# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functionality to sign and verify models with keys."""

from cryptography.hazmat.primitives.serialization import \
    load_pem_public_key, load_pem_private_key, Encoding, PublicFormat, \
    PrivateFormat, NoEncryption
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives.hashes import SHA256
from google.protobuf import json_format
from in_toto_attestation.v1 import statement
from in_toto_attestation.v1 import statement_pb2 as statement_pb
from sigstore_protobuf_specs.dev.sigstore.bundle import v1 as bundle_pb
from sigstore_protobuf_specs.dev.sigstore.common import v1 as common_pb
from sigstore_protobuf_specs.io import intoto as intoto_pb

from . import encoding
from .signing import Signer
from .verifying import VerificationError
from .verifying import Verifier
import ctypes
import torch
import re
import numpy as np
import cupy as cp
from cuda.bindings import driver
from ..cuda.compiler import compileCuda, checkCudaErrors
import os


class gsv_params_t(ctypes.Structure):
    _fields_ = [("TPB",           ctypes.c_uint32),
                ("MAX_ROTATION",  ctypes.c_uint32),
                ("SHM_LIMIT",     ctypes.c_uint32),
                ("CONSTANT_TIME", ctypes.c_bool),
                ("TPI",           ctypes.c_uint32),]


class gsv_mem_t(ctypes.Structure):
    _fields_ = [("_limbs", ctypes.c_uint32 * 8),]


class gsv_sign_t(ctypes.Structure):
    _fields_ = [("e",           gsv_mem_t),
                ("priv_key",    gsv_mem_t),
                ("k",           gsv_mem_t),
                ("r",           gsv_mem_t),
                ("s",           gsv_mem_t),]


class gsv_verify_t(ctypes.Structure):
    _fields_ = [("r",           gsv_mem_t),
                ("s",           gsv_mem_t),
                ("e",           gsv_mem_t),
                ("key_x",       gsv_mem_t),
                ("key_y",       gsv_mem_t),]


def load_ec_private_key(
    path: str, password: str | None = None
) -> ec.EllipticCurvePrivateKey:
    private_key: ec.EllipticCurvePrivateKey
    with open(path, "rb") as fd:
        serialized_key = fd.read()
    private_key = load_pem_private_key(
        serialized_key, password=password
    )
    return private_key


class ECKeySigner(Signer):
    """Provides a Signer using an elliptic curve private key for signing."""

    def __init__(self, private_key: ec.EllipticCurvePrivateKey, device='gpu',
                 num_sigs=1, hasher=None):

        self._private_key = private_key
        self._priv_val = self._private_key.private_numbers().private_value.to_bytes(32, byteorder='big')
        self._device = device
        self._num_sigs = num_sigs

        if device == 'gpu':
            self._gsv = ctypes.CDLL('./RapidEC/gsv.so')
            self._gsv.sign_init.argtypes = [ctypes.c_int]
            self._gsv.sign_init.restype = None
            self._gsv.sign_exec.argtypes = [ctypes.c_int,
                ctypes.POINTER(gsv_sign_t), ctypes.POINTER(ctypes.c_uint64)]
            self._gsv.sign_exec.restype = ctypes.POINTER(gsv_sign_t)
            self._gsv.sign_close.argtypes = []
            self._gsv.sign_close.restype = None

            self._gsv.sign_init(self._num_sigs)
            self._hasher = hasher

            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'cuda', 'encoder.cuh')
            
            if not hasattr(ECKeySigner, '_binToHex'):
                _, [ECKeySigner._binToHex] = compileCuda(path, ['binToHex'])

    def __exit__(self):
        if self._device == 'gpu':
            self._gsv.sign_close(self._num_sigs)

    @classmethod
    def from_path(cls, key_path: str, password: str = None, device='gpu',
                  num_sigs=1, hasher=None):
        private_key = load_ec_private_key(key_path, password)
        return cls(private_key, device, num_sigs, hasher)

    def _gpu_bin_to_hex(self, hashes_d):
        n = len(hashes_d)
        hash_bin_d = cp.ndarray([n, self._hasher.digestSize], dtype=cp.uint8)
        hash_hex_d = cp.ndarray([n, 2*self._hasher.digestSize], dtype=cp.uint8)

        for i, hash_d in enumerate(hashes_d):
            driver.cuMemcpy(hash_bin_d[i].data.ptr, hash_d, self._hasher.digestSize)

        hex_in = np.array([hash_bin_d.data.ptr], dtype=np.uint64)
        hex_out = np.array([hash_hex_d.data.ptr], dtype=np.uint64)
        args = [hex_in, hex_out]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        stream = checkCudaErrors(driver.cuStreamCreate(0))
        checkCudaErrors(driver.cuLaunchKernel(
            ECKeySigner._binToHex, n, 1, 1, self._hasher.digestSize, 1, 1, 0,
            stream, args.ctypes.data, 0
        ))
        checkCudaErrors(driver.cuStreamSynchronize(stream))
        checkCudaErrors(driver.cuStreamDestroy(stream))
        return hash_hex_d

    def _find_dummy_pos(self, data: bytearray):
        dummy_value = bytes(range(self._hasher.digestSize)).hex().encode()
        return [m.start() for m in re.finditer(dummy_value, data)]

    def sign(self, stmnts: list[statement.Statement], hashes=None) -> list[bundle_pb.Bundle]:
        bundles = []
        sigs = []

        if self._device == 'cpu':
            for stmnt in stmnts:
                pae = encoding.pae(stmnt.pb)
                sig = self._private_key.sign(pae, ec.ECDSA(SHA256()))
                sigs.append(sig)

        elif self._device == 'gpu':
            identities_hashes_hex_d = [self._gpu_bin_to_hex(h) for h in hashes]
            paes = {}
            for i, (stmnt, identity_hashes_hex_d) in enumerate(zip(stmnts, identities_hashes_hex_d)):
                pae = encoding.pae(stmnt.pb)
                dummy_pos = self._find_dummy_pos(pae)
                assert(len(dummy_pos) == len(identity_hashes_hex_d))
                pae_d = cp.frombuffer(pae, dtype=cp.uint8)
                for j, pos in enumerate(dummy_pos):
                    checkCudaErrors(driver.cuMemcpy(pae_d.data.ptr + pos,
                        identity_hashes_hex_d[j].data.ptr, 2*self._hasher.digestSize))
                paes[i] = pae_d
            
            self._hasher.update(paes)
            digestMap = self._hasher.compute()
            digests = [v[1] for k, v in digestMap.items()]
            digests = (ctypes.c_uint64 * self._num_sigs)(*digests)

            sign_tasks = [gsv_sign_t(
                priv_key=gsv_mem_t.from_buffer_copy(self._priv_val)
            ) for _ in range(self._num_sigs)]
            sig_pending = (gsv_sign_t * self._num_sigs)(*sign_tasks)
            sign_res = self._gsv.sign_exec(self._num_sigs, sig_pending, digests)
            sign_res = ctypes.cast(sign_res, ctypes.POINTER((gsv_sign_t * self._num_sigs)))
            rsPair = [(
                int.from_bytes(sig.r, byteorder='big'),
                int.from_bytes(sig.s, byteorder='big')
            ) for sig in sign_res.contents]
            sigs = [utils.encode_dss_signature(r, s) for r, s in rsPair]

        for stmnt, sig, identity_hashes_hex_d in zip(stmnts, sigs, identities_hashes_hex_d):
            if self._device == 'gpu':
                for i, hash_hex_d in enumerate(identity_hashes_hex_d):
                    hash_hex_h = bytes(hash_hex_d.nbytes)
                    checkCudaErrors(driver.cuMemcpyDtoH(hash_hex_h, hash_hex_d.data.ptr, hash_hex_d.nbytes))
                    stmnt.pb.subject[i].digest.update({'sha256': hash_hex_h.decode()})
            payload = json_format.MessageToJson(stmnt.pb).encode()
            env = intoto_pb.Envelope(
                payload=payload,
                payload_type=encoding.PAYLOAD_TYPE,
                signatures=[intoto_pb.Signature(sig=sig, keyid=None)],
            )
            bundles.append(bundle_pb.Bundle(
                media_type="application/vnd.dev.sigstore.bundle.v0.3+json",
                verification_material=bundle_pb.VerificationMaterial(
                    public_key=common_pb.PublicKey(
                        raw_bytes=self._private_key.public_key().public_bytes(
                            encoding=Encoding.PEM,
                            format=PublicFormat.SubjectPublicKeyInfo,
                        ),
                        key_details=common_pb.PublicKeyDetails.PKIX_ECDSA_P256_SHA_256,
                    )
                ),
                dsse_envelope=env,
            ))

        return bundles


class ECKeyVerifier(Verifier):
    """Provides a verifier using a public key."""

    def __init__(self, public_key: ec.EllipticCurvePublicKey, device='gpu',
                 num_sigs=1, hasher=None):

        self._public_key = public_key
        self._pub_x = public_key.public_numbers().x.to_bytes(32, byteorder='big')
        self._pub_y = public_key.public_numbers().y.to_bytes(32, byteorder='big')
        self._device = device
        self._num_sigs = num_sigs

        if device == 'gpu':
            self._gsv = ctypes.CDLL('./RapidEC/gsv.so')
            self._gsv.verify_init.argtypes = [ctypes.c_int]
            self._gsv.verify_init.restype = None
            self._gsv.verify_exec.argtypes = [ctypes.c_int, ctypes.POINTER(gsv_verify_t)]
            self._gsv.verify_exec.restype = ctypes.POINTER(ctypes.c_int)
            self._gsv.verify_close.argtypes = []
            self._gsv.verify_close.restype = None

            self._gsv.verify_init(self._num_sigs)
            self._hasher = hasher

    def __exit__(self):
        if self._device == 'gpu':
            self._gsv.verify_close(self._num_sigs)

    @classmethod
    def from_path(cls, key_path: str, device='gpu', num_sigs=1, hasher=None):
        with open(key_path, "rb") as fd:
            serialized_key = fd.read()
        public_key = load_pem_public_key(serialized_key)
        return cls(public_key, device, num_sigs, hasher)

    def verify(self, bundles: list[bundle_pb.Bundle]) -> None:
        paes = []
        sigs = []

        for bundle in bundles:
            statement = json_format.Parse(
                bundle.dsse_envelope.payload,
                statement_pb.Statement(),  # pylint: disable=no-member
            )
            paes.append(encoding.pae(statement))
            sigs.append(bundle.dsse_envelope.signatures[0].sig)

        if self._device == 'cpu':
            try:
                [self._public_key.verify(sig, pae, ec.ECDSA(SHA256()))
                 for pae, sig in zip(paes, sigs)]
            except Exception as e:
                raise VerificationError(
                    "signature verification failed " + str(e)
                ) from e
        elif self._device == 'gpu':
            paes_d = {}
            for i, (pae, sig) in enumerate(zip(paes, sigs)):
                paes_d[i] = cp.frombuffer(pae, dtype=cp.uint8)

            self._hasher.update(paes_d)
            digestMap = self._hasher.compute()
            digests = [v[1] for k, v in digestMap.items()]
            digests = (ctypes.c_uint64 * self._num_sigs)(*digests)

            r, s = utils.decode_dss_signature(sig)
            verify_tasks = [gsv_verify_t(
                r=gsv_mem_t.from_buffer_copy(r.to_bytes(32, byteorder='big')),
                s=gsv_mem_t.from_buffer_copy(s.to_bytes(32, byteorder='big')),
                key_x=gsv_mem_t.from_buffer_copy(self._pub_x),
                key_y=gsv_mem_t.from_buffer_copy(self._pub_y),
            )]
            ver_pending = (gsv_verify_t * self._num_sigs)(*verify_tasks)
            ver_res = self._gsv.verify_exec(self._num_sigs, ver_pending, digests)
            ver_res = ver_res[:self._num_sigs]
            # TODO: verify ec key results
