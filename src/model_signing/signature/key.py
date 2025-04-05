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
from ..cuda import utils
from cuda.bindings import driver
from ..cuda.utils import checkCudaErrors
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

    def __init__(self, private_key: ec.EllipticCurvePrivateKey, device='cpu',
                 num_sigs=1, hasher=None):

        self._private_key = private_key
        self._device = device
        self._num_sigs = num_sigs

        if device == 'gpu':
            self._gsv = ctypes.CDLL('./RapidEC/gsv.so')
            self._gsv.sign_init.argtypes = [ctypes.c_int]
            self._gsv.sign_init.restype = None
            self._gsv.sign_exec.argtypes = [ctypes.c_int, ctypes.POINTER(gsv_sign_t)]
            self._gsv.sign_exec.restype = ctypes.POINTER(gsv_sign_t)
            self._gsv.sign_close.argtypes = []
            self._gsv.sign_close.restype = None

            self._gsv.sign_init(self._num_sigs)
            self._hasher = hasher

            path = os.path.join(os.sep, 'home', 'src', 'model_signing', 'cuda',
                                'binToHex.cuh')
            self._ctx, [self._binToHex] = utils.compile(path, ['binToHex'])

    def __exit__(self):
        if self._device == 'gpu':
            self._gsv.sign_close(self._num_sigs)

    @classmethod
    def from_path(cls, key_path: str, password: str = None, device='cpu',
                  num_sigs=1, hasher=None):
        private_key = load_ec_private_key(key_path, password)
        return cls(private_key, device, num_sigs, hasher)

    def _gpu_convert_bin_to_hex(self, hashes_d):
        n = len(hashes_d)
        hash_bin_d = cp.ndarray([n, 64])
        hash_hex_d = cp.ndarray([n, 128])
        for i, hash_d in enumerate(hashes_d):
            hash_bin_d[i] = hash_d
        hex_in = np.array([hash_bin_d.data.ptr], dtype=np.uint64)
        hex_out = np.array([hash_hex_d.data.ptr], dtype=np.uint64)
        args = [hex_in, hex_out]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        stream = checkCudaErrors(driver.cuStreamCreate(0))

        print(type(self._ctx), type(self._binToHex), flush=True)
        checkCudaErrors(driver.cuCtxSetCurrent(self._ctx))
        checkCudaErrors(driver.cuLaunchKernel(
            self._binToHex, 1, 1, 1, 64, n, 1, 0, stream, args.ctypes.data, 0
        ))
        checkCudaErrors(driver.cuStreamSynchronize(stream))
        checkCudaErrors(driver.cuStreamDestroy(stream))
        return hash_hex_d

    def sign(self, stmnts: list[statement.Statement], hashes_d=None) -> list[bundle_pb.Bundle]:
        sign_tasks = []
        bundles = []
        paes = []

        for stmnt in stmnts:
            pae = encoding.pae(stmnt.pb)
            if self._device == 'cpu':
                paes.append(pae)

            if self._device == 'gpu':
                hash_hex_d = self._gpu_convert_bin_to_hex(hashes_d)

                dummy_pos = [m.start() for m in re.finditer(b'0'*128, pae)]
                assert(len(dummy_pos) == len(hashes_d))
                pae_d = cp.frombuffer(pae, dtype=cp.uint8)
                for i, pos in enumerate(dummy_pos):
                    driver.cuMemcpy(pae_d+pos, hash_hex_d[i].data.ptr, 128)

                # debug
                tmp = bytearray(len(pae))
                driver.cuMemcpy(tmp, pae_d, len(pae))
                print(tmp)

                paes.append(pae_d)
        
        return

        if self._device == 'cpu':
            sigs = [self._private_key.sign(pae, ec.ECDSA(SHA256())) for pae in paes]
        elif self._device == 'gpu':
            for pae in enumerate(paes):
                self._hasher.update({'pae': pae}, len(pae))
                digest = self._hasher.compute().digest_value
                priv_key = self._private_key.private_numbers().private_value.to_bytes(32)
                sign_tasks.append(gsv_sign_t(
                    e=gsv_mem_t.from_buffer_copy(digest),
                    priv_key=gsv_mem_t.from_buffer_copy(priv_key)))

            sig_pending = (gsv_sign_t * self._num_sigs)(*sign_tasks)
            sigs_uncoded = self._gsv.sign_exec(self._num_sigs, sig_pending)
            sigs_uncoded = ctypes.cast(sigs_uncoded, ctypes.POINTER((gsv_sign_t * self._num_sigs)))
            rsPair = [(int.from_bytes(sig.r), int.from_bytes(sig.s)) for sig in sigs_uncoded.contents]
            sigs = [utils.encode_dss_signature(rs[0], rs[1]) for rs in rsPair]

        for stmnt, sig in zip(stmnts, sigs):
            env = intoto_pb.Envelope(
                payload=json_format.MessageToJson(stmnt.pb).encode(),
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

    def __init__(self, public_key: ec.EllipticCurvePublicKey, device='cpu',
                 num_sigs=1, hasher=None):

        self._public_key = public_key
        self._device = device
        self._num_sigs = num_sigs

        if device == 'gpu':
            self._gsv = ctypes.CDLL('./RapidEC/gsv.so')
            self._gsv.verify_init.argtypes = [ctypes.c_int]
            self._gsv.verify_init.restype = None
            self._gsv.verify_exec.argtypes = [ctypes.c_int, ctypes.POINTER(gsv_sign_t)]
            self._gsv.verify_exec.restype = ctypes.POINTER(ctypes.c_int)
            self._gsv.verify_close.argtypes = []
            self._gsv.verify_close.restype = None

            self._gsv.verify_init(self._num_sigs)
            self._hasher = hasher

    def __exit__(self):
        if self._device == 'gpu':
            self._gsv.verify_close(self._num_sigs)

    @classmethod
    def from_path(cls, key_path: str, device='cpu', num_sigs=1, hasher=None):
        with open(key_path, "rb") as fd:
            serialized_key = fd.read()
        public_key = load_pem_public_key(serialized_key)
        return cls(public_key, device, num_sigs, hasher)

    def verify(self, bundles: list[bundle_pb.Bundle]) -> None:
        verify_tasks = []
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
            for pae, sig in zip(paes, sigs):
                pae_d = torch.frombuffer(bytearray(pae), dtype=torch.uint8).cuda()
                self._hasher.update({'pae': pae_d}, len(pae))
                digest = self._hasher.compute().digest_value
                pub_x = self._public_key.public_numbers().x.to_bytes(32)
                pub_y = self._public_key.public_numbers().y.to_bytes(32)
                r, s = utils.decode_dss_signature(sig)
                verify_tasks.append(gsv_verify_t(
                    r=gsv_mem_t.from_buffer_copy(r.to_bytes(32)),
                    s=gsv_mem_t.from_buffer_copy(s.to_bytes(32)),
                    e=gsv_mem_t.from_buffer_copy(digest),
                    key_x=gsv_mem_t.from_buffer_copy(pub_x),
                    key_y=gsv_mem_t.from_buffer_copy(pub_y),
                ))
            ver_pending = (gsv_verify_t * self._num_sigs)(*verify_tasks)
            results = self._gsv.verify_exec(self._num_sigs, ver_pending)
            results = list(results)
            print(results)
