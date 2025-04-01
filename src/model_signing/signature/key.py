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

    def __init__(self, private_key: ec.EllipticCurvePrivateKey):
        self._private_key = private_key

    @classmethod
    def from_path(cls, private_key_path: str, password: str | None = None):
        private_key = load_ec_private_key(private_key_path, password)
        return cls(private_key)

    def sign(self, stmnt: statement.Statement) -> bundle_pb.Bundle:
        pae = encoding.pae(stmnt.pb)
        sig = self._private_key.sign(pae, ec.ECDSA(SHA256()))
        env = intoto_pb.Envelope(
            payload=json_format.MessageToJson(stmnt.pb).encode(),
            payload_type=encoding.PAYLOAD_TYPE,
            signatures=[intoto_pb.Signature(sig=sig, keyid=None)],
        )
        bdl = bundle_pb.Bundle(
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
        )

        return bdl


class ECKeyVerifier(Verifier):
    """Provides a verifier using a public key."""

    def __init__(self, public_key: ec.EllipticCurvePublicKey):
        self._public_key = public_key

    @classmethod
    def from_path(cls, key_path: str):
        with open(key_path, "rb") as fd:
            serialized_key = fd.read()
        public_key = load_pem_public_key(serialized_key)
        return cls(public_key)

    def verify(self, bundle: bundle_pb.Bundle) -> None:
        statement = json_format.Parse(
            bundle.dsse_envelope.payload,
            statement_pb.Statement(),  # pylint: disable=no-member
        )
        pae = encoding.pae(statement)
        try:
            self._public_key.verify(
                bundle.dsse_envelope.signatures[0].sig, pae, ec.ECDSA(SHA256())
            )
        except Exception as e:
            raise VerificationError(
                "signature verification failed " + str(e)
            ) from e


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


class ECKeyGPUSigner(Signer):
    """Provides a Signer using an elliptic curve private key for signing."""

    def __init__(self, private_key: ec.EllipticCurvePrivateKey, num_sigs, hasher):
        self._private_key = private_key
        self._num_sigs = num_sigs
        self.gsv = ctypes.CDLL('./RapidEC/gsv.so')
        self.gsv.sign_init.argtypes = [ctypes.c_int]
        self.gsv.sign_init.restype = None
        self.gsv.sign_exec.argtypes = [ctypes.c_int, ctypes.POINTER(gsv_sign_t)]
        self.gsv.sign_exec.restype = ctypes.POINTER(gsv_sign_t)
        self.gsv.sign_close.argtypes = [ctypes.c_int]
        self.gsv.sign_close.restype = None

        self.gsv.sign_init(self._num_sigs)
        self._hasher = hasher

    def __del__(self):
        self.gsv.sign_close(self._num_sigs)

    @classmethod
    def from_path(cls, private_key_path: str, password: str=None, num_sigs=1, hasher=None):
        private_key = load_ec_private_key(private_key_path, password)
        return cls(private_key, num_sigs, hasher)

    def sign(self, stmnts: list[statement.Statement]) -> bundle_pb.Bundle:
        sign_tasks = []

        for stmnt in stmnts:
            pae = encoding.pae(stmnt.pb)
            pae_d = torch.frombuffer(bytearray(pae), dtype=torch.uint8).cuda()
            self._hasher.update({'pae': pae_d}, len(pae))
            digest = self._hasher.compute().digest_value
            priv_key = self._private_key.private_numbers().private_value.to_bytes(32)
            sign_tasks.append(gsv_sign_t(
                e=gsv_mem_t.from_buffer_copy(digest),
                priv_key=gsv_mem_t.from_buffer_copy(priv_key)))

        gsv_sign_arr_t = gsv_sign_t * self._num_sigs
        sig_pending = gsv_sign_arr_t(*sign_tasks)
        sig_completed = self.gsv.sign_exec(self._num_sigs, sig_pending)

        bundles = []
        for i in range(self._num_sigs):
            sig = sig_completed[i]
            r, s = int.from_bytes(sig.r), int.from_bytes(sig.s)
            sig_encoded = utils.encode_dss_signature(r, s)

            env = intoto_pb.Envelope(
                payload=json_format.MessageToJson(stmnt.pb).encode(),
                payload_type=encoding.PAYLOAD_TYPE,
                signatures=[intoto_pb.Signature(sig=sig_encoded, keyid=None)],
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

# for use with verify
# pub_x = self._private_key.private_numbers().public_numbers.x.to_bytes(32)
# pub_y = self._private_key.private_numbers().public_numbers.y.to_bytes(32)
