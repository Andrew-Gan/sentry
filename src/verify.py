# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script can be used to verify model signatures."""

import argparse
import logging
import pathlib

from .model_signing.hashing import hashing, file, state
from .model_signing.serialization import serialize_by_file, serialize_by_state
from .model_signing import model
from .model_signing.signature import fake
from .model_signing.signature import key
from .model_signing.signature import pki
from .model_signing.signature import verifying
from .model_signing.signing import in_toto, in_toto_signature
from .model_signing.signing import signing
from .model_signing.signing import sigstore
from .compile import HashType, Topology, InputType, compile_hasher


log = logging.getLogger(__name__)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Script to verify models")
    parser.add_argument(
        "--sig_path",
        help="the path to the signature",
        required=True,
        type=pathlib.Path,
        dest="sig_path",
    )
    parser.add_argument(
        "--model_path",
        help="the path to the model's base folder",
        type=pathlib.Path,
        dest="model_path",
    )

    method_cmd = parser.add_subparsers(
        required=True,
        dest="method",
        help="method to verify the model: [pki, private-key, sigstore, skip]",
    )
    # pki subcommand
    pki = method_cmd.add_parser("pki")
    pki.add_argument(
        "--root_certs",
        help="paths to PEM encoded certificate files or a single file"
        + "used as the root of trust",
        required=False,
        type=list[str],
        default=[],
        dest="root_certs",
    )
    # private key subcommand
    p_key = method_cmd.add_parser("private-key")
    p_key.add_argument(
        "--public_key",
        help="the path to the public key used for verification",
        required=True,
        type=pathlib.Path,
        dest="key",
    )
    # sigstore subcommand
    sigstore = method_cmd.add_parser("sigstore")
    sigstore.add_argument(
        "--identity",
        help="the expected identity of the signer e.g. name@example.com",
        required=True,
        type=str,
        dest="identity",
    )
    sigstore.add_argument(
        "--identity-provider",
        help="the identity provider expected e.g. https://accounts.example.com",
        required=True,
        type=str,
        dest="identity_provider",
    )
    # skip subcommand
    method_cmd.add_parser("skip")

    return parser.parse_args()


def _get_verifier(args: argparse.Namespace, device='cpu', num_sigs=1) -> signing.Verifier:
    if args.method == "private-key":
        _check_private_key_flags(args)
        verifierHasher = None
        if device == 'gpu':
            verifierHasher = compile_hasher(HashType.SHA256, Topology.SERIAL, InputType.MODULE)

        verifier = key.ECKeyVerifier.from_path(args.key, device, num_sigs, verifierHasher)
        if device == 'cpu':
            return in_toto_signature.IntotoVerifier(verifier)
        elif device == 'gpu':
            return in_toto_signature.IntotoVerifier(verifier)

    elif args.method == "pki":
        _check_pki_flags(args)
        verifier = pki.PKIVerifier.from_paths(args.root_certs)
        return in_toto_signature.IntotoVerifier(verifier)
    elif args.method == "sigstore":
        return sigstore.SigstoreDSSEVerifier(
            identity=args.identity, oidc_issuer=args.identity_provider
        )
    elif args.method == "skip":
        return in_toto_signature.IntotoVerifier(fake.FakeVerifier())
    else:
        log.error(f"unsupported verification method {args.method}")
        log.error(
            'supported methods: ["pki", "private-key", "sigstore", "skip"]'
        )
        exit(-1)


def _check_private_key_flags(args: argparse.Namespace):
    if args.key == "":
        log.error("--public_key must be defined")
        exit()


def _check_pki_flags(args: argparse.Namespace):
    if not args.root_certs:
        log.warning("no root of trust is set using system default")


def _get_signature(args: argparse.Namespace) -> signing.Signature:
    if args.method == "sigstore":
        return sigstore.SigstoreSignature.read(args.sig_path)
    else:
        return in_toto_signature.IntotoSignature.read(args.sig_path)


def build(hashType: HashType, topology: Topology, inputType: InputType, num_sigs=1):
    if inputType is not InputType.FILE:
        hasher = compile_hasher(hashType, topology, inputType)

    args = _arguments()
    dev = 'cpu' if inputType == InputType.FILE else 'gpu'
    verifier = _get_verifier(args, dev, num_sigs)
    
    if inputType == InputType.DIGEST:
        serializer = serialize_by_state.ManifestSerializer(
            state_hasher_factory=None)
    
    elif inputType == InputType.FILE:
        def hasher_factory(item) -> hashing.HashEngine:
            return file.SimpleFileHasher(
                file=item, content_hasher=hashType.value[2]())
        serializer = serialize_by_file.ManifestSerializer(
            file_hasher_factory=hasher_factory)

    elif inputType == InputType.MODULE:
        def hasher_factory(item) -> hashing.HashEngine:
            return state.SimpleStateHasher(state=item, content_hasher=hasher)
        serializer = serialize_by_state.ManifestSerializer(
            state_hasher_factory=hasher_factory)

    return hasher, verifier, serializer


def verify_item(item, verifier, serializer, inputType: InputType):
    args = _arguments()
    sig = _get_signature(args)

    if inputType == InputType.FILE:
        item = pathlib.Path(item)
    elif inputType == InputType.MODULE:
        item = item.to('cuda').state_dict()

    try:
        if inputType == InputType.DIGEST:
            model.verify(
                sig=sig,
                item=item,
                verifier=verifier,
                serializer=serializer,
                ignore_paths=[args.sig_path],
                skipHash=True,
            )
        else:
            model.verify(
                sig=sig,
                item=item,
                verifier=verifier,
                serializer=serializer,
                ignore_paths=[args.sig_path],
            )
    except verifying.VerificationError as err:
        log.error(f"verification failed: {err}")

    log.info("all checks passed")
