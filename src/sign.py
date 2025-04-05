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

"""Script to sign models."""

import argparse
import logging
import pathlib

from .model_signing.hashing import hashing, file, state
from .model_signing.serialization import serialize_by_file, serialize_by_state
from .model_signing import model
from .model_signing.signature import fake
from .model_signing.signature import key
from .model_signing.signature import pki
from .model_signing.signing import in_toto
from .model_signing.signing import in_toto_signature
from .model_signing.signing import signing
from .model_signing.signing import sigstore
from .compile import HashType, Topology, InputType, compile_hasher

log = logging.getLogger(__name__)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Script to sign models")
    parser.add_argument(
        "--model_path",
        help="path to the model to sign",
        required=True,
        type=pathlib.Path,
        dest="model_path",
    )
    parser.add_argument(
        "--sig_out",
        help="the output file, it defaults ./model.sig",
        required=False,
        type=pathlib.Path,
        default=pathlib.Path("./model.sig"),
        dest="sig_out",
    )

    method_cmd = parser.add_subparsers(
        required=True,
        dest="method",
        help="method to sign the model: [pki, private-key, sigstore, skip]",
    )
    # PKI
    pki = method_cmd.add_parser("pki")
    pki.add_argument(
        "--cert_chain",
        help="paths to pem encoded certificate files or a single file"
        + "containing a chain",
        required=False,
        type=list[str],
        default=[],
        nargs="+",
        dest="cert_chain_path",
    )
    pki.add_argument(
        "--signing_cert",
        help="the pem encoded signing cert",
        required=True,
        type=pathlib.Path,
        dest="signing_cert_path",
    )
    pki.add_argument(
        "--private_key",
        help="the path to the private key PEM file",
        required=True,
        type=pathlib.Path,
        dest="key_path",
    )
    # private key
    p_key = method_cmd.add_parser("private-key")
    p_key.add_argument(
        "--private_key",
        help="the path to the private key PEM file",
        required=True,
        type=pathlib.Path,
        dest="key_path",
    )
    # sigstore
    sigstore = method_cmd.add_parser("sigstore")
    sigstore.add_argument(
        "--use_ambient_credentials",
        help="use ambient credentials (also known as Workload Identity,"
        + "default is true)",
        required=False,
        type=bool,
        default=True,
        dest="use_ambient_credentials",
    )
    sigstore.add_argument(
        "--staging",
        help="Use Sigstore's staging instances, instead of the default"
        " production instances",
        action="store_true",
        dest="sigstore_staging",
    )
    sigstore.add_argument(
        "--identity-token",
        help="the OIDC identity token to use",
        required=False,
        dest="identity_token",
    )
    # skip
    method_cmd.add_parser("skip")

    return parser.parse_args()


def _get_payload_signer(args: argparse.Namespace, device='cpu', num_sigs=1) -> signing.Signer:
    if args.method == "private-key":
        _check_private_key_options(args)
        signerHasher = None
        if device == 'gpu':
            signerHasher = compile_hasher(HashType.SHA256, Topology.MERKLE, InputType.MODULE)
        payload_signer = key.ECKeySigner.from_path(
            key_path=args.key_path,
            device=device,
            num_sigs=num_sigs,
            hasher=signerHasher)
        if device == 'cpu':
            return in_toto_signature.IntotoSigner(payload_signer)
        elif device == 'gpu':
            return in_toto_signature.IntotoSigner(payload_signer)

    elif args.method == "pki":
        _check_pki_options(args)
        payload_signer = pki.PKISigner.from_path(
            args.key_path, args.signing_cert_path, args.cert_chain_path
        )
        return in_toto_signature.IntotoSigner(payload_signer)
    elif args.method == "sigstore":
        return sigstore.SigstoreDSSESigner(
            use_ambient_credentials=args.use_ambient_credentials,
            use_staging=args.sigstore_staging,
            identity_token=args.identity_token,
        )
    elif args.method == "skip":
        return in_toto_signature.IntotoSigner(fake.FakeSigner())
    else:
        log.error(f"unsupported signing method {args.method}")
        log.error(
            'supported methods: ["pki", "private-key", "sigstore", "skip"]'
        )
        exit(-1)


def _check_private_key_options(args: argparse.Namespace):
    if args.key_path == "":
        log.error("--private_key must be set to a valid private key PEM file")
        exit()


def _check_pki_options(args: argparse.Namespace):
    _check_private_key_options(args)
    if args.signing_cert_path == "":
        log.error(
            (
                "--signing_cert must be set to a valid ",
                "PEM encoded signing certificate",
            )
        )
        exit()
    if args.cert_chain_path == "":
        log.warning("No certificate chain provided")


def build(hashType: HashType, topology: Topology, inputType: InputType, num_sigs=1):
    if inputType is not InputType.FILE:
        hasher = compile_hasher(hashType, topology, inputType)

    args = _arguments()
    dev = 'cpu' if inputType == InputType.FILE else 'gpu'
    payload_signer = _get_payload_signer(args, dev, num_sigs)
    
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

    return hasher, payload_signer, serializer


# item to pass in differs depending on inputType
# InputType.FILE   : directory path containing model files
# InputType.MODULE : loaded PyTorch model of type torch.nn.Module
# InputType.DIGEST : hash digest of type hashing.Digest. This skips hashing.
def sign_item(item, payload_signer, serializer, inputType: InputType):
    args = _arguments()

    if inputType == InputType.FILE:
        item = pathlib.Path(item)
    elif inputType == InputType.MODULE:
        item = item.to('cuda').state_dict()

    if inputType == InputType.DIGEST:
        sigs = model.sign(
            item=item,
            signer=payload_signer,
            payload_generator=in_toto.DigestsIntotoPayload.from_manifest,
            serializer=serializer,
            skipHash=True,
        )
    else:
        sigs = model.sign(
            item=item,
            signer=payload_signer,
            payload_generator=in_toto.DigestsIntotoPayload.from_manifest,
            serializer=serializer,
            ignore_paths=[args.sig_out],
        )
    for sig in sigs:
        sig.write(args.sig_out)


# if __name__ == "__main__":
#     PATH = './model.pth'
#     models = [
#         ('pytorch/vision:v0.10.0', 'resnet152'),
#         ('huggingface/pytorch-transformers', 'model', 'bert-base-uncased'),
#         ('huggingface/transformers', 'modelForCausalLM', 'gpt2'),
#         ('pytorch/vision:v0.10.0', 'vgg19'),
#         ('huggingface/transformers', 'modelForCausalLM', 'gpt2-large'),
#         ('huggingface/transformers', 'modelForCausalLM', 'gpt2-xl'),
#     ]

#     for m in models:
#         if len(m) == 2:
#             net = torch.hub.load(m[0], m[1], pretrained=True)
#         elif len(m) == 3:
#             net = torch.hub.load(m[0], m[1], m[2])

#         print(f'Hashing {net.__class__.__name__}, num layers: {len(net.state_dict())}, num param: {sum(p.numel() for p in net.parameters())}')
#         t0 = time.monotonic()
#         torch.save(net, PATH)
#         t1 = time.monotonic()
#         print(f'Write to file: {1000*(t1-t0):.2f} ms')

#         t0 = time.monotonic()
#         torch.load(PATH, weights_only=False)
#         t1 = time.monotonic()
#         print(f'Read from file: {1000*(t1-t0):.2f} ms')

#         for hashType in HashType:
#             print(f'CPU Hashing from file using {hashType.name}')
#             sign_item(PATH, hashType, Topology.SERIAL, InputType.FILE)

#             print(f'SeqGPU-{hashType.name}')
#             sign_item(net, hashType, Topology.SERIAL, InputType.MODULE)

#             print(f'MerkleGPU-{hashType.name}')
#             sign_item(net, hashType, Topology.MERKLE, InputType.MODULE)

#         print(f'HomomorphicGPU-lattice')
#         sign_item(net, HashType.LATTICE, Topology.HOMOMORPHIC, InputType.MODULE)
        
#         del net
