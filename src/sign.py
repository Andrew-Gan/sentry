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

import sys
import argparse
import logging
import pathlib
import collections

from model_signing import model
from model_signing.hashing import file
from model_signing.hashing import state
from model_signing.hashing import memory
from model_signing.serialization import serialize_by_file
from model_signing.serialization import serialize_by_state
from model_signing.signature import fake
from model_signing.signature import key
from model_signing.signature import pki
from model_signing.signing import in_toto
from model_signing.signing import in_toto_signature
from model_signing.signing import signing
from model_signing.signing import sigstore
import torch
import time
from cuda.bindings import driver, nvrtc, runtime
import numpy as np

import torchvision.transforms.v2

log = logging.getLogger(__name__)
SAMPLE_SIZE = 8


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


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


def _get_payload_signer(args: argparse.Namespace) -> signing.Signer:
    if args.method == "private-key":
        _check_private_key_options(args)
        payload_signer = key.ECKeySigner.from_path(
            private_key_path=args.key_path
        )
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


def sign_files(path, hasher, chunk=8192):
    path = pathlib.Path(path)
    logging.basicConfig(level=logging.INFO)
    args = _arguments()

    payload_signer = _get_payload_signer(args)

    def hasher_factory(file_path: pathlib.Path) -> file.FileHasher:
        return file.SimpleFileHasher(
            file=file_path, content_hasher=hasher, chunk_size=chunk
        )

    serializer = serialize_by_file.ManifestSerializer(
        file_hasher_factory=hasher_factory
    )

    t0 = time.monotonic()
    for _ in range(SAMPLE_SIZE):
        sig = model.sign_file(
            model_path=path,
            signer=payload_signer,
            payload_generator=in_toto.DigestsIntotoPayload.from_manifest,
            serializer=serializer,
            ignore_paths=[args.sig_out],
        )
    t1 = time.monotonic()
    print(f'Runtime: {1000*(t1-t0)/SAMPLE_SIZE:.2f} ms')
    sig.write(args.sig_out)


def sign_model(net, hasher):
    net = net.to('cuda')
    logging.basicConfig(level=logging.INFO)
    args = _arguments()

    payload_signer = _get_payload_signer(args)

    def hasher_factory(state_dict: collections.OrderedDict) -> state.StateHasher:
        return state.SimpleStateHasher(
            state=state_dict, content_hasher=hasher
        )

    serializer = serialize_by_state.ManifestSerializer(
        state_hasher_factory=hasher_factory
    )

    states = {'state_dict': net.state_dict(),}

    t0 = time.monotonic()
    for _ in range(SAMPLE_SIZE):
        sig = model.sign_state(
            states=states,
            signer=payload_signer,
            payload_generator=in_toto.DigestsIntotoPayload.from_manifest,
            serializer=serializer,
        )
    t1 = time.monotonic()
    print(f'Runtime: {1000*(t1-t0)/SAMPLE_SIZE:.2f} ms')
    sig.write(args.sig_out)

def compile_sha256(arch_arg, ctx, opts):
    with open('model_signing/cuda/sha256.cu', 'r') as f:
        code = f.read()
    # parse cuda code from file
    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(code), b'sha256.cu', 0, [], []))
    # compile code into program and extract ptx
    checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
    ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b' ' * ptxSize
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))
    ptx = np.char.array(ptx)
    # obtain global functions as entrypoints into gpu
    module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))
    streamHash = checkCudaErrors(driver.cuModuleGetFunction(module, b'stream_hash_sha256'))
    pre = checkCudaErrors(driver.cuModuleGetFunction(module, b'merkle_pre_sha256'))
    treeHash = checkCudaErrors(driver.cuModuleGetFunction(module, b'merkle_hash_sha256'))
    return streamHash, pre, treeHash

def compile_blake2(arch_arg, ctx, opts):
    with open('model_signing/cuda/blake2.cu', 'r') as f:
        code = f.read()
    # parse cuda code from file
    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(code), b'blake2.cu', 0, [], []))
    # compile code into program and extract ptx
    checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
    ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b' ' * ptxSize
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))
    ptx = np.char.array(ptx)
    # obtain global functions as entrypoints into gpu
    module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))
    streamHash = checkCudaErrors(driver.cuModuleGetFunction(module, b'stream_hash_blake2'))
    pre = checkCudaErrors(driver.cuModuleGetFunction(module, b'merkle_pre_blake2'))
    treeHash = checkCudaErrors(driver.cuModuleGetFunction(module, b'merkle_hash_blake2'))
    return streamHash, pre, treeHash

if __name__ == "__main__":
    PATH = './model.pth'
    models = []
    models.append(torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True))
    # models.append(torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased'))
    # models.append(torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2'))
    # models.append(torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True))
    # models.append(torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2-large'))
    # models.append(torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2-xl'))

    driver.cuInit(0)
    cuDevice = checkCudaErrors(runtime.cudaGetDevice())
    ctx = checkCudaErrors(driver.cuCtxCreate(0, cuDevice))
    # get device arch
    major = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice))
    minor = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice))
    arch_arg = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')
    opts = [b'--fmad=false', arch_arg]

    streamHashSha256, preHashSha256, treeHashSha256 = compile_sha256(arch_arg, ctx, opts)
    streamHashBlake2, preHashBlake2, treeHashBlake2 = compile_blake2(arch_arg, ctx, opts)

    for net in models:
        print(f'Hashing {net.__class__.__name__}, num param: {sum(p.numel() for p in net.parameters())}')
        t0 = time.monotonic()
        torch.save(net, PATH)
        t1 = time.monotonic()
        print(f'Write to file: {1000*(t1-t0):.2f} ms')

        t0 = time.monotonic()
        torch.load(PATH, weights_only=False)
        t1 = time.monotonic()
        print(f'Read from file: {1000*(t1-t0):.2f} ms')

        # print('Hashing from file using SHA256')
        # sign_files(PATH, memory.SHA256())

        # print('Hashing from device using StreamGPU-SHA256')
        # sign_model(net, memory.StreamGPU(streamHashSha256, ctx, 32))

        print('Hashing from device using MerkleGPU-SHA256')
        sign_model(net, memory.MerkleGPU(preHashSha256, treeHashSha256, ctx, 32))

        # print('Hashing from file using BLAKE2')
        # sign_files(PATH, memory.BLAKE2())

        # print('Hashing from device using StreamGPU-Blake2')
        # sign_model(net, memory.StreamGPU(streamHashBlake2, ctx, 64))

        print('Hashing from device using MerkleGPU-Blake2')
        sign_model(net, memory.MerkleGPU(preHashBlake2, treeHashBlake2, ctx, 64))
    
    # checkCudaErrors(driver.cuModuleUnload(streamHash))
    # checkCudaErrors(driver.cuModuleUnload(pre))
    # checkCudaErrors(driver.cuModuleUnload(treeHash))
    # checkCudaErrors(driver.cuCtxDestroy(ctx))
