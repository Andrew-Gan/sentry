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

from .model_signing import model
from .model_signing.hashing import hashing
from .model_signing.hashing import file
from .model_signing.hashing import state
from .model_signing.hashing import memory
from .model_signing.serialization import serialize_by_file
from .model_signing.serialization import serialize_by_state
from .model_signing.signature import fake
from .model_signing.signature import key
from .model_signing.signature import pki
from .model_signing.signing import in_toto
from .model_signing.signing import in_toto_signature
from .model_signing.signing import signing
from .model_signing.signing import sigstore

import torch
import time
from cuda.bindings import driver, nvrtc, runtime
import numpy as np
from enum import Enum
import os

log = logging.getLogger(__name__)

class HashType(Enum):
    SHA256  = ('sha256.cuh', 32, memory.SHA256)
    BLAKE2B = ('blake2b.cuh', 64, memory.BLAKE2)
    SHA3    = ('sha3.cuh', 64, memory.SHA3)
    LATTICE = ('ltHash.cuh', 64)

class Topology(Enum):
    SERIAL = 1
    MERKLE = 2
    HOMOMORPHIC = 3

class InputType(Enum):
    """
    FILE mode uses Hashlib/Sigstore, MODEL or DATASET mode uses Sentry
    """
    FILE = 1
    MODULE = 2
    DIGEST = 3


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
        raise RuntimeError("CUDA error code={}({})".format(result[0].value,
            _cudaGetErrorEnum(result[0])))
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


def compile(srcPath, function_names):
    cuDevice = checkCudaErrors(runtime.cudaGetDevice())
    ctx = checkCudaErrors(driver.cuCtxGetCurrent())
    if repr(ctx) == '<CUcontext 0x0>':
        ctx = checkCudaErrors(driver.cuCtxCreate(0, cuDevice))
    # get device arch
    major = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        cuDevice))
    minor = checkCudaErrors(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        cuDevice))
    arch_arg = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')

    inclPath = os.path.dirname(srcPath)
    opts = [b'--fmad=false', arch_arg, b'-I' + inclPath.encode()]

    with open(srcPath, 'r') as f:
        code = f.read()
    # parse cuda code from file
    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(code), 
        bytes(srcPath[-1], 'utf-8'), 0, [], []))
    
    # compile code into program and extract ptx
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
        log = bytes(logSize)
        checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
        print(log.decode("utf-8"))
        checkCudaErrors(err)

    ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b' ' * ptxSize
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))
    ptx = np.char.array(ptx)
    # obtain global functions as entrypoints into gpu
    module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))
    funcs = []
    for func in function_names:
        funcs.append(checkCudaErrors(driver.cuModuleGetFunction(module,
            bytes(f'{func}', 'utf-8'))))
    return ctx, funcs


def compile_cuda_hasher(hashType: HashType, topology: Topology, inputType: InputType):
    digestSize = hashType.value[1]
    srcPath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'model_signing', 'cuda', hashType.value[0])

    if topology == Topology.SERIAL:
        ctx, [seq] = compile(srcPath, [f'seq'])
        hasher = memory.SeqGPU(seq, ctx, digestSize)

    elif topology == Topology.MERKLE:
        ctx, [hashB, reduce] = compile(srcPath, ['hash', 'reduce'])
        hasher = memory.MerkleGPU(hashB, reduce, ctx, digestSize)

    elif topology == Topology.HOMOMORPHIC:
        if hashType != HashType.LATTICE:
            raise RuntimeError('Homomorphic Hashing must use Lattice Hashing')
        if inputType == InputType.MODULE:
            ctx, [hashB, reduce] = compile(srcPath, ['hash_ltHash', 'reduce_ltHash'])
            hasher = memory.HomomorphicGPU(hashB, reduce, ctx, digestSize)
        elif inputType == InputType.DIGEST:
            ctx, [hashB, reduce] = compile(srcPath, ['hash_dataset_ltHash', 'reduce_ltHash'])
            hasher = memory.HomomorphicGPU(hashB, reduce, ctx, digestSize)

    if not hasher:
        raise RuntimeError('Failed to build hasher due to invalid arguments')

    return hasher


def compile_cuda_signer():
    srcDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'RapidEC', 'gsv.cu')
    ctx, [signer] = compile(srcDir, ['GSV_sign_exec'])

    return signer


# item to pass in differs depending on inputType
# InputType.FILE   : directory path containing model files
# InputType.MODULE : loaded PyTorch model of type torch.nn.Module
# InputType.DIGEST : hash digest of type hashing.Digest. This skips hashing.
def sign(item, hashType: HashType, topology: Topology, inputType: InputType,
        hasher=None, signer=None):

    if inputType == InputType.FILE and hashType == HashType.LATTICE:
        raise RuntimeError('Lattice Hashing not supported for CPU file hashing')
    # early compilation of cuda modules
    if not hasher and inputType == InputType.MODULE:
        hasher = compile_cuda_hasher(hashType, topology, inputType)

    logging.basicConfig(level=logging.INFO)
    args = _arguments()

    payload_signer = _get_payload_signer(args)
    # if inputType is InputType.FILE: # use argument specified signer
    #     payload_signer = _get_payload_signer(args)
    # else: # use GPU ECDSA
    #     signer = key.ECKeyGPUSigner.from_path(args.key_path, signer=signer)
    #     payload_signer = in_toto_signature.IntotoSigner(signer)
    
    if inputType == InputType.DIGEST:
        serializer = serialize_by_state.ManifestSerializer(
            state_hasher_factory=None)
        sig = model.sign_hash(item, payload_signer,
            in_toto.DigestsIntotoPayload.from_manifest, serializer)

    else:
        if inputType == InputType.FILE:
            def hasher_factory(item) -> hashing.HashEngine:
                return file.SimpleFileHasher(
                    file=item, content_hasher=hashType.value[2]())
            serializer = serialize_by_file.ManifestSerializer(
                file_hasher_factory=hasher_factory)
            myitem = pathlib.Path(item)

        elif inputType == InputType.MODULE:
            def hasher_factory(item) -> hashing.HashEngine:
                return state.SimpleStateHasher(state=item, content_hasher=hasher)
            serializer = serialize_by_state.ManifestSerializer(
                state_hasher_factory=hasher_factory)
            myitem = item.to('cuda').state_dict()

        sig, hashRuntime = model.sign(
            item=myitem,
            signer=payload_signer,
            payload_generator=in_toto.DigestsIntotoPayload.from_manifest,
            serializer=serializer,
            ignore_paths=[args.sig_out],
        )

        print(f'[Sigstore] hash-only runtime: {1000*hashRuntime:.2f} ms')

    sig.write(args.sig_out)


if __name__ == "__main__":
    PATH = './model.pth'
    models = [
        ('pytorch/vision:v0.10.0', 'resnet152'),
        ('huggingface/pytorch-transformers', 'model', 'bert-base-uncased'),
        ('huggingface/transformers', 'modelForCausalLM', 'gpt2'),
        ('pytorch/vision:v0.10.0', 'vgg19'),
        ('huggingface/transformers', 'modelForCausalLM', 'gpt2-large'),
        ('huggingface/transformers', 'modelForCausalLM', 'gpt2-xl'),
    ]

    for m in models:
        if len(m) == 2:
            net = torch.hub.load(m[0], m[1], pretrained=True)
        elif len(m) == 3:
            net = torch.hub.load(m[0], m[1], m[2])

        print(f'Hashing {net.__class__.__name__}, num layers: {len(net.state_dict())}, num param: {sum(p.numel() for p in net.parameters())}')
        t0 = time.monotonic()
        torch.save(net, PATH)
        t1 = time.monotonic()
        print(f'Write to file: {1000*(t1-t0):.2f} ms')

        # t0 = time.monotonic()
        # torch.load(PATH, weights_only=False)
        # t1 = time.monotonic()
        # print(f'Read from file: {1000*(t1-t0):.2f} ms')

        for hashType in HashType:
            # print(f'CPU Hashing from file using {hashType.name}')
            # sign(PATH, hashType, Topology.SERIAL, InputType.FILE)

            # print(f'SeqGPU-{hashType.name}')
            # sign(net, hashType, Topology.SERIAL, InputType.MODULE)

            print(f'MerkleGPU-{hashType.name}')
            sign(net, hashType, Topology.MERKLE, InputType.MODULE)

        print(f'HomomorphicGPU-lattice')
        sign(net, HashType.LATTICE, Topology.HOMOMORPHIC, InputType.MODULE)
        
        del net
