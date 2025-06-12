from enum import Enum
import os
from .model_signing.hashing import memory
from .model_signing.cuda import mycuda

class HashAlgo(Enum):
    SHA256  = ('sha256.cuh', 32, memory.SHA256)
    BLAKE2B = ('blake2b.cuh', 64, memory.BLAKE2)
    SHA3    = ('sha3.cuh', 64, memory.SHA3)
    LATTICE = ('lattice.cuh', 64)

class Topology(Enum):
    SERIAL = 1
    MERKLE = 2
    HADD = 3

class InputType(Enum):
    """
    FILE mode uses Hashlib/Sigstore, MODULE or DIGEST mode uses Sentry
    """
    FILE = 1
    MODULE = 2
    DIGEST = 3

def get_hasher(hashAlgo: HashAlgo, topology: Topology, inputType: InputType, device: str):
    digestSize = hashAlgo.value[1]
    srcPath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'model_signing', 'cuda', hashAlgo.value[0])

    if device == 'cpu':
        if topology == Topology.MERKLE:
            hasher = memory.MerkleCPU(hashAlgo.value[2])

    elif device == 'gpu':
        if topology == Topology.SERIAL:
            ctx, [seq] = mycuda.compile(srcPath, [f'seq'])
            hasher = memory.SeqGPU(seq, ctx, digestSize)

        elif topology == Topology.MERKLE:
            ctx, [hashB, reduce] = mycuda.compile(srcPath, ['hash', 'reduce'])
            hasher = memory.MerkleGPU(hashB, reduce, ctx, digestSize)

        elif topology == Topology.HADD:
            if hashAlgo != HashAlgo.LATTICE:
                raise RuntimeError('Homomorphic Hashing must use Lattice Hashing')
            if inputType == InputType.MODULE:
                ctx, [hashB, reduce] = mycuda.compile(srcPath,
                    ['hash_ltHash', 'reduce_ltHash'])
                hasher = memory.HomomorphicGPU(hashB, reduce, ctx, digestSize)
            elif inputType == InputType.DIGEST:
                ctx, [hashB, reduce] = mycuda.compile(srcPath,
                    ['hash_dataset_ltHash', 'reduce_ltHash'])
                hasher = memory.HomomorphicGPU(hashB, reduce, ctx, digestSize)

    if not hasher:
        raise NotImplementedError(f'Unsupported for {hashAlgo.name}, \
            {topology.name}, {inputType.name}, {device}')

    return hasher
