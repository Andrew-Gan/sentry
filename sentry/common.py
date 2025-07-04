from enum import Enum
import os
from .model_signing.hashing import memory
from .model_signing.cuda import nvrtc

# cuda source file, digest size, hashlib module
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
    if device == 'cpu':
        if topology == Topology.MERKLE:
            hasher = memory.MerkleCPU(hashAlgo.value[2])
    elif device == 'gpu':
        if topology == Topology.SERIAL:
            hasher = memory.SeqGPU(hashAlgo)
        elif topology == Topology.MERKLE:
            hasher = memory.MerkleGPU(hashAlgo)
        elif topology == Topology.HADD:
            hasher = memory.HomomorphicGPU(hashAlgo, inputType)

    if not hasher:
        raise NotImplementedError(f'Unsupported for {hashAlgo.name}, \
            {topology.name}, {inputType.name}, {device}')

    return hasher
