# Copyright 2024 The Sigstore Authors
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

"""Digests for memory objects.

These can only compute hashes of objects residing in memory, after they get
converted to bytes.

Example usage:
```python
>>> hasher = SHA256()
>>> hasher.update(b"abcd")
>>> digest = hasher.compute()
>>> digest.digest_hex
'88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589'
```

Or, passing the data directly in the constructor:
```python
>>> hasher = SHA256(b"abcd")
>>> digest = hasher.compute()
>>> digest.digest_hex
'88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589'
```
"""

import hashlib

from typing_extensions import override

from model_signing.hashing import hashing

from cuda.bindings import driver, nvrtc, runtime
import numpy as np
import collections
import math


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


class SHA256(hashing.StreamingHashEngine):
    """A wrapper around `hashlib.sha256`."""

    def __init__(self, initial_data: bytes = b""):
        self._hasher = hashlib.sha256(initial_data)

    @override
    def update(self, data: bytes) -> None:
        self._hasher.update(data)

    @override
    def reset(self, data: bytes = b"") -> None:
        self._hasher = hashlib.sha256(data)

    @override
    def compute(self) -> hashing.Digest:
        return hashing.Digest(self.digest_name, self._hasher.digest())

    @property
    @override
    def digest_name(self) -> str:
        return "sha256"

    @property
    @override
    def digest_size(self) -> int:
        return self._hasher.digest_size


class BLAKE2(hashing.StreamingHashEngine):
    """A wrapper around `hashlib.blake2b`."""

    def __init__(self, initial_data: bytes = b""):
        """Initializes an instance of a BLAKE2 hash engine.

        Args:
            initial_data: Optional initial data to hash.
        """
        self._hasher = hashlib.blake2b(initial_data)

    @override
    def update(self, data: bytes) -> None:
        self._hasher.update(data)

    @override
    def reset(self, data: bytes = b"") -> None:
        self._hasher = hashlib.blake2b(data)

    @override
    def compute(self) -> hashing.Digest:
        return hashing.Digest(self.digest_name, self._hasher.digest())

    @property
    @override
    def digest_name(self) -> str:
        return "blake2b"

    @property
    @override
    def digest_size(self) -> int:
        return self._hasher.digest_size

class MerkleGPU(hashing.StreamingHashEngine):
    def __init__(self, pre, hash, ctx):
        self.ctx = ctx
        self.pre = pre
        self.hash = hash
    
    @override
    def update(self, data: collections.OrderedDict, blockSize: int) -> None:
        self.digest = bytes(32)
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        total_size = sum(d.nbytes for d in data.values())
        nThread = (total_size + (blockSize-1)) // blockSize
        nThread = 2**math.ceil(math.log2(nThread))

        content = checkCudaErrors(runtime.cudaMalloc(nThread*blockSize))
        stream = checkCudaErrors(runtime.cudaStreamCreate())
        i = 0
        for v in data.values():
            checkCudaErrors(runtime.cudaMemcpy(content+i, v.data_ptr(),
                v.nbytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
            i += v.nbytes

        contentA = np.array([content], dtype=np.uint64)
        buffer = checkCudaErrors(runtime.cudaMalloc(nThread*32))
        nThreadA = np.array([nThread], dtype=np.uint64)
        bufferA = np.array([buffer], dtype=np.uint64)
        blockSizeA = np.array([blockSize], dtype=np.uint64)

        args = [bufferA, contentA, blockSizeA, nThreadA]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        block = min(1024, nThread)
        grid = (nThread + (block-1)) // block

        checkCudaErrors(driver.cuLaunchKernel(
            self.pre, grid, 1, 1, block, 1, 1,
            0, stream, args.ctypes.data, 0,
        ))
        nThread //= 2

        while nThread > 0:
            nThreadA = np.array([nThread], dtype=np.uint64)
            args = [contentA, bufferA, nThreadA]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            block = min(1024, nThread)
            grid = (nThread + (block-1)) // block

            checkCudaErrors(driver.cuLaunchKernel(
                self.hash, grid, 1, 1, block, 1, 1,
                0, stream, args.ctypes.data, 0,
            ))
            checkCudaErrors(runtime.cudaMemcpy2D(buffer, 32, content,
                2*1024*32, 32, grid, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
            nThread //= 2048

        checkCudaErrors(runtime.cudaMemcpy(self.digest, buffer, 32,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        checkCudaErrors(runtime.cudaFree(content))
        checkCudaErrors(runtime.cudaFree(buffer))

    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self) -> hashing.Digest:
        return hashing.Digest('Merkle-SHA256', self.digest)

    @property
    @override
    def digest_name(self) -> str:
        return "Merkle-SHA256"

    @property
    @override
    def digest_size(self) -> int:
        return 32