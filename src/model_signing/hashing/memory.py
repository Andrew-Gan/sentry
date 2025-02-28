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

class SeqGPU(hashing.StreamingHashEngine):
    def __init__(self, hash, ctx, digest_size):
        self.ctx = ctx
        self.hash = hash
        self.digestSize = digest_size
    
    @override
    def update(self, data: collections.OrderedDict, blockSize: int) -> None:
        self.digest = bytes(self.digestSize)
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        total_size = sum(d.nbytes for d in data.values())
        stream = checkCudaErrors(runtime.cudaStreamCreate())
        iData = checkCudaErrors(runtime.cudaMalloc(total_size))
        oData = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
        i = 0
        for v in data.values():
            checkCudaErrors(runtime.cudaMemcpyAsync(iData+i, v.data_ptr(),
                v.nbytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))
            i += v.nbytes

        iDataA = np.array([iData], dtype=np.uint64)
        oDataA = np.array([oData], dtype=np.uint64)
        blockSizeA = np.array([blockSize], dtype=np.uint64)
        nA = np.array([total_size // blockSize], dtype=np.uint64)

        args = [oDataA, iDataA, blockSizeA, nA]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        checkCudaErrors(driver.cuLaunchKernel(
            self.hash, 1, 1, 1, 1, 1, 1, 0, stream, args.ctypes.data, 0,
        ))

        checkCudaErrors(runtime.cudaMemcpyAsync(self.digest, oData, self.digestSize,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        checkCudaErrors(runtime.cudaFreeAsync(iData, stream))
        checkCudaErrors(runtime.cudaFreeAsync(oData, stream))
        checkCudaErrors(runtime.cudaStreamSynchronize(stream))
        checkCudaErrors(runtime.cudaStreamDestroy(stream))
    
    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self):
        return hashing.Digest(self.digest_name, self.digest)

    @property
    @override
    def digest_name(self) -> str:
        return "SeqGPU"

    @property
    @override
    def digest_size(self) -> int:
        return self.digestSize

class HashAndReduceGPU(hashing.StreamingHashEngine):
    # reduce factor is 1 for normal hashing
    # for lattice hash we spawn 8 times more threads to handle reduction
    # because each 64 byte digest can be represented as 8 uint64_t for summation
    def __init__(self, hashblock, reduce, ctx, digestsize, reducefactor):
        self.ctx = ctx
        self.hashblock = hashblock
        self.reduce = reduce
        self.digestSize = digestsize
        self.reduceFactor = reducefactor

    @override
    def update(self, data: collections.OrderedDict, blockSize) -> None:
        self.digest = bytes(self.digestSize)
        digests = checkCudaErrors(runtime.cudaMalloc(len(data)*self.digestSize))
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        iData = []
        oData = []
        streams = [checkCudaErrors(runtime.cudaStreamCreate()) for i in range(len(data))]

        for v in data.values():
            size = v.numel() * v.element_size()
            nThread = (size + (blockSize-1)) // blockSize
            iData.append(checkCudaErrors(runtime.cudaMalloc(nThread*self.digestSize)))
            oData.append(checkCudaErrors(runtime.cudaMalloc(nThread*self.digestSize)))
        
        for i, v in enumerate(data.values()):
            size = v.numel() * v.element_size()
            nThread = (size + (blockSize-1)) // blockSize

            sizeA = np.array([size], dtype=np.uint64)
            tensorA = np.array([v.data_ptr()], dtype=np.uint64)
            myIn = (iData[i], np.array([iData[i]], dtype=np.uint64))
            myOut = (oData[i], np.array([oData[i]], dtype=np.uint64))
            blockSizeA = np.array([blockSize], dtype=np.uint64)

            args = [myOut[1], tensorA, blockSizeA, sizeA]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            block = min(512, nThread)
            grid = (nThread + (block-1)) // block

            checkCudaErrors(driver.cuLaunchKernel(
                self.hashblock, grid, 1, 1, block, 1, 1, 0, streams[i],
                args.ctypes.data, 0,
            ))
            nThread *= self.reduceFactor
            nThread //= 2

            while nThread > 0:
                myIn, myOut = myOut, myIn
                nThreadA = np.array([nThread], dtype=np.uint64)
                args = [myOut[1], myIn[1], nThreadA]
                args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
                block = min(512, nThread)
                grid = (nThread + (block-1)) // block

                checkCudaErrors(driver.cuLaunchKernel(
                    self.reduce, grid, 1, 1, block, 1, 1, block * self.digestSize,
                    streams[i], args.ctypes.data, 0,
                ))
                nThread //= 2 * block

            checkCudaErrors(runtime.cudaMemcpyAsync(digests + i * self.digestSize,
                myOut[0], self.digestSize,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, streams[i]))
        
        for i, o, s in zip(iData, oData, streams):
            checkCudaErrors(runtime.cudaFreeAsync(i, s))
            checkCudaErrors(runtime.cudaFreeAsync(o, s))
            checkCudaErrors(runtime.cudaStreamSynchronize(s))
            checkCudaErrors(runtime.cudaStreamDestroy(s))

        nThread = (len(data) * self.reduceFactor + 1) // 2
        stream = checkCudaErrors(runtime.cudaStreamCreate())
        output = checkCudaErrors(runtime.cudaMalloc(nThread*self.digestSize))
        iData = (digests, np.array([digests], dtype=np.uint64))
        oData = (output, np.array([output], dtype=np.uint64))

        while nThread > 0:
            nThreadA = np.array([nThread], dtype=np.uint64)
            args = [oData[1], iData[1], nThreadA]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            block = min(512, nThread)
            grid = (nThread + (block-1)) // block

            checkCudaErrors(driver.cuLaunchKernel(
                self.reduce, grid, 1, 1, block, 1, 1, block * self.digestSize,
                stream, args.ctypes.data, 0,
            ))
            nThread //= 2 * block

        checkCudaErrors(runtime.cudaMemcpyAsync(self.digest, digests, self.digestSize,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        checkCudaErrors(runtime.cudaFreeAsync(iData[0], stream))
        checkCudaErrors(runtime.cudaFreeAsync(oData[0], stream))
        checkCudaErrors(runtime.cudaStreamSynchronize(stream))
        checkCudaErrors(runtime.cudaStreamDestroy(stream))

    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self) -> hashing.Digest:
        return hashing.Digest(self.digest_name, self.digest)

    @property
    @override
    def digest_name(self) -> str:
        return "HashAndReduceGPU"

    @property
    @override
    def digest_size(self) -> int:
        return self.digestSize
