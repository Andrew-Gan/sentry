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
import torch
import time

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
        i = 0
        for v in data.values():
            checkCudaErrors(runtime.cudaMemcpyAsync(iData+i, v.data_ptr(),
                v.nbytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))
            i += v.nbytes

        iDataA = np.array([iData], dtype=np.uint64)
        oData = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
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
        free, total = checkCudaErrors(runtime.cudaMemGetInfo())
        print(f'Peak memory consumption: {(total-free) // 1000000} / {total // 1000000}')
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


class MerkleGPU(hashing.StreamingHashEngine):
    def __init__(self, hashblock, reduce, ctx, digestsize, reducefactor):
        self.ctx = ctx
        self.hashblock = hashblock
        self.reduce = reduce
        self.digestSize = digestsize
        self.reduceFactor = reducefactor
        self.runtime = 0

    @override
    def update(self, data: collections.OrderedDict, blockSize: int) -> None:
        # prevfree, _ = checkCudaErrors(runtime.cudaMemGetInfo())

        start = time.monotonic()

        self.digest = bytes(self.digestSize)
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        stream = checkCudaErrors(runtime.cudaStreamCreate())
        starts = []
        workAddr = []
        workSize = []
        nBlock = 0
        for v in data.values():
            starts.append(nBlock)
            blk = (v.nbytes + (blockSize-1)) // blockSize
            workAddr.append(v.data_ptr())
            workSize.append(v.nbytes)
            nBlock += blk

        starts = torch.tensor(starts, device='cuda')
        workAddr = torch.tensor(workAddr, device='cuda')
        workSize = torch.tensor(workSize, device='cuda')

        nThread = nBlock
        block = min(512, nThread)
        grid = (nThread + (block-1)) // block

        iData = checkCudaErrors(runtime.cudaMalloc((grid+1) * block * self.digestSize))
        oData = checkCudaErrors(runtime.cudaMalloc((grid+1) * block * self.digestSize))

        iData = (iData, np.array([iData], dtype=np.uint64))
        oData = (oData, np.array([oData], dtype=np.uint64))
        blockSize = np.array([blockSize], dtype=np.uint64)
        startT = np.array([starts.data_ptr()], dtype=np.uint64)
        workAddr = np.array([workAddr.data_ptr()], dtype=np.uint64)
        workSize = np.array([workSize.data_ptr()], dtype=np.uint64)
        startsLen = np.array([len(starts)], dtype=np.uint64)
        nThread = np.array([nThread], dtype=np.uint64)

        args = [oData[1], blockSize, startT, workSize, workAddr, startsLen, nThread]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        checkCudaErrors(driver.cuLaunchKernel(
            self.hashblock, grid, 1, 1, block, 1, 1, 0, stream, args.ctypes.data, 0,
        ))

        while nBlock > 1:
            iData, oData = oData, iData
            nBlock = ((nBlock + 1) & ~0b1)
            nThread = (nBlock // 2) * self.reduceFactor
            block = min(512, nThread)
            grid = (nThread + (block-1)) // block

            nThread = np.array([nThread], dtype=np.uint64)
            args = [oData[1], iData[1], nThread]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.reduce, grid, 1, 1, block, 1, 1, block * self.digestSize,
                stream, args.ctypes.data, 0,
            ))
            nBlock = grid

        checkCudaErrors(runtime.cudaMemcpyAsync(self.digest, oData[0], self.digestSize,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))

        # free, total = checkCudaErrors(runtime.cudaMemGetInfo())
        # print(f'Peak memory consumption: {(prevfree-free) // 1000000} / {total // 1000000}')

        checkCudaErrors(runtime.cudaFreeAsync(iData[0], stream))
        checkCudaErrors(runtime.cudaFreeAsync(oData[0], stream))
        checkCudaErrors(runtime.cudaStreamSynchronize(stream))
        checkCudaErrors(runtime.cudaStreamDestroy(stream))

        self.runtime += time.monotonic()-start

    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self) -> hashing.Digest:
        return hashing.Digest(self.digest_name, self.digest)

    @property
    @override
    def digest_name(self) -> str:
        return "MerkleGPU"

    @property
    @override
    def digest_size(self) -> int:
        return self.digestSize

class AddGPU(hashing.StreamingHashEngine):
    def __init__(self, hashblock, adder, ctx, digestsize, reducefactor, batchsizepow2):
        self.ctx = ctx
        self.hashblock = hashblock
        self.adder = adder
        self.digestSize = digestsize
        self.reduceFactor = reducefactor
        if batchsizepow2 > 7 or batchsizepow2 < 0:
            raise RuntimeError('batchsizepow2 must be between 0 and 7')
        self.batchSize = 1 << batchsizepow2
        self.runtime = 0

    @override
    def update(self, data: collections.OrderedDict, blockSize) -> None:

        start = time.monotonic()
        t = [0] * 4

        self.digest = bytes(self.digestSize)
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        blockSize = (blockSize, np.array([blockSize], dtype=np.uint64))

        nBlocks = sum([(v.nbytes + blockSize[0] - 1) // blockSize[0] for v in data.values()])
        numStreams = (nBlocks + self.batchSize - 1) // self.batchSize
        print('numStreams ', numStreams)
        streams = [checkCudaErrors(runtime.cudaStreamCreate()) for _ in range(numStreams)]

        digestSum = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
        digestSum = (digestSum, np.array([digestSum], dtype=np.uint64))

        startT = checkCudaErrors(runtime.cudaMalloc(8 * self.batchSize))
        for i in range(self.batchSize):
            checkCudaErrors(runtime.cudaMemset(startT+i*8, i, 1))
        startT = (startT, np.array([startT], dtype=np.uint64))
        startsLen = np.array([self.batchSize], dtype=np.uint64)

        dictIter = iter(data.values())
        currItem = next(dictIter)
        currAddr = currItem.data_ptr()
        for i, s in enumerate(streams):
            # checkCudaErrors(runtime.cudaDeviceSynchronize())
            t0 = time.monotonic()

            workAddr = []
            workSize = []

            for nThread in range(self.batchSize):
                workAddr.append(currAddr)
                itemEnd = currItem.data_ptr() + currItem.nbytes
                nextStart = currAddr + blockSize[0]
                if itemEnd <= nextStart:
                    workSize.append(itemEnd - currAddr)
                    try:
                        currItem = next(dictIter)
                        currAddr = currItem.data_ptr()
                    except StopIteration:
                        break
                else:
                    workSize.append(blockSize[0])
                    currAddr += blockSize[0]
            nThread += 1
            
            # checkCudaErrors(runtime.cudaDeviceSynchronize())
            t1 = time.monotonic()
            t[0] += t1-t0
            
            workAddr = np.array(workAddr, dtype=np.uint64)
            workSize = np.array(workSize, dtype=np.uint64)
            
            workAddrA = checkCudaErrors(runtime.cudaMallocAsync(workAddr.nbytes, s))
            workSizeA = checkCudaErrors(runtime.cudaMallocAsync(workSize.nbytes, s))
            
            checkCudaErrors(runtime.cudaMemcpyAsync(workAddrA, workAddr.ctypes.data,
                workAddr.nbytes, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, s))
            
            checkCudaErrors(runtime.cudaMemcpyAsync(workSizeA, workSize.ctypes.data,
                workSize.nbytes, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, s))
           
            workAddr = np.array([workAddrA], dtype=np.uint64)
            workSize = np.array([workSizeA], dtype=np.uint64)

            buffer = checkCudaErrors(runtime.cudaMallocAsync(nThread*self.digestSize, s))
            buffer = (buffer, np.array([buffer], dtype=np.uint64))

            # checkCudaErrors(runtime.cudaDeviceSynchronize())
            t2 = time.monotonic()
            t[1] += t2-t1

            nThreadA = np.array([nThread], dtype=np.uint64)
            args = [buffer[1], blockSize[1], startT[1], workSize, workAddr, startsLen, nThreadA]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.hashblock, 1, 1, 1, nThread, 1, 1, 0, s, args.ctypes.data, 0,
            ))

            # checkCudaErrors(runtime.cudaDeviceSynchronize())
            t3 = time.monotonic()
            t[2] += t3-t2

            nThread = ((nThread + 1) & ~0b1)
            nThread = (nThread // 2) * self.reduceFactor

            addOut = np.array([True], dtype=bool)
            args = [digestSum[1], buffer[1], addOut]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.adder, 1, 1, 1, nThread, 1, 1, nThread * self.digestSize,
                s, args.ctypes.data, 0,
            ))

            checkCudaErrors(runtime.cudaFreeAsync(workAddrA, s))
            checkCudaErrors(runtime.cudaFreeAsync(workSizeA, s))

            # checkCudaErrors(runtime.cudaDeviceSynchronize())
            t4 = time.monotonic()
            t[3] += t4-t3

        for s in streams:
            checkCudaErrors(runtime.cudaStreamSynchronize(s))
            checkCudaErrors(runtime.cudaStreamDestroy(s))

        checkCudaErrors(runtime.cudaMemcpy(self.digest, digestSum[0],
            self.digestSize, runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        checkCudaErrors(runtime.cudaFree(digestSum[0]))
        checkCudaErrors(runtime.cudaFree(startT[0]))

        # print([f'{i*1000:.2f}' for i in t])

        self.runtime += time.monotonic() - start

    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self) -> hashing.Digest:
        return hashing.Digest(self.digest_name, self.digest)

    @property
    @override
    def digest_name(self) -> str:
        return "MerkleGPU"

    @property
    @override
    def digest_size(self) -> int:
        return self.digestSize
