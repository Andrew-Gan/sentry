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

from . import hashing

from cuda.bindings import driver, runtime
import numpy as np
import cupy as cp
import collections
from ..cuda.nvrtc import rtcompile, checkCudaErrors
import threading


srcPath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'model_signing', 'cuda', 'topology')


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

class SHA3(hashing.StreamingHashEngine):
    """A wrapper around `hashlib.sha256`."""

    def __init__(self, initial_data: bytes = b""):
        self._hasher = hashlib.sha3_512(initial_data)

    @override
    def update(self, data: bytes) -> None:
        self._hasher.update(data)

    @override
    def reset(self, data: bytes = b"") -> None:
        self._hasher = hashlib.sha3_512(data)

    @override
    def compute(self) -> hashing.Digest:
        return hashing.Digest(self.digest_name, self._hasher.digest())

    @property
    @override
    def digest_name(self) -> str:
        return "sha3"

    @property
    @override
    def digest_size(self) -> int:
        return self._hasher.digest_size

class SeqGPU(hashing.StreamingHashEngine):
    def __init__(self, hashAlgo):
        self.digestSize = hashAlgo.value[1]
        srcPath = os.path.join(srcPath, hashAlgo.value[0])
        self.ctx, [self.hash] = nvrtc.rtcompile(srcPath, [f'hash'], hashAlgo.name)
    
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

class MerkleCPU(hashing.StreamingHashEngine):
    def __init__(self, hasher : hashing.StreamingHashEngine):
        self.hasher = hasher
        self.digestSize = hasher().digest_size
        self.digests = {}
    
    def worker_thread(self, layer, v, myBuff0, myBuff1, nBlock, blockSize):
        # hash blocks of data into equal number of digests
        hasher = self.hasher()
        rem = v.nbytes
        for i in range(nBlock):
            hasher.reset()
            sizeToRead = min(rem, blockSize)
            data = bytes(sizeToRead)
            checkCudaErrors(runtime.cudaMemcpy(data, v.data_ptr()+(i*blockSize),
                sizeToRead, runtime.cudaMemcpyKind.cudaMemcpyHostToHost))
            if sizeToRead > 0:
                hasher.update(data)
                myBuff0[i] = hasher.compute().digest_value
            rem -= sizeToRead

        # merkle tree reduction to single digest
        pos = False
        while nBlock > 1:
            nBlock = (nBlock + 1) & ~0b1 # padding to multiple of 2
            for i in range(nBlock // 2):
                hasher.reset()
                pos = not pos
                if pos:
                    hasher.update(myBuff0[2*i])
                    hasher.update(myBuff0[2*i+1])
                    myBuff1[i] = hasher.compute().digest_value
                else:
                    hasher.update(myBuff1[2*i])
                    hasher.update(myBuff1[2*i+1])
                    myBuff0[i] = hasher.compute().digest_value
            nBlock //= 2
        self.digests[layer] = myBuff1[0] if pos else myBuff0[0]
        return pos

    @override
    def update(self, data: collections.OrderedDict, blockSize=8192) -> None:
        # min number of blocks for each layer
        nBlocks = [(v.nbytes + blockSize - 1) // blockSize for v in data.values()]
        # pad number of blocks for each layer to multiple of 2
        nBlocks = [(n + 1) & ~0b1 for n in nBlocks]
        buff0 = [[bytearray(self.digestSize) for _ in range(nBlock)] for nBlock in nBlocks]
        buff1 = [[bytearray(self.digestSize) for _ in range(nBlock)] for nBlock in nBlocks]

        self.digests = {layer: bytearray(self.digestSize) for layer in data.keys()}
        self.digests['model'] = bytearray(self.digestSize)

        threads = []
        for (layer, v), myBuff0, myBuff1, nBlock in zip(data.items(), buff0, buff1, nBlocks):
            thread = threading.Thread(target=self.worker_thread,
                args=[layer, v, myBuff0, myBuff1, nBlock, blockSize])
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

        # reduce layer digests into one model digest
        nLayer = len(data)
        hasher = self.hasher()
        pos = False
        buff0 = [bytearray(self.digestSize) for _ in range(nLayer)]
        buff1 = [bytearray(self.digestSize) for _ in range(nLayer)]
        for i, layer in enumerate(data.keys()):
            buff0[i] = self.digests[layer]
        while nLayer > 1:
            nLayer = (nLayer + 1) & ~0b1 # padding to multiple of 2
            for i in range(nLayer // 2):
                hasher.reset()
                pos = not pos
                if pos:
                    hasher.update(buff0[2*i])
                    hasher.update(buff0[2*i+1])
                    buff1[i] = hasher.compute().digest_value
                else:
                    hasher.update(buff1[2*i])
                    hasher.update(buff1[2*i+1])
                    buff0[i] = hasher.compute().digest_value
            nLayer //= 2
        self.digests['model'] = buff1[0] if pos else buff0[0]

    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self) -> collections.OrderedDict:
        digests = {}
        for key, trueHash in self.digests.items():
            digest = hashing.Digest(self.digest_name, bytes(range(64)))
            digests[key] = (digest, trueHash)
        return digests

    @property
    @override
    def digest_name(self) -> str:
        return "MerkleCPU"

    @property
    @override
    def digest_size(self) -> int:
        return self.hasher.digestSize

class MerkleGPU(hashing.StreamingHashEngine):
    # reduce factor is 1 for normal hashing
    # for lattice hash we spawn 8 times more threads to handle reduction
    # because each 64 byte digest can be represented as 8 uint64_t for summation
    def __init__(self, hashAlgo, mode=1):
        self.digestSize = hashAlgo.value[1]
        self.digests = {}
        self.mode = mode
        srcPath = os.path.join(srcPath, 'merkle.cuh')
        if self.mode == 2:
            self.ctx, [self.hashBlock, self.reduce] =
                nvrtc.rtcompile(srcPath, ['hash_dict', 'reduce'], hashAlgo.name)
        else:
            self.ctx, [self.hashBlock, self.reduce] =
                nvrtc.rtcompile(srcPath, ['hash_tensor', 'reduce'], hashAlgo.name)
    
    def _reduce_tree(self, nDigest, inputBuffer, outputBuffer, stream):
        inputBuffer = (inputBuffer, np.array([inputBuffer], dtype=np.uint64))
        outputBuffer = (outputBuffer, np.array([outputBuffer], dtype=np.uint64))
        while nDigest > 1:
            nDigest = (nDigest + 1) & ~0b1 # padding to multiple of 2
            nThread = nDigest // 2 # need half as many threads to compress
            block = min(512, nThread)
            grid = (nThread + (block-1)) // block
            nThread = np.array([nThread], dtype=np.uint64)
            args = [outputBuffer[1], inputBuffer[1], nThread]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.reduce, grid, 1, 1, block, 1, 1, (block+1)*self.digestSize,
                stream, args.ctypes.data, 0,
            ))
            nDigest = grid
            inputBuffer, outputBuffer = outputBuffer, inputBuffer
        return inputBuffer[0]
    
    @override
    def update(self, data: collections.OrderedDict, blockSize=8192) -> None:
        if self.mode == 0: # coalesced
            self.update_coalesced(data, blockSize)
        elif self.mode == 1: # separated
            self.update_perlayer(data, blockSize)
        elif self.mode == 2: # inplace
            self.update_inplace(data, blockSize)
    
    def update_coalesced(self, data: collections.OrderedDict, blockSize=8192) -> None:
        self.digests['model'] = bytes(self.digestSize)
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        total_size = sum(d.nbytes for d in data.values())
        nBlock = (total_size + (blockSize-1)) // blockSize

        stream = checkCudaErrors(runtime.cudaStreamCreate())
        iData = checkCudaErrors(runtime.cudaMalloc(nBlock*blockSize))
        oData = checkCudaErrors(runtime.cudaMalloc((nBlock+1)*self.digestSize))
        iData = (iData, np.array([iData], dtype=np.uint64))
        oData = (oData, np.array([oData], dtype=np.uint64))

        i = 0
        for v in data.values():
            checkCudaErrors(runtime.cudaMemcpyAsync(iData[0]+i, v.data_ptr(),
                v.nbytes, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))
            i += v.nbytes

        nThread = nBlock
        block = min(512, nThread)
        grid = (nThread + (block-1)) // block

        blockSize = np.array([blockSize], dtype=np.uint64)
        nThread = np.array([nThread], dtype=np.uint64)

        args = [oData[1], iData[1], blockSize, nThread]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        checkCudaErrors(driver.cuLaunchKernel(
            self.hashBlock, grid, 1, 1, block, 1, 1, 0, stream, args.ctypes.data, 0,
        ))

        while nBlock > 1:
            nBlock = ((nBlock + 1) & ~0b1)
            iData, oData = oData, iData
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

        checkCudaErrors(runtime.cudaMemcpyAsync(self.digests['model'], oData[0], self.digestSize,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))

        checkCudaErrors(runtime.cudaFreeAsync(iData[0], stream))
        checkCudaErrors(runtime.cudaFreeAsync(oData[0], stream))
        checkCudaErrors(runtime.cudaStreamSynchronize(stream))
        checkCudaErrors(runtime.cudaStreamDestroy(stream))

    def update_perlayer(self, data: collections.OrderedDict, blockSize=8192) -> None:
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        streams = [checkCudaErrors(runtime.cudaStreamCreate()) for i in range(len(data))]

        # min number of blocks for each layer
        nBlocks = [(v.nbytes + blockSize - 1) // blockSize for v in data.values()]
        # pad number of blocks for each layer to multiple of 2
        nBlocks = [(n + 1) & ~0b1 for n in nBlocks]
        iData = checkCudaErrors(runtime.cudaMalloc(sum(nBlocks)*self.digestSize))
        oData = checkCudaErrors(runtime.cudaMalloc(sum(nBlocks)*self.digestSize))
        checkCudaErrors(runtime.cudaMemset(iData, 0, sum(nBlocks)*self.digestSize))
        checkCudaErrors(runtime.cudaMemset(oData, 0, sum(nBlocks)*self.digestSize))

        for layer in data.keys():
            self.digests[layer] = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
        self.digests['model'] = checkCudaErrors(runtime.cudaMalloc(self.digestSize))

        blockSize = (blockSize, np.array([blockSize], dtype=np.uint64))
        currBlock = 0
        for stream, (layer, v), nBlock in zip(streams, data.items(), nBlocks):
            # hash data blocks to equal number of digests
            nThread = nBlock
            block = min(512, nThread)
            grid = (nThread + (block-1)) // block

            nBytes = np.array([v.nbytes], dtype=np.uint64)
            ptr = v.data_ptr() if hasattr(v, 'data_ptr') else v.data.ptr
            tensor = np.array([ptr], dtype=np.uint64)
            myIn = iData + self.digestSize * currBlock
            myOut = oData + self.digestSize * currBlock
            args = [np.array([myIn], dtype=np.uint64), tensor, blockSize[1], nBytes]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

            checkCudaErrors(driver.cuLaunchKernel(
                self.hashBlock, grid, 1, 1, block, 1, 1, 0, stream, args.ctypes.data, 0,
            ))

            # merkle tree reduction to single digest
            layerDigest = self._reduce_tree(nBlock, myIn, myOut, stream)
            checkCudaErrors(runtime.cudaMemcpyAsync(self.digests[layer], layerDigest,
                self.digestSize, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))
            currBlock += nBlock
        
        for i, (stream, key) in enumerate(zip(streams, data.keys())):
            checkCudaErrors(runtime.cudaStreamSynchronize(stream))
            checkCudaErrors(runtime.cudaStreamDestroy(stream))
            inBuff = iData + i * self.digestSize
            checkCudaErrors(runtime.cudaMemcpy(inBuff, self.digests[key],
                self.digestSize, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice))

        stream = checkCudaErrors(runtime.cudaStreamCreate())
        final = self._reduce_tree(len(data), iData, oData, stream)
        checkCudaErrors(runtime.cudaStreamSynchronize(stream))
        checkCudaErrors(runtime.cudaStreamDestroy(stream))

        checkCudaErrors(runtime.cudaMemcpy(self.digests['model'], final,
            self.digestSize, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice))

        checkCudaErrors(runtime.cudaFree(iData))
        checkCudaErrors(runtime.cudaFree(oData))

    def update_inplace(self, data: collections.OrderedDict, blockSize: int) -> None:
        self.digests['model'] = bytes(self.digestSize)
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
            self.hashBlock, grid, 1, 1, block, 1, 1, 0, stream, args.ctypes.data, 0,
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

        checkCudaErrors(runtime.cudaMemcpyAsync(self.digests['model'], oData[0], self.digestSize,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))

        checkCudaErrors(runtime.cudaFreeAsync(iData[0], stream))
        checkCudaErrors(runtime.cudaFreeAsync(oData[0], stream))
        checkCudaErrors(runtime.cudaStreamSynchronize(stream))
        checkCudaErrors(runtime.cudaStreamDestroy(stream))

    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self) -> collections.OrderedDict:
        digests = {}
        for key, trueHash in self.digests.items():
            digest = hashing.Digest(self.digest_name, bytes(range(64)))
            digests[key] = (digest, trueHash)
        return digests

    @property
    @override
    def digest_name(self) -> str:
        return f'MerkleGPU-{self.mode}'

    @property
    @override
    def digest_size(self) -> int:
        return self.digestSize

class HomomorphicGPU(hashing.StreamingHashEngine):
    def __init__(self, hashAlgo, inputType):
        self.digestSize = hashAlgo.value[1]
        self.allocatedSpace = 1
        self.iDataFull = checkCudaErrors(runtime.cudaMalloc(1))
        self.oDataFull = checkCudaErrors(runtime.cudaMalloc(1))
        self.digests = {}
        totalSum = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
        self.totalSum = (totalSum, np.array([totalSum], dtype=np.uint64))

        srcPath = os.path.join(srcPath, hashAlgo.value[0])
        if inputType == InputType.MODULE:
            self.ctx, [self.hashBlock, self.adder] = nvrtc.rtcompile(srcPath,
                ['hash_ltHash', 'reduce_ltHash'])
        elif inputType == InputType.DIGEST:
            self.ctx, [self.hashBlock, self.adder] = nvrtc.rtcompile(srcPath,
                ['hash_dataset_ltHash', 'reduce_ltHash'])
    
    def __exit__(self):
        checkCudaErrors(runtime.cudaFree(self.iDataFull))
        checkCudaErrors(runtime.cudaFree(self.oDataFull))
        for s in self.digests:
            checkCudaErrors(runtime.cudaFree(s[0]))
        checkCudaErrors(runtime.cudaFree(self.totalSum[0]))
    
    def __str__(self):
        hashString = ''
        for k in range(len(self.digests)):
            v = self.digests[k]
            tmp = bytes(self.digestSize)
            checkCudaErrors(runtime.cudaMemcpy(tmp, v[0], self.digestSize,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            hashString += f'{k}: {tmp.hex()[:16]}\n'
        return hashString

    @override
    def update(self, data: collections.OrderedDict, blockSize) -> None:
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        streams = [checkCudaErrors(runtime.cudaStreamCreate()) for _ in range(len(data))]

        blockSizeA = np.array([blockSize], dtype=np.uint64)
        unsortedLayers = []
        totalBlocks = 0
        for k, v in data.items():
            # register new hash category in dictionary
            if k not in self.digests:
                mem = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
                self.digests[k] = (mem, np.array([mem], dtype=np.uint64))
            unsortedLayers.append((k, v.data_ptr(), v.nbytes))
            sortedLayers = sorted(unsortedLayers, key=lambda x: x[2])
            totalBlocks += (v.nbytes + blockSize - 1) // blockSize
        
        spaceToAlloc = totalBlocks * self.digestSize
        if self.allocatedSpace < spaceToAlloc:
            checkCudaErrors(runtime.cudaFree(self.iDataFull))
            checkCudaErrors(runtime.cudaFree(self.oDataFull))
            self.iDataFull = checkCudaErrors(runtime.cudaMalloc(spaceToAlloc))
            self.oDataFull = checkCudaErrors(runtime.cudaMalloc(spaceToAlloc))
            self.allocatedSpace = spaceToAlloc
        outputStored = [True] * len(data.values()) # True: iData, False: oData
        currBytes = 0

        for i, (stream, (_, value, size)) in enumerate(zip(streams, sortedLayers)):
            sizeA = np.array([size], dtype=np.uint64)
            nBlock = (size + (blockSize-1)) // blockSize
            tensor = np.array([value], dtype=np.uint64)
            iData = np.array([self.iDataFull + currBytes], dtype=np.uint64)
            oData = np.array([self.oDataFull + currBytes], dtype=np.uint64)

            args = [oData, tensor, blockSizeA, sizeA]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            nThread = nBlock
            block = min(512, nThread)
            grid = (nThread + block - 1) // block

            checkCudaErrors(driver.cuLaunchKernel(
                self.hashBlock, grid, 1, 1, block, 1, 1, 0, stream,
                args.ctypes.data, 0,
            ))
            outputStored[i] = not outputStored[i]

            while nBlock > 1:
                iData, oData = oData, iData
                nBlock = ((nBlock + 1) & ~0b1)
                nThread = (nBlock // 2) * 8
                block = min(512, nThread)
                grid = (nThread + (block-1)) // block
                args = [oData, iData]
                args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

                checkCudaErrors(driver.cuLaunchKernel(
                    self.adder, grid, 1, 1, block, 1, 1, block * self.digestSize,
                    stream, args.ctypes.data, 0,
                ))
                checkCudaErrors(runtime.cudaStreamSynchronize(stream))
                nBlock = grid
                outputStored[i] = not outputStored[i]
            currBytes += nBlock * self.digestSize
        
        currBytes = 0
        for i, (stream, (key, _, size)) in enumerate(zip(streams, sortedLayers)):
            checkCudaErrors(runtime.cudaStreamSynchronize(stream))
            base = self.iDataFull if outputStored[i] else self.oDataFull
            layerDigest = np.array([base + currBytes], dtype=np.uint64)

            # add layer hash to layer sum
            args = [self.digests[key][1], layerDigest]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.adder, 1, 1, 1, 8, 1, 1, self.digestSize,
                stream, args.ctypes.data, 0,
            ))

            # add layer hash to total sum
            args = [self.totalSum[1], layerDigest]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.adder, 1, 1, 1, 8, 1, 1, self.digestSize,
                stream, args.ctypes.data, 0,
            ))

            checkCudaErrors(runtime.cudaStreamSynchronize(stream))
            checkCudaErrors(runtime.cudaStreamDestroy(stream))
            nBlock = (size + (blockSize-1)) // blockSize
            currBytes += nBlock * self.digestSize
    
    def update_dataset(self, data: collections.OrderedDict, blockSize) -> None:
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        streams = [checkCudaErrors(runtime.cudaStreamCreate()) for _ in range(len(data))]

        blockSizeA = np.array([blockSize], dtype=np.uint64)
        totalBlocks = 0
        for source, samples in data.items():
            # register new hash category in dictionary
            if source not in self.digests:
                mem = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
                self.digests[source] = (mem, np.array([mem], dtype=np.uint64))
            partitionBytes = sum([v.nbytes for v in samples])
            totalBlocks += (partitionBytes + blockSize - 1) // blockSize
        
        spaceToAlloc = (totalBlocks + 1) * self.digestSize
        if self.allocatedSpace < spaceToAlloc:
            checkCudaErrors(runtime.cudaFree(self.iDataFull))
            checkCudaErrors(runtime.cudaFree(self.oDataFull))
            self.iDataFull = checkCudaErrors(runtime.cudaMalloc(spaceToAlloc))
            self.oDataFull = checkCudaErrors(runtime.cudaMalloc(spaceToAlloc))
            self.allocatedSpace = spaceToAlloc

        currBytes = 0
        for stream, (src, samples) in zip(streams, data.items()):
            samplePtrs = np.array([s.data.ptr for s in samples], dtype=np.uint64)
            samplePtrsA = checkCudaErrors(runtime.cudaMallocAsync(samplePtrs.nbytes, stream))
            checkCudaErrors(runtime.cudaMemcpyAsync(samplePtrsA, samplePtrs.ctypes.data,
                samplePtrs.nbytes, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
            samplePtrsA = (samplePtrsA, np.array([samplePtrsA], dtype=np.uint64))
            sampleSizes = cp.array([sample.nbytes for sample in samples])
            sampleSizesA = np.array([sampleSizes.data.ptr], dtype=np.uint64)

            nA = np.array([len(samples)], dtype=np.uint64)
            iData = np.array([self.iDataFull + currBytes], dtype=np.uint64)
            oData = np.array([self.oDataFull + currBytes], dtype=np.uint64)

            args = [oData, samplePtrsA[1], sampleSizesA, nA]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.hashBlock, 1, 1, 1, len(samples), 1, 1, 0, stream,
                args.ctypes.data, 0,
            ))
            checkCudaErrors(runtime.cudaFreeAsync(samplePtrsA[0], stream))

            nDigests = len(samples)
            while nDigests > 1:
                iData, oData = oData, iData
                nThread = ((nDigests + 1) >> 1) * 8
                block = min(512, nThread)
                grid = (nThread + (block-1)) // block
                args = [oData, iData, np.array([nThread], dtype=np.uint64)]
                args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
                checkCudaErrors(driver.cuLaunchKernel(
                    self.adder, grid, 1, 1, block, 1, 1, block * self.digestSize,
                    stream, args.ctypes.data, 0,
                ))
                nDigests = grid
            
            # add layer hash to layer sum
            args = [self.digests[src][1], oData, np.array([8], dtype=np.uint64)]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.adder, 1, 1, 1, 8, 1, 1, self.digestSize,
                stream, args.ctypes.data, 0,
            ))

            currBytes += len(samples) * self.digestSize

        for stream in streams:
            checkCudaErrors(runtime.cudaStreamSynchronize(stream))
            checkCudaErrors(runtime.cudaStreamDestroy(stream))

    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self) -> collections.OrderedDict:
        digests = {}
        for key, trueHash in self.digests.items():
            digest = hashing.Digest(self.digest_name, bytes(range(32)))
            digests[key] = (digest, trueHash[0])
        return digests

    @property
    @override
    def digest_name(self) -> str:
        return "HomomorphicGPU"

    @property
    @override
    def digest_size(self) -> int:
        return self.digestSize
