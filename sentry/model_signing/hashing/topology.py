from . import hashing, memory
import numpy as np
import cupy as cp
from cuda.bindings import driver, runtime
from ..cuda.compiler import compileCuda, checkCudaErrors
import collections
import threading
import os
from enum import Enum, IntEnum

from typing_extensions import override

# (source file, digest size, hashlib)
class HashAlgo(Enum):
    SHA256   = ('../algorithm/sha256.cuh', 32, memory.SHA256)
    BLAKE2B  = ('../algorithm/blake2b.cuh', 64, memory.BLAKE2)
    SHA3     = ('../algorithm/sha3.cuh', 64, memory.SHA3)
    BLAKE2XB = ('../algorithm/blake2xb.cuh', 64, None)

class Topology(Enum):
    SERIAL  = 'serial.cuh'
    MERKLE  = 'merkle.cuh'
    LATTICE = 'lattice.cuh'

class Workflow(IntEnum):
    COALESCED       = 0
    LAYERED         = 1
    INPLACE         = 2
    LAYERED_SORTED  = 3

srcPath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'model_signing', 'cuda', 'topology')

class SeqGPU(hashing.StreamingHashEngine):
    def __init__(self, hashAlgo):
        self.digestSize = hashAlgo.value[1]
        global srcPath
        myPath = os.path.join(srcPath, 'serial.cuh')
        self.ctx, [self.hash] = compileCuda(myPath, [f'hash'], [hashAlgo.value[0]])
    
    @override
    def update(self, data: collections.OrderedDict, blockSize=8192) -> None:
        self.digest = bytes(self.digestSize)
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        total_size = sum(d.nbytes for d in data.values())
        stream = checkCudaErrors(runtime.cudaStreamCreate())
        iData = checkCudaErrors(runtime.cudaMalloc(total_size))
        oData = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
        i = 0
        for v in data.values():
            ptr = v.data_ptr() if hasattr(v, 'data_ptr') else v.data.ptr
            checkCudaErrors(runtime.cudaMemcpyAsync(iData+i, ptr, v.nbytes,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))
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
        dummyDigest = hashing.Digest(self.digest_name, bytes(self.digest_size))
        return dummyDigest, self.digest

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
        self.digests['all'] = bytearray(self.digestSize)

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
        self.digests['all'] = buff1[0] if pos else buff0[0]

    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self) -> collections.OrderedDict:
        digests = {}
        for key, trueHash in self.digests.items():
            digest = hashing.Digest(self.digest_name, bytes(self.digest_size))
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

class HasherGPU(hashing.StreamingHashEngine):
    def __init__(self, hashAlgo, topology, workflow):
        self.digestSize = hashAlgo.value[1]
        self.digests = {}
        self.hashAlgo = hashAlgo
        self.topology = topology
        self.workflow = workflow
        global srcPath
        myPath = os.path.join(srcPath, topology.value)
        self.ctx, [self.hashBlock, self.reduce] = compileCuda(
            myPath, ['hashBlock', 'reduce'], [hashAlgo.value[0]], [workflow.name])

    @override
    def update(self, data: collections.OrderedDict, blockSize=8192) -> None:
        if self.workflow == Workflow.COALESCED:
            self.update_coalesced(data, blockSize)
        elif self.workflow == Workflow.LAYERED or self.workflow == Workflow.LAYERED_SORTED:
            self.update_perlayer(data, blockSize)
        elif self.workflow == Workflow.INPLACE:
            self.update_inplace(data, blockSize)

    def update_coalesced(self, data: collections.OrderedDict, blockSize=8192):
        self.digests['all'] = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        total_size = sum(d.nbytes for d in data.values())
        nBlock = (total_size + (blockSize-1)) // blockSize
        stream = checkCudaErrors(runtime.cudaStreamCreate())
        iData = checkCudaErrors(runtime.cudaMalloc(nBlock*blockSize))
        oData = checkCudaErrors(runtime.cudaMalloc((nBlock+1)*self.digestSize))
        iData = (iData, np.array([iData], dtype=np.uint64))
        oData = (oData, np.array([oData], dtype=np.uint64))

        i = 0
        for tensor in data.values():
            ptr = tensor.data_ptr() if hasattr(tensor, 'data_ptr') else tensor.data.ptr
            checkCudaErrors(runtime.cudaMemcpy(iData[0]+i, ptr, tensor.nbytes,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
            i += tensor.nbytes

        nThread = nBlock
        block = min(512, nThread)
        grid = (nThread + (block-1)) // block

        blockSize = np.array([blockSize], dtype=np.uint64)
        nThread = np.array([nThread], dtype=np.uint64)

        args = [oData[1], iData[1], blockSize, nBlock*blockSize]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        checkCudaErrors(driver.cuLaunchKernel(
            self.hashBlock, grid, 1, 1, block, 1, 1, 0, stream, args.ctypes.data, 0,
        ))

        # merkle tree reduction to single digest
        if self.topology == Topology.MERKLE:
            threadPerReduce = 1
        elif self.topology == Topology.LATTICE:
            threadPerReduce = 8
        finalDigest = _reduce(self.reduce, oData[0], iData[0], nBlock,
            self.digestSize, threadPerReduce, stream)
        checkCudaErrors(runtime.cudaMemcpyAsync(self.digests['all'], finalDigest,
            self.digestSize, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))

        checkCudaErrors(runtime.cudaFreeAsync(iData[0], stream))
        checkCudaErrors(runtime.cudaFreeAsync(oData[0], stream))
        checkCudaErrors(runtime.cudaStreamSynchronize(stream))
        checkCudaErrors(runtime.cudaStreamDestroy(stream))

    def update_perlayer(self, data: collections.OrderedDict, blockSize=8192):
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        streams = [checkCudaErrors(runtime.cudaStreamCreate()) for _ in range(len(data))]

        nBlocks = []
        for layerName, tensor in data.items():
            self.digests[layerName] = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
            nBlock = ((tensor.nbytes + blockSize - 1) // blockSize)
            # pad number of blocks to multiple of 2
            nBlocks.append((nBlock + 1) & ~0b1)
        self.digests['all'] = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
        spaceToAlloc = sum(nBlocks)*self.digestSize
        iData = checkCudaErrors(runtime.cudaMalloc(spaceToAlloc))
        oData = checkCudaErrors(runtime.cudaMalloc(spaceToAlloc))
        checkCudaErrors(runtime.cudaMemset(iData, 0, spaceToAlloc))
        checkCudaErrors(runtime.cudaMemset(oData, 0, spaceToAlloc))
        if self.workflow == Workflow.LAYERED_SORTED:
            sortedLayer = sorted(data, key=lambda x: data[x].nbytes)
            data = {k: data[k] for k in sortedLayer}

        blockSize = np.array([blockSize], dtype=np.uint64)
        currBlock = 0
        for stream, (layerName, tensor), nBlock in zip(streams, data.items(), nBlocks):
            nThread = nBlock
            nBytes = np.array([tensor.nbytes], dtype=np.uint64)
            ptr = tensor.data_ptr() if hasattr(tensor, 'data_ptr') else tensor.data.ptr
            tensor_ptr = np.array([ptr], dtype=np.uint64)
            myIn = iData + self.digestSize * currBlock
            myOut = oData + self.digestSize * currBlock
            args = [np.array([myIn], dtype=np.uint64), tensor_ptr, blockSize, nBytes]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            block = min(512, nThread)
            grid = (nThread + (block-1)) // block
            checkCudaErrors(driver.cuLaunchKernel(
                self.hashBlock, grid, 1, 1, block, 1, 1, 0, stream,
                args.ctypes.data, 0,
            ))

            if self.topology == Topology.MERKLE:
                threadPerReduce = 1
            elif self.topology == Topology.LATTICE:
                threadPerReduce = 8
            layerDigest = _reduce(self.reduce, myIn, myOut, nBlock,
                self.digestSize, threadPerReduce, stream)
            checkCudaErrors(runtime.cudaMemcpyAsync(self.digests[layerName], layerDigest,
                self.digestSize, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))
            currBlock += nBlock
        
        for i, (stream, layerName) in enumerate(zip(streams, data.keys())):
            checkCudaErrors(runtime.cudaStreamSynchronize(stream))
            checkCudaErrors(runtime.cudaStreamDestroy(stream))
            inBuff = iData + i * self.digestSize
            checkCudaErrors(runtime.cudaMemcpy(inBuff, self.digests[layerName],
                self.digestSize, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice))

        stream = checkCudaErrors(runtime.cudaStreamCreate())
        if self.topology == Topology.MERKLE:
            threadPerReduce = 1
        elif self.topology == Topology.LATTICE:
            threadPerReduce = 8
        finalDigest = _reduce(self.reduce, iData, oData, len(data),
            self.digestSize, threadPerReduce, stream)
        checkCudaErrors(runtime.cudaStreamSynchronize(stream))
        checkCudaErrors(runtime.cudaStreamDestroy(stream))

        checkCudaErrors(runtime.cudaMemcpy(self.digests['all'], finalDigest,
            self.digestSize, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice))

        checkCudaErrors(runtime.cudaFree(iData))
        checkCudaErrors(runtime.cudaFree(oData))

    def update_inplace(self, data: collections.OrderedDict, blockSize: int):
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        self.digests['all'] = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
        stream = checkCudaErrors(runtime.cudaStreamCreate())
        startThread = []
        workAddr = []
        workSize = []
        nBlock = 0
        for v in data.values():
            ptr = v.data_ptr() if hasattr(v, 'data_ptr') else v.data.ptr
            startThread.append(nBlock)
            workAddr.append(ptr)
            workSize.append(v.nbytes)
            nBlock += (v.nbytes + (blockSize-1)) // blockSize
        
        startThread = cp.array(startThread, dtype=np.uint64)
        workAddr = cp.array(workAddr, dtype=np.uint64)
        workSize = cp.array(workSize, dtype=np.uint64)

        nThread = nBlock
        block = min(512, nThread)
        grid = (nThread + (block-1)) // block
        iData = checkCudaErrors(runtime.cudaMalloc((grid+1) * block * self.digestSize))
        oData = checkCudaErrors(runtime.cudaMalloc((grid+1) * block * self.digestSize))
        iData = (iData, np.array([iData], dtype=np.uint64))
        oData = (oData, np.array([oData], dtype=np.uint64))
        blockSize = np.array([blockSize], dtype=np.uint64)
        startT = np.array([startThread.data.ptr], dtype=np.uint64)
        workAddr = np.array([workAddr.data.ptr], dtype=np.uint64)
        workSize = np.array([workSize.data.ptr], dtype=np.uint64)
        workLen = np.array([len(startThread)], dtype=np.uint64)
        nThread = np.array([nThread], dtype=np.uint64)

        args = [iData[1], blockSize, startT, workSize, workAddr, workLen, nThread]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        checkCudaErrors(driver.cuLaunchKernel(
            self.hashBlock, grid, 1, 1, block, 1, 1, 0, stream, args.ctypes.data, 0,
        ))

        # merkle tree reduction to single digest
        if self.topology == Topology.MERKLE:
            threadPerReduce = 1
        elif self.topology == Topology.LATTICE:
            threadPerReduce = 8
        finalDigest = _reduce(self.reduce, iData[0], oData[0], nBlock,
            self.digestSize, threadPerReduce, stream)
        checkCudaErrors(runtime.cudaMemcpyAsync(self.digests['all'], finalDigest,
            self.digestSize, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))

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
        for layerName, trueHash in self.digests.items():
            digest = hashing.Digest(self.digest_name, bytes(range(self.digest_size)))
            digests[layerName] = (digest, trueHash)
        return digests

    @property
    @override
    def digest_name(self) -> str:
        return f'{self.topology.name}-{self.workflow.name}-{self.hashAlgo.name}'

    @property
    @override
    def digest_size(self) -> int:
        return self.digestSize

class DatasetHasherGPU(hashing.StreamingHashEngine):
    def __init__(self, hashAlgo, topology):
        self.hashAlgo = hashAlgo
        self.topology = topology
        self.digestSize = hashAlgo.value[1]
        self.digests = {}

        global srcPath
        myPath = os.path.join(srcPath, topology.value)
        self.ctx, [self.hashBlock, self.reduce] = compileCuda(
            myPath, ['hash_dataset', 'reduce'], [hashAlgo.value[0]])

    @override
    def update(self, data: collections.OrderedDict, blockSize) -> None:
        checkCudaErrors(driver.cuCtxSetCurrent(self.ctx))
        blockSizeA = np.array([blockSize], dtype=np.uint64)
        iData = []
        oData = []
        streams = []
        for identity, samples in data.items():
            streams.append(checkCudaErrors(runtime.cudaStreamCreate()))
            if identity not in self.digests:
                self.digests[identity] = checkCudaErrors(runtime.cudaMalloc(self.digestSize))
                checkCudaErrors(runtime.cudaMemset(self.digests[identity], 0, self.digestSize))
            numBytes = sum([v.nbytes for v in samples])
            numBlk = (numBytes + blockSize - 1) // blockSize
            if numBlk % 2 == 1:
                numBlk += 1
            iData.append(checkCudaErrors(runtime.cudaMalloc(numBlk*self.digestSize)))
            oData.append(checkCudaErrors(runtime.cudaMalloc(numBlk*self.digestSize)))
            bytesOffset = (numBlk - 1) * self.digestSize
            checkCudaErrors(runtime.cudaMemset(iData[-1]+bytesOffset, 0, self.digestSize))
            checkCudaErrors(runtime.cudaMemset(oData[-1]+bytesOffset, 0, self.digestSize))

        for stream, iDatum, oDatum, (identity, samples) in zip(streams, iData, oData, data.items()):
            iDatum = np.array([iDatum], dtype=np.uint64)
            oDatum = np.array([oDatum], dtype=np.uint64)

            samplePtrs = cp.array([s.data.ptr for s in samples], dtype=np.uint64)
            nA = np.array([len(samples)], dtype=np.uint64)
            args = [oDatum, np.array([samplePtrs.data.ptr], dtype=np.uint64), blockSizeA, nA]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.hashBlock, 1, 1, 1, len(samples), 1, 1, 0, stream,
                args.ctypes.data, 0,
            ))

            nDigests = len(samples)
            while nDigests > 1:
                iDatum, oDatum = oDatum, iDatum
                nThread = ((nDigests + 1) >> 1) * 8
                block = min(512, nThread)
                grid = (nThread + (block-1)) // block
                args = [oDatum, iDatum, np.array([nThread], dtype=np.uint64)]
                args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
                checkCudaErrors(driver.cuLaunchKernel(
                    self.reduce, grid, 1, 1, block, 1, 1, block * self.digestSize,
                    stream, args.ctypes.data, 0,
                ))
                nDigests = grid

            # add layer hash to layer sum
            digestSum = np.array([self.digests[identity]], dtype=np.uint64)
            args = [digestSum, oDatum, np.array([8], dtype=np.uint64)]
            args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
            checkCudaErrors(driver.cuLaunchKernel(
                self.reduce, 1, 1, 1, 8, 1, 1, self.digestSize,
                stream, args.ctypes.data, 0,
            ))

        for stream in streams:
            checkCudaErrors(runtime.cudaStreamSynchronize(stream))
            checkCudaErrors(runtime.cudaStreamDestroy(stream))

    @override
    def reset(self, data: bytes = b"") -> None:
        pass

    @override
    def compute(self) -> collections.OrderedDict:
        digests = {}
        for identity, trueHash in self.digests.items():
            digest = hashing.Digest(self.digest_name, bytes(range(self.digest_size)))
            digests[identity] = (digest, trueHash)
        return digests

    @property
    @override
    def digest_name(self) -> str:
        return f'{self.topology.name}-{Workflow.LAYERED.name}-{self.hashAlgo.name}'

    @property
    @override
    def digest_size(self) -> int:
        return self.digestSize

def get_model_hasher(hashAlgo: HashAlgo, topology: Topology, workflow: Workflow, device: str):
    hasher = None
    if device == 'cpu':
        if topology >= Topology.MERKLE_LAYERED and topology <= Topology.MERKLE_INPLACE:
            hasher = MerkleCPU(hashAlgo.value[2])
    elif device == 'gpu':
        if topology == Topology.SERIAL:
            hasher = SeqGPU(hashAlgo)
        else:
            hasher = HasherGPU(hashAlgo, topology, workflow)

    if not hasher:
        raise NotImplementedError(f'Unsupported for {hashAlgo.name}, \
            {topology.name}, {workflow.name}, {device}')

    return hasher

# utility functions

def _reduce(reduceKernel, iData, oData, nDigest, digestSize, threadPerReduce, stream):
    iData = (iData, np.array([iData], dtype=np.uint64))
    oData = (oData, np.array([oData], dtype=np.uint64))
    while nDigest > 1:
        # padding to multiple of 2
        nDigest = (nDigest + 1) & ~0b1
        nThread = (nDigest // 2) * threadPerReduce
        block = min(512, nThread)
        grid = (nThread + (block-1)) // block
        nThread = np.array([nThread], dtype=np.uint64)
        args = [oData[1], iData[1], nThread]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        checkCudaErrors(driver.cuLaunchKernel(
            reduceKernel, grid, 1, 1, block, 1, 1, (block+1)*digestSize,
            stream, args.ctypes.data, 0,
        ))
        nDigest = grid
        iData, oData = oData, iData

    return iData[0]
