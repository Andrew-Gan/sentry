#if defined(COALESCED) || defined(LAYERED)
extern "C" __global__
void hashBlock(uint8_t *out, uint8_t *in, uint64_t blockSize, uint64_t nbytes) {
    CTX ctx;
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t *myIn = in + idx * blockSize;
    if (myIn >= in + nbytes) return;
    uint64_t rem = (in + nbytes) - myIn;

    init(&ctx);
    update(&ctx, myIn, rem < blockSize ? rem : blockSize);
    final(&ctx, &out[idx * OUT_BYTES]);
}

#elif defined(INPLACE)
extern "C" __global__
void hashBlock(uint8_t *out, uint64_t blockSize, uint64_t *startThread,
    uint64_t *workSize, uint8_t **workAddr, uint64_t l, uint64_t nThread) {
    CTX ctx;
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nThread) {
        init(&ctx);
        uint64_t workId = 0;
        while (workId < l && idx >= startThread[workId]) {
            workId++;
        }
        workId--;
        uint8_t *my_in = workAddr[workId] + blockSize * (idx - startThread[workId]);
        uint8_t *workEnd = workAddr[workId] + workSize[workId];
        update(&ctx, my_in, blockSize < workEnd - my_in ? blockSize : workEnd - my_in);
    }
    __syncthreads();
    if (idx < nThread)
        final(&ctx, &out[idx * OUT_BYTES]);
}

#endif

extern "C" __global__
void reduce(uint8_t *out, uint8_t *in, uint64_t n) {
    extern __shared__ uint8_t shMem[];
    CTX ctx;
    int glbIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int locIdx = threadIdx.x;
    int activeThreads = min((uint64_t)blockDim.x, n - (blockIdx.x * blockDim.x));
    if (activeThreads % 2 == 1 && locIdx == 0)
        memset(&shMem[activeThreads*OUT_BYTES], 0, OUT_BYTES);
    if (locIdx < activeThreads) {
        init(&ctx);
        update(&ctx, &in[(2*glbIdx)*OUT_BYTES], OUT_BYTES);
        update(&ctx, &in[(2*glbIdx+1)*OUT_BYTES], OUT_BYTES);
        final(&ctx, &shMem[locIdx*OUT_BYTES]);
    }
    __syncthreads();
    if (activeThreads > 1) {
        activeThreads = (activeThreads + 1) / 2;
        for (; activeThreads > 0; activeThreads /= 2) {
            if (locIdx < activeThreads) {
                update(&ctx, &shMem[(2*locIdx)*OUT_BYTES], OUT_BYTES);
                update(&ctx, &shMem[(2*locIdx+1)*OUT_BYTES], OUT_BYTES);
            }
            __syncthreads();
            if (locIdx < activeThreads)
                final(&ctx, &shMem[locIdx*OUT_BYTES]);
            __syncthreads();
            if (activeThreads > 1 && activeThreads & 0b1 == 1) activeThreads++;
        }
    }
    if (locIdx == 0) {
        memcpy(out + blockIdx.x * OUT_BYTES, shMem, OUT_BYTES);
    }
}
