#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#define uint8_t unsigned char
#define uint32_t unsigned int
#define uint64_t unsigned long

#define sequential(init, update, final) { \
    init(&ctx); \
    for (uint64_t i = 0; i < n; i++) { \
        update(&ctx, &in[i * blockSize], blockSize); \
    } \
    final(&ctx, out); \
} \

#define merkle_pre(init, update, final, outBytes) { \
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
    uint8_t *myIn = in + idx * blockSize; \
	uint8_t *myEnd = myIn + blockSize; \
    uint64_t rem = (in + size) - myIn; \
	if (myIn < in + size) { \
        init(&ctx); \
        update(&ctx, myIn, rem < blockSize ? rem : blockSize); \
        final(&ctx, &out[idx * outBytes]); \
    } \
} \

#define merkle_step(init, update, final, outBytes) { \
	int glbIdx = blockIdx.x * blockDim.x + threadIdx.x; \
	int locIdx = threadIdx.x; \
	int activeThreads = min((uint64_t)blockDim.x, n - (blockIdx.x * blockDim.x)); \
    if (locIdx == 0) \
        memset(&shMem[(blockDim.x)*outBytes], 0, outBytes); \
    if (locIdx < activeThreads) { \
        init(&ctx); \
		update(&ctx, &in[(2*glbIdx)*outBytes], outBytes); \
		update(&ctx, &in[(2*glbIdx+1)*outBytes], outBytes); \
        final(&ctx, &shMem[locIdx*outBytes]); \
	} \
    __syncthreads(); \
    if (activeThreads > 1) { \
        activeThreads = (activeThreads + 1) / 2; \
        for (; activeThreads > 0; activeThreads /= 2) { \
            if (locIdx < activeThreads) { \
                update(&ctx, &shMem[(2*locIdx)*outBytes], outBytes); \
                update(&ctx, &shMem[(2*locIdx+1)*outBytes], outBytes); \
            } \
            __syncthreads(); \
            if (locIdx < activeThreads) \
                final(&ctx, &shMem[locIdx*outBytes]); \
            __syncthreads(); \
            if (activeThreads > 1 && activeThreads & 0b1 == 1) activeThreads++; \
        } \
    } \
    if (locIdx == 0) { \
        memcpy(out + blockIdx.x*outBytes, shMem, outBytes); \
    } \
} \

#endif
