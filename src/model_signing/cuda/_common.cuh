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
	if (idx < n) { \
        init(&ctx); \
		uint64_t workId = 0; \
		while (workId < l && idx >= startThread[workId]) { \
			workId++; \
		} \
		workId--; \
		uint8_t *my_in = workAddr[workId] + blockSize * (idx - startThread[workId]); \
		uint8_t *workEnd = workAddr[workId] + workSize[workId]; \
		update(&ctx, my_in, blockSize < workEnd - my_in ? blockSize : workEnd - my_in); \
    } \
    __syncthreads(); \
	if (idx < n) \
        final(&ctx, &out[idx * outBytes]); \
} \

#define merkle_step(init, update, final, outBytes) { \
	int glbIdx = blockIdx.x * blockDim.x + threadIdx.x; \
	int locIdx = threadIdx.x; \
	int activeThreads = min((uint64_t)blockDim.x, n - (blockIdx.x * blockDim.x)); \
    memset(&shMem[locIdx*outBytes], 0, outBytes); \
    if (locIdx < activeThreads) { \
        init(&ctx); \
		update(&ctx, &in[(2*glbIdx)*outBytes], outBytes); \
		update(&ctx, &in[(2*glbIdx+1)*outBytes], outBytes); \
        final(&ctx, &shMem[locIdx*outBytes]); \
	} \
	__syncthreads(); \
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
    if (locIdx == 0) { \
        memcpy(&out[blockIdx.x*outBytes], shMem, outBytes); \
    } \
} \

#endif
