#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#define OUTBYTES 32

#define sequential(init, update, final, output, input, blockSize, n) { \
    const unsigned long k = 0xFEDCBA9876543210UL; \
    init(&ctx); \
    for (size_t i = 0; i < n; i++) { \
        update(&ctx, &input[i * blockSize], blockSize); \
    } \
    final(&ctx, output); \
} \

#define merkle_pre(init, update, final, output, input, blockSize, n) { \
    int i = blockIdx.x * blockDim.x + threadIdx.x; \
    const unsigned long k = 0xFEDCBA9876543210UL; \
	if (i < n) { \
        init(&ctx); \
        update(&ctx, &input[i * blockSize], blockSize); \
    } \
	__syncthreads(); \
	if (i < n) \
        final(&ctx, &output[i * OUTBYTES]); \
} \

#define merkle_step(init, update, final, shMem, output, input, n) { \
	int glbIdx = blockIdx.x * blockDim.x + threadIdx.x; \
	int locIdx = threadIdx.x; \
    if (glbIdx < n) { \
        init(&ctx); \
		update(&ctx, &input[(2*glbIdx)*OUTBYTES], OUTBYTES); \
		update(&ctx, &input[(2*glbIdx+1)*OUTBYTES], OUTBYTES); \
        final(&ctx, &shMem[locIdx*OUTBYTES]); \
	} \
    for (int block = blockDim.x / 2; block >= 1; block /= 2) { \
		if (glbIdx < n && locIdx < block) { \
			update(&ctx, &shMem[(2*locIdx)*OUTBYTES], OUTBYTES); \
			update(&ctx, &shMem[(2*locIdx+1)*OUTBYTES], OUTBYTES); \
		} \
		__syncthreads(); \
		if (glbIdx < n && locIdx < block) \
            final(&ctx, &shMem[locIdx*OUTBYTES]); \
	} \
    if (locIdx == 0) { \
        memcpy(&output[blockIdx.x*(blockDim.x*2)*OUTBYTES], shMem, OUTBYTES); \
    } \
} \

#endif