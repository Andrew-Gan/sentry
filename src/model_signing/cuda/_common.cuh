#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#define OUTBYTES 32
#define uint8_t unsigned char
#define uint64_t unsigned long

#define sequential(init, update, final) { \
    init(&ctx); \
    for (uint64_t i = 0; i < n; i++) { \
        update(&ctx, &in[i * blockSize], blockSize); \
    } \
    final(&ctx, out); \
} \

#define merkle_pre(init, update, final) { \
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
	uint8_t *my_in = in + idx * blockSize; \
	if (idx < nThread) { \
        init(&ctx); \
        update(&ctx, my_in, my_in + blockSize < in + sBytes ? blockSize : in + sBytes - my_in); \
    } \
	__syncthreads(); \
	if (idx < nThread) \
        final(&ctx, &out[idx * OUTBYTES]); \
} \

#define merkle_step(init, update, final) { \
	int glbIdx = blockIdx.x * blockDim.x + threadIdx.x; \
	int locIdx = threadIdx.x; \
    if (glbIdx < n) { \
        init(&ctx); \
		update(&ctx, &in[(2*glbIdx)*OUTBYTES], OUTBYTES); \
		update(&ctx, &in[(2*glbIdx+1)*OUTBYTES], OUTBYTES); \
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
    if (locIdx == 0) \
        memcpy(&out[blockIdx.x*(blockDim.x*2)*OUTBYTES], shMem, OUTBYTES); \
} \

#endif