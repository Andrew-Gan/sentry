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
    uint64_t myIn = in + idx * blockSize; \
	if (myIn < in + size) { \
        init(&ctx); \
        update(&ctx, in + idx * blockSize, blockSize); \
        final(&ctx, &out[idx * outBytes]); \
    } \
} \

#define merkle_step(init, update, final, outBytes) { \
	int glbIdx = blockIdx.x * blockDim.x + threadIdx.x; \
	int locIdx = threadIdx.x; \
    if (glbIdx < n) { \
        init(&ctx); \
		update(&ctx, &in[(2*glbIdx)*outBytes], outBytes); \
		update(&ctx, &in[(2*glbIdx+1)*outBytes], outBytes); \
        final(&ctx, &shMem[locIdx*outBytes]); \
	} \
    for (int block = blockDim.x / 2; block >= 1; block /= 2) { \
		if (glbIdx < n && locIdx < block) { \
			update(&ctx, &shMem[(2*locIdx)*outBytes], outBytes); \
			update(&ctx, &shMem[(2*locIdx+1)*outBytes], outBytes); \
		} \
		__syncthreads(); \
		if (glbIdx < n && locIdx < block) {\
            final(&ctx, &shMem[locIdx*outBytes]); \
        } \
        __syncthreads(); \
	} \
    if (locIdx == 0) { \
        memcpy(out + blockIdx.x*outBytes, shMem, outBytes); \
    } \
} \

#endif
