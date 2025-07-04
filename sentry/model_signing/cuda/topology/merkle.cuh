#if defined(SHA256)
    #include "../algorithm/sha256.cuh"
#elif defined(BLAKE2)
    #include "../algorithm/blake2.cuh"
#elif defined(SHA3)
    #include "../algorithm/sha3.cuh"
#endif

#define OUT_BYTES 32UL

extern "C" __global__
void hash_tensor(unsigned char *out, unsigned char *in, unsigned long blockSize, unsigned long size) {
	CTX ctx;
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char *myIn = in + idx * blockSize;
    unsigned long rem = (in + size) - myIn;
	if (myIn < in + size) {
        init(&ctx);
        update(&ctx, myIn, rem < blockSize ? rem : blockSize);
        final(&ctx, &out[idx * OUT_BYTES]);
    }
}

extern "C" __global__
void hash_dict(unsigned char *out, unsigned long blockSize, unsigned long *startThread,
	unsigned long *workSize, unsigned char **workAddr, unsigned long l, unsigned long n) {
	CTX ctx;
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
        init(&ctx);
		unsigned long workId = 0;
		while (workId < l && idx >= startThread[workId]) {
			workId++;
		}
		workId--;
		unsigned char *my_in = workAddr[workId] + blockSize * (idx - startThread[workId]);
		unsigned char *workEnd = workAddr[workId] + workSize[workId];
		update(&ctx, my_in, blockSize < workEnd - my_in ? blockSize : workEnd - my_in);
    }
    __syncthreads();
	if (idx < n)
        final(&ctx, &out[idx * OUT_BYTES]);
}

extern "C" __global__
void reduce(unsigned char *out, unsigned char *in, size_t n) {
	extern __shared__ unsigned char shMem[];
	CTX ctx;
	int glbIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int locIdx = threadIdx.x;
	int activeThreads = min((unsigned long)blockDim.x, n - (blockIdx.x * blockDim.x));
    if (locIdx == 0)
        memset(&shMem[(blockDim.x)*OUT_BYTES], 0, OUT_BYTES);
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
        memcpy(out + blockIdx.x*OUT_BYTES, shMem, OUT_BYTES);
    }
}
