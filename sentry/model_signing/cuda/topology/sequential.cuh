#if defined(SHA256)
    #include "../algorithm/sha256.cuh"
#elif defined(BLAKE2)
    #include "../algorithm/blake2.cuh"
#elif defined(SHA3)
    #include "../algorithm/sha3.cuh"
#endif

extern "C" __global__
void hash(unsigned char *out, unsigned char *in, unsigned long blockSize, unsigned long n) {
	CTX ctx;
    init(&ctx);
    for (unsigned long i = 0; i < n; i++) {
        update(&ctx, &in[i * blockSize], blockSize);
    }
    final(&ctx, out);
}
