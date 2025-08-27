extern "C" __global__
void hash(uint8_t *out, uint8_t *in, uint64_t blockSize, uint64_t n) {
	CTX ctx;
    init(&ctx);
    for (uint64_t i = 0; i < n; i++) {
        update(&ctx, &in[i * blockSize], blockSize);
    }
    final(&ctx, out);
}
