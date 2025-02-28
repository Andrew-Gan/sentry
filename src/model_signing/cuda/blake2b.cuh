/*
 * blake2b.cu CUDA Implementation of BLAKE2B Hashing
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is released into the Public Domain.
 */

#ifndef __BLAKE2B_H__
#define __BLAKE2B_H__

#include "_common.cuh"

#define BLAKE2B_ROUNDS 12
#define BLAKE2B_BLOCK_LENGTH 128
#define BLAKE2B_CHAIN_SIZE 8
#define BLAKE2B_CHAIN_LENGTH (BLAKE2B_CHAIN_SIZE * sizeof(long))
#define BLAKE2B_STATE_SIZE 16
#define BLAKE2B_STATE_LENGTH (BLAKE2B_STATE_SIZE * sizeof(long))

typedef struct {
    unsigned int digestlen;
    unsigned char key[64];
    unsigned int keylen;

    unsigned char buff[BLAKE2B_BLOCK_LENGTH];
    long chain[BLAKE2B_CHAIN_SIZE];
    long state[BLAKE2B_STATE_SIZE];

    unsigned int pos;
    long t0;
    long t1;
    long f0;
} BLAKE2B_CTX;

__constant__ long BLAKE2B_IVS[8] = {
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
};


__constant__ unsigned char BLAKE2B_SIGMAS[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
};

__device__ long cuda_blake2b_leuint64(unsigned char *in) {
    long a;
    memcpy(&a, in, 8);
    return a;

/* If memory is not little endian
unsigned char *a = (unsigned char *)in;
return ((long)(a[0]) << 0) | ((long)(a[1]) << 8) | ((long)(a[2]) << 16) | ((long)(a[3]) << 24) |((long)(a[4]) << 32)
    | ((long)(a[5]) << 40) | ((long)(a[6]) << 48) | 	((long)(a[7]) << 56);
 */
}

__device__ long cuda_blake2b_ROTR64(long a, unsigned char b) {
    return (a >> b) | (a << (64 - b));
}

__device__
void cuda_blake2b_G(BLAKE2B_CTX *ctx, long m1, long m2, int a, int b, int c, int d) {
    ctx->state[a] += ctx->state[b] + m1;
    ctx->state[d] = cuda_blake2b_ROTR64(ctx->state[d] ^ ctx->state[a], 32);
    ctx->state[c] += ctx->state[d];
    ctx->state[b] = cuda_blake2b_ROTR64(ctx->state[b] ^ ctx->state[c], 24);
    ctx->state[a] += ctx->state[b] + m2;
    ctx->state[d] = cuda_blake2b_ROTR64(ctx->state[d] ^ ctx->state[a], 16);
    ctx->state[c] += ctx->state[d];
    ctx->state[b] = cuda_blake2b_ROTR64(ctx->state[b] ^ ctx->state[c], 63);
}

__device__ __forceinline__ void cuda_blake2b_init_state(BLAKE2B_CTX *ctx) {
    memcpy(ctx->state, ctx->chain, BLAKE2B_CHAIN_LENGTH);
    for (int i = 0; i < 4; i++)
        ctx->state[BLAKE2B_CHAIN_SIZE + i] = BLAKE2B_IVS[i];

    ctx->state[12] = ctx->t0 ^ BLAKE2B_IVS[4];
    ctx->state[13] = ctx->t1 ^ BLAKE2B_IVS[5];
    ctx->state[14] = ctx->f0 ^ BLAKE2B_IVS[6];
    ctx->state[15] = BLAKE2B_IVS[7];
}

__device__ __forceinline__ void cuda_blake2b_compress(BLAKE2B_CTX *ctx, unsigned char *in, unsigned int inoffset) {
    cuda_blake2b_init_state(ctx);

    long  m[16] = {0};
    for (int j = 0; j < 16; j++)
        m[j] = cuda_blake2b_leuint64(in + inoffset + (j << 3));

    for (int round = 0; round < BLAKE2B_ROUNDS; round++)
    {
        cuda_blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][0]], m[BLAKE2B_SIGMAS[round][1]], 0, 4, 8, 12);
        cuda_blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][2]], m[BLAKE2B_SIGMAS[round][3]], 1, 5, 9, 13);
        cuda_blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][4]], m[BLAKE2B_SIGMAS[round][5]], 2, 6, 10, 14);
        cuda_blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][6]], m[BLAKE2B_SIGMAS[round][7]], 3, 7, 11, 15);
        cuda_blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][8]], m[BLAKE2B_SIGMAS[round][9]], 0, 5, 10, 15);
        cuda_blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][10]], m[BLAKE2B_SIGMAS[round][11]], 1, 6, 11, 12);
        cuda_blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][12]], m[BLAKE2B_SIGMAS[round][13]], 2, 7, 8, 13);
        cuda_blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][14]], m[BLAKE2B_SIGMAS[round][15]], 3, 4, 9, 14);
    }

    for (int offset = 0; offset < BLAKE2B_CHAIN_SIZE; offset++)
        ctx->chain[offset] ^= ctx->state[offset] ^ ctx->state[offset + 8];
}

__device__
void cuda_blake2b_init(BLAKE2B_CTX *ctx) {
    memset(ctx, 0, sizeof(BLAKE2B_CTX));
    const unsigned long key = 0xFEDCBA9876543210UL;
    const unsigned long keylen = sizeof(key);

    ctx->keylen = keylen;
    ctx->digestlen = 64 >> 3;
    ctx->pos = 0;
    ctx->t0 = 0;
    ctx->t1 = 0;
    ctx->f0 = 0;
    ctx->chain[0] = BLAKE2B_IVS[0] ^ (ctx->digestlen | (ctx->keylen << 8) | 0x1010000);
    ctx->chain[1] = BLAKE2B_IVS[1];
    ctx->chain[2] = BLAKE2B_IVS[2];
    ctx->chain[3] = BLAKE2B_IVS[3];
    ctx->chain[4] = BLAKE2B_IVS[4];
    ctx->chain[5] = BLAKE2B_IVS[5];
    ctx->chain[6] = BLAKE2B_IVS[6];
    ctx->chain[7] = BLAKE2B_IVS[7];

    memcpy(ctx->buff, &key, keylen);
    memcpy(ctx->key, &key, keylen);
    ctx->pos = BLAKE2B_BLOCK_LENGTH;
}

__device__
void cuda_blake2b_update(BLAKE2B_CTX *ctx, unsigned char* in, long inlen) {
    if (inlen == 0)
        return;

    unsigned int start = 0;
    long in_index = 0, block_index = 0;

    if (ctx->pos) {
        start = BLAKE2B_BLOCK_LENGTH - ctx->pos;
        if (start < inlen){
            memcpy(ctx->buff + ctx->pos, in, start);
            ctx->t0 += BLAKE2B_BLOCK_LENGTH;

            if (ctx->t0 == 0) ctx->t1++;

            cuda_blake2b_compress(ctx, ctx->buff, 0);
            ctx->pos = 0;
            memset(ctx->buff, 0, BLAKE2B_BLOCK_LENGTH);
        } else {
            memcpy(ctx->buff + ctx->pos, in, inlen);//read the whole *in
            ctx->pos += inlen;
            return;
        }
    }

    block_index =  inlen - BLAKE2B_BLOCK_LENGTH;
    for (in_index = start; in_index < block_index; in_index += BLAKE2B_BLOCK_LENGTH) {
        ctx->t0 += BLAKE2B_BLOCK_LENGTH;
        if (ctx->t0 == 0)
            ctx->t1++;

        cuda_blake2b_compress(ctx, in, in_index);
    }

    memcpy(ctx->buff, in + in_index, inlen - in_index);
    ctx->pos += inlen - in_index;
}

__device__
void cuda_blake2b_final(BLAKE2B_CTX *ctx, unsigned char* out) {
    ctx->f0 = 0xFFFFFFFFFFFFFFFFL;
    ctx->t0 += ctx->pos;
    if (ctx->pos > 0 && ctx->t0 == 0)
        ctx->t1++;

    cuda_blake2b_compress(ctx, ctx->buff, 0);
    memset(ctx->buff, 0, BLAKE2B_BLOCK_LENGTH);
    memset(ctx->state, 0, BLAKE2B_STATE_LENGTH);

    for (int i = 0; i < BLAKE2B_CHAIN_SIZE && (i*8 < ctx->digestlen); i++) {
        unsigned char * tmp = (unsigned char*)(&ctx->chain[i]);
        if (i*8 < ctx->digestlen - 8)
            memcpy(out + i*8, tmp, 8);
        else
            memcpy(out + i*8, tmp, ctx->digestlen - i*8);
    }
}

extern "C" __global__
void seq_blake2b(uint8_t *out, uint8_t *in, uint64_t blockSize, uint64_t n) {
    BLAKE2B_CTX ctx;
	sequential(cuda_blake2b_init, cuda_blake2b_update, cuda_blake2b_final);
}

extern "C" __global__
void merkle_hash_blake2b(uint8_t *out, uint64_t blockSize, uint64_t *startThread,
	uint64_t *workSize, uint8_t **workAddr, uint64_t l, uint64_t n) {

    BLAKE2B_CTX ctx;
    merkle_pre(cuda_blake2b_init, cuda_blake2b_update, cuda_blake2b_final, 64);
}

extern "C" __global__
void merkle_reduce_blake2b(uint8_t *out, uint8_t *in, size_t n) {
	extern __shared__ uint8_t shMem[];
    BLAKE2B_CTX ctx;
	merkle_step(cuda_blake2b_init, cuda_blake2b_update, cuda_blake2b_final, 64);
}

#endif
