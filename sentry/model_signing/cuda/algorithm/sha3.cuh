/*
 * sha3.cu  Implementation of Keccak/SHA3 digest
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is released into the Public Domain.
 */

#include <cuda/std/cstdint>

#define KECCAK_ROUND 24
#define KECCAK_STATE_SIZE 25
#define KECCAK_Q_SIZE 192

__constant__ uint64_t CUDA_KECCAK_CONSTS[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

typedef struct {
    uint8_t sha3_flag;
    uint32_t digestbitlen;
    uint64_t rate_bits;
    uint64_t rate_bytes;
    uint64_t absorb_round;

    long state[KECCAK_STATE_SIZE];
    uint8_t q[KECCAK_Q_SIZE];

    uint64_t bits_in_queue;
} SHA3_CTX;

__device__ uint64_t cuda_sha3_leuint64(void *in)
{
    uint64_t a;
    memcpy(&a, in, 8);
    return a;
}

__device__ long cuda_sha3_MIN(long a, long b)
{
    if (a > b) return b;
    return a;
}

__device__ uint64_t cuda_sha3_UMIN(uint64_t a, uint64_t b)
{
    if (a > b) return b;
    return a;
}

__device__ void cuda_sha3_extract(SHA3_CTX *ctx)
{
    uint64_t len = ctx->rate_bits >> 6;
    long a;
    int s = sizeof(uint64_t);

    for (int i = 0;i < len;i++) {
        a = cuda_sha3_leuint64((long*)&ctx->state[i]);
        memcpy(ctx->q + (i * s), &a, s);
    }
}

__device__ __forceinline__ uint64_t cuda_sha3_ROTL64(uint64_t a, uint64_t  b)
{
    return (a << b) | (a >> (64 - b));
}

__device__ void cuda_sha3_permutations(SHA3_CTX * ctx)
{

    long* A = ctx->state;;

    long *a00 = A, *a01 = A + 1, *a02 = A + 2, *a03 = A + 3, *a04 = A + 4;
    long *a05 = A + 5, *a06 = A + 6, *a07 = A + 7, *a08 = A + 8, *a09 = A + 9;
    long *a10 = A + 10, *a11 = A + 11, *a12 = A + 12, *a13 = A + 13, *a14 = A + 14;
    long *a15 = A + 15, *a16 = A + 16, *a17 = A + 17, *a18 = A + 18, *a19 = A + 19;
    long *a20 = A + 20, *a21 = A + 21, *a22 = A + 22, *a23 = A + 23, *a24 = A + 24;

    for (int i = 0; i < KECCAK_ROUND; i++) {

        /* Theta */
        long c0 = *a00 ^ *a05 ^ *a10 ^ *a15 ^ *a20;
        long c1 = *a01 ^ *a06 ^ *a11 ^ *a16 ^ *a21;
        long c2 = *a02 ^ *a07 ^ *a12 ^ *a17 ^ *a22;
        long c3 = *a03 ^ *a08 ^ *a13 ^ *a18 ^ *a23;
        long c4 = *a04 ^ *a09 ^ *a14 ^ *a19 ^ *a24;

        long d1 = cuda_sha3_ROTL64(c1, 1) ^ c4;
        long d2 = cuda_sha3_ROTL64(c2, 1) ^ c0;
        long d3 = cuda_sha3_ROTL64(c3, 1) ^ c1;
        long d4 = cuda_sha3_ROTL64(c4, 1) ^ c2;
        long d0 = cuda_sha3_ROTL64(c0, 1) ^ c3;

        *a00 ^= d1;
        *a05 ^= d1;
        *a10 ^= d1;
        *a15 ^= d1;
        *a20 ^= d1;
        *a01 ^= d2;
        *a06 ^= d2;
        *a11 ^= d2;
        *a16 ^= d2;
        *a21 ^= d2;
        *a02 ^= d3;
        *a07 ^= d3;
        *a12 ^= d3;
        *a17 ^= d3;
        *a22 ^= d3;
        *a03 ^= d4;
        *a08 ^= d4;
        *a13 ^= d4;
        *a18 ^= d4;
        *a23 ^= d4;
        *a04 ^= d0;
        *a09 ^= d0;
        *a14 ^= d0;
        *a19 ^= d0;
        *a24 ^= d0;

        /* Rho pi */
        c1 = cuda_sha3_ROTL64(*a01, 1);
        *a01 = cuda_sha3_ROTL64(*a06, 44);
        *a06 = cuda_sha3_ROTL64(*a09, 20);
        *a09 = cuda_sha3_ROTL64(*a22, 61);
        *a22 = cuda_sha3_ROTL64(*a14, 39);
        *a14 = cuda_sha3_ROTL64(*a20, 18);
        *a20 = cuda_sha3_ROTL64(*a02, 62);
        *a02 = cuda_sha3_ROTL64(*a12, 43);
        *a12 = cuda_sha3_ROTL64(*a13, 25);
        *a13 = cuda_sha3_ROTL64(*a19, 8);
        *a19 = cuda_sha3_ROTL64(*a23, 56);
        *a23 = cuda_sha3_ROTL64(*a15, 41);
        *a15 = cuda_sha3_ROTL64(*a04, 27);
        *a04 = cuda_sha3_ROTL64(*a24, 14);
        *a24 = cuda_sha3_ROTL64(*a21, 2);
        *a21 = cuda_sha3_ROTL64(*a08, 55);
        *a08 = cuda_sha3_ROTL64(*a16, 45);
        *a16 = cuda_sha3_ROTL64(*a05, 36);
        *a05 = cuda_sha3_ROTL64(*a03, 28);
        *a03 = cuda_sha3_ROTL64(*a18, 21);
        *a18 = cuda_sha3_ROTL64(*a17, 15);
        *a17 = cuda_sha3_ROTL64(*a11, 10);
        *a11 = cuda_sha3_ROTL64(*a07, 6);
        *a07 = cuda_sha3_ROTL64(*a10, 3);
        *a10 = c1;

        /* Chi */
        c0 = *a00 ^ (~*a01 & *a02);
        c1 = *a01 ^ (~*a02 & *a03);
        *a02 ^= ~*a03 & *a04;
        *a03 ^= ~*a04 & *a00;
        *a04 ^= ~*a00 & *a01;
        *a00 = c0;
        *a01 = c1;

        c0 = *a05 ^ (~*a06 & *a07);
        c1 = *a06 ^ (~*a07 & *a08);
        *a07 ^= ~*a08 & *a09;
        *a08 ^= ~*a09 & *a05;
        *a09 ^= ~*a05 & *a06;
        *a05 = c0;
        *a06 = c1;

        c0 = *a10 ^ (~*a11 & *a12);
        c1 = *a11 ^ (~*a12 & *a13);
        *a12 ^= ~*a13 & *a14;
        *a13 ^= ~*a14 & *a10;
        *a14 ^= ~*a10 & *a11;
        *a10 = c0;
        *a11 = c1;

        c0 = *a15 ^ (~*a16 & *a17);
        c1 = *a16 ^ (~*a17 & *a18);
        *a17 ^= ~*a18 & *a19;
        *a18 ^= ~*a19 & *a15;
        *a19 ^= ~*a15 & *a16;
        *a15 = c0;
        *a16 = c1;

        c0 = *a20 ^ (~*a21 & *a22);
        c1 = *a21 ^ (~*a22 & *a23);
        *a22 ^= ~*a23 & *a24;
        *a23 ^= ~*a24 & *a20;
        *a24 ^= ~*a20 & *a21;
        *a20 = c0;
        *a21 = c1;

        /* Iota */
        *a00 ^= CUDA_KECCAK_CONSTS[i];
    }
}


__device__ void cuda_sha3_absorb(SHA3_CTX *ctx, uint8_t* in)
{

    uint64_t offset = 0;
    for (uint64_t i = 0; i < ctx->absorb_round; ++i) {
        ctx->state[i] ^= cuda_sha3_leuint64(in + offset);
        offset += 8;
    }

    cuda_sha3_permutations(ctx);
}

__device__ void cuda_sha3_pad(SHA3_CTX *ctx)
{
    ctx->q[ctx->bits_in_queue >> 3] |= (1L << (ctx->bits_in_queue & 7));

    if (++(ctx->bits_in_queue) == ctx->rate_bits) {
        cuda_sha3_absorb(ctx, ctx->q);
        ctx->bits_in_queue = 0;
    }

    uint64_t full = ctx->bits_in_queue >> 6;
    uint64_t partial = ctx->bits_in_queue & 63;

    uint64_t offset = 0;
    for (int i = 0; i < full; ++i) {
        ctx->state[i] ^= cuda_sha3_leuint64(ctx->q + offset);
        offset += 8;
    }

    if (partial > 0) {
        uint64_t mask = (1L << partial) - 1;
        ctx->state[full] ^= cuda_sha3_leuint64(ctx->q + offset) & mask;
    }

    ctx->state[(ctx->rate_bits - 1) >> 6] ^= 9223372036854775808ULL;/* 1 << 63 */

    cuda_sha3_permutations(ctx);
    cuda_sha3_extract(ctx);

    ctx->bits_in_queue = ctx->rate_bits;
}

/*
 * Digestbitlen must be 128 224 256 288 384 512
 */
__device__ void init(SHA3_CTX *ctx)
{
    memset(ctx, 0, sizeof(SHA3_CTX));
    ctx->sha3_flag = 0;
    ctx->digestbitlen = (64 << 3);
    ctx->rate_bits = 1600 - ((ctx->digestbitlen) << 1);
    ctx->rate_bytes = ctx->rate_bits >> 3;
    ctx->absorb_round = ctx->rate_bits >> 6;
    ctx->bits_in_queue = 0;
}

__device__ void update(SHA3_CTX *ctx, uint8_t *data, uint64_t len)
{
    long bytes = ctx->bits_in_queue >> 3;
    long count = 0;
    while (count < len) {
        if (bytes == 0 && count <= ((long)(len - ctx->rate_bytes))) {
            do {
                cuda_sha3_absorb(ctx, data + count);
                count += ctx->rate_bytes;
            } while (count <= ((long)(len - ctx->rate_bytes)));
        } else {
            long partial = cuda_sha3_MIN(ctx->rate_bytes - bytes, len - count);
            memcpy(ctx->q + bytes, data + count, partial);

            bytes += partial;
            count += partial;

            if (bytes == ctx->rate_bytes) {
                cuda_sha3_absorb(ctx, ctx->q);
                bytes = 0;
            }
        }
    }
    ctx->bits_in_queue = bytes << 3;
}

__device__ void final(SHA3_CTX *ctx, uint8_t *out)
{
    if (ctx->sha3_flag) {
        int mask = (1 << 2) - 1;
        ctx->q[ctx->bits_in_queue >> 3] = (uint8_t)(0x02 & mask);
        ctx->bits_in_queue += 2;
    }

    cuda_sha3_pad(ctx);
    uint64_t i = 0;

    while (i < ctx->digestbitlen) {
        if (ctx->bits_in_queue == 0) {
            cuda_sha3_permutations(ctx);
            cuda_sha3_extract(ctx);
            ctx->bits_in_queue = ctx->rate_bits;
        }

        uint64_t partial_block = cuda_sha3_UMIN(ctx->bits_in_queue, ctx->digestbitlen - i);
        memcpy(out + (i >> 3), ctx->q + (ctx->rate_bytes - (ctx->bits_in_queue >> 3)), partial_block >> 3);
        ctx->bits_in_queue -= partial_block;
        i += partial_block;
    }
}

typedef SHA3_CTX CTX;