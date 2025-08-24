/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda/std/cstdint>
#include "blake2b.cuh"

#define kMinOutputLength 1
#define kMaxOutputLength 0xfffffffeULL
#define kUnknownOutputLength 0

#define BLAKE2B_KEYBYTES_MIN 16U
#define BLAKE2B_KEYBYTES_MAX 64U

#define BLAKE2B_BYTES_MIN 16U
#define BLAKE2B_BYTES_MAX 64U

#define BLAKE2B_SALTBYTES 16U
#define BLAKE2B_PERSONALBYTES 16U

#define kUnknownOutputLengthMagic 0xffffffffULL
#define BLAKE2XB_STATE_SIZE 8

typedef struct {
    uint8_t fanout;
    uint8_t depth;
    uint32_t leafLength;
    uint32_t nodeOffset;
    uint32_t xofLength;
    uint8_t nodeDepth;
    uint8_t innerLength;
    uint8_t reserved[14];
    uint8_t salt[16];
    uint8_t personal[16];
    BLAKE2B_CTX blake2b_ctx;
} BLAKE2XB_CTX;

__device__
void initStateFromParams(BLAKE2XB_CTX *ctx, uint8_t *key, uint64_t key_nBytes) {
    auto p = (uint64_t*)ctx;
    for (int i = 0; i < 8; ++i) {
        ctx->blake2b_ctx.state[i] = BLAKE2B_IVS[i] ^ p[i];
    }
    uint64_t s = (BLAKE2B_STATE_SIZE - BLAKE2XB_STATE_SIZE) * sizeof(*ctx->blake2b_ctx.state);
    memset(ctx->blake2b_ctx.state + BLAKE2XB_STATE_SIZE, 0, s);
    if (key) {
        // assert (key_nBytes < BLAKE2B_KEYBYTES_MIN ||
        //     key_nBytes > BLAKE2B_KEYBYTES_MAX);
        uint8_t block[128];
        memcpy(block, key, key_nBytes);
        memset(block + key_nBytes, 0, 128 - key_nBytes);
        update(&ctx->blake2b_ctx, block, 128);
        memset(block, 0, 128); // erase key from stack
    }
}

__device__
void cuda_blake2xb_init(BLAKE2XB_CTX *ctx, uint64_t outputLength,
    uint8_t *key = nullptr, uint64_t key_nBytes = 0,
    uint8_t *salt = nullptr, uint8_t *personalization = nullptr) {

    memset(ctx, 0, sizeof(ctx));
    memcpy(ctx->blake2b_ctx.key, key, key_nBytes);
    ctx->blake2b_ctx.digestlen = BLAKE2B_BYTES_MAX;
    ctx->blake2b_ctx.keylen = (uint8_t)key_nBytes;
    ctx->fanout = 1;
    ctx->depth = 1;
    ctx->xofLength = (uint32_t)outputLength;
    if (salt)
        memcpy(ctx->salt, salt, sizeof(ctx->salt));
    if (personalization)
        memcpy(ctx->personal, personalization, sizeof(ctx->personal));
    initStateFromParams(ctx, key, key_nBytes);
}

__device__
void cuda_blake2xb_update(BLAKE2XB_CTX *ctx, uint8_t *data, uint64_t data_nBytes) {
    update(&ctx->blake2b_ctx, data, data_nBytes);
}

__device__
void cuda_blake2xb_final(BLAKE2XB_CTX *ctx, uint8_t *out) {
    // TODO: proof check
    uint8_t h0[BLAKE2B_BYTES_MAX];
    final(&ctx->blake2b_ctx, h0);

    ctx->blake2b_ctx.keylen = 0;
    ctx->fanout = 0;
    ctx->depth = 0;
    ctx->leafLength = (uint32_t)BLAKE2B_BYTES_MAX;
    ctx->innerLength = BLAKE2B_BYTES_MAX;
    uint64_t pos = 0;
    uint64_t remaining = 64;
    while (remaining > 0) {
        ctx->nodeOffset = (uint32_t)pos / BLAKE2B_BYTES_MAX;
        uint64_t len = BLAKE2B_BYTES_MAX < remaining ? BLAKE2B_BYTES_MAX : remaining;
        ctx->blake2b_ctx.digestlen = (uint8_t)len;
        initStateFromParams(ctx, nullptr, 0);
        update(&ctx->blake2b_ctx, h0, BLAKE2B_BYTES_MAX);
        final(&ctx->blake2b_ctx, out + pos);
        pos += len;
        remaining -= len;
    }
}
