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

#ifndef __BLAKE2XB_H__
#define __BLAKE2XB_H__

#include <cstdint>
#include <cstring>
#include <stdexcept>
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

bool outputLengthKnown;

__device__
void initStateFromParams(BLAKE2XB_CTX *ctx, uint8_t *key, uint64_t key_nBytes) {
    auto p = reinterpret_cast<const uint64_t*>(ctx);
    for (int i = 0; i < 8; ++i) {
        ctx->blake2b_ctx.state[i] = BLAKE2B_IVS[i] ^ p[i];
    }
    uint64_t s = (BLAKE2B_STATE_SIZE - BLAKE2XB_STATE_SIZE) * sizeof(*ctx->blake2b_ctx.state);
    memset(ctx->blake2b_ctx.state + BLAKE2XB_STATE_SIZE, 0, s);
    if (key) {
        if (key_nBytes < BLAKE2B_KEYBYTES_MIN ||
            key_nBytes > BLAKE2B_KEYBYTES_MAX) {
            throw std::runtime_error("invalid key size");
        }
        uint8_t block[128];
        memcpy(block, key, key_nBytes);
        memset(block + key_nBytes, 0, 128 - key_nBytes);
        cuda_blake2b_update(&ctx->blake2b_ctx, block, 128);
        memset(block, 0, 128); // erase key from stack
    }
}

__device__
void cuda_blake2xb_init(BLAKE2XB_CTX *ctx, uint64_t outputLength,
    uint8_t *key = nullptr, uint64_t key_nBytes = 0,
    uint8_t *salt = nullptr, uint64_t salt_nBytes = 0,
    uint8_t *personalization = nullptr, uint64_t personalization_nBytes = 0) {

    if (outputLength == kUnknownOutputLength) {
        outputLengthKnown = false;
        outputLength = kUnknownOutputLengthMagic;
    } else if (outputLength > kMaxOutputLength) {
        throw std::runtime_error("Output length too large");
    } else {
        outputLengthKnown = true;
    }
    memset(&ctx, 0, sizeof(ctx));
    ctx->blake2b_ctx.digestlen = BLAKE2B_BYTES_MAX;
    ctx->blake2b_ctx.keylen = static_cast<uint8_t>(key_nBytes);
    ctx->fanout = 1;
    ctx->depth = 1;
    ctx->xofLength = static_cast<uint32_t>(outputLength);
    if (salt) {
        if (salt_nBytes != BLAKE2B_SALTBYTES) {
        throw std::runtime_error("Invalid salt length, must be 16 bytes");
        }
        std::memcpy(ctx->salt, salt, sizeof(ctx->salt));
    }
    if (personalization) {
        if (personalization_nBytes != BLAKE2B_PERSONALBYTES) {
        throw std::runtime_error(
            "Invalid personalization length, must be 16 bytes");
        }
        std::memcpy(ctx->personal, personalization, sizeof(ctx->personal));
    }
    initStateFromParams(ctx, key, key_nBytes);
}

__device__
void cuda_blake2xb_update(BLAKE2XB_CTX *ctx, uint8_t *data, uint64_t data_nBytes) {
    cuda_blake2b_update(&ctx->blake2b_ctx, data, data_nBytes);
}

__device__
void cuda_blake2xb_final(BLAKE2XB_CTX *ctx, uint8_t *out, uint64_t out_nBytes) {
    if (outputLengthKnown) {
        auto outLength = static_cast<uint32_t>(out_nBytes);
        if (outLength != ctx->xofLength) {
            throw std::runtime_error("out_nBytes must equal output length");
        }
    }

    uint8_t h0[BLAKE2B_BYTES_MAX];
    cuda_blake2b_final(&ctx->blake2b_ctx, h0);

    ctx->blake2b_ctx.keylen = 0;
    ctx->fanout = 0;
    ctx->depth = 0;
    ctx->leafLength = static_cast<uint32_t>(BLAKE2B_BYTES_MAX);
    ctx->innerLength = BLAKE2B_BYTES_MAX;
    uint64_t pos = 0;
    uint64_t remaining = out_nBytes;
    while (remaining > 0) {
        ctx->nodeOffset = static_cast<uint32_t>(pos / BLAKE2B_BYTES_MAX);
        uint64_t len = std::min(
            static_cast<uint64_t>(BLAKE2B_BYTES_MAX), remaining);
        ctx->blake2b_ctx.digestlen = static_cast<uint8_t>(len);
        initStateFromParams(ctx, nullptr, 0);
        cuda_blake2b_update(&ctx->blake2b_ctx, h0, BLAKE2B_BYTES_MAX);
        cuda_blake2b_update(&ctx->blake2b_ctx, out + pos, len);
        pos += len;
        remaining -= len;
    }
}

#endif
