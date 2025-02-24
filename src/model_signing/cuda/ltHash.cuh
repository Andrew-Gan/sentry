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

#ifndef __LTHASH_H__
#define __LTHASH_H__

#include "blake2xb.cuh"

__device__
void add(uint64_t bitsPerElem, const uint64_t b1, const uint64_t b2,
    uint64_t *out) {

    // assert(kCacheLineSize % sizeof(uint64_t) == 0);
    // static constexpr uint64_t kValsPerCacheLine = kCacheLineSize / sizeof(uint64_t);
    // assert(kValsPerCacheLine > 0);
    // assert(bitsPerElem == 16 || bitsPerElem == 32);

    // When bitsPerElem is 16:
    // There are no padding bits, 4x 16-bit values fit exactly into a uint64_t:
    // uint64_t U = [ uint16_t W, uint16_t X, uint16_t Y, uint16_t Z ].
    // We break them up into A and B groups, with each group containing
    // alternating elements, such that A | B = the original number:
    // uint64_t A = [ uint16_t W,          0, uint16_t Y,          0 ]
    // uint64_t B = [          0, uint16_t X,          0, uint16_t Z ]
    // Then we add the A group and B group independently, and bitwise-OR
    // the results.
    // When bitsPerElem is 32:
    // There are no padding bits, 2x 32-bit values fit exactly into a uint64_t.
    // We independently add the high and low halves and then XOR them together.
    const uint64_t kMaskA = bitsPerElem == 16 ? 0xffff0000ffff0000ULL : 0xffffffff00000000ULL;
    const uint64_t kMaskB = ~kMaskA;

    uint64_t v1a = b1 & kMaskA;
    uint64_t v1b = b1 & kMaskB;
    uint64_t v2a = b2 & kMaskA;
    uint64_t v2b = b2 & kMaskB;
    uint64_t v3a = (v1a + v2a) & kMaskA;
    uint64_t v3b = (v1b + v2b) & kMaskB;
    *out = v3a | v3b;
}

__device__
void sub(uint64_t bitsPerElem, const uint64_t b1, const uint64_t b2,
    uint64_t *out) {

    // assert(kCacheLineSize % sizeof(uint64_t) == 0);
    // static constexpr uint64_t kValsPerCacheLine = kCacheLineSize / sizeof(uint64_t);
    // assert(kValsPerCacheLine > 0);
    // assert(bitsPerElem == 16 || bitsPerElem == 32);

    const uint64_t kMaskA = bitsPerElem == 16 ? 0xffff0000ffff0000ULL : 0xffffffff00000000ULL;
    const uint64_t kMaskB = ~kMaskA;

    uint64_t v1a = b1 & kMaskA;
    uint64_t v1b = b1 & kMaskB;
    uint64_t v2a = b2 & kMaskA;
    uint64_t v2b = b2 & kMaskB;
    uint64_t v3a = (v1a + (kMaskB - v2a)) & kMaskA;
    uint64_t v3b = (v1b + (kMaskA - v2b)) & kMaskB;
    *out = v3a | v3b;
}

__device__
size_t getChecksumSizeBytes(uint64_t N, uint64_t B) {
    size_t elemsPerUint64 = (sizeof(uint64_t) * 8) / B;
    return (N / elemsPerUint64) * sizeof(uint64_t);
}

extern "C" __global__ 
void ltHash_pre(uint8_t *out, uint8_t *in, uint64_t block, uint64_t nThread) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nThread) return;

    BLAKE2XB_CTX ctx;
    uint64_t key = 0xfedcba9876543210UL;
    cuda_blake2xb_init(&ctx, BLAKE2B_BYTES_MAX, (uint8_t*)&key, sizeof(key));
    cuda_blake2xb_update(&ctx, in + i * block, block);
    cuda_blake2xb_final(&ctx, out);
}

__device__
void ltHash_add_warp(uint64_t B, uint64_t *sdata, uint64_t tid) {
    for (int numThread = 32; numThread >= 8; numThread /= 2) {
        if (tid < numThread)
            add(B, sdata[tid], sdata[tid + numThread], sdata + tid);
    }
}

extern "C" __global__
void ltHash_add(uint64_t B, uint64_t *dataIO, uint64_t nBytes) {
    __shared__ uint64_t sdata[512];
    uint64_t tid = threadIdx.x;
    uint64_t digestId = (2 * blockDim.x) * blockIdx.x + tid;
    uint64_t *lhs = dataIO + digestId;
    uint64_t *rhs = dataIO + digestId + blockDim.x;
    add(B, *lhs, *rhs, sdata + tid);

    for (uint64_t numThread = blockDim.x / 2; numThread > 32; numThread /= 2) {
        if (tid < numThread) {
            add(B, sdata[tid], sdata[tid + numThread], sdata + tid);
        }
        __syncthreads();
    }
    if (tid < 32) ltHash_add_warp(B, sdata, tid);
    if (tid == 0) memcpy(lhs, sdata, BLAKE2B_BYTES_MAX);
}

__device__
void ltHash_sub_warp(uint64_t B, uint64_t *sdata, uint64_t tid) {
    for (int numThread = 32; numThread >= 8; numThread /= 2) {
        if (tid < numThread)
            sub(B, sdata[tid], sdata[tid + numThread], sdata + tid);
    }
}

extern "C" __global__
void ltHash_sub(uint64_t B, uint64_t *dataIO, uint64_t nBytes) {
    __shared__ uint64_t sdata[1024];
    uint64_t tid = threadIdx.x;
    uint64_t digestId = (2 * blockDim.x) * blockIdx.x + tid;
    uint64_t *lhs = dataIO + digestId;
    uint64_t *rhs = dataIO + digestId + blockDim.x;
    sub(B, *lhs, *rhs, sdata + tid);

    for (uint64_t numThread = blockDim.x / 2; numThread > 32; numThread /= 2) {
        if (tid < numThread) {
            sub(B, sdata[tid], sdata[tid + numThread], sdata + tid);
        }
        __syncthreads();
    }
    if (tid < 32) ltHash_sub_warp(B, sdata, tid);
    if (tid == 0) memcpy(lhs, sdata, BLAKE2B_BYTES_MAX);
}

// using LtHash16_1024 = LtHash<16, 1024>;
// using LtHash32_1024 = LtHash<32, 1024>;

#endif
