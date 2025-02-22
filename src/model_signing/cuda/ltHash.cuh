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

#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "blake2xb.cuh"

constexpr uint64_t kCacheLineSize = 128;

// add<<<b1_nBytes / kCacheLineSize, kValsPerCacheLine>>>
__device__
void add(uint64_t bitsPerElem, const uint8_t *b1, const uint8_t *b2,
    uint8_t *out, uint64_t nBytes) {

    assert(kCacheLineSize % sizeof(uint64_t) == 0);
    static constexpr uint64_t kValsPerCacheLine = kCacheLineSize / sizeof(uint64_t);
    assert(kValsPerCacheLine > 0);
    assert(bitsPerElem == 16 || bitsPerElem == 32);

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
    const uint64_t kMaskA =
        bitsPerElem == 16 ? 0xffff0000ffff0000ULL : 0xffffffff00000000ULL;
    const uint64_t kMaskB = ~kMaskA;

    // for (uint64_t pos = 0; pos < b1_nBytes; pos += kCacheLineSize) {
    uint64_t pos = blockIdx.x * blockDim.x;

    auto v1p = reinterpret_cast<const uint64_t*>(b1 + pos);
    auto v2p = reinterpret_cast<const uint64_t*>(b2 + pos);

    // for (uint64_t i = 0; i < kValsPerCacheLine; i++) {
    uint64_t i = threadIdx.x;

    uint64_t v1 = *(v1p + i);
    uint64_t v2 = *(v2p + i);
    uint64_t v1a = v1 & kMaskA;
    uint64_t v1b = v1 & kMaskB;
    uint64_t v2a = v2 & kMaskA;
    uint64_t v2b = v2 & kMaskB;
    uint64_t v3a = (v1a + v2a) & kMaskA;
    uint64_t v3b = (v1b + v2b) & kMaskB;
    *(out + pos + i) = v3a | v3b;
    //  }
    // }
}

// add<<<b1_nBytes / kCacheLineSize, kValsPerCacheLine>>>
__device__
void sub(uint64_t bitsPerElem, const uint8_t *b1, const uint8_t *b2,
    uint8_t *out, uint64_t nBytes) {

    assert(kCacheLineSize % sizeof(uint64_t) == 0);
    static constexpr uint64_t kValsPerCacheLine = kCacheLineSize / sizeof(uint64_t);
    assert(kValsPerCacheLine > 0);
    assert(bitsPerElem == 16 || bitsPerElem == 32);

    // modified from add() by inverting second value before addition
    const uint64_t kMaskA =
        bitsPerElem == 16 ? 0xffff0000ffff0000ULL : 0xffffffff00000000ULL;
    const uint64_t kMaskB = ~kMaskA;
    // for (uint64_t pos = 0; pos < b1_nBytes; pos += kCacheLineSize) {
    uint64_t pos = blockIdx.x * blockDim.x;

    auto v1p = reinterpret_cast<const uint64_t*>(b1 + pos);
    auto v2p = reinterpret_cast<const uint64_t*>(b2 + pos);

    //   for (uint64_t i = 0; i < kValsPerCacheLine; i++) {
    uint64_t i = threadIdx.x;

    uint64_t v1 = *(v1p + i);
    uint64_t v2 = *(v2p + i);
    uint64_t v1a = v1 & kMaskA;
    uint64_t v1b = v1 & kMaskB;
    uint64_t v2a = v2 & kMaskA;
    uint64_t v2b = v2 & kMaskB;
    uint64_t v3a = (v1a + (kMaskB - v2a)) & kMaskA;
    uint64_t v3b = (v1b + (kMaskA - v2b)) & kMaskB;
    *(out + pos + i) = v3a | v3b;
    //   }
    // }
}

__device__
size_t getChecksumSizeBytes(uint64_t N, uint64_t B) {
    size_t elemsPerUint64 = (sizeof(uint64_t) * 8) / B;
    assert(N % elemsPerUint64 == 0);
    return (N / elemsPerUint64) * sizeof(uint64_t);
}

extern "C" __global__ 
void ltHash_pre(uint8_t *out, uint8_t *in, uint64_t block, uint64_t nThread,
    uint64_t N, uint64_t B) {

    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nThread) return;
    uint64_t s = getChecksumSizeBytes(N, B);
    uint8_t *h = new uint8_t[s];

    BLAKE2XB_CTX ctx;
    uint64_t key = 0xfedcba9876543210UL;
    cuda_blake2xb_init(&ctx, 64, (uint8_t*)&key, sizeof(key));
    cuda_blake2xb_update(&ctx, in + i * block, block);
    cuda_blake2xb_final(&ctx, out, s);
    delete[] h;
}

extern "C" __global__
void ltHash_add(uint64_t B, uint8_t *lhs, const uint8_t *rhs, uint64_t nBytes) {
    add(B, lhs, rhs, lhs, nBytes);
}

extern "C" __global__
void ltHash_remove(uint64_t B, uint8_t *lhs, const uint8_t *rhs, uint64_t nBytes) {
    sub(B, lhs, rhs, lhs, nBytes);
}

// using LtHash16_1024 = LtHash<16, 1024>;
// using LtHash32_1024 = LtHash<32, 1024>;

#endif
