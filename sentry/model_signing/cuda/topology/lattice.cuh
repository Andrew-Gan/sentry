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

__device__
void add(const uint64_t b1, const uint64_t b2, uint64_t *out) {
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
    const uint64_t kMaskA = 0xffff0000ffff0000ULL;
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
void sub(const uint64_t b1, const uint64_t b2, uint64_t *out) {

    const uint64_t kMaskA = 0xffff0000ffff0000ULL;
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

#if defined(COALESCED) || defined(LAYERED) || defined(LAYERED_SORTED)
extern "C" __global__ 
void hashBlock(uint8_t *out, uint8_t *in, uint64_t blockSize, uint64_t nbytes) {
    BLAKE2XB_CTX ctx;
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t *myIn = in + idx * blockSize;
    if (myIn >= in + nbytes) return;
    uint64_t rem = (in + nbytes) - myIn;

    cuda_blake2xb_init(&ctx);
    cuda_blake2xb_update(&ctx, myIn, rem < blockSize ? rem : blockSize);
    cuda_blake2xb_final(&ctx, &out[idx * OUT_BYTES]);
}

#elif defined(INPLACE)
extern "C" __global__
void hashBlock(uint8_t *out, uint64_t blockSize, uint64_t *startThread,
    uint64_t *workSize, uint8_t **workAddr, uint64_t l, uint64_t nThread) {
    BLAKE2XB_CTX ctx;
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nThread) {
        cuda_blake2xb_init(&ctx);
        uint64_t workId = 0;
        while (workId < l && idx >= startThread[workId]) {
            workId++;
        }
        workId--;
        uint8_t *my_in = workAddr[workId] + blockSize * (idx - startThread[workId]);
        uint8_t *workEnd = workAddr[workId] + workSize[workId];
        cuda_blake2xb_update(&ctx, my_in, blockSize < workEnd - my_in ? blockSize : workEnd - my_in);
    }
    __syncthreads();
    if (idx < nThread)
        cuda_blake2xb_final(&ctx, &out[idx * OUT_BYTES]);
}

#endif

extern "C" __global__ 
void hash_dataset(uint8_t *out, uint8_t **in, uint64_t blockSize, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    BLAKE2XB_CTX ctx;
    cuda_blake2xb_init(&ctx);
    cuda_blake2xb_update(&ctx, (uint8_t*)&i, sizeof(i));
    cuda_blake2xb_update(&ctx, in[i], blockSize);
    cuda_blake2xb_final(&ctx, out + i * BLAKE2B_BYTES_MAX);
}

extern "C" __global__
void reduce(uint64_t *out, uint64_t *in, uint64_t n) {
    extern __shared__ uint64_t shMem[];
    uint64_t locIdx = threadIdx.x;
    int activeThreads = min((uint64_t)blockDim.x, n - (blockIdx.x * blockDim.x));
    uint64_t numDigest = 2 * (activeThreads / 8);

    if (locIdx < activeThreads) {
        uint64_t offset = (2 * blockDim.x * blockIdx.x) + locIdx;
        add(in[offset], in[offset+activeThreads], shMem+locIdx);
    }
    numDigest /= 2;
    __syncthreads();
    if (numDigest > 1) {
        // if odd number of digests at start, pad one zero digest
        if (numDigest % 2 == 1 && locIdx < 8)
            shMem[8*numDigest+locIdx] = 0;
        numDigest = (numDigest + 1) & ~0b1;
        __syncthreads();
    }
    while (numDigest > 8) {
        activeThreads = numDigest / 2 * 8;
        if (locIdx < activeThreads)
            add(shMem[locIdx], shMem[locIdx + activeThreads], shMem + locIdx);
        if (numDigest > 1) numDigest = (numDigest / 2 + 1) & ~0b1;
        __syncthreads();
    }
    while (numDigest > 1) {
        activeThreads = numDigest / 2 * 8;
        if (locIdx < activeThreads)
            add(shMem[locIdx], shMem[locIdx + activeThreads], shMem + locIdx);
        numDigest /= 2;
        if (numDigest > 1) numDigest = (numDigest + 1) & ~0b1;
    }
    if (locIdx < 8) {
        uint64_t blockOffsetU64 = 8 * blockIdx.x;
        *(out + blockOffsetU64 + locIdx) = shMem[locIdx];
    }
}

