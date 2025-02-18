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
void add(uint64_t bitsPerElement, uint8_t *b1, uint8_t *b2, uint8_t *out,
    uint64_t nBytes) {

    assert(b1_nBytes % kCacheLineSize == 0);
    assert(kCacheLineSize % sizeof(uint64_t) == 0);
    static constexpr uint64_t kValsPerCacheLine = kCacheLineSize / sizeof(uint64_t);
    assert(kValsPerCacheLine > 0);
    assert(bitsPerElement == 16 || bitsPerElement == 32);

    // When bitsPerElement is 16:
    // There are no padding bits, 4x 16-bit values fit exactly into a uint64_t:
    // uint64_t U = [ uint16_t W, uint16_t X, uint16_t Y, uint16_t Z ].
    // We break them up into A and B groups, with each group containing
    // alternating elements, such that A | B = the original number:
    // uint64_t A = [ uint16_t W,          0, uint16_t Y,          0 ]
    // uint64_t B = [          0, uint16_t X,          0, uint16_t Z ]
    // Then we add the A group and B group independently, and bitwise-OR
    // the results.
    // When bitsPerElement is 32:
    // There are no padding bits, 2x 32-bit values fit exactly into a uint64_t.
    // We independently add the high and low halves and then XOR them together.
    const uint64_t kMaskA =
        bitsPerElement == 16 ? 0xffff0000ffff0000ULL : 0xffffffff00000000ULL;
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
void sub(uint64_t bitsPerElement,
    uint8_t *b1, uint64_t b1_nBytes, uint8_t *b2, uint64_t b2_nBytes,
    uint8_t *out, uint64_t out_nBytes) {
    assert(b1_nBytes == b2_nBytes);
    assert(b1_nBytes == out_nBytes);
    assert(b1_nBytes % kCacheLineSize == 0);
    assert(kCacheLineSize % sizeof(uint64_t) == 0);
    static constexpr uint64_t kValsPerCacheLine = kCacheLineSize / sizeof(uint64_t);
    assert(kValsPerCacheLine > 0);
    assert(bitsPerElement == 16 || bitsPerElement == 32);

    // modified from add() by inverting second value before addition
    const uint64_t kMaskA =
        bitsPerElement == 16 ? 0xffff0000ffff0000ULL : 0xffffffff00000000ULL;
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

/**
 * Templated homomorphic hash, using LtHash (lattice-based crypto).
 * Template parameters: B = element size in bits, N = number of elements.
 *
 * Current constraints (checked at compile time with static asserts):
 * (1) B must be 16, 20 or 32.
 * (2) N must be > 999.
 * (3) when B is 16, N must be divisible by 32.
 * (4) when B is 32, N must be divisible by 16.
 */
template <std::size_t B, std::size_t N>
class LtHash {
private:
    // current checksum
    BLAKE2XB_CTX ctx;
    uint8_t *checksum = nullptr;
    uint64_t checksum_nBytes;

public:
    LtHash(uint8_t *initialChecksum, uint64_t initialChecksum_nBytes) {
        assert(N > 999);
        assert(B == 16 || B == 32);
        uint64_t size = getChecksumSizeBytes();

        if (!initialChecksum) {
            checksum = new uint8_t[size + sizeof(uint64_t)];
            ((uint64_t*)checksum)[size / sizeof(uint64_t)] = size;
            memset(checksum, 0, checksum_nBytes);
        }
        else
            setChecksum(initialChecksum, getChecksumSizeBytes());
    }

    virtual ~LtHash() {
        clearKey(); // securely erase the old key if there is one
    }

    /**
     * Sets the secret Blake2xb key. The key will be used to hash every element
     * added with addObject() / removed with removeObject(). This can be used
     * to compute a keyed LtHash value for a set of elements, if desired.
     *
     * The key must be between 16 and 64 bytes long (inclusive) and should be
     * a cryptographic key (e.g. a random value generated by a CPRNG).
     *
     * Note that if the LtHash value is transmitted from one user to another, the
     * two users will have to securely share the secret key before the receiver
     * can verify the integrity of the LtHash value they got from the sender.
     */
    void setKey(uint8_t *key, uint64_t key_nBytes) {
        if (key_nBytes < BLAKE2B_KEYBYTES_MIN ||
            key_nBytes > BLAKE2B_KEYBYTES_MAX) {
            throw std::runtime_error("invalid key size");
        }
        clearKey(); // securely erase the old key if there is one
        memcpy(ctx.blake2b_ctx.key, key, key_nBytes);
        ctx.blake2b_ctx.keylen = key_nBytes;
    }

    /**
     * Unsets the secret Blake2xb key and erases the key contents from memory.
     */
    void clearKey() {
        if (ctx.blake2b_ctx.keylen != 0) {
            memset(ctx.blake2b_ctx.key, 0, ctx.blake2b_ctx.keylen);
            ctx.blake2b_ctx.keylen = 0;
        }
    }

    static bool keysEqual(const LtHash<B, N>& h1, const LtHash<B, N>& h2)  {
        if ((h1.key && !h2.key) || (!h1.key && h2.key))
            return false;
        if (!h1)
            return true; // both LtHashes have empty keys
        if (h1.key_nBytes != h2.key_nBytes)
            return false;
        return memcmp(h1.key, h2.key, h1.key_nBytes) == 0;
    }

    LtHash<B, N>& operator+=(const LtHash<B, N>& rhs) {
        if (!keysEqual(*this, rhs)) {
            throw std::runtime_error("Cannot add 2 LtHashes with different keys");
        }
        add(B, checksum, checksum_nBytes, rhs.checksum, rhs.checksum_nBytes,
            checksum, checksum_nBytes);
        return *this;
    }

    LtHash<B, N>& operator+=(std::vector<const LtHash<B, N>&> rhs) {
        for(auto r : rhs) {
            if (!keysEqual(*this, r)) {
                throw std::runtime_error("Cannot add 2 LtHashes with different keys");
            }
            add(B, checksum, checksum_nBytes, r.checksum, r.checksum_nBytes,
                checksum, checksum_nBytes);
        }
        return *this;
    }

    LtHash<B, N>& operator-=(const LtHash<B, N>& rhs) {
        if (!keysEqual(*this, rhs)) {
            throw std::runtime_error("Cannot subtract 2 LtHashes with different keys");
        }
        sub(B, checksum, checksum_nBytes, rhs.checksum, rhs.checksum_nBytes,
            checksum, checksum_nBytes);
        return *this;
    }

    LtHash<B, N>& operator-=(std::vector<const LtHash<B, N>&> rhs) {
        for(auto r : rhs) {
            if (!keysEqual(*this, r)) {
                throw std::runtime_error("Cannot add 2 LtHashes with different keys");
            }
            sub(B, checksum, checksum_nBytes, r.checksum, r.checksum_nBytes,
                checksum, checksum_nBytes);
        }
        return *this;
    }

    /**
     * Equality comparison operator, implemented in a data-independent way to
     * guard against timing attacks. Always use this to check if two LtHash
     * values are equal instead of manually comparing checksum buffers.
     */
    bool operator==(const LtHash<B, N>& that) const {
        if (this == &that) { // same memory location means it's the same object
            return true;
        } else if (this->checksum_nBytes != that.checksum_nBytes) {
            return false;
        } else if (this->checksum_nBytes == 0) {
            // both objects must have been moved away from
            return true;
        } else {
            int cmp = memcmp(this->checksum, that.checksum, this->checksum_nBytes);
            return cmp == 0;
        }
    }

    /**
     * Inequality comparison operator.
     */
    bool operator!=(const LtHash<B, N>& that) const {
        return !(*this == that);
    }

    /**
     * Returns the number of elements that get packed into a single uint64_t.
     */
    static constexpr size_t getElementsPerUint64() {
        // how many elements fit into a 64-bit int? If padding is needed, assumes that
        // there is 1 padding bit between elements and any partial space is not used.
        // If padding is not needed, the computation is a trivial division.
        return (sizeof(uint64_t) * 8) / B;
    }

    /**
     * Returns the total length of the checksum (element_count * element_length)
     */
    static constexpr size_t getChecksumSizeBytes() {
        constexpr size_t elemsPerUint64 = getElementsPerUint64();
        assert(N % elemsPerUint64 == 0);
        return (N / elemsPerUint64) * sizeof(uint64_t);
    }

    __global__
    void hashBlockToDigest(uint8_t *in, uint64_t block, uint64_t n) {
        uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        uint64_t s = getChecksumSizeBytes();
        uint8_t *h = new uint8_t[s];
        cuda_blake2xb_init(&ctx, h, ctx.blake2b_ctx.key);
        cuda_blake2xb_update(ctx, in + i * block, block);
        cuda_blake2xb_final(&ctx, out, s);
        add(B, checksum, checksum_nBytes, h, s, checksum, checksum_nBytes);
        delete[] h;
    }
};

// This is the fastest and smallest specialization and should be
// preferred in most cases. It provides over 200 bits of security
// which should be good enough for most cases.
using LtHash16_1024 = LtHash<16, 1024>;

// These specializations are available to users who want a higher
// level of cryptographic security. They are slower and larger than
// the one above.
using LtHash32_1024 = LtHash<32, 1024>;

#endif
