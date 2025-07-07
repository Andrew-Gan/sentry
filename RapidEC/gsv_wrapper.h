#ifndef _GSV_WRAPPER_H_
#define _GSV_WRAPPER_H_

#include <cuda/std/cstdint>

#define GSV_TPI 4
#define GSV_BITS 256

// #define GSV_KNOWN_PKEY  // speed up verification if public key is known

typedef struct {
    uint32_t _limbs[(GSV_BITS + 31) / 32];
} gsv_mem_t;

typedef struct {
    gsv_mem_t e;         // digest
    gsv_mem_t priv_key;  // private key
    gsv_mem_t k;         // random number, no need to fill in
    gsv_mem_t r;         // sig->r, return value
    gsv_mem_t s;         // sig->s, return value
} gsv_sign_t;

typedef struct {
    gsv_mem_t r;  // sig->r
    gsv_mem_t s;  // sig->s
    gsv_mem_t e;  // digest
#ifndef GSV_KNOWN_PKEY
    gsv_mem_t key_x;  // public key
    gsv_mem_t key_y;  // public key
#endif
} gsv_verify_t;

#endif  // _GSV_WRAPPER_H_
