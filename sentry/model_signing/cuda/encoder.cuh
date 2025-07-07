#include <cuda/std/cstdint>

extern "C"
__global__ void binToHex(uint8_t *bin_in, int8_t *hex_out) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t c0 = bin_in[idx] & 0xf;
    uint8_t c1 = bin_in[idx] >> 4;

    hex_out[2*idx] = (c0 < 10) ? (c0 + '0') : (c0 - 10 + 'a');
    hex_out[2*idx+1] = (c1 < 10) ? (c1 + '0') : (c1 - 10 + 'a');
}
