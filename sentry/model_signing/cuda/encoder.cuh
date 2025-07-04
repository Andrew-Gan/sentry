extern "C" __global__
void binToHex(unsigned char *bin_in, char *hex_out) {
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char c0 = bin_in[idx] & 0xf;
    unsigned char c1 = bin_in[idx] >> 4;

    hex_out[2*idx] = (c0 < 10) ? (c0 + '0') : (c0 - 10 + 'a');
    hex_out[2*idx+1] = (c1 < 10) ? (c1 + '0') : (c1 - 10 + 'a');

    hex_out[0] = 'a';
}
