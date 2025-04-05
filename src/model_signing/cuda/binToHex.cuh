extern "C" __global__
void binToHex(unsigned char *bin_in, char *hex_out) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned long index = i * blockDim.x + j;

    unsigned char c0 = bin_in[index] & 0xff;
    unsigned char c1 = bin_in[index] >> 4;

    hex_out[2*index] = (c0 < 10) ? (c0 + '0') : (c0 - 10 + 'A');
    hex_out[2*index+1] = (c1 < 10) ? (c1 + '0') : (c1 - 10 + 'A');
}
