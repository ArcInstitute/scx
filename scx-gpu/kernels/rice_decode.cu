// GPU Rice decoder kernel — one thread per block of 256 values.
//
// CPU pre-scans the byte-aligned block boundaries and extracts each block's
// k parameter. Each GPU thread sequentially decodes its assigned block's
// Rice-coded values from the bitstream.
//
// Bitstream format (LSB-first):
//   Per value: unary quotient (q ones + 0 bit) + k-bit remainder (LSB-first)
//   Reconstruction: output = ((q << k) | r) + 1

extern "C" __global__ void rice_decode_kernel(
    const unsigned char* __restrict__ bitstream,      // full encoded bitstream
    const unsigned int*  __restrict__ block_offsets,   // byte offset where each block's data starts (after 1-byte header)
    const unsigned char* __restrict__ block_k,         // k parameter per block
    unsigned int*        __restrict__ output,          // decoded values (all >= 1)
    unsigned int n_blocks,
    unsigned int block_size,       // 256
    unsigned int last_block_len    // values in the last (potentially partial) block
) {
    unsigned int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) return;

    unsigned int n_vals = (block_id == n_blocks - 1) ? last_block_len : block_size;
    unsigned int k = block_k[block_id];
    unsigned int out_base = block_id * block_size;

    // Start reading from this block's data (after the 1-byte header)
    const unsigned char* bs = bitstream + block_offsets[block_id];
    unsigned int bit_pos = 0;

    for (unsigned int i = 0; i < n_vals; i++) {
        // Read unary quotient: count 1-bits until 0-bit (LSB-first within each byte)
        unsigned int q = 0;
        while (1) {
            unsigned int byte_idx = bit_pos >> 3;
            unsigned int bit_idx = bit_pos & 7;
            if ((bs[byte_idx] >> bit_idx) & 1) {
                q++;
                bit_pos++;
            } else {
                bit_pos++;  // skip the terminating 0-bit
                break;
            }
        }

        // Read k-bit remainder (LSB-first)
        unsigned int r = 0;
        if (k > 0) {
            // Optimized: read up to 32 bits at once using word access
            unsigned int byte_idx = bit_pos >> 3;
            unsigned int bit_idx = bit_pos & 7;

            // Read up to 5 bytes to get enough bits (bit_idx + k <= 40)
            unsigned long long word = 0;
            for (unsigned int b = 0; b < 5 && (byte_idx + b) < 0xFFFFFFFF; b++) {
                word |= ((unsigned long long)bs[byte_idx + b]) << (b * 8);
            }
            word >>= bit_idx;
            unsigned int mask = (1u << k) - 1;
            r = (unsigned int)(word & mask);
            bit_pos += k;
        }

        unsigned int shifted = (q << k) | r;
        output[out_base + i] = shifted + 1;
    }
}
