// GPU FOR-BP decoder kernel — one thread per non-empty row.
//
// CPU pre-parses all block headers (block_nnz, n_rows_in_block, LEB128 varints,
// frame_min, frame_bits) and uploads per-row metadata. The GPU kernel handles
// only fixed-width bit extraction + prefix-sum reconstruction.
//
// Bitstream format (LSB-first):
//   Per row: frame_bits-wide packed deltas, delta[0]=0, delta[j]=idx[j]-idx[j-1]
//   Reconstruction: idx[0] = frame_min + delta[0], idx[j] = idx[j-1] + delta[j]

extern "C" __global__ void forbp_decode_kernel(
    const unsigned char* __restrict__ bitstream,          // full encoded data
    const unsigned int*  __restrict__ row_bit_offsets,     // bit offset of packed deltas for each row
    const unsigned int*  __restrict__ row_frame_min,       // frame_min per row
    const unsigned char* __restrict__ row_frame_bits,      // frame_bits per row
    const unsigned int*  __restrict__ row_nnz,             // nnz per row
    const unsigned int*  __restrict__ row_output_offset,   // start index in output for each row
    unsigned int*        __restrict__ output,              // flat output indices
    unsigned int n_rows                                    // number of non-empty rows
) {
    unsigned int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= n_rows) return;

    unsigned int nnz = row_nnz[row_id];
    if (nnz == 0) return;

    unsigned int frame_min = row_frame_min[row_id];
    unsigned int frame_bits = row_frame_bits[row_id];
    unsigned int out_base = row_output_offset[row_id];
    unsigned int bit_off = row_bit_offsets[row_id];

    // Prefix-sum to reconstruct absolute indices from deltas
    unsigned int prev = frame_min;

    for (unsigned int j = 0; j < nnz; j++) {
        unsigned int delta = 0;

        if (frame_bits > 0) {
            // Read frame_bits bits starting at bit_off + j * frame_bits (LSB-first)
            unsigned int cur_bit = bit_off + j * frame_bits;
            unsigned int byte_idx = cur_bit >> 3;
            unsigned int bit_idx = cur_bit & 7;

            // Optimized: read up to 5 bytes as a word to extract the delta at once
            unsigned long long word = 0;
            for (unsigned int b = 0; b < 5; b++) {
                word |= ((unsigned long long)bitstream[byte_idx + b]) << (b * 8);
            }
            word >>= bit_idx;
            unsigned int mask = (1u << frame_bits) - 1;
            delta = (unsigned int)(word & mask);
        }

        prev += delta;
        output[out_base + j] = prev;
    }
}
