// GPU FOR-BP BitPacker4x decoder kernel — one thread per row (index_packing == 2).
//
// Complements forbp_decode_kernel (the scalar / index_packing == 1 path). Handles
// rows whose deltas the encoder packed with the `bitpacking` crate's BitPacker4x
// SIMD layout for full 128-value chunks (nnz >= SIMD_THRESHOLD = 128), with the
// trailing remainder (nnz % 128 values) in the scalar LSB-first layout.
//
// BitPacker4x chunk layout (chunk = frame_bits SIMD-words of 4x u32 lanes,
// chunk_bytes = frame_bits * 16, chunks concatenated with no padding):
//   value at chunk-position p (0..128): lane l = p & 3, register j = p >> 2 (0..32).
//   lane l's packed stream is the strided little-endian u32 view
//       u32 @ (chunk_base + w*16 + l*4),  w = 0..frame_bits
//   and the value occupies bits [j*frame_bits, (j+1)*frame_bits) of that lane
//   stream (spans at most two consecutive lane words → <= 2 unaligned u32 reads).
//
// Deltas: delta[0] = 0, delta[j] = idx[j] - idx[j-1]; idx = frame_min + prefix-sum.
// Produces output **bit-identical** to scx_codec::forbp::forbp_decode.
//
// Safety: `bitstream_len` bounds all byte reads to prevent out-of-bounds access.

#define BP4X_BLOCK_LEN 128u

// Assemble a little-endian u32 from up to 4 bytes, clamping each read to
// `bitstream_len` (the encoder's chunk offsets are byte-aligned but not
// 4-byte-aligned, so this must be an unaligned byte-wise read — never a u32* cast).
__device__ __forceinline__ unsigned int bp4x_read_u32le(
    const unsigned char* __restrict__ bitstream,
    unsigned int byte_idx,
    unsigned int bitstream_len
) {
    unsigned int v = 0;
#pragma unroll
    for (unsigned int b = 0; b < 4u; b++) {
        if (byte_idx + b < bitstream_len) {
            v |= ((unsigned int)bitstream[byte_idx + b]) << (b * 8);
        }
    }
    return v;
}

extern "C" __global__ void forbp_decode_bp4x_kernel(
    const unsigned char* __restrict__ bitstream,          // full encoded data
    const unsigned int*  __restrict__ row_bit_offsets,     // bit offset of packed deltas for each row
    const unsigned int*  __restrict__ row_frame_min,       // frame_min per row
    const unsigned char* __restrict__ row_frame_bits,      // frame_bits per row
    const unsigned int*  __restrict__ row_nnz,             // nnz per row
    const unsigned int*  __restrict__ row_output_offset,   // start index in output for each row
    unsigned int*        __restrict__ output,              // flat output indices
    unsigned int n_rows,                                   // number of rows in this (bp4x) group
    unsigned int bitstream_len                             // total length of bitstream in bytes
) {
    unsigned int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= n_rows) return;

    unsigned int nnz = row_nnz[row_id];
    if (nnz == 0) return;

    unsigned int frame_min = row_frame_min[row_id];
    unsigned int frame_bits = row_frame_bits[row_id];
    unsigned int out_base = row_output_offset[row_id];
    unsigned int base_byte = row_bit_offsets[row_id] >> 3;  // byte-aligned for packing==2

    // Guard against UB: `1u << 32` is undefined. frame_bits can be 32 for
    // u32-index shards (n_vars > 65535) whose deltas need 32 bits.
    unsigned int mask =
        (frame_bits >= 32) ? 0xFFFFFFFFu : ((1u << frame_bits) - 1u);

    unsigned int full_chunks = nnz / BP4X_BLOCK_LEN;
    unsigned int chunk_bytes = frame_bits * 16u;  // = frame_bits * 128 / 8
    unsigned int simd_count = full_chunks * BP4X_BLOCK_LEN;

    unsigned int prev = frame_min;

    for (unsigned int j = 0; j < nnz; j++) {
        unsigned int delta = 0;

        if (frame_bits > 0) {
            if (j < simd_count) {
                // BitPacker4x full chunk: transposed SIMD lane stream.
                unsigned int c = j / BP4X_BLOCK_LEN;
                unsigned int p = j - c * BP4X_BLOCK_LEN;   // 0..128
                unsigned int lane = p & 3u;
                unsigned int reg = p >> 2;                 // 0..32
                unsigned int chunk_base = base_byte + c * chunk_bytes;
                unsigned int bitpos = reg * frame_bits;
                unsigned int w0 = bitpos >> 5;
                unsigned int bit_in = bitpos & 31u;

                unsigned int lo = bp4x_read_u32le(
                    bitstream, chunk_base + w0 * 16u + lane * 4u, bitstream_len);
                unsigned int val = lo >> bit_in;
                if (bit_in + frame_bits > 32u) {
                    // Value straddles two consecutive lane words.
                    unsigned int hi = bp4x_read_u32le(
                        bitstream, chunk_base + (w0 + 1u) * 16u + lane * 4u, bitstream_len);
                    val |= hi << (32u - bit_in);
                }
                delta = val & mask;
            } else {
                // Remainder: contiguous LSB-first scalar stream (same extraction
                // as forbp_decode_kernel), based after the full chunks.
                unsigned int r = j - simd_count;
                unsigned int rem_base_byte = base_byte + full_chunks * chunk_bytes;
                unsigned int cur_bit = rem_base_byte * 8u + r * frame_bits;
                unsigned int byte_idx = cur_bit >> 3;
                unsigned int bit_idx = cur_bit & 7u;
                unsigned long long word = 0;
                for (unsigned int b = 0; b < 5u && (byte_idx + b) < bitstream_len; b++) {
                    word |= ((unsigned long long)bitstream[byte_idx + b]) << (b * 8);
                }
                word >>= bit_idx;
                delta = (unsigned int)(word & mask);
            }
        }

        prev += delta;
        output[out_base + j] = prev;
    }
}
