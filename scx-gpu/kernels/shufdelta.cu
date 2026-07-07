// GPU decode transforms for ShufDeltaZstd (codec_id = 5) shards.
//
// The CPU host still runs zstd; these kernels take over the two byte-plane
// transforms that Phase-0 profiling showed dominate the CPU decode (~63-73%
// of it, running only ~0.4-0.7 GB/s scalar): the per-plane wrapping-u8 delta
// prefix scan (`undelta`) and the plane-major -> element-major transpose
// fused with the widen-to-i32/f32 convert (`unshuffle_convert`).
//
// Byte layout (mirrors scx-codec `byte_shuffle`/`byte_delta_planes`):
//   plane-major: buf = [ plane0[0..n], plane1[0..n], ... plane(width-1)[0..n] ]
//   byte j of element i lives at buf[(size_t)j * n + i].
// Encode is shuffle -> delta -> zstd; decode reverses to zstd -> undelta ->
// unshuffle. Integer values skip the delta step (shuffle-only on encode).
//
// Correctness-first design: `undelta_planes_kernel` uses one thread block per
// plane with a running carry (no cross-block scan), which is trivially correct
// and still ~50-100x the scalar CPU baseline. A single fused multi-block
// decoupled-lookback scan is a future optimization.

/// In-place per-plane wrapping-u8 inclusive prefix scan (undo byte-delta).
///
/// Launch with grid_dim.x == width (one block per plane), block_dim.x == 256.
/// Each block strides its plane in tiles of blockDim.x, scanning each tile in
/// shared memory (Hillis-Steele) and threading a running carry across tiles.
extern "C" __global__ void undelta_planes_kernel(
    unsigned char* buf,   // plane-major, mutated in place
    unsigned int n,       // elements per plane
    unsigned int width    // number of planes (== gridDim.x)
) {
    unsigned int p = blockIdx.x;
    if (p >= width) return;
    size_t base = (size_t)p * (size_t)n;

    __shared__ unsigned char s[256];
    unsigned char carry = 0;

    for (unsigned int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        unsigned int idx = tile_start + threadIdx.x;
        unsigned char v = (idx < n) ? buf[base + (size_t)idx] : (unsigned char)0;
        s[threadIdx.x] = v;
        __syncthreads();

        // Inclusive scan over the tile (padding lanes hold 0, contribute nothing).
        for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
            unsigned char add = (threadIdx.x >= offset) ? s[threadIdx.x - offset] : (unsigned char)0;
            __syncthreads();
            if (threadIdx.x >= offset) {
                s[threadIdx.x] = (unsigned char)(s[threadIdx.x] + add);
            }
            __syncthreads();
        }

        if (idx < n) {
            buf[base + (size_t)idx] = (unsigned char)(s[threadIdx.x] + carry);
        }
        __syncthreads();

        // Advance the carry by this tile's total (last valid lane's inclusive
        // sum + prior carry). Uniform across the block.
        unsigned int count = (n - tile_start) < blockDim.x ? (n - tile_start) : blockDim.x;
        carry = (unsigned char)(s[count - 1] + carry);
        __syncthreads();
    }
}

/// Unshuffle (plane-major -> element-major) fused with widen-to-i32/f32.
/// One thread per element; reassembles the little-endian value from `width`
/// planes and writes it as i32 (indices) or f32 (values).
extern "C" __global__ void unshuffle_convert_kernel(
    const unsigned char* __restrict__ src,  // plane-major (already undelta'd)
    void*                __restrict__ dst,   // int32* or float*
    unsigned int n,
    unsigned int width,
    unsigned int out_is_float
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int v = 0;
    for (unsigned int j = 0; j < width; ++j) {
        v |= ((unsigned int)src[(size_t)j * (size_t)n + (size_t)i]) << (8u * j);
    }
    if (out_is_float) {
        ((float*)dst)[i] = (float)v;
    } else {
        ((int*)dst)[i] = (int)v;
    }
}
