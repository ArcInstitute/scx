// Fused per-row normalize_total + log1p on GPU-resident CSR.
//
// Each thread handles one row: divides by row sum, multiplies by target_sum,
// applies log1p. Modifies data[] in-place.
//
// Matches the CPU implementation in scx-engine/src/fused_ops.rs:
//   fused_normalize_log1p(data, indptr, row_idx, target_sum)
//
// NOTE: The CPU path computes `(value_f64 * factor) as f32).ln_1p()`.
// The GPU path uses plain f32 arithmetic: `log1pf(value * scale)`.
// For NLP (normalize→log1p), both paths converge because:
//   - Row sums are accumulated in f32 on GPU vs f64 on CPU
//   - The scaling factor is computed in f32 on GPU vs f64 on CPU
// Small rounding differences (~1e-6 relative) are expected.
// Use rtol=1e-6 for validation, not bitwise identity.

extern "C" __global__ void normalize_log1p_kernel(
    const long long* __restrict__ indptr,   // [n_rows + 1], int64_t
    float* __restrict__ data,               // [nnz], modified in-place
    int n_rows,
    float target_sum
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long start = indptr[row];
    long long end = indptr[row + 1];

    // Compute row sum
    float sum = 0.0f;
    for (long long i = start; i < end; i++) {
        sum += data[i];
    }
    if (sum == 0.0f) return;

    // Normalize + log1p
    float scale = target_sum / sum;
    for (long long i = start; i < end; i++) {
        data[i] = log1pf(data[i] * scale);
    }
}

// Normalize-only variant: per-row normalize_total without log1p.
// data[i] = data[i] / row_sum * target_sum
extern "C" __global__ void normalize_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    float target_sum
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long start = indptr[row];
    long long end = indptr[row + 1];

    float sum = 0.0f;
    for (long long i = start; i < end; i++) {
        sum += data[i];
    }
    if (sum == 0.0f) return;

    float scale = target_sum / sum;
    for (long long i = start; i < end; i++) {
        data[i] *= scale;
    }
}

// Log1p-only variant: applies log1p to all non-zero values.
// data[i] = log1p(data[i])
extern "C" __global__ void log1p_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long start = indptr[row];
    long long end = indptr[row + 1];

    for (long long i = start; i < end; i++) {
        data[i] = log1pf(data[i]);
    }
}

// Row-scale variant: multiplies each row's non-zero values by an explicit
// per-row factor. data[i] *= factors[row_offset + row].
//
// Unlike normalize_kernel (which computes the factor from the row sum), the
// factor is supplied. `factors` is a global per-row vector indexed in the
// source's iteration row order; `row_offset` is this shard's first global
// row, so one device factor vector serves every shard in a streamed source.
extern "C" __global__ void row_scale_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    int row_offset,
    const float* __restrict__ factors
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long start = indptr[row];
    long long end = indptr[row + 1];

    float factor = factors[row_offset + row];
    for (long long i = start; i < end; i++) {
        data[i] *= factor;
    }
}

// ---------------------------------------------------------------------------
// Nonzero-parallel variants (Phase 2.1 / V2 §F).
//
// The thread-per-row kernels above serialize a whole row's nonzeros on one
// lane — fine for sparse rows, wasteful for dense single-cell rows (hundreds
// to thousands of nonzeros). These variants parallelize over nonzeros:
//   - log1p (no row reduction)   → one thread per nonzero.
//   - normalize / normalize+log1p (per-row sum reduction) → one warp per row
//     (warp-shuffle reduce) or one block per row (shared-memory reduce).
// The host (`apply_fused_ops_inner`) selects the granularity by mean row
// density; the thread-per-row kernels above remain the sparse fallback.
//
// f32 accumulation throughout, matching the thread-per-row kernels and their
// ~1e-5 rel-tolerance vs the f64-intermediate CPU reference.
// ---------------------------------------------------------------------------

// Log1p over all nonzeros, one thread per nonzero, grid-strided.
// data[i] = log1p(data[i]) for i in [0, nnz). indptr is irrelevant: log1p is
// elementwise and row-independent.
extern "C" __global__ void log1p_nnz_kernel(
    float* __restrict__ data,
    long long nnz
) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < nnz; i += stride) {
        data[i] = log1pf(data[i]);
    }
}

// Reduce one float across a warp (32 lanes); result returned in lane 0.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Reduce one float across `blockDim.x` threads (≤ 1024 = 32 warps) via a
// warp-shuffle pass then a shared-memory pass over the warp partials. The
// total is returned to EVERY thread (broadcast), so callers can use the scale
// without a second sync. Mirrors `block_reduce_sum_sumsq` in colmajor_ops.cu.
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_partial[32];
    __shared__ float block_total;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    int n_warps = (blockDim.x + warpSize - 1) / warpSize;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_partial[warp_id] = v;
    __syncthreads();

    if (warp_id == 0) {
        float w = (threadIdx.x < (unsigned int)n_warps) ? warp_partial[threadIdx.x] : 0.0f;
        w = warp_reduce_sum(w);
        if (lane == 0) block_total = w;
    }
    __syncthreads();
    return block_total;
}

// One warp per row: lanes stride the row's nonzeros to reduce the sum, then
// (after normalize) stride again to write. `do_log1p` selects the fused form.
__device__ __forceinline__ void normalize_warp_row(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    float target_sum,
    bool do_log1p
) {
    int lane = threadIdx.x % warpSize;
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    if (warp_global >= n_rows) return;

    long long start = indptr[warp_global];
    long long end = indptr[warp_global + 1];

    float partial = 0.0f;
    for (long long i = start + lane; i < end; i += warpSize) {
        partial += data[i];
    }
    float sum = warp_reduce_sum(partial);
    // Broadcast the lane-0 total to all lanes.
    sum = __shfl_sync(0xffffffff, sum, 0);
    if (sum == 0.0f) return;

    float scale = target_sum / sum;
    for (long long i = start + lane; i < end; i += warpSize) {
        float v = data[i] * scale;
        data[i] = do_log1p ? log1pf(v) : v;
    }
}

extern "C" __global__ void normalize_warp_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    float target_sum
) {
    normalize_warp_row(indptr, data, n_rows, target_sum, false);
}

extern "C" __global__ void normalize_log1p_warp_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    float target_sum
) {
    normalize_warp_row(indptr, data, n_rows, target_sum, true);
}

// One block per row: all threads stride the row's nonzeros to reduce the sum,
// then (after normalize) stride again to write. For very dense rows.
__device__ __forceinline__ void normalize_block_row(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    float target_sum,
    bool do_log1p
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    long long start = indptr[row];
    long long end = indptr[row + 1];

    float partial = 0.0f;
    for (long long i = start + threadIdx.x; i < end; i += blockDim.x) {
        partial += data[i];
    }
    float sum = block_reduce_sum(partial);
    if (sum == 0.0f) return;

    float scale = target_sum / sum;
    for (long long i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = data[i] * scale;
        data[i] = do_log1p ? log1pf(v) : v;
    }
}

extern "C" __global__ void normalize_block_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    float target_sum
) {
    normalize_block_row(indptr, data, n_rows, target_sum, false);
}

extern "C" __global__ void normalize_log1p_block_kernel(
    const long long* __restrict__ indptr,
    float* __restrict__ data,
    int n_rows,
    float target_sum
) {
    normalize_block_row(indptr, data, n_rows, target_sum, true);
}
