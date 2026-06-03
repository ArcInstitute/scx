// Column-major GPU kernels for scatter, gather, mean-correction, and column sums.
//
// These replace CPU round-trip operations in the GPU PCA pipeline that were
// downloading the full global matrix (n_obs × k) to host per shard, modifying
// a small region, and re-uploading — causing massive PCIe transfer overhead.
//
// All kernels use 1D thread indexing with col-major addressing:
//   For a matrix M (rows × cols): M[r, c] = flat[c * rows + r]
//   Given flat idx: col = idx / rows, row = idx % rows

// Mean-correct a strided slice of a col-major matrix in place:
//   Y[global_row + r, c] -= mc[c]   for r in [0, shard_rows), c in [0, k)
//
// Y is laid out as (ld × k) col-major: Y[r, c] = flat[c * ld + r].
// The kernel touches only the sub-region (rows [global_row, global_row +
// shard_rows)) — other rows are not read or written.
//
// Used by the strided PCA matmat path to apply mean correction directly to
// a shard's slice of the global (n_obs × k) buffer without copying back
// through a contiguous shard temporary.
//
// total threads = shard_rows × k
extern "C" __global__ void mean_correct_colmajor_strided_kernel(
    float* __restrict__ Y,           // [ld × k], col-major; full matrix
    const float* __restrict__ mc,    // [k]
    int shard_rows,
    int k,
    int global_row,
    int ld
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = shard_rows * k;
    if (idx >= total) return;

    int col = idx / shard_rows;
    int row = idx % shard_rows;
    long long flat = (long long)col * (long long)ld + (long long)global_row + (long long)row;
    Y[flat] -= mc[col];
}

// Compute column sums of a col-major matrix.
//
// X: (m × k) col-major: X[r, c] = flat[c * m + r]
// out: [k] column sums
//
// Strategy: one block per column, each block reduces m elements.
// For PCA, k is small (50-60), m is large (up to 1M cells).
// Uses shared memory reduction within each block, then atomicAdd.
extern "C" __global__ void column_sum_kernel(
    const float* __restrict__ X,     // [m × k], col-major
    float* __restrict__ out,         // [k], output column sums
    int m,
    int k
) {
    // Grid: blockIdx.x = which chunk of rows, blockIdx.y = which column
    int col = blockIdx.y;
    if (col >= k) return;

    const float* col_ptr = X + col * m;

    // Each thread sums a strided range of rows
    float local_sum = 0.0f;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int r = row; r < m; r += stride) {
        local_sum += col_ptr[r];
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Block reduction via shared memory (one value per warp).
    // Uses dynamic shared memory (extern __shared__) so the host controls the
    // allocation size, preventing silent overflow if block size changes.
    // Caller must pass shared_mem_bytes >= (blockDim.x / warpSize) * sizeof(float).
    assert(blockDim.x <= 1024 && "column_sum_kernel: block size must be <= 1024 threads");
    extern __shared__ float warp_sums[];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp reduces the warp sums
    int n_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < (unsigned int)n_warps) {
        local_sum = warp_sums[threadIdx.x];
    } else {
        local_sum = 0.0f;
    }

    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(&out[col], local_sum);
        }
    }
}

// Select the top-k eigenvectors (in descending eigenvalue order) from an
// ascending-ordered (n × n) column-major eigenvector matrix, writing them
// into an (n × k) column-major output.
//
// src: (n × n) col-major eigvecs, eigenvalues ascending by column index.
// dst: (n × k) col-major, column j of dst ← column (n-1-j) of src.
//
// Element (row, col=j) in dst corresponds to (row, src_col = n-1-j) in src.
// total threads = n × k
extern "C" __global__ void select_top_eigvecs_desc_kernel(
    const float* __restrict__ src,   // [n × n], col-major
    float* __restrict__ dst,         // [n × k], col-major
    int n,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * k;
    if (idx >= total) return;

    int col = idx / n;           // output column 0..k
    int row = idx % n;
    int src_col = n - 1 - col;   // descending: largest first

    dst[col * n + row] = src[src_col * n + row];
}

// Copy the last `k` entries of a length-`n` vector into a length-`k` output
// in reversed order — i.e. out[j] = in[n - 1 - j] for j in [0, k).
//
// total threads = k
extern "C" __global__ void reverse_tail_vec_kernel(
    const float* __restrict__ src,   // [n]
    float* __restrict__ dst,         // [k]
    int n,
    int k
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    dst[j] = src[n - 1 - j];
}

// Broadcast-scale each column of a col-major matrix by a scalar.
//
// U[r, c] *= s[c]   for all (r, c)
//
// U: (m × k) col-major, in-place
// s: [k] per-column scale factors
//
// total threads = m × k
extern "C" __global__ void scale_columns_kernel(
    float* __restrict__ U,           // [m × k], col-major, in-place
    const float* __restrict__ s,     // [k]
    int m,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * k;
    if (idx >= total) return;

    int col = idx / m;
    U[idx] *= s[col];
}

// Per-column nonzero accumulation: Σ x and Σ x² in f64.
//
// indices: [nnz] column index per nonzero
// data:    [nnz] f32 value per nonzero
// col_sum, col_sum_sq: [n_vars] f64 accumulators (must be zero-initialized by caller)
//
// Thread-per-nonzero, using atomicAdd on f64 (requires compute 6.x+).
//
// total threads = nnz
extern "C" __global__ void col_sum_sq_nonzeros_kernel(
    const int* __restrict__ indices,
    const float* __restrict__ data,
    long long nnz,
    double* __restrict__ col_sum,
    double* __restrict__ col_sum_sq
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int c = indices[i];
    double v = (double)data[i];
    atomicAdd(&col_sum[c], v);
    atomicAdd(&col_sum_sq[c], v * v);
}

// Per-column clipped nonzero accumulation: Σ min(x, clip[c]) and Σ min(x, clip[c])²
//
// indices: [nnz] column index per nonzero
// data:    [nnz] f32 value per nonzero
// clip_val: [n_vars] per-column clip threshold (f64)
// batch_count_sum:    [n_vars] Σ clipped_v
// sq_batch_count_sum: [n_vars] Σ clipped_v²
//
// Thread-per-nonzero, using atomicAdd on f64.
//
// total threads = nnz
extern "C" __global__ void col_clip_sq_nonzeros_kernel(
    const int* __restrict__ indices,
    const float* __restrict__ data,
    long long nnz,
    const double* __restrict__ clip_val,
    double* __restrict__ batch_count_sum,
    double* __restrict__ sq_batch_count_sum
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int c = indices[i];
    double v = (double)data[i];
    double cv = clip_val[c];
    double vc = v > cv ? cv : v;
    atomicAdd(&batch_count_sum[c], vc);
    atomicAdd(&sq_batch_count_sum[c], vc * vc);
}

// Outer product subtraction: Z[v, j] -= mu[v] * sum_q[j]
//
// Z: (n_vars × k) col-major
// mu: [n_vars]
// sum_q: [k]
//
// total threads = n_vars × k
extern "C" __global__ void outer_sub_kernel(
    float* __restrict__ Z,           // [n_vars × k], col-major
    const float* __restrict__ mu,    // [n_vars]
    const float* __restrict__ sum_q, // [k]
    int n_vars,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_vars * k;
    if (idx >= total) return;

    int col = idx / n_vars;
    int row = idx % n_vars;

    Z[idx] -= mu[row] * sum_q[col];
}

// Per-(batch, column) nonzero accumulation: Σ x and Σ x² in f64 grouped by batch.
//
// indptr:        [n_rows+1]    CSR row pointers (i64)
// indices:       [nnz]         column index per nonzero
// data:          [nnz]         f32 value per nonzero
// row_to_batch:  [n_rows]      batch id per row (or -1 to skip the row)
// col_sum_per_batch:    [n_batches × n_vars] f64 accumulators (caller zero-inits)
// col_sum_sq_per_batch: [n_batches × n_vars] f64 accumulators (caller zero-inits)
//
// Layout for the per-batch buffers: row-major over (batch, gene), i.e.
//   col_sum_per_batch[b * n_vars + c]
//
// Thread-per-row. Each thread reads its row's batch from row_to_batch[],
// skips when -1, and scans that row's nonzero range atomic-adding into the
// (b, col) slot. f64 atomicAdd (requires compute 6.x+).
extern "C" __global__ void col_sum_sq_nonzeros_batched_kernel(
    const long long* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data,
    const int* __restrict__ row_to_batch,
    int n_rows,
    int n_vars,
    int n_batches,
    double* __restrict__ col_sum_per_batch,
    double* __restrict__ col_sum_sq_per_batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    int b = row_to_batch[row];
    if (b < 0 || b >= n_batches) return;

    long long row_base = (long long)b * (long long)n_vars;
    long long start = indptr[row];
    long long end = indptr[row + 1];
    for (long long i = start; i < end; ++i) {
        int c = indices[i];
        double v = (double)data[i];
        atomicAdd(&col_sum_per_batch[row_base + c], v);
        atomicAdd(&col_sum_sq_per_batch[row_base + c], v * v);
    }
}

// Per-(batch, column) clipped nonzero accumulation:
//   Σ min(x, clip[b, c]) and Σ min(x, clip[b, c])²
//
// indptr, indices, data, row_to_batch: same layout as the batched mean/var kernel.
// clip_val_per_batch:    [n_batches × n_vars] f64 clip threshold per (batch, gene)
// batch_sum_per_batch:    [n_batches × n_vars] Σ clipped_v
// sq_batch_sum_per_batch: [n_batches × n_vars] Σ clipped_v²
//
// Thread-per-row. Mirrors the unbatched col_clip_sq_nonzeros_kernel but groups
// the output by row's batch and indexes clip_val by that batch's row in the
// flat (n_batches × n_vars) buffer.
extern "C" __global__ void col_clip_sq_nonzeros_batched_kernel(
    const long long* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data,
    const int* __restrict__ row_to_batch,
    int n_rows,
    int n_vars,
    int n_batches,
    const double* __restrict__ clip_val_per_batch,
    double* __restrict__ batch_sum_per_batch,
    double* __restrict__ sq_batch_sum_per_batch
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    int b = row_to_batch[row];
    if (b < 0 || b >= n_batches) return;

    long long row_base = (long long)b * (long long)n_vars;
    long long start = indptr[row];
    long long end = indptr[row + 1];
    for (long long i = start; i < end; ++i) {
        int c = indices[i];
        double v = (double)data[i];
        double cv = clip_val_per_batch[row_base + c];
        double vc = v > cv ? cv : v;
        atomicAdd(&batch_sum_per_batch[row_base + c], vc);
        atomicAdd(&sq_batch_sum_per_batch[row_base + c], vc * vc);
    }
}

// ---------------------------------------------------------------------------
// CSC-reduce HVG column statistics (one block per column, no cross-column
// atomics). Column-major sidecar: each gene's nonzeros are contiguous in
// `data[col_indptr[c] .. col_indptr[c+1]]`, so a single thread block owns one
// column, reduces its nonzeros in shared memory, and writes one value per
// accumulator. Shards cover disjoint global column ranges and each column is
// owned by exactly one block, so the writes are plain stores (`=`), never
// atomics — eliminating the hot-gene `atomicAdd` contention of the CSR path.
// Single-batch only (rows are ignored); multi-batch HVG stays on CSR.
// ---------------------------------------------------------------------------

// Block-reduce two doubles (sum, sum_sq) across `blockDim.x` threads using a
// warp-shuffle pass followed by a shared-memory pass over the warp partials.
// Caps at 1024 threads (32 warps). Returns the totals in lane 0 of warp 0;
// other threads receive undefined values.
__device__ __forceinline__ void block_reduce_sum_sumsq(
    double& sum, double& sum_sq
) {
    // Warp-level reduction.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum    += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ double warp_sum[32];
    __shared__ double warp_sum_sq[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        warp_sum[warp_id] = sum;
        warp_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();
    int n_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        sum    = (threadIdx.x < (unsigned int)n_warps) ? warp_sum[threadIdx.x]    : 0.0;
        sum_sq = (threadIdx.x < (unsigned int)n_warps) ? warp_sum_sq[threadIdx.x] : 0.0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum    += __shfl_down_sync(0xffffffff, sum, offset);
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
}

// Per-column Σx and Σx² on a CSC shard. One block per local column.
//
// col_indptr: [n_cols_in_shard + 1] CSC offsets (i64)
// data:       [nnz] f32 values
// col_start:  global column id of local column 0 (shard offset)
// col_sum, col_sum_sq: [n_vars] f64 accumulators (written, not accumulated)
extern "C" __global__ void csc_col_mean_sq_reduce_kernel(
    const long long* __restrict__ col_indptr,
    const float* __restrict__ data,
    int n_cols_in_shard,
    int col_start,
    int n_vars,
    double* __restrict__ col_sum,
    double* __restrict__ col_sum_sq
) {
    int local_col = blockIdx.x;
    if (local_col >= n_cols_in_shard) return;
    int global_col = col_start + local_col;
    if (global_col < 0 || global_col >= n_vars) return;

    long long s = col_indptr[local_col];
    long long e = col_indptr[local_col + 1];
    double sum = 0.0, sum_sq = 0.0;
    for (long long i = s + threadIdx.x; i < e; i += blockDim.x) {
        double v = (double)data[i];
        sum += v;
        sum_sq += v * v;
    }
    block_reduce_sum_sumsq(sum, sum_sq);
    if (threadIdx.x == 0) {
        col_sum[global_col] = sum;
        col_sum_sq[global_col] = sum_sq;
    }
}

// Per-column clipped Σ min(x, clip[c]) and Σ min(x, clip[c])² on a CSC shard.
// One block per local column. `clip_val` is indexed by global column id.
extern "C" __global__ void csc_col_clip_sq_reduce_kernel(
    const long long* __restrict__ col_indptr,
    const float* __restrict__ data,
    int n_cols_in_shard,
    int col_start,
    int n_vars,
    const double* __restrict__ clip_val,
    double* __restrict__ clipped_sum,
    double* __restrict__ clipped_sum_sq
) {
    int local_col = blockIdx.x;
    if (local_col >= n_cols_in_shard) return;
    int global_col = col_start + local_col;
    if (global_col < 0 || global_col >= n_vars) return;

    double cv = clip_val[global_col];
    long long s = col_indptr[local_col];
    long long e = col_indptr[local_col + 1];
    double sum = 0.0, sum_sq = 0.0;
    for (long long i = s + threadIdx.x; i < e; i += blockDim.x) {
        double v = (double)data[i];
        double vc = v > cv ? cv : v;
        sum += vc;
        sum_sq += vc * vc;
    }
    block_reduce_sum_sumsq(sum, sum_sq);
    if (threadIdx.x == 0) {
        clipped_sum[global_col] = sum;
        clipped_sum_sq[global_col] = sum_sq;
    }
}
