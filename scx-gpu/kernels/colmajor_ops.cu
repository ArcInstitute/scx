// Column-major GPU kernels for scatter, gather, mean-correction, and column sums.
//
// These replace CPU round-trip operations in the GPU PCA pipeline that were
// downloading the full global matrix (n_obs × k) to host per shard, modifying
// a small region, and re-uploading — causing massive PCIe transfer overhead.
//
// All kernels use 1D thread indexing with col-major addressing:
//   For a matrix M (rows × cols): M[r, c] = flat[c * rows + r]
//   Given flat idx: col = idx / rows, row = idx % rows

// Scatter a shard's col-major result into the global matrix.
//
// src: (shard_rows × k) col-major — the shard SpMM output
// dst: (n_obs × k) col-major — the global accumulator
//
// For each element (row, col) in src:
//   dst[(global_row + row), col] = src[row, col]
//   i.e. dst[col * n_obs + global_row + row] = src[col * shard_rows + row]
//
// total threads = shard_rows × k
extern "C" __global__ void scatter_colmajor_kernel(
    const float* __restrict__ src,   // [shard_rows × k], col-major
    float* __restrict__ dst,         // [n_obs × k], col-major
    int shard_rows,
    int n_obs,
    int k,
    int global_row
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = shard_rows * k;
    if (idx >= total) return;

    int col = idx / shard_rows;
    int row = idx % shard_rows;

    dst[col * n_obs + global_row + row] = src[col * shard_rows + row];
}

// Gather shard rows from global col-major matrix into a contiguous shard buffer.
//
// src: (n_obs × k) col-major — the global matrix (e.g., Q)
// dst: (shard_rows × k) col-major — contiguous shard buffer
//
// For each element (row, col) in dst:
//   dst[row, col] = src[(global_row + row), col]
//   i.e. dst[col * shard_rows + row] = src[col * n_obs + global_row + row]
//
// total threads = shard_rows × k
extern "C" __global__ void gather_colmajor_kernel(
    const float* __restrict__ src,   // [n_obs × k], col-major
    float* __restrict__ dst,         // [shard_rows × k], col-major
    int shard_rows,
    int n_obs,
    int k,
    int global_row
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = shard_rows * k;
    if (idx >= total) return;

    int col = idx / shard_rows;
    int row = idx % shard_rows;

    dst[col * shard_rows + row] = src[col * n_obs + global_row + row];
}

// Mean-correct a col-major matrix: Y[r, c] -= mc[c].
//
// Y: (m × k) col-major: Y[r, c] = flat[c * m + r]
// mc: [k] correction vector
//
// total threads = m × k
extern "C" __global__ void mean_correct_colmajor_kernel(
    float* __restrict__ Y,           // [m × k], col-major
    const float* __restrict__ mc,    // [k]
    int m,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * k;
    if (idx >= total) return;

    int col = idx / m;
    Y[idx] -= mc[col];
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
