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

    // Block reduction via shared memory (one value per warp)
    __shared__ float warp_sums[32]; // max 32 warps per block (1024 threads)
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
