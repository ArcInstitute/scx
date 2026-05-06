// Harmony2 batch integration GPU kernels.
//
// Layout conventions (match scx-accel/src/harmony.rs CPU code):
//   Z_*: (d x N) column-major, f32. Column i = embedding for cell i.
//        Element [t, i] = flat[i * d + t].
//   Y:   (d x K) column-major, f32. Column k = centroid k.
//   R:   (K x N) row-major, f32. Row k contains R[k, :].
//        Element [k, i] = flat[k * N + i].
//   dist:(K x N) row-major, f32. Same layout as R.
//
// All kernels assume row/col indexing described above; callers must respect
// the contract or they will silently corrupt the computation.

// ──────────────────────────────────────────────────────────────────────
// Kernel 1 — Pairwise cosine distance (cell-tiled).
//
//   dist[k, i] = 2 * (1 - dot(Y[:, k], Z_cos[:, i])) for i in [i_offset, i_offset + n_chunk)
//
// Y is (d x K) col-major, Z_cos is (d x N) col-major, dist is (K x N)
// row-major. Each launch processes a slab of n_chunk cells starting at
// i_offset; the host wrapper tiles N to keep K * n_chunk under 2^31 so
// the launch grid (block count = ceil(K*n_chunk/256)) fits in 32 bits.
// Internal indexing uses 64-bit math to address the global K*N output
// — at K=200, N=20M the total can exceed 2^31 even though no single
// launch does.
//
// For typical d (~20-50), per-thread sequential dot product fits in
// registers; shared-memory tiling adds complexity without much benefit.
// Revisit if d grows large.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_distances_kernel(
    const float* __restrict__ Y,      // [d x K], col-major
    const float* __restrict__ Z_cos,  // [d x N], col-major
    float* __restrict__ dist,         // [K x N], row-major
    int d,
    int K,
    int N,
    long long i_offset,
    long long n_chunk
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)K * n_chunk;
    if (idx >= total) return;

    int k = (int)(idx / n_chunk);
    long long i_local = idx - (long long)k * n_chunk;
    long long i = i_offset + i_local;

    const float* y_col = Y + (long long)k * d;
    const float* z_col = Z_cos + i * d;

    float dot = 0.0f;
    #pragma unroll 4
    for (int t = 0; t < d; ++t) {
        dot += y_col[t] * z_col[t];
    }
    dist[(long long)k * N + i] = 2.0f * (1.0f - dot);
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 1b — GEMM-backed distance finalize.
//
// `gpu_harmony_distances_gemm` computes `D = -2 · Z_cos^T · Y` via
// cuBLAS sgemm; this kernel adds the constant 2 in-place to yield
// `D[k, i] = 2 - 2 · dot(Y[:, k], Z_cos[:, i])`. Each thread updates one
// element. 64-bit indexing — D has K*N elements which can exceed 2^31.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_dist_finalize_kernel(
    float* __restrict__ dist,
    long long total
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dist[idx] = 2.0f + dist[idx];
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 2 — Fused softmax with diversity penalty, per cell.
//
// For each cell i, for each cluster k:
//   s[k] = exp(-dist[k, i] / sigma[k]) * diversity[k, i]
// Then column-normalize s to sum to 1 and write R[k, i] = s[k].
//
// diversity[k, i] = prod_c ((2*E[k, b_c(i)] + 1) / (O[k, b_c(i)] + E[k, b_c(i)] + 1)) ^ theta[b_c(i)]
//
// batch_labels_flat encodes per-covariate labels as a row-major (C x N)
// int32 matrix so one cell's C batch levels are reachable with a stride-N
// lookup. cov_offset[c] gives the global batch index of covariate c's
// level 0 (so a cell's global batch = cov_offset[c] + label).
//
// Each thread block handles one cell (up to blockDim.x = 256 threads);
// threads within a block cooperate on the softmax reduction via shared
// memory. For K > 256, threads loop with stride.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_softmax_penalty_kernel(
    const float* __restrict__ dist,          // [K x N], row-major
    const float* __restrict__ sigma,         // [K]
    const float* __restrict__ O,             // [K x B], row-major
    const float* __restrict__ E,             // [K x B], row-major
    const float* __restrict__ theta,         // [B]
    const int*   __restrict__ batch_labels,  // [C x N], row-major (i32)
    const int*   __restrict__ cov_offset,    // [C]
    int C,
    int K,
    int N,
    int B,
    float* __restrict__ R                    // [K x N], row-major, written
) {
    int i = blockIdx.x;
    if (i >= N) return;

    extern __shared__ float smem[];          // size blockDim.x floats
    int tid = threadIdx.x;
    int bsize = blockDim.x;

    // Pass 1: compute the log-weight l[k] = -dist[k,i]/sigma[k] + sum_c theta_c * log_ratio_c
    // and its per-block max for numerical stabilization.
    float local_max = -INFINITY;
    for (int k = tid; k < K; k += bsize) {
        float val = -dist[(long long)k * N + i] / sigma[k];
        for (int c = 0; c < C; ++c) {
            int gb = cov_offset[c] + batch_labels[(long long)c * N + i];
            float o_kb = O[(long long)k * B + gb];
            float e_kb = E[(long long)k * B + gb];
            float num = 2.0f * e_kb + 1.0f;
            float den = o_kb + e_kb + 1.0f;
            float ratio = (den > 0.0f) ? (num / den) : 1.0f;
            float th = theta[gb];
            // ratio^th = exp(th * log(ratio)); guard log(0).
            if (ratio > 0.0f) {
                val += th * logf(ratio);
            }
        }
        // Stash l[k] in R temporarily so we can avoid recomputing in pass 2.
        R[(long long)k * N + i] = val;
        if (val > local_max) local_max = val;
    }
    // Reduce local_max across block threads.
    smem[tid] = local_max;
    __syncthreads();
    for (int s = bsize >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float other = smem[tid + s];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }
    float m = smem[0];
    __syncthreads();

    // Pass 2: exp(l - m) and per-block sum.
    float local_sum = 0.0f;
    for (int k = tid; k < K; k += bsize) {
        float ex = expf(R[(long long)k * N + i] - m);
        R[(long long)k * N + i] = ex;
        local_sum += ex;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = bsize >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float total = smem[0];
    __syncthreads();

    // Pass 3: normalize.
    if (total > 0.0f) {
        float inv = 1.0f / total;
        for (int k = tid; k < K; k += bsize) {
            R[(long long)k * N + i] *= inv;
        }
    } else {
        // Degenerate: uniform.
        float u = 1.0f / (float)K;
        for (int k = tid; k < K; k += bsize) {
            R[(long long)k * N + i] = u;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 3 — Column-wise L2 normalization of a (d x N) col-major matrix.
//
// For each column i, compute ||M[:, i]||_2 and divide in place.
// One thread per column; serial reduction over d. d is typically 20-50.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_l2_normalize_cols_kernel(
    float* __restrict__ M,  // [d x N], col-major
    int d,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float* col = M + (long long)i * d;
    float s = 0.0f;
    for (int t = 0; t < d; ++t) {
        s += col[t] * col[t];
    }
    if (s > 0.0f) {
        float inv = rsqrtf(s);
        for (int t = 0; t < d; ++t) {
            col[t] *= inv;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 3b — Per-cluster z-sum reduction.
//
// For one cluster k, computes the regression right-hand side
//     z_sum[j, t] = sum over cells i in kept-batch j of
//                   R[k, i] * Z_orig[t, i]
// for j ∈ [0, n_kept), t ∈ [0, d).
//
// `cells_concat` and `batch_offsets` use the same convention as the
// grouped correction kernel — cells for batch j are
// `cells_concat[batch_offsets[j]..batch_offsets[j+1]]`. Empty kept
// batches are allowed (offsets[j+1] == offsets[j]); the corresponding
// z_sum row stays zero.
//
// Grid: (n_kept, 1, 1). Block: blockDim.x ≥ d threads (padded to warp
// boundary). One block per (j); each thread handles one t. The thread
// scans the batch's cells sequentially, accumulating in f32. f32 is
// sufficient given R ∈ [0, 1] and Z entries are bounded post-PCA;
// f64 accumulation could be added later if drift becomes an issue.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_z_sum_kernel(
    const float* __restrict__ R_row_k,        // [N]
    const float* __restrict__ Z_orig,         // [d x N], col-major
    const int* __restrict__ cells_concat,     // [n_kept_total]
    const int* __restrict__ batch_offsets,    // [n_kept + 1]
    int n_kept,
    int d,
    int N,
    float* __restrict__ z_sum                 // [n_kept x d], row-major
) {
    int j = blockIdx.x;
    int t = threadIdx.x;
    if (j >= n_kept || t >= d) return;

    int start = batch_offsets[j];
    int end = batch_offsets[j + 1];

    float acc = 0.0f;
    for (int c = start; c < end; ++c) {
        int cell = cells_concat[c];
        float r = R_row_k[cell];
        if (r != 0.0f) {
            acc += r * Z_orig[(long long)cell * d + t];
        }
    }
    z_sum[j * d + t] = acc;
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 4 — Batched correction scatter-subtract.
//
// For a single cluster k and a single qualifying batch b (a list of
// cell indices `cells` of length `n_cells`, along with per-PC weight
// vector `w_row` of length d), apply:
//     Z_corr[:, cells[j]] -= w_row * R_row_k[cells[j]]   for j in 0..n_cells
//
// R_row_k is R[k, :] (length N, from the K×N row-major R matrix).
//
// Grid: 2D launch with (n_cells, d) threads mapped via 1D global index.
// Each thread updates one (cell-in-batch, t) slot in Z_corr.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_correction_kernel(
    float* __restrict__ Z_corr,          // [d x N], col-major
    int d,
    int N,
    const int* __restrict__ cells,       // [n_cells]
    int n_cells,
    const float* __restrict__ R_row_k,   // [N]
    const float* __restrict__ w_row      // [d]
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_cells * (long long)d;
    if (idx >= total) return;

    int j = (int)(idx / d);
    int t = (int)(idx - (long long)j * d);

    int cell = cells[j];
    float r = R_row_k[cell];
    if (r == 0.0f) return;  // nothing to subtract

    Z_corr[(long long)cell * d + t] -= w_row[t] * r;
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 4b — Grouped correction scatter-subtract for one cluster.
//
// Replaces the K * B' per-(cluster, batch) launches with one launch
// per cluster. For cluster k, all kept batches j ∈ [0, n_kept) are
// processed in a single grid; each cell's batch identity is recovered
// via a binary search over `batch_offsets`.
//
// For each (cell_local ∈ [0, n_kept_total), t ∈ [0, d)):
//     j     = batch index s.t. batch_offsets[j] <= cell_local < batch_offsets[j+1]
//     cell  = cells_concat[cell_local]
//     Z_corr[:, cell] -= W[j, t] * R_row_k[cell]
//
// Grid: 1D over `n_kept_total` blocks; each block has `d` threads.
// One block per cell_local; threads cover t ∈ [0, d). Block size is
// padded to the next warp multiple for hardware-friendly launch.
// Thread 0 in each block performs the binary search and stores `j` in
// shared memory; remaining threads use the shared `j`.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_correction_grouped_kernel(
    float* __restrict__ Z_corr,             // [d x N], col-major, in-place
    const float* __restrict__ R_row_k,      // [N]
    const float* __restrict__ W,            // [n_kept x d], row-major (j, t)
    const int* __restrict__ cells_concat,   // [n_kept_total]
    const int* __restrict__ batch_offsets,  // [n_kept + 1]
    int n_kept,
    int n_kept_total,
    int d,
    int N
) {
    int cell_local = blockIdx.x;
    int t = threadIdx.x;
    if (cell_local >= n_kept_total) return;
    if (t >= d) return;  // block dim padded to warp boundary; mask out extras

    __shared__ int j_shared;
    __shared__ int cell_shared;
    __shared__ float r_shared;
    if (t == 0) {
        int cell = cells_concat[cell_local];
        cell_shared = cell;
        r_shared = R_row_k[cell];
        // Binary search batch_offsets for j s.t.
        // batch_offsets[j] <= cell_local < batch_offsets[j+1].
        int lo = 0, hi = n_kept;
        while (lo < hi - 1) {
            int mid = (lo + hi) >> 1;
            if (batch_offsets[mid] <= cell_local) lo = mid;
            else hi = mid;
        }
        j_shared = lo;
    }
    __syncthreads();

    float r = r_shared;
    if (r == 0.0f) return;
    int cell = cell_shared;
    int j = j_shared;

    Z_corr[(long long)cell * d + t] -= W[(long long)j * d + t] * r;
}
