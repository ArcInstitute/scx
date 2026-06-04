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

#include <cassert>

// The shared-memory tree reductions in this file (`for (s = bsize >> 1; s > 0;
// s >>= 1)`) assume a power-of-two block size: a non-pow2 block drops the odd
// top partial and silently under-counts the reduction. Every launch site uses
// pow2 blocks (callers double from 32), so the bug is latent — this guard turns
// a future non-pow2 launch into a device-side trap instead of a silent wrong
// result (finding ACC2). Asserted once per affected kernel after `bsize` is set.
#define SCX_ASSERT_POW2_BLOCK() assert((blockDim.x & (blockDim.x - 1)) == 0)

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
    SCX_ASSERT_POW2_BLOCK();

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
// Kernel 2b — Plain softmax (no diversity penalty).
//
// Identical to `harmony_softmax_penalty_kernel` with C = 0 — produces
// `R[:, i] = softmax(-dist[:, i] / sigma)` for every cell. Used by
// the GPU orchestrator at iter > 0 cold-start, where the diversity
// penalty is *not* applied yet (CPU `softmax_r_from_dist` does the
// same; the per-block penalty is folded into `update_r`'s inner
// softmax via leave-block-out O/E).
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_softmax_kernel(
    const float* __restrict__ dist,   // [K x N], row-major
    const float* __restrict__ sigma,  // [K]
    int K,
    int N,
    float* __restrict__ R             // [K x N], row-major, written
) {
    int i = blockIdx.x;
    if (i >= N) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bsize = blockDim.x;
    SCX_ASSERT_POW2_BLOCK();

    // Pass 1: l[k] = -dist/sigma; per-block max for numerical stability.
    float local_max = -INFINITY;
    for (int k = tid; k < K; k += bsize) {
        float val = -dist[(long long)k * N + i] / sigma[k];
        R[(long long)k * N + i] = val;
        if (val > local_max) local_max = val;
    }
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
        float u = 1.0f / (float)K;
        for (int k = tid; k < K; k += bsize) {
            R[(long long)k * N + i] = u;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 2c — Block softmax+penalty for the k-means update_r sub-loop.
//
// Like `harmony_softmax_penalty_kernel` but operates on an explicit
// list of cell indices `block_cells[blockIdx.x]` instead of all N
// cells. Each block in the grid handles one cell. The diversity
// penalty uses the *current* O/E values (callers are responsible for
// passing leave-block-out O/E by subtracting the block's contribution
// before launching this kernel).
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_block_softmax_penalty_kernel(
    const float* __restrict__ dist,           // [K x N]
    const float* __restrict__ sigma,          // [K]
    const float* __restrict__ O,              // [K x B] (leave-block-out)
    const float* __restrict__ E,              // [K x B] (leave-block-out)
    const float* __restrict__ theta,          // [B]
    const int*   __restrict__ batch_labels,   // [C x N]
    const int*   __restrict__ cov_offset,     // [C]
    const int*   __restrict__ block_cells,    // [n_block_cells] global cell indices
    int C,
    int K,
    int N,
    int B,
    int n_block_cells,
    float* __restrict__ R                     // [K x N], rows for block cells written
) {
    int bi = blockIdx.x;
    if (bi >= n_block_cells) return;
    int i = block_cells[bi];
    if (i < 0 || i >= N) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bsize = blockDim.x;
    SCX_ASSERT_POW2_BLOCK();

    // Pass 1: l[k] = -dist/sigma + Σ_c θ_c * log(ratio_c); per-block max.
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
            if (ratio > 0.0f) val += th * logf(ratio);
        }
        R[(long long)k * N + i] = val;
        if (val > local_max) local_max = val;
    }
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
        float u = 1.0f / (float)K;
        for (int k = tid; k < K; k += bsize) {
            R[(long long)k * N + i] = u;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 2d — Signed O/E update for a block of cells.
//
// For each (cell-in-block, cluster, covariate) triple, atomically adds
// `sign * R[k, cell]` to `O[k, gb]` and `sign * pr_b[gb] * R[k, cell]`
// to `E[k, gb]`. With `sign = -1` this implements the block-decrement
// step of `update_r` (subtract the block's R contributions); with
// `sign = +1` it implements the increment step (add the new R back
// after softmax). f32 atomicAdd is supported natively on H100 / A100.
//
// Atomic contention scales with cells_per_(k, gb) bin in the block;
// in typical single-covariate runs (block_size ≈ 5% of N, K ≈ 100,
// B ≈ 10–100) the contention is bounded.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_block_oe_update_kernel(
    const float* __restrict__ R,             // [K x N]
    const int*   __restrict__ block_cells,   // [n_block_cells]
    const int*   __restrict__ batch_labels,  // [C x N]
    const int*   __restrict__ cov_offset,    // [C]
    const float* __restrict__ pr_b,          // [B]
    float sign,                              // +1.0 or -1.0
    int C,
    int K,
    int N,
    int B,
    int n_block_cells,
    float* __restrict__ O,                   // [K x B]
    float* __restrict__ E                    // [K x B]
) {
    long long total = (long long)n_block_cells * (long long)K * (long long)C;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = (int)(idx % (long long)C);
    long long ki_idx = idx / (long long)C;
    int k = (int)(ki_idx % (long long)K);
    int bi = (int)(ki_idx / (long long)K);

    int i = block_cells[bi];
    if (i < 0 || i >= N) return;

    float r = R[(long long)k * N + i];
    if (r == 0.0f) return;

    int gb = cov_offset[c] + batch_labels[(long long)c * N + i];
    float r_signed = sign * r;
    atomicAdd(&O[(long long)k * B + gb], r_signed);
    atomicAdd(&E[(long long)k * B + gb], r_signed * pr_b[gb]);
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 2e — Compute O / E from R (full-N initial reduction).
//
// O[k, gb] = Σ_i R[k, i] (where gb is computed per cell from the
// covariate label table). E[k, gb] = pr_b[gb] * Σ_i R[k, i] = pr_b[gb]
// * row_sum[k]. We accumulate into both O and a per-cluster row_sum
// scratch in a single pass; a second small kernel finalises E from
// pr_b * row_sum. Only used at iter > 0 cold-start to refresh O/E
// from a freshly-computed (penalty-free) R.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_compute_o_kernel(
    const float* __restrict__ R,             // [K x N]
    const int*   __restrict__ batch_labels,  // [C x N]
    const int*   __restrict__ cov_offset,    // [C]
    int C,
    int K,
    int N,
    int B,
    float* __restrict__ O,                   // [K x B]
    float* __restrict__ row_sum              // [K]
) {
    long long total = (long long)N * (long long)K * (long long)C;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = (int)(idx % (long long)C);
    long long ki_idx = idx / (long long)C;
    int k = (int)(ki_idx % (long long)K);
    int i = (int)(ki_idx / (long long)K);

    float r = R[(long long)k * N + i];
    if (r == 0.0f) return;

    int gb = cov_offset[c] + batch_labels[(long long)c * N + i];
    atomicAdd(&O[(long long)k * B + gb], r);
    if (c == 0) {
        atomicAdd(&row_sum[k], r);
    }
}

// Finalize E from pr_b * row_sum. One thread per (k, gb).
extern "C" __global__ void harmony_compute_e_finalize_kernel(
    const float* __restrict__ row_sum,  // [K]
    const float* __restrict__ pr_b,     // [B]
    int K,
    int B,
    float* __restrict__ E               // [K x B]
) {
    int k = blockIdx.y;
    int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K || gb >= B) return;
    E[(long long)k * B + gb] = pr_b[gb] * row_sum[k];
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 2f — Per-cell kmeans + entropy objective contribution.
//
// obj_cell[i] = Σ_k R[k, i] * dist[k, i]              (kmeans_err)
//             + Σ_k sigma[k] * R[k, i] * log(R[k, i]) (entropy)
//
// One block per cell; threads cooperate on the K-reduction in shared
// memory. Reducing `obj_cell` over i gives `kmeans_err + entropy`
// (the cross-entropy term needs its own per-(k, gb) reduction).
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_obj_kmeans_entropy_kernel(
    const float* __restrict__ R,       // [K x N]
    const float* __restrict__ dist,    // [K x N]
    const float* __restrict__ sigma,   // [K]
    int K,
    int N,
    float* __restrict__ obj_cell       // [N]
) {
    int i = blockIdx.x;
    if (i >= N) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bsize = blockDim.x;
    SCX_ASSERT_POW2_BLOCK();

    float local = 0.0f;
    for (int k = tid; k < K; k += bsize) {
        float r = R[(long long)k * N + i];
        float d = dist[(long long)k * N + i];
        local += r * d;
        if (r > 0.0f) {
            local += sigma[k] * r * logf(r);
        }
    }
    smem[tid] = local;
    __syncthreads();
    for (int s = bsize >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) obj_cell[i] = smem[0];
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 2g — Per-(k, gb) cross-entropy objective contribution.
//
// cross_kgb[k, gb] = sigma[k] * O[k, gb] * theta[gb] * log((O+E+1)/(2E+1))
//
// Reducing `cross_kgb` over (k, gb) gives the cross-entropy term.
// ──────────────────────────────────────────────────────────────────────
extern "C" __global__ void harmony_obj_cross_kernel(
    const float* __restrict__ O,       // [K x B]
    const float* __restrict__ E,       // [K x B]
    const float* __restrict__ sigma,   // [K]
    const float* __restrict__ theta,   // [B]
    int K,
    int B,
    float* __restrict__ cross_kgb      // [K x B]
) {
    int k = blockIdx.y;
    int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K || gb >= B) return;

    float o_kb = O[(long long)k * B + gb];
    float e_kb = E[(long long)k * B + gb];
    float num = o_kb + e_kb + 1.0f;
    float den = 2.0f * e_kb + 1.0f;
    float val = 0.0f;
    if (num > 0.0f && den > 0.0f) {
        val = sigma[k] * o_kb * theta[gb] * logf(num / den);
    }
    cross_kgb[(long long)k * B + gb] = val;
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

    // Block dim is padded to a warp multiple, so threads with t >= d
    // exist but must stay alive through __syncthreads() — exiting
    // before the barrier within a warp is UB per the CUDA spec.
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

    if (t >= d) return;  // mask out warp-padding threads after the barrier
    float r = r_shared;
    if (r == 0.0f) return;
    int cell = cell_shared;
    int j = j_shared;

    Z_corr[(long long)cell * d + t] -= W[(long long)j * d + t] * r;
}
