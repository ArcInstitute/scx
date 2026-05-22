// GPU kernels for differential expression (pdex_ref + Wilcoxon MWU).
//
// Chunked Mann-Whitney U via
// per-gene block radix sort (CUB), batched warp-cooperative searchsorted,
// merge-walk combined tie term, and on-device normal-tail p-value via erfc.
// Mirrors the CPU formulae in `scx-accel/src/diffexp.rs`:
//   * U1 = Σ_i (n_ref_less(x_i) + 0.5 · n_ref_equal(x_i))      // ref vs group
//   * Tie correction = Σ_v (c_v^3 − c_v) over distinct values v with count c_v
//   * z = (U1 − μ) / σ;   σ² = (n1·n2/12) · ((N+1) − tc / (N·(N−1)))
//   * p = erfc(|z| / √2)                                       // two-sided
//
// Numerical policy: U statistics are integer-valued and computed in f64
// register accumulators (exact within input range); p-values are tolerance-
// matched against the CPU `libm::erfc` path (atol 1e-9, rtol 1e-6).
//
// Block-radix-sort capacity: BLOCK_THREADS × ITEMS_PER_THREAD = 8192 keys
// per gene. Callers must reject (n_pool > 8192) before invoking the sort
// kernels; the v1 GPU path falls back to CPU for larger reference pools.

#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>  // CUDART_INF_F

// Block-sort tunables (must be compile-time constants for cub::BlockRadixSort).
#define BLOCK_THREADS 1024
#define ITEMS_PER_THREAD 8
#define BLOCK_SORT_CAPACITY (BLOCK_THREADS * ITEMS_PER_THREAD)  // 8192

// ---------------------------------------------------------------------------
// Scatter dense [n_obs × chunk_size] (row-major) into a gene-major slab.
//
// Input  dense[cell * chunk_size + gene]
// Permutation cell_indices[i] (i = 0..n_perm), e.g. ref cells or group cells.
// Output slab[gene * n_perm + i] = dense[cell_indices[i] * chunk_size + gene]
//
// Grid: 2D (n_perm, chunk_size). One thread per output element.
// ---------------------------------------------------------------------------
extern "C" __global__ void scatter_perm_to_gene_major_kernel(
    const float* __restrict__ dense,         // [n_obs × chunk_size]
    const int*   __restrict__ cell_indices,  // [n_perm]
    float*       __restrict__ slab,          // [chunk_size × n_perm]
    int n_obs,
    int n_perm,
    int chunk_size
) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;  // permutation index
    int gene = blockIdx.y * blockDim.y + threadIdx.y;  // gene in chunk
    if (i >= n_perm || gene >= chunk_size) return;

    int cell = cell_indices[i];
    // cell is assumed in [0, n_obs); guard with a finite default to keep the
    // kernel total — out-of-range cell ids are filtered host-side, but we
    // still avoid an out-of-bounds load on a corrupted permutation.
    float v = (cell >= 0 && cell < n_obs)
                  ? dense[(long long)cell * chunk_size + gene]
                  : 0.0f;
    slab[(long long)gene * n_perm + i] = v;
}

// ---------------------------------------------------------------------------
// Per-gene CUB BlockRadixSort.
//
// Sorts each row of `slab` (shape [chunk_size × n_per_gene], row-major)
// ascending in-place. Caller must guarantee n_per_gene <= BLOCK_SORT_CAPACITY;
// CPU pre-check rejects larger pools to avoid silent truncation.
//
// Grid: 1D (chunk_size). Block: BLOCK_THREADS.
// ---------------------------------------------------------------------------
extern "C" __global__ void block_radix_sort_per_gene_kernel(
    float* __restrict__ slab,
    int chunk_size,
    int n_per_gene
) {
    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    int gene = blockIdx.x;
    if (gene >= chunk_size) return;

    float* row = slab + (long long)gene * n_per_gene;

    // Per-thread items; pad above n_per_gene with +inf so they sort to the
    // end of the row and can be cheaply discarded on writeback.
    //
    // cub::BlockRadixSort::Sort consumes AND emits the "blocked" layout —
    // thread `tid` holds items at indices [tid * ITEMS_PER_THREAD,
    // (tid+1) * ITEMS_PER_THREAD). Both load and store must use the same
    // indexing convention or the smallest 1024 sorted items get scattered
    // across the row.
    float keys[ITEMS_PER_THREAD];
    int tid = threadIdx.x;

    #pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
        int idx = tid * ITEMS_PER_THREAD + it;
        keys[it] = (idx < n_per_gene) ? row[idx] : CUDART_INF_F;
    }

    BlockRadixSort(temp_storage).Sort(keys);

    #pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
        int idx = tid * ITEMS_PER_THREAD + it;
        if (idx < n_per_gene) {
            row[idx] = keys[it];
        }
    }
}

// ---------------------------------------------------------------------------
// Tie-term Σ(c^3 − c) over a single sorted row.
//
// One block per gene; thread 0 walks the row sequentially. The walk is
// O(n_per_gene) per gene, fine for v1 with n_per_gene ≤ 8192.
// ---------------------------------------------------------------------------
extern "C" __global__ void tie_term_sorted_kernel(
    const float*  __restrict__ sorted_slab,  // [chunk_size × n_per_gene]
    double*       __restrict__ tie_term,     // [chunk_size]
    int chunk_size,
    int n_per_gene
) {
    int gene = blockIdx.x;
    if (gene >= chunk_size) return;
    if (threadIdx.x != 0) return;

    const float* row = sorted_slab + (long long)gene * n_per_gene;
    double sum = 0.0;
    int i = 0;
    while (i < n_per_gene) {
        int j = i + 1;
        while (j < n_per_gene && row[j] == row[i]) ++j;
        long long c = (long long)(j - i);
        if (c > 1) {
            sum += (double)(c * c * c - c);
        }
        i = j;
    }
    tie_term[gene] = sum;
}

// ---------------------------------------------------------------------------
// Combined tie term Σ((r_v + g_v)^3 − (r_v + g_v)) via merge-walk of two
// pre-sorted rows. One block per gene; thread 0 walks.
// ---------------------------------------------------------------------------
extern "C" __global__ void combined_tie_term_kernel(
    const float*  __restrict__ sorted_ref,    // [chunk_size × n_ref]
    const float*  __restrict__ sorted_group,  // [chunk_size × n_g]
    double*       __restrict__ tie_term,      // [chunk_size]
    int chunk_size,
    int n_ref,
    int n_g
) {
    int gene = blockIdx.x;
    if (gene >= chunk_size) return;
    if (threadIdx.x != 0) return;

    const float* R = sorted_ref + (long long)gene * n_ref;
    const float* G = sorted_group + (long long)gene * n_g;

    double sum = 0.0;
    int i = 0, j = 0;
    while (i < n_ref || j < n_g) {
        float v;
        if (i >= n_ref)       v = G[j];
        else if (j >= n_g)    v = R[i];
        else                  v = fminf(R[i], G[j]);

        long long cr = 0, cg = 0;
        while (i < n_ref && R[i] == v) { ++i; ++cr; }
        while (j < n_g   && G[j] == v) { ++j; ++cg; }
        long long c = cr + cg;
        if (c > 1) {
            sum += (double)(c * c * c - c);
        }
    }
    tie_term[gene] = sum;
}

// ---------------------------------------------------------------------------
// Batched searchsorted U1: U1 = Σ_i (n_ref_less(x_i) + 0.5 · n_ref_equal(x_i))
// for x_i in group_slab[gene, :] against sorted_ref[gene, :].
//
// One block per gene; threads stride over the group cells, partial sums are
// reduced via shared memory. Output u_stats[gene] is exact (integer-valued
// when group/ref are integer counts).
// ---------------------------------------------------------------------------
extern "C" __global__ void searchsorted_u_stat_kernel(
    const float*  __restrict__ sorted_ref,  // [chunk_size × n_ref]
    const float*  __restrict__ group_slab,  // [chunk_size × n_g]
    double*       __restrict__ u_stats,     // [chunk_size]
    int chunk_size,
    int n_ref,
    int n_g
) {
    extern __shared__ double partials[];

    int gene = blockIdx.x;
    if (gene >= chunk_size) return;

    const float* R = sorted_ref + (long long)gene * n_ref;
    const float* G = group_slab + (long long)gene * n_g;

    int tid   = threadIdx.x;
    int nthr  = blockDim.x;
    double local = 0.0;

    for (int i = tid; i < n_g; i += nthr) {
        float x = G[i];
        // lower_bound: first index lo such that R[lo] >= x
        int lo = 0, hi = n_ref;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (R[mid] < x) lo = mid + 1;
            else            hi = mid;
        }
        int n_ref_less = lo;
        // upper_bound: first index hi2 such that R[hi2] > x
        int lo2 = lo, hi2 = n_ref;
        while (lo2 < hi2) {
            int mid = (lo2 + hi2) >> 1;
            if (R[mid] <= x) lo2 = mid + 1;
            else             hi2 = mid;
        }
        int n_ref_equal = lo2 - lo;
        local += (double)n_ref_less + 0.5 * (double)n_ref_equal;
    }

    partials[tid] = local;
    __syncthreads();
    for (int s = nthr >> 1; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        __syncthreads();
    }
    if (tid == 0) u_stats[gene] = partials[0];
}

// ---------------------------------------------------------------------------
// Mid-rank sum from searchsorted against a single global sorted pool.
//
// Used by the 1-vs-rest Wilcoxon path: with all `n_total` cells sorted per
// gene (the "all" pool), the 1-based mid-rank of value x within the pool is
//   rank(x) = lower_bound(x) + (count_equal(x) + 1) / 2
// Σ over group cells gives R_g, and U_g = R_g − n_g(n_g+1)/2.
//
// One block per gene; threads stride over group cells.
// ---------------------------------------------------------------------------
extern "C" __global__ void searchsorted_ranksum_kernel(
    const float*  __restrict__ sorted_all,  // [chunk_size × n_total]
    const float*  __restrict__ group_slab,  // [chunk_size × n_g]
    double*       __restrict__ ranksum,     // [chunk_size]  (output)
    int chunk_size,
    int n_total,
    int n_g
) {
    extern __shared__ double partials[];

    int gene = blockIdx.x;
    if (gene >= chunk_size) return;

    const float* A = sorted_all + (long long)gene * n_total;
    const float* G = group_slab + (long long)gene * n_g;

    int tid  = threadIdx.x;
    int nthr = blockDim.x;
    double local = 0.0;

    for (int i = tid; i < n_g; i += nthr) {
        float x = G[i];
        int lo = 0, hi = n_total;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (A[mid] < x) lo = mid + 1;
            else            hi = mid;
        }
        int n_all_less = lo;
        int lo2 = lo, hi2 = n_total;
        while (lo2 < hi2) {
            int mid = (lo2 + hi2) >> 1;
            if (A[mid] <= x) lo2 = mid + 1;
            else             hi2 = mid;
        }
        int n_all_equal = lo2 - lo;
        // 1-based mid-rank
        double rank = (double)n_all_less + 0.5 * (double)(n_all_equal + 1);
        local += rank;
    }

    partials[tid] = local;
    __syncthreads();
    for (int s = nthr >> 1; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        __syncthreads();
    }
    if (tid == 0) ranksum[gene] = partials[0];
}

// ---------------------------------------------------------------------------
// MWU p-value via normal approximation with tie correction.
//
// Matches `scx-accel::diffexp::wilcoxon_full_from_ranks`:
//   μ      = n1 * n2 / 2
//   σ²     = (n1 * n2 / 12) * ((N + 1) − tc / (N * (N − 1)))   (N = n1+n2)
//   z      = (U1 − μ) / σ
//   p      = erfc(|z| / √2)                                    (two-sided)
//
// No continuity correction (matches CPU exactly). Clips p to [0, 1] before
// writing, mirroring pdex_ref's clamp.
// ---------------------------------------------------------------------------
extern "C" __global__ void pvalue_erfc_kernel(
    const double* __restrict__ u_stats,
    const double* __restrict__ tie_term,
    double*       __restrict__ p_values,
    int chunk_size,
    int n1,
    int n2
) {
    int gene = blockIdx.x * blockDim.x + threadIdx.x;
    if (gene >= chunk_size) return;

    double u1  = u_stats[gene];
    double n1d = (double)n1;
    double n2d = (double)n2;
    double n   = n1d + n2d;

    if (n1 == 0 || n2 == 0) {
        p_values[gene] = 1.0;
        return;
    }
    double mu = n1d * n2d * 0.5;
    double tc = tie_term[gene];
    double denom = n * (n - 1.0);
    double sigma_sq = (n1d * n2d / 12.0) * ((n + 1.0) - (denom > 0.0 ? tc / denom : 0.0));
    if (!(sigma_sq > 0.0)) {
        p_values[gene] = 1.0;
        return;
    }
    double z = (u1 - mu) / sqrt(sigma_sq);
    double p = erfc(fabs(z) * 0.7071067811865475);  // 1/√2
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    p_values[gene] = p;
}
