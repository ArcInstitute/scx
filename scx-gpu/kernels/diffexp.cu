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
// Scatter one CSR shard's rows into a global dense [n_obs × chunk_size]
// row-major buffer at row offset `global_row_offset`, filtered to columns
// [c0, c1). Output column index is (col - c0). Caller MUST zero the
// affected row range before invoking — zeros are implicit in the CSR.
//
// One block per shard row (gridDim.x = n_shard_rows). Each block walks
// its row's nonzero entries cooperatively; threads stride over
// (indptr[r+1] − indptr[r]) and predicate `c0 <= col < c1`.
// ---------------------------------------------------------------------------
extern "C" __global__ void csr_shard_to_dense_chunk_kernel(
    const long*  __restrict__ indptr,   // [n_shard_rows + 1]
    const int*   __restrict__ indices,  // [nnz]
    const float* __restrict__ data,     // [nnz]
    float*       __restrict__ dense,    // [n_obs × chunk_size], row-major
    int n_shard_rows,
    int global_row_offset,
    int chunk_size,
    int c0,
    int c1
) {
    int r = blockIdx.x;
    if (r >= n_shard_rows) return;
    long start = indptr[r];
    long end   = indptr[r + 1];
    long out_row_base = (long long)(global_row_offset + r) * chunk_size;
    for (long e = start + threadIdx.x; e < end; e += blockDim.x) {
        int col = indices[e];
        if (col >= c0 && col < c1) {
            dense[out_row_base + (col - c0)] = data[e];
        }
    }
}

// ---------------------------------------------------------------------------
// All-groups pseudobulk fold.
//
// One block per (gene, group). Threads stride over the group's cell list and
// accumulate the `pre(x)`-transformed value into a single f64 sum via shared-
// memory reduction — no atomics, no per-block contention.
//
// `mode_id` selects the per-cell transform applied before summation, matching
// the host `GeomMeanMode::pre`:
//   0  ArithRaw          → f(x) = x
//   1  ArithLog1pExpand  → f(x) = expm1(x)
//   2  GeomRaw           → f(x) = log1p(x)
//   3  GeomLog1p         → f(x) = x          (post on host does expm1)
//
// The host divides each sum by the group's cell count and applies the matching
// `mode.post` transform (cheap, no kernel needed).
//
// Grid: 2D (chunk_size, n_groups). Block: PSEUDOBULK_BLOCK_THREADS.
// Shared memory: blockDim.x × sizeof(double) bytes (block reduction scratch).
// ---------------------------------------------------------------------------

#define PSEUDOBULK_BLOCK_THREADS 256

__device__ __forceinline__ double apply_pre_transform(float x, int mode_id) {
    double xd = (double)x;
    switch (mode_id) {
        case 0:  return xd;          // ArithRaw / GeomLog1p (identity)
        case 1:  return expm1(xd);   // ArithLog1pExpand
        case 2:  return log1p(xd);   // GeomRaw
        case 3:  return xd;
        default: return xd;
    }
}

extern "C" __global__ void pseudobulk_all_groups_kernel(
    const float* __restrict__ dense,            // [n_obs × chunk_size]
    const int*   __restrict__ all_group_cells,  // flattened, length = sum(n_g)
    const int*   __restrict__ group_offsets,    // [n_groups + 1] CSR-style
    double*      __restrict__ sums,             // [n_groups × chunk_size]
    int n_obs,
    int chunk_size,
    int n_groups,
    int mode_id
) {
    extern __shared__ double sdata[];

    int gene  = blockIdx.x;
    int group = blockIdx.y;
    if (gene >= chunk_size || group >= n_groups) return;

    int start = group_offsets[group];
    int end   = group_offsets[group + 1];

    int tid  = threadIdx.x;
    int nthr = blockDim.x;
    double local_sum = 0.0;
    for (int i = start + tid; i < end; i += nthr) {
        int cell = all_group_cells[i];
        if (cell < 0 || cell >= n_obs) continue;
        float x = dense[(long long)cell * chunk_size + gene];
        local_sum += apply_pre_transform(x, mode_id);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthr >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) sums[(long long)group * chunk_size + gene] = sdata[0];
}

// ---------------------------------------------------------------------------
// Per-gene CUB BlockRadixSort.
//
// Sorts each row of `slab` (shape [chunk_size × n_per_gene], row-major)
// ascending in-place. Used as the fast-path when n_per_gene ≤ BLOCK_SORT_CAPACITY;
// above that, the host dispatches to `tile_block_radix_sort_kernel` +
// `merge_pass_per_gene_kernel` (tiled bottom-up merge sort).
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
// Tiled BlockRadixSort: same body as `block_radix_sort_per_gene_kernel` but
// indexed by (gene, tile). Each block sorts a contiguous up-to-BLOCK_SORT_CAPACITY
// slice of the gene's row in place. Tiles that fall past `n_per_gene` exit early;
// the final partial tile uses the +inf padding trick to stop at the real-data
// boundary.
//
// Grid: 2D (chunk_size, n_tiles). Block: BLOCK_THREADS.
//
// Caller pairs this with `merge_pass_per_gene_kernel` to sort rows of arbitrary
// length via bottom-up merge sort.
// ---------------------------------------------------------------------------
extern "C" __global__ void tile_block_radix_sort_kernel(
    float* __restrict__ slab,
    int chunk_size,
    int n_per_gene,
    int tile_size
) {
    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    int gene = blockIdx.x;
    int tile = blockIdx.y;
    if (gene >= chunk_size) return;

    long long tile_start = (long long)tile * tile_size;
    if (tile_start >= (long long)n_per_gene) return;

    long long tile_end = tile_start + tile_size;
    if (tile_end > (long long)n_per_gene) tile_end = (long long)n_per_gene;
    int n_in_tile = (int)(tile_end - tile_start);

    float* row = slab + (long long)gene * n_per_gene + tile_start;

    // Per-thread items; pad above n_in_tile with +inf so they sort to the
    // end of the tile and can be discarded on writeback.
    float keys[ITEMS_PER_THREAD];
    int tid = threadIdx.x;

    #pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
        int idx = tid * ITEMS_PER_THREAD + it;
        keys[it] = (idx < n_in_tile) ? row[idx] : CUDART_INF_F;
    }

    BlockRadixSort(temp_storage).Sort(keys);

    #pragma unroll
    for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
        int idx = tid * ITEMS_PER_THREAD + it;
        if (idx < n_in_tile) {
            row[idx] = keys[it];
        }
    }
}

// ---------------------------------------------------------------------------
// Merge-path co-rank.
//
// For sorted arrays A[0..m) and B[0..n) being merged into C[0..m+n), find the
// index `i` in [max(0, diag-n), min(diag, m)] such that:
//     A[i-1] ≤ B[j]      AND     B[j-1] ≤ A[i]      (where j = diag - i)
// — i.e. the merge has consumed exactly `i` from A and `j` from B by output
// position `diag`. Boundaries treated as -∞ / +∞ via index guards.
//
// Binary-search-based — O(log(min(m, n))) per call. Called twice per thread
// in `merge_pass_per_gene_kernel` (once for the slice start, once for the end).
//
// Reference: Green/McColl/Bader 2012, "GPU merge path: a GPU merging algorithm"
// (the "MGPU merge" algorithm).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int merge_path_co_rank(
    const float* __restrict__ A, int m,
    const float* __restrict__ B, int n,
    int diag
) {
    int i_lo = max(0, diag - n);
    int i_hi = min(diag, m);
    while (i_lo < i_hi) {
        int i = (i_lo + i_hi) >> 1;
        int j = diag - i;
        // Two violation checks; ≤ in both directions means stable merge
        // (ties from A come first when equal values cross the diagonal).
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            // i too large — pull back, take fewer from A.
            i_hi = i;
        } else if (i < m && j > 0 && B[j - 1] > A[i]) {
            // i too small — push forward, take more from A.
            i_lo = i + 1;
        } else {
            return i;
        }
    }
    return i_lo;
}

// ---------------------------------------------------------------------------
// Merge pass: block-cooperative merge of two adjacent sorted runs of size
// `run_size` into one sorted run of size up to `2 * run_size`. One block per
// (gene, pair); within each block, threads partition the output via merge-path
// co-rank and run sequential merge over their slice.
//
// in_slab and out_slab are distinct buffers (ping-pong managed host-side).
// The host launches `⌈log₂(K)⌉` of these per gene chunk, doubling `run_size`
// each pass, until run_size ≥ n_per_gene.
//
// Edge cases handled inline:
//   * Pair starts at-or-past n_per_gene: early return.
//   * B is empty (last pair when only A's run is non-empty): copy A → C verbatim.
//   * Final partial pair where B's length < run_size: m and n computed per-pair.
//
// Grid: 2D (chunk_size, ⌈n_per_gene / (2 * run_size)⌉). Block: MERGE_BLOCK_THREADS.
// ---------------------------------------------------------------------------

#define MERGE_BLOCK_THREADS 256

extern "C" __global__ void merge_pass_per_gene_kernel(
    const float* __restrict__ in_slab,
    float*       __restrict__ out_slab,
    int chunk_size,
    int n_per_gene,
    int run_size
) {
    int gene = blockIdx.x;
    int pair = blockIdx.y;
    if (gene >= chunk_size) return;

    long long pair_start_ll = (long long)pair * 2 * run_size;
    if (pair_start_ll >= (long long)n_per_gene) return;

    long long a_end_ll = pair_start_ll + run_size;
    if (a_end_ll > (long long)n_per_gene) a_end_ll = (long long)n_per_gene;
    long long b_end_ll = pair_start_ll + 2 * (long long)run_size;
    if (b_end_ll > (long long)n_per_gene) b_end_ll = (long long)n_per_gene;

    int pair_start = (int)pair_start_ll;
    int a_end = (int)a_end_ll;
    int b_start = a_end;
    int b_end = (int)b_end_ll;

    int m = a_end - pair_start;
    int n = b_end - b_start;

    const float* A = in_slab + (long long)gene * n_per_gene + pair_start;
    const float* B = in_slab + (long long)gene * n_per_gene + b_start;
    float*       C = out_slab + (long long)gene * n_per_gene + pair_start;

    int tid  = threadIdx.x;
    int nthr = blockDim.x;

    // B empty (odd run count at last level): just copy A → C.
    if (n == 0) {
        for (int i = tid; i < m; i += nthr) C[i] = A[i];
        return;
    }

    int total = m + n;

    // Each thread handles a contiguous slice of the output [out_start, out_end).
    int per_thread = (total + nthr - 1) / nthr;
    int out_start = tid * per_thread;
    if (out_start >= total) return;
    int out_end = out_start + per_thread;
    if (out_end > total) out_end = total;

    // Co-rank at slice boundaries.
    int i = merge_path_co_rank(A, m, B, n, out_start);
    int j = out_start - i;
    int i_end = merge_path_co_rank(A, m, B, n, out_end);
    int j_end = out_end - i_end;

    // Sequential merge over the slice.
    int k = out_start;
    while (i < i_end && j < j_end) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < i_end) C[k++] = A[i++];
    while (j < j_end) C[k++] = B[j++];
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
