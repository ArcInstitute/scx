// GPU kernels for differential expression (pdex_ref + Wilcoxon MWU).
//
// Chunked Mann-Whitney U via
// per-gene block radix sort (CUB), batched warp-cooperative searchsorted,
// merge-walk combined tie term, and on-device normal-tail p-value via erfc.
// Mirrors the CPU formulae in `scx-accel/src/diffexp.rs`:
//   * U1 = Σ_i (n_ref_less(x_i) + 0.5 · n_ref_equal(x_i))      // group's U:
//     for each group value x_i, count ref values it exceeds (+½ per tie)
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
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>  // CUDART_INF_F
#include <cassert>

// The hand-rolled shared-memory tree reductions below (`for (s = nthr >> 1;
// s > 0; s >>= 1)`) assume a power-of-two block size; a non-pow2 block drops
// the odd top partial and silently under-counts. Every launch site uses pow2
// blocks (128/64/32 are hardcoded), so the bug is latent — this guard turns a
// future non-pow2 launch into a device-side trap instead of a silently wrong
// statistic (finding ACC2). Asserted once per affected kernel after `nthr` is
// set. (The cub::BlockReduce / cub::BlockRadixSort paths are non-pow2-safe and
// need no guard.)
#define SCX_ASSERT_POW2_BLOCK() assert((blockDim.x & (blockDim.x - 1)) == 0)

// Block-sort tunables (must be compile-time constants for cub::BlockRadixSort).
#define BLOCK_THREADS 1024
#define ITEMS_PER_THREAD 8
#define BLOCK_SORT_CAPACITY (BLOCK_THREADS * ITEMS_PER_THREAD)  // 8192

// ---------------------------------------------------------------------------
// Per-row gene-chunk window (§9.11).
//
// Every CSR row-scan kernel below is launched once per gene chunk and keeps
// only the nonzeros whose column falls in `[c0, c1)`. Scanning the row's full
// nonzero range and predicating per element makes the *kernel* cost O(nnz) per
// chunk — with 61,497 genes at a 500-gene chunk that is the whole matrix walked
// 123 times, the same quadratic on the compute side that device residency
// removes on the decode side.
//
// Every one of those kernels already requires **strictly increasing per-row
// column indices** (each documents it; `shard_validate::validate_shard` enforces it
// release-active at the host staging boundary, since a duplicate column races
// the scatter). Sorted indices mean the chunk's columns are a contiguous
// sub-range of the row, so two binary searches replace the linear scan and the
// per-chunk cost becomes O(nnz / n_chunks + log(row_len)).
//
// `lower_bound` semantics: the first index in `[lo, hi)` whose column is `>=
// key`, or `hi` when none is. Applied as `[lower_bound(c0), lower_bound(c1))`
// this selects exactly `{e : c0 <= indices[e] < c1}` — the identical element
// set the predicate selected, so the surviving writes and their values are
// unchanged.
//
// Every thread in the block runs the same search redundantly rather than
// computing it once and broadcasting through shared memory: the searches are
// warp-uniform, hit the same cache lines, and cost far less than the
// `__syncthreads()` a broadcast would need.
// ---------------------------------------------------------------------------
__device__ __forceinline__ long long scx_row_lower_bound(
    const int* __restrict__ indices,
    long long lo,
    long long hi,
    int key
) {
    while (lo < hi) {
        long long mid = lo + ((hi - lo) >> 1);
        if (indices[mid] < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

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
// One block per shard row (gridDim.x = n_shard_rows). Each block binary-
// searches its row down to the `[c0, c1)` column window (see
// `scx_row_lower_bound`) and strides over that window cooperatively.
//
// PRECONDITION: each row's column indices must be strictly increasing
// (no duplicates). Threads write `dense[...] = data[e]` in parallel, so
// duplicate `(row, col)` entries race and the winning value is
// nondeterministic. The Rust wrapper enforces this via
// `shard_validate::validate_shard` at the host-side staging boundary as a
// release-active check (returns `GpuError::InvalidShard`), not a
// debug_assert.
//
// `indptr` is 64-bit (matches Rust `i64` and CSR-shard nnz ≥ 2^31);
// `global_row_offset` is 64-bit so atlases with n_obs > 2^31 don't
// overflow the row-base address computation. `long long` is used in
// preference to `long` for portability across LP64 (Linux) and LLP64
// (Windows) ABIs.
// ---------------------------------------------------------------------------
extern "C" __global__ void csr_shard_to_dense_chunk_kernel(
    const long long* __restrict__ indptr,   // [n_shard_rows + 1]
    const int*       __restrict__ indices,  // [nnz]
    const float*     __restrict__ data,     // [nnz]
    float*           __restrict__ dense,    // [n_obs × chunk_size], row-major
    int       n_shard_rows,
    long long global_row_offset,
    int       chunk_size,
    int       c0,
    int       c1
) {
    int r = blockIdx.x;
    if (r >= n_shard_rows) return;
    // Narrow to the row's [c0, c1) window; see `scx_row_lower_bound`.
    long long w0 = scx_row_lower_bound(indices, indptr[r], indptr[r + 1], c0);
    long long w1 = scx_row_lower_bound(indices, w0, indptr[r + 1], c1);
    long long out_row_base = (global_row_offset + (long long)r) * (long long)chunk_size;
    for (long long e = w0 + threadIdx.x; e < w1; e += blockDim.x) {
        dense[out_row_base + (indices[e] - c0)] = data[e];
    }
}

// ---------------------------------------------------------------------------
// G4 v2: Scatter one CSR shard's rows directly into a gene-major slab,
// skipping the dense `[n_obs × chunk_size]` intermediate.
//
// Per-thread output: slab[(col - c0) * n_perm + pool_pos] = data[e]
//   where pool_pos = cell_to_pool[global_row_offset + r]
//
// `cell_to_pool` is a length-n_obs i32 array; entries are -1 for cells
// that are NOT in this pool (ref or test group) and 0..n_perm-1 for cells
// that ARE. The caller computes it once per (chunk-loop, pool) via
// `inverse_perm()`.
//
// Whole-block early exit: if the global cell is not in this pool, the
// block returns before iterating its CSR row — this is the common case
// for test-group pools (~n_test cells of n_obs).
//
// `slab` MUST be pre-zeroed by the caller before the first shard's
// scatter (`memset_zeros(&mut slab.slice_mut(..n_perm * chunk_size))`),
// because zeros are implicit in the CSR representation.
//
// NO RACE: each (cell_global, col) pair writes one unique output cell
// — pool_pos is unique per cell (it's an inverse permutation), and CSR
// rows have strictly increasing column indices (no duplicates).
// ---------------------------------------------------------------------------
extern "C" __global__ void csr_shard_to_gene_major_kernel(
    const long long* __restrict__ indptr,        // [n_shard_rows + 1]
    const int*       __restrict__ indices,       // [nnz]
    const float*     __restrict__ data,          // [nnz]
    const int*       __restrict__ cell_to_pool,  // [n_obs], -1 if not in pool
    float*           __restrict__ slab,          // [chunk_size × n_perm]
    int       n_shard_rows,
    long long global_row_offset,
    int       n_perm,
    int       c0,
    int       c1
) {
    int r = blockIdx.x;
    if (r >= n_shard_rows) return;
    long long cell_global = global_row_offset + (long long)r;
    int pool_pos = cell_to_pool[cell_global];
    if (pool_pos < 0) return;  // cell not in this pool; whole-block early exit

    // Narrow to the row's [c0, c1) window; see `scx_row_lower_bound`.
    long long w0 = scx_row_lower_bound(indices, indptr[r], indptr[r + 1], c0);
    long long w1 = scx_row_lower_bound(indices, w0, indptr[r + 1], c1);
    for (long long e = w0 + threadIdx.x; e < w1; e += blockDim.x) {
        long long gene_local = (long long)(indices[e] - c0);
        slab[gene_local * (long long)n_perm + (long long)pool_pos] = data[e];
    }
}

// ---------------------------------------------------------------------------
// G4 v3 (post code-review #6, CSR variant): identical scatter shape to
// `csr_shard_to_gene_major_kernel` above, but reads the cell membership
// from combined `cell_to_group + cell_to_pos` tables (with a launch-time
// `this_group_id` filter). Used by the v3 drivers ONLY; the original
// per-pool kernel above stays in place for v2's caller in G4.1's drivers,
// since collapsing v2 is out of scope for PR #133.
//
// Same race/correctness story as the v2 variant: whole-block early exit
// when the row's cell is not in `this_group_id`; threads stride the row's
// nonzeros and write to `slab[gene_local × n_perm + pos]` with `pos =
// cell_to_pos[cell]`. NO atomicAdd.
// ---------------------------------------------------------------------------
extern "C" __global__ void csr_shard_to_gene_major_filtered_kernel(
    const long long* __restrict__ indptr,         // [n_shard_rows + 1]
    const int*       __restrict__ indices,        // [nnz]
    const float*     __restrict__ data,           // [nnz]
    const int*       __restrict__ cell_to_group,  // [n_obs] group id or -1
    const int*       __restrict__ cell_to_pos,    // [n_obs] in-group pos or -1
    int this_group_id,
    float*           __restrict__ slab,           // [chunk_size × n_perm]
    int       n_shard_rows,
    long long global_row_offset,
    int       n_perm,
    int       c0,
    int       c1
) {
    int r = blockIdx.x;
    if (r >= n_shard_rows) return;
    long long cell_global = global_row_offset + (long long)r;
    if (cell_to_group[cell_global] != this_group_id) return;  // whole-block exit
    int pos = cell_to_pos[cell_global];

    // Narrow to the row's [c0, c1) window; see `scx_row_lower_bound`.
    long long w0 = scx_row_lower_bound(indices, indptr[r], indptr[r + 1], c0);
    long long w1 = scx_row_lower_bound(indices, w0, indptr[r + 1], c1);
    for (long long e = w0 + threadIdx.x; e < w1; e += blockDim.x) {
        long long gene_local = (long long)(indices[e] - c0);
        slab[gene_local * (long long)n_perm + (long long)pos] = data[e];
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
    SCX_ASSERT_POW2_BLOCK();
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
// G4 v3 (CSC-direct): all-groups pseudobulk fold over a CSC shard.
//
// One block per gene in the chunk; threads cooperate over the column's
// nonzeros, accumulating per-thread per-group sums in shared memory, then
// tree-reducing. Thread 0 of the block adds the block's result to the
// running per-(group, gene_local) sum.
//
// NO ATOMICS: each (gene, group) pair has exactly one writer (block 0 of
// gene_local). Cross-shard accumulation is safe because shards are
// serialized on the device stream — each launch reads the running sum,
// adds this shard's contribution, writes back.
//
// Shared memory: [n_groups × blockDim.x] doubles. With blockDim.x = 128
// and n_groups ≤ 32, that's ≤ 32 KB per block — fits comfortably in the
// 48 KB default shared-mem allocation.
//
// `sums` MUST be pre-zeroed before the FIRST shard's invocation.
//
// `n_obs` bounds `cell_to_group`. Two host-side layers already reject a row
// index outside it — the shard decoder's `check_minor_indices` on any backed
// file, then `shard_validate::validate_shard` before staging — so this guard is
// unreachable in practice and deliberately kept anyway: an out-of-bounds device
// read poisons the whole CUDA context, and review §8.3 was precisely a
// host-side check that existed but had never been wired to this route.
// ---------------------------------------------------------------------------
extern "C" __global__ void csc_shard_pseudobulk_kernel(
    const long long* __restrict__ col_indptr,     // [n_cols_in_shard + 1]
    const int*       __restrict__ row_indices,    // [nnz] global row indices
    const float*     __restrict__ data,           // [nnz]
    const int*       __restrict__ cell_to_group,  // [n_obs], -1 if not in any group
    double*          __restrict__ sums,           // [n_groups × chunk_size]
    int n_cols_in_shard,
    int shard_col_start,
    int c0,
    int chunk_size,
    int n_groups,
    int mode_id,
    int n_obs
) {
    int gene_local = blockIdx.x;
    if (gene_local >= chunk_size) return;
    int gene_global = c0 + gene_local;
    int col_in_shard = gene_global - shard_col_start;
    if (col_in_shard < 0 || col_in_shard >= n_cols_in_shard) return;

    extern __shared__ double sdata[];
    int tid  = threadIdx.x;
    int nthr = blockDim.x;
    SCX_ASSERT_POW2_BLOCK();
    for (int g = 0; g < n_groups; g++) {
        sdata[(long long)g * nthr + tid] = 0.0;
    }
    __syncthreads();

    long long start = col_indptr[col_in_shard];
    long long end   = col_indptr[col_in_shard + 1];
    for (long long e = start + tid; e < end; e += nthr) {
        int cell = row_indices[e];
        if (cell < 0 || cell >= n_obs) continue;
        int g = cell_to_group[cell];
        if (g >= 0 && g < n_groups) {
            sdata[(long long)g * nthr + tid] += apply_pre_transform(data[e], mode_id);
        }
    }
    __syncthreads();

    // Tree-reduce per group.
    for (int s = nthr >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            for (int g = 0; g < n_groups; g++) {
                sdata[(long long)g * nthr + tid] +=
                    sdata[(long long)g * nthr + tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int g = 0; g < n_groups; g++) {
            sums[(long long)g * chunk_size + gene_local] += sdata[(long long)g * nthr];
        }
    }
}

// ---------------------------------------------------------------------------
// G4 v3 (CSC-direct, many-groups): global-atomic pseudobulk fold over a CSC
// shard. Same one-block-per-gene structure and arguments as
// `csc_shard_pseudobulk_kernel`, but with NO shared-memory per-group
// accumulator — threads `atomicAdd` directly into the global
// `sums[group × chunk_size + gene_local]`.
//
// This lifts the `n_groups` ceiling of the SMEM-staged kernel (whose
// `n_groups × blockDim.x × sizeof(double)` shared-memory footprint exceeds the
// per-block hardware limit beyond ~900 groups on H100). The host wrapper
// selects this kernel when no `blockDim.x ∈ {128,64,32}` lets the SMEM variant
// fit; that regime has *many* groups, where per-(group,gene) atomic contention
// is naturally low (one block per gene; within a block only same-group cells of
// that gene contend). Mirrors `csr_shard_pseudobulk_kernel`.
//
// Cross-shard accumulation is identical to the SMEM/CSR kernels: `sums` MUST be
// pre-zeroed before the FIRST shard, and per-shard launches are serialized on
// the device stream. NOTE: f64 `atomicAdd` makes the summation order
// run-to-run nondeterministic (matching the CSR-direct path), unlike the
// deterministic tree-reduce of `csc_shard_pseudobulk_kernel`.
// ---------------------------------------------------------------------------
extern "C" __global__ void csc_shard_pseudobulk_global_kernel(
    const long long* __restrict__ col_indptr,     // [n_cols_in_shard + 1]
    const int*       __restrict__ row_indices,    // [nnz] global row indices
    const float*     __restrict__ data,           // [nnz]
    const int*       __restrict__ cell_to_group,  // [n_obs], -1 if not in any group
    double*          __restrict__ sums,           // [n_groups × chunk_size]
    int n_cols_in_shard,
    int shard_col_start,
    int c0,
    int chunk_size,
    int n_groups,
    int mode_id,
    int n_obs                                     // bounds cell_to_group; see above
) {
    int gene_local = blockIdx.x;
    if (gene_local >= chunk_size) return;
    int gene_global = c0 + gene_local;
    int col_in_shard = gene_global - shard_col_start;
    if (col_in_shard < 0 || col_in_shard >= n_cols_in_shard) return;

    long long start = col_indptr[col_in_shard];
    long long end   = col_indptr[col_in_shard + 1];
    for (long long e = start + threadIdx.x; e < end; e += blockDim.x) {
        int cell = row_indices[e];
        if (cell < 0 || cell >= n_obs) continue;
        int g = cell_to_group[cell];
        if (g >= 0 && g < n_groups) {
            atomicAdd(&sums[(long long)g * (long long)chunk_size + (long long)gene_local],
                      apply_pre_transform(data[e], mode_id));
        }
    }
}

// ---------------------------------------------------------------------------
// G4 v3 (CSC-direct, post code-review #6): scatter one CSC shard's nonzeros
// into a gene-major pool slab using combined cell_to_group + cell_to_pos
// tables instead of K+1 per-pool `cell_to_pool` tables. Collapses
// `(K+1) × n_obs × 4` device memory to `2 × n_obs × 4` regardless of K
// (atlas-scale wins: 4 GB → 80 MB at n_obs=10M, K=100).
//
// One block per gene; threads stride the column's nonzeros. The launch
// caller passes `this_group_id` (0 = ref, 1..=K = test groups); the kernel
// writes only the cells where `cell_to_group[cell] == this_group_id`,
// using `cell_to_pos[cell]` as the in-group position.
//
// NO RACE, NO atomicAdd: each (gene_local, pos) cell has at most one
// writer per launch. K+1 launches per shard (one per pool); same launch
// count as the pre-#6 kernel.
//
// `slab` MUST be pre-zeroed by the caller before the FIRST shard's scatter.
//
// `n_obs` bounds both cell tables; see `csc_shard_pseudobulk_kernel` above for
// why the guard is here as well as on the host.
// ---------------------------------------------------------------------------
extern "C" __global__ void csc_shard_to_gene_major_kernel(
    const long long* __restrict__ col_indptr,     // [n_cols_in_shard + 1]
    const int*       __restrict__ row_indices,    // [nnz] global row indices
    const float*     __restrict__ data,           // [nnz]
    const int*       __restrict__ cell_to_group,  // [n_obs] group id or -1
    const int*       __restrict__ cell_to_pos,    // [n_obs] in-group pos or -1
    int this_group_id,                            // which group's slab this fills
    float*           __restrict__ slab,           // [chunk_size × n_perm]
    int n_cols_in_shard,
    int shard_col_start,
    int c0,
    int chunk_size,
    int n_perm,
    int n_obs                                     // bounds cell_to_group / cell_to_pos
) {
    int gene_local = blockIdx.x;
    if (gene_local >= chunk_size) return;
    int gene_global = c0 + gene_local;
    int col_in_shard = gene_global - shard_col_start;
    if (col_in_shard < 0 || col_in_shard >= n_cols_in_shard) return;

    long long start = col_indptr[col_in_shard];
    long long end   = col_indptr[col_in_shard + 1];
    for (long long e = start + threadIdx.x; e < end; e += blockDim.x) {
        int cell = row_indices[e];
        if (cell < 0 || cell >= n_obs) continue;
        if (cell_to_group[cell] == this_group_id) {
            int pos = cell_to_pos[cell];
            slab[(long long)gene_local * (long long)n_perm + (long long)pos] = data[e];
        }
    }
}

// ---------------------------------------------------------------------------
// G4 v3 (CSR-direct): CSR-shard pseudobulk fallback for SCX files without
// a CSC sidecar. Accumulates per-(group, gene) f64 sums by streaming a CSR
// shard view directly, skipping the [n_obs × chunk_size] dense intermediate
// that `pseudobulk_all_groups_kernel` reads from.
//
// One block per shard row; whole-block early exit when the cell is not in
// any group (`cell_to_group[global_cell] < 0`). Threads stride over the
// row's `[c0, c1)` column window (see `scx_row_lower_bound`), apply the
// matching `pre()` transform per element, and
// f64-atomicAdd into `sums[group × chunk_size + gene_local]`. f64 atomicAdd
// is hardware-native on H100 (CC 9.0; shipped in CC 6.0).
//
// Implicit zeros (CSR sparsity) contribute nothing because
// `apply_pre_transform(0.0)` is 0.0 for all four mode_ids — iterating only
// over nonzeros is mathematically equivalent to the dense-read path.
//
// `sums` MUST be pre-zeroed before the FIRST shard's invocation; atomicAdd
// accumulates. NO RACE within a single CSR row (column indices strictly
// increasing); cross-cell contention on hot (group, gene) cells is what
// atomicAdd handles.
// ---------------------------------------------------------------------------
extern "C" __global__ void csr_shard_pseudobulk_kernel(
    const long long* __restrict__ indptr,         // [n_shard_rows + 1]
    const int*       __restrict__ indices,        // [nnz]
    const float*     __restrict__ data,           // [nnz]
    const int*       __restrict__ cell_to_group,  // [n_obs], -1 if not in any group
    double*          __restrict__ sums,           // [n_groups × chunk_size]
    int       n_shard_rows,
    long long global_row_offset,
    int       chunk_size,
    int       c0,
    int       c1,
    int       mode_id
) {
    int r = blockIdx.x;
    if (r >= n_shard_rows) return;
    long long cell_global = global_row_offset + (long long)r;
    int g = cell_to_group[cell_global];
    if (g < 0) return;  // cell not in any group; whole-block early exit

    // Narrow to the row's [c0, c1) window; see `scx_row_lower_bound`.
    long long w0 = scx_row_lower_bound(indices, indptr[r], indptr[r + 1], c0);
    long long w1 = scx_row_lower_bound(indices, w0, indptr[r + 1], c1);
    for (long long e = w0 + threadIdx.x; e < w1; e += blockDim.x) {
        long long gene_local = (long long)(indices[e] - c0);
        double val = apply_pre_transform(data[e], mode_id);
        atomicAdd(&sums[(long long)g * (long long)chunk_size + gene_local], val);
    }
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
//
// Precondition: input MUST be finite. The kernel pads with +INF and sorts on
// the raw IEEE-754 bit pattern (cub::BlockRadixSort), so a NaN — whose bit
// pattern lies above +INF / outside the normal ordering — would land at the
// wrong position and corrupt the downstream U statistic and tie counts.
// Callers must guarantee finite input. This kernel sorts slabs filled from
// EITHER layout, so both host-side staging validators enforce it release-active
// (`GpuError::InvalidShard`): one `shard_validate::validate_shard` covers both
// layouts, at the `ValidationLevel::Ranking` rung the DE routes ask for. Two
// separate validators, with only the CSR one named here, is what review §8.3
// was — the CSC-direct route reached this kernel without ever passing the
// validator its comment cited. The CPU DE entry point rejects
// non-finite input symmetrically.
// ---------------------------------------------------------------------------
// `__launch_bounds__(BLOCK_THREADS)` is REQUIRED, not an optimization hint.
// This kernel is always launched with exactly BLOCK_THREADS (1024) threads/block
// (see `gpu_de_single_tile_block_sort`). The HW caps a block at 65,536 32-bit
// registers on every current arch, so 1024 threads ⇒ ≤64 regs/thread. The PTX is
// built `-arch=compute_70` and JIT-compiled to the runtime arch (e.g. sm_90);
// without the launch bound the JIT has no signal that the block is 1024 threads,
// optimizes `cub::BlockRadixSort<float,1024,8>::Sort()` for lower occupancy, and
// emits >64 regs/thread → the 1024-thread launch is rejected at runtime with
// CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES. The launch bound emits `.maxntid` into the
// PTX so the JIT caps registers at ≤64/thread (spilling to local if needed) on
// every arch. Occupancy is already pinned to 1 block/SM at 1024 threads, so this
// costs nothing. Same applies to `tile_block_radix_sort_kernel` below.
extern "C" __global__ void __launch_bounds__(BLOCK_THREADS)
block_radix_sort_per_gene_kernel(
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
// `__launch_bounds__(BLOCK_THREADS)` required for the same reason as
// `block_radix_sort_per_gene_kernel` — see that kernel's comment. Launched at
// 1024 threads/block by `gpu_de_tile_block_sort`.
extern "C" __global__ void __launch_bounds__(BLOCK_THREADS)
tile_block_radix_sort_kernel(
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
        // Stable A-first merge invariants at the canonical partition:
        //   A[i-1] <= B[j]   (A's tie at i-1 went into A's range — `<=` allows tie)
        //   B[j-1] <  A[i]   (B's tie at j-1 would have been preceded by A — strict)
        //
        // Violation checks (the binary search advances on either):
        //   A[i-1] >  B[j]   → i too large, pull back (consume fewer from A).
        //   B[j-1] >= A[i]   → i too small, push forward (consume more from A).
        //
        // The non-strict `>=` on the second check is what makes the co-rank
        // monotonic in diag when ties span the diagonal. With strict `>`, an
        // identical pair (A[i] == B[j-1]) can produce different `i` values for
        // adjacent diagonals, leading to the same A[i] being consumed by two
        // different threads when this function partitions across a block.
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            // i too large — pull back, take fewer from A.
            i_hi = i;
        } else if (i < m && j > 0 && B[j - 1] >= A[i]) {
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
// Tie-term Σ(c^3 − c) over a single sorted row — single-thread fast path.
//
// One block per gene, one thread active. O(n_per_gene) per gene. The
// block-cooperative kernel below pays a ~10 µs fixed overhead (BlockReduce
// + cross-warp shmem relays + boundary stitch) that dominates the work
// when n_per_gene is small. For n_per_gene < TIE_DISPATCH_THRESHOLD
// (≈ block-radix-sort capacity, single-tile sort regime) the host dispatches
// to this simple kernel; above the threshold the block-cooperative kernel
// wins. See `gpu_de_tie_term` in `scx-gpu/src/gpu_diffexp.rs`.
// ---------------------------------------------------------------------------
extern "C" __global__ void tie_term_sorted_simple_kernel(
    const float*  __restrict__ sorted_slab,
    double*       __restrict__ tie_term,
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
            sum += (double)c * c * c - (double)c;
        }
        i = j;
    }
    tie_term[gene] = sum;
}

// ---------------------------------------------------------------------------
// Tie-term Σ(c^3 − c) over a single sorted row — block-cooperative.
//
// One block per gene; TIE_BLOCK_THREADS threads partition the sorted row
// into contiguous slices and each thread walks its slice in O(n/BT). Each
// thread reports:
//   * head: the first run in its slice (which may continue from the
//           previous thread's tail)
//   * tail: the last run in its slice (which may continue into the next
//           thread's head)
//   * inner_sum: Σ(c³ − c) over runs that fully close inside the slice
//                (neither the head nor the tail).
//
// Thread 0 sequentially stitches the per-thread (head, tail) pairs to
// detect boundary-spanning runs and compute their corrected c³ − c. The
// O(BT) stitch is dwarfed by the O(n/BT) per-thread walk on census-tier
// inputs.
//
// Invariant exploited by the stitch: the input slab is sorted, so within
// any contiguous slice `head.value == tail.value` ⇔ the slice contains
// a single run.
// ---------------------------------------------------------------------------

#define TIE_BLOCK_THREADS 256

extern "C" __global__ void tie_term_sorted_kernel(
    const float*  __restrict__ sorted_slab,  // [chunk_size × n_per_gene]
    double*       __restrict__ tie_term,     // [chunk_size]
    int chunk_size,
    int n_per_gene
) {
    int gene = blockIdx.x;
    if (gene >= chunk_size) return;
    int tid = threadIdx.x;

    if (n_per_gene == 0) {
        if (tid == 0) tie_term[gene] = 0.0;
        return;
    }

    const float* row = sorted_slab + (long long)gene * n_per_gene;

    int per_thread = (n_per_gene + TIE_BLOCK_THREADS - 1) / TIE_BLOCK_THREADS;
    int start = tid * per_thread;
    int end   = start + per_thread;
    if (start > n_per_gene) start = n_per_gene;
    if (end > n_per_gene)   end   = n_per_gene;

    double inner_sum = 0.0;
    float  head_value = 0.0f;
    long long head_count = 0;
    float  tail_value = 0.0f;
    long long tail_count = 0;

    if (start < end) {
        // Head: the first run in this slice.
        head_value = row[start];
        int hi = start;
        while (hi < end && row[hi] == head_value) ++hi;
        head_count = (long long)(hi - start);

        if (hi == end) {
            // Slice is a single run; tail mirrors head.
            tail_value = head_value;
            tail_count = head_count;
        } else {
            // Walk subsequent runs. Each run that fully closes inside the
            // slice contributes c³ − c. The last run found becomes the tail.
            int j = hi;
            float    cur_v = row[j];
            long long cur_c = 0;
            while (j < end) {
                if (row[j] == cur_v) {
                    ++cur_c;
                    ++j;
                } else {
                    // (cur_v, cur_c) closed strictly inside the slice → inner.
                    if (cur_c > 1) inner_sum += (double)cur_c * cur_c * cur_c - (double)cur_c;
                    cur_v = row[j];
                    cur_c = 1;
                    ++j;
                }
            }
            tail_value = cur_v;
            tail_count = cur_c;
        }
    }

    // ---- Boundary stitch ----
    __shared__ float     s_head_value[TIE_BLOCK_THREADS];
    __shared__ long long s_head_count[TIE_BLOCK_THREADS];
    __shared__ float     s_tail_value[TIE_BLOCK_THREADS];
    __shared__ long long s_tail_count[TIE_BLOCK_THREADS];

    s_head_value[tid] = head_value;
    s_head_count[tid] = head_count;
    s_tail_value[tid] = tail_value;
    s_tail_count[tid] = tail_count;

    using Reduce = cub::BlockReduce<double, TIE_BLOCK_THREADS>;
    __shared__ typename Reduce::TempStorage reduce_tmp;
    double block_inner = Reduce(reduce_tmp).Sum(inner_sum);
    __syncthreads();

    if (tid == 0) {
        double boundary_sum = 0.0;

        // Find the first non-empty slice.
        int t = 0;
        while (t < TIE_BLOCK_THREADS && s_head_count[t] == 0) ++t;

        if (t < TIE_BLOCK_THREADS) {
            float    open_value;
            long long open_count;

            if (s_head_value[t] == s_tail_value[t]) {
                // Single-run slice; the open run continues.
                open_value = s_head_value[t];
                open_count = s_head_count[t];
            } else {
                // Multi-run slice: head closes within this slice (it's the
                // first run of the whole row, can't continue from earlier).
                long long c = s_head_count[t];
                if (c > 1) boundary_sum += (double)c * c * c - (double)c;
                open_value = s_tail_value[t];
                open_count = s_tail_count[t];
            }

            for (int u = t + 1; u < TIE_BLOCK_THREADS; ++u) {
                if (s_head_count[u] == 0) continue;
                bool u_single = (s_head_value[u] == s_tail_value[u]);

                if (s_head_value[u] == open_value) {
                    // Open run extends into u's head run.
                    open_count += s_head_count[u];
                    if (!u_single) {
                        // u has more runs after the head → open closes now.
                        long long c = open_count;
                        if (c > 1) boundary_sum += (double)c * c * c - (double)c;
                        open_value = s_tail_value[u];
                        open_count = s_tail_count[u];
                    }
                    // u_single: open keeps growing into u's tail (still open).
                } else {
                    // Open run closes; u's head does not extend it.
                    long long c = open_count;
                    if (c > 1) boundary_sum += (double)c * c * c - (double)c;
                    if (!u_single) {
                        // u's head closes inside u; only u's tail stays open.
                        long long c2 = s_head_count[u];
                        if (c2 > 1) boundary_sum += (double)c2 * c2 * c2 - (double)c2;
                        open_value = s_tail_value[u];
                        open_count = s_tail_count[u];
                    } else {
                        open_value = s_head_value[u];
                        open_count = s_head_count[u];
                    }
                }
            }

            // Close the final open run.
            long long c = open_count;
            if (c > 1) boundary_sum += (double)c * c * c - (double)c;
        }

        tie_term[gene] = block_inner + boundary_sum;
    }
}

// ---------------------------------------------------------------------------
// Combined tie term Σ((r_v + g_v)^3 − (r_v + g_v)) — single-thread fast path.
// Mirrors `tie_term_sorted_simple_kernel`: cheaper than the block-cooperative
// kernel below for small n_ref + n_g (host dispatches by total size).
// ---------------------------------------------------------------------------
extern "C" __global__ void combined_tie_term_simple_kernel(
    const float*  __restrict__ sorted_ref,
    const float*  __restrict__ sorted_group,
    double*       __restrict__ tie_term,
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
            sum += (double)c * c * c - (double)c;
        }
    }
    tie_term[gene] = sum;
}

// ---------------------------------------------------------------------------
// Combined tie term Σ((r_v + g_v)^3 − (r_v + g_v)) over the conceptual merge
// of two pre-sorted rows — block-cooperative via merge-path partitioning.
//
// One block per gene; TIE_BLOCK_THREADS threads partition the combined
// output of length (n_ref + n_g) via `merge_path_co_rank`. Each
// thread merge-walks its sub-range of R and G, reporting:
//   * head: the first merged run in its sub-range
//   * tail: the last merged run in its sub-range
//   * inner_sum: Σ(c³ − c) over runs that fully close inside the slice
// Thread 0 then stitches per-thread (head, tail) pairs the same way as
// `tie_term_sorted_kernel`. The merge of two sorted streams is itself
// sorted, so the "head.value == tail.value ⇒ single-run" invariant holds
// here too.
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
    int tid = threadIdx.x;

    int total = n_ref + n_g;
    if (total == 0) {
        if (tid == 0) tie_term[gene] = 0.0;
        return;
    }

    const float* R = sorted_ref + (long long)gene * n_ref;
    const float* G = sorted_group + (long long)gene * n_g;

    // Partition the conceptual merge across threads.
    int per_thread = (total + TIE_BLOCK_THREADS - 1) / TIE_BLOCK_THREADS;
    int diag_start = tid * per_thread;
    int diag_end   = diag_start + per_thread;
    if (diag_start > total) diag_start = total;
    if (diag_end > total)   diag_end   = total;

    int i_start = merge_path_co_rank(R, n_ref, G, n_g, diag_start);
    int j_start = diag_start - i_start;
    int i_end   = merge_path_co_rank(R, n_ref, G, n_g, diag_end);
    int j_end   = diag_end - i_end;

    // Per-thread merge-walk: emit (head, tail, inner_sum).
    int i = i_start, j = j_start;
    int n_runs = 0;
    float    head_value = 0.0f, last_value = 0.0f;
    long long head_count = 0, last_count = 0;
    double inner_sum = 0.0;

    while (i < i_end || j < j_end) {
        float v;
        if (j >= j_end)       v = R[i];
        else if (i >= i_end)  v = G[j];
        else                  v = fminf(R[i], G[j]);

        long long c = 0;
        while (i < i_end && R[i] == v) { ++i; ++c; }
        while (j < j_end && G[j] == v) { ++j; ++c; }

        if (n_runs == 0) {
            head_value = v;
            head_count = c;
        } else if (n_runs >= 2) {
            // The previous `last` run was displaced by this new run → inner.
            // (When n_runs == 1, `last` is the head, which we don't promote
            // to inner here — head is its own slice-boundary category.)
            if (last_count > 1) inner_sum += (double)last_count * last_count * last_count - (double)last_count;
        }
        last_value = v;
        last_count = c;
        ++n_runs;
    }

    float    tail_value = (n_runs == 0) ? 0.0f : last_value;
    long long tail_count = (n_runs == 0) ? 0 : last_count;

    // ---- Boundary stitch (identical structure to tie_term_sorted_kernel) ----
    __shared__ float     s_head_value[TIE_BLOCK_THREADS];
    __shared__ long long s_head_count[TIE_BLOCK_THREADS];
    __shared__ float     s_tail_value[TIE_BLOCK_THREADS];
    __shared__ long long s_tail_count[TIE_BLOCK_THREADS];

    s_head_value[tid] = head_value;
    s_head_count[tid] = head_count;
    s_tail_value[tid] = tail_value;
    s_tail_count[tid] = tail_count;

    using Reduce = cub::BlockReduce<double, TIE_BLOCK_THREADS>;
    __shared__ typename Reduce::TempStorage reduce_tmp;
    double block_inner = Reduce(reduce_tmp).Sum(inner_sum);
    __syncthreads();

    if (tid == 0) {
        double boundary_sum = 0.0;

        int t = 0;
        while (t < TIE_BLOCK_THREADS && s_head_count[t] == 0) ++t;

        if (t < TIE_BLOCK_THREADS) {
            float    open_value;
            long long open_count;

            if (s_head_value[t] == s_tail_value[t]) {
                open_value = s_head_value[t];
                open_count = s_head_count[t];
            } else {
                long long c = s_head_count[t];
                if (c > 1) boundary_sum += (double)c * c * c - (double)c;
                open_value = s_tail_value[t];
                open_count = s_tail_count[t];
            }

            for (int u = t + 1; u < TIE_BLOCK_THREADS; ++u) {
                if (s_head_count[u] == 0) continue;
                bool u_single = (s_head_value[u] == s_tail_value[u]);

                if (s_head_value[u] == open_value) {
                    open_count += s_head_count[u];
                    if (!u_single) {
                        long long c = open_count;
                        if (c > 1) boundary_sum += (double)c * c * c - (double)c;
                        open_value = s_tail_value[u];
                        open_count = s_tail_count[u];
                    }
                } else {
                    long long c = open_count;
                    if (c > 1) boundary_sum += (double)c * c * c - (double)c;
                    if (!u_single) {
                        long long c2 = s_head_count[u];
                        if (c2 > 1) boundary_sum += (double)c2 * c2 * c2 - (double)c2;
                        open_value = s_tail_value[u];
                        open_count = s_tail_count[u];
                    } else {
                        open_value = s_head_value[u];
                        open_count = s_head_count[u];
                    }
                }
            }

            long long c = open_count;
            if (c > 1) boundary_sum += (double)c * c * c - (double)c;
        }

        tie_term[gene] = block_inner + boundary_sum;
    }
}

// ---------------------------------------------------------------------------
// Batched searchsorted U1 (the group's Mann-Whitney U): for each group value
// x_i, count the ref values it exceeds (+½ per tie):
//   U1 = Σ_i (n_ref_less(x_i) + 0.5 · n_ref_equal(x_i))
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
    SCX_ASSERT_POW2_BLOCK();
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
    SCX_ASSERT_POW2_BLOCK();
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
// Matches `scx-accel::diffexp::wilcoxon_full_from_ranks(.., continuity=true)`:
//   μ      = n1 * n2 / 2
//   σ²     = (n1 * n2 / 12) * ((N + 1) − tc / (N * (N − 1)))   (N = n1+n2)
//   z      = max(|U1 − μ| − 0.5, 0) / σ                         (continuity)
//   p      = erfc(z / √2)                                       (two-sided)
//
// The continuity correction (subtract 0.5 from |U1 − μ|) matches upstream pdex
// / numba_mwu (`use_continuity=True`, scipy's default) and the CPU pdex_ref
// path. This kernel is used ONLY by the pdex_ref GPU sequences; the Wilcoxon
// GPU path computes its p-value elsewhere and stays uncorrected for scanpy
// parity. Clips p to [0, 1] before writing, mirroring pdex_ref's clamp.
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
    // Continuity correction: subtract 0.5 from |U1 − μ| (floored at 0) before
    // standardizing, matching upstream pdex / scipy and the CPU pdex_ref path.
    double dev_abs = fabs(u1 - mu) - 0.5;
    if (dev_abs < 0.0) dev_abs = 0.0;
    double z = dev_abs / sqrt(sigma_sq);
    double p = erfc(z * 0.7071067811865475);  // 1/√2 (z ≥ 0 already)
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    p_values[gene] = p;
}
