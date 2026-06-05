// Device-resident UMAP fuzzy-simplicial-set construction (V3 plan Phase 2.4).
//
// Builds the symmetrized fuzzy connectivity graph directly on the GPU from a
// raw CAGRA kNN result (DeviceKnnGraph), removing the host round-trip through
// `scx_accel::neighbors::compute_connectivities`.
//
// Pipeline (orchestrated from scx-gpu/src/gpu_fuzzy.rs):
//   1. fuzzy_membership_kernel  — per-row self-filter + sqrt + bandwidth (sigma)
//      search + membership strengths, emitting a forward CSR M (k entries/row,
//      sorted by column).
//   2. cuSPARSE csr2csc(M) -> T = Mᵀ (column-sorted) on the host side.
//   3. fuzzy_union_count_kernel — per-row merge-count |union(M[i], T[i])|.
//   4. fuzzy_union_fill_kernel  — per-row merge writing conn = a + b - a*b.
//
// The fuzzy union c = a + b - a*b (probabilistic t-conorm) is associative and
// commutative, so accumulating it over every entry sharing a column — across
// both M and T, collapsing duplicates — yields exactly the host result:
//   - a real edge present in both directions -> mu_ij ⊕ mu_ji
//   - a one-directional edge               -> the present membership (other = 0)
//   - collapsed self-pads (value 1.0)       -> 1.0 (idempotent under ⊕)
//
// Must match scx_accel::neighbors::compute_connectivities within f32 tolerance.

#include <math.h>

// Maximum supported n_neighbors. Per-row scratch lives in thread-local arrays;
// the host entry point rejects k beyond this bound.
#define FUZZY_MAX_K 256

// Per-row: filter self-hits, sqrt L2² → Euclidean, pad to exactly k, search the
// UMAP bandwidth sigma, compute membership strengths, and emit a column-sorted
// forward CSR row of length k at offset i*k.
extern "C" __global__ void fuzzy_membership_kernel(
    const unsigned int* __restrict__ knn_indices, // [n_obs * search_k] raw CAGRA
    const float* __restrict__ knn_dist_sq,        // [n_obs * search_k] L2 squared
    int n_obs,
    int search_k,
    int k,                                         // n_neighbors (<= FUZZY_MAX_K)
    int* __restrict__ col_m,                       // out [n_obs * k]
    float* __restrict__ val_m)                     // out [n_obs * k]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_obs) return;

    int cols[FUZZY_MAX_K];
    float vals[FUZZY_MAX_K]; // distances first, then overwritten with memberships

    // 1. Self-filter + sqrt + pad to exactly k (mirrors DeviceKnnGraph::to_host).
    int count = 0;
    int base = i * search_k;
    for (int j = 0; j < search_k && count < k; ++j) {
        int nb = (int)knn_indices[base + j];
        if (nb == i) continue; // skip self-hit
        cols[count] = nb;
        float dsq = knn_dist_sq[base + j];
        vals[count] = sqrtf(dsq < 0.0f ? 0.0f : dsq);
        ++count;
    }
    while (count < k) { // pad short rows with (self, 0.0)
        cols[count] = i;
        vals[count] = 0.0f;
        ++count;
    }

    // 2. Bandwidth search: solve sum_j exp(-max(d_j - rho, 0) / sigma) = log2(k).
    float rho = vals[0];
    float target = log2f((float)k);
    float lo = 1e-10f, hi = 1000.0f;
    for (int it = 0; it < 50; ++it) {
        float mid = 0.5f * (lo + hi);
        float sum = 0.0f;
        for (int t = 0; t < k; ++t) {
            float adj = vals[t] - rho;
            if (adj < 0.0f) adj = 0.0f;
            sum += expf(-adj / mid);
        }
        if (sum > target) hi = mid; else lo = mid;
    }
    float sigma = 0.5f * (lo + hi);

    // 3. Membership strengths (overwrite vals in place).
    for (int t = 0; t < k; ++t) {
        float d = vals[t];
        vals[t] = (d <= rho || sigma <= 1e-10f) ? 1.0f : expf(-(d - rho) / sigma);
    }

    // 4. Insertion sort (cols, vals) ascending by column (k is small).
    for (int a = 1; a < k; ++a) {
        int ck = cols[a];
        float vk = vals[a];
        int b = a - 1;
        while (b >= 0 && cols[b] > ck) {
            cols[b + 1] = cols[b];
            vals[b + 1] = vals[b];
            --b;
        }
        cols[b + 1] = ck;
        vals[b + 1] = vk;
    }

    // 5. Emit forward CSR row (regular layout: k entries at offset i*k).
    int out = i * k;
    for (int t = 0; t < k; ++t) {
        col_m[out + t] = cols[t];
        val_m[out + t] = vals[t];
    }
}

// Per-row merge of the column-sorted forward row M[i] and transpose row T[i],
// counting the number of union columns with a positive fuzzy-union value.
extern "C" __global__ void fuzzy_union_count_kernel(
    const int* __restrict__ col_m,    // [n_obs * k] forward columns
    const float* __restrict__ val_m,  // [n_obs * k] forward memberships
    const int* __restrict__ t_indptr, // [n_obs + 1] transpose row pointers (i32)
    const int* __restrict__ t_col,    // [nnz] transpose columns
    const float* __restrict__ t_val,  // [nnz] transpose memberships
    int n_obs,
    int k,
    int* __restrict__ row_nnz)        // out [n_obs]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_obs) return;

    int mp = i * k, me = i * k + k;
    int tp = t_indptr[i], te = t_indptr[i + 1];
    int cnt = 0;

    while (mp < me || tp < te) {
        int cm = (mp < me) ? col_m[mp] : n_obs;
        int ct = (tp < te) ? t_col[tp] : n_obs;
        int c = (cm < ct) ? cm : ct;
        float acc = 0.0f;
        while (mp < me && col_m[mp] == c) { float v = val_m[mp]; acc = acc + v - acc * v; ++mp; }
        while (tp < te && t_col[tp] == c) { float v = t_val[tp]; acc = acc + v - acc * v; ++tp; }
        if (acc > 0.0f) ++cnt;
    }
    row_nnz[i] = cnt;
}

// Per-row merge writing the symmetric CSR row: conn = a + b - a*b, dropping any
// column whose accumulated value is 0 (matches the host conn > 0 filter).
extern "C" __global__ void fuzzy_union_fill_kernel(
    const int* __restrict__ col_m,
    const float* __restrict__ val_m,
    const int* __restrict__ t_indptr,
    const int* __restrict__ t_col,
    const float* __restrict__ t_val,
    const long long* __restrict__ out_indptr, // [n_obs + 1] i64 row pointers
    int n_obs,
    int k,
    int* __restrict__ out_indices,            // [total_nnz]
    float* __restrict__ out_data)             // [total_nnz]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_obs) return;

    int mp = i * k, me = i * k + k;
    int tp = t_indptr[i], te = t_indptr[i + 1];
    long long w = out_indptr[i];

    while (mp < me || tp < te) {
        int cm = (mp < me) ? col_m[mp] : n_obs;
        int ct = (tp < te) ? t_col[tp] : n_obs;
        int c = (cm < ct) ? cm : ct;
        float acc = 0.0f;
        while (mp < me && col_m[mp] == c) { float v = val_m[mp]; acc = acc + v - acc * v; ++mp; }
        while (tp < te && t_col[tp] == c) { float v = t_val[tp]; acc = acc + v - acc * v; ++tp; }
        if (acc > 0.0f) {
            out_indices[w] = c;
            out_data[w] = acc;
            ++w;
        }
    }
}
