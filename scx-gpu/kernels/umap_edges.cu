// Device-side UMAP edge-list + sampling-schedule preparation (V3 plan Phase 2.4).
//
// Lets `gpu_umap_from_device_graph` build the SGD inputs directly from a
// device-resident fuzzy graph (DeviceFuzzyGraph CSR), so the connectivity
// buffers never round-trip through the host before optimization. Mirrors the
// host helpers `scx_sparse::umap_math::compute_epochs_per_sample` and the CSR→
// edge-list expansion previously done on the CPU in `gpu_umap_native`.

#include <cassert>

// The hand-rolled shared-memory tree reduction in `umap_max_weight_kernel`
// (`for (s = blockDim.x >> 1; s > 0; s >>= 1)`) assumes a power-of-two block
// size; a non-pow2 block drops the odd top partial and silently under-counts
// the max weight. The only launch site hardcodes block=256 (pow2), so the bug
// is latent — this guard turns a future non-pow2 launch into a device-side trap
// instead of a silently wrong sampling schedule (mirrors the #182 fix in
// diffexp.cu / harmony.cu).
#define SCX_ASSERT_POW2_BLOCK() assert((blockDim.x & (blockDim.x - 1)) == 0)

// Expand CSR row pointers into a per-edge head (source) index: thread per row
// writes its row index into every edge slot in [indptr[i], indptr[i+1]).
// The tail (destination) array is the CSR column-index buffer itself.
extern "C" __global__ void umap_expand_head_kernel(
    const long long* __restrict__ indptr, // [n_obs + 1]
    int n_obs,
    int* __restrict__ head)                // out [nnz]
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_obs) return;
    long long start = indptr[i];
    long long end = indptr[i + 1];
    for (long long p = start; p < end; ++p) {
        head[p] = i;
    }
}

// CAS-based float atomic max (handles all finite floats; weights are >= 0 here).
__device__ static inline void atomic_max_f32(float* addr, float val) {
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (cur >= val) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

// Reduce the maximum connectivity weight into out_max[0] (block shared-mem
// reduction + a single float atomicMax per block). out_max must be pre-zeroed
// (weights are non-negative, matching the host `max_weight <= 0` guard).
extern "C" __global__ void umap_max_weight_kernel(
    const float* __restrict__ data, // [nnz]
    int nnz,
    float* __restrict__ out_max)     // [1], pre-zeroed
{
    SCX_ASSERT_POW2_BLOCK();
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int p = blockIdx.x * blockDim.x + tid; p < nnz; p += blockDim.x * gridDim.x) {
        float v = data[p];
        if (v > local) local = v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomic_max_f32(out_max, sdata[0]);
}

// Per-edge UMAP sampling schedule:
//   eps = (w <= 0) ? n_epochs + 1 : n_epochs / max(w / max_w * n_epochs, 1)
// matching `compute_epochs_per_sample`. Also initializes the mutable
// epoch_of_next_sample = eps. When max_w <= 0, every edge is set to never-sample.
extern "C" __global__ void umap_epochs_kernel(
    const float* __restrict__ data, // [nnz] connectivity weights
    const float* __restrict__ max_w, // [1] from umap_max_weight_kernel
    int nnz,
    int n_epochs,
    float* __restrict__ eps,         // out [nnz] epochs_per_sample
    float* __restrict__ eps_next)    // out [nnz] epoch_of_next_sample (= eps)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz) return;
    float mw = max_w[0];
    float ne = (float)n_epochs;
    float w = data[p];
    float e;
    if (mw <= 0.0f || w <= 0.0f) {
        e = ne + 1.0f; // never sample
    } else {
        float n_samples = w / mw * ne;
        if (n_samples < 1.0f) n_samples = 1.0f;
        e = ne / n_samples;
    }
    eps[p] = e;
    eps_next[p] = e;
}
