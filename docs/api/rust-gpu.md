# scx-gpu — GPU Analysis

> Part of the [SCX API reference](README.md). Setup is in [docs/gpu-setup.md](../gpu-setup.md).

GPU-accelerated analysis APIs. Requires CUDA Toolkit ≥ 12.0 at build time.

## cuSPARSE SpMM

### `spmm_csr_view_with_alg(handle, stream, dev, pool, a, b_view, c_view, alpha, beta, alg) → Result<(), GpuError>`
Sparse × dense matrix multiply: C = α·A·B + β·C. A is GPU-resident CSR; B/C are
`DnMatView` / `DnMatViewMut` over dense column-major f32, so a sub-block of a
larger buffer can be read or written without a copy. `pool` reuses the cuSPARSE
workspace across calls (the PCA power loop passes `Some`).

### `spmm_csr_transpose_view_with_alg(...) → Result<(), GpuError>`
Transposed SpMM: C = α·A^T·B + β·C, same view-based signature.

> The contiguous-buffer `spmm_csr` / `spmm_csr_transpose` wrappers were removed —
> every production caller uses the strided-view entry points above.

## cuSOLVER Dense Operations

### `gpu_qr_q(handle, stream, a, m, n) → Result<CudaSlice<f32>, GpuError>`
Economy QR decomposition on GPU: A = Q·R. Returns Q (m × n). Uses `cusolverDnSgeqrf` + `cusolverDnSorgqr`.

## cuRAND

### `random_gaussian_gpu(stream, rows, cols, seed) → Result<CudaSlice<f32>, GpuError>`
Generate a random Gaussian matrix directly on GPU via cuRAND XORWOW generator.

## GPU PCA Pipeline

### `gpu_randomized_pca(dev, reader, n_components, n_oversamples, n_power_iterations, zero_center, seed) → Result<GpuPcaResult, GpuError>`
Complete GPU-accelerated randomized PCA. Streams SpMM shard-by-shard via cuSPARSE, QR via cuSOLVER, SVD via CPU faer, final projection via GPU GEMM. Returns `GpuPcaResult { embeddings, components, variance_explained, variance_ratio, mean }`.

> **Row-major `mean_correct_gpu`** — removed. It had no caller after the GPU PCA
> path moved to the col-major operator; centering now happens inside
> `spmm_forward_segment` via `mean_correct_colmajor_strided_kernel`, which both
> the streaming and the device-resident PCA operators call (the streaming one
> once per shard, the resident one once).

## GPU kNN

### `gpu_knn_cagra_device(dev, embedding, n_neighbors) → Result<DeviceKnnGraph, GpuError>`
Build kNN graph on GPU using NVIDIA CAGRA (cuVS), **device-resident** input and output. Reads a `DeviceEmbedding` (e.g. straight from `gpu_randomized_pca_device`) directly as the CAGRA dataset — no host upload — and returns a `DeviceKnnGraph` on the GPU; call `.to_host(dev)` for the host `GpuKnnResult { indices, distances, n_obs, n_neighbors }`. L2 distance, optimized for PCA embeddings. The fused PCA→kNN pipeline uses this to keep the embedding resident across the handoff. (The host-bounce `gpu_knn_cagra` wrapper and the standalone `scx-accel` `build_knn_graph_gpu` entry point were removed — no production path called them after the rapids-singlecell transition.)

### `cuvs_available() → bool`
Check if `libcuvs.so` is available at runtime.

> **GPU UMAP** — the native CUDA SGD kernel (`gpu_umap_native`) was removed.
> In-VRAM `device="gpu"` UMAP routes to
> rapids-singlecell (`rsc.tl.umap`); backed/lazy inputs fall through to the cuML
> fallback and then CPU SGD.

## GPU Preprocessing

### `gpu_normalize_log1p(dev, csr, target_sum) → Result<(), GpuError>`
Fused per-row normalize_total + log1p on GPU-resident CSR (in-place).

### `gpu_normalize(dev, csr, target_sum) → Result<(), GpuError>`
Per-row normalize_total on GPU-resident CSR (in-place).

### `gpu_log1p(dev, csr) → Result<(), GpuError>`
Element-wise log1p on GPU-resident CSR (in-place).

### `gpu_apply_fused_ops(dev, csr, normalize, log1p, target_sum) → Result<(), GpuError>`
Apply configurable fused preprocessing ops on GPU-resident CSR.

## GpuError Variants

| Variant | Description |
|---------|-------------|
| `CudaError(String)` | CUDA runtime/driver error |
| `KernelLaunchFailed(String)` | Kernel launch failure |
| `GdsUnavailable(String)` | GDS not available |
| `DeviceNotFound(usize)` | GPU device not found |
| `InvalidShard(String)` | Malformed shard data |
| `ShapeMismatch { expected, got }` | Matrix dimension mismatch |
| `CodecError(CodecError)` | Codec decode error |
| `CuSparseError(String)` | cuSPARSE API error |
| `CuSolverError(String)` | cuSOLVER QR/SVD error |
| `CuRandError(String)` | cuRAND generation error |
| `StreamError(String)` | CUDA stream error |
| `OutOfMemory(String)` | GPU OOM |
| `ModuleLoadError(String)` | PTX/CUDA module load failure |
| `CuVsError(String)` | cuVS/CAGRA error |
| `LibraryNotFound(String)` | Runtime library missing (libcuvs.so) |
