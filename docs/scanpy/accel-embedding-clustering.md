# PCA, kNN, UMAP, and Leiden

> Part of the [SCX + scanpy guide](README.md). Shared conventions (`device=`, route metadata, the
compatibility matrix) are on the [accelerators overview](accelerators.md).

## PCA (`pyscx.accel.pca`)

Two methods, auto-routed by the number of variables:

- **Covariance PCA** (CPU: n_vars ≤ 5,000): Builds the covariance matrix
  `X^T @ X` directly from CSR nonzeros via sparse outer product accumulation
  (exploiting symmetry), then eigendecomposes. Faster than randomized SVD for
  HVG-selected data. Parallel accumulation into one shared
  matrix whose columns are partitioned across rayon workers (see
  [Reproducibility](#reproducibility)). CPU-only; the former native GPU covariance path
  (`cusolverDnSsyevd`) was removed in Phase 3.2.

  **It is exact only on well-conditioned input**, and that qualifier is load-bearing
  rather than pedantic. Mean-centering a sparse cross-product means computing
  `Σxy − n·μₓ·μ_y`, a difference of same-order quantities, in every one of the
  matrix's `n_vars²` entries. When the column means are large relative to their
  variances — un-normalized counts, a raw `use_rep`, an uncentered embedding — that
  subtraction loses most of its significant digits and the eigenvalues stop
  meaning anything. Measured on a synthetic f32 fixture with a column mean of
  1e7 and a per-cell variation of 1, `variance_ratio` came back
  `[0.0, 0.0, 0.0]`; at a mean of 1e12, `variance_ratio[0]` came back as **3.72**
  — one component explaining 372% of the total variance.

  Since v0.14.0 the route detects this and says so: `total_var` is computed
  through the same guarded entry point the randomized routes use (so it is no
  longer derived from a sum of round-off-contaminated eigenvalues, and no longer
  collapses to zero), and a `warn`-level log names the condition and points at
  the remedy. The remedy is `method="randomized"`, which decomposes the data
  rather than a differenced cross-product, or normalizing / log-transforming
  first. The eigenvalues themselves cannot be repaired in this route: the
  textbook fix — center each shard before accumulating — has a nonzero term for
  every cell where *both* genes are zero, so on sparse input it is
  `O(n_obs · n_vars²)` and defeats the purpose of the method.
- **Randomized SVD** (CPU: n_vars > 5,000): Streaming shard-by-shard SpMM
  with zero-copy `MatRef::from_row_major_slice` views. Skips intermediate QR
  on transpose results for n_power_iterations ≤ 2 (matching sklearn's default).
- **GPU PCA** routes to `rapids_singlecell` (`rsc.pp.pca`) via
  `to_gpu_anndata()`. The data is handed off as a GPU-resident AnnData with
  `cupyx.scipy.sparse.csr_matrix` X — no host round-trip. The streaming/
  randomized CPU PCA path (>VRAM datasets) survives natively as a fallback.

> [!NOTE]
> **GPU PCA VRAM usage.** SCX preserves sparse CSR when handing `X` to
> rapids, but `rsc.pp.pca()` internally allocates dense working buffers
> (cuBLAS matmul) — peak VRAM during GPU PCA can be substantially higher
> than the sparse `X` footprint alone. Use
> `pyscx.accel.estimate_gpu_memory(adata, operation="pca")` to check
> whether the operation fits before launching. For datasets that exceed
> VRAM, use `backed=True` — the native streaming/randomized PCA path
> processes shards one at a time with bounded VRAM (one shard + working
> matrices). See [gpu-setup.md § GPU memory model](../gpu-setup.md#gpu-memory-model)
> for the full VRAM sizing model.

Both methods work in backed mode without materializing the full matrix.

```python
import pyscx

adata = pyscx.open("atlas.scx").to_anndata(backed=True)
pyscx.accel.pca(adata, n_comps=50)

# Results written to standard scanpy slots:
#   adata.obsm["X_pca"]           — (n_obs × n_comps) float32
#   adata.varm["PCs"]             — (n_vars × n_comps) float32
#   adata.uns["pca"]["variance"]  — explained variance per PC
#   adata.uns["pca"]["variance_ratio"]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_comps` | 50 | Number of principal components |
| `zero_center` | True | Mean-center data (True = standard PCA, False = TruncatedSVD) |
| `random_state` | 0 | Random seed. Seeds the randomized SVD's Ω only — the covariance method draws no randomness. Both are deterministic; see [Reproducibility](#reproducibility) |
| `n_oversamples` | 10 | Extra dimensions for accuracy (randomized SVD only) |
| `n_power_iterations` | 2 | Power iterations for spectral accuracy (randomized SVD only) |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |
| `method` | `"auto"` | `"auto"`, `"covariance"`, or `"randomized"`. `"auto"` routes by `n_vars` (covariance when small, randomized otherwise). Explicit override is useful when benchmarking or when the auto threshold doesn't fit your data. |
| `qr_method` | `"householder"` | CPU randomized-path QR algorithm: `"householder"` (always stable) or `"cholesky"` (CholeskyQR2 — ~3× faster on well-conditioned inputs). **Ignored** by the covariance path and by the GPU rapids path. Non-SPD failures surface as `RuntimeError` with a clear "retry with qr_method='householder'" hint. |

**Key advantage:** On HVG-selected data (2,000 genes), covariance PCA
completes in 4.2s on 1M cells — 5× faster than the previous randomized
SVD and 1.9× faster than scanpy. The method is auto-selected based on
`n_vars`; no user configuration needed. On GPU, PCA routes to
`rapids_singlecell` (`rsc.pp.pca`) which handles method selection
internally. Peak memory on CPU is one shard plus working matrices (plus
~30 MB covariance matrix for 2K genes). Peak VRAM on GPU includes the
sparse `X` plus rapids' internal dense working buffers — use
`pyscx.accel.estimate_gpu_memory(adata, operation="pca")` for pre-flight
sizing (see [gpu-setup.md § GPU memory model](../gpu-setup.md#gpu-memory-model)).

### Reproducibility

**Running the same call twice on the same machine gives bit-identical results**, on both CPU
methods and on in-memory, backed and lazy `X`. Peak memory is also lower than it looks: the
reductions hold one accumulator, not one per thread.

That is worth stating because it was not always true. Through v0.13.0 inclusive, the streaming covariance
build and transpose SpMM accumulated into per-thread buffers whose merge order came from rayon
work-stealing, so five consecutive `method="covariance"` runs produced five different results
— and this table used to claim the covariance method was deterministic. Both reductions now
partition their *output* across workers rather than their input rows, so nothing is merged and
the thread schedule cannot reach the result.

One boundary is worth knowing:

| Change | Same bits? |
|---|---|
| Re-running the same call | ✅ |
| `SCX_ACCEL_PREFETCH_DEPTH` or `SCX_ACCEL_NUM_THREADS` | ✅ |
| A different `RAYON_NUM_THREADS`, or a machine with a different core count | ⚠️ see below |
| A different CPU (different SIMD width) | ❌ |

SCX's own reductions are identical at any thread count. The **dense** decomposition
underneath them — faer's QR and self-adjoint eigendecomposition — blocks its work by the
ambient rayon width, so its low bits move when that changes. This is the same contract
numpy/scipy give, where LAPACK's bits likewise move with `OMP_NUM_THREADS`, and it is why
pinning `RAYON_NUM_THREADS` is the usual advice for cross-machine comparison.

Set `SCX_ACCEL_DETERMINISTIC_LINALG=1` to remove that last dependency: it pins faer to
sequential execution, making PCA's results identical regardless of thread count. It is
opt-in because it is not free — measured at ~2.3× slower on the covariance route's
eigendecomposition, though ~1.65× *faster* on the randomized route's thin QR. Read once, at
the **first CPU PCA / PFlog call**: set it before then.

> [!IMPORTANT]
> **The guarantee is scoped to CPU PCA and PFlog, but the side effect is process-wide.**
> Those are different sets and it matters which you are relying on.
>
> - **Guaranteed reproducible:** CPU PCA and PFlog. Nothing is pinned until one of them
>   runs, and only their determinism is tested.
> - **Slowed but not made reproducible:** every *implicit* faer decomposition in the
>   process, because the setting is one global. After the first pinned PCA call, Harmony's
>   LU fallback, NB-GLM's LLT/LU/QR and the native-GPU PCA's host SVD run sequentially too.
>   If you set this knob, expect those to get slower — a call-order-dependent effect, since
>   before that first PCA call they are unaffected.
> - **Untouched:** call sites that pass faer an explicit `Par` — the exact-kNN gemm and the
>   eval-metrics distance gemm hand it `Par::rayon(0)` directly and ignore the global, so
>   pinning makes them neither sequential nor reproducible.
>
> Do not read this knob as an accelerator-wide reproducibility switch.

Version-to-version bits are not promised. The first release after v0.13.0 changes them once, by
fixing the above — so a result computed with v0.13.0 or earlier will not reproduce exactly on a
later build, and was not reproducible run to run in the first place.

## kNN graph (`pyscx.accel.neighbors`)

Approximate nearest neighbors via HNSW (Hierarchical Navigable Small
World), followed by UMAP-style fuzzy set connectivities.

```python
pyscx.accel.neighbors(adata, n_neighbors=15)

# Results written to standard scanpy slots:
#   adata.obsp["distances"]        — sparse CSR (n_obs × n_obs)
#   adata.obsp["connectivities"]   — sparse CSR (n_obs × n_obs)
#   adata.uns["neighbors"]         — metadata dict
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbors` | 15 | Number of nearest neighbors |
| `use_rep` | `"X_pca"` | Key in `adata.obsm` to use as input |
| `random_state` | 0 | Random seed |
| `ef_construction` | 200 | HNSW build parameter (higher = more accurate) |
| `ef_search` | 200 | HNSW search parameter (higher = more accurate) |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |

On CPU, uses HNSW (instant-distance) with Euclidean distance — except at
`n_obs ≤ 5,000`, where the CPU path silently dispatches to an exact kNN
via a faer matmul + per-row partial top-k sort; `ef_construction` /
`ef_search` are ignored on the exact path. The exact path allocates an
`n_obs × n_obs`
f32 Gram matrix (~100 MB at the threshold) — keep this in mind if
calling at the boundary on memory-constrained hosts.
On GPU, routes to `rapids_singlecell` (`rsc.pp.neighbors`) via
`to_gpu_anndata()`. The standalone native CAGRA dispatch was removed in
Phase 3.3; device-resident CAGRA kNN is retained only within the fused
pipeline path. Benchmarked at 4.4× on 100K cells and 9.4× on 1M cells.

## Fused PCA → kNN (`pyscx.accel.pca_neighbors`)

Runs PCA then the kNN graph in a single call. On a GPU host with
`rapids_singlecell` available, the entire pipeline runs through
`rsc.pp.pca` + `rsc.pp.neighbors` — data stays GPU-resident via
`to_gpu_anndata()`, eliminating the `obsm["X_pca"]` GPU→host→GPU
round-trip that calling `pca` then `neighbors` separately incurs (~240 MB
of host traffic at 1M cells × 60 PCs). Output is identical to the two
sequential calls; it writes every slot they do (`obsm["X_pca"]`,
`varm["PCs"]`, `uns["pca"]`, `obsp["distances"]`,
`obsp["connectivities"]`, `uns["neighbors"]`).

```python
# One call instead of pca(...) + neighbors(...).
pyscx.accel.pca_neighbors(adata, n_comps=50, n_neighbors=15, device="gpu")

# When the rapids fused path runs, all three are stamped "rapids_singlecell_gpu":
#   adata.uns["scx_accel"]["pca"]["route"]
#   adata.uns["scx_accel"]["neighbors"]["route"]
#   adata.uns["scx_accel"]["pca_neighbors"]["route"]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_comps` | 50 | Number of principal components |
| `n_neighbors` | 15 | Number of nearest neighbors |
| `zero_center` | `True` | Mean-center before PCA |
| `random_state` | 0 | Random seed |
| `n_oversamples` / `n_power_iterations` | 10 / 2 | Randomized-PCA accuracy knobs |
| `method` | `"auto"` | PCA method: `"auto"`, `"covariance"`, `"randomized"` |
| `qr_method` | `"householder"` | CPU randomized-PCA QR: `"householder"` or `"cholesky"` (ignored on GPU rapids path) |
| `use_rep` | `"X_pca"` | obsm key the `neighbors` step reads. A non-default value always runs the sequential path — the fused path runs kNN on the freshly-computed PCA embedding, so honoring `obsm[use_rep]` requires the standalone `neighbors`. |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |
| `prefer_format` | `"csr"` | Only `"csr"` is supported (PCA's SpMM path is row-major) |

Fallback: when `rapids_singlecell` is not importable — or the GPU is
unavailable — `pca_neighbors` transparently runs the standalone `pca` then
`neighbors` (each with its own normal routing and warnings), and the
`pca_neighbors` route records the fallback reason as `no_rapids`
(e.g. `cpu_csr`). The fuzzy-graph (connectivity) step runs on the CPU in
the fallback path. Accepts the same `X` inputs as `pca` (backed SCX, lazy
transform, or a materialized scipy/dense matrix).

## Fused PCA → kNN → UMAP (`pyscx.accel.pca_neighbors_umap`)

Extends the fused path through the embedding. On a GPU host with
`rapids_singlecell` available, the full pipeline runs through
`rsc.pp.pca` → `rsc.pp.neighbors` → `rsc.tl.umap` — data stays
GPU-resident via `to_gpu_anndata()` with no host round-trip. The native
CUDA SGD UMAP kernel, fuzzy simplicial set kernel, and device-resident
CAGRA kNN were removed in Phase 3 (3.1, 3.5, 3.3 respectively); rapids
is now the sole GPU path. Writes every slot `pca` + `neighbors` + `umap`
do, including `obsm["X_umap"]`.

```python
# One call instead of pca(...) + neighbors(...) + umap(...).
pyscx.accel.pca_neighbors_umap(adata, n_comps=50, n_neighbors=15,
                               n_components=2, device="gpu")

# When the rapids fused path runs, all four are stamped "rapids_singlecell_gpu":
#   adata.uns["scx_accel"]["pca"|"neighbors"|"umap"|"pca_neighbors_umap"]["route"]
```

Takes the [`pca_neighbors`](#fused-pca--knn-pyscxaccelpca_neighbors) parameters
plus the UMAP knobs: `n_components` (output dims, default 2), `n_epochs`
(default 200), `min_dist` (default 0.1), `spread` (default 1.0),
`negative_sample_rate` (default 5), `umap_learning_rate` (default 1.0).

Fallback: when `rapids_singlecell` is not importable (or no GPU), it
transparently runs the standalone `pca` → `neighbors` → `umap` on CPU (each
with its own routing/warnings) and the `pca_neighbors_umap` route records the
fallback reason as `no_rapids`. GPU UMAP via rapids is non-deterministic, so
the embedding differs run-to-run but preserves cluster structure — pin
`device="cpu"` for reproducible coordinates.

## UMAP (`pyscx.accel.umap`)

UMAP embedding with spectral initialization. On CPU, uses an SGD-based
implementation with negative sampling. On GPU, routes to
`rapids_singlecell` (`rsc.tl.umap`). Takes the kNN connectivity graph as
input.

```python
pyscx.accel.umap(adata)

# Result written to:
#   adata.obsm["X_umap"]  — (n_obs × 2) float32
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 2 | Output dimensions |
| `n_epochs` | 200 | SGD epochs (more = better quality, slower) |
| `min_dist` | 0.1 | Minimum distance in embedding |
| `spread` | 1.0 | Spread of embedded points |
| `negative_sample_rate` | 5 | Negative samples per positive edge |
| `learning_rate` | 1.0 | Initial learning rate |
| `random_state` | 0 | Random seed |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |

On GPU, routes to `rapids_singlecell` (`rsc.tl.umap`) via
`to_gpu_anndata()`. The native CUDA SGD UMAP kernel was removed in
Phase 3.1. Falls back to CPU if `rapids_singlecell` is not importable
(fallback reason: `no_rapids`).

## Leiden clustering (`pyscx.accel.leiden`)

Rust-native implementation of the Leiden algorithm (Traag, Waltman & van
Eck, 2019) with the Reichardt-Bornholdt (RB) configuration model quality
function on the CPU path; cuGraph on the GPU path. Operates directly on
the kNN connectivities CSR — no Python `igraph` / `leidenalg` dependency
required.

```python
pyscx.accel.leiden(adata, resolution=1.0)

# Results written to:
#   adata.obs["leiden"]              — categorical community labels
#   adata.uns["leiden"]["params"]    — resolution, random_state, device,
#                                     parallel, theta (cugraph), gpu_id
#                                     (cugraph), and `ignored` list of
#                                     kwargs the chosen backend dropped
#   adata.uns["leiden"]["backend"]   — "scx-accel" or "cugraph"
#   adata.uns["leiden"]["modularity"] — normalized generalized RB modularity,
#                                     `quality / 2m`. Comparable across graphs
#                                     and across the two backends. At the
#                                     default resolution=1.0 this IS Newman
#                                     modularity, bounded by [-0.5, 1]; at
#                                     resolution != 1 the γ term does not
#                                     cancel and it is NOT so bounded (on
#                                     Zachary's graph, -0.996 at γ=20 and
#                                     -2.490 at γ=50, where leidenalg's own
#                                     `modularity` reports -0.0498).
#   adata.uns["leiden"]["quality"]   — the raw, un-normalized RB objective
#                                     Σ_c [2·w_in(c) − γ·k_c²/2m], i.e.
#                                     `modularity × 2m`. Scales with total
#                                     edge weight, so it is comparable only
#                                     between partitions of the *same* graph.
#                                     `None` on the cuGraph backend, which
#                                     does not expose it.
```

> **Changed in 0.20.** `uns["leiden"]["modularity"]` used to carry the *raw* RB
> quality on the CPU backend — leidenalg's internal `quality()`, which returns
> ~10⁶ on a 1M-edge graph — while the cuGraph backend wrote a genuinely
> normalized value into the same key. Any threshold or cross-dataset comparison
> on the old CPU number was meaningless. It is now `quality / 2m` on both
> backends; the old value is still available as `uns["leiden"]["quality"]`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | 1.0 | Resolution parameter γ — higher values yield more communities |
| `key_added` | `"leiden"` | Key in `adata.obs` for community labels |
| `random_state` | 0 | Random seed for reproducibility |
| `n_iterations` | 2 | **Unit differs by backend.** Rust-native (CPU): leidenalg-style outer iterations (default 2 is plenty — each is a full multilevel cycle). cuGraph (GPU): maps to cuGraph's `max_iter` (a *coarsening-pass* count). The leidenalg default of 2 would starve cuGraph's coarsening and produce a degenerate, over-partitioned result, so the cuGraph path uses cuGraph's own default of **100** whenever `n_iterations <= 2` (including the `-1`/`0` convergence sentinels); only values `> 2` are forwarded verbatim. The effective cap is recorded in `uns["leiden"]["params"]["max_iter"]`. |
| `parallel` | `False` | Run the **Rust-native** Leiden in conflict-free batched mode. `False` (default) reproduces C++ leidenalg's sequential move-node *ordering* (the refinement omits the paper's well-connectedness admissibility conditions). **Ignored on the cuGraph path** (warns when `True`). |
| `device` | `"auto"` | `"auto"` (cuGraph if available, else Rust-native), `"cpu"` (Rust-native), `"gpu"` / `"gpu:N"` (cuGraph on CUDA device 0 or N — `gpu:N` pins via `cupy.cuda.Device(N)`). |
| `theta` | 1.0 | cuGraph-only resolution scaling knob (forwarded to `cugraph.leiden(theta=...)`). **Ignored on the Rust-native path** (warns when non-default). |

**Dispatch (post-spec):** two backends as peers, selected by `device`:

* `device="cpu"` → Rust-native (`scx_accel::leiden`). ARI ≈ 0.97 vs
  leidenalg on pbmc3k. Always available.
* `device="gpu"` → cuGraph. ARI ≈ 0.92 vs leidenalg, by design (different
  refinement strategy). Hard error if cuGraph is missing — no fallback.
* `device="auto"` (default) → cuGraph when a CUDA device is visible and
  `cugraph` imports cleanly, else Rust-native. Matches the rest of
  `pyscx.accel.*`.

**Migration note (vs the pre-spec dispatcher):** `device="auto"` previously
ran Rust-native first regardless of host. After the spec it runs cuGraph
on GPU hosts where cuGraph is installed, which produces a different
partition (ARI 0.97 → 0.92 vs leidenalg). Pin `device="cpu"` to preserve
the old behavior — required when downstream DE / annotation transfer /
UMAP coloring is keyed on specific cluster IDs from previous runs. The
Python `leidenalg` fallback has been deleted; callers who want it run
`scanpy.tl.leiden(flavor="leidenalg")` directly.

Benchmarked at 55s on 1M cells (**40× faster** than Python leidenalg's 2,226s
in same-conditions comparison) on the Rust-native path; ~3.5s on the cuGraph
path. ARI 0.92 vs Python leidenalg on census_1m for the cuGraph path. The
two backends converge to different local optima — both produce valid
high-quality community structures. Compare via ARI or NMI when switching
backends.

## See also

- [GPU acceleration](accel-gpu.md) — GPU vs CPU numerical differences for
  PCA, kNN, UMAP, and Leiden.
- [Batch integration and LISI](accel-integration.md) — Harmony on the PCA
  embedding.
