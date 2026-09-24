# Batch integration and LISI

> Part of the [SCX + scanpy guide](README.md). Shared conventions are on the
[accelerators overview](accelerators.md).

## Batch integration / Harmony2 (`pyscx.accel.harmony_integrate`)

Clean-room Rust implementation of the Harmony2 algorithm (Korsunsky et
al., 2019): iterative soft k-means clustering with a diversity penalty
over batch covariates, followed by ridge-regression correction of the
PCA embedding. Drop-in replacement for
`scanpy.external.pp.harmony_integrate` — the parameter names
(`key`, `basis`, `adjusted_basis`, `theta`, `lamb`) match, so existing
scanpy pipelines can swap in without other changes.

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="batch")

# PCA first — Harmony corrects the PCA embedding, not the raw matrix.
pyscx.accel.pca(adata, n_comps=30)

# Default (scanpy-compatible): write the corrected embedding to a new
# obsm key "X_pca_harmony" and leave the raw "X_pca" intact.
pyscx.accel.harmony_integrate(adata, "batch")

# Or overwrite the input embedding in place:
pyscx.accel.harmony_integrate(
    adata, "batch", adjusted_basis="X_pca"
)

# Multi-covariate integration (e.g., donor + assay):
pyscx.accel.harmony_integrate(adata, ["donor_id", "assay"])

# Downstream scanpy works on the corrected embedding just like raw PCA:
pyscx.accel.neighbors(adata, use_rep="X_pca_harmony")
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `key` | (required) | `obs` column name, or list of column names, for the batch covariate(s). Each is factorised via `pandas.factorize(sort=False)`. |
| `basis` | `"X_pca"` | `obsm` key holding the input embedding. |
| `adjusted_basis` | `"X_pca_harmony"` | `obsm` key for the corrected embedding. Default writes a **new** key, preserving `basis` (scanpy-compatible). Pass `adjusted_basis=basis` (e.g. `"X_pca"`) to overwrite in place. |
| `n_clusters` | `None` | Soft cluster count K. `None` → `min(N/30, 100)`, clamped to `[2, N/2]`. |
| `theta` | `2.0` | Diversity-penalty strength. Scalar broadcasts to every covariate. |
| `sigma` | `0.1` | Gaussian bandwidth for soft assignments. |
| `lamb` | `None` | Ridge penalty. `None` enables dynamic estimation (`alpha × E[k,b]`). |
| `max_iter` | `10` | Maximum Harmony outer iterations (cluster → correct rounds). |
| `max_iter_kmeans` | `6` | Maximum k-means sub-iterations per Harmony iter (must be ≥ 2×window_size so the convergence check can fire). |
| `random_state` | `0` | RNG seed (`ChaCha8Rng` for determinism across runs). |
| `device` | `"auto"` | `"cpu"` / `"gpu"` / `"auto"`. GPU path requires pyscx built with `--features gpu`. |

Results:

- `adata.obsm[adjusted_basis]` — corrected embedding (N × d, f32); default key
  `"X_pca_harmony"`, leaving `basis` (`"X_pca"`) intact.
- `adata.uns["harmony"]` — dict with `params`, `converged`, `n_iterations`,
  `objective_harmony` (per-iteration objective curve), and `backend`
  (`"scx-accel-cpu"` or `"scx-gpu"`).
- `adata.uns["scx_accel"]["harmony_integrate"]` — the canonical route envelope
  shared with PCA / kNN / UMAP (`route` ∈ `gpu_dense` / `cpu_dense`,
  `fallback_reason`), so you can prove GPU-vs-CPU dispatch the same way as the
  other accelerator ops. See [docs/api/python-accel.md § Accelerator route metadata](../../docs/api/python-accel.md#accelerator-route-metadata).

**Numerical parity.** The clustering primitives are pinned against
**harmonypy 0.2.0** in `scx-accel/src/harmony/harmony_reference_values.rs`,
so `cargo test` gates them with no Python installed: the M-step
(`Y = normalize(Z_cos·Rᵀ)`), the cosine-distance kernel, the ridge
correction against `torch.linalg.inv`, and `update_R`'s softmax half. Two
of the three objective components match; the third is a **documented
divergence** (below).

End-to-end agreement is a *correlation* claim, not a numerical one, and
cannot be otherwise: SCX seeds k-means++ from `rand_chacha` where
harmonypy uses `sklearn.KMeans` and R uses Mersenne Twister, so the runs
start from different cluster geometry. The `accel_harmony` benchmark
gates mean per-PC Pearson r vs harmonypy as an absolute floor
(`benchmarks/comprehensive/thresholds.yaml`).

> The figure previously quoted here — *mean per-PC Pearson r 0.989–0.999
> against R `harmony` v2.x* — was measured **before** the soft k-means
> M-step landed, on `.npz` fixtures under `benchmarks/results/harmony/reference/`
> that are gitignored, so neither CI nor any contributor could reproduce
> it. It is not restated until it is re-measured on the current code. Two
> further corrections: the installed R package is **1.2.4** (the
> *algorithm* is Harmony2 — the version string was wrong), and
> `pyscx/tests/test_harmony_validation.py::test_per_pc_pearson_ge_095` (renamed
> in Phase 7e; it was `..._ge_0998`)
> asserts **0.95** per PC and 0.97 on the mean, not the 0.998 its name
> claims.

**Two documented divergences from harmonypy**, asserted as such rather
than left as unexplained looseness:

* The diversity penalty is `((2E+1)/(O+E+1))^θ` where harmonypy 0.2.0
  uses `(E/(O+E))^θ`. The factor of 2 cancels — it is constant across
  clusters for a fixed cell, so the per-cell L1 normalization removes it —
  but the `+1` smoothing does not.
* The objective's cross-entropy is `log((O+E+1)/(2E+1))` where harmonypy
  uses `log((O+E)/E)`, which puts SCX's term below harmonypy's by
  `log(2)·(2000/N)·Σ σ·O·θ`. Both convergence checks are ratio-based, so
  the two can converge at different sub-iterations.

Both forms are self-consistent and matched across the CPU and GPU arms.

Regenerate the reference tables with
`benchmarks/scripts/generate_harmony_references.py harmony` (and
`… lisi` for LISI), which owns the fixtures and the expected values
together so they cannot drift apart. It drives harmonypy's own
`cluster` / `moe_correct_ridge` / `update_R` / `compute_objective` —
only the driver loop is monkeypatched out, never a formula.

**Scaling** (5M cells × 30 PCs × 100 clusters, single covariate):
scx-accel CPU 37.5 min, scx-accel GPU 31.1 min, harmonypy 22.4 min,
R harmony 80.5 min. Full curves in `benchmarks/results/harmony/REPORT.md`.

## LISI — local batch mixing (`pyscx.accel.compute_lisi`)

Local Inverse Simpson Index (Korsunsky et al., 2019) — per-cell measure
of local categorical diversity. Values approach 1 when a cell's
neighbours share a single label (poor mixing) and approach the number
of categories under uniform mixing (good mixing). Useful as a
batch-integration QC summary: run before and after `harmony_integrate`
and compare the distribution shift.

```python
import pyscx
import numpy as np

# Run on the uncorrected PCA first
lisi_pre = pyscx.accel.compute_lisi(adata, "batch", basis="X_pca")

# Run Harmony, then LISI on the corrected embedding
pyscx.accel.harmony_integrate(adata, "batch", adjusted_basis="X_pca_harmony")
lisi_post = pyscx.accel.compute_lisi(
    adata, "batch", basis="X_pca_harmony"
)

# Integration improves local mixing — mean LISI should rise toward n_batches.
print(f"LISI pre={np.mean(lisi_pre):.2f}  post={np.mean(lisi_post):.2f}")

# Also written to adata.obs:
print(adata.obs["lisi_batch"].describe())
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `key` | (required) | `obs` column with the categorical label to score. |
| `basis` | `"X_pca"` | `obsm` key for the embedding to compute neighbourhoods over. |
| `perplexity` | `30.0` | Gaussian-kernel target perplexity (t-SNE-style bandwidth search). |
| `n_neighbors` | `None` | k for the kNN graph. `None` → `ceil(3 × perplexity) − 1` = **89** at the default perplexity. The `−1` is harmonypy's shape, not a fencepost: `harmonypy.lisi.compute_lisi` retrieves `3 × perplexity` neighbours and then drops column 0, its own self-match, while SCX's sweep skips `j == i` as it collects. Before v0.15 this was `ceil(3 × perplexity)` in three places — the Rust default and both bindings — giving one neighbour more than harmonypy. Pinned in `scx-accel/src/lisi_reference_values.rs`. |
| `approximate_knn` | `False` | Use HNSW approximate kNN instead of the exact O(N²) sweep. ~10× faster at N ≳ 100k, with ~0.01–0.05 mean-LISI drift. |

Returns a `numpy.ndarray` of length N and also writes the values to
`adata.obs[f"lisi_{key}"]`.

By default the implementation uses an exact brute-force kNN (per-row
squared-norm expansion + per-cell top-k heap) and follows the harmonypy /
R `lisi` LISI formulation (raw-distance Gaussian kernel `exp(-D·β)`). On
D1–D4 it is **~10× faster** than R `lisi::compute_lisi`; the previously
reported mean-LISI agreement of 0.8–2.4 % predates the 2026-07 raw-distance
kernel fix and is pending a benchmark recapture.
Brute-force kNN is O(N²·d); above ~50k cells the exact path logs a hint
to set `approximate_knn=True`, which swaps in an HNSW kNN for an
order-of-magnitude speed-up at census scale (D5+) at the cost of small
numerical drift (~0.01–0.05 on mean LISI).

```python
# Census-scale: avoid the O(N²) exact sweep.
lisi = pyscx.accel.compute_lisi(adata, "batch", approximate_knn=True)
```

## See also

- [PCA, kNN, UMAP, and Leiden](accel-embedding-clustering.md) — the PCA
  Harmony corrects, and the graph built on its output.
- [Common scanpy workflows § Batch integration](workflows.md#batch-integration).
