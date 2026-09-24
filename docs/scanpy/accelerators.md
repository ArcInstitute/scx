# Rust-native accelerators

> Part of the [SCX + scanpy guide](README.md).

SCX includes optional Rust-native implementations of PCA, kNN graph
construction, UMAP embedding, Leiden clustering, differential expression, and
the perturbation-evaluation metrics (`perturbation_metrics`, `energy_distance`)
via `pyscx.accel`. These accelerators are 2–40× faster than their scanpy
equivalents at scale (>100K cells) while writing results to the same AnnData
slots — so downstream scanpy functions (plotting, etc.) work identically.

All accelerators that support GPU expose a `device` parameter:
- `device="auto"` (default) — use GPU if available, fall back to CPU
- `device="cpu"` — force CPU
- `device="gpu"` — force GPU (raises error if unavailable)
- `device="gpu:1"` — select a specific GPU on multi-GPU systems

`auto` resolves availability **once, before the op starts**. A GPU that is
present but then fails while running — out of memory, a driver fault — raises;
`auto` does not silently re-run the op on CPU, because a CPU re-run at atlas
scale is hours of work you did not ask for. The error names the shortfall and
the remedy. See [docs/gpu-setup.md § What `device="auto"` does and does not
do](../gpu-setup.md#what-deviceauto-does-and-does-not-do).

> **Seeing the route at op start / diagnosing a slow GPU op.** Each accelerator
> logs its resolved route the moment it starts, at INFO — enable it with
> `import logging; logging.basicConfig(level=logging.INFO)` (or
> `logging.getLogger("pyscx.accel").setLevel(logging.INFO)`). On **backed /
> atlas-scale** input the streaming GPU path (`highly_variable_genes`,
> streaming PCA) is **CPU-decode-bound** — the GPU can sit near 0% util while
> shards decode; this is expected, not a hang. An explicit `device="gpu"` request
> that silently lands on CPU emits a `UserWarning` naming the `fallback_reason`.
> See the "slow GPU op / is it hung?" entry in [docs/gpu-setup.md § Troubleshooting](../gpu-setup.md#troubleshooting).

> **GPU is fastest only when the input layout matches the op.** For
> `pdex_ref` the column-major CSC-direct GPU route is the high-performance
> path, and it requires a *backed* SCX file with a CSC sidecar
> (built by default under `csc="auto"` on qualifying datasets, or explicitly via `csc="always"` / `scx build-csc`); v3
> CSC-direct is the default GPU DE route. In-memory scipy CSR can run on GPU but
> may be slower than CPU. (`rank_genes_groups` / Wilcoxon rank-sum shares the same
> CSC-direct (`gpu_csc_v3`) and CSR-direct (`gpu_csr_v3`) routes as `pdex_ref`;
> dense-host input is densified to CSR and also records `gpu_csr_v3`.) When comparing performance,
> **check the recorded route** at
> `adata.uns["scx_accel"][<op>]["route"]` (e.g. `gpu_csc_v3` vs `gpu_csr_v3`) —
> every DE call records which route it actually took and the `fallback_reason`
> if it didn't take the ideal one. See
> [docs/api/python-accel.md § Accelerator route metadata](../api/python-accel.md#accelerator-route-metadata).

## Accelerator pages

| Page | Ops |
|------|-----|
| [PCA, kNN, UMAP, and Leiden](accel-embedding-clustering.md) | `pca`, `neighbors`, `pca_neighbors`, `pca_neighbors_umap`, `umap`, `leiden` |
| [Batch integration and LISI](accel-integration.md) | `harmony_integrate`, `compute_lisi` |
| [Gene-set scoring and PFlog](accel-scoring-normalization.md) | `score_genes`, `pflog` |
| [Differential expression](accel-differential-expression.md) | `rank_genes_groups`, `pdex_ref`, `pseudobulk_dex`, stratified DE, `nb_glm` / `pdex_nb_glm` |
| [Perturbation evaluation metrics](accel-perturbation-metrics.md) | `pseudobulk_means`, `perturbation_metrics`, `discrimination_score`, `energy_distance`, `knockdown_efficiency`, `clustering_agreement`, `rank_genes_groups_df` |
| [Column-major (CSC) dispatch](accel-csc.md) | `prefer_format=` on the ops that accept it |
| [GPU acceleration](accel-gpu.md) | GPU routing, data layout, numerical differences, tolerances, backend checks |
| [Multithreading](threading.md) | Threading and GPU DE device residency |

Preprocessing accelerators (`filter_cells`, `filter_genes`, `normalize_total`,
`log1p`, `highly_variable_genes`, `calculate_qc_metrics`) are covered with
backed mode — see [scanpy operations in backed mode](backed-mode.md#scanpy-operations-in-backed-mode)
and [Lazy preprocessing](lazy-preprocessing.md).

## Axis-subsetting ops and the aligned members

`filter_cells`, `filter_genes`, `subset_obs`, `subset_var`, and
`highly_variable_genes(subset=True)` subset one axis of the AnnData in place.
**anndata performs the subset** — pyscx only makes a backed `X` subsettable and
keeps its lazy mappings off disk — so `obs`, `var`, `uns`, `raw`, unused
categorical levels and every aligned member (`layers`, `obsm`, `obsp` on the obs
axis; `layers`, `varm`, `varp` on the var axis) behave exactly as on an in-memory
AnnData, and a failure part-way leaves the object untouched.

What stays SCX-specific is what you would lose otherwise: the matrix is never
materialized (`type(adata.X)` is unchanged by a filter), and a backed `obsp` /
`varp` / `varm` is never pulled off disk — the subset is recorded and applied on
first read, so `filter_cells` on a file carrying a kNN graph costs nothing extra
unless you read `obsp`.

Plain anndata indexing works on a backed `X` too: `adata[:, mask]` is a lazy view,
`adata[mask].copy()` subsets and materializes, `adata[mask].to_memory()` gives a
fully in-memory AnnData. Full table in
[docs/api/python-accel.md § Axis subsetting and aligned members](../api/python-accel.md#axis-subsetting-and-aligned-members).

Handing that view to an accelerator works as well, but it is not free: any
`pyscx.accel.*` op that writes results back **rebuilds the view in place as a
regular AnnData first**, keeping `X` lazy, and warns
(`ImplicitModificationWarning`) that it did. Afterwards the object no longer
tracks its parent and the results land on it, not on the parent. Nothing is
copied — that is the point of the rebuild, since anndata's own copy-on-write
would get to the same place by materializing the matrix. The rebuild happens on
entry, so `is_view` flips to `False` even if the op then raises. So the
canonical scanpy ordering is safe out-of-core:

```python
adata = pyscx.open("atlas.scx").to_anndata(backed=True)
pyscx.accel.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
pyscx.accel.subset_var(adata, adata.var["highly_variable"].values)  # no view at all
pyscx.accel.normalize_total(adata, target_sum=1e4)
pyscx.accel.log1p(adata)
pyscx.accel.pca(adata, n_comps=50)
```

`subset_var` / `subset_obs` are preferred over `adata[:, mask]` here because
they subset in place and never produce a view, so there is no rebuild and no
warning. Both keep `X` lazy; only the plain-indexing route has the detachment
to explain.

On a **lazy** `X` with an active column projection, `filter_cells` thresholds the
visible-gene totals — the same numbers `adata.obs["total_counts"]` and
`adata.X.sum(axis=1)` report, and what scanpy would compute on the sliced object.

## Compatibility matrix

| Op                       | CPU | GPU | Scanpy-parity kwargs                                                  | scx-only kwargs                                |
|--------------------------|:---:|:---:|-----------------------------------------------------------------------|------------------------------------------------|
| `normalize_total`        | ✓   | ✓   | `target_sum`                                                          | `device`                                       |
| `log1p`                  | ✓   | ✓   | —                                                                     | `device`                                       |
| `filter_cells`           | ✓   | —   | `min_genes`, `max_genes`, `min_counts`, `max_counts`                  | —                                              |
| `filter_genes`           | ✓   | —   | `min_cells`, `max_cells`, `min_counts`, `max_counts`                  | —                                              |
| `subset_obs`             | ✓   | —   | — (`adata[mask].copy()`, without materializing)                       | `mask_or_indices`                              |
| `subset_var`             | ✓   | —   | — (`adata[:, mask].copy()`, without materializing)                    | `mask_or_indices`                              |
| `calculate_qc_metrics`   | ✓   | —   | `qc_vars`, `log1p`, `inplace`, `layer`, `percent_top`                 | `prefer_format`                                |
| `highly_variable_genes`  | ✓   | ✓   | `n_top_genes`, `flavor`, `batch_key`, `span`, `subset`, `n_bins`, `layer` | `device`, `prefer_format`                  |
| `score_genes`            | ✓   | —   | `gene_list`, `ctrl_size`, `gene_pool`, `n_bins`, `score_name`, `random_state` | `method`, `layer`, `device`, `ctrl_genes` |
| `pflog`                 | ✓   | —   | — (no scanpy equivalent)                                             | `alpha`, `store`, `n_components`, `store_repr`, `out`, `shard_size`, `layer`, `device` |
| `pca`                    | ✓   | ✓   | `n_comps`, `zero_center`, `random_state`                              | `device`, `method`, `qr_method`, `prefer_format`, `allow_tf32`, `n_oversamples`, `n_power_iterations`, `spmm_policy`, `memory_budget` |
| `neighbors`              | ✓   | ✓   | `n_neighbors`, `use_rep`, `random_state`                              | `device`, `ef_construction`, `ef_search`       |
| `pca_neighbors`          | ✓   | ✓   | (PCA + neighbors kwargs, see below)                                   | `device`, `method`, `qr_method`, `prefer_format` |
| `pca_neighbors_umap`     | ✓   | ✓   | (PCA + neighbors + UMAP kwargs, see below)                            | `device`, `method`, `qr_method`, `prefer_format` |
| `umap`                   | ✓   | ✓   | `n_components`, `n_epochs`, `min_dist`, `spread`, `learning_rate`, `random_state` | `device`, `negative_sample_rate`   |
| `leiden`                 | ✓   | ✓¹  | `resolution`, `key_added`, `random_state`, `n_iterations`             | `device`, `parallel`, `theta`                  |
| `harmony_integrate`      | ✓   | —   | `key`, `basis`, `theta`, `sigma`, `lamb`, `max_iter`                  | `adjusted_basis`, `block_size`, `n_clusters`, `alpha`, `max_iter_kmeans`, `random_state`, `device` |
| `compute_lisi`           | ✓   | —   | `key`, `basis`, `perplexity`, `n_neighbors`, `approximate_knn`        | —                                              |
| `rank_genes_groups`      | ✓   | ✓   | `groupby`, `groups`, `reference`, `n_genes`, `method`, `pts`, `corr_method` (BH only), `rankby_abs`, `tie_correct`, `use_raw`, `layer` | `gene_chunk_size`, `stratify_by`, `prefer_format`, `device` |
| `pdex_ref`               | ✓   | ✓   | `groupby`, `reference`                                                | `groups`, `is_log1p`, `geometric_mean`, `epsilon`, `cpm_filter`, `gene_chunk_size`, `prefer_format`, `device`, `output`, `use_raw`, `layer` |
| `pseudobulk_dex`         | ✓   | —   | `design`, `reference` (**`groupby` is spelled the same but means the opposite** — sample-defining columns, not the compared one; `str \| list[str]`) | `test_col`, `sample_cols`/`sample_key` (aliases for `groupby`), `aggr_method`, `stratify_by`, `prefer_format`, `backend`, `nbglm_options`, `gene_indices`, `n_cpus` |
| `nb_glm`                 | ✓   | —   | — (no scanpy equivalent)                                             | `counts`, `design`, `contrast`                 |
| `pdex_nb_glm`            | ✓   | —   | — (no scanpy equivalent)                                             | `groupby`, `reference`, `stratify_by`          |
| `pseudobulk_means`       | ✓   | ✓   | — (no scanpy equivalent)                                             | `groupby` (`str \| list[str]`), `min_cells_per_group`, `device` |
| `perturbation_metrics`   | ✓   | ✓   | — (cell-eval metric)                                                 | `pert_col`, `control`, `metrics`, `min_cells_per_group`, `device` |
| `energy_distance`        | ✓   | ✓²  | — (cell-eval metric)                                                 | `pert_col`, `control`, `metric`, `embed_key`, `backend`, `dtype`, `device` |
| `discrimination_score`   | ✓   | —   | — (cell-eval metric)                                                 | `pert_col`, `control`, `metric`, `exclude_target_gene`, `embed_key` |

¹ GPU Leiden has a documented label-stability divergence vs `leidenalg` —
pin `device="cpu"` to preserve label stability for downstream DE / annotation
transfer. See `CLAUDE.md § Known Limitations`.

² GPU `energy_distance` covers **euclidean + cosine** at `dtype="f32"` (gemm
decomposition). `metric="l1"` and `dtype="f64"` stay on CPU even under
`device="gpu"` (route `cpu_csr`, no error). `discrimination_score` has no GPU
kernel yet — it needs exact-rank parity that f32 gemm can't guarantee, and is
already fast on the small `[P×G]` effect matrix; `device` is accepted for
symmetry but always runs CPU.

**`calculate_qc_metrics` head-to-head.** Since 0.18 one native kernel handles
every kind of `X`, and on the standard opening sequence —
`calculate_qc_metrics(qc_vars=["mt","ribo"])` → `filter_cells` →
`filter_genes`, with `percent_top` passed explicitly on both sides — a backed
SCX handle runs it in **23.2 s / 2.2 GB on 1M cells against scanpy's 52.5 s /
33.4 GB**, 2.3× faster on 15× less resident memory. The two engines agree to
0.0 on every fixture whose per-cell totals stay inside float32's exact-integer
range; past it SCX is the accurate side, because scanpy reduces `X` in the
array's own dtype and SCX accumulates in f64. Tables, smaller fixtures and
provenance:
[docs/performance/accel-qc-de-integration.md § Fused QC + filtering vs scanpy](../performance/accel-qc-de-integration.md#fused-qc--filtering-vs-scanpy-accel_qc_filter).

## Mixing accelerators with scanpy

The accelerators write to the same AnnData slots as scanpy, so they are
fully interchangeable. You can mix and match:

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var["highly_variable"]].copy()

# Use SCX accelerators for compute-heavy steps
pyscx.accel.pca(adata, n_comps=50)             # 5× faster (covariance method for HVGs)
pyscx.accel.neighbors(adata)                    # HNSW kNN
pyscx.accel.umap(adata)                         # faster than sc.tl.umap
pyscx.accel.leiden(adata)                        # 40× faster than leidenalg
pyscx.accel.rank_genes_groups(adata, "leiden")   # 3× faster than sc.tl.rank_genes_groups

# Downstream scanpy works identically
sc.pl.umap(adata, color="leiden")         # uses adata.obsm["X_umap"]
```

Or use scanpy for everything — no changes needed:

```python
sc.pp.pca(adata)        # works fine with SCX data
sc.pp.neighbors(adata)  # uses pynndescent
sc.tl.umap(adata)       # uses umap-learn
```

The choice is purely about performance. At <50K cells, the difference is
negligible. At >100K cells, the Rust accelerators provide meaningful
speedups. At >1M cells, GPU acceleration (`device="gpu"`) transforms
interactive exploration from "go get coffee" to "instant."
