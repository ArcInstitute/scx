# Processing reference (pyscx)

Self-contained reference for processing single-cell data with pyscx. There are
three approaches; pick one based on dataset size and what you need out.

## Choosing the approach

```
                  Fits in memory (~500K cells)?
                   ┌───── yes ─────┴───── no ─────┐
              In-memory                  Need the full dataset?
            (sc.pp.* / sc.tl.*)      ┌─── yes ───┴─── no ───┐
                              Backed + lazy            Query pipeline
                            (pyscx.accel.*)         (extract a subset)
```

| | In-memory | Backed + lazy | Query pipeline |
|--|-----------|---------------|----------------|
| API | `to_anndata()` + `sc.pp.*` | `to_anndata(backed=True)` + `pyscx.accel.*` | `.query().filter_obs().collect()` |
| Peak memory | full matrix | ~1 shard (~128 MB) | subset (materialized) |
| scanpy compat | full | partial (use `accel.*` for preprocessing) | full (result is a regular AnnData) |
| GPU accel | `accel.*(device="gpu")` | `accel.*(device="gpu")` | preprocess after `.to_anndata()` |
| Predicate pushdown | no | no (filters via deletion vectors) | yes (skips non-matching shards) |
| Lazy normalize/log1p | no | yes (`accel.normalize_total/log1p`) | yes (`with_normalize/with_log1p`) |
| Streaming PCA | no | yes (`accel.pca` streams transforms) | no (after materialize) |
| Write-back | in-place on X | read-only (`pyscx.preprocess` for copy) | read-only |
| Typical size | < 500K cells | 500K–10M+ | any (output is a subset) |

In-memory CSR footprint ≈ `8·n_obs + 8·nnz` bytes (i32 indices + f32 data, plus
small indptr). E.g. 1M cells × 30K genes @ 5% density (1.5B nnz) ≈ 12 GB. SCX's
advantage is on-disk size; the in-memory AnnData is identical to one loaded from
h5ad. **Hybrid:** query a subset, then go in-memory with standard scanpy.

## PyExperiment (returned by `pyscx.open(path)`)

- `to_anndata(backed=False, cache_shards=4, var_names=None, obs_filter=None, layers=None, preserve_slots=False, modality=None, eager=False, memory_budget=None, obsm=None)` — convert to AnnData.
  - `var_names`: gene-name list to project (column subset).
  - `obs_filter`: predicate string. **Non-backed mode** routes through the query engine (shard pushdown). **Backed mode** evaluates it with **pandas `.query()`** (richer grammar, no pushdown) and folds the matches into the dataset's row set — so the same expression can resolve via different engines depending on `backed`.
  - `layers`: layer names to load (default all).
  - `obsm`: obsm keys to load (default `None` = all; byte-identical to before). `obsm=[...]` loads only the listed embeddings (unknown key → `KeyError`; `[]` = none) **and** switches the obsm materialisation mode (see next bullet). Use `obsm=[embed_key]` on the random-access dataloader path to drop every unused embedding's per-worker RAM.
  - `backed=True`: `X` and layers are lazy `ScxBackedSparseDataset`.
  - `modality`: select one modality of a multimodal file (requires `backed=True`; incompatible with `var_names`/`obs_filter`/`layers`/`obsm`).
  - `eager=False`: `obsp`/`varp`/`varm` (and `layers` in non-backed mode) are lazy bridges decoded on first access; `eager=True` materializes everything and detaches from the file handle (use before closing the experiment or shipping to a subprocess). `uns` is always eager.
  - **obsm modes**: `obsm=None` → all keys eager (default). `obsm=[...]` + (`eager=True` or `obs_filter`) → selected keys eager. `obsm=[...]` + `backed=False` + `eager=False` → `ScxLazyObsmMapping` (each key → dense numpy on first access). `obsm=[...]` + `backed=True` + `eager=False` + no `obs_filter` → `ScxLazyObsmMapping` of `ScxBackedObsmDataset` (shard-aware dense row-gather: `m[idx]` reads only touched obsm shards, per-key LRU = `cache_shards`; the scalable path for one huge embedding). Deletion vectors compose via `kept_to_global`.
- `to_mudata(backed=False, cache_shards=4)` — multimodal `mudata.MuData` (eager raises on single-modality; backed wraps single-modality in a one-modality MuData).
- `query() -> PyQueryPipeline`.
- `mark_deleted(mask)`, `validate()`.
- `detection_counts(axis="var", modality=None)` / `cells_expressing(gene, modality=None)` — bitmap fast path when sidecars exist, CSR scan otherwise.
- Properties: `n_obs`, `n_vars`, `nnz`, `shard_count`, `format_version`, `codec_id`, `layer_names`.

## Query pipeline (`PyQueryPipeline`)
- `filter_obs(expr)` / `filter_var(expr)` — predicate strings (e.g. `"tissue == 'lung'"`).
- `select_genes(indices)` — gene projection.
- `with_normalize(target_sum=1e4)` / `with_log1p()` — applied in Rust during `collect`.
- `limit(n)`, `count() -> int`, `collect() -> PyQueryResult`.
- `PyQueryResult.to_anndata()` — zero-copy CSR.

`collect()` **always materializes** the matching subset into an in-memory scipy
CSR (there is no `collect(backed=True)` today). Pushdown skips non-matching
shards **only if predicate indexes were written at convert time**
(`index_obs`/`index_preset`); otherwise the filter still applies, just via a full
obs scan. If the matching subset is still too big to materialize, use
`to_anndata(backed=True, obs_filter=...)` for a lazy on-disk filtered view —
note that path evaluates the predicate via pandas `.query()`, not the engine
grammar that `filter_obs` uses.

## ScxBackedSparseDataset (backed `adata.X`)
Lazy CSR; only requested shards decode on access. `shape`, `dtype` (always
float32), `format` (`"csr"`), `backend` (`"scx"`), `ndim` (2). Cache size via
`to_anndata(backed=True, cache_shards=N)` (default 4; `0` = no cache). Backed
mode sees only non-deleted cells. `ScxLazyTransformedDataset` is the wrapper
produced when you apply lazy `normalize_total`/`log1p`.

## pyscx.accel.* — Rust-native accelerators
All write to standard AnnData slots, so downstream scanpy works unchanged. Most
take `device="auto"|"cpu"|"gpu"|"gpu:N"`. Several take `prefer_format="csr"`
(default) or `"csc"` (requires a CSC sidecar from `csc="always"` at convert).

**Preprocessing / QC (non-materializing on backed/lazy):**
- `normalize_total(adata, target_sum=10000.0)` — on backed/lazy, appends a transform; on scipy CSR delegates to `sc.pp.normalize_total`.
- `log1p(adata)` — appends `Log1p` (fuses with preceding `NormalizeTotal`); on scipy CSR delegates to `sc.pp.log1p`.
- `filter_cells(adata, min_genes=None, max_genes=None, min_counts=None, max_counts=None)` — streaming; updates the deletion vector.
- `filter_genes(adata, min_cells=None, max_cells=None, min_counts=None, max_counts=None)` — streaming; sets the column projection.
- `subset_obs(adata, mask_or_indices)` — boolean mask **or** integer array. **Integer arrays become a boolean mask: order not preserved, duplicates collapsed** (unlike NumPy fancy indexing). Use a boolean mask for unambiguous results.
- `calculate_qc_metrics(adata, qc_vars=None, log1p=True, inplace=True, prefer_format="csr")` — streaming per-cell `n_genes_by_counts`/`total_counts` and per-gene `n_cells_by_counts`/`total_counts`. `qc_vars=["mt"]` needs `adata.var["mt"]` tagged yourself (else `pct_counts_mt` is not produced).
- `highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", batch_key=None, span=0.3, subset=False, n_bins=20, device="auto", prefer_format="csr", layer=None)` — streaming. **seurat_v3 expects raw counts** — run before normalize/log1p or pass `layer="counts"`. Runs the scx-native kernel on backed, lazy, **and** in-memory scipy/dense `X` for `flavor` in `seurat_v3`/`seurat_v3_paper`/`seurat` (a materialized `X` is wrapped in a single-shard `ShardSource`); only `cell_ranger` delegates to scanpy. **High-cardinality `batch_key`** (e.g. CELLxGENE `dataset_id` → many <150-cell batches) makes some per-batch loess fits singular; the native path catches each, warns naming the batch, and drops it from the ranking, so HVG completes — prefer a coarser `batch_key` if many drop (`filter_genes(min_cells=10)` only helps the no-`batch_key` global fit). Writes `var["highly_variable"]`, `var["means"]`, `var["variances"]`, `var["variances_norm"]`, `var["highly_variable_rank"]`.

**Dimensionality reduction / graph:**
- `pca(adata, n_comps=50, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto")` — randomized SVD with streaming SpMM. Writes `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]`. **PCA rejects CSC.**
- `neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0, ef_construction=200, ef_search=200, device="auto")` — CPU HNSW / GPU CAGRA. Writes `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`.
- `umap(adata, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, learning_rate=1.0, random_state=0, device="auto")` — writes `obsm["X_umap"]`.
- `leiden(adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=-1, device="auto")` — reads `obsp["connectivities"]`. **Pin `device="cpu"` for label stability** (GPU Leiden diverges from `leidenalg`; documented). Writes `obs[key_added]` (categorical) + `uns["leiden"]`.

**DE / perturbation:**
- `rank_genes_groups(adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, log_transformed=False, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr")` — parallel Wilcoxon + BH. Writes `uns["rank_genes_groups"]` (or returns a DataFrame when `stratify_by` set).
- `rank_genes_groups_df(...) -> polars.DataFrame` — same Wilcoxon in cell-eval's `DEResults` schema.
- `pdex_ref(adata, groupby, *, reference="non-targeting", is_log1p=None, geometric_mean=True, epsilon=0.0, gene_chunk_size=None, prefer_format="csr", device="auto") -> polars.DataFrame` — perturbation-screen DE (Mann–Whitney U + pseudobulk geometric-mean LFC vs one reference group). Set `SCX_GPU_DE_V3=1` to use the CSC-direct GPU driver (needs a CSC sidecar).
- `pseudobulk_dex(adata, groupby, test_col, reference, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr", gene_indices=None) -> DataFrame` — streaming pseudobulk + pydeseq2 (optional dep). CSC needs a gene subset.
- `pseudobulk_means(adata, groupby, min_cells_per_group=1) -> (ndarray[P,G] f64, group_names)`.
- `perturbation_metrics(adata_real, adata_pred, pert_col="perturbation", control="control", metrics=None, min_cells_per_group=1) -> dict` — `{pearson_delta, mse, mae, mse_delta, mae_delta}`.
- `energy_distance(adata_real, adata_pred, pert_col=..., control=..., metric="euclidean", embed_key=None, backend=None, dtype=None) -> float`; `energy_distance_details(...)` returns the per-pert breakdown.
- `discrimination_score(...)`, `knockdown_efficiency(adata, pert_col=..., control=..., eps=1e-8)` (input must be normalized, NOT log1p'd; writes `obs["KnockDownEfficiency"]`, `obs["KnockDownGeneFC"]`), `clustering_agreement(...)`.
- `adjusted_mutual_info(a, b)` / `normalized_mutual_info(a, b)` / `adjusted_rand_index(a, b)` (rescaled to `[0,1]`).

**Integration / metrics:**
- `harmony_integrate(adata, key, *, basis="X_pca", adjusted_basis=None, ..., random_state=0, device="auto")` — Rust Harmony2. `key` is one obs column or a list. Writes corrected embedding to `obsm[adjusted_basis or basis]` + `uns["harmony"]`. Param names match `scanpy.external.pp.harmony_integrate`.
- `compute_lisi(adata, key, *, basis="X_pca", perplexity=30.0, n_neighbors=None) -> np.ndarray` — Local Inverse Simpson Index; also writes `obs[f"lisi_{key}"]`.

**Streaming column stats** (on `ScxBackedSparseDataset` / `ScxLazyTransformedDataset`; honor `col_projection` / deletion vector):
- `col_sums(dataset, prefer_format="csr") -> f64[]`, `col_nnz -> i64[]`, `col_min`, `col_max`, `col_var`.

**GPU helpers:**
- `gpu_info() -> dict | None` — `{device, total_vram_gb, free_vram_gb}` or `None` if unavailable.
- `estimate_gpu_memory(adata, operation, **kwargs) -> {required_gb, fits_in_vram}` — `operation` ∈ `"pca"` / `"knn"` / `"umap"` / `"leiden"`.

## Streaming write-back (copy-on-write, no full materialization)
- `pyscx.preprocess(source, target, ops, target_sum=None)` — shard-by-shard transform to a new file.
- `pyscx.save_layer(source, target, layer_name, ops, target_sum=None)` — write transformed data as a named layer.
- `pyscx.iter_chunks(adata, chunk_size="shard")` — shard-aligned or fixed-size chunk iterator.

## Worked examples

In-memory:
```python
import pyscx, scanpy as sc
adata = pyscx.open("experiment.scx").to_anndata()
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[(adata.obs.n_genes_by_counts >= 200) & (adata.obs.pct_counts_mt < 20)].copy()
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts")
adata = adata[:, adata.var.highly_variable].copy()
sc.pp.pca(adata); sc.pp.neighbors(adata); sc.tl.umap(adata); sc.tl.leiden(adata)
```

Backed + lazy:
```python
import pyscx
from pyscx import accel
adata = pyscx.open("atlas.scx").to_anndata(backed=True)
adata.var["mt"] = adata.var["feature_name"].str.upper().str.startswith("MT-")  # Census: symbols in feature_name
accel.calculate_qc_metrics(adata, qc_vars=["mt"])
adata = adata[(adata.obs.n_genes_by_counts >= 200) & (adata.obs.pct_counts_mt < 20)].copy()
accel.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", batch_key="dataset_id")
adata = adata[:, adata.var.highly_variable].copy()
accel.normalize_total(adata, target_sum=1e4); accel.log1p(adata)
accel.pca(adata, n_comps=50); accel.neighbors(adata, n_neighbors=15, use_rep="X_pca")
accel.leiden(adata, resolution=1.0, device="cpu"); accel.umap(adata)
```
