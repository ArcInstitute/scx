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

## Experiment (returned by `pyscx.open(path)`)

The Python-visible class is `Experiment` (Rust type `PyExperiment`). `repr` is
AnnData-style (`Experiment object with n_obs × n_vars = …` + indented
`obs:`/`var:`/`uns:`/`obsm:`/`varm:`/`layers:` key lists); codec/shard internals
are on `.info()`. Key accessors: `obs_keys`, `var_keys`, `obsm_keys`,
`varm_keys`, `uns_keys`, `layer_names`, plus `has_csc` / `has_deletions`.
One-liners: `pyscx.read(path, **kwargs)` (= `open(path).to_anndata(**kwargs)`)
and `pyscx.write(adata, path, **kwargs)` (= `from_anndata`).

- `to_anndata(backed=False, cache_shards=4, var_names=None, obs_filter=None, layers=None, preserve_slots=False, modality=None, eager=False, memory_budget=None, obsm=None, preserve_var_order=False, strict_var_names=True, container="csr", data_dtype=None, index_dtype=None, allow_lossy=False, obsp=None, varp=None, varm=None, raw=True)` — convert to AnnData.
  - `var_names`: gene-name list to project (column subset). Sorted-order set selector by default; `preserve_var_order=True` returns the requested order (first-wins dedup). `strict_var_names=True` (default) raises `KeyError` on unknown names — `False` drops them silently.
  - `obs_filter`: predicate string. **Non-backed mode** routes through the query engine (shard pushdown). **Backed mode** evaluates it with **pandas `.query()`** (richer grammar, no pushdown) and folds the matches into the dataset's row set — so the same expression can resolve via different engines depending on `backed`.
  - `layers`: layer names to load (default all).
  - `obsm`: obsm keys to load (default `None` = all; byte-identical to before). `obsm=[...]` loads only the listed embeddings (unknown key → `KeyError`; `[]` = none) **and** switches the obsm materialisation mode (see next bullet). Use `obsm=[embed_key]` on the random-access dataloader path to drop every unused embedding's per-worker RAM.
  - `backed=True`: `X` and layers are lazy `ScxBackedSparseDataset`.
  - `obsp` / `varp` / `varm`: keys to load from those slots (default `None` = all). Same contract as `obsm=` — `[]` = none, a list = that subset, unknown key → `KeyError` — except that an empty/fully-excluding list builds **no bridge at all**, so `adata.obsp` is anndata's own empty mapping. Worth setting: anndata decodes a whole slot on the first `adata.obsp` access (`AlignedActual.__init__` validates every entry), so an `n_obs × n_obs` kNN graph comes off disk uninvited. `to_anndata(layers=[], obsm=[], obsp=[], varp=[], varm=[], raw=False)` is the X/obs/var/uns-only read.
  - `raw`: `True` (default) reconstructs `adata.raw` where the mode allows and emits the `dropped_raw` notice where it cannot (backed, obs-filtered query, deletion-vector-active file). `False` does neither — the opt-out for workloads that never want raw.
  - `container` / `data_dtype` / `index_dtype` / `allow_lossy`: read-side materialisation. Default (`csr` / `float32` / `int32`) is zero-copy; a narrow request assembles X and raw directly at the target width (`backed=False` only). A lossy narrow raises unless `allow_lossy=True`.
  - `layers=` / `var_names=` force eager assembly of the aligned slots (`obsp` / `varp` / `varm`), whatever `eager=` says — pair them with the slot filters above to keep that bounded. Under `var_names=`, `X` and each selected layer are the exception: they are assembled already projected, shard by shard, and never exist at full width. `.raw` is never gene-projected (anndata does not slice it on that axis), so `raw=False` is what drops it.
  - `modality`: select one modality of a multimodal file (requires `backed=True`; incompatible with `var_names`/`obs_filter`/`layers`/`obsm`/`obsp`/`varp`/`varm`/`raw=False` — it would silently ignore them, so it raises instead).
  - `eager=False`: `obsp`/`varp`/`varm` (and `layers` in non-backed mode) are lazy bridges decoded on first access; `eager=True` materializes everything and detaches from the file handle (use before closing the experiment or shipping to a subprocess). `uns` is always eager.
  - **obsm modes**: `obsm=None` → all keys eager (default). `obsm=[...]` + (`eager=True` or `obs_filter`) → selected keys eager. `obsm=[...]` + `backed=False` + `eager=False` → `ScxLazyObsmMapping` (each key → dense numpy on first access). `obsm=[...]` + `backed=True` + `eager=False` + no `obs_filter` → `ScxLazyObsmMapping` of `ScxBackedObsmDataset` (shard-aware dense row-gather: `m[idx]` reads only touched obsm shards, per-key LRU = `cache_shards`; the scalable path for one huge embedding). Deletion vectors compose via `kept_to_global`.
- `to_mudata(backed=False, cache_shards=4)` — multimodal `mudata.MuData` (eager raises on single-modality; backed wraps single-modality in a one-modality MuData).
- `query() -> PyQueryPipeline`.
- `mark_deleted(mask)`, `validate()`.
- `detection_counts(axis="var", modality=None)` / `cells_expressing(gene, modality=None)` — bitmap fast path when sidecars exist, CSR scan otherwise.
- `gather_rows_sparse(rows, modality=None, cache_shards=4, layer=None, logical=True)` — random-access sparse row gather (bool mask or integer array-like; request order, duplicates allowed; logical rows by default, `layer=` for a layer); returns a scipy CSR submatrix with peak memory = result + the shard cache + up to `cache_shards` shards decoding in flight while it fills (never a second copy of the result). The same gather is `adata.X[rows]` / `adata.layers[name][rows]` on a backed handle.
- `provenance()` — returns the file's provenance chain.
- Properties: `n_obs`, `n_vars`, `nnz`, `shard_count`, `format_version`, `codec_id`, `layer_names`; `value_encoding` (`"uint16"` / `"mixed (uint8, uint16)"`), `is_integer` (every CSR shard integer-encoded — "are these counts?" without decoding), `max_value` (largest stored value from catalog stats; 0 for a float file). `info()` prints the same three as tokens.
- Backed handle column selectors: `adata.X[:, 5]`, `X[:, [7, 2, 11]]`, `X[:, 10:20]`, `X[:, mask]`, any-order int ndarray → a projected **handle**, no decode (request order kept). Repeated columns (`X[:, [3, 1, 3]]`) → scipy from the projected unique columns. `np.asarray(handle)` raises `TypeError` — use `to_memory()` / `toarray()`. `handle.stored_dtype` is the on-disk encoding (`dtype` stays `float32`); `handle.cache_shards` reads the LRU size back.

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
(default) or `"csc"` (requires a CSC sidecar from `csc="auto"|"always"` at
convert).

**How to get GPU CSC-direct DE (`gpu_csc_v3`).** `prefer_format` and `device`
are two *different* axes, and the GPU CSC-direct route is selected by the
planner, **not** by `prefer_format="csc"`:

- For **GPU-fast** DE, keep the **default `prefer_format="csr"`** and pass
  `device="gpu"` (or `"auto"`). When the file has a CSC sidecar the planner
  routes `rank_genes_groups`/`pdex_ref` to `gpu_csc_v3` automatically; without a
  sidecar it uses `gpu_csr_v3`.
- `prefer_format="csc"` selects the **CPU** column-major streaming path
  (`cpu_csc`); it has no GPU kernel. With `device="auto"` it silently runs on
  CPU; with an explicit `device="gpu"` it raises a `RuntimeError` explaining
  that `prefer_format='csc'` is the CPU path and that GPU CSC-direct comes from
  the default `prefer_format='csr'` + `device='gpu'`.

Always confirm the backend that actually ran via
`adata.uns["scx_accel"][op]["route"]` — `gpu_csc_v3` (GPU CSC-direct),
`gpu_csr_v3` (GPU, no sidecar), or `cpu_csc` / `cpu_csr` (plus `cpu_csc_nnz`,
the 1-vs-rest exact-nnz Wilcoxon kernel, when `SCX_ACCEL_WILCOXON_NNZ=1`). All accel ops stamp
this envelope, including `harmony_integrate` (`gpu_dense` / `cpu_dense`).

**Preprocessing / QC (non-materializing on backed/lazy):**
- `normalize_total(adata, target_sum=None, device="auto")` — on backed/lazy, appends a transform; on scipy CSR delegates to `sc.pp.normalize_total`.
- `log1p(adata, device="auto")` — appends `Log1p` (fuses with preceding `NormalizeTotal`); on scipy CSR delegates to `sc.pp.log1p`.
- `filter_cells(adata, min_genes=None, max_genes=None, min_counts=None, max_counts=None)` — streaming; updates the deletion vector.
- `filter_genes(adata, min_cells=None, max_cells=None, min_counts=None, max_counts=None)` — streaming; sets the column projection.
- `subset_obs(adata, mask_or_indices)` — boolean mask **or** integer array. **Integer arrays become a boolean mask: order not preserved, duplicates collapsed** (unlike NumPy fancy indexing). Use a boolean mask for unambiguous results.
- `pflog(adata, *, alpha=None, layer=None, store="pca", n_components=50, n_oversamples=10, n_power_iterations=2, zero_center=True, random_state=0, obsm_key="X_pflog_pca", baseline_key="pflog_baseline", layer_out=None, out=None, store_repr="delta_baseline", shard_size=None, dense_max_elems=200_000_000, device="auto")` — PFlog (v4) shifted-log normalization on **raw counts** (Booeshaghi et al.): shift by the matrix-wide Anscombe pseudocount `1/(4α)`, `log1p`, center — no per-cell depth. `alpha=None` estimates the NB overdispersion `α` once from the counts (a float pins it); the fit is stamped into `uns["pflog"]`. It does **not** transform `X` in place — the delta source is the lazy `scale{4α}→log1p` chain. `store`: `"pca"` (default, stream randomized PCA into `obsm[obsm_key]`), `"baseline"` (write `obs[baseline_key]` only), `"dense"` (materialize full normalized layer to `out=` SCX or `layer_out=`), `"all"` (PCA + dense). `store_repr`: `"delta_baseline"` (compact sparse delta + per-cell baseline) or `"dense"` (literal). Always writes `obs[baseline_key]` (default `"pflog_baseline"`). Companion: `pflog_reconstruct(adata, baseline_key="pflog_baseline")` reconstructs the dense layer from the decomposition.
- `score_genes(adata, gene_list, ctrl_size=50, gene_pool=None, n_bins=25, score_name="score", random_state=0, method="control", layer=None, device="auto", *, ctrl_genes=None)` — gene-set scoring. `method`: `"control"` (default, the `sc.tl.score_genes` algorithm), `"mean"`, or `"zscore"` (decoupler-compatible). Works on backed/lazy data via streaming. The `"control"` sampler is not numpy's, so it picks different control genes from scanpy and the scores differ by roughly 10% of their range at the defaults — pass `ctrl_genes=[...]` to supply the control set yourself and get exact parity (and it works on a backed `X`, where `sc.tl.score_genes` raises). `ctrl_genes=` needs `method="control"` and rejects an explicit `gene_pool=`.
- `calculate_qc_metrics(adata, qc_vars=None, log1p=True, inplace=True, prefer_format="csr", *, layer=None, percent_top=None)` — scanpy's full QC column set (obs `n_genes_by_counts` / `log1p_n_genes_by_counts` / `total_counts` / `log1p_total_counts`; var `n_cells_by_counts` / `mean_counts` / `log1p_mean_counts` / `pct_dropout_by_counts` / `total_counts` / `log1p_total_counts`), from one native streaming kernel on **any** `X` — backed, lazy, a backed layer handle, or in-memory scipy/dense. Needs no scanpy. `qc_vars=["mt"]` needs `adata.var["mt"]` tagged yourself (else `pct_counts_mt` is not produced). `layer=` reads a named layer. `percent_top=(50, 100)` adds `pct_counts_in_top_<n>_genes` in the same pass; it defaults to `None` rather than scanpy's `(50, 100, 200, 500)`, which raises on files with fewer than 500 genes.
- `highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", batch_key=None, span=0.3, subset=False, n_bins=20, device="auto", prefer_format="csr", layer=None)` — streaming. **seurat_v3 expects raw counts** — run before normalize/log1p or pass `layer="counts"`. Runs the scx-native kernel on backed, lazy, **and** in-memory scipy/dense `X` for `flavor` in `seurat_v3`/`seurat_v3_paper`/`seurat` (a materialized `X` is wrapped in a single-shard `ShardSource`); only `cell_ranger` delegates to scanpy. **High-cardinality `batch_key`** (e.g. CELLxGENE `dataset_id` → many <150-cell batches) makes some per-batch loess fits singular; the native path catches each, warns naming the batch, and drops it from the ranking, so HVG completes — prefer a coarser `batch_key` if many drop (`filter_genes(min_cells=10)` only helps the no-`batch_key` global fit). Writes `var["highly_variable"]`, `var["means"]`, `var["variances"]`, `var["variances_norm"]`, `var["highly_variable_rank"]`.

**Dimensionality reduction / graph:**
- `pca(adata, n_comps=50, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", prefer_format="csr", allow_tf32=False, spmm_policy="default", memory_budget=None, mask_var=None)` — in-VRAM data routes to `rsc.pp.pca` (rapids-singlecell); >VRAM data uses native randomized SVD with streaming shards. `method`: `"auto"` (selects covariance or randomized by var count), `"covariance"`, `"randomized"`. Writes `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]`. **PCA rejects CSC.**
- `neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0, ef_construction=200, ef_search=200, device="auto")` — CPU HNSW; GPU routes to `rsc.pp.neighbors` (rapids-singlecell). Writes `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`.
- `umap(adata, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, learning_rate=1.0, random_state=0, device="auto")` — GPU routes to `rsc.tl.umap` (rapids-singlecell); CPU falls back to scanpy. Writes `obsm["X_umap"]`.
- `leiden(adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=2, device="auto", parallel=False, theta=1.0)` — reads `obsp["connectivities"]`. `n_iterations` is a leidenalg-style outer-iteration unit on CPU; on the cuGraph (GPU) path values `<= 2` (incl. `-1`/`0` sentinels) map to cuGraph's `max_iter=100` (forwarding the bare leidenalg default of 2 used to starve cuGraph coarsening into a degenerate ~116k-cluster partition — fixed). With that fix, GPU and CPU give comparable cluster counts (e.g. 48 vs 53 on 1 M cells, ARI ≈ 0.72). **Still prefer `device="cpu"` when downstream cares about exact cluster identity** (DE, annotation transfer) — the two backends differ in label stability by design, so they won't match exactly. Effective GPU cap recorded in `uns["leiden"]["params"]["max_iter"]`. Writes `obs[key_added]` (categorical) + `uns["leiden"]`.
- `pca_neighbors(adata, n_comps=50, n_neighbors=15, ...)` — fused PCA→kNN GPU pipeline (rapids-singlecell). Avoids GPU↔CPU roundtrip.
- `pca_neighbors_umap(adata, n_comps=50, n_neighbors=15, n_components=2, ...)` — fused PCA→kNN→UMAP GPU pipeline.

**DE / perturbation:**
- `rank_genes_groups(adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, stratify_by=None, min_cells_per_stratum=50, rankby_abs=False, tie_correct=False, prefer_format="auto", device="auto", use_raw=None, layer=None, *, pts=False, groups=None, corr_method="benjamini-hochberg")` — parallel Wilcoxon + BH. Log-transform state auto-detected from `adata.uns["log1p"]`. Writes `uns["rank_genes_groups"]` (or returns a DataFrame when `stratify_by` set). `pts=True` adds scanpy's `pts` / `pts_rest` frames (and `pct_nz_*` columns in `rank_genes_groups_df`; needs unique `var_names`); `groups=` restricts the *reported* groups without changing "rest"; **any participating group with < 2 cells raises**, as in scanpy — every level when `groups=` is omitted, the named ones when it is given, an unused category included (its message names `remove_unused_categories()`). Cells with no `groupby` label get no group of their own, but for `reference="rest"` they are in the rank pool and in every group's "rest", as scanpy does. `corr_method` accepts only BH (else raises). Both frames survive `from_anndata` / `to_anndata` and h5ad (uns DataFrame envelope).
- `rank_genes_groups_df(..., output="pandas", ...) -> pandas.DataFrame` — same Wilcoxon in cell-eval's `DEResults` column schema, plus a `group=` extract mode that is the `sc.get.rank_genes_groups_df` alias. Pass `output="polars"` when feeding `cell_eval` (its `DEResults.data` is typed `pl.DataFrame`).
- `pdex_ref(adata, groupby, *, reference="non-targeting", is_log1p=None, geometric_mean=True, epsilon=1e-9, cpm_filter=None, gene_chunk_size=None, prefer_format="auto", device="auto", output="pandas", use_raw=None, layer=None, groups=None) -> pandas.DataFrame` — perturbation-screen DE (Mann–Whitney U + pseudobulk geometric-mean LFC vs one reference group). `groups=[...]` tests only those targets (rows identical to the full run's, work scales with the request). `epsilon` (default `1e-9`): finite-guard for LFC denominators matching pdex ≥ 0.2.x; pass `0.0` for the legacy ±inf behaviour. `cpm_filter`: optional per-gene CPM floor — genes below this in both test and reference are dropped and FDR recomputed. With a CSC sidecar, `device="auto"` selects the CSC-direct driver (`gpu_csc_v3`); without one it uses `gpu_csr_v3`. **Do not combine `device="gpu"` with `prefer_format="csc"` — it raises (use `device="auto"`).** `output="polars"` returns the polars frame upstream `pdex` / `cell_eval` use (needs the `eval` extra); the default pandas needs no extra.
- `pseudobulk_dex(adata, groupby=None, test_col=None, reference=None, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr", gene_indices=None, n_cpus=None, backend=None, nbglm_options=None, *, sample_cols=None, sample_key=None) -> DataFrame` — streaming pseudobulk DE. `backend`: `None` → `"nb_glm"` (default since v0.13; Rust-native DESeq2-style NB GLM, CPU-only, no optional dependency) or `"pydeseq2"` (exact DESeq2 numerics; needs `pip install 'pyscx[pydeseq2]'`, and is required for `stratify_by=` and `aggr_method="mean"`). CSC needs a gene subset.
- `pseudobulk_means(adata, groupby, min_cells_per_group=1, device="auto") -> (ndarray[P,G] f64, group_names)` — `groupby` is `str | list[str]`; group names are tuples when several columns are given.
- `nb_glm(counts, design, size_factors=None, contrast=None, gene_names=None, sample_names=None, options=None, counts_axis="samples_by_genes", device="auto")` — Rust-native DESeq2-style NB GLM on pre-aggregated count matrices. CPU-only (route `cpu_nb_glm`).
- `pdex_nb_glm(adata, groupby, reference, stratify_by=None, min_cells_per_group=10, min_cells_per_stratum=50, is_log1p=None, nbglm_options=None, gene_chunk_size=None, prefer_format="csr", device="auto", design=None, output="pandas") -> pandas.DataFrame` — pseudobulk from AnnData + Rust NB GLM in one call. CPU-only. Same cell-eval column schema as `rank_genes_groups_df` / `pdex_ref`; pass `output="polars"` when feeding `cell_eval`.
- `perturbation_metrics(adata_real, adata_pred, pert_col="perturbation", control="control", metrics=None, embed_key=None, min_cells_per_group=1, device="auto") -> dict` — `{pearson_delta, mse, mae, mse_delta, mae_delta}`.
- `energy_distance(adata_real, adata_pred, pert_col=..., control=..., metric="euclidean", embed_key=None, backend=None, dtype=None, device="auto") -> float`; `energy_distance_details(...)` returns the per-pert breakdown.
- `discrimination_score(...)`, `knockdown_efficiency(adata, pert_col=..., control=..., eps=1e-8, device="auto")` (input must be normalized, NOT log1p'd; writes `obs["KnockDownEfficiency"]`, `obs["KnockDownGeneFC"]`), `clustering_agreement(...)`.
- `adjusted_mutual_info(labels_a, labels_b)` / `normalized_mutual_info(labels_a, labels_b)` / `adjusted_rand_index(labels_a, labels_b, rescaled=False)` — `rescaled=True` maps ARI onto `[0,1]`; the default leaves it on its native `[-1,1]`.

**Integration / metrics:**
- `harmony_integrate(adata, key, *, basis="X_pca", adjusted_basis="X_pca_harmony", ..., random_state=0, device="auto")` — Rust Harmony2. `key` is one obs column or a list. Writes the corrected embedding to a **new** `obsm[adjusted_basis]` key (default `"X_pca_harmony"`), preserving the input `basis` — matching `scanpy.external.pp.harmony_integrate`. Pass `adjusted_basis="X_pca"` (or `=basis`) to overwrite in place. Also writes `uns["harmony"]`. Param names match scanpy.
- `compute_lisi(adata, key, *, basis="X_pca", perplexity=30.0, n_neighbors=None, approximate_knn=False) -> np.ndarray` — Local Inverse Simpson Index (iLISI batch-mixing / cLISI label-purity); also writes `obs[f"lisi_{key}"]`. Higher iLISI = better batch mixing — useful pre/post `harmony_integrate` (e.g. 3.2 → 6.0 on a 116-batch census file confirms integration worked). **Slow at scale by default:** exact kNN is O(N²) and ran ~44 min per call on 1 M cells (the exact path logs a hint above ~50k cells). Pass `approximate_knn=True` to swap in an HNSW kNN — ~10× faster at N≳100k with small drift (~0.01–0.05 mean LISI) — for atlas-scale runs; the default stays exact for byte-for-byte R `lisi` parity.

**Streaming column stats** (on `ScxBackedSparseDataset` / `ScxLazyTransformedDataset`; honor `col_projection` / deletion vector):
- `col_sums(dataset, prefer_format="csr") -> f64[]`, `col_nnz -> i64[]`, `col_min`, `col_max`, `col_var`.
- Restrict to a gene subset first with `dataset[:, cols]` (a projected handle, no decode) rather than materialising.

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
