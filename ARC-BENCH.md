# Feasibility Report: arc-bench / cell-eval Metrics as SCX Rust Accelerators

## Executive Summary

The `arc-bench` and `cell-eval` libraries perform evaluation of single-cell
perturbation predictions. `cell-eval` defines 24 registered metrics; `arc-bench`
adds upstream preprocessing (normalization, knockdown efficiency, HVG selection)
and an MLP-based intrinsic/extrinsic evaluator.

The compute-intensive operations fall into three categories:

1. **Pairwise distance matrices** (e-distance, discrimination score) — O(N²) per
   perturbation, the dominant bottleneck. Strong Rust candidate.
2. **Pseudobulk aggregation** (mean per perturbation group from sparse matrix) —
   foundation for 6+ metrics. The Rust side of SCX already implements this
   (`pseudobulk_aggregate` with `AggregationMethod::Mean`), but the Python
   binding only exposes it through the `pseudobulk_dex` → pydeseq2 pipeline.
   A direct Python binding returning the dense means matrix is the key gap.
3. **Per-cell perturbation metrics** (knockdown efficiency, log deviation) —
   require extracting single gene columns per perturbation from sparse matrices.

The remaining operations are either already fast (polars DataFrame metrics on
small aggregated data, <100ms), depend on external ecosystems that shouldn't
be replaced (PyTorch MLP training, scanpy Seurat v3 HVG), or already exist
in SCX (Wilcoxon DE, normalize/log1p, Leiden clustering).

---

## Source code locations

| Library | Path | Key files |
|---------|------|-----------|
| **arc-bench** | `/home/nickyoungblut/dev/python/arc-bench/` | `src/arc_bench/tools/normalize_transform/core.py`, `tools/filter_h5ad/core.py`, `tools/hvg_union/core.py`, `tools/pert_eval/cli.py` |
| **cell-eval** | `/home/nickyoungblut/dev/python/cell-eval/` | `src/cell_eval/metrics/_anndata.py`, `metrics/_de.py`, `_types/_anndata.py`, `_evaluator.py`, `_pipeline/_runner.py` |
| **scx-accel** | `scx-accel/src/` | `pseudobulk.rs`, `diffexp.rs`, `pca.rs`, `neighbors.rs`, `umap.rs` |
| **pyscx bindings** | `pyscx/src/accel.rs` | All `#[pyfunction]` definitions |

---

## Detailed analysis of arc-bench computations

### normalize_transform/core.py

**Processing order per file** (`normalize_log_transform_single`):
1. Load h5ad, ensure CSR via `adata.X.tocsr()`
2. Optional gene alignment (`align_to_gene_set`) — reindex sparse columns to a
   reference gene set, filling missing genes with zero columns
3. Optional binomial downsampling (`downsample_counts`) — `rng.binomial(X.data, frac)`
   on CSR `.data` array
4. `sc.pp.normalize_total(adata, target_sum=...)` — scanpy
5. Compute knockdown efficiency (BEFORE log1p)
6. `sc.pp.log1p(adata)` — scanpy
7. Compute log deviation (AFTER log1p)

**Knockdown efficiency** (`compute_knockdown_efficiency`, lines 353–411):
```
For each non-control perturbation p:
    gene_idx = index of gene named p in var_names
    cells = rows where obs[pert_col] == p
    x_target = X[cells, gene_idx]        # extract single column slice
    μ_control = control_baseline[gene_idx]  # precomputed mean of control cells
    KD[cells] = 1.0 - x_target / (μ_control + ε)
```
The Python code calls `X_raw[pert_mask, gene_idx]` which on a CSR matrix
requires scanning all non-zeros in those rows. For each perturbation, it calls
`.toarray()` on the resulting column slice. With hundreds of perturbations,
this means hundreds of sparse→dense round-trips.

**Log deviation** (`compute_log_deviation`, lines 414–468):
Same pattern but operates on already-log1p-transformed data:
```
FC[cells] = log1p(x_target) - log1p(μ_control[gene_idx])
```

**Control baseline** (`compute_control_baseline`, lines 308–350):
```
control_mask = obs[pert_col] == control_perturbation
result = X[control_mask].mean(axis=0)  # sparse mean over control cells
```

### filter_h5ad/core.py

**Pseudobulk** (`compute_pseudobulk`, lines 87–139):
Uses `sc.get.aggregate(adata, by=perturbation_col, func="mean"|"sum")`.
Result is an AnnData with one row per perturbation.

**HVG selection** (`select_hvgs_from_pseudobulk`, lines 142–194):
Concatenates pseudobulk AnnDatas with `ad.concat(axis=0, join="inner")`,
converts to dense, calls `sc.pp.highly_variable_genes(flavor="seurat_v3")`.

**Two-pass architecture** (lines 285–655):
- Pass 1: Load → filter cells → pseudobulk → save temp → discard
- HVG: Load pseudobulk temps → concatenate → Seurat v3 selection
- Pass 2: Reload original → refilter → subset to HVGs → write output

### hvg_union/core.py

Two strategies:
- `compute_hvg_union_per_file`: Per-file HVG → set union → sort by reference order
- `compute_hvgs_from_concatenated_pseudobulks`: Per-file pseudobulk (via `adpbulk`)
  → concatenate → single HVG selection

### pert_eval/cli.py

**Standard evaluation** (`_run_standard`):
Delegates entirely to `cell_eval.MetricsEvaluator`. Clips adata.X to [0, 14],
splits by cell type (context), runs MetricsEvaluator per context.

**Intrinsic/extrinsic evaluation** (`_run_inextrinsic`):
Per cell type: train a SingleLayerMLP (Linear → LayerNorm → GELU → Dropout →
Linear) on intrinsic data to classify perturbations, evaluate on extrinsic
(predicted) data. Uses PyTorch, Adam optimizer, CrossEntropyLoss.

---

## Detailed analysis of cell-eval metrics

### Data structures

**PerturbationAnndataPair** (`_types/_anndata.py`, lines 17–38):
Core container holding `real` and `pred` AnnData objects, perturbation column
name, control label, and cached pseudobulk data.

Key fields:
- `perts: NDArray[str]` — unique perturbation names (excluding control)
- `genes: NDArray[str]` — gene names (validated equal between real/pred)
- `pert_mask_real/pred: dict[str, NDArray[int]]` — cell indices per perturbation
- `bulk_real/pred: dict[str, tuple[NDArray[str], NDArray[float64]]]` — cached
  pseudobulk means, keyed by embed_key

**BulkArrays** (lines 269–293):
Per-perturbation pseudobulk data with 4 vectors (all 1D, length = n_genes
or n_embedding_dims):
- `pert_real`, `pert_pred` — mean expression for this perturbation
- `ctrl_real`, `ctrl_pred` — mean expression for control
- `perturbation_effect(which)` returns `pert_{which} - ctrl_{which}`

**CellArrays** (lines 296–304):
Per-perturbation single-cell data with 4 matrices (2D):
- `pert_real[N_pert_real, G]`, `pert_pred[N_pert_pred, G]`
- `ctrl_real[N_ctrl_real, G]`, `ctrl_pred[N_ctrl_pred, G]`

**Pseudobulk computation** (`_bulk_anndata`, lines 112–142):
```python
matrix = adata.X  # or adata.obsm[embed_key]
if issparse(matrix):
    matrix = matrix.toarray()           # full dense conversion
frame = pl.DataFrame(matrix).with_columns(
    groupby_key=adata.obs[pert_col].to_numpy(str)
)
bulked = frame.group_by("groupby_key").mean().sort("groupby_key")
```
This converts the entire sparse matrix to dense, loads it into polars, and
computes group means. For large datasets (>100K cells × 30K genes), this
allocates multi-GB dense arrays.

### AnnData metrics (`_anndata.py`)

All metrics below operate on `PerturbationAnndataPair`. Metrics that use
pseudobulk data (BulkArrays) call `iter_bulk_arrays()` which triggers the
polars-based pseudobulk computation once (memoized by embed_key).

#### pearson_delta (line 24)
```
For each perturbation p:
    delta_pred = bulk_pred[p] - bulk_pred[control]    # 1D, length G
    delta_real = bulk_real[p] - bulk_real[control]
    result[p] = scipy.stats.pearsonr(delta_pred, delta_real).correlation
```
Operates on BulkArrays. Complexity: O(P × G).

#### mse, mae, mse_delta, mae_delta (lines 36–69)
```
For each perturbation p:
    # mse/mae: compare pseudobulk means directly
    result[p] = sklearn.metrics.mean_squared_error(bulk_real[p], bulk_pred[p])
    # mse_delta/mae_delta: compare perturbation effects (delta from control)
    result[p] = sklearn.metrics.mean_squared_error(delta_real, delta_pred)
```
Operates on BulkArrays. Complexity: O(P × G).

#### edistance (lines 72–126)
**Returns a single float** (Pearson correlation), not per-perturbation values.

Algorithm:
1. Precompute control self-distances (reused across all perturbations):
   ```
   sigma_ctrl_real = pairwise_distances(ctrl_real_cells, metric="euclidean").mean()
   sigma_ctrl_pred = pairwise_distances(ctrl_pred_cells, metric="euclidean").mean()
   ```
2. For each perturbation p, compute e-distance on both real and predicted sides:
   ```
   e_real[p] = 2 * D(pert_real, ctrl_real) - D(pert_real, pert_real) - sigma_ctrl_real
   e_pred[p] = 2 * D(pert_pred, ctrl_pred) - D(pert_pred, pert_pred) - sigma_ctrl_pred
   ```
   where `D(A, B) = pairwise_distances(A, B, metric).mean()`
3. Return `pearsonr(e_real, e_pred).correlation`

**Operates on CellArrays** (per-cell matrices, not pseudobulk).
Complexity: O(P × max(N_ctrl, N_pert)²) where N is cells per group.
This is the most expensive metric — each `pairwise_distances(A, B)` allocates
an `[N_A, N_B]` dense matrix and computes all pairwise Euclidean distances.

Key calls: `sklearn.metrics.pairwise_distances(X, Y, metric)` (6 calls per
perturbation: 2 self-distances + 1 cross-distance for each side).

#### discrimination_score (lines 129–198)
Three registered variants: `discrimination_score_l1`, `_l2`, `_cosine`.

Algorithm:
1. If metric is L1/manhattan/cityblock, force `embed_key=None` (use raw gene
   expression, not embeddings)
2. Compute perturbation effects for ALL perturbations at once:
   ```
   real_effects[P, G] = stack([bulk_real[p] - bulk_real[ctrl] for p in perts])
   pred_effects[P, G] = stack([bulk_pred[p] - bulk_pred[ctrl] for p in perts])
   ```
3. For each perturbation p:
   - If `exclude_target_gene=True` and not using embeddings: remove the column
     corresponding to gene named p (avoids trivial matching on knockdown gene)
   - Compute distances from `pred_effects[p]` to all `real_effects`:
     ```
     distances = pairwise_distances(
         real_effects[:, include_mask],
         pred_effects[p, include_mask].reshape(1, -1),
         metric=metric
     ).flatten()    # shape: [P]
     ```
   - Find rank of correct perturbation in sorted distances
   - `score[p] = 1 - rank / P`    (1.0 = best, 0.0 = worst)

**Operates on BulkArrays** (pseudobulk means).
Complexity: O(P² × G) for the distance matrix (P perturbations × G genes,
computed P times). Per-perturbation gene exclusion requires masking on each
iteration.

#### ClusteringAgreement (lines 227–353)
Configurable: `metric` ∈ {ami, nmi, ari}, `real_resolution=1.0`,
`pred_resolutions=(0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)`, `n_neighbors=15`.

Algorithm:
1. Build centroids (pseudobulk means per perturbation, excluding control):
   ```
   centroids[i] = adata.X[obs[pert_col] == pert_i].mean(axis=0)
   ```
   Result: AnnData with shape `[P, G]` (typically 50–200 rows)
2. Build kNN graph on centroids: `sc.pp.neighbors(centroid_adata, n_neighbors=min(15, P-1), use_rep="X")`
3. Cluster real centroids at fixed resolution: `sc.tl.leiden(real, resolution=1.0, flavor="igraph")`
4. For each of 7 predicted resolutions: cluster predicted centroids, compute
   agreement metric between real and predicted labels
5. Return best score across resolutions

Scoring:
- AMI: `sklearn.metrics.adjusted_mutual_info_score()`
- NMI: `sklearn.metrics.normalized_mutual_info_score()`
- ARI: `(sklearn.metrics.adjusted_rand_score() + 1) / 2` (rescaled to [0,1])

**Operates on centroid matrices** (small, ~50–200 rows). The bottleneck is
the 7× Leiden clustering sweep, not the distance computation.

### DE metrics (`_de.py`)

All DE metrics operate on `DEComparison`, which wraps two `DEResults` (polars
DataFrames with columns: target, feature, fold_change, p_value, fdr,
log2_fold_change, abs_log2_fold_change).

| Metric | Returns | Key polars/sklearn calls |
|--------|---------|--------------------------|
| `de_spearman_sig` | `float` | `pl.corr(method="spearman")` on group-by counts |
| `de_direction_match` | `dict[str, float]` | `sign() ==` comparison, group-by mean |
| `de_spearman_lfc_sig` | `dict[str, float]` | `pl.corr(method="spearman")` per perturbation |
| `de_sig_genes_recall` | `dict[str, float]` | Inner join + group-by length ratio |
| `de_nsig_counts` | `dict[str, dict[str, int]]` | Filter + count per perturbation |
| `pr_auc` | `dict[str, float]` | `average_precision_score(labels, -log10(fdr))` |
| `roc_auc` | `dict[str, float]` | `roc_curve()` + `auc(fpr, tpr)` |
| `overlap_at_N` | `dict[str, float]` | Top-k gene intersection (k=None,50,100,200,500) |
| `precision_at_N` | `dict[str, float]` | Same as overlap but using pred's top-k |

These operate on pre-computed DE DataFrames (hundreds of perturbations ×
thousands of genes). Individual metric latency is typically <100ms.

### Metric profiles

The `MetricPipeline` selects metrics by profile:
- **full**: all 24 metrics
- **minimal**: pearson_delta, mse, mae, discrimination_score_l1, overlap_at_N, precision_at_N, de_nsig_counts
- **vcc**: mae, discrimination_score_l1, overlap_at_N
- **pds**: discrimination_score_l1 only

---

## What SCX already provides

| arc-bench / cell-eval need | SCX equivalent | Gap |
|----------------------------|----------------|-----|
| Pseudobulk means (group-by mean on sparse X) | `pseudobulk_aggregate(method=Mean)` in `scx-accel/src/pseudobulk.rs` returns dense `Vec<f64>` `[n_groups × n_vars]` | **Python binding missing.** `pyscx.accel.pseudobulk_dex()` passes the aggregated counts to pydeseq2 for DE testing. No way to get just the means matrix back as numpy. |
| Wilcoxon rank-sum DE | `wilcoxon_rank_sum_streaming()` in `scx-accel/src/diffexp.rs` returns `DiffExpResult` (z-scores, p-values, adjusted p-values, log fold changes per group) | **Output format mismatch.** cell-eval's `DEComparison` expects a polars DataFrame with columns `(target, feature, fold_change, p_value, fdr, log2_fold_change)`. SCX returns per-group arrays. Need a conversion layer. Also, `pdex` may use different statistical methodology than SCX's Wilcoxon. |
| Normalize total + log1p | `pyscx.accel.normalize_total()` / `log1p()` — lazy transforms, zero materialization | **Sufficient.** |
| kNN graph | `pyscx.accel.neighbors()` — HNSW (CPU) or cuVS CAGRA (GPU) | **Sufficient.** |
| Leiden clustering | `pyscx.accel.leiden()` — cuGraph (GPU) or igraph (CPU) | **Sufficient.** |
| Pairwise distance computation | Not in SCX | **New code needed.** |
| E-distance | Not in SCX | **New code needed.** |
| Discrimination score | Not in SCX | **New code needed.** |
| Knockdown efficiency | Not in SCX | **New code needed.** |
| AMI/NMI/ARI scoring | Not in SCX | **New code needed** (~50 lines). |

---

## Recommended Rust accelerators

### Tier 1: High impact, strong fit for Rust

#### 1. `pyscx.accel.pseudobulk_means()` — new Python binding for existing Rust code

The Rust implementation already exists: `pseudobulk_aggregate()` in
`scx-accel/src/pseudobulk.rs` with `AggregationMethod::Mean` streams shards
via `BackedCsrReader` and returns `PseudobulkResult { counts: Vec<f64>, ... }`
as a dense `[n_groups × n_vars]` row-major matrix. The in-memory variant
`pseudobulk_aggregate_inmemory()` handles scipy CSR inputs.

**What's needed:** A new `#[pyfunction]` in `pyscx/src/accel.rs` that:
1. Extracts `adata.X` (backed or in-memory) and `adata.obs[groupby]` labels
2. Calls the existing Rust `pseudobulk_aggregate` with `AggregationMethod::Mean`
3. Returns the result as a numpy `[P, G]` float64 array + perturbation name list

This replaces cell-eval's `_bulk_anndata()` which converts the entire sparse
matrix to dense via `.toarray()`, loads it into a polars DataFrame, and
computes `group_by().mean()`. For a 1M-cell × 30K-gene matrix, cell-eval
allocates ~120 GB of f64 dense data; the SCX streaming path holds one shard
(~16K cells) at a time.

**Signature:**
```python
pyscx.accel.pseudobulk_means(
    adata,
    groupby,                    # str: column name in adata.obs
    min_cells_per_group=1,      # int: exclude groups with fewer cells
) -> tuple[np.ndarray, list[str]]
# Returns: (means[P, G] float64, group_names)
```

#### 2. `pyscx.accel.energy_distance()` — new module

The most expensive cell-eval metric. Computes O(N²) pairwise Euclidean
distances per perturbation, 6 distance matrices per perturbation (3 per side).

**Exact algorithm to implement:**
```
Input: two AnnData objects (real, pred), perturbation column, control label
Output: single float (Pearson correlation of per-perturbation e-distances)

1. Extract control cells: ctrl_real[N_ctrl, G], ctrl_pred[N_ctrl, G]
2. Precompute: sigma_ctrl_real = mean(pairwise_euclidean(ctrl_real, ctrl_real))
               sigma_ctrl_pred = mean(pairwise_euclidean(ctrl_pred, ctrl_pred))
3. For each perturbation p (parallelizable with rayon):
     pert_real = real.X[real.obs[pert_col] == p]     # [N_p, G]
     pert_pred = pred.X[pred.obs[pert_col] == p]     # [N_p, G]
     
     e_real[p] = 2 * mean(pairwise_euclidean(pert_real, ctrl_real))
                 - mean(pairwise_euclidean(pert_real, pert_real))
                 - sigma_ctrl_real
     e_pred[p] = 2 * mean(pairwise_euclidean(pert_pred, ctrl_pred))
                 - mean(pairwise_euclidean(pert_pred, pert_pred))
                 - sigma_ctrl_pred
4. Return: pearsonr(e_real, e_pred)
```

**Optimization opportunities:**
- The three pairwise distance calls per side can be fused: iterate cell pairs
  once, accumulate all three sums simultaneously, avoiding 3 separate
  `[N, N]` allocations.
- The control self-distance (sigma) is computed once and reused across all
  perturbations.
- Per-perturbation computation is independent → rayon `par_iter`.
- Distance computation is SIMD-friendly (sum of squared differences).
- Control cells can be shared across perturbations as a read-only reference.

**Rust module:** `scx-accel/src/eval_metrics/edistance.rs`

**Python signature:**
```python
pyscx.accel.energy_distance(
    adata_real,
    adata_pred,
    pert_col="perturbation",    # str: perturbation column in obs
    control="control",          # str: control perturbation label
    metric="euclidean",         # str: distance metric
    embed_key=None,             # str|None: use obsm[embed_key] instead of X
) -> float
```

#### 3. `pyscx.accel.discrimination_score()` — new module

Computes how well each predicted perturbation effect ranks among all real
effects by pairwise distance.

**Exact algorithm to implement:**
```
Input: two AnnData objects (real, pred), perturbation column, control label,
       distance metric (l1/l2/cosine), exclude_target_gene flag
Output: dict[str, float] mapping perturbation → normalized rank [0,1]

1. Compute pseudobulk means for real and pred (reuse pseudobulk_means)
2. Compute effects: real_effects[P, G] = means_real - means_real[ctrl]
                     pred_effects[P, G] = means_pred - means_pred[ctrl]
3. For each perturbation p (parallelizable):
     if exclude_target_gene and not using embeddings:
         mask = all gene indices except gene named p
     else:
         mask = all gene indices
     
     distances[P] = pairwise_distances(
         real_effects[:, mask],           # [P, G']
         pred_effects[p, mask][1, G'],    # [1, G']
         metric
     )    # one distance from pred_p to each real effect
     
     sorted = argsort(distances)
     rank = position of p in sorted order
     score[p] = 1 - rank / P
```

**Key detail:** When `metric` is L1/manhattan/cityblock, cell-eval forces
`embed_key=None` (always uses gene expression, not embeddings).

**Key detail:** `exclude_target_gene=True` (default) removes the column for
the gene whose name matches the perturbation name. This prevents trivially
high scores from the knockdown gene itself dominating the distance.

**Rust module:** `scx-accel/src/eval_metrics/discrimination.rs`

**Python signature:**
```python
pyscx.accel.discrimination_score(
    adata_real,
    adata_pred,
    pert_col="perturbation",
    control="control",
    metric="l1",                    # "l1", "l2", "cosine"
    exclude_target_gene=True,       # exclude gene named after perturbation
    embed_key=None,                 # str|None: use obsm[embed_key] instead of X
) -> dict[str, float]
```

#### 4. `pyscx.accel.knockdown_efficiency()` — new function

Computes per-cell knockdown efficiency and log fold change. The Python code
loops over perturbations, extracting a single gene column from the sparse
matrix per iteration via `X[mask, gene_idx].toarray()`.

**Exact algorithm to implement:**
```
Input: adata with sparse X, perturbation column, control label, eps=1e-8
Output: two float32 arrays of length n_obs (KD efficiency, log deviation)

1. Compute control baseline: μ_control[G] = mean(X[control_cells], axis=0)
2. Build gene name → column index map
3. For each non-control perturbation p:
     gene_idx = gene_name_to_col[p]   # perturbation name matches gene name
     if gene not found: skip (leave NaN)
     cells = obs[pert_col] == p
     x_target = X[cells, gene_idx]    # single column extraction from CSR
     KD[cells] = 1.0 - x_target / (μ_control[gene_idx] + eps)
4. After log1p transformation:
     μ_control_log[G] = log1p(μ_control)    # or recompute from log-space
     For each perturbation p:
         x_log = X_log[cells, gene_idx]
         FC[cells] = x_log - μ_control_log[gene_idx]
```

In SCX's CSR shard format, extracting column `j` from row `i` requires
scanning `indices[indptr[i]..indptr[i+1]]` for value `j`. This is efficient
in Rust with a binary search on sorted indices. The key advantage over Python:
no `.toarray()` allocation per perturbation, and the shard decode + column
extraction can happen in a single pass.

**Rust module:** `scx-accel/src/eval_metrics/knockdown.rs`

**Python signature:**
```python
pyscx.accel.knockdown_efficiency(
    adata,
    pert_col="perturbation",
    control="control",
    eps=1e-8,
) -> None  # writes KnockDownEfficiency and KnockDownGeneFC to adata.obs
```

#### 5. Bulk metric functions (pearson_delta, mse, mae, mse_delta, mae_delta)

These are simple vectorized operations on the pseudobulk means matrix. Once
`pseudobulk_means()` returns the dense `[P, G]` matrix, computing these in
Rust avoids Python per-perturbation loop overhead.

**Exact algorithm for all 5:**
```
Input: means_real[P, G], means_pred[P, G], ctrl_idx (index of control row)
Output: dict[str, float] per perturbation

For pearson_delta:
    delta_real[p] = means_real[p] - means_real[ctrl_idx]
    delta_pred[p] = means_pred[p] - means_pred[ctrl_idx]
    result[p] = pearson(delta_real[p], delta_pred[p])

For mse:     result[p] = mean((means_real[p] - means_pred[p])²)
For mae:     result[p] = mean(|means_real[p] - means_pred[p]|)
For mse_delta: result[p] = mean((delta_real[p] - delta_pred[p])²)
For mae_delta: result[p] = mean(|delta_real[p] - delta_pred[p]|)
```

These can be fused into a single pass over the means matrices.

**Python signature:**
```python
pyscx.accel.perturbation_metrics(
    adata_real,
    adata_pred,
    pert_col="perturbation",
    control="control",
    metrics=["pearson_delta", "mse", "mae", "mse_delta", "mae_delta"],
    embed_key=None,
) -> dict[str, dict[str, float]]
# Returns: {metric_name: {perturbation: value}}
```

### Tier 2: Moderate impact, reuses existing infrastructure

#### 6. `pyscx.accel.clustering_agreement()` — chains existing accelerators

ClusteringAgreement operates on centroid matrices (typically 50–200 rows ×
G columns), which are just pseudobulk means per perturbation. The expensive
parts (kNN graph, Leiden clustering) already exist in SCX.

**What's needed:**
1. Compute centroids using `pseudobulk_means()` (Tier 1.1)
2. Build kNN graph using existing `neighbors()` with `use_rep="X"` on the
   centroid AnnData (small, so CPU HNSW is sufficient)
3. Run Leiden at 7 resolutions using existing `leiden()` — **can parallelize
   across resolutions with rayon** since each resolution is independent
4. Compute AMI/NMI/ARI between real and predicted cluster labels (new, ~50
   lines of Rust implementing the standard information-theoretic formulas)
5. Return best score

The AMI/NMI/ARI implementations are straightforward: build contingency table
from two label vectors, compute entropies and mutual information. Standard
formulas from sklearn's implementations.

**Python signature:**
```python
pyscx.accel.clustering_agreement(
    adata_real,
    adata_pred,
    pert_col="perturbation",
    control="control",
    metric="ami",                     # "ami", "nmi", "ari"
    real_resolution=1.0,
    pred_resolutions=(0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
    n_neighbors=15,
    embed_key=None,
) -> float
```

#### 7. DE result format bridge

SCX's `DiffExpResult` stores results as per-group arrays:
```rust
pub struct DiffExpResult {
    pub group_names: Vec<String>,
    pub names: Vec<Vec<String>>,         // [n_groups][n_genes]
    pub scores: Vec<Vec<f64>>,           // z-scores
    pub pvals: Vec<Vec<f64>>,
    pub pvals_adj: Vec<Vec<f64>>,        // BH-adjusted
    pub logfoldchanges: Vec<Vec<f64>>,   // log2 fold changes
}
```

cell-eval's `DEResults` expects a polars DataFrame with columns:
`(target, feature, fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change)`.

**What's needed:** A conversion function (Python-side or Rust-side) that
reshapes `DiffExpResult` into the expected DataFrame format. This is a
reshaping operation, not a computation — no performance concern, just glue.

### Tier 3: Low impact or impractical — do not implement

| Operation | Reason to skip |
|-----------|---------------|
| **DE DataFrame metrics** (de_spearman_sig, de_direction_match, etc.) | Operate on small polars DataFrames (hundreds of rows). Individual latency <100ms. Polars is already columnar + vectorized. |
| **ROC/PR AUC** | Per-perturbation `average_precision_score` / `roc_curve` on small arrays. sklearn is already efficient. Marginal Rust benefit doesn't justify the integration cost. |
| **Overlap/precision at N** | Sorted top-k intersection on small gene lists. Trivial compute. |
| **Inextrinsic MLP training/eval** | PyTorch neural network. GPU-optimized. Rust has no advantage. |
| **Seurat v3 HVG selection** | Complex statistical method (variance-stabilizing transformation). scanpy's implementation is mature and authoritative. High reimplementation risk for low benefit. |
| **Binomial downsampling** | `rng.binomial(X.data, frac)` on CSR data array. numpy is fast, operation is I/O-bound. |
| **Gene alignment** (`align_to_gene_set`) | Sparse matrix column reindexing. One-time preprocessing, not a bottleneck. |
| **Hydra/CLI orchestration** | Pure Python workflow logic. |

---

## Architecture

### New module structure in scx-accel

```
scx-accel/src/
├── eval_metrics/
│   ├── mod.rs              # re-exports
│   ├── edistance.rs        # energy distance (pairwise + Pearson)
│   ├── discrimination.rs   # discrimination score (L1/L2/cosine)
│   ├── knockdown.rs        # knockdown efficiency + log deviation
│   ├── bulk_metrics.rs     # pearson_delta, mse, mae, mse_delta, mae_delta
│   ├── distances.rs        # shared pairwise distance kernels (euclidean, L1, cosine)
│   └── clustering.rs       # AMI/NMI/ARI scoring
├── pseudobulk.rs           # existing (unchanged)
├── diffexp.rs              # existing (unchanged)
├── pca.rs                  # existing (unchanged)
├── neighbors.rs            # existing (unchanged)
├── umap.rs                 # existing (unchanged)
├── error.rs                # add EvalMetrics variant
└── lib.rs                  # add pub mod eval_metrics
```

### Dependencies to add to scx-accel/Cargo.toml

No new external crate dependencies required. The pairwise distance
computations use standard f64 arithmetic (SIMD via autovectorization). The
Pearson/Spearman correlations are ~20 lines each. AMI/NMI/ARI are ~80 lines
each (contingency table + entropy computation). All parallelism via existing
`rayon` dependency.

### Key Rust types

```rust
// scx-accel/src/eval_metrics/mod.rs

/// Distance metric for pairwise computations.
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    L1,
    Cosine,
}

/// Result from energy_distance computation.
pub struct EDistanceResult {
    /// Per-perturbation e-distances (real side).
    pub d_real: Vec<f64>,
    /// Per-perturbation e-distances (predicted side).
    pub d_pred: Vec<f64>,
    /// Pearson correlation between d_real and d_pred.
    pub correlation: f64,
    /// Perturbation names in order.
    pub pert_names: Vec<String>,
}

/// Result from discrimination_score computation.
pub struct DiscriminationResult {
    /// Per-perturbation normalized rank scores (1.0 = best).
    pub scores: Vec<f64>,
    /// Perturbation names in order.
    pub pert_names: Vec<String>,
}

/// Result from knockdown_efficiency computation.
pub struct KnockdownResult {
    /// Per-cell knockdown efficiency (NaN for control/unmatched cells).
    pub efficiency: Vec<f32>,
    /// Per-cell log fold change (NaN for control/unmatched cells).
    pub log_fc: Vec<f32>,
}

/// Result from bulk perturbation metrics.
pub struct BulkMetricsResult {
    /// Metric name → (perturbation name → value).
    pub metrics: HashMap<String, Vec<f64>>,
    /// Perturbation names in order.
    pub pert_names: Vec<String>,
}
```

### Python binding pattern

Each new function in `pyscx/src/accel.rs` follows the existing pattern:

1. Extract `adata.X` — detect if `ScxBackedSparseDataset`,
   `ScxLazyTransformedDataset`, scipy sparse CSR, or dense numpy
2. Extract `adata.obs[pert_col]` as string array via PyArrow
3. Call the appropriate Rust function
4. Convert result to numpy arrays / Python dicts
5. Write results back to AnnData (for functions like `knockdown_efficiency`)
   or return directly (for metric functions)

For functions taking two AnnData objects (energy_distance, discrimination_score,
perturbation_metrics), the binding extracts from both and passes to Rust.

### Pairwise distance kernel design

The shared `distances.rs` module provides a fused pairwise distance function
that avoids materializing the full `[N, N]` distance matrix when only the
mean is needed:

```rust
/// Compute mean pairwise distance without materializing the full matrix.
/// Returns: sum of all pairwise distances / (n_a * n_b)
pub fn mean_pairwise_distance(
    a: &[f64],         // [N_A × D] row-major
    b: &[f64],         // [N_B × D] row-major (or same as a for self-distance)
    n_a: usize,
    n_b: usize,
    n_dims: usize,
    metric: DistanceMetric,
) -> f64 {
    // Iterate row pairs, accumulate distances without allocating [N_A, N_B]
    // For self-distance (a == b), only compute upper triangle
}

/// Fused e-distance: compute 2*D(X,Y) - D(X,X) - D(Y,Y) in one pass
/// when the mean pairwise distances of Y (sigma_y) is precomputed.
pub fn fused_edistance(
    x: &[f64],         // [N_X × D] perturbation cells
    y: &[f64],         // [N_Y × D] control cells
    n_x: usize,
    n_y: usize,
    n_dims: usize,
    sigma_y: f64,       // precomputed mean self-distance of Y
    metric: DistanceMetric,
) -> f64 {
    let sigma_x = mean_pairwise_distance(x, x, n_x, n_x, n_dims, metric);
    let delta = mean_pairwise_distance(x, y, n_x, n_y, n_dims, metric);
    2.0 * delta - sigma_x - sigma_y
}
```

This avoids the sklearn pattern of allocating an `[N, N]` matrix and calling
`.mean()`. For a perturbation with 500 cells and 500 control cells, this
saves 500² × 8 bytes = 2 MB per distance computation, and avoids 6 such
allocations per perturbation.

---

## What should NOT be reimplemented in Rust

1. **MLP training/eval** (inextrinsic) — PyTorch is the right tool. The MLP
   is a simple 2-layer network that trains in seconds on GPU.
2. **Seurat v3 HVG selection** — Complex statistical method with edge cases.
   scanpy's implementation is authoritative. High risk of subtle divergence.
3. **DE polars DataFrame metrics** — Polars is already columnar and vectorized.
   These metrics run in <100ms on typical data. Rewriting in Rust adds
   complexity for negligible speedup.
4. **ROC/PR AUC** — sklearn's implementations are well-optimized for the small
   per-perturbation arrays these operate on.
5. **Binomial downsampling** — numpy's RNG is fast; single call on CSR `.data`.
6. **Gene alignment** — One-time sparse matrix reindexing, not a bottleneck.
7. **CLI/workflow orchestration** — Pure Python, no compute involved.

---

## Implementation plan

### Phase 1: Pseudobulk means Python binding + bulk metrics

Expose the existing Rust `pseudobulk_aggregate(method=Mean)` to Python and
implement the simple per-perturbation metrics that consume it.

- [x] **1.1** Add `pseudobulk_means` pyfunction to `pyscx/src/accel.rs`
  - Extract `adata.X` (backed SCX, lazy-transformed, scipy CSR, or dense)
  - Extract `adata.obs[groupby]` as `Vec<String>`
  - Call `pseudobulk_aggregate` / `pseudobulk_aggregate_inmemory` with `AggregationMethod::Mean`
  - Return `(numpy[P, G] float64, list[str] group_names)` to Python
  - Handle edge case: `min_cells_per_group` filtering may reduce P
- [x] **1.2** Register `pseudobulk_means` in `pyscx/src/lib.rs` accel submodule
- [x] **1.3** Add `scx-accel/src/eval_metrics/mod.rs` module skeleton with
  `DistanceMetric` enum and re-exports
- [x] **1.4** Implement `scx-accel/src/eval_metrics/bulk_metrics.rs`
  - `pearson_correlation(x: &[f64], y: &[f64]) -> f64` — Welford-style
    single-pass Pearson
  - `compute_bulk_metrics(means_real, means_pred, ctrl_idx, n_perts, n_genes, metrics) -> BulkMetricsResult`
  - Fused loop: iterate perturbations once, compute all requested metrics
    (pearson_delta, mse, mae, mse_delta, mae_delta) per perturbation
- [x] **1.5** Add `perturbation_metrics` pyfunction to `pyscx/src/accel.rs`
  - Calls `pseudobulk_means` for both adata_real and adata_pred
  - Identifies control index in group_names
  - Calls `compute_bulk_metrics` from Rust
  - Returns `dict[str, dict[str, float]]` (metric → perturbation → value)
- [x] **1.6** Register `perturbation_metrics` in `pyscx/src/lib.rs`
- [x] **1.7** Add Rust unit tests in `scx-accel/src/eval_metrics/bulk_metrics.rs`
  - Test Pearson against known values (hand-computed)
  - Test MSE/MAE against known values
  - Test delta variants (subtract control row)
  - Test with single perturbation, many perturbations, zero-variance case
- [x] **1.8** Add Python integration tests in `pyscx/tests/test_eval_metrics.py`
  - Create synthetic AnnData pair with known perturbation effects
  - Verify `pseudobulk_means()` against `polars.group_by().mean()` (cell-eval's method)
  - Verify `perturbation_metrics()` against cell-eval's `pearson_delta`, `mse`,
    `mae`, `mse_delta`, `mae_delta` on same data
  - Test with backed SCX input and in-memory scipy CSR input
  - Test edge cases: single perturbation, empty groups, all-zero expression

### Phase 2: Pairwise distance kernels + energy distance

Build the shared distance infrastructure and implement the most expensive
cell-eval metric.

- [x] **2.1** Implement `scx-accel/src/eval_metrics/distances.rs`
  - `euclidean_distance(a: &[f64], b: &[f64], n_dims: usize) -> f64`
  - `l1_distance(a: &[f64], b: &[f64], n_dims: usize) -> f64`
  - `cosine_distance(a: &[f64], b: &[f64], n_dims: usize) -> f64`
  - `mean_pairwise_distance(a, b, n_a, n_b, n_dims, metric) -> f64` — streaming
    accumulation without `[N, N]` allocation. For self-distance (a == b),
    compute only upper triangle and double
  - `mean_pairwise_distance_self(a, n_a, n_dims, metric) -> f64` — optimized
    self-distance variant
  - All distance functions should use `#[inline]` for autovectorization
- [x] **2.2** Add Rust unit tests for distance kernels
  - Verify euclidean/L1/cosine against hand-computed values
  - Verify `mean_pairwise_distance` against brute-force `[N, N]` matrix mean
  - Verify self-distance symmetry optimization matches full computation
  - Test with 1-dim, high-dim, zero vectors, identical vectors
- [x] **2.3** Implement `scx-accel/src/eval_metrics/edistance.rs`
  - `fused_edistance(x, y, n_x, n_y, n_dims, sigma_y, metric) -> f64`
  - `compute_energy_distance(real_cells, pred_cells, real_groups, pred_groups,
    ctrl_group_real, ctrl_group_pred, pert_names, n_dims, metric) -> EDistanceResult`
    - Precompute sigma_ctrl_real, sigma_ctrl_pred (once each)
    - `rayon::par_iter` over perturbations, compute fused e-distance for each
    - Compute Pearson correlation between d_real and d_pred vectors
  - Input: per-cell dense matrices + group label arrays (u32 indices)
- [x] **2.4** Add Rust unit tests for energy distance
  - Verify against Python `_edistance` implementation on small test data
  - Verify Pearson correlation of e-distance vectors
  - Test with identical real/pred (should give correlation ~1.0)
  - Test with random real/pred (should give correlation ~0.0)
- [x] **2.5** Add `energy_distance` pyfunction to `pyscx/src/accel.rs`
  - Extract `adata_real.X` and `adata_pred.X` as dense numpy (or convert
    from sparse via `.toarray()` — e-distance requires per-cell access,
    not pseudobulk)
  - Extract perturbation labels from both adata.obs
  - If `embed_key` is set, use `adata.obsm[embed_key]` instead of X
  - Build group index arrays (perturbation name → integer index)
  - Call Rust `compute_energy_distance`
  - Return `float` (the Pearson correlation)
- [x] **2.6** Register `energy_distance` in `pyscx/src/lib.rs`
- [x] **2.7** Add Python integration tests for energy distance
  - Verify against cell-eval's `edistance()` on synthetic data
  - Test with different distance metrics
  - Test with embed_key (obsm embeddings)

### Phase 3: Discrimination score

Builds on the pairwise distance kernels from Phase 2 and pseudobulk means
from Phase 1.

- [x] **3.1** Implement `scx-accel/src/eval_metrics/discrimination.rs`
  - `compute_discrimination_score(real_effects, pred_effects, n_perts, n_genes,
    pert_names, gene_names, metric, exclude_target_gene) -> DiscriminationResult`
  - Per-perturbation:
    - Build include_mask (exclude gene column matching perturbation name if
      `exclude_target_gene` is true and not using embeddings)
    - Compute distances from `pred_effects[p]` to all `real_effects[:, mask]`
      using the appropriate metric
    - `argsort` distances, find rank of correct perturbation, normalize
  - `rayon::par_iter` across perturbations (each is independent)
- [x] **3.2** Add Rust unit tests for discrimination score
  - Verify against cell-eval's `discrimination_score()` on small test data
  - Test L1, L2, cosine metrics separately
  - Test `exclude_target_gene` behavior (gene column removed when name matches)
  - Test perfect prediction (score should be 1.0 for all perturbations)
  - Test random prediction (scores should be ~0.5 on average)
- [x] **3.3** Add `discrimination_score` pyfunction to `pyscx/src/accel.rs`
  - Compute pseudobulk means for both adata_real and adata_pred (reuse
    `pseudobulk_means` logic)
  - Compute perturbation effects (subtract control)
  - Extract gene_names from adata.var_names (needed for target gene exclusion)
  - If `embed_key` is set, use obsm instead of X for pseudobulk computation
  - If metric is L1/manhattan/cityblock, force embed_key=None
  - Call Rust `compute_discrimination_score`
  - Return `dict[str, float]`
- [x] **3.4** Register `discrimination_score` in `pyscx/src/lib.rs`
- [x] **3.5** Add Python integration tests
  - Verify against cell-eval's `discrimination_score()` for L1, L2, cosine
  - Test with `exclude_target_gene=True` and `False`
  - Test with `embed_key` (e.g., "X_pca")

### Phase 4: Knockdown efficiency + log deviation

Per-cell metrics from arc-bench that require efficient single-column
extraction from CSR sparse matrices.

- [x] **4.1** Implement `scx-accel/src/eval_metrics/knockdown.rs`
  - `compute_control_baseline(source, group_labels, ctrl_group, n_vars) -> Vec<f64>`
    - Stream shards, accumulate sum for control cells, divide by count
    - For in-memory CSR: iterate control rows directly
  - `compute_knockdown_efficiency(source_or_csr, pert_labels, ctrl_label,
    gene_names, eps) -> KnockdownResult`
    - Build gene_name → column_index map
    - For each non-control perturbation: look up gene index matching
      perturbation name. For each cell of that perturbation, extract
      `X[cell, gene_idx]` via binary search on CSR indices
    - Compute `KD = 1.0 - x / (baseline[gene_idx] + eps)`
    - For backed SCX: stream shards, process cells within each shard
  - `compute_log_deviation(source_or_csr, pert_labels, ctrl_label,
    gene_names, baseline_log) -> Vec<f32>`
    - Same column extraction pattern
    - Compute `FC = x_log - baseline_log[gene_idx]`
    - Expects already log1p-transformed input (matches arc-bench pipeline order)
- [x] **4.2** Add Rust unit tests for knockdown efficiency
  - Hand-crafted CSR with known values; verify KD and FC against manual calc
  - Test missing gene (perturbation name not in var_names → NaN)
  - Test control cells (should remain NaN)
  - Test streaming (multi-shard backed reader) matches in-memory
- [x] **4.3** Add `knockdown_efficiency` pyfunction to `pyscx/src/accel.rs`
  - Extract `adata.X` (backed or in-memory)
  - Extract `adata.obs[pert_col]` and `adata.var_names`
  - Call Rust function
  - Write `adata.obs["KnockDownEfficiency"]` = efficiency array
  - Write `adata.obs["KnockDownGeneFC"]` = log_fc array
- [x] **4.4** Register in `pyscx/src/lib.rs`
- [x] **4.5** Add Python integration tests
  - Verify against arc-bench's `compute_knockdown_efficiency()` and
    `compute_log_deviation()` on synthetic data
  - Test with backed SCX and in-memory scipy CSR

### Phase 5: Clustering agreement

Chains existing SCX accelerators with new AMI/NMI/ARI scoring.

- [x] **5.1** Implement `scx-accel/src/eval_metrics/clustering.rs`
  - `build_contingency_table(labels_a: &[u32], labels_b: &[u32], n_a: u32,
    n_b: u32) -> Vec<u64>` — `[n_a × n_b]` count matrix
  - `adjusted_mutual_info(labels_a, labels_b) -> f64`
  - `normalized_mutual_info(labels_a, labels_b) -> f64`
  - `adjusted_rand_index(labels_a, labels_b) -> f64` — then rescale
    `(score + 1) / 2` to match cell-eval convention
  - All implementations follow sklearn formulas
- [x] **5.2** Add Rust unit tests for clustering metrics
  - Verify AMI/NMI/ARI against sklearn on known label pairs
  - Test perfect agreement (score = 1.0)
  - Test random labels
  - Test single-cluster edge case
- [x] **5.3** Add `clustering_agreement` pyfunction to `pyscx/src/accel.rs`
  - Compute centroids using `pseudobulk_means()` for real and pred
  - Exclude control perturbation from centroid sets
  - Sort centroids by perturbation name (to align between real and pred)
  - Build kNN graph on real centroids via `sc.pp.neighbors()` (Python-side,
    since these are small matrices ~50–200 rows)
  - Run Leiden at fixed resolution for real
  - For each of 7 pred resolutions: run Leiden, compute scoring metric
  - Return best score
  - Note: Leiden and kNN can stay on the Python side (calling into scanpy/igraph)
    since the centroid matrices are small. The Rust side provides the
    AMI/NMI/ARI scoring.
- [x] **5.4** Register in `pyscx/src/lib.rs`
- [x] **5.5** Add Python integration tests
  - Verify against cell-eval's `ClusteringAgreement` on synthetic data
  - Test all three metrics (ami, nmi, ari)
  - Test different resolution sweeps

### Phase 6: DE result format bridge + integration

Connect SCX's existing Wilcoxon DE to cell-eval's DE metric pipeline.

- [x] **6.1** Add `de_results_to_dataframe` helper in `pyscx/src/accel.rs`
  - Convert `DiffExpResult` (per-group arrays) to a flat polars DataFrame with
    columns matching cell-eval's `DEResults` schema:
    `(target, feature, fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change)`
  - `target` = group name (perturbation)
  - `feature` = gene name
  - `fold_change` = 2^(logfoldchanges) (convert from log2 to linear)
  - `p_value` = pvals
  - `fdr` = pvals_adj
  - `log2_fold_change` = logfoldchanges
  - `abs_log2_fold_change` = |logfoldchanges|
- [x] **6.2** Expose as `pyscx.accel.rank_genes_groups_df()` — same as
  existing `rank_genes_groups()` but returns a polars DataFrame in cell-eval
  format instead of pandas DataFrame in scanpy format
- [x] **6.3** Register in `pyscx/src/lib.rs`
- [x] **6.4** Add Python integration tests
  - Run SCX DE + format conversion on synthetic data
  - Feed result into cell-eval's DE metrics (de_spearman_sig, etc.)
  - Verify metrics produce valid outputs (not that values match pdex exactly,
    since the underlying statistical tests may differ)
- [x] **6.5** Add end-to-end integration test
   - Synthetic perturbation dataset (real + pred AnnDatas)
   - Run full pipeline: pseudobulk_means → perturbation_metrics →
     discrimination_score → energy_distance → knockdown_efficiency
   - Verify all outputs match cell-eval reference implementations within
     numerical tolerance (1e-6 for Pearson/MSE, 1e-4 for e-distance)

### Phase 7: End-to-end correctness validation vs cell-eval / arc-bench

Validate that the complete SCX-accelerated metric pipeline produces
results numerically equivalent to the Python `cell-eval` and `arc-bench`
reference implementations. This is the final gate: every Tier 1 metric
must match the Python reference within specified tolerance on the **same**
input data before the Rust accelerators can replace the Python codepath.

**External repos:**
- `cell-eval` — `/home/nickyoungblut/dev/python/cell-eval/` (v0.7.0)
- `arc-bench` — `/home/nickyoungblut/dev/python/arc-bench/`

**Key cell-eval entry points:**
- `cell_eval.MetricsEvaluator` — orchestrates DE computation (via `pdex`)
  and all AnnData-pair metrics via `MetricPipeline`
- `cell_eval.metrics.metrics_registry` — 24 registered metrics in
  `_impl.py`, split into `MetricType.DE` and `MetricType.ANNDATA_PAIR`
- `cell_eval.data.build_random_anndata()` — synthetic data factory
  (1000 cells × 100 genes, 10 perturbations + control, optional sparse)
- `cell_eval._score.score_agg_metrics()` — per-metric normalized scoring
  against a baseline (norm-by-zero for MSE/MAE, norm-by-one for Pearson)

**Key arc-bench entry points:**
- `arc_bench.tools.pert_eval.cli._run_standard()` — splits by context,
  clips to [0, 14], runs `MetricsEvaluator` per cell type
- `arc_bench.tools.normalize_transform.core` —
  `compute_knockdown_efficiency()`, `compute_log_deviation()`,
  `compute_control_baseline()` (Python reference implementations)

#### 7.1 Environment setup

Following the established pattern in `benchmarks/comprehensive/envs/`,
create a dedicated conda environment for cell-eval / arc-bench parity
testing. This keeps validation deps (pdex, polars, cell-eval, arc-bench)
isolated from the main benchmark and development environments.

- [x] **7.1.1** Create `benchmarks/comprehensive/envs/scx-bench-eval.yml`:
  ```yaml
  # SCX Benchmark Suite — cell-eval / arc-bench Parity Validation
  #
  # Create:   conda env create -f benchmarks/comprehensive/envs/scx-bench-eval.yml
  # Activate: conda activate scx-bench-eval
  # Build pyscx: cd pyscx && maturin develop --release && cd ..
  #
  name: scx-bench-eval
  channels:
    - conda-forge
    - defaults
  dependencies:
    - python=3.13
    # Core single-cell stack (matches scx-bench)
    - anndata>=0.12.7
    - scanpy>=1.12
    - scipy>=1.16
    - numpy>=2.2
    - pandas>=2.3
    - h5py>=3.16
    # Analysis (needed by cell-eval and clustering agreement tests)
    - scikit-learn>=1.8
    - python-igraph>=1.0
    - leidenalg>=0.11
    # Data handling
    - pyarrow>=18.0
    # Build tooling (pyscx from source)
    - maturin>=1.5.0
    # System monitoring
    - psutil>=7.0
    - pip:
      # cell-eval and arc-bench (editable from local repos)
      - -e /home/nickyoungblut/dev/python/cell-eval
      - -e /home/nickyoungblut/dev/python/arc-bench
      # cell-eval's transitive deps
      - pdex>=0.2.0
      - polars>=1.30.0
      - tqdm>=4.67
  ```
- [x] **7.1.2** Add `--eval` flag to `install_dependencies.sh`:
  ```bash
  bash benchmarks/comprehensive/scripts/install_dependencies.sh --eval
  ```
  This should create the `scx-bench-eval` conda env and build `pyscx`
  inside it (release mode, no GPU features needed).
- [x] **7.1.3** Create `pyscx/tests/test_cell_eval_parity.py` — the main
  validation test file. All tests in this file import both `pyscx.accel`
  and `cell_eval` / `arc_bench` to compare outputs head-to-head.
  Tests should be run from within the `scx-bench-eval` conda env:
  ```bash
  conda activate scx-bench-eval
  cd pyscx && maturin develop --release && cd ..
  pytest pyscx/tests/test_cell_eval_parity.py -v
  ```
- [x] **7.1.4** Add a SLURM script
  `benchmarks/comprehensive/scripts/slurm_eval_parity.sh` for running
  the parity tests on the cluster:
  - Partition: `cpu_preemptible`
  - Resources: 16 CPUs, 32 GB (synthetic data is small)
  - Activates `scx-bench-eval` conda env
  - Rebuilds pyscx in release mode
  - Runs `pytest pyscx/tests/test_cell_eval_parity.py -v`

#### 7.2 Shared synthetic dataset

- [x] **7.2.1** Create a `_make_cell_eval_adata()` helper that produces
  data consumable by both SCX and cell-eval:
  ```python
  def _make_cell_eval_adata(
      n_obs=500, n_vars=100, n_perts=8, seed=42, as_sparse=True,
  ) -> tuple[ad.AnnData, ad.AnnData]:
      """Paired real/pred AnnData matching cell-eval conventions.

      - obs column: "perturbation" (matches CANONICAL_PERTURBATION_COL)
      - control label: "control" (matches CANONICAL_CONTROL_LABEL)
      - Perturbation names match gene names (gene_0..gene_{n_perts-2})
        so knockdown_efficiency can look up target genes
      - Normalize-total + log1p applied (cell-eval expects lognorm input)
      - Predicted data = real + Gaussian noise (correlated but imperfect)
      """
  ```
  Key constraints:
  - Gene names must match perturbation names for `n_perts - 1` entries
    (excluding control) — required for knockdown/discrimination target-gene
    exclusion tests
  - Data must pass `cell_eval.utils.guess_is_lognorm()` (values in
    [0, 15), has fractional component)
  - At least 20 cells per perturbation (required for stable pseudobulk)
  - Both real and pred must share the same `var_names` (cell-eval validates
    this in `PerturbationAnndataPair.__init__`)

#### 7.3 Pseudobulk means parity

- [x] **7.3.1** Test: `test_pseudobulk_vs_cell_eval()`
  - Compute pseudobulk via `pyscx.accel.pseudobulk_means(adata, "perturbation")`
  - Compute pseudobulk via cell-eval's `_bulk_anndata()` method:
    ```python
    from cell_eval._types._anndata import PerturbationAnndataPair
    pair = PerturbationAnndataPair(
        real=adata_real, pred=adata_real,
        control_pert="control", pert_col="perturbation",
    )
    # Triggers polars group_by().mean() pseudobulk
    bulk = pair.bulk_real  # dict[str, tuple[NDArray, NDArray]]
    ```
  - Assert: per-perturbation mean vectors match within `atol=1e-6`
  - Also verify group ordering matches (cell-eval sorts by group name)

#### 7.4 Bulk perturbation metrics parity

- [x] **7.4.1** Test: `test_pearson_delta_vs_cell_eval()`
  - Run `pyscx.accel.perturbation_metrics(real, pred, metrics=["pearson_delta"])`
  - Run `cell_eval.metrics.pearson_delta(pair)` where `pair` is a
    `PerturbationAnndataPair`
  - Assert: per-perturbation Pearson values match within `atol=1e-6`
- [x] **7.4.2** Test: `test_mse_mae_vs_cell_eval()`
  - Same pattern for `mse`, `mae`, `mse_delta`, `mae_delta`
  - Assert: `atol=1e-6`
- [x] **7.4.3** Test: `test_perturbation_metrics_agg_vs_cell_eval()`
  - Compare aggregated (mean across perturbations) metrics
  - This is what `cell_eval.MetricPipeline.get_agg_results()` returns
  - Assert: mean/std match within `atol=1e-5`

#### 7.5 Energy distance parity

- [x] **7.5.1** Test: `test_edistance_vs_cell_eval()`
  - Run `pyscx.accel.energy_distance(real, pred, pert_col="perturbation", control="control")`
  - Run `cell_eval.metrics.edistance(pair)` (returns Pearson correlation
    of per-perturbation e-distances)
  - Assert: `atol=1e-4` (e-distance involves O(N²) accumulations with
    potential ordering-dependent floating-point summation)
- [x] **7.5.2** Test: `test_edistance_intermediate_values()`
  - Exposed via the new `pyscx.accel.energy_distance_details()` binding,
    which returns `{correlation, d_real, d_pred, pert_names}`.
  - Per-perturbation `d_real` / `d_pred` values verified against a
    direct `scipy.spatial.distance.cdist` reference within `atol=1e-4`.

#### 7.6 Discrimination score parity

- [x] **7.6.1** Test: `test_discrimination_score_l1_vs_cell_eval()`
  - Run `pyscx.accel.discrimination_score(real, pred, metric="l1")`
  - Run `cell_eval.metrics.discrimination_score(pair, metric="l1")`
  - Assert: per-perturbation rank scores match **exactly** (these are
    integer-rank-based, so should be bit-identical unless tie-breaking
    differs)
- [x] **7.6.2** Test: `test_discrimination_score_l2_cosine_vs_cell_eval()`
  - Same for L2 and cosine metrics
- [x] **7.6.3** Test: `test_discrimination_target_exclusion_parity()`
  - Verify `exclude_target_gene=True` behavior: SCX must exclude the
    same gene column that cell-eval excludes (matched by perturbation
    name → gene name)
  - Compare with `exclude_target_gene=False` to confirm the delta

#### 7.7 Knockdown efficiency parity

- [x] **7.7.1** Test: `test_knockdown_vs_arc_bench()`
  - Create raw-count AnnData (NOT log1p) with known perturbation names
    matching gene names
  - Compute control baseline:
    - SCX: via `pyscx.accel.knockdown_efficiency(adata, ...)`
    - arc-bench: via `arc_bench.tools.normalize_transform.core.compute_control_baseline()`
      + `compute_knockdown_efficiency()`
  - Assert: per-cell knockdown efficiency arrays match within `atol=1e-6`
- [x] **7.7.2** Test: `test_log_deviation_vs_arc_bench()`
  - Apply normalize_total + log1p, then compare log deviation:
    - SCX: `adata.obs["KnockDownGeneFC"]` (written by knockdown_efficiency)
    - arc-bench: `compute_log_deviation()`
  - Assert: per-cell FC arrays match within `atol=1e-6`
- [x] **7.7.3** Test: `test_knockdown_missing_gene()`
  - Perturbation name not in `var_names` — both implementations should
    produce NaN for those cells
  - Verify NaN positions match exactly

#### 7.8 Clustering agreement parity

- [x] **7.8.1** Test: `test_clustering_agreement_vs_cell_eval()`
  - Run `pyscx.accel.clustering_agreement(real, pred, metric="ami")`
  - Run `cell_eval.metrics.ClusteringAgreement(metric="ami")(pair)`
  - **Note**: Exact score match not expected due to stochastic Leiden.
    Assert: scores agree within `atol=0.15` or both return the same best
    resolution. Verify the AMI/NMI/ARI **scoring functions** independently
    on identical label vectors (these must match exactly via
    `sklearn.metrics` cross-check).
- [x] **7.8.2** Test: `test_clustering_scoring_functions_vs_sklearn()`
  - Generate known label pairs
  - Verify `pyscx.accel.adjusted_mutual_info()` matches
    `sklearn.metrics.adjusted_mutual_info_score()` within `atol=1e-10`
  - Same for NMI and ARI (noting cell-eval's `(ARI + 1) / 2` rescaling)

#### 7.9 DE result format bridge parity

- [x] **7.9.1** Test: `test_de_dataframe_format()`
  - Run `pyscx.accel.rank_genes_groups_df(adata, "perturbation", ...)`
  - Verify output DataFrame has the required cell-eval columns:
    `(target, feature, fold_change, p_value, fdr, log2_fold_change,
     abs_log2_fold_change)`
  - Verify column types: `target` and `feature` are `Utf8`, numeric
    columns are `Float64`
- [x] **7.9.2** Test: `test_de_bridge_feeds_cell_eval_metrics()`
  - Compute DE via SCX, convert to cell-eval format
  - Feed the resulting `DEComparison` into cell-eval's DE metrics
    (`de_spearman_sig`, `overlap_at_N`, `precision_at_N`, `pr_auc`, etc.)
  - Assert: all metrics return valid float values (no NaN/Inf/errors)
  - **Note**: Exact metric value match not required because SCX uses
    Wilcoxon rank-sum while cell-eval defaults to `pdex` (which uses
    a different statistical test). The goal is format compatibility, not
    statistical equivalence.

#### 7.10 Full pipeline integration test

- [x] **7.10.1** Test: `test_full_pipeline_vs_cell_eval()`
  - Run the complete cell-eval `MetricsEvaluator.compute(profile="anndata")`
    pipeline on the shared synthetic dataset
  - Run the equivalent SCX pipeline:
    ```python
    results_scx = {}
    results_scx["pseudobulk"] = pyscx.accel.pseudobulk_means(...)
    results_scx["bulk_metrics"] = pyscx.accel.perturbation_metrics(...)
    results_scx["discrimination_l1"] = pyscx.accel.discrimination_score(..., metric="l1")
    results_scx["discrimination_l2"] = pyscx.accel.discrimination_score(..., metric="l2")
    results_scx["discrimination_cosine"] = pyscx.accel.discrimination_score(..., metric="cosine")
    results_scx["edistance"] = pyscx.accel.energy_distance(...)
    results_scx["clustering"] = pyscx.accel.clustering_agreement(...)
    ```
  - Compare per-perturbation results for every `MetricType.ANNDATA_PAIR`
    metric that SCX implements
  - Tolerance table:
    | Metric | Tolerance | Rationale |
    |--------|-----------|-----------|
    | pearson_delta | 1e-6 | Deterministic, same algorithm |
    | mse / mae | 1e-6 | Deterministic |
    | mse_delta / mae_delta | 1e-6 | Deterministic |
    | discrimination_score_* | 0 (exact) | Integer rank, deterministic |
    | pearson_edistance | 1e-4 | O(N²) float accumulation |
    | clustering_agreement | 0.15 | Stochastic Leiden |

- [x] **7.10.2** Test: `test_full_pipeline_arc_bench_cli_parity()`
  - Simulate the `arc_bench.tools.pert_eval.cli._run_standard()` workflow:
    1. Clip X to [0, 14]
    2. Split by context column
    3. Per context: run MetricsEvaluator + SCX pipeline
  - Verify that the SCX pathway produces equivalent per-context results
  - This validates the end-to-end arc-bench integration, not just
    individual metrics

#### 7.11 Performance comparison

- [x] **7.11.1** Test: `test_performance_vs_cell_eval()`
  - Create a larger synthetic dataset (10K cells × 2K genes × 50 perts)
  - Time both implementations for:
    - Pseudobulk means
    - pearson_delta + mse + mae + mse_delta + mae_delta (bundled)
    - discrimination_score_l1
    - energy_distance (if feasible at this scale, else 5K × 500)
  - Print speedup ratios (SCX / cell-eval wall time)
  - No assertion on speedup — this is informational only, but should
    document the expected improvement factors
- [x] **7.11.2** Memory profiling (manual / optional) — deferred; not
  gating. The performance test already records wall-clock. Peak RSS
  comparison is tracked under the broader benchmark suite in
  `benchmarks/comprehensive/`.
  - Run both pipelines under `tracemalloc` or `/usr/bin/time -v`
  - Document peak RSS for cell-eval vs SCX on the 10K-cell dataset
  - Expected: SCX should use significantly less memory due to streaming
    pseudobulk and fused distance computation (no `[N, N]` allocation)

#### 7.12 Scoring parity

- [x] **7.12.1** Test: `test_score_agg_metrics_parity()`
  - Run `cell_eval.score_agg_metrics()` on aggregated results from both
    the SCX pipeline and the cell-eval pipeline
  - Verify that the normalized scores (norm-by-zero for MSE/MAE,
    norm-by-one for Pearson/discrimination) are numerically equivalent
  - This validates that SCX metric outputs are compatible with cell-eval's
    standard scoring workflow

## Results

### Correctness parity (7.3–7.10, 7.12)

`pytest pyscx/tests/test_cell_eval_parity.py` — **30 passed, 0 skipped**
(2026-04-16, local .venv with cell-eval v0.7.0 + arc-bench editable).

All metrics match the cell-eval / arc-bench references within the
tolerance table in §7.10:

| Metric                    | Tolerance  | Observed | Notes                              |
|---------------------------|------------|----------|------------------------------------|
| pseudobulk_means          | `atol=1e-6` | ✅       | vs `PerturbationAnndataPair._bulk_anndata` |
| pearson_delta             | `atol=1e-6` | ✅       |                                    |
| mse / mae                 | `atol=1e-6` | ✅       |                                    |
| mse_delta / mae_delta     | `atol=1e-6` | ✅       |                                    |
| pearson_edistance         | `atol=1e-4` | ✅       |                                    |
| discrimination_score (l1/l2/cosine) | exact | ✅   | integer-rank, bit-identical to cell-eval |
| knockdown_efficiency      | `atol=1e-6` | ✅       | vs `arc_bench.compute_knockdown_efficiency` |
| log_deviation             | `atol=1e-6` | ✅       | vs `arc_bench.compute_log_deviation` |
| AMI / NMI / ARI           | `atol=1e-10` | ✅      | vs `sklearn.metrics.*` (ARI uses cell-eval's `(ARI+1)/2`) |
| clustering_agreement      | `atol=0.15` | ✅       | loose tolerance, stochastic Leiden |

### Performance (§7.11.1)

The one-shot `test_performance_vs_cell_eval` in
`pyscx/tests/test_cell_eval_parity.py` provides a 10K-cell sanity
snapshot. The authoritative scaling numbers come from the new
`cell_eval_parity_perf` benchmark in the comprehensive suite
(`benchmarks/comprehensive/benchmarks/cell_eval_parity_perf.py`), which
runs the same operations on synthetic perturbation datasets at
100K / 500K / 1M cells with warmup + 3 repeats and a fresh
`PerturbationAnndataPair` per reference call (to neutralize the
pseudobulk cache that would otherwise make the reference look
artificially fast). Full JSON at
`benchmarks/comprehensive/results/raw/cell_eval_parity_perf__scx_auto__pert_synth_*.json`;
rendered table in the comprehensive `BENCHMARK_REPORT.md`.

Speedup (SCX vs cell-eval / arc-bench), single node (Intel Xeon Platinum
8468, 32 CPUs allocated, 192 logical host):

| Operation | 10K | 100K | 500K | 1M |
|---|---:|---:|---:|---:|
| Pseudobulk means | 7.8× | 11.6× | 13.8× | **19.4×** |
| Bulk metrics (5 bundled) | 9.1× | 12.1× | 13.6× | **21.9×** |
| Discrimination score (L1) | 8.1× | 12.0× | 12.9× | **20.1×** |
| Energy distance | 4.0× | **14.4×** | skipped¹ | skipped¹ |
| Clustering agreement (AMI) | 4.9× | 7.6× | **24.6×** | 10.0× |
| Knockdown efficiency + log deviation | 0.6× | 0.9× | **1.3×** | 0.7× |

¹ `energy_distance` is skipped at ≥500K — at 100K the reference's
`sklearn.metrics.pairwise_distances` path already runs ~18 s per
perturbation × 49 perts × 4 measurement runs = ~60 min per dataset;
projecting to N² at 500K+ would require ~24 h for the reference alone.

Notes:
- **Pseudobulk-driven metrics** (pseudobulk, bulk_metrics, discrimination)
  scale cleanly — Rust's single-pass streaming aggregation wins harder
  as cell count grows, reaching ~20× at 1M.
- **Discrimination score at 10K appeared slow in the earlier snapshot**
  (0.4×) because that measurement used cell-eval with a warm pseudobulk
  cache. With a fair cold-start reference, SCX is 8.1× at 10K and
  ~12–20× at larger scales.
- **Knockdown efficiency** is within ±40% of arc-bench's tight NumPy
  column-access loop and is not currently a speedup target.
- **Clustering agreement** variability (10×–25× range) comes from the
  stochastic Leiden sweep over 7 resolutions.

### Phase 7 status

All gating correctness tasks (7.1–7.10, 7.12) are green. 7.11.2
(tracemalloc/RSS profiling) is deferred to the broader benchmark suite.
