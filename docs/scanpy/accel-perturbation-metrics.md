# Perturbation evaluation metrics (cell-eval parity)

> Part of the [SCX + scanpy guide](README.md). Shared conventions are on the
[accelerators overview](accelerators.md).

SCX ships Rust-accelerated equivalents of the metrics in
[`cell-eval`](https://github.com/arcinstitute/cell-eval). The outputs are
numerically equivalent to the Python references within the tolerances
below, so an existing cell-eval pipeline can swap in `pyscx.accel.*` for 10–20×
wall-clock speedup at census-scale perturbation datasets (see
[`docs/performance/perturbation-metrics.md`](../performance/perturbation-metrics.md#perturbation-metrics-cell-eval-parity)
for numbers at 10K / 100K / 500K / 1M cells).

**How each tolerance is gated.** Every row below is pinned twice:

- **Rust-side**, against reference values `cell-eval`
  *produced*, checked into `scx-accel/src/eval_metrics/cell_eval_reference_values.rs`
  and asserted by `cell_eval_reference_tests.rs`. This runs under plain
  `cargo test` with no Python installed, and it is what makes these claims
  reproducible. Regenerate with
  `.venv/bin/python benchmarks/scripts/generate_eval_metrics_references.py`.
- **Python-side**, by `pyscx/tests/test_cell_eval_parity.py` against the live
  libraries. That file `importorskip`s `cell_eval` / `polars`,
  which are editable installs in the repo's `.venv` and are in **no** conda env
  and not in CI — so it strengthens the local gate and is not on its own
  evidence for anything here.

The two clustering rows are Python-side only: `clustering_agreement` wraps a
stochastic Leiden, and AMI / NMI / ARI are pinned against sklearn in
`eval_metrics/clustering.rs` rather than against cell-eval.

| Metric | Tolerance | Rationale |
|---|---|---|
| AMI / NMI / ARI on label vectors | `atol=1e-10` | Integer-label inputs; limited by double-precision floor (~2.2e-16). |
| pseudobulk_means, pearson_delta, mse/mae (and `_delta` variants), knockdown_efficiency, log_deviation | `atol=1e-6` **plus** `rtol=1e-7` | f32 CSR promoted to f64 before accumulation; expected rounding `O(n_cells · 2⁻²³) ≈ 1e-7` at 1M cells. The relative term is not new: the parity tests write `np.testing.assert_allclose(…, atol=1e-6)`, and **numpy's default `rtol` is `1e-7`**, so this has always been the enforced bar. The absolute half alone cannot be the whole claim — cell-eval stores these in f32, so the divergence scales with the value: `mse_delta` of `23.0172` differs by `1.25e-6`, inside numpy's bar and outside a bare `atol=1e-6`. |
| energy_distance / pearson_edistance | `atol=1e-4`, correlation and per-pert alike | O(N²) pairwise reduction; faer-gemm reduction order differs from sklearn BLAS GEMM. f32 + gemm matches f64 + scalar within these bounds (test parametrised over both dtypes). |
| clustering_agreement (AMI over Leiden sweep) | `atol=0.15` aggregate | Native-Rust kNN (HNSW) + Leiden replaces scanpy under the hood; the two algorithms produce within-permutation labels on graphs with `n_perts ≥ 16` (parity test scaffold uses `n_perts=30`). |
| discrimination_score rank | exact (`abs=0`) on untied distances; **empirically** exact on totally-tied ones (tested, not guaranteed — the reference's sort is unstable); not claimed on mixed ties | Integer rank computation; any non-zero diff on untied input is a correctness regression, and on a total tie it means numpy's tie order moved. Exactness requires matching the reference on duplicated gene symbols and zero-norm effect vectors, both of which diverged through v0.13.0 — and the parity fixture (400×20 continuous random) contains neither, so it did not see them. **Mixed ties are explicitly out of scope**: the reference's order comes from an unstable `np.argsort` and is not reproducible. See [Discrimination score](#discrimination-score-pyscxacceldiscrimination_score). |

All functions accept in-memory, backed, or lazy-transformed inputs. They
expect the `cell-eval` data conventions: an `obs` column with
perturbation labels, a designated control label, and — for the knockdown
and discrimination metrics — perturbation names that match gene names in
`var_names` so the target gene can be looked up.

## Pseudobulk means (`pyscx.accel.pseudobulk_means`)

Group-by mean on sparse `X`. Foundation for the pairwise metrics below.

```python
means, groups = pyscx.accel.pseudobulk_means(adata, "perturbation")
# means.shape == (n_perturbations, n_genes), dtype float64
# groups == ["control", "drug_A", "drug_B", ...]  (sorted)

# Several columns key by the per-cell tuple (the `str | list[str]` groupby
# `pseudobulk_dex` takes); group names are then tuples, one entry per column:
means, groups = pyscx.accel.pseudobulk_means(adata, ["perturbation", "donor"])
# groups == [("control", "d1"), ("control", "d2"), ("drug_A", "d1"), ...]
```

Streams directly from CSR shards with no full-matrix materialization. On
backed data, processes shard-by-shard; on lazy-transformed data, applies
the transform stack before aggregation. **Dense fast-path**: when
`adata.X` is a dense numpy array (the common shape after
`pp.normalize_total + log1p`), the in-memory aggregation
runs through `scx_accel::pseudobulk_aggregate_dense` directly, bypassing
the historical `scipy.sparse.csr_matrix(dense_array)` round-trip. At
24K-cell × 18K-gene shapes this cut `pseudobulk_means` from ~22 s to ~5 s.

## Bulk perturbation metrics (`pyscx.accel.perturbation_metrics`)

Pearson of the perturbation→control delta plus MSE/MAE — the five metrics
`cell-eval` computes on pseudobulked pairs, bundled into a single pass:

```python
results = pyscx.accel.perturbation_metrics(adata_real, adata_pred)
# {
#   "pearson_delta": {"drug_A": 0.95, ...},
#   "mse":          {"drug_A": 0.12, ...},
#   "mae":          {"drug_A": 0.08, ...},
#   "mse_delta":    {...},
#   "mae_delta":    {...},
# }

# Pick a subset:
results = pyscx.accel.perturbation_metrics(
    adata_real, adata_pred, metrics=["pearson_delta", "mse"],
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pert_col` | `"perturbation"` | `obs` column containing perturbation labels |
| `control` | `"control"` | Control label |
| `metrics` | all 5 | Subset of `{pearson_delta, mse, mae, mse_delta, mae_delta}` |
| `min_cells_per_group` | 1 | Skip perturbations with fewer cells |
| `device` | `"auto"` | `"auto"`/`"cpu"`/`"gpu"`/`"gpu:N"` — GPU runs the per-group pseudobulk means on the device (f64), bulk metrics on the host; CPU parity `atol≈1e-6`. Route `gpu_csr` / `cpu_csr`. |

## Discrimination score (`pyscx.accel.discrimination_score`)

For each perturbation, ranks how well the predicted effect matches the
correct real effect among all perturbations by pairwise distance. Returns
a normalized rank in `[0, 1]` where 1 = correct perturbation is the closest
match, 0 = furthest.

```python
scores = pyscx.accel.discrimination_score(
    adata_real, adata_pred, metric="l1",  # "l1" | "l2" | "cosine"
)
# scores["drug_A"] == 0.96
```

With `exclude_target_gene=True` (default), **every** gene column matching a
perturbation's name is dropped from that perturbation's distance — prevents
trivially high scores from knockdown-gene dominance and matches cell-eval's
`np.flatnonzero(genes != p)`. "Every" is load-bearing: `var_names` are not unique
in practice (10x matrices routinely repeat a gene symbol), and leaving one copy
of the target column in place restores exactly the self-match the flag exists to
remove. Through v0.13.0 inclusive one copy did survive.

**Ties break by ascending index** — a deterministic, stable rule. Also fixed in
v0.14.0: the rank was previously the number of *strictly* smaller distances, which
is the position of the first tied element rather than of the perturbation being
scored. The visible symptom was at the extreme — a model whose predicted effects
cannot separate its perturbations at all made every distance tie, and scored a
perfect 1.0 on every perturbation instead of `1 - p/P`.

⚠️ **On a tie, SCX does not claim bit-parity with cell-eval, and cannot.** The
reference ranks by `np.argsort`, whose default kind is `quicksort` — not a stable
sort — so its tie order is implementation-defined. Measured on numpy 2.4.4:

| distances | `np.argsort` (default) | `kind="stable"` | |
|---|---|---|---|
| `[5, 5, 5]` | `[0, 1, 2]` | `[0, 1, 2]` | agrees |
| `[1, 1, 2]` | `[0, 1, 2]` | `[0, 1, 2]` | agrees |
| `[3, 3, 1, 1]` | `[3, 2, 1, 0]` | `[2, 3, 0, 1]` | **differs** |

On a **total** tie the two readings coincide *on every numpy measured so far*,
and there SCX matches the reference exactly — but that is an empirical
compatibility point, not a guarantee anyone owes you. An unstable sort has no
contract to preserve the order of equal keys, including when every key is equal,
so a future numpy could change it without breaking any promise.
`TestDiscriminationTieParity` pins it at `abs=0` against the installed cell-eval,
which means a change is caught rather than assumed away.

On a **mixed** tie — some distances equal, some not — the two readings *may*
diverge and are not guaranteed to agree: `[1, 1, 2]` happens to agree,
`[3, 3, 1, 1]` does not, and which one you get depends on introsort internals
rather than on anything you can predict from the data. Where it diverges the
scores differ outright: for `[3, 3, 1, 1]` SCX scores `[0.50, 0.25, 1.00, 0.75]`
and cell-eval on numpy 2.4.4 scores `[0.25, 0.50, 0.75, 1.00]`.

So the contract has one guaranteed half and one observed half. **Guaranteed:**
SCX's own rule — ties break by ascending index, deterministically, on every
platform and version. **Observed, and tested rather than promised:** that this
coincides with cell-eval on untied and totally-tied distances for the numpy
versions exercised. **Not claimed at all:** mixed ties, whether or not a
particular one happens to agree.

SCX keeps the stable rule deliberately. Reproducing the reference would mean
reimplementing NumPy's introsort and would break on any release that touched it,
whereas a documented stable rule is reproducible across versions, platforms and
languages. If you need scores that track a specific cell-eval run tie-for-tie,
compare against that run directly rather than relying on this metric's tie order.

A zero-norm effect vector under `metric="cosine"` is distance `1.0` — maximally
distant — not `0.0`. This is sklearn's `cosine_distances` convention and is now
shared with every other distance in `scx-accel`; the masked path used to return
`0.0`, which undercut every genuine distance and stole rank 0.

## Energy distance (`pyscx.accel.energy_distance`)

Per-perturbation e-distance between perturbation cells and control cells on
both real and predicted sides, returning the Pearson correlation of the
two e-distance vectors.

```python
corr = pyscx.accel.energy_distance(
    adata_real, adata_pred,
    pert_col="perturbation", control="control",
    metric="euclidean",          # "euclidean" | "l1" | "cosine"
    backend="auto",              # "auto" (default) | "gemm" | "scalar"
    dtype="f32",                 # "f32" (default) | "f64"
)

# For per-perturbation details (individual e_real / e_pred values):
details = pyscx.accel.energy_distance_details(adata_real, adata_pred)
# {
#   "correlation": 0.85,
#   "d_real": {"drug_A": 12.34, ...},
#   "d_pred": {"drug_A": 11.82, ...},
#   "pert_names": [...],
# }
```

SCX's pairwise kernel runs in two backend modes:

- **`backend="gemm"`** (the `auto` default for euclidean / cosine): a
  faer-dispatched matmul builds the `‖a‖² + ‖b‖² − 2·aᵀb` decomposition
  per pert, with row-norm² and the `sqrt(max(0, ·))` expansion in `f64`.
  L1 has no gemm formulation and `backend="gemm"` with `metric="l1"`
  raises `RuntimeError`.
- **`backend="scalar"`**: row-by-row `point_distance` reduction. Always
  valid; matches the original implementation and serves as the legacy
  back-compat path for callers that need bit-stable historical numbers.

The `dtype` kwarg controls the matmul / per-pair arithmetic precision —
reductions always accumulate in `f64` regardless. Default `"f32"` is
~2× faster than `"f64"` on AVX2 and matches `f64` within `atol=1e-4`
(verified by `pyscx/tests/test_cell_eval_parity.py::TestEdistanceParity`,
parametrised over `dtype ∈ {"f32", "f64"}`).

**GPU** (`device="gpu"`/`"auto"` on a GPU host, requires a `--features gpu`
build): the gemm decomposition runs on the device (one cuBLAS `sgemm` for
the `aᵀb` gram, per-point squared norms + a finalize/reduce kernel summing
distances in `f64`), reusing the harmony L2-normalize-columns kernel for the
cosine pre-normalization. GPU covers **euclidean + cosine** at `dtype="f32"`
only; `metric="l1"` (no gemm decomposition) and `dtype="f64"` stay on the CPU
path even under `device="gpu"` — the route is stamped `cpu_csr` rather than
erroring. GPU parity with the CPU is at the same `atol=1e-4` correlation bar.
Route `gpu_dense` / `cpu_csr`, recorded on
`adata.uns["scx_accel"]["energy_distance"]`. Because energy distance is
`O(N²)` in cells per perturbation (auto-skipped ≥ 500K cells on the CPU
reference), the GPU path is what makes it tractable at scale.

SCX's implementation never materializes the `[N, N]` *distance* matrix per
perturbation — the reduction keeps one `f64` per row — precomputes control
self-distance once, and parallelizes across perturbations with rayon. At 20K
cells × 2K genes × 50 perturbations the default `gemm + f32` path is **52×**
faster than cell-eval's `sklearn.metrics.pairwise_distances`; above ~500K the
reference becomes infeasible while SCX remains usable.

The gemm backend does need a Gram (`a·bᵀ`), and that is the allocation the
budget governs. It is built **one row block of `a` at a time**, bounded by
`SCX_ACCEL_PAIRWISE_MEMORY_BUDGET` (default 256 MiB) and reduced before the next
block overwrites it, rather than as `n_a · n_b · itemsize` — which at 100K
control cells on the default `dtype="f32"` is 40 GB. Inputs whose whole Gram
already fits the budget are a single block, i.e. the unblocked computation.

Two things the knob does **not** cover, since it is what operators will tune:

- The block is `max(budget, one Gram row)`. A single row wider than the budget
  still gets its own block — the floor that guarantees progress instead of an
  error.
- **`metric="cosine"` allocates outside it.** Row normalization materialises a
  full `n_a · n_dims` copy, plus `n_b · n_dims` when the two sides differ; at
  100K × 2000 f32 that is ~800 MB for a self-distance and ~1.6 GB for a cross,
  untouched by lowering the budget. `metric="euclidean"` keeps only `O(n)`
  row-norm buffers.

The knob changes how the same pairs are batched, never *which* pairs are summed
or the order the row sums reduce in. It is not bit-neutral above the budget,
though: a different block width makes faer panel the gemm differently, so Gram
ulps move — the same class of drift `Par::rayon(0)` already has across thread
counts, and well inside the `atol=1e-4` parity bound. Below the budget there is
one block and the *blocking* contributes nothing — the **cross** distance is then
bit-identical to pyscx ≤ 0.13.0's gemm. That is not a statement about
`energy_distance` as a whole: its self terms changed convention regardless of
blocking (see the upper-triangle note above), so the metric's output is not
generally bit-identical to the previous release.

The self-distance term (`d(X, X)`, and the once-per-side control self-distance)
additionally takes the **strict upper triangle** rather than the full square,
which is what `backend="scalar"` has always done and what
`sklearn.metrics.pairwise.cosine_distances` does (it forces the self diagonal to
0 when `X is Y` rather than evaluating it). On ordinary input the discarded half
is the symmetric mirror plus a diagonal that evaluates to ~0, so values move only
in the last bits and land *closer* to `backend="scalar"` than before.

> [!IMPORTANT]
> On a **zero-norm row under `metric="cosine"`** it is not a last-bit change.
> Row normalization leaves such a row as zeros, so its raw self-similarity is
> `1 - 0 = 1`, not `1 - 1 = 0` — the full square counted a whole unit per
> zero-norm row and the triangle does not. Two all-zero rows: `1.0` before,
> `0.5` now. This **fixes** a divergence — `backend="gemm"` (the `auto` default)
> and `backend="scalar"` used to disagree on such input, and gemm was the one
> that was wrong. If you have compared the two backends on data containing
> all-zero cells, the gemm numbers change.

> [!NOTE]
> The budget bounds **one block**, not the op. `energy_distance` evaluates
> perturbations on a rayon `par_iter`, so up to `RAYON_NUM_THREADS` blocks are
> live at once, and each task additionally holds its own dense copy of that
> perturbation's and the control's rows. Measured end to end at 30K control +
> 10K perturbation cells × 50 dims with 16 threads, peak RSS is **1.76 GB** —
> well above the 256 MiB budget, and dominated by those per-task buffers rather
> than by any single Gram. (Unblocked, the same run peaks at 3.85 GB.)

**Tuning.** Blocking is a memory/throughput trade and it only engages above the
budget. Which way it goes depends on `n_dims`: on a 50-dim embedding the gemm is
memory-bound and blocking is *faster* (the measurement above runs 3.9 s → 1.0 s),
while on a 2000-dim raw-gene input it costs up to 2× once it engages. If you are
running on raw genes with RAM to spare, raising
`SCX_ACCEL_PAIRWISE_MEMORY_BUDGET` until the Gram fits in one block restores the
unblocked throughput exactly; if the host is tight, lowering it — or
`RAYON_NUM_THREADS` — shrinks the Gram term. See
[performance/perturbation-metrics.md § Pairwise-distance kernels](../performance/perturbation-metrics.md#pairwise-distance-kernels-gram-blocking-and-the-with_min_len-fix).

## Knockdown efficiency (`pyscx.accel.knockdown_efficiency`)

Per-cell CRISPR knockdown efficiency and log-fold change against a
control baseline. Writes two columns to `adata.obs`:

```python
import scanpy as sc
adata = pyscx.open("perturb_seq.scx").to_anndata(backed=True)
sc.pp.normalize_total(adata)          # input must be normalized, not log1p'd

pyscx.accel.knockdown_efficiency(
    adata, pert_col="perturbation", control="control",
)
# adata.obs["KnockDownEfficiency"]  — 1 - x_target / (mu_control[target] + eps)
# adata.obs["KnockDownGeneFC"]      — x_log[target] - log1p(mu_control[target])
```

Input is expected on the normalized (linear) scale; the log-deviation pass
applies `log1p` internally. Control cells and cells whose perturbation name
isn't in `var_names` get `NaN` in both columns.

## Clustering agreement (`pyscx.accel.clustering_agreement`)

Builds perturbation-centroid matrices (pseudobulks excluding control), runs
kNN + Leiden at multiple resolutions, and scores the real-vs-predicted
cluster assignments via AMI / NMI / ARI. Matches
`cell_eval.metrics._anndata.ClusteringAgreement` within `atol=0.15`.

The implementation is **all native Rust** — no scanpy / anndata / igraph
calls. The kNN graph uses `scx_accel::neighbors::build_knn_graph`, which
auto-dispatches between two backends based on `n_obs` (the perturbation
count after filtering control):

- **`n_obs ≤ 5,000` (default)** — exact kNN via a faer matmul of
  `Centroids · Centroidsᵀ`, per-row partial top-k sort. Wins at small
  `n_obs` because the matmul runs at AVX-GEMM throughput while HNSW's
  inner loops are scalar. This is the active path on every realistic
  perturbation-evaluation workload (Replogle-scale n_perts ≈ 2–3K);
- **`n_obs > 5,000`** — HNSW via `instant-distance`
  (`ef_construction=200`, `ef_search=50`, `seed=0`).

Leiden uses `scx_accel::leiden` sequential mode (`max_iterations=2`,
`parallel=false`, `seed=0` — matches scanpy's
`flavor="igraph", n_iterations=2`). The pred-side kNN graph is built
once and reused across the resolution sweep, so only the Leiden pass
re-runs per resolution. Resolutions are evaluated in parallel via
rayon's `par_iter`. The whole hot path runs under `py.allow_threads`.

Per-phase profile timers can be enabled at runtime — set the env var
`RUST_LOG=pyscx::accel::eval_metrics::clustering_agreement=debug` and
the function logs `real_knn / real_leiden / pred_knn / sweep / total`
walls in milliseconds, alongside their fraction of total time.

**Caveat — small-graph divergence (`n_perts ≲ 10`).** The Rust-native
Leiden's RB-modularity tie-break differs from scanpy's
`flavor="igraph"` on graphs with very few nodes. On centroid graphs
with ≤ ~10 perturbations the two algorithms can produce different
community counts at `resolution=1.0`, and AMI / NMI are not
permutation-invariant across different partition cardinalities, so
scores can diverge by > 0.15 vs the scanpy-based reference. The
algorithms agree exactly at `n_perts ≥ 16` on the synthetic parity
fixtures (test scaffold uses `n_perts=30` for a comfortable margin).
If you have a small-perturbation experiment and need bit-stable
comparison against an existing scanpy-based pipeline, hand the
centroid matrices to `scanpy.tl.leiden` directly and feed the labels
into `pyscx.accel.adjusted_mutual_info` for the scoring step.

```python
score = pyscx.accel.clustering_agreement(
    adata_real, adata_pred,
    pert_col="perturbation", control="control",
    metric="ami",                    # "ami" | "nmi" | "ari" (ARI rescaled to [0,1])
    pred_resolutions=(0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
    n_neighbors=15,
)
```

The underlying scoring functions are also exposed for direct use on label
vectors (equivalent to `sklearn.metrics.*` within 1e-10, ARI uses
cell-eval's `(ARI+1)/2` rescaling):

```python
ami = pyscx.accel.adjusted_mutual_info(labels_a, labels_b)
nmi = pyscx.accel.normalized_mutual_info(labels_a, labels_b)
ari = pyscx.accel.adjusted_rand_index(labels_a, labels_b)  # sklearn ARI (negative = worse than random)
ari01 = pyscx.accel.adjusted_rand_index(labels_a, labels_b, rescaled=True)  # cell-eval (ARI+1)/2, [0, 1]
```

## DE result format bridge (`pyscx.accel.rank_genes_groups_df`)

Same computation as `rank_genes_groups()` but returns a DataFrame in cell-eval's
`DEResults` column schema. It defaults to **pandas**; pass `output="polars"` to
feed `cell_eval.initialize_de_comparison()` and `MetricPipeline(profile="de")`,
whose `DEResults.data` is typed `pl.DataFrame` and rejects a pandas frame:

```python
df = pyscx.accel.rank_genes_groups_df(
    adata, "perturbation", reference="control", output="polars",
)
# Columns: target, feature, fold_change, p_value, fdr,
#          log2_fold_change, abs_log2_fold_change
```

Useful when you want SCX's faster Wilcoxon rank-sum but cell-eval's DE metrics
downstream (overlap@N, precision@N, pr_auc, etc.).

> **`group=` is the scanpy extractor alias**, and `group=None` (or omitting it,
> with no `groupby=`) extracts **every** group — matching
> `sc.get.rank_genes_groups_df`'s "All groups are returned if group is None".
> Both modes return a **pandas** DataFrame, so scanpy-shaped idioms such as
> `.map` and `df[col] = ...` work directly; pass `output="polars"` for the polars
> frame `cell_eval` consumes (it needs the `eval` extra). Calling it the scanpy way —
> `pyscx.accel.rank_genes_groups_df(adata, group="0")` — does **not** recompute;
> it extracts the precomputed `adata.uns["rank_genes_groups"]` and returns
> scanpy's columns (`names, scores, logfoldchanges, pvals, pvals_adj`, plus
> `pct_nz_group` / `pct_nz_reference` when `uns` carries `pts` / `pts_rest` —
> i.e. after `rank_genes_groups(pts=True)`), a
> drop-in for `sc.get.rank_genes_groups_df`. Use `groupby=` to recompute (cell-eval
> columns), `group=` to extract (scanpy columns); pass one, not both. The scanpy
> filters `pval_cutoff` / `log2fc_min` / `log2fc_max` apply to the `group=` path.


## See also

- [Differential expression accelerators](accel-differential-expression.md).
- [docs/training.md](../training.md) — the perturbation training loaders.
