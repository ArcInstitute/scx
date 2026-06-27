# Pseudobulk negative-binomial GLM (NB-GLM)

SCX ships a Rust-native pseudobulk **negative-binomial GLM** for differential
expression — a CPU, `f64`, dependency-free alternative to PyDESeq2 for
DESeq2-style workflows at Perturb-seq scale. It lets you stay on the Rust
acceleration path after pseudobulk aggregation instead of handing off to
pydeseq2.

> **DESeq2-*style*, not DESeq2-*identical*.** SCX implements the DESeq2 *core* —
> IRLS / Fisher scoring for the mean coefficients, a Cox–Reid adjusted
> profile-likelihood dispersion estimate, a parametric mean→dispersion **trend
> fit**, **empirical-Bayes dispersion shrinkage**, and (default-on) DESeq2
> **Cook's-distance outlier filtering** and **base-mean independent filtering** —
> but it does **not** reproduce DESeq2 bit-for-bit. It still omits apeglm/ashr LFC
> shrinkage. The bar is **ranking / effect-sign / significance parity**, not
> numerical equality. If you need exact DESeq2 behaviour, keep using PyDESeq2
> (`backend="pydeseq2"`, the default).

> **Behavior change.** Cook's-distance outlier filtering and base-mean
> independent filtering are **on by default** (matching DESeq2 `results()`). Versus
> the first NB-GLM release, `accel.nb_glm` / `pdex_nb_glm` /
> `pseudobulk_dex(backend="nb_glm")` can now emit `NaN` in `padj` (and `fdr`) — and,
> for Cook's outliers, in `pvalue` — for outlier and low-base-mean genes. Set
> `cooks_filtering=False` and/or `independent_filtering=False` to restore the
> unfiltered behavior.

## When to use it

Pseudobulk NB-GLM is the right tool when you have **biological replicates** —
multiple pseudobulk samples per condition (e.g. several donors / batches / wells
per perturbation). It models per-gene over-dispersion and borrows strength across
genes via the dispersion trend + shrinkage, which is what makes low-replicate
results trustworthy.

It is the **wrong** tool when you have **one profile per condition** (no
replicates): a pseudobulk NB-GLM cannot estimate dispersion from a single sample,
so the estimator degenerates. For no-replicate layouts use the per-cell tests
instead — [`pyscx.accel.pdex_ref`](api.md#python-api-pyscx) (Mann–Whitney U +
pseudobulk log fold change) or `pyscx.accel.rank_genes_groups` (Wilcoxon rank-sum). The
`pdex_nb_glm` entry point **enforces** this: it errors with guidance when no
stratifier is supplied (see [§ Replicate requirement](#replicate-requirement)).

| Situation | Recommended |
|---|---|
| ≥ 2 pseudobulk replicates per condition, want DESeq2-style NB-GLM | `nb_glm` / `pdex_nb_glm` / `pseudobulk_dex(backend="nb_glm")` |
| One profile per condition (no replicates) | `pdex_ref` / `rank_genes_groups` (per-cell) |
| Need exact DESeq2 numerics | `pseudobulk_dex(backend="pydeseq2")` (default) |

## Three entry points

All three accept `device="auto"` (`"auto"` / `"cpu"` / `"gpu"` / `"gpu:N"`). The
core NB-GLM fit runs in `f64` end-to-end; GPU routes (`gpu_nb_glm_csr`,
`gpu_nb_glm_csc`) accelerate the pseudobulk aggregation stage while the IRLS /
Cox–Reid fit itself remains CPU. `pseudobulk_dex(backend="nb_glm")` is CPU-only
(no `device` parameter).

### 1. `pyscx.accel.nb_glm` — direct, already-pseudobulked

For callers who have already built a pseudobulk count matrix and a numeric design
and want a DESeq2-shaped result. Returns a **pandas** DataFrame with PyDESeq2-style
column names.

```python
import numpy as np, pyscx

# counts: [n_samples × n_genes] (default) or [n_genes × n_samples]
# design: [n_samples × n_features], full column rank (intercept + covariates)
df = pyscx.accel.nb_glm(
    counts, design,
    size_factors=None,          # None → DESeq2 median-ratio factors
    contrast=1,                 # coefficient index, weight vector, or None (last coef)
    gene_names=gene_ids,
    options={"dispersion": "cox_reid_shrunk"},
    counts_axis="samples_by_genes",
)
# columns: gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj,
#          dispersion, cooks, converged, n_iter
```

### 2. `pyscx.accel.pdex_nb_glm` — cell-eval / pdex path (from AnnData)

Builds pseudobulk **replicates** straight from an AnnData and emits the
**cell-eval/pdex polars schema**, so it is a drop-in DE method for
[`cell-eval`](https://github.com/arcinstitute/cell-eval). A **stratifier is
required** (it forms the replicates), passed as a **list** of obs column names —
e.g. `stratify_by=["donor"]`; a bare string is rejected with a clear error.

```python
df = pyscx.accel.pdex_nb_glm(
    adata,
    "perturbation",             # groupby: perturbation column in obs
    "control",                  # reference / control label
    stratify_by=["donor"],      # forms pseudobulk REPLICATES (required)
    min_cells_per_group=10,
    is_log1p=None,              # auto-detected from adata.uns["log1p"]
)
# polars columns (cell-eval DEResults schema):
#   target, feature, fold_change, p_value, fdr, log2_fold_change,
#   abs_log2_fold_change
```

The output schema is byte-compatible with `pyscx.accel.pdex_ref` /
`rank_genes_groups_df`, so it feeds straight into `cell_eval.data.DEComparison`.
On the `cell-eval-scx` side this is a one-line `ScxDeMethod` addition
(`de_method="nb_glm"`).

### 3. `pseudobulk_dex(backend="nb_glm")` — alongside the PyDESeq2 bridge

`pyscx.accel.pseudobulk_dex` gains a `backend` argument
(`"pydeseq2"` default, or `"nb_glm"`). The NB-GLM backend emits the **same pandas
schema** as the pydeseq2 path, so existing consumers need no changes, and it has
**no pydeseq2 dependency**.

```python
df = pyscx.accel.pseudobulk_dex(
    adata,
    groupby=["perturbation", "donor"],
    test_col="perturbation",
    reference="control",
    backend="nb_glm",           # default is "pydeseq2"
)
# columns: gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj,
#          target, reference
```

## Replicate requirement

A pseudobulk NB-GLM needs **≥ 2 pseudobulk samples per condition** to estimate
dispersion. `pdex_nb_glm` (and `pseudobulk_dex(backend="nb_glm")`) form one
pseudobulk sample per `(condition × stratum)` combination, so you need a
stratifier (batch / donor / well / replicate id) that spans **≥ 2 strata** per
condition.

`pdex_nb_glm` enforces this:

- No `stratify_by` (or an empty list) → `ValueError` pointing you to
  `pdex_ref` / `wilcoxon` (rank-sum).
- A perturbation with fewer than 2 replicates per condition is **skipped with a
  `UserWarning`** (rather than silently returning a meaningless dispersion).

This is the honest boundary of where pseudobulk NB-GLM applies. cell-eval's
default aggregation is one profile per perturbation; without a stratifier the
NB-GLM is unidentifiable, hence the hard error.

## How it works

For each gene independently (rayon gene-parallel):

1. **Size factors** — if not supplied, DESeq2 **median-ratio** size factors are
   computed on the pseudobulk counts (per-gene geometric mean over positive
   counts; per-sample median ratio; normalised to geometric mean 1; library-size
   fallback when too few genes have a valid geometric mean).
2. **Mean fit (IRLS / Fisher scoring)** — given the current dispersion, `beta` is
   fit by iteratively reweighted least squares with NB working weights
   `W = μ / (1 + α·μ)` and a log link with a `log(size_factor)` offset. The final
   `XᵀWX` is the expected Fisher information, reused for the Wald step.
3. **Dispersion fit (Cox–Reid)** — given `beta`/`μ`, `log(α)` maximises the
   Cox–Reid adjusted profile log-likelihood via a safeguarded 1-D root find
   (Illinois / regula-falsi, bracketed in `[min_disp, max_disp]`). The Cox–Reid
   adjustment removes the small-sample bias plain MLE has when `n_samples ≈
   n_features` — exactly the pseudobulk regime.
4. Steps 2–3 alternate to convergence (1–3 outer passes typically).

Then, across genes:

5. **Dispersion trend fit** — a parametric `α_trend(μ̄) = a0 + a1/μ̄` is fit by a
   gamma-family GLM on the per-gene MLE dispersions vs base mean.
6. **Empirical-Bayes shrinkage** — each gene's dispersion is shrunk toward the
   trend with a log-normal prior whose variance is estimated (robustly, via the
   MAD of log-residuals about the trend). `beta` is refit once at the shrunken
   dispersion so the SEs use the final value. This stabilises low-replicate genes.

7. **Wald inference** — the contrast effect `c·beta`, SE `√(cᵀ·cov·c)` from the
   Fisher inverse, Wald statistic, two-sided p-value, and `log2FoldChange =
   effect / ln 2`. Ill-conditioned / non-PD information (after a small ridge)
   yields conservative output (`pvalue = 1`, `stat = 0`, `lfcSE = ∞`) rather than
   a spurious call.

8. **Cook's-distance outlier filtering** (DESeq2 default, on) — per gene, the
   maximum Cook's distance over samples is computed from the leverage (reusing the
   Wald covariance) and the Pearson residual. Genes whose max Cook's distance
   exceeds `qf(0.99, p, m−p)` (the F-quantile cutoff; override via `cooks_cutoff`)
   have their `pvalue` **and** `padj` set to `NaN`. Only applied when the residual
   df `m − p ≥ 3` (smaller designs can't localize an outlier). The maximum Cook's
   distance is reported in the `cooks` column. `log2FoldChange`/`lfcSE`/`stat` are
   still reported for flagged genes. There is no count replacement (that needs ≥ 7
   replicates and is out of scope).

9. **Independent filtering** (DESeq2 default, on) — a base-mean cutoff is chosen to
   maximize the number of rejections (genefilter algorithm: a quantile grid, lowess
   smoothing, and the 1-SE rule), and genes below it have **only** their `padj` set
   to `NaN`. Benjamini–Hochberg is then computed over the retained genes. When the
   filter is uninformative (e.g. near-uniform base means) it is a no-op and every
   gene keeps a finite `padj`.

Both filtering steps are on by default to match DESeq2; set
`cooks_filtering=False` / `independent_filtering=False` to disable them. Genes with
`NaN` `padj`/`fdr` are treated as not-significant by downstream consumers (cell-eval
thresholds `fdr < 0.05`).

> **BH scope.** In `pdex_nb_glm` / `pseudobulk_dex(backend="nb_glm")` each
> non-reference target is fit independently, so the `padj` / `fdr` column is BH
> corrected **per target** (within that contrast's gene set), not globally across
> all target × gene rows. This matches `pdex_ref` and cell-eval's per-target
> convention; if you need a global correction, re-adjust the pooled `p_value`
> column yourself.

### Contrasts

The Rust API and `accel.nb_glm` take a numeric contrast:

- an **integer coefficient index** (test that coefficient against zero), or
- a **weight vector** `c` (test `c·beta = 0`), or
- `None` → the **last coefficient** (the DESeq2 "last coefficient" convention),
  since a numeric design carries no column names.

`pdex_nb_glm` / `pseudobulk_dex(backend="nb_glm")` build a `[intercept,
is_target]` design per non-reference level and test the `is_target` coefficient
(1-vs-reference). Stratifiers enter only as **replicates** (extra rows), not as
design covariates — this keeps the marginal effect aligned with the per-cell
`pdex_ref` test for ranking parity (batch-as-covariate is a v2 refinement).

## Options

`accel.nb_glm(options=...)` and `pdex_nb_glm(nbglm_options=...)` accept an optional
dict; unspecified keys keep their defaults:

| Key | Default | Meaning |
|---|---|---|
| `dispersion` | `"cox_reid_shrunk"` | `"moments"` (fast, noisy), `"cox_reid_mle"` (per-gene MLE, no shrinkage), or `"cox_reid_shrunk"` (MLE + trend + EB shrinkage) |
| `fit_dispersion_trend` | `True` | Fit the parametric mean→dispersion trend |
| `shrink_dispersion` | `True` | Apply empirical-Bayes shrinkage toward the trend |
| `min_disp` / `max_disp` | `1e-8` / `100.0` | Dispersion clamps |
| `max_irls_iters` | `100` | IRLS iteration cap (converges in ~5–15) |
| `irls_tol` | `1e-8` | Relative-deviance convergence tolerance |
| `max_outer_iters` | `10` | Mean ↔ dispersion outer-loop cap |
| `cooks_filtering` | `True` | Apply DESeq2 Cook's-distance outlier filtering (`pvalue`/`padj` → NaN) |
| `cooks_cutoff` | `None` | Cook's cutoff; `None` ⇒ `qf(0.99, p, m−p)` |
| `independent_filtering` | `True` | Apply base-mean independent filtering (`padj` → NaN below the cutoff) |
| `independent_filter_alpha` | `0.1` | Significance level independent filtering optimizes rejections at |

## Diagnostics

- **`converged`** (per-gene boolean) and **`n_iter`** (outer iterations) are
  surfaced in the `accel.nb_glm` output.
- The route is recorded on `adata.uns["scx_accel"]["pseudobulk_dex"]` /
  `["pdex_nb_glm"]` as `{"route": "cpu_nb_glm", "fallback_reason": "none"}` — a
  first-class native CPU route (it is **never** stamped `no_rapids`; that reason
  applies only to rapids-absent GPU fallbacks). See
  [docs/api.md § Accelerator route metadata](api.md#accelerator-route-metadata).
- Genes that hit a dispersion clamp, fail to converge, or are all-zero
  (`pvalue = 1`, `dispersion = NaN`, `log2FoldChange = 0`) are counted in internal
  diagnostics.
- Cook's-distance outliers, the number of independent-filtered genes, the chosen
  base-mean threshold, and the Cook's cutoff used are recorded in internal
  diagnostics; the per-gene maximum Cook's distance is surfaced in the `cooks`
  column of `accel.nb_glm`.

## Performance

The fitter is rayon gene-parallel and `f64` throughout; IRLS converges in ~10
iterations (vs hundreds for the prior Adam approach). A
`30k genes × 200 samples × 5 features` fit completes in a few seconds on 8 CPU
threads — well within the budget for any pseudobulk DE.

## See also

- [docs/scanpy.md § Pseudobulk Differential Expression](scanpy.md#pseudobulk-differential-expression-pyscxaccelpseudobulk_dex)
- [docs/api.md § Python API](api.md#python-api-pyscx) — exact signatures
- [docs/api.md § Accelerator route metadata](api.md#accelerator-route-metadata)
