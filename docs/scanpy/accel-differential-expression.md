# Differential expression accelerators

> Part of the [SCX + scanpy guide](README.md). Shared conventions are on the
[accelerators overview](accelerators.md).

## Differential Expression

SCX offers several DE functions covering different experimental designs:

| Function | Use case | Method |
|----------|----------|--------|
| `rank_genes_groups` | Standard cluster marker genes (scanpy-compatible) | Wilcoxon rank-sum |
| `pdex_ref` | Perturbation-specific fold changes against a control | Wilcoxon rank-sum (perturbation semantics) |
| `rank_genes_groups_df` | Same as `rank_genes_groups` but returns a DataFrame (cell-eval schema), or extracts precomputed results | Wilcoxon rank-sum |
| `pseudobulk_dex` | Pseudobulk DE with biological replicates | PyDESeq2 or Rust-native NB-GLM |
| `nb_glm` | Direct NB-GLM on a pre-aggregated pseudobulk count matrix | Rust-native negative-binomial GLM |
| `pdex_nb_glm` | Perturbation NB-GLM with replicate-forming stratification | Rust-native negative-binomial GLM |

All Wilcoxon rank-sum-based functions support GPU via `device="gpu"` (CSC-direct
`gpu_csc_v3` when a sidecar is present **and** the handle's row window still
spans at least half the CSR shards, CSR-direct `gpu_csr_v3` otherwise).
The NB-GLM functions are CPU-only.

### `pyscx.accel.rank_genes_groups`

Parallel Wilcoxon rank-sum test with rayon. Uses a pre-ranking approach:
for 1-vs-rest, all cells are ranked once per gene and the ranks are reused
across groups (10× fewer sorts than the naive per-group approach). Compares
each cluster against the rest (or a specific reference group) and applies
Benjamini–Hochberg correction. Results are written to the same
`adata.uns["rank_genes_groups"]` format as scanpy, so
`sc.pl.rank_genes_groups()` and `sc.get.rank_genes_groups_df()` work
identically.

```python
pyscx.accel.rank_genes_groups(adata, "leiden")

# Results written to:
#   adata.uns["rank_genes_groups"]["names"]           — structured array
#   adata.uns["rank_genes_groups"]["scores"]           — z-scores
#   adata.uns["rank_genes_groups"]["pvals"]             — raw p-values
#   adata.uns["rank_genes_groups"]["pvals_adj"]         — BH-adjusted
#   adata.uns["rank_genes_groups"]["logfoldchanges"]    — log2 FC
#   adata.uns["rank_genes_groups"]["pts"]               — with pts=True: genes × groups
#   adata.uns["rank_genes_groups"]["pts_rest"]          — with pts=True and reference="rest"

# Downstream scanpy works identically:
sc.pl.rank_genes_groups(adata, n_genes=20)
df = sc.get.rank_genes_groups_df(adata, group="0")

# Restrict the reported groups; fraction-expressing tables as scanpy writes them.
pyscx.accel.rank_genes_groups(adata, "leiden", groups=["0", "3"], pts=True)
df = pyscx.accel.rank_genes_groups_df(adata, group="0")  # + pct_nz_group, pct_nz_reference
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groupby` | (required) | Column in `adata.obs` to group cells by. Cells with no label — a pandas missing value (`NaN` / `None` / `pd.NA`, decided by `pandas.isna` and not by how it prints), or a value outside the column's categories — get **no group of their own**: no result row, no `pts` column. For `reference="rest"` they are still in the rank pool and in every group's `"rest"`, numerator and denominator alike, exactly as scanpy 1.12 does (**changed in 0.17**; before that pyscx left them out, and results on partially labelled `obs` differed from scanpy's). A pairwise run against a named reference compares `group ∪ reference`, so such a cell takes no part in it. A `UserWarning` reports how many there were and which of the two applies. A group whose *name* merely looks like a missing value — `"nan"`, `"None"`, `""` — is a real group and is kept. |
| `reference` | `"rest"` | Compare against a specific group or `"rest"` (1-vs-rest) |
| `groups` | `None` (all) | Report only these groups, in this order. An **output** filter: the pool each group is compared against does not change, so a group's statistics are identical with or without it — and identical to scanpy's `groups=`. Unknown names, repeats and an empty list raise; the reference group is silently not tested (scanpy's rule) but stays a `pts` column. Compute is not reduced. |
| `n_genes` | all | Number of top genes to report per group |
| `method` | `"wilcoxon"` | Statistical method (currently only `"wilcoxon"`) |
| `pts` | `False` | Also write `uns[key]["pts"]` — and `["pts_rest"]` when `reference="rest"` — the fraction of cells in each group with a nonzero value, as scanpy does: `genes × groups` float64 DataFrames indexed by the analysed var names, over **every** gene whatever `n_genes` says. See below. |
| `corr_method` | `"benjamini-hochberg"` | Recorded in `params`. Only BH is implemented; any other value (e.g. `"bonferroni"`) **raises** rather than silently applying BH. |
| `use_raw` | `None` | Analyse `adata.raw.X` (with `adata.raw.var` names). `None` → `True` iff `adata.raw` exists and `layer` is `None` (scanpy's rule). Mutually exclusive with `layer`. |
| `layer` | `None` | Analyse `adata.layers[layer]` instead of `X`. Works on a file opened backed, where the layer is an `ScxBackedLayerDataset` and streams shard-by-shard like `X` does — the case the kwarg exists for, since raw counts usually live in a layer once `X` is normalised. A CSC sidecar belongs to `X`, so a layer request stays on the CSR route (`route="cpu_csr"`) rather than reading the sidecar's columns. |
| `rankby_abs` | `False` | Sort genes by absolute z-score instead of signed score. `False` (default) matches scanpy's default: highest positive z-score first. `True` ranks by significance regardless of direction. |
| `tie_correct` | `False` | Apply the `Σ(t³−t)` tie correction to the Wilcoxon rank-sum variance estimate. The default `False` matches **scanpy's default** (`scanpy.tl.rank_genes_groups` takes the same parameter, also defaulting to `False`); `True` matches **scipy**, which always corrects. These are two different answers, not two precisions — see [Numerical parity](#numerical-parity-against-scanpy-and-scipy) below. |
| `gene_chunk_size` | `None` | Process genes in chunks of this size to limit memory. `None` processes all genes at once. |
| `prefer_format` | `"auto"` | `"auto"` (default; CPU routes CSC-direct when a valid sidecar is present **and** the row window spans at least half the CSR shards, else CSR), `"csr"`, or `"csc"` — the explicit `"csc"` skips that policy. |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"`. GPU routes to CSC-direct (`gpu_csc_v3`) when a sidecar is present and the row window spans at least half the CSR shards, or CSR-direct (`gpu_csr_v3`) otherwise. |

**A group with fewer than two cells is refused, whichever way you ask.** Any
**participating** group raises scanpy's `Could not calculate statistics for
groups <g> since they only contain one sample.` — every level when `groups` is
omitted, the named ones (plus a named reference) when it is given. Before 0.17
only the named path checked, so the default call returned finite, plausible
z-scores computed from a single cell. An unused category counts as zero cells
and raises too, exactly as in scanpy (its `value_counts()` reports every
category, and `groups="all"` selects every category); since the usual cause is
a subset that kept its parent's levels, the message names
`remove_unused_categories()`. Under `stratify_by` this is a per-stratum failure
like any other: it warns and drops that stratum.

Benchmarked at 5.4s on 1M cells (3.2× faster than scanpy's 17.2s).

**`pts` is one extra pass, not a kernel change.** The count of nonzero values
per (group, gene) is made in a separate streaming pass over the analysed matrix
— the same backed / lazy / scipy / dense source the test ran on — so every
route reports the same number from the same code, CSC-direct and the GPU
drivers included; `pts=True` costs one more read of `X` (a second decode pass
on a backed handle — the default four-shard LRU does not keep a full
sequential scan resident) and nothing when off. "Expressing" is scanpy's `!= 0`: an explicit zero
stored in a scipy CSR is not counted, a negative value is. `pts_rest[g]` is
scanpy's `X[~mask_g]` fraction — over **every other cell of the matrix**, cells
with no `groupby` label included — so the table equals scanpy's on partially
labelled input as well. Since 0.17 the rank-sum statistic uses that same pool,
so `pts_rest` and `pvals` describe one reference population rather than two.
`rank_genes_groups_df(group=…)` then appends `pct_nz_group` and (for
`reference="rest"`) `pct_nz_reference`, looked up by gene name, as
`sc.get.rank_genes_groups_df` does. Both refuse duplicate `var_names` (run
`adata.var_names_make_unique()` first): a by-name join cannot tell two genes
with one name apart, and scanpy's merge silently multiplies the rows instead.
Both frames survive `pyscx.from_anndata` → `to_anndata` and both h5ad
directions: `uns` carries a `pandas.DataFrame` envelope preserving the index,
the column order and per-column dtypes (see
[docs/api/python-experiment.md § `uns` serialization](../api/python-experiment.md#uns-serialization)).

#### Numerical parity against scanpy and scipy

Pinned against both references, in the Rust suite (so `cargo test` gates it
with no Python installed) and in `pyscx/tests/test_accel.py`. Every tolerance
below is the **observed** max |Δ| plus one decimal order. Regenerate the
reference *tables* with `benchmarks/scripts/generate_de_parity_references.py`,
which owns the fixtures and the expected values together; it also prints the
Python-side max |Δ| measurements quoted below, so every number on this page
comes out of one script rather than out of a session someone ran once.

**Which reference applies depends on `tie_correct`**, and this is a difference
in definition rather than in precision. scipy *always* applies the tie
correction. scanpy takes the same `tie_correct` parameter and **defaults it to
`False`**, so it can produce either convention and simply does not correct
unless asked — SCX's default matches scanpy's. On a fixture with ties the two
conventions differ by **4.4e-01** in `z`, so pinning one arm against the other's
reference would be wrong, not merely loose.

Rust-side pins (`scx-accel`, CPU dense and analytic-nnz kernels):

| `tie_correct` | `z` reference | bar | `p` reference | bar |
|---|---|---|---|---|
| `True` | `scanpy(tie_correct=True).scores` | `1e-6` | `scipy.stats.mannwhitneyu(...).pvalue` | `1e-15` |
| `False` (default) | `scanpy(tie_correct=False).scores` | `1e-6` | `scanpy(...).pvals` | `1e-12` |

Both `z` bars are `1e-6` because scanpy stores `scores` as **float32** in its
recarray; its `pvals` are float64, and scipy is f64 throughout, hence the tight
`p` bars. scipy exposes no `z`, so the `z` reference is scanpy on both arms —
deriving one from scipy's `U` would mean re-implementing the variance formula
being tested. On the corrected arm scipy's and scanpy's `p` agree at **exactly
0.0**, and that agreement is itself asserted: two independent implementations of
the tie term is what makes either of them an oracle for SCX.

The **GPU** arm is held to the same `z` bar and a looser `p` bar (`1e-9`), which
is a CUDA reduction-order bound rather than a convention difference. `pvals_adj`
is **not** pinned Rust-side — BH is applied above the kernel — and is covered on
the Python side below.

Python-side pins (`pyscx/tests/test_accel.py`), against scanpy on real fixtures,
keyed by gene name:

| Field | Observed max &#124;Δ&#124; | Bar |
|---|---|---|
| `scores` | 1.4e-07 | `1e-6` |
| `pvals` | 1.1e-16 | `1e-12` |
| `pvals_adj` | 3.3e-16 | `1e-12` |
| `logfoldchanges` (log1p'd input) | 2.3e-07 | `1e-6` |
| `pts` / `pts_rest` (`pts=True`) | 0.0 | `1e-12` |

`pts` is an exact integer count divided once, so the two implementations agree
bit-for-bit (`pyscx/tests/test_rank_genes_groups_pts.py`); its bar is the
division's, not a tolerance for a differing formula.

The observed column is the pytest fixture's; the generator re-measures on a
second, independent fixture and **fails** if any field there exceeds the same
bar (it lands within 5e-07 / 3e-16 / 3e-16 / 2e-07). Run it in the `.venv`, where
scanpy and `pyscx` are importable — it refuses to exit 0 having skipped that
half, unless you ask for the Rust tables alone with `SCX_SKIP_PYTHON_BARS=1`. The bars are what is
claimed — the observed figures are fixture-dependent by nature, and pinning a
tolerance to one fixture's exact divergence is how a bar stops surviving a
change of input.

Two caveats that are load-bearing rather than fine print:

- **Gene *order* within a group is not claimed.** scanpy's tie order comes from
  `np.argsort`'s default `quicksort`, which is not stable, so two genes with
  equal scores may come out in either order. Compare by gene name, never by
  position. (The *set* of names is claimed, and asserted.)
- **`logfoldchanges` on raw counts is deliberately not scanpy's.** scanpy
  `expm1`s the group means unconditionally, assuming log1p'd input — it emits a
  warning when the data looks like counts. SCX detects the untransformed case
  and uses `log2(mean + ε)` differences instead. On raw counts the two differ by
  ~6e+01. Log-transform first if you want the two to agree.

In-memory and backed/streaming DE are **bit-identical** to each other on every
field including gene order, at every `gene_chunk_size` — an identity, not a
tolerance.

### `pyscx.accel.pdex_ref`

Perturbation DE in reference mode — computes Wilcoxon rank-sum fold changes
for each non-reference group against the reference (e.g. `"non-targeting"`
or `"control"`). Designed for Perturb-seq experiments where you compare each
perturbation against a common control population. Returns a **pandas**
DataFrame in cell-eval's `DEResults` column schema; pass `output="polars"` for
the polars frame `cell_eval` (and upstream `pdex`) use.

```python
df = pyscx.accel.pdex_ref(adata, "perturbation", reference="non-targeting")
# Columns (upstream pdex v0.2.x schema): target, feature, target_mean, ref_mean,
#          target_membership, ref_membership, fold_change (= log2_fold_change,
#          deprecated alias), log2_fold_change, percent_change, p_value,
#          statistic, fdr

# Test only these targets (rows equal the full run's, in this order):
df = pyscx.accel.pdex_ref(adata, "perturbation", groups=["KO_1", "KO_7"])
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groupby` | (required) | Column in `adata.obs` containing perturbation labels |
| `reference` | `"non-targeting"` | Control group label |
| `groups` | `None` (all) | Restrict the tested targets to these `groupby` levels, reported in this order. A target is only ever compared with the reference (MWU, pseudobulk fold change, per-target `cpm_filter`, per-target BH), so each selected target's rows equal the unrestricted run's, while the work — and GPU memory — scales with the number of targets asked for (the other levels' cells are dropped before the kernel). Unknown names, repeats, an empty list and the reference itself raise. A pyscx extension: upstream pdex has no such knob. |
| `use_raw` | `None` | Analyse `adata.raw.X` (with `adata.raw.var` names); `None` → `True` iff `adata.raw` exists and `layer` is `None`. Recorded on `uns["scx_accel"]["pdex_ref"]`. |
| `layer` | `None` | Analyse `adata.layers[layer]` instead of `X`. Mutually exclusive with `use_raw=True`. Works on a file opened backed (the layer streams like `X`); `is_log1p` auto-detection reads the layer's own catalog bound. |
| `is_log1p` | `None` | Whether input X is log1p-transformed. `None` auto-detects, layout-independently — a backed handle and an in-memory `AnnData` over the same data resolve the same mode. Order: `adata.uns["log1p"]` → a lazy `X`'s `Log1p` transform → a backed `X`'s catalog `value_max` against a `< 30` heuristic (catalog-only, no decode) → in-memory `max(X) < 30`. **Two cases raise `ValueError` instead of guessing**: a backed file whose catalog cannot bound its value range — shards that are float-encoded (the format records no range for those), carry no statistics, or hold no values — and a lazy `X` carrying a rescaling-only chain such as `normalize_total` (which detaches the values from the recorded range). Pass `True`/`False` to resolve either — or run `pyscx.accel.log1p`, which stamps `uns["log1p"]` and settles it. |
| `geometric_mean` | `True` | Use geometric mean for fold-change computation |
| `epsilon` | `1e-9` | Finite-guard pseudocount on count-space means before fold-/percent-change (not CPM/MWU). Default keeps outputs finite; `0/0 → 0.0`. Pass `0.0` for legacy `±inf` on reference-undetected genes. |
| `cpm_filter` | `None` | Optional CPM floor `T`: keep a gene iff `target_cpm > T` or `ref_cpm > T` (pooled arithmetic CPM, mode-independent); drops other rows, FDR recomputed over survivors. |
| `gene_chunk_size` | `None` | Process genes in chunks to limit memory |
| `prefer_format` | `"auto"` | `"auto"` (default), `"csr"`, or `"csc"` |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"`. GPU takes the CSC-direct route (`gpu_csc_v3`) when a sidecar is present and the row window spans at least half the CSR shards. |
| `output` | `"pandas"` | `"pandas"` (needs no extra) or `"polars"` (needs the `eval` extra; what `cell_eval` consumes) |

## Pseudobulk Differential Expression (`pyscx.accel.pseudobulk_dex`)

Streaming pseudobulk aggregation in Rust + negative binomial GLM testing via
[pydeseq2](https://pydeseq2.readthedocs.io/). Designed for perturbation
sequencing (Perturb-seq) experiments with biological replicates.

> **`groupby` means the opposite of what it means in `rank_genes_groups`.**
> In `accel.rank_genes_groups` — and everywhere in scanpy — `groupby` names the
> column whose levels are compared. In `pseudobulk_dex` it names the columns
> that together define one pseudobulk **sample**: condition *plus* replicate,
> e.g. `["disease", "donor_id"]`. The column being compared is `test_col`.
>
> Passing only the condition column produces one pseudobulk sample per
> condition and therefore no replication. Because that mistake is easy to make
> and quiet, the replicate-role spellings **`sample_cols=`** and
> **`sample_key=`** are accepted as aliases for `groupby` (`sample_key` also
> takes a bare string). Pass exactly one of the three.

The aggregation phase streams shards from `BackedCsrReader` without
materializing the full matrix — peak memory is one shard plus the pseudobulk
count matrix (n_groups × n_vars).

```python
import pyscx

adata = pyscx.open("perturb_seq.scx").to_anndata(backed=True)

# Run pseudobulk DE: drug vs control. The pseudobulk sample is
# (perturbation, donor) — donor is the replicate that makes the test possible.
result = pyscx.accel.pseudobulk_dex(
    adata,
    groupby=["perturbation", "donor"],   # or sample_cols=[...] — same thing
    test_col="perturbation",             # the column actually compared
    reference="control",
)

# result is a pandas DataFrame:
#   gene | baseMean | log2FoldChange | lfcSE | stat | pvalue | padj | target | reference
print(result.sort_values("padj").head(20))
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groupby` | (required) | Obs columns defining a pseudobulk sample — condition **plus** replicate (e.g. `["disease", "donor_id"]`). **Not** the compared column |
| `sample_cols` | `None` | Alias for `groupby`, named for the role it plays |
| `sample_key` | `None` | Single-column alias for `groupby`; accepts a bare string (`sample_key="donor_id"`) |
| `test_col` | (required) | Which of those columns holds the condition to compare |
| `reference` | (required) | Reference level in `test_col` (e.g., `"control"`) |
| `design` | `"~ test_col"` | DESeq2 design formula (auto-generated if not specified) |
| `aggr_method` | `"sum"` | Aggregation method: `"sum"` or `"mean"`. `"mean"` requires `backend="pydeseq2"` — the negative-binomial count model is defined on summed replicate counts |
| `min_cells_per_group` | 10 | Groups with fewer cells are excluded |
| `backend` | `None` → `"nb_glm"` | DE engine: `"nb_glm"` (Rust-native NB-GLM, no optional dependency — see [§ NB-GLM backend](#nb-glm-backend-rust-native-pseudobulk-de)) or `"pydeseq2"` (exact DESeq2 numerics; needs `pip install 'pyscx[pydeseq2]'`). Both emit the same column schema. Defaulted to `"pydeseq2"` through v0.12. |

## Stratified Differential Expression

Both `rank_genes_groups()` and `pseudobulk_dex()` support automatic
stratification via the `stratify_by` parameter. DE is run independently
within each stratum and the results are concatenated into a single
DataFrame with stratum columns appended.

```python
import pyscx

adata = pyscx.open("perturb_seq.scx").to_anndata()

# Single-cell DE stratified by cell type:
result = pyscx.accel.rank_genes_groups(
    adata, "perturbation",
    stratify_by=["cell_type"],
    min_cells_per_stratum=50,
)
# Returns a DataFrame with columns:
#   gene | scores | pvals | pvals_adj | logfoldchanges | group | cell_type

# Multi-column stratification (composite strata):
result = pyscx.accel.rank_genes_groups(
    adata, "perturbation",
    stratify_by=["cell_type", "tissue"],
    min_cells_per_stratum=30,
)
# Returns DataFrame with both cell_type and tissue columns

# Pseudobulk DE stratified by cell type. `stratify_by` is pydeseq2-only —
# the default NB-GLM backend takes replicates as rows of one design, so it
# rejects stratification (put the replicate column in `groupby` instead).
result = pyscx.accel.pseudobulk_dex(
    adata,
    groupby=["perturbation", "donor"],
    test_col="perturbation",
    reference="control",
    stratify_by=["cell_type"],
    min_cells_per_stratum=50,
    backend="pydeseq2",                     # required for stratify_by
)
# Returns DataFrame with cell_type column added
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stratify_by` | `None` | Column(s) in `adata.obs` to stratify by. Single string or list of strings. |
| `min_cells_per_stratum` | 50 | Strata with fewer cells are skipped (with warning). |

Strata with insufficient cells are skipped with a `UserWarning`. If all
strata are filtered, a `ValueError` is raised. `stratify_by` columns must
not collide with `groupby` or `test_col`.

> [!NOTE]
> When `stratify_by` is provided, `rank_genes_groups()` returns a pandas
> DataFrame instead of writing to `adata.uns`. Without `stratify_by`, it
> writes to `adata.uns["rank_genes_groups"]` as usual and returns `None`.

> [!NOTE]
> `pydeseq2` is an **optional** runtime dependency, and since v0.13 it is no
> longer on the default path: `pseudobulk_dex()` defaults to `backend="nb_glm"`
> (below), which needs nothing extra. Reach for `backend="pydeseq2"` when you
> need exact DESeq2 numerics, `stratify_by=`, or `aggr_method="mean"` — and
> install it with `pip install 'pyscx[pydeseq2]'`.

## NB-GLM backend (Rust-native pseudobulk DE)

SCX includes a CPU, `f64`, dependency-free **negative-binomial GLM** that
implements the DESeq2 *core* (IRLS / Fisher scoring + Cox–Reid dispersion +
parametric trend fit + empirical-Bayes shrinkage + Wald inference). It is a
DESeq2-*style* — **not** DESeq2-*identical* — estimator: the bar is ranking /
effect-sign / significance parity, not bit-for-bit numerics. Keep PyDESeq2 when
you need exact DESeq2 behaviour. There is **no GPU path** (no `device=` argument).

By default it fits a fixed `[intercept, is_target]` design per non-reference level.
Pass a `design=` **formula** (e.g. `"~ perturbation + donor"`) to fit a
covariate-adjusted joint model instead — built via `formulaic` (the parser pydeseq2
uses; needs the `nbglm` extra) with one shared-dispersion fit and per-level
contrasts. See [docs/pseudobulk_nb_glm.md § Custom designs](../pseudobulk_nb_glm.md#custom-designs-formula).

Three entry points, all CPU-only:

```python
import pyscx

# 1. pseudobulk_dex with backend="nb_glm" — same pandas schema as pydeseq2.
df = pyscx.accel.pseudobulk_dex(
    adata, groupby=["perturbation", "donor"],
    test_col="perturbation", reference="control",
    backend="nb_glm",                       # no pydeseq2 needed
)

# 2. pdex_nb_glm — cell-eval/pdex column schema, from an AnnData.
df = pyscx.accel.pdex_nb_glm(
    adata, "perturbation", "control",
    stratify_by=["donor"],                  # forms pseudobulk REPLICATES (required)
    # output="polars",                      # opt in when feeding cell_eval
)
# columns: target, feature, fold_change, p_value, fdr, log2_fold_change,
#          abs_log2_fold_change  (byte-compatible with pdex_ref / rank_genes_groups_df)

# 3. accel.nb_glm — direct, on an already-pseudobulked matrix + numeric design.
df = pyscx.accel.nb_glm(counts, design, contrast=1)
# columns: gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj,
#          dispersion, cooks, converged, n_iter
```

> [!IMPORTANT]
> **Replicate requirement.** A pseudobulk NB-GLM needs **≥ 2 pseudobulk samples
> per condition** to estimate dispersion. `pdex_nb_glm` forms one sample per
> `(perturbation × stratum)`, so a `stratify_by` spanning ≥ 2 strata (batch /
> donor / well / replicate) is **required** — it errors with a clear message
> (pointing to `pdex_ref` / `rank_genes_groups`) when absent. cell-eval's default
> one-profile-per-perturbation layout has no replicates and is **not** a valid
> NB-GLM input.

The route is recorded as `route="cpu_nb_glm"` on
`adata.uns["scx_accel"]["pseudobulk_dex"]` / `["pdex_nb_glm"]`. Full guide,
options, and algorithm details:
[docs/pseudobulk_nb_glm.md](../pseudobulk_nb_glm.md).

## See also

- [Column-major (CSC) dispatch](accel-csc.md) and
  [GPU-supported vs GPU-fast](accel-gpu.md#gpu-supported-vs-gpu-fast) — which DE route runs
  and why.
- [Perturbation evaluation metrics](accel-perturbation-metrics.md) — including
  `rank_genes_groups_df`, the cell-eval DE format bridge.
- [docs/pseudobulk_nb_glm.md](../pseudobulk_nb_glm.md) — the NB-GLM backend in
  depth.
