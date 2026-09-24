# Gene-set scoring and PFlog normalization

> Part of the [SCX + scanpy guide](README.md). Shared conventions are on the
[accelerators overview](accelerators.md).

## Gene-set scoring (`pyscx.accel.score_genes`)

CPU-native equivalent of `sc.tl.score_genes` — a per-cell score for a gene
signature, written to `adata.obs[score_name]`. Streams shard-by-shard, so it
runs identically on in-memory, backed, and lazy `X` with bounded memory.

Three methods via `method=`:

| `method`    | Score per cell                                                   | Notes |
|-------------|------------------------------------------------------------------|-------|
| `"control"` | `mean(gene_list) − mean(control)` (default; scanpy `score_genes`)| Control genes sampled from expression-matched bins. Pass `ctrl_genes=` to supply them instead and get exact scanpy parity. |
| `"mean"`    | `mean(gene_list)`                                                | Fastest; no control set, ignores `gene_pool`/`ctrl_size`/`n_bins`. |
| `"zscore"`  | `Σ (xᵍ − meanᵍ)/stdᵍ / √k` over the set                          | decoupler [`mt.zscore`](https://decoupler.readthedocs.io/en/latest/api/generated/decoupler.mt.zscore.html); per-gene std uses ddof=1. |

```python
import pyscx

adata = pyscx.open("pbmc.scx").to_anndata(backed=True)

# scanpy-style control scoring (default method)
pyscx.accel.score_genes(
    adata,
    ["CD3D", "CD3E", "CD8A", "GZMB"],   # gene_list (symbols, resolved vs var_names)
    ctrl_size=50,
    n_bins=25,
    score_name="t_cell_score",
)
adata.obs["t_cell_score"]   # per-cell signature score

# lightweight alternatives when score_genes' control sampling is too slow
pyscx.accel.score_genes(adata, marker_genes, method="mean",   score_name="sig_mean")
pyscx.accel.score_genes(adata, marker_genes, method="zscore", score_name="sig_z")

# exact parity: score against a control set you chose (or scanpy chose)
pyscx.accel.score_genes(
    adata,
    ["CD3D", "CD3E", "CD8A", "GZMB"],
    ctrl_genes=my_control_genes,       # no binning, no sampling
    score_name="t_cell_score",
)
```

> **Divergence from scanpy, and how to avoid it.** The `control` method
> replicates scanpy's rank-binning + control-gene sampling algorithm, but the
> sampler is Rust-native (a fixed `ChaCha8` stream seeded by `random_state`,
> reproducible across `rand` upgrades) and seeded independently of numpy. It
> therefore draws *different* control genes, and the scores differ by more than
> rounding.
>
> Measured against scanpy 1.12 on a synthetic 400 × 2000 matrix with
> log-normally distributed gene means, at **all defaults** (`ctrl_size=50`,
> `n_bins=25`): Spearman **0.958**, maximum absolute difference **0.124** against
> a score range of 1.095 — about **11 %** of the range. At 5000 genes, 0.971 and
> 0.232 of 1.824. Forcing more sampling widens it: `ctrl_size=25` gives 0.891,
> `ctrl_size=10` gives 0.721.
>
> The divergence disappears only when scanpy does not sample at all — when
> `ctrl_size` is at least the bin size (`round(n_genes / (n_bins − 1))`), whole
> bins are taken and both implementations pick the same set, agreeing to ~1e-7.
> That is worth knowing before writing a parity check: at ~1200 genes with the
> defaults the bin size *is* 50, so a test written there passes without
> exercising anything.
>
> **`ctrl_genes=` is the exact-parity route.** It takes the control set as an
> argument and skips the selection entirely, so the score matches
> `sc.tl.score_genes` given the controls scanpy used — and it works on a backed
> `X`, which `sc.tl.score_genes` refuses outright (`NotImplementedError`). It
> requires `method="control"` and rejects an explicit `gene_pool=`, which exists
> only as the universe to sample from; `ctrl_size` / `n_bins` / `random_state`
> are ignored, there being no sampling left to steer.
>
> Genes in `gene_list`, `ctrl_genes` (or an explicit `gene_pool`) not present in
> `adata.var_names` are dropped with a `UserWarning`; an empty `gene_list` or
> `ctrl_genes` after resolution raises. Use `layer=` to score a named layer
> instead of `X`. CPU-only — `device` is accepted for API symmetry but there is
> no GPU kernel.

**What `ctrl_genes=` on a backed `X` is worth.** Scoring K = 25 / 100 / 500
panels, `pyscx.accel.score_genes(method="control", ctrl_genes=...)` on a
backed `X` against `sc.tl.score_genes` on an eager one — which is the only
comparison available, since scanpy refuses the backed input:

| Dataset | pyscx (backed) | scanpy (eager) | speedup | peak RSS |
|---|---:|---:|---:|---|
| pbmc10k (11.5K) | 1.28 s | 0.94 s | **0.7×** | 327 MB vs 848 MB |
| tabula_sapiens_100k | 3.21 s | 22.09 s | 6.9× | 487 MB vs 3,666 MB |
| census_1m | 18.76 s | 55.76 s | 3.0× | **821 MB vs 23,531 MB** |

Scores are **bit-identical** — Spearman 1.0, maximum absolute difference 0.0
at all three panel sizes on every fixture except `smartseq2`, where scanpy's
float32 accumulator puts it at 0.017–0.038 (Spearman still 1.0). SCX is
slower below ~10K cells, where the matrix fits in cache and the streaming
decode is overhead the eager path does not pay. The memory ratio never
crosses, and at 1M cells it is 29×. Full tables and provenance:
[docs/performance/accel-qc-de-integration.md § Gene-set scoring vs scanpy](../performance/accel-qc-de-integration.md#gene-set-scoring-vs-scanpy-accel_score_genes).

## PFlog normalization (`pyscx.accel.pflog`)

PFlog (v4, the **shifted-log** transform on raw counts, Booeshaghi et al.,
DOI 10.1101/2022.05.06.490859) is a variance-stabilizing transform with **no
direct scanpy function**. Counts are shifted by a single **matrix-wide** Anscombe
pseudocount `1/(4α)`, log-transformed, and centered by subtracting the within-cell
mean:

```
z_ij = log(x_ij + 1/(4α)) − (1/D) Σ_k log(x_ik + 1/(4α))
```

`α` is the negative-binomial overdispersion of the matrix (`Var = μ + α·μ²`),
estimated once from the counts (`alpha=None`, the default) or pinned
(`alpha=<float>`, e.g. a reference `α` reused across datasets). Unlike v2 there is
**no per-cell depth** — it cancels under the Anscombe scale.

**How `α` is estimated.** Per gene, method-of-moments
`α_g = (var_g − mean_g) / mean_g²`; the matrix-wide `α` is the **median over
every gene whose mean exceeds `1e-3`**, negative `α_g` included. Two details are
load-bearing:

- *Nothing is filtered by dispersion.* Dropping the genes with `var_g ≤ mean_g`
  before the median keeps only the upper tail of sampling noise and biases `α`
  high by a factor that grows as the true dispersion falls. On a matrix where 20
  of 24 genes are under-dispersed, that version reported the remaining four
  genes' dispersion as the whole matrix's and reported success while doing it.
  Fixed; a run from before the fix will show a smaller `n_genes_used` and a larger
  `α` on the same counts.
- *A median, not a mean or a `Σ(var−mean)/Σmean²` ratio.* The median is the only
  one of the three that survives an outlier: a single highly-expressed
  over-dispersed gene — a mitochondrial or ambient-RNA spike, routine in real
  counts — moves the moment-pooled estimate by more than an order of magnitude
  and leaves the median where it was.

If the pooled median comes out non-positive the matrix carries no NB
overdispersion to report, so `α` falls back to `0.25` (pseudocount `1.0`, i.e.
plain `log1p`) and `uns["pflog"]["fell_back"]` is `True` — a clamp to a small
positive floor would instead return a confident number the counts do not support.
The estimator is pinned against counts simulated from a **known** `α` in
`scx-accel/src/pflog_reference_tests.rs`; regenerate those fixtures with
`.venv/bin/python benchmarks/scripts/generate_pflog_alpha_references.py`.

The exact output is **dense** (zeros map to a per-cell baseline), so a naïve
materialization is `O(N·D)`. The accelerator avoids that by exploiting the
decomposition `Z = delta + baseline·1ᵀ`, where `delta = log1p(4α·x)` is exactly the
lazy `scale(4α) → log1p` chain (sparse, same pattern as `X`) and
`baseline_i = −(1/D) Σ_j delta_ij` is one float per cell. So the out-of-core PCA
never densifies, and a compact on-disk form stores only `delta` + `baseline`.

Operates on **raw counts** — run it on the raw-count `X`, not a normalized
layer. Streams shard-by-shard, so it runs identically on in-memory, backed, and
lazy `X` (a lazy `X` that already carries transforms is rejected).

> **Default is a PCA embedding, not an in-place `X` transform.** Unlike
> `normalize_total` / `log1p` (which overwrite `adata.X`), `pflog` defaults
> to `store="pca"`: it writes a baseline-aware PCA embedding to
> `adata.obsm[obsm_key]` (plus the per-cell baseline to `adata.obs[baseline_key]`)
> and **leaves `X` as raw counts**. To get the normalized matrix itself, pass
> `store="dense"` (with `out=<path.scx>` for data too large to densify in memory).
> The fit is recorded in `adata.uns["pflog"]` (`alpha`, `pseudocount`, …).

| `store`            | writes                                                          | transforms `X`? |
| ------------------ | --------------------------------------------------------------- | --------------- |
| `"pca"` (default)  | `obsm[obsm_key]` + `uns[f"{obsm_key}_singular_values"]`, `obs[baseline_key]` | no |
| `"baseline"`       | `obs[baseline_key]` only                                        | no              |
| `"dense"`          | `layers[layer_out]` (or a new SCX file via `out=`)              | no (a layer/file) |
| `"all"`            | both the `"pca"` and `"dense"` outputs                          | no              |

```python
import pyscx

adata = pyscx.open("pbmc.scx").to_anndata(backed=True)

# Headline path: out-of-core baseline-aware PCA embedding (α estimated once).
pyscx.accel.pflog(adata, store="pca", n_components=50)
adata.obsm["X_pflog_pca"]      # cells × n_components
adata.obs["pflog_baseline"]    # per-cell baseline (always written)
adata.uns["pflog"]             # {"alpha", "pseudocount", "alpha_source", ...}

# Precompute-once / train-many: stream the transform to a compact SCX file
# (sparse `delta` + `baseline` obs column), then reconstruct exact dense rows.
pyscx.accel.pflog(adata, store="dense", out="pbmc_pflog.scx")  # store_repr="delta_baseline"
re = pyscx.open("pbmc_pflog.scx").to_anndata()
Z = pyscx.accel.pflog_reconstruct(re)   # exact dense Z = delta + baseline[:, None]
```

> **Representations & codecs.** `store_repr="delta_baseline"` (default) is the
> compact `O(M)` form — the `delta` layer is written as a sparse CSR with
> **Pcodec** float values (the natural codec for log-ratios) and `baseline`
> rides in `obs`; reconstruct with `pyscx.accel.pflog_reconstruct` (or feed
> the file to `TrainingDataset` with its transform mode off). `store_repr="dense"`
> writes the literal full-density `Z` as a CSR with **forced Zstd** values and a
> small default `shard_size` (peak RAM per shard ≈ `2·shard_rows·n_vars·4 B`),
> for downstream tools that need a plain dense layer — it is `O(N·D)` on disk, so
> prefer the compact default at atlas scale. Without `out=`, `store="dense"`
> materializes into `adata.layers[layer_out]` guarded by `dense_max_elems`.
> CPU-only — `device` is accepted for API symmetry but there is no GPU kernel.

## See also

- [Lazy preprocessing](lazy-preprocessing.md) — the lazy
  `normalize_total` → `log1p` chain.
- [docs/training.md](../training.md) — PFlog in the ML training loader.
