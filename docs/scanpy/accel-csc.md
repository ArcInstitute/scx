# `prefer_format="auto"|"csr"|"csc"`: column-major dispatch

> Part of the [SCX + scanpy guide](README.md). See also the [accelerators overview](accelerators.md).

A subset of accelerators take a `prefer_format` kwarg that selects
between the row-major CSR path and the column-major CSC sidecar path.
Entries that accept it:

| Function | CSC win |
|----------|---------|
| `pyscx.accel.highly_variable_genes` | **None on CPU — it is slower.** Single-batch seurat_v3 only (multi-batch and non-seurat_v3 raise). Measured 4.4× slower than CSR at tabula_sapiens_100k (9.8 s vs 2.2 s) and 6.4× at census_1m (84.7 s vs 13.3 s), at 1.5–2.2× the peak RSS, selecting the same genes: the CSC mean/var pass walks every column of every CSC shard for statistics one row-major sweep produces. The kwarg stays because on `device="gpu"` it is also how a *filtered* handle reaches the column-major reduce (`gpu_csc_v3`), which the default auto-route leaves on `gpu_csr`. |
| `pyscx.accel.rank_genes_groups` | Per gene chunk: read CSC slab + scatter into row-major dense buffer (vs decode every row + project for CSR). Clearest CSC win — how large depends on the shard cache, see below. |
| `pyscx.accel.pdex_ref` | Per gene chunk: read CSC slab + scatter into row-major dense buffer (vs decode every row + project for CSR). Takes `prefer_format="auto"` (default), `"csr"`, or `"csc"`. |
| `pyscx.accel.pseudobulk_dex` | Filtered-gene subsets only (`gene_indices=...` or column projection on `adata.X`). Full-gene pseudobulk has no CSC win and raises. |
| `pyscx.accel.calculate_qc_metrics` | Gene-axis aggregations only (`total_counts`, `n_cells_by_counts`, and the `mean_counts` / `pct_dropout_by_counts` derived from them); cell-axis stays CSR. Both axes take one shard pass each for up to 64 `qc_vars`, with one extra row pass per additional 64, and `percent_top` adds none. `prefer_format="csc"` rejects a scipy/dense `X` and a layer source (`layer=`, or a layer handle assigned to `X`); `layer=` works on the default CSR route. |
| `pyscx.accel.col_sums` / `col_nnz` / `col_min` / `col_max` / `col_var` | **None on a whole-matrix reduction — CSR is faster on all five.** Measured 2.4× (`col_var`) to 4.8× (`col_sums`) slower at tabula_sapiens_100k and 4.0× (`col_var`, 42.1 s vs 10.5 s) to 8.5× (`col_sums`, 42.2 s vs 5.0 s) at census_1m, identical results. What `"csc"` offers is the one thing CSR cannot do: serve a column reduction on a **lazy** `ScxLazyTransformedDataset`, which the CSR route refuses. Peak memory is bounded by one CSC shard: until this release an unprojected handle decoded the whole sidecar as one slab (12.9 GB at census_1m, against 5.7 GB now). |
| `pyscx.accel.pca` | **Rejects `prefer_format="csc"`** with `ValueError`. Covariance build and randomized SpMM are row-major; CSC offers no measurable speed-up. |

## What the CSC route is worth, and in which regime

Every CPU number below was taken on a **backed** handle, and for DE the shard
cache the handle was opened with decides most of the answer. `to_anndata(backed=True)`
defaults to a 4-shard cache; once a Wilcoxon pass visits more CSR shards than
that, every gene chunk re-decodes every shard (a pass at the default
`gene_chunk_size` is ~123 chunks on a 61.5K-gene file). The CSC route reads each
column once either way, so its advantage is largest exactly where the CSR route
thrashes:

- **DE at the default cache** — 13.7× (CSC densify) and 21.7× (exact-nnz
  kernel) over CSR on tabula_sapiens_100k, in
  [performance/accel-qc-de-integration.md § Differential expression (CPU, full-matrix)](../performance/accel-qc-de-integration.md#differential-expression-cpu-full-matrix);
  6.2× on census_1m behind `normalize_total → log1p` (1,459.7 s → 235.2 s,
  7.5 → 5.8 GB peak RSS; a one-off A/B, one build, two runs each, spread
  < 2.5 %).
- **DE with a cache sized to the file** — the CSR route stops re-decoding and
  the sidecar's margin drops to 1.45× at census_1m (585.9 s vs 404.1 s) and
  1.95× at tabula_sapiens_100k, with the CSR arm at 83 GB of RSS to get there.
  That is the regime `bench_csc_dispatch`'s `de_csr` / `de_csc` arms measure (the
  `LATEST` baseline); its `de_csr_bounded` arm is the same CSR call at the
  default cache.
- **HVG and the whole-matrix `col_*` reductions** are one pass over the matrix,
  so the cache does not enter into it — and the CSC route loses on both (the
  table above has the numbers). They make no argument for a sidecar.

The HVG and `col_*` figures are one-off captures on one 16-core `cpu` node, the
HVG arms at `d55ebf4f` and the `col_*` arms on the build that introduced the
shard-bounded CSC walk and the unprojected CSR kernels (median of 2–3 runs, a
fresh process per run, both arms of a pair on the same file and binary), not
`benchmarks/comprehensive` captures; the
`bench_csc_dispatch` arms `hvg_*` and `col_sums_*` / `col_var_*` measure the
same calls in the suite.

**DE (`rank_genes_groups`, `pdex_ref`) defaults to `"auto"`; every
other `prefer_format`-taking function defaults to `"csr"`.**
`"auto"` (a **compatibility change** in the CPU-accelerator Phase-2
work — DE previously defaulted to `"csr"`) resolves at call time
against the *selected* matrix: on CPU it takes the CSC-direct route when the
file has a sidecar **and** the handle's row window still spans at least half
the CSR shards, and CSR otherwise; on GPU it stays CSR so the planner routes
`gpu_csc_v3` under the same condition. The route and
`csc_available` flag are recorded on `adata.uns["scx_accel"][<op>]`
(`cpu_csc` vs `cpu_csr`; `cpu_csc_nnz` for 1-vs-rest `rank_genes_groups`, whose
CSC route runs the exact-nnz Wilcoxon kernel by default since 0.20 —
bit-identical to the densify kernel and 3.1× / 4.1× faster than it on
tabula_sapiens_100k / census_1m; `SCX_ACCEL_WILCOXON_NNZ=0` restores densify, and
`reference=` / `rankby_abs=True` always use it). Pass `prefer_format="csr"` explicitly to pin
the pre-change behaviour. The non-DE functions keep `"csr"`, and that is a
measured decision rather than a gap: HVG and the `col_*` reductions are slower
on CSC (above), and `calculate_qc_metrics` / `pseudobulk_dex` serve only part of
what they compute from it. No thread-local default; no
env-var override; each call sets the choice locally.

`prefer_format="csc"` requires **one** thing, and raises `RuntimeError`
naming it otherwise: the file has a CSC sidecar
(`pyscx.from_anndata(csc="always"|"auto")`, `scx convert --csc=always|auto`,
`scx build-csc`, or the standalone `pyscx.build_csc(path)` to add one to an
existing file in place — pass an `output` to write a copy instead).

Neither the transform chain, nor a row filter, nor a column projection is a
condition, and all three used to be. Each was a real barrier as written, and
each was removed by making the read path present the handle's own view rather
than the file:

- **The transform chain.** The gate required every transform to be
  *column-local* (output depending only on the element's own column), which
  `Log1p` and `Scale` satisfy and `NormalizeTotal` and `RowScale` do not — so
  the standard `normalize_total → log1p` chain was refused. That was the wrong
  question for a CSC reader, which always knows a nonzero's row because
  `ScxCsc::indices` *is* the global row, and both row-indexed transforms carry
  their per-row vector (`row_sums`, `factors`) with them. They are applied
  column-major by lookup, bit-identically to the CSR path — element-wise maps
  with no accumulation, so exact agreement is achievable and is asserted, not
  approximated.
- **A row filter** (`filter_cells`, `subset_obs`, `adata[mask]`, or a
  `mark_deleted` deletion vector) renumbers the live rows while the sidecar's
  `indices` keep addressing the file's, so a slab handed over unchanged would
  have scattered into the wrong output rows. The read path now renumbers it —
  one pass over the slab, after the transforms and before the projection — so
  what a kernel receives is addressed in live row space, which is what it
  already assumed.
- **A column projection on a backed handle** (`filter_genes`, `subset_var`,
  `highly_variable_genes(subset=True)`, `adata[:, mask]`) used to reach the
  kernel as a full-axis sidecar under a visible-width gene axis, and DE and HVG
  refused rather than risk it. A backed handle now serves its CSC reads through
  the same view a lazy one does, which remaps each shard's columns into the
  projected axis — so every CSC consumer honours a projection on either handle
  kind.

One consequence worth stating plainly: because `"auto"` is DE's default, an
ordinary `filter_cells → filter_genes → normalize_total → log1p →
rank_genes_groups` pipeline now records `cpu_csc` where it recorded `cpu_csr`.
The output is the same (pinned bit-for-bit against the CSR route); the wall
time and the recorded route are not.

**`"auto"` asks a second question that an explicit `"csc"` does not.** Being
*able* to serve a window is not a reason to prefer it: a CSC column shard spans
the whole row axis, so a narrow row window (`adata[:10_000]` of a million
cells) decodes every physical cell of the columns it asks for and discards most
of them, where the CSR path skips the shards the window empties outright. So
`auto` takes CSC only while the kept rows still span **at least half** the CSR
shards — true for the ordinary `filter_cells` that keeps ~99 % of cells, where
every shard retains survivors and nothing is skippable, and false for a slice
confined to a few shards. That cut is coarse and deliberately so: it is chosen
to be right at both ends rather than tuned, and where exactly it belongs in
between is not something the measurements here establish. `prefer_format="csc"`
bypasses it and is served on any window the reader can compact.

Unknown values (e.g. `"CSC"`, `"bogus"`) raise `ValueError`. `"auto"`
is accepted by `rank_genes_groups` / `pdex_ref` (and is their default);
the other `prefer_format`-taking functions accept only `"csr"` / `"csc"`.

```python
import pyscx

# Open a CSC-equipped file
exp = pyscx.open("atlas.scx")  # e.g. written via pyscx.from_anndata(csc="auto") or csc="always"
adata = exp.to_anndata(backed=True)

# DE on a small target gene set — CSC slab read avoids decoding every row
pyscx.accel.rank_genes_groups(
    adata, "perturbation", reference="control",
    prefer_format="csc",
)

# A lazy transform chain keeps CSC capability — including the standard
# normalize_total -> log1p, whose per-row factor is read at the global row
# CSC `indices` already carries.
pyscx.accel.normalize_total(adata, target_sum=1e4)
pyscx.accel.log1p(adata)
pyscx.accel.col_sums(adata.X, prefer_format="csc")  # works

# A row filter keeps it too: the slab's rows are renumbered onto the live
# row space on the way out, so the answer is the visible matrix's.
pyscx.accel.filter_cells(adata, min_genes=200)
pyscx.accel.col_sums(adata.X, prefer_format="csc")  # works

# And so does a gene filter, on a backed handle as well as a lazy one.
pyscx.accel.filter_genes(adata, min_cells=3)
pyscx.accel.rank_genes_groups(adata, "perturbation")  # 1-vs-rest default
adata.uns["scx_accel"]["rank_genes_groups"]["route"]  # 'cpu_csc_nnz' under the "auto" default
# (or 'cpu_csc' with an explicit reference="control" or rankby_abs=True, which use densify)
```

For the on-disk format and sharding granularity, see
[docs/sharding.md § CSC sharding](../sharding.md#csc-sharding) and
[docs/format.md § 4.1 CSC Shard Internal Layout](../format.md#41-csc-shard-internal-layout).

## See also

- [GPU-supported vs GPU-fast](accel-gpu.md#gpu-supported-vs-gpu-fast) — the GPU CSC-direct
  route, which the planner selects without `prefer_format`.
- [docs/sharding.md § CSC sharding](../sharding.md#csc-sharding) — building and
  carrying the CSC sidecar.
