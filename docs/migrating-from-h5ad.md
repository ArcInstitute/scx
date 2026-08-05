# Migrating from h5ad to SCX

For analysts who already have an h5ad/scanpy workflow. This page answers three
questions: **which loader do I use, what changes when I convert, and what (if
anything) is lost in the round trip?**

> **Status:** pre-1.0 (v0.7.x). The on-disk format is stable for read-back at
> `format_version` 3; the spec freezes and gains published conformance vectors at
> 1.0.

## TL;DR

```python
import pyscx

# Convert once (Python):
pyscx.write(adata, "data.scx")          # mirrors adata.write_h5ad(...)

# Read it back:
adata = pyscx.read("data.scx")          # mirrors sc.read_h5ad(...)
exp   = pyscx.open("data.scx")          # a handle (lazy); .to_anndata() / .query()
```

Or from the command line (streaming by default):

```bash
# Minimal:
scx convert data.h5ad data.scx

# Production-ready (indexes + CSC sidecar + detection bitmaps):
scx convert data.h5ad data.scx \
  --index-preset cellxgene \
  --csc auto \
  --bitmap auto
```

`pyscx.read` / `pyscx.write` are the flat one-liners. `pyscx.open(path)` returns
an `Experiment` handle whose `repr` looks like an AnnData:

```
>>> pyscx.open("data.scx")
Experiment object with n_obs × n_vars = 2700 × 32738
    obs: 'n_genes', 'percent_mito', 'louvain'
    var: 'gene_ids', 'n_cells'
    uns: 'louvain_colors'
    obsm: 'X_pca', 'X_umap'
```

`Experiment.info()` carries the on-disk codec / shard / format-version details.

## Which loader do I use?

```
                         Is your dataset small enough
                         to fit in memory (~500K cells)?
                                    │
                        ┌───yes─────┴──────no───┐
                        ▼                       ▼
                  pyscx.read(path)        Do you need
              (full scanpy compat)        the full dataset?
                                                │
                                   ┌───yes───────┴──────no───┐
                                   ▼                         ▼
                          to_anndata(backed=True)      .query().filter_obs()
                          + pyscx.accel.*              .collect()  (extract a
                          (out-of-core)                subset, then in-memory)
```

| Need | Use | Notes |
|------|-----|-------|
| Dataset fits in RAM | `pyscx.read(path)` / `exp.to_anndata()` | Returns a regular AnnData; every `sc.pp.*` / `sc.tl.*` works |
| Atlas-scale, full dataset | `exp.to_anndata(backed=True)` + `pyscx.accel.*` | ~1-shard working set; use `pyscx.accel.*` for preprocessing, not `sc.pp.*` |
| A cell/gene subset of a large file | `exp.query().filter_obs(...).select_genes(...).collect()` | Predicate pushdown skips non-matching shards |
| Reading from cloud storage | `pyscx.read_cloud(url, obs_filter=..., var_names=...)` | `gs://` / `s3://` / `az://` / `file://`; fetches only matching shards |
| A dense / narrow matrix for sklearn / PyTorch / scVI | `exp.to_anndata(container="dense", data_dtype=...)` | Row-major ndarray and/or narrowed dtype (`float16`/`uint8`/…), fail-loud gate — see [docs/api.md](api.md#container-and-dtype-materialization) |
| Training a model | `TrainingDataset` / `IndexPlanDataset` | See [docs/training.md](training.md) |

Full trade-off table:
[docs/scanpy.md § Choosing the right approach](scanpy.md#choosing-the-right-approach).

## Converting h5ad to SCX

Three entry points, from simplest to most scalable:

| Entry point | When to use |
|---|---|
| `pyscx.write(adata, path)` / `pyscx.from_anndata(adata, path)` | You already have an AnnData in memory |
| `pyscx.from_h5ad(path, out)` | Large h5ad on disk — never materialises X in Python; bounded RSS |
| `scx convert data.h5ad data.scx` | Shell scripts, pipelines, no Python needed |

All three accept the same optimization kwargs / flags and produce identical
on-disk output. Streaming is the default for `from_h5ad` and the CLI. For
detailed semantics of each entry point, see
[docs/scanpy.md § Converting existing data to SCX](scanpy.md#converting-existing-data-to-scx).

### Basic usage

**CLI:**

```bash
# Format is auto-detected from file extensions:
scx convert data.h5ad data.scx

# Explicit format flags (useful for non-standard extensions):
scx convert input output --from h5ad
```

**Python:**

```python
import pyscx

# From an in-memory AnnData:
pyscx.write(adata, "data.scx")               # alias for from_anndata
pyscx.from_anndata(adata, "data.scx")

# Streaming directly from h5ad on disk (bounded RSS):
pyscx.from_h5ad("data.h5ad", "data.scx")
```

### Conversion optimizations

SCX supports several on-disk sidecars and indexes that accelerate downstream
operations. All of them can be built at conversion time with a single command.

#### Predicate indexes

Predicate indexes enable `filter_obs()` pushdown — the query engine skips entire
shards whose obs values don't match the predicate, giving sub-second cell-type
lookups on atlas-scale files.

**CLI** (`--index-obs`, `--index-var`, `--index-preset`)**:**

```bash
# Index specific obs/var columns:
scx convert data.h5ad data.scx \
  --index-obs cell_type,tissue,donor_id \
  --index-var feature_name

# Or use a named preset (missing columns warn, don't fail):
scx convert data.h5ad data.scx --index-preset cellxgene
```

**Python** (`index_obs=`, `index_var=`, `index_preset=`)**:**

```python
# Index specific columns:
pyscx.from_h5ad("data.h5ad", "data.scx",
                index_obs=["cell_type", "tissue", "donor_id"],
                index_var=["feature_name"])

# Named preset:
pyscx.from_anndata(adata, "data.scx", index_preset="cellxgene")
```

Available presets:

| Preset | obs columns | var columns |
|--------|-------------|-------------|
| `cellxgene` | `cell_type`, `cell_type_ontology_term_id`, `tissue`, `tissue_ontology_term_id`, `disease`, `assay`, `donor_id`, `development_stage`, `sex`, `suspension_type` | `feature_name`, `feature_type` |
| `perturbseq` | `cell_type`, `donor`, `batch`, `condition`, `perturbation`, `guide_id`, `target_gene`, `control`, `split` | `feature_name`, `feature_type` |
| `training` | `cell_type`, `donor`, `batch`, `dataset_id`, `split`, `organism`, `tissue` | `feature_name`, `feature_type` |

When no explicit columns or preset are supplied, low-cardinality categorical
columns (≤ 1000 categories by default) are auto-detected and indexed. Tune the
threshold with `--index-auto-threshold N` / `index_auto_threshold=N`.

#### CSC sidecar

A CSC (column-major) sidecar accelerates gene-axis operations (DE, pseudobulk,
GPU-accelerated DE). For full details on when to use it, the `off`/`auto`/`always`
policy, and multi-shard layout, see
[docs/sharding.md § CSC sharding](sharding.md#csc-sharding).

**CLI** (`--csc`, `--csc-cols-per-shard`)**:**

```bash
scx convert data.h5ad data.scx --csc auto
scx convert data.h5ad data.scx --csc always --csc-cols-per-shard 10000
```

**Python** (`csc=`, `csc_cols_per_shard=`)**:**

```python
pyscx.from_h5ad("data.h5ad", "data.scx", csc="auto")
pyscx.from_anndata(adata, "data.scx", csc="always", csc_cols_per_shard=10000)
```

> **Note:** The `training` and `perturbseq` index presets automatically upgrade
> an unset `csc` to `"auto"`. An explicit `csc="off"` / `--csc off` overrides
> this.

#### Detection bitmaps

Detection bitmaps are per-shard Roaring bitmap sidecars that accelerate
`pyscx.detection_counts` / `cells_expressing` queries. For the wire format and
`auto` heuristic details, see
[docs/format.md § 12. Detection Bitmap](format.md#12-detection-bitmap-optional).

**CLI** (`--bitmap`)**:**

```bash
scx convert data.h5ad data.scx --bitmap auto
scx convert data.h5ad data.scx --bitmap always
```

**Python** (`bitmap=`)**:**

```python
pyscx.from_h5ad("data.h5ad", "data.scx", bitmap="auto")
pyscx.from_anndata(adata, "data.scx", bitmap="always")
```

#### Sort-on-convert

Pre-sort cells by a key so CSR shards — and the predicate index — are contiguous
per category, giving optimal read locality for the dominant query axis. For full
rationale, sort strategy selection, and benchmarks, see
[docs/sharding.md § Sorting for read locality](sharding.md#sorting-for-read-locality-scx-sort).

**CLI** (`--sort-by`, `--sort-reverse`)**:**

```bash
scx convert data.h5ad data.scx \
  --sort-by cell_type,tissue \
  --index-obs cell_type,tissue
```

**Python** (`sort_by=`, `reverse=`)**:**

```python
pyscx.from_h5ad("data.h5ad", "data.scx",
                sort_by=["cell_type", "tissue"],
                index_obs=["cell_type", "tissue"])

pyscx.from_anndata(adata, "data.scx",
                   sort_by=["donor_id"], reverse=True)
```

### Codec and shard tuning

`codec=` is a three-profile **intent axis**: `auto` (default, cost-aware adaptive
— adopts the compact ShufDeltaZstd per integer shard where it wins by a margin,
else the Scx1/Zstd heuristic; float → Pcodec), `fast` (decode-speed-max heuristic
— the pre-flip default), and `compact` (size-max). For the full spec see
[docs/codec.md § The codec intent axis](codec.md#the-codec-intent-axis-auto--fast--compact).
For shard sizing guidelines, see
[docs/sharding.md § Shard sizing guidelines](sharding.md#shard-sizing-guidelines).

> **Default change (2026-07-12).** The default `codec="auto"` now writes
> **size-optimizing adaptive** output (predominantly ShufDeltaZstd, 1.3–2× smaller
> on integer counts) instead of the old decode-max Scx1. Round-trip fidelity is
> unchanged. If you need the old decode-max behavior for latency-critical CPU
> training, pass `codec="fast"`. The former `codec="auto_v2"` + `decode_target`
> were removed (pre-1.0 clean break) — use `auto` (or `compact` for max size).

**CLI** (`--codec`, `--shard-size`)**:**

```bash
scx convert data.h5ad data.scx --codec auto      # recommended (default, adaptive/size)
scx convert data.h5ad data.scx --codec fast      # decode-speed-max (old default)
scx convert data.h5ad data.scx --codec compact   # size-max
scx convert data.h5ad data.scx --shard-size 8192 # smaller shards (default: 16384)
```

**Python** (`codec=`, `shard_size=`)**:**

```python
pyscx.from_h5ad("data.h5ad", "data.scx", codec="fast", shard_size=8192)
pyscx.from_anndata(adata, "data.scx", codec="compact", shard_size=32768)
```

### Memory and parallelism

For the full multithreading architecture, see
[docs/multithreading.md](multithreading.md).

**CLI** (`--memory-budget`, `--reader-threads`, `--writer-queue-depth`, `--temp-dir`)**:**

```bash
scx convert data.h5ad data.scx --memory-budget 8G
scx convert data.h5ad data.scx --reader-threads 4
scx convert data.h5ad data.scx --writer-queue-depth 8
scx convert data.h5ad data.scx --temp-dir /scratch/tmp
```

**Python** (`memory_budget=`, `reader_threads=`, `writer_queue_depth=`, `temp_dir=`)**:**

```python
# from_h5ad supports all parallelism knobs:
pyscx.from_h5ad("data.h5ad", "data.scx",
                memory_budget="8G",
                reader_threads=4,
                writer_queue_depth=8,
                temp_dir="/scratch/tmp")

# from_anndata accepts memory_budget (no reader_threads — data is already
# in memory, so there is no HDF5 reader to parallelise):
pyscx.from_anndata(adata, "data.scx", memory_budget="8G")
```

### Other flags

**CLI:**

```bash
scx convert data.h5ad data.scx --strict-uns            # fail on unsupported uns
scx convert data.h5ad data.scx --dense-zero-epsilon 1e-7 # sparsification threshold
scx convert data.h5ad data.scx --stream=false           # legacy materialising path
```

**Python:**

```python
pyscx.from_h5ad("data.h5ad", "data.scx",
                strict_uns=True,
                dense_zero_epsilon=1e-7)

pyscx.from_h5ad("data.h5ad", "data.scx", stream=False)
```

### Production-ready recipes

For a typical atlas-scale dataset destined for query + analysis:

**CLI:**

```bash
scx convert atlas.h5ad atlas.scx \
  --index-preset cellxgene \
  --csc auto \
  --bitmap auto \
  --sort-by cell_type \
  --memory-budget 16G \
  --reader-threads 8
```

**Python:**

```python
# Streaming from disk (recommended for large files):
pyscx.from_h5ad("atlas.h5ad", "atlas.scx",
                index_preset="cellxgene",
                csc="auto",
                bitmap="auto",
                sort_by=["cell_type"],
                memory_budget="16G",
                reader_threads=8)

# From an in-memory AnnData:
pyscx.write(adata, "atlas.scx",
            index_preset="cellxgene",
            csc="auto",
            bitmap="auto",
            sort_by=["cell_type"])
```

### Post-conversion commands

#### Verify the output

For full validation semantics (checksum vs deep, production checkpoints), see
[docs/scanpy.md § Validating files after write or transfer](scanpy.md#validating-files-after-write-or-transfer).

```bash
scx info data.scx                # shape, codec, sidecars, indexes
scx validate data.scx            # checksum validation
scx validate --deep data.scx     # decode every shard + verify canonical CSR
```

```python
exp = pyscx.open("data.scx")
print(exp)                       # shape, obs/var/obsm/uns keys
print(exp.info())                # codec, shard count, format version, has_csc
pyscx.validate("data.scx", deep=True)
```

#### Add a CSC sidecar after the fact

If you converted without `--csc` / `csc=`, you can build the sidecar later. See
[docs/sharding.md § CSC sharding](sharding.md#csc-sharding) for layout details.

```bash
# In place (temp file + atomic rename — a failure leaves data.scx intact)
scx build-csc data.scx --memory-limit 8G

# Or write a copy, leaving the input alone
scx build-csc data.scx data_with_csc.scx --memory-limit 8G
```

```python
pyscx.build_csc("data.scx", memory_limit="8G", csc_cols_per_shard=10000)

# Or copy out:
pyscx.build_csc("data.scx", "data_with_csc.scx",
                memory_limit="8G", csc_cols_per_shard=10000)
```

#### Upgrade an older file (re-encode + row-group-frame)

`scx optimize` re-encodes and canonicalizes CSR shards and row-group-frames them
(codec-agnostic random access via the row-group `BlockIndex`), stamping
`format_version` 4. For full semantics, see
[docs/operations.md § Optimize](operations.md#optimize).

```bash
scx optimize data.scx data_v4.scx
scx optimize data.scx data.scx --force   # in-place
scx validate --deep data_v4.scx
```

#### Convert back to h5ad

For full streaming export details, see
[docs/scanpy.md § Exporting back to h5ad / h5mu](scanpy.md#exporting-back-to-h5ad--h5mu-to_h5ad-to_h5mu).

```bash
scx convert data.scx data.h5ad --reader-threads 4 --memory-budget 8G
```

```python
pyscx.to_h5ad("data.scx", "data.h5ad",
              reader_threads=4, memory_budget="8G")
```

#### Doublet detection after conversion

If your pipeline currently calls a doublet caller inline —
`sc.pp.scrublet(adata)`, `doubletdetection`, `solo`, or scDblFinder in R — that
still works: load the file into memory and call the tool exactly as before.
Nothing about conversion changes it.

What changes is what happens when the file stops fitting in memory. Every
doublet caller is an in-memory, single-sample tool, so the pooled-atlas
`sc.pp.scrublet(adata)` was never really the right shape anyway — it was one
library's worth of tool applied to a merged file. SCX gives you the per-sample
split without materialising the pool, and a way to land the answers back:

```python
r = pyscx.export_batches("data.scx", "batches/", batch_key="donor_id")
# ... run the tool on each r["batches"][i]["path"] ...
pyscx.doublet_import("data.scx", "all_calls.csv", tool="scrublet")
```

Three things differ from the inline version and are worth knowing before you
switch:

- **The join is by key, not row position.** A tool's output can come back in any
  order and it will still land correctly — but the key has to be unique. Run
  `pyscx.diagnose_obs_key("data.scx")` first; on a merged atlas the obs index is
  often *not* unique and you will need `key=["sample_id", "barcode"]`.
- **Uncovered cells are `null`, not `0.0`.** Inline scanpy leaves every cell
  scored because every cell was passed to the tool. Here, a cell no tool saw
  keeps a null score — which is what makes `pyscx.doublet_consensus` able to
  distinguish "unassessed" from "all tools said singlet".
- **The columns are canonical, not tool-native.** `doublet_import` maps each
  tool's spelling onto `<K>_score` / `<K>_predicted`, so code downstream of it
  never branches on which caller ran. The native columns are kept alongside as
  `<K>_<native>` unless you pass `keep_native_columns=False`.

Full workflow: [docs/scanpy.md § Landing external per-cell
annotations](scanpy.md#landing-external-per-cell-annotations-doublet-detection).

## What changes when you convert

- **`X` is stored as float32 CSR.** A float64 source matrix is downcast (a
  `UserWarning` fires at write). Counts and most normalized values are unaffected
  in practice; if you depend on float64 precision, keep the h5ad. On **read** you
  can go the other way and materialize `X` into a chosen container/dtype —
  `to_anndata(container="dense")` for a row-major ndarray, or
  `data_dtype="float16"` / `"uint8"` to shrink the in-memory footprint (with a
  fail-loud cast gate; pass `allow_lossy=True` to force a narrowing). See
  [docs/api.md § Container and dtype materialization](api.md#container-and-dtype-materialization).
- **obs/var are Arrow-backed**, not HDF5 datasets. Column dtypes, categorical
  categories, and the pandas `ordered` flag round-trip. The index (`obs_names` /
  `var_names`) round-trips as the pandas index.
- **`pyscx.accel.*` replaces `sc.pp.*` in backed mode.** In-memory mode is plain
  scanpy. Backed mode needs the accelerators for preprocessing — `sc.pp.*` would
  force a full materialization. See
  [docs/scanpy.md § scanpy operations in backed mode](scanpy.md#scanpy-operations-in-backed-mode).

## What does and does not round-trip

The canonical, always-current table lives in the API reference:
[docs/api.md § Round-trip fidelity](api.md#round-trip-fidelity). Every conversion
that drops or transforms something also emits a structured warning — see
[docs/api.md § Conversion warnings](api.md#conversion-warnings-convertwarning).
Headlines:

- **Preserved:** obs/var columns + dtypes, ordered categoricals (including
  **integer- and float-keyed** categoricals — e.g. integer cluster labels or
  dose levels), `obsm`/`varm` embeddings, layers, `obsp`/`varp` (as float32 CSR),
  `adata.raw` ([docs/api.md § `adata.raw`](api.md#adataraw)), most `uns` entries
  including `None` scalars (round-trip as Python `None`, e.g.
  `uns['log1p']['base']`).
- **Lossy / transformed (warns):** `X` and `obsp`/`varp` float64 → float32;
  a dense `obsp`/`varp` is re-emitted as sparse (values identical).
- **Dropped (warns):** CSC/unsupported `obsp`/`varp`, pickled `uns` objects,
  obs/var columns with an unsupported dtype, `uns` pandas DataFrames (kept as a
  nested dict + a `FlattenedUnsDataframe` warning, not silently flattened), and
  **compound/structured `uns` arrays** — notably scanpy's `rank_genes_groups`
  (stored as a per-cluster recarray). Export DE results separately before
  converting (`sc.get.rank_genes_groups_df(...)` → CSV/Parquet); the skip emits a
  short `SkippedUnsKey` warning naming the key.

## scanpy-divergence gotchas

A short list of places SCX's accelerators behave differently from a naive
scanpy script. Each is documented in full in `docs/scanpy.md`.

- **MT genes must be tagged explicitly.** `pyscx.accel.calculate_qc_metrics`
  follows scanpy's contract: without `qc_vars=["mt"]` there is no
  `pct_counts_mt` column. The accelerator emits a `UserWarning` when MT-prefixed
  symbols are present but `qc_vars=None`.
- **`seurat_v3` HVG needs raw counts**, not log-normalized values — stash counts
  in a layer before normalizing. A high-cardinality `batch_key` can trigger a
  per-batch LOESS singularity; SCX drops those batches with a warning rather than
  crashing. See
  [docs/scanpy.md § Lazy preprocessing](scanpy.md#lazy-preprocessing-in-backed-mode).
- **Leiden on GPU is not label-stable** vs Python `leidenalg`. `device="auto"`
  on a GPU host uses cuGraph (ARI ≈ 0.92 vs leidenalg). Pin `device="cpu"` to
  preserve label stability for downstream DE / annotation transfer. See
  [docs/scanpy.md § Leiden clustering](scanpy.md#leiden-clustering-pyscxaccelleiden).
- **Backed-mode preprocessing is lazy on CPU, eager on GPU.** CPU wraps `X` in a
  lazy transform; GPU streams through a fused kernel and materializes a scipy
  CSR. See [docs/scanpy.md § Lazy vs eager preprocessing](scanpy.md#lazy-vs-eager-preprocessing).
- **`pct_counts_<qc_var>` changed on lazy input.** After a lazy
  `normalize_total`, `calculate_qc_metrics` now takes the `qc_var` subset sums
  *through* the transform chain, so the ratio divides a transformed numerator by
  a transformed denominator. It previously used a pre-transform numerator, which
  could be off by more than 2×. There is **no runtime signal** for this: a
  pipeline that filters on `pct_counts_mt` after a lazy normalize will keep and
  drop different cells than it did before. Re-check any thresholds tuned against
  older output.
- **Row-axis aggregations on a projected lazy `X` changed.** They used to sum the
  full physical width: after `filter_genes` (or any column projection) on a lazy
  dataset, `adata.X.sum(axis=1)` included the genes that had just been removed,
  and `adata.X.mean(axis=1)` divided that physical-width sum by the *visible*
  column count. `filter_cells` on a lazy dataset thresholded the same
  physical-width totals. All of them now cover only the visible genes, matching
  `adata.obs["total_counts"]` and scanpy on the sliced object. There is **no
  runtime signal** for the change — a pipeline that filtered cells after a lazy
  `filter_genes` will now keep and drop different cells. Re-check thresholds
  tuned against older output. (`getnnz` was already projection-aware, and the
  backed — non-lazy — dunders were already correct.)
- **`adata[:, mask]` on a backed AnnData used to raise.** anndata had no view
  registration for an SCX handle, so the var axis was unreachable through the
  public API — including `sc.pp.filter_genes`, which calls
  `adata._inplace_subset_var`. It now works: `adata[:, mask]` is a lazy view,
  `adata[:, mask].copy()` materializes, and the backed filter ops subset `raw`
  and drop unused categorical levels like scanpy does. See [docs/api.md § Axis
  subsetting and aligned
  members](api.md#axis-subsetting-and-aligned-members).
- **A backed axis subset now deep-copies `uns`.** anndata builds the replacement
  object with `deepcopy(uns)`, so an entry that cannot be deep-copied — a lock,
  an open file handle, a live client — makes `filter_cells` / `filter_genes`
  raise `TypeError` where the older backed path silently left `uns` alone. This
  matches what an in-memory AnnData has always done. Keep non-copyable objects
  out of `uns`, or drop them before filtering.

## Errors you might see

SCX maps failures to the Python exception a scanpy user expects:

- Missing input file → `FileNotFoundError`.
- Truncated / corrupt / wrong-magic / version-mismatch file → `ValueError` (the
  message says the file looks corrupt or was written by an incompatible SCX).
- A stale CSC sidecar → `ValueError` naming the `--rebuild-csc` fix.
- A non-canonical scipy CSR passed to an accelerator → `ValueError` suggesting
  `X = X.tocsr(); X.sort_indices()` or a re-run of `pyscx.from_anndata`.

## See also

- [docs/quickstart.md](quickstart.md) — 5-minute end-to-end pipeline.
- [docs/scanpy.md](scanpy.md) — the full scanpy integration story.
- [docs/api.md](api.md) — API reference, fidelity table, conversion warnings.
- [docs/training.md](training.md) — ML training loaders.
