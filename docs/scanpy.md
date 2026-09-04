# Using SCX with scanpy

SCX integrates directly with [scanpy](https://scanpy.readthedocs.io/) and the
[scverse](https://scverse.org/) ecosystem through `pyscx`. Every `pyscx` method
that returns data produces a standard `anndata.AnnData` object — so any scanpy
function works out of the box with zero glue code.

> Migrating an existing h5ad workflow? Start with
> [docs/migrating-from-h5ad.md](migrating-from-h5ad.md) — it has the loader
> decision tree, the round-trip fidelity table, and the handful of
> scanpy-divergence gotchas consolidated in one place.

## Choosing the right approach

SCX offers three ways to work with data in Python. Each makes different
trade-offs between memory usage, scanpy compatibility, and performance:

### Decision tree

```
                         Is your dataset small enough
                         to fit in memory (~500K cells)?
                                    │
                        ┌───yes─────┴──────no───┐
                        ▼                       ▼
                   In-memory              Do you need
               (simplest, full           the full dataset?
              scanpy compat)                    │
                                    ┌───yes─────┴──────no───┐
                                    ▼                       ▼
                            Backed + lazy             Query pipeline
                          (out-of-core with         (extract a subset,
                          pyscx.accel.*)             then work in-memory)
```

### Feature comparison

|  | In-memory | Backed + lazy | Query pipeline |
|--|-----------|---------------|----------------|
| **API** | `to_anndata()` + `sc.pp.*` | `to_anndata(backed=True)` + `pyscx.accel.*` | `.query().filter_obs().collect()` |
| **When to use** | Small–medium datasets that fit in RAM | Atlas-scale datasets (500K–10M+ cells) | Extract a cell/gene subset from a large file |
| **Peak memory** | Full matrix in RAM | ~1 shard working set (~128 MB) | Subset only |
| **scanpy compatibility** | ✅ Full — every `sc.pp.*` / `sc.tl.*` works | ⚠️ Partial — use `pyscx.accel.*` for preprocessing; `sc.tl.*` and `sc.pl.*` work normally | ✅ Full — result is a regular AnnData |
| **Parallel shard decode** | ✅ All shards decoded in parallel via rayon | ✅ Per-access shard decode (parallel for streaming ops) | ✅ Parallel decode of matching shards |
| **GPU accelerators** | ✅ via `pyscx.accel.*(device="gpu")` | ✅ via `pyscx.accel.*(device="gpu")` | ❌ Preprocess in query pipeline runs on CPU; use accelerators after `.to_anndata()` |
| **Predicate pushdown** | ❌ All data loaded | ❌ All data accessible (filtering via deletion vectors) | ✅ Skips non-matching shards entirely |
| **Lazy normalize/log1p** | ❌ Materializes (standard scanpy) | ✅ `pyscx.accel.normalize_total()` / `log1p()` — zero materialization | ✅ `with_normalize()` / `with_log1p()` — applied in Rust during collect |
| **Streaming PCA** | ❌ Requires full matrix | ✅ `pyscx.accel.pca()` streams through lazy transforms | ❌ PCA runs after materialization |
| **Write-back** | ✅ In-place modification of X | ❌ Read-only (use `pyscx.preprocess()` for copy-on-write) | ❌ Read-only |
| **Typical dataset size** | < 500K cells | 500K–10M+ cells | Any size (output is a subset) |

### Pros and cons summary

**In-memory** (`to_anndata()`):
- ✅ Simplest — zero learning curve if you already know scanpy
- ✅ Every scanpy function works without modification
- ✅ Fastest for datasets that fit in RAM (no per-access overhead)
- ❌ Full matrix must fit in memory (e.g., 1M cells × 30K genes at 5% density ≈ 6 GB)
- ❌ No lazy preprocessing — `normalize_total()` and `log1p()` operate on the full matrix

**Backed + lazy** (`to_anndata(backed=True)` + `pyscx.accel.*`):
- ✅ Handles 10M+ cells on modest hardware (~16 GB RAM)
- ✅ Lazy preprocessing keeps data on disk (normalize, log1p, filter)
- ✅ Streaming PCA, kNN, UMAP through lazy transforms
- ✅ GPU accelerators via `device="gpu"`
- ⚠️ Must use `pyscx.accel.*` instead of `sc.pp.*` for preprocessing
- ⚠️ Some scanpy functions still force materialization (see [compatibility table](#scanpy-operations-in-backed-mode))

**Query pipeline** (`.query().filter_obs().collect()`):
- ✅ Predicate pushdown skips non-matching shards (bandwidth savings up to 20×)
- ✅ Normalize + log1p computed in Rust during collect (fast)
- ✅ Result is a regular AnnData — full scanpy compatibility downstream
- ❌ Only useful when you want a subset, not the full dataset
- ❌ No streaming PCA or lazy transforms — analysis starts after materialization

> [!TIP]
> **Hybrid approach:** Use the query pipeline to extract a subset, then
> work in-memory with standard scanpy:
> ```python
> adata = (pyscx.open("atlas.scx")
>     .query()
>     .filter_obs("tissue == 'lung'")
>     .with_normalize(1e4)
>     .with_log1p()
>     .collect()
>     .to_anndata())
> sc.pp.pca(adata)  # regular scanpy from here
> ```
> The query pipeline always **materializes** the matching subset into
> memory. If the subset is still too large to materialize, open in backed
> mode with filtering instead — the data stays on disk:
> ```python
> adata = pyscx.open("atlas.scx").to_anndata(
>     backed=True, obs_filter="tissue == 'lung'")
> pyscx.accel.normalize_total(adata, target_sum=1e4)  # lazy
> pyscx.accel.log1p(adata)                             # lazy
> pyscx.accel.pca(adata, n_comps=50)                   # streaming
> ```

> [!TIP]
> **Multimodal files:** scope the query to one modality with
> `query(modality="rna")`. The obs predicate resolves against the shared
> global obs axis; X / `select_genes` resolve against that modality's var:
> ```python
> rna = (pyscx.open("citeseq.scx")
>     .query(modality="rna")
>     .filter_obs("cell_type == 'T cell'")
>     .select_genes(["MS4A1", "CD3D"])
>     .collect()
>     .to_anndata())
> ```
> On a multimodal file `modality=` is required (omitting raises `ValueError`).
> See [docs/multimodal.md § 3.4](multimodal.md#34-modality-scoped-queries--querymodality).

## Quick start: in-memory

The simplest approach. `to_anndata()` loads the entire expression matrix into
memory as a scipy CSR matrix (via **zero-copy** transfer from Rust). This is
ideal for datasets that fit comfortably in RAM, since the full standard scanpy
API works without modification.

```python
import pyscx
import scanpy as sc

# Open an SCX file and load everything into memory
adata = pyscx.open("experiment.scx").to_anndata()

# Standard scanpy pipeline — nothing changes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color="leiden")
```

> [!IMPORTANT]
> `to_anndata()` **materializes the full matrix** into a standard scipy CSR.
> The resulting AnnData is identical in memory whether loaded from SCX or
> h5ad — SCX's advantage is a smaller file on disk, not a smaller in-memory
> object. For a 1M-cell dataset with 30K genes at 5% density, the scipy
> CSR requires ~12 GB (see [memory formula](#default-behavior-no-extra-params)
> below). If this exceeds your available memory, use
> [backed mode](#backed-mode-lazy-loading) or the
> [query pipeline](#querying-subsets-before-loading) instead.

## Quick start: backed mode (out-of-core)

For large datasets, backed mode keeps data on disk and preprocesses
lazily — peak memory is one shard (~128 MB) rather than the full matrix:

```python
import pyscx
import scanpy as sc

# Open in backed mode — X stays on disk
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# QC and filtering — fully streaming, no materialization
pyscx.accel.filter_cells(adata, min_genes=200)
pyscx.accel.filter_genes(adata, min_cells=3)

# Preprocessing — lazy, data stays on disk
pyscx.accel.normalize_total(adata, target_sum=1e4)
pyscx.accel.log1p(adata)

# Analysis — PCA streams through lazy transforms
sc.pp.highly_variable_genes(adata)
pyscx.accel.pca(adata, n_comps=50)
pyscx.accel.neighbors(adata, n_neighbors=15)
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)                    # Rust-native, 40× faster than leidenalg
sc.pl.umap(adata, color="leiden")
```

> [!NOTE]
> In backed mode, use `pyscx.accel.*` for preprocessing functions
> (`normalize_total`, `log1p`, `filter_cells`, `filter_genes`, `pca`,
> `neighbors`, `umap`, `leiden`). These are designed for out-of-core data and avoid
> materializing the full matrix. Standard `sc.pp.*` functions work for
> operations that don't modify X (e.g., `highly_variable_genes`), but
> `sc.pp.normalize_total()` and `sc.pp.log1p()` will force full
> materialization. See the [compatibility table](#scanpy-operations-in-backed-mode)
> for details.

## Converting existing data to SCX

> **What round-trips?** For the canonical table of which AnnData fields are
> preserved, lossy, or dropped on conversion (and which warning fires for each),
> see [docs/api.md § Round-trip fidelity](api.md#round-trip-fidelity).

> **Benchmarks**: for h5ad → SCX conversion throughput across datasets, codecs,
> and thread counts (including `full` mode that covers the h5ad read + SCX
> write), see [docs/performance.md §Conversion (h5ad → format)](performance.md#conversion-h5ad--format)
> and [§Write Scaling (parallel shard encoding)](performance.md#write-scaling-parallel-shard-encoding).

> **Migrating an existing h5ad workflow?**
> [docs/migrating-from-h5ad.md § Converting h5ad to SCX](migrating-from-h5ad.md#converting-h5ad-to-scx)
> has end-to-end conversion recipes — predicate indexes, CSC sidecars,
> detection bitmaps, sort-on-convert, codec/shard tuning, memory budgets,
> and production-ready CLI + Python examples — all in one place.

### From AnnData / h5ad

> [!NOTE]
> **All h5ad / h5mu ingestion and export entry points stream by default.**
> `pyscx.from_h5ad`, `pyscx.from_h5mu`, `pyscx.to_h5ad`, and
> `pyscx.to_h5mu` all default to `stream=True`; `pyscx.from_anndata`
> auto-routes to the streaming pipeline when given a backed AnnData.
> Peak RSS is bounded by one shard's worth of CSR per matrix
> (plus one shard's worth of obs/var per column when the source
> carries `ObsMetadataShard` / `VarMetadataShard` sections — atlas-scale
> obs no longer needs to live in memory at once during export)
> regardless of total file size. Pass `stream=False` to opt into the
> legacy materialising paths (only useful when you specifically need
> the in-memory shape, e.g. for AnnData mutations the streaming path
> doesn't forward).

```python
import scanpy as sc
import pyscx

# From an AnnData object in memory
adata = sc.read_h5ad("dataset.h5ad")
pyscx.from_anndata(adata, "dataset.scx")

# Optional: control codec and shard size
pyscx.from_anndata(adata, "dataset.scx", codec="auto", shard_size=8192)
```

The `codec` parameter accepts `"auto"` (default — selects best codec per shard),
`"scx1"` (domain-specific integer codec), `"zstd"`, `"pcodec"` (optimal for float layers),
`"lz4"` (byte-shuffle + LZ4 frame), or `"none"`. With `"auto"`, integer data uses Scx1 or Zstd
and float data (e.g., log-normalized layers) uses Pcodec for 7–16% better compression than Zstd.

#### Sharded obs/var metadata

For datasets with `n_obs > shard_target_rows` (default 16,384),
`from_anndata` emits obs and var as sharded `ObsMetadataShard` /
`VarMetadataShard` sections rather than single monolithic sections.
This bounds metadata write memory and produces files compatible with the
streaming merge pipeline. Pass `force_legacy_metadata=True` to opt into
single-section metadata when needed for backward compatibility:

```python
pyscx.from_anndata(adata, "output.scx", force_legacy_metadata=True)
```

#### Incremental mapping writes

Obsm, varm, obsp, and varp are extracted and written one key at a time —
the full set of mappings is never collected in memory simultaneously.
When a single mapping's estimated peak footprint exceeds `memory_budget`,
a `MappingPeakFootprintHigh` warning is emitted:

```python
pyscx.from_anndata(adata, "output.scx", memory_budget="4G")
```

### From h5ad on disk — streaming (`from_h5ad`)

For h5ad files that don't fit in RAM, use `pyscx.from_h5ad(path, out)`. It
opens the file in Rust via `scx-convert` and writes one shard's worth of
rows at a time, bounding peak memory to roughly `shard_target_rows × n_vars × density × ~16 bytes` per X shard, plus one row-shard per
`obsm` / `varm` / `obsp` / `varp` matrix (hyperslab-read from the source
h5ad and written as row-aligned sections), plus the always-resident
`indptr` (`(n_obs + 1) × 8` bytes — ~80 MB at 10M cells, ~800 MB at
100M cells). No Python AnnData object is constructed.

```python
import pyscx

# Stream directly from disk — works on files larger than RAM.
pyscx.from_h5ad("very_large.h5ad", "very_large.scx")

# Same codec / shard / CSC options as from_anndata.
pyscx.from_h5ad("very_large.h5ad", "very_large.scx",
                codec="auto", csc="always")
```

`csc="always"` performs a two-pass write: the streaming path emits CSR
shards, then `scx_ops::rebuild_csc_inplace` regenerates the CSC sidecar
over the just-written file. Peak disk briefly reaches ~2× the output
size during the rebuild.

Limitations:

- `obsp` / `varp` ingest comes through only when the on-disk h5ad has them
  in a form anndata exposes (matches the non-streaming CLI converter).

Source-layout handling:

- **CSR-on-disk**: native streaming path.
- **Dense-on-disk**: row-slab streaming with per-shard sparsification
  (zero-drop). Use `dense_zero_epsilon` to threshold near-zero values;
  default `0.0` matches scipy's `csr_matrix(dense)` behavior.
- **CSC-on-disk**: in-memory transpose when the file fits `memory_budget`;
  otherwise an external bucketed transpose to `temp_dir` (scipy
  `sum_duplicates` semantics on duplicate coordinates).


### Backed AnnData — auto-streams via `from_anndata`

`pyscx.from_anndata(adata, out)` now accepts a backed AnnData (one
loaded with `sc.read_h5ad(path, backed='r')`). When `adata.isbacked` is
true, the call routes to the streaming pipeline against the underlying
h5ad file. In-memory mutations to `obs` / `var` / `uns` / `obsm` /
`varm` / `obsp` / `varp` made before the call are preserved verbatim —
they're extracted from Python and passed as overrides so the on-disk
read doesn't clobber them.

```python
adata = sc.read_h5ad("big.h5ad", backed="r")
adata.obs["pheno"] = compute_phenotype(adata)   # in-memory mutation OK
pyscx.from_anndata(adata, "big.scx")            # mutation preserved
```

Caveats:

- Layers always come from the on-disk h5ad. If you mutated
  `adata.layers["foo"]` in Python, `from_anndata` will warn you that
  the in-memory edit is dropped and the on-disk layer is used
  instead. Write the AnnData out to a fresh h5ad first, then convert.
- Zarr-backed or duck-typed AnnData-likes without a resolvable
  `filename` raise `NotImplementedError` pointing at
  `pyscx.from_h5ad(path, out)`.

### From 10x HDF5

```python
pyscx.from_10x("filtered_feature_bc_matrix.h5", "dataset.scx")
```

### From Cell Ranger MTX directory

```python
pyscx.from_mtx("/path/to/filtered_feature_bc_matrix", "dataset.scx")
```

Cell Ranger writes `matrix.mtx` as **features × barcodes** (the size line is
`<genes> <cells> <nnz>`). The reader detects orientation by matching the matrix
dimensions against the `barcodes.tsv` / `features.tsv` lengths and transposes a
features×barcodes matrix to the cells×genes layout SCX stores, so a standard
`filtered_feature_bc_matrix/` converts correctly and `to_anndata()` returns shape
`(n_cells, n_genes)`. An already-cells×genes matrix is kept as-is; a square matrix
is assumed to be Cell Ranger's features×barcodes (transposed, with a warning); and
a matrix whose dimensions match neither layout is a hard error rather than a silent
mis-orientation.

### adata.raw

`adata.raw` (pre-normalization counts on its own, usually wider, var axis) is
preserved. `from_h5ad` ingests the h5ad `/raw` group, `pyscx.open(...).to_anndata()`
reconstructs `adata.raw`, and `to_h5ad` re-emits `/raw/X` + `/raw/var`, so
`h5ad → scx → h5ad` round-trips raw with integer counts bit-exact. The in-memory
`from_anndata(adata)` / `write(adata, path)` path writes the same section family from
`adata.raw.X` + `adata.raw.var`, so both doors preserve raw identically. Raw is dropped
(with a `DroppedRaw` warning) under obs-filtered `to_anndata`, backed mode, and
deletion-vector-active files, and (with `DroppedRawOnWrite`) on two write paths: the
SCX-backed / lazy-`X` rewrite, where the in-memory AnnData's `.raw` is `None`, and
reorder-on-convert (`--sort-by` / `--group-by`), where raw streams unpermuted and would
end up attached to the wrong cells. That warning's remedy is per call site — convert
from the h5ad for the first, convert without the reorder for the second.
`adata.raw.varm` has no section and is dropped with `DroppedRawVarm`.
See [docs/api.md § `adata.raw`](api.md#adataraw).

### Exporting back to MTX

```python
pyscx.to_mtx("dataset.scx", "/path/to/output_dir")
```

### Exporting back to h5ad / h5mu (`to_h5ad`, `to_h5mu`)

Symmetric to `from_h5ad` / `from_h5mu`. Both default to `stream=True`
— peak RSS is bounded by one shard's worth of CSR per matrix
written, regardless of total file size. When the source SCX file
carries sharded obs/var (`ObsMetadataShard` / `VarMetadataShard`
sections — produced by `from_anndata` for `n_obs > shard_target_rows`,
or by `merge` / `append` on atlas-scale inputs), obs and var also
stream column-by-column through pre-allocated HDF5 datasets with
hyperslab writes per shard; categorical columns unify disjoint
per-shard vocabularies via a running global dictionary. Legacy
single-section obs/var sources transparently fall back to the eager
writer. Deletion vectors are respected on every column (obs row
count matches `/X[0]`); for high-cardinality categorical obs columns
the writer emits the modern `categorical` group form so
`anndata.read_h5ad` reads them cleanly at census scale.

`to_h5ad` also takes an optional obs-axis row filter, so a large file can be
exported as a subset without materialising it: `obs_mask=` (a boolean array) and
`min_counts=` (a per-cell total-UMI floor, computed with one streaming pass over
the CSR shards). Both are indexed in the **global / physical** obs row space —
length must equal `pyscx.open(path).n_obs_physical`, not `.n_obs`, which is the
post-deletion live count — and both are ANDed with the deletion-vector mask
rather than replacing it, so a logically deleted row stays dropped. A filtered
export records what it dropped in `uns["scx_export"]`. Both require
`stream=True`.

The motivating case is feeding a raw all-droplet file to CellBender
`remove-background`:

```python
# Result-preserving pre-trim: CellBender's own prior estimation ignores
# droplets at or below its --low-count-threshold, and never-analyzed barcodes
# are all-zero rows in its output.
pyscx.to_h5ad("raw.scx", "raw_trimmed.h5ad", min_counts=5)
```

Bring the corrected counts back with `pyscx.cellbender_import` /
`scx cellbender-import` — see
[docs/operations.md § CellBender import](operations.md#cellbender-import).

```python
import pyscx

# Single-modality export.
pyscx.to_h5ad("dataset.scx", "dataset.h5ad")

# Multimodal export (per-modality X + layers).
pyscx.to_h5mu("citeseq.scx", "citeseq.h5mu")

# Extract one modality from a multimodal SCX file as an h5ad.
pyscx.to_h5ad("citeseq.scx", "rna.h5ad", modality="rna")

# Opt out of streaming if you specifically want the legacy
# materialising path.
pyscx.to_h5ad("dataset.scx", "dataset.h5ad", stream=False)
```

## Landing external per-cell annotations (doublet detection)

Doublet callers are in-memory, single-sample tools that live in four different
ecosystems — scDblFinder and scds in R, Scrublet and DoubletDetection in
scanpy, Solo in scvi-tools. SCX does not reimplement any of them. It gives you
the plumbing to run them where they already live and land the results back on
the file: export per batch, run the tool, import by key.

Nothing in this path is doublet-specific except one lookup table of column
names. `pyscx.obs_import` lands any per-cell annotation table; `doublet_import`
is a thin wrapper that normalises each tool's spellings.

More generally: whenever the thing you computed is **per-cell columns on a
file that already exists** — a batch key for a downstream tool, a QC flag, a
cluster label — reach for the attach seam, not a `from_anndata` rewrite:
`obs_import` for a table on disk, `pyscx.attach_obs_columns(path, df)` for a
DataFrame already in memory (key-joined by default; `positional=True` for
columns you computed row-for-row from this file's own `read_obs()`, where the
obs index may not be unique). Either patches obs in place with no re-encode of
`X` (seconds, not minutes, at atlas scale), preserves the CSC sidecar /
`.raw` / deletion vectors / bitmaps / predicate indexes, and is
`scx rollback`-able. Pipeline runners can call both from restricted
(`no __import__`) `python` steps — see
[docs/api.md § Restricted-exec (sandbox) safety](api.md#restricted-exec-sandbox-safety).

### The per-tool column table

This is that lookup table. Read it before writing the import, not after it
surprises you: `--tool` selects which spellings the importer looks for, so a
table whose call column is named something else imports **score-only** and the
tool cannot vote on a call in `doublet_consensus`. (It warns when that happens,
names what it expected, and preserves your column under the `<K>_` prefix — so
the fix is `call_column=`, not re-running the caller.)

| `--tool` | score columns | score prefix | call columns | call prefix | doublet / singlet tokens | emits a call? |
|---|---|---|---|---|---|---|
| `scdblfinder` | `scDblFinder.score` | — | `scDblFinder.class` | — | `doublet` / `singlet` | yes |
| `scrublet` | `doublet_score` | — | `predicted_doublet` | — | `true` / `false` | yes |
| `doubletfinder` | — | `pANN_` | — | `DF.classifications_` | `Doublet` / `Singlet` | yes |
| `doubletdetection` | `doublet_score` | — | `doublet_label` | — | numeric `0`/`1` | yes |
| `solo` | `softmax_score`, `score` | — | `prediction` | — | `doublet` / `singlet` | yes |
| `scds` | `hybrid_score`, `cxds_score`, `bcds_score` | — | *(none)* | — | — | **no** |
| `generic` | *(caller-supplied)* | — | *(caller-supplied)* | — | *(caller-supplied)* | **no** — only via `call_column=` |

Score/call aliases are tried in the order listed, first match wins. A prefix
match must land on **exactly one** column — DoubletFinder run twice with
different `pK` leaves two `pANN_*` columns behind, and picking one silently
would be a coin flip. `generic` requires an explicit `score_column=`.

Rather than trusting this table to stay in sync with the code, read it from
Python: `pyscx.doublet_profiles()` returns the same rows straight from the
profile definitions, and a test asserts the two agree.

### Settle the join key first

This is the step worth doing before anything else, because everything
downstream joins on it. A tool sees only the file you hand it and returns rows
in whatever order it pleased, so the key is the only thing tying its answers
back to your cells.

```python
d = pyscx.diagnose_obs_key("atlas.scx")
print(d["summary"])          # what would be resolved, and whether it is unique
print(d["unique_columns"])   # usable keys that ARE unique, best first
print(d["unique_pairs"])     # two-column composites that are
print(d["unusable_unique_columns"])   # unique, but refused as a key
```

Every name it reports is one you can paste straight into `key=` — including
`obs_names` for the obs index. `unique_columns` is ordered best-candidate-first
(obs index, then `barcode`/`cell_id`-style names, then other strings, then
integers), and it lists only columns that can actually key a join. A unique
column the join would *refuse* is reported separately under
`unusable_unique_columns`: a float score column is often unique per row, but two
independently written sides are not guaranteed to format the same float
identically, so joining on one could silently half-match.

On a single library the obs index is usually unique and there is nothing to
think about. On a merged atlas it often is not: measured on a real
CELLxGENE-derived 1M-cell file, the obs index was a 10×-duplicated stringified
`RangeIndex`, no batch-column composite rescued it, and the only unique column
was `soma_joinid` — a name no fallback list would have guessed. Two ways out:

```python
# A column that is unique file-wide.
pyscx.obs_import("atlas.scx", "calls.csv", key=["soma_joinid"])

# Or a composite: barcodes repeat across libraries but are unique within one.
pyscx.obs_import("atlas.scx", "calls.csv", key=["sample_id", "barcode"])
```

A composite key needs both components on both sides, so the tool's output table
has to carry `sample_id` too. `export_batches` writes the batch column into
each per-batch h5ad for exactly this reason.

The two sides may *name* them differently. `source_key=` gives the source-side
column for each `key` component, pairing positionally like pandas
`left_on` / `right_on` — which is what a merged atlas usually needs, since its
identity is (`sample_id`, obs index) while the tool wrote a `barcode` column:

```python
pyscx.obs_import("atlas.scx", "calls.csv",
                 key=["sample_id", "obs_names"],
                 source_key=["sample_id", "barcode"])
```

`obs_names` is how you name the obs index anywhere a key is accepted — on the
target *and* on the source, so a tool table whose key is its own unnamed index
(what a plain `df.to_csv()` writes) needs no `source_key=`:

```python
pyscx.obs_import("atlas.scx", "scrublet.csv", key="obs_names")
```

It is also the spelling `diagnose_obs_key` reports, so whatever it suggests can
be pasted straight back. The underlying pyarrow field is called
`__index_level_0__`, but `read_obs()` hands it back as the frame's *unnamed*
index, so that name is not something you can address.

### Export one file per batch

```python
r = pyscx.export_batches("atlas.scx", "batches/", batch_key="donor_id")
r["key"]                    # the key that was resolved
r["key_is_globally_unique"] # decides how you import, below
for b in r["batches"]:
    b["path"], b["n_cells"]
```

The pooled matrix is never materialised — peak RSS is one library. The check
this adds over the loop you would write yourself is on the key: two cells
sharing a key *inside one batch* leave the tool's output with nothing to join
on, and that surfaces much later as a duplicate-key error at import time or,
worse, as scores landing on the wrong cell. Both the resolved key and
`obs_names` must be unique within a batch, since the tools read `obs_names`
while the import joins on the key. `on_ambiguous_key="error"` (the default)
refuses before writing anything; `"skip"` omits the batch and records why.

### Run the tool, then import

```python
# scanpy-side, per batch. Take the paths from the result rather than
# reconstructing them — the filenames are sanitised from the batch values.
import scanpy as sc

for b in r["batches"]:
    adata = sc.read_h5ad(b["path"])
    sc.pp.scrublet(adata)
    # The unnamed index column this writes is obs_names, and the import
    # resolves it as the join key.
    adata.obs[["doublet_score", "predicted_doublet"]].to_csv(
        f"calls/{b['batch']}.csv")
```

```r
# R-side — no intermediate file in either direction: read one batch straight
# out of SCX, run the tool, hand the data.frame back.
library(rscx)
library(scDblFinder)

res <- scx_open("atlas.scx") |>
  scx_query() |>
  filter_obs("donor_id == 'A'") |>   # a STRING expression, not NSE
  collect()
sce <- scDblFinder(res$to_sce())

# colData() carries the cell keys as ROWNAMES, not as a column.
df <- as.data.frame(colData(sce)[, c("scDblFinder.score", "scDblFinder.class")])

# One batch at a time would keep only the last, same as on the Python side.
# rbind() the per-batch data.frames and attach once.
scx_attach_obs("atlas.scx", df, key = rownames(df))
```

Then import. **`overwrite` replaces, it never merges**, so how you batch the
import depends on the key:

```python
import pandas as pd

if r["key_is_globally_unique"]:
    # Concatenate every batch's output and import ONCE.
    pd.concat([pd.read_csv(f"calls/{b['batch']}.csv") for b in r["batches"]]
              ).to_csv("all.csv", index=False)
    pyscx.doublet_import("atlas.scx", "all.csv", tool="scrublet")
else:
    # Keys only distinguish cells inside their own batch, so join on a
    # composite — which means the CSV above has to carry those columns too:
    #     adata.obs[["donor_id", "barcode",
    #                "doublet_score", "predicted_doublet"]].to_csv(...)
    # export_batches writes the batch column into each per-batch h5ad so they
    # are there to select.
    for b in r["batches"]:
        pyscx.doublet_import("atlas.scx", f"calls/{b['batch']}.csv",
                             tool="scrublet",
                             key=["donor_id", "barcode"], overwrite=True)
```

Importing several per-batch tables one after another *without* a composite key
keeps only the last — the second import replaces the first's columns rather
than filling in the rows it did not cover.

Cells the tool never saw come back `null`, never `0.0`. That distinction is
load-bearing: it is what lets the consensus step below tell "no tool assessed
this cell" apart from "every tool called it a singlet".

### If the tool wrote back into an h5ad

The scanpy-resident tools mutate `adata.obs` in place, so the h5ad itself is a
valid source — no CSV step:

```python
# One batch's h5ad, written back in place by the tool.
pyscx.doublet_import("atlas.scx", r["batches"][0]["path"], tool="scrublet",
                     keep_native_columns=False)
```

The same batching rule applies here as above: an h5ad holds one batch, so
importing each in turn without a composite key would keep only the last. With a
globally unique key, go through a concatenated CSV instead.

**Pass `keep_native_columns=False` on this route.** An h5ad exported from the
target file carries the *whole* original obs, and the default (`True`)
re-imports every one of those columns under the tool prefix — a real run wrote
32 obs columns (`scrublet_soma_joinid`, `scrublet_tissue`, …) where three were
wanted. Nothing is lost and nothing is wrong, but it is a lot of duplicated
metadata. The CSV route does not have this problem because you choose the
columns when you write the CSV. The h5ad source needs a pyscx built with the
`hdf5` feature; the CSV route does not.

### Combine several callers

Once N tools' results are canonical obs columns on one file, the consensus is
arithmetic:

```python
pyscx.doublet_import("atlas.scx", "scdbl.csv",   tool="scdblfinder")
pyscx.doublet_import("atlas.scx", "scrublet.csv", tool="scrublet")

cons = pyscx.doublet_consensus("atlas.scx", keys=["scdblfinder", "scrublet"])
cons["n_predicted_doublet"], cons["n_no_vote"]
```

This writes `obs["doublet_predicted"]` (pandas nullable `boolean`),
`obs["doublet_n_tools_calling"]` and `obs["doublet_n_tools_voting"]`. Read the
voting count before trusting a `False`: a `0` there means no tool assessed the
cell, which is why `doublet_predicted` is null beside it. `method="majority"`
(default) needs more than half of the *voting* tools; `"any"` / `"all"` are
also available, and `"mean_rank"` ignores the calls and combines the scores,
requiring an explicit `quantile` because a score cutoff is a scientific
decision the helper does not own.

Every one of these ops is in place and undoable — `pyscx.rollback("atlas.scx")`
reverts the last one. `X`, layers, `var`, the CSC sidecar, `.raw` and deletion
vectors are never touched. On a file target a **first** consensus writes
through `pyscx.attach_obs_columns(positional=True)` — a pure column add plus a
one-key uns merge in one commit — so the file's predicate index and the rest
of its obs and `uns` survive byte-identical. A re-run that overwrites existing
consensus columns (and any call passing `index_obs`/`index_preset`) takes the
whole-frame `modify_metadata` route instead, the seam that rebuilds a
predicate index over the rewritten values rather than dropping it. Full
behaviour and the predicate-index interaction:
[docs/operations.md § External obs import](operations.md#external-obs-import).

## Validating files after write or transfer

`pyscx.open(path)` and `pyscx.open(path, verify=True)` authenticate the file
header, catalog offsets/lengths, and the catalog checksum, but they do **not**
re-hash section payload bytes. That's the right default for normal reads —
the catalog already records per-section BLAKE3 checksums, so opening is fast
and detects file-header / catalog corruption immediately.

For full payload integrity, run `pyscx.validate(path)` (or `scx validate`
from the CLI), which BLAKE3-hashes every section's bytes against the
catalog and reports per-section pass/fail. Cost is proportional to total
section bytes.

Production checkpoints to call `pyscx.validate()`:

- After `pyscx.from_anndata(...)` / `pyscx.from_10x(...)` writes a new
  file, before downstream consumers depend on it.
- After `pyscx.pull(...)` (or `scx pull`) downloads from cloud, before
  treating the local file as canonical.
- After any file transfer (rsync, gcloud cp, scp, etc.) into a path that
  will be read by training or analysis pipelines.

```python
results = pyscx.validate("data.scx")
for name, passed in results:
    if not passed:
        raise RuntimeError(f"section {name} failed checksum")
```

### Deep validation (`deep=True`)

Checksum validation proves the section bytes match what was written, but not
that those bytes *decode* to a well-formed sparse matrix. For decode-level
integrity, pass `deep=True` (the equivalent of `scx validate --deep`):

```python
results = pyscx.validate("data.scx", deep=True)   # or exp.validate(deep=True)
for name, passed in results:
    if not passed:
        raise RuntimeError(f"{name} failed validation")
```

On top of the per-section checksums, deep mode:

- decodes every sparse shard and verifies the **v3 canonical CSR invariant**
  (column indices sorted and in range, no explicit zeros, `indptr` starting at
  0 and monotonically increasing, metadata consistent with the decoded data);
- verifies every framed shard's **row-group `BlockIndex`** — structural linkage
  to its source shard plus a decode-parity check that seeking to each row group's
  recorded block offset and decoding it reproduces the canonical decode
  byte-for-byte.

Deep-check results are appended to the returned list with `canonical-csr ` and
`block-index ` prefixed names. Canonical-CSR checks run only on v3+ files
(pre-v3 files may legitimately carry unsorted shards, so they are skipped).
Unlike a checksum failure on an essential section — which raises — deep-check
failures report `False` in the result list rather than raising, so iterate the
list to surface them. Cost is higher than a checksum-only pass because every
shard is decoded; reserve it for post-write or post-transfer integrity gates
where decode correctness matters.

## Understanding `to_anndata()`

### Full signature

```python
exp.to_anndata(
    backed=False,         # True for lazy loading (X stays on disk)
    cache_shards=4,       # LRU cache size for backed mode
    var_names=None,       # List of gene names to project (column subset)
    obs_filter=None,      # Predicate string to filter cells (e.g. "cell_type == 'T cell'")
    layers=None,          # None = load all layers; pass a list to select specific layers
                          # (e.g. ["raw_counts"]), or [] to skip loading layers entirely
    obsm=None,            # None = load all obsm keys (default); pass a list to load only
                          # those embeddings (e.g. ["X_pca"]), or [] to skip obsm. Selecting
                          # keys also switches obsm to a lazy / backed row-gather bridge
                          # (see "Selective + lazy obsm" below).
    eager=False,          # False (default): obsp/varp/varm and non-backed layers are
                          # wrapped in lazy bridges that decode each entry on first
                          # access. True: materialise everything up front so the
                          # AnnData is fully detached from the SCX file handle.
    memory_budget=None,   # None (default: 8 GiB), int (bytes), or a binary-prefixed
                          # size str: K/M/G/T or KiB/MiB/GiB/TiB ("4G" / "512MiB"); decimal KB/MB rejected.
                          # When the estimated eager assembly footprint exceeds this
                          # budget, a UserWarning is emitted recommending backed mode.
                          # Advisory only — assembly still proceeds.
)
```

### Default behavior (no extra params)

Calling `to_anndata()` performs a **full read** of the SCX file. Here is
exactly what happens:

1. **Reads and decompresses every CSR shard** from disk. Each shard is decoded
   using its per-shard codec (Scx1, Zstd, or raw), then all shards are
   assembled into a single contiguous CSR matrix.
2. **Applies deletion vectors.** If any cells have been logically deleted via
   `mark_deleted()`, those rows are excluded from both the matrix and the obs
   metadata. You always get a clean view — no manual filtering needed.
3. **Transfers the CSR matrix to scipy via zero-copy.** The Rust `Vec`s for
   indptr (`i64`), indices (`i32`), and data (`f32`) are moved directly into
   numpy arrays with no memory copy. These dtypes match what scipy expects, so
   `scipy.sparse.csr_matrix` wraps them without conversion.
4. **Reads obs/var metadata** as Arrow RecordBatches, converts to pandas
   DataFrames via `pyarrow.to_pandas()`.
5. **Reads obsm and uns eagerly**, and **wires lazy bridges for obsp,
   varp, varm, and (non-backed) layers** — each entry is decoded on
   first access rather than during `to_anndata()` itself.

The returned `anndata.AnnData` is fully populated:

| Slot | Source | Type |
|------|--------|------|
| `X` | CSR shards | `scipy.sparse.csr_matrix` (zero-copy) |
| `obs` | Obs metadata section | pandas DataFrame |
| `var` | Var metadata section | pandas DataFrame |
| `obsm` | Obsm sections | dict of numpy arrays (e.g. `X_pca`, `X_umap`); `ScxLazyObsmMapping` when `obsm=[...]` selected (see below) |
| `varm` | Varm sections | `ScxLazyVarmMapping` (lazy) / dict of numpy arrays (`eager=True`) |
| `obsp` | Obsp sections (COO Arrow IPC) | `ScxLazyPairwiseMapping` (lazy) / dict of `scipy.sparse.csr_matrix` (`eager=True`) |
| `varp` | Varp sections (COO Arrow IPC) | `ScxLazyPairwiseMapping` (lazy) / dict of `scipy.sparse.csr_matrix` (`eager=True`) |
| `uns` | Uns section | dict (tagged-JSON round-tripped; see below) |
| `layers` | Layer shards | `ScxLazyLayersMapping` (lazy, non-backed) / `ScxBackedLayerDataset` per entry (backed) / dict (`eager=True`) |

> Scanpy workflows that produce `obsp` / `varp` / `varm` (`sc.pp.neighbors`
> writes `obsp["distances"]` + `obsp["connectivities"]`; `pyscx.accel.pca`
> writes `varm["PCs"]`) survive `pyscx.from_anndata` → `to_anndata`
> verbatim. Sparse pairwise matrices are stored as float32 COO; higher
> precision is downcast on write. When cells are logically deleted via
> `mark_deleted` (or excluded by `obs_filter` in backed mode), `obsp` is
> subset to the kept rows and columns when the entry is decoded so the
> in-memory AnnData stays shape-consistent. The on-disk section keeps
> its original axis until `compact` rebuilds the file. `varp` and `varm`
> are unaffected by the deletion vector (var axis).

> **Lazy `obsp` / `varp` / `varm` / `layers`** (default `eager=False`):
> the four slots above are `MutableMapping`-compatible bridges
> (`ScxLazyPairwiseMapping`, `ScxLazyVarmMapping`,
> `ScxLazyLayersMapping`) that decode each entry from the SCX file
> only on first access — `ad.obsp["distances"]`,
> `for k, v in ad.varm.items():`, `dict(ad.layers)`, etc. — and cache
> the materialised value. Lookups via `__contains__` and key iteration
> stay catalog-only (no I/O). Mutations are in-memory and never write
> back to disk. The bridges keep a sibling `Arc<ScxReader>` alive so
> the returned AnnData stays usable after the source `Experiment`
> drops. Pass `eager=True` to substitute a plain `dict` and fully
> detach the AnnData from the SCX file handle — required when you
> intend to close the experiment, hand the AnnData to a subprocess,
> or otherwise outlive the underlying mmap. See
> [`docs/api.md` § `Experiment`](api.md#experiment) for the kwarg
> table.
>
> The lazy default bounds the peak RSS of `to_anndata()` itself for
> files that carry large kNN graphs (`obsp["distances"]` /
> `obsp["connectivities"]`) or embeddings (`varm["PCs"]`): the
> sections are not decoded until consumer code touches the slot. User
> code that does access them pays the same one-time decode cost it
> would have paid at construction time. Repeat accesses of the same
> key return the cached object.

#### Selective + lazy `obsm` (`obsm=[...]`)

`obsm` is eager by default (it tends to be small relative to
`obsp`/`varp`/`varm`), so `obsm=None` is byte-identical to prior
behaviour. Passing `obsm=[...]` opts into selective loading — only the
listed embeddings are read — and changes *how* obsm is materialised:

```python
# Selective eager: load just X_pca (and skip X_umap / X_state / …).
adata = exp.to_anndata(obsm=["X_pca"])

# Lazy (non-backed): X_pca is a ScxLazyObsmMapping — decoded to a dense
# numpy array on first `adata.obsm["X_pca"]` access, cached thereafter.
adata = exp.to_anndata(obsm=["X_pca"], eager=False)

# Backed dense row-gather: X_pca is a ScxBackedObsmDataset. m[idx] reads
# only the touched obsm shards (per-key LRU = cache_shards), so a single
# huge embedding (e.g. 10M cells × 2000-d) stays O(batch) per access.
adata = exp.to_anndata(backed=True, obsm=["X_pca"])
emb = adata.obsm["X_pca"]          # ScxBackedObsmDataset
batch = emb[cell_indices]          # dense (len(idx), n_cols) float array
```

| `obsm=` | `backed` | `eager` | obsm value type | When it reads |
|---|---|---|---|---|
| `None` | any | any | dict of dense numpy arrays | all keys, at `to_anndata()` |
| `[...]` | any | `True` | dict of dense numpy arrays | listed keys, at `to_anndata()` |
| `[...]` | `False` | `False` | `ScxLazyObsmMapping` → numpy | listed key, on first access |
| `[...]` | `True` | `False` | `ScxLazyObsmMapping` → `ScxBackedObsmDataset` | only touched rows, per `m[idx]` |

An unknown key raises `KeyError`; `obsm=[]` loads no embeddings.
Deletion vectors compose with the row gather via the same
`kept_to_global` remap as `X`. Under `obs_filter`, the backed-obsm path
falls back to selective eager obsm (composing a pandas-query row mask
with shard gather is deferred). This is the fix for per-worker obsm
memory blow-up on the random-access `embed_key`=`<obsm key>` dataloader
path — `obsm=[embed_key]` drops every unused embedding, and `backed=True`
keeps a single huge key off the per-worker heap.

> **`ScxBackedObsmDataset` is registered as `anndata.abc.CSRDataset`.**
> AnnData's `obsm` (`AxisArrays`) re-validates every value on each public
> `adata.obsm[key]` access and only accepts a fixed allowlist of array
> types; the lazy dense types it allows (`h5py.Dataset` / `zarr.Array` /
> `dask.array`) are concrete classes we can't subclass. Registering the
> backed dataset as a `CSRDataset` virtual subclass is what lets
> `adata.obsm[key]` return it (and `m[idx]` gather rows) rather than
> raising. The dataset is **dense** despite the `CSRDataset` label:
> `m[idx]` / `np.asarray(m)` / `m.toarray()` all return dense numpy. It
> does **not** implement CSR-only methods (`.tocsr()`), so code that
> introspects `adata.obsm[key]` as a sparse matrix will not work — treat
> it as a backed dense array (index it, or `np.asarray` it).

> **`uns` round-trip fidelity:** `from_anndata()` defaults to
> `uns_format="tagged"`, which preserves NumPy `dtype` and `shape`,
> bit-exact `float32` values, NaN/Inf inside arrays, structured
> recarrays (e.g. `uns["rank_genes_groups"]["names"]`), and
> `pd.Categorical` / `pd.Index` / `pd.Series` metadata (`name`, `codes`,
> `categories`, `ordered`). Legacy callers that depended on the previous
> behavior — where every NumPy array readback was a plain `list` — can
> opt back into it with `pyscx.from_anndata(adata, path,
> uns_format="plain")`. Both modes are read-compatible: the
> auto-detecting reader passes plain JSON through unchanged and decodes
> tagged envelopes back to their original Python types. See
> [`docs/api.md` § `uns` serialization](api.md#uns-serialization) for
> the on-disk envelope schema.

**Memory implications:** Once materialized, the in-memory AnnData is
**identical** whether the source was an `.scx` file or an `.h5ad` file —
the same scipy CSR matrix with the same dtypes (`i64` indptr, `i32` indices,
`f32` data). SCX's compression advantage applies only to the on-disk file
(e.g., an SCX file may be 200 MB where the equivalent h5ad is 1.5 GB), but
after `to_anndata()` both produce the same in-memory CSR.

The CSR memory formula:
```
memory ≈ (n_obs + 1) × 8 bytes           # indptr (i64)
       + nnz × 4 bytes                   # indices (i32)
       + nnz × 4 bytes                   # data (f32)

# Example: 1M cells × 30K genes × 5% density = 1.5B non-zeros
# ≈ 8 MB + 5.6 GB + 5.6 GB ≈ 11.2 GB
#
# At 2% density (more typical for 10x Chromium):
# nnz = 600M → ≈ 8 MB + 2.2 GB + 2.2 GB ≈ 4.5 GB
```

> [!WARNING]
> The formula above covers only the CSR arrays for X. The **total** memory
> footprint includes obs/var DataFrames, obsm embeddings, layers, and
> Python/h5py overhead — which can be substantial. For example, a 10M-cell ×
> 61K-gene dataset (176 GB h5ad) was OOM-killed during h5py streaming
> metadata reads with 80 GB of RAM available. As a rule of thumb, budget
> **2–3× the CSR size** for a comfortable working set, or use backed mode
> for datasets over ~500K cells.

`to_anndata()` performs a catalog-only estimate of the eager assembly
footprint before loading data. When the estimate exceeds `memory_budget`
(default 8 GiB), a `UserWarning` is emitted recommending `backed=True`
or `pyscx.open(path).query()`. The warning is advisory — assembly still
proceeds. Override the threshold with `memory_budget=`:

```python
exp.to_anndata(memory_budget="16G")   # raise the threshold
exp.to_anndata(backed=True)           # or use backed mode instead
```

If this exceeds your available memory, use
[selective loading](#selective-loading) or the
[query pipeline](#querying-subsets-before-loading) to load only what you
need. For fully out-of-core analysis, use
[backed mode](#backed-mode-lazy-loading).

#### Narrowing the output dtype / dense output

The memory formula above assumes `i32` indices and `f32` data. You can cut that
in half (or more) by requesting a narrower `data_dtype`, or skip the scipy CSR
entirely with `container="dense"` — useful when the next step wants a dense,
narrow array anyway (sklearn, a PyTorch `Tensor`, scVI):

```python
# Half the X footprint: float16 values (10x/count data is small-valued).
adata = exp.to_anndata(data_dtype="float16")

# uint8 counts (0–255): a quarter of the f32 footprint.
adata = exp.to_anndata(data_dtype="uint8")

# Dense row-major ndarray straight out (no CSR → dense re-densify later).
X = exp.to_anndata(container="dense", data_dtype="float32").X   # numpy.ndarray
```

The **default** (`container="csr"`, no dtype kwargs) is unchanged and stays
zero-copy — the `i64/i32/f32` Vecs are moved into numpy with no cast. Any
non-default request is a read-then-convert (an extra cast/copy of `X`).

Narrowing is **fail-loud** by default: a value that cannot be represented in the
requested dtype (out of range, fractional into an integer, negative into an
unsigned type, or a count above 2²⁴ into `float16`) raises `ValueError` naming
the offending value. Pass `allow_lossy=True` to narrow anyway. This also fixes a
prior silent `u32 → f32` rounding above 2²⁴ (e.g. pseudobulk / aggregated
counts). See [`docs/api.md` § Container and dtype materialization](api.md#container-and-dtype-materialization)
for the full reference. (Note: these kwargs apply to the eager and query paths;
`to_gpu_anndata` is f32-native and rejects them — narrow on the host first.)

### Selective loading

All selective loading parameters work in both non-backed and backed modes.

#### Gene projection (`var_names`)

Load only specific genes, reducing memory and computation:

```python
# Load only marker genes
adata = pyscx.open("atlas.scx").to_anndata(
    var_names=["CD3E", "CD4", "CD8A", "MS4A1", "NCAM1"]
)
print(adata.n_vars)  # 5
```

In non-backed mode, this applies column projection to the materialized CSR.
In backed mode, projection is applied lazily — full rows are decoded from
disk, but only the requested columns are retained in the returned CSR.

By default `var_names` is a **set selector**: the returned gene axis is in
sorted original-column order, and duplicate names collapse. Pass
`preserve_var_order=True` to return columns in the order you listed them
instead (duplicates still collapse, first occurrence wins) — useful when the
order carries meaning (e.g. a fixed signature panel):

```python
adata = pyscx.open("atlas.scx").to_anndata(
    var_names=["CD8A", "CD4", "CD3E"], preserve_var_order=True
)
list(adata.var_names)  # ['CD8A', 'CD4', 'CD3E']  (request order)
```

`preserve_var_order` works on the eager, backed, GPU, and query-engine
(`obs_filter` + `var_names`) paths. It is **not** supported by the streaming
accelerators on the resulting **backed** dataset: they decode columns in
sorted on-disk order, so a request-ordered gene axis would silently misalign
the result against `adata.var`. `highly_variable_genes`, `normalize_total`,
`log1p`, `calculate_qc_metrics`, `score_genes`, `pflog`, `pca`,
`pca_neighbors`, `pca_neighbors_umap`, `rank_genes_groups`, `pdex_ref`,
`pseudobulk_means`, `pseudobulk_dex` and `pdex_nb_glm` raise `RuntimeError`
rather than return misaligned output — run them before projecting by name, or
re-open without `preserve_var_order`.

Unknown names raise `KeyError` by default (`strict_var_names=True`). Pass
`strict_var_names=False` to silently drop names absent from the var metadata
(the pre-0.8.6 behaviour, which only errored when *every* name was unknown).

#### Cell filtering (`obs_filter`)

Filter cells using a predicate string. In non-backed mode, this leverages
the query engine with predicate pushdown (shard skipping). In backed mode,
it evaluates the predicate with pandas `.query()` on the (already
deletion-vector-filtered) obs DataFrame and folds the matches into the backed
dataset's row set — a different grammar (see [Filter Expression
Compatibility](#filter-expression-compatibility) below):

```python
# Load only T cells from lung tissue
adata = pyscx.open("atlas.scx").to_anndata(
    obs_filter="cell_type == 'T cell' and tissue == 'lung'"
)
```

##### Filter Expression Compatibility

`to_anndata()` evaluates `obs_filter` via one of three paths that fall into
**two grammars** — the `scx-engine` parser or pandas. They accept overlapping
but **not identical** expressions, so knowing which fires matters when a filter
string is reused across calls or pipelines (e.g. moving a filter from a
`query().filter_obs()` call to `to_anndata(backed=True, obs_filter=...)`).

| Path | Engine | Grammar | When it fires |
|---|---|---|---|
| SCX predicate engine | `scx-engine` predicate parser | engine | non-backed default (`preserve_slots=False`); `query().filter_obs(...)`; `pyscx.pull(...)` selective pulls |
| pandas `.query()` | `pandas.DataFrame.query` | pandas | `backed=True` with `obs_filter` set |
| pandas `.eval()` | `pandas.DataFrame.eval` | pandas | non-backed `preserve_slots=True` with `obs_filter` set |

`.query()` and `.eval()` share pandas's grammar, so the only split that matters
in practice is **engine vs pandas**: `backed=True` and `preserve_slots=True` both
accept the pandas-only forms below, while the default non-backed path and
`query().filter_obs()` use the stricter engine grammar.

**Portable subset (works in both paths):**

```python
"cell_type == 'T cell'"
"n_counts > 50"
"n_counts >= 50 and cell_type == 'T cell'"
"cell_type in ['T cell', 'B cell']"           # bracket-delimited list
"(n_counts > 50) or (cell_type == 'NK cell')"

# Parse in both, but select DIFFERENT rows when the column has nulls —
# see "Nulls: the one semantic divergence" below.
"not (cell_type == 'NK cell')"
"cell_type != 'NK cell'"
```

This subset uses comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`),
keyword-form boolean operators (`and`, `or`, `not`), the `in` operator
against a `[...]` list literal, and parenthesised sub-expressions. Tests in
`pyscx/tests/test_to_anndata_integration.py` assert that both paths select
identical rows for the entries above — `test_obs_filter_grammar_parity_common_ground`
on a fixture with **no** missing values, and
`test_obs_filter_grammar_parity_with_null_categorical` on one with nulls. The
null-bearing test deliberately omits `!=` and `not (...)`, which is the
divergence spelled out below.

**Divergences (work in one grammar only)** — the "pandas" column covers both
`backed=True` (`.query()`) and `preserve_slots=True` (`.eval()`):

| Expression | SCX engine | pandas (`.query()` / `.eval()`) |
|---|---|---|
| `n_counts > 50 & cell_type == 'T cell'` | ❌ parse error — use `and` | ✅ accepted as bitwise-and |
| `cell_type in ('T cell', 'B cell')` (tuple) | ❌ parse error — `in` requires `[...]` | ✅ accepted |
| `n_counts > 50 \| cell_type == 'NK cell'` | ❌ parse error — use `or` | ✅ accepted |
| Arithmetic on obs columns (e.g. `n_counts + n_genes > 100`) | ❌ not supported | ✅ accepted |
| String-method calls (e.g. `cell_type.str.startswith('T')`) | ❌ not supported | ✅ accepted |

**Nulls: the one *semantic* divergence.** Everything above is about which
expressions parse. This one is about what an expression that parses in both
grammars *means* when the column has missing values — an unannotated
`cell_type`, an obs column added by a join that didn't cover every cell.

The SCX engine uses three-valued (Kleene) logic, like SQL: a comparison
against a NULL cell is UNKNOWN, and only the final mask turns a surviving
UNKNOWN into "not matched". pandas is two-valued — `NaN == 'v'` is `False`
and `NaN != 'v'` is `True`. The two agree on `and`, `or` and `in`, and part
company on `!=` and `not`:

| For a row whose `cell_type` is NULL | SCX engine | pandas |
|---|---|---|
| `cell_type == 'B cell' or n_counts > 50` (and `n_counts` is 90) | ✅ matches | ✅ matches |
| `cell_type == 'B cell'` | ❌ | ❌ |
| `cell_type != 'B cell'` | ❌ UNKNOWN → not matched | ✅ matches |
| `not (cell_type == 'B cell')` | ❌ UNKNOWN → not matched | ✅ matches |

So a filter using `!=` or `not` on a null-bearing column selects a different
set of cells under `backed=True` than under the default path. Prefer the
positive form (`cell_type in [...]`) on columns that may have missing values,
or do the selection in Python where you can be explicit about `NaN`.

When `preserve_slots=True` is used with an `obs_filter`, `to_anndata()` emits
a `UserWarning` noting that the filter was evaluated via pandas.eval — this
surfaces in notebook output so the grammar shift is visible without reading
this section.

**Recommendation:** write filters in the portable subset above, and on a
column that may have missing values prefer the positive forms — `!=` and
`not (...)` parse everywhere but do not select the same rows. If a filter
truly needs pandas-only syntax, do the row selection in Python after
`to_anndata()` instead of inside `obs_filter` — that keeps the SCX call site
portable across `preserve_slots`, `backed=True`, and cloud selective pulls.

#### Layer selection (`layers`)

Load only specific layers instead of all:

```python
# Load raw counts layer only
adata = pyscx.open("atlas.scx").to_anndata(layers=["raw_counts"])

# Load no layers at all (X only)
adata = pyscx.open("atlas.scx").to_anndata(layers=[])
```

#### Combining parameters

All parameters can be combined:

```python
adata = pyscx.open("atlas.scx").to_anndata(
    backed=True,
    var_names=["CD3E", "CD4", "CD8A"],
    obs_filter="cell_type == 'T cell'",
    layers=["raw_counts"],
)
```

## Backed mode (lazy loading)

For atlas-scale datasets where full materialization is impractical, SCX
supports **backed mode** — data stays on disk and is loaded on demand, one
shard at a time:

```python
import pyscx
import scanpy as sc

# Open in backed mode — X stays on disk
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

print(type(adata.X))  # <class 'pyscx.ScxBackedSparseDataset'>
print(adata.X.shape)   # (1000000, 33694) — no data in memory yet

# Slicing loads only the needed shards
subset = adata.X[100:200]  # returns scipy.sparse.csr_matrix, ~1 shard decode
```

### How it works

When `backed=True`:

- `X` is an `ScxBackedSparseDataset` (not a materialized CSR matrix)
- **Layers** are wrapped in `ScxBackedLayerDataset` — also lazy
- **obs, var, uns** are loaded eagerly (same as non-backed — these are
  small relative to X)
- **obsm** is loaded eagerly by default, but `obsm=[...]` (without
  `eager=True` / `obs_filter`) makes each selected key a lazy
  `ScxBackedObsmDataset` row-gather dataset — see [Selective + lazy
  `obsm`](#selective--lazy-obsm-obsm) above
- The dataset is **read-only** (matching AnnData's `backed="r"` semantics)

Each access to `adata.X[rows, cols]` decompresses only the CSR shards that
overlap the requested rows. For a 1M-cell file with 64 shards, slicing 1K
cells touches ≤ 1 shard.

### Testing whether a matrix is lazy: `pyscx.is_backed_handle`

```python
if pyscx.is_backed_handle(adata.X):
    adata.X = adata.X.to_memory()   # scipy CSR
```

True for all four handle classes — `ScxBackedSparseDataset`,
`ScxBackedLayerDataset`, `ScxBackedObsmDataset`,
`ScxLazyTransformedDataset` — and False for anything already in memory.

**Do not reach for `scipy.sparse.issparse` here.** It returns `False` for a
handle and cannot be made to return `True`: scipy's `sparray` / `spmatrix` are
concrete classes rather than ABCs, so there is no `register()` seam. Two
consequences, neither of which raises:

```python
sp.issparse(adata.X)          # False  -> the usual guard takes the DENSE arm
np.asarray(adata.X).shape     # ()     -> a 0-d object array, not an error
```

The dense arm is the wrong arm, and the 0-d array surfaces as an unrelated
failure much later ("setting an array element with a sequence"). Slicing a
handle *does* yield genuine scipy sparse, so `sp.issparse(adata.X[0:10])` is
`True` — test the slice, or use this predicate on the matrix.

### Indexing patterns

| Access | Behavior |
|--------|----------|
| `X[100:200]` | Row slice → decodes 1 shard |
| `X[[0, 5, 10]]` | Fancy index → one decode per touched shard, result assembled once in request order (duplicates and negative indices allowed) |
| `X[mask]` | Boolean mask (length must equal `n_obs`) → one decode per touched shard, result assembled once; peak memory = result + the shard cache (`cache_shards` shards) + one shard's transient |
| `X[:]` | Whole matrix → exact-size result; without deletion vectors the shards decode uncached in parallel (costs what `to_memory()` costs), with deletion vectors it is the row gather over the kept rows |
| `X[100:200, :500]` | Row slice + column filter → 1 shard + post-filter |
| `X[:, hvg_idx]` | Column-only → must decode all shards (CSR is row-major) |
| `X[0, 5]` | Scalar → returns `float` |
| `X[[0, 10**9]]` | Out-of-range row → `IndexError` (never a shorter matrix) |

The same rows are reachable without an AnnData through
`Experiment.gather_rows_sparse(rows, layer=None, logical=True)` — see
[api.md](api.md#experiment). `adata.layers[name][rows]` gathers from a layer.

### scanpy operations in backed mode

| Operation | Works? | Notes |
|-----------|--------|-------|
| `sc.pp.filter_cells()` | ✅ | Native Rust row sums/NNZ |
| `sc.pp.filter_genes()` | ✅ | Native Rust column sums/NNZ |
| `sc.pp.calculate_qc_metrics()` | ✅ | `(X > 0).sum()` short-circuits to `getnnz()` |
| `sc.pp.calculate_qc_metrics(qc_vars=)` | ✅ | Column-projected streaming aggregation (non-materializing) |
| `sc.pp.highly_variable_genes()` | ✅ | Native Rust column variance |
| `adata[mask].copy()` | ✅ | Subset → materialize → preprocess |
| `sc.pp.pca()` | ✅ | Forces materialization of HVG columns |
| `pyscx.accel.normalize_total()` | ✅ | Lazy — no materialization |
| `pyscx.accel.log1p()` | ✅ | Lazy — no materialization |
| `sc.pp.normalize_total()` | ⚠️ | Use `pyscx.accel.normalize_total()` instead — see [Lazy preprocessing](#lazy-preprocessing-in-backed-mode) |
| `sc.pp.log1p()` | ⚠️ | Use `pyscx.accel.log1p()` instead — see [Lazy preprocessing](#lazy-preprocessing-in-backed-mode) |

> [!TIP]
> For `normalize_total` and `log1p` in backed mode, use `pyscx.accel.*`
> instead of `sc.pp.*`. The accelerator functions create lazy transform
> wrappers that keep data on disk — no materialization needed. See the
> [Lazy preprocessing](#lazy-preprocessing-in-backed-mode) section below.

**Traditional workflow** (materialize → preprocess):

```python
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# Subset to cells of interest, then materialize
adata_sub = adata[adata.obs["cell_type"] == "T cell"].copy()

# Now X is a regular scipy CSR — in-place ops work
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)
sc.pp.pca(adata_sub)
sc.pp.neighbors(adata_sub)
sc.tl.leiden(adata_sub)
```

**Zero-materialization workflow** (recommended for large datasets):

```python
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# Preprocessing stays lazy — data never leaves disk
pyscx.accel.normalize_total(adata, target_sum=1e4)  # → ScxLazyTransformedDataset
pyscx.accel.log1p(adata)                             # → appends transform

# PCA streams through the lazy transforms
pyscx.accel.pca(adata, n_comps=50)
pyscx.accel.neighbors(adata, n_neighbors=15)
pyscx.accel.umap(adata)
sc.tl.leiden(adata)
```

> **HVG masking without subsetting.** `pyscx.accel.pca(adata, mask_var=...)`
> restricts PCA to the selected genes in-place (scanpy semantics): pass a
> var-column name or a boolean array, or leave `mask_var=None` to auto-consume
> `adata.var["highly_variable"]` when present. It projects columns on the fly
> (backed/lazy/in-memory) — no HVG subset copy — and `varm["PCs"]` stays aligned
> to the full `var` axis (excluded genes filled with 0), so you never need the
> `adata[:, adata.var["highly_variable"]]` slice before PCA.
>
> **Behavior change (v0.11.6+):** matching scanpy, `mask_var=None` now
> **auto-consumes** `adata.var["highly_variable"]` when that column exists — so a
> PCA that previously ran on all genes will run on the HVG subset if you have
> flagged HVGs. `uns["pca"]["params"]["use_highly_variable"]` records whether a
> mask was applied. To force all genes, pass an all-`True` `mask_var`.

### Backed layers

Layers are also lazy in backed mode:

```python
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

print(type(adata.layers["raw"]))  # ScxBackedLayerDataset
raw_slice = adata.layers["raw"][0:1000]  # decodes only needed shards
```

### Deletion vector support

Backed mode transparently handles files with deleted rows. If cells have
been marked deleted via `mark_deleted()`, the backed dataset automatically:

- Excludes deleted rows from `shape`
- Remaps row indices so user-visible indices are contiguous
- Filters obs metadata to match

```python
exp = pyscx.open("experiment.scx")
exp.mark_deleted(doublet_mask)

# Backed mode sees only non-deleted cells
adata = exp.to_anndata(backed=True)
assert adata.X.shape[0] == exp.n_obs - doublet_mask.sum()
```

### Cache configuration

Decoded shards are optionally cached in an LRU cache to speed up repeated
access to the same rows:

```python
# Default: 4-shard LRU cache
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# Larger cache for repeated access patterns
adata = pyscx.open("atlas.scx").to_anndata(backed=True, cache_shards=16)

# No cache (minimum memory footprint)
adata = pyscx.open("atlas.scx").to_anndata(backed=True, cache_shards=0)
```

### Comparison with h5ad backed mode

| | h5ad `backed="r"` | SCX `backed=True` |
|--|---|---|
| File open | ~50 ms | ~5 ms (mmap + catalog parse) |
| Row slice 1K | ~20 ms | ~5 ms (1 shard decode) |
| Idle memory (1M cells) | ~8 MB | ~2 MB |
| Column slice (all rows) | ~2 s | ~800 ms (parallel shard decode) |
| `isinstance(X, CSRDataset)` | ✅ | ✅ (ABC registered) |

## Querying subsets before loading

For large datasets, you don't need to load everything into memory. The SCX
query engine filters at the shard level, skipping data that can't match:

```python
# Load only T cells from lung tissue
result = (pyscx.open("atlas.scx")
    .query()
    .filter_obs("cell_type == 'T cell' and tissue == 'lung'")
    .collect())

adata = result.to_anndata()
print(f"Loaded {adata.n_obs} cells, skipped {result.skipped_shards}/{result.total_shards} shards")

# Continue with scanpy as usual
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata)
```

### Query pipeline options

The query pipeline supports chaining multiple operations:

```python
result = (pyscx.open("atlas.scx")
    .query()
    .filter_obs("cell_type == 'B cell'")       # filter cells
    .filter_var("highly_variable == True")      # filter genes
    .select_genes([0, 1, 2, 100, 200])         # or select by index
    .with_normalize(1e4)                        # normalize in Rust (faster)
    .with_log1p()                               # log1p in Rust
    .limit(5000)                                # cap returned cells
    .collect())

adata = result.to_anndata()
```

When you use `with_normalize()` and `with_log1p()` in the query pipeline,
normalization runs in compiled Rust — significantly faster than the Python
equivalent on large datasets. The resulting AnnData is ready for downstream
analysis (PCA, clustering, etc.) without calling `sc.pp.normalize_total()`
or `sc.pp.log1p()` again.

## Out-of-core chunk iteration

For workflows that need to process data shard-by-shard without loading the
entire matrix, use `iter_chunks()`:

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# Iterate shard-aligned chunks (default)
for chunk in pyscx.iter_chunks(adata):
    print(f"Chunk: {chunk.n_obs} cells, {chunk.n_vars} genes")
    # Each chunk is a fully materialized AnnData with obs/var/obsm sliced
    sc.pp.normalize_total(chunk, target_sum=1e4)
    sc.pp.log1p(chunk)
    # ... accumulate results ...

# Fixed-size chunks
for chunk in pyscx.iter_chunks(adata, chunk_size=5000):
    # Each chunk has at most 5000 cells
    pass
```

Shard-aligned chunking (default) avoids decoding any shard twice. Each
chunk's `obs`, `var`, and `obsm` are correctly sliced to match the rows
in that chunk.

## Lazy preprocessing in backed mode

SCX supports **materialization-free preprocessing** for backed mode.
Instead of loading the entire expression matrix into RAM, `pyscx.accel.normalize_total()`
and `pyscx.accel.log1p()` create a **lazy transform wrapper** that applies
transformations on-read — data stays on disk.

### Lazy vs eager preprocessing

| `device=` | Behaviour |
|-----------|-----------|
| `"cpu"` / `"auto"` (no GPU) | **Lazy**: wraps `adata.X` in an `ScxLazyTransformedDataset` (see below). No materialization. |
| `"gpu"` / `"auto"` (GPU available) | **Eager**: streams shards through GPU kernels (`gpu_preprocess_to_csr`) and replaces `adata.X` with a materialized scipy CSR. A `UserWarning` is emitted when the prior X was already lazy so the broken chain is visible. |

**Fusion detection** — `normalize_total(device="gpu")` stashes a marker on
`adata.uns["__scx_gpu_pending_normalize__"]`. A subsequent
`log1p(device="gpu")` on the same AnnData consumes the marker and re-runs a
**single fused normalize+log1p pass over the original backed source** rather
than reading the already-materialized scipy CSR. Any intervening op (PCA,
kNN, …) silently forfeits the fusion, producing correct — but 1× redundant —
results.

**Scipy/dense fallback (symmetric for `normalize_total` and `log1p`)** — when
`device="gpu"` is passed but X is already a materialized scipy/dense matrix,
GPU dispatch on these per-row ops is dominated by H→D + D→H copies and
makes no sense. Both `pyscx.accel.normalize_total(device="gpu")` and
`pyscx.accel.log1p(device="gpu")` detect this, emit a `UserWarning` naming
the gating condition (X must be `ScxBackedSparseDataset` or
`ScxLazyTransformedDataset`), and delegate to `sc.pp.normalize_total` /
`sc.pp.log1p` on the host. To get the GPU fast path — including the
single-pass `normalize+log1p` fusion via the marker on
`adata.uns["__scx_gpu_pending_normalize__"]` — open via
`pyscx.open(...).to_anndata(backed=True)` (or keep the source as
`ScxLazyTransformedDataset`) and avoid `.copy()` between `normalize_total`
and `log1p` (that materialises X to scipy CSR and unreachably forfeits the
fast path for the rest of the pipeline).

**HVG input — backed, lazy, or materialized X.** `flavor` in `seurat_v3` /
`seurat_v3_paper` / `seurat` runs the scx-native streaming kernel whether
`adata.X` is an `ScxBackedSparseDataset`, an `ScxLazyTransformedDataset`, or a
plain materialized scipy/dense matrix (a materialized `X` is wrapped in a
single-shard `ShardSource`). So the common `pyscx.open(...).query()...collect()
.to_anndata()` (eager) idiom gets the same numerics — and the same per-batch
LOESS-singularity tolerance — as the backed path. Only flavors scx does not
implement natively (today `cell_ranger`) delegate to
`scanpy.pp.highly_variable_genes`, with a one-shot `UserWarning`.

> **`batch_key` cardinality is the usual LOESS-singularity trigger.** `seurat_v3`
> fits one `skmisc.loess` per batch; a high-cardinality key such as CELLxGENE
> `dataset_id` produces many tiny batches whose log-mean / log-variance
> regression is singular. The native path catches each such fit, warns naming
> the batch, and drops it from the ranking — so HVG completes. If many batches
> drop, prefer a coarser `batch_key` (or none). `filter_genes(min_cells=10)`
> only helps the single global fit, not the per-batch case.

**HVG on GPU** — `pyscx.accel.highly_variable_genes(device="gpu")` routes
through GPU atomicAdd kernels for `streaming_mean_var` and
`streaming_clip_square_sum`, including per-batch variants when `batch_key`
is set (materialized X uses the same single-shard `ShardSource`). GPU dispatch
is active for any `seurat_v3` configuration regardless of `batch_key`;
`flavor="seurat"` still falls back to CPU with a `UserWarning`. The per-batch
loess fits run on CPU via `skmisc.loess` — a batch whose log-mean / log-variance
regression is too degenerate to fit (small batch sizes, near-collinear inputs)
is caught, surfaced as a `UserWarning` naming the batch, and excluded from the
per-batch ranking; other batches proceed normally.


### Quick example

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# These do NOT materialize — they create/extend a lazy wrapper
pyscx.accel.normalize_total(adata, target_sum=1e4)
pyscx.accel.log1p(adata)

# adata.X is now an ScxLazyTransformedDataset (still on disk)
print(type(adata.X))  # <class 'pyscx.ScxLazyTransformedDataset'>

# Slicing decodes, transforms, and returns a scipy CSR
chunk = adata.X[0:1000]  # normalized + log1p'd scipy CSR (1000 × n_vars)

# PCA streams through the transforms shard-by-shard
pyscx.accel.pca(adata, n_comps=50)
```

### Memory savings

| Step | `sc.pp.*` (materializes) | `pyscx.accel.*` (lazy) |
|------|--------------------------|------------------------|
| After `normalize_total()` | ~8 GB (1M cells) | ~8 MB (cached row sums) |
| After `log1p()` | ~8 GB (in-place on CSR) | ~0 bytes (transform enum) |
| PCA peak | ~8 GB + working matrices | ~128 MB (1 shard) + working matrices |

### How it works: ScxLazyTransformedDataset

When you call `pyscx.accel.normalize_total(adata)` on backed data, it:

1. **Computes row sums** via streaming (no materialization — `BackedCsrReader::row_sums()`)
2. **Creates an `ScxLazyTransformedDataset`** wrapping the backed reader with a `NormalizeTotal` transform
3. **Replaces `adata.X`** with this lazy wrapper

When you subsequently call `pyscx.accel.log1p(adata)`, it:

1. **Detects** that `adata.X` is already an `ScxLazyTransformedDataset`
2. **Appends** a `Log1p` transform to the existing chain

When data is accessed (e.g., `adata.X[100:200]` or by `pyscx.accel.pca()`),
each shard is decoded → normalized → log1p'd on the fly. Peak memory is
one shard (~128 MB), not the full matrix.

### Transform types

| Transform | Created by | Effect |
|-----------|------------|--------|
| `NormalizeTotal` | `pyscx.accel.normalize_total()` | Divides each row by its precomputed sum, multiplies by `target_sum` |
| `Log1p` | `pyscx.accel.log1p()` | Element-wise `ln(x + 1)` on non-zero values |
| `RowScale` | `__truediv__` interception (experimental) | Multiplies each row by a per-row scalar |

When `NormalizeTotal` + `Log1p` are chained (the common case), they are
automatically **fused** into a single pass: `ln(x × target_sum / row_sum + 1)`,
avoiding an intermediate normalized array.

### Aggregation through transforms

The lazy wrapper supports streaming aggregation *after* transforms:

```python
# These stream shards, apply transforms, then aggregate — no materialization
adata.X.sum(axis=0)    # column sums of normalized+log1p'd data
adata.X.sum(axis=1)    # row sums of normalized+log1p'd data
adata.X.var(axis=0)    # column variance of transformed data
```

> [!NOTE]
> Aggregation through transforms is ~2× slower than raw aggregation
> (each shard must be decoded, transformed, then aggregated). However,
> peak memory remains O(shard_size) — ~128 MB vs ~8+ GB for full
> materialization.

### Materializing when needed

To get a full scipy CSR matrix (e.g., for a function that requires it):

```python
# Explicit materialization
csr = adata.X.to_memory()  # scipy CSR with all transforms applied
# or
csr = adata.X.copy()       # same as to_memory()
```

### Complete out-of-core pipeline

This pipeline processes 1M+ cells with ~11 GB peak RSS (vs ~22 GB materialised — 51% reduction). The lazy preprocessing path (normalize + log1p + streaming PCA) peaks at ~3.5 GB; the remaining RSS is dominated by kNN graph construction and UMAP:

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# QC — fully streaming, including gene subsets (column-projected streaming aggregation)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# HVG — streaming variance
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Preprocessing — lazy, no materialization
pyscx.accel.normalize_total(adata, target_sum=1e4)
pyscx.accel.log1p(adata)

# Dimensionality reduction — streams through lazy transforms
pyscx.accel.pca(adata, n_comps=50)
pyscx.accel.neighbors(adata, n_neighbors=15)
pyscx.accel.umap(adata)
sc.tl.leiden(adata)

sc.pl.umap(adata, color="leiden")
```

### `__truediv__` interception (experimental)

As an optional convenience, `ScxBackedSparseDataset.__truediv__()` detects
when it receives a per-row scaling vector (as produced by scanpy's
`normalize_total` internals) and returns a lazy `ScxLazyTransformedDataset`
instead of materializing. This means `sc.pp.normalize_total(adata)` may
avoid materialization in some cases.

> [!WARNING]
> **This interception is experimental.** Modern scanpy (≥1.10) uses a
> numba-jitted `_normalize_csr()` function that accesses `.data`/`.indptr`
> attributes directly for CSR matrices. Since `ScxBackedSparseDataset`
> doesn't expose raw CSR attributes, scanpy falls back to the non-CSR path
> where `__truediv__` interception *does* work — but this behavior depends
> on scanpy's internal code paths and may change between versions.
>
> **Recommendation:** Always use `pyscx.accel.normalize_total()` and
> `pyscx.accel.log1p()` for reliable, version-independent lazy preprocessing.

## Preprocessing pipeline (write-back)

For in-place modifications that need to persist, use the streaming
preprocessing pipeline. This reads shards one at a time, applies fused
operations in Rust, and writes the result to a new SCX file:

```python
import pyscx

# Normalize + log1p → new file (no full materialization)
pyscx.preprocess(
    "raw.scx",
    "preprocessed.scx",
    ops=["normalize_total", "log1p"],
    target_sum=1e4,
)

# Save as a named layer in an existing file
pyscx.save_layer(
    "experiment.scx",
    "experiment_with_norm.scx",
    layer_name="normalized",
    ops=["normalize_total", "log1p"],
    target_sum=1e4,
)

# Then use the preprocessed file
adata = pyscx.open("preprocessed.scx").to_anndata()
sc.pp.pca(adata)  # Already normalized — skip normalize/log1p
```

The preprocessing pipeline uses fused `normalize_total + log1p` in a
single pass over each shard, which is faster than the equivalent scanpy
calls on large datasets.

> [!TIP]
> **Choosing between lazy preprocessing and write-back:**
> - **Lazy** (`pyscx.accel.*`): Best for interactive analysis — no disk I/O,
>   instant, transforms applied on-read. Original data preserved.
> - **Write-back** (`pyscx.preprocess()`): Best when you need a persistent
>   preprocessed file (e.g., for repeated analysis or sharing). Writes a new
>   SCX file with transforms baked in.

## Rust-native accelerators

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
do](gpu-setup.md#what-deviceauto-does-and-does-not-do).

> **Seeing the route at op start / diagnosing a slow GPU op.** Each accelerator
> logs its resolved route the moment it starts, at INFO — enable it with
> `import logging; logging.basicConfig(level=logging.INFO)` (or
> `logging.getLogger("pyscx.accel").setLevel(logging.INFO)`). On **backed /
> atlas-scale** input the streaming GPU path (`highly_variable_genes`,
> streaming PCA) is **CPU-decode-bound** — the GPU can sit near 0% util while
> shards decode; this is expected, not a hang. An explicit `device="gpu"` request
> that silently lands on CPU emits a `UserWarning` naming the `fallback_reason`.
> See the "slow GPU op / is it hung?" entry in [docs/gpu-setup.md § Troubleshooting](gpu-setup.md#troubleshooting).

> **GPU is fastest only when the input layout matches the op.** For
> `pdex_ref` the column-major CSC-direct GPU route is the high-performance
> path, and it requires a *backed* SCX file with a CSC sidecar
> (`pyscx.from_anndata(..., csc="always")` / `scx convert --csc=always`); v3
> CSC-direct is the default GPU DE route. In-memory scipy CSR can run on GPU but
> may be slower than CPU. (`rank_genes_groups` / Wilcoxon rank-sum shares the same
> CSC-direct (`gpu_csc_v3`) and CSR-direct (`gpu_csr_v3`) routes as `pdex_ref`;
> dense-host input is densified to CSR and also records `gpu_csr_v3`.) When comparing performance,
> **check the recorded route** at
> `adata.uns["scx_accel"][<op>]["route"]` (e.g. `gpu_csc_v3` vs `gpu_csr_v3`) —
> every DE call records which route it actually took and the `fallback_reason`
> if it didn't take the ideal one. See
> [docs/api.md § Accelerator route metadata](api.md#accelerator-route-metadata).

### Axis-subsetting ops and the aligned members

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
[docs/api.md § Axis subsetting and aligned members](api.md#axis-subsetting-and-aligned-members).

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

### Compatibility matrix

| Op                       | CPU | GPU | Scanpy-parity kwargs                                                  | scx-only kwargs                                |
|--------------------------|:---:|:---:|-----------------------------------------------------------------------|------------------------------------------------|
| `normalize_total`        | ✓   | ✓   | `target_sum`                                                          | `device`                                       |
| `log1p`                  | ✓   | ✓   | —                                                                     | `device`                                       |
| `filter_cells`           | ✓   | —   | `min_genes`, `max_genes`, `min_counts`, `max_counts`                  | —                                              |
| `filter_genes`           | ✓   | —   | `min_cells`, `max_cells`, `min_counts`, `max_counts`                  | —                                              |
| `subset_obs`             | ✓   | —   | — (`adata[mask].copy()`, without materializing)                       | `mask_or_indices`                              |
| `subset_var`             | ✓   | —   | — (`adata[:, mask].copy()`, without materializing)                    | `mask_or_indices`                              |
| `calculate_qc_metrics`   | ✓   | —   | `qc_vars`, `log1p`, `inplace`                                         | `prefer_format`                                |
| `highly_variable_genes`  | ✓   | ✓   | `n_top_genes`, `flavor`, `batch_key`, `span`, `subset`, `n_bins`, `layer` | `device`, `prefer_format`                  |
| `score_genes`            | ✓   | —   | `gene_list`, `ctrl_size`, `gene_pool`, `n_bins`, `score_name`, `random_state` | `method`, `layer`, `device`           |
| `pflog`                 | ✓   | —   | — (no scanpy equivalent)                                             | `alpha`, `store`, `n_components`, `store_repr`, `out`, `shard_size`, `layer`, `device` |
| `pca`                    | ✓   | ✓   | `n_comps`, `zero_center`, `random_state`                              | `device`, `method`, `qr_method`, `prefer_format`, `allow_tf32`, `n_oversamples`, `n_power_iterations`, `spmm_policy`, `memory_budget` |
| `neighbors`              | ✓   | ✓   | `n_neighbors`, `use_rep`, `random_state`                              | `device`, `ef_construction`, `ef_search`       |
| `pca_neighbors`          | ✓   | ✓   | (PCA + neighbors kwargs, see below)                                   | `device`, `method`, `qr_method`, `prefer_format` |
| `pca_neighbors_umap`     | ✓   | ✓   | (PCA + neighbors + UMAP kwargs, see below)                            | `device`, `method`, `qr_method`, `prefer_format` |
| `umap`                   | ✓   | ✓   | `n_components`, `n_epochs`, `min_dist`, `spread`, `learning_rate`, `random_state` | `device`, `negative_sample_rate`   |
| `leiden`                 | ✓   | ✓¹  | `resolution`, `key_added`, `random_state`, `n_iterations`             | `device`, `parallel`, `theta`                  |
| `harmony_integrate`      | ✓   | —   | `key`, `basis`, `theta`, `sigma`, `lamb`, `max_iter`                  | `adjusted_basis`, `block_size`, `n_clusters`, `alpha`, `max_iter_kmeans`, `random_state`, `device` |
| `compute_lisi`           | ✓   | —   | `key`, `basis`, `perplexity`, `n_neighbors`, `approximate_knn`        | —                                              |
| `rank_genes_groups`      | ✓   | ✓   | `groupby`, `reference`, `n_genes`, `method`                           | `gene_chunk_size`, `stratify_by`, `prefer_format`, `tie_correct`, `rankby_abs`, `device` |
| `pdex_ref`               | ✓   | ✓   | `groupby`, `reference`                                                | `is_log1p`, `geometric_mean`, `epsilon`, `cpm_filter`, `gene_chunk_size`, `prefer_format`, `device`, `output` |
| `pseudobulk_dex`         | ✓   | —   | `design`, `reference` (**`groupby` is spelled the same but means the opposite** — sample-defining columns, not the compared one) | `test_col`, `sample_cols`/`sample_key` (aliases for `groupby`), `aggr_method`, `stratify_by`, `prefer_format`, `backend`, `nbglm_options`, `gene_indices`, `n_cpus` |
| `nb_glm`                 | ✓   | —   | — (no scanpy equivalent)                                             | `counts`, `design`, `contrast`                 |
| `pdex_nb_glm`            | ✓   | —   | — (no scanpy equivalent)                                             | `groupby`, `reference`, `stratify_by`          |
| `pseudobulk_means`       | ✓   | ✓   | — (no scanpy equivalent)                                             | `groupby`, `min_cells_per_group`, `device`     |
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

### `prefer_format="auto"|"csr"|"csc"`: column-major dispatch

A subset of accelerators take a `prefer_format` kwarg that selects
between the row-major CSR path and the column-major CSC sidecar path.
Entries that accept it:

| Function | CSC win |
|----------|---------|
| `pyscx.accel.highly_variable_genes` | Single-batch seurat_v3 only — single-pass per-column accumulators with no `O(n_vars)` row-wise scratch. Multi-batch and non-seurat_v3 raise. |
| `pyscx.accel.rank_genes_groups` | Per gene chunk: read CSC slab + scatter into row-major dense buffer (vs decode every row + project for CSR). Clearest CSC win. |
| `pyscx.accel.pseudobulk_dex` | Filtered-gene subsets only (`gene_indices=...` or column projection on `adata.X`). Full-gene pseudobulk has no CSC win and raises. |
| `pyscx.accel.calculate_qc_metrics` | Gene-axis aggregations only (`total_counts`, `n_cells_by_counts`); cell-axis stays CSR. Both axes take one shard pass each, whatever the `qc_vars` count. |
| `pyscx.accel.col_sums` / `col_nnz` / `col_min` / `col_max` / `col_var` | Per-column aggregations on `ScxBackedSparseDataset` / `ScxLazyTransformedDataset`. |
| `pyscx.accel.pca` | **Rejects `prefer_format="csc"`** with `ValueError`. Covariance build and randomized SpMM are row-major; CSC offers no measurable speed-up. |

**DE (`rank_genes_groups`, `pdex_ref`) defaults to `"auto"`; every
other `prefer_format`-taking function defaults to `"csr"`.**
`"auto"` (a **compatibility change** in the CPU-accelerator Phase-2
work — DE previously defaulted to `"csr"`) resolves at call time
against the *selected* matrix: on CPU it takes the CSC-direct route
when a valid sidecar is available (sidecar present ∧ no active row
deletion vector ∧ column-local transform chain — the same capability
gate `"csc"` enforces) and CSR otherwise; on GPU it stays CSR so the
planner routes `gpu_csc_v3` when a sidecar is present. The route and
`csc_available` flag are recorded on `adata.uns["scx_accel"][<op>]`
(`cpu_csc` vs `cpu_csr`). Pass `prefer_format="csr"` explicitly to pin
the pre-change behaviour. The non-DE functions keep `"csr"` — the
runtime does not yet auto-route them. No thread-local default; no
env-var override; each call sets the choice locally.

`prefer_format="csc"` requires *all* of the following; otherwise it
raises `RuntimeError` with a message naming the missing capability:

1. The file has a CSC sidecar (`pyscx.from_anndata(csc="always"|"auto")`,
   `scx convert --csc=always|auto`, `scx build-csc`, or the standalone
   `pyscx.build_csc(path)` to add one to an existing file in place, or pass an `output` to write a copy).
2. The transform chain on `adata.X` contains only column-local
   operations. `Log1p` is column-local; `NormalizeTotal` and
   `RowScale` are not (per-row state). The common `normalize_total →
   log1p` chain is *not* column-local — use `prefer_format="csr"`.
3. No active row deletion vector. After
   `pyscx.accel.filter_cells()` or `pyscx.accel.subset_obs()`, the
   dataset has `kept_to_global` set; CSC dispatch then raises until
   you `materialize()` or rebuild the file.
4. No active column projection. After `pyscx.accel.filter_genes()`,
   `pyscx.accel.subset_var()`, `highly_variable_genes(subset=True)` or
   `adata[:, mask]`, the sidecar (written against the *full* gene axis)
   no longer describes the visible one, so CSC dispatch raises. The CSR
   path streams the projected window and works normally.

Both subset cases are refusals, not silent fallbacks — the CSR default
handles them, so reach for `prefer_format="csc"` before you subset, not
after.

Unknown values (e.g. `"CSC"`, `"bogus"`) raise `ValueError`. `"auto"`
is accepted by `rank_genes_groups` / `pdex_ref` (and is their default);
the other `prefer_format`-taking functions accept only `"csr"` / `"csc"`.

```python
import pyscx

# Open a CSC-equipped file
exp = pyscx.open("atlas.scx")  # written via pyscx.from_anndata(csc="always")
adata = exp.to_anndata(backed=True)

# DE on a small target gene set — CSC slab read avoids decoding every row
pyscx.accel.rank_genes_groups(
    adata, "perturbation", reference="control",
    prefer_format="csc",
)

# log1p preserves CSC capability (column-local)
pyscx.accel.log1p(adata)
pyscx.accel.col_sums(adata.X, prefer_format="csc")  # works

# normalize_total breaks it (row-local)
pyscx.accel.normalize_total(adata, target_sum=1e4)
pyscx.accel.col_sums(adata.X, prefer_format="csc")  # raises RuntimeError
```

For the on-disk format and sharding granularity, see
[docs/sharding.md § CSC sharding](sharding.md#csc-sharding) and
[docs/format.md § 4.1 CSC Shard Internal Layout](format.md#41-csc-shard-internal-layout).

### GPU-supported vs GPU-fast

`device="gpu"` runs a column algorithm on the GPU, but **running on the GPU
is not the same as running fast on the GPU**. For differential expression,
peak GPU throughput requires a *column-major* substrate so the kernel reads
contiguous gene columns instead of decoding and projecting every row.

- **`pdex_ref` is GPU-fast only with a CSC sidecar.** With a backed SCX file
  that has a CSC sidecar, the dispatch takes the
  CSC-direct route (`route == "gpu_csc_v3"`): it drops the per-chunk dense
  intermediate and skips non-overlapping CSC shards via a column-range
  pre-filter. Without a sidecar — e.g. an in-memory scipy CSR — the same call
  falls back to `gpu_csr_v3` (`fallback_reason == "no_csc_sidecar"`), which is
  *GPU-supported but not GPU-fast*.
- **Wilcoxon rank-sum (`rank_genes_groups`) takes the same v3 routes as `pdex_ref`.** With a
  CSC sidecar it runs CSC-direct (`gpu_csc_v3`); without one it runs CSR-direct
  (`gpu_csr_v3`, `fallback_reason == "no_csc_sidecar"`) — GPU-supported but not
  GPU-fast. So a CSC sidecar makes Wilcoxon rank-sum GPU-fast too.
- **PCA / kNN / UMAP / Leiden are not column algorithms** — they operate on
  row-major `X` or on PCA embeddings / kNN graphs, so CSC does not apply.
- **CSC-direct does not double VRAM usage.** The `gpu_csc_v3` route reads CSC
  shards *instead of* CSR shards — it does not load both representations
  simultaneously. VRAM usage for the CSC-direct path is comparable to the CSR
  path (proportional to NNZ), plus the per-chunk dense intermediate replaced
  by the shared-memory tree-reduce.

> **Which `(device, prefer_format)` selects `gpu_csc_v3`?** `device` and
> `prefer_format` are independent axes, and the GPU CSC-direct route is chosen
> by the *route planner*, **not** by `prefer_format="csc"`:
>
> - **GPU-fast DE:** pass `device="gpu"` (or `"auto"`) with `prefer_format`
>   left at its `"auto"` default (or set to `"csr"`) — both keep GPU on the
>   planner-driven path. When the backed file has a CSC sidecar the planner
>   routes to `gpu_csc_v3` automatically; without one it uses `gpu_csr_v3`. This
>   is the intended GPU-fast entry point.
> - `prefer_format="csc"` selects the **CPU** column-major streaming path
>   (`cpu_csc`) — there is no GPU kernel behind that knob. With `device="auto"`
>   it runs on CPU; combining it with an explicit `device="gpu"` raises a
>   `RuntimeError` that points you back to `prefer_format="csr"`/`"auto"` +
>   `device="gpu"` for GPU CSC-direct.
>
> In short: do **not** reach for `prefer_format="csc"` to get GPU speed — it is
> the CPU path. A CSC *sidecar on the file* (built at conversion) is what makes
> the default-`csr` GPU call fast.

> **The CSC sidecar lives on disk — only a *backed* AnnData carries it.** The
> sidecar is reachable for GPU DE only through a backed dataset
> (`exp.to_anndata(backed=True)`, whose `X` is a `ScxBackedSparseDataset`). If you
> **materialize** with the plain `exp.to_anndata()`, `X` becomes an in-memory
> scipy CSR with no link back to the file, so GPU DE silently takes the slower
> `gpu_csr_v3` route (`fallback_reason == "no_csc_sidecar"`) **even though
> `exp.has_csc` is `True`**. Reach for `to_anndata(backed=True)` whenever you want
> the `gpu_csc_v3` fast route. As a guard, pyscx emits a one-time `UserWarning`
> when `device="gpu"` DE runs on an `X` that was materialized from a file with a
> CSC sidecar (it stamps `adata.uns["scx_source_has_csc_sidecar"] = True` at
> materialization to detect exactly this case).

To make a file GPU-fast for DE, build the sidecar at conversion time:
`pyscx.from_anndata(adata, path, csc="auto")` (built automatically once the
dataset clears the size thresholds) or `csc="always"`, `scx convert --csc=auto`,
or `scx build-csc` after the fact. Then **always confirm the route actually
taken** via `adata.uns["scx_accel"][op]["route"]` before drawing performance
conclusions — a silent CSR fallback measured as "GPU DE" is exactly the
benchmarking trap the [route metadata](api.md#accelerator-route-metadata) exists
to catch.

**Malformed input is rejected, not ranked.** Every GPU route validates each
shard on the host before it is staged, and raises rather than producing
numbers. Two invariants the kernels cannot enforce themselves:

- **Finite values.** The per-gene sort pads with `+INF` and sorts on the raw
  IEEE-754 bit pattern, so a NaN would land above `+INF` and corrupt the U
  statistic, the tie counts and every p-value in the gene — silently. Filter or
  QC NaN / Inf before DE; the CPU paths reject the same input.

  This check is scoped to the operations whose kernels need it, and the scoping
  is decided per entry point rather than per op:

  | GPU entry point | non-finite input |
  |---|---|
  | `rank_genes_groups`, `pdex_ref` | **rejected** at the staging boundary; the error names the op |
  | `highly_variable_genes` — the clipped-sum reducers, and the batched mean/variance | **rejected** at the staging boundary |
  | `highly_variable_genes` — plain and CSC mean/variance | rejected *after* accumulation, naming the offending gene column |
  | `pca`, `normalize_total` / `log1p`, `pseudobulk` | accepted; propagates as it would on the CPU |

  The split inside `highly_variable_genes` is not arbitrary. Plain mean/variance
  can afford to skip the O(nnz) input scan because a non-finite value survives
  into its column's sums, where `first_non_finite_column` still catches it. The
  clipped reducers cannot: the clip kernel evaluates `v > clip ? clip : v`, so
  `+Inf` compares true and is **replaced by the clip value** before it is ever
  accumulated — the sums come out finite and plausible, and no post-hoc check on
  the output can tell. The batched mean/variance is excluded for a different
  reason: its kernel returns early on a row belonging to no batch, so a
  non-finite value in an excluded row never reaches the sums either.

  Note this differs from the CPU HVG path, which rejects **every** non-finite
  input up front via `ensure_finite_values`, including for the plain
  mean/variance reducers where the GPU defers to the post-accumulation check.
- **One value per `(cell, gene)`.** The scatter runs one thread per nonzero, so
  a duplicated entry would put two threads on one output cell with a
  nondeterministic winner. On the CSC side this is checked as *strictly
  increasing* row indices per column, which is what `scx build-csc` and every
  other sidecar writer emits; a hand-built sidecar with distinct-but-unordered
  rows is refused conservatively rather than raced.

  This check is likewise scoped, and **independently** of the finiteness check
  above — the two are separate switches, not a strictness ladder, precisely
  because they do not nest:

  | GPU entry point | sorted indices required |
  |---|---|
  | `rank_genes_groups`, `pdex_ref` | yes — the scatter races a duplicate `(cell, gene)` |
  | `pca` | yes — cuSPARSE SpMM is undefined on unsorted column indices |
  | `pseudobulk` | yes — the kernel binary-searches each row's column window |
  | `highly_variable_genes` (all entry points) | **no** — it accumulates atomically or clips in place |
  | `normalize_total` / `log1p` | **no** — it rewrites values in place |

  So `highly_variable_genes` accepts an unsorted `scipy.sparse.csr_matrix`
  (`has_sorted_indices == False`) exactly as the CPU path does, while still
  rejecting a non-finite value on the entry points listed in the previous
  table. An earlier design made these a cumulative ladder, which meant asking
  for the finiteness check silently also demanded sorted indices — and GPU HVG
  then refused input its own kernels handle fine.

Both surface as a `RuntimeError` naming the offending gene column or nonzero
index. The CSC-direct route previously ran neither check, so a NaN in a file
with a CSC sidecar returned a complete `rank_genes_groups` / `pdex_ref` result
on `device="gpu"` — finite, plausible scores and p-values, no error — while the
same file on CPU, or on GPU without a sidecar, was rejected.

**Benchmark route gates.** The benchmark suite enforces correct GPU dispatch
via absolute-floor gates in `thresholds.yaml`. Every `accel_*.py` GPU variant
emits an `<op>_route_gpu_correct` signal (1.0 when a GPU route ran, 0.0 on a
silent CPU fallback); `bench_csc_dispatch.py` emits `csc_dispatch_correct` for
CSC-labelled variants; `accel_de.py` emits `de_route_csc_direct` for the
pdex_ref CSC-direct path; and `accel_eval_metrics.py` emits
`perturbation_metrics_route_gpu_correct` / `energy_distance_route_gpu_correct`
for the perturbation-evaluation metric GPU variants. See
[benchmarks/README.md § Regression Gating](../benchmarks/README.md#regression-gating)
for the full gate table.

### Data layout for fast GPU decode (`to_gpu_anndata` / device-resident analysis)

The device-handoff path — `pyscx.open(...).to_gpu_anndata()`, then chained
`pyscx.accel.*` / `rsc.*` ops on the device-resident AnnData
(`transfer_mode` ∈ `scx_device_decode_gpu` / `scx_device_handoff_streamed` /
`scx_device_handoff`) — is only as fast as the cost of
getting each shard onto the GPU. That cost is dominated by **decode**, and
decode cost is set by the **value representation you persisted on disk**, not by
the analysis op. So the layout choice matters as much as the device flag:

- **Prefer storing raw integer counts (`X` → Scx1) and deriving log-norm
  on-device.** Raw scRNA-seq counts auto-route to the Scx1 codec
  (Delta-Golomb / FOR-BP / Rice) — the codec with GPU decode kernels, so it is
  the codec the device-side decode path targets; framed Scx1 shards decode
  group-by-group in VRAM (random access via the row-group block index). Open the
  counts and run `normalize_total` / `log1p` in VRAM (the
  `ScxLazyTransformedDataset` chain, or `rsc.pp.*` on the device AnnData) so the
  log-normalized matrix is produced on the GPU and **never round-trips through a
  host float buffer**. This is the recommended flow.
- **A *persisted* log-normalized `X` is float → Pcodec, which decodes on the
  host.** Public h5ad / CELLxGENE files often ship `X` already log-normalized.
  That matrix has **no GPU decoder** (Pcodec is CPU-only),
  so the handoff pays a host pcodec-decompress + HtoD per shard — *GPU-supported,
  but not GPU-fast on the decode side*. If the file also carries a `counts`
  layer, prefer opening that and deriving log-norm on-device (bullet above).
- **If you must persist log-norm for the GPU and want decode-free upload, store
  it uncompressed (`None` codec → raw `f32` / `f16`).** A raw float array needs
  no decode — the handoff is a straight HtoD `memcpy` — at the cost of
  compression ratio (`f16` halves the bytes if its precision is acceptable for
  log-norm). This is the only float option that is GPU-fast to upload today; a
  GPU-decodable *compressed* float codec does not exist.
- **Device decode is Scx1-only.** Framed Scx1 shards decode group-by-group
  directly in VRAM; Zstd / Pcodec / LZ4 shards and float layers have no GPU
  decoder, so the device path falls back to host decode + HtoD for them. "Make
  `X` GPU-fast to decode" therefore means "store the GPU-relevant matrix as Scx1
  counts," **not** "re-codec a float layer."
- **Upgrade an older file in place with `scx optimize`.** A pre-v4 file (Scx1
  counts but not row-group-framed to v4) does not need a full reconvert to become
  device-decode-fast — run `scx optimize in.scx out.scx` (or
  `pyscx.optimize("in.scx", "out.scx")`; pass `codec="scx1"` to force Scx1 on
  every integer shard). It re-encodes + canonicalizes every CSR shard and
  row-group-frames it (`format_version=4`), preserving rows / obs / var / obsm /
  uns / indexes (see [operations.md § Optimize](operations.md#optimize)). Only Scx1
  integer shards decode in VRAM — a persisted float (Pcodec) `X` still won't
  (store counts per the first bullet).

Confirm the path actually taken via
`adata.uns["scx_accel"][op]["transfer_mode"]`, the same way you confirm `route`
for DE. `to_gpu_anndata` stamps one of:
- `scx_device_decode_gpu` — framed Scx1 shards decoded **fully in VRAM**
  group-by-group; only the tiny indptr is uploaded (`bytes_uploaded` ≈ indptr).
  The fast path you want. As of the BitPacker4x GPU kernel, this covers **every**
  Scx1 row, including dense (≥128-nnz) cells — those no longer host-fall-back.
- `scx_device_handoff_streamed` — on-device, but some shard still bounced through
  the host because it is **not** an Scx1 shard: a non-Scx1 codec (the float
  Pcodec case above) host-bounces. `bytes_uploaded` is the real HtoD total.
- `scx_device_handoff` — host-assembled CSR (filtered / projected / multimodal
  input), or an `X` that was already device-resident on entry.

(rapids ops that have to upload a host `X` instead stamp `anndata_to_gpu` — a host
re-upload, not a `to_gpu_anndata` mode.)

**rapids-singlecell knows nothing about the SCX device decode path.** The
in-VRAM group-by-group decode lives entirely on the SCX side of the handoff: it
accelerates SCX's own decode→device step (`to_gpu_anndata`), which *produces* the
`cupyx.scipy.sparse.csr_matrix` that rapids then operates on. rapids only ever
sees that already-decoded, device-resident matrix (it validates inputs via its
own `_check_gpu_X`) and has no knowledge of the SCX format or codecs. So choosing
an Scx1-counts layout speeds up the SCX→device handoff that *feeds* rapids — it is
not something rapids consumes, and it changes no rapids call.

### PCA (`pyscx.accel.pca`)

Two methods, auto-routed by the number of variables:

- **Covariance PCA** (CPU: n_vars ≤ 5,000): Builds the covariance matrix
  `X^T @ X` directly from CSR nonzeros via sparse outer product accumulation
  (exploiting symmetry), then eigendecomposes. Faster than randomized SVD for
  HVG-selected data. Parallel accumulation into one shared
  matrix whose columns are partitioned across rayon workers (see
  [Reproducibility](#reproducibility)). CPU-only; the former native GPU covariance path
  (`cusolverDnSsyevd`) was removed in Phase 3.2.

  **It is exact only on well-conditioned input**, and that qualifier is load-bearing
  rather than pedantic. Mean-centering a sparse cross-product means computing
  `Σxy − n·μₓ·μ_y`, a difference of same-order quantities, in every one of the
  matrix's `n_vars²` entries. When the column means are large relative to their
  variances — un-normalized counts, a raw `use_rep`, an uncentered embedding — that
  subtraction loses most of its significant digits and the eigenvalues stop
  meaning anything. Measured on a synthetic f32 fixture with a column mean of
  1e7 and a per-cell variation of 1, `variance_ratio` came back
  `[0.0, 0.0, 0.0]`; at a mean of 1e12, `variance_ratio[0]` came back as **3.72**
  — one component explaining 372% of the total variance.

  Since v0.14.0 the route detects this and says so: `total_var` is computed
  through the same guarded entry point the randomized routes use (so it is no
  longer derived from a sum of round-off-contaminated eigenvalues, and no longer
  collapses to zero), and a `warn`-level log names the condition and points at
  the remedy. The remedy is `method="randomized"`, which decomposes the data
  rather than a differenced cross-product, or normalizing / log-transforming
  first. The eigenvalues themselves cannot be repaired in this route: the
  textbook fix — center each shard before accumulating — has a nonzero term for
  every cell where *both* genes are zero, so on sparse input it is
  `O(n_obs · n_vars²)` and defeats the purpose of the method.
- **Randomized SVD** (CPU: n_vars > 5,000): Streaming shard-by-shard SpMM
  with zero-copy `MatRef::from_row_major_slice` views. Skips intermediate QR
  on transpose results for n_power_iterations ≤ 2 (matching sklearn's default).
- **GPU PCA** routes to `rapids_singlecell` (`rsc.pp.pca`) via
  `to_gpu_anndata()`. The data is handed off as a GPU-resident AnnData with
  `cupyx.scipy.sparse.csr_matrix` X — no host round-trip. The streaming/
  randomized CPU PCA path (>VRAM datasets) survives natively as a fallback.

> [!NOTE]
> **GPU PCA VRAM usage.** SCX preserves sparse CSR when handing `X` to
> rapids, but `rsc.pp.pca()` internally allocates dense working buffers
> (cuBLAS matmul) — peak VRAM during GPU PCA can be substantially higher
> than the sparse `X` footprint alone. Use
> `pyscx.accel.estimate_gpu_memory(adata, operation="pca")` to check
> whether the operation fits before launching. For datasets that exceed
> VRAM, use `backed=True` — the native streaming/randomized PCA path
> processes shards one at a time with bounded VRAM (one shard + working
> matrices). See [gpu-setup.md § GPU memory model](gpu-setup.md#gpu-memory-model)
> for the full VRAM sizing model.

Both methods work in backed mode without materializing the full matrix.

```python
import pyscx

adata = pyscx.open("atlas.scx").to_anndata(backed=True)
pyscx.accel.pca(adata, n_comps=50)

# Results written to standard scanpy slots:
#   adata.obsm["X_pca"]           — (n_obs × n_comps) float32
#   adata.varm["PCs"]             — (n_vars × n_comps) float32
#   adata.uns["pca"]["variance"]  — explained variance per PC
#   adata.uns["pca"]["variance_ratio"]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_comps` | 50 | Number of principal components |
| `zero_center` | True | Mean-center data (True = standard PCA, False = TruncatedSVD) |
| `random_state` | 0 | Random seed. Seeds the randomized SVD's Ω only — the covariance method draws no randomness. Both are deterministic; see [Reproducibility](#reproducibility) |
| `n_oversamples` | 10 | Extra dimensions for accuracy (randomized SVD only) |
| `n_power_iterations` | 2 | Power iterations for spectral accuracy (randomized SVD only) |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |
| `method` | `"auto"` | `"auto"`, `"covariance"`, or `"randomized"`. `"auto"` routes by `n_vars` (covariance when small, randomized otherwise). Explicit override is useful when benchmarking or when the auto threshold doesn't fit your data. |
| `qr_method` | `"householder"` | CPU randomized-path QR algorithm: `"householder"` (always stable) or `"cholesky"` (CholeskyQR2 — ~3× faster on well-conditioned inputs). **Ignored** by the covariance path and by the GPU rapids path. Non-SPD failures surface as `RuntimeError` with a clear "retry with qr_method='householder'" hint. |

**Key advantage:** On HVG-selected data (2,000 genes), covariance PCA
completes in 4.2s on 1M cells — 5× faster than the previous randomized
SVD and 1.9× faster than scanpy. The method is auto-selected based on
`n_vars`; no user configuration needed. On GPU, PCA routes to
`rapids_singlecell` (`rsc.pp.pca`) which handles method selection
internally. Peak memory on CPU is one shard plus working matrices (plus
~30 MB covariance matrix for 2K genes). Peak VRAM on GPU includes the
sparse `X` plus rapids' internal dense working buffers — use
`pyscx.accel.estimate_gpu_memory(adata, operation="pca")` for pre-flight
sizing (see [gpu-setup.md § GPU memory model](gpu-setup.md#gpu-memory-model)).

#### Reproducibility

**Running the same call twice on the same machine gives bit-identical results**, on both CPU
methods and on in-memory, backed and lazy `X`. Peak memory is also lower than it looks: the
reductions hold one accumulator, not one per thread.

That is worth stating because it was not always true. Through v0.13.0 inclusive, the streaming covariance
build and transpose SpMM accumulated into per-thread buffers whose merge order came from rayon
work-stealing, so five consecutive `method="covariance"` runs produced five different results
— and this table used to claim the covariance method was deterministic. Both reductions now
partition their *output* across workers rather than their input rows, so nothing is merged and
the thread schedule cannot reach the result.

One boundary is worth knowing:

| Change | Same bits? |
|---|---|
| Re-running the same call | ✅ |
| `SCX_ACCEL_PREFETCH_DEPTH` or `SCX_ACCEL_NUM_THREADS` | ✅ |
| A different `RAYON_NUM_THREADS`, or a machine with a different core count | ⚠️ see below |
| A different CPU (different SIMD width) | ❌ |

SCX's own reductions are identical at any thread count. The **dense** decomposition
underneath them — faer's QR and self-adjoint eigendecomposition — blocks its work by the
ambient rayon width, so its low bits move when that changes. This is the same contract
numpy/scipy give, where LAPACK's bits likewise move with `OMP_NUM_THREADS`, and it is why
pinning `RAYON_NUM_THREADS` is the usual advice for cross-machine comparison.

Set `SCX_ACCEL_DETERMINISTIC_LINALG=1` to remove that last dependency: it pins faer to
sequential execution, making PCA's results identical regardless of thread count. It is
opt-in because it is not free — measured at ~2.3× slower on the covariance route's
eigendecomposition, though ~1.65× *faster* on the randomized route's thin QR. Read once, at
the **first CPU PCA / PFlog call**: set it before then.

> [!IMPORTANT]
> **The guarantee is scoped to CPU PCA and PFlog, but the side effect is process-wide.**
> Those are different sets and it matters which you are relying on.
>
> - **Guaranteed reproducible:** CPU PCA and PFlog. Nothing is pinned until one of them
>   runs, and only their determinism is tested.
> - **Slowed but not made reproducible:** every *implicit* faer decomposition in the
>   process, because the setting is one global. After the first pinned PCA call, Harmony's
>   LU fallback, NB-GLM's LLT/LU/QR and the native-GPU PCA's host SVD run sequentially too.
>   If you set this knob, expect those to get slower — a call-order-dependent effect, since
>   before that first PCA call they are unaffected.
> - **Untouched:** call sites that pass faer an explicit `Par` — the exact-kNN gemm and the
>   eval-metrics distance gemm hand it `Par::rayon(0)` directly and ignore the global, so
>   pinning makes them neither sequential nor reproducible.
>
> Do not read this knob as an accelerator-wide reproducibility switch.

Version-to-version bits are not promised. The first release after v0.13.0 changes them once, by
fixing the above — so a result computed with v0.13.0 or earlier will not reproduce exactly on a
later build, and was not reproducible run to run in the first place.

### kNN graph (`pyscx.accel.neighbors`)

Approximate nearest neighbors via HNSW (Hierarchical Navigable Small
World), followed by UMAP-style fuzzy set connectivities.

```python
pyscx.accel.neighbors(adata, n_neighbors=15)

# Results written to standard scanpy slots:
#   adata.obsp["distances"]        — sparse CSR (n_obs × n_obs)
#   adata.obsp["connectivities"]   — sparse CSR (n_obs × n_obs)
#   adata.uns["neighbors"]         — metadata dict
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbors` | 15 | Number of nearest neighbors |
| `use_rep` | `"X_pca"` | Key in `adata.obsm` to use as input |
| `random_state` | 0 | Random seed |
| `ef_construction` | 200 | HNSW build parameter (higher = more accurate) |
| `ef_search` | 200 | HNSW search parameter (higher = more accurate) |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |

On CPU, uses HNSW (instant-distance) with Euclidean distance — except at
`n_obs ≤ 5,000`, where the CPU path silently dispatches to an exact kNN
via a faer matmul + per-row partial top-k sort; `ef_construction` /
`ef_search` are ignored on the exact path. The exact path allocates an
`n_obs × n_obs`
f32 Gram matrix (~100 MB at the threshold) — keep this in mind if
calling at the boundary on memory-constrained hosts.
On GPU, routes to `rapids_singlecell` (`rsc.pp.neighbors`) via
`to_gpu_anndata()`. The standalone native CAGRA dispatch was removed in
Phase 3.3; device-resident CAGRA kNN is retained only within the fused
pipeline path. Benchmarked at 4.4× on 100K cells and 9.4× on 1M cells.

### Fused PCA → kNN (`pyscx.accel.pca_neighbors`)

Runs PCA then the kNN graph in a single call. On a GPU host with
`rapids_singlecell` available, the entire pipeline runs through
`rsc.pp.pca` + `rsc.pp.neighbors` — data stays GPU-resident via
`to_gpu_anndata()`, eliminating the `obsm["X_pca"]` GPU→host→GPU
round-trip that calling `pca` then `neighbors` separately incurs (~240 MB
of host traffic at 1M cells × 60 PCs). Output is identical to the two
sequential calls; it writes every slot they do (`obsm["X_pca"]`,
`varm["PCs"]`, `uns["pca"]`, `obsp["distances"]`,
`obsp["connectivities"]`, `uns["neighbors"]`).

```python
# One call instead of pca(...) + neighbors(...).
pyscx.accel.pca_neighbors(adata, n_comps=50, n_neighbors=15, device="gpu")

# When the rapids fused path runs, all three are stamped "rapids_singlecell_gpu":
#   adata.uns["scx_accel"]["pca"]["route"]
#   adata.uns["scx_accel"]["neighbors"]["route"]
#   adata.uns["scx_accel"]["pca_neighbors"]["route"]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_comps` | 50 | Number of principal components |
| `n_neighbors` | 15 | Number of nearest neighbors |
| `zero_center` | `True` | Mean-center before PCA |
| `random_state` | 0 | Random seed |
| `n_oversamples` / `n_power_iterations` | 10 / 2 | Randomized-PCA accuracy knobs |
| `method` | `"auto"` | PCA method: `"auto"`, `"covariance"`, `"randomized"` |
| `qr_method` | `"householder"` | CPU randomized-PCA QR: `"householder"` or `"cholesky"` (ignored on GPU rapids path) |
| `use_rep` | `"X_pca"` | obsm key the `neighbors` step reads. A non-default value always runs the sequential path — the fused path runs kNN on the freshly-computed PCA embedding, so honoring `obsm[use_rep]` requires the standalone `neighbors`. |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |
| `prefer_format` | `"csr"` | Only `"csr"` is supported (PCA's SpMM path is row-major) |

Fallback: when `rapids_singlecell` is not importable — or the GPU is
unavailable — `pca_neighbors` transparently runs the standalone `pca` then
`neighbors` (each with its own normal routing and warnings), and the
`pca_neighbors` route records the fallback reason as `no_rapids`
(e.g. `cpu_csr`). The fuzzy-graph (connectivity) step runs on the CPU in
the fallback path. Accepts the same `X` inputs as `pca` (backed SCX, lazy
transform, or a materialized scipy/dense matrix).

### Fused PCA → kNN → UMAP (`pyscx.accel.pca_neighbors_umap`)

Extends the fused path through the embedding. On a GPU host with
`rapids_singlecell` available, the full pipeline runs through
`rsc.pp.pca` → `rsc.pp.neighbors` → `rsc.tl.umap` — data stays
GPU-resident via `to_gpu_anndata()` with no host round-trip. The native
CUDA SGD UMAP kernel, fuzzy simplicial set kernel, and device-resident
CAGRA kNN were removed in Phase 3 (3.1, 3.5, 3.3 respectively); rapids
is now the sole GPU path. Writes every slot `pca` + `neighbors` + `umap`
do, including `obsm["X_umap"]`.

```python
# One call instead of pca(...) + neighbors(...) + umap(...).
pyscx.accel.pca_neighbors_umap(adata, n_comps=50, n_neighbors=15,
                               n_components=2, device="gpu")

# When the rapids fused path runs, all four are stamped "rapids_singlecell_gpu":
#   adata.uns["scx_accel"]["pca"|"neighbors"|"umap"|"pca_neighbors_umap"]["route"]
```

Takes the [`pca_neighbors`](#fused-pca--knn-pyscxaccelpca_neighbors) parameters
plus the UMAP knobs: `n_components` (output dims, default 2), `n_epochs`
(default 200), `min_dist` (default 0.1), `spread` (default 1.0),
`negative_sample_rate` (default 5), `umap_learning_rate` (default 1.0).

Fallback: when `rapids_singlecell` is not importable (or no GPU), it
transparently runs the standalone `pca` → `neighbors` → `umap` on CPU (each
with its own routing/warnings) and the `pca_neighbors_umap` route records the
fallback reason as `no_rapids`. GPU UMAP via rapids is non-deterministic, so
the embedding differs run-to-run but preserves cluster structure — pin
`device="cpu"` for reproducible coordinates.

### UMAP (`pyscx.accel.umap`)

UMAP embedding with spectral initialization. On CPU, uses an SGD-based
implementation with negative sampling. On GPU, routes to
`rapids_singlecell` (`rsc.tl.umap`). Takes the kNN connectivity graph as
input.

```python
pyscx.accel.umap(adata)

# Result written to:
#   adata.obsm["X_umap"]  — (n_obs × 2) float32
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 2 | Output dimensions |
| `n_epochs` | 200 | SGD epochs (more = better quality, slower) |
| `min_dist` | 0.1 | Minimum distance in embedding |
| `spread` | 1.0 | Spread of embedded points |
| `negative_sample_rate` | 5 | Negative samples per positive edge |
| `learning_rate` | 1.0 | Initial learning rate |
| `random_state` | 0 | Random seed |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |

On GPU, routes to `rapids_singlecell` (`rsc.tl.umap`) via
`to_gpu_anndata()`. The native CUDA SGD UMAP kernel was removed in
Phase 3.1. Falls back to CPU if `rapids_singlecell` is not importable
(fallback reason: `no_rapids`).

### Leiden clustering (`pyscx.accel.leiden`)

Rust-native implementation of the Leiden algorithm (Traag, Waltman & van
Eck, 2019) with the Reichardt-Bornholdt (RB) configuration model quality
function on the CPU path; cuGraph on the GPU path. Operates directly on
the kNN connectivities CSR — no Python `igraph` / `leidenalg` dependency
required.

```python
pyscx.accel.leiden(adata, resolution=1.0)

# Results written to:
#   adata.obs["leiden"]              — categorical community labels
#   adata.uns["leiden"]["params"]    — resolution, random_state, device,
#                                     parallel, theta (cugraph), gpu_id
#                                     (cugraph), and `ignored` list of
#                                     kwargs the chosen backend dropped
#   adata.uns["leiden"]["backend"]   — "scx-accel" or "cugraph"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | 1.0 | Resolution parameter γ — higher values yield more communities |
| `key_added` | `"leiden"` | Key in `adata.obs` for community labels |
| `random_state` | 0 | Random seed for reproducibility |
| `n_iterations` | 2 | **Unit differs by backend.** Rust-native (CPU): leidenalg-style outer iterations (default 2 is plenty — each is a full multilevel cycle). cuGraph (GPU): maps to cuGraph's `max_iter` (a *coarsening-pass* count). The leidenalg default of 2 would starve cuGraph's coarsening and produce a degenerate, over-partitioned result, so the cuGraph path uses cuGraph's own default of **100** whenever `n_iterations <= 2` (including the `-1`/`0` convergence sentinels); only values `> 2` are forwarded verbatim. The effective cap is recorded in `uns["leiden"]["params"]["max_iter"]`. |
| `parallel` | `False` | Run the **Rust-native** Leiden in conflict-free batched mode. `False` (default) reproduces C++ leidenalg's sequential move-node *ordering* (the refinement omits the paper's well-connectedness admissibility conditions). **Ignored on the cuGraph path** (warns when `True`). |
| `device` | `"auto"` | `"auto"` (cuGraph if available, else Rust-native), `"cpu"` (Rust-native), `"gpu"` / `"gpu:N"` (cuGraph on CUDA device 0 or N — `gpu:N` pins via `cupy.cuda.Device(N)`). |
| `theta` | 1.0 | cuGraph-only resolution scaling knob (forwarded to `cugraph.leiden(theta=...)`). **Ignored on the Rust-native path** (warns when non-default). |

**Dispatch (post-spec):** two backends as peers, selected by `device`:

* `device="cpu"` → Rust-native (`scx_accel::leiden`). ARI ≈ 0.97 vs
  leidenalg on pbmc3k. Always available.
* `device="gpu"` → cuGraph. ARI ≈ 0.92 vs leidenalg, by design (different
  refinement strategy). Hard error if cuGraph is missing — no fallback.
* `device="auto"` (default) → cuGraph when a CUDA device is visible and
  `cugraph` imports cleanly, else Rust-native. Matches the rest of
  `pyscx.accel.*`.

**Migration note (vs the pre-spec dispatcher):** `device="auto"` previously
ran Rust-native first regardless of host. After the spec it runs cuGraph
on GPU hosts where cuGraph is installed, which produces a different
partition (ARI 0.97 → 0.92 vs leidenalg). Pin `device="cpu"` to preserve
the old behavior — required when downstream DE / annotation transfer /
UMAP coloring is keyed on specific cluster IDs from previous runs. The
Python `leidenalg` fallback has been deleted; callers who want it run
`scanpy.tl.leiden(flavor="leidenalg")` directly.

Benchmarked at 55s on 1M cells (**40× faster** than Python leidenalg's 2,226s
in same-conditions comparison) on the Rust-native path; ~3.5s on the cuGraph
path. ARI 0.92 vs Python leidenalg on census_1m for the cuGraph path. The
two backends converge to different local optima — both produce valid
high-quality community structures. Compare via ARI or NMI when switching
backends.

### Batch integration / Harmony2 (`pyscx.accel.harmony_integrate`)

Clean-room Rust implementation of the Harmony2 algorithm (Korsunsky et
al., 2019): iterative soft k-means clustering with a diversity penalty
over batch covariates, followed by ridge-regression correction of the
PCA embedding. Drop-in replacement for
`scanpy.external.pp.harmony_integrate` — the parameter names
(`key`, `basis`, `adjusted_basis`, `theta`, `lamb`) match, so existing
scanpy pipelines can swap in without other changes.

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="batch")

# PCA first — Harmony corrects the PCA embedding, not the raw matrix.
pyscx.accel.pca(adata, n_comps=30)

# Default (scanpy-compatible): write the corrected embedding to a new
# obsm key "X_pca_harmony" and leave the raw "X_pca" intact.
pyscx.accel.harmony_integrate(adata, "batch")

# Or overwrite the input embedding in place:
pyscx.accel.harmony_integrate(
    adata, "batch", adjusted_basis="X_pca"
)

# Multi-covariate integration (e.g., donor + assay):
pyscx.accel.harmony_integrate(adata, ["donor_id", "assay"])

# Downstream scanpy works on the corrected embedding just like raw PCA:
pyscx.accel.neighbors(adata, use_rep="X_pca_harmony")
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `key` | (required) | `obs` column name, or list of column names, for the batch covariate(s). Each is factorised via `pandas.factorize(sort=False)`. |
| `basis` | `"X_pca"` | `obsm` key holding the input embedding. |
| `adjusted_basis` | `"X_pca_harmony"` | `obsm` key for the corrected embedding. Default writes a **new** key, preserving `basis` (scanpy-compatible). Pass `adjusted_basis=basis` (e.g. `"X_pca"`) to overwrite in place. |
| `n_clusters` | `None` | Soft cluster count K. `None` → `min(N/30, 100)`, clamped to `[2, N/2]`. |
| `theta` | `2.0` | Diversity-penalty strength. Scalar broadcasts to every covariate. |
| `sigma` | `0.1` | Gaussian bandwidth for soft assignments. |
| `lamb` | `None` | Ridge penalty. `None` enables dynamic estimation (`alpha × E[k,b]`). |
| `max_iter` | `10` | Maximum Harmony outer iterations (cluster → correct rounds). |
| `max_iter_kmeans` | `6` | Maximum k-means sub-iterations per Harmony iter (must be ≥ 2×window_size so the convergence check can fire). |
| `random_state` | `0` | RNG seed (`ChaCha8Rng` for determinism across runs). |
| `device` | `"auto"` | `"cpu"` / `"gpu"` / `"auto"`. GPU path requires pyscx built with `--features gpu`. |

Results:

- `adata.obsm[adjusted_basis]` — corrected embedding (N × d, f32); default key
  `"X_pca_harmony"`, leaving `basis` (`"X_pca"`) intact.
- `adata.uns["harmony"]` — dict with `params`, `converged`, `n_iterations`,
  `objective_harmony` (per-iteration objective curve), and `backend`
  (`"scx-accel-cpu"` or `"scx-gpu"`).
- `adata.uns["scx_accel"]["harmony_integrate"]` — the canonical route envelope
  shared with PCA / kNN / UMAP (`route` ∈ `gpu_dense` / `cpu_dense`,
  `fallback_reason`), so you can prove GPU-vs-CPU dispatch the same way as the
  other accelerator ops. See [docs/api.md § Accelerator route metadata](../docs/api.md#accelerator-route-metadata).

**Numerical parity.** The clustering primitives are pinned against
**harmonypy 0.2.0** in `scx-accel/src/harmony/harmony_reference_values.rs`,
so `cargo test` gates them with no Python installed: the M-step
(`Y = normalize(Z_cos·Rᵀ)`), the cosine-distance kernel, the ridge
correction against `torch.linalg.inv`, and `update_R`'s softmax half. Two
of the three objective components match; the third is a **documented
divergence** (below).

End-to-end agreement is a *correlation* claim, not a numerical one, and
cannot be otherwise: SCX seeds k-means++ from `rand_chacha` where
harmonypy uses `sklearn.KMeans` and R uses Mersenne Twister, so the runs
start from different cluster geometry. The `accel_harmony` benchmark
gates mean per-PC Pearson r vs harmonypy as an absolute floor
(`benchmarks/comprehensive/thresholds.yaml`).

> The figure previously quoted here — *mean per-PC Pearson r 0.989–0.999
> against R `harmony` v2.x* — was measured **before** the soft k-means
> M-step landed, on `.npz` fixtures under `benchmarks/results/harmony/reference/`
> that are gitignored, so neither CI nor any contributor could reproduce
> it. It is not restated until it is re-measured on the current code. Two
> further corrections: the installed R package is **1.2.4** (the
> *algorithm* is Harmony2 — the version string was wrong), and
> `pyscx/tests/test_harmony_validation.py::test_per_pc_pearson_ge_095` (renamed
> in Phase 7e; it was `..._ge_0998`)
> asserts **0.95** per PC and 0.97 on the mean, not the 0.998 its name
> claims.

**Two documented divergences from harmonypy**, asserted as such rather
than left as unexplained looseness:

* The diversity penalty is `((2E+1)/(O+E+1))^θ` where harmonypy 0.2.0
  uses `(E/(O+E))^θ`. The factor of 2 cancels — it is constant across
  clusters for a fixed cell, so the per-cell L1 normalization removes it —
  but the `+1` smoothing does not.
* The objective's cross-entropy is `log((O+E+1)/(2E+1))` where harmonypy
  uses `log((O+E)/E)`, which puts SCX's term below harmonypy's by
  `log(2)·(2000/N)·Σ σ·O·θ`. Both convergence checks are ratio-based, so
  the two can converge at different sub-iterations.

Both forms are self-consistent and matched across the CPU and GPU arms.

Regenerate the reference tables with
`benchmarks/scripts/generate_harmony_references.py harmony` (and
`… lisi` for LISI), which owns the fixtures and the expected values
together so they cannot drift apart. It drives harmonypy's own
`cluster` / `moe_correct_ridge` / `update_R` / `compute_objective` —
only the driver loop is monkeypatched out, never a formula.

**Scaling** (5M cells × 30 PCs × 100 clusters, single covariate):
scx-accel CPU 37.5 min, scx-accel GPU 31.1 min, harmonypy 22.4 min,
R harmony 80.5 min. Full curves in `benchmarks/results/harmony/REPORT.md`.

### LISI — local batch mixing (`pyscx.accel.compute_lisi`)

Local Inverse Simpson Index (Korsunsky et al., 2019) — per-cell measure
of local categorical diversity. Values approach 1 when a cell's
neighbours share a single label (poor mixing) and approach the number
of categories under uniform mixing (good mixing). Useful as a
batch-integration QC summary: run before and after `harmony_integrate`
and compare the distribution shift.

```python
import pyscx
import numpy as np

# Run on the uncorrected PCA first
lisi_pre = pyscx.accel.compute_lisi(adata, "batch", basis="X_pca")

# Run Harmony, then LISI on the corrected embedding
pyscx.accel.harmony_integrate(adata, "batch", adjusted_basis="X_pca_harmony")
lisi_post = pyscx.accel.compute_lisi(
    adata, "batch", basis="X_pca_harmony"
)

# Integration improves local mixing — mean LISI should rise toward n_batches.
print(f"LISI pre={np.mean(lisi_pre):.2f}  post={np.mean(lisi_post):.2f}")

# Also written to adata.obs:
print(adata.obs["lisi_batch"].describe())
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `key` | (required) | `obs` column with the categorical label to score. |
| `basis` | `"X_pca"` | `obsm` key for the embedding to compute neighbourhoods over. |
| `perplexity` | `30.0` | Gaussian-kernel target perplexity (t-SNE-style bandwidth search). |
| `n_neighbors` | `None` | k for the kNN graph. `None` → `ceil(3 × perplexity) − 1` = **89** at the default perplexity. The `−1` is harmonypy's shape, not a fencepost: `harmonypy.lisi.compute_lisi` retrieves `3 × perplexity` neighbours and then drops column 0, its own self-match, while SCX's sweep skips `j == i` as it collects. Before v0.15 this was `ceil(3 × perplexity)` in three places — the Rust default and both bindings — giving one neighbour more than harmonypy. Pinned in `scx-accel/src/lisi_reference_values.rs`. |
| `approximate_knn` | `False` | Use HNSW approximate kNN instead of the exact O(N²) sweep. ~10× faster at N ≳ 100k, with ~0.01–0.05 mean-LISI drift. |

Returns a `numpy.ndarray` of length N and also writes the values to
`adata.obs[f"lisi_{key}"]`.

By default the implementation uses an exact brute-force kNN (per-row
squared-norm expansion + per-cell top-k heap) and follows the harmonypy /
R `lisi` LISI formulation (raw-distance Gaussian kernel `exp(-D·β)`). On
D1–D4 it is **~10× faster** than R `lisi::compute_lisi`; the previously
reported mean-LISI agreement of 0.8–2.4 % predates the 2026-07 raw-distance
kernel fix and is pending a benchmark recapture.
Brute-force kNN is O(N²·d); above ~50k cells the exact path logs a hint
to set `approximate_knn=True`, which swaps in an HNSW kNN for an
order-of-magnitude speed-up at census scale (D5+) at the cost of small
numerical drift (~0.01–0.05 on mean LISI).

```python
# Census-scale: avoid the O(N²) exact sweep.
lisi = pyscx.accel.compute_lisi(adata, "batch", approximate_knn=True)
```

### Gene-set scoring (`pyscx.accel.score_genes`)

CPU-native equivalent of `sc.tl.score_genes` — a per-cell score for a gene
signature, written to `adata.obs[score_name]`. Streams shard-by-shard, so it
runs identically on in-memory, backed, and lazy `X` with bounded memory.

Three methods via `method=`:

| `method`    | Score per cell                                                   | Notes |
|-------------|------------------------------------------------------------------|-------|
| `"control"` | `mean(gene_list) − mean(control)` (default; scanpy `score_genes`)| Control genes sampled from expression-matched bins. |
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
```

> **Divergence from scanpy.** The `control` method replicates scanpy's
> rank-binning + control-gene sampling algorithm, but the sampler is
> Rust-native (a fixed `ChaCha8` stream seeded by `random_state`, reproducible
> across `rand` upgrades) and seeded independently of numpy, so the *specific*
> control genes (and therefore the absolute scores) differ from
> `sc.tl.score_genes`. The score is deterministic for a fixed `random_state` and
> rank-correlates near-perfectly with scanpy in practice. Genes in `gene_list`
> (or an explicit `gene_pool`) not present in `adata.var_names` are dropped with
> a `UserWarning`. Use `layer=` to score a named layer instead of `X`. CPU-only —
> `device` is accepted for API symmetry but there is no GPU kernel.

### PFlog normalization (`pyscx.accel.pflog`)

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

### Differential Expression

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
`gpu_csc_v3` when a sidecar is present, CSR-direct `gpu_csr_v3` otherwise).
The NB-GLM functions are CPU-only.

#### `pyscx.accel.rank_genes_groups`

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

# Downstream scanpy works identically:
sc.pl.rank_genes_groups(adata, n_genes=20)
df = sc.get.rank_genes_groups_df(adata, group="0")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groupby` | (required) | Column in `adata.obs` to group cells by. Cells with no label — NaN, empty, or a value outside the column's categories — are **excluded from the test entirely**: they are not part of a group, not part of `"rest"`, and not in the rank pool, matching scanpy, which subsets them out before ranking. A `UserWarning` reports how many were dropped. |
| `reference` | `"rest"` | Compare against a specific group or `"rest"` (1-vs-rest) |
| `n_genes` | all | Number of top genes to report per group |
| `method` | `"wilcoxon"` | Statistical method (currently only `"wilcoxon"`) |
| `rankby_abs` | `False` | Sort genes by absolute z-score instead of signed score. `False` (default) matches scanpy's default: highest positive z-score first. `True` ranks by significance regardless of direction. |
| `tie_correct` | `False` | Apply the `Σ(t³−t)` tie correction to the Wilcoxon rank-sum variance estimate. The default `False` matches **scanpy's default** (`scanpy.tl.rank_genes_groups` takes the same parameter, also defaulting to `False`); `True` matches **scipy**, which always corrects. These are two different answers, not two precisions — see [Numerical parity](#numerical-parity-against-scanpy-and-scipy) below. |
| `gene_chunk_size` | `None` | Process genes in chunks of this size to limit memory. `None` processes all genes at once. |
| `prefer_format` | `"auto"` | `"auto"` (default; CPU routes CSC-direct when a valid sidecar is present, else CSR), `"csr"`, or `"csc"`. |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"`. GPU routes to CSC-direct (`gpu_csc_v3`) when a sidecar is present, or CSR-direct (`gpu_csr_v3`) otherwise. |

Benchmarked at 5.4s on 1M cells (3.2× faster than scanpy's 17.2s).

##### Numerical parity against scanpy and scipy

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
conventions differ by **7.6e-02** in `z`, so pinning one arm against the other's
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

#### `pyscx.accel.pdex_ref`

Perturbation DE in reference mode — computes Wilcoxon rank-sum fold changes
for each non-reference group against the reference (e.g. `"non-targeting"`
or `"control"`). Designed for Perturb-seq experiments where you compare each
perturbation against a common control population. Returns a **pandas**
DataFrame in cell-eval's `DEResults` column schema; pass `output="polars"` for
the polars frame `cell_eval` (and upstream `pdex`) use.

```python
df = pyscx.accel.pdex_ref(adata, "perturbation", reference="non-targeting")
# Columns: target, feature, fold_change, p_value, fdr,
#          log2_fold_change, abs_log2_fold_change
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groupby` | (required) | Column in `adata.obs` containing perturbation labels |
| `reference` | `"non-targeting"` | Control group label |
| `is_log1p` | `None` | Whether input X is log1p-transformed. `None` auto-detects, layout-independently — a backed handle and an in-memory `AnnData` over the same data resolve the same mode. Order: `adata.uns["log1p"]` → a lazy `X`'s `Log1p` transform → a backed `X`'s catalog `value_max` against a `< 30` heuristic (catalog-only, no decode) → in-memory `max(X) < 30`. **Two cases raise `ValueError` instead of guessing**: a backed file whose catalog cannot bound its value range — shards that are float-encoded (the format records no range for those), carry no statistics, or hold no values — and a lazy `X` carrying a rescaling-only chain such as `normalize_total` (which detaches the values from the recorded range). Pass `True`/`False` to resolve either — or run `pyscx.accel.log1p`, which stamps `uns["log1p"]` and settles it. |
| `geometric_mean` | `True` | Use geometric mean for fold-change computation |
| `epsilon` | `1e-9` | Finite-guard pseudocount on count-space means before fold-/percent-change (not CPM/MWU). Default keeps outputs finite; `0/0 → 0.0`. Pass `0.0` for legacy `±inf` on reference-undetected genes. |
| `cpm_filter` | `None` | Optional CPM floor `T`: keep a gene iff `target_cpm > T` or `ref_cpm > T` (pooled arithmetic CPM, mode-independent); drops other rows, FDR recomputed over survivors. |
| `gene_chunk_size` | `None` | Process genes in chunks to limit memory |
| `prefer_format` | `"auto"` | `"auto"` (default), `"csr"`, or `"csc"` |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"`. GPU takes the CSC-direct route (`gpu_csc_v3`) when a sidecar is present. |
| `output` | `"pandas"` | `"pandas"` (needs no extra) or `"polars"` (needs the `eval` extra; what `cell_eval` consumes) |

### Pseudobulk Differential Expression (`pyscx.accel.pseudobulk_dex`)

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

### Stratified Differential Expression

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

### NB-GLM backend (Rust-native pseudobulk DE)

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
contrasts. See [docs/pseudobulk_nb_glm.md § Custom designs](pseudobulk_nb_glm.md#custom-designs-formula).

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
[docs/pseudobulk_nb_glm.md](pseudobulk_nb_glm.md).

### Perturbation evaluation metrics (cell-eval / arc-bench parity)

SCX ships Rust-accelerated equivalents of the metrics in
[`cell-eval`](https://github.com/arcinstitute/cell-eval) and
[`arc-bench`](https://github.com/arcinstitute/arc-bench). The outputs are
numerically equivalent to the Python references within the tolerances
below, so an existing cell-eval pipeline can swap in `pyscx.accel.*` for 10–20×
wall-clock speedup at census-scale perturbation datasets (see
[`docs/performance.md`](performance.md#perturbation-metrics-cell-eval--arc-bench-parity)
for numbers at 10K / 100K / 500K / 1M cells).

**How each tolerance is gated.** Every row below is pinned twice:

- **Rust-side**, against reference values `cell-eval` and `arc-bench`
  *produced*, checked into `scx-accel/src/eval_metrics/cell_eval_reference_values.rs`
  and asserted by `cell_eval_reference_tests.rs`. This runs under plain
  `cargo test` with no Python installed, and it is what makes these claims
  reproducible. Regenerate with
  `.venv/bin/python benchmarks/scripts/generate_eval_metrics_references.py`.
- **Python-side**, by `pyscx/tests/test_cell_eval_parity.py` against the live
  libraries. That file `importorskip`s `cell_eval` / `arc_bench` / `polars`,
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

#### Pseudobulk means (`pyscx.accel.pseudobulk_means`)

Group-by mean on sparse `X`. Foundation for the pairwise metrics below.

```python
means, groups = pyscx.accel.pseudobulk_means(adata, "perturbation")
# means.shape == (n_perturbations, n_genes), dtype float64
# groups == ["control", "drug_A", "drug_B", ...]  (sorted)
```

Streams directly from CSR shards with no full-matrix materialization. On
backed data, processes shard-by-shard; on lazy-transformed data, applies
the transform stack before aggregation. **Dense fast-path**: when
`adata.X` is a dense numpy array (the common shape after
`pp.normalize_total + log1p`), the in-memory aggregation
runs through `scx_accel::pseudobulk_aggregate_dense` directly, bypassing
the historical `scipy.sparse.csr_matrix(dense_array)` round-trip. At
24K-cell × 18K-gene shapes this cut `pseudobulk_means` from ~22 s to ~5 s.

#### Bulk perturbation metrics (`pyscx.accel.perturbation_metrics`)

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

#### Discrimination score (`pyscx.accel.discrimination_score`)

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

#### Energy distance (`pyscx.accel.energy_distance`)

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
[performance.md § Pairwise-distance kernels](performance.md#pairwise-distance-kernels-gram-blocking-and-the-with_min_len-fix).

#### Knockdown efficiency (`pyscx.accel.knockdown_efficiency`)

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
isn't in `var_names` get `NaN` in both columns — matching `arc-bench`.

#### Clustering agreement (`pyscx.accel.clustering_agreement`)

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

#### DE result format bridge (`pyscx.accel.rank_genes_groups_df`)

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
> scanpy's columns (`names, scores, logfoldchanges, pvals, pvals_adj`), a
> drop-in for `sc.get.rank_genes_groups_df`. Use `groupby=` to recompute (cell-eval
> columns), `group=` to extract (scanpy columns); pass one, not both. The scanpy
> filters `pval_cutoff` / `log2fc_min` / `log2fc_max` apply to the `group=` path.


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

### GPU helper functions

```python
# Query GPU availability and memory
info = pyscx.accel.gpu_info()
# {'device': 'NVIDIA A100-SXM4-80GB', 'total_vram_gb': 80.0, 'free_vram_gb': 72.3}

# Estimate GPU memory requirements
est = pyscx.accel.estimate_gpu_memory(adata, operation="pca", n_comps=50)
# {'required_gb': 2.1, 'fits_in_vram': True}
```

### GPU vs CPU numerical differences

GPU and CPU accelerators may produce slightly different results due to:

| Factor | Impact | When it matters |
|--------|--------|-----------------|
| **PCA precision** | GPU uses f32 throughout; CPU uses f64 | Native GPU vs CPU: per-PC cosine ≥ 0.99 on the leading PCs — no biological impact |
| **kNN algorithm** | GPU uses CAGRA (graph-based ANN); CPU uses HNSW | Both are approximate; exact agreement is not expected. **No GPU-vs-CPU recall bar is enforced** — see the table below |
| **UMAP non-determinism** | GPU uses `atomicAdd` (race conditions are intentional) | Embedding coordinates differ; cluster structure preserved |
| **Leiden** | Rust-native vs cuGraph vs leidenalg may produce different partitions | Label stability differs **by design**; the enforced GPU-vs-CPU bar is a degeneracy floor, not agreement — see the table below |

### Tolerance thresholds (correctness tests)

Each row below names the test that enforces it. Where nothing enforces a row,
it says so rather than implying a gate that does not exist.

| Test | Metric | Threshold | Enforced by |
|------|--------|-----------|-------------|
| **Native** GPU PCA vs CPU PCA | Per-PC cosine, leading `n_clusters - 1` PCs | ≥ 0.99 | `test_gpu_randomized_householder_vs_cpu_randomized` — pins `SCX_FORCE_NATIVE_GPU=1`, so it does **not** cover the `device="gpu"` default, which is rapids-singlecell |
| **Native** GPU PCA, Cholesky QR vs Householder QR | Per-PC cosine | ≥ 0.999 | `test_gpu_cholesky_matches_householder` — GPU vs GPU, not a CPU comparison, and it pins `SCX_FORCE_NATIVE_GPU=1` too: rapids ignores `qr_method`, so without the pin both arms are the same path and the comparison is vacuous |
| `normalize_total`→`log1p`→`pca` at `device="gpu"` on an SCX-round-tripped **in-memory** AnnData, vs scanpy | Per-PC cosine, top 10 PCs | ≥ 0.99 — but see below: this is **not** a GPU-route gate | `test_normalize_log1p_pca_matches_scanpy` |
| GPU normalize + log1p vs CPU | Element-wise relative | < 1e-5 | `scx-gpu/src/gpu_preprocess_tests.rs::test_gpu_normalize_log1p_matches_cpu` |
| GPU Leiden vs CPU Leiden | ARI | ≥ 0.10, plus a cluster-count bound | `test_gpu_leiden_not_degenerate` |
| GPU kNN vs CPU HNSW | Recall@k | **not enforced** | — |
| GPU UMAP | Trustworthiness | **not enforced** | — |

Four notes on why these are what they are, since several look alarming out of
context:

- **Leiden's ARI floor is 0.10 on purpose, not by neglect.** GPU label stability
  differs from `leidenalg` by design (see the note above and README's Known
  Limitations), so a high ARI would be asserting a property SCX explicitly does
  not promise. The test pairs the loose ARI — a degeneracy canary, against an
  observed 0.0002 for a collapsed partition — with a cluster-count bound, which
  is the assertion that actually has teeth. Pin `device="cpu"` when you need
  label stability.
- **kNN recall and UMAP trustworthiness have CPU-path tests, not GPU ones.**
  `pyscx/tests/test_accel.py` asserts HNSW recall ≥ 0.90 against brute force and
  UMAP trustworthiness > 0.75, both on `device="cpu"`. Neither compares a GPU
  result to anything, so neither backs a GPU tolerance.
- **Two of these rows pin `SCX_FORCE_NATIVE_GPU=1`, and that is not a detail.**
  The `device="gpu"` default routes in-memory PCA to rapids-singlecell, so both
  native-path bars are invisible to it.
- **The third PCA row asks for the default path but does not gate it.** It calls
  `normalize_total` / `log1p` / `pca` at `device="gpu"` on a materialized AnnData
  — which is the rapids route — and its numeric bar is real. What it never does
  is assert *which route served the request*: none of the three ops has its
  `uns["scx_accel"][op]["route"]` checked. On a CUDA host without
  rapids-singlecell installed, every one of them takes the documented
  `NoRapidsCpu` fallback (`pyscx/src/accel/pca.rs`) and the whole SCX side runs
  on the **CPU** while the test still passes. Read it as a correctness check of
  the op chain, not as evidence that a GPU kernel ran.
  ⚠️ An earlier revision of this note said "no row here gates the default path",
  which was wrong in the other direction — this row does target it. Both
  statements were mis-citations of the kind this table exists to remove.
- **The PCA rows are split three ways on purpose.** An earlier revision of this
  table had one "GPU PCA vs CPU PCA" row claiming "≥ 0.999 (GPU arms), ≥ 0.99 (vs
  scanpy)". The 0.999 bar is `test_gpu_cholesky_matches_householder`, which
  compares two **GPU** QR methods to each other — attributing it to a GPU-vs-CPU
  row overstated what is gated, which is the same mis-citation this table exists
  to remove. The real GPU-vs-CPU bar is ≥ 0.99 on the leading `n_clusters - 1`
  PCs, and it pins `SCX_FORCE_NATIVE_GPU=1`, so it says nothing about the
  `device="gpu"` default path (rapids-singlecell) that most users actually take.
- Earlier revisions of this table cited `pyscx/tests/test_accel_gpu.py` as the
  enforcement mechanism for all five rows. **That file has never existed**, and
  four of the five thresholds it was said to enforce did not match any
  assertion in the tree.

### Checking which backend was used

All accelerators record the backend in `adata.uns`:

```python
pyscx.accel.pca(adata, device="gpu")
print(adata.uns["pca"]["backend"])          # "rapids_singlecell_gpu"
print(adata.uns["pca"]["device"])           # "NVIDIA A100-SXM4-80GB"
print(adata.uns["pca"]["gpu_time_ms"])      # 1234.5

pyscx.accel.neighbors(adata, device="gpu")
print(adata.uns["neighbors"]["backend"])    # "rapids_singlecell_gpu"

pyscx.accel.umap(adata, device="gpu")
print(adata.uns["umap"]["backend"])         # "rapids_singlecell_gpu"
```

On GPU, PCA, kNN, and UMAP all route to `rapids_singlecell` (`rsc.pp.pca`,
`rsc.pp.neighbors`, `rsc.tl.umap`). Preprocessing ops (`normalize_total`,
`log1p`, `highly_variable_genes`) also route to `rsc.pp.*` on GPU. GPU
Leiden uses cuGraph: `device="cpu"` runs the Rust-native CPU path,
`device="gpu"` runs cuGraph and hard-errors if cuGraph is absent (the
Python `leidenalg` fallback was deleted — call
`scanpy.tl.leiden(flavor="leidenalg")` directly if you need it). Set
`SCX_FORCE_NATIVE_GPU=1` to pin
surviving native GPU paths (HVG `seurat_v3`, DE, Harmony, Leiden); set
`SCX_DISABLE_RAPIDS=1` to force the rapids-absent fallback for testing.
When rapids is unavailable, a one-shot `UserWarning` is emitted and the
fallback reason `no_rapids` is recorded in the route metadata.

## Multithreading

Most scx-accel and pyscx entry points are multithreaded via rayon by default,
and release the GIL (`py.allow_threads()`) so Python stays responsive. For the
full architecture — runtimes, thread pools, channel topology, and how to control
parallelism — see [docs/multithreading.md](multithreading.md).

### Per-function threading

| Function | Threading | Notes |
|----------|-----------|-------|
| `pyscx.open()`, `to_anndata()`, `read_layer()` | Rayon parallel shard decode | `scx-format` with `parallel` feature; SIMD BitPacker4x within each shard |
| `ScxBackedDataset` slicing / column projection | Rayon per access | Each `X[...]` call decodes touched shards in parallel |
| `pyscx.query().where(...).collect()` | Rayon parallel shard decode | Only shards surviving catalog pushdown are decoded |
| `pyscx.iter_chunks()`, `pyscx.preprocess()`, `pyscx.save_layer()` | Rayon parallel shard decode + encode | Parallel encode achieves up to 3.2× at 32 threads |
| `pyscx.accel.pca` (CPU) | Rayon | Covariance and transpose SpMM partition their output columns across workers — one shared accumulator, no merge |
| `pyscx.accel.neighbors` (CPU) | Rayon | Parallel kNN queries on HNSW index |
| `pyscx.accel.umap` (CPU) | Single-threaded SGD | Edge updates are serial on CPU; GPU path uses CUDA kernel parallelism |
| `pyscx.accel.leiden` | Opt-in rayon via `parallel=True` | Sequential by default (reproduces C++ leidenalg's move-node ordering); parallel uses conflict-free graph coloring |
| `pyscx.accel.rank_genes_groups` | Rayon | Parallel Wilcoxon rank-sum across genes |
| `pyscx.accel.pseudobulk_dex` | Rayon (aggregation) | Streaming aggregation is parallel; downstream `pydeseq2` testing runs single-threaded |
| `pyscx.accel.highly_variable_genes` | Rayon (via streaming reader) | Parallelism comes from shard decode; the mean/var reduction itself is serial |
| `pyscx.accel.perturbation_metrics`, `energy_distance` | Rayon (CPU) or GPU | CPU parallelizes across perturbations / pairwise-distance rows; GPU runs pseudobulk means (`perturbation_metrics`) or a gemm pairwise-distance mean (`energy_distance`, euclidean/cosine f32) on the device |
| `pyscx.accel.discrimination_score` | Rayon (CPU only) | Parallelizes across perturbations; no GPU kernel (exact-rank parity not f32-safe) |
| `pyscx.accel.knockdown_efficiency`, `clustering_agreement` | Rayon | Parallel per-perturbation / per-label reductions |
| `pyscx.accel.harmony` (batch correction) | Rayon | Parallel per-cluster correction |
| `pyscx.accel.normalize_total`, `log1p`, `filter_cells`, `filter_genes`, `calculate_qc_metrics` | Rayon (via streaming reader) | Lazy — no work until materialized or consumed |
| `pyscx.pull()` / `pyscx.push()` (cloud) | Tokio async | `parallelism` parameter controls concurrent transfers (default 8) |
| `pyscx.TrainingDataset` | Triple-buffered (tokio I/O + rayon decode + Python consumer) | See [multithreading.md §Training data loader](multithreading.md#training-data-loader-triple-buffered-pipeline) |
| `pyscx.accel.*` with `device="gpu"` | CUDA kernel parallelism | CPU side launches kernels and manages transfers |

### Controlling parallelism

```python
import os
os.environ["RAYON_NUM_THREADS"] = "8"   # must be set before `import pyscx`
import pyscx
```

`RAYON_NUM_THREADS` sizes the process-wide rayon pool that most accelerators use.
`SCX_ACCEL_NUM_THREADS` is a narrower ceiling for the accelerators' private rayon work —
Harmony batch integration, and the number of column blocks the PCA reductions split their
output into — so you can cap those on a fat node without shrinking every op. It is read
once at first use (set it before the first accelerator call); unset (the default) leaves
today's behaviour unchanged. On PCA it bounds **speed and memory only**: the block count
cannot change the numbers, because the blocks write to disjoint slices and are never merged
(see [PCA § Reproducibility](#reproducibility)). Other `SCX_ACCEL_*` knobs (`SCX_ACCEL_PREFETCH_DEPTH`,
`SCX_ACCEL_REDUCTION_MODE`, `SCX_ACCEL_DE_MEMORY_BUDGET`) are documented in
[performance.md](performance.md); `SCX_ACCEL_PAIRWISE_MEMORY_BUDGET`, which bounds one
`energy_distance` Gram block and so interacts with `RAYON_NUM_THREADS` multiplicatively, is
in the [architecture.md environment table](architecture.md#environment-variables). `SCX_ACCEL_PREFETCH_DEPTH` bounds the
decode-prefetch pipeline, which since Phase 4.2 also covers the backed
aggregation kernels (QC, filtering, `col_*`, `normalize_total`'s row sums), their
column-projected **CSR** and lazy/transformed twins, and GPU staging — so raising it
raises peak memory (`depth` decoded shards in flight) across all of those, not
just HVG.

Three consequences worth knowing before you tune it:

- **On a memory-tight GPU host, consider `SCX_ACCEL_PREFETCH_DEPTH=2`.** GPU
  staging keeps host RSS low otherwise (1.7–4.5 GB in the Phase-4.2 capture), so
  the extra `depth − 1` decoded shards are plainly visible there: **+19–46 %**
  peak host RSS across every measured op. Depth 2 keeps most of the overlap at
  roughly a third of the extra footprint. The default of 4 is tuned for
  throughput, not for the tightest node.
- **`RAYON_NUM_THREADS=1` now makes GPU staging fully sequential.** Before 4.2 it
  had a dedicated `std::thread` that overlapped one shard ahead *unconditionally*;
  the shared pipeline instead declines to engage on a single-thread pool (that
  guard is what prevents a nested-call deadlock). If you pin
  `RAYON_NUM_THREADS=1` for reproducibility, GPU HVG/DE will be **slower than
  before 4.2**, not faster. For **CPU PCA** you no longer need to pin it at all —
  repeated runs agree at any thread count, and `SCX_ACCEL_DETERMINISTIC_LINALG=1`
  covers the cross-thread-count case more cheaply than serializing everything
  (see [PCA § Reproducibility](#reproducibility)).
- **`prefer_format="csc"` does not inherit the 4.2 speedups.** The CSC column
  kernels reach their source through `&dyn ColumnShardSource` and still decode
  serially; only the CSR paths are prefetched. They do benefit from the `col_*`
  GIL release.

Since Phase 4.5, `SCX_GPU_STAGING_MEMORY_BUDGET` (bytes) expresses the same
bound in a unit that does not depend on the file: GPU staging derates the
prefetch depth so `depth × per-shard-decoded-bytes` fits the budget, using the
per-shard `nnz` the catalog already carries. Unset — the default — nothing is
derated and the depth is exactly `SCX_ACCEL_PREFETCH_DEPTH`.

### GPU DE device residency

GPU DE's CSR route (`gpu_csr_v3`, the mandatory route for any file **without** a
CSC sidecar) has no column-range prefilter, so before Phase 4.5 it walked every
shard once **per gene chunk** — `n_gene_chunks × n_shards` host decodes and
uploads. On a 61 497-gene file at the default 500-gene chunk that is 123 full
passes over the matrix, and it made GPU DE almost entirely host-decode-bound.

4.5 drains the source **once** into device-resident per-shard CSR buffers and
serves every later chunk from VRAM, and narrows each row to the chunk's column
window with a binary search instead of scanning the whole row and predicating
per element. Neither changes what the kernels compute.

The cost is device memory: one f32 value plus one i32 index per nonzero, so
roughly `8 × nnz` bytes for the whole matrix on top of the per-chunk scratch.
Two knobs:

- **`SCX_GPU_DE_RESIDENT_MAX_FRAC`** (default `0.5`) — the fraction of *free*
  VRAM the resident matrix may occupy. The remainder is what the per-chunk
  gene-slab budget then sizes itself against, so a matrix that takes half the
  card simply yields a smaller gene chunk rather than an OOM.
- **`SCX_GPU_DE_RESIDENT=0`** — kill switch. Restores the pre-4.5 streaming
  behaviour exactly.

Native GPU **PCA** has the same two-path structure and its own kill switch,
**`SCX_GPU_PCA_RESIDENT=0`**, which forces the streaming power loop (the whole
matrix re-decoded and re-uploaded on every multiply). Both ops record which path
ran on `resident_csr` — see
[api.md § Accelerator route metadata](api.md#accelerator-route-metadata). The
PCA knob is also what makes the streaming loop reachable from a test: residency
is decided against *free* VRAM at call time, so on a large card no fixture can
force the streaming branch by shape alone.

A third knob, `SCX_GPU_VALIDATE_PAR_MIN_NNZ`, sets the shard size above which
the per-shard GPU-DE validation scan (strictly-increasing columns + finiteness)
runs in parallel; default 65 536 nnz, and pinning it above any real shard's nnz
restores the serial scan. It exists mainly so that choice stays measurable —
the parallel scan is a **measured no-op** at current scales, because it runs on
the consuming thread while the prefetch workers decode ahead and is therefore
hidden behind decode.

Residency is declined — silently and correctly — when the matrix does not fit
the budget, or when there is only one gene chunk (streaming would run one pass
anyway, so retaining the matrix would be pure cost). Because a declined run
produces *identical output*, just slower, the only way to tell is the route
stamp: `adata.uns["scx_accel"][<op>]["resident_csr"]` is `True` when residency
engaged, `False` when the CSR route streamed, and `None` on a route with no
residency decision to make (CPU, dense, or CSC-direct). A benchmark gate
(`de_route_resident_csr`) floors it for exactly that reason.

The CSC-direct route is unaffected: it already prefilters by column range and
never re-decodes.

The heavy accelerators, including `pyscx.accel.col_sums` and its siblings,
release the GIL for their streaming scan, so they can be called concurrently
from Python threads without serialising each other.

A CPU accelerator that releases the GIL works from an **owned snapshot** of `X`,
taken before the release: the in-flight result reflects the matrix as it was
when the kernel started, never a half-updated blend. On an **in-memory** `X`
that snapshot is a copy; it is taken in parallel, so it costs rather less than
`X.copy()` (measured 14 ms for a 192 MB dense matrix, against 119 ms for
`np.copy`), and `pseudobulk_means` warns once it exceeds 1 GB. A **backed or
lazy** `X` needs no copy at all: it streams shard by shard from the file, which
is the cheaper way to run these ops on a large matrix regardless.

That is a statement about those kernels, not a general licence to mutate `X`
from another thread mid-call. The GPU in-VRAM fast lane reads `X`'s buffers
directly and is safe only because it holds the GIL throughout — nothing is
snapshotted — and conversion (`pyscx.from_anndata`) makes several passes over
`X` that are not guaranteed to see one consistent state. Mutating a matrix
while any scx call is reading it remains unsupported; what changed is that a
detached kernel can no longer read a buffer out from under you.

The cloud runtime exposes its own knob:

```python
pyscx.pull("gs://bucket/atlas.scxd/", "atlas.scx", parallelism=16)
```

## Common scanpy workflows

### Clustering and visualization

```python
import pyscx
import scanpy as sc

adata = pyscx.open("experiment.scx").to_anndata()

# QC
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

# Normalize and select HVGs
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var["highly_variable"]].copy()

# Dimensionality reduction and clustering (standard scanpy)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# Visualization
sc.pl.umap(adata, color=["leiden", "cell_type"])
```

### Accelerated pipeline for large datasets

For datasets with >100K cells, use `pyscx.accel.*` for a fully
out-of-core pipeline — no materialization from file open through
clustering:

```python
import pyscx
import scanpy as sc

# Open in backed mode — X stays on disk
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# QC works natively in backed mode (no materialization)
# Gene subsets (e.g., mitochondrial) use column-projected streaming
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# HVG selection — streaming variance
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Lazy preprocessing — NO materialization
pyscx.accel.normalize_total(adata, target_sum=1e4)   # → lazy wrapper
pyscx.accel.log1p(adata)                              # → appends to chain

# PCA streams through lazy transforms shard-by-shard
pyscx.accel.pca(adata, n_comps=50)
pyscx.accel.neighbors(adata, n_neighbors=15)
pyscx.accel.umap(adata)
sc.tl.leiden(adata)

sc.pl.umap(adata, color="leiden")
```

> Peak RSS for this pipeline on census_1m (1M cells × 61K genes) is ~11 GB
> (vs ~22 GB with the traditional materialize-then-preprocess approach — a
> 51% reduction). The lazy preprocessing path streams shard-by-shard through
> transforms without materializing the full matrix. The remaining RSS is
> dominated by kNN graph construction and UMAP embedding, not preprocessing.
> The Python interpreter + library baseline is ~450 MB.

### GPU-accelerated analysis pipeline

When a CUDA GPU is available, use `device="gpu"` for GPU-accelerated
analysis (up to 16× per-op, 3.8× end-to-end on 1M cells). See
[`docs/gpu-setup.md`](gpu-setup.md) for installation instructions (conda,
system CUDA, or container).

```python
import pyscx
import scanpy as sc

# Open in backed mode for memory-efficient QC
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# QC works natively in backed mode
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Subset to HVGs and materialize
adata_sub = adata[:, adata.var["highly_variable"]].copy()
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

# GPU-accelerated pipeline (falls back to CPU if no GPU)
pyscx.accel.pca(adata_sub, n_comps=50, device="gpu")
pyscx.accel.neighbors(adata_sub, n_neighbors=15, device="gpu")
pyscx.accel.umap(adata_sub, device="gpu")
sc.tl.leiden(adata_sub)  # CPU Leiden; use device="gpu" for cuGraph

sc.pl.umap(adata_sub, color="leiden")

# Check which backends were used
print(adata_sub.uns["pca"]["backend"])       # "rapids_singlecell_gpu"
print(adata_sub.uns["neighbors"]["backend"]) # "rapids_singlecell_gpu"
print(adata_sub.uns["umap"]["backend"])      # "rapids_singlecell_gpu"
```

### Differential expression

```python
adata = pyscx.open("experiment.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Using scanpy directly (works fine):
sc.tl.rank_genes_groups(adata, groupby="cell_type", method="wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=20)

# Or using pyscx accelerator (3× faster, supports GPU):
pyscx.accel.rank_genes_groups(adata, "cell_type")
sc.pl.rank_genes_groups(adata, n_genes=20)  # scanpy plotting works identically

# Perturbation DE (Perturb-seq experiments):
df = pyscx.accel.pdex_ref(adata, "perturbation", reference="non-targeting")

# Pseudobulk DE with biological replicates:
result = pyscx.accel.pseudobulk_dex(
    adata, groupby=["perturbation", "donor"],
    test_col="perturbation", reference="control",
    backend="nb_glm",  # Rust-native NB-GLM, no pydeseq2 needed
)
```

### Batch integration

Two integrated batch-correction paths, neither of which requires leaving
the pyscx stack:

**Harmony2 (fast, linear).** Clean-room Rust port of Harmony2. Operates
on the PCA embedding — correct once, feed the result into kNN / UMAP /
Leiden as if it were raw PCA. Drop-in replacement for
`scanpy.external.pp.harmony_integrate`.

```python
import pyscx
import scanpy as sc

adata = pyscx.open("multi_batch.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="batch")

pyscx.accel.pca(adata, n_comps=30)
pyscx.accel.harmony_integrate(adata, "batch")  # writes obsm["X_pca_harmony"], keeps "X_pca"

# Use corrected embedding for downstream:
pyscx.accel.neighbors(adata, use_rep="X_pca_harmony")
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)

# QC: batch mixing before / after — both embeddings are available by default
lisi_pre = pyscx.accel.compute_lisi(adata, "batch", basis="X_pca")
lisi = pyscx.accel.compute_lisi(adata, "batch", basis="X_pca_harmony")
print(f"mean LISI batch = {lisi.mean():.2f}  (ideal: ≈ n_batches)")
```

**scVI (deep-learning, nonlinear).** When you need nonlinear
integration or want to learn a latent space for transfer tasks.

```python
import pyscx
import scanpy as sc
import scvi

adata = pyscx.open("multi_batch.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, batch_key="batch")

scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
model = scvi.model.SCVI(adata)
model.train()
adata.obsm["X_scVI"] = model.get_latent_representation()
```

Harmony runs in seconds-to-minutes on 1M cells (see
[`docs/performance.md`](performance.md#harmony2-batch-integration--lisi));
scVI adds a GPU-minutes training phase but captures nonlinear
batch effects Harmony cannot. For most routine integration tasks start
with Harmony; reach for scVI when Harmony underperforms on a
`compute_lisi` gate.

## File operations with scanpy

### Removing doublets and saving back

```python
import pyscx
import scanpy as sc
import scrublet

adata = pyscx.open("experiment.scx").to_anndata()

# Run doublet detection
scrub = scrublet.Scrublet(adata.X)
scores, predicted = scrub.scrub_doublets()

# Mark doublets as deleted (instant, no data rewrite)
exp = pyscx.open("experiment.scx")
exp.mark_deleted(predicted)  # boolean numpy array

# Or save the filtered result as a new file
adata_clean = adata[~predicted].copy()
pyscx.from_anndata(adata_clean, "experiment_clean.scx")
```

### Appending new batches

```python
# Append cells from another SCX file (streaming — reads one shard at a time)
pyscx.append("atlas.scx", "new_batch.scx")

# Append cells from an AnnData object
new_adata = sc.read_h5ad("new_batch.h5ad")
pyscx.append_from_anndata("atlas.scx", new_adata)
```

`pyscx.append` uses a streaming SCX → SCX path: source shards are decoded
(or raw-copied when codec and encoding match) one at a time, so memory
usage is bounded by a single shard rather than the full source matrix.

Obs metadata is appended as new `ObsMetadataShard` sections — the
existing obs is not rewritten. Legacy single-section obs files are
promoted to shard 0 on first append, and new obs rows are added as
subsequent shards.

### Merging datasets

`pyscx.merge` combines multiple SCX files into one atlas-scale output.
All inputs must share the same var axis (gene set, order, and metadata).

```python
# Basic merge — var identity validated by default
pyscx.merge(["batch1.scx", "batch2.scx", "batch3.scx"], "atlas.scx")

# Then analyze the merged atlas
adata = pyscx.open("atlas.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.combat(adata, key="batch")  # batch correction
```

#### Full merge signature

```python
pyscx.merge(
    inputs,                              # list[str] — at least 2 SCX file paths
    output,                              # str — output SCX path
    index_obs=None,                      # list[str] — obs columns to index for query pushdown
    index_var=None,                      # list[str] — var columns to index
    index_preset=None,                   # "cellxgene" | "perturbseq" | "training"
    index_auto_threshold=None,           # int — auto-index cardinality threshold
    assume_identical_var=False,          # skip var identity validation
    assume_identical_obs=False,          # skip obs schema validation
    uns_policy=None,                     # "first" | "require-equal" | "namespace" | "summary"
)
```

#### Streaming behavior

Merge operates shard-by-shard at every level — **no full-dataset
materialization** at any point in the pipeline:

- **Obs metadata**: each input's obs is read one shard at a time and
  written as `ObsMetadataShard` sections in the output. Peak obs memory
  is bounded by one shard (~16K rows) rather than the total cell count.
  This removes the previous ~2 GB Arrow IPC narrow-offset ceiling.
- **X and layers**: CSR shards are decoded one at a time per input and
  re-encoded into output shards. Single-modality merge now matches the
  multimodal streaming pattern.
- **Obsm / varm**: dense mapping sections are read and written one shard
  at a time. Legacy single-section inputs are treated as one source shard.
- **Predicate indexes**: built incrementally from the obs shard stream
  without materializing a full obs table.

> [!TIP]
> For atlas-scale merges (>10M cells), merge's peak RSS is now dominated by
> the per-shard working set (~128 MB) rather than total obs/layer size.
> Merges that previously OOM'd or hit Arrow offset overflows at ~67M cells
> now complete with bounded memory.

#### Var identity validation

> [!IMPORTANT]
> **Breaking change**: `merge()` now validates var identity by default.
> Pre-existing code that merged files with different var metadata (but
> the same `n_vars`) will error. This prevents silent column-axis
> corruption where gene indices in later inputs are misinterpreted
> against the first input's var table.

By default (`assume_identical_var=False`), merge compares every input's
var batch column-by-column against input 0 and errors on mismatch. If
you have already validated var identity upstream:

```python
pyscx.merge(inputs, output, assume_identical_var=True)
```

Similarly, `assume_identical_obs=False` validates obs schema (column
names and dtypes) across inputs.

#### Uns conflict policy

The `uns_policy` kwarg controls how conflicting `uns` sections are
handled across inputs:

| Policy | Behavior |
|--------|----------|
| `"first"` (default) | Keep the first input's uns verbatim; warn on disagreement |
| `"require-equal"` | Error if any input's uns differs from the first |
| `"namespace"` | Wrap each input's uns under `"input_0"`, `"input_1"`, etc. |
| `"summary"` | Keep the first input's uns and record conflicts in `uns["_scx_uns_conflicts"]` |

```python
# Error if uns sections differ across inputs
pyscx.merge(inputs, output, uns_policy="require-equal")

# Namespace each input's uns to preserve all metadata
pyscx.merge(inputs, output, uns_policy="namespace")
```

## ML training data loading

For large-scale model training, SCX provides a high-performance data loader
that bypasses Python I/O entirely. Three dataset types cover different
ML patterns:

- **`TrainingDataset`** — sequential streaming for standard training loops
  (autoencoder, scVI, scGPT). 82× faster than TileDB-SOMA-ML on 1M cells.
- **`IndexPlanDataset`** — paired `(perturbed, control)` cell reads for
  perturbation training, contrastive learning, and donor-matched designs.
- **`MultimodalTrainingDataset`** — cell-aligned multi-assay batches
  (RNA + ADT + ATAC) from a single multimodal SCX file.

```python
import pyscx
import torch

dataset = pyscx.TrainingDataset(
    "atlas.scx",
    batch_size=1024,
    hvg_indices=hvg_array,   # decode only HVGs → less data
    normalize=True,
    log1p=True,
    obs_columns=["cell_type", "batch"],  # metadata in each batch
)

for batch in dataset:
    x = torch.from_numpy(batch["X"]).to(device)
    cell_types = batch["obs"]["cell_type"]  # {"codes": ndarray, "categories": list}
    # model.forward(), loss.backward(), ...

dataset.close()
```

The training loader uses a triple-buffered Rust pipeline (I/O → decode → GPU)
with zero Python on the hot path. For scVI, use the built-in
[`ScxDataModule`](api.md#scvi-integration-pyscxscx_integrationsscvi)
PyTorch Lightning DataModule.

See the dedicated [ML Training Guide](training.md) for end-to-end examples,
train/val split handling, PyTorch DataLoader compatibility, and migration
from h5ad-based training loops.

## File inspection

```python
exp = pyscx.open("experiment.scx")
print(exp)           # AnnData-style repr:
#   Experiment object with n_obs × n_vars = 10000 × 33694
#       obs: 'cell_type', 'sample'
#       var: 'gene_ids'
#       layers: 'raw_counts', 'spliced'
print(exp.n_obs)     # 10000
print(exp.n_vars)    # 33694
print(exp.nnz)       # 5234891
print(exp.obs_keys())  # ['cell_type', 'sample'] — callable, like adata.obs_keys()
print(exp.layer_names())  # ["raw_counts", "spliced"] — callable method too
print(exp.info())    # codec / shard / format-version internals

# Validate checksums
results = exp.validate()
for section, passed in results:
    print(f"  {section}: {'✓' if passed else '✗'}")
```

## Dependencies

```
pyscx          # SCX Python bindings
scanpy         # scverse analysis toolkit
anndata        # AnnData data structure
scipy          # sparse matrix support
numpy          # array support
pyarrow        # metadata transfer (Arrow IPC)
```

Install with:

```bash
uv pip install scanpy anndata scipy numpy pyarrow
cd pyscx && maturin develop --release
```
