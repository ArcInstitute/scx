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

- The on-disk `X` must be CSR or absent — CSC-on-disk and dense `X`
  are rejected with a clear error. Pre-convert upstream or use the
  non-streaming `from_anndata` path with enough RAM.
- `obsp` / `varp` are silently skipped (no on-disk readers yet — same
  gap the non-streaming CLI converter has had).

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
`h5ad → scx → h5ad` round-trips raw with integer counts bit-exact. Raw is dropped
(with a `DroppedRaw` warning) under obs-filtered `to_anndata`, backed mode, and
deletion-vector-active files; the in-memory `from_anndata(adata)` path does not yet
write raw. See [docs/api.md § `adata.raw`](api.md#adataraw).

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
"not (cell_type == 'NK cell')"
"(n_counts > 50) or (cell_type == 'NK cell')"
```

This subset uses comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`),
keyword-form boolean operators (`and`, `or`, `not`), the `in` operator
against a `[...]` list literal, and parenthesised sub-expressions. Tests in
`pyscx/tests/test_to_anndata_integration.py` (`test_obs_filter_grammar_parity_common_ground`)
assert that both paths select identical rows for the entries above.

**Divergences (work in one grammar only)** — the "pandas" column covers both
`backed=True` (`.query()`) and `preserve_slots=True` (`.eval()`):

| Expression | SCX engine | pandas (`.query()` / `.eval()`) |
|---|---|---|
| `n_counts > 50 & cell_type == 'T cell'` | ❌ parse error — use `and` | ✅ accepted as bitwise-and |
| `cell_type in ('T cell', 'B cell')` (tuple) | ❌ parse error — `in` requires `[...]` | ✅ accepted |
| `n_counts > 50 \| cell_type == 'NK cell'` | ❌ parse error — use `or` | ✅ accepted |
| Arithmetic on obs columns (e.g. `n_counts + n_genes > 100`) | ❌ not supported | ✅ accepted |
| String-method calls (e.g. `cell_type.str.startswith('T')`) | ❌ not supported | ✅ accepted |

When `preserve_slots=True` is used with an `obs_filter`, `to_anndata()` emits
a `UserWarning` noting that the filter was evaluated via pandas.eval — this
surfaces in notebook output so the grammar shift is visible without reading
this section.

**Recommendation:** write filters in the portable subset above. If a filter
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

### Indexing patterns

| Access | Behavior |
|--------|----------|
| `X[100:200]` | Row slice → decodes 1 shard |
| `X[[0, 5, 10]]` | Fancy index → decodes needed shards only |
| `X[mask]` | Boolean mask → decodes matching shards |
| `X[100:200, :500]` | Row slice + column filter → 1 shard + post-filter |
| `X[:, hvg_idx]` | Column-only → must decode all shards (CSR is row-major) |
| `X[0, 5]` | Scalar → returns `float` |

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
construction, UMAP embedding, Leiden clustering, and differential expression
via `pyscx.accel`. These accelerators are 2–40× faster than their scanpy
equivalents at scale (>100K cells) while writing results to the same AnnData
slots — so downstream scanpy functions (plotting, etc.) work identically.

All accelerators that support GPU expose a `device` parameter:
- `device="auto"` (default) — use GPU if available, fall back to CPU
- `device="cpu"` — force CPU
- `device="gpu"` — force GPU (raises error if unavailable)
- `device="gpu:1"` — select a specific GPU on multi-GPU systems

> **GPU is fastest only when the input layout matches the op.** For
> `pdex_ref` the column-major CSC-direct GPU route is the high-performance
> path, and it requires a *backed* SCX file with a CSC sidecar
> (`pyscx.from_anndata(..., csc="always")` / `scx convert --csc=always`); v3
> CSC-direct is the default GPU DE route. In-memory scipy CSR can run on GPU but
> may be slower than CPU. (`rank_genes_groups` / Wilcoxon shares the same
> CSC-direct (`gpu_csc_v3`) and CSR-direct (`gpu_csr_v3`) routes as `pdex_ref`;
> dense-host input is densified to CSR and also records `gpu_csr_v3`.) When comparing performance,
> **check the recorded route** at
> `adata.uns["scx_accel"][<op>]["route"]` (e.g. `gpu_csc_v3` vs `gpu_csr_v3`) —
> every DE call records which route it actually took and the `fallback_reason`
> if it didn't take the ideal one. See
> [docs/api.md § Accelerator route metadata](api.md#accelerator-route-metadata).

### Compatibility matrix

| Op                       | CPU | GPU | Scanpy-parity kwargs                                                  | scx-only kwargs                                |
|--------------------------|:---:|:---:|-----------------------------------------------------------------------|------------------------------------------------|
| `normalize_total`        | ✓   | ✓   | `target_sum`                                                          | `device`                                       |
| `log1p`                  | ✓   | ✓   | —                                                                     | `device`                                       |
| `calculate_qc_metrics`   | ✓   | —   | `qc_vars`, `log1p`, `inplace`                                         | `prefer_format`                                |
| `highly_variable_genes`  | ✓   | ✓   | `n_top_genes`, `flavor`, `batch_key`, `span`, `subset`, `n_bins`, `layer` | `device`, `prefer_format`                  |
| `pca`                    | ✓   | ✓   | `n_comps`, `zero_center`, `random_state`                              | `device`, `method`, `qr_method`, `prefer_format`, `allow_tf32` |
| `neighbors`              | ✓   | ✓   | `n_neighbors`, `use_rep`, `random_state`                              | `device`, `ef_construction`, `ef_search`       |
| `pca_neighbors`          | ✓   | ✓   | (PCA + neighbors kwargs, see below)                                   | `device`, `method`, `qr_method`, `prefer_format` |
| `pca_neighbors_umap`     | ✓   | ✓   | (PCA + neighbors + UMAP kwargs, see below)                            | `device`, `method`, `qr_method`, `prefer_format` |
| `umap`                   | ✓   | ✓   | `n_components`, `n_epochs`, `min_dist`, `spread`, `learning_rate`, `random_state` | `device`                           |
| `leiden`                 | ✓   | ✓¹  | `resolution`, `key_added`, `random_state`, `n_iterations`             | `device`, `parallel`, `theta`                  |
| `harmony_integrate`      | ✓   | —   | `key`, `basis`, `theta`, `sigma`, `lamb`, `max_iter`                  | `adjusted_basis`, `block_size`                 |
| `compute_lisi`           | ✓   | —   | `key`, `basis`, `perplexity`, `n_neighbors`                           | —                                              |
| `rank_genes_groups`      | ✓   | —   | `groupby`, `reference`, `n_genes`, `method`                           | `gene_chunk_size`, `stratify_by`, `prefer_format` |
| `pseudobulk_dex`         | ✓   | —   | `groupby`, `design`, `reference`                                      | `test_col`, `aggr_method`, `stratify_by`, `prefer_format` |

¹ GPU Leiden has a documented label-stability divergence vs `leidenalg` —
pin `device="cpu"` to preserve label stability for downstream DE / annotation
transfer. See `CLAUDE.md § Known Limitations`.

### `prefer_format="csr"|"csc"`: explicit column-major dispatch

A subset of accelerators take a `prefer_format` kwarg that selects
between the row-major CSR path (default) and the column-major CSC
sidecar path. Entries that accept it:

| Function | CSC win |
|----------|---------|
| `pyscx.accel.highly_variable_genes` | Single-batch seurat_v3 only — single-pass per-column accumulators with no `O(n_vars)` row-wise scratch. Multi-batch and non-seurat_v3 raise. |
| `pyscx.accel.rank_genes_groups` | Per gene chunk: read CSC slab + scatter into row-major dense buffer (vs decode every row + project for CSR). Clearest CSC win. |
| `pyscx.accel.pseudobulk_dex` | Filtered-gene subsets only (`gene_indices=...` or column projection on `adata.X`). Full-gene pseudobulk has no CSC win and raises. |
| `pyscx.accel.calculate_qc_metrics` | Gene-axis aggregations only (`total_counts`, `n_cells_by_counts`); cell-axis stays CSR. |
| `pyscx.accel.col_sums` / `col_nnz` / `col_min` / `col_max` / `col_var` | Per-column aggregations on `ScxBackedSparseDataset` / `ScxLazyTransformedDataset`. |
| `pyscx.accel.pca` | **Rejects `prefer_format="csc"`** with `ValueError`. Covariance build and randomized SpMM are row-major; CSC offers no measurable speed-up. |

**Default is `"csr"` everywhere.** No `"auto"` — the runtime can't
guess whether CSC dispatch is safe (depends on the file having a
sidecar AND the user's transform chain being column-local). No
thread-local default. No env-var override. Each call sets the
choice locally.

`prefer_format="csc"` requires *all* of the following; otherwise it
raises `RuntimeError` with a message naming the missing capability:

1. The file has a CSC sidecar (`pyscx.from_anndata(csc="always"|"auto")`,
   `scx convert --csc=always|auto`, or `scx build-csc`).
2. The transform chain on `adata.X` contains only column-local
   operations. `Log1p` is column-local; `NormalizeTotal` and
   `RowScale` are not (per-row state). The common `normalize_total →
   log1p` chain is *not* column-local — use `prefer_format="csr"`.
3. No active row deletion vector. After
   `pyscx.accel.filter_cells()` or `pyscx.accel.subset_obs()`, the
   dataset has `kept_to_global` set; CSC dispatch then raises until
   you `materialize()` or rebuild the file.

Invalid values (e.g. `"auto"`, `"CSC"`) raise `ValueError`.

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
- **Wilcoxon (`rank_genes_groups`) takes the same v3 routes as `pdex_ref`.** With a
  CSC sidecar it runs CSC-direct (`gpu_csc_v3`); without one it runs CSR-direct
  (`gpu_csr_v3`, `fallback_reason == "no_csc_sidecar"`) — GPU-supported but not
  GPU-fast. So a CSC sidecar makes Wilcoxon GPU-fast too.
- **PCA / kNN / UMAP / Leiden are not column algorithms** — they operate on
  row-major `X` or on PCA embeddings / kNN graphs, so CSC does not apply.

To make a file GPU-fast for DE, build the sidecar at conversion time:
`pyscx.from_anndata(adata, path, csc="auto")` (built automatically once the
dataset clears the size thresholds) or `csc="always"`, `scx convert --csc=auto`,
or `scx build-csc` after the fact. Then **always confirm the route actually
taken** via `adata.uns["scx_accel"][op]["route"]` before drawing performance
conclusions — a silent CSR fallback measured as "GPU DE" is exactly the
benchmarking trap the [route metadata](api.md#accelerator-route-metadata) exists
to catch.

**Benchmark route gates.** The benchmark suite enforces correct GPU dispatch
via absolute-floor gates in `thresholds.yaml`. Every `accel_*.py` GPU variant
emits an `<op>_route_gpu_correct` signal (1.0 when a GPU route ran, 0.0 on a
silent CPU fallback); `bench_csc_dispatch.py` emits `csc_dispatch_correct` for
CSC-labelled variants; and `accel_de.py` emits `de_route_csc_direct` for the
pdex_ref CSC-direct path. See
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
  (Delta-Golomb / FOR-BP / Rice) — the one codec with a decode-metadata sidecar
  ([format.md §4.2](format.md#42-decode-metadata-sidecar)) and GPU decode
  kernels, so it is the codec the device-side decode path targets. Open the
  counts and run `normalize_total` / `log1p` in VRAM (the
  `ScxLazyTransformedDataset` chain, or `rsc.pp.*` on the device AnnData) so the
  log-normalized matrix is produced on the GPU and **never round-trips through a
  host float buffer**. This is the recommended flow.
- **A *persisted* log-normalized `X` is float → Pcodec, which decodes on the
  host.** Public h5ad / CELLxGENE files often ship `X` already log-normalized.
  That matrix has **no decode sidecar and no GPU decoder** (Pcodec is CPU-only),
  so the handoff pays a host pcodec-decompress + HtoD per shard — *GPU-supported,
  but not GPU-fast on the decode side*. If the file also carries a `counts`
  layer, prefer opening that and deriving log-norm on-device (bullet above).
- **If you must persist log-norm for the GPU and want decode-free upload, store
  it uncompressed (`None` codec → raw `f32` / `f16`).** A raw float array needs
  no decode — the handoff is a straight HtoD `memcpy` — at the cost of
  compression ratio (`f16` halves the bytes if its precision is acceptable for
  log-norm). This is the only float option that is GPU-fast to upload today; a
  GPU-decodable *compressed* float codec does not exist.
- **The decode sidecar is Scx1-only.** Zstd / Pcodec / LZ4 shards and float
  layers carry no sidecar; the device decode path falls back to host decode +
  HtoD for them. "Make `X` GPU-fast to decode" therefore means "store the
  GPU-relevant matrix as Scx1 counts," **not** "add a sidecar to a float layer."
- **Upgrade a sidecar-less file in place with `scx optimize`.** A pre-v3 or
  reconverted-without-sidecars file (Scx1 counts but no `decode/*` sections) does
  not need a full reconvert to become device-decode-fast — run
  `scx optimize in.scx --output out.scx`. It re-encodes + canonicalizes every CSR
  shard so decode sidecars are emitted and the file is stamped `format_version=3`,
  preserving rows / obs / var / obsm / uns / indexes (see
  [operations.md § Optimize](operations.md#optimize)). Only Scx1 integer shards
  gain a sidecar — a persisted float (Pcodec) `X` still won't (store counts per
  the first bullet).

Confirm the path actually taken via
`adata.uns["scx_accel"][op]["transfer_mode"]`, the same way you confirm `route`
for DE. `to_gpu_anndata` stamps one of:
- `scx_device_decode_gpu` — Scx1 shards decoded **fully in VRAM** from the decode
  sidecar; only the tiny indptr is uploaded (`bytes_uploaded` ≈ indptr). The fast
  path you want. As of the BitPacker4x GPU kernel, this covers **every** Scx1 row,
  including dense (≥128-nnz) cells — those no longer host-fall-back.
- `scx_device_handoff_streamed` — on-device, but some shard still bounced through
  the host because it is **not** an Scx1 sidecar shard: a non-Scx1 codec (the float
  Pcodec case above) or a sidecar-less Scx1 shard (e.g. byte-passthrough output).
  `bytes_uploaded` is the real HtoD total.
- `scx_device_handoff` — host-assembled CSR (filtered / projected / multimodal
  input), or an `X` that was already device-resident on entry.

(rapids ops that have to upload a host `X` instead stamp `anndata_to_gpu` — a host
re-upload, not a `to_gpu_anndata` mode.)

**rapids-singlecell does not read the decode sidecar.** The sidecar lives
entirely on the SCX side of the handoff: it accelerates SCX's own
decode→device step (`to_gpu_anndata`, the
[format.md §4.2](format.md#42-decode-metadata-sidecar) sidecar consumer), which
*produces* the `cupyx.scipy.sparse.csr_matrix` that rapids then operates on.
rapids only ever sees that already-decoded, device-resident matrix (it validates
inputs via its own `_check_gpu_X`) and has no knowledge of the SCX format,
codecs, or sidecars. So choosing an Scx1-counts layout speeds up the
SCX→device handoff that *feeds* rapids — it is not something rapids consumes, and
it changes no rapids call.

### PCA (`pyscx.accel.pca`)

Two methods, auto-routed by the number of variables:

- **Covariance PCA** (CPU: n_vars ≤ 5,000): Builds the covariance matrix
  `X^T @ X` directly from CSR nonzeros via sparse outer product accumulation
  (exploiting symmetry), then eigendecomposes. Exact results, faster than
  randomized SVD for HVG-selected data. Parallel accumulation via rayon
  thread-local matrices. CPU-only; the former native GPU covariance path
  (`cusolverDnSsyevd`) was removed in Phase 3.2.
- **Randomized SVD** (CPU: n_vars > 5,000): Streaming shard-by-shard SpMM
  with zero-copy `MatRef::from_row_major_slice` views. Skips intermediate QR
  on transpose results for n_power_iterations ≤ 2 (matching sklearn's default).
- **GPU PCA** routes to `rapids_singlecell` (`rsc.pp.pca`) via
  `to_gpu_anndata()`. The data is handed off as a GPU-resident AnnData with
  `cupyx.scipy.sparse.csr_matrix` X — no host round-trip. The streaming/
  randomized CPU PCA path (>VRAM datasets) survives natively as a fallback.

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
| `random_state` | 0 | Random seed (used by randomized SVD; covariance method is deterministic) |
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
~30 MB covariance matrix for 2K genes).

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
| `n_iterations` | 2 | Outer iterations: 2 matches the leidenalg package default; raise to e.g. 100 on the cuGraph path for tighter modularity convergence (rapids-singlecell's default). |
| `parallel` | `False` | Run the **Rust-native** Leiden in conflict-free batched mode. `False` (default) matches C++ leidenalg sequential moving. **Ignored on the cuGraph path** (warns when `True`). |
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

# Default: overwrite adata.obsm["X_pca"] with the corrected embedding.
pyscx.accel.harmony_integrate(adata, "batch")

# Or keep the raw PCA and write the corrected embedding to a new obsm key:
pyscx.accel.harmony_integrate(
    adata, "batch", adjusted_basis="X_pca_harmony"
)

# Multi-covariate integration (e.g., donor + assay):
pyscx.accel.harmony_integrate(adata, ["donor_id", "assay"])

# Downstream scanpy works on the corrected embedding just like raw PCA:
pyscx.accel.neighbors(adata, use_rep="X_pca")
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `key` | (required) | `obs` column name, or list of column names, for the batch covariate(s). Each is factorised via `pandas.factorize(sort=False)`. |
| `basis` | `"X_pca"` | `obsm` key holding the input embedding. |
| `adjusted_basis` | `None` | `obsm` key for the corrected embedding. `None` overwrites `basis` in place (scanpy-compatible default). |
| `n_clusters` | `None` | Soft cluster count K. `None` → `min(N/30, 100)`, clamped to `[2, N/2]`. |
| `theta` | `2.0` | Diversity-penalty strength. Scalar broadcasts to every covariate. |
| `sigma` | `0.1` | Gaussian bandwidth for soft assignments. |
| `lamb` | `None` | Ridge penalty. `None` enables dynamic estimation (`alpha × E[k,b]`). |
| `max_iter` | `10` | Maximum Harmony outer iterations (cluster → correct rounds). |
| `max_iter_kmeans` | `4` | Maximum k-means sub-iterations per Harmony iter. |
| `random_state` | `0` | RNG seed (`ChaCha8Rng` for determinism across runs). |
| `device` | `"auto"` | `"cpu"` / `"gpu"` / `"auto"`. GPU path requires pyscx built with `--features gpu`. |

Results:

- `adata.obsm[adjusted_basis or basis]` — corrected embedding (N × d, f32).
- `adata.uns["harmony"]` — dict with `params`, `converged`, `n_iterations`,
  `objective_harmony` (per-iteration objective curve), and `backend`
  (`"scx-accel-cpu"` or `"scx-gpu"`).

**Numerical parity** against R `harmony` v2.x on the validation fixtures
in `benchmarks/results/harmony/reference/`: mean per-PC Pearson r is
0.989–0.999. Rust uses `rand_chacha` while R uses Mersenne Twister, so
tail PCs can drift by a few percent on high-batch-count inputs — see
[`docs/performance.md`](performance.md#harmony2-batch-integration--lisi)
and `pyscx/tests/test_harmony_validation.py`.

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
| `n_neighbors` | `None` | k for the exact kNN graph. `None` → `ceil(3 × perplexity)`. |

Returns a `numpy.ndarray` of length N and also writes the values to
`adata.obs[f"lisi_{key}"]`.

The implementation uses an exact brute-force kNN (per-row squared-norm
expansion + per-cell top-k heap) to stay numerically in lockstep with
the R `lisi` reference. On D1–D4 it is **~10× faster** than
R `lisi::compute_lisi` with mean-LISI agreement within 0.8–2.4 %.
Brute-force kNN is O(N²·d); at census scale (D5+) you'd want to pair
this with an HNSW-approximate kNN step instead.

### Differential Expression (`pyscx.accel.rank_genes_groups`)

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
| `groupby` | (required) | Column in `adata.obs` to group cells by |
| `reference` | `"rest"` | Compare against a specific group or `"rest"` (1-vs-rest) |
| `n_genes` | all | Number of top genes to report per group |
| `method` | `"wilcoxon"` | Statistical method (currently only `"wilcoxon"`) |
| `rankby_abs` | `False` | Sort genes by absolute z-score instead of signed score. `False` (default) matches scanpy's default: highest positive z-score first. `True` ranks by significance regardless of direction. |

Benchmarked at 5.4s on 1M cells (3.2× faster than scanpy's 17.2s).

### Pseudobulk Differential Expression (`pyscx.accel.pseudobulk_dex`)

Streaming pseudobulk aggregation in Rust + negative binomial GLM testing via
[pydeseq2](https://pydeseq2.readthedocs.io/). Designed for perturbation
sequencing (Perturb-seq) experiments with biological replicates.

The aggregation phase streams shards from `BackedCsrReader` without
materializing the full matrix — peak memory is one shard plus the pseudobulk
count matrix (n_groups × n_vars).

```python
import pyscx

adata = pyscx.open("perturb_seq.scx").to_anndata(backed=True)

# Run pseudobulk DE: drug vs control, grouped by (perturbation, donor)
result = pyscx.accel.pseudobulk_dex(
    adata,
    groupby=["perturbation", "donor"],
    test_col="perturbation",
    reference="control",
)

# result is a pandas DataFrame:
#   gene | baseMean | log2FoldChange | lfcSE | stat | pvalue | padj | target | reference
print(result.sort_values("padj").head(20))
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groupby` | (required) | List of obs columns to group cells by |
| `test_col` | (required) | Column in `groupby` containing the condition variable |
| `reference` | (required) | Reference level in `test_col` (e.g., `"control"`) |
| `design` | `"~ test_col"` | DESeq2 design formula (auto-generated if not specified) |
| `aggr_method` | `"sum"` | Aggregation method: `"sum"` or `"mean"` |
| `min_cells_per_group` | 10 | Groups with fewer cells are excluded |

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

# Pseudobulk DE stratified by cell type:
result = pyscx.accel.pseudobulk_dex(
    adata,
    groupby=["perturbation", "donor"],
    test_col="perturbation",
    reference="control",
    stratify_by=["cell_type"],
    min_cells_per_stratum=50,
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
> `pydeseq2` is an **optional** runtime dependency. Install with
> `pip install pydeseq2` before calling `pseudobulk_dex()`.

### Perturbation evaluation metrics (cell-eval / arc-bench parity)

SCX ships Rust-accelerated equivalents of the metrics in
[`cell-eval`](https://github.com/arcinstitute/cell-eval) and
[`arc-bench`](https://github.com/arcinstitute/arc-bench). The outputs are
numerically equivalent to the Python references within the tolerances
below (30/30 parity tests in `pyscx/tests/test_cell_eval_parity.py` pass),
so an existing cell-eval pipeline can swap in `pyscx.accel.*` for 10–20×
wall-clock speedup at census-scale perturbation datasets (see
[`docs/performance.md`](performance.md#perturbation-metrics-cell-eval--arc-bench-parity)
for numbers at 10K / 100K / 500K / 1M cells).

| Metric | Tolerance | Rationale |
|---|---|---|
| AMI / NMI / ARI on label vectors | `atol=1e-10` | Integer-label inputs; limited by double-precision floor (~2.2e-16). |
| pseudobulk_means, pearson_delta, mse/mae (and `_delta` variants), knockdown_efficiency, log_deviation | `atol=1e-6` | f32 CSR promoted to f64 before accumulation; expected rounding `O(n_cells · 2⁻²³) ≈ 1e-7` at 1M cells. |
| energy_distance / pearson_edistance | `atol=1e-4` correlation, `atol=1e-3` per-pert | O(N²) pairwise reduction; faer-gemm reduction order differs from sklearn BLAS GEMM. f32 + gemm matches f64 + scalar within these bounds (test parametrised over both dtypes). |
| clustering_agreement (AMI over Leiden sweep) | `atol=0.05` per-resolution, `atol=0.15` aggregate | Native-Rust kNN (HNSW) + Leiden replaces scanpy under the hood; the two algorithms produce within-permutation labels on graphs with `n_perts ≥ 16` (parity test scaffold uses `n_perts=30`). |
| discrimination_score rank | exact (`abs=0`) | Integer rank computation; any non-zero diff is a correctness regression. |

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

With `exclude_target_gene=True` (default), the gene matching each
perturbation's name is dropped from the distance — prevents trivially high
scores from knockdown-gene dominance and matches cell-eval's default.

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

SCX's implementation avoids materializing the `[N, N]` distance matrix per
perturbation (streaming accumulation of per-row sums even on the gemm
path), precomputes control self-distance once, and parallelizes across
perturbations with rayon. At 20K cells × 2K genes × 50 perturbations the
default `gemm + f32` path is **52×** faster than cell-eval's
`sklearn.metrics.pairwise_distances`; above ~500K the reference becomes
infeasible while SCX remains usable.

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
ari = pyscx.accel.adjusted_rand_index(labels_a, labels_b)  # rescaled
```

#### DE result format bridge (`pyscx.accel.rank_genes_groups_df`)

Same computation as `rank_genes_groups()` but returns a **polars DataFrame**
in cell-eval's `DEResults` schema — ready to feed into
`cell_eval.initialize_de_comparison()` and `MetricPipeline(profile="de")`:

```python
df = pyscx.accel.rank_genes_groups_df(
    adata, "perturbation", reference="control",
)
# Columns: target, feature, fold_change, p_value, fdr,
#          log2_fold_change, abs_log2_fold_change
```

Useful when you want SCX's faster Wilcoxon but cell-eval's DE metrics
downstream (overlap@N, precision@N, pr_auc, etc.).


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
| **PCA precision** | GPU uses f32 throughout; CPU uses f64 | Cosine similarity per PC > 0.99 — no biological impact |
| **kNN algorithm** | GPU uses CAGRA (graph-based ANN); CPU uses HNSW | Both are approximate; recall@k > 0.95 |
| **UMAP non-determinism** | GPU uses `atomicAdd` (race conditions are intentional) | Embedding coordinates differ; cluster structure preserved |
| **Leiden** | Rust-native vs cuGraph vs leidenalg may produce different partitions | ARI > 0.90; biological conclusions equivalent |

### Tolerance thresholds (correctness tests)

The GPU test suite (`pyscx/tests/test_accel_gpu.py`) enforces these thresholds vs the CPU reference:

| Test | Metric | Threshold | Notes |
|------|--------|-----------|-------|
| GPU PCA vs CPU PCA | Cosine similarity per PC | > 0.99 | Sign-invariant; GPU=f32, CPU=f64 |
| GPU kNN vs CPU HNSW | Recall@k | > 0.95 | Different algorithms (rapids vs HNSW); exact match not expected |
| GPU UMAP | Trustworthiness | > 0.95 | Non-deterministic (rapids_singlecell GPU UMAP) |
| GPU Leiden vs CPU Leiden | ARI | > 0.90 | Graph partitioning is inherently non-deterministic |
| GPU normalize + log1p | Element-wise | rtol=1e-7 | Possible f32 rounding differences vs CPU |

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
Leiden uses cuGraph. The Leiden dispatch order is: Rust-native CPU →
cuGraph GPU → Python leidenalg. Set `SCX_FORCE_NATIVE_GPU=1` to pin
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
| `pyscx.accel.pca` (CPU) | Rayon | Parallel covariance accumulation (thread-local matrices); streaming SpMM parallelizes inner products |
| `pyscx.accel.neighbors` (CPU) | Rayon | Parallel kNN queries on HNSW index |
| `pyscx.accel.umap` (CPU) | Single-threaded SGD | Edge updates are serial on CPU; GPU path uses CUDA kernel parallelism |
| `pyscx.accel.leiden` | Opt-in rayon via `parallel=True` | Sequential by default (matches C++ leidenalg); parallel uses conflict-free graph coloring |
| `pyscx.accel.rank_genes_groups` | Rayon | Parallel Wilcoxon rank-sum across genes |
| `pyscx.accel.pseudobulk_dex` | Rayon (aggregation) | Streaming aggregation is parallel; downstream `pydeseq2` testing runs single-threaded |
| `pyscx.accel.highly_variable_genes` | Rayon (via streaming reader) | Parallelism comes from shard decode; the mean/var reduction itself is serial |
| `pyscx.accel.perturbation_metrics`, `discrimination_score`, `energy_distance` | Rayon | Parallelizes across perturbations / pairwise-distance rows |
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

sc.tl.rank_genes_groups(adata, groupby="cell_type", method="wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=20)
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
pyscx.accel.harmony_integrate(adata, "batch")  # overwrites obsm["X_pca"]

# Use corrected embedding for downstream:
pyscx.accel.neighbors(adata, use_rep="X_pca")
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)

# QC: batch mixing before / after
adata.obsm["X_pca_raw"] = adata.obsm["X_pca"]  # (not actually raw after overwrite — use adjusted_basis if you need both)
lisi = pyscx.accel.compute_lisi(adata, "batch")
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
