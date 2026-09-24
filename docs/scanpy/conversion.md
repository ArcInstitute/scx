# Converting existing data to SCX

> Part of the [SCX + scanpy guide](README.md).

> **What round-trips?** For the canonical table of which AnnData fields are
> preserved, lossy, or dropped on conversion (and which warning fires for each),
> see [docs/api/conversion.md § Round-trip fidelity](../api/conversion.md#round-trip-fidelity).

> **Benchmarks**: for h5ad → SCX conversion throughput across datasets, codecs,
> and thread counts (including `full` mode that covers the h5ad read + SCX
> write), see [docs/performance/conversion.md §Conversion (h5ad → format)](../performance/conversion.md#conversion-h5ad--format)
> and [§Write Scaling (parallel shard encoding)](../performance/conversion.md#write-scaling-parallel-shard-encoding).

> **Migrating an existing h5ad workflow?**
> [docs/migrating-from-h5ad.md § Converting h5ad to SCX](../migrating-from-h5ad.md#converting-h5ad-to-scx)
> has end-to-end conversion recipes — predicate indexes, CSC sidecars,
> detection bitmaps, sort-on-convert, codec/shard tuning, memory budgets,
> and production-ready CLI + Python examples — all in one place.

## From AnnData / h5ad

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

### Sharded obs/var metadata

For datasets with `n_obs > shard_target_rows` (default 16,384),
`from_anndata` emits obs and var as sharded `ObsMetadataShard` /
`VarMetadataShard` sections rather than single monolithic sections.
This bounds metadata write memory and produces files compatible with the
streaming merge pipeline. Pass `force_legacy_metadata=True` to opt into
single-section metadata when needed for backward compatibility:

```python
pyscx.from_anndata(adata, "output.scx", force_legacy_metadata=True)
```

### Incremental mapping writes

Obsm, varm, obsp, and varp are extracted and written one key at a time —
the full set of mappings is never collected in memory simultaneously.
When a single mapping's estimated peak footprint exceeds `memory_budget`,
a `MappingPeakFootprintHigh` warning is emitted:

```python
pyscx.from_anndata(adata, "output.scx", memory_budget="4G")
```

## From h5ad on disk — streaming (`from_h5ad`)

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

`csc="always"` builds the sidecar in the same pass as X: the writer pushes
each streamed CSR shard into a `CscBuilder` and emits the CSC shards right
after X — no extra read of the output and no second copy on disk. Under
`memory_budget=`, the builder's buckets get a quarter of the budget (capped at
what the builder would stage alone) and ingest sizes itself against the rest.

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


## Backed AnnData — auto-streams via `from_anndata`

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

## From 10x HDF5

```python
pyscx.from_10x("filtered_feature_bc_matrix.h5", "dataset.scx")
```

`from_10x` reads through `scanpy.read_10x_h5` and hands the in-memory AnnData to
`from_anndata`, so peak memory scales with the whole matrix. That is fine for a
*filtered* matrix and wrong for a raw all-droplet one. For
`raw_feature_bc_matrix.h5` — the CellBender input, millions of droplets — use
the CLI, which streams by default and bounds peak memory to the resident
`indptr` plus one shard per outstanding worker:

```bash
scx convert --from 10x raw_feature_bc_matrix.h5 raw.scx
```

## From Cell Ranger MTX directory

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

## adata.raw

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
`to_anndata(raw=False)` opts out of raw entirely — no rebuild on the paths that would
rebuild it, and no `DroppedRaw` notice on the three that would drop it.
See [docs/api/conversion.md § `adata.raw`](../api/conversion.md#adataraw).

## Exporting back to MTX

```python
pyscx.to_mtx("dataset.scx", "/path/to/output_dir")
```

## Exporting back to h5ad / h5mu (`to_h5ad`, `to_h5mu`)

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
the CSR shards). `obs_mask=` is accepted in either obs row space, told apart by
length — `n_obs` entries (the live rows `read_obs()` describes; a pandas Series
with a labelled index is also checked for order) or `n_obs_physical` entries
(every physical row) — and both filters are ANDed with the deletion-vector mask
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
[docs/operations.md § CellBender import](../operations.md#cellbender-import).

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

## See also

- [Loading SCX data into AnnData](loading.md) — reading the converted file.
- [File operations](file-operations.md) — validating a file after write or
  transfer, appending, and merging.
- [Landing external annotations](external-annotations.md) — bringing tool
  output (doublet scores, CellBender counts) back onto an SCX file.
- [docs/migrating-from-h5ad.md](../migrating-from-h5ad.md) — conversion recipes
  and the round-trip fidelity table.
