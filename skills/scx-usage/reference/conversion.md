# Conversion reference (pyscx + scx-cli)

Self-contained reference for getting data into and out of SCX. All h5ad/h5mu
ingest and export entry points stream by default (`stream=True`); peak RSS is
bounded by one shard's worth of CSR per matrix regardless of file size. Pass
`stream=False` to opt into the legacy materializing path.

## Ingest into SCX

### `pyscx.from_h5ad(path, out, ...)` — stream an h5ad directly to SCX
Reads obs/var/uns via pure-Rust HDF5 (bypasses `anndata.read_h5ad` entirely), so
it never pays anndata's eager `obsm` allocation. Recommended for files larger
than RAM. Full kwargs:

```
from_h5ad(path, out, codec=None, shard_size=None, csc="off",
          csc_cols_per_shard=5000, uns_format="tagged", stream=True,
          strict_uns=False, dense_zero_epsilon=0.0, memory_budget=None,
          temp_dir=None, index_obs=None, index_var=None, index_preset=None,
          index_auto_threshold=1000, bitmap="off", reader_threads=None,
          writer_queue_depth=4, obs_override=None, var_override=None,
          uns_override=None)
```

- **Source layout** auto-detected: CSR streams natively; dense `/X` streams via
  row-slab sparsification (`dense_zero_epsilon` thresholds near-zero values,
  default `0.0` matches `scipy.csr_matrix(dense)`); CSC-on-disk uses an
  in-memory transpose when it fits `memory_budget`, else an external bucketed
  transpose to `temp_dir`.
- `obs_override` / `var_override` (pandas DataFrame) / `uns_override` (dict):
  use in place of on-disk values for read-mutate-write flows that add
  annotations without the `obsm` allocation. `obs_override.shape[0]` must equal
  on-disk n_obs; `var_override.shape[0]` must equal n_vars. `uns_override`
  replaces the entire uns section (not a merge). Any override with
  `stream=False` raises `ValueError`. `obsm`/`varm`/`obsp`/`varp` are
  intentionally **not** override-able (would re-introduce the OOM class this
  API avoids).
- `strict_uns=True` raises on the first unrepresentable uns entry; default
  `False` emits a `UserWarning` per skipped key. Other structured warnings:
  `InferredEncoding`, `DenseSparsified`, `DuplicateCoordinatesMerged`.
- `reader_threads`: parallel streaming reader. `None` → `RAYON_NUM_THREADS` or
  `os.cpu_count()`; `1` forces sequential; `>1` requests rayon workers
  (byte-identical output). Requires a thread-safe libhdf5 (conda-forge default);
  a static libhdf5 falls back to sequential with a one-shot `Hdf5NotThreadsafe`
  warning. `memory_budget` derates the granted count.
- `writer_queue_depth` (default 4): backpressure window; outstanding shards
  capped at `reader_threads + writer_queue_depth`.

### `pyscx.from_anndata(adata, path, ...)` — write an in-memory (or backed) AnnData
```
from_anndata(adata, path, codec=None, shard_size=None, in_place=False,
             csc="off", csc_cols_per_shard=5000, uns_format="tagged",
             index_obs=None, index_var=None, index_preset=None,
             index_auto_threshold=1000, bitmap="off",
             force_legacy_metadata=False, memory_budget=None,
             shard_target_rows=None)
```
Persists `X`, `obs`, `var`, `layers`, `obsm`, `varm`, `uns`, and the sparse
pairwise `obsp`/`varp` (stored as float32 COO Arrow IPC; higher precision is
downcast). Accepts a backed AnnData (`sc.read_h5ad(p, backed='r')`) and
auto-routes to the streaming converter. Also streams **SCX → SCX** when
`adata.X` is an `ScxBackedSparseDataset` / `ScxLazyTransformedDataset`
(byte-passthrough fast path when shard layouts match; decode-encode otherwise;
lazy `X` always decode-encodes with transforms applied per shard).
`force_legacy_metadata=True` forces single `ObsMetadata`/`VarMetadata` sections;
default emits sharded sections when `n_obs > shard_target_rows`.

### `pyscx.read_h5ad_metadata(path, strict_uns=False) -> H5adMetadata`
Reads just obs/var/uns + X shape via pure-Rust HDF5 (no anndata, no `obsm`
allocation). Returns attributes `obs` (DataFrame), `var` (DataFrame), `uns`
(dict), `n_obs`, `n_vars`, `x_format` (`"csr"`/`"csc"`/`"dense"`). Pair with
`from_h5ad(..., obs_override=, uns_override=)` for cheap read-mutate-write.

### Other ingest
- `pyscx.from_h5mu(path, out, codec=None, shard_size=None, csc="off", csc_cols_per_shard=5000, stream=True, strict_uns=False, memory_budget=None, temp_dir=None, modalities=None, modality_types=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4)` — multimodal SCX v2. `modalities=["rna","adt"]` keeps a subset (unknown names raise); `modality_types={"adt":"protein","peaks":"atac"}` overrides inferred types (others emit `ModalityTypeInferred`). Per-modality dispatch mirrors `from_h5ad`.
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None, in_place=False, csc="off", csc_cols_per_shard=5000, uns_format="tagged", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off")`.
- `pyscx.from_mtx(mtx_dir, scx_path, codec=None, shard_size=None)` — Cell Ranger MTX dir (`matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, `features.tsv[.gz]`). Default shard size 16384.

## Export out of SCX

- `pyscx.to_h5ad(path_or_experiment, out, stream=True, modality=None, reader_threads=None, writer_queue_depth=4, memory_budget=None)` — **free function**, not a method; source may be a path or a `pyscx.Experiment`. Mirror of `from_h5ad`. Deletion vectors honored (only kept rows written; `shape[0] = n_obs - n_deleted`). For a multimodal SCX file pass `modality="rna"` to extract one modality, else it raises. `reader_threads` here does **not** require a thread-safe libhdf5 (HDF5 writes stay on the calling thread).
- `pyscx.to_h5mu(path, out, stream=True, reader_threads=None, writer_queue_depth=4, memory_budget=None)` — requires `reader.is_multimodal()`; single-modality files raise (use `to_h5ad`).
- `pyscx.to_mtx(scx_path, output_dir)` — Cell Ranger–style MTX (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`).

> `to_h5ad(exp, ...)` re-exports the **unmodified on-disk** file. To save an
> in-memory analysis, use `adata.write_h5ad(...)` or `from_anndata(adata, "out.scx")`.

## Shared kwarg semantics

- **`codec`**: `"auto"` (default, per-shard selection by median value), `"scx1"`
  (Delta-Golomb + FOR-BP + Rice; integer-only), `"zstd"`, `"pcodec"` (best for
  float layers), `"lz4"`, `"none"`.
- **`csc`**: `"off"` (default) or `"always"`. `"always"` writes a column-major
  sidecar via a two-pass write (streaming CSR → in-place rebuild); transient
  disk briefly reaches ~2× the output size. Required to use the
  `prefer_format="csc"` accel paths and the v3-CSC GPU DE driver.
- **`index_obs` / `index_var` / `index_preset` / `index_auto_threshold`**:
  materialize predicate indexes at write time. `index_preset` expands curated
  column lists: `"cellxgene"` (cell_type / disease / tissue / assay / donor_id /
  development_stage / sex / suspension_type), `"perturbseq"`, `"training"`.
  **Without an index, `query().filter_obs(...)` silently regresses to a full obs
  scan.** Set these whenever the file will be queried. The CLI `--index-*` flags
  only build indexes on SCX-writing ingest (`h5ad → scx`, `10x → scx`; `h5mu → scx`
  accepts and skips with a warning); on `mtx → scx` and the SCX-export directions
  (`scx → h5ad/h5mu/mtx`) they are a **hard error**, not a silent no-op. Rebuild an
  index on an existing SCX file with `scx compact` / `scx append` / `scx merge`.
- **`bitmap`**: `"off"` / `"auto"` / `"always"` — per-shard gene→row detection
  bitmaps consumed by `Experiment.detection_counts()` / `cells_expressing()`.
- **`memory_budget`**: a bare byte count or a binary-prefixed size — `K`/`M`/`G`/`T`
  or `KiB`/`MiB`/`GiB`/`TiB` (all powers of 1024); decimal `KB`/`MB`/`GB`/`TB` is
  **rejected** to avoid 1000-vs-1024 ambiguity. E.g. `"4G"` / `"512M"` / `"2GiB"`.
  Caps dense slabs and CSC external-transpose buffers; derates `reader_threads`.
  The same parser backs the CLI `--memory-budget` and `scx build-csc --memory-limit`.
- **`uns_format`**: `"tagged"` (default) wraps NumPy/pandas containers in
  `__scx_type__` envelopes for bit-exact round-trip; `"plain"` collapses to JSON
  primitives (raises on NaN/Inf in arrays). The reader auto-detects per value,
  so older/plain files read identically on a modern build. Unsupported in both:
  `bytes`, and `datetime64`/`complex`/`timedelta` ndarray dtypes.

## File ops (Python)

- `pyscx.append(target, input, codec=None, shard_size=None, index_*=...)` — streaming append from an SCX file (raw-copy fast path when codec/encoding match).
- `pyscx.append_from_anndata(target, adata, ...)`.
- `pyscx.mark_deleted(path, cell_indices)` — logical deletion (deletion vector).
- `pyscx.compact(input, output, index_*=...)` — rewrite reclaiming deleted space.
- `pyscx.rollback(path, to_seq=None)`.
- `pyscx.merge(inputs, output, index_*=..., assume_identical_var=False, uns_policy="first", shard_target_rows=None)` — streams obs shard-by-shard. `assume_identical_var=False` validates var index/columns/values across inputs (`True` checks only `n_vars`). `uns_policy`: `"first"` / `"require_equal"` / `"namespace"` / `"summary"`. Pass `index_*` or pushdown regresses to a full scan on the merged file.

## CLI (`scx`)

```
scx convert <input> <output> [--from h5ad|10x|h5mu|mtx|scx] [--to h5ad|h5mu|scx]
    [--codec auto|none|scx1|zstd|lz4|pcodec] [--shard-size N]
    [--stream[=true|false]] [--csc off|always] [--csc-cols-per-shard N]
    [--modality NAME] [--memory-budget SIZE] [--strict-uns]
    [--dense-zero-epsilon F] [--temp-dir DIR] [--modalities CSV]
    [--modality-types NAME:TYPE,...] [--index-obs CSV] [--index-var CSV]
    [--index-preset cellxgene|perturbseq|training] [--index-auto-threshold N]
    [--bitmap off|auto|always] [--reader-threads N] [--writer-queue-depth N]
```
- `--stream` (default true) bounds peak memory; supported h5ad ↔ SCX and h5mu ↔ SCX both directions.
- On ingest, `--csc always` does the two-pass CSR-then-rebuild write.
- For multimodal SCX → h5ad, combine `--to h5ad --modality NAME`.

Other commands:
- `scx info <file> [--json] [--history]` — metadata; multimodal shows a per-modality table with a `has_csc` column.
- `scx validate <file> [--verbose]` — verifies BLAKE3 checksums section-by-section.
- `scx subset <input> [--output P] [--filter EXPR] [--genes PATH] [--dry-run] [--shard-size N] [--codec ...]`.
- `scx append <target> --input <source> [--codec ...] [--shard-size N] [--index-* ...]`.
- `scx delete <file> --filter <expr> [--dry-run]`.
- `scx compact <input> --output <path> [--force] [--index-* ...]`.
- `scx merge <f1> <f2> [...] --output <path> [--index-* ...]`.
- `scx rollback <file> [--to-seq N]`.
- `scx build-csc <input> <output> [--memory-limit 4G] [--force]` — `--memory-limit` takes the same size forms as `--memory-budget` (`K`/`M`/`G`/`T`, `KiB`/`MiB`/`GiB`/`TiB`; decimals rejected).
- `scx upgrade <input> [output] [--in-place]`.
- `scx query <input> <filter> [--count] [--output P] [--select-genes PATH] [--normalize N] [--log1p] [--limit N] [--json]` — `<input>` accepts a local `.scx`, an exploded `.scxd/` dir, or a cloud URL (`gs://`/`s3://`/`az://`/`file://`).

The CLI binary is named `scx`. h5ad/h5mu support and cloud ops are opt-in
build-time features; if `scx convert` errors about HDF5, the build lacks that
feature.

## Cloud (requires the cloud feature)
- `pyscx.pull(source, dest, filter=None, parallelism=None)` / `pyscx.push(source, dest, parallelism=None)`.
- `pyscx.open_cloud(url) -> PyCloudExperiment` — direct range reads; `.query().filter_obs(...).select_genes(...).collect()` with no `scx pull` step.
- `pyscx.cloud_optimize(input, output=None)`, `pyscx.explode(input, output)`, `pyscx.pack(input, output)`.
- CLI: `scx cloud-optimize`, `scx explode`, `scx pack`, `scx pull <url> <dest> [--filter EXPR]`, `scx push <src> <url>`.
