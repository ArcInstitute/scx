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
from_h5ad(path, out, codec=None, shard_size=None, shard_obs="auto", csc=None,
          csc_cols_per_shard=5000, uns_format="tagged", stream=True,
          strict_uns=False, dense_zero_epsilon=0.0, memory_budget=None,
          temp_dir=None, index_obs=None, index_var=None, index_preset=None,
          index_auto_threshold=1000, bitmap="off", reader_threads=None,
          writer_queue_depth=4, sort_by=None, reverse=False,
          obs_override=None, var_override=None, uns_override=None)
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
  `InferredEncoding`, `DenseSparsified`, `DuplicateCoordinatesMerged`,
  `SkippedColumn` (unreadable obs/var column), `SkippedObsm` (unreadable
  obsm/varm), `LayerSkipped` (unreadable / shape-mismatched layer),
  `DroppedObsp` (CSC or unsupported pairwise `obsp`/`varp` dropped — CSR and
  dense are preserved as COO), `UnsupportedUnsDataframeColumn` (one column of a
  `uns` pandas DataFrame is in a nullable encoding with no lossless `uns` form
  and is left out; the index, the column order and every other column are
  intact). All formerly-silent skips now warn. For the full preserved/lossy/dropped matrix
  see [docs/api.md § Round-trip fidelity](../../../docs/api.md#round-trip-fidelity).
- `reader_threads`: parallel streaming reader. `None` → `RAYON_NUM_THREADS` or
  `os.cpu_count()`; `1` forces sequential; `>1` requests rayon workers
  (byte-identical output). Requires a thread-safe libhdf5 (conda-forge default);
  a static libhdf5 falls back to sequential with a one-shot `Hdf5NotThreadsafe`
  warning. `memory_budget` derates the granted count.
- `writer_queue_depth` (default 4): backpressure window; outstanding shards
  capped at `reader_threads + writer_queue_depth`.
- `sort_by`: list of obs column names to globally reorder the cell axis by
  (lexicographic; the leading key gets the best X-read-locality benefit).
  `reverse=True` for descending. Sort-on-convert forces `stream=True`.
  Also available on `from_anndata`. CLI: `--sort-by CSV [--sort-reverse]`.

### `pyscx.from_anndata(adata, path, ...)` — write an in-memory (or backed) AnnData
```
from_anndata(adata, path, codec=None, shard_size=None, in_place=False,
             csc=None, csc_cols_per_shard=5000, uns_format="tagged",
             index_obs=None, index_var=None, index_preset=None,
             index_auto_threshold=1000, bitmap="off", memory_budget=None,
             force_legacy_metadata=False, sort_by=None, reverse=False)
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
- `pyscx.from_h5mu(path, out, codec=None, shard_size=None, shard_obs="auto", csc=None, csc_cols_per_shard=5000, stream=True, strict_uns=False, memory_budget=None, temp_dir=None, modalities=None, modality_types=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4)` — multimodal SCX v2. `modalities=["rna","adt"]` keeps a subset (unknown names raise); `modality_types={"adt":"protein","peaks":"atac"}` overrides inferred types (others emit `ModalityTypeInferred`). Per-modality dispatch mirrors `from_h5ad`.
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None, csc=None, csc_cols_per_shard=5000, uns_format="tagged", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", memory_budget=None, force_legacy_metadata=False)`. `csc=None` is `"auto"` — a CSC sidecar when `n_obs ≥ 50000` and `n_vars ≥ 5000` (same rule as `from_anndata` / `from_h5ad` / `from_h5mu`); `csc="off"` opts out.
- `pyscx.from_mtx(mtx_dir, scx_path, codec=None, shard_size=None, shard_obs="auto", allow_lossy=False)` — Cell Ranger MTX dir (`matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, `features.tsv[.gz]`). Default shard size 16384. Orientation is auto-detected: Cell Ranger's native **features × barcodes** matrix is transposed to cells×genes (an already-cells×genes matrix is kept; a square matrix assumes Cell Ranger's layout with a warning; a dimension mismatch is a hard error), so a standard `filtered_feature_bc_matrix/` converts to a `(n_cells, n_genes)` SCX file.
- `pyscx.from_mudata(mu, path, codec=None, shard_size=None, csc="off", csc_cols_per_shard=5000, codec_per_modality=True, uns_format="tagged", row_group_rows=256)` — write an in-memory `mudata.MuData` to SCX. Requires the `[mudata]` extra. Unlike the four entry points above it takes **no** `index_*` / `bitmap` / `memory_budget` kwargs, and its default stays `csc="off"`: it cannot build per-modality sidecars (it refuses `"always"` and degrades `"auto"`), so the `"auto"` default of the entry points above would buy nothing here.

## Export out of SCX

- `pyscx.to_h5ad(path_or_experiment, out, stream=True, modality=None, reader_threads=None, writer_queue_depth=4, memory_budget=None)` — **free function**, not a method; source may be a path or a `pyscx.Experiment`. Mirror of `from_h5ad`. Deletion vectors honored (only kept rows written; `shape[0] = n_obs - n_deleted`). For a multimodal SCX file pass `modality="rna"` to extract one modality, else it raises. `reader_threads` here does **not** require a thread-safe libhdf5 (HDF5 writes stay on the calling thread).
- `pyscx.to_h5mu(path, out, stream=True, reader_threads=None, writer_queue_depth=4, memory_budget=None)` — requires `reader.is_multimodal()`; single-modality files raise (use `to_h5ad`).
- `pyscx.to_mtx(scx_path, output_dir, modality=None)` — Cell Ranger–style MTX (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`). Streams shard by shard; `modality=` is required on a multimodal file.

> `to_h5ad(exp, ...)` re-exports the **unmodified on-disk** file. To save an
> in-memory analysis, use `adata.write_h5ad(...)` or `from_anndata(adata, "out.scx")`.

### adata.raw

`adata.raw` (pre-normalization counts on its own, usually wider, var axis) round-trips
end-to-end on **both** write paths: `from_h5ad` ingests the `/raw` group (streamed
shard-by-shard on `stream=True`, the default) and the in-memory
`from_anndata(adata)` / `write(adata, path)` writes it from `adata.raw.X` +
`adata.raw.var`; `pyscx.open(...).to_anndata()` reconstructs `adata.raw`, and
`to_h5ad` re-emits `/raw/X` + `/raw/var`. Caveats: raw
is dropped with a `DroppedRaw` warning under obs-filtered `to_anndata`, backed mode,
and deletion-vector-active files (those paths don't yet re-filter raw's obs axis),
and with `DroppedRawOnWrite` on two write paths: the SCX-backed / lazy-`X` rewrite
(`pyscx.open(f).to_anndata(backed=True)` → `from_anndata`), where backed
reconstruction leaves `.raw` unset — convert from the h5ad to keep raw there — and
reorder-on-convert (`from_h5ad(..., sort_by=…)` / `scx convert --sort-by` /
`--group-by`), where raw streams in source order while X/obs/obsm/layers are
permuted, so keeping it would misalign raw's rows — convert without the reorder to
keep raw. The warning carries a per-call-site remedy for exactly this reason.
`adata.raw.varm` has no section in the raw family and is dropped with
`DroppedRawVarm` on both ingest doors. A raw whose `X` row count disagrees with
`n_obs` is rejected rather than written.

## Shared kwarg semantics

- **`codec`**: `"auto"` (default, per-shard selection by median value), `"scx1"`
  (Delta-Golomb + FOR-BP + Rice; integer-only), `"zstd"`, `"pcodec"` (best for
  float layers), `"lz4"`, `"none"`.
  Also `"compact-trial"` (requires `row_group_rows=N`, e.g. 256 — trial-encodes
  each shard with both the auto-selected codec and ShufDeltaZstd, keeps the
  per-shard smaller; 1.3–2.1× smaller integer files at the cost of ~2× encode
  time; produces framed v4 output) and `"shufdelta"` (force ShufDeltaZstd).
  **Guidance:** `"auto"` (default) is best for GPU ML training and CPU analysis
  (Scx1 has in-VRAM GPU decode + faster CPU decode); `"compact-trial"` is best
  for storage-constrained archival or cloud hosting (smaller files, random
  access via framing). Float data always routes to Pcodec regardless of the
  codec setting. See `docs/codec.md § Codec tradeoff summary` for the full
  comparison table.
- **`csc`**: `None` (default; resolved to `"auto"`), `"off"` (the opt-out),
  `"auto"`, or `"always"`. `"always"`/`"auto"`
  write a column-major sidecar, built in the same pass as X (each streamed CSR
  shard is pushed into the builder; no extra read, no second copy on disk;
  under `memory_budget` the builder takes a quarter of it). `"auto"` builds
  it only when the dataset clears the size thresholds (`n_obs ≥ 50000` **and**
  `n_vars ≥ 5000`, env-tunable via `SCX_CSC_AUTO_OBS_THRESHOLD` /
  `SCX_CSC_AUTO_VARS_THRESHOLD` — set either to `0` to force a build). Required
  to use the `prefer_format="csc"` accel paths and the v3-CSC GPU DE driver.
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

## Grouped sharding

Cluster rows by an obs label so every shard holds exactly one group (e.g. one
guide target, one donor). Reference/control cells land in the leading shard(s).

**Python:**
```python
pyscx.from_h5ad("screen.h5ad", "screen.scx",
                group_by="target_gene", reference=["non-targeting"])

# boolean obs column as reference selector
pyscx.from_h5ad("screen.h5ad", "screen.scx",
                group_by="target_gene", reference={"column": "is_control"})

# explicit byte budget per shard
pyscx.from_h5ad("screen.h5ad", "screen.scx",
                group_by="target_gene", reference=["non-targeting"],
                group_target_bytes=256*1024*1024)

# re-sort an existing file into grouped order
pyscx.sort("screen.scx", "screen_grouped.scx",
           group_by="target_gene", reference=["non-targeting"])
```

**CLI:**
```bash
scx convert screen.h5ad screen.scx --group-by target_gene --reference non-targeting
scx convert screen.h5ad screen.scx --group-by target_gene --reference non-targeting \
    --group-target-bytes 256MiB
scx sort screen.scx screen_grouped.scx --group-by target_gene --reference non-targeting
```

**Kwargs / flags:**

- **`group_by`** (`str`): obs column whose labels cluster rows into shards
  (e.g. `target_gene`, `cell_type`, `donor_id`).
- **`reference`** (`list[str]` | `dict`): labels that mark reference/control
  cells. `["non-targeting"]` → label match; `{"column": "is_control"}` →
  boolean obs column. Requires `group_by`.
- **`group_target_bytes`** (`int`, optional): byte budget per shard for the
  bin-packer. Defaults to a value derived from `shard_size`.
- **`group_max_bytes`** (`int`, optional): oversize warning threshold (default
  4× target).
- **`group_pass`** (`str`, optional): `"auto"` (default), `"one"`, or `"two"`.
  `auto` routes CSR sources to one-pass streaming gather and dense sources to
  two-pass (convert then sort). One-pass is ~1.7× faster / ~2× lighter on CSR.

**Behaviour:**

- Reference cells are isolated in the leading shard(s) (shard 0). A group never
  straddles a shard boundary.
- The `group_by` column is auto-indexed for predicate pushdown.
- `append` drops the group index (with a warning); re-sort to restore grouping.
- Requires single-modality inputs (multimodal `--group-by` errors).
- CSC sidecars and h5mu inputs are not supported with `--group-by`.

## File ops (Python)

- `pyscx.append(target, input, codec=None, shard_size=None, index_*=...)` — streaming append from an SCX file (raw-copy fast path when codec/encoding match).
- `pyscx.append_from_anndata(target, adata, ...)`.
- `pyscx.mark_deleted(path, cell_indices)` — logical deletion (deletion vector).
- `pyscx.compact(input, output, index_*=...)` — rewrite reclaiming deleted space.
- `pyscx.rollback(path, to_seq=None)`.
- `pyscx.merge(inputs, output, index_*=..., assume_identical_var=False, uns_policy="first", shard_target_rows=None)` — streams obs shard-by-shard. `assume_identical_var=False` validates var index/columns/values across inputs (`True` checks only `n_vars`). `uns_policy`: `"first"` / `"require_equal"` / `"namespace"` / `"summary"`. Pass `index_*` or pushdown regresses to a full scan on the merged file.
- `pyscx.subset(input_path, output_path=None, *, filter=None, genes=None, modality=None, dry_run=False, codec=None, shard_size=None, memory_budget=None, index_obs=None, index_var=None, index_preset=None)` — Python wrapper for `scx subset`.
- `pyscx.delete(path, *, filter)` — Python wrapper for `scx delete --filter`.

## CLI (`scx`)

```
scx convert <input> <output> [--force] [--from h5ad|10x|h5mu|mtx|scx] [--to h5ad|h5mu|scx]
    [--codec auto|none|scx1|zstd|lz4|pcodec] [--shard-size N]
    [--stream[=true|false]] [--csc off|auto|always] [--csc-cols-per-shard N]
    [--modality NAME] [--memory-budget SIZE] [--strict-uns]
    [--dense-zero-epsilon F] [--temp-dir DIR] [--modalities CSV]
    [--modality-types NAME:TYPE,...] [--index-obs CSV] [--index-var CSV]
    [--index-preset cellxgene|perturbseq|training] [--index-auto-threshold N]
    [--bitmap off|auto|always] [--reader-threads N] [--writer-queue-depth N]
    [--sort-by CSV] [--sort-reverse]
    [--obs-override PATH] [--var-override PATH] [--uns-override PATH] [--allow-lossy]
```
- `--force` is required to overwrite an existing destination. For `--to mtx`
  that means an existing `matrix.mtx.gz` / `barcodes.tsv.gz` / `features.tsv.gz`
  / `genes.tsv.gz` inside the output directory, not the directory itself. The
  export never deletes anything up front: it writes its three members under
  temp names and swaps them in together once all three have succeeded, so a
  failure leaves the previous export intact. The alternative spellings the
  writer does not produce (`genes.tsv[.gz]`, the uncompressed forms) are
  cleared afterwards, under `--force`, so no stale file is left describing a
  different matrix.
- `--stream` bounds peak memory. It applies to h5ad ↔ SCX and h5mu ↔ SCX in both
  directions and to 10x → SCX, all of which stream by default. MTX ↔ SCX has a
  single path in each direction with nothing to select — SCX → MTX always
  streams one decoded shard at a time, MTX → SCX always materialises — so omit
  the flag (an explicit `--stream` there is an error; `--stream=false` is
  accepted as a no-op). A value needs the `=` form.
- `--allow-lossy` applies to `--from mtx` only. A `%%MatrixMarket ... integer ...`
  header is taken at its word: values parse as integers, and one whose magnitude
  exceeds 2²⁴ (the largest integer `float32` holds exactly) is refused, because
  SCX ingest carries values as f32 and could only store it rounded.
  `--allow-lossy` accepts the rounding.
- On ingest, `--csc always` (or `--csc auto` over a dataset above the size
  thresholds) builds the sidecar in the same pass as X.
- For multimodal SCX → h5ad, combine `--to h5ad --modality NAME`; `--to mtx
  --modality NAME` does the same and is **required** on a multimodal input, since
  an MTX directory holds one matrix over one feature space.

Other commands:
- `scx info <source> [--json] [--history]` — metadata; `<source>` accepts a file, `.scxd/` dir, or cloud URL. Multimodal shows a per-modality table with a `has_csc` column.
- `scx validate <file> [--verbose] [--deep]` — verifies BLAKE3 checksums section-by-section. `--deep` also decodes sparse shards and validates the canonical CSR invariant.
- `scx subset <input> [output] [--force] [--filter EXPR] [--genes PATH] [--modality NAME] [--dry-run] [--shard-size N] [--codec ...] [--memory-budget SIZE] [--csc carry|always|off] [--csc-cols-per-shard N] [--csc-memory-limit SIZE]` (`output` optional with `--dry-run`).
- `scx append <target> <source> [--codec ...] [--shard-size N] [--index-* ...] [--modality NAME] [--rebuild-csc]`.
- `scx delete <file> --filter <expr> [--dry-run]`.
- `scx compact <input> <output> [--force] [--index-* ...] [--codec ...] [--memory-budget SIZE] [--csc carry|always|off] [--csc-cols-per-shard N] [--csc-memory-limit SIZE] [--reshape-obs]`. `--csc carry` (the default on `compact`/`merge`/`optimize`/`sort`/`subset`) rebuilds the output's CSC sidecar in the same pass iff the input had one (merge: any input); `always` builds one regardless, `off` none. A multimodal input's sidecars are dropped (`always` refused). `--rebuild-csc` there is a deprecated alias for `--csc always`; on `append` it still means an in-place rebuild after the append (append otherwise drops the sidecar).
- `scx optimize <input> <output> [--force] [--codec auto|scx1] [--shard-obs off|auto|always] [--csc carry|always|off] [--csc-cols-per-shard N] [--csc-memory-limit SIZE]` — in-place upgrade (single-modality): re-encode + canonicalize + row-group-frame CSR shards and stamp `format_version=4` (preserves rows/obs/var/obsm/uns/indexes/deletion-vectors; carries CSC, rebuilt from the re-canonicalized X in the same pass); no decode sidecar is written. Use to make an older file row-group random-access + GPU-device-decode-fast without a full reconvert. `--shard-obs` (default `auto`) also migrates a **legacy single-section** obs to the sharded `ObsMetadataShard` layout when `n_obs > shard_target_rows` (`always` = unconditional, `off` = keep single section); already-sharded obs is preserved as-is. Python: `pyscx.optimize(input, output, codec="auto", shard_obs="auto")`.
- `scx merge <f1> <f2> [...] --output <path> [--force] [--index-* ...] [--assume-identical-var] [--uns-policy first|require-equal|namespace|summary] [--sort-by CSV] [--sort-reverse] [--codec ...] [--memory-budget SIZE] [--csc carry|always|off] [--csc-cols-per-shard N] [--csc-memory-limit SIZE]`.
- `scx sort <input> <output> --by CSV [--reverse] [--force] [--shard-size N] [--codec ...] [--index-* ...] [--memory-budget SIZE] [--temp-dir DIR] [--bitmap off|auto|always] [--csc carry|always|off] [--csc-cols-per-shard N] [--csc-memory-limit SIZE]` — globally reorder cells by obs columns for query locality.
- `scx sort <input> <output> --shuffle [--seed N] [--codec ...]` — the training counterpart: reorder cells by a seeded random permutation so a loader gets i.i.d. batches at any `shard_group_size`. Mutually exclusive with `--by` / `--group-by` / `--reverse`; the seed (default 42) is recorded in provenance. Pass `--codec` to hold the input's encoding — left at `auto` the adaptive codec re-selects and the file can grow substantially. Python: `pyscx.shuffle(input, output, seed=42)`.
- `scx set-uns <file> --uns JSON_FILE [--merge]` — replace the `uns` block in place (no X re-encode); `--merge` shallow-merges the JSON object's top-level keys into the existing block instead (`pyscx.update_uns`).
- `scx modify-metadata <file> [--uns JSON] [--obs PARQUET] [--var PARQUET] [--obsm NAME=PATH.npy ...] [--varm NAME=PATH.npy ...] [--index-* ...] [--modality NAME]` — replace metadata sections in place.
- `scx rollback <file> [--to-seq N]`.
- `scx build-csc <input> <output> [--memory-limit 4G] [--force] [--csc-cols-per-shard N]` — `--memory-limit` takes the same size forms as `--memory-budget`.
- `scx upgrade <input> [output] [--force] [--in-place]`.
- `scx query <input> <filter> [--filter EXPR] [--count] [--output P] [--force] [--select-genes PATH] [--normalize N] [--log1p] [--limit N] [--json] [--explain]` — `<filter>` is a positional obs predicate; alternatively pass `--filter EXPR` (one form, not both). `<input>` accepts a local `.scx`, `.scxd/` dir, or cloud URL. `--explain` prints the pushdown plan.
- `scx benchmark <file> [--compare-h5ad] [--runs N] [--json]` — read/query benchmarks.

The CLI binary is named `scx`. h5ad/h5mu support and cloud ops are opt-in
build-time features; if `scx convert` errors about HDF5, the build lacks that
feature.

## Cloud (requires the cloud feature)
- `pyscx.pull(source, dest, filter=None, parallelism=None)` / `pyscx.push(source, dest, parallelism=None)`.
- `pyscx.open_cloud(url) -> CloudExperiment` — direct range reads; `.query().filter_obs(...).select_genes(...).collect()` with no `scx pull` step.
- `pyscx.read_cloud(url, *, obs_filter=None, var_names=None) -> AnnData` — one-call cloud read (wraps `open_cloud(url).query()…collect().to_anndata()`); `file://` / local paths work too.
- `pyscx.cloud_optimize(input, output=None)`, `pyscx.explode(input, output)`, `pyscx.pack(input, output)`.
- CLI: `scx cloud-optimize`, `scx explode`, `scx pack`, `scx pull <url> <dest> [--filter EXPR]`, `scx push <src> <url>`.
