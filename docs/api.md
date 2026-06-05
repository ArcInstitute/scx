# SCX API Reference

## Section Types

26 section types are defined in `scx-format/src/section.rs`:

```
ObsMetadata (0)        — Arrow IPC metadata for observations
ObsIndex (1)           — Arrow IPC index for observations
VarMetadata (2)        — Arrow IPC metadata for variables
VarIndex (3)           — Arrow IPC index for variables
CsrShard (4)           — Main expression matrix (row-major)
CscShard (5)           — Column-major (gene-major) sparse shard. Used
                         for column-axis analytical workloads (DE, HVG,
                         per-gene QC). Optional sidecar; written via
                         `scx convert --csc=always` or `scx build-csc`.
BitmapShard (6)        — Per-shard detection bitmap sidecar (gene →
                         local-row roaring bitmaps; `SCXB` magic).
                         Written by `scx convert --bitmap auto|always`
                         and the pyscx `bitmap="..."` kwarg; consumed
                         by `PyExperiment.detection_counts` /
                         `cells_expressing`. See
                         [docs/format.md § Detection Bitmap](format.md#12-detection-bitmap-optional).
LayerCsrShard (7)      — Alternative expression layers
ObsmEmbedding (8)      — Embeddings (obsm)
ObspCsrShard (9)       — Reserved (legacy); current obsp persistence uses
                         ObspEmbedding (18).
UnsBlob (10)           — Unstructured metadata (JSON)
Provenance (11)        — Operation history
DeletionVectors (12)   — Logical deletion tracking (Roaring Bitmap)
ObsPredicateIndex (13) — Obs predicate index for query pushdown
VarPredicateIndex (14) — Var predicate index for query pushdown
ModalityTable (15)     — v2; ordered list of named modalities (CITE-seq,
                         10x Multiome, …). See docs/format.md § 13.
LayerCscShard (16)     — v2; per-modality CSC sidecar for a named layer
                         (parallel to LayerCsrShard).
VarmEmbedding (17)     — Dense var embeddings (varm); same wire format
                         as ObsmEmbedding but indexed by var.
ObspEmbedding (18)     — Sparse obs×obs pairwise matrices (e.g.
                         kNN connectivities/distances). COO Arrow IPC
                         with schema metadata n_rows / n_cols; data is
                         stored as float32.
VarpEmbedding (19)     — Sparse var×var pairwise matrices. Same wire
                         format as ObspEmbedding.
ObsmEmbeddingShard (20)— Row-sharded obsm (section per shard × key).
VarmEmbeddingShard (21)— Row-sharded varm (mirror of 20).
ObspEmbeddingShard (22)— Row-sharded obsp (section per shard × key).
VarpEmbeddingShard (23)— Row-sharded varp (mirror of 22).
ObsMetadataShard (24)  — Row-sharded obs Arrow IPC. Produced by merge,
                         append, and from_anndata when n_obs exceeds
                         shard_target_rows. Mutually exclusive with
                         ObsMetadata (0) in the same file.
VarMetadataShard (25)  — Row-sharded var Arrow IPC (mirror of 24).
```

## ScxReader (`scx-format/src/reader.rs`)

- `open(path)` — Open and validate file (mmap-based, validates magic/version/minimum size)
- `header()`, `root_catalog()`, `catalog()` — Access file metadata
- `n_obs()`, `n_vars()`, `nnz()` — Quick dimension access
- `read_obs()`/`read_var()` — Arrow RecordBatch metadata
- `read_csr_shard(idx)` — Single shard as `(Vec<i64>, Vec<i32>, Vec<f32>)`
- `read_all_csr_shards()` — Full matrix as `ScxCsr` (parallel via rayon)
- `read_csc_shard(idx)` — Single CSC sidecar shard as `ScxCsc`
- `read_all_csc_shards()` — Concatenated CSC matrix as `ScxCsc`
- `read_csc_columns(col_range)` — CSC columns covering a half-open
  `Range<u32>`; only shards overlapping the range are decoded
- `read_csc_columns_subset(cols)` — CSC columns gathered from a sorted
  unique `&[u32]`; contiguous runs are read in a single decode
- `csc_shard_count()` — Number of CSC sidecar shards (0 when
  `header.has_csc()` is false)
- `read_layer(name)` — Named layer as `ScxCsr`
- `layer_names()` — List available layer names
- `read_obsm(name)` / `read_all_obsm()` — Embeddings as Arrow RecordBatch
- `read_uns()` — Unstructured metadata as `serde_json::Value`
- `read_provenance()` — Operation history
- `validate()` — Check all section BLAKE3 checksums
- `section_bytes(entry)` — Direct byte access to a section
- `read_obs_schema()` / `read_var_schema()` — Arrow schema (without data)
- `read_obs_schema_physical()` / `read_var_schema_physical()` — Physical Arrow schema from the first on-disk section (without data)
- `read_obs_schema_logical_lossy()` / `read_var_schema_logical_lossy()` — Logical schema assembled from all shards (field union; lossy because cross-shard type conflicts are resolved by first-seen-wins)
- `obs_shard_count()` / `var_shard_count()` — Number of `ObsMetadataShard` / `VarMetadataShard` sections (0 on legacy single-section files)
- `read_obs_shard(idx)` / `read_var_shard(idx)` — Single metadata shard as Arrow RecordBatch
- `obs_shards()` / `var_shards()` — Iterator over all metadata shards
- `read_obs_assembled()` / `read_var_assembled()` — Reassemble all metadata shards into one Arrow RecordBatch (transparent on legacy single-section files)
- `debug_counts()` — `ReaderDebugCounts` with `AtomicU64` I/O counters (`cfg(debug_assertions)` only)
- `read_obs_predicate_index_bytes()` / `read_var_predicate_index_bytes()` — Predicate index raw bytes
- `read_deletion_vectors()` — Roaring Bitmap deletion vectors
- `read_all_csr_shards_filtered()` — Full matrix with deletion vector filtering
- `read_shard_header()` / `read_raw_shard_bytes()` / `read_shard_from_entry()` — Low-level shard access
- `mmap()` — Direct mmap access to the underlying file

## ScxWriter (`scx-format/src/writer.rs`)

- `new(path, header)` — Create writer (writes to temp file)
- `write_obs(batch)`/`write_var(batch)` — Arrow IPC metadata
- `write_csr_shard(indptr, indices, values, ...)` — CSR expression data
- `write_layer_csr_shard(name, ...)` — Named layer CSR data
- `write_obsp_shard(name, ...)` — Cell-cell graph CSR data
- `write_obsm(name, batch)` — Embeddings
- `write_uns(json)` — JSON metadata
- `write_provenance(operations)` — Operation history
- `write_csc_shard(indptr, indices, values, codec_id, value_encoding, col_start)`
  — Write a CSC sidecar shard. Fully supported. The catalog
  `section_type` is `CscShard (5)`; the on-disk shard header carries
  `shard_type = 1` going forward (readers also accept the legacy
  `shard_type = 0` when the catalog `section_type` is `CscShard`).
  `finish()` updates `header.n_csc_shards` from the writer's
  internal counter and sets the `has_csc` flag bit.
- `write_obs_predicate_index(data)` / `write_var_predicate_index(data)` — Predicate indexes for pushdown
- `write_deletion_vectors(dv)` — Roaring Bitmap deletion vectors
- `write_raw_shard(raw_bytes, section_type, name, stats, nnz)` — Pre-encoded shard passthrough
- `set_shard_column_stats(column_stats)` — Per-column shard statistics for catalog
- `finish()` — Atomic write: full catalog at EOF -> pwrite root catalog -> pwrite header -> fsync -> rename

## Multimodal API

v2 SCX files carry multiple modalities (RNA + ADT + ATAC + …) routed
via a 1-byte `modality_id` stamped on each catalog entry. See
[docs/format.md § 13](format.md#13-multimodal-extension) for the
on-disk layout and [docs/multimodal.md](multimodal.md) for the
end-to-end usage guide.

### `ModalityType` enum (`scx-format/src/modality.rs`)

```
ModalityType::Rna           = 0
ModalityType::Protein       = 1   // ADT
ModalityType::Atac          = 2
ModalityType::Spatial       = 3
ModalityType::Methylation   = 4
ModalityType::Custom        = 255
```

Drives per-modality codec selection (see
[docs/codec.md § Per-modality codec defaults](codec.md#8a-per-modality-codec-defaults)).

### `ModalityInfo` struct

Per-modality record carried on disk by the `ModalityTable` section:

```rust
pub struct ModalityInfo {
    pub name: String,                  // UTF-8, ≤ 64 bytes, unique
    pub modality_type: ModalityType,
    pub default_codec_id: u8,          // CodecId u8 repr
    pub default_value_encoding: u8,    // ValueEncoding u8 repr
    pub n_vars: u64,                   // Per-modality variable count
    pub nnz: u64,                      // Per-modality non-zeros
    pub n_csr_shards: u32,
    pub n_csc_shards: u32,
    pub flags: ModalityFlags,          // HAS_CSC, HAS_OBSM, HAS_OBSP, HAS_LAYERS, HAS_UNS
}
```

### `ScxReader` — multimodal accessors

- `is_multimodal()` / `n_modalities()` / `has_modalities()` — capability checks.
- `modality_table()` → `Option<&ModalityTable>` — full parsed table.
- `modality_names()` → `Vec<&str>` — names in registration order.
- `modality_id(name)` → `Option<u8>` — name → id resolution.
- `modality_info(id)` → `Option<&ModalityInfo>` — per-modality record.
- `read_var_for(modality_id)` — per-modality var as `RecordBatch`.
- `read_csr_shard_for(modality_id, shard_idx)` — single shard from
  the modality's CSR shard list.
- `read_all_csr_shards_for(modality_id)` — concatenated CSR for the
  modality. Patches `n_cols` from `modality_info(id).n_vars` so the
  resulting `ScxCsr` has the modality's per-modality `n_vars` rather
  than the file-wide `header.n_vars` (which is the max across modalities).
- `csr_shard_count_for(modality_id)` — CSR shard count.
- `read_all_csc_shards_for(modality_id)` / `csc_shard_count_for(modality_id)`
  — same shape on the column-major axis.
- `read_obsm_for(modality_id, key)` — per-modality embeddings.

### `ScxWriter` — multimodal writers

- `add_modality(name, modality_type, default_codec, default_value_encoding)` →
  `Result<u8>` — register a modality (1-based id). Validates the name
  (≤ 64 bytes, UTF-8, unique).
- `set_modality_n_vars(modality_id, n_vars)` — record the per-modality
  variable count (called once per modality, before the first per-modality
  shard write).
- `modality_id(name)` → `Option<u8>` / `modality_name_for(id)` — lookups.
- `write_var_for(modality_id, batch)` — per-modality var.
- `write_csr_shard_for(modality_id, indptr, indices, values, codec_id, encoding, row_start)`
  — per-modality CSR shard. Uses the `X/{name}/shard_{i}` naming convention.
  Updates the modality's `n_csr_shards` / `nnz` automatically.
- `write_csc_shard_for(modality_id, indptr, indices, values, codec_id, encoding, col_start)`
  — per-modality CSC shard (`X_csc/{name}/shard_{i}`). Section type is
  `CscShard` (5).
- `write_layer_for(modality_id, layer_name, ...)` /
  `write_obsm_for(modality_id, key, batch)` /
  `write_obsp_for(modality_id, key, ...)` /
  `write_uns_for(modality_id, json)` — per-modality variants of the
  legacy section writers. Stamp the catalog entry with the chosen
  `modality_id`.
- `finish()` — emits the `ModalityTable` section automatically when
  any modality was registered. Updates header `n_modalities` /
  `modality_table_offset` / `modality_table_length` and sets the
  `has_modalities` flag bit.

### Codec selection — `select_codec_for_modality`

`select_codec_for_modality(raw_values, value_encoding, modality_type)`
extends `select_codec` with per-modality routing:
RNA / Custom / Methylation / Spatial → delegate to `select_codec`;
Protein/ADT → Zstd for integers, Pcodec for floats; ATAC → Zstd for
binary peak presence (sample max ≤ 1) else Lz4Shuffle, Pcodec for
floats. See [docs/codec.md § Per-modality codec defaults](codec.md#8a-per-modality-codec-defaults).

### Per-modality `BackedCscReader`

`BackedCscReader::for_modality(reader, modality_id, cache_shards)` —
builds a column-major reader scoped to one modality so multimodal
training (e.g. totalVI) can hold separate caches per modality without
LRU thrashing across modalities.

### Python (`pyscx`)

- `pyscx.from_mudata(mu, path, codec="auto", ...)` — write a MuData
  object as a multimodal SCX file. Per-modality CSR shards stamped
  with `modality_id` derived from the registration order; per-modality
  codec resolved via `select_codec_for_modality`.
- `pyscx.open(path)` returns `PyExperiment`. New attrs / methods:
  - `is_multimodal: bool`, `n_modalities: int`, `modality_names: list[str]`.
  - `modality_id(name) -> int | None`, `modality_info(id) -> dict | None`.
  - `to_mudata() -> mudata.MuData` — round-trips back to MuData.
- `pyscx.MultimodalTrainingDataset(path, modalities=[…], …)` — yields
  per-batch dicts `{"X": {modality_name: ndarray}, "obs": {...},
  "cell_indices": ndarray}` (or tuples in `return_dict=False` mode).
- Backward compat: `pyscx.TrainingDataset(path)` on a multimodal file
  emits `UserWarning` and falls back to the alphabetically-first
  modality. Pass `modality="rna"` explicitly to suppress.

### R (`rscx`)

- `from_seurat(seu, path)` detects Seurat v5 multi-assay objects
  (`length(seu@assays) > 1`) and routes to a multimodal write path
  with one modality per assay. Single-assay objects keep the legacy
  path.
- `from_mae(mae, path)` writes a Bioconductor `MultiAssayExperiment`
  as multimodal SCX. Cells must align across experiments
  (`colnames(experiments[[i]])` identical); on misalignment, raises
  with a clear error directing to `intersectColumns(mae)`.
- `scx_open(path)$to_seurat()` builds a Seurat v5 multi-assay object
  on multimodal files (one `Assay5` per modality, shared `meta.data`).
- `scx_open(path)$to_mae()` builds a `MultiAssayExperiment` (one
  `SingleCellExperiment` per modality, shared `colData`).
- `scx_open(path)$is_multimodal()` / `$modality_names()` — capability
  checks.

### CLI surface

```
scx info path.scx                 # Modalities (N): name, type, n_vars, nnz, csr/csc, codec
scx validate path.scx             # ModalityTable checksum + n_modalities cross-check
scx convert --from h5mu in.h5mu --to scx out.scx          # h5mu → SCX (streaming by default)
scx convert --to h5ad out.scx out.h5ad --modality rna     # SCX → h5ad (streaming by default)
scx convert --to h5mu out.scx out.h5mu                    # SCX → h5mu (streaming by default)
scx convert --to h5ad out.scx out.h5ad --stream=false     # opt-out: materialising path
scx append target.scx --input new.scx --modality rna
scx subset in.scx --modality rna --output rna_only.scx
```

## Codec Selection (`scx-format/src/codec_select.rs`)

- `select_codec(values, encoding)` — Auto-select best codec per shard based on data type and distribution:
  - Integer values with median ≤ 8 → Scx1 (Rice coding, optimal for typical 10x UMI counts)
  - Integer values with median > 8 → Zstd (LZ77 dictionary wins for larger values)
  - Float values (Float32, Float16) → Pcodec (7–16% better compression than Zstd on log-normalized data)
- LZ4+shuffle (`codec="lz4"`) and Pcodec (`codec="pcodec"`) also available as explicit overrides

**Codec tradeoffs:**

| Codec | Best for | Compression | Read speed | Write speed |
|-------|----------|-------------|------------|-------------|
| `auto` | General use (recommended default) | Best per-shard | Best per-shard | Best per-shard |
| `scx1` | Small UMI counts (median ≤ 8) | Best for 10x data (~4.8×) | Fastest (SIMD decode) | Moderate |
| `zstd` | Large integers, general fallback | Good (~4.3× UMI, ~3.8× float) | Fast | Fast |
| `pcodec` | Log-normalized, PCA embeddings, float layers | Best for floats (~4.1–4.7×) | Moderate (19–39% slower than Zstd) | Slower (35–40% slower than Zstd) |
| `lz4` | Speed-critical pipelines | Lower (~2.2–3.1×) | Fast | Fastest compressed |
| `none` | GDS bypass, debugging | 1× (no compression) | Fastest (I/O bound) | Fastest |

For raw count data (integer-valued), `auto` selects Scx1 or Zstd — Pcodec falls through to Zstd internally since its advantage is specific to float values. For storage-constrained workflows with normalized float data, explicitly selecting `pcodec` gives the best compression. For latency-sensitive pipelines, `zstd` or `lz4` are better choices.

## Provenance

- `ProvenanceEntry`: timestamp, action, tool, params_json, input_checksums
- Auto-populated on `ScxWriter::finish()` with operation info
- `scx info` displays full provenance history
- Read/write via `ScxReader::read_provenance()` / `ScxWriter::write_provenance()`
- Conversion runs include a `params_json.warnings` summary (per-category
  counts emitted via [`WarningSink::summary_json`](#conversion-warnings-convertwarning)),
  along with `stream`, `source_format`, `source_matrix_format`,
  `indexed_obs`, `indexed_var`, `bitmap`, `memory_budget_mb`,
  `modalities`, and modality-type overrides where applicable.

## Conversion warnings (`ConvertWarning`)

Conversion paths surface structured warnings via a `WarningSink`
(`scx-convert/src/warnings.rs`). The CLI prints a per-category summary
at the end of `scx convert`; `pyscx` aggregates them and emits one
Python `UserWarning` per category. The full count + sample is also
recorded under `ProvenanceEntry.params_json.warnings`.

| Variant | Emitted by | Meaning |
| --- | --- | --- |
| `InferredEncoding { path, inferred }` | h5ad layout detection (`detect_matrix_format_at`, `open_x_streaming`) | h5ad `encoding-type` was missing or ambiguous; layout was inferred from group children or dataset shape. |
| `SkippedUnsKey { key, reason }` | `read_uns` / `read_uns_entry` | `uns` entry was unrepresentable; skipped under default `strict_uns=false`. `strict_uns=true` turns this into an error on the first occurrence. |
| `DenseSparsified { path, density }` | Dense h5ad streaming reader | Dense `/X` slab was sparsified during streaming. Reports density to help users decide whether dense storage is worth keeping. |
| `DuplicateCoordinatesMerged { count, policy }` | CSC h5ad streaming reader | CSC input contained duplicate `(row, col)` coordinates; values were summed (scipy `sum_duplicates` semantics). |
| `ModalityTypeInferred { name, modality_type }` | h5mu streaming reader | Modality name → `ModalityType` was inferred by name; override via `--modality-types NAME:TYPE` / `modality_types={...}`. |
| `MissingPresetIndexColumn { column }` | Predicate-index builder | A preset (`cellxgene` / `perturbseq` / `training`) referenced an obs/var column not present in the source; preset misses warn, conversion continues. |
| `UnsupportedIndexColumn { column, reason }` | Predicate-index builder | A user-forced (`--index-obs` / `--index-var`) or preset column has an unsupported dtype; forced columns hard-error, preset columns warn and skip. |
| `PredicateIndexSkippedMultimodal` | Predicate-index builder | Predicate indexes are unimodal-only on the read side today; emitted (and indexes skipped) when conversion input is multimodal. |
| `BitmapSkipped { reason }` | Detection bitmap auto policy | `--bitmap auto` rejected emission (e.g. `n_vars > 1_000_000`, estimated bitmap size > 15% of encoded CSR, dense X). |
| `DroppedObsp { name, reason }` | `scx merge` (multimodal) | `obsp` could not be merged (axis semantics don't compose); default-dropped with a warning. |
| `MappingPeakFootprintHigh { mapping, estimated_bytes, budget_bytes }` | `pyscx.from_anndata` | A single mapping's estimated in-memory footprint exceeds `memory_budget`. |
| `EagerAssemblyMemoryHigh { estimated_bytes, budget_bytes }` | `PyExperiment.to_anndata` | Estimated eager assembly footprint exceeds `memory_budget` (default 8 GiB). Warn-only, does not block. |
| `ThreadsafeHdf5Unavailable` | Parallel streaming reader fallback | libhdf5 was not built thread-safe; parallel streaming fell back to a single reader thread. |

## Memory budgets

`MemoryBudget::parse(s)` (`scx-format/src/mem.rs`) is the shared parser
behind `--memory-budget` and `build-csc --memory-limit` (CLI) and the
`memory_budget=` kwarg on `from_h5ad` / `from_h5mu`. It caps dense row
slabs in the h5ad streaming reader, CSC external-transpose buffers when
CSC-on-disk exceeds the budget, and the parallel streaming reader's
worker derate. It lives in `scx-format` (re-exported as
`scx_convert::MemoryBudget`) so sibling crates such as `scx-ops` —
which owns `build-csc` — can share it without a dependency cycle.

Accepted forms:

- bare byte counts (`"1048576"`, `1048576`),
- binary-prefix shorthand `K` / `M` / `G` / `T` (= `KiB` / `MiB` / …),
- explicit binary prefixes `KiB` / `MiB` / `GiB` / `TiB`.

Decimal prefixes (`KB`, `MB`, `GB`, `TB`) are **rejected** to avoid
1000-vs-1024 ambiguity. When the requested budget cannot fit even one
shard's metadata plus one worker, conversion refuses to start with
an actionable error rather than OOMing partway through.

## Conversion-time predicate indexes and detection bitmaps

CLI flags `--index-obs`, `--index-var`, `--index-preset`,
`--index-auto-threshold` (plus the equivalent
`index_obs=` / `index_var=` / `index_preset=` /
`index_auto_threshold=` kwargs on `from_anndata` / `from_h5ad` /
`from_10x` / `from_h5mu`) materialise `ObsPredicateIndex` /
`VarPredicateIndex` sections at write time so subsequent
`pyscx.open(...).query()` and `scx pull --filter` calls can push
predicates down without an obs scan.

The same `index_*` knobs are also accepted by the rewrite ops —
`pyscx.merge` / `pyscx.append` / `pyscx.append_from_anndata` /
`pyscx.compact` and `scx merge` / `scx append` / `scx compact`.
Without them, those ops drop (or, on append, leave stale) the
predicate-index sections — pushdown silently falls back to a full
obs scan. Pass `index_obs=[...]` (or the matching CLI flag) to
rebuild a fresh index covering every row of the output in the same
pass; this is the only supported way to keep pushdown working
across multi-input atlas builds (fan-out per-perturbation
conversions → `pyscx.merge(..., index_obs=[...])`).

Behaviour:

- Force-listed columns (`--index-obs`/`--index-var`) **hard-error** if
  the column is missing or has an unsupported dtype.
- Preset columns warn (`MissingPresetIndexColumn` /
  `UnsupportedIndexColumn`) and are skipped without aborting the
  conversion.
- Auto-indexing picks up categorical-like columns with cardinality
  `≤ index_auto_threshold` (default `1000`).
- Multimodal inputs emit `PredicateIndexSkippedMultimodal` and skip
  predicate-index emission entirely — the read path is unimodal-only
  today. The same skip-with-warning applies to multimodal
  `merge` / `append` / `compact`.
- The flags only build indexes on the SCX-writing ingest directions
  (`h5ad → scx`, `10x → scx`; `h5mu → scx` accepts them and skips with
  the warning above). On any other `scx convert` direction —
  `mtx → scx` and the SCX-export directions `scx → h5ad/h5mu/mtx` —
  passing `--index-*` is a **hard error** rather than a silent no-op,
  since those paths cannot build a predicate index. To (re)build an
  index on an existing SCX file, use `scx compact` / `scx append` /
  `scx merge` or `pyscx.from_anndata`.

Index presets:

| Preset | Expanded obs columns |
| --- | --- |
| `cellxgene` | `cell_type`, `cell_type_ontology_term_id`, `tissue`, `tissue_ontology_term_id`, `disease`, `assay`, `donor_id`, `development_stage`, `sex`, `suspension_type` |
| `perturbseq` | `cell_type`, `donor`, `batch`, `condition`, `perturbation`, `guide_id`, `target_gene`, `control`, `split` |
| `training` | `cell_type`, `donor`, `batch`, `dataset_id`, `split`, `organism`, `tissue` |

`--bitmap off|auto|always` (and the `bitmap=` kwarg) write
`BitmapShard` (section id 6) sidecars carrying per-shard
gene → local-row roaring bitmaps. Auto policy requires sparse X,
`n_vars ≤ 1_000_000`, and an estimated bitmap size ≤ 15 % of the
encoded CSR; ATAC modalities are always-on under `auto`. Consumed by
`PyExperiment.detection_counts(axis="var", modality=...)` and
`PyExperiment.cells_expressing(gene, modality=...)`; the backed reader
falls back to a CSR scan when sidecars are absent. The wire format is
specified in [docs/format.md § Detection Bitmap](format.md#12-detection-bitmap-optional).

## `pyscx.from_anndata` — backed and lazy `X`

`pyscx.from_anndata(adata, out)` accepts `adata.X` in three shapes:

1. **scipy / numpy** — the existing in-memory path; extracts CSR
   arrays, partitions into shards, writes.
2. **`ScxBackedSparseDataset`** (returned by
   `pyscx.open(p).to_anndata(backed=True)`) — streams the source SCX
   file shard-by-shard without materialising `X`.
3. **`ScxLazyTransformedDataset`** (after
   `pyscx.accel.normalize_total` / `pyscx.accel.log1p` / row scaling
   are stacked on a backed source) — same streaming write, with
   transforms applied per shard.

For shape 2 the writer chooses between two modes:

- **Byte-passthrough**: when the source and target shard layouts
  agree (matching `shard_size`, matching `codec`, source built from a
  single modality with no row deletions and no column projection),
  pre-encoded CSR shards are copied verbatim via
  `ScxWriter::copy_section_verbatim`. Provenance records
  `passthrough=true`.
- **Decode + encode**: any precondition mismatch (e.g.
  `shard_size=` override, `codec=` override, a deletion vector marked
  via `mark_deleted`, a `[:, gene_subset]` column projection) falls
  back to iterating the wrapper's user-visible shard boundaries,
  materialising each shard via `wrapper[start:end]` (which already
  applies deletions / projection), and re-encoding through
  `encode_one_shard` + `write_preencoded_shard`. The output `n_obs`
  / `n_vars` reflect the user-visible shape.

For shape 3 the rewrite is always decode + encode; transforms are
applied per shard inside `wrapper[start:end]`. The same
`csc="always"` two-pass rebuild is honoured (matches `from_h5ad`).
Any source CSC sidecar that would be invalidated by the rewrite is
dropped with a `UserWarning` unless `csc="always"` opts into a
fresh rebuild.

Each rewrite emits a `from_anndata` provenance entry with the
following `params_json` fields:

| Field | Type | Description |
| --- | --- | --- |
| `x_source` | `"scipy" \| "backed" \| "lazy"` | source of `X` |
| `passthrough` | `bool` | `true` only on the byte-passthrough fast path |
| `source_path` | `str \| null` | source SCX path (when the wrapper carried one) |
| `lazy_transforms` | `list[{"name","params"}]` | only for `x_source="lazy"`; per-row factor / row-sum vectors are summarised by length |
| `csc_dropped` | `bool` | source had a CSC sidecar that was dropped |

Read the chain via `PyExperiment.provenance()` — each entry is a
dict with `params_json` as a raw JSON string (parse with
`json.loads`).

The full transform list is also surfaced on the wrapper itself via
`ScxLazyTransformedDataset.transforms_repr() → list[{"name","params"}]`,
which the writer uses to build the `lazy_transforms` payload.

## BackedCsrReader (`scx-format/src/backed.rs`)

- `new(reader, cache_shards)` — Create backed reader from `ScxReader` with LRU shard cache
- `read_rows(start, end)` → `ScxCsr` — Decode and concatenate rows from relevant shards
- `read_row_indices(indices)` → `ScxCsr` — Decode specific rows by index (fancy indexing)
- `read_shard_cached(idx)` → `ScxCsr` — Read shard through LRU cache (clones on hit)
- `read_shard_uncached(idx)` → `ScxCsr` — Read shard bypassing cache (preferred for streaming)
- `row_sums()` / `col_sums()` — Streaming per-row/column sums
- `row_nnz()` / `col_nnz()` — Streaming per-row/column NNZ
- `row_var()` / `col_var()` — Streaming per-row/column variance
- `row_max()` / `col_max()` / `row_min()` / `col_min()` — Streaming extrema
- `total_nnz()` — Total NNZ across all shards
- `col_means_and_sum_sq(zero_center)` — Single-pass column statistics for PCA
- Masked variants (deletion-vector aware): `col_sums_masked(kept_rows)`, `col_nnz_masked(kept_rows)`, `col_max_masked(kept_rows)`, `col_min_masked(kept_rows)`, `col_var_masked(kept_rows)`

## ShardSource Trait (`scx-format/src/shard_source.rs`)

A uniform interface for streaming CSR data shard-by-shard, enabling algorithms like PCA to process data without materializing the full matrix. Defined in `scx-format`, available to both `scx-accel` and `pyscx`.

```rust
pub trait ShardSource {
    fn n_shards(&self) -> usize;
    fn n_obs(&self) -> usize;
    fn n_vars(&self) -> usize;
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }
    fn read_shard(&self, shard_idx: usize) -> Result<ScxCsr>;
    fn col_means_and_sum_sq(&self, zero_center: bool)
        -> Result<(Option<Vec<f64>>, Vec<f64>)>; // default impl provided
}
```

**Implementations:**

| Type | Crate | Behavior |
|------|-------|---------|
| `BackedCsrReader` | `scx-format` | Reads raw decoded shards via `read_shard_uncached()` |
| `LazyShardSource` | `pyscx` (internal) | Applies per-shard transforms (NormalizeTotal, Log1p, RowScale), column projection (`project_csr`), and deletion vector filtering before returning |

**Used by:** `scx-accel::randomized_pca()`, `streaming_spmm_forward()`, `streaming_spmm_transpose()` — all generic over `<S: ShardSource>` for zero-cost abstraction across the crate boundary.

**Design note:** The trait is defined in `scx-format` (not `scx-accel`) so that `pyscx`'s `LazyShardSource` can implement it without creating a dependency on `scx-accel`. PCA functions in `scx-accel` are generic (`<S: ShardSource>`) rather than using `&dyn ShardSource` to allow monomorphization.

## BackedCscReader (`scx-format/src/backed.rs`)

Column-major counterpart to `BackedCsrReader`. Streams CSC sidecar
shards from disk with an LRU shard cache, parallel to the CSR side.

- `new(reader, cache_shards)` — Create from `ScxReader`. `cache_shards = 0` disables caching (one decode per call)
- `index()` — Per-shard column-range index, sorted by `col_start`
- `n_shards()` / `n_obs()` / `n_vars()` — Dimensions
- `read_shard_uncached(idx)` → `ScxCsc` — Single CSC shard, bypass cache
- `read_shard_cached(idx)` → `Arc<ScxCsc>` — Single CSC shard, through cache
- `read_csc_columns(col_range)` → `ScxCsc` — Decode only shards overlapping the half-open range; partial-overlap shards are sliced post-decode
- `read_csc_columns_subset(cols)` → `ScxCsc` — Gather columns from a sorted unique `&[u32]`; contiguous runs share a single decode
- `enable_metrics()` / `metrics()` — Per-call hits / misses / decoded-bytes counters

## ColumnShardSource Trait (`scx-format/src/shard_source.rs`)

Sibling to `ShardSource` for column-major streaming. Defined in
`scx-format` so consumers in `scx-accel` and `pyscx` can take the
bound generically without depending on each other.

```rust
pub trait ColumnShardSource {
    fn n_csc_shards(&self) -> usize;
    fn n_obs(&self) -> usize;
    fn n_vars(&self) -> usize;
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }
    fn read_csc_shard(&self, shard_idx: usize) -> Result<ScxCsc>;
    fn read_csc_columns(&self, col_range: Range<u32>) -> Result<ScxCsc>;
    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)>;
}
```

**Implementations:**

| Type | Crate | Behavior |
|------|-------|---------|
| `BackedCscReader` | `scx-format` | Reads raw decoded CSC shards from disk (cache + index-driven shard skip) |
| `LazyShardSource` | `pyscx` (internal) | Applies the column-local subset of transforms (Log1p only) post-decode; honors column projection |

**Capability gate:** the trait is *not* a sub-trait of `ShardSource`.
Consumers that want CSC dispatch take the bound explicitly (`fn
require_csc<S: ColumnShardSource>(...)`); the runtime "does this
dataset support CSC?" question is answered exactly once at
`ScxBackedSparseDataset::as_column_source()` /
`ScxLazyTransformedDataset::as_column_source()`. Returns `Some` iff
the file has a CSC sidecar AND the transform chain is column-local
AND no row deletion vector is active. See `pyscx.accel.*
prefer_format` below.

**Used by:** `scx-accel::csc::{streaming_mean_var_csc,
streaming_clip_square_sum_csc, wilcoxon_rank_sum_streaming_csc,
pseudobulk_aggregate_csc}` and the `pyscx::projected_agg::*_csc`
column-aggregation kernels.

## scx-ops — File Operations

### `scx_ops::append(path, obs, indptr, indices, values, value_encoding, options: &AppendOptions) → Result<()>`
Append new cells at EOF from raw CSR arrays. `AppendOptions` bundles codec selection, shard sizing, and modality routing. Advisory flock for concurrent safety.

### `scx_ops::append_from_reader(target_path, source_reader, options: &AppendOptions, source_modality_id) → Result<()>`
Streaming SCX → SCX append. Reads one source CSR shard at a time (or copies raw bytes verbatim when codec / value encoding / index dtype / per-modality `n_vars` all match), avoiding materializing the entire source matrix in memory. Supports multimodal targets via `options.modality_id` / `source_modality_id` routing. CSC sidecars are dropped on append.

### `scx_ops::mark_deleted(path, cell_indices) → Result<u64>`
Logical deletion via Roaring Bitmap deletion vectors. Returns total deleted count.

### `scx_ops::compact(input, output) → Result<()>`
Rewrite file reclaiming deleted/orphaned space.

### `scx_ops::rollback(path) → Result<()>` / `rollback_to(path, seq) → Result<()>`
Revert to previous (or specific) manifest version — header-only update.

### `scx_ops::merge(inputs, output) → Result<()>`
Streaming merge of multiple SCX files into one.

## scx-engine — Query Engine

### `QueryPipeline`
Lazy pipeline builder — no I/O until `.collect()`:

```rust
QueryPipeline::open("file.scx")?
    .filter_obs("cell_type == 'T cell' and tissue == 'lung'")?
    .select_genes(hvg_indices)
    .with_normalize(1e4)
    .with_log1p()
    .limit(1000)
    .collect()?   // → QueryResult { x: ScxCsr, obs, var, skipped_shards, total_shards }
```

- Schema validation at construction time (fail-fast)
- Two-level predicate pushdown: catalog shard pruning + index row pruning
- Fused normalize+log1p in single CSR row scan
- Parallel shard processing via rayon

## scx-loader — Training Data Loader

### `TrainingPipeline`
Triple-buffered Rust pipeline: tokio I/O → rayon decode → Python/GPU.

```rust
let pipeline = TrainingPipeline::new("file.scx", LoaderConfig {
    batch_size: 1024,
    hvg_indices: Some(hvg_array),
    normalize: true,
    log1p: true,
    ..Default::default()
})?;
pipeline.start_epoch();
while let Some(batch) = pipeline.next_batch() {
    // batch.x: dense f32 matrix, batch.obs_columns: Vec<Vec<f32>>
}
```

## scx-cloud — Cloud Operations

### `cloud_optimize(input, output) → Result<()>`
Rewrite file with front-of-file catalog for single-read cloud opens.

### `explode(input, output_dir) → Result<()>`
Packed `.scx` → exploded `.scxd` directory (one file per section).

### `pack(input_dir, output) → Result<()>`
Exploded `.scxd` directory → packed `.scx` (cloud-ready by default).

### `pull(source, dest, options) → Result<PullStats>`
Streaming cloud → local packed file. Supports parallel downloads and selective filter.

### `push(source, dest, options) → Result<PushStats>`
Streaming local → cloud exploded directory. Parallel uploads.

### `CloudReader::open_cloud(url) → Result<CloudReader>`
Direct cloud reads without full download. Supports `.scxd/`, cloud-ready `.scx`, and non-cloud-ready `.scx`.

## scx-gpu — GPU Analysis

GPU-accelerated analysis APIs. Requires CUDA Toolkit ≥ 12.0 at build time.

### cuSPARSE SpMM

#### `spmm_csr(handle, stream, a, b, c, m, k, n, alpha, beta) → Result<(), GpuError>`
Sparse × dense matrix multiply: C = α·A·B + β·C. A is GPU-resident CSR, B/C are dense column-major f32.

#### `spmm_csr_transpose(handle, stream, a, b, c, m, k, n, alpha, beta) → Result<(), GpuError>`
Transposed SpMM: C = α·A^T·B + β·C.

### cuSOLVER Dense Operations

#### `gpu_qr_q(handle, stream, a, m, n) → Result<CudaSlice<f32>, GpuError>`
Economy QR decomposition on GPU: A = Q·R. Returns Q (m × n). Uses `cusolverDnSgeqrf` + `cusolverDnSorgqr`.

### cuRAND

#### `random_gaussian_gpu(stream, rows, cols, seed) → Result<CudaSlice<f32>, GpuError>`
Generate a random Gaussian matrix directly on GPU via cuRAND XORWOW generator.

### GPU PCA Pipeline

#### `gpu_randomized_pca(dev, reader, n_components, n_oversamples, n_power_iterations, zero_center, seed) → Result<GpuPcaResult, GpuError>`
Complete GPU-accelerated randomized PCA. Streams SpMM shard-by-shard via cuSPARSE, QR via cuSOLVER, SVD via CPU faer, final projection via GPU GEMM. Returns `GpuPcaResult { embeddings, components, variance_explained, variance_ratio, mean }`.

#### `mean_correct_gpu(dev, y, mc, n_obs, k) → Result<(), GpuError>`
Mean-centering correction kernel: Y[i,j] -= mc[j] for all rows.

### GPU kNN

#### `gpu_knn_cagra(dev, embeddings, n_obs, n_dims, n_neighbors) → Result<GpuKnnResult, GpuError>`
Build kNN graph on GPU using NVIDIA CAGRA (cuVS). L2 distance, optimized for PCA embeddings. Returns `GpuKnnResult { indices, distances, n_obs, n_neighbors }`.

#### `cuvs_available() → bool`
Check if `libcuvs.so` is available at runtime.

### GPU UMAP

#### `gpu_umap_native(dev, knn, n_components, min_dist, spread, n_epochs, seed) → Result<GpuUmapResult, GpuError>`
GPU UMAP via native CUDA SGD optimization kernel. Edge-parallel with `atomicAdd` for concurrent embedding updates.

### GPU Preprocessing

#### `gpu_normalize_log1p(dev, csr, target_sum) → Result<(), GpuError>`
Fused per-row normalize_total + log1p on GPU-resident CSR (in-place).

#### `gpu_normalize(dev, csr, target_sum) → Result<(), GpuError>`
Per-row normalize_total on GPU-resident CSR (in-place).

#### `gpu_log1p(dev, csr) → Result<(), GpuError>`
Element-wise log1p on GPU-resident CSR (in-place).

#### `gpu_apply_fused_ops(dev, csr, normalize, log1p, target_sum) → Result<(), GpuError>`
Apply configurable fused preprocessing ops on GPU-resident CSR.

### GpuError Variants

| Variant | Description |
|---------|-------------|
| `CudaError(String)` | CUDA runtime/driver error |
| `KernelLaunchFailed(String)` | Kernel launch failure |
| `GdsUnavailable(String)` | GDS not available |
| `DeviceNotFound(usize)` | GPU device not found |
| `InvalidShard(String)` | Malformed shard data |
| `ShapeMismatch { expected, got }` | Matrix dimension mismatch |
| `CodecError(CodecError)` | Codec decode error |
| `CuSparseError(String)` | cuSPARSE API error |
| `CuSolverError(String)` | cuSOLVER QR/SVD error |
| `CuRandError(String)` | cuRAND generation error |
| `StreamError(String)` | CUDA stream error |
| `OutOfMemory(String)` | GPU OOM |
| `ModuleLoadError(String)` | PTX/CUDA module load failure |
| `CuVsError(String)` | cuVS/CAGRA error |
| `LibraryNotFound(String)` | Runtime library missing (libcuvs.so) |

## Python API (`pyscx`)

> [!NOTE]
> For the most up-to-date function signatures and type annotations, see the
> [auto-generated Python API reference](python_api.rst) built from the Rust
> docstrings via autodoc.

### Module-level functions

- `pyscx.open(path) -> PyExperiment` — Open SCX file (local)
- `pyscx.from_anndata(adata, path, codec=None, shard_size=None, in_place=False, csc="off", csc_cols_per_shard=5000, uns_format="tagged", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", force_legacy_metadata=False, memory_budget=None, shard_target_rows=None)` — Write AnnData to SCX.
  Persists `X`, `obs`, `var`, `layers`, `obsm`, `varm`, `uns`, and the sparse
  pairwise slots `obsp` / `varp`. Pairwise matrices are stored as float32 COO
  Arrow IPC; higher-precision inputs are downcast on write. `uns_format`
  selects how `adata.uns` is serialized — see [`uns` serialization](#uns-serialization).
  Accepts backed AnnData (`sc.read_h5ad(path, backed='r')`) and auto-routes
  to the streaming converter — see `pyscx.from_h5ad` below for the
  underlying mechanics. `index_*` / `bitmap` materialise query
  predicate indexes and detection bitmaps at conversion time — see
  [Conversion-time predicate indexes and detection bitmaps](#conversion-time-predicate-indexes-and-detection-bitmaps).
  `force_legacy_metadata=True` forces a single `ObsMetadata` /
  `VarMetadata` section regardless of size; the default (`False`)
  emits `ObsMetadataShard` / `VarMetadataShard` sections when
  `n_obs > shard_target_rows`. `memory_budget` (`"4G"`, `"512M"`,
  bytes) emits `MappingPeakFootprintHigh` when an individual mapping's
  estimated footprint exceeds the budget. `shard_target_rows` overrides
  the default obs shard size. Obsm, varm, obsp, and varp are extracted
  and written one key at a time (incremental, not collected).
- `pyscx.from_h5ad(path, out, codec=None, shard_size=None, csc="off", csc_cols_per_shard=5000, uns_format="tagged", stream=True, strict_uns=False, dense_zero_epsilon=0.0, memory_budget=None, temp_dir=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4, obs_override=None, var_override=None, uns_override=None)` — Stream an h5ad file directly to SCX without materialising `X` in Python or Rust.
  Bounded peak memory: `shard_target_rows × n_vars × density × ~16` bytes
  per X shard, plus `shard_target_rows × k × 4` bytes per `obsm` / `varm` /
  `obsp` / `varp` matrix (each is now hyperslab-read and emitted as
  row-sharded sections — see [§ Sharded obsm/varm/obsp/varp in the
  format spec](format.md#sharded-layout-section-types-20-23)), plus the
  always-resident `indptr` (`(n_obs + 1) × 8` bytes). Recommended
  entry point for files larger than RAM. `csc="always"` performs a
  two-pass write (streaming CSR → `rebuild_csc_inplace` on the
  finished file) — peak disk briefly reaches ~2× the output size during
  the rebuild. `uns_format` is a no-op for the on-disk `uns` read but
  controls the envelope shape applied to `uns_override` when supplied
  (`"tagged"` default wraps NumPy / pandas containers in `__scx_type__`
  envelopes for bit-exact round-trip; `"plain"` collapses them to JSON
  primitives). Internally bypasses
  `anndata.read_h5ad` entirely — obs/var/uns are read via pure-Rust
  HDF5, so callers that don't supply overrides also dodge the eager
  `obsm` materialisation that `anndata.read_h5ad(path, backed='r')`
  performs (anndata reads `obsm` into Python heap on every call,
  including in backed mode).
  - `obs_override`, `var_override`, `uns_override` (optional): supply a
    pandas DataFrame (obs/var) or Python dict (uns) to use in place of
    the on-disk values. Intended for read-mutate-write flows where the
    caller wants to add annotations without paying the full `obsm`
    allocation that `anndata.read_h5ad` would trigger. Typically paired
    with `pyscx.read_h5ad_metadata(path)` (below): fetch on-disk obs /
    var / uns cheaply, mutate them, pass them back. `obs_override.shape[0]`
    must equal n_obs on disk; `var_override.shape[0]` must equal n_vars
    on disk. `uns_override` replaces the entire `uns` section (not a
    merge). Any override with `stream=False` raises `ValueError`
    because the non-streaming path does not apply overrides. `obsm` /
    `varm` / `obsp` / `varp` are intentionally not exposed as overrides
    — accepting them would re-introduce the OOM class this API exists
    to avoid; mutate them via `pyscx.from_anndata(backed_adata, ...)`
    instead if needed.
  - Source layout: CSR streams natively. Dense `/X`
    streams via row-slab sparsification — set `dense_zero_epsilon` to
    threshold near-zero values (default `0.0` matches scipy's
    `csr_matrix(dense)`). CSC-on-disk uses an in-memory transpose
    when the file fits `memory_budget`, otherwise an external
    bucketed transpose to `temp_dir` (scipy `sum_duplicates` semantics
    on duplicate coordinates).
  - `strict_uns=True`: raise on the first unrepresentable `uns`
    entry; default `False` emits a `UserWarning` per skipped key
    (`SkippedUnsKey`). Other structured warnings: `InferredEncoding`
    (h5ad encoding-type missing/ambiguous), `DenseSparsified`,
    `DuplicateCoordinatesMerged`. See
    [Conversion warnings](#conversion-warnings-convertwarning).
  - `memory_budget`: caps dense slabs and the CSC external-transpose
    buffers. Accepts an int byte count or a binary-prefixed size —
    `K`/`M`/`G`/`T` or `KiB`/`MiB`/`GiB`/`TiB` (powers of 1024); decimal
    `KB`/`MB`/`GB`/`TB` is rejected to avoid 1000-vs-1024 ambiguity
    (see [Memory budgets](#memory-budgets)). E.g. `"4G"` / `"512M"` / `"2GiB"`.
  - `stream=False` falls back to the materialising path (kept for
    parity / debugging).
  - `obsm` / `varm` / `obsp` / `varp` on the input are hyperslab-read
    one row-range at a time and emitted as row-sharded sections
    (`<section>/<name>_shard_<idx>`); peak memory per matrix is
    bounded by one shard's worth of rows.
  - `reader_threads`: streaming reader worker count.
    `None` (default) resolves to `RAYON_NUM_THREADS` if set, else
    `os.cpu_count()`. `1` forces the sequential coordinator. `> 1`
    requests rayon workers; output is byte-identical to sequential.
    Requires a thread-safe libhdf5 build (conda-forge default); falls
    back to sequential with a `Hdf5NotThreadsafe` warning emitted at
    most once per process otherwise. `--memory-budget` derates the
    granted count to fit a per-worker estimate; the estimate is
    delegated to the reader: sparse readers assume density 5 % (RNA
    and general) or 10 % (ATAC), times `n_vars × 16 B/nnz`; the
    dense reader sizes the dense slab buffer
    (`shard_target_rows × n_vars × sizeof(dtype) × 2`). When the
    dense reader's `memory_budget`-derived slab cap is tighter than
    `shard_target_rows`, the parallel coordinator silently clamps
    its partition to that cap (matching the sequential path).
  - `writer_queue_depth`: backpressure window between the parallel
    encoder pool and the ordered writer. Default 4. The parallel
    coordinator caps outstanding shards (encoding + in channel + in
    reorder buffer) at `reader_threads + writer_queue_depth` via a
    rolling-window spawn, so peak RSS scales with that sum, not with
    the total shard count. Larger values give a slow shard a deeper
    look-ahead buffer; smaller values risk starving encoders when
    one shard takes much longer than its siblings.
- `pyscx.read_h5ad_metadata(path, strict_uns=False) -> H5adMetadata` —
  Read just `obs`, `var`, `uns`, and the X shape from an h5ad file via
  pure-Rust HDF5 readers. Skips `anndata.read_h5ad` (and therefore
  anndata's eager `obsm` allocation) entirely. Returns an
  `H5adMetadata` object with attributes `obs` (`pandas.DataFrame`),
  `var` (`pandas.DataFrame`), `uns` (`dict`), `n_obs` (`int`),
  `n_vars` (`int`), `x_format` (`"csr"` / `"csc"` / `"dense"`). Intended
  for read-mutate-write flows: read this, mutate `obs` / `uns`, pass
  the mutated values back via `pyscx.from_h5ad(..., obs_override=, uns_override=)`.
  Categoricals, pandas Index metadata, and nullable-boolean columns
  round-trip through the same Arrow IPC path that `pyscx.open(...).to_anndata()`
  uses, so the result is semantically equivalent to the obs / var that
  `anndata.read_h5ad` would have returned — without the obsm allocation
  cost. `strict_uns=True` mirrors `from_h5ad`'s strict-uns semantics.
- `pyscx.from_h5mu(path, out, codec=None, shard_size=None, csc="off", csc_cols_per_shard=5000, stream=True, strict_uns=False, memory_budget=None, temp_dir=None, modalities=None, modality_types=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4)` — Stream an h5mu file to a multimodal SCX v2 file. Mirrors `from_h5ad` for h5mu inputs; per-modality `n_vars`/`nnz` come from `/mod/{name}/X` attributes so there is no pre-pass materialisation. `reader_threads`/`writer_queue_depth` carry the same semantics as `from_h5ad` — each modality runs through the same dispatcher independently.
  - `modalities`: optional list of modality names to keep
    (case-sensitive). Unknown names raise `ValueError` with the
    available list.
  - `modality_types`: optional dict `{name: "rna" | "protein" | "atac"
    | "spatial" | "methylation" | "custom"}`. Modalities not listed
    fall back to name inference and emit `ModalityTypeInferred`.
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None, in_place=False, csc="off", csc_cols_per_shard=5000, uns_format="tagged", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off")` — 10x HDF5 to SCX.
- `pyscx.from_mtx(mtx_dir, scx_path, codec=None, shard_size=None)` — Cell Ranger MTX directory (`matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, `features.tsv[.gz]`) to SCX. Default shard size is 16384.
- `pyscx.to_mtx(scx_path, output_dir)` — SCX to Cell Ranger–style MTX directory (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`).
- `pyscx.to_h5ad(path, out, stream=True, modality=None, reader_threads=None, writer_queue_depth=4, memory_budget=None)` — Stream SCX → h5ad
  without materialising `X` in memory. Mirror of `pyscx.from_h5ad` in the
  opposite direction. Bounded peak memory: one shard's worth of CSR
  plus encode buffers per matrix written, plus the always-resident
  `indptr` (`(n_obs + 1) × 8` bytes). When deletion vectors are
  present, only kept rows appear in the output (`shape[0] = n_obs -
  n_deleted`); a single pre-scan pass computes the filtered nnz before
  pre-allocating the HDF5 triplet so the on-disk layout is
  deterministic. For multimodal SCX files, pass `modality="rna"` to
  extract a single modality as h5ad; otherwise the call raises (use
  `pyscx.to_h5mu`). `stream=False` falls back to the materialising
  path (kept for parity / debugging).
  - `reader_threads`: parallel shard decoder pool. `None` (default)
    auto-resolves to `RAYON_NUM_THREADS` if set, else
    `os.cpu_count()`. `1` forces the sequential coordinator.
    `> 1` requests rayon workers; output is byte-identical to
    sequential. HDF5 writes stay on the calling thread, so this
    knob does **not** require a thread-safe libhdf5 build (unlike
    the ingest direction).
  - `writer_queue_depth`: bounded reorder buffer depth between the
    parallel decoder pool and the ordered HDF5 writer. Default 4.
    Outstanding decoded shards are capped at `reader_threads +
    writer_queue_depth` so a slow shard 0 can't let the buffer
    accumulate the rest of the file.
  - `memory_budget`: `"4G"`, `"512M"`, `"2GiB"`, or bytes — same
    parser as `from_h5ad`. Derates the granted `reader_threads`
    against `max_shard_bytes` (computed exactly from
    `FullCatalogEntry::stats.nnz` and row count, no density
    heuristic). A single shard exceeding the budget raises with an
    actionable message; smaller mismatches emit
    `ReaderThreadsDerated` and proceed with fewer workers.
- `pyscx.to_h5mu(path, out, stream=True, reader_threads=None, writer_queue_depth=4, memory_budget=None)` — Stream a multimodal SCX
  file to h5mu. Iterates each modality and writes
  `/mod/{name}/X` and any `/mod/{name}/layers/{layer}` shard-by-shard;
  global `/obs`, per-modality `/var` / `/obsm`, and `/uns` reuse the
  in-memory metadata writers. Requires `reader.is_multimodal()`;
  single-modality files raise (use `to_h5ad`). `reader_threads`,
  `writer_queue_depth`, and `memory_budget` carry the same
  semantics as `to_h5ad`; each modality runs through the same
  dispatcher independently.
- `pyscx.iter_chunks(adata, chunk_size="shard")` — Shard-aligned or fixed-size chunk iterator
- `pyscx.preprocess(source, target, ops, target_sum=None)` — Streaming shard-by-shard preprocessing
- `pyscx.save_layer(source, target, layer_name, ops, target_sum=None)` — Save transformed data as layer

### File operations
- `pyscx.append(target, input, codec=None, shard_size=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None)` — Streaming append from SCX file (reads one shard at a time; raw-copy fast path when codec/encoding match). `index_*` kwargs rebuild predicate indexes covering all rows post-append — see [Conversion-time predicate indexes and detection bitmaps](#conversion-time-predicate-indexes-and-detection-bitmaps).
- `pyscx.append_from_anndata(target, adata, codec=None, shard_size=None, in_place=False, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None)` — Append from AnnData. Same `index_*` semantics as `append`.
- `pyscx.mark_deleted(path, cell_indices)` — Logical deletion
- `pyscx.compact(input, output, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, reshape_obs=False)` — Rewrite reclaiming space. `index_*` kwargs rebuild predicate indexes against the compacted output. `reshape_obs=True` migrates legacy single-section obs metadata to the sharded `ObsMetadataShard` layout (mirrors `scx compact --reshape-obs`; useful after a backed `from_anndata` conversion).
- `pyscx.rollback(path, to_seq=None)` — Revert to previous manifest
- `pyscx.merge(inputs, output, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, assume_identical_var=False, uns_policy="first", shard_target_rows=None)` — Merge multiple files. `index_*` kwargs rebuild predicate indexes against the merged output — without them, pushdown silently regresses to a full obs scan on the merged file. `assume_identical_var` (default `False`) validates var identity (index, column names, values) across all inputs; set `True` to check only `n_vars` (breaking change from pre-branch where var was unchecked). `uns_policy` controls conflicting uns sections: `"first"` (keep first input), `"require_equal"` (error on difference), `"namespace"` (prefix keys with input filename), `"summary"` (write conflict report as `uns["_merge_uns_summary"]`). `shard_target_rows` overrides the default obs shard size during merge. Merge now streams obs shard-by-shard and builds predicate indexes incrementally from the shard stream.

### Cloud operations (requires `--features cloud`)
- `pyscx.pull(source, dest, filter=None, parallelism=None)` — Streaming cloud → local
- `pyscx.push(source, dest, parallelism=None)` — Streaming local → cloud
- `pyscx.cloud_optimize(input, output=None)` — Front-of-file catalog
- `pyscx.explode(input, output)` — Packed → exploded directory
- `pyscx.pack(input, output)` — Exploded → packed
- `pyscx.open_cloud(url) -> PyCloudExperiment` — Direct cloud reads

### PyExperiment

- `to_anndata(backed=False, cache_shards=4, var_names=None, obs_filter=None, layers=None, preserve_slots=False, modality=None, eager=False, memory_budget=None, obsm=None)` — Convert to AnnData
  - `var_names`: list of gene names to project (column subset)
  - `obs_filter`: predicate string for cell filtering. Non-backed mode uses the scx-engine query parser with shard pushdown; `backed=True` evaluates it with pandas `.query()` (different grammar — see [Filter Expression Compatibility](scanpy.md#filter-expression-compatibility) in docs/scanpy.md)
  - `layers`: list of layer names to load (default: all)
  - `obsm`: list of obsm keys to load (default `None` = all keys, byte-identical to prior behaviour). When set, only the listed embeddings are read — dropping the per-process RAM of unused keys on the random-access dataloader path. An unknown key raises `KeyError`; `obsm=[]` loads no embeddings. Selecting keys also changes *how* obsm is materialised (see `obsm` loading modes under `eager`, below).
  - `backed`: when True, X and layers are lazy `ScxBackedSparseDataset` instances
  - `modality`: select one modality of a multimodal file and
    return a backed AnnData scoped to that modality (per-modality X /
    var / obsm; global obs shared). Currently requires `backed=True`;
    incompatible with `var_names` / `obs_filter` / `layers` (use
    `scx subset --modality NAME --filter` to pre-materialise a
    filtered single-modality file).
  - `eager` (default `False`): when `False`, the `obsp` / `varp` /
    `varm` slots (plus `layers` in non-backed mode) are returned as
    lazy bridges — `ScxLazyPairwiseMapping` / `ScxLazyVarmMapping` /
    `ScxLazyLayersMapping` — that decode each entry on first
    access. Keeps the peak RSS of `to_anndata()` itself bounded for
    files that carry large kNN graphs or embeddings; user code that
    accesses these slots pays the same one-time decode cost it would
    otherwise pay at construction time. Bridges expose the full
    `MutableMapping` protocol (`__getitem__` / `__contains__` /
    `__iter__` / `__len__` / `keys` / `items` / `values` / `get` /
    `__setitem__` / `__delitem__`); mutations stay in memory and are
    not written back. Pass `eager=True` to materialise everything up
    front and detach the returned AnnData from the SCX file handle
    (use this before closing the experiment or shipping the AnnData
    to a subprocess). `uns` is always eager regardless of this flag.
  - **`obsm` loading modes** (selected by the combination of `obsm`,
    `backed`, `eager`, `obs_filter`):
    - `obsm=None` (default): every obsm key is materialised eagerly as
      a dense numpy array — byte-identical to prior behaviour.
    - `obsm=[...]`, `eager=True`, **or** `obs_filter` set: the selected
      keys are materialised eagerly (selective eager).
    - `obsm=[...]`, `backed=False`, `eager=False`: obsm becomes a lazy
      `ScxLazyObsmMapping` bridge — each selected key is decoded to a
      dense numpy array on *first* access (`adata.obsm[key]`), so an
      unused/late key costs nothing. Same `MutableMapping` protocol as
      the other bridges.
    - `obsm=[...]`, `backed=True`, `eager=False`, no `obs_filter`:
      obsm becomes a `ScxLazyObsmMapping` whose values are
      `ScxBackedObsmDataset` — a shard-aware **dense row-gather**
      dataset. `m[idx]` / `m[idx_array]` decode only the touched
      `ObsmEmbeddingShard`s (bounded per-key LRU = `cache_shards`), so
      per-access memory is `O(batch × n_cols)` and independent of
      `n_obs`. This is the scalable path for a single huge embedding
      (e.g. millions of cells × thousands of dims) on the random-access
      `embed_key` dataloader. Under `obs_filter` the backed-obsm path
      falls back to eager obsm (composing a pandas-query row mask with
      shard gather is deferred). Deletion vectors compose via the same
      `kept_to_global` remap as `X`. `ScxBackedObsmDataset` is registered
      as an `anndata.abc.CSRDataset` virtual subclass so AnnData's `obsm`
      coercion accepts it on public `adata.obsm[key]` access — it is
      nonetheless **dense** (`m[idx]` / `np.asarray(m)` / `m.toarray()`
      return dense numpy; it has no `.tocsr()`).
  - `memory_budget` (default `None`, treated as 8 GiB): emits
    `EagerAssemblyMemoryHigh` `UserWarning` when the estimated eager
    footprint exceeds the budget. Warn-only — does not block
    assembly.
  - Returns `obsm` (dense), `varm` (dense), `obsp` (scipy CSR), and
    `varp` (scipy CSR) when present in the file. `obsp` / `varp` are
    not subject to deletion-vector row filtering — when cells are
    logically deleted, the pairwise matrices still cover the full
    original axis; `compact` resolves this by rebuilding from scratch.
- `to_mudata(backed=False, cache_shards=4)` — Materialise a multimodal file as `mudata.MuData`
  - Eager (`backed=False`): per-modality scipy CSR AnnData sharing the
    global obs (existing behaviour). Raises on single-modality files.
  - **Backed (`backed=True`)**: per-modality
    `ScxBackedSparseDataset` AnnData sharing the global obs.
    Single-modality files are wrapped in a one-modality MuData rather
    than raising, so the call works uniformly across layouts.
- `query() -> PyQueryPipeline` — Start lazy query pipeline
- `mark_deleted(mask)` — Delete cells matching boolean array
- `validate()` — Check checksums, returns list of `(section_name, passed)`
- `detection_counts(axis="var", modality=None) -> np.ndarray`
  — Per-gene non-zero counts. Reads `BitmapShard` sidecars when
  present (one roaring decode per shard); falls back to a CSR scan
  otherwise. `axis="obs"` returns per-cell gene counts. For
  multimodal v2 files, pass `modality="rna"` to scope the result.
- `cells_expressing(gene, modality=None) -> np.ndarray` —
  Indices of cells with non-zero expression for `gene` (name or
  integer). Bitmap fast path when sidecars exist; CSR fallback
  otherwise.
- Properties: `n_obs`, `n_vars`, `nnz`, `shard_count`, `format_version`, `codec_id`, `layer_names`

### `uns` serialization

`adata.uns` is written into the `UnsBlob` section (id 10) as JSON. The
`uns_format` kwarg on `from_anndata()` / `from_10x()` selects the envelope:

| Mode | Default | NumPy ndarray | NumPy scalar | tuple | pandas Cat/Index/Series | NaN/Inf in array |
| --- | --- | --- | --- | --- | --- | --- |
| `"tagged"` | ✓ | `__scx_type__: "ndarray"` envelope (base64-LE bytes for numeric; JSON string list for object/string; `__scx_type__: "recarray"` for structured) | `__scx_type__: "scalar"` envelope (1-element base64) | `__scx_type__: "tuple"` envelope | `__scx_type__: "categorical" / "pandas.Index" / "pandas.Series"` envelope preserving name/codes/categories/ordered | preserved bit-exact (numeric path) |
| `"plain"` |  | collapses to nested list (`.tolist()`) | collapses to Python scalar | collapses to JSON array | collapses to JSON array via `.tolist()` | raises `ValueError` |

The reader **auto-detects** per value: dicts with a `__scx_type__` key are
decoded back to their original Python type; everything else passes through
as plain JSON. This means files written by older `pyscx` (or with
`uns_format="plain"`) read identically on a modern build, and modern
tagged files are forward-compatible — unknown future tags emit a
`UserWarning` and return the raw envelope dict for inspection.

Unsupported in both modes: `bytes` objects (no portable JSON
representation) and `datetime64` / `complex` / `timedelta` ndarray dtypes.
Non-finite raw Python `float` *scalars* (outside an ndarray) still raise.

The tagged envelope is plain JSON, so the section can be inspected with
any JSON tool. Example for a `float32` array:
```json
{
  "pca_variance": {
    "__scx_type__": "ndarray",
    "dtype": "float32",
    "shape": [50],
    "encoding": "base64le",
    "data": "zczMPc3MTD4AAIA/..."
  }
}
```

### PyQueryPipeline

- `filter_obs(expr)` / `filter_var(expr)` — Predicate filtering
- `select_genes(indices)` — Gene projection
- `with_normalize(target_sum=1e4)` — Total-count normalization
- `with_log1p()` — Log1p transformation
- `limit(n)` — Row limit
- `collect() -> PyQueryResult` — Execute pipeline
- `count() -> int` — Convenience: matching cell count

### PyQueryResult

- `to_anndata()` — Convert result to AnnData (zero-copy CSR)
- `to_csr()` — Return just the scipy CSR matrix
- Properties: `n_obs`, `n_vars`, `nnz`, `skipped_shards`, `total_shards`

### PyCloudExperiment

Returned by `pyscx.open_cloud()`. Cloud-hosted SCX handle. Supports
metadata accessors plus a cloud-native query path served over
`object_store` range reads; full `to_anndata()` and `validate()` still
require `pyscx.pull()` to materialise the file locally.

- `n_obs` `→ int` — Number of observations (cells)
- `n_vars` `→ int` — Number of variables (genes)
- `nnz` `→ int` — Total non-zero entries
- `shard_count` `→ int` — Number of CSR shards in the file
- `format_version` `→ int` — SCX format version
- `codec_id` `→ int` — Default codec ID
- `query() → PyQueryPipeline` — Start a lazy cloud query. Backed by
  the same `QueryPipeline` as `pyscx.open(...).query()`, wired over a
  `CloudReader`-backed `SectionReader`. Predicate pushdown uses the
  catalog (and predicate indexes when present); only matching shards
  are range-read from object storage. Example:

  ```python
  adata = (
      pyscx.open_cloud("gs://bucket/atlas.scxd/")
            .query()
            .filter_obs("cell_type == 'T cell'")
            .select_genes(hvg)
            .collect()
            .to_anndata()
  )
  ```

  Deferred follow-ons: `CloudQueryOptions` (parallelism,
  max-inflight bytes, cache-dir, retry policy), the
  `pyscx.read_cloud(...)` flat helper, and a batched async section
  fetcher (current cloud reads block per shard from the rayon worker).

### pyscx.accel — Rust-Native Accelerators

All accelerators write results to standard AnnData slots (same as scanpy), so downstream functions work identically.

#### `prefer_format="csr"|"csc"` kwarg

Several accelerators take an explicit `prefer_format` kwarg that selects
between the row-major CSR path (default) and the column-major CSC sidecar
path. See [scanpy.md § prefer_format](scanpy.md#prefer_formatcsrcsc-explicit-column-major-dispatch)
for the full dispatch rules and requirements.

##### CSC decision matrix

CSC sidecars are the column-major substrate for column (gene-axis)
algorithms. Build one at conversion time with `csc="auto"` / `csc="always"`
(`pyscx.from_anndata` / `from_h5ad` / `from_10x`) or `scx convert --csc=auto`,
or after the fact with `scx build-csc`. `csc="auto"` builds a sidecar only
when the dataset is large enough to benefit — `n_obs ≥ 50000` **and**
`n_vars ≥ 5000` by default, tunable via `SCX_CSC_AUTO_OBS_THRESHOLD` /
`SCX_CSC_AUTO_VARS_THRESHOLD`.

| Op | `supports_csc` | Default format | GPU-fast with CSC | Notes |
|----|:--:|:--:|:--:|-------|
| `pdex_ref` | ✅ | CSR | ✅ (`gpu_csc_v3`) | CSC-direct GPU route by default when a CSC sidecar is present; in-memory CSR falls back to `gpu_csr_v3`. |
| `rank_genes_groups` (Wilcoxon) | ✅ | CSR | ✅ (`gpu_csc_v3`) | CSC-direct GPU route by default when a CSC sidecar is present; in-memory CSR falls back to `gpu_csr_v3`. CPU CSC kernel via `prefer_format="csc"`. |
| `rank_genes_groups_df` | ✅ | CSR | ✅ (`gpu_csc_v3`) | Same Wilcoxon engine as above. |
| `pseudobulk_dex` | ✅ (gene subset) | CSR | N/A (CPU + pydeseq2) | CSC requires a gene subset (`gene_indices` or `col_projection`); full-gene CSC has no win. |
| `highly_variable_genes` (seurat_v3) | ✅ (single-batch) | CSR | ❌ | CSC routes single-batch seurat_v3; multi-batch / GPU / other flavors raise on CSC. |
| `calculate_qc_metrics` | ✅ (gene axis) | CSR | N/A | Gene-axis aggregation uses CSC; cell-axis stays CSR. |
| `col_sums` / `col_nnz` / `col_min` / `col_max` / `col_var` | ✅ | CSR | N/A | Column reductions; CSC requires no row deletion vector. |
| `pca` | ❌ (rejects CSC) | CSR | N/A | Inherently row-major; `prefer_format="csc"` raises `ValueError`. |
| `neighbors` / `umap` / `leiden` | N/A | — | N/A | Operate on PCA embeddings / kNN graphs, not on `X`. |

"GPU-fast with CSC" means the op reaches peak GPU throughput **only** with a
backed SCX file that has a CSC sidecar — see
[scanpy.md § GPU-supported vs GPU-fast](scanpy.md#gpu-supported-vs-gpu-fast).
Confirm which path actually ran via the [route metadata](#accelerator-route-metadata).

- `pyscx.accel.pca(adata, n_comps=50, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto")` — Randomized SVD PCA with streaming SpMM. Writes `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]`. On GPU: cuSPARSE SpMM + cuSOLVER QR (f32).
- `pyscx.accel.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0, ef_construction=200, ef_search=200, device="auto")` — kNN graph + UMAP-style connectivities. CPU: HNSW. GPU: CAGRA (cuVS). Writes `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`.
- `pyscx.accel.pca_neighbors(adata, n_comps=50, n_neighbors=15, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", use_rep="X_pca", prefer_format="csr")` — Fused PCA → kNN in one call. On a GPU host with cuSPARSE 12.5+ and cuVS, the PCA embedding stays GPU-resident and feeds straight into CAGRA (no `X_pca` host round-trip), recording route `gpu_device_resident`; otherwise it falls back to sequential `pca` + `neighbors`. Writes the union of both ops' slots (`obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]`, `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`). See [docs/scanpy.md § Fused PCA → kNN](scanpy.md#fused-pca--knn-pyscxaccelpca_neighbors).
- `pyscx.accel.umap(adata, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, learning_rate=1.0, random_state=0, device="auto")` — Spectral-init SGD UMAP. GPU: native CUDA kernel or cuML fallback. Writes `obsm["X_umap"]`.
- `pyscx.accel.rank_genes_groups(adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, log_transformed=False, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr")` — Parallel Wilcoxon rank-sum with BH correction. Writes `uns["rank_genes_groups"]`, or returns DataFrame when `stratify_by` is set. `prefer_format="csc"` routes per-chunk reads through the column-major sidecar (see kwarg docs above). The execution route is recorded on `uns["rank_genes_groups"]["scx_accel_route"]` and `uns["scx_accel"]["rank_genes_groups"]` (see [Accelerator route metadata](#accelerator-route-metadata)).
- `pyscx.accel.pseudobulk_dex(adata, groupby, test_col, reference, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr", gene_indices=None)` — Streaming pseudobulk aggregation (Rust) + pydeseq2 testing. Returns DataFrame. Requires optional `pydeseq2` dependency. `prefer_format="csc"` requires a gene subset (either an explicit `gene_indices` argument or a `col_projection` already set on `adata.X`); full-gene CSC pseudobulk has no measurable speed-up.
- `pyscx.accel.rank_genes_groups_df(adata, groupby, reference="rest", n_genes=None, gene_chunk_size=None, rankby_abs=False, tie_correct=False) → polars.DataFrame` — Same Wilcoxon as `rank_genes_groups()` but returns a polars DataFrame in cell-eval's `DEResults` schema: `(target, feature, fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change)`. Ready to feed into `cell_eval.initialize_de_comparison()`.
- `pyscx.accel.pdex_ref(adata, groupby, *, reference="non-targeting", is_log1p=None, geometric_mean=True, epsilon=0.0, gene_chunk_size=None, prefer_format="csr", device="auto") → polars.DataFrame` — Perturbation-screen differential expression: Mann–Whitney U + pseudobulk geometric-mean log fold change vs a single reference group. Pinned bit-for-bit to upstream [`pdex`](https://github.com/ArcInstitute/pdex) (`pyscx/tests/test_pdex_ref_parity.py`). Returns one row per (target group, gene) excluding the reference group, with columns `target`, `feature`, `target_mean`, `ref_mean`, `target_membership`, `ref_membership`, `log2_fold_change` (also exposed as `fold_change` for migration), `percent_change`, `p_value`, `statistic`, `fdr`. Note: here `fold_change` is a deprecated alias of `log2_fold_change` (log2 scale) — this differs from `rank_genes_groups_df`'s `fold_change` column, which is linear (`exp2` of the log2 value). `is_log1p=None` auto-detects via `adata.uns["log1p"]` + a max-value heuristic; pass `True`/`False` to override. `device="auto"` picks GPU when available and falls back to CPU otherwise. **GPU v3-CSC path (default):** the GPU dispatch routes through a CSC-direct driver that drops the per-chunk dense intermediate and uses a CSC shard source with pipelining + per-chunk shard-range pre-filter. This is the default GPU DE route (the former `SCX_GPU_DE_V3` opt-in gate was removed when v3 became the unconditional default). CSC-direct requires a CSC sidecar on the SCX file — build it with `pyscx.from_anndata(adata, path, csc="always")` or `scx convert --csc=always`. In-memory inputs (scipy CSR) and files without a sidecar fall back to the v3-CSR-direct path automatically. The route that actually ran is recorded on `adata.uns["scx_accel"]["pdex_ref"]` (see [Accelerator route metadata](#accelerator-route-metadata)) — `gpu_csc_v3` confirms the CSC-direct path, `gpu_csr_v3` + `fallback_reason="no_csc_sidecar"` confirms the CSR fallback. Wall-time numbers and the disposition live in [`docs/performance.md` § Per-operation timing](performance.md#per-operation-timing).
- `pyscx.accel.pseudobulk_means(adata, groupby, min_cells_per_group=1) → (ndarray, list[str])` — Group-by mean on sparse X, streaming shard-by-shard (works on backed, lazy, scipy CSR, or dense). Returns `(means[P, G] float64, sorted group names)`. Foundation for the perturbation evaluation metrics below.
- `pyscx.accel.perturbation_metrics(adata_real, adata_pred, pert_col="perturbation", control="control", metrics=None, min_cells_per_group=1) → dict[str, dict[str, float]]` — Bundled bulk metrics `{pearson_delta, mse, mae, mse_delta, mae_delta}` between paired real/pred AnnData. Matches cell-eval's metrics within atol=1e-6.
- `pyscx.accel.energy_distance(adata_real, adata_pred, pert_col="perturbation", control="control", metric="euclidean", embed_key=None, backend=None, dtype=None) → float` — Pearson correlation of per-perturbation e-distance vectors (real vs pred). Avoids `[N, N]` distance materialization (per-row sum reduction even on the gemm path); precomputes control self-distance once; rayon-parallel across perturbations. `backend ∈ {"auto" (default), "gemm", "scalar"}` — `"auto"` picks faer-dispatched gemm for euclidean/cosine and the scalar row-by-row path for L1; `"gemm" + metric="l1"` raises `RuntimeError` (no decomposition exists). `dtype ∈ {"f32" (default), "f64"}` controls only the matmul / per-pair arithmetic precision; reductions always accumulate in `f64`. f32 + gemm matches f64 + scalar within `atol=1e-4` correlation / `atol=1e-3` per-pert.
- `pyscx.accel.energy_distance_details(...)` — Same signature (including `backend` / `dtype`) as `energy_distance` but returns `{"correlation": float, "d_real": {pert: float}, "d_pred": {pert: float}, "pert_names": [...]}`.
- `pyscx.accel.discrimination_score(adata_real, adata_pred, pert_col="perturbation", control="control", metric="l1", exclude_target_gene=True, embed_key=None, min_cells_per_group=1) → dict[str, float]` — Per-perturbation normalized rank of the predicted perturbation effect's distance to the correct real effect. `metric ∈ {"l1", "l2"/"euclidean", "cosine"}`. `exclude_target_gene=True` drops the gene matching each perturbation's name from the distance (matches cell-eval's default).
- `pyscx.accel.knockdown_efficiency(adata, pert_col="perturbation", control="control", eps=1e-8)` — Per-cell knockdown efficiency + log-fold change vs control baseline. Input must be normalized (NOT log1p'd); log1p is applied internally. Writes `adata.obs["KnockDownEfficiency"]` and `adata.obs["KnockDownGeneFC"]` (both float32, NaN for control cells and cells whose perturbation name isn't in `var_names`). Matches `arc_bench.tools.normalize_transform.core` within atol=1e-6.
- `pyscx.accel.clustering_agreement(adata_real, adata_pred, pert_col="perturbation", control="control", metric="ami", real_resolution=1.0, pred_resolutions=None, n_neighbors=15, embed_key=None, min_cells_per_group=1) → float` — Builds perturbation-centroid kNN graphs, sweeps Leiden resolutions, scores best real-vs-pred agreement via AMI / NMI / ARI. **All-native-Rust** — no scanpy / anndata / igraph dispatch; uses `scx_accel::neighbors::build_knn_graph` (HNSW via `instant-distance`, `ef_construction=200, ef_search=50, seed=0`) plus `scx_accel::leiden` sequential mode (`max_iterations=2, parallel=false, seed=0` — matches scanpy's `flavor="igraph", n_iterations=2`). Pred-side kNN graph built once and reused across the resolution sweep; whole hot path runs under `py.allow_threads`. Matches cell-eval's `ClusteringAgreement` within `atol=0.15` aggregate (stochastic Leiden; exact score match not expected — algorithms agree exactly on graphs with `n_perts ≥ 16`).
- `pyscx.accel.adjusted_mutual_info(labels_a, labels_b) → float` — AMI on integer label arrays (arithmetic-mean convention). Matches `sklearn.metrics.adjusted_mutual_info_score` within atol=1e-10.
- `pyscx.accel.normalized_mutual_info(labels_a, labels_b) → float` — NMI (arithmetic-mean). Matches `sklearn.metrics.normalized_mutual_info_score` within atol=1e-10.
- `pyscx.accel.adjusted_rand_index(labels_a, labels_b) → float` — ARI rescaled to `[0, 1]` via `(ARI + 1) / 2` (cell-eval convention). For the raw sklearn ARI (in `[-0.5, 1]`), compute `2 * adjusted_rand_index(a, b) - 1`.
- `pyscx.accel.leiden(adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=-1, device="auto")` — Leiden community detection on kNN graph. Reads `obsp["connectivities"]` (from `neighbors()`). GPU: cuGraph Leiden (up to 47× faster). CPU fallback: leidenalg via igraph. Writes `adata.obs[key_added]` (categorical) and `adata.uns["leiden"]` (params + backend metadata). GPU and CPU may produce different partitions due to algorithmic differences; compare via ARI/NMI.
- `pyscx.accel.harmony_integrate(adata, key, *, basis="X_pca", adjusted_basis=None, n_clusters=None, theta=None, sigma=0.1, lamb=None, alpha=0.2, max_iter=10, max_iter_kmeans=4, epsilon_harmony=1e-2, epsilon_kmeans=1e-3, block_size=0.05, batch_prop_cutoff=1e-5, tau=0.0, random_state=0, device="auto")` — Clean-room Rust implementation of Harmony2 (Korsunsky et al., 2019). Soft k-means clustering with a diversity penalty, followed by ridge-regression correction of `adata.obsm[basis]` (`"X_pca"` by default). `key` accepts a single `obs` column name or a list for multi-covariate batch correction; each is factorised via `pandas.factorize(sort=False)`. Writes corrected embedding (f32, N × d) to `adata.obsm[adjusted_basis or basis]` and convergence metadata to `adata.uns["harmony"]` (`params`, `converged`, `n_iterations`, `objective_harmony`, `backend`). Parameter names match `scanpy.external.pp.harmony_integrate`, so existing scanpy pipelines can swap in. GPU path (when built with `--features gpu`) accelerates distance / L2-norm / batched scatter-subtract kernels; k-means++ init and covariance inversion stay on CPU. Numerical parity vs R `harmony` v2.x: mean per-PC Pearson r 0.989–0.999 on the three validation fixtures.
- `pyscx.accel.compute_lisi(adata, key, *, basis="X_pca", perplexity=30.0, n_neighbors=None) → np.ndarray` — Local Inverse Simpson Index on an `obsm` embedding. Exact brute-force kNN (matches R `FNN::get.knn`) + per-cell Gaussian-bandwidth search (t-SNE Hbeta routine) + Simpson index over kernel-weighted neighbour category probabilities. Returns LISI vector of length N and also writes to `adata.obs[f"lisi_{key}"]`. Values near 1 → poor mixing (neighbourhoods dominated by one category); values approaching the number of categories → uniform mixing. `n_neighbors` defaults to `ceil(3 × perplexity)`. ~10× faster than R `lisi::compute_lisi` on D1–D4 with mean-LISI agreement within 0.8–2.4 %.
- `pyscx.accel.normalize_total(adata, target_sum=10000.0)` — Materialization-free row normalization. On `ScxBackedSparseDataset`: computes row sums via streaming, creates `ScxLazyTransformedDataset` wrapper. On `ScxLazyTransformedDataset`: appends `NormalizeTotal` transform to chain. On scipy CSR: delegates to `sc.pp.normalize_total()`.
- `pyscx.accel.log1p(adata)` — Materialization-free log1p. On `ScxBackedSparseDataset`: creates `ScxLazyTransformedDataset` with `Log1p` transform. On `ScxLazyTransformedDataset`: appends `Log1p` to chain (fuses with preceding `NormalizeTotal` when possible). On scipy CSR: delegates to `sc.pp.log1p()`.
- `pyscx.accel.gpu_info() → dict` — Query GPU device info: `{'device': ..., 'total_vram_gb': ..., 'free_vram_gb': ...}`. Returns `None` if no GPU available.
- `pyscx.accel.estimate_gpu_memory(adata, operation, **kwargs) → dict` — Estimate GPU VRAM required for an operation. Returns `{'required_gb': float, 'fits_in_vram': bool}`. Supported operations:
  - `"pca"`: kwargs `n_components` (default 50), `n_oversamples` (default 10), `shard_size` (default 16384)
  - `"knn"`: kwargs `n_neighbors` (default 15), `n_dims` (default 50)
  - `"umap"`: kwargs `n_components` (default 2)
  - `"leiden"`: no additional kwargs

  Raises `ValueError` for unknown operations. Note: estimates are approximate — cuSOLVER QR workspace may be undercounted by ~1.5×.
- `pyscx.accel.calculate_qc_metrics(adata, qc_vars=None, log1p=True, inplace=True, prefer_format="csr")` — Streaming QC metrics for backed/lazy data without materialization. Computes per-cell `n_genes_by_counts`, `total_counts` and per-gene `n_cells_by_counts`, `total_counts`. Supports `qc_vars` for gene subsets (e.g., `["mt"]` for mitochondrial percentage). When `inplace=True`, writes to `adata.obs`/`adata.var`; when `False`, returns `(obs_df, var_df)`. `prefer_format="csc"` routes the gene-axis aggregation through the CSC sidecar (cell-axis stays CSR — row aggregations have no CSC win). Falls back to `sc.pp.calculate_qc_metrics()` for scipy/dense.
- `pyscx.accel.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", batch_key=None, span=0.3, subset=False, n_bins=20, device="auto", prefer_format="csr", layer=None)` — Streaming HVG selection. Default CPU + CSR streams `mean_var` and clipped sums shard-by-shard via `ShardSource`; multi-batch runs CSR. Works on `ScxBackedSparseDataset`, `ScxLazyTransformedDataset`, **and a materialized scipy/dense `X`** for `flavor` in `seurat_v3` / `seurat_v3_paper` / `seurat` (a materialized `X` is wrapped in a single-shard `ShardSource`), so the eager `to_anndata()` idiom gets the same numerics and the same per-batch LOESS-singularity tolerance as the backed path. Only flavors scx does not implement natively (e.g. `cell_ranger`) delegate to `scanpy.pp.highly_variable_genes` (one-shot `UserWarning`). `prefer_format="csc"` routes single-batch seurat_v3 through `streaming_mean_var_csc` and `streaming_clip_square_sum_csc` (multi-batch + GPU + non-seurat_v3 flavors raise on CSC). Writes `var["highly_variable"]`, `var["means"]`, `var["variances"]`, `var["variances_norm"]`, `var["highly_variable_rank"]`.
- `pyscx.accel.col_sums(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column sums on `ScxBackedSparseDataset`. Honors `col_projection` and `kept_to_global` on the CSR path; CSC dispatch requires no row deletion vector and (currently) only supports `ScxBackedSparseDataset` and `ScxLazyTransformedDataset` (CSR scipy / dense raises a helpful message — use the array-protocol `dataset.sum(axis=0)` for those).
- `pyscx.accel.col_nnz(dataset, prefer_format="csr") → np.ndarray (i64)` — Streaming per-column NNZ. Same dispatch as `col_sums`.
- `pyscx.accel.col_min(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column min. Implicit-zero correction (`mins[c] = min(mins[c], 0.0)` when `col_nnz[c] < n_obs`) applied on both paths.
- `pyscx.accel.col_max(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column max. Symmetric implicit-zero correction.
- `pyscx.accel.col_var(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column variance. CSR uses a two-pass formulation; CSC uses single-pass `(sum_x² - n·mean²) / n`. Numerically equivalent within f64 epsilon (verified by `pyscx/tests/test_csc_dispatch.py`).
- `pyscx.accel.filter_cells(adata, min_genes=None, max_genes=None, min_counts=None, max_counts=None)` — Non-materializing cell QC filter for backed/lazy data. Computes row NNZ and/or row sums via streaming, builds a boolean mask, and updates the deletion vector (`kept_to_global`) on `ScxBackedSparseDataset` or `ScxLazyTransformedDataset`. Also slices `adata.obs`, `adata.obsm`, and updates `adata.layers` with the new deletion vector. Falls back to `sc.pp.filter_cells()` for scipy/dense.
- `pyscx.accel.filter_genes(adata, min_cells=None, max_cells=None, min_counts=None, max_counts=None)` — Non-materializing gene QC filter for backed/lazy data. Computes column NNZ and/or column sums via streaming, builds a boolean mask, and sets `col_projection` on `ScxBackedSparseDataset` or `ScxLazyTransformedDataset`. Slices `adata.var` to match. Composes with existing column projections. Falls back to `sc.pp.filter_genes()` for scipy/dense.
- `pyscx.accel.subset_obs(adata, mask_or_indices)` — Subset observations (cells) without materializing. Accepts a **boolean numpy mask** or **integer index array**. Creates a new deletion vector (`kept_to_global`) on the backing dataset, slices `adata.obs` and `adata.obsm`, and updates `adata.layers`. Composes correctly with existing deletion vectors. Falls back to numpy slicing for non-SCX data.

  > [!WARNING]
  > **Integer index semantics differ from NumPy.** When `mask_or_indices` is an integer array, it is internally converted to a boolean mask. This means:
  > - **Duplicate indices are silently collapsed** — `[0, 0, 5]` is equivalent to `[0, 5]`
  > - **Order is not preserved** — `[5, 0, 10]` produces the same result as `[0, 5, 10]`
  >
  > This differs from NumPy's fancy indexing where `a[[5, 0, 10]]` returns rows in the order `[5, 0, 10]` with duplicates preserved. Use a boolean mask for unambiguous results.

#### Accelerator route metadata

Every `pyscx.accel.*` call records the execution route it actually took on `adata.uns["scx_accel"][<op>]`. `rank_genes_groups` additionally copies the route string to `adata.uns["rank_genes_groups"]["scx_accel_route"]`. The dict carries:

- `route` — the concrete path taken. One of `cpu_dense`, `cpu_csr`, `cpu_csc`, `gpu_csr_v3`, `gpu_csc_v3` (the column-major perf path), `gpu_csr` / `gpu_dense` (non-DE GPU routes — see **Non-DE ops** below), or `gpu_device_resident` (the fused PCA→kNN path keeps the embedding on the GPU between stages — see **Non-DE ops**).
- `fallback_reason` — why the ideal route wasn't taken: `none`, `no_cuda`, `no_csc_sidecar`, `unsupported_dimensions`, `unsupported_input_layout`, `user_forced_cpu`, or `perf_policy`.
- `chunk_size`, `csc_available`, `graph_replay`, `shards_decoded`, `shards_uploaded` — optional detail (`None` when not tracked).

**Op keys:**

| Op key | Covers |
|--------|--------|
| `"pdex_ref"` | `pyscx.accel.pdex_ref` |
| `"rank_genes_groups"` | `pyscx.accel.rank_genes_groups` |
| `"rank_genes_groups_df"` | `pyscx.accel.rank_genes_groups_df` |
| `"highly_variable_genes"` | `pyscx.accel.highly_variable_genes` |
| `"pca"` | `pyscx.accel.pca` |
| `"neighbors"` | `pyscx.accel.neighbors` |
| `"pca_neighbors"` | `pyscx.accel.pca_neighbors` (also stamps `"pca"` + `"neighbors"`) |
| `"umap"` | `pyscx.accel.umap` |
| `"leiden"` | `pyscx.accel.leiden` |
| `"normalize_total"` | `pyscx.accel.normalize_total` |
| `"log1p"` | `pyscx.accel.log1p` |
| `"calculate_qc_metrics"` | `pyscx.accel.calculate_qc_metrics` |
| `"pseudobulk_dex"` | `pyscx.accel.pseudobulk_dex` |

This is the canonical way to confirm which path ran when comparing CPU vs GPU performance — GPU is fastest only when the input layout matches the op. The CSC-direct route (`route == "gpu_csc_v3"`) requires a backed SCX file with a CSC sidecar; in-memory CSR inputs (and files without a sidecar) report `gpu_csr_v3` with `fallback_reason == "no_csc_sidecar"`. v3 is the unconditional default GPU DE route (the `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3` opt-in gates were removed when v3 became the default). The route is decided by a single internal planner (`scx_accel::route::plan_de_route`) that also *drives* dispatch — the GPU `pdex_ref` **and** Wilcoxon (`rank_genes_groups` / `rank_genes_groups_df`) entry points `match` on the planned route to select the kernel, and CPU dispatch calls the same planner — so the recorded value always matches the kernel that executed. Both DE ops take `gpu_csc_v3` (CSC sidecar) or `gpu_csr_v3` (otherwise) on GPU; dense-host input is densified to CSR and also records `gpu_csr_v3`. Route-specific benchmark gates in `thresholds.yaml` enforce correct dispatch across all GPU ops — see [benchmarks/README.md § Regression Gating](../benchmarks/README.md#regression-gating). The legacy `SCX_GPU_DE_V3_TRACE` stderr trace remains only as a debug fallback.

**Deprecated route identifiers.** The v1/v2 dense-materialization GPU DE drivers were removed (GPU DE is now always v3), and the non-DE GPU routes were renamed from the misleading `_v1` suffix to plain `gpu_csr` / `gpu_dense` in the follow-on taxonomy cleanup. So `gpu_csr_v1`, `gpu_dense_v1`, and `gpu_csr_v2` are all **historical** — they no longer appear in fresh output. Interpret them in old benchmark JSON as: `gpu_csr_v2` = legacy v2 DE; DE `gpu_dense_v1` = legacy v1-dense DE; non-DE `gpu_csr_v1` / `gpu_dense_v1` = today's `gpu_csr` / `gpu_dense`.

**Non-DE ops.** PCA, kNN (`neighbors`), Leiden, HVG, and preprocessing each have a single GPU route, so their metadata is the GPU-vs-CPU dispatch contract rather than a version cascade. On a GPU host they record `gpu_csr` (`pca` cuSPARSE+cuBLAS, `neighbors` cuVS CAGRA, `leiden` cuGraph, `highly_variable_genes` seurat_v3 atomic-CSR), except `umap`, whose native CUDA / cuML SGD runs on a dense embedding and records `gpu_dense`. When the op's GPU library is unavailable the route falls back to `cpu_csr` / `cpu_dense` with `fallback_reason="unsupported_input_layout"` (vs `no_cuda` when CUDA itself is absent, or `user_forced_cpu` for `device="cpu"`). PCA's covariance-vs-randomized choice is a math-policy detail recorded separately in `uns["pca"]["backend"]`, not in the route string.

**Fused device-resident route.** `pyscx.accel.pca_neighbors` runs PCA then kNN in one call; when both the modern cuSPARSE ABI and cuVS are available it keeps the PCA embedding GPU-resident across the handoff and stamps `gpu_device_resident` on `uns["scx_accel"]["pca"]`, `["neighbors"]`, **and** `["pca_neighbors"]`. When the fused path can't run (no GPU / old libcusparse / no cuVS) it delegates to the standalone `pca` + `neighbors` — each records its usual route — and the `pca_neighbors` summary records the non-device-resident result. (The fuzzy-graph step runs on the CPU either way; full GPU residency through UMAP is the V3 plan's Phase 2.4.)

**Preprocessing caveat.** `normalize_total` / `log1p` only reach the GPU shard-streaming kernel when `adata.X` is a backed (`ScxBackedSparseDataset`) or lazy (`ScxLazyTransformedDataset`) dataset; on a materialized scipy/dense `X` they record `cpu_csr` (with `fallback_reason="unsupported_input_layout"` on a GPU host) because the eager kernel needs a shard source — open via `pyscx.open(...).to_anndata(backed=True)` to engage GPU. `calculate_qc_metrics` and `pseudobulk_dex` are CPU-only; their route reflects only the gene-axis layout (`cpu_csc` under `prefer_format="csc"`, else `cpu_csr`). `harmony_integrate` does not record route metadata yet — its dispatch detail lives in `uns["harmony"]["backend"]`.

### ScxBackedSparseDataset

PyO3 class for backed-mode lazy access to the main expression matrix (`adata.X`). Data stays on disk; only requested shards are decoded on access. Registered with `anndata.abc.CSRDataset`.

**Properties:**
- `shape` `→ (int, int)` — `(n_obs, n_vars)`, adjusted for deletion vectors and column projection
- `dtype` `→ numpy.dtype` — Always `float32`
- `format` `→ str` — Always `"csr"`
- `backend` `→ str` — Always `"scx"`
- `ndim` `→ int` — Always `2`
- `non_negative` `→ bool` — Whether the data is known to be non-negative (enables `(X > 0).sum() → getnnz()` short-circuit)
- `nnz` `→ int` — Total non-zero count
- `n_shards` `→ int` — Number of CSR shards in the backing file

**Column projection:**
- `set_col_projection(col_indices)` — Restrict all access and aggregation to a subset of columns. Used internally by `to_anndata(var_names=...)` and streaming QC with gene subsets (`qc_vars`).

**Slicing:**
- `__getitem__(row_slice)` `→ scipy.sparse.csr_matrix` — Decode requested shards, return scipy CSR.
- `__getitem__(row_slice, col_slice)` — Row decode + column post-filter.
- Supports integer, slice, boolean mask, and fancy indexing.

**Aggregation (streaming, no materialization):**
- `sum(axis=0|1)` `→ numpy.ndarray` — Column or row sums via native Rust streaming.
- `mean(axis=0|1)` `→ numpy.ndarray` — Column or row means.
- `var(axis=0|1)` `→ numpy.ndarray` — Column or row variance (two-pass).
- `getnnz(axis=0|1)` `→ numpy.ndarray` — Non-zero counts per column or row.
- `max(axis=0|1)` `→ numpy.ndarray` — Column or row max.
- `min(axis=0|1)` `→ numpy.ndarray` — Column or row min.

**Materialization:**
- `to_memory()` `→ scipy.sparse.csr_matrix` — Decode all shards → full CSR.
- `copy()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()`.
- `toarray()` `→ numpy.ndarray` — Dense array.
- `tocsr()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()` (scipy compat).
- `tocsc()` `→ scipy.sparse.csc_matrix` — Materialize and convert to CSC.
- `.A` `→ numpy.ndarray` — Dense array property (scipy compat).

**Comparison operators:**
- `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` — Return `ScxComparisonResult` for lazy boolean operations.

**Arithmetic:**
- `__truediv__(other)` — If `other` is a per-row vector, returns `ScxLazyTransformedDataset` with `RowScale(1/factors)` (lazy). Otherwise materializes.
- `__mul__(other)` — If `other` is a per-row vector, returns `ScxLazyTransformedDataset` with `RowScale(factors)` (lazy). Otherwise materializes.

  *Per-row vector* means an **unambiguously row-oriented** operand: a 1-D `(n_obs,)` array or an explicit `(n_obs, 1)` column vector. A `(1, n_obs)` row vector is treated as a per-*column* broadcast (numpy/scipy semantics), **not** intercepted as a transposed row scale — on a non-square matrix it falls through to scipy and raises a shape error rather than being silently mis-applied. On a **square** matrix (`n_obs == n_vars`) a bare 1-D `(n,)` operand is orientation-ambiguous (could be per-gene), so it is not intercepted and instead materializes; pass an explicit `(n_obs, 1)` column vector to force the lazy row-scale path. scanpy's `normalize_total` reshapes its row factors to `(n_obs, 1)`, so that path stays lazy.
- `__add__(other)` — Materializes and adds.
- `__sub__(other)` — Materializes and subtracts.
- `__matmul__(other)` — Matrix multiply (materializes).
- `multiply(other)` — Element-wise Hadamard product (materializes).
- `power(n)` — Element-wise power (materializes).

**Introspection:**
- `shard_boundaries()` `→ list[(int, int)]` — List of `(row_start, row_end)` tuples per shard.

### ScxBackedLayerDataset

PyO3 class for backed-mode layer access (e.g., `adata.layers["raw_counts"]`). Wraps a `ScxBackedSparseDataset` for a named layer. Registered with `anndata.abc.CSRDataset`.

- Same interface as `ScxBackedSparseDataset` (`shape`, `dtype`, `format`, `backend`, `ndim`, `__getitem__`, `to_memory`, `toarray`, `tocsr`, `tocsc`, `copy`, `sum`, `mean`, `var`, `getnnz`, `max`, `min`)
- `layer_name` `→ str` — Name of the backing layer

### ScxComparisonResult

Lazy comparison result returned by `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` on `ScxBackedSparseDataset` and `ScxLazyTransformedDataset`. Exposed as `_ComparisonResult` in Python.

**Key optimization:** `(X > 0).sum(axis)` short-circuits to `getnnz(axis)` without materializing the full boolean matrix. This is the critical path for `sc.pp.calculate_qc_metrics()`, `sc.pp.filter_cells()`, and `sc.pp.filter_genes()`. The short-circuit only activates for non-negative data (raw counts, normalized, log1p).

- `shape` `→ (int, int)`, `dtype` `→ numpy.dtype` (bool), `ndim` `→ int` (2)
- `sum(axis=None)` — Short-circuits `(X > 0).sum()` → `getnnz()` for non-negative data; otherwise materializes.
- `getnnz(axis=None)`, `mean(axis=None)` — Materialize and delegate.
- `toarray()`, `tocsr()`, `tocsc()` — Materialize to dense/sparse.
- `multiply(other)` — Element-wise product (materializes).
- `.A` `→ numpy.ndarray` — Dense array property.

### ScxLazyTransformedDataset

PyO3 class wrapping `ScxBackedSparseDataset` with chained per-row transforms. Created by `pyscx.accel.normalize_total()` and `pyscx.accel.log1p()`. Implements the same interface as `ScxBackedSparseDataset` and is registered with `anndata.abc.CSRDataset`.

**Properties:**
- `shape` `→ (int, int)` — `(n_obs, n_vars)`
- `dtype` `→ numpy.dtype` — Always `float32`
- `format` `→ str` — Always `"csr"`
- `ndim` `→ int` — Always `2`
- `backend` `→ str` — Always `"scx-lazy"` (distinguishes from `ScxBackedSparseDataset.backend` which is `"scx"`)
- `non_negative` `→ bool` — Whether the transformed data is non-negative

**Slicing:**
- `__getitem__(row_slice)` `→ scipy.sparse.csr_matrix` — Decode requested shards, apply all transforms in order, return scipy CSR. Peak memory = 1 shard.
- `__getitem__(row_slice, col_slice)` — Row decode + transform + column projection.

**Aggregation (streaming through transforms):**
- `sum(axis=0|1)` `→ numpy.ndarray` — Column or row sums of transformed data.
- `mean(axis=0|1)` `→ numpy.ndarray` — Column or row means of transformed data.
- `var(axis=0|1)` `→ numpy.ndarray` — Column or row variance (two-pass streaming).
- `getnnz(axis=0|1)` `→ numpy.ndarray` — Non-zero counts (unchanged by normalize/log1p).
- `max(axis=0|1)` `→ numpy.ndarray` — Column or row max of transformed data.
- `min(axis=0|1)` `→ numpy.ndarray` — Column or row min of transformed data.

**Materialization:**
- `to_memory()` `→ scipy.sparse.csr_matrix` — Decode all shards + apply transforms → full CSR.
- `copy()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()`.
- `toarray()` `→ numpy.ndarray` — Dense array (via `to_memory().toarray()`).
- `tocsr()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()` (scipy compat).
- `tocsc()` `→ scipy.sparse.csc_matrix` — Materialize and convert to CSC.
- `.A` `→ numpy.ndarray` — Dense array property (scipy compat, same as `toarray()`).

**Comparison operators:**
- `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` — Return `ScxComparisonResult` for lazy boolean operations.

**Arithmetic:**
- `__truediv__(other)` — If `other` is a per-row vector, appends `RowScale` transform (lazy). Otherwise materializes.
- `__mul__(other)` — If `other` is a per-row vector, appends `RowScale` transform (lazy). Otherwise materializes.

  *Per-row vector* is defined identically to `ScxBackedSparseDataset` above: `(n_obs,)` or `(n_obs, 1)` are intercepted lazily; `(1, n_obs)` and ambiguous square-matrix `(n,)` operands fall through to materialization.
- `__add__(other)` — Materializes and adds.
- `__sub__(other)` — Materializes and subtracts.
- `__matmul__(other)` — Matrix multiply (materializes).
- `multiply(other)` — Element-wise Hadamard product (materializes).
- `power(n)` — Element-wise power (materializes).

**Introspection:**
- `nnz` `→ int` — Total non-zero count in backing file.
- `n_shards` `→ int` — Number of CSR shards in backing file.
- `shard_boundaries()` `→ list[(int, int)]` — List of `(row_start, row_end)` tuples per shard.

**Transform chain:**
- Backed data → `NormalizeTotal` → `Log1p` is fused into `ln(x × target_sum / row_sum + 1)` in a single pass.
- `repr()` shows the transform chain: `ScxLazyTransformedDataset(shape=(1000000, 33694), transforms=[NormalizeTotal, Log1p])`

### scVI Integration (`pyscx.scx_integrations.scvi`)

Pure-Python PyTorch Lightning DataModule wrapping `TrainingDataset` for scVI model training. Requires `lightning` or `pytorch_lightning`.

- `ScxDataModule(scx_path, batch_size=1024, hvg_indices=None, normalize=True, log1p=True, target_sum=1e4, seed=42, **kwargs)` — Creates a PyTorch Lightning `LightningDataModule`.
  - `scx_path` — Path to the `.scx` file.
  - `batch_size` — Mini-batch size (default: 1024).
  - `hvg_indices` — Gene indices for HVG projection. `None` = all genes.
  - `normalize` — Apply total-count normalization (default: `True`).
  - `log1p` — Apply log1p transformation (default: `True`).
  - `target_sum` — Normalization target sum (default: `1e4`).
  - `seed` — RNG seed for reproducibility (default: `42`).
  - `**kwargs` — Additional keyword arguments passed to `TrainingDataset`.
- Properties: `n_obs`, `n_vars`, `n_output_genes`
- `train_dataloader()` `→ DataLoader` — Uses `batch_size=None` and `num_workers=0` (Rust handles batching and threading internally).
- `val_dataloader()` `→ None` — Validation not currently supported.

```python
from pyscx.scx_integrations.scvi import ScxDataModule
import scvi

dm = ScxDataModule("atlas.scx", batch_size=1024, hvg_indices=hvg_array)
model = scvi.model.SCVI(dm.adata_manager)
model.train(datamodule=dm)
```

### TrainingDataset

High-throughput sequential streaming dataset. Wraps the triple-buffered
Rust pipeline (tokio I/O → rayon decode → Python/GPU). Each `for batch in dataset:`
loop is one epoch; shards are reshuffled between epochs for training randomization.

**Constructor kwargs**

| Argument | Default | Notes |
|---|---|---|
| `path` | — | Path to `.scx` file. |
| `batch_size` | `1024` | Mini-batch size. Auto-tuned downward if `max_memory_mb` is exceeded. |
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. |
| `obs_columns` | `[]` | Obs metadata column names included in each batch. |
| `normalize` | `True` | Total-count normalize (fused with `log1p` in a single CSR row scan). |
| `log1p` | `True` | Apply `log1p` after normalize. |
| `target_sum` | `1e4` | Normalization target sum. |
| `shard_group_size` | `8` | Shards per I/O group. Sequential I/O within each group for disk efficiency. |
| `prefetch_batches` | `4` | Ring buffer depth — number of pre-built batches to buffer ahead. |
| `seed` | `42` | RNG seed for reproducibility. Deterministic shuffle via `(seed, epoch)`. |
| `max_memory_mb` | `512` | Memory budget. Pipeline auto-tunes `shard_group_size`, `prefetch_batches`, and `batch_size` to fit. |
| `modality` | `None` | For multimodal v2 files: name of the modality to load (e.g. `"rna"`). Ignored on single-modality files. |

**Properties**

- `n_obs` → `int` — Total number of observations (cells) in the dataset.
- `n_vars` → `int` — Total number of variables (genes) in the dataset.
- `n_output_genes` → `int` — Genes per batch (HVG count if projection active, else `n_vars`).
- `effective_batch_size` → `int` — Actual batch size after memory budget auto-tuning.

**Methods**

- `close()` — Explicitly shut the pipeline down (join I/O + decode threads, release rayon pool). Idempotent. Recommended before process exit; see [Fork safety](#fork-safety-under-pytorch-dataloadernum_workers--0).
- `memory_budget()` → `dict` — Memory budget diagnostics including `shard_group_size`, `prefetch_batches`, `batch_size`, `estimated_mb`, `mmap_mb`, `budget_exceeded`, and a nested `breakdown` dict.

**Batch dict schema**

```python
{
    "X":            np.ndarray[B, n_output_genes, float32],   # dense expression
    "obs":          dict[str, np.ndarray | {"codes", "categories"}],
    "cell_indices":  np.ndarray[B, int64],   # global row indices
}
```

Categorical obs columns encode as `{"codes": ndarray[int32], "categories": list[str]}`.
Numeric obs columns are `ndarray[int64]` or `ndarray[float64]`.

**Epoch and shuffling semantics**

Each `for batch in dataset:` loop is one epoch. On each epoch:
- **Level 1 (shard order)**: Shard indices `[0..n_shards)` are randomly permuted using a
  deterministic RNG seeded from `(seed, epoch_number)`. Permuted shards are grouped into
  contiguous I/O groups of `shard_group_size` for disk-sequential reads.
- **Level 2 (row shuffle)**: Within each shard group, cell indices are Fisher-Yates shuffled
  and sliced into `batch_size`-sized batches.

This two-level shuffle provides training randomization without random I/O. The same `seed`
and epoch always produce the identical ordering.

```python
dataset = pyscx.TrainingDataset("file.scx", batch_size=1024,
    hvg_indices=hvg_array, normalize=True, log1p=True,
    obs_columns=["cell_type", "batch"])

for epoch in range(n_epochs):
    for batch in dataset:
        x = batch["X"]              # [B, n_output_genes] float32
        obs = batch["obs"]          # {"cell_type": {"codes": ..., "categories": ...}, ...}
        idx = batch["cell_indices"]  # [B] int64 — global row indices

dataset.close()
```

### IndexPlanDataset

Plan-driven paired-batch reader for ML workloads where each batch is a list
of `(perturbed_cell, control_cell)` index pairs (perturbation training,
contrastive learning, donor-matched designs). Sibling to `TrainingDataset`:
`TrainingDataset` streams shards in catalog order for the highest possible
sequential throughput; `IndexPlanDataset` consumes a Python iterator of
plans and yields paired dense batches, trading sequential streaming for
per-cell pairing flexibility.

**Constructor kwargs**

| Argument | Default | Notes |
|---|---|---|
| `path` | — | Path to `.scx` file. |
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. |
| `obs_columns` | `[]` | Obs metadata column names included in each batch. |
| `normalize` | `True` | Total-count normalize (fused with `log1p`). |
| `log1p` | `True` | Apply `log1p` after normalize. |
| `target_sum` | `1e4` | Normalization target sum. |
| `cache_shards` | `128` | LRU shard cache budget. Auto-tuned downward to fit `max_memory_mb`; check via `effective_cache_shards()`. |
| `sort_by_shard` | `True` | Reorder each plan by `min(shard_of(p), shard_of(c))` so the returned `X`/`X_paired` rows land in shard locality order. Disable to preserve caller's input pair order. |
| `lookahead` | `4` | Default lookahead for `iter_with_plans` when not overridden. `0` disables shard prefetching; auto-tuned downward to fit `max_memory_mb`; check via `effective_lookahead()`. |
| `max_plan_size` | `16384` | Upper bound on rows-per-batch for the memory budget calculation. |
| `max_memory_mb` | `512` | On overflow, `lookahead` is reduced first (down to 1), then `cache_shards` (down to 1); construction fails with `RuntimeError` if neither fits. |

**Batch dict schema** (yielded by `iter_with_plans`):

```python
{
    "X":          np.ndarray[B, n_output_genes, float32],   # perturbed rows
    "X_paired":   np.ndarray[B, n_output_genes, float32],   # control rows
    "pairs":      list[tuple[int, int]],                    # post-sort plan
    "obs":        dict[str, np.ndarray | {"codes", "categories"}],
    "obs_paired": dict[str, np.ndarray | {"codes", "categories"}],
}
```

`pairs[i]` always corresponds to `X[i]` and `X_paired[i]`. Categorical obs
columns encode as `{"codes": ndarray[i32], "categories": list[str]}` —
schema matches `TrainingDataset`.

**Iterator semantics**

- `plans` is any Python iterable yielding `list[tuple[int, int]]` (plain lists,
  generators, queues all work).
- Plan iteration is lazy: the loader pulls the next plan only when it is
  ready to schedule a prefetch for it.
- The loader keeps `lookahead` plans in flight at once: the head plan is
  decoding while shards for the next `lookahead - 1` are being warmed via
  `tokio::task::spawn_blocking` calls into `BackedCsrReader::read_shard_cached_arc`.
- `StopIteration` from `plans` ends the batch stream cleanly. Other Python
  exceptions from `plans` propagate as `RuntimeError("plan iterator raised: ...")`.
- Empty plans inside a stream are silently skipped.
- Out-of-range row indices raise `IndexError` immediately when validating
  the offending plan; missing obs columns raise `KeyError` at construction.
- `os.fork()` after construction raises `RuntimeError` with `num_workers=0`
  guidance — the shard cache and mmap state are not fork-safe; consumers
  must lazily construct the dataset post-fork in each DataLoader worker.

**Example — bare iterator**

```python
import pyscx

ds = pyscx.IndexPlanDataset(
    "atlas.scx",
    hvg_indices=hvg_array,
    obs_columns=["cell_type", "perturbation"],
    normalize=True,
    target_sum=1e4,
)

plans = [
    [(0, 5), (2, 7)],
    [(10, 100), (50, 75)],
]
for batch in ds.iter_with_plans(iter(plans)):
    X, X_paired = batch["X"], batch["X_paired"]
    pairs = batch["pairs"]
    # ... training step ...
```

**Example — paired with a `BaseMappingStrategy`-style plan generator** (the
cell-load-scx pattern):

```python
def plan_generator(strategy, perturbed_indices, batch_size):
    """Wrap any pairing policy that exposes `get_control_index(idx) -> int`."""
    buf = []
    for p_idx in perturbed_indices:
        c_idx = strategy.get_control_index(p_idx)
        if c_idx is not None:
            buf.append((p_idx, c_idx))
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf

ds = pyscx.IndexPlanDataset("atlas.scx", obs_columns=["cell_type"])
for batch in ds.iter_with_plans(plan_generator(my_strategy, perm, 1024),
                                lookahead=4):
    ...
```

**Memory budget surfaces**

```python
ds = pyscx.IndexPlanDataset("atlas.scx", cache_shards=128, lookahead=4,
                            max_plan_size=16384, max_memory_mb=256)
print(ds.effective_cache_shards(), ds.effective_lookahead())
# Detects when auto-tuning kicked in.
```

### Fork safety under PyTorch `DataLoader(num_workers > 0)`

`pyscx.TrainingDataset` and `pyscx.IndexPlanDataset` are fork-safe under
`torch.utils.data.DataLoader(num_workers > 0, start_method="fork")` —
the Linux PyTorch default — when the dataset is **constructed lazily
inside the worker's `__iter__`** (the pattern `cell-load-scx` and
`state-scx` already use). Both the per-pipeline tokio runtime (current-
thread, per-epoch) and per-pipeline `rayon::ThreadPool` (lazily built on
first iteration) are constructed inside the worker process, so a forked
child inherits no fork-hostile state from the parent.

```python
# Recommended IterableDataset wrapper for DataLoader(num_workers=2)
import torch.utils.data as data

class TrainingShim(data.IterableDataset):
    def __init__(self, scx_path):
        self.scx_path = scx_path  # paths only — no inner dataset yet

    def __iter__(self):
        # Construct the inner dataset HERE (in the worker process, post-fork).
        ds = pyscx.TrainingDataset(self.scx_path, batch_size=1024, hvg_indices=...)
        try:
            yield from ds
        finally:
            ds.close()  # release the per-pipeline rayon pool / tokio runtime

loader = torch.utils.data.DataLoader(
    TrainingShim("atlas.scx"),
    batch_size=None,
    num_workers=2,
    persistent_workers=False,
)
```

**Do**: construct lazily inside `__iter__`; call `dataset.close()` (or
register `weakref.finalize(dataset, dataset.close)`) before process exit;
prefer `multiprocessing.set_start_method("spawn")` if your workload
allows — spawn re-execs Python in the child and is genuinely fork-safe
because there is no fork.

**Don't**: construct a `TrainingDataset` / `IndexPlanDataset` in the
parent and share it across forked workers — the PID check in `__next__`
raises `RuntimeError`. Don't pickle a constructed dataset across
processes either; it owns thread handles that don't survive transfer.

> [!IMPORTANT]
> Calling **any** `rayon::par_*`-using pyscx API in the parent before
> fork (e.g., `pyscx.from_anndata(...)` to write the fixture) initialises
> rayon's process-global pool. Pre-fix this prerequisite was sufficient
> to wedge `DataLoader(num_workers=2)` indefinitely; the per-pipeline
> rayon pool in `scx-loader` removed that hazard. See
> `pyscx/tests/test_fork_safety.py` for the durable regression test.

## CLI (`scx`)

The CLI binary is named `scx` (built from the `scx-cli` crate via `cargo build -p scx-cli`).


### Core
- `scx convert <input> <output> [--from h5ad|10x|h5mu|scx] [--to h5ad|h5mu|scx] [--codec auto|none|scx1|zstd|lz4|pcodec] [--shard-size N] [--stream[=true|false]] [--csc off|always] [--csc-cols-per-shard N] [--modality NAME] [--memory-budget SIZE] [--strict-uns] [--dense-zero-epsilon F] [--temp-dir DIR] [--modalities CSV] [--modality-types NAME:TYPE,...] [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N] [--bitmap off|auto|always] [--reader-threads N] [--writer-queue-depth N]` — `--stream` (default `true`) bounds peak memory to one shard's worth of CSR plus encode buffers; supported on h5ad ↔ SCX and h5mu ↔ SCX in both directions. On ingestion (h5ad/h5mu → SCX), combine with `--csc always` for a two-pass CSR-then-`rebuild_csc_inplace` write (transient disk ~2× the output). On export (SCX → h5ad/h5mu), the streaming writer pre-allocates the `/X/{indptr,indices,data}` HDF5 triplet from catalog stats (or a single pre-scan when deletion vectors are active) so the on-disk layout is deterministic. Pass `--stream=false` to opt into the legacy materialising path on either side. For multimodal SCX → h5ad, combine `--to h5ad --modality NAME` to extract a single modality.

  Wild-h5ad hardening: `--memory-budget 4G` caps dense row slabs and
  CSC external-transpose buffers (binary prefixes only; see
  [Memory budgets](#memory-budgets)). `--strict-uns` aborts on the
  first unrepresentable `uns` entry rather than warning.
  `--dense-zero-epsilon F` thresholds near-zero values during
  dense→CSR sparsification (default `0.0`). `--temp-dir DIR` selects
  the scratch directory for CSC external-transpose runs.

  h5mu inputs: `--modalities rna,adt` restricts to a subset of
  modalities; `--modality-types adt:Protein,peaks:ATAC` overrides
  inferred types.

  Query-readiness: `--index-obs cell_type,donor` and
  `--index-var gene_name` force predicate indexes; `--index-preset
  cellxgene|perturbseq|training` expands curated column lists;
  `--index-auto-threshold N` controls automatic categorical indexing.
  `--bitmap auto|always` writes per-shard detection bitmap sidecars
  consumed by `detection_counts` / `cells_expressing`. See
  [Conversion-time predicate indexes and detection bitmaps](#conversion-time-predicate-indexes-and-detection-bitmaps).

- `scx info <file> [--json] [--history]` — Pretty-prints file
  metadata. Multimodal v2 files show a per-modality table including
  a `has_csc` column (`✓` / `—`) and the underlying boolean lands in
  the `--json` payload under each modality's entry. The file-level
  header `has_csc` flag means "at least one CSC sidecar exists" for
  v2 multimodal files; the per-modality column resolves which
  modalities own one (relevant after a partial-CSC `scx append`).
- `scx validate <file> [--verbose]` — Walks the catalog and verifies
  BLAKE3 checksums section-by-section. Partial per-modality CSC
  sidecars are accepted naturally.
- `scx benchmark <file> [--compare-h5ad <path>] [--runs N] [--json]`

### File operations
- `scx append <target> --input <source> [--codec auto|none|scx1|zstd|lz4|pcodec] [--shard-size N] [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N]` — Streaming append (reads source one shard at a time). `--index-*` rebuilds predicate indexes covering all rows post-append — see [Conversion-time predicate indexes and detection bitmaps](#conversion-time-predicate-indexes-and-detection-bitmaps).
- `scx delete <file> --filter <expr> [--dry-run]`
- `scx compact <input> --output <path> [--force] [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N]` — Rewrite reclaiming space; `--index-*` rebuilds the predicate index against the compacted output.
- `scx rollback <file> [--to-seq N]`
- `scx merge <file1> <file2> [<...>] --output <path> [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N]` — Merge multiple files; `--index-*` rebuilds the predicate index against the merged output (without it, pushdown regresses to a full obs scan on the merged file).
- `scx query <input> <filter> [--count] [--output <path>] [--select-genes <path>] [--normalize N] [--log1p] [--limit N] [--json]` — `<input>` accepts a local `.scx` file path, an exploded `.scxd/` directory, or a cloud URL (`gs://`, `s3://`, `az://`, `file://`). For cloud inputs the query is served via the `SectionReader` cloud path with no `scx pull` step. See [docs/cloud.md § Cloud-native query](cloud.md#cloud-native-query).
- `scx subset <input> [--output <path>] [--filter <expr>] [--genes <path>] [--dry-run] [--shard-size N] [--codec auto|none|scx1|zstd|lz4|pcodec]` — Extract a subset of cells and/or genes into a new SCX file
- `scx build-csc <input> <output> [--memory-limit 4G] [--force]` — Build CSC (column-major) shards from existing CSR data. `--memory-limit` accepts the same size forms as `--memory-budget` (see [Memory budgets](#memory-budgets)).
- `scx upgrade <input> [output] [--in-place]` — Upgrade an SCX file to the latest format version

### Cloud operations (`--features cloud`)
- `scx cloud-optimize <input> [--output <path>]`
- `scx explode <input> <output>`
- `scx pack <input> <output>`
- `scx pull <source-url> <dest> [--parallelism N] [--no-cloud-ready] [--filter <expr>]`
- `scx push <source> <dest-url> [--parallelism N]`

Note: the `scx-cli` crate has optional `hdf5` and `cloud` feature flags. HDF5 support is opt-in (`--features hdf5`). Cloud operations are opt-in (`--features cloud`). End-users installing via `cargo install` can pass `--features default-bin` to get an h5ad-capable build in one command.
