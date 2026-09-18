# SCX API Reference

## Section Types

29 section types (IDs 0–29, with ID 26 reserved) are defined in `scx-format/src/section.rs`:

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
                         by `Experiment.detection_counts` /
                         `cells_expressing`. See
                         [docs/format.md § Detection Bitmap](format.md#12-detection-bitmap-optional).
LayerCsrShard (7)      — Alternative expression layers
ObsmEmbedding (8)      — Embeddings (obsm)
ObspCsrShard (9)       — CSR-backed obs x obs graph. Written by
                         `ScxWriter::write_obsp_shard`, checked by
                         `scx validate --deep` against the v3 canonical-CSR
                         invariant, re-encoded by `scx optimize`, and
                         preserved by `scx upgrade` / `scx build-csc` —
                         build-csc copies it verbatim, while upgrade
                         canonicalizes a non-canonical pre-v3 graph and
                         re-encodes it, carrying the source shard's own
                         minor extent rather than re-deriving one. No read
                         API materialises it: `read_obsp` / `list_obsp` — and
                         so `to_anndata`'s `obsp` — see only the COO forms,
                         ObspEmbedding (18) / ObspEmbeddingShard (22), which
                         is what every conversion path writes.
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
                         append, from_anndata, optimize --shard-obs and
                         every scx convert / from_h5ad / from_h5mu /
                         from_mtx ingest,
                         all on the same n_obs > shard_size boundary;
                         --shard-obs off|always overrides it on the
                         convert paths. Mutually exclusive with
                         ObsMetadata (0) in the same file.
VarMetadataShard (25)  — Row-sharded var Arrow IPC (mirror of 24). NOT
                         emitted by convert ingest at any n_vars — only by
                         merge, append and from_anndata.
(26)                   — RESERVED (formerly DecodeMetadataShard, removed;
                         random access now via the codec-agnostic row-group
                         BlockIndex, framing). Legacy files carrying id 26 are
                         skipped by the catalog reader.
RawCsrShard (27)       — Row-sharded `raw.X` CSR (anndata `.raw` layer),
                         parallel to CsrShard (4).
RawVarMetadata (28)    — Arrow IPC var metadata for the `.raw` layer
                         (mirror of VarMetadata (2)).
GroupIndex (29)        — Condition/label-grouped sharding sidecar (one per
                         file; `group_index` JSON). Written by
                         `scx sort --group-by`; consumed by the grouped-read
                         API. See docs/format.md § section ids.
```

## ScxReader (`scx-format-io/src/reader/`)

- `open(path)` — Open and validate file (mmap-based, validates magic/version/minimum size)
- `header()`, `root_catalog()`, `catalog()` — Access file metadata
- `n_obs()`, `n_vars()`, `nnz()` — Quick dimension access
- `read_obs()`/`read_var()` — Arrow RecordBatch metadata
- `read_csr_shard(idx)` — Single shard as `(Vec<i64>, Vec<i32>, Vec<f32>)`
- `read_all_csr_shards()` — Full matrix as `ScxCsr` (parallel via rayon)
- `read_all_csr_shards_typed(plan)` — Full matrix as a `TypedCsr` at the plan's value / index dtypes, assembled directly at that width (also parallel via rayon)
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
- `obs_categorical(col)` / `obs_categorical_many(&[cols])` — `(codes: Vec<i32>, categories: Vec<String>)` for string/categorical obs columns, folded **one shard at a time** into a running global dictionary (never concatenates the column, unlike `read_obs_keys`). Null → `-1` (pandas convention), so a literal `"NaN"` string stays a real category; category order is first-seen. Accepts both `Dictionary(_, Utf8|LargeUtf8)` (as `from_anndata` writes) and plain `Utf8`/`LargeUtf8` (as `append` writes), including a file mixing both across shards. `_many` costs **one** projected read per shard for N columns. See [Obs categorical codes](performance.md#obs-categorical-codes-without-pandas-data-load-phase-1-1c).
- `debug_counts()` — `ReaderDebugCounts` with `AtomicU64` I/O counters (`cfg(debug_assertions)` only). `read_obs_shard_projected` counts column-scoped shard reads separately from `read_obs_shard`, so a test can assert the cheap path was *taken* rather than only that the materialising ones were avoided.
- `read_obs_predicate_index_bytes()` / `read_var_predicate_index_bytes()` — Predicate index raw bytes
- `read_deletion_vectors()` — Roaring Bitmap deletion vectors
- `deletion_keep_mask()` / `deletion_keep_mask_for(modality_id)` — `Option<Vec<bool>>` (`true` = keep); `None` when nothing is deleted
- `read_all_csr_shards_filtered()` / `read_all_csr_shards_for_filtered(modality_id)` / `read_layer_filtered(name)` — Matrix reads with deletion vectors applied
- `read_obs_filtered()` / `read_obs_keys_filtered(&[cols])` / `obs_categorical_filtered(col)` / `obs_categorical_many_filtered(&[cols])` — The obs half of the above: the whole frame, the projected read and the codes fold with the deletion keep mask applied. `read_obs()` / `read_obs_keys()` / `obs_categorical()` are *physical* (`header.n_obs` rows regardless of deletions), so any caller materialising a matrix for a user needs both halves and must move them together — an obs frame longer than its matrix is worse than either being stale alone. (`pyscx`'s `read_obs()` / `obs_categorical()` route to the filtered twins by default since 0.17.) `scatter_batch_to_physical(batch, keep)` is the inverse of `filter_batch_by_keep_mask`: a live-length batch back onto the physical axis, null at deleted rows — what the in-place obs writers use to accept a live-length frame
- `read_shard_header()` / `read_raw_shard_bytes()` / `read_shard_from_entry()` — Low-level shard access
- `mmap()` — Direct mmap access to the underlying file

## ScxWriter (`scx-format-io/src/writer.rs`)

- `new(path, header)` — Create writer (writes to temp file)
- `write_obs(batch)`/`write_var(batch)` — Arrow IPC metadata
- `write_csr_shard(indptr, indices, values, ...)` — CSR expression data
- `write_layer_csr_shard(name, ...)` — Named layer CSR data
- `write_obsp_shard(name, ...)` — Cell-cell graph CSR data. The shard's
  minor extent is stamped from `header.n_obs` (an obsp graph is obs x obs), so
  the per-shard index width follows the cell axis and not the gene axis; on a
  multimodal file `write_obsp_shard_for` stamps the same global `n_obs`, never
  the modality's `n_vars`. Canonical CSR is a precondition — this writer does
  not canonicalize.
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
[docs/format.md § 13](format.md#13-multimodal-extension-optional) for the
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
- `read_var_schema_for(modality_id)` — per-modality var schema without
  materialising the batch (`modality_id == 0` → global `var`). Backs the
  engine's modality-scoped `QueryPipeline`.
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

### Modality-scoped queries — `QueryPipeline` (`scx-engine`)

- `QueryPipeline::open_for_modality(path, modality_id: u8)` /
  `from_reader_for_modality(reader, modality_id)` — scope a query pipeline to
  one modality (`modality_id == 0` = global / single-modality, the default that
  `open` / `from_reader` delegate to). X assembly (`csr_shards_for_modality`),
  the cached `n_vars`, `filter_var`, and `select_genes` resolve against the
  modality; `filter_obs` evaluates against the shared global obs axis. The
  Level-2 obs-predicate-index fast path is disabled for modality-scoped queries
  (its shard ids are keyed to the flattened all-modality shard order); Level-1
  catalog-stats pruning runs per modality.
- Errors: `EngineError::UnknownModality { requested, available }` and
  `ModalityRequired { available }`. A modality-scoped query on a file that
  carries deletion vectors is now fully supported — v2 deletion vectors are a
  global-obs bitmap that applies identically to every modality, so the query
  returns deletion-filtered rows for the queried modality (no error and no
  `scx compact` workaround).
- The `SectionReader` trait gains modality-aware methods
  (`read_var_for` / `read_var_schema_for` / `modality_n_vars` /
  `modality_id_by_name` / `n_modalities` / `is_multimodal` / `modality_names`),
  with single-modality default impls. Both backends override them: `ScxReader`
  (local) and `CloudSectionReader` (cloud — routes per-modality `var`/CSR
  sections over packed and exploded `.scxd/` layouts), so
  `open_cloud(url).query(modality=…)` runs the same modality-scoped pushdown as
  the local path.

### Python (`pyscx`)

- `pyscx.from_mudata(mu, path, codec="auto", ...)` — write a MuData
  object as a multimodal SCX file. Per-modality CSR shards stamped
  with `modality_id` derived from the registration order; per-modality
  codec resolved via `select_codec_for_modality`.
- `pyscx.open(path)` returns an `Experiment`. New attrs / methods:
  - `is_multimodal: bool`, `n_modalities: int`, `modality_names: list[str]`.
  - `modality_id(name) -> int | None`, `modality_info(id) -> dict | None`.
  - `to_mudata() -> mudata.MuData` — round-trips back to MuData.
  - `query(modality="rna")` — modality-scoped predicate pushdown: X /
    `select_genes` / `filter_var` resolve against that modality's var while
    `filter_obs` stays on the shared global obs axis. On a multimodal file
    `modality=` is required (omitting → `ValueError`); unknown name →
    `KeyError`. Omit it on single-modality files. See
    [docs/multimodal.md § 3.4](multimodal.md#34-modality-scoped-queries--querymodality).
- `pyscx.MultimodalTrainingDataset(path, modalities=[…], …)` — yields
  per-batch dicts `{"X": {modality_name: ndarray}, "obs": {...},
  "cell_indices": ndarray}` (or tuples in `return_dict=False` mode). Pins a
  uniform effective `batch_size`/`shard_group_size` across modalities so a
  wide modality (e.g. ATAC) can't desync the per-modality batching; pass a
  larger `max_memory_mb` to lift the pinned batch. See
  [docs/multimodal.md § Training](multimodal.md#33-training--pyscxmultimodaltrainingdataset).
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
- `scx_query(exp, modality = "rna")` — modality-scoped query pipeline (R
  equivalent of pyscx `query(modality=…)`). `modality = NULL` is the global /
  single-modality axis; on a multimodal file a name is required (omitting or an
  unknown name `stop()`s).

#### Backed (out-of-core) sparse access

- `scx_open(path)$x_backed(cache_shards=128)` / `scx_backed_sparse(path,
  cache_shards=128)` — open a lazy, on-demand view of X that reads rows from
  disk with a shard-level LRU cache instead of materialising the whole matrix
  (the R equivalent of pyscx `to_anndata(backed=True)` X). Returns an
  `ScxBackedSparse` that behaves like a read-only sparse matrix:
  - `dim()` / `nrow()` / `ncol()`; `bsd[i, ]` / `bsd[i, j]` return a `dgCMatrix`
    of the selected cells (rows) × genes (columns), rows in requested order.
    Row indices are 1-based; positive-integer and logical indices are supported
    (negative/character/non-finite raise). Fractional indices truncate, matching
    a `dgCMatrix` (`M[1.9, ]` is row 1). Contiguous ascending `i` uses a single
    range read; arbitrary/reordered `i` uses a fancy gather.
  - `bsd$read_rows(start, end)` / `bsd$read_row_indices(idx)` are the raw
    methods underneath `[`. They are **0-based** (`end` is an exclusive bound)
    and, unlike `[`, strict: negative, `NaN`, fractional, and non-finite values
    raise rather than being coerced.
  - `bsd$row_sums()` / `bsd$col_sums()` / `bsd$nnz()` — streamed aggregations
    with no full decode; `bsd$to_dgcmatrix()` / `as(bsd, "dgCMatrix")` to
    materialise the whole matrix.
  - Column projection is applied in R on the returned matrix; CSC-sidecar /
    deletion-vector / multimodal-modality routing (present in pyscx) are not yet
    wired through the rscx backed path.

#### Lazy transform chains

- `scx_open(path)$x_lazy(cache_shards=128)` / `scx_lazy_transform(path,
  cache_shards=128)` — open a lazy-transform view of X (the R equivalent of
  pyscx `ScxLazyTransformedDataset`). Layer preprocessing transforms that are
  applied **on read** (out-of-core), never materialising the matrix:
  - `scx_normalize_total(lt, target_sum=1e4)`, `scx_log1p(lt)`,
    `scx_row_scale(lt, factors)` — pipe-friendly, immutable verbs; each returns
    a new `ScxLazyTransformed` with the op appended. `normalize_total` uses the
    per-cell sums of the data as transformed by the chain so far.
  - `lt[i, j]` returns the transformed `dgCMatrix` for the requested cells/genes
    (same indexing rules as the backed view); `lt$row_sums()` / `lt$col_sums()`
    stream over the transformed data; `lt$to_dgcmatrix()` / `as(lt,
    "dgCMatrix")` materialise the full transformed matrix.
  - Transforms preserve sparsity, so `nnz` is unchanged. The pyscx extras
    (arithmetic/comparison interception, CSC dispatch, deletion vectors) are not
    wired through the rscx path.

#### Analysis accelerators (`scx-accel`, CPU)

Pipe-friendly front ends over the same Rust kernels the Python accelerators
use, so the full Normalize → HVG → PCA → Neighbors → UMAP → Leiden →
FindMarkers pipeline runs in R. Each accepts a `Seurat` object (results
written into the expected slot, object returned invisibly) **or** a raw
genes × cells `dgCMatrix` / embedding matrix (the raw result list is returned,
usable without Seurat installed):

- `scx_highly_variable_genes(obj, n_top_genes=2000, span=0.3)` — seurat_v3 /
  vst HVG (two native passes + an R-side `loess` fit); sets
  `VariableFeatures()`.
- `scx_pca(obj, n_components=50, ...)` — randomized / covariance SVD; writes
  `obj[["pca"]]`.
- `scx_neighbors(obj, dims, k=20)` — HNSW kNN; writes the fuzzy connectivity
  graph to the `<assay>_nn` slot.
- `scx_umap(obj, dims, ..., neighbors=NULL)` — UMAP from the kNN connectivity;
  writes `obj[["umap"]]`.
- `scx_leiden(obj, resolution=1.0, ..., neighbors=NULL)` — Leiden clustering;
  sets `Idents()` / `seurat_clusters`.
- `scx_rank_genes_groups(obj, group.by, reference=NULL, tie_correct=FALSE)` —
  Wilcoxon rank-sum DE (FindAllMarkers analog); returns a tidy `data.frame`.
- `scx_score_genes(obj, gene_list, method="control", ctrl_size=50, n_bins=25,
  random_state=0, gene_pool=NULL, score_name="score")` — gene-set scoring
  (scanpy `score_genes` analog; methods `"control"` / `"mean"` / `"zscore"`).
  Missing genes are dropped with a warning. On a Seurat object writes
  `obj[[score_name]]` (returned invisibly); on a matrix returns the per-cell
  score vector.
- `scx_pseudobulk(obj, group_by, method="sum", min_cells_per_group=0)` —
  pseudobulk aggregation (cells → group×gene; `"sum"` / `"mean"`). `group_by`
  is one or more `meta.data` column names (Seurat) or a per-cell label
  vector/`data.frame` (matrix). Returns a `list(counts, samples, gene_names)`
  where `counts` is a features×samples matrix and `samples` is a `data.frame`
  of the groupby columns plus `n_cells` per group.
- `scx_pseudobulk_dex(obj, group_by, test_col, reference, aggr_method="sum",
  min_cells_per_group=10, dispersion="cox_reid_shrunk", cooks_filtering=TRUE,
  independent_filtering=TRUE)` — pseudobulk DE via the **Rust-native
  negative-binomial GLM** (DESeq2-style, CPU-only; the `pseudobulk_dex(backend=
  "nb_glm")` / `pdex_nb_glm` analog). Aggregates cells, then fits each
  non-reference level of `test_col` vs `reference`. **Requires ≥2 pseudobulk
  replicates per condition** — `group_by` must include `test_col` **and** a
  replicate column (donor/batch); under-replicated targets are skipped with a
  warning. Input must be **raw counts** (`layer="counts"`). Returns a tidy
  `data.frame` with DESeq2-style columns `gene, baseMean, log2FoldChange, lfcSE,
  stat, pvalue, padj, target, reference` (per-target BH-adjusted `padj`; `lfcSE`
  on the log2 scale).
- `scx_nb_glm(counts, design, contrast=NULL, size_factors=NULL,
  dispersion="cox_reid_shrunk", ...)` — direct NB-GLM on a pre-aggregated
  genes×samples count matrix + a samples×features `design` (e.g.
  `model.matrix(~ condition, sampleinfo)`); `contrast` is a 1-based coefficient
  (default: last). Returns `gene, baseMean, log2FoldChange, lfcSE, stat, pvalue,
  padj, dispersion, converged`. For no-replicate / log-normalized data use
  `scx_rank_genes_groups` (Wilcoxon); `pdex_ref` (Wilcoxon pseudobulk) is not
  separately wired.

Notes:
- `scx_umap` / `scx_leiden` build their **own** kNN (default `n_neighbors`)
  and do not read the `_nn` slot; pass `neighbors=scx_neighbors(emb)` (matrix
  form) to reuse a precomputed graph.
- `scx_rank_genes_groups` densifies the matrix, so on a Seurat object it
  defaults to `VariableFeatures()` when set; `tie_correct`/`rankby_abs` default
  to `FALSE` to match `pyscx`/scanpy.
- `scx_highly_variable_genes` runs the numeric passes in Rust but the loess fit
  in R (`stats::loess`); the selected HVG set may differ slightly from the
  Python accelerator's `skmisc.loess`.

All are CPU-only (rscx links no GPU feature); for GPU runs use the Python
accelerators. Joining `scx_harmony_integrate()` / `RunHarmony_scx()` and
`scx_compute_lisi()`, which predate this set.

#### File operations (options)

`scx_merge` / `scx_compact` / `scx_append` accept the same option surface as the
matching pyscx functions:

- `scx_merge(inputs, output, index_obs=NULL, index_var=NULL, index_preset=NULL,
  index_auto_threshold=0, assume_identical_var=FALSE, assume_identical_obs=FALSE,
  uns_policy="first", sort_by=NULL, reverse=FALSE)` — `uns_policy` ∈
  `first`/`require-equal`/`namespace`/`summary`; `sort_by` is a sorted **k-way**
  merge (each input must be pre-sorted by the key).
- `scx_compact(input, output, index_obs=NULL, index_var=NULL, index_preset=NULL,
  index_auto_threshold=0, reshape_obs=FALSE)`.
- `scx_append(target, input, codec=NULL, shard_size=16384, index_obs=NULL,
  index_var=NULL, index_preset=NULL, index_auto_threshold=0, modality=NULL)` —
  `codec` ∈ `auto`/`none`/`scx1`/`zstd`/`lz4`/`pcodec`; `modality` is a NAME on a
  multimodal target (streams from a reader, preserving per-shard encodings).

Predicate-index presets are `cellxgene`/`perturbseq`/`training`.
`shard_target_rows` is inherited from the input (not exposed on merge/compact);
CSC sidecars are dropped by these ops and must be rebuilt separately.

### CLI surface

```
scx info path.scx                 # Modalities (N): name, type, n_vars, nnz, csr/csc, codec
scx validate path.scx             # ModalityTable checksum + n_modalities cross-check
scx query cite.scx --modality rna --filter "cell_type == 'T cell'" --count
                                  # modality-scoped predicate pushdown (local only)
scx convert --from h5mu in.h5mu --to scx out.scx          # h5mu → SCX (streaming by default)
scx convert --from 10x raw_feature_bc_matrix.h5 out.scx    # 10x → SCX (streaming by default)
scx convert --to h5ad out.scx out.h5ad --modality rna     # SCX → h5ad (streaming by default)
scx convert --to h5mu out.scx out.h5mu                    # SCX → h5mu (streaming by default)
scx convert --to h5ad out.scx out.h5ad --stream=false     # opt-out: materialising path
# NOTE: `scx append --modality` into a multimodal file is deferred (rejected with
# MultimodalUnsupported): a single-modality append would leave sibling modalities
# under-covering the shared obs axis. Extract → append → re-merge instead:
scx subset in.scx rna_only.scx --modality rna
```

## Codec Selection (`scx-format/src/codec_select.rs`)

The user-facing `codec=` argument is a single **intent axis** with three profiles,
resolved by `resolve_codec`:

- **`auto`** (default) — cost-aware adaptive. Per framed integer shard, dual-encode
  the heuristic vs ShufDeltaZstd and adopt ShufDeltaZstd only when it is smaller by
  at least `ADOPT_MARGIN` (5%), so a marginal size win never pays the ShufDeltaZstd
  decode tax. The heuristic itself is `select_codec`: integer median ≤ 8 → Scx1 (Rice,
  optimal for 10x UMI counts), median > 8 → Zstd, float → Pcodec. Unframed writes fall
  back to the heuristic single-encode. `auto` files are typically **mixed-codec**;
  `scx info` prints the per-shard breakdown.
- **`fast`** — decode-speed-max: the heuristic single-encode (Scx1/Zstd), never
  ShufDeltaZstd. This is the pre-flip default; pin it for latency-critical CPU training.
- **`compact`** — size-max: adopt ShufDeltaZstd on ties (framed only).

Explicit forces (`none`/`scx1`/`zstd`/`lz4`/`pcodec`/`shufdelta`) and `compact-trial`
remain available. (The prior `auto_v2` profile + `decode_target` knob were removed as
a pre-1.0 clean break — use `auto`/`compact`.)

**Codec / profile tradeoffs:**

| Codec | Best for | Compression | Read speed | Write speed |
|-------|----------|-------------|------------|-------------|
| `auto` | General use (recommended default) — size-optimizing adaptive | Best per-shard (adopts ShufDeltaZstd where it wins) | Near-best (~6% ShufDeltaZstd tax on adopted shards) | Moderate (dual-encode) |
| `fast` | Latency-critical CPU training | Heuristic per-shard | Fastest (SIMD Scx1 decode) | Best (single-encode) |
| `compact` | Storage/egress-bound archival | Highest (adopts on ties) | ~6% ShufDeltaZstd tax | Moderate (dual-encode) |
| `scx1` | Small UMI counts (median ≤ 8) | Best for 10x data (~4.8×) | Fastest (SIMD decode) | Moderate |
| `zstd` | Large integers, general fallback | Good (~4.3× UMI, ~3.8× float) | Fast | Fast |
| `shufdelta` | Force the compact integer codec | ~1.5–2.5× smaller than Scx1 | ~6% slower than Scx1 | Moderate |
| `pcodec` | Log-normalized, PCA embeddings, float layers | Best for floats (~4.1–4.7×) | Moderate (19–39% slower than Zstd) | Slower (35–40% slower than Zstd) |
| `lz4` | Speed-critical pipelines | Lower (~2.2–3.1×) | Fast | Fastest compressed |
| `none` | GDS bypass, debugging | 1× (no compression) | Fastest (I/O bound) | Fastest |

For storage-constrained workflows with normalized float data, explicitly selecting `pcodec` gives the best compression. For latency-sensitive pipelines, `fast`, `zstd`, or `lz4` are better choices.

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
| `SkippedUnsKey { key, reason }` | `read_uns` / `read_uns_entry` (ingest); `write_uns_value` (export) | **Ingest:** a `uns` entry was unrepresentable; skipped under default `strict_uns=false`, an error on the first occurrence under `strict_uns=true`. **Export:** the key is not usable as an HDF5 member name (empty, `.`, `..`, or containing `/`, which HDF5 would resolve as a *path*), so the entry is dropped. `strict_uns` is an ingest option and does not govern the export case. |
| `UnsupportedUnsDataframeColumn { key, column, reason }` | `read_uns_dataframe_group` | **Ingest.** One column of a `uns` pandas DataFrame had no lossless `uns` encoding (anndata's `nullable-integer` / `nullable-boolean` / `nullable-string-array`, or an unrecognised column encoding) and was left out of the reconstructed frame. The index, the column order and every other column are intact. Under `strict_uns=true` this is an error instead. |
| `UnsExportedAsRawEnvelope { key, reason }` | `try_write_uns_dataframe` | **Export.** A `uns` pandas DataFrame had no faithful h5ad dataframe spelling and was written as a raw `__scx_type__` envelope subgroup. No data is lost; anndata reads it back as a nested dict rather than a DataFrame. |
| `FlattenedUnsSparse { key, format }` | `read_uns_entry` | A `uns` scipy-sparse matrix (`encoding-type` = `"csr_matrix"` / `"csc_matrix"` / `"coo_matrix"`) was preserved as a nested dict of `data` / `indices` / `indptr` arrays rather than reconstructed as a sparse matrix — the sparse type tag is not restored on read. Data is not dropped. |
| `SkippedColumn { group, name, reason }` | `read_dataframe_group` | An obs/var column could not be read (unsupported encoding-type or read error); skipped. The column is absent from output. |
| `SkippedObsm { name, reason }` | `read_obsm_at` | An obsm/varm embedding could not be read; skipped. |
| `LayerSkipped { name, reason }` | `read_layers_at` / streaming layer reader | A layer was skipped (open failed, shape mismatch with `/X`, or width exceeds `u32::MAX`). |
| `DenseSparsified { path, density }` | Dense h5ad streaming reader | Dense `/X` slab was sparsified during streaming. Reports density to help users decide whether dense storage is worth keeping. |
| `DuplicateCoordinatesMerged { count, policy }` | CSC h5ad streaming reader | CSC input contained duplicate `(row, col)` coordinates; values were summed (scipy `sum_duplicates` semantics). |
| `ModalityTypeInferred { name, modality_type }` | h5mu streaming reader | Modality name → `ModalityType` was inferred by name; override via `--modality-types NAME:TYPE` / `modality_types={...}`. |
| `MissingPresetIndexColumn { column }` | Predicate-index builder | A preset (`cellxgene` / `perturbseq` / `training`) referenced an obs/var column not present in the source; preset misses warn, conversion continues. |
| `PresetNoColumnsMatched { preset, axis, missing }` | Predicate-index builder | `--index-preset <preset>` matched **none** of the source obs/var columns. The user almost certainly picked the wrong preset for the input format; the per-column warnings are collapsed into a single actionable diagnosis. |
| `UnsupportedIndexColumn { column, reason }` | Predicate-index builder | A user-forced (`--index-obs` / `--index-var`) or preset column has an unsupported dtype; forced columns hard-error, preset columns warn and skip. |
| `PredicateIndexSkippedMultimodal` | Predicate-index builder | Predicate indexes are unimodal-only on the read side today; emitted (and indexes skipped) when conversion input is multimodal. |
| `CscSkippedStreamingMultimodal { modalities }` | Streaming multimodal writer | CSC sidecar emission was skipped because streaming multimodal conversion does not support CSC output. Run `scx build-csc` on the output to add the sidecar. |
| `BitmapSkipped { reason }` | Detection bitmap auto policy | `--bitmap auto` rejected emission (e.g. `n_vars > 1_000_000`, estimated bitmap size > 15% of encoded CSR, dense X). |
| `DroppedObsp { name, reason }` | h5ad ingest (obsp/varp routing) and `scx merge` (multimodal) | A pairwise `obsp`/`varp` matrix could not be preserved: on ingest, a CSC or otherwise unsupported pairwise layout is dropped (CSR is stored directly; a **dense** pairwise matrix is preserved as nonzero COO, not dropped); on `scx merge`, a **modality-scoped** `obsp`/`varp` graph is dropped (the format has no per-modality pairwise reader), as is a **CSR-backed** `obsp` in any scope. A file-scope COO `obsp` is carried and rebased into the merged obs axis since Phase 5b — the older "axis semantics don't compose" was true of merge before that. Default-dropped with a warning. |
| `MappingPeakFootprintHigh { mapping, estimated_bytes, budget_bytes }` | `pyscx.from_anndata` | A single mapping's estimated in-memory footprint exceeds `memory_budget`. |
| `EagerAssemblyMemoryHigh { estimated_bytes, budget_bytes, nnz, value_bytes, index_bytes, dense_shape, dense_bytes }` | `Experiment.to_anndata` | Estimated eager assembly footprint exceeds `memory_budget` (default 8 GiB). Warn-only, does not block. `estimated_bytes` covers **assembly** — `X`, `adata.raw` when the read includes it, and obs/var — and nothing that reads the matrix afterwards, so it is a floor on process RSS, not a job size; the message says so. Every figure the message prints is derived from these rather than assumed: `value_bytes` is the caller's `data_dtype` width and `index_bytes` is scipy's choice from `max(nnz, n_rows)`, so an `f32` CSR reads as 8 or 12 B/nnz and a `float64` one as 12 or 16. `nnz` is `None` when the call assembles no host `X` (the `to_gpu_anndata` device path), and `index_bytes` is `None` where the width genuinely is not knowable — a dense request has no index array, and above `i32::MAX` physical nonzeros on a file with deletion vectors scipy decides from the live count the catalog cannot supply. `dense_shape` is the `(rows, columns)` a dense request will allocate, which under a `var_names` projection is the *selected* column count. `dense_bytes` is `n_obs × n_vars × value_width` and is `Some` only when this call assembles a host CSR `X` **and** dense would be smaller than that sparse `X` — compared against the `X` term alone, since a large `adata.raw` is assembled either way and would otherwise make a larger dense `X` look like a saving. |
| `Hdf5NotThreadsafe` | Parallel streaming reader fallback | libhdf5 was not built thread-safe; parallel streaming fell back to the sequential coordinator. |
| `DroppedRaw { raw_n_vars }` | `Experiment.to_anndata` | The file carries an `adata.raw` matrix but the current reconstruction mode (obs-filtered query, backed mode, or deletion-vectors active) cannot reproduce raw's obs-axis filtering, so raw is omitted. The on-disk raw sections are preserved. Pass `to_anndata(raw=False)` to opt out of raw entirely — no rebuild and no notice. |
| `DroppedRawOnWrite { raw_n_vars, reason }` | `pyscx.from_anndata` with SCX-backed / lazy `X`; `from_h5ad` / `scx convert` with `--sort-by` / `--group-by` | The **source** carries raw that the file being written will not. Distinct from `DroppedRaw`: its "on-disk raw sections are preserved" reassurance is true of the source and says nothing about the output, where raw is gone for good. `reason` is supplied **per call site**, because the doors differ — the SCX → SCX rewrite loses raw because the in-memory AnnData does not hold it (convert from the h5ad instead), while reorder-on-convert loses it because raw streams unpermuted (convert without the reorder instead). A single baked-in remedy would be wrong on one of them. |
| `DroppedRawVarm { keys }` | h5ad ingest and `pyscx.from_anndata` | `adata.raw.varm` is not representable — the raw section family stores `raw/X` and `raw/var` only — so raw's own var-axis mappings are dropped. `raw.X` / `raw.var` are unaffected. |

## Round-trip fidelity

What survives an `h5ad → scx → h5ad` (or `→ AnnData`) conversion, and what is
lossy or dropped. **Status** is one of: **preserved** (faithful round-trip),
**lossy** (round-trips with a documented value/structure change), or **dropped**
(not written; surfaced as a warning). The **Warning** column names the
[`ConvertWarning`](#conversion-warnings-convertwarning) that fires; "—" means no
warning (the behaviour is by-design and documented here). Nothing in this table
is lost *silently*.

| Feature | Status | Warning | Notes |
| --- | --- | --- | --- |
| `X` counts / values | lossy | — | On-disk `u8`–`u32` ↔ in-memory `f32`; `float64` `X` is **downcast to `f32`** to match scipy CSR zero-copy. Integer counts are bit-exact. |
| obs/var columns (numeric, string, bool, nullable, categorical) | preserved | — | Nullable int/string/bool and categoricals round-trip via anndata's nullable-group / categorical encodings. |
| obs/var **ordered** categoricals | preserved | — | The `ordered` bit + category order round-trip both `h5ad → scx → h5ad` and `pyscx.open(...).to_anndata()` (carried in Arrow field metadata, re-applied to the reconstructed pandas factor). Order is the **declared** one, independent of the order the values happen to appear in the data. |
| obs/var declared-but-unused categories | preserved (dropped when a row filter is active) | — | A category no row uses is part of the declared factor and survives an ordinary export. It is dropped only when the export applies a row filter — a deletion vector, or `obs_mask=` / `min_counts=` on `pyscx.to_h5ad` — matching pandas' `remove_unused_categories()` on a subset. |
| obs/var **MultiIndex** | lossy | — | Only the single pandas `_index` is preserved; additional index levels are not carried. |
| unreadable obs/var column | dropped | `SkippedColumn` | Unsupported encoding-type or read error; column absent from output. |
| obsm / varm embeddings | preserved | `SkippedObsm` (on failure) | Dense embeddings round-trip; an unreadable embedding is dropped with the warning. |
| layers | preserved | `LayerSkipped` (on failure) | CSR layers round-trip; a layer whose shape disagrees with `/X` (or whose width exceeds `u32::MAX`) is dropped with the warning. |
| **dense** `obsp` / `varp` | lossy | — | Ingested as nonzero **float32 COO** (only nonzeros stored); both `to_anndata` and `to_h5ad` re-emit it as a **sparse** matrix (a dense input becomes sparse; values identical). |
| CSR `obsp` / `varp` | lossy | — | Round-trips `h5ad → scx → h5ad` (and via `to_anndata`) as **float32 CSR**; values downcast to `f32`. Under deletion vectors, `obsp` is filtered on both axes; `varp` (var axis) is never obs-deleted. |
| CSC / unsupported `obsp` / `varp` | dropped | `DroppedObsp` | CSC and other non-CSR/non-dense pairwise layouts are dropped on ingest. |
| `adata.raw` | preserved² | `DroppedRaw` / `DroppedRawOnWrite` (some modes) | Round-trips raw counts bit-exact with the wider var axis through **both** write doors — `h5ad → scx → h5ad` and in-memory `pyscx.from_anndata` / `pyscx.write`. ² Dropped, with a warning, under obs-filtered `to_anndata`, backed mode, and deletion-vector-active files, on the SCX-backed / lazy-`X` rewrite, and under reorder-on-convert (`--sort-by` / `--group-by`). `to_anndata(raw=False)` opts out of raw and its warning. See [`adata.raw`](#adataraw). |
| `adata.raw.varm` | dropped | `DroppedRawVarm` | Raw's own var-axis mappings have no section in the raw family (`raw/X` + `raw/var` only). `raw.X` and `raw.var` are unaffected. |
| `uns` scalars / 1-D & 2-D numeric arrays / nested dicts | preserved | — | Round-trip through the `uns` JSON representation. |
| `uns` pandas **DataFrame** | preserved | `UnsupportedUnsDataframeColumn` / `UnsExportedAsRawEnvelope` | Round-trips as a `pd.DataFrame` through both write doors, index name, column order and per-column dtypes (ordered categoricals included) intact — see [`uns` serialization](#uns-serialization). Two exceptions, each warned: on **ingest** a column in one of anndata's nullable encodings is left out (an error under `strict_uns=true`); on **export** a frame h5ad cannot spell is demoted whole to a raw envelope subgroup, losing no data but arriving at anndata as a dict. |
| `uns` scipy-sparse matrix | lossy | `FlattenedUnsSparse` | Preserved as a nested dict of `data` / `indices` / `indptr` arrays; **not** reconstructed as a sparse matrix (the sparse type tag is not restored). Data survives. |
| `uns` pickled / unrepresentable entry | dropped | `SkippedUnsKey` | Skipped under default `strict_uns=false`; `strict_uns=true` errors on the first occurrence. |
| `uns` nested more than 60 levels deep | rejected / truncated | `SkippedUnsKey` (h5ad ingest) | See [`uns` nesting depth](#uns-nesting-depth). Writing raises `ValueError` naming the key path; an over-deep h5ad `/uns` group chain is truncated at the limit with the warning (or errors under `strict_uns=true`). |

### `uns` nesting depth

`uns` container nesting — dicts, lists, tuples, and nested h5ad `/uns` groups —
is capped at **60 levels**. Real metadata is nowhere near this: scanpy's deepest
standard structure, `rank_genes_groups`, is 3 deep.

The cap exists because the walk is recursive on every path (Python→JSON,
h5ad→JSON, JSON→Python, JSON→h5ad), and exceeding the stack **aborts the
process** rather than raising — a `SIGSEGV` no `try`/`except` can catch. Cycle
detection does not help: it works by object identity, and a merely-deep tree
contains no repeated object.

The specific value is derived, not chosen. `serde_json`'s serializer has no
depth limit but its parser stops at 127 levels, so before the cap the writer
could emit an `uns` section that no reader — including SCX's own — could ever
parse back. Under `uns_format="tagged"` a tuple is stored as
`{"__scx_type__": "tuple", "data": [...]}`, two JSON levels per Python
container, so 60 containers is the worst case that still fits.

Writers (`from_anndata`, `set_uns`, `modify_metadata`, h5ad ingest) enforce 60.
Readers stop at 127 instead, so a file written before the cap existed still
opens if its `uns` is parseable; one written deeper than 127 raises a message
saying so rather than `serde_json`'s bare `recursion limit exceeded`.

Underneath both sits a storage gate at 127: the section writer itself refuses
an `uns` tree deeper than the reader will parse, whatever built it. That is
what makes "an SCX file always has a readable `uns`" true for paths with no
Python or h5ad in them — `scx merge`, `scx subset`, and any Rust caller
handing the writer a tree it assembled itself.

## `adata.raw`

`adata.raw` (pre-normalization counts on its own, usually wider, var axis) is
preserved end-to-end. `pyscx.from_h5ad` ingests the h5ad `/raw` group into a
dedicated raw section family (CSR shards on `X`'s obs axis + a `raw/var` Arrow IPC
section; see [docs/format.md § raw section family](format.md#adataraw-raw-section-family)).
On the streaming convert path (`stream=True`, the default) `raw/X` is read and
written shard-by-shard through the same coordinator as `/X`, so peak RSS stays
bounded; the materializing path (`stream=False`) reads it eagerly.
The in-memory `pyscx.from_anndata(adata)` / `pyscx.write(adata, path)` path writes the
same section family from `adata.raw.X` + `adata.raw.var`, so both doors into SCX
preserve raw identically. It is materialized eagerly (the AnnData is already
resident), unlike the h5ad streaming ingest. Raw keeps its own column count: the raw
shards' `index_dtype`, minor-axis extent, and codec are all resolved against
`raw_n_vars`, independent of `X`.

`pyscx.open(...).to_anndata()` reconstructs `adata.raw` (an AnnData with raw `X` +
`var`), and `pyscx.to_h5ad` re-emits `/raw/X` + `/raw/var`, so
`h5ad → scx → h5ad` round-trips raw with integer counts bit-exact and the wider var
axis intact. Raw is **dropped with a `DroppedRaw` warning** under obs-filtered
`to_anndata`, backed mode, and deletion-vector-active files (those modes do not yet
re-filter raw's obs axis).

`to_anndata(raw=False)` opts out: raw is not rebuilt on the paths that would rebuild
it, and the `DroppedRaw` notice is not emitted on the three that would drop it. Use it
when raw is irrelevant to the workload — otherwise the notice fires on every backed
open of a raw-bearing file. `raw=True` is the default and is unchanged.

A **gene projection does not narrow raw**: anndata hands `.raw` only the obs index when
slicing, so `to_anndata(var_names=[...])` returns raw on its full gene axis, exactly as
a plain read does. That makes raw the dominant cost of a projected read of a
raw-bearing file (the projected `X` is small and raw is not), which is what `raw=False`
is for — measured on a 20-shard fixture, a three-gene read peaked at 61.6 MB with raw,
20.5 MB once raw stopped being copied along the way, and 3.1 MB with `raw=False`.

Three write paths still cannot carry raw and say so rather than dropping it in silence.
The first two warn with `DroppedRawOnWrite`, whose `reason` is supplied per call site —
they lose raw for different causes, so a single baked-in remedy would misdirect one of
them:

- **SCX-backed / lazy `X` rewrite.** `pyscx.from_anndata` writes the sections the
  in-memory AnnData holds, and backed reconstruction sets `.raw` to `None`, so a
  `pyscx.open(f).to_anndata(backed=True)` → `from_anndata` round-trip loses raw. Keyed
  off the *source* file rather than the object — `DroppedRaw`'s "the on-disk raw
  sections are preserved" would be misleading here, since the file being written has
  none. Remedy: convert from the h5ad, or from an in-memory AnnData whose `.raw` is set.
- **Reorder-on-convert** (`from_h5ad(..., sort_by=…)` / `scx convert --sort-by` /
  `--group-by`). The permutation is applied to `X` / `obs` / `obsm` / `layers` while raw
  is streamed in source order, so carrying it would leave raw's rows attached to the
  wrong cells. Remedy: convert without the reorder. (`from_anndata(sort_by=…)` on an
  in-memory AnnData is rejected outright, so it cannot reach this.)
- **`merge` / `compact` / `sort` / `append` / `preprocess`**, which need raw's obs axis
  filtered in lockstep with `X`. The first four warn; `append` and `preprocess` refuse.

`adata.raw.varm` is not stored — the raw family holds `raw/X` and `raw/var` only — and
is dropped with a `DroppedRawVarm` warning on both the h5ad ingest and the in-memory
write. A raw whose `X` row count disagrees with `adata.n_obs` is **rejected**, not
written: `adata.raw.shape` reports the parent's `n_obs` rather than
`adata.raw.X.shape[0]`, so a misaligned raw looks correct from Python.

## Memory budgets

`MemoryBudget::parse(s)` (`scx-format-io/src/mem.rs`) is the shared parser
behind `--memory-budget` and `build-csc --memory-limit` (CLI) and the
`memory_budget=` kwarg on `from_h5ad` / `from_h5mu`. It caps dense row
slabs in the h5ad streaming reader, CSC external-transpose buffers when
CSC-on-disk exceeds the budget, the **CSC sidecar** transpose chunk (which
therefore affects `n_csc_shards`), and the parallel streaming reader's
worker derate. It lives in `scx-format-io` (re-exported as
`scx_convert::MemoryBudget`) so sibling crates such as `scx-ops` —
which owns `build-csc` — can share it without a dependency cycle.
It parses a byte count and nothing more; what a budget *buys* is the
allocation table below.

Accepted forms:

- bare byte counts (`"1048576"`, `1048576`),
- binary-prefix shorthand `K` / `M` / `G` / `T` (= `KiB` / `MiB` / …),
- explicit binary prefixes `KiB` / `MiB` / `GiB` / `TiB`.

Decimal prefixes (`KB`, `MB`, `GB`, `TB`) are **rejected** to avoid
1000-vs-1024 ambiguity. When the requested budget cannot fit even one
shard's metadata plus one worker, conversion refuses to start with
an actionable error rather than OOMing partway through.

### The allocation table

What a budget *buys* is declared in one place, `scx-convert/src/budget.rs`,
rather than derived at each site. Two things were previously tangled in a
single division and are now separate:

- a **share** — what fraction of the budget one concurrent unit may claim.
  One in-flight shard takes a quarter, which is what leaves room for the
  derate to grant more than one worker.
- a **cost model** — how many bytes that unit actually holds, across the
  worker's *whole* phase rather than one stage. A CSR shard costs 48 B/nnz plus
  the indptr: 8 for the resident payload, 8 for the encoder's own copy of the
  values, and 32 for the framed encode's live buffers (it holds every row
  group's encoded bytes alongside the streams assembled from them, and
  `codec="auto"` runs two candidate codecs concurrently). A dense source
  element costs 44 B: 12 for the f32 slab plus the sparsified indices and
  values at exact capacity, plus the same 32 for the nonzero it may become.

Reservations are declared per *phase*, and only reservations in the same phase
are concurrent — the CSC external transpose claims half the budget for a
column chunk in pass 1 and a quarter for bucket records in pass 2, and those
never coexist. A unit test asserts that each phase's concurrent claims sum to
at most the whole budget.

Seven claims are declared but **not enforced**, and the `enforced` flag is the
difference between "we sized this" and "nothing exceeds this" — read it before
quoting a row as a guarantee:

- the two CSC **bucket** rows are sized from the *mean* nnz per row, so a
  right-skewed sequencing-depth distribution overshoots them;
- the CSC **sidecar** row's budget sizes the transpose chunk, while the writer's
  full-length index and value copies and the encoder's streams are live next to
  it, and the rebuild path additionally retains every source shard. It controls
  column and shard sizing, not a ceiling;
- the three per-shard **ingest / export** rows: on ingest, `encoded <= payload`
  is an estimate rather than a codec guarantee (frames can expand, a codec
  holds its raw, shuffled and compressed planes at once, the encoded indptr is
  priced at zero for an `nnz = 0` shard, the detection bitmap is uncharged, and
  readers on the trait default take a density guess); on export,
  `filter_shard` holds a second indptr and, when masked, doubling-grown output
  buffers alongside the originals;
- the CSC **column-chunk** row's scan always reads the first column whole before
  testing the budget, so one wide column exceeds the share (on a large atlas
  that is an ordinary ubiquitous gene), and its floor exceeds the share for
  budgets under 128 bytes.

They are named in the table so each gap is visible rather than silent, and a
unit test pins the count so an eighth cannot arrive unannounced.

The three per-shard **ingest / export** rows are still on that list, but their
*estimates* changed. Each used to size a reader working set while the worker
also held the encoded shard. The two ingest rows now size the whole worker
phase — 3x larger on sparse, 3.7x on dense — and the export row is sized from
its own decode model rather than borrowing the ingest one. The share did not
change; what widened is the cost model the share is applied to. So a budget now
buys **fewer concurrent workers and smaller shards** rather than the same
concurrency over an unpriced buffer, and a budget too small to hold one whole
phase is refused outright instead of being silently over-committed. Budgets are
opt-in (`memory_budget` defaults to unset), so nothing derates that did not ask
to.

They remain unenforced because a better estimate is not a proof: each row names
the terms it still does not bound (codec frame expansion and intra-codec planes
on ingest, `filter_shard`'s second indptr and doubling-grown buffers on
export).

Three different things are called "no budget", and they are not
interchangeable: an unset `memory_budget` means *no cap at all*; pyscx's
eager-materialisation path warns above 8 GiB; and the CSC sidecar builder
defaults to 4 GiB when no budget is given.

### Ops that bound themselves without a budget knob

Some ops are bounded structurally rather than by a `memory_budget=`, because
there is nothing to trade off — they stream, or they do not.

`pyscx.obs_import` / `attach_obs_columns` / `doublet_import` / `cellbender_import`
(and their `scx` subcommands) rewrite the target's `obs` **one shard at a time** whenever the
target's obs is sharded — anything `from_anndata` wrote above
`shard_target_rows`, and anything `merge` or `append` produced. Peak is one obs
shard plus one row index and one key string per target cell, so landing a
100-cell annotation on a 10M-cell atlas does not cost the atlas's obs table.
The input's obs shard boundaries are preserved rather than re-derived.

A target whose `obs` is a single legacy `ObsMetadata` section has no per-shard
reader, so the whole table is assembled; the op warns and reports
`obs_streamed = false` in its summary. Run `scx optimize` on such a file first
to migrate obs to the sharded layout. The remaining unbounded paths — the key
diagnosis printed on a *failed* join, an `obsm` embedding when one is requested
(`cellbender_import(latent_embedding=True)` is the reachable case), and the
source table itself — are enumerated in
[docs/operations.md § Memory: bounded on the target, resident on the source](operations.md#memory-bounded-on-the-target-resident-on-the-source).

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
  the column is missing or has an unsupported dtype. Unsupported means
  anything that is neither a string nor a number: `Boolean`, and
  equally a boolean-valued pandas `Categorical`
  (`Dictionary(_, Boolean)`), which is a dictionary-encoded `Boolean`
  and no more indexable than the plain column.
- Preset columns warn (`MissingPresetIndexColumn` /
  `UnsupportedIndexColumn`) and are skipped without aborting the
  conversion.
- Auto-indexing picks up categorical-like columns with cardinality
  `≤ index_auto_threshold` (default `1000`).
- **A high-cardinality numeric obs column is indexed, not skipped.** The
  high-cardinality cap applies to categorical columns; the numeric branch
  ignores it deliberately, because a numeric index is **one entry per shard**
  (that column's `[min, max]` over the shard's rows) and so costs the same
  bytes for a million distinct values as for ten. Force-listing
  `--index-obs total_counts` on an atlas is cheap and gives Level-1 shard
  pruning on that column.

  The cap does still apply, unchanged, to **`var`** (`--index-var`), which is
  built by the batch builder. It also applies to obs on the batch builder
  itself (`build_obs_predicate_index_bytes`), where a capped column is
  **dropped from the index entirely** — not merely reported: it is absent from
  `indexed_columns`, absent from the serialised bytes, and gets no Level-1
  pruning. No conversion front end reaches that path, though: `scx convert`
  and `pyscx.from_anndata` both go through
  `build_and_write_conversion_predicate_indexes`, which hands obs to the
  streaming builder with a one-item iterator. So the distinction is real where
  it applies, and not one users of the conversion APIs can observe on obs.
- **A column is classified by what it holds, not by how it is stored.**
  pandas writes every `Categorical` as an Arrow dictionary, whatever the
  categories are, so the dictionary's *value* type decides:
  `pd.Categorical(["A", "B"])` → the categorical index (Level-1
  `CategoryBitset` pruning + Level-2 row-set pushdown), while
  `pd.Categorical([1, 2, 3])` → the **numeric** index, exactly as the
  plain integer column it holds (Level-1 `MinMax` pruning; numeric
  operators are residual at Level 2, so no row-set pushdown). Query
  syntax follows the value type too: `batch == 3`, not `batch == '3'`.
- Multimodal inputs emit `PredicateIndexSkippedMultimodal` and skip
  predicate-index emission entirely — the read path is unimodal-only
  today. The same skip-with-warning applies to multimodal
  `merge` / `append` / `compact`. The reason is that a `shard_id` is resolved to
  "the i-th CSR shard" **by position**, and on a multimodal file each modality's
  shards independently tile `[0, n_obs)`, so the i-th shard is ambiguous.
  Adding multimodal indexing means giving `ShardRange` a modality scope.

  Correctness does **not** rest on the file lacking an index —
  `ScxWriter::write_obs_predicate_index` is public and accepts a multimodal
  writer, so skipping emission is a convention of the high-level writers. Three
  independent guards do the work: the write side
  (`assign_csr_shard_column_stats` counts only `modality_id == 0` entries and
  returns `ColumnStatsShardCountMismatch` rather than mis-assigning), Level-2
  read (`build_plan` forces `obs_predicate_index` to `None` for any
  `modality_id != 0` pipeline), and Level-1 read, which reads per-entry
  `column_stats` and never consults an index `shard_id` at all.
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
`Experiment.detection_counts(axis="var", modality=...)` and
`Experiment.cells_expressing(gene, modality=...)`; the backed reader
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

### Which codec a shape-2 (backed) decode + encode rewrite uses

Shape 3 (lazy) is different and is covered below — do not read this subsection
as the rule for both.

With an explicit `codec=`, that codec. Without one, the codec is resolved
against the **source's shard headers**, never against its file header:

- every source CSR shard shares one codec → reuse it, so an explicit
  `codec="scx1"` / `"pcodec"` / `"none"` source round-trips unchanged;
- they differ → the per-shard adaptive heuristic
  (`select_codec_for_modality`) chooses again for each output shard.

The file header's `codec_id` is only a *default* and each shard header
overrides it (see *Codec Selection* above, and `docs/codec.md` §1), so it can
describe no shard at all — a `codec="auto"` write leaves whichever codec its
first shard picked, and files written by older versions of the streaming
h5ad → SCX converter carry a `0` (`none`) placeholder over compressed shards.
Inheriting it wrote every shard raw: on a 966,728 × 6,143 source
(nnz 2.67 × 10⁹) with such a header, dropping 537 rows turned 2.19 GB into
10.85 GB. Any row deletion disables byte-passthrough, so this reached every
subset rewrite of an affected file.

For shape 3 the rewrite is always decode + encode; transforms are
applied per shard inside `wrapper[start:end]`. **Its codec rule differs from
shape 2's:** an explicit `codec=`, otherwise per-shard adaptive selection —
never the source's codec, from its shard headers or anywhere else. The
transforms have already rewritten the values (`normalize_total` / `log1p` turn
integer counts into f32), so the codec that suited the source's integers does
not suit the output's floats; a uniform `scx1` or `none` source therefore does
**not** round-trip its codec through a transformed write. The same
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

Read the chain via `Experiment.provenance()` — each entry is a
dict with `params_json` as a raw JSON string (parse with
`json.loads`).

The full transform list is also surfaced on the wrapper itself via
`ScxLazyTransformedDataset.transforms_repr() → list[{"name","params"}]`,
which the writer uses to build the `lazy_transforms` payload.

### Zero rows and zero columns

`pyscx.from_anndata` accepts an AnnData with `n_obs == 0` (an empty QC or
guide filter, `adata[mask]` with an all-`False` mask, `pyscx.accel.subset_obs`
keeping nothing) or `n_vars == 0`, on every `X` shape above. Before 0.17 both
raised `RuntimeError: Arrow IPC contains no batches` — pyarrow's `write_table`
emits zero IPC batches for a 0-row frame; the format, the Rust writer and every
reader already handled zero shards (`scx subset` with a predicate matching
nothing has always written a valid 0-row file). What a 0-row file holds:

- **`X`'s shape and the full schema.** The header carries `n_obs = 0` and
  `n_vars`; there are **no** CSR shards (the format forbids framed zero-row
  shards — "emit no shard at all instead", [format.md § Block index](format.md#block-index)).
  `obs` and `var` are single 0-row sections with every column, dtype and
  declared category — `pd.Categorical([], categories=["A", "B"], ordered=True)`
  reads back with both categories and `ordered=True`, because the 0-row frame
  crosses the pandas → Arrow boundary as a real 0-row batch
  (`pyarrow.RecordBatch.from_pandas`), not as an empty batch rebuilt from the
  schema. `obsm` / `varm` (as `(0, k)` / `(n_vars, k)`), `obsp` / `varp` and
  `uns` survive.
- **Empty `object` columns, and the obs index, are stored as string.** With no
  values to look at pyarrow types them Arrow `null`, and a categorical that an
  anndata subset pruned to zero categories as `dictionary<null>` — types no
  populated frame produces. They are stored as `string` / `dictionary<string>`
  so the 0-row file's schema is the one its populated sibling has: `append`'s
  obs-schema check, `merge`'s column unification and a forced `index_obs=`
  all compare against it. Only 0-row frames are touched; a populated all-`None`
  column still round-trips as `null` → `object`.
- **`layers` and `adata.raw` are dropped, with a `UserWarning`** naming them.
  Both exist on disk only as their CSR shards (`has_raw` is derived from shard
  presence), and a 0-row file has none, so not even the layer's name can be
  recorded. `to_anndata()` of a 0-row file has `layers == {}` and `raw is None`.
- **No CSC sidecar and no obs predicate index**, whatever `csc=` / `index_obs=`
  say, and no warning about either: a sidecar over an empty matrix indexes
  nothing, and a 0-row obs has nothing to index (`filter_obs` scans the empty
  obs). `index_var=` on a 0-row file still indexes `var`, which has rows. The
  same rule holds for `n_vars == 0` (`csc` skipped; `index_var` builds nothing)
  and for every op that rebuilds them (`compact`, `merge`, `sort`, `build-csc`
  — see [operations.md § Empty inputs and outputs](operations.md#empty-inputs-and-outputs)).
  The SCX-backed / lazy `X` rewrite (shapes 2 and 3) applies the same
  layers / `raw` / CSC policy, but it has never built a predicate index for any
  row count: passing `index_obs=` / `index_var=` / `index_preset=` there raises
  a `UserWarning` and writes none — build it afterwards with
  `pyscx.compact(out, out2, index_obs=[...])`.

Reading one back: `to_anndata()` is `(0, n_vars)` with the obs / var frames
above; `to_anndata(backed=True).X` is a handle with `n_shards == 0` whose
`to_memory()` / `sum(axis=0)` / column selectors all work (`stored_dtype`
reports `float32`, the decode dtype, since there is no shard to read an
encoding from, while `Experiment.value_encoding` reports `"n/a"` — both
documented above); `read_obs()` / `read_var()`, `query().collect()`,
`to_h5ad` (streaming and eager) and `scx info` all answer for the empty
file. A `(n_obs, 0)` file reads back the same way with the axes swapped.

Not covered: `pyscx.from_mudata` still rejects `n_obs == 0` (it derives the
global cell count from the outer obs), and `adata.X is None` is not accepted
on any row count.

## Restricted-exec (sandbox) safety

pyscx entry points work when called from code `exec`'d under **restricted
globals whose `__builtins__` lack `__import__`** — the sandbox shape pipeline
runners use for escape-hatch `python` steps (imports forbidden, modules
pre-injected). This is a maintained guarantee, not an accident:

- CPython resolves `__import__` for C-level imports (`PyImport_Import`) from
  the **innermost Python frame's builtins**, so a native function called
  directly from such a frame would otherwise die with
  `KeyError: '__import__'` at its first call-time import.
- All call-time imports in pyscx and scx-loader go through the
  frame-insensitive `scx_loader::pyimport::import_module` (`sys.modules`
  lookup, then the core import machinery — neither consults frame builtins).
  A workspace `clippy.toml` `disallowed-methods` entry rejects bare
  `Python::import` / `PyModule::import`.
- Third-party lazy imports are handled too: rust-numpy's C-API and
  borrow-capsule inits are primed at `import pyscx` time. numpy re-imports
  `numpy._core._dtype` on **every** `dtype.name` / `str(dtype)` /
  `repr(dtype)`, so pyscx never stringifies a dtype on a native path: the
  write-path dtype-name reads route through the pure-Python trampoline
  `pyscx._frame_safe.dtype_name` (whose frame carries real builtins), and
  the backed/lazy `__getitem__` selector paths read `dtype.kind`, a plain C
  descriptor that triggers no import.

Covered surfaces (regression-tested in `pyscx/tests/test_sandbox_exec.py`,
which runs each one inside `exec(code, {"__builtins__": <no __import__>})`):
`pyscx.write`, the native `pyscx.pyscx.from_anndata`, `pyscx.obs_import`,
`pyscx.attach_obs_columns` (a positional DataFrame attach),
`pyscx.modify_metadata` (with an obs replacement) / `pyscx.set_uns`,
`open(...).to_anndata()` eager and `backed=True`, boolean-mask and
integer-array indexing on backed and lazy `X`, and the import helper's
core-import fallback (via `sys.modules` eviction — there is deliberately no
importable probe hook, since an arbitrary-name importer on the module would
re-create the `__import__` the sandbox removed).
Note that numpy itself is *not* sandbox-safe (`str(x.dtype)` in step code
will still raise); the guarantee covers pyscx's own entry points.

## BackedCsrReader (`scx-format-io/src/backed/csr.rs`)

- `new(reader, cache_shards)` — Create backed reader from `ScxReader` with LRU shard cache
- `cache_shards()` → `usize` — The requested LRU count cap this reader was built with (`0` = no cache; the cache clamps its own capacity to ≥ 1 internally). Read back by the pyscx handles' `cache_shards` getter.
- `stored_value_encoding()` → `Result<Option<ValueEncoding>>` — The on-disk value encoding of this reader's shard family (X, the layer, or the modality it is scoped to — it walks the same `shard_entry` table every read does): one 76-byte header read per shard, no payload decode, memoised. A uniform family reports its own encoding; a mixed one the widest via `ValueEncoding::widest` (any float ⇒ `Float32`, else the widest integer); `None` for no shards. The encoding lives only in the shard header — `ShardStats` has no encoding field, and a `value_max` of 0 cannot tell a float shard from an all-zero one — so this is the one place a caller learns whether the stored values are integer counts without decoding.
- `read_rows(start, end)` → `ScxCsr` — Decode rows `[start, end)` from the overlapping shards into one pre-sized result. A range that fits the LRU is warmed and copied from the cache; a bulk range (more full shards than `cache_shards`, e.g. the whole matrix) decodes uncached in parallel chunks of `cache_shards`, keeping the LRU's entries as they were (the residents it copies are promoted, as any hit is) — peak = result + up to `cache_shards` shards in flight, on top of whatever the LRU already holds (itself capped at `cache_shards`). `end > n_obs` is an error, and so is a catalog whose shards do not tile the range exactly (a gap, or the overlapping modalities of an unscoped reader on a multimodal file).
- `read_row_indices(indices)` → `ScxCsr` — Decode specific rows by index (fancy indexing), in request order, duplicates allowed. Assembles the result once: an indptr-only prescan of each touched shard sizes the output exactly, then the `read_rows_with` scatter copies each row into place — peak = result + the shard cache + up to `cache_shards` shards decoding in flight while `warm_shards` fills it (at most `2 × cache_shards` decoded shards beside the result on a full cache) + the block-index transient, which is one pool-width **chunk** of row groups (they are decoded a chunk at a time and each chunk is scattered before the next is decoded, so an over-budget gather holds one chunk rather than all its groups). A sparse request on a row-group-framed shard that is not resident whole decodes only the touched row groups (block index) and retains **those groups** in the same LRU under the same byte budget — the whole shard is never inserted — so a repeated small gather over one region is served from cache (`row_group_hits`). This method decides admission for itself: its groups are retained only if all of them fit the byte budget (sized from the block index before any decode); over budget, it decodes and drops. `read_row_indices_with_admission(indices, admit_row_groups)` is the same read taking the verdict from the caller instead — the pair is exactly `read_rows_with` / `read_rows_with_admission` below, and `scx-loader`'s cell-set gather reads a whole plan through it so the L1 gather and the L2 warm act on one verdict. Whether the read takes the block-index route or decodes the shard whole is decided from the **distinct** rows it touches, not from the request positions, so a plan that repeats rows does not read as a dense request. The prescan reads a resident shard's indptr rather than decoding it again, which is invisible except as time: it counts no hit and does not touch LRU recency. An out-of-range row is an error (it used to be dropped silently).
- `read_rows_with(rows, scatter)` / `read_rows_with_admission(rows, admit_row_groups, scatter)` → `()` — The scatter primitive under `read_row_indices` and the plan loaders: `scatter(i, indices, data)` once per requested row with zero-copy views into the decoded shard or row group. `read_rows_with` decides row-group admission per call, as above; `read_rows_with_admission` takes the verdict from the caller — `scx-loader`'s prefetch engine decides once per plan, over every gather the plan will make and every file it touches, against `budget / (lookahead + 1)`, so the L1 gathers and the L2 warm act on one admission verdict (the warm pre-decodes the eligible subset of what the gathers may retain) and a plan of individually-fitting gathers whose union is over budget cannot retain that union. The verdict sizes the plan's whole footprint in the shared LRU — its row groups and the shards it takes whole. A plan that fits its share is retained whole; a plan that does not keeps only the row groups another plan of the lookahead window also touches, hottest-first and only while they fit **the room that share leaves after the shards the plan takes whole** (`share - whole_shard_bytes` — those shards are inserted regardless), with its cold tail decoded and dropped. A resident group is served as a hit either way; `Admit::None` stops every miss from being inserted and `Admit::Groups` all but the named keys. ⚠️ **Scatter order is per pass, not sorted by row**: every full-shard fallback fires before every block-index row group, so on a mixed request a later full-shard row precedes an earlier block-index one. The `i` argument — the position in `rows` — is the only ordering guarantee, and every in-tree consumer indexes by it. `read_rows(start, end)` applies the same idea to its own edge windows: one verdict over every row-range window of the read.
- `read_shard_cached(idx)` → `ScxCsr` — Read shard through LRU cache (clones on hit)
- `read_shard_uncached(idx)` → `ScxCsr` — Read shard bypassing cache (preferred for streaming)
- `row_sums()` / `col_sums()` — Streaming per-row/column sums
- `row_nnz()` / `col_nnz()` — Streaming per-row/column NNZ
- `row_var()` / `col_var()` — Streaming per-row/column variance
- `row_max()` / `col_max()` / `row_min()` / `col_min()` — Streaming extrema

- `total_nnz()` — Total NNZ across all shards
- `col_means_and_sum_sq(zero_center)` — Single-pass column statistics for PCA.
  Decodes one shard at a time on the calling thread; a caller that can name a
  `Sync` source should prefer `scx_format_io::col_means_and_sum_sq_prefetched`,
  which overlaps decode across shards and is bit-identical to it (the trait
  default is that function's test oracle, so the two cannot drift). Both CPU and
  GPU PCA take the prefetched form.
- Masked variants (deletion-vector aware): `col_sums_masked(kept_rows)`, `col_nnz_masked(kept_rows)`, `col_max_masked(kept_rows)`, `col_min_masked(kept_rows)`, `col_var_masked(kept_rows)`

### Overfull-axis rejection in the aggregations

`row_var` / `col_var` and the four extrema fold implicit zeros into their result,
and so have to know how many there are: `extent − nnz`, where `extent` is
`n_cols` on the row axis and `n_obs` (or the kept-row count) on the column axis.
That is only valid on a **canonical** CSR — `docs/format.md` § "v3 canonical CSR
invariant" requires per-row column indices to be strictly increasing, so an axis
of `extent` cells holds at most `extent` entries. Ordinary reads do not verify
that: the decode seam bounds index *values*, and only `scx validate --deep`
checks ordering. They therefore return
`ScxError::Csr(CsrError::NonCanonicalAxis { extent, nnz })` when **an axis holds
more stored entries than it has cells**, rather than reporting a number derived
from a wrapped count. Surfaced through pyscx as a `RuntimeError` from
`X.var(axis=0)` / `X.var(axis=1)` / `X.max(axis=…)` / `X.min(axis=…)` on a backed
matrix, with or without a column projection.

⚠️ **That predicate is narrower than "the shard is canonical", and the
difference matters.** `nnz > extent` proves a repeated coordinate, but the
converse does not hold: a sparse row with a couple of repeats stays under its
extent and is *not* detected — a 1×3 row storing indices `[0, 0]` yields a
variance computed from both entries plus one implicit zero, with no error. So
these are overfull-axis guards, not uniqueness guards.

Which routes reject, and which answer:

| Route | On an overfull axis |
|---|---|
| `X.var(axis=0)` / `X.var(axis=1)`, CSR or CSC, projected or not | `RuntimeError` |
| `X.max(axis=0)` / `X.min(axis=0)`, projected or not | `RuntimeError` |
| `X.max(axis=1)` / `X.min(axis=1)`, **unprojected** | `RuntimeError` |
| `X.max(axis=1)` / `X.min(axis=1)`, **with a column projection** | **answers** — that branch materializes via `to_memory()` and defers to scipy, so no guard runs |
| CSC Wilcoxon DE (`scx-accel`, nnz fast path) | `AccelError::InvalidInput` when the **labelled-nonzero** count overruns; explicit stored zeros are excluded from that count, so a column overfull purely with duplicate zeros is not caught |
| `X.var(axis=None)` scalar, unprojected | clamped to `0.0` — no per-column count exists on that path |
| `scx-accel` PCA total variance | `saturating_sub`, absorbed |
| A duplicate leaving the axis under its extent | not detected anywhere |
| An **unsorted** overfull row seen through a column projection | not detected — `project_csr` drops indices smaller than one already seen, so the projected row is no longer overfull |

Variance results preserve `NaN`: the clamps use `if v < 0.0 { 0.0 } else { v }`
rather than `v.max(0.0)`, which would return `0.0` for a NaN input because
Rust's `f64::max` ignores NaN.

Use `scx validate --deep` when you need an actual canonicality verdict on a file.
None of the read-path aggregations is a substitute for it.

## ShardSource Trait (`scx-format-io/src/shard_source.rs`)

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

## BackedCscReader (`scx-format-io/src/backed/csc.rs`)

Column-major counterpart to `BackedCsrReader`. Streams CSC sidecar
shards from disk through the same `ShardCache` the CSR and dense readers use —
a byte-budgeted LRU plus a singleflight table, so concurrent readers of one cold
shard decode it once.

- `new(reader, cache_shards)` — Create from `ScxReader`. `cache_shards = 0` disables caching (one decode per call). Count-only: the byte budget is `usize::MAX`, so only the shard count bounds the cache
- `with_byte_budget(reader, cache_shards, bytes)` — As `new`, but also caps the cache in bytes. A decoded `ScxCsc` is measured by its components (`indptr.len()*8 + indices.len()*4 + data.len()*4`), the same model `IndexPlanLoader`'s memory-budget auto-tune uses
- `index()` — Per-shard column-range index, sorted by `col_start`
- `n_shards()` / `n_obs()` / `n_vars()` — Dimensions
- `read_shard_uncached(idx)` → `ScxCsc` — Single CSC shard, bypass cache
- `read_shard_cached(idx)` → `Arc<ScxCsc>` — Single CSC shard, through cache and singleflight
- `read_csc_columns(col_range)` → `ScxCsc` — Decode only shards overlapping the half-open range; partial-overlap shards are sliced post-decode
- `read_csc_columns_subset(cols)` → `ScxCsc` — Gather columns from a sorted unique `&[u32]`; contiguous runs share a single decode
- `enable_metrics()` / `metrics()` — Hits / misses / evictions / decoded-bytes counters. **Idempotent**: repeat calls return the same accumulating handle rather than resetting. To measure an interval, snapshot and subtract — the same contract `BackedCsrReader` and `BackedDenseReader` have

## ColumnShardSource Trait (`scx-format-io/src/shard_source.rs`)

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
Streaming SCX → SCX append. Reads one source CSR shard at a time (or copies raw bytes verbatim when codec / value encoding / index dtype / per-modality `n_vars` all match), avoiding materializing the entire source matrix in memory. **Multimodal targets are rejected** with `OpsError::MultimodalUnsupported` (deferred — a single-modality append would leave sibling modalities under-covering the shared obs axis; extract → append → re-merge instead). CSC sidecars are dropped on append.

### `scx_ops::mark_deleted(path, cell_indices) → Result<u64>`
Logical deletion via Roaring Bitmap deletion vectors. Returns total deleted count.

### `scx_ops::compact(input, output) → Result<()>`
Rewrite file reclaiming deleted/orphaned space.

### `scx_ops::optimize(input, output, codec, obs_shard_policy) → Result<()>`

Three entry points, and the difference is the output's format version:

- `optimize(input, output, codec, obs_shard_policy)` — delegates with no framing, so it writes an **unframed v3** file.
- `optimize_with_framing(…, framing) → Result<OptimizeStats>` — `Some(FramingConfig)` with `row_group_rows > 0` writes framed **v4**; `None` is the v3 path above.
- `optimize_with_budget(…, framing, memory_budget) → Result<OptimizeStats>` — the same, plus a cap on what the parallel shard re-encode holds in flight. The other two delegate here with `None`.
Faithful upgrade of a single-modality file: re-encode + `canonicalize_csr`
every CSR-backed shard (X / layer / obsp-CSR) and row-group-frames it, stamping
`format_version=4`, without a full reconvert. No decode sidecar is written. Preserves row layout, obs/var, obsm/varm/obsp/varp, uns, predicate
indexes, and the deletion-vector section (does not apply deletions); drops the
CSC sidecar. Multimodal inputs are rejected (use `compact`). `obs_shard_policy`
(`ObsShardPolicy::{Off, Auto, Always}`, default `Auto`) migrates a
*single-section* legacy obs to the sharded `ObsMetadataShard` layout: `Auto`
shards when `n_obs > shard_target_rows` (the `from_anndata` threshold), `Always`
always shards, `Off` keeps the single section (the historical 1:1 copy).
Already-sharded obs is always stream-preserved regardless of the policy. See
[docs/operations.md § Optimize](operations.md#optimize).

### `scx_ops::rollback(path) → Result<()>` / `rollback_to(path, seq) → Result<()>`
Revert to previous (or specific) manifest version — header-only update.

### `scx_ops::merge(inputs, output) → Result<()>`
Streaming merge of multiple SCX files into one.

### `scx_ops::sort(input, output, opts: &SortOptions) → Result<SortSummary>`
Global obs-axis row reorder. `SortOptions.by` selects a lexicographic key sort;
`SortOptions.shuffle = Some(seed)` instead applies a **seeded random
permutation** (`scx sort --shuffle`), and is mutually exclusive with `by`,
`reverse` and `group_by` — the engine rejects the combination. Both modes share
the same emission, alignment and bounded-memory machinery: the order is computed
once in pass 0 and every downstream stage keys off destination row. The seed is
recorded in provenance and is the only record of the permutation. See
[operations.md](operations.md) and
[sharding.md § Shuffling for training](sharding.md#shuffling-for-training-scx-sort---shuffle).

### `scx_ops::modify_metadata(path, patch: &MetadataPatch) → Result<ModifyMetadataSummary>`
Replace metadata sections (`uns` / `obs` / `var` / `obsm` / `varm`) of an existing file **in place, without re-encoding `X`**. Appends only the replaced section bytes at EOF and atomically repoints the catalog — cost is O(replaced sections), the matrix shards are never read or rewritten. The CSC sidecar and `data_generation` are preserved (no `--rebuild-csc`). Any `None` field on `MetadataPatch` is left untouched. **A replaced `obs`/`var` keeps the predicate index it had**: the old section describes values that are gone, so it is rebuilt over the same columns (which re-derives the per-shard column stats, so pushdown survives the edit); `patch.index` names a different set instead. `n_obs` / `n_vars` are invariants — `var` must have `n_vars` rows; `obs` may arrive in either row space: `header.n_obs` rows (written as handed in) or the **live** count (`n_obs` minus the deletion-vector popcount, what `pyscx` `read_obs()` returns since 0.17 — scattered onto the physical axis with `scatter_batch_to_physical`, so a deleted row is `null` in every supplied column and keeps its obs-index barcode from the file). Any other length is rejected before any write (`OpsError::ShapeMismatch`, naming both counts); `obsm` is always physical-length. Replace semantics, not merge. Multimodal (`modality_id != 0`) returns `OpsError::MultimodalUnsupported`. Advisory flock; rollback-able via the catalog chain.

The returned `ModifyMetadataSummary` carries the per-column build outcomes (in the same shape every other rewrite op reports them) plus:

- `obs_columns_not_carried` / `var_columns_not_carried` — the columns the file indexed that the new index does not, so a caller can surface the loss instead of leaving it silent. Render these through `obs_not_carried_unreported()` / `var_not_carried_unreported()`, which drop the columns a build outcome already names: the two channels overlap on the carry path, and a front end printing both raw warns twice about one column.
- `obs_carried_forward` / `var_carried_forward` — per axis, and **success-gated**: true only when that axis was carried *and* the rebuild produced an index. A carry whose every column turned out unindexable reports `false` here and `true` in `obs_predicate_index_dropped`, rather than both at once.
- `obs_predicate_index_dropped` / `var_predicate_index_dropped` — the file had an index on that axis and the output has none.

`set_uns` discards the summary: a `uns`-only patch touches no axis.

### `scx_ops::set_uns(path, uns: &serde_json::Value) → Result<()>`
Convenience wrapper over `modify_metadata` for the headline case — replace the whole `uns` block (O(uns bytes)).

### `scx_ops::update_uns(path, patch: &serde_json::Value) → Result<()>`
The shallow-merge twin: `patch` must be a JSON object, and its top-level keys are laid over the file's existing `uns` (`MetadataPatch { uns, uns_merge: true }` underneath). A patch key replaces a same-named key wholesale; every other key survives verbatim; a file with no `uns` section gets `patch` as-is. The existing blob is read through the op's own file lock (no mmap) and must be a JSON object — anything else is refused before any write. Provenance records `"uns_merge": true` so `scx info` can tell a merge from a replace.

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
- **Level-1 stats are re-derived or dropped, never silently reused, when obs is
  rewritten.** Catalog shard pruning reads a per-shard `MinMax` /
  `CategoryBitset` for each indexed obs column — for the numeric arm without
  consulting the predicate index at all. So the in-place ops that rewrite obs
  values (`modify_metadata(obs=…)`, `obs_import` / `doublet_import`,
  `cellbender_import`) never leave a bound describing values that are gone.
  `modify_metadata` carries the file's index forward and re-derives the stats
  with it, so an indexed file keeps its pushdown; the import ops clear the stats
  for the columns they replace. Where a clear does happen the cost is a slower
  query, never a short result. Rebuild with
  `modify_metadata(..., index_obs=[…])` or a copy-out `sort` / `compact` with
  `--index-obs` / `--index-preset`. Full per-op table in
  [docs/operations.md § Per-shard column stats and the in-place ops](operations.md#per-shard-column-stats-and-the-in-place-ops).
- **A dictionary miss short-circuits only on a complete vocabulary.** A
  `filter_obs("cell_type == 'Typo'")` whose value is absent from an indexed
  column's category dictionary can be answered instantly by pruning every
  shard, instead of scanning obs to find nothing. That inference is global
  ("this value is in *no* shard") drawn from a local artifact (the vocabulary
  this file's index happens to hold), so it is made only when the vocabulary
  is known to cover every shard being pruned — established by the catalog
  itself. Every shard must carry, for that column, a `CategoryBitset` (not just
  any stat under the same hash — `MinMax` answers to it too) whose byte length
  is the one the vocabulary implies, consistently across shards. The index
  build emits exactly that for *every* shard of an indexed categorical column,
  including an all-zero bitset where the column has no values there, and sizes
  each from the same entry list the dictionary comes from. So a missing bitset
  means that shard was never seen by the build, and a wrong-length one means
  the stats and the index section came from different builds — where bit *i*
  no longer means entry *i*. A column appearing **twice within one shard**
  disqualifies it too: nothing says which of the two bitsets the dictionary's
  positions belong to, and counting stat records rather than shards would let
  one shard's surplus stand in for a shard that carries none.
  - In practice the vocabulary is complete on anything written by `convert`,
    `merge`, `compact`, `sort` or `subset` with `--index-*`. It is **not**
    complete after `append` without `--index-obs`, which adds shards the
    file-scope index has never seen. Such a file answers correctly either way
    (an appended shard carries no column stats, so pruning never touches it),
    but a miss on it costs a full obs scan; `--index-obs` on the append, or a
    copy-out `compact --index-obs`, restores the short-circuit.
  - The same bit gates Level-2. There a partial vocabulary is worse than slow:
    `categorical_eq` reports an absent value as an *exact empty row-set*, and
    reports a present one with only the rows that were recorded — so an
    incomplete column is residual (decode + mask) rather than resolvable.
  - An **empty** vocabulary is never complete, however well covered. It cannot
    tell "this column has no values" apart from "this build recorded none of
    them". That distinction is load-bearing on files written before
    integer-valued categoricals were classified: such a file carries an
    entry-less categorical index, every shard gets a zero-length
    `CategoryBitset` for it, and `batch == '1'` would prune every shard and
    return nothing — silently, because the type mismatch that rejects a string
    literal against an integer column lives in the residual evaluator, which
    never runs once Level-1 has short-circuited. Those columns now fall back to
    a scan (and still raise the type error). Re-index with `--index-obs` to
    make the column queryable.
  - `in [...]` and `==` make the identical inference. They used to disagree —
    `==` pruned on a miss unconditionally while `in` refused to — which meant
    one predicate had two answers depending on how it was spelled.
- Fused normalize+log1p in single CSR row scan
- Parallel shard processing via rayon
- **Bounded obs memory on row-sharded files.** `filter_obs` / `count`
  evaluate obs predicates one metadata shard at a time into an `n_obs`
  boolean mask rather than concatenating every obs shard into RAM, and
  skip decoding obs shards that don't overlap a surviving CSR shard (when
  per-shard catalog row-range stats are present — written going forward,
  so existing atlas files still get the memory bound but not the I/O
  skip). Peak obs memory is ~`(rayon width × one shard) + n_obs` bytes,
  not the full obs table. This bound applies **only to the query engine
  path**: other `read_obs()` callers (`compact` / `merge` / streaming
  export / CLI `subset` / `to_anndata`) still assemble the full obs table
  and remain unbounded on atlas-scale sharded files.
- **Filtered-obs categorical semantics.** A `collect()` whose rows the caller
  narrowed — `filter_obs(...)` or `limit(...)` — returns categorical obs
  columns carrying only the categories present in the surviving rows, in
  declared order, not the full parent dictionary — standard AnnData/pandas
  behavior (`remove_unused_categories` on a subset), on both obs layouts and
  whatever the result size, an empty result included. An unfiltered
  `collect()`, like `read_obs()` on the file itself, keeps the full declared
  list. One pre-existing gap: on a file grown by `append` (which still writes
  the rows it adds as plain strings), a filtered `collect()` whose surviving
  rows all fall in appended shards sees no dictionary shard to reconcile
  against and returns that column as plain strings — the values are right, the
  `category` dtype, declared order and `ordered` bit are not; the fix is
  dictionary output from `append` (tracked in the ROADMAP). Downstream code that
  compares `.cat.categories` against the source file (e.g. plotting that
  assumes a fixed palette) should re-derive categories from the result.
- **Null semantics — three-valued (Kleene) logic, like a SQL `WHERE`
  clause.** A comparison against a NULL cell is UNKNOWN, not `false`.
  `and` / `or` combine UNKNOWN accordingly — **`null OR true` is `true`**
  and `null AND false` is `false` — `not` propagates UNKNOWN, and only the
  final mask turns a surviving UNKNOWN into "not matched". In full:

  | | `and` | `or` |
  |---|---|---|
  | `TRUE` ∘ `UNKNOWN` | `UNKNOWN` | **`TRUE`** |
  | `FALSE` ∘ `UNKNOWN` | **`FALSE`** | `UNKNOWN` |
  | `UNKNOWN` ∘ `UNKNOWN` | `UNKNOWN` | `UNKNOWN` |

  with `not UNKNOWN = UNKNOWN`, and a top-level `UNKNOWN` → not matched. So
  `filter_obs("cell_type == 'B cell' or n_genes > 5000")` returns a cell
  with an unannotated `cell_type` and 9000 genes; pandas, polars and SQL
  agree. The same rules apply to `filter_var`, and to every other surface
  driven by this evaluator: `scx delete --filter`, `scx subset --filter`,
  and `to_anndata(obs_filter=...)` on the non-backed default path.
  **Divergence from pandas on `!=` / `not`:** the engine leaves a NULL cell
  UNKNOWN, so a NULL row does *not* match `x != 'v'`; pandas is two-valued
  (`NaN != 'v'` is `True`) and returns it. This matters when the same filter
  string is reused across `backed=True` (pandas) and the default path
  (engine) — see [Filter Expression Compatibility](scanpy.md#filter-expression-compatibility).

### Grouped reads (condition/label-grouped sharding)

On an archive written by `scx sort --group-by` (see
[sharding.md § Condition/label-grouped sharding](sharding.md)),
`QueryPipeline` exposes a targeted-read API over the `group_index` sidecar:

```rust
let pipe = QueryPipeline::open("grouped.scx")?;
pipe.read_group("MYC")?;        // QueryResult — just the MYC cells (Route B slice)
pipe.read_reference()?;          // Option<QueryResult> — the full reference region
pipe.group_labels()?;            // Vec<String>
pipe.iter_group_shards()?;       // Vec<GroupShardHandle> — per non-reference shard
pipe.read_row_range(start, stop)?; // the underlying contiguous-range read
```

- **Raw reads.** These ignore builder state (`filter_obs`/`filter_var`/
  `select_genes`/`with_normalize`/`with_log1p`/`limit`) — they return the
  range's cells only. Deletion vectors **are** honored (rows deleted after the
  sort are dropped), so results match `query().collect()` for the same cells.
- **Errors:** `EngineError::NotGrouped` when the archive has no sidecar;
  `EngineError::UnknownGroupLabel { suggestions }` (with `strsim` close matches)
  for an unknown label; bounds/coverage errors from `read_row_range` on an
  out-of-range or gapped range.

In pyscx these surface on `Experiment` (each opens a fresh pipeline, so builder
bypass cannot happen):

- `Experiment.read_group(label) -> AnnData` — raises `KeyError` (with
  suggestions) on miss, `ValueError` if not grouped.
- `Experiment.read_reference() -> AnnData | None`.
- `Experiment.group_labels() -> list[str]`.
- `Experiment.iter_group_shards() -> list[GroupShard]` — each `GroupShard` has
  `.shard_index`, `.global_start`, `.global_stop`, `.labels`, and `.to_anndata()`
  (deferred per-shard I/O).

The same grouped-read methods are available on `open_cloud(...)` (over range
reads) and via `rscx`. The sidecar is dropped by `append` (and not propagated by
`compact` / `merge` / `subset`); re-sort or re-convert with `--group-by` to
regroup.

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
pipeline.start_epoch()?;
// `next_batch` returns `Result<Option<Batch>>`: `Ok(None)` = clean epoch end,
// `Err(e)` = a mid-epoch I/O/decode fault (propagate it — don't treat as EOF).
while let Some(batch) = pipeline.next_batch()? {
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

#### `spmm_csr_view_with_alg(handle, stream, dev, pool, a, b_view, c_view, alpha, beta, alg) → Result<(), GpuError>`
Sparse × dense matrix multiply: C = α·A·B + β·C. A is GPU-resident CSR; B/C are
`DnMatView` / `DnMatViewMut` over dense column-major f32, so a sub-block of a
larger buffer can be read or written without a copy. `pool` reuses the cuSPARSE
workspace across calls (the PCA power loop passes `Some`).

#### `spmm_csr_transpose_view_with_alg(...) → Result<(), GpuError>`
Transposed SpMM: C = α·A^T·B + β·C, same view-based signature.

> The contiguous-buffer `spmm_csr` / `spmm_csr_transpose` wrappers were removed —
> every production caller uses the strided-view entry points above.

### cuSOLVER Dense Operations

#### `gpu_qr_q(handle, stream, a, m, n) → Result<CudaSlice<f32>, GpuError>`
Economy QR decomposition on GPU: A = Q·R. Returns Q (m × n). Uses `cusolverDnSgeqrf` + `cusolverDnSorgqr`.

### cuRAND

#### `random_gaussian_gpu(stream, rows, cols, seed) → Result<CudaSlice<f32>, GpuError>`
Generate a random Gaussian matrix directly on GPU via cuRAND XORWOW generator.

### GPU PCA Pipeline

#### `gpu_randomized_pca(dev, reader, n_components, n_oversamples, n_power_iterations, zero_center, seed) → Result<GpuPcaResult, GpuError>`
Complete GPU-accelerated randomized PCA. Streams SpMM shard-by-shard via cuSPARSE, QR via cuSOLVER, SVD via CPU faer, final projection via GPU GEMM. Returns `GpuPcaResult { embeddings, components, variance_explained, variance_ratio, mean }`.

> **Row-major `mean_correct_gpu`** — removed. It had no caller after the GPU PCA
> path moved to the col-major operator; centering now happens inside
> `spmm_forward_segment` via `mean_correct_colmajor_strided_kernel`, which both
> the streaming and the device-resident PCA operators call (the streaming one
> once per shard, the resident one once).

### GPU kNN

#### `gpu_knn_cagra_device(dev, embedding, n_neighbors) → Result<DeviceKnnGraph, GpuError>`
Build kNN graph on GPU using NVIDIA CAGRA (cuVS), **device-resident** input and output. Reads a `DeviceEmbedding` (e.g. straight from `gpu_randomized_pca_device`) directly as the CAGRA dataset — no host upload — and returns a `DeviceKnnGraph` on the GPU; call `.to_host(dev)` for the host `GpuKnnResult { indices, distances, n_obs, n_neighbors }`. L2 distance, optimized for PCA embeddings. The fused PCA→kNN pipeline uses this to keep the embedding resident across the handoff. (The host-bounce `gpu_knn_cagra` wrapper and the standalone `scx-accel` `build_knn_graph_gpu` entry point were removed — no production path called them after the rapids-singlecell transition.)

#### `cuvs_available() → bool`
Check if `libcuvs.so` is available at runtime.

> **GPU UMAP** — the native CUDA SGD kernel (`gpu_umap_native`) was removed.
> In-VRAM `device="gpu"` UMAP routes to
> rapids-singlecell (`rsc.tl.umap`); backed/lazy inputs fall through to the cuML
> fallback and then CPU SGD.

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
> For the most up-to-date function signatures, see the
> [auto-generated Python API reference](python_api.rst). For wrapped functions
> (`open`, `from_h5ad`, `obs_import`, …) the canonical docstring is the one on
> the **Python wrapper** — it is what `help()` and the rendered site show, and
> the wrapper's input coercions (`Experiment` paths, pandas Series masks) are
> part of the contract; the Rust `///` docs on those natives are pointers back
> at it. Unwrapped functions (`from_anndata`, `merge`, `compact`, …) render
> from the Rust docstrings via autodoc. A pytest guard
> (`pyscx/tests/test_docstring_coverage.py`) pins every native kwarg to an
> `Args:` entry on the wrapper docstring.

### Module-level functions

- `pyscx.open(path, verify=True) -> Experiment` — Open SCX file (local), returning a lazy `Experiment` handle.
- `pyscx.read(path, *, verify=True, **kwargs) -> AnnData` — One-liner read mirroring `sc.read_h5ad`: shorthand for `pyscx.open(path).to_anndata(**kwargs)`. `**kwargs` forward to [`Experiment.to_anndata`](#experiment) (`backed=`, `var_names=`, `obs_filter=`, `layers=`, `container=`, `data_dtype=`, `index_dtype=`, `allow_lossy=`, …).
- `pyscx.write(adata, path, **kwargs)` — One-liner write mirroring `AnnData.write_h5ad`: shorthand for `pyscx.from_anndata(adata, path, **kwargs)`.
- `pyscx.from_anndata(adata, path, codec=None, shard_size=None, in_place=False, csc=None, csc_cols_per_shard=5000, uns_format="tagged", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", memory_budget=None, force_legacy_metadata=False, sort_by=None, reverse=False, row_group_rows=256, row_group_target_nnz=None)` — Write AnnData to SCX. A float64 `X` is downcast to float32 with a `UserWarning`. `csc=None` resolves to `"off"` unless `index_preset` implies `"auto"` (see `resolve_csc_policy`).
  Persists `X`, `obs`, `var`, `layers`, `obsm`, `varm`, `uns`, and the sparse
  pairwise slots `obsp` / `varp`. Pairwise matrices are stored as float32 COO
  Arrow IPC; higher-precision inputs are downcast on write. `in_place=True`
  permits sorting the caller's CSR indices in place (avoids a copy when `X` is
  an unsorted scipy CSR); leave it `False` (default) to keep the input AnnData
  untouched. `uns_format`
  selects how `adata.uns` is serialized — see [`uns` serialization](#uns-serialization).
  Accepts backed AnnData (`sc.read_h5ad(path, backed='r')`) and auto-routes
  to the streaming converter — see `pyscx.from_h5ad` below for the
  underlying mechanics. `index_*` / `bitmap` materialise query
  predicate indexes and detection bitmaps at conversion time — see
  [Conversion-time predicate indexes and detection bitmaps](#conversion-time-predicate-indexes-and-detection-bitmaps).
  `force_legacy_metadata=True` forces a single `ObsMetadata` /
  `VarMetadata` section regardless of size; the default (`False`)
  emits `ObsMetadataShard` / `VarMetadataShard` sections when
  `n_obs > shard_size`. `memory_budget` (`"4G"`, `"512M"`,
  bytes) emits `MappingPeakFootprintHigh` when an individual mapping's
  estimated footprint exceeds the budget, and — when a CSC sidecar is
  built — is passed to the sidecar transpose (capped at its 4 GiB
  default), so it also changes the CSC shard count; a budget too small for
  one column chunk now **raises** (`RuntimeError: CSC transpose failed:
  memory limit too small…`) where it previously succeeded. `shard_size` sets the per-shard
  row count for both `X` and the obs/var metadata shards. Obsm, varm,
  obsp, and varp are extracted and written one key at a time (incremental,
  not collected).
- `pyscx.from_h5ad(path, out, codec=None, shard_size=None, shard_obs="auto", csc=None, csc_cols_per_shard=5000, uns_format="tagged", stream=True, strict_uns=False, dense_zero_epsilon=0.0, memory_budget=None, temp_dir=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4, sort_by=None, reverse=False, group_by=None, reference=None, group_target_bytes=None, group_max_bytes=None, group_pass="auto", obs_override=None, var_override=None, uns_override=None, row_group_rows=256, row_group_target_nnz=None)` — Stream an h5ad file directly to SCX without materialising `X` in Python or Rust. `csc=None` resolves to `"off"` unless `index_preset` implies `"auto"`. `shard_obs` (`"off"`/`"auto"`/`"always"`, default `"auto"`) writes obs as row-sharded `ObsMetadataShard` sections when `n_obs > shard_size` — the same tri-state and threshold as `scx optimize --shard-obs` and `pyscx.from_anndata`; **obs axis only**, var is always a single section on ingest. It is a storage-layout choice, not a memory one: obs is read whole either way — though each string column's payload is copied **once**, straight from the HDF5 read into the Arrow value buffer at its exact size, rather than through a `Vec<String>` and a `Vec<&str>` first.
  Bounded peak memory: `shard_target_rows × n_vars × density × ~16` bytes
  per X shard, plus `shard_target_rows × k × 4` bytes per `obsm` / `varm` /
  `obsp` / `varp` matrix (each is now hyperslab-read and emitted as
  row-sharded sections — see [§ Sharded obsm/varm/obsp/varp in the
  format spec](format.md#sharded-layout-section-types-2023)), plus the
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
  - `memory_budget`: caps dense slabs, the CSC external-transpose
    buffers, and the CSC **sidecar** transpose chunk (so it changes
    `n_csc_shards` when `csc=` builds one). Accepts an int byte count or
    a binary-prefixed size —
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
    and general) or 10 % (ATAC), times `n_vars × 48 B/nnz`; the
    dense reader sizes its slab at
    `shard_target_rows × n_vars × 44 B/element`. Both figures are the
    **whole worker phase** — the payload or slab, the encoder's own copy
    of the values, and the framed encode's buffers for the two codec
    candidates `codec="auto"` runs concurrently — not the reader stage
    alone. When the dense reader's `memory_budget`-derived slab cap is
    tighter than `shard_target_rows`, the parallel coordinator silently
    clamps its partition to that cap (matching the sequential path).
    The 44 B/element and the quarter-of-the-budget share both come
    from the allocation table described under
    [Memory budgets](#memory-budgets); the dense figure does **not**
    scale with the source dtype width, because the resident slab is
    f32 whatever the input was.
    Export is sized separately at 16 B/nnz, since it decodes and never
    runs the encoder.
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
- `pyscx.from_h5mu(path, out, codec=None, shard_size=None, shard_obs="auto", csc=None, csc_cols_per_shard=5000, stream=True, strict_uns=False, memory_budget=None, temp_dir=None, modalities=None, modality_types=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4, row_group_rows=256)` — Stream an h5mu file to a multimodal SCX v2 file. `csc=None` resolves to `"off"` unless `index_preset` implies `"auto"`. Mirrors `from_h5ad` for h5mu inputs; per-modality `n_vars`/`nnz` come from `/mod/{name}/X` attributes so there is no pre-pass materialisation. `reader_threads`/`writer_queue_depth` carry the same semantics as `from_h5ad` — each modality runs through the same dispatcher independently.
  - `modalities`: optional list of modality names to keep
    (case-sensitive). Unknown names raise `ValueError` with the
    available list.
  - `modality_types`: optional dict `{name: "rna" | "protein" | "atac"
    | "spatial" | "methylation" | "custom"}`. Modalities not listed
    fall back to name inference and emit `ModalityTypeInferred`.
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None, csc=None, csc_cols_per_shard=5000, uns_format="tagged", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", memory_budget=None, force_legacy_metadata=False, row_group_rows=256, row_group_target_nnz=None)` — 10x HDF5 to SCX. `csc=None` resolves to `"off"` unless `index_preset` implies `"auto"`. **Does not stream**: it imports `scanpy.read_10x_h5` and hands the in-memory AnnData to `from_anndata`, so peak memory scales with the whole matrix. The bounded-memory 10x path is `scx convert --from 10x` (streaming by default since OPT-CONVERT-9) — use it for a raw all-droplet `raw_feature_bc_matrix.h5`.
- `pyscx.from_mtx(mtx_dir, scx_path, codec=None, shard_size=None, shard_obs="auto")` — Cell Ranger MTX directory (`matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, `features.tsv[.gz]`) to SCX. Default shard size is 16384.
- `pyscx.to_mtx(scx_path, output_dir)` — SCX to Cell Ranger–style MTX directory (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`).
- `pyscx.cellbender_import(path, cellbender_h5, *, layer="cellbender", obs_key=None, var_key=None, prefix="cellbender_", uns_key="cellbender", overwrite=False, on_missing_rows="zero", on_extra_rows="warn", gene_axis="identical", latent_embedding=False, dry_run=False)` —
  Attach a CellBender `remove-background` output to an existing SCX file as a
  layer, **in place**, joined by barcode. Returns a summary dict; inspect
  `n_matched` (or run with `dry_run=True`) before trusting the result. Since it
  writes `var` columns too, the dict also carries the var predicate index
  outcome — `var_index_rebuilt` / `var_index_dropped` /
  `var_columns_not_carried` — on the same terms as `var_import`. See
  [docs/operations.md § CellBender import](operations.md#cellbender-import).
- `pyscx.is_cellbender_h5(path)` — True when a `.h5` looks like a CellBender
  `remove-background` output rather than a plain 10x CellRanger matrix.
- `pyscx.obs_import(path, table, *, key=None, source_key=None, columns=None, rename=None, prefix="", keep_key_columns=False, delimiter=None, status_column=None, uns_key=None, uns_keys=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  Import a delimited annotation table (CSV/TSV) — or an `.h5ad` whose `/obs`
  holds the columns, on an `hdf5`-feature build — as obs columns on an existing
  file, **in place**. The join is by key string, never by row position; target
  rows the table does not cover get `null`, never a fabricated `0.0`. `path`
  accepts a str, `os.PathLike`, or an open `Experiment`; `key` accepts a str
  (one column), `"obs_names"` (the obs index), or a list (length > 1 builds a
  composite key — the right answer for a multi-library merge where `sample_id` +
  `barcode` is unique but neither is alone). `"obs_names"` resolves on **both** sides, so a table keyed on its own unnamed
  index needs no `source_key`. `source_key` names the **source**
  side's column for each `key` component when the table spells the key
  differently, pairing positionally like pandas `left_on` / `right_on`:
  `key=["sample_id", "obs_names"], source_key=["sample_id", "barcode"]`. Omitted,
  both sides use the `key` names. `on_missing_rows`: `"null"` (default) leaves
  uncovered target rows NULL; `"error"` refuses. `"zero"` is an accepted legacy
  alias for `"null"` — the shared policy's zero is literal only where the
  missing thing is a matrix row (`cellbender_import`, which still spells its
  default `"zero"` and rejects `"null"`), which really is zeros.
  **`overwrite` replaces, it does not merge** — importing several
  per-batch tables in turn keeps only the last. Returns a summary dict
  (`n_obs`, `n_matched`, `n_target_rows_absent`, `n_source_rows_absent`,
  `obs_key_column`, `obs_columns_added`, `obs_index_dropped`, `obs_streamed`,
  and a `key_diagnosis` on failure or `dry_run`); inspect `n_matched`, or run
  with `dry_run=True`, before trusting the result. `obs_streamed` is `False`
  when the target's obs is a legacy single section and had to be assembled
  whole — see [Ops that bound themselves without a budget knob](#ops-that-bound-themselves-without-a-budget-knob).
  Undone by `pyscx.rollback`. See
  [docs/operations.md § External obs import](operations.md#external-obs-import).
- `pyscx.attach_obs_columns(path, df, *, key=None, positional=False, status_column=None, uns=None, uns_key=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  The DataFrame twin of `obs_import`, on the same `attach_external_obs` seam:
  land an in-memory pandas `DataFrame` (or pyarrow `Table`) as obs columns, in
  place, without writing a temp CSV or replacing the whole frame through
  `modify_metadata`. Key-joined by default — `key=None` resolves each side
  independently, exactly as `obs_import` with no `key=` (the source uses `df`'s
  index, named or not, then the barcode-style fallbacks; the target its own obs
  index / fallbacks); a str names one column (matched to the **same name** on
  the target, as rscx's `scx_attach_obs`), a list builds a composite; key
  columns and the pandas index are consumed by the join, not re-imported. `positional=True` (mutually exclusive with `key`)
  skips the join: row `i` annotates obs row `i`, for frames computed
  in-process from this file's own `read_obs()` — never for external tool
  output. It accepts **either row space**, told apart by length: `n_obs` rows
  (`read_obs()`, the live rows — deleted rows are left `null`) or
  `n_obs_physical` rows (`read_obs(logical=False)`, every physical row,
  written as handed in); any other length raises naming both counts. Because
  dispatch is by length, a frame **sorted or reindexed** after `read_obs()`
  would land every value on the wrong cell — so a frame carrying a labelled
  pandas index (every `read_obs()` frame does) is checked: labels that are the
  file's own barcodes in a different order raise naming the first misplaced
  row; labels that are not the file's barcodes are ignored as before, and a
  `RangeIndex` frame is not checked. Under a live-length attach
  `n_matched` is the live count and `n_target_rows_absent` the deleted rows;
  provenance records `row_space` and `positional_index_checked`.
  `status_column` is rejected there. `uns=` lands in the same commit as the columns, so one
  `pyscx.rollback` undoes obs and uns together: alone, it must be a dict and
  its top-level keys are merged into `uns` (several keys per attach); with
  `uns_key="K"` the whole payload nests under `uns["K"]` instead. Untouched
  `uns` keys are left as they were; a colliding key is an error without
  `overwrite=True`, and `uns_key=` without `uns=` is an error. Same policies, summary
  dict and index behaviour as `obs_import` (`obs_key_column` is
  `"<positional>"` under positional; a pure add keeps the predicate index).
  Ungated (no libhdf5). This is `doublet_consensus`'s first-run write path
  (its overwriting re-runs take `modify_metadata` — see below). Categoricals
  survive every in-place obs edit (`attach_obs_columns`, `obs_import`,
  `doublet_import`, `cellbender_import`, `modify_metadata(obs=…)`, rscx
  `scx_attach_obs`): a pandas `category` column — the file's existing ones and
  the one being attached — keeps its dtype, its declared category order, its
  unused levels and its `ordered` bit, exactly as `from_anndata` writes them,
  for string, boolean and numeric levels alike.
  (Before pyscx 0.17 every one of these writers demoted every categorical obs
  column to plain strings.) `append` / `merge` still write the rows they add as
  plain strings. Only an `append` onto an already-sharded dictionary base leaves
  a dictionary/plain mix that a full `read_obs()` reconciles back to `category`
  (with the union vocabulary); a `merge` output, or an `append` onto a legacy
  single-section obs, is plain strings throughout and reads back as `object`
  until the tracked follow-on lands.
- `pyscx.diagnose_obs_key(path, key=None)` — Read-only. Report which obs columns
  could serve as a join key: `n_obs`, `resolved_key`, `resolved_cardinality`,
  `unique_columns`, `unusable_unique_columns`, `unique_pairs` (two-column
  composites that are unique), `pair_search_capped`, `suggestion`, `summary`.
  Worth running before an import onto a merged atlas, where the obvious
  candidates are often not unique and the one that is may be a column no
  fallback list would guess. Every name reported is one `obs_import(key=...)`
  accepts, including `"obs_names"` for the obs index — the physical
  `__index_level_0__` field is never surfaced, because `read_obs()` hands it back
  as the frame's *unnamed index*. `unique_columns` is ordered
  best-candidate-first (obs index, then `barcode`/`cell_id`-style names, then
  other strings, then integers) and holds only columns that can actually key a
  join; a unique column the join would refuse — a float, whose text form is not
  guaranteed to agree across two independently written sides — is listed
  separately under `unusable_unique_columns` rather than offered.
- `pyscx.var_import(path, table, *, key=None, source_key=None, columns=None, rename=None, prefix="", keep_key_columns=False, delimiter=None, status_column=None, uns_key=None, uns_keys=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  The var-axis twin of `obs_import`: land per-**gene** annotations computed
  elsewhere — a normalised symbol from a reference release, an ATAC peak
  annotation, a curated flag — as `var` columns, **in place**, joined by key
  string and never by row position. Genes the table does not cover get `null`,
  never a fabricated `0.0`. `key=None` auto-resolves each side independently:
  the var index, then `gene_id` / `gene_ids` / `id` / `feature_id` /
  `gene_name` / `gene_symbol` / `name`; `"var_names"` names the var index on
  either side, and a list builds a composite. Same `source_key` / `columns` /
  `rename` / `prefix` / `keep_key_columns` / `delimiter` / `status_column` /
  `uns_key` / `uns_keys` / `overwrite` / `on_missing_rows` / `on_extra_rows` /
  `dry_run` semantics as `obs_import`, including **overwrite replaces, never
  merges**. `X`, layers, `obs`, the CSC sidecar, `.raw`, deletion vectors and
  the *obs* predicate index are all untouched; a sharded var keeps its shard
  boundaries and a single-section var stays one section. Returns a summary dict
  (`n_vars`, `n_matched`, `n_target_rows_absent`, `n_source_rows_absent`,
  `var_key_column`, `var_columns_added`, `var_index_rebuilt`,
  `var_index_dropped`, `var_columns_not_carried`, `var_streamed`, the source's `format` /
  `delimiter` / `n_rows_in_source` / `uns_keys_imported`, and a `key_diagnosis`
  on a dry run). A **multimodal** file is refused — each modality owns its own
  var table. Undone by `pyscx.rollback`. See
  [docs/operations.md § External var import](operations.md#external-var-import).
- `pyscx.attach_var_columns(path, df, *, key=None, positional=False, status_column=None, uns=None, uns_key=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  The DataFrame twin of `var_import`, on the same `attach_external_var` seam.
  Key-joined by default; `positional=True` (mutually exclusive with `key`)
  lands row `i` on var row `i` and requires exactly `n_vars` rows — var has no
  deletion vector and therefore no second row space, so unlike
  `attach_obs_columns` there is no length-based dispatch and any other length
  raises. A frame carrying a labelled pandas index (every `read_var()` frame
  does) is checked under `positional`: labels that are the file's own gene
  names in a different order raise, naming the first misplaced row, because a
  frame sorted after `read_var()` would otherwise land every value on the wrong
  gene. Categoricals survive with their declared order, unused levels and
  `ordered` bit. `uns=` lands in the same commit as the columns. Ungated (no
  libhdf5).
- `pyscx.diagnose_var_key(path, key=None)` — The var-axis twin of
  `diagnose_obs_key`, returning the same dict with `n_vars` in place of
  `n_obs`. Worth running on a concatenated or merged file, where `var_names` is
  not always unique.
- `pyscx.doublet_import(path, table, *, tool, key=None, source_key=None, key_added=None, score_column=None, call_column=None, call_true=None, call_false=None, keep_native_columns=True, delimiter=None, uns_keys=None, overwrite=False, on_missing_rows="null", on_extra_rows="warn", dry_run=False)` —
  The doublet-caller wrapper over `obs_import`: each tool names its score and
  call differently, and this maps them onto canonical columns so downstream code
  never branches on which tool ran. For `key_added="K"` (defaulting to the tool
  name) it writes `obs["K_score"]` (`float32`), `obs["K_predicted"]` (pandas
  nullable `boolean`), `obs["K_status"]` (`object`, "present"/"absent"),
  `obs["K_<native>"]` for every other source column, and `uns["K"]`. A tool that
  emits no call (`scds`) gets **no**
  `K_predicted` — thresholding a score is a decision the importer does not make
  for you; pass `call_column=` to opt in. `tool="generic"` requires
  `score_column=`. When a profile *does* declare a call column and the table
  carries none of its spellings, the import **warns** and returns
  `call_column_status="declared_but_absent"` plus `expected_call_columns`
  (also recorded in `uns["<K>"]`), rather than silently producing a score-only
  import — the column is still preserved as `<K>_<native>`, so the fix is
  `call_column=` and not re-running the caller. `call_column_status` is
  `"resolved"` / `"not_declared"` (scds — emits no call by design) /
  `"declared_but_absent"`. Pass `keep_native_columns=False` when the source is an h5ad
  exported from the target file, or every original obs column is re-imported
  under the tool prefix.
- `pyscx.doublet_tools()` — The valid `tool=` values, in table order:
  `scdblfinder`, `scrublet`, `doubletfinder`, `doubletdetection`, `solo`,
  `scds`, `generic`.
- `pyscx.doublet_profiles()` — `{tool: {score_columns, score_prefix,
  call_columns, call_prefix, call_tokens, emits_call}}`, read straight from the
  profile definitions. What each `tool=` actually looks for, so a surprising
  import is a REPL lookup rather than a source read. `emits_call` is the
  load-bearing one: `False` (scds, generic) means `<K>_predicted` can never
  appear without `call_column=`. Rendered as a table in
  [docs/scanpy.md § The per-tool column table](scanpy.md#the-per-tool-column-table),
  which a test pins against this accessor.
- `pyscx.doublet_consensus(target, *, keys=None, method="majority", key_added="doublet", quantile=None, overwrite=False, index_obs=None, index_preset=None)` —
  Combine several callers' imported columns into one consensus. `target` is an
  SCX file (written in place, one commit, `pyscx.rollback`-able) or an in-memory
  `AnnData` (mutated directly). Writes `obs["K_predicted"]` (nullable boolean),
  `obs["K_n_tools_calling"]`, `obs["K_n_tools_voting"]` (both int32),
  `obs["K_score"]` (`mean_rank` only) and `uns["K_consensus"]`, and returns that
  same dict. **Null-aware throughout**: a tool that never saw a cell does not
  vote on it, and a cell nobody voted on comes out `null`, not `False` — read
  `K_n_tools_voting` before trusting a `False`. `method` is `"majority"` (more
  than half the *voting* tools; an even split is `False`, not null), `"any"`,
  `"all"`, or `"mean_rank"` (ignores the calls, rank-normalises each tool's
  scores within the cells it covered and averages; requires an explicit
  `quantile`, since a score cutoff is a scientific decision this helper does not
  own). `keys=None` discovers every key on obs carrying the column the method
  needs, **excluding previous consensus outputs** — a consensus writes the same
  `<K>_predicted` / `<K>_score` columns a caller does, and counting one would
  double-weight whichever callers fed it and inflate `n_tools_voting`. Naming a
  consensus key explicitly in `keys` is still allowed (combining two disjoint
  tool panels is coherent) but warns, and is recorded in the returned
  `keys_that_are_consensus`; `keys_excluded` records what discovery skipped. On
  a file target a **first run** writes through
  `attach_obs_columns(positional=True)` — a pure column add plus a one-key uns
  merge in one commit, so the file's obs predicate index (and every other obs
  column's stats) survives untouched and the rest of `uns` stays
  byte-identical. Passing `index_obs` / `index_preset` — or **overwriting
  existing consensus columns** on a re-run — selects the whole-frame
  `modify_metadata` route instead, the only seam that can rebuild the
  predicate index in the same commit: an index covering a rewritten consensus
  column is rebuilt over the new values, never silently dropped. The `index_*`
  kwargs change the indexed column set; they are not needed to preserve it.
  Pure Python — nothing in it knows what a doublet is.
- `pyscx.export_batches(path, out_dir, *, batch_key, key=None, batches=None, on_ambiguous_key="error", overwrite=False, **kwargs)` —
  Write one h5ad per batch, ready to run a per-sample tool on, without
  materialising the pooled file (peak RSS is one library). `**kwargs` pass
  through to `pyscx.to_h5ad`. What it adds over the loop you would write
  yourself is the key check: a tool sees only the h5ad it is handed, so two
  cells sharing a key inside one batch leave nothing to join its answers back
  on. **Two identities are checked** — the tools read `obs_names`, the import
  joins on the resolved `key`, and those diverge exactly when auto-resolution
  picks a unique non-index column over a duplicated index; both must be unique
  within a batch. `on_ambiguous_key` is `"error"` (refuses before writing
  anything), `"skip"` or `"warn"`. Returns `key`, `key_is_obs_index`,
  `key_is_globally_unique`, `out_dir`, `n_batches`, `n_cells_exported` and a
  per-batch list. Read `key_is_globally_unique` before planning the import: when
  True, concatenate every tool output and import once (which is what you want,
  since `overwrite` replaces rather than merges); when False the keys only
  distinguish cells inside their own batch, so import with a composite key that
  includes `batch_key`.
- `pyscx.to_h5ad(path, out, stream=True, modality=None, reader_threads=None, writer_queue_depth=4, memory_budget=None, obs_mask=None, min_counts=None)` — Stream SCX → h5ad. `obs_mask` is a boolean row mask in either obs row space, told apart by length: `n_obs` entries (the live rows `read_obs()` describes; expanded through the deletion keep mask) or `n_obs_physical` entries (every physical row); any other length raises naming both counts, and an already-deleted row stays dropped either way
  without materialising `X` in memory. Mirror of `pyscx.from_h5ad` in the
  opposite direction. Bounded peak memory: one shard's worth of CSR
  plus encode buffers per matrix written, plus the always-resident
  `indptr` (`(n_obs + 1) × 8` bytes). "Per matrix" covers `X`, every
  `layers` entry, and `adata.raw` — raw goes through the same shard
  walk on its own (usually wider) gene axis. `obsm` / `varm` / `obsp` /
  `varp` are still read whole and are the remaining unbounded term.
  `memory_budget=` is likewise evaluated **per matrix**, so size it for the
  largest one rather than for `X`: raw is captured before HVG subsetting and
  its shards are usually the binding constraint on a `.raw`-bearing file. Raw
  is written last, so a budget that admits `X` but not raw raises only after
  `/X` and the layers are on disk, leaving a partial output. The check runs
  **only on the parallel route** (`reader_threads` > 1) — the budget bounds
  how many shards are in flight, and at one thread there is nothing to
  derate, which is why the refusal offers `--reader-threads 1` as the
  alternative to raising the budget. When deletion vectors are
  present, only kept rows appear in the output (`shape[0] = n_obs -
  n_deleted`); a single pre-scan pass computes the filtered nnz before
  pre-allocating the HDF5 triplet so the on-disk layout is
  deterministic. For multimodal SCX files, pass `modality="rna"` to
  extract a single modality as h5ad; otherwise the call raises (use
  `pyscx.to_h5mu`). `stream=False` falls back to the materialising
  path (kept for parity / debugging).
  - **Categorical obs/var on append-grown files (handled):** a file whose sharded
    obs/var mixes `Dictionary` (original) and plain (appended) representations for a
    categorical column — the layout `append`/`append_from_anndata` produces — exports
    a single h5ad categorical with the full unioned, de-duplicated vocabulary,
    matching `pyscx.open(f).to_anndata()`. This holds for string **and** numeric
    (`Int*`/`Float*`) categoricals, and for both `stream=True` and `stream=False`
    (the fix lives in the shared streaming dataframe writer, which both paths use for
    a sharded axis — there is no separate eager path to fall back to). Category code
    *order* is best-effort and may differ from the read path under deletion vectors;
    per-row values are authoritative.
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
    parser as `from_h5ad`. **Parallel route only** (`reader_threads` > 1);
    at one thread there is nothing to derate, which is why the refusal
    offers `--reader-threads 1`. Evaluated per matrix written (`/X`, each
    layer, `/raw/X`) — size it for the largest, usually raw.
    Derates the granted `reader_threads`
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
- `pyscx.preprocess(source, target, ops, target_sum=None)` — Streaming shard-by-shard preprocessing. Preserves obs/var/obsm/uns and carries the deletion-vector section through (the rewrite is 1:1 in obs row space, so deleted cells stay deleted — see [Operations § Deletion vectors](operations.md#deletion-vectors-carried-vs-applied)); **rejects multimodal and `adata.raw`-bearing inputs** (would otherwise corrupt X / drop raw) — extract a single modality first, or transform before attaching raw.
- `pyscx.save_layer(source, target, layer_name, ops, target_sum=None)` — Save transformed data as a layer. Preserves obs/var/original-X/obsm/uns and the deletion-vector section (pre-existing layers, raw, CSC, varm/obsp/varp, indexes and bitmaps are not carried over); same multimodal/raw rejection as `preprocess`.

### File operations
- `pyscx.append(target, input, codec=None, shard_size=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, modality=None)` — Streaming append from SCX file (reads one shard at a time; raw-copy fast path when codec/encoding match). **Append into a multimodal target is deferred** — it raises `ValueError` (`MultimodalUnsupported`); a single-modality append would leave sibling modalities under-covering the shared obs axis. Extract a modality with `scx subset --modality`, append to that single-modality file, then re-merge. Single-modality append is unaffected. `index_*` kwargs rebuild predicate indexes covering all rows post-append — see [Conversion-time predicate indexes and detection bitmaps](#conversion-time-predicate-indexes-and-detection-bitmaps). Appending categorical `obs`/`var` onto a **row-sharded** base reassembles the existing metadata through the shared canonical assembler (`scx_format_io::assemble_sharded_metadata`), so disjoint/duplicate per-shard categorical vocabularies and a mixed `Dictionary`/plain-string shard layout are reconciled rather than failing to read back. A genuinely corrupt/gapped obs/var shard cover (non-contiguous `row_start` stamps) is now rejected with a clear error instead of being silently mis-assembled.
- `pyscx.append_from_anndata(target, adata, codec=None, shard_size=None, in_place=False, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, modality=None)` — Append from AnnData. Same `index_*` semantics as `append`. **Append into a multimodal target is deferred** and raises `ValueError` (`MultimodalUnsupported`) — extract → append → re-merge. Single-modality append is unaffected.
- `pyscx.mark_deleted(path, cell_indices)` — Logical deletion
- `pyscx.compact(input, output, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, reshape_obs=False)` — Rewrite reclaiming space. `index_*` kwargs rebuild predicate indexes against the compacted output. `reshape_obs=True` migrates legacy single-section obs metadata to the sharded `ObsMetadataShard` layout (mirrors `scx compact --reshape-obs`). A **no-op on already-sharded obs** — which since phase 6c is every `scx convert` / `from_h5ad` / `from_h5mu` / `from_mtx` output above `n_obs > shard_size`, a backed `from_anndata` included. Reach for it on pre-6c files, on `pyscx.from_mudata` output, or after a conversion pinned with `shard_obs="off"` / `force_legacy_metadata=True`.
- `pyscx.optimize(input, output, codec="auto", shard_obs="auto", memory_budget=None)` — Re-encode + canonicalize every CSR shard (X / layers / obsp graphs). **Unframed: this stamps `format_version=3`, not 4.** It is *not* the full equivalent of `scx optimize`, which frames by default — this binding exposes no framing knob and calls the unframed entry point, so it produces no row-group `BlockIndex` and none of the codec-agnostic sub-shard random access (or framed-Scx1 GPU device decode) that a v4 file carries. Use the CLI (`scx optimize --row-group-rows N`) for framed output. Single-modality files only (multimodal → `RuntimeError`; use `compact`). `codec="auto"` (default) keeps the per-shard codec choice; `codec="scx1"` forces Scx1 on every integer shard (full `to_gpu_anndata` device-decode coverage — framed Scx1 shards decode in VRAM); any other value → `ValueError`. `shard_obs` (`"off"|"auto"|"always"`, default `"auto"`) migrates a legacy single-section obs to the sharded `ObsMetadataShard` layout — `"auto"` shards when `n_obs > shard_target_rows` (the `from_anndata` threshold), `"always"` unconditionally, `"off"` keeps the single section; an already-sharded obs is preserved regardless; an invalid value → `ValueError`. No `force` kwarg — pass `output == input` for an in-place upgrade (atomic rename) or remove the target first. Drops the CSC sidecar (rerun `build_csc`). `memory_budget` (a binary-prefixed size string, or `None`) caps what the parallel shard re-encode holds in flight; `None` holds 1 GiB rather than being unbounded, and a small value pins the one-shard-at-a-time behaviour this call had before the re-encode became parallel. See [docs/operations.md § `scx optimize --memory-budget`](operations.md#scx-optimize---memory-budget) and [docs/operations.md § Optimize](operations.md#optimize).
- `pyscx.rollback(path, to_seq=None)` — Revert to previous manifest
- `pyscx.set_uns(path, uns)` — Replace the whole `uns` block in place, **without re-encoding `X`** (cost O(uns bytes)). Replace semantics, not merge (see `update_uns`). The CSC sidecar and `data_generation` are preserved. Rollback-able via `pyscx.rollback`. **`set_uns` is a strict subset of `modify_metadata`** — `pyscx.modify_metadata(path, uns=...)` does the same thing and also reaches `obs`/`var`/`obsm`/`varm`; prefer `modify_metadata` unless you only need the one-arg `uns` convenience.
- `pyscx.update_uns(path, uns)` — **Shallow-merge** a dict's top-level keys into the file's `uns` block in place, same cost and guarantees as `set_uns` (O(uns bytes), `X` untouched, one atomic commit, undone by one `pyscx.rollback`). A key that exists is replaced wholesale (nested dicts are not deep-merged; `None` sets `null`, it does not delete); every other key survives untouched; a file with no `uns` yet gets the dict as-is. A non-dict payload is a `ValueError` (a tuple / array has no top-level keys to merge), and so is a file whose existing `uns` is not a JSON object (replace it with `set_uns`). CLI: `scx set-uns <file> --uns patch.json --merge`.
- `pyscx.modify_metadata(path, *, uns=None, obs=None, var=None, obsm=None, varm=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, modality=None)` — Replace metadata sections (`uns` / `obs` / `var` / `obsm` / `varm`) in place without touching `X`. `obs`/`var` accept a pandas `DataFrame` (or pyarrow `Table`); `var` must have `n_vars` rows and `obs` either `n_obs` rows (what `read_obs()` returns — the live rows; a deleted row keeps its barcode, every index level paired by position so a renamed or multi-level index is preserved too, and is `null` in every other column — **including columns it had before, since this is a replace**; to add a column while keeping deleted rows' values use `attach_obs_columns(positional=True)`) or `n_obs_physical` rows (what `read_obs(logical=False)` returns, written as handed in); any other length → `ValueError` naming both counts. A live-length frame holding the file's own live barcodes in a different order is refused by row (a sort / reindex after `read_obs()`; renamed barcodes pass). Provenance records `row_space`. `obsm`/`varm` accept `dict[str, np.ndarray]` and are always physical-length (a dense mapping has no `null` for a deleted row). Any omitted arg is left untouched. **A replaced `obs`/`var` keeps the predicate index it had** — rebuilt over the same columns, which also re-derives the per-shard column stats, so `filter_obs` pushdown survives an ordinary obs edit. `index_*` kwargs name a different column set instead, **per axis** (`index_obs` does not change what a same-call var replacement carries; `index_preset` / `index_auto_threshold` span both); a column the file indexed and the new set omits raises a `UserWarning` rather than vanishing silently. Replace semantics, not merge; for a shallow `uns` merge use `pyscx.update_uns`. Only the global modality is supported today (`modality != 0` → error). For a **pure obs column add**, prefer `pyscx.attach_obs_columns` (or `obs_import`): it knows which columns it writes, so the predicate index and untouched columns' stats survive without the rebuild.
- `pyscx.sort(input, output, by, reverse=False, shard_size=None, codec="auto", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, memory_budget=None, temp_dir=None, bitmap="off", rebuild_csc=False, csc_cols_per_shard=5000, csc_memory_limit="4G", group_by=None, reference=None, group_target_bytes=None, group_max_bytes=None, group_write_block_bytes=None)` — Globally reorder cells by an obs key for X-read locality and contiguous predicate-index shard ranges. `codec` (`auto`/`none`/`scx1`/`zstd`/`lz4`/`pcodec`/`shufdelta`) pins the output encoding — without it the writer re-selects per shard, so a reorder can change file size for reasons unrelated to the reorder. Pass `memory_budget` (e.g. `"4G"`) to force the bounded external partition sort. Drops the CSC sidecar and detection bitmap (`rebuild_csc=True` / `bitmap=` to re-emit); `adata.raw` is not preserved; deletions are materialized away. See [sharding.md § Sorting for read locality](sharding.md#sorting-for-read-locality-scx-sort).
- `pyscx.shuffle(input, output, seed=42, shard_size=None, codec="auto", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, memory_budget=None, temp_dir=None, bitmap="off", rebuild_csc=False, csc_cols_per_shard=5000, csc_memory_limit="4G")` — Globally reorder cells by a **seeded random permutation** (`scx sort --shuffle`), so a training loader gets i.i.d. batches at any `shard_group_size`. Same engine and same drop semantics as `sort`, minus the key arguments — shuffle is an order *source*, not a modifier, so `by` / `reverse` / `group_by` are not offered. `seed` is recorded in provenance and is the only record of the permutation; the same seed on the same input always reproduces the same file. The permutation runs over **live** rows, so a file with deletion vectors shuffles differently from the same file without them. Two consequences worth knowing: the output is the inverse of a sorted file for queries (it maximally scatters predicate-index shard ranges), and a permutation inherently costs some cross-row redundancy for codecs whose compression spans rows — ~6–12% for `zstd`, under 1% for `lz4`/`shufdelta`. **Leave `codec="auto"`**: it runs the same adaptive per-shard selection `scx convert` does. (Earlier releases told you to pin the input's own codec because `auto` grew X 1.86–2.09×. That was a derived-file bug, not a property of shuffling, and is fixed — pinning now selects a specific encoding, it does not hold size. See [sharding.md § Shuffling for training](sharding.md#shuffling-for-training-scx-sort---shuffle).) The provenance entry's `action` stays `"sort"` — the seed lives at `params.shuffle.seed`, so a consumer looking for `action == "shuffle"` will not find it. See [sharding.md § Shuffling for training](sharding.md#shuffling-for-training-scx-sort---shuffle).
- `pyscx.merge(inputs, output, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, assume_identical_var=False, assume_identical_obs=False, uns_policy=None, sort_by=None, reverse=False)` — Merge multiple files. `index_*` kwargs rebuild predicate indexes against the merged output — without them, pushdown silently regresses to a full obs scan on the merged file. `assume_identical_var` (default `False`) validates var identity (index, column names, values) across all inputs; set `True` to check only `n_vars` (breaking change from pre-branch where var was unchecked). `assume_identical_obs` (default `False`) validates each input's obs schema (column names + dtypes) against input 0; set `True` to skip. `uns_policy` controls conflicting uns sections: `None` / `"first"` (keep first input), `"require-equal"` (error on difference), `"namespace"` (prefix keys with input filename), `"summary"` (keep first input and record a `_scx_uns_conflicts` array). `sort_by` / `reverse` optionally sort the merged obs by a column (a sorted merge refuses `obsm` and COO `obsp` — merge without `sort_by`, then `scx sort`). **An `obsm` / `obsp` key that some inputs carry and others lack is a hard error** (`RuntimeError`) naming the axis, the key and the input — the same answer a missing layer has always had; it used to be a silent drop. An `obsp` key whose `data` column disagrees in dtype or nullability across inputs is refused too, since the merged shards are read back under one schema. File-scope COO `obsp` is carried and rebased into the merged obs space, and `varp` comes from input 0; the **CSR-backed** `obsp` encoding and modality-scoped pairwise graphs are dropped with a warning. Merge streams obs shard-by-shard and builds predicate indexes incrementally from the shard stream.

### Cloud operations (requires `--features cloud`)
- `pyscx.pull(source, dest, filter=None, parallelism=None)` — Streaming cloud → local
- `pyscx.push(source, dest, parallelism=None)` — Streaming local → cloud
- `pyscx.cloud_optimize(input, output=None)` — Front-of-file catalog
- `pyscx.explode(input, output)` — Packed → exploded directory
- `pyscx.pack(input, output)` — Exploded → packed
- `pyscx.open_cloud(url) -> CloudExperiment` — Direct cloud reads.
  `CloudExperiment.query(modality=…)` scopes to one modality of a multimodal
  file (same semantics as the local `Experiment.query`).
- `pyscx.read_cloud(url, *, obs_filter=None, var_names=None, modality=None, data_dtype=None, index_dtype=None, allow_lossy=False) -> AnnData` — One-call cloud read (= `open_cloud(url).query(modality=…)…collect().to_anndata()`); see [docs/cloud.md § `pyscx.read_cloud(...)`](cloud.md#pyscxread_cloud--one-liner-cloud-read).

### Experiment

The Python-visible class is `Experiment` (the Rust type is `PyExperiment`).
`repr(exp)` is AnnData-style — a `Experiment object with n_obs × n_vars = …`
header followed by indented `obs:` / `var:` / `uns:` / `obsm:` / `varm:` /
`layers:` key lists. On-disk codec / shard / format-version internals moved off
the repr onto `Experiment.info() -> str`, whose tokens now include
`value_encoding=`, `is_integer=` and `max_value=` (rendering them reads one
76-byte header per CSR shard, so `info()` is O(shards), not O(1)).

- `value_encoding` `→ str` / `is_integer` `→ bool` / `max_value` `→ int` — What the file stores, without decoding it. `value_encoding` is the on-disk value encoding of the CSR shards as `scx info` prints it: a numpy dtype name (`"uint16"`, `"float32"`, …) when every shard agrees, `"mixed (uint8, uint16)"` when they differ (append keeps each shard's own encoding), `"n/a"` with no shards; on a multimodal file it folds every modality's X shards, and a layer's encoding is `adata.layers[name].stored_dtype` on a backed handle. `is_integer` is `True` when every shard is integer-encoded (`uint8` / `uint16` / `uint32`), i.e. the values are counts — `False` for any float shard or a shard-less file. `max_value` is the largest stored value from the per-shard catalog stats: float-encoded shards record no value range and a shard without stats contributes nothing, so it is `0` for a float file — read it together with `is_integer`; it is physical, like `nnz` (values in logically deleted rows still count). Each costs one shard-header read per CSR shard on first access (`max_value` is catalog-only); the fold is memoised per `Experiment` — shared by `value_encoding`, `is_integer` and `info()`, reset by `reload()` / `close()`, and a stale handle refuses before it can answer from the memo — so later reads are free. None decodes a value.

- `read_obs(columns=None, *, logical=True)` / `read_var(columns=None, *, modality=None)` — Read the cell / gene metadata table as a pandas DataFrame **without touching `X`**. Reach for these instead of `to_anndata().obs` / `.var` when you only want the metadata: on a 500k × 61k Census file `read_var()` is ~0.05 s against ~1.4 s and ~10 GB peak for the full materialisation. Both always retain the pandas index column (barcodes / gene names), so a projected frame indexes the same as an unprojected one, and an unknown column name raises `KeyError` listing what is available. Boolean columns come back as the pandas nullable `boolean` dtype whether or not they contain nulls, so the dtype follows the schema rather than the data and agrees with what a `to_h5ad` round trip returns. On a column with nulls that means `.astype(bool)` raises — deliberately, since coercing "not covered" to `False` is the mistake nullability exists to prevent — and `.fillna(False)` is the explicit form.
  - `read_obs`'s `columns` is a genuine **pushdown** — unselected columns are never materialised, which matters because `obs` scales with `n_obs`.
  - **Row space.** `read_obs()` returns the **live** rows: deletion vectors applied, so `len(read_obs()) == n_obs == len(to_anndata(backed=True).obs)`, row for row and index for index — the same frame `query().collect()`, `gather_rows_sparse` and rscx's `$obs()` describe. `read_obs(logical=False)` is the **physical** table (`n_obs_physical` rows, deleted rows in place). `to_h5ad(obs_mask=)` and `Experiment.mark_deleted(mask)` accept a mask in either row space (a live-length mask is expanded through the keep mask; a pandas Series with a labelled index is checked for order), so a mask derived from `read_obs()` works. Identical on a file with no deletions (and the logical read costs no copy); `has_deletions` says whether they differ. `obs_categorical(col, *, logical=True)` / `obs_categorical_many(cols, *, logical=True)` follow the same rule (one code per live row by default; `categories` never loses a level to the filter). **Changed in 0.17**: `read_obs()` and `obs_categorical()` used to return the physical table on every file, so on a `mark_deleted` file `read_obs()` was longer than `to_anndata(backed=True).obs` with no warning — a pre-1.0 clean break, no `FutureWarning` cycle; pass `logical=False` for the old frame. A frame computed from `read_obs()` can still be landed positionally: `attach_obs_columns(positional=True)` and `modify_metadata(obs=)` accept either row space (a live-length frame leaves the deleted rows `null`).
  - `read_var`'s `columns` is a convenience projection applied **after** the decode. `var` is one section sized by `n_vars` (5.5 MB for 61k genes), so there is nothing to save at the I/O layer; it does not read less off disk.
  - `read_var(modality=…)` selects one modality's gene axis on a multimodal file. Omitting it reads the global / single-modality `var`, which on a multimodal file is usually not what you want. Unknown name → `KeyError`.
  - For enumerating one categorical obs column, prefer `distinct_values()` / `obs_categorical()`, which scan per-shard dictionaries instead of assembling the table.
  - `CloudExperiment` mirrors both, `logical=` included (its `n_obs` / `shape` / `repr` are the live count too since 0.17 — one small section read on a file with deletions, memoised per handle; `n_obs_physical` stays the header count). There, `read_obs(columns=…)` *is* a genuine network pushdown (per-column projected range reads); `read_var(columns=…)` is still post-fetch, for the same reason as locally.
- `to_anndata(backed=False, cache_shards=4, var_names=None, obs_filter=None, layers=None, preserve_slots=False, modality=None, eager=False, memory_budget=None, obsm=None, preserve_var_order=False, strict_var_names=True, container="csr", data_dtype=None, index_dtype=None, allow_lossy=False, obsp=None, varp=None, varm=None, raw=True)` — Convert to AnnData
  - `var_names`: list of gene names to project (column subset). A set selector by default (sorted original-column order, duplicates collapsed). `X` and each selected layer are assembled **already projected**, shard by shard, so a few genes out of tens of thousands cost roughly the projected result rather than the whole matrix twice (measured on a 40-shard 20 480 × 2 000 fixture carrying one layer: 74.0 MB → 8.9 MB peak; `container="dense"` 48.8 MB → 8.8 MB). The bound is on the **var axis only** — `obs`, `obsm` and `obsp` are on the other axis, so they are in the result at full size and anndata's copy transiently doubles them. On a file carrying an `n_obs × n_obs` kNN graph that term dominates whatever `var_names` says; pair it with the slot filters (`to_anndata(var_names=[...], obsp=[], varp=[], varm=[])`) for the tight bound. `.raw` is not projected either — anndata never var-slices it, so it stays on its own (usually wider) gene axis; pass `raw=False` if you do not want to pay for it
  - `preserve_var_order`: when True, return the gene axis in the order `var_names` was listed (first-occurrence-wins dedup) instead of sorted order. Works on eager / backed / GPU / query-engine paths. The streaming accelerators decode columns in sorted on-disk order and so cannot express a request-ordered gene axis; on a *backed* `X` they refuse rather than misalign the result against `adata.var` — `highly_variable_genes`, `normalize_total`, `log1p`, `calculate_qc_metrics`, `score_genes`, `pflog`, `pca`, `pca_neighbors`, `pca_neighbors_umap`, `rank_genes_groups`, `rank_genes_groups_df` (its compute mode; the `adata.uns[key]` extraction mode reads no matrix and is unaffected), `pdex_ref`, `pseudobulk_means`, `pseudobulk_dex` and `pdex_nb_glm` raise `RuntimeError`. The same refusal applies to a backed `X` reordered at the handle level — `adata[:, idx]` with a non-ascending `idx`, or `adata.X = adata.X[:, [7, 2, 11]]` / `X[:, ::-1]` — which installs the same presentation permutation. The refusal covers the matrix the op actually reads, so a named `layer=` and a layer handle assigned to `X` are rejected too — materialising `adata.X` does not help when the permutation is on the layer. Run them before reordering, select with a sorted index or a boolean mask, or materialise the matrix being read (`adata.X = adata.X.to_memory()`, or `adata.layers[name] = adata.layers[name].to_memory()`) — and only that matrix: materialising a named layer is sufficient for `layer=`, even while `adata.X` stays presentation-ordered, because each op is guarded on the matrix it selects rather than on `X`
  - `strict_var_names`: when True (default), any name absent from the var metadata raises `KeyError`. Pass False to silently drop unknown names (pre-0.8.6 behaviour)
  - `obs_filter`: predicate string for cell filtering. Non-backed mode uses the scx-engine query parser with shard pushdown; `backed=True` evaluates it with pandas `.query()` (different grammar — see [Filter Expression Compatibility](scanpy.md#filter-expression-compatibility) in docs/scanpy.md)
  - `layers`: list of layer names to load (default `None` = all; `[]` loads none). **Changed in 0.18:** an unknown layer name raises `KeyError` naming the available layers. It previously loaded nothing and said so only indirectly, via the obs-filtered query path's "does not load" warning — which the slot filters made unreliable, since an excluded slot and a misspelled key both select nothing. All five slot filters (`layers`, `obsm`, `obsp`, `varp`, `varm`) now fail the same way, on every path.
  - `obsm`: list of obsm keys to load (default `None` = all keys, byte-identical to prior behaviour). When set, only the listed embeddings are read — dropping the per-process RAM of unused keys on the random-access dataloader path. An unknown key raises `KeyError`; `obsm=[]` loads no embeddings. Selecting keys also changes *how* obsm is materialised (see `obsm` loading modes under `eager`, below).
  - `obsp` / `varp` / `varm`: lists of keys to load from those slots (default `None` = all keys, byte-identical to prior behaviour). Same contract as `obsm=`: `[]` loads none, a list loads that subset, an unknown key raises `KeyError`. Unlike `obsm=`, an empty or fully-excluding list builds **no lazy bridge at all** for that slot — `adata.obsp` is then anndata's own empty mapping, so there is nothing left that could decode. This is the knob that bounds the cost described under the lazy-mapping note in [docs/scanpy.md](scanpy.md#understanding-to_anndata): anndata's `AlignedMappingProperty` builds an `AlignedActual` on the first `adata.obsp` access, and that validates — and therefore decodes — **every** key of the slot, so a census-scale kNN graph comes off disk whether or not the caller wanted it. Honoured on the eager, backed and `preserve_slots` paths, and on `to_gpu_anndata`; on the obs-filtered query path these slots are dropped anyway, so excluding one simply removes it from the "not loaded" warning. Key validation is at the entry point, so an unknown key raises on every path including that one.
  - `raw`: when `True` (default), `adata.raw` is reconstructed on the paths that can (see [`adata.raw`](#adataraw)) and the `DroppedRaw` notice is emitted on the three that cannot — backed mode, an obs-filtered query, and a deletion-vector-active file. When `False`, neither happens: no rebuild, no notice. The explicit opt-out for callers who never wanted raw and do not want the warning on every open.
  - `layers=` and `var_names=` force eager assembly of `obsp` / `varp` / `varm` regardless of `eager=False`: those are sliced by anndata, whose `AlignedMapping` validation would drag every bridge through the slice anyway, so fragmenting the cost across implicit slicing helps nobody. Combine them with the slot filters above to keep that assembly small. Under `var_names=` the **matrices** are the exception — `X` and each selected layer are projected while assembling and never exist at full width.
  - `backed`: when True, X and layers are lazy `ScxBackedSparseDataset` instances
  - `modality`: select one modality of a multimodal file and
    return a backed AnnData scoped to that modality (per-modality X /
    var / obsm; global obs shared). Currently requires `backed=True`.
    Every selection kwarg is **rejected**, not ignored — `var_names`,
    `obs_filter`, `layers`, `obsm`, `obsp`, `varp`, `varm` and
    `raw=False` all raise `ValueError`, because the per-modality builder
    accepts none of them (use `scx subset --modality NAME --filter` to
    pre-materialise a filtered single-modality file).
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
    assembly. Under `var_names=` the estimate is scaled by the fraction
    of genes selected, so taking the warning's own advice actually
    silences it; the catalog carries no per-column `nnz`, so that
    scaling assumes an even spread of nonzeros across genes.
    The estimate covers **assembly only** — `X`, `adata.raw` when the read
    includes it, and obs/var — and is a floor on process RSS rather than a
    figure a job can be sized from: whatever reads the matrix afterwards
    (scanpy's per-group copies, a write-back) is not in it and cannot be.
    It prices the plan the read will actually use: `container="dense"` is
    `n_obs × n_vars × value_width` with no index array, and the value width
    comes from `data_dtype` (a `float64` read holds 8 B per value). For a CSR
    read it counts that value width plus the column-index width the assembled
    CSR will hold — **4 B below `i32::MAX` nonzeros and 8 B above it**, which
    is scipy's choice and not the caller's, so `index_dtype=` does not enter
    into it. `to_gpu_anndata`'s device path assembles no host `X` and so is not
    charged for one; its host-assembling fallback is. It is a floor on the objects an ordinary read
    leaves resident, not a bound on its transients — the dense reader builds a
    CSR and scatters from it, and the assembler's bounded in-flight shard
    decodes are not in it either. On a file with **deletion vectors** it is
    neither: the counts are physical, so it overstates the compacted matrix the
    read returns and understates the peak, where both buffers are live at once. It was a flat 16 B —
    right for a wide matrix before the widened decode below landed, and 2×
    conservative for every matrix under the line, which never paid an upcast.
    So a file between roughly 0.5 and 1 billion nonzeros no longer trips the
    default 8 GiB budget. Eagerly-materialized `layers` are still not counted
    (they assemble `f32` and cast afterwards).
  - **Container / dtype materialization** (`container`, `data_dtype`,
    `index_dtype`, `allow_lossy`) — control the output container and numeric
    dtype of `X` (and layers). Eager (`backed=False`) only; a non-default
    request with `backed=True` raises (the backed dataset is lazy and
    f32-native). See [Container and dtype materialization](#container-and-dtype-materialization) below for the full reference.
    - `container` (default `"csr"`): `"csr"` → scipy `csr_matrix` (unchanged);
      `"dense"` → row-major `numpy.ndarray` (no scipy CSR).
    - `data_dtype` (default `None` → `float32`, today's behaviour): one of
      `float16` / `float32` / `float64` / `int8` / `int16` / `int32` /
      `int64` / `uint8` / `uint16` / `uint32`.
    - `index_dtype` (default `None` → `int32`): CSR column-index dtype
      (`int16` / `int32` / `int64`). Ignored (with a `RuntimeWarning`) for
      `container="dense"`.
    - `allow_lossy` (default `False`): the fail-loud cast gate. When `False`,
      any narrowing that would lose data (out-of-range, fractional-into-int,
      negative-into-unsigned, or a count above 2²⁴ into `float16`) raises
      `ValueError`; `True` performs the narrowing anyway.
    - The default (`container="csr"`, no dtype kwargs) is **byte-identical and
      zero-copy** — the Vec is moved into numpy with no cast. A non-default eager
      request narrows **in-decode**: `X` (and `adata.raw`) assemble directly at the
      target width, never building the full-matrix f32 CSR, so a narrow
      `data_dtype=` lowers peak RSS. `to_anndata(obs_filter=…, data_dtype=…)`
      decodes at the requested dtype too — that route resolves the plan before it
      collects. (Eager `layers` still cast post-assembly.)
  - Returns `obsm` (dense), `varm` (dense), `obsp` (scipy CSR), and
    `varp` (scipy CSR) when present in the file. When cells are
    logically deleted, `obsp` is subset to the kept rows and columns
    when the entry is decoded (via `filter_coo_obsp_by_kept_rows`), so
    the in-memory AnnData stays shape-consistent. `varp` and `varm`
    are unaffected (var axis has no deletion vector). The on-disk
    section retains the original axis until `compact` rebuilds the file.
- `to_mudata(backed=False, cache_shards=4, container=None, data_dtype=None, index_dtype=None, allow_lossy=False)` — Materialise a multimodal file as `mudata.MuData`
  - Eager (`backed=False`): per-modality scipy CSR AnnData sharing the
    global obs. Raises on single-modality files.
  - ⚠️ **Deletion vectors are not applied** — unlike `to_anndata()`, `query()`
    and every other materialising read, `to_mudata()` returns *physical* rows,
    so cells marked by `mark_deleted` are present. This is deliberate: every
    modality's X has to stay in lockstep with the single shared global obs, and
    the per-modality typed reader is unfiltered for that reason
    (`scx_format_io::read_all_csr_shards_for_typed`). Run `scx compact` first if
    you need the deletions materialised. `Experiment.has_deletions` tells you
    whether a given file is affected (and `n_obs` vs `n_obs_physical` by how
    much). See also
    [docs/multimodal.md](multimodal.md) and
    [Operations § Deletion vectors](operations.md#deletion-vectors-carried-vs-applied).
  - **Per-modality in-decode narrow.** `container` / `data_dtype` /
    `index_dtype` each accept **either a scalar** (applied to every modality)
    **or a dict keyed by modality name** — e.g.
    `data_dtype={"rna": "uint16", "atac": "uint8"}`. A modality with no
    override keeps the **byte-identical zero-copy `f32` CSR** path; a
    narrowed modality assembles directly at the target width via the typed
    per-modality reader (`scx_format_io::read_all_csr_shards_for_typed`),
    never building the intermediate f32 CSR. This matters most for
    multimodal reads: modalities differ in range (shallow/binarized ATAC and
    small ADT fit `uint8` = 4× on the value buffer; RNA/deep counts fit
    `uint16` = 2×) and `to_mudata` materialises *N* matrices at once. The
    same fail-loud cast gate applies **per modality** (a lossy narrow raises
    unless `allow_lossy=True`); integer→integer narrows are exact, including
    `> 2²⁴`. A dict key naming no modality in the file raises `ValueError`.
    `index_dtype` is accepted for symmetry but is a **no-op for the returned
    CSR** on any modality that fits in int32 — scipy resolves the width from
    the contents and canonicalizes in both directions (same as `to_anndata`,
    below).
    `container="dense"` is **not yet supported** for `to_mudata` (CSR only).
    The narrow kwargs require `backed=False`.
  - **Backed (`backed=True`)**: per-modality
    `ScxBackedSparseDataset` AnnData sharing the global obs (lazily
    f32-native — the narrow kwargs are rejected). Single-modality files are
    wrapped in a one-modality MuData rather than raising, so the call works
    uniformly across layouts.
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
- `gather_rows_sparse(rows, modality=None, cache_shards=4, layer=None, logical=True) -> scipy.sparse.csr_matrix`
  — The bounded shard-wise row gather, without building an AnnData. `rows` is a
  boolean mask or any 1-D integer array-like (list, `range`, ndarray of any
  integer dtype); duplicates allowed, order preserved, negative indices wrap
  once. Each touched shard is decoded once (a sparse request on a
  row-group-framed shard decodes only the touched row groups) and the result is
  assembled once into exact-size buffers, so peak memory is the result plus the
  shard cache, plus up to `cache_shards` shards decoding in flight while that
  cache fills (at most `2 × cache_shards` decoded shards beside the result) —
  never a second copy of the result.
  `logical=True` (default) indexes the rows `n_obs` / `read_obs()` describe
  (deletion vectors applied, exactly as `to_anndata(backed=True).X[rows]`);
  `logical=False` indexes physical file rows (`n_obs_physical`). **Changed in
  0.17**: before, this method addressed physical rows only, so on a file with
  deletion vectors the same ids now select different cells — pass
  `logical=False` for the old behaviour. `layer=`
  gathers from that layer instead of `X` (`ValueError` if absent; not supported
  on a multimodal file — index the modality's layer handle from `to_mudata()`).
  Out-of-range ids and a boolean mask whose length is not the row count raise
  `IndexError`. Gene indices are raw-local (no global-vocab remap). A fresh
  reader is opened per call (fork-safe; `cache_shards` bounds its memory, it is
  not a speedup knob). Equivalent to `adata.X[rows]` / `adata.layers[name][rows]`
  on a backed handle.
- `to_gpu_anndata(var_names=None, obs_filter=None, layers=None, obsm=None, device="gpu", memory_budget=None, preserve_var_order=False, strict_var_names=True, container="csr", data_dtype=None, index_dtype=None, allow_lossy=False, obsp=None, varp=None, varm=None, raw=True)` — Minimal-copy on-device handoff: decodes shards, transfers to the GPU, and returns a GPU-resident AnnData whose `X` is a `cupyx.scipy.sparse.csr_matrix`. The returned object is suitable for direct use with rapids-singlecell (`rsc.pp.*`, `rsc.tl.*`) without additional host↔device copies. Requires `cupy` and a CUDA-capable GPU. Records its `transfer_mode` and real `bytes_uploaded` on `uns["scx_accel"]["to_gpu_anndata"]` — `scx_device_decode_gpu` (Scx1 sidecar shards decoded fully in VRAM, including dense ≥128-nnz rows via the BitPacker4x kernel; only indptr uploaded), `scx_device_handoff_streamed` (some shard host-bounced because it is not an Scx1 sidecar shard — a non-Scx1 codec or a sidecar-less Scx1 shard), or `scx_device_handoff` (host-assembled filtered/projected/multimodal input). If the in-VRAM decode *fails* on a request that qualified for the fast path, the call does not raise: it falls through to the host-assemble path, which reaches the same `cupyx` `X` by a different road, and records `transfer_mode="scx_device_handoff"` with `fallback_reason="gpu_runtime_error"` plus a `UserWarning` naming the device error. An out-of-memory failure is excluded — both paths end with the same CSR resident on the device, so host-assemble cannot fix a VRAM shortfall and the `>VRAM` error is raised directly instead. See **Accelerator route metadata** below.

  **Memory semantics.** The result is the **complete** sparse matrix in VRAM — this is not a streaming/partial representation. Sparse CSR format is preserved throughout (VRAM scales with NNZ, not N×M). Shards are decoded one at a time into pre-allocated combined device buffers; peak device memory during transfer is the combined buffer plus one shard. A **VRAM pre-flight check** (1.2× headroom factor) compares the required bytes against free device memory and raises `ValueError` if insufficient, with an actionable message pointing to `backed=True` streaming workflows. See [gpu-setup.md § GPU memory model](gpu-setup.md#gpu-memory-model) for sizing formulas.

  `to_gpu_anndata` accepts `container` / `data_dtype` / `index_dtype` / `allow_lossy` for signature parity with `to_anndata`, but the device path is **f32-native**: a non-default request raises `ValueError`. To obtain a narrow/dense matrix, materialize it on the host with `to_anndata(container=..., data_dtype=...)`.
- `info() -> str` — One-line codec / shard / format-version internals (kept off the AnnData-style `repr`).
- `reload()` — Re-open the file, picking up anything written since the handle was opened. See **Handles and files that change underneath them** below.
- `close()` — Release the file mapping. Idempotent; reads afterwards raise. Also available as a context manager (`with pyscx.open(p) as exp:`).
- Properties: `n_obs`, `n_vars`, `nnz`, `shard_count`, `format_version`, `codec_id`, `index_dtype`, `path`, `has_csc`, `has_deletions`, `closed`.
- List-returning accessors — callable **methods** (not properties): `layer_names()`, and the AnnData-style key accessors `obs_keys()`, `var_keys()`, `obsm_keys()`, `obsp_keys()`, `varm_keys()`, `uns_keys()` (all cheap — schema/catalog reads, no matrix decode; `obs_keys()`/`var_keys()` exclude the pandas index column; `obsp_keys()` lists only the COO forms every conversion path writes, since the CSR-backed `ObspCsrShard` has no read API).
- `read_obsp_rows(key, start, stop, *, logical=True)` — rows `[start, stop)` of an `obsp` graph as a scipy CSR, decoding only the shards the range covers. The bounded counterpart to `to_anndata().obsp[key]`, which materialises the whole matrix. `logical=True` (the default, matching `read_obs`) takes the bounds in **live** row space, drops any edge whose either endpoint is deleted and renumbers both axes into live space; `logical=False` is the physical graph, unfiltered, with column extent `n_obs_physical`. ⚠️ A graph stored as one unsharded section is decoded whole whatever range is asked for. Every path that *writes* an `obsp` emits shards — `scx sort` / `scx compact` since phase 9, the h5ad and `from_anndata` paths always, `scx merge` per input section — so an unsharded graph means a file written before phase 9, or one whose graph came from the low-level `write_obsp` and was then carried by `scx optimize` (which copies a mapping section verbatim). `scx subset` drops the families outright. The obs-axis *embeddings* are a separate question with more unsharded producers: see [sharding.md § Obsm / varm / obsp / varp sharding](sharding.md#obsm--varm--obsp--varp-sharding).

#### Handles and files that change underneath them

An `Experiment` maps the file it was opened from, and no SCX write path ever
edits bytes a reader is looking at — an in-place op (`obs_import`,
`doublet_import`, `modify_metadata`, `set_uns`, `update_uns`, `mark_deleted`, `build_csc`,
`rollback`) appends and rewrites the header, and a copy-out op (`compact`,
`sort`, `merge`, `subset`) renames a new file into place. Either way the
mapping stays intact and readable while describing a file state that is no
longer on disk.

Reading through such a handle **raises** rather than answering:

```python
exp = pyscx.open("atlas.scx")
pyscx.obs_import("atlas.scx", "calls.csv", key="obs_names")

exp.read_obs()
# RuntimeError: 'atlas.scx' changed on disk since it was opened
# (manifest_sequence 1 → 2). This handle still maps the file as it was
# when it was opened … re-open the file to read the current ones
# (in pyscx: `Experiment.reload()`)

exp.reload()
exp.read_obs()      # now carries the imported columns
```

The rules, in full:

- **It covers the objects the handle hands out, not just the handle.**
  `adata = pyscx.open(p).to_anndata(backed=True)` drops the `Experiment` on the
  same line, and the backed `X` / `obsm` / layers, a `query()` pipeline, and a
  backed MuData each hold their own reader. All of them refuse a changed file.
- **`reload()` does not reach them.** It refreshes the `Experiment` only;
  re-derive anything taken out of it (`exp.to_anndata(backed=True)` again).
  Reviving them in place would silently mix arrays from two file versions.
- **Mutating *through* a handle is fine.** `pyscx.obs_import(exp, "calls.csv")`
  and `exp.mark_deleted(mask)` reload the handle for you.
- **`path`, `closed`, `close()`, `reload()` and `repr()` never raise** — a
  stale handle reprs as `<Experiment 'atlas.scx' [stale: …]>`. Everything else
  on the class does.
- **Only pyscx handles are watched.** `scx-ops`, the CLI and the ML training
  loader open readers around their own writes and are unaffected; the check
  costs them nothing.
- **Cloud handles are not covered.** `open_cloud` has no local mapping and no
  inode to compare; detecting a changed object would need a request per read.
- **Timestamps are not the signal.** The check compares the file's inode and
  its catalog pointer, so `touch`, a metadata-preserving copy, or a backup pass
  does not invalidate a handle — and, in the other direction, a mutation that
  leaves size and mtime untouched is still caught. (Both happen: Linux updates
  inode timestamps from a coarse clock, so a fast open→mutate sequence can land
  with an *identical* `st_mtime_ns`.)

`close()` releases the mapping without waiting for the garbage collector, which
is what you want before rewriting a file in place, and is required on Windows,
where a mapped file cannot be replaced at all:

```python
with pyscx.open("atlas.scx") as exp:
    obs = exp.read_obs()
pyscx.compact("atlas.scx", "atlas.scx")   # mapping already released
```

Objects taken out of a `with` block keep their own readers and stay usable
after it exits; closing the handle you opened them from does not close them.


### Container and dtype materialization

The read APIs materialize `X` (and layers) as a scipy `csr_matrix` of
`(int64 indptr, int32 indices, float32 data)` by default. The `container`,
`data_dtype`, `index_dtype`, and `allow_lossy` kwargs let a reader choose the
output container and numeric dtype directly, so downstream consumers
(sklearn / PyTorch / scVI, GPU batches) don't over-allocate or re-densify.

Surfaced on `Experiment.to_anndata`, `PyQueryResult.to_anndata`, and
`PyQueryResult.to_csr` (`Experiment.to_gpu_anndata` accepts them for parity but
rejects any non-default request — the device path is f32-native).
`QueryPipeline.collect` takes the two that decide the *decode* — `data_dtype` and
`allow_lossy`, plus `index_dtype` — and not `container`, which is a presentation
choice applied afterwards and cannot lose anything.

| kwarg | values | default | notes |
|-------|--------|---------|-------|
| `container` | `"csr"` \| `"dense"` | `"csr"` | `"dense"` returns a row-major `numpy.ndarray` (no scipy CSR) |
| `data_dtype` | `float16/32/64`, `int8/16/32/64`, `uint8/16/32` | `None` → `float32` | numeric dtype of the values |
| `index_dtype` | `int16` \| `int32` \| `int64` | `None` → `int32` below `i32::MAX` nonzeros, `int64` above (pyscx does not choose `int64` on a file with deletion vectors; scipy still may) | CSR column-index dtype; ignored (warns) for `"dense"`. **Note:** `csr_matrix` resolves the index dtype from `max(nnz, n_rows)` and ignores the width it was handed, so on a matrix that fits in int32 **neither** `int16` nor `int64` survives — int16 is upcast, int64 is downcast, both to int32. The narrow int16 buffer is still built and range-gated on the way through (a column index ≥ 32768 fails loud). The default is resolved per matrix (`X` and `adata.raw` decide separately) — see **int64 above 2³¹ nonzeros** below |
| `allow_lossy` | `bool` | `False` | fail-loud cast gate — see below |

**Zero-copy default preserved.** `container="csr"` with no dtype kwargs takes the
exact pre-existing path: the decoded `Vec`s are moved into numpy with `copy=False`
and no cast. This is guaranteed byte-identical and is the performance-sensitive
common case.

**int64 above 2³¹ nonzeros.** `csr_matrix` resolves **one** index dtype for
`indices` and `indptr`, from `max(nnz, n_rows)` rather than from the arrays it
was handed. Above `i32::MAX` it picks int64 whether or not anyone asked — and
gets there by **copying** the int32 array pyscx handed it, while that array is
still alive. Measured on a 960,195 × 6,143 file with 2,650,704,199 nonzeros: a
16.3 B/nnz assembly transient settling to 11.9 B/nnz. The default eager read
therefore decodes indices at int64 directly once the matrix is over that line,
via the same typed reader a non-default `index_dtype=` uses. The returned scipy
object is identical — same values, same dtypes, still `copy=False` — and no
whole extra index array is alive at the peak.

The gate matters in both directions: at or below the line scipy **downcasts**
int64 inputs, so widening a matrix that does not need it would add a copy rather
than remove one. That is also why it is **off entirely on a file with deletion
vectors**: those are applied after assembly, so the catalog's nnz is an upper
bound on what scipy is handed, and a matrix whose physical nnz is over the line
but whose live nnz is under it would pay an oversized index buffer *and* the
downcast copy. The live count is not derivable from the catalog, so such files
keep the pre-existing behaviour exactly. `X` and `adata.raw` decide
independently, because scipy decides per `csr_matrix`. An explicit
`index_dtype=` is never overridden, and `container="dense"` has no column-index
array to widen.
`SCX_EAGER_INT64_NNZ_THRESHOLD` overrides the threshold; it exists so a small
fixture can exercise the widened decode, not as a tuning knob — the threshold is
scipy's, and below it the widen is a pessimization.

**In-decode narrow (eager `X` / `raw`).** A non-default request on the eager
`to_anndata` path narrows **in-decode**: each shard is decoded to its native
stream (integer counts as `u32`, floats as `f32`) and cast straight into a
full-matrix buffer *of the target dtype*, so the intermediate f32 CSR is never
allocated. A narrow `data_dtype="uint16"` read therefore **lowers** peak RSS
(2 B/nnz for the value buffer, not 4 B/nnz + a narrow copy) — see
[performance.md § Full read → AnnData](performance.md#full-read--anndata-wall-s--peak-rss-mb).
Eagerly-materialized **layers** still cast post-assembly (correct, no RSS win),
because there is no typed layer reader yet. `to_gpu_anndata` is unchanged
(f32-native device path).

**Fail-loud cast gate (`allow_lossy`).** With `allow_lossy=False` (the default),
any narrowing that would lose data raises `ValueError` rather than silently
corrupting values:

- out-of-range for the target integer dtype (e.g. `300 → uint8`),
- a fractional value into an integer dtype (e.g. `1.5 → int32`),
- a negative value into an unsigned dtype (sign loss, e.g. `-1 → uint16`),
- a count above 2²⁴ into `float16` (IEEE-754 cannot represent it exactly).

The error names the offending value and suggests a wider dtype or
`allow_lossy=True`. Widening casts (e.g. `uint8 → float32`, the default) are
always safe and are never gated.

**Decode-loss guard (the `u32 → f32` case).** The guard fires per *target dtype*:
a read fails loud when the shards' `value_max` cannot be represented exactly in
the requested `data_dtype`. Because the eager path now narrows **in-decode**
(integer counts cast straight from the native `u32` stream, never through f32), a
`> 2²⁴` integer count read into an **exactly representable** dtype
(`uint32` / `int64` / `float64`) now **succeeds losslessly** — where it previously
failed loud. The guard still fires (unless `allow_lossy=True`) for targets that
genuinely cannot hold the value: the plain `to_anndata()` default (`float32`, exact
only to 2²⁴), and `float16` (exact only to 2¹¹). Notes:

- The guard is **integer-encoding-only** and O(1): float-encoded shards record
  `value_max = 0` in the catalog, so continuous / log-normalized data never trips
  it, and the check is a single per-shard comparison (no data scan).
- For `float32` it is **conservative**: the catalog carries only the per-shard
  maximum, so a shard whose max exceeds 2²⁴ trips the guard even if that particular
  value is itself f32-exact. Read into `uint32` / `int64` / `float64` (exact), or
  pass `allow_lossy=True`, to bypass.
- The **query** path decodes at the requested dtype **when the dtype is declared
  before the decode** — `to_anndata(obs_filter=…, data_dtype=…)`, which resolves
  the plan and collects in one call, or `query().collect(data_dtype=…)`. A `> 2²⁴`
  count read that way is exact. A dtype named *after* a plain `collect()` is a
  **cast of values that were already decoded as f32**, so it fails loud and the
  message says where the dtype belongs; `allow_lossy=True` still accepts the
  rounding. See [Declaring the dtype at `collect()`](#declaring-the-dtype-at-collect).
- A **`var_names=` projection** assembles f32 too (it streams shard by shard through
  the same projecting reader the backed handles use, which has no typed variant), so
  a narrow `data_dtype=` takes that route only when the file's values survive an f32
  round trip — `value_max ≤ 2²⁴`, which includes every float-encoded file, since
  those record `value_max = 0`. Above that the projection stands down and the read
  falls back to today's full-width assemble-then-slice: same values, same peak as
  before the projection existed. Nothing is silently rounded either way. In practice
  the fallback is unreachable through pyscx's own write doors — both `from_anndata`
  and h5ad ingest route `X` through f32, so a count above 2²⁴ that is *not*
  f32-exact cannot be written from Python; it exists for files other scx tooling
  produces.

**Which matrices are guarded.** The guard covers every eagerly-decoded count
matrix: `X` (all `to_anndata` paths — default, `var_names`, `obs_filter`,
`preserve_slots`, and `to_gpu_anndata`), `adata.raw`, eagerly-materialized
`layers` (`to_anndata(eager=True)`), and each modality's `X` in
`to_mudata()`. The following are **not** gated — they decode lazily per-slice,
so a whole-file check would spuriously error on partial reads that never touch
the large-count shard:

- **Backed reads** (`backed=True`, including `to_mudata(backed=True)`).
- **Lazy `layers`** on the default (`eager=False`) `to_anndata()` — the layer
  is decoded only on later `adata.layers[...]` access.

The layers guard folds `value_max` over **the selected layers only**, so
`to_anndata(layers=["cpm"])` is refused by a `> 2²⁴` count in `cpm` and never by
one in a layer the call does not read. (`layers=` forces eager assembly, so a
filtered read is always a guarded read.) That holds at all three eager-layer
sites — the bridge branch, the `var_names=` projected assemble, and the
post-assembly narrow a non-default `data_dtype=` triggers — so none of them
disagrees about when a read raises. The retype site reads the layer keys off the
assembled object rather than re-deriving the filter, which is what keeps it in
step with the f32 decode guard that already ran.

For those ungated paths, a `> 2²⁴` count still rounds silently on access; pass
`allow_lossy` where available, or read eagerly to get the guard. To *know*
before reading: `Experiment.is_integer` / `Experiment.max_value` (catalog
stats, no decode) say whether a file holds counts and how large they get, and a
backed handle's `stored_dtype` names its on-disk encoding.

The R bindings (`rscx`) wire the same guard behind the same `allow_lossy`
opt-out: eager reads — `$x_matrix()`, `$layer()`, `$to_seurat()` / `$to_mae()`
(per modality) — use the shared catalog `value_max` folds on `FullCatalog`
(`csr_max_value` / `layer_csr_max_value`), the same implementation pyscx's
eager reads use; the R query path, like Python's, guards on the
engine-computed max over the shards the query actually selected, not a
catalog-wide fold.

> **Note.** A scipy `csr_matrix` with `float16` `data` is valid but cannot be
> densified by scipy's own `.toarray()` (a scipy limitation) — call
> `.astype(np.float32).toarray()`, or request `container="dense"` directly.

Backed reads (`backed=True`) are lazy and f32-native, so a non-default plan with
`backed=True` raises; materialize eagerly (`backed=False`) to narrow.

**Caveats.**

- **`index_dtype` for CSR is best-effort.** scipy canonicalizes a
  `csr_matrix`'s index arrays on construction (typically to `int32`, `int64`
  above the 32-bit nnz/dimension limit), so `index_dtype="int16"` will usually be
  upcast back to `int32` and delivers no reliable memory saving for CSR output.
  The `int64` widen sticks. For a guaranteed narrow-index layout, use
  `container="dense"` (no index array) instead.
- **`adata.raw` is retyped in-decode.** When the file carries a raw count
  matrix, the reconstructed `adata.raw.X` is assembled directly at the
  requested `data_dtype` (same typed reader as `X`) and gated by the same
  decode-loss check; it stays CSR even under `container="dense"` (the
  conventional raw representation).
- **Scope.** The kwargs are surfaced on the three read entry points above, on
  `QueryPipeline.collect` (`data_dtype` / `index_dtype` / `allow_lossy` — see
  [Declaring the dtype at `collect()`](#declaring-the-dtype-at-collect)), and on
  the flat `pyscx.read_cloud(...)` cloud helper, which is a one-call query and so
  takes them for the same reason `collect()` does. The grouped-shard read helpers
  (`read_group` / `read_reference` / `iter_group_shards`) are f32-native, and say
  so in their refusal rather than recommending a `data_dtype` they do not accept.

### `uns` serialization

`adata.uns` is written into the `UnsBlob` section (id 10) as JSON. The
`uns_format` kwarg on `from_anndata()` / `from_10x()` selects the envelope:

| Mode | Default | NumPy ndarray | NumPy scalar | tuple | pandas Cat/Index/Series | pandas DataFrame | NaN/Inf in array |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `"tagged"` | ✓ | `__scx_type__: "ndarray"` envelope (base64-LE bytes for numeric; JSON string list for object/string; `__scx_type__: "recarray"` for structured) | `__scx_type__: "scalar"` envelope (1-element base64) | `__scx_type__: "tuple"` envelope | `__scx_type__: "categorical" / "pandas.Index" / "pandas.Series"` envelope preserving name/codes/categories/ordered | `__scx_type__: "pandas.DataFrame"` envelope: a `pandas.Index` envelope for the index, an explicit ordered `columns` list, and one `ndarray` / `categorical` envelope per column | preserved bit-exact — ndarray bytes, and a bare non-finite Python `float` (anywhere: top level, list, tuple, dict, structured-array field, pandas `name`) becomes a float64 `scalar` envelope that reads back as `np.float64` |
| `"plain"` |  | collapses to nested list (`.tolist()`) | collapses to Python scalar | collapses to JSON array | collapses to JSON array via `.tolist()` | raises `ValueError` | raises `ValueError` |

`uns` is one JSON document, parsed whole on every `to_anndata()`, every
`Experiment.read_uns()` and every `open_cloud(...).read_uns()`, and rewritten
whole by every in-place `set_uns` / `update_uns` / `attach_obs_columns(uns=)`
(which leaves the superseded section behind — `scx info` reports the
fragmentation). That is worth knowing before putting a large table there: a
`rank_genes_groups(pts=True)` pair on a 60 k-gene × 30-group atlas is roughly
29 MB of raw float64, ≈ 38 MB once base64-encoded. Numeric columns always take
the base64-LE `ndarray` form (≈ 10.7 B/value) rather than a JSON number list
(≈ 18 B/value and 2–3× the parse time), but the whole-document cost is
inherent to the section.

The reader **auto-detects** per value: dicts with a `__scx_type__` key are
decoded back to their original Python type; everything else passes through
as plain JSON. This means files written by older `pyscx` (or with
`uns_format="plain"`) read identically on a modern build, and modern
tagged files are forward-compatible — unknown future tags emit a
`UserWarning` and return the raw envelope dict for inspection.

Unsupported in both modes: `bytes` objects (no portable JSON
representation) and `datetime64` / `complex` / `timedelta` ndarray dtypes.
For a `DataFrame` specifically, also unsupported: a `MultiIndex` on either
axis, a non-string or duplicated column name (names key the envelope's `data`
object, so a duplicate would silently collapse two columns into one), and any
pandas extension dtype other than `category` (`Int64`, `boolean`,
`string[python]`, …) — cast it first. Each raises a `ValueError` naming the
column. One documented gap: a frame with **zero columns** reads back with
pandas' default empty `RangeIndex` for `columns`, since with no names there is
nothing for a columns dtype to travel on.

The **h5ad export** writes the anndata dataframe group directly and keeps every
column's exact dtype — an `int8` column lands as `int8`, a `bool` as a plain
`bool` dataset. It is all-or-nothing per frame: if any column has no faithful
anndata spelling (a `null` inside an object column, which a plain h5ad string
dataset cannot hold; a column whose name equals the index's, which would
collide in the group; a name HDF5 cannot carry as a single member — empty,
`.`, `..`, or containing `/`; a non-string index name, which h5ad has nowhere
to put), the
whole frame is written as a raw `__scx_type__` envelope subgroup instead, with
an `uns_exported_as_raw_envelope` warning. anndata reads the subgroup as a
nested dict, so DataFrame consumers will not recognise it. Declining beats
dropping the column, because the fallback keeps what dropping would discard —
every value **whose key HDF5 can carry**. A column named `"/evil"` or `""` has
no HDF5 member spelling anywhere, so the fallback drops it too, with its own
`skipped_uns_key` warning.
Non-finite raw Python `float` scalars (`nan` / `±inf`) are preserved under
`"tagged"` via the `scalar` envelope (they read back as `np.float64`, a
`float` subclass, bit-exact) and raise under `"plain"`, whose contract is
"lossless or refuse". Finite doubles round-trip bit-exact in both modes: the
writer is ryu-exact and every reader parses with `serde_json`'s
`float_roundtrip`, so a 17-significant-digit value comes back as the same
IEEE-754 pattern.

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
- `collect(*, data_dtype=None, index_dtype=None, allow_lossy=False) -> PyQueryResult` — Execute pipeline. The kwargs are keyword-only and choose the dtype `X` is **decoded at** (see [Declaring the dtype at `collect()`](#declaring-the-dtype-at-collect)); the default is the unchanged zero-copy f32 decode. `data_dtype="float32"` takes the default route too, so the fused transforms keep working.
- `count() -> int` — Matching cell count, without decoding `X` (ignores `limit`)
- `exists() -> bool` — Whether any cell matches, without decoding `X`

**Consumption semantics.** Builder steps mutate the pipeline in place and return it, so both
`p.filter_obs(...).limit(5)` and the statement form work on one object. A **failed builder step
leaves the pipeline unchanged and usable** — a bad predicate (`ValueError`) or an unknown gene name
(`KeyError`) neither applies partially nor invalidates the object, so an interactive typo can simply
be retyped. `count()` and `exists()` borrow. `collect()` consumes the pipeline **only when it
succeeds**; a failed `collect()` leaves it usable, so the offending step can be corrected and
re-collected. After a successful `collect()` every method raises `RuntimeError` — build a new
pipeline with `Experiment.query()`.

### PyQueryResult

- `to_anndata(container="csr", data_dtype=None, index_dtype=None, allow_lossy=False)` — Convert result to AnnData. The default is a zero-copy CSR; the four kwargs have the same semantics as [`Experiment.to_anndata`](#experiment) (`"dense"` container, narrow `data_dtype`/`index_dtype`, fail-loud `allow_lossy` gate), with one difference described below: here they **cast** what `collect()` already decoded. Any non-default request forgoes zero-copy (a cast/copy of `X`).
- `to_csr(container="csr", data_dtype=None, index_dtype=None, allow_lossy=False)` — Return just the scipy CSR matrix (or a dense `numpy.ndarray` for `container="dense"`), with the same dtype kwargs.
- Properties: `n_obs`, `n_vars`, `nnz`, `skipped_shards`, `total_shards`

Both conversions consume the result (zero-copy hand-off), but only on success: a **tripped
`allow_lossy` guard leaves the result intact**, so the retry the error message recommends
(`result.to_anndata(allow_lossy=True)`) works on the same object.

#### Declaring the dtype at `collect()`

**The dtype passed to `collect()` is the dtype the data is *decoded* at; a dtype
passed to `to_anndata()` / `to_csr()` is a *cast* of what was already decoded.**
`collect()` is where the I/O and the decode happen, so it is the only place a
dtype can change what comes off disk:

```python
exp = pyscx.open("atlas.scx")                       # value_max > 2**24
q = exp.query().filter_obs("cell_type == 'fibroblast'")

q.collect(data_dtype="float64").to_csr()            # exact, float64
exp.query().filter_obs(...).collect().to_csr(data_dtype="float64")   # ValueError
```

The second call raises **on this file** because the values are f32 by then:
returning them as `float64` would report f32-rounded numbers at the wider dtype.
The error names the dtype and points at `collect()`. It is the guard firing, not
a blanket rule — the same call on a file whose selected shards hold no count
above 2²⁴ succeeds and casts post-assembly, because there was nothing to lose. `uint32` and `int64` behave like `float64`
(each holds every `u32` exactly); `float16`, `uint16` and the plain `float32`
default still fail loud on a `> 2²⁴` count, because no decode order helps a
target that cannot hold the value; `allow_lossy=True` accepts the rounding
anywhere.

Three details worth knowing:

- On a result collected with an explicit `data_dtype`, `to_anndata()` /
  `to_csr()` with **no** `data_dtype` return that dtype (not `float32`). Passing
  a *different* one raises: the caller wants either another decode (re-collect)
  or a numpy `.astype()` of what they hold, and `allow_lossy` does not unlock it.
- `container=` stays on the materialize call, since it is applied after the
  decode. `container="dense"` works on either kind of result, and on the
  one-shot `to_anndata(obs_filter=…, data_dtype=…, container="dense")` route,
  which decodes natively and scatters afterwards.
- `index_dtype` at `collect()` narrows the index buffer, but scipy normalises a
  `csr_matrix`'s index width on construction, so the returned matrix does not
  report it (the same caveat as `index_dtype="int16"` above). Passing a
  *different* `index_dtype` to `to_anndata()` / `to_csr()` on a dtype-selected
  result raises rather than being ignored; for `container="dense"` it is
  accepted and irrelevant, since dense output has no indices.
- `with_normalize()` / `with_log1p()` with a non-`float32` `data_dtype` is
  **refused**: those replace the stored counts with floating-point values, so no
  dtype reproduces the stored data exactly, and serving it from the f32 route
  would silently change which guard ran. Collect without a dtype (the values are
  transformed anyway) or drop the transform.

The same applies over cloud, on both spellings: `open_cloud(url).query()` returns
the same pipeline, so `collect(data_dtype=…)` decodes losslessly there — and
`pyscx.read_cloud(url, …)` takes `data_dtype` / `index_dtype` / `allow_lossy`
directly, for the same reason `collect()` does (that call *is* the decode).

### CloudExperiment

The Python-visible class is `CloudExperiment` (the Rust type is
`PyCloudExperiment`). Returned by `pyscx.open_cloud()`. Cloud-hosted SCX
handle. Supports metadata accessors plus a cloud-native query path served over
`object_store` range reads; full `to_anndata()` and `validate()` still
require `pyscx.pull()` to materialise the file locally. For a one-call read see
[`pyscx.read_cloud(...)`](cloud.md#pyscxread_cloud--one-liner-cloud-read). Its
`repr` is the AnnData-style header line (no key lists — listing them would need
network reads).

- `n_obs` `→ int` — Number of observations (cells)
- `n_vars` `→ int` — Number of variables (genes)
- `shape` `→ tuple[int, int]` — `(n_obs, n_vars)`
- `nnz` `→ int` — Total non-zero entries
- `shard_count` `→ int` — Number of CSR shards in the file
- `format_version` `→ int` — SCX format version
- `codec_id` `→ int` — Default codec ID
- `obs_keys() → list[str]` / `var_keys() → list[str]` — column names (one range read each)
- `is_multimodal() → bool` / `n_modalities() → int` / `modality_names() → list[str]` /
  `modality_id(name) → int | None` / `modality_info(id) → dict | None` — modality discovery,
  mirroring the local `Experiment` (the modality table is fetched once at open). On a
  single-modality file `is_multimodal()` is `False` and `modality_names()` is empty.
- `to_mudata()` — raises: cloud per-modality reads aren't supported yet; `pyscx.pull()` the file
  locally and open it with `pyscx.open(...).to_mudata()`.
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

  The flat one-liner `pyscx.read_cloud(url, obs_filter=..., var_names=...)`
  wraps this chain and returns an AnnData directly — see
  [docs/cloud.md § `pyscx.read_cloud(...)`](cloud.md#pyscxread_cloud--one-liner-cloud-read).

  Deferred follow-ons: `CloudQueryOptions` (parallelism,
  max-inflight bytes, cache-dir, retry policy) and a batched async section
  fetcher (current cloud reads block per shard from the rayon worker).

### Per-surface capability matrix

The three handle surfaces — local `pyscx.Experiment`, `pyscx.CloudExperiment`,
and rscx `ScxExperiment` — overlap but are not identical. This table shows where
each capability lives so you don't have to rediscover it per surface. "via
query" means the method isn't on the handle directly; reach it through
`query().collect()` (the result object) instead.

| Capability                       | `Experiment` (local) | `CloudExperiment`        | rscx `ScxExperiment` |
| -------------------------------- | -------------------- | ------------------------ | -------------------- |
| `n_obs` / `n_vars` / `nnz`       | ✓                    | ✓                        | ✓ (`$n_obs()` …)     |
| `shape`                          | ✓                    | ✓                        | — (use `n_obs`/`n_vars`) |
| `obs_keys()` / `var_keys()`      | ✓                    | ✓                        | — (`$obs()` / `$var()` data.frames) |
| `is_multimodal()` / `modality_names()` | ✓              | ✓                        | ✓                    |
| `to_anndata()` / extraction      | ✓                    | via query / `read_cloud` | via `query() …$collect()` |
| backed (out-of-core) X           | ✓ (`to_anndata(backed=True)`) | — (pull locally) | ✓ (`$x_backed()` / `scx_backed_sparse()`) |
| lazy transform chain             | ✓ (`ScxLazyTransformedDataset`) | — (pull locally) | ✓ (`$x_lazy()` / `scx_lazy_transform()` + `scx_normalize_total`/`scx_log1p`/`scx_row_scale`) |
| `to_mudata()` (multimodal)       | ✓                    | — (pull locally)         | `$to_mae()` / `$to_seurat()` |
| `detection_counts()` / `cells_expressing()` | ✓        | — (pull locally)         | —                    |
| `query()` builder                | ✓                    | ✓                        | ✓ (`scx_query()`)    |
| query `.count()`                 | ✓                    | ✓                        | ✓ (`count()`)        |
| `provenance()`                   | ✓                    | —                        | —                    |

Extraction off a query result is symmetric: pyscx `result.to_anndata()` /
`.to_csr()`; rscx `result$to_dgcmatrix()` / `$to_sce()` / `$to_seurat()` /
`as(result, "SingleCellExperiment")` (rscx adds S3/S4 generics — `dim()`,
`as.matrix()`, `as.data.frame()`, `as(res, "dgCMatrix")` — over the `$`-methods).

### pyscx.accel — Rust-Native Accelerators

All accelerators write results to standard AnnData slots (same as scanpy), so downstream functions work identically.

#### `prefer_format="auto"|"csr"|"csc"` kwarg

Several accelerators take a `prefer_format` kwarg selecting between the row-major
CSR path and the column-major CSC sidecar path. The DE ops (`rank_genes_groups`,
`pdex_ref`) default to `"auto"` (CPU routes CSC-direct when a valid sidecar is
present, else CSR; GPU stays CSR for the `gpu_csc_v3` planner route); every other
`prefer_format`-taking op defaults to `"csr"`. See
[scanpy.md § prefer_format](scanpy.md#prefer_formatautocsrcsc-column-major-dispatch)
for the full dispatch rules and requirements.

##### CSC decision matrix

CSC sidecars are the column-major substrate for column (gene-axis)
algorithms. Build one at conversion time with `csc="auto"` / `csc="always"`
(`pyscx.from_anndata` / `from_h5ad` / `from_10x`) or `scx convert --csc=auto`,
or after the fact with `scx build-csc` / `pyscx.build_csc(input, output=None)` (omit `output` to rebuild in place).
`csc="auto"` builds a sidecar only
when the dataset is large enough to benefit — `n_obs ≥ 50000` **and**
`n_vars ≥ 5000` by default, tunable via `SCX_CSC_AUTO_OBS_THRESHOLD` /
`SCX_CSC_AUTO_VARS_THRESHOLD`.

| Op | `supports_csc` | Default format | GPU-fast with CSC | Notes |
|----|:--:|:--:|:--:|-------|
| `pdex_ref` | ✅ | CSR | ✅ (`gpu_csc_v3`) | CSC-direct GPU route by default when a CSC sidecar is present; in-memory CSR falls back to `gpu_csr_v3`. |
| `rank_genes_groups` (Wilcoxon rank-sum) | ✅ | CSR | ✅ (`gpu_csc_v3`) | CSC-direct GPU route by default when a CSC sidecar is present; in-memory CSR falls back to `gpu_csr_v3`. CPU CSC kernel via `prefer_format="csc"`. |
| `rank_genes_groups_df` | ✅ | CSR | ✅ (`gpu_csc_v3`) | Same Wilcoxon rank-sum engine as above. |
| `pseudobulk_dex` | ✅ (gene subset) | CSR | N/A (CPU + pydeseq2) | CSC requires a gene subset (`gene_indices` or `col_projection`); full-gene CSC has no win. |
| `highly_variable_genes` (seurat_v3) | ✅ (single-batch) | CSR | ❌ | CSC routes single-batch seurat_v3; multi-batch / GPU / other flavors raise on CSC. |
| `calculate_qc_metrics` | ✅ (gene axis) | CSR | N/A | Gene-axis aggregation uses CSC; cell-axis stays CSR. `prefer_format="csc"` specifically rejects a scipy/dense `X` and a layer source — the sidecar belongs to `X`. `layer=` itself is supported on the default CSR route. |
| `col_sums` / `col_nnz` / `col_min` / `col_max` / `col_var` | ✅ | CSR | N/A | Column reductions; CSC requires no row deletion vector. |
| `pca` | ❌ (rejects CSC) | CSR | N/A | Inherently row-major; `prefer_format="csc"` raises `ValueError`. |
| `neighbors` / `umap` / `leiden` | N/A | — | N/A | Operate on PCA embeddings / kNN graphs, not on `X`. |

"GPU-fast with CSC" means the op reaches peak GPU throughput **only** with a
backed SCX file that has a CSC sidecar — see
[scanpy.md § GPU-supported vs GPU-fast](scanpy.md#gpu-supported-vs-gpu-fast).
Confirm which path actually ran via the [route metadata](#accelerator-route-metadata).

#### `gene_chunk_size` and DE memory

`rank_genes_groups`, `pdex_ref`, and `rank_genes_groups_df` **in compute mode**
(`groupby=`) process genes in chunks, densifying `n_obs × gene_chunk_size` `f32`
per chunk (default `None` → 500 genes) and ranking that. Extraction mode
(`group=`) reads `uns[key]` and densifies nothing. `pdex_nb_glm` accepts the
kwarg but does **not** chunk — it holds all genes in memory and warns that
`gene_chunk_size` is not implemented — so none of this section applies to it. The chunk is clamped to
`SCX_ACCEL_DE_MEMORY_BUDGET` (default 4 GiB) — an atlas-scale `n_obs` with a
large chunk would otherwise request hundreds of GB — and a request that cannot
fit even 32 genes raises rather than letting the allocation fail.

On a **backed or lazy** `X` the shard walk is *inner* to the chunk walk: every
shard the call visits is read once per gene chunk. Three things follow.

- **Size the decoded-shard LRU to the shards the call visits**
  (`to_anndata(backed=True, cache_shards=…)`), or each chunk re-decodes what the
  last one evicted. Below that, pyscx warns once per call — against the *visited*
  shard count, so a row-windowed read is not told to size a cache for the whole
  file.
- **A row projection skips the shards it empties** — a deletion vector,
  `adata[mask]`, `adata[:n]` — so the cost is `n_gene_chunks × visited shards`,
  not `× n_shards`. See
  [docs/sharding.md § Row-projected reads](sharding.md#row-projected-reads-skip-whole-shards).
- **Decode is prefetched** up to `SCX_ACCEL_PREFETCH_DEPTH` (default 4) shards
  ahead, bounded by whatever the dense workspace left of the DE budget. The
  pipeline holds `depth` decoded shards where a sequential loop held one; on a
  plain backed handle those are the same objects the LRU already holds, so the
  extra cost appears only where the handle rebuilds each shard (a view, or a
  lazy transform chain), and a budget-bound file resolves to depth 1 and keeps
  its sequential footprint.

- `pyscx.accel.pca(adata, n_comps=50, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", prefer_format="csr", allow_tf32=False, spmm_policy="default", memory_budget=None, mask_var=None)` — Randomized/covariance SVD PCA with streaming SpMM. Writes `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]` (incl. `uns["pca"]["params"]`). On GPU: cuSPARSE SpMM + cuSOLVER QR (f32). **`mask_var`** (scanpy semantics): a var-column name, a boolean array of length `n_vars`, or `None`. `None` auto-consumes `adata.var["highly_variable"]` when present, else uses all genes. PCA runs on the selected columns only (backed/lazy/in-memory alike, no materialization — via a column-projecting shard source), while `varm["PCs"]` stays aligned to the **full** `var` axis (excluded vars filled with 0). An all-false mask or a length mismatch raises `ValueError`. **`memory_budget`** (`"8G"`, `"512M"`, bytes, or `None` → 8 GiB) is the RAM ceiling for out-of-core PCA on a backed `X`. Out-of-core PCA re-reads every shard once per pass — ~6–7 passes for randomized, 3 for covariance — so the budget sizes a decoded-shard LRU that lets each shard decode once per pass instead of once per *read*; a budget too small to hold the working set logs a warning and the passes re-decode. It also bounds decode-prefetch, on every input shape: up to `depth` shards decode concurrently ahead of the reduction, and the depth is resolved against this ceiling rather than taken from the process-wide default. On a backed `X` the reserve comes out of the shard LRU, so worst-case live bytes are `budget + one shard` — the same high-water the pre-prefetch loop had. A lazy/transformed `X` has no LRU, so the ceiling bounds the pipeline directly; a shard larger than the budget falls back to depth 1 and prefetches nothing. Prefetch depth is `SCX_ACCEL_PREFETCH_DEPTH` (default 4, capped by the rayon pool and by `SCX_ACCEL_NUM_THREADS`); setting it to `1` disables the pipeline and restores the strictly sequential decode.
- `pyscx.accel.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0, ef_construction=200, ef_search=200, device="auto")` — kNN graph + UMAP-style connectivities. CPU: HNSW. In-VRAM `device="gpu"` routes to rapids-singlecell (`rsc.pp.neighbors`); the native standalone CAGRA dispatch was removed, so with rapids absent / `SCX_FORCE_NATIVE_GPU=1` the standalone op falls back to CPU HNSW (device-resident CAGRA survives only inside the fused `pca_neighbors` path). Writes `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`.
- `pyscx.accel.pca_neighbors(adata, n_comps=50, n_neighbors=15, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", use_rep="X_pca", prefer_format="csr")` — Fused PCA → kNN in one call. For an in-memory `X`, in-VRAM `device="gpu"` routes to the rapids-singlecell pipeline (`rsc.pp.pca` → `rsc.pp.neighbors`, route `rapids_singlecell_gpu`). For backed/lazy `X` (or under `SCX_FORCE_NATIVE_GPU=1`) the native device-resident path runs: the PCA embedding stays GPU-resident and feeds straight into CAGRA (no `X_pca` host round-trip), recording route `gpu_device_resident`. With neither GPU path available it falls back to sequential `pca` + `neighbors`. Writes the union of both ops' slots (`obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]`, `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`). **Note:** the fused entry points do not expose `mask_var`; the CPU/rapids routes delegate to `pca()` and so auto-consume `adata.var["highly_variable"]` when present, but the native device-resident GPU route analyzes all genes — so the PCA gene set is route-dependent when HVGs are flagged. For a deterministic HVG-masked pipeline, run `pca(mask_var=...)` then `neighbors()`/`umap()` separately. See [docs/scanpy.md § Fused PCA → kNN](scanpy.md#fused-pca--knn-pyscxaccelpca_neighbors).
- `pyscx.accel.pca_neighbors_umap(adata, n_comps=50, n_neighbors=15, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, umap_learning_rate=1.0, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", use_rep="X_pca", prefer_format="csr")` — Fused PCA → kNN → UMAP in one call. For an in-memory `X`, in-VRAM `device="gpu"` runs the full rapids-singlecell pipeline (`rsc.pp.pca` → `rsc.pp.neighbors` → `rsc.tl.umap`, route `rapids_singlecell_gpu` on every stage). The native device-resident UMAP path was removed, so backed/lazy inputs (and any GPU host without rapids) fall back to sequential `pca` + `neighbors` + `umap` (the umap stage uses the cuML fallback then CPU SGD). Writes their union including `obsm["X_umap"]`. See [docs/scanpy.md § Fused PCA → kNN → UMAP](scanpy.md#fused-pca--knn--umap-pyscxaccelpca_neighbors_umap).
- `pyscx.accel.umap(adata, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, learning_rate=1.0, random_state=0, device="auto")` — Spectral-init SGD UMAP. In-VRAM `device="gpu"` routes to rapids-singlecell (`rsc.tl.umap`); the native CUDA SGD kernel was removed, so with rapids absent it falls back to cuML and then CPU SGD. Writes `obsm["X_umap"]`.
- `pyscx.accel.rank_genes_groups(adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, stratify_by=None, min_cells_per_stratum=50, rankby_abs=False, tie_correct=False, prefer_format="auto", device="auto", use_raw=None, layer=None, *, pts=False, groups=None, corr_method="benjamini-hochberg")` — Parallel Wilcoxon rank-sum with BH correction. Writes `uns["rank_genes_groups"]`, or returns DataFrame when `stratify_by` is set. `pts=True` also writes scanpy's `uns[key]["pts"]` (and `["pts_rest"]` for `reference="rest"`): `genes × groups` float64 DataFrames of the fraction of cells with a nonzero value, indexed by var name, over every gene — one extra streaming pass over the analysed matrix, route-independent; `pts_rest` is scanpy's `X[~mask_g]` fraction over every other cell, unlabelled cells included. `groups=` restricts which groups are **reported**, in the given order; the pool each group is compared against is unchanged, so its numbers equal the unrestricted run's and scanpy's `groups=` (the reference is silently not tested; unknown names / repeats / an empty list raise). Any **participating** group with fewer than two cells raises scanpy's "only contain one sample" error — every level when `groups=` is omitted, the named ones plus a named reference when it is given; an unused category counts as zero cells and raises too, with `remove_unused_categories()` named in the message. Cells with no `groupby` label get no group, result row or `pts` column, but for `reference="rest"` they are in the rank pool and in every group's `"rest"` (scanpy's rule, **changed in 0.17**); a pairwise run against a named reference leaves them out entirely. "No label" means `pandas.isna` — `NaN`, `None` and `pd.NA` alike — not how a value prints, so a group *named* `"nan"` / `"None"` / `""` is a real group and is kept. `pts=True` refuses duplicate `var_names` (its table is joined by name; `adata.var_names_make_unique()`), and `rank_genes_groups_df` refuses a `pts` table whose index is duplicated. `corr_method` accepts only `"benjamini-hochberg"` (recorded in `params`); any other value raises. Both frames round-trip through `from_anndata` / `to_anndata` and h5ad via the `uns` `pandas.DataFrame` envelope. `prefer_format="csc"` routes per-chunk reads through the column-major sidecar (see kwarg docs above). The execution route is recorded on `uns["rank_genes_groups"]["scx_accel_route"]` and `uns["scx_accel"]["rank_genes_groups"]` (see [Accelerator route metadata](#accelerator-route-metadata)).
- `pyscx.accel.pseudobulk_dex(adata, groupby=None, test_col=None, reference=None, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr", gene_indices=None, n_cpus=None, backend=None, nbglm_options=None, *, sample_cols=None, sample_key=None)` — Streaming pseudobulk aggregation (Rust) + DE testing. Returns DataFrame. **`groupby` here is not what it is in `rank_genes_groups`**: it names the obs columns that together define one pseudobulk *sample* — condition **plus** replicate, e.g. `["disease", "donor_id"]` — while the column actually compared is `test_col`. (In `rank_genes_groups`, and all of scanpy, `groupby` *is* the compared column.) `sample_cols=` / `sample_key=` are aliases named for that role, with `sample_key` also accepting a bare string; pass exactly one of the three, or a `ValueError` names the ones you supplied. `groupby` / `test_col` / `reference` remain semantically required — they are typed optional only so the aliases can exist, and each has its own error when omitted. `backend=None` resolves to `"nb_glm"`, the Rust-native NB-GLM (no optional dependency, route `cpu_nb_glm`); `backend="pydeseq2"` uses [pydeseq2](https://pydeseq2.readthedocs.io/) (the `pydeseq2` extra) for exact DESeq2 numerics and returns the identical column schema. The default was `"pydeseq2"` through v0.12 and changed because pydeseq2 is optional, so the default path did not run on a base install; `stratify_by=` and `aggr_method="mean"` are pydeseq2-only and raise under the default with that guidance, and a one-time `UserWarning` fires on a default-backend call when pydeseq2 is importable. On the NB-GLM route the gene-axis layout is reported in `csc_available` rather than in `route`. A custom `design=` formula is honored on the NB-GLM backend too (built via `formulaic`, the parser pydeseq2 uses — one shared-dispersion fit, per-level contrasts; needs the `nbglm` extra), with an optional explicit `nbglm_options["contrast"]`. `prefer_format="csc"` requires a gene subset (either an explicit `gene_indices` argument or a `col_projection` already set on `adata.X`); full-gene CSC pseudobulk has no measurable speed-up. See [docs/pseudobulk_nb_glm.md](pseudobulk_nb_glm.md).
- `pyscx.accel.nb_glm(counts, design, size_factors=None, contrast=None, gene_names=None, sample_names=None, options=None, counts_axis="samples_by_genes", device="auto") → pandas.DataFrame` — Direct Rust-native negative-binomial GLM on an **already-pseudobulked** count matrix + numeric design (DESeq2 replacement). `f64` end-to-end; GPU routes (`gpu_nb_glm_csr`, `gpu_nb_glm_csc`) accelerate the pseudobulk aggregation while the IRLS / Cox–Reid fit itself remains CPU. `counts` is `[n_samples × n_genes]` (`counts_axis="samples_by_genes"`, default) or `[n_genes × n_samples]`; `design` is `[n_samples × n_features]`, full column rank. `size_factors=None` → DESeq2 median-ratio factors. `contrast` is an integer coefficient index, a weight vector, or `None` (last coefficient, DESeq2 convention). `options` is an optional dict (`dispersion ∈ {"moments","cox_reid_mle","cox_reid_shrunk"}`, `min_disp`, `max_disp`, `max_irls_iters`, `irls_tol`, `max_outer_iters`, `fit_dispersion_trend`, `shrink_dispersion`). Returns PyDESeq2-style columns `gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj, dispersion, converged, n_iter` (`lfcSE` on the log2 scale). DESeq2-*style*, not DESeq2-*identical* — includes Cook's-distance outlier filtering and base-mean independent filtering (both default-on), but omits apeglm/ashr LFC shrinkage. See [docs/pseudobulk_nb_glm.md](pseudobulk_nb_glm.md).
- `pyscx.accel.pdex_nb_glm(adata, groupby, reference, stratify_by=None, min_cells_per_group=10, min_cells_per_stratum=50, is_log1p=None, nbglm_options=None, gene_chunk_size=None, prefer_format="csr", device="auto", design=None, output="pandas") → pandas.DataFrame` — Pseudobulk NB-GLM DE for the cell-eval/pdex consumer. Aggregates `groupby × stratify_by` pseudobulk **replicates**, fits one NB-GLM per non-reference perturbation vs `reference`, and returns the **same** cell-eval `DEResults` column schema as `rank_genes_groups_df` (`target, feature, fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change`). **`output="polars"` is required to feed `cell_eval`** — its `DEResults.data` is typed `pl.DataFrame`; the default `output="pandas"` needs no optional dependency. **`stratify_by` is required**, as a **list** of obs column names (e.g. `["donor"]`; a bare string is rejected with a clear error) — it forms the replicates: with no stratifier the dispersion is unidentifiable, so it raises `ValueError` pointing to `pdex_ref` / `rank_genes_groups`. `is_log1p=None` auto-detects via `adata.uns["log1p"]`; NB-GLM requires **raw counts** and errors on log1p-normalized input. Records route on `adata.uns["scx_accel"]["pdex_nb_glm"]` (GPU routes `gpu_nb_glm_csr` / `gpu_nb_glm_csc`; CPU route `cpu_nb_glm`). See [docs/pseudobulk_nb_glm.md](pseudobulk_nb_glm.md).
- `pyscx.accel.rank_genes_groups_df(adata, groupby=None, reference="rest", n_genes=None, gene_chunk_size=None, rankby_abs=False, tie_correct=False, device="auto", output="pandas", *, group=None, key="rank_genes_groups", pval_cutoff=None, log2fc_min=None, log2fc_max=None) → pandas.DataFrame` — **Two modes.** *Compute* (`groupby=`): same Wilcoxon rank-sum as `rank_genes_groups()`, returns the cell-eval `DEResults` schema `(target, feature, fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change)`; pass `output="polars"` for `cell_eval.initialize_de_comparison()`. *Extract* (`group=`): the **scanpy `sc.get.rank_genes_groups_df` alias** — does not recompute; reads the precomputed `adata.uns[key]` and returns scanpy's columns `(names, scores, logfoldchanges, pvals, pvals_adj)`, with a leading `group` column when `group` is a list **or `None`**, and `pct_nz_group` / `pct_nz_reference` appended when `uns[key]` carries `pts` / `pts_rest` (after `rank_genes_groups(pts=True)`), looked up by gene name as `sc.get.rank_genes_groups_df` does. `pval_cutoff` / `log2fc_min` / `log2fc_max` are scanpy-style row filters (extraction only). Pass either `groupby=` or `group=`, not both. **`group=None` (or omitting it) with no `groupby=` extracts every group** in `uns[key]`, matching scanpy's "All groups are returned if group is None"; with neither and no `uns[key]` it errors naming both remedies. Both modes return a **pandas** DataFrame by default, so scanpy-shaped idioms (`.map`, `df[col] = …`) work directly; `output="polars"` returns the polars frame `cell_eval` consumes and requires the `eval` extra. (`gene_symbols=` var-name remap is not supported yet.)
- `pyscx.accel.pdex_ref(adata, groupby, *, reference="non-targeting", is_log1p=None, geometric_mean=True, epsilon=1e-9, cpm_filter=None, gene_chunk_size=None, prefer_format="auto", device="auto", output="pandas", use_raw=None, layer=None, groups=None) → pandas.DataFrame` — Perturbation-screen differential expression: Mann–Whitney U + pseudobulk geometric-mean log fold change vs a single reference group. **`groups`** (a pyscx extension; upstream pdex has none) restricts the tested targets to those `groupby` levels, reported in that order — each target is only ever compared with the reference, so its rows equal the unrestricted run's while the work and GPU footprint scale with the targets asked for; unknown names, repeats, an empty list and the reference itself raise. Pinned bit-for-bit to upstream [`pdex`](https://github.com/ArcInstitute/pdex) (`pyscx/tests/test_pdex_ref_parity.py`). Returns one row per (target group, gene) excluding the reference group, with columns `target`, `feature`, `target_mean`, `ref_mean`, `target_membership`, `ref_membership`, `log2_fold_change` (also exposed as `fold_change` for migration), `percent_change`, `p_value`, `statistic`, `fdr`. Note: here `fold_change` is a deprecated alias of `log2_fold_change` (log2 scale) — this differs from `rank_genes_groups_df`'s `fold_change` column, which is linear (`exp2` of the log2 value). **`epsilon`** (default `1e-9`, matching pdex ≥ 0.2.x) is a finite-guard pseudocount added to the count-space means before the fold-/percent-change ratio — not applied to CPM or the MWU test; with the default the outputs stay finite, and `0/0` (a gene unexpressed in both groups) is `0.0`. Pass `epsilon=0.0` to recover the legacy behaviour where genes undetected in the reference yield `±inf`. **`cpm_filter`** (optional float `T`) keeps a gene for a test group iff its pooled arithmetic counts-per-million `target_cpm > T` **or** `ref_cpm > T` (strict `>`, mode-independent); dropped rows are removed and FDR is recomputed over the surviving genes only. `is_log1p=None` auto-detects, and the answer does not depend on how `X` is stored — a backed handle and an in-memory `AnnData` over the same data resolve the same mode. Resolution order: `adata.uns["log1p"]` (present → log1p); else, for a lazy `X`, a `Log1p` in its transform chain; else, for a backed `X`, the catalog's integer `value_max` against the same `< 30` heuristic the in-memory path uses (catalog-only — no decode, no materialization); else, in memory, `max(X) < 30`. The matrix probed is the one `use_raw=` / `layer=` selected, not always `adata.X`. **A backed file whose catalog cannot bound its value range, with no `uns["log1p"]`, raises `ValueError`** rather than guessing — shards that are float-encoded (the format writes `value_max = 0` for those), carry no statistics, or hold no values at all. Answering `False` there is what used to make a backed `pdex_ref` disagree with its in-memory twin by orders of magnitude. Pass `is_log1p=True`/`False` to override or to resolve that case (`adata.X.max()` computes the heuristic's input by streaming, without materializing). **`output`** is `"pandas"` by default (no optional dependency) or `"polars"` for the frame upstream `pdex` and `cell_eval` use; columns are identical either way. `device="auto"` picks GPU when available and falls back to CPU otherwise. **GPU v3-CSC path (default):** the GPU dispatch routes through a CSC-direct driver that drops the per-chunk dense intermediate and uses a CSC shard source with pipelining + per-chunk shard-range pre-filter. This is the default GPU DE route (the former `SCX_GPU_DE_V3` opt-in gate was removed when v3 became the unconditional default). CSC-direct requires a CSC sidecar on the SCX file — build it with `pyscx.from_anndata(adata, path, csc="always")` or `scx convert --csc=always`. In-memory inputs (scipy CSR) and files without a sidecar fall back to the v3-CSR-direct path automatically. The route that actually ran is recorded on `adata.uns["scx_accel"]["pdex_ref"]` (see [Accelerator route metadata](#accelerator-route-metadata)) — `gpu_csc_v3` confirms the CSC-direct path, `gpu_csr_v3` + `fallback_reason="no_csc_sidecar"` confirms the CSR fallback. Wall-time numbers and the disposition live in [`docs/performance.md` § Per-operation timing](performance.md#per-operation-timing).
- `pyscx.accel.pseudobulk_means(adata, groupby, min_cells_per_group=1, device="auto") → (ndarray, list[str] | list[tuple[str, ...]])` — Group-by mean on sparse X, streaming shard-by-shard (works on backed, lazy, scipy CSR, or dense). `groupby` is one obs column or a list of columns keyed by the per-cell tuple (the `str | list[str]` `pseudobulk_dex` takes). Returns `(means[P, G] float64, sorted group names)` — plain strings for one column, tuples (one entry per column) for several. Foundation for the perturbation evaluation metrics below. GPU (`device="gpu"`/`"auto"` on a GPU host): wraps the DE pseudobulk kernels — f64 accumulation matches the CPU within `atol≈1e-5` (dense `embed_key` at the f32 bar). Route `gpu_csr` / `cpu_csr` (the GPU pseudobulk reuses the DE CSR kernels). An **in-memory** `X` is copied once into an owned buffer so the kernel can run with the GIL released — a parallel copy, measured at ~14 ms per 192 MB, with a `UserWarning` above 1 GB; a backed or lazy `X` streams and never pays it. See [Coding conventions § Python Bindings](conventions.md#python-bindings-pyscx).
- `pyscx.accel.perturbation_metrics(adata_real, adata_pred, pert_col="perturbation", control="control", metrics=None, embed_key=None, min_cells_per_group=1, device="auto") → dict[str, dict[str, float]]` — Bundled bulk metrics `{pearson_delta, mse, mae, mse_delta, mae_delta}` between paired real/pred AnnData. Matches cell-eval's metrics within atol=1e-6. GPU (`device="gpu"`/`"auto"` on a GPU host): the per-group pseudobulk means run on the GPU (f64), the five bulk metrics on the host — CPU parity `atol≈1e-6`. Route `gpu_csr` / `cpu_csr`.
- `pyscx.accel.energy_distance(adata_real, adata_pred, pert_col="perturbation", control="control", metric="euclidean", embed_key=None, backend=None, dtype=None, device="auto") → float` — Pearson correlation of per-perturbation e-distance vectors (real vs pred). **Polarity:** despite the name this is a *score*, not a distance — it lies in `[-1.0, 1.0]` where `1.0` = perfect and **higher is better**; use it directly, don't invert. Returns `nan` when either side's e-distance vector is constant (e.g. a control-broadcast predictor) or there is only one non-control perturbation. Never materializes the `[N, N]` distance matrix (the reduction keeps one `f64` per row); precomputes control self-distance once; rayon-parallel across perturbations. The gemm backend's Gram is built one row block at a time, bounded by `SCX_ACCEL_PAIRWISE_MEMORY_BUDGET` (default 256 MiB) — see [scanpy.md § Energy distance](scanpy.md#energy-distance-pyscxaccelenergy_distance) for what that budget does and does not cover. `backend ∈ {"auto" (default), "gemm", "scalar"}` — `"auto"` picks faer-dispatched gemm for euclidean/cosine and the scalar row-by-row path for L1; `"gemm" + metric="l1"` raises `RuntimeError` (no decomposition exists). `dtype ∈ {"f32" (default), "f64"}` controls only the matmul / per-pair arithmetic precision; reductions always accumulate in `f64`. f32 + gemm matches f64 + scalar within `atol=1e-4` correlation / `atol=1e-3` per-pert. GPU (`device="gpu"`/`"auto"` on a GPU host): a gemm-based GPU pairwise-distance mean (`‖x−y‖²=‖x‖²+‖y‖²−2·xyᵀ`) handles **euclidean + cosine** at f32 with f64 reductions — CPU parity at the same `atol=1e-4` correlation bar. `metric="l1"` (no gemm decomposition) and `dtype="f64"` stay on the CPU path even under `device="gpu"` (route `cpu_csr`, no crash). GPU route `gpu_dense`.
- `pyscx.accel.energy_distance_details(...)` — Same signature (including `backend` / `dtype` / `device`) as `energy_distance` but returns `{"correlation": float, "d_real": {pert: float}, "d_pred": {pert: float}, "pert_names": [...]}`.
- `pyscx.accel.discrimination_score(adata_real, adata_pred, pert_col="perturbation", control="control", metric="l1", exclude_target_gene=True, embed_key=None, min_cells_per_group=1, device="auto") → dict[str, float]` — Per-perturbation normalized rank of the predicted perturbation effect's distance to the correct real effect. `metric ∈ {"l1", "l2"/"euclidean", "cosine"}`. `exclude_target_gene=True` drops the gene matching each perturbation's name from the distance (matches cell-eval's default).
- `pyscx.accel.knockdown_efficiency(adata, pert_col="perturbation", control="control", eps=1e-8, device="auto")` — Per-cell knockdown efficiency + log-fold change vs control baseline. Input must be normalized (NOT log1p'd); log1p is applied internally. Writes `adata.obs["KnockDownEfficiency"]` and `adata.obs["KnockDownGeneFC"]` (both float32, NaN for control cells and cells whose perturbation name isn't in `var_names`). Matches `arc_bench.tools.normalize_transform.core` within atol=1e-6.
- `pyscx.accel.clustering_agreement(adata_real, adata_pred, pert_col="perturbation", control="control", metric="ami", real_resolution=1.0, pred_resolutions=None, n_neighbors=15, embed_key=None, min_cells_per_group=1, device="auto") → float` — Builds perturbation-centroid kNN graphs, sweeps Leiden resolutions, scores best real-vs-pred agreement via AMI / NMI / ARI. **All-native-Rust** — no scanpy / anndata / igraph dispatch; uses `scx_accel::neighbors::build_knn_graph` (HNSW via `instant-distance`, `ef_construction=200, ef_search=50, seed=0`) plus `scx_accel::leiden` sequential mode (`max_iterations=2, parallel=false, seed=0` — matches scanpy's `flavor="igraph", n_iterations=2`). Pred-side kNN graph built once and reused across the resolution sweep; whole hot path runs under `py.allow_threads`. Matches cell-eval's `ClusteringAgreement` within `atol=0.15` aggregate (stochastic Leiden; exact score match not expected — algorithms agree exactly on graphs with `n_perts ≥ 16`).
- `pyscx.accel.adjusted_mutual_info(labels_a, labels_b) → float` — AMI on integer label arrays (arithmetic-mean convention). Matches `sklearn.metrics.adjusted_mutual_info_score` within atol=1e-10.
- `pyscx.accel.normalized_mutual_info(labels_a, labels_b) → float` — NMI (arithmetic-mean). Matches `sklearn.metrics.normalized_mutual_info_score` within atol=1e-10.
- `pyscx.accel.adjusted_rand_index(labels_a, labels_b, rescaled=False) → float` — ARI. Default matches `sklearn.metrics.adjusted_rand_score` (`1.0` identical, `~0` random, negative for worse-than-random; consistent with NMI/AMI above). Pass `rescaled=True` for cell-eval's `(ARI + 1) / 2` in `[0, 1]`.
- `pyscx.accel.leiden(adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=2, device="auto", parallel=False, theta=1.0)` — Community detection on the kNN graph. Reads `obsp["connectivities"]` (from `neighbors()`). GPU: cuGraph Leiden (up to 47× faster). CPU: **Rust-native** — the Python `igraph`/`leidenalg` shim was removed (call `scanpy.tl.leiden(flavor="leidenalg")` directly if you need it). The CPU path is **Louvain + constrained refinement**: it omits the Leiden paper's well-connectedness admissibility conditions and CPU `theta` (non-default `theta` is warned and ignored on CPU), so do not treat it as a full well-connected-Leiden guarantee. Writes `adata.obs[key_added]` (categorical) and `adata.uns["leiden"]` (params + backend metadata). GPU and CPU may produce different partitions due to algorithmic differences; compare via ARI/NMI.
- `pyscx.accel.harmony_integrate(adata, key, *, basis="X_pca", adjusted_basis="X_pca_harmony", n_clusters=None, theta=None, sigma=0.1, lamb=None, alpha=0.2, max_iter=10, max_iter_kmeans=6, epsilon_harmony=1e-2, epsilon_kmeans=1e-3, block_size=0.05, batch_prop_cutoff=1e-5, tau=0.0, random_state=0, device="auto")` — Clean-room Rust implementation of Harmony2 (Korsunsky et al., 2019). Soft k-means clustering with a diversity penalty, followed by ridge-regression correction of `adata.obsm[basis]` (`"X_pca"` by default). `key` accepts a single `obs` column name or a list for multi-covariate batch correction; each is factorised via `pandas.factorize(sort=False)`. Writes the corrected embedding (f32, N × d) to `adata.obsm[adjusted_basis]` — by default a **new** key `"X_pca_harmony"`, leaving the input `basis` intact (matching scanpy); pass `adjusted_basis=basis` (e.g. `"X_pca"`) to overwrite in place. Convergence metadata goes to `adata.uns["harmony"]` (`params`, `converged`, `n_iterations`, `objective_harmony`, `backend`). Parameter names and the non-destructive default match `scanpy.external.pp.harmony_integrate`, so existing scanpy pipelines can swap in. GPU path (when built with `--features gpu`) accelerates distance / L2-norm / batched scatter-subtract kernels; k-means++ init and covariance inversion stay on CPU. Clustering primitives pinned against harmonypy 0.2.0 Rust-side (`scx-accel/src/harmony/harmony_reference_values.rs`, gated by `cargo test`); end-to-end agreement is a per-PC correlation floor in the `accel_harmony` benchmark, not a numerical-parity claim — see [docs/scanpy.md § Harmony](scanpy.md#batch-integration--harmony2-pyscxaccelharmony_integrate) for the two documented divergences and for why the previous "r 0.989–0.999 vs R harmony v2.x" figure is withdrawn rather than restated.
- `pyscx.accel.compute_lisi(adata, key, *, basis="X_pca", perplexity=30.0, n_neighbors=None, approximate_knn=False) → np.ndarray` — Local Inverse Simpson Index on an `obsm` embedding. Exact brute-force kNN (matches R `FNN::get.knn`) + per-cell Gaussian-bandwidth search (t-SNE Hbeta routine) + Simpson index over kernel-weighted neighbour category probabilities. Returns LISI vector of length N and also writes to `adata.obs[f"lisi_{key}"]`. Values near 1 → poor mixing (neighbourhoods dominated by one category); values approaching the number of categories → uniform mixing. `n_neighbors` defaults to `ceil(3 × perplexity) − 1` (89 at the default perplexity) — harmonypy retrieves `3 × perplexity` and drops the self-match, while SCX's sweep skips it while collecting, so asking for `3 × perplexity` here would be one neighbour wider than the reference. Pinned against `harmonypy.lisi.compute_lisi` in `scx-accel/src/lisi_reference_values.rs`. ~10× faster than R `lisi::compute_lisi` on D1–D4 (the previously reported 0.8–2.4 % mean-LISI agreement predates the 2026-07 raw-distance kernel fix and is pending a benchmark recapture). Set `approximate_knn=True` to swap the exact O(N²) sweep for an HNSW approximate kNN (~10× faster at N ≳ 100k, ~0.01–0.05 mean-LISI drift); the exact path logs a hint to do so above ~50k cells.
- `pyscx.accel.normalize_total(adata, target_sum=None, device="auto")` — Materialization-free row normalization. `target_sum=None` (default) scales each cell to the **median of positive per-cell totals** (scanpy `target_sum=None` semantics); a float pins an explicit target. **Compatibility break (v0.11.6+):** the previous default was a hard `1e4` — pass `target_sum=1e4` to restore it. On `ScxBackedSparseDataset`: computes row sums via streaming (deriving the median deletion-correctly), creates `ScxLazyTransformedDataset` wrapper. On `ScxLazyTransformedDataset`: appends `NormalizeTotal` transform to chain. On scipy CSR: delegates to `sc.pp.normalize_total()` (forwarding `None`).
- `pyscx.accel.log1p(adata, device="auto")` — Materialization-free log1p. On `ScxBackedSparseDataset`: creates `ScxLazyTransformedDataset` with `Log1p` transform. On `ScxLazyTransformedDataset`: appends `Log1p` to chain (fuses with preceding `NormalizeTotal` when possible). On scipy CSR: delegates to `sc.pp.log1p()`. **Every arm stamps `adata.uns["log1p"] = {"base": None}`**, the same annotation `sc.pp.log1p` writes — so `rank_genes_groups` takes its `expm1` logFC branch, `pdex_ref` resolves the log1p mean mode, and `pdex_nb_glm` correctly refuses the result as non-raw-counts. (The lazy arms previously wrote nothing, leaving the SCX-native pipeline the only one whose log1p was invisible downstream.)
- `pyscx.accel.gpu_info() → dict` — Query GPU device info: `{'device': ..., 'total_vram_gb': ..., 'free_vram_gb': ...}`. Returns `None` if no GPU available.
- `pyscx.accel.estimate_gpu_memory(adata, operation, **kwargs) → dict` — Estimate GPU VRAM required for an operation. Returns `{'required_gb': float, 'fits_in_vram': bool}`. Supported operations:
  - `"pca"`: kwargs `n_components` (default 50), `n_oversamples` (default 10), `shard_size` (default 16384)
  - `"knn"`: kwargs `n_neighbors` (default 15), `n_dims` (default 50)
  - `"umap"`: kwargs `n_components` (default 2)
  - `"leiden"`: no additional kwargs

  Raises `ValueError` for unknown operations. Note: estimates are approximate — cuSOLVER QR workspace may be undercounted by ~1.5×. When NNZ is not available from the input data (e.g. a scipy matrix without a backing SCX file), the estimator falls back to a ~10% density assumption.

  **Estimation model per operation:**
  - **PCA**: `Y (n_obs × k × 4)` + `Ω+B (2 × n_vars × k × 4)` + `shard_dense (shard_rows × n_vars × 4)` + `QR (2 × n_obs × k × 4)`, where `k = n_components + n_oversamples`. This covers the native GPU randomized PCA working set; rapids PCA may use different internal allocations.
  - **kNN**: `embeddings (n_obs × n_dims × 4)` + CAGRA graph index overhead.
  - **UMAP**: `embeddings (n_obs × n_components × 4)` + graph working memory.
  - **Leiden**: graph adjacency + working buffers.

  See [gpu-setup.md § GPU memory model](gpu-setup.md#gpu-memory-model) for the full VRAM sizing guide.
- `pyscx.accel.calculate_qc_metrics(adata, qc_vars=None, log1p=True, inplace=True, prefer_format="csr", *, layer=None, percent_top=None)` — Streaming QC metrics, **one native kernel for every kind of `X`**: backed, lazy, a backed layer handle, or an in-memory scipy/dense matrix. Writes scanpy's obs columns (`n_genes_by_counts`, `log1p_n_genes_by_counts`, `total_counts`, `log1p_total_counts`, per-`qc_var` `total_counts_<v>` / `log1p_total_counts_<v>` / `pct_counts_<v>`) and var columns (`n_cells_by_counts`, `mean_counts`, `log1p_mean_counts`, `pct_dropout_by_counts`, `total_counts`, `log1p_total_counts`), in scanpy's order. When `inplace=True`, writes to `adata.obs`/`adata.var`; when `False`, returns `(obs_df, var_df)`. `prefer_format="csc"` routes the gene-axis aggregation through the CSC sidecar (cell-axis stays CSR — row aggregations have no CSC win); it rejects a scipy/dense `X` and rejects `layer=` (the sidecar belongs to `X`). `layer=` reads `adata.layers[name]` instead of `adata.X`.

  `percent_top=` adds `pct_counts_in_top_<n>_genes` for each 1-indexed position, computed in the same shard pass. It defaults to `None`, **not** scanpy's `(50, 100, 200, 500)`: that default raises `IndexError` on any file with fewer than 500 genes. Positions outside `1..=<visible genes>` raise `ValueError`; under a column projection the bound is the visible width, not the file's.

  *This op no longer requires `scanpy`.* It used to delegate to `sc.pp.calculate_qc_metrics` whenever `X` was neither of pyscx's two SCX handles — so not only in-memory scipy/dense, but also an anndata h5ad-backed `SparseDataset`, a cupy or dask matrix, and anything else duck-typed. All of those now go through `owned_csr` (`scipy.sparse.csr_matrix(x)`) instead, which accepts scipy and dense inputs and raises on a handle it does not recognise. That delegation meant two implementations behind one name — and they wrote different column sets, since the streaming route never produced `log1p_n_genes_by_counts`, `mean_counts`, `log1p_mean_counts` or `pct_dropout_by_counts`. The in-memory arm accepts a scipy sparse or numpy `X` and **refuses** anything else — a dask array, a cupy matrix, an h5ad-backed `CSRDataset` — because `owned_csr` would materialise it in full (for dask, a whole `compute()`) before producing a single number, and the `nnz × 8` large-copy warning is sized from the result of that materialisation, i.e. after the peak. Those inputs previously reached scanpy's own backend-aware arms; materialise deliberately (`adata.X = adata.X.compute()`) or open the file backed. For a caller who was on that route: the in-memory matrix is now copied (`nnz × 8` bytes, the shared >2 GB warning applies) where handing it to scanpy copied nothing; values accumulate in f64 rather than f32 (and a float64 `X` narrows to f32 on the way in, as elsewhere in pyscx); `pct_counts_<v>` for a cell with no counts is `0.0` rather than `NaN`; and explicitly-stored zeros are dropped from the copy before counting, so `n_genes_by_counts` counts real nonzeros as scanpy does — without `eliminate_zeros()`-ing the caller's matrix the way scanpy does.

  **Column projections** are honored on every route: per-cell totals cover only the visible genes, the gene axis is returned at the visible width, and `qc_vars` masks are read against `adata.var`. Projections arrive from `filter_genes`, `highly_variable_genes(subset=True)`, a gene-selected `to_anndata(backed=True, var_names=[...])`, `X.set_col_projection(...)`, or `adata.X = adata.X[:, cols]` — *not* from `adata[:, mask]`, which raises on a backed SCX `X` because anndata has no registered view type for it (that's the deferred `as_view`/`_subset` work).

  *Behavior change, and it differs by route.* Under `prefer_format="csr"` (default) a projection previously produced a physical-width gene axis, so the call raised `ValueError: Length mismatch` — loud, no wrong numbers. Under `prefer_format="csc"` the gene axis was already correct, so lengths matched and the call **returned silently wrong per-cell numbers** (projection-blind row axis; `qc_vars` visible positions applied to the on-disk axis). Projected QC results published from the CSC route are worth re-checking. Separately, on a lazy `X` the `qc_vars` subset sums are now taken **through** the transform chain, so `pct_counts_<v>` divides a transformed numerator by a transformed denominator — a silent numeric change that shifts which cells a `pct_counts_mt` filter keeps.

  Requires `adata.var` row *i* to describe visible column *i*. `set_col_projection` sorts and dedups, so slicing `var` in a different (unsorted) order mislabels the gene axis without erroring.

  All of the above is computed in **two** shard passes — one row-axis, one column-axis — for up to 64 `qc_vars`; beyond that the row pass repeats once per additional 64. `percent_top` adds no pass: the per-row top-N is selected from the values the row pass already has in hand.
- `pyscx.accel.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", batch_key=None, span=0.3, subset=False, n_bins=20, device="auto", prefer_format="csr", layer=None)` — Streaming HVG selection. Default CPU + CSR streams `mean_var` and clipped sums shard-by-shard via `ShardSource`; multi-batch runs CSR. Works on `ScxBackedSparseDataset`, `ScxLazyTransformedDataset`, **and a materialized scipy/dense `X`** for `flavor` in `seurat_v3` / `seurat_v3_paper` / `seurat` (a materialized `X` is wrapped in a single-shard `ShardSource`), so the eager `to_anndata()` idiom gets the same numerics and the same per-batch LOESS-singularity tolerance as the backed path. Only flavors scx does not implement natively (e.g. `cell_ranger`) delegate to `scanpy.pp.highly_variable_genes` (one-shot `UserWarning`). `prefer_format="csc"` routes single-batch seurat_v3 through `streaming_mean_var_csc` and `streaming_clip_square_sum_csc` (multi-batch + GPU + non-seurat_v3 flavors raise on CSC). Writes `var["highly_variable"]`, `var["means"]`, `var["variances"]`, `var["variances_norm"]`, `var["highly_variable_rank"]`.
- `pyscx.accel.score_genes(adata, gene_list, ctrl_size=50, gene_pool=None, n_bins=25, score_name="score", random_state=0, method="control", layer=None, device="auto", *, ctrl_genes=None)` — CPU-native `sc.tl.score_genes` equivalent. Streams shard-by-shard via `ShardSource`, so it runs on `ScxBackedSparseDataset`, `ScxLazyTransformedDataset`, and materialized scipy/dense `X` with bounded memory. Writes a per-cell float64 score to `adata.obs[score_name]`. `method="control"` (default) = `mean(gene_list) − mean(control)` with control genes sampled from expression-matched bins by a Rust-native deterministic sampler, which is **not** numpy's — it draws different control genes from scanpy, so the absolute scores differ (see `docs/scanpy.md` for measured numbers). `method="mean"` = per-cell mean over `gene_list`; `method="zscore"` = decoupler `mt.zscore` (`Σ z / √k`, ddof=1 std). **`ctrl_genes=` supplies the control set directly and skips the sampling — the exact-parity route**: given the controls scanpy used, the score matches it, and it works on a backed `X`, where `sc.tl.score_genes` raises `NotImplementedError`. It requires `method="control"` and rejects an explicit `gene_pool=` (which exists only to be sampled from); `ctrl_size` / `n_bins` / `random_state` are ignored because there is no sampling left to control. `gene_list`/`gene_pool`/`ctrl_genes` are symbols resolved against `adata.var_names`; missing genes are dropped with a `UserWarning`. CPU-only (`device` accepted for symmetry; no GPU kernel). Records its route (`cpu_csr`) on `adata.uns["scx_accel"]["score_genes"]`.
- `pyscx.accel.pflog(adata, *, alpha=None, layer=None, store="pca", n_components=50, n_oversamples=10, n_power_iterations=2, zero_center=True, random_state=0, obsm_key="X_pflog_pca", baseline_key="pflog_baseline", layer_out=None, out=None, store_repr="delta_baseline", shard_size=None, dense_max_elems=200_000_000, device="auto")` — **PFlog (v4) / shifted-log on raw counts** normalization (Booeshaghi et al., DOI 10.1101/2022.05.06.490859; no direct scanpy equivalent). **By default (`store="pca"`) it produces a baseline-aware PCA embedding in `adata.obsm[obsm_key]` and leaves `X` as raw counts — it does *not* transform `X` in place like `normalize_total`/`log1p`; pass `store="dense"` (+ `out=` for large data) for the matrix itself.** Operates on **raw counts**: shift by the matrix-wide Anscombe pseudocount `1/(4α)`, `log`, then subtract the within-cell mean — no per-cell depth (it cancels under the Anscombe scale). `α` (NB overdispersion `Var = μ + α·μ²`) is estimated once from the counts when `alpha=None`, or pinned to a float (e.g. a reference `α` reused across datasets); the fit is stamped into `adata.uns["pflog"]` (`alpha`, `pseudocount`, `alpha_source`, and — when estimated — `n_genes_used`/`fell_back`). The estimate is the **median of the per-gene method-of-moments `α_g = (var_g − mean_g)/mean_g²` over every gene with `mean_g > 1e-3`**, negatives included; see [docs/scanpy.md](scanpy.md#pflog-normalization-pyscxaccelpflog) for why there is no dispersion filter and why the pool is combined by a median. **`n_genes_used` changed meaning when that filter was removed**: it now counts every gene in that pool, where before it counted only the over-dispersed ones, so it is larger on the same counts. The exact transform `Z = delta + baseline·1ᵀ` is dense, but decomposes into a sparse `delta` (= `scale{4α}` → `log1p`, i.e. `log1p(4α·x)`) plus a per-cell `baseline = −(1/D)·Σ_j delta_ij`, so PCA never densifies. Streams shard-by-shard via `ShardSource` (backed / lazy / in-memory `X`; a transformed lazy `X` is rejected — raw-count guard). Always writes `adata.obs[baseline_key]`. `store ∈ {"pca","all"}` → baseline-aware out-of-core randomized PCA into `adata.obsm[obsm_key]` (+ `adata.uns[f"{obsm_key}_singular_values"]`). `store ∈ {"dense","all"}` materializes the exact dense `Z`: with `out=None` into `adata.layers[layer_out]` (guarded by `dense_max_elems`), or with `out=<path.scx>` streamed to a new SCX file (no size guard) per `store_repr` — `"delta_baseline"` (default, compact `O(M)`: Pcodec `delta` X + `baseline` obs, reconstruct with `pyscx.accel.pflog_reconstruct`) or `"dense"` (literal full-density CSR, forced Zstd, small default `shard_size`). CPU-only (`device` accepted for symmetry; no GPU kernel). Records its route (`cpu_csr`) on `adata.uns["scx_accel"]["pflog"]`.
- `pyscx.accel.pflog_reconstruct(adata, baseline_key="pflog_baseline") → np.ndarray (f32)` — Reconstruct the exact dense PFlog `Z = delta + baseline[:, None]` from a compact `store_repr="delta_baseline"` file (`X` is `delta`, `obs[baseline_key]` is the baseline). Same cheap kernel the training loader applies on read; a compact file can equivalently be streamed through `TrainingDataset` with its transform mode off (precompute-once / train-many-epochs).
- `pyscx.accel.col_sums(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column sums on `ScxBackedSparseDataset`. Honors `col_projection` and `kept_to_global` on the CSR path; CSC dispatch requires no row deletion vector and (currently) only supports `ScxBackedSparseDataset` and `ScxLazyTransformedDataset` (CSR scipy / dense raises a helpful message — use the array-protocol `dataset.sum(axis=0)` for those).
- `pyscx.accel.col_nnz(dataset, prefer_format="csr") → np.ndarray (i64)` — Streaming per-column NNZ. Same dispatch as `col_sums`.
- `pyscx.accel.col_min(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column min. Implicit-zero correction (`mins[c] = min(mins[c], 0.0)` when `col_nnz[c] < n_obs`) applied on both paths.
- `pyscx.accel.col_max(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column max. Symmetric implicit-zero correction.
- `pyscx.accel.col_var(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column variance. CSR uses a two-pass formulation; CSC uses single-pass `(sum_x² - n·mean²) / n`. Numerically equivalent within f64 epsilon (verified by `pyscx/tests/test_csc_dispatch.py`).
- `pyscx.accel.filter_cells(adata, min_genes=None, max_genes=None, min_counts=None, max_counts=None)` — Non-materializing cell QC filter for backed/lazy data. Computes row NNZ and/or row sums via streaming, builds a boolean mask, and updates the deletion vector (`kept_to_global`) on `ScxBackedSparseDataset` or `ScxLazyTransformedDataset`. Slices `adata.obs` and every obs-aligned member (see [Axis subsetting and aligned members](#axis-subsetting-and-aligned-members)). Falls back to `sc.pp.filter_cells()` for scipy/dense. On a lazy `X` under an active column projection, the thresholds are evaluated against the **visible** genes, matching scanpy on the sliced object.
- `pyscx.accel.filter_genes(adata, min_cells=None, max_cells=None, min_counts=None, max_counts=None)` — Non-materializing gene QC filter for backed/lazy data. Computes column NNZ and/or column sums via streaming, builds a boolean mask, and sets `col_projection` on `ScxBackedSparseDataset` or `ScxLazyTransformedDataset`. Slices `adata.var` and every var-aligned member. Composes with existing column projections. Falls back to `sc.pp.filter_genes()` for scipy/dense.
- `pyscx.accel.subset_obs(adata, mask_or_indices)` — Subset observations (cells) without materializing. Accepts a **boolean numpy mask** or **integer index array**. Creates a new deletion vector (`kept_to_global`) on the backing dataset; `adata.obs`, `raw` and every obs-aligned member follow via anndata. Composes correctly with existing deletion vectors. Equivalent to `adata[mask].copy()` except that `X` is not materialized.
- `pyscx.accel.subset_var(adata, mask_or_indices)` — The var-axis twin: subset genes (columns) in place without materializing. Equivalent to `adata[:, mask].copy()`, minus the materialization. Prefer this over `adata.X.set_col_projection(...)`, which moves `X` alone.

  > [!WARNING]
  > **Integer index semantics differ from NumPy.** When `mask_or_indices` is an integer array, it is internally converted to a boolean mask. This means:
  > - **Duplicate indices are silently collapsed** — `[0, 0, 5]` is equivalent to `[0, 5]`
  > - **Order is not preserved** — `[5, 0, 10]` produces the same result as `[0, 5, 10]`
  >
  > This differs from NumPy's fancy indexing where `a[[5, 0, 10]]` returns rows in the order `[5, 0, 10]` with duplicates preserved. Use a boolean mask for unambiguous results.

#### Axis subsetting and aligned members

`filter_cells`, `filter_genes`, `subset_obs`, `subset_var`, and `highly_variable_genes(subset=True)` all mutate one axis of the AnnData in place. **anndata performs the subset**; pyscx only makes a backed `X` subsettable and keeps its lazy mappings off disk. So `obs`, `var`, `uns`, `raw`, unused categorical levels and every aligned member behave exactly as they do on an in-memory AnnData:

| parent axis | members updated |
|---|---|
| obs (`filter_cells`, `subset_obs`) | `obs`, `raw`, `layers` (rows), `obsm`, `obsp` (both axes) |
| var (`filter_genes`, `subset_var`, HVG `subset=True`) | `var`, `layers` (columns), `varm`, `varp` (both axes) |

Two properties are worth stating explicitly, because they are what the SCX-specific machinery exists to preserve:

- **Nothing materializes.** The subset of a backed / lazy `X` is a `kept_to_global` / `col_projection` update, and so is the subset of an SCX-handle layer or embedding — no copy, no decode. `type(adata.X)` is unchanged by a filter.
- **Nothing decodes.** `obsp` / `varp` / `varm` on `to_anndata(backed=True)` (and `obsm` under `to_anndata(backed=True, obsm=[...])`) are lazy mappings that read a section only on first key access. A subset is *recorded* and applied when the key is first read, so `filter_cells` on a file carrying a large kNN graph costs nothing extra unless you actually read `obsp`. This does not apply to the default (non-backed) `to_anndata()`, where the values are already in memory.

Members you added yourself — a numpy array, scipy matrix, or pandas DataFrame — are subset by anndata in the usual way.

**Accelerators run on the subset, not the file.** After any of these ops, a backed `X` is a *window* onto the file: `kept_to_global` selects the visible rows, `col_projection` the visible columns. Every streaming `pyscx.accel.*` op reads through that window — `normalize_total`, `log1p`, `calculate_qc_metrics`, `pca`, `pca_neighbors`, `neighbors`, `umap`, `leiden`, `harmony_integrate`, `highly_variable_genes`, `score_genes`, `pflog`, the `col_*` aggregations, `pseudobulk_means`, `pseudobulk_dex`, `rank_genes_groups`, `pdex_ref`, `pdex_nb_glm` and the perturbation-evaluation metrics all see exactly the matrix `adata.X` presents. Results are aligned to `adata.obs` / `adata.var`, and a `mask_var=` selection is resolved against the visible gene axis — so `filter_genes` → `highly_variable_genes` → `pca` masks the genes you filtered down to, not the on-disk ones at the same positions.

The one thing the window cannot express is a **request-ordered** gene axis (`preserve_var_order=True`), because the source emits columns in sorted on-disk order. Every op above refuses that combination on a backed `X` rather than returning a silent gene/column permutation — the `preserve_var_order` entry under `Experiment.to_anndata` lists them.

> [!NOTE]
> **`prefer_format="csc"` is the exception.** The gene-major sidecar is written against the full axis and has no projection surface, so an explicit CSC request on a subset dataset is refused with an actionable `RuntimeError` (naming the row deletion vector or the column projection) rather than reading the wrong columns. Use the default `prefer_format="csr"`, which streams the window. On GPU this is automatic: the dispatch routes a subset backed dataset to the CSR path and keeps the CSC-direct `gpu_csc_v3` route for unsubset ones.

A subset is **atomic**: anndata builds the replacement object and swaps it in, so a failure part-way through leaves the original untouched.

> [!NOTE]
> **`uns` is deep-copied by the subset.** anndata's replacement object carries a `deepcopy` of `uns`, so an entry that cannot be deep-copied — a lock, an open file handle, a live client object — makes the whole subset raise `TypeError`. The pre-4.0b backed path left `uns` alone and so tolerated these. Store non-copyable objects outside `uns`, or drop them before filtering. Everything the accelerators themselves put in `uns` (`scx_accel` route metadata, `pflog`, neighbors params) is plain data and copies fine.

pyscx registers SCX handles with three private anndata `singledispatch` hooks to make this work (`as_view`, `_subset`, `to_memory`), drives `_mutated_copy` / `_init_as_actual` directly, and reads `is_view` / `_adata_ref` to recognise a view. See [docs/compatibility-matrix.md § Private anndata APIs](compatibility-matrix.md#private-anndata-apis-pyscx-depends-on) for the supported versions and the compat test that fails loudly on an upgrade.

**Plain anndata indexing works too.** `adata[:, mask]` builds a lazy view of a backed `X` (before, it raised `NotImplementedError`), `adata[mask].copy()` subsets and **materializes** — the documented "subset, then run scanpy" workflow — and `adata[mask].to_memory()` returns a fully in-memory AnnData. Use `pyscx.accel.subset_obs` / `subset_var` when you want the subset applied in place *without* materializing.

**Accelerators on a view.** Handing that view straight to an accelerator also works: any `pyscx.accel.*` op that writes results back **rebuilds the view in place as a regular `AnnData` before it starts**, keeping `X` lazy, and emits an `ImplicitModificationWarning` saying so. Nothing is copied — the point of the rebuild is to avoid the copy. Two consequences to know about:

- The object you passed in stops tracking its parent (`adata.is_view` becomes `False`), and the results land on it, not on the parent. That is the same end state anndata's own copy-on-write reaches; the difference is that copy-on-write gets there by calling `.copy()`, which **materializes** a backed `X`. On a 500k × 3k gene subset that was a 1.8 GB → 10.3 GB jump.
- The rebuild happens on **entry**, before the op runs, so it is not conditional on the op succeeding: if the call then raises, `is_view` has still flipped to `False`. The rebuild has to precede the first write, and by then it is too late to know whether the op will finish.
- `pyscx.accel.subset_var` / `subset_obs` avoid the transition entirely — they subset in place and never produce a view. Prefer them in a pipeline you intend to keep out-of-core.

The rebuild is declined, and anndata's copy-on-write left to do its normal job, when `X` is a plain scipy/dense matrix (there is no lazy handle to protect). One carve-out: a view whose index is not expressible as a window — a duplicated or descending selection such as `adata[[2, 2, 7]]` — already holds a materialized `X`, so the rebuild installs scipy. The op still succeeds; it just is not out-of-core any more.

#### Accelerator route metadata

Every `pyscx.accel.*` call records the execution route it actually took on `adata.uns["scx_accel"][<op>]`. `rank_genes_groups` additionally copies the route string to `adata.uns["rank_genes_groups"]["scx_accel_route"]`.

> [!IMPORTANT]
> **Once a call returns, `adata.uns["scx_accel"][<op>]` is present if and only if that op completed.** An op that raises leaves no entry, so the key's presence is a usable "this ran" signal and not merely "this was attempted". If an earlier run of the same op had recorded an entry, a later failing run **restores that earlier entry unchanged** rather than deleting it — a bad re-run cannot erase a good stamp. A failing *first* accel op does not create `uns["scx_accel"]` at all.
>
> The "once a call returns" qualifier is real but narrow: thirteen ops stamp their route *before* dispatch, so the route is inspectable while a long backed run is still in flight, and the rollback happens as the call unwinds. Anything that reads `uns["scx_accel"]` from another thread, or from a debugger paused mid-call, can therefore see a stamp for an op that has not finished yet.
>
> There is deliberately no `status` / `success` field: absence *is* the failure signal, and adding one would mean every consumer had to check it to avoid trusting a stamp from a crashed op.

The dict carries:

- `route` — the concrete path taken. One of `cpu_dense`, `cpu_csr`, `cpu_csc`, `cpu_nb_glm` (native pseudobulk NB-GLM DE — see [docs/pseudobulk_nb_glm.md](pseudobulk_nb_glm.md)), `gpu_nb_glm_csr` / `gpu_nb_glm_csc` (GPU-accelerated pseudobulk aggregation for NB-GLM), `gpu_csr_v3`, `gpu_csc_v3` (the column-major perf path), `gpu_csr` / `gpu_dense` (non-DE GPU routes — see **Non-DE ops** below), `rapids_singlecell_gpu` (in-VRAM ops routed to rapids-singlecell — see **Non-DE ops**), or `gpu_device_resident` (the fused PCA→kNN path keeps the embedding on the GPU between stages when running the native backed/lazy path; see **Fused device-resident route**).
- `fallback_reason` — why the ideal route wasn't taken: `none`, `no_cuda`, `no_rapids`, `no_csc_sidecar`, `unsupported_dimensions`, `unsupported_input_layout`, `user_forced_cpu`, `perf_policy`, or `gpu_runtime_error`. All but the last are **pre-flight** conditions, resolved before any device work. `gpu_runtime_error` is the exception: it means the faster path failed on the device at run time and a slower one produced the result, so it is only ever recorded on an op that **succeeded** by another route — today, `to_gpu_anndata` falling through to host-assemble. A GPU op that fails outright raises; it does not silently re-run on CPU, and its route stamp is rolled back rather than rewritten to claim a CPU run that never happened (see the invariant above, and [gpu-setup.md § What `device="auto"` does and does not do](gpu-setup.md#what-deviceauto-does-and-does-not-do)).
- `chunk_size`, `csc_available`, `shards_decoded`, `shards_uploaded` — optional detail (`None` when not tracked). `shards_decoded` counts **slab passes** (`n_gene_chunks × n_shards` on the chunked routes), not host decodes — with `resident_csr` true a shard is decoded once and replayed from VRAM, and this counter does not move.
- `resident_csr` — GPU CSR DE routes **and native GPU PCA**. `True` when the whole matrix was held **device-resident**, `False` when the route streamed, `None` where there is no residency decision to make — CPU, dense, or a CSC-direct route, which prefilters by column range and never re-decodes. On DE this is across gene chunks (streamed = over the VRAM budget, only one gene chunk, or `SCX_GPU_DE_RESIDENT=0`); on PCA it is across the randomized power loop (streamed = over the VRAM budget or `SCX_GPU_PCA_RESIDENT=0`, in which case the operator re-decodes and re-uploads the whole matrix on every multiply). The decision is dynamic on both — taken against *free* VRAM at call time — so the same input can go either way run to run, which is why it is recorded rather than inferred from shape. See [scanpy.md § GPU DE device residency](scanpy.md#gpu-de-device-residency).
- `transfer_mode`, `bytes_uploaded`, `device_id`, `rapids_version`, `cuml_version`, `cupy_version` — GPU device-handoff / rapids detail. `transfer_mode` is how the GPU-resident matrix was produced, one of: `scx_device_decode_gpu` (framed Scx1 shards decoded fully in VRAM group-by-group — only the indptr is uploaded; covers dense ≥128-nnz rows via the BitPacker4x kernel; from `to_gpu_anndata`), `scx_device_handoff_streamed` (on-device but a shard host-bounced because it is not an Scx1 shard — a non-Scx1 codec; from `to_gpu_anndata`), `scx_device_handoff` (host-assembled CSR or an already-device-resident input; from `to_gpu_anndata` or a rapids op consuming a device-resident `X`), or `anndata_to_gpu` (rapids uploaded a host `X` via `rsc.get.anndata_to_GPU`). `bytes_uploaded` is the real host→device byte count for that mode (`None` when not tracked). `scx_device_handoff` has two causes that the mode string alone cannot separate — the request needed filtering, or the in-VRAM decode *failed* and the host path took over — so read `fallback_reason` alongside it: `gpu_runtime_error` means the fast path broke (a `UserWarning` names the device error) and is worth investigating; `none` means host-assemble was chosen up front.
- `math_mode`, `spmm_policy` — GPU PCA only (Task 2.5): the cuBLAS/cuSPARSE float math mode (`strict_fp32` / `allow_tf32`) and the cuSPARSE SpMM algorithm policy (`default` / `deterministic` / `benchmark_once`) the run used. `benchmark_once` is **reserved and not yet implemented** — it resolves to the same heuristic algorithm as `default`, and is echoed back so the wire API is stable for when per-shape timing lands; treat it as `default` when reading provenance. `None` on CPU / ops without these knobs. Both describe **whichever power loop ran** — the device-resident one and the streaming one apply the same requested policy, so a `>VRAM` run is not silently downgraded. Read `resident_csr` alongside them to tell the two apart. (Before v0.13.1 the streaming loop hardcoded the heuristic `CUSPARSE_SPMM_ALG_DEFAULT`, so `spmm_policy="deterministic"` was reported but not applied whenever the matrix exceeded VRAM.) **`"deterministic"` names an algorithm, not a guarantee:** it pins `CUSPARSE_SPMM_CSR_ALG2` instead of letting cuSPARSE choose heuristically, but it does **not** make GPU PCA bit-reproducible. cuSPARSE gives no reproducibility guarantee for transpose operations, and the randomized power loop issues `Aᵀ · Y` every iteration — so no algorithm choice makes the loop run-to-run bitwise identical. For guaranteed-reproducible PCA use `device="cpu"` (see [scanpy.md § Reproducibility](scanpy.md#reproducibility)).
- `graph_replay` — whether a captured CUDA graph was replayed. Meaningful for `harmony_integrate`, the only op that captures: `True` when the k-means sub-iter graph was captured and replayed. `False` means "no graph was replayed" and has four causes, not three: capture failed, capture produced no graph, `SCX_DISABLE_CUDA_GRAPHS=1` turned it off, or the run never reached a capture attempt — the first k-means sub-iteration is always a direct warm-up, so a call configured with only one sub-iteration in total (e.g. `max_iter=1, max_iter_kmeans=1`) has no second sub-iter on which to capture. In every case the kernels ran directly; read it as that, not as "capture was tried and failed" (the WARN log distinguishes the failure cases). Results are identical either way; throughput is not, which is why a failure is recorded (and logged at WARN naming the device error) rather than swallowed. `None` on CPU routes, which make no capture decision, **and on GPU PCA**, which no longer makes a capture decision: SpMM-segment capture was removed (`cusparseSpMM` is not capture-safe on current cuSPARSE), so the field could only ever read `False` there. The device-resident PCA power loop still avoids re-decoding/re-uploading the matrix each power iteration — read `resident_csr` for that, which is the decision PCA actually makes.

**Op keys:**

| Op key | Covers |
|--------|--------|
| `"pdex_ref"` | `pyscx.accel.pdex_ref` |
| `"rank_genes_groups"` | `pyscx.accel.rank_genes_groups` |
| `"rank_genes_groups_df"` | `pyscx.accel.rank_genes_groups_df` |
| `"highly_variable_genes"` | `pyscx.accel.highly_variable_genes` |
| `"score_genes"` | `pyscx.accel.score_genes` |
| `"pflog"` | `pyscx.accel.pflog` |
| `"pca"` | `pyscx.accel.pca` |
| `"neighbors"` | `pyscx.accel.neighbors` |
| `"pca_neighbors"` | `pyscx.accel.pca_neighbors` (also stamps `"pca"` + `"neighbors"`) |
| `"pca_neighbors_umap"` | `pyscx.accel.pca_neighbors_umap` (also stamps `"pca"` + `"neighbors"` + `"umap"`) |
| `"umap"` | `pyscx.accel.umap` |
| `"leiden"` | `pyscx.accel.leiden` |
| `"harmony_integrate"` | `pyscx.accel.harmony_integrate` (`gpu_dense` native cuBLAS path / `cpu_dense`; records `graph_replay`) |
| `"normalize_total"` | `pyscx.accel.normalize_total` |
| `"log1p"` | `pyscx.accel.log1p` |
| `"calculate_qc_metrics"` | `pyscx.accel.calculate_qc_metrics` |
| `"pseudobulk_dex"` | `pyscx.accel.pseudobulk_dex` |
| `"pseudobulk_means"` | `pyscx.accel.pseudobulk_means` (`gpu_csr` GPU pseudobulk / `cpu_csr`) |
| `"perturbation_metrics"` | `pyscx.accel.perturbation_metrics` (`gpu_csr` GPU pseudobulk / `cpu_csr`) |
| `"energy_distance"` | `pyscx.accel.energy_distance` / `energy_distance_details` (`gpu_dense` gemm for euclidean/cosine f32 / `cpu_csr` for L1, f64) |
| `"discrimination_score"` | `pyscx.accel.discrimination_score` (`cpu_*` — GPU deferred: exact-rank parity not f32-safe) |
| `"to_gpu_anndata"` | `pyscx.Experiment.to_gpu_anndata` (records `transfer_mode` / `bytes_uploaded`) |

This is the canonical way to confirm which path ran when comparing CPU vs GPU performance — GPU is fastest only when the input layout matches the op. The CSC-direct route (`route == "gpu_csc_v3"`) requires a backed SCX file with a CSC sidecar; in-memory CSR inputs (and files without a sidecar) report `gpu_csr_v3` with `fallback_reason == "no_csc_sidecar"`. v3 is the unconditional default GPU DE route (the `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3` opt-in gates were removed when v3 became the default). The route is decided by a single internal planner (`scx_accel::route::plan_de_route`) that also *drives* dispatch — the GPU `pdex_ref` **and** Wilcoxon rank-sum (`rank_genes_groups` / `rank_genes_groups_df`) entry points `match` on the planned route to select the kernel, and CPU dispatch calls the same planner — so the recorded value always matches the kernel that executed. Both DE ops take `gpu_csc_v3` (CSC sidecar) or `gpu_csr_v3` (otherwise) on GPU; dense-host input is densified to CSR and also records `gpu_csr_v3`. Route-specific benchmark gates in `thresholds.yaml` enforce correct dispatch across all GPU ops — see [benchmarks/README.md § Regression Gating](../benchmarks/README.md#regression-gating).

**Deprecated route identifiers.** The v1/v2 dense-materialization GPU DE drivers were removed (GPU DE is now always v3), and the non-DE GPU routes were renamed from the misleading `_v1` suffix to plain `gpu_csr` / `gpu_dense` in the follow-on taxonomy cleanup. So `gpu_csr_v1`, `gpu_dense_v1`, and `gpu_csr_v2` are all **historical** — they no longer appear in fresh output. Interpret them in old benchmark JSON as: `gpu_csr_v2` = legacy v2 DE; DE `gpu_dense_v1` = legacy v1-dense DE; non-DE `gpu_csr_v1` / `gpu_dense_v1` = today's `gpu_csr` / `gpu_dense`.

**Non-DE ops.** For an **in-memory** `X`, in-VRAM `device="gpu"` PCA, kNN (`neighbors`), UMAP, HVG-extra-flavors, and preprocessing route to **rapids-singlecell** (route `rapids_singlecell_gpu`, `fallback_reason="none"`) — see [§ rapids routing](#accelerator-route-metadata). The **native** GPU routes still apply for backed/lazy/streaming inputs and under `SCX_FORCE_NATIVE_GPU=1`: `pca` records `gpu_csr` (cuSPARSE+cuBLAS randomized — the in-VRAM covariance core was removed), `leiden` `gpu_csr` (cuGraph), `highly_variable_genes` `gpu_csr` (seurat_v3 atomic-CSR). Standalone `neighbors` and `umap` no longer have a native in-VRAM kernel (Phase 3.1/3.3): standalone `neighbors` falls back to CPU HNSW (`cpu_csr`) and `umap` to the cuML fallback (`gpu_dense`) then CPU SGD (`cpu_dense`); native CAGRA survives only inside the fused `pca_neighbors` path. When a GPU library is unavailable the route falls back to `cpu_csr` / `cpu_dense` with `fallback_reason="unsupported_input_layout"` (vs `no_cuda` when CUDA itself is absent, `no_rapids` when rapids-singlecell is absent, or `user_forced_cpu` for `device="cpu"`). Set `SCX_DISABLE_RAPIDS=1` to force the rapids-absent fallback for testing (CPU path with `no_rapids` reason); set `SCX_FORCE_NATIVE_GPU=1` to pin the surviving native GPU paths and bypass rapids routing.

**Fused device-resident route.** `pyscx.accel.pca_neighbors` runs PCA then kNN in one call. For an in-memory `X`, in-VRAM `device="gpu"` routes to the rapids-singlecell pipeline (`rapids_singlecell_gpu` on `pca` / `neighbors` / `pca_neighbors`). For backed/lazy `X` — or under `SCX_FORCE_NATIVE_GPU=1` — and when the modern cuSPARSE ABI + cuVS are available, the native path keeps the PCA embedding GPU-resident across the handoff into CAGRA and stamps `gpu_device_resident` on `uns["scx_accel"]["pca"]`, `["neighbors"]`, **and** `["pca_neighbors"]`. When neither GPU path can run it delegates to the standalone `pca` + `neighbors` — each records its usual route — and the `pca_neighbors` summary records the non-device-resident result. (The `pca_neighbors` fuzzy-graph step runs on the CPU either way.) `pyscx.accel.pca_neighbors_umap` no longer has a native device-resident UMAP path — the device fuzzy-simplicial-set and native UMAP SGD kernels were removed — so in-VRAM it runs the full rapids pipeline (`rapids_singlecell_gpu` on all four stages) and backed/lazy inputs fall back to sequential `pca` + `neighbors` + `umap` (the umap stage uses cuML then CPU; the summary mirrors the `umap` stage route).

**Preprocessing caveat.** `normalize_total` / `log1p` only reach the GPU shard-streaming kernel when `adata.X` is a backed (`ScxBackedSparseDataset`) or lazy (`ScxLazyTransformedDataset`) dataset; on a materialized scipy/dense `X` they record `cpu_csr` (with `fallback_reason="unsupported_input_layout"` on a GPU host) because the eager kernel needs a shard source — open via `pyscx.open(...).to_anndata(backed=True)` to engage GPU. `calculate_qc_metrics` and `pseudobulk_dex` are CPU-only; their route reflects only the gene-axis layout (`cpu_csc` under `prefer_format="csc"`, else `cpu_csr`). `harmony_integrate` records `gpu_dense` (its native cuBLAS + custom-kernel path) or `cpu_dense` on `uns["scx_accel"]["harmony_integrate"]`, with the usual `fallback_reason` (`none` / `no_cuda` / `user_forced_cpu`); its legacy backend string remains at `uns["harmony"]["backend"]` for back-compat.

**R (rscx).** The rscx accelerators stamp the **same record** — produced by the same `scx-accel` planners and serialized from the same key list (`AccelExecutionInfo::fields()`), so the keys above apply verbatim. Where it lands depends on the input: on a `Seurat` object it is written to `object@misc$scx_accel[[op]]` (mirroring `object@misc$pflog`); a matrix-form call returns it as the `scx_accel` element of the result list, or as the `"scx_accel"` attribute when the result is a `data.frame` / vector (`scx_rank_genes_groups`, `scx_pseudobulk_dex`, `scx_nb_glm`, `scx_highly_variable_genes`, `scx_score_genes`). rscx is CPU-only (no `device=` selector — see the ROADMAP R-parity list), so routes are `cpu_*` with `fallback_reason="user_forced_cpu"` — the pair a pyscx call with `device="cpu"` records — with op-specific notes matching pyscx semantics: `scx_pseudobulk_dex` stamps `cpu_nb_glm`/`"none"` (the NB-GLM is a first-class native CPU route, not a fallback), `scx_rank_genes_groups` honestly stamps `cpu_dense` because its kernel densifies, and `scx_harmony_integrate` stamps `harmony_integrate` (`cpu_dense`). The one deliberate gap is `scx_pseudobulk` (plain aggregation): pyscx stamps no `"pseudobulk"` op, and rscx does not invent keys Python never writes. `scx_pca` additionally returns `method` (`"covariance"`/`"randomized"`) — also stored as `@misc$scx_accel$pca_method` on the Seurat path — making the shared auto-routing rule's arm observable. Gated by the `accel_r_route` benchmark floors (`r_*_route_*_correct` in `thresholds.yaml`) — rscx has no CI lane, so that local gate is its automated dispatch signal.

### ScxBackedSparseDataset

PyO3 class for backed-mode lazy access to the main expression matrix (`adata.X`). Data stays on disk; only requested shards are decoded on access. Registered with `anndata.abc.CSRDataset`.

**Properties:**
- `shape` `→ (int, int)` — `(n_obs, n_vars)`, adjusted for deletion vectors and column projection
- `dtype` `→ numpy.dtype` — Always `float32`: the type every read decodes to (scipy CSR interop, anndata's `CSRDataset` expectations). The on-disk encoding is `stored_dtype`.
- `stored_dtype` `→ numpy.dtype` — The on-disk value encoding of this handle's shards: `uint8` / `uint16` / `uint32` for integer counts, `float32` / `float16` for continuous data. When shards mix (an `append` keeps each shard's own encoding) it is the widest — any float ⇒ `float32`, else the widest integer; a file with no shards reports `float32`. One 76-byte header read per shard, no decode, memoised on the reader, so `stored_dtype.kind in "ui"` answers "are these counts?" in O(shards). A layer handle reports the layer's own family; a lazy handle reports its *source*'s encoding (the transforms produce floats on read regardless).
- `cache_shards` `→ int` — The decoded-shard LRU size this handle was opened with (`to_anndata(backed=True, cache_shards=…)`, default 4). `0` is the **uncached** path — every read decodes afresh and retains nothing — and is the right setting when the streaming footprint must be one shard. Read-only: the count is fixed when a handle's reader is built, and one `to_anndata` call builds `X` and each layer's reader with the same count (there is no `cache_shards` on `pyscx.open`). Note the split: `to_anndata(cache_shards=0)` is legal and meaningful, while `IndexPlanDataset(cache_shards=0)` raises `RuntimeError` on purpose (the loader's prefetcher needs a cache to prefetch into).
- `format` `→ str` — Always `"csr"`
- `backend` `→ str` — Always `"scx"`
- `ndim` `→ int` — Always `2`
- `non_negative` `→ bool` — Whether the data is known to be non-negative (enables `(X > 0).sum() → getnnz()` short-circuit)
- `nnz` `→ int` — Total non-zero count
- `n_shards` `→ int` — Number of CSR shards in the backing file
- `__array__(dtype=None, copy=None)` — **Raises `TypeError`.** numpy's array protocol is implemented only to refuse: `np.asarray(adata.X)` on a handle would decode the whole `n_obs × n_vars` matrix at once, and before this it returned a 0-d object array that failed far away ("setting an array element with a sequence"). Call `to_memory()` (scipy CSR) or `toarray()` (dense) — both work on a column-projected handle such as `X[:, genes]`, which is itself a handle and refuses `np.asarray` the same way — take a row window with `handle[rows]` (a scipy CSR), or open the file with `pyscx.open(path).to_anndata()` for an in-memory AnnData. The dense `ScxBackedObsmDataset` keeps a materialising `__array__` — an embedding is small.

**Column projection:**
- `set_col_projection(col_indices)` — Restrict all access and aggregation to a subset of columns. Used internally by `to_anndata(var_names=...)` — on the backed path to project the handle, and on the eager path to assemble `X` and each layer already projected — and by streaming QC with gene subsets (`qc_vars`).

  > [!WARNING]
  > **This is a handle-level knob, not an axis subset.** It moves `X` only — `var`, `layers`, `varm` and `varp` are left at the old width, so the AnnData is inconsistent until you slice them yourself. Use `pyscx.accel.subset_var(adata, mask)` (or `adata[:, mask]`) for a real gene subset — see [Axis subsetting and aligned members](#axis-subsetting-and-aligned-members). Reach for this only when you want to reproject a bare handle.

**Slicing:**
- `__getitem__(row_slice)` `→ scipy.sparse.csr_matrix` — Decode requested shards, return scipy CSR.
- `__getitem__(rows, cols)` — With a non-`:` row selector: the row gather below, then a scipy column slice on the result (materialises the selected rows only). With `:` rows: the column projection below.
- **`handle[:, cols]` is a column projection, not a read.** Every column
  selector form — an `int` (`X[:, 5]` is an `(n_obs, 1)` handle, as scipy's is
  `(n, 1)`), a `list` / `range` / `tuple`, an integer ndarray of **any order**
  (signed negatives wrap once; unsigned is bounds-checked as `uint64`), a
  boolean mask of length `n_vars`, or a non-full `slice` (`X[:, 10:20]`,
  `X[:, ::-1]`) — is resolved once and composed through the handle's current
  window, and a new `ScxBackedSparseDataset` comes back with **no decode**: one
  decode per touched shard happens later, when that handle is read or
  aggregated. Ascending-unique selectors install a sorted projection; a
  reordered-but-unique selector rides on the presentation permutation
  (`sum(axis=0)`, `to_memory()` and every read honour the request order), and
  composing on top of an existing presentation (`X[:, [7, 2, 11]][:, [2, 0]]`
  → columns `[11, 7]`) is exact. Two forms are not handles: a selector with
  **repeated** columns (`X[:, [3, 1, 3]]`) — a permutation cannot express a
  repeat, so it materialises the *projected unique columns* (one `to_memory()`
  over 2 columns here, never the whole matrix) and gathers them with scipy —
  and the full `X[:, :]`, which returns the row CSR like `X[:]` (scipy's
  `X[:, :]` is a copy too). An out-of-range column, a wrong-length mask, a
  float or 2-D selector raise `IndexError` (numpy's rule; a float selector
  used to slip through and decode everything). Before 0.17 only an ascending
  int / bool ndarray projected; every other form decoded the whole matrix and
  sliced it.
- **`handle[rows]` is the bounded row gather.** A boolean mask or any 1-D
  integer array-like (unsorted, duplicates, negative indices wrapping once) is
  resolved in user-visible row space and gathered in request order by
  `BackedCsrReader::read_row_indices`: each touched shard is decoded once — a
  sparse request on a row-group-framed shard decodes only the touched row groups
  — and the result is assembled once into exact-size buffers, so peak memory is
  the result plus the shard cache, plus up to `cache_shards` shards decoding in
  flight while that cache fills (at most `2 × cache_shards` decoded shards
  beside the result), never a second copy of the result. `X[:]` and other
  contiguous slices on a handle **without** deletion vectors go through
  `read_rows`, which sizes the result exactly and, for a range larger than the
  shard cache, decodes uncached in parallel chunks of `cache_shards` on top of
  whatever the LRU already holds — `X[:]` costs what `to_memory()` costs. With
  deletion vectors, `X[:]` is the row gather over the kept rows (still one
  decode per shard and one assembly, but through the LRU rather than the
  uncached bulk path). An out-of-range row, and a boolean mask whose length is
  not the row count, raise `IndexError` (numpy's rule) rather than returning a
  shorter matrix. Deletion vectors and column projection compose with all of
  this. The same gather is available without an AnnData as
  `Experiment.gather_rows_sparse`.

**Aggregation (streaming, no materialization):**
- `sum(axis=0|1)` `→ numpy.ndarray` — Column or row sums via native Rust streaming.
- `mean(axis=0|1)` `→ numpy.ndarray` — Column or row means.
- `var(axis=0|1)` `→ numpy.ndarray` — Column or row variance (two-pass).
- `getnnz(axis=0|1)` `→ numpy.ndarray` — Non-zero counts per column or row.
- `max(axis=0|1)` `→ numpy.ndarray` — Column or row max.
- `min(axis=0|1)` `→ numpy.ndarray` — Column or row min.

**Materialization:**
- `to_memory()` `→ scipy.sparse.csr_matrix` — Decode all shards → full CSR (one exact-size assembly, `read_all`). On a handle with a **column projection** it assembles shard by shard instead: each shard is decoded once, projected (and row-filtered) while still one shard wide, and the narrow pieces are concatenated — peak is 2× the *projected* result plus one shard, never the whole matrix (it used to decode everything and project afterwards, so `X[:, [1, 3]].to_memory()` peaked at the full matrix). Sequential by design; parallel decode would hold every shard at once.
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
- `shard_boundaries()` `→ list[(int, int)]` — `(row_start, row_end)` pairs in user-visible row space, one per on-disk shard that still has visible rows. **Tiling contract** (relied on by `pyscx.iter_chunks(chunk_size="shard")`): `b[0][0] == 0`, `b[-1][1] == n_obs`, and `b[i][0] == b[i-1][1]` — the pairs tile `[0, n_obs)` exactly, with no gaps or overlap. With deletion vectors the counts exclude deleted rows, and a shard whose rows are all deleted is omitted, so `len(b)` may be less than `n_shards`.

### ScxBackedLayerDataset

PyO3 class for backed-mode layer access (e.g., `adata.layers["raw_counts"]`). Wraps a `ScxBackedSparseDataset` for a named layer. Registered with `anndata.abc.CSRDataset`.

- Same interface as `ScxBackedSparseDataset` (`shape`, `dtype`, `format`, `backend`, `ndim`, `__getitem__`, `to_memory`, `toarray`, `tocsr`, `tocsc`, `copy`, `sum`, `mean`, `var`, `getnnz`, `max`, `min`, `shard_boundaries`)
- `layer_name` `→ str` — Name of the backing layer
- `stored_dtype` `→ numpy.dtype` — The **layer's** on-disk encoding (its reader walks the layer's shard family, not X's), so `adata.layers["counts"].stored_dtype` can be `uint16` while a float `X` reports `float32`. `cache_shards` is this layer reader's setting — one `to_anndata` call builds `X` and each layer's own reader with the same count. `__array__` raises `TypeError`, as on `X`.
- `adata.layers[name][rows]` is the same bounded row gather as on `X` (one decode per touched shard, result assembled once, `IndexError` semantics as above) — the layer's own shard family is read, so gathering counts from a layer needs no round trip through the `Experiment`.
- `adata.layers[name][:, cols]` is the same column projection as on `X` and comes back as a `ScxBackedLayerDataset` (the wrapper, and with it `layer_name`, is kept — it used to return the bare inner class).

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
- `stored_dtype` `→ numpy.dtype` — The on-disk encoding of the **source** matrix, before the transforms (`normalize_total` / `log1p` produce floats on read regardless); `uint16` on a counts file.
- `cache_shards` `→ int` — The LRU size of the reader this handle shares with the `ScxBackedSparseDataset` it was derived from.
- `format` `→ str` — Always `"csr"`
- `ndim` `→ int` — Always `2`
- `backend` `→ str` — Always `"scx-lazy"` (distinguishes from `ScxBackedSparseDataset.backend` which is `"scx"`)
- `non_negative` `→ bool` — Whether the transformed data is non-negative
- `__array__(dtype=None, copy=None)` — Raises `TypeError`, as on `ScxBackedSparseDataset` (the transforms would have to run over the whole matrix to answer).

**Slicing:**
- `__getitem__(row_slice)` `→ scipy.sparse.csr_matrix` — Decode requested shards, apply all transforms in order, return scipy CSR.
- `__getitem__(:, cols)` — The same column selector forms as on `ScxBackedSparseDataset` (`int`, `list`, `range`, `slice`, any-order ndarray, bool mask; same `IndexError` rules). An **ascending-unique** selection (after composing through the current projection) is a projected `ScxLazyTransformedDataset` with no decode. This class stores its projection sorted and has no presentation permutation, so a **reordered or repeated** request (`X[:, [7, 2]]`, `X[:, [3, 1, 3]]`) materialises the projected *unique* columns — transforms applied, `to_memory()` over just those columns — and gathers them with scipy; it never decodes the whole matrix. (Reorder through `adata[:, idx]` on a lazy `X` still raises, because `var` and `X` must be sliced by one rule there.)
- `__getitem__(rows, cols)` — Row gather + transform, then a scipy column slice.
- A boolean mask or integer array-like is the same bounded row gather as on `ScxBackedSparseDataset` (one decode per touched shard, result assembled once in request order, `IndexError` for out-of-range rows or a wrong-length mask), with each output row's transform parameters looked up by its global row id — duplicates and unsorted requests included.

**Aggregation (streaming through transforms):**
- `sum(axis=0|1)` `→ numpy.ndarray` — Column or row sums of transformed data.
- `mean(axis=0|1)` `→ numpy.ndarray` — Column or row means of transformed data.
- `var(axis=0|1)` `→ numpy.ndarray` — Column or row variance. `axis=0` is two-pass streaming; `axis=1` and `axis=None` **materialize** the visible matrix via `to_memory()` and compute `E[X²] − E[X]²` in f32, so they are neither out-of-core nor clamped at zero (a near-constant row can return a small negative). Prefer the backed `X.var(axis=1)`, which streams in f64 and clamps.
- `getnnz(axis=0|1)` `→ numpy.ndarray` — Non-zero counts (unchanged by normalize/log1p).
- `max(axis=0|1)` `→ numpy.ndarray` — Column or row max of transformed data.
- `min(axis=0|1)` `→ numpy.ndarray` — Column or row min of transformed data.

**Materialization:**
- `to_memory()` `→ scipy.sparse.csr_matrix` — Decode all shards + apply transforms → full CSR. With a column projection it assembles shard by shard (decode → transforms → project → drop deleted rows, one shard at a time, then concatenate), so it peaks at 2× the projected result plus one shard rather than the whole transformed matrix.
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
- `shard_boundaries()` `→ list[(int, int)]` — Same tiling contract as `ScxBackedSparseDataset.shard_boundaries()` (user-visible row space; `b[0][0] == 0`, `b[-1][1] == n_obs`, contiguous). `pyscx.iter_chunks` uses it on a lazily transformed `X` too.

**Transform chain:**
- Backed data → `NormalizeTotal` → `Log1p` is fused into `ln(x × target_sum / row_sum + 1)` in a single pass.
- `repr()` shows the transform chain: `ScxLazyTransformedDataset(shape=(1000000, 33694), transforms=[NormalizeTotal, Log1p])`

### scVI Integration (`pyscx.scx_integrations.scvi`)

Pure-Python PyTorch Lightning DataModule wrapping `TrainingDataset` for scVI model training. Requires `lightning` or `pytorch_lightning`.

> **scVI needs raw counts.** `normalize`/`log1p` default to `True` (inherited from `TrainingDataset`), so batches are log-normalized by default. scVI and other count-likelihood models require raw integer counts — construct with `ScxDataModule(..., normalize=False, log1p=False)`.

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

> **Transforms are ON by default.** `normalize` **and** `log1p` both default to
> `True`, so batches are total-count normalized (`target_sum=1e4`) and
> `log1p`-transformed even though the file stores raw counts — the yielded `X` is
> log-normalized, **not** raw counts. Count-likelihood models (scVI, scANVI,
> count autoencoders, NB/ZINB decoders) need raw counts: pass
> `normalize=False, log1p=False`.

**Constructor kwargs**

| Argument | Default | Notes |
|---|---|---|
| `path` | — | Path to `.scx` file. |
| `batch_size` | `1024` | Mini-batch size. Auto-tuned downward if `max_memory_mb` is exceeded. |
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. Every index must be `< n_vars` — an out-of-range index is rejected at construction, because it matches no column and would otherwise yield an output feature that is silently always zero. **The panel is sorted and deduplicated**, so batch columns are in ascending gene-index order regardless of the order you pass, and duplicates shrink the batch width (check `n_output_genes`). Passing a panel that is not already ascending-unique emits a `UserWarning`; `np.unique(hvg_indices)` reproduces the column order the batches use. |
| `obs_columns` | `[]` | Obs metadata column names included in each batch. |
| `normalize` | `True` | Total-count normalize (fused with `log1p` in a single CSR row scan). **On by default** — set `normalize=False` (with `log1p=False`) for raw-count output. |
| `log1p` | `True` | Apply `log1p` after normalize. **On by default** — set `log1p=False` for raw-count output. |
| `target_sum` | `1e4` | Normalization target sum. |
| `pflog` | `False` | Apply PFlog (v4) / shifted-log normalization on raw counts (Booeshaghi et al.) instead of `normalize`/`log1p`. Mutually exclusive with them — when `True` it takes precedence and those flags are ignored. |
| `pflog_alpha` | `None` | PFlog NB overdispersion `α` (matrix-wide pseudocount `1/(4α)`). `None` estimates `α` once at loader construction from the raw counts (single-modality only); a float pins it. Only used when `pflog=True`. |
| `shard_group_size` | `8` | Shards per I/O group. Sequential I/O within each group for disk efficiency. |
| `prefetch_batches` | `4` | Ring buffer depth — number of pre-built batches to buffer ahead. |
| `seed` | `42` | RNG seed for reproducibility. Deterministic shuffle via `(seed, epoch)`. |
| `max_memory_mb` | adaptive (≥512) | Memory budget. **On every loader class it excludes the mmap'd file** — kernel page cache, evictable under pressure, reported (`mmap_mb`) but never budgeted. What each class *does* charge differs: this one costs the shard buffers and the batch ring, while `SparseCellSetDataset` costs only its shard cache (see its row for the two terms it leaves out). **When omitted**, the budget is *adaptive*: it scales up to fit the file's requested configuration (so a full-width ~33k-gene file keeps its requested `batch_size` instead of silently shrinking), floored at 512 MB and capped at 4096 MB. Pass an explicit value to pin a **hard ceiling** — then the pipeline auto-tunes `prefetch_batches` (to 2), then `shard_group_size` (to 1), then `batch_size` (halved, to 64), in that order. If it still does not fit, `budget_exceeded` is set and a `UserWarning` is raised. |
| `modality` | `None` | For multimodal v2 files: name of the modality to load (e.g. `"rna"`). Ignored on single-modality files. |

**Properties**

- `n_obs` → `int` — Total number of observations (cells) in the dataset.
- `n_vars` → `int` — Total number of variables (genes) in the dataset.
- `n_output_genes` → `int` — Genes per batch (HVG count if projection active, else `n_vars`).
- `effective_batch_size` → `int` — Actual batch size after memory budget auto-tuning.

**Methods**

- `close()` — Explicitly shut the pipeline down (join I/O + decode threads, release rayon pool). Idempotent, and **not** terminal: the next `__iter__` rebuilds and starts a fresh epoch. Recommended before process exit; see [Fork safety](#fork-safety-under-pytorch-dataloadernum_workers--0) and **Lifecycle — `close()` and `closed`** under [IndexPlanDataset](#indexplandataset).
- `memory_budget()` → `dict` — `breakdown` (the six-key per-component estimate every class that reports a budget uses) plus this class's own `shard_group_size`, `prefetch_batches`, `batch_size`, `estimated_mb`, `mmap_mb`, `budget_exceeded`. `mmap_mb` appears here and nowhere else, but it is **not** budgeted anywhere: `estimated_mb` equals `breakdown["total_bytes"]`. `budget_exceeded` means the auto-tune could not fit even at its minimums, and also raises a `UserWarning` at construction.

**Properties (lifecycle)**

- `closed` → `bool` — True from `close()` until the next `__iter__` rebuilds. Never raises.

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

> [!NOTE]
> The `seed` → ordering mapping changed in v0.13.1: both levels now chain
> `(seed, domain tag, epoch)` through SplitMix64 instead of adding them, which
> removes a collision where `(seed, epoch)` aliased `(seed + φ, epoch − 1)`.
> Determinism within a version is unchanged; a given seed produces a different
> ordering than it did before. See [docs/training.md](training.md#epoch-and-shuffling).

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
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. Every index must be `< n_vars` — an out-of-range index is rejected at construction, because it matches no column and would otherwise yield an output feature that is silently always zero. **The panel is sorted and deduplicated**, so batch columns are in ascending gene-index order regardless of the order you pass, and duplicates shrink the batch width (check `n_output_genes`). Passing a panel that is not already ascending-unique emits a `UserWarning`; `np.unique(hvg_indices)` reproduces the column order the batches use. |
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
- The loader keeps *up to* `lookahead` plans in flight at once: the head plan is
  decoding while shards for the next `lookahead - 1` are being warmed via
  `tokio::task::spawn_blocking` calls into `BackedCsrReader::read_shard_cached_arc`.
- **`lookahead` is a budget, not an obligation on the generator.** The loader
  waits for the first plan of a batch and fills the remaining slots only from
  plans the generator has already produced. A generator that yields plan *i+1*
  only after inspecting batch *i* — curriculum sampling, hard-negative mining —
  therefore makes progress rather than deadlocking; it simply runs
  un-prefetched. (Before v0.13.1 it hung: the generator waited for the batch,
  the loader waited for `lookahead` plans.)
- The plan-pull thread is eager and never waits for the consumer, so such a
  generator must **block on an explicit feedback signal** after `yield` — the
  loader will otherwise ask for the next plan before the current batch exists.
  See [training.md § Feedback and curriculum plan generators](training.md#feedback-and-curriculum-plan-generators)
  for the rendezvous.
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
print(ds.memory_budget()["max_blocking_threads"])
```

`max_blocking_threads` is the cap on simultaneously-running shard decodes. It
lives on the prefetch engine this class shares with `SparseCellSetDataset`, and
is sized from the **constructor** `lookahead` — which bounds in-flight *plans*,
not the blocking task a plan spawns per distinct `(file, shard)` it touches.

**Lifecycle — `close()` and `closed`**

All four dataset classes expose `close()` and a `closed` property, and all four
release the GIL around teardown, so tearing a dataset down cannot stall other
Python threads while an in-flight shard decode finishes. The bound is not one
number: the plan-driven pair drop a tokio runtime under a single 5 s deadline,
while the training pair bound-join two threads *sequentially* (`~2 ×
SHUTDOWN_DEADLINE`), and `MultimodalTrainingDataset` repeats that per modality.
Dropping the object does the same, so `close()` buys determinism rather than
correctness. `closed` and `repr()` never raise on any of them.

`close()` is **terminal on `IndexPlanDataset` and `SparseCellSetDataset`**:
their tokio runtime is built exactly once so that a forked child can never
inherit live tokio threads, which also means it cannot be rebuilt. Every other
method raises `RuntimeError` afterwards; construct a new dataset. On
`TrainingDataset` and `MultimodalTrainingDataset` it is not — their pool and
runtime are per-epoch anyway, so the next `__iter__` rebuilds them.

`closed` follows that split rather than papering over it: on the terminal pair
it means closed for good, on the re-usable pair it means torn down *right now*
and returns to `False` after the next `__iter__`. So `if not ds.closed:
ds.close()` is portable across all four; `assert ds.closed` after an epoch is
not.

A still-alive `IndexPlanBatchIter` / `SparseCellSetBatchIter` holds its own
reference, so `close()` releases only the dataset's. The iterator then does the
teardown itself — under the same 5 s bound and with the GIL detached — on
whichever comes first, its `Drop` or the end-of-stream branch of `__next__`. So
this ordering, which the API supports, is safe:

```python
it = ds.iter_with_plans(plans)
ds.close()      # releases the dataset's reference only
list(it)        # the iterator's own teardown, bounded and off-GIL
```

**Iterator metrics — `metrics()`**

Both `IndexPlanBatchIter` and `SparseCellSetBatchIter` expose `metrics()`,
returning `{"cache": {...}, "prefetch": {...}}`. The `cache` half is
loader-cumulative and identical to the dataset's `cache_metrics()`; the
`prefetch` half is per-iter and resets on each `iter_with_plans` call:

| key | meaning |
|---|---|
| `prefetch_tasks_spawned` | shards warmed into the LRU ahead of the gather |
| `prefetch_skipped_cache_hit` | already resident |
| `prefetch_skipped_in_flight` | a peer was already decoding it |
| `prefetch_skipped_block_index` | not warmed *whole*, so the gather takes the row-group path; its touched row groups are pre-decoded into the row-group LRU instead when the plan fits `budget / (lookahead + 1)` |
| `prefetch_skipped_reader_limit` | the plan touches more distinct files than `reader_limit` can keep resident, so it was not prefetched at all — the prefetcher would have had to pin the plan's whole width. Structurally 0 on `IndexPlanDataset` (single-file) and on any dataset that did not set `reader_limit`. The plan still gets a real admission verdict. |

Both handles are cloned when the iterator is built, so `metrics()` is safe to
call after the iterator has been drained.

`prefetch_skipped_block_index` counts the **L2 prefetch-time** decision: shards
not warmed whole so the gather could take the row-group path. The name predates
the row-group LRU and is a contract; such a shard is no longer left cold. The
engine takes **one row-group admission verdict per plan** — the row-group bytes
of every framed shard the plan touches, over every file, sized from the block
index with no decode, against `max_memory_mb`'s cache share
`budget / (lookahead + 1)` — and that verdict drives both sides: an admitted
plan's eligible shards are pre-decoded into the row-group half of the LRU and
the gather reports them as `cache_metrics()["row_group_hits"]`; an over-share
plan is not warmed, and retains only the groups **another plan of the lookahead
window also touches**, hottest-first and only while they fit the same share —
its cold tail still decodes and drops, so `row_group_misses` keeps growing while
`row_group_bytes_inserted` moves only by the reused groups.
`cache_metrics()["reuse_admissions"]` counts the plans that got such a partial
verdict, and `admitted_group_bytes` / `rejected_group_bytes` is the split it
decided. Before that, an over-share plan retained nothing at all —
`SCX_ROW_GROUP_ADMIT=plan` restores it as the same-build A/B arm. The bound on
the partial verdict is load-bearing, not belt-and-braces: a wide random plan can
touch nearly every row group a file has, at which point every key is "reused"
and an unbounded rule degenerates into admitting everything. It is a useful
confirmation that `scatter_block_index=True` had an effect — it stays 0 against
an unframed file. Both classes now warn at construction when the kwarg is set
and no shard is framed, so this is a confirmation rather than the only signal;
on `SparseCellSetDataset` the warning tests *any file in the set*, so a mixed
set is silent and these counters are still what tell you the route ran.

It is **not** the same signal as `cache_metrics()["block_index_groups"]`, and
the two can legitimately differ. `block_index_groups` is the route the *gather*
actually took; the prefetch counter only exists when prefetching runs. At
`lookahead=0` no prefetch runs at all, so every counter here is 0 while
`block_index_groups` is still positive — and with concurrent iterators sharing
one cache, a peer can warm a shard between the prefetch decision and the
gather. **Use `block_index_groups` as the route signal**; use these counters to
see what the prefetcher decided.

```python
ds = pyscx.IndexPlanDataset("atlas.scx")
try:
    for batch in ds.iter_with_plans(plans):
        ...
finally:
    ds.close()
assert ds.closed
```

### Fork safety under PyTorch `DataLoader(num_workers > 0)`

`pyscx.TrainingDataset`, `pyscx.IndexPlanDataset` and
`pyscx.SparseCellSetDataset` are fork-safe under
`torch.utils.data.DataLoader(num_workers > 0, start_method="fork")` —
the Linux PyTorch default — when the dataset is **constructed lazily
inside the worker's `__iter__`** (the pattern `cell-load-scx` and
`state-scx` already use). The per-pipeline tokio runtime (current-thread,
per-epoch), the per-pipeline `rayon::ThreadPool` (lazily built on first
iteration) and the process-wide `scx_loader::pool::cpu_pool()` used during
construction are all created inside the worker process, so a forked child
inherits no fork-hostile state from the parent. `cpu_pool()` is keyed on the
PID and rebuilt when it changes, which is what makes "created in the worker"
true even if the parent touched a loader path first. Construction therefore
does own worker threads — they are simply the child's own. See
[multithreading.md § Per-worker thread footprint](multithreading.md#per-worker-thread-footprint)
for how many.

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

### SparseCellSetDataset

Plan-driven **sparse** reader for role-tagged cell sets spanning several `.scx`
files — the shape perturbation screens and contrastive set-based models want.
Where `IndexPlanDataset` yields a dense paired batch from one file,
`SparseCellSetDataset` gathers flat CSR rows across files and delimits each set
with `set_offsets`.

**Constructor kwargs**

| Argument | Default | Notes |
|---|---|---|
| `paths` | — | List of `.scx` paths. `file_id` in a plan is the index into this list. |
| `cache_shards` | `None` → 128 | Shard-cache count cap. Size it with `suggested_cache_shards`, not by guessing — see [Sizing the shard cache](training.md#sizing-the-shard-cache). |
| `max_memory_mb` | `None` | Memory budget. Since ORG-9.10-5 the constant interpreter/numpy/Arrow overhead is subtracted before the cache is sized, so this is no longer a bare cache cap — but it is **not** a hard ceiling on the process either: the batch's per-row transients are never charged and the batch itself only when `max_plan_rows` declares how wide plans get (this path has no `max_plan_size`, so plan output size is otherwise caller-controlled), and one above-average shard can sit above the byte cap because the LRU keeps an oversize entry rather than refusing to cache it. `None` resolves adaptively (never unbounded). Both the count cap and the byte cap are enforced, and on large-shard files the byte cap binds first. |
| `remap_tables` / `n_global_genes` | `None` | Per-file `local → global` gene tables. Required for a cell **set** that spans files: raw-local indices from different files are not comparable. |
| `normalize` / `log1p` / `target_sum` | `None` → off | Off by default here, unlike the training classes. |
| `lookahead` | `None` → tuned | Prefetch depth; `0` disables prefetch. |
| `scatter_block_index` | `False` | Opt into the row-group block-index scattered route. Defaulted **off**, and there is no route that wins everywhere: full-shard beats block-index 117–167× on 100k-cell corpora but **loses 1.53×** on census_1m with grouped plans, so the two cross near 1M cells. The default serves the common case; pick per dataset. See [Cell-set scatter routes, re-measured after the row-group LRU](performance.md#cell-set-scatter-routes-re-measured-after-the-row-group-lru-phase-0-gate) and [sharding.md § Row-group framing & scattered reads](sharding.md#row-group-framing--scattered-reads). |
| `max_plan_rows` | `None` | Upper bound on rows per plan, **enforced**: a wider plan raises `RuntimeError`, because a cache sized for `N` rows while the gather accepts any width is not a bound. **Uncharged and unenforced by default** — the resolved cache is then exactly what it would be without the argument. Declared, it charges one gathered CSR batch — `presize_nnz(rows, mean_nnz_per_row) × 8` (the mean plus a ⅛ bias) + `(rows + 1) × 8` — against `max_memory_mb` before the cache is sized, reported as `breakdown["batch_buffer_bytes"]`, and on the configurations whose gather holds a second buffer (a remap, a downsample, or more than one file) the same figure again as `breakdown["transient_bytes"]`. The gather itself no longer allocates the biased figure — it sizes the batch exactly from an indptr prescan — so the ⅛ is headroom on an estimate rather than a model of the allocation. It bounds the plan's **row count**, not its bytes: the density is a manifest-wide mean, so a denser-than-average plan can exceed the charge. Opt-in because this class has no `max_plan_size`: plan width is the caller's, and a guessed default would shrink the shard cache on every existing dataset. |
| `reader_limit` | `None` | Cap on simultaneously-open readers. `None` opens every path and never closes one — today's behaviour, byte-identical in sizing, throughput and gather output. What it bounds is **resident memory, not file descriptors**: opening an SCX file mmaps it and closes the descriptor, and these readers do not watch their files, so a 5,000-file manifest constructs and gathers under `ulimit -n 1024` with the descriptor count flat. An open reader costs ~104 kB (tabula) to ~121 kB (census_1m) resident, over 90% of it the parsed catalog, so a 26k-file manifest is ~2.8-3.2 GB per process. A reopen re-parses that catalog (0.09-20 ms/file), which is exactly why the saving is real. Caps handles the loader may drop, not handles in existence — a plan needing more files at once exceeds it rather than blocking, and `cache_metrics()["reader_hwm"]` reports the truth. On the streaming path the figure to size against is roughly `files-per-plan x (lookahead + 1)`, since each in-flight plan leases every file it touches; a plan wider than the limit is **not prefetched** (counted by `metrics()["prefetch"]["prefetch_skipped_reader_limit"]` on the iterator, not by `cache_metrics()` — it is a per-iterator counter) and falls back to the one-file-at-a-time gather. Not charged to `max_memory_mb`. See [Very large manifests](training.md#very-large-manifests). |
| `downsample_target_library_size` / `downsample_method` / `downsample_seed` | `None` | Seeded per-row count downsample, applied before the batch leaves Rust. |

**Plans.** Each plan is a tuple `(file_ids, rows, role_tags, set_offsets)`:
`file_ids` and `rows` are parallel per-row arrays, `role_tags` labels each row
(perturbed / control / …), and `set_offsets` has length `n_sets + 1` and
delimits each set's row range in the flat batch.

**Properties**

- `n_files` → `int`, `n_cols` → `int` — reader count and CSR column count.
- `closed` → `bool` — True once `close()` has run. Terminal on this class; never raises.

**Methods**

- `iter_with_plans(plans, lookahead=None)` → `SparseCellSetBatchIter` — stream plans into §4.4 sparse batch dicts.
- `gather(file_ids, rows, role_tags, set_offsets)` → `dict` — gather **one** plan synchronously, returning the same batch dict the iterator yields (`ds.gather(*plan)`). Admission is decided once over the whole plan against the whole byte budget, where `iter_with_plans` takes one verdict per plan against its divided share (`budget / (lookahead + 1)`); that changes what the cache *retains*, not what is read, and the output is identical. Gathered on the calling thread, so it is not the way to drive an epoch. Raises exactly as the iterator does.
- `suggested_cache_shards(plan)` → `int` — distinct `(file_id, shard)` pairs one plan touches. Takes the same four-tuple `iter_with_plans` consumes; `role_tags` / `set_offsets` are ignored.
- `cache_metrics()` → `dict` — cumulative shard-cache counters, including `full_shard_groups` / `block_index_groups`, which report the scattered-read route the gathers actually took, and the `row_group_*` set (`hits`, `misses`, `evictions`, `bytes_inserted`, `duplicate_waiters`) for the decoded row groups a framed `scatter_block_index=True` gather retains. `hits` / `misses` / `evictions` / `bytes_inserted` keep meaning **whole-shard** entries; both kinds share one LRU and one byte budget, and `peak_bytes_in_cache` gauges both, so `peak <= bytes_inserted + row_group_bytes_inserted`. `SCX_ROW_GROUP_CACHE=0` disables row-group retention process-wide. Four further keys report the **reader registry** rather than the cache — `reader_opens`, `reader_evictions`, `reader_resident`, `reader_hwm`. They are absent from `IndexPlanDataset.cache_metrics()`, which is single-file and would report them as permanently zero. At the default `reader_limit=None`, `reader_opens` equals the manifest size and the other three never move. `reader_opens` counts the registry's opens — what it was handed, plus every reopen since — not the constructor's manifest scan, which opens every file once whatever the limit.
- `memory_budget()` → `dict` — `breakdown` plus `max_memory_mb`, `cache_shards`, `effective_cache_shards`, `shard_decoded_bytes`, `max_plan_rows`, `mean_nnz_per_row`, `max_blocking_threads`, `reader_limit`, `budget_exceeded`. `cache_bytes` and `python_overhead_bytes` are the non-zero terms by default; `max_plan_rows` adds `batch_buffer_bytes`, and `transient_bytes` beside it when the loader's configuration forces the gather to assemble the batch from a second buffer. `budget_exceeded` means even a one-shard cache does not fit. It is a statement about the **cache this loader sizes**, not a guarantee about process RSS — see the `max_memory_mb` row. `max_blocking_threads` is the cap on simultaneously-running shard decodes: `lookahead` bounds in-flight *plans*, not the tasks a plan spawns (one per distinct `(file, shard)` it touches). `reader_limit` is reported but deliberately **outside** `breakdown`: the breakdown is the byte model the shard cache is sized against, and an open reader's catalog is not one of its terms, so charging readers there would shrink the cache by something the tuner has never accounted for.
- `close()` — release the prefetch engine's tokio runtime, GIL detached, 5 s bound. Idempotent and **terminal** — see **Lifecycle — `close()` and `closed`** under [IndexPlanDataset](#indexplandataset).

### Tokenisation kernels (`pyscx.tokenize`)

Numeric kernels over a gathered CSR batch — the per-cell steps a transformer-class model's tokeniser is built from. Each takes the whole batch as `(indptr, indices, data)`, runs its row loop in Rust with the GIL released, and returns numpy arrays **moved** (not copied) out of Rust. Full semantics, parameter-by-parameter, and the divergences from each reference implementation: [docs/tokenize.md](tokenize.md).

`pyscx.tokenize.CONTRACT_VERSION` → `int` pins the kernels' numeric semantics, the parameters each requires, and the seed derivation for the RNG-driven ones. A module constant, not also a function. Assert it at setup. It does not cover array shapes or dict key names.

Gene ids must be non-negative and strictly ascending within each row — what a gathered batch always is, and **enforced** at these entries, since a `scipy` CSR is not index-sorted until `sort_indices()` and an unsorted or negative id otherwise reached a panic or a silently wrong mask.

| Kernel | Signature | Returns |
|---|---|---|
| `top_k` | `(indptr, indices, data, k, n_genes_total)` | `{ids, values, mask, pad}`, each `[n_rows * k]`, plus `n_rows` / `k`. Value descending, gene id ascending on ties; unfilled slots PAD; a row with no positive value gets one GENE_MASK token. The withheld-gene masking `collate_cellset_gathered` layers on this kernel is reachable only there. |
| `rank_tokens` | `(indptr, indices, data, gene_stats, l_max, vocabulary_version, target_sum=1e4)` | `{ids, lengths, norm_identity, n_rows, l_max}`. `gene_stats` is indexed by global gene id, every entry finite and strictly positive. Slots past a row's length are **undefined** — this kernel reports a length, the consumer owns padding. `norm_identity` is a blake3-64 stamp over the statistics and the vocabulary version; record it with the run. ⚠️ Ties break by gene id ascending; Geneformer's `np.argsort` default is unstable and follows no rule. |
| `bin_values` | `(indptr, indices, data, n_bins, edges=None, tie="left", seed=0, file_identity=0, rows=None)` | `{bins, n_rows, n_bins}`; `bins` is `[nnz]`, parallel to `data`. Zeros stay at bin 0. `edges=None` recomputes per-cell quantile edges (scGPT's `Preprocessor`); explicit `edges` must be exactly `n_bins - 1` finite non-decreasing values (checked before the all-zero early return) and become part of the tokeniser's identity. ⚠️ `tie` — `"left"` / `"right"` are `np.digitize`'s two deterministic bounds; `"seeded"` is scGPT's randomisation keyed on content instead of numpy's global RNG, which reproduces its distribution but never its draws. |
| `sample_genes` | `(indptr, indices, data, n, seed, file_identity, weight="log1p", rows=None)` | `{ids, lengths, n_rows, n}`. Draws **with replacement** (UCE's `replace=True`), weights `log1p(count)` renormalised. Pass real row ids in `rows` and the file's `downsample_file_identity`, or the draw is keyed on batch position and is reproducible only for that batch. |
| `transform_values` | `(indptr, data, mode, target_sum=1e4, pflog_alpha=None, n_measured=None)` | a new `data` array. `mode` is the collate kernel's: `pass_through` / `log1p_raw` / `normalize_log1p` / `pflog_raw`. `pflog_raw` is PFlog **v4** and requires both extra parameters. |
| `library_size` | `(indptr, data)` | `[n_rows]` `float64`. ⚠️ The sum of the row **as given** — on a panel-projected batch that is the library size after feature filtering, not the cell's sequencing depth. |
| `measured_mask` | `(indptr, indices, panel)` | `[n_rows * len(panel)]` `uint8`. "Measured", not "non-zero": a gene the row does not carry is absent from the CSR, which on a heterogeneous panel is a different claim from a zero count. |
| `gene_mask_id` / `pad_id` | `(n_genes_total)` | the two crop sentinels, `n` and `n + 1`. `top_k` rejects any gene id `>= n_genes_total` for exactly this reason. |

### Neighbourhood plans

Turn a stored `obsp` graph, or `obsm` coordinates, into the plans
`SparseCellSetDataset.gather` / `.iter_with_plans` already take. Each plan is
**one set**: the centre at position 0 with `role_tag` 0, then its neighbours
with `role_tag` 1. See [docs/training.md § Neighbourhood
plans](training.md#neighbourhood-plans) for the worked example.

| Function | Signature | Returns |
|---|---|---|
| `neighborhood_plans_from_graph` | `(path, key="connectivities", *, file_id, k=None, weight_order=None, include_center=True, drop_deleted=True, chunk_rows=65536)` | `(plans, centers)`. `plans` is a list of `(file_ids, rows, role_tags, set_offsets)` numpy tuples, one per surviving centre; `centers` is those centres' physical rows. `k` keeps `k` edges from the `weight_order` end (required with `k`), ties by column ascending — a rule this function declares because a stored graph carries none; `k=None` keeps every stored edge in column order. Non-finite weights sort last either way, so a distance graph's `inf` is never picked as a near neighbour. The graph is read one `chunk_rows` range at a time. |
| `neighborhood_plans_from_coords` | `(path, obsm_key="spatial", *, file_id, k=None, radius=None, include_center=True, drop_deleted=True)` | `(plans, centers)`, as above. Exactly one of `k` / `radius` is required; neither or both raises rather than resolving to one. A uniform grid is built at call time (O(n), no on-disk index) and searched ring by ring, so the answer is **exact**; ties order by squared distance then row ascending. **1-D, 2-D or 3-D only** — the search is exponential in the dimensionality, so a wide key raises rather than not returning. Integer and float64 columns narrow to float32. |
| `batch_plans` | `(plans, sets_per_batch, *, shuffle_seed=None)` | Single-set plans concatenated into batch plans. `shuffle_seed` reorders the **sets**, never a set's members. The last batch is short, not dropped. |

- `path` accepts a `str`, an `os.PathLike`, or an open `Experiment`.
- ⚠️ **`file_id` is a manifest position, not a file identity** — the index of
  this file in the `SparseCellSetDataset` the plans will be gathered with.
  Plans built from file A and fed to a dataset whose manifest puts A third
  gather the *first* file's rows, silently. Keyword-only and **required**: a
  default of `0` would make the documented hazard the quiet path.
- ⚠️ **`weight_order` is required whenever `k` is given, and has no default.**
  `"desc"` suits an affinity graph (`connectivities`: larger = closer); `"asc"`
  suits a distance graph (`distances`: larger = farther). A default of `"desc"`
  would be right for the default key and silently wrong the moment a caller
  changed only the key — returning each cell's `k` **farthest** neighbours —
  so the direction is stated rather than documented.
- ⚠️ **The two builders order a set's neighbours differently.**
  `_from_graph` emits them **column-ascending** — a stored graph carries no
  other order, and re-sorting by weight would make which edges `k` picked also
  change where they land. `_from_coords` emits them **nearest first**, the
  order it computed. Position 1 is therefore the nearest neighbour on the
  coordinate path and simply the lowest-numbered one on the graph path.
- With `k`, a deleted row is **not a candidate**, so the `k` best of the *live*
  neighbours are taken and the set is still `k` wide. Without `k`, a set is
  simply short by whatever is gone. Different answers, both intended.
- A `Float64` COO `data` column is narrowed to `f32` before ranking, so `k`
  over a float64 distance graph can order two very close weights differently
  from `to_anndata().obsp[key]`, which keeps them wide.
- Rows are **physical**. A deleted centre yields no set (so `centers` is
  shorter than `n_obs`); a deleted neighbour is never emitted. What that costs
  a set is the `k` / no-`k` split above. `drop_deleted=False` opts out.
- A NaN or infinite coordinate on a kept cell **raises**. That differs from the
  tokenisation kernels, which clip a NaN value to zero: zero is a meaningful
  expression level, and a NaN coordinate has no defensible grid cell.
- Sets never span files, so these plans never need `remap_tables`.
- `scx-accel`'s kNN is deliberately not used: it dispatches to approximate HNSW
  above 5,000 cells and has no radius mode.

## CLI (`scx`)

The CLI binary is named `scx` (built from the `scx-cli` crate via `cargo build -p scx-cli`).


### Core
- `scx convert <input> <output> [--from h5ad|10x|h5mu|scx] [--to h5ad|h5mu|scx] [--codec auto|none|scx1|zstd|lz4|pcodec] [--shard-size N] [--shard-obs off|auto|always] [--stream[=true|false]] [--csc off|always] [--csc-cols-per-shard N] [--modality NAME] [--memory-budget SIZE] [--strict-uns] [--dense-zero-epsilon F] [--temp-dir DIR] [--modalities CSV] [--modality-types NAME:TYPE,...] [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N] [--bitmap off|auto|always] [--reader-threads N] [--writer-queue-depth N]` — `--stream` bounds peak memory to one shard's worth of CSR plus encode buffers. It applies to h5ad ↔ SCX and h5mu ↔ SCX in both directions and to 10x → SCX, all of which stream by default; MTX ↔ SCX has a single materialising path, so omit the flag there (an explicit `--stream` on it is an error, and `--stream=false` is accepted as a no-op). On ingestion (h5ad/h5mu/10x → SCX), combine with `--csc always` for a two-pass CSR-then-`rebuild_csc_inplace` write (transient disk ~2× the output). On export (SCX → h5ad/h5mu), the streaming writer pre-allocates the `/X/{indptr,indices,data}` HDF5 triplet from catalog stats (or a single pre-scan when deletion vectors are active) so the on-disk layout is deterministic. Pass `--stream=false` to opt into the legacy materialising path on either side. For multimodal SCX → h5ad, combine `--to h5ad --modality NAME` to extract a single modality.

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
- `scx append <target> <source> [--codec auto|none|scx1|zstd|lz4|pcodec|shufdelta] [--shard-size N] [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N]` — Streaming append (reads source one shard at a time). Note: on `append` (like `merge`/`compact`/`from_h5mu`) `--codec auto` is the **legacy median heuristic** (Scx1/Zstd), not the adaptive default; the framed intent profiles (`fast`/`compact`/`compact-trial`) are not offered here — use `scx convert` / `pyscx.from_anndata` for those. `--index-*` rebuilds predicate indexes covering all rows post-append — see [Conversion-time predicate indexes and detection bitmaps](#conversion-time-predicate-indexes-and-detection-bitmaps).
- `scx delete <file> --filter <expr> [--dry-run]`
- `scx compact <input> <output> [--force] [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N]` — Rewrite reclaiming space; `--index-*` rebuilds the predicate index against the compacted output.
- `scx optimize <input> <output> [--force] [--codec {auto|fast|compact|scx1|shufdelta|compact-trial}] [--shard-obs {off|auto|always}] [--memory-budget SIZE]` — In-place upgrade (single-modality): re-encode + canonicalize every CSR shard and row-group-frame it (codec-agnostic random access via the row-group `BlockIndex`), stamping `format_version=4`, preserving row layout / obs / var / obsm / uns / indexes / deletion vectors. Default `auto` re-encodes adaptively (adopts ShufDeltaZstd where it wins by the margin); `--codec fast` forces the heuristic (Scx1 on low-median integer shards — keeps the GPU device-decode route, framed Scx1 shards decode in VRAM); `--codec scx1` forces Scx1 on every integer shard. `--shard-obs` (default `auto`) migrates a legacy single-section obs to the sharded layout — `auto` shards when `n_obs > shard_target_rows`, `always` unconditionally, `off` keeps the single section; an already-sharded obs is preserved regardless. Drops the CSC sidecar (rerun `scx build-csc`). `<output>` may equal `<input>` (atomic rename). `--memory-budget` caps what the parallel shard re-encode holds in flight (binary-prefixed size; decimal `KB`/`MB`/`GB` rejected) — **omitting it is not unbounded**: the default holds 1 GiB, so peak RSS does not scale with the machine's core count, and a budget below one shard's phase still encodes one shard at a time rather than refusing (unlike `scx convert --memory-budget`, which can tell you to lower `--shard-size`). See [docs/operations.md § `scx optimize --memory-budget`](operations.md#scx-optimize---memory-budget) and [docs/operations.md § Optimize](operations.md#optimize).
- `scx rollback <file> [--to-seq N]`
- `scx merge <file1> <file2> [<...>] --output <path> [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N]` — Merge multiple files; `--index-*` rebuilds the predicate index against the merged output (without it, pushdown regresses to a full obs scan on the merged file).
- `scx query <input> (--filter <expr> | <filter>) [--count] [--output <path>] [--select-genes <path>] [--normalize N] [--log1p] [--limit N] [--json]` — the obs predicate may be given via `--filter` (consistent with `scx subset` / `scx delete`) or positionally (back-compat); supply one form, not both. `<input>` accepts a local `.scx` file path, an exploded `.scxd/` directory, or a cloud URL (`gs://`, `s3://`, `az://`, `file://`). For cloud inputs the query is served via the `SectionReader` cloud path with no `scx pull` step. See [docs/cloud.md § Cloud-native query](cloud.md#cloud-native-query).
- `scx subset <input> [output] [--filter <expr>] [--genes <path>] [--modality NAME] [--dry-run] [--shard-size N] [--codec auto|none|scx1|zstd|lz4|pcodec] [--rebuild-csc] [--index-obs <csv>] [--index-var <csv>] [--index-preset {cellxgene,perturbseq,training}] [--index-auto-threshold N]` — Extract a subset of cells and/or genes into a new SCX file (`output` is optional with `--dry-run`). An input's predicate index cannot be carried over — row / column projection invalidates every shard range in it — so the `--index-*` flags **rebuild** it against the subset, mirroring `scx merge` / `scx compact` / `scx sort`. Without any of them the output has no predicate-index sections and `filter_obs` pushdown falls back to a full obs scan (a warning names the flags). A forced column missing from the subset's obs/var fails before anything is written, `--dry-run` included; the indexed columns are recorded in the `subset` provenance entry.
- `scx build-csc <input> <output> [--memory-limit 4G] [--force]` — Build CSC (column-major) shards from existing CSR data. `--memory-limit` accepts the same size forms as `--memory-budget` (see [Memory budgets](#memory-budgets)).
- `scx upgrade <input> [output] [--in-place]` — Upgrade an SCX file to the latest **unframed** format version (v3). Does not add row-group framing, so it does not reach v4 — use `scx optimize --row-group-rows N` for the framed v4 layout. A file that is already v3, **or already v4**, is left untouched (exit 0, message explaining why): rewriting a v4 file here would decode and re-emit every shard unframed, silently costing it the sub-shard random access v4 exists to provide, and `--in-place` is not rollback-able. Since `scx convert` / `pyscx.from_anndata` frame by default, most modern files are v4 and hit that branch. On a genuinely pre-v3 file it **canonicalizes** X and every layer before stamping v3 (that is what a v3 header asserts, so `nnz` can change), and obs and var pass through 1:1 with a sharded layout kept sharded. A CSC sidecar is preserved unless canonicalizing actually rewrote X, in which case it is dropped with a warning (rerun `scx build-csc`) rather than carried forward as a view of a different matrix. **Multimodal input is refused** — the rewrite is not modality-aware. It carries `varm`, `obsp`, `varp`, `.raw`, the group index, `obsm`, `uns`, layers, predicate indexes and deletion vectors; it does **not** carry layer CSC sidecars (rerun `scx build-csc`), and it drops detection bitmaps with a warning when canonicalizing actually rewrote X, since a bitmap records which genes a row *stores* and canonicalizing drops explicit zeros. It warns before writing whatever it will not carry. See [docs/operations.md § `scx upgrade` declines a file newer than its target](operations.md#scx-upgrade-declines-a-file-newer-than-its-target).

### External annotation import
- `scx cellbender-import <target.scx> <cellbender_out.h5> [--layer NAME] [--obs-key NAME] [--var-key NAME] [--prefix P] [--uns-key K] [--overwrite] [--on-missing-rows zero|error] [--on-extra-rows warn|error] [--gene-axis identical|reorder|subset] [--latent-embedding] [--dry-run]` — Attach a CellBender `remove-background` output as a layer, in place, joined by barcode. `--gene-axis` defaults to `identical`: a silent gene permutation is biologically wrong, so reordering must be opted into. Needs `--features hdf5`. See [docs/operations.md § CellBender import](operations.md#cellbender-import).
- `scx obs-import <target.scx> <table.csv> [--key CSV] [--source-key CSV] [--columns CSV] [--rename SRC=DST]... [--prefix P] [--keep-key-columns] [--delimiter C] [--status-column NAME] [--uns-key K] [--uns-key-from-source K]... [--overwrite] [--on-missing-rows null|zero|error] [--on-extra-rows warn|error] [--dry-run]` — Import a delimited annotation table (CSV/TSV) as obs columns, in place. Joins by key string, never by row position; uncovered target rows get `null`, not `0`. `--key a,b` is a composite key, and `--key obs_names` keys on the obs index. `--source-key` names the source side per component when the table spells the key differently (`--key sample_id,obs_names --source-key sample_id,barcode`), pairing positionally. Ungated — a delimited-table reader needs no libhdf5; an `.h5ad` source does (`--features hdf5`). `--dry-run` runs the join and a key diagnosis and writes nothing. Undo with `scx rollback`. See [docs/operations.md § External obs import](operations.md#external-obs-import).
- `scx var-import <target.scx> <genes.csv> [--key CSV] [--source-key CSV] [--columns CSV] [--rename SRC=DST]... [--prefix P] [--keep-key-columns] [--delimiter C] [--status-column NAME] [--uns-key K] [--uns-key-from-source K]... [--overwrite] [--on-missing-rows null|zero|error] [--on-extra-rows warn|error] [--dry-run]` — The var-axis twin of `obs-import`: import a delimited annotation table as var columns, in place. Joins by key string, never by row position; genes the table does not cover get `null`, not `0`. `--key var_names` keys on the var index; omitted, it auto-resolves through the var index then the gene-id spellings. A sharded var keeps its shard boundaries. Ungated; an `.h5ad` source needs `--features hdf5`. Refuses a multimodal file. `--dry-run` runs the join and a key diagnosis and writes nothing. Undo with `scx rollback`. See [docs/operations.md § External var import](operations.md#external-var-import).
- `scx doublet-import <target.scx> <table.csv> --tool {scdblfinder|scrublet|doubletfinder|doubletdetection|solo|scds|generic} [--key CSV] [--source-key CSV] [--key-added K] [--score-column NAME] [--call-column NAME] [--call-true TOK] [--call-false TOK] [--drop-native-columns] [--delimiter C] [--uns-key-from-source K]... [--overwrite] [--on-missing-rows null|zero|error] [--on-extra-rows warn|error] [--dry-run]` — The doublet wrapper over `obs-import`, normalising each tool's spellings onto `<K>_score` / `<K>_predicted` / `<K>_status` (+ `uns["<K>"]`). `--tool scds` emits no call column, so no `<K>_predicted` unless `--call-column` opts in; `--tool generic` requires `--score-column`. Use `--drop-native-columns` when the source is an h5ad exported from the target file.

### Cloud operations (`--features cloud`)
- `scx cloud-optimize <input> [--output <path>]`
- `scx explode <input> <output>`
- `scx pack <input> <output>`
- `scx pull <source-url> <dest> [--parallelism N] [--no-cloud-ready] [--filter <expr>]`
- `scx push <source> <dest-url> [--parallelism N]`

Note: the `scx-cli` crate has optional `hdf5` and `cloud` feature flags. HDF5 support is opt-in (`--features hdf5`). Cloud operations are opt-in (`--features cloud`). End-users building from a clone with `cargo install --path scx-cli` can pass `--features default-bin` to get an h5ad-capable build in one command (the crate is not on crates.io, so plain `cargo install scx-cli` does not work — prefer the pre-built binary from GitHub Releases).
