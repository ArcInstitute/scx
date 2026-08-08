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
                         shard_size. Mutually exclusive with
                         ObsMetadata (0) in the same file.
VarMetadataShard (25)  — Row-sharded var Arrow IPC (mirror of 24).
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

## ScxReader (`scx-format-io/src/reader.rs`)

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
- `obs_categorical(col)` / `obs_categorical_many(&[cols])` — `(codes: Vec<i32>, categories: Vec<String>)` for string/categorical obs columns, folded **one shard at a time** into a running global dictionary (never concatenates the column, unlike `read_obs_keys`). Null → `-1` (pandas convention), so a literal `"NaN"` string stays a real category; category order is first-seen. Accepts both `Dictionary(_, Utf8|LargeUtf8)` (as `from_anndata` writes) and plain `Utf8`/`LargeUtf8` (as `append` writes), including a file mixing both across shards. `_many` costs **one** projected read per shard for N columns. See [Obs categorical codes](performance.md#obs-categorical-codes-without-pandas-data-load-phase-1-1c).
- `debug_counts()` — `ReaderDebugCounts` with `AtomicU64` I/O counters (`cfg(debug_assertions)` only). `read_obs_shard_projected` counts column-scoped shard reads separately from `read_obs_shard`, so a test can assert the cheap path was *taken* rather than only that the materialising ones were avoided.
- `read_obs_predicate_index_bytes()` / `read_var_predicate_index_bytes()` — Predicate index raw bytes
- `read_deletion_vectors()` — Roaring Bitmap deletion vectors
- `read_all_csr_shards_filtered()` — Full matrix with deletion vector filtering
- `read_shard_header()` / `read_raw_shard_bytes()` / `read_shard_from_entry()` — Low-level shard access
- `mmap()` — Direct mmap access to the underlying file

## ScxWriter (`scx-format-io/src/writer.rs`)

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
| `SkippedUnsKey { key, reason }` | `read_uns` / `read_uns_entry` | `uns` entry was unrepresentable; skipped under default `strict_uns=false`. `strict_uns=true` turns this into an error on the first occurrence. |
| `FlattenedUnsDataframe { key }` | `read_uns_entry` | A `uns` pandas DataFrame (`encoding-type == "dataframe"`) was preserved as a nested dict (per-column values + `_index`) rather than reconstructed as a DataFrame; column order and per-column categorical dtypes are not restored. Data is not dropped. |
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
| `DroppedObsp { name, reason }` | h5ad ingest (obsp/varp routing) and `scx merge` (multimodal) | A pairwise `obsp`/`varp` matrix could not be preserved: on ingest, a CSC or otherwise unsupported pairwise layout is dropped (CSR is stored directly; a **dense** pairwise matrix is preserved as nonzero COO, not dropped); on `scx merge`, `obsp` axis semantics don't compose. Default-dropped with a warning. |
| `MappingPeakFootprintHigh { mapping, estimated_bytes, budget_bytes }` | `pyscx.from_anndata` | A single mapping's estimated in-memory footprint exceeds `memory_budget`. |
| `EagerAssemblyMemoryHigh { estimated_bytes, budget_bytes }` | `Experiment.to_anndata` | Estimated eager assembly footprint exceeds `memory_budget` (default 8 GiB). Warn-only, does not block. |
| `Hdf5NotThreadsafe` | Parallel streaming reader fallback | libhdf5 was not built thread-safe; parallel streaming fell back to the sequential coordinator. |
| `DroppedRaw { raw_n_vars }` | `Experiment.to_anndata` | The file carries an `adata.raw` matrix but the current reconstruction mode (obs-filtered query, backed mode, or deletion-vectors active) cannot reproduce raw's obs-axis filtering, so raw is omitted. The on-disk raw sections are preserved. |
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
| obs/var **ordered** categoricals | preserved | — | The `ordered` bit + category order round-trip both `h5ad → scx → h5ad` and `pyscx.open(...).to_anndata()` (carried in Arrow field metadata, re-applied to the reconstructed pandas factor). |
| obs/var **MultiIndex** | lossy | — | Only the single pandas `_index` is preserved; additional index levels are not carried. |
| unreadable obs/var column | dropped | `SkippedColumn` | Unsupported encoding-type or read error; column absent from output. |
| obsm / varm embeddings | preserved | `SkippedObsm` (on failure) | Dense embeddings round-trip; an unreadable embedding is dropped with the warning. |
| layers | preserved | `LayerSkipped` (on failure) | CSR layers round-trip; a layer whose shape disagrees with `/X` (or whose width exceeds `u32::MAX`) is dropped with the warning. |
| **dense** `obsp` / `varp` | lossy | — | Ingested as nonzero **float32 COO** (only nonzeros stored); both `to_anndata` and `to_h5ad` re-emit it as a **sparse** matrix (a dense input becomes sparse; values identical). |
| CSR `obsp` / `varp` | lossy | — | Round-trips `h5ad → scx → h5ad` (and via `to_anndata`) as **float32 CSR**; values downcast to `f32`. Under deletion vectors, `obsp` is filtered on both axes; `varp` (var axis) is never obs-deleted. |
| CSC / unsupported `obsp` / `varp` | dropped | `DroppedObsp` | CSC and other non-CSR/non-dense pairwise layouts are dropped on ingest. |
| `adata.raw` | preserved² | `DroppedRaw` / `DroppedRawOnWrite` (some modes) | Round-trips raw counts bit-exact with the wider var axis through **both** write doors — `h5ad → scx → h5ad` and in-memory `pyscx.from_anndata` / `pyscx.write`. ² Dropped, with a warning, under obs-filtered `to_anndata`, backed mode, and deletion-vector-active files, on the SCX-backed / lazy-`X` rewrite, and under reorder-on-convert (`--sort-by` / `--group-by`). See [`adata.raw`](#adataraw). |
| `adata.raw.varm` | dropped | `DroppedRawVarm` | Raw's own var-axis mappings have no section in the raw family (`raw/X` + `raw/var` only). `raw.X` and `raw.var` are unaffected. |
| `uns` scalars / 1-D & 2-D numeric arrays / nested dicts | preserved | — | Round-trip through the `uns` JSON representation. |
| `uns` pandas **DataFrame** | lossy | `FlattenedUnsDataframe` | Preserved as a nested dict (per-column values + `_index`); **not** reconstructed as a `pd.DataFrame` (column order / categorical dtypes not restored). |
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

Read the chain via `Experiment.provenance()` — each entry is a
dict with `params_json` as a raw JSON string (parse with
`json.loads`).

The full transform list is also surfaced on the wrapper itself via
`ScxLazyTransformedDataset.transforms_repr() → list[{"name","params"}]`,
which the writer uses to build the `lazy_transforms` payload.

## BackedCsrReader (`scx-format-io/src/backed.rs`)

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

## BackedCscReader (`scx-format-io/src/backed.rs`)

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
Replace metadata sections (`uns` / `obs` / `var` / `obsm` / `varm`) of an existing file **in place, without re-encoding `X`**. Appends only the replaced section bytes at EOF and atomically repoints the catalog — cost is O(replaced sections), the matrix shards are never read or rewritten. The CSC sidecar and `data_generation` are preserved (no `--rebuild-csc`). Any `None` field on `MetadataPatch` is left untouched. **A replaced `obs`/`var` keeps the predicate index it had**: the old section describes values that are gone, so it is rebuilt over the same columns (which re-derives the per-shard column stats, so pushdown survives the edit); `patch.index` names a different set instead. `n_obs` / `n_vars` are invariants — a shape mismatch is rejected before any write (`OpsError::ShapeMismatch`). Replace semantics, not merge. Multimodal (`modality_id != 0`) returns `OpsError::MultimodalUnsupported`. Advisory flock; rollback-able via the catalog chain.

The returned `ModifyMetadataSummary` carries the per-column build outcomes (in the same shape every other rewrite op reports them) plus:

- `obs_columns_not_carried` / `var_columns_not_carried` — the columns the file indexed that the new index does not, so a caller can surface the loss instead of leaving it silent. Render these through `obs_not_carried_unreported()` / `var_not_carried_unreported()`, which drop the columns a build outcome already names: the two channels overlap on the carry path, and a front end printing both raw warns twice about one column.
- `obs_carried_forward` / `var_carried_forward` — per axis, and **success-gated**: true only when that axis was carried *and* the rebuild produced an index. A carry whose every column turned out unindexable reports `false` here and `true` in `obs_predicate_index_dropped`, rather than both at once.
- `obs_predicate_index_dropped` / `var_predicate_index_dropped` — the file had an index on that axis and the output has none.

`set_uns` discards the summary: a `uns`-only patch touches no axis.

### `scx_ops::set_uns(path, uns: &serde_json::Value) → Result<()>`
Convenience wrapper over `modify_metadata` for the headline case — replace the whole `uns` block (O(uns bytes)).

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
- **Filtered-obs categorical semantics.** A `filter_obs(...).collect()`
  result's categorical (`Dictionary<_, Utf8>`) obs columns carry only the
  categories present in the surviving rows, not the full parent
  dictionary — standard AnnData/pandas behavior. Downstream code that
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
> For the most up-to-date function signatures and type annotations, see the
> [auto-generated Python API reference](python_api.rst) built from the Rust
> docstrings via autodoc.

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
  estimated footprint exceeds the budget. `shard_size` sets the per-shard
  row count for both `X` and the obs/var metadata shards. Obsm, varm,
  obsp, and varp are extracted and written one key at a time (incremental,
  not collected).
- `pyscx.from_h5ad(path, out, codec=None, shard_size=None, csc=None, csc_cols_per_shard=5000, uns_format="tagged", stream=True, strict_uns=False, dense_zero_epsilon=0.0, memory_budget=None, temp_dir=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4, sort_by=None, reverse=False, group_by=None, reference=None, group_target_bytes=None, group_max_bytes=None, group_pass="auto", obs_override=None, var_override=None, uns_override=None, row_group_rows=256, row_group_target_nnz=None)` — Stream an h5ad file directly to SCX without materialising `X` in Python or Rust. `csc=None` resolves to `"off"` unless `index_preset` implies `"auto"`.
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
- `pyscx.from_h5mu(path, out, codec=None, shard_size=None, csc=None, csc_cols_per_shard=5000, stream=True, strict_uns=False, memory_budget=None, temp_dir=None, modalities=None, modality_types=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", reader_threads=None, writer_queue_depth=4, row_group_rows=256)` — Stream an h5mu file to a multimodal SCX v2 file. `csc=None` resolves to `"off"` unless `index_preset` implies `"auto"`. Mirrors `from_h5ad` for h5mu inputs; per-modality `n_vars`/`nnz` come from `/mod/{name}/X` attributes so there is no pre-pass materialisation. `reader_threads`/`writer_queue_depth` carry the same semantics as `from_h5ad` — each modality runs through the same dispatcher independently.
  - `modalities`: optional list of modality names to keep
    (case-sensitive). Unknown names raise `ValueError` with the
    available list.
  - `modality_types`: optional dict `{name: "rna" | "protein" | "atac"
    | "spatial" | "methylation" | "custom"}`. Modalities not listed
    fall back to name inference and emit `ModalityTypeInferred`.
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None, csc="off", csc_cols_per_shard=5000, uns_format="tagged", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000, bitmap="off", memory_budget=None, force_legacy_metadata=False, row_group_rows=256, row_group_target_nnz=None)` — 10x HDF5 to SCX.
- `pyscx.from_mtx(mtx_dir, scx_path, codec=None, shard_size=None)` — Cell Ranger MTX directory (`matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, `features.tsv[.gz]`) to SCX. Default shard size is 16384.
- `pyscx.to_mtx(scx_path, output_dir)` — SCX to Cell Ranger–style MTX directory (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`).
- `pyscx.cellbender_import(path, cellbender_h5, *, layer="cellbender", obs_key=None, var_key=None, prefix="cellbender_", uns_key="cellbender", overwrite=False, on_missing_rows="zero", on_extra_rows="warn", gene_axis="identical", latent_embedding=False, dry_run=False)` —
  Attach a CellBender `remove-background` output to an existing SCX file as a
  layer, **in place**, joined by barcode. Returns a summary dict; inspect
  `n_matched` (or run with `dry_run=True`) before trusting the result. See
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
  both sides use the `key` names. `on_missing_rows`
  accepts `"null"` (the default), `"zero"` or `"error"`; `"null"` and `"zero"`
  are the **same policy** (leave the row NULL) — `zero` is the spelling the
  shared `MissingRowPolicy` enum carries from the CellBender importer, where a
  missing *matrix* row genuinely is zeros, so `cellbender_import` still spells
  its default `"zero"` and rejects `"null"`.
  **`overwrite` replaces, it does not merge** — importing several
  per-batch tables in turn keeps only the last. Returns a summary dict
  (`n_obs`, `n_matched`, `n_target_rows_absent`, `n_source_rows_absent`,
  `obs_key_column`, `obs_columns_added`, `obs_index_dropped`, and a
  `key_diagnosis` on failure or `dry_run`); inspect `n_matched`, or run with
  `dry_run=True`, before trusting the result. Undone by `pyscx.rollback`. See
  [docs/operations.md § External obs import](operations.md#external-obs-import).
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
  a file target the write goes through `modify_metadata`, so the file's obs
  predicate index is **carried forward** — `index_obs` / `index_preset` are for
  changing the indexed column set, not for restoring it. Pure Python — nothing
  in it knows what a doublet is.
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
- `pyscx.to_h5ad(path, out, stream=True, modality=None, reader_threads=None, writer_queue_depth=4, memory_budget=None, obs_mask=None, min_counts=None)` — Stream SCX → h5ad
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
- `pyscx.preprocess(source, target, ops, target_sum=None)` — Streaming shard-by-shard preprocessing. Preserves obs/var/obsm/uns only; **rejects multimodal and `adata.raw`-bearing inputs** (would otherwise corrupt X / drop raw) — extract a single modality first, or transform before attaching raw.
- `pyscx.save_layer(source, target, layer_name, ops, target_sum=None)` — Save transformed data as a layer. Preserves obs/var/original-X/obsm/uns only (pre-existing layers, raw, CSC, varm/obsp/varp, indexes, bitmaps, deletion vectors are not carried over); same multimodal/raw rejection as `preprocess`.

### File operations
- `pyscx.append(target, input, codec=None, shard_size=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, modality=None)` — Streaming append from SCX file (reads one shard at a time; raw-copy fast path when codec/encoding match). **Append into a multimodal target is deferred** — it raises `ValueError` (`MultimodalUnsupported`); a single-modality append would leave sibling modalities under-covering the shared obs axis. Extract a modality with `scx subset --modality`, append to that single-modality file, then re-merge. Single-modality append is unaffected. `index_*` kwargs rebuild predicate indexes covering all rows post-append — see [Conversion-time predicate indexes and detection bitmaps](#conversion-time-predicate-indexes-and-detection-bitmaps). Appending categorical `obs`/`var` onto a **row-sharded** base reassembles the existing metadata through the shared canonical assembler (`scx_format_io::assemble_sharded_metadata`), so disjoint/duplicate per-shard categorical vocabularies and a mixed `Dictionary`/plain-string shard layout are reconciled rather than failing to read back. A genuinely corrupt/gapped obs/var shard cover (non-contiguous `row_start` stamps) is now rejected with a clear error instead of being silently mis-assembled.
- `pyscx.append_from_anndata(target, adata, codec=None, shard_size=None, in_place=False, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, modality=None)` — Append from AnnData. Same `index_*` semantics as `append`. **Append into a multimodal target is deferred** and raises `ValueError` (`MultimodalUnsupported`) — extract → append → re-merge. Single-modality append is unaffected.
- `pyscx.mark_deleted(path, cell_indices)` — Logical deletion
- `pyscx.compact(input, output, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, reshape_obs=False)` — Rewrite reclaiming space. `index_*` kwargs rebuild predicate indexes against the compacted output. `reshape_obs=True` migrates legacy single-section obs metadata to the sharded `ObsMetadataShard` layout (mirrors `scx compact --reshape-obs`; useful after a backed `from_anndata` conversion).
- `pyscx.optimize(input, output, codec="auto", shard_obs="auto")` — Re-encode + canonicalize every CSR shard (X / layers / obsp graphs) and row-group-frame it (codec-agnostic random access via the row-group `BlockIndex`), stamping `format_version=4` — the Python equivalent of `scx optimize`. Single-modality files only (multimodal → `RuntimeError`; use `compact`). `codec="auto"` (default) keeps the per-shard codec choice; `codec="scx1"` forces Scx1 on every integer shard (full `to_gpu_anndata` device-decode coverage — framed Scx1 shards decode in VRAM); any other value → `ValueError`. `shard_obs` (`"off"|"auto"|"always"`, default `"auto"`) migrates a legacy single-section obs to the sharded `ObsMetadataShard` layout — `"auto"` shards when `n_obs > shard_target_rows` (the `from_anndata` threshold), `"always"` unconditionally, `"off"` keeps the single section; an already-sharded obs is preserved regardless; an invalid value → `ValueError`. No `force` kwarg — pass `output == input` for an in-place upgrade (atomic rename) or remove the target first. Drops the CSC sidecar (rerun `build_csc`). See [docs/operations.md § Optimize](operations.md#optimize).
- `pyscx.rollback(path, to_seq=None)` — Revert to previous manifest
- `pyscx.set_uns(path, uns)` — Replace the whole `uns` block in place, **without re-encoding `X`** (cost O(uns bytes)). Replace semantics, not merge. The CSC sidecar and `data_generation` are preserved. Rollback-able via `pyscx.rollback`. **`set_uns` is a strict subset of `modify_metadata`** — `pyscx.modify_metadata(path, uns=...)` does the same thing and also reaches `obs`/`var`/`obsm`/`varm`; prefer `modify_metadata` unless you only need the one-arg `uns` convenience.
- `pyscx.modify_metadata(path, *, uns=None, obs=None, var=None, obsm=None, varm=None, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, modality=None)` — Replace metadata sections (`uns` / `obs` / `var` / `obsm` / `varm`) in place without touching `X`. `obs`/`var` accept a pandas `DataFrame` (or pyarrow `Table`) and must match `n_obs` / `n_vars` (wrong shape → `ValueError`); `obsm`/`varm` accept `dict[str, np.ndarray]`. Any omitted arg is left untouched. **A replaced `obs`/`var` keeps the predicate index it had** — rebuilt over the same columns, which also re-derives the per-shard column stats, so `filter_obs` pushdown survives an ordinary obs edit. `index_*` kwargs name a different column set instead, **per axis** (`index_obs` does not change what a same-call var replacement carries; `index_preset` / `index_auto_threshold` span both); a column the file indexed and the new set omits raises a `UserWarning` rather than vanishing silently. Replace semantics, not merge; for a shallow `uns` merge, read-modify-write (`adata = pyscx.open(path).to_anndata(); adata.uns[...] = ...; pyscx.set_uns(path, dict(adata.uns))`). Only the global modality is supported today (`modality != 0` → error).
- `pyscx.sort(input, output, by, reverse=False, shard_size=None, codec="auto", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, memory_budget=None, temp_dir=None, bitmap="off", rebuild_csc=False, csc_cols_per_shard=5000, csc_memory_limit="4G", group_by=None, reference=None, group_target_bytes=None, group_max_bytes=None, group_write_block_bytes=None)` — Globally reorder cells by an obs key for X-read locality and contiguous predicate-index shard ranges. `codec` (`auto`/`none`/`scx1`/`zstd`/`lz4`/`pcodec`/`shufdelta`) pins the output encoding — without it the writer re-selects per shard, so a reorder can change file size for reasons unrelated to the reorder. Pass `memory_budget` (e.g. `"4G"`) to force the bounded external partition sort. Drops the CSC sidecar and detection bitmap (`rebuild_csc=True` / `bitmap=` to re-emit); `adata.raw` is not preserved; deletions are materialized away. See [sharding.md § Sorting for read locality](sharding.md#sorting-for-read-locality-scx-sort).
- `pyscx.shuffle(input, output, seed=42, shard_size=None, codec="auto", index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, memory_budget=None, temp_dir=None, bitmap="off", rebuild_csc=False, csc_cols_per_shard=5000, csc_memory_limit="4G")` — Globally reorder cells by a **seeded random permutation** (`scx sort --shuffle`), so a training loader gets i.i.d. batches at any `shard_group_size`. Same engine and same drop semantics as `sort`, minus the key arguments — shuffle is an order *source*, not a modifier, so `by` / `reverse` / `group_by` are not offered. `seed` is recorded in provenance and is the only record of the permutation; the same seed on the same input always reproduces the same file. The permutation runs over **live** rows, so a file with deletion vectors shuffles differently from the same file without them. Two consequences worth knowing: the output is the inverse of a sorted file for queries (it maximally scatters predicate-index shard ranges), and a permutation inherently costs some cross-row redundancy for codecs whose compression spans rows — ~6–12% for `zstd`, under 1% for `lz4`/`shufdelta`. **Leave `codec="auto"`**: it runs the same adaptive per-shard selection `scx convert` does. (Earlier releases told you to pin the input's own codec because `auto` grew X 1.86–2.09×. That was a derived-file bug, not a property of shuffling, and is fixed — pinning now selects a specific encoding, it does not hold size. See [sharding.md § Shuffling for training](sharding.md#shuffling-for-training-scx-sort---shuffle).) The provenance entry's `action` stays `"sort"` — the seed lives at `params.shuffle.seed`, so a consumer looking for `action == "shuffle"` will not find it. See [sharding.md § Shuffling for training](sharding.md#shuffling-for-training-scx-sort---shuffle).
- `pyscx.merge(inputs, output, index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None, assume_identical_var=False, assume_identical_obs=False, uns_policy=None, sort_by=None, reverse=False)` — Merge multiple files. `index_*` kwargs rebuild predicate indexes against the merged output — without them, pushdown silently regresses to a full obs scan on the merged file. `assume_identical_var` (default `False`) validates var identity (index, column names, values) across all inputs; set `True` to check only `n_vars` (breaking change from pre-branch where var was unchecked). `assume_identical_obs` (default `False`) validates each input's obs schema (column names + dtypes) against input 0; set `True` to skip. `uns_policy` controls conflicting uns sections: `None` / `"first"` (keep first input), `"require-equal"` (error on difference), `"namespace"` (prefix keys with input filename), `"summary"` (keep first input and record a `_scx_uns_conflicts` array). `sort_by` / `reverse` optionally sort the merged obs by a column. Merge streams obs shard-by-shard and builds predicate indexes incrementally from the shard stream.

### Cloud operations (requires `--features cloud`)
- `pyscx.pull(source, dest, filter=None, parallelism=None)` — Streaming cloud → local
- `pyscx.push(source, dest, parallelism=None)` — Streaming local → cloud
- `pyscx.cloud_optimize(input, output=None)` — Front-of-file catalog
- `pyscx.explode(input, output)` — Packed → exploded directory
- `pyscx.pack(input, output)` — Exploded → packed
- `pyscx.open_cloud(url) -> CloudExperiment` — Direct cloud reads.
  `CloudExperiment.query(modality=…)` scopes to one modality of a multimodal
  file (same semantics as the local `Experiment.query`).
- `pyscx.read_cloud(url, *, obs_filter=None, var_names=None, modality=None) -> AnnData` — One-call cloud read (= `open_cloud(url).query(modality=…)…collect().to_anndata()`); see [docs/cloud.md § `pyscx.read_cloud(...)`](cloud.md#pyscxread_cloud--one-liner-cloud-read).

### Experiment

The Python-visible class is `Experiment` (the Rust type is `PyExperiment`).
`repr(exp)` is AnnData-style — a `Experiment object with n_obs × n_vars = …`
header followed by indented `obs:` / `var:` / `uns:` / `obsm:` / `varm:` /
`layers:` key lists. On-disk codec / shard / format-version internals moved off
the repr onto `Experiment.info() -> str`.

- `read_obs(columns=None)` / `read_var(columns=None, *, modality=None)` — Read the cell / gene metadata table as a pandas DataFrame **without touching `X`**. Reach for these instead of `to_anndata().obs` / `.var` when you only want the metadata: on a 500k × 61k Census file `read_var()` is ~0.05 s against ~1.4 s and ~10 GB peak for the full materialisation. Both always retain the pandas index column (barcodes / gene names), so a projected frame indexes the same as an unprojected one, and an unknown column name raises `KeyError` listing what is available. Boolean columns come back as the pandas nullable `boolean` dtype whether or not they contain nulls, so the dtype follows the schema rather than the data and agrees with what a `to_h5ad` round trip returns. On a column with nulls that means `.astype(bool)` raises — deliberately, since coercing "not covered" to `False` is the mistake nullability exists to prevent — and `.fillna(False)` is the explicit form.
  - `read_obs`'s `columns` is a genuine **pushdown** — unselected columns are never materialised, which matters because `obs` scales with `n_obs`.
  - `read_var`'s `columns` is a convenience projection applied **after** the decode. `var` is one section sized by `n_vars` (5.5 MB for 61k genes), so there is nothing to save at the I/O layer; it does not read less off disk.
  - `read_var(modality=…)` selects one modality's gene axis on a multimodal file. Omitting it reads the global / single-modality `var`, which on a multimodal file is usually not what you want. Unknown name → `KeyError`.
  - For enumerating one categorical obs column, prefer `distinct_values()` / `obs_categorical()`, which scan per-shard dictionaries instead of assembling the table.
  - `CloudExperiment` mirrors both. There, `read_obs(columns=…)` *is* a genuine network pushdown (per-column projected range reads); `read_var(columns=…)` is still post-fetch, for the same reason as locally.
- `to_anndata(backed=False, cache_shards=4, var_names=None, obs_filter=None, layers=None, preserve_slots=False, modality=None, eager=False, memory_budget=None, obsm=None, preserve_var_order=False, strict_var_names=True, container="csr", data_dtype=None, index_dtype=None, allow_lossy=False)` — Convert to AnnData
  - `var_names`: list of gene names to project (column subset). A set selector by default (sorted original-column order, duplicates collapsed)
  - `preserve_var_order`: when True, return the gene axis in the order `var_names` was listed (first-occurrence-wins dedup) instead of sorted order. Works on eager / backed / GPU / query-engine paths. The streaming accelerators decode columns in sorted on-disk order and so cannot express a request-ordered gene axis; on a *backed* `X` they refuse rather than misalign the result against `adata.var` — `highly_variable_genes`, `normalize_total`, `log1p`, `calculate_qc_metrics`, `score_genes`, `pflog`, `pca`, `pca_neighbors`, `pca_neighbors_umap`, `rank_genes_groups`, `pdex_ref`, `pseudobulk_means`, `pseudobulk_dex` and `pdex_nb_glm` raise `RuntimeError`. Run them before projecting by name, or re-open without `preserve_var_order`
  - `strict_var_names`: when True (default), any name absent from the var metadata raises `KeyError`. Pass False to silently drop unknown names (pre-0.8.6 behaviour)
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
      `data_dtype=` lowers peak RSS. (The `obs_filter` query path and eager
      `layers` still cast post-assembly.)
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
    CSR** — scipy upcasts `int16 → int32` (same as `to_anndata`, below).
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
- `to_gpu_anndata()` — Minimal-copy on-device handoff: decodes shards, transfers to the GPU, and returns a GPU-resident AnnData whose `X` is a `cupyx.scipy.sparse.csr_matrix`. The returned object is suitable for direct use with rapids-singlecell (`rsc.pp.*`, `rsc.tl.*`) without additional host↔device copies. Requires `cupy` and a CUDA-capable GPU. Records its `transfer_mode` and real `bytes_uploaded` on `uns["scx_accel"]["to_gpu_anndata"]` — `scx_device_decode_gpu` (Scx1 sidecar shards decoded fully in VRAM, including dense ≥128-nnz rows via the BitPacker4x kernel; only indptr uploaded), `scx_device_handoff_streamed` (some shard host-bounced because it is not an Scx1 sidecar shard — a non-Scx1 codec or a sidecar-less Scx1 shard), or `scx_device_handoff` (host-assembled filtered/projected/multimodal input). See **Accelerator route metadata** below.

  **Memory semantics.** The result is the **complete** sparse matrix in VRAM — this is not a streaming/partial representation. Sparse CSR format is preserved throughout (VRAM scales with NNZ, not N×M). Shards are decoded one at a time into pre-allocated combined device buffers; peak device memory during transfer is the combined buffer plus one shard. A **VRAM pre-flight check** (1.2× headroom factor) compares the required bytes against free device memory and raises `ValueError` if insufficient, with an actionable message pointing to `backed=True` streaming workflows. See [gpu-setup.md § GPU memory model](gpu-setup.md#gpu-memory-model) for sizing formulas.

  `to_gpu_anndata` accepts `container` / `data_dtype` / `index_dtype` / `allow_lossy` for signature parity with `to_anndata`, but the device path is **f32-native**: a non-default request raises `ValueError`. To obtain a narrow/dense matrix, materialize it on the host with `to_anndata(container=..., data_dtype=...)`.
- `info() -> str` — One-line codec / shard / format-version internals (kept off the AnnData-style `repr`).
- Properties: `n_obs`, `n_vars`, `nnz`, `shard_count`, `format_version`, `codec_id`, `index_dtype`, `path`, `has_csc`, `has_deletions`.
- List-returning accessors — callable **methods** (not properties): `layer_names()`, and the AnnData-style key accessors `obs_keys()`, `var_keys()`, `obsm_keys()`, `varm_keys()`, `uns_keys()` (all cheap — schema/catalog reads, no matrix decode; `obs_keys()`/`var_keys()` exclude the pandas index column).

### Container and dtype materialization

The read APIs materialize `X` (and layers) as a scipy `csr_matrix` of
`(int64 indptr, int32 indices, float32 data)` by default. The `container`,
`data_dtype`, `index_dtype`, and `allow_lossy` kwargs let a reader choose the
output container and numeric dtype directly, so downstream consumers
(sklearn / PyTorch / scVI, GPU batches) don't over-allocate or re-densify.

Surfaced on `Experiment.to_anndata`, `PyQueryResult.to_anndata`, and
`PyQueryResult.to_csr` (`Experiment.to_gpu_anndata` accepts them for parity but
rejects any non-default request — the device path is f32-native).

| kwarg | values | default | notes |
|-------|--------|---------|-------|
| `container` | `"csr"` \| `"dense"` | `"csr"` | `"dense"` returns a row-major `numpy.ndarray` (no scipy CSR) |
| `data_dtype` | `float16/32/64`, `int8/16/32/64`, `uint8/16/32` | `None` → `float32` | numeric dtype of the values |
| `index_dtype` | `int16` \| `int32` \| `int64` | `None` → `int32` | CSR column-index dtype; ignored (warns) for `"dense"`. **Note:** scipy `csr_matrix` does not support 16-bit indices, so `index_dtype="int16"` is upcast back to `int32` by scipy on construction — the narrow int16 buffer is built (and range-gated: a column index ≥ 32768 fails loud) but does not persist in the returned CSR |
| `allow_lossy` | `bool` | `False` | fail-loud cast gate — see below |

**Zero-copy default preserved.** `container="csr"` with no dtype kwargs takes the
exact pre-existing path: the decoded `Vec`s are moved into numpy with `copy=False`
and no cast. This is guaranteed byte-identical and is the performance-sensitive
common case.

**In-decode narrow (eager `X` / `raw`).** A non-default request on the eager
`to_anndata` path narrows **in-decode**: each shard is decoded to its native
stream (integer counts as `u32`, floats as `f32`) and cast straight into a
full-matrix buffer *of the target dtype*, so the intermediate f32 CSR is never
allocated. A narrow `data_dtype="uint16"` read therefore **lowers** peak RSS
(2 B/nnz for the value buffer, not 4 B/nnz + a narrow copy) — see
[performance.md § Full read → AnnData](performance.md#full-read--anndata-wall-s--peak-rss-mb). Two
paths still cast post-assembly (correct, no RSS win): the `obs_filter` **query**
path (it assembles f32 after predicate pushdown) and eagerly-materialized
**layers**. `to_gpu_anndata` is unchanged (f32-native device path).

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
- The **query** (`obs_filter`) path assembles f32 first (post-assembly cast), so a
  `> 2²⁴` integer request there still fails loud (or rounds under `allow_lossy`) —
  the lossless-wide read lands on the eager path. Extending it to the query path is
  a planned follow-up.

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
- The R bindings (`rscx`) do not yet wire the guard.

For those, a `> 2²⁴` count still rounds silently on access; pass `allow_lossy`
where available, or read eagerly to get the guard.

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
- **`adata.raw` is not retyped.** When the file carries a raw count matrix, the
  reconstructed `adata.raw.X` stays `float32` CSR regardless of `data_dtype` —
  only `X` and `layers` are materialized in the requested dtype.
- **Scope.** The kwargs are surfaced on the three read entry points above. Other
  read surfaces (the grouped-shard read helpers, the flat `pyscx.read_cloud(...)`
  cloud helper) are f32-native for now; the cloud *query* path
  (`open_cloud(...).query()...collect()`) returns a `PyQueryResult` and so does
  honor them.

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

- `to_anndata(container="csr", data_dtype=None, index_dtype=None, allow_lossy=False)` — Convert result to AnnData. The default is a zero-copy CSR; the four kwargs have the same semantics as [`Experiment.to_anndata`](#experiment) (`"dense"` container, narrow `data_dtype`/`index_dtype`, fail-loud `allow_lossy` gate). Any non-default request forgoes zero-copy (a cast/copy of `X`).
- `to_csr(container="csr", data_dtype=None, index_dtype=None, allow_lossy=False)` — Return just the scipy CSR matrix (or a dense `numpy.ndarray` for `container="dense"`), with the same dtype kwargs.
- Properties: `n_obs`, `n_vars`, `nnz`, `skipped_shards`, `total_shards`

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
| `calculate_qc_metrics` | ✅ (gene axis) | CSR | N/A | Gene-axis aggregation uses CSC; cell-axis stays CSR. |
| `col_sums` / `col_nnz` / `col_min` / `col_max` / `col_var` | ✅ | CSR | N/A | Column reductions; CSC requires no row deletion vector. |
| `pca` | ❌ (rejects CSC) | CSR | N/A | Inherently row-major; `prefer_format="csc"` raises `ValueError`. |
| `neighbors` / `umap` / `leiden` | N/A | — | N/A | Operate on PCA embeddings / kNN graphs, not on `X`. |

"GPU-fast with CSC" means the op reaches peak GPU throughput **only** with a
backed SCX file that has a CSC sidecar — see
[scanpy.md § GPU-supported vs GPU-fast](scanpy.md#gpu-supported-vs-gpu-fast).
Confirm which path actually ran via the [route metadata](#accelerator-route-metadata).

- `pyscx.accel.pca(adata, n_comps=50, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", prefer_format="csr", memory_budget=None, mask_var=None)` — Randomized/covariance SVD PCA with streaming SpMM. Writes `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]` (incl. `uns["pca"]["params"]`). On GPU: cuSPARSE SpMM + cuSOLVER QR (f32). **`mask_var`** (scanpy semantics): a var-column name, a boolean array of length `n_vars`, or `None`. `None` auto-consumes `adata.var["highly_variable"]` when present, else uses all genes. PCA runs on the selected columns only (backed/lazy/in-memory alike, no materialization — via a column-projecting shard source), while `varm["PCs"]` stays aligned to the **full** `var` axis (excluded vars filled with 0). An all-false mask or a length mismatch raises `ValueError`. **`memory_budget`** (`"8G"`, `"512M"`, bytes, or `None` → 8 GiB) is the RAM ceiling for out-of-core PCA on a backed `X`. Out-of-core PCA re-reads every shard once per pass — ~6–7 passes for randomized, 3 for covariance — so the budget sizes a decoded-shard LRU that lets each shard decode once per pass instead of once per *read*; a budget too small to hold the working set logs a warning and the passes re-decode. It also bounds decode-prefetch, on every input shape: up to `depth` shards decode concurrently ahead of the reduction, and the depth is resolved against this ceiling rather than taken from the process-wide default. On a backed `X` the reserve comes out of the shard LRU, so worst-case live bytes are `budget + one shard` — the same high-water the pre-prefetch loop had. A lazy/transformed `X` has no LRU, so the ceiling bounds the pipeline directly; a shard larger than the budget falls back to depth 1 and prefetches nothing. Prefetch depth is `SCX_ACCEL_PREFETCH_DEPTH` (default 4, capped by the rayon pool and by `SCX_ACCEL_NUM_THREADS`); setting it to `1` disables the pipeline and restores the strictly sequential decode.
- `pyscx.accel.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0, ef_construction=200, ef_search=200, device="auto")` — kNN graph + UMAP-style connectivities. CPU: HNSW. In-VRAM `device="gpu"` routes to rapids-singlecell (`rsc.pp.neighbors`); the native standalone CAGRA dispatch was removed, so with rapids absent / `SCX_FORCE_NATIVE_GPU=1` the standalone op falls back to CPU HNSW (device-resident CAGRA survives only inside the fused `pca_neighbors` path). Writes `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`.
- `pyscx.accel.pca_neighbors(adata, n_comps=50, n_neighbors=15, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", use_rep="X_pca", prefer_format="auto")` — Fused PCA → kNN in one call. For an in-memory `X`, in-VRAM `device="gpu"` routes to the rapids-singlecell pipeline (`rsc.pp.pca` → `rsc.pp.neighbors`, route `rapids_singlecell_gpu`). For backed/lazy `X` (or under `SCX_FORCE_NATIVE_GPU=1`) the native device-resident path runs: the PCA embedding stays GPU-resident and feeds straight into CAGRA (no `X_pca` host round-trip), recording route `gpu_device_resident`. With neither GPU path available it falls back to sequential `pca` + `neighbors`. Writes the union of both ops' slots (`obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]`, `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`). **Note:** the fused entry points do not expose `mask_var`; the CPU/rapids routes delegate to `pca()` and so auto-consume `adata.var["highly_variable"]` when present, but the native device-resident GPU route analyzes all genes — so the PCA gene set is route-dependent when HVGs are flagged. For a deterministic HVG-masked pipeline, run `pca(mask_var=...)` then `neighbors()`/`umap()` separately. See [docs/scanpy.md § Fused PCA → kNN](scanpy.md#fused-pca--knn-pyscxaccelpca_neighbors).
- `pyscx.accel.pca_neighbors_umap(adata, n_comps=50, n_neighbors=15, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, umap_learning_rate=1.0, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", use_rep="X_pca", prefer_format="csr")` — Fused PCA → kNN → UMAP in one call. For an in-memory `X`, in-VRAM `device="gpu"` runs the full rapids-singlecell pipeline (`rsc.pp.pca` → `rsc.pp.neighbors` → `rsc.tl.umap`, route `rapids_singlecell_gpu` on every stage). The native device-resident UMAP path was removed, so backed/lazy inputs (and any GPU host without rapids) fall back to sequential `pca` + `neighbors` + `umap` (the umap stage uses the cuML fallback then CPU SGD). Writes their union including `obsm["X_umap"]`. See [docs/scanpy.md § Fused PCA → kNN → UMAP](scanpy.md#fused-pca--knn--umap-pyscxaccelpca_neighbors_umap).
- `pyscx.accel.umap(adata, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, learning_rate=1.0, random_state=0, device="auto")` — Spectral-init SGD UMAP. In-VRAM `device="gpu"` routes to rapids-singlecell (`rsc.tl.umap`); the native CUDA SGD kernel was removed, so with rapids absent it falls back to cuML and then CPU SGD. Writes `obsm["X_umap"]`.
- `pyscx.accel.rank_genes_groups(adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, stratify_by=None, min_cells_per_stratum=50, rankby_abs=False, tie_correct=False, prefer_format="csr", device="auto")` — Parallel Wilcoxon rank-sum with BH correction. Writes `uns["rank_genes_groups"]`, or returns DataFrame when `stratify_by` is set. `prefer_format="csc"` routes per-chunk reads through the column-major sidecar (see kwarg docs above). The execution route is recorded on `uns["rank_genes_groups"]["scx_accel_route"]` and `uns["scx_accel"]["rank_genes_groups"]` (see [Accelerator route metadata](#accelerator-route-metadata)).
- `pyscx.accel.pseudobulk_dex(adata, groupby=None, test_col=None, reference=None, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr", gene_indices=None, n_cpus=None, backend="pydeseq2", nbglm_options=None, *, sample_cols=None, sample_key=None)` — Streaming pseudobulk aggregation (Rust) + DE testing. Returns DataFrame. **`groupby` here is not what it is in `rank_genes_groups`**: it names the obs columns that together define one pseudobulk *sample* — condition **plus** replicate, e.g. `["disease", "donor_id"]` — while the column actually compared is `test_col`. (In `rank_genes_groups`, and all of scanpy, `groupby` *is* the compared column.) `sample_cols=` / `sample_key=` are aliases named for that role, with `sample_key` also accepting a bare string; pass exactly one of the three, or a `ValueError` names the ones you supplied. `groupby` / `test_col` / `reference` remain semantically required — they are typed optional only so the aliases can exist, and each has its own error when omitted. `backend="pydeseq2"` (default) uses [pydeseq2](https://pydeseq2.readthedocs.io/) (optional dependency); `backend="nb_glm"` uses the Rust-native NB-GLM (no pydeseq2 dependency, route `cpu_nb_glm`) and returns the identical column schema; a custom `design=` formula is honored on this backend too (built via `formulaic`, the parser pydeseq2 uses — one shared-dispersion fit, per-level contrasts; needs the `nbglm` extra), with an optional explicit `nbglm_options["contrast"]`. `prefer_format="csc"` requires a gene subset (either an explicit `gene_indices` argument or a `col_projection` already set on `adata.X`); full-gene CSC pseudobulk has no measurable speed-up. See [docs/pseudobulk_nb_glm.md](pseudobulk_nb_glm.md).
- `pyscx.accel.nb_glm(counts, design, size_factors=None, contrast=None, gene_names=None, sample_names=None, options=None, counts_axis="samples_by_genes", device="auto") → pandas.DataFrame` — Direct Rust-native negative-binomial GLM on an **already-pseudobulked** count matrix + numeric design (DESeq2 replacement). `f64` end-to-end; GPU routes (`gpu_nb_glm_csr`, `gpu_nb_glm_csc`) accelerate the pseudobulk aggregation while the IRLS / Cox–Reid fit itself remains CPU. `counts` is `[n_samples × n_genes]` (`counts_axis="samples_by_genes"`, default) or `[n_genes × n_samples]`; `design` is `[n_samples × n_features]`, full column rank. `size_factors=None` → DESeq2 median-ratio factors. `contrast` is an integer coefficient index, a weight vector, or `None` (last coefficient, DESeq2 convention). `options` is an optional dict (`dispersion ∈ {"moments","cox_reid_mle","cox_reid_shrunk"}`, `min_disp`, `max_disp`, `max_irls_iters`, `irls_tol`, `max_outer_iters`, `fit_dispersion_trend`, `shrink_dispersion`). Returns PyDESeq2-style columns `gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj, dispersion, converged, n_iter` (`lfcSE` on the log2 scale). DESeq2-*style*, not DESeq2-*identical* — includes Cook's-distance outlier filtering and base-mean independent filtering (both default-on), but omits apeglm/ashr LFC shrinkage. See [docs/pseudobulk_nb_glm.md](pseudobulk_nb_glm.md).
- `pyscx.accel.pdex_nb_glm(adata, groupby, reference, stratify_by=None, min_cells_per_group=10, min_cells_per_stratum=50, is_log1p=None, nbglm_options=None, gene_chunk_size=None, prefer_format="csr", device="auto", design=None, output="pandas") → pandas.DataFrame` — Pseudobulk NB-GLM DE for the cell-eval/pdex consumer. Aggregates `groupby × stratify_by` pseudobulk **replicates**, fits one NB-GLM per non-reference perturbation vs `reference`, and returns the **same** cell-eval `DEResults` column schema as `rank_genes_groups_df` (`target, feature, fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change`). **`output="polars"` is required to feed `cell_eval`** — its `DEResults.data` is typed `pl.DataFrame`; the default `output="pandas"` needs no optional dependency. **`stratify_by` is required**, as a **list** of obs column names (e.g. `["donor"]`; a bare string is rejected with a clear error) — it forms the replicates: with no stratifier the dispersion is unidentifiable, so it raises `ValueError` pointing to `pdex_ref` / `rank_genes_groups`. `is_log1p=None` auto-detects via `adata.uns["log1p"]`; NB-GLM requires **raw counts** and errors on log1p-normalized input. Records route on `adata.uns["scx_accel"]["pdex_nb_glm"]` (GPU routes `gpu_nb_glm_csr` / `gpu_nb_glm_csc`; CPU route `cpu_nb_glm`). See [docs/pseudobulk_nb_glm.md](pseudobulk_nb_glm.md).
- `pyscx.accel.rank_genes_groups_df(adata, groupby=None, reference="rest", n_genes=None, gene_chunk_size=None, rankby_abs=False, tie_correct=False, device="auto", output="pandas", *, group=None, key="rank_genes_groups", pval_cutoff=None, log2fc_min=None, log2fc_max=None) → pandas.DataFrame` — **Two modes.** *Compute* (`groupby=`): same Wilcoxon rank-sum as `rank_genes_groups()`, returns the cell-eval `DEResults` schema `(target, feature, fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change)`; pass `output="polars"` for `cell_eval.initialize_de_comparison()`. *Extract* (`group=`): the **scanpy `sc.get.rank_genes_groups_df` alias** — does not recompute; reads the precomputed `adata.uns[key]` and returns scanpy's columns `(names, scores, logfoldchanges, pvals, pvals_adj)`, with a leading `group` column when `group` is a list **or `None`**. `pval_cutoff` / `log2fc_min` / `log2fc_max` are scanpy-style row filters (extraction only). Pass either `groupby=` or `group=`, not both. **`group=None` (or omitting it) with no `groupby=` extracts every group** in `uns[key]`, matching scanpy's "All groups are returned if group is None"; with neither and no `uns[key]` it errors naming both remedies. Both modes return a **pandas** DataFrame by default, so scanpy-shaped idioms (`.map`, `df[col] = …`) work directly; `output="polars"` returns the polars frame `cell_eval` consumes and requires the `eval` extra. (`gene_symbols=` var-name remap is not supported yet.)
- `pyscx.accel.pdex_ref(adata, groupby, *, reference="non-targeting", is_log1p=None, geometric_mean=True, epsilon=1e-9, cpm_filter=None, gene_chunk_size=None, prefer_format="auto", device="auto", output="pandas", use_raw=None, layer=None) → pandas.DataFrame` — Perturbation-screen differential expression: Mann–Whitney U + pseudobulk geometric-mean log fold change vs a single reference group. Pinned bit-for-bit to upstream [`pdex`](https://github.com/ArcInstitute/pdex) (`pyscx/tests/test_pdex_ref_parity.py`). Returns one row per (target group, gene) excluding the reference group, with columns `target`, `feature`, `target_mean`, `ref_mean`, `target_membership`, `ref_membership`, `log2_fold_change` (also exposed as `fold_change` for migration), `percent_change`, `p_value`, `statistic`, `fdr`. Note: here `fold_change` is a deprecated alias of `log2_fold_change` (log2 scale) — this differs from `rank_genes_groups_df`'s `fold_change` column, which is linear (`exp2` of the log2 value). **`epsilon`** (default `1e-9`, matching pdex ≥ 0.2.x) is a finite-guard pseudocount added to the count-space means before the fold-/percent-change ratio — not applied to CPM or the MWU test; with the default the outputs stay finite, and `0/0` (a gene unexpressed in both groups) is `0.0`. Pass `epsilon=0.0` to recover the legacy behaviour where genes undetected in the reference yield `±inf`. **`cpm_filter`** (optional float `T`) keeps a gene for a test group iff its pooled arithmetic counts-per-million `target_cpm > T` **or** `ref_cpm > T` (strict `>`, mode-independent); dropped rows are removed and FDR is recomputed over the surviving genes only. `is_log1p=None` auto-detects, and the answer does not depend on how `X` is stored — a backed handle and an in-memory `AnnData` over the same data resolve the same mode. Resolution order: `adata.uns["log1p"]` (present → log1p); else, for a lazy `X`, a `Log1p` in its transform chain; else, for a backed `X`, the catalog's integer `value_max` against the same `< 30` heuristic the in-memory path uses (catalog-only — no decode, no materialization); else, in memory, `max(X) < 30`. The matrix probed is the one `use_raw=` / `layer=` selected, not always `adata.X`. **A backed file whose catalog cannot bound its value range, with no `uns["log1p"]`, raises `ValueError`** rather than guessing — shards that are float-encoded (the format writes `value_max = 0` for those), carry no statistics, or hold no values at all. Answering `False` there is what used to make a backed `pdex_ref` disagree with its in-memory twin by orders of magnitude. Pass `is_log1p=True`/`False` to override or to resolve that case (`adata.X.max()` computes the heuristic's input by streaming, without materializing). **`output`** is `"pandas"` by default (no optional dependency) or `"polars"` for the frame upstream `pdex` and `cell_eval` use; columns are identical either way. `device="auto"` picks GPU when available and falls back to CPU otherwise. **GPU v3-CSC path (default):** the GPU dispatch routes through a CSC-direct driver that drops the per-chunk dense intermediate and uses a CSC shard source with pipelining + per-chunk shard-range pre-filter. This is the default GPU DE route (the former `SCX_GPU_DE_V3` opt-in gate was removed when v3 became the unconditional default). CSC-direct requires a CSC sidecar on the SCX file — build it with `pyscx.from_anndata(adata, path, csc="always")` or `scx convert --csc=always`. In-memory inputs (scipy CSR) and files without a sidecar fall back to the v3-CSR-direct path automatically. The route that actually ran is recorded on `adata.uns["scx_accel"]["pdex_ref"]` (see [Accelerator route metadata](#accelerator-route-metadata)) — `gpu_csc_v3` confirms the CSC-direct path, `gpu_csr_v3` + `fallback_reason="no_csc_sidecar"` confirms the CSR fallback. Wall-time numbers and the disposition live in [`docs/performance.md` § Per-operation timing](performance.md#per-operation-timing).
- `pyscx.accel.pseudobulk_means(adata, groupby, min_cells_per_group=1, device="auto") → (ndarray, list[str])` — Group-by mean on sparse X, streaming shard-by-shard (works on backed, lazy, scipy CSR, or dense). Returns `(means[P, G] float64, sorted group names)`. Foundation for the perturbation evaluation metrics below. GPU (`device="gpu"`/`"auto"` on a GPU host): wraps the DE pseudobulk kernels — f64 accumulation matches the CPU within `atol≈1e-5` (dense `embed_key` at the f32 bar). Route `gpu_csr` / `cpu_csr` (the GPU pseudobulk reuses the DE CSR kernels). An **in-memory** `X` is copied once into an owned buffer so the kernel can run with the GIL released — a parallel copy, measured at ~14 ms per 192 MB, with a `UserWarning` above 1 GB; a backed or lazy `X` streams and never pays it. See [Coding conventions § Python Bindings](conventions.md#python-bindings-pyscx).
- `pyscx.accel.perturbation_metrics(adata_real, adata_pred, pert_col="perturbation", control="control", metrics=None, min_cells_per_group=1, device="auto") → dict[str, dict[str, float]]` — Bundled bulk metrics `{pearson_delta, mse, mae, mse_delta, mae_delta}` between paired real/pred AnnData. Matches cell-eval's metrics within atol=1e-6. GPU (`device="gpu"`/`"auto"` on a GPU host): the per-group pseudobulk means run on the GPU (f64), the five bulk metrics on the host — CPU parity `atol≈1e-6`. Route `gpu_csr` / `cpu_csr`.
- `pyscx.accel.energy_distance(adata_real, adata_pred, pert_col="perturbation", control="control", metric="euclidean", embed_key=None, backend=None, dtype=None, device="auto") → float` — Pearson correlation of per-perturbation e-distance vectors (real vs pred). **Polarity:** despite the name this is a *score*, not a distance — it lies in `[-1.0, 1.0]` where `1.0` = perfect and **higher is better**; use it directly, don't invert. Returns `nan` when either side's e-distance vector is constant (e.g. a control-broadcast predictor) or there is only one non-control perturbation. Avoids `[N, N]` distance materialization (per-row sum reduction even on the gemm path); precomputes control self-distance once; rayon-parallel across perturbations. `backend ∈ {"auto" (default), "gemm", "scalar"}` — `"auto"` picks faer-dispatched gemm for euclidean/cosine and the scalar row-by-row path for L1; `"gemm" + metric="l1"` raises `RuntimeError` (no decomposition exists). `dtype ∈ {"f32" (default), "f64"}` controls only the matmul / per-pair arithmetic precision; reductions always accumulate in `f64`. f32 + gemm matches f64 + scalar within `atol=1e-4` correlation / `atol=1e-3` per-pert. GPU (`device="gpu"`/`"auto"` on a GPU host): a gemm-based GPU pairwise-distance mean (`‖x−y‖²=‖x‖²+‖y‖²−2·xyᵀ`) handles **euclidean + cosine** at f32 with f64 reductions — CPU parity at the same `atol=1e-4` correlation bar. `metric="l1"` (no gemm decomposition) and `dtype="f64"` stay on the CPU path even under `device="gpu"` (route `cpu_csr`, no crash). GPU route `gpu_dense`.
- `pyscx.accel.energy_distance_details(...)` — Same signature (including `backend` / `dtype` / `device`) as `energy_distance` but returns `{"correlation": float, "d_real": {pert: float}, "d_pred": {pert: float}, "pert_names": [...]}`.
- `pyscx.accel.discrimination_score(adata_real, adata_pred, pert_col="perturbation", control="control", metric="l1", exclude_target_gene=True, embed_key=None, min_cells_per_group=1) → dict[str, float]` — Per-perturbation normalized rank of the predicted perturbation effect's distance to the correct real effect. `metric ∈ {"l1", "l2"/"euclidean", "cosine"}`. `exclude_target_gene=True` drops the gene matching each perturbation's name from the distance (matches cell-eval's default).
- `pyscx.accel.knockdown_efficiency(adata, pert_col="perturbation", control="control", eps=1e-8)` — Per-cell knockdown efficiency + log-fold change vs control baseline. Input must be normalized (NOT log1p'd); log1p is applied internally. Writes `adata.obs["KnockDownEfficiency"]` and `adata.obs["KnockDownGeneFC"]` (both float32, NaN for control cells and cells whose perturbation name isn't in `var_names`). Matches `arc_bench.tools.normalize_transform.core` within atol=1e-6.
- `pyscx.accel.clustering_agreement(adata_real, adata_pred, pert_col="perturbation", control="control", metric="ami", real_resolution=1.0, pred_resolutions=None, n_neighbors=15, embed_key=None, min_cells_per_group=1) → float` — Builds perturbation-centroid kNN graphs, sweeps Leiden resolutions, scores best real-vs-pred agreement via AMI / NMI / ARI. **All-native-Rust** — no scanpy / anndata / igraph dispatch; uses `scx_accel::neighbors::build_knn_graph` (HNSW via `instant-distance`, `ef_construction=200, ef_search=50, seed=0`) plus `scx_accel::leiden` sequential mode (`max_iterations=2, parallel=false, seed=0` — matches scanpy's `flavor="igraph", n_iterations=2`). Pred-side kNN graph built once and reused across the resolution sweep; whole hot path runs under `py.allow_threads`. Matches cell-eval's `ClusteringAgreement` within `atol=0.15` aggregate (stochastic Leiden; exact score match not expected — algorithms agree exactly on graphs with `n_perts ≥ 16`).
- `pyscx.accel.adjusted_mutual_info(labels_a, labels_b) → float` — AMI on integer label arrays (arithmetic-mean convention). Matches `sklearn.metrics.adjusted_mutual_info_score` within atol=1e-10.
- `pyscx.accel.normalized_mutual_info(labels_a, labels_b) → float` — NMI (arithmetic-mean). Matches `sklearn.metrics.normalized_mutual_info_score` within atol=1e-10.
- `pyscx.accel.adjusted_rand_index(labels_a, labels_b, rescaled=False) → float` — ARI. Default matches `sklearn.metrics.adjusted_rand_score` (`1.0` identical, `~0` random, negative for worse-than-random; consistent with NMI/AMI above). Pass `rescaled=True` for cell-eval's `(ARI + 1) / 2` in `[0, 1]`.
- `pyscx.accel.leiden(adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=2, device="auto", parallel=False, theta=1.0)` — Community detection on the kNN graph. Reads `obsp["connectivities"]` (from `neighbors()`). GPU: cuGraph Leiden (up to 47× faster). CPU: **Rust-native** — the Python `igraph`/`leidenalg` shim was removed (call `scanpy.tl.leiden(flavor="leidenalg")` directly if you need it). The CPU path is **Louvain + constrained refinement**: it omits the Leiden paper's well-connectedness admissibility conditions and CPU `theta` (non-default `theta` is warned and ignored on CPU), so do not treat it as a full well-connected-Leiden guarantee. Writes `adata.obs[key_added]` (categorical) and `adata.uns["leiden"]` (params + backend metadata). GPU and CPU may produce different partitions due to algorithmic differences; compare via ARI/NMI.
- `pyscx.accel.harmony_integrate(adata, key, *, basis="X_pca", adjusted_basis="X_pca_harmony", n_clusters=None, theta=None, sigma=0.1, lamb=None, alpha=0.2, max_iter=10, max_iter_kmeans=6, epsilon_harmony=1e-2, epsilon_kmeans=1e-3, block_size=0.05, batch_prop_cutoff=1e-5, tau=0.0, random_state=0, device="auto")` — Clean-room Rust implementation of Harmony2 (Korsunsky et al., 2019). Soft k-means clustering with a diversity penalty, followed by ridge-regression correction of `adata.obsm[basis]` (`"X_pca"` by default). `key` accepts a single `obs` column name or a list for multi-covariate batch correction; each is factorised via `pandas.factorize(sort=False)`. Writes the corrected embedding (f32, N × d) to `adata.obsm[adjusted_basis]` — by default a **new** key `"X_pca_harmony"`, leaving the input `basis` intact (matching scanpy); pass `adjusted_basis=basis` (e.g. `"X_pca"`) to overwrite in place. Convergence metadata goes to `adata.uns["harmony"]` (`params`, `converged`, `n_iterations`, `objective_harmony`, `backend`). Parameter names and the non-destructive default match `scanpy.external.pp.harmony_integrate`, so existing scanpy pipelines can swap in. GPU path (when built with `--features gpu`) accelerates distance / L2-norm / batched scatter-subtract kernels; k-means++ init and covariance inversion stay on CPU. Numerical parity vs R `harmony` v2.x: mean per-PC Pearson r 0.989–0.999 on the three validation fixtures.
- `pyscx.accel.compute_lisi(adata, key, *, basis="X_pca", perplexity=30.0, n_neighbors=None, approximate_knn=False) → np.ndarray` — Local Inverse Simpson Index on an `obsm` embedding. Exact brute-force kNN (matches R `FNN::get.knn`) + per-cell Gaussian-bandwidth search (t-SNE Hbeta routine) + Simpson index over kernel-weighted neighbour category probabilities. Returns LISI vector of length N and also writes to `adata.obs[f"lisi_{key}"]`. Values near 1 → poor mixing (neighbourhoods dominated by one category); values approaching the number of categories → uniform mixing. `n_neighbors` defaults to `ceil(3 × perplexity)`. ~10× faster than R `lisi::compute_lisi` on D1–D4 (the previously reported 0.8–2.4 % mean-LISI agreement predates the 2026-07 raw-distance kernel fix and is pending a benchmark recapture). Set `approximate_knn=True` to swap the exact O(N²) sweep for an HNSW approximate kNN (~10× faster at N ≳ 100k, ~0.01–0.05 mean-LISI drift); the exact path logs a hint to do so above ~50k cells.
- `pyscx.accel.normalize_total(adata, target_sum=None, device="auto")` — Materialization-free row normalization. `target_sum=None` (default) scales each cell to the **median of positive per-cell totals** (scanpy `target_sum=None` semantics); a float pins an explicit target. **Compatibility break (v0.11.6+):** the previous default was a hard `1e4` — pass `target_sum=1e4` to restore it. On `ScxBackedSparseDataset`: computes row sums via streaming (deriving the median deletion-correctly), creates `ScxLazyTransformedDataset` wrapper. On `ScxLazyTransformedDataset`: appends `NormalizeTotal` transform to chain. On scipy CSR: delegates to `sc.pp.normalize_total()` (forwarding `None`).
- `pyscx.accel.log1p(adata)` — Materialization-free log1p. On `ScxBackedSparseDataset`: creates `ScxLazyTransformedDataset` with `Log1p` transform. On `ScxLazyTransformedDataset`: appends `Log1p` to chain (fuses with preceding `NormalizeTotal` when possible). On scipy CSR: delegates to `sc.pp.log1p()`. **Every arm stamps `adata.uns["log1p"] = {"base": None}`**, the same annotation `sc.pp.log1p` writes — so `rank_genes_groups` takes its `expm1` logFC branch, `pdex_ref` resolves the log1p mean mode, and `pdex_nb_glm` correctly refuses the result as non-raw-counts. (The lazy arms previously wrote nothing, leaving the SCX-native pipeline the only one whose log1p was invisible downstream.)
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
- `pyscx.accel.calculate_qc_metrics(adata, qc_vars=None, log1p=True, inplace=True, prefer_format="csr")` — Streaming QC metrics for backed/lazy data without materialization. Computes per-cell `n_genes_by_counts`, `total_counts` and per-gene `n_cells_by_counts`, `total_counts`. Supports `qc_vars` for gene subsets (e.g., `["mt"]` for mitochondrial percentage). When `inplace=True`, writes to `adata.obs`/`adata.var`; when `False`, returns `(obs_df, var_df)`. `prefer_format="csc"` routes the gene-axis aggregation through the CSC sidecar (cell-axis stays CSR — row aggregations have no CSC win). Falls back to `sc.pp.calculate_qc_metrics()` for scipy/dense.

  **Column projections** are honored on every route: per-cell totals cover only the visible genes, the gene axis is returned at the visible width, and `qc_vars` masks are read against `adata.var`. Projections arrive from `filter_genes`, `highly_variable_genes(subset=True)`, a gene-selected `to_anndata(backed=True, var_names=[...])`, `X.set_col_projection(...)`, or `adata.X = adata.X[:, cols]` — *not* from `adata[:, mask]`, which raises on a backed SCX `X` because anndata has no registered view type for it (that's the deferred `as_view`/`_subset` work).

  *Behavior change, and it differs by route.* Under `prefer_format="csr"` (default) a projection previously produced a physical-width gene axis, so the call raised `ValueError: Length mismatch` — loud, no wrong numbers. Under `prefer_format="csc"` the gene axis was already correct, so lengths matched and the call **returned silently wrong per-cell numbers** (projection-blind row axis; `qc_vars` visible positions applied to the on-disk axis). Projected QC results published from the CSC route are worth re-checking. Separately, on a lazy `X` the `qc_vars` subset sums are now taken **through** the transform chain, so `pct_counts_<v>` divides a transformed numerator by a transformed denominator — a silent numeric change that shifts which cells a `pct_counts_mt` filter keeps.

  Requires `adata.var` row *i* to describe visible column *i*. `set_col_projection` sorts and dedups, so slicing `var` in a different (unsorted) order mislabels the gene axis without erroring.

  All of the above is computed in **two** shard passes — one row-axis, one column-axis — for up to 64 `qc_vars`; beyond that the row pass repeats once per additional 64.
- `pyscx.accel.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", batch_key=None, span=0.3, subset=False, n_bins=20, device="auto", prefer_format="csr", layer=None)` — Streaming HVG selection. Default CPU + CSR streams `mean_var` and clipped sums shard-by-shard via `ShardSource`; multi-batch runs CSR. Works on `ScxBackedSparseDataset`, `ScxLazyTransformedDataset`, **and a materialized scipy/dense `X`** for `flavor` in `seurat_v3` / `seurat_v3_paper` / `seurat` (a materialized `X` is wrapped in a single-shard `ShardSource`), so the eager `to_anndata()` idiom gets the same numerics and the same per-batch LOESS-singularity tolerance as the backed path. Only flavors scx does not implement natively (e.g. `cell_ranger`) delegate to `scanpy.pp.highly_variable_genes` (one-shot `UserWarning`). `prefer_format="csc"` routes single-batch seurat_v3 through `streaming_mean_var_csc` and `streaming_clip_square_sum_csc` (multi-batch + GPU + non-seurat_v3 flavors raise on CSC). Writes `var["highly_variable"]`, `var["means"]`, `var["variances"]`, `var["variances_norm"]`, `var["highly_variable_rank"]`.
- `pyscx.accel.score_genes(adata, gene_list, ctrl_size=50, gene_pool=None, n_bins=25, score_name="score", random_state=0, method="control", layer=None, device="auto")` — CPU-native `sc.tl.score_genes` equivalent. Streams shard-by-shard via `ShardSource`, so it runs on `ScxBackedSparseDataset`, `ScxLazyTransformedDataset`, and materialized scipy/dense `X` with bounded memory. Writes a per-cell float64 score to `adata.obs[score_name]`. `method="control"` (default) = `mean(gene_list) − mean(control)` with control genes sampled from expression-matched bins (Rust-native deterministic sampler — **not** numpy-RNG-compatible, so absolute scores differ from scanpy but rank-correlate near-perfectly); `method="mean"` = per-cell mean over `gene_list`; `method="zscore"` = decoupler `mt.zscore` (`Σ z / √k`, ddof=1 std). `gene_list`/`gene_pool` are symbols resolved against `adata.var_names`; missing genes are dropped with a `UserWarning`. CPU-only (`device` accepted for symmetry; no GPU kernel). Records its route (`cpu_csr`) on `adata.uns["scx_accel"]["score_genes"]`.
- `pyscx.accel.pflog(adata, *, alpha=None, layer=None, store="pca", n_components=50, n_oversamples=10, n_power_iterations=2, zero_center=True, random_state=0, obsm_key="X_pflog_pca", baseline_key="pflog_baseline", layer_out=None, out=None, store_repr="delta_baseline", shard_size=None, dense_max_elems=200_000_000, device="auto")` — **PFlog (v4) / shifted-log on raw counts** normalization (Booeshaghi et al., DOI 10.1101/2022.05.06.490859; no direct scanpy equivalent). **By default (`store="pca"`) it produces a baseline-aware PCA embedding in `adata.obsm[obsm_key]` and leaves `X` as raw counts — it does *not* transform `X` in place like `normalize_total`/`log1p`; pass `store="dense"` (+ `out=` for large data) for the matrix itself.** Operates on **raw counts**: shift by the matrix-wide Anscombe pseudocount `1/(4α)`, `log`, then subtract the within-cell mean — no per-cell depth (it cancels under the Anscombe scale). `α` (NB overdispersion `Var = μ + α·μ²`) is estimated once from the counts when `alpha=None`, or pinned to a float (e.g. a reference `α` reused across datasets); the fit is stamped into `adata.uns["pflog"]` (`alpha`, `pseudocount`, `alpha_source`, and — when estimated — `n_genes_used`/`fell_back`). The exact transform `Z = delta + baseline·1ᵀ` is dense, but decomposes into a sparse `delta` (= `scale{4α}` → `log1p`, i.e. `log1p(4α·x)`) plus a per-cell `baseline = −(1/D)·Σ_j delta_ij`, so PCA never densifies. Streams shard-by-shard via `ShardSource` (backed / lazy / in-memory `X`; a transformed lazy `X` is rejected — raw-count guard). Always writes `adata.obs[baseline_key]`. `store ∈ {"pca","all"}` → baseline-aware out-of-core randomized PCA into `adata.obsm[obsm_key]` (+ `adata.uns[f"{obsm_key}_singular_values"]`). `store ∈ {"dense","all"}` materializes the exact dense `Z`: with `out=None` into `adata.layers[layer_out]` (guarded by `dense_max_elems`), or with `out=<path.scx>` streamed to a new SCX file (no size guard) per `store_repr` — `"delta_baseline"` (default, compact `O(M)`: Pcodec `delta` X + `baseline` obs, reconstruct with `pyscx.accel.pflog_reconstruct`) or `"dense"` (literal full-density CSR, forced Zstd, small default `shard_size`). CPU-only (`device` accepted for symmetry; no GPU kernel). Records its route (`cpu_csr`) on `adata.uns["scx_accel"]["pflog"]`.
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
- `fallback_reason` — why the ideal route wasn't taken: `none`, `no_cuda`, `no_rapids`, `no_csc_sidecar`, `unsupported_dimensions`, `unsupported_input_layout`, `user_forced_cpu`, or `perf_policy`.
- `chunk_size`, `csc_available`, `graph_replay`, `shards_decoded`, `shards_uploaded` — optional detail (`None` when not tracked). `shards_decoded` counts **slab passes** (`n_gene_chunks × n_shards` on the chunked routes), not host decodes — with `resident_csr` true a shard is decoded once and replayed from VRAM, and this counter does not move.
- `resident_csr` — GPU CSR DE routes only. `True` when the whole matrix was held **device-resident** across gene chunks, `False` when the route streamed (over the VRAM budget, only one gene chunk, or `SCX_GPU_DE_RESIDENT=0`), `None` where there is no residency decision to make — CPU, dense, or a CSC-direct route, which prefilters by column range and never re-decodes. See [scanpy.md § GPU DE device residency](scanpy.md#gpu-de-device-residency).
- `transfer_mode`, `bytes_uploaded`, `device_id`, `rapids_version`, `cuml_version`, `cupy_version` — GPU device-handoff / rapids detail. `transfer_mode` is how the GPU-resident matrix was produced, one of: `scx_device_decode_gpu` (framed Scx1 shards decoded fully in VRAM group-by-group — only the indptr is uploaded; covers dense ≥128-nnz rows via the BitPacker4x kernel; from `to_gpu_anndata`), `scx_device_handoff_streamed` (on-device but a shard host-bounced because it is not an Scx1 shard — a non-Scx1 codec; from `to_gpu_anndata`), `scx_device_handoff` (host-assembled CSR or an already-device-resident input; from `to_gpu_anndata` or a rapids op consuming a device-resident `X`), or `anndata_to_gpu` (rapids uploaded a host `X` via `rsc.get.anndata_to_GPU`). `bytes_uploaded` is the real host→device byte count for that mode (`None` when not tracked).
- `math_mode`, `spmm_policy` — GPU PCA only (Task 2.5): the cuBLAS/cuSPARSE float math mode (`strict_fp32` / `allow_tf32`) and the cuSPARSE SpMM algorithm policy (`default` / `deterministic` / `benchmark_once`) the run used. `None` on CPU / ops without these knobs. `graph_replay` is always `False` for GPU PCA: SpMM-segment CUDA-graph capture was removed (`cusparseSpMM` is not capture-safe on current cuSPARSE). The device-resident PCA power loop still runs and avoids re-decoding/re-uploading the matrix each power iteration; the field is retained for metadata continuity.

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
| `"harmony_integrate"` | `pyscx.accel.harmony_integrate` (`gpu_dense` native cuBLAS path / `cpu_dense`) |
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

  > [!WARNING]
  > **This is a handle-level knob, not an axis subset.** It moves `X` only — `var`, `layers`, `varm` and `varp` are left at the old width, so the AnnData is inconsistent until you slice them yourself. Use `pyscx.accel.subset_var(adata, mask)` (or `adata[:, mask]`) for a real gene subset — see [Axis subsetting and aligned members](#axis-subsetting-and-aligned-members). Reach for this only when you want to reproject a bare handle.

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
- `var(axis=0|1)` `→ numpy.ndarray` — Column or row variance. `axis=0` is two-pass streaming; `axis=1` and `axis=None` **materialize** the visible matrix via `to_memory()` and compute `E[X²] − E[X]²` in f32, so they are neither out-of-core nor clamped at zero (a near-constant row can return a small negative). Prefer the backed `X.var(axis=1)`, which streams in f64 and clamps.
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
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. |
| `obs_columns` | `[]` | Obs metadata column names included in each batch. |
| `normalize` | `True` | Total-count normalize (fused with `log1p` in a single CSR row scan). **On by default** — set `normalize=False` (with `log1p=False`) for raw-count output. |
| `log1p` | `True` | Apply `log1p` after normalize. **On by default** — set `log1p=False` for raw-count output. |
| `target_sum` | `1e4` | Normalization target sum. |
| `pflog` | `False` | Apply PFlog (v4) / shifted-log normalization on raw counts (Booeshaghi et al.) instead of `normalize`/`log1p`. Mutually exclusive with them — when `True` it takes precedence and those flags are ignored. |
| `pflog_alpha` | `None` | PFlog NB overdispersion `α` (matrix-wide pseudocount `1/(4α)`). `None` estimates `α` once at loader construction from the raw counts (single-modality only); a float pins it. Only used when `pflog=True`. |
| `shard_group_size` | `8` | Shards per I/O group. Sequential I/O within each group for disk efficiency. |
| `prefetch_batches` | `4` | Ring buffer depth — number of pre-built batches to buffer ahead. |
| `seed` | `42` | RNG seed for reproducibility. Deterministic shuffle via `(seed, epoch)`. |
| `max_memory_mb` | adaptive (≥512) | Memory budget. **When omitted**, the budget is *adaptive*: it scales up to fit the file's requested configuration (so a full-width ~33k-gene file keeps its requested `batch_size` instead of silently shrinking), floored at 512 MB and capped at 4096 MB. Pass an explicit value to pin a **hard ceiling** — then the pipeline auto-tunes `shard_group_size`, `prefetch_batches`, and `batch_size` down to fit (the prior behaviour). |
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
- `scx convert <input> <output> [--from h5ad|10x|h5mu|scx] [--to h5ad|h5mu|scx] [--codec auto|none|scx1|zstd|lz4|pcodec] [--shard-size N] [--stream[=true|false]] [--csc off|always] [--csc-cols-per-shard N] [--modality NAME] [--memory-budget SIZE] [--strict-uns] [--dense-zero-epsilon F] [--temp-dir DIR] [--modalities CSV] [--modality-types NAME:TYPE,...] [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N] [--bitmap off|auto|always] [--reader-threads N] [--writer-queue-depth N]` — `--stream` bounds peak memory to one shard's worth of CSR plus encode buffers. It applies to h5ad ↔ SCX and h5mu ↔ SCX in both directions, which stream by default; MTX ↔ SCX and 10x → SCX have a single materialising path, so omit the flag there (an explicit `--stream` on those directions is an error, and `--stream=false` is accepted as a no-op). On ingestion (h5ad/h5mu → SCX), combine with `--csc always` for a two-pass CSR-then-`rebuild_csc_inplace` write (transient disk ~2× the output). On export (SCX → h5ad/h5mu), the streaming writer pre-allocates the `/X/{indptr,indices,data}` HDF5 triplet from catalog stats (or a single pre-scan when deletion vectors are active) so the on-disk layout is deterministic. Pass `--stream=false` to opt into the legacy materialising path on either side. For multimodal SCX → h5ad, combine `--to h5ad --modality NAME` to extract a single modality.

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
- `scx optimize <input> <output> [--force] [--codec {auto|fast|compact|scx1|shufdelta|compact-trial}] [--shard-obs {off|auto|always}]` — In-place upgrade (single-modality): re-encode + canonicalize every CSR shard and row-group-frame it (codec-agnostic random access via the row-group `BlockIndex`), stamping `format_version=4`, preserving row layout / obs / var / obsm / uns / indexes / deletion vectors. Default `auto` re-encodes adaptively (adopts ShufDeltaZstd where it wins by the margin); `--codec fast` forces the heuristic (Scx1 on low-median integer shards — keeps the GPU device-decode route, framed Scx1 shards decode in VRAM); `--codec scx1` forces Scx1 on every integer shard. `--shard-obs` (default `auto`) migrates a legacy single-section obs to the sharded layout — `auto` shards when `n_obs > shard_target_rows`, `always` unconditionally, `off` keeps the single section; an already-sharded obs is preserved regardless. Drops the CSC sidecar (rerun `scx build-csc`). `<output>` may equal `<input>` (atomic rename). See [docs/operations.md § Optimize](operations.md#optimize).
- `scx rollback <file> [--to-seq N]`
- `scx merge <file1> <file2> [<...>] --output <path> [--index-obs CSV] [--index-var CSV] [--index-preset NAME] [--index-auto-threshold N]` — Merge multiple files; `--index-*` rebuilds the predicate index against the merged output (without it, pushdown regresses to a full obs scan on the merged file).
- `scx query <input> (--filter <expr> | <filter>) [--count] [--output <path>] [--select-genes <path>] [--normalize N] [--log1p] [--limit N] [--json]` — the obs predicate may be given via `--filter` (consistent with `scx subset` / `scx delete`) or positionally (back-compat); supply one form, not both. `<input>` accepts a local `.scx` file path, an exploded `.scxd/` directory, or a cloud URL (`gs://`, `s3://`, `az://`, `file://`). For cloud inputs the query is served via the `SectionReader` cloud path with no `scx pull` step. See [docs/cloud.md § Cloud-native query](cloud.md#cloud-native-query).
- `scx subset <input> [output] [--filter <expr>] [--genes <path>] [--modality NAME] [--dry-run] [--shard-size N] [--codec auto|none|scx1|zstd|lz4|pcodec] [--rebuild-csc] [--index-obs <csv>] [--index-var <csv>] [--index-preset {cellxgene,perturbseq,training}] [--index-auto-threshold N]` — Extract a subset of cells and/or genes into a new SCX file (`output` is optional with `--dry-run`). An input's predicate index cannot be carried over — row / column projection invalidates every shard range in it — so the `--index-*` flags **rebuild** it against the subset, mirroring `scx merge` / `scx compact` / `scx sort`. Without any of them the output has no predicate-index sections and `filter_obs` pushdown falls back to a full obs scan (a warning names the flags). A forced column missing from the subset's obs/var fails before anything is written, `--dry-run` included; the indexed columns are recorded in the `subset` provenance entry.
- `scx build-csc <input> <output> [--memory-limit 4G] [--force]` — Build CSC (column-major) shards from existing CSR data. `--memory-limit` accepts the same size forms as `--memory-budget` (see [Memory budgets](#memory-budgets)).
- `scx upgrade <input> [output] [--in-place]` — Upgrade an SCX file to the latest **unframed** format version (v3). Does not add row-group framing, so it does not reach v4 — use `scx optimize --row-group-rows N` for the framed v4 layout.

### External annotation import
- `scx cellbender-import <target.scx> <cellbender_out.h5> [--layer NAME] [--obs-key NAME] [--var-key NAME] [--prefix P] [--uns-key K] [--overwrite] [--on-missing-rows zero|error] [--on-extra-rows warn|error] [--gene-axis identical|reorder|subset] [--latent-embedding] [--dry-run]` — Attach a CellBender `remove-background` output as a layer, in place, joined by barcode. `--gene-axis` defaults to `identical`: a silent gene permutation is biologically wrong, so reordering must be opted into. Needs `--features hdf5`. See [docs/operations.md § CellBender import](operations.md#cellbender-import).
- `scx obs-import <target.scx> <table.csv> [--key CSV] [--source-key CSV] [--columns CSV] [--rename SRC=DST]... [--prefix P] [--keep-key-columns] [--delimiter C] [--status-column NAME] [--uns-key K] [--uns-key-from-source K]... [--overwrite] [--on-missing-rows null|zero|error] [--on-extra-rows warn|error] [--dry-run]` — Import a delimited annotation table (CSV/TSV) as obs columns, in place. Joins by key string, never by row position; uncovered target rows get `null`, not `0`. `--key a,b` is a composite key, and `--key obs_names` keys on the obs index. `--source-key` names the source side per component when the table spells the key differently (`--key sample_id,obs_names --source-key sample_id,barcode`), pairing positionally. Ungated — a delimited-table reader needs no libhdf5; an `.h5ad` source does (`--features hdf5`). `--dry-run` runs the join and a key diagnosis and writes nothing. Undo with `scx rollback`. See [docs/operations.md § External obs import](operations.md#external-obs-import).
- `scx doublet-import <target.scx> <table.csv> --tool {scdblfinder|scrublet|doubletfinder|doubletdetection|solo|scds|generic} [--key CSV] [--source-key CSV] [--key-added K] [--score-column NAME] [--call-column NAME] [--call-true TOK] [--call-false TOK] [--drop-native-columns] [--delimiter C] [--uns-key-from-source K]... [--overwrite] [--on-missing-rows null|zero|error] [--on-extra-rows warn|error] [--dry-run]` — The doublet wrapper over `obs-import`, normalising each tool's spellings onto `<K>_score` / `<K>_predicted` / `<K>_status` (+ `uns["<K>"]`). `--tool scds` emits no call column, so no `<K>_predicted` unless `--call-column` opts in; `--tool generic` requires `--score-column`. Use `--drop-native-columns` when the source is an h5ad exported from the target file.

### Cloud operations (`--features cloud`)
- `scx cloud-optimize <input> [--output <path>]`
- `scx explode <input> <output>`
- `scx pack <input> <output>`
- `scx pull <source-url> <dest> [--parallelism N] [--no-cloud-ready] [--filter <expr>]`
- `scx push <source> <dest-url> [--parallelism N]`

Note: the `scx-cli` crate has optional `hdf5` and `cloud` feature flags. HDF5 support is opt-in (`--features hdf5`). Cloud operations are opt-in (`--features cloud`). End-users building from a clone with `cargo install --path scx-cli` can pass `--features default-bin` to get an h5ad-capable build in one command (the crate is not on crates.io, so plain `cargo install scx-cli` does not work — prefer the pre-built binary from GitHub Releases).
