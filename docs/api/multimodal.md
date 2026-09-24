# Multimodal API

> Part of the [SCX API reference](README.md). For the layout and task-level guide, see
[docs/multimodal.md](../multimodal.md).

v2 SCX files carry multiple modalities (RNA + ADT + ATAC + …) routed
via a 1-byte `modality_id` stamped on each catalog entry. See
[docs/format.md § 13](../format.md#13-multimodal-extension-optional) for the
on-disk layout and [docs/multimodal.md](../multimodal.md) for the
end-to-end usage guide.

## `ModalityType` enum (`scx-format/src/modality.rs`)

```
ModalityType::Rna           = 0
ModalityType::Protein       = 1   // ADT
ModalityType::Atac          = 2
ModalityType::Spatial       = 3
ModalityType::Methylation   = 4
ModalityType::Custom        = 255
```

Drives per-modality codec selection (see
[docs/codec.md § Per-modality codec defaults](../codec.md#8a-per-modality-codec-defaults)).

## `ModalityInfo` struct

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

## `ScxReader` — multimodal accessors

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

## `ScxWriter` — multimodal writers

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

## Codec selection — `select_codec_for_modality`

`select_codec_for_modality(raw_values, value_encoding, modality_type)`
extends `select_codec` with per-modality routing:
RNA / Custom / Methylation / Spatial → delegate to `select_codec`;
Protein/ADT → Zstd for integers, Pcodec for floats; ATAC → Zstd for
binary peak presence (sample max ≤ 1) else Lz4Shuffle, Pcodec for
floats. See [docs/codec.md § Per-modality codec defaults](../codec.md#8a-per-modality-codec-defaults).

## Per-modality `BackedCscReader`

`BackedCscReader::for_modality(reader, modality_id, cache_shards)` —
builds a column-major reader scoped to one modality so multimodal
training (e.g. totalVI) can hold separate caches per modality without
LRU thrashing across modalities.

## Modality-scoped queries — `QueryPipeline` (`scx-engine`)

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

## Python (`pyscx`)

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
    [docs/multimodal.md § 3.4](../multimodal.md#34-modality-scoped-queries--querymodality).
- `pyscx.MultimodalTrainingDataset(path, modalities=[…], …)` — yields
  per-batch dicts `{"X": {modality_name: ndarray}, "obs": {...},
  "cell_indices": ndarray}` (or tuples in `return_dict=False` mode). Pins a
  uniform effective `batch_size`/`shard_group_size` across modalities so a
  wide modality (e.g. ATAC) can't desync the per-modality batching; pass a
  larger `max_memory_mb` to lift the pinned batch. See
  [docs/multimodal.md § Training](../multimodal.md#33-training--pyscxmultimodaltrainingdataset).
- Backward compat: `pyscx.TrainingDataset(path)` on a multimodal file
  emits `UserWarning` and falls back to the alphabetically-first
  modality. Pass `modality="rna"` explicitly to suppress.

## R (`rscx`)

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

### Backed (out-of-core) sparse access

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

### Lazy transform chains

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

### Analysis accelerators (`scx-accel`, CPU)

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

### File operations (options)

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
a CSC sidecar is carried — rebuilt from the output's X in the same pass iff an
input had one (rscx exposes no `csc` argument; `append` still drops it).

## CLI surface

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
