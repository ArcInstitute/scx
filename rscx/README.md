# rscx — R Bindings for SCX

R package providing native access to the SCX (Sparse Cell eXpression) binary
file format via Rust. Features a lazy reader, pipe-friendly query pipeline,
Seurat v5 and SingleCellExperiment interop, and file lifecycle operations.

## Prerequisites

- **R** >= 4.2.0
- **Rust** >= 1.78 ([install via rustup](https://rustup.rs/))
- **R packages**: `Matrix`, `methods` (required); `Seurat` >= 5.0.0,
  `SingleCellExperiment` (optional, for interop)

## Installation

### From source (requires Rust toolchain)

```r
# Option 1: Using devtools (recommended)
# install.packages("devtools")
devtools::install_github("ArcInstitute/scx", subdir = "rscx")

# Option 2: Using remotes
# install.packages("remotes")
remotes::install_github("ArcInstitute/scx", subdir = "rscx")
```

### From a local clone

```bash
git clone https://github.com/ArcInstitute/scx.git
cd scx
R CMD INSTALL rscx/
```

> **Using conda R?** Build from an **activated** conda env
> (`conda activate <env>`) — or add `$CONDA_PREFIX/bin` to `PATH`. R's
> `Makeconf` points at the conda C compiler (e.g. `x86_64-conda-linux-gnu-cc`),
> which is only on `PATH` when the env is active; otherwise the link step fails
> with `x86_64-conda-linux-gnu-cc: not found`. Running `<env>/bin/R CMD INSTALL`
> by full path without activating does **not** fix it.

### Pre-built binaries (no Rust needed)

```r
# Via r-universe (when available)
install.packages("rscx", repos = "https://arcinstitute.r-universe.dev")
```

## Quick Start

```r
library(rscx)

# Open a file (lazy, no data loaded yet)
exp <- scx_open("experiment.scx")
exp$n_obs()     # number of cells
exp$n_vars()    # number of genes

# Read metadata
obs <- exp$obs()     # R data.frame
var <- exp$var()     # R data.frame

# Read expression matrix as dgCMatrix
dgc <- exp$x_matrix()

# Lazy query pipeline (pipe-friendly)
result <- scx_open("experiment.scx") |>
  scx_query() |>
  filter_obs("tissue == 'lung'") |>
  select_genes(hvg_indices) |>
  with_normalize(target_sum = 1e4) |>
  with_log1p() |>
  collect()

# Convert to Seurat v5
seu <- result$to_seurat()

# Convert to SingleCellExperiment
sce <- result$to_sce()

# File operations
scx_append("atlas.scx", "new_batch.scx")
scx_delete("experiment.scx", c(0L, 5L, 10L))
scx_compact("experiment.scx", "clean.scx")
scx_rollback("experiment.scx")
scx_merge(c("batch1.scx", "batch2.scx"), "atlas.scx")

# ...with options (mirroring pyscx):
scx_merge(c("b1.scx", "b2.scx"), "atlas.scx",
          assume_identical_var = TRUE, uns_policy = "namespace",
          index_obs = "cell_type")
scx_append("atlas.scx", "rna_batch.scx", codec = "zstd", shard_size = 32768L,
           modality = "rna")               # modality = a name on a multimodal file
scx_compact("experiment.scx", "clean.scx", index_obs = "cell_type",
            reshape_obs = TRUE)

# File info and validation
scx_info("experiment.scx")
scx_validate("experiment.scx")
```

`scx_merge` / `scx_compact` / `scx_append` accept the same option set as pyscx
(predicate-index columns via `index_obs`/`index_var`/`index_preset`, `uns_policy`
and `assume_identical_var`/`sort_by` on merge, `codec`/`shard_size`/`modality` on
append, `reshape_obs` on compact). `shard_target_rows` is inherited from the
input; CSC sidecars are dropped by these ops (rebuild separately).

## Codecs and framing on import

`from_seurat()` / `from_sce()` / `from_mae()` take the same `codec` intent axis
as `pyscx.from_anndata` and the `scx` CLI:

```r
from_seurat(seu, "auto.scx")                       # default: cost-aware adaptive
from_seurat(seu, "fast.scx",    codec = "fast")     # decode-optimized heuristic
from_seurat(seu, "compact.scx", codec = "compact")  # size-optimized (adaptive ShufDeltaZstd)
from_seurat(seu, "z.scx",       codec = "zstd")     # explicit codec
from_seurat(seu, "u.scx",       row_group_rows = 0L)  # opt out of row-group framing
```

`codec` accepts `"auto"` (default), `"fast"`, `"compact"`, or an explicit
`"none"`/`"scx1"`/`"zstd"`/`"lz4"`/`"pcodec"`/`"shufdelta"`, resolved by the same
`scx_format::resolve_codec` used everywhere else. `row_group_rows` (default 256)
controls row-group framing; the defaults produce v4-framed files matching
pyscx/CLI.

**Parity note.** A few pyscx/CLI write-side knobs are intentionally not exposed
in R (tracked in `ROADMAP.md` § 3.3): streaming-convert threading
(`reader_threads`/`writer_queue_depth`), grouped-write args (`group_by` /
`reference` — R has the F2 *reads* `read_group`/`read_reference` only), the accel
`device=` selector (rscx accelerators are CPU-only), and the byte/nnz-aware
framing knob `row_group_target_nnz` (R exposes `row_group_rows` only). Use the
`scx` CLI or pyscx when you need those.

## Backed (out-of-core) access

`exp$x_matrix()` materialises the whole matrix in memory. For atlas-scale files
that don't fit in RAM, open a **backed** view instead: it reads rows from disk on
demand with a shard-level LRU cache, and behaves like a read-only sparse matrix.

```r
exp <- scx_open("atlas.scx")
bsd <- exp$x_backed()                  # lazy; no data read yet
# (or, standalone:)  bsd <- scx_backed_sparse("atlas.scx")

dim(bsd)                               # c(n_obs, n_vars), so nrow()/ncol() work
first100 <- bsd[1:100, ]               # dgCMatrix of those cells x all genes
subset   <- bsd[c(5, 1, 3), 1:10]      # arbitrary rows (in order) x first 10 genes
mask_rows <- bsd[exp$obs()$tissue == "lung", ]   # logical row index

gene_totals <- bsd$col_sums()          # streamed aggregations, no full decode
cell_totals <- bsd$row_sums()
```

Row indices are 1-based (R convention); positive-integer and logical indices are
supported (negative/character row indices are not). A column index is applied to
the returned `dgCMatrix`. Use `bsd$to_dgcmatrix()` / `as(bsd, "dgCMatrix")` to
materialise the full matrix when you do want it all in memory.

### Lazy transform chains

`scx_lazy_transform()` (or `exp$x_lazy()`) layers a chain of preprocessing
transforms — `normalize_total`, `log1p`, `row_scale` — on top of the backed
reader. The transforms are applied **on read**, so the canonical
normalize → log1p pipeline runs out-of-core: nothing is materialised until you
slice. The verbs are pipe-friendly and immutable (each returns a new handle):

```r
lt <- scx_lazy_transform("atlas.scx") |>
  scx_normalize_total(target_sum = 1e4) |>
  scx_log1p()

lt[1:100, ]          # transformed dgCMatrix (cells x genes), computed on demand
lt$col_sums()        # streamed over the transformed data, no full decode
as(lt, "dgCMatrix")  # materialise the whole transformed matrix when needed
```

`scx_row_scale(lt, factors)` multiplies each cell by a per-cell factor (length
`nrow(lt)`). Indexing and coercion behave the same as the backed view above.

## Getting data out of a query

`collect()` returns an `RQueryResult`. Because it is an extendr object, its
extraction API is a set of `$`-methods (not S3 generics), so `methods()` won't
list them — they are:

| Method | Returns |
| --- | --- |
| `res$to_dgcmatrix()` | `Matrix::dgCMatrix` (cells × genes) |
| `res$to_sce()` | `SingleCellExperiment` (needs the package) |
| `res$to_seurat()` | `Seurat` v5 object (needs the package) |
| `res$obs()` / `res$var()` | cell / gene metadata `data.frame` |
| `res$n_obs()` / `res$n_vars()` / `res$nnz()` | dimensions (numeric) |

For convenience, thin S3/S4 generics are also registered over `RQueryResult`,
so the idiomatic R verbs work too:

```r
res <- scx_open("experiment.scx") |> scx_query() |> collect()

dim(res)                              # c(n_obs, n_vars)
m   <- as.matrix(res)                 # dense base matrix
df  <- as.data.frame(res)             # obs (cell) metadata
dgc <- as(res, "dgCMatrix")           # sparse matrix
sce <- as(res, "SingleCellExperiment")# when SingleCellExperiment is installed
```

The query builder also has `count()` — a one-liner to get just the matching
cell count without decoding the expression matrix (plan + obs-mask only):

```r
n <- scx_open("experiment.scx") |> scx_query() |>
  filter_obs("tissue == 'lung'") |> count()
```

If both `rscx` and `dplyr` are attached, `count` is masked — use
`rscx::count()` / `dplyr::count()` to disambiguate.

> **Note:** an `RQueryResult` is consumed by the first extraction call, so call
> one of the above once per `collect()`.

## Analysis accelerators

CPU-native, pipe-friendly front ends over the same Rust kernels the Python
accelerators use. Each accepts a `Seurat` object (results written into the
expected slot, object returned invisibly) **or** a raw genes × cells
`dgCMatrix` (the raw result is returned, no Seurat needed):

```r
obj <- scx_highly_variable_genes(obj)            # seurat_v3 HVG
obj <- scx_pca(obj)                              # randomized / covariance PCA
obj <- scx_neighbors(obj, dims = 1:30)           # HNSW kNN graph
obj <- scx_umap(obj, dims = 1:30)                # UMAP
obj <- scx_leiden(obj, resolution = 1.0)         # Leiden clustering
de  <- scx_rank_genes_groups(obj, group.by = "seurat_clusters")  # Wilcoxon DE

# Gene-set scoring (scanpy score_genes analog: "control" / "mean" / "zscore"):
obj <- scx_score_genes(obj, gene_list = c("CD3D", "CD3E", "CD8A"),
                       score_name = "t_cell_score")

# Pseudobulk aggregation (cells -> group x gene; "sum" or "mean"):
pb <- scx_pseudobulk(obj, group_by = c("condition", "donor"), method = "sum")
pb$counts    # genes x pseudobulk-samples matrix
pb$samples   # per-sample metadata (groupby columns + n_cells)

# Pseudobulk DE: Rust-native negative-binomial GLM (DESeq2-style, CPU-only).
# Needs >=2 replicates per condition, so group_by includes a replicate column
# (donor/batch), and raw counts (layer = "counts"):
de <- scx_pseudobulk_dex(obj, group_by = c("condition", "donor"),
                         test_col = "condition", reference = "ctrl")
# data.frame: gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj,
#             target, reference  (one block per non-reference level)

# Or run the NB-GLM directly on a pre-aggregated counts matrix + design:
de2 <- scx_nb_glm(pb$counts, model.matrix(~ condition, pb$samples))
```

For no-replicate or log-normalized data use `scx_rank_genes_groups()` (Wilcoxon)
instead of the NB-GLM. `scx_harmony_integrate()` / `RunHarmony_scx()` and
`scx_compute_lisi()` round out the set. All paths are CPU-only (rscx links no
GPU feature).

## Build Notes

The package compiles a Rust shared library during installation. The first
install takes several minutes while Cargo downloads and compiles dependencies.
Subsequent installs are faster thanks to Cargo's build cache.

The Rust crate depends on sibling workspace crates (`scx-format`, `scx-codec`,
`scx-sparse`, `scx-engine`, `scx-ops`) via path dependencies. These are all
compiled automatically during `R CMD INSTALL`.
