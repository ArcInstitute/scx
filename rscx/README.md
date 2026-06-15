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

# File info and validation
scx_info("experiment.scx")
scx_validate("experiment.scx")
```

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

## Build Notes

The package compiles a Rust shared library during installation. The first
install takes several minutes while Cargo downloads and compiles dependencies.
Subsequent installs are faster thanks to Cargo's build cache.

The Rust crate depends on sibling workspace crates (`scx-format`, `scx-codec`,
`scx-sparse`, `scx-engine`, `scx-ops`) via path dependencies. These are all
compiled automatically during `R CMD INSTALL`.
