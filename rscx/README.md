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

## Build Notes

The package compiles a Rust shared library during installation. The first
install takes several minutes while Cargo downloads and compiles dependencies.
Subsequent installs are faster thanks to Cargo's build cache.

The Rust crate depends on sibling workspace crates (`scx-format`, `scx-codec`,
`scx-sparse`, `scx-engine`, `scx-ops`) via path dependencies. These are all
compiled automatically during `R CMD INSTALL`.
