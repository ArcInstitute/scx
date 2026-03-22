#!/usr/bin/env Rscript
# BPCells benchmark: read h5ad, convert to BPCells format, iterate batches.
# Outputs a single JSON line to stdout with timing results.
#
# Usage:
#   Rscript benchmark_bpcells.R <h5ad_path> [batch_size]
#
# Requirements:
#   - BPCells R package
#   - hdf5r R package (for reading h5ad)

suppressPackageStartupMessages({
  library(BPCells)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript benchmark_bpcells.R <h5ad_path> [batch_size]")
}

h5ad_path <- args[1]
batch_size <- if (length(args) >= 2) as.integer(args[2]) else 1024L

if (!file.exists(h5ad_path)) {
  stop(paste("File not found:", h5ad_path))
}

# Derive dataset name from filename (strip path and extension)
dataset_name <- tools::file_path_sans_ext(basename(h5ad_path))

# Temporary directory for BPCells on-disk format
bpcells_dir <- tempfile(pattern = "bpcells_bench_")

# --- Step 1: Read h5ad and convert to BPCells ---
# BPCells can open H5AD files directly via open_matrix_anndata_hdf5
cat_to_stderr <- function(...) cat(..., file = stderr())

cat_to_stderr("Reading h5ad: ", h5ad_path, "\n")
mat <- open_matrix_anndata_hdf5(h5ad_path)

# Get dimensions before conversion
n_obs <- nrow(mat)
n_vars <- ncol(mat)
n_batches <- ceiling(n_obs / batch_size)

cat_to_stderr("Matrix: ", n_obs, " x ", n_vars, "\n")
cat_to_stderr("Batch size: ", batch_size, ", n_batches: ", n_batches, "\n")

# Write to BPCells on-disk format for efficient iteration
cat_to_stderr("Converting to BPCells on-disk format...\n")
write_matrix_dir(mat, dir = bpcells_dir)
bpc_mat <- open_matrix_dir(bpcells_dir)

# --- Step 2: Iterate through batches, measuring time ---
cat_to_stderr("Starting batch iteration...\n")

ttfb <- NA_real_
total_start <- proc.time()["elapsed"]

for (i in seq_len(n_batches)) {
  row_start <- (i - 1L) * batch_size + 1L
  row_end <- min(i * batch_size, n_obs)

  # Slice rows and materialize to dense matrix (simulates ML data loading)
  batch <- bpc_mat[row_start:row_end, , drop = FALSE]
  # Force materialization by converting to dgCMatrix
  batch_mat <- as(batch, "dgCMatrix")

  if (i == 1L) {
    ttfb <- as.numeric(proc.time()["elapsed"] - total_start)
    cat_to_stderr("First batch loaded in ", round(ttfb, 4), "s\n")
  }
}

total_time <- as.numeric(proc.time()["elapsed"] - total_start)
batches_per_sec <- n_batches / total_time

# --- Step 3: Cleanup ---
unlink(bpcells_dir, recursive = TRUE)

# --- Step 4: Output JSON ---
json <- sprintf(
  '{"benchmark": "bpcells", "dataset": "%s", "batches_per_sec": %.2f, "ttfb_s": %.4f, "n_batches": %d, "total_time_s": %.4f, "n_obs": %d, "n_vars": %d, "batch_size": %d}',
  dataset_name, batches_per_sec, ttfb, n_batches, total_time, n_obs, n_vars, batch_size
)
cat(json, "\n")
