#!/usr/bin/env Rscript
#
# scDblFinder runner for the SCX doublet-interop benchmark.
#
# Reads a JSON config from stdin, runs scDblFinder on one batch, writes a CSV
# of the tool's NATIVE output columns, and emits a JSON run record on stdout.
# Same protocol as benchmarks/r_scripts/benchmark_bpcells.R.
#
# The input is the SCX file itself, read through rscx — not an exported h5ad.
# That is not a shortcut: R has no h5ad reader in this env (no zellkonverter,
# no anndata), and reading SCX directly is the shorter path Phase 6 built and
# verified. The Python runners consume the per-batch h5ad export instead, so
# between them the two ecosystems' real entry points are both covered.
#
# The output columns are deliberately the tool's own spellings
# (`scDblFinder.score` / `scDblFinder.class`) so the benchmark drives the real
# `doublet_import(tool="scdblfinder")` profile rather than pre-canonicalising
# and testing nothing.
#
# Config:
#   scx_path    path to the .scx file
#   out_csv     where to write the per-cell table
#   batch_key   obs column to filter on (or null for the whole file)
#   batch       the value of batch_key to select
#   seed        RNG seed
#
# Required R packages: jsonlite, rscx, scDblFinder, SingleCellExperiment

for (pkg in c("jsonlite", "rscx", "scDblFinder", "SingleCellExperiment")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf('{"error": "Required R package \'%s\' is not installed."}\n', pkg))
    quit(save = "no", status = 1)
  }
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(rscx)
  library(scDblFinder)
  library(SingleCellExperiment)
})

get_peak_rss_kb <- function() {
  status_file <- "/proc/self/status"
  if (!file.exists(status_file)) return(0L)
  lines <- readLines(status_file, warn = FALSE)
  hit <- grep("^VmHWM:", lines, value = TRUE)
  if (length(hit) == 0) return(0L)
  as.integer(sub("^VmHWM:\\s+", "", sub("\\s+kB$", "", hit[1])))
}

emit_result <- function(result) {
  cat(toJSON(result, auto_unbox = TRUE))
  cat("\n")
  quit(save = "no", status = 0)
}

emit_error <- function(msg) {
  cat(toJSON(list(error = msg), auto_unbox = TRUE))
  cat("\n")
  quit(save = "no", status = 1)
}

stdin_lines <- readLines("stdin", warn = FALSE)
if (length(stdin_lines) == 0) emit_error("No input received on stdin")

config <- tryCatch(
  fromJSON(paste(stdin_lines, collapse = "\n")),
  error = function(e) emit_error(paste("JSON parse error:", e$message))
)

for (field in c("scx_path", "out_csv")) {
  if (is.null(config[[field]])) emit_error(sprintf("Missing '%s' in config", field))
}

seed <- if (is.null(config$seed)) 1L else as.integer(config$seed)
set.seed(seed)

t_start <- Sys.time()

# --------------------------------------------------------------------------
# Read one batch straight out of SCX
# --------------------------------------------------------------------------
exp <- tryCatch(
  rscx::scx_open(config$scx_path),
  error = function(e) emit_error(paste("scx_open failed:", e$message))
)

q <- rscx::scx_query(exp)
if (!is.null(config$batch_key) && !is.null(config$batch)) {
  # filter_obs takes a STRING expression, not NSE (Phase 4).
  expr <- sprintf("%s == '%s'", config$batch_key, config$batch)
  q <- rscx::filter_obs(q, expr)
}

res <- tryCatch(
  rscx::collect(q),
  error = function(e) emit_error(paste("collect failed:", e$message))
)
sce <- tryCatch(
  res$to_sce(),
  error = function(e) emit_error(paste("to_sce failed:", e$message))
)

t_read <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))

if (ncol(sce) == 0) {
  emit_error(sprintf("batch '%s' selected 0 cells", as.character(config$batch)))
}
# The join key. A tool whose output cannot be joined back is worthless, and
# the failure is far cheaper to see here than as a duplicate-key error at
# import time or, worse, as scores landing on the wrong cell.
if (is.null(colnames(sce))) {
  emit_error("to_sce() produced NULL colnames — no join key for the import")
}
if (any(duplicated(colnames(sce)))) {
  emit_error(sprintf(
    "the join key is not unique within this batch (%d duplicated of %d cells)",
    sum(duplicated(colnames(sce))), ncol(sce)
  ))
}

# --------------------------------------------------------------------------
# scDblFinder
# --------------------------------------------------------------------------
t0 <- Sys.time()
sce <- tryCatch(
  scDblFinder(sce),
  error = function(e) emit_error(paste("scDblFinder failed:", e$message))
)
t_tool <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

# Which columns the tool emitted is version-dependent: 1.24 emits no
# `mostLikelyOrigin`, so a fixed column list silently drops or errors depending
# on the release. Select adaptively and report what was found.
cols <- grep("^scDblFinder\\.", colnames(colData(sce)), value = TRUE)
if (!("scDblFinder.score" %in% cols) || !("scDblFinder.class" %in% cols)) {
  emit_error(paste(
    "scDblFinder emitted neither score nor class; got:",
    paste(cols, collapse = ", ")
  ))
}

out <- as.data.frame(colData(sce)[, cols, drop = FALSE])
out$barcode <- colnames(sce)
out <- out[, c("barcode", cols), drop = FALSE]

write.csv(out, file = config$out_csv, row.names = FALSE, quote = TRUE)

n_called <- sum(as.character(out$scDblFinder.class) == "doublet")

emit_result(list(
  tool = "scdblfinder",
  batch = if (is.null(config$batch)) NA else as.character(config$batch),
  out_csv = config$out_csv,
  n_cells = ncol(sce),
  n_genes = nrow(sce),
  n_called = n_called,
  called_rate = if (ncol(sce) > 0) n_called / ncol(sce) else NA,
  score_column = "scDblFinder.score",
  call_column = "scDblFinder.class",
  emitted_columns = cols,
  read_s = t_read,
  tool_s = t_tool,
  peak_rss_kb = get_peak_rss_kb(),
  seed = seed,
  versions = list(
    scDblFinder = as.character(packageVersion("scDblFinder")),
    rscx = as.character(packageVersion("rscx")),
    R = paste0(R.version$major, ".", R.version$minor)
  )
))
