#!/usr/bin/env Rscript
#
# BPCells benchmark helper for the SCX comprehensive benchmark suite.
#
# Reads a JSON config from stdin, executes the requested operation,
# and writes a JSON result to stdout.
#
# Operations:
#   convert    — Read h5ad, write BPCells matrix directory
#   read_full  — Open BPCells dir, materialise as dgCMatrix
#   read_subset — Open BPCells dir, subset rows/cols, materialise
#
# Required R packages: jsonlite, BPCells, Matrix

# --------------------------------------------------------------------------
# Dependency check
# --------------------------------------------------------------------------
for (pkg in c("jsonlite", "BPCells", "Matrix")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required R package '%s' is not installed.", pkg), call. = FALSE)
  }
}

library(jsonlite)
library(BPCells)
library(Matrix)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

#' Read peak RSS from /proc/self/status (Linux only).
#' Returns peak RSS in kilobytes, or 0 if unavailable.
get_peak_rss_kb <- function() {
  status_file <- "/proc/self/status"
  if (!file.exists(status_file)) return(0L)
  lines <- readLines(status_file, warn = FALSE)
  vm_peak_line <- grep("^VmHWM:", lines, value = TRUE)
  if (length(vm_peak_line) == 0) return(0L)
  # Line looks like: "VmHWM:    123456 kB"
  as.integer(sub("^VmHWM:\\s+", "", sub("\\s+kB$", "", vm_peak_line[1])))
}

#' Write the result list as JSON to stdout and exit.
emit_result <- function(result) {
  cat(toJSON(result, auto_unbox = TRUE))
  cat("\n")
  quit(save = "no", status = 0)
}

#' Exit with an error JSON message.
emit_error <- function(msg) {
  cat(toJSON(list(error = msg), auto_unbox = TRUE))
  cat("\n")
  quit(save = "no", status = 1)
}

# --------------------------------------------------------------------------
# Read config from stdin
# --------------------------------------------------------------------------
stdin_lines <- readLines("stdin", warn = FALSE)
if (length(stdin_lines) == 0) {
  emit_error("No input received on stdin")
}

config <- tryCatch(
  fromJSON(paste(stdin_lines, collapse = "\n")),
  error = function(e) emit_error(paste("JSON parse error:", e$message))
)

operation <- config$operation
if (is.null(operation)) {
  emit_error("Missing 'operation' field in config")
}

# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------

if (operation == "convert") {
  # ------------------------------------------------------------------
  # Convert h5ad -> BPCells matrix directory
  # ------------------------------------------------------------------
  h5ad_path   <- config$h5ad_path
  output_path <- config$output_path

  if (is.null(h5ad_path) || is.null(output_path)) {
    emit_error("'convert' requires 'h5ad_path' and 'output_path'")
  }

  timing <- system.time({
    mat <- open_matrix_anndata_hdf5(h5ad_path)
    write_matrix_dir(mat, dir = output_path, overwrite = TRUE)
  })

  emit_result(list(
    wall_s         = as.numeric(timing["elapsed"]),
    peak_rss_kb    = get_peak_rss_kb()
  ))

} else if (operation == "read_full") {
  # ------------------------------------------------------------------
  # Read full matrix from BPCells directory -> dgCMatrix
  # ------------------------------------------------------------------
  path <- config$path
  if (is.null(path)) {
    emit_error("'read_full' requires 'path'")
  }

  timing <- system.time({
    mat <- open_matrix_dir(path)
    dense <- as(mat, "dgCMatrix")
    # Force evaluation (dim access ensures the matrix is realised)
    stopifnot(nrow(dense) > 0)
  })

  emit_result(list(
    wall_s      = as.numeric(timing["elapsed"]),
    peak_rss_kb = get_peak_rss_kb()
  ))

} else if (operation == "read_subset") {
  # ------------------------------------------------------------------
  # Read a subset of cells/genes -> dgCMatrix
  # ------------------------------------------------------------------
  path <- config$path
  if (is.null(path)) {
    emit_error("'read_subset' requires 'path'")
  }

  cell_indices <- config$cell_indices  # already 1-based from Python
  gene_indices <- config$gene_indices  # already 1-based from Python

  timing <- system.time({
    mat <- open_matrix_dir(path)

    if (!is.null(cell_indices) && !is.null(gene_indices)) {
      sub <- mat[cell_indices, gene_indices]
    } else if (!is.null(cell_indices)) {
      sub <- mat[cell_indices, ]
    } else if (!is.null(gene_indices)) {
      sub <- mat[, gene_indices]
    } else {
      sub <- mat
    }

    dense <- as(sub, "dgCMatrix")
    stopifnot(nrow(dense) > 0)
  })

  emit_result(list(
    wall_s      = as.numeric(timing["elapsed"]),
    peak_rss_kb = get_peak_rss_kb()
  ))

} else {
  emit_error(paste("Unknown operation:", operation))
}
