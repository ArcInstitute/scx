# Backed (lazy, out-of-core) sparse access.
#
# `scx_backed_sparse()` / `exp$x_backed()` return an `ScxBackedSparse`: an
# environment with `class(self) <- "ScxBackedSparse"` and `$`-methods
# (`n_obs`, `n_vars`, `nnz`, `read_rows`, `read_row_indices`, `row_sums`,
# `col_sums`, `to_dgcmatrix`) defined in R/extendr-wrappers.R. This file layers
# the idiomatic R matrix generics (`dim`, `[`, `print`, `as.matrix`) on top so
# the handle behaves like a read-only sparse matrix that streams rows from disk.

#' Open a backed (lazy) sparse view of an SCX file's X matrix
#'
#' Returns an \code{ScxBackedSparse} that reads rows from disk on demand with a
#' shard-level LRU cache, instead of materialising the whole matrix in memory
#' (\code{\link{scx_open}}\code{(path)$x_matrix()}). Index it like a matrix:
#' \code{bsd[1:100, ]} returns a \code{dgCMatrix} of the requested cells (rows)
#' by all genes (columns). This is the R equivalent of pyscx's
#' \code{to_anndata(backed=TRUE)} X.
#'
#' @param path Path to the \code{.scx} file.
#' @param cache_shards Number of decoded CSR shards to keep in the LRU cache
#'   (default 128). Larger values trade memory for fewer re-decodes on repeated
#'   or sequential access.
#' @return An \code{ScxBackedSparse} object.
#' @seealso \code{\link{scx_open}}; the \code{$x_backed()} method on an
#'   \code{ScxExperiment} is the equivalent accessor for an already-open file.
#' @export
#' @examples
#' \dontrun{
#' bsd <- scx_backed_sparse("experiment.scx")
#' dim(bsd)
#' first100 <- bsd[1:100, ]      # dgCMatrix, cells x genes
#' subset <- bsd[c(5, 1, 3), 1:10]
#' gene_totals <- bsd$col_sums()  # streamed, no full materialization
#' }
scx_backed_sparse <- function(path, cache_shards = 128) {
  ScxBackedSparse$.wrap(
    .Call(wrap__RBackedSparse__new, path, as.numeric(cache_shards))
  )
}

#' Dimensions of a backed sparse dataset
#'
#' @param x An \code{ScxBackedSparse}.
#' @return Integer vector \code{c(n_obs, n_vars)} (cells, genes). Enables
#'   \code{nrow()} / \code{ncol()}.
#' @export
dim.ScxBackedSparse <- function(x) {
  as.integer(c(x$n_obs(), x$n_vars()))
}

#' Index a backed sparse dataset
#'
#' Reads the requested rows (cells) from disk on demand and returns a
#' \code{dgCMatrix} of cells x genes. Row indices select cells; the optional
#' column index selects genes (applied to the returned matrix). Rows are
#' returned in the requested order.
#'
#' Positive-integer and logical row indices are supported. Negative and
#' character row indices are not (they raise an error); subset by computing the
#' positive indices yourself, e.g. \code{bsd[setdiff(seq_len(nrow(bsd)), drop), ]}.
#'
#' @param x An \code{ScxBackedSparse}.
#' @param i Row (cell) selector: positive integers or a logical vector of length
#'   \code{nrow(x)}. Missing selects all rows.
#' @param j Column (gene) selector, applied to the returned matrix. Missing
#'   selects all columns.
#' @param ... Unused.
#' @param drop Ignored; a \code{dgCMatrix} is always returned (sparse, never
#'   dropped to a vector).
#' @return A \code{dgCMatrix} (cells x genes).
#' @export
`[.ScxBackedSparse` <- function(x, i, j, ..., drop = FALSE) {
  n_obs <- x$n_obs()

  if (missing(i)) {
    mat <- x$to_dgcmatrix()
  } else {
    if (is.logical(i)) {
      if (length(i) != n_obs) {
        stop(sprintf(
          "logical row index has length %d but nrow is %.0f",
          length(i), n_obs
        ), call. = FALSE)
      }
      i <- which(i)
    } else if (is.character(i)) {
      stop("character row indices are not supported for ScxBackedSparse", call. = FALSE)
    } else {
      i <- as.numeric(i)
      if (anyNA(i)) {
        stop("NA row indices are not supported for ScxBackedSparse", call. = FALSE)
      }
      # R is 1-based; reject 0 and negatives (0 would underflow to -1 below and
      # surface as a confusing out-of-bounds error from the Rust core).
      if (length(i) && any(i < 1)) {
        stop("row indices must be >= 1 (1-based); 0 / negative indices are not supported for ScxBackedSparse",
             call. = FALSE)
      }
    }

    if (length(i) == 0) {
      # Empty selection: 0-row slice with the right column count.
      mat <- x$read_rows(0, 0)
    } else if (length(i) > 1 && all(diff(i) == 1)) {
      # Contiguous ascending run → single half-open range read (fast path).
      mat <- x$read_rows(i[1] - 1, i[length(i)])
    } else {
      # Arbitrary / reordered / single row → fancy gather (0-based).
      mat <- x$read_row_indices(i - 1)
    }
  }

  if (!missing(j)) {
    mat <- mat[, j, drop = FALSE]
  }
  mat
}

#' Print a backed sparse dataset
#'
#' @param x An \code{ScxBackedSparse}.
#' @param ... Unused.
#' @return \code{x}, invisibly.
#' @export
print.ScxBackedSparse <- function(x, ...) {
  cat(sprintf(
    "<ScxBackedSparse: %.0f x %.0f (cells x genes), backed>\n",
    x$n_obs(), x$n_vars()
  ))
  invisible(x)
}

#' Coerce a backed sparse dataset to a dense matrix
#'
#' Materialises the full matrix and densifies it; for large files prefer row
#' slicing (\code{x[i, ]}) or \code{x$to_dgcmatrix()}.
#'
#' @param x An \code{ScxBackedSparse}.
#' @param ... Unused.
#' @return A base dense matrix (cells x genes).
#' @export
as.matrix.ScxBackedSparse <- function(x, ...) {
  as.matrix(x$to_dgcmatrix())
}
