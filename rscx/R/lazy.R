# Lazy transform chains.
#
# `scx_lazy_transform()` / `exp$x_lazy()` return an `ScxLazyTransformed`: an
# environment with `class(self) <- "ScxLazyTransformed"` and `$`-methods
# (`n_obs`, `n_vars`, `nnz`, `normalize_total`, `log1p`, `row_scale`,
# `read_rows`, `read_row_indices`, `row_sums`, `col_sums`, `to_dgcmatrix`,
# `transform_names`) defined in R/extendr-wrappers.R. Transforms are applied on
# read (slicing / materialization), never to the whole matrix at once. The
# transform verbs are immutable: each returns a NEW handle with the op appended,
# so chains are pipe-friendly. This file adds the pipe-friendly verbs plus the
# idiomatic matrix generics (`dim`, `[`, `print`, `as.matrix`).

#' Open a lazy-transform view of an SCX file's X matrix
#'
#' Returns an \code{ScxLazyTransformed} that streams rows from disk and applies
#' a chain of preprocessing transforms (\code{normalize_total},
#' \code{log1p}, \code{row_scale}) lazily, on read — never materialising the
#' whole matrix. The R equivalent of pyscx's \code{ScxLazyTransformedDataset}.
#'
#' Build a chain with the pipe-friendly verbs and slice the result like a
#' matrix:
#' \preformatted{
#' lt <- scx_lazy_transform("atlas.scx") |>
#'   scx_normalize_total(target_sum = 1e4) |>
#'   scx_log1p()
#' lt[1:100, ]            # transformed dgCMatrix (cells x genes)
#' }
#'
#' @param path Path to the \code{.scx} file.
#' @param cache_shards Decoded-shard LRU cache size (default 128).
#' @return An \code{ScxLazyTransformed} with an empty transform chain.
#' @seealso \code{\link{scx_backed_sparse}} (untransformed backed access);
#'   \code{scx_normalize_total}, \code{scx_log1p}, \code{scx_row_scale}.
#' @export
scx_lazy_transform <- function(path, cache_shards = 128) {
  ScxLazyTransformed$.wrap(
    .Call(wrap__RLazyTransformed__new, path, as.numeric(cache_shards))
  )
}

#' Lazy transform verbs
#'
#' Append a preprocessing transform to a lazy chain. Each returns a **new**
#' \code{ScxLazyTransformed} (the input is unchanged), so they compose with the
#' pipe. Transforms are applied on read, not eagerly.
#'
#' @param x An \code{ScxLazyTransformed} (from \code{\link{scx_lazy_transform}}
#'   or \code{exp$x_lazy()}).
#' @param target_sum Per-cell target count for \code{scx_normalize_total}
#'   (default \code{1e4}, matching scanpy / pyscx). The per-cell sums used are
#'   those of the data as transformed by the chain so far.
#' @param factors Per-cell numeric scaling vector for \code{scx_row_scale}
#'   (length \code{nrow(x)}).
#' @return A new \code{ScxLazyTransformed} with the transform appended.
#' @name scx-lazy-transforms
NULL

#' @rdname scx-lazy-transforms
#' @export
scx_normalize_total <- function(x, target_sum = 1e4) {
  x$normalize_total(target_sum)
}

#' @rdname scx-lazy-transforms
#' @export
scx_log1p <- function(x) {
  x$log1p()
}

#' @rdname scx-lazy-transforms
#' @export
scx_row_scale <- function(x, factors) {
  x$row_scale(factors)
}

#' Dimensions of a lazy-transformed dataset
#'
#' @param x An \code{ScxLazyTransformed}.
#' @return Integer vector \code{c(n_obs, n_vars)} (cells, genes).
#' @export
dim.ScxLazyTransformed <- function(x) {
  as.integer(c(x$n_obs(), x$n_vars()))
}

#' Index a lazy-transformed dataset
#'
#' Reads the requested rows (cells) from disk, applies the transform chain, and
#' returns a transformed \code{dgCMatrix} of cells x genes. Same indexing rules
#' as \code{\link{[.ScxBackedSparse}}: positive-integer and logical row indices,
#' fractional ones truncated as a \code{dgCMatrix} does, \code{NA} / non-finite
#' rejected; an optional column index is applied to the returned matrix. The
#' underlying \code{$read_rows()} / \code{$read_row_indices()} methods are
#' 0-based and strict.
#'
#' @param x An \code{ScxLazyTransformed}.
#' @param i Row (cell) selector: positive integers or a logical vector of length
#'   \code{nrow(x)}. Missing selects all rows.
#' @param j Column (gene) selector applied to the returned matrix. Missing
#'   selects all columns.
#' @param ... Unused.
#' @param drop Ignored; a \code{dgCMatrix} is always returned.
#' @return A transformed \code{dgCMatrix} (cells x genes).
#' @export
`[.ScxLazyTransformed` <- function(x, i, j, ..., drop = FALSE) {
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
      stop("character row indices are not supported for ScxLazyTransformed", call. = FALSE)
    } else {
      i <- as.numeric(i)
      if (anyNA(i)) {
        stop("NA row indices are not supported for ScxLazyTransformed", call. = FALSE)
      }
      if (length(i) && !all(is.finite(i))) {
        stop("row indices must be finite; Inf / -Inf are not supported for ScxLazyTransformed",
             call. = FALSE)
      }
      # R is 1-based; reject 0 and negatives (0 would underflow to -1 below and
      # surface as a confusing out-of-bounds error from the Rust core).
      if (length(i) && any(i < 1)) {
        stop("row indices must be >= 1 (1-based); 0 / negative indices are not supported for ScxLazyTransformed",
             call. = FALSE)
      }
      # Truncate fractional subscripts to match a dgCMatrix; see
      # `[.ScxBackedSparse` for why this must precede the contiguity check.
      i <- trunc(i)
    }

    if (length(i) == 0) {
      mat <- x$read_rows(0, 0)
    } else if (length(i) > 1 && all(diff(i) == 1)) {
      mat <- x$read_rows(i[1] - 1, i[length(i)])
    } else {
      mat <- x$read_row_indices(i - 1)
    }
  }

  if (!missing(j)) {
    mat <- mat[, j, drop = FALSE]
  }
  mat
}

#' Print a lazy-transformed dataset
#'
#' @param x An \code{ScxLazyTransformed}.
#' @param ... Unused.
#' @return \code{x}, invisibly.
#' @export
print.ScxLazyTransformed <- function(x, ...) {
  tn <- x$transform_names()
  chain <- if (length(tn) == 0L) "none" else paste(tn, collapse = " -> ")
  cat(sprintf(
    "<ScxLazyTransformed: %.0f x %.0f (cells x genes), transforms: %s>\n",
    x$n_obs(), x$n_vars(), chain
  ))
  invisible(x)
}

#' Coerce a lazy-transformed dataset to a dense matrix
#'
#' Materialises the full transformed matrix; for large files prefer row slicing
#' (\code{x[i, ]}) or \code{x$to_dgcmatrix()}.
#'
#' @param x An \code{ScxLazyTransformed}.
#' @param ... Unused.
#' @return A base dense matrix (cells x genes).
#' @export
as.matrix.ScxLazyTransformed <- function(x, ...) {
  as.matrix(x$to_dgcmatrix())
}
