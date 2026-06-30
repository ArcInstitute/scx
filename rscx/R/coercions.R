# S3 / S4 dispatch over the RQueryResult extraction surface.
#
# The extendr wrapper for a collected query is an environment with
# `class(self) <- "RQueryResult"` and `$`-methods (`to_dgcmatrix`, `to_sce`,
# `to_seurat`, `obs`, `var`, `n_obs`, `n_vars`, `nnz`). Those are discoverable
# only via `ls(res)` — `methods(class = "RQueryResult")` returns `character(0)`.
# This file layers the idiomatic R generics on top so users can reach the data
# the way they expect (`as.matrix(res)`, `as.data.frame(res)`, `dim(res)`, and
# `as(res, "SingleCellExperiment")`), without changing the Rust surface.

#' Dimensions of a query result
#'
#' @param x An \code{RQueryResult} (from \code{collect()}).
#' @return Integer-like vector \code{c(n_obs, n_vars)} (cells, genes).
#' @export
dim.RQueryResult <- function(x) {
  c(x$n_obs(), x$n_vars())
}

#' Coerce a query result to a dense matrix
#'
#' Densifies the sparse result; for large results prefer
#' \code{res$to_dgcmatrix()} or \code{as(res, "dgCMatrix")}.
#'
#' @param x An \code{RQueryResult} (from \code{collect()}).
#' @param ... Unused.
#' @return A base dense matrix (cells x genes).
#' @export
as.matrix.RQueryResult <- function(x, ...) {
  as.matrix(x$to_dgcmatrix())
}

#' Coerce a query result's cell metadata to a data.frame
#'
#' @param x An \code{RQueryResult} (from \code{collect()}).
#' @param row.names,optional Accepted for S3 consistency with the
#'   \code{base::as.data.frame} generic; ignored (obs already has row names).
#' @param ... Unused.
#' @return A \code{data.frame} of obs (cell) metadata.
#' @export
as.data.frame.RQueryResult <- function(x, row.names = NULL, optional = FALSE, ...) {
  x$obs()
}

# Register S4 coercions so `as(res, <class>)` works. The single-cell containers
# are optional (DESCRIPTION Suggests), so bind them only when present — done in
# .onLoad rather than at namespace-build time so a missing Suggests dependency
# is never a hard error.
.onLoad <- function(libname, pkgname) {
  methods::setOldClass("RQueryResult")
  methods::setOldClass("ScxBackedSparse")
  methods::setOldClass("ScxLazyTransformed")

  # Matrix is an Imports dependency, so dgCMatrix is always available.
  methods::setAs(
    "RQueryResult", "dgCMatrix",
    function(from) from$to_dgcmatrix()
  )
  methods::setAs(
    "ScxBackedSparse", "dgCMatrix",
    function(from) from$to_dgcmatrix()
  )
  methods::setAs(
    "ScxLazyTransformed", "dgCMatrix",
    function(from) from$to_dgcmatrix()
  )

  if (requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    methods::setAs(
      "RQueryResult", "SingleCellExperiment",
      function(from) from$to_sce()
    )
  }

  if (requireNamespace("SeuratObject", quietly = TRUE) ||
    requireNamespace("Seurat", quietly = TRUE)) {
    methods::setAs(
      "RQueryResult", "Seurat",
      function(from) from$to_seurat()
    )
  }

  invisible(NULL)
}
