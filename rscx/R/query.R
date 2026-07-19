#' Start a query pipeline from an ScxExperiment
#'
#' @param exp An ScxExperiment object (from \code{scx_open()}).
#' @return An RQueryPipeline object.
#' @export
#' @examples
#' \dontrun{
#' hvg_indices <- which(scx_highly_variable_genes(exp)$highly_variable)  # 1-based
#' result <- scx_open("experiment.scx") |>
#'   scx_query() |>
#'   filter_obs("tissue == 'lung'") |>
#'   select_genes(hvg_indices) |>   # 1-based gene indices
#'   with_normalize(target_sum = 1e4) |>
#'   with_log1p() |>
#'   collect()
#'
#' dgc <- result$to_dgcmatrix()
#' seu <- result$to_seurat()
#' sce <- result$to_sce()
#' }
scx_query <- function(exp) {
  exp$query()
}

#' Filter observations (cells) by a predicate expression
#'
#' @param pipeline An RQueryPipeline object.
#' @param expr Character string predicate, e.g. \code{"tissue == 'lung'"}.
#' @return A new RQueryPipeline object (for pipe chaining).
#' @export
filter_obs <- function(pipeline, expr) {
  pipeline$filter_obs(expr)
}

#' Filter variables (genes) by a predicate expression
#'
#' @param pipeline An RQueryPipeline object.
#' @param expr Character string predicate.
#' @return A new RQueryPipeline object (for pipe chaining).
#' @export
filter_var <- function(pipeline, expr) {
  pipeline$filter_var(expr)
}

#' Select specific gene indices for projection
#'
#' @param pipeline An RQueryPipeline object.
#' @param indices Numeric vector of 1-based gene indices (R convention, matching
#'   the \code{[} operator). For example, obtain them from
#'   \code{which(scx_highly_variable_genes(...)$highly_variable)}. Indices must be
#'   \code{>= 1}; \code{0} / negatives raise an error.
#' @return A new RQueryPipeline object (for pipe chaining).
#' @export
select_genes <- function(pipeline, indices) {
  indices <- as.numeric(indices)
  if (anyNA(indices)) {
    stop("NA gene indices are not supported", call. = FALSE)
  }
  # R is 1-based (like the `[` operator); reject 0 / negatives before the
  # conversion to the 0-based core index.
  if (length(indices) && any(indices < 1)) {
    stop("gene indices must be >= 1 (1-based); 0 / negative indices are not supported",
         call. = FALSE)
  }
  pipeline$select_genes(indices - 1)
}

#' Enable total-count normalization
#'
#' @param pipeline An RQueryPipeline object.
#' @param target_sum Target sum for normalization (e.g. 1e4).
#' @return A new RQueryPipeline object (for pipe chaining).
#' @export
with_normalize <- function(pipeline, target_sum) {
  pipeline$with_normalize(target_sum)
}

#' Enable log1p transformation
#'
#' @param pipeline An RQueryPipeline object.
#' @return A new RQueryPipeline object (for pipe chaining).
#' @export
with_log1p <- function(pipeline) {
  pipeline$with_log1p()
}

#' Limit the number of returned cells
#'
#' @param pipeline An RQueryPipeline object.
#' @param n Maximum number of cells to return.
#' @return A new RQueryPipeline object (for pipe chaining).
#' @export
limit <- function(pipeline, n) {
  pipeline$limit(n)
}

#' Execute the query pipeline and collect results
#'
#' @param pipeline An RQueryPipeline object.
#' @return An RQueryResult object.
#' @export
collect <- function(pipeline) {
  pipeline$collect()
}

#' Count matching cells without decoding the matrix
#'
#' Convenience wrapper mirroring pyscx's \code{query().count()}: runs only the
#' plan + obs-mask half of the pipeline (no expression-matrix shards are
#' decoded) and returns the number of matching cells. Returns the true match
#' count (any \code{limit()} is not applied) and does \emph{not} consume the
#' pipeline.
#'
#' Note: if both \pkg{rscx} and \pkg{dplyr} are attached, \code{count} is
#' masked — use \code{rscx::count()} / \code{dplyr::count()} to disambiguate.
#'
#' @param pipeline An RQueryPipeline object.
#' @return The number of matching cells (numeric).
#' @export
count <- function(pipeline) {
  pipeline$count()
}
