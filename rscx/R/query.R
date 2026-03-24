#' Start a query pipeline from an ScxExperiment
#'
#' @param exp An ScxExperiment object (from \code{scx_open()}).
#' @return An RQueryPipeline object.
#' @export
#' @examples
#' \dontrun{
#' result <- scx_open("experiment.scx") |>
#'   scx_query() |>
#'   filter_obs("tissue == 'lung'") |>
#'   select_genes(hvg_indices) |>
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
#' @param indices Integer vector of 0-based gene indices.
#' @return A new RQueryPipeline object (for pipe chaining).
#' @export
select_genes <- function(pipeline, indices) {
  pipeline$select_genes(indices)
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
