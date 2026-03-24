#' @title Pipe-friendly SCX query API
#' @description
#' Build lazy queries using R's native pipe operator (|>).
#'
#' @examples
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
#'
#' # Metadata getters work even after conversion:
#' result$n_obs()
#' result$n_vars()
#' result$skipped_shards()

#' Start a query pipeline from an ScxExperiment
#'
#' @param exp An ScxExperiment object (from scx_open())
#' @return An RQueryPipeline object
#' @export
scx_query <- function(exp) {
  exp$query()
}
