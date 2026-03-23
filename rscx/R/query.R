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
NULL
