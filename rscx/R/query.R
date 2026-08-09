#' Start a query pipeline from an ScxExperiment
#'
#' @param exp An ScxExperiment object (from \code{scx_open()}).
#' @param modality Optional modality NAME to scope the query to one modality of
#'   a multimodal file (X / \code{select_genes} / \code{filter_var} resolve
#'   against that modality's var; the obs predicate stays on the shared global
#'   obs axis). On a multimodal file a modality is required (omitting it errors);
#'   an unknown name errors. Omit (\code{NULL}) on single-modality files.
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
#'
#' # Multimodal: scope to one modality.
#' rna <- scx_open("cite.scx") |>
#'   scx_query(modality = "rna") |>
#'   filter_obs("cell_type == 'T cell'") |>
#'   collect()
#' }
scx_query <- function(exp, modality = NULL) {
  exp$query(modality)
}

#' Filter observations (cells) by a predicate expression
#'
#' @param pipeline An RQueryPipeline object.
#' @param expr Character string predicate, e.g. \code{"tissue == 'lung'"}.
#' @return A new RQueryPipeline object (for pipe chaining). The input pipeline is
#'   handed on to the returned object; on error nothing is handed on and the
#'   input pipeline is left unchanged and still usable.
#' @export
filter_obs <- function(pipeline, expr) {
  pipeline$filter_obs(expr)
}

#' Filter variables (genes) by a predicate expression
#'
#' @param pipeline An RQueryPipeline object.
#' @param expr Character string predicate.
#' @return A new RQueryPipeline object (for pipe chaining). The input pipeline is
#'   handed on to the returned object; on error nothing is handed on and the
#'   input pipeline is left unchanged and still usable.
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
#' @return A new RQueryPipeline object (for pipe chaining). The input pipeline is
#'   handed on to the returned object; on error nothing is handed on and the
#'   input pipeline is left unchanged and still usable.
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
  # Reject fractional indices explicitly rather than letting the extendr wrapper's
  # `as.integer` silently truncate them.
  if (length(indices) && any(indices != floor(indices))) {
    stop("non-integer gene index: gene indices must be whole numbers", call. = FALSE)
  }
  pipeline$select_genes(indices - 1)
}

#' Enable total-count normalization
#'
#' @param pipeline An RQueryPipeline object.
#' @param target_sum Target sum for normalization (e.g. 1e4).
#' @return A new RQueryPipeline object (for pipe chaining). The input pipeline is
#'   handed on to the returned object; on error nothing is handed on and the
#'   input pipeline is left unchanged and still usable.
#' @export
with_normalize <- function(pipeline, target_sum) {
  pipeline$with_normalize(target_sum)
}

#' Enable log1p transformation
#'
#' @param pipeline An RQueryPipeline object.
#' @return A new RQueryPipeline object (for pipe chaining). The input pipeline is
#'   handed on to the returned object; on error nothing is handed on and the
#'   input pipeline is left unchanged and still usable.
#' @export
with_log1p <- function(pipeline) {
  pipeline$with_log1p()
}

#' Limit the number of returned cells
#'
#' @param pipeline An RQueryPipeline object.
#' @param n Maximum number of cells to return.
#' @return A new RQueryPipeline object (for pipe chaining). The input pipeline is
#'   handed on to the returned object; on error nothing is handed on and the
#'   input pipeline is left unchanged and still usable.
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
