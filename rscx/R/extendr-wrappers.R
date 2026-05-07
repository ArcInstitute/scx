# nolint start
# Extendr R class wrappers for rscx Rust structs.
# These wrap .Call() routines into R class objects.
# Free functions (scx_open, scx_info, scx_append, etc.) are defined
# in R/scx.R, R/ops.R with roxygen2 documentation.

# ── ScxExperiment class ─────────────────────────────────────────
#' @export
ScxExperiment <- new.env(parent = emptyenv())
ScxExperiment$new <- function(path) {
  ptr <- .Call(wrap__ScxExperiment__new, path)
  self <- new.env(parent = emptyenv())
  self$.ptr <- ptr
  self$n_obs <- function() .Call(wrap__ScxExperiment__n_obs, self$.ptr)
  self$n_vars <- function() .Call(wrap__ScxExperiment__n_vars, self$.ptr)
  self$nnz <- function() .Call(wrap__ScxExperiment__nnz, self$.ptr)
  self$shard_count <- function() .Call(wrap__ScxExperiment__shard_count, self$.ptr)
  self$obs <- function() .Call(wrap__ScxExperiment__obs, self$.ptr)
  self$var <- function() .Call(wrap__ScxExperiment__var, self$.ptr)
  self$x_matrix <- function() .Call(wrap__ScxExperiment__x_matrix, self$.ptr)
  self$layer <- function(name) .Call(wrap__ScxExperiment__layer, self$.ptr, name)
  self$layer_names <- function() .Call(wrap__ScxExperiment__layer_names, self$.ptr)
  self$query <- function() {
    ptr <- .Call(wrap__ScxExperiment__query, self$.ptr)
    RQueryPipeline$.wrap(ptr)
  }
  class(self) <- "ScxExperiment"
  self
}
class(ScxExperiment) <- "ScxExperiment__class"

# ── RQueryPipeline class ────────────────────────────────────────
#' @export
RQueryPipeline <- new.env(parent = emptyenv())
RQueryPipeline$.wrap <- function(ptr) {
  self <- new.env(parent = emptyenv())
  self$.ptr <- ptr
  self$filter_obs <- function(expr) RQueryPipeline$.wrap(.Call(wrap__RQueryPipeline__filter_obs, self$.ptr, expr))
  self$filter_var <- function(expr) RQueryPipeline$.wrap(.Call(wrap__RQueryPipeline__filter_var, self$.ptr, expr))
  self$select_genes <- function(indices) RQueryPipeline$.wrap(.Call(wrap__RQueryPipeline__select_genes, self$.ptr, as.integer(indices)))
  self$with_normalize <- function(target_sum) RQueryPipeline$.wrap(.Call(wrap__RQueryPipeline__with_normalize, self$.ptr, target_sum))
  self$with_log1p <- function() RQueryPipeline$.wrap(.Call(wrap__RQueryPipeline__with_log1p, self$.ptr))
  self$limit <- function(n) RQueryPipeline$.wrap(.Call(wrap__RQueryPipeline__limit, self$.ptr, as.integer(n)))
  self$collect <- function() RQueryResult$.wrap(.Call(wrap__RQueryPipeline__collect, self$.ptr))
  class(self) <- "RQueryPipeline"
  self
}
class(RQueryPipeline) <- "RQueryPipeline__class"

# ── RQueryResult class ──────────────────────────────────────────
#' @export
RQueryResult <- new.env(parent = emptyenv())
RQueryResult$.wrap <- function(ptr) {
  self <- new.env(parent = emptyenv())
  self$.ptr <- ptr
  self$to_dgcmatrix <- function() .Call(wrap__RQueryResult__to_dgcmatrix, self$.ptr)
  self$to_seurat <- function() .Call(wrap__RQueryResult__to_seurat, self$.ptr)
  self$to_sce <- function() .Call(wrap__RQueryResult__to_sce, self$.ptr)
  self$obs <- function() .Call(wrap__RQueryResult__obs, self$.ptr)
  self$var <- function() .Call(wrap__RQueryResult__var, self$.ptr)
  self$n_obs <- function() .Call(wrap__RQueryResult__n_obs, self$.ptr)
  self$n_vars <- function() .Call(wrap__RQueryResult__n_vars, self$.ptr)
  self$nnz <- function() .Call(wrap__RQueryResult__nnz, self$.ptr)
  self$skipped_shards <- function() .Call(wrap__RQueryResult__skipped_shards, self$.ptr)
  self$total_shards <- function() .Call(wrap__RQueryResult__total_shards, self$.ptr)
  class(self) <- "RQueryResult"
  self
}
class(RQueryResult) <- "RQueryResult__class"

# ── Harmony batch integration ──────────────────────────────────
# Free function wrapper; documentation lives in R/harmony.R.
scx_harmony_integrate <- function(embeddings, batch,
                                  n_clusters = NULL,
                                  theta = 2.0,
                                  sigma = 0.1,
                                  lambda = NULL,
                                  alpha = 0.2,
                                  max_iter = 10L,
                                  max_iter_kmeans = 4L,
                                  epsilon_harmony = 1e-2,
                                  epsilon_kmeans = 1e-3,
                                  block_size = 0.05,
                                  batch_prop_cutoff = 1e-5,
                                  tau = 0.0,
                                  random_state = 0L) {
  .Call(wrap__scx_harmony_integrate,
        embeddings,
        as.character(batch),
        if (is.null(n_clusters)) NULL else as.integer(n_clusters),
        as.numeric(theta),
        as.numeric(sigma),
        if (is.null(lambda)) NULL else as.numeric(lambda),
        as.numeric(alpha),
        as.integer(max_iter),
        as.integer(max_iter_kmeans),
        as.numeric(epsilon_harmony),
        as.numeric(epsilon_kmeans),
        as.numeric(block_size),
        as.numeric(batch_prop_cutoff),
        as.numeric(tau),
        as.integer(random_state))
}

# ── LISI metric ────────────────────────────────────────────────
scx_compute_lisi <- function(embeddings, labels,
                             perplexity = 30.0,
                             n_neighbors = NULL) {
  .Call(wrap__scx_compute_lisi,
        embeddings,
        as.character(labels),
        as.numeric(perplexity),
        if (is.null(n_neighbors)) NULL else as.integer(n_neighbors))
}

# ── Import functions (interop module) ──────────────────────────
#' @export
from_seurat <- function(seurat_obj, output_path, codec = NULL,
                        csc = FALSE, csc_cols_per_shard = 5000L) {
  invisible(.Call(
    wrap__from_seurat,
    seurat_obj,
    output_path,
    if (is.null(codec)) NULL else as.character(codec),
    as.logical(csc),
    as.integer(csc_cols_per_shard)
  ))
}

#' @export
from_sce <- function(sce_obj, output_path, codec = NULL,
                     csc = FALSE, csc_cols_per_shard = 5000L) {
  invisible(.Call(
    wrap__from_sce,
    sce_obj,
    output_path,
    if (is.null(codec)) NULL else as.character(codec),
    as.logical(csc),
    as.integer(csc_cols_per_shard)
  ))
}

# nolint end
