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
  self$x_matrix <- function(allow_lossy = FALSE) .Call(wrap__ScxExperiment__x_matrix, self$.ptr, as.logical(allow_lossy))
  self$layer <- function(name, allow_lossy = FALSE) .Call(wrap__ScxExperiment__layer, self$.ptr, name, as.logical(allow_lossy))
  self$layer_names <- function() .Call(wrap__ScxExperiment__layer_names, self$.ptr)
  self$query <- function(modality = NULL) {
    ptr <- .Call(
      wrap__ScxExperiment__query,
      self$.ptr,
      if (is.null(modality)) NULL else as.character(modality)
    )
    RQueryPipeline$.wrap(ptr)
  }
  # Phase I.1 / I.2: multimodal accessors.
  self$is_multimodal <- function()
    .Call(wrap__ScxExperiment__is_multimodal, self$.ptr)
  self$modality_names <- function()
    .Call(wrap__ScxExperiment__modality_names, self$.ptr)
  self$to_seurat <- function(allow_lossy = FALSE) .Call(wrap__ScxExperiment__to_seurat, self$.ptr, as.logical(allow_lossy))
  self$to_mae <- function(allow_lossy = FALSE) .Call(wrap__ScxExperiment__to_mae, self$.ptr, as.logical(allow_lossy))
  # F2 grouped reads (7.2c).
  self$read_group <- function(label)
    RQueryResult$.wrap(.Call(wrap__ScxExperiment__read_group, self$.ptr, label))
  self$read_reference <- function() {
    ptr <- .Call(wrap__ScxExperiment__read_reference, self$.ptr)
    if (is.null(ptr)) NULL else RQueryResult$.wrap(ptr)
  }
  self$group_labels <- function() .Call(wrap__ScxExperiment__group_labels, self$.ptr)
  self$iter_group_shards <- function()
    lapply(.Call(wrap__ScxExperiment__iter_group_shards, self$.ptr), RGroupShardHandle$.wrap)
  # Backed (lazy) X access — see R/backed.R for the [ / dim / print methods.
  self$x_backed <- function(cache_shards = 128) {
    ScxBackedSparse$.wrap(
      .Call(wrap__ScxExperiment__x_backed, self$.ptr, as.numeric(cache_shards))
    )
  }
  # Lazy transform chain — see R/lazy.R for the chainable transform verbs.
  self$x_lazy <- function(cache_shards = 128) {
    ScxLazyTransformed$.wrap(
      .Call(wrap__ScxExperiment__x_lazy, self$.ptr, as.numeric(cache_shards))
    )
  }
  class(self) <- "ScxExperiment"
  self
}
class(ScxExperiment) <- "ScxExperiment__class"

# ── ScxBackedSparse class ───────────────────────────────────────
#' @export
ScxBackedSparse <- new.env(parent = emptyenv())
ScxBackedSparse$.wrap <- function(ptr) {
  self <- new.env(parent = emptyenv())
  self$.ptr <- ptr
  self$n_obs <- function() .Call(wrap__RBackedSparse__n_obs, self$.ptr)
  self$n_vars <- function() .Call(wrap__RBackedSparse__n_vars, self$.ptr)
  self$nnz <- function() .Call(wrap__RBackedSparse__nnz, self$.ptr)
  self$read_rows <- function(start, end)
    .Call(wrap__RBackedSparse__read_rows, self$.ptr, as.numeric(start), as.numeric(end))
  self$read_row_indices <- function(indices)
    .Call(wrap__RBackedSparse__read_row_indices, self$.ptr, as.numeric(indices))
  self$row_sums <- function() .Call(wrap__RBackedSparse__row_sums, self$.ptr)
  self$col_sums <- function() .Call(wrap__RBackedSparse__col_sums, self$.ptr)
  self$to_dgcmatrix <- function() .Call(wrap__RBackedSparse__to_dgcmatrix, self$.ptr)
  class(self) <- "ScxBackedSparse"
  self
}
class(ScxBackedSparse) <- "ScxBackedSparse__class"

# ── ScxLazyTransformed class ────────────────────────────────────
#' @export
ScxLazyTransformed <- new.env(parent = emptyenv())
ScxLazyTransformed$.wrap <- function(ptr) {
  self <- new.env(parent = emptyenv())
  self$.ptr <- ptr
  self$n_obs <- function() .Call(wrap__RLazyTransformed__n_obs, self$.ptr)
  self$n_vars <- function() .Call(wrap__RLazyTransformed__n_vars, self$.ptr)
  self$nnz <- function() .Call(wrap__RLazyTransformed__nnz, self$.ptr)
  self$transform_names <- function() .Call(wrap__RLazyTransformed__transform_names, self$.ptr)
  # Transform verbs each return a NEW lazy handle (immutable chain).
  self$normalize_total <- function(target_sum)
    ScxLazyTransformed$.wrap(.Call(wrap__RLazyTransformed__normalize_total, self$.ptr, as.numeric(target_sum)))
  self$log1p <- function()
    ScxLazyTransformed$.wrap(.Call(wrap__RLazyTransformed__log1p, self$.ptr))
  self$row_scale <- function(factors)
    ScxLazyTransformed$.wrap(.Call(wrap__RLazyTransformed__row_scale, self$.ptr, as.numeric(factors)))
  self$read_rows <- function(start, end)
    .Call(wrap__RLazyTransformed__read_rows, self$.ptr, as.numeric(start), as.numeric(end))
  self$read_row_indices <- function(indices)
    .Call(wrap__RLazyTransformed__read_row_indices, self$.ptr, as.numeric(indices))
  self$row_sums <- function() .Call(wrap__RLazyTransformed__row_sums, self$.ptr)
  self$col_sums <- function() .Call(wrap__RLazyTransformed__col_sums, self$.ptr)
  self$to_dgcmatrix <- function() .Call(wrap__RLazyTransformed__to_dgcmatrix, self$.ptr)
  class(self) <- "ScxLazyTransformed"
  self
}
class(ScxLazyTransformed) <- "ScxLazyTransformed__class"

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
  self$count <- function() .Call(wrap__RQueryPipeline__count, self$.ptr)
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
  self$to_dgcmatrix <- function(allow_lossy = FALSE) .Call(wrap__RQueryResult__to_dgcmatrix, self$.ptr, as.logical(allow_lossy))
  self$to_seurat <- function(allow_lossy = FALSE) .Call(wrap__RQueryResult__to_seurat, self$.ptr, as.logical(allow_lossy))
  self$to_sce <- function(allow_lossy = FALSE) .Call(wrap__RQueryResult__to_sce, self$.ptr, as.logical(allow_lossy))
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

# ── RGroupShardHandle class (F2 grouped reads) ──────────────────
#' @export
RGroupShardHandle <- new.env(parent = emptyenv())
RGroupShardHandle$.wrap <- function(ptr) {
  self <- new.env(parent = emptyenv())
  self$.ptr <- ptr
  self$shard_index <- function() .Call(wrap__RGroupShardHandle__shard_index, self$.ptr)
  self$global_start <- function() .Call(wrap__RGroupShardHandle__global_start, self$.ptr)
  self$global_stop <- function() .Call(wrap__RGroupShardHandle__global_stop, self$.ptr)
  self$labels <- function() .Call(wrap__RGroupShardHandle__labels, self$.ptr)
  self$groups <- function() .Call(wrap__RGroupShardHandle__groups, self$.ptr)
  self$read_group <- function(label)
    RQueryResult$.wrap(.Call(wrap__RGroupShardHandle__read_group, self$.ptr, label))
  self$to_query_result <- function()
    RQueryResult$.wrap(.Call(wrap__RGroupShardHandle__to_query_result, self$.ptr))
  class(self) <- "RGroupShardHandle"
  self
}
class(RGroupShardHandle) <- "RGroupShardHandle__class"

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

# ── Analysis accelerators (accel module) ──────────────────────
# Low-level matrix/graph wrappers; the Seurat-aware front ends
# (scx_pca, scx_neighbors, …) live in R/accel.R.
scx_pca_matrix <- function(counts, n_components, zero_center,
                           n_oversamples, n_power_iterations, seed) {
  .Call(wrap__scx_pca_matrix,
        counts,
        as.integer(n_components),
        as.logical(zero_center),
        as.integer(n_oversamples),
        as.integer(n_power_iterations),
        as.numeric(seed))
}

scx_pflog_matrix <- function(counts, alpha, n_components, zero_center,
                             n_oversamples, n_power_iterations, seed) {
  .Call(wrap__scx_pflog_matrix,
        counts,
        as.numeric(alpha),
        as.integer(n_components),
        as.logical(zero_center),
        as.integer(n_oversamples),
        as.integer(n_power_iterations),
        as.numeric(seed))
}

scx_knn_matrix <- function(embeddings, n_neighbors,
                           ef_construction, ef_search, seed) {
  .Call(wrap__scx_knn_matrix,
        embeddings,
        as.integer(n_neighbors),
        as.integer(ef_construction),
        as.integer(ef_search),
        as.numeric(seed))
}

scx_umap_graph <- function(conn_indptr, conn_indices, conn_data,
                           n_obs, n_components, n_epochs, min_dist, spread,
                           negative_sample_rate, learning_rate, seed) {
  .Call(wrap__scx_umap_graph,
        as.numeric(conn_indptr),
        as.integer(conn_indices),
        as.numeric(conn_data),
        as.integer(n_obs),
        as.integer(n_components),
        as.integer(n_epochs),
        as.numeric(min_dist),
        as.numeric(spread),
        as.integer(negative_sample_rate),
        as.numeric(learning_rate),
        as.numeric(seed))
}

scx_leiden_graph <- function(indptr, indices, weights, n_nodes,
                             resolution, seed, max_iterations) {
  .Call(wrap__scx_leiden_graph,
        as.numeric(indptr),
        as.integer(indices),
        as.numeric(weights),
        as.integer(n_nodes),
        as.numeric(resolution),
        as.numeric(seed),
        as.integer(max_iterations))
}

scx_rank_genes <- function(counts, gene_names, groups, reference,
                           log_transformed, rankby_abs = FALSE,
                           tie_correct = FALSE) {
  .Call(wrap__scx_rank_genes,
        counts,
        as.character(gene_names),
        as.character(groups),
        if (is.null(reference)) NULL else as.character(reference),
        as.logical(log_transformed),
        as.logical(rankby_abs),
        as.logical(tie_correct))
}

scx_hvg_mean_var <- function(counts) {
  .Call(wrap__scx_hvg_mean_var, counts)
}

scx_hvg_clipped_sums <- function(counts, clip_val) {
  .Call(wrap__scx_hvg_clipped_sums, counts, as.numeric(clip_val))
}

scx_score_genes_matrix <- function(counts, gene_list_idx, gene_pool_idx, method,
                                   ctrl_size, n_bins, random_state) {
  .Call(wrap__scx_score_genes_matrix,
        counts,
        as.integer(gene_list_idx),
        as.integer(gene_pool_idx),
        as.character(method),
        as.integer(ctrl_size),
        as.integer(n_bins),
        as.numeric(random_state))
}

scx_pseudobulk_matrix <- function(counts, groupby, groupby_columns, gene_names,
                                  method, min_cells_per_group) {
  .Call(wrap__scx_pseudobulk_matrix,
        counts,
        groupby,
        as.character(groupby_columns),
        as.character(gene_names),
        as.character(method),
        as.integer(min_cells_per_group))
}

scx_pseudobulk_dex_matrix <- function(counts, groupby, groupby_columns, test_col,
                                      reference, gene_names, aggr_method,
                                      min_cells_per_group, dispersion,
                                      cooks_filtering, independent_filtering) {
  .Call(wrap__scx_pseudobulk_dex_matrix,
        counts,
        groupby,
        as.character(groupby_columns),
        as.character(test_col),
        as.character(reference),
        as.character(gene_names),
        as.character(aggr_method),
        as.integer(min_cells_per_group),
        as.character(dispersion),
        as.logical(cooks_filtering),
        as.logical(independent_filtering))
}

scx_nb_glm_matrix <- function(counts, design, contrast_index, size_factors,
                              gene_names, dispersion, cooks_filtering,
                              independent_filtering) {
  .Call(wrap__scx_nb_glm_matrix,
        counts,
        design,
        if (is.null(contrast_index)) NULL else as.integer(contrast_index),
        if (is.null(size_factors)) NULL else as.numeric(size_factors),
        as.character(gene_names),
        as.character(dispersion),
        as.logical(cooks_filtering),
        as.logical(independent_filtering))
}

# ── Import functions (interop module) ──────────────────────────
#' @export
from_seurat <- function(seurat_obj, output_path, codec = NULL,
                        csc = FALSE, csc_cols_per_shard = 5000L,
                        row_group_rows = NULL) {
  invisible(.Call(
    wrap__from_seurat,
    seurat_obj,
    output_path,
    if (is.null(codec)) NULL else as.character(codec),
    as.logical(csc),
    as.integer(csc_cols_per_shard),
    if (is.null(row_group_rows)) NULL else as.integer(row_group_rows)
  ))
}

#' @export
from_sce <- function(sce_obj, output_path, codec = NULL,
                     csc = FALSE, csc_cols_per_shard = 5000L,
                     row_group_rows = NULL) {
  invisible(.Call(
    wrap__from_sce,
    sce_obj,
    output_path,
    if (is.null(codec)) NULL else as.character(codec),
    as.logical(csc),
    as.integer(csc_cols_per_shard),
    if (is.null(row_group_rows)) NULL else as.integer(row_group_rows)
  ))
}

# Phase I.2: import a Bioconductor MultiAssayExperiment to a multimodal
# SCX file. Requires aligned cell axes across experiments; raises with
# a clear message on mismatch.
#' @export
from_mae <- function(mae_obj, output_path, codec = NULL,
                     csc = FALSE, csc_cols_per_shard = 5000L,
                     row_group_rows = NULL) {
  invisible(.Call(
    wrap__from_mae,
    mae_obj,
    output_path,
    if (is.null(codec)) NULL else as.character(codec),
    as.logical(csc),
    as.integer(csc_cols_per_shard),
    if (is.null(row_group_rows)) NULL else as.integer(row_group_rows)
  ))
}

# nolint end
