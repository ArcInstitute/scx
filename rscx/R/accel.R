#' Rust-native analysis accelerators for R
#'
#' Pipe-friendly front ends over the `scx-accel` CPU kernels, mirroring the
#' Seurat verbs they replace. Each function accepts **either** a `Seurat`
#' object (results are written back into the slot Seurat expects and the
#' object is returned invisibly) **or** a raw genes x cells `dgCMatrix` /
#' embedding matrix (the raw result list is returned). The matrix form makes
#' the kernels usable without Seurat installed.
#'
#' The full pipeline is reachable from R:
#' Normalize -> `scx_highly_variable_genes` -> `scx_pca` -> `scx_neighbors`
#' -> `scx_umap` -> `scx_leiden` -> `scx_rank_genes_groups`.
#'
#' @name scx-accelerators
NULL

# Internal: TRUE for a Seurat object.
.is_seurat <- function(x) methods::is(x, "Seurat")

# Internal: genes x cells dgCMatrix for `layer` of `object`'s `assay`.
.scx_layer_matrix <- function(object, assay, layer) {
  if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
  m <- SeuratObject::LayerData(object, layer = layer, assay = assay)
  methods::as(m, "CsparseMatrix")
}

#' @describeIn scx-accelerators PCA via randomized / covariance SVD.
#' @param object A `Seurat` object or a genes x cells `dgCMatrix`.
#' @param assay Assay to pull from (default: the object's default assay).
#' @param layer Layer to decompose (default `"data"`, i.e. log-normalized).
#' @param features Genes to use (default: `VariableFeatures(object)`, or all
#'   genes for a matrix input).
#' @param n_components Number of principal components (default `50`).
#' @param zero_center Mean-center genes before SVD (default `TRUE`).
#' @param n_oversamples,n_power_iterations Randomized-SVD accuracy knobs.
#' @param seed RNG seed (default `0`).
#' @param reduction.name,reduction.key Output reduction slot name / key.
#' @export
scx_pca <- function(object, assay = NULL, layer = "data", features = NULL,
                    n_components = 50L, zero_center = TRUE,
                    n_oversamples = 10L, n_power_iterations = 2L, seed = 0L,
                    reduction.name = "pca", reduction.key = "PC_") {
  if (.is_seurat(object)) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    if (is.null(features)) features <- SeuratObject::VariableFeatures(object)
    mat <- .scx_layer_matrix(object, assay, layer)
    if (length(features) > 0L) mat <- mat[features, , drop = FALSE]
  } else {
    mat <- methods::as(object, "CsparseMatrix")
  }

  res <- scx_pca_matrix(mat, n_components, zero_center,
                        n_oversamples, n_power_iterations, seed)
  if (!.is_seurat(object)) return(res)

  rownames(res$embeddings) <- colnames(mat)
  colnames(res$embeddings) <- paste0(reduction.key, seq_len(ncol(res$embeddings)))
  rownames(res$loadings) <- rownames(mat)
  colnames(res$loadings) <- colnames(res$embeddings)
  red <- Seurat::CreateDimReducObject(
    embeddings = res$embeddings,
    loadings = res$loadings,
    stdev = sqrt(res$variance_explained),
    key = reduction.key,
    assay = assay
  )
  object[[reduction.name]] <- red
  invisible(object)
}

#' @describeIn scx-accelerators PFlog (v4) / shifted-log normalization on raw
#'   counts (Booeshaghi et al.) returning a baseline-aware PCA embedding.
#' @param alpha NB overdispersion `α` (matrix-wide pseudocount `1/(4α)`).
#'   `NULL` (default) estimates `α` once from the counts; a finite positive
#'   value pins it (e.g. a reference `α` reused across datasets).
#'
#' Operates on **raw counts** (hence `layer = "counts"`), not log-normalized
#' data: counts are shifted by the matrix-wide Anscombe pseudocount `1/(4α)`,
#' log-transformed, then centered within each cell — no per-cell depth (it
#' cancels under the Anscombe scale). The exact transform is dense but
#' decomposes into a sparse `delta = log1p(4α·x)` plus a per-cell `baseline`, so
#' PCA never densifies. For a `Seurat` object the embedding is written to the
#' `reduction.name` reduction, the per-cell baseline to `object$pflog_baseline`,
#' and the fit (`alpha`, `pseudocount`) to `object@misc$pflog`. This is the
#' **in-memory** R path; the streaming / atlas-scale out-of-core path is
#' `pyscx`-only.
#'
#' Unlike [scx_pca], this does **not** accept a `features` argument: PFlog's
#' centering denominator `D` is defined over the full transcriptome, so
#' subsetting to variable features before the transform would silently change
#' the statistic (and the stored `baseline`). The transform and its PCA are
#' always computed over the full counts layer.
#' @export
scx_pflog <- function(object, assay = NULL, layer = "counts",
                      alpha = NULL, n_components = 50L, zero_center = TRUE,
                      n_oversamples = 10L, n_power_iterations = 2L, seed = 0L,
                      reduction.name = "pflog", reduction.key = "PFLOG_") {
  if (.is_seurat(object)) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    # No feature subsetting: PFlog centering must span the full transcriptome
    # (see @describeIn note above).
    mat <- .scx_layer_matrix(object, assay, layer)
  } else {
    mat <- methods::as(object, "CsparseMatrix")
  }

  # NULL alpha ⇒ NA_real_ sentinel (the Rust side estimates on NaN).
  alpha_arg <- if (is.null(alpha)) NA_real_ else as.numeric(alpha)
  res <- scx_pflog_matrix(mat, alpha_arg, n_components, zero_center,
                          n_oversamples, n_power_iterations, seed)
  if (!.is_seurat(object)) return(res)

  rownames(res$embeddings) <- colnames(mat)
  colnames(res$embeddings) <- paste0(reduction.key, seq_len(ncol(res$embeddings)))
  rownames(res$loadings) <- rownames(mat)
  colnames(res$loadings) <- colnames(res$embeddings)
  red <- Seurat::CreateDimReducObject(
    embeddings = res$embeddings,
    loadings = res$loadings,
    stdev = sqrt(res$variance_explained),
    key = reduction.key,
    assay = assay
  )
  object[[reduction.name]] <- red
  object$pflog_baseline <- res$baseline
  object@misc$pflog <- list(
    alpha = res$alpha, pseudocount = res$pseudocount, version = res$version
  )
  invisible(object)
}

#' @describeIn scx-accelerators kNN graph (HNSW) from an embedding.
#' @param reduction Input reduction (default `"pca"`).
#' @param dims Reduction dimensions to use (default `1:30`).
#' @param k Number of neighbors (default `20`).
#' @param ef_construction,ef_search HNSW accuracy knobs.
#' @param graph.name.prefix Prefix for the written graph slots.
#' @export
scx_neighbors <- function(object, reduction = "pca", dims = 1:30, k = 20L,
                          ef_construction = 200L, ef_search = 200L, seed = 0L,
                          assay = NULL, graph.name.prefix = NULL) {
  if (.is_seurat(object)) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    emb <- SeuratObject::Embeddings(object[[reduction]])
    dims <- dims[dims <= ncol(emb)]
    emb <- emb[, dims, drop = FALSE]
  } else {
    emb <- as.matrix(object)
  }

  res <- scx_knn_matrix(emb, k, ef_construction, ef_search, seed)
  if (!.is_seurat(object)) return(res)

  cells <- rownames(emb)
  g <- .scx_conn_to_graph(res, cells, assay)
  # The slot holds the fuzzy nearest-neighbor connectivity graph (UMAP-style),
  # not a shared-nearest-neighbor (SNN) graph, so name it `_nn` to avoid
  # implying SNN semantics to FindClusters consumers.
  graph.name <- if (is.null(graph.name.prefix)) {
    paste0(assay, "_nn")
  } else {
    paste0(graph.name.prefix, "_nn")
  }
  object[[graph.name]] <- g
  invisible(object)
}

# Internal: connectivity CSR (from scx_knn_matrix) -> SeuratObject Graph.
.scx_conn_to_graph <- function(res, cells, assay) {
  n_obs <- res$n_obs
  per_row <- diff(res$conn_indptr)
  sm <- Matrix::sparseMatrix(
    i = rep(seq_len(n_obs), times = per_row),
    j = res$conn_indices + 1L,
    x = res$conn_data,
    dims = c(n_obs, n_obs)
  )
  dimnames(sm) <- list(cells, cells)
  g <- SeuratObject::as.Graph(sm)
  SeuratObject::DefaultAssay(g) <- assay
  g
}

#' @describeIn scx-accelerators UMAP embedding from a kNN graph.
#' @param n_neighbors Neighbors for the internal kNN build (default `15`).
#' @param n_epochs SGD epochs (default `200`).
#' @param min_dist,spread UMAP layout parameters.
#' @param neighbors Optional precomputed kNN result from [scx_neighbors()]
#'   (matrix form). When supplied, its connectivity graph is reused; when
#'   `NULL`, `scx_umap` builds its **own** kNN (`n_neighbors`) and does **not**
#'   read any graph slot written by `scx_neighbors()` on a Seurat object.
#' @export
scx_umap <- function(object, reduction = "pca", dims = 1:30,
                     n_neighbors = 15L, n_components = 2L, n_epochs = 200L,
                     min_dist = 0.1, spread = 1.0, seed = 0L,
                     assay = NULL, reduction.name = "umap",
                     reduction.key = "UMAP_", neighbors = NULL) {
  if (.is_seurat(object)) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    emb <- SeuratObject::Embeddings(object[[reduction]])
    dims <- dims[dims <= ncol(emb)]
    emb <- emb[, dims, drop = FALSE]
  } else {
    emb <- as.matrix(object)
  }

  knn <- if (is.null(neighbors)) {
    scx_knn_matrix(emb, n_neighbors, 200L, 200L, seed)
  } else {
    neighbors
  }
  um <- scx_umap_graph(knn$conn_indptr, knn$conn_indices, knn$conn_data,
                       knn$n_obs, n_components, n_epochs, min_dist, spread,
                       5L, 1.0, seed)
  if (!.is_seurat(object)) return(um)

  rownames(um$embeddings) <- rownames(emb)
  colnames(um$embeddings) <- paste0(reduction.key, seq_len(ncol(um$embeddings)))
  red <- Seurat::CreateDimReducObject(
    embeddings = um$embeddings, key = reduction.key, assay = assay
  )
  object[[reduction.name]] <- red
  invisible(object)
}

#' @describeIn scx-accelerators Leiden clustering from a kNN graph.
#' @param resolution Resolution parameter (default `1.0`).
#' @param max_iterations Maximum Leiden outer iterations (default `100`).
#' @param neighbors Optional precomputed kNN result from [scx_neighbors()]
#'   (matrix form). When supplied, its connectivity graph is reused; when
#'   `NULL`, `scx_leiden` builds its **own** kNN (`n_neighbors`) and does
#'   **not** read any graph slot written by `scx_neighbors()`.
#' @export
scx_leiden <- function(object, reduction = "pca", dims = 1:30,
                       n_neighbors = 20L, resolution = 1.0, seed = 0L,
                       max_iterations = 100L, assay = NULL, neighbors = NULL) {
  if (.is_seurat(object)) {
    emb <- SeuratObject::Embeddings(object[[reduction]])
    dims <- dims[dims <= ncol(emb)]
    emb <- emb[, dims, drop = FALSE]
  } else {
    emb <- as.matrix(object)
  }

  knn <- if (is.null(neighbors)) {
    scx_knn_matrix(emb, n_neighbors, 200L, 200L, seed)
  } else {
    neighbors
  }
  res <- scx_leiden_graph(knn$conn_indptr, knn$conn_indices, knn$conn_data,
                          knn$n_obs, resolution, seed, max_iterations)
  if (!.is_seurat(object)) return(res)

  clusters <- factor(res$membership)
  names(clusters) <- rownames(emb)
  object$seurat_clusters <- clusters
  SeuratObject::Idents(object) <- clusters
  invisible(object)
}

#' @describeIn scx-accelerators Wilcoxon rank-sum DE (FindAllMarkers analog).
#' @param group.by `meta.data` column with per-cell group labels.
#' @param groups Subset of group labels to test (character vector); `NULL`
#'   (default) tests every group.
#' @param reference Reference baseline: for `scx_rank_genes_groups`, the group to
#'   test against (`NULL` tests each group vs the rest); for `scx_pseudobulk_dex` /
#'   `scx_nb_glm`, the reference level of `test_col`.
#' @param log_transformed Whether `layer` is already log-transformed.
#' @param rankby_abs,tie_correct Wilcoxon options; defaults (`FALSE`/`FALSE`)
#'   match `pyscx`/scanpy. The kernel densifies the matrix internally, so on a
#'   Seurat object with `features = NULL` this defaults to `VariableFeatures()`
#'   when set; passing the full gene set materializes a dense matrix.
#' @return For DE, a long-format `data.frame` with columns `group`, `gene`,
#'   `score`, `pval`, `pval_adj`, `logfoldchange`.
#' @export
scx_rank_genes_groups <- function(object, group.by = NULL, assay = NULL,
                                  layer = "data", features = NULL,
                                  groups = NULL, reference = NULL,
                                  log_transformed = TRUE, rankby_abs = FALSE,
                                  tie_correct = FALSE) {
  if (.is_seurat(object)) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    if (is.null(group.by)) {
      labels <- as.character(SeuratObject::Idents(object))
    } else {
      labels <- as.character(object[[group.by, drop = TRUE]])
    }
    # The DE kernel densifies the matrix; default to the variable features
    # when the caller did not restrict `features` (matches scanpy's
    # rank_genes_groups-on-HVGs expectation and bounds memory).
    if (is.null(features)) {
      vf <- SeuratObject::VariableFeatures(object)
      if (length(vf) > 0L) features <- vf
    }
    mat <- .scx_layer_matrix(object, assay, layer)
    if (!is.null(features)) mat <- mat[features, , drop = FALSE]
  } else {
    mat <- methods::as(object, "CsparseMatrix")
    labels <- as.character(groups)
  }
  gene_names <- rownames(mat)
  if (is.null(gene_names)) gene_names <- as.character(seq_len(nrow(mat)))

  # scx_rank_genes() materializes a dense n_cells x n_genes matrix; warn before
  # a large allocation so an accidental full-matrix call is not silent.
  if (as.double(nrow(mat)) * as.double(ncol(mat)) > 5e7) {
    warning(sprintf(
      paste0("scx_rank_genes_groups densifies a %d x %d matrix (~%.1f GB); ",
             "subset to highly variable genes via `features=` to bound memory."),
      nrow(mat), ncol(mat), as.double(nrow(mat)) * as.double(ncol(mat)) * 4 / 1e9),
      call. = FALSE)
  }

  res <- scx_rank_genes(mat, gene_names, labels, reference, log_transformed,
                        rankby_abs = rankby_abs, tie_correct = tie_correct)
  .scx_de_to_dataframe(res)
}

# Internal: flatten the per-group DE lists into one long data.frame.
.scx_de_to_dataframe <- function(res) {
  groups <- res$group_names
  parts <- lapply(seq_along(groups), function(gi) {
    data.frame(
      group = groups[[gi]],
      gene = res$names[[gi]],
      score = res$scores[[gi]],
      pval = res$pvals[[gi]],
      pval_adj = res$pvals_adj[[gi]],
      logfoldchange = res$logfoldchanges[[gi]],
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, parts)
}

#' @describeIn scx-accelerators Highly variable genes (seurat_v3 / vst).
#' @param n_top_genes Number of HVGs to select (default `2000`).
#' @param span Loess span for the mean-variance fit (default `0.3`).
#' @section Note:
#' The mean/variance and clipped-sum passes run in Rust (identical to the
#' Python accelerator), but the loess fit uses R's native `stats::loess`
#' whereas `pyscx` uses `skmisc.loess`. The two LOESS implementations can give
#' slightly different fitted values, so the selected HVG set may differ
#' marginally between R and Python on the same data.
#' @export
scx_highly_variable_genes <- function(object, assay = NULL, layer = "counts",
                                      n_top_genes = 2000L, span = 0.3) {
  if (.is_seurat(object)) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    mat <- .scx_layer_matrix(object, assay, layer)
  } else {
    mat <- methods::as(object, "CsparseMatrix")
  }
  n_genes <- nrow(mat)
  n_cells <- ncol(mat)

  stats <- scx_hvg_mean_var(mat)
  mean <- stats$means
  variance <- stats$variances

  # seurat_v3: loess of log10(var) ~ log10(mean) over non-constant genes;
  # regularized std defines the clip threshold for the second pass.
  estimat_var <- rep(0, n_genes)
  not_const <- variance > 0
  if (sum(not_const) > 1L) {
    x <- log10(mean[not_const])
    y <- log10(variance[not_const])
    fitted <- tryCatch({
      fit <- stats::loess(y ~ x, span = span, degree = 2,
                          family = "gaussian", surface = "direct")
      stats::predict(fit, x)
    }, error = function(e) {
      # R's loess can fail on very small / degenerate gene sets where
      # skmisc.loess would not. Fall back to the unregularized variance
      # (reg_std = observed sd) so selection still proceeds.
      y
    })
    estimat_var[not_const] <- fitted
  }
  reg_std <- sqrt(10^estimat_var)
  clip_val <- reg_std * sqrt(n_cells) + mean

  clipped <- scx_hvg_clipped_sums(mat, clip_val)
  counts_sum <- clipped$counts_sum
  sq_counts_sum <- clipped$sq_counts_sum

  norm_gene_var <- rep(0, n_genes)
  denom <- (n_cells - 1) * reg_std^2
  ok <- denom > 0
  norm_gene_var[ok] <- (1 / denom[ok]) *
    ((n_cells * mean[ok]^2) + sq_counts_sum[ok] - 2 * counts_sum[ok] * mean[ok])

  rank_order <- order(norm_gene_var, decreasing = TRUE)
  n_top <- min(n_top_genes, n_genes)
  highly_variable <- logical(n_genes)
  highly_variable[rank_order[seq_len(n_top)]] <- TRUE

  df <- data.frame(
    means = mean,
    variances = variance,
    variances_norm = norm_gene_var,
    highly_variable = highly_variable,
    row.names = rownames(mat),
    stringsAsFactors = FALSE
  )
  if (!.is_seurat(object)) return(df)

  hvg_names <- rownames(mat)[highly_variable]
  SeuratObject::VariableFeatures(object, assay = assay) <- hvg_names
  invisible(object)
}

#' @describeIn scx-accelerators Gene-set scoring (scanpy `score_genes` analog).
#' @param gene_list Character vector of gene names to score. Genes absent from
#'   the matrix are dropped with a warning; an all-missing set is an error.
#' @param method One of `"control"` (scanpy `score_genes`: mean(gene_list) minus
#'   a binned control set), `"mean"` (per-cell mean over `gene_list`), or
#'   `"zscore"` (decoupler `mt.zscore`). Default `"control"`.
#' @param ctrl_size,n_bins Control-method knobs (control genes per expression
#'   bin / number of bins). Defaults `50` / `25`.
#' @param random_state Seed for the deterministic control sampler (not
#'   numpy-compatible). Default `0`.
#' @param gene_pool Optional character vector restricting the control binning
#'   universe (control method only); default `NULL` = all genes.
#' @param score_name obs/meta.data column to write the score into on a Seurat
#'   object. Default `"score"`.
#' @return For `scx_score_genes`: on a Seurat object, the object with
#'   `score_name` added to `meta.data` (returned invisibly); on a matrix, a
#'   numeric per-cell score vector (named by cell when column names exist).
#' @export
scx_score_genes <- function(object, gene_list, method = "control",
                            ctrl_size = 50L, n_bins = 25L, random_state = 0L,
                            gene_pool = NULL, score_name = "score",
                            assay = NULL, layer = "data") {
  if (.is_seurat(object)) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    mat <- .scx_layer_matrix(object, assay, layer)
  } else {
    mat <- methods::as(object, "CsparseMatrix")
  }
  gene_names <- rownames(mat)
  if (is.null(gene_names)) {
    stop("score_genes needs gene (row) names to resolve `gene_list`", call. = FALSE)
  }

  # Resolve gene_list -> 0-based indices; drop (warn) the missing, error if none.
  gene_list <- unique(as.character(gene_list))
  pos <- match(gene_list, gene_names)
  missing <- gene_list[is.na(pos)]
  if (length(missing) > 0L) {
    shown <- missing[seq_len(min(10L, length(missing)))]
    warning(sprintf("score_genes: %d/%d genes not found and dropped: %s",
                    length(missing), length(gene_list),
                    paste(shown, collapse = ", ")),
            call. = FALSE)
  }
  gene_list_idx <- pos[!is.na(pos)] - 1L
  if (length(gene_list_idx) == 0L) {
    stop("score_genes: no genes in `gene_list` matched the matrix", call. = FALSE)
  }

  # gene_pool: only the control method bins it; default to all genes.
  if (identical(method, "control")) {
    if (is.null(gene_pool)) {
      gene_pool_idx <- seq_len(nrow(mat)) - 1L
    } else {
      gp <- match(unique(as.character(gene_pool)), gene_names)
      gene_pool_idx <- gp[!is.na(gp)] - 1L
    }
  } else {
    gene_pool_idx <- integer(0)
  }

  scores <- scx_score_genes_matrix(mat, gene_list_idx, gene_pool_idx, method,
                                   ctrl_size, n_bins, random_state)

  if (!.is_seurat(object)) {
    names(scores) <- colnames(mat)
    return(scores)
  }
  object[[score_name]] <- scores
  invisible(object)
}

#' @describeIn scx-accelerators Pseudobulk aggregation (cells -> group x gene).
#' @param group_by Grouping of cells. On a Seurat object: a character vector of
#'   `meta.data` column name(s). On a matrix: a per-cell label vector, or a
#'   `data.frame` / named list of per-cell label columns for multi-column
#'   groupby.
#' @param min_cells_per_group Drop groups with fewer than this many cells
#'   (default `0` = keep all).
#' @return For `scx_pseudobulk`: a list with `counts` (a genes x groups matrix,
#'   rownames = genes, colnames = the group labels joined by `"_"`), `samples`
#'   (a `data.frame` of the groupby columns plus `n_cells` per group), and
#'   `gene_names`. Returned for both Seurat and matrix inputs (pseudobulk
#'   collapses the cell axis, so it cannot be written back into the object).
#' @export
scx_pseudobulk <- function(object, group_by, method = "sum",
                           min_cells_per_group = 0L, assay = NULL,
                           layer = "counts") {
  if (.is_seurat(object)) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    mat <- .scx_layer_matrix(object, assay, layer)
    # group_by names meta.data columns.
    cols <- as.character(group_by)
    groupby <- lapply(cols, function(cn) as.character(object[[cn, drop = TRUE]]))
    names(groupby) <- cols
  } else {
    mat <- methods::as(object, "CsparseMatrix")
    # group_by is a vector / data.frame / list of per-cell labels.
    if (is.data.frame(group_by) || is.list(group_by)) {
      groupby <- lapply(group_by, as.character)
      cols <- names(groupby)
      if (is.null(cols)) cols <- paste0("group", seq_along(groupby))
      names(groupby) <- cols
    } else {
      groupby <- list(group = as.character(group_by))
      cols <- "group"
    }
  }

  gene_names <- rownames(mat)
  if (is.null(gene_names)) gene_names <- as.character(seq_len(nrow(mat)))

  res <- scx_pseudobulk_matrix(mat, groupby, cols, gene_names, method,
                               min_cells_per_group)

  # Kernel returns counts as n_groups x n_genes; transpose to the R-native
  # features x samples (genes x groups) orientation.
  counts <- t(res$counts)
  rownames(counts) <- res$gene_names
  # group_labels: one character vector per groupby column, length n_groups.
  label_df <- as.data.frame(res$group_labels, stringsAsFactors = FALSE)
  names(label_df) <- res$groupby_columns
  sample_ids <- do.call(paste, c(label_df, sep = "_"))
  colnames(counts) <- sample_ids

  samples <- label_df
  samples$n_cells <- res$cell_counts
  rownames(samples) <- sample_ids

  list(counts = counts, samples = samples, gene_names = res$gene_names)
}

#' @describeIn scx-accelerators Pseudobulk differential expression via a
#'   Rust-native negative-binomial GLM (DESeq2-style, CPU-only). Aggregates
#'   cells into pseudobulk samples and fits each non-reference level of
#'   `test_col` vs `reference`.
#' @param test_col The `group_by` column holding the condition being tested.
#' @param reference Reference baseline: for `scx_rank_genes_groups`, the group to
#'   test against (`NULL` tests each group vs the rest); for `scx_pseudobulk_dex` /
#'   `scx_nb_glm`, the reference level of `test_col`.
#' @param aggr_method Pseudobulk aggregation, `"sum"` (default) or `"mean"`.
#'   `scx_pseudobulk_dex()` accepts only `"sum"`: its negative-binomial count
#'   model is defined on summed replicate counts, not fractional aggregates.
#' @param dispersion Dispersion estimator: `"cox_reid_shrunk"` (default),
#'   `"cox_reid_mle"`, or `"moments"`.
#' @param cooks_filtering,independent_filtering DESeq2 results-stage filters
#'   (both `TRUE` by default).
#' @section Replicates &amp; input:
#' NB-GLM requires **≥2 pseudobulk replicates per condition**, so `group_by`
#' must include `test_col` **and** a replicate column (donor/batch/well), e.g.
#' `group_by = c("condition", "donor")`. Targets with fewer than 2 replicates
#' per side are skipped with a warning. Input must be **raw counts** (use
#' `layer = "counts"`); for no-replicate or log-normalized data use
#' [scx_rank_genes_groups] (Wilcoxon).
#'
#' Each non-reference level is fit as an independent pairwise 2-coefficient
#' NB-GLM (intercept + treatment) on only that target's and the reference's
#' pseudobulk samples, with size factors recomputed per contrast. The replicate
#' column is **not** entered as a covariate, so batch/donor confounders are not
#' adjusted and `baseMean`/`dispersion` differ per contrast. (rscx exposes only
#' this fixed-design fit; the pyscx `pseudobulk_dex(backend="nb_glm", design=...)`
#' path additionally supports a covariate-adjusted joint fit — not yet wired into
#' rscx.)
#' @return For `scx_pseudobulk_dex`: a long-format `data.frame` with one row per
#'   gene per non-reference target and DESeq2-style columns `gene`, `baseMean`,
#'   `log2FoldChange`, `lfcSE`, `stat`, `pvalue`, `padj`, `target`, `reference`.
#' @export
scx_pseudobulk_dex <- function(object, group_by, test_col, reference,
                               aggr_method = "sum", min_cells_per_group = 10L,
                               dispersion = "cox_reid_shrunk",
                               cooks_filtering = TRUE,
                               independent_filtering = TRUE,
                               assay = NULL, layer = "counts") {
  if (.is_seurat(object)) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    mat <- .scx_layer_matrix(object, assay, layer)
    cols <- as.character(group_by)
    groupby <- lapply(cols, function(cn) as.character(object[[cn, drop = TRUE]]))
    names(groupby) <- cols
  } else {
    mat <- methods::as(object, "CsparseMatrix")
    if (is.data.frame(group_by) || is.list(group_by)) {
      groupby <- lapply(group_by, as.character)
      cols <- names(groupby)
      if (is.null(cols)) cols <- paste0("group", seq_along(groupby))
      names(groupby) <- cols
    } else {
      stop(paste0("scx_pseudobulk_dex needs replicates: pass `group_by` as a ",
                  "data.frame / named list including the test column and a ",
                  "replicate column (e.g. donor)"), call. = FALSE)
    }
  }
  if (!(test_col %in% cols)) {
    stop(sprintf("test_col '%s' is not among group_by columns (%s)",
                 test_col, paste(cols, collapse = ", ")), call. = FALSE)
  }
  # Refuse rather than warn: the NB-GLM is the only DE engine rscx exposes and
  # it is defined on summed counts, so there is no degraded-but-working path to
  # warn about. Warning and continuing sent the fractional values into the
  # model's input validator, which failed naming an interior cell instead of
  # the argument at fault. Matches the pyscx guard.
  if (identical(aggr_method, "mean")) {
    stop(paste0("scx_pseudobulk_dex requires aggr_method='sum' (got 'mean'): ",
                "the negative-binomial count model is defined on summed ",
                "replicate counts, not fractional mean aggregates. Use ",
                "aggr_method='sum', or for mean aggregation use pyscx ",
                "pseudobulk_dex(backend='pydeseq2', aggr_method='mean'), ",
                "which rscx does not expose."), call. = FALSE)
  }

  gene_names <- rownames(mat)
  if (is.null(gene_names)) gene_names <- as.character(seq_len(nrow(mat)))

  res <- scx_pseudobulk_dex_matrix(mat, groupby, cols, test_col, reference,
                                   gene_names, aggr_method, min_cells_per_group,
                                   dispersion, cooks_filtering,
                                   independent_filtering)

  if (length(res$skipped) > 0L) {
    warning(sprintf(
      "scx_pseudobulk_dex skipped %d target(s) with <2 replicates per condition: %s",
      length(res$skipped), paste(res$skipped, collapse = ", ")), call. = FALSE)
  }

  data.frame(
    gene = res$gene,
    baseMean = res$baseMean,
    log2FoldChange = res$log2FoldChange,
    lfcSE = res$lfcSE,
    stat = res$stat,
    pvalue = res$pvalue,
    padj = res$padj,
    target = res$target,
    reference = res$reference,
    stringsAsFactors = FALSE
  )
}

#' @describeIn scx-accelerators Direct negative-binomial GLM on a pre-aggregated
#'   pseudobulk count matrix and design (the `nb_glm` building block).
#' @param counts A genes x samples numeric matrix of pseudobulk counts (DESeq2
#'   orientation).
#' @param gene_names Character vector of gene names (length `nrow(counts)`);
#'   defaults to `rownames(counts)`, falling back to positional indices.
#' @param design A samples x features numeric design matrix of full column rank,
#'   e.g. `model.matrix(~ condition, sampleinfo)`.
#' @param contrast 1-based design column (coefficient) to test; `NULL` (default)
#'   tests the last coefficient.
#' @param size_factors Per-sample size factors; `NULL` (default) uses the
#'   DESeq2 median-ratio estimate.
#' @return For `scx_nb_glm`: a `data.frame` with columns `gene`, `baseMean`,
#'   `log2FoldChange`, `lfcSE`, `stat`, `pvalue`, `padj`, `dispersion`,
#'   `converged`.
#' @export
scx_nb_glm <- function(counts, design, contrast = NULL, size_factors = NULL,
                       gene_names = rownames(counts),
                       dispersion = "cox_reid_shrunk", cooks_filtering = TRUE,
                       independent_filtering = TRUE) {
  counts <- as.matrix(counts)
  design <- as.matrix(design)
  storage.mode(counts) <- "double"
  storage.mode(design) <- "double"
  if (is.null(gene_names)) gene_names <- as.character(seq_len(nrow(counts)))

  res <- scx_nb_glm_matrix(counts, design, contrast, size_factors, gene_names,
                           dispersion, cooks_filtering, independent_filtering)

  data.frame(
    gene = res$gene,
    baseMean = res$baseMean,
    log2FoldChange = res$log2FoldChange,
    lfcSE = res$lfcSE,
    stat = res$stat,
    pvalue = res$pvalue,
    padj = res$padj,
    dispersion = res$dispersion,
    converged = res$converged,
    stringsAsFactors = FALSE
  )
}
