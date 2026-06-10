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
#' @param reference Reference group, or `NULL` to test each group vs rest.
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
