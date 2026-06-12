# Phase 4 T4.1 — Rust-native analysis accelerators (CPU).
#
# The matrix-input path of each wrapper runs with only the `Matrix` package
# installed (no Seurat), so these exercise the full kernel chain
# HVG -> PCA -> Neighbors -> UMAP -> Leiden -> rank-genes. The Seurat-object
# path is covered by guarded tests that skip when Seurat is absent.

make_counts <- function(n_genes = 40L, n_cells = 120L, seed = 1L) {
  set.seed(seed)
  m <- Matrix::rsparsematrix(
    n_genes, n_cells,
    density = 0.3,
    rand.x = function(n) sample.int(20L, n, replace = TRUE)
  )
  m <- abs(m)
  m <- methods::as(m, "CsparseMatrix")
  rownames(m) <- paste0("g", seq_len(n_genes))
  colnames(m) <- paste0("c", seq_len(n_cells))
  m
}

test_that("scx_pca_matrix returns finite embeddings + loadings of the right shape", {
  counts <- make_counts()
  res <- scx_pca(counts, n_components = 10L, seed = 0L)
  expect_type(res, "list")
  expect_named(res, c("embeddings", "loadings", "variance_explained",
                      "variance_ratio", "n_components"))
  expect_equal(nrow(res$embeddings), ncol(counts)) # cells
  expect_equal(ncol(res$embeddings), 10L)
  expect_equal(nrow(res$loadings), nrow(counts))   # genes
  expect_true(all(is.finite(res$embeddings)))
  expect_true(all(res$variance_ratio >= 0))
})

test_that("scx_pflog1ppf returns embeddings + a per-cell baseline matching the reference", {
  counts <- make_counts()
  # PFlog1pPF requires raw counts with positive cell depth; drop empty cells.
  counts <- counts[, Matrix::colSums(counts) > 0, drop = FALSE]
  cc <- 1.0
  res <- scx_pflog1ppf(counts, c = cc, n_components = 10L, seed = 0L)

  expect_type(res, "list")
  expect_named(res, c("embeddings", "loadings", "variance_explained",
                      "variance_ratio", "n_components", "baseline"))
  expect_equal(nrow(res$embeddings), ncol(counts)) # cells
  expect_equal(ncol(res$embeddings), 10L)
  expect_equal(nrow(res$loadings), nrow(counts))   # genes
  expect_true(all(is.finite(res$embeddings)))
  expect_true(all(res$variance_ratio >= 0))

  # Baseline is rotation/sign-free, so check it exactly against the reference:
  # baseline_i = -(1/D) * sum_j log(x_ij/s_i + c).
  expect_equal(length(res$baseline), ncol(counts))
  expect_true(all(is.finite(res$baseline)))
  Xc <- as.matrix(counts)                 # genes x cells
  depth <- colSums(Xc)
  L <- log(t(Xc) / depth + cc)            # cells x genes
  ref_baseline <- -rowMeans(L)
  # unname: rowMeans carries cell names; the Rust path returns a plain vector.
  expect_equal(unname(res$baseline), unname(ref_baseline), tolerance = 1e-4)
})

test_that("scx_highly_variable_genes selects n_top genes with finite normalized variance", {
  counts <- make_counts(n_genes = 50L)
  df <- scx_highly_variable_genes(counts, n_top_genes = 15L)
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), nrow(counts))
  expect_true(all(c("means", "variances", "variances_norm", "highly_variable")
                  %in% colnames(df)))
  expect_equal(sum(df$highly_variable), 15L)
  expect_true(all(is.finite(df$variances_norm)))
})

test_that("kNN -> UMAP -> Leiden chain runs on a dense embedding", {
  counts <- make_counts()
  pca <- scx_pca(counts, n_components = 10L, seed = 0L)
  emb <- pca$embeddings

  knn <- scx_neighbors(emb, k = 10L, seed = 0L)
  expect_named(knn, c("n_obs", "n_neighbors", "indices", "distances",
                      "conn_indptr", "conn_indices", "conn_data"))
  expect_equal(knn$n_obs, nrow(emb))
  expect_equal(knn$n_neighbors, 10L)

  um <- scx_umap(emb, n_neighbors = 10L, n_epochs = 50L, seed = 0L)
  expect_equal(nrow(um$embeddings), nrow(emb))
  expect_equal(ncol(um$embeddings), 2L)
  expect_true(all(is.finite(um$embeddings)))

  cl <- scx_leiden(emb, n_neighbors = 10L, resolution = 1.0, seed = 0L)
  expect_length(cl$membership, nrow(emb))
  expect_true(all(cl$membership >= 1L))            # 1-based labels
  expect_equal(cl$n_communities, length(unique(cl$membership)))

  # Reuse a precomputed kNN graph (PR #222 review #2): same connectivity in,
  # same UMAP out — no internal recompute.
  um_reuse <- scx_umap(emb, neighbors = knn, n_epochs = 50L, seed = 0L)
  expect_equal(um_reuse$embeddings, um$embeddings)
})

test_that("scx_rank_genes_groups returns a tidy DE data.frame", {
  counts <- make_counts(n_genes = 30L, n_cells = 100L)
  groups <- rep(c("A", "B"), each = 50L)
  de <- scx_rank_genes_groups(counts, groups = groups, log_transformed = FALSE)
  expect_s3_class(de, "data.frame")
  expect_true(all(c("group", "gene", "score", "pval", "pval_adj",
                    "logfoldchange") %in% colnames(de)))
  expect_true(all(de$gene %in% rownames(counts)))
  expect_true(all(de$pval_adj >= 0 & de$pval_adj <= 1))
})

# ── Numerical correctness (value-level, no Python/Seurat needed) ──────
# Full cross-language parity against pyscx (the T4.1 AC's "cross-check on
# pbmc3k") needs a Python-enabled env and is deferred; these assert the
# kernels' sign/selection semantics directly, which is what would have caught
# the tie_correct default divergence flagged in PR #222 review.

test_that("scx_rank_genes_groups ranks a planted marker top with positive score", {
  set.seed(3)
  ng <- 25L; nc <- 120L
  m <- matrix(rpois(ng * nc, 1), nrow = ng, ncol = nc)
  groups <- rep(c("A", "B"), each = nc / 2L)
  # Gene 1 is strongly up in group A.
  m[1L, groups == "A"] <- m[1L, groups == "A"] + 50L
  counts <- methods::as(Matrix::Matrix(m, sparse = TRUE), "CsparseMatrix")
  rownames(counts) <- paste0("g", seq_len(ng))

  de <- scx_rank_genes_groups(counts, groups = groups, reference = "B",
                              log_transformed = FALSE)
  a <- de[de$group == "A", ]
  a <- a[order(-a$score), ]
  expect_equal(a$gene[1], "g1")            # planted marker ranks first
  expect_gt(a$score[1], 0)                 # up in A → positive score
  # Deterministic across runs.
  de2 <- scx_rank_genes_groups(counts, groups = groups, reference = "B",
                               log_transformed = FALSE)
  expect_equal(de$score, de2$score)
})

test_that("scx_hvg_mean_var matches base-R mean/var (Bessel-corrected)", {
  set.seed(5)
  ng <- 30L; nc <- 200L
  m <- matrix(rpois(ng * nc, 2), nrow = ng, ncol = nc)
  m[7L, ] <- m[7L, ] + rbinom(nc, 1, 0.3) * 40L   # gene 7: clearly overdispersed
  counts <- methods::as(Matrix::Matrix(m, sparse = TRUE), "CsparseMatrix")

  st <- scx_hvg_mean_var(counts)
  # Per-gene mean and (ddof=1) variance must match base R row-wise stats.
  expect_equal(st$means, rowMeans(m), tolerance = 1e-6)
  expect_equal(st$variances, apply(m, 1, var), tolerance = 1e-5)
  # The planted overdispersed gene has the largest raw variance.
  expect_equal(which.max(st$variances), 7L)
})

# ── Seurat-object path (skips when Seurat is absent) ──────────────────

test_that("the accelerators write into Seurat slots end to end", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  library(Seurat)

  counts <- make_counts(n_genes = 60L, n_cells = 150L)
  obj <- CreateSeuratObject(counts = counts)
  obj <- NormalizeData(obj, verbose = FALSE)

  obj <- scx_highly_variable_genes(obj, n_top_genes = 30L)
  expect_length(VariableFeatures(obj), 30L)

  obj <- scx_pca(obj, n_components = 10L)
  expect_true("pca" %in% names(obj@reductions))
  expect_equal(ncol(Embeddings(obj[["pca"]])), 10L)

  obj <- scx_neighbors(obj, dims = 1:10, k = 15L)
  expect_true(any(grepl("_nn$", names(obj@graphs))))

  obj <- scx_umap(obj, dims = 1:10, n_neighbors = 15L)
  expect_true("umap" %in% names(obj@reductions))

  obj <- scx_leiden(obj, dims = 1:10, resolution = 1.0)
  expect_true(!is.null(obj$seurat_clusters))

  de <- scx_rank_genes_groups(obj, group.by = "seurat_clusters")
  expect_s3_class(de, "data.frame")
})
