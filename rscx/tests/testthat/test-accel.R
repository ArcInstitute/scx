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

test_that("scx_pflog1ppf baseline matches the reference for c != 1", {
  counts <- make_counts()
  counts <- counts[, Matrix::colSums(counts) > 0, drop = FALSE]
  cc <- 0.5
  res <- scx_pflog1ppf(counts, c = cc, n_components = 8L, seed = 0L)
  expect_true(all(is.finite(res$embeddings)))

  # For c != 1 the reference must use the log1p(x/(c*s)) form (the log(x/s + c)
  # shortcut only matches at c = 1, since the two differ by the constant log(c)
  # which cancels under centering only when c = 1 in that specific expansion).
  Xc <- as.matrix(counts)                 # genes x cells
  depth <- colSums(Xc)
  delta <- log1p(t(Xc) / (cc * depth))    # cells x genes
  # Centering denominator D = number of genes = nrow(Xc); rowMeans over the
  # cells x genes delta divides by exactly that.
  ref_baseline <- -rowMeans(delta)
  expect_equal(unname(res$baseline), unname(ref_baseline), tolerance = 1e-4)
})

test_that("scx_pflog1ppf rejects invalid c and empty cells", {
  counts <- make_counts()
  counts <- counts[, Matrix::colSums(counts) > 0, drop = FALSE]
  expect_error(scx_pflog1ppf(counts, c = 0))
  expect_error(scx_pflog1ppf(counts, c = -1))
  expect_error(scx_pflog1ppf(counts, c = Inf))

  # An empty cell (zero column in the genes x cells matrix) has undefined depth.
  empty <- make_counts(n_genes = 10L, n_cells = 5L)
  empty[, 1] <- 0
  empty <- methods::as(empty, "CsparseMatrix")
  expect_error(scx_pflog1ppf(empty, c = 1))
})

test_that("scx_pflog1ppf Seurat path writes reduction + baseline, ignores features", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")

  counts <- make_counts(n_genes = 30L, n_cells = 60L)
  counts <- counts[, Matrix::colSums(counts) > 0, drop = FALSE]
  obj <- SeuratObject::CreateSeuratObject(counts = counts)

  obj <- scx_pflog1ppf(obj, c = 1, n_components = 5L, seed = 0L)
  expect_true("pflog1ppf" %in% names(obj@reductions))
  emb <- SeuratObject::Embeddings(obj[["pflog1ppf"]])
  expect_equal(nrow(emb), ncol(counts))
  expect_equal(ncol(emb), 5L)
  expect_true("pflog1ppf_baseline" %in% colnames(obj@meta.data))

  # Regression for fix #7: setting VariableFeatures must NOT change the
  # full-transcriptome baseline (no feature subsetting in PFlog1pPF).
  SeuratObject::VariableFeatures(obj) <- rownames(counts)[1:5]
  obj2 <- scx_pflog1ppf(obj, c = 1, n_components = 5L, seed = 0L)
  expect_equal(
    unname(obj2$pflog1ppf_baseline),
    unname(obj$pflog1ppf_baseline),
    tolerance = 1e-6
  )
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

# ─── score_genes ────────────────────────────────────────────────────────

test_that("scx_score_genes method='mean' equals the per-cell mean over the set", {
  counts <- make_counts()
  gene_list <- c("g1", "g5", "g10")
  scores <- scx_score_genes(counts, gene_list, method = "mean")

  expect_type(scores, "double")
  expect_length(scores, ncol(counts)) # one score per cell
  expect_equal(names(scores), colnames(counts))

  # Reference: mean of the selected gene rows, per cell.
  idx <- match(gene_list, rownames(counts))
  ref <- colMeans(as.matrix(counts[idx, , drop = FALSE]))
  expect_equal(unname(scores), unname(ref), tolerance = 1e-5)
})

test_that("scx_score_genes control method is deterministic for a fixed seed", {
  counts <- make_counts()
  gl <- paste0("g", 1:5)
  s1 <- scx_score_genes(counts, gl, method = "control", random_state = 7L)
  s2 <- scx_score_genes(counts, gl, method = "control", random_state = 7L)
  expect_equal(s1, s2)
  expect_true(all(is.finite(s1)))
  expect_length(s1, ncol(counts))
})

test_that("scx_score_genes warns on missing genes and errors when none match", {
  counts <- make_counts()
  expect_warning(
    scx_score_genes(counts, c("g1", "nope_gene"), method = "mean"),
    "not found"
  )
  expect_error(
    scx_score_genes(counts, c("nope1", "nope2"), method = "mean"),
    "no genes"
  )
})

# ─── pseudobulk aggregation ─────────────────────────────────────────────

test_that("scx_pseudobulk sum matches a manual per-group rowSums", {
  counts <- make_counts(n_genes = 12L, n_cells = 30L)
  grp <- rep(c("A", "B", "C"), length.out = 30L)

  res <- scx_pseudobulk(counts, group_by = grp, method = "sum")

  # counts is genes x groups.
  expect_equal(nrow(res$counts), nrow(counts))
  expect_equal(ncol(res$counts), 3L)
  expect_equal(rownames(res$counts), rownames(counts))
  expect_setequal(colnames(res$counts), c("A", "B", "C"))

  dense <- as.matrix(counts)
  for (g in c("A", "B", "C")) {
    manual <- rowSums(dense[, grp == g, drop = FALSE])
    expect_equal(unname(res$counts[, g]), unname(manual), tolerance = 1e-6)
    expect_equal(res$samples[g, "n_cells"], sum(grp == g))
  }
})

test_that("scx_pseudobulk mean equals sum divided by cells-per-group", {
  counts <- make_counts(n_genes = 10L, n_cells = 24L)
  grp <- rep(c("X", "Y"), each = 12L)

  s_sum <- scx_pseudobulk(counts, group_by = grp, method = "sum")
  s_mean <- scx_pseudobulk(counts, group_by = grp, method = "mean")

  for (g in c("X", "Y")) {
    n <- s_sum$samples[g, "n_cells"]
    expect_equal(s_mean$counts[, g], s_sum$counts[, g] / n, tolerance = 1e-6)
  }
})

test_that("scx_pseudobulk min_cells_per_group drops small groups", {
  counts <- make_counts(n_genes = 8L, n_cells = 20L)
  # Group "rare" has only 2 cells; "big" has 18.
  grp <- c(rep("rare", 2L), rep("big", 18L))

  res <- scx_pseudobulk(counts, group_by = grp, method = "sum",
                        min_cells_per_group = 5L)
  expect_equal(colnames(res$counts), "big")
  expect_equal(res$samples[["n_cells"]], 18L)
})

test_that("scx_pseudobulk supports multi-column groupby", {
  counts <- make_counts(n_genes = 6L, n_cells = 24L)
  cond <- rep(c("ctrl", "drug"), each = 12L)
  donor <- rep(c("d1", "d2", "d3"), length.out = 24L)
  meta <- data.frame(cond = cond, donor = donor, stringsAsFactors = FALSE)

  res <- scx_pseudobulk(counts, group_by = meta, method = "sum")

  # 2 conditions x 3 donors = 6 groups.
  expect_equal(ncol(res$counts), 6L)
  expect_true(all(c("cond", "donor", "n_cells") %in% colnames(res$samples)))
  expect_equal(sum(res$samples$n_cells), 24L)

  # Spot-check one cell-aligned group sum.
  sel <- cond == "ctrl" & donor == "d1"
  manual <- rowSums(as.matrix(counts)[, sel, drop = FALSE])
  sid <- res$samples$cond == "ctrl" & res$samples$donor == "d1"
  expect_equal(unname(res$counts[, which(sid)]), unname(manual), tolerance = 1e-6)
})

# ─── pseudobulk DE (NB-GLM) ─────────────────────────────────────────────

# Replicated single-cell-like counts: conds x donors pseudobulk groups, with a
# planted up-regulated gene (g1) in the `drug` condition.
.make_replicated <- function(n_genes = 8L, cells_per_group = 20L, seed = 11L) {
  set.seed(seed)
  conds <- c("ctrl", "drug")
  donors <- c("d1", "d2", "d3")
  cell_cond <- character(0)
  cell_donor <- character(0)
  for (co in conds) {
    for (dn in donors) {
      cell_cond <- c(cell_cond, rep(co, cells_per_group))
      cell_donor <- c(cell_donor, rep(dn, cells_per_group))
    }
  }
  n_cells <- length(cell_cond)
  m <- matrix(rpois(n_genes * n_cells, lambda = 5), nrow = n_genes, ncol = n_cells)
  drug <- cell_cond == "drug"
  m[1, drug] <- m[1, drug] + rpois(sum(drug), lambda = 40) # g1 up in drug
  rownames(m) <- paste0("g", seq_len(n_genes))
  colnames(m) <- paste0("c", seq_len(n_cells))
  counts <- methods::as(Matrix::Matrix(m, sparse = TRUE), "CsparseMatrix")
  list(counts = counts,
       meta = data.frame(condition = cell_cond, donor = cell_donor,
                         stringsAsFactors = FALSE),
       n_genes = n_genes)
}

test_that("scx_pseudobulk_dex (NB-GLM) recovers a planted up-regulated gene", {
  d <- .make_replicated()
  de <- scx_pseudobulk_dex(d$counts, group_by = d$meta, test_col = "condition",
                           reference = "ctrl", min_cells_per_group = 5L)

  expect_s3_class(de, "data.frame")
  expect_true(all(c("gene", "baseMean", "log2FoldChange", "lfcSE", "stat",
                    "pvalue", "padj", "target", "reference") %in% colnames(de)))
  expect_equal(nrow(de), d$n_genes) # 1 non-reference target x n_genes
  expect_true(all(de$target == "drug"))
  expect_true(all(de$reference == "ctrl"))
  expect_true(all(de$pvalue >= 0 & de$pvalue <= 1, na.rm = TRUE))

  g1 <- de[de$gene == "g1", ]
  expect_gt(g1$log2FoldChange, 0) # up in drug
})

test_that("scx_pseudobulk_dex skips targets with <2 replicates (warning)", {
  set.seed(12)
  n_genes <- 6L
  cell_cond <- rep(c("ctrl", "drug"), each = 20L)
  cell_donor <- rep("d1", 40L) # single donor -> 1 pseudobulk per condition
  m <- matrix(rpois(n_genes * 40L, 4), nrow = n_genes)
  rownames(m) <- paste0("g", seq_len(n_genes))
  colnames(m) <- paste0("c", seq_len(40L))
  counts <- methods::as(Matrix::Matrix(m, sparse = TRUE), "CsparseMatrix")
  meta <- data.frame(condition = cell_cond, donor = cell_donor,
                     stringsAsFactors = FALSE)

  expect_warning(
    de <- scx_pseudobulk_dex(counts, group_by = meta, test_col = "condition",
                             reference = "ctrl", min_cells_per_group = 5L),
    "skipped"
  )
  expect_equal(nrow(de), 0L)
})

test_that("scx_pseudobulk_dex errors on an unknown test_col / reference", {
  d <- .make_replicated(n_genes = 4L, cells_per_group = 12L)
  expect_error(
    scx_pseudobulk_dex(d$counts, group_by = d$meta, test_col = "nope",
                       reference = "ctrl", min_cells_per_group = 5L),
    "test_col"
  )
  expect_error(
    scx_pseudobulk_dex(d$counts, group_by = d$meta, test_col = "condition",
                       reference = "nope", min_cells_per_group = 5L),
    "reference"
  )
})

test_that("scx_nb_glm fits a planted effect on pre-aggregated counts", {
  set.seed(13)
  n_genes <- 6L
  base <- matrix(rpois(n_genes * 6L, 200), nrow = n_genes) # 3 ctrl + 3 drug samples
  drug_cols <- 4:6
  base[1, drug_cols] <- base[1, drug_cols] + rpois(3L, 400) # g1 up in drug
  rownames(base) <- paste0("g", seq_len(n_genes))
  cond <- factor(c("ctrl", "ctrl", "ctrl", "drug", "drug", "drug"))
  design <- stats::model.matrix(~cond)

  de <- scx_nb_glm(base, design) # default contrast = last coef (conddrug)
  expect_s3_class(de, "data.frame")
  expect_true(all(c("gene", "baseMean", "log2FoldChange", "lfcSE", "stat",
                    "pvalue", "padj", "dispersion", "converged") %in% colnames(de)))
  expect_equal(nrow(de), n_genes)
  expect_true(all(de$pvalue >= 0 & de$pvalue <= 1, na.rm = TRUE))

  g1 <- de[de$gene == "g1", ]
  expect_gt(g1$log2FoldChange, 0) # up in drug

  # n_samples must exceed n_features: a rank-deficient/too-small design errors.
  expect_error(scx_nb_glm(base[, 1:2], design[1:2, , drop = FALSE]))
})
