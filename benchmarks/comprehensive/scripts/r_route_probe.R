#!/usr/bin/env Rscript
# Route-metadata probe for the rscx accelerators (ORG-10.16-3b).
#
# Runs a handful of rscx accelerator calls on small synthetic matrices and
# prints, as JSON on stdout, the `scx_accel` route records they stamped plus
# the facts the `accel_r_route` gate floors:
#
#   * `scx_pca` on both sides of COVARIANCE_PCA_THRESHOLD — the one real
#     dispatch branch rscx owns, observable via the returned `method`;
#   * `scx_rank_genes_groups` — must honestly stamp `cpu_dense` (the kernel
#     densifies), not the `cpu_csr` the dgCMatrix input suggests;
#   * `scx_pseudobulk_dex` — must stamp `cpu_nb_glm` with fallback `"none"`
#     (pyscx parity: NB-GLM is a first-class native CPU route).
#
# Synthetic data on purpose: the gate asserts routing facts, which are
# dataset-independent, so it must not depend on a converted fixture existing.
# Runs inside the `rscx` conda env (see TOOL ENV resolution in
# `benchmarks/comprehensive/benchmarks/accel_r_route.py`).

suppressPackageStartupMessages({
  library(rscx)
  library(Matrix)
  library(jsonlite)
})

make_counts <- function(n_genes, n_cells, seed = 1L) {
  set.seed(seed)
  m <- Matrix::rsparsematrix(
    n_genes, n_cells,
    density = 0.3,
    rand.x = function(n) sample.int(20L, n, replace = TRUE)
  )
  m <- methods::as(abs(m), "CsparseMatrix")
  rownames(m) <- paste0("g", seq_len(n_genes))
  colnames(m) <- paste0("c", seq_len(n_cells))
  m
}

t0 <- proc.time()[["elapsed"]]

# --- PCA: both arms of the shared covariance-vs-randomized auto rule -------
small <- make_counts(40L, 120L)
pca_small <- scx_pca(small, n_components = 5L, seed = 0L)

set.seed(7)
big <- Matrix::rsparsematrix(5001L, 30L, density = 0.01,
                             rand.x = function(n) rep(1, n))
big <- methods::as(abs(big), "CsparseMatrix")
pca_big <- scx_pca(big, n_components = 3L, seed = 0L)

# --- Wilcoxon: the densifying kernel must stamp cpu_dense -------------------
counts <- make_counts(30L, 100L)
groups <- rep(c("A", "B"), each = 50L)
de <- scx_rank_genes_groups(counts, groups = groups, log_transformed = FALSE)
de_route <- attr(de, "scx_accel")

# --- pseudobulk_dex: cpu_nb_glm / none (pyscx parity) -----------------------
set.seed(11)
conds <- rep(rep(c("ctrl", "drug"), each = 60L))
donors <- rep(rep(c("d1", "d2", "d3"), each = 20L), times = 2L)
m <- matrix(rpois(8L * 120L, lambda = 5), nrow = 8L)
rownames(m) <- paste0("g", 1:8)
colnames(m) <- paste0("c", 1:120)
dex_counts <- methods::as(Matrix::Matrix(m, sparse = TRUE), "CsparseMatrix")
meta <- data.frame(condition = conds, donor = donors, stringsAsFactors = FALSE)
dex <- scx_pseudobulk_dex(dex_counts, group_by = meta, test_col = "condition",
                          reference = "ctrl", min_cells_per_group = 5L)
dex_route <- attr(dex, "scx_accel")

wall_s <- proc.time()[["elapsed"]] - t0

result <- list(
  pca_route = pca_small$scx_accel$route,
  pca_fallback = pca_small$scx_accel$fallback_reason,
  pca_method_small = pca_small$method,
  pca_method_big = pca_big$method,
  pca_record_keys = names(pca_small$scx_accel),
  wilcoxon_route = de_route$route,
  wilcoxon_fallback = de_route$fallback_reason,
  dex_route = dex_route$route,
  dex_fallback = dex_route$fallback_reason,
  wall_s = wall_s
)
cat(toJSON(result, auto_unbox = TRUE))
