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

# Deliberately no jsonlite (or any package outside rscx's own Imports): the
# gate maps a load failure of *rscx* to a typed `no_rscx_env` skip, and a
# missing convenience package must not be able to masquerade as that skip
# and silently vacate the floors (review round 1, Cursor Agent). The JSON is
# a handful of scalars plus one string array — base R emits it directly.
suppressPackageStartupMessages({
  library(rscx)
  library(Matrix)
})

json_str <- function(x) paste0('"', gsub('"', '\\"', x, fixed = TRUE), '"')
json_arr <- function(v) paste0("[", paste(vapply(v, json_str, ""), collapse = ","), "]")

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

# --- harmony: pyscx stamps harmony_integrate; rscx must too ----------------
set.seed(3)
hm <- scx_harmony_integrate(matrix(rnorm(80 * 6), nrow = 80, ncol = 6),
                            rep(c("A", "B"), each = 40L), max_iter = 2L)

wall_s <- proc.time()[["elapsed"]] - t0

# One JSON object on the LAST stdout line (the gate parses the final {...}
# from stdout, so R warnings or profile chatter above cannot corrupt it).
cat(sprintf(
  paste0('{"pca_route":%s,"pca_fallback":%s,"pca_method_small":%s,',
         '"pca_method_big":%s,"pca_record_keys":%s,"wilcoxon_route":%s,',
         '"wilcoxon_fallback":%s,"dex_route":%s,"dex_fallback":%s,',
         '"harmony_route":%s,"harmony_fallback":%s,"wall_s":%.3f}'),
  json_str(pca_small$scx_accel$route),
  json_str(pca_small$scx_accel$fallback_reason),
  json_str(pca_small$method),
  json_str(pca_big$method),
  json_arr(names(pca_small$scx_accel)),
  json_str(de_route$route),
  json_str(de_route$fallback_reason),
  json_str(dex_route$route),
  json_str(dex_route$fallback_reason),
  json_str(hm$scx_accel$route),
  json_str(hm$scx_accel$fallback_reason),
  wall_s
))
