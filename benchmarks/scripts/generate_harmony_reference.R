#!/usr/bin/env Rscript
# Generate R harmony v2.x reference outputs for Phase 6 numerical validation.
#
# Inputs:
#   --pca <path>            Path to pre-computed PCA embeddings (.npy, N x d).
#   --batch <path>           Path to batch labels (.npy, length N, int32).
#   --out-prefix <path>      Output prefix; writes <prefix>.rds + <prefix>.npz.
#   --theta <num>            Diversity penalty (default 2.0).
#   --nclust <int>           Number of clusters K (default 100).
#   --max-iter <int>         Max Harmony iterations (default 10).
#   --seed <int>             RNG seed (default 0).
#
# Writes an RDS + a companion .npz (via reticulate) so Python validation tests
# can load the reference without depending on R at test time.

suppressPackageStartupMessages({
  library(harmony)
  # Use the scx-bench Python (which has numpy) unless the caller overrode it.
  if (Sys.getenv("RETICULATE_PYTHON") == "") {
    Sys.setenv(RETICULATE_PYTHON = "/home/nickyoungblut/miniforge3/envs/scx-bench/bin/python")
  }
  library(reticulate)
})

# --- argument parsing -------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NA) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[i + 1]
}

pca_path    <- get_arg("--pca")
batch_path  <- get_arg("--batch")
out_prefix  <- get_arg("--out-prefix")
theta       <- as.numeric(get_arg("--theta", "2.0"))
nclust      <- as.integer(get_arg("--nclust", "100"))
max_iter    <- as.integer(get_arg("--max-iter", "10"))
seed        <- as.integer(get_arg("--seed", "0"))

stopifnot(
  !is.na(pca_path) && file.exists(pca_path),
  !is.na(batch_path) && file.exists(batch_path),
  !is.na(out_prefix)
)

dir.create(dirname(out_prefix), showWarnings = FALSE, recursive = TRUE)

# --- load inputs via reticulate/numpy --------------------------------
np <- import("numpy", convert = TRUE)
pca <- np$load(pca_path)    # expected shape: N x d, dtype float32/64
batch <- np$load(batch_path)  # expected shape: (N,), int32

# Harmony needs a data.frame for meta_data.
meta <- data.frame(batch = as.character(batch), stringsAsFactors = FALSE)

cat(sprintf("[R] PCA: %d x %d; batch: %d levels\n",
            nrow(pca), ncol(pca), length(unique(meta$batch))))
cat(sprintf("[R] theta=%g, nclust=%d, max_iter=%d, seed=%d\n",
            theta, nclust, max_iter, seed))

# --- run Harmony ------------------------------------------------------
set.seed(seed)
t0 <- Sys.time()
result <- harmony::RunHarmony(
  data_mat = as.matrix(pca),
  meta_data = meta,
  vars_use = "batch",
  theta = theta,
  lambda = NULL,
  nclust = nclust,
  max_iter = max_iter,
  return_object = TRUE,
  verbose = FALSE
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[R] RunHarmony: %.2fs\n", elapsed))

# --- extract state ----------------------------------------------------
# Harmony v2 stores the corrected embedding in result$Z_corr (d x N col-major).
# Transpose so the Python side gets N x d.
z_corr <- t(result$Z_corr)
r_mat  <- result$R
y_mat  <- result$Y
obj    <- as.numeric(result$objective_harmony)
kmeans_rounds <- tryCatch(as.integer(result$kmeans_rounds),
                           error = function(e) length(obj))
# harmony::RunHarmony (Rcpp_harmony reference class) has no `converged`
# field — completion implies it ran to max_iter or hit its internal tol.
# Our Python validation only uses n_iterations ±1, which we track below.
n_iter <- length(obj)

# --- save outputs -----------------------------------------------------
saveRDS(
  list(
    Z_corr = z_corr,
    R = r_mat,
    Y = y_mat,
    objective = obj,
    n_iterations = n_iter,
    kmeans_rounds = kmeans_rounds,
    elapsed_s = elapsed,
    theta = theta, nclust = nclust, max_iter = max_iter, seed = seed
  ),
  paste0(out_prefix, ".rds")
)

# Companion NPZ for Python tests — only the arrays Python consumes.
# Using np$savez would require a Python-side `kwargs` dict which reticulate
# doesn't expose cleanly, so we build the payload via py_run_string.
# Build the NPZ via a tiny Python snippet. `do.call(np$savez, ...)` in
# reticulate doesn't forward R named args as Python **kwargs, so the R
# list collapses to a single positional `arr_0`. Pass the arrays through
# R globals (reticulate exposes R objects to Python via `r.<name>`).
out_npz <- paste0(out_prefix, ".npz")
py_run_string(sprintf("
import numpy as np
np.savez(r'%s',
         Z_corr=np.asarray(r.z_corr, dtype='float64'),
         R=np.asarray(r.r_mat, dtype='float64'),
         Y=np.asarray(r.y_mat, dtype='float64'),
         objective=np.asarray(r.obj, dtype='float64'),
         n_iterations=np.asarray(r.n_iter, dtype='int32'),
         kmeans_rounds=np.asarray(r.kmeans_rounds, dtype='int32'),
         elapsed_s=np.asarray(r.elapsed, dtype='float64'))
", out_npz))

cat(sprintf("[R] wrote %s.{rds,npz}\n", out_prefix))
