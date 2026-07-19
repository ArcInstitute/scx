#' Harmony2 batch integration
#'
#' Corrects batch effects in PCA-style embeddings using a clean-room Rust
#' implementation of the Harmony2 algorithm (Korsunsky et al.). The core
#' is `scx_accel::harmony_integrate` in the `scx-accel` Rust crate; this
#' is the R front end.
#'
#' Defaults match those in the Python binding
#' (`pyscx.accel.harmony_integrate`) and the scanpy-external API.
#'
#' @param embeddings Numeric matrix of shape `N x d` (cells x PCs). Any
#'   storage mode that coerces to `double` via `as.matrix` is accepted.
#' @param batch Character vector (or factor coerced with `as.character`)
#'   of length `N` with one batch label per cell. Factorised internally.
#' @param n_clusters Number of soft clusters `K` (integer). `NULL` uses
#'   `min(N/30, 100)` clamped to `[2, N/2]`.
#' @param theta Diversity-penalty strength. Scalar; default `2.0`.
#' @param sigma Soft-assignment kernel bandwidth. Scalar; default `0.1`.
#' @param lambda Ridge penalty. `NULL` (default) enables dynamic
#'   estimation (`alpha * E[k, b]`). Supply a non-negative scalar to use
#'   a fixed penalty for every batch level.
#' @param alpha Dynamic-lambda scale factor. Default `0.2`.
#' @param max_iter Maximum Harmony iterations. Default `10L`.
#' @param max_iter_kmeans Maximum k-means sub-iterations per Harmony
#'   iteration. Default `4L`.
#' @param epsilon_harmony Harmony convergence tolerance (signed relative
#'   change). Default `1e-2`.
#' @param epsilon_kmeans K-means convergence tolerance (absolute relative
#'   window change). Default `1e-3`.
#' @param block_size Stochastic block size as a fraction of `N`. Default
#'   `0.05`.
#' @param batch_prop_cutoff Minimum batch proportion per cluster to apply
#'   correction. Default `1e-5`.
#' @param tau Overcorrection protection — scales theta per batch size.
#'   `0` (default) disables scaling; larger values protect smaller
#'   batches.
#' @param random_state RNG seed (integer). Default `0L`.
#'
#' @return A named list:
#'   * `embeddings`: corrected `N x d` numeric matrix.
#'   * `converged`: logical.
#'   * `n_iterations`: integer.
#'   * `n_clusters`: integer — the resolved `K`.
#'   * `objective`: numeric vector of per-iteration Harmony objectives.
#'
#' @examples
#' \dontrun{
#' set.seed(0)
#' N <- 400; d <- 10
#' pca <- matrix(rnorm(N * d), nrow = N, ncol = d)
#' batch <- sample(c("A", "B"), N, replace = TRUE)
#' res <- scx_harmony_integrate(pca, batch, max_iter = 5L)
#' str(res)
#' }
#' @name scx_harmony_integrate
#' @rdname scx_harmony_integrate
#' @export
NULL

#' Seurat-compatible Harmony2 wrapper
#'
#' Extracts a dimensional reduction (e.g. `"pca"`) from a Seurat object,
#' runs [scx_harmony_integrate()] using a `meta.data` column as the batch
#' variable, and writes the corrected embeddings back as a new reduction
#' (default `"harmony"`). This mirrors `harmony::RunHarmony` so existing
#' Seurat pipelines can swap it in with minimal changes.
#'
#' @param object A `Seurat` object (S4).
#' @param group.by.vars Name of a `meta.data` column holding batch
#'   labels. A length-1 character vector. (Multi-covariate support is
#'   exposed at the Rust layer but not yet wired through this wrapper.)
#' @param reduction Name of the input reduction. Default `"pca"`.
#' @param reduction.save Name of the output reduction. Default
#'   `"harmony"`.
#' @param dims.use Optional integer vector of component indices to use
#'   (e.g. `1:30`). `NULL` uses all components.
#' @param ... Forwarded to [scx_harmony_integrate()]. Use to set
#'   `n_clusters`, `theta`, `sigma`, `max_iter`, `random_state`, etc.
#'
#' @return The Seurat object with a new reduction added at
#'   `object[[reduction.save]]`. Also returns invisibly so the function
#'   composes cleanly in pipelines.
#' @export
RunHarmony_scx <- function(object,
                           group.by.vars,
                           reduction = "pca",
                           reduction.save = "harmony",
                           dims.use = NULL,
                           ...) {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("'Seurat' is required for RunHarmony_scx(); install it with `install.packages('Seurat')`",
         call. = FALSE)
  }
  if (length(group.by.vars) != 1L) {
    stop("RunHarmony_scx currently supports exactly one group.by.vars column; ",
         "call scx_harmony_integrate() directly for multi-covariate correction.",
         call. = FALSE)
  }

  red <- object[[reduction]]
  emb <- Seurat::Embeddings(red)
  if (!is.null(dims.use)) {
    emb <- emb[, dims.use, drop = FALSE]
  }

  md <- object[[]]  # full meta.data
  if (!group.by.vars %in% colnames(md)) {
    stop(sprintf("group.by.vars '%s' not found in object@meta.data",
                 group.by.vars),
         call. = FALSE)
  }
  batch <- as.character(md[[group.by.vars]])

  result <- scx_harmony_integrate(emb, batch, ...)
  corrected <- result$embeddings
  dimnames(corrected) <- dimnames(emb)

  new_red <- Seurat::CreateDimReducObject(
    embeddings = corrected,
    key = paste0(toupper(substr(reduction.save, 1, 1)),
                 substring(reduction.save, 2), "_"),
    assay = Seurat::DefaultAssay(object)
  )
  object[[reduction.save]] <- new_red
  invisible(object)
}
