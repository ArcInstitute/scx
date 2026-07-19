#' Local Inverse Simpson Index (LISI)
#'
#' Computes the per-cell LISI metric (Korsunsky et al., 2019) on a
#' numeric embedding matrix with categorical labels. LISI approaches 1
#' when neighbourhoods share a single label (poor mixing) and approaches
#' the number of categories under uniform mixing (good mixing). Useful
#' as a batch-integration QC summary before and after running
#' [scx_harmony_integrate()].
#'
#' @param embeddings Numeric matrix (N rows × d columns).
#' @param labels Character vector (or factor coerced with `as.character`)
#'   of length `N`.
#' @param perplexity Gaussian-kernel target perplexity (default 30).
#' @param n_neighbors Number of neighbours to use. `NULL` uses
#'   `ceil(3 * perplexity)`.
#'
#' @return Numeric vector of length `N` with LISI values.
#' @export
#' @examples
#' \dontrun{
#' set.seed(0)
#' N <- 300; d <- 10
#' emb <- matrix(rnorm(N * d), nrow = N, ncol = d)
#' lab <- sample(c("A", "B"), N, replace = TRUE)
#' mean(scx_compute_lisi(emb, lab))
#' }
#' @name scx_compute_lisi
#' @rdname scx_compute_lisi
NULL
