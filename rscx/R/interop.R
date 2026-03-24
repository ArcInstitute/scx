#' @title SCX interop with Seurat v5 and SingleCellExperiment
#' @name rscx-interop
#' @useDynLib rscx, .registration = TRUE
#' @importFrom methods new
#' @importFrom Matrix t
#' @description
#' Convert between SCX and R single-cell data structures.
#'
#' The \code{to_seurat()} and \code{to_sce()} methods on \code{RQueryResult}
#' construct Seurat v5 and SingleCellExperiment objects directly from query
#' results. For import, use \code{from_seurat()} and \code{from_sce()} to
#' write R objects to SCX files.
#'
#' @section Orientation:
#' SCX stores CSR (cells x genes). Seurat and SingleCellExperiment both
#' expect genes x cells. The conversion functions handle the transpose
#' automatically.
#'
#' @examples
#' \dontrun{
#' # ── Export: SCX → Seurat v5 ──
#' result <- scx_open("experiment.scx") |>
#'   scx_query() |>
#'   filter_obs("tissue == 'lung'") |>
#'   collect()
#' seu <- result$to_seurat()
#'
#' # ── Export: SCX → SingleCellExperiment ──
#' sce <- result$to_sce()
#'
#' # ── Import: Seurat → SCX ──
#' from_seurat(seu, "output.scx")
#'
#' # ── Import: SingleCellExperiment → SCX ──
#' from_sce(sce, "output.scx")
#' }
NULL
