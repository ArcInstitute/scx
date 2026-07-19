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
#' @section Codecs and framing:
#' \code{from_seurat()}, \code{from_sce()}, and \code{from_mae()} accept a
#' \code{codec} argument selecting the same codec intent axis as
#' \code{pyscx.from_anndata} and the \code{scx} CLI (resolved by the shared
#' \code{scx_format::resolve_codec}):
#' \describe{
#'   \item{\code{"auto"} (default)}{Cost-aware adaptive: per integer shard,
#'     adopt \code{ShufDeltaZstd} when it is smaller by a margin, else the
#'     heuristic (Scx1 for low-count / Zstd otherwise). Float always uses
#'     Pcodec.}
#'   \item{\code{"fast"}}{Decode-optimized heuristic single-encode (the prior
#'     default behaviour); never trials \code{ShufDeltaZstd}.}
#'   \item{\code{"compact"}}{Size-optimized: adopts \code{ShufDeltaZstd} on ties.
#'     Requires framing (\code{row_group_rows > 0}).}
#'   \item{explicit codecs}{\code{"none"}, \code{"scx1"}, \code{"zstd"},
#'     \code{"lz4"}, \code{"pcodec"}, \code{"shufdelta"}.}
#' }
#' \code{row_group_rows} controls row-group framing (sub-shard random access);
#' the default (\code{NULL}) frames at 256 rows to match pyscx/CLI, producing v4
#' files. Pass \code{0L} to opt out of framing (which the \code{"compact"} /
#' \code{"compact-trial"} profiles reject). Framed files require an SCX reader
#' that understands the v4 layout.
#'
#' @param seurat_obj A \code{Seurat} object to import (\code{from_seurat}).
#' @param sce_obj A \code{SingleCellExperiment} to import (\code{from_sce}).
#' @param mae_obj A \code{MultiAssayExperiment} to import as a multimodal SCX
#'   file (\code{from_mae}); requires aligned cell axes across experiments.
#' @param output_path Destination \code{.scx} path.
#' @param codec Codec intent (\code{"auto"}/\code{"fast"}/\code{"compact"} or an
#'   explicit codec name; see \dQuote{Codecs and framing}). \code{NULL} =
#'   \code{"auto"}.
#' @param csc Also write a gene-major CSC sidecar (logical; default
#'   \code{FALSE}).
#' @param csc_cols_per_shard Columns per CSC shard when \code{csc = TRUE}
#'   (default \code{5000L}).
#' @param row_group_rows Row-group framing size; \code{NULL} frames at 256 rows
#'   (v4), \code{0L} disables framing.
#' @return Invisibly \code{NULL}; called for the side effect of writing
#'   \code{output_path}.
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
#'
#' # ── Size-optimized import (adaptive ShufDeltaZstd, framed) ──
#' from_seurat(seu, "compact.scx", codec = "compact")
#' }
NULL

# Companion doc blocks attaching the extendr-generated importers (defined in
# R/extendr-wrappers.R) to the shared `rscx-interop` help page. The `@export`
# for each comes from the wrapper; these only supply the doc destination.

#' @rdname rscx-interop
#' @name from_seurat
NULL

#' @rdname rscx-interop
#' @name from_sce
NULL

#' @rdname rscx-interop
#' @name from_mae
NULL
