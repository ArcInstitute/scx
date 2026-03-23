#' @title SCX file operations
#'
#' @description
#' Append, delete, compact, rollback, and merge SCX files from R.
#'
#' @examples
#' # Append new cells
#' scx_append("atlas.scx", "new_batch.scx")
#'
#' # Delete specific cells
#' scx_delete("experiment.scx", c(0L, 5L, 10L, 42L))
#'
#' # Compact to reclaim space
#' scx_compact("experiment.scx", "compacted.scx")
#'
#' # Roll back one version
#' scx_rollback("experiment.scx")
#'
#' # Merge multiple files
#' scx_merge(c("batch1.scx", "batch2.scx"), "atlas.scx")
NULL
