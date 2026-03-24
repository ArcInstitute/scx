#' @title SCX file operations
#'
#' @description
#' Append, delete, compact, rollback, and merge SCX files from R.
#' Info and validation utilities.

#' Append cells from one SCX file to another
#'
#' Reads all CSR shards and obs metadata from the input file and appends
#' them to the target file. Both files must have the same number of variables.
#'
#' @param target Path to the target SCX file (modified in-place).
#' @param input Path to the input SCX file to append from.
#' @export
#' @examples
#' \dontrun{
#' scx_append("atlas.scx", "new_batch.scx")
#' }
scx_append <- function(target, input) {
  invisible(.Call(wrap__scx_append, target, input))
}

#' Mark cells as logically deleted
#'
#' Marks the specified cell indices as deleted using deletion vectors.
#' Deleted cells are excluded from queries but remain on disk until
#' compaction.
#'
#' @param path Path to the SCX file.
#' @param cell_indices Integer vector of 0-based cell indices to delete.
#' @return Total number of deleted cells (including previously deleted).
#' @export
#' @examples
#' \dontrun{
#' total <- scx_delete("experiment.scx", c(0L, 5L, 10L, 42L))
#' }
scx_delete <- function(path, cell_indices) {
  .Call(wrap__scx_delete, path, as.integer(cell_indices))
}

#' Compact an SCX file
#'
#' Rewrites the file, removing deleted rows and reclaiming space from
#' orphaned sections and stale catalogs.
#'
#' @param input Path to the input SCX file.
#' @param output Path for the compacted output file.
#' @export
#' @examples
#' \dontrun{
#' scx_compact("experiment.scx", "compacted.scx")
#' }
scx_compact <- function(input, output) {
  invisible(.Call(wrap__scx_compact, input, output))
}

#' Roll back to a previous manifest version
#'
#' Reverts the SCX file to a previous state. If \code{to_seq} is NULL,
#' rolls back one version. If specified, rolls back to that manifest
#' sequence number.
#'
#' @param path Path to the SCX file.
#' @param to_seq Optional manifest sequence number (integer or NULL).
#' @export
#' @examples
#' \dontrun{
#' scx_rollback("experiment.scx")
#' scx_rollback("experiment.scx", to_seq = 3L)
#' }
scx_rollback <- function(path, to_seq = NULL) {
  invisible(.Call(wrap__scx_rollback, path, to_seq))
}

#' Merge multiple SCX files
#'
#' Combines two or more SCX files into a single output file.
#' All input files must have the same number of variables (genes).
#'
#' @param inputs Character vector of input file paths (>= 2).
#' @param output Path for the merged output file.
#' @export
#' @examples
#' \dontrun{
#' scx_merge(c("batch1.scx", "batch2.scx"), "atlas.scx")
#' }
scx_merge <- function(inputs, output) {
  invisible(.Call(wrap__scx_merge, inputs, output))
}


