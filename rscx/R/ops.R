# SCX file operations: append, delete, compact, rollback, and merge SCX files
# from R, plus info/validation utilities. Each exported verb is documented on
# its own help page below (this header is a plain comment so roxygen does not
# attach it to the internal helper that follows).

# Internal: NULL -> character(0); otherwise coerce to character (for the
# optional index-column / sort-column vectors passed to the Rust ops).
.scx_chr_or_empty <- function(x) if (is.null(x)) character(0) else as.character(x)

#' Append cells from one SCX file to another
#'
#' Streams the input's X / obs from a reader and appends them to the target
#' (in place). Both files must have the same number of variables (per modality
#' when \code{modality} is given).
#'
#' @param target Path to the target SCX file (modified in-place).
#' @param input Path to the input SCX file to append from.
#' @param codec Codec for the new shards: \code{NULL}/\code{"auto"} (per-shard
#'   auto-selection) or one of \code{"none"}, \code{"scx1"}, \code{"zstd"},
#'   \code{"lz4"}, \code{"pcodec"}. Note \code{"scx1"} encodes integers only.
#' @param shard_size Target rows per CSR shard (default 16384; must be > 0).
#' @param index_obs,index_var Obs/var columns to build predicate indexes for.
#' @param index_preset Named index preset
#'   (\code{"cellxgene"}/\code{"perturbseq"}/\code{"training"}).
#' @param index_auto_threshold Auto-index cardinality cap (0 = disabled).
#' @param modality Modality NAME to append into on a multimodal file
#'   (\code{NULL} for single-modality / the global axis).
#' @export
#' @examples
#' \dontrun{
#' scx_append("atlas.scx", "new_batch.scx")
#' scx_append("atlas.scx", "rna_batch.scx", codec = "zstd", modality = "rna")
#' }
scx_append <- function(target, input, codec = NULL, shard_size = 16384L,
                       index_obs = NULL, index_var = NULL, index_preset = NULL,
                       index_auto_threshold = 0L, modality = NULL) {
  invisible(.Call(
    wrap__scx_append,
    target,
    input,
    if (is.null(codec)) NULL else as.character(codec),
    as.integer(shard_size),
    .scx_chr_or_empty(index_obs),
    .scx_chr_or_empty(index_var),
    if (is.null(index_preset)) NULL else as.character(index_preset),
    as.integer(index_auto_threshold),
    if (is.null(modality)) NULL else as.character(modality)
  ))
}

#' Mark cells as logically deleted
#'
#' Marks the specified cell indices as deleted using deletion vectors.
#' Deleted cells are excluded from queries but remain on disk until
#' compaction.
#'
#' @param path Path to the SCX file.
#' @param cell_indices Numeric vector of 1-based cell indices to delete (R
#'   convention, matching the \code{[} operator). Passed as doubles so indices
#'   above \code{.Machine$integer.max} (2^31) are addressable (mirrors the backed
#'   reader); values must be whole and \code{>= 1} (\code{0} / negatives raise an
#'   error).
#' @return Total number of deleted cells (including previously deleted).
#' @export
#' @examples
#' \dontrun{
#' total <- scx_delete("experiment.scx", c(1, 6, 11, 43))
#' }
scx_delete <- function(path, cell_indices) {
  idx <- as.numeric(cell_indices)
  if (anyNA(idx)) {
    stop("NA cell indices are not supported", call. = FALSE)
  }
  # R is 1-based (like the `[` operator); reject 0 / negatives before converting
  # to the 0-based core index.
  if (length(idx) && any(idx < 1)) {
    stop("cell indices must be >= 1 (1-based); 0 / negative indices are not supported",
         call. = FALSE)
  }
  # Reject fractional indices up front so the error reports the original value
  # (the Rust core also guards this after the 1->0 conversion, as defense-in-depth).
  if (length(idx) && any(idx != floor(idx))) {
    bad <- idx[idx != floor(idx)][1]
    stop(sprintf("non-integer cell index: %s", format(bad)), call. = FALSE)
  }
  .Call(wrap__scx_delete, path, idx - 1)
}

#' Compact an SCX file
#'
#' Rewrites the file, removing deleted rows and reclaiming space from
#' orphaned sections and stale catalogs. \code{shard_target_rows} is inherited
#' from the input header; CSC sidecars are dropped (rebuild separately).
#'
#' @param input Path to the input SCX file.
#' @param output Path for the compacted output file.
#' @param index_obs,index_var Obs/var columns to build predicate indexes for.
#' @param index_preset Named index preset
#'   (\code{"cellxgene"}/\code{"perturbseq"}/\code{"training"}).
#' @param index_auto_threshold Auto-index cardinality cap (0 = disabled).
#' @param reshape_obs Migrate legacy single-section obs metadata to the
#'   row-sharded layout.
#' @export
#' @examples
#' \dontrun{
#' scx_compact("experiment.scx", "compacted.scx")
#' scx_compact("experiment.scx", "compacted.scx", index_obs = "cell_type")
#' }
scx_compact <- function(input, output, index_obs = NULL, index_var = NULL,
                        index_preset = NULL, index_auto_threshold = 0L,
                        reshape_obs = FALSE) {
  invisible(.Call(
    wrap__scx_compact,
    input,
    output,
    .scx_chr_or_empty(index_obs),
    .scx_chr_or_empty(index_var),
    if (is.null(index_preset)) NULL else as.character(index_preset),
    as.integer(index_auto_threshold),
    as.logical(reshape_obs)
  ))
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
#' Combines two or more SCX files into a single output file. Var identity is
#' validated by default. \code{shard_target_rows} is inherited from the first
#' input; CSC sidecars are dropped (rebuild separately).
#'
#' @param inputs Character vector of input file paths (>= 2).
#' @param output Path for the merged output file.
#' @param index_obs,index_var Obs/var columns to build predicate indexes for.
#' @param index_preset Named index preset
#'   (\code{"cellxgene"}/\code{"perturbseq"}/\code{"training"}).
#' @param index_auto_threshold Auto-index cardinality cap (0 = disabled).
#' @param assume_identical_var,assume_identical_obs Skip the var / obs schema
#'   identity checks across inputs.
#' @param uns_policy How to combine \code{uns}: \code{"first"} (default),
#'   \code{"require-equal"}, \code{"namespace"}, or \code{"summary"}.
#' @param sort_by Obs columns for a globally-ordered (sorted) k-way merge;
#'   \code{NULL} = legacy concatenation.
#' @param reverse Descending order when \code{sort_by} is set.
#' @export
#' @examples
#' \dontrun{
#' scx_merge(c("batch1.scx", "batch2.scx"), "atlas.scx")
#' scx_merge(c("b1.scx", "b2.scx"), "atlas.scx",
#'           uns_policy = "namespace", sort_by = "cell_type")
#' }
scx_merge <- function(inputs, output, index_obs = NULL, index_var = NULL,
                      index_preset = NULL, index_auto_threshold = 0L,
                      assume_identical_var = FALSE, assume_identical_obs = FALSE,
                      uns_policy = "first", sort_by = NULL, reverse = FALSE) {
  invisible(.Call(
    wrap__scx_merge,
    as.character(inputs),
    output,
    .scx_chr_or_empty(index_obs),
    .scx_chr_or_empty(index_var),
    if (is.null(index_preset)) NULL else as.character(index_preset),
    as.integer(index_auto_threshold),
    as.logical(assume_identical_var),
    as.logical(assume_identical_obs),
    as.character(uns_policy),
    .scx_chr_or_empty(sort_by),
    as.logical(reverse)
  ))
}



#' Attach a data.frame of per-cell annotations to an SCX file, in place
#'
#' The R end of the doublet-caller interop. scDblFinder, DoubletFinder and scds
#' all run directly on an rscx-loaded object, so on this path there is no
#' intermediate file in either direction: the caller hands the tool's
#' \code{data.frame} straight back to the SCX file.
#'
#' The join is **by key string, never by row position**. A caller returns rows
#' in whatever order it pleased, and a positional attach would put every score
#' on the wrong cell while still producing a correctly-shaped column. Target
#' rows the \code{data.frame} does not cover become \code{NA}, never a
#' fabricated \code{0}.
#'
#' In place via the same harness \code{scx_append} uses: X, layers, var, the
#' CSC sidecar, .raw, deletion vectors and predicate indexes are preserved, and
#' \code{scx_rollback(path)} undoes the whole attach.
#'
#' @param path Path to the SCX file (modified in place).
#' @param df A \code{data.frame} of annotations. Each column becomes an obs
#'   column; a \code{factor} is preserved as a categorical.
#' @param key Character vector with one key per row of \code{df} — usually
#'   \code{rownames(df)}. Supplying \code{NULL} or an empty vector is an error
#'   rather than a silent fall-through: \code{rownames()} returns \code{NULL}
#'   on an object with no names, and joining on nothing would match nothing.
#'   Omit the argument entirely to resolve a key column from \code{df} instead.
#' @param key_columns Columns **of \code{df}** to fuse into a composite key,
#'   joined against target obs columns of the same names — the right answer for
#'   a multi-library merge where \code{sample_id} + \code{barcode} is unique but
#'   neither is alone. Mutually exclusive with \code{key}.
#' @param key_column Target obs column to join \code{key} against.
#'   \code{NULL} auto-resolves it.
#' @param prefix Prepended to every attached column name.
#' @param status_column Obs column recording \code{"present"}/\code{"absent"}
#'   per row. \code{NULL} omits it.
#' @param uns_key \code{uns} key for the run metadata. \code{NULL} omits it.
#' @param overwrite Replace colliding columns. **Replaces, never merges** — so
#'   attaching several per-batch results in turn keeps only the last. Combine
#'   them into one \code{data.frame} and attach once.
#' @param on_missing_rows \code{"null"} (default) leaves uncovered target rows
#'   \code{NA} and marks them absent; \code{"error"} refuses. \code{"zero"} is
#'   an accepted alias for \code{"null"} — it names the shared policy enum,
#'   whose \code{zero} is literal only for a matrix layer.
#' @param on_extra_rows \code{"warn"} (default) skips source rows the target
#'   lacks; \code{"error"} refuses.
#' @param dry_run Run every validation and the join, then return the summary
#'   without writing.
#' @return A named list with \code{n_obs}, \code{n_matched},
#'   \code{n_target_rows_absent}, \code{n_source_rows_absent},
#'   \code{obs_key_column} and \code{obs_columns_added}.
#' @export
#' @examples
#' \dontrun{
#' library(scDblFinder)
#'
#' exp <- scx_open("library_A.scx")
#' sce <- collect(scx_query(exp))$to_sce()
#' sce <- scDblFinder(sce)
#'
#' # colData() carries the cell keys as ROWNAMES, not as a column.
#' df <- as.data.frame(colData(sce)[, c("scDblFinder.score", "scDblFinder.class")])
#' scx_attach_obs("library_A.scx", df, key = rownames(df))
#' }
scx_attach_obs <- function(path, df, key, key_columns = NULL,
                           key_column = NULL, prefix = "",
                           status_column = NULL, uns_key = NULL,
                           overwrite = FALSE, on_missing_rows = "null",
                           on_extra_rows = "warn", dry_run = FALSE) {
  # `key` supplied but empty is the rownames(df) == NULL trap: an SCE built
  # from a file whose obs carries barcodes only as a named column has NULL
  # colnames, so rownames(colData(sce)) is NULL too. Joining on nothing would
  # match nothing and look like the tool covered no cells.
  if (!missing(key) && (is.null(key) || length(key) == 0L)) {
    stop("`key` was supplied but is NULL/empty — rownames(df) is NULL when the ",
         "object has no cell names. Give the keys explicitly, or use ",
         "`key_columns=` to join on columns of `df`.", call. = FALSE)
  }
  if (missing(key)) key <- character(0)
  if (!is.null(key_columns) && length(key) > 0L) {
    stop("pass either `key` or `key_columns`, not both.", call. = FALSE)
  }

  .Call(
    wrap__scx_attach_obs,
    path,
    df,
    as.character(key),
    .scx_chr_or_empty(key_columns),
    if (is.null(key_column)) NULL else as.character(key_column),
    as.character(prefix),
    if (is.null(status_column)) NULL else as.character(status_column),
    if (is.null(uns_key)) NULL else as.character(uns_key),
    as.logical(overwrite),
    as.character(on_missing_rows),
    as.character(on_extra_rows),
    as.logical(dry_run)
  )
}
