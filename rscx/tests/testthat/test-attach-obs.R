# scx_attach_obs: land an R data.frame of per-cell annotations on an SCX file.
#
# The R end of the doublet-caller interop, where scDblFinder / DoubletFinder /
# scds run directly on an rscx-loaded object and there is no intermediate file
# in either direction. What has to be right is the join: a caller returns rows
# in its own order, so these fixtures deliberately scramble the data.frame
# relative to the file.

# Read a fresh copy of the fixture's obs as a data.frame.
attach_fixture <- function() {
  src <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  file.copy(src, tmp)
  tmp
}

obs_of <- function(path) {
  collect(scx_query(scx_open(path)))$obs()
}

test_that("scx_attach_obs joins by key, not by row position", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))
  expect_gt(length(keys), 3)

  # Reverse the order and cover only the first three cells: a positional
  # attach would put every score on the wrong cell and still look fine.
  covered <- keys[1:3]
  df <- data.frame(
    dbl_score = c(0.3, 0.2, 0.1),
    dbl_class = c("doublet", "singlet", "singlet"),
    stringsAsFactors = FALSE
  )
  rownames(df) <- rev(covered)

  res <- scx_attach_obs(path, df, key = rownames(df),
                        status_column = "dbl_status")

  expect_equal(res$n_matched, 3)
  expect_equal(res$n_obs, length(keys))
  expect_true("dbl_score" %in% res$obs_columns_added)

  back <- obs_of(path)
  # File order, not data.frame order: cell 1 got 0.1, cell 3 got 0.3.
  expect_equal(back$dbl_score[1], 0.1)
  expect_equal(back$dbl_score[3], 0.3)
  # A cell the caller did not cover is NA, never a fabricated 0.
  expect_true(is.na(back$dbl_score[4]))
  expect_equal(back$dbl_status[4], "absent")
})

test_that("a factor column survives as a factor", {
  # scDblFinder's class column is a factor. It must come back as one — levels
  # in the declared order, an unused level kept — not as the character vector
  # the attach used to decode it to. (The old assertion accepted either, which
  # is how the decode went unnoticed.)
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))[1:2]
  df <- data.frame(dbl_class = factor(c("singlet", "doublet"),
                                      levels = c("singlet", "doublet", "unsure")))
  rownames(df) <- keys

  scx_attach_obs(path, df, key = rownames(df))
  back <- obs_of(path)
  expect_true(is.factor(back$dbl_class))
  expect_false(is.ordered(back$dbl_class))
  expect_equal(levels(back$dbl_class), c("singlet", "doublet", "unsure"))
  expect_equal(as.character(back$dbl_class[1]), "singlet")
  expect_equal(as.character(back$dbl_class[2]), "doublet")
  # A cell the caller did not cover is NA, not a fabricated level.
  expect_true(is.na(back$dbl_class[3]))
})

test_that("an ordered factor keeps its order through a prefixed attach", {
  # `prefix=` renames every attached column; the rename must carry the field
  # metadata that records `ordered`, or the factor comes back unordered.
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))[1:3]
  df <- data.frame(phase = factor(c("G2M", "G1", "S"),
                                  levels = c("G1", "S", "G2M", "M"),
                                  ordered = TRUE))
  rownames(df) <- keys

  scx_attach_obs(path, df, key = rownames(df), prefix = "cc_")
  back <- obs_of(path)
  expect_true(is.ordered(back$cc_phase))
  expect_equal(levels(back$cc_phase), c("G1", "S", "G2M", "M"))
  expect_equal(as.character(back$cc_phase[1:3]), c("G2M", "G1", "S"))
})

test_that("prefix is applied to the attached columns", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))[1:2]
  df <- data.frame(score = c(0.1, 0.9))
  rownames(df) <- keys

  res <- scx_attach_obs(path, df, key = rownames(df), prefix = "scdbl_")
  expect_true("scdbl_score" %in% res$obs_columns_added)
  expect_true("scdbl_score" %in% names(obs_of(path)))
})

test_that("dry_run reports the join without writing", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  before <- file.size(path)
  keys <- rownames(obs_of(path))[1:2]
  df <- data.frame(score = c(0.1, 0.9))
  rownames(df) <- keys

  res <- scx_attach_obs(path, df, key = rownames(df), dry_run = TRUE)
  expect_equal(res$n_matched, 2)
  expect_true(res$dry_run)
  expect_equal(file.size(path), before)
  expect_false("score" %in% names(obs_of(path)))
})

test_that("scx_rollback undoes an attach", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))[1:2]
  df <- data.frame(score = c(0.1, 0.9))
  rownames(df) <- keys

  scx_attach_obs(path, df, key = rownames(df))
  expect_true("score" %in% names(obs_of(path)))

  scx_rollback(path)
  expect_false("score" %in% names(obs_of(path)))
})

test_that("a second attach errors and overwrite replaces", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))
  a <- data.frame(score = c(1, 2)); rownames(a) <- keys[1:2]
  b <- data.frame(score = c(3, 4)); rownames(b) <- keys[3:4]

  scx_attach_obs(path, a, key = rownames(a))
  expect_error(scx_attach_obs(path, b, key = rownames(b)), "already exists")

  scx_attach_obs(path, b, key = rownames(b), overwrite = TRUE)
  back <- obs_of(path)
  # overwrite REPLACES: batch a's values are gone. Attaching per-batch results
  # in turn keeps only the last -- combine and attach once.
  expect_true(is.na(back$score[1]))
  expect_equal(back$score[3], 3)
})

# B3: fallible ops must raise a clean R stop() carrying the real message, not
# the opaque "User function panicked" from extendr's unwrap()-panic.
test_that("scx_attach_obs raises clean R errors, not Rust panics", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  df <- data.frame(score = c(0.1, 0.9))
  rownames(df) <- c("NOT-A-CELL-1", "NOT-A-CELL-2")

  msg <- tryCatch(scx_attach_obs(path, df, key = rownames(df)),
                  error = function(e) conditionMessage(e))
  expect_false(grepl("panicked", msg, fixed = TRUE))
  # Zero overlap names both sides so the mismatch is diagnosable.
  expect_true(grepl("NOT-A-CELL-1", msg, fixed = TRUE))
})

test_that("a NULL or empty key is refused rather than joined on nothing", {
  # The real trap: rownames(colData(sce)) is NULL when the SCE was built from a
  # file whose obs carries barcodes only as a named column, so colnames(sce) is
  # NULL too. Falling through to auto-resolution would match nothing and look
  # like the tool covered no cells.
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  df <- data.frame(score = c(0.1, 0.9))
  expect_error(scx_attach_obs(path, df, key = NULL), "NULL/empty")
  expect_error(scx_attach_obs(path, df, key = character(0)), "NULL/empty")
})

test_that("key length must match the data.frame row count", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))
  df <- data.frame(score = c(0.1, 0.9))
  msg <- tryCatch(scx_attach_obs(path, df, key = keys[1:3]),
                  error = function(e) conditionMessage(e))
  expect_true(grepl("3 values", msg) || grepl("rows", msg))
})

test_that("key and key_columns are mutually exclusive", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))[1:2]
  df <- data.frame(barcode = keys, score = c(0.1, 0.9))
  expect_error(
    scx_attach_obs(path, df, key = keys, key_columns = "barcode"),
    "not both"
  )
})

test_that("key_column without key is an error, not a silent no-op", {
  # `key_column` names the TARGET obs column that `key`'s values are matched
  # against, and the Rust side reads it only on the `key` branch — so alone, or
  # beside `key_columns`, it used to be accepted and ignored while the join ran
  # against something else. Same guard as scx_attach_var.
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))
  df <- data.frame(score = seq_along(keys) * 1.0, alt = keys,
                   stringsAsFactors = FALSE)

  expect_error(scx_attach_obs(path, df, key_columns = "alt",
                              key_column = "barcode"),
               "needs `key=")
})

test_that("key_columns joins on a column of the data.frame", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  obs <- obs_of(path)
  # `key_columns` names columns that must exist on BOTH sides: they are how
  # the source rows are fused and what the target obs is matched against.
  ids <- obs$cell_id[1:2]
  df <- data.frame(cell_id = ids, score = c(0.1, 0.9), stringsAsFactors = FALSE)

  res <- scx_attach_obs(path, df, key_columns = "cell_id", status_column = "st")
  expect_equal(res$n_matched, 2)
  back <- obs_of(path)
  expect_equal(back$score[1], 0.1)
  # The key column itself is consumed: it is already on the obs axis.
  expect_false("cell_id" %in% res$obs_columns_added)
})

test_that("the rownames column is consumed, not attached alongside the key", {
  # `dataframe_to_record_batch` turns rownames(df) into `__index_level_0__`.
  # Those rownames ARE the key in the flagship recipe, so re-attaching them
  # would collide with the target's own index column on every single attach.
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  keys <- rownames(obs_of(path))[1:2]
  df <- data.frame(score = c(0.1, 0.9))
  rownames(df) <- keys

  res <- scx_attach_obs(path, df, key = rownames(df))
  expect_equal(res$obs_columns_added, "score")
  expect_false("__index_level_0__" %in% res$obs_columns_added)
})

test_that("an empty data.frame is refused", {
  path <- attach_fixture()
  on.exit(unlink(path), add = TRUE)

  msg <- tryCatch(
    scx_attach_obs(path, data.frame(score = numeric(0)), key = character(0)),
    error = function(e) conditionMessage(e)
  )
  expect_false(grepl("panicked", msg, fixed = TRUE))
})
