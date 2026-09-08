# scx_attach_var: land an R data.frame of per-gene annotations on an SCX file.
#
# The var-axis twin of test-attach-obs.R. What has to be right is the same
# thing: the join. An annotation table returns rows in its own order, so these
# fixtures deliberately scramble the data.frame relative to the file — a
# positional attach would put every value on the wrong gene and still produce a
# correctly-shaped column.

attach_var_fixture <- function() {
  src <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  file.copy(src, tmp)
  tmp
}

var_of <- function(path) {
  collect(scx_query(scx_open(path)))$var()
}

test_that("scx_attach_var joins by key, not by row position", {
  path <- attach_var_fixture()
  on.exit(unlink(path), add = TRUE)

  genes <- rownames(var_of(path))
  expect_gt(length(genes), 3)

  # Reverse the order and cover only the first three genes.
  covered <- genes[1:3]
  df <- data.frame(
    peak_score = c(0.3, 0.2, 0.1),
    peak_class = c("promoter", "enhancer", "enhancer"),
    stringsAsFactors = FALSE
  )
  rownames(df) <- rev(covered)

  res <- scx_attach_var(path, df, key = rownames(df),
                        status_column = "ann_status")

  expect_equal(res$n_matched, 3)
  expect_equal(res$n_vars, length(genes))
  expect_true("peak_score" %in% res$var_columns_added)

  back <- var_of(path)
  # File order, not data.frame order: gene 1 got 0.1, gene 3 got 0.3.
  expect_equal(back$peak_score[1], 0.1)
  expect_equal(back$peak_score[3], 0.3)
  # A gene the caller did not cover is NA, never a fabricated 0.
  expect_true(is.na(back$peak_score[4]))
  expect_equal(back$ann_status[4], "absent")
})

test_that("a factor column survives as a factor", {
  path <- attach_var_fixture()
  on.exit(unlink(path), add = TRUE)

  genes <- rownames(var_of(path))
  df <- data.frame(
    peak_class = factor(rep(c("promoter", "enhancer"), length.out = length(genes)),
                        levels = c("promoter", "enhancer", "intergenic"))
  )
  rownames(df) <- genes

  scx_attach_var(path, df, key = rownames(df))
  back <- var_of(path)
  expect_s3_class(back$peak_class, "factor")
  # The declared levels, including the unused one, and their order.
  expect_equal(levels(back$peak_class), c("promoter", "enhancer", "intergenic"))
})

test_that("prefix is applied to every imported column", {
  path <- attach_var_fixture()
  on.exit(unlink(path), add = TRUE)

  genes <- rownames(var_of(path))
  df <- data.frame(score = seq_along(genes) * 1.0)
  rownames(df) <- genes

  res <- scx_attach_var(path, df, key = rownames(df), prefix = "pk_")
  expect_true("pk_score" %in% res$var_columns_added)
  expect_true("pk_score" %in% names(var_of(path)))
})

test_that("dry_run reports the join without writing", {
  path <- attach_var_fixture()
  on.exit(unlink(path), add = TRUE)

  before <- file.size(path)
  genes <- rownames(var_of(path))
  df <- data.frame(score = c(1.0, 2.0))
  rownames(df) <- genes[1:2]

  res <- scx_attach_var(path, df, key = rownames(df), dry_run = TRUE)
  expect_true(res$dry_run)
  expect_equal(res$n_matched, 2)
  expect_equal(file.size(path), before)
  expect_false("score" %in% names(var_of(path)))
})

test_that("scx_rollback undoes the attach", {
  path <- attach_var_fixture()
  on.exit(unlink(path), add = TRUE)

  genes <- rownames(var_of(path))
  df <- data.frame(score = seq_along(genes) * 1.0)
  rownames(df) <- genes
  scx_attach_var(path, df, key = rownames(df))
  expect_true("score" %in% names(var_of(path)))

  scx_rollback(path)
  expect_false("score" %in% names(var_of(path)))
})

test_that("a second attach errors and overwrite replaces rather than merging", {
  path <- attach_var_fixture()
  on.exit(unlink(path), add = TRUE)

  genes <- rownames(var_of(path))
  first <- data.frame(score = c(1.0, 1.0))
  rownames(first) <- genes[1:2]
  scx_attach_var(path, first, key = rownames(first))

  second <- data.frame(score = c(2.0, 2.0))
  rownames(second) <- genes[3:4]
  expect_error(scx_attach_var(path, second, key = rownames(second)),
               "overwrite")

  scx_attach_var(path, second, key = rownames(second), overwrite = TRUE)
  back <- var_of(path)
  # Replace, not merge: the first attach's genes are NA again.
  expect_true(is.na(back$score[1]))
  expect_equal(back$score[3], 2.0)
})

test_that("bad arguments raise clean R errors, not Rust panics", {
  path <- attach_var_fixture()
  on.exit(unlink(path), add = TRUE)

  genes <- rownames(var_of(path))
  df <- data.frame(score = seq_along(genes) * 1.0)
  rownames(df) <- genes

  expect_error(scx_attach_var(path, df, key = NULL), "NULL/empty")
  expect_error(
    scx_attach_var(path, df, key = rownames(df), key_columns = "score"),
    "not both"
  )
  # `key_column` names the TARGET side of a `key` join, so it means nothing
  # without one. It used to be accepted and silently ignored.
  expect_error(
    scx_attach_var(path, df, key_columns = "score", key_column = "gene_id"),
    "needs `key="
  )
  expect_error(scx_attach_var(path, df, key = genes[1:2]),
               "`key` has 2 values")
  expect_error(
    scx_attach_var(path, df, key = rownames(df), on_missing_rows = "nope"),
    "on_missing_rows"
  )
  # `key` omitted, not empty: an explicit `character(0)` trips the
  # supplied-but-empty guard above first, which is the right order.
  expect_error(scx_attach_var(path, data.frame()), "no rows")
})

test_that("key_columns joins on a column of df", {
  path <- attach_var_fixture()
  on.exit(unlink(path), add = TRUE)

  genes <- rownames(var_of(path))
  # The var index of the fixture is the only unique gene identifier it has, so
  # the composite / named-column path is exercised against it by name.
  df <- data.frame(
    var_names = rev(genes),
    score = seq_along(genes) * 1.0,
    stringsAsFactors = FALSE
  )

  res <- scx_attach_var(path, df, key_columns = "var_names")
  expect_equal(res$n_matched, length(genes))
  back <- var_of(path)
  # Reversed source: gene 1 must carry the LAST score.
  expect_equal(back$score[1], length(genes) * 1.0)
})
