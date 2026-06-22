# Regression tests for dgCMatrix slot validation on the import path.
#
# `dgcmatrix_to_csr` (used by `from_seurat` / `from_sce`) consumes the @i / @p /
# @Dim slots of an arbitrary, possibly host-built S4 object as loop bounds and
# indices. A malformed slot must surface as a catchable R error (`stop()` via
# `throw_on_err`), NOT an uncatchable R-session abort from an out-of-bounds
# index panic across the FFI boundary.
#
# Slot assignment (`@<-`) bypasses validObject, so we corrupt a valid dgCMatrix
# after construction and wrap it in an SCE whose dim checks still pass.

skip_if_no_sce <- function() {
  skip_if_not_installed("Matrix")
  skip_if_not_installed("SingleCellExperiment")
}

# A small, valid 5-gene x 4-cell dgCMatrix (genes x cells, as Seurat/SCE store).
valid_dgc <- function() {
  Matrix::Matrix(
    matrix(
      c(0, 1, 0, 2, 3, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7),
      nrow = 5, byrow = TRUE
    ),
    sparse = TRUE
  )
}

sce_from <- function(m) {
  SingleCellExperiment::SingleCellExperiment(assays = list(counts = m))
}

test_that("from_sce rejects an out-of-range @i without aborting the session", {
  skip_if_no_sce()
  m <- valid_dgc()
  m@i[1] <- 99L # row index 99 >= n_rows (5)
  sce <- sce_from(m)
  expect_error(
    from_sce(sce, tempfile(fileext = ".scx")),
    "@i value 99 out of range"
  )
})

test_that("from_sce rejects a truncated @p without aborting the session", {
  skip_if_no_sce()
  m <- valid_dgc()
  m@p <- m@p[seq_len(length(m@p) - 1L)] # one pointer short of n_cols + 1
  sce <- sce_from(m)
  expect_error(
    from_sce(sce, tempfile(fileext = ".scx")),
    "@p length"
  )
})

test_that("from_sce rejects a non-monotonic @p without aborting the session", {
  skip_if_no_sce()
  m <- valid_dgc()
  # Force a decreasing step in @p (column 2 ends before it starts).
  m@p[3] <- 0L
  sce <- sce_from(m)
  expect_error(
    from_sce(sce, tempfile(fileext = ".scx")),
    "@p not monotonic|@p\\[.*\\] ="
  )
})

test_that("from_sce still succeeds on a well-formed dgCMatrix", {
  skip_if_no_sce()
  m <- valid_dgc()
  sce <- sce_from(m)
  path <- tempfile(fileext = ".scx")
  expect_no_error(from_sce(sce, path))
  expect_true(file.exists(path))
})
