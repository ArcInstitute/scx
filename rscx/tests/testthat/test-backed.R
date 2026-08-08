# Backed (lazy) sparse access. The eager `exp$x_matrix()` is the ground truth;
# every backed read must match the corresponding rows/cols of that dgCMatrix.

test_that("x_backed / scx_backed_sparse open an ScxBackedSparse with right dims", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)

  bsd <- exp$x_backed()
  expect_s3_class(bsd, "ScxBackedSparse")
  expect_equal(dim(bsd), c(20L, 10L))
  expect_equal(nrow(bsd), 20L)
  expect_equal(ncol(bsd), 10L)

  # Standalone constructor is equivalent.
  bsd2 <- scx_backed_sparse(path)
  expect_s3_class(bsd2, "ScxBackedSparse")
  expect_equal(dim(bsd2), dim(bsd))
})

test_that("contiguous row slice matches the eager matrix", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- exp$x_matrix()
  bsd <- exp$x_backed()

  got <- bsd[1:5, ]
  expect_s4_class(got, "dgCMatrix")
  expect_equal(dim(got), c(5L, 10L))
  expect_equal(as.matrix(got), as.matrix(eager[1:5, , drop = FALSE]))
})

test_that("arbitrary / reordered row indices preserve requested order", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- exp$x_matrix()
  bsd <- exp$x_backed()

  idx <- c(5L, 1L, 3L)
  got <- bsd[idx, ]
  expect_equal(dim(got), c(3L, 10L))
  expect_equal(as.matrix(got), as.matrix(eager[idx, , drop = FALSE]))
})

test_that("single row read returns a 1 x n_vars dgCMatrix", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- exp$x_matrix()
  bsd <- exp$x_backed()

  got <- bsd[2, ]
  expect_s4_class(got, "dgCMatrix")
  expect_equal(dim(got), c(1L, 10L))
  expect_equal(as.matrix(got), as.matrix(eager[2, , drop = FALSE]))
})

test_that("column subset is applied to the returned matrix", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- exp$x_matrix()
  bsd <- exp$x_backed()

  got <- bsd[1:5, 1:3]
  expect_equal(dim(got), c(5L, 3L))
  expect_equal(as.matrix(got), as.matrix(eager[1:5, 1:3, drop = FALSE]))
})

test_that("logical row index selects the right rows", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- exp$x_matrix()
  bsd <- exp$x_backed()

  mask <- rep(c(TRUE, FALSE), length.out = 20)
  got <- bsd[mask, ]
  expect_equal(as.matrix(got), as.matrix(eager[mask, , drop = FALSE]))
})

test_that("missing row index materialises the full matrix", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- exp$x_matrix()
  bsd <- exp$x_backed()

  expect_equal(as.matrix(bsd[]), as.matrix(eager))
  expect_equal(as.matrix(as(bsd, "dgCMatrix")), as.matrix(eager))
  expect_equal(as.matrix(bsd), as.matrix(eager))
})

test_that("row_sums / col_sums match the eager matrix", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- exp$x_matrix()
  bsd <- exp$x_backed()

  expect_equal(bsd$row_sums(), unname(Matrix::rowSums(eager)), tolerance = 1e-6)
  expect_equal(bsd$col_sums(), unname(Matrix::colSums(eager)), tolerance = 1e-6)
  expect_equal(bsd$nnz(), Matrix::nnzero(eager))
})

test_that("out-of-bounds row index raises a clean error", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  bsd <- exp$x_backed()

  expect_error(bsd[21, ])
  expect_error(bsd[c(1L, 100L), ])
  expect_error(bsd[-1, ])
  expect_error(bsd[0, ], "1-based")            # 0 is rejected (1-based)
  expect_error(bsd[c(1L, 0L), ], "1-based")
  expect_error(bsd[rep(TRUE, 3), ])  # wrong-length logical
  expect_error(bsd$read_rows(5, 2), "start") # start > end guard
})

test_that("raw read methods reject indices `f64 as u64` would saturate", {
  path <- skip_if_no_fixture()
  bsd <- scx_open(path)$x_backed()

  # `$read_rows()` / `$read_row_indices()` are exported and bypass `[`'s guards.
  # Rust's `f64 as u64` saturates (-1 -> 0, NaN -> 0) and truncates fractions, so
  # every one of these silently returned the WRONG CELLS before the fix rather
  # than erroring: read_rows(-1, 10) was identical to read_rows(0, 10).
  expect_error(bsd$read_rows(-1, 10), "negative")
  expect_error(bsd$read_rows(NaN, 10), "NA or NaN")
  expect_error(bsd$read_rows(1.5, 3), "non-integer")
  expect_error(bsd$read_rows(0, Inf), "non-finite")
  expect_error(bsd$read_row_indices(c(NA, 3, -2)), "NA or NaN")
  expect_error(bsd$read_row_indices(c(0, -1)), "negative")
  expect_error(bsd$read_row_indices(c(0, 1.5)), "non-integer")

  # The offending element is named by 1-based position, not just by value.
  expect_error(bsd$read_row_indices(c(0, 1, -2)), "position 3")

  # Valid boundary values must keep working: `end == n_obs` is a legal half-open
  # bound (not an index), and 0 is a legal 0-based row.
  expect_equal(nrow(bsd$read_rows(0, 20)), 20L)
  expect_equal(nrow(bsd$read_rows(0, 0)), 0L)
})

test_that("`[` keeps Matrix's fractional-subscript semantics", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- exp$x_matrix()
  bsd <- exp$x_backed()

  # A dgCMatrix truncates fractional subscripts (`M[1.9, ]` is row 1), and
  # `bsd[i, j]` hands `j` to Matrix's own `[`. If the row path went strict while
  # the column path truncated, one call would error on `i` and silently truncate
  # `j`. `[` therefore truncates; the raw `$read_*` methods stay strict.
  expect_equal(as.matrix(bsd[1.9, ]), as.matrix(eager[1, , drop = FALSE]))
  expect_equal(as.matrix(bsd[c(1.9, 2.9, 3.9), ]), as.matrix(eager[1:3, , drop = FALSE]))
  expect_equal(as.matrix(bsd[c(1.2, 2.9), ]), as.matrix(eager[1:2, , drop = FALSE]))
  expect_equal(as.matrix(bsd[2.5, 1.9]), as.matrix(eager[2, 1, drop = FALSE]))

  # Non-finite subscripts are rejected in R, with R's spelling of Inf.
  expect_error(bsd[Inf, ], "finite")
  expect_error(bsd[c(1, Inf), ], "finite")
})

test_that("backed + lazy read correctly across multiple shards", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  file.copy(path, tmp)
  # Append with a tiny shard_size so the file spans several CSR shards; this
  # exercises the n_shards > 1 streaming paths (global-vs-local row ids).
  scx_append(tmp, path, shard_size = 8L)
  expect_gt(scx_info(tmp)$n_shards, 1L)

  exp <- scx_open(tmp)
  eager <- as.matrix(exp$x_matrix())

  bsd <- exp$x_backed()
  expect_equal(as.matrix(bsd[5:30, ]), eager[5:30, , drop = FALSE]) # cross-shard
  expect_equal(bsd$col_sums(), unname(Matrix::colSums(eager)), tolerance = 1e-6)
  expect_equal(bsd$row_sums(), unname(Matrix::rowSums(eager)), tolerance = 1e-6)

  # Lazy normalize_total streams all shards for the per-cell sums.
  lt <- scx_lazy_transform(tmp) |> scx_normalize_total(1e4)
  nonempty <- rowSums(eager) > 0
  expect_equal(lt$row_sums()[nonempty], rep(1e4, sum(nonempty)), tolerance = 1e-3)
})
