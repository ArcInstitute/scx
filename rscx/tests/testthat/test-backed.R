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
