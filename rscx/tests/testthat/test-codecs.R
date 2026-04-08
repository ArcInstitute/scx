# Tests for reading SCX files written with different codecs (LZ4Shuffle, Pcodec).
# Ensures the R FFI boundary correctly delegates to all codec decode paths.

# --- LZ4Shuffle codec ---

test_that("LZ4Shuffle fixture opens with correct dimensions", {
  path <- skip_if_no_fixture("tiny_lz4.scx")
  exp <- scx_open(path)

  expect_s3_class(exp, "ScxExperiment")
  expect_equal(exp$n_obs(), 20)
  expect_equal(exp$n_vars(), 10)
  expect_true(exp$nnz() > 0)
})

test_that("LZ4Shuffle fixture returns correct matrix", {
  path <- skip_if_no_fixture("tiny_lz4.scx")
  ref_path <- skip_if_no_fixture("tiny.scx")

  exp <- scx_open(path)
  ref <- scx_open(ref_path)

  mat <- exp$x_matrix()
  ref_mat <- ref$x_matrix()

  expect_s4_class(mat, "dgCMatrix")
  expect_equal(dim(mat), c(20, 10))
  expect_equal(Matrix::nnzero(mat), Matrix::nnzero(ref_mat))
  expect_equal(as.matrix(mat), as.matrix(ref_mat))
})

test_that("LZ4Shuffle fixture obs/var match reference", {
  path <- skip_if_no_fixture("tiny_lz4.scx")
  ref_path <- skip_if_no_fixture("tiny.scx")

  exp <- scx_open(path)
  ref <- scx_open(ref_path)

  expect_equal(exp$obs(), ref$obs())
  expect_equal(exp$var(), ref$var())
})

# --- Pcodec codec ---

test_that("Pcodec fixture opens with correct dimensions", {
  path <- skip_if_no_fixture("tiny_pcodec.scx")
  exp <- scx_open(path)

  expect_s3_class(exp, "ScxExperiment")
  expect_equal(exp$n_obs(), 20)
  expect_equal(exp$n_vars(), 10)
  expect_true(exp$nnz() > 0)
})

test_that("Pcodec fixture returns correct matrix", {
  path <- skip_if_no_fixture("tiny_pcodec.scx")
  ref_path <- skip_if_no_fixture("tiny.scx")

  exp <- scx_open(path)
  ref <- scx_open(ref_path)

  mat <- exp$x_matrix()
  ref_mat <- ref$x_matrix()

  expect_s4_class(mat, "dgCMatrix")
  expect_equal(dim(mat), c(20, 10))
  expect_equal(Matrix::nnzero(mat), Matrix::nnzero(ref_mat))
  expect_equal(as.matrix(mat), as.matrix(ref_mat))
})

test_that("Pcodec fixture obs/var match reference", {
  path <- skip_if_no_fixture("tiny_pcodec.scx")
  ref_path <- skip_if_no_fixture("tiny.scx")

  exp <- scx_open(path)
  ref <- scx_open(ref_path)

  expect_equal(exp$obs(), ref$obs())
  expect_equal(exp$var(), ref$var())
})

# --- Validation across codecs ---

test_that("scx_validate passes for all codec fixtures", {
  for (name in c("tiny.scx", "tiny_lz4.scx", "tiny_pcodec.scx")) {
    path <- skip_if_no_fixture(name)
    result <- scx_validate(path)
    expect_true(result, info = paste("validation failed for", name))
  }
})

# --- Info metadata ---

test_that("scx_info returns metadata for all codec fixtures", {
  for (name in c("tiny.scx", "tiny_lz4.scx", "tiny_pcodec.scx")) {
    path <- skip_if_no_fixture(name)
    info <- scx_info(path)
    expect_type(info, "list")
    expect_equal(info$n_obs, 20)
    expect_equal(info$n_vars, 10)
  }
})
