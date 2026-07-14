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

# --- Codec intent axis on the write path (E1) ---
#
# rscx `from_sce` / `from_seurat` / `from_mae` route through the shared
# `scx_format::resolve_codec`, so R can request the same `auto` / `fast` /
# `compact` profiles (and explicit codecs) as pyscx/CLI. These write a small SCE
# with each profile, read it back, and assert values round-trip losslessly.

skip_if_no_sce_codecs <- function() {
  skip_if_not_installed("Matrix")
  skip_if_not_installed("SingleCellExperiment")
}

# 6-gene x 8-cell integer count dgCMatrix (genes x cells, as SCE stores).
codec_count_sce <- function() {
  set.seed(1)
  dense <- matrix(rpois(6 * 8, lambda = 3), nrow = 6, ncol = 8)
  m <- Matrix::Matrix(dense, sparse = TRUE)
  SingleCellExperiment::SingleCellExperiment(assays = list(counts = m))
}

test_that("from_sce accepts the codec intent axis and round-trips losslessly", {
  skip_if_no_sce_codecs()
  sce <- codec_count_sce()

  # Reference: explicit zstd (a plain lossless codec).
  ref_path <- tempfile(fileext = ".scx")
  from_sce(sce, ref_path, codec = "zstd")
  ref_mat <- as.matrix(scx_open(ref_path)$x_matrix())

  for (codec in c("auto", "fast", "compact")) {
    path <- tempfile(fileext = ".scx")
    expect_no_error(from_sce(sce, path, codec = codec))
    expect_true(scx_validate(path), info = paste("validate failed for", codec))
    mat <- as.matrix(scx_open(path)$x_matrix())
    expect_equal(mat, ref_mat, info = paste("values diverged for codec", codec))
  }
})

test_that("codec='compact' with row_group_rows=0 errors cleanly (no session abort)", {
  skip_if_no_sce_codecs()
  sce <- codec_count_sce()
  expect_error(
    from_sce(sce, tempfile(fileext = ".scx"), codec = "compact", row_group_rows = 0L),
    "requires row_group_rows"
  )
})

test_that("unknown codec name errors via resolve_codec", {
  skip_if_no_sce_codecs()
  sce <- codec_count_sce()
  expect_error(
    from_sce(sce, tempfile(fileext = ".scx"), codec = "bogus"),
    "codec|bogus"
  )
})
