# u32 -> f32 decode-loss guard (F4).
#
# scx decodes on-disk integer counts to an f32 CSR before materialization, so
# counts above 2^24 (16,777,216) are silently rounded. Eager R reads now fail
# loud unless allow_lossy = TRUE.

# Write an SCX whose counts contain one entry above 2^24 (f32-exact) via a
# Seurat -> SCX round-trip, and return its path.
write_big_count_scx <- function() {
  m <- matrix(0, nrow = 5, ncol = 6)
  m[1, 1] <- 20000000 # > 2^24, exact in f32 -> uint32 on disk, value_max > 2^24
  m[2, 2] <- 7
  rownames(m) <- paste0("gene", seq_len(nrow(m)))
  colnames(m) <- paste0("cell", seq_len(ncol(m)))
  counts <- as(m, "CsparseMatrix")
  seu <- SeuratObject::CreateSeuratObject(counts = counts)
  path <- tempfile(fileext = ".scx")
  from_seurat(seu, path)
  path
}

test_that("to_seurat fails loud on >2^24 counts; allow_lossy escapes", {
  skip_if_not_installed("Seurat", "5.0.0")
  path <- write_big_count_scx()
  on.exit(unlink(path), add = TRUE)
  exp <- scx_open(path)
  expect_error(exp$to_seurat(), "allow_lossy")
  seu <- exp$to_seurat(allow_lossy = TRUE)
  expect_s4_class(seu, "Seurat")
})

test_that("x_matrix fails loud on >2^24 counts; allow_lossy escapes", {
  skip_if_not_installed("Seurat", "5.0.0")
  path <- write_big_count_scx()
  on.exit(unlink(path), add = TRUE)
  exp <- scx_open(path)
  expect_error(exp$x_matrix(), "allow_lossy")
  m <- exp$x_matrix(allow_lossy = TRUE)
  expect_s4_class(m, "dgCMatrix")
})

test_that("query path fails loud on >2^24 counts; allow_lossy escapes", {
  skip_if_not_installed("Seurat", "5.0.0")
  path <- write_big_count_scx()
  on.exit(unlink(path), add = TRUE)
  exp <- scx_open(path)
  res <- exp$query()$collect()
  expect_error(res$to_dgcmatrix(), "allow_lossy")

  # The guard fires before consuming the result, so the SAME object can be
  # retried with allow_lossy = TRUE (non-destructive guard error).
  m <- res$to_dgcmatrix(allow_lossy = TRUE)
  expect_s4_class(m, "dgCMatrix")
})

test_that("small-count archive reads without spurious error", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  expect_no_error(exp$x_matrix())
})
