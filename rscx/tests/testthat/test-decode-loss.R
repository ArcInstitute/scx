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

# `$layer()` is the surface the catalog's name-scoped `value_max` fold reaches,
# and it had no coverage: the fold resolved only the per-modality
# `layer/{modality}/{layer}/shard_{idx}` section naming, so on a
# single-modality file it answered 0 — indistinguishable from "no large values
# here" — and this guard never fired. rscx cannot write a layer (`from_seurat`
# reads only the Seurat `counts` layer), so the fixture is written by pyscx and
# committed: 6 cells x 4 genes, a `narrow` layer and a `wide` one holding
# 20,000,000.
test_that("layer() guards per layer: narrow reads, wide fails loud, allow_lossy escapes", {
  path <- skip_if_no_fixture("tiny_big_layer.scx")
  exp <- scx_open(path)
  expect_setequal(exp$layer_names(), c("narrow", "wide"))

  # The scoping that matters: a narrow layer reads even though a wide one
  # exists in the same file.
  narrow <- exp$layer("narrow")
  expect_s4_class(narrow, "dgCMatrix")
  expect_equal(max(narrow), 14)

  # And the wide one still refuses, rather than silently rounding.
  expect_error(exp$layer("wide"), "allow_lossy")
  wide <- exp$layer("wide", allow_lossy = TRUE)
  expect_s4_class(wide, "dgCMatrix")
  expect_equal(max(wide), 20000000)
})
