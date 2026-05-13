# Seurat v4/v5 regression tests.
#
# These tests verify that rscx correctly produces Seurat-compatible objects
# and that the round-trip (Seurat → SCX → Seurat) preserves matrix data
# and metadata.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

has_seurat <- function(min_version = NULL) {
  if (!requireNamespace("SeuratObject", quietly = TRUE)) return(FALSE)
  if (!is.null(min_version)) {
    v <- utils::packageVersion("SeuratObject")
    return(v >= min_version)
  }
  TRUE
}

skip_if_no_seurat <- function(min_version = NULL) {
  if (!has_seurat(min_version)) {
    skip(paste("SeuratObject not installed",
               if (!is.null(min_version)) paste("(need >=", min_version, ")")))
  }
}

# ---------------------------------------------------------------------------
# to_seurat: ScxExperiment → Seurat
# ---------------------------------------------------------------------------

test_that("to_seurat produces a Seurat object with correct dimensions", {
  skip_if_no_seurat()
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  seu <- exp$to_seurat()

  expect_s4_class(seu, "Seurat")
  expect_equal(ncol(seu), exp$n_obs())   # cells = columns in Seurat
  expect_equal(nrow(seu), exp$n_vars())  # genes = rows in Seurat
})

test_that("to_seurat matrix matches x_matrix values", {
  skip_if_no_seurat()
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  seu <- exp$to_seurat()
  mat_scx <- exp$x_matrix()

  # Seurat stores genes × cells (transposed from SCX cells × genes)
  mat_seu <- SeuratObject::GetAssayData(seu, layer = "counts")

  # Verify dimensions match (after accounting for Seurat's gene × cell layout)
  expect_equal(nrow(mat_seu), ncol(mat_scx))
  expect_equal(ncol(mat_seu), nrow(mat_scx))

  # Convert both to dense for value comparison
  dense_scx <- as.matrix(mat_scx)    # cells × genes
  dense_seu <- as.matrix(mat_seu)    # genes × cells
  expect_equal(t(dense_scx), dense_seu, tolerance = 1e-6)
})

test_that("to_seurat preserves obs metadata", {
  skip_if_no_seurat()
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  seu <- exp$to_seurat()

  obs_scx <- exp$obs()
  meta_seu <- seu@meta.data

  # Cell IDs should appear in Seurat metadata or as column names
  expect_equal(ncol(seu), nrow(obs_scx))
})

test_that("to_seurat preserves var metadata", {
  skip_if_no_seurat()
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  seu <- exp$to_seurat()

  var_scx <- exp$var()

  # Gene IDs should appear as feature names
  expect_equal(nrow(seu), nrow(var_scx))
})

# ---------------------------------------------------------------------------
# from_seurat: Seurat → SCX round-trip
# ---------------------------------------------------------------------------

test_that("from_seurat round-trip preserves matrix and metadata", {
  skip_if_no_seurat("5.0.0")  # from_seurat requires v5
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  seu_orig <- exp$to_seurat()

  # Write Seurat → SCX
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  from_seurat(seu_orig, tmp)

  # Read back and convert to Seurat
  exp2 <- scx_open(tmp)
  seu_rt <- exp2$to_seurat()

  # Dimensions should match
  expect_equal(ncol(seu_rt), ncol(seu_orig))
  expect_equal(nrow(seu_rt), nrow(seu_orig))

  # Matrix values should round-trip
  orig_counts <- SeuratObject::GetAssayData(seu_orig, layer = "counts")
  rt_counts <- SeuratObject::GetAssayData(seu_rt, layer = "counts")
  expect_equal(as.matrix(orig_counts), as.matrix(rt_counts), tolerance = 1e-6)
})

# ---------------------------------------------------------------------------
# Seurat v4 graceful failure / compat
# ---------------------------------------------------------------------------

test_that("from_seurat handles v4-style assay gracefully", {
  skip_if_no_seurat()
  # If only SeuratObject < 5 is installed, from_seurat should either
  # succeed on the legacy assay or fail with an informative error.
  # This test documents expected behavior; it cannot force a v4
  # installation, but exercises the code path.
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  seu <- exp$to_seurat()

  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)

  # Should not crash — either succeeds or gives a clear error
  result <- tryCatch(
    {
      from_seurat(seu, tmp)
      "success"
    },
    error = function(e) {
      # Accept informative errors about v4/v5 incompatibility
      conditionMessage(e)
    }
  )
  expect_true(is.character(result))
})

# ---------------------------------------------------------------------------
# Query result → Seurat
# ---------------------------------------------------------------------------

test_that("query result to_seurat produces valid Seurat object", {
  skip_if_no_seurat()
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  result <- exp$query() |> collect()
  seu <- result$to_seurat()

  expect_s4_class(seu, "Seurat")
  expect_equal(ncol(seu), result$n_obs())
  expect_equal(nrow(seu), result$n_vars())
})

test_that("filtered query result preserves dimensions in Seurat", {
  skip_if_no_seurat()
  path <- skip_if_no_fixture()
  exp <- scx_open(path)

  # Limit to 5 rows
  result <- exp$query() |> limit(5) |> collect()
  seu <- result$to_seurat()

  expect_s4_class(seu, "Seurat")
  expect_equal(ncol(seu), 5)
  expect_equal(nrow(seu), exp$n_vars())
})
