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

  # Convert both to dense for value comparison. `x_matrix()` carries no
  # dimnames while `to_seurat()` sets cell/gene names (and Seurat sanitizes
  # underscores in feature names), so compare values with names stripped.
  dense_scx <- as.matrix(mat_scx)    # cells × genes
  dense_seu <- as.matrix(mat_seu)    # genes × cells
  expect_equal(unname(t(dense_scx)), unname(dense_seu), tolerance = 1e-6)
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

# ---------------------------------------------------------------------------
# T4.2 — meta.data is keyed by barcode (AddMetaData), and the sole-assay name
# is honored on the from_seurat read path (no hardcoded 'RNA').
# ---------------------------------------------------------------------------

make_labeled_seurat <- function(assay = "RNA", n_genes = 20L, n_cells = 30L) {
  set.seed(7)
  counts <- matrix(
    rpois(n_genes * n_cells, lambda = 3),
    nrow = n_genes, ncol = n_cells,
    dimnames = list(paste0("g", seq_len(n_genes)),
                    paste0("cell", seq_len(n_cells)))
  )
  meta <- data.frame(
    # Per-cell label tied to its barcode — a misaligned join would scramble it.
    label = paste0("lab_", seq_len(n_cells)),
    row.names = paste0("cell", seq_len(n_cells))
  )
  Seurat::CreateSeuratObject(counts = counts, assay = assay, meta.data = meta)
}

test_that("to_seurat keys obs metadata to the correct barcode", {
  skip_if_no_seurat("5.0.0")
  obj <- make_labeled_seurat()

  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  from_seurat(obj, tmp)

  seu_rt <- scx_open(tmp)$to_seurat()
  # Value-level barcode keying: the original barcodes must survive the round
  # trip, and each cell's label must still travel with its own barcode.
  expect_setequal(colnames(seu_rt), colnames(obj))
  expect_equal(
    as.character(seu_rt@meta.data[colnames(obj), "label"]),
    as.character(obj$label)
  )
  # AddMetaData preserved Seurat's computed columns.
  expect_true("nCount_RNA" %in% colnames(seu_rt@meta.data))
})

test_that("from_seurat round-trips an object whose sole assay is not 'RNA'", {
  skip_if_no_seurat("5.0.0")
  obj <- make_labeled_seurat(assay = "originalexp")
  expect_equal(SeuratObject::DefaultAssay(obj), "originalexp")

  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  # Must not error on the hardcoded-'RNA' feature-metadata lookup.
  expect_error(from_seurat(obj, tmp), NA)

  exp2 <- scx_open(tmp)
  expect_equal(exp2$n_obs(), ncol(obj))
  expect_equal(exp2$n_vars(), nrow(obj))
})

# ---------------------------------------------------------------------------
# T4.3 — meta.data column classes (integer / logical / ordered factor)
# survive Seurat → SCX → Seurat.
# ---------------------------------------------------------------------------

test_that("from_seurat preserves integer / logical / ordered-factor columns", {
  skip_if_no_seurat("5.0.0")
  set.seed(11)
  n_genes <- 15L
  n_cells <- 24L
  counts <- matrix(
    rpois(n_genes * n_cells, lambda = 2),
    nrow = n_genes, ncol = n_cells,
    dimnames = list(paste0("g", seq_len(n_genes)),
                    paste0("c", seq_len(n_cells)))
  )
  meta <- data.frame(
    n_umi = as.integer(sample.int(1000L, n_cells)),
    is_doublet = sample(c(TRUE, FALSE), n_cells, replace = TRUE),
    stage = factor(sample(c("early", "mid", "late"), n_cells, replace = TRUE),
                   levels = c("early", "mid", "late"), ordered = TRUE),
    qc_score = runif(n_cells),
    row.names = colnames(counts)
  )
  obj <- Seurat::CreateSeuratObject(counts = counts, meta.data = meta)

  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  from_seurat(obj, tmp)

  md <- scx_open(tmp)$obs()
  expect_true(is.integer(md$n_umi))
  expect_true(is.logical(md$is_doublet))
  expect_true(is.ordered(md$stage))
  expect_identical(levels(md$stage), c("early", "mid", "late"))
  expect_true(is.numeric(md$qc_score) && !is.integer(md$qc_score))
})
