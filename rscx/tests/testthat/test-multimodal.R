# Phase I tests — Seurat v5 multi-assay + MultiAssayExperiment interop.
#
# These tests require optional R packages (Seurat >= 5.0, Matrix,
# MultiAssayExperiment). They skip cleanly when the packages are not
# installed, so the rscx test suite stays green on minimal R
# environments. See rscx/tests/testthat/test-reader.R for the
# `skip_if_no_fixture()` pattern used elsewhere in this directory.


# --- Phase I.1: Seurat v5 multi-assay round-trip ------------------------------

test_that("from_seurat detects v5 multi-assay and writes a multimodal SCX file", {
  skip_if_not_installed("Seurat", minimum_version = "5.0.0")
  skip_if_not_installed("Matrix")

  set.seed(42)
  n_cells <- 20
  rna_n <- 12
  adt_n <- 4
  rna_counts <- Matrix::Matrix(
    matrix(rpois(rna_n * n_cells, lambda = 0.5), nrow = rna_n, ncol = n_cells),
    sparse = TRUE
  )
  adt_counts <- Matrix::Matrix(
    matrix(rpois(adt_n * n_cells, lambda = 0.5), nrow = adt_n, ncol = n_cells),
    sparse = TRUE
  )
  rownames(rna_counts) <- paste0("g", seq_len(rna_n))
  rownames(adt_counts) <- paste0("a", seq_len(adt_n))
  colnames(rna_counts) <- paste0("cell_", seq_len(n_cells))
  colnames(adt_counts) <- paste0("cell_", seq_len(n_cells))

  seu <- Seurat::CreateSeuratObject(counts = rna_counts, assay = "rna")
  seu[["adt"]] <- SeuratObject::CreateAssay5Object(counts = adt_counts)

  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)
  from_seurat(seu, out)

  exp <- scx_open(out)
  expect_true(exp$is_multimodal())
  modalities <- exp$modality_names()
  expect_setequal(modalities, c("rna", "adt"))
  expect_equal(exp$n_obs(), n_cells)
})


test_that("scx_open(...)$to_seurat() reconstructs a v5 multi-assay object", {
  skip_if_not_installed("Seurat", minimum_version = "5.0.0")
  skip_if_not_installed("Matrix")

  set.seed(0)
  n_cells <- 16
  rna_n <- 10
  adt_n <- 3
  rna_counts <- Matrix::Matrix(
    matrix(rpois(rna_n * n_cells, lambda = 0.5), nrow = rna_n, ncol = n_cells),
    sparse = TRUE
  )
  adt_counts <- Matrix::Matrix(
    matrix(rpois(adt_n * n_cells, lambda = 0.5), nrow = adt_n, ncol = n_cells),
    sparse = TRUE
  )
  rownames(rna_counts) <- paste0("g", seq_len(rna_n))
  rownames(adt_counts) <- paste0("a", seq_len(adt_n))
  colnames(rna_counts) <- paste0("cell_", seq_len(n_cells))
  colnames(adt_counts) <- paste0("cell_", seq_len(n_cells))
  seu <- Seurat::CreateSeuratObject(counts = rna_counts, assay = "rna")
  seu[["adt"]] <- SeuratObject::CreateAssay5Object(counts = adt_counts)

  scx_path <- tempfile(fileext = ".scx")
  on.exit(unlink(scx_path), add = TRUE)
  from_seurat(seu, scx_path)
  seu_back <- scx_open(scx_path)$to_seurat()
  expect_s4_class(seu_back, "Seurat")
  # Both assays survived the round-trip.
  expect_setequal(names(seu_back@assays), c("rna", "adt"))
  # Per-assay shapes match the original.
  expect_equal(nrow(seu_back[["rna"]]), rna_n)
  expect_equal(nrow(seu_back[["adt"]]), adt_n)
  expect_equal(ncol(seu_back), n_cells)
})


# --- Phase I.2: MultiAssayExperiment round-trip ------------------------------

test_that("from_mae writes a multimodal SCX file from a cell-aligned MAE", {
  skip_if_not_installed("MultiAssayExperiment")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("Matrix")

  set.seed(1)
  n_cells <- 12
  rna_n <- 8
  adt_n <- 4
  rna_mat <- Matrix::Matrix(
    matrix(rpois(rna_n * n_cells, 0.5), nrow = rna_n, ncol = n_cells),
    sparse = TRUE
  )
  adt_mat <- Matrix::Matrix(
    matrix(rpois(adt_n * n_cells, 0.5), nrow = adt_n, ncol = n_cells),
    sparse = TRUE
  )
  cell_ids <- paste0("cell_", seq_len(n_cells))
  colnames(rna_mat) <- cell_ids
  colnames(adt_mat) <- cell_ids
  rownames(rna_mat) <- paste0("g", seq_len(rna_n))
  rownames(adt_mat) <- paste0("a", seq_len(adt_n))

  rna_sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = rna_mat)
  )
  adt_sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = adt_mat)
  )
  col_data <- S4Vectors::DataFrame(
    cell_id = cell_ids,
    row.names = cell_ids
  )
  mae <- MultiAssayExperiment::MultiAssayExperiment(
    experiments = list(rna = rna_sce, adt = adt_sce),
    colData = col_data
  )

  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)
  from_mae(mae, out)

  exp <- scx_open(out)
  expect_true(exp$is_multimodal())
  expect_setequal(exp$modality_names(), c("rna", "adt"))
  expect_equal(exp$n_obs(), n_cells)
})


test_that("from_mae raises when experiments have different cell axes", {
  skip_if_not_installed("MultiAssayExperiment")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("Matrix")

  set.seed(2)
  rna_mat <- Matrix::Matrix(
    matrix(rpois(8 * 12, 0.5), nrow = 8, ncol = 12),
    sparse = TRUE
  )
  adt_mat <- Matrix::Matrix(
    matrix(rpois(4 * 10, 0.5), nrow = 4, ncol = 10), # different cell count
    sparse = TRUE
  )
  colnames(rna_mat) <- paste0("cell_", seq_len(12))
  colnames(adt_mat) <- paste0("cell_", seq_len(10))
  rownames(rna_mat) <- paste0("g", seq_len(8))
  rownames(adt_mat) <- paste0("a", seq_len(4))

  rna_sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = rna_mat)
  )
  adt_sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = adt_mat)
  )
  col_data <- S4Vectors::DataFrame(
    cell_id = paste0("cell_", seq_len(12)),
    row.names = paste0("cell_", seq_len(12))
  )
  mae <- MultiAssayExperiment::MultiAssayExperiment(
    experiments = list(rna = rna_sce, adt = adt_sce),
    colData = col_data
  )

  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)
  expect_error(
    from_mae(mae, out),
    regexp = "non-aligned cell axes"
  )
})


test_that("scx_open(...)$to_mae() reconstructs a MultiAssayExperiment", {
  skip_if_not_installed("MultiAssayExperiment")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("Matrix")

  set.seed(3)
  n_cells <- 10
  rna_n <- 6
  adt_n <- 3
  rna_mat <- Matrix::Matrix(
    matrix(rpois(rna_n * n_cells, 0.5), nrow = rna_n, ncol = n_cells),
    sparse = TRUE
  )
  adt_mat <- Matrix::Matrix(
    matrix(rpois(adt_n * n_cells, 0.5), nrow = adt_n, ncol = n_cells),
    sparse = TRUE
  )
  cell_ids <- paste0("cell_", seq_len(n_cells))
  colnames(rna_mat) <- cell_ids
  colnames(adt_mat) <- cell_ids
  rownames(rna_mat) <- paste0("g", seq_len(rna_n))
  rownames(adt_mat) <- paste0("a", seq_len(adt_n))

  rna_sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = rna_mat)
  )
  adt_sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = adt_mat)
  )
  col_data <- S4Vectors::DataFrame(
    cell_id = cell_ids,
    row.names = cell_ids
  )
  mae <- MultiAssayExperiment::MultiAssayExperiment(
    experiments = list(rna = rna_sce, adt = adt_sce),
    colData = col_data
  )

  scx_path <- tempfile(fileext = ".scx")
  on.exit(unlink(scx_path), add = TRUE)
  from_mae(mae, scx_path)
  mae_back <- scx_open(scx_path)$to_mae()
  expect_s4_class(mae_back, "MultiAssayExperiment")
  expect_setequal(
    names(MultiAssayExperiment::experiments(mae_back)),
    c("rna", "adt")
  )
})


# --- Multimodal whole-cell delete ---------------------------------------------
# A whole-cell scx_delete on a multimodal file removes the cell from every
# modality. Before compact, a modality-scoped query already reflects the
# deletion (the engine applies the global deletion bitmap to each modality);
# after compact the physical n_obs drops for both modalities. Builds the file
# via from_seurat (like the query test) so it skips cleanly without Seurat.

test_that("scx_delete on a multimodal file removes the cell from every modality", {
  skip_if_not_installed("Seurat", minimum_version = "5.0.0")
  skip_if_not_installed("Matrix")

  set.seed(11)
  n_cells <- 18
  rna_n <- 9
  adt_n <- 4
  rna_counts <- Matrix::Matrix(
    matrix(rpois(rna_n * n_cells, lambda = 0.6), nrow = rna_n, ncol = n_cells),
    sparse = TRUE
  )
  adt_counts <- Matrix::Matrix(
    matrix(rpois(adt_n * n_cells, lambda = 0.6), nrow = adt_n, ncol = n_cells),
    sparse = TRUE
  )
  rownames(rna_counts) <- paste0("g", seq_len(rna_n))
  rownames(adt_counts) <- paste0("a", seq_len(adt_n))
  colnames(rna_counts) <- paste0("cell_", seq_len(n_cells))
  colnames(adt_counts) <- paste0("cell_", seq_len(n_cells))

  seu <- Seurat::CreateSeuratObject(counts = rna_counts, assay = "rna")
  seu[["adt"]] <- SeuratObject::CreateAssay5Object(counts = adt_counts)

  path <- tempfile(fileext = ".scx")
  on.exit(unlink(path), add = TRUE)
  from_seurat(seu, path)

  # Delete three cells (1-based R indices).
  deleted <- c(2L, 7L, 15L)
  total <- scx_delete(path, deleted)
  expect_equal(total, length(deleted))

  # Before compact: a modality-scoped query drops the deleted rows from each
  # modality (deletion applies to the shared global obs axis). This is the R
  # equivalent of the CLI/engine "modality query with deletions succeeds" test.
  kept <- n_cells - length(deleted)
  rna <- scx_open(path) |> scx_query(modality = "rna") |> collect()
  expect_equal(rna$n_obs(), kept)
  expect_equal(rna$n_vars(), rna_n)
  adt <- scx_open(path) |> scx_query(modality = "adt") |> collect()
  expect_equal(adt$n_obs(), kept)
  expect_equal(adt$n_vars(), adt_n)

  # After compact: the deletions are physically reclaimed and n_obs drops.
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)
  scx_compact(path, out)
  expect_equal(scx_info(out)$n_obs, kept)
  expect_true(scx_validate(out))

  # Compacted file is still multimodal with both modalities aligned.
  exp_out <- scx_open(out)
  expect_true(exp_out$is_multimodal())
  expect_setequal(exp_out$modality_names(), c("rna", "adt"))
  rna_c <- scx_open(out) |> scx_query(modality = "rna") |> collect()
  adt_c <- scx_open(out) |> scx_query(modality = "adt") |> collect()
  expect_equal(rna_c$n_obs(), kept)
  expect_equal(adt_c$n_obs(), kept)
})
