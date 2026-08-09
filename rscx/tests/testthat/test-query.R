test_that("basic query pipeline collects all cells", {
  path <- skip_if_no_fixture()

  result <- scx_open(path) |>
    scx_query() |>
    collect()

  expect_s3_class(result, "RQueryResult")
  expect_equal(result$n_obs(), 20)
  expect_equal(result$n_vars(), 10)
  expect_true(result$nnz() > 0)
  expect_true(result$total_shards() >= 1)
})

test_that("filter_obs reduces cell count", {
  path <- skip_if_no_fixture()

  result <- scx_open(path) |>
    scx_query() |>
    filter_obs("batch == 'A'") |>
    collect()

  expect_true(result$n_obs() > 0)
  expect_true(result$n_obs() < 20)
})

test_that("limit restricts output rows", {
  path <- skip_if_no_fixture()

  result <- scx_open(path) |>
    scx_query() |>
    limit(5L) |>
    collect()

  expect_equal(result$n_obs(), 5)
})

test_that("to_dgcmatrix returns valid sparse matrix", {
  path <- skip_if_no_fixture()

  result <- scx_open(path) |>
    scx_query() |>
    collect()

  dgc <- result$to_dgcmatrix()
  expect_s4_class(dgc, "dgCMatrix")
  expect_equal(nrow(dgc), 20)
  expect_equal(ncol(dgc), 10)
})

test_that("query result obs/var accessible after to_dgcmatrix", {
  path <- skip_if_no_fixture()

  result <- scx_open(path) |>
    scx_query() |>
    collect()

  # Consume the matrix
  dgc <- result$to_dgcmatrix()

  # Metadata should still be accessible
  obs <- result$obs()
  var <- result$var()
  expect_s3_class(obs, "data.frame")
  expect_s3_class(var, "data.frame")
  expect_equal(nrow(obs), 20)
  expect_equal(nrow(var), 10)
})

test_that("select_genes projects columns", {
  path <- skip_if_no_fixture()

  result <- scx_open(path) |>
    scx_query() |>
    select_genes(c(1L, 2L, 3L)) |>   # 1-based (R4)
    collect()

  expect_equal(result$n_vars(), 3)
  expect_equal(result$n_obs(), 20)
})

test_that("select_genes rejects 0 / negative / non-integer indices (1-based guard, R4)", {
  path <- skip_if_no_fixture()
  pipe <- scx_open(path) |> scx_query()
  expect_error(select_genes(pipe, 0L), regexp = "1-based")
  expect_error(select_genes(pipe, -1L), regexp = "1-based")
  expect_error(select_genes(pipe, 1.5), regexp = "non-integer")
})

# --- A failed step must not poison the pipeline -----------
# The engine's consuming builders drop the pipeline on Err, so a binding that
# take()s before applying loses it for good: the next call then reports
# "already consumed" for an operation that never ran.

test_that("a failed builder step leaves the pipeline usable", {
  path <- skip_if_no_fixture()
  pipe <- scx_open(path) |> scx_query()

  # A predicate typo is rejected by the engine...
  expect_error(filter_obs(pipe, "batch ==== 'A'"))
  # ...and the very same object still works, with nothing half-applied.
  result <- pipe |> filter_obs("batch == 'A'") |> collect()
  expect_true(result$n_obs() > 0)
  expect_true(result$n_obs() < 20)

  # Same for a rejected gene index.
  pipe2 <- scx_open(path) |> scx_query()
  expect_error(select_genes(pipe2, 0L), regexp = "1-based")
  expect_equal(select_genes(pipe2, c(1L, 2L))$collect()$n_vars(), 2)
})

test_that("a failed collect() leaves the pipeline usable", {
  path <- skip_if_no_fixture()
  pipe <- scx_open(path) |> scx_query()

  # Out-of-range gene indices are only detected during execution, so this
  # fails inside collect() rather than in the builder. Note the failure is
  # currently a *panic* from arrow's `take` (an unchecked index — a separate
  # pre-existing defect, mirrored by pyscx's BaseException harness), not a
  # clean Err. What is asserted here is only that the failure, however it
  # arrives, does not take the pipeline with it.
  pipe2 <- select_genes(pipe, 10000L)
  expect_error(collect(pipe2))

  # The pipeline survived the failed execution: the bad selection can be
  # replaced and the query re-run on the same object.
  result <- select_genes(pipe2, c(1L, 2L)) |> collect()
  expect_equal(result$n_vars(), 2)
})

test_that("a successful collect() consumes the pipeline", {
  path <- skip_if_no_fixture()
  pipe <- scx_open(path) |> scx_query()

  expect_equal(collect(pipe)$n_obs(), 20)
  expect_error(count(pipe), regexp = "already consumed")
  expect_error(collect(pipe), regexp = "already consumed")
})

# --- Multimodal-scoped query -----------
# Builds a multimodal SCX on the fly via from_seurat (like test-multimodal.R),
# then scopes the query to one modality. Skips cleanly without Seurat/Matrix.

test_that("scx_query(modality=) scopes to one modality of a multimodal file", {
  skip_if_not_installed("Seurat", minimum_version = "5.0.0")
  skip_if_not_installed("Matrix")

  set.seed(7)
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

  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)
  from_seurat(seu, out)

  # Each modality collects at its own var width, over the shared obs axis.
  rna <- scx_open(out) |> scx_query(modality = "rna") |> collect()
  expect_equal(rna$n_obs(), n_cells)
  expect_equal(rna$n_vars(), rna_n)

  adt <- scx_open(out) |> scx_query(modality = "adt") |> collect()
  expect_equal(adt$n_obs(), n_cells)
  expect_equal(adt$n_vars(), adt_n)
})

test_that("scx_query on a multimodal file errors without / with a bad modality", {
  skip_if_not_installed("Seurat", minimum_version = "5.0.0")
  skip_if_not_installed("Matrix")

  set.seed(8)
  n_cells <- 12
  rna_counts <- Matrix::Matrix(
    matrix(rpois(6 * n_cells, lambda = 0.5), nrow = 6, ncol = n_cells), sparse = TRUE
  )
  adt_counts <- Matrix::Matrix(
    matrix(rpois(3 * n_cells, lambda = 0.5), nrow = 3, ncol = n_cells), sparse = TRUE
  )
  rownames(rna_counts) <- paste0("g", seq_len(6))
  rownames(adt_counts) <- paste0("a", seq_len(3))
  colnames(rna_counts) <- paste0("cell_", seq_len(n_cells))
  colnames(adt_counts) <- paste0("cell_", seq_len(n_cells))
  seu <- Seurat::CreateSeuratObject(counts = rna_counts, assay = "rna")
  seu[["adt"]] <- SeuratObject::CreateAssay5Object(counts = adt_counts)
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)
  from_seurat(seu, out)

  expect_error(scx_open(out) |> scx_query(), regexp = "multimodal")
  expect_error(scx_open(out) |> scx_query(modality = "atac"), regexp = "atac")
})
