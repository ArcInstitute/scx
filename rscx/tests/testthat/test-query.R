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
    select_genes(c(0L, 1L, 2L)) |>
    collect()

  expect_equal(result$n_vars(), 3)
  expect_equal(result$n_obs(), 20)
})
