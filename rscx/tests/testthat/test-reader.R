test_that("scx_open returns ScxExperiment with correct dimensions", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)

  expect_s3_class(exp, "ScxExperiment")
  expect_equal(exp$n_obs(), 20)
  expect_equal(exp$n_vars(), 10)
  expect_true(exp$nnz() > 0)
  expect_true(exp$shard_count() >= 1)
})

test_that("obs returns a data.frame with expected columns", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  obs <- exp$obs()

  expect_s3_class(obs, "data.frame")
  expect_equal(nrow(obs), 20)
  expect_true("cell_id" %in% names(obs))
  expect_true("batch" %in% names(obs))
})

test_that("var returns a data.frame with expected columns", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  var <- exp$var()

  expect_s3_class(var, "data.frame")
  expect_equal(nrow(var), 10)
  expect_true("gene_id" %in% names(var))
})

test_that("x_matrix returns a dgCMatrix with correct dimensions", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  mat <- exp$x_matrix()

  expect_s4_class(mat, "dgCMatrix")
  expect_equal(nrow(mat), 20)
  expect_equal(ncol(mat), 10)
  expect_true(Matrix::nnzero(mat) > 0)
})

test_that("layer_names returns character vector", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  names <- exp$layer_names()

  expect_type(names, "character")
})

test_that("scx_open errors on non-existent file", {
  expect_error(scx_open("/nonexistent/path.scx"))
})

# B3: a failed open must surface a clean R stop() carrying the real message,
# not the opaque "User function panicked: new" that extendr's unwrap()-panic
# produced before the constructor was routed through util::throw_on_err.
# This exercises the same new -> open_impl -> Err -> throw_on_err path as a
# format-version skew (both are an Err out of ScxReader::open).
test_that("scx_open raises a clean error (not a Rust panic) on a bad file", {
  msg <- tryCatch(scx_open("/nonexistent/path.scx"),
                  error = function(e) conditionMessage(e))
  expect_false(grepl("panicked", msg, fixed = TRUE))
  expect_match(msg, "failed to open SCX file")
})
