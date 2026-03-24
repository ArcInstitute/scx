test_that("scx_info returns correct metadata", {
  path <- skip_if_no_fixture()
  info <- scx_info(path)

  expect_type(info, "list")
  expect_equal(info$n_obs, 20)
  expect_equal(info$n_vars, 10)
  expect_true(info$nnz > 0)
  expect_equal(info$format_version, 1L)
  expect_true(info$n_shards >= 1L)
})

test_that("scx_validate returns TRUE for valid file", {
  path <- skip_if_no_fixture()
  expect_true(scx_validate(path))
})

test_that("scx_validate errors on invalid path", {
  expect_error(scx_validate("/nonexistent/path.scx"))
})

test_that("scx_delete marks cells as deleted", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  file.copy(path, tmp)

  total <- scx_delete(tmp, c(0L, 1L))
  expect_equal(total, 2)
})

test_that("scx_compact produces valid file with fewer cells", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(c(tmp, out)), add = TRUE)
  file.copy(path, tmp)

  scx_delete(tmp, c(0L, 1L, 2L))
  scx_compact(tmp, out)

  info <- scx_info(out)
  expect_equal(info$n_obs, 17)
  expect_true(scx_validate(out))
})

test_that("scx_rollback reverts to previous state", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  file.copy(path, tmp)

  # Delete some cells (creates a new manifest version)
  scx_delete(tmp, c(0L, 1L))

  # Rollback
  scx_rollback(tmp)

  # After rollback, info should show original state
  info <- scx_info(tmp)
  expect_equal(info$n_obs, 20)
})

test_that("scx_merge combines two files", {
  path <- skip_if_no_fixture()
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)

  scx_merge(c(path, path), out)

  info <- scx_info(out)
  expect_equal(info$n_obs, 40)
  expect_equal(info$n_vars, 10)
  expect_true(scx_validate(out))
})

test_that("scx_append adds cells to target", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  file.copy(path, tmp)

  scx_append(tmp, path)

  info <- scx_info(tmp)
  expect_equal(info$n_obs, 40)
  expect_equal(info$n_vars, 10)
})
