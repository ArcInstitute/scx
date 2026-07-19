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

# B3: fallible ops free functions must raise a clean R stop() carrying the real
# message, not the opaque "User function panicked" from extendr's unwrap()-panic.
test_that("scx ops raise clean errors (not Rust panics) on a bad file", {
  for (fn in list(
    function() scx_validate("/nonexistent/path.scx"),
    function() scx_info("/nonexistent/path.scx")
  )) {
    msg <- tryCatch(fn(), error = function(e) conditionMessage(e))
    expect_false(grepl("panicked", msg, fixed = TRUE))
  }
})

test_that("scx_delete marks cells as deleted", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  file.copy(path, tmp)

  total <- scx_delete(tmp, c(1L, 2L))
  expect_equal(total, 2)
})

test_that("scx_delete addresses indices above 2^31 and validates values (R3)", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  file.copy(path, tmp)

  # A >2^31 index reaches Rust as a double (the old as.integer() path would
  # truncate it to NA). On this small fixture it is out of bounds — the error
  # carries the real index, proving no NA truncation / i32 overflow.
  big <- 2^31 + 5
  expect_gt(big, .Machine$integer.max)
  expect_error(scx_delete(tmp, big), regexp = "out of bounds")

  # Value validation: 0 / negatives trip the R-side 1-based guard (R4);
  # non-integers are caught by the Rust core after the 1->0 conversion.
  expect_error(scx_delete(tmp, 0), regexp = "1-based")
  expect_error(scx_delete(tmp, -1), regexp = "1-based")
  expect_error(scx_delete(tmp, 1.5), regexp = "non-integer")
})

test_that("scx_compact produces valid file with fewer cells", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(c(tmp, out)), add = TRUE)
  file.copy(path, tmp)

  scx_delete(tmp, c(1L, 2L, 3L))
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
  scx_delete(tmp, c(1L, 2L))

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

# ── merge options ───────────────────────────────────────────────────────

test_that("scx_merge honours assume_identical_var + uns_policy", {
  path <- skip_if_no_fixture()
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)

  scx_merge(c(path, path), out,
            assume_identical_var = TRUE, uns_policy = "first")
  info <- scx_info(out)
  expect_equal(info$n_obs, 40)
  expect_true(scx_validate(out))
})

test_that("scx_merge sort_by is plumbed through (k-way merge needs sorted inputs)", {
  path <- skip_if_no_fixture()
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)

  # sort_by is a sorted k-way merge: it requires each input pre-sorted by the
  # key. The fixture is not sorted by "batch", so this must raise (confirming
  # the option reaches the Rust merge) rather than silently concatenate.
  expect_error(scx_merge(c(path, path), out, sort_by = "batch"), "sort")
})

test_that("scx_merge rejects an unknown uns_policy", {
  path <- skip_if_no_fixture()
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)
  expect_error(scx_merge(c(path, path), out, uns_policy = "nope"),
               "uns_policy")
})

test_that("scx_merge with index_obs builds an openable, valid file", {
  path <- skip_if_no_fixture()
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(out), add = TRUE)
  scx_merge(c(path, path), out, index_obs = "batch")
  expect_true(scx_validate(out))
  expect_equal(scx_info(out)$n_obs, 40)
})

# ── compact options ─────────────────────────────────────────────────────

test_that("scx_compact honours reshape_obs + index_obs", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  out <- tempfile(fileext = ".scx")
  on.exit(unlink(c(tmp, out)), add = TRUE)
  file.copy(path, tmp)

  scx_delete(tmp, c(1L, 2L, 3L))
  scx_compact(tmp, out, index_obs = "batch", reshape_obs = TRUE)

  info <- scx_info(out)
  expect_equal(info$n_obs, 17)
  expect_true(scx_validate(out))
})

# ── append options ──────────────────────────────────────────────────────

test_that("scx_append honours codec + shard_size", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  file.copy(path, tmp)

  scx_append(tmp, path, codec = "zstd", shard_size = 8L)
  info <- scx_info(tmp)
  expect_equal(info$n_obs, 40)
  expect_equal(info$n_vars, 10)
  expect_true(scx_validate(tmp))
})

test_that("scx_append with index_obs stays valid", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  file.copy(path, tmp)

  scx_append(tmp, path, index_obs = "batch")
  expect_equal(scx_info(tmp)$n_obs, 40)
  expect_true(scx_validate(tmp))
})

test_that("scx_append rejects a bad codec and a modality on a single-modality file", {
  path <- skip_if_no_fixture()
  tmp <- tempfile(fileext = ".scx")
  on.exit(unlink(tmp), add = TRUE)
  file.copy(path, tmp)

  expect_error(scx_append(tmp, path, codec = "bogus"))
  expect_error(scx_append(tmp, path, modality = "rna"), "modality")
})
