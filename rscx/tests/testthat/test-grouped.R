# F2 grouped-read bindings (7.2c): read_group / read_reference / group_labels /
# iter_group_shards on the pre-built `grouped.scx` fixture (12 cells, 4 target
# genes, "nt" reference; produced by pyscx.sort(group_by="target_gene")).

test_that("group_labels lists every label", {
  path <- skip_if_no_fixture("grouped.scx")
  exp <- scx_open(path)
  labels <- exp$group_labels()
  expect_setequal(labels, c("nt", "MYC", "TP53", "GATA1"))
})

test_that("read_group returns the labelled cells as an RQueryResult", {
  path <- skip_if_no_fixture("grouped.scx")
  exp <- scx_open(path)

  res <- exp$read_group("MYC")
  expect_s4_class(res$to_dgcmatrix(), "dgCMatrix")
  # Re-read for metadata (to_dgcmatrix consumes the matrix).
  res2 <- exp$read_group("MYC")
  expect_equal(res2$n_obs(), 3)
  expect_true(all(res2$obs()$target_gene == "MYC"))
})

test_that("read_reference returns the reference cells", {
  path <- skip_if_no_fixture("grouped.scx")
  exp <- scx_open(path)
  ref <- exp$read_reference()
  expect_false(is.null(ref))
  expect_equal(ref$n_obs(), 4)
})

test_that("unknown label raises a clean R error (not a panic)", {
  path <- skip_if_no_fixture("grouped.scx")
  exp <- scx_open(path)
  expect_error(exp$read_group("NOPE"))
})

test_that("read_group on an ungrouped file errors cleanly", {
  path <- skip_if_no_fixture("tiny.scx")
  exp <- scx_open(path)
  expect_error(exp$group_labels())
  expect_error(exp$read_group("anything"))
})

test_that("iter_group_shards covers all rows and exposes per-label slices", {
  path <- skip_if_no_fixture("grouped.scx")
  exp <- scx_open(path)
  shards <- exp$iter_group_shards()
  expect_true(is.list(shards))
  expect_true(length(shards) >= 1)

  total <- 0
  seen <- character(0)
  for (gs in shards) {
    qr <- gs$to_query_result()
    total <- total + qr$n_obs()
    g <- gs$groups()
    expect_s3_class(g, "data.frame")
    expect_true(all(c("label", "start", "stop") %in% names(g)))
    seen <- c(seen, gs$labels())
    # Per-label read out of the shard matches the whole-file read_group.
    lab <- gs$labels()[1]
    expect_equal(gs$read_group(lab)$n_obs(), exp$read_group(lab)$n_obs())
  }
  # Non-reference shards exclude "nt"; rows + reference == n_obs.
  expect_false("nt" %in% seen)
  ref <- exp$read_reference()
  ref_rows <- if (is.null(ref)) 0 else ref$n_obs()
  expect_equal(total + ref_rows, exp$n_obs())
})
