# Lazy transform chains. The eager `exp$x_matrix()` (cells x genes, after the
# same transform applied in R) is the ground truth; every lazy read must match.

# Reference: replicate normalize_total/log1p/row_scale in R on a cells x genes
# dense matrix (X has cells as rows, genes as columns — same as x_matrix()).
.ref_normalize <- function(x, target_sum = 1e4) {
  rs <- rowSums(x)
  f <- ifelse(rs > 0, target_sum / rs, 0)
  x * f # recycles f down columns -> per-row scaling
}

test_that("x_lazy / scx_lazy_transform open an empty-chain ScxLazyTransformed", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)

  lt <- exp$x_lazy()
  expect_s3_class(lt, "ScxLazyTransformed")
  expect_equal(dim(lt), c(20L, 10L))
  expect_length(lt$transform_names(), 0L)

  lt2 <- scx_lazy_transform(path)
  expect_s3_class(lt2, "ScxLazyTransformed")
  expect_equal(dim(lt2), dim(lt))

  # Empty chain is an identity view of X.
  expect_equal(as.matrix(lt), as.matrix(exp$x_matrix()))
})

test_that("scx_normalize_total matches an eager per-cell normalization", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- as.matrix(exp$x_matrix())

  lt <- scx_lazy_transform(path) |> scx_normalize_total(target_sum = 1e4)
  expect_equal(lt$transform_names(), "normalize_total")

  got <- as.matrix(lt[])
  ref <- .ref_normalize(eager, 1e4)
  expect_equal(got, ref, tolerance = 1e-4)

  # A row slice applies the same per-cell factor (global row identity preserved).
  expect_equal(as.matrix(lt[3:8, ]), ref[3:8, , drop = FALSE], tolerance = 1e-4)
})

test_that("scx_log1p matches eager log1p", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- as.matrix(exp$x_matrix())

  lt <- scx_lazy_transform(path) |> scx_log1p()
  expect_equal(as.matrix(lt[]), log1p(eager), tolerance = 1e-5)
})

test_that("normalize_total -> log1p chain matches the eager equivalent", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- as.matrix(exp$x_matrix())

  lt <- scx_lazy_transform(path) |>
    scx_normalize_total(target_sum = 1e4) |>
    scx_log1p()
  expect_equal(lt$transform_names(), c("normalize_total", "log1p"))

  ref <- log1p(.ref_normalize(eager, 1e4))
  expect_equal(as.matrix(lt[]), ref, tolerance = 1e-4)
  # Reordered fancy index keeps per-row identity correct.
  idx <- c(10L, 1L, 5L)
  expect_equal(as.matrix(lt[idx, ]), ref[idx, , drop = FALSE], tolerance = 1e-4)
})

test_that("scx_row_scale applies per-cell factors and validates length", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- as.matrix(exp$x_matrix())

  set.seed(1)
  factors <- runif(nrow(eager), 0.5, 2.0)
  lt <- scx_lazy_transform(path) |> scx_row_scale(factors)
  got <- as.matrix(lt[])
  ref <- eager * factors # recycles per-row
  expect_equal(got, ref, tolerance = 1e-4)

  expect_error(scx_lazy_transform(path) |> scx_row_scale(c(1, 2, 3)))
})

test_that("lazy chains are immutable / pipe-composable", {
  path <- skip_if_no_fixture()
  base <- scx_lazy_transform(path)
  norm <- scx_normalize_total(base, 1e4)
  # Appending to `norm` does not mutate `base`.
  expect_length(base$transform_names(), 0L)
  expect_equal(norm$transform_names(), "normalize_total")
})

test_that("row_sums after normalize_total equal target_sum per (non-empty) cell", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  eager <- as.matrix(exp$x_matrix())
  nonempty <- rowSums(eager) > 0

  lt <- scx_lazy_transform(path) |> scx_normalize_total(target_sum = 1e4)
  rs <- lt$row_sums()
  expect_equal(rs[nonempty], rep(1e4, sum(nonempty)), tolerance = 1e-3)

  # col_sums over transformed data matches the eager column sums.
  expect_equal(lt$col_sums(), unname(colSums(.ref_normalize(eager, 1e4))),
               tolerance = 1e-3)
})

test_that("nnz is preserved across transforms; out-of-bounds slices error", {
  path <- skip_if_no_fixture()
  exp <- scx_open(path)
  lt <- scx_lazy_transform(path) |> scx_normalize_total() |> scx_log1p()

  expect_equal(lt$nnz(), Matrix::nnzero(exp$x_matrix()))
  expect_error(lt[21, ])
  expect_error(lt[0, ], "1-based")          # 0 rejected (1-based)
  expect_error(lt$read_rows(5, 2), "start") # start > end guard
})

test_that("lazy raw read methods reject indices `f64 as u64` would saturate", {
  path <- skip_if_no_fixture()
  eager <- scx_open(path)$x_matrix()
  lt <- scx_lazy_transform(path)

  # Same defect as the backed handle (see test-backed.R): every one of these
  # returned the wrong rows silently, because the bounds checks that follow the
  # cast only reject values ABOVE n_obs, never a saturated 0.
  expect_error(lt$read_rows(-1, 10), "negative")
  expect_error(lt$read_rows(NaN, 10), "NA or NaN")
  expect_error(lt$read_rows(1.5, 3), "non-integer")
  expect_error(lt$read_rows(0, Inf), "non-finite")
  expect_error(lt$read_row_indices(c(NA, 3, -2)), "NA or NaN")
  expect_error(lt$read_row_indices(c(0, -1)), "negative")
  expect_error(lt$read_row_indices(c(0, 1.5)), "non-integer")

  # `[` still truncates like a dgCMatrix, and the identity chain is a no-op.
  expect_equal(as.matrix(lt[1.9, ]), as.matrix(eager[1, , drop = FALSE]))
  expect_error(lt[Inf, ], "finite")

  # Mirror of the backed pin: `[.ScxLazyTransformed` is a *separate copy* of the
  # dispatcher, so the `trunc(i)`-before-contiguity ordering has to be held here
  # independently or an edit to one file alone would silently lose it.
  # `diff(c(1.5, 2.5)) == 1`, so this takes the range branch and would reach the
  # strict Rust layer as read_rows(0.5, 2.5) without the truncation.
  expect_equal(as.matrix(lt[c(1.5, 2.5), ]), as.matrix(eager[1:2, , drop = FALSE]))
})
