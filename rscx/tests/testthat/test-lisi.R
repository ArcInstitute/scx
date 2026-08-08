test_that("scx_compute_lisi returns a length-N numeric vector", {
  set.seed(0)
  n <- 150
  d <- 6
  emb <- matrix(rnorm(n * d), nrow = n, ncol = d)
  labels <- sample(c("A", "B", "C"), n, replace = TRUE)

  lisi <- scx_compute_lisi(emb, labels, perplexity = 15)

  expect_type(lisi, "double")
  expect_length(lisi, n)
  expect_true(all(is.finite(lisi)))
  expect_true(all(lisi >= 1 - 1e-6))
  expect_true(all(lisi <= 3 + 1e-3))
})

test_that("single-label input yields LISI = 1", {
  set.seed(1)
  emb <- matrix(rnorm(80 * 4), nrow = 80, ncol = 4)
  labels <- rep("only", 80)

  lisi <- scx_compute_lisi(emb, labels, perplexity = 10)
  expect_true(all(abs(lisi - 1) < 1e-6))
})

test_that("scx_compute_lisi errors on invalid perplexity", {
  emb <- matrix(rnorm(30 * 3), nrow = 30, ncol = 3)
  labels <- rep(c("A", "B"), each = 15)
  expect_error(
    suppressWarnings(scx_compute_lisi(emb, labels, perplexity = -1))
  )
})

test_that("scx_compute_lisi rejects a NaN perplexity instead of returning all-1.0", {
  set.seed(2)
  emb <- matrix(rnorm(30 * 3), nrow = 30, ncol = 3)
  labels <- rep(c("A", "B"), each = 15)

  # `NaN <= 0.0` is false, so NaN slipped past the `perplexity > 0` guard.
  # Downstream `(NaN * 3).ceil() as usize` is 0 -> clamped to k = 1, and
  # `target_logu = NaN.ln()` makes the calibration loop's `abs() > tol` false,
  # so it never iterates. The result was a full-length vector of exactly 1.0 --
  # which reads as "perfectly unmixed", a plausible answer, with no warning.
  expect_error(
    suppressWarnings(scx_compute_lisi(emb, labels, perplexity = NaN)),
    "finite"
  )

  # A fractional perplexity is legitimate and must keep working: the derived
  # neighbour count is `ceil(3 * perplexity)`, so 15.5 -> 47.
  expect_length(scx_compute_lisi(emb, labels, perplexity = 5.5), 30L)
})
