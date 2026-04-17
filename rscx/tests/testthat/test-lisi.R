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
