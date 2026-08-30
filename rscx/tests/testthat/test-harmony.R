test_that("scx_harmony_integrate returns list with correct dimensions", {
  set.seed(0)
  n <- 120
  d <- 8
  emb <- matrix(rnorm(n * d), nrow = n, ncol = d)
  batch <- sample(c("A", "B", "C"), n, replace = TRUE)

  res <- scx_harmony_integrate(emb, batch, max_iter = 3L, random_state = 7L)

  expect_type(res, "list")
  expect_named(
    res,
    c("embeddings", "converged", "n_iterations", "n_clusters", "objective",
      "scx_accel")
  )
  # Route metadata (ORG-10.16-3b review round 1): pyscx parity — the same
  # cpu_dense / user_forced_cpu pair pyscx stamps for device="cpu".
  expect_equal(res$scx_accel$route, "cpu_dense")
  expect_equal(res$scx_accel$fallback_reason, "user_forced_cpu")
  expect_equal(nrow(res$embeddings), n)
  expect_equal(ncol(res$embeddings), d)
  expect_true(is.logical(res$converged))
  expect_true(is.integer(res$n_iterations))
  expect_gte(res$n_iterations, 1L)
  expect_lte(res$n_iterations, 3L)
  expect_true(all(is.finite(res$embeddings)))
})

test_that("scx_harmony_integrate is deterministic for identical seed", {
  set.seed(42)
  emb <- matrix(rnorm(200 * 6), nrow = 200, ncol = 6)
  batch <- rep(c("A", "B"), each = 100)

  r1 <- scx_harmony_integrate(emb, batch, max_iter = 2L, random_state = 99L)
  r2 <- scx_harmony_integrate(emb, batch, max_iter = 2L, random_state = 99L)

  expect_identical(r1$embeddings, r2$embeddings)
})

test_that("scx_harmony_integrate rejects single-level batch", {
  emb <- matrix(rnorm(50 * 4), nrow = 50, ncol = 4)
  batch <- rep("A", 50)

  # extendr wraps Rust `Error::Other(...)` as a generic
  # "User function panicked" error; the detailed "requires >= 2" text is
  # printed to stderr rather than becoming the R condition message.
  expect_error(
    suppressWarnings(scx_harmony_integrate(emb, batch, max_iter = 1L))
  )
})

test_that("RunHarmony_scx writes a new reduction onto a Seurat object", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  library(Seurat)
  set.seed(0)
  n <- 150
  n_genes <- 20
  counts <- matrix(
    rpois(n * n_genes, lambda = 2),
    nrow = n_genes,
    ncol = n,
    dimnames = list(paste0("g", seq_len(n_genes)),
                    paste0("c", seq_len(n)))
  )
  meta <- data.frame(
    batch = sample(c("A", "B"), n, replace = TRUE),
    row.names = paste0("c", seq_len(n))
  )
  obj <- CreateSeuratObject(counts = counts, meta.data = meta)
  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(obj, verbose = FALSE)
  obj <- ScaleData(obj, verbose = FALSE)
  obj <- RunPCA(obj, npcs = 10, verbose = FALSE)

  obj <- RunHarmony_scx(obj, group.by.vars = "batch",
                       max_iter = 2L, random_state = 0L)

  expect_true("harmony" %in% names(obj@reductions))
  expect_equal(obj@misc$scx_accel[["harmony_integrate"]]$route, "cpu_dense")
  emb <- Embeddings(obj[["harmony"]])
  expect_equal(nrow(emb), n)
  expect_equal(ncol(emb), 10L)
  expect_true(all(is.finite(emb)))
})
