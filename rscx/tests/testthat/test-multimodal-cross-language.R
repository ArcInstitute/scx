# Phase K.2.2 — R → Python cross-language round-trip.
#
# Writes a small Seurat v5 multi-assay object via `rscx::from_seurat`,
# then invokes a Python subprocess to read the file via
# `pyscx.open(path).to_mudata()` and confirm the modality count + per-
# modality `n_vars` survive the R → Rust → Python chain. Gated on
# the presence of:
#   * Seurat >= 5.0 (writer side)
#   * Matrix (sparse type)
#   * a Python with `pyscx` + `mudata` installed (reader side)
# The runner skips when any prereq is missing so the rscx test suite
# stays green on minimal R / Python environments.


test_that("from_seurat → pyscx.open round-trips modality metadata", {
  skip_if_not_installed("Seurat", minimum_version = "5.0.0")
  skip_if_not_installed("Matrix")

  python <- Sys.which("python")
  if (!nzchar(python)) {
    python <- Sys.getenv("PYTHON", unset = NA)
  }
  if (is.na(python) || !nzchar(python) || !file.exists(python)) {
    skip("python not found on PATH; set PYTHON env var to enable cross-language test")
  }

  # Verify the target Python has pyscx + mudata. Skip gracefully if
  # not — this test is only useful when both bindings exist.
  probe <- system2(
    python,
    args = c("-c", "import pyscx, mudata"),
    stdout = TRUE, stderr = TRUE
  )
  status <- attr(probe, "status")
  if (!is.null(status) && status != 0) {
    skip(paste0("python at ", python, " missing pyscx or mudata"))
  }

  set.seed(0)
  n_cells <- 24
  rna_n <- 10
  adt_n <- 4
  rna_counts <- Matrix::Matrix(
    matrix(rpois(rna_n * n_cells, lambda = 0.5), nrow = rna_n, ncol = n_cells),
    sparse = TRUE
  )
  adt_counts <- Matrix::Matrix(
    matrix(rpois(adt_n * n_cells, lambda = 0.5), nrow = adt_n, ncol = n_cells),
    sparse = TRUE
  )
  rownames(rna_counts) <- paste0("g", seq_len(rna_n))
  rownames(adt_counts) <- paste0("a", seq_len(adt_n))
  colnames(rna_counts) <- paste0("cell_", seq_len(n_cells))
  colnames(adt_counts) <- paste0("cell_", seq_len(n_cells))

  rna_assay <- Seurat::CreateAssay5Object(counts = rna_counts)
  adt_assay <- Seurat::CreateAssay5Object(counts = adt_counts)
  seu <- Seurat::CreateSeuratObject(rna_assay, assay = "rna")
  seu[["adt"]] <- adt_assay

  tmp_dir <- tempfile("rscx_xlang_")
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)
  scx_path <- file.path(tmp_dir, "r_to_py.scx")

  rscx::from_seurat(seu, scx_path)
  expect_true(file.exists(scx_path))

  # Drive a Python subprocess that opens the file via pyscx and prints
  # `n_obs n_rna n_adt` to stdout for parsing.
  py_script <- sprintf(
    "import pyscx; r=pyscx.open(%s); mu=r.to_mudata(); print(mu.n_obs, mu.mod['rna'].n_vars, mu.mod['adt'].n_vars)",
    paste0("'", scx_path, "'")
  )
  out <- system2(python, args = c("-c", py_script), stdout = TRUE, stderr = TRUE)
  status <- attr(out, "status")
  expect_true(is.null(status) || status == 0,
              info = paste("python subprocess failed:", paste(out, collapse = "\n")))

  parts <- strsplit(trimws(out[[length(out)]]), "\\s+")[[1]]
  expect_length(parts, 3)
  n_obs_py <- as.integer(parts[[1]])
  n_rna_py <- as.integer(parts[[2]])
  n_adt_py <- as.integer(parts[[3]])
  expect_equal(n_obs_py, n_cells)
  expect_equal(n_rna_py, rna_n)
  expect_equal(n_adt_py, adt_n)
})
