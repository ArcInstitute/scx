# Shared test helpers and fixture paths.

# Path to the pre-built tiny SCX fixture (20 cells x 10 genes).
fixture_path <- function(name = "tiny.scx") {
  system.file("tests", "testthat", "fixtures", name, package = "rscx",
              mustWork = FALSE)
}

# Fallback: find fixture relative to test directory (for devtools::test)
fixture_dir <- function() {
  candidates <- c(
    file.path("fixtures"),
    file.path("..", "testthat", "fixtures"),
    file.path("..", "..", "tests", "testthat", "fixtures")
  )
  for (d in candidates) {
    if (dir.exists(d)) return(d)
  }
  NULL
}

get_fixture <- function(name = "tiny.scx") {
  # Try installed location first
  p <- fixture_path(name)
  if (nzchar(p) && file.exists(p)) return(p)
  # Try relative path (devtools::test / testthat::test_local)
  d <- fixture_dir()
  if (!is.null(d)) {
    p <- file.path(d, name)
    if (file.exists(p)) return(p)
  }
  NULL
}

skip_if_no_fixture <- function(name = "tiny.scx") {
  p <- get_fixture(name)
  if (is.null(p)) skip(paste("Fixture not found:", name))
  p
}
