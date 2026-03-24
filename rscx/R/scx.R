#' Open an SCX file
#'
#' Opens an SCX file and returns a lazy handle. No data is read until
#' you access obs, var, x_matrix, or call query methods.
#'
#' @param path Path to the .scx file
#' @return An ScxExperiment object
#' @export
#' @examples
#' exp <- scx_open("experiment.scx")
#' exp$n_obs()
#' exp$n_vars()
scx_open <- function(path) {
  ScxExperiment$new(path)
}

#' Show SCX file information
#'
#' @param path Path to the .scx file
#' @return A named list with file metadata
#' @export
scx_info <- function(path) {
  .Call(wrap__scx_info, path)
}

#' Validate an SCX file
#'
#' @param path Path to the .scx file
#' @return TRUE if valid, raises an error otherwise
#' @export
scx_validate <- function(path) {
  .Call(wrap__scx_validate, path)
}
