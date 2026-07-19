# Documentation for the extendr-generated handle classes. The class objects
# themselves (environment-based generators) and their `@export` tags are emitted
# into R/extendr-wrappers.R by rextendr; these companion blocks only supply the
# help page so the exported objects are documented (avoids the R CMD check
# "undocumented code objects" warning). Do not add method bodies here.

#' rscx handle classes
#'
#' Environment-based handle classes returned by the rscx entry points. Each
#' wraps a pointer to a Rust-side object; call methods with the `$` operator
#' (e.g. \code{exp$n_obs()}). Construct them through the documented front ends
#' rather than directly.
#'
#' @details
#' \describe{
#'   \item{\code{ScxExperiment}}{Open-file handle returned by
#'     \code{\link{scx_open}}. Methods include \code{n_obs()}, \code{n_vars()},
#'     \code{nnz()}, \code{obs()}, \code{var()}, \code{x_matrix()},
#'     \code{layer()}, \code{query()}, \code{is_multimodal()},
#'     \code{modality_names()}, \code{to_seurat()}, \code{to_mae()},
#'     \code{read_group()}, \code{iter_group_shards()}, \code{x_backed()},
#'     \code{x_lazy()}.}
#'   \item{\code{ScxBackedSparse}}{Lazily-decoded backed sparse matrix from
#'     \code{\link{scx_backed_sparse}}; supports \code{[}, \code{dim},
#'     \code{as.matrix}, \code{row_sums()}, \code{col_sums()}.}
#'   \item{\code{ScxLazyTransformed}}{Backed matrix with a deferred
#'     normalize/log1p/row-scale chain from \code{\link{scx_lazy_transform}}.}
#'   \item{\code{RQueryPipeline}}{Query builder from \code{\link{scx_query}};
#'     chain \code{\link{filter_obs}}, \code{\link{filter_var}},
#'     \code{\link{select_genes}}, \code{\link{with_normalize}},
#'     \code{\link{with_log1p}}, \code{\link{limit}}, then
#'     \code{\link{collect}} / \code{\link{count}}.}
#'   \item{\code{RQueryResult}}{Materialised query result; coerce with
#'     \code{as.matrix}, \code{as.data.frame}, or \code{to_seurat()}.}
#'   \item{\code{RGroupShardHandle}}{One shard of a grouped (sorted) layout,
#'     yielded by \code{ScxExperiment$iter_group_shards()}.}
#' }
#'
#' @name rscx-classes
#' @rdname rscx-classes
NULL

#' @rdname rscx-classes
#' @name ScxExperiment
NULL

#' @rdname rscx-classes
#' @name ScxBackedSparse
NULL

#' @rdname rscx-classes
#' @name ScxLazyTransformed
NULL

#' @rdname rscx-classes
#' @name RQueryPipeline
NULL

#' @rdname rscx-classes
#' @name RQueryResult
NULL

#' @rdname rscx-classes
#' @name RGroupShardHandle
NULL
