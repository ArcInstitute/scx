pub mod convert;
pub mod csc;
pub mod csr;
pub mod materialize;
pub mod moments;
pub mod transpose;
pub mod umap_math;
pub mod validate;

pub use convert::{csr_to_dense, dense_to_csr};
pub use csc::{CscError, ScxCsc};
pub use csr::{
    concatenate_csr, finalize_implicit_zero_variance, implicit_zero_count,
    total_variance_from_col_sq, CsrArrays, CsrError, ScxCsr,
};
pub use materialize::{
    Container, IndexBuffer, IndexDtype, MaterializePlan, TypedCsr, TypedDense, ValueBuffer,
    ValueDtype,
};
pub use moments::{
    closed_form_variance_health, closed_form_variance_unstable, finalize_column_moments,
    first_non_finite_column, residual_lost_to_cancellation, ClosedFormVarianceHealth,
    ColumnMoments, CLOSED_FORM_VAR_REL_EPS,
};
pub use transpose::{
    compute_chunk_cols_with_cap, streaming_csr_to_csc_iter_with_cap, CscArrays, CscShardIterator,
    TransposeError,
};
pub use umap_math::{compute_epochs_per_sample, find_ab_params, random_init_f32, random_init_f64};
pub use validate::{
    canonicalize_csr, coalesce_sorted_coo, drop_explicit_zeros_inplace, is_canonical_csr,
    rebase_csr_shard, shard_nnz_bounds, sort_csr_rows_in_place, validate_csr_arrays,
    validate_sparse_layout, SparseLayout,
};
