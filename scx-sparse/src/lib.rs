pub mod convert;
pub mod csc;
pub mod csr;
pub mod transpose;
pub mod umap_math;
pub mod validate;

pub use convert::{csr_to_dense, dense_to_csr};
pub use csc::{CscError, ScxCsc};
pub use csr::{CsrError, ScxCsr};
pub use transpose::{
    compute_chunk_cols_with_cap, streaming_csr_to_csc_iter_with_cap, CscArrays, CscShardIterator,
    TransposeError,
};
pub use umap_math::{compute_epochs_per_sample, find_ab_params, random_init_f32, random_init_f64};
pub use validate::{
    canonicalize_csr, coalesce_sorted_coo, drop_explicit_zeros_inplace, rebase_csr_shard,
    shard_nnz_bounds, sort_csr_rows_in_place, validate_csr_arrays, validate_sparse_layout,
    SparseLayout,
};
