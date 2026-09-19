//! CSC (column-major) consumer kernels.
//!
//! Submodules implement column-major variants of the streaming kernels
//! used by analysis accelerators. Each kernel takes a generic
//! `S: ColumnShardSource` from `scx-format` so it works with both
//! `BackedCscReader` and the pyscx `LazyShardSource` (transform-aware).
//!
//! Dispatch is explicit: callers thread a `PreferFormat` enum or
//! `prefer_format` kwarg through their entry points and use
//! [`require_csc`] to translate that into a CSC-capable source or a
//! clean error when the dataset doesn't support CSC. There is no
//! heuristic / thread-local default — every CSC dispatch is opt-in
//! at the call site.

pub mod dispatch;
pub mod mean_var;
pub mod pdex;
pub mod pseudobulk;
pub mod wilcoxon;

// Reached from `crate::hvg::moments_golden` as well as this module's own
// tests, so it is `pub(crate)` rather than private.
#[cfg(test)]
pub(crate) mod test_helpers;

#[cfg(test)]
mod parity_test;

pub use dispatch::{require_csc, PreferFormat};
pub use mean_var::{streaming_clip_square_sum_csc, streaming_mean_var_csc};
#[cfg(feature = "gpu")]
pub use mean_var::{streaming_clip_square_sum_csc_with_device, streaming_mean_var_csc_with_device};
pub use pdex::pdex_ref_streaming_csc;
pub use pseudobulk::pseudobulk_aggregate_csc;
pub use wilcoxon::{csc_wilcoxon_uses_nnz_kernel, wilcoxon_rank_sum_streaming_csc};
