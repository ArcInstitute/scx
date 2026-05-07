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
//! heuristic / thread-local default — see CSC-SUPPORT.md.

pub mod dispatch;
pub mod mean_var;
pub mod pseudobulk;
pub mod wilcoxon;

#[cfg(test)]
mod test_helpers;

#[cfg(test)]
mod parity_test;

pub use dispatch::{require_csc, PreferFormat};
pub use mean_var::{streaming_clip_square_sum_csc, streaming_mean_var_csc};
pub use pseudobulk::pseudobulk_aggregate_csc;
pub use wilcoxon::wilcoxon_rank_sum_streaming_csc;
