//! The crate's error type.
//!
//! Its own module because every other submodule returns it, so leaving it in
//! `mod.rs` would make `mod.rs` a dependency of everything rather than the
//! re-export surface it is.

use scx_format_io::error::ScxError;

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),

    #[error("SCX error: {0}")]
    Scx(#[from] ScxError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),

    /// Narrowing an HDF5 source value to the target Rust type would
    /// truncate. Used by `read_*_dataset` / `read_slice_*` when a
    /// source `i64` / `u32` / `u64` value falls outside the target
    /// range. Silent truncation of CSR indptr / indices would corrupt
    /// the on-disk sparse layout, so the conversion fails loudly.
    #[error("value {value} from {source_dtype} dataset '{path}' overflows {target} target range")]
    IndexOverflow {
        path: String,
        source_dtype: &'static str,
        target: &'static str,
        value: String,
    },

    #[error("format mismatch: expected {expected}, got {got}")]
    FormatMismatch { expected: String, got: String },

    /// An `uns` tree nests deeper than the format can carry. Raised by the
    /// h5ad reader on a deep `/uns` group chain and by the h5ad writer on a
    /// deep JSON tree. Without it, both walk the tree with unbounded
    /// recursion and abort the process on a stack overflow — which is not a
    /// catchable error in any binding.
    ///
    /// Reached through the ordinary `strict_uns` channel on the read side, so
    /// the default is a [`crate::ConvertWarning::SkippedUnsKey`] naming the key
    /// rather than a failed conversion: one pathological key in someone
    /// else's h5ad should not make the file unconvertible.
    #[error("uns/{path}: nesting is deeper than the maximum of {max_depth} levels")]
    UnsTooDeep { path: String, max_depth: usize },

    #[error("streaming unsupported: {0}")]
    StreamingUnsupported(String),

    /// A per-shard read or encode failed on one of the
    /// parallel reader workers. The wrapper carries the row range and
    /// source-matrix name so the user knows exactly which shard
    /// produced the error.
    #[error(
        "shard read failed at rows [{row_start}, {}) of '{source}': {inner}",
        row_start + *n_rows as u64
    )]
    ShardRead {
        row_start: u64,
        n_rows: u32,
        source: String,
        #[source]
        inner: Box<ConvertError>,
    },

    #[error("{0}")]
    Other(String),
}

/// A failure of the shared parallel drain itself — pool construction, or the
/// worker channel closing early — as opposed to a failure of any one shard.
/// `crate::parallel_drain` is generic over the error type so it can stay
/// feature-free (and therefore testable without libhdf5); this is how it
/// re-enters `ConvertError`.
impl From<crate::parallel_drain::DrainFailure> for ConvertError {
    fn from(f: crate::parallel_drain::DrainFailure) -> Self {
        ConvertError::Other(f.0)
    }
}
