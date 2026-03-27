//! Rust-native analysis accelerators for SCX files.
//!
//! Provides high-performance implementations of common single-cell analysis
//! operations (PCA, kNN, etc.) that stream data shard-by-shard from SCX's
//! backed mode, avoiding full matrix materialization.

pub mod error;
pub mod pca;

pub use error::{AccelError, Result};
pub use pca::{randomized_pca, randomized_pca_inmemory, PcaResult};
