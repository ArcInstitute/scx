//! Backed (lazy, out-of-core) sparse access for rscx.
//!
//! [`RBackedSparse`] wraps the Rust-core [`BackedCsrReader`], which reads X rows
//! on demand from disk with an O(log n) shard lookup and a shard-level LRU
//! cache. This is the R-side equivalent of pyscx's `ScxBackedSparseDataset`:
//! atlas-scale files that don't fit in RAM can be sliced row-by-row instead of
//! materialised whole via [`ScxExperiment::x_matrix`](crate::ScxExperiment).
//!
//! Every fallible `#[extendr]` method returns `Robj` and routes its `Result`
//! through [`crate::util::throw_on_err`] so R sees a clean `stop()` rather than
//! the opaque "User function panicked" extendr 0.8.0 emits on a `Result` `Err`.
//!
//! Row indices cross the FFI boundary as R doubles (`f64`), matching how
//! `n_obs`/`n_vars`/`nnz` are returned, so files with >2^31 cells stay
//! addressable. The R-side `[` method converts 1-based R indices to the 0-based
//! forms the core expects; the Rust layer here treats indices as already
//! 0-based and bounds-checks them so an out-of-range request raises a clean
//! error instead of silently truncating.

use std::path::PathBuf;
use std::sync::Arc;

use extendr_api::prelude::*;
use scx_format_io::backed::BackedCsrReader;
use scx_format_io::ScxReader;

use crate::util::throw_on_err;

/// Map a core `scx-format-io` error into an extendr error carrying its message.
fn to_other<E: std::fmt::Display>(e: E) -> Error {
    Error::Other(e.to_string())
}

/// R class wrapping a [`BackedCsrReader`] for lazy, on-demand row access to X.
#[extendr]
pub struct RBackedSparse {
    backed: Arc<BackedCsrReader>,
    n_obs: usize,
    n_vars: usize,
}

impl RBackedSparse {
    /// Fallible body of `new`: open a fresh reader and wrap it in a backed CSR
    /// reader. Mirrors `ScxExperiment::query_impl`, which likewise re-opens the
    /// file (the backed reader takes ownership of its own `ScxReader`).
    pub(crate) fn open_impl(path: &str, cache_shards: usize) -> Result<Self> {
        let path_buf = PathBuf::from(path);
        let reader = ScxReader::open(&path_buf)
            .map_err(|e| Error::Other(format!("failed to open SCX file '{}': {}", path, e)))?;
        // A cache of zero shards would defeat the LRU; clamp to at least one.
        let cache_shards = cache_shards.max(1);
        let backed = BackedCsrReader::new(reader, cache_shards);
        let (n_obs, n_vars) = backed.shape();
        Ok(Self {
            backed: Arc::new(backed),
            n_obs,
            n_vars,
        })
    }

    /// Fallible body of `read_rows`: 0-based, half-open `[start, end)`.
    fn read_rows_impl(&self, start: u64, end: u64) -> Result<Robj> {
        if end > self.n_obs as u64 {
            return Err(Error::Other(format!(
                "row range end {} exceeds n_obs {}",
                end, self.n_obs
            )));
        }
        let csr = self.backed.read_rows(start, end).map_err(to_other)?;
        crate::interop::csr_to_dgcmatrix(&csr)
    }

    /// Fallible body of `read_row_indices`: arbitrary 0-based rows, in the order
    /// given (the core re-sorts internally but restores the requested order).
    fn read_row_indices_impl(&self, idx0: &[u64]) -> Result<Robj> {
        for &r in idx0 {
            if r >= self.n_obs as u64 {
                return Err(Error::Other(format!(
                    "row index {} out of bounds for n_obs {}",
                    r, self.n_obs
                )));
            }
        }
        let csr = self.backed.read_row_indices(idx0).map_err(to_other)?;
        crate::interop::csr_to_dgcmatrix(&csr)
    }

    /// Fallible body of `to_dgcmatrix`: materialise the full matrix (escape
    /// hatch matching `ScxExperiment$x_matrix()`).
    fn to_dgcmatrix_impl(&self) -> Result<Robj> {
        let csr = self.backed.read_all().map_err(to_other)?;
        crate::interop::csr_to_dgcmatrix(&csr)
    }
}

#[extendr]
impl RBackedSparse {
    /// Open a backed sparse handle over an SCX file's X matrix.
    ///
    /// Returns `Robj` (the externalptr wrapping `Self`) and throws a clean R
    /// error via `throw_on_err` on failure, for the same reason the
    /// `ScxExperiment` constructor does (see B3 / `crate::util::throw_on_err`).
    #[allow(clippy::new_ret_no_self)]
    fn new(path: &str, cache_shards: f64) -> Robj {
        throw_on_err(Self::open_impl(path, cache_shards as usize))
    }

    /// Number of observations (cells). R numeric (f64) to allow >2B cells.
    fn n_obs(&self) -> Robj {
        Robj::from(self.n_obs as f64)
    }

    /// Number of variables (genes).
    fn n_vars(&self) -> Robj {
        Robj::from(self.n_vars as f64)
    }

    /// Total non-zero entries (streamed from catalog stats, no decode).
    fn nnz(&self) -> Robj {
        throw_on_err(
            self.backed
                .total_nnz()
                .map(|n| Robj::from(n as f64))
                .map_err(to_other),
        )
    }

    /// Read a contiguous 0-based, half-open `[start, end)` row range as a
    /// dgCMatrix. The R `[` method converts from 1-based indices.
    fn read_rows(&self, start: f64, end: f64) -> Robj {
        throw_on_err(self.read_rows_impl(start as u64, end as u64))
    }

    /// Read arbitrary 0-based rows (in the given order) as a dgCMatrix.
    fn read_row_indices(&self, indices: Vec<f64>) -> Robj {
        let idx0: Vec<u64> = indices.iter().map(|&r| r as u64).collect();
        throw_on_err(self.read_row_indices_impl(&idx0))
    }

    /// Per-row sums over all genes (streamed shard-by-shard, no full decode).
    fn row_sums(&self) -> Robj {
        throw_on_err(self.backed.row_sums().map_err(to_other))
    }

    /// Per-column (per-gene) sums over all cells (streamed, no full decode).
    fn col_sums(&self) -> Robj {
        throw_on_err(self.backed.col_sums().map_err(to_other))
    }

    /// Materialise the full matrix as a dgCMatrix (escape hatch; loads all
    /// shards). Prefer row slicing via `[` for large files.
    fn to_dgcmatrix(&self) -> Robj {
        throw_on_err(self.to_dgcmatrix_impl())
    }
}

extendr_module! {
    mod backed;
    impl RBackedSparse;
}
