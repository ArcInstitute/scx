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
//! 0-based.
//!
//! Every index is run through [`crate::util::r_whole_u64`] **before** the cast
//! to `u64`, and only then bounds-checked. Both halves are needed: the bounds
//! check alone rejects values above `n_obs`, but `f64 as u64` saturates, so
//! `-1.0` and `NaN` arrive as `0` — in range, and silently the wrong cells.
//! `$read_rows()` / `$read_row_indices()` are exported `$`-methods that bypass
//! `[`'s R-side guards entirely, so this layer is the only check they get.

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
        // Clamp on both ends. Zero would defeat the LRU; the upper bound is
        // load-bearing because `cache_shards` flows into `LruCache::new`, which
        // pre-allocates a `HashMap` of that capacity — `cache_shards = 1e9`
        // would reserve tens of gigabytes and abort the R session with no
        // catchable error. A cache larger than the shard count is useless
        // anyway, so the file's own shard count is the natural ceiling.
        let n_shards = reader.header().n_csr_shards as usize;
        let cache_shards = crate::util::clamp_cache_shards(cache_shards, n_shards);
        let backed = BackedCsrReader::new(reader, cache_shards);
        let (n_obs, n_vars) = backed.shape();
        Ok(Self {
            backed: Arc::new(backed),
            n_obs,
            n_vars,
        })
    }

    // NOTE: backed reads are not gated for the u32→f32 decode loss (see the
    // decode-loss guard). They decode lazily per row-slice through a
    // `BackedCsrReader` (no `ScxReader`/catalog in scope), so a whole-file check
    // would spuriously error on partial reads that never touch the large-count
    // shard — consistent with ungated backed reads on the Python side.

    /// Fallible body of `read_rows`: 0-based, half-open `[start, end)`.
    fn read_rows_impl(&self, start: u64, end: u64) -> Result<Robj> {
        if start > end {
            return Err(Error::Other(format!(
                "row range start ({start}) must be <= end ({end})"
            )));
        }
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
        throw_on_err((|| -> Result<Self> {
            let cache_shards = crate::util::r_whole_usize(cache_shards, "cache_shards")?;
            Self::open_impl(path, cache_shards)
        })())
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
    ///
    /// Both arguments are validated before the cast: `f64 as u64` saturates, so
    /// an unchecked `-1` or `NaN` would read row 0 and return the wrong cells.
    fn read_rows(&self, start: f64, end: f64) -> Robj {
        throw_on_err((|| -> Result<Robj> {
            let start = crate::util::r_whole_u64(start, "0-based row range start")?;
            let end = crate::util::r_whole_u64(end, "0-based row range end")?;
            self.read_rows_impl(start, end)
        })())
    }

    /// Read arbitrary 0-based rows (in the given order) as a dgCMatrix.
    ///
    /// The `Vec<f64>` form is where validation matters most: extendr rejects
    /// `NA` for a scalar `f64` argument, but not for an element of a vector.
    fn read_row_indices(&self, indices: Vec<f64>) -> Robj {
        throw_on_err((|| -> Result<Robj> {
            let idx0 = crate::util::r_whole_u64_slice(&indices, "0-based row index")?;
            self.read_row_indices_impl(&idx0)
        })())
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
