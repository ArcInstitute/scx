//! Lazy transform chains for rscx.
//!
//! [`RLazyTransformed`] wraps the same Rust-core [`BackedCsrReader`] as
//! [`crate::backed::RBackedSparse`] but carries an ordered chain of cheap
//! per-row transforms (`normalize_total` → `log1p` → `row_scale`) that are
//! applied **on read**, never materialising the whole matrix. This is the
//! R-side equivalent of pyscx's `ScxLazyTransformedDataset`: the canonical
//! out-of-core preprocessing chain on an atlas-scale file.
//!
//! The transform logic lives in pyscx (`pyscx/src/lazy_transform/`) behind
//! PyO3 types, so it is not reusable from R; the kernels here are a small,
//! faithful re-implementation against the shared [`ScxCsr`] / [`BackedCsrReader`]
//! surface (the transforms are simple element-wise / per-row scalings).
//!
//! Chaining is immutable: each transform method returns a **new**
//! `RLazyTransformed` that shares the underlying `Arc<BackedCsrReader>` (and its
//! shard cache) with the same transform chain plus the appended op — mirroring
//! how `ScxLazyTransformedDataset` composes. Transforms preserve the sparsity
//! pattern (`normalize_total`/`row_scale` scale nonzeros; `log1p(0) = 0`), so
//! `nnz` is unchanged and reported straight from the backed catalog.

use std::sync::Arc;

use extendr_api::prelude::*;
use scx_format_io::backed::BackedCsrReader;
use scx_format_io::ScxReader;
use scx_sparse::ScxCsr;

use crate::util::throw_on_err;

/// Map a core `scx-format-io` error into an extendr error carrying its message.
fn to_other<E: std::fmt::Display>(e: E) -> Error {
    Error::Other(e.to_string())
}

/// One step in a lazy preprocessing chain. Per-row state (`row_sums`,
/// `factors`) is precomputed once at append time and shared via `Arc`.
#[derive(Clone)]
enum RTransform {
    /// `x * (target_sum / row_sum)` — `sc.pp.normalize_total`.
    NormalizeTotal {
        row_sums: Arc<Vec<f64>>,
        target_sum: f64,
    },
    /// `ln(x + 1)` — `sc.pp.log1p`.
    Log1p,
    /// `x * factors[row]` — per-cell scaling.
    RowScale { factors: Arc<Vec<f64>> },
}

/// Apply a transform chain in-place to a decoded CSR whose rows correspond to
/// the global row ids in `global_rows` (parallel to the CSR rows). Using an
/// explicit id slice keeps it correct for both contiguous ranges and arbitrary
/// (reordered) fancy-index reads.
fn apply_transforms(transforms: &[RTransform], csr: &mut ScxCsr, global_rows: &[u64]) {
    for t in transforms {
        match t {
            RTransform::NormalizeTotal {
                row_sums,
                target_sum,
            } => {
                for (row, &g) in global_rows.iter().enumerate() {
                    let sum = row_sums[g as usize];
                    if sum > 0.0 {
                        let factor = *target_sum / sum;
                        let start = csr.indptr[row] as usize;
                        let end = csr.indptr[row + 1] as usize;
                        for v in &mut csr.data[start..end] {
                            *v = (*v as f64 * factor) as f32;
                        }
                    }
                }
            }
            RTransform::Log1p => {
                for v in &mut csr.data {
                    *v = v.ln_1p();
                }
            }
            RTransform::RowScale { factors } => {
                for (row, &g) in global_rows.iter().enumerate() {
                    let factor = factors[g as usize];
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    for v in &mut csr.data[start..end] {
                        *v = (*v as f64 * factor) as f32;
                    }
                }
            }
        }
    }
}

/// R class: a backed CSR reader plus an ordered, lazily-applied transform chain.
#[extendr]
pub struct RLazyTransformed {
    backed: Arc<BackedCsrReader>,
    transforms: Vec<RTransform>,
    n_obs: usize,
    n_vars: usize,
}

impl RLazyTransformed {
    /// Open a fresh backed reader with an empty transform chain (mirrors
    /// `RBackedSparse::open_impl`).
    pub(crate) fn open_impl(path: &str, cache_shards: usize) -> Result<Self> {
        let reader = ScxReader::open(path)
            .map_err(|e| Error::Other(format!("failed to open SCX file '{}': {}", path, e)))?;
        let backed = BackedCsrReader::new(reader, cache_shards.max(1));
        let (n_obs, n_vars) = backed.shape();
        Ok(Self {
            backed: Arc::new(backed),
            transforms: Vec::new(),
            n_obs,
            n_vars,
        })
    }

    fn with_transform(&self, t: RTransform) -> Self {
        let mut transforms = self.transforms.clone();
        transforms.push(t);
        Self {
            backed: Arc::clone(&self.backed),
            transforms,
            n_obs: self.n_obs,
            n_vars: self.n_vars,
        }
    }

    /// Per-row sums of the data **as transformed by the current chain** — the
    /// state a freshly-appended `normalize_total` needs. Fast-paths to the
    /// backed reader's raw `row_sums` when the chain is empty.
    fn transformed_row_sums(&self) -> Result<Vec<f64>> {
        if self.transforms.is_empty() {
            return self.backed.row_sums().map_err(to_other);
        }
        let mut sums = vec![0.0f64; self.n_obs];
        self.stream_transformed(|csr, s_start| {
            for row in 0..csr.n_rows() {
                let start = csr.indptr[row] as usize;
                let end = csr.indptr[row + 1] as usize;
                let acc: f64 = csr.data[start..end].iter().map(|&v| v as f64).sum();
                sums[s_start as usize + row] = acc;
            }
        })?;
        Ok(sums)
    }

    /// Stream every shard with the chain applied, invoking `f(csr, shard_start)`.
    fn stream_transformed<F: FnMut(&ScxCsr, u64)>(&self, mut f: F) -> Result<()> {
        let n_shards = self.backed.index().n_shards();
        for s in 0..n_shards {
            let (s_start, _s_end) = self
                .backed
                .index()
                .shard_range(s)
                .ok_or_else(|| Error::Other(format!("shard {s} out of range")))?;
            let mut csr = self.backed.read_shard_uncached(s).map_err(to_other)?;
            let nrow = csr.n_rows();
            let global_rows: Vec<u64> = (s_start..s_start + nrow as u64).collect();
            apply_transforms(&self.transforms, &mut csr, &global_rows);
            f(&csr, s_start);
        }
        Ok(())
    }

    // NOTE: lazy-transform reads are not gated for the u32→f32 decode loss —
    // they decode per row-slice through a `BackedCsrReader` (no catalog in
    // scope), and the normalize/log1p/row_scale transforms already break the
    // integer-counts invariant. Consistent with ungated backed reads elsewhere.
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
        let mut csr = self.backed.read_rows(start, end).map_err(to_other)?;
        let global_rows: Vec<u64> = (start..start + csr.n_rows() as u64).collect();
        apply_transforms(&self.transforms, &mut csr, &global_rows);
        crate::interop::csr_to_dgcmatrix(&csr)
    }

    fn read_row_indices_impl(&self, idx0: &[u64]) -> Result<Robj> {
        for &r in idx0 {
            if r >= self.n_obs as u64 {
                return Err(Error::Other(format!(
                    "row index {} out of bounds for n_obs {}",
                    r, self.n_obs
                )));
            }
        }
        let mut csr = self.backed.read_row_indices(idx0).map_err(to_other)?;
        // read_row_indices returns rows in the requested order, so the global id
        // of CSR row k is idx0[k].
        apply_transforms(&self.transforms, &mut csr, idx0);
        crate::interop::csr_to_dgcmatrix(&csr)
    }

    fn to_dgcmatrix_impl(&self) -> Result<Robj> {
        let mut csr = self.backed.read_all().map_err(to_other)?;
        let global_rows: Vec<u64> = (0..self.n_obs as u64).collect();
        apply_transforms(&self.transforms, &mut csr, &global_rows);
        crate::interop::csr_to_dgcmatrix(&csr)
    }

    fn normalize_total_impl(&self, target_sum: f64) -> Result<Self> {
        if target_sum <= 0.0 || target_sum.is_nan() {
            return Err(Error::Other("target_sum must be > 0".into()));
        }
        let row_sums = Arc::new(self.transformed_row_sums()?);
        Ok(self.with_transform(RTransform::NormalizeTotal {
            row_sums,
            target_sum,
        }))
    }

    fn row_scale_impl(&self, factors: Vec<f64>) -> Result<Self> {
        if factors.len() != self.n_obs {
            return Err(Error::Other(format!(
                "row_scale factors length {} != n_obs {}",
                factors.len(),
                self.n_obs
            )));
        }
        Ok(self.with_transform(RTransform::RowScale {
            factors: Arc::new(factors),
        }))
    }

    fn col_sums_impl(&self) -> Result<Vec<f64>> {
        let mut sums = vec![0.0f64; self.n_vars];
        self.stream_transformed(|csr, _s_start| {
            for k in 0..csr.data.len() {
                sums[csr.indices[k] as usize] += csr.data[k] as f64;
            }
        })?;
        Ok(sums)
    }
}

#[extendr]
impl RLazyTransformed {
    /// Open a lazy-transform handle (empty chain) over an SCX file's X matrix.
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

    /// Total non-zero entries (unchanged by the chain; from catalog stats).
    fn nnz(&self) -> Robj {
        throw_on_err(
            self.backed
                .total_nnz()
                .map(|n| Robj::from(n as f64))
                .map_err(to_other),
        )
    }

    /// Names of the transforms in the current chain (for printing).
    fn transform_names(&self) -> Vec<String> {
        self.transforms
            .iter()
            .map(|t| {
                match t {
                    RTransform::NormalizeTotal { .. } => "normalize_total",
                    RTransform::Log1p => "log1p",
                    RTransform::RowScale { .. } => "row_scale",
                }
                .to_string()
            })
            .collect()
    }

    /// Append `normalize_total(target_sum)`; returns a new lazy handle.
    fn normalize_total(&self, target_sum: f64) -> Robj {
        throw_on_err(self.normalize_total_impl(target_sum))
    }

    /// Append `log1p`; returns a new lazy handle.
    fn log1p(&self) -> Robj {
        self.with_transform(RTransform::Log1p).into()
    }

    /// Append `row_scale(factors)` (per-cell factors, length n_obs); returns a
    /// new lazy handle.
    fn row_scale(&self, factors: Vec<f64>) -> Robj {
        throw_on_err(self.row_scale_impl(factors))
    }

    /// Read a contiguous 0-based, half-open `[start, end)` row range as a
    /// transformed dgCMatrix.
    fn read_rows(&self, start: f64, end: f64) -> Robj {
        throw_on_err(self.read_rows_impl(start as u64, end as u64))
    }

    /// Read arbitrary 0-based rows (in the given order) as a transformed dgCMatrix.
    fn read_row_indices(&self, indices: Vec<f64>) -> Robj {
        let idx0: Vec<u64> = indices.iter().map(|&r| r as u64).collect();
        throw_on_err(self.read_row_indices_impl(&idx0))
    }

    /// Per-row sums over the transformed data (streamed, no full materialize).
    fn row_sums(&self) -> Robj {
        throw_on_err(self.transformed_row_sums())
    }

    /// Per-column sums over the transformed data (streamed, no full materialize).
    fn col_sums(&self) -> Robj {
        throw_on_err(self.col_sums_impl())
    }

    /// Materialise the full transformed matrix as a dgCMatrix (escape hatch).
    fn to_dgcmatrix(&self) -> Robj {
        throw_on_err(self.to_dgcmatrix_impl())
    }
}

extendr_module! {
    mod lazy;
    impl RLazyTransformed;
}
