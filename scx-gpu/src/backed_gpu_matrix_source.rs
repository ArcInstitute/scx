//! [`GpuMatrixSource`] impl over a CPU-side CSR shard source plus an optional
//! CSC sidecar.
//!
//! [`BackedGpuMatrixSource`] covers every untransformed input the GPU DE /
//! column-algorithm paths see today — they all reduce to "a CSR
//! [`ShardSource`] + an optional CSC [`ColumnShardSource`]":
//!
//! - backed SCX with a CSC sidecar → [`with_csc`](BackedGpuMatrixSource::with_csc)
//!   (reports `CSR | CSC`);
//! - backed SCX without a sidecar, or a lazy `dyn ShardSource` → [`new`](BackedGpuMatrixSource::new)
//!   (reports `CSR`);
//! - in-memory scipy CSR → [`new`](BackedGpuMatrixSource::new) over an
//!   [`InMemoryCsrShardSource`](crate::staging::InMemoryCsrShardSource).

use std::ops::Range;

use scx_format_io::{ColumnShardSource, ShardSource};

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_csc_shard_source::{GpuCscShardSource, GpuCscShardView, RawGpuCscShardSource};
use crate::gpu_matrix_source::{GpuMatrixSource, LayoutSet, ValidationPolicy};
use crate::gpu_shard_source::{GpuShardSource, RawGpuShardSource};
use crate::staging::GpuCsrSlot;

/// A [`GpuMatrixSource`] backed by a CSR shard source, a CSC sidecar source, or
/// both.
///
/// Which layouts [`available_layouts`](GpuMatrixSource::available_layouts)
/// reports follows the constructor, and **CSR is not guaranteed**:
///
/// | constructor | layouts |
/// |---|---|
/// | [`new`](Self::new) | `CSR` |
/// | [`with_csc`](Self::with_csc) | `CSR \| CSC` |
/// | [`csc_only`](Self::csc_only) | `CSC` |
///
/// A `csc_only` source returns [`GpuError::UnsupportedLayout`] from the CSR
/// iterator. It exists because the column-major HVG reduces are handed a
/// [`ColumnShardSource`] and nothing else; before it, those two call sites
/// could not reach `GpuMatrixSource` at all and so reached past it.
pub struct BackedGpuMatrixSource<'a> {
    csr: Option<RawGpuShardSource<'a>>,
    csc: Option<RawGpuCscShardSource<'a>>,
    /// Shape when there is no CSR adapter to ask. `None` whenever `csr` is
    /// `Some`, so the two can never disagree.
    csc_shape: Option<(usize, usize)>,
}

impl<'a> BackedGpuMatrixSource<'a> {
    /// CSR-only source. Use for lazy `dyn ShardSource`, in-memory CSR (via
    /// [`InMemoryCsrShardSource`](crate::staging::InMemoryCsrShardSource)), or a
    /// backed reader without a CSC sidecar.
    pub fn new(
        dev: &'a GpuDevice,
        csr_source: &'a (dyn ShardSource + Sync),
    ) -> Result<Self, GpuError> {
        Ok(Self {
            csr: Some(RawGpuShardSource::new(dev, csr_source)?),
            csc: None,
            csc_shape: None,
        })
    }

    /// CSC-only source. Reports `CSC` and returns
    /// [`GpuError::UnsupportedLayout`] from the CSR iterator.
    ///
    /// Exists because the column-major HVG reduces are handed a
    /// [`ColumnShardSource`] and nothing else — there is no CSR side to pair
    /// with. Without it those two sites cannot reach `GpuMatrixSource` at all,
    /// which is what kept them calling `RawGpuCscShardSource` directly while
    /// the CSR reduces two functions above went through the unified surface
    /// (review §8.20: *the same file using both conventions*).
    ///
    /// `shape()` comes from the CSC source, so `n_obs` is the file-wide cell
    /// count the sidecar's global row indices are numbered against.
    pub fn csc_only(
        dev: &'a GpuDevice,
        csc_source: &'a (dyn ColumnShardSource + Sync),
    ) -> Result<Self, GpuError> {
        Ok(Self {
            csr: None,
            csc: Some(RawGpuCscShardSource::new(dev, csc_source)?),
            csc_shape: Some(csc_source.shape()),
        })
    }

    /// CSR + CSC source. `csc_source` is the column-major sidecar; the result
    /// reports `CSR | CSC` and serves
    /// [`for_each_gpu_csc_shard_in_range`](GpuMatrixSource::for_each_gpu_csc_shard_in_range).
    pub fn with_csc(
        dev: &'a GpuDevice,
        csr_source: &'a (dyn ShardSource + Sync),
        csc_source: &'a (dyn ColumnShardSource + Sync),
    ) -> Result<Self, GpuError> {
        Ok(Self {
            csr: Some(RawGpuShardSource::new(dev, csr_source)?),
            csc: Some(RawGpuCscShardSource::new(dev, csc_source)?),
            csc_shape: None,
        })
    }

    /// Set the validation policy on whichever layouts this source provides.
    ///
    /// Applied to both inner adapters, so a consumer cannot end up validating a
    /// CSR shard to one depth and the CSC sidecar of the same matrix to
    /// another — which is the second half of §8.14: HVG's CSR and CSC entry
    /// points used to disagree about whether a given file was valid.
    pub fn with_validation(mut self, validation: ValidationPolicy) -> Self {
        self.csr = self.csr.map(|c| c.with_validation(validation));
        self.csc = self.csc.map(|c| c.with_validation(validation));
        self
    }
}

impl GpuMatrixSource for BackedGpuMatrixSource<'_> {
    fn shape(&self) -> (usize, usize) {
        match (&self.csr, self.csc_shape) {
            (Some(csr), _) => csr.shape(),
            (None, Some(shape)) => shape,
            // Unreachable: every constructor supplies at least one adapter,
            // and `csc_shape` is `Some` exactly when `csr` is `None`. Panicking
            // rather than answering `(0, 0)` because a silently empty matrix is
            // the worst possible way for an invariant slip to surface — every
            // downstream loop would just do nothing and report success.
            (None, None) => unreachable!(
                "BackedGpuMatrixSource has neither a CSR adapter nor a CSC shape; \
                 every constructor establishes one of the two"
            ),
        }
    }

    fn available_layouts(&self) -> LayoutSet {
        let mut set = LayoutSet::EMPTY;
        if self.csr.is_some() {
            set = set | LayoutSet::CSR;
        }
        if self.csc.is_some() {
            set = set | LayoutSet::CSC;
        }
        set
    }

    fn for_each_gpu_csr_shard(
        &mut self,
        f: &mut dyn FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        match self.csr.as_mut() {
            Some(csr) => csr.for_each_gpu_shard(|idx, slot| f(idx, slot)),
            None => Err(GpuError::UnsupportedLayout(
                "BackedGpuMatrixSource was constructed CSC-only".to_string(),
            )),
        }
    }

    fn for_each_gpu_csc_shard_in_range(
        &mut self,
        col_range: Range<u32>,
        f: &mut dyn FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        match self.csc.as_mut() {
            Some(csc) => csc.for_each_gpu_csc_shard_in_range(col_range, |idx, view| f(idx, view)),
            None => Err(GpuError::UnsupportedLayout(
                "BackedGpuMatrixSource was constructed without a CSC sidecar".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_sparse::{ScxCsc, ScxCsr};

    // In-memory CSR `ShardSource` (one `ScxCsr` per shard).
    struct InMemCsr {
        shards: Vec<ScxCsr>,
        n_obs: usize,
        n_vars: usize,
    }
    impl ShardSource for InMemCsr {
        fn n_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_shard(&self, idx: usize) -> scx_format_io::Result<ScxCsr> {
            Ok(self.shards[idx].clone())
        }
    }

    // In-memory CSC `ColumnShardSource` (mirrors the fixture in
    // `gpu_csc_shard_source::tests`).
    struct InMemCsc {
        shards: Vec<ScxCsc>,
        ranges: Vec<(u32, u32)>,
        n_obs: usize,
        n_vars: usize,
    }
    impl ColumnShardSource for InMemCsc {
        fn n_csc_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_csc_shard(&self, idx: usize) -> scx_format_io::Result<ScxCsc> {
            Ok(self.shards[idx].clone())
        }
        fn read_csc_columns(&self, _r: Range<u32>) -> scx_format_io::Result<ScxCsc> {
            unimplemented!("test stub")
        }
        fn csc_shard_col_range(&self, idx: usize) -> Option<(u32, u32)> {
            self.ranges.get(idx).copied()
        }
    }

    // Two CSR shards: rows 0..3 and 3..5, 4 cols, one nonzero per row.
    fn csr_fixture() -> InMemCsr {
        let s0 =
            ScxCsr::new_unchecked((3, 4), vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        let s1 = ScxCsr::new_unchecked((2, 4), vec![0, 1, 2], vec![3, 0], vec![4.0, 5.0]);
        InMemCsr {
            shards: vec![s0, s1],
            n_obs: 5,
            n_vars: 4,
        }
    }

    fn csc_fixture() -> InMemCsc {
        // One nonzero per column, value = col index + 1.
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for c in 0..4usize {
            indices.push((c % 5) as i32);
            data.push((c + 1) as f32);
            indptr.push(indices.len() as i64);
        }
        let shard = ScxCsc::new_unchecked((5, 4), indptr, indices, data);
        InMemCsc {
            shards: vec![shard],
            ranges: vec![(0, 4)],
            n_obs: 5,
            n_vars: 4,
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn csr_only_reports_csr_and_rejects_csc() {
        let dev = require_gpu!();
        let csr = csr_fixture();
        let mut src = BackedGpuMatrixSource::new(&dev, &csr).unwrap();
        assert_eq!(src.available_layouts(), LayoutSet::CSR);
        assert!(!src.route_metadata().csc_available);
        assert_eq!(src.shape(), (5, 4));

        // CSC iteration on a CSR-only source errors (no panic).
        let err = src.for_each_gpu_csc_shard_in_range(0..4, &mut |_, _| Ok(()));
        assert!(matches!(err, Err(GpuError::UnsupportedLayout(_))));
    }

    /// The mirror of `csr_only_reports_csr_and_rejects_csc`, for the
    /// constructor this PR added.
    ///
    /// Missing until review (Cursor Agent - Grok 4.6 High): the HVG CSC reduces
    /// exercise `csc_only`'s happy path, but nothing pinned its layout bits or
    /// its rejection of the CSR iterator — the two properties that make it a
    /// `GpuMatrixSource` rather than a way to smuggle a CSC adapter through one.
    ///
    /// `shape()` is asserted because a `csc_only` source has no CSR adapter to
    /// ask, so it answers from `csc_shape`; getting that wrong yields an empty
    /// matrix rather than an error, and every downstream loop would quietly do
    /// nothing.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn csc_only_reports_csc_and_rejects_csr() {
        let dev = require_gpu!();
        let csc = csc_fixture();
        let mut src = BackedGpuMatrixSource::csc_only(&dev, &csc).unwrap();
        assert_eq!(src.available_layouts(), LayoutSet::CSC);
        assert!(src.route_metadata().csc_available);
        assert_eq!(
            src.shape(),
            (5, 4),
            "shape must come from the CSC source when there is no CSR adapter"
        );

        // CSR iteration on a CSC-only source errors (no panic).
        let err = src.for_each_gpu_csr_shard(&mut |_, _| Ok(()));
        assert!(matches!(err, Err(GpuError::UnsupportedLayout(_))));

        // ...and the CSC side still works, so the rejection above is about the
        // missing layout and not a source that is broken outright.
        let mut seen = 0usize;
        src.for_each_gpu_csc_shard_in_range(0..4, &mut |_, _| {
            seen += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, 1, "the one CSC shard must still be delivered");
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn with_csc_reports_both_layouts() {
        let dev = require_gpu!();
        let csr = csr_fixture();
        let csc = csc_fixture();
        let src = BackedGpuMatrixSource::with_csc(&dev, &csr, &csc).unwrap();
        assert_eq!(src.available_layouts(), LayoutSet::CSR | LayoutSet::CSC);
        assert!(src.route_metadata().csc_available);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn csr_iteration_yields_source_values() {
        let dev = require_gpu!();
        let csr = csr_fixture();
        let mut src = BackedGpuMatrixSource::new(&dev, &csr).unwrap();
        let mut seen = Vec::<f32>::new();
        src.for_each_gpu_csr_shard(&mut |_, slot| {
            let v = slot.view();
            let mut host = vec![0.0f32; v.data.len()];
            dev.stream()
                .memcpy_dtoh(&v.data, &mut host)
                .map_err(|e| GpuError::CudaError(format!("dtoh: {e}")))?;
            seen.extend(host);
            Ok(())
        })
        .unwrap();
        dev.synchronize().unwrap();
        // Concatenated nonzeros across both shards, in shard order.
        assert_eq!(seen, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn csc_iteration_yields_source_values() {
        let dev = require_gpu!();
        let csr = csr_fixture();
        let csc = csc_fixture();
        let mut src = BackedGpuMatrixSource::with_csc(&dev, &csr, &csc).unwrap();
        let mut seen = Vec::<f32>::new();
        src.for_each_gpu_csc_shard_in_range(0..4, &mut |_, view| {
            let mut host = vec![0.0f32; view.data.len()];
            dev.stream()
                .memcpy_dtoh(&view.data, &mut host)
                .map_err(|e| GpuError::CudaError(format!("dtoh: {e}")))?;
            seen.extend(host);
            Ok(())
        })
        .unwrap();
        dev.synchronize().unwrap();
        assert_eq!(seen, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
