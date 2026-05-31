//! [`GpuMatrixSource`] impl that applies `normalize_total` / `log1p` on device
//! before yielding CSR shards.
//!
//! Wraps [`GpuPreprocessedShardSource`]; CSR-only (no CSC preprocessing path
//! exists yet) and reports the applied transforms via
//! [`transforms`](GpuMatrixSource::transforms) so a downstream kernel can detect
//! a double-application mismatch.

use scx_format::ShardSource;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_matrix_source::{GpuMatrixSource, GpuTransformSpec, LayoutSet};
use crate::gpu_shard_source::{GpuPreprocessedShardSource, GpuShardSource};
use crate::staging::GpuCsrSlot;

/// A [`GpuMatrixSource`] that applies per-row normalization and/or `log1p` on
/// device. `CSR`-only.
pub struct PreprocessedGpuMatrixSource<'a> {
    inner: GpuPreprocessedShardSource<'a>,
    // Mirrored here because `GpuPreprocessedShardSource`'s fields are private;
    // surfaced via `transforms()`.
    normalize: Option<f32>,
    log1p: bool,
}

impl<'a> PreprocessedGpuMatrixSource<'a> {
    /// Construct a preprocessing source. `normalize = Some(target)` applies
    /// per-row `normalize_total` to `target`; `log1p` applies `log(1 + x)`
    /// after any normalization.
    pub fn new(
        dev: &'a GpuDevice,
        source: &'a (dyn ShardSource + Sync),
        normalize: Option<f32>,
        log1p: bool,
    ) -> Result<Self, GpuError> {
        Ok(Self {
            inner: GpuPreprocessedShardSource::new(dev, source, normalize, log1p)?,
            normalize,
            log1p,
        })
    }
}

impl GpuMatrixSource for PreprocessedGpuMatrixSource<'_> {
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    fn available_layouts(&self) -> LayoutSet {
        LayoutSet::CSR
    }

    fn transforms(&self) -> GpuTransformSpec {
        GpuTransformSpec {
            normalize: self.normalize,
            log1p: self.log1p,
            row_scale: false,
        }
    }

    fn for_each_gpu_csr_shard(
        &mut self,
        f: &mut dyn FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        self.inner.for_each_gpu_shard(|idx, slot| f(idx, slot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_sparse::ScxCsr;

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
        fn read_shard(&self, idx: usize) -> scx_format::Result<ScxCsr> {
            Ok(self.shards[idx].clone())
        }
    }

    #[test]
    fn reports_csr_layout_and_applied_transforms() {
        let dev = require_gpu!();
        let src = InMemCsr {
            shards: vec![ScxCsr::new_unchecked(
                (2, 3),
                vec![0, 1, 2],
                vec![0, 2],
                vec![10.0, 20.0],
            )],
            n_obs: 2,
            n_vars: 3,
        };
        let pp = PreprocessedGpuMatrixSource::new(&dev, &src, Some(1e4), true).unwrap();
        assert_eq!(pp.available_layouts(), LayoutSet::CSR);
        let t = pp.transforms();
        assert_eq!(t.normalize, Some(1e4));
        assert!(t.log1p);
        assert!(!t.row_scale);
        // CSC is unavailable on a preprocessing source.
        assert!(!pp.route_metadata().csc_available);
    }
}
