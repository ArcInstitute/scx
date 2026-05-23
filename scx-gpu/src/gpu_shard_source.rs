//! Device-resident shard sources for fully-GPU pipelines.
//!
//! [`GpuShardSource`] is the GPU counterpart of [`scx_format::ShardSource`]:
//! a sequence of GPU-resident CSR shards that consumers iterate over
//! without materialising the full matrix on the host.
//!
//! Compared to [`crate::shard_pipeline::DoubleBufferedShardLoader`], the
//! `GpuShardSource` abstraction:
//!
//! - Reuses the **same device CSR buffers** ([`crate::staging::GpuCsrSlot`])
//!   across all shards, avoiding the per-shard `dev.alloc_zeros` round-trip.
//! - Caches the cuSPARSE `CusparseSpMatDescr` on the slot, so iterative
//!   callers (PCA power iteration, Harmony correction) pay the descriptor
//!   build cost only when the slot's pointers or shape change.
//! - Surfaces a borrowed [`crate::staging::GpuCsrShardView`] callback so
//!   `view.indices.len()` returns the live shard's `nnz` rather than the
//!   slot's grow-only capacity.
//!
//! ## Variants
//!
//! - [`RawGpuShardSource`] — raw decoded CSR shards (no transforms).
//! - [`GpuPreprocessedShardSource`] — applies `normalize_total` and / or
//!   `log1p` to the `data` slot in place before the callback. Used by
//!   `pyscx.accel.normalize_total(device="gpu")` /
//!   `log1p(device="gpu")` and as the input contract for the future scVI
//!   device-resident dataloader (G12).
//!
//! ## Host-returning APIs
//!
//! [`crate::gpu_preprocess::gpu_preprocess_to_csr`] is implemented as a
//! terminal D→H copy on top of [`GpuPreprocessedShardSource`]: shards
//! are normalised / log1p'd on device, the transformed `(indptr, indices,
//! data)` triple is downloaded once per shard, and concatenation happens
//! on the host. The per-shard `dev.synchronize()` of the legacy
//! implementation is gone — `dtoh_copy` for pageable destinations already
//! enforces host-side ordering, and a single `dev.synchronize()` at the
//! API boundary remains as a defence-in-depth barrier.

use std::sync::Arc;

use cudarc::driver::safe::CudaStream;
use scx_format::ShardSource;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_preprocess::gpu_apply_fused_ops;
use crate::staging::{GpuCsrSlot, PinnedCsrSlot};

/// Sequence of GPU-resident CSR shards.
///
/// Implementors own a shared device CSR slot; the trait yields borrowed
/// [`GpuCsrShardView`]s over the slot's live shard. The callback's
/// `&GpuCsrShardView` is invalidated when the iterator advances to the
/// next shard (the slot is mutated in place).
pub trait GpuShardSource {
    /// Number of shards in this source (0 when empty).
    fn n_shards(&self) -> usize;

    /// Total observation count `n_obs` across all shards.
    fn n_obs(&self) -> usize;

    /// Variable count `n_vars` (constant across shards).
    fn n_vars(&self) -> usize;

    /// `(n_obs, n_vars)`.
    fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    /// Iterate over each non-empty shard, invoking the callback with
    /// `&mut GpuCsrSlot` (the loader's reusable device slot positioned
    /// at the live shard).
    ///
    /// The callback typically constructs a view via [`GpuCsrSlot::view`]
    /// for kernel arguments, or asks for the cached cuSPARSE descriptor
    /// via [`GpuCsrSlot::cached_sp_descr`]. The two are not
    /// simultaneously addressable through this `&mut` borrow — request
    /// the descriptor first (its lifetime ends with the cuSPARSE call),
    /// then construct the view. See `tests` for examples.
    fn for_each_gpu_shard<F>(&mut self, f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>;
}

// --------------------------------------------------------------------------
// RawGpuShardSource
// --------------------------------------------------------------------------

/// Adapter from a CPU-side [`ShardSource`] to [`GpuShardSource`].
///
/// Owns the staging pipeline:
/// - One reusable [`PinnedCsrSlot`] (host-side staging buffer).
/// - One reusable [`GpuCsrSlot`] (device CSR + cached descriptor).
/// - A dedicated `copy_stream` and per-shard upload events.
///
/// The decode loop runs on a scoped worker thread that pre-decodes the
/// next shard while the main thread processes the current one. The slot
/// is reused — only the live shard's bytes are uploaded.
pub struct RawGpuShardSource<'a> {
    dev: &'a GpuDevice,
    source: &'a (dyn ShardSource + Sync),
    pinned: PinnedCsrSlot,
    slot: GpuCsrSlot,
    copy_stream: Arc<CudaStream>,
}

impl<'a> RawGpuShardSource<'a> {
    /// Construct a raw GPU shard source. Sizes the staging buffers from
    /// the catalog's `max_shard_rows` when available; otherwise grows
    /// lazily on first upload.
    pub fn new(dev: &'a GpuDevice, source: &'a (dyn ShardSource + Sync)) -> Result<Self, GpuError> {
        // Lazy-grow staging slots — we deliberately do not call
        // `max_shard_rows()` here. The default `ShardSource` impl reads
        // every shard, which would defeat the worker-thread pipelining
        // and inflate I/O on backed readers without an O(1) override.
        // Backed readers with the O(1) override and callers that know
        // their max can pre-size manually if profiling shows the lazy
        // first-grow cost dominates.
        let pinned = PinnedCsrSlot::new(dev.context(), 1, 1);
        let slot = GpuCsrSlot::new(dev, 1, 1)?;

        // Dedicated copy stream — used only when n_shards > 1 (single-
        // shard sources reuse the compute stream below).
        let copy_stream = if source.n_shards() > 1 {
            dev.context()
                .new_stream()
                .map_err(|e| GpuError::StreamError(format!("new_stream: {e}")))?
        } else {
            dev.stream().clone()
        };

        Ok(Self {
            dev,
            source,
            pinned,
            slot,
            copy_stream,
        })
    }

    /// Run `f` over each non-empty shard. Internal driver shared between
    /// the raw and preprocessed sources — the `transform` closure is
    /// invoked AFTER the upload event handshake but BEFORE the user
    /// callback, giving preprocessing variants a hook to modify the
    /// slot's `data` in place.
    fn run<F, T>(&mut self, mut f: F, mut transform: T) -> Result<(), GpuError>
    where
        F: FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>,
        T: FnMut(&GpuDevice, &mut GpuCsrSlot) -> Result<(), GpuError>,
    {
        let n_shards = self.source.n_shards();
        if n_shards == 0 {
            return Ok(());
        }

        // Single-shard fast path (no worker thread, no extra copy stream).
        if n_shards == 1 {
            let csr = self
                .source
                .read_shard(0)
                .map_err(|e| GpuError::InvalidShard(format!("shard 0: {e}")))?;
            if csr.n_rows() == 0 {
                return Ok(());
            }
            self.pinned.stage(&csr)?;
            self.pinned.upload_to(
                self.dev.stream(),
                &mut self.slot,
                csr.n_rows(),
                csr.data.len(),
                csr.n_cols(),
            )?;
            transform(self.dev, &mut self.slot)?;
            f(0, &mut self.slot)?;
            return Ok(());
        }

        // Multi-shard: scoped worker decodes one shard ahead of main.
        // Borrow split: hoist references to inner fields up-front so the
        // scoped thread closure can capture them without going through
        // `&mut self`.
        let source = self.source;
        let dev = self.dev;
        let pinned = &mut self.pinned;
        let slot = &mut self.slot;
        let copy_stream = &self.copy_stream;
        let compute_stream = dev.stream();

        std::thread::scope(|scope| -> Result<(), GpuError> {
            use std::sync::mpsc;
            type Msg = Result<(usize, scx_sparse::ScxCsr), scx_format::ScxError>;
            let (tx, rx) = mpsc::sync_channel::<Msg>(1);

            scope.spawn(move || {
                for i in 0..n_shards {
                    let out = source.read_shard(i).map(|c| (i, c));
                    if tx.send(out).is_err() {
                        break;
                    }
                }
            });

            while let Ok(msg) = rx.recv() {
                let (i, csr) = match msg {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(GpuError::InvalidShard(format!(
                            "SCX read error during streaming decode: {e}"
                        )))
                    }
                };
                if csr.n_rows() == 0 {
                    continue;
                }
                pinned.stage(&csr)?;
                pinned.upload_to(
                    copy_stream,
                    slot,
                    csr.n_rows(),
                    csr.data.len(),
                    csr.n_cols(),
                )?;
                // Event handshake: upload on copy_stream, kernel reads
                // on compute_stream. The wait guarantees the kernel
                // observes the just-uploaded shard data.
                let upload_event = copy_stream
                    .record_event(None)
                    .map_err(|e| GpuError::CudaError(format!("record event: {e}")))?;
                compute_stream
                    .wait(&upload_event)
                    .map_err(|e| GpuError::CudaError(format!("compute wait: {e}")))?;

                transform(dev, slot)?;

                f(i, slot)?;

                // Record compute-stream completion so the next shard's
                // upload (on copy_stream) waits for any reads of this
                // slot's buffers to finish before re-staging into them.
                let compute_event = compute_stream
                    .record_event(None)
                    .map_err(|e| GpuError::CudaError(format!("record event: {e}")))?;
                copy_stream
                    .wait(&compute_event)
                    .map_err(|e| GpuError::CudaError(format!("copy wait: {e}")))?;
            }
            Ok(())
        })
    }
}

impl<'a> GpuShardSource for RawGpuShardSource<'a> {
    fn n_shards(&self) -> usize {
        self.source.n_shards()
    }

    fn n_obs(&self) -> usize {
        self.source.n_obs()
    }

    fn n_vars(&self) -> usize {
        self.source.n_vars()
    }

    fn for_each_gpu_shard<F>(&mut self, f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>,
    {
        self.run(f, |_dev, _slot| Ok(()))
    }
}

// --------------------------------------------------------------------------
// GpuPreprocessedShardSource
// --------------------------------------------------------------------------

/// `GpuShardSource` that applies in-place `normalize_total` and / or
/// `log1p` to each shard's `data` buffer on device before yielding the
/// view to the consumer.
///
/// The transforms run on the compute stream (after the upload-event
/// handshake) and mutate the slot's `data` buffer in place. Consumers
/// that need to read the transformed values back to the host should issue
/// a `memcpy_dtoh` on the compute stream — it is ordered after the
/// transform kernel.
///
/// This is the input contract for [`crate::gpu_preprocess::gpu_preprocess_to_csr`]
/// and, in future, the scVI device-resident dataloader (G12).
pub struct GpuPreprocessedShardSource<'a> {
    inner: RawGpuShardSource<'a>,
    normalize: Option<f32>,
    log1p: bool,
}

impl<'a> GpuPreprocessedShardSource<'a> {
    /// Construct a preprocessing source. Pass `normalize = Some(target)`
    /// to apply per-row normalization to `target` total counts; `log1p`
    /// applies `log(1 + x)` after any normalization. Pass `None` /
    /// `false` to skip a transform; passing both `None` and `false`
    /// reduces to a [`RawGpuShardSource`].
    pub fn new(
        dev: &'a GpuDevice,
        source: &'a (dyn ShardSource + Sync),
        normalize: Option<f32>,
        log1p: bool,
    ) -> Result<Self, GpuError> {
        Ok(Self {
            inner: RawGpuShardSource::new(dev, source)?,
            normalize,
            log1p,
        })
    }
}

impl<'a> GpuShardSource for GpuPreprocessedShardSource<'a> {
    fn n_shards(&self) -> usize {
        self.inner.n_shards()
    }

    fn n_obs(&self) -> usize {
        self.inner.n_obs()
    }

    fn n_vars(&self) -> usize {
        self.inner.n_vars()
    }

    fn for_each_gpu_shard<F>(&mut self, f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>,
    {
        let normalize = self.normalize;
        let log1p = self.log1p;
        self.inner.run(f, move |dev, slot| {
            if normalize.is_none() && !log1p {
                return Ok(());
            }
            let n_rows = slot.shape().0;
            // Field-level split borrow: indptr (immutable) + data (mutable)
            // touch disjoint fields, which Rust permits through the
            // dedicated `split_indptr_data_mut` accessor.
            let (indptr, mut data) = slot.split_indptr_data_mut();
            apply_fused_ops_to_views(dev, &indptr, &mut data, n_rows, normalize, log1p)
        })
    }
}

/// View-based wrapper around the CSR-level preprocessing kernels
/// (`normalize`, `log1p`, fused). Mirrors
/// [`crate::gpu_preprocess::gpu_apply_fused_ops`] but operates on
/// [`cudarc::driver::safe::CudaView`] / [`cudarc::driver::safe::CudaViewMut`]
/// so the slot's grow-only buffers can be transformed in place at the
/// live shard size.
fn apply_fused_ops_to_views(
    dev: &GpuDevice,
    indptr: &cudarc::driver::safe::CudaView<'_, i64>,
    data: &mut cudarc::driver::safe::CudaViewMut<'_, f32>,
    n_rows: usize,
    normalize: Option<f32>,
    log1p: bool,
) -> Result<(), GpuError> {
    // The kernels in `crate::gpu_preprocess` were written against
    // `&CudaSlice<T>`. The PushKernelArg trait is also implemented for
    // `&CudaView` / `&mut CudaViewMut`, so we can launch the same
    // kernels with view arguments directly — this helper just calls them
    // through.
    use cudarc::driver::safe::LaunchConfig;
    use cudarc::driver::PushKernelArg;

    if n_rows == 0 {
        return Ok(());
    }

    const NORMALIZE_LOG1P_PTX: &str =
        include_str!(concat!(env!("OUT_DIR"), "/normalize_log1p.ptx"));
    let module = dev.load_module_cached(NORMALIZE_LOG1P_PTX)?;

    let n_rows_i32 = n_rows as i32;
    let threads: u32 = 256;
    let blocks = (n_rows as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let _ = gpu_apply_fused_ops; // anchor for IDE jump-to-def
    match (normalize, log1p) {
        (Some(target_sum), true) => {
            let func = module
                .load_function("normalize_log1p_kernel")
                .map_err(|e| GpuError::KernelLaunchFailed(format!("normalize_log1p: {e}")))?;
            unsafe {
                dev.stream()
                    .launch_builder(&func)
                    .arg(indptr)
                    .arg(data)
                    .arg(&n_rows_i32)
                    .arg(&target_sum)
                    .launch(cfg)
            }
            .map_err(|e| GpuError::KernelLaunchFailed(format!("normalize_log1p: {e}")))?;
        }
        (Some(target_sum), false) => {
            let func = module
                .load_function("normalize_kernel")
                .map_err(|e| GpuError::KernelLaunchFailed(format!("normalize: {e}")))?;
            unsafe {
                dev.stream()
                    .launch_builder(&func)
                    .arg(indptr)
                    .arg(data)
                    .arg(&n_rows_i32)
                    .arg(&target_sum)
                    .launch(cfg)
            }
            .map_err(|e| GpuError::KernelLaunchFailed(format!("normalize: {e}")))?;
        }
        (None, true) => {
            let func = module
                .load_function("log1p_kernel")
                .map_err(|e| GpuError::KernelLaunchFailed(format!("log1p: {e}")))?;
            unsafe {
                dev.stream()
                    .launch_builder(&func)
                    .arg(indptr)
                    .arg(data)
                    .arg(&n_rows_i32)
                    .launch(cfg)
            }
            .map_err(|e| GpuError::KernelLaunchFailed(format!("log1p: {e}")))?;
        }
        (None, false) => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_sparse::ScxCsr;

    struct InMemorySource {
        shards: Vec<ScxCsr>,
        n_obs: usize,
        n_vars: usize,
    }

    impl ShardSource for InMemorySource {
        fn n_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_shard(&self, shard_idx: usize) -> scx_format::Result<ScxCsr> {
            Ok(self.shards[shard_idx].clone())
        }
    }

    fn make_csr(rows: usize, n_vars: usize, base: f32) -> ScxCsr {
        // One nonzero per row at column (row % n_vars), value (base + row).
        let mut indptr = vec![0i64];
        let mut indices = Vec::with_capacity(rows);
        let mut data = Vec::with_capacity(rows);
        for r in 0..rows {
            indices.push((r % n_vars) as i32);
            data.push(base + r as f32);
            indptr.push(indices.len() as i64);
        }
        ScxCsr::new_unchecked((rows, n_vars), indptr, indices, data)
    }

    /// Raw source yields the staged shard verbatim (slot.view returns
    /// the uploaded data).
    #[test]
    fn test_raw_source_round_trip() {
        let dev = require_gpu!();
        let shards = vec![make_csr(3, 5, 1.0), make_csr(4, 5, 100.0)];
        let src = InMemorySource {
            shards: shards.clone(),
            n_obs: 7,
            n_vars: 5,
        };
        let mut gpu = RawGpuShardSource::new(&dev, &src).unwrap();

        let mut seen: Vec<f32> = Vec::new();
        gpu.for_each_gpu_shard(|_idx, slot| {
            let view = slot.view();
            let mut host = vec![0.0f32; view.data.len()];
            dev.stream()
                .memcpy_dtoh(&view.data, &mut host)
                .map_err(|e| GpuError::CudaError(format!("dtoh: {e}")))?;
            seen.extend(host);
            Ok(())
        })
        .unwrap();
        dev.synchronize().unwrap();

        // Shard 0 = [1, 2, 3]; shard 1 = [100, 101, 102, 103].
        assert_eq!(seen, vec![1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0]);
    }

    /// `GpuPreprocessedShardSource` applies normalize+log1p in place;
    /// reading the view's `data` back yields the transformed values.
    #[test]
    fn test_preprocessed_source_normalize_log1p() {
        let dev = require_gpu!();
        // Tiny 2-row CSR: row 0 = [5, 5] (sum 10), row 1 = [1, 4] (sum 5).
        let csr = ScxCsr::new_unchecked(
            (2, 3),
            vec![0i64, 2, 4],
            vec![0i32, 1, 0, 2],
            vec![5.0f32, 5.0, 1.0, 4.0],
        );
        let src = InMemorySource {
            shards: vec![csr],
            n_obs: 2,
            n_vars: 3,
        };
        let mut gpu = GpuPreprocessedShardSource::new(&dev, &src, Some(10.0f32), true).unwrap();

        let mut seen: Vec<f32> = Vec::new();
        gpu.for_each_gpu_shard(|_idx, slot| {
            let view = slot.view();
            let mut host = vec![0.0f32; view.data.len()];
            dev.stream()
                .memcpy_dtoh(&view.data, &mut host)
                .map_err(|e| GpuError::CudaError(format!("dtoh: {e}")))?;
            seen.extend(host);
            Ok(())
        })
        .unwrap();
        dev.synchronize().unwrap();

        // Expected (CPU reference, f64 intermediate then ln_1p):
        // Row 0: 5/10*10 = 5  → ln(6)  ≈ 1.7917595
        //        5/10*10 = 5  → ln(6)  ≈ 1.7917595
        // Row 1: 1/5*10 = 2   → ln(3)  ≈ 1.0986123
        //        4/5*10 = 8   → ln(9)  ≈ 2.1972246
        let expected: Vec<f32> = vec![(6.0f32).ln(), (6.0f32).ln(), (3.0f32).ln(), (9.0f32).ln()];
        assert_eq!(seen.len(), expected.len());
        for (g, c) in seen.iter().zip(expected.iter()) {
            assert!(
                (g - c).abs() < 1e-5,
                "transformed value mismatch: got {g}, expected {c}"
            );
        }
    }

    /// Cached cuSPARSE descriptor is reused across power-iteration-style
    /// repeated callbacks on the same shard. Each shard yields one
    /// descriptor build; subsequent in-loop accesses to
    /// `slot.cached_sp_descr` reuse it.
    #[test]
    fn test_cached_descr_reused_per_shard() {
        let dev = require_gpu!();
        let shards = vec![make_csr(4, 6, 1.0), make_csr(4, 6, 1.0)];
        let src = InMemorySource {
            shards,
            n_obs: 8,
            n_vars: 6,
        };
        let mut gpu = RawGpuShardSource::new(&dev, &src).unwrap();

        let mut total_descr_addrs: Vec<u64> = Vec::new();
        gpu.for_each_gpu_shard(|_idx, slot| {
            // Multiple accesses within one shard — second access must
            // hit the cache.
            let p1 = slot.cached_sp_descr(&dev, dev.stream())?.raw();
            let p2 = slot.cached_sp_descr(&dev, dev.stream())?.raw();
            assert_eq!(
                p1 as usize, p2 as usize,
                "within-shard repeated cached_sp_descr calls must reuse the descriptor"
            );
            total_descr_addrs.push(p1 as u64);
            Ok(())
        })
        .unwrap();
        dev.synchronize().unwrap();

        assert_eq!(total_descr_addrs.len(), 2, "two shards → two descriptors");
        // Across shards, the cache is invalidated (different shape /
        // different uploaded contents), so addresses may differ — we
        // don't assert that.
    }
}
