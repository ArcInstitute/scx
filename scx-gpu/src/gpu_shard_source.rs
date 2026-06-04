//! Device-resident shard sources for fully-GPU pipelines.
//!
//! [`GpuShardSource`] is the GPU counterpart of [`scx_format::ShardSource`]:
//! a sequence of GPU-resident CSR shards that consumers iterate over
//! without materialising the full matrix on the host.
//!
//! Key properties of the `GpuShardSource` abstraction:
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

use cudarc::driver::safe::{CudaEvent, CudaSlice, CudaStream};
use scx_format::ShardSource;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_preprocess::apply_fused_ops_inner;
use crate::staging::{GpuCsrSlot, PinnedCsrSlot};

/// Release-active validation of a CSR shard at the host-side GPU DE staging
/// boundary. Returns [`GpuError::InvalidShard`] rather than relying on a
/// `debug_assert!` that vanishes in release builds (findings ACC3 + ACC11 /
/// the always-on-boundary-validation policy).
///
/// Two invariants the GPU DE kernels require but cannot themselves enforce:
///
/// 1. **Strictly-increasing per-row columns (ACC3).** The CSR-to-dense scatter
///    (`csr_shard_to_dense_chunk_kernel`) writes `dense[row, col - c0] =
///    data[e]` with one thread per nonzero, so a duplicate `(row, col)` pair
///    races on the same output cell and the winning value is nondeterministic.
///    SCX canonicalisation sorts but does not dedup columns.
///
/// 2. **Finite values (ACC11).** `block_radix_sort_per_gene_kernel` pads with
///    `+INF` and sorts on the raw IEEE-754 bit pattern, so a NaN lands at the
///    wrong position and corrupts the U statistic and tie counts.
///
/// O(nnz) — negligible beside the H2D copy and the per-gene device sort. Run
/// once per shard for every GPU DE consumer (the `for_each_gpu_shard` driver
/// calls it before staging).
fn validate_shard_for_gpu_de(csr: &scx_sparse::ScxCsr) -> Result<(), GpuError> {
    for r in 0..csr.n_rows() {
        let s = csr.indptr[r] as usize;
        let e = csr.indptr[r + 1] as usize;
        for w in csr.indices[s..e].windows(2) {
            if w[0] >= w[1] {
                return Err(GpuError::InvalidShard(format!(
                    "ScxCsr row {r} has unsorted or duplicate column indices ({} >= {}): \
                     GPU shard scatter requires strictly-increasing per-row indices for \
                     deterministic output",
                    w[0], w[1]
                )));
            }
        }
    }
    if let Some(pos) = csr.data.iter().position(|v| !v.is_finite()) {
        return Err(GpuError::InvalidShard(format!(
            "ScxCsr contains a non-finite value ({}) at nonzero index {pos}: GPU DE ranking \
             requires finite input (NaN corrupts the radix sort; sanitise/QC before DE)",
            csr.data[pos]
        )));
    }
    Ok(())
}

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
/// - A **2-slot ring** of [`PinnedCsrSlot`]s (host-side staging buffers).
///   Two are required to fix a host-side reuse race that the previous
///   single-slot design exhibited: with pinned memory the H→D copy issued
///   on `copy_stream` is truly asynchronous, so the next iteration's
///   `stage()` would otherwise overwrite the pinned source buffer while
///   the prior shard's DMA was still in flight. The ring pings between
///   the two slots; before re-staging, the loop host-waits on the
///   captured copy-stream event for that slot.
/// - One reusable [`GpuCsrSlot`] (device CSR + cached descriptor).
/// - A dedicated `copy_stream` and per-shard upload events for the
///   device-side handshake (compute waits on copy; next copy waits on
///   compute reads of the prior shard's device slot).
///
/// The decode loop runs on a scoped worker thread that pre-decodes the
/// next shard while the main thread processes the current one.
pub struct RawGpuShardSource<'a> {
    dev: &'a GpuDevice,
    source: &'a (dyn ShardSource + Sync),
    pinned: [PinnedCsrSlot; 2],
    /// Per-pinned-slot copy-stream events captured immediately after the
    /// slot's most recent `upload_to`. Host-waited on before the slot is
    /// reused for the next `stage()` so the CPU never overwrites a buffer
    /// whose DMA is still in flight.
    pinned_events: [Option<CudaEvent>; 2],
    slot: GpuCsrSlot,
    copy_stream: Arc<CudaStream>,
}

impl<'a> RawGpuShardSource<'a> {
    /// Construct a raw GPU shard source with lazy-grow staging buffers.
    ///
    /// Deliberately does **not** call `ShardSource::max_shard_rows()`
    /// to pre-size: the default trait impl reads every shard, which would
    /// defeat the worker-thread pipelining and inflate I/O on backed
    /// readers without an O(1) override. Backed readers with the O(1)
    /// override and callers that know their max should use
    /// [`Self::with_max_shard_rows`] instead.
    pub fn new(dev: &'a GpuDevice, source: &'a (dyn ShardSource + Sync)) -> Result<Self, GpuError> {
        Self::build(dev, source, 1, 1)
    }

    /// Construct a raw GPU shard source with pinned and device buffers
    /// pre-sized for the largest expected shard.
    ///
    /// `max_rows` is the maximum `n_rows` across shards, `max_nnz` is the
    /// maximum `nnz`. The slots will grow on demand if these are
    /// underestimates; over-estimates only cost extra pinned host memory
    /// and one-time device alloc. Use catalog stats
    /// (`FullCatalog::max_shard_rows`, `ShardStats::nnz`) when available.
    pub fn with_max_shard_rows(
        dev: &'a GpuDevice,
        source: &'a (dyn ShardSource + Sync),
        max_rows: usize,
        max_nnz: usize,
    ) -> Result<Self, GpuError> {
        Self::build(
            dev,
            source,
            max_rows.saturating_add(1).max(1),
            max_nnz.max(1),
        )
    }

    fn build(
        dev: &'a GpuDevice,
        source: &'a (dyn ShardSource + Sync),
        indptr_cap: usize,
        nnz_cap: usize,
    ) -> Result<Self, GpuError> {
        let pinned = [
            PinnedCsrSlot::new(dev.context(), indptr_cap, nnz_cap),
            PinnedCsrSlot::new(dev.context(), indptr_cap, nnz_cap),
        ];
        let slot = GpuCsrSlot::new(dev, indptr_cap, nnz_cap)?;

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
            pinned_events: [None, None],
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

        // Single-shard fast path (no worker thread, no extra copy stream,
        // no pinned-ring rotation). Safe because there is no successor
        // `stage()` that could race the in-flight DMA — once `f` returns,
        // the caller's next host action implicitly orders against the
        // compute stream and the pinned buffer is free to reuse.
        if n_shards == 1 {
            let csr = self
                .source
                .read_shard(0)
                .map_err(|e| GpuError::InvalidShard(format!("shard 0: {e}")))?;
            if csr.n_rows() == 0 {
                return Ok(());
            }
            validate_shard_for_gpu_de(&csr)?;
            self.pinned[0].stage(&csr)?;
            self.pinned[0].upload_to(
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
        let pinned_events = &mut self.pinned_events;
        let slot = &mut self.slot;
        let copy_stream = &self.copy_stream;
        let compute_stream = dev.stream();

        let scope_result = std::thread::scope(|scope| -> Result<(), GpuError> {
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

            let mut pinned_idx: usize = 0;
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

                // Host-side gate: if this pinned slot still has an
                // outstanding copy-stream event from a previous shard, we
                // must wait for that DMA to drain on the host before
                // overwriting the pinned buffer. The device-side gates
                // below only order device streams against each other;
                // they do not prevent the CPU from racing the DMA's
                // source memory.
                if let Some(evt) = pinned_events[pinned_idx].take() {
                    evt.synchronize()
                        .map_err(|e| GpuError::CudaError(format!("pinned event sync: {e}")))?;
                }

                validate_shard_for_gpu_de(&csr)?;
                pinned[pinned_idx].stage(&csr)?;
                pinned[pinned_idx].upload_to(
                    copy_stream,
                    slot,
                    csr.n_rows(),
                    csr.data.len(),
                    csr.n_cols(),
                )?;
                // Device-side gate (copy → compute): kernel reads must
                // observe the just-uploaded shard data.
                let upload_event = copy_stream
                    .record_event(None)
                    .map_err(|e| GpuError::CudaError(format!("record upload event: {e}")))?;
                compute_stream
                    .wait(&upload_event)
                    .map_err(|e| GpuError::CudaError(format!("compute wait: {e}")))?;
                // Stash the same event against this pinned slot so the
                // next iteration that recycles it can host-wait.
                pinned_events[pinned_idx] = Some(upload_event);

                transform(dev, slot)?;
                f(i, slot)?;

                // Device-side gate (compute → copy): the next shard's
                // upload (which writes into `slot`) must wait for the
                // current shard's compute reads of `slot` to finish.
                let compute_event = compute_stream
                    .record_event(None)
                    .map_err(|e| GpuError::CudaError(format!("record compute event: {e}")))?;
                copy_stream
                    .wait(&compute_event)
                    .map_err(|e| GpuError::CudaError(format!("copy wait: {e}")))?;

                pinned_idx ^= 1;
            }
            Ok(())
        });

        // Drain any remaining pinned events so the caller may safely
        // mutate or drop the pinned host buffers immediately after this
        // function returns. Cheap in the common case — by the time we
        // reach this point the DMAs are typically already complete.
        for evt in self.pinned_events.iter_mut() {
            if let Some(e) = evt.take() {
                e.synchronize()
                    .map_err(|err| GpuError::CudaError(format!("pinned drain sync: {err}")))?;
            }
        }

        scope_result
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
    /// Per-row scale factors uploaded once to device, indexed in iteration row
    /// order (length == `source.n_obs()`). `None` = no row-scale transform.
    row_scale: Option<CudaSlice<f32>>,
}

impl<'a> GpuPreprocessedShardSource<'a> {
    /// Construct a preprocessing source. Pass `normalize = Some(target)`
    /// to apply per-row normalization to `target` total counts; `log1p`
    /// applies `log(1 + x)` after any normalization; `row_scale = Some(factors)`
    /// multiplies each row by an explicit factor (applied last). `factors` is a
    /// per-row vector in iteration row order; its length must equal the
    /// source's `n_obs`. Pass `None` / `false` to skip a transform; passing
    /// all of `None` / `false` / `None` reduces to a [`RawGpuShardSource`].
    pub fn new(
        dev: &'a GpuDevice,
        source: &'a (dyn ShardSource + Sync),
        normalize: Option<f32>,
        log1p: bool,
        row_scale: Option<&[f32]>,
    ) -> Result<Self, GpuError> {
        let row_scale = match row_scale {
            Some(factors) => {
                let n_obs = source.n_obs();
                if factors.len() != n_obs {
                    return Err(GpuError::InvalidShard(format!(
                        "row_scale factor length {} != source n_obs {n_obs}",
                        factors.len()
                    )));
                }
                Some(dev.htod_copy(factors)?)
            }
            None => None,
        };
        Ok(Self {
            inner: RawGpuShardSource::new(dev, source)?,
            normalize,
            log1p,
            row_scale,
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
        // Disjoint-field borrow: `&self.row_scale` and `&mut self.inner` touch
        // different fields, so binding it before `self.inner.run` is allowed.
        let row_scale = self.row_scale.as_ref();
        // Cumulative first-global-row of the current shard, advanced per shard.
        // `run` invokes the transform only for non-empty shards in 0..n_shards
        // order, and empty shards contribute 0 rows — so this matches the
        // global row layout the factor vector is indexed against.
        let mut global_row_offset = 0usize;
        self.inner.run(f, move |dev, slot| {
            let n_rows = slot.shape().0;
            let result = if normalize.is_none() && !log1p && row_scale.is_none() {
                Ok(())
            } else {
                // Field-level split borrow: indptr (immutable) + data (mutable)
                // touch disjoint fields, which Rust permits through the
                // dedicated `split_indptr_data_mut` accessor.
                let (indptr, mut data) = slot.split_indptr_data_mut();
                let rs = row_scale.map(|fs| (fs.slice(..), global_row_offset));
                apply_fused_ops_inner(
                    dev,
                    &indptr,
                    &mut data,
                    n_rows,
                    normalize,
                    log1p,
                    rs.as_ref().map(|(v, off)| (v, *off)),
                )
            };
            global_row_offset += n_rows;
            result
        })
    }
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

    /// ACC3: a row with non-increasing (here duplicate) column indices is
    /// rejected with `InvalidShard` — always-on, not a debug_assert. Pure
    /// CPU; needs no GPU device.
    #[test]
    fn validate_rejects_duplicate_columns() {
        // 1 row, columns [1, 1] — a duplicate the GPU scatter would race on.
        let csr = ScxCsr::new_unchecked((1, 4), vec![0i64, 2], vec![1i32, 1], vec![1.0f32, 2.0]);
        let err = validate_shard_for_gpu_de(&csr).unwrap_err();
        assert!(
            matches!(err, GpuError::InvalidShard(_)),
            "expected InvalidShard, got {err:?}"
        );
    }

    /// ACC11: a non-finite value is rejected with `InvalidShard` — always-on,
    /// not a debug_assert. Pure CPU; needs no GPU device.
    #[test]
    fn validate_rejects_non_finite_values() {
        let csr =
            ScxCsr::new_unchecked((1, 4), vec![0i64, 2], vec![0i32, 2], vec![1.0f32, f32::NAN]);
        let err = validate_shard_for_gpu_de(&csr).unwrap_err();
        assert!(
            matches!(err, GpuError::InvalidShard(_)),
            "expected InvalidShard, got {err:?}"
        );
    }

    #[test]
    fn validate_accepts_clean_shard() {
        let csr = make_csr(8, 4, 1.0);
        assert!(validate_shard_for_gpu_de(&csr).is_ok());
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
        let mut gpu =
            GpuPreprocessedShardSource::new(&dev, &src, Some(10.0f32), true, None).unwrap();

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

    /// Cached cuSPARSE descriptor is reused **across** shards when the
    /// shape (`n_rows`, `n_cols`, `nnz`) is identical and the slot's
    /// device buffers haven't grown. Complements
    /// [`test_cached_descr_reused_per_shard`] which only checks within-
    /// shard reuse.
    #[test]
    fn test_cross_shard_descr_reuse_same_shape() {
        let dev = require_gpu!();
        // Two identically-shaped shards (same n_rows, same nnz). The
        // first shard's stage() grows the slot from (1,1) once; the
        // second shard fits in the same capacity so no further grow
        // happens and the descriptor (built on shard 0) stays valid.
        let shards = vec![make_csr(4, 6, 1.0), make_csr(4, 6, 100.0)];
        let src = InMemorySource {
            shards,
            n_obs: 8,
            n_vars: 6,
        };
        let mut gpu = RawGpuShardSource::new(&dev, &src).unwrap();

        let mut descr_addrs: Vec<usize> = Vec::new();
        gpu.for_each_gpu_shard(|_idx, slot| {
            let p = slot.cached_sp_descr(&dev, dev.stream())?.raw() as usize;
            descr_addrs.push(p);
            Ok(())
        })
        .unwrap();
        dev.synchronize().unwrap();

        assert_eq!(descr_addrs.len(), 2);
        // Same shape across both shards → same descriptor object. The
        // contents of the device buffers differ between shards, but
        // cuSPARSE re-reads through the captured pointers on each SpMM
        // call, so the descriptor remains semantically valid.
        assert_eq!(
            descr_addrs[0], descr_addrs[1],
            "cross-shard cached_sp_descr must reuse the descriptor when shape is unchanged"
        );
    }

    /// Regression test for the pinned-host-reuse race fixed by the
    /// 2-slot pinned ring + host-side event sync.
    ///
    /// Pre-fix, `pinned.stage(&csr)` for shard `i+1` could overwrite the
    /// pinned source buffer while shard `i`'s `memcpy_htod_async` was
    /// still in flight on `copy_stream`. The device-side event handshake
    /// only orders streams against each other — it doesn't host-block
    /// the CPU writer. With pinned memory the H→D copy is truly async,
    /// so the corruption is observable.
    ///
    /// The test stresses the bug by:
    ///   - Using ≥3 shards so the ring cycles at least once.
    ///   - Sizing each shard ~10⁴ nnz so the DMA isn't trivially short.
    ///   - Using a **device-only** consumer (per-shard `memcpy_dtod` into
    ///     a private capture buffer on the compute stream) — no
    ///     per-callback host-blocking dtoh that would mask the race.
    ///   - Repeating across a small outer loop to amplify the race window.
    #[test]
    fn test_multi_shard_pinned_no_corruption() {
        use cudarc::driver::safe::CudaSlice;

        let dev = require_gpu!();
        let n_vars = 32usize;
        let rows_per_shard = 10_000usize; // ~10k rows × 1 nnz/row per shard
        let n_shards = 5usize;

        // Build per-shard CSRs with a per-shard tag value: shard `s`'s
        // data is `[s*1e6 + row]`. That makes any cross-shard bleed
        // numerically obvious.
        let make_tagged_csr = |shard: usize| -> ScxCsr {
            let mut indptr = Vec::with_capacity(rows_per_shard + 1);
            let mut indices = Vec::with_capacity(rows_per_shard);
            let mut data = Vec::with_capacity(rows_per_shard);
            indptr.push(0i64);
            for r in 0..rows_per_shard {
                indices.push((r % n_vars) as i32);
                data.push(shard as f32 * 1.0e6 + r as f32);
                indptr.push(indices.len() as i64);
            }
            ScxCsr::new_unchecked((rows_per_shard, n_vars), indptr, indices, data)
        };

        for repeat in 0..5 {
            let shards: Vec<ScxCsr> = (0..n_shards).map(make_tagged_csr).collect();
            let src = InMemorySource {
                shards: shards.clone(),
                n_obs: n_shards * rows_per_shard,
                n_vars,
            };
            let mut gpu = RawGpuShardSource::new(&dev, &src).unwrap();

            // Per-shard device capture buffers. Filled via dtod on the
            // compute stream inside each callback — strictly device-side,
            // so no host-blocking mask of the race.
            let mut captures: Vec<CudaSlice<f32>> = (0..n_shards)
                .map(|_| dev.alloc_zeros::<f32>(rows_per_shard).unwrap())
                .collect();

            gpu.for_each_gpu_shard(|idx, slot| {
                let view = slot.view();
                assert_eq!(view.data.len(), rows_per_shard, "shard {idx} nnz mismatch");
                dev.stream()
                    .memcpy_dtod(&view.data, &mut captures[idx])
                    .map_err(|e| GpuError::CudaError(format!("dtod: {e}")))?;
                Ok(())
            })
            .unwrap();
            // Single boundary sync — everything queued on compute_stream
            // (the per-shard dtod copies) must complete before we read
            // the captures back to the host.
            dev.synchronize().unwrap();

            // Dtoh each capture and verify against the per-shard tagged
            // values. Any cross-shard pinned-buffer corruption would
            // produce values from a neighbouring shard.
            for (idx, capture) in captures.iter().enumerate() {
                let mut host = vec![0.0f32; rows_per_shard];
                dev.stream().memcpy_dtoh(capture, &mut host).unwrap();
                dev.synchronize().unwrap();
                for (r, &v) in host.iter().enumerate() {
                    let expected = idx as f32 * 1.0e6 + r as f32;
                    assert!(
                        (v - expected).abs() < 0.5,
                        "repeat {repeat}, shard {idx}, row {r}: got {v}, expected {expected} \
                         (pinned-host-reuse race?)"
                    );
                }
            }
        }
    }

    /// `check_no_duplicate_columns` accepts a canonical CSR with strictly
    /// increasing per-row column indices. No panic, no GPU required.
    #[cfg(debug_assertions)]
    #[test]
    fn test_check_no_duplicate_columns_accepts_canonical_csr() {
        // 2 rows, 5 cols, each row strictly increasing.
        let csr = ScxCsr::new_unchecked(
            (2, 5),
            vec![0i64, 3, 5],
            vec![0i32, 2, 4, 1, 3],
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
        );
        check_no_duplicate_columns(&csr);
    }

    /// `check_no_duplicate_columns` panics in debug builds on an
    /// intra-row duplicate column index. Catches the foot-gun before the
    /// CSR is staged to GPU, where parallel writes would race
    /// nondeterministically.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "unsorted or duplicate column indices")]
    fn test_check_no_duplicate_columns_panics_on_duplicate() {
        // Row 0: [0, 3, 3] — duplicate column 3.
        let csr = ScxCsr::new_unchecked(
            (1, 5),
            vec![0i64, 3],
            vec![0i32, 3, 3],
            vec![1.0f32, 2.0, 3.0],
        );
        check_no_duplicate_columns(&csr);
    }
}
