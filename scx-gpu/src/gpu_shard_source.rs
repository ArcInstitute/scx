//! Device-resident shard sources for fully-GPU pipelines.
//!
//! [`GpuShardSource`] is the GPU counterpart of [`scx_format_io::ShardSource`]:
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
use scx_format_io::{prefetch, ShardSource};

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_matrix_source::{ValidationChecks, ValidationPolicy};
use crate::gpu_preprocess::apply_fused_ops_inner;
use crate::profile::{self, CodecClass};
use crate::shard_validate::{validate_shard, ShardToValidate};
use crate::staging::{GpuCsrSlot, PinnedCsrSlot};
use crate::staging_driver::{
    drive_shards, ShardConsumer, ShardFeeder, ShardStager, StagingPlan, ValidationMemo,
};

/// Approximate the host→device bytes a staged CSR shard moves (data f32 +
/// indices i32 + indptr i64). Used only for the Phase 0.2 GPU profiler.
#[inline]
fn csr_htod_bytes(csr: &scx_sparse::ScxCsr) -> usize {
    csr.data.len() * 4 + csr.indices.len() * 4 + (csr.n_rows() + 1) * 8
}

/// Host-RAM budget, in bytes, for the shards the GPU staging path holds
/// decoded-but-not-yet-staged. Unset means "no budget" — the depth is whatever
/// `SCX_ACCEL_PREFETCH_DEPTH` / `DEFAULT_PREFETCH_DEPTH` says.
const STAGING_MEMORY_BUDGET_ENV: &str = "SCX_GPU_STAGING_MEMORY_BUDGET";

fn staging_memory_budget() -> Option<u64> {
    static B: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var(STAGING_MEMORY_BUDGET_ENV)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|b| *b > 0)
    })
}

/// Decode-prefetch depth for the multi-shard staging loop, derated to a host
/// memory budget when one is set (§9.13 follow-on / the #373 review finding).
///
/// The pipeline holds up to `depth` decoded shards in host RAM at once, and GPU
/// staging is the path with the largest shards in the system — a census_500k
/// shard is ~190 MB decoded, so the default depth of 4 is ~760 MB. The #373
/// review found that `clamp_prefetch_depth` existed but clamped nothing
/// anywhere: its two call sites derated a `max_workers` that `StableOrder`
/// discards, and every prefetch call site passed the raw depth. This is the
/// first place it actually binds.
///
/// **No silent behaviour change**: with `SCX_GPU_STAGING_MEMORY_BUDGET` unset
/// (the default) this returns exactly what the old code used. Setting it makes
/// the advice "lower your prefetch depth on a memory-tight host" actionable in
/// terms a user can reason about — bytes — instead of a shard count whose cost
/// depends on the file.
///
/// Sources without a [`shard_size_hint`](ShardSource::shard_size_hint) cannot
/// be budgeted (there is no per-shard byte estimate to divide by) and keep the
/// unclamped depth.
fn resolve_staging_prefetch_depth(source: &(dyn ShardSource + Sync)) -> usize {
    resolve_staging_prefetch_depth_for(source.shard_size_hint())
}

/// The layout-agnostic half of [`resolve_staging_prefetch_depth`].
///
/// Takes the hint rather than the source so the column-major staging path can
/// share it: `ColumnShardSource::csc_shard_size_hint` reports the same
/// `ShardSizeHint` shape (its `max_rows` counts columns, but `decoded_bytes()`
/// is the same arithmetic either way, because a `ScxCsc` has the same three
/// arrays as a `ScxCsr`). Before this the budget existed on one layout only —
/// and CSC is the layout whose decode depth this phase raises from 1 to 4.
pub(crate) fn resolve_staging_prefetch_depth_for(
    hint: Option<scx_format_io::ShardSizeHint>,
) -> usize {
    let requested = prefetch::prefetch_depth();
    let (Some(budget), Some(hint)) = (staging_memory_budget(), hint) else {
        return requested;
    };
    let depth = prefetch::clamp_prefetch_depth(requested, hint.decoded_bytes(), budget);
    if depth < requested {
        log::debug!(
            "GPU staging: clamped decode-prefetch depth {requested} -> {depth} to fit the \
             {budget}-byte {STAGING_MEMORY_BUDGET_ENV} budget (~{} B/shard)",
            hint.decoded_bytes(),
        );
    }
    depth
}

/// `ShardSource` adapter that times every decode into the GPU profiler's
/// host-decode bucket.
///
/// Before Phase 4.2 the staging loop owned its decode call and could time it
/// inline. The bounded decode-prefetch pipeline owns it now, so the
/// instrumentation moves onto the source — otherwise the multi-shard path
/// silently stops reporting host-decode time, and "the host-decode bucket
/// dropped" becomes unfalsifiable.
///
/// As on the CPU side after 2.1, the reported total is the **sum over
/// concurrent workers** and so can exceed wall-clock; the ratio against wall is
/// the signal, not the absolute.
struct ProfiledDecode<'a>(&'a (dyn ShardSource + Sync));

impl ShardSource for ProfiledDecode<'_> {
    fn n_shards(&self) -> usize {
        self.0.n_shards()
    }
    fn n_obs(&self) -> usize {
        self.0.n_obs()
    }
    fn n_vars(&self) -> usize {
        self.0.n_vars()
    }
    fn max_shard_rows(&self) -> scx_format_io::Result<usize> {
        self.0.max_shard_rows()
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<scx_sparse::ScxCsr> {
        let t_decode = profile::start();
        let out = self.0.read_shard(shard_idx);
        profile::record_host_decode_since(CodecClass::Generic, t_decode);
        out
    }
}

/// Sequence of GPU-resident CSR shards.
///
/// Implementors own a shared device CSR slot; the trait yields borrowed
/// [`GpuCsrShardView`]s over the slot's live shard. The callback's
/// `&GpuCsrShardView` is invalidated when the iterator advances to the
/// next shard (the slot is mutated in place).
pub(crate) trait GpuShardSource {
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
pub(crate) struct RawGpuShardSource<'a> {
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
    /// Decode-prefetch depth for the multi-shard path, resolved once at
    /// construction by [`resolve_staging_prefetch_depth`].
    prefetch_depth: usize,
    /// Shard indices already host-validated by this adapter.
    ///
    /// New on this layout: the CSC adapter has had it since §8.3, but the CSR
    /// one re-ran both O(nnz) scans on every drive — and GPU DE drives a CSR
    /// source once per gene chunk. Sound for the same reason it is sound there:
    /// `source` is a shared borrow held for the adapter's whole lifetime, and
    /// `ShardSource::read_shard`'s stability contract requires structurally
    /// equivalent output for a given index across calls.
    ///
    /// The borrow alone would **not** be enough — a `&self` method may return
    /// whatever it likes, and this crate's own `GenerationalCsrSource` test
    /// fixture returns different values on every call. What makes the memo
    /// sound is the documented contract on the trait, which that fixture
    /// honours: it varies values only, never index order, bounds or
    /// finiteness. Raised by codex - gpt-5.6-sol, who noted that a shared
    /// borrow does not imply repeatable output.
    memo: ValidationMemo,
    /// How deeply to validate, and what to call the operation in an error.
    /// Defaults to full DE-strength checking so a consumer that forgets to
    /// choose fails closed.
    validation: ValidationPolicy,
}

impl<'a> RawGpuShardSource<'a> {
    /// Construct a raw GPU shard source, pre-sizing the staging buffers from
    /// the source's [`shard_size_hint`](ShardSource::shard_size_hint) when it
    /// offers one and growing them on demand when it does not.
    ///
    /// Deliberately does **not** call `ShardSource::max_shard_rows()` directly:
    /// its default trait impl decodes every shard, which would defeat the
    /// decode-prefetch pipelining and inflate I/O on sources without an O(1)
    /// override. `shard_size_hint` is `None` unless the implementor can answer
    /// cheaply — a backed reader reads it straight from catalog statistics — so
    /// consulting it is always safe. Callers who know their bounds
    /// independently can still pass them via [`Self::with_max_shard_rows`].
    ///
    /// Before 4.5 this always built at capacity 1 and let the slots
    /// grow-and-realloc up to the first shard's size, and
    /// `with_max_shard_rows` — the obvious pre-sizing hook, on the path with
    /// the largest shards in the system — had no production caller at all.
    pub fn new(dev: &'a GpuDevice, source: &'a (dyn ShardSource + Sync)) -> Result<Self, GpuError> {
        match source.shard_size_hint() {
            Some(hint) => Self::with_max_shard_rows(dev, source, hint.max_rows, hint.max_nnz),
            None => Self::build(dev, source, 1, 1),
        }
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
        let prefetch_depth = resolve_staging_prefetch_depth(source);

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
            prefetch_depth,
            memo: ValidationMemo::new(source.n_shards()),
            validation: ValidationPolicy::default(),
        })
    }

    /// Set the validation policy — how deeply each shard is checked, and the
    /// operation name its errors speak in.
    ///
    /// **Resets the validation memo.** The memo records only *that* a shard
    /// passed, not at which rung, so carrying it across a policy change lets a
    /// weaker pass satisfy a stronger one: drive at `Bounds` (on CSR, no scan
    /// at all), call `with_validation(Ranking)`, drive again — every shard is
    /// already marked seen and the finiteness scan never runs. Found by
    /// codex - gpt-5.6-sol. Resetting unconditionally is cheaper to reason
    /// about than tracking a high-water rung, and re-validating after a policy
    /// change is the conservative direction.
    pub fn with_validation(mut self, validation: ValidationPolicy) -> Self {
        self.validation = validation;
        self.memo = ValidationMemo::new(self.source.n_shards());
        self
    }

    /// Capacity of the reusable device CSR slot. Test-only: lets the
    /// pre-sizing test assert the staging buffers never grow-and-realloc when
    /// the source offered a `shard_size_hint`.
    #[cfg(test)]
    pub(crate) fn slot_capacity(&self) -> (usize, usize, usize) {
        self.slot.capacity()
    }

    /// Run `f` over each non-empty shard, via the shared
    /// [`drive_shards`](crate::staging_driver::drive_shards) lifecycle.
    ///
    /// The `transform` closure is invoked AFTER the upload event handshake but
    /// BEFORE the user callback, giving preprocessing variants a hook to
    /// modify the slot's `data` in place.
    ///
    /// Everything about the ordering — the pinned ring, the host-side DMA gate,
    /// both device-side event gates, the empty-shard skip, the drain — now
    /// lives in the driver, which the CSC adapter shares and which
    /// `staging_driver_tests.rs` exercises on a CPU host. What is left here is
    /// what the *layout* means.
    fn run<F, T>(&mut self, f: F, transform: T) -> Result<(), GpuError>
    where
        F: FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>,
        T: FnMut(&GpuDevice, &mut GpuCsrSlot) -> Result<(), GpuError>,
    {
        let plan = StagingPlan::for_source(self.source, self.prefetch_depth);
        // `ProfiledDecode` still wraps the source, because the prefetch
        // pipeline owns the decode call and inline timing would silently stop
        // reporting host-decode time on the multi-shard path.
        let feeder = CsrFeeder(ProfiledDecode(self.source));
        let mut stager = CsrStaging {
            dev: self.dev,
            pinned: &mut self.pinned,
            pinned_events: &mut self.pinned_events,
            slot: &mut self.slot,
            memo: &mut self.memo,
            validation: self.validation,
            copy_stream: &self.copy_stream,
            f,
            transform,
        };
        drive_shards(&feeder, &mut stager, &plan)
    }
}

/// Row-major [`ShardFeeder`]: the bounded, ordered decode-prefetch over CSR
/// shards.
struct CsrFeeder<'a>(ProfiledDecode<'a>);

impl ShardFeeder for CsrFeeder<'_> {
    type Shard = scx_sparse::ScxCsr;

    fn feed(
        &self,
        plan: &StagingPlan,
        consume: ShardConsumer<'_, Self::Shard>,
    ) -> Result<(), GpuError> {
        // `_uncached`: a single staging pass over a caching source gains
        // nothing from warming its LRU and would only evict a co-resident
        // reader's entries.
        prefetch::for_each_shard_ordered_uncached_selected(
            &self.0,
            &plan.indices,
            plan.depth,
            |i, csr| consume(i, &csr),
        )
    }
}

/// Row-major [`ShardStager`]: every CUDA call one CSR shard's staging needs.
///
/// Holds field references rather than `&mut RawGpuShardSource` because the
/// feeder needs the source shared while the ring and the device slot are
/// exclusive — the same split the hand-rolled loop performed by hoisting these
/// exact bindings out of `&mut self`, now checked by the compiler.
struct CsrStaging<'r, F, T> {
    dev: &'r GpuDevice,
    pinned: &'r mut [PinnedCsrSlot; 2],
    pinned_events: &'r mut [Option<CudaEvent>; 2],
    slot: &'r mut GpuCsrSlot,
    memo: &'r mut ValidationMemo,
    validation: ValidationPolicy,
    copy_stream: &'r Arc<CudaStream>,
    f: F,
    transform: T,
}

impl<F, T> ShardStager for CsrStaging<'_, F, T>
where
    F: FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>,
    T: FnMut(&GpuDevice, &mut GpuCsrSlot) -> Result<(), GpuError>,
{
    type Shard = scx_sparse::ScxCsr;

    /// Rows, not nonzeros: a CSR shard with rows but no nonzeros must still be
    /// dispatched, because `GpuPreprocessedShardSource` advances a
    /// `global_row_offset` per dispatched shard.
    fn is_stageable(&self, shard: &Self::Shard) -> bool {
        shard.n_rows() != 0
    }

    fn validate(&mut self, _idx: usize, shard: &Self::Shard) -> Result<(), GpuError> {
        validate_shard(ShardToValidate::Csr(shard), &self.validation)
    }

    fn memo(&mut self) -> &mut ValidationMemo {
        self.memo
    }

    fn host_wait(&mut self, slot: usize) -> Result<(), GpuError> {
        if let Some(evt) = self.pinned_events[slot].take() {
            evt.synchronize()
                .map_err(|e| GpuError::CudaError(format!("pinned event sync: {e}")))?;
        }
        Ok(())
    }

    fn stage_and_upload(&mut self, slot: usize, shard: &Self::Shard) -> Result<(), GpuError> {
        let t_stage = profile::start();
        self.pinned[slot].stage(shard)?;
        profile::record_htod_since(CodecClass::Generic, t_stage, csr_htod_bytes(shard));
        self.pinned[slot].upload_to(
            self.dev,
            self.copy_stream,
            self.slot,
            shard.n_rows(),
            shard.data.len(),
            shard.n_cols(),
        )
    }

    fn gate_copy_to_compute(&mut self, slot: usize) -> Result<(), GpuError> {
        let upload_event = self
            .copy_stream
            .record_event(None)
            .map_err(|e| GpuError::CudaError(format!("record upload event: {e}")))?;
        self.dev
            .stream()
            .wait(&upload_event)
            .map_err(|e| GpuError::CudaError(format!("compute wait: {e}")))?;
        // The same event is stashed against this slot so the next drive that
        // recycles it has something to host-wait on.
        self.pinned_events[slot] = Some(upload_event);
        Ok(())
    }

    fn dispatch(&mut self, idx: usize, _shard: &Self::Shard) -> Result<(), GpuError> {
        (self.transform)(self.dev, self.slot)?;
        (self.f)(idx, self.slot)
    }

    fn gate_compute_to_copy(&mut self) -> Result<(), GpuError> {
        let compute_event = self
            .dev
            .stream()
            .record_event(None)
            .map_err(|e| GpuError::CudaError(format!("record compute event: {e}")))?;
        self.copy_stream
            .wait(&compute_event)
            .map_err(|e| GpuError::CudaError(format!("copy wait: {e}")))
    }

    fn drain(&mut self) -> Result<(), GpuError> {
        for evt in self.pinned_events.iter_mut() {
            if let Some(e) = evt.take() {
                e.synchronize()
                    .map_err(|err| GpuError::CudaError(format!("pinned drain sync: {err}")))?;
            }
        }
        Ok(())
    }
}

impl<'a> GpuShardSource for RawGpuShardSource<'a> {
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
pub(crate) struct GpuPreprocessedShardSource<'a> {
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
            // `Bounds`: the fused kernels rewrite `data` per row in place and
            // never index by column, so neither sortedness nor finiteness is a
            // precondition — `normalize_total` on a NaN-bearing matrix is the
            // CPU behaviour too. On CSR `Bounds` runs no scan at all, which is
            // the point: this path used to pay both DE scans per shard.
            inner: RawGpuShardSource::new(dev, source)?.with_validation(ValidationPolicy::new(
                ValidationChecks::IN_RANGE,
                "normalize_total/log1p",
            )),
            normalize,
            log1p,
            row_scale,
        })
    }
}

impl<'a> GpuShardSource for GpuPreprocessedShardSource<'a> {
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
        // `run` invokes the transform only for non-empty shards, in the
        // ascending order of its `StagingPlan` — every shard when the source
        // reports no row filter, otherwise only the shards its
        // `visible_shard_indices` names. Empty shards contribute 0 rows either
        // way, so this matches the *visible* row layout the factor vector is
        // indexed against. (It said `0..n_shards` before `StagingPlan::for_source`
        // made a filtered plan possible; the row_scale factors are host-side and
        // sized to the visible axis, so a disagreeing plan mis-scales rather
        // than reading out of bounds — the DE and pseudobulk passes index a
        // device `cell_to_group` and so carry `VisibleRowCursor` instead.)
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
    use crate::shard_validate::validate_par_min_nnz;
    use scx_sparse::ScxCsr;

    /// The pre-§8.14 CSR entry point, as a test shim at the default policy.
    ///
    /// The eight scanner tests below predate `ValidationChecks` and assert
    /// things that have nothing to do with it — that the parallel and serial
    /// arms name the same offender, that `position_first` is used, that rows
    /// are scanned independently. Routing them through the default policy keeps
    /// them asserting exactly what they did, and keeps the level work from
    /// quietly reducing what they cover.
    fn validate_shard_for_gpu_de(csr: &ScxCsr) -> Result<(), GpuError> {
        validate_shard(ShardToValidate::Csr(csr), &ValidationPolicy::default())
    }

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
        fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
            Ok(self.shards[shard_idx].clone())
        }
    }

    /// Same as `InMemorySource` but advertises a `shard_size_hint`, standing in
    /// for a catalog-backed reader.
    struct HintedSource {
        inner: InMemorySource,
    }

    impl ShardSource for HintedSource {
        fn n_shards(&self) -> usize {
            self.inner.n_shards()
        }
        fn n_obs(&self) -> usize {
            self.inner.n_obs()
        }
        fn n_vars(&self) -> usize {
            self.inner.n_vars()
        }
        fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
            self.inner.read_shard(shard_idx)
        }
        fn shard_size_hint(&self) -> Option<scx_format_io::ShardSizeHint> {
            Some(scx_format_io::ShardSizeHint {
                max_rows: self.inner.shards.iter().map(|s| s.n_rows()).max()?,
                max_nnz: self.inner.shards.iter().map(|s| s.data.len()).max()?,
            })
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

    fn hinted(shards: Vec<ScxCsr>, n_vars: usize) -> HintedSource {
        let n_obs = shards.iter().map(|s| s.n_rows()).sum();
        HintedSource {
            inner: InMemorySource {
                shards,
                n_obs,
                n_vars,
            },
        }
    }

    /// §9.13 follow-on: with no budget set — the default — the depth clamp is
    /// inert. This is the "no silent behaviour change" claim, so it is asserted
    /// rather than argued, including the premise that the knob really is unset
    /// in this process (a leaked env var would make the test vacuous).
    #[test]
    fn staging_depth_is_unclamped_without_a_budget() {
        assert!(
            std::env::var(STAGING_MEMORY_BUDGET_ENV).is_err(),
            "premise: {STAGING_MEMORY_BUDGET_ENV} must be unset for this test to mean anything"
        );
        let src = hinted(vec![make_csr(4, 8, 1.0), make_csr(4, 8, 2.0)], 8);
        assert_eq!(
            resolve_staging_prefetch_depth(&src),
            prefetch::prefetch_depth()
        );
    }

    /// A source that cannot describe its shards cannot be budgeted — there is
    /// no per-shard byte estimate to divide by — so it keeps the full depth
    /// even when a budget is set.
    #[test]
    fn staging_depth_needs_a_hint_to_be_clamped() {
        let src = InMemorySource {
            shards: vec![make_csr(4, 8, 1.0), make_csr(4, 8, 2.0)],
            n_obs: 8,
            n_vars: 8,
        };
        assert!(ShardSource::shard_size_hint(&src).is_none());
        assert_eq!(
            resolve_staging_prefetch_depth(&src),
            prefetch::prefetch_depth()
        );
    }

    /// The clamp arithmetic the budget path performs, exercised directly since
    /// the budget itself is a process-global `OnceLock` that a test cannot vary.
    #[test]
    fn staging_depth_clamp_arithmetic() {
        let src = hinted(vec![make_csr(4, 8, 1.0)], 8);
        let hint = ShardSource::shard_size_hint(&src).unwrap();
        let per_shard = hint.decoded_bytes();
        assert!(per_shard > 0);

        // Budget for exactly two shards -> depth 2, whatever was requested.
        assert_eq!(
            prefetch::clamp_prefetch_depth(8, per_shard, per_shard * 2),
            2
        );
        // A budget smaller than one shard still floors at 1: refusing to decode
        // anything would be worse than exceeding the budget.
        assert_eq!(prefetch::clamp_prefetch_depth(8, per_shard, 1), 1);
        // A generous budget never raises the requested depth.
        assert_eq!(
            prefetch::clamp_prefetch_depth(2, per_shard, per_shard * 100),
            2
        );
    }

    /// A CSR source whose single shard carries a **different payload on every
    /// read**, tagged by a generation counter.
    ///
    /// This is the load-bearing half of the cross-drive test below. With a
    /// fixed fixture, a pinned slot overwritten mid-DMA is overwritten with the
    /// *same* bytes, so the readback matches and the assertion cannot fail —
    /// which is what the first version of that test did (found by
    /// codex - gpt-5.6-sol in review).
    struct GenerationalCsrSource {
        rows: usize,
        n_vars: usize,
        generation: std::sync::atomic::AtomicUsize,
    }

    impl GenerationalCsrSource {
        /// Value base for read `g`. Spaced so two generations share no value,
        /// and small enough that every value is an exactly-representable f32
        /// integer (< 2^24).
        fn base_for(g: usize) -> f32 {
            (100_000 * (g + 1)) as f32
        }
    }

    impl ShardSource for GenerationalCsrSource {
        fn n_shards(&self) -> usize {
            1
        }
        fn n_obs(&self) -> usize {
            self.rows
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_shard(&self, _shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
            let g = self
                .generation
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(make_csr(self.rows, self.n_vars, Self::base_for(g)))
        }
    }

    /// Driving one adapter repeatedly stages each drive's own bytes.
    ///
    /// The hardware half of §8.18, and the row-major twin of
    /// `driving_a_single_shard_csc_source_repeatedly_stages_own_bytes`. Both layouts
    /// used to special-case a single-shard drive: no ring, no copy stream, and
    /// **no event recorded**, on the stated grounds that "there is no successor
    /// `stage()` that could race the in-flight DMA". True within one drive.
    /// Across two, the second drive re-entered `pinned[0]` while the first
    /// drive's `memcpy_htod_async` could still be reading it, with nothing on
    /// the host having synchronised.
    ///
    /// `staging_driver_tests.rs` proves the driver *orders* the host-wait; only
    /// a device can show the bytes survive.
    ///
    /// Two properties make this able to fail, and the first version of this
    /// test had neither (codex - gpt-5.6-sol):
    ///
    /// 1. **Every drive carries distinct bytes** (`GenerationalCsrSource`), so
    ///    a slot overwritten mid-DMA shows up as another generation's values.
    /// 2. **No host synchronisation between drives.** Device buffers are
    ///    allocated up front — `cudaMalloc` is itself device-synchronising, so
    ///    allocating inside the loop would silently insert the very barrier the
    ///    eventless path lacked — and the single `dev.synchronize()` comes only
    ///    after all `DRIVES` drives are queued. Readback happens afterwards.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn driving_a_single_shard_csr_source_repeatedly_stages_own_bytes() {
        let dev = require_gpu!();
        const ROWS: usize = 65_536;
        const N_VARS: usize = 16;
        const DRIVES: usize = 8;

        // `make_csr` emits exactly one nonzero per row.
        const NNZ: usize = ROWS;

        let src = GenerationalCsrSource {
            rows: ROWS,
            n_vars: N_VARS,
            generation: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut gpu = RawGpuShardSource::new(&dev, &src).unwrap();

        // Pre-allocated: no allocation, and therefore no implicit device
        // synchronisation, inside the drive loop.
        let mut captures: Vec<CudaSlice<f32>> = (0..DRIVES)
            .map(|_| dev.alloc_zeros::<f32>(NNZ).unwrap())
            .collect();

        for dst in captures.iter_mut() {
            gpu.for_each_gpu_shard(|_idx, slot| {
                let view = slot.view();
                assert_eq!(view.data.len(), NNZ, "fixture nnz must be stable");
                dev.stream()
                    .memcpy_dtod(&view.data, dst)
                    .map_err(|e| GpuError::CudaError(format!("dtod: {e}")))?;
                Ok(())
            })
            .unwrap();
            // Deliberately no synchronisation here.
        }
        dev.synchronize().unwrap();

        for (g, dst) in captures.iter().enumerate() {
            let mut host = vec![0.0f32; NNZ];
            dev.stream().memcpy_dtoh(dst, &mut host).unwrap();
            dev.synchronize().unwrap();
            let expected = make_csr(ROWS, N_VARS, GenerationalCsrSource::base_for(g)).data;
            assert_eq!(
                host, expected,
                "drive {g} did not stage its own generation — the pinned slot was rewritten \
                 while this drive's DMA was still reading it"
            );
        }
    }

    /// A hinted source pre-sizes the staging slot at construction, so the drain
    /// never grows-and-reallocs. Before 4.5 `with_max_shard_rows` — the hook for
    /// this, on the path with the largest shards in the system — had no
    /// production caller and every source started at capacity 1.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn hinted_source_presizes_the_staging_slot() {
        let dev = require_gpu!();
        let shards = vec![make_csr(3, 5, 1.0), make_csr(7, 5, 100.0)];
        let src = hinted(shards, 5);

        let mut gpu_src = RawGpuShardSource::new(&dev, &src).unwrap();
        let at_construction = gpu_src.slot_capacity();
        assert!(
            at_construction.0 >= 8 && at_construction.1 >= 7 && at_construction.2 >= 7,
            "slot must be pre-sized for the largest shard (7 rows / 7 nnz), got \
             {at_construction:?}"
        );

        gpu_src.for_each_gpu_shard(|_idx, _slot| Ok(())).unwrap();
        assert_eq!(
            gpu_src.slot_capacity(),
            at_construction,
            "a pre-sized slot must not grow during the drain"
        );
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

    /// Skip rather than fail when `SCX_GPU_VALIDATE_PAR_MIN_NNZ` puts the
    /// parallel path out of reach for a fixture of `nnz`. A premise assertion
    /// should fire when the *fixture* is wrong, not when the environment
    /// deliberately disables the feature under test — the A/B arm that pins the
    /// threshold high would otherwise report four failures instead of a
    /// measurement.
    fn parallel_path_reachable(nnz: usize) -> bool {
        if nnz >= validate_par_min_nnz() {
            return true;
        }
        eprintln!(
            "SCX_GPU_VALIDATE_PAR_MIN_NNZ={} puts the parallel scan out of reach for a \
             {nnz}-nnz fixture — skipping",
            validate_par_min_nnz()
        );
        false
    }

    /// Build a shard above [`validate_par_min_nnz`] so validation takes the
    /// parallel path: `rows` rows × `per_row` strictly-increasing columns.
    fn make_big_csr(rows: usize, per_row: usize) -> ScxCsr {
        let mut indptr = Vec::with_capacity(rows + 1);
        indptr.push(0i64);
        let mut indices = Vec::with_capacity(rows * per_row);
        let mut data = Vec::with_capacity(rows * per_row);
        for r in 0..rows {
            for c in 0..per_row {
                indices.push(c as i32);
                data.push((r * per_row + c) as f32);
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsr::new_unchecked((rows, per_row), indptr, indices, data)
    }

    /// §9.13: the parallel scan must name the **first** offending row, not
    /// whichever one a worker happens to reach first. Three rows are corrupted;
    /// only the lowest may be reported, and the message must be the one the
    /// serial scan produced.
    #[test]
    fn validate_parallel_reports_the_minimum_offending_row() {
        let per_row = 64;
        let rows = 4096; // 262 144 nnz — comfortably over the parallel threshold
        let mut csr = make_big_csr(rows, per_row);
        if !parallel_path_reachable(csr.data.len()) {
            return;
        }

        // Corrupt rows 3000, 977 and 2500 (inserted out of order on purpose).
        for &r in &[3000usize, 977, 2500] {
            let s = csr.indptr[r] as usize;
            csr.indices[s + 5] = csr.indices[s + 4]; // duplicate → a >= b
        }

        let err = validate_shard_for_gpu_de(&csr).unwrap_err();
        let GpuError::InvalidShard(msg) = err else {
            panic!("expected InvalidShard, got {err:?}");
        };
        assert_eq!(
            msg,
            "ScxCsr row 977 has unsorted or duplicate column indices (4 >= 4): the GPU shard \
             scatter behind this GPU operation requires strictly-increasing per-row indices \
             for deterministic output",
            "parallel scan must report the same first offending row as the serial one"
        );

        // The identical shard truncated below the threshold takes the serial
        // path and must agree on the row it names.
        let small = {
            let rows_small = 1000;
            let nnz = csr.indptr[rows_small] as usize;
            ScxCsr::new_unchecked(
                (rows_small, per_row),
                csr.indptr[..=rows_small].to_vec(),
                csr.indices[..nnz].to_vec(),
                csr.data[..nnz].to_vec(),
            )
        };
        assert!(
            small.data.len() < validate_par_min_nnz(),
            "premise: the truncated fixture must take the serial path"
        );
        let GpuError::InvalidShard(small_msg) = validate_shard_for_gpu_de(&small).unwrap_err()
        else {
            panic!("expected InvalidShard from the serial path");
        };
        assert_eq!(small_msg, msg, "serial and parallel scans must agree");
    }

    /// Same for the finiteness scan: `position_first`, not "any position".
    #[test]
    fn validate_parallel_reports_the_first_non_finite_position() {
        let mut csr = make_big_csr(4096, 64);
        if !parallel_path_reachable(csr.data.len()) {
            return;
        }

        csr.data[200_000] = f32::INFINITY;
        csr.data[12_345] = f32::NAN;
        csr.data[99_999] = f32::NEG_INFINITY;

        let GpuError::InvalidShard(msg) = validate_shard_for_gpu_de(&csr).unwrap_err() else {
            panic!("expected InvalidShard");
        };
        assert!(
            msg.contains("at nonzero index 12345"),
            "must name the first non-finite position, got: {msg}"
        );
    }

    /// A large clean shard is accepted — guards against the parallel path
    /// inventing an offender (e.g. a window straddling a row boundary).
    #[test]
    fn validate_accepts_large_clean_shard() {
        let csr = make_big_csr(4096, 64);
        if !parallel_path_reachable(csr.data.len()) {
            return;
        }
        assert!(validate_shard_for_gpu_de(&csr).is_ok());
    }

    /// Rows are validated independently: `indices` is one flat array, so a
    /// naive `windows(2)` over the whole array would see the boundary pair
    /// (last column of row r, first column of row r+1) and reject a perfectly
    /// legal shard. Every row here ends high and starts low.
    #[test]
    fn validate_does_not_straddle_row_boundaries() {
        let rows = 2048;
        let per_row = 64;
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in 0..rows {
            for c in 0..per_row {
                indices.push(c as i32); // every row restarts at column 0
                data.push(1.0f32 + r as f32);
            }
            indptr.push(indices.len() as i64);
        }
        let csr = ScxCsr::new_unchecked((rows, per_row), indptr, indices, data);
        if !parallel_path_reachable(csr.data.len()) {
            return;
        }
        assert!(validate_shard_for_gpu_de(&csr).is_ok());
    }

    /// Raw source yields the staged shard verbatim (slot.view returns
    /// the uploaded data).
    #[test]
    #[ignore = "requires a CUDA GPU"]
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
    #[ignore = "requires a CUDA GPU"]
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
    #[ignore = "requires a CUDA GPU"]
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
    #[ignore = "requires a CUDA GPU"]
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
    #[ignore = "requires a CUDA GPU"]
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

    /// A policy can require a check another policy does not, in **either**
    /// direction — which is the premise the behavioural test below rests on,
    /// and the reason these are independent switches rather than a ladder.
    ///
    /// The `FINITE` / `SORTED` pair is the case an ordered ladder could not
    /// express: neither is a superset of the other, so under a ladder the only
    /// way to reach `finite` was to also demand `sorted`
    /// (codex - gpt-5.6-sol).
    #[test]
    fn check_sets_are_independent_and_neither_implies_the_other() {
        let finite = ValidationChecks::FINITE;
        let sorted = ValidationChecks::SORTED;
        assert!(finite.finite && !finite.sorted);
        assert!(sorted.sorted && !sorted.finite);
        // Both keep the floor, which is the only check that is genuinely common.
        assert!(finite.in_range && sorted.in_range);
        // ALL is the fail-closed default and is a superset of both.
        let all = ValidationChecks::ALL;
        assert!(all.in_range && all.sorted && all.finite);
        assert_eq!(ValidationPolicy::default().checks, all);
        // IN_RANGE runs no scan on a row-major shard.
        let floor = ValidationChecks::IN_RANGE;
        assert!(!floor.sorted && !floor.finite);
    }

    /// Driving at `Bounds`, then upgrading to `Ranking`, must re-validate.
    ///
    /// Found by codex - gpt-5.6-sol. The memo is a bare `Vec<bool>` recording
    /// *that* a shard passed, never at which rung, so carrying it across a
    /// policy change lets a weaker pass satisfy a stronger one — and the
    /// weakest pass on a row-major shard is **no scan at all**:
    ///
    /// ```text
    /// drive at Bounds          → CSR runs zero scans, every shard marked seen
    /// with_validation(Ranking)
    /// drive again              → memo says validated → finiteness never runs
    /// ```
    ///
    /// A NaN then reaches `block_radix_sort_per_gene_kernel` — the exact silent
    /// corruption `Ranking` exists to prevent, reached by *asking for more*
    /// validation.
    ///
    /// GPU-gated because `RawGpuShardSource` cannot be built without a device.
    /// The cheap structural half — that both `with_validation` bodies reset the
    /// memo — is an `ORG-8.20-1` CI grep, so a revert is caught on a CPU runner
    /// even though this behaviour is not.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn upgrading_the_policy_revalidates_shards_the_weaker_policy_waved_through() {
        let dev = require_gpu!();
        let mut csr = make_csr(64, 8, 1.0);
        csr.data[7] = f32::NAN;
        let src = InMemorySource {
            shards: vec![csr],
            n_obs: 64,
            n_vars: 8,
        };

        let mut gpu =
            RawGpuShardSource::new(&dev, &src)
                .unwrap()
                .with_validation(ValidationPolicy::new(
                    ValidationChecks::IN_RANGE,
                    "normalize_total",
                ));
        gpu.for_each_gpu_shard(|_, _| Ok(()))
            .expect("premise: Bounds runs no CSR scan, so the NaN is accepted");

        let mut gpu = gpu.with_validation(ValidationPolicy::new(
            ValidationChecks::FINITE,
            "rank_genes_groups",
        ));
        let err = gpu
            .for_each_gpu_shard(|_, _| Ok(()))
            .expect_err("Ranking must re-scan a shard the Bounds pass never looked at");
        let GpuError::InvalidShard(msg) = err else {
            panic!("expected InvalidShard, got {err:?}");
        };
        assert!(
            msg.contains("non-finite") && msg.contains("rank_genes_groups"),
            "message must name the non-finite value and the upgraded op, got: {msg}"
        );
    }
}
