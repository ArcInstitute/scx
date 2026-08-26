//! Device-resident CSC shard view + source trait (parallel hierarchy to the
//! CSR `GpuShardSource`).
//!
//! CSC sidecars store gene-major data (`docs/format.md` section 5,
//! `b"SCXS"` shard header with `shard_type = 1`). Each on-disk shard
//! covers a contiguous column range `[col_start, col_end)` × all rows
//! (column-only sharding; multi-shard via `--csc-cols-per-shard`).
//!
//! The view + source defined here are the GPU-side mirror of
//! `scx-format::shard_source::ColumnShardSource`. Built for the G4 v3 DE
//! pipeline (`pdex_ref_gpu_chunked_v3_csc`) — see
//! `scx-accel/src/diffexp/gpu.rs` for the consumer.
//!
//! [`RawGpuCscShardSource`] mirrors [`crate::gpu_shard_source::RawGpuShardSource`]'s
//! G3-shaped pipelining: a 2-slot pinned host ring, a dedicated copy
//! stream, a bounded rayon decode-prefetch (`scx_format_io::prefetch`,
//! depth 4 by default, derated to fit `SCX_GPU_STAGING_MEMORY_BUDGET` **when
//! the source supplies a size hint** — without one there is nothing to price
//! the budget against and the requested depth stands), and
//! an event-driven handshake between the copy and compute streams. The
//! synchronous baseline from G4.3 has been replaced because it
//! serialised decode + H→D + compute and regressed wall time at
//! scale (smartseq2 / tabula) where it dominates the per-shard
//! overhead.

use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::safe::{CudaEvent, CudaSlice, CudaStream, CudaView};

use scx_format_io::prefetch;
use scx_format_io::shard_source::ColumnShardSource;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_matrix_source::ValidationPolicy;
use crate::gpu_shard_source::resolve_staging_prefetch_depth_for;
use crate::profile::{self, CodecClass};
use crate::shard_validate::{validate_shard, ShardToValidate};
use crate::staging::PinnedCscSlot;
use crate::staging_driver::{
    drive_shards, ShardConsumer, ShardFeeder, ShardStager, StagingPlan, ValidationMemo,
};

/// Borrowed CSC view backed by a device-resident CSC shard slot.
///
/// Column-major: `col_indptr` indexes into `row_indices` / `data` per
/// column. `row_indices` stores **global** row (cell) ids — the cell
/// numbering is consistent across all shards in the file. This matches the
/// on-disk encoding documented in `docs/format.md` § 5 (`shard_type = 1`).
///
/// All fields are exact-sized: `col_indptr.len() == n_cols_in_shard + 1`,
/// `row_indices.len() == data.len() == nnz`. Callers can pass `&col_indptr`,
/// `&row_indices`, `&data` directly to kernel launches.
pub struct GpuCscShardView<'a> {
    /// `[n_cols_in_shard + 1]` cumulative offsets (CSC-style).
    pub col_indptr: CudaView<'a, i64>,
    /// `[nnz]` global row (cell) indices.
    pub row_indices: CudaView<'a, i32>,
    /// `[nnz]` values.
    pub data: CudaView<'a, f32>,
    /// First global column id covered by this shard (inclusive).
    pub col_start: usize,
    /// One-past-last global column id (exclusive); `col_end - col_start` is
    /// the shard's column count.
    pub col_end: usize,
    /// Cell count (n_obs) — same across all shards in a file.
    pub n_obs: usize,
}

impl<'a> GpuCscShardView<'a> {
    /// Number of CSC columns held by this shard.
    pub fn n_cols(&self) -> usize {
        self.col_end - self.col_start
    }

    /// nnz from the view's data length (matches `col_indptr[n_cols]`).
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Sequence of GPU-resident CSC shards.
///
/// Parallel hierarchy to [`crate::gpu_shard_source::GpuShardSource`]; CSC is
/// not part of that trait because the slot layouts are incompatible (CSR
/// stores row-major indptr + col indices; CSC stores col-major indptr +
/// row indices).
pub trait GpuCscShardSource {
    /// Number of CSC shards in this source.
    fn n_csc_shards(&self) -> usize;

    /// Total observation count (rows). Same across all shards by CSC's
    /// column-only sharding invariant.
    fn n_obs(&self) -> usize;

    /// Variable count (columns) across all shards combined.
    fn n_vars(&self) -> usize;

    /// `(n_obs, n_vars)`.
    fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    /// Iterate over each non-empty CSC shard, invoking the callback with
    /// `(shard_idx, &GpuCscShardView)` positioned at the live shard.
    ///
    /// The view is invalidated when the iterator advances to the next
    /// shard (device slots are mutated in place).
    fn for_each_gpu_csc_shard<F>(&mut self, f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>;

    /// Iterate over CSC shards whose `[col_start, col_end)` overlaps
    /// `col_range`, skipping all others **before** decode + upload.
    ///
    /// Default impl falls back to a full iteration with the overlap
    /// filter applied inside the callback (i.e., still decodes +
    /// uploads every shard but ignores non-overlapping ones).
    /// Implementors that can answer "does this shard overlap?" via
    /// cheap catalog metadata (e.g. [`RawGpuCscShardSource`]) override
    /// this to skip the decode entirely — material at scale (smartseq2
    /// / tabula) where most shards don't overlap a given gene chunk.
    fn for_each_gpu_csc_shard_in_range<F>(
        &mut self,
        col_range: Range<u32>,
        mut f: F,
    ) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    {
        // Post-decode filter, for an impl that cannot plan which shards to
        // read. `RawGpuCscShardSource` overrides this with a real prefilter.
        // The boundary condition is `scx_format_io`'s, not a local restatement
        // of it — this used to be the fourth hand-written copy of the same
        // half-open rule.
        self.for_each_gpu_csc_shard(|idx, view| {
            if !scx_format_io::col_range_overlaps(
                view.col_start as u32,
                view.col_end as u32,
                &col_range,
            ) {
                return Ok(());
            }
            f(idx, view)
        })
    }
}

/// Adapter from a CPU-side [`ColumnShardSource`] to [`GpuCscShardSource`]
/// with G3-shaped pipelining.
///
/// Owns the staging pipeline:
/// - A **2-slot ring** of [`PinnedCscSlot`]s. Two are required to fix the
///   same host-side reuse race the CSR `RawGpuShardSource` documents: with
///   pinned memory the H→D copy issued on `copy_stream` is truly
///   asynchronous, so the next iteration's `stage()` would otherwise
///   overwrite the pinned source buffer while the prior shard's DMA was
///   still in flight. The ring pings between the two slots; before
///   re-staging, the loop host-waits on the captured copy-stream event for
///   that slot.
/// - Three reusable device CSC buffers (`col_indptr`, `row_indices`,
///   `data`) with grow-only capacity tracking.
/// - A dedicated `copy_stream` and per-shard upload events for the
///   device-side handshake (compute waits on copy; next copy waits on
///   compute reads of the prior shard's device buffers).
///
/// The decode loop is `scx_format_io::prefetch`'s bounded rayon pipeline —
/// depth 4 by default — and not the single scoped worker thread this said
/// before ORG-8.20-1 PR B. The difference is the live decoded-shard set, and
/// so the host RSS, which is a blast-radius item rather than a detail.
///
/// `SCX_GPU_STAGING_MEMORY_BUDGET` derates that depth only when the source also
/// supplies a `csc_shard_size_hint`: `resolve_staging_prefetch_depth_for`
/// returns the requested depth unchanged if either the budget or the hint is
/// absent, because there is nothing to price the budget against. A
/// `ColumnShardSource` that does not override the hint is therefore **not**
/// bounded by that env var (codex - gpt-5.6-sol).
///
/// CSC kernels in this pipeline (`csc_shard_pseudobulk_kernel`,
/// `csc_shard_to_gene_major_kernel`) are custom — no cuSPARSE descriptor
/// cache is needed.
///
/// Every shard is run through [`crate::shard_validate::validate_shard`] before it is
/// staged, the column-major counterpart of what the CSR pipeline has always
/// done. Until review §8.3 this path validated nothing at all, so the
/// CSC-direct DE route — the default whenever a sidecar exists — fed NaN,
/// duplicate `(cell, gene)` pairs and out-of-range cell ids straight to the
/// kernels.
pub struct RawGpuCscShardSource<'a> {
    dev: &'a GpuDevice,
    source: &'a (dyn ColumnShardSource + Sync),
    /// Shard indices already validated by this adapter, so a source iterated
    /// once per gene chunk pays the O(nnz) scans once per shard rather than
    /// once per chunk. Sound because `source` is a shared borrow held for the
    /// adapter's whole lifetime: the bytes behind a given shard index cannot
    /// change underneath it.
    memo: ValidationMemo,
    /// How deeply to validate, and what to call the operation in an error.
    /// Defaults to full DE-strength checking so a consumer that forgets to
    /// choose fails closed.
    validation: ValidationPolicy,
    pinned: [PinnedCscSlot; 2],
    /// Per-pinned-slot copy-stream events captured immediately after the
    /// slot's most recent `upload_to`. Host-waited on before the slot is
    /// reused for the next `stage()` so the CPU never overwrites a buffer
    /// whose DMA is still in flight.
    pinned_events: [Option<CudaEvent>; 2],
    col_indptr: CudaSlice<i64>,
    row_indices: CudaSlice<i32>,
    data: CudaSlice<f32>,
    col_indptr_cap: usize,
    nnz_cap: usize,
    copy_stream: Arc<CudaStream>,
    /// Decode-prefetch depth, resolved once at construction and derated to
    /// `SCX_GPU_STAGING_MEMORY_BUDGET` when one is set. Before ORG-8.20-1 this
    /// path had no depth at all: it hand-rolled a `sync_channel(1)`, so exactly
    /// one shard was ever decoded ahead, and the budget knob priced nothing.
    prefetch_depth: usize,
    n_obs: usize,
    n_vars: usize,
}

impl<'a> RawGpuCscShardSource<'a> {
    /// Construct with lazy-grow staging buffers and a dedicated copy
    /// stream when more than one shard is present.
    ///
    /// Pre-sizes the pinned and device buffers from
    /// [`ColumnShardSource::csc_shard_size_hint`] when the source offers one,
    /// growing on demand when it does not.
    ///
    /// It used to say pre-sizing was impossible here because "the trait doesn't
    /// expose them". The trait didn't; `BackedCscReader` did, and threw the
    /// `nnz` away while reading the column range out of the same stats block.
    /// The row-major side has pre-sized since Phase 4.5.
    pub fn new(
        dev: &'a GpuDevice,
        source: &'a (dyn ColumnShardSource + Sync),
    ) -> Result<Self, GpuError> {
        let (n_obs, n_vars) = source.shape();
        let ctx = dev.context();
        let hint = source.csc_shard_size_hint();
        // `max_rows` is the major axis on this layout, i.e. columns — so the
        // indptr wants one more than that. Over-estimating costs pinned host
        // memory and one device alloc; under-estimating costs a realloc.
        let (col_indptr_cap, nnz_cap) = match hint {
            Some(h) => (h.max_rows.saturating_add(1).max(1), h.max_nnz.max(1)),
            None => (1, 1),
        };
        let pinned = [
            PinnedCscSlot::new(ctx, col_indptr_cap, nnz_cap),
            PinnedCscSlot::new(ctx, col_indptr_cap, nnz_cap),
        ];
        let col_indptr = dev.alloc_zeros::<i64>(col_indptr_cap)?;
        let row_indices = dev.alloc_zeros::<i32>(nnz_cap)?;
        let data = dev.alloc_zeros::<f32>(nnz_cap)?;

        // Dedicated copy stream — used only when n_csc_shards > 1
        // A single-shard file aliases this to the compute stream: with one
        // shard there is no second slot to overlap with, so a dedicated copy
        // stream buys nothing.
        //
        // This is NOT the deleted single-shard *fast path* — that one skipped
        // the event record entirely, which is the §8.18 bug. Every drive still
        // records and waits on its upload event whatever this alias resolves
        // to. Spelled out because the earlier wording named the fast path and
        // would invite the next reader to put it back.
        let copy_stream = if source.n_csc_shards() > 1 {
            dev.context()
                .new_stream()
                .map_err(|e| GpuError::StreamError(format!("new_stream: {e}")))?
        } else {
            dev.stream().clone()
        };

        Ok(Self {
            dev,
            source,
            memo: ValidationMemo::new(source.n_csc_shards()),
            validation: ValidationPolicy::default(),
            pinned,
            pinned_events: [None, None],
            col_indptr,
            row_indices,
            data,
            col_indptr_cap,
            nnz_cap,
            copy_stream,
            prefetch_depth: resolve_staging_prefetch_depth_for(hint),
            n_obs,
            n_vars,
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
        self.memo = ValidationMemo::new(self.source.n_csc_shards());
        self
    }

    /// Number of shards this adapter has validated so far. Test-only: lets the
    /// caching test assert that a second pass over the same source re-scans
    /// nothing, and that a pass did not skip a shard it staged.
    #[cfg(test)]
    pub(crate) fn validated_count(&self) -> usize {
        self.memo.validated_count()
    }

    /// Device buffer capacities `(col_indptr, nnz)`. Test-only: lets the
    /// pre-sizing test assert the staging buffers never grow-and-realloc when
    /// the source offered a `csc_shard_size_hint`.
    #[cfg(test)]
    pub(crate) fn device_capacity(&self) -> (usize, usize) {
        (self.col_indptr_cap, self.nnz_cap)
    }

    /// Shared driver for both trait methods, via the
    /// [`drive_shards`](crate::staging_driver::drive_shards) lifecycle.
    ///
    /// `indices` is the list of shard indices to process in order — either
    /// `0..n_csc_shards()` (for `for_each_gpu_csc_shard`) or the pre-filtered
    /// subset (for `for_each_gpu_csc_shard_in_range`).
    ///
    /// This used to hand-roll the whole thing: a `std::thread::scope` with a
    /// `sync_channel(1)`, so exactly **one** shard was decoded ahead where the
    /// row-major path has decoded four since Phase 4.2; a duplicated
    /// capacity-growth closure shadowing the method beside it; a single-shard
    /// fast path that recorded no event; no profiling; and a read failure
    /// classified as `CudaError` where `error.rs` says `InvalidShard`. All of
    /// that is gone — the ordering is the shared driver's and the decode is
    /// `scx_format_io::prefetch`'s.
    fn run_shards<F>(&mut self, indices: Vec<usize>, f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    {
        let plan = StagingPlan::selected(indices, self.prefetch_depth);
        let feeder = CscFeeder(ProfiledCscDecode(self.source));
        let mut stager = CscStaging {
            dev: self.dev,
            source: self.source,
            pinned: &mut self.pinned,
            pinned_events: &mut self.pinned_events,
            col_indptr: &mut self.col_indptr,
            row_indices: &mut self.row_indices,
            data: &mut self.data,
            col_indptr_cap: &mut self.col_indptr_cap,
            nnz_cap: &mut self.nnz_cap,
            memo: &mut self.memo,
            validation: self.validation,
            copy_stream: &self.copy_stream,
            n_obs: self.n_obs,
            f,
        };
        drive_shards(&feeder, &mut stager, &plan)
    }
}

/// `ColumnShardSource` adapter that times every decode into the GPU profiler's
/// host-decode bucket.
///
/// The column-major twin of `gpu_shard_source::ProfiledDecode`, and the reason
/// this path reports host-decode time at all: it previously made zero
/// `profile::` calls, so "GPU CSC DE is host-decode-bound" was not a claim the
/// profiler could confirm or refute.
struct ProfiledCscDecode<'a>(&'a (dyn ColumnShardSource + Sync));

impl ColumnShardSource for ProfiledCscDecode<'_> {
    fn n_csc_shards(&self) -> usize {
        self.0.n_csc_shards()
    }
    fn n_obs(&self) -> usize {
        self.0.n_obs()
    }
    fn n_vars(&self) -> usize {
        self.0.n_vars()
    }
    fn read_csc_shard(&self, shard_idx: usize) -> scx_format_io::Result<scx_sparse::ScxCsc> {
        let t_decode = profile::start();
        let out = self.0.read_csc_shard(shard_idx);
        profile::record_host_decode_since(CodecClass::Generic, t_decode);
        out
    }
    fn read_csc_columns(&self, col_range: Range<u32>) -> scx_format_io::Result<scx_sparse::ScxCsc> {
        self.0.read_csc_columns(col_range)
    }
    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
        self.0.csc_shard_col_range(shard_idx)
    }
    fn csc_shard_size_hint(&self) -> Option<scx_format_io::ShardSizeHint> {
        self.0.csc_shard_size_hint()
    }
    fn csc_shards_for_col_range(&self, col_range: Range<u32>) -> Vec<usize> {
        self.0.csc_shards_for_col_range(col_range)
    }
}

/// Column-major [`ShardFeeder`]: the bounded, ordered decode-prefetch over the
/// planned CSC shards.
struct CscFeeder<'a>(ProfiledCscDecode<'a>);

impl ShardFeeder for CscFeeder<'_> {
    type Shard = scx_sparse::ScxCsc;

    fn feed(
        &self,
        plan: &StagingPlan,
        consume: ShardConsumer<'_, Self::Shard>,
    ) -> Result<(), GpuError> {
        prefetch::for_each_csc_shard_ordered_selected(
            &self.0,
            &plan.indices,
            plan.depth,
            |i, csc| consume(i, &csc),
        )
    }
}

/// Column-major [`ShardStager`].
struct CscStaging<'r, F> {
    dev: &'r GpuDevice,
    /// Held for the O(1) `csc_shard_col_range` lookup: the view a consumer
    /// receives is addressed in **global** column space, and the decoded shard
    /// does not carry its own offset.
    source: &'r (dyn ColumnShardSource + Sync),
    pinned: &'r mut [PinnedCscSlot; 2],
    pinned_events: &'r mut [Option<CudaEvent>; 2],
    col_indptr: &'r mut CudaSlice<i64>,
    row_indices: &'r mut CudaSlice<i32>,
    data: &'r mut CudaSlice<f32>,
    col_indptr_cap: &'r mut usize,
    nnz_cap: &'r mut usize,
    memo: &'r mut ValidationMemo,
    validation: ValidationPolicy,
    copy_stream: &'r Arc<CudaStream>,
    n_obs: usize,
    f: F,
}

impl<F> CscStaging<'_, F> {
    /// This shard's global `[col_start, col_end)`.
    ///
    /// `InvalidShard`, not `CudaError`: a catalog that cannot say where a shard
    /// starts is a bad input, and `error.rs` reserves the runtime classes for
    /// device failures — which is what `decline_on_runtime_failure` keys on.
    fn col_range(&self, idx: usize) -> Result<(usize, usize), GpuError> {
        let (s, e) = self.source.csc_shard_col_range(idx).ok_or_else(|| {
            GpuError::InvalidShard(format!(
                "csc_shard_col_range({idx}) returned None: the CSC sidecar's catalog does not \
                 record this shard's column range, so its columns cannot be placed in the global \
                 gene axis"
            ))
        })?;
        Ok((s as usize, e as usize))
    }

    /// Grow the device buffers to fit `col_indptr_len` / `nnz`. No-op when the
    /// current capacity already suffices.
    fn ensure_device_capacity(
        &mut self,
        col_indptr_len: usize,
        nnz: usize,
    ) -> Result<(), GpuError> {
        if col_indptr_len > *self.col_indptr_cap {
            let new_cap = col_indptr_len.next_power_of_two().max(col_indptr_len);
            *self.col_indptr = self.dev.alloc_zeros::<i64>(new_cap)?;
            *self.col_indptr_cap = new_cap;
        }
        if nnz > *self.nnz_cap {
            let new_cap = nnz.next_power_of_two().max(nnz);
            *self.row_indices = self.dev.alloc_zeros::<i32>(new_cap)?;
            *self.data = self.dev.alloc_zeros::<f32>(new_cap)?;
            *self.nnz_cap = new_cap;
        }
        Ok(())
    }
}

impl<F> ShardStager for CscStaging<'_, F>
where
    F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
{
    type Shard = scx_sparse::ScxCsc;

    /// Nonzeros **and** columns: unlike the row-major side, an empty CSC shard
    /// carries no addressing the consumer needs, so skipping it is free.
    fn is_stageable(&self, shard: &Self::Shard) -> bool {
        !shard.data.is_empty() && shard.shape.1 != 0
    }

    fn validate(&mut self, idx: usize, shard: &Self::Shard) -> Result<(), GpuError> {
        let (col_start, _) = self.col_range(idx)?;
        validate_shard(
            ShardToValidate::Csc {
                csc: shard,
                n_obs: self.n_obs,
                col_start,
            },
            &self.validation,
        )
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
        let col_indptr_len = shard.indptr.len();
        let nnz = shard.data.len();
        self.ensure_device_capacity(col_indptr_len, nnz)?;
        let t_stage = profile::start();
        self.pinned[slot].stage(shard)?;
        profile::record_htod_since(CodecClass::Generic, t_stage, csc_htod_bytes(shard));
        self.pinned[slot].upload_to(
            self.copy_stream,
            self.col_indptr,
            self.row_indices,
            self.data,
            col_indptr_len,
            nnz,
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
        self.pinned_events[slot] = Some(upload_event);
        Ok(())
    }

    fn dispatch(&mut self, idx: usize, shard: &Self::Shard) -> Result<(), GpuError> {
        let (col_start, col_end) = self.col_range(idx)?;
        let col_indptr_len = shard.indptr.len();
        let nnz = shard.data.len();
        let view = GpuCscShardView {
            col_indptr: self.col_indptr.slice(..col_indptr_len),
            row_indices: self.row_indices.slice(..nnz),
            data: self.data.slice(..nnz),
            col_start,
            col_end,
            n_obs: self.n_obs,
        };
        (self.f)(idx, &view)
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

/// Bytes one CSC shard moves host->device: `indptr` i64 + `indices` i32 +
/// `data` f32. Mirrors `gpu_shard_source::csr_htod_bytes` (same three arrays,
/// major axis differs).
fn csc_htod_bytes(csc: &scx_sparse::ScxCsc) -> usize {
    csc.data.len() * 4 + csc.indices.len() * 4 + csc.indptr.len() * 8
}

impl<'a> GpuCscShardSource for RawGpuCscShardSource<'a> {
    fn n_csc_shards(&self) -> usize {
        self.source.n_csc_shards()
    }

    fn n_obs(&self) -> usize {
        self.n_obs
    }

    fn n_vars(&self) -> usize {
        self.n_vars
    }

    fn for_each_gpu_csc_shard<F>(&mut self, f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    {
        let indices: Vec<usize> = (0..self.source.n_csc_shards()).collect();
        self.run_shards(indices, f)
    }

    fn for_each_gpu_csc_shard_in_range<F>(
        &mut self,
        col_range: Range<u32>,
        f: F,
    ) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    {
        // Pre-filter shard indices so non-overlapping shards are never decoded
        // or uploaded. At smartseq2 / tabula scale with many CSC shards (13+)
        // and small gene chunks most shards skip, and eliminating decode + H→D
        // for them collapses cache thrashing (default `cache_shards=4`) and the
        // per-chunk fixed cost.
        //
        // The overlap predicate used to be spelled out here, making a fourth
        // copy of a rule that already existed on `FullCatalog` and on
        // `BackedCscIndex`. `ColumnShardSource::csc_shards_for_col_range` now
        // owns it: a backed reader answers by binary search, everyone else by
        // the trait's linear default, and an exhaustive differential test in
        // `scx-format-io` pins the two to the same boundary condition.
        self.run_shards(self.source.csc_shards_for_col_range(col_range), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_sparse::ScxCsc;

    /// In-memory `ColumnShardSource` stub: one `ScxCsc` per shard, each
    /// covering a precomputed `(col_start, col_end)` global column
    /// range. `read_csc_columns` is unimplemented because the pipelined
    /// driver only calls `read_csc_shard` + `csc_shard_col_range`.
    struct InMemoryCscSource {
        shards: Vec<ScxCsc>,
        ranges: Vec<(u32, u32)>,
        n_obs: usize,
        n_vars: usize,
        /// `None` reproduces a source that cannot answer cheaply — the state
        /// every `ColumnShardSource` was in before ORG-8.20-1.
        hint: Option<scx_format_io::ShardSizeHint>,
        /// Every shard index actually decoded. The range prefilter's whole
        /// purpose is that a non-overlapping shard never reaches here.
        reads: std::sync::Mutex<Vec<usize>>,
    }

    impl InMemoryCscSource {
        fn new(shards: Vec<ScxCsc>, ranges: Vec<(u32, u32)>, n_obs: usize, n_vars: usize) -> Self {
            Self {
                shards,
                ranges,
                n_obs,
                n_vars,
                hint: None,
                reads: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// Decoded shard indices, ascending. Sorted because decode order is
        /// concurrent and so nondeterministic — only the *set* is a property of
        /// the plan; delivery order is asserted separately.
        fn reads_sorted(&self) -> Vec<usize> {
            let mut v = self.reads.lock().unwrap().clone();
            v.sort_unstable();
            v
        }

        /// Attach the exact bounds of the shards held, as a real catalog would.
        fn hinted(mut self) -> Self {
            self.hint = Some(scx_format_io::ShardSizeHint {
                max_rows: self.shards.iter().map(|c| c.shape.1).max().unwrap_or(0),
                max_nnz: self.shards.iter().map(|c| c.data.len()).max().unwrap_or(0),
            });
            self
        }
    }

    impl ColumnShardSource for InMemoryCscSource {
        fn n_csc_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_csc_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsc> {
            self.reads.lock().unwrap().push(shard_idx);
            Ok(self.shards[shard_idx].clone())
        }
        fn read_csc_columns(
            &self,
            _col_range: std::ops::Range<u32>,
        ) -> scx_format_io::Result<ScxCsc> {
            unimplemented!("test stub does not implement read_csc_columns")
        }
        fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
            self.ranges.get(shard_idx).copied()
        }
        fn csc_shard_size_hint(&self) -> Option<scx_format_io::ShardSizeHint> {
            self.hint
        }
    }

    /// Build a tagged CSC shard with one nonzero per column at row
    /// `(col_local % n_obs)` and value `base + col_local`.
    fn make_csc(n_obs: usize, n_cols: usize, base: f32) -> ScxCsc {
        let mut indptr = vec![0i64];
        let mut indices = Vec::with_capacity(n_cols);
        let mut data = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            indices.push((c % n_obs) as i32);
            data.push(base + c as f32);
            indptr.push(indices.len() as i64);
        }
        ScxCsc::new_unchecked((n_obs, n_cols), indptr, indices, data)
    }

    /// `n_cols` columns holding rows `0..per_col` in strictly increasing
    /// order — enough nonzeros per column that a duplicate can be planted.
    fn make_dense_csc(n_obs: usize, n_cols: usize, per_col: usize) -> ScxCsc {
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for c in 0..n_cols {
            for r in 0..per_col {
                indices.push(r as i32);
                data.push((c * per_col + r) as f32 + 1.0);
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsc::new_unchecked((n_obs, n_cols), indptr, indices, data)
    }

    /// §8.3: the CSC staging path validated nothing, so a malformed sidecar
    /// reached the kernels — a NaN corrupting the radix sort, a duplicate
    /// racing one `slab` cell, an out-of-range row as an OOB device read.
    ///
    /// Each fixture must be rejected **before** anything is staged: the
    /// callback never fires, and the shard is not recorded as validated.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_csc_source_rejects_malformed_shards_before_staging() {
        let dev = require_gpu!();

        let nan = {
            let mut s = make_dense_csc(8, 3, 4);
            s.data[5] = f32::NAN;
            s
        };
        let duplicate = {
            let mut s = make_dense_csc(8, 3, 4);
            s.indices[5] = s.indices[4];
            s
        };
        let out_of_range = {
            let mut s = make_dense_csc(8, 3, 4);
            s.indices[5] = 8; // == n_obs
            s
        };

        for (label, shard) in [
            ("non-finite", nan),
            ("duplicate row", duplicate),
            ("out-of-range row", out_of_range),
        ] {
            let src = InMemoryCscSource::new(vec![shard], vec![(0, 3)], 8, 3);
            let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();
            let mut calls = 0usize;
            let err = gpu
                .for_each_gpu_csc_shard(|_, _| {
                    calls += 1;
                    Ok(())
                })
                .unwrap_err();
            assert!(
                matches!(err, GpuError::InvalidShard(_)),
                "{label}: expected InvalidShard, got {err:?}"
            );
            assert_eq!(calls, 0, "{label}: must be rejected before staging");
            assert_eq!(
                gpu.validated_count(),
                0,
                "{label}: a rejected shard must not be recorded as validated"
            );
        }
    }

    /// Every shard the source stages is validated — including on the
    /// multi-shard pipelined path, which is a separate call site from the
    /// single-shard fast path and was the one §8.3's absent check hid in.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_csc_source_validates_every_staged_shard() {
        let dev = require_gpu!();
        let src = InMemoryCscSource::new(
            vec![
                make_dense_csc(8, 3, 4),
                make_dense_csc(8, 4, 4),
                make_dense_csc(8, 3, 4),
            ],
            vec![(0, 3), (3, 7), (7, 10)],
            8,
            10,
        );
        let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();
        gpu.for_each_gpu_csc_shard(|_, _| Ok(())).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(
            gpu.validated_count(),
            3,
            "every staged shard must be scanned"
        );

        // A second pass re-yields all three; the per-shard record is what keeps
        // the O(nnz) scans from repeating once per gene chunk.
        gpu.for_each_gpu_csc_shard(|_, _| Ok(())).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(gpu.validated_count(), 3);
    }

    /// A malformed shard in the middle of a multi-shard source aborts the
    /// iteration rather than being staged.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_csc_source_rejects_malformed_shard_on_the_pipelined_path() {
        let dev = require_gpu!();
        let bad = {
            let mut s = make_dense_csc(8, 4, 4);
            s.data[2] = f32::NAN;
            s
        };
        let src = InMemoryCscSource::new(
            vec![make_dense_csc(8, 3, 4), bad, make_dense_csc(8, 3, 4)],
            vec![(0, 3), (3, 7), (7, 10)],
            8,
            10,
        );
        let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();
        let mut seen: Vec<usize> = Vec::new();
        let err = gpu
            .for_each_gpu_csc_shard(|idx, _| {
                seen.push(idx);
                Ok(())
            })
            .unwrap_err();
        assert!(
            matches!(err, GpuError::InvalidShard(_)),
            "expected InvalidShard, got {err:?}"
        );
        assert_eq!(seen, vec![0], "iteration must stop at the malformed shard");
    }

    /// `for_each_gpu_csc_shard_in_range` **decodes** only the shards whose
    /// column range overlaps, not merely delivers only those.
    ///
    /// There was no test for this at all before ORG-8.20-1 — on the path whose
    /// entire reason for prefiltering is that GPU DE drives it once per gene
    /// chunk (123× at census_500k) against a 13-shard sidecar. A version that
    /// decoded all thirteen and filtered on delivery would have produced
    /// identical numbers and thrown away the whole benefit, silently.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn csc_range_drive_decodes_only_overlapping_shards() {
        let dev = require_gpu!();
        // Four shards over [0,3) [3,7) [7,10) [10,14).
        let src = InMemoryCscSource::new(
            vec![
                make_csc(8, 3, 1.0),
                make_csc(8, 4, 100.0),
                make_csc(8, 3, 200.0),
                make_csc(8, 4, 300.0),
            ],
            vec![(0, 3), (3, 7), (7, 10), (10, 14)],
            8,
            14,
        );
        let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();

        let mut delivered: Vec<usize> = Vec::new();
        gpu.for_each_gpu_csc_shard_in_range(4..8, |idx, _view| {
            delivered.push(idx);
            Ok(())
        })
        .unwrap();

        // [4,8) meets shard 1 = [3,7) and shard 2 = [7,10); it does not meet
        // shard 0 (ends at 3) or shard 3 (starts at 10).
        assert_eq!(delivered, vec![1, 2]);
        assert_eq!(
            src.reads_sorted(),
            vec![1, 2],
            "a non-overlapping shard was decoded — the prefilter plans which \
             shards to read, it does not filter after reading them"
        );
    }

    /// The half-open boundary, on the surface a consumer actually calls.
    ///
    /// Hand-computed against the same four shards: a range that ends exactly
    /// where a shard begins excludes it, and one that begins exactly where a
    /// shard ends excludes that one.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn csc_range_drive_is_half_open_at_both_ends() {
        let dev = require_gpu!();
        let ranges = vec![(0u32, 3u32), (3, 7), (7, 10), (10, 14)];
        for (range, want) in [
            (0..3u32, vec![0usize]),
            (3..7, vec![1]),
            (2..4, vec![0, 1]),
            (0..14, vec![0, 1, 2, 3]),
            (10..14, vec![3]),
            (14..20, vec![]),
        ] {
            let src = InMemoryCscSource::new(
                vec![
                    make_csc(8, 3, 1.0),
                    make_csc(8, 4, 100.0),
                    make_csc(8, 3, 200.0),
                    make_csc(8, 4, 300.0),
                ],
                ranges.clone(),
                8,
                14,
            );
            let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();
            let mut delivered: Vec<usize> = Vec::new();
            gpu.for_each_gpu_csc_shard_in_range(range.clone(), |idx, _view| {
                delivered.push(idx);
                Ok(())
            })
            .unwrap();
            assert_eq!(delivered, want, "range {range:?}");
            assert_eq!(
                src.reads_sorted(),
                want,
                "range {range:?} decoded the wrong set"
            );
        }
    }

    /// A hinted source pre-sizes the device buffers at construction, so no
    /// shard grows-and-reallocs during the drive.
    ///
    /// The column-major twin of `hinted_source_presizes_the_staging_slot`.
    /// Before ORG-8.20-1 this path always started at `(1, 1)` and the doc said
    /// pre-sizing was impossible because "the trait doesn't expose"
    /// per-shard stats — the trait didn't, `BackedCscReader` did, and it read
    /// the `nnz` out of the same stats block it read the column range from and
    /// discarded it.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn hinted_csc_source_presizes_the_device_buffers() {
        let dev = require_gpu!();
        let src = InMemoryCscSource::new(
            vec![make_dense_csc(8, 3, 4), make_dense_csc(8, 5, 4)],
            vec![(0, 3), (3, 8)],
            8,
            8,
        )
        .hinted();

        let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();
        let at_construction = gpu.device_capacity();
        assert!(
            at_construction.0 >= 6 && at_construction.1 >= 20,
            "device buffers must be pre-sized for the largest shard (5 cols / 20 nnz), got \
             {at_construction:?}"
        );

        gpu.for_each_gpu_csc_shard(|_idx, _view| Ok(())).unwrap();
        assert_eq!(
            gpu.device_capacity(),
            at_construction,
            "pre-sized device buffers must not grow during the drive"
        );

        // The accept-side half: an unhinted source still works, it just starts
        // small. Without this the assertion above would pass on a build that
        // had made the hint mandatory.
        let unhinted = InMemoryCscSource::new(
            vec![make_dense_csc(8, 3, 4), make_dense_csc(8, 5, 4)],
            vec![(0, 3), (3, 8)],
            8,
            8,
        );
        let mut gpu2 = RawGpuCscShardSource::new(&dev, &unhinted).unwrap();
        assert_eq!(gpu2.device_capacity(), (1, 1));
        gpu2.for_each_gpu_csc_shard(|_idx, _view| Ok(())).unwrap();
    }

    /// A CSC source whose single shard carries a **different payload on every
    /// read**, tagged by a generation counter. Column-major twin of
    /// `GenerationalCsrSource`; see that type for why a fixed fixture makes the
    /// cross-drive test unable to fail.
    struct GenerationalCscSource {
        n_obs: usize,
        n_cols: usize,
        per_col: usize,
        generation: std::sync::atomic::AtomicUsize,
    }

    impl GenerationalCscSource {
        /// Offset added to every value on read `g`. Spaced wider than one
        /// shard's value span so no two generations share a value.
        fn offset_for(g: usize) -> f32 {
            (100_000 * (g + 1)) as f32
        }
        fn shard_for(&self, g: usize) -> ScxCsc {
            let mut csc = make_dense_csc(self.n_obs, self.n_cols, self.per_col);
            let off = Self::offset_for(g);
            for v in csc.data.iter_mut() {
                *v += off;
            }
            csc
        }
    }

    impl ColumnShardSource for GenerationalCscSource {
        fn n_csc_shards(&self) -> usize {
            1
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_cols
        }
        fn read_csc_shard(&self, _shard_idx: usize) -> scx_format_io::Result<ScxCsc> {
            let g = self
                .generation
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(self.shard_for(g))
        }
        fn read_csc_columns(
            &self,
            _col_range: std::ops::Range<u32>,
        ) -> scx_format_io::Result<ScxCsc> {
            unimplemented!("test stub does not implement read_csc_columns")
        }
        fn csc_shard_col_range(&self, _shard_idx: usize) -> Option<(u32, u32)> {
            Some((0, self.n_cols as u32))
        }
    }

    /// Driving one adapter repeatedly stages each drive's own bytes.
    ///
    /// Column-major twin of
    /// `driving_a_single_shard_csr_source_repeatedly_stages_own_bytes`; the
    /// reasoning, and the two properties that make it able to fail at all, are
    /// documented there.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn driving_a_single_shard_csc_source_repeatedly_stages_own_bytes() {
        let dev = require_gpu!();
        const N_OBS: usize = 4096;
        const N_COLS: usize = 4;
        const PER_COL: usize = 512;
        const DRIVES: usize = 8;
        const NNZ: usize = N_COLS * PER_COL;

        let src = GenerationalCscSource {
            n_obs: N_OBS,
            n_cols: N_COLS,
            per_col: PER_COL,
            generation: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();

        // Allocated up front: `cudaMalloc` is device-synchronising, so
        // allocating inside the loop would insert the barrier this test exists
        // to do without.
        let mut captures: Vec<cudarc::driver::safe::CudaSlice<f32>> = (0..DRIVES)
            .map(|_| dev.alloc_zeros::<f32>(NNZ).unwrap())
            .collect();

        for dst in captures.iter_mut() {
            gpu.for_each_gpu_csc_shard(|_idx, view| {
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
            let expected = GenerationalCscSource {
                n_obs: N_OBS,
                n_cols: N_COLS,
                per_col: PER_COL,
                generation: std::sync::atomic::AtomicUsize::new(0),
            }
            .shard_for(g)
            .data;
            assert_eq!(
                host, expected,
                "drive {g} did not stage its own generation — the pinned slot was rewritten \
                 while this drive's DMA was still reading it"
            );
        }
    }

    /// Pipelined source yields the staged shards verbatim — dtoh of
    /// the view's `data` returns each shard's tagged values.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_raw_csc_source_round_trip() {
        let dev = require_gpu!();
        let shards = vec![make_csc(8, 3, 1.0), make_csc(8, 4, 100.0)];
        let src = InMemoryCscSource::new(shards, vec![(0, 3), (3, 7)], 8, 7);
        let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();

        let mut seen: Vec<f32> = Vec::new();
        let mut col_ranges: Vec<(usize, usize)> = Vec::new();
        gpu.for_each_gpu_csc_shard(|_idx, view| {
            col_ranges.push((view.col_start, view.col_end));
            let mut host = vec![0.0f32; view.data.len()];
            dev.stream()
                .memcpy_dtoh(&view.data, &mut host)
                .map_err(|e| GpuError::CudaError(format!("dtoh: {e}")))?;
            seen.extend(host);
            Ok(())
        })
        .unwrap();
        dev.synchronize().unwrap();

        // Shard 0 = [1, 2, 3] cols, shard 1 = [100, 101, 102, 103] cols.
        assert_eq!(seen, vec![1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0]);
        assert_eq!(col_ranges, vec![(0, 3), (3, 7)]);
    }

    /// Regression test for the pinned-host-reuse race that the 2-slot
    /// pinned ring + host-side event sync fixes.
    ///
    /// Mirrors `test_multi_shard_pinned_no_corruption` in
    /// `gpu_shard_source.rs`. Pre-fix, `pinned.stage(&csc)` for shard
    /// `i+1` could overwrite the pinned source buffer while shard
    /// `i`'s `memcpy_htod_async` was still in flight on `copy_stream`.
    /// The device-side event handshake only orders streams against
    /// each other — it doesn't host-block the CPU writer. With pinned
    /// memory the H→D copy is truly async, so the corruption is
    /// observable as cross-shard bleed.
    ///
    /// The test stresses the bug by:
    ///   - Using ≥3 shards so the ring cycles at least once.
    ///   - Sizing each shard to ~10⁴ nnz so the DMA isn't trivially short.
    ///   - Using a **device-only** consumer (per-shard `memcpy_dtod`
    ///     into a private capture buffer on the compute stream) — no
    ///     per-callback host-blocking dtoh that would mask the race.
    ///   - Repeating across a small outer loop to amplify the window.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_multi_shard_csc_pinned_no_corruption() {
        use cudarc::driver::safe::CudaSlice;

        let dev = require_gpu!();
        let n_obs = 32usize;
        let cols_per_shard = 10_000usize; // ~10k cols × 1 nnz/col per shard
        let n_shards = 5usize;

        // Build per-shard CSCs with a per-shard tag value: shard `s`'s
        // data is `[s*1e6 + col]`. Any cross-shard bleed shows up
        // numerically.
        let make_tagged_csc = |shard: usize| -> ScxCsc {
            let mut indptr = Vec::with_capacity(cols_per_shard + 1);
            let mut indices = Vec::with_capacity(cols_per_shard);
            let mut data = Vec::with_capacity(cols_per_shard);
            indptr.push(0i64);
            for c in 0..cols_per_shard {
                indices.push((c % n_obs) as i32);
                data.push(shard as f32 * 1.0e6 + c as f32);
                indptr.push(indices.len() as i64);
            }
            ScxCsc::new_unchecked((n_obs, cols_per_shard), indptr, indices, data)
        };

        for repeat in 0..5 {
            let shards: Vec<ScxCsc> = (0..n_shards).map(make_tagged_csc).collect();
            let ranges: Vec<(u32, u32)> = (0..n_shards)
                .map(|s| {
                    (
                        (s * cols_per_shard) as u32,
                        ((s + 1) * cols_per_shard) as u32,
                    )
                })
                .collect();
            let src = InMemoryCscSource::new(shards, ranges, n_obs, n_shards * cols_per_shard);
            let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();

            // Per-shard device capture buffers. Filled via dtod on the
            // compute stream inside each callback — strictly
            // device-side, so no host-blocking mask of the race.
            let mut captures: Vec<CudaSlice<f32>> = (0..n_shards)
                .map(|_| dev.alloc_zeros::<f32>(cols_per_shard).unwrap())
                .collect();

            // Track which shard idx fills which capture (the source
            // yields in order, so capture index == shard index, but
            // assert via the view's col_start).
            gpu.for_each_gpu_csc_shard(|_idx, view| {
                assert_eq!(view.data.len(), cols_per_shard, "shard nnz mismatch");
                let shard = view.col_start / cols_per_shard;
                dev.stream()
                    .memcpy_dtod(&view.data, &mut captures[shard])
                    .map_err(|e| GpuError::CudaError(format!("dtod: {e}")))?;
                Ok(())
            })
            .unwrap();
            // Single boundary sync — everything queued on
            // compute_stream (the per-shard dtod copies) must complete
            // before we read the captures back to the host.
            dev.synchronize().unwrap();

            for (idx, capture) in captures.iter().enumerate() {
                let mut host = vec![0.0f32; cols_per_shard];
                dev.stream().memcpy_dtoh(capture, &mut host).unwrap();
                dev.synchronize().unwrap();
                for (c, &v) in host.iter().enumerate() {
                    let expected = idx as f32 * 1.0e6 + c as f32;
                    assert!(
                        (v - expected).abs() < 0.5,
                        "repeat {repeat}, shard {idx}, col {c}: got {v}, expected {expected} \
                         (pinned-host-reuse race?)"
                    );
                }
            }
        }
    }
}
