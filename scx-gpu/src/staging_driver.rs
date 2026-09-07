//! One staging driver for both device layouts.
//!
//! `RawGpuShardSource::run` (row-major CSR) and
//! `RawGpuCscShardSource::run_shards` (column-major CSC) each hand-rolled the
//! same seven-step shard lifecycle — skip empty, validate, host-wait the pinned
//! slot, stage + enqueue H→D, gate copy→compute, dispatch, gate compute→copy,
//! rotate — and the two copies drifted. The CSC one never gained the decode
//! prefetch, the host-RAM budget, the staging pre-size or the profiling hooks
//! that Phase 4.2 gave the CSR one, and it classified a host-side read failure
//! as `CudaError` where the crate's own rule (`error.rs`) says `InvalidShard`.
//!
//! [`drive_shards`] is that lifecycle written once. It contains **no CUDA call
//! and names no layout**: every device interaction is behind a
//! [`ShardStager`] method, and every source interaction is behind a
//! [`ShardFeeder`] method.
//!
//! # Why the seam is here and not somewhere more convenient
//!
//! `cargo test -p scx-gpu --lib` on a CPU host runs 82 tests and *ignores* 166.
//! Every test of the ring, both event handshakes, capacity growth, the
//! validation memo and the range prefilter is in the ignored set, because they
//! all open a device. A refactor of code that no reachable test exercises
//! cannot be watched red, which is the series' first ground rule — so the seam
//! has to put the ordering policy on one side of a trait boundary and the CUDA
//! calls on the other. `staging_driver_tests.rs` then drives the policy with a
//! recording fake, on CPU, with no `#[ignore]`.
//!
//! What the fake **cannot** prove stays with the GPU suite: that an event
//! really orders a DMA against a kernel, that pinned memory is page-locked,
//! that `upload_to` writes the right bytes, that the descriptor cache is
//! invalidated when pointers move, and that staged data is numerically
//! identical.

use scx_format_io::ShardSource;

use crate::error::GpuError;

/// Pinned staging slots in the ring. Two is enough to overlap one shard's DMA
/// with the previous shard's compute; a third buys nothing because the
/// compute→copy gate already serialises against the live device slot.
pub(crate) const RING: usize = 2;

/// A drive's shard list plus the decode-prefetch depth to run it at.
///
/// `indices` is always an explicit list, even on the CSR path where it is
/// exactly `0..n_shards`. An earlier revision had an `All(n)` / `Selected(v)`
/// enum so the dense sweep could skip the allocation; it was collapsed because
/// only one layout could ever produce each arm, which is coverage that reads as
/// real and is not. The cost is one `Vec<usize>` per drive — tens of `usize`
/// against a shard decode.
///
/// The list must be strictly ascending: "in plan order" is what the prefetch
/// pipeline preserves, so a shuffled list is delivered shuffled, not sorted.
#[derive(Debug, Clone)]
pub(crate) struct StagingPlan {
    /// Which shards, in what order.
    pub(crate) indices: Vec<usize>,
    /// Max shards decoded-but-unconsumed. Resolved by the caller from
    /// `resolve_staging_prefetch_depth_for`, which derates
    /// `SCX_ACCEL_PREFETCH_DEPTH` to `SCX_GPU_STAGING_MEMORY_BUDGET`.
    pub(crate) depth: usize,
}

impl StagingPlan {
    /// Plan covering every shard of a source, in order.
    pub(crate) fn all(n_shards: usize, depth: usize) -> Self {
        Self {
            indices: (0..n_shards).collect(),
            depth,
        }
    }

    /// Plan covering an explicit ascending subset.
    pub(crate) fn selected(indices: Vec<usize>, depth: usize) -> Self {
        Self { indices, depth }
    }

    /// Plan for a row-major source, honouring a row filter.
    ///
    /// A source that filters rows inside `read_shard` answers
    /// [`ShardSource::visible_shard_indices`] with the shards that still hold
    /// one; the rest would be decoded, uploaded and found empty. `StagingPlan`
    /// has to ask, because the driver cannot: it feeds through
    /// `for_each_shard_ordered_uncached_selected`, which deliberately does not
    /// consult the hook — second-guessing an explicit plan is how a staging
    /// path skips a shard it meant to stage. Asking here is what keeps the GPU
    /// DE routes in step with the CPU ones, which get the skip from the driver.
    ///
    /// `drive_shards` already skips an empty shard *after* decoding it; this
    /// removes the decode.
    pub(crate) fn for_source<S: ShardSource + ?Sized>(source: &S, depth: usize) -> Self {
        match source.visible_shard_indices() {
            Some(indices) => Self::selected(indices, depth),
            None => Self::all(source.n_shards(), depth),
        }
    }
}

/// Which shard indices a stager has already host-validated.
///
/// A source iterated once per gene chunk (GPU DE drives it 123× at
/// census_500k) would otherwise pay the O(nnz) validation scans once per chunk
/// rather than once per shard. Sound because a stager holds its source by
/// shared borrow for its whole lifetime: the bytes behind a given shard index
/// cannot change underneath it.
#[derive(Debug, Default)]
pub(crate) struct ValidationMemo {
    seen: Vec<bool>,
}

impl ValidationMemo {
    /// Memo over `n_shards` indices, none validated yet.
    pub(crate) fn new(n_shards: usize) -> Self {
        Self {
            seen: vec![false; n_shards],
        }
    }

    /// Whether `idx` has already passed validation in this stager's lifetime.
    ///
    /// An out-of-range index reports `false`, so it is re-validated rather than
    /// silently trusted — fail-closed on a memo that was sized against a
    /// different shard count.
    pub(crate) fn is_validated(&self, idx: usize) -> bool {
        self.seen.get(idx).copied().unwrap_or(false)
    }

    /// Record that `idx` passed. Called by the driver **after** a successful
    /// `validate`, never before: marking first would let a failing shard be
    /// accepted on the next pass.
    pub(crate) fn mark_validated(&mut self, idx: usize) {
        if let Some(v) = self.seen.get_mut(idx) {
            *v = true;
        }
    }

    /// How many indices have been validated. Test-only.
    #[cfg(test)]
    pub(crate) fn validated_count(&self) -> usize {
        self.seen.iter().filter(|v| **v).count()
    }
}

/// The driver's per-shard callback, as a feeder receives it.
///
/// Boxed rather than generic so [`ShardFeeder`] stays object-safe and the two
/// layout feeders can be held behind one type when a future entry-point
/// collapse wants that. One indirect call per shard, against a decode plus an
/// H→D upload plus a kernel launch.
pub(crate) type ShardConsumer<'a, S> = &'a mut dyn FnMut(usize, &S) -> Result<(), GpuError>;

/// Delivers a plan's decoded shards, in plan order, on the calling thread.
///
/// Split from [`ShardStager`] on the `&self` / `&mut self` line: the source and
/// the prefetch pipeline are shared and `Sync`, while the pinned ring and the
/// device buffers are exclusively owned. Both hand-rolled loops performed this
/// same split by hoisting field references out of `&mut self` before entering a
/// scope; here it is a type, so the compiler checks it.
pub(crate) trait ShardFeeder {
    /// The decoded host-side shard this layout carries (`ScxCsr` / `ScxCsc`).
    type Shard;

    /// Invoke `consume(shard_idx, shard)` for each planned shard, in plan
    /// order, on the calling thread.
    ///
    /// A read failure must surface as [`GpuError::InvalidShard`] — the host
    /// could not produce the shard, which `error.rs` classifies as a *bad
    /// input*, not a device runtime failure. Getting this wrong routes a
    /// corrupt-file diagnosis into the GPU→CPU fallback logic.
    fn feed(
        &self,
        plan: &StagingPlan,
        consume: ShardConsumer<'_, Self::Shard>,
    ) -> Result<(), GpuError>;
}

/// Every CUDA interaction one shard's staging needs, one method each.
///
/// The driver calls these in a fixed order and does nothing else; a layout impl
/// decides what each one means. Nothing here may reorder, skip or repeat a
/// step — that is the driver's job and the reason the policy is testable.
pub(crate) trait ShardStager {
    /// The decoded host-side shard this layout carries.
    type Shard;

    /// Whether this shard is worth staging.
    ///
    /// **Layout-specific on purpose.** CSR skips a shard with no rows; CSC
    /// skips one with no nonzeros or no columns. Unifying them to "no major
    /// entries or no nonzeros" would make CSR skip a rows-but-no-nonzeros
    /// shard, and `GpuPreprocessedShardSource` advances a `global_row_offset`
    /// per dispatched shard — so that shard's rows would vanish from every
    /// subsequent shard's addressing.
    fn is_stageable(&self, shard: &Self::Shard) -> bool;

    /// Release-active host validation of a shard the kernels are about to
    /// read. Runs at most once per shard index per stager, gated by the memo.
    fn validate(&mut self, idx: usize, shard: &Self::Shard) -> Result<(), GpuError>;

    /// The memo the driver gates `validate` on.
    fn memo(&mut self) -> &mut ValidationMemo;

    /// Host-wait for any DMA still reading pinned slot `slot`.
    ///
    /// The device-side gates below order device streams against each other;
    /// they do not stop the **CPU** from overwriting a DMA's source buffer.
    /// This is the only thing that does.
    fn host_wait(&mut self, slot: usize) -> Result<(), GpuError>;

    /// Grow device buffers to fit, copy the shard into pinned slot `slot`, and
    /// enqueue the H→D copy on the copy stream.
    fn stage_and_upload(&mut self, slot: usize, shard: &Self::Shard) -> Result<(), GpuError>;

    /// Record the upload event, make the compute stream wait on it, and stash
    /// it against `slot` so a later `host_wait(slot)` has something to wait on.
    fn gate_copy_to_compute(&mut self, slot: usize) -> Result<(), GpuError>;

    /// Hand the live device view to the consumer.
    fn dispatch(&mut self, idx: usize, shard: &Self::Shard) -> Result<(), GpuError>;

    /// Record the compute event and make the copy stream wait on it, so the
    /// next shard's upload cannot overwrite device memory this shard's kernels
    /// are still reading.
    fn gate_compute_to_copy(&mut self) -> Result<(), GpuError>;

    /// Host-wait every outstanding pinned event, so the caller may mutate or
    /// drop the pinned buffers the moment the drive returns.
    fn drain(&mut self) -> Result<(), GpuError>;
}

/// Run one plan's worth of shards through the staging lifecycle.
///
/// Order, per non-skipped shard:
///
/// 1. `is_stageable` — a skip consumes no slot and does **not** rotate the ring
/// 2. `validate`, if the memo has not already seen this index
/// 3. `host_wait(slot)`
/// 4. `stage_and_upload(slot, …)`
/// 5. `gate_copy_to_compute(slot)`
/// 6. `dispatch`
/// 7. `gate_compute_to_copy`
/// 8. rotate: `slot = (slot + 1) % RING`
///
/// Then `drain`, exactly once, on **both** the success and the error path.
///
/// # Two normalisations against the code this replaces
///
/// * **Validate before host-wait.** The CSR copy host-waited first, the CSC
///   copy validated first. Validating first is the survivor: a shard that is
///   going to be rejected should not first block the calling thread on an
///   unrelated DMA.
/// * **No single-shard fast path.** Both copies special-cased a one-shard drive
///   to skip the ring, the copy stream and the event records entirely, on the
///   stated grounds that "there is no successor `stage()` that could race the
///   in-flight DMA". That holds within one drive and not across two: driving
///   the same adapter twice re-enters with `pinned[0]` still the source of an
///   un-awaited `memcpy_htod_async`, and nothing on the host has synchronised.
///   Latent today only because every real consumer downloads a result, which
///   syncs. The uniform path costs one event record plus one wait per drive.
pub(crate) fn drive_shards<F, S>(
    feeder: &F,
    stager: &mut S,
    plan: &StagingPlan,
) -> Result<(), GpuError>
where
    F: ShardFeeder,
    S: ShardStager<Shard = F::Shard>,
{
    if plan.indices.is_empty() {
        return Ok(());
    }

    let mut slot = 0usize;
    let feed_result = feeder.feed(plan, &mut |idx, shard| {
        if !stager.is_stageable(shard) {
            return Ok(());
        }

        if !stager.memo().is_validated(idx) {
            stager.validate(idx, shard)?;
            stager.memo().mark_validated(idx);
        }

        stager.host_wait(slot)?;
        stager.stage_and_upload(slot, shard)?;
        stager.gate_copy_to_compute(slot)?;
        stager.dispatch(idx, shard)?;
        stager.gate_compute_to_copy()?;

        slot = (slot + 1) % RING;
        Ok(())
    });

    // Drain on both paths: an error that leaves a DMA in flight would let the
    // caller drop the pinned buffers underneath it. A drain error does not
    // mask the original failure.
    let drain_result = stager.drain();
    feed_result.and(drain_result)
}

#[cfg(test)]
#[path = "staging_driver_tests.rs"]
mod tests;
