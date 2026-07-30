//! Device-resident CSR shard retention for multi-pass GPU consumers (§9.11).
//!
//! # The problem
//!
//! GPU DE's CSR route has no column-range prefilter, so it runs a **full**
//! `for_each_gpu_csr_shard` pass per gene chunk. Cost is
//! `n_gene_chunks × n_shards` host decodes + H→D uploads — the same quadratic
//! Phase 2.2 removed from the CPU path by defaulting to the CSC-direct route,
//! but `gpu_csr_v3` is the mandatory route for every file **without** a CSC
//! sidecar. Measured at census_500k (61 497 genes ÷ a 500-gene chunk = 123
//! chunks over 31 shards): 1 005.6 s of host decode against a 1 019.7 s wall,
//! i.e. **98.6 % of the op is re-decoding data it already decoded**.
//!
//! # The fix
//!
//! [`ResidentGpuCsrSource`] drains an inner [`GpuMatrixSource`] **once**,
//! retaining every shard in its own device-resident [`GpuCsrSlot`], and then
//! satisfies each subsequent `for_each_gpu_csr_shard` call from VRAM. Decode
//! and H→D collapse from `n_chunks × n_shards` to `n_shards`.
//!
//! # Why per-shard retention rather than one concatenated CSR
//!
//! Two builders in this crate already concatenate shards into a single
//! [`GpuCsr`](crate::shard_decode::GpuCsr):
//! [`decode_csr_shards_to_device`](crate::gpu_csr_assemble::decode_csr_shards_to_device)
//! (the `to_gpu_anndata` handoff) and `gpu_pca_resident::try_build_resident_csr`
//! (the PCA power loop). Both do so because their consumer needs **one cuSPARSE
//! descriptor over the whole matrix**.
//!
//! GPU DE does not. Its kernels take a per-shard `GpuCsrShardView` plus a
//! `global_row` offset, so concatenation would buy nothing and cost something:
//! collapsing 31 shards into one 500 000-row "shard" changes every kernel's
//! grid shape and, for the f64 `atomicAdd` pseudobulk fold, the accumulation
//! interleaving. Retaining per shard keeps the callback's view of the world
//! **byte-for-byte what it was** — same shard indices, same shapes, same launch
//! geometry, same arguments — so residency is a pure "where did these bytes
//! come from" change rather than a numerical one. It also avoids needing a
//! total-nnz hint up front, an `i32::MAX` nnz ceiling on the concatenated row
//! offsets, and a single multi-GB allocation.
//!
//! Total VRAM is identical either way: one f32 value + one i32 index per
//! nonzero, ~6 GB for census_500k's 747 M nnz.
//!
//! # Budget
//!
//! Residency is refused (returning `None`, so the caller streams exactly as
//! before) when the retained set would exceed `max_frac` of *free* device
//! memory. The budget is re-checked before each shard is retained, so an
//! over-budget matrix aborts the drain at the offending shard instead of
//! OOMing — it never retains more than it checked for. Callers pass a fraction
//! below 1.0 because they still need VRAM for their own scratch; GPU DE leaves
//! half the card.

use std::ops::Range;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_csc_shard_source::GpuCscShardView;
use crate::gpu_matrix_source::{GpuMatrixSource, GpuTransformSpec, LayoutSet, SourceRouteMetadata};
use crate::staging::GpuCsrSlot;

/// Default fraction of free VRAM a resident CSR may occupy. Half the card
/// leaves the DE per-chunk scratch (`GpuDeChunkScratch`, whose own budget
/// clamp adapts to what is left) room to size itself.
pub const DEFAULT_RESIDENT_MAX_FRAC: f64 = 0.5;

/// A [`GpuMatrixSource`] whose CSR shards are already on the device.
///
/// Built by [`try_build_resident`]. Replays the retained shards in their
/// original order with their original shard indices; every call is a pure VRAM
/// read, with no host decode, no pinned staging and no H→D copy.
///
/// # Consumers must not mutate the slot
///
/// The callback receives `&mut GpuCsrSlot` — the signature the trait requires
/// so a streaming source can hand over its reusable staging slot — but here
/// the slot is the **retained** shard, not a scratch buffer refilled on the
/// next iteration. An in-place transform (the shape
/// [`GpuPreprocessedShardSource`](crate::gpu_shard_source::GpuPreprocessedShardSource)
/// uses) would persist into every later pass and compound: normalize applied
/// once per gene chunk instead of once.
///
/// Today's consumers are the GPU DE CSR drivers, which read through
/// [`GpuCsrSlot::view`] and never write. A source's *own* transforms are safe
/// and correct — they run during the drain, so the retained bytes are already
/// post-transform — but a consumer that mutates must not be handed a resident
/// source.
pub struct ResidentGpuCsrSource {
    /// `(original shard index, retained device CSR)`, in iteration order.
    /// Empty shards are absent — the streaming driver skips them too, so the
    /// callback sees the identical subsequence.
    slots: Vec<(usize, GpuCsrSlot)>,
    shape: (usize, usize),
    /// Mirrored from the inner source: the transforms were applied **during**
    /// the drain, so the retained bytes are post-transform and a downstream
    /// consumer must see the same spec it would have seen while streaming.
    transforms: GpuTransformSpec,
    /// Mirrored from the inner source so route stamping is unchanged by
    /// residency (in particular `csc_available`, which the resident source
    /// cannot serve but which the planner already decided on).
    route_metadata: SourceRouteMetadata,
    /// Total device bytes retained. Surfaced for capture/diagnostics.
    resident_bytes: u64,
}

impl ResidentGpuCsrSource {
    /// Number of retained (non-empty) shards.
    pub fn n_retained_shards(&self) -> usize {
        self.slots.len()
    }

    /// Total device bytes held by the retained shards.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

impl GpuMatrixSource for ResidentGpuCsrSource {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn available_layouts(&self) -> LayoutSet {
        // CSR only: the drain went through the CSR iterator, so a CSC sidecar
        // the inner source may have had is *not* retained. Callers wrap only
        // after the route planner has already chosen CSR.
        LayoutSet::CSR
    }

    fn transforms(&self) -> GpuTransformSpec {
        self.transforms.clone()
    }

    fn route_metadata(&self) -> SourceRouteMetadata {
        self.route_metadata.clone()
    }

    fn for_each_gpu_csr_shard(
        &mut self,
        f: &mut dyn FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        for (idx, slot) in self.slots.iter_mut() {
            f(*idx, slot)?;
        }
        Ok(())
    }

    fn for_each_gpu_csc_shard_in_range(
        &mut self,
        _col_range: Range<u32>,
        _f: &mut dyn FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        Err(GpuError::UnsupportedLayout(
            "ResidentGpuCsrSource retains CSR shards only".to_string(),
        ))
    }
}

/// Marker text for the internal "stop draining, the budget is spent" signal.
///
/// `GpuMatrixSource`'s callback can only stop iteration by returning `Err`, so
/// the budget check raises a sentinel that [`try_build_resident`] recognises
/// and converts into a plain `Ok(None)` decline. Deliberately unmistakable —
/// it must never be confused with a genuine device OOM, which the caller has
/// to see.
const BUDGET_ABORT: &str = "__scx_resident_csr_budget_abort__";

fn budget_abort() -> GpuError {
    GpuError::OutOfMemory(BUDGET_ABORT.to_string())
}

fn is_budget_abort(e: &GpuError) -> bool {
    matches!(e, GpuError::OutOfMemory(msg) if msg == BUDGET_ABORT)
}

/// Drain `inner`'s CSR shards once and retain them on the device.
///
/// Returns `Ok(None)` when residency is declined — the caller then uses `inner`
/// unchanged and behaviour is bit-for-bit what it was. Declined when:
///
/// - the source has no CSR layout, or is empty;
/// - the retained set would exceed `max_frac` of free device memory (re-checked
///   before every shard, and aborting the drain the moment it would be).
///
/// A decline costs only the shards drained before the budget was hit, not a
/// full pass. Sizing it exactly would need a total-nnz pass up front — which is
/// the very cost this type exists to remove.
///
/// # Errors
/// Propagates decode / CUDA errors from the inner source's drain, and any
/// device allocation failure while retaining a shard.
pub fn try_build_resident(
    dev: &GpuDevice,
    inner: &mut dyn GpuMatrixSource,
    max_frac: f64,
) -> Result<Option<ResidentGpuCsrSource>, GpuError> {
    if !inner.available_layouts().contains(LayoutSet::CSR) {
        return Ok(None);
    }
    let shape = inner.shape();
    if shape.0 == 0 || shape.1 == 0 {
        return Ok(None);
    }

    let (free, _total) = dev.free_memory()?;
    let budget = (free as f64 * max_frac.clamp(0.0, 1.0)) as u64;
    if budget == 0 {
        return Ok(None);
    }

    let transforms = inner.transforms();
    let route_metadata = inner.route_metadata();

    let mut slots: Vec<(usize, GpuCsrSlot)> = Vec::new();
    let mut resident_bytes: u64 = 0;

    let drain = inner.for_each_gpu_csr_shard(&mut |idx, slot| {
        let (n_rows, _) = slot.shape();
        if n_rows == 0 {
            return Ok(());
        }
        // What `clone_exact` will allocate for this shard, spelled the same way
        // it spells it — including the `max(1)` floors, so the prediction the
        // budget is checked against and the `device_bytes()` accumulated
        // afterwards are the *same number* rather than two estimates that drift
        // by a few bytes on a rows-but-no-nonzeros shard. The check has to
        // predict (there is nothing to measure until it allocates), so the only
        // way to keep the two honest is to derive them identically.
        let want = (n_rows as u64 + 1) * 8 + (slot.nnz().max(1) as u64) * 8;
        if resident_bytes.saturating_add(want) > budget {
            // Abort the drain rather than finish a pass whose results we are
            // about to discard: a matrix far too large for the budget would
            // otherwise pay a full useless decode before the caller streams.
            // `RawGpuShardSource::run` drains its pinned events before
            // propagating this, so the inner source stays reusable.
            return Err(budget_abort());
        }
        let retained = slot.clone_exact(dev)?;
        debug_assert_eq!(
            retained.device_bytes(),
            want,
            "budget prediction and actual retained bytes must agree"
        );
        resident_bytes += retained.device_bytes();
        slots.push((idx, retained));
        Ok(())
    });
    match drain {
        Ok(()) => {}
        Err(e) if is_budget_abort(&e) => {
            // Dropping the slots returns the retained memory to cudarc's CUDA
            // memory pool — but the pool keeps it charged to the process until
            // trimmed, so `cuMemGetInfo` would still report it as used. The
            // caller's very next act is to size its per-chunk scratch against
            // free VRAM, and on this path it has *no* residency to release in
            // exchange, so an untrimmed pool would shrink its chunk (or fail
            // the budget outright) to pay for memory nothing is holding.
            drop(slots);
            dev.reclaim_memory_pool()?;
            return Ok(None);
        }
        Err(e) => {
            // Same reasoning as the budget abort, for a genuine mid-drain
            // failure (a `clone_exact` OOM, a CUDA fault): whatever was
            // retained before the failure has to come off the process's books,
            // or the caller's next free-VRAM probe pays for it. Best-effort and
            // deliberately not `?` — a failing trim must never replace the
            // error the caller actually needs to see. `GpuDevice::Drop` would
            // eventually trim anyway, but only once the per-op device unwinds
            // several frames up; doing it here keeps the two failure paths
            // symmetric rather than one relying on a distant destructor.
            drop(slots);
            let _ = dev.reclaim_memory_pool();
            return Err(e);
        }
    }

    if slots.is_empty() {
        return Ok(None);
    }

    // The retain copies are queued on the compute stream, as is every
    // downstream kernel that will read them, so ordering is already
    // guaranteed. Synchronize anyway: the drain is a one-off and a caller
    // sampling free VRAM (or deciding a chunk budget) immediately after must
    // see the allocations settled.
    dev.synchronize()?;

    Ok(Some(ResidentGpuCsrSource {
        slots,
        shape,
        transforms,
        route_metadata,
        resident_bytes,
    }))
}

#[cfg(test)]
#[path = "resident_gpu_csr_source_tests.rs"]
mod tests;
