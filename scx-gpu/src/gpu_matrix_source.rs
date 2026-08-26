//! Unified device-side sparse matrix source.
//!
//! [`GpuShardSource`](crate::gpu_shard_source::GpuShardSource) (row-major CSR)
//! and [`GpuCscShardSource`](crate::gpu_csc_shard_source::GpuCscShardSource)
//! (column-major CSC) are two parallel, un-unified iteration contracts: a
//! consumer that wants "DE on whatever layout this source can provide" has to
//! know the concrete source type and branch on it. `GpuMatrixSource` composes
//! the two into a single capability surface — a consumer queries
//! [`available_layouts`](GpuMatrixSource::available_layouts) and calls the
//! matching iterator, without downcasting.
//!
//! This trait does **not** replace the low-level traits — concrete impls wrap a
//! [`RawGpuShardSource`](crate::gpu_shard_source::RawGpuShardSource) and/or
//! [`RawGpuCscShardSource`](crate::gpu_csc_shard_source::RawGpuCscShardSource)
//! and delegate. The `&mut GpuCsrSlot` (CSR) vs `&GpuCscShardView` (CSC)
//! callback asymmetry is preserved — unifying them into one enum view would
//! force every callback to pattern-match and lose the cuSPARSE descriptor cache
//! on the CSR path.
//!
//! The iteration methods take **boxed** callbacks (`&mut dyn FnMut`) rather than
//! generic `<F>` so the trait is object-safe: a future entry-point collapse can
//! dispatch over `&mut dyn GpuMatrixSource` without redefining the trait. The
//! cost is one indirect call per shard — negligible against per-shard decode +
//! H→D upload + kernel launch.

use std::ops::Range;

use crate::error::GpuError;
use crate::gpu_csc_shard_source::GpuCscShardView;
use crate::staging::GpuCsrSlot;

/// Set of device-format capabilities a [`GpuMatrixSource`] can provide.
///
/// Hand-rolled bitset (no `bitflags` dependency) — only two flags exist today.
/// Extensible: a future `DENSE` / `COO` flag is another `const`.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct LayoutSet(u8);

impl LayoutSet {
    /// Empty set (no layouts).
    pub const EMPTY: Self = Self(0);
    /// Row-major CSR shards (via [`GpuMatrixSource::for_each_gpu_csr_shard`]).
    pub const CSR: Self = Self(0b0001);
    /// Column-major CSC shards (via
    /// [`GpuMatrixSource::for_each_gpu_csc_shard_in_range`]).
    pub const CSC: Self = Self(0b0010);

    /// True when `self` contains every flag in `other`.
    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Union of two flag sets.
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// True when no layout is available.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for LayoutSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

/// Which host-side shard checks a consumer needs, as three **independent**
/// switches.
///
/// Independent, not a ladder. An ordered `Bounds < Scatter < Ranking` ladder was
/// the first shape of this and it was wrong, because the requirements do not
/// nest: HVG's clipped reducers need **finite without sorted**, while PCA and
/// pseudobulk need **sorted without finite**. Under a cumulative ladder the only
/// way to ask for finiteness was to also demand sortedness, which made GPU HVG
/// reject an unsorted `scipy.sparse.csr_matrix` that its own kernels handle fine
/// and that the CPU path accepts (found by codex - gpt-5.6-sol).
///
/// Validation is not free — each enabled check is an O(nnz) scan on the
/// consuming thread, competing with the decode-prefetch workers feeding it — so
/// asking for a check a kernel does not need costs throughput, and asking for
/// one it does need is a silent corruption. Each field below names the kernel
/// hazard it removes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ValidationChecks {
    /// Minor-axis indices are in range.
    ///
    /// **CSC only** — the row-major validator has never range-checked column
    /// indices, because the CSR kernels bound the column window themselves and
    /// the shard decoder already rejects an out-of-range minor index. So this
    /// switch is a no-op on CSR, and a CSR consumer that enables only this runs
    /// no scan at all.
    ///
    /// On CSC it matters: `csc_shard_pseudobulk_kernel`,
    /// `csc_shard_pseudobulk_global_kernel` and `csc_shard_to_gene_major_kernel`
    /// index `cell_to_group[row_indices[e]]` directly, so an out-of-range row is
    /// an out-of-bounds device read — which on CUDA poisons the whole context.
    pub in_range: bool,

    /// Major-axis indices are strictly increasing (hence unique).
    ///
    /// Three distinct kernels need this and none of them rank:
    /// `csc_shard_to_gene_major_kernel` writes `slab[gene, pos]` with one thread
    /// per nonzero, so a duplicate pair is two threads racing one cell;
    /// cuSPARSE SpMM (streaming PCA) is undefined on unsorted column indices;
    /// and `csr_shard_pseudobulk_kernel` narrows each row with
    /// `scx_row_lower_bound`, a binary search.
    pub sorted: bool,

    /// Values are finite.
    ///
    /// Two independent ways a non-finite value corrupts a result silently:
    /// `block_radix_sort_per_gene_kernel` pads with `+INF` and sorts on the raw
    /// IEEE-754 bit pattern, so a NaN lands above `+INF`; and the HVG clip
    /// kernels evaluate `v > cv ? cv : v`, so `+Inf` compares true and is
    /// **replaced by the clip value** before accumulation, which no check on the
    /// output can detect.
    pub finite: bool,
}

impl ValidationChecks {
    /// Everything. The fail-closed default.
    pub const ALL: Self = Self {
        in_range: true,
        sorted: true,
        finite: true,
    };

    /// In-range only — the floor. On CSR this runs no scan.
    ///
    /// For kernels that neither index by a bound table nor rank nor binary
    /// search: the fused preprocessing rewrites, and the HVG mean/variance
    /// reducers whose non-finite input is caught after accumulation instead.
    pub const IN_RANGE: Self = Self {
        in_range: true,
        sorted: false,
        finite: false,
    };

    /// In-range and sorted, **without** the finiteness scan.
    ///
    /// Streaming PCA and pseudobulk: their kernels require ordered indices but
    /// propagate a NaN exactly as the CPU would.
    pub const SORTED: Self = Self {
        in_range: true,
        sorted: true,
        finite: false,
    };

    /// In-range and finite, **without** the sortedness scan.
    ///
    /// The combination a ladder could not express, and the reason this is a set:
    /// HVG's clipped reducers and its batched mean/variance must reject a
    /// non-finite value, and accept whatever index order the caller had.
    pub const FINITE: Self = Self {
        in_range: true,
        sorted: false,
        finite: true,
    };
}

impl Default for ValidationChecks {
    fn default() -> Self {
        Self::ALL
    }
}

/// A consumer's validation checks plus the operation name its errors speak in.
///
/// `op` is the **user-facing** operation — `"normalize_total"`, `"pca"`,
/// `"rank_genes_groups"` — not the kernel or the crate function. It is
/// interpolated into every message the scanners raise, which is the textual half
/// of the §8.14 fix; the structural half is that a consumer never runs a check
/// it did not ask for. Either alone is fragile: a message can be reworded back,
/// and a check can be re-enabled back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ValidationPolicy {
    /// Which checks to run.
    pub checks: ValidationChecks,
    /// What to call the operation in an error.
    pub op: &'static str,
}

impl ValidationPolicy {
    /// A policy running `checks`, speaking as `op`.
    pub fn new(checks: ValidationChecks, op: &'static str) -> Self {
        Self { checks, op }
    }
}

impl Default for ValidationPolicy {
    /// Every check, attributed to no particular op.
    ///
    /// The pre-§8.14 behaviour and the fail-closed default: a source built
    /// without an explicit policy validates everything.
    fn default() -> Self {
        Self {
            checks: ValidationChecks::ALL,
            op: "this GPU operation",
        }
    }
}

/// Record of which transforms a source applies on the device before yielding
/// shard data.
///
/// Distinct from the CPU-side lazy transform chain: this records what the GPU
/// source **actually applies**, so a consumer that needs untransformed data can
/// detect a mismatch (e.g. a source with `log1p == true` handed to a kernel
/// that would log1p again).
#[derive(Clone, Default, Debug, PartialEq)]
pub struct GpuTransformSpec {
    /// `None` = no normalization; `Some(target)` = per-row `normalize_total`.
    pub normalize: Option<f32>,
    /// Whether `log(1 + x)` is applied.
    pub log1p: bool,
    /// Whether per-row scaling is applied (not yet supported on GPU).
    pub row_scale: bool,
}

/// Provenance metadata so a route planner / result stamping can record what ran
/// without downcasting to a concrete source type.
///
/// Deliberately **omits** `scx_accel::route::InputLayout`: `scx-accel` depends
/// on `scx-gpu`, so referencing that enum here would be a circular dependency.
/// The concrete-source → `InputLayout` mapping is done in `scx-accel` (where the
/// Python input type is known) when the planner is wired to consume a source.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct SourceRouteMetadata {
    /// Whether a CSC sidecar was available at source construction time.
    pub csc_available: bool,
    /// Total observation count (rows) at the source level.
    pub source_n_obs: usize,
    /// Total variable count (columns) at the source level.
    pub source_n_vars: usize,
}

/// Unified device-side sparse matrix source.
///
/// Consumers query [`available_layouts`](Self::available_layouts) to decide
/// which iterator to call. The CSR / CSC iterators default to returning
/// [`GpuError::UnsupportedLayout`] so a source that lacks a layout errors
/// cleanly rather than panics; impls override only the layouts they provide.
pub trait GpuMatrixSource {
    /// `(n_obs, n_vars)`.
    fn shape(&self) -> (usize, usize);

    /// Which device-format iterators this source supports.
    fn available_layouts(&self) -> LayoutSet;

    /// Transforms the source applies before yielding device data.
    fn transforms(&self) -> GpuTransformSpec {
        GpuTransformSpec::default()
    }

    /// Provenance metadata for route stamping without downcasting.
    fn route_metadata(&self) -> SourceRouteMetadata {
        let (n_obs, n_vars) = self.shape();
        SourceRouteMetadata {
            csc_available: self.available_layouts().contains(LayoutSet::CSC),
            source_n_obs: n_obs,
            source_n_vars: n_vars,
        }
    }

    /// Iterate CSR shards on device, invoking `f` with `(shard_idx, &mut
    /// GpuCsrSlot)` positioned at the live shard. Returns
    /// [`GpuError::UnsupportedLayout`] unless `available_layouts()` contains
    /// [`LayoutSet::CSR`].
    fn for_each_gpu_csr_shard(
        &mut self,
        _f: &mut dyn FnMut(usize, &mut GpuCsrSlot) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        Err(GpuError::UnsupportedLayout(
            "source does not provide a CSR layout".to_string(),
        ))
    }

    /// Iterate CSC shards overlapping `col_range` on device, invoking `f` with
    /// `(shard_idx, &GpuCscShardView)`. Returns [`GpuError::UnsupportedLayout`]
    /// unless `available_layouts()` contains [`LayoutSet::CSC`].
    fn for_each_gpu_csc_shard_in_range(
        &mut self,
        _col_range: Range<u32>,
        _f: &mut dyn FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        Err(GpuError::UnsupportedLayout(
            "source does not provide a CSC layout".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_set_contains_and_union() {
        let csr = LayoutSet::CSR;
        let both = LayoutSet::CSR | LayoutSet::CSC;
        assert!(csr.contains(LayoutSet::CSR));
        assert!(!csr.contains(LayoutSet::CSC));
        assert!(both.contains(LayoutSet::CSR));
        assert!(both.contains(LayoutSet::CSC));
        assert!(both.contains(LayoutSet::CSR | LayoutSet::CSC));
        // CSR-only does not contain the {CSR,CSC} set.
        assert!(!csr.contains(both));
    }

    #[test]
    fn layout_set_empty_default() {
        assert_eq!(LayoutSet::default(), LayoutSet::EMPTY);
        assert!(LayoutSet::EMPTY.is_empty());
        assert!(!LayoutSet::CSR.is_empty());
        // Empty is contained in everything; everything contains empty.
        assert!(LayoutSet::CSR.contains(LayoutSet::EMPTY));
    }

    #[test]
    fn transform_spec_default_is_identity() {
        let t = GpuTransformSpec::default();
        assert_eq!(t.normalize, None);
        assert!(!t.log1p);
        assert!(!t.row_scale);
    }

    #[test]
    fn route_metadata_default() {
        let m = SourceRouteMetadata::default();
        assert!(!m.csc_available);
        assert_eq!(m.source_n_obs, 0);
        assert_eq!(m.source_n_vars, 0);
    }
}
