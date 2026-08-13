//! Release-active validation of decoded shards at the host-side GPU staging
//! boundary, shared by the CSR and CSC pipelines.
//!
//! The GPU shard kernels carry preconditions they cannot themselves enforce —
//! finite values, one writer per output cell, in-range cell ids. Each is
//! checked here on the host, before anything is staged or uploaded, and
//! reported as [`GpuError::InvalidShard`] rather than a `debug_assert!` that
//! vanishes in release builds.
//!
//! Both layouts go through the same three scanners so the two cannot drift:
//! `gpu_shard_source::validate_shard_for_gpu_de` for row-major CSR and
//! [`validate_csc_shard_for_gpu`] for the column-major
//! sidecar. Before this module existed only the CSR path validated, so the
//! CSC-direct DE route — the *default* whenever a CSC sidecar is present —
//! staged NaN, duplicate `(cell, gene)` pairs and out-of-range cell ids
//! straight into the kernels (review §8.3).
//!
//! ## Determinism
//!
//! Each scanner keeps the serial scan's **exact** answer, not merely the same
//! accept/reject decision: the structural scan reduces by minimum major-axis
//! index rather than taking whichever offender a worker reaches first, and the
//! flat scans use `position_first`. A first-hit early exit would make the error
//! message nondeterministic under load.
//!
//! Small shards scan serially: below [`VALIDATE_PAR_MIN_NNZ`] the rayon
//! split/join costs more than the scan.

use rayon::prelude::*;

use crate::error::GpuError;

/// Default nnz below which the scanners run serially. Real shards are orders of
/// magnitude above this (a census_500k shard carries ~24 M nnz); the threshold
/// exists so unit fixtures and degenerate single-row shards don't pay a pool
/// round-trip. Deliberately low enough that a test can exceed it with a
/// ~256 KB fixture and still exercise the parallel path.
pub(crate) const VALIDATE_PAR_MIN_NNZ: usize = 65_536;

/// Effective threshold, overridable by `SCX_GPU_VALIDATE_PAR_MIN_NNZ`.
///
/// Exists because the parallel scan is not free for every consumer. It runs on
/// the **consuming** thread, so on a decode-bound op it competes with the very
/// decode-prefetch workers that are feeding it — GPU DE wins (validation is on
/// its critical path now that each shard is decoded once) while GPU HVG, which
/// validates but gains nothing from residency, can only lose. Setting the knob
/// above any real shard's nnz restores the pre-4.5 serial scan **exactly**: the
/// serial arms below are the original code, unchanged, so this is a genuine
/// baseline rather than an "off" arm that means something new.
pub(crate) fn validate_par_min_nnz() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("SCX_GPU_VALIDATE_PAR_MIN_NNZ")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(VALIDATE_PAR_MIN_NNZ)
    })
}

/// First `(major, a, b)` whose major-axis group holds `a >= b` at adjacent
/// positions, minimised over the major axis.
///
/// `indptr` / `indices` are the raw CSR (row → columns) or CSC (column → rows)
/// arrays; `n_major` is `indptr.len() - 1`'s worth of groups the caller wants
/// scanned. Groups are scanned **independently** — `indices` is one flat array,
/// so a naive `windows(2)` over the whole of it would see the boundary pair
/// (last minor index of group `g`, first of group `g + 1`) and reject a
/// perfectly legal shard.
pub(crate) fn first_unsorted_major(
    indptr: &[i64],
    indices: &[i32],
    n_major: usize,
) -> Option<(usize, i32, i32)> {
    fn in_group(indptr: &[i64], indices: &[i32], m: usize) -> Option<(usize, i32, i32)> {
        let s = indptr[m] as usize;
        let e = indptr[m + 1] as usize;
        indices[s..e]
            .windows(2)
            .find(|w| w[0] >= w[1])
            .map(|w| (m, w[0], w[1]))
    }

    if indices.len() >= validate_par_min_nnz() {
        (0..n_major)
            .into_par_iter()
            .filter_map(|m| in_group(indptr, indices, m))
            .min_by_key(|(m, _, _)| *m)
    } else {
        (0..n_major).find_map(|m| in_group(indptr, indices, m))
    }
}

/// Position of the first index outside `[0, bound)`.
pub(crate) fn first_out_of_range(indices: &[i32], bound: usize) -> Option<usize> {
    let bad = |v: &i32| *v < 0 || *v as usize >= bound;
    if indices.len() >= validate_par_min_nnz() {
        indices.par_iter().position_first(bad)
    } else {
        indices.iter().position(bad)
    }
}

/// Position of the first non-finite (NaN / ±Inf) value.
pub(crate) fn first_non_finite(data: &[f32]) -> Option<usize> {
    if data.len() >= validate_par_min_nnz() {
        data.par_iter().position_first(|v| !v.is_finite())
    } else {
        data.iter().position(|v| !v.is_finite())
    }
}

/// Release-active validation of a CSC sidecar shard at the host-side GPU
/// staging boundary.
///
/// `n_obs` is the file-wide cell count the shard's **global** row indices are
/// numbered against; `col_start` is the shard's first global column, used only
/// so the error names the gene column a user can look up rather than a
/// shard-local offset.
///
/// Three invariants the CSC kernels require but cannot themselves enforce:
///
/// 1. **In-range row indices.** `csc_shard_pseudobulk_kernel`,
///    `csc_shard_pseudobulk_global_kernel` and `csc_shard_to_gene_major_kernel`
///    all read `cell_to_group[row_indices[e]]` — an out-of-range row is an
///    out-of-bounds device read, which on CUDA poisons the whole context.
///
///    This is **not** the primary guard against that: on any backed file the
///    shard decoder's `check_minor_indices` already bounds CSC row indices
///    against the shard header's `n_minor` (= `n_obs`) and refuses to hand one
///    back, and the kernels carry their own `cell` bound as a last line. What
///    this check adds is coverage of `ColumnShardSource` impls that do not
///    decode through that seam, and an actionable error rather than the
///    kernels' silent skip.
///
/// 2. **Strictly-increasing per-column row indices.**
///    `csc_shard_to_gene_major_kernel` writes `slab[gene, pos]` with one thread
///    per nonzero, so a duplicate `(cell, gene)` pair is two threads racing one
///    output cell with a nondeterministic winner.
///
///    Note this is **stricter than [`scx_sparse::ScxCsc`]'s own contract**,
///    which permits arbitrary row order within a column. The kernels need only
///    distinctness, but proving distinctness on unordered input costs a seen-set
///    over `n_obs`, whereas strict increase is one flat pass — and every sidecar
///    that can reach the GPU is written by `scx_sparse::transpose`, which lays
///    each column's rows down in strictly increasing order by construction. So
///    the check is a fail-closed false negative on a shape SCX never emits, the
///    same trade the CSR validator already makes.
///
/// 3. **Finite values.** `block_radix_sort_per_gene_kernel` pads with `+INF` and
///    sorts on the raw IEEE-754 bit pattern, so a NaN lands above `+INF` and
///    corrupts the U statistic and tie counts with no error. The CPU CSC DE path
///    rejects the same input (`scx_accel`'s `ensure_finite_values`).
pub(crate) fn validate_csc_shard_for_gpu(
    csc: &scx_sparse::ScxCsc,
    n_obs: usize,
    col_start: usize,
) -> Result<(), GpuError> {
    if let Some(pos) = first_out_of_range(&csc.indices, n_obs) {
        return Err(GpuError::InvalidShard(format!(
            "ScxCsc row index {} at nonzero index {pos} is outside [0, {n_obs}): GPU shard \
             kernels index the per-cell group and position tables with it directly, so an \
             out-of-range row is an out-of-bounds device read",
            csc.indices[pos]
        )));
    }

    let n_cols = csc.indptr.len().saturating_sub(1);
    if let Some((c, a, b)) = first_unsorted_major(&csc.indptr, &csc.indices, n_cols) {
        return Err(GpuError::InvalidShard(format!(
            "ScxCsc column {} has unsorted or duplicate row indices ({a} >= {b}): GPU shard \
             scatter requires strictly-increasing per-column row indices so every (gene, cell) \
             has exactly one writer (SCX-written CSC sidecars are strictly increasing by \
             construction)",
            col_start + c
        )));
    }

    if let Some(pos) = first_non_finite(&csc.data) {
        return Err(GpuError::InvalidShard(format!(
            "ScxCsc contains a non-finite value ({}) at nonzero index {pos}: GPU shard kernels \
             require finite input (NaN corrupts the radix sort; sanitise/QC before running)",
            csc.data[pos]
        )));
    }

    Ok(())
}

#[cfg(test)]
#[path = "shard_validate_tests.rs"]
mod tests;
