//! GPU pseudobulk group-means for perturbation-evaluation metrics.
//!
//! Thin wrappers around the DE-path pseudobulk primitives
//! ([`gpu_de_pseudobulk_csr_direct`], [`gpu_de_pseudobulk_all_groups`]) that
//! produce per-group arithmetic means `[n_groups × n_cols]` (row-major f64).
//! The GPU accumulates per-`(group, col)` sums in **f64** with the identity
//! pre-transform (`mode_id = 0`); the host divides by per-group cell counts.
//!
//! This mirrors the CPU `pseudobulk_aggregate*` (`AggregationMethod::Mean`)
//! contract exactly — same f64 sum → divide-by-count means — so the downstream
//! host bulk-metric math (`scx_accel::compute_bulk_metrics`) is unchanged and
//! GPU/CPU parity is bounded only by f64 atomic-add ordering.

use scx_format_io::ShardSource;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_diffexp::{gpu_de_pseudobulk_all_groups, gpu_de_pseudobulk_csr_direct};
use crate::gpu_matrix_source::{ValidationChecks, ValidationPolicy};
use crate::gpu_shard_source::{GpuShardSource, RawGpuShardSource};

/// Divide row-major `[n_groups × n_cols]` f64 sums in place by per-group cell
/// counts, turning group sums into arithmetic means. Groups with a zero count
/// are left as-is (all-zero rows).
fn divide_by_counts(sums: &mut [f64], n_groups: usize, n_cols: usize, counts: &[usize]) {
    for g in 0..n_groups {
        let c = counts[g];
        if c == 0 {
            continue;
        }
        let cf = c as f64;
        for v in &mut sums[g * n_cols..(g + 1) * n_cols] {
            *v /= cf;
        }
    }
}

/// Per-group means from a CSR [`ShardSource`] (a backed reader, or an in-memory
/// single-shard source), computed on the GPU.
///
/// `cell_to_group[obs]` is the group id per observation (`-1` = excluded —
/// e.g. below `min_cells_per_group`, or not a kept group). `counts[g]` is the
/// number of cells in group `g` (the mean divisor). Returns row-major
/// `[n_groups × n_cols]` f64 means.
///
/// Streams shards via [`RawGpuShardSource`]; each shard's nonzeros are
/// scatter-added into a pre-zeroed device `sums` buffer by
/// [`gpu_de_pseudobulk_csr_direct`] over the full column range.
pub fn gpu_pseudobulk_means_csr(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    cell_to_group: &[i32],
    n_groups: usize,
    n_cols: usize,
    counts: &[usize],
) -> Result<Vec<f64>, GpuError> {
    if n_groups == 0 || n_cols == 0 {
        return Ok(vec![0.0; n_groups * n_cols]);
    }
    let cell_to_group_dev = dev.htod_copy(cell_to_group)?;
    let mut sums = dev.alloc_zeros::<f64>(n_groups * n_cols)?;

    // `Scatter`, not `Bounds`: the accumulation is `atomicAdd`, so duplicate
    // `(cell, gene)` pairs would sum correctly — but `csr_shard_pseudobulk_kernel`
    // first narrows each row to its `[c0, c1)` window with `scx_row_lower_bound`,
    // a binary search that silently returns the wrong window on unsorted column
    // indices. Reasoning from the accumulator alone gets this rung wrong.
    let mut src = RawGpuShardSource::new(dev, source)?.with_validation(ValidationPolicy::new(
        ValidationChecks::SORTED,
        "pseudobulk",
    ));
    let mut global_row = 0usize;
    src.for_each_gpu_shard(|_idx, slot| {
        let view = slot.view();
        let n_rows = view.shape.0;
        gpu_de_pseudobulk_csr_direct(
            dev,
            &view,
            &cell_to_group_dev,
            &mut sums,
            global_row,
            n_cols, // chunk_size = full column range
            0,
            n_cols,
            0, // mode_id = identity (plain sum)
        )?;
        global_row += n_rows;
        Ok(())
    })?;
    dev.synchronize()?;

    let mut host = dev.dtoh_copy(&sums)?;
    divide_by_counts(&mut host, n_groups, n_cols, counts);
    Ok(host)
}

/// Per-group means from a dense row-major `[n_obs × n_cols]` f32 host matrix
/// (an in-memory dense `X`, or an `obsm` embedding), computed on the GPU.
///
/// `all_group_cells` is the concatenated obs indices grouped by group, and
/// `group_offsets` (length `n_groups + 1`) are the prefix offsets into it — the
/// representation [`gpu_de_pseudobulk_all_groups`] consumes. `counts[g]` is the
/// mean divisor (typically `group_offsets[g+1] - group_offsets[g]`). Returns
/// row-major `[n_groups × n_cols]` f64 means.
#[allow(clippy::too_many_arguments)]
pub fn gpu_pseudobulk_means_dense(
    dev: &GpuDevice,
    dense: &[f32],
    n_obs: usize,
    n_cols: usize,
    all_group_cells: &[i32],
    group_offsets: &[i32],
    n_groups: usize,
    counts: &[usize],
) -> Result<Vec<f64>, GpuError> {
    if n_groups == 0 || n_cols == 0 {
        return Ok(vec![0.0; n_groups * n_cols]);
    }
    let dense_dev = dev.htod_copy(dense)?;
    let all_group_cells_dev = dev.htod_copy(all_group_cells)?;
    let group_offsets_dev = dev.htod_copy(group_offsets)?;
    let mut sums = dev.alloc_zeros::<f64>(n_groups * n_cols)?;

    gpu_de_pseudobulk_all_groups(
        dev,
        &dense_dev,
        &all_group_cells_dev,
        &group_offsets_dev,
        &mut sums,
        n_obs,
        n_cols,
        n_groups,
        0, // mode_id = identity
    )?;
    dev.synchronize()?;

    let mut host = dev.dtoh_copy(&sums)?;
    divide_by_counts(&mut host, n_groups, n_cols, counts);
    Ok(host)
}
