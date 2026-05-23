//! GPU dispatch for pdex_ref and Wilcoxon rank-sum DE.
//!
//! The chunk loop mirrors
//! `pdex_ref_streaming` / `wilcoxon_rank_sum_streaming` in `diffexp.rs` —
//! same host-side dense materialisation, same merge-chunks-then-BH semantics
//! — but the per-gene MWU work happens on GPU via the primitives in
//! `scx_gpu::gpu_diffexp::*`.
//!
//! Returns the same `PdexRefResult` / `DiffExpResult` types the CPU path
//! emits, so `pyscx`'s DataFrame conversion is unchanged.
//!
//! Boundaries:
//!   * Per-gene sort: single-tile CUB `BlockRadixSort` when the pool is
//!     ≤ [`scx_gpu::GPU_DE_BLOCK_SORT_CAPACITY`] (= 8192) keys; bottom-up
//!     tiled merge sort otherwise. No upper limit beyond available VRAM —
//!     `gpu_de_block_sort` dispatches internally.
//!   * Pseudobulk fold (`target_mean` / `ref_mean` for `pdex_ref`; raw
//!     per-group gene sums for Wilcoxon logFC) runs on device via
//!     `gpu_de_pseudobulk_all_groups`. The host only does the divide and the
//!     `GeomMeanMode::post` transform after downloading the per-(group, gene)
//!     f64 sums.
//!   * `compute_logfc_hostside` runs on host: it's O(chunk_size) per group
//!     and uses `libm` log2 / expm1 to match the CPU path bit-for-bit.

#![cfg(feature = "gpu")]

use scx_gpu::{
    default_gpu_de_gene_chunk_size, gpu_de_block_sort, gpu_de_combined_tie_term,
    gpu_de_pseudobulk_all_groups, gpu_de_pvalues, gpu_de_scatter_gene_major,
    gpu_de_searchsorted_ranksum, gpu_de_searchsorted_u_stat, gpu_de_tie_term, gpu_de_upload_chunk,
    CudaSlice, GpuDevice,
};

use crate::diffexp::{benjamini_hochberg, merge_diff_exp_results, DiffExpResult, PdexRefResult};
use crate::pseudobulk::GeomMeanMode;
use crate::{AccelError, Result};

/// Pseudocount used by the CPU `compute_logfc` helper. Mirrors
/// `diffexp::LOGFC_PSEUDOCOUNT` (private there).
const LOGFC_PSEUDOCOUNT: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// pdex `mode="ref"` on a dense `[n_obs × n_vars]` row-major buffer (single
/// chunk; for sparse / backed inputs use [`pdex_ref_gpu_sparse`] /
/// [`pdex_ref_gpu_streaming`]).
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_gpu_dense(
    device_id: usize,
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    mode: GeomMeanMode,
    epsilon: f64,
) -> Result<PdexRefResult> {
    validate_pdex_inputs(
        data.len(),
        n_obs,
        n_vars,
        gene_names.len(),
        groups.len(),
        group_names.len(),
        reference,
        epsilon,
    )?;

    let dev = open_device(device_id)?;
    let chunk_size = n_vars.min(default_gpu_de_gene_chunk_size(&dev, n_obs, n_obs));
    pdex_ref_gpu_chunked(
        &dev,
        n_obs,
        n_vars,
        chunk_size,
        gene_names,
        groups,
        group_names,
        reference,
        mode,
        epsilon,
        |c0, sz, buf| {
            // Slice the dense row-major source into the chunk buffer.
            // `data[cell * n_vars + var]` → `buf[cell * sz + (var - c0)]`.
            for cell in 0..n_obs {
                let src_off = cell * n_vars + c0;
                let dst_off = cell * sz;
                buf[dst_off..dst_off + sz].copy_from_slice(&data[src_off..src_off + sz]);
            }
            Ok(())
        },
    )
}

/// pdex `mode="ref"` on an in-memory `ScxCsr` (gene-chunked).
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_gpu_sparse(
    device_id: usize,
    csr: &scx_sparse::ScxCsr,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    gene_chunk_size: Option<usize>,
    mode: GeomMeanMode,
    epsilon: f64,
) -> Result<PdexRefResult> {
    let (n_obs, n_vars) = csr.shape;
    validate_pdex_inputs(
        n_obs * n_vars, // length match
        n_obs,
        n_vars,
        gene_names.len(),
        groups.len(),
        group_names.len(),
        reference,
        epsilon,
    )?;

    let dev = open_device(device_id)?;
    let chunk_size = resolve_chunk_size(&dev, n_obs, gene_chunk_size, n_vars);

    pdex_ref_gpu_chunked(
        &dev,
        n_obs,
        n_vars,
        chunk_size,
        gene_names,
        groups,
        group_names,
        reference,
        mode,
        epsilon,
        |c0, sz, buf| {
            let col_indices: Vec<u32> = (c0 as u32..(c0 + sz) as u32).collect();
            let projected = scx_engine::project_csr(csr, &col_indices);
            scatter_projected_into_buf(&projected, n_obs, sz, buf, 0);
            Ok(())
        },
    )
}

/// pdex `mode="ref"` on an SCX-backed reader (shard-streaming, gene-chunked).
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_gpu_streaming(
    device_id: usize,
    reader: &scx_format::backed::BackedCsrReader,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    gene_chunk_size: Option<usize>,
    mode: GeomMeanMode,
    epsilon: f64,
) -> Result<PdexRefResult> {
    let n_obs = reader.n_obs();
    let n_vars = gene_names.len();
    validate_pdex_inputs(
        n_obs * n_vars,
        n_obs,
        n_vars,
        gene_names.len(),
        groups.len(),
        group_names.len(),
        reference,
        epsilon,
    )?;

    let dev = open_device(device_id)?;
    let chunk_size = resolve_chunk_size(&dev, n_obs, gene_chunk_size, n_vars);

    let n_shards = reader.index().n_shards();
    pdex_ref_gpu_chunked(
        &dev,
        n_obs,
        n_vars,
        chunk_size,
        gene_names,
        groups,
        group_names,
        reference,
        mode,
        epsilon,
        |c0, sz, buf| {
            let col_indices: Vec<u32> = (c0 as u32..(c0 + sz) as u32).collect();
            buf.fill(0.0);
            let mut global_row = 0usize;
            for shard_idx in 0..n_shards {
                let shard_csr = reader
                    .read_shard_cached_arc(shard_idx)
                    .map_err(AccelError::Scx)?;
                let projected = scx_engine::project_csr(&shard_csr, &col_indices);
                scatter_projected_into_buf(&projected, n_obs, sz, buf, global_row);
                global_row += projected.n_rows();
            }
            Ok(())
        },
    )
}

/// Wilcoxon rank-sum (1-vs-rest or vs-reference) on a dense row-major buffer.
#[allow(clippy::too_many_arguments)]
pub fn wilcoxon_rank_sum_gpu_dense(
    device_id: usize,
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
) -> Result<DiffExpResult> {
    validate_wilcoxon_inputs(data.len(), n_obs, n_vars, gene_names.len(), groups.len())?;

    let dev = open_device(device_id)?;
    let chunk_size = n_vars.min(default_gpu_de_gene_chunk_size(&dev, n_obs, n_obs));

    wilcoxon_rank_sum_gpu_chunked(
        &dev,
        n_obs,
        n_vars,
        chunk_size,
        gene_names,
        groups,
        group_names,
        reference,
        log_transformed,
        rankby_abs,
        tie_correct,
        |c0, sz, buf| {
            for cell in 0..n_obs {
                let src_off = cell * n_vars + c0;
                let dst_off = cell * sz;
                buf[dst_off..dst_off + sz].copy_from_slice(&data[src_off..src_off + sz]);
            }
            Ok(())
        },
    )
}

/// Wilcoxon rank-sum on an in-memory `ScxCsr` (gene-chunked).
#[allow(clippy::too_many_arguments)]
pub fn wilcoxon_rank_sum_gpu_sparse(
    device_id: usize,
    csr: &scx_sparse::ScxCsr,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    gene_chunk_size: Option<usize>,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
) -> Result<DiffExpResult> {
    let (n_obs, n_vars) = csr.shape;
    validate_wilcoxon_inputs(
        n_obs * n_vars,
        n_obs,
        n_vars,
        gene_names.len(),
        groups.len(),
    )?;

    let dev = open_device(device_id)?;
    let chunk_size = resolve_chunk_size(&dev, n_obs, gene_chunk_size, n_vars);

    wilcoxon_rank_sum_gpu_chunked(
        &dev,
        n_obs,
        n_vars,
        chunk_size,
        gene_names,
        groups,
        group_names,
        reference,
        log_transformed,
        rankby_abs,
        tie_correct,
        |c0, sz, buf| {
            let col_indices: Vec<u32> = (c0 as u32..(c0 + sz) as u32).collect();
            let projected = scx_engine::project_csr(csr, &col_indices);
            scatter_projected_into_buf(&projected, n_obs, sz, buf, 0);
            Ok(())
        },
    )
}

/// Wilcoxon rank-sum on an SCX-backed reader (shard-streaming, gene-chunked).
#[allow(clippy::too_many_arguments)]
pub fn wilcoxon_rank_sum_gpu_streaming(
    device_id: usize,
    reader: &scx_format::backed::BackedCsrReader,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    gene_chunk_size: Option<usize>,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
) -> Result<DiffExpResult> {
    let n_obs = reader.n_obs();
    let n_vars = gene_names.len();
    validate_wilcoxon_inputs(
        n_obs * n_vars,
        n_obs,
        n_vars,
        gene_names.len(),
        groups.len(),
    )?;

    let dev = open_device(device_id)?;
    let chunk_size = resolve_chunk_size(&dev, n_obs, gene_chunk_size, n_vars);
    let n_shards = reader.index().n_shards();

    wilcoxon_rank_sum_gpu_chunked(
        &dev,
        n_obs,
        n_vars,
        chunk_size,
        gene_names,
        groups,
        group_names,
        reference,
        log_transformed,
        rankby_abs,
        tie_correct,
        |c0, sz, buf| {
            let col_indices: Vec<u32> = (c0 as u32..(c0 + sz) as u32).collect();
            buf.fill(0.0);
            let mut global_row = 0usize;
            for shard_idx in 0..n_shards {
                let shard_csr = reader
                    .read_shard_cached_arc(shard_idx)
                    .map_err(AccelError::Scx)?;
                let projected = scx_engine::project_csr(&shard_csr, &col_indices);
                scatter_projected_into_buf(&projected, n_obs, sz, buf, global_row);
                global_row += projected.n_rows();
            }
            Ok(())
        },
    )
}

// ---------------------------------------------------------------------------
// Shared chunked drivers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn pdex_ref_gpu_chunked<F>(
    dev: &GpuDevice,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    mode: GeomMeanMode,
    epsilon: f64,
    mut materialise: F,
) -> Result<PdexRefResult>
where
    F: FnMut(usize, usize, &mut [f32]) -> Result<()>,
{
    if chunk_size == 0 {
        return Err(AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }

    let n_groups = group_names.len();
    let (group_indices, _oor) = bucket_cells_by_group(groups, n_groups);
    let ref_cells = &group_indices[reference];
    let n_ref = ref_cells.len();
    if n_ref == 0 {
        return Err(AccelError::InvalidInput(format!(
            "reference group '{}' has zero cells",
            group_names[reference]
        )));
    }
    // G1.5: the 8192-cell v1 capacity cap has been lifted by the tiled
    // merge-sort path inside `gpu_de_block_sort` — any pool size now
    // dispatches correctly. The `GPU_DE_BLOCK_SORT_CAPACITY` constant is
    // now the fast-path threshold, not a hard ceiling.
    let test_groups: Vec<usize> = (0..n_groups).filter(|&g| g != reference).collect();
    let target_memberships: Vec<usize> = test_groups
        .iter()
        .map(|&g| group_indices[g].len())
        .collect();

    let n_g_max = target_memberships.iter().copied().max().unwrap_or(0);

    // Pre-encode cell index permutations as i32 (the kernel signature).
    let ref_idx_i32: Vec<i32> = ref_cells.iter().map(|&c| c as i32).collect();
    let group_idx_i32: Vec<Vec<i32>> = test_groups
        .iter()
        .map(|&g| group_indices[g].iter().map(|&c| c as i32).collect())
        .collect();

    // Flattened cell permutation for the all-groups pseudobulk fold (G1.6).
    // Group 0 = reference; groups 1..=n_test = test_groups in input order.
    let n_groups_for_means = 1 + test_groups.len();
    let mut all_cells_host: Vec<i32> = Vec::with_capacity(n_ref + n_g_max * test_groups.len());
    let mut offsets_host: Vec<i32> = Vec::with_capacity(n_groups_for_means + 1);
    offsets_host.push(0);
    all_cells_host.extend(ref_idx_i32.iter().copied());
    offsets_host.push(all_cells_host.len() as i32);
    for cells in &group_idx_i32 {
        all_cells_host.extend(cells.iter().copied());
        offsets_host.push(all_cells_host.len() as i32);
    }
    let d_all_cells = dev
        .htod_copy(&all_cells_host)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE alloc cell perm: {e}")))?;
    let d_offsets = dev
        .htod_copy(&offsets_host)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE alloc offsets: {e}")))?;
    let mode_id = geom_mean_mode_id(mode);

    // Device-side scratch sized to max-pool-per-gene = max(n_ref, n_g_max).
    let n_pool_max = n_ref.max(n_g_max).max(1);
    let mut scratch = scx_gpu::GpuDeChunkScratch::new(dev, n_obs, chunk_size, n_pool_max)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE scratch alloc failed: {e}")))?;
    // G2: pre-grow per-chunk reusable slots to their worst-case sizes for
    // this DE call so the chunk loop never re-allocates. Mirrors the slab
    // sizing already done by `GpuDeChunkScratch::new` for the (ref-or-pool)
    // path. `ensure_aux_capacity` covers the multi-tile sort ping-pong on
    // census-scale inputs where `chunk_size × max_pool > 8192`.
    scratch
        .ensure_ref_slab_capacity(dev, n_ref)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE ensure ref_slab: {e}")))?;
    scratch
        .ensure_group_slab_capacity(dev, n_g_max.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE ensure group_slab: {e}")))?;
    scratch
        .ensure_sums_capacity(dev, n_groups_for_means)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE ensure sums: {e}")))?;
    scratch
        .ensure_aux_capacity(dev, chunk_size * n_pool_max)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE ensure aux: {e}")))?;

    // Accumulators (per test group × gene, in input gene order).
    let n_test = test_groups.len();
    let mut target_means: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut ref_means: Vec<f64> = Vec::with_capacity(n_vars);
    let mut log2_fold_changes: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut percent_changes: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut statistics: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut p_values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];

    // Host-side chunk dense buffer (row-major n_obs × chunk_size). Allocate
    // once with the max chunk capacity; refill via the materialise closure.
    let mut chunk_dense = vec![0.0f32; n_obs * chunk_size];

    for c0 in (0..n_vars).step_by(chunk_size) {
        let c1 = (c0 + chunk_size).min(n_vars);
        let sz = c1 - c0;
        // Reuse the high end of `chunk_dense` for chunks smaller than chunk_size.
        let buf = &mut chunk_dense[..n_obs * sz];
        // Zero the buf before the materialise closure: the sparse and
        // streaming closures write only the non-zero CSR entries, so
        // stale values from chunk N-1 leak into zero positions of chunk
        // N without this. The dense closure overwrites every cell so
        // the fill is redundant there but cheap (memset is ~50 GB/s).
        buf.fill(0.0);
        materialise(c0, sz, buf)?;

        // --- GPU work for this chunk ---

        // 1. Upload dense chunk.
        gpu_de_upload_chunk(dev, &mut scratch, buf, n_obs, sz)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE upload: {e}")))?;

        // 2. Pseudobulk fold on device (G1.6). Replaces the host rayon
        //    `compute_pdex_means` that previously dominated host wall time
        //    on census-tier inputs. G2: reuses `scratch.sums` instead of a
        //    per-chunk allocation.
        let (chunk_ref_means, chunk_target_means) = compute_pdex_means_gpu(
            dev,
            &scratch.dense,
            &d_all_cells,
            &d_offsets,
            &mut scratch.sums,
            n_obs,
            sz,
            n_ref,
            &target_memberships,
            mode,
            mode_id,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU DE pseudobulk: {e}")))?;

        // log2_fc and percent_change (per gene per group) — host, cheap.
        let chunk_log2_fc: Vec<Vec<f64>> = chunk_target_means
            .iter()
            .map(|tm| {
                tm.iter()
                    .zip(chunk_ref_means.iter())
                    .map(|(t, r)| ((t + epsilon) / (r + epsilon)).log2())
                    .collect()
            })
            .collect();
        let chunk_percent: Vec<Vec<f64>> = chunk_target_means
            .iter()
            .map(|tm| {
                tm.iter()
                    .zip(chunk_ref_means.iter())
                    .map(|(t, r)| (t - r) / (r + epsilon))
                    .collect()
            })
            .collect();

        // 2. Build sorted ref slab and ref tie term. G2 reuses
        //    `scratch.ref_slab` across chunks — no per-chunk allocation. The
        //    scatter kernel writes (sz × n_ref) of (chunk_max × n_ref_max)
        //    capacity; subsequent kernels are told the logical
        //    `(chunk_size, n_per_gene) = (sz, n_ref)` so the layout matches.
        gpu_de_scatter_gene_major(
            dev,
            &scratch.dense,
            &ref_idx_i32,
            &mut scratch.ref_slab,
            n_obs,
            sz,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU DE scatter ref: {e}")))?;
        gpu_de_block_sort(dev, &mut scratch.ref_slab, &mut scratch.slab_aux, sz, n_ref)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE sort ref: {e}")))?;
        // Ref tie term is reusable across all test groups; keep it on host.
        gpu_de_tie_term(dev, &scratch.ref_slab, &mut scratch.tie_term, sz, n_ref)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE tie term ref: {e}")))?;
        let mut tie_ref_host = dev
            .dtoh_copy(&scratch.tie_term)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE dtoh ref tie: {e}")))?;
        tie_ref_host.truncate(sz);

        // 3. Per test group: scatter, sort, combined tie, U1, p-value.
        ref_means.extend_from_slice(&chunk_ref_means);
        for (tg_idx, &g) in test_groups.iter().enumerate() {
            let group_cells = &group_indices[g];
            let n_g = group_cells.len();

            // Push the gene-input-order chunk slices into the global accumulators.
            target_means[tg_idx].extend_from_slice(&chunk_target_means[tg_idx]);
            log2_fold_changes[tg_idx].extend_from_slice(&chunk_log2_fc[tg_idx]);
            percent_changes[tg_idx].extend_from_slice(&chunk_percent[tg_idx]);

            // Empty target group → NaN U, p = 1.
            if n_g == 0 {
                statistics[tg_idx].extend(std::iter::repeat_n(f64::NAN, sz));
                p_values[tg_idx].extend(std::iter::repeat_n(1.0, sz));
                continue;
            }

            // G2: scratch.group_slab is grown to chunk_size × n_g_max at
            // driver entry; the scatter writes (sz × n_g) into its prefix.
            gpu_de_scatter_gene_major(
                dev,
                &scratch.dense,
                &group_idx_i32[tg_idx],
                &mut scratch.group_slab,
                n_obs,
                sz,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU DE scatter group: {e}")))?;

            // U1 from searchsorted before sorting the group (the kernel
            // reads group_slab in any order). Write into scratch.u_or_rank.
            gpu_de_searchsorted_u_stat(
                dev,
                &scratch.ref_slab,
                &scratch.group_slab,
                &mut scratch.u_or_rank,
                sz,
                n_ref,
                n_g,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU DE searchsorted U: {e}")))?;

            // Sort group slab for combined tie term.
            gpu_de_block_sort(dev, &mut scratch.group_slab, &mut scratch.slab_aux, sz, n_g)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE sort group: {e}")))?;
            gpu_de_combined_tie_term(
                dev,
                &scratch.ref_slab,
                &scratch.group_slab,
                &mut scratch.tie_term,
                sz,
                n_ref,
                n_g,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU DE combined tie: {e}")))?;

            // p-value on device.
            gpu_de_pvalues(
                dev,
                &scratch.u_or_rank,
                &scratch.tie_term,
                &mut scratch.p_values,
                sz,
                n_g,
                n_ref,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU DE p-value: {e}")))?;

            let mut u_host = dev
                .dtoh_copy(&scratch.u_or_rank)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE dtoh U: {e}")))?;
            u_host.truncate(sz);
            let mut p_host = dev
                .dtoh_copy(&scratch.p_values)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE dtoh p: {e}")))?;
            p_host.truncate(sz);

            statistics[tg_idx].extend(u_host);
            p_values[tg_idx].extend(p_host.iter().map(|&p| p.clamp(0.0, 1.0)));
        }
    }

    dev.synchronize()
        .map_err(|e| AccelError::LinAlg(format!("GPU DE synchronize: {e}")))?;

    // BH per group across all genes (matches CPU `pdex_ref` post-pass).
    let fdrs: Vec<Vec<f64>> = p_values
        .iter()
        .map(|pv| {
            let clipped: Vec<f64> = pv.iter().map(|&p| p.clamp(0.0, 1.0)).collect();
            benjamini_hochberg(&clipped)
        })
        .collect();

    // Clip p-values to [0, 1] (matches CPU; harmless after the GPU clip).
    for pv in p_values.iter_mut() {
        for p in pv.iter_mut() {
            *p = p.clamp(0.0, 1.0);
        }
    }

    Ok(PdexRefResult {
        group_names: test_groups
            .iter()
            .map(|&g| group_names[g].clone())
            .collect(),
        feature_names: gene_names.to_vec(),
        target_means,
        ref_means,
        target_memberships,
        ref_membership: n_ref,
        log2_fold_changes,
        percent_changes,
        statistics,
        p_values,
        fdrs,
    })
}

#[allow(clippy::too_many_arguments)]
fn wilcoxon_rank_sum_gpu_chunked<F>(
    dev: &GpuDevice,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
    mut materialise: F,
) -> Result<DiffExpResult>
where
    F: FnMut(usize, usize, &mut [f32]) -> Result<()>,
{
    if chunk_size == 0 {
        return Err(AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }

    let n_groups = group_names.len();
    let (group_indices, _oor) = bucket_cells_by_group(groups, n_groups);

    let test_groups: Vec<usize> = match reference {
        Some(r) => (0..n_groups).filter(|&g| g != r).collect(),
        None => (0..n_groups).collect(),
    };

    // For 1-vs-rest the pool size is n_obs (all cells). For ref mode it's
    // max(n_ref, max test group). G1.5 tiled merge-sort lifts the prior 8192
    // cap on this pool — `gpu_de_block_sort` dispatches single-tile vs
    // multi-tile internally, so any pool size sorts correctly.
    let (max_pool, ref_cells_opt): (usize, Option<Vec<usize>>) = match reference {
        Some(r) => {
            let n_ref = group_indices[r].len();
            let n_g_max = test_groups
                .iter()
                .map(|&g| group_indices[g].len())
                .max()
                .unwrap_or(0);
            (n_ref.max(n_g_max), Some(group_indices[r].clone()))
        }
        None => (n_obs, None),
    };

    // Pre-encode all-cells permutation (for 1-vs-rest sort-all) and per-group
    // permutations.
    let all_idx_i32: Option<Vec<i32>> = if reference.is_none() {
        Some((0..n_obs as i32).collect())
    } else {
        None
    };
    let ref_idx_i32: Option<Vec<i32>> = ref_cells_opt
        .as_ref()
        .map(|v| v.iter().map(|&c| c as i32).collect());
    let group_idx_i32: Vec<Vec<i32>> = test_groups
        .iter()
        .map(|&g| group_indices[g].iter().map(|&c| c as i32).collect())
        .collect();

    let mut scratch = scx_gpu::GpuDeChunkScratch::new(dev, n_obs, chunk_size, max_pool.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE scratch alloc failed: {e}")))?;
    // G2: pre-grow the per-chunk reusable slots once (max_pool already
    // covers both ref-mode pool-size = n_ref and 1-vs-rest pool-size =
    // n_obs). `n_g_max_for_wil` is the largest test-group size; for
    // 1-vs-rest n_g_max ≤ max_pool, for ref-mode it's already in max_pool.
    let n_g_max_for_wil = test_groups
        .iter()
        .map(|&g| group_indices[g].len())
        .max()
        .unwrap_or(0);
    scratch
        .ensure_ref_slab_capacity(dev, max_pool.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE ensure ref_slab: {e}")))?;
    scratch
        .ensure_group_slab_capacity(dev, n_g_max_for_wil.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE ensure group_slab: {e}")))?;
    scratch
        .ensure_sums_capacity(dev, n_groups)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE ensure sums: {e}")))?;
    scratch
        .ensure_aux_capacity(dev, chunk_size * max_pool.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE ensure aux: {e}")))?;

    // Flatten the (cell → group) labelling into a CSR-style permutation for
    // `gpu_de_pseudobulk_all_groups` (G1.6). Cells with `groups[i] >= n_groups`
    // (the out-of-range sentinel) get dropped; groups appear in input order so
    // `host_sums[g * chunk_size + var]` matches `group_gene_sums[g][var]`.
    let mut wilcoxon_cells_by_group: Vec<Vec<i32>> = vec![Vec::new(); n_groups];
    for (cell, &g) in groups.iter().enumerate() {
        if g < n_groups {
            wilcoxon_cells_by_group[g].push(cell as i32);
        }
    }
    let mut all_cells_host: Vec<i32> = Vec::with_capacity(n_obs);
    let mut offsets_host: Vec<i32> = Vec::with_capacity(n_groups + 1);
    offsets_host.push(0);
    for cells in &wilcoxon_cells_by_group {
        all_cells_host.extend(cells.iter().copied());
        offsets_host.push(all_cells_host.len() as i32);
    }
    let d_all_cells = dev
        .htod_copy(&all_cells_host)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE alloc cell perm: {e}")))?;
    let d_offsets = dev
        .htod_copy(&offsets_host)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE alloc offsets: {e}")))?;

    // Accumulators in input gene order.
    let n_test = test_groups.len();
    let mut chunk_results: Vec<DiffExpResult> = Vec::new();

    let mut chunk_dense = vec![0.0f32; n_obs * chunk_size];

    for c0 in (0..n_vars).step_by(chunk_size) {
        let c1 = (c0 + chunk_size).min(n_vars);
        let sz = c1 - c0;
        let buf = &mut chunk_dense[..n_obs * sz];
        // See `pdex_ref_gpu_chunked`: sparse / streaming closures only
        // write non-zero entries; zero the slot to keep chunks isolated.
        buf.fill(0.0);
        materialise(c0, sz, buf)?;

        gpu_de_upload_chunk(dev, &mut scratch, buf, n_obs, sz)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE upload: {e}")))?;

        // Per-group per-gene raw sums on device (G1.6) — feeds the host-side
        // logFC computation further down. `mode_id = 0` selects the identity
        // pre-transform; Wilcoxon doesn't apply `GeomMeanMode`. G2: reuses
        // `scratch.sums` instead of a per-chunk allocation.
        let group_gene_sums = compute_group_gene_sums_gpu(
            dev,
            &scratch.dense,
            &d_all_cells,
            &d_offsets,
            &mut scratch.sums,
            n_obs,
            sz,
            n_groups,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon pseudobulk: {e}")))?;

        // Build the sort pool slab once per chunk. G2: the pool slab lives in
        // `scratch.ref_slab` (already sized to chunk_size × max_pool); the
        // scatter writes (sz × pool_len) into the prefix.
        let (pool_len, _pool_perm_i32) = match (&ref_idx_i32, &all_idx_i32) {
            (Some(ref_i32), _) => {
                let n_ref = ref_i32.len();
                gpu_de_scatter_gene_major(
                    dev,
                    &scratch.dense,
                    ref_i32,
                    &mut scratch.ref_slab,
                    n_obs,
                    sz,
                )
                .map_err(|e| AccelError::LinAlg(format!("GPU DE scatter ref: {e}")))?;
                (n_ref, Some(ref_i32))
            }
            (None, Some(all_i32)) => {
                gpu_de_scatter_gene_major(
                    dev,
                    &scratch.dense,
                    all_i32,
                    &mut scratch.ref_slab,
                    n_obs,
                    sz,
                )
                .map_err(|e| AccelError::LinAlg(format!("GPU DE scatter all: {e}")))?;
                (n_obs, Some(all_i32))
            }
            (None, None) => unreachable!(),
        };

        gpu_de_block_sort(
            dev,
            &mut scratch.ref_slab,
            &mut scratch.slab_aux,
            sz,
            pool_len,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU DE sort pool: {e}")))?;

        // Tie term over the pool (used for ref mode + as the base for combined
        // ties in ref mode; for 1-vs-rest this is the global tie correction).
        gpu_de_tie_term(dev, &scratch.ref_slab, &mut scratch.tie_term, sz, pool_len)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE pool tie: {e}")))?;
        let mut pool_tie_host = dev
            .dtoh_copy(&scratch.tie_term)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE dtoh pool tie: {e}")))?;
        pool_tie_host.truncate(sz);

        // Per test group: searchsorted, p-value, z, logFC.
        // Tuple = (group_name, gene_names, scores, pvals, logfc) — packaged
        // per chunk and consumed by `assemble_chunk_diffexp_result`.
        type ChunkGroupRow = (String, Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>);
        let mut chunk_per_group: Vec<ChunkGroupRow> = Vec::with_capacity(n_test);

        for (tg_idx, &g) in test_groups.iter().enumerate() {
            let group_cells = &group_indices[g];
            let n_g = group_cells.len();
            let group_name = group_names[g].clone();

            if n_g == 0 {
                let names = gene_names[c0..c1].to_vec();
                let scores = vec![f64::NAN; sz];
                let pvals = vec![1.0f64; sz];
                let logfc = vec![f64::NAN; sz];
                chunk_per_group.push((group_name, names, scores, pvals, logfc));
                continue;
            }

            // G2: scratch.group_slab is pre-sized for the largest test group;
            // the scatter writes (sz × n_g) into its prefix.
            gpu_de_scatter_gene_major(
                dev,
                &scratch.dense,
                &group_idx_i32[tg_idx],
                &mut scratch.group_slab,
                n_obs,
                sz,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU DE scatter group: {e}")))?;

            // For ref mode: U1 via ref-searchsorted. For 1-vs-rest: rank sum
            // via all-searchsorted → U = R_g − n_g(n_g+1)/2.
            let (u_host, tie_host_for_p, n1, n2) = if reference.is_some() {
                gpu_de_searchsorted_u_stat(
                    dev,
                    &scratch.ref_slab,
                    &scratch.group_slab,
                    &mut scratch.u_or_rank,
                    sz,
                    pool_len,
                    n_g,
                )
                .map_err(|e| AccelError::LinAlg(format!("GPU DE searchsorted U: {e}")))?;
                let mut u_host = dev
                    .dtoh_copy(&scratch.u_or_rank)
                    .map_err(|e| AccelError::LinAlg(format!("GPU DE dtoh U: {e}")))?;
                u_host.truncate(sz);

                // Combined tie term for ref vs group.
                gpu_de_block_sort(dev, &mut scratch.group_slab, &mut scratch.slab_aux, sz, n_g)
                    .map_err(|e| AccelError::LinAlg(format!("GPU DE sort group: {e}")))?;
                gpu_de_combined_tie_term(
                    dev,
                    &scratch.ref_slab,
                    &scratch.group_slab,
                    &mut scratch.tie_term,
                    sz,
                    pool_len,
                    n_g,
                )
                .map_err(|e| AccelError::LinAlg(format!("GPU DE combined tie: {e}")))?;
                let mut combined_tie = dev
                    .dtoh_copy(&scratch.tie_term)
                    .map_err(|e| AccelError::LinAlg(format!("GPU DE dtoh combined tie: {e}")))?;
                combined_tie.truncate(sz);
                (u_host, combined_tie, n_g, pool_len)
            } else {
                gpu_de_searchsorted_ranksum(
                    dev,
                    &scratch.ref_slab,
                    &scratch.group_slab,
                    &mut scratch.u_or_rank,
                    sz,
                    pool_len,
                    n_g,
                )
                .map_err(|e| AccelError::LinAlg(format!("GPU DE searchsorted rank: {e}")))?;
                let mut rank_host = dev
                    .dtoh_copy(&scratch.u_or_rank)
                    .map_err(|e| AccelError::LinAlg(format!("GPU DE dtoh ranksum: {e}")))?;
                rank_host.truncate(sz);
                // R_g → U_g.
                let n1d = n_g as f64;
                let u_host: Vec<f64> = rank_host
                    .iter()
                    .map(|&r| r - n1d * (n1d + 1.0) / 2.0)
                    .collect();
                (u_host, pool_tie_host.clone(), n_g, n_obs - n_g)
            };

            // Host-side: compute z (signed) and (if !tie_correct) zero-out
            // the tie correction before p-value comes back from device. The
            // device-side p-value used `combined_tie` (or `pool_tie`); when
            // `tie_correct == false` the CPU path zeroes ties — so we
            // recompute p on host in that case to match exactly.
            let (scores, pvals) =
                compute_scores_and_pvals(&u_host, &tie_host_for_p, n1, n2, tie_correct);

            // logFC per gene: target_mean and rest_mean.
            let mut logfc = Vec::with_capacity(sz);
            for (var, &s_g) in group_gene_sums[g].iter().enumerate().take(sz) {
                let mean_g = if n_g > 0 { s_g / n_g as f64 } else { f64::NAN };
                let mean_ref = match reference {
                    Some(r) => {
                        let n_r = group_indices[r].len();
                        if n_r == 0 {
                            f64::NAN
                        } else {
                            group_gene_sums[r][var] / n_r as f64
                        }
                    }
                    None => {
                        let rest_sum: f64 = (0..n_groups)
                            .filter(|&gg| gg != g)
                            .map(|gg| group_gene_sums[gg][var])
                            .sum();
                        let rest_n = n_obs - n_g;
                        if rest_n == 0 {
                            f64::NAN
                        } else {
                            rest_sum / rest_n as f64
                        }
                    }
                };
                logfc.push(compute_logfc_hostside(mean_g, mean_ref, log_transformed));
            }

            let names = gene_names[c0..c1].to_vec();
            chunk_per_group.push((group_name, names, scores, pvals, logfc));
        }

        // Assemble a DiffExpResult for this chunk: per-group sort by score
        // (matching `wilcoxon_rank_sum`), apply per-chunk BH (discarded later
        // by `merge_diff_exp_results`'s global BH).
        let chunk_de = assemble_chunk_diffexp_result(chunk_per_group, rankby_abs)
            .ok_or_else(|| AccelError::InvalidInput("empty test_groups in chunk".into()))?;
        chunk_results.push(chunk_de);
    }

    dev.synchronize()
        .map_err(|e| AccelError::LinAlg(format!("GPU DE synchronize: {e}")))?;

    merge_diff_exp_results(chunk_results, rankby_abs)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn open_device(device_id: usize) -> Result<GpuDevice> {
    GpuDevice::new(device_id).map_err(|e| AccelError::LinAlg(format!("GPU init failed: {e}")))
}

fn resolve_chunk_size(dev: &GpuDevice, n_obs: usize, user: Option<usize>, n_vars: usize) -> usize {
    match user {
        Some(v) if v > 0 => v.min(n_vars).max(1),
        _ => default_gpu_de_gene_chunk_size(dev, n_obs, n_obs)
            .min(n_vars)
            .max(1),
    }
}

fn bucket_cells_by_group(groups: &[usize], n_groups: usize) -> (Vec<Vec<usize>>, usize) {
    let mut indices: Vec<Vec<usize>> = vec![vec![]; n_groups];
    let mut oor = 0usize;
    for (i, &g) in groups.iter().enumerate() {
        if g < n_groups {
            indices[g].push(i);
        } else {
            oor += 1;
        }
    }
    (indices, oor)
}

fn scatter_projected_into_buf(
    projected: &scx_sparse::ScxCsr,
    n_obs: usize,
    chunk_size: usize,
    buf: &mut [f32],
    global_row_offset: usize,
) {
    let _ = n_obs;
    for row in 0..projected.n_rows() {
        let s = projected.indptr[row] as usize;
        let e = projected.indptr[row + 1] as usize;
        let base = (global_row_offset + row) * chunk_size;
        for j in s..e {
            let col = projected.indices[j] as usize;
            buf[base + col] = projected.data[j];
        }
    }
}

/// Mode-id encoding for `gpu_de_pseudobulk_all_groups`. Must match the
/// `apply_pre_transform` switch in `scx-gpu/kernels/diffexp.cu`. The kernel
/// only applies the `pre()` transform; the matching `mode.post()` runs on host.
fn geom_mean_mode_id(mode: GeomMeanMode) -> i32 {
    match mode {
        GeomMeanMode::ArithRaw => 0,
        GeomMeanMode::ArithLog1pExpand => 1,
        GeomMeanMode::GeomRaw => 2,
        GeomMeanMode::GeomLog1p => 3,
    }
}

/// GPU version of [`compute_pdex_means`] (G1.6).
///
/// Launches `gpu_de_pseudobulk_all_groups` over the already-uploaded `d_dense`
/// slab with the flattened cell permutation (group 0 = ref, groups 1.. = test
/// groups in order). Downloads the f64 sums, divides by per-group cell count,
/// and applies `mode.post()` on host. Output shape matches the host helper's:
/// `(Vec<f64>, Vec<Vec<f64>>)` = (ref_means, target_means).
#[allow(clippy::too_many_arguments)]
fn compute_pdex_means_gpu(
    dev: &GpuDevice,
    d_dense: &CudaSlice<f32>,
    d_all_cells: &CudaSlice<i32>,
    d_offsets: &CudaSlice<i32>,
    d_sums: &mut CudaSlice<f64>,
    n_obs: usize,
    chunk_size: usize,
    n_ref: usize,
    target_memberships: &[usize],
    mode: GeomMeanMode,
    mode_id: i32,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n_test = target_memberships.len();
    let n_groups = 1 + n_test;

    // G2: `d_sums` is a borrowed slice from `GpuDeChunkScratch::sums` sized
    // to hold at least `n_groups * chunk_size` f64 values; the caller has
    // already grown it via `ensure_sums_capacity` before the chunk loop.
    gpu_de_pseudobulk_all_groups(
        dev,
        d_dense,
        d_all_cells,
        d_offsets,
        d_sums,
        n_obs,
        chunk_size,
        n_groups,
        mode_id,
    )
    .map_err(|e| AccelError::LinAlg(format!("GPU DE pseudobulk kernel: {e}")))?;
    // Slice to the populated prefix — `d_sums` is sized for `chunk_max ×
    // sums_capacity` but the kernel only writes `n_groups * chunk_size`
    // elements. Downloading the full buffer wastes PCIe on the final chunk
    // and whenever `n_groups < sums_capacity`.
    let host_sums = dev
        .stream()
        .clone_dtoh(&d_sums.slice(..n_groups * chunk_size))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE dtoh pseudobulk sums: {e}")))?;

    // Group 0 = reference.
    let ref_means: Vec<f64> = host_sums
        .iter()
        .take(chunk_size)
        .map(|&s| mode.post(if n_ref == 0 { 0.0 } else { s / n_ref as f64 }))
        .collect();

    // Groups 1..=n_test = test groups, in input order.
    let mut target_means: Vec<Vec<f64>> = Vec::with_capacity(n_test);
    for (i, &n_g) in target_memberships.iter().enumerate() {
        let base = (i + 1) * chunk_size;
        let mut tm = Vec::with_capacity(chunk_size);
        for var in 0..chunk_size {
            if n_g == 0 {
                tm.push(f64::NAN);
            } else {
                let s = host_sums[base + var];
                tm.push(mode.post(s / n_g as f64));
            }
        }
        target_means.push(tm);
    }

    Ok((ref_means, target_means))
}

/// GPU version of [`compute_group_gene_sums`] (G1.6). Uses identity pre-transform
/// (`mode_id = 0`); Wilcoxon doesn't apply `GeomMeanMode` and just needs raw
/// `Σ x` per (group, gene) for the host-side logFC computation.
#[allow(clippy::too_many_arguments)]
fn compute_group_gene_sums_gpu(
    dev: &GpuDevice,
    d_dense: &CudaSlice<f32>,
    d_all_cells: &CudaSlice<i32>,
    d_offsets: &CudaSlice<i32>,
    d_sums: &mut CudaSlice<f64>,
    n_obs: usize,
    chunk_size: usize,
    n_groups: usize,
) -> Result<Vec<Vec<f64>>> {
    // G2: `d_sums` is a borrowed slice from `GpuDeChunkScratch::sums`; see
    // `compute_pdex_means_gpu` for the lifetime / sizing contract.
    gpu_de_pseudobulk_all_groups(
        dev,
        d_dense,
        d_all_cells,
        d_offsets,
        d_sums,
        n_obs,
        chunk_size,
        n_groups,
        0, // mode_id = ArithRaw / identity
    )
    .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon pseudobulk kernel: {e}")))?;
    // See `compute_pdex_means_gpu` for the slice rationale.
    let host_sums = dev
        .stream()
        .clone_dtoh(&d_sums.slice(..n_groups * chunk_size))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE dtoh Wilcoxon sums: {e}")))?;

    let mut sums: Vec<Vec<f64>> = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let base = g * chunk_size;
        sums.push(host_sums[base..base + chunk_size].to_vec());
    }
    Ok(sums)
}

fn compute_logfc_hostside(mean_group: f64, mean_ref: f64, log_transformed: bool) -> f64 {
    if log_transformed {
        let g = mean_group.exp_m1();
        let r = mean_ref.exp_m1();
        ((g + LOGFC_PSEUDOCOUNT) / (r + LOGFC_PSEUDOCOUNT)).log2()
    } else {
        (mean_group + LOGFC_PSEUDOCOUNT).log2() - (mean_ref + LOGFC_PSEUDOCOUNT).log2()
    }
}

/// Reconstruct (signed z-score, two-sided p-value) per gene from U + tie term.
///
/// Matches `scx-accel::diffexp::wilcoxon_full_from_ranks`:
///   sigma² = (n1·n2 / 12) · ((N + 1) − tie / (N · (N − 1)))
///   z      = (U − μ) / σ
///   p      = erfc(|z| / √2)                            (two-sided)
///
/// When `tie_correct == false` the CPU path passes `tc = 0.0` to the variance
/// formula — mirror that. The device-side p-value kernel uses the supplied
/// tie term, so we redo p on host whenever ties are intentionally ignored.
fn compute_scores_and_pvals(
    u_host: &[f64],
    tie_host: &[f64],
    n1: usize,
    n2: usize,
    tie_correct: bool,
) -> (Vec<f64>, Vec<f64>) {
    let n1d = n1 as f64;
    let n2d = n2 as f64;
    let n = n1d + n2d;
    let mu = n1d * n2d * 0.5;

    let mut scores = Vec::with_capacity(u_host.len());
    let mut pvals = Vec::with_capacity(u_host.len());
    for (i, &u) in u_host.iter().enumerate() {
        let tc = if tie_correct { tie_host[i] } else { 0.0 };
        let denom = n * (n - 1.0);
        let sigma_sq =
            (n1d * n2d / 12.0) * ((n + 1.0) - if denom > 0.0 { tc / denom } else { 0.0 });
        let (z, p) = if sigma_sq > 0.0 {
            let sigma = sigma_sq.sqrt();
            let z = (u - mu) / sigma;
            let p = libm::erfc(z.abs() / std::f64::consts::SQRT_2);
            (z, p.clamp(0.0, 1.0))
        } else {
            (0.0, 1.0)
        };
        scores.push(z);
        pvals.push(p);
    }
    (scores, pvals)
}

#[allow(clippy::type_complexity)]
fn assemble_chunk_diffexp_result(
    chunk_per_group: Vec<(String, Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>)>,
    rankby_abs: bool,
) -> Option<DiffExpResult> {
    if chunk_per_group.is_empty() {
        return None;
    }
    let mut group_names = Vec::with_capacity(chunk_per_group.len());
    let mut names = Vec::with_capacity(chunk_per_group.len());
    let mut scores = Vec::with_capacity(chunk_per_group.len());
    let mut pvals = Vec::with_capacity(chunk_per_group.len());
    let mut pvals_adj = Vec::with_capacity(chunk_per_group.len());
    let mut logfc = Vec::with_capacity(chunk_per_group.len());

    for (gn, mut g_names, g_scores, g_pvals, g_logfc) in chunk_per_group {
        // Per-group sort by score descending (matches wilcoxon_rank_sum).
        let n = g_scores.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let ka = if rankby_abs {
                g_scores[a].abs()
            } else {
                g_scores[a]
            };
            let kb = if rankby_abs {
                g_scores[b].abs()
            } else {
                g_scores[b]
            };
            kb.partial_cmp(&ka)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let sorted_names: Vec<String> = order
            .iter()
            .map(|&i| std::mem::take(&mut g_names[i]))
            .collect();
        let sorted_scores: Vec<f64> = order.iter().map(|&i| g_scores[i]).collect();
        let sorted_pvals: Vec<f64> = order.iter().map(|&i| g_pvals[i]).collect();
        let sorted_logfc: Vec<f64> = order.iter().map(|&i| g_logfc[i]).collect();
        let bh = benjamini_hochberg(&sorted_pvals);

        group_names.push(gn);
        names.push(sorted_names);
        scores.push(sorted_scores);
        pvals.push(sorted_pvals);
        pvals_adj.push(bh);
        logfc.push(sorted_logfc);
    }

    Some(DiffExpResult {
        group_names,
        names,
        scores,
        pvals,
        pvals_adj,
        logfoldchanges: logfc,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_pdex_inputs(
    data_len: usize,
    n_obs: usize,
    n_vars: usize,
    gene_names_len: usize,
    groups_len: usize,
    group_names_len: usize,
    reference: usize,
    epsilon: f64,
) -> Result<()> {
    if gene_names_len == 0 {
        return Err(AccelError::InvalidInput("gene_names is empty".into()));
    }
    if groups_len == 0 {
        return Err(AccelError::InvalidInput("groups is empty".into()));
    }
    if data_len != n_obs * n_vars {
        return Err(AccelError::InvalidInput(format!(
            "data length {data_len} != n_obs {n_obs} × n_vars {n_vars}"
        )));
    }
    if groups_len != n_obs {
        return Err(AccelError::InvalidInput(format!(
            "groups length {groups_len} != n_obs {n_obs}"
        )));
    }
    if gene_names_len != n_vars {
        return Err(AccelError::InvalidInput(format!(
            "gene_names length {gene_names_len} != n_vars {n_vars}"
        )));
    }
    if reference >= group_names_len {
        return Err(AccelError::InvalidInput(format!(
            "reference index {reference} out of range (n_groups = {group_names_len})"
        )));
    }
    if epsilon < 0.0 || !epsilon.is_finite() {
        return Err(AccelError::InvalidInput(format!(
            "epsilon must be non-negative and finite (got {epsilon})"
        )));
    }
    Ok(())
}

fn validate_wilcoxon_inputs(
    data_len: usize,
    n_obs: usize,
    n_vars: usize,
    gene_names_len: usize,
    groups_len: usize,
) -> Result<()> {
    if gene_names_len == 0 {
        return Err(AccelError::InvalidInput("gene_names is empty".into()));
    }
    if groups_len == 0 {
        return Err(AccelError::InvalidInput("groups is empty".into()));
    }
    if data_len != n_obs * n_vars {
        return Err(AccelError::InvalidInput(format!(
            "data length {data_len} != n_obs {n_obs} × n_vars {n_vars}"
        )));
    }
    if groups_len != n_obs {
        return Err(AccelError::InvalidInput(format!(
            "groups length {groups_len} != n_obs {n_obs}"
        )));
    }
    if gene_names_len != n_vars {
        return Err(AccelError::InvalidInput(format!(
            "gene_names length {gene_names_len} != n_vars {n_vars}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffexp::{pdex_ref, wilcoxon_rank_sum};

    /// Skip the test cleanly when no CUDA GPU is available.
    macro_rules! require_gpu_or_skip {
        () => {
            match scx_gpu::GpuDevice::new(0) {
                Ok(_) => 0usize,
                Err(_) => {
                    eprintln!("CUDA not available — skipping GPU DE test");
                    return;
                }
            }
        };
    }

    /// Deterministic small fixture: 60 cells × 8 genes, 3 groups (ref +
    /// 2 KOs) with engineered fold changes. Mirrors the shape of
    /// `pyscx/tests/test_pdex_ref_parity.py::_make_adata`.
    fn make_fixture() -> (
        Vec<f32>,
        usize,
        usize,
        Vec<String>,
        Vec<usize>,
        Vec<String>,
        usize,
    ) {
        let n_obs = 60usize;
        let n_vars = 8usize;
        // group 0: ref (20 cells), group 1: KO_A (20 cells), group 2: KO_B (20 cells).
        let groups: Vec<usize> = (0..n_obs).map(|i| i / 20).collect();
        let group_names = vec![
            "non-targeting".to_string(),
            "KO_A".to_string(),
            "KO_B".to_string(),
        ];
        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();

        let mut data = vec![0.0f32; n_obs * n_vars];
        let mut state: u64 = 0xDEADBEEFCAFE;
        let mut next_uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // [0, 1)
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };

        for cell in 0..n_obs {
            let g = groups[cell];
            for gene in 0..n_vars {
                // Base rate ~3 per (cell, gene), perturbed for groups 1/2 on
                // a couple of genes.
                let base = 3.0_f64;
                let mut lambda = base;
                if g == 1 && gene == 2 {
                    lambda = base * 2.5;
                }
                if g == 2 && gene == 5 {
                    lambda = base * 2.0;
                }
                // Poisson-ish: approximate by sampling N=10 Bernoulli with
                // p = lambda/10 — produces integer counts in [0, 10].
                let p = (lambda / 10.0).clamp(0.0, 1.0);
                let mut k = 0u32;
                for _ in 0..10 {
                    if next_uniform() < p {
                        k += 1;
                    }
                }
                data[cell * n_vars + gene] = k as f32;
            }
        }
        (data, n_obs, n_vars, gene_names, groups, group_names, 0)
    }

    /// Per-(group × gene) U statistic must match CPU exactly; p-value must
    /// match within `< 1e-9 abs / 1e-6 rel`, target/ref/log2fc within
    /// `< 1e-4 rel` (mirrors `test_pdex_ref_parity.py` tolerance).
    #[test]
    fn test_pdex_ref_gpu_dense_matches_cpu() {
        let _ = require_gpu_or_skip!();

        let (data, n_obs, n_vars, gene_names, groups, group_names, reference) = make_fixture();
        let mode = crate::pseudobulk::GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;

        let cpu = pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            reference,
            mode,
            epsilon,
        )
        .expect("CPU pdex_ref failed");

        let gpu = pdex_ref_gpu_dense(
            0,
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            reference,
            mode,
            epsilon,
        )
        .expect("GPU pdex_ref_dense failed");

        assert_eq!(cpu.group_names, gpu.group_names);
        assert_eq!(cpu.feature_names, gpu.feature_names);
        assert_eq!(cpu.ref_membership, gpu.ref_membership);
        assert_eq!(cpu.target_memberships, gpu.target_memberships);
        for tg in 0..cpu.group_names.len() {
            for var in 0..n_vars {
                let u_cpu = cpu.statistics[tg][var];
                let u_gpu = gpu.statistics[tg][var];
                if u_cpu.is_finite() && u_gpu.is_finite() {
                    assert!(
                        (u_cpu - u_gpu).abs() < 1e-6,
                        "U mismatch tg={tg} gene={var}: cpu={u_cpu}, gpu={u_gpu}"
                    );
                }

                let p_cpu = cpu.p_values[tg][var];
                let p_gpu = gpu.p_values[tg][var];
                assert!(
                    (p_cpu - p_gpu).abs() < 1e-9
                        || (p_cpu - p_gpu).abs() / p_cpu.abs().max(1e-12) < 1e-6,
                    "p-value mismatch tg={tg} gene={var}: cpu={p_cpu}, gpu={p_gpu}"
                );

                let tm_cpu = cpu.target_means[tg][var];
                let tm_gpu = gpu.target_means[tg][var];
                assert!(
                    (tm_cpu - tm_gpu).abs() < 1e-4
                        || (tm_cpu - tm_gpu).abs() / tm_cpu.abs().max(1e-9) < 1e-4,
                    "target_mean mismatch tg={tg} gene={var}: cpu={tm_cpu}, gpu={tm_gpu}"
                );

                let l_cpu = cpu.log2_fold_changes[tg][var];
                let l_gpu = gpu.log2_fold_changes[tg][var];
                if l_cpu.is_finite() && l_gpu.is_finite() {
                    assert!(
                        (l_cpu - l_gpu).abs() < 1e-4
                            || (l_cpu - l_gpu).abs() / l_cpu.abs().max(1e-9) < 1e-4,
                        "log2_fold_change mismatch tg={tg} gene={var}: cpu={l_cpu}, gpu={l_gpu}"
                    );
                }
            }
            for var in 0..n_vars {
                let r_cpu = cpu.ref_means[var];
                let r_gpu = gpu.ref_means[var];
                assert!(
                    (r_cpu - r_gpu).abs() < 1e-4
                        || (r_cpu - r_gpu).abs() / r_cpu.abs().max(1e-9) < 1e-4,
                    "ref_mean mismatch gene={var}: cpu={r_cpu}, gpu={r_gpu}"
                );
            }
        }
    }

    /// Wilcoxon 1-vs-rest: per-(group × gene) U and p must match CPU within
    /// tolerance. Test compares raw p (not BH) since the per-group sort
    /// ordering can differ on ties.
    #[test]
    fn test_wilcoxon_gpu_dense_matches_cpu_one_vs_rest() {
        let _ = require_gpu_or_skip!();

        let (data, n_obs, n_vars, gene_names, groups, group_names, _reference) = make_fixture();

        let cpu = wilcoxon_rank_sum(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            None,  // 1-vs-rest
            false, // not log-transformed
            false, // rankby_abs
            true,  // tie_correct
        )
        .expect("CPU wilcoxon failed");

        let gpu = wilcoxon_rank_sum_gpu_dense(
            0,
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            None,
            false,
            false,
            true,
        )
        .expect("GPU wilcoxon failed");

        // Convert per-group sorted lists into hashmap (gene_name → (score, pval))
        // for stable comparison regardless of tie-broken sort order.
        use std::collections::HashMap;
        let group_to_map = |res: &DiffExpResult, g: usize| -> HashMap<String, (f64, f64)> {
            res.names[g]
                .iter()
                .zip(res.scores[g].iter().zip(res.pvals[g].iter()))
                .map(|(n, (&s, &p))| (n.clone(), (s, p)))
                .collect()
        };

        for g in 0..cpu.group_names.len() {
            let cpu_map = group_to_map(&cpu, g);
            let gpu_map = group_to_map(&gpu, g);
            for gene in &gene_names {
                let (s_cpu, p_cpu) = cpu_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
                let (s_gpu, p_gpu) = gpu_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
                if s_cpu.is_finite() && s_gpu.is_finite() {
                    assert!(
                        (s_cpu - s_gpu).abs() < 1e-6,
                        "score mismatch group={} gene={}: cpu={}, gpu={}",
                        cpu.group_names[g],
                        gene,
                        s_cpu,
                        s_gpu
                    );
                }
                assert!(
                    (p_cpu - p_gpu).abs() < 1e-9
                        || (p_cpu - p_gpu).abs() / p_cpu.abs().max(1e-12) < 1e-6,
                    "pval mismatch group={} gene={}: cpu={}, gpu={}",
                    cpu.group_names[g],
                    gene,
                    p_cpu,
                    p_gpu
                );
            }
        }
    }

    /// Multi-chunk regression: a sparse CSR with `n_vars > gene_chunk_size`
    /// must produce the same per-(group × gene) statistics as the dense
    /// path. This catches the buf-zeroing bug that the 90×15 dense fixture
    /// in `test_pdex_ref_gpu_dense_matches_cpu` is too small to surface:
    /// without a per-chunk `buf.fill(0.0)`, non-zero entries from chunk
    /// N-1 leak into the zero positions of chunk N and corrupt every U.
    #[test]
    fn test_pdex_ref_gpu_sparse_multi_chunk_matches_dense() {
        let _ = require_gpu_or_skip!();

        // 80 cells × 200 genes — chunk_size = 64 forces 4 chunks. Keep
        // the matrix sparse-on-purpose (mostly zeros) so the leak would
        // manifest as non-zero leakage into zero positions.
        let n_obs = 80usize;
        let n_vars = 200usize;
        let groups: Vec<usize> = (0..n_obs).map(|i| i / 40).collect(); // 2 groups of 40
        let group_names = vec!["ref".to_string(), "test".to_string()];
        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g_{i}")).collect();

        // Deterministic sparse fixture: ~10% density, integer counts in [0, 5].
        let mut data = vec![0.0f32; n_obs * n_vars];
        let mut state: u64 = 0xBEEFCAFEBABE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for cell in 0..n_obs {
            for gene in 0..n_vars {
                if (next() % 10) == 0 {
                    data[cell * n_vars + gene] = (next() % 6) as f32;
                }
            }
        }

        // Build an ScxCsr from the dense matrix.
        let mut indptr: Vec<i64> = Vec::with_capacity(n_obs + 1);
        let mut indices: Vec<i32> = Vec::new();
        let mut sparse_data: Vec<f32> = Vec::new();
        indptr.push(0);
        for cell in 0..n_obs {
            for gene in 0..n_vars {
                let v = data[cell * n_vars + gene];
                if v != 0.0 {
                    indices.push(gene as i32);
                    sparse_data.push(v);
                }
            }
            indptr.push(indices.len() as i64);
        }
        let csr = scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, sparse_data);

        let mode = GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;

        let dense_result = pdex_ref_gpu_dense(
            0,
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            mode,
            epsilon,
        )
        .expect("dense GPU pdex_ref failed");

        let sparse_result = pdex_ref_gpu_sparse(
            0,
            &csr,
            &gene_names,
            &groups,
            &group_names,
            0,
            Some(64),
            mode,
            epsilon,
        )
        .expect("sparse GPU pdex_ref failed");

        assert_eq!(dense_result.group_names, sparse_result.group_names);
        for tg in 0..dense_result.group_names.len() {
            for var in 0..n_vars {
                let u_d = dense_result.statistics[tg][var];
                let u_s = sparse_result.statistics[tg][var];
                if u_d.is_finite() && u_s.is_finite() {
                    assert!(
                        (u_d - u_s).abs() < 1e-6,
                        "multi-chunk U mismatch tg={tg} gene={var}: dense={u_d}, sparse={u_s}"
                    );
                }
                let p_d = dense_result.p_values[tg][var];
                let p_s = sparse_result.p_values[tg][var];
                assert!(
                    (p_d - p_s).abs() < 1e-9 || (p_d - p_s).abs() / p_d.abs().max(1e-12) < 1e-6,
                    "multi-chunk p mismatch tg={tg} gene={var}: dense={p_d}, sparse={p_s}"
                );
            }
        }
    }

    /// G1.5 regression: `pdex_ref` GPU vs CPU on a fixture large enough to
    /// force the tiled merge-sort path. At `n_obs = 12_000` with a 50/50
    /// split, `n_ref ≈ 6_000` (fast path) but the test group cells used to
    /// rank against ref also exceed 8192 when chained with ref — the
    /// combined-tie sort + group sort step actually only sees ≤ n_g cells,
    /// so this hits the fast path on ref. To genuinely exercise the
    /// multi-tile path inside `gpu_de_block_sort`, we use a fixture where
    /// the reference group itself is above 8192 cells (12K total, with
    /// 9000 reference cells and 3000 test cells).
    #[test]
    fn test_pdex_ref_gpu_multi_tile_matches_cpu() {
        let _ = require_gpu_or_skip!();

        let n_obs = 12_000usize;
        let n_vars = 6usize;
        // Reference = first 9000 cells (above 8192 → multi-tile sort on ref).
        // Test groups = next 1500 + last 1500.
        let groups: Vec<usize> = (0..n_obs)
            .map(|i| {
                if i < 9000 {
                    0
                } else if i < 10_500 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let group_names = vec![
            "non-targeting".to_string(),
            "KO_A".to_string(),
            "KO_B".to_string(),
        ];
        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();

        // Deterministic Poisson-ish counts; perturb gene 2 in KO_A, gene 4 in KO_B.
        let mut data = vec![0.0f32; n_obs * n_vars];
        let mut state: u64 = 0x1357_2468_ACE0_BDF1;
        let mut next_uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for cell in 0..n_obs {
            let g = groups[cell];
            for gene in 0..n_vars {
                let lambda: f64 = if g == 1 && gene == 2 {
                    7.5
                } else if g == 2 && gene == 4 {
                    6.0
                } else {
                    3.0
                };
                let p = (lambda / 10.0).clamp(0.0, 1.0);
                let mut k = 0u32;
                for _ in 0..10 {
                    if next_uniform() < p {
                        k += 1;
                    }
                }
                data[cell * n_vars + gene] = k as f32;
            }
        }

        let mode = crate::pseudobulk::GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;

        let cpu = pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            mode,
            epsilon,
        )
        .expect("CPU pdex_ref failed");

        let gpu = pdex_ref_gpu_dense(
            0,
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            mode,
            epsilon,
        )
        .expect("GPU pdex_ref_dense failed at n_obs=12000 (n_ref=9000 > 8192)");

        assert_eq!(cpu.group_names, gpu.group_names);
        assert_eq!(cpu.ref_membership, gpu.ref_membership);
        for tg in 0..cpu.group_names.len() {
            for var in 0..n_vars {
                let u_cpu = cpu.statistics[tg][var];
                let u_gpu = gpu.statistics[tg][var];
                if u_cpu.is_finite() && u_gpu.is_finite() {
                    // U statistic is integer-valued; allow a tiny float epsilon
                    // for the host-side accumulator's f64 rounding.
                    assert!(
                        (u_cpu - u_gpu).abs() < 1e-6,
                        "multi-tile U mismatch tg={tg} gene={var}: cpu={u_cpu}, gpu={u_gpu}"
                    );
                }
                let p_cpu = cpu.p_values[tg][var];
                let p_gpu = gpu.p_values[tg][var];
                assert!(
                    (p_cpu - p_gpu).abs() < 1e-9
                        || (p_cpu - p_gpu).abs() / p_cpu.abs().max(1e-12) < 1e-6,
                    "multi-tile p mismatch tg={tg} gene={var}: cpu={p_cpu}, gpu={p_gpu}"
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // G1.7 — dedicated edge-case probes.
    //
    // Each test builds a tiny synthetic fixture targeting one explicit edge
    // case the original G1 spec called out, then asserts CPU↔GPU parity via
    // `pdex_ref_gpu_dense` against the CPU `pdex_ref`. Earlier coverage was
    // transitive (via Poisson-sampled parity); G1.7 adds standalone probes
    // so each edge case has a named failure mode.
    // ---------------------------------------------------------------------

    /// Shared assertion: CPU↔GPU exact U + tolerance-based p / means. Mirrors
    /// the comparison block in `test_pdex_ref_gpu_dense_matches_cpu`.
    fn assert_pdex_parity(cpu: &PdexRefResult, gpu: &PdexRefResult, label: &str) {
        assert_eq!(cpu.group_names, gpu.group_names, "{label}: group_names");
        assert_eq!(
            cpu.feature_names, gpu.feature_names,
            "{label}: feature_names"
        );
        assert_eq!(
            cpu.ref_membership, gpu.ref_membership,
            "{label}: ref_membership"
        );
        assert_eq!(
            cpu.target_memberships, gpu.target_memberships,
            "{label}: target_memberships"
        );
        let n_vars = cpu.feature_names.len();
        for tg in 0..cpu.group_names.len() {
            for var in 0..n_vars {
                let u_cpu = cpu.statistics[tg][var];
                let u_gpu = gpu.statistics[tg][var];
                if u_cpu.is_finite() && u_gpu.is_finite() {
                    assert!(
                        (u_cpu - u_gpu).abs() < 1e-6,
                        "{label}: U mismatch tg={tg} gene={var}: cpu={u_cpu}, gpu={u_gpu}"
                    );
                } else {
                    assert_eq!(
                        u_cpu.is_finite(),
                        u_gpu.is_finite(),
                        "{label}: U finite-mask mismatch tg={tg} gene={var}"
                    );
                }
                let p_cpu = cpu.p_values[tg][var];
                let p_gpu = gpu.p_values[tg][var];
                assert!(
                    (p_cpu - p_gpu).abs() < 1e-9
                        || (p_cpu - p_gpu).abs() / p_cpu.abs().max(1e-12) < 1e-6,
                    "{label}: p mismatch tg={tg} gene={var}: cpu={p_cpu}, gpu={p_gpu}"
                );
                let tm_cpu = cpu.target_means[tg][var];
                let tm_gpu = gpu.target_means[tg][var];
                if tm_cpu.is_finite() && tm_gpu.is_finite() {
                    assert!(
                        (tm_cpu - tm_gpu).abs() < 1e-9
                            || (tm_cpu - tm_gpu).abs() / tm_cpu.abs().max(1e-9) < 1e-6,
                        "{label}: target_mean mismatch tg={tg} gene={var}: cpu={tm_cpu}, gpu={tm_gpu}"
                    );
                }
                let l_cpu = cpu.log2_fold_changes[tg][var];
                let l_gpu = gpu.log2_fold_changes[tg][var];
                if l_cpu.is_finite() && l_gpu.is_finite() {
                    assert!(
                        (l_cpu - l_gpu).abs() < 1e-4
                            || (l_cpu - l_gpu).abs() / l_cpu.abs().max(1e-9) < 1e-4,
                        "{label}: log2_fc mismatch tg={tg} gene={var}: cpu={l_cpu}, gpu={l_gpu}"
                    );
                }
            }
        }
    }

    fn run_pdex_pair(
        data: &[f32],
        n_obs: usize,
        n_vars: usize,
        groups: &[usize],
        group_names: &[String],
        reference: usize,
    ) -> (PdexRefResult, PdexRefResult) {
        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g_{i}")).collect();
        let mode = crate::pseudobulk::GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;
        let cpu = pdex_ref(
            data,
            n_obs,
            n_vars,
            &gene_names,
            groups,
            group_names,
            reference,
            mode,
            epsilon,
        )
        .expect("CPU pdex_ref failed");
        let gpu = pdex_ref_gpu_dense(
            0,
            data,
            n_obs,
            n_vars,
            &gene_names,
            groups,
            group_names,
            reference,
            mode,
            epsilon,
        )
        .expect("GPU pdex_ref_dense failed");
        (cpu, gpu)
    }

    /// Edge case 1 — test group with a single cell.
    /// 11 cells × 3 genes; reference = 10 cells, test_A = {cell 10}.
    /// Hand-crafted counts: gene 0 puts the singleton above the entire ref
    /// distribution; gene 1 puts it below; gene 2 puts it inside.
    #[test]
    fn test_pdex_ref_gpu_group_of_one_cell() {
        let _ = require_gpu_or_skip!();
        let n_obs = 11usize;
        let n_vars = 3usize;
        let mut data = vec![0.0f32; n_obs * n_vars];
        // gene 0: ref counts 0..9 (cells 0..10), singleton = 100 (above all).
        // gene 1: ref counts 10..19, singleton = 0 (below all).
        // gene 2: ref counts 1,2,2,3,3,3,4,4,5,5; singleton = 3 (mid-range with ties).
        let ref_g0: [f32; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let ref_g1: [f32; 10] = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        let ref_g2: [f32; 10] = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0];
        for cell in 0..10 {
            data[cell * n_vars] = ref_g0[cell];
            data[cell * n_vars + 1] = ref_g1[cell];
            data[cell * n_vars + 2] = ref_g2[cell];
        }
        data[10 * n_vars] = 100.0;
        data[10 * n_vars + 1] = 0.0;
        data[10 * n_vars + 2] = 3.0;
        let groups: Vec<usize> = (0..11).map(|i| if i < 10 { 0 } else { 1 }).collect();
        let group_names = vec!["ref".to_string(), "test".to_string()];

        let (cpu, gpu) = run_pdex_pair(&data, n_obs, n_vars, &groups, &group_names, 0);
        // Sanity-check structure before parity assert: 1 test group, 1-cell membership.
        assert_eq!(gpu.target_memberships, vec![1usize]);
        assert_eq!(gpu.ref_membership, 10);
        // All p_values must be finite (no NaN from divide-by-zero sigma at n_g=1).
        for var in 0..n_vars {
            assert!(
                gpu.p_values[0][var].is_finite(),
                "group-of-one p-value must be finite at gene {var}, got {}",
                gpu.p_values[0][var]
            );
        }
        assert_pdex_parity(&cpu, &gpu, "group_of_one_cell");
    }

    /// Edge case 2 — gene with zero counts in every cell.
    /// 20 cells × 4 genes, gene 2 is all-zero. ref={0..9}, test_A={10..19}.
    #[test]
    fn test_pdex_ref_gpu_all_zero_gene() {
        let _ = require_gpu_or_skip!();
        let n_obs = 20usize;
        let n_vars = 4usize;
        let mut data = vec![0.0f32; n_obs * n_vars];
        // Non-zero genes: deterministic spread.
        for cell in 0..n_obs {
            data[cell * n_vars + 0] = (cell as f32) % 5.0;
            data[cell * n_vars + 1] = (cell as f32) * 0.5;
            // gene 2 left as 0.0
            data[cell * n_vars + 3] = if cell < 10 { 1.0 } else { 3.0 };
        }
        let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();
        let group_names = vec!["ref".to_string(), "test".to_string()];

        let (cpu, gpu) = run_pdex_pair(&data, n_obs, n_vars, &groups, &group_names, 0);
        // For gene 2 (all-zero), both means = 0 and the CPU reports U = n1·n2/2,
        // p = 1 (everything tied). Verify GPU matches:
        assert_eq!(gpu.target_means[0][2], 0.0);
        assert_eq!(gpu.ref_means[2], 0.0);
        assert!(
            (gpu.p_values[0][2] - 1.0).abs() < 1e-9,
            "all-zero gene must give p = 1, got {}",
            gpu.p_values[0][2]
        );
        assert_pdex_parity(&cpu, &gpu, "all_zero_gene");
    }

    /// Edge case 3 — reference smaller than test group (n_ref=3, n_test=30).
    /// Exercises the asymmetric `n1 / n2` path in the variance formula.
    #[test]
    fn test_pdex_ref_gpu_ref_smaller_than_test_group() {
        let _ = require_gpu_or_skip!();
        let n_obs = 33usize;
        let n_vars = 4usize;
        let mut data = vec![0.0f32; n_obs * n_vars];
        // Deterministic-ish values; doesn't matter much, just want spread.
        let mut state: u64 = 0xCAFEBABE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 20) as f32
        };
        for v in data.iter_mut() {
            *v = next();
        }
        // Reference = first 3 cells; test group A = remaining 30.
        let groups: Vec<usize> = (0..n_obs).map(|i| if i < 3 { 0 } else { 1 }).collect();
        let group_names = vec!["ref".to_string(), "test".to_string()];

        let (cpu, gpu) = run_pdex_pair(&data, n_obs, n_vars, &groups, &group_names, 0);
        assert_eq!(gpu.ref_membership, 3);
        assert_eq!(gpu.target_memberships, vec![30usize]);
        assert_pdex_parity(&cpu, &gpu, "ref_smaller_than_test_group");
    }

    /// Edge case 4 — test group has identical values for one gene
    /// (zero-variance group). Exercises the combined-tie-term path on a
    /// pathologically tie-heavy input.
    #[test]
    fn test_pdex_ref_gpu_all_equal_values_in_group() {
        let _ = require_gpu_or_skip!();
        let n_obs = 20usize;
        let n_vars = 3usize;
        let mut data = vec![0.0f32; n_obs * n_vars];
        // Gene 0: normal spread.
        // Gene 1: ref has spread; test group A is constant 5.0 (zero-variance).
        // Gene 2: both groups have spread but several values tie with each other.
        for cell in 0..n_obs {
            data[cell * n_vars + 0] = ((cell as f32) % 7.0) + 0.5;
            data[cell * n_vars + 1] = if cell < 10 {
                (cell as f32) % 11.0 // ref: spread including 5.0 a few times
            } else {
                5.0 // test_A: constant
            };
            data[cell * n_vars + 2] = if cell % 3 == 0 { 2.0 } else { 4.0 };
        }
        let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();
        let group_names = vec!["ref".to_string(), "test".to_string()];

        let (cpu, gpu) = run_pdex_pair(&data, n_obs, n_vars, &groups, &group_names, 0);
        // Sanity-check the test group for gene 1 truly is constant.
        for cell in 10..n_obs {
            assert_eq!(data[cell * n_vars + 1], 5.0);
        }
        assert_pdex_parity(&cpu, &gpu, "all_equal_values_in_group");
    }
}
