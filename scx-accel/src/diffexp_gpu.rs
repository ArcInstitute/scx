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

use scx_format::ShardSource;
use scx_gpu::{
    build_cell_to_group_dev, build_cell_to_pos_dev, cuda_graphs_enabled,
    default_gpu_de_gene_chunk_size, gpu_de_block_sort, gpu_de_combined_tie_term,
    gpu_de_pseudobulk_csc_direct, gpu_de_pseudobulk_csr_direct, gpu_de_pvalues,
    gpu_de_scatter_csc_to_gene_major, gpu_de_scatter_csr_to_gene_major_filtered,
    gpu_de_searchsorted_ranksum, gpu_de_searchsorted_u_stat, gpu_de_tie_term,
    BackedGpuMatrixSource, CudaSlice, GpuDevice, GpuMatrixSource,
};

use crate::diffexp::{benjamini_hochberg, merge_diff_exp_results, DiffExpResult, PdexRefResult};
use crate::pseudobulk::GeomMeanMode;
use crate::route::{
    plan_de_route_from_source, AccelExecutionInfo, AccelRoute, DeviceRequest, InputLayout,
};
use crate::{AccelError, Result};

/// Stamp the planned execution info onto a `pdex_ref` GPU result and emit a
/// structured route log line. The route is decided up front by
/// [`plan_de_route`] (the single source of truth) so the stamped route always
/// matches the kernel branch that actually ran. Replaces the old
/// `SCX_GPU_DE_V3_TRACE` stderr trace as the primary signal (the trace
/// survives as a debug-only fallback inside the v3 drivers).
fn finish_pdex(
    result: Result<PdexRefResult>,
    mut info: AccelExecutionInfo,
) -> Result<PdexRefResult> {
    log::debug!(
        "scx-accel pdex_ref GPU route: {} (fallback: {})",
        info.route.as_str(),
        info.fallback_reason.as_str()
    );
    result.map(|mut r| {
        // The route/fallback come from the planner; carry the driver-measured
        // shard counts (set on the chunk driver's result) through the stamp.
        info.shards_decoded = info.shards_decoded.or(r.exec_info.shards_decoded);
        info.shards_uploaded = info.shards_uploaded.or(r.exec_info.shards_uploaded);
        r.exec_info = info;
        r
    })
}

/// Stamp the planned execution info onto a Wilcoxon GPU result. Shard inputs
/// are routed by [`plan_de_route_from_source`] (CSC-direct `gpu_csc_v3` when a
/// CSC sidecar is present, else CSR-direct `gpu_csr_v3`); dense-host input is
/// densified to CSR and follows the `gpu_csr_v3` path.
fn finish_de(result: Result<DiffExpResult>, mut info: AccelExecutionInfo) -> Result<DiffExpResult> {
    log::debug!(
        "scx-accel wilcoxon GPU route: {} (fallback: {})",
        info.route.as_str(),
        info.fallback_reason.as_str()
    );
    result.map(|mut r| {
        // Carry the driver-measured shard counts through the planner stamp.
        info.shards_decoded = info.shards_decoded.or(r.exec_info.shards_decoded);
        info.shards_uploaded = info.shards_uploaded.or(r.exec_info.shards_uploaded);
        r.exec_info = info;
        r
    })
}

/// Pseudocount used by the CPU `compute_logfc` helper. Mirrors
/// `diffexp::LOGFC_PSEUDOCOUNT` (private there).
const LOGFC_PSEUDOCOUNT: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// pdex `mode="ref"` on a dense `[n_obs × n_vars]` row-major buffer (single
/// chunk; for sparse / backed / lazy inputs use [`pdex_ref_gpu`] with the
/// matching [`GpuDeShardInput`] variant).
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
        Some(data.len()),
        n_obs,
        n_vars,
        gene_names.len(),
        groups.len(),
        group_names.len(),
        reference,
        epsilon,
    )?;

    // Dense host input is densified into an in-memory CSR and routed through the
    // shared v3 CSR path. The transient CSR carries no CSC
    // sidecar, so the planner records `gpu_csr_v3` with `NoCscSidecar`. Dense-host
    // DE is the rarest GPU DE input, so the one-time O(n_obs × n_vars) densify→CSR
    // cost is negligible versus the DE compute.
    let csr = scx_sparse::dense_to_csr(data, n_obs, n_vars)
        .map_err(|e| AccelError::LinAlg(format!("dense→CSR for GPU DE: {e}")))?;
    pdex_ref_gpu(
        device_id,
        GpuDeShardInput::Csr(&csr),
        gene_names,
        groups,
        group_names,
        reference,
        None,
        mode,
        epsilon,
    )
}

/// Already-extracted Rust input for the unified GPU DE entry points
/// ([`pdex_ref_gpu`] / [`wilcoxon_rank_sum_gpu`]). Python-level type detection
/// stays in `pyscx`; this captures the three shard-shaped inputs that all
/// reduce to "a CSR shard source + an optional CSC sidecar". Dense host input
/// is handled separately by [`pdex_ref_gpu_dense`] /
/// [`wilcoxon_rank_sum_gpu_dense`], which densify the host buffer to an
/// in-memory CSR and delegate here via [`GpuDeShardInput::Csr`] (the v3 CSR
/// path); it carries no CSC sidecar.
pub enum GpuDeShardInput<'a> {
    /// In-memory scipy-style CSR. Never carries a CSC sidecar.
    Csr(&'a scx_sparse::ScxCsr),
    /// SCX-backed CSR reader plus an optional gene-major CSC sidecar. When the
    /// sidecar is present and v3 is enabled, dispatch routes CSC-direct.
    Backed {
        csr: &'a scx_format::backed::BackedCsrReader,
        csc: Option<&'a scx_format::backed::BackedCscReader>,
    },
    /// Generic lazy `ShardSource` (CSR-shaped; no CSC capability surface).
    Lazy(&'a (dyn ShardSource + Sync)),
}

impl GpuDeShardInput<'_> {
    /// The route-planner input layout for this input shape.
    fn input_layout(&self) -> InputLayout {
        match self {
            GpuDeShardInput::Csr(_) => InputLayout::CsrHost,
            GpuDeShardInput::Backed { csc: Some(_), .. } => InputLayout::BackedCsc,
            GpuDeShardInput::Backed { csc: None, .. } => InputLayout::BackedCsr,
            GpuDeShardInput::Lazy(_) => InputLayout::LazyCsr,
        }
    }

    /// `(n_obs, n_vars)`. The in-memory CSR shape is authoritative; backed/lazy
    /// take `n_vars` from `gene_names` / the source (matching the former
    /// per-shape entry points).
    fn shape(&self, gene_names_len: usize) -> (usize, usize) {
        match self {
            GpuDeShardInput::Csr(csr) => csr.shape,
            GpuDeShardInput::Backed { csr, .. } => (csr.n_obs(), gene_names_len),
            GpuDeShardInput::Lazy(source) => (source.n_obs(), source.n_vars()),
        }
    }
}

/// pdex `mode="ref"` on any shard-shaped input (in-memory CSR / SCX-backed
/// CSR+optional-CSC / lazy `ShardSource`). Replaces the former
/// `pdex_ref_gpu_{sparse,streaming,lazy}` trio: all three reduce to a
/// [`BackedGpuMatrixSource`] whose [`available_layouts`](GpuMatrixSource::available_layouts)
/// drive the route via [`plan_de_route_from_source`]. Dense input uses
/// [`pdex_ref_gpu_dense`].
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_gpu(
    device_id: usize,
    input: GpuDeShardInput<'_>,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    gene_chunk_size: Option<usize>,
    mode: GeomMeanMode,
    epsilon: f64,
) -> Result<PdexRefResult> {
    let (n_obs, n_vars) = input.shape(gene_names.len());
    validate_pdex_inputs(
        None, // no dense buffer — dim guard still runs in the validator
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
    let layout = input.input_layout();

    // Construct the matrix source per input shape — the in-memory `Csr` arm
    // needs a local `InMemoryCsrShardSource` adaptor that must outlive the
    // source, so construction stays in each arm; the shared route dispatch
    // runs on `&mut dyn GpuMatrixSource` in `pdex_ref_gpu_dispatch`.
    match input {
        GpuDeShardInput::Csr(csr) => {
            let src = scx_gpu::InMemoryCsrShardSource::new(csr);
            let mut source = BackedGpuMatrixSource::new(&dev, &src)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE matrix source init: {e}")))?;
            pdex_ref_gpu_dispatch(
                &dev,
                &mut source,
                layout,
                n_obs,
                n_vars,
                chunk_size,
                gene_names,
                groups,
                group_names,
                reference,
                mode,
                epsilon,
            )
        }
        GpuDeShardInput::Lazy(source_dyn) => {
            let mut source = BackedGpuMatrixSource::new(&dev, source_dyn)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE matrix source init: {e}")))?;
            pdex_ref_gpu_dispatch(
                &dev,
                &mut source,
                layout,
                n_obs,
                n_vars,
                chunk_size,
                gene_names,
                groups,
                group_names,
                reference,
                mode,
                epsilon,
            )
        }
        GpuDeShardInput::Backed { csr, csc } => {
            let mut source = match csc {
                Some(csc) => BackedGpuMatrixSource::with_csc(&dev, csr, csc),
                None => BackedGpuMatrixSource::new(&dev, csr),
            }
            .map_err(|e| AccelError::LinAlg(format!("GPU DE matrix source init: {e}")))?;
            pdex_ref_gpu_dispatch(
                &dev,
                &mut source,
                layout,
                n_obs,
                n_vars,
                chunk_size,
                gene_names,
                groups,
                group_names,
                reference,
                mode,
                epsilon,
            )
        }
    }
}

/// Shared pdex_ref route dispatch over a `&mut dyn GpuMatrixSource`. The route
/// is decided once by [`plan_de_route_from_source`] (reading the source's CSC
/// capability) so the stamped route always matches the kernel that runs.
#[allow(clippy::too_many_arguments)]
fn pdex_ref_gpu_dispatch(
    dev: &GpuDevice,
    source: &mut dyn GpuMatrixSource,
    layout: InputLayout,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    mode: GeomMeanMode,
    epsilon: f64,
) -> Result<PdexRefResult> {
    let mut exec_info = plan_de_route_from_source(DeviceRequest::Gpu, source, layout);
    exec_info.chunk_size = Some(chunk_size);
    finish_pdex(
        match exec_info.route {
            AccelRoute::GpuCscV3 => pdex_ref_gpu_chunked_v3_csc(
                dev,
                n_obs,
                n_vars,
                chunk_size,
                gene_names,
                groups,
                group_names,
                reference,
                mode,
                epsilon,
                source,
            ),
            AccelRoute::GpuCsrV3 => pdex_ref_gpu_chunked_v3_csr(
                dev,
                n_obs,
                n_vars,
                chunk_size,
                gene_names,
                groups,
                group_names,
                reference,
                mode,
                epsilon,
                source,
            ),
            other => Err(AccelError::LinAlg(format!(
                "pdex_ref_gpu: planner returned unreachable route {} for {:?}",
                other.as_str(),
                layout
            ))),
        },
        exec_info,
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
    validate_wilcoxon_inputs(
        Some(data.len()),
        n_obs,
        n_vars,
        gene_names.len(),
        groups.len(),
    )?;

    // Dense host input is densified into an in-memory CSR and routed through the
    // shared v3 CSR path, reporting `gpu_csr_v3` with
    // `NoCscSidecar` — identical discipline to `pdex_ref_gpu_dense`.
    let csr = scx_sparse::dense_to_csr(data, n_obs, n_vars)
        .map_err(|e| AccelError::LinAlg(format!("dense→CSR for GPU DE: {e}")))?;
    wilcoxon_rank_sum_gpu(
        device_id,
        GpuDeShardInput::Csr(&csr),
        gene_names,
        groups,
        group_names,
        reference,
        None,
        log_transformed,
        rankby_abs,
        tie_correct,
    )
}

/// Wilcoxon rank-sum on any shard-shaped input (in-memory CSR / SCX-backed /
/// lazy `ShardSource`). The route is decided by [`plan_de_route_from_source`]
/// from the input layout + the source's CSC capability: a `Backed` input with a
/// CSC sidecar reaches the CSC-direct v3 driver (`gpu_csc_v3`); otherwise the
/// CSR-direct v3 driver (`gpu_csr_v3`). Dense input is densified to CSR by
/// [`wilcoxon_rank_sum_gpu_dense`] and also follows the `gpu_csr_v3` path.
#[allow(clippy::too_many_arguments)]
pub fn wilcoxon_rank_sum_gpu(
    device_id: usize,
    input: GpuDeShardInput<'_>,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    gene_chunk_size: Option<usize>,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
) -> Result<DiffExpResult> {
    let (n_obs, n_vars) = input.shape(gene_names.len());
    validate_wilcoxon_inputs(None, n_obs, n_vars, gene_names.len(), groups.len())?;

    let dev = open_device(device_id)?;
    let chunk_size = resolve_chunk_size(&dev, n_obs, gene_chunk_size, n_vars);

    match input {
        GpuDeShardInput::Csr(csr) => {
            let src = scx_gpu::InMemoryCsrShardSource::new(csr);
            let mut source = BackedGpuMatrixSource::new(&dev, &src)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE matrix source init: {e}")))?;
            wilcoxon_rank_sum_gpu_dispatch(
                &dev,
                &mut source,
                InputLayout::CsrHost,
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
            )
        }
        GpuDeShardInput::Lazy(source_dyn) => {
            let mut source = BackedGpuMatrixSource::new(&dev, source_dyn)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE matrix source init: {e}")))?;
            wilcoxon_rank_sum_gpu_dispatch(
                &dev,
                &mut source,
                InputLayout::LazyCsr,
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
            )
        }
        GpuDeShardInput::Backed { csr, csc } => {
            // Hand the CSC sidecar (when present) to the matrix source so the
            // planner can reach the CSC-direct v3 Wilcoxon driver.
            let layout = if csc.is_some() {
                InputLayout::BackedCsc
            } else {
                InputLayout::BackedCsr
            };
            let mut source = match csc {
                Some(csc) => BackedGpuMatrixSource::with_csc(&dev, csr, csc),
                None => BackedGpuMatrixSource::new(&dev, csr),
            }
            .map_err(|e| AccelError::LinAlg(format!("GPU DE matrix source init: {e}")))?;
            wilcoxon_rank_sum_gpu_dispatch(
                &dev,
                &mut source,
                layout,
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
            )
        }
    }
}

/// Shared Wilcoxon route dispatch over a `&mut dyn GpuMatrixSource`. The route
/// is decided once by [`plan_de_route_from_source`] (reading the source's CSC
/// capability) so the stamped route always matches the kernel that runs —
/// identical discipline to [`pdex_ref_gpu_dispatch`]. Sparse GPU DE is always
/// v3: `gpu_csc_v3` with a CSC sidecar, `gpu_csr_v3` otherwise.
#[allow(clippy::too_many_arguments)]
fn wilcoxon_rank_sum_gpu_dispatch(
    dev: &GpuDevice,
    source: &mut dyn GpuMatrixSource,
    layout: InputLayout,
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
) -> Result<DiffExpResult> {
    let mut exec_info = plan_de_route_from_source(DeviceRequest::Gpu, source, layout);
    exec_info.chunk_size = Some(chunk_size);
    finish_de(
        match exec_info.route {
            AccelRoute::GpuCscV3 => wilcoxon_rank_sum_gpu_chunked_v3_csc(
                dev,
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
                source,
            ),
            AccelRoute::GpuCsrV3 => wilcoxon_rank_sum_gpu_chunked_v3_csr(
                dev,
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
                source,
            ),
            other => Err(AccelError::LinAlg(format!(
                "wilcoxon_rank_sum_gpu: planner returned unreachable route {} for {:?}",
                other.as_str(),
                layout
            ))),
        },
        exec_info,
    )
}

// ---------------------------------------------------------------------------
// Shared chunked drivers
// ---------------------------------------------------------------------------

/// Per-chunk Wilcoxon GPU statistical sequence for the CSC/CSR-direct drivers.
/// It does **not** scatter from a dense buffer: the pool slab
/// (`scratch.ref_slab`, `pool_len` rows) and each test group's slab
/// (`scratch.per_tg_pool_slabs[tg_idx]`, `n_g` rows) are pre-populated by the
/// shard pass (CSC / CSR-direct scatter) — like
/// [`pdex_ref_chunk_gpu_sequence_v2`] — while keeping Wilcoxon's
/// U-stat / ranksum / combined-tie statistical kernels.
///
/// Pre-condition: `scratch.ref_slab[..sz * pool_len]` and, for each non-empty
/// test group, `scratch.per_tg_pool_slabs[tg_idx][..sz * n_g]` are populated;
/// `scratch.slab_aux` / `scratch.u_or_rank` / `scratch.tie_term` /
/// per-group slabs pre-grown by the caller. Empty test groups (`n_g == 0`)
/// are skipped (the host post-pass synthesises NaN scores / p = 1 for them).
#[allow(clippy::too_many_arguments)]
fn wilcoxon_chunk_gpu_sequence_v3(
    dev: &GpuDevice,
    scratch: &mut scx_gpu::GpuDeChunkScratch,
    test_groups: &[usize],
    group_indices: &[Vec<usize>],
    sz: usize,
    pool_len: usize,
    chunk_max: usize,
    is_ref_mode: bool,
) -> Result<()> {
    // Pool slab is pre-populated by the shard pass; sort + tie on it. The pool
    // is the reference group in ref-mode or all cells in 1-vs-rest; either way
    // `scratch.ref_slab` holds it and `scratch.tie_term` holds its tie term.
    gpu_de_block_sort(
        dev,
        &mut scratch.ref_slab,
        &mut scratch.slab_aux,
        sz,
        pool_len,
    )
    .map_err(|e| AccelError::LinAlg(format!("GPU DE sort pool (wil v3): {e}")))?;
    gpu_de_tie_term(dev, &scratch.ref_slab, &mut scratch.tie_term, sz, pool_len)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE pool tie (wil v3): {e}")))?;

    // Per-test-group sequence. Each tg's slab is pre-populated; no scatter.
    for (tg_idx, &g) in test_groups.iter().enumerate() {
        let n_g = group_indices[g].len();
        if n_g == 0 {
            continue;
        }

        if is_ref_mode {
            // U1 from ref searchsorted → scratch.u_or_rank.
            gpu_de_searchsorted_u_stat(
                dev,
                &scratch.ref_slab,
                &scratch.per_tg_pool_slabs[tg_idx],
                &mut scratch.u_or_rank,
                sz,
                pool_len,
                n_g,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU DE searchsorted U (wil v3): {e}")))?;
            // Combined tie for ref ∪ group → scratch.tie_term (overwrites the
            // pool tie; unused downstream in ref-mode).
            gpu_de_block_sort(
                dev,
                &mut scratch.per_tg_pool_slabs[tg_idx],
                &mut scratch.slab_aux,
                sz,
                n_g,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU DE sort group (wil v3): {e}")))?;
            gpu_de_combined_tie_term(
                dev,
                &scratch.ref_slab,
                &scratch.per_tg_pool_slabs[tg_idx],
                &mut scratch.tie_term,
                sz,
                pool_len,
                n_g,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU DE combined tie (wil v3): {e}")))?;
            let tie_off = tg_idx * chunk_max;
            let tie_src = scratch
                .tie_term
                .try_slice(..sz)
                .ok_or_else(|| AccelError::LinAlg("tie_term source slice OOB (wil v3)".into()))?;
            let mut tie_dst = scratch
                .tie_per_group
                .try_slice_mut(tie_off..tie_off + sz)
                .ok_or_else(|| {
                    AccelError::LinAlg("tie_per_group dest slice OOB (wil v3)".into())
                })?;
            dev.stream()
                .memcpy_dtod(&tie_src, &mut tie_dst)
                .map_err(|e| AccelError::LinAlg(format!("stage tie (wil v3): {e}")))?;
        } else {
            // Rank sum via all-searchsorted → scratch.u_or_rank. No per-tg tie
            // work; the pool tie in scratch.tie_term is the global correction.
            gpu_de_searchsorted_ranksum(
                dev,
                &scratch.ref_slab,
                &scratch.per_tg_pool_slabs[tg_idx],
                &mut scratch.u_or_rank,
                sz,
                pool_len,
                n_g,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU DE searchsorted rank (wil v3): {e}")))?;
        }

        // Stage u_or_rank → u_per_group (both modes).
        let u_off = tg_idx * chunk_max;
        let u_src = scratch
            .u_or_rank
            .try_slice(..sz)
            .ok_or_else(|| AccelError::LinAlg("u_or_rank source slice OOB (wil v3)".into()))?;
        let mut u_dst = scratch
            .u_per_group
            .try_slice_mut(u_off..u_off + sz)
            .ok_or_else(|| AccelError::LinAlg("u_per_group dest slice OOB (wil v3)".into()))?;
        dev.stream()
            .memcpy_dtod(&u_src, &mut u_dst)
            .map_err(|e| AccelError::LinAlg(format!("stage U (wil v3): {e}")))?;
    }
    Ok(())
}

/// G4 v2: pdex_ref per-chunk GPU kernel sequence WITHOUT per-chunk scatter
/// kernel calls. The v2 chunk loop pre-populates `scratch.ref_slab` and
/// `scratch.per_tg_pool_slabs[..]` directly from the CSR shard source via
/// [`gpu_de_scatter_shard_to_gene_major`] before invoking this sequence,
/// so this function starts with sort + tie on already-filled slabs.
///
/// Versus a scatter-from-dense sequence:
/// - Skips `gpu_de_scatter_gene_major` for ref (slab is pre-populated).
/// - Skips `gpu_de_scatter_gene_major` for each test group; reads from
///   `scratch.per_tg_pool_slabs[tg_idx]` instead of `scratch.group_slab`.
///
/// Same numerical contract as v1 — produces identical U / p / tie values
/// from the same input data; the only change is HOW the slabs got
/// populated. (The v2-vs-v1 parity test was removed with the V1b
/// default-flip, after which the planner no longer reaches this driver.)
///
/// Empty test groups (`n_g == 0`) are skipped — those rows in the
/// per-group U / p slabs are LEFT UNINITIALIZED; the caller handles the
/// empty-group case via the same host-side post-pass as v1.
#[allow(clippy::too_many_arguments)]
fn pdex_ref_chunk_gpu_sequence_v2(
    dev: &GpuDevice,
    scratch: &mut scx_gpu::GpuDeChunkScratch,
    test_groups: &[usize],
    group_indices: &[Vec<usize>],
    sz: usize,
    n_ref: usize,
    chunk_max: usize,
) -> Result<()> {
    // Ref slab is pre-populated by the v2 shard pass; sort + tie on it.
    gpu_de_block_sort(dev, &mut scratch.ref_slab, &mut scratch.slab_aux, sz, n_ref)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE sort ref (v2): {e}")))?;
    gpu_de_tie_term(dev, &scratch.ref_slab, &mut scratch.tie_term, sz, n_ref)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE tie term ref (v2): {e}")))?;

    // Per-test-group sequence. The pool slab for each tg is pre-populated;
    // sort + searchsort + combined-tie + p-values + stage_dtod on it.
    for (tg_idx, &g) in test_groups.iter().enumerate() {
        let n_g = group_indices[g].len();
        if n_g == 0 {
            continue;
        }
        // Pull the per-tg slab via index (can't borrow the vec mutably
        // while sharing `scratch` — use swap_remove-style indexed access).
        // We hold `&mut scratch.per_tg_pool_slabs[tg_idx]` for the duration
        // of this group's kernel calls; cleared at end of block.
        // ref_slab borrow ends after the searchsorted call below.

        // U1 = searchsorted(ref, group)
        gpu_de_searchsorted_u_stat(
            dev,
            &scratch.ref_slab,
            &scratch.per_tg_pool_slabs[tg_idx],
            &mut scratch.u_or_rank,
            sz,
            n_ref,
            n_g,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU DE searchsorted U (v2): {e}")))?;

        // Sort the per-tg slab in place, then combined-tie over ref ∪ group.
        gpu_de_block_sort(
            dev,
            &mut scratch.per_tg_pool_slabs[tg_idx],
            &mut scratch.slab_aux,
            sz,
            n_g,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU DE sort group (v2): {e}")))?;
        gpu_de_combined_tie_term(
            dev,
            &scratch.ref_slab,
            &scratch.per_tg_pool_slabs[tg_idx],
            &mut scratch.tie_term,
            sz,
            n_ref,
            n_g,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU DE combined tie (v2): {e}")))?;

        gpu_de_pvalues(
            dev,
            &scratch.u_or_rank,
            &scratch.tie_term,
            &mut scratch.p_values,
            sz,
            n_g,
            n_ref,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU DE p-value (v2): {e}")))?;

        // Stage U and p into per-group slabs via on-device memcpy_dtod.
        let u_off = tg_idx * chunk_max;
        let p_off = tg_idx * chunk_max;
        let u_src = scratch
            .u_or_rank
            .try_slice(..sz)
            .ok_or_else(|| AccelError::LinAlg("u_or_rank source slice OOB (v2)".into()))?;
        let mut u_dst = scratch
            .u_per_group
            .try_slice_mut(u_off..u_off + sz)
            .ok_or_else(|| AccelError::LinAlg("u_per_group dest slice OOB (v2)".into()))?;
        dev.stream()
            .memcpy_dtod(&u_src, &mut u_dst)
            .map_err(|e| AccelError::LinAlg(format!("stage U (v2): {e}")))?;
        let p_src = scratch
            .p_values
            .try_slice(..sz)
            .ok_or_else(|| AccelError::LinAlg("p_values source slice OOB (v2)".into()))?;
        let mut p_dst = scratch
            .p_per_group
            .try_slice_mut(p_off..p_off + sz)
            .ok_or_else(|| AccelError::LinAlg("p_per_group dest slice OOB (v2)".into()))?;
        dev.stream()
            .memcpy_dtod(&p_src, &mut p_dst)
            .map_err(|e| AccelError::LinAlg(format!("stage p (v2): {e}")))?;
    }
    Ok(())
}

/// G4 v3 helper: compute pdex_ref target/reference means from a
/// pre-populated `d_sums` `[n_groups × chunk_size]` f64 buffer. It does **not**
/// launch a pseudobulk kernel — the v3 driver has already accumulated sums via
/// either the CSC-direct or CSR-direct pseudobulk kernels. The only work left
/// is dtoh + per-group divide + `mode.post`.
fn compute_pdex_means_from_sums(
    dev: &GpuDevice,
    d_sums: &CudaSlice<f64>,
    chunk_size: usize,
    n_ref: usize,
    target_memberships: &[usize],
    mode: GeomMeanMode,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n_test = target_memberships.len();
    let n_groups = 1 + n_test;
    let host_sums = dev
        .stream()
        .clone_dtoh(&d_sums.slice(..n_groups * chunk_size))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 dtoh sums: {e}")))?;
    let ref_means: Vec<f64> = host_sums
        .iter()
        .take(chunk_size)
        .map(|&s| mode.post(if n_ref == 0 { 0.0 } else { s / n_ref as f64 }))
        .collect();
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

/// G4 v3 CSR-fallback driver for `pdex_ref`. Same shape as v2 but drops
/// the `[n_obs × chunk_size]` dense materialization step: per chunk, zeros
/// ref + per-tg pool slabs + sums, walks the CSR shard source ONCE per
/// chunk, populating the per-tg pool slabs (via G4.1's
/// `gpu_de_scatter_shard_to_gene_major`) and pseudobulk sums (via the new
/// `gpu_de_pseudobulk_csr_direct`) in a single pass per shard. No dense
/// scatter, no `gpu_de_pseudobulk_all_groups` call.
///
/// Used when the source's
/// [`available_layouts`](GpuMatrixSource::available_layouts) does not contain
/// [`LayoutSet::CSC`](scx_gpu::LayoutSet) (in-memory CSR, or backed/lazy with no
/// CSC sidecar). The CSC-direct equivalent [`pdex_ref_gpu_chunked_v3_csc`] is
/// the primary v3 path when a CSC sidecar is available.
#[allow(clippy::too_many_arguments)]
fn pdex_ref_gpu_chunked_v3_csr(
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
    source: &mut dyn GpuMatrixSource,
) -> Result<PdexRefResult> {
    if chunk_size == 0 {
        return Err(AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }

    // One-shot dispatch trace for bench validation. Gated on a separate
    // env var so it doesn't fire in unit tests or production. Sets
    // SCX_GPU_DE_V3_DISPATCH=v3-csr in the process env as a parallel
    // signal that callers (bench harness) can read post-call if they
    // can't capture stderr.
    if std::env::var("SCX_GPU_DE_V3_TRACE").is_ok() {
        eprintln!("[scx-accel/pdex_ref] v3 dispatch route: csr-direct (no CSC sidecar)");
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
    let test_groups: Vec<usize> = (0..n_groups).filter(|&g| g != reference).collect();
    let target_memberships: Vec<usize> = test_groups
        .iter()
        .map(|&g| group_indices[g].len())
        .collect();
    let n_g_max = target_memberships.iter().copied().max().unwrap_or(0);

    let ref_idx_i32: Vec<i32> = ref_cells.iter().map(|&c| c as i32).collect();
    let group_idx_i32: Vec<Vec<i32>> = test_groups
        .iter()
        .map(|&g| group_indices[g].iter().map(|&c| c as i32).collect())
        .collect();

    // Build all_cells / offsets / cell_to_group for the v3 pseudobulk kernel.
    let n_groups_for_means = 1 + test_groups.len();
    let mut all_cells_host: Vec<i32> = Vec::with_capacity(n_ref + n_g_max * test_groups.len());
    let mut offsets_host: Vec<i32> = Vec::with_capacity(n_groups_for_means + 1);
    offsets_host.push(0);
    all_cells_host.extend(ref_idx_i32.iter().copied());
    offsets_host.push(checked_offset_i32(
        all_cells_host.len(),
        "GPU DE cell-permutation offsets",
    )?);
    for cells in &group_idx_i32 {
        all_cells_host.extend(cells.iter().copied());
        offsets_host.push(checked_offset_i32(
            all_cells_host.len(),
            "GPU DE cell-permutation offsets",
        )?);
    }
    let mode_id = geom_mean_mode_id(mode);

    // Code-review #6: replace K+1 per-pool `cell_to_pool` tables with two
    // tables that scale constant in K (group_id table + in-group-pos table).
    // Device memory: (K+2)·n_obs·4 → 2·n_obs·4 (atlas-scale win — 4 GB → 80 MB
    // at n_obs=10M, K=100). v3 scatter launches pass `this_group_id` per
    // launch; the filtered scatter kernel checks `cell_to_group[cell] ==
    // this_group_id` and writes to `slab[…, cell_to_pos[cell]]`. v2 driver
    // still uses the per-pool pattern (out of PR #133 scope).
    let cell_to_group_dev = build_cell_to_group_dev(dev, &all_cells_host, &offsets_host, n_obs)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 alloc cell_to_group: {e}")))?;
    let cell_to_pos_dev = build_cell_to_pos_dev(dev, &all_cells_host, &offsets_host, n_obs)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 alloc cell_to_pos: {e}")))?;

    let n_pool_max = n_ref.max(n_g_max).max(1);
    let mut scratch = scx_gpu::GpuDeChunkScratch::new(dev, n_obs, chunk_size, n_pool_max)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE scratch alloc failed: {e}")))?;
    scratch
        .ensure_ref_slab_capacity(dev, n_ref)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure ref_slab: {e}")))?;
    scratch
        .ensure_group_slab_capacity(dev, n_g_max.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure group_slab: {e}")))?;
    scratch
        .ensure_sums_capacity(dev, n_groups_for_means)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure sums: {e}")))?;
    scratch
        .ensure_aux_capacity(dev, chunk_size * n_pool_max)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure aux: {e}")))?;
    let n_test = test_groups.len();
    scratch
        .ensure_per_group_capacity(dev, n_test.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure per_group: {e}")))?;
    scratch
        .ensure_per_tg_pool_slabs_capacity(dev, n_test, n_g_max.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure per_tg_pool_slabs: {e}")))?;

    let mut target_means: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut ref_means: Vec<f64> = Vec::with_capacity(n_vars);
    let mut log2_fold_changes: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut percent_changes: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut statistics: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut p_values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];

    let pts: std::sync::Arc<scx_gpu::CudaStream> = dev.context().per_thread_stream();
    let dev_pts = dev.with_stream(pts.clone());

    // CSR has no column-range prefilter — every shard is decoded for every
    // gene chunk, so this equals n_csr_shards × n_gene_chunks (recorded for
    // route observability symmetry with the CSC path).
    let mut shards_decoded = 0usize;

    for (chunk_idx, c0) in (0..n_vars).step_by(chunk_size).enumerate() {
        let c1 = (c0 + chunk_size).min(n_vars);
        let sz = c1 - c0;

        // Pre-zero slabs and sums for atomicAdd accumulation.
        {
            let nelem_ref = sz * n_ref;
            let mut ref_view = scratch.ref_slab.slice_mut(..nelem_ref);
            dev.stream()
                .memset_zeros(&mut ref_view)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 memset ref_slab: {e}")))?;
        }
        for tg_idx in 0..n_test {
            let n_g = group_indices[test_groups[tg_idx]].len();
            if n_g == 0 {
                continue;
            }
            let nelem_tg = sz * n_g;
            let mut tg_view = scratch.per_tg_pool_slabs[tg_idx].slice_mut(..nelem_tg);
            dev.stream()
                .memset_zeros(&mut tg_view)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 memset tg slab: {e}")))?;
        }
        {
            let nelem_sums = n_groups_for_means * sz;
            let mut sums_view = scratch.sums.slice_mut(..nelem_sums);
            dev.stream()
                .memset_zeros(&mut sums_view)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 memset sums: {e}")))?;
        }

        // Single CSR shard pass per chunk: scatter to ref + per-tg slabs +
        // accumulate pseudobulk sums via the v3 CSR-direct kernel.
        let mut global_row = 0usize;
        source
            .for_each_gpu_csr_shard(&mut |_idx, slot| {
                shards_decoded += 1;
                let view = slot.view();
                let n_rows = view.shape.0;
                // group_id = 0 is the reference; 1..=n_test are the test groups
                // (matches offsets_host layout: ref cells first, then each tg).
                gpu_de_scatter_csr_to_gene_major_filtered(
                    dev,
                    &view,
                    &cell_to_group_dev,
                    &cell_to_pos_dev,
                    0,
                    &mut scratch.ref_slab,
                    global_row,
                    n_ref,
                    sz,
                    c0,
                    c1,
                )?;
                for tg_idx in 0..n_test {
                    let n_g = group_indices[test_groups[tg_idx]].len();
                    if n_g == 0 {
                        continue;
                    }
                    gpu_de_scatter_csr_to_gene_major_filtered(
                        dev,
                        &view,
                        &cell_to_group_dev,
                        &cell_to_pos_dev,
                        (tg_idx + 1) as i32,
                        &mut scratch.per_tg_pool_slabs[tg_idx],
                        global_row,
                        n_g,
                        sz,
                        c0,
                        c1,
                    )?;
                }
                gpu_de_pseudobulk_csr_direct(
                    dev,
                    &view,
                    &cell_to_group_dev,
                    &mut scratch.sums,
                    global_row,
                    sz,
                    c0,
                    c1,
                    mode_id,
                )?;
                global_row += n_rows;
                Ok(())
            })
            .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 CSR shard pass: {e}")))?;

        let (chunk_ref_means, chunk_target_means) =
            compute_pdex_means_from_sums(dev, &scratch.sums, sz, n_ref, &target_memberships, mode)?;

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

        // Per-chunk DE sequence (sort + tie + searchsort + pvalues) is
        // unchanged from v2 — reads from per-tg slabs (already populated by
        // the shard loop). Mode 4 reserved for v3 CSR if graph capture is
        // later re-enabled; for now skip capture (shard loop count is
        // dynamic so the per-chunk kernel count diverges from v2).
        let chunk_max = scratch.chunk_max();
        let target_dev = if cuda_graphs_enabled() { &dev_pts } else { dev };
        let _ = chunk_idx;
        pdex_ref_chunk_gpu_sequence_v2(
            target_dev,
            &mut scratch,
            &test_groups,
            &group_indices,
            sz,
            n_ref,
            chunk_max,
        )?;

        ref_means.extend_from_slice(&chunk_ref_means);

        let u_batch_len = n_test * chunk_max;
        let p_batch_len = n_test * chunk_max;
        let u_batch = if n_test == 0 {
            Vec::new()
        } else {
            let view = scratch
                .u_per_group
                .try_slice(..u_batch_len)
                .ok_or_else(|| AccelError::LinAlg("u_per_group batch slice OOB (v3csr)".into()))?;
            dev.stream()
                .clone_dtoh(&view)
                .map_err(|e| AccelError::LinAlg(format!("dtoh U batch (v3csr): {e}")))?
        };
        let p_batch = if n_test == 0 {
            Vec::new()
        } else {
            let view = scratch
                .p_per_group
                .try_slice(..p_batch_len)
                .ok_or_else(|| AccelError::LinAlg("p_per_group batch slice OOB (v3csr)".into()))?;
            dev.stream()
                .clone_dtoh(&view)
                .map_err(|e| AccelError::LinAlg(format!("dtoh p batch (v3csr): {e}")))?
        };

        for (tg_idx, &g) in test_groups.iter().enumerate() {
            target_means[tg_idx].extend_from_slice(&chunk_target_means[tg_idx]);
            log2_fold_changes[tg_idx].extend_from_slice(&chunk_log2_fc[tg_idx]);
            percent_changes[tg_idx].extend_from_slice(&chunk_percent[tg_idx]);
            let n_g = group_indices[g].len();
            if n_g == 0 {
                statistics[tg_idx].extend(std::iter::repeat_n(f64::NAN, sz));
                p_values[tg_idx].extend(std::iter::repeat_n(1.0, sz));
            } else {
                let off = tg_idx * chunk_max;
                statistics[tg_idx].extend_from_slice(&u_batch[off..off + sz]);
                p_values[tg_idx].extend(p_batch[off..off + sz].iter().map(|&p| p.clamp(0.0, 1.0)));
            }
        }
    }

    dev.synchronize()
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 CSR synchronize: {e}")))?;

    let fdrs: Vec<Vec<f64>> = p_values
        .iter()
        .map(|pv| {
            let clipped: Vec<f64> = pv.iter().map(|&p| p.clamp(0.0, 1.0)).collect();
            benjamini_hochberg(&clipped)
        })
        .collect();

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
        exec_info: crate::route::AccelExecutionInfo {
            shards_decoded: Some(shards_decoded),
            shards_uploaded: Some(shards_decoded),
            ..Default::default()
        },
    })
}

/// G4 v3 CSC-direct driver for `pdex_ref` (primary v3 path when a CSC
/// sidecar is available). Same shape as the CSR fallback above but the
/// per-chunk shard pass walks gene columns (CSC) instead of cell rows
/// (CSR): per CSC shard overlapping `[c0, c1)`, one launch per pool for
/// the gene-major scatter (K+1 total) + one pseudobulk launch that
/// tree-reduces per-(gene, group) sums with **no atomicAdd**.
///
/// This is the perf-winning v3 path — the CSC tree-reduce matches the
/// existing `pseudobulk_all_groups_kernel` reduction shape exactly, so the
/// only thing v3 changes from v2 is dropping the dense intermediate (and
/// the K+1 dense→slab scatter kernels from v1, already dropped in v2).
#[allow(clippy::too_many_arguments)]
fn pdex_ref_gpu_chunked_v3_csc(
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
    source: &mut dyn GpuMatrixSource,
) -> Result<PdexRefResult> {
    if chunk_size == 0 {
        return Err(AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }

    // One-shot dispatch trace for bench validation. See the matching
    // trace in `pdex_ref_gpu_chunked_v3_csr`.
    if std::env::var("SCX_GPU_DE_V3_TRACE").is_ok() {
        eprintln!("[scx-accel/pdex_ref] v3 dispatch route: csc-direct (CSC sidecar present)");
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
    let test_groups: Vec<usize> = (0..n_groups).filter(|&g| g != reference).collect();
    let target_memberships: Vec<usize> = test_groups
        .iter()
        .map(|&g| group_indices[g].len())
        .collect();
    let n_g_max = target_memberships.iter().copied().max().unwrap_or(0);

    let ref_idx_i32: Vec<i32> = ref_cells.iter().map(|&c| c as i32).collect();
    let group_idx_i32: Vec<Vec<i32>> = test_groups
        .iter()
        .map(|&g| group_indices[g].iter().map(|&c| c as i32).collect())
        .collect();

    let n_groups_for_means = 1 + test_groups.len();
    let mut all_cells_host: Vec<i32> = Vec::with_capacity(n_ref + n_g_max * test_groups.len());
    let mut offsets_host: Vec<i32> = Vec::with_capacity(n_groups_for_means + 1);
    offsets_host.push(0);
    all_cells_host.extend(ref_idx_i32.iter().copied());
    offsets_host.push(checked_offset_i32(
        all_cells_host.len(),
        "GPU DE cell-permutation offsets",
    )?);
    for cells in &group_idx_i32 {
        all_cells_host.extend(cells.iter().copied());
        offsets_host.push(checked_offset_i32(
            all_cells_host.len(),
            "GPU DE cell-permutation offsets",
        )?);
    }
    let mode_id = geom_mean_mode_id(mode);

    // Code-review #6 (same as v3-CSR): collapse K+1 per-pool cell_to_pool
    // tables to 2 constant-in-K tables. See pdex_ref_gpu_chunked_v3_csr for
    // the rationale.
    let cell_to_group_dev = build_cell_to_group_dev(dev, &all_cells_host, &offsets_host, n_obs)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 alloc cell_to_group: {e}")))?;
    let cell_to_pos_dev = build_cell_to_pos_dev(dev, &all_cells_host, &offsets_host, n_obs)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 alloc cell_to_pos: {e}")))?;

    let n_pool_max = n_ref.max(n_g_max).max(1);
    let mut scratch = scx_gpu::GpuDeChunkScratch::new(dev, n_obs, chunk_size, n_pool_max)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE scratch alloc failed: {e}")))?;
    scratch
        .ensure_ref_slab_capacity(dev, n_ref)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure ref_slab: {e}")))?;
    scratch
        .ensure_group_slab_capacity(dev, n_g_max.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure group_slab: {e}")))?;
    scratch
        .ensure_sums_capacity(dev, n_groups_for_means)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure sums: {e}")))?;
    scratch
        .ensure_aux_capacity(dev, chunk_size * n_pool_max)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure aux: {e}")))?;
    let n_test = test_groups.len();
    scratch
        .ensure_per_group_capacity(dev, n_test.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure per_group: {e}")))?;
    scratch
        .ensure_per_tg_pool_slabs_capacity(dev, n_test, n_g_max.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 ensure per_tg_pool_slabs: {e}")))?;

    let mut target_means: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut ref_means: Vec<f64> = Vec::with_capacity(n_vars);
    let mut log2_fold_changes: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut percent_changes: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut statistics: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut p_values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];

    let pts: std::sync::Arc<scx_gpu::CudaStream> = dev.context().per_thread_stream();
    let dev_pts = dev.with_stream(pts.clone());

    // Count CSC shards actually decoded+uploaded across the whole call. With
    // range prefiltering this is < n_csc_shards × n_gene_chunks, proving
    // non-overlapping shards were skipped (§B.10 criterion 4).
    let mut shards_decoded = 0usize;

    for (chunk_idx, c0) in (0..n_vars).step_by(chunk_size).enumerate() {
        let c1 = (c0 + chunk_size).min(n_vars);
        let sz = c1 - c0;

        // Pre-zero ref + per-tg slabs + sums.
        {
            let nelem_ref = sz * n_ref;
            let mut ref_view = scratch.ref_slab.slice_mut(..nelem_ref);
            dev.stream()
                .memset_zeros(&mut ref_view)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 memset ref_slab: {e}")))?;
        }
        for tg_idx in 0..n_test {
            let n_g = group_indices[test_groups[tg_idx]].len();
            if n_g == 0 {
                continue;
            }
            let nelem_tg = sz * n_g;
            let mut tg_view = scratch.per_tg_pool_slabs[tg_idx].slice_mut(..nelem_tg);
            dev.stream()
                .memset_zeros(&mut tg_view)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 memset tg slab: {e}")))?;
        }
        {
            let nelem_sums = n_groups_for_means * sz;
            let mut sums_view = scratch.sums.slice_mut(..nelem_sums);
            dev.stream()
                .memset_zeros(&mut sums_view)
                .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 memset sums: {e}")))?;
        }

        // Single CSC shard pass per chunk: each shard that overlaps
        // [c0, c1) launches K+1 scatter-to-gene-major kernels + 1
        // pseudobulk kernel. `for_each_gpu_csc_shard_in_range` pre-
        // filters non-overlapping shards via cheap catalog lookup, so
        // they're never decoded or uploaded.
        source
            .for_each_gpu_csc_shard_in_range(c0 as u32..c1 as u32, &mut |_idx, csc_view| {
                shards_decoded += 1;
                // group_id = 0 is the reference; 1..=n_test are tg slabs.
                gpu_de_scatter_csc_to_gene_major(
                    dev,
                    csc_view,
                    &cell_to_group_dev,
                    &cell_to_pos_dev,
                    0,
                    &mut scratch.ref_slab,
                    c0,
                    c1,
                    sz,
                    n_ref,
                )?;
                for tg_idx in 0..n_test {
                    let n_g = group_indices[test_groups[tg_idx]].len();
                    if n_g == 0 {
                        continue;
                    }
                    gpu_de_scatter_csc_to_gene_major(
                        dev,
                        csc_view,
                        &cell_to_group_dev,
                        &cell_to_pos_dev,
                        (tg_idx + 1) as i32,
                        &mut scratch.per_tg_pool_slabs[tg_idx],
                        c0,
                        c1,
                        sz,
                        n_g,
                    )?;
                }
                gpu_de_pseudobulk_csc_direct(
                    dev,
                    csc_view,
                    &cell_to_group_dev,
                    &mut scratch.sums,
                    c0,
                    c1,
                    sz,
                    n_groups_for_means,
                    mode_id,
                )?;
                Ok(())
            })
            .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 CSC shard pass: {e}")))?;

        let (chunk_ref_means, chunk_target_means) =
            compute_pdex_means_from_sums(dev, &scratch.sums, sz, n_ref, &target_memberships, mode)?;

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

        let chunk_max = scratch.chunk_max();
        let target_dev = if cuda_graphs_enabled() { &dev_pts } else { dev };
        let _ = chunk_idx;
        pdex_ref_chunk_gpu_sequence_v2(
            target_dev,
            &mut scratch,
            &test_groups,
            &group_indices,
            sz,
            n_ref,
            chunk_max,
        )?;

        ref_means.extend_from_slice(&chunk_ref_means);

        let u_batch_len = n_test * chunk_max;
        let p_batch_len = n_test * chunk_max;
        let u_batch = if n_test == 0 {
            Vec::new()
        } else {
            let view = scratch
                .u_per_group
                .try_slice(..u_batch_len)
                .ok_or_else(|| AccelError::LinAlg("u_per_group batch slice OOB (v3csc)".into()))?;
            dev.stream()
                .clone_dtoh(&view)
                .map_err(|e| AccelError::LinAlg(format!("dtoh U batch (v3csc): {e}")))?
        };
        let p_batch = if n_test == 0 {
            Vec::new()
        } else {
            let view = scratch
                .p_per_group
                .try_slice(..p_batch_len)
                .ok_or_else(|| AccelError::LinAlg("p_per_group batch slice OOB (v3csc)".into()))?;
            dev.stream()
                .clone_dtoh(&view)
                .map_err(|e| AccelError::LinAlg(format!("dtoh p batch (v3csc): {e}")))?
        };

        for (tg_idx, &g) in test_groups.iter().enumerate() {
            target_means[tg_idx].extend_from_slice(&chunk_target_means[tg_idx]);
            log2_fold_changes[tg_idx].extend_from_slice(&chunk_log2_fc[tg_idx]);
            percent_changes[tg_idx].extend_from_slice(&chunk_percent[tg_idx]);
            let n_g = group_indices[g].len();
            if n_g == 0 {
                statistics[tg_idx].extend(std::iter::repeat_n(f64::NAN, sz));
                p_values[tg_idx].extend(std::iter::repeat_n(1.0, sz));
            } else {
                let off = tg_idx * chunk_max;
                statistics[tg_idx].extend_from_slice(&u_batch[off..off + sz]);
                p_values[tg_idx].extend(p_batch[off..off + sz].iter().map(|&p| p.clamp(0.0, 1.0)));
            }
        }
    }

    dev.synchronize()
        .map_err(|e| AccelError::LinAlg(format!("GPU DE v3 CSC synchronize: {e}")))?;

    let fdrs: Vec<Vec<f64>> = p_values
        .iter()
        .map(|pv| {
            let clipped: Vec<f64> = pv.iter().map(|&p| p.clamp(0.0, 1.0)).collect();
            benjamini_hochberg(&clipped)
        })
        .collect();

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
        exec_info: crate::route::AccelExecutionInfo {
            shards_decoded: Some(shards_decoded),
            shards_uploaded: Some(shards_decoded),
            ..Default::default()
        },
    })
}

// ---------------------------------------------------------------------------
// Wilcoxon v3 (CSC-direct / CSR-direct) chunk drivers
// ---------------------------------------------------------------------------

/// Pre-built device tables + group bookkeeping shared by the v3 Wilcoxon
/// drivers. Built once per DE call and reused across the chunk loop.
///
/// Two distinct cell→group/pos table sets are needed because the Wilcoxon
/// comparison pool differs from `pdex_ref` (§C.5):
/// - **Main tables** (`cell_to_group_dev` / `cell_to_pos_dev`): slot 0 is the
///   reference group (ref-mode) or empty (1-vs-rest); slots `1..=n_test` are
///   the test groups. Used for the per-test-group scatters (`group_id =
///   tg_idx + 1`) and for the pseudobulk (which spans all `n_slots` slots,
///   covering every cell). `slot_to_group[slot]` maps a slot back to its
///   original group id (for reading per-group sums back).
/// - **Pool tables** (`pool_group_dev` / `pool_pos_dev`): map every pool cell
///   to `group_id = 0`. In ref-mode the pool is the reference group; in
///   1-vs-rest the pool is *all* cells, which overlaps the test groups and so
///   cannot share the main table. Used only for the `group_id = 0` pool
///   scatter into `ref_slab`.
struct WilcoxonV3Prep {
    cell_to_group_dev: CudaSlice<i32>,
    cell_to_pos_dev: CudaSlice<i32>,
    pool_group_dev: CudaSlice<i32>,
    pool_pos_dev: CudaSlice<i32>,
    /// `slot_to_group[slot] = Some(original_group)` for populated slots; slot 0
    /// is `None` in 1-vs-rest (empty), `Some(reference)` in ref-mode.
    slot_to_group: Vec<Option<usize>>,
    n_slots: usize,
    pool_len: usize,
    is_ref_mode: bool,
    group_indices: Vec<Vec<usize>>,
    test_groups: Vec<usize>,
}

/// Build the [`WilcoxonV3Prep`] tables for a Wilcoxon GPU v3 call.
fn prepare_wilcoxon_v3(
    dev: &GpuDevice,
    n_obs: usize,
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
) -> Result<WilcoxonV3Prep> {
    let n_groups = group_names.len();
    let (group_indices, _oor) = bucket_cells_by_group(groups, n_groups);
    let is_ref_mode = reference.is_some();
    let test_groups: Vec<usize> = match reference {
        Some(r) => (0..n_groups).filter(|&g| g != r).collect(),
        None => (0..n_groups).collect(),
    };

    // Pool permutation: reference cells (ref-mode) or all cells (1-vs-rest).
    let pool_host: Vec<i32> = match reference {
        Some(r) => group_indices[r].iter().map(|&c| c as i32).collect(),
        None => (0..n_obs as i32).collect(),
    };
    let pool_len = pool_host.len();
    if pool_len == 0 {
        return Err(AccelError::InvalidInput(
            "Wilcoxon GPU v3: empty comparison pool (reference group has zero cells)".to_string(),
        ));
    }
    let pool_offsets: Vec<i32> = vec![0, checked_offset_i32(pool_len, "Wilcoxon v3 pool offsets")?];
    let pool_group_dev = build_cell_to_group_dev(dev, &pool_host, &pool_offsets, n_obs)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 alloc pool group: {e}")))?;
    let pool_pos_dev = build_cell_to_pos_dev(dev, &pool_host, &pool_offsets, n_obs)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 alloc pool pos: {e}")))?;

    // Main table: slot 0 = reference (ref-mode) or empty (1-vs-rest); slots
    // 1..=n_test = test groups, in `test_groups` order.
    let mut all_cells_host: Vec<i32> = Vec::new();
    let mut offsets_host: Vec<i32> = Vec::with_capacity(test_groups.len() + 2);
    offsets_host.push(0);
    if let Some(r) = reference {
        all_cells_host.extend(group_indices[r].iter().map(|&c| c as i32));
    }
    offsets_host.push(checked_offset_i32(
        all_cells_host.len(),
        "Wilcoxon v3 main offsets",
    )?);
    for &g in &test_groups {
        all_cells_host.extend(group_indices[g].iter().map(|&c| c as i32));
        offsets_host.push(checked_offset_i32(
            all_cells_host.len(),
            "Wilcoxon v3 main offsets",
        )?);
    }
    let n_slots = test_groups.len() + 1;
    let cell_to_group_dev = build_cell_to_group_dev(dev, &all_cells_host, &offsets_host, n_obs)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 alloc cell_to_group: {e}")))?;
    let cell_to_pos_dev = build_cell_to_pos_dev(dev, &all_cells_host, &offsets_host, n_obs)
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 alloc cell_to_pos: {e}")))?;

    let mut slot_to_group: Vec<Option<usize>> = vec![None; n_slots];
    if let Some(r) = reference {
        slot_to_group[0] = Some(r);
    }
    for (i, &g) in test_groups.iter().enumerate() {
        slot_to_group[i + 1] = Some(g);
    }

    Ok(WilcoxonV3Prep {
        cell_to_group_dev,
        cell_to_pos_dev,
        pool_group_dev,
        pool_pos_dev,
        slot_to_group,
        n_slots,
        pool_len,
        is_ref_mode,
        group_indices,
        test_groups,
    })
}

/// Zero the per-chunk ref/pool slab, each non-empty per-tg slab, and the
/// pseudobulk sums before the shard scatter pass populates them.
fn zero_wilcoxon_v3_slabs(
    dev: &GpuDevice,
    scratch: &mut scx_gpu::GpuDeChunkScratch,
    pool_len: usize,
    test_groups: &[usize],
    group_indices: &[Vec<usize>],
    n_slots: usize,
    sz: usize,
) -> Result<()> {
    {
        let mut v = scratch.ref_slab.slice_mut(..sz * pool_len);
        dev.stream()
            .memset_zeros(&mut v)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 memset ref_slab: {e}")))?;
    }
    for (tg_idx, &g) in test_groups.iter().enumerate() {
        let n_g = group_indices[g].len();
        if n_g == 0 {
            continue;
        }
        let mut v = scratch.per_tg_pool_slabs[tg_idx].slice_mut(..sz * n_g);
        dev.stream()
            .memset_zeros(&mut v)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 memset tg slab: {e}")))?;
    }
    {
        let mut v = scratch.sums.slice_mut(..n_slots * sz);
        dev.stream()
            .memset_zeros(&mut v)
            .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 memset sums: {e}")))?;
    }
    Ok(())
}

/// Core Wilcoxon v3 driver shared by the CSC-direct and CSR-direct paths. The
/// `populate_slabs` closure is responsible for zeroing + populating
/// `scratch.ref_slab` (the pool), each `scratch.per_tg_pool_slabs[tg_idx]`,
/// and `scratch.sums` (per-slot raw pseudobulk sums) for the chunk
/// `[c0, c1)` — it writes the gene-major slabs directly (no dense
/// intermediate). The per-chunk statistical sequence + host post-pass
/// (z / p / logFC) matches the CPU Wilcoxon reference; only slab population
/// differs between the CSC-direct and CSR-direct callers.
#[allow(clippy::too_many_arguments)]
fn wilcoxon_rank_sum_gpu_chunked_v3<P>(
    dev: &GpuDevice,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
    gene_names: &[String],
    group_names: &[String],
    group_indices: &[Vec<usize>],
    test_groups: &[usize],
    reference: Option<usize>,
    slot_to_group: &[Option<usize>],
    n_slots: usize,
    pool_len: usize,
    is_ref_mode: bool,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
    mut populate_slabs: P,
) -> Result<DiffExpResult>
where
    // Returns the number of shards decoded for the chunk (for the
    // shards_decoded route signal); the core accumulates across chunks.
    P: FnMut(&mut scx_gpu::GpuDeChunkScratch, usize, usize, usize) -> Result<usize>,
{
    if chunk_size == 0 {
        return Err(AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }

    let n_groups = group_names.len();
    let n_test = test_groups.len();
    let n_g_max = test_groups
        .iter()
        .map(|&g| group_indices[g].len())
        .max()
        .unwrap_or(0);

    let mut scratch = scx_gpu::GpuDeChunkScratch::new(dev, n_obs, chunk_size, pool_len.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE scratch alloc failed: {e}")))?;
    scratch
        .ensure_ref_slab_capacity(dev, pool_len.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 ensure ref_slab: {e}")))?;
    scratch
        .ensure_sums_capacity(dev, n_slots.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 ensure sums: {e}")))?;
    scratch
        .ensure_aux_capacity(dev, chunk_size * pool_len.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 ensure aux: {e}")))?;
    scratch
        .ensure_per_group_capacity(dev, n_test.max(1))
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 ensure per_group: {e}")))?;
    scratch
        .ensure_per_tg_pool_slabs_capacity(dev, n_test, n_g_max.max(1))
        .map_err(|e| {
            AccelError::LinAlg(format!("GPU DE Wilcoxon v3 ensure per_tg_pool_slabs: {e}"))
        })?;

    let mut chunk_results: Vec<DiffExpResult> = Vec::new();
    let mut shards_decoded = 0usize;
    let pts: std::sync::Arc<scx_gpu::CudaStream> = dev.context().per_thread_stream();
    let dev_pts = dev.with_stream(pts.clone());

    for c0 in (0..n_vars).step_by(chunk_size) {
        let c1 = (c0 + chunk_size).min(n_vars);
        let sz = c1 - c0;

        // Populate ref/pool + per-tg slabs + per-slot pseudobulk sums.
        shards_decoded += populate_slabs(&mut scratch, c0, c1, sz)?;

        // Read per-(slot, gene) raw sums → group_gene_sums[original_group][var].
        let sums_host: Vec<f64> = {
            let view = scratch
                .sums
                .try_slice(..n_slots * sz)
                .ok_or_else(|| AccelError::LinAlg("sums slice OOB (wil v3)".into()))?;
            dev.stream()
                .clone_dtoh(&view)
                .map_err(|e| AccelError::LinAlg(format!("dtoh sums (wil v3): {e}")))?
        };
        let mut group_gene_sums: Vec<Vec<f64>> = vec![vec![0.0f64; sz]; n_groups];
        for (slot, maybe_g) in slot_to_group.iter().enumerate().take(n_slots) {
            if let Some(g) = *maybe_g {
                group_gene_sums[g].copy_from_slice(&sums_host[slot * sz..slot * sz + sz]);
            }
        }

        // 1-vs-rest reference sum in O(1): precompute the chunk-local total gene
        // sum across all groups once, then derive each group's rest_sum via
        // subtraction (mirrors `diffexp.rs::wilcoxon_rank_sum`). Avoids the
        // O(n_groups) re-scan per (gene, group) — i.e. O(sz·n_groups²) per chunk.
        // Only needed for the 1-vs-rest arm.
        let total_gene_sum: Vec<f64> = if reference.is_none() {
            let mut totals = vec![0.0f64; sz];
            for sums in &group_gene_sums {
                for (total, &s) in totals.iter_mut().zip(sums.iter()) {
                    *total += s;
                }
            }
            totals
        } else {
            Vec::new()
        };

        let chunk_max = scratch.chunk_max();
        let target_dev = if cuda_graphs_enabled() { &dev_pts } else { dev };
        wilcoxon_chunk_gpu_sequence_v3(
            target_dev,
            &mut scratch,
            test_groups,
            group_indices,
            sz,
            pool_len,
            chunk_max,
            is_ref_mode,
        )?;

        // dtoh of the pool tie (1-vs-rest only; ref-mode overwrites tie_term
        // per tg with the combined tie and stages it into tie_per_group).
        let pool_tie_host: Vec<f64> = if !is_ref_mode {
            let view = scratch
                .tie_term
                .try_slice(..sz)
                .ok_or_else(|| AccelError::LinAlg("tie_term pool slice OOB (wil v3)".into()))?;
            dev.stream()
                .clone_dtoh(&view)
                .map_err(|e| AccelError::LinAlg(format!("dtoh pool tie (wil v3): {e}")))?
        } else {
            Vec::new()
        };

        let u_batch_len = n_test * chunk_max;
        let u_batch = if n_test == 0 {
            Vec::new()
        } else {
            let view = scratch
                .u_per_group
                .try_slice(..u_batch_len)
                .ok_or_else(|| AccelError::LinAlg("u_per_group batch slice OOB (wil v3)".into()))?;
            dev.stream()
                .clone_dtoh(&view)
                .map_err(|e| AccelError::LinAlg(format!("dtoh U batch (wil v3): {e}")))?
        };
        let tie_batch: Vec<f64> = if is_ref_mode && n_test > 0 {
            let view = scratch
                .tie_per_group
                .try_slice(..u_batch_len)
                .ok_or_else(|| {
                    AccelError::LinAlg("tie_per_group batch slice OOB (wil v3)".into())
                })?;
            dev.stream()
                .clone_dtoh(&view)
                .map_err(|e| AccelError::LinAlg(format!("dtoh tie batch (wil v3): {e}")))?
        } else {
            Vec::new()
        };

        // Host post-pass: per-tg slice from batched dtoh, compute z + p, logFC.
        type ChunkGroupRow = (String, Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>);
        let mut chunk_per_group: Vec<ChunkGroupRow> = Vec::with_capacity(n_test);
        for (tg_idx, &g) in test_groups.iter().enumerate() {
            let n_g = group_indices[g].len();
            let group_name = group_names[g].clone();

            if n_g == 0 {
                let names = gene_names[c0..c1].to_vec();
                chunk_per_group.push((
                    group_name,
                    names,
                    vec![f64::NAN; sz],
                    vec![1.0f64; sz],
                    vec![f64::NAN; sz],
                ));
                continue;
            }

            let off = tg_idx * chunk_max;
            let (u_host, tie_host_for_p, n1, n2) = if is_ref_mode {
                let u_host: Vec<f64> = u_batch[off..off + sz].to_vec();
                let combined_tie: Vec<f64> = tie_batch[off..off + sz].to_vec();
                (u_host, combined_tie, n_g, pool_len)
            } else {
                let rank_host: &[f64] = &u_batch[off..off + sz];
                let n1d = n_g as f64;
                let u_host: Vec<f64> = rank_host
                    .iter()
                    .map(|&r| r - n1d * (n1d + 1.0) / 2.0)
                    .collect();
                (u_host, pool_tie_host.clone(), n_g, n_obs - n_g)
            };

            let (scores, pvals) =
                compute_scores_and_pvals(&u_host, &tie_host_for_p, n1, n2, tie_correct);

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
                        let rest_sum = total_gene_sum[var] - group_gene_sums[g][var];
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

        let chunk_de = assemble_chunk_diffexp_result(chunk_per_group, rankby_abs, c0)
            .ok_or_else(|| AccelError::InvalidInput("empty test_groups in chunk".into()))?;
        chunk_results.push(chunk_de);
    }

    dev.synchronize()
        .map_err(|e| AccelError::LinAlg(format!("GPU DE Wilcoxon v3 synchronize: {e}")))?;

    let mut result = merge_diff_exp_results(chunk_results, rankby_abs)?;
    result.exec_info.shards_decoded = Some(shards_decoded);
    result.exec_info.shards_uploaded = Some(shards_decoded);
    Ok(result)
}

/// CSC-direct Wilcoxon rank-sum GPU driver. Populates gene-major slabs
/// directly from CSC column shards (skipping non-overlapping shards via
/// `for_each_gpu_csc_shard_in_range`), bypassing the v1 `[n_obs × chunk]`
/// dense buffer. Mirrors [`pdex_ref_gpu_chunked_v3_csc`].
#[allow(clippy::too_many_arguments)]
fn wilcoxon_rank_sum_gpu_chunked_v3_csc(
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
    source: &mut dyn GpuMatrixSource,
) -> Result<DiffExpResult> {
    if std::env::var("SCX_GPU_DE_V3_TRACE").is_ok() {
        eprintln!("[scx-accel/wilcoxon] v3 dispatch route: csc-direct (CSC sidecar present)");
    }
    let prep = prepare_wilcoxon_v3(dev, n_obs, groups, group_names, reference)?;
    let WilcoxonV3Prep {
        cell_to_group_dev,
        cell_to_pos_dev,
        pool_group_dev,
        pool_pos_dev,
        slot_to_group,
        n_slots,
        pool_len,
        is_ref_mode,
        group_indices,
        test_groups,
    } = prep;
    let n_test = test_groups.len();

    wilcoxon_rank_sum_gpu_chunked_v3(
        dev,
        n_obs,
        n_vars,
        chunk_size,
        gene_names,
        group_names,
        &group_indices,
        &test_groups,
        reference,
        &slot_to_group,
        n_slots,
        pool_len,
        is_ref_mode,
        log_transformed,
        rankby_abs,
        tie_correct,
        |scratch, c0, c1, sz| {
            zero_wilcoxon_v3_slabs(
                dev,
                scratch,
                pool_len,
                &test_groups,
                &group_indices,
                n_slots,
                sz,
            )?;
            let mut shards = 0usize;
            source
                .for_each_gpu_csc_shard_in_range(c0 as u32..c1 as u32, &mut |_idx, csc_view| {
                    shards += 1;
                    // group_id = 0 is the pool (ref cells / all cells); 1..=n_test
                    // are the per-test-group slabs.
                    gpu_de_scatter_csc_to_gene_major(
                        dev,
                        csc_view,
                        &pool_group_dev,
                        &pool_pos_dev,
                        0,
                        &mut scratch.ref_slab,
                        c0,
                        c1,
                        sz,
                        pool_len,
                    )?;
                    for tg_idx in 0..n_test {
                        let n_g = group_indices[test_groups[tg_idx]].len();
                        if n_g == 0 {
                            continue;
                        }
                        gpu_de_scatter_csc_to_gene_major(
                            dev,
                            csc_view,
                            &cell_to_group_dev,
                            &cell_to_pos_dev,
                            (tg_idx + 1) as i32,
                            &mut scratch.per_tg_pool_slabs[tg_idx],
                            c0,
                            c1,
                            sz,
                            n_g,
                        )?;
                    }
                    gpu_de_pseudobulk_csc_direct(
                        dev,
                        csc_view,
                        &cell_to_group_dev,
                        &mut scratch.sums,
                        c0,
                        c1,
                        sz,
                        n_slots,
                        0,
                    )?;
                    Ok(())
                })
                .map_err(|e| {
                    AccelError::LinAlg(format!("GPU DE Wilcoxon v3 CSC shard pass: {e}"))
                })?;
            Ok(shards)
        },
    )
}

/// CSR-direct Wilcoxon rank-sum GPU driver — the CSC-absent fallback (in-memory
/// CSR, lazy sources). Populates gene-major slabs from CSR row shards with a
/// running global row offset. Mirrors [`pdex_ref_gpu_chunked_v3_csr`].
#[allow(clippy::too_many_arguments)]
fn wilcoxon_rank_sum_gpu_chunked_v3_csr(
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
    source: &mut dyn GpuMatrixSource,
) -> Result<DiffExpResult> {
    if std::env::var("SCX_GPU_DE_V3_TRACE").is_ok() {
        eprintln!("[scx-accel/wilcoxon] v3 dispatch route: csr-direct (no CSC sidecar)");
    }
    let prep = prepare_wilcoxon_v3(dev, n_obs, groups, group_names, reference)?;
    let WilcoxonV3Prep {
        cell_to_group_dev,
        cell_to_pos_dev,
        pool_group_dev,
        pool_pos_dev,
        slot_to_group,
        n_slots,
        pool_len,
        is_ref_mode,
        group_indices,
        test_groups,
    } = prep;
    let n_test = test_groups.len();

    wilcoxon_rank_sum_gpu_chunked_v3(
        dev,
        n_obs,
        n_vars,
        chunk_size,
        gene_names,
        group_names,
        &group_indices,
        &test_groups,
        reference,
        &slot_to_group,
        n_slots,
        pool_len,
        is_ref_mode,
        log_transformed,
        rankby_abs,
        tie_correct,
        |scratch, c0, c1, sz| {
            zero_wilcoxon_v3_slabs(
                dev,
                scratch,
                pool_len,
                &test_groups,
                &group_indices,
                n_slots,
                sz,
            )?;
            let mut global_row = 0usize;
            let mut shards = 0usize;
            source
                .for_each_gpu_csr_shard(&mut |_idx, slot| {
                    shards += 1;
                    let view = slot.view();
                    let n_rows = view.shape.0;
                    gpu_de_scatter_csr_to_gene_major_filtered(
                        dev,
                        &view,
                        &pool_group_dev,
                        &pool_pos_dev,
                        0,
                        &mut scratch.ref_slab,
                        global_row,
                        pool_len,
                        sz,
                        c0,
                        c1,
                    )?;
                    for tg_idx in 0..n_test {
                        let n_g = group_indices[test_groups[tg_idx]].len();
                        if n_g == 0 {
                            continue;
                        }
                        gpu_de_scatter_csr_to_gene_major_filtered(
                            dev,
                            &view,
                            &cell_to_group_dev,
                            &cell_to_pos_dev,
                            (tg_idx + 1) as i32,
                            &mut scratch.per_tg_pool_slabs[tg_idx],
                            global_row,
                            n_g,
                            sz,
                            c0,
                            c1,
                        )?;
                    }
                    gpu_de_pseudobulk_csr_direct(
                        dev,
                        &view,
                        &cell_to_group_dev,
                        &mut scratch.sums,
                        global_row,
                        sz,
                        c0,
                        c1,
                        0,
                    )?;
                    global_row += n_rows;
                    Ok(())
                })
                .map_err(|e| {
                    AccelError::LinAlg(format!("GPU DE Wilcoxon v3 CSR shard pass: {e}"))
                })?;
            Ok(shards)
        },
    )
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
    gene_index_base: usize,
) -> Option<DiffExpResult> {
    if chunk_per_group.is_empty() {
        return None;
    }
    let mut group_names = Vec::with_capacity(chunk_per_group.len());
    let mut names = Vec::with_capacity(chunk_per_group.len());
    let mut gene_indices = Vec::with_capacity(chunk_per_group.len());
    let mut scores = Vec::with_capacity(chunk_per_group.len());
    let mut pvals = Vec::with_capacity(chunk_per_group.len());
    let mut pvals_adj = Vec::with_capacity(chunk_per_group.len());
    let mut logfc = Vec::with_capacity(chunk_per_group.len());

    for (gn, mut g_names, g_scores, g_pvals, g_logfc) in chunk_per_group {
        // Per-group sort via the shared comparator (matches wilcoxon_rank_sum):
        // the in-chunk position `i` maps to the global var index
        // `gene_index_base + i`, so ties break on the global index and NaN
        // scores sort last — identical to the CPU paths.
        let n = g_scores.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            crate::diffexp::de_rank_cmp(
                g_scores[a],
                gene_index_base + a,
                g_scores[b],
                gene_index_base + b,
                rankby_abs,
            )
        });
        let sorted_names: Vec<String> = order
            .iter()
            .map(|&i| std::mem::take(&mut g_names[i]))
            .collect();
        let sorted_indices: Vec<usize> = order.iter().map(|&i| gene_index_base + i).collect();
        let sorted_scores: Vec<f64> = order.iter().map(|&i| g_scores[i]).collect();
        let sorted_pvals: Vec<f64> = order.iter().map(|&i| g_pvals[i]).collect();
        let sorted_logfc: Vec<f64> = order.iter().map(|&i| g_logfc[i]).collect();
        let bh = benjamini_hochberg(&sorted_pvals);

        group_names.push(gn);
        names.push(sorted_names);
        gene_indices.push(sorted_indices);
        scores.push(sorted_scores);
        pvals.push(sorted_pvals);
        pvals_adj.push(bh);
        logfc.push(sorted_logfc);
    }

    Some(DiffExpResult {
        group_names,
        names,
        gene_indices,
        scores,
        pvals,
        pvals_adj,
        logfoldchanges: logfc,
        exec_info: crate::route::AccelExecutionInfo::default(),
    })
}

/// GPU DE kernels pass cell indices/offsets to CUDA as `i32` and stage dense
/// buffers as `n_obs × chunk`. Reject dimensions that overflow the 32-bit index
/// space or the `usize` element-count product before any device allocation, so
/// atlas-scale callers get a clear error pointing at the CPU path instead of
/// silent integer wraparound. Returns the checked `n_obs × n_vars` product.
fn validate_gpu_de_dims(n_obs: usize, n_vars: usize) -> Result<usize> {
    if n_obs > i32::MAX as usize {
        return Err(AccelError::InvalidInput(format!(
            "n_obs {n_obs} exceeds the GPU DE limit of i32::MAX ({}); GPU cell \
             indices are 32-bit. Use the CPU differential-expression path for \
             datasets with more than {} cells.",
            i32::MAX,
            i32::MAX
        )));
    }
    n_obs.checked_mul(n_vars).ok_or_else(|| {
        AccelError::InvalidInput(format!(
            "n_obs {n_obs} × n_vars {n_vars} overflows usize; use the CPU \
             differential-expression path for datasets of this size."
        ))
    })
}

/// Checked `usize → i32` for cumulative cell / permutation offsets, which can
/// exceed `n_obs` (a cell may appear in both the reference and a test group).
fn checked_offset_i32(v: usize, ctx: &str) -> Result<i32> {
    i32::try_from(v).map_err(|_| {
        AccelError::InvalidInput(format!(
            "{ctx}: cumulative cell offset {v} exceeds i32::MAX ({})",
            i32::MAX
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_pdex_inputs(
    data_len: Option<usize>,
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
    let expected = validate_gpu_de_dims(n_obs, n_vars)?;
    if let Some(data_len) = data_len {
        if data_len != expected {
            return Err(AccelError::InvalidInput(format!(
                "data length {data_len} != n_obs {n_obs} × n_vars {n_vars}"
            )));
        }
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
    data_len: Option<usize>,
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
    let expected = validate_gpu_de_dims(n_obs, n_vars)?;
    if let Some(data_len) = data_len {
        if data_len != expected {
            return Err(AccelError::InvalidInput(format!(
                "data length {data_len} != n_obs {n_obs} × n_vars {n_vars}"
            )));
        }
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

    // ---- Synthetic dimension-bound guards (pure, no GPU required) ----

    #[test]
    fn test_validate_gpu_de_dims_rejects_oversized_n_obs() {
        // n_obs above i32::MAX cannot be expressed as 32-bit GPU cell indices.
        let too_many = i32::MAX as usize + 1;
        let err = validate_gpu_de_dims(too_many, 10).unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
        // A normal case returns the checked product.
        assert_eq!(validate_gpu_de_dims(1_000, 50).unwrap(), 50_000);
    }

    #[test]
    fn test_validate_gpu_de_dims_rejects_product_overflow() {
        // n_obs within i32 but n_obs × n_vars overflows usize.
        let n_obs = i32::MAX as usize;
        let n_vars = usize::MAX / 2;
        let err = validate_gpu_de_dims(n_obs, n_vars).unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_checked_offset_i32_boundary() {
        assert_eq!(
            checked_offset_i32(i32::MAX as usize, "ctx").unwrap(),
            i32::MAX
        );
        let err = checked_offset_i32(i32::MAX as usize + 1, "ctx").unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
    }

    #[test]
    fn test_validate_pdex_inputs_enforces_dim_guard_without_buffer() {
        // No dense buffer (sparse/streaming) still rejects oversized n_obs.
        let n_obs = i32::MAX as usize + 1;
        let err = validate_pdex_inputs(None, n_obs, 10, 10, n_obs, 3, 0, 1e-6).unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
        // A dense buffer whose length disagrees with n_obs × n_vars still errors.
        let err = validate_pdex_inputs(Some(99), 10, 10, 10, 10, 3, 0, 1e-6).unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
        // Consistent small dims pass.
        validate_pdex_inputs(Some(100), 10, 10, 10, 10, 3, 0, 1e-6).unwrap();
    }

    #[test]
    fn test_validate_wilcoxon_inputs_enforces_dim_guard_without_buffer() {
        let n_obs = i32::MAX as usize + 1;
        let err = validate_wilcoxon_inputs(None, n_obs, 10, 10, n_obs).unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
        let err = validate_wilcoxon_inputs(Some(99), 10, 10, 10, 10).unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
        validate_wilcoxon_inputs(Some(100), 10, 10, 10, 10).unwrap();
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

    /// G10.4 parity: graph-captured per-chunk path produces identical
    /// results to the direct per-chunk path under the same fixture.
    /// Unlike UMAP (where atomicAdd races create irreducible run-to-run
    /// jitter), pdex_ref's kernels are deterministic given fixed input
    /// — sort + searchsorted + tie-correct + pvalues — so the two
    /// paths should agree to fp32 tolerance bit-for-bit on U / p /
    /// log2_fc / means. Any divergence implies the graph-replay path
    /// is feeding stale buffer pointers or missing a kernel.
    ///
    /// Uses `set_cuda_graphs_enabled_override` to flip the kill switch
    /// in-process so both branches run in the same test invocation
    /// (the `SCX_DISABLE_CUDA_GRAPHS=1` env var is `OnceLock`-cached
    /// at process start and can't be re-read).
    #[test]
    fn test_pdex_ref_gpu_graph_vs_direct_parity() {
        let _ = require_gpu_or_skip!();

        let (data, n_obs, n_vars, gene_names, groups, group_names, reference) = make_fixture();
        let mode = crate::pseudobulk::GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;

        let prev = scx_gpu::set_cuda_graphs_enabled_override(Some(false));
        let direct = pdex_ref_gpu_dense(
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
        .expect("direct pdex_ref_gpu_dense failed");

        scx_gpu::set_cuda_graphs_enabled_override(Some(true));
        let graph = pdex_ref_gpu_dense(
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
        .expect("graph pdex_ref_gpu_dense failed");

        scx_gpu::set_cuda_graphs_enabled_override(prev);

        assert_eq!(direct.group_names, graph.group_names);
        assert_eq!(direct.feature_names, graph.feature_names);

        // The graph path should produce bit-for-bit identical outputs
        // to the direct path (same kernels, same inputs, deterministic
        // sort/searchsort/pvalues — no atomics in this DE family).
        // A modest tolerance accommodates kernel-launch reordering
        // between per_thread_stream and NULL stream, but anything
        // beyond fp32 rounding suggests a real bug.
        for tg in 0..direct.group_names.len() {
            for var in 0..n_vars {
                let u_d = direct.statistics[tg][var];
                let u_g = graph.statistics[tg][var];
                if u_d.is_finite() && u_g.is_finite() {
                    assert!(
                        (u_d - u_g).abs() < 1e-6,
                        "U mismatch tg={tg} gene={var}: direct={u_d}, graph={u_g}"
                    );
                }
                let p_d = direct.p_values[tg][var];
                let p_g = graph.p_values[tg][var];
                assert!(
                    (p_d - p_g).abs() < 1e-9 || (p_d - p_g).abs() / p_d.abs().max(1e-12) < 1e-6,
                    "p-value mismatch tg={tg} gene={var}: direct={p_d}, graph={p_g}"
                );
                let tm_d = direct.target_means[tg][var];
                let tm_g = graph.target_means[tg][var];
                assert!(
                    (tm_d - tm_g).abs() < 1e-6 || (tm_d - tm_g).abs() / tm_d.abs().max(1e-9) < 1e-6,
                    "target_mean mismatch tg={tg} gene={var}: direct={tm_d}, graph={tm_g}"
                );
            }
            for var in 0..n_vars {
                let r_d = direct.ref_means[var];
                let r_g = graph.ref_means[var];
                assert!(
                    (r_d - r_g).abs() < 1e-6 || (r_d - r_g).abs() / r_d.abs().max(1e-9) < 1e-6,
                    "ref_mean mismatch gene={var}: direct={r_d}, graph={r_g}"
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
            0,     // gene_index_base
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

    /// G10.5 parity (1-vs-rest): graph-captured wilcoxon path
    /// produces identical results to the direct path. Wilcoxon's
    /// captureable kernels (scatter / block_sort / tie / searchsorted /
    /// ranksum) are deterministic given fixed input — atomicAdd lives
    /// only in `gpu_de_pseudobulk_all_groups`, which runs OUTSIDE the
    /// captured region — so we can assert fp32-tight tolerance on U /
    /// p / score, same shape as
    /// `test_pdex_ref_gpu_graph_vs_direct_parity`.
    #[test]
    fn test_wilcoxon_gpu_one_vs_rest_graph_vs_direct_parity() {
        let _ = require_gpu_or_skip!();

        let (data, n_obs, n_vars, gene_names, groups, group_names, _reference) = make_fixture();

        let prev = scx_gpu::set_cuda_graphs_enabled_override(Some(false));
        let direct = wilcoxon_rank_sum_gpu_dense(
            0,
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
        .expect("direct wilcoxon 1-vs-rest failed");

        scx_gpu::set_cuda_graphs_enabled_override(Some(true));
        let graph = wilcoxon_rank_sum_gpu_dense(
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
        .expect("graph wilcoxon 1-vs-rest failed");

        scx_gpu::set_cuda_graphs_enabled_override(prev);

        assert_eq!(direct.group_names, graph.group_names);

        use std::collections::HashMap;
        let group_to_map = |res: &DiffExpResult, g: usize| -> HashMap<String, (f64, f64)> {
            res.names[g]
                .iter()
                .zip(res.scores[g].iter().zip(res.pvals[g].iter()))
                .map(|(n, (&s, &p))| (n.clone(), (s, p)))
                .collect()
        };

        for g in 0..direct.group_names.len() {
            let d_map = group_to_map(&direct, g);
            let g_map = group_to_map(&graph, g);
            for gene in &gene_names {
                let (s_d, p_d) = d_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
                let (s_g, p_g) = g_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
                if s_d.is_finite() && s_g.is_finite() {
                    assert!(
                        (s_d - s_g).abs() < 1e-6,
                        "score mismatch group={} gene={}: direct={}, graph={}",
                        direct.group_names[g],
                        gene,
                        s_d,
                        s_g
                    );
                }
                assert!(
                    (p_d - p_g).abs() < 1e-9 || (p_d - p_g).abs() / p_d.abs().max(1e-12) < 1e-6,
                    "pval mismatch group={} gene={}: direct={}, graph={}",
                    direct.group_names[g],
                    gene,
                    p_d,
                    p_g
                );
            }
        }
    }

    /// G10.5 parity (ref-mode): graph-captured wilcoxon path matches
    /// direct in ref-mode. Same kernel determinism contract as the
    /// 1-vs-rest test above, but exercises the second `GraphKey`
    /// variant (`mode=1`) and the per-tg combined-tie + tie_per_group
    /// staging path.
    ///
    /// The shared `make_fixture` provides a reference group via its
    /// last return value; passing it as `reference: Some(...)` routes
    /// the GPU driver into the ref-mode capture path.
    #[test]
    fn test_wilcoxon_gpu_ref_mode_graph_vs_direct_parity() {
        let _ = require_gpu_or_skip!();

        let (data, n_obs, n_vars, gene_names, groups, group_names, reference) = make_fixture();
        // `reference` is a `usize` (group index) — make_fixture always
        // supplies one for pdex_ref. For wilcoxon we wrap it in `Some`
        // to drive the ref-mode capture path (mode = 1).

        let prev = scx_gpu::set_cuda_graphs_enabled_override(Some(false));
        let direct = wilcoxon_rank_sum_gpu_dense(
            0,
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            Some(reference),
            false,
            false,
            true,
        )
        .expect("direct wilcoxon ref-mode failed");

        scx_gpu::set_cuda_graphs_enabled_override(Some(true));
        let graph = wilcoxon_rank_sum_gpu_dense(
            0,
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            Some(reference),
            false,
            false,
            true,
        )
        .expect("graph wilcoxon ref-mode failed");

        scx_gpu::set_cuda_graphs_enabled_override(prev);

        assert_eq!(direct.group_names, graph.group_names);

        use std::collections::HashMap;
        let group_to_map = |res: &DiffExpResult, g: usize| -> HashMap<String, (f64, f64)> {
            res.names[g]
                .iter()
                .zip(res.scores[g].iter().zip(res.pvals[g].iter()))
                .map(|(n, (&s, &p))| (n.clone(), (s, p)))
                .collect()
        };

        for g in 0..direct.group_names.len() {
            let d_map = group_to_map(&direct, g);
            let g_map = group_to_map(&graph, g);
            for gene in &gene_names {
                let (s_d, p_d) = d_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
                let (s_g, p_g) = g_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
                if s_d.is_finite() && s_g.is_finite() {
                    assert!(
                        (s_d - s_g).abs() < 1e-6,
                        "ref-mode score mismatch group={} gene={}: direct={}, graph={}",
                        direct.group_names[g],
                        gene,
                        s_d,
                        s_g
                    );
                }
                assert!(
                    (p_d - p_g).abs() < 1e-9 || (p_d - p_g).abs() / p_d.abs().max(1e-12) < 1e-6,
                    "ref-mode pval mismatch group={} gene={}: direct={}, graph={}",
                    direct.group_names[g],
                    gene,
                    p_d,
                    p_g
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

        let sparse_result = pdex_ref_gpu(
            0,
            GpuDeShardInput::Csr(&csr),
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

    /// The lazy arm of `pdex_ref_gpu` / `wilcoxon_rank_sum_gpu`
    /// matches the dense (`gpu_de_upload_chunk`) reference on the same fixture.
    /// We feed the same `ScxCsr` through both paths: dense via
    /// `pdex_ref_gpu_dense` (legacy host upload), and lazy via
    /// `pdex_ref_gpu(GpuDeShardInput::Lazy(&InMemoryCsrShardSource(&csr)))`
    /// (new device-resident
    /// scatter). The two paths must agree bit-for-bit on the U statistic
    /// (integer-valued) and within the documented p-value / FDR
    /// tolerance.
    #[test]
    fn test_gpu_lazy_entry_points_match_dense_reference() {
        let _ = require_gpu_or_skip!();

        let n_obs = 60usize;
        let n_vars = 150usize;
        let groups: Vec<usize> = (0..n_obs).map(|i| i / 30).collect(); // 2 groups of 30
        let group_names = vec!["ref".to_string(), "test".to_string()];
        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g_{i}")).collect();

        // Reproducible sparse fixture (~12% density, integer counts ≤ 5).
        let mut data = vec![0.0f32; n_obs * n_vars];
        let mut state: u64 = 0xDEADBEEFCAFE_F00D;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for cell in 0..n_obs {
            for gene in 0..n_vars {
                if (next() % 9) == 0 {
                    data[cell * n_vars + gene] = (next() % 6) as f32;
                }
            }
        }

        // Build the matching ScxCsr.
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
        let in_mem = scx_gpu::InMemoryCsrShardSource::new(&csr);

        // ----- pdex_ref -----
        let mode = GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;
        let dense_res = pdex_ref_gpu_dense(
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
        .expect("dense pdex_ref failed");
        let lazy_res = pdex_ref_gpu(
            0,
            GpuDeShardInput::Lazy(&in_mem),
            &gene_names,
            &groups,
            &group_names,
            0,
            Some(50), // 3 chunks
            mode,
            epsilon,
        )
        .expect("lazy pdex_ref failed");

        assert_eq!(dense_res.group_names, lazy_res.group_names);
        for tg in 0..dense_res.group_names.len() {
            for var in 0..n_vars {
                let u_d = dense_res.statistics[tg][var];
                let u_l = lazy_res.statistics[tg][var];
                if u_d.is_finite() && u_l.is_finite() {
                    assert!(
                        (u_d - u_l).abs() < 1e-6,
                        "pdex U mismatch tg={tg} gene={var}: dense={u_d}, lazy={u_l}"
                    );
                }
                let p_d = dense_res.p_values[tg][var];
                let p_l = lazy_res.p_values[tg][var];
                assert!(
                    (p_d - p_l).abs() < 1e-9 || (p_d - p_l).abs() / p_d.abs().max(1e-12) < 1e-6,
                    "pdex p mismatch tg={tg} gene={var}: dense={p_d}, lazy={p_l}"
                );
            }
        }

        // ----- wilcoxon_rank_sum (1-vs-rest) -----
        let dense_w = wilcoxon_rank_sum_gpu_dense(
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
        .expect("dense wilcoxon failed");
        let lazy_w = wilcoxon_rank_sum_gpu(
            0,
            GpuDeShardInput::Lazy(&in_mem),
            &gene_names,
            &groups,
            &group_names,
            None,
            Some(50),
            false,
            false,
            true,
        )
        .expect("lazy wilcoxon failed");

        assert_eq!(dense_w.group_names.len(), lazy_w.group_names.len());
        // wilcoxon_rank_sum sorts within each group; merging restores
        // gene-name ordering. Compare per-(group, gene) by name lookup.
        for g in 0..dense_w.group_names.len() {
            let mut d_map: std::collections::HashMap<String, (f64, f64)> =
                std::collections::HashMap::new();
            for (i, name) in dense_w.names[g].iter().enumerate() {
                d_map.insert(name.clone(), (dense_w.scores[g][i], dense_w.pvals[g][i]));
            }
            for (i, name) in lazy_w.names[g].iter().enumerate() {
                let (d_score, d_pval) = d_map[name];
                let l_score = lazy_w.scores[g][i];
                let l_pval = lazy_w.pvals[g][i];
                if d_score.is_finite() && l_score.is_finite() {
                    assert!(
                        (d_score - l_score).abs() < 1e-6,
                        "wilcoxon z mismatch group={} gene={}: dense={}, lazy={}",
                        dense_w.group_names[g],
                        name,
                        d_score,
                        l_score
                    );
                }
                assert!(
                    (d_pval - l_pval).abs() < 1e-9
                        || (d_pval - l_pval).abs() / d_pval.abs().max(1e-12) < 1e-6,
                    "wilcoxon p mismatch group={} gene={}: dense={}, lazy={}",
                    dense_w.group_names[g],
                    name,
                    d_pval,
                    l_pval
                );
            }
        }
    }
}
