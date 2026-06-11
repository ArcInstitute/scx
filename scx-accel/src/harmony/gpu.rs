//! Harmony2 GPU orchestration (behind the `gpu` feature).
//!
//! Drives the `scx-gpu` device kernels through the CPU-side [`HarmonyState`].
//! Declared as a submodule of [`super`] (the CPU module) so it can reach the
//! private clustering state and helpers it shares with the host path.

use super::*;
use rand::seq::SliceRandom;
use scx_gpu::{
    capture_graph, cuda_graphs_enabled, gpu_harmony_block_oe_update,
    gpu_harmony_block_softmax_penalty, gpu_harmony_compute_o_e_full,
    gpu_harmony_correction_grouped, gpu_harmony_distances, gpu_harmony_distances_gemm,
    gpu_harmony_l2_normalize_cols, gpu_harmony_obj_cross, gpu_harmony_obj_kmeans_entropy,
    gpu_harmony_softmax, gpu_harmony_z_sum, CublasHandle, CudaSlice, GpuDevice,
};
use std::sync::Arc;

/// Test-only counter for `capture_graph` calls made from the
/// Harmony k-means sub-iter capture site. Used by the parity test
/// to assert "capture once across all outer iters" — pre-G10
/// behaviour would have been `max_iter` captures, the hoisted
/// behaviour should be exactly 1. Increments live on the capture
/// path only; warm-up and replay don't touch this counter.
#[cfg(test)]
pub(super) static HARMONY_CAPTURE_ATTEMPTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// cuBLAS sgemm has fixed launch overhead; the hand-written kernel
/// is faster at small N. Above this threshold the GEMM path wins.
const GEMM_N_THRESHOLD: usize = 100_000;

/// Helper: convert a `Vec<f64>` slice to f32 (GPU kernels use f32).
fn f64_to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// Auto-route distance computation: GEMM for large N (cuBLAS
/// dispatches optimised tiles), hand-written kernel for small N
/// (avoids GEMM launch overhead) or whenever either of N / K
/// exceeds 2^31 (cuBLAS sgemm dimensions are i32).
#[allow(clippy::too_many_arguments)]
fn dispatch_harmony_distances(
    dev: &GpuDevice,
    handle: &CublasHandle,
    d_y: &CudaSlice<f32>,
    d_z_cos: &CudaSlice<f32>,
    d_dist: &mut CudaSlice<f32>,
    d: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    let gemm_dims_fit = (n as u64) <= i32::MAX as u64 && (k as u64) <= i32::MAX as u64;
    if n >= GEMM_N_THRESHOLD && gemm_dims_fit {
        gpu_harmony_distances_gemm(dev, handle, d_y, d_z_cos, d_dist, d, k, n)
            .map_err(|e| AccelError::LinAlg(format!("GPU distances (gemm): {e}")))
    } else {
        gpu_harmony_distances(dev, d_y, d_z_cos, d_dist, d, k, n)
            .map_err(|e| AccelError::LinAlg(format!("GPU distances (kernel): {e}")))
    }
}

/// Reduce `obj_cell` (length N) and `cross_kgb` (length K·B) to two f64
/// scalars **on-device** (Task 2.6) and return the Harmony objective.
///
/// Previously this did a full `(N + K·B)`-element D→H copy followed by a
/// host f64 sum. The device reduction (`gpu_harmony_reduce_objective`)
/// accumulates in f64 on the GPU and downloads only 2 scalars per iter,
/// removing the per-iteration `(N + K·B) · 4 bytes` PCIe traffic (≈4 MB at
/// N=1 M, K=B=100). f64 accumulation preserves the prior host-f64 result
/// within floating-point reassociation.
fn compute_objective_gpu(
    dev: &GpuDevice,
    d_obj_cell: &CudaSlice<f32>,
    d_cross_kgb: &CudaSlice<f32>,
    n: usize,
) -> Result<f64> {
    let (kmeans_entropy, cross) = scx_gpu::gpu_harmony_reduce_objective(
        dev,
        d_obj_cell,
        d_obj_cell.len(),
        d_cross_kgb,
        d_cross_kgb.len(),
    )
    .map_err(|e| AccelError::LinAlg(format!("GPU objective reduce: {e}")))?;
    let norm = 2000.0 / n as f64;
    Ok((kmeans_entropy + cross) * norm)
}

/// Factored kernel sequence for one k-means sub-iteration.
/// Runs `n_blocks` × (memcpy_dtod + O/E decrement + block softmax +
/// O/E increment), then the two objective-partial kernels.
///
/// `dev` may be either the canonical NULL-stream `GpuDevice` (for
/// the warm-up sub-iter that populates the module cache, or when
/// `SCX_DISABLE_CUDA_GRAPHS=1`) or a `dev.with_stream(per_thread_
/// stream)` clone (when the call site wants to capture or replay
/// these kernels on a capturable stream).
///
/// Does NOT include the `memcpy_htod(order)` (caller must issue
/// that on the same stream BEFORE calling, so the captured graph
/// reads from a freshly-uploaded `d_order`), nor the post-
/// sub-iter sync / dtoh / convergence check (those break stream
/// capture and stay on the host orchestrator).
#[allow(clippy::too_many_arguments)]
fn run_kmeans_subiter_kernels(
    dev: &GpuDevice,
    d_order: &CudaSlice<i32>,
    d_block_cells: &mut CudaSlice<i32>,
    d_r: &mut CudaSlice<f32>,
    d_o: &mut CudaSlice<f32>,
    d_e: &mut CudaSlice<f32>,
    d_dist: &CudaSlice<f32>,
    d_sigma: &CudaSlice<f32>,
    d_theta: &CudaSlice<f32>,
    d_labels: &CudaSlice<i32>,
    d_cov_offset: &CudaSlice<i32>,
    d_pr_b: &CudaSlice<f32>,
    d_obj_cell: &mut CudaSlice<f32>,
    d_cross_kgb: &mut CudaSlice<f32>,
    n: usize,
    k: usize,
    b: usize,
    c_count: usize,
    n_blocks: usize,
    block_len: usize,
) -> Result<()> {
    for blk in 0..n_blocks {
        let start = blk * block_len;
        if start >= n {
            break;
        }
        let end = (start + block_len).min(n);
        let n_block_cells = end - start;

        let order_view = d_order
            .try_slice(start..end)
            .ok_or_else(|| AccelError::LinAlg("d_order slice out of bounds".into()))?;
        let mut block_view = d_block_cells
            .try_slice_mut(0..n_block_cells)
            .ok_or_else(|| AccelError::LinAlg("d_block_cells slice out of bounds".into()))?;
        dev.stream()
            .memcpy_dtod(&order_view, &mut block_view)
            .map_err(|e| AccelError::LinAlg(format!("memcpy block_cells: {e}")))?;

        gpu_harmony_block_oe_update(
            dev,
            d_r,
            d_block_cells,
            d_labels,
            d_cov_offset,
            d_pr_b,
            d_o,
            d_e,
            -1.0,
            c_count,
            k,
            n,
            b,
            n_block_cells,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU O/E decrement: {e}")))?;

        gpu_harmony_block_softmax_penalty(
            dev,
            d_dist,
            d_sigma,
            d_o,
            d_e,
            d_theta,
            d_labels,
            d_cov_offset,
            d_block_cells,
            d_r,
            c_count,
            k,
            n,
            b,
            n_block_cells,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU block softmax: {e}")))?;

        gpu_harmony_block_oe_update(
            dev,
            d_r,
            d_block_cells,
            d_labels,
            d_cov_offset,
            d_pr_b,
            d_o,
            d_e,
            1.0,
            c_count,
            k,
            n,
            b,
            n_block_cells,
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU O/E increment: {e}")))?;
    }

    gpu_harmony_obj_kmeans_entropy(dev, d_r, d_dist, d_sigma, d_obj_cell, k, n)
        .map_err(|e| AccelError::LinAlg(format!("GPU obj k+e: {e}")))?;
    gpu_harmony_obj_cross(dev, d_o, d_e, d_sigma, d_theta, d_cross_kgb, k, b)
        .map_err(|e| AccelError::LinAlg(format!("GPU obj cross: {e}")))?;
    Ok(())
}

/// GPU-accelerated Harmony2 integration.
///
/// End-to-end GPU orchestration: distance, plain softmax (cold
/// start), block softmax+penalty (k-means inner sub-loop), atomic
/// O/E updates, objective reduction, regression z-sum, and
/// scatter-subtract correction all run on the device. K-means++
/// seeding (HarmonyState::new) and the small `(B'+1) x (B'+1)`
/// regression solve stay on CPU.
///
/// Output is bit-compatible in shape with the CPU path
/// (`HarmonyResult`), but values differ from the CPU reference by
/// f32 rounding plus atomic-ordering nondeterminism in the O/E
/// updates. The validation gate is per-PC Pearson r ≥ 0.95
/// (`test_gpu_vs_cpu_per_pc_correlation`); the CPU
/// `test_determinism_same_seed` bit-exact contract applies only
/// to the CPU path.
pub fn harmony_integrate_gpu(
    device_id: usize,
    embeddings: &[f32],
    n_obs: usize,
    n_pcs: usize,
    covariates: &[BatchCovariate],
    config: &HarmonyConfig,
) -> Result<HarmonyResult> {
    // Initialise CPU state (validates inputs, runs kmeans++/Lloyd, sets up
    // initial R/O/E). Reusing this keeps the two paths algorithmically in
    // step for the first iteration.
    let mut state = HarmonyState::new(embeddings, n_obs, n_pcs, covariates, config)?;
    let n = state.n;
    let d = state.d;
    let k = state.k;
    let b = state.layout.b;
    let c_count = state.layout.c;

    let dev = GpuDevice::new(device_id)
        .map_err(|e| AccelError::LinAlg(format!("GPU init failed: {e}")))?;
    let cublas = CublasHandle::new()
        .map_err(|e| AccelError::LinAlg(format!("cuBLAS handle init failed: {e}")))?;

    // Upload Z_orig once — it never changes.
    let d_z_orig = dev
        .htod_copy(&f64_to_f32(&state.z_orig))
        .map_err(|e| AccelError::LinAlg(format!("upload Z_orig: {e}")))?;

    // Persistent device buffers (reused across iterations). All
    // are kept resident across the iter loop and across the inner
    // k-means sub-loop. Only the final Z_corr download (after the
    // loop) round-trips to host.
    let mut d_z_corr = dev
        .htod_copy(&f64_to_f32(&state.z_orig))
        .map_err(|e| AccelError::LinAlg(format!("alloc Z_corr: {e}")))?;
    let mut d_z_cos = dev
        .alloc_zeros::<f32>(d * n)
        .map_err(|e| AccelError::LinAlg(format!("alloc Z_cos: {e}")))?;
    let mut d_dist = dev
        .alloc_zeros::<f32>(k * n)
        .map_err(|e| AccelError::LinAlg(format!("alloc dist: {e}")))?;
    let mut d_y = dev
        .alloc_zeros::<f32>(d * k)
        .map_err(|e| AccelError::LinAlg(format!("alloc Y: {e}")))?;

    // R / O / E persist on GPU across the entire run; CPU mirror
    // is kept only for the small (B+1)×(B+1) regression solve in
    // the correction step (O/E downloaded once per outer iter).
    let mut d_r = dev
        .htod_copy(&state.r)
        .map_err(|e| AccelError::LinAlg(format!("upload R: {e}")))?;
    let mut d_o = dev
        .htod_copy(&f64_to_f32(&state.o))
        .map_err(|e| AccelError::LinAlg(format!("upload O: {e}")))?;
    let mut d_e = dev
        .htod_copy(&f64_to_f32(&state.e))
        .map_err(|e| AccelError::LinAlg(format!("upload E: {e}")))?;
    // Upload the cold-start dist matrix from CPU `HarmonyState::new`
    // so iter 0's GPU update_r has the same dist values that CPU
    // `update_r` would read from `state.dist_mat`. After iter 0,
    // d_dist is overwritten by GPU distance at the iter > 0
    // cold-start branch.
    dev.stream()
        .memcpy_htod(&state.dist_mat, &mut d_dist)
        .map_err(|e| AccelError::LinAlg(format!("upload dist (iter 0 cold-start): {e}")))?;

    // Read-only constants — uploaded once.
    let d_sigma = dev
        .htod_copy(&f64_to_f32(&state.sigma))
        .map_err(|e| AccelError::LinAlg(format!("upload sigma: {e}")))?;
    let d_theta = dev
        .htod_copy(&f64_to_f32(&state.theta))
        .map_err(|e| AccelError::LinAlg(format!("upload theta: {e}")))?;
    let d_pr_b = dev
        .htod_copy(&f64_to_f32(&state.pr_b))
        .map_err(|e| AccelError::LinAlg(format!("upload pr_b: {e}")))?;
    let cov_offset_i32: Vec<i32> = state.layout.cov_offset.iter().map(|&v| v as i32).collect();
    let d_cov_offset = dev
        .htod_copy(&cov_offset_i32)
        .map_err(|e| AccelError::LinAlg(format!("upload cov_offset: {e}")))?;

    // Flatten per-covariate labels to (C x N) row-major i32. Batch
    // count per covariate is bounded by `max_batches` (default
    // 1024) << i32::MAX.
    let mut labels_flat = vec![0i32; c_count * n];
    for (ci, cov) in covariates.iter().enumerate() {
        for (i, &lab) in cov.labels.iter().enumerate() {
            debug_assert!(lab <= i32::MAX as u32, "batch label {lab} exceeds i32::MAX");
            labels_flat[ci * n + i] = lab as i32;
        }
    }
    let d_labels = dev
        .htod_copy(&labels_flat)
        .map_err(|e| AccelError::LinAlg(format!("upload batch labels: {e}")))?;

    // Per-cluster R-row scratch (size N) — used by z-sum and
    // grouped correction kernels. Filled per cluster via
    // memcpy_dtod from `d_r[ku*n..(ku+1)*n]`.
    let mut d_r_row = dev
        .alloc_zeros::<f32>(n)
        .map_err(|e| AccelError::LinAlg(format!("alloc R-row scratch: {e}")))?;

    // Task 2.6: pre-upload the global per-batch cell membership ONCE — it
    // is fixed for the whole run. `all_cells` concatenates each global
    // batch's cell indices in canonical (covariate, level) order;
    // `batch_start[gb]..batch_start[gb+1]` delimits batch `gb`. The
    // correction loop gathers each cluster's kept-batch cells from this
    // device buffer via `memcpy_dtod` instead of rebuilding `cells_concat`
    // on the host and re-uploading it per cluster (O(N) host work + PCIe
    // every cluster × outer-iter → one upload total).
    let mut all_cells: Vec<i32> = Vec::new();
    let mut batch_start: Vec<i32> = Vec::with_capacity(b + 1);
    batch_start.push(0);
    for gb in 0..b {
        let (ci, lvl) = state.gb_to_cov_level(gb);
        let cells = &state.batch_index[ci][lvl];
        all_cells.extend(cells.iter().map(|&i| i as i32));
        batch_start.push(all_cells.len() as i32);
    }
    let all_cells_total = all_cells.len();
    let d_all_cells = dev
        .htod_copy(&all_cells)
        .map_err(|e| AccelError::LinAlg(format!("upload global cell membership: {e}")))?;

    // Task 2.6: persistent correction scratch hoisted out of the
    // per-cluster K-loop. `b` (total batch levels) upper-bounds any
    // cluster's kept-batch count `b_prime`. Each is sized once and reused;
    // the z-sum / correction kernels only touch the valid `b_prime`-prefix.
    let mut d_z_sum_scratch = dev
        .alloc_zeros::<f32>((b * d).max(1))
        .map_err(|e| AccelError::LinAlg(format!("alloc z_sum scratch: {e}")))?;
    let mut d_w_scratch = dev
        .alloc_zeros::<f32>((b * d).max(1))
        .map_err(|e| AccelError::LinAlg(format!("alloc W scratch: {e}")))?;
    let mut d_cells_concat_scratch = dev
        .alloc_zeros::<i32>(all_cells_total.max(1))
        .map_err(|e| AccelError::LinAlg(format!("alloc cells_concat scratch: {e}")))?;
    let mut d_offsets_scratch = dev
        .alloc_zeros::<i32>(b + 1)
        .map_err(|e| AccelError::LinAlg(format!("alloc batch_offsets scratch: {e}")))?;

    // Per-sub-iter shuffled cell order (CPU shuffle, GPU consumes
    // contiguous block ranges). Allocated once at full size N.
    let mut d_order = dev
        .alloc_zeros::<i32>(n)
        .map_err(|e| AccelError::LinAlg(format!("alloc d_order: {e}")))?;

    // Block-cells scratch (size = max possible block_len). Filled
    // per block via memcpy_dtod from a slice of d_order.
    let block_size_cfg = state.config.block_size.max(1.0 / n as f64);
    let n_blocks = (1.0 / block_size_cfg).ceil() as usize;
    let block_len = n.div_ceil(n_blocks.max(1));
    let mut d_block_cells = dev
        .alloc_zeros::<i32>(block_len)
        .map_err(|e| AccelError::LinAlg(format!("alloc d_block_cells: {e}")))?;

    // Objective scratch.
    let mut d_obj_cell = dev
        .alloc_zeros::<f32>(n)
        .map_err(|e| AccelError::LinAlg(format!("alloc d_obj_cell: {e}")))?;
    let mut d_cross_kgb = dev
        .alloc_zeros::<f32>(k * b)
        .map_err(|e| AccelError::LinAlg(format!("alloc d_cross_kgb: {e}")))?;

    let mut converged = false;
    let mut iters_used = 0usize;

    // Reusable CPU-side cell-order buffer (avoids per-sub-iter
    // allocation). Shuffled deterministically via state.rng.
    let mut order_usize: Vec<usize> = (0..n).collect();
    let mut order_i32: Vec<i32> = vec![0i32; n];

    // G10.2: graph-capture state for the k-means sub-iter kernel
    // sequence. Hoisted outside the outer iter loop so a single
    // capture is reused across all max_iter × max_iter_kmeans
    // sub-iters of this call. All buffers and shape scalars passed
    // to `run_kmeans_subiter_kernels` (d_order, d_r, d_o, d_e,
    // d_dist, ..., n, k, b, c_count, n_blocks, block_len) are
    // allocated/computed above and remain stable for the rest of
    // the function — the only inter-sub-iter change is d_order's
    // contents, which the captured graph reads by pointer.
    let graphs_enabled = cuda_graphs_enabled();
    let pts: Arc<scx_gpu::CudaStream> = dev.context().per_thread_stream();
    let dev_pts = dev.with_stream(pts.clone());
    let mut sub_graph: Option<scx_gpu::CudaGraph> = None;
    // Warm-up runs once across the entire Harmony call to populate
    // `dev`'s module cache (so the captured region in the next
    // sub-iter doesn't trigger a module load). After this flips
    // true the dispatch picks replay (sub_graph is Some), capture
    // (sub_graph is None and capture not yet failed), or direct
    // fallback (capture_failed is true).
    let mut warmed_up = false;
    // If the first capture attempt fails, don't keep retrying on
    // every later sub-iter — fall back to direct dispatch for the
    // rest of the call. Honours the existing "don't try capture
    // again for this call" contract that the hoisted layout would
    // otherwise quietly violate.
    let mut capture_failed = false;

    for iter in 0..state.config.max_iter {
        iters_used = iter + 1;

        if iter > 0 {
            // Cold-start R on GPU: Z_cos = l2_normalize(Z_corr),
            // then dist = 2 · (1 − Y⊤ Z_cos), then plain softmax
            // (no penalty) → R, then refresh O/E from the new R.
            // No CPU round-trip in this branch.
            dev.stream()
                .memcpy_dtod(&d_z_corr, &mut d_z_cos)
                .map_err(|e| AccelError::LinAlg(format!("memcpy Z_corr->Z_cos: {e}")))?;
            gpu_harmony_l2_normalize_cols(&dev, &mut d_z_cos, d, n)
                .map_err(|e| AccelError::LinAlg(format!("GPU L2 normalize: {e}")))?;

            // Refresh d_y from CPU state.y (mutated by previous
            // iter's correction step).
            let y_f32 = f64_to_f32(&state.y);
            dev.stream()
                .memcpy_htod(&y_f32, &mut d_y)
                .map_err(|e| AccelError::LinAlg(format!("upload Y: {e}")))?;

            dispatch_harmony_distances(&dev, &cublas, &d_y, &d_z_cos, &mut d_dist, d, k, n)?;
            gpu_harmony_softmax(&dev, &d_dist, &d_sigma, &mut d_r, k, n)
                .map_err(|e| AccelError::LinAlg(format!("GPU softmax: {e}")))?;
            gpu_harmony_compute_o_e_full(
                &dev,
                &d_r,
                &d_labels,
                &d_cov_offset,
                &d_pr_b,
                &mut d_o,
                &mut d_e,
                c_count,
                k,
                n,
                b,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU compute_o_e: {e}")))?;
        }

        // G10.2: k-means sub-loop dispatch. See the hoisted
        // `sub_graph` / `warmed_up` setup above the outer iter
        // loop for the capture-once contract.
        //
        // Capture stream: `dev_pts = dev.with_stream(per_thread_
        // stream())`. The per-thread default stream is capturable
        // and does NOT flip cudarc into multi-stream-mode (see
        // gpu_graph module docs for why that matters). The clone
        // shares the module cache via shallow-clone, so the
        // captured kernels reuse the HARMONY_PTX module that the
        // warm-up sub-iter loaded into `dev`'s cache.
        //
        // CPU shuffle + memcpy_htod(d_order) stays outside the
        // captured region; the captured graph reads the freshly-
        // shuffled order at replay time (the d_order pointer is
        // stable, only its contents change). Likewise the
        // post-sub-iter sync + dtoh + convergence check stay on
        // the host orchestrator.
        //
        // `local_obj` is the per-outer-iter k-means objective
        // window consumed by `check_convergence_kmeans`; it must
        // reset per outer iter (each outer iter runs an
        // independent k-means trajectory), so it stays declared
        // here rather than alongside the hoisted graph state.
        let mut local_obj: Vec<f64> = Vec::new();

        for _sub in 0..state.config.max_iter_kmeans {
            // Reshuffle order for this sub-iter (mirrors CPU
            // update_r's per-sub-iter shuffle).
            order_usize.shuffle(&mut state.rng);
            for (dst, &src) in order_i32.iter_mut().zip(order_usize.iter()) {
                *dst = src as i32;
            }

            // Upload onto the same stream the kernels will run
            // on. When graphs are enabled the kernels run on
            // per_thread_stream (via dev_pts); upload there too
            // so the captured graph sees a fully-uploaded order
            // without a cross-stream wait.
            let upload_stream = if graphs_enabled {
                dev_pts.stream()
            } else {
                dev.stream()
            };
            upload_stream
                .memcpy_htod(&order_i32, &mut d_order)
                .map_err(|e| AccelError::LinAlg(format!("upload order: {e}")))?;

            // Sub-iter execution strategy:
            //
            // 1. First sub-iter of the whole Harmony call (when
            //    graphs are enabled): run directly to populate
            //    `dev`'s module cache. dev_pts shares this cache
            //    via shallow clone, so the subsequent capture
            //    will not issue a module load inside the captured
            //    region. Fires once across all outer iters.
            // 2. After warm-up: try to capture the sub-iter the
            //    first time `sub_graph` is None. Subsequent
            //    sub-iters (in this AND every later outer iter)
            //    launch the cached graph.
            // 3. Kill switch (`!graphs_enabled`) or capture
            //    failure: run directly on `dev`.
            if !graphs_enabled || !warmed_up || capture_failed {
                // Direct path (warm-up, kill switch, or post-
                // capture-failure for the rest of the call).
                let active_dev = if graphs_enabled { &dev_pts } else { &dev };
                run_kmeans_subiter_kernels(
                    active_dev,
                    &d_order,
                    &mut d_block_cells,
                    &mut d_r,
                    &mut d_o,
                    &mut d_e,
                    &d_dist,
                    &d_sigma,
                    &d_theta,
                    &d_labels,
                    &d_cov_offset,
                    &d_pr_b,
                    &mut d_obj_cell,
                    &mut d_cross_kgb,
                    n,
                    k,
                    b,
                    c_count,
                    n_blocks,
                    block_len,
                )?;
                warmed_up = true;
            } else if let Some(graph) = &sub_graph {
                // Replay path (the common case after the first
                // captured sub-iter).
                graph.launch().map_err(|e| {
                    AccelError::LinAlg(format!("harmony sub-iter graph.launch: {e}"))
                })?;
            } else {
                // First capture attempt — fires on the sub-iter
                // after the warm-up direct run, then never again
                // for the rest of the Harmony call (subsequent
                // sub-iters across all outer iters take the
                // replay branch above). capture_graph records
                // into the graph WITHOUT executing; we then
                // launch the returned graph to actually run the
                // work for this sub-iter.
                #[cfg(test)]
                HARMONY_CAPTURE_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let pts_for_capture = pts.clone();
                let capture_result = capture_graph(&pts_for_capture, |_stream| {
                    run_kmeans_subiter_kernels(
                        &dev_pts,
                        &d_order,
                        &mut d_block_cells,
                        &mut d_r,
                        &mut d_o,
                        &mut d_e,
                        &d_dist,
                        &d_sigma,
                        &d_theta,
                        &d_labels,
                        &d_cov_offset,
                        &d_pr_b,
                        &mut d_obj_cell,
                        &mut d_cross_kgb,
                        n,
                        k,
                        b,
                        c_count,
                        n_blocks,
                        block_len,
                    )
                    .map_err(|e| scx_gpu::GpuError::CudaError(format!("{e}")))
                });
                match capture_result {
                    Ok(Some(g)) => {
                        // Capture succeeded — actually run the
                        // work via the captured graph and stash
                        // for subsequent replays.
                        g.launch().map_err(|e| {
                            AccelError::LinAlg(format!("harmony sub-iter first graph.launch: {e}"))
                        })?;
                        sub_graph = Some(g);
                    }
                    _ => {
                        // Capture failed or returned no graph —
                        // fall back to direct run for THIS
                        // sub-iter and don't try capture again
                        // for this call. The `capture_failed`
                        // flag pushes every later sub-iter through
                        // the direct path above.
                        capture_failed = true;
                        run_kmeans_subiter_kernels(
                            &dev_pts,
                            &d_order,
                            &mut d_block_cells,
                            &mut d_r,
                            &mut d_o,
                            &mut d_e,
                            &d_dist,
                            &d_sigma,
                            &d_theta,
                            &d_labels,
                            &d_cov_offset,
                            &d_pr_b,
                            &mut d_obj_cell,
                            &mut d_cross_kgb,
                            n,
                            k,
                            b,
                            c_count,
                            n_blocks,
                            block_len,
                        )?;
                    }
                }
            }

            // Per-sub-iter sync + objective. When graphs are
            // enabled the kernels ran on per_thread_stream; sync
            // there. dtoh through dev_pts uses the same stream
            // so ordering is correct.
            let active_for_sync = if graphs_enabled { &dev_pts } else { &dev };
            active_for_sync
                .synchronize()
                .map_err(|e| AccelError::LinAlg(format!("sync: {e}")))?;
            let obj = compute_objective_gpu(active_for_sync, &d_obj_cell, &d_cross_kgb, n)?;
            local_obj.push(obj);
            state.objective_kmeans.push(obj);
            if check_convergence_kmeans(
                &local_obj,
                state.config.window_size,
                state.config.epsilon_kmeans,
            ) {
                break;
            }
        }

        // Sync state.r, state.o, state.e back to CPU for the
        // correction step's small regression solve. R is needed
        // implicitly (we slice d_r per cluster on-device, no CPU
        // R copy needed); O/E feed the per-cluster cov matrix.
        let o_back = dev
            .dtoh_copy(&d_o)
            .map_err(|e| AccelError::LinAlg(format!("download O: {e}")))?;
        let e_back = dev
            .dtoh_copy(&d_e)
            .map_err(|e| AccelError::LinAlg(format!("download E: {e}")))?;
        for (dst, src) in state.o.iter_mut().zip(o_back.iter()) {
            *dst = *src as f64;
        }
        for (dst, src) in state.e.iter_mut().zip(e_back.iter()) {
            *dst = *src as f64;
        }

        // --- Correction step ---
        //
        // Reset Z_corr = Z_orig on device.
        dev.stream()
            .memcpy_dtod(&d_z_orig, &mut d_z_corr)
            .map_err(|e| AccelError::LinAlg(format!("reset Z_corr: {e}")))?;

        // Mirror CPU correction logic, with the per-batch scatter done on
        // GPU. Centroid rows W[0, :] still flow through CPU state.y.
        for ku in 0..k {
            let (kept, active_cov) = prune_batches_for_cluster(&state, ku);
            if active_cov == 0 || kept.is_empty() {
                continue;
            }
            let b_prime = kept.len();
            let size = b_prime + 1;

            let lambda = match &state.lambda_fixed {
                Some(lam) => {
                    let mut local = vec![0f64; size];
                    for (j, &gb) in kept.iter().enumerate() {
                        local[j + 1] = lam[gb + 1];
                    }
                    local
                }
                None => build_dynamic_lambda(&state, ku, &kept),
            };

            let mut cov = vec![0f64; size * size];
            let mut sum_o = 0f64;
            for (j, &gb) in kept.iter().enumerate() {
                let o_kb = state.o[ku * b + gb];
                cov[j + 1] = o_kb;
                cov[(j + 1) * size] = o_kb;
                cov[(j + 1) * size + (j + 1)] = o_kb;
                sum_o += o_kb;
            }
            cov[0] = sum_o;
            for j in 0..size {
                cov[j * size + j] += lambda[j];
            }

            let inv_cov = if c_count == 1 {
                match arrowhead_inverse(&cov, size) {
                    Ok(inv) => inv,
                    Err(AccelError::NumericalInstability(msg)) => {
                        log::warn!(
                            "harmony: arrowhead_inverse unstable ({msg}); \
                                 falling back to full LU inverse"
                        );
                        full_matrix_inverse(&cov, size)?
                    }
                    Err(e) => return Err(e),
                }
            } else {
                full_matrix_inverse(&cov, size)?
            };

            // Task 2.6: gather this cluster's kept-batch cells from the
            // pre-uploaded global membership (`d_all_cells`) into the
            // reused `d_cells_concat_scratch` via on-device `memcpy_dtod`,
            // and build only the tiny `batch_offsets` (length b_prime+1) on
            // host. Empty kept batches contribute zero-length spans, exactly
            // as the previous host rebuild did. Replaces the per-cluster
            // O(N) host concatenation + full `cells_concat` H→D upload.
            let mut batch_offsets: Vec<i32> = Vec::with_capacity(b_prime + 1);
            batch_offsets.push(0);
            let mut off: usize = 0;
            for &gb in &kept {
                let gstart = batch_start[gb] as usize;
                let gend = batch_start[gb + 1] as usize;
                let len = gend - gstart;
                if len > 0 {
                    let src = d_all_cells
                        .try_slice(gstart..gend)
                        .ok_or_else(|| AccelError::LinAlg("d_all_cells slice oob".into()))?;
                    let mut dst = d_cells_concat_scratch
                        .try_slice_mut(off..off + len)
                        .ok_or_else(|| {
                            AccelError::LinAlg("cells_concat scratch slice oob".into())
                        })?;
                    dev.stream()
                        .memcpy_dtod(&src, &mut dst)
                        .map_err(|e| AccelError::LinAlg(format!("gather kept cells: {e}")))?;
                }
                off += len;
                batch_offsets.push(off as i32);
            }
            let n_kept_total = off;
            // Upload the tiny offsets into the reused scratch prefix.
            {
                let mut off_dst = d_offsets_scratch
                    .try_slice_mut(0..batch_offsets.len())
                    .ok_or_else(|| AccelError::LinAlg("offsets scratch slice oob".into()))?;
                dev.stream()
                    .memcpy_htod(&batch_offsets, &mut off_dst)
                    .map_err(|e| AccelError::LinAlg(format!("upload batch_offsets: {e}")))?;
            }

            // R[k, :] is already on device — slice d_r[ku*n..(ku+1)*n]
            // and memcpy_dtod into the hoisted d_r_row scratch.
            // No host round-trip.
            let r_view = d_r
                .try_slice(ku * n..(ku + 1) * n)
                .ok_or_else(|| AccelError::LinAlg("d_r slice out of bounds".into()))?;
            dev.stream()
                .memcpy_dtod(&r_view, &mut d_r_row)
                .map_err(|e| AccelError::LinAlg(format!("memcpy R row: {e}")))?;

            // GPU z-sum: z_sum[j, t] = Σ_{i in batch j} R[k,i] · Z_orig[t,i].
            // Writes into the reused `d_z_sum_scratch` (only the b_prime×d
            // prefix; the kernel overwrites every touched slot). f32 result
            // is downloaded and promoted to f64 for the regression solve.
            gpu_harmony_z_sum(
                &dev,
                &d_r_row,
                &d_z_orig,
                &d_cells_concat_scratch,
                &d_offsets_scratch,
                &mut d_z_sum_scratch,
                b_prime,
                d,
                n,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU z-sum: {e}")))?;
            // Download only the valid b_prime×d prefix of the reused scratch
            // (not the full b×d capacity).
            let z_sum_f32 = {
                let view = d_z_sum_scratch
                    .try_slice(0..b_prime * d)
                    .ok_or_else(|| AccelError::LinAlg("z_sum scratch slice oob".into()))?;
                let mut host = vec![0f32; b_prime * d];
                dev.stream()
                    .memcpy_dtoh(&view, &mut host)
                    .map_err(|e| AccelError::LinAlg(format!("download z_sum: {e}")))?;
                dev.synchronize()
                    .map_err(|e| AccelError::LinAlg(format!("sync z_sum: {e}")))?;
                host
            };
            let z_sum: Vec<f64> = z_sum_f32.iter().map(|&v| v as f64).collect();
            let mut z_sum_all = vec![0f64; d];
            for j in 0..b_prime {
                let row_off = j * d;
                for t in 0..d {
                    z_sum_all[t] += z_sum[row_off + t];
                }
            }

            // CPU regression solve: small (B'+1) x (B'+1) inv_cov
            // times (z_sum_all, z_sum[0..b_prime]) → W of shape
            // (size, d). Stays on host because B' is small and
            // inv_cov already lives here.
            let mut w = vec![0f64; size * d];
            for r in 0..size {
                let ic0 = inv_cov[r * size];
                let row_off = r * d;
                for t in 0..d {
                    w[row_off + t] = ic0 * z_sum_all[t];
                }
                for j in 0..b_prime {
                    let icj = inv_cov[r * size + (j + 1)];
                    if icj == 0.0 {
                        continue;
                    }
                    let zj_off = j * d;
                    for t in 0..d {
                        w[row_off + t] += icj * z_sum[zj_off + t];
                    }
                }
            }

            // Extract centroid into CPU state.y; zero W[0, :].
            let y_off = ku * d;
            for t in 0..d {
                state.y[y_off + t] = w[t];
                w[t] = 0.0;
            }

            // Build w_flat = W rows 1..size (row 0 was the centroid
            // and is zeroed above). One row per kept batch, including
            // empties (kernel never reads their slots since
            // batch_offsets[j+1] == batch_offsets[j] for empties).
            let w_flat: Vec<f32> = w[d..(b_prime + 1) * d].iter().map(|&v| v as f32).collect();

            if n_kept_total == 0 {
                // All kept batches are empty — nothing to scatter.
                continue;
            }
            // Upload W into the reused scratch prefix (Task 2.6).
            {
                let mut w_dst = d_w_scratch
                    .try_slice_mut(0..w_flat.len())
                    .ok_or_else(|| AccelError::LinAlg("W scratch slice oob".into()))?;
                dev.stream()
                    .memcpy_htod(&w_flat, &mut w_dst)
                    .map_err(|e| AccelError::LinAlg(format!("upload W: {e}")))?;
            }
            gpu_harmony_correction_grouped(
                &dev,
                &mut d_z_corr,
                &d_r_row,
                &d_w_scratch,
                &d_cells_concat_scratch,
                &d_offsets_scratch,
                b_prime,
                n_kept_total,
                d,
                n,
            )
            .map_err(|e| AccelError::LinAlg(format!("GPU correction (grouped): {e}")))?;
        }

        // L2-normalize Y columns on GPU (d x K). Y was just mutated
        // by CPU correction; refresh d_y in place and run the kernel.
        let y_f32 = f64_to_f32(&state.y);
        dev.stream()
            .memcpy_htod(&y_f32, &mut d_y)
            .map_err(|e| AccelError::LinAlg(format!("upload Y for norm: {e}")))?;
        gpu_harmony_l2_normalize_cols(&dev, &mut d_y, d, k)
            .map_err(|e| AccelError::LinAlg(format!("GPU L2 normalize Y: {e}")))?;
        dev.synchronize()
            .map_err(|e| AccelError::LinAlg(format!("sync: {e}")))?;
        let y_back = dev
            .dtoh_copy(&d_y)
            .map_err(|e| AccelError::LinAlg(format!("download Y: {e}")))?;
        for (dst, src) in state.y.iter_mut().zip(y_back.iter()) {
            *dst = *src as f64;
        }

        // Z_corr stays on the device. The next iter's distance step
        // will memcpy_dtod it into d_z_cos and L2-normalize there;
        // there's no consumer of state.z_corr inside the loop on the
        // GPU path. We download once at the end of the run.

        if let Some(&last) = state.objective_kmeans.last() {
            state.objective_harmony.push(last);
        }
        if check_convergence_harmony(&state.objective_harmony, state.config.epsilon_harmony) {
            converged = true;
            break;
        }
    }

    // Final download of Z_corr to CPU state so the row-major output
    // builder below can read it. Saves the per-iter PCIe round-trip
    // that the previous orchestration did.
    let z_corr_back = dev
        .dtoh_copy(&d_z_corr)
        .map_err(|e| AccelError::LinAlg(format!("download Z_corr: {e}")))?;
    for (dst, src) in state.z_corr.iter_mut().zip(z_corr_back.iter()) {
        *dst = *src as f64;
    }

    // Build output as (N x d) row-major f64.
    let mut z_out = vec![0f64; n * d];
    for i in 0..n {
        for j in 0..d {
            z_out[i * d + j] = state.z_corr[j + i * d];
        }
    }

    Ok(HarmonyResult {
        z_corrected: z_out,
        n_obs: n,
        n_pcs: d,
        n_clusters: state.k,
        objective_harmony: std::mem::take(&mut state.objective_harmony),
        n_iterations: iters_used,
        converged,
    })
}
