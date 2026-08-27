//! Device-resident randomized-PCA power loop (V3 plan Task 2.5).
//!
//! The streaming randomized-PCA core (`gpu_pca::randomized_pca_core`) drives the
//! power loop through [`crate::linear_operator::CenteredSparseOperator`], which
//! re-runs the host-orchestrated `RawGpuShardSource`
//! streaming pipeline on **every** `matmat`/`rmatmat` — re-reading, re-decoding
//! and re-uploading the entire matrix once per call (≈7 full passes for the
//! default 2 power iterations). For inputs whose full CSR fits device memory,
//! this module instead:
//!
//! 1. Drains the [`ShardSource`] **once** into a single device-resident
//!    [`GpuCsr`] + cuSPARSE descriptor ([`try_build_resident_csr`]).
//! 2. Runs the power loop as two plain `cusparseSpMM` calls per iteration on
//!    that fixed descriptor — no per-iteration decode/upload
//!    ([`run_resident_power_loop`]).
//!
//! SpMM-segment CUDA-graph capture (an opt-in `cusparseSpMM` graph replay) was
//! removed: it poisoned the CUDA context on the cuSPARSE versions tested
//! (CUDA 12.x on H100) and was never a measured win over the direct resident
//! path, which already delivers the residency benefit.
//!
//! ## Math mode / SpMM algorithm
//!
//! The [`GpuPcaTuning`] math mode is applied to the shared cuBLAS handle by the
//! caller. The SpMM algorithm follows the `SpmmAlgPolicy` math policy.

use cudarc::driver::safe::CudaSlice;

use scx_format_io::ShardSource;

use crate::cublas::CublasHandle;
use crate::cusolver::{CusolverHandle, QrMethod};
use crate::cusparse::{CuSparseWorkspacePool, CusparseHandle, CusparseSpMatDescr};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_pca::{
    forward_mean_prefactor, qr_swap_into, spmm_forward_segment, spmm_transpose_segment,
    transpose_centering_correction, GpuPcaScratch, PcaMultiplyCtx, RowSegment,
};
use crate::math_policy::GpuPcaTuning;
use crate::pca_operator::{run_power_loop, ForwardOperand, PcaBuf, PcaOperator};
use crate::shard_decode::GpuCsr;

/// VRAM headroom factor applied to the resident-CSR + scratch estimate before
/// comparing against free device memory.
const RESIDENT_VRAM_HEADROOM: f64 = 1.2;

/// Kill switch for GPU PCA CSR residency. `SCX_GPU_PCA_RESIDENT=0` forces the
/// streaming power loop (the whole matrix re-decoded and re-uploaded on every
/// `matmat`/`rmatmat`) — an escape hatch for a host where the extra VRAM is not
/// available, and the "off" arm for an A/B.
///
/// The mirror of `SCX_GPU_DE_RESIDENT` on the GPU-DE side, down to being read
/// **once per process**: `pyscx/tests/test_gpu_pca_resident.py` runs its two
/// arms as subprocesses precisely because the setting is supposed to be
/// process-stable, and a per-call `env::var` would quietly make that isolation
/// unnecessary — and the promise untrue for anyone relying on it.
///
/// This is also what makes the streaming path *reachable* from a test at all.
/// Residency is otherwise decided dynamically against free VRAM, so no fixture
/// can force the streaming branch on an 80 GB card, and the branch that ignored
/// `spmm_policy` was therefore never exercised.
fn pca_resident_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("SCX_GPU_PCA_RESIDENT").as_deref(), Ok("0")))
}

/// Drain `source` into a single device-resident CSR, or return `None` when the
/// resident matrix plus PCA dense scratch would not fit device memory, or when
/// [`pca_resident_enabled`] is off (the caller then falls back to the streaming
/// power loop).
///
/// `k` is the oversampled rank used to size the dense scratch in the VRAM
/// pre-flight (`2·n_obs·k + n_vars·k` f32 for `d_y` / `d_z` + small vectors).
///
/// cuSPARSE SpMM tolerates unsorted column indices, so — unlike the GPU-DE
/// staging path — this builder does not reject unsorted/duplicate columns; it
/// concatenates shards verbatim.
pub(crate) fn try_build_resident_csr(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    k: usize,
) -> Result<Option<GpuCsr>, GpuError> {
    if !pca_resident_enabled() {
        return Ok(None);
    }
    let (n_obs, n_vars) = source.shape();
    let n_shards = source.n_shards();
    if n_obs == 0 || n_vars == 0 || n_shards == 0 {
        return Ok(None);
    }

    // Preflight the nnz budget BEFORE accumulating any shard on the host, so a
    // matrix that will not fit device memory cannot first OOM host RAM
    // (finding 1). The fixed (nnz-independent) cost is the dense PCA scratch
    // (`d_y`/`d_z` + a transient transpose/U buffer + small vectors) plus the
    // resident indptr (i64 + the i32 copy the cuSPARSE descriptor owns); each
    // nnz then costs 8 bytes (i32 index + f32 value). Compute in u64 with
    // saturating arithmetic so pathological shapes can't overflow.
    let free_budget = {
        let (free, _total) = dev.free_memory()?;
        (free as f64 / RESIDENT_VRAM_HEADROOM) as u64
    };
    let nk = (n_obs as u64).saturating_mul(k as u64);
    let scratch_bytes = nk
        .saturating_mul(2)
        .saturating_add((n_vars as u64).saturating_mul(k as u64))
        .saturating_add((k as u64).saturating_mul(4))
        .saturating_mul(4);
    let indptr_bytes = (n_obs as u64 + 1).saturating_mul(8 + 4);
    let fixed_bytes = scratch_bytes.saturating_add(indptr_bytes);
    if fixed_bytes >= free_budget {
        return Ok(None);
    }
    // Max resident nnz allowed by VRAM, capped at i32::MAX: the cuSPARSE
    // descriptor downcasts the (concatenated) row offsets to i32
    // (`GpuCsr::to_cusparse_csr`), which the per-shard streaming path never
    // exceeded but the full concatenation can — so total nnz > i32::MAX must
    // hard-fall back to streaming (finding 2).
    let max_nnz = ((free_budget - fixed_bytes) / 8).min(i32::MAX as u64) as usize;
    if max_nnz == 0 {
        return Ok(None);
    }

    let mut indptr: Vec<i64> = Vec::with_capacity(n_obs + 1);
    indptr.push(0);
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    let mut row_acc: i64 = 0;

    for s in 0..n_shards {
        let csr = source
            .read_shard(s)
            .map_err(|e| GpuError::InvalidShard(format!("resident PCA read shard {s}: {e}")))?;
        let nr = csr.n_rows();
        if nr == 0 {
            continue;
        }
        // Bail to the streaming fallback the moment the running nnz exceeds the
        // VRAM/i32 budget — this bounds the host accumulation to ~one device's
        // worth of CSR rather than materialising an over-budget matrix first.
        if indices.len() + csr.indices.len() > max_nnz {
            return Ok(None);
        }
        indices.extend_from_slice(&csr.indices);
        data.extend_from_slice(&csr.data);
        // Shard indptr starts at 0; offset each row pointer by the running
        // global nnz so the concatenated indptr is globally monotonic.
        for r in 0..nr {
            indptr.push(row_acc + csr.indptr[r + 1]);
        }
        row_acc += *csr.indptr.last().unwrap_or(&0);
    }

    // Empty / all-zero matrix → let the streaming path handle it.
    let nnz = indices.len();
    if indptr.len() != n_obs + 1 || nnz == 0 {
        return Ok(None);
    }

    let d_indptr = dev.htod_copy(&indptr)?;
    let d_indices = dev.htod_copy(&indices)?;
    let d_data = dev.htod_copy(&data)?;
    Ok(Some(GpuCsr {
        indptr: d_indptr,
        indices: d_indices,
        data: d_data,
        shape: (n_obs, n_vars),
    }))
}

/// [`PcaOperator`] over a device-resident CSR.
///
/// The whole matrix is one segment, `(0, n_obs)`, against a fixed cuSPARSE
/// descriptor — so every method here is the shared segment function called once.
/// The streaming operator calls the same functions once per shard. That is the
/// entire difference between the two paths, and stating it this way is what
/// stops the next divergence: review §8.11 was `spmm_policy` reaching one of
/// these two multiplies and not the other, and there is now one place for it to
/// reach.
///
/// `d_mc` and `d_sum_q` are length-`k` scratch held across the power loop rather
/// than allocated per multiply.
struct ResidentPcaOperator<'a> {
    ctx: PcaMultiplyCtx<'a>,
    cublas: &'a CublasHandle,
    cusolver: &'a CusolverHandle,
    qr_method: QrMethod,
    desc: CusparseSpMatDescr,
    d_means: Option<&'a CudaSlice<f32>>,
    d_omega: &'a CudaSlice<f32>,
    scratch: &'a mut GpuPcaScratch,
    d_mc: CudaSlice<f32>,
    d_sum_q: CudaSlice<f32>,
    pool: CuSparseWorkspacePool,
}

impl PcaOperator for ResidentPcaOperator<'_> {
    fn matmat(&mut self, src: ForwardOperand) -> Result<(), GpuError> {
        let Self {
            ctx,
            cublas,
            desc,
            d_means,
            d_omega,
            scratch,
            d_mc,
            pool,
            ..
        } = self;
        let GpuPcaScratch { d_y, d_z } = &mut **scratch;
        let d_v: &CudaSlice<f32> = match src {
            ForwardOperand::Omega => d_omega,
            ForwardOperand::Z => d_z,
        };

        // Kept for the same measured reason as the streaming operator's — see
        // the note there. One segment covers every row here, so the only thing
        // this insures against is cuSPARSE ceasing to write `C`'s all-zero rows
        // under β = 0, which it does write today and does not promise to.
        ctx.dev
            .stream()
            .memset_zeros(d_y)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("resident matmat zero: {e}")))?;
        if let Some(d_mu) = d_means {
            forward_mean_prefactor(ctx, cublas, d_v, d_mu, d_mc)?;
        }
        let mc = d_means.is_some().then_some(&*d_mc);
        spmm_forward_segment(ctx, pool, desc, d_v, d_y, mc, RowSegment::whole(ctx.n_obs))
    }

    fn rmatmat(&mut self) -> Result<(), GpuError> {
        let Self {
            ctx,
            desc,
            d_means,
            scratch,
            d_sum_q,
            pool,
            ..
        } = self;
        let GpuPcaScratch { d_y, d_z } = &mut **scratch;

        // Required, not defensive: the segment function accumulates with β = 1
        // so that a multi-shard streaming drive can sum across shards. With one
        // segment the accumulation is into zeros, which is the same arithmetic
        // the pre-unification `β = 0` did.
        ctx.dev
            .stream()
            .memset_zeros(d_z)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("resident rmatmat zero: {e}")))?;
        spmm_transpose_segment(ctx, pool, desc, d_y, d_z, RowSegment::whole(ctx.n_obs))?;
        if let Some(d_mu) = d_means {
            transpose_centering_correction(ctx, d_y, d_z, d_mu, d_sum_q)?;
        }
        Ok(())
    }

    fn qr(&mut self, buf: PcaBuf) -> Result<(), GpuError> {
        let Self {
            ctx,
            cublas,
            cusolver,
            qr_method,
            scratch,
            ..
        } = self;
        let (slot, rows) = match buf {
            PcaBuf::Y => (&mut scratch.d_y, ctx.n_obs),
            PcaBuf::Z => (&mut scratch.d_z, ctx.n_vars),
        };
        qr_swap_into(ctx.dev, cusolver, cublas, *qr_method, slot, rows, ctx.k)
    }
}

/// Run the randomized-PCA power loop on a device-resident CSR, filling
/// `scratch.d_y` with the final `Q` (`n_obs × k`, col-major) and `scratch.d_z`
/// with the final `B = (X − μ)ᵀ·Q` (`n_vars × k`, col-major) — exactly the two
/// buffers the streaming core leaves for the downstream SVD.
///
/// The loop itself is [`run_power_loop`]; this function only assembles the
/// operator. SpMM-segment CUDA-graph capture was removed, so nothing here
/// requires a fixed device pointer across iterations — which is why the QR
/// adapter is the shared swap form and not a second, copy-based one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_resident_power_loop(
    dev: &GpuDevice,
    gpu_csr: &GpuCsr,
    scratch: &mut GpuPcaScratch,
    d_omega: &CudaSlice<f32>,
    d_means: Option<&CudaSlice<f32>>,
    cusparse: &CusparseHandle,
    cublas: &CublasHandle,
    cusolver: &CusolverHandle,
    qr_method: QrMethod,
    n_obs: usize,
    n_vars: usize,
    k: usize,
    n_power_iterations: usize,
    tuning: GpuPcaTuning,
) -> Result<(), GpuError> {
    // SpMM-segment CUDA-graph capture was removed: capturing `cusparseSpMM`
    // poisons the CUDA context on the cuSPARSE versions tested (CUDA 12.x on
    // H100) and was never a measured win. The residency benefit (single upload +
    // one SpMM per segment, no per-iteration re-decode) is delivered by the
    // direct loop regardless.
    //
    // Resident scratch + cuSPARSE descriptor are built on the default stream.
    // The descriptor downcasts indptr i64→i32.
    let d_mc = dev.alloc_zeros::<f32>(k)?;
    let d_sum_q = dev.alloc_zeros::<f32>(k)?;
    let desc = gpu_csr.to_cusparse_csr(dev, dev.stream())?;
    dev.synchronize()?;

    let mut op = ResidentPcaOperator {
        ctx: PcaMultiplyCtx {
            dev,
            cusparse,
            n_obs,
            n_vars,
            k,
            alg: tuning.spmm_policy.to_alg(),
        },
        cublas,
        cusolver,
        qr_method,
        desc,
        d_means,
        d_omega,
        scratch,
        d_mc,
        d_sum_q,
        pool: CuSparseWorkspacePool::new(),
    };
    run_power_loop(&mut op, n_power_iterations)?;
    dev.synchronize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::cusolver::QrMethod;
    use crate::gpu_graph::set_cuda_graphs_enabled_override;
    use crate::gpu_pca::{gpu_column_sums_into, gpu_randomized_pca};
    use crate::math_policy::GpuPcaTuning;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use scx_format_io::ShardSource;
    use scx_sparse::ScxCsr;

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

    /// Random CSR (sorted columns per row) split into `n_shards` row-shards.
    fn random_source(n_rows: usize, n_cols: usize, n_shards: usize, seed: u64) -> InMemorySource {
        let mut rng = StdRng::seed_from_u64(seed);
        let rows_per = n_rows.div_ceil(n_shards);
        let mut shards = Vec::new();
        let mut r0 = 0;
        while r0 < n_rows {
            let r1 = (r0 + rows_per).min(n_rows);
            let mut indptr = vec![0i64];
            let mut indices: Vec<i32> = Vec::new();
            let mut data: Vec<f32> = Vec::new();
            for _ in r0..r1 {
                for c in 0..n_cols {
                    if rng.gen_bool(0.15) {
                        indices.push(c as i32); // ascending c → sorted
                        data.push(rng.gen_range(0.0..5.0));
                    }
                }
                indptr.push(indices.len() as i64);
            }
            shards.push(ScxCsr::new_unchecked(
                (r1 - r0, n_cols),
                indptr,
                indices,
                data,
            ));
            r0 = r1;
        }
        InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        }
    }

    /// |cosine| between column `c` of two row-major (n_obs × k) embeddings.
    fn abs_cosine(a: &[f32], b: &[f32], n_obs: usize, k: usize, c: usize) -> f32 {
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for i in 0..n_obs {
            let x = a[i * k + c] as f64;
            let y = b[i * k + c] as f64;
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        (dot.abs() / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32
    }

    /// Task 2.5: PCA results must be identical in subspace whether CUDA graphs
    /// are enabled or disabled. SpMM-segment capture was removed in Phase 3.4, so
    /// both runs take the direct resident path; this guards that toggling the
    /// graph kill switch does not perturb the resident PCA, and that the small
    /// (fits-VRAM) input routes through the resident path on both. Compared by
    /// |cosine| (sign-free), not bitwise.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_resident_capture_vs_direct_subspace() {
        let dev = require_gpu!();
        let (n_rows, n_cols, k) = (400usize, 60usize, 8usize);
        let source = random_source(n_rows, n_cols, 4, 17);
        let tuning = GpuPcaTuning::default();

        // Graphs enabled (PCA SpMM capture was removed in Phase 3.4 → direct path).
        set_cuda_graphs_enabled_override(Some(true));
        let captured = gpu_randomized_pca(
            &dev,
            &source,
            k,
            10,
            3,
            true,
            7,
            QrMethod::Householder,
            tuning,
        )
        .unwrap();

        // Capture off: direct resident path (never replays).
        set_cuda_graphs_enabled_override(Some(false));
        let direct = gpu_randomized_pca(
            &dev,
            &source,
            k,
            10,
            3,
            true,
            7,
            QrMethod::Householder,
            tuning,
        )
        .unwrap();

        // Restore env-controlled behaviour for sibling tests.
        set_cuda_graphs_enabled_override(None);

        for c in 0..k {
            let cos = abs_cosine(&captured.embeddings, &direct.embeddings, n_rows, k, c);
            assert!(
                cos > 0.99,
                "PC {c}: |cosine| {cos} between captured and direct resident PCA"
            );
        }
    }

    /// Materialize an [`InMemorySource`] into a dense row-major `(n_obs × n_vars)`
    /// f64 matrix for the CPU reference PCA below.
    fn materialize_dense(src: &InMemorySource) -> Vec<f64> {
        let (n_obs, n_vars) = (src.n_obs, src.n_vars);
        let mut dense = vec![0f64; n_obs * n_vars];
        let mut row0 = 0usize;
        for shard in &src.shards {
            let rows = shard.shape.0;
            for r in 0..rows {
                let lo = shard.indptr[r] as usize;
                let hi = shard.indptr[r + 1] as usize;
                for p in lo..hi {
                    let c = shard.indices[p] as usize;
                    dense[(row0 + r) * n_vars + c] = shard.data[p] as f64;
                }
            }
            row0 += rows;
        }
        dense
    }

    /// Dense-stored CSR carrying a strong rank-2 signal on top of a large
    /// per-column baseline offset. The big column means make this fixture
    /// sensitive to the mean-correction path; the clean rank-2 structure lets
    /// randomized PCA converge to the exact top-2 subspace, so a GPU-vs-CPU
    /// `|cosine|` comparison on the leading components is tight.
    fn low_rank_source_with_offset(
        n_obs: usize,
        n_vars: usize,
        n_shards: usize,
        seed: u64,
    ) -> InMemorySource {
        let mut rng = StdRng::seed_from_u64(seed);
        let w1: Vec<f64> = (0..n_vars).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let w2: Vec<f64> = (0..n_vars).map(|_| rng.gen_range(-1.0..1.0)).collect();
        // Large per-column baseline → large column means (the lever the bug hits).
        let base: Vec<f64> = (0..n_vars).map(|_| rng.gen_range(10.0..30.0)).collect();

        let rows_per = n_obs.div_ceil(n_shards);
        let mut shards = Vec::new();
        let mut r0 = 0;
        while r0 < n_obs {
            let r1 = (r0 + rows_per).min(n_obs);
            let mut indptr = vec![0i64];
            let mut indices: Vec<i32> = Vec::new();
            let mut data: Vec<f32> = Vec::new();
            for _ in r0..r1 {
                let f1: f64 = rng.gen_range(-3.0..3.0);
                let f2: f64 = rng.gen_range(-3.0..3.0);
                for c in 0..n_vars {
                    let noise: f64 = rng.gen_range(-0.05..0.05);
                    let v = base[c] + f1 * w1[c] + f2 * w2[c] + noise;
                    indices.push(c as i32); // dense, ascending c → sorted
                    data.push(v as f32);
                }
                indptr.push(indices.len() as i64);
            }
            shards.push(ScxCsr::new_unchecked(
                (r1 - r0, n_vars),
                indptr,
                indices,
                data,
            ));
            r0 = r1;
        }
        InMemorySource {
            shards,
            n_obs,
            n_vars,
        }
    }

    /// Regression for the device-resident randomized-PCA mean over-subtraction
    /// bug (finding G1): `d_sum_q` was allocated once and reused across the power
    /// loop while `column_sum_kernel` at the time combined blocks via
    /// `atomicAdd` (accumulating into the reused buffer), so the mean correction
    /// `Z[v,j] -= μ[v]·Σ_r Q[r,j]` was inflated on iteration ≥2 and on the final
    /// `B`, rotating the returned basis. (The kernel was later reworked under
    /// finding G2 to a single-block plain-store reduction, which also removes
    /// this reuse hazard structurally.) The sibling
    /// `test_resident_capture_vs_direct_subspace` compares two GPU paths that
    /// share the buggy code and cannot catch this; here we compare the GPU
    /// device-resident PCA (default config: `zero_center = true`,
    /// `n_power_iterations = 2`) against an exact CPU SVD reference on mean-heavy,
    /// low-rank data. Pre-fix the top subspace is rotated and this fails;
    /// post-fix it passes.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_resident_pca_matches_cpu_reference() {
        let dev = require_gpu!();
        // A small matrix suffices: G1 was cross-*iteration* accumulation into a
        // reused `d_sum_q` (buffer reused across the 2 power iterations + final
        // B), not within-iteration multi-block atomicAdd — so it fires even at
        // n_obs < 256 where only one column-sum block would ever launch.
        let (n_obs, n_vars, n_shards) = (300usize, 40usize, 3usize);
        let n_components = 4usize;
        let source = low_rank_source_with_offset(n_obs, n_vars, n_shards, 2024);

        set_cuda_graphs_enabled_override(None);
        let gpu = gpu_randomized_pca(
            &dev,
            &source,
            n_components,
            10,   // n_oversamples
            2,    // n_power_iterations (accelerator default)
            true, // zero_center (accelerator default)
            2024,
            QrMethod::Householder,
            GpuPcaTuning::default(),
        )
        .unwrap();
        assert!(
            gpu.mean.is_some(),
            "zero_center=true must record column means"
        );

        // Exact CPU reference: column-center, then thin SVD via faer (f64).
        let dense = materialize_dense(&source);
        let mut means = vec![0f64; n_vars];
        for i in 0..n_obs {
            for j in 0..n_vars {
                means[j] += dense[i * n_vars + j];
            }
        }
        for m in &mut means {
            *m /= n_obs as f64;
        }
        let mut c_mat = faer::Mat::<f64>::zeros(n_obs, n_vars);
        for i in 0..n_obs {
            for j in 0..n_vars {
                c_mat[(i, j)] = dense[i * n_vars + j] - means[j];
            }
        }
        let svd = c_mat.thin_svd().expect("CPU SVD");
        let v = svd.V().to_owned(); // (n_vars × min) right singular vectors

        // CPU embeddings E = C · V[:, :n_components], row-major (n_obs × n_components).
        let mut cpu_emb = vec![0f32; n_obs * n_components];
        for i in 0..n_obs {
            for pc in 0..n_components {
                let mut acc = 0f64;
                for j in 0..n_vars {
                    acc += (dense[i * n_vars + j] - means[j]) * v[(j, pc)];
                }
                cpu_emb[i * n_components + pc] = acc as f32;
            }
        }

        // Compare the top-2 (well-separated signal) PCs by sign-free cosine.
        // Randomized PCA with 2 power iterations recovers this subspace to
        // ~1.0; the mean bug rotates it well below the 0.98 bar.
        for pc in 0..2 {
            let cos = abs_cosine(&gpu.embeddings, &cpu_emb, n_obs, n_components, pc);
            assert!(
                cos > 0.98,
                "PC {pc}: GPU-vs-CPU |cosine| = {cos} (mean-correction regression?)"
            );
        }
    }

    /// Build a col-major `(m × k)` matrix as a flat `Vec<f32>` (element `(r, c)`
    /// at index `c * m + r`) with non-dyadic fractional values, plus the exact
    /// per-column f64 reference sums. The magnitude/count push the running sum
    /// into the range where f32 accumulation drifts.
    fn colmajor_fixture(m: usize, k: usize) -> (Vec<f32>, Vec<f64>) {
        let mut data = vec![0f32; m * k];
        let mut refs = vec![0f64; k];
        for c in 0..k {
            let mut acc = 0f64;
            for r in 0..m {
                // Non-power-of-two fractions so f32 partial sums lose low bits.
                let val = 1.0 + ((r % 7) as f32) * 0.1 + (c as f32) * 0.01;
                data[c * m + r] = val;
                acc += val as f64;
            }
            refs[c] = acc;
        }
        (data, refs)
    }

    /// Regression for finding G2: `column_sum_kernel` must accumulate in f64 and
    /// be deterministic (single-block, no cross-block atomicAdd). Asserts (1)
    /// GPU column sums match an exact f64 CPU reference within a tight relative
    /// tolerance the old f32 cross-block atomicAdd misses at this scale, (2) two
    /// runs are bit-identical, and (3) writing into a reused buffer fully
    /// overwrites it (also guards the G1 concern).
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_column_sum_kernel_f64_deterministic() {
        let dev = require_gpu!();
        let (m, k) = (262_144usize, 3usize);
        let (host_a, refs_a) = colmajor_fixture(m, k);
        let d_a = dev.htod_copy(&host_a).unwrap();

        // (1) Precision vs exact f64 reference.
        let mut out_a = dev.alloc_zeros::<f32>(k).unwrap();
        gpu_column_sums_into(&dev, &d_a, &mut out_a, m, k).unwrap();
        let sums_a = dev.dtoh_copy(&out_a).unwrap();
        for c in 0..k {
            let rel = (sums_a[c] as f64 - refs_a[c]).abs() / refs_a[c].abs().max(1e-12);
            assert!(
                rel < 1e-6,
                "col {c}: GPU sum {} vs f64 ref {} (rel {rel:.2e})",
                sums_a[c],
                refs_a[c]
            );
        }

        // (2) Determinism: identical input → bit-identical output across runs.
        let mut out_a2 = dev.alloc_zeros::<f32>(k).unwrap();
        gpu_column_sums_into(&dev, &d_a, &mut out_a2, m, k).unwrap();
        let sums_a2 = dev.dtoh_copy(&out_a2).unwrap();
        assert_eq!(
            sums_a, sums_a2,
            "column sums must be deterministic run-to-run"
        );

        // (3) Reuse/overwrite: a second input into the SAME buffer must equal a
        // fresh reduction of that input (no leakage from the first call).
        let (host_b, _refs_b) = colmajor_fixture(m, k);
        // Perturb B so it differs from A.
        let host_b: Vec<f32> = host_b.iter().map(|v| v + 0.5).collect();
        let d_b = dev.htod_copy(&host_b).unwrap();
        let mut reused = dev.alloc_zeros::<f32>(k).unwrap();
        gpu_column_sums_into(&dev, &d_a, &mut reused, m, k).unwrap();
        gpu_column_sums_into(&dev, &d_b, &mut reused, m, k).unwrap();
        let reused_host = dev.dtoh_copy(&reused).unwrap();
        let mut out_b = dev.alloc_zeros::<f32>(k).unwrap();
        gpu_column_sums_into(&dev, &d_b, &mut out_b, m, k).unwrap();
        let fresh_b = dev.dtoh_copy(&out_b).unwrap();
        assert_eq!(
            reused_host, fresh_b,
            "reused buffer must be fully overwritten by the second call"
        );
    }
}
