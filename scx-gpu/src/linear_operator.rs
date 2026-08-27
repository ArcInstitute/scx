//! Implicit-centering sparse operator over a streamed `ShardSource`.
//!
//! [`StreamingPcaOperator`] is the [`PcaOperator`] for inputs whose CSR does not
//! fit device memory: each multiply drives the source shard by shard, and each
//! shard is one call to the same segment function
//! ([`spmm_forward_segment`] / [`spmm_transpose_segment`]) the device-resident
//! operator calls once. Neither file owns a copy of the centered arithmetic and
//! neither owns a power loop — [`run_power_loop`] owns the sequence.
//!
//! # The source is built once, not once per multiply
//!
//! Before ORG-8.20-2 this file constructed a fresh `RawGpuShardSource` inside
//! every `matmat` / `rmatmat`. A default two-power-iteration PCA runs seven
//! multiplies, so it built the pinned ring, the copy stream and the event pair
//! seven times — and, because `ValidationMemo` lives on the adapter, re-ran the
//! O(nnz) sortedness scan on all seven drives instead of the first. Holding one
//! [`BackedGpuMatrixSource`] for the operator's lifetime collapses both to one.
//!
//! What has **not** changed: the matrix is still decoded and re-uploaded on
//! every multiply. Residency is what avoids that, and it is the other operator.
//!
//! # Validation
//!
//! `ValidationChecks::SORTED`, attributed to `"pca"`. cuSPARSE SpMM is undefined
//! on unsorted CSR column indices, so sortedness is a precondition here rather
//! than a preference — and `in_range` alone would be a no-op on this layout
//! (the row-major validator does not range-check minor indices), leaving the
//! SpMM entirely unguarded. Finiteness is deliberately **not** requested: this
//! path propagates a NaN exactly as the CPU one does.

use cudarc::cusparse::sys as csp;
use cudarc::driver::safe::CudaSlice;
use scx_format_io::ShardSource;

use crate::backed_gpu_matrix_source::BackedGpuMatrixSource;
use crate::cusolver::QrMethod;
use crate::cusparse::CuSparseWorkspacePool;
use crate::error::GpuError;
use crate::gpu_matrix_source::{GpuMatrixSource, ValidationChecks, ValidationPolicy};
use crate::gpu_pca::{
    forward_mean_prefactor, qr_swap_into, spmm_forward_segment, spmm_transpose_segment,
    transpose_centering_correction, GpuPcaScratch, PcaHandles, PcaMultiplyCtx, RowSegment,
};
use crate::math_policy::SpmmAlgPolicy;
use crate::pca_operator::{ForwardOperand, PcaBuf, PcaOperator};

/// [`PcaOperator`] that streams the matrix from a [`ShardSource`], one segment
/// per CSR shard.
///
/// `d_means` is `None` when `zero_center == false`; both multiplies then reduce
/// to their uncentered equivalents.
///
/// `spmm_policy` is held as the *policy*, not the resolved cuSPARSE enum, so the
/// operator carries the caller's intent — but it is resolved once, into
/// [`PcaMultiplyCtx::alg`], and that is the only value any launch reads. Review
/// §8.11 was this path hardcoding `CUSPARSE_SPMM_ALG_DEFAULT` while
/// `uns["scx_accel"]["pca"]["spmm_policy"]` reported whatever the caller asked
/// for; there is now one field for it to be wrong in, shared with the resident
/// operator.
pub struct StreamingPcaOperator<'a> {
    ctx: PcaMultiplyCtx<'a>,
    handles: PcaHandles<'a>,
    qr_method: QrMethod,
    spmm_policy: SpmmAlgPolicy,
    source: BackedGpuMatrixSource<'a>,
    d_means: Option<&'a CudaSlice<f32>>,
    d_omega: &'a CudaSlice<f32>,
    scratch: &'a mut GpuPcaScratch,
    d_mc: CudaSlice<f32>,
    d_sum_q: CudaSlice<f32>,
    pool: CuSparseWorkspacePool,
}

impl<'a> StreamingPcaOperator<'a> {
    /// Build the operator, taking the source's staging adapter with it.
    ///
    /// `d_omega` is the `(n_vars × k)` seed matrix the first forward multiply
    /// reads; `scratch` supplies the `Y` and `Z` buffers the loop rewrites. `k`
    /// is passed rather than inferred from `scratch` — the buffers' lengths
    /// happen to determine it, and a constructor that quietly re-derives a
    /// caller's parameter is one shape change away from being silently wrong.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        handles: PcaHandles<'a>,
        source: &'a (dyn ShardSource + Sync),
        d_means: Option<&'a CudaSlice<f32>>,
        d_omega: &'a CudaSlice<f32>,
        scratch: &'a mut GpuPcaScratch,
        k: usize,
        qr_method: QrMethod,
        spmm_policy: SpmmAlgPolicy,
    ) -> Result<Self, GpuError> {
        let (n_obs, n_vars) = source.shape();
        let source = BackedGpuMatrixSource::new(handles.dev, source)?
            .with_validation(ValidationPolicy::new(ValidationChecks::SORTED, "pca"));
        Ok(Self {
            ctx: PcaMultiplyCtx {
                dev: handles.dev,
                cusparse: handles.cusparse,
                n_obs,
                n_vars,
                k,
                alg: spmm_policy.to_alg(),
            },
            handles,
            qr_method,
            spmm_policy,
            source,
            d_means,
            d_omega,
            d_mc: handles.dev.alloc_zeros::<f32>(k)?,
            d_sum_q: handles.dev.alloc_zeros::<f32>(k)?,
            scratch,
            pool: CuSparseWorkspacePool::new(),
        })
    }

    /// The cuSPARSE SpMM algorithm this operator launches with — the resolution
    /// of the [`SpmmAlgPolicy`] it was constructed with.
    pub fn spmm_alg(&self) -> csp::cusparseSpMMAlg_t {
        debug_assert_eq!(self.spmm_policy.to_alg(), self.ctx.alg);
        self.ctx.alg
    }
}

impl PcaOperator for StreamingPcaOperator<'_> {
    fn matmat(&mut self, src: ForwardOperand) -> Result<(), GpuError> {
        let Self {
            ctx,
            handles,
            source,
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

        // Zero up front: each shard's SpMM writes only its own row window, so
        // any row no shard covers would otherwise keep the previous multiply's
        // values. (ORG-8.20-2 PR D decides whether a tiling check makes this
        // removable; until it is measured, it stays.)
        ctx.dev
            .stream()
            .memset_zeros(d_y)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("matmat: zero out: {e}")))?;
        if let Some(d_mu) = d_means {
            forward_mean_prefactor(ctx, handles.cublas, d_v, d_mu, d_mc)?;
        }
        let mc = d_means.is_some().then_some(&*d_mc);

        let mut offset = 0usize;
        source.for_each_gpu_csr_shard(&mut |_idx, slot| {
            let rows = slot.view().shape.0;
            if rows == 0 {
                return Ok(());
            }
            // Cached per slot: reused across power iterations while the slot's
            // pointers and shape are unchanged.
            let desc = slot.cached_sp_descr(ctx.dev, ctx.dev.stream())?;
            spmm_forward_segment(ctx, pool, desc, d_v, d_y, mc, RowSegment { offset, rows })?;
            offset += rows;
            Ok(())
        })
    }

    fn rmatmat(&mut self) -> Result<(), GpuError> {
        let Self {
            ctx,
            source,
            d_means,
            scratch,
            d_sum_q,
            pool,
            ..
        } = self;
        let GpuPcaScratch { d_y, d_z } = &mut **scratch;

        // Required: the segment function accumulates with β = 1 across shards.
        ctx.dev
            .stream()
            .memset_zeros(d_z)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("rmatmat: zero out: {e}")))?;

        let mut offset = 0usize;
        source.for_each_gpu_csr_shard(&mut |_idx, slot| {
            let rows = slot.view().shape.0;
            if rows == 0 {
                return Ok(());
            }
            let desc = slot.cached_sp_descr(ctx.dev, ctx.dev.stream())?;
            spmm_transpose_segment(ctx, pool, desc, d_y, d_z, RowSegment { offset, rows })?;
            offset += rows;
            Ok(())
        })?;

        // Once, after every shard: the correction needs Y's column sums over all
        // rows, so it cannot be folded into the per-shard segment.
        if let Some(d_mu) = d_means {
            transpose_centering_correction(ctx, d_y, d_z, d_mu, d_sum_q)?;
        }
        Ok(())
    }

    fn qr(&mut self, buf: PcaBuf) -> Result<(), GpuError> {
        let Self {
            ctx,
            handles,
            qr_method,
            scratch,
            ..
        } = self;
        let (slot, rows) = match buf {
            PcaBuf::Y => (&mut scratch.d_y, ctx.n_obs),
            PcaBuf::Z => (&mut scratch.d_z, ctx.n_vars),
        };
        qr_swap_into(
            handles.dev,
            handles.cusolver,
            handles.cublas,
            *qr_method,
            slot,
            rows,
            ctx.k,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cublas::CublasHandle;
    use crate::cusolver::CusolverHandle;
    use crate::cusparse::CusparseHandle;
    use crate::device::GpuDevice;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use scx_sparse::ScxCsr;

    // ---- In-memory ShardSource fixture ----

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

    /// Build a random CSR (sparse) with ~`density` fraction of nonzeros.
    fn random_csr(n_rows: usize, n_cols: usize, density: f32, seed: u64) -> ScxCsr {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        indptr.push(0);
        for _ in 0..n_rows {
            for c in 0..n_cols {
                if rng.gen_bool(density as f64) {
                    indices.push(c as i32);
                    data.push(rng.gen_range(-1.0..1.0));
                }
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data)
    }

    /// Split `csr` into `n_shards` row-shards (roughly equal rows each).
    fn split_into_shards(csr: &ScxCsr, n_shards: usize) -> Vec<ScxCsr> {
        let (n_rows, n_cols) = (csr.n_rows(), csr.n_cols());
        let rows_per = n_rows.div_ceil(n_shards);
        let mut out = Vec::new();
        let mut row_start = 0;
        while row_start < n_rows {
            let row_end = (row_start + rows_per).min(n_rows);
            let p0 = csr.indptr[row_start] as usize;
            let p1 = csr.indptr[row_end] as usize;
            let shard_indptr: Vec<i64> = csr.indptr[row_start..=row_end]
                .iter()
                .map(|&p| p - csr.indptr[row_start])
                .collect();
            let shard_indices = csr.indices[p0..p1].to_vec();
            let shard_data = csr.data[p0..p1].to_vec();
            out.push(ScxCsr::new_unchecked(
                (row_end - row_start, n_cols),
                shard_indptr,
                shard_indices,
                shard_data,
            ));
            row_start = row_end;
        }
        out
    }

    /// Densify CSR to row-major Vec<f32> (n_rows × n_cols).
    fn densify(csr: &ScxCsr) -> Vec<f32> {
        let (n_rows, n_cols) = (csr.n_rows(), csr.n_cols());
        let mut out = vec![0.0f32; n_rows * n_cols];
        for r in 0..n_rows {
            let p0 = csr.indptr[r] as usize;
            let p1 = csr.indptr[r + 1] as usize;
            for nz in p0..p1 {
                let c = csr.indices[nz] as usize;
                out[r * n_cols + c] = csr.data[nz];
            }
        }
        out
    }

    /// Per-column means of a row-major matrix (`n_rows × n_cols`).
    fn col_means(x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
        let mut means = vec![0.0f32; n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                means[c] += x[r * n_cols + c];
            }
        }
        for m in &mut means {
            *m /= n_rows as f32;
        }
        means
    }

    /// Relative Frobenius error between GPU and CPU results.
    fn rel_error(gpu: &[f32], cpu: &[f32]) -> f32 {
        let mut num = 0.0f32;
        let mut denom = 0.0f32;
        for (&g, &c) in gpu.iter().zip(cpu.iter()) {
            num += (g - c).powi(2);
            denom += c.powi(2);
        }
        (num.sqrt()) / (denom.sqrt().max(1e-12))
    }

    // ---- Rig ----

    /// The three handles every operator needs, owned so a test can hand out a
    /// [`PcaHandles`] without three `let` bindings per test.
    struct Rig {
        cusparse: CusparseHandle,
        cublas: CublasHandle,
        cusolver: CusolverHandle,
    }

    impl Rig {
        fn new() -> Self {
            Self {
                cusparse: CusparseHandle::new().unwrap(),
                cublas: CublasHandle::new().unwrap(),
                cusolver: CusolverHandle::new().unwrap(),
            }
        }

        fn handles<'a>(&'a self, dev: &'a GpuDevice) -> PcaHandles<'a> {
            PcaHandles {
                dev,
                cusparse: &self.cusparse,
                cublas: &self.cublas,
                cusolver: &self.cusolver,
            }
        }
    }

    /// `Y = (X − μ)·V` through the operator, returned host-side col-major.
    ///
    /// `V` is supplied as the operator's Ω, which is what the seed multiply
    /// reads — so this exercises exactly the call `run_power_loop` makes first.
    fn forward(
        dev: &GpuDevice,
        rig: &Rig,
        source: &(dyn ShardSource + Sync),
        d_means: Option<&CudaSlice<f32>>,
        v_host: &[f32],
        k: usize,
    ) -> Vec<f32> {
        let (n_obs, n_vars) = source.shape();
        let d_v = dev.htod_copy(v_host).unwrap();
        let mut scratch = GpuPcaScratch::new(dev, n_obs, n_vars, k).unwrap();
        {
            let mut op = StreamingPcaOperator::new(
                rig.handles(dev),
                source,
                d_means,
                &d_v,
                &mut scratch,
                k,
                QrMethod::Householder,
                SpmmAlgPolicy::Default,
            )
            .unwrap();
            op.matmat(ForwardOperand::Omega).unwrap();
        }
        dev.synchronize().unwrap();
        dev.dtoh_copy(&scratch.d_y).unwrap()
    }

    /// `Z = (X − μ)ᵀ·Y` through the operator, returned host-side col-major.
    ///
    /// `Y` is seeded straight into the scratch slot the operator reads, which is
    /// how the power loop supplies it — the transpose multiply has no operand
    /// parameter precisely because there is only ever one source for it.
    fn transpose(
        dev: &GpuDevice,
        rig: &Rig,
        source: &(dyn ShardSource + Sync),
        d_means: Option<&CudaSlice<f32>>,
        y_host: &[f32],
        k: usize,
    ) -> Vec<f32> {
        let (n_obs, n_vars) = source.shape();
        let d_omega = dev.alloc_zeros::<f32>(n_vars * k).unwrap();
        let mut scratch = GpuPcaScratch::new(dev, n_obs, n_vars, k).unwrap();
        scratch.d_y = dev.htod_copy(y_host).unwrap();
        {
            let mut op = StreamingPcaOperator::new(
                rig.handles(dev),
                source,
                d_means,
                &d_omega,
                &mut scratch,
                k,
                QrMethod::Householder,
                SpmmAlgPolicy::Default,
            )
            .unwrap();
            op.rmatmat().unwrap();
        }
        dev.synchronize().unwrap();
        dev.dtoh_copy(&scratch.d_z).unwrap()
    }

    /// CPU reference for `(X − μ)·V`, col-major `(n_rows × k)`.
    fn cpu_forward(
        x_dense: &[f32],
        means: Option<&[f32]>,
        v_host: &[f32],
        n_rows: usize,
        n_cols: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n_rows * k];
        for j in 0..k {
            for r in 0..n_rows {
                let mut acc = 0.0f32;
                for c in 0..n_cols {
                    let mu = means.map_or(0.0, |m| m[c]);
                    acc += (x_dense[r * n_cols + c] - mu) * v_host[j * n_cols + c];
                }
                out[j * n_rows + r] = acc;
            }
        }
        out
    }

    /// CPU reference for `(X − μ)ᵀ·Y`, col-major `(n_cols × k)`.
    fn cpu_transpose(
        x_dense: &[f32],
        means: &[f32],
        y_host: &[f32],
        n_rows: usize,
        n_cols: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n_cols * k];
        for j in 0..k {
            for c in 0..n_cols {
                let mut acc = 0.0f32;
                for r in 0..n_rows {
                    acc += (x_dense[r * n_cols + c] - means[c]) * y_host[j * n_rows + r];
                }
                out[j * n_cols + c] = acc;
            }
        }
        out
    }

    // ---- Numerical oracle ----
    //
    // These are the only CPU↔GPU comparison the streaming path has. They were
    // written against `CenteredSparseOperator::matmat` / `::rmatmat` and are
    // ported, not folded into the loop tests: a power-loop test compares a
    // composition, and would pass on two multiplies that are individually wrong
    // in cancelling ways.

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_matches_cpu_centered() {
        let dev = require_gpu!();
        let rig = Rig::new();
        let (n_rows, n_cols, k) = (500usize, 80usize, 8usize);
        let csr = random_csr(n_rows, n_cols, 0.1, 42);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let source = InMemorySource {
            shards: split_into_shards(&csr, 4),
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(7);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_mu = dev.htod_copy(&means).unwrap();

        let out_gpu = forward(&dev, &rig, &source, Some(&d_mu), &v_host, k);
        let out_cpu = cpu_forward(&x_dense, Some(&means), &v_host, n_rows, n_cols, k);
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "matmat centered rel_error = {err}");
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_matches_cpu_not_centered() {
        let dev = require_gpu!();
        let rig = Rig::new();
        let (n_rows, n_cols, k) = (300usize, 50usize, 6usize);
        let csr = random_csr(n_rows, n_cols, 0.08, 13);
        let x_dense = densify(&csr);
        let source = InMemorySource {
            shards: split_into_shards(&csr, 3),
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(11);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let out_gpu = forward(&dev, &rig, &source, None, &v_host, k);
        let out_cpu = cpu_forward(&x_dense, None, &v_host, n_rows, n_cols, k);
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "matmat uncentered rel_error = {err}");
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_rmatmat_matches_cpu_centered() {
        let dev = require_gpu!();
        let rig = Rig::new();
        let (n_rows, n_cols, k) = (400usize, 60usize, 7usize);
        let csr = random_csr(n_rows, n_cols, 0.1, 23);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let source = InMemorySource {
            shards: split_into_shards(&csr, 4),
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(31);
        let y_host: Vec<f32> = (0..n_rows * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_mu = dev.htod_copy(&means).unwrap();

        let out_gpu = transpose(&dev, &rig, &source, Some(&d_mu), &y_host, k);
        let out_cpu = cpu_transpose(&x_dense, &means, &y_host, n_rows, n_cols, k);
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "rmatmat centered rel_error = {err}");
    }

    /// G2 regression: each shard's SpMM writes directly into the output at row
    /// offset `RowSegment::offset` with `ld = n_obs`. Uneven shard sizes + odd
    /// `k` exercise the ld/offset arithmetic in
    /// `mean_correct_colmajor_strided_kernel` — any off-by-one would corrupt the
    /// last shard's rows or leak into untouched ones.
    ///
    /// n_rows = 503 (prime) split into 4 shards → 126+126+126+125. The last
    /// shard is undersized by 1 row, so its strided SpMM writes rows
    /// `[378, 503)` only and the mean-correct touches a shorter region.
    ///
    /// This is also the test that distinguishes the two operators: the resident
    /// one has a single segment at offset 0, so it cannot observe an offset bug
    /// at all.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_uneven_shards_centered() {
        let dev = require_gpu!();
        let rig = Rig::new();
        let (n_rows, n_cols, k) = (503usize, 73usize, 15usize); // k not a multiple of 32
        let csr = random_csr(n_rows, n_cols, 0.1, 1009);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let shards = split_into_shards(&csr, 4);
        assert!(
            shards.last().unwrap().n_rows() < shards.first().unwrap().n_rows(),
            "uneven-shard test requires a strictly smaller last shard; shard sizes = {:?}",
            shards.iter().map(|s| s.n_rows()).collect::<Vec<_>>()
        );
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(2027);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_mu = dev.htod_copy(&means).unwrap();

        let out_gpu = forward(&dev, &rig, &source, Some(&d_mu), &v_host, k);
        let out_cpu = cpu_forward(&x_dense, Some(&means), &v_host, n_rows, n_cols, k);
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "matmat uneven shards rel_error = {err}");
    }

    /// Mirror of the above for the transpose path. The strided B view reads `Y`
    /// at `RowSegment::offset` with `ld = n_obs`; an off-by-one would either
    /// read past the end of `Y` or skip rows.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_rmatmat_uneven_shards_centered() {
        let dev = require_gpu!();
        let rig = Rig::new();
        let (n_rows, n_cols, k) = (503usize, 73usize, 15usize);
        let csr = random_csr(n_rows, n_cols, 0.1, 4099);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let source = InMemorySource {
            shards: split_into_shards(&csr, 4),
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(8191);
        let y_host: Vec<f32> = (0..n_rows * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_mu = dev.htod_copy(&means).unwrap();

        let out_gpu = transpose(&dev, &rig, &source, Some(&d_mu), &y_host, k);
        let out_cpu = cpu_transpose(&x_dense, &means, &y_host, n_rows, n_cols, k);
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "rmatmat uneven shards rel_error = {err}");
    }

    // ---- Workspace pool ----

    /// G2 power-iteration pool reuse: alternating forward and transpose
    /// multiplies across `n_power_iters` iterations on one operator. Each
    /// iteration's shapes are constant, so the pool should grow at most twice
    /// (once per multiply direction) regardless of the iteration count.
    ///
    /// Driven through `matmat` / `rmatmat` rather than `run_power_loop` on
    /// purpose: this is a test of the SpMM workspace, and the loop would drag
    /// cuSOLVER QR into it for no added coverage.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_rmatmat_pool_no_realloc_across_power_iters() {
        let dev = require_gpu!();
        let rig = Rig::new();
        let (n_rows, n_cols, k) = (800usize, 120usize, 12usize);
        let n_power_iters = 8;
        let csr = random_csr(n_rows, n_cols, 0.1, 271);
        let means = col_means(&densify(&csr), n_rows, n_cols);
        let source = InMemorySource {
            shards: split_into_shards(&csr, 4),
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(2);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_v = dev.htod_copy(&v_host).unwrap();
        let d_mu = dev.htod_copy(&means).unwrap();
        let mut scratch = GpuPcaScratch::new(&dev, n_rows, n_cols, k).unwrap();

        let mut op = StreamingPcaOperator::new(
            rig.handles(&dev),
            &source,
            Some(&d_mu),
            &d_v,
            &mut scratch,
            k,
            QrMethod::Householder,
            SpmmAlgPolicy::Default,
        )
        .unwrap();

        op.matmat(ForwardOperand::Omega).unwrap();
        for _ in 0..n_power_iters {
            op.rmatmat().unwrap();
            op.matmat(ForwardOperand::Z).unwrap();
        }
        dev.synchronize().unwrap();

        let metrics = op.pool.metrics();
        // 1 + 2·n_power_iters = 17 multiplies, each over 4 shards = 68 SpMM
        // invocations of the pool.
        let expected_calls = (1 + 2 * n_power_iters) * 4;
        assert_eq!(
            metrics.alloc_count + metrics.reuse_count,
            expected_calls as u64,
            "pool counters should sum to total SpMM invocations"
        );
        assert!(
            metrics.alloc_count <= 2,
            "expected ≤ 2 grow events over {} power iters, got {} (capacity {} bytes)",
            n_power_iters,
            metrics.alloc_count,
            metrics.current_capacity_bytes,
        );
    }

    /// The pool is an operator field, so it is reused across calls as well as
    /// across the shards within one call: two same-shape forward multiplies over
    /// 4 shards should allocate at most once and reuse for the other 7.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_pooled_reuses_workspace_across_calls() {
        let dev = require_gpu!();
        let rig = Rig::new();
        let (n_rows, n_cols, k) = (400usize, 60usize, 8usize);
        let csr = random_csr(n_rows, n_cols, 0.1, 7);
        let source = InMemorySource {
            shards: split_into_shards(&csr, 4),
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(3);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_v = dev.htod_copy(&v_host).unwrap();
        let mut scratch = GpuPcaScratch::new(&dev, n_rows, n_cols, k).unwrap();

        let mut op = StreamingPcaOperator::new(
            rig.handles(&dev),
            &source,
            None,
            &d_v,
            &mut scratch,
            k,
            QrMethod::Householder,
            SpmmAlgPolicy::Default,
        )
        .unwrap();
        op.matmat(ForwardOperand::Omega).unwrap();
        op.matmat(ForwardOperand::Omega).unwrap();
        dev.synchronize().unwrap();

        let metrics = op.pool.metrics();
        assert!(
            metrics.alloc_count <= 1,
            "pool alloc_count = {} (capacity {} bytes); expected ≤ 1 across 8 same-shape SpMM calls",
            metrics.alloc_count,
            metrics.current_capacity_bytes
        );
        assert_eq!(
            metrics.alloc_count + metrics.reuse_count,
            8,
            "expected 8 with_workspace calls (2 matmat × 4 shards); got alloc={} reuse={}",
            metrics.alloc_count,
            metrics.reuse_count
        );
    }

    // ---- SpMM policy ----

    /// The streaming operator must launch the algorithm its `SpmmAlgPolicy`
    /// resolves to, on **both** multiplies. This is the path a `>VRAM` PCA
    /// takes; review §8.11 is it hardcoding `CUSPARSE_SPMM_ALG_DEFAULT` while
    /// `uns["scx_accel"]["pca"]["spmm_policy"]` reported the caller's request.
    /// (The policy pins an *algorithm*, not the bits — cuSPARSE guarantees no
    /// reproducibility for the transpose multiply this loop issues. See
    /// `SpmmAlgPolicy::Deterministic`.)
    ///
    /// Two things are checked, and it is worth being precise about which is
    /// which. The `spmm_alg()` assertion is a real regression guard: it fails if
    /// the field stops being wired to the policy. The numeric halves are
    /// **acceptance** checks — they catch `CUSPARSE_STATUS_NOT_SUPPORTED` for
    /// `CSR_ALG2` (notably under `CUSPARSE_OPERATION_TRANSPOSE`) and confirm the
    /// two algorithms agree. They would pass even if the operator ignored the
    /// policy entirely; nothing observable from the host reports which algorithm
    /// cuSPARSE actually ran.
    ///
    /// The structural guard is what carries the wiring, and it is narrower than
    /// "unbypassable": with no algorithm-less strided-SpMM entry point left, a
    /// call site can no longer *omit* the algorithm and inherit a hidden
    /// default. It can still pass `CUSPARSE_SPMM_ALG_DEFAULT` deliberately — the
    /// `cusparse.rs` tests do. The guard is against an invisible default, not
    /// against a wrong choice.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn streaming_operator_honours_the_spmm_policy() {
        let dev = require_gpu!();
        let rig = Rig::new();
        let (n_rows, n_cols, k) = (400usize, 70usize, 6usize);
        let csr = random_csr(n_rows, n_cols, 0.12, 91);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let source = InMemorySource {
            shards: split_into_shards(&csr, 4),
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(19);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let y_host: Vec<f32> = (0..n_rows * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_v = dev.htod_copy(&v_host).unwrap();
        let d_mu = dev.htod_copy(&means).unwrap();

        let mut forward_out = Vec::new();
        let mut transpose_out = Vec::new();
        for policy in [SpmmAlgPolicy::Default, SpmmAlgPolicy::Deterministic] {
            let mut scratch = GpuPcaScratch::new(&dev, n_rows, n_cols, k).unwrap();
            {
                let mut op = StreamingPcaOperator::new(
                    rig.handles(&dev),
                    &source,
                    Some(&d_mu),
                    &d_v,
                    &mut scratch,
                    k,
                    QrMethod::Householder,
                    policy,
                )
                .unwrap();
                assert_eq!(
                    op.spmm_alg(),
                    policy.to_alg(),
                    "operator launches {:?} but was constructed with {policy:?}",
                    op.spmm_alg()
                );
                op.matmat(ForwardOperand::Omega).unwrap();
            }
            dev.synchronize().unwrap();
            forward_out.push(dev.dtoh_copy(&scratch.d_y).unwrap());

            // A second operator for the transpose: `Y` has to be seeded into the
            // scratch the operator borrows, and the forward multiply above
            // overwrote it. `CSR_ALG2` under `CUSPARSE_OPERATION_TRANSPOSE` is
            // the combination this operator had never issued.
            transpose_out.push({
                let d_omega = dev.alloc_zeros::<f32>(n_cols * k).unwrap();
                let mut scratch = GpuPcaScratch::new(&dev, n_rows, n_cols, k).unwrap();
                scratch.d_y = dev.htod_copy(&y_host).unwrap();
                {
                    let mut op = StreamingPcaOperator::new(
                        rig.handles(&dev),
                        &source,
                        Some(&d_mu),
                        &d_omega,
                        &mut scratch,
                        k,
                        QrMethod::Householder,
                        policy,
                    )
                    .unwrap();
                    op.rmatmat().unwrap();
                }
                dev.synchronize().unwrap();
                dev.dtoh_copy(&scratch.d_z).unwrap()
            });
        }

        let fwd_err = rel_error(&forward_out[0], &forward_out[1]);
        assert!(
            fwd_err < 1e-4,
            "matmat under Deterministic diverged from Default: rel_error = {fwd_err}"
        );
        let rev_err = rel_error(&transpose_out[0], &transpose_out[1]);
        assert!(
            rev_err < 1e-4,
            "rmatmat under Deterministic diverged from Default: rel_error = {rev_err}"
        );
    }
}
