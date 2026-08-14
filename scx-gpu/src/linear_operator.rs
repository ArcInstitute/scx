//! Implicit-centering sparse LinearOperator used by covariance and randomized PCA.
//!
//! [`CenteredSparseOperator`] exposes the three operations we need without
//! materializing the mean-centered matrix `(X − μ)`:
//!
//! * [`matmat`](CenteredSparseOperator::matmat) — `out = (X − μ) · V` for an
//!   `n_vars × k` right-hand side (the "forward" power-iteration step).
//! * [`rmatmat`](CenteredSparseOperator::rmatmat) — `out = (X − μ)ᵀ · Y` for
//!   an `n_obs × k` right-hand side (the "transpose" power-iteration step).
//! * [`accumulate_gram`](CenteredSparseOperator::accumulate_gram) — dense
//!   `n_vars × n_vars` Gram matrix `(X − μ)ᵀ (X − μ)` (the covariance-PCA
//!   accumulator).
//!
//! All routines stream shards via [`RawGpuShardSource`] (the G3 staging path,
//! with a cached cuSPARSE descriptor per reusable slot), reuse the
//! existing column-major helpers from `gpu_pca.rs`, and — critically for
//! later phases — compute the mean-correction pre-factor `mc = Vᵀ · μ` via
//! cuBLAS `sgemv` on the GPU, avoiding the D→H round-trip at
//! `gpu_pca.rs:326-327` in the current randomized-PCA implementation.

use cudarc::cublas::sys as cbs;
use cudarc::cusparse::sys as csp;
use cudarc::driver::safe::CudaSlice;
use scx_format_io::ShardSource;

use crate::cublas::{gpu_sgemm, gpu_sgemv, gpu_sger, CublasHandle};
use crate::cusparse::{
    spmm_csr_transpose_view_with_alg, spmm_csr_view_with_alg, CuSparseWorkspacePool,
    CusparseHandle, DnMatView, DnMatViewMut,
};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_pca::{gpu_column_sums, gpu_mean_correct_colmajor_strided, gpu_outer_sub};
use crate::gpu_shard_source::{GpuShardSource, RawGpuShardSource};
use crate::math_policy::SpmmAlgPolicy;
use crate::sparse_dense::sparse_to_dense_gpu_into_view;

/// Implicit-centering sparse operator over a `&dyn ShardSource`.
///
/// `d_means` is `None` when `zero_center == false`; all three ops then
/// reduce to their non-centered equivalents (`X · V`, `Xᵀ · Y`, `Xᵀ X`).
///
/// `spmm_policy` is the caller's [`SpmmAlgPolicy`] — held as the *policy*, not
/// the resolved cuSPARSE enum, so the operator carries the caller's intent and
/// resolves it at each launch. It applies to both [`Self::matmat`] and
/// [`Self::rmatmat`]; [`Self::accumulate_gram`] issues no SpMM and is
/// unaffected. Before this field existed the streaming PCA path hardcoded
/// `CUSPARSE_SPMM_ALG_DEFAULT` while the resident path honoured the policy, so
/// a `>VRAM` run reported `spmm_policy="deterministic"` having used atomics.
pub struct CenteredSparseOperator<'a> {
    dev: &'a GpuDevice,
    cusparse: &'a CusparseHandle,
    cublas: &'a CublasHandle,
    source: &'a (dyn ShardSource + Sync),
    d_means: Option<&'a CudaSlice<f32>>,
    spmm_policy: SpmmAlgPolicy,
}

impl<'a> CenteredSparseOperator<'a> {
    /// Construct from borrowed components. `d_means`, when provided, must
    /// have length `source.n_vars()`.
    pub fn new(
        dev: &'a GpuDevice,
        cusparse: &'a CusparseHandle,
        cublas: &'a CublasHandle,
        source: &'a (dyn ShardSource + Sync),
        d_means: Option<&'a CudaSlice<f32>>,
        spmm_policy: SpmmAlgPolicy,
    ) -> Self {
        Self {
            dev,
            cusparse,
            cublas,
            source,
            d_means,
            spmm_policy,
        }
    }

    /// The cuSPARSE SpMM algorithm this operator launches with — the resolution
    /// of the [`SpmmAlgPolicy`] it was constructed with.
    pub fn spmm_alg(&self) -> csp::cusparseSpMMAlg_t {
        self.spmm_policy.to_alg()
    }

    /// `out (n_obs × k, col-major) = (X − μ) · V`.
    ///
    /// `V` is col-major `(n_vars × k)`. `d_out` is overwritten (no accumulate).
    ///
    /// Each shard's SpMM writes directly into a strided view of `d_out` at row
    /// offset `global_row` with `ld = n_obs`, avoiding the per-shard
    /// `(shard_rows × k)` dense temporary that the pre-G2 implementation
    /// scatter-kernelled into `d_out`. A single-shot `CuSparseWorkspacePool`
    /// is constructed inside the call so the cuSPARSE SpMM workspace is
    /// reused across shards within one matmat (~N shards × one shape).
    /// Iterative callers (PCA power loop) should prefer
    /// [`Self::matmat_pooled`] so the pool also amortises across iterations.
    pub fn matmat(
        &self,
        d_v: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
        k: usize,
    ) -> Result<(), GpuError> {
        let mut pool = CuSparseWorkspacePool::new();
        self.matmat_pooled(d_v, d_out, k, &mut pool)
    }

    /// Pool-driven variant of [`Self::matmat`]. Reuses `pool`'s SpMM workspace
    /// across all shards in this call (and across calls if the caller threads
    /// the same pool through, e.g. PCA power iterations).
    pub fn matmat_pooled(
        &self,
        d_v: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
        k: usize,
        pool: &mut CuSparseWorkspacePool,
    ) -> Result<(), GpuError> {
        let (n_obs, n_vars) = self.source.shape();

        // Zero `d_out` — the strided SpMM writes only the populated row range
        // for each shard, so we must zero rows outside any shard up-front.
        self.dev
            .stream()
            .memset_zeros(d_out)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("matmat: zero out: {e}")))?;

        // Precompute `mc = Vᵀ · μ` on GPU (length k) — cuBLAS sgemv replaces
        // the D→H round-trip used in the legacy `streaming_gpu_spmm_forward`.
        let d_mc: Option<CudaSlice<f32>> = if let Some(d_mu) = self.d_means {
            let mut mc = self.dev.alloc_zeros::<f32>(k)?;
            gpu_sgemv(
                self.cublas,
                self.dev.stream(),
                d_v,
                d_mu,
                &mut mc,
                n_vars,
                k,
                1.0,
                0.0,
                cbs::cublasOperation_t::CUBLAS_OP_T,
            )?;
            Some(mc)
        } else {
            None
        };

        let mut src = RawGpuShardSource::new(self.dev, self.source)?;
        let mut global_row = 0usize;

        src.for_each_gpu_shard(|_idx, slot| {
            let shard_rows = slot.view().shape.0;
            if shard_rows == 0 {
                return Ok(());
            }

            // Get-or-build the cached cuSPARSE descriptor for this slot's live
            // shard (the descriptor is reused across power iterations when the
            // slot's pointers/shape are unchanged).
            let a_desc = slot.cached_sp_descr(self.dev, self.dev.stream())?;

            // V is col-major (n_vars × k); SpMM B operand stays contiguous.
            let b_view = DnMatView::contiguous(d_v, n_vars as i64, k as i64);
            // Strided view into d_out: write rows [global_row, global_row +
            // shard_rows) of the (n_obs × k) col-major output buffer. `ld =
            // n_obs` because that's the stride between columns in d_out.
            let c_view = DnMatViewMut {
                buf: d_out,
                offset_elems: global_row,
                rows: shard_rows as i64,
                cols: k as i64,
                ld: n_obs as i64,
            };
            // Y_shard = A · V   (β = 0 → overwrite the strided sub-region)
            spmm_csr_view_with_alg(
                self.cusparse,
                self.dev.stream(),
                self.dev,
                Some(pool),
                a_desc,
                b_view,
                c_view,
                1.0,
                0.0,
                self.spmm_policy.to_alg(),
            )?;

            if let Some(ref mc) = d_mc {
                gpu_mean_correct_colmajor_strided(
                    self.dev, d_out, mc, shard_rows, k, global_row, n_obs,
                )?;
            }

            global_row += shard_rows;
            Ok(())
        })?;

        Ok(())
    }

    /// `out (n_vars × k, col-major) = (X − μ)ᵀ · Y`.
    ///
    /// `Y` is col-major `(n_obs × k)`. `d_out` is overwritten.
    ///
    /// Each shard's transposed SpMM reads a strided view of `d_y` at row
    /// offset `global_row` with `ld = n_obs` and accumulates (β = 1) into the
    /// contiguous `(n_vars × k)` output. Mirrors the matmat scatter-removal
    /// for the transpose direction. Iterative callers should prefer
    /// [`Self::rmatmat_pooled`].
    pub fn rmatmat(
        &self,
        d_y: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
        k: usize,
    ) -> Result<(), GpuError> {
        let mut pool = CuSparseWorkspacePool::new();
        self.rmatmat_pooled(d_y, d_out, k, &mut pool)
    }

    /// Pool-driven variant of [`Self::rmatmat`]. See [`Self::matmat_pooled`].
    pub fn rmatmat_pooled(
        &self,
        d_y: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
        k: usize,
        pool: &mut CuSparseWorkspacePool,
    ) -> Result<(), GpuError> {
        let (n_obs, n_vars) = self.source.shape();

        // Zero `d_out` — the per-shard SpMM uses β = 1 to accumulate.
        self.dev
            .stream()
            .memset_zeros(d_out)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("rmatmat: zero out: {e}")))?;

        let mut src = RawGpuShardSource::new(self.dev, self.source)?;
        let mut global_row = 0usize;

        src.for_each_gpu_shard(|_idx, slot| {
            let shard_rows = slot.view().shape.0;
            if shard_rows == 0 {
                return Ok(());
            }

            let a_desc = slot.cached_sp_descr(self.dev, self.dev.stream())?;

            // Strided view into d_y: read rows [global_row, global_row +
            // shard_rows) of the (n_obs × k) col-major buffer. SpMM walks
            // columns with stride `ld = n_obs`.
            let b_view = DnMatView {
                buf: d_y,
                offset_elems: global_row,
                rows: shard_rows as i64,
                cols: k as i64,
                ld: n_obs as i64,
            };
            // out += Aᵀ · Y_shard   (β = 1 → accumulate into contiguous out)
            let c_view = DnMatViewMut::contiguous(d_out, n_vars as i64, k as i64);
            spmm_csr_transpose_view_with_alg(
                self.cusparse,
                self.dev.stream(),
                self.dev,
                Some(pool),
                a_desc,
                b_view,
                c_view,
                1.0,
                1.0,
                self.spmm_policy.to_alg(),
            )?;
            global_row += shard_rows;
            Ok(())
        })?;

        // Centering correction: out[v, j] −= μ[v] · (Σ_r Y[r, j]).
        if let Some(d_mu) = self.d_means {
            let d_sum_q = gpu_column_sums(self.dev, d_y, n_obs, k)?;
            gpu_outer_sub(self.dev, d_out, d_mu, &d_sum_q, n_vars, k)?;
        }

        Ok(())
    }

    /// `out (n_vars × n_vars, col-major) = (X − μ)ᵀ (X − μ)`.
    ///
    /// Per shard: densify `X_shard` (row-major `shard_rows × n_vars`) and
    /// accumulate `out += X_shardᵀ · X_shard` via cuBLAS `sgemm`. After the
    /// loop, if centered, apply the rank-1 correction `out −= n_obs · μ · μᵀ`
    /// via `sger`.
    ///
    /// A single `max_shard_rows × n_vars × 4 B` densification scratch is
    /// allocated once above the loop and reused across shards (Phase 10):
    /// the leading `shard_rows × n_vars` slice is zeroed before each
    /// `sparse_to_dense_gpu_into` call (the kernel only writes positions
    /// for nonzeros, so reuse without zeroing would carry over stale rows
    /// from the previous, possibly larger shard). `gpu_sgemm` receives
    /// `&d_dense` whose leading-dimension is `n_vars`; passing `k = shard_rows`
    /// makes cuBLAS ignore the rows past `shard_rows × n_vars`.
    pub fn accumulate_gram(&self, d_out: &mut CudaSlice<f32>) -> Result<(), GpuError> {
        let (n_obs, n_vars) = self.source.shape();

        // Row-major dense X_shard of shape (shard_rows × n_vars) has the same
        // byte layout as column-major (n_vars × shard_rows). We use it as A
        // with `trans_a = N` and as B with `trans_b = T`, so cuBLAS computes
        // A · Bᵀ = X_cm · X_cmᵀ = (X_rm)ᵀ · X_rm = Gram contribution.
        self.dev
            .stream()
            .memset_zeros(d_out)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("gram: zero out: {e}")))?;

        // Hoist the dense scratch to a single allocation sized for the
        // largest shard. `alloc_zeros` zero-initialises it; per-shard zeroing
        // happens inside the closure below.
        let max_shard_rows = self
            .source
            .max_shard_rows()
            .map_err(|e| GpuError::InvalidShard(format!("max_shard_rows: {e}")))?;
        let scratch_len = max_shard_rows.saturating_mul(n_vars);
        let mut d_dense = self.dev.alloc_zeros::<f32>(scratch_len)?;

        let mut src = RawGpuShardSource::new(self.dev, self.source)?;
        src.for_each_gpu_shard(|_idx, slot| {
            let view = slot.view();
            let shard_rows = view.shape.0;
            if shard_rows == 0 {
                return Ok(());
            }
            // Zero only the leading region we're about to write — the kernel
            // does not touch positions outside `[0, shard_rows * n_vars)`, but
            // it also does not write empty rows or absent (row, col) entries,
            // so any leftover values from the previous shard would corrupt the
            // Gram contribution.
            let used = shard_rows * n_vars;
            self.dev
                .stream()
                .memset_zeros(&mut d_dense.slice_mut(0..used))
                .map_err(|e| GpuError::KernelLaunchFailed(format!("gram: zero scratch: {e}")))?;
            sparse_to_dense_gpu_into_view(
                self.dev,
                &view.indptr,
                &view.indices,
                &view.data,
                shard_rows,
                None,
                n_vars,
                &mut d_dense,
            )?;
            gpu_sgemm(
                self.cublas,
                self.dev.stream(),
                &d_dense,
                &d_dense,
                d_out,
                n_vars,
                n_vars,
                shard_rows,
                1.0,
                1.0,
                cbs::cublasOperation_t::CUBLAS_OP_N,
                cbs::cublasOperation_t::CUBLAS_OP_T,
            )?;
            Ok(())
        })?;

        if let Some(d_mu) = self.d_means {
            gpu_sger(
                self.cublas,
                self.dev.stream(),
                d_mu,
                d_mu,
                d_out,
                n_vars,
                n_vars,
                -(n_obs as f32),
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_matches_cpu_centered() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 500;
        let n_cols = 80;
        let k = 8;
        let csr = random_csr(n_rows, n_cols, 0.1, 42);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        // Random V (n_cols × k), col-major.
        let mut rng = StdRng::seed_from_u64(7);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_v = dev.htod_copy(&v_host).unwrap();
        let d_mu = dev.htod_copy(&means).unwrap();

        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            Some(&d_mu),
            SpmmAlgPolicy::Default,
        );
        let mut d_out = dev.alloc_zeros::<f32>(n_rows * k).unwrap();
        op.matmat(&d_v, &mut d_out, k).unwrap();
        dev.synchronize().unwrap();
        let out_gpu = dev.dtoh_copy(&d_out).unwrap();

        // CPU ref: out (n_rows × k) col-major: out[r, j] = Σ_c (x[r,c] − μ[c]) * V[c, j].
        // V is col-major (n_cols × k): V[c, j] = v_host[j * n_cols + c].
        let mut out_cpu = vec![0.0f32; n_rows * k];
        for j in 0..k {
            for r in 0..n_rows {
                let mut acc = 0.0f32;
                for c in 0..n_cols {
                    acc += (x_dense[r * n_cols + c] - means[c]) * v_host[j * n_cols + c];
                }
                out_cpu[j * n_rows + r] = acc;
            }
        }
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "matmat centered rel_error = {err}");
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_matches_cpu_not_centered() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 300;
        let n_cols = 50;
        let k = 6;
        let csr = random_csr(n_rows, n_cols, 0.08, 13);
        let x_dense = densify(&csr);
        let shards = split_into_shards(&csr, 3);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(11);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_v = dev.htod_copy(&v_host).unwrap();

        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            None,
            SpmmAlgPolicy::Default,
        );
        let mut d_out = dev.alloc_zeros::<f32>(n_rows * k).unwrap();
        op.matmat(&d_v, &mut d_out, k).unwrap();
        dev.synchronize().unwrap();
        let out_gpu = dev.dtoh_copy(&d_out).unwrap();

        let mut out_cpu = vec![0.0f32; n_rows * k];
        for j in 0..k {
            for r in 0..n_rows {
                let mut acc = 0.0f32;
                for c in 0..n_cols {
                    acc += x_dense[r * n_cols + c] * v_host[j * n_cols + c];
                }
                out_cpu[j * n_rows + r] = acc;
            }
        }
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "matmat uncentered rel_error = {err}");
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_rmatmat_matches_cpu_centered() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 400;
        let n_cols = 60;
        let k = 7;
        let csr = random_csr(n_rows, n_cols, 0.1, 23);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(31);
        let y_host: Vec<f32> = (0..n_rows * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_y = dev.htod_copy(&y_host).unwrap();
        let d_mu = dev.htod_copy(&means).unwrap();

        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            Some(&d_mu),
            SpmmAlgPolicy::Default,
        );
        let mut d_out = dev.alloc_zeros::<f32>(n_cols * k).unwrap();
        op.rmatmat(&d_y, &mut d_out, k).unwrap();
        dev.synchronize().unwrap();
        let out_gpu = dev.dtoh_copy(&d_out).unwrap();

        // CPU ref: out (n_cols × k) col-major: out[c, j] = Σ_r (x[r,c] − μ[c]) * Y[r, j].
        // Y col-major (n_rows × k): Y[r, j] = y_host[j * n_rows + r].
        let mut out_cpu = vec![0.0f32; n_cols * k];
        for j in 0..k {
            for c in 0..n_cols {
                let mut acc = 0.0f32;
                for r in 0..n_rows {
                    acc += (x_dense[r * n_cols + c] - means[c]) * y_host[j * n_rows + r];
                }
                out_cpu[j * n_cols + c] = acc;
            }
        }
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "rmatmat centered rel_error = {err}");
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_accumulate_gram_matches_cpu_centered() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 250;
        let n_cols = 40;
        let csr = random_csr(n_rows, n_cols, 0.12, 5);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let shards = split_into_shards(&csr, 3);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let d_mu = dev.htod_copy(&means).unwrap();
        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            Some(&d_mu),
            SpmmAlgPolicy::Default,
        );
        let mut d_gram = dev.alloc_zeros::<f32>(n_cols * n_cols).unwrap();
        op.accumulate_gram(&mut d_gram).unwrap();
        dev.synchronize().unwrap();
        let gram_gpu = dev.dtoh_copy(&d_gram).unwrap();

        // CPU ref: Gram[a, b] = Σ_r (x[r, a] − μ[a]) (x[r, b] − μ[b]), col-major.
        let mut gram_cpu = vec![0.0f32; n_cols * n_cols];
        for b in 0..n_cols {
            for a in 0..n_cols {
                let mut acc = 0.0f32;
                for r in 0..n_rows {
                    acc +=
                        (x_dense[r * n_cols + a] - means[a]) * (x_dense[r * n_cols + b] - means[b]);
                }
                gram_cpu[b * n_cols + a] = acc;
            }
        }
        let err = rel_error(&gram_gpu, &gram_cpu);
        assert!(err < 1e-3, "accumulate_gram centered rel_error = {err}");
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_accumulate_gram_matches_cpu_not_centered() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 250;
        let n_cols = 40;
        let csr = random_csr(n_rows, n_cols, 0.12, 17);
        let x_dense = densify(&csr);
        let shards = split_into_shards(&csr, 3);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            None,
            SpmmAlgPolicy::Default,
        );
        let mut d_gram = dev.alloc_zeros::<f32>(n_cols * n_cols).unwrap();
        op.accumulate_gram(&mut d_gram).unwrap();
        dev.synchronize().unwrap();
        let gram_gpu = dev.dtoh_copy(&d_gram).unwrap();

        let mut gram_cpu = vec![0.0f32; n_cols * n_cols];
        for b in 0..n_cols {
            for a in 0..n_cols {
                let mut acc = 0.0f32;
                for r in 0..n_rows {
                    acc += x_dense[r * n_cols + a] * x_dense[r * n_cols + b];
                }
                gram_cpu[b * n_cols + a] = acc;
            }
        }
        let err = rel_error(&gram_gpu, &gram_cpu);
        assert!(err < 1e-3, "accumulate_gram uncentered rel_error = {err}");
    }

    /// Phase 10 regression: hoisting the densification scratch out of the
    /// shard loop means the same `d_dense` buffer is reused across shards.
    /// If the per-shard zero pass were ever dropped, nonzero positions
    /// written by shard *i* would survive into shard *(i+1)*'s leading
    /// region at coordinates the new shard happens to leave empty (no
    /// nonzero), and `gpu_sgemm` would fold that stale data into the Gram.
    ///
    /// Uses heterogeneous shard sizes (`221 / 4` → 56, 56, 56, 53 via
    /// ceil-div) and verifies *both* bit-identical determinism across
    /// runs *and* approximate equality against a CPU reference. The
    /// determinism check alone is insufficient — under a dropped
    /// zero-pass, both runs allocate fresh `d_dense` and corrupt
    /// identically, so `gram_a == gram_b` would still hold. The CPU
    /// reference is what actually catches the bug.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_accumulate_gram_scratch_reuse() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 221;
        let n_cols = 35;
        let csr = random_csr(n_rows, n_cols, 0.12, 99);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);

        let shards = split_into_shards(&csr, 4);
        // Sanity: the test premise depends on at least one shard boundary
        // being heterogeneous. If split_into_shards' rounding ever changes
        // and produces uniform shards on these dimensions, surface it here
        // rather than silently weakening the test.
        assert!(
            shards.windows(2).any(|w| w[0].n_rows() != w[1].n_rows()),
            "test precondition: shards must have at least one heterogeneous boundary"
        );
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let d_mu = dev.htod_copy(&means).unwrap();
        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            Some(&d_mu),
            SpmmAlgPolicy::Default,
        );

        let mut d_gram_a = dev.alloc_zeros::<f32>(n_cols * n_cols).unwrap();
        op.accumulate_gram(&mut d_gram_a).unwrap();
        dev.synchronize().unwrap();
        let gram_a = dev.dtoh_copy(&d_gram_a).unwrap();

        let mut d_gram_b = dev.alloc_zeros::<f32>(n_cols * n_cols).unwrap();
        op.accumulate_gram(&mut d_gram_b).unwrap();
        dev.synchronize().unwrap();
        let gram_b = dev.dtoh_copy(&d_gram_b).unwrap();

        assert_eq!(
            gram_a, gram_b,
            "accumulate_gram must be bit-identical across runs"
        );

        // CPU reference: Gram[a, b] = Σ_r (x[r, a] − μ[a]) (x[r, b] − μ[b]),
        // col-major. This is the check that actually catches a dropped
        // per-shard zero pass — determinism alone holds under that bug.
        let mut gram_cpu = vec![0.0f32; n_cols * n_cols];
        for b in 0..n_cols {
            for a in 0..n_cols {
                let mut acc = 0.0f32;
                for r in 0..n_rows {
                    acc +=
                        (x_dense[r * n_cols + a] - means[a]) * (x_dense[r * n_cols + b] - means[b]);
                }
                gram_cpu[b * n_cols + a] = acc;
            }
        }
        let err = rel_error(&gram_a, &gram_cpu);
        assert!(err < 1e-3, "scratch reuse rel_error vs CPU = {err}");
    }

    /// G2 regression: the strided SpMM path writes shard SpMM results
    /// directly into `d_out` at row offset `global_row` with `ld = n_obs`.
    /// Uneven shard sizes + odd `k` exercise the ld/offset arithmetic in
    /// `mean_correct_colmajor_strided_kernel` — any off-by-one would
    /// corrupt the last shard's rows or leak into untouched rows.
    ///
    /// n_rows = 503 (prime) split into 4 shards → 126+126+126+125. The
    /// last shard is undersized by 1 row, so its strided SpMM writes to
    /// rows `[378, 503)` only and the mean-correct touches a shorter
    /// `(shard_rows × k)` region than the others.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_uneven_shards_centered() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 503;
        let n_cols = 73;
        let k = 15; // not a multiple of 32 (warp size)
        let csr = random_csr(n_rows, n_cols, 0.1, 1009);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let shards = split_into_shards(&csr, 4);
        // Sanity: confirm the last shard is smaller than the others.
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
        let d_v = dev.htod_copy(&v_host).unwrap();
        let d_mu = dev.htod_copy(&means).unwrap();

        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            Some(&d_mu),
            SpmmAlgPolicy::Default,
        );
        let mut d_out = dev.alloc_zeros::<f32>(n_rows * k).unwrap();
        op.matmat(&d_v, &mut d_out, k).unwrap();
        dev.synchronize().unwrap();
        let out_gpu = dev.dtoh_copy(&d_out).unwrap();

        let mut out_cpu = vec![0.0f32; n_rows * k];
        for j in 0..k {
            for r in 0..n_rows {
                let mut acc = 0.0f32;
                for c in 0..n_cols {
                    acc += (x_dense[r * n_cols + c] - means[c]) * v_host[j * n_cols + c];
                }
                out_cpu[j * n_rows + r] = acc;
            }
        }
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "matmat uneven shards rel_error = {err}");
    }

    /// G2 regression mirror of `test_matmat_uneven_shards_centered` for the
    /// transpose path. Strided B view reads `d_y` at row offset `global_row`
    /// with `ld = n_obs`; an off-by-one in the view's offset/ld would either
    /// read past the end of `d_y` (silently zero or garbage) or skip rows.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_rmatmat_uneven_shards_centered() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 503;
        let n_cols = 73;
        let k = 15; // not a multiple of 32
        let csr = random_csr(n_rows, n_cols, 0.1, 4099);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(8191);
        let y_host: Vec<f32> = (0..n_rows * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_y = dev.htod_copy(&y_host).unwrap();
        let d_mu = dev.htod_copy(&means).unwrap();

        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            Some(&d_mu),
            SpmmAlgPolicy::Default,
        );
        let mut d_out = dev.alloc_zeros::<f32>(n_cols * k).unwrap();
        op.rmatmat(&d_y, &mut d_out, k).unwrap();
        dev.synchronize().unwrap();
        let out_gpu = dev.dtoh_copy(&d_out).unwrap();

        let mut out_cpu = vec![0.0f32; n_cols * k];
        for j in 0..k {
            for c in 0..n_cols {
                let mut acc = 0.0f32;
                for r in 0..n_rows {
                    acc += (x_dense[r * n_cols + c] - means[c]) * y_host[j * n_rows + r];
                }
                out_cpu[j * n_cols + c] = acc;
            }
        }
        let err = rel_error(&out_gpu, &out_cpu);
        assert!(err < 1e-4, "rmatmat uneven shards rel_error = {err}");
    }

    /// G2 power-iteration pool reuse: mimic the `gpu_randomized_pca` shape —
    /// alternating `matmat_pooled` (forward, n_obs × k output) and
    /// `rmatmat_pooled` (transpose, n_vars × k output) across `n_power_iters`
    /// iterations sharing one pool. Each iteration shape stays constant, so
    /// the pool's `alloc_count` should bump at most twice (once for the
    /// forward workspace shape, once for the transpose shape) regardless of
    /// the iteration count.
    ///
    /// This is the matmat-equivalent of the plan's
    /// `test_randomized_pca_pool_no_realloc_across_power_iters` — it
    /// exercises the same SpMM pattern without dragging in cuSOLVER QR.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_rmatmat_pool_no_realloc_across_power_iters() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 800;
        let n_cols = 120;
        let k = 12;
        let n_power_iters = 8;
        let csr = random_csr(n_rows, n_cols, 0.1, 271);
        let means = col_means(&densify(&csr), n_rows, n_cols);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(2);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_v = dev.htod_copy(&v_host).unwrap();
        let d_mu = dev.htod_copy(&means).unwrap();

        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            Some(&d_mu),
            SpmmAlgPolicy::Default,
        );
        let mut pool = CuSparseWorkspacePool::new();
        let mut d_y = dev.alloc_zeros::<f32>(n_rows * k).unwrap();
        let mut d_z = dev.alloc_zeros::<f32>(n_cols * k).unwrap();

        // First forward to populate d_y; then n_power_iters of (rmatmat into
        // d_z, matmat back into d_y) — exactly the SpMM pattern the GPU
        // randomized PCA power loop runs.
        op.matmat_pooled(&d_v, &mut d_y, k, &mut pool).unwrap();
        for _ in 0..n_power_iters {
            op.rmatmat_pooled(&d_y, &mut d_z, k, &mut pool).unwrap();
            op.matmat_pooled(&d_z, &mut d_y, k, &mut pool).unwrap();
        }
        dev.synchronize().unwrap();

        let metrics = pool.metrics();
        // Total SpMM calls: 1 + 2 * n_power_iters = 17. Each shard run = 4
        // SpMM calls, total = 17 × 4 = 68 SpMM invocations of the pool.
        let expected_calls = (1 + 2 * n_power_iters) * 4;
        assert_eq!(
            metrics.alloc_count + metrics.reuse_count,
            expected_calls as u64,
            "pool counters should sum to total SpMM invocations"
        );
        // The matmat and rmatmat workspace sizes may differ (different B/C
        // dimensions for forward vs transpose), so the pool can grow at
        // most twice. In practice on most hardware both fit the same
        // bucket and we see exactly 1 alloc.
        assert!(
            metrics.alloc_count <= 2,
            "expected ≤ 2 grow events over {} power iters, got {} (capacity {} bytes)",
            n_power_iters,
            metrics.alloc_count,
            metrics.current_capacity_bytes,
        );
    }

    /// G2 pool-threading: `matmat_pooled` with the caller's pool reuses
    /// workspace across the 4 shards; the second invocation reuses it
    /// across another 4 shards without re-allocating. Asserts that the
    /// pool's `alloc_count` is at most 1 after both calls (single grow
    /// event for cuSPARSE's largest workspace shape).
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_matmat_pooled_reuses_workspace_across_calls() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 400;
        let n_cols = 60;
        let k = 8;
        let csr = random_csr(n_rows, n_cols, 0.1, 7);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(3);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_v = dev.htod_copy(&v_host).unwrap();

        let op = CenteredSparseOperator::new(
            &dev,
            &cusparse,
            &cublas,
            &source,
            None,
            SpmmAlgPolicy::Default,
        );
        let mut pool = CuSparseWorkspacePool::new();
        let mut d_out = dev.alloc_zeros::<f32>(n_rows * k).unwrap();

        // Two back-to-back invocations on the same shape.
        op.matmat_pooled(&d_v, &mut d_out, k, &mut pool).unwrap();
        op.matmat_pooled(&d_v, &mut d_out, k, &mut pool).unwrap();
        dev.synchronize().unwrap();

        let metrics = pool.metrics();
        // Two calls × 4 shards = 8 SpMM invocations. Workspace shape is
        // the same for every shard (same V, same n_vars, same k), so the
        // pool should allocate at most once and reuse for the other 7.
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

    /// The streaming operator must launch the algorithm its `SpmmAlgPolicy`
    /// resolves to, on **both** multiplies.
    ///
    /// This is the path a `>VRAM` PCA takes. Before the policy was threaded in
    /// it hardcoded `CUSPARSE_SPMM_ALG_DEFAULT` while
    /// `uns["scx_accel"]["pca"]["spmm_policy"]` reported whatever the caller
    /// asked for, so a caller who requested determinism got atomics and was
    /// told otherwise.
    ///
    /// Two things are checked, and it is worth being precise about which is
    /// which. The `spmm_alg()` assertion is a real regression guard: it fails
    /// if the field stops being wired to the policy. The numeric halves are
    /// **acceptance** checks — they catch `CUSPARSE_STATUS_NOT_SUPPORTED` for
    /// `CSR_ALG2` (notably under `CUSPARSE_OPERATION_TRANSPOSE`, which the
    /// resident loop already relies on but this operator never exercised) and
    /// confirm the two algorithms agree. They would pass even if the operator
    /// ignored the policy entirely; nothing observable from the host reports
    /// which algorithm cuSPARSE actually ran. What makes the wiring
    /// unbypassable is structural, not this test: there is no algorithm-less
    /// strided-SpMM entry point left to call.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn streaming_operator_honours_the_spmm_policy() {
        let dev = require_gpu!();
        let cusparse = CusparseHandle::new().unwrap();
        let cublas = CublasHandle::new().unwrap();

        let n_rows = 400;
        let n_cols = 70;
        let k = 6;
        let csr = random_csr(n_rows, n_cols, 0.12, 91);
        let x_dense = densify(&csr);
        let means = col_means(&x_dense, n_rows, n_cols);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let mut rng = StdRng::seed_from_u64(19);
        let v_host: Vec<f32> = (0..n_cols * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let y_host: Vec<f32> = (0..n_rows * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let d_v = dev.htod_copy(&v_host).unwrap();
        let d_y = dev.htod_copy(&y_host).unwrap();
        let d_mu = dev.htod_copy(&means).unwrap();

        let mut forward = Vec::new();
        let mut transpose = Vec::new();
        for policy in [SpmmAlgPolicy::Default, SpmmAlgPolicy::Deterministic] {
            let op =
                CenteredSparseOperator::new(&dev, &cusparse, &cublas, &source, Some(&d_mu), policy);
            assert_eq!(
                op.spmm_alg(),
                policy.to_alg(),
                "operator launches {:?} but was constructed with {policy:?}",
                op.spmm_alg()
            );

            // Forward: (X − μ) · V, n_rows × k col-major.
            let mut d_fwd = dev.alloc_zeros::<f32>(n_rows * k).unwrap();
            op.matmat(&d_v, &mut d_fwd, k).unwrap();
            // Transpose: (X − μ)ᵀ · Y, n_cols × k col-major. `CSR_ALG2` under
            // CUSPARSE_OPERATION_TRANSPOSE is the combination this operator
            // had never issued.
            let mut d_rev = dev.alloc_zeros::<f32>(n_cols * k).unwrap();
            op.rmatmat(&d_y, &mut d_rev, k).unwrap();
            dev.synchronize().unwrap();

            forward.push(dev.dtoh_copy(&d_fwd).unwrap());
            transpose.push(dev.dtoh_copy(&d_rev).unwrap());
        }

        let fwd_err = rel_error(&forward[0], &forward[1]);
        assert!(
            fwd_err < 1e-4,
            "matmat under Deterministic diverged from Default: rel_error = {fwd_err}"
        );
        let rev_err = rel_error(&transpose[0], &transpose[1]);
        assert!(
            rev_err < 1e-4,
            "rmatmat under Deterministic diverged from Default: rel_error = {rev_err}"
        );
    }
}
