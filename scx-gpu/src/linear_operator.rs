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
//! All routines stream shards via [`DoubleBufferedShardLoader`], reuse the
//! existing column-major helpers from `gpu_pca.rs`, and — critically for
//! later phases — compute the mean-correction pre-factor `mc = Vᵀ · μ` via
//! cuBLAS `sgemv` on the GPU, avoiding the D→H round-trip at
//! `gpu_pca.rs:326-327` in the current randomized-PCA implementation.

use cudarc::cublas::sys as cbs;
use cudarc::driver::safe::CudaSlice;
use scx_format::ShardSource;

use crate::cublas::{gpu_sgemm, gpu_sgemv, gpu_sger, CublasHandle};
use crate::cusparse::{spmm_csr, spmm_csr_transpose, CusparseHandle};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_pca::{
    gpu_column_sums, gpu_gather_colmajor, gpu_mean_correct_colmajor, gpu_outer_sub,
    gpu_scatter_colmajor,
};
use crate::shard_pipeline::DoubleBufferedShardLoader;
use crate::sparse_dense::sparse_to_dense_gpu;

/// Implicit-centering sparse operator over a `&dyn ShardSource`.
///
/// `d_means` is `None` when `zero_center == false`; all three ops then
/// reduce to their non-centered equivalents (`X · V`, `Xᵀ · Y`, `Xᵀ X`).
pub struct CenteredSparseOperator<'a> {
    dev: &'a GpuDevice,
    cusparse: &'a CusparseHandle,
    cublas: &'a CublasHandle,
    source: &'a (dyn ShardSource + Sync),
    d_means: Option<&'a CudaSlice<f32>>,
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
    ) -> Self {
        Self {
            dev,
            cusparse,
            cublas,
            source,
            d_means,
        }
    }

    /// `out (n_obs × k, col-major) = (X − μ) · V`.
    ///
    /// `V` is col-major `(n_vars × k)`. `d_out` is overwritten (no accumulate).
    pub fn matmat(
        &self,
        d_v: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
        k: usize,
    ) -> Result<(), GpuError> {
        let (n_obs, n_vars) = self.source.shape();

        // Zero `d_out` (scatter_colmajor writes but doesn't zero untouched entries
        // if any shard spans < n_obs — we also zero to be explicit).
        self.dev
            .stream()
            .memset_zeros(d_out)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("matmat: zero out: {e}")))?;

        // Precompute `mc = Vᵀ · μ` on GPU (length k) — cuBLAS sgemv replaces
        // the D→H round-trip used in the current `streaming_gpu_spmm_forward`.
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

        let loader = DoubleBufferedShardLoader::new(self.dev, self.source)?;
        let mut global_row = 0usize;

        loader.for_each_shard(|_idx, gpu_csr| {
            let shard_rows = gpu_csr.shape.0;
            if shard_rows == 0 {
                return Ok(());
            }

            let a_desc = gpu_csr.to_cusparse_csr(self.dev, self.dev.stream())?;
            let mut d_y_shard = self.dev.alloc_zeros::<f32>(shard_rows * k)?;

            // Y_shard = A · V   (β = 0 → overwrite)
            spmm_csr(
                self.cusparse,
                self.dev.stream(),
                self.dev,
                &a_desc,
                d_v,
                &mut d_y_shard,
                shard_rows,
                n_vars,
                k,
                1.0,
                0.0,
            )?;

            if let Some(ref mc) = d_mc {
                gpu_mean_correct_colmajor(self.dev, &mut d_y_shard, mc, shard_rows, k)?;
            }

            gpu_scatter_colmajor(
                self.dev, &d_y_shard, d_out, shard_rows, k, global_row, n_obs,
            )?;
            global_row += shard_rows;
            Ok(())
        })?;

        Ok(())
    }

    /// `out (n_vars × k, col-major) = (X − μ)ᵀ · Y`.
    ///
    /// `Y` is col-major `(n_obs × k)`. `d_out` is overwritten.
    pub fn rmatmat(
        &self,
        d_y: &CudaSlice<f32>,
        d_out: &mut CudaSlice<f32>,
        k: usize,
    ) -> Result<(), GpuError> {
        let (n_obs, n_vars) = self.source.shape();

        // Zero `d_out` — the per-shard SpMM uses β = 1 to accumulate.
        self.dev
            .stream()
            .memset_zeros(d_out)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("rmatmat: zero out: {e}")))?;

        let loader = DoubleBufferedShardLoader::new(self.dev, self.source)?;
        let mut global_row = 0usize;

        loader.for_each_shard(|_idx, gpu_csr| {
            let shard_rows = gpu_csr.shape.0;
            if shard_rows == 0 {
                return Ok(());
            }

            let a_desc = gpu_csr.to_cusparse_csr(self.dev, self.dev.stream())?;
            let d_y_shard = gpu_gather_colmajor(self.dev, d_y, shard_rows, k, global_row, n_obs)?;

            // out += Aᵀ · Y_shard   (β = 1 → accumulate)
            spmm_csr_transpose(
                self.cusparse,
                self.dev.stream(),
                self.dev,
                &a_desc,
                &d_y_shard,
                d_out,
                shard_rows,
                n_vars,
                k,
                1.0,
                1.0,
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
    /// Per-shard memory: one dense scratch buffer of `shard_rows × n_vars × 4 B`.
    /// Reusing the scratch across shards is a Phase 2 optimization; for now
    /// we pay the small allocator cost for simplicity.
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

        let loader = DoubleBufferedShardLoader::new(self.dev, self.source)?;
        loader.for_each_shard(|_idx, gpu_csr| {
            let shard_rows = gpu_csr.shape.0;
            if shard_rows == 0 {
                return Ok(());
            }
            let d_dense = sparse_to_dense_gpu(self.dev, gpu_csr, None, n_vars)?;
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
        fn read_shard(&self, shard_idx: usize) -> scx_format::Result<ScxCsr> {
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

        let op = CenteredSparseOperator::new(&dev, &cusparse, &cublas, &source, Some(&d_mu));
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

        let op = CenteredSparseOperator::new(&dev, &cusparse, &cublas, &source, None);
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

        let op = CenteredSparseOperator::new(&dev, &cusparse, &cublas, &source, Some(&d_mu));
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
        let op = CenteredSparseOperator::new(&dev, &cusparse, &cublas, &source, Some(&d_mu));
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

        let op = CenteredSparseOperator::new(&dev, &cusparse, &cublas, &source, None);
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
}
