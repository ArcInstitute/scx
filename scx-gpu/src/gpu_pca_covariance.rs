//! GPU covariance PCA — the HVG-shaped (small `n_vars`) PCA path.
//!
//! Mirrors the CPU [`scx_accel::covariance_pca`] algorithm on GPU:
//!
//! 1. Streaming column means + `col_sum_sq` pass (CPU — small buffers).
//! 2. Upload means (if centering) to GPU as `d_means: CudaSlice<f32>`.
//! 3. [`CenteredSparseOperator::accumulate_gram`] — builds
//!    `G = (X − μ)ᵀ(X − μ)` as a dense `n_vars × n_vars` col-major matrix on GPU.
//! 4. [`gpu_eigh_sym`] — solves `G · V = V · diag(Λ)` in-place; `G` becomes `V`,
//!    eigenvalues are returned in ascending order.
//! 5. Pick the top `n_components` eigenpairs (last k in ascending order →
//!    reversed to descending), form `V_topk (n_vars × k)` on GPU, compute
//!    `σ = sqrt(max(λ, 0))`.
//! 6. Embeddings `U · Σ = (X − μ) · V_topk` via a second streaming pass over
//!    `CenteredSparseOperator::matmat`.
//! 7. D→H copy embeddings and loadings once at the end.
//!
//! This eliminates the randomized-PCA regression on 2K HVGs (the covariance
//! path is `O(n_vars^3)` for eigh but avoids iterative SpMM × QR, which is
//! where the GPU randomized path currently pays 0.9× vs CPU).

use cudarc::cublas::sys as cbs;
use cudarc::driver::safe::CudaSlice;
use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::PushKernelArg;

use scx_format::{total_variance_from_col_sq, ShardSource};

use crate::cublas::{gpu_sgemm, CublasHandle};
use crate::cusolver::{gpu_eigh_sym, CusolverHandle};
use crate::cusparse::CusparseHandle;
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_pca::GpuPcaResult;
use crate::linear_operator::CenteredSparseOperator;

// Reuse the existing column-major PTX module — it already hosts
// scatter/gather/mean_correct/column_sum/outer_sub. We add a new
// `select_top_eigvecs_desc_kernel` in `colmajor_ops.cu` and a simple
// `reverse_vec_kernel` for flipping eigenvalues ascending → descending.
const COLMAJOR_OPS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/colmajor_ops.ptx"));

/// Internal: format a `scx_format::ScxError` as [`GpuError::InvalidShard`].
fn format_scx_error(e: scx_format::ScxError) -> GpuError {
    GpuError::InvalidShard(format!("SCX read error: {e}"))
}

/// GPU-accelerated covariance PCA.
///
/// Streams shards twice — once to build the Gram matrix, once to project the
/// embeddings. Designed for `n_vars <= 8000` (HVG-selected data); callers
/// with larger `n_vars` should use [`crate::gpu_randomized_pca`] instead.
///
/// See module-level docs for algorithmic detail. `Sync` is required on `source`
/// so that internal `CenteredSparseOperator` / `DoubleBufferedShardLoader`
/// usages can borrow it across scoped worker threads.
pub fn gpu_covariance_pca(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    n_components: usize,
    zero_center: bool,
) -> Result<GpuPcaResult, GpuError> {
    let (n_obs, n_vars) = source.shape();

    if n_components == 0 || n_obs == 0 || n_vars == 0 {
        return Err(GpuError::ShapeMismatch {
            expected: "n_components > 0, n_obs > 0, n_vars > 0".into(),
            got: format!("n_components={n_components}, n_obs={n_obs}, n_vars={n_vars}"),
        });
    }
    if n_components > n_obs.min(n_vars) {
        return Err(GpuError::ShapeMismatch {
            expected: format!("n_components <= min(n_obs, n_vars) = {}", n_obs.min(n_vars)),
            got: format!("n_components = {n_components}"),
        });
    }

    // Pre-flight VRAM check — dominated by the Gram matrix, embeddings,
    // and one dense-shard scratch allocated inside accumulate_gram.
    {
        let max_shard_rows = source.max_shard_rows().map_err(format_scx_error)?.max(1);
        let gram_bytes = n_vars * n_vars * 4;
        let emb_bytes = n_obs * n_components * 4;
        let dense_scratch_bytes = max_shard_rows * n_vars * 4;
        let means_bytes = n_vars * 4;
        // cuSOLVER syevd workspace — approximate 3× the matrix. Actual size
        // returned by bufferSize() is usually smaller; we budget generously.
        let syevd_bytes = 3 * n_vars * n_vars * 4;
        let peak_bytes = gram_bytes + emb_bytes + dense_scratch_bytes + means_bytes + syevd_bytes;
        let peak_with_headroom = (peak_bytes as f64 * 1.1) as usize;
        let (free, _total) = dev.free_memory()?;
        if peak_with_headroom > free {
            return Err(GpuError::OutOfMemory(format!(
                "GPU covariance PCA requires ~{} MB but only {} MB free on device",
                peak_with_headroom / (1 << 20),
                free / (1 << 20)
            )));
        }
    }

    // Handles.
    let cusparse = CusparseHandle::new()?;
    let cublas = CublasHandle::new()?;
    let cusolver = CusolverHandle::new()?;

    // Step 1: column means + col_sum_sq (CPU, single streaming pass).
    let (means, col_sum_sq) = source
        .col_means_and_sum_sq(zero_center)
        .map_err(format_scx_error)?;

    // Step 2: upload means.
    let d_means: Option<CudaSlice<f32>> = means
        .as_ref()
        .map(|mu| {
            let mu_f32: Vec<f32> = mu.iter().map(|&v| v as f32).collect();
            dev.htod_copy(&mu_f32)
        })
        .transpose()?;

    // Step 3: Gram matrix on GPU (n_vars × n_vars col-major).
    let mut d_gram = dev.alloc_zeros::<f32>(n_vars * n_vars)?;
    {
        let op = CenteredSparseOperator::new(dev, &cusparse, &cublas, source, d_means.as_ref());
        op.accumulate_gram(&mut d_gram)?;
    }
    dev.synchronize()?;

    // Step 4: symmetric eigendecomposition. Overwrites `d_gram` with
    // eigenvectors (col-major n_vars × n_vars); eigenvalues ascending.
    let d_eigvals_asc = gpu_eigh_sym(&cusolver, dev.stream(), dev, &mut d_gram, n_vars)?;

    // Step 5: pick the top-k eigenpairs — the LAST k entries of the ascending
    // eigvals / eigvecs. Reverse order via a small kernel so loadings are
    // output in descending variance order (conventional PCA convention).
    let mut d_v_topk = dev.alloc_zeros::<f32>(n_vars * n_components)?;
    let mut d_eigvals_topk = dev.alloc_zeros::<f32>(n_components)?;
    select_top_eigvecs_desc(
        dev,
        &d_gram,
        &d_eigvals_asc,
        &mut d_v_topk,
        &mut d_eigvals_topk,
        n_vars,
        n_components,
    )?;

    // Step 6: embeddings `U · Σ = (X − μ) · V_topk`  — stream shards again.
    let mut d_embeddings = dev.alloc_zeros::<f32>(n_obs * n_components)?;
    {
        let op = CenteredSparseOperator::new(dev, &cusparse, &cublas, source, d_means.as_ref());
        op.matmat(&d_v_topk, &mut d_embeddings, n_components)?;
    }
    dev.synchronize()?;

    // Step 7: D→H copy results.
    let embeddings_colmajor = dev.dtoh_copy(&d_embeddings)?;
    let v_topk_host = dev.dtoh_copy(&d_v_topk)?;
    let eigvals_topk = dev.dtoh_copy(&d_eigvals_topk)?;

    // Embeddings: col-major (n_obs × n_components) → row-major for AnnData.
    let mut embeddings = vec![0.0f32; n_obs * n_components];
    for j in 0..n_components {
        for i in 0..n_obs {
            embeddings[i * n_components + j] = embeddings_colmajor[j * n_obs + i];
        }
    }

    // Components: V is col-major (n_vars × n_components); transpose to
    // row-major (n_components × n_vars) — the `GpuPcaResult` convention.
    let mut components = vec![0.0f32; n_components * n_vars];
    for pc in 0..n_components {
        for v in 0..n_vars {
            components[pc * n_vars + v] = v_topk_host[pc * n_vars + v];
        }
    }

    // Variance explained = λ / (n − 1). Clamp to ≥ 0 for numerical safety
    // (syevd may return tiny-negative eigenvalues on nearly-singular inputs).
    let denom = (n_obs as f64 - 1.0).max(1.0);
    let variance_explained: Vec<f64> = eigvals_topk
        .iter()
        .map(|&l| (l.max(0.0) as f64) / denom)
        .collect();

    let total_var = total_variance_from_col_sq(&col_sum_sq, means.as_deref(), n_obs);
    let variance_ratio: Vec<f64> = if total_var > 0.0 {
        variance_explained.iter().map(|&v| v / total_var).collect()
    } else {
        vec![0.0; n_components]
    };

    Ok(GpuPcaResult {
        embeddings,
        components,
        variance_explained,
        variance_ratio,
        mean: means,
        n_components,
        n_obs,
        n_vars,
    })
}

/// Select top-`k` eigenpairs from an ascending-ordered `eigvecs/eigvals` pair
/// and reorder them descending (largest-first).
///
/// Implemented as two cuBLAS `sgemm`-free column copies: we just reverse-copy
/// the last `k` columns of `eigvecs_asc` into `eigvecs_topk_desc`. Done via
/// a small CUDA kernel (`select_top_eigvecs_desc_kernel`) in `colmajor_ops.cu`.
/// Eigenvalues are reversed in a separate kernel call (`reverse_vec_kernel`).
fn select_top_eigvecs_desc(
    dev: &GpuDevice,
    eigvecs_asc: &CudaSlice<f32>,      // (n × n) col-major
    eigvals_asc: &CudaSlice<f32>,      // (n,)
    eigvecs_topk: &mut CudaSlice<f32>, // (n × k) col-major
    eigvals_topk: &mut CudaSlice<f32>, // (k,)
    n: usize,
    k: usize,
) -> Result<(), GpuError> {
    if k == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;

    // 1) Copy columns [n-k .. n) of eigvecs_asc into eigvecs_topk[:, 0..k]
    //    in REVERSE column order (so column 0 of output is the max-eigval
    //    eigenvector).
    {
        let func = module
            .load_function("select_top_eigvecs_desc_kernel")
            .map_err(|e| GpuError::KernelLaunchFailed(format!("select_top_eigvecs_desc: {e}")))?;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let total = n * k;
        let threads: u32 = 256;
        let blocks = (total as u32).div_ceil(threads);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(eigvecs_asc)
                .arg(eigvecs_topk)
                .arg(&n_i32)
                .arg(&k_i32)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("select_top_eigvecs_desc: {e}")))?;
    }

    // 2) Copy eigvals[n-k..n] into eigvals_topk reversed.
    {
        let func = module
            .load_function("reverse_tail_vec_kernel")
            .map_err(|e| GpuError::KernelLaunchFailed(format!("reverse_tail_vec: {e}")))?;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let threads: u32 = 256;
        let blocks = (k as u32).div_ceil(threads).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(eigvals_asc)
                .arg(eigvals_topk)
                .arg(&n_i32)
                .arg(&k_i32)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("reverse_tail_vec: {e}")))?;
    }
    Ok(())
}

// Suppress unused-import warnings in the CPU build where cublas ops aren't
// actually invoked (they are, via CenteredSparseOperator — but clippy may not
// see through the trait impl).
#[allow(dead_code)]
fn _unused_imports_guard(a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>) {
    let _ = (a, b, c);
    let _ = cbs::cublasOperation_t::CUBLAS_OP_N;
    let _: fn(_, _, _, _, _, _, _, _, _, _, _, _) -> _ = gpu_sgemm;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_randomized_pca;
    use faer::Mat;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
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
        fn read_shard(&self, shard_idx: usize) -> scx_format::Result<ScxCsr> {
            Ok(self.shards[shard_idx].clone())
        }
    }

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

    /// Sign-agnostic cosine similarity between two flat row-major matrices
    /// of shape (k × d). Returns the average absolute cosine over rows.
    fn row_abs_cosine(a: &[f32], b: &[f32], k: usize, d: usize) -> f32 {
        assert_eq!(a.len(), k * d);
        assert_eq!(b.len(), k * d);
        let mut total = 0.0f32;
        for i in 0..k {
            let row_a = &a[i * d..(i + 1) * d];
            let row_b = &b[i * d..(i + 1) * d];
            let dot: f32 = row_a.iter().zip(row_b).map(|(x, y)| x * y).sum();
            let na: f32 = row_a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = row_b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = (na * nb).max(1e-12);
            total += (dot / denom).abs();
        }
        total / k as f32
    }

    /// CPU covariance-PCA reference using faer directly on a dense version.
    /// Builds Gram = (X-μ)ᵀ(X-μ), runs eigh, and returns the top-k eigenvectors
    /// as row-major (k × n_cols). Used for parity testing inside scx-gpu
    /// (we can't depend on scx-accel here because scx-accel depends on us).
    fn cpu_cov_pca_components(csr: &ScxCsr, k: usize, zero_center: bool) -> Vec<f32> {
        let (n_rows, n_cols) = (csr.n_rows(), csr.n_cols());
        // Densify.
        let mut x = vec![0.0f64; n_rows * n_cols];
        for r in 0..n_rows {
            let p0 = csr.indptr[r] as usize;
            let p1 = csr.indptr[r + 1] as usize;
            for nz in p0..p1 {
                let c = csr.indices[nz] as usize;
                x[r * n_cols + c] = csr.data[nz] as f64;
            }
        }
        let mut means = vec![0.0f64; n_cols];
        if zero_center {
            for r in 0..n_rows {
                for c in 0..n_cols {
                    means[c] += x[r * n_cols + c];
                }
            }
            for m in &mut means {
                *m /= n_rows as f64;
            }
        }
        // Gram (col-major n_cols × n_cols).
        let mut gram = Mat::<f64>::zeros(n_cols, n_cols);
        for r in 0..n_rows {
            for a in 0..n_cols {
                let xa = x[r * n_cols + a] - means[a];
                for b in 0..n_cols {
                    let xb = x[r * n_cols + b] - means[b];
                    gram[(a, b)] += xa * xb;
                }
            }
        }
        let eigh = gram
            .self_adjoint_eigen(faer::Side::Upper)
            .expect("faer eigh");
        // faer returns eigenvalues/vectors in the natural order returned by
        // LAPACK divide-and-conquer: ascending. Take the last k columns and
        // reverse to descending.
        let eigvecs = eigh.U();
        let mut components = vec![0.0f32; k * n_cols];
        for j in 0..k {
            let src_col = n_cols - 1 - j;
            for v in 0..n_cols {
                components[j * n_cols + v] = eigvecs[(v, src_col)] as f32;
            }
        }
        components
    }

    #[test]
    fn test_gpu_covariance_vs_cpu_covariance() {
        let dev = require_gpu!();

        let n_rows = 500;
        let n_cols = 100;
        let k = 20;
        let csr = random_csr(n_rows, n_cols, 0.05, 42);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let gpu = gpu_covariance_pca(&dev, &source, k, true).unwrap();
        assert_eq!(gpu.components.len(), k * n_cols);
        assert_eq!(gpu.embeddings.len(), n_rows * k);

        let cpu_components = cpu_cov_pca_components(&csr, k, true);
        let cos = row_abs_cosine(&gpu.components, &cpu_components, k, n_cols);
        assert!(
            cos > 0.999,
            "GPU cov PCA vs CPU cov PCA: sign-agnostic cosine = {cos} (want > 0.999)"
        );
    }

    #[test]
    fn test_gpu_covariance_uncentered() {
        let dev = require_gpu!();

        let n_rows = 400;
        let n_cols = 80;
        let k = 10;
        let csr = random_csr(n_rows, n_cols, 0.07, 17);
        let shards = split_into_shards(&csr, 3);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let gpu = gpu_covariance_pca(&dev, &source, k, false).unwrap();
        let cpu_components = cpu_cov_pca_components(&csr, k, false);
        let cos = row_abs_cosine(&gpu.components, &cpu_components, k, n_cols);
        assert!(cos > 0.999, "GPU cov (uncentered) vs CPU: cosine = {cos}");
    }

    #[test]
    fn test_gpu_covariance_vs_gpu_randomized() {
        let dev = require_gpu!();

        // 2000 × 200 — spec-sized cross-method parity test. (2000×2000 is
        // slow in CI; 200 cols still exercises the full dispatch and keeps
        // the Gram matrix tractable.)
        let n_rows = 2000;
        let n_cols = 200;
        let k = 50;
        let csr = random_csr(n_rows, n_cols, 0.05, 7);
        let shards = split_into_shards(&csr, 8);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let cov = gpu_covariance_pca(&dev, &source, k, true).unwrap();
        let rand = gpu_randomized_pca(
            &dev,
            &source,
            k,
            10,
            4,
            true,
            123,
            crate::cusolver::QrMethod::default(),
        )
        .unwrap();

        let cos = row_abs_cosine(&cov.components, &rand.components, k, n_cols);
        assert!(
            cos > 0.99,
            "GPU cov PCA vs GPU randomized PCA: cosine = {cos} (want > 0.99)"
        );
    }
}
