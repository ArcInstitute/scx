//! GPU pseudobulk NB-GLM fitter wrapper (Stage A).
//!
//! Thin host-side driver for `kernels/nb_glm.cu::nb_glm_fit_kernel`. The host
//! has already aggregated the small dense pseudobulk `[n_genes × n_sub]` (f64,
//! gene-major) and the shared design `[n_sub × p]`; this uploads them, launches
//! the thread-per-gene fitter, and downloads the per-gene fit. Two passes share
//! the one kernel:
//!
//!   * [`GpuNbGlmPass::Mle`] — moments init → IRLS → outer Cox-Reid alternation,
//!     returning the per-gene MLE dispersion + warm β.
//!   * [`GpuNbGlmPass::Shrink`] — recompute μ from the warm β, shrink dispersion
//!     against the empirical-Bayes prior, refit β once.
//!
//! Cross-gene steps (trend fit, prior variance, Wald, Cook's, filtering, BH)
//! stay on the host — see `scx-accel/src/nb_glm/gpu.rs`.

use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;

/// Compiled PTX for the NB-GLM kernel (produced by build.rs via nvcc --ptx).
const NB_GLM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/nb_glm.ptx"));

/// Maximum `n_features` (`p`) and `n_sub` the kernel supports — must match the
/// `NB_PMAX` / `NB_NSUB_MAX` `#define`s in `kernels/nb_glm.cu`. Larger inputs
/// fall back to the CPU path via the route planner.
pub const GPU_NB_GLM_PMAX: usize = 8;
pub const GPU_NB_GLM_NSUB_MAX: usize = 64;

/// Dispersion-method tags shared with the kernel (mirror `DispersionMethod`).
pub const GPU_NB_GLM_METHOD_MOMENTS: i32 = 0;
pub const GPU_NB_GLM_METHOD_CR_MLE: i32 = 1;
pub const GPU_NB_GLM_METHOD_CR_SHRUNK: i32 = 2;

/// Scalar fitter options forwarded to the kernel (mirror the relevant
/// `NbGlmOptions` fields).
#[derive(Debug, Clone, Copy)]
pub struct GpuNbGlmOpts {
    pub method: i32,
    pub max_irls_iters: i32,
    pub irls_tol: f64,
    pub min_mu: f64,
    pub eta_min: f64,
    pub eta_max: f64,
    pub beta_ridge: f64,
    pub min_disp: f64,
    pub max_disp: f64,
    pub disp_newton_iters: i32,
    pub max_outer_iters: i32,
    pub outer_tol: f64,
}

/// Which pass to run, with its pass-specific inputs.
pub enum GpuNbGlmPass<'a> {
    /// MLE pass: `base_mean[g]` seeds the intercept (`ln(max(base_mean, 1e-4))`).
    Mle { base_mean: &'a [f64] },
    /// Shrinkage pass: warm β + MLE dispersion + per-gene log prior target.
    Shrink {
        beta_in: &'a [f64],
        alpha_in: &'a [f64],
        prior_log_target: &'a [f64],
        prior_var: f64,
    },
}

/// Per-gene fit downloaded from the device.
#[derive(Debug, Clone)]
pub struct GpuNbGlmFit {
    /// `[n_genes × p]` row-major coefficients.
    pub beta: Vec<f64>,
    /// `[n_genes × n_sub]` row-major fitted means.
    pub mu: Vec<f64>,
    /// `[n_genes × p²]` row-major Fisher information (`XᵀWX + ridge·I`).
    pub fisher: Vec<f64>,
    /// `[n_genes]` dispersion (MLE or shrunken per pass).
    pub alpha: Vec<f64>,
    /// `[n_genes]` convergence flag (IRLS ∧ outer for MLE; IRLS for Shrink).
    pub converged: Vec<u8>,
    /// `[n_genes]` outer-iteration count (0 on the Shrink pass).
    pub n_iter: Vec<u32>,
    /// `[n_genes]` dispersion hit the lower clamp.
    pub at_low: Vec<u8>,
    /// `[n_genes]` dispersion hit the upper clamp.
    pub at_high: Vec<u8>,
}

/// Fit the per-gene NB-GLM on the GPU.
///
/// * `counts` — `[n_genes × n_sub]` row-major, non-negative finite pseudobulk.
/// * `design` — `[n_sub × p]` row-major.
/// * `log_sf` / `sf` — `[n_sub]` log-size-factors and size factors.
///
/// Returns a [`GpuNbGlmFit`]. Errors if `p > GPU_NB_GLM_PMAX` or
/// `n_sub > GPU_NB_GLM_NSUB_MAX` (caller should route those to CPU).
#[allow(clippy::too_many_arguments)]
pub fn gpu_nb_glm_fit(
    dev: &GpuDevice,
    counts: &[f64],
    design: &[f64],
    log_sf: &[f64],
    sf: &[f64],
    n_genes: usize,
    n_sub: usize,
    p: usize,
    opts: GpuNbGlmOpts,
    pass: GpuNbGlmPass<'_>,
) -> Result<GpuNbGlmFit, GpuError> {
    if p == 0 || p > GPU_NB_GLM_PMAX {
        return Err(GpuError::UnsupportedLayout(format!(
            "gpu_nb_glm_fit: p={p} out of range (1..={GPU_NB_GLM_PMAX})"
        )));
    }
    if n_sub == 0 || n_sub > GPU_NB_GLM_NSUB_MAX {
        return Err(GpuError::UnsupportedLayout(format!(
            "gpu_nb_glm_fit: n_sub={n_sub} out of range (1..={GPU_NB_GLM_NSUB_MAX})"
        )));
    }
    if counts.len() != n_genes * n_sub {
        return Err(GpuError::ShapeMismatch {
            expected: format!("counts n_genes*n_sub={}", n_genes * n_sub),
            got: format!("{}", counts.len()),
        });
    }
    if design.len() != n_sub * p || log_sf.len() != n_sub || sf.len() != n_sub {
        return Err(GpuError::ShapeMismatch {
            expected: format!("design {n_sub}*{p}, log_sf/sf {n_sub}"),
            got: format!(
                "design {}, log_sf {}, sf {}",
                design.len(),
                log_sf.len(),
                sf.len()
            ),
        });
    }
    if n_genes == 0 {
        return Ok(GpuNbGlmFit {
            beta: vec![],
            mu: vec![],
            fisher: vec![],
            alpha: vec![],
            converged: vec![],
            n_iter: vec![],
            at_low: vec![],
            at_high: vec![],
        });
    }

    let module = dev.load_module_cached(NB_GLM_PTX)?;
    let kernel = module
        .load_function("nb_glm_fit_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load nb_glm_fit_kernel: {e}")))?;

    // Shared inputs.
    let d_counts = dev.htod_copy(counts)?;
    let d_design = dev.htod_copy(design)?;
    let d_log_sf = dev.htod_copy(log_sf)?;
    let d_sf = dev.htod_copy(sf)?;

    // Pass-specific inputs. Unused pointers in a given mode are never
    // dereferenced by the kernel, but must be valid device pointers — each arm
    // allocates a 1-element placeholder buffer for the pointers it doesn't use.
    let (mode, prior_var, d_base_mean, d_beta_in, d_alpha_in, d_prior) = match pass {
        GpuNbGlmPass::Mle { base_mean } => {
            if base_mean.len() != n_genes {
                return Err(GpuError::ShapeMismatch {
                    expected: format!("base_mean {n_genes}"),
                    got: format!("{}", base_mean.len()),
                });
            }
            (
                0i32,
                0.0f64,
                dev.htod_copy(base_mean)?,
                dev.alloc_zeros::<f64>(1)?,
                dev.alloc_zeros::<f64>(1)?,
                dev.alloc_zeros::<f64>(1)?,
            )
        }
        GpuNbGlmPass::Shrink {
            beta_in,
            alpha_in,
            prior_log_target,
            prior_var,
        } => {
            if beta_in.len() != n_genes * p
                || alpha_in.len() != n_genes
                || prior_log_target.len() != n_genes
            {
                return Err(GpuError::ShapeMismatch {
                    expected: format!("beta_in {}, alpha_in/prior {n_genes}", n_genes * p),
                    got: format!(
                        "beta_in {}, alpha_in {}, prior {}",
                        beta_in.len(),
                        alpha_in.len(),
                        prior_log_target.len()
                    ),
                });
            }
            (
                1i32,
                prior_var,
                dev.alloc_zeros::<f64>(1)?,
                dev.htod_copy(beta_in)?,
                dev.htod_copy(alpha_in)?,
                dev.htod_copy(prior_log_target)?,
            )
        }
    };

    // Outputs.
    let mut d_beta = dev.alloc_zeros::<f64>(n_genes * p)?;
    let mut d_mu = dev.alloc_zeros::<f64>(n_genes * n_sub)?;
    let mut d_fisher = dev.alloc_zeros::<f64>(n_genes * p * p)?;
    let mut d_alpha = dev.alloc_zeros::<f64>(n_genes)?;
    let mut d_converged = dev.alloc_zeros::<u8>(n_genes)?;
    let mut d_niter = dev.alloc_zeros::<u32>(n_genes)?;
    let mut d_atlow = dev.alloc_zeros::<u8>(n_genes)?;
    let mut d_athigh = dev.alloc_zeros::<u8>(n_genes)?;

    // Scalar args (kernel takes `int` / `double`).
    let n_genes_i = n_genes as i32;
    let n_sub_i = n_sub as i32;
    let p_i = p as i32;
    let method = opts.method;
    let mode_i = mode;
    let max_irls = opts.max_irls_iters;
    let irls_tol = opts.irls_tol;
    let min_mu = opts.min_mu;
    let eta_min = opts.eta_min;
    let eta_max = opts.eta_max;
    let ridge = opts.beta_ridge;
    let min_disp = opts.min_disp;
    let max_disp = opts.max_disp;
    let disp_iters = opts.disp_newton_iters;
    let max_outer = opts.max_outer_iters;
    let outer_tol = opts.outer_tol;

    let threads: u32 = 256;
    let grid = (n_genes as u32).div_ceil(threads);
    let smem = ((n_sub * p + 2 * n_sub) * std::mem::size_of::<f64>()) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem,
    };

    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(&d_counts)
            .arg(&d_design)
            .arg(&d_log_sf)
            .arg(&d_sf)
            .arg(&d_base_mean)
            .arg(&d_beta_in)
            .arg(&d_alpha_in)
            .arg(&d_prior)
            .arg(&prior_var)
            .arg(&n_genes_i)
            .arg(&n_sub_i)
            .arg(&p_i)
            .arg(&method)
            .arg(&mode_i)
            .arg(&max_irls)
            .arg(&irls_tol)
            .arg(&min_mu)
            .arg(&eta_min)
            .arg(&eta_max)
            .arg(&ridge)
            .arg(&min_disp)
            .arg(&max_disp)
            .arg(&disp_iters)
            .arg(&max_outer)
            .arg(&outer_tol)
            .arg(&mut d_beta)
            .arg(&mut d_mu)
            .arg(&mut d_fisher)
            .arg(&mut d_alpha)
            .arg(&mut d_converged)
            .arg(&mut d_niter)
            .arg(&mut d_atlow)
            .arg(&mut d_athigh)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("nb_glm_fit_kernel: {e}")))?;

    Ok(GpuNbGlmFit {
        beta: dev.dtoh_copy(&d_beta)?,
        mu: dev.dtoh_copy(&d_mu)?,
        fisher: dev.dtoh_copy(&d_fisher)?,
        alpha: dev.dtoh_copy(&d_alpha)?,
        converged: dev.dtoh_copy(&d_converged)?,
        n_iter: dev.dtoh_copy(&d_niter)?,
        at_low: dev.dtoh_copy(&d_atlow)?,
        at_high: dev.dtoh_copy(&d_athigh)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: a tiny 2-group design fits to finite, sensible values on a
    /// GPU host; skipped gracefully when no CUDA device is present.
    #[test]
    fn test_gpu_nb_glm_smoke() {
        let dev = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no CUDA device — skipping GPU NB-GLM smoke test");
                return;
            }
        };
        // 4 samples, 2 groups (control/treat), p=2 design [1, is_treat].
        let n_sub = 4;
        let p = 2;
        let design = vec![
            1.0, 0.0, // ctrl
            1.0, 0.0, // ctrl
            1.0, 1.0, // treat
            1.0, 1.0, // treat
        ];
        let sf = vec![1.0, 1.0, 1.0, 1.0];
        let log_sf = vec![0.0, 0.0, 0.0, 0.0];
        // Two genes: one with a clear treat effect, one flat.
        let counts = vec![
            10.0, 12.0, 40.0, 44.0, // up in treat
            20.0, 22.0, 19.0, 21.0, // flat
        ];
        let n_genes = 2;
        let base_mean: Vec<f64> = (0..n_genes)
            .map(|g| counts[g * n_sub..(g + 1) * n_sub].iter().sum::<f64>() / n_sub as f64)
            .collect();
        let opts = GpuNbGlmOpts {
            method: GPU_NB_GLM_METHOD_CR_MLE,
            max_irls_iters: 100,
            irls_tol: 1e-8,
            min_mu: 1e-10,
            eta_min: -30.0,
            eta_max: 30.0,
            beta_ridge: 1e-6,
            min_disp: 1e-8,
            max_disp: 100.0,
            disp_newton_iters: 25,
            max_outer_iters: 10,
            outer_tol: 1e-4,
        };
        let fit = gpu_nb_glm_fit(
            &dev,
            &counts,
            &design,
            &log_sf,
            &sf,
            n_genes,
            n_sub,
            p,
            opts,
            GpuNbGlmPass::Mle {
                base_mean: &base_mean,
            },
        )
        .expect("gpu fit");
        assert_eq!(fit.beta.len(), n_genes * p);
        assert!(fit.beta.iter().all(|v| v.is_finite()), "beta finite");
        assert!(fit.alpha.iter().all(|v| v.is_finite()), "alpha finite");
        // Gene 0 has ~4× higher treat mean → positive slope; gene 1 ~flat.
        assert!(fit.beta[1] > 0.5, "gene0 slope positive: {}", fit.beta[1]);
        assert!(fit.beta[3].abs() < 0.3, "gene1 slope ~0: {}", fit.beta[3]);
    }
}
