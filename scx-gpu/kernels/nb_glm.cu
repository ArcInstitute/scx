// GPU pseudobulk negative-binomial GLM fitter (Stage A).
//
// Thread-per-gene (G1) port of the CPU fitter in
// `scx-accel/src/nb_glm/{math,irls,dispersion}.rs`. Each thread owns one gene:
// it runs the full IRLS (Fisher-scoring) mean fit alternated with the Cox-Reid
// adjusted dispersion root-find, entirely on-chip (no per-iteration global
// round-trip, no library solves). `f64` throughout (native on H100); the design
// `X`, `log(size_factor)` offset, and size factors are staged once per block in
// shared memory and broadcast across the genes a block fits.
//
// The numerical primitives (`nb_loglik`, `digamma`, `trigamma`,
// `nb_loglik_sum_dlogalpha`, the Cox-Reid gradient, the safeguarded Illinois
// root-find, method-of-moments init) are ported verbatim from `math.rs` /
// `dispersion.rs` / `irls.rs`, including every magic constant, so the GPU result
// tracks the CPU `f64` reference within the cross-validation tolerances in
// `DE-GPU-ACC.md` §10. The cheap per-gene tail (Wald, Cook's) and all cross-gene
// steps (trend, prior var, filtering, BH) stay on the host.
//
// A single kernel serves both passes via `mode`:
//   * mode 0 (MLE):    moments init -> IRLS -> outer{ dispersion(no prior) -> IRLS }.
//   * mode 1 (SHRINK): recompute mu from warm beta -> dispersion(prior) -> IRLS refit.
//
// Layout bounds: `p <= NB_PMAX`, `n_sub <= NB_NSUB_MAX`; larger inputs fall back
// to the CPU path via the route planner.

#include <math_constants.h>

#define NB_PMAX 8
#define NB_NSUB_MAX 64

// Dispersion method tags (mirror `DispersionMethod` in types.rs).
#define NB_METHOD_MOMENTS 0
#define NB_METHOD_CR_MLE 1
#define NB_METHOD_CR_SHRUNK 2

// Pass mode.
#define NB_MODE_MLE 0
#define NB_MODE_SHRINK 1

// Threshold below which the NB collapses to its Poisson limit (math.rs:13).
__device__ __forceinline__ double nb_poisson_eps() { return 1e-12; }

__device__ __forceinline__ double nb_clamp(double v, double lo, double hi) {
    return fmin(fmax(v, lo), hi);
}

// ---------------------------------------------------------------------------
// Special functions (math.rs:78-105). Recurrence to x>=10 then asymptotic series.
// ---------------------------------------------------------------------------
__device__ double d_digamma(double x) {
    double result = 0.0;
    while (x < 10.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    double inv = 1.0 / x;
    double inv2 = inv * inv;
    return result + log(x) - 0.5 * inv -
           inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0));
}

__device__ double d_trigamma(double x) {
    double result = 0.0;
    while (x < 10.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    double inv = 1.0 / x;
    double inv2 = inv * inv;
    return result + inv + 0.5 * inv2 +
           inv * inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0));
}

// NB2 log-likelihood of one observation (math.rs:30-50).
__device__ double d_nb_loglik(double y, double mu, double alpha) {
    if (alpha <= nb_poisson_eps()) {
        double ll = -mu - lgamma(y + 1.0);
        if (y > 0.0) ll += y * log(mu);
        return ll;
    }
    double theta = 1.0 / alpha;
    double log1p_am = log1p(alpha * mu);
    double ll = lgamma(y + theta) - lgamma(theta) - lgamma(y + 1.0) - theta * log1p_am;
    if (y > 0.0) ll += y * (log(alpha * mu) - log1p_am);
    return ll;
}

__device__ double d_nb_loglik_sum(const double* y, const double* mu, int n, double alpha) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += d_nb_loglik(y[i], mu[i], alpha);
    return s;
}

// d(Σ logNB)/d(log alpha) (math.rs:332-344).
__device__ double d_nb_loglik_sum_dlogalpha(const double* y, const double* mu, int n,
                                            double alpha) {
    double theta = 1.0 / alpha;
    double psi_theta = d_digamma(theta);
    double log_theta = log(theta);
    double grad_theta = 0.0;
    for (int s = 0; s < n; s++) {
        double tpm = theta + mu[s];
        grad_theta += d_digamma(y[s] + theta) - psi_theta + log_theta - log(tpm) + 1.0 -
                      theta / tpm - y[s] / tpm;
    }
    return -theta * grad_theta;
}

// ---------------------------------------------------------------------------
// Small dense linear algebra (p <= NB_PMAX), per-thread local scratch.
// ---------------------------------------------------------------------------

// Solve A x = b (A is p*p row-major). Closed form for p<=2, else Gauss
// elimination with partial pivoting on a local copy. Returns false on a
// non-finite / singular result (mirrors `solve_spd` returning None).
__device__ bool d_solve(const double* A, const double* b, double* x, int p) {
    if (p == 1) {
        double a = A[0];
        if (a == 0.0) return false;
        x[0] = b[0] / a;
        return isfinite(x[0]);
    }
    if (p == 2) {
        double det = A[0] * A[3] - A[1] * A[2];
        if (det == 0.0) return false;
        x[0] = (b[0] * A[3] - A[1] * b[1]) / det;
        x[1] = (A[0] * b[1] - b[0] * A[2]) / det;
        return isfinite(x[0]) && isfinite(x[1]);
    }
    double m[NB_PMAX * NB_PMAX];
    double rhs[NB_PMAX];
    for (int i = 0; i < p * p; i++) m[i] = A[i];
    for (int i = 0; i < p; i++) rhs[i] = b[i];
    for (int col = 0; col < p; col++) {
        // Partial pivot.
        int piv = col;
        double best = fabs(m[col * p + col]);
        for (int r = col + 1; r < p; r++) {
            double v = fabs(m[r * p + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best == 0.0) return false;
        if (piv != col) {
            for (int k = 0; k < p; k++) {
                double t = m[col * p + k]; m[col * p + k] = m[piv * p + k]; m[piv * p + k] = t;
            }
            double t = rhs[col]; rhs[col] = rhs[piv]; rhs[piv] = t;
        }
        double diag = m[col * p + col];
        for (int r = col + 1; r < p; r++) {
            double f = m[r * p + col] / diag;
            for (int k = col; k < p; k++) m[r * p + k] -= f * m[col * p + k];
            rhs[r] -= f * rhs[col];
        }
    }
    for (int i = p - 1; i >= 0; i--) {
        double acc = rhs[i];
        for (int k = i + 1; k < p; k++) acc -= m[i * p + k] * x[k];
        x[i] = acc / m[i * p + i];
        if (!isfinite(x[i])) return false;
    }
    return true;
}

// Invert A (p*p row-major) into inv. Closed form for p<=2, else Gauss-Jordan
// with partial pivoting. Returns false on a non-finite / singular result
// (mirrors `m_stats` returning None).
__device__ bool d_invert(const double* A, double* inv, int p) {
    if (p == 1) {
        if (A[0] == 0.0) return false;
        inv[0] = 1.0 / A[0];
        return isfinite(inv[0]);
    }
    if (p == 2) {
        double det = A[0] * A[3] - A[1] * A[2];
        if (det == 0.0) return false;
        double id = 1.0 / det;
        inv[0] = A[3] * id;
        inv[1] = -A[1] * id;
        inv[2] = -A[2] * id;
        inv[3] = A[0] * id;
        return isfinite(inv[0]) && isfinite(inv[3]);
    }
    double m[NB_PMAX * NB_PMAX];
    for (int i = 0; i < p * p; i++) m[i] = A[i];
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) inv[i * p + j] = (i == j) ? 1.0 : 0.0;
    for (int col = 0; col < p; col++) {
        int piv = col;
        double best = fabs(m[col * p + col]);
        for (int r = col + 1; r < p; r++) {
            double v = fabs(m[r * p + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best == 0.0) return false;
        if (piv != col) {
            for (int k = 0; k < p; k++) {
                double t = m[col * p + k]; m[col * p + k] = m[piv * p + k]; m[piv * p + k] = t;
                double u = inv[col * p + k]; inv[col * p + k] = inv[piv * p + k]; inv[piv * p + k] = u;
            }
        }
        double diag = m[col * p + col];
        double idiag = 1.0 / diag;
        for (int k = 0; k < p; k++) { m[col * p + k] *= idiag; inv[col * p + k] *= idiag; }
        for (int r = 0; r < p; r++) {
            if (r == col) continue;
            double f = m[r * p + col];
            for (int k = 0; k < p; k++) {
                m[r * p + k] -= f * m[col * p + k];
                inv[r * p + k] -= f * inv[col * p + k];
            }
        }
    }
    for (int i = 0; i < p * p; i++)
        if (!isfinite(inv[i])) return false;
    return true;
}

// Assemble XᵀWX + ridge·I (p*p row-major, symmetric) (irls.rs:34-61).
__device__ void d_assemble_xtwx(const double* design, const double* w, int n, int p,
                                double ridge, double* xtwx) {
    for (int i = 0; i < p * p; i++) xtwx[i] = 0.0;
    for (int s = 0; s < n; s++) {
        const double* row = &design[s * p];
        double ws = w[s];
        for (int j = 0; j < p; j++) {
            double wxj = ws * row[j];
            for (int k = j; k < p; k++) xtwx[j * p + k] += wxj * row[k];
        }
    }
    for (int j = 0; j < p; j++) {
        for (int k = j + 1; k < p; k++) xtwx[k * p + j] = xtwx[j * p + k];
        xtwx[j * p + j] += ridge;
    }
}

// eta/mu/w/z for the current beta (irls.rs:87-112).
__device__ void d_working(const double* y, const double* design, const double* log_sf,
                          const double* beta, int n, int p, double alpha, double min_mu,
                          double eta_min, double eta_max, double* mu, double* w, double* z) {
    for (int s = 0; s < n; s++) {
        const double* row = &design[s * p];
        double eta = log_sf[s];
        for (int k = 0; k < p; k++) eta += row[k] * beta[k];
        double eta_c = nb_clamp(eta, eta_min, eta_max);
        double mu_s = fmax(exp(eta_c), min_mu);
        mu[s] = mu_s;
        w[s] = mu_s / (1.0 + alpha * mu_s);
        z[s] = eta_c - log_sf[s] + (y[s] - mu_s) / mu_s;
    }
}

// Per-gene IRLS at fixed alpha (irls.rs:120-200). Updates beta in place; fills
// mu and fisher. Returns whether the deviance converged.
__device__ bool d_irls(const double* y, const double* design, const double* log_sf, int n,
                       int p, double alpha, double* beta, double* mu, double* fisher,
                       int max_iters, double tol, double min_mu, double eta_min,
                       double eta_max, double ridge) {
    double w[NB_NSUB_MAX];
    double z[NB_NSUB_MAX];
    double xtwx[NB_PMAX * NB_PMAX];
    double xtwz[NB_PMAX];
    double beta_next[NB_PMAX];
    double prev_dev = CUDART_INF;
    bool converged = false;

    for (int iter = 0; iter < max_iters; iter++) {
        d_working(y, design, log_sf, beta, n, p, alpha, min_mu, eta_min, eta_max, mu, w, z);
        double dev = -2.0 * d_nb_loglik_sum(y, mu, n, alpha);
        if (iter > 0 && fabs(prev_dev - dev) <= tol * (fabs(dev) + 1e-8)) {
            converged = true;
            break;
        }
        prev_dev = dev;
        d_assemble_xtwx(design, w, n, p, ridge, xtwx);
        for (int j = 0; j < p; j++) xtwz[j] = 0.0;
        for (int s = 0; s < n; s++) {
            const double* row = &design[s * p];
            double wz = w[s] * z[s];
            for (int j = 0; j < p; j++) xtwz[j] += wz * row[j];
        }
        if (d_solve(xtwx, xtwz, beta_next, p)) {
            for (int j = 0; j < p; j++) beta[j] = beta_next[j];
        } else {
            break;  // non-finite solve -> stop, mark non-converged
        }
    }
    // Final consistent pass so mu / fisher match the returned beta.
    d_working(y, design, log_sf, beta, n, p, alpha, min_mu, eta_min, eta_max, mu, w, z);
    d_assemble_xtwx(design, w, n, p, ridge, fisher);
    return converged;
}

// dCR/d(log alpha), optionally MAP-penalized (dispersion.rs:154-195). `mu` held
// fixed. Returns +1.0 sentinel if XᵀWX is non-invertible (dispersion.rs:173).
__device__ double d_cr_grad(const double* y, const double* mu, const double* design, int n,
                           int p, double ridge, double alpha, bool have_prior,
                           double log_alpha_trend, double sigma2) {
    double w[NB_NSUB_MAX];
    for (int s = 0; s < n; s++) w[s] = mu[s] / (1.0 + alpha * mu[s]);
    double xtwx[NB_PMAX * NB_PMAX];
    double minv[NB_PMAX * NB_PMAX];
    d_assemble_xtwx(design, w, n, p, ridge, xtwx);
    if (!d_invert(xtwx, minv, p)) return 1.0;

    double dlogdet_dalpha = 0.0;
    for (int s = 0; s < n; s++) {
        const double* row = &design[s * p];
        double h = 0.0;
        for (int j = 0; j < p; j++) {
            double mij_xj = 0.0;
            for (int k = 0; k < p; k++) mij_xj += minv[j * p + k] * row[k];
            h += row[j] * mij_xj;
        }
        dlogdet_dalpha += -(w[s] * w[s]) * h;
    }
    double dlogdet_dt = alpha * dlogdet_dalpha;
    double g = d_nb_loglik_sum_dlogalpha(y, mu, n, alpha) - 0.5 * dlogdet_dt;
    if (have_prior) g -= (log(alpha) - log_alpha_trend) / sigma2;
    return g;
}

// Maximize the Cox-Reid objective over alpha by a bracketed Illinois root-find
// on g(t)=0, t=log(alpha) (dispersion.rs:201-316). Writes at_low/at_high flags.
__device__ double d_fit_dispersion(const double* y, const double* mu, const double* design,
                                   int n, int p, double ridge, double alpha_init,
                                   bool have_prior, double log_alpha_trend, double sigma2,
                                   double min_disp, double max_disp, int newton_iters,
                                   unsigned char* at_low, unsigned char* at_high) {
    *at_low = 0;
    *at_high = 0;
    double t_lo = log(min_disp);
    double t_hi = log(max_disp);

    double t0 = log(nb_clamp(alpha_init, min_disp, max_disp));
    double g0 = d_cr_grad(y, mu, design, n, p, ridge, exp(t0), have_prior, log_alpha_trend, sigma2);
    if (!isfinite(g0)) {
        return nb_clamp(alpha_init, min_disp, max_disp);
    }

    double a, ga, b, gb;
    if (g0 >= 0.0) {
        a = t0; ga = g0;
        double step = 0.5;
        double tb = fmin(t0 + step, t_hi);
        while (true) {
            double gtb = d_cr_grad(y, mu, design, n, p, ridge, exp(tb), have_prior,
                                   log_alpha_trend, sigma2);
            if (!isfinite(gtb) || gtb <= 0.0) { b = tb; gb = gtb; break; }
            if (tb >= t_hi) { *at_high = 1; return max_disp; }
            a = tb; ga = gtb;
            step *= 2.0;
            tb = fmin(tb + step, t_hi);
        }
    } else {
        b = t0; gb = g0;
        double step = 0.5;
        double ta = fmax(t0 - step, t_lo);
        while (true) {
            double gta = d_cr_grad(y, mu, design, n, p, ridge, exp(ta), have_prior,
                                   log_alpha_trend, sigma2);
            if (!isfinite(gta) || gta >= 0.0) { a = ta; ga = gta; break; }
            if (ta <= t_lo) { *at_low = 1; return min_disp; }
            b = ta; gb = gta;
            step *= 2.0;
            ta = fmax(ta - step, t_lo);
        }
    }

    double t = b;
    for (int it = 0; it < newton_iters; it++) {
        double c = (a * gb - b * ga) / (gb - ga);
        double clo = fmin(a, b);
        double chi = fmax(a, b);
        if (!isfinite(c) || c <= clo || c >= chi) c = 0.5 * (a + b);
        double gc = d_cr_grad(y, mu, design, n, p, ridge, exp(c), have_prior, log_alpha_trend,
                              sigma2);
        t = c;
        if (fabs(gc) < 1e-8 || fabs(a - b) < 1e-10) break;
        if (gc * gb < 0.0) {
            a = b; ga = gb;
        } else {
            ga *= 0.5;
        }
        b = c; gb = gc;
    }
    double alpha = nb_clamp(exp(t), min_disp, max_disp);
    if (alpha <= min_disp * (1.0 + 1e-9)) *at_low = 1;
    if (alpha >= max_disp * (1.0 - 1e-9)) *at_high = 1;
    return alpha;
}

// Method-of-moments dispersion init (dispersion.rs:40-67).
__device__ double d_moments(const double* y, const double* sf, int n, double min_disp,
                            double max_disp) {
    double sum = 0.0, sum_sq = 0.0;
    for (int s = 0; s < n; s++) {
        double yn = y[s] / sf[s];
        sum += yn;
        sum_sq += yn * yn;
    }
    double dn = (double)n;
    double mean = sum / dn;
    if (mean <= 0.0) return min_disp;
    double var = (dn > 1.0) ? fmax((sum_sq - dn * mean * mean) / (dn - 1.0), 0.0) : 0.0;
    double alpha0 = (var - mean) / (mean * mean);
    return nb_clamp(alpha0, min_disp, max_disp);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void nb_glm_fit_kernel(
    const double* __restrict__ counts,           // [n_genes * n_sub] gene-major
    const double* __restrict__ design,           // [n_sub * p] row-major
    const double* __restrict__ log_sf,           // [n_sub]
    const double* __restrict__ sf,               // [n_sub]
    const double* __restrict__ base_mean,        // [n_genes] (MLE beta init)
    const double* __restrict__ beta_in,          // [n_genes * p] warm start (SHRINK) or null
    const double* __restrict__ alpha_in,         // [n_genes] disp init (SHRINK) or null
    const double* __restrict__ prior_log_target, // [n_genes] (SHRINK) or null
    double prior_var,
    int n_genes, int n_sub, int p, int method, int mode,
    int max_irls_iters, double irls_tol, double min_mu, double eta_min, double eta_max,
    double beta_ridge, double min_disp, double max_disp, int disp_newton_iters,
    int max_outer_iters, double outer_tol,
    double* __restrict__ beta_out,             // [n_genes * p]
    double* __restrict__ mu_out,               // [n_genes * n_sub]
    double* __restrict__ fisher_out,           // [n_genes * p * p]
    double* __restrict__ alpha_out,            // [n_genes]
    unsigned char* __restrict__ converged_out, // [n_genes]
    unsigned int* __restrict__ niter_out,      // [n_genes]
    unsigned char* __restrict__ atlow_out,     // [n_genes]
    unsigned char* __restrict__ athigh_out) {  // [n_genes]
    // Stage shared design + offsets once per block (broadcast across genes).
    extern __shared__ double smem[];
    double* s_design = smem;             // n_sub * p
    double* s_log_sf = s_design + n_sub * p;  // n_sub
    double* s_sf = s_log_sf + n_sub;          // n_sub
    for (int i = threadIdx.x; i < n_sub * p; i += blockDim.x) s_design[i] = design[i];
    for (int i = threadIdx.x; i < n_sub; i += blockDim.x) {
        s_log_sf[i] = log_sf[i];
        s_sf[i] = sf[i];
    }
    __syncthreads();

    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_genes) return;

    const double* y = &counts[(size_t)g * n_sub];
    double beta[NB_PMAX];
    double mu[NB_PMAX > 0 ? NB_NSUB_MAX : 1];
    double fisher[NB_PMAX * NB_PMAX];

    // All-zero gene -> fixed all-zero state (mod.rs GeneState::all_zero).
    bool all_zero = true;
    for (int s = 0; s < n_sub; s++) {
        if (y[s] != 0.0) { all_zero = false; break; }
    }
    if (all_zero) {
        for (int j = 0; j < p; j++) beta_out[(size_t)g * p + j] = 0.0;
        for (int s = 0; s < n_sub; s++) mu_out[(size_t)g * n_sub + s] = 0.0;
        for (int j = 0; j < p * p; j++) fisher_out[(size_t)g * p * p + j] = 0.0;
        alpha_out[g] = CUDART_NAN;
        converged_out[g] = 1;
        niter_out[g] = 0;
        atlow_out[g] = 0;
        athigh_out[g] = 0;
        return;
    }

    double alpha;
    unsigned char at_low = 0, at_high = 0;
    unsigned int n_outer = 1;
    bool converged_irls;
    bool outer_converged = true;

    if (mode == NB_MODE_MLE) {
        double bm = base_mean[g];
        for (int j = 0; j < p; j++) beta[j] = 0.0;
        beta[0] = log(fmax(bm, 1e-4));  // MEAN_FLOOR
        alpha = d_moments(y, s_sf, n_sub, min_disp, max_disp);
        converged_irls = d_irls(y, s_design, s_log_sf, n_sub, p, alpha, beta, mu, fisher,
                                max_irls_iters, irls_tol, min_mu, eta_min, eta_max, beta_ridge);
        if (method != NB_METHOD_MOMENTS) {
            outer_converged = false;
            for (int it = 0; it < max_outer_iters; it++) {
                unsigned char lo = 0, hi = 0;
                double new_alpha = d_fit_dispersion(y, mu, s_design, n_sub, p, beta_ridge, alpha,
                                                    false, 0.0, 1.0, min_disp, max_disp,
                                                    disp_newton_iters, &lo, &hi);
                double prev_alpha = alpha;
                alpha = new_alpha;
                at_low = lo;
                at_high = hi;
                converged_irls = d_irls(y, s_design, s_log_sf, n_sub, p, alpha, beta, mu, fisher,
                                        max_irls_iters, irls_tol, min_mu, eta_min, eta_max,
                                        beta_ridge);
                n_outer += 1;
                double rel_a = fabs(alpha - prev_alpha) / (fabs(prev_alpha) + 1.0);
                if (rel_a < outer_tol) {
                    outer_converged = true;
                    break;
                }
            }
        }
    } else {
        // SHRINK pass: recompute mu from warm beta (mu depends only on beta),
        // shrink dispersion with the MAP prior, then refit beta once.
        for (int j = 0; j < p; j++) beta[j] = beta_in[(size_t)g * p + j];
        double w_tmp[NB_NSUB_MAX];
        double z_tmp[NB_NSUB_MAX];
        d_working(y, s_design, s_log_sf, beta, n_sub, p, alpha_in[g], min_mu, eta_min, eta_max,
                  mu, w_tmp, z_tmp);
        alpha = d_fit_dispersion(y, mu, s_design, n_sub, p, beta_ridge, alpha_in[g], true,
                                 prior_log_target[g], prior_var, min_disp, max_disp,
                                 disp_newton_iters, &at_low, &at_high);
        converged_irls = d_irls(y, s_design, s_log_sf, n_sub, p, alpha, beta, mu, fisher,
                                max_irls_iters, irls_tol, min_mu, eta_min, eta_max, beta_ridge);
        // n_iter / mle-converged are combined on the host from the MLE pass.
        n_outer = 0;
        outer_converged = true;
    }

    for (int j = 0; j < p; j++) beta_out[(size_t)g * p + j] = beta[j];
    for (int s = 0; s < n_sub; s++) mu_out[(size_t)g * n_sub + s] = mu[s];
    for (int j = 0; j < p * p; j++) fisher_out[(size_t)g * p * p + j] = fisher[j];
    alpha_out[g] = alpha;
    converged_out[g] = (converged_irls && outer_converged) ? 1 : 0;
    niter_out[g] = n_outer;
    atlow_out[g] = at_low;
    athigh_out[g] = at_high;
}
