//! GPU math-mode and SpMM-algorithm policies for the PCA power loop (Task 2.5).
//!
//! These two knobs let the PCA path trade numerical strictness and algorithm
//! pinning for speed, and record the choice as an explicit artifact (surfaced
//! through
//! `scx_accel::route::AccelExecutionInfo` → `adata.uns["scx_accel"]["pca"]`).
//! Numerics under [`GpuMathMode::AllowTf32`] are validated by subspace/variance
//! agreement, **not** bitwise (TF32 truncates the SpMM/GEMM mantissa).
//!
//! ## What the SpMM algorithm policy is, and what it is NOT
//!
//! `CUSPARSE_SPMM_ALG_DEFAULT` lets cuSPARSE pick heuristically, which may
//! select an atomics-based kernel. [`SpmmAlgPolicy::Deterministic`] instead
//! pins `CUSPARSE_SPMM_CSR_ALG2` explicitly.
//!
//! **It is not a bit-reproducibility guarantee for PCA, and must not be
//! documented as one.** cuSPARSE states plainly that "operations using
//! transpose or conjugate-transpose have no reproducibility guarantees"
//! (CUDA 12.6 cuSPARSE § Result Reproducibility). The randomized PCA power
//! loop necessarily issues `Aᵀ · Y` through
//! `CUSPARSE_OPERATION_TRANSPOSE`, so **no** choice of algorithm makes the
//! loop as a whole run-to-run bitwise identical. What the policy buys is a
//! pinned, non-heuristic algorithm selection — reproducible *dispatch*, and
//! the removal of the heuristic's freedom to switch kernels between runs or
//! across shapes — not reproducible bits.
//!
//! The policy reaches **both** PCA power loops — the device-resident one and
//! the streaming operator used when the matrix exceeds VRAM. That symmetry is
//! load-bearing rather than incidental: `uns["scx_accel"]["pca"]["spmm_policy"]`
//! is stamped from the caller's request, and residency is decided dynamically
//! against free VRAM, so a policy honoured on only one path would make the
//! recorded provenance false on runs the caller cannot predict.
//!
//! (Two earlier versions of this note were wrong. The first said captured PCA
//! SpMM segments force `Deterministic` regardless of the requested policy —
//! SpMM-segment CUDA-graph capture has since been removed, so nothing
//! overrides the request today. The second called the result
//! "bit-reproducible", which the transpose multiply rules out.)

use cudarc::cublas::sys as cbs;
use cudarc::cusparse::sys as csp;

/// cuBLAS / cuSPARSE floating-point math mode for the dense + sparse multiplies
/// in PCA.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GpuMathMode {
    /// Prescribed fp32 arithmetic with no reduced-precision substitution
    /// (`CUBLAS_PEDANTIC_MATH`). `CUBLAS_DEFAULT_MATH` is *not* used here: NVIDIA
    /// documents the default as performance-oriented and free to use Tensor
    /// Cores (incl. TF32) for some routines, so only the pedantic mode actually
    /// guarantees the "strict fp32" label. The reproducible default.
    #[default]
    StrictFp32,
    /// Allow TF32 tensor-op acceleration (`CUBLAS_TF32_TENSOR_OP_MATH`) — faster
    /// on Ampere+ at reduced mantissa precision. Validate by subspace, not bits.
    AllowTf32,
}

impl GpuMathMode {
    /// Stable snake-case identifier for Python / benchmark JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            GpuMathMode::StrictFp32 => "strict_fp32",
            GpuMathMode::AllowTf32 => "allow_tf32",
        }
    }

    /// The cuBLAS math mode this maps to.
    pub fn to_cublas(self) -> cbs::cublasMath_t {
        match self {
            GpuMathMode::StrictFp32 => cbs::cublasMath_t::CUBLAS_PEDANTIC_MATH,
            GpuMathMode::AllowTf32 => cbs::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
        }
    }

    /// Parse from the Python kwarg form. Unknown strings map to the strict default.
    pub fn from_allow_tf32(allow_tf32: bool) -> Self {
        if allow_tf32 {
            GpuMathMode::AllowTf32
        } else {
            GpuMathMode::StrictFp32
        }
    }
}

/// cuSPARSE SpMM algorithm-selection policy for the PCA power-loop multiplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpmmAlgPolicy {
    /// `CUSPARSE_SPMM_ALG_DEFAULT` — cuSPARSE's heuristic pick (may use atomics;
    /// run-to-run nondeterministic). Fastest general default.
    #[default]
    Default,
    /// Pin `CUSPARSE_SPMM_CSR_ALG2` explicitly instead of letting cuSPARSE
    /// choose. Applied to both the forward (`NON_TRANSPOSE`) and transpose
    /// multiplies of the power loop, on the resident and streaming paths alike.
    ///
    /// **The name overpromises: this does not make GPU PCA bit-reproducible.**
    /// cuSPARSE gives no reproducibility guarantee for transpose operations,
    /// and the power loop issues one every iteration. It pins the algorithm
    /// choice; it does not pin the bits. See the module docs.
    Deterministic,
    /// Reserved: intended to time the candidate algorithms once per shape and
    /// cache the winner. **Not yet implemented** — currently resolves to the
    /// same heuristic algorithm as [`SpmmAlgPolicy::Default`]
    /// (`CUSPARSE_SPMM_ALG_DEFAULT`). The variant exists so the wire API is
    /// stable when per-shape timing lands.
    BenchmarkOnce,
}

impl SpmmAlgPolicy {
    /// Stable snake-case identifier for Python / benchmark JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            SpmmAlgPolicy::Default => "default",
            SpmmAlgPolicy::Deterministic => "deterministic",
            SpmmAlgPolicy::BenchmarkOnce => "benchmark_once",
        }
    }

    /// The concrete cuSPARSE algorithm this policy resolves to.
    ///
    /// [`SpmmAlgPolicy::BenchmarkOnce`] is not yet implemented and resolves to
    /// the same heuristic algorithm as [`SpmmAlgPolicy::Default`]
    /// (`CUSPARSE_SPMM_ALG_DEFAULT`); when per-shape timing lands it will pick
    /// between `CUSPARSE_SPMM_ALG_DEFAULT` and `CUSPARSE_SPMM_CSR_ALG2` here.
    /// (Naming ALG2 "the deterministic algorithm" would repeat the overclaim
    /// documented on [`SpmmAlgPolicy::Deterministic`].)
    pub fn to_alg(self) -> csp::cusparseSpMMAlg_t {
        match self {
            SpmmAlgPolicy::Default | SpmmAlgPolicy::BenchmarkOnce => {
                csp::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT
            }
            SpmmAlgPolicy::Deterministic => csp::cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG2,
        }
    }
}

/// Bundle of PCA GPU tuning knobs, threaded through the PCA entry points.
///
/// [`Default`] is strict-fp32 + the cuSPARSE heuristic SpMM — the
/// behaviour-preserving configuration that matches pre-2.5 results.
///
/// Neither policy makes GPU PCA bit-reproducible; the transpose multiply in
/// the power loop has no cuSPARSE reproducibility guarantee under any
/// algorithm. [`SpmmAlgPolicy::Deterministic`] pins the algorithm rather than
/// the bits — see [`SpmmAlgPolicy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GpuPcaTuning {
    /// cuBLAS / cuSPARSE math mode.
    pub math_mode: GpuMathMode,
    /// SpMM algorithm-selection policy.
    pub spmm_policy: SpmmAlgPolicy,
}

impl GpuPcaTuning {
    /// Bundle an explicit math mode and SpMM policy. For the
    /// behaviour-preserving configuration use [`GpuPcaTuning::default`], whose
    /// reproducibility caveat is documented on the struct.
    pub fn new(math_mode: GpuMathMode, spmm_policy: SpmmAlgPolicy) -> Self {
        Self {
            math_mode,
            spmm_policy,
        }
    }
}
