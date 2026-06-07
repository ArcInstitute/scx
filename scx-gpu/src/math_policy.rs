//! GPU math-mode and SpMM-algorithm policies for the PCA power loop (Task 2.5).
//!
//! These two knobs let the PCA path trade reproducibility for speed and record
//! the choice as an explicit artifact (surfaced through
//! `scx_accel::route::AccelExecutionInfo` → `adata.uns["scx_accel"]["pca"]`).
//! Numerics under [`GpuMathMode::AllowTf32`] are validated by subspace/variance
//! agreement, **not** bitwise (TF32 truncates the SpMM/GEMM mantissa).
//!
//! ## Why a deterministic SpMM algorithm matters for graph capture
//!
//! `CUSPARSE_SPMM_ALG_DEFAULT` heuristically picks an algorithm that may use
//! atomics (run-to-run nondeterministic) and may size its workspace
//! differently across shapes. CUDA-graph capture bakes in buffer pointers and
//! forbids allocations inside the captured region, so the captured PCA SpMM
//! segments force [`SpmmAlgPolicy::Deterministic`] (`CUSPARSE_SPMM_CSR_ALG2`,
//! the no-atomics CSR algorithm) regardless of the requested policy.

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
    /// `CUSPARSE_SPMM_CSR_ALG2` — the no-atomics, deterministic CSR algorithm.
    /// Required inside CUDA-graph capture and for bit-reproducible runs.
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
    /// between the heuristic and deterministic algorithms here.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct GpuPcaTuning {
    /// cuBLAS / cuSPARSE math mode.
    pub math_mode: GpuMathMode,
    /// SpMM algorithm-selection policy.
    pub spmm_policy: SpmmAlgPolicy,
}

impl GpuPcaTuning {
    /// Strict-fp32 + heuristic-default — the reproducible, behaviour-preserving
    /// default that matches pre-2.5 results.
    pub fn new(math_mode: GpuMathMode, spmm_policy: SpmmAlgPolicy) -> Self {
        Self {
            math_mode,
            spmm_policy,
        }
    }
}
