//! Pseudobulk negative-binomial GLM (NB-GLM) for differential expression.
//!
//! A Rust-native, **CPU-only**, `f64` end-to-end DESeq2-*core* fitter (IRLS /
//! Fisher scoring for the mean, Cox–Reid adjusted dispersion with a parametric
//! trend + empirical-Bayes shrinkage, Wald inference). See `GPU-NB-GLM-SPEC.md`.
//! There is no GPU path in v1 — a future GPU extension (spec §13) is additive.
//!
//! This module is built up across phases. Phase 1 (here) provides the shared
//! public types ([`types`]), input validation ([`validate`]), median-ratio size
//! factors ([`size_factors`]), and the numerical primitives ([`math`]) — the NB
//! log-likelihood, the analytic IRLS working weights, and the Cox–Reid objective
//! and dispersion gradient. The fitter orchestration (IRLS loop, dispersion
//! trend/shrinkage, Wald) and the public `pseudobulk_nb_glm` entry point land in
//! Phase 2.

pub mod math;
pub mod types;

// Phase-1 scaffolding: these helpers are fully tested here but are first *called*
// by the Phase-2 `pseudobulk_nb_glm` orchestration, so they read as dead code
// under a non-test build until then. Drop the allow when Phase 2 wires them in.
#[allow(dead_code)]
pub(crate) mod size_factors;
#[allow(dead_code)]
pub(crate) mod validate;

pub use types::{
    DispersionMethod, DispersionTrend, NbGlmContrast, NbGlmDiagnostics, NbGlmOptions, NbGlmResult,
};

// Phase-1 helpers consumed by the Phase-2 orchestration; crate-internal until the
// public `pseudobulk_nb_glm` entry point wires them together (hence unused in a
// non-test build for now). The tests reach them through these re-exports.
#[allow(unused_imports)]
pub(crate) use size_factors::median_ratio_size_factors;
#[allow(unused_imports)]
pub(crate) use validate::{validate_contrast, validate_inputs};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
