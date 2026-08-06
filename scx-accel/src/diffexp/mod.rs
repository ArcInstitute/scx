//! Differential expression (Wilcoxon rank-sum / pdex reference mode).
//!
//! [`cpu`] holds the host implementations and the parity oracle; [`gpu`]
//! (behind the `gpu` feature) holds the GPU orchestration that drives the
//! `scx-gpu` device primitives. Routing is decided by `crate::route`.
//!
//! [`groups`] owns the one definition of "which cells take part" — every
//! kernel, CPU or GPU, dense or sparse, derives its rest denominator and rank
//! pool from it rather than from `n_obs`.

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod groups;

pub use groups::{partition_by_group, GroupPartition};

pub use cpu::{
    finalize_pdex, merge_diff_exp_results, pdex_ref, pdex_ref_core, pdex_ref_sparse,
    pdex_ref_streaming, wilcoxon_rank_sum, wilcoxon_rank_sum_sparse, wilcoxon_rank_sum_streaming,
    DiffExpResult, PdexRefResult,
};
// Shared inference helpers reused by the NB-GLM Wald + BH steps (spec §7.7–7.8).
pub(crate) use cpu::{benjamini_hochberg, normal_sf};
#[cfg(feature = "gpu")]
pub use gpu::{
    pdex_ref_gpu, pdex_ref_gpu_dense, wilcoxon_rank_sum_gpu, wilcoxon_rank_sum_gpu_dense,
    GpuDeShardInput,
};
