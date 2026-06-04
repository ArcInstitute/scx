//! Memory-budget breakdowns shared by the sequential and plan-driven loaders.
//!
//! Both [`crate::pipeline::TrainingPipeline`] and
//! [`crate::index_plan::IndexPlanLoader`] auto-tune their config to fit a
//! caller-supplied `max_memory_mb`. The two paths have different shapes
//! but decompose into the same handful of components: cache, batch buffer,
//! per-path lookahead/prefetch overhead, transient temporaries, and
//! constant Python overhead.
//!
//! [`BudgetBreakdown`] is the shared component type that both paths produce
//! at construction time. The auto-tune algorithms remain in the respective
//! modules — only the breakdown shape and its Python representation are
//! shared here.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

/// Constant Python/Arrow/numpy/threads overhead estimate (bytes). Both
/// budget models add this to their per-component sum.
pub const PYTHON_OVERHEAD_BYTES: usize = 50 * 1024 * 1024;

/// Per-component memory breakdown produced by both budget paths so that
/// callers (Python `memory_budget()` accessors, the benchmark harness, the
/// `SCX_LOADER_PROFILE` Drop dump) can render a consistent "where the bytes
/// go" report.
///
/// Fields are intentionally additive — `total_bytes` is the saturating sum
/// of the others. None of them count mmap-resident pages: the kernel page
/// cache is treated as evictable under pressure on both paths and explicitly
/// excluded from the auto-tune budget. (The sequential path keeps its
/// `mmap_bytes` term separate on the existing
/// [`crate::pipeline::MemoryBudget`] surface for backwards compatibility.)
#[derive(Debug, Clone, Copy, Default)]
pub struct BudgetBreakdown {
    /// Decoded shard cache budget (bytes). Sequential:
    /// `(shard_group_size + 1) × decoded_shard_bytes`. Plan-driven:
    /// `effective_cache_shards × shard_decoded_bytes`.
    pub cache_bytes: usize,
    /// Dense batch buffer(s) carried in flight. Sequential:
    /// `(prefetch_batches.max(2) + 1) × batch_size × n_output_genes × 4`.
    /// Plan-driven: `2 × max_plan_size × n_output_cols × 4` (paired
    /// `x` / `x_paired`).
    pub batch_buffer_bytes: usize,
    /// Per-path lookahead/prefetch staging overhead (e.g. plan-tuple Vecs in
    /// the index-plan iterator). Sequential paths report `0` here.
    pub lookahead_overhead_bytes: usize,
    /// Short-lived temporaries that are co-resident with the batch buffer
    /// during a `process_plan` call: per-batch obs `Vec`s allocated by
    /// `extract_obs_columns`, the `PairRequest` sort scratch in
    /// `gather_pairs_dense`, etc. Empty (`0`) on the sequential path.
    pub transient_bytes: usize,
    /// Constant Python interpreter / numpy / Arrow / thread-stack overhead.
    /// Equal to [`PYTHON_OVERHEAD_BYTES`] on both paths.
    pub python_overhead_bytes: usize,
    /// Saturating sum of every other field.
    pub total_bytes: usize,
}

impl BudgetBreakdown {
    /// Construct a breakdown from its component parts. `total_bytes` is the
    /// saturating sum of the inputs — callers don't pass it explicitly.
    pub fn new(
        cache_bytes: usize,
        batch_buffer_bytes: usize,
        lookahead_overhead_bytes: usize,
        transient_bytes: usize,
        python_overhead_bytes: usize,
    ) -> Self {
        let total_bytes = cache_bytes
            .saturating_add(batch_buffer_bytes)
            .saturating_add(lookahead_overhead_bytes)
            .saturating_add(transient_bytes)
            .saturating_add(python_overhead_bytes);
        BudgetBreakdown {
            cache_bytes,
            batch_buffer_bytes,
            lookahead_overhead_bytes,
            transient_bytes,
            python_overhead_bytes,
            total_bytes,
        }
    }

    /// Returns `true` iff `total_bytes <= max_memory_mb × 1 MiB`.
    pub fn fits_within(&self, max_memory_mb: usize) -> bool {
        self.total_bytes <= max_memory_mb.saturating_mul(1024 * 1024)
    }

    /// Render as a Python dict with keys
    /// `{cache_bytes, batch_buffer_bytes, lookahead_overhead_bytes,
    ///   transient_bytes, python_overhead_bytes, total_bytes}`. All values
    /// are `int`.
    #[cfg(feature = "python")]
    pub fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("cache_bytes", self.cache_bytes)?;
        dict.set_item("batch_buffer_bytes", self.batch_buffer_bytes)?;
        dict.set_item("lookahead_overhead_bytes", self.lookahead_overhead_bytes)?;
        dict.set_item("transient_bytes", self.transient_bytes)?;
        dict.set_item("python_overhead_bytes", self.python_overhead_bytes)?;
        dict.set_item("total_bytes", self.total_bytes)?;
        Ok(dict)
    }
}

/// Returns `true` if `SCX_LOADER_PROFILE` is set to `"1"` or `"true"`.
///
/// Shared helper for the env-var check used across `pipeline.rs`,
/// `decode_stage.rs`, `io_stage.rs`, and `index_plan.rs`. Note this is not
/// the sole reader of the variable: some call sites (e.g. the inner
/// `spawn_blocking` path in `io_stage.rs`) re-read `SCX_LOADER_PROFILE`
/// inline rather than calling this function.
pub(crate) fn profiling_enabled() -> bool {
    std::env::var("SCX_LOADER_PROFILE")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breakdown_total_is_saturating_sum() {
        let b = BudgetBreakdown::new(100, 200, 30, 40, 50);
        assert_eq!(b.total_bytes, 100 + 200 + 30 + 40 + 50);
    }

    #[test]
    fn breakdown_fits_within_threshold() {
        let b = BudgetBreakdown::new(0, 0, 0, 0, 1024 * 1024); // 1 MiB
        assert!(b.fits_within(1));
        assert!(b.fits_within(2));
    }

    #[test]
    fn breakdown_does_not_fit_above_threshold() {
        let b = BudgetBreakdown::new(0, 0, 0, 0, 2 * 1024 * 1024); // 2 MiB
        assert!(!b.fits_within(1));
    }

    #[test]
    fn breakdown_handles_overflow_safely() {
        let b = BudgetBreakdown::new(usize::MAX, 1, 0, 0, 0);
        assert_eq!(b.total_bytes, usize::MAX);
    }
}
