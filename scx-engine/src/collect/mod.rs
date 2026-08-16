//! Pipeline execution engine.
//!
//! Ties together pushdown, decode, projection, filtering, and fused operations
//! to execute a `QueryPipeline`. Called by `QueryPipeline::collect()`.
//!
//! Three layers, bottom up:
//!
//! | Module | Holds |
//! |---|---|
//! | `retry` | `par_map_with_shard_retry` — generic resilient parallel map, nothing query-specific |
//! | `rows` | `filter_csr_rows` — row selection inside one decoded shard |
//! | `plan` | the `ExecutionPlan`: catalog-level (Level-1) shard pruning and the category dictionaries |
//! | `mask` | the row-set fast path, the legacy full-decode fallback, and the fork between them |
//! | `execute` | the public entry points and the X-shard decode they drive |
//!
//! The layering exists so that `mask` is the *only* module that decides
//! row-set-pushdown vs legacy-full-decode. `execute` consumes a `MaskResult`
//! without knowing which arm produced it, which is what makes that fork an
//! interface rather than a branch buried mid-file.
//!
//! The submodules are private; everything reachable at `collect::<name>`
//! before the split is re-exported here at the same visibility.

mod execute;
mod mask;
mod plan;
mod retry;
mod rows;

pub use execute::{count, execute, exists};
pub use rows::filter_csr_rows;

pub(crate) use execute::materialize_filtered_obs;
pub(crate) use mask::obs_shard_ranges_from_catalog;
pub(crate) use plan::scan_shards;

// Reached only from `collect_tests.rs`, which is attached to this module and
// so cannot see the submodules' items directly. Gated on `cfg(test)` so that a
// non-test caller of a helper widened purely for a test fails to compile
// rather than quietly acquiring a crate-wide dependency on it.
#[cfg(test)]
pub(crate) use execute::{materialize, plan_and_mask};
#[cfg(test)]
pub(crate) use plan::{build_plan, collect_bitset_coverage};

/// Env var (diagnostic only) that forces the legacy full-decode obs path,
/// disabling row-set predicate pushdown.
///
/// Read **once per pipeline**, at construction, into
/// [`QueryPipeline::rowset_pushdown`](crate::QueryPipeline::rowset_pushdown)'s
/// backing field — not once per query. That field is the only thing
/// [`mask::compute_mask`] consults, so nothing has to mutate the process
/// environment to exercise the legacy path; `tests/rowset_differential.rs`
/// sets the field.
pub(crate) fn rowset_pushdown_disabled_by_env() -> bool {
    std::env::var_os("SCX_DISABLE_ROWSET_PUSHDOWN").is_some()
}

#[cfg(test)]
#[path = "../collect_tests.rs"]
mod tests;
