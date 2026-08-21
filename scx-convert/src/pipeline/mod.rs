//! h5ad / h5mu / 10x ingest and the SCX -> h5ad export drivers.
//!
//! This was one 3727-line file -- the largest production file in the workspace --
//! until ORG-11.16-6, holding nine concerns with no section markers and no module
//! doc comment: line 1 was a bare `use`. The table below is the seam. Submodules
//! are private; the re-export block below preserves every `pipeline::<name>` path
//! that had a caller, so nothing outside `scx-convert` changed. It is **not** a
//! same-visibility carve -- five internal paths were deliberately dropped, and
//! they are enumerated below rather than glossed.
//!
//! | Module            | Holds                                                     |
//! |-------------------|-----------------------------------------------------------|
//! | `error`           | [`ConvertError`] and its `DrainFailure` conversion         |
//! | `bitmap`          | detection-bitmap eligibility and the per-shard writer      |
//! | `threads`         | reader-thread derating against `crate::budget`'s shares    |
//! | `index`           | convert-time predicate-index build and its outcomes        |
//! | `entry`           | the eager entry points: h5ad, 10x, SCX -> h5ad             |
//! | `entry_streaming` | streaming h5ad ingest and its grouped two-pass sibling     |
//! | `coordinator`     | the four shard coordinators and the encode worker          |
//! | `shards`          | X / raw / layer / CSC shard writers                        |
//! | `mappings`        | obsm / varm / obsp / varp section writers                  |
//!
//! The direction-specific option types are [`IngestOptions`] and
//! [`crate::ExportOptions`], both in `crate::options` (ORG-11.16-3); what a memory
//! budget *buys* is `crate::budget` (ORG-11.16-5). Neither is a submodule here,
//! for different reasons: `options` is hdf5-gated exactly like this module but is
//! policy rather than pipeline, and keeping the ingest/export pair in one file is
//! the whole point of ORG-11.16-3; `budget` is **ungated**, so its arithmetic
//! invariants run in the default test job and `csc_sidecar_bytes` stays reachable
//! from the non-hdf5 `pyscx.from_anndata` sidecar path.

//! ### Not a "same visibility" carve, and the difference is worth naming
//!
//! Five items were reachable in production at `crate::pipeline::*` before the
//! split and are not now: `BitmapBuildOutcome`, `maybe_build_bitmap_shard` and
//! `ensure_shard_fits_budget` have no re-export at all, and
//! `compute_shard_row_ranges` / `process_predicate_index_outcomes` are re-exported
//! only under `cfg(test)`. Nothing outside `pipeline` called any of them, so no
//! caller broke -- but the internal `pub(crate)` surface was **deliberately
//! reduced**, which is a narrowing and not merely a move.

mod bitmap;
mod coordinator;
mod entry;
mod entry_streaming;
mod error;
mod index;
mod mappings;
mod shards;
mod threads;

// Re-exported at the visibility each item already had. `ensure_shard_fits_budget`,
// `maybe_build_bitmap_shard` and `BitmapBuildOutcome` are deliberately absent:
// nothing outside `pipeline` names them, and re-exporting an item no one reaches
// is how a module surface grows without anyone deciding it should.
pub use coordinator::{run_streaming_writer_coordinator, streaming_writer_coordinator};
pub use entry::{h5ad_to_scx, scx_to_h5ad, scx_to_h5ad_streaming, tenx_to_scx};
pub use entry_streaming::{h5ad_to_scx_streaming, StreamingOverrides};
pub use error::ConvertError;

pub(crate) use threads::{derate_threads_and_depth, resolve_reader_threads};

// Reached only from `convert_tests_parallel.rs` / `convert_tests_dataframe.rs`,
// which are attached to this module and so cannot see the submodules' items
// directly. Gated on `cfg(test)` so a non-test caller of a helper widened purely
// for a test fails to compile rather than quietly acquiring a crate-wide
// dependency on it. (The submodules that genuinely call these reach them as
// `super::coordinator::…` / `super::index::…`, not through here.)
#[cfg(test)]
pub(crate) use coordinator::compute_shard_row_ranges;
#[cfg(test)]
pub(crate) use index::process_predicate_index_outcomes;

// `IngestOptions` and `codec_selection_json` live in `crate::options` beside
// `ExportOptions` (ORG-11.16-3); re-exported here so
// `scx_convert::pipeline::{IngestOptions, codec_selection_json}` still resolve.
// `codec_selection_json` is pure option policy whose only caller is
// `IngestOptions::codec_selection_value`; leaving it here made `options` import
// `pipeline` while `pipeline` re-exported `options`, which contradicted the
// one-way dependency this module doc claims.
pub use crate::options::{codec_selection_json, IngestOptions};

use arrow::record_batch::RecordBatch;
use scx_format_io::writer::ScxWriter;

/// Detection-bitmap generation policy.
///
/// Re-exported from [`scx_format_io::BitmapPolicy`] so callers that depend
/// on `scx-convert` (CLI, pyscx with hdf5) can name it without an
/// extra `scx_format` import. The actual definition lives in
/// `scx-format` so the CPU-only pyscx build (which doesn't pull in
/// `scx-convert`) can still drive bitmap generation from its in-memory
/// write path.
pub use scx_format_io::BitmapPolicy;
/// Re-exported from [`scx_format_io::CscPolicy`] so callers depending only on
/// `scx-convert` get the CSC policy type without an explicit `scx-format` dep.
pub use scx_format_io::CscPolicy;

/// Write the obs table for an ingest, sharded or not per
/// [`IngestOptions::obs_shard_policy`].
///
/// Every `scx-convert` ingest path funnels through here — the eager and
/// streaming h5ad routes, 10x, and both h5mu routes — so the threshold is
/// decided once and the shard boundaries come from
/// [`scx_format_io::write_obs_section`] — the same routine `scx compact
/// --reshape-obs`, `scx optimize --shard-obs` and `scx-mtx` use. Convert's sharded obs is therefore
/// layout-identical to theirs by construction rather than by review.
///
/// The var axis deliberately does not have a counterpart: it stays a single
/// section on every ingest path (organization phase 6c is obs-scoped, matching
/// `ObsShardPolicy`'s own scope).
///
/// This does **not** lower peak memory. `obs` is already resident and
/// `RecordBatch::slice` is zero-copy; what sharding buys is the bounded layout
/// for downstream streaming / cloud / export readers, and — the reason the
/// phase exists — putting the multi-shard h5ad dataframe writer on the path
/// that `scx convert` output actually takes.
pub(crate) fn write_ingest_obs(
    writer: &mut ScxWriter,
    obs: &RecordBatch,
    opts: &IngestOptions,
) -> Result<(), ConvertError> {
    let reshape = opts
        .obs_shard_policy
        .should_shard_single_section(obs.num_rows() as u64, opts.shard_target_rows);
    Ok(scx_format_io::write_obs_section(
        writer,
        obs,
        reshape,
        opts.shard_target_rows,
    )?)
}

#[cfg(test)]
#[path = "../pipeline_test_hooks.rs"]
pub(crate) mod test_hooks;
