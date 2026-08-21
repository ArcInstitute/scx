//! h5ad / h5mu / 10x ingest and the SCX -> h5ad export drivers.
//!
//! This was one 3727-line file -- the largest production file in the workspace --
//! until ORG-11.16-6. Nine concerns were interleaved behind `// ====` banners; the
//! table below is the seam. Submodules are private and everything reachable at
//! `pipeline::<name>` before is re-exported here at the SAME visibility, so no
//! caller inside or outside the crate changed a path.
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
//! budget *buys* is `crate::budget` (ORG-11.16-5). Neither is a submodule here --
//! both are crate-level policy that the non-hdf5 build paths also name.

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

// `IngestOptions` lives in `crate::options` beside `ExportOptions` (ORG-11.16-3);
// re-exported here so `scx_convert::pipeline::IngestOptions` still resolves.
pub use crate::options::IngestOptions;

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

/// Per-modality density assumptions used by the
/// `IndexedCsrShardStream::per_worker_bytes` default impl. The
/// dispatcher uses this estimate to derate workers under
/// `memory_budget`. Over-estimating routes the convert to the
/// sequential coordinator (safe failure mode), so values err
/// conservative. The dense reader overrides `per_worker_bytes`
/// entirely; these constants only affect sparse readers.
pub(crate) use crate::budget::{PARALLEL_DENSITY_ATAC_DEN, PARALLEL_DENSITY_DEFAULT_DEN};

/// Build the `codec_selection` provenance value from a write's codec choice.
/// Shared by the streaming coordinators and the pyscx in-memory writer so the
/// stamp is identical across paths. `decode_target` is the internal mechanism
/// behind the adaptive intent profiles: `Auto` → `auto`, `Storage` → `compact`.
/// `None` (no adaptive bias) stamps `fast` (or the explicit codec name).
pub fn codec_selection_json(
    codec: Option<scx_codec::CodecId>,
    codec_trial: bool,
    decode_target: Option<scx_format_io::DecodeTarget>,
) -> serde_json::Value {
    use scx_format_io::DecodeTarget;
    let profile = if let Some(dt) = decode_target {
        match dt {
            DecodeTarget::Auto => "auto",
            DecodeTarget::Storage => "compact",
        }
    } else if codec_trial {
        "compact-trial"
    } else {
        // No adaptive bias and no explicit codec → heuristic single-encode
        // (the `fast` profile). An explicit codec stamps its own name.
        codec.map(|c| c.display_name()).unwrap_or("fast")
    };
    serde_json::json!({ "profile": profile })
}

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
