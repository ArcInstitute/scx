// Rust 1.98's `clippy::chunks_exact_to_as_chunks` fires here on code this PR
// does not touch; see the crate-root note in `scx-codec/src/lib.rs` for why it
// is suppressed rather than rewritten, and who owns the rewrite.
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod append;
pub mod build_csc;
pub mod carry;
pub mod checksum;
pub mod codec_intent;
pub mod compact;
pub mod delete;
pub mod error;
pub mod external_layer;
pub mod external_obs;
pub mod flock;
pub mod group_plan;
pub mod helpers;
mod in_place;
pub mod merge;
pub mod merge_options;
mod merge_pairwise;
mod merge_sorted;
pub mod modify_metadata;
pub mod optimize;
pub mod predicate_index;
pub mod rebuild_csc;
pub mod rewrite_helpers;
pub mod rollback;
pub mod shuffle_order;
pub mod sort;
pub mod sort_engine;

#[cfg(test)]
mod test_utils;

pub use append::{
    append, append_from_reader, append_from_reader_with_index_options, append_with_index_options,
    AppendOptions,
};
pub use build_csc::run_build_csc;
pub use codec_intent::{framing_for_rewrite, intent_from_codec_selection, seed_codec};
pub use compact::{compact, compact_with_index_options, compact_with_options, CompactOptions};
// Re-exported from `scx-format-io`, which is where the obs shard-boundary loop
// now lives so `scx-mtx` (which cannot depend on this crate) shares it too.
pub use delete::mark_deleted;
pub use error::{OpsError, Result};
pub use external_layer::{
    attach_external_layer, axis_index_alias, display_key_name, resolve_key_alias,
    AttachLayerOptions, AttachLayerSummary, ColumnAxisMatch, ColumnAxisPolicy, ExternalLayerData,
    ExtraRowPolicy, MissingRowPolicy, ShardRangeSource,
};
pub use external_obs::{
    attach_external_obs, build_composite_key, diagnose_obs_key, obs_key_values,
    resolve_obs_key_column, AttachObsOptions, AttachObsSummary, ExternalObsData, KeyDiagnosis,
    ObsJoinKey, ObsRewrite, COMPOSITE_KEY_SEPARATOR,
};
pub use group_plan::{plan_group_shards, GroupPlan, GroupRecord, Role};
pub use merge::{merge, merge_with_index_options, merge_with_options};
pub use merge_options::{MergeOptions, UnsPolicy};
pub use modify_metadata::{modify_metadata, set_uns, MetadataPatch, ModifyMetadataSummary};
pub use optimize::{optimize, optimize_with_framing, OptimizeStats};
pub use predicate_index::PredicateIndexBuildSummary;
pub use rebuild_csc::{framing_for_csc_rebuild, rebuild_csc_inplace};
pub use rewrite_helpers::{
    codec_for_canonicalized, copy_auxiliary_sections, copy_auxiliary_sections_canonicalizing,
    copy_obs_var_preserving_layout, encoding_for_canonicalized, LayerCanonicalization,
};
pub use rollback::{rollback, rollback_to};
pub use scx_format_io::write_obs_section;
pub use shuffle_order::seeded_permutation;
pub use sort::{ReferenceSpec, SortOptions, SortStrategy, SortSummary};
pub use sort_engine::{
    compute_grouped_order, sort, sort_with_strategy, GroupedOrder, GROUP_BYTES_PER_NNZ,
    GROUP_MAX_BYTES_MULTIPLE,
};
