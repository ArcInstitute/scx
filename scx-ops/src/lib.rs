pub mod append;
pub mod build_csc;
pub mod checksum;
pub mod compact;
pub mod delete;
pub mod error;
pub mod flock;
pub mod helpers;
mod in_place;
pub mod merge;
pub mod merge_options;
mod merge_sorted;
pub mod modify_metadata;
pub mod optimize;
pub mod predicate_index;
pub mod rebuild_csc;
pub mod rewrite_helpers;
pub mod rollback;
pub mod sort;
pub mod sort_engine;

#[cfg(test)]
mod test_utils;

pub use append::{
    append, append_from_reader, append_from_reader_with_index_options, append_with_index_options,
    AppendOptions,
};
pub use build_csc::run_build_csc;
pub use compact::{compact, compact_with_index_options};
pub use delete::mark_deleted;
pub use error::{OpsError, Result};
pub use merge::{merge, merge_with_index_options, merge_with_options};
pub use merge_options::{MergeOptions, UnsPolicy};
pub use modify_metadata::{modify_metadata, set_uns, MetadataPatch};
pub use optimize::optimize;
pub use predicate_index::PredicateIndexBuildSummary;
pub use rebuild_csc::rebuild_csc_inplace;
pub use rewrite_helpers::copy_auxiliary_sections;
pub use rollback::{rollback, rollback_to};
pub use sort::{SortOptions, SortStrategy, SortSummary};
pub use sort_engine::{sort, sort_with_strategy};
