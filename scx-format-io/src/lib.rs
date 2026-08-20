//! Runtime I/O for the SCX format.
//!
//! This crate holds everything that touches the filesystem or decodes bytes:
//! the [`ScxReader`]/[`ScxWriter`], the backed/streaming readers, shard
//! encode/decode dispatch, and the CSC sidecar. The pure, no-I/O on-disk *layout*
//! (header, catalog, shard structs, modality, codec selection) lives in the
//! [`scx_format`] crate, which this crate depends on.
//!
//! For convenience the entire [`scx_format`] public surface is re-exported here,
//! so a downstream crate that depends on `scx-format-io` can reach both the
//! layout types and the runtime through `scx_format_io::…` paths. `scx-format`
//! has no `pub(crate)` items, so every layout item this crate's modules reference
//! via `crate::…` resolves through these re-exports.

pub use scx_format::*;

pub mod arrow_compat;
pub mod backed;
#[cfg(feature = "deletion-vectors")]
pub mod bitmap;
pub mod categorical;
pub mod csc_sidecar;
#[cfg(feature = "deletion-vectors")]
pub mod deletion_vectors;
pub mod distinct;
pub mod encoder;
pub mod freshness;
pub mod mem;
pub mod prefetch;
pub mod profile;
pub mod reader;
pub mod shard_decode;
pub mod shard_report;
pub mod shard_source;
pub mod typed_read;
pub(crate) mod validated_section;
pub mod writer;

pub use arrow_compat::{
    downcast_large_types, downcast_large_types_schema, ensure_pandas_index_metadata,
    pandas_index_columns, resolve_index_columns, upcast_to_large_types,
};
pub use backed::{
    BackedCscIndex, BackedCscReader, BackedCsrIndex, BackedCsrReader, BackedDenseReader,
    CacheMetrics, ShardCache, SharedShardCache, SizeHint,
};
#[cfg(feature = "deletion-vectors")]
pub use bitmap::{
    BitmapPolicy, BitmapShard, BITMAP_ORIENTATION_GENE_TO_ROWS, BITMAP_SHARD_MAGIC,
    BITMAP_SHARD_VERSION,
};
#[cfg(feature = "deletion-vectors")]
pub use categorical::GlobalCategoryAccum;
#[cfg(feature = "deletion-vectors")]
pub use deletion_vectors::DeletionVectors;
pub use distinct::DistinctAccumulator;
pub use encoder::{
    encode_one_shard, encode_one_shard_from_bytes, encode_one_shard_with_value_encoding,
    encode_shard_adaptive, encode_shard_framed, FramingConfig, DEFAULT_ROW_GROUP_ROWS,
};
pub use mem::MemoryBudget;
pub use prefetch::{
    accumulate_shards, clamp_prefetch_depth, col_means_and_sum_sq_prefetched,
    for_each_csc_shard_ordered, for_each_shard_ordered, for_each_shard_ordered_uncached,
    prefetch_depth, reduce_shards_budgeted, reduction_mode, PrefetchError, ReductionMode,
    DEFAULT_PREFETCH_DEPTH,
};
pub use profile::{
    record_decode_since, record_io_since, record_marshalling_since, record_reduction_since,
    reduction_guard, CodecClass, CpuProfileSnapshot, ReductionGuard, StageStat,
};
pub use reader::{
    assemble_filtered_metadata, assemble_sharded_metadata, compact_key_shard,
    decode_arrow_ipc_schema, ScxReader,
};
pub use shard_decode::{
    decode_shard_bytes, decode_shard_bytes_native, decode_shard_indptr_bytes,
    decode_shard_regions_native, decode_shard_regions_scipy,
};
pub use shard_report::{codec_id_histogram, distinct_sorted_shard_field, total_shard_bytes};
pub use shard_source::{ColumnShardSource, ShardSizeHint, ShardSource};
pub use writer::{
    assign_csr_shard_column_stats, chmod_to_umask, clear_all_csr_shard_column_stats,
    clear_csr_shard_column_stats_for, compute_shard_stats, fsync_parent_dir, make_sibling_tempfile,
    write_obs_section, MajorAxis, PreEncodedSection, ScxWriter, SECTIONS_START_OFFSET,
};
