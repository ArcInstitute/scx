pub mod arrow_compat;
pub mod backed;
#[cfg(feature = "deletion-vectors")]
pub mod bitmap;
pub mod catalog;
pub mod catalog_view;
pub mod checksum;
pub mod codec_select;
pub mod csc_policy;
#[cfg(feature = "deletion-vectors")]
pub mod deletion_vectors;
pub mod encoder;
pub mod error;
pub mod header;
pub mod mem;
pub mod modality;
pub mod provenance;
pub mod reader;
pub mod section;
pub mod shard;
pub mod shard_decode;
pub mod shard_source;
pub mod writer;

pub use arrow_compat::{
    downcast_large_types, downcast_large_types_schema, ensure_pandas_index_metadata,
    pandas_index_columns, upcast_to_large_types,
};
pub use backed::{
    concatenate_csr, total_variance_from_col_sq, BackedCscIndex, BackedCscReader, BackedCsrIndex,
    BackedCsrReader, BackedDenseReader, CacheMetrics,
};
#[cfg(feature = "deletion-vectors")]
pub use bitmap::{
    BitmapPolicy, BitmapShard, BITMAP_ORIENTATION_GENE_TO_ROWS, BITMAP_SHARD_MAGIC,
    BITMAP_SHARD_VERSION,
};
#[allow(deprecated)]
pub use catalog::SHARD_STATS_BASE_SIZE;
pub use catalog::{
    column_name_hash, ColumnStat, FullCatalog, FullCatalogEntry, LazyShardStats, RootCatalog,
    RootCatalogEntry, ShardStats, CURRENT_CATALOG_VERSION, ROOT_CATALOG_ENTRY_SIZE,
    ROOT_CATALOG_MAX_SIZE, SHARD_STATS_BASE_SIZE_V1, SHARD_STATS_BASE_SIZE_V2,
};
pub use catalog_view::{CatalogView, CatalogViewEntry, ShardStatsLite};
pub use checksum::{blake3_hash, blake3_truncated_64};
pub use codec_select::{
    select_codec, select_codec_for_modality, select_codec_with_profile, CodecProfile,
};
pub use csc_policy::{
    auto_obs_threshold, auto_vars_threshold, CscPolicy, AUTO_CSC_OBS_THRESHOLD,
    AUTO_CSC_VARS_THRESHOLD,
};
#[cfg(feature = "deletion-vectors")]
pub use deletion_vectors::{DeletionVectors, ShardDeletion};
pub use encoder::encode_one_shard;
pub use error::{validate_allocation, Result, ScxError};
pub use header::{FileHeader, CURRENT_FORMAT_VERSION, HEADER_SIZE, MAGIC};
pub use mem::MemoryBudget;
pub use modality::{
    ModalityFlags, ModalityInfo, ModalityTable, ModalityType, MAX_MODALITIES,
    MODALITY_NAME_MAX_BYTES, MODALITY_TABLE_MAGIC, MODALITY_TABLE_VERSION,
};
pub use provenance::{Provenance, ProvenanceEntry};
pub use reader::{assemble_sharded_metadata, ScxReader};
pub use section::{align_to_8, SectionType};
pub use shard::{
    derive_shard_type, BlockIndex, BlockIndexEntry, ShardHeader, BLOCK_INDEX_ENTRY_SIZE,
    SHARD_HEADER_SIZE, SHARD_MAGIC,
};
pub use shard_decode::decode_shard_bytes;
pub use shard_source::{ColumnShardSource, ShardSource};
pub use writer::{
    chmod_to_umask, compute_shard_stats, fsync_parent_dir, make_sibling_tempfile, MajorAxis,
    PreEncodedSection, ScxWriter, SECTIONS_START_OFFSET,
};

/// Default number of rows per CSR shard when callers don't override it.
///
/// 16 384 is a power of two (aligns with typical GPU batch sizes and
/// memory page boundaries) and matches the `ScxWriter` test-fixture
/// default. Used by the pyscx ops bindings and the `scx convert` /
/// `scx append` / `scx subset` CLI subcommands so identical inputs
/// through CLI and Python produce identical shard layouts.
pub const DEFAULT_SHARD_TARGET_ROWS: u32 = 16384;
