pub mod arrow_compat;
pub mod backed;
pub mod catalog;
pub mod checksum;
pub mod codec_select;
#[cfg(feature = "deletion-vectors")]
pub mod deletion_vectors;
pub mod error;
pub mod header;
pub mod modality;
pub mod provenance;
pub mod reader;
pub mod section;
pub mod shard;
pub mod shard_source;
pub mod writer;

pub use arrow_compat::{downcast_large_types, upcast_to_large_types};
pub use backed::{
    concatenate_csr, total_variance_from_col_sq, BackedCscIndex, BackedCscReader, BackedCsrIndex,
    BackedCsrReader, CacheMetrics,
};
#[allow(deprecated)]
pub use catalog::SHARD_STATS_BASE_SIZE;
pub use catalog::{
    column_name_hash, ColumnStat, FullCatalog, FullCatalogEntry, RootCatalog, RootCatalogEntry,
    ShardStats, CURRENT_CATALOG_VERSION, ROOT_CATALOG_ENTRY_SIZE, ROOT_CATALOG_MAX_SIZE,
    SHARD_STATS_BASE_SIZE_V1, SHARD_STATS_BASE_SIZE_V2,
};
pub use checksum::{blake3_hash, blake3_truncated_64};
pub use codec_select::{
    select_codec, select_codec_for_modality, select_codec_with_profile, CodecProfile,
};
#[cfg(feature = "deletion-vectors")]
pub use deletion_vectors::{DeletionVectors, ShardDeletion};
pub use error::{Result, ScxError};
pub use header::{FileHeader, CURRENT_FORMAT_VERSION, HEADER_SIZE, MAGIC};
pub use modality::{
    ModalityFlags, ModalityInfo, ModalityTable, ModalityType, MAX_MODALITIES,
    MODALITY_NAME_MAX_BYTES, MODALITY_TABLE_MAGIC, MODALITY_TABLE_VERSION,
};
pub use provenance::{Provenance, ProvenanceEntry};
pub use reader::ScxReader;
pub use section::{align_to_8, SectionType};
pub use shard::{
    derive_shard_type, BlockIndex, BlockIndexEntry, ShardHeader, BLOCK_INDEX_ENTRY_SIZE,
    SHARD_HEADER_SIZE, SHARD_MAGIC,
};
pub use shard_source::{ColumnShardSource, ShardSource};
pub use writer::{
    chmod_to_umask, compute_shard_stats, fsync_parent_dir, make_sibling_tempfile, MajorAxis,
    PreEncodedSection, ScxWriter, SECTIONS_START_OFFSET,
};
