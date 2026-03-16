pub mod catalog;
pub mod checksum;
pub mod error;
pub mod header;
pub mod provenance;
pub mod reader;
pub mod section;
pub mod shard;
pub mod writer;

pub use catalog::{
    FullCatalog, FullCatalogEntry, RootCatalog, RootCatalogEntry, ShardStats,
    ROOT_CATALOG_ENTRY_SIZE, ROOT_CATALOG_MAX_SIZE, SHARD_STATS_BASE_SIZE,
};
pub use checksum::{blake3_hash, blake3_truncated_64};
pub use error::{Result, ScxError};
pub use header::{FileHeader, HEADER_SIZE, MAGIC};
pub use provenance::{Provenance, ProvenanceEntry};
pub use section::{align_to_8, SectionType};
pub use shard::{
    BlockIndex, BlockIndexEntry, ShardHeader, BLOCK_INDEX_ENTRY_SIZE, SHARD_HEADER_SIZE,
    SHARD_MAGIC,
};
pub use reader::ScxReader;
pub use writer::{ScxWriter, SECTIONS_START_OFFSET};
