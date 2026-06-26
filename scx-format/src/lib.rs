//! On-disk layout and spec for the SCX format: file header, catalogs, shard
//! structs, modality table, provenance, codec selection, and the error/checksum
//! primitives. This crate is **pure** — no `std::fs`, no `memmap2`, no decode
//! dispatch. The runtime reader/writer and backed/streaming access live in the
//! `scx-format-io` crate, which depends on this one. This is the surface an
//! independent reader implementation and the format conformance vectors verify
//! against.

pub mod catalog;
pub mod catalog_view;
pub mod checksum;
pub mod codec_select;
pub mod csc_policy;
pub mod error;
pub mod group_index;
pub mod header;
pub mod modality;
pub mod provenance;
pub mod section;
pub mod shard;
pub mod versioned;

/// Arrow `Field::metadata` key marking a dictionary column as an *ordered*
/// categorical (R `ordered` factor / pandas ordered Categorical). Canonical
/// home shared by every binding (scx-convert re-exports it; pyscx and rscx
/// read it from here) so the wire key has a single definition.
pub const CATEGORICAL_ORDERED_KEY: &str = "scx.categorical.ordered";

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
pub use error::{validate_allocation, Result, ScxError, ScxErrorClass};
pub use group_index::{GroupIndexPayload, GroupRecordWire};
pub use header::{
    rewrite_output_format_version, FileHeader, CURRENT_FORMAT_VERSION, HEADER_SIZE, MAGIC,
};
pub use modality::{
    ModalityFlags, ModalityInfo, ModalityTable, ModalityType, MAX_MODALITIES,
    MODALITY_NAME_MAX_BYTES, MODALITY_TABLE_MAGIC, MODALITY_TABLE_VERSION,
};
pub use provenance::{Provenance, ProvenanceEntry};
pub use section::{align_to_8, SectionType};
pub use shard::{
    derive_shard_type, BlockIndex, BlockIndexEntry, ShardHeader, BLOCK_INDEX_ENTRY_SIZE,
    SHARD_HEADER_SIZE, SHARD_MAGIC,
};
pub use versioned::VersionedSection;

/// Default number of rows per CSR shard when callers don't override it.
///
/// 16 384 is a power of two (aligns with typical GPU batch sizes and
/// memory page boundaries) and matches the `ScxWriter` test-fixture
/// default. Used by the pyscx ops bindings and the `scx convert` /
/// `scx append` / `scx subset` CLI subcommands so identical inputs
/// through CLI and Python produce identical shard layouts.
pub const DEFAULT_SHARD_TARGET_ROWS: u32 = 16384;
