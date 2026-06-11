//! Runtime I/O for the SCX format.
//!
//! This crate holds everything that touches the filesystem or decodes bytes:
//! the [`ScxReader`]/[`ScxWriter`], the backed/streaming readers, shard
//! encode/decode dispatch, and the sidecars. The pure, no-I/O on-disk *layout*
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
pub mod csc_sidecar;
pub mod decode_sidecar;
#[cfg(feature = "deletion-vectors")]
pub mod deletion_vectors;
pub mod encoder;
pub mod mem;
pub mod reader;
pub mod shard_decode;
pub mod shard_source;
pub(crate) mod validated_section;
pub mod writer;

pub use arrow_compat::{
    downcast_large_types, downcast_large_types_schema, ensure_pandas_index_metadata,
    pandas_index_columns, upcast_to_large_types,
};
pub use backed::{
    BackedCscIndex, BackedCscReader, BackedCsrIndex, BackedCsrReader, BackedDenseReader,
    CacheMetrics,
};
#[cfg(feature = "deletion-vectors")]
pub use bitmap::{
    BitmapPolicy, BitmapShard, BITMAP_ORIENTATION_GENE_TO_ROWS, BITMAP_SHARD_MAGIC,
    BITMAP_SHARD_VERSION,
};
pub use decode_sidecar::{
    decode_scx1_parallel, DecodeRowEntry, DecodeSidecar, RiceBlockEntry,
    DECODE_SIDECAR_KIND_SCX1_CSR, DECODE_SIDECAR_MAGIC, DECODE_SIDECAR_VERSION,
    DEFAULT_DECODE_SIDECAR_MAX_OVERHEAD_RATIO,
};
#[cfg(feature = "deletion-vectors")]
pub use deletion_vectors::{DeletionVectors, ShardDeletion};
pub use encoder::encode_one_shard;
pub use mem::MemoryBudget;
pub use reader::{assemble_filtered_metadata, assemble_sharded_metadata, ScxReader};
pub use shard_decode::decode_shard_bytes;
pub use shard_source::{ColumnShardSource, ShardSource};
pub use writer::{
    assign_csr_shard_column_stats, chmod_to_umask, compute_shard_stats, fsync_parent_dir,
    make_sibling_tempfile, MajorAxis, PreEncodedSection, ScxWriter, SECTIONS_START_OFFSET,
};
