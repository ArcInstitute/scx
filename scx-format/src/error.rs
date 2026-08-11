use thiserror::Error;

/// Errors produced by the SCX format crate.
#[derive(Debug, Error)]
pub enum ScxError {
    #[error("invalid file magic bytes")]
    InvalidMagic,

    #[error("unsupported format version")]
    UnsupportedVersion,

    #[error("unsupported format version: found {found}, this build supports 1..={max_supported}")]
    UnsupportedFormatVersion { found: u16, max_supported: u16 },

    #[error("unsupported endianness (only little-endian is supported)")]
    UnsupportedEndian,

    #[error("checksum mismatch in section '{section}'")]
    ChecksumMismatch { section: String },

    #[error("invalid shard magic bytes")]
    InvalidShardMagic,

    #[error("unknown codec ID: {0}")]
    UnknownCodec(u8),

    #[error("unknown value encoding: {0}")]
    UnknownValueEncoding(u8),

    #[error("inconsistent CSR array lengths")]
    InconsistentCsr,

    #[error("root catalog exceeds 4096 byte limit: {0} bytes")]
    RootCatalogTooLarge(usize),

    #[error("unknown section type: {0}")]
    UnknownSectionType(u8),

    #[error("section not found: {0}")]
    SectionNotFound(String),

    // Shared by `distinct_values` and `obs_categorical` — both require a
    // string/categorical column — so the message names neither.
    #[error(
        "column '{column}' has unsupported type {dtype}; only string/categorical \
         columns (Utf8, LargeUtf8, Dictionary) are supported"
    )]
    UnsupportedColumnType { column: String, dtype: String },

    #[error("shard index {index} out of bounds (count: {count})")]
    ShardIndexOutOfBounds { index: usize, count: usize },

    #[error("invalid catalog: {0}")]
    InvalidCatalog(String),

    #[error(
        "duplicate section: (modality_id={modality_id}, section_type={section_type:?}, \
         name='{name}') was already written; readers resolve names to the first match"
    )]
    DuplicateSection {
        name: String,
        section_type: String,
        modality_id: u8,
    },

    #[error("unsupported {section} version: found {found}, expected {expected}")]
    UnsupportedSectionVersion {
        section: &'static str,
        found: u16,
        expected: u16,
    },

    #[error("bitmap shard gene_id {gene_id} out of range (n_vars = {n_vars})")]
    BitmapGeneIdOutOfRange { gene_id: u32, n_vars: u32 },

    #[error(
        "stale CSC sidecar: built against data generation {built_generation} but the file is \
         now at data generation {data_generation}; the column-major sidecar no longer matches \
         the CSR data — rebuild it with `scx build-csc` (or `--rebuild-csc`)"
    )]
    StaleCscSidecar {
        built_generation: u64,
        data_generation: u64,
    },

    /// A handle that was opened with change-watching enabled was asked to read
    /// a file that has changed since it was opened.
    ///
    /// Nothing is wrong with either file — the handle's mapping is intact and
    /// still perfectly readable. That is precisely the problem: it maps the
    /// *previous* contents (an in-place op appends and rewrites the header) or
    /// an unlinked inode (a copy-out op renames a new file into place), so
    /// every answer it gives is right about a file that is no longer there.
    ///
    /// Only handed to callers that opted in via `ScxReader::open_watched`; the
    /// ops / CLI / loader paths open readers around their own mutations and
    /// never see this.
    #[error(
        "'{path}' {detail}. This handle still maps the file as it was when it was opened, \
         so its answers describe contents that are no longer on disk — re-open the file to \
         read the current ones (in pyscx: `Experiment.reload()`)"
    )]
    FileChangedOnDisk { path: String, detail: String },

    #[error("block n_rows {0} exceeds u16::MAX (65535)")]
    BlockRowsOverflow(u32),

    #[error("block nnz {0} exceeds u32::MAX")]
    BlockNnzOverflow(u64),

    #[error("shard sub-stream too large for a u32 offset/length field: {0}")]
    ShardStreamTooLarge(String),

    #[error("malformed block index: {0}")]
    InvalidBlockIndex(String),

    /// A decoded minor-axis index (column for row-major shards, row for CSC)
    /// falls outside `[0, n_minor)`.
    ///
    /// The catalog's BLAKE3 covers catalog bytes, not shard payloads, so a
    /// decoded index is **unauthenticated** — see
    /// [`ScxError::AllocationTooLarge`] and `clamped_reserve` for the same
    /// reasoning applied to declared lengths. Every consumer of a decoded shard
    /// uses the index to address a buffer sized by `n_minor`; the readers
    /// validate it once at the decode seam so those consumers can index
    /// unchecked.
    ///
    /// `index` is reported as the decoded `u32`: a negative `i32` from a
    /// scipy-side decode reinterprets to a value `≥ 2³¹` here, which is how it
    /// is detected in the first place.
    #[error(
        "shard '{shard}': minor-axis index {index} at position {position} is out of range \
         for n_minor {n_minor} (a negative index reads as ≥ 2^31 here)"
    )]
    ShardIndexOutOfRange {
        shard: String,
        index: u32,
        position: usize,
        n_minor: u32,
    },

    /// A row-group-framed shard was asked for with zero rows.
    ///
    /// Framing emits one [`crate::BlockIndexEntry`] per row group, so a zero-row
    /// shard produces an **empty** block index — which `resolve_block_index`
    /// rejects on every read. Such a shard writes and checksums cleanly and then
    /// fails every read, so the writer refuses it instead. (Readers still accept
    /// an empty block index when the header agrees the shard is empty, so files
    /// written before this guard remain readable.)
    #[error(
        "refusing to row-group-frame a zero-row shard: framing would emit an empty block \
         index, which no reader accepts. Skip the shard instead of writing an empty one."
    )]
    ZeroRowFramedShard,

    #[error("n_vars {0} exceeds u32::MAX, cannot fit in shard header n_minor field")]
    NVarsOverflow(u64),

    #[error("section bytes out of bounds: offset {offset} + length {length} exceeds file size {file_size}")]
    SectionOutOfBounds {
        offset: u64,
        length: u64,
        file_size: usize,
    },

    #[error("writer has already been finished")]
    WriterAlreadyFinished,

    #[error("empty indptr array")]
    EmptyIndptr,

    #[error("column_stats count {0} exceeds u8::MAX (255)")]
    ColumnStatsOverflow(usize),

    #[error("per-shard column_stats count {got} != CSR shard count {expected}")]
    ColumnStatsShardCountMismatch { got: usize, expected: usize },

    #[error(
        "invalid shard_type byte {got} for section_type {section_type:#x} (expected {expected})"
    )]
    InvalidShardType {
        expected: u8,
        got: u8,
        section_type: u8,
    },

    #[error("CSR error: {0}")]
    Csr(#[from] scx_sparse::CsrError),

    #[error("codec error: {0}")]
    Codec(#[from] scx_codec::CodecError),

    #[error("CSC transpose failed: {0}")]
    CscTranspose(String),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    /// The `uns` section nests deeper than the JSON reader will parse.
    ///
    /// Raised on **both** sides of the section, which is why the message names
    /// neither a file nor a write:
    ///
    /// - [`crate::validate_uns_depth`], refusing to *store* a tree no reader
    ///   could take back — before any file exists;
    /// - [`crate::parse_uns_json`], on a file written before `uns` depth was
    ///   capped, when an uncapped writer could still emit one.
    ///
    /// Both exist because the JSON *serializer* has no depth limit while the
    /// deserializer stops at [`crate::SERDE_JSON_MAX_NESTING`]. Split out from
    /// [`ScxError::Json`] because the bare `recursion limit exceeded` that
    /// `serde_json` produces names neither the section nor the cause, and this
    /// is the one JSON error a user can act on.
    #[error(
        "uns nests deeper than the {max_nesting} levels the JSON reader accepts, so it can \
         neither be stored nor read back — flatten the metadata, or, for an existing file \
         written before uns depth was capped, rewrite it from its source"
    )]
    UnsTooDeep { max_nesting: usize },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("allocation too large: requested {requested} bytes but only {available} bytes remain in section")]
    AllocationTooLarge { requested: usize, available: usize },

    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),

    #[error(
        "cannot write '{attempted}' on a writer that already wrote '{existing}' for the same axis: \
         obs and var metadata must be either entirely single-section ({single_kind}) or entirely \
         sharded ({sharded_kind}), not mixed"
    )]
    ObsLayoutConflict {
        attempted: &'static str,
        existing: &'static str,
        single_kind: &'static str,
        sharded_kind: &'static str,
    },
}

/// Coarse semantic category of an [`ScxError`], used by the language
/// bindings to choose how to surface an error without each binding
/// re-matching every `ScxError` variant.
///
/// The categories intentionally collapse many variants into a handful of
/// buckets that map cleanly onto host-language error conventions
/// (e.g. Python `ValueError` / `FileNotFoundError` / `RuntimeError`).
/// Message wording remains the binding's responsibility — this only
/// decides the *kind* of failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScxErrorClass {
    /// Bad user input or internally inconsistent data (not a file-format
    /// corruption signal): inconsistent CSR, overflowing dimensions, a
    /// stale CSC sidecar.
    Validation,
    /// The file is corrupt or was written by an incompatible/newer SCX:
    /// bad magic, unsupported version/endianness/section version, checksum
    /// mismatch, or an invalid catalog.
    CorruptFile,
    /// An underlying I/O failure; forwards the [`std::io::ErrorKind`] so the
    /// binding can distinguish missing-file / permission / truncation cases.
    Io(std::io::ErrorKind),
    /// Anything not covered above; bindings should treat as a generic
    /// runtime failure.
    Other,
}

impl ScxError {
    /// Classify this error into a coarse [`ScxErrorClass`]. Used by the
    /// `pyscx` / `rscx` bindings so the variant→category decision lives in
    /// one place instead of being re-spelled per binding.
    ///
    /// The match is intentionally exhaustive (no wildcard arm) so adding a
    /// new [`ScxError`] variant is a compile error here until it is given a
    /// class. Guiding principle: malformed/unreadable on-disk data is
    /// [`ScxErrorClass::CorruptFile`]; bad caller input or a writer-side
    /// inconsistency is [`ScxErrorClass::Validation`]; wrapped lower-level
    /// errors and genuine runtime failures are [`ScxErrorClass::Other`].
    pub fn class(&self) -> ScxErrorClass {
        match self {
            // Bad input / internally inconsistent data (not file corruption).
            ScxError::InconsistentCsr
            | ScxError::NVarsOverflow(_)
            | ScxError::BlockRowsOverflow(_)
            | ScxError::BlockNnzOverflow(_)
            | ScxError::ShardStreamTooLarge(_)
            | ScxError::StaleCscSidecar { .. }
            | ScxError::ColumnStatsOverflow(_)
            // Write-path only: no file exists yet, so `CorruptFile`'s "re-run
            // conversion" suffix would point the caller at nothing.
            | ScxError::ZeroRowFramedShard
            | ScxError::UnsupportedColumnType { .. }
            | ScxError::DuplicateSection { .. }
            | ScxError::ObsLayoutConflict { .. }
            // Deliberately NOT `CorruptFile`. On the write path no file exists
            // yet; on the read path the file is intact and was written by an
            // *older*, uncapped SCX. The binding's "appears corrupt or was
            // written by an incompatible version; re-run conversion" suffix is
            // wrong on both. A writer-side inconsistency, which is what
            // `Validation` is for.
            | ScxError::UnsTooDeep { .. } => ScxErrorClass::Validation,
            // File is corrupt or was written by an incompatible/newer SCX:
            // bad magic/version/endian/checksum/catalog, an unknown on-disk
            // codec / value-encoding / section / shard-type byte, or an
            // out-of-bounds offset/index read from a malformed structure.
            ScxError::InvalidMagic
            | ScxError::InvalidShardMagic
            | ScxError::UnsupportedVersion
            | ScxError::UnsupportedFormatVersion { .. }
            | ScxError::UnsupportedEndian
            | ScxError::UnsupportedSectionVersion { .. }
            | ScxError::ChecksumMismatch { .. }
            | ScxError::InvalidCatalog(_)
            | ScxError::UnknownCodec(_)
            | ScxError::UnknownValueEncoding(_)
            | ScxError::RootCatalogTooLarge(_)
            | ScxError::UnknownSectionType(_)
            | ScxError::SectionNotFound(_)
            | ScxError::ShardIndexOutOfBounds { .. }
            | ScxError::BitmapGeneIdOutOfRange { .. }
            | ScxError::InvalidShardType { .. }
            | ScxError::EmptyIndptr
            | ScxError::SectionOutOfBounds { .. }
            | ScxError::AllocationTooLarge { .. }
            | ScxError::InvalidBlockIndex(_)
            | ScxError::ShardIndexOutOfRange { .. }
            | ScxError::ColumnStatsShardCountMismatch { .. } => ScxErrorClass::CorruptFile,
            ScxError::Io(io_err) => ScxErrorClass::Io(io_err.kind()),
            // Deliberately NOT `CorruptFile`: both files are intact. The
            // handle is simply looking at the older one, and the binding's
            // "appears corrupt; re-run conversion" suffix would send the
            // caller to fix a file that has nothing wrong with it. `Other`
            // maps to a plain `RuntimeError`, which is what a "your handle
            // is out of date" condition is.
            ScxError::FileChangedOnDisk { .. }
            // Wrapped lower-level errors and genuine runtime failures.
            | ScxError::WriterAlreadyFinished
            | ScxError::CscTranspose(_)
            | ScxError::Csr(_)
            | ScxError::Codec(_)
            | ScxError::Json(_)
            | ScxError::Arrow(_) => ScxErrorClass::Other,
        }
    }
}

pub type Result<T> = std::result::Result<T, ScxError>;

/// Validate that a requested allocation does not exceed the remaining
/// bytes in the enclosing section. Returns `ScxError::AllocationTooLarge`
/// if `requested > section_remaining`.
///
/// Use this before every `Vec::with_capacity(n)` or `vec![0u8; n]` where
/// `n` derives from an on-disk field to prevent a single malformed `u32`
/// from requesting a multi-GB allocation.
pub fn validate_allocation(requested: usize, section_remaining: usize) -> Result<()> {
    if requested > section_remaining {
        return Err(ScxError::AllocationTooLarge {
            requested,
            available: section_remaining,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::ErrorKind;

    #[test]
    fn class_buckets_variants_as_expected() {
        // Validation
        assert_eq!(ScxError::InconsistentCsr.class(), ScxErrorClass::Validation);
        assert_eq!(
            ScxError::NVarsOverflow(1 << 33).class(),
            ScxErrorClass::Validation
        );
        assert_eq!(
            ScxError::BlockRowsOverflow(70000).class(),
            ScxErrorClass::Validation
        );
        assert_eq!(
            ScxError::BlockNnzOverflow(1 << 33).class(),
            ScxErrorClass::Validation
        );
        assert_eq!(
            ScxError::StaleCscSidecar {
                built_generation: 1,
                data_generation: 2,
            }
            .class(),
            ScxErrorClass::Validation
        );

        // CorruptFile
        assert_eq!(ScxError::InvalidMagic.class(), ScxErrorClass::CorruptFile);
        assert_eq!(
            ScxError::InvalidShardMagic.class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::UnsupportedVersion.class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::UnsupportedFormatVersion {
                found: 99,
                max_supported: 3,
            }
            .class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::UnsupportedEndian.class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::UnsupportedSectionVersion {
                section: "catalog",
                found: 9,
                expected: 1,
            }
            .class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::ChecksumMismatch {
                section: "X".into(),
            }
            .class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::InvalidCatalog("bad".into()).class(),
            ScxErrorClass::CorruptFile
        );
        // Validation: writer-side limits / inconsistencies.
        assert_eq!(
            ScxError::ColumnStatsOverflow(256).class(),
            ScxErrorClass::Validation
        );
        assert_eq!(
            ScxError::ObsLayoutConflict {
                attempted: "sharded",
                existing: "single",
                single_kind: "ObsMetadata",
                sharded_kind: "ObsMetadataShard",
            }
            .class(),
            ScxErrorClass::Validation
        );
        // Write-path only — no file exists yet, so `CorruptFile`'s "re-run
        // conversion" suffix would send the caller to fix nothing.
        assert_eq!(
            ScxError::ZeroRowFramedShard.class(),
            ScxErrorClass::Validation
        );

        // CorruptFile: malformed/unreadable on-disk data (previously Other,
        // now surfaced as ValueError rather than RuntimeError in pyscx).
        assert_eq!(
            ScxError::UnknownCodec(99).class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::UnknownValueEncoding(7).class(),
            ScxErrorClass::CorruptFile
        );
        // A decoded index outside `[0, n_minor)` means the shard payload is
        // malformed — the payload is not covered by the catalog checksum, so
        // this is the read-path detection of genuine corruption.
        assert_eq!(
            ScxError::ShardIndexOutOfRange {
                shard: "X_shard_0".to_string(),
                index: u32::MAX,
                position: 3,
                n_minor: 100,
            }
            .class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::RootCatalogTooLarge(99999).class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::UnknownSectionType(200).class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::SectionNotFound("X".into()).class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::ShardIndexOutOfBounds { index: 5, count: 2 }.class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::BitmapGeneIdOutOfRange {
                gene_id: 9,
                n_vars: 4,
            }
            .class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::InvalidShardType {
                expected: 1,
                got: 9,
                section_type: 2,
            }
            .class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(ScxError::EmptyIndptr.class(), ScxErrorClass::CorruptFile);
        assert_eq!(
            ScxError::SectionOutOfBounds {
                offset: 10,
                length: 999,
                file_size: 100,
            }
            .class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::AllocationTooLarge {
                requested: 1 << 40,
                available: 16,
            }
            .class(),
            ScxErrorClass::CorruptFile
        );
        assert_eq!(
            ScxError::ColumnStatsShardCountMismatch {
                got: 3,
                expected: 4
            }
            .class(),
            ScxErrorClass::CorruptFile
        );

        // Io forwards the kind
        assert_eq!(
            ScxError::Io(std::io::Error::from(ErrorKind::NotFound)).class(),
            ScxErrorClass::Io(ErrorKind::NotFound)
        );
        assert_eq!(
            ScxError::Io(std::io::Error::from(ErrorKind::PermissionDenied)).class(),
            ScxErrorClass::Io(ErrorKind::PermissionDenied)
        );

        // Other: wrapped lower-level errors and genuine runtime failures.
        assert_eq!(
            ScxError::WriterAlreadyFinished.class(),
            ScxErrorClass::Other
        );
        assert_eq!(
            ScxError::CscTranspose("boom".into()).class(),
            ScxErrorClass::Other
        );
    }
}
