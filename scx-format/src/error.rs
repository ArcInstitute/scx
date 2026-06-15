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

    #[error("shard index {index} out of bounds (count: {count})")]
    ShardIndexOutOfBounds { index: usize, count: usize },

    #[error("invalid catalog: {0}")]
    InvalidCatalog(String),

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

    #[error("block n_rows {0} exceeds u16::MAX (65535)")]
    BlockRowsOverflow(u32),

    #[error("block nnz {0} exceeds u32::MAX")]
    BlockNnzOverflow(u64),

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
            | ScxError::StaleCscSidecar { .. }
            | ScxError::ColumnStatsOverflow(_)
            | ScxError::ObsLayoutConflict { .. } => ScxErrorClass::Validation,
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
            | ScxError::ColumnStatsShardCountMismatch { .. } => ScxErrorClass::CorruptFile,
            ScxError::Io(io_err) => ScxErrorClass::Io(io_err.kind()),
            // Wrapped lower-level errors and genuine runtime failures.
            ScxError::WriterAlreadyFinished
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
