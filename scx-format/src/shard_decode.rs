//! Stateless CSR shard decoder.
//!
//! Decodes a single shard's bytes (header + payload) into scipy-compatible
//! (`Vec<i64>`, `Vec<i32>`, `Vec<f32>`) triples. Shared between the local
//! [`ScxReader`](crate::reader::ScxReader) (which slices bytes from mmap)
//! and the cloud `CloudSectionReader` (which gets bytes from
//! `object_store` range reads), so the codec path is identical regardless
//! of how the bytes were fetched.
//!
//! [`ScxReader::read_shard_from_entry`](crate::reader::ScxReader::read_shard_from_entry)
//! is a thin wrapper over this function.

use std::io::Cursor;

use scx_codec::{CodecId, EncodedShardRef, ValueEncoding};

use crate::catalog::FullCatalogEntry;
use crate::error::{Result, ScxError};
use crate::shard::{ShardHeader, SHARD_HEADER_SIZE};

/// Bounds-checked extraction of `section[off..off + len]`.
///
/// Shard-header offset/length fields are read raw from untrusted bytes
/// (the catalog BLAKE3 authenticates catalog bytes, *not* shard payloads),
/// so a corrupt or hostile shard can point a region past the section.
/// Return [`ScxError::SectionOutOfBounds`] rather than panicking on the
/// slice. The `ValidatedSection` newtype is the long-term home for this
/// (Phase 1); this keeps the four call-sites uniform in the interim.
pub(crate) fn checked_subslice(section: &[u8], off: u32, len: u32) -> Result<&[u8]> {
    let start = off as usize;
    let end = (off as u64).checked_add(len as u64);
    match end {
        Some(end) if end <= section.len() as u64 => Ok(&section[start..end as usize]),
        _ => Err(ScxError::SectionOutOfBounds {
            offset: off as u64,
            length: len as u64,
            file_size: section.len(),
        }),
    }
}

/// Bounds-checked slice of the fixed-size shard header prefix.
///
/// A section shorter than [`SHARD_HEADER_SIZE`] would panic the raw
/// `&section[..SHARD_HEADER_SIZE]` slice; reject it as out-of-bounds.
pub(crate) fn shard_header_slice(section: &[u8]) -> Result<&[u8]> {
    if section.len() < SHARD_HEADER_SIZE {
        return Err(ScxError::SectionOutOfBounds {
            offset: 0,
            length: SHARD_HEADER_SIZE as u64,
            file_size: section.len(),
        });
    }
    Ok(&section[..SHARD_HEADER_SIZE])
}

/// Decode a single CSR shard from its on-disk bytes.
///
/// `section` is the full shard section (header + payload). `entry` is
/// the catalog entry naming the section (used for error messages and
/// `shard_type` validation against the catalog). `catalog_version` is
/// the parsed `FullCatalog::catalog_version`; v2 catalogs trigger the
/// strict shard-type validation against `entry.section_type` that v1
/// catalogs tolerate.
///
/// When `verify_checksum` is `true`, recomputes the BLAKE3 hash of the
/// shard payload and compares it to the truncated 8-byte checksum in
/// the shard header. Use this for `scx validate` or when integrity
/// must be confirmed per-shard.
pub fn decode_shard_bytes(
    section: &[u8],
    entry: &FullCatalogEntry,
    catalog_version: u16,
    verify_checksum: bool,
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    // Parse shard header
    let sh = ShardHeader::read_from(&mut Cursor::new(shard_header_slice(section)?))?;

    // v2 strict shard_type validation: a v2 catalog must not carry
    // CSC entries with shard_type != 1. v1 catalogs preserve the
    // legacy catalog-wins tolerance (the writer hardcoded
    // shard_type = 0 for CSC pre-CSC-SUPPORT).
    if catalog_version >= 2 {
        sh.validate_csc_strict(entry.section_type)?;
    }

    // Extract encoded byte slices (bounds-checked against the untrusted header)
    let indptr_bytes = checked_subslice(section, sh.indptr_rel_offset, sh.indptr_length)?;
    let indices_bytes = checked_subslice(section, sh.indices_rel_offset, sh.indices_length)?;
    let values_bytes = checked_subslice(section, sh.values_rel_offset, sh.values_length)?;
    let block_index_bytes =
        checked_subslice(section, sh.block_index_rel_offset, sh.block_index_length)?;

    if verify_checksum {
        let mut shard_hasher = blake3::Hasher::new();
        shard_hasher.update(indptr_bytes);
        shard_hasher.update(indices_bytes);
        shard_hasher.update(values_bytes);
        shard_hasher.update(block_index_bytes);
        let hash = shard_hasher.finalize();
        let mut computed = [0u8; 8];
        computed.copy_from_slice(&hash.as_bytes()[..8]);
        if computed != sh.checksum {
            return Err(ScxError::ChecksumMismatch {
                section: format!("shard '{}'", entry.name),
            });
        }
    }

    // Resolve codec and encoding from shard header (NOT file header)
    let codec_id = CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?;
    let value_encoding = ValueEncoding::from_u8(sh.value_encoding)
        .ok_or(ScxError::UnknownValueEncoding(sh.value_encoding))?;
    let index_dtype_u16 = sh.index_dtype == 0;

    let encoded = EncodedShardRef {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    };

    let (indptr, indices, data) = scx_codec::decode_shard_scipy(
        &encoded,
        codec_id,
        value_encoding,
        sh.n_major as usize,
        sh.nnz as usize,
        index_dtype_u16,
    )?;

    Ok((indptr, indices, data))
}

/// Decode **only** the indptr region of a shard.
///
/// Parses the [`ShardHeader`], slices out `indptr_bytes`, and dispatches
/// to [`scx_codec::decode_indptr_only`]. No checksum verification — the
/// catalog already authenticates section offsets/lengths, and indptr-only
/// callers (e.g. the streaming SCX → h5ad export's `precompute_total_nnz`)
/// are followed by a full shard decode that will surface any corruption.
pub fn decode_shard_indptr_bytes(
    section: &[u8],
    entry: &FullCatalogEntry,
    catalog_version: u16,
) -> Result<Vec<i64>> {
    let sh = ShardHeader::read_from(&mut Cursor::new(shard_header_slice(section)?))?;

    if catalog_version >= 2 {
        sh.validate_csc_strict(entry.section_type)?;
    }

    let indptr_bytes = checked_subslice(section, sh.indptr_rel_offset, sh.indptr_length)?;

    let codec_id = CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?;

    let indptr = scx_codec::decode_indptr_only(indptr_bytes, codec_id, sh.n_major as usize)?;
    Ok(indptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::FullCatalogEntry;
    use crate::shard::{ShardHeader, SHARD_MAGIC};
    use crate::SectionType;

    fn dummy_entry() -> FullCatalogEntry {
        FullCatalogEntry {
            name: "X_shard_0".to_string(),
            offset: 0,
            length: 0,
            section_type: SectionType::CsrShard,
            checksum: [0u8; 32],
            modality_id: 0,
            stats: None,
        }
    }

    /// A header whose declared region offsets/lengths point past the
    /// section must yield `SectionOutOfBounds`, not a slice panic (F1).
    #[test]
    fn decode_shard_rejects_out_of_bounds_offsets() {
        let header = ShardHeader {
            magic: SHARD_MAGIC,
            shard_format_version: 1,
            shard_type: 0,
            codec_id: 0,
            value_encoding: 2,
            index_dtype: 1,
            reserved_flags: [0u8; 3],
            n_major: 1,
            n_minor: 1,
            nnz: 0,
            global_offset: 0,
            // indptr region claims to extend far past the section end.
            indptr_rel_offset: SHARD_HEADER_SIZE as u32,
            indptr_length: 4096,
            indices_rel_offset: SHARD_HEADER_SIZE as u32,
            indices_length: 0,
            values_rel_offset: SHARD_HEADER_SIZE as u32,
            values_length: 0,
            block_index_rel_offset: SHARD_HEADER_SIZE as u32,
            block_index_length: 0,
            checksum: [0u8; 8],
        };
        let mut section = Vec::new();
        header.write_to(&mut section).unwrap();
        assert_eq!(section.len(), SHARD_HEADER_SIZE);

        let err = decode_shard_bytes(&section, &dummy_entry(), 1, false).unwrap_err();
        assert!(
            matches!(err, ScxError::SectionOutOfBounds { .. }),
            "expected SectionOutOfBounds, got {err:?}"
        );
    }

    /// A section shorter than the fixed 76-byte header must be rejected
    /// instead of panicking on `&section[..SHARD_HEADER_SIZE]` (F1).
    #[test]
    fn decode_shard_rejects_truncated_section() {
        let section = vec![0u8; SHARD_HEADER_SIZE - 1];
        let err = decode_shard_bytes(&section, &dummy_entry(), 1, false).unwrap_err();
        assert!(
            matches!(err, ScxError::SectionOutOfBounds { .. }),
            "expected SectionOutOfBounds, got {err:?}"
        );
    }

    #[test]
    fn checked_subslice_bounds() {
        let buf = vec![0u8; 100];
        assert!(checked_subslice(&buf, 0, 100).is_ok());
        assert!(checked_subslice(&buf, 50, 50).is_ok());
        assert!(matches!(
            checked_subslice(&buf, 50, 51),
            Err(ScxError::SectionOutOfBounds { .. })
        ));
        // offset + length overflow must not wrap to a small in-bounds value.
        assert!(matches!(
            checked_subslice(&buf, u32::MAX, u32::MAX),
            Err(ScxError::SectionOutOfBounds { .. })
        ));
    }
}
