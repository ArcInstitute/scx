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
    let sh = ShardHeader::read_from(&mut Cursor::new(&section[..SHARD_HEADER_SIZE]))?;

    // v2 strict shard_type validation: a v2 catalog must not carry
    // CSC entries with shard_type != 1. v1 catalogs preserve the
    // legacy catalog-wins tolerance (the writer hardcoded
    // shard_type = 0 for CSC pre-CSC-SUPPORT).
    if catalog_version >= 2 {
        sh.validate_csc_strict(entry.section_type)?;
    }

    // Extract encoded byte slices
    let indptr_bytes = &section[sh.indptr_rel_offset as usize..][..sh.indptr_length as usize];
    let indices_bytes = &section[sh.indices_rel_offset as usize..][..sh.indices_length as usize];
    let values_bytes = &section[sh.values_rel_offset as usize..][..sh.values_length as usize];
    let block_index_bytes =
        &section[sh.block_index_rel_offset as usize..][..sh.block_index_length as usize];

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
