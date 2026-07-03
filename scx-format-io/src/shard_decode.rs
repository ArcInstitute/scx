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
use crate::shard::{resolve_block_index, ShardHeader, DEFAULT_WRITE_SHARD_FORMAT_VERSION};
use crate::validated_section::ValidatedSection;

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
    let vs = ValidatedSection::new(section);
    let sh = ShardHeader::read_from(&mut Cursor::new(vs.header()?))?;

    // v2 strict shard_type validation: a v2 catalog must not carry
    // CSC entries with shard_type != 1. v1 catalogs preserve the
    // legacy catalog-wins tolerance (the writer hardcoded
    // shard_type = 0 for CSC pre-CSC-SUPPORT).
    if catalog_version >= 2 {
        sh.validate_csc_strict(entry.section_type)?;
    }

    // Extract encoded byte slices (bounds-checked against the untrusted header)
    let indptr_bytes = vs.subslice(sh.indptr_rel_offset, sh.indptr_length)?;
    let indices_bytes = vs.subslice(sh.indices_rel_offset, sh.indices_length)?;
    let values_bytes = vs.subslice(sh.values_rel_offset, sh.values_length)?;
    let block_index_bytes = vs.subslice(sh.block_index_rel_offset, sh.block_index_length)?;

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

    // Row-group-framed (v2) shards: the indptr sub-stream is a concatenation of
    // per-group local-rebased indptrs, so the whole-stream decoder can't read it
    // — iterate the block index and reassemble the global CSR (byte-identical to
    // an unframed decode of the same data). Legacy (v1) shards take the direct
    // whole-shard path below.
    if sh.shard_format_version > DEFAULT_WRITE_SHARD_FORMAT_VERSION {
        return decode_framed_shard_scipy(
            &sh,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            block_index_bytes,
            codec_id,
            value_encoding,
            index_dtype_u16,
        );
    }

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

/// Reassemble a whole framed (v2) shard into a global scipy CSR by iterating its
/// row-group block index. Each group decodes to a *local* CSR
/// (`indptr[0] == 0`); we rebase the group indptrs into a single monotonic
/// global indptr and concatenate indices/values. Output is byte-identical to a
/// legacy whole-shard decode of the same matrix.
#[allow(clippy::too_many_arguments)]
fn decode_framed_shard_scipy(
    sh: &ShardHeader,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    block_index_bytes: &[u8],
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    let spans = resolve_block_index(sh, block_index_bytes)?;
    let n_major = sh.n_major as usize;
    let nnz = sh.nnz as usize;

    let mut indptr = Vec::with_capacity(n_major + 1);
    indptr.push(0i64);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    let mut running: i64 = 0;

    for span in &spans {
        let decoded = scx_codec::decode_row_group(
            codec_id,
            span,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            value_encoding,
            index_dtype_u16,
        )?;
        // Convert this group (local CSR) to scipy types, then rebase indptr.
        let (g_indptr, g_indices, g_data) =
            scx_codec::decoded_shard_to_scipy(decoded, value_encoding)?;
        for &local in &g_indptr[1..] {
            indptr.push(running + local);
        }
        if let Some(&last) = g_indptr.last() {
            running += last;
        }
        indices.extend_from_slice(&g_indices);
        data.extend_from_slice(&g_data);
    }

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
    let vs = ValidatedSection::new(section);
    let sh = ShardHeader::read_from(&mut Cursor::new(vs.header()?))?;

    if catalog_version >= 2 {
        sh.validate_csc_strict(entry.section_type)?;
    }

    let indptr_bytes = vs.subslice(sh.indptr_rel_offset, sh.indptr_length)?;

    let codec_id = CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?;

    // Framed (v2) shards store per-group local indptrs; reconstruct the global
    // indptr from the row-group index rather than reading the stream directly.
    if sh.shard_format_version > DEFAULT_WRITE_SHARD_FORMAT_VERSION {
        let value_encoding = ValueEncoding::from_u8(sh.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(sh.value_encoding))?;
        let index_dtype_u16 = sh.index_dtype == 0;
        let indices_bytes = vs.subslice(sh.indices_rel_offset, sh.indices_length)?;
        let values_bytes = vs.subslice(sh.values_rel_offset, sh.values_length)?;
        let block_index_bytes = vs.subslice(sh.block_index_rel_offset, sh.block_index_length)?;
        let spans = resolve_block_index(&sh, block_index_bytes)?;
        let mut indptr = Vec::with_capacity(sh.n_major as usize + 1);
        indptr.push(0i64);
        let mut running: i64 = 0;
        for span in &spans {
            let (g_indptr, _, _) = scx_codec::decoded_shard_to_scipy(
                scx_codec::decode_row_group(
                    codec_id,
                    span,
                    indptr_bytes,
                    indices_bytes,
                    values_bytes,
                    value_encoding,
                    index_dtype_u16,
                )?,
                value_encoding,
            )?;
            for &local in &g_indptr[1..] {
                indptr.push(running + local);
            }
            if let Some(&last) = g_indptr.last() {
                running += last;
            }
        }
        return Ok(indptr);
    }

    let indptr = scx_codec::decode_indptr_only(indptr_bytes, codec_id, sh.n_major as usize)?;
    Ok(indptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::FullCatalogEntry;
    use crate::shard::{ShardHeader, SHARD_HEADER_SIZE, SHARD_MAGIC};
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
}
