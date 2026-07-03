// Per-shard encode helper shared by the in-memory pyscx converter
// and the `scx-convert` streaming writer. Both paths route through
// `encode_one_shard` so output is bit-identical regardless of how
// the (indptr, indices, values) slice was assembled upstream.

use blake3;

use scx_codec::value_encoding::{detect_value_encoding, values_to_raw_bytes};
use scx_codec::{encode_shard, CodecId, EncodedShard, ValueEncoding};
use scx_sparse::is_canonical_csr;

use crate::codec_select::select_codec_for_modality;
use crate::decode_sidecar::{DecodeSidecar, DEFAULT_DECODE_SIDECAR_MAX_OVERHEAD_RATIO};
use crate::error::ScxError;
use crate::modality::ModalityType;
use crate::section::SectionType;
use crate::shard::{
    BlockIndex, BlockIndexEntry, ShardHeader, CURRENT_SHARD_FORMAT_VERSION,
    DEFAULT_WRITE_SHARD_FORMAT_VERSION, MAX_BLOCK_ROWS, SHARD_HEADER_SIZE, SHARD_MAGIC,
};
use crate::writer::{compute_shard_stats, MajorAxis, PreEncodedSection};

/// Encode a single shard's CSR triplet into a `PreEncodedSection`
/// ready for sequential write via
/// [`crate::ScxWriter::write_preencoded_shard`].
///
/// Inputs must already be **canonical CSR**:
/// - shard-local (`shard_indptr[0] == 0`, length `n_rows + 1`),
/// - column indices sorted ascending within each row,
/// - no duplicate column indices within a row,
/// - no explicit stored zeros.
///
/// Canonicalize upstream with [`scx_sparse::canonicalize_csr`] (and
/// validate with `scx_sparse::validate_csr_arrays`). This contract is not
/// cosmetic: the Scx1 codec assumes canonical input, and a non-canonical
/// shard (e.g. a stored zero combined with a small per-row median) can hit
/// a hard `CodecError::FloatWithScx1` / bitstream error — or silently
/// encode incorrect values. Debug builds assert canonicality at this
/// boundary so a forgetful new call site fails fast with a clear message;
/// release builds trust the contract (no extra scan).
///
/// `explicit_codec = None` selects per-shard auto-codec via
/// [`select_codec_for_modality`]. `Some(CodecId::Scx1)` on non-integer
/// data falls back to `CodecId::Zstd` to match the existing pyscx
/// behaviour (Scx1 only handles integer payloads).
///
/// `index_dtype` encodes the *file-wide* index encoding choice
/// (`0` → u16 indices, `1` → u32). This drives the codec's index
/// packing path and is recorded in the shard header for the reader.
///
/// # Examples
///
/// ```
/// use scx_format_io::encoder::encode_one_shard;
/// use scx_format_io::section::SectionType;
/// use scx_format_io::modality::ModalityType;
///
/// // Canonical CSR: indptr[0] == 0, indices sorted per row, no dup, no zeros.
/// let indptr = [0u64, 2];
/// let indices = [0u32, 2];
/// let values = [1.0f32, 3.0];
/// let section = encode_one_shard(
///     &indptr, &indices, &values, None, 1, 3, 0,
///     SectionType::CsrShard, ModalityType::Rna, "X".to_string(),
///     None, // framing: unframed
/// )
/// .unwrap();
/// assert!(section.section_length > 0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn encode_one_shard(
    shard_indptr: &[u64],
    shard_indices: &[u32],
    shard_values: &[f32],
    explicit_codec: Option<CodecId>,
    index_dtype: u8,
    n_vars: u32,
    global_row_offset: u64,
    section_type: SectionType,
    modality_type: ModalityType,
    name: String,
    framing: Option<FramingConfig>,
) -> Result<PreEncodedSection, ScxError> {
    // Enforce the canonical-CSR contract in debug builds. Release builds
    // trust the caller (callers canonicalize upstream); this catches a
    // forgetful new call site before it produces a hard codec error or
    // silently-wrong Scx1 output.
    debug_assert!(
        is_canonical_csr(shard_indptr, shard_indices, shard_values),
        "encode_one_shard requires canonical CSR (sorted, deduped, no explicit \
         zeros); call scx_sparse::canonicalize_csr upstream"
    );

    let index_dtype_u16 = index_dtype == 0;

    // 3. Detect value encoding and encode values.
    let shard_value_encoding: ValueEncoding = detect_value_encoding(shard_values);
    let shard_values_bytes = values_to_raw_bytes(shard_values, shard_value_encoding)?;

    // 4. Select codec (heuristic when not explicit).
    let mut shard_codec = match explicit_codec {
        Some(codec_id) => {
            if codec_id == CodecId::Scx1 && !shard_value_encoding.is_integer() {
                CodecId::Zstd
            } else {
                codec_id
            }
        }
        None => select_codec_for_modality(&shard_values_bytes, shard_value_encoding, modality_type),
    };

    // 5–6. Encode shard + build block index. Two layouts:
    //   - Unframed (default): monolithic per-stream encode + a single whole-shard
    //     BlockIndex entry (or ≤MAX_BLOCK_ROWS split) with zero byte offsets —
    //     byte-identical to the legacy layout; shard v1.
    //   - Row-group-framed (`framing`): each row-group is encoded independently
    //     and the multi-entry BlockIndex records per-group byte offsets, enabling
    //     codec-agnostic sub-shard random access (SIDECAR-LONG-TERM-FIX.md
    //     Option B); shard v2. `trial` picks the smaller of {heuristic winner,
    //     ShufDeltaZstd} per shard.
    let n_major = (shard_indptr.len() - 1) as u32;
    let nnz = *shard_indptr.last().unwrap_or(&0);
    let (encoded, block_index, shard_version) = match framing {
        Some(fc) if fc.row_group_rows > 0 => {
            let frame = |codec: CodecId| {
                encode_shard_framed(
                    shard_indptr,
                    shard_indices,
                    &shard_values_bytes,
                    codec,
                    shard_value_encoding,
                    index_dtype_u16,
                    fc.row_group_rows,
                    fc.target_nnz,
                )
            };
            let (enc, bi) = if fc.trial && shard_codec != CodecId::ShufDeltaZstd {
                // Encode with the heuristic winner and ShufDeltaZstd; keep smaller.
                // compact-trial optimizes for size + random access and therefore
                // forgoes the Scx1 GPU/per-row sidecar (framed shards use the
                // block index). The §4.5 GPU cost-model knob is a future refinement.
                let (e_h, bi_h) = frame(shard_codec)?;
                let (e_s, bi_s) = frame(CodecId::ShufDeltaZstd)?;
                if framed_size(&e_s) < framed_size(&e_h) {
                    shard_codec = CodecId::ShufDeltaZstd;
                    (e_s, bi_s)
                } else {
                    (e_h, bi_h)
                }
            } else {
                frame(shard_codec)?
            };
            (enc, bi, CURRENT_SHARD_FORMAT_VERSION)
        }
        _ => {
            let enc = encode_shard(
                shard_indptr,
                shard_indices,
                &shard_values_bytes,
                shard_codec,
                shard_value_encoding,
                index_dtype_u16,
            )?;
            (
                enc,
                BlockIndex::for_shard(n_major, shard_indptr)?,
                DEFAULT_WRITE_SHARD_FORMAT_VERSION,
            )
        }
    };
    let mut block_index_bytes = Vec::new();
    block_index.write_to(&mut block_index_bytes)?;

    // 7. Shard-level checksum (truncated 8-byte BLAKE3).
    let mut shard_hasher = blake3::Hasher::new();
    shard_hasher.update(&encoded.indptr_bytes);
    shard_hasher.update(&encoded.indices_bytes);
    shard_hasher.update(&encoded.values_bytes);
    shard_hasher.update(&block_index_bytes);
    let shard_hash = shard_hasher.finalize();
    let mut shard_checksum = [0u8; 8];
    shard_checksum.copy_from_slice(&shard_hash.as_bytes()[..8]);

    // 8. Build ShardHeader with relative offsets.
    let indptr_rel_offset = SHARD_HEADER_SIZE as u32;
    let indptr_length = encoded.indptr_bytes.len() as u32;
    let indices_rel_offset = indptr_rel_offset + indptr_length;
    let indices_length = encoded.indices_bytes.len() as u32;
    let values_rel_offset = indices_rel_offset + indices_length;
    let values_length = encoded.values_bytes.len() as u32;
    let block_index_rel_offset = values_rel_offset + values_length;
    let block_index_length = block_index_bytes.len() as u32;

    let shard_header = ShardHeader {
        magic: SHARD_MAGIC,
        shard_format_version: shard_version,
        shard_type: 0, // CSR
        codec_id: shard_codec as u8,
        value_encoding: shard_value_encoding as u8,
        index_dtype,
        reserved_flags: [0; 3],
        n_major,
        n_minor: n_vars,
        nnz,
        global_offset: global_row_offset,
        indptr_rel_offset,
        indptr_length,
        indices_rel_offset,
        indices_length,
        values_rel_offset,
        values_length,
        block_index_rel_offset,
        block_index_length,
        checksum: shard_checksum,
    };

    let mut header_buf = Vec::with_capacity(SHARD_HEADER_SIZE);
    shard_header.write_to(&mut header_buf)?;

    // 9. Section-level checksum (full 32-byte BLAKE3).
    let mut section_hasher = blake3::Hasher::new();
    section_hasher.update(&header_buf);
    section_hasher.update(&encoded.indptr_bytes);
    section_hasher.update(&encoded.indices_bytes);
    section_hasher.update(&encoded.values_bytes);
    section_hasher.update(&block_index_bytes);
    let section_checksum = *section_hasher.finalize().as_bytes();

    let section_length = (header_buf.len()
        + encoded.indptr_bytes.len()
        + encoded.indices_bytes.len()
        + encoded.values_bytes.len()
        + block_index_bytes.len()) as u64;

    // Build the decode sidecar from the **encoder-produced** metadata (single
    // source of truth for the bit layout) — never re-derived. Only Scx1 integer
    // CSR shards carry `scx1_decode`.
    let decode_sidecar = match (shard_codec, &encoded.scx1_decode) {
        (CodecId::Scx1, Some(meta)) => DecodeSidecar::from_codec_metadata(
            meta,
            shard_value_encoding,
            index_dtype,
            n_vars,
            global_row_offset,
            section_type,
            0,
            section_length,
            section_checksum,
        )?
        .filter(|s| s.within_overhead_budget(DEFAULT_DECODE_SIDECAR_MAX_OVERHEAD_RATIO)),
        _ => None,
    };

    // 10. Compute shard stats. pyscx writes row-major CSR shards
    // exclusively; CSC sidecars use a separate path.
    let stats = compute_shard_stats(
        &shard_values_bytes,
        shard_value_encoding,
        MajorAxis::Row,
        global_row_offset,
        n_major as u64,
        n_vars as u64,
        nnz,
    );

    Ok(PreEncodedSection {
        encoded,
        block_index_bytes,
        header_buf,
        section_checksum,
        section_length,
        stats,
        name,
        section_type,
        nnz,
        decode_sidecar,
    })
}

/// Frame a `CodecId::None` CSR shard into row-groups for codec-agnostic
/// sub-shard random access (SIDECAR-LONG-TERM-FIX.md Option B, Phase 1).
///
/// Returns the re-framed **indptr sub-stream** (a concatenation of per-group
/// *local-rebased* indptrs — each group `[r0, r1)` contributes `r1-r0+1` u64s
/// starting at 0) and the multi-entry [`BlockIndex`]. The indices and values
/// sub-streams are left contiguous (None stores them as fixed-width LE arrays,
/// so a group's slice is `[indptr[r0]·w, indptr[r1]·w)`); their entry offsets
/// point into those global streams. Groups are capped at `min(row_group_rows,
/// MAX_BLOCK_ROWS)` rows. This is exactly the layout [`resolve_block_index`]
/// validates and `scx_codec::decode_row_group` consumes.
/// Config controlling row-group framing (F5-b). `row_group_rows` caps a group's
/// row count; `target_nnz` (if set) additionally caps its nnz (byte/nnz-aware
/// sizing, §4.3); `trial` selects the smaller of {heuristic winner,
/// ShufDeltaZstd} per shard.
#[derive(Debug, Clone, Copy)]
pub struct FramingConfig {
    pub row_group_rows: u32,
    pub target_nnz: Option<u64>,
    pub trial: bool,
}

/// Total encoded size of a shard's three sub-streams (trial-encode comparison key).
fn framed_size(e: &EncodedShard) -> usize {
    e.indptr_bytes.len() + e.indices_bytes.len() + e.values_bytes.len()
}

/// Encode a shard **row-group-framed** (F5-b / SIDECAR-LONG-TERM-FIX.md Option B):
/// partition the major axis into groups (≤ `row_group_rows` rows and, if set,
/// ≤ `target_nnz` nnz — always ≥1 row), encode each group independently as a
/// standalone sub-shard via [`encode_shard`], and concatenate the three
/// sub-streams while recording per-group byte offsets in a multi-entry
/// [`BlockIndex`]. Codec-agnostic: a group decodes via
/// `scx_codec::decode_row_group` (→ the ordinary per-shard decoder). For
/// `CodecId::None` this is byte-identical to the legacy contiguous layout with a
/// framed indptr. `scx1_decode` is dropped — framed shards use the block index
/// for random access, not the monolithic Scx1 sidecar.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_shard_framed(
    indptr: &[u64],
    indices: &[u32],
    values_bytes: &[u8],
    codec: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    row_group_rows: u32,
    target_nnz: Option<u64>,
) -> Result<(EncodedShard, BlockIndex), ScxError> {
    let n_rows = indptr.len().saturating_sub(1);
    let w_v = value_encoding.byte_width();
    let g = row_group_rows.clamp(1, MAX_BLOCK_ROWS) as usize;
    let nnz_cap = target_nnz.unwrap_or(u64::MAX);

    let mut indptr_stream: Vec<u8> = Vec::new();
    let mut indices_stream: Vec<u8> = Vec::new();
    let mut values_stream: Vec<u8> = Vec::new();
    let mut entries = Vec::with_capacity(n_rows.div_ceil(g).max(1));

    let mut r0 = 0usize;
    while r0 < n_rows {
        let base = indptr[r0];
        // Grow the group to ≤ g rows, stopping before nnz exceeds the cap (but
        // always keep ≥1 row even if that single row alone exceeds the cap).
        let max_r1 = (r0 + g).min(n_rows);
        let mut r1 = r0 + 1;
        while r1 < max_r1 && (indptr[r1 + 1] - base) <= nnz_cap {
            r1 += 1;
        }
        let end = indptr[r1] as usize;
        let start = base as usize;
        let nnz_in_block = indptr[r1] - base;

        let local_indptr: Vec<u64> = indptr[r0..=r1].iter().map(|&v| v - base).collect();
        let group_indices = &indices[start..end];
        let group_values = &values_bytes[start * w_v..end * w_v];
        let enc = encode_shard(
            &local_indptr,
            group_indices,
            group_values,
            codec,
            value_encoding,
            index_dtype_u16,
        )?;

        let ip_off = indptr_stream.len() as u32;
        let ix_off = indices_stream.len() as u32;
        let vv_off = values_stream.len() as u32;
        indptr_stream.extend_from_slice(&enc.indptr_bytes);
        indices_stream.extend_from_slice(&enc.indices_bytes);
        values_stream.extend_from_slice(&enc.values_bytes);

        entries.push(BlockIndexEntry::new(
            r0 as u32,
            (r1 - r0) as u32,
            ip_off,
            ix_off,
            vv_off,
            nnz_in_block,
        )?);
        r0 = r1;
    }

    Ok((
        EncodedShard {
            indptr_bytes: indptr_stream,
            indices_bytes: indices_stream,
            values_bytes: values_stream,
            scx1_decode: None,
        },
        BlockIndex { entries },
    ))
}
