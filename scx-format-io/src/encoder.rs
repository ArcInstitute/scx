// Per-shard encode helper shared by the in-memory pyscx converter
// and the `scx-convert` streaming writer. Both paths route through
// `encode_one_shard` so output is bit-identical regardless of how
// the (indptr, indices, values) slice was assembled upstream.

use blake3;

use scx_codec::value_encoding::{detect_value_encoding, values_to_raw_bytes};
use scx_codec::{encode_shard, CodecId, EncodedShard, ValueEncoding};
use scx_sparse::is_canonical_csr;

use crate::codec_select::{pick_codec_v2, select_codec_for_modality, DecodeTarget};
use crate::error::ScxError;
use crate::modality::ModalityType;
use crate::section::SectionType;
use crate::shard::{
    BlockIndex, BlockIndexEntry, ShardHeader, CURRENT_SHARD_FORMAT_VERSION,
    DEFAULT_WRITE_SHARD_FORMAT_VERSION, MAX_BLOCK_ROWS, SHARD_HEADER_SIZE, SHARD_MAGIC,
};
use crate::writer::{compute_shard_stats, MajorAxis, PreEncodedSection};

/// Which rule picks the framed codec when a ShufDeltaZstd trial-encode competes
/// with the heuristic winner: `Trial` = compact-trial (keep strictly smaller);
/// `V2` = auto_v2 (target-biased [`pick_codec_v2`]).
enum FramedPick {
    Trial,
    V2(DecodeTarget),
}

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
    // Default: auto-detect the value encoding per shard (the narrowest that fits
    // this shard's values). Callers needing a caller-fixed encoding (e.g. the
    // sort engine's file-wide encoding, for byte-identical output) use
    // [`encode_one_shard_with_value_encoding`].
    encode_one_shard_with_value_encoding(
        shard_indptr,
        shard_indices,
        shard_values,
        explicit_codec,
        index_dtype,
        n_vars,
        global_row_offset,
        section_type,
        modality_type,
        name,
        framing,
        None,
    )
}

/// Like [`encode_one_shard`], but with an optional caller-supplied
/// `value_encoding` override. When `Some(enc)`, the shard's values are encoded
/// with `enc` instead of the per-shard auto-detected narrowest encoding — this
/// is what lets a parallel encode reproduce a *file-wide* value encoding and
/// thus produce byte-identical output to a path that fixed the encoding up
/// front (e.g. `scx-ops` grouped-write fast path vs. `CsrEmitter`). `None`
/// preserves the auto-detect behaviour.
#[allow(clippy::too_many_arguments)]
pub fn encode_one_shard_with_value_encoding(
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
    value_encoding: Option<ValueEncoding>,
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

    // 3. Determine value encoding (caller override wins) and encode values.
    let shard_value_encoding: ValueEncoding =
        value_encoding.unwrap_or_else(|| detect_value_encoding(shard_values));
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
    //     codec-agnostic sub-shard random access; shard v2. `trial` picks the
    //     smaller of {heuristic winner,
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
            // Decide whether to trial-encode ShufDeltaZstd as a second candidate.
            // `decode_target` (auto_v2) takes precedence over `trial`
            // (compact-trial): Cpu keeps the heuristic and never trials;
            // Gpu/Storage/Auto trial only integer shards; compact-trial trials any
            // non-ShufDeltaZstd heuristic. In all cases both candidates are
            // row-group-framed (shard v2) with codec-agnostic BlockIndex access.
            let is_integer = shard_value_encoding.is_integer();
            let pick_mode: Option<FramedPick> = if shard_codec == CodecId::ShufDeltaZstd {
                None
            } else if let Some(dt) = fc.decode_target {
                (dt != DecodeTarget::Cpu && is_integer).then_some(FramedPick::V2(dt))
            } else if fc.trial {
                Some(FramedPick::Trial)
            } else {
                None
            };
            match pick_mode {
                Some(mode) => {
                    let (e_h, bi_h) = frame(shard_codec)?;
                    let (e_s, bi_s) = frame(CodecId::ShufDeltaZstd)?;
                    let (size_h, size_s) = (framed_size(&e_h), framed_size(&e_s));
                    let pick_shufdelta = match mode {
                        FramedPick::Trial => size_s < size_h,
                        FramedPick::V2(dt) => {
                            pick_codec_v2(shard_codec, dt, size_h, size_s, true)
                                == CodecId::ShufDeltaZstd
                        }
                    };
                    let (framed_codec, framed_enc, framed_bi) = if pick_shufdelta {
                        (CodecId::ShufDeltaZstd, e_s, bi_s)
                    } else {
                        (shard_codec, e_h, bi_h)
                    };
                    shard_codec = framed_codec;
                    (framed_enc, framed_bi, CURRENT_SHARD_FORMAT_VERSION)
                }
                None => {
                    let (enc, bi) = frame(shard_codec)?;
                    (enc, bi, CURRENT_SHARD_FORMAT_VERSION)
                }
            }
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

    // 8. Build ShardHeader with relative offsets. The header's offset/length
    // fields are u32; a shard whose sub-streams (or their cumulative offset)
    // exceed 4 GiB would silently wrap. Fail loud instead (F-c).
    let len_u32 = |n: usize, what: &str| -> Result<u32, ScxError> {
        u32::try_from(n).map_err(|_| {
            ScxError::ShardStreamTooLarge(format!("{what} length {n} exceeds u32::MAX"))
        })
    };
    let add_u32 = |a: u32, b: u32, what: &str| -> Result<u32, ScxError> {
        a.checked_add(b).ok_or_else(|| {
            ScxError::ShardStreamTooLarge(format!("{what} relative offset exceeds u32::MAX"))
        })
    };
    let indptr_rel_offset = SHARD_HEADER_SIZE as u32;
    let indptr_length = len_u32(encoded.indptr_bytes.len(), "indptr")?;
    let indices_rel_offset = add_u32(indptr_rel_offset, indptr_length, "indices")?;
    let indices_length = len_u32(encoded.indices_bytes.len(), "indices")?;
    let values_rel_offset = add_u32(indices_rel_offset, indices_length, "values")?;
    let values_length = len_u32(encoded.values_bytes.len(), "values")?;
    let block_index_rel_offset = add_u32(values_rel_offset, values_length, "block_index")?;
    let block_index_length = len_u32(block_index_bytes.len(), "block_index")?;

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
    })
}

/// Frame a `CodecId::None` CSR shard into row-groups for codec-agnostic
/// sub-shard random access.
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
///
/// `trial` (compact-trial) picks the smaller of {framed heuristic winner, framed
/// ShufDeltaZstd} per shard. Both are row-group-framed (shard v2) and use the
/// codec-agnostic `BlockIndex` for random access; the historical unframed-Scx1 +
/// decode-sidecar representation was removed once framed Scx1 gained an in-VRAM
/// GPU decode path.
/// `decode_target` (auto_v2) is the workload-aware alternative to `trial`: when
/// `Some`, the per-shard integer codec is picked by [`pick_codec_v2`] using the
/// same dual-encode as `trial` but biased toward/away from ShufDeltaZstd by the
/// downstream target (`Cpu` keeps the heuristic; `Gpu`/`Storage`/`Auto` adopt
/// ShufDeltaZstd where it compresses ≤/< the heuristic). `decode_target` takes
/// precedence over `trial` (the CLI/pyscx layers keep them mutually exclusive).
#[derive(Debug, Clone, Copy)]
pub struct FramingConfig {
    pub row_group_rows: u32,
    pub target_nnz: Option<u64>,
    pub trial: bool,
    pub decode_target: Option<DecodeTarget>,
}

/// Default row-group size (G) for framed writes. Chosen from the F5 follow-up
/// sweep (Phase B/B7): compression ratio is flat across G (±0.3%), while finer G
/// is 1.4–2.1× faster for scattered/training reads on multi-shard files, so 256
/// is the scatter-friendly middle between decode latency and block-index size.
/// This is the single chokepoint shared by every framed-by-default write site
/// (`ConvertOptions::default`, the CLI `--row-group-rows` default, and the pyscx
/// `from_*` defaults); a plain `codec="auto"` write frames at this G (framing is
/// codec-agnostic — see the encoder's framing branch), so it costs no extra
/// encode work. Pass `row_group_rows = 0` to opt out (unframed v3 output).
pub const DEFAULT_ROW_GROUP_ROWS: u32 = 256;

impl Default for FramingConfig {
    fn default() -> Self {
        Self {
            row_group_rows: DEFAULT_ROW_GROUP_ROWS,
            target_nnz: None,
            trial: false,
            decode_target: None,
        }
    }
}

/// Total encoded size of a shard's three sub-streams (trial-encode comparison key).
fn framed_size(e: &EncodedShard) -> usize {
    e.indptr_bytes.len() + e.indices_bytes.len() + e.values_bytes.len()
}

/// Encode a shard **row-group-framed** (F5-b):
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
pub fn encode_shard_framed(
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

        // Sub-stream offsets are u32 in BlockIndexEntry; a >4 GiB concatenated
        // sub-stream would silently wrap. Fail loud instead (F-c).
        let offset_u32 = |stream: &[u8], name: &str| -> Result<u32, ScxError> {
            u32::try_from(stream.len()).map_err(|_| {
                ScxError::ShardStreamTooLarge(format!(
                    "framed {name} sub-stream offset {} exceeds u32::MAX",
                    stream.len()
                ))
            })
        };
        let ip_off = offset_u32(&indptr_stream, "indptr")?;
        let ix_off = offset_u32(&indices_stream, "indices")?;
        let vv_off = offset_u32(&values_stream, "values")?;
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

#[cfg(test)]
mod auto_v2_tests {
    use super::*;
    use crate::modality::ModalityType;
    use scx_codec::CodecId;

    /// Census-like integer shard: `n_rows` rows, `nnz` sorted unique column
    /// indices each. `max_count` sets the value range (≤8 median → Scx1 heuristic;
    /// larger → Zstd heuristic). Sorted indices make ShufDeltaZstd (shuffle+delta+
    /// zstd on the index stream) a strong candidate.
    fn gen_int_shard(
        n_rows: usize,
        nnz: usize,
        n_cols: u32,
        max_count: u32,
    ) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::with_capacity(n_rows * nnz);
        let mut values = Vec::with_capacity(n_rows * nnz);
        indptr.push(0u64);
        let mut state: u64 = 0x1234_5678_9abc_def0;
        for _ in 0..n_rows {
            // Pick `nnz` sorted, unique columns via a fixed stride + jitter.
            let stride = (n_cols as usize / nnz.max(1)).max(1);
            let mut col = 0usize;
            for _ in 0..nnz {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                col += 1 + (state as usize % stride);
                if col >= n_cols as usize {
                    break;
                }
                indices.push(col as u32);
                values.push((1 + (state % max_count as u64) as u32) as f32);
            }
            indptr.push(indices.len() as u64);
        }
        (indptr, indices, values)
    }

    fn encode(
        indptr: &[u64],
        indices: &[u32],
        values: &[f32],
        n_cols: u32,
        framing: FramingConfig,
    ) -> PreEncodedSection {
        encode_one_shard_with_value_encoding(
            indptr,
            indices,
            values,
            None, // auto-select heuristic
            0,    // u16 indices
            n_cols,
            0,
            SectionType::CsrShard,
            ModalityType::Rna,
            "X_shard_0".to_string(),
            Some(framing),
            Some(ValueEncoding::Uint8),
        )
        .expect("encode")
    }

    /// auto_v2 with decode_target=Cpu is byte-identical to plain auto: Cpu keeps
    /// the heuristic winner and never runs the ShufDeltaZstd dual-encode.
    #[test]
    fn auto_v2_cpu_matches_auto() {
        let (indptr, indices, values) = gen_int_shard(512, 40, 5000, 4);
        let base = FramingConfig {
            row_group_rows: 256,
            target_nnz: None,
            trial: false,
            decode_target: None,
        };
        let auto = encode(&indptr, &indices, &values, 5000, base);
        let v2_cpu = encode(
            &indptr,
            &indices,
            &values,
            5000,
            FramingConfig {
                decode_target: Some(DecodeTarget::Cpu),
                ..base
            },
        );
        assert_eq!(
            auto.codec_id(),
            v2_cpu.codec_id(),
            "Cpu must keep the heuristic codec"
        );
        assert_eq!(
            auto.encoded.indices_bytes, v2_cpu.encoded.indices_bytes,
            "Cpu auto_v2 must be byte-identical to auto"
        );
        assert_eq!(auto.section_length, v2_cpu.section_length);
    }

    /// auto_v2 with decode_target=Storage adopts ShufDeltaZstd on this
    /// census-like integer shard (where it compresses better) and never exceeds
    /// the plain-auto size — the "no size regression vs auto" guard.
    #[test]
    fn auto_v2_storage_adopts_shufdelta_and_no_size_regression() {
        // Larger counts (median > 8) → heuristic Zstd, which ShufDeltaZstd's
        // delta+shuffle on the structured index/value streams reliably beats.
        let (indptr, indices, values) = gen_int_shard(2048, 60, 20000, 255);
        let base = FramingConfig {
            row_group_rows: 256,
            target_nnz: None,
            trial: false,
            decode_target: None,
        };
        let auto = encode(&indptr, &indices, &values, 20000, base);
        let v2_storage = encode(
            &indptr,
            &indices,
            &values,
            20000,
            FramingConfig {
                decode_target: Some(DecodeTarget::Storage),
                ..base
            },
        );
        assert_eq!(
            auto.codec_id(),
            CodecId::Zstd as u8,
            "large-count heuristic is Zstd"
        );
        assert_eq!(
            v2_storage.codec_id(),
            CodecId::ShufDeltaZstd as u8,
            "Storage should adopt ShufDeltaZstd where it compresses better"
        );
        // Core no-regression guarantee: Storage picks the smaller-or-equal codec.
        assert!(
            v2_storage.section_length <= auto.section_length,
            "auto_v2/Storage ({}) must not exceed auto ({})",
            v2_storage.section_length,
            auto.section_length
        );
        assert!(v2_storage.section_length > 0);
    }

    /// Float modality stays Pcodec under every decode target (ShufDeltaZstd never
    /// competes for float).
    #[test]
    fn auto_v2_float_stays_pcodec() {
        let (indptr, indices, _) = gen_int_shard(256, 20, 5000, 4);
        let nnz = *indptr.last().unwrap() as usize;
        let values: Vec<f32> = (0..nnz).map(|i| (i as f32) * 0.5 + 0.25).collect();
        let base = FramingConfig {
            row_group_rows: 256,
            target_nnz: None,
            trial: false,
            decode_target: None,
        };
        for dt in [DecodeTarget::Auto, DecodeTarget::Gpu, DecodeTarget::Storage] {
            let sec = encode_one_shard_with_value_encoding(
                &indptr,
                &indices,
                &values,
                None,
                0,
                5000,
                0,
                SectionType::CsrShard,
                ModalityType::Rna,
                "X_shard_0".to_string(),
                Some(FramingConfig {
                    decode_target: Some(dt),
                    ..base
                }),
                Some(ValueEncoding::Float32),
            )
            .expect("encode");
            assert_eq!(
                sec.codec_id(),
                CodecId::Pcodec as u8,
                "float must stay Pcodec under {dt:?}"
            );
        }
    }
}
