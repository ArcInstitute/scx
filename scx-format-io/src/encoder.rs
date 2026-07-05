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
            if fc.trial && shard_codec != CodecId::ShufDeltaZstd {
                // Two-layer cost model: pick the
                // smallest *random-access-safe* representation per shard.
                //   (A) framed heuristic winner, (B) framed ShufDeltaZstd,
                //   (C) unframed Scx1 + DecodeSidecar (GPU/per-row fast path),
                //       eligible only when the heuristic itself is Scx1
                //       (integer, low median) and the sidecar fits its budget.
                let heuristic_is_scx1 = shard_codec == CodecId::Scx1;
                let (e_h, bi_h) = frame(shard_codec)?;
                let (framed_codec, framed_enc, framed_bi) = {
                    let (e_s, bi_s) = frame(CodecId::ShufDeltaZstd)?;
                    if framed_size(&e_s) < framed_size(&e_h) {
                        (CodecId::ShufDeltaZstd, e_s, bi_s)
                    } else {
                        (shard_codec, e_h, bi_h)
                    }
                };
                let framed_sz = framed_size(&framed_enc);

                // Representation (C): only worth encoding when the heuristic
                // picked Scx1 (i.e. the GPU-friendly integer/low-median class).
                let keep_scx1 = if heuristic_is_scx1 {
                    let enc_c = encode_shard(
                        shard_indptr,
                        shard_indices,
                        &shard_values_bytes,
                        CodecId::Scx1,
                        shard_value_encoding,
                        index_dtype_u16,
                    )?;
                    let bi_c = BlockIndex::for_shard(n_major, shard_indptr)?;
                    let mut bi_c_bytes = Vec::new();
                    bi_c.write_to(&mut bi_c_bytes)?;
                    // The sidecar (and thus the GPU/per-row path) survives only
                    // if it fits the overhead budget — predict that here so we
                    // never commit a sidecar-less unframed shard into a v4 file.
                    let section_len_c = (SHARD_HEADER_SIZE
                        + enc_c.indptr_bytes.len()
                        + enc_c.indices_bytes.len()
                        + enc_c.values_bytes.len()
                        + bi_c_bytes.len()) as u64;
                    let sidecar_viable = if let Some(meta) = &enc_c.scx1_decode {
                        DecodeSidecar::from_codec_metadata(
                            meta,
                            shard_value_encoding,
                            index_dtype,
                            n_vars,
                            global_row_offset,
                            section_type,
                            0,
                            section_len_c,
                            [0u8; 32],
                        )?
                        .filter(|s| {
                            s.within_overhead_budget(DEFAULT_DECODE_SIDECAR_MAX_OVERHEAD_RATIO)
                        })
                        .is_some()
                    } else {
                        false
                    };
                    let unframed_sz = framed_size(&enc_c);
                    let keep = sidecar_viable
                        && (fc.prefer_gpu_sidecar
                            || unframed_sz as f64
                                <= framed_sz as f64 * (1.0 + compact_trial_gpu_margin()));
                    keep.then_some((enc_c, bi_c))
                } else {
                    None
                };

                if let Some((enc_c, bi_c)) = keep_scx1 {
                    // Unframed Scx1 + sidecar (shard v1); random access via the
                    // per-row DecodeSidecar layered on top (§4.3).
                    shard_codec = CodecId::Scx1;
                    (enc_c, bi_c, DEFAULT_WRITE_SHARD_FORMAT_VERSION)
                } else {
                    shard_codec = framed_codec;
                    (framed_enc, framed_bi, CURRENT_SHARD_FORMAT_VERSION)
                }
            } else {
                let (enc, bi) = frame(shard_codec)?;
                (enc, bi, CURRENT_SHARD_FORMAT_VERSION)
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
///
/// `prefer_gpu_sidecar` engages the two-layer cost model
/// (`SIDECAR-LONG-TERM-FIX.md` §4.3) inside a `trial` (compact-trial) encode: a
/// shard whose heuristic codec is Scx1 (integer, low median — the GPU/per-row
/// class) is stored **unframed Scx1 with its `DecodeSidecar`** instead of a
/// framed codec, preserving the FOR-BP/Rice GPU device-decode + bit-level
/// per-row random access. Both representations are random-access-safe (framed →
/// group `BlockIndex`; unframed Scx1 → per-row sidecar), so a `compact-trial`
/// file legitimately mixes them. When `false`, Scx1 is only kept if it is no
/// larger than the framed winner (free GPU, no size regression) — governed by
/// the `SCX_COMPACT_TRIAL_GPU_MARGIN` slack (default 0).
#[derive(Debug, Clone, Copy)]
pub struct FramingConfig {
    pub row_group_rows: u32,
    pub target_nnz: Option<u64>,
    pub trial: bool,
    pub prefer_gpu_sidecar: bool,
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
            prefer_gpu_sidecar: false,
        }
    }
}

/// Size slack (fraction) by which an unframed Scx1+sidecar representation may
/// exceed the framed winner and still be kept for its GPU/per-row fast path when
/// `prefer_gpu_sidecar` is not set. Default 0 → keep Scx1 only when it does not
/// regress size. Env: `SCX_COMPACT_TRIAL_GPU_MARGIN`.
///
/// Read once (process-lifetime) rather than per shard — `encode_one_shard`'s
/// trial branch runs for every shard, and a `std::env::var` syscall per shard on
/// a many-thousand-shard file is pure overhead.
fn compact_trial_gpu_margin() -> f64 {
    static MARGIN: std::sync::LazyLock<f64> = std::sync::LazyLock::new(|| {
        std::env::var("SCX_COMPACT_TRIAL_GPU_MARGIN")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|m| m.is_finite() && *m >= 0.0)
            .unwrap_or(0.0)
    });
    *MARGIN
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
        let ip_off = u32::try_from(indptr_stream.len()).map_err(|_| {
            ScxError::ShardStreamTooLarge(format!(
                "framed indptr sub-stream offset {} exceeds u32::MAX",
                indptr_stream.len()
            ))
        })?;
        let ix_off = u32::try_from(indices_stream.len()).map_err(|_| {
            ScxError::ShardStreamTooLarge(format!(
                "framed indices sub-stream offset {} exceeds u32::MAX",
                indices_stream.len()
            ))
        })?;
        let vv_off = u32::try_from(values_stream.len()).map_err(|_| {
            ScxError::ShardStreamTooLarge(format!(
                "framed values sub-stream offset {} exceeds u32::MAX",
                values_stream.len()
            ))
        })?;
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
