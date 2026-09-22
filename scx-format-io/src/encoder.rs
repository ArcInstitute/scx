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
/// `V2` = the adaptive profiles (`auto`/`compact`), target-biased via
/// [`pick_codec_v2`].
enum FramedPick {
    Trial,
    V2(DecodeTarget),
}

/// Options and metadata for encoding a single CSR shard via
/// [`encode_one_shard`] / [`encode_one_shard_from_bytes`].
#[derive(Debug, Clone)]
pub struct EncodeShardOptions {
    pub explicit_codec: Option<CodecId>,
    pub index_dtype: u8,
    pub n_vars: u64,
    pub global_row_offset: u64,
    pub section_type: SectionType,
    pub modality_type: ModalityType,
    pub name: String,
    pub framing: Option<FramingConfig>,
    pub value_encoding: Option<ValueEncoding>,
}

impl EncodeShardOptions {
    /// `index_dtype` is a required argument, not a field set after
    /// construction: it is a correctness-required, file-wide choice (u16 vs
    /// u32 index packing) with no safe default — silently defaulting it to
    /// `0` let a forgetful call site encode u16 indices for a file with
    /// `n_vars > 65535` with no error. Pass the same value every existing
    /// call site already derives (typically `if n_vars <= 65535 { 0 } else
    /// { 1 }`, or an input file's carried `index_dtype`).
    ///
    /// There is deliberately no `Default` impl: one would have to pick a
    /// value for `index_dtype`, reopening exactly the hazard above via
    /// `EncodeShardOptions { n_vars: 70_000, ..Default::default() }`.
    pub fn new(
        name: impl Into<String>,
        section_type: SectionType,
        n_vars: u64,
        global_row_offset: u64,
        index_dtype: u8,
    ) -> Self {
        Self {
            name: name.into(),
            section_type,
            n_vars,
            global_row_offset,
            index_dtype,
            explicit_codec: None,
            modality_type: ModalityType::Rna,
            framing: None,
            value_encoding: None,
        }
    }
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
/// `opts.value_encoding`: when `Some(enc)`, the shard's values are encoded
/// with `enc` instead of the per-shard auto-detected narrowest encoding —
/// this is what lets a parallel encode reproduce a *file-wide* value
/// encoding and thus produce byte-identical output to a path that fixed
/// the encoding up front (e.g. `scx-ops` grouped-write fast path vs.
/// `CsrEmitter`). `None` preserves the auto-detect behaviour. The
/// byte-oriented [`encode_one_shard_from_bytes`] sibling takes its value
/// encoding from its own `shard_value_encoding` argument instead and does
/// **not** consult this field.
///
/// # Examples
///
/// ```
/// use scx_format_io::encoder::{encode_one_shard, EncodeShardOptions};
/// use scx_format_io::section::SectionType;
/// use scx_format_io::modality::ModalityType;
///
/// // Canonical CSR: indptr[0] == 0, indices sorted per row, no dup, no zeros.
/// let indptr = [0u64, 2];
/// let indices = [0u32, 2];
/// let values = [1.0f32, 3.0];
/// let mut opts = EncodeShardOptions::new("X", SectionType::CsrShard, 3, 0, 1);
/// opts.modality_type = ModalityType::Rna;
/// let section = encode_one_shard(&indptr, &indices, &values, &opts).unwrap();
/// assert!(section.section_length > 0);
/// ```
pub fn encode_one_shard(
    shard_indptr: &[u64],
    shard_indices: &[u32],
    shard_values: &[f32],
    opts: &EncodeShardOptions,
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

    // 3. Determine value encoding (caller override wins) and encode values.
    let shard_value_encoding: ValueEncoding = opts
        .value_encoding
        .unwrap_or_else(|| detect_value_encoding(shard_values));
    let shard_values_bytes = values_to_raw_bytes(shard_values, shard_value_encoding)?;

    // The remaining steps (codec selection, framed/unframed encode, block index,
    // checksums, stats) operate purely on value BYTES — no f32 dependency — so
    // they live in the byte-oriented helper below, shared with callers (e.g. rscx)
    // that already hold raw LE value bytes + a fixed ValueEncoding.
    encode_one_shard_from_bytes(
        shard_indptr,
        shard_indices,
        &shard_values_bytes,
        shard_value_encoding,
        opts,
    )
}

/// Byte-oriented sibling of [`encode_one_shard`]: encode a CSR shard whose values
/// are already serialized to raw little-endian bytes under a fixed
/// [`ValueEncoding`]. This is the shared adaptive core — codec selection
/// (heuristic or explicit), row-group framing with the `auto`/`compact` adaptive
/// [`pick_codec_v2`] bias (or `compact-trial` dual-encode), block index,
/// checksums, and shard stats — reused by the f32 entry point
/// ([`encode_one_shard`]) and by callers that hold raw bytes directly (e.g. rscx, which serializes counts
/// f64→uN and would lose >2^24 counts on an f32 round-trip).
///
/// `shard_values_bytes` MUST equal
/// `values_to_raw_bytes(values, shard_value_encoding)` for canonical CSR values;
/// the caller owns canonicalization (this fn does not re-validate, unlike the f32
/// [`encode_one_shard`] debug assert).
pub fn encode_one_shard_from_bytes(
    shard_indptr: &[u64],
    shard_indices: &[u32],
    shard_values_bytes: &[u8],
    shard_value_encoding: ValueEncoding,
    opts: &EncodeShardOptions,
) -> Result<PreEncodedSection, ScxError> {
    let index_dtype_u16 = opts.index_dtype == 0;

    // 4. Select codec (heuristic when not explicit).
    let mut shard_codec = match opts.explicit_codec {
        Some(codec_id) => {
            if codec_id == CodecId::Scx1 && !shard_value_encoding.is_integer() {
                CodecId::Zstd
            } else {
                codec_id
            }
        }
        None => {
            select_codec_for_modality(shard_values_bytes, shard_value_encoding, opts.modality_type)
        }
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
    let (encoded, block_index, shard_version, chosen_codec) = encode_shard_adaptive(
        shard_indptr,
        shard_indices,
        shard_values_bytes,
        shard_codec,
        shard_value_encoding,
        index_dtype_u16,
        opts.framing,
    )?;
    shard_codec = chosen_codec;
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

    // `n_vars` is `u64` on `EncodeShardOptions` (matching `FileHeader`), but the
    // shard header's `n_minor` is `u32` — mirror the same guard `ScxWriter`'s
    // internal shard writers use rather than truncating silently, which would
    // desync the header's column count from the `u64` value `compute_shard_stats`
    // below records unchanged.
    if opts.n_vars > u32::MAX as u64 {
        return Err(ScxError::NVarsOverflow(opts.n_vars));
    }

    let shard_header = ShardHeader {
        magic: SHARD_MAGIC,
        shard_format_version: shard_version,
        shard_type: 0, // CSR
        codec_id: shard_codec as u8,
        value_encoding: shard_value_encoding as u8,
        index_dtype: opts.index_dtype,
        reserved_flags: [0; 3],
        n_major,
        n_minor: opts.n_vars as u32,
        nnz,
        global_offset: opts.global_row_offset,
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
        shard_values_bytes,
        shard_value_encoding,
        MajorAxis::Row,
        opts.global_row_offset,
        n_major as u64,
        opts.n_vars,
        nnz,
    );

    Ok(PreEncodedSection {
        encoded,
        block_index_bytes,
        header_buf,
        section_checksum,
        section_length,
        stats,
        name: opts.name.clone(),
        section_type: opts.section_type,
        nnz,
    })
}

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
/// `decode_target` is the adaptive alternative to `trial`, backing the
/// `auto`/`compact` intent profiles: when `Some`, the per-shard integer codec is
/// picked by [`pick_codec_v2`] using the same dual-encode as `trial` but biased
/// by the profile (`Auto` adopts ShufDeltaZstd only when it wins by
/// [`pick_codec_v2`]'s margin; `Storage` adopts on ≤ ties). `None` = heuristic
/// single-encode (`fast`, or an explicit codec — no ShufDeltaZstd trial).
/// `decode_target` takes precedence over `trial` (the CLI/pyscx layers keep them
/// mutually exclusive).
///
/// # Contract: `decode_target = Some(_)` authorises codec re-selection
///
/// Setting `decode_target` (or `trial`) grants every write path that consumes
/// this config — [`encode_one_shard`] *and* [`crate::ScxWriter::write_csr_shard`]
/// — permission to **override the `codec_id` the caller passed in** for integer
/// shards, via the dual-encode in [`encode_shard_adaptive`].
///
/// Callers that mean to **preserve** a source shard's codec (rather than pick a
/// new one) MUST therefore pass `decode_target: None` — i.e.
/// `FramingConfig::default()`. `scx_ops::rewrite_helpers::copy_layers` relies
/// on this: it re-writes shards at the codec read off the source header, and
/// re-selection would silently defeat that. (`scx_ops::build_csc` did too
/// while it rewrote the CSR; it now appends only the sidecar, and clears
/// `decode_target` itself.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// (`IngestOptions::default`, the CLI `--row-group-rows` default, and the pyscx
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

/// Encode a shard with adaptive codec selection under optional framing.
///
/// Under framing, when `fc.decode_target` is `auto` or `compact`, an integer
/// shard is dual-encoded against `ShufDeltaZstd` and **the returned codec may
/// differ from `seed_codec`**. With `decode_target: None` and `trial: false` —
/// the `fast` profile, and every explicit codec — the returned codec always
/// equals `seed_codec`.
///
/// Callers MUST stamp the returned codec into the shard header rather than the
/// one they passed in; it is what the reader dispatches on.
pub fn encode_shard_adaptive(
    indptr: &[u64],
    indices: &[u32],
    values_bytes: &[u8],
    seed_codec: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    framing: Option<FramingConfig>,
) -> Result<(EncodedShard, BlockIndex, u8, CodecId), ScxError> {
    let n_major = (indptr.len().saturating_sub(1)) as u32;
    match framing {
        Some(fc) if fc.row_group_rows > 0 => {
            let frame = |codec: CodecId| {
                encode_shard_framed(
                    indptr,
                    indices,
                    values_bytes,
                    codec,
                    value_encoding,
                    index_dtype_u16,
                    fc,
                )
            };
            // Decide whether to trial-encode ShufDeltaZstd as a second candidate.
            // The adaptive `decode_target` (`auto`/`compact`) takes precedence over
            // `trial` (compact-trial): the adaptive profiles trial only integer
            // shards (`pick_codec_v2` biases the pick by target); compact-trial
            // trials any non-ShufDeltaZstd heuristic. In all cases both candidates
            // are row-group-framed (shard v2) with codec-agnostic BlockIndex access.
            let is_integer = value_encoding.is_integer();
            let pick_mode: Option<FramedPick> = if seed_codec == CodecId::ShufDeltaZstd {
                None
            } else if let Some(dt) = fc.decode_target {
                is_integer.then_some(FramedPick::V2(dt))
            } else if fc.trial {
                Some(FramedPick::Trial)
            } else {
                None
            };
            match pick_mode {
                Some(mode) => {
                    // Two independent full encodes of the same input, compared
                    // only by size — so they run concurrently. `?` is applied
                    // in the serial order, so the seed candidate's error still
                    // wins.
                    //
                    // This is where the change costs peak memory, and the cost
                    // compounds with the parallel groups below rather than
                    // being free. Serially, one candidate's internal transient
                    // (its encoded groups plus the streams assembled from them)
                    // had collapsed to a single `EncodedShard` before the other
                    // started, so the peak was ~2x one shard's encoded bytes.
                    // Under `join` the two transients overlap: ~4x.
                    // `scx-convert/src/budget.rs` carries the measured figures
                    // and which codecs can reach the worst case.
                    #[cfg(feature = "parallel")]
                    let (h, s) =
                        rayon::join(|| frame(seed_codec), || frame(CodecId::ShufDeltaZstd));
                    #[cfg(not(feature = "parallel"))]
                    let (h, s) = (frame(seed_codec), frame(CodecId::ShufDeltaZstd));
                    let (e_h, bi_h) = h?;
                    let (e_s, bi_s) = s?;
                    let (size_h, size_s) = (framed_size(&e_h), framed_size(&e_s));
                    let pick_shufdelta = match mode {
                        FramedPick::Trial => size_s < size_h,
                        FramedPick::V2(dt) => {
                            pick_codec_v2(seed_codec, dt, size_h, size_s, true)
                                == CodecId::ShufDeltaZstd
                        }
                    };
                    let (codec, enc, bi) = if pick_shufdelta {
                        (CodecId::ShufDeltaZstd, e_s, bi_s)
                    } else {
                        (seed_codec, e_h, bi_h)
                    };
                    Ok((enc, bi, CURRENT_SHARD_FORMAT_VERSION, codec))
                }
                None => {
                    let (enc, bi) = frame(seed_codec)?;
                    Ok((enc, bi, CURRENT_SHARD_FORMAT_VERSION, seed_codec))
                }
            }
        }
        _ => {
            let enc = encode_shard(
                indptr,
                indices,
                values_bytes,
                seed_codec,
                value_encoding,
                index_dtype_u16,
            )?;
            Ok((
                enc,
                BlockIndex::for_shard(n_major, indptr)?,
                DEFAULT_WRITE_SHARD_FORMAT_VERSION,
                seed_codec,
            ))
        }
    }
}

/// Encode a shard **row-group-framed** (F5-b):
/// partition the major axis into groups (≤ `framing.row_group_rows` rows and, if set,
/// ≤ `framing.target_nnz` nnz — always ≥1 row), encode each group independently as a
/// standalone sub-shard via [`encode_shard`], and concatenate the three
/// sub-streams while recording per-group byte offsets in a multi-entry
/// [`BlockIndex`]. Codec-agnostic: a group decodes via
/// `scx_codec::decode_row_group` (→ the ordinary per-shard decoder). For
/// `CodecId::None` this is byte-identical to the legacy contiguous layout with a
/// framed indptr.
pub fn encode_shard_framed(
    indptr: &[u64],
    indices: &[u32],
    values_bytes: &[u8],
    codec: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    framing: FramingConfig,
) -> Result<(EncodedShard, BlockIndex), ScxError> {
    let n_rows = indptr.len().saturating_sub(1);
    // The group loop below runs once per row group, so zero rows means zero
    // entries — and `resolve_block_index` rejects an empty index on every read
    // path. Such a shard writes and checksums cleanly and then fails every read,
    // so refuse it here rather than emitting it. This is the single seam both
    // write paths funnel through: `write_shard_inner` and `encode_one_shard*`
    // (the subset path, which bypasses the writer) both arrive via
    // `encode_shard_adaptive`.
    //
    // Only the *framed* encoding is affected. An unframed zero-row shard emits
    // the legacy single-entry index, is never resolved, and stays legal.
    if n_rows == 0 {
        return Err(ScxError::ZeroRowFramedShard);
    }
    let w_v = value_encoding.byte_width();
    let g = framing.row_group_rows.clamp(1, MAX_BLOCK_ROWS) as usize;
    let nnz_cap = framing.target_nnz.unwrap_or(u64::MAX);

    // Pass 1 — group boundaries, serially and before any encode. The growth
    // loop peeks one row past `r1`, so the boundaries are data-dependent
    // whenever `target_nnz` is set and cannot be derived per group in
    // isolation. Deriving them here is what makes pass 2 order-free.
    let mut bounds: Vec<(usize, usize)> = Vec::with_capacity(n_rows.div_ceil(g).max(1));
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
        bounds.push((r0, r1));
        r0 = r1;
    }

    // Pass 2 — encode each group. Every group reads disjoint slices of the
    // caller's three inputs and writes only its own `EncodedShard`, so the
    // work is order-free and the bytes depend on nothing but the group's own
    // rows: running the groups concurrently cannot change the output.
    let encode_group = |&(r0, r1): &(usize, usize)| -> Result<EncodedShard, ScxError> {
        let base = indptr[r0];
        let start = base as usize;
        let end = indptr[r1] as usize;
        let local_indptr: Vec<u64> = indptr[r0..=r1].iter().map(|&v| v - base).collect();
        Ok(encode_shard(
            &local_indptr,
            &indices[start..end],
            &values_bytes[start * w_v..end * w_v],
            codec,
            value_encoding,
            index_dtype_u16,
        )?)
    };
    // `Vec<Result<_>>`, not `collect::<Result<Vec<_>, _>>()`: rayon does not
    // define *which* error a short-circuiting collect returns, and the loop
    // this replaced returned the lowest-indexed group's. Pass 3 takes the
    // first `Err` in group order, which keeps that exactly.
    //
    // The `cfg` switches the iterator and nothing else — passes 1 and 3 are
    // shared — so the sequential build cannot drift from the parallel one.
    // Two implementations could, and the proof that these do not is direct:
    // `cargo test -p scx-format-io --no-default-features` builds and runs 366
    // tests including `encoder_framed_tests`, so the same byte pin runs in
    // both configurations and reports the same digests. CI clippies three
    // `--all-targets` legs of this crate (with `deletion-vectors`, with
    // `parallel`, and with neither).
    #[cfg(feature = "parallel")]
    let groups: Vec<Result<EncodedShard, ScxError>> = {
        use rayon::prelude::*;
        bounds.par_iter().map(encode_group).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let groups: Vec<Result<EncodedShard, ScxError>> = bounds.iter().map(encode_group).collect();

    // Pass 3 — concatenate in group order and record each group's offsets
    // before its bytes are appended, exactly as the serial loop did.
    //
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
    // Surface a failed group **before** reserving anything. The three streams
    // below are sized for the whole shard, which at census scale is hundreds of
    // megabytes — allocating them only to drop them on the next line can turn a
    // recoverable encode error into an OOM.
    //
    // A *sequential* `collect` into `Result` short-circuits on the first `Err`
    // in index order, which is the precedence the serial loop had. That is
    // exactly the promise rayon's parallel collect does not make, and the
    // reason pass 2 collects `Vec<Result<_>>` and the conversion happens here
    // instead of there.
    let encoded: Vec<EncodedShard> = groups.into_iter().collect::<Result<Vec<_>, _>>()?;

    // Exact sizes, now that every group is encoded: the three streams are
    // allocated once rather than doubling-grown. The overshoot this removes is
    // load-bearing beyond tidiness — see `scx-convert/src/budget.rs` on the
    // transient these buffers form with the group results they are built from.
    let (ip_len, ix_len, vv_len) = encoded
        .iter()
        .fold((0usize, 0usize, 0usize), |(a, b, c), e| {
            (
                a + e.indptr_bytes.len(),
                b + e.indices_bytes.len(),
                c + e.values_bytes.len(),
            )
        });
    // Bound the *totals* here, before reserving them. `offset_u32` below runs
    // before each append, so it validates every group's starting offset and
    // never the last group's end: a >4 GiB sub-stream whose final group starts
    // under the limit slipped through — and did so before this function was
    // parallelised, so this closes a pre-existing gap rather than one the
    // change opened. Checking the fold also means the multi-gigabyte reserve
    // never happens on the way to the error.
    for (len, name) in [(ip_len, "indptr"), (ix_len, "indices"), (vv_len, "values")] {
        if u32::try_from(len).is_err() {
            return Err(ScxError::ShardStreamTooLarge(format!(
                "framed {name} sub-stream is {len} bytes, which exceeds u32::MAX"
            )));
        }
    }
    let mut indptr_stream: Vec<u8> = Vec::with_capacity(ip_len);
    let mut indices_stream: Vec<u8> = Vec::with_capacity(ix_len);
    let mut values_stream: Vec<u8> = Vec::with_capacity(vv_len);
    let mut entries = Vec::with_capacity(bounds.len());

    for (&(r0, r1), enc) in bounds.iter().zip(encoded) {
        let nnz_in_block = indptr[r1] - indptr[r0];
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
    }

    Ok((
        EncodedShard {
            indptr_bytes: indptr_stream,
            indices_bytes: indices_stream,
            values_bytes: values_stream,
        },
        BlockIndex { entries },
    ))
}

#[cfg(test)]
mod adaptive_codec_tests {
    use super::*;
    use scx_codec::CodecId;

    /// Framing emits one `BlockIndexEntry` per row group, so a zero-row shard
    /// makes the group loop never run and produces an **empty** block index —
    /// which every read path rejects via `resolve_block_index`. The shard writes
    /// and checksums cleanly, then fails every read.
    ///
    /// Refuse it at the encoder, which is the one function that would produce
    /// the empty index and the seam both write paths funnel through
    /// (`write_shard_inner` → `encode_shard_adaptive` → here, and
    /// `encode_one_shard*` → the same, which is the subset path that bypasses
    /// `write_shard_inner` entirely).
    #[test]
    fn zero_row_framed_shard_is_refused_at_write() {
        let err = encode_shard_framed(
            &[0u64],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            false,
            FramingConfig {
                row_group_rows: 4,
                ..Default::default()
            },
        )
        .expect_err("framing a zero-row shard must be refused");
        assert!(
            matches!(err, ScxError::ZeroRowFramedShard),
            "expected ZeroRowFramedShard, got {err:?}"
        );

        // Same refusal through the adaptive wrapper, which is what the writers
        // actually call.
        let err = encode_shard_adaptive(
            &[0u64],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            false,
            Some(FramingConfig {
                row_group_rows: 4,
                ..Default::default()
            }),
        )
        .expect_err("framing a zero-row shard must be refused via the adaptive path");
        assert!(matches!(err, ScxError::ZeroRowFramedShard));
    }

    /// The control the guard above needs: an **unframed** zero-row shard is
    /// legal and stays legal. Its indptr stream is a valid `[0]` and no block
    /// index is involved, so nothing about it is unreadable — rejecting it too
    /// would break every writer that emits an empty matrix (e.g. a 0-row
    /// `optimize` output). Without this arm the guard could over-reject and the
    /// test above would not notice.
    #[test]
    fn zero_row_unframed_shard_is_still_accepted() {
        let (_enc, bi, version, _codec) = encode_shard_adaptive(
            &[0u64],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            false,
            None,
        )
        .expect("an unframed zero-row shard is legal");
        assert_eq!(
            version,
            crate::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION,
            "unframed shards stay v1"
        );
        // The legacy single-entry index, not the empty one framing would emit.
        // `resolve_block_index` is never called on a v1 shard, so this entry's
        // all-zero offsets are inert — which is precisely why the unframed
        // zero-row shard was always readable and the framed one never was.
        assert_eq!(bi.entries.len(), 1);
        assert_eq!(bi.entries[0].n_rows, 0);
    }

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
        let mut opts =
            EncodeShardOptions::new("X_shard_0", SectionType::CsrShard, n_cols as u64, 0, 0);
        opts.framing = Some(framing);
        opts.value_encoding = Some(ValueEncoding::Uint8);
        encode_one_shard(indptr, indices, values, &opts).expect("encode")
    }

    /// The `fast` profile resolves to `decode_target: None` — a plain heuristic
    /// single-encode that keeps the heuristic winner and never runs the
    /// ShufDeltaZstd dual-encode, even on a shard where ShufDeltaZstd would win
    /// (which `auto`/`compact` adopt — see `auto_and_compact_adopt_shufdelta_on_large_win`).
    #[test]
    fn fast_keeps_heuristic_even_when_shufdelta_would_win() {
        // Large-count shard: heuristic is Zstd and ShufDeltaZstd beats it by a
        // wide margin, so the adaptive profiles adopt it. `fast` must not.
        let (indptr, indices, values) = gen_int_shard(2048, 60, 20000, 255);
        let fast = FramingConfig {
            row_group_rows: 256,
            target_nnz: None,
            trial: false,
            decode_target: None,
        };
        let encoded = encode(&indptr, &indices, &values, 20000, fast);
        assert_eq!(
            encoded.codec_id(),
            CodecId::Zstd as u8,
            "fast (decode_target=None) must keep the heuristic Zstd, never trial ShufDeltaZstd"
        );
        assert!(encoded.section_length > 0);
    }

    /// The `auto` (cost-aware) and `compact` (tie-adopt) profiles both adopt
    /// ShufDeltaZstd on this census-like integer shard, where it wins by well
    /// more than `ADOPT_MARGIN`, and neither exceeds the heuristic size — the
    /// "no size regression vs the heuristic" guard.
    #[test]
    fn auto_and_compact_adopt_shufdelta_on_large_win() {
        // Larger counts (median > 8) → heuristic Zstd, which ShufDeltaZstd's
        // delta+shuffle on the structured index/value streams reliably beats by
        // a wide margin (far past ADOPT_MARGIN).
        let (indptr, indices, values) = gen_int_shard(2048, 60, 20000, 255);
        let base = FramingConfig {
            row_group_rows: 256,
            target_nnz: None,
            trial: false,
            decode_target: None,
        };
        let heuristic = encode(&indptr, &indices, &values, 20000, base);
        assert_eq!(
            heuristic.codec_id(),
            CodecId::Zstd as u8,
            "large-count heuristic is Zstd"
        );
        for dt in [DecodeTarget::Auto, DecodeTarget::Storage] {
            let adaptive = encode(
                &indptr,
                &indices,
                &values,
                20000,
                FramingConfig {
                    decode_target: Some(dt),
                    ..base
                },
            );
            assert_eq!(
                adaptive.codec_id(),
                CodecId::ShufDeltaZstd as u8,
                "{dt:?} should adopt ShufDeltaZstd on a large win"
            );
            assert!(
                adaptive.section_length <= heuristic.section_length,
                "{dt:?} ({}) must not exceed the heuristic ({})",
                adaptive.section_length,
                heuristic.section_length
            );
            assert!(adaptive.section_length > 0);
        }
    }

    /// Float modality stays Pcodec under every decode target (ShufDeltaZstd never
    /// competes for float).
    #[test]
    fn adaptive_float_stays_pcodec() {
        let (indptr, indices, _) = gen_int_shard(256, 20, 5000, 4);
        let nnz = *indptr.last().unwrap() as usize;
        let values: Vec<f32> = (0..nnz).map(|i| (i as f32) * 0.5 + 0.25).collect();
        let base = FramingConfig {
            row_group_rows: 256,
            target_nnz: None,
            trial: false,
            decode_target: None,
        };
        for dt in [DecodeTarget::Auto, DecodeTarget::Storage] {
            let mut opts = EncodeShardOptions::new("X_shard_0", SectionType::CsrShard, 5000, 0, 0);
            opts.framing = Some(FramingConfig {
                decode_target: Some(dt),
                ..base
            });
            opts.value_encoding = Some(ValueEncoding::Float32);
            let sec = encode_one_shard(&indptr, &indices, &values, &opts).expect("encode");
            assert_eq!(
                sec.codec_id(),
                CodecId::Pcodec as u8,
                "float must stay Pcodec under {dt:?}"
            );
        }
    }

    /// The byte-oriented [`encode_one_shard_from_bytes`] must produce a
    /// byte-identical `PreEncodedSection` to the f32 [`encode_one_shard`]
    /// for the same canonical CSR + value encoding, proving the extraction is a
    /// no-op refactor. Checked across unframed, `fast`, and adaptive `auto`/`compact`
    /// framings so the shared adaptive core is exercised on both paths.
    #[test]
    fn from_bytes_matches_f32_path_byte_for_byte() {
        let (indptr, indices, values) = gen_int_shard(2048, 60, 20000, 255);
        let enc = ValueEncoding::Uint8;
        let bytes = values_to_raw_bytes(&values, enc).expect("values_to_raw_bytes");
        let base = FramingConfig {
            row_group_rows: 256,
            target_nnz: None,
            trial: false,
            decode_target: None,
        };
        let framings: [Option<FramingConfig>; 4] = [
            None, // unframed
            Some(base),
            Some(FramingConfig {
                decode_target: Some(DecodeTarget::Auto),
                ..base
            }),
            Some(FramingConfig {
                decode_target: Some(DecodeTarget::Storage),
                ..base
            }),
        ];
        for framing in framings {
            let mut opts = EncodeShardOptions::new("X_shard_0", SectionType::CsrShard, 20000, 0, 0);
            opts.framing = framing;
            opts.value_encoding = Some(enc);
            let f32_sec = encode_one_shard(&indptr, &indices, &values, &opts).expect("f32 encode");
            let bytes_sec = encode_one_shard_from_bytes(&indptr, &indices, &bytes, enc, &opts)
                .expect("bytes encode");
            assert_eq!(
                f32_sec.section_checksum, bytes_sec.section_checksum,
                "section checksum diverged for framing {framing:?}"
            );
            assert_eq!(
                f32_sec.section_length, bytes_sec.section_length,
                "section length diverged for framing {framing:?}"
            );
            assert_eq!(f32_sec.codec_id(), bytes_sec.codec_id());
        }
    }

    /// `opts.n_vars` is `u64` (matching `FileHeader`), but the shard header's
    /// `n_minor` is `u32`. A value past `u32::MAX` must be rejected with
    /// `NVarsOverflow`, not silently truncated into a header/stats mismatch.
    #[test]
    fn n_vars_past_u32_max_is_rejected_not_truncated() {
        let mut opts = EncodeShardOptions::new(
            "X_shard_0",
            SectionType::CsrShard,
            u32::MAX as u64 + 1,
            0,
            1,
        );
        opts.value_encoding = Some(ValueEncoding::Uint8);
        match encode_one_shard(&[0u64, 1], &[0u32], &[1.0f32], &opts) {
            Err(ScxError::NVarsOverflow(n)) => assert_eq!(n, u32::MAX as u64 + 1),
            Err(other) => panic!("expected NVarsOverflow, got {other:?}"),
            Ok(_) => panic!("n_vars past u32::MAX must be refused, not truncated"),
        }
    }
}

#[cfg(test)]
#[path = "encoder_framed_tests.rs"]
mod encoder_framed_tests;
