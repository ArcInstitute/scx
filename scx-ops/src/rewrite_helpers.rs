// Shared helpers for copying auxiliary sections between SCX files.
//
// Used by build_csc and upgrade to avoid duplicating layer/obsm/uns/
// predicate-index/provenance copy logic.

use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::catalog::{FullCatalogEntry, ShardStats};
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::shard::{ShardHeader, DEFAULT_WRITE_SHARD_FORMAT_VERSION, SHARD_HEADER_SIZE};
use scx_format_io::writer::ScxWriter;
use scx_format_io::ResolvedCodec;
use scx_format_io::{compute_shard_stats, MajorAxis, ScxReader};

use crate::error::{OpsError, Result as OpsResult};

/// Core eligibility for raw-copying a CSR shard's section bytes verbatim:
/// the source shard's index dtype, column extent, and codec all match what
/// the target would emit, so a byte-copy reproduces the decode/re-encode
/// output. Callers AND their own extra preconditions onto this:
/// `append` adds `sh.n_major <= shard_target_rows` (it may re-split shards);
/// `merge` adds `!assume_identical_var` (column indices are not guaranteed
/// identical when the var axis is assumed-but-not-verified equal).
///
/// **Framing gate (`output_framed`).** A shard may only be byte-copied when its
/// framing matches the output file's: a framed (shard v2) shard into a v4 file,
/// or an unframed (v1) shard into a ≤v3 file. A framing mismatch forces the
/// decode-encode path, which re-emits the shard at the output's framing so the
/// file stays self-consistent (and passes
/// [`ScxWriter::guard_no_legacy_shard_in_v4`], which rejects an unframed CSR
/// shard in a v4 file). Byte-copying a v2 shard into a ≤v3 file would leave a
/// framed shard under a header a pre-framing reader accepts, which then
/// mis-decodes the per-group local-rebased indptr as global; byte-copying a v1
/// shard into a v4 file would advertise sub-shard random access it cannot honor.
/// **Codec gate.** A byte-copy preserves the source shard's codec exactly, so
/// whether it is eligible depends on what the caller's intent asked for:
///
/// | intent | raw-copy | why |
/// |---|---|---|
/// | explicit codec | only if `sh.codec_id` already matches | the copy would ignore the force |
/// | `auto` / `fast` | **yes** | preserving the source codec is size-neutral and free; re-encoding to "re-decide" a codec the source already chose adaptively is pure cost |
/// | `compact` / `compact-trial` | **no** | the caller explicitly asked for a size-max re-encode; a byte-copy would silently ignore it |
pub(crate) fn raw_copy_csr_eligible(
    sh: &ShardHeader,
    target_index_dtype: u8,
    target_n_vars: u64,
    codec: ResolvedCodec,
    output_framed: bool,
) -> bool {
    let shard_framed = sh.shard_format_version > DEFAULT_WRITE_SHARD_FORMAT_VERSION;
    let codec_ok = match codec.explicit_codec {
        Some(c) => sh.codec_id == c as u8,
        // `compact`/`compact-trial` request a size-max re-encode; honour it.
        // `auto`/`fast` are satisfied by keeping what the source already has.
        None => !codec.requires_framing && !codec.codec_trial,
    };
    shard_framed == output_framed
        && sh.index_dtype == target_index_dtype
        && (sh.n_minor as u64) == target_n_vars
        && codec_ok
}

/// Build the patched section bytes + stats for a raw-copied CSR shard,
/// sink-agnostic.
///
/// Patches only `ShardHeader.n_minor` (→ `target_n_vars`) and `global_offset`
/// (→ `global_row_start`); the indptr/indices/values/block-index payload is
/// byte-identical to the source. The caller writes the returned bytes through
/// its own sink — the in-place `FileLock` (append) or
/// [`ScxWriter::write_csr_shard_raw_copy`] (merge) — and records the catalog
/// entry. Reuses the source entry's stats (only the row range is
/// position-dependent); decodes to recompute stats only when the source entry
/// is missing them (format-permitted but not produced by the current writer).
pub(crate) fn build_raw_copied_csr_section(
    source: &ScxReader,
    entry: &FullCatalogEntry,
    sh: &ShardHeader,
    target_n_vars: u64,
    global_row_start: u64,
    value_encoding: ValueEncoding,
) -> OpsResult<(Vec<u8>, ShardStats)> {
    // `read_raw_shard_bytes` returns an mmap slice. We never mutate it: the
    // patched header is built in a separate `header_buf` and the payload is
    // copied read-only into `section_data`, so borrow it directly (no alloc).
    let src_bytes = source.read_raw_shard_bytes(entry)?;
    if src_bytes.len() < SHARD_HEADER_SIZE {
        return Err(OpsError::Format(scx_format_io::ScxError::InvalidCatalog(
            format!("source shard '{}' too small for header", entry.name),
        )));
    }

    let new_sh = ShardHeader {
        n_minor: target_n_vars as u32,
        global_offset: global_row_start,
        ..sh.clone()
    };
    let mut header_buf = Vec::with_capacity(SHARD_HEADER_SIZE);
    new_sh.write_to(&mut header_buf)?;

    let mut section_data = Vec::with_capacity(src_bytes.len());
    section_data.extend_from_slice(&header_buf);
    section_data.extend_from_slice(&src_bytes[SHARD_HEADER_SIZE..]);

    // Reuse the source entry's stats; only the row range is position-dependent.
    // `nnz`, `value_min/max/sum`, `col_start/col_end`, and `column_stats` are
    // invariant under raw copy because eligibility already requires
    // `sh.n_minor == target_n_vars` (so the column extent is preserved).
    let stats = match entry.stats.as_ref() {
        Some(src_stats) => {
            let mut s = src_stats.clone();
            s.row_start = global_row_start;
            s.row_end = global_row_start + sh.n_major as u64;
            s
        }
        None => {
            let codec_id =
                CodecId::from_u8(sh.codec_id).ok_or(OpsError::UnknownCodec(sh.codec_id))?;
            if codec_id == CodecId::None {
                let values_start = sh.values_rel_offset as usize;
                let values_end = values_start + sh.values_length as usize;
                // Defend against a malformed/corrupt header: error rather than
                // panic on an out-of-bounds values slice (readers never panic
                // on bad input).
                if values_start > values_end || values_end > src_bytes.len() {
                    return Err(OpsError::Format(scx_format_io::ScxError::InvalidCatalog(
                        format!(
                            "source shard '{}' has invalid values offset/length \
                             ({values_start}..{values_end} of {} bytes)",
                            entry.name,
                            src_bytes.len(),
                        ),
                    )));
                }
                compute_shard_stats(
                    &src_bytes[values_start..values_end],
                    value_encoding,
                    MajorAxis::Row,
                    global_row_start,
                    sh.n_major as u64,
                    target_n_vars,
                    sh.nnz,
                )
            } else {
                let (_, _, val_f32) = source.read_shard_from_entry(entry)?;
                let raw = scx_codec::values_to_raw_bytes(&val_f32, value_encoding)?;
                compute_shard_stats(
                    &raw,
                    value_encoding,
                    MajorAxis::Row,
                    global_row_start,
                    sh.n_major as u64,
                    target_n_vars,
                    sh.nnz,
                )
            }
        }
    };

    Ok((section_data, stats))
}

/// Copy obs and var through a rewrite that preserves **both axes 1:1**,
/// keeping a row-sharded layout sharded.
///
/// This is the safe front door to `compact::write_obs_shards_streaming` /
/// `optimize::write_var_shards_streaming`, and the reason they are not exported
/// directly. The obs helper takes a `keep_mask` and a `total_kept`, and neither
/// is checkable from inside it: a mask shorter than the shards it is applied to
/// panics on the slice in `filtered_obs_shards`, and a wrong `total_kept` is
/// stamped into every output shard, producing a file whose own `n_rows_total`
/// disagrees with its cover. Both are fine for the in-crate callers that derive
/// them from `build_keep_mask(n_obs, …)`; neither is something a downstream
/// caller should have to know. So the filtering primitive stays crate-private
/// and this — the only case a rewrite outside `scx-ops` needs — takes no
/// arguments beyond the two files and derives the totals from the reader.
///
/// The alternative each caller reaches for otherwise is `read_obs()` +
/// `write_obs()`, which assembles the whole axis into one in-memory batch and
/// emits a single legacy section: peak RSS O(n_obs) — the OOM the sharded
/// layout exists to prevent — plus the silent loss of the row-sharded-obs
/// precondition Level-2 row-set pushdown depends on. `build_csc` and
/// `scx upgrade` had both written that by hand; this is the shared version.
///
/// A legacy single-section input has no per-shard reader and falls through to
/// the materialising path, which is what it already was.
pub fn copy_obs_var_preserving_layout(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let (n_obs, n_vars) = (reader.header().n_obs, reader.header().n_vars);
    if reader.obs_metadata_shard_count() > 0 {
        crate::compact::write_obs_shards_streaming(reader, writer, None, n_obs as usize)?;
    } else {
        writer.write_obs(&reader.read_obs()?)?;
    }
    if reader.var_metadata_shard_count() > 0 {
        crate::optimize::write_var_shards_streaming(reader, writer, n_vars)?;
    } else {
        writer.write_var(&reader.read_var()?)?;
    }
    Ok(())
}

/// The value encoding to re-emit canonicalized values under, given the one the
/// source shard used.
///
/// Canonicalization **sums duplicate coordinates**, so it can produce a value
/// larger than any the source held — and the source's encoding was chosen to fit
/// the source's values. Re-encoding through it has two failure modes, and the
/// quiet one is the dangerous one: `Uint8` refuses `200 + 200 = 400` outright, so
/// the upgrade fails on exactly the non-canonical input it exists to repair,
/// while `Float16` is unchecked — `f16::from_f32` maps anything past 65504 to
/// **infinity** and the op reports success.
///
/// So widen to the narrowest encoding that actually holds the result. Integers
/// stay integers (canonicalizing sums counts; it never makes them fractional)
/// and floats stay floats. Returns the source encoding unchanged whenever it
/// still fits, which is the overwhelmingly common case — this only widens for a
/// shard canonicalization actually rewrote.
pub fn encoding_for_canonicalized(source: ValueEncoding, data: &[f32]) -> ValueEncoding {
    // One pass, folding both limits together. `Float16`'s ceiling is a
    // magnitude while the integer ceilings are signed, so both are needed —
    // using the signed max for f16 is how the negative half went unguarded: it
    // starts at 0.0, so an all-negative shard reports `max == 0.0` and never
    // widens. The rungs below are a flat ladder rather than a recursive one
    // because recursing re-folded the whole slice per rung, and `copy_layers`
    // calls this on every layer shard of an upgrade.
    //
    // Folded in f64 to match `detect_value_encoding`; every threshold here is
    // exact in f32 too, so the width is convention rather than correctness —
    // the *constant* on the `Uint32` rung is what has to be right.
    let (max, max_abs) = data.iter().fold((0.0f64, 0.0f64), |(m, ma), &v| {
        let v = v as f64;
        (m.max(v), ma.max(v.abs()))
    });
    match source {
        // f16's finite ceiling. Past it `from_f32` yields ±inf, silently.
        ValueEncoding::Float16 if max_abs > 65504.0 => ValueEncoding::Float32,
        ValueEncoding::Uint8 | ValueEncoding::Uint16 | ValueEncoding::Uint32 => {
            if max > U32_DECODE_CEILING {
                ValueEncoding::Float32
            } else if max > u16::MAX as f64 {
                ValueEncoding::Uint32
            } else if max > u8::MAX as f64 {
                // Never narrower than the source: a shard whose values happen to
                // fit a smaller width keeps the encoding the file already chose.
                widest_of_two(source, ValueEncoding::Uint16)
            } else {
                source
            }
        }
        other => other,
    }
}

/// The wider of two **integer** encodings.
///
/// Only ever called with the source rung and the rung its values need, so the
/// ordering question is `Uint8 < Uint16 < Uint32` and nothing else; floats never
/// reach it.
fn widest_of_two(a: ValueEncoding, b: ValueEncoding) -> ValueEncoding {
    let rank = |e: ValueEncoding| match e {
        ValueEncoding::Uint8 => 0,
        ValueEncoding::Uint16 => 1,
        _ => 2,
    };
    if rank(b) > rank(a) {
        b
    } else {
        a
    }
}

/// The largest `f32` any on-disk `u32` decodes to: 2³², because `u32::MAX as
/// f32` rounds *up*. Values at or below it may be decoded originals; values
/// above it cannot be.
///
/// This is why [`encoding_for_canonicalized`]'s top rung is bounded here rather
/// than at `u32::MAX`, and the gap between the two is the whole point. Strictly
/// above 2³² a value is provably a sum, so widening is unambiguous — and
/// necessary, because `encode_f32` refuses such a value outright and would abort
/// the rewrite. **At** 2³² the provenances are indistinguishable: it is equally
/// the f32 image of a decoded `u32::MAX` and a sum that reached it. Keeping
/// `Uint32` is the better half of that irreducible choice — `as u32` saturates
/// the decoded value back to `u32::MAX` exactly, so the rewrite stays lossless
/// on a format-valid archive, whereas widening writes 2³², one larger and not a
/// value `u32` can hold at all, so the file stops reading as `uint32`. A sum
/// landing exactly on 2³² is written one low; that is the residue of the f32
/// round-trip these paths already have, not something this rung can fix.
///
/// **Only the rewrite paths are ambiguous**, and not because their values came
/// off disk — an external file can hand over a `u32`-derived 2³² just as easily.
/// What separates them is evidence: `attach_external_layer` runs
/// `scx_codec::detect_value_encoding` over its pre-canonical values, and that
/// detector sends 2³² to `Float32`, so a `Uint32` layer encoding *proves* no
/// input held the alias and any later 2³² is a sum. A rewrite has no such step —
/// its encoding comes from the shard header it is copying — so it cannot tell
/// the two apart and must not use the fresh-data rule.
const U32_DECODE_CEILING: f64 = (1u64 << 32) as f64;

/// The codec to pair with a value encoding [`encoding_for_canonicalized`] may
/// have widened.
///
/// `Scx1` encodes integers only, so `Scx1` + `Float32` is
/// [`scx_codec::CodecError::FloatWithScx1`] — the rewrite aborts on exactly the
/// input the widening exists to let through. `encode_one_shard_from_bytes`
/// already performs this downgrade for callers that pick a codec through it;
/// the rewrite writers (`write_csr_shard` / `write_layer_csr_shard`) pass their
/// codec straight to `encode_shard_adaptive`, which does not, so they have to
/// ask for it. Kept next to the ladder because the two decisions are one
/// decision: whenever the encoding can stop being integral, the codec can stop
/// being valid.
pub fn codec_for_canonicalized(source: CodecId, encoding: ValueEncoding) -> CodecId {
    if source == CodecId::Scx1 && !encoding.is_integer() {
        CodecId::Zstd
    } else {
        source
    }
}

/// Section families this file's copy helpers do **not** carry, checked against
/// the input so the loss can be reported rather than discovered later.
///
/// The list mirrors what `copy_auxiliary_sections` actually copies; keep the
/// two in step.
///
/// It used to be eleven entries long. Review §6.3 is why: `varm`, `obsp`,
/// `varp` and `.raw` are user data with no rebuild path, and both callers
/// rename a wholly new file over the target carrying no prior catalog, so
/// `scx rollback` could not bring any of it back. They are all carried now
/// (Phase 5b) — both callers preserve the global obs row space 1:1, which is
/// this helper's stated precondition and is exactly what makes a verbatim copy
/// of an obs-axis section sound.
///
/// What is left is one entry, and it is **unreachable through either caller**:
/// a `LayerCscShard` can only be written for a registered modality
/// (`write_layer_csc_shard_for` refuses `modality_id == 0`), and both callers
/// refuse multimodal input outright. It stays on the list rather than being
/// deleted because the list is what makes a *future* section type surface as a
/// named loss instead of a silent one — this is an allowlist, and that is the
/// bug class it was written against.
///
/// **Detection bitmaps** are deliberately not here. They are carried, except by
/// a rewrite that canonicalised the matrix underneath them — a condition, not a
/// family, so it lives in [`LayerCanonicalization`] and warns from
/// `copy_bitmaps`.
const DROPPED_SECTION_FAMILIES: &[(SectionType, &str)] = &[(
    SectionType::LayerCscShard,
    "layer CSC sidecars (rebuild: scx build-csc)",
)];

/// Warn, once per family, about input sections this rewrite is about to drop.
///
/// `copy_auxiliary_sections` is an allowlist, so anything it does not name is
/// dropped — silently, until this existed. Both its callers rename a wholly new
/// file over the target with no prior catalog, so `scx rollback` cannot recover
/// what goes missing.
fn warn_dropped_sections(reader: &ScxReader, action: &str) {
    let dropped = dropped_section_labels(reader);
    if !dropped.is_empty() {
        // No rollback clause: this helper does not know whether the caller is
        // writing to a separate output (where the input is untouched) or
        // renaming over it. Stating the loss and the remedy is true of both;
        // claiming irreversibility on the copy-out form would be the same wrong
        // rationale for a right warning that `run_upgrade`'s decline message
        // had. The in-place hazard is documented in docs/operations.md.
        log::warn!(
            "scx {action}: the output will not carry {} — rebuild them against the \
             output if you need them.",
            dropped.join(", ")
        );
    }
}

/// The distinct family labels this rewrite would drop from `reader`, in
/// [`DROPPED_SECTION_FAMILIES`] order. Split out from [`warn_dropped_sections`]
/// so the set can be asserted directly — a warning is only worth documenting if
/// it names the right things.
pub(crate) fn dropped_section_labels(reader: &ScxReader) -> Vec<&'static str> {
    let mut seen: Vec<&'static str> = Vec::new();
    for entry in &reader.catalog().entries {
        if let Some((_, label)) = DROPPED_SECTION_FAMILIES
            .iter()
            .find(|(ty, _)| *ty == entry.section_type)
        {
            if !seen.contains(label) {
                seen.push(label);
            }
        }
    }
    seen
}

/// Whether a rewrite canonicalizes what it re-emits, and — if it does —
/// whether canonicalizing the **primary matrix** actually changed anything.
///
/// One type rather than two `bool` parameters because the second question only
/// exists inside the first, and because it decides something a caller would
/// otherwise have to know to ask about: a detection bitmap records which genes
/// each row *stores*, so `drop_explicit_zeros_inplace` — which
/// `canonicalize_csr` runs — can remove a gene the sidecar still claims. A
/// bitmap carried across a rewrite that rewrote X therefore over-reports, and
/// `Experiment.detection_counts` answers from it with nothing to notice by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerCanonicalization {
    /// Re-emit layers exactly as they are.
    ///
    /// `build_csc`'s case: it deliberately clamps its output version to the
    /// source's (SCX-005) precisely so it does not have to canonicalize, and
    /// canonicalizing would change the nnz of a file it promises to re-emit
    /// unchanged. Nothing under the bitmaps moves, so they carry.
    Off,
    /// Re-sort, dedup-sum and zero-drop every layer CSR shard before
    /// re-encoding, widening the value encoding if the sums need it
    /// ([`encoding_for_canonicalized`]).
    ///
    /// `scx upgrade`'s case: it stamps the output `DEFAULT_WRITE_FORMAT_VERSION`
    /// and v3's contract *is* canonical CSR, so it must canonicalize what it
    /// re-emits. `x_was_rewritten` is the caller's report of whether
    /// canonicalizing X already changed the matrix — the same flag that decides
    /// whether the CSC sidecar can be carried, and for the same reason.
    On { x_was_rewritten: bool },
}

impl LayerCanonicalization {
    fn canonicalizes(self) -> bool {
        matches!(self, Self::On { .. })
    }

    /// Whether detection bitmaps written against the input still describe the
    /// output. False only when canonicalization actually rewrote X.
    fn bitmaps_still_valid(self) -> bool {
        !matches!(
            self,
            Self::On {
                x_was_rewritten: true
            }
        )
    }
}

/// Copy every auxiliary section from reader to writer, then append a new
/// provenance entry.
///
/// **Only valid for a rewrite that preserves the global obs row space 1:1** —
/// its two callers, `build_csc` and `scx upgrade`, both do. That precondition
/// is what licenses nearly everything here: v2 deletion vectors store global obs
/// row indices, detection bitmaps are keyed to CSR-shard-local rows, and the
/// group index records global output-row ranges, so all three stay valid exactly
/// as long as the row order and shard boundaries do. An op that drops or
/// reorders rows must remap instead (see `compact` / `sort`).
///
/// Carries: layers, `obsm`, `varm`, COO `obsp`/`varp`, the CSR-backed `obsp`
/// graph, `uns`, `adata.raw`, detection bitmaps, the grouped-sort group index,
/// the predicate-index sections, and the deletion vector.
///
/// Does **not** carry layer CSC sidecars (rebuild with `scx build-csc`), and
/// carries detection bitmaps only when canonicalization left the matrix alone —
/// see [`LayerCanonicalization`]. [`warn_dropped_sections`] reports whatever is
/// dropped rather than leaving the user to discover it; see
/// [`DROPPED_SECTION_FAMILIES`], which must be kept in step with what is
/// actually copied below.
///
/// **This is still an allowlist**, so a section type added to the format is
/// dropped here until someone adds it — which is the shape of the bug this
/// carry was written to fix, and why `scx_ops::carry`'s audit runs over the
/// output regardless of what this function believes it copied.
///
/// Layers are re-emitted as they are. For the canonicalizing variant — which
/// `scx upgrade` needs and `build_csc` must not have — see
/// [`copy_auxiliary_sections_canonicalizing`].
pub fn copy_auxiliary_sections(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    action: &str,
    params_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    copy_auxiliary_sections_canonicalizing(
        reader,
        writer,
        action,
        params_json,
        LayerCanonicalization::Off,
    )
}

/// [`copy_auxiliary_sections`] with control over canonicalization.
///
/// The fifth parameter was a bare `canonicalize: bool` until Phase 5b, split out
/// of the four-argument form rather than widening it because only one of the two
/// callers was making the choice. It is now [`LayerCanonicalization`], because
/// the caller that *is* making it also has to answer a second question the
/// bitmap carry depends on — and answering "did canonicalization change the
/// matrix?" with a separate `bool` would let a caller pass a combination
/// (`Off` + `changed`) that cannot happen.
pub fn copy_auxiliary_sections_canonicalizing(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    action: &str,
    params_json: &str,
    canonicalization: LayerCanonicalization,
) -> Result<(), Box<dyn std::error::Error>> {
    warn_dropped_sections(reader, action);
    copy_layers(reader, writer, canonicalization.canonicalizes())?;
    copy_dense_and_pairwise(reader, writer)?;
    copy_uns(reader, writer)?;
    copy_raw(reader, writer)?;
    copy_bitmaps(reader, writer, canonicalization, action)?;
    copy_group_index(reader, writer)?;
    copy_predicate_indices(reader, writer)?;
    copy_deletion_vectors(reader, writer)?;
    append_provenance(reader, writer, action, params_json)?;
    Ok(())
}

/// Copy `obsm` / `varm` / COO `obsp` / COO `varp` and the CSR-backed `obsp`
/// graph, byte for byte.
///
/// Verbatim rather than read-and-rewrite, matching `optimize`: it preserves a
/// sharded layout (`ObsmEmbeddingShard` and friends) as sharded, and it avoids
/// decoding a large `obsp`/`varp` graph into one in-memory section — which is
/// both an OOM risk and a way to hit Arrow IPC's 2 GB narrow-offset ceiling on a
/// graph that was sharded precisely to stay under it.
///
/// `ObspCsrShard` is included here, unlike in `optimize` — `optimize`
/// re-encodes it in its own CSR shard loop, whereas `build_csc`'s loop filters
/// to `SectionType::CsrShard` and never sees it. An upgrade re-emits it
/// unchanged rather than canonicalizing it; a pairwise graph is not what the v3
/// canonical-CSR contract is about, and rewriting a user's graph structure is
/// not something this op should do on the way past.
fn copy_dense_and_pairwise(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e.section_type,
                SectionType::ObsmEmbedding
                    | SectionType::ObsmEmbeddingShard
                    | SectionType::VarmEmbedding
                    | SectionType::VarmEmbeddingShard
                    | SectionType::ObspEmbedding
                    | SectionType::ObspEmbeddingShard
                    | SectionType::ObspCsrShard
                    | SectionType::VarpEmbedding
                    | SectionType::VarpEmbeddingShard
            )
        })
        .collect();
    // Sorted by name for deterministic output, as `optimize` does.
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    for entry in entries {
        let bytes = reader.section_bytes(entry)?;
        writer.copy_section_verbatim(entry, bytes)?;
    }
    Ok(())
}

/// Copy `adata.raw` — its CSR shards and its own var axis.
///
/// Raw shares the obs axis with X, which this helper's 1:1 precondition keeps
/// aligned; its `var` axis is its own and is copied with it. `optimize` does not
/// carry raw, so this is the one family where `build-csc` now carries more than
/// `optimize` does. That asymmetry is real and is recorded in the carry table
/// rather than smoothed over: `optimize`'s raw drop is a separate open item.
fn copy_raw(reader: &ScxReader, writer: &mut ScxWriter) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e.section_type,
                SectionType::RawCsrShard | SectionType::RawVarMetadata
            )
        })
        .collect();
    if entries.is_empty() {
        return Ok(());
    }
    // No `set_raw_n_vars` needed. That writer field exists to stamp `n_minor`
    // into a shard header being *built*; a verbatim copy carries the source's
    // header bytes, and `ScxReader::raw_n_vars` recovers the extent from the
    // shards' own stats rather than from anything on the file header.
    entries.sort_by_key(|e| e.stats.as_ref().map(|s| s.row_start).unwrap_or(0));
    for entry in entries {
        let bytes = reader.section_bytes(entry)?;
        writer.copy_section_verbatim(entry, bytes)?;
    }
    Ok(())
}

/// Copy the detection-bitmap sidecars, unless canonicalization invalidated them.
///
/// Bitmaps are keyed to CSR-shard-local rows and read back in `row_start` order
/// (`bitmap_shards_for_modality`), so a rewrite that preserves row order and
/// shard boundaries keeps them valid — which is this helper's precondition.
///
/// What it does *not* survive is canonicalization changing the matrix.
/// `BitmapShard::build_from_csr` records a gene for a row if the row **stores**
/// that column, regardless of the value there, and `canonicalize_csr` drops
/// explicit zeros. So a sidecar carried across a canonicalizing rewrite of a
/// non-canonical input claims genes the output no longer has, and
/// `detection_counts` / `cells_expressing` over-report from it silently. Dropped
/// with a warning in exactly that case, and only that case — which is rare,
/// since `canonicalize_csr` short-circuits on already-canonical input and every
/// file a current writer produces is canonical.
fn copy_bitmaps(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    canonicalization: LayerCanonicalization,
    action: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::BitmapShard)
        .collect();
    if entries.is_empty() {
        return Ok(());
    }
    if !canonicalization.bitmaps_still_valid() {
        log::warn!(
            "scx {action}: canonicalizing the matrix changed which genes each row \
             stores, so the detection bitmaps built against the old matrix would \
             over-report and have been dropped. Rebuild them with `scx sort --bitmap \
             always` or a re-convert if you need `detection_counts`."
        );
        return Ok(());
    }
    entries.sort_by_key(|e| e.stats.as_ref().map(|s| s.row_start).unwrap_or(0));
    for entry in entries {
        let bytes = reader.section_bytes(entry)?;
        writer.copy_section_verbatim(entry, bytes)?;
    }
    Ok(())
}

/// Copy the F1 grouped-sharding sidecar.
///
/// It records **global** output-row ranges plus shard indices, so it survives
/// any rewrite that keeps both — which this helper requires. Contrast `append`,
/// which extends the row universe past what the index describes and must drop
/// it, and `compact`/`sort`, which re-shard or permute.
fn copy_group_index(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(entry) = reader
        .catalog()
        .entries
        .iter()
        .find(|e| e.section_type == SectionType::GroupIndex)
    {
        let bytes = reader.section_bytes(entry)?;
        writer.copy_section_verbatim(entry, bytes)?;
    }
    Ok(())
}

/// Carry the deletion-vector section through a 1:1 rewrite.
///
/// Without this the rows come back. The output header is built by copying the
/// input's flags, so the `has_deletion_vectors` bit looks preserved — but
/// `FileHeader::sync_from_catalog` re-derives every flag from the sections that
/// were actually written, finds no deletion-vector section, and clears it. The
/// result is a file that has silently forgotten which cells were deleted, with
/// no dangling flag to notice it by.
fn copy_deletion_vectors(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(dv) = reader.read_deletion_vectors()? {
        writer.write_deletion_vectors(&dv)?;
    }
    Ok(())
}

/// Copy all layer CSR shards from reader to writer, preserving per-shard codec.
///
/// `canonicalize` sorts each row's column indices, sums duplicate coordinates
/// and drops explicit zeros before re-encoding — the v3 canonical-CSR contract.
/// See [`copy_auxiliary_sections`] for why it is the caller's choice.
fn copy_layers(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    canonicalize: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let layer_names = reader.layer_names();
    for layer_name in &layer_names {
        let layer_prefix = format!("{layer_name}_shard_");
        let layer_shard_entries: Vec<&scx_format_io::FullCatalogEntry> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&layer_prefix)
            })
            .collect();

        let mut sorted_entries = layer_shard_entries;
        sorted_entries.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));

        for (shard_idx, entry) in sorted_entries.iter().enumerate() {
            let sh = reader.read_shard_header(entry)?;
            let ve = ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(format!("unknown value encoding: {}", sh.value_encoding))?;
            let ci =
                CodecId::from_u8(sh.codec_id).ok_or(format!("unknown codec: {}", sh.codec_id))?;

            let (indptr, indices, mut data) = reader.read_shard_from_entry(entry)?;
            let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
            let mut indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
            let mut indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            // Canonicalizing sums duplicates, which can exceed what the source
            // encoding holds — see `encoding_for_canonicalized`.
            let (ve, ci) = if canonicalize {
                scx_sparse::canonicalize_csr(&mut indptr_u64, &mut indices_u32, &mut data);
                let widened = encoding_for_canonicalized(ve, &data);
                // The codec has to follow the encoding: this writer hands `ci`
                // straight to `encode_shard_adaptive`, which will not downgrade
                // Scx1 for a float encoding on its own.
                (widened, codec_for_canonicalized(ci, widened))
            } else {
                (ve, ci)
            };
            let mut raw_values = Vec::new();
            for &v in &data {
                ve.encode_f32(&mut raw_values, v)?;
            }
            writer.write_layer_csr_shard(
                &indptr_u64,
                &indices_u32,
                &raw_values,
                ci,
                ve,
                row_start,
                layer_name,
                shard_idx as u32,
            )?;
        }
    }
    Ok(())
}

/// Copy uns section from reader to writer.
fn copy_uns(reader: &ScxReader, writer: &mut ScxWriter) -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }
    Ok(())
}

/// Copy predicate index sections from reader to writer.
pub(crate) fn copy_predicate_indices(
    reader: &ScxReader,
    writer: &mut ScxWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(Some(data)) = reader.read_obs_predicate_index_bytes() {
        writer.write_obs_predicate_index(data)?;
    }
    if let Ok(Some(data)) = reader.read_var_predicate_index_bytes() {
        writer.write_var_predicate_index(data)?;
    }
    Ok(())
}

/// Read existing provenance, append a new entry, and write to writer.
pub(crate) fn append_provenance(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    action: &str,
    params_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut prov_entries = if let Ok(prov) = reader.read_provenance() {
        prov.operations
    } else {
        Vec::new()
    };
    prov_entries.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: action.to_string(),
        tool: format!("scx-cli {}", env!("CARGO_PKG_VERSION")),
        params_json: params_json.to_string(),
        input_checksums: vec![],
    });
    writer.write_provenance(prov_entries)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_format_io::shard::{ShardHeader, SHARD_MAGIC};

    fn header(shard_format_version: u8, codec_id: u8) -> ShardHeader {
        ShardHeader {
            magic: SHARD_MAGIC,
            shard_format_version,
            shard_type: 0,
            codec_id,
            value_encoding: ValueEncoding::Uint8 as u8,
            index_dtype: 0,
            reserved_flags: [0u8; 3],
            n_major: 4,
            n_minor: 10,
            nnz: 8,
            global_offset: 0,
            indptr_rel_offset: 0,
            indptr_length: 0,
            indices_rel_offset: 0,
            indices_length: 0,
            values_rel_offset: 0,
            values_length: 0,
            block_index_rel_offset: 0,
            block_index_length: 0,
            checksum: [0u8; 8],
        }
    }

    /// Framing gate: raw-copy requires the source shard's framing to match the
    /// output's framing exactly. A framed (v2) source is eligible only into a
    /// framed (v4) output; an unframed (v1) source only into a ≤v3 output.
    /// Byte-copying across a framing boundary would let a reader mis-decode the
    /// shard (a v2 shard under a pre-framing reader, or an unframed v1 shard
    /// under a v4 header whose readers expect frames).
    #[test]
    fn framed_shard_raw_copy_eligibility_tracks_output_framing() {
        let v1 = header(scx_format_io::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION, 0);
        let v2 = header(
            scx_format_io::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION + 1,
            0,
        );
        // Unframed (v1) shard: eligible ONLY into unframed output.
        assert!(
            raw_copy_csr_eligible(&v1, 0, 10, ResolvedCodec::AUTO, false),
            "unframed v1 shard must be eligible into unframed output"
        );
        assert!(
            !raw_copy_csr_eligible(&v1, 0, 10, ResolvedCodec::AUTO, true),
            "unframed v1 shard must NOT be raw-copy eligible into framed output"
        );
        // Framed (v2) shard: eligible ONLY into framed (v4) output.
        assert!(
            !raw_copy_csr_eligible(&v2, 0, 10, ResolvedCodec::AUTO, false),
            "framed v2 shard must NOT be raw-copy eligible into unframed output"
        );
        assert!(
            raw_copy_csr_eligible(&v2, 0, 10, ResolvedCodec::AUTO, true),
            "framed v2 shard must be raw-copy eligible into framed output"
        );
    }

    /// `copy_auxiliary_sections` is an allowlist, so what it does not name is
    /// dropped — and both its callers rename over the target with no prior
    /// catalog, so the drop cannot be rolled back. The warning is the only
    /// notice a user gets, which makes "does it name the right families?" worth
    /// asserting rather than assuming: a stale [`DROPPED_SECTION_FAMILIES`]
    /// produces a *confidently wrong* warning, which is worse than none.
    ///
    /// This test used to assert `["varm"]` on this exact fixture, which is the
    /// clearest statement of what review §6.3 was: `varm` is user data with no
    /// rebuild path, and the op's own warning listed it as expected collateral.
    /// It is carried now, so the fixture must report **nothing** — and the
    /// still-dropped family gets its own file below, because "the list is
    /// empty" and "the list is right" are different claims.
    #[test]
    fn dropped_families_are_detected_on_a_file_that_has_them() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};
        use arrow::array::{Float32Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("with_varm.scx");
        let (n_obs, n_vars) = (4usize, 3usize);
        let mut w = ScxWriter::new(&path, sample_header(n_obs as u64, n_vars as u64)).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        let varm = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "pc1",
                DataType::Float32,
                false,
            )])),
            vec![Arc::new(Float32Array::from(vec![0.1f32, 0.2, 0.3]))],
        )
        .unwrap();
        w.write_varm("loadings", &varm).unwrap();
        w.write_csr_shard(
            &vec![0u64; n_obs + 1],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        w.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        assert!(
            dropped_section_labels(&reader).is_empty(),
            "varm is carried since Phase 5b, so it must no longer be reported as \
             a loss — got {:?}",
            dropped_section_labels(&reader)
        );

        // The accept side: the one family still on the list must still be
        // named. Without it, emptying `DROPPED_SECTION_FAMILIES` outright would
        // satisfy every other assertion in this test.
        //
        // It takes a multimodal file to build one, because
        // `write_layer_csc_shard_for` refuses `modality_id == 0` — which is the
        // same reason the entry is unreachable through the two ops that consult
        // this list, and why the constant's doc says so.
        let with_layer_csc = dir.path().join("with_layer_csc.scx");
        let mut w =
            ScxWriter::new(&with_layer_csc, sample_header(n_obs as u64, n_vars as u64)).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        let rna = w
            .add_modality(
                "rna",
                scx_format_io::modality::ModalityType::Rna,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        w.write_var_for(rna, &sample_var(n_vars)).unwrap();
        w.set_modality_n_vars(rna, n_vars as u64).unwrap();
        w.write_csr_shard_for(
            rna,
            &vec![0u64; n_obs + 1],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        w.write_layer_csc_shard_for(
            rna,
            "spliced",
            0,
            &vec![0u64; n_vars + 1],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        w.finish().unwrap();
        assert_eq!(
            dropped_section_labels(&ScxReader::open(&with_layer_csc).unwrap()),
            vec!["layer CSC sidecars (rebuild: scx build-csc)"],
            "the one family still on the list must still be named when present"
        );

        // And a file with none of them must warn about nothing — otherwise the
        // warning fires on every ordinary upgrade and stops being read.
        let plain = dir.path().join("plain.scx");
        let mut w = ScxWriter::new(&plain, sample_header(n_obs as u64, n_vars as u64)).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        w.write_csr_shard(
            &vec![0u64; n_obs + 1],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        w.finish().unwrap();
        assert!(dropped_section_labels(&ScxReader::open(&plain).unwrap()).is_empty());
    }

    /// The integer ladder must not stop at `Uint32`. A sum past the decode
    /// ceiling re-encoded under `Uint32` hard-fails `ValueEncoding::encode_f32`
    /// — aborting the rewrite on exactly the non-canonical input this function
    /// exists to repair.
    #[test]
    fn canonicalized_uint32_sum_past_the_decode_ceiling_widens_to_float32() {
        let past = (1u128 << 33) as f32;
        assert_eq!(
            encoding_for_canonicalized(ValueEncoding::Uint32, &[1.0, past]),
            ValueEncoding::Float32
        );
    }

    /// Each rung must hand off to the next, not terminate. A `Uint8` source
    /// whose sums outgrow the whole ladder has to climb all the way to
    /// `Float32`; stopping at `Uint32` lands it on the arm above.
    #[test]
    fn canonicalized_uint8_cascades_all_the_way_to_float32() {
        let past = (1u128 << 33) as f32;
        assert_eq!(
            encoding_for_canonicalized(ValueEncoding::Uint8, &[past]),
            ValueEncoding::Float32
        );
    }

    /// The boundary is 2³² **exclusive**, and that is load-bearing in both
    /// directions.
    ///
    /// No `u32` decodes above 2³² — `u32::MAX as f32` rounds *up* to exactly
    /// 2³² — so a value strictly greater can only have come from summing, which
    /// makes the arm provenance-proof. At 2³² itself the two provenances are
    /// indistinguishable, and keeping `Uint32` is the better half: `as u32`
    /// saturates a decoded `u32::MAX` back to itself exactly, whereas widening
    /// writes 2³², which no longer fits `u32` at all.
    #[test]
    fn canonicalized_uint32_widens_only_strictly_above_the_decode_ceiling() {
        let two_pow_32 = (1u128 << 32) as f32;
        assert_eq!(u32::MAX as f32, two_pow_32, "the alias this arm turns on");

        assert_eq!(
            encoding_for_canonicalized(ValueEncoding::Uint32, &[4_294_967_040.0]),
            ValueEncoding::Uint32
        );
        assert_eq!(
            encoding_for_canonicalized(ValueEncoding::Uint32, &[two_pow_32]),
            ValueEncoding::Uint32,
            "a decoded u32::MAX must survive the rewrite as u32::MAX"
        );
        assert_eq!(
            encoding_for_canonicalized(ValueEncoding::Uint32, &[two_pow_32 * 2.0]),
            ValueEncoding::Float32,
            "a sum no u32 can decode to must widen rather than abort the encoder"
        );
    }

    /// `Float16`'s ceiling is a *magnitude*, but the fold only ever looked at
    /// the signed maximum — which starts at `0.0`, so an all-negative shard
    /// reports `max == 0.0` and never widens. Two duplicate `-40000`s sum to
    /// `-80000`, `f16::from_f32` maps that to `-inf`, and the rewrite reports
    /// success. Layers hold arbitrary floats, so this is reachable data.
    #[test]
    fn canonicalized_float16_widens_on_negative_overflow_too() {
        assert_eq!(
            encoding_for_canonicalized(ValueEncoding::Float16, &[-80_000.0]),
            ValueEncoding::Float32
        );
        assert_eq!(
            encoding_for_canonicalized(ValueEncoding::Float16, &[-65_504.0, 1.0]),
            ValueEncoding::Float16,
            "the negative end of the representable range must not widen"
        );
    }

    /// A widened encoding has to drag the codec with it: `Scx1` encodes
    /// integers only, so pairing it with `Float32` is `FloatWithScx1` — the
    /// widening arm would abort the very rewrite it exists to let through.
    #[test]
    fn canonicalized_codec_drops_scx1_when_the_encoding_goes_float() {
        assert_eq!(
            codec_for_canonicalized(CodecId::Scx1, ValueEncoding::Float32),
            CodecId::Zstd
        );
        assert_eq!(
            codec_for_canonicalized(CodecId::Scx1, ValueEncoding::Uint32),
            CodecId::Scx1,
            "an encoding Scx1 can hold must keep it"
        );
        assert_eq!(
            codec_for_canonicalized(CodecId::Zstd, ValueEncoding::Float32),
            CodecId::Zstd
        );
    }
}
