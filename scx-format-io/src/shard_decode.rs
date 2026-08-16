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

use scx_codec::{CodecId, EncodedShardRef, ShardValuesNative, ValueEncoding};

use crate::catalog::FullCatalogEntry;
use crate::error::{Result, ScxError};
use crate::section::SectionType;
use crate::shard::{
    clamped_reserve, resolve_block_index, ShardHeader, DEFAULT_WRITE_SHARD_FORMAT_VERSION,
};
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

    // Reconcile the shard's own header against the catalog before trusting
    // either. This is the seam every full-entry read funnels through — the local
    // mmap reader, the cloud range reader, the ML loader's `io_stage`, and
    // `scx-engine`'s `read_row_range` — so a check here covers all of them,
    // where the `BackedCsrReader`-only check covered just the backed cache.
    check_header_against_catalog(&sh, entry)?;

    // Delegate the codec-resolve + framed/unframed decode to the shared region
    // decoder (also used by the scx-gpu host-bounce, so both honor framing).
    let (indptr, indices, data) = decode_shard_regions_scipy(
        &sh,
        indptr_bytes,
        indices_bytes,
        values_bytes,
        block_index_bytes,
    )?;

    // The decoded major-axis extent, against the catalog's. `decode_shard_regions_*`
    // cannot do this — it never sees the entry.
    if let Some(expected) = catalog_major_extent(&sh, entry)? {
        let got = indptr.len().saturating_sub(1) as u64;
        if got != expected {
            return Err(ScxError::InvalidCatalog(format!(
                "shard '{}': catalog says {expected} rows, decoded {got} \
                 (truncated or corrupt file)",
                entry.name
            )));
        }
    }

    Ok((indptr, indices, data))
}

/// Whether the **catalog** says this section is column-major.
///
/// Named explicitly rather than derived from the shard header — the distinction
/// from `ShardHeader::is_csc` is the whole point at the call sites, since the
/// header is unauthenticated bytes. The answer itself comes from
/// [`crate::shard::is_column_major`], so this cannot drift from what the writer
/// stamped or from what either catalog representation reads.
pub(crate) fn catalog_says_column_major(section_type: SectionType) -> bool {
    crate::shard::is_column_major(section_type)
}

/// Reconcile a shard's declared minor extent against a width the caller knows
/// from an authenticated source.
///
/// O(1) — two integers. The expensive part (bounding every decoded index)
/// already happened inside the codec using the header's `n_minor`, so once that
/// number is confirmed equal to the real width, the bound it enforced was the
/// right one. That is what lets the backed reader, which deliberately carries no
/// catalog stats on its transient entries, close the same hole for the price of
/// a comparison.
///
/// `0` means "undeclared" — legacy multimodal shards stamp the file-level
/// `n_vars`, which is `0` there — and is left alone, as everywhere else.
pub(crate) fn reconcile_declared_minor(
    declared: u32,
    authenticated: u64,
    what: &str,
) -> Result<()> {
    if declared == 0 || authenticated == 0 {
        return Ok(());
    }
    if declared as u64 != authenticated {
        return Err(ScxError::InvalidCatalog(format!(
            "{what}: header declares n_minor {declared} but the file says \
             {authenticated} (corrupt file; the shard payload is not covered by \
             the catalog checksum, so it is not the authority here)"
        )));
    }
    Ok(())
}

/// The major-axis extent the **catalog** assigns this shard, or `None` when the
/// entry carries no stats.
///
/// `ShardEntryLite::into_transient_full_entry` deliberately sets `stats: None` —
/// the backed reader keeps row ranges in `BackedCsrIndex` instead — so the
/// backed path returns `None` here and is covered by
/// `BackedCsrReader::check_decoded_shard_rows`.
fn catalog_major_extent(_sh: &ShardHeader, entry: &FullCatalogEntry) -> Result<Option<u64>> {
    let Some(stats) = entry.stats.as_ref() else {
        return Ok(None);
    };
    // CSR-class shards tile the row axis (`row_start..row_end`); a CSC sidecar
    // tiles the column axis. `compute_shard_stats` writes the *other* pair as
    // `0..n_minor`, so reading the wrong one would compare an extent against a
    // width.
    let (lo, hi) = if catalog_says_column_major(entry.section_type) {
        (stats.col_start, stats.col_end)
    } else {
        (stats.row_start, stats.row_end)
    };
    hi.checked_sub(lo).map(Some).ok_or_else(|| {
        ScxError::InvalidCatalog(format!(
            "shard '{}': catalog major range {lo}..{hi} is inverted",
            entry.name
        ))
    })
}

/// Reject a shard whose header disagrees with the catalog about the minor axis.
///
/// The two carry the same number by construction — `compute_shard_stats` derives
/// `col_end` (CSR) / `row_end` (CSC) from the very `n_minor` stamped into the
/// header — but they live in different places, and only one of them is
/// authenticated: the catalog's BLAKE3 covers catalog bytes, the shard payload
/// is not covered at all.
///
/// That asymmetry is the point. Bounding decoded indices by the payload's own
/// `n_minor` lets a corrupt shard raise its declared width and smuggle an index
/// past the bound — one that is still out of range for the matrix the catalog
/// describes, and which then reaches `scipy.sparse.csr_matrix` as a structurally
/// invalid matrix whose `.toarray()` misplaces the value. Requiring the two to
/// agree removes that move: the payload cannot widen itself.
///
/// A v1 catalog writes `0` for the v2-only `col_start`/`col_end` pair, so there
/// is nothing to reconcile against and the header stands alone — strictly the
/// pre-existing position, not a regression.
fn check_header_against_catalog(sh: &ShardHeader, entry: &FullCatalogEntry) -> Result<()> {
    let Some(stats) = entry.stats.as_ref() else {
        return Ok(());
    };
    // Dispatch on the **catalog's** section type, never on `ShardHeader::is_csc`.
    // `is_csc` is true when *either* the catalog says CSC or the payload's own
    // `shard_type` byte does, and `validate_csc_strict` only rejects the one
    // direction (`CscShard` + `shard_type != 1`). A `CsrShard` entry carrying a
    // self-consistent transposed payload (`shard_type = 1`, major and minor
    // swapped) would therefore choose which catalog axes validated it — and both
    // comparisons would pass while the catalog says the section is row-major.
    // Letting unauthenticated bytes pick their own referee defeats the point of
    // reconciling against the catalog at all.
    let authenticated = if catalog_says_column_major(entry.section_type) {
        stats.row_end
    } else {
        stats.col_end
    };
    // Having fixed the axis from the catalog, require the payload to agree about
    // the layout too, in *both* directions.
    let payload_says_csc = sh.shard_type == 1;
    if payload_says_csc != catalog_says_column_major(entry.section_type) {
        return Err(ScxError::InvalidCatalog(format!(
            "shard '{}': catalog section type {:?} disagrees with the payload's \
             shard_type byte {} (corrupt file)",
            entry.name, entry.section_type, sh.shard_type
        )));
    }
    if authenticated == 0 {
        return Ok(()); // v1 catalog: no minor extent recorded.
    }
    if authenticated != sh.n_minor as u64 {
        return Err(ScxError::InvalidCatalog(format!(
            "shard '{}': header declares n_minor {} but the catalog says {authenticated} \
             (corrupt file; the shard payload is not covered by the catalog checksum, \
             so the catalog is the authority here)",
            entry.name, sh.n_minor
        )));
    }
    Ok(())
}

/// Decode a shard's already-extracted byte regions into scipy triples
/// (`Vec<i64>` indptr, `Vec<i32>` indices, `Vec<f32>` data), transparently
/// handling both **framed** (v2, row-group) and **legacy** (v1, whole-stream)
/// layouts.
///
/// The caller has parsed the [`ShardHeader`] and sliced the four regions
/// (indptr / indices / values / block_index) from the shard section. This is the
/// codec seam shared by [`decode_shard_bytes`] (mmap/cloud CPU path) and the
/// scx-gpu host-bounce, so a framed shard decodes identically on either path.
///
/// For a framed shard the indptr sub-stream is a concatenation of per-group
/// local-rebased indptrs, so the whole-stream decoder cannot read it — the block
/// index is iterated and the global CSR reassembled (byte-identical to an
/// unframed decode of the same data).
pub fn decode_shard_regions_scipy(
    sh: &ShardHeader,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    block_index_bytes: &[u8],
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    // Resolve codec and encoding from shard header (NOT file header)
    let codec_id = CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?;
    let value_encoding = ValueEncoding::from_u8(sh.value_encoding)
        .ok_or(ScxError::UnknownValueEncoding(sh.value_encoding))?;
    let index_dtype_u16 = sh.index_dtype == 0;

    // Row-group-framed (v2) shards reassemble from the block index; legacy (v1)
    // shards take the direct whole-shard path. Both fall through to the same
    // bound check below rather than returning early, so neither layout can be
    // the one that skips it.
    let (indptr, indices, data) = if sh.shard_format_version > DEFAULT_WRITE_SHARD_FORMAT_VERSION {
        decode_framed_shard_scipy(
            sh,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            block_index_bytes,
            codec_id,
            value_encoding,
            index_dtype_u16,
        )?
    } else {
        let encoded = EncodedShardRef {
            indptr_bytes,
            indices_bytes,
            values_bytes,
        };
        scx_codec::decode_shard_scipy(
            &encoded,
            codec_id,
            value_encoding,
            sh.n_major as usize,
            sh.nnz as usize,
            index_dtype_u16,
            scx_codec::clamp_index_bound(sh.n_minor),
        )?
    };

    // No standalone pass on this path: the bound rides on the scan `scx-codec`
    // already performs to keep a `> i32::MAX` value from reinterpreting to a
    // negative `i32`. Only the comparand changed, so the check is free here.
    // The native twin below has no such existing scan and does pay for one.
    Ok((indptr, indices, data))
}

/// Reject any decoded minor-axis index outside `[0, n_minor)`, for the **native**
/// (`u32`) decode path.
///
/// # Why the reader validates at all
///
/// `ScxCsr::new_unchecked` states as invariant 5 that every index lies in
/// `[0, shape.1)`, and roughly twenty sites across this crate and `scx-sparse`
/// rely on it — `dense[base + col]`, `result[c] += v`, `col_sums[col]`, and so
/// on — indexing a buffer sized by `n_minor` without a check. Nothing enforced
/// the invariant. The catalog's BLAKE3 covers catalog bytes, not shard payloads
/// (see [`crate::reader::ScxReader::read_shard_from_entry`]), so a decoded index
/// is unauthenticated in exactly the way `clamped_reserve` already documents for
/// declared lengths.
///
/// Enforcing it at the decode, rather than at each consumer, is what lets those
/// sites keep indexing unchecked — and keeps the two streaming statistics
/// kernels (`backed.rs` and `prefetch.rs`), which are each other's oracle, from
/// diverging on malformed input because only one of them grew a guard.
///
/// # Why only the native path calls this
///
/// The scipy path gets the same check for free: `scx-codec` already walks every
/// index there, to keep a `> i32::MAX` value from reinterpreting to a negative
/// `i32`, so passing `clamp_index_bound(n_minor)` into `decode_shard_scipy`
/// changes only that scan's comparand. The native path has no such existing pass
/// and pays for this one.
///
/// The difference is worth the asymmetry: measured as a standalone pass on the
/// scipy path — which is what the ML loader, the GPU host-bounce and every
/// mmap/cloud read use — it cost **+5.1–6.4%** of per-shard decode time
/// (`tabula_sapiens_100k`, 194.9M nnz, three runs). The pass runs at 10.0 GB/s,
/// i.e. memory-bandwidth-bound, so it could not be made cheaper in place; the
/// only way to make it free was to stop making it a separate pass.
///
/// # Cost, on the path that still pays
///
/// One `max` reduction over data still hot from the decode, then a single
/// comparison. `max` rather than a short-circuiting `find` because the good case
/// is every case: `find` would run a scalar loop over the whole slice anyway,
/// while `max` auto-vectorizes. Locating the offending position for the error
/// message happens only on the cold path.
///
/// # `n_minor == 0` is "not declared", not "zero columns"
///
/// See [`scx_codec::clamp_index_bound`], which encodes the same rule for the
/// scipy path. Older writers stamped the **file-level** `n_vars` into every
/// shard header, and on a multimodal file that field is `0` — the real column
/// count lives in the per-modality metadata. Both multimodal conformance
/// fixtures (`v2_multimodal_citeseq.scx`, `v2_multimodal_partial_csc.scx`) carry
/// `n_minor = 0` on shards with hundreds of nonzeros, and reading them is
/// correct behaviour, not corruption.
///
/// Skipping is safe rather than merely convenient: a genuinely zero-column
/// matrix has no nonzeros to check. Current writers resolve the per-modality
/// `n_vars` (`ScxWriter::write_shard_inner`), so files written today do get the
/// bound; only legacy multimodal shards fall through unvalidated, which is the
/// right trade against bricking them.
///
/// # Not covered
///
/// `scx-gpu`'s in-VRAM decode paths never pass through this seam — their indices
/// stay device-resident and remain unvalidated.
#[inline]
fn check_minor_indices<T: MinorIndexBits>(indices: &[T], n_minor: u32) -> Result<()> {
    if n_minor == 0 {
        return Ok(());
    }
    let Some(max) = indices.iter().map(|&c| c.bits()).max() else {
        return Ok(());
    };
    if max < n_minor {
        return Ok(());
    }
    // Cold: re-walk only to name the *first* offender. Reporting `max` with the
    // first offender's position would pair a value and a position that need not
    // belong together.
    let (position, index) = indices
        .iter()
        .map(|&c| c.bits())
        .enumerate()
        .find(|&(_, bits)| bits >= n_minor)
        .unwrap_or((0, max));
    Err(ScxError::ShardIndexOutOfRange {
        index,
        position,
        n_minor,
    })
}

/// The `u32` bit pattern of a decoded minor-axis index, for the single unsigned
/// compare in [`check_minor_indices`]. Implemented for both decode domains:
/// `i32` (scipy) and `u32` (native).
trait MinorIndexBits: Copy {
    fn bits(self) -> u32;
}

impl MinorIndexBits for i32 {
    #[inline]
    fn bits(self) -> u32 {
        self as u32
    }
}

impl MinorIndexBits for u32 {
    #[inline]
    fn bits(self) -> u32 {
        self
    }
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

    // `n_major` / `nnz` come straight off the untrusted header, so reserve
    // through `clamped_reserve` rather than trusting them: `Vec::with_capacity`
    // aborts the process on allocation failure, and a ~100-byte file can declare
    // `nnz = u32::MAX`. Honest shards are unaffected (the declared count wins);
    // a hostile one under-reserves and then fails in `decode_row_group` below,
    // which length-checks every frame against its own bytes.
    let mut indptr = Vec::with_capacity(clamped_reserve(n_major + 1, indptr_bytes.len(), 8));
    indptr.push(0i64);
    let mut indices = Vec::with_capacity(clamped_reserve(nnz, indices_bytes.len(), 4));
    let mut data = Vec::with_capacity(clamped_reserve(nnz, values_bytes.len(), 4));
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
        let (g_indptr, g_indices, g_data) = scx_codec::decoded_shard_to_scipy(
            decoded,
            value_encoding,
            scx_codec::clamp_index_bound(sh.n_minor),
        )?;
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

/// Native-value twin of [`decode_shard_bytes`]: decodes a single CSR shard to
/// `(Vec<i64>` indptr, `Vec<u32>` indices, [`ShardValuesNative`] values`)`,
/// keeping integer values as `u32` (never rounded through `f32`) so the
/// in-assembly narrow reader can cast directly to the target dtype. Handles both
/// legacy (v1) and framed (v2) layouts.
pub fn decode_shard_bytes_native(
    section: &[u8],
    entry: &FullCatalogEntry,
    catalog_version: u16,
    verify_checksum: bool,
) -> Result<(Vec<i64>, Vec<u32>, ShardValuesNative)> {
    let vs = ValidatedSection::new(section);
    let sh = ShardHeader::read_from(&mut Cursor::new(vs.header()?))?;

    if catalog_version >= 2 {
        sh.validate_csc_strict(entry.section_type)?;
    }

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

    // Same reconciliation as the scipy twin — this is a full-entry seam too, and
    // it is the one the typed / dense reads take. Guarding only the scipy entry
    // point left `to_anndata(container="dense")` accepting a shard whose header
    // had widened its own declared width past the catalog's.
    check_header_against_catalog(&sh, entry)?;

    let (indptr, indices, values) = decode_shard_regions_native(
        &sh,
        indptr_bytes,
        indices_bytes,
        values_bytes,
        block_index_bytes,
    )?;

    if let Some(expected) = catalog_major_extent(&sh, entry)? {
        let got = indptr.len().saturating_sub(1) as u64;
        if got != expected {
            return Err(ScxError::InvalidCatalog(format!(
                "shard '{}': catalog says {expected} rows, decoded {got} \
                 (truncated or corrupt file)",
                entry.name
            )));
        }
    }

    Ok((indptr, indices, values))
}

/// Native-value twin of [`decode_shard_regions_scipy`]. Same framed/legacy split,
/// but delegates to [`scx_codec::decode_shard_native`] /
/// [`scx_codec::decoded_shard_to_native`] so integer values stay `u32`.
pub fn decode_shard_regions_native(
    sh: &ShardHeader,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    block_index_bytes: &[u8],
) -> Result<(Vec<i64>, Vec<u32>, ShardValuesNative)> {
    let codec_id = CodecId::from_u8(sh.codec_id).ok_or(ScxError::UnknownCodec(sh.codec_id))?;
    let value_encoding = ValueEncoding::from_u8(sh.value_encoding)
        .ok_or(ScxError::UnknownValueEncoding(sh.value_encoding))?;
    let index_dtype_u16 = sh.index_dtype == 0;

    // Both layouts fall through to the same bound check — see the scipy twin.
    let (indptr, indices, values) = if sh.shard_format_version > DEFAULT_WRITE_SHARD_FORMAT_VERSION
    {
        decode_framed_shard_native(
            sh,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            block_index_bytes,
            codec_id,
            value_encoding,
            index_dtype_u16,
        )?
    } else {
        let encoded = EncodedShardRef {
            indptr_bytes,
            indices_bytes,
            values_bytes,
        };
        scx_codec::decode_shard_native(
            &encoded,
            codec_id,
            value_encoding,
            sh.n_major as usize,
            sh.nnz as usize,
            index_dtype_u16,
        )?
    };

    check_minor_indices(&indices, sh.n_minor)?;
    Ok((indptr, indices, values))
}

/// Native-value twin of [`decode_framed_shard_scipy`]. Reassembles a framed (v2)
/// shard from its block index, keeping integer values as `u32`. The value
/// accumulator variant is fixed by `value_encoding.is_integer()` (a shard is
/// uniformly integer- or float-encoded).
#[allow(clippy::too_many_arguments)]
fn decode_framed_shard_native(
    sh: &ShardHeader,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    block_index_bytes: &[u8],
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<(Vec<i64>, Vec<u32>, ShardValuesNative)> {
    let spans = resolve_block_index(sh, block_index_bytes)?;
    let n_major = sh.n_major as usize;
    let nnz = sh.nnz as usize;

    // Same untrusted-header clamp as `decode_framed_shard_scipy` — see there.
    let mut indptr = Vec::with_capacity(clamped_reserve(n_major + 1, indptr_bytes.len(), 8));
    indptr.push(0i64);
    let mut indices = Vec::with_capacity(clamped_reserve(nnz, indices_bytes.len(), 4));
    let values_reserve = clamped_reserve(nnz, values_bytes.len(), 4);
    let mut values_u32: Vec<u32> = if value_encoding.is_integer() {
        Vec::with_capacity(values_reserve)
    } else {
        Vec::new()
    };
    let mut values_f32: Vec<f32> = if value_encoding.is_integer() {
        Vec::new()
    } else {
        Vec::with_capacity(values_reserve)
    };
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
        let (g_indptr, g_indices, g_values) =
            scx_codec::decoded_shard_to_native(decoded, value_encoding)?;
        for &local in &g_indptr[1..] {
            indptr.push(running + local);
        }
        if let Some(&last) = g_indptr.last() {
            running += last;
        }
        indices.extend_from_slice(&g_indices);
        match g_values {
            ShardValuesNative::U32(v) => values_u32.extend_from_slice(&v),
            ShardValuesNative::F32(v) => values_f32.extend_from_slice(&v),
        }
    }

    let values = if value_encoding.is_integer() {
        ShardValuesNative::U32(values_u32)
    } else {
        ShardValuesNative::F32(values_f32)
    };
    Ok((indptr, indices, values))
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
        // Indptr-only: decode just each group's local indptr frame (F-b); the
        // indices/values frames are never touched.
        let block_index_bytes = vs.subslice(sh.block_index_rel_offset, sh.block_index_length)?;
        let spans = resolve_block_index(&sh, block_index_bytes)?;
        // Untrusted `n_major`: clamp the reservation (see
        // `decode_framed_shard_scipy`). A block index of only ~45 KB can declare
        // 134M rows, which is a 1 GB `Vec<i64>` before any frame is decoded.
        let mut indptr = Vec::with_capacity(clamped_reserve(
            sh.n_major as usize + 1,
            indptr_bytes.len(),
            8,
        ));
        indptr.push(0i64);
        let mut running: i64 = 0;
        for span in &spans {
            let g_indptr = scx_codec::decode_row_group_indptr_only(codec_id, span, indptr_bytes)?;
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

    /// The shared codec seam must reject a shard whose decoded arrays do not
    /// agree with each other.
    ///
    /// `decode_shard_regions_scipy` is the seam every non-mmap consumer uses —
    /// including the scx-gpu host bounce — and it validated only the *minor
    /// axis bound* on the decoded indices, never a length. The whole-file read
    /// (`ScxReader::assemble_shards`) does check lengths, which is why this
    /// went unnoticed: the defect was only reachable through the callers that
    /// skip it.
    ///
    /// The shard here is self-consistent everywhere a single sub-stream
    /// decoder can see — FOR-BP holds 3 indices, Rice holds 3 values, the
    /// header declares `nnz = 3` — but its indptr says the one row ends at 2.
    /// A structurally invalid CSR: `ScxCsr::new_unchecked` only `debug_assert`s
    /// the difference, so in release `csr_to_csc` would walk off the end of
    /// `indices`.
    #[test]
    fn decode_shard_regions_rejects_indptr_disagreeing_with_nnz() {
        use scx_codec::delta_golomb::delta_golomb_encode;
        use scx_codec::forbp::forbp_encode;
        use scx_codec::rice::{rice_encode, B_VAL};

        let indptr_bytes = delta_golomb_encode(&[0, 2]).unwrap();
        let indices_bytes = forbp_encode(&[1, 2, 3], &[3], false).unwrap();
        let values_bytes = rice_encode(&[1, 1, 1], B_VAL).unwrap();

        let header = ShardHeader {
            magic: SHARD_MAGIC,
            shard_format_version: 1, // legacy whole-shard, not framed
            shard_type: 0,
            codec_id: 1,       // Scx1
            value_encoding: 2, // Uint32
            index_dtype: 1,    // u32
            reserved_flags: [0u8; 3],
            n_major: 1,
            n_minor: 100,
            nnz: 3,
            global_offset: 0,
            indptr_rel_offset: SHARD_HEADER_SIZE as u32,
            indptr_length: indptr_bytes.len() as u32,
            indices_rel_offset: SHARD_HEADER_SIZE as u32,
            indices_length: indices_bytes.len() as u32,
            values_rel_offset: SHARD_HEADER_SIZE as u32,
            values_length: values_bytes.len() as u32,
            block_index_rel_offset: SHARD_HEADER_SIZE as u32,
            block_index_length: 0,
            checksum: [0u8; 8],
        };

        let err =
            decode_shard_regions_scipy(&header, &indptr_bytes, &indices_bytes, &values_bytes, &[])
                .unwrap_err();
        assert!(
            err.to_string().contains("indptr ends at"),
            "expected a shape MalformedInput, got {err:?}"
        );

        // The native twin decodes through its own Scx1 short-circuit, so it
        // needs its own assertion rather than riding on the scipy one.
        match decode_shard_regions_native(
            &header,
            &indptr_bytes,
            &indices_bytes,
            &values_bytes,
            &[],
        ) {
            Err(other) => assert!(
                other.to_string().contains("indptr ends at"),
                "expected a shape MalformedInput, got {other:?}"
            ),
            Ok((indptr, indices, _)) => panic!(
                "accepted a malformed shard: indptr={indptr:?}, indices.len()={}",
                indices.len()
            ),
        }
    }

    /// A framed shard with zero rows — writable before the encoder learned to
    /// refuse it, and unreadable on *every* path, because framing emits an empty
    /// block index and `resolve_block_index` rejected it outright.
    ///
    /// The bytes are assembled by hand rather than by the writer, because the
    /// writer now refuses to produce them: the point of the test is that a file
    /// already on disk is not bricked by the new guard. Both value domains are
    /// covered — a `native` read has its own framed reassembly path.
    #[test]
    fn a_zero_row_framed_shard_written_before_the_guard_still_reads() {
        use crate::shard::BlockIndex;

        let header = ShardHeader {
            magic: SHARD_MAGIC,
            shard_format_version: 2, // framed
            shard_type: 0,
            codec_id: 0, // None
            value_encoding: 2,
            index_dtype: 1,
            reserved_flags: [0u8; 3],
            n_major: 0,
            n_minor: 16,
            nnz: 0,
            global_offset: 0,
            indptr_rel_offset: SHARD_HEADER_SIZE as u32,
            indptr_length: 0,
            indices_rel_offset: SHARD_HEADER_SIZE as u32,
            indices_length: 0,
            values_rel_offset: SHARD_HEADER_SIZE as u32,
            values_length: 0,
            block_index_rel_offset: SHARD_HEADER_SIZE as u32,
            block_index_length: 4, // just the u32 count == 0
            checksum: [0u8; 8],
        };
        let mut block_index_bytes = Vec::new();
        BlockIndex { entries: vec![] }
            .write_to(&mut block_index_bytes)
            .unwrap();

        let (indptr, indices, data) =
            decode_shard_regions_scipy(&header, &[], &[], &[], &block_index_bytes)
                .expect("a zero-row framed shard must decode, not error");
        assert_eq!(
            indptr,
            vec![0i64],
            "an empty CSR still carries indptr = [0]"
        );
        assert!(indices.is_empty());
        assert!(data.is_empty());

        let (indptr, indices, _vals) =
            decode_shard_regions_native(&header, &[], &[], &[], &block_index_bytes)
                .expect("the native framed path must agree");
        assert_eq!(indptr, vec![0i64]);
        assert!(indices.is_empty());
    }

    // -----------------------------------------------------------------------
    // Minor-axis index bounds
    // -----------------------------------------------------------------------

    /// Encode a shard, then hand the decoder a header whose `n_minor` is
    /// `claimed_n_minor` — i.e. a shard carrying indices past the end of its own
    /// declared column axis.
    ///
    /// Patching the header rather than the payload is what makes this a *decode*
    /// test: the indices themselves encode and decode perfectly, so no codec
    /// length check fires. Only a comparison against `n_minor` can catch it, and
    /// nothing performed one.
    /// `(scipy decode, native decode reduced to its error)` — the native values
    /// are irrelevant here, only whether the seam accepted the shard.
    type SeamOutcome = (Result<(Vec<i64>, Vec<i32>, Vec<f32>)>, Result<()>);

    fn decode_with_claimed_n_minor(
        indices: &[u32],
        index_dtype: u8,
        encoded_n_cols: u32,
        claimed_n_minor: u32,
        framing: Option<crate::encoder::FramingConfig>,
    ) -> SeamOutcome {
        use crate::encoder::encode_one_shard;
        use crate::modality::ModalityType;

        let indptr: Vec<u64> = vec![0, indices.len() as u64];
        let values: Vec<f32> = (0..indices.len()).map(|i| (i + 1) as f32).collect();
        let s = encode_one_shard(
            &indptr,
            indices,
            &values,
            Some(CodecId::None),
            index_dtype,
            encoded_n_cols,
            0,
            SectionType::CsrShard,
            ModalityType::Rna,
            "X_shard_0".to_string(),
            framing,
        )
        .expect("encode_one_shard");
        let mut sh = ShardHeader::read_from(&mut Cursor::new(&s.header_buf[..])).unwrap();
        sh.n_minor = claimed_n_minor;

        let scipy = decode_shard_regions_scipy(
            &sh,
            &s.encoded.indptr_bytes,
            &s.encoded.indices_bytes,
            &s.encoded.values_bytes,
            &s.block_index_bytes,
        );
        let native = decode_shard_regions_native(
            &sh,
            &s.encoded.indptr_bytes,
            &s.encoded.indices_bytes,
            &s.encoded.values_bytes,
            &s.block_index_bytes,
        )
        .map(|_| ());
        (scipy, native)
    }

    /// A decoded column index at or past `n_minor` must be rejected at the
    /// decode seam.
    ///
    /// ~20 sites across this crate and `scx-sparse` index a buffer sized by
    /// `n_minor` with a decoded column and none of them check it, because
    /// `ScxCsr::new_unchecked`'s invariant 5 says every index is in range. That
    /// invariant was never enforced anywhere: the catalog's BLAKE3 covers
    /// catalog bytes, not shard payloads. Validating here is what makes it true.
    ///
    /// Both value domains and both layouts, because they are four separate
    /// reassembly paths.
    #[test]
    fn out_of_range_column_index_is_rejected() {
        for framing in [
            None,
            Some(crate::encoder::FramingConfig {
                row_group_rows: 2,
                ..Default::default()
            }),
        ] {
            let (scipy, native) = decode_with_claimed_n_minor(&[0, 3, 9], 0, 16, 8, framing);
            for (what, r) in [("scipy", scipy.map(|_| ())), ("native", native)] {
                let err = r.expect_err("index 9 with n_minor 8 must be rejected ({what})");
                assert!(
                    matches!(err, ScxError::ShardIndexOutOfRange { index: 9, .. }),
                    "{what}: expected ShardIndexOutOfRange{{index:9}}, got {err:?}"
                );
            }
        }
    }

    /// An index at or above 2^31 must be rejected identically on both value
    /// domains, because the *file* is equally corrupt either way.
    ///
    /// Indices are `u32` on disk and `i32` in scipy's CSR, so such a value would
    /// reinterpret to a *negative* `i32`, and in `scatter_typed_csr_to_dense` (or
    /// `ScxCsr::to_dense`) `dense[base + col as usize]` would make it ~2^64,
    /// wrap the add in release, and write to some other cell — a plausible,
    /// wrong answer with no panic and no error.
    ///
    /// The scipy path has always rejected it, in `scx-codec`'s
    /// `u32_vec_to_i32`. What it did *not* do was classify it the same way: the
    /// sign branch produced `MalformedInput` → `ScxError::Codec` → `Other`,
    /// while the native seam produced `ShardIndexOutOfRange` → `CorruptFile`.
    /// One corrupt file, two Python exception types, chosen by whether the
    /// caller asked for `f32` or a narrowed dtype — the same defect the typed
    /// `IndexOutOfRange` variant exists to remove. An earlier version of this
    /// test asserted the *divergence*, which is how it survived.
    #[test]
    fn an_index_above_i32_max_is_rejected_the_same_way_on_both_paths() {
        // index_dtype = 1 (u32) so the value survives the round-trip.
        let (scipy, native) = decode_with_claimed_n_minor(&[0, 0x8000_0000], 1, 100, 100, None);
        for (what, r) in [("scipy", scipy.map(|_| ())), ("native", native)] {
            let err = r.expect_err("2^31 must not reach a CSR");
            assert!(
                matches!(
                    err,
                    ScxError::ShardIndexOutOfRange {
                        index: 0x8000_0000,
                        ..
                    }
                ),
                "{what}: expected ShardIndexOutOfRange, got {err:?}"
            );
        }
    }

    /// The sign guard itself still exists, and still has its own message, for
    /// the case where there is no column axis to be out of.
    ///
    /// `NO_INDEX_BOUND` is what a caller passes when it is decoding something
    /// that is not addressing a column axis. There the only hazard is the `i32`
    /// reinterpretation, and the legacy diagnostic is the accurate one — calling
    /// it "out of range for n_minor 2147483648" would invent a bound the caller
    /// never declared.
    #[test]
    fn without_a_declared_bound_the_sign_guard_keeps_its_own_message() {
        use scx_codec::{CodecError, NO_INDEX_BOUND};
        let err = scx_codec::decoded_shard_to_scipy(
            (vec![0, 2], vec![0, 0x8000_0000], vec![1, 2]),
            ValueEncoding::Uint8,
            NO_INDEX_BOUND,
        )
        .expect_err("2^31 must not reinterpret to a negative i32");
        assert!(
            matches!(err, CodecError::MalformedInput(ref m) if m.contains("exceeds i32::MAX")),
            "expected the sign-only diagnostic, got {err:?}"
        );
    }

    /// The native (`u32`) decode has no such conversion — it hands the value
    /// back as-is — so for it, 2^31 is just another out-of-range index and the
    /// seam check is what catches it.
    #[test]
    fn an_index_above_i32_max_is_rejected_on_the_native_path() {
        let (_scipy, native) = decode_with_claimed_n_minor(&[0, 0x8000_0000], 1, 100, 100, None);
        let err = native.expect_err("2^31 exceeds n_minor 100");
        assert!(
            matches!(
                err,
                ScxError::ShardIndexOutOfRange {
                    index: 0x8000_0000,
                    ..
                }
            ),
            "expected ShardIndexOutOfRange, got {err:?}"
        );
    }

    /// `n_minor == 0` means the shard header does not declare a column bound —
    /// it is not a claim that the matrix has zero columns.
    ///
    /// Older writers stamped the file-level `n_vars` into every shard header,
    /// and that field is `0` on a multimodal file (the real count is
    /// per-modality). Both multimodal conformance fixtures carry `n_minor = 0`
    /// on shards with hundreds of nonzeros; treating `0` as a bound rejected
    /// them outright, which is how this was found.
    ///
    /// Pinned as a test because the natural "tightening" — dropping the
    /// exemption so every shard is validated — silently breaks reading every
    /// multimodal file written before the per-modality `n_minor` fix.
    #[test]
    fn n_minor_zero_means_undeclared_and_does_not_reject() {
        let (scipy, native) = decode_with_claimed_n_minor(&[0, 3, 9], 0, 16, 0, None);
        let (_, indices, _) = scipy.expect("a legacy multimodal shard must still decode");
        assert_eq!(indices, vec![0i32, 3, 9]);
        native.expect("native path agrees");
    }

    /// Control: the largest legal index (`n_minor - 1`) still decodes. Without
    /// this an off-by-one that rejected every full-width matrix would pass the
    /// two tests above.
    #[test]
    fn the_largest_legal_column_index_still_decodes() {
        let (scipy, native) = decode_with_claimed_n_minor(&[0, 7], 0, 16, 8, None);
        let (_, indices, _) = scipy.expect("index 7 is legal for n_minor 8");
        assert_eq!(indices, vec![0i32, 7]);
        native.expect("native path agrees");
    }

    /// `decode_shard_regions_scipy` on a **framed (v2)** shard reassembles the
    /// same global CSR as decoding the identical matrix unframed (v1). Covers the
    /// helper shared with the scx-gpu host-bounce.
    #[test]
    fn decode_shard_regions_framed_matches_unframed() {
        use crate::encoder::{encode_one_shard, FramingConfig};
        use crate::modality::ModalityType;

        // Canonical CSR: strictly increasing indices per row, integer values.
        let indptr = [0u64, 2, 2, 5, 7];
        let indices = [0u32, 3, 1, 4, 9, 2, 8];
        let values: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 9.0, 3.0, 7.0];
        let n_cols: u32 = 16;

        let assemble = |framing: Option<FramingConfig>| {
            let s = encode_one_shard(
                &indptr,
                &indices,
                &values,
                Some(CodecId::ShufDeltaZstd),
                0,
                n_cols,
                0,
                SectionType::CsrShard,
                ModalityType::Rna,
                "X_shard_0".to_string(),
                framing,
            )
            .expect("encode_one_shard");
            let sh = ShardHeader::read_from(&mut Cursor::new(&s.header_buf[..])).unwrap();
            let regions = decode_shard_regions_scipy(
                &sh,
                &s.encoded.indptr_bytes,
                &s.encoded.indices_bytes,
                &s.encoded.values_bytes,
                &s.block_index_bytes,
            )
            .expect("decode_shard_regions_scipy");
            (sh.shard_format_version, regions)
        };

        let (v_unframed, unframed) = assemble(None);
        let (v_framed, framed) = assemble(Some(FramingConfig {
            row_group_rows: 2,
            target_nnz: None,
            trial: false,
            decode_target: None,
        }));

        assert_eq!(v_unframed, 1, "control must be unframed (v1)");
        assert_eq!(v_framed, 2, "framing must produce v2");
        assert_eq!(framed.0, unframed.0, "indptr differs framed vs unframed");
        assert_eq!(framed.1, unframed.1, "indices differ framed vs unframed");
        assert_eq!(framed.2, unframed.2, "data differs framed vs unframed");
        // And equals the source arrays.
        assert_eq!(framed.0, vec![0i64, 2, 2, 5, 7]);
        assert_eq!(
            framed.1,
            indices.iter().map(|&v| v as i32).collect::<Vec<_>>()
        );
        assert_eq!(framed.2, values);
    }

    /// F-b: the indptr-only framed path (`decode_shard_indptr_bytes`) yields the
    /// same global indptr as a full decode, decoding only the per-group indptr
    /// frames (never the indices/values).
    #[test]
    fn decode_shard_indptr_only_framed_matches_full() {
        use crate::encoder::{encode_one_shard, FramingConfig};
        use crate::modality::ModalityType;

        let indptr = [0u64, 2, 2, 5, 7];
        let indices = [0u32, 3, 1, 4, 9, 2, 8];
        let values: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 9.0, 3.0, 7.0];
        let n_cols: u32 = 16;

        let s = encode_one_shard(
            &indptr,
            &indices,
            &values,
            Some(CodecId::ShufDeltaZstd),
            0,
            n_cols,
            0,
            SectionType::CsrShard,
            ModalityType::Rna,
            "X_shard_0".to_string(),
            Some(FramingConfig {
                row_group_rows: 2,
                target_nnz: None,
                trial: false,
                decode_target: None,
            }),
        )
        .expect("encode_one_shard");

        let sh = ShardHeader::read_from(&mut Cursor::new(&s.header_buf[..])).unwrap();
        assert_eq!(sh.shard_format_version, 2, "fixture must be framed (v2)");

        // Assemble the full on-disk section: header ++ sub-streams ++ block index
        // (order matches the header's relative offsets).
        let mut section = Vec::new();
        section.extend_from_slice(&s.header_buf);
        section.extend_from_slice(&s.encoded.indptr_bytes);
        section.extend_from_slice(&s.encoded.indices_bytes);
        section.extend_from_slice(&s.encoded.values_bytes);
        section.extend_from_slice(&s.block_index_bytes);

        let ip_only =
            decode_shard_indptr_bytes(&section, &dummy_entry(), 2).expect("indptr-only decode");
        assert_eq!(ip_only, vec![0i64, 2, 2, 5, 7]);

        // Parity with the full decode's indptr.
        let full = decode_shard_regions_scipy(
            &sh,
            &s.encoded.indptr_bytes,
            &s.encoded.indices_bytes,
            &s.encoded.values_bytes,
            &s.block_index_bytes,
        )
        .expect("full decode");
        assert_eq!(
            ip_only, full.0,
            "indptr-only differs from full-decode indptr"
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

    /// Every site that dispatches on storage order must agree with
    /// `shard::is_column_major`, for **every** section type — not just the two
    /// that are column-major today.
    ///
    /// The predicate being centralised is not itself the guarantee: nothing
    /// stops a future edit from re-spelling `matches!(st, CscShard | ...)` at
    /// one of these sites, which is exactly the state this series found the
    /// tree in (`LayerCscShard` was column-major to three callers and row-major
    /// to five, with no producer to make them disagree out loud). Walking the
    /// discriminants is what turns "adding a column-major section type is a
    /// one-line change" into something a test can fail on.
    #[test]
    fn column_major_dispatch_is_exhaustive() {
        use crate::shard::{derive_shard_type, is_column_major};

        let mut seen_column_major = 0usize;
        for raw in 0u8..=255 {
            let Some(st) = SectionType::from_u8(raw) else {
                continue;
            };
            let expected = is_column_major(st);
            seen_column_major += usize::from(expected);

            assert_eq!(
                derive_shard_type(st) == 1,
                expected,
                "{st:?} (id {raw}): the on-disk shard_type byte disagrees with \
                 is_column_major — a reader would reject what the writer emits"
            );
            assert_eq!(
                catalog_says_column_major(st),
                expected,
                "{st:?} (id {raw}): the decoder's catalog dispatch disagrees with \
                 is_column_major — it would reconcile against the wrong stats pair"
            );
        }

        // Guards the guard: if `from_u8` or the predicate regressed to answering
        // `false` everywhere, every assertion above would hold vacuously.
        assert_eq!(
            seen_column_major, 2,
            "expected exactly CscShard and LayerCscShard to be column-major; \
             update this count deliberately when adding one"
        );
    }
}
