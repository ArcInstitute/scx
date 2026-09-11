//! The per-shard framing layout a scattered read resolves **once**, and the
//! one-group decode built on it.
//!
//! `decode_block_index_row_runs` used to do all of this per call: re-read the
//! 76-byte shard header, re-slice the sub-streams, re-run `resolve_block_index`
//! over every `BlockIndexEntry`, decode the touched groups into a per-call map
//! and drop them. [`FramedShardLayout`] is the part of that work that depends
//! only on the shard — `BackedCsrReader` memoizes one per shard for the
//! reader's lifetime — and [`ScxReader::decode_framed_row_group`] is the part
//! that depends on the group, which the shard LRU retains under
//! `CacheKey::Group` (OPT-FORMATIO-1).

use std::ops::Range;

use scx_codec::RowGroupSpan;

use super::*;

/// Everything about a **row-group-framed (v2)** shard that a scattered read
/// needs and that does not change between reads: the authenticated header
/// scalars, the byte ranges of the three sub-streams within the section, and
/// the resolved block index.
///
/// Holds a clone of the shard's `FullCatalogEntry` so a later decode can
/// re-borrow the section from the mmap without another linear scan of the
/// catalog (`full_entry_at_offset`), and so the `stats`-authenticated minor
/// extent the layout was reconciled against travels with it.
#[derive(Debug)]
pub(crate) struct FramedShardLayout {
    pub(crate) entry: FullCatalogEntry,
    pub(crate) codec: CodecId,
    pub(crate) venc: ValueEncoding,
    pub(crate) index_dtype_u16: bool,
    /// Rows in the shard (`header.n_major`), the bound every per-call row /
    /// run validation is checked against.
    pub(crate) n_major: usize,
    /// Declared minor extent, already reconciled against the catalog.
    pub(crate) n_minor: u32,
    indptr: Range<usize>,
    indices: Range<usize>,
    values: Range<usize>,
    /// One span per row group, ascending `row_start`, contiguous, tiling
    /// `[0, n_major)` — `resolve_block_index` validates all of that.
    pub(crate) spans: Vec<RowGroupSpan>,
}

impl FramedShardLayout {
    /// Index of the row group holding shard-local `row`. `row` must be
    /// `< n_major` (callers validate first); an in-range row always falls in
    /// exactly one span because the spans tile the shard.
    pub(crate) fn find_group(&self, row: usize) -> usize {
        self.spans
            .partition_point(|s| (s.row_start as usize) <= row)
            .saturating_sub(1)
    }

    pub(crate) fn span(&self, g: usize) -> &RowGroupSpan {
        &self.spans[g]
    }

    /// Bytes the decoded group `g` will occupy in the shard LRU — the same
    /// model [`crate::backed::SizeHint`] charges once it is decoded
    /// (`(n_rows + 1) × 8 + nnz × 4 + nnz × 4`), so a caller can budget a
    /// warm **before** decoding.
    pub(crate) fn group_bytes(&self, g: usize) -> usize {
        let s = &self.spans[g];
        crate::backed::csr_component_bytes(s.n_rows as usize + 1, s.nnz as usize, s.nnz as usize)
    }

    /// Reject a shard-local run that reaches past the shard. Runs are built by
    /// callers from `ShardStats` ranges, which are not otherwise checked
    /// against the header's `n_major`; an out-of-range run would make
    /// `find_group` / the local offset overshoot the decoded group CSR and
    /// panic on OOB indexing. Fail loud instead (readers return errors, not
    /// panics, on malformed input). Per call, not per layout: it depends on
    /// the request.
    pub(crate) fn check_run(&self, run_start: usize, run_len: usize) -> Result<()> {
        let run_end = run_start
            .checked_add(run_len)
            .filter(|&e| e <= self.n_major);
        if run_end.is_none() {
            return Err(ScxError::InvalidCatalog(format!(
                "shard {} row run [{run_start}, +{run_len}) exceeds shard n_major {}",
                self.entry.name, self.n_major
            )));
        }
        Ok(())
    }
}

impl ScxReader {
    /// Resolve the framing layout of the shard `entry` names, or `Ok(None)`
    /// for a legacy (v1, unframed) shard so the caller falls back to a
    /// full-shard decode. Reads the 76-byte header and the block index; never
    /// the payload.
    ///
    /// The minor extent comes from the **catalog**, never from a re-derivation
    /// off the `FileHeader`. The catalog is checksummed and the shard payload
    /// is not, which is the whole point of reconciling at all — and it is what
    /// `check_header_against_catalog` already compares the whole-shard decode
    /// against, so the two seams accept the same set of files. Re-deriving it
    /// was wrong for a multimodal shard (stamped with its modality's `n_vars`,
    /// compared against the file-wide max), for `.raw` (its own gene axis, not
    /// on the header), and for an `ObspCsrShard` written before
    /// OPT-FORMATIO-4 (the legacy gene-axis stamp, which the catalog agrees
    /// with). No stats, no authority: skip, exactly as
    /// `check_header_against_catalog` does.
    pub(crate) fn framed_shard_layout(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<Option<FramedShardLayout>> {
        self.guard_csc_sidecar_fresh(entry)?;
        let header = self.read_shard_header(entry)?;
        // Only framed shards carry a resolvable multi-entry block index.
        if header.shard_format_version <= crate::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION {
            return Ok(None);
        }
        let codec =
            CodecId::from_u8(header.codec_id).ok_or(ScxError::UnknownCodec(header.codec_id))?;
        let venc = ValueEncoding::from_u8(header.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(header.value_encoding))?;
        let index_dtype_u16 = header.index_dtype == 0;
        let section = self.section_bytes(entry)?;

        let range = |rel: u32, len: u32, label: &str| -> Result<Range<usize>> {
            let start = rel as usize;
            let end = start
                .checked_add(len as usize)
                .filter(|&e| e <= section.len())
                .ok_or_else(|| {
                    ScxError::InvalidCatalog(format!(
                        "shard {} {label} stream out of bounds",
                        entry.name
                    ))
                })?;
            Ok(start..end)
        };
        let indptr = range(header.indptr_rel_offset, header.indptr_length, "indptr")?;
        let indices = range(header.indices_rel_offset, header.indices_length, "indices")?;
        let values = range(header.values_rel_offset, header.values_length, "values")?;
        let block_index = range(
            header.block_index_rel_offset,
            header.block_index_length,
            "block_index",
        )?;

        let authenticated_minor = entry
            .stats
            .as_ref()
            .map(|stats| crate::shard_decode::catalog_minor_extent(stats, entry.section_type))
            .unwrap_or(0);
        crate::shard_decode::reconcile_declared_minor(
            header.n_minor,
            authenticated_minor,
            &format!("shard '{}'", entry.name),
        )?;

        let spans = crate::shard::resolve_block_index(&header, &section[block_index])?;
        Ok(Some(FramedShardLayout {
            entry: entry.clone(),
            codec,
            venc,
            index_dtype_u16,
            n_major: header.n_major as usize,
            n_minor: header.n_minor,
            indptr,
            indices,
            values,
            spans,
        }))
    }

    /// Decode row group `g` of a framed shard into a **group-local** CSR
    /// (`indptr[0] == 0`, `shape.0 == span.n_rows`), byte-identical to the
    /// matching rows of a full decode. Cost is O(group), not O(shard).
    pub(crate) fn decode_framed_row_group(
        &self,
        layout: &FramedShardLayout,
        g: usize,
    ) -> Result<ScxCsr> {
        let section = self.section_bytes(&layout.entry)?;
        let span = &layout.spans[g];
        let decoded = scx_codec::decode_row_group(
            layout.codec,
            span,
            &section[layout.indptr.clone()],
            &section[layout.indices.clone()],
            &section[layout.values.clone()],
            layout.venc,
            layout.index_dtype_u16,
        )
        .map_err(|e| {
            ScxError::InvalidCatalog(format!(
                "row-group decode of {} group {g}: {e}",
                layout.entry.name
            ))
        })?;
        // The third decode seam: this path calls `scx_codec::decode_row_group`
        // directly and never passes through `decode_shard_regions_scipy`, so
        // it has to supply the minor-axis bound itself. Without it a partial
        // (block-index) read would be the one remaining way to get an
        // unvalidated column index out of the reader.
        let (indptr, indices, data) = scx_codec::decoded_shard_to_scipy(
            decoded,
            layout.venc,
            scx_codec::clamp_index_bound(layout.n_minor),
        )
        // An out-of-range index must keep its structured
        // `ShardIndexOutOfRange` identity, so let the promoting
        // `From<CodecError>` handle that variant and only wrap the rest with
        // the shard name. Blanket-wrapping in `InvalidCatalog` kept the
        // error class right but destroyed the variant, so a caller matching
        // on `ShardIndexOutOfRange` saw this seam behave differently from
        // every other one.
        .map_err(|e| match e {
            scx_codec::CodecError::IndexOutOfRange { .. } => ScxError::from(e),
            other => ScxError::InvalidCatalog(format!(
                "row-group convert of {} group {g}: {other}",
                layout.entry.name
            )),
        })?;
        Ok(ScxCsr::new_unchecked(
            (span.n_rows as usize, layout.n_minor as usize),
            indptr,
            indices,
            data,
        ))
    }
}

/// Build the run-local CSR (`indptr[0] == 0`) for shard-local rows
/// `[run_start, run_start + run_len)` by copying each row out of its group,
/// fetching groups through `lookup` (a cache lookup or a decode — the caller
/// decides). Rows are visited in order, so a group is looked up once per
/// contiguous stretch of rows it holds. The run must already have passed
/// [`FramedShardLayout::check_run`].
pub(crate) fn assemble_row_run(
    layout: &FramedShardLayout,
    run_start: usize,
    run_len: usize,
    mut lookup: impl FnMut(usize) -> Result<Arc<ScxCsr>>,
) -> Result<scx_codec::ScipyShard> {
    let mut r_indptr = Vec::with_capacity(run_len + 1);
    r_indptr.push(0i64);
    let mut r_indices = Vec::new();
    let mut r_data = Vec::new();
    let mut running = 0i64;
    let mut current: Option<(usize, Arc<ScxCsr>)> = None;
    for row in run_start..run_start + run_len {
        let g = layout.find_group(row);
        if current.as_ref().is_none_or(|(cur_g, _)| *cur_g != g) {
            current = Some((g, lookup(g)?));
        }
        let group = &current.as_ref().expect("just set").1;
        let local = row - layout.spans[g].row_start as usize;
        let lo = group.indptr[local] as usize;
        let hi = group.indptr[local + 1] as usize;
        r_indices.extend_from_slice(&group.indices[lo..hi]);
        r_data.extend_from_slice(&group.data[lo..hi]);
        running += (hi - lo) as i64;
        r_indptr.push(running);
    }
    Ok((r_indptr, r_indices, r_data))
}
