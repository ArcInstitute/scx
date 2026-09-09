// scx build-csc — Build CSC (column-major) shards from existing CSR data.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::FramingConfig;
use scx_format_io::MemoryBudget;
use scx_format_io::ScxReader;

use crate::rewrite_helpers;

/// Build CSC (column-major) shards from an existing file's CSR data, rewriting
/// the whole file (CSR is decoded and re-encoded, then the CSC sidecar appended).
///
/// `framing`: when `Some`, both the re-written CSR shards and the emitted CSC
/// shards are row-group-framed (shard v2) and the output header is v4. When
/// `None`, the output is unframed (v1 shards, v3 header) — note this **strips
/// framing from a source that was already framed**; the framing-preserving entry
/// point for arbitrary files is `scx optimize`, so mutating-op callers that don't
/// thread framing pass `None` deliberately.
/// What `build-csc` did, so callers print the truth rather than "Built".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildCscOutcome {
    /// At least one CSC shard was written.
    Built,
    /// The matrix is empty (0 rows or 0 columns): there is nothing to
    /// transpose, the output carries no CSC sidecar — and if the input had a
    /// stale one, it is gone.
    NoSidecar,
}

/// `input` and `output` name the same file — by canonical path when both
/// resolve, else lexically. `std::fs::copy` opens the destination with
/// `O_TRUNC`, so copying a file onto itself through a different spelling
/// (`./a.scx` vs `a.scx`) would truncate the source before reading it.
fn same_file(input: &Path, output: &Path) -> bool {
    match (std::fs::canonicalize(input), std::fs::canonicalize(output)) {
        (Ok(a), Ok(b)) => a == b,
        _ => input == output,
    }
}

pub fn run_build_csc(
    input: &Path,
    output: &Path,
    memory_limit: &str,
    force: bool,
    csc_cols_per_shard: usize,
    framing: Option<FramingConfig>,
) -> Result<BuildCscOutcome, Box<dyn std::error::Error>> {
    // 1. Parse memory limit string ("4G" → 4 * 1024^3 bytes). Shares the
    //    workspace parser so --memory-limit accepts the same forms as
    //    `scx convert --memory-budget` (K/M/G/T, KiB/MiB/GiB/TiB; decimals
    //    rejected).
    let max_bytes = usize::try_from(MemoryBudget::parse(memory_limit)?)?;

    // 2. Validate input exists
    if !input.exists() {
        return Err(format!("input file does not exist: {}", input.display()).into());
    }

    // 3. `output` must be a different file from `input`, by canonical path.
    //    The `--force` arm below unlinks `output` before `input` is opened, so
    //    an alias (`a.scx` vs `./a.scx`) would delete the source and then fail
    //    to open it. The in-place form (no `<OUTPUT>`) is `rebuild_csc_inplace`,
    //    which stages a temp file; the pyscx wrapper has refused this alias
    //    since it was written — the guard belongs here so every caller gets it.
    if same_file(input, output) {
        return Err(format!(
            "input and output must be different files ({} names the input); omit <OUTPUT> \
             to add the CSC sidecar in place, or give a distinct output path",
            output.display()
        )
        .into());
    }

    // 3b. Check output doesn't exist (unless --force)
    if output.exists() && !force {
        return Err(format!(
            "{} already exists (use --force to overwrite)",
            output.display()
        )
        .into());
    }
    if output.exists() && force {
        std::fs::remove_file(output)?;
    }

    // 4. Open input file
    let reader = ScxReader::open(input)?;
    let in_header = reader.header();

    // 5. An empty matrix (0 rows or 0 columns) has nothing for a sidecar to
    //    index, but the requested output must still exist — `rebuild_csc_inplace`
    //    and the convert / sort / subset `--rebuild-csc` callers rename it into
    //    place, and a rewrite that yielded zero rows must not fail after the
    //    fact. An input without a sidecar is copied verbatim (its CSR shards,
    //    if a 0-column file has any, come along). An input that still carries
    //    one (written under the pre-0.17 `CscPolicy::Always`, which emitted a
    //    CSC shard of empty columns) is rewritten without it, so "an empty
    //    matrix has no sidecar" does not depend on how the file was made: a
    //    0-row file has no CSR shards, so its rewrite is obs / var plus the
    //    auxiliary sections; a 0-column file with rows falls through to the
    //    normal path, which re-emits its CSR shards and writes zero CSC shards.
    //    A header that *claims* rows but has no CSR shards is malformed and
    //    stays an error (below).
    // A header that claims rows but carries no CSR shards is malformed, and
    // must be refused before the empty-matrix fast path can copy it through as
    // a success. (A 0-row file legitimately has none: the format forbids framed
    // zero-row shards, so that case is exempt.)
    if in_header.n_obs > 0 && in_header.n_csr_shards == 0 {
        return Err("Input file has no CSR shards".into());
    }
    let empty_matrix = in_header.n_obs == 0 || in_header.n_vars == 0;
    let has_csc = reader
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == scx_format_io::section::SectionType::CscShard);
    if empty_matrix && !has_csc {
        log::info!(
            "build-csc: {} is an empty matrix ({} x {}); no CSC sidecar to build",
            input.display(),
            in_header.n_obs,
            in_header.n_vars
        );
        std::fs::copy(input, output)?;
        return Ok(BuildCscOutcome::NoSidecar);
    }
    if in_header.n_obs == 0 {
        log::info!(
            "build-csc: {} has no rows; dropping its stale CSC sidecar",
            input.display()
        );
        let out_header = FileHeader {
            manifest_sequence: in_header.manifest_sequence + 1,
            ..in_header.clone()
        };
        let mut writer = ScxWriter::new(output, out_header)?
            .with_data_generation(reader.catalog().data_generation);
        rewrite_helpers::copy_obs_var_preserving_layout(&reader, &mut writer)?;
        let params_json = format!(
            "{{\"memory_limit\":\"{memory_limit}\",\"csc_cols_per_shard\":{csc_cols_per_shard}}}"
        );
        rewrite_helpers::copy_auxiliary_sections(&reader, &mut writer, "build-csc", &params_json)?;
        crate::carry::audit_staged(
            crate::carry::RewriteOp::BuildCsc,
            &[reader.catalog()],
            &writer,
        )?;
        writer.finish()?;
        return Ok(BuildCscOutcome::NoSidecar);
    }

    // This function is not modality-aware — it flattens every CSR shard against
    // the single top-level n_obs × n_vars shape, which would corrupt the sidecar
    // on a multimodal input. `pyscx.build_csc` has guarded this since it was
    // written; the CLI did not, and reached `reader.read_var()` to fail with an
    // opaque `section not found: var`. The guard belongs here so every caller —
    // `scx build-csc` in both its forms, `rebuild_csc_inplace`, and the pyscx
    // wrapper — gets the same actionable message.
    if reader.is_multimodal() {
        return Err(format!(
            "build-csc does not support multimodal files ({} has {} modalities); \
             extract a single modality first with \
             `scx subset {} out.scx --modality NAME`",
            input.display(),
            in_header.n_modalities,
            input.display(),
        )
        .into());
    }

    // SCX-005: build-csc re-emits CSR shards without re-canonicalizing them, so
    // it must not *raise* the format version's canonicality claim above what
    // the input already guarantees. Framing (v4) structurally requires the v3
    // canonical-CSR invariant; refuse to frame a pre-v3 input rather than stamp
    // a v4 file whose shards may be unsorted/duplicated. Run `scx optimize`
    // first (which canonicalizes) for such inputs.
    let in_version = in_header.format_version;
    if framing.is_some() && in_version < scx_format_io::header::DEFAULT_WRITE_FORMAT_VERSION {
        return Err(format!(
            "build-csc cannot row-group-frame a format v{in_version} input (framing implies the \
             v3 canonical-CSR invariant, which build-csc does not re-establish); run \
             `scx optimize` first to canonicalize, then build-csc"
        )
        .into());
    }

    let n_rows = in_header.n_obs as usize;
    let n_cols = in_header.n_vars as usize;

    // Show progress
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!(
        "Building CSC shards ({} rows × {} cols)...",
        n_rows, n_cols
    ));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // 6. Read CSR shard entries and pick a CSC value encoding wide enough to
    //    cover EVERY shard. The CSC sidecar transposes all shards into shared
    //    columns, so a first-shard-only encoding can truncate later shards
    //    (SCX-004): a `Uint8` first shard followed by a `Float32` shard would
    //    otherwise encode `1.5` as integer `1`. Scan every header for the
    //    float/integer kind; if any is float the CSC must be Float32 (+ a
    //    float-safe codec), else use the widest integer width across shards.
    let csr_entries = reader.catalog().csr_shards_sorted();
    // One header read per shard, not two: this scan, the CSC codec/encoding
    // pick below and the per-shard CSR re-emit at step 11 all want the same
    // two bytes off each shard header, and the re-emit used to read it again.
    let mut per_shard: Vec<(CodecId, ValueEncoding)> = Vec::with_capacity(csr_entries.len());
    let mut max_int_val: u32 = 0;
    for entry in &csr_entries {
        let sh = reader.read_shard_header(entry)?;
        let ve = ValueEncoding::from_u8(sh.value_encoding)
            .ok_or(format!("unknown value encoding: {}", sh.value_encoding))?;
        let ci = CodecId::from_u8(sh.codec_id).ok_or(format!("unknown codec: {}", sh.codec_id))?;
        per_shard.push((ci, ve));
        if let Some(stats) = entry.stats.as_ref() {
            max_int_val = max_int_val.max(stats.value_max);
        }
    }
    // Floor the integer width on each shard's declared encoding, not only on
    // `stats.value_max`: a shard may lack stats (format-permitted), in which
    // case `value_max` contributes nothing and a wide integer shard could be
    // under-picked as Uint8. `widest_value_encoding` is the crate-wide spelling
    // of that rule (`compact` re-shards under the same one), and it maps any
    // float shard to `Float32` — so its result doubles as the `any_float` test.
    let declared: Vec<ValueEncoding> = per_shard.iter().map(|&(_, ve)| ve).collect();
    let (csc_value_encoding, csc_codec) = match crate::helpers::widest_value_encoding(&declared) {
        // Pcodec is the canonical float codec; the first shard's codec may be
        // an integer-only codec (Scx1) that cannot represent float values.
        ValueEncoding::Float32 | ValueEncoding::Float16 => {
            (ValueEncoding::Float32, CodecId::Pcodec)
        }
        widest_declared => {
            let by_value = if max_int_val <= u8::MAX as u32 {
                ValueEncoding::Uint8
            } else if max_int_val <= u16::MAX as u32 {
                ValueEncoding::Uint16
            } else {
                ValueEncoding::Uint32
            };
            // The wider of the value-derived and header-declared widths. Both
            // are integer encodings here, so the helper's float arm cannot fire.
            let enc = crate::helpers::widest_value_encoding(&[widest_declared, by_value]);
            // `.first()` rather than `csr_entries[0]`: the guard above means a
            // file with rows always has shards, so this reports the same
            // condition that guard already names instead of panicking.
            let (codec, _) = *per_shard.first().ok_or("Input file has no CSR shards")?;
            (enc, codec)
        }
    };

    // 7. Read CSR shards individually (preserves shard boundaries for streaming transpose)
    let csr_shards: Vec<scx_sparse::ScxCsr> = csr_entries
        .iter()
        .map(|entry| {
            let (indptr, indices, data) = reader.read_shard_from_entry(entry)?;
            let n_shard_rows = indptr.len() - 1;
            Ok(scx_sparse::ScxCsr::new(
                (n_shard_rows, n_cols),
                indptr,
                indices,
                data,
            )?)
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    // 9. Set up output header (preserve flags/codec/index dtype, bump manifest;
    // writer fills nnz + shard counts).
    let out_header = FileHeader {
        flags: in_header.flags,
        n_obs: in_header.n_obs,
        n_vars: in_header.n_vars,
        shard_target_rows: in_header.shard_target_rows,
        codec_id: in_header.codec_id,
        index_dtype: in_header.index_dtype,
        manifest_sequence: in_header.manifest_sequence + 1,
        // Row-group framing produces a v4 file (guarded above so the input is
        // already ≥ v3 canonical). Otherwise clamp the unframed version to what
        // the input guarantees — build-csc does not canonicalize, so it must
        // not claim v3 for a pre-v3 input (SCX-005).
        format_version: if framing.is_some() {
            scx_format_io::header::CURRENT_FORMAT_VERSION
        } else {
            scx_format_io::header::rewrite_output_format_version(&[in_version], 1)
        },
        ..Default::default()
    };

    // 10. Create writer and write metadata
    pb.set_message("Writing output file...");
    // build-csc does NOT change the CSR data — it only adds the column-major
    // sidecar. Preserve the source data generation (no bump) so the freshly
    // emitted CSC sidecar reads as fresh: `write_shard_inner` records
    // `csc_build_generation = data_generation` for each CSC shard written,
    // yielding `csc_build_generation == data_generation`.
    let mut writer =
        ScxWriter::new(output, out_header)?.with_data_generation(reader.catalog().data_generation);
    // F5-b: frame the re-written CSR shards and the CSC sidecar (both go through
    // `write_shard_inner`, which consults the writer's framing). No-op when None.
    writer.set_framing(framing);
    // obs/var pass through 1:1, so a row-sharded input must come out
    // row-sharded — see `copy_obs_var_preserving_layout` for what collapsing it
    // would cost. build-csc never drops rows, so it takes no keep mask.
    crate::rewrite_helpers::copy_obs_var_preserving_layout(&reader, &mut writer)?;

    // 11. Re-write CSR shards from input (decode + re-encode, per-shard codec)
    // Re-encode from the shards step 7 already decoded rather than decoding the
    // whole matrix a second time: `ScxCsr::new` validates only, so
    // `csr_shards[i]`'s `(indptr, indices, data)` *is* what a second
    // `read_shard_from_entry` would return. `csr_shards` is borrowed, not
    // consumed, because the transpose at step 13 still needs it.
    //
    // `csr_shards` and `per_shard` are both built by mapping over `csr_entries`
    // with no early exit in between, so all three have the same length. Checked
    // rather than assumed, and in release too: a three-way zip over lists of
    // different lengths truncates to the shortest, which would drop shards from
    // the output with no error anywhere. Two comparisons per op, once.
    assert_eq!(
        csr_entries.len(),
        csr_shards.len(),
        "one decoded shard per catalog entry"
    );
    assert_eq!(
        csr_entries.len(),
        per_shard.len(),
        "one (codec, encoding) pair per catalog entry"
    );
    for ((shard_entry, shard), &(ci, ve)) in csr_entries.iter().zip(&csr_shards).zip(&per_shard) {
        let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);

        let indices_u32: Vec<u32> = shard.indices.iter().map(|&i| i as u32).collect();
        let indptr_u64: Vec<u64> = shard.indptr.iter().map(|&v| v as u64).collect();
        let raw_values = ve.encode_f32_batch(&shard.data)?;

        writer.write_csr_shard(
            &indptr_u64,
            &indices_u32,
            &raw_values,
            ci,
            ve,
            shard_row_start,
        )?;
    }

    // 13. Write CSC shards via the streaming transpose iterator. Each
    //     iterator chunk becomes one CSC shard; col_start is read from
    //     the iterator *before* advancing to the next chunk.
    //
    //     NOTE: this is the inline twin of `scx_format_io::csc_sidecar::write_csc_sidecar`
    //     (the shared helper the convert/pyscx/rscx paths use). It stays inline
    //     here because it drives a progress bar per chunk; keep the encode +
    //     write_csc_shard logic in sync with that helper.
    pb.set_message("Transposing CSR → CSC (streaming)...");
    let mut iter = scx_sparse::streaming_csr_to_csc_iter_with_cap(
        &csr_shards,
        n_rows,
        n_cols,
        max_bytes,
        csc_cols_per_shard,
    )?;

    let mut total_csc_nnz: usize = 0;
    let mut n_csc_shards_written: u32 = 0;
    loop {
        // current_col_start() returns the start of the NEXT chunk
        // (== end of the previous chunk, == 0 on first iteration).
        let col_start = iter.current_col_start() as u64;
        let chunk = match iter.next() {
            Some(c) => c?,
            None => break,
        };

        let csc_indptr_u64: Vec<u64> = chunk.indptr.iter().map(|&v| v as u64).collect();
        let csc_indices_u32: Vec<u32> = chunk.indices.iter().map(|&i| i as u32).collect();
        let csc_raw_values = csc_value_encoding.encode_f32_batch(&chunk.data)?;

        writer.write_csc_shard(
            &csc_indptr_u64,
            &csc_indices_u32,
            &csc_raw_values,
            csc_codec,
            csc_value_encoding,
            col_start,
        )?;

        total_csc_nnz += chunk.data.len();
        n_csc_shards_written += 1;
    }

    // 14. Copy auxiliary sections (layers, obsm, uns, predicate indices, provenance)
    let params_json = format!(
        "{{\"memory_limit\":\"{memory_limit}\",\"csc_cols_per_shard\":{csc_cols_per_shard}}}"
    );
    // The non-canonicalizing (four-argument) form: build-csc clamps its output
    // `format_version` to the source's rather than claiming v3 (SCX-005),
    // precisely so it does not have to canonicalize — and re-emitting a layer
    // with different nnz would contradict its contract of leaving matrix data
    // unchanged.
    rewrite_helpers::copy_auxiliary_sections(&reader, &mut writer, "build-csc", &params_json)?;

    // `copy_auxiliary_sections` carries the predicate-index bytes verbatim, which
    // the carry policy requires — but step 11 above re-encoded every CSR shard
    // through `write_csr_shard`, and `compute_shard_stats` emits no
    // `column_stats`. Level-1 pruning reads *those*, not the section, so without
    // this the index survived while the pruning silently stopped: `filter_obs`
    // fell back to a full scan and still returned the right rows, which is why it
    // went unnoticed long enough to be pinned as a documented gap.
    //
    // The stats are **carried from the input**, not re-derived from the index.
    // Step 11 writes one output shard per input shard at the same `row_start`, so
    // the input's statistics are already exactly right for the output's rows —
    // and re-deriving would have to *infer* that the index is keyed to the CSR
    // partition, which cannot be proven from the index bytes. See
    // `scx_format_io::carry_csr_shard_column_stats`.
    let carried_stats = writer.carry_csr_shard_column_stats_from(&reader.catalog().entries);
    if carried_stats == 0 && reader.read_obs_predicate_index_bytes()?.is_some() {
        log::debug!(
            "build-csc: the input carries an obs predicate index but no per-shard column \
             statistics, so there are none to carry forward. Level-1 shard pruning was \
             already off on the input; rebuild the index with `scx sort`/`scx compact` plus \
             --index-obs / --index-preset to enable it."
        );
    }

    // 15. Check the staged catalog against the declared carry policy, then
    //     finalize. Before `finish()`, not after: `finish()` renames over the
    //     target, and `scx build-csc` with no `<OUTPUT>` makes that target the
    //     input — so an audit afterwards could only report a loss it was too
    //     late to stop.
    crate::carry::audit_staged(
        crate::carry::RewriteOp::BuildCsc,
        &[reader.catalog()],
        &writer,
    )?;
    writer.finish()?;
    pb.finish_and_clear();

    if n_csc_shards_written == 0 {
        // A 0-column matrix with a stale sidecar: the column loop had nothing
        // to emit, and the CLI reports "no CSC sidecar to build" from the
        // outcome — printing "Built CSC … 0 CSC shards" here would contradict it.
        return Ok(BuildCscOutcome::NoSidecar);
    }
    println!(
        "Built CSC: {} → {} ({} rows × {} cols, {} nnz, {} CSC shard{})",
        input.display(),
        output.display(),
        n_rows,
        n_cols,
        total_csc_nnz,
        n_csc_shards_written,
        if n_csc_shards_written == 1 { "" } else { "s" },
    );

    Ok(BuildCscOutcome::Built)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{sample_header, sample_obs, sample_var};

    /// Write a test SCX file with CSR shards.
    fn write_test_input(
        dir: &tempfile::TempDir,
        n_obs: usize,
        n_vars: usize,
    ) -> std::path::PathBuf {
        let path = dir.path().join("input.scx");
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        // Build CSR data: each row has 2 nnz
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_obs {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            indices.push(col0 as u32);
            indices.push(col1 as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }

        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        writer.finish().unwrap();
        path
    }

    /// build-csc is documented as preserving obs metadata, and a row-sharded
    /// layout is part of what "preserved" has to mean.
    ///
    /// Collapsing an `ObsMetadataShard` input into one legacy `ObsMetadata`
    /// section costs peak RSS O(n_obs) — precisely the OOM the sharded layout
    /// exists to avoid — and destroys the precondition Level-2 row-set pushdown
    /// depends on, on the atlas files where pushdown matters most.
    #[test]
    fn build_csc_preserves_sharded_obs_layout() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("sharded.scx");
        let (n_obs, n_vars) = (6usize, 4usize);
        {
            let mut w = ScxWriter::new(&input, sample_header(n_obs as u64, n_vars as u64)).unwrap();
            let obs = sample_obs(n_obs);
            for shard_idx in 0u32..2 {
                let row_start = shard_idx as usize * 3;
                let slice = obs.slice(row_start, 3);
                w.write_obs_shard(shard_idx, row_start as u64, 3, n_obs as u64, &slice)
                    .unwrap();
            }
            w.write_var(&sample_var(n_vars)).unwrap();
            let mut indptr = vec![0u64];
            let (mut indices, mut values) = (Vec::new(), Vec::new());
            for row in 0..n_obs {
                indices.push(((row * 2) % n_vars) as u32);
                indices.push(((row * 2 + 1) % n_vars) as u32);
                values.push(((row + 1) % 256) as u8);
                values.push(((row + 2) % 256) as u8);
                indptr.push(indptr.last().unwrap() + 2);
            }
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            w.finish().unwrap();
        }

        let output = dir.path().join("out.scx");
        run_build_csc(&input, &output, "4G", false, 5000, None).unwrap();

        let reader = ScxReader::open(&output).unwrap();
        assert_eq!(
            reader.obs_metadata_shard_count(),
            2,
            "sharded obs must stay sharded through build-csc"
        );
        assert!(
            !reader
                .catalog()
                .entries
                .iter()
                .any(|e| e.section_type == scx_format_io::section::SectionType::ObsMetadata),
            "no collapsed single-section obs may be emitted"
        );
        // ...and the content still round-trips.
        assert_eq!(reader.read_obs().unwrap().num_rows(), n_obs);
        assert!(reader.header().has_csc());
    }

    /// build-csc adds a sidecar; it must not un-delete anything on the way.
    ///
    /// The in-place form is the dangerous one: it renames a wholly new file
    /// over the target carrying no prior catalog, so `scx rollback` cannot
    /// recover a deletion it dropped. And it is the documented way to restore a
    /// sidecar another op dropped, which puts `mark_deleted` → `build-csc`
    /// directly on the happy path.
    #[test]
    fn build_csc_carries_deletion_vectors() {
        for in_place in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let input = write_test_input(&dir, 6, 4);
            crate::mark_deleted(&input, &[1, 4]).unwrap();

            let read_path = if in_place {
                crate::rebuild_csc::rebuild_csc_inplace(&input, 5000, "4G", None).unwrap();
                input.clone()
            } else {
                let output = dir.path().join("output.scx");
                run_build_csc(&input, &output, "4G", false, 5000, None).unwrap();
                output
            };

            let reader = ScxReader::open(&read_path).unwrap();
            assert!(
                reader.header().has_deletion_vectors(),
                "in_place={in_place}: deletion-vector flag must survive build-csc"
            );
            let dv = reader
                .read_deletion_vectors()
                .unwrap()
                .expect("deletion-vector section present after build-csc");
            assert_eq!(dv.total_deleted(), 2, "in_place={in_place}");
            assert!(dv.is_deleted_global(1) && dv.is_deleted_global(4));

            // Carried, not applied: build-csc is a 1:1 re-emit, so the physical
            // rows (and the row-indexed CSC sidecar built over them) stay put.
            assert_eq!(reader.n_obs(), 6, "in_place={in_place}");
            assert_eq!(reader.read_obs().unwrap().num_rows(), 6);
            assert_eq!(
                reader.read_all_csr_shards_filtered().unwrap().shape.0,
                4,
                "in_place={in_place}: a deletion-aware read sees 6 - 2 rows"
            );
        }
    }

    #[test]
    fn test_build_csc_creates_csc_shards() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 6, 4);
        let output = dir.path().join("output.scx");

        run_build_csc(&input, &output, "4G", false, 5000, None).unwrap();

        // Verify output file
        let reader = ScxReader::open(&output).unwrap();
        let hdr = reader.header();

        // Should have CSR + CSC shards
        assert!(hdr.n_csr_shards > 0, "output must have CSR shards");
        assert!(hdr.n_csc_shards > 0, "output must have CSC shards");
        assert!(hdr.has_csc(), "has_csc flag must be set");

        // CSR data should match original
        let orig_reader = ScxReader::open(&input).unwrap();
        let orig_csr = orig_reader.read_all_csr_shards().unwrap();
        let new_csr = reader.read_all_csr_shards().unwrap();

        assert_eq!(new_csr.shape, orig_csr.shape);
        assert_eq!(new_csr.indptr, orig_csr.indptr);
        assert_eq!(new_csr.indices, orig_csr.indices);
        assert_eq!(new_csr.data, orig_csr.data);

        // Verify CSC data is a valid transpose
        // Convert both CSR and CSC to dense and compare
        let dense_csr = orig_csr.to_dense().unwrap();

        // Read CSC shard from catalog
        let csc_entries: Vec<_> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == scx_format_io::section::SectionType::CscShard)
            .collect();
        assert_eq!(csc_entries.len(), 1, "should have exactly 1 CSC shard");

        // Decode the CSC shard
        let (csc_indptr, csc_indices, csc_data) =
            reader.read_shard_from_entry(csc_entries[0]).unwrap();

        // Reconstruct dense from CSC
        let (n_rows, n_cols) = orig_csr.shape;
        let mut dense_csc = vec![0.0f32; n_rows * n_cols];
        for col in 0..n_cols {
            let start = csc_indptr[col] as usize;
            let end = csc_indptr[col + 1] as usize;
            for j in start..end {
                let row = csc_indices[j] as usize;
                dense_csc[row * n_cols + col] = csc_data[j];
            }
        }
        assert_eq!(dense_csc, dense_csr, "CSC transpose must match CSR data");
    }

    #[test]
    fn test_build_csc_memory_limit() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 10, 6);
        let output = dir.path().join("output_mem.scx");

        // Use a small memory limit that still allows at least 1 col per pass
        // 10 rows × 12 bytes = 120 bytes/col, so 150 → 1 col per pass → 6 passes
        run_build_csc(&input, &output, "150", false, 5000, None).unwrap();

        let reader = ScxReader::open(&output).unwrap();
        assert!(reader.header().has_csc());
        assert!(reader.header().n_csc_shards > 0);

        // Verify data integrity
        let orig_reader = ScxReader::open(&input).unwrap();
        let orig_csr = orig_reader.read_all_csr_shards().unwrap();
        let new_csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(new_csr.data, orig_csr.data);
    }

    #[test]
    fn test_build_csc_force_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 4, 3);
        let output = dir.path().join("output_force.scx");

        // Create the output file first
        std::fs::write(&output, b"placeholder").unwrap();

        // Without --force should fail
        let err = run_build_csc(&input, &output, "4G", false, 5000, None);
        assert!(err.is_err());

        // With --force should succeed
        run_build_csc(&input, &output, "4G", true, 5000, None).unwrap();
        let reader = ScxReader::open(&output).unwrap();
        assert!(reader.header().has_csc());
    }

    #[test]
    fn test_build_csc_no_csr_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.scx");

        // Write a file with obs/var but no CSR shards
        let header = sample_header(3, 2);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(3)).unwrap();
        writer.write_var(&sample_var(2)).unwrap();

        // Need at least one shard for finish to write a catalog with data
        // Actually, let's just test with a proper file that has no shards
        writer.finish().unwrap();

        let output = dir.path().join("output.scx");
        let err = run_build_csc(&path, &output, "4G", false, 5000, None);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("no CSR shards"));
    }

    /// An empty matrix (`n_obs == 0` here) has nothing to transpose, so
    /// `build-csc` is a no-op that still produces the requested output — the
    /// `--rebuild-csc` / `rebuild_csc=True` callers rename that output into
    /// place and must not fail on a rewrite that yielded zero rows. Distinct
    /// from `test_build_csc_no_csr_error`, whose header *claims* rows.
    #[test]
    fn test_build_csc_empty_matrix_is_a_noop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.scx");
        let mut writer = ScxWriter::new(&path, sample_header(0, 2)).unwrap();
        writer.write_obs(&sample_obs(0)).unwrap();
        writer.write_var(&sample_var(2)).unwrap();
        writer.finish().unwrap();

        let output = dir.path().join("output.scx");
        let outcome = run_build_csc(&path, &output, "4G", false, 5000, None)
            .expect("build-csc on an empty matrix must succeed");
        assert_eq!(outcome, BuildCscOutcome::NoSidecar);
        let reader = ScxReader::open(&output).unwrap();
        assert!(!reader.header().has_csc());
        assert_eq!(reader.header().n_obs, 0);
        assert_eq!(reader.header().n_vars, 2);
        assert_eq!(reader.read_obs().unwrap().num_rows(), 0);
        assert_eq!(reader.read_var().unwrap().num_rows(), 2);
    }

    /// A 0-row file written under the pre-0.17 `CscPolicy::Always` carries a
    /// CSC shard of empty columns. `build-csc` must not preserve it through the
    /// copy shortcut — the output has no sidecar whatever the input had.
    #[test]
    fn test_build_csc_empty_matrix_drops_a_stale_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty_with_csc.scx");
        let mut writer = ScxWriter::new(&path, sample_header(0, 2)).unwrap();
        writer.write_obs(&sample_obs(0)).unwrap();
        writer.write_var(&sample_var(2)).unwrap();
        writer.write_uns(&serde_json::json!({"k": "v"})).unwrap();
        // Two empty columns, zero rows: what the old policy emitted.
        writer
            .write_csc_shard(&[0, 0, 0], &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
            .unwrap();
        writer.finish().unwrap();
        assert!(ScxReader::open(&path).unwrap().header().has_csc());

        let output = dir.path().join("output.scx");
        let outcome = run_build_csc(&path, &output, "4G", false, 5000, None).unwrap();
        assert_eq!(outcome, BuildCscOutcome::NoSidecar);
        let reader = ScxReader::open(&output).unwrap();
        assert!(!reader.header().has_csc(), "the stale sidecar must be gone");
        assert_eq!(reader.header().n_obs, 0);
        assert_eq!(reader.read_obs().unwrap().num_rows(), 0);
        assert_eq!(reader.read_var().unwrap().num_rows(), 2);
        assert_eq!(reader.read_uns().unwrap()["k"], "v", "uns is carried");
    }

    /// `input` and `output` naming the same file through different spellings
    /// is refused before the `--force` arm can unlink it: the source stays
    /// byte-identical.
    #[cfg(unix)]
    #[test]
    fn test_build_csc_refuses_an_aliased_output_before_force_unlinks_it() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 4, 3);
        let before = std::fs::read(&input).unwrap();
        // A symlink is a different path (lexically unequal) to the same file;
        // only the canonical comparison sees through it. (`Path` equality
        // already normalises `.` components, so `<dir>/./<name>` would not be
        // a real alias here.)
        let alias = dir.path().join("alias.scx");
        std::os::unix::fs::symlink(&input, &alias).unwrap();
        assert_ne!(alias, input);
        let err = run_build_csc(&input, &alias, "4G", true, 5000, None).unwrap_err();
        assert!(err.to_string().contains("must be different files"), "{err}");
        assert_eq!(
            std::fs::read(&input).unwrap(),
            before,
            "the source must be untouched"
        );
        assert!(
            alias.symlink_metadata().is_ok(),
            "the alias itself must not be unlinked"
        );
    }

    /// The empty-matrix fast path must not launder a malformed file: a header
    /// that claims rows with no CSR shards stays an error on the 0-column axis
    /// too (the 2-column twin is `test_build_csc_no_csr_error`).
    #[test]
    fn test_build_csc_zero_columns_with_rows_but_no_shards_is_still_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("malformed.scx");
        let mut writer = ScxWriter::new(&path, sample_header(3, 0)).unwrap();
        writer.write_obs(&sample_obs(3)).unwrap();
        writer.write_var(&sample_var(0)).unwrap();
        writer.finish().unwrap();
        let output = dir.path().join("output.scx");
        let err = run_build_csc(&path, &output, "4G", false, 5000, None).unwrap_err();
        assert!(err.to_string().contains("no CSR shards"), "{err}");
        assert!(
            !output.exists(),
            "nothing may be written for a malformed input"
        );
        let err = crate::rebuild_csc::rebuild_csc_inplace(&path, 5000, "4G", None).unwrap_err();
        assert!(err.to_string().contains("no CSR shards"), "{err}");
    }

    /// A 0-column matrix with rows and no sidecar is copied verbatim (CSR
    /// shards included); with a stale sidecar it takes the normal path, which
    /// re-emits its CSR shards and writes zero CSC shards.
    #[test]
    fn test_build_csc_zero_columns_writes_no_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zero_cols.scx");
        let mut writer = ScxWriter::new(&path, sample_header(3, 0)).unwrap();
        writer.write_obs(&sample_obs(3)).unwrap();
        writer.write_var(&sample_var(0)).unwrap();
        writer
            .write_csr_shard(
                &[0, 0, 0, 0],
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer.finish().unwrap();

        let output = dir.path().join("output.scx");
        let outcome = run_build_csc(&path, &output, "4G", false, 5000, None).unwrap();
        assert_eq!(outcome, BuildCscOutcome::NoSidecar);
        let reader = ScxReader::open(&output).unwrap();
        assert!(!reader.header().has_csc());
        assert_eq!(reader.header().n_obs, 3);
        assert_eq!(reader.header().n_vars, 0);
        assert_eq!(reader.read_all_csr_shards().unwrap().shape, (3, 0));

        // Same shape carrying a stale sidecar: the normal path strips it.
        let stale = dir.path().join("zero_cols_csc.scx");
        let mut writer = ScxWriter::new(&stale, sample_header(3, 0)).unwrap();
        writer.write_obs(&sample_obs(3)).unwrap();
        writer.write_var(&sample_var(0)).unwrap();
        writer
            .write_csr_shard(
                &[0, 0, 0, 0],
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer
            .write_csc_shard(&[0], &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
            .unwrap();
        writer.finish().unwrap();
        assert!(ScxReader::open(&stale).unwrap().header().has_csc());
        let output2 = dir.path().join("output2.scx");
        let outcome = run_build_csc(&stale, &output2, "4G", false, 5000, None).unwrap();
        assert_eq!(outcome, BuildCscOutcome::NoSidecar);
        let reader = ScxReader::open(&output2).unwrap();
        assert!(!reader.header().has_csc());
        assert_eq!(reader.read_all_csr_shards().unwrap().shape, (3, 0));
    }

    /// write a small CSR-only file, run build-csc with
    /// `--csc-cols-per-shard 3` over n_vars=10, and verify that the
    /// output has exactly ceil(10/3) = 4 CSC shards with correct,
    /// non-overlapping `[col_start, col_end)` ranges.
    #[test]
    fn test_build_csc_multi_shard_layout() {
        let dir = tempfile::tempdir().unwrap();
        // 4 rows × 10 cols. write_test_input gives each row 2 nnz.
        let input = write_test_input(&dir, 4, 10);
        let output = dir.path().join("multi_csc.scx");

        run_build_csc(&input, &output, "4G", false, 3, None).unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let hdr = reader.header();
        assert!(hdr.has_csc());
        assert_eq!(hdr.n_csc_shards, 4, "ceil(10/3) = 4 CSC shards");

        let csc_entries = reader.catalog().csc_shards_sorted();
        assert_eq!(csc_entries.len(), 4);
        let ranges: Vec<std::ops::Range<u64>> = csc_entries
            .iter()
            .map(|e| e.stats.as_ref().unwrap().col_range())
            .collect();
        // Chunks: [0..3), [3..6), [6..9), [9..10).
        assert_eq!(ranges, vec![0..3, 3..6, 6..9, 9..10]);

        // CSC ranges are contiguous and cover [0, n_vars).
        for w in ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, hdr.n_vars);

        // Every CSC shard's on-disk shard_type byte is 1.
        let data = std::fs::read(&output).unwrap();
        for entry in &csc_entries {
            let section = &data[entry.offset as usize..][..entry.length as usize];
            let sh = scx_format_io::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format_io::shard::SHARD_HEADER_SIZE],
            ))
            .unwrap();
            assert_eq!(sh.shard_type, 1);
        }

        // High-level reads round-trip: CSC == densify(CSR).
        let orig_reader = ScxReader::open(&input).unwrap();
        let dense_csr = orig_reader
            .read_all_csr_shards()
            .unwrap()
            .to_dense()
            .unwrap();
        let csc_concat = reader.read_all_csc_shards().unwrap();
        assert_eq!(csc_concat.shape, (4, 10));
        assert_eq!(csc_concat.to_dense().unwrap(), dense_csr);
    }

    /// `read_csc_columns(2..7)` over the multi-shard layout
    /// returns the same densified slice as densifying the full matrix
    /// then column-slicing. Validates that partial-overlap shards are
    /// `col_slice`d post-decode and that fully-skipped shards do not
    /// affect the result.
    #[test]
    fn test_build_csc_multi_shard_read_columns_range() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 4, 10);
        let output = dir.path().join("multi_csc_range.scx");

        run_build_csc(&input, &output, "4G", false, 3, None).unwrap();

        let orig_reader = ScxReader::open(&input).unwrap();
        let dense = orig_reader
            .read_all_csr_shards()
            .unwrap()
            .to_dense()
            .unwrap();
        let n_cols = orig_reader.header().n_vars as usize;

        let reader = ScxReader::open(&output).unwrap();

        // Reference dense slice for [c_lo, c_hi).
        let dense_slice = |c_lo: usize, c_hi: usize| -> Vec<f32> {
            let cols = c_hi - c_lo;
            let mut out = vec![0.0f32; 4 * cols];
            for r in 0..4 {
                for (oc, sc) in (c_lo..c_hi).enumerate() {
                    out[r * cols + oc] = dense[r * n_cols + sc];
                }
            }
            out
        };

        let cases = [
            (2u32, 7u32), // partial overlap on shards 0 and 2; full shard 1
            (0, 10),      // entire range
            (3, 6),       // exact-boundary single shard
            (4, 5),       // single column
            (8, 10),      // crosses the trailing 1-col shard
            (0, 0),       // empty
        ];
        for (c_lo, c_hi) in cases {
            let csc = reader.read_csc_columns(c_lo..c_hi).unwrap();
            let got = csc.to_dense().unwrap();
            let want = dense_slice(c_lo as usize, c_hi as usize);
            assert_eq!(got, want, "mismatch on cols [{c_lo}..{c_hi})");
        }
    }

    ///  codec parity: rerun the A.4 codec sweep idea
    /// (None / Scx1 / Zstd / Lz4Shuffle / Pcodec × Uint8) on a
    /// multi-shard CSC layout. Inputs are integer Uint8 throughout; the
    /// codec from the input shards drives the output codec.
    #[test]
    fn test_build_csc_multi_shard_codec_parity() {
        let codecs = [
            CodecId::None,
            CodecId::Scx1,
            CodecId::Zstd,
            CodecId::Lz4Shuffle,
            CodecId::Pcodec,
        ];

        for codec in codecs {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("input.scx");
            // Build a 4×10 file using the requested codec.
            let n_obs = 4usize;
            let n_vars = 10usize;
            let header = sample_header(n_obs as u64, n_vars as u64);
            let mut writer = ScxWriter::new(&path, header).unwrap();
            writer.write_obs(&sample_obs(n_obs)).unwrap();
            writer.write_var(&sample_var(n_vars)).unwrap();

            let mut indptr = vec![0u64];
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for row in 0..n_obs {
                let c0 = (row * 2) % n_vars;
                let c1 = (row * 2 + 1) % n_vars;
                indices.push(c0 as u32);
                indices.push(c1 as u32);
                values.push(((row + 1) % 256) as u8);
                values.push(((row + 2) % 256) as u8);
                indptr.push(indptr.last().unwrap() + 2);
            }
            writer
                .write_csr_shard(&indptr, &indices, &values, codec, ValueEncoding::Uint8, 0)
                .unwrap();
            writer.finish().unwrap();

            let output = dir.path().join("multi_csc.scx");
            run_build_csc(&path, &output, "4G", false, 3, None).unwrap();

            let reader = ScxReader::open(&output).unwrap();
            assert_eq!(reader.header().n_csc_shards, 4, "codec={codec:?}");

            // CSC density equals CSR density.
            let orig = ScxReader::open(&path).unwrap();
            let dense_csr = orig.read_all_csr_shards().unwrap().to_dense().unwrap();
            let csc = reader.read_all_csc_shards().unwrap();
            assert_eq!(
                csc.to_dense().unwrap(),
                dense_csr,
                "round-trip mismatch for codec={codec:?}"
            );
        }
    }
}
