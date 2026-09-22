// scx build-csc — Build CSC (column-major) shards from existing CSR data.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::csc_budget;
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
    // `temp_dir`: root for the CSC builder's column-bucket spill files. `None`
    // uses the output file's own directory — see `TempDirSpillStore` for why
    // that, rather than the platform temp dir the other spilling ops default
    // to.
    temp_dir: Option<&Path>,
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
    //    An alias (`a.scx` vs `./a.scx`) would otherwise have the rewrite
    //    write over its own source. The in-place form (no `<OUTPUT>`) is
    //    `rebuild_csc_inplace`,
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
    // `symlink_metadata`, not `exists()`: the latter follows the link and
    // answers `false` for a dangling symlink, which would let an unforced
    // rewrite replace it.
    if std::fs::symlink_metadata(output).is_ok() && !force {
        return Err(format!(
            "{} already exists (use --force to overwrite)",
            output.display()
        )
        .into());
    }
    // Deliberately no `remove_file`: `ScxWriter::finish` persists with
    // `rename(2)`, which replaces the output atomically, so unlinking first
    // would only widen a window in which neither the old nor the new file
    // exists. Same-path is already refused above.

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
            "{{\"memory_limit\":\"{memory_limit}\",\"csc_cols_per_shard\":{csc_cols_per_shard},\
          \"temp_dir\":{}}}",
            temp_dir.map_or_else(
                || "null".to_string(),
                |p| format!("{:?}", p.display().to_string())
            )
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
    // One *standalone* `read_shard_header` per shard, not two plus one: this
    // scan, the CSC codec/encoding pick below and the per-shard CSR re-emit at
    // step 11 all want the same two bytes, and the re-emit used to read them
    // again (as did a third call for shard 0's codec). `decode_shard_bytes`
    // still parses the header out of the section it already fetched on the one
    // surviving decode, so total header *parses* go 4n+1 -> 2n; what this loop
    // removes is n+1 standalone reads, and with them the second decode pass.
    //
    // The rule itself is `scx_format_io::pick_csc_encoding`, shared with
    // `ScxWriter`'s finish-time sidecar emit, which used to carry a
    // hand-rolled copy of it. This loop's job is only to collect its two
    // inputs. Flooring on each shard's *declared* encoding matters beyond
    // `stats.value_max`: a shard may lack stats (format-permitted), so
    // `value_max` would contribute nothing and a wide integer shard could be
    // under-picked as Uint8.
    let mut per_shard: Vec<(CodecId, ValueEncoding)> = Vec::with_capacity(csr_entries.len());
    let mut declared_encs: Vec<ValueEncoding> = Vec::with_capacity(csr_entries.len());
    let mut max_int_val: u32 = 0;
    let mut worst_decoded_bytes: u64 = 0;
    for entry in &csr_entries {
        let sh = reader.read_shard_header(entry)?;
        let ve = ValueEncoding::from_u8(sh.value_encoding).ok_or(
            crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
        )?;
        let ci = CodecId::from_u8(sh.codec_id)
            .ok_or(crate::error::OpsError::UnknownCodec(sh.codec_id))?;
        per_shard.push((ci, ve));
        declared_encs.push(ve);
        // The exact decoded cost of THIS shard, from the header that is
        // already in hand. `ShardHeader` carries both `nnz` and `n_major`, so
        // neither term needs `ShardStats` — which is format-permitted to be
        // absent, and whose absence would otherwise let a shard contribute
        // nothing to the refusal below.
        worst_decoded_bytes = worst_decoded_bytes.max(csc_budget::decoded_csr_shard_bytes(
            sh.nnz,
            sh.n_major as u64,
        ));
        if let Some(stats) = entry.stats.as_ref() {
            max_int_val = max_int_val.max(stats.value_max);
        }
    }
    // One spelling of the SCX-004 widening rule, shared with
    // `ScxWriter`'s finish-time sidecar emit. `None` means the integer path
    // found no source shard to take a codec from; the guard above means a file
    // with rows always has shards, so this reports that same condition instead
    // of panicking.
    let (csc_value_encoding, csc_codec) = scx_format_io::pick_csc_encoding(
        &declared_encs,
        max_int_val,
        per_shard.first().map(|&(codec, _)| codec),
    )
    .ok_or_else(|| "Input file has no CSR shards".to_string())?;

    // 7. Refuse a budget that cannot admit one decoded source shard.
    //
    // This is what makes the push-phase budget row enforceable rather than
    // aspirational: the walk below holds exactly one decoded shard at a time,
    // so if the largest one does not fit its share, no amount of spilling
    // helps and the op should say so before writing anything. The byte figure
    // comes from `Share::min_budget_for`, which exists so a refusal and its
    // "raise it to at least N" message cannot drift apart.
    //
    // `worst_decoded_bytes` is folded from each `ShardHeader` above, not from
    // `ShardStats`, and both of its terms are per-shard. The first version of
    // this guard got both halves wrong: it took `nnz` from `entry.stats` — so
    // a stats-less shard contributed nothing, and a file whose shards all lack
    // stats skipped the check entirely — and it charged `n_obs` for the indptr
    // instead of the shard's own `n_major`, which on a 1M-row file with 20k-row
    // shards is ~8 MB of phantom indptr per shard and refuses budgets that
    // actually fit. The header carries both numbers exactly.
    if worst_decoded_bytes > 0 {
        let need = csc_budget::CSC_BUILD_INPUT_SHARE.min_budget_for(worst_decoded_bytes);
        if (max_bytes as u64) < need {
            return Err(format!(
                "build-csc: --memory-limit {memory_limit} is too small for {}: decoding its \
                 largest CSR shard needs ~{worst_decoded_bytes} bytes; raise --memory-limit \
                 to at least {need}",
                input.display()
            )
            .into());
        }
    }

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

    // 11 + 13. One walk: decode a shard, re-emit it as CSR, feed it to the CSC
    // builder, drop it.
    //
    // These used to be two passes over a `Vec<ScxCsr>` holding every decoded
    // shard at once — 11.2 GB of census_1m's 14.7 GB peak — because
    // `streaming_csr_to_csc_iter_with_cap` borrowed `&[ScxCsr]` for its whole
    // lifetime. A push sink removes the borrow, and then the only reason to
    // keep the shards was that two consumers wanted each one. So both consume
    // it in the same iteration and it is dropped at the end of the body: peak
    // goes from every shard to one.
    //
    // Not a second decode pass instead: on census_1m that is roughly a third
    // of the op's wall.
    //
    // `per_shard` stays a list beside `csr_entries` rather than being folded
    // in, because the encoding scan at step 6 runs *before* this walk (it
    // reads headers and catalog stats only, never a payload) and its answer is
    // needed to construct the builder.
    //
    // One check, both directions. Indexing only catches `per_shard` being
    // *longer* (a bounds panic); a shorter one would silently stop early and
    // drop shards from the output with no error anywhere, the same hazard a
    // truncating `zip` has.
    assert_eq!(
        per_shard.len(),
        csr_entries.len(),
        "one header scan per catalog entry"
    );

    let spill_root = temp_dir
        .map(Path::to_path_buf)
        .or_else(|| output.parent().map(Path::to_path_buf));
    let store = scx_format_io::TempDirSpillStore::new(spill_root.as_deref())?;
    let mut builder = scx_sparse::CscBuilder::new(
        n_rows,
        n_cols,
        scx_sparse::CscBuilderConfig {
            cols_per_shard: csc_cols_per_shard,
            memory_bytes: max_bytes,
            spill_after_bytes: csc_budget::CSC_BUILD_BUCKET_SHARE.of(max_bytes as u64) as usize,
            ..Default::default()
        },
        Box::new(store),
    )?;

    pb.set_message("Re-writing CSR shards and routing them into the CSC builder...");
    let mut rows_pushed: u64 = 0;
    for (i, shard_entry) in csr_entries.iter().enumerate() {
        let (ci, ve) = per_shard[i];
        let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);

        let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
        let n_shard_rows = indptr.len() - 1;
        let shard = scx_sparse::ScxCsr::new((n_shard_rows, n_cols), indptr, indices, data)?;

        // Consumer 1: the verbatim-intent CSR re-emit. Unchanged.
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
        drop((indptr_u64, indices_u32, raw_values));

        // Consumer 2: the CSC builder. The row axis it is fed is the walk's
        // own running count, and `push_shard` checks it against the entry's
        // declared `row_start`, so a catalog whose shards do not tile
        // `[0, n_obs)` in order is an error rather than a silently shifted
        // sidecar. (`csr_shards_sorted` sorts by `major_start`, but an entry
        // with no stats sorts last at `u64::MAX` and keeps catalog order, so
        // the two can genuinely disagree.)
        if shard_row_start != rows_pushed {
            return Err(format!(
                "build-csc: CSR shard {i} declares row_start {shard_row_start}, but \
                 {rows_pushed} rows precede it; the shards do not tile [0, {n_rows}) in order"
            )
            .into());
        }
        builder.push_shard(rows_pushed, &shard)?;
        rows_pushed += n_shard_rows as u64;

        pb.set_message(format!("shard {}/{} re-written", i + 1, csr_entries.len()));
    }

    // 13. Drain the builder into CSC sections. The `on_shard` callback is what
    //     let the inline twin of this loop be deleted; its NOTE said it stayed
    //     inline "because it drives a progress bar per chunk".
    pb.set_message("Emitting CSC shards...");
    let mut emitter = builder.finish()?;
    let n_planned = emitter.plan().len();
    let emit_opts = scx_format_io::CscEmitOptions {
        value_encoding: csc_value_encoding,
        codec_id: csc_codec,
        modality_id: None,
    };
    let csc_stats = scx_format_io::emit_csc_shards(
        &mut writer,
        &mut emitter,
        &emit_opts,
        |idx, col_start, nnz| {
            pb.set_message(format!(
                "CSC shard {}/{n_planned} (from column {col_start}, {nnz} nnz)",
                idx + 1
            ));
        },
    )?;
    let total_csc_nnz = csc_stats.total_nnz as usize;
    let n_csc_shards_written = csc_stats.n_shards;
    if csc_stats.spill_bytes > 0 {
        log::info!(
            "build-csc: spilled {} bytes of column buckets under a {max_bytes}-byte budget",
            csc_stats.spill_bytes
        );
    }
    if let Some(col) = csc_stats.first_non_strict_column {
        log::warn!(
            "build-csc: column {col} has a duplicate (row, col) in the source, so its CSC rows \
             are not strictly increasing; GPU routes validating `sorted` will reject this sidecar"
        );
    }

    // 14. Copy auxiliary sections (layers, obsm, uns, predicate indices, provenance)
    let params_json = format!(
        "{{\"memory_limit\":\"{memory_limit}\",\"csc_cols_per_shard\":{csc_cols_per_shard},\
          \"temp_dir\":{}}}",
        temp_dir.map_or_else(
            || "null".to_string(),
            |p| format!("{:?}", p.display().to_string())
        )
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

    /// The same, split across `n_shards` row-shards, so a test can say
    /// something about multi-shard order rather than about a single section.
    fn write_test_input_multi_shard(
        dir: &tempfile::TempDir,
        n_obs: usize,
        n_vars: usize,
        n_shards: usize,
    ) -> std::path::PathBuf {
        let path = dir.path().join("input_multi.scx");
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let rows_per = n_obs.div_ceil(n_shards);
        for s in 0..n_shards {
            let lo = s * rows_per;
            let hi = ((s + 1) * rows_per).min(n_obs);
            let mut indptr = vec![0u64];
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for row in lo..hi {
                let mut cols = [(row * 2) % n_vars, (row * 2 + 1) % n_vars];
                cols.sort_unstable();
                for (k, c) in cols.iter().enumerate() {
                    indices.push(*c as u32);
                    values.push(((row + k + 1) % 256) as u8);
                }
                indptr.push(indptr.last().unwrap() + 2);
            }
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    lo as u64,
                )
                .unwrap();
        }
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
        run_build_csc(&input, &output, "4G", false, 5000, None, None).unwrap();

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
                crate::rebuild_csc::rebuild_csc_inplace(&input, 5000, "4G", None, None).unwrap();
                input.clone()
            } else {
                let output = dir.path().join("output.scx");
                run_build_csc(&input, &output, "4G", false, 5000, None, None).unwrap();
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

        run_build_csc(&input, &output, "4G", false, 5000, None, None).unwrap();

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

    /// The freshness stamp and the section order, both of which the merged
    /// shard walk could have broken quietly.
    ///
    /// `csc_build_generation` is stamped in exactly one place —
    /// `write_shard_inner`, on any column-major section — so a CSC emit that
    /// ever wrote a pre-encoded section instead would leave it `None` and
    /// every read of the new sidecar would raise `StaleCscSidecar`. That is
    /// what this half pins; note it cannot see a *bumped* generation, because
    /// the stamp is taken from `data_generation` and the two move together.
    /// And all CSR writes must still precede all CSC
    /// writes: the walk interleaves a `write_csr_shard` with each
    /// `push_shard`, and only the `finish()` drain afterwards emits CSC — if
    /// anyone later moves the drain inside the walk, the two families
    /// interleave and `carry_csr_shard_column_stats_from` stops lining up
    /// with the input's entries.
    #[test]
    fn build_csc_keeps_the_sidecar_fresh_and_the_sections_ordered() {
        let dir = tempfile::tempdir().unwrap();
        // Three CSR shards, so "CSR before CSC" is a claim about order and not
        // about there being one of each.
        let input = write_test_input_multi_shard(&dir, 9, 8, 3);
        let output = dir.path().join("ordered.scx");
        run_build_csc(&input, &output, "4G", false, 3, None, None).unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let cat = reader.catalog();
        assert_eq!(
            cat.csc_build_generation, cat.data_generation,
            "a sidecar stamped at a different generation than the data reads as stale"
        );

        let mut seen_csc = false;
        let mut n_csr = 0usize;
        let mut n_csc = 0usize;
        for e in &cat.entries {
            match e.section_type {
                scx_format::SectionType::CsrShard => {
                    assert!(!seen_csc, "CSR shard {} written after a CSC shard", e.name);
                    n_csr += 1;
                }
                scx_format::SectionType::CscShard => {
                    seen_csc = true;
                    n_csc += 1;
                }
                _ => {}
            }
        }
        assert_eq!(n_csr, 3, "one output CSR shard per input shard");
        assert!(n_csc >= 3, "8 columns at 3 per shard is 3 CSC shards");
    }

    #[test]
    fn test_build_csc_memory_limit() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 10, 6);
        let output = dir.path().join("output_mem.scx");

        // Small enough to bind the shard width — 10 rows x 12 B = 120 B per
        // column, so 1200 gives 10 columns and the 6-column matrix lands in
        // one shard — while still admitting one decoded source shard, which
        // the refusal below requires.
        run_build_csc(&input, &output, "1200", false, 5000, None, None).unwrap();

        let reader = ScxReader::open(&output).unwrap();
        assert!(reader.header().has_csc());
        assert!(reader.header().n_csc_shards > 0);

        // Verify data integrity
        let orig_reader = ScxReader::open(&input).unwrap();
        let orig_csr = orig_reader.read_all_csr_shards().unwrap();
        let new_csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(new_csr.data, orig_csr.data);
    }

    /// A budget too small for **one decoded source shard** is refused, with
    /// the byte figure to raise it to.
    ///
    /// This is a deliberate behaviour change, and the reason the allocation
    /// table's push row can say `enforced: true`. The predecessor accepted any
    /// budget down to one column's dense worst case and then ignored it: the
    /// `build_csc` benchmark measured a declared 512 MiB producing a 2,233 MB
    /// allocation on tabula, 4.4x over, with no error. The op cannot run
    /// bounded below one shard, so saying so beats overshooting silently.
    /// The refusal must size the shard it is about to decode, not the matrix.
    ///
    /// Both halves of this were wrong in the first version and neither was
    /// visible to `test_build_csc_refuses_a_budget_below_one_source_shard`,
    /// whose fixture is a single 10-row shard *with* stats — so `n_obs`
    /// equalled `n_major` and `entry.stats` was always present.
    ///
    /// (a) Charging `n_obs` for every shard's indptr over-states a multi-shard
    /// file: 40 rows in 4 shards is 10 rows of indptr each, not 40. At census
    /// proportions — 20k-row shards in a 1M-row file — that is ~8 MB of
    /// phantom indptr per shard, multiplied by four by the 1/4 share, and it
    /// refuses budgets that fit.
    #[test]
    fn the_refusal_sizes_one_shard_not_the_whole_matrix() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input_multi_shard(&dir, 40, 6, 4);
        let output = dir.path().join("sized.scx");

        // Each shard is 10 rows x 2 nnz = 20 nnz: 20*8 + 11*8 = 248 B, so the
        // 1/4 share needs 992 B. Charging all 40 rows of indptr would make it
        // 20*8 + 41*8 = 488 B -> 1952 B, so a budget between the two is the
        // discriminating case: it must be accepted.
        run_build_csc(&input, &output, "1200", false, 5000, None, None).expect(
            "1200 B admits a 10-row shard; only the whole-matrix indptr made it look too small",
        );
        assert!(ScxReader::open(&output).unwrap().header().has_csc());
    }

    #[test]
    fn test_build_csc_refuses_a_budget_below_one_source_shard() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 10, 6);
        let output = dir.path().join("output_too_small.scx");

        let err = run_build_csc(&input, &output, "150", false, 5000, None, None)
            .expect_err("150 bytes cannot hold a decoded shard")
            .to_string();
        assert!(err.contains("--memory-limit 150"), "{err}");
        assert!(err.contains("raise --memory-limit to at least"), "{err}");
        assert!(
            !output.exists(),
            "the refusal must come before anything is written"
        );

        // Premise: the same input succeeds once the budget admits a shard, so
        // the test is about the budget and not about the fixture.
        run_build_csc(&input, &output, "4G", false, 5000, None, None).unwrap();
        assert!(ScxReader::open(&output).unwrap().header().has_csc());
    }

    #[test]
    fn test_build_csc_force_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 4, 3);
        let output = dir.path().join("output_force.scx");

        // Create the output file first
        std::fs::write(&output, b"placeholder").unwrap();

        // Without --force should fail
        let err = run_build_csc(&input, &output, "4G", false, 5000, None, None);
        assert!(err.is_err());

        // With --force should succeed
        run_build_csc(&input, &output, "4G", true, 5000, None, None).unwrap();
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
        let err = run_build_csc(&path, &output, "4G", false, 5000, None, None);
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
        let outcome = run_build_csc(&path, &output, "4G", false, 5000, None, None)
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
        let outcome = run_build_csc(&path, &output, "4G", false, 5000, None, None).unwrap();
        assert_eq!(outcome, BuildCscOutcome::NoSidecar);
        let reader = ScxReader::open(&output).unwrap();
        assert!(!reader.header().has_csc(), "the stale sidecar must be gone");
        assert_eq!(reader.header().n_obs, 0);
        assert_eq!(reader.read_obs().unwrap().num_rows(), 0);
        assert_eq!(reader.read_var().unwrap().num_rows(), 2);
        assert_eq!(reader.read_uns().unwrap()["k"], "v", "uns is carried");
    }

    /// `input` and `output` naming the same file through different spellings
    /// is refused outright: the source stays byte-identical.
    #[cfg(unix)]
    #[test]
    fn test_build_csc_refuses_an_aliased_output() {
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
        let err = run_build_csc(&input, &alias, "4G", true, 5000, None, None).unwrap_err();
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
        let err = run_build_csc(&path, &output, "4G", false, 5000, None, None).unwrap_err();
        assert!(err.to_string().contains("no CSR shards"), "{err}");
        assert!(
            !output.exists(),
            "nothing may be written for a malformed input"
        );
        let err =
            crate::rebuild_csc::rebuild_csc_inplace(&path, 5000, "4G", None, None).unwrap_err();
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
        let outcome = run_build_csc(&path, &output, "4G", false, 5000, None, None).unwrap();
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
        let outcome = run_build_csc(&stale, &output2, "4G", false, 5000, None, None).unwrap();
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

        run_build_csc(&input, &output, "4G", false, 3, None, None).unwrap();

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

        run_build_csc(&input, &output, "4G", false, 3, None, None).unwrap();

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
            run_build_csc(&path, &output, "4G", false, 3, None, None).unwrap();

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
