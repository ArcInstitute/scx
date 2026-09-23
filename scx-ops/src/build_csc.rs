// scx build-csc — add a CSC (column-major) sidecar to an SCX file by appending
// it in place.

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::catalog::{FullCatalog, FullCatalogEntry};
use scx_format_io::checksum::blake3_hash;
use scx_format_io::csc_budget;
use scx_format_io::provenance::{Provenance, ProvenanceEntry};
use scx_format_io::section::{write_alignment_padding, SectionType};
use scx_format_io::writer::ScxWriter;
use scx_format_io::FramingConfig;
use scx_format_io::MemoryBudget;
use scx_format_io::ScxReader;

use crate::flock::FileLock;
use crate::in_place::{commit_in_place, prepare_in_place, read_provenance_ops};

type BoxError = Box<dyn std::error::Error>;

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

/// Everything decided before a byte is written: the refusals, and the header
/// pre-pass the builder and the emit need.
///
/// Split out so the copy-out form can run it against the *input* and refuse
/// before paying for a copy of a multi-gigabyte file, and the in-place form can
/// run it again under the lock against the file it will actually append to.
enum Plan<'r> {
    /// 0 rows or 0 columns. Nothing to transpose; `has_csc` says whether there
    /// is a stale sidecar to drop.
    Empty {
        has_csc: bool,
    },
    Build(BuildPlan<'r>),
}

struct BuildPlan<'r> {
    /// CSR shards in row order, by `ShardHeader.global_offset`.
    csr_entries: Vec<&'r FullCatalogEntry>,
    /// Each shard's first row, parallel to `csr_entries`.
    row_starts: Vec<u64>,
    csc_value_encoding: ValueEncoding,
    csc_codec: CodecId,
    /// `Some` iff the file is v4: a sidecar follows the file's layout and
    /// never changes it.
    csc_framing: Option<FramingConfig>,
    max_bytes: usize,
    n_rows: usize,
    n_cols: usize,
}

fn plan<'r>(
    reader: &'r ScxReader,
    path: &Path,
    memory_limit: &str,
    framing: Option<FramingConfig>,
) -> Result<Plan<'r>, BoxError> {
    // Shares the workspace parser so --memory-limit accepts the same forms as
    // `scx convert --memory-budget` (K/M/G/T, KiB/MiB/GiB/TiB; decimals
    // rejected).
    let max_bytes = usize::try_from(MemoryBudget::parse(memory_limit)?)?;
    let header = reader.header();

    // A header that claims rows but carries no CSR shards is malformed, and
    // must be refused before the empty-matrix path can report it as a success.
    // (A 0-row file legitimately has none: the format forbids framed zero-row
    // shards, so that case is exempt.)
    if header.n_obs > 0 && header.n_csr_shards == 0 {
        return Err("Input file has no CSR shards".into());
    }
    let has_csc = reader
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == SectionType::CscShard);
    // An empty matrix without a sidecar is answered before anything else,
    // including the multimodal and framing refusals below: there is nothing to
    // build and nothing to drop, so the op is a no-op (the copy-out form copies
    // the input verbatim) whatever the file's layout. Ahead of the framing
    // refusal on purpose: refusing to frame a sidecar that will not be written
    // would fail a caller — an `--csc` post-pass over an empty output, say — for
    // no benefit.
    let empty_matrix = header.n_obs == 0 || header.n_vars == 0;
    if empty_matrix && !has_csc {
        return Ok(Plan::Empty { has_csc });
    }

    // Not modality-aware — it flattens every CSR shard against the single
    // top-level n_obs × n_vars shape, which would corrupt the sidecar on a
    // multimodal input. The guard lives here so every caller — `scx build-csc`
    // in both its forms, `rebuild_csc_inplace`, and the pyscx wrapper — gets
    // the same actionable message.
    if reader.is_multimodal() {
        return Err(format!(
            "build-csc does not support multimodal files ({} has {} modalities); \
             extract a single modality first with \
             `scx subset {} out.scx --modality NAME`",
            path.display(),
            header.n_modalities,
            path.display(),
        )
        .into());
    }

    // The sidecar is framed iff the file is v4, because an append cannot change
    // the CSR's layout and v4 promises sub-shard random access on every sparse
    // shard. `Some(framing)` on a ≤ v3 file used to mean "re-frame the whole
    // file", which only a rewrite can do — `scx optimize` is that rewrite.
    // `decode_target` and `trial` are cleared rather than trusted: they
    // authorise the encoder to re-select a codec, and the sidecar's codec is
    // `pick_csc_encoding`'s decision.
    let file_v4 = header.format_version >= scx_format_io::CURRENT_FORMAT_VERSION;
    if !file_v4 && framing.is_some() {
        return Err(format!(
            "build-csc cannot row-group-frame a sidecar on a format v{} file: framing is \
             a property of the whole file, and build-csc appends without touching the CSR. \
             Run `scx optimize` first to frame it, then build-csc",
            header.format_version
        )
        .into());
    }
    let csc_framing = file_v4.then(|| FramingConfig {
        trial: false,
        decode_target: None,
        ..framing.unwrap_or_default()
    });

    if empty_matrix {
        return Ok(Plan::Empty { has_csc });
    }

    // Pick a CSC value encoding wide enough to cover EVERY shard. The sidecar
    // transposes all shards into shared columns, so a first-shard-only
    // encoding can truncate later shards (SCX-004): a `Uint8` first shard
    // followed by a `Float32` shard would otherwise encode `1.5` as `1`. The
    // rule is `scx_format_io::pick_csc_encoding`, shared with `ScxWriter`'s
    // finish-time sidecar emit; this loop only collects its inputs. Flooring on
    // each shard's *declared* encoding matters beyond `stats.value_max`: a
    // shard may lack stats (format-permitted), so `value_max` would contribute
    // nothing and a wide integer shard could be under-picked as Uint8.
    //
    // The row start is `ShardHeader.global_offset`, which the format defines as
    // the CSR shard's first row — exact, and present on every shard, where
    // `ShardStats` is format-permitted to be absent.
    let csr_entries = reader.catalog().csr_shards_sorted();
    let mut per_shard: Vec<(CodecId, ValueEncoding, u64)> = Vec::with_capacity(csr_entries.len());
    let mut max_int_val: u32 = 0;
    let mut worst_decoded_bytes: u64 = 0;
    for entry in &csr_entries {
        let sh = reader.read_shard_header(entry)?;
        let ve = ValueEncoding::from_u8(sh.value_encoding).ok_or(
            crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
        )?;
        let ci = CodecId::from_u8(sh.codec_id)
            .ok_or(crate::error::OpsError::UnknownCodec(sh.codec_id))?;
        per_shard.push((ci, ve, sh.global_offset));
        // The exact decoded cost of THIS shard, from the header in hand — both
        // `nnz` and `n_major` are header fields, so a stats-less shard still
        // counts toward the refusal below.
        worst_decoded_bytes = worst_decoded_bytes.max(csc_budget::decoded_csr_shard_bytes(
            sh.nnz,
            sh.n_major as u64,
        ));
        if let Some(stats) = entry.stats.as_ref() {
            max_int_val = max_int_val.max(stats.value_max);
        }
    }
    // Re-order by the header's own offset, not the catalog's.
    //
    // `csr_shards_sorted` keys on `ShardStats::major_start` and sinks a
    // stats-less entry to `u64::MAX`. That is fine when every shard has stats
    // or none does, but the format permits a stats-less entry *beside* a
    // stats-bearing one — and then an early stats-less shard sorts last, the
    // walk sees a non-zero `global_offset` first, and the tiling guard rejects
    // a perfectly valid file. A stable sort, so shards that genuinely share an
    // offset keep catalog order and the guard reports the duplicate.
    let mut order: Vec<usize> = (0..csr_entries.len()).collect();
    order.sort_by_key(|&i| per_shard[i].2);
    let csr_entries: Vec<_> = order.iter().map(|&i| csr_entries[i]).collect();
    let per_shard: Vec<_> = order.iter().map(|&i| per_shard[i]).collect();

    let declared_encs: Vec<ValueEncoding> = per_shard.iter().map(|&(_, ve, _)| ve).collect();
    let (csc_value_encoding, csc_codec) = scx_format_io::pick_csc_encoding(
        &declared_encs,
        max_int_val,
        per_shard.first().map(|&(codec, _, _)| codec),
    )
    .ok_or_else(|| "Input file has no CSR shards".to_string())?;

    // Refuse a budget that cannot admit one decoded source shard.
    //
    // The walk holds exactly one decoded shard at a time, so if the largest
    // does not fit its share no amount of spilling helps, and the op should
    // say so before writing anything. The figure comes from
    // `Share::min_budget_for`, so the refusal and its "raise it to at least N"
    // message cannot drift apart. Both terms of `worst_decoded_bytes` are
    // per-shard header fields: charging the file's `n_obs` for the indptr, or
    // reading `nnz` from optional stats, were both wrong in a first version.
    if worst_decoded_bytes > 0 {
        let need = csc_budget::CSC_BUILD_INPUT_SHARE.min_budget_for(worst_decoded_bytes);
        if (max_bytes as u64) < need {
            return Err(format!(
                "build-csc: --memory-limit {memory_limit} is too small for {}: decoding its \
                 largest CSR shard needs ~{worst_decoded_bytes} bytes; raise --memory-limit \
                 to at least {need}",
                path.display()
            )
            .into());
        }
    }

    Ok(Plan::Build(BuildPlan {
        csr_entries,
        row_starts: per_shard.iter().map(|&(_, _, r)| r).collect(),
        csc_value_encoding,
        csc_codec,
        csc_framing,
        max_bytes,
        n_rows: header.n_obs as usize,
        n_cols: header.n_vars as usize,
    }))
}

/// Run `f`, and if it fails, cut the file back to `len`.
///
/// For the stretch of an in-place op between its first append and
/// `commit_in_place`: nothing references those bytes until the header write
/// repoints the catalog, so dropping them is safe, and a failed census-scale
/// build would otherwise leave a gigabyte of orphan tail that only `scx
/// compact` could reclaim.
fn truncate_on_error<T>(
    lock: &mut FileLock,
    len: u64,
    f: impl FnOnce(&mut FileLock) -> Result<T, BoxError>,
) -> Result<T, BoxError> {
    let result = f(lock);
    if result.is_err() {
        // The original error is the one worth reporting; a failed truncation
        // leaves orphan bytes, which is exactly the pre-existing behaviour of
        // every other in-place op.
        if let Err(e) = lock.file().set_len(len) {
            log::warn!("build-csc: could not truncate the failed append back to {len} bytes: {e}");
        }
    }
    result
}

/// Add a CSC sidecar to `path` in place — `scx build-csc` with no `<OUTPUT>`,
/// and `scx append --rebuild-csc`. The CSC shards are appended at EOF
/// and the catalog repointed with `prepare_in_place` / `commit_in_place`, the
/// harness `append` and the attach ops use.
///
/// Nothing else in the file moves. Every CSR shard, obs/var section, index and
/// bitmap keeps its bytes at its offset — [`crate::carry::audit_in_place`]
/// checks that before committing — so `data_generation` is unchanged and
/// `scx rollback` undoes the build. A sidecar already on the file is replaced;
/// its bytes become orphans that `scx compact` reclaims (carrying a fresh
/// sidecar), as with
/// every in-place op. A *layer* sidecar is kept when it is fresh and dropped
/// with a warning when it is not, because the generation stamp written here is
/// the one freshness field every column-major section shares.
///
/// `framing`: `None` or `Some` on a v4 file (the sidecar is framed with it, or
/// with the default when `None`); must be `None` on a ≤ v3 file, which an
/// append cannot frame. [`crate::framing_for_csc_rebuild`] returns exactly the
/// admissible value, and the streaming convert path passes its own so a custom
/// `--row-group-rows` reaches the sidecar.
///
/// `temp_dir`: root for the CSC builder's column-bucket spill files. `None`
/// uses the file's own directory — see `TempDirSpillStore` for why that,
/// rather than the platform temp dir the other spilling ops default to.
pub fn rebuild_csc_inplace(
    path: &Path,
    csc_cols_per_shard: usize,
    memory_limit: &str,
    framing: Option<FramingConfig>,
    temp_dir: Option<&Path>,
) -> Result<BuildCscOutcome, BoxError> {
    if !path.exists() {
        return Err(format!("input file does not exist: {}", path.display()).into());
    }
    let (mut lock, mut prep) = prepare_in_place(path, 0)?;
    // Opened after the lock, as the attach ops do. The reader takes no lock of
    // its own, and this op only ever appends, so its mapping stays valid.
    let reader = ScxReader::open(path)?;

    let plan = plan(&reader, path, memory_limit, framing)?;
    let (build, had_csc) = match plan {
        Plan::Empty { has_csc: false } => {
            log::info!(
                "build-csc: {} is an empty matrix ({} x {}); no CSC sidecar to build",
                path.display(),
                prep.header.n_obs,
                prep.header.n_vars
            );
            return Ok(BuildCscOutcome::NoSidecar);
        }
        Plan::Empty { has_csc: true } => {
            log::info!(
                "build-csc: {} is an empty matrix; dropping its stale CSC sidecar",
                path.display()
            );
            (None, true)
        }
        Plan::Build(b) => {
            let had = prep
                .old_catalog
                .entries
                .iter()
                .any(|e| e.section_type == SectionType::CscShard);
            (Some(b), had)
        }
    };

    let params_json = format!(
        "{{\"memory_limit\":\"{memory_limit}\",\"csc_cols_per_shard\":{csc_cols_per_shard},\
          \"temp_dir\":{}}}",
        temp_dir.map_or_else(
            || "null".to_string(),
            |p| format!("{:?}", p.display().to_string())
        )
    );

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let original_len = lock.seek(SeekFrom::End(0))?;
    let old_catalog_offset = prep.old_catalog_offset;
    let spill_root = temp_dir
        .map(Path::to_path_buf)
        .or_else(|| path.parent().map(Path::to_path_buf));

    let new_catalog = truncate_on_error(&mut lock, original_len, |lock| {
        let csc_entries = match &build {
            None => Vec::new(),
            Some(b) => append_csc_shards(
                lock,
                &prep,
                &reader,
                b,
                csc_cols_per_shard,
                spill_root.as_deref(),
                &pb,
            )?,
        };

        // Provenance: the existing chain plus one entry, written as a fresh
        // section (the old one is dropped from the catalog below).
        let mut prov_ops = read_provenance_ops(lock, &prep.old_catalog)?;
        prov_ops.push(ProvenanceEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            action: "build-csc".to_string(),
            tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
            params_json: params_json.clone(),
            input_checksums: vec![],
        });
        let mut prov_bytes = Vec::new();
        Provenance {
            version: 1,
            operations: prov_ops,
        }
        .write_to(&mut prov_bytes)?;
        let mut woff = lock.seek(SeekFrom::End(0))?;
        woff += write_alignment_padding(&mut *lock, woff)? as u64;
        lock.write_all(&prov_bytes)?;

        // Old entries minus the sidecar being replaced and the provenance being
        // superseded; then the new sidecar and provenance. The adopted writer
        // names CSC shards from `X_csc_shard_0`, which is sound only because
        // every old CSC entry is dropped here — its duplicate guard sees only
        // its own entries.
        let orphaned: u64 = prep
            .old_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard)
            .map(|e| e.length)
            .sum();
        if orphaned > 0 && build.is_some() {
            log::warn!(
                "build-csc: replacing the existing CSC sidecar on {} leaves {orphaned} bytes \
                 unreferenced; `scx compact` reclaims them",
                path.display()
            );
        }
        // A layer sidecar shares the one freshness stamp with X's, and this op
        // re-stamps it below. One that was already stale (built before an op
        // that bumped `data_generation` and did not drop it) would be blessed by
        // that stamp, so it is dropped instead — nothing here can rebuild it.
        // A fresh one is carried untouched.
        let old_fresh = prep.old_catalog.csc_build_generation == prep.old_catalog.data_generation;
        let stale_layer_csc = !old_fresh
            && prep
                .old_catalog
                .entries
                .iter()
                .any(|e| e.section_type == SectionType::LayerCscShard);
        if stale_layer_csc {
            let bytes: u64 = prep
                .old_catalog
                .entries
                .iter()
                .filter(|e| e.section_type == SectionType::LayerCscShard)
                .map(|e| e.length)
                .sum();
            log::warn!(
                "build-csc: dropping the stale layer CSC sidecar on {} (built at generation \
                 {}, file is at {}); nothing rebuilds a layer sidecar, and its {bytes} bytes \
                 stay unreferenced until `scx compact`",
                path.display(),
                prep.old_catalog.csc_build_generation,
                prep.old_catalog.data_generation
            );
        }
        let mut entries: Vec<FullCatalogEntry> = prep
            .old_catalog
            .entries
            .iter()
            .filter(|e| {
                e.section_type != SectionType::CscShard
                    && e.section_type != SectionType::Provenance
                    && !(stale_layer_csc && e.section_type == SectionType::LayerCscShard)
            })
            .cloned()
            .collect();
        entries.extend(csc_entries);
        entries.push(FullCatalogEntry {
            name: "provenance".to_string(),
            offset: woff,
            length: prov_bytes.len() as u64,
            section_type: SectionType::Provenance,
            checksum: blake3_hash(&prov_bytes),
            modality_id: 0,
            stats: None,
        });

        let has_sidecar = entries
            .iter()
            .any(|e| e.section_type == SectionType::CscShard);
        let new_catalog = FullCatalog {
            catalog_version: scx_format_io::CURRENT_CATALOG_VERSION,
            manifest_sequence: prep.header.manifest_sequence + 1,
            // What makes `scx rollback` undo the build.
            prev_catalog_offset: old_catalog_offset,
            n_obs: prep.old_n_obs,
            entries,
            // X is untouched, so its generation is too — and the sidecar is
            // built against exactly that generation, which is what the reader's
            // staleness guard checks. With no X sidecar written (an empty
            // matrix dropping a stale one) the field is left as it was: it is
            // also what keeps a carried *layer* sidecar fresh, and zeroing it
            // would stale that for nothing.
            data_generation: prep.old_catalog.data_generation,
            csc_build_generation: if has_sidecar {
                prep.old_catalog.data_generation
            } else {
                prep.old_catalog.csc_build_generation
            },
        };

        // Checked before the commit, so a violation leaves the file on its old
        // catalog (and `truncate_on_error` drops the appended bytes).
        crate::carry::audit_in_place(
            crate::carry::RewriteOp::BuildCsc,
            &prep.old_catalog,
            &new_catalog,
        )?;
        Ok(new_catalog)
    })?;

    // Header: the sidecar counters, set explicitly as `append` does — not
    // `sync_from_catalog`, which would re-derive `nnz` from optional stats.
    let n_csc_shards = new_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .count() as u32;
    prep.header.n_csc_shards = n_csc_shards;
    if n_csc_shards > 0 {
        prep.header.set_csc();
    } else {
        prep.header.clear_csc();
    }
    let (mt_off, mt_len) = (
        prep.header.modality_table_offset,
        prep.header.modality_table_length,
    );
    commit_in_place(&mut lock, &mut prep.header, &new_catalog, mt_off, mt_len)?;
    pb.finish_and_clear();

    if n_csc_shards == 0 {
        if had_csc {
            log::info!(
                "build-csc: dropped the stale CSC sidecar from {}",
                path.display()
            );
        }
        return Ok(BuildCscOutcome::NoSidecar);
    }
    log::info!(
        "build-csc: appended {n_csc_shards} CSC shard(s) to {}",
        path.display()
    );
    Ok(BuildCscOutcome::Built)
}

/// Decode each CSR shard once, route it into the builder, then emit the CSC
/// shards at EOF through an adopted writer. Returns their catalog entries.
///
/// No CSR shard is written. The rewrite this replaced decoded every shard to
/// re-encode it — under `auto`, the framed encoder's two candidates per shard —
/// to produce bytes that were, by the op's own contract, the input's.
fn append_csc_shards(
    lock: &mut FileLock,
    prep: &crate::in_place::InPlacePrep,
    reader: &ScxReader,
    b: &BuildPlan<'_>,
    csc_cols_per_shard: usize,
    spill_root: Option<&Path>,
    pb: &ProgressBar,
) -> Result<Vec<FullCatalogEntry>, BoxError> {
    let store = scx_format_io::TempDirSpillStore::new(spill_root)?;
    let mut builder = scx_sparse::CscBuilder::new(
        b.n_rows,
        b.n_cols,
        scx_sparse::CscBuilderConfig {
            cols_per_shard: csc_cols_per_shard,
            memory_bytes: b.max_bytes,
            spill_after_bytes: csc_budget::CSC_BUILD_BUCKET_SHARE.of(b.max_bytes as u64) as usize,
            ..Default::default()
        },
        Box::new(store),
    )?;

    let mut rows_pushed: u64 = 0;
    for (i, (shard_entry, &shard_row_start)) in b.csr_entries.iter().zip(&b.row_starts).enumerate()
    {
        // The row axis this shard claims, checked against the walk's own
        // running count before the shard is decoded. A catalog whose shards do
        // not tile `[0, n_obs)` in order is an error rather than a silently
        // shifted sidecar. (`row_starts` was built from the same list, so the
        // `zip` cannot truncate.)
        if shard_row_start != rows_pushed {
            return Err(format!(
                "build-csc: CSR shard {i} declares row_start {shard_row_start}, but \
                 {rows_pushed} rows precede it; the shards do not tile [0, {}) in order",
                b.n_rows
            )
            .into());
        }
        let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
        let n_shard_rows = indptr.len() - 1;
        let shard = scx_sparse::ScxCsr::new((n_shard_rows, b.n_cols), indptr, indices, data)?;
        builder.push_shard(rows_pushed, &shard)?;
        rows_pushed += n_shard_rows as u64;
        pb.set_message(format!(
            "shard {}/{} routed into the CSC builder",
            i + 1,
            b.csr_entries.len()
        ));
    }

    let mut emitter = builder.finish()?;
    let n_planned = emitter.plan().len();

    let write_offset = lock.seek(SeekFrom::End(0))?;
    let mut writer = ScxWriter::adopt_in_place(
        lock.file().try_clone()?,
        prep.header.clone(),
        write_offset,
        Vec::new(),
    )?;
    // Framed iff the file is v4 (decided in `plan`). `adopt_in_place` defaults
    // framing to `None`, which would put unframed shards in a v4 file.
    writer.set_framing(b.csc_framing);
    let emit_opts = scx_format_io::CscEmitOptions {
        value_encoding: b.csc_value_encoding,
        codec_id: b.csc_codec,
        modality_id: None,
    };
    let stats = scx_format_io::emit_csc_shards(
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
    let (file, new_offset, entries) = writer.into_in_place_parts()?;
    drop(file);
    lock.seek(SeekFrom::Start(new_offset))?;

    if stats.spill_bytes > 0 {
        log::info!(
            "build-csc: spilled {} bytes of column buckets under a {}-byte budget",
            stats.spill_bytes,
            b.max_bytes
        );
    }
    if let Some(col) = stats.first_non_strict_column {
        log::warn!(
            "build-csc: column {col} has a duplicate (row, col) in the source, so its CSC rows \
             are not strictly increasing; GPU routes validating `sorted` will reject this sidecar"
        );
    }
    Ok(entries)
}

/// Write a copy of `input` carrying a CSC sidecar to `output`, leaving `input`
/// untouched.
///
/// The copy is `input`'s bytes verbatim followed by the in-place append of
/// [`rebuild_csc_inplace`], staged beside `output` and renamed over it. So the
/// output's previous catalog is the input's, and `scx rollback` on the output
/// yields the input. Every refusal is checked against `input` first, so a
/// refused build costs no copy.
///
/// `framing` and `temp_dir` are as for [`rebuild_csc_inplace`]; `temp_dir`
/// defaults to `output`'s directory.
pub fn run_build_csc(
    input: &Path,
    output: &Path,
    memory_limit: &str,
    force: bool,
    csc_cols_per_shard: usize,
    framing: Option<FramingConfig>,
    temp_dir: Option<&Path>,
) -> Result<BuildCscOutcome, BoxError> {
    if !input.exists() {
        return Err(format!("input file does not exist: {}", input.display()).into());
    }
    // `output` must be a different file from `input`, by canonical path. The
    // in-place form (no `<OUTPUT>`) is `rebuild_csc_inplace`.
    if same_file(input, output) {
        return Err(format!(
            "input and output must be different files ({} names the input); omit <OUTPUT> \
             to add the CSC sidecar in place, or give a distinct output path",
            output.display()
        )
        .into());
    }
    // `symlink_metadata`, not `exists()`: the latter follows the link and
    // answers `false` for a dangling symlink, which would let an unforced
    // write replace it.
    if std::fs::symlink_metadata(output).is_ok() && !force {
        return Err(format!(
            "{} already exists (use --force to overwrite)",
            output.display()
        )
        .into());
    }

    // Refuse against the input before copying it.
    let empty_without_sidecar = {
        let reader = ScxReader::open(input)?;
        matches!(
            plan(&reader, input, memory_limit, framing)?,
            Plan::Empty { has_csc: false }
        )
    };

    // A sibling temp file, so the rename is atomic and a failure leaves no
    // partial output (the `TempPath` deletes itself on drop) — for the
    // empty-matrix copy too, which used to write `output` directly and so could
    // leave half a file, or write through a symlink rather than replacing it.
    // No `remove_file` of an existing `output` first: the rename replaces it,
    // and unlinking would only widen a window in which neither exists.
    let (file, staging) = scx_format_io::make_sibling_tempfile(output)?;
    drop(file);
    std::fs::copy(input, &staging)?;
    let outcome = if empty_without_sidecar {
        log::info!(
            "build-csc: {} is an empty matrix; no CSC sidecar to build",
            input.display()
        );
        BuildCscOutcome::NoSidecar
    } else {
        // `temp_dir: None` spills beside the staging file, i.e. in `output`'s
        // directory.
        rebuild_csc_inplace(
            &staging,
            csc_cols_per_shard,
            memory_limit,
            framing,
            temp_dir,
        )?
    };
    staging.persist(output)?;
    // `std::fs::copy` carried the input's mode onto the staging file; every
    // other writer leaves its output at `0o666 & !umask`, so this does too.
    scx_format_io::chmod_to_umask(output)?;

    // Best-effort: the output is already persisted, so failing to reopen it
    // for the summary must not turn a finished build into an error.
    if let (BuildCscOutcome::Built, Ok(r)) = (outcome, ScxReader::open(output)) {
        let csc: Vec<_> = r
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard)
            .collect();
        let nnz: u64 = csc
            .iter()
            .filter_map(|e| e.stats.as_ref().map(|s| s.nnz))
            .sum();
        println!(
            "Built CSC: {} → {} ({} rows × {} cols, {} nnz, {} CSC shard{})",
            input.display(),
            output.display(),
            r.header().n_obs,
            r.header().n_vars,
            nnz,
            csc.len(),
            if csc.len() == 1 { "" } else { "s" },
        );
    }
    Ok(outcome)
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

    /// Clear `ShardStats` from the CSR shard entries whose position **in
    /// catalog order among CSR shards** is listed in `drop_at`, by appending a
    /// rewritten full catalog and repointing the header at it.
    ///
    /// No in-tree writer emits a `stats: None` CSR entry — `write_shard_inner`
    /// and every sibling always attach `Some(stats)` — but the format permits
    /// one, and `csr_shards_sorted` sinks it to `u64::MAX`. So the mixed
    /// `Some`/`None` catalog that reorders has to be constructed rather than
    /// written.
    ///
    /// Only the catalog moves: every section body, and therefore every
    /// per-entry checksum, is untouched. `header.file_checksum` does go stale,
    /// which is harmless here because `ScxReader::open` does not verify it
    /// (`verify_file_checksum` is opt-in, for `scx validate`).
    fn strip_csr_stats(path: &std::path::Path, drop_at: &[usize]) {
        let (mut catalog, mut header) = {
            let r = ScxReader::open(path).unwrap();
            (r.catalog().clone(), r.header().clone())
        };
        let mut seen = 0usize;
        for e in catalog.entries.iter_mut() {
            if e.section_type == scx_format::SectionType::CsrShard {
                if drop_at.contains(&seen) {
                    e.stats = None;
                }
                seen += 1;
            }
        }
        assert!(
            drop_at.iter().all(|&i| i < seen),
            "strip_csr_stats: asked to clear shard {drop_at:?} of {seen}"
        );

        let mut catalog_buf = Vec::new();
        catalog.write_to(&mut catalog_buf).unwrap();

        let mut bytes = std::fs::read(path).unwrap();
        while !bytes.len().is_multiple_of(8) {
            bytes.push(0);
        }
        header.full_catalog_offset = bytes.len() as u64;
        header.full_catalog_length = catalog_buf.len() as u64;
        bytes.extend_from_slice(&catalog_buf);

        let mut header_buf = Vec::new();
        header.write_to(&mut header_buf).unwrap();
        bytes[..header_buf.len()].copy_from_slice(&header_buf);
        std::fs::write(path, &bytes).unwrap();
    }

    /// A file whose **first** CSR shard has no `ShardStats` and whose later
    /// shards do must build a sidecar, and the same one as if every shard
    /// carried stats.
    ///
    /// `csr_shards_sorted` keys on `ShardStats::major_start` and sinks a
    /// stats-less entry to `u64::MAX`. All-stats and no-stats files are both
    /// fine — the first sorts correctly, the second keeps catalog order — but a
    /// mixed catalog sorts the stats-less shard *last*, so the walk meets
    /// `global_offset = rows_per` with `rows_pushed = 0` and the tiling guard
    /// rejects a valid file. `ShardHeader.global_offset` is present on every
    /// shard, so the walk re-orders on that instead.
    ///
    /// Reported by review (codex - gpt-5.6-sol), which also asked for this
    /// case specifically: the all-statsless coverage in
    /// `scx-ops/tests/column_stats_staleness.rs` cannot see it, because with
    /// no stats anywhere the sort is a no-op.
    #[test]
    fn a_stats_less_shard_beside_a_stats_bearing_one_still_tiles() {
        let dir = tempfile::tempdir().unwrap();

        // Control: the same fixture with stats on every shard.
        let pristine = write_test_input_multi_shard(&dir, 40, 6, 4);
        let control = dir.path().join("control.scx");
        run_build_csc(&pristine, &control, "4G", false, 2, None, None).unwrap();

        // The case: shard 0 — the one that must sort FIRST — loses its stats.
        let mixed = dir.path().join("mixed.scx");
        std::fs::copy(&pristine, &mixed).unwrap();
        strip_csr_stats(&mixed, &[0]);

        // Premise: the catalog really is mixed, and `csr_shards_sorted` really
        // does mis-order it. Without this the test could pass because the
        // strip silently did nothing.
        {
            let r = ScxReader::open(&mixed).unwrap();
            let entries = r.catalog().csr_shards_sorted();
            assert_eq!(entries.len(), 4);
            assert!(
                entries.iter().filter(|e| e.stats.is_none()).count() == 1
                    && entries.iter().filter(|e| e.stats.is_some()).count() == 3,
                "setup: exactly one stats-less entry beside three stats-bearing ones"
            );
            assert!(
                entries.last().unwrap().stats.is_none(),
                "setup: the stats-less shard must sort LAST, or there is nothing to re-order"
            );
        }

        let out = dir.path().join("mixed_csc.scx");
        run_build_csc(&mixed, &out, "4G", false, 2, None, None)
            .expect("a stats-less shard beside a stats-bearing one is a valid file");

        // Same sidecar, not merely some sidecar: a re-order that got the
        // permutation wrong would still tile and still produce CSC shards.
        let got = ScxReader::open(&out).unwrap();
        let want = ScxReader::open(&control).unwrap();
        assert_eq!(got.header().n_csc_shards, want.header().n_csc_shards);
        let got_csc = got.read_all_csc_shards().unwrap();
        let want_csc = want.read_all_csc_shards().unwrap();
        assert_eq!(got_csc.shape, want_csc.shape);
        assert_eq!(got_csc.indptr, want_csc.indptr);
        assert_eq!(got_csc.indices, want_csc.indices);
        assert_eq!(got_csc.data, want_csc.data);

        // And the CSR is the input's own entries, untouched — stats-less one
        // included. (While build-csc re-encoded the CSR, this decoded it back
        // and compared; the append never writes a CSR entry, so the entries
        // themselves are the stronger check, and `read_all_csr_shards` refuses
        // a stats-less entry anyway.)
        let csr = |r: &ScxReader| -> Vec<(String, u64, u64, [u8; 32], bool)> {
            r.catalog()
                .entries
                .iter()
                .filter(|e| e.section_type == SectionType::CsrShard)
                .map(|e| {
                    (
                        e.name.clone(),
                        e.offset,
                        e.length,
                        e.checksum,
                        e.stats.is_some(),
                    )
                })
                .collect()
        };
        assert_eq!(csr(&got), csr(&ScxReader::open(&mixed).unwrap()));
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
    /// It is the documented way to restore a sidecar another op dropped, which
    /// puts `mark_deleted` → `build-csc` directly on the happy path. (While the
    /// in-place form renamed a wholly new file over the target, a deletion it
    /// dropped was unrecoverable; it is an append now, and rollback-able, but a
    /// silently un-deleted cell would still be wrong.)
    #[test]
    fn build_csc_carries_deletion_vectors() {
        for in_place in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let input = write_test_input(&dir, 6, 4);
            crate::mark_deleted(&input, &[1, 4]).unwrap();

            let read_path = if in_place {
                crate::rebuild_csc_inplace(&input, 5000, "4G", None, None).unwrap();
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
    /// And every CSR section must still precede every CSC section: the append
    /// leaves the CSR entries where they were and writes the CSC sidecar after
    /// the file's old end, so a CSC section before a CSR one would mean
    /// something rewrote CSR.
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
    /// `build-csc` is a no-op that still produces the requested output — its
    /// callers (`scx append --rebuild-csc`, MTX `convert --csc`) run it on
    /// whatever they produced and must not fail on one that yielded zero rows. Distinct
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
        let err = crate::rebuild_csc_inplace(&path, 5000, "4G", None, None).unwrap_err();
        assert!(err.to_string().contains("no CSR shards"), "{err}");
    }

    /// A 0-column matrix with rows and no sidecar is copied verbatim (CSR
    /// shards included); with a stale sidecar, the append drops the old CSC
    /// entries and writes none, leaving its CSR shards untouched.
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

#[cfg(test)]
#[path = "build_csc_in_place_tests.rs"]
mod in_place_tests;
