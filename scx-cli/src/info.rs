use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;

use crate::format::human_size;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::catalog::FullCatalog;
use scx_format_io::deletion_vectors::DeletionVectors;
use scx_format_io::header::FileHeader;
use scx_format_io::modality::{ModalityTable, ModalityType};
use scx_format_io::provenance::Provenance;
use scx_format_io::reader::ScxReader;
use scx_format_io::section::SectionType;

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

/// One row of `--history` manifest output.
struct ManifestHistoryEntry {
    manifest_sequence: u64,
    n_obs: u64,
    action: String,
    timestamp: String,
}

/// Reader-agnostic snapshot of everything `scx info` renders.
///
/// Both the local (`ScxReader`) and cloud (`CloudReader`) collectors
/// populate the same owned model, and both renderers consume it — so the
/// text/JSON output cannot diverge between a local file and a cloud URL by
/// construction.
struct InfoModel {
    header: FileHeader,
    catalog: FullCatalog,
    /// Distinct codec ids across all CSR shards, sorted ascending. Formatted
    /// into the per-shard codec summary via the shared `codec_id_name` logic.
    distinct_codec_ids: Vec<u8>,
    /// Distinct value-encoding bytes across all CSR shards, sorted ascending.
    distinct_value_encodings: Vec<u8>,
    file_size: u64,
    /// Orphaned (non-live) bytes, or `None` when the concept doesn't apply
    /// (an exploded `.scxd/` directory has no single file and no orphans).
    orphaned: Option<u64>,
    modality_table: Option<ModalityTable>,
    deletion_vectors: Option<DeletionVectors>,
    provenance: Option<Provenance>,
    /// Manifest history rows, populated only for local `--history` (the
    /// chain walk needs packed-file byte offsets that cloud inputs lack).
    history: Option<Vec<ManifestHistoryEntry>>,
}

/// Live byte region: header + root-catalog placeholder, the active full
/// catalog, and every live section.
fn live_bytes(header: &FileHeader, catalog: &FullCatalog) -> u64 {
    scx_format_io::SECTIONS_START_OFFSET
        + header.full_catalog_length
        + catalog.entries.iter().map(|e| e.length).sum::<u64>()
}

/// Format a distinct, sorted set of shard-header bytes into the summary
/// string `scx info` prints: the single name when uniform, `mixed (a, b)`
/// when shards differ, or `n/a` when there are no shards.
fn format_distinct(bytes: &[u8], name: impl Fn(u8) -> &'static str) -> String {
    match bytes {
        [] => "n/a".to_string(),
        [one] => name(*one).to_string(),
        many => format!(
            "mixed ({})",
            many.iter().map(|&b| name(b)).collect::<Vec<_>>().join(", ")
        ),
    }
}

pub fn run_info(source: &str, json_output: bool, history: bool) -> CliResult<()> {
    if crate::cloud_url::is_cloud_url(source) {
        return run_info_cloud(source, json_output, history);
    }
    let path = Path::new(source);
    let reader = ScxReader::open_unchecked(path)?;
    let model = collect_local(&reader, path, history)?;
    if json_output {
        render_json(&model)
    } else {
        render_text(&model)
    }
}

#[cfg(feature = "cloud")]
fn run_info_cloud(source: &str, json_output: bool, history: bool) -> CliResult<()> {
    if history {
        eprintln!(
            "note: --history is local-only; manifest history is not available for \
             cloud URLs or exploded .scxd/ directories"
        );
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    // One runtime entry: open and collect in the same async block so the
    // reader stays scoped to it and we don't pay the block_on round-trip twice.
    let model = rt.block_on(async {
        let reader = scx_cloud::open_cloud(source).await?;
        collect_cloud(&reader).await
    })?;
    if json_output {
        render_json(&model)
    } else {
        render_text(&model)
    }
}

#[cfg(not(feature = "cloud"))]
fn run_info_cloud(source: &str, _json_output: bool, _history: bool) -> CliResult<()> {
    Err(format!(
        "'{source}' looks like a cloud URL or exploded .scxd/ directory; \
         `scx info` on these requires scx-cli built with --features cloud"
    )
    .into())
}

/// Gather everything `scx info` renders from a local mmap reader.
fn collect_local(reader: &ScxReader, path: &Path, history: bool) -> CliResult<InfoModel> {
    let header = reader.header().clone();
    let catalog = reader.catalog().clone();

    // Codec / value encoding are chosen per shard (auto-routing mixes
    // Scx1/Zstd; a subset can mix uint16/uint32), so summarize across ALL CSR
    // shards rather than trusting the file-level header default.
    let csr_shards = catalog.shards(SectionType::CsrShard);
    let (distinct_codec_ids, distinct_value_encodings) = if csr_shards.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        (
            crate::shard_utils::distinct_sorted_shard_field(reader, &csr_shards, |h| h.codec_id)?,
            crate::shard_utils::distinct_sorted_shard_field(reader, &csr_shards, |h| {
                h.value_encoding
            })?,
        )
    };

    let file_size = std::fs::metadata(path)?.len();
    let orphaned = Some(file_size.saturating_sub(live_bytes(&header, &catalog)));

    let history = if history {
        Some(collect_history_local(reader)?)
    } else {
        None
    };

    Ok(InfoModel {
        modality_table: reader.modality_table().cloned(),
        deletion_vectors: reader.read_deletion_vectors().ok().flatten(),
        provenance: reader.read_provenance().ok(),
        header,
        catalog,
        distinct_codec_ids,
        distinct_value_encodings,
        file_size,
        orphaned,
        history,
    })
}

/// Gather everything `scx info` renders from a cloud reader. No degradation
/// vs. the local path: file size comes from a `HEAD` (packed) or summed live
/// bytes (exploded), and the codec/value-encoding summaries range-read each
/// CSR shard's 76-byte header.
#[cfg(feature = "cloud")]
async fn collect_cloud(reader: &scx_cloud::CloudReader) -> CliResult<InfoModel> {
    let header = reader.header().clone();
    let catalog = reader.catalog().clone();
    let (distinct_codec_ids, distinct_value_encodings) = reader.csr_shard_field_summaries().await?;
    let file_size = reader.object_size().await?;
    // Exploded directories have no single file and therefore no orphaned
    // bytes; packed remote files do, computed exactly as the local path.
    let orphaned = if reader.is_exploded() {
        None
    } else {
        Some(file_size.saturating_sub(live_bytes(&header, &catalog)))
    };

    // Soft sections degrade the same way as the local path (`collect_local`):
    // a malformed modality-table / deletion-vector / provenance section drops
    // to `None` rather than aborting the whole report.
    Ok(InfoModel {
        modality_table: reader.modality_table().await.ok().flatten(),
        deletion_vectors: reader.read_deletion_vectors().await.ok().flatten(),
        provenance: reader.read_provenance().await.ok().flatten(),
        header,
        catalog,
        distinct_codec_ids,
        distinct_value_encodings,
        file_size,
        orphaned,
        history: None,
    })
}

/// Render the human-readable `scx info` report.
fn render_text(model: &InfoModel) -> CliResult<()> {
    let header = &model.header;
    let catalog = &model.catalog;

    // Line 1: overview
    println!(
        "SCX v{} | {} cells x {} genes | {} nnz",
        header.format_version,
        fmt_num(header.n_obs),
        fmt_num(header.n_vars),
        fmt_num(header.nnz),
    );

    let codec_name = format_distinct(&model.distinct_codec_ids, codec_id_name);

    let index_dtype = match header.index_dtype {
        0 => "u16",
        1 => "u32",
        _ => "unknown",
    };

    // Line 2: shards/codec/index. Suppress the CSC count when zero so
    // the line stays clean for files that ship without a sidecar.
    if header.n_csc_shards > 0 {
        println!(
            "Shards: {} CSR | {} CSC | Codec: {} | Index dtype: {}",
            header.n_csr_shards, header.n_csc_shards, codec_name, index_dtype,
        );
    } else {
        println!(
            "Shards: {} CSR | Codec: {} | Index dtype: {}",
            header.n_csr_shards, codec_name, index_dtype,
        );
    }

    let value_enc_name = format_distinct(&model.distinct_value_encodings, value_encoding_name);

    // Line 3: value encoding and shard target
    println!(
        "Value encoding: {} | Shard target: {} rows",
        value_enc_name,
        fmt_num(header.shard_target_rows as u64),
    );

    // Line 4: manifest sequence
    println!("Manifest: sequence {}", header.manifest_sequence);

    // Line 5: file size. For an exploded directory there is no single file —
    // this is the summed live bytes (no padding / per-object overhead), so
    // label it to avoid a mismatch against `du` / `gsutil du`. (`orphaned` is
    // `None` exactly for exploded inputs.)
    if model.orphaned.is_none() {
        println!("File size: {} (live bytes)", human_size(model.file_size));
    } else {
        println!("File size: {}", human_size(model.file_size));
    }

    // Line 6: orphaned bytes — file size minus live regions. Repeated
    // in-place edits (`scx set-uns` / `modify-metadata`, `append`) leave
    // superseded sections behind until `scx compact` reclaims them. Suppressed
    // for exploded directories, which have no single file and no orphans.
    if let Some(orphaned) = model.orphaned {
        if orphaned > 0 {
            println!(
                "Orphaned bytes: ~{} (run 'scx compact' to reclaim)",
                human_size(orphaned)
            );
        }
    }

    // Modality table (Phase F.1): when n_modalities > 0, print one
    // row per modality. Single-modality files skip this block to
    // keep their summary unchanged.
    if let Some(table) = &model.modality_table {
        if !table.entries.is_empty() {
            println!();
            println!(
                "Modalities ({}):  {:<14} {:<10} {:>10} {:>14} {:>6} {:>6} {:<6} {:<10}",
                table.entries.len(),
                "name",
                "type",
                "n_vars",
                "nnz",
                "csr",
                "csc",
                "has_csc",
                "codec",
            );
            for info in &table.entries {
                let type_name = modality_type_name(info.modality_type);
                let codec_name = codec_id_name(info.default_codec_id);
                let has_csc = if info.flags.has_csc() { "yes" } else { "no" };
                println!(
                    "                  {:<14} {:<10} {:>10} {:>14} {:>6} {:>6} {:<6} {:<10}",
                    info.name,
                    type_name,
                    fmt_num(info.n_vars),
                    fmt_num(info.nnz),
                    info.n_csr_shards,
                    info.n_csc_shards,
                    has_csc,
                    codec_name,
                );
            }
        }
    }

    // Sections table
    println!();
    println!("Sections:");
    let mut groups: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for entry in &catalog.entries {
        let label = section_label(&entry.name, &entry.section_type);
        let g = groups.entry(label).or_insert((0, 0));
        g.0 += 1;
        g.1 += entry.length;
    }
    for (label, (count, size)) in &groups {
        if *count > 1 {
            println!(
                "  {:30} {:>4} sections  {}",
                label,
                count,
                human_size(*size)
            );
        } else {
            println!("  {:30} {}", label, human_size(*size));
        }
    }

    // Flags
    let mut flags = Vec::new();
    if header.has_csc() {
        flags.push("csc");
    }
    if header.has_bitmap() {
        flags.push("bitmap");
    }
    if header.has_obsm() {
        flags.push("obsm");
    }
    if header.has_obsp() {
        flags.push("obsp");
    }
    if header.has_deletion_vectors() {
        flags.push("deletion_vectors");
    }
    if !flags.is_empty() {
        println!();
        println!("Flags: {}", flags.join(", "));
    }

    // CSC sidecar layout: when more than one CSC shard is present,
    // print the per-shard column range + nnz so users can see the
    // sharding granularity. One-shard files leave it implicit — the
    // count on the Shards line says it all.
    if header.n_csc_shards > 1 {
        println!();
        println!(
            "CSC layout ({} shards, {} cols/shard avg):",
            header.n_csc_shards,
            header.n_vars / header.n_csc_shards as u64,
        );
        for (i, entry) in catalog.csc_shards_sorted().iter().enumerate() {
            let range = entry.stats.as_ref().map(|s| s.col_range());
            let nnz = entry.stats.as_ref().map(|s| s.nnz).unwrap_or(0);
            match range {
                Some(r) => println!(
                    "  shard {:>3}: cols {}..{} ({} cols), nnz {}",
                    i,
                    fmt_num(r.start),
                    fmt_num(r.end),
                    fmt_num(r.end - r.start),
                    fmt_num(nnz),
                ),
                None => println!("  shard {:>3}: (no stats)", i),
            }
        }
    }

    // Deletion vector detail
    if let Some(dv) = &model.deletion_vectors {
        let n_shards_with_dv = dv.shards.len();
        let total_deleted = dv.total_deleted();
        println!(
            "Deletion vectors: {} shards, {} cells deleted",
            n_shards_with_dv,
            fmt_num(total_deleted),
        );
    }

    // Provenance history
    if let Some(prov) = &model.provenance {
        if !prov.operations.is_empty() {
            println!();
            println!("Provenance:");
            for op in &prov.operations {
                println!("  {} | {} | {}", op.timestamp, op.action, op.tool);
                if op.params_json != "{}" && !op.params_json.is_empty() {
                    println!("    params: {}", op.params_json);
                }
            }
        }
    }

    // Manifest history
    if let Some(history) = &model.history {
        println!();
        println!("Manifest history:");
        for entry in history {
            println!(
                "  seq {}: n_obs={} ({}, {})",
                entry.manifest_sequence,
                fmt_num(entry.n_obs),
                entry.action,
                entry.timestamp,
            );
        }
    }

    Ok(())
}

/// Human-readable name for a raw `value_encoding` header byte.
fn value_encoding_name(byte: u8) -> &'static str {
    match ValueEncoding::from_u8(byte) {
        Some(ValueEncoding::Uint8) => "uint8",
        Some(ValueEncoding::Uint16) => "uint16",
        Some(ValueEncoding::Uint32) => "uint32",
        Some(ValueEncoding::Float32) => "float32",
        Some(ValueEncoding::Float16) => "float16",
        None => "unknown",
    }
}

/// Render all info as JSON.
fn render_json(model: &InfoModel) -> CliResult<()> {
    let header = &model.header;
    let catalog = &model.catalog;

    // `codec` reflects what shards actually use; `codec_default` keeps the raw
    // file-level header default for debugging.
    let codec_name = format_distinct(&model.distinct_codec_ids, codec_id_name);
    let codec_default = codec_id_name(header.codec_id);

    let mut sections = Vec::new();
    for entry in &catalog.entries {
        sections.push(serde_json::json!({
            "name": entry.name,
            "type": section_label(&entry.name, &entry.section_type),
            "offset": entry.offset,
            "length": entry.length,
        }));
    }

    let mut obj = serde_json::json!({
        "format_version": header.format_version,
        "n_obs": header.n_obs,
        "n_vars": header.n_vars,
        "nnz": header.nnz,
        "n_csr_shards": header.n_csr_shards,
        "n_csc_shards": header.n_csc_shards,
        "has_csc": header.has_csc(),
        "codec": codec_name,
        "codec_default": codec_default,
        "index_dtype": if header.index_dtype == 0 { "u16" } else { "u32" },
        "shard_target_rows": header.shard_target_rows,
        "manifest_sequence": header.manifest_sequence,
        "file_size_bytes": model.file_size,
        "sections": sections,
    });

    // Phase F.1: per-modality table in JSON output.
    if let Some(table) = &model.modality_table {
        if !table.entries.is_empty() {
            let modalities: Vec<_> = table
                .entries
                .iter()
                .map(|info| {
                    serde_json::json!({
                        "name": info.name,
                        "modality_type": modality_type_name(info.modality_type),
                        "n_vars": info.n_vars,
                        "nnz": info.nnz,
                        "n_csr_shards": info.n_csr_shards,
                        "n_csc_shards": info.n_csc_shards,
                        "has_csc": info.flags.has_csc(),
                        "default_codec": codec_id_name(info.default_codec_id),
                        "flags": info.flags.bits(),
                    })
                })
                .collect();
            obj["modalities"] = serde_json::json!(modalities);
        }
    }

    // Per-CSC-shard layout for files with multiple CSC shards.
    if header.n_csc_shards > 1 {
        let csc_layout: Vec<_> = catalog
            .csc_shards_sorted()
            .iter()
            .map(|entry| {
                let range = entry.stats.as_ref().map(|s| s.col_range());
                let nnz = entry.stats.as_ref().map(|s| s.nnz).unwrap_or(0);
                serde_json::json!({
                    "name": entry.name,
                    "col_start": range.as_ref().map(|r| r.start),
                    "col_end": range.as_ref().map(|r| r.end),
                    "nnz": nnz,
                })
            })
            .collect();
        obj["csc_layout"] = serde_json::json!(csc_layout);
    }

    // Deletion vectors
    if let Some(dv) = &model.deletion_vectors {
        obj["deletion_vectors"] = serde_json::json!({
            "n_shards": dv.shards.len(),
            "total_deleted": dv.total_deleted(),
        });
    }

    // Provenance
    if let Some(prov) = &model.provenance {
        let ops: Vec<_> = prov
            .operations
            .iter()
            .map(|op| {
                serde_json::json!({
                    "timestamp": op.timestamp,
                    "action": op.action,
                    "tool": op.tool,
                    "params": op.params_json,
                })
            })
            .collect();
        obj["provenance"] = serde_json::json!(ops);
    }

    println!("{}", serde_json::to_string_pretty(&obj)?);
    Ok(())
}

/// Walk the prev_catalog_offset chain and collect manifest history rows.
/// Local-only: the chain walk needs packed-file byte offsets the cloud path
/// doesn't have. The current manifest is the first row.
fn collect_history_local(reader: &ScxReader) -> CliResult<Vec<ManifestHistoryEntry>> {
    let mmap = reader.mmap();
    let catalog = reader.catalog();

    // Read provenance for action labels
    let prov_ops = reader
        .read_provenance()
        .map(|p| p.operations)
        .unwrap_or_default();

    let mut rows = Vec::new();

    // Current manifest
    rows.push(ManifestHistoryEntry {
        manifest_sequence: catalog.manifest_sequence,
        n_obs: catalog.n_obs,
        action: prov_ops
            .last()
            .map(|op| op.action.clone())
            .unwrap_or_else(|| "create".to_string()),
        timestamp: prov_ops
            .last()
            .map(|op| format_timestamp(op.timestamp))
            .unwrap_or_else(|| "unknown".to_string()),
    });

    // Walk the chain. Each prior `FullCatalog` is self-describing but its
    // length isn't recorded, so `try_read_catalog_at` probes sizes until the
    // BLAKE3 checksum validates.
    let mut prev_offset = catalog.prev_catalog_offset;
    let mut prov_idx = if prov_ops.len() >= 2 {
        prov_ops.len() - 2
    } else {
        0
    };

    while prev_offset != 0 {
        let fc = try_read_catalog_at(mmap, prev_offset as usize)?;

        let (action, ts) = if !prov_ops.is_empty() && prov_idx < prov_ops.len() {
            (
                prov_ops[prov_idx].action.clone(),
                format_timestamp(prov_ops[prov_idx].timestamp),
            )
        } else {
            ("create".to_string(), "unknown".to_string())
        };

        rows.push(ManifestHistoryEntry {
            manifest_sequence: fc.manifest_sequence,
            n_obs: fc.n_obs,
            action,
            timestamp: ts,
        });

        prev_offset = fc.prev_catalog_offset;
        prov_idx = prov_idx.saturating_sub(1);
    }

    Ok(rows)
}

/// Read the `FullCatalog` at `offset` in the mmap.
///
/// Prior catalogs in the manifest chain don't record their own byte length,
/// but a catalog is self-delimiting: the header records `n_entries` and each
/// entry carries `name_len` / `stats_len` prefixes, so the exact serialized
/// size is computable in one O(n_entries) pass over the length prefixes. We
/// compute `total_len` that way and hand it to `FullCatalog::read_from` once —
/// no size probing. The per-entry arithmetic mirrors `FullCatalog::write_to`.
///
/// All offsets/lengths are bounds-checked against the mmap, so a corrupt
/// `prev_catalog_offset` (past EOF) or a garbage `n_entries` / length prefix
/// returns `Err` rather than panicking or scanning the whole file.
fn try_read_catalog_at(
    mmap: &[u8],
    offset: usize,
) -> Result<FullCatalog, Box<dyn std::error::Error>> {
    // Catalog payload header: u16 version + u64 manifest_seq + u64 prev_offset
    // + u64 n_obs + u32 n_entries = 30 bytes. Trailing 32-byte BLAKE3 checksum.
    const HEADER_LEN: usize = 2 + 8 + 8 + 8 + 4;
    const CHECKSUM_LEN: usize = 32;
    // Smallest serialized entry (v1, empty name, no stats): 2 + 0 + 8 + 8 + 1
    // + 32 + 2 = 53 bytes. Matches `FullCatalog::read_from`'s alloc guard.
    const MIN_ENTRY_BYTES: usize = 53;

    // (a) A corrupt `prev_catalog_offset` can point past EOF — `checked_sub`
    // avoids the underflow/panic.
    let remaining = mmap
        .len()
        .checked_sub(offset)
        .ok_or("catalog offset points beyond end of file")?;
    if remaining < HEADER_LEN + CHECKSUM_LEN {
        return Err("catalog truncated: fewer bytes remain than a minimal catalog".into());
    }
    let slice = &mmap[offset..]; // safe: offset <= mmap.len()

    let read_u16 = |p: usize| -> Option<usize> {
        slice
            .get(p..p + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
    };
    let catalog_version = read_u16(0).ok_or("catalog header truncated")?;
    // n_entries is the u32 at byte offset 2+8+8+8 = 26.
    let n_entries = slice
        .get(26..30)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .ok_or("catalog header truncated")?;

    // (b) Guard against a garbage `n_entries` driving a long walk — the
    // entries can't fit if even their minimum size exceeds the remaining bytes.
    if n_entries.saturating_mul(MIN_ENTRY_BYTES) > remaining {
        return Err("catalog entry count exceeds remaining bytes (corrupt offset)".into());
    }

    // Bytes between an entry's name and its stats-length prefix: offset(8) +
    // length(8) + section_type(1) + checksum(32) + modality_id(1, v2+ only).
    let entry_mid = 8 + 8 + 1 + 32 + if catalog_version >= 2 { 1 } else { 0 };

    // Walk the length prefixes to find the exact end of the entry block,
    // bounds-checking every step against `remaining`.
    let mut pos = HEADER_LEN;
    for _ in 0..n_entries {
        let name_len = read_u16(pos).ok_or("catalog entry name length truncated")?;
        // Advance past name_len(2) + name + entry_mid to the stats-length prefix.
        let stats_len_pos = pos
            .checked_add(2 + name_len + entry_mid)
            .filter(|&p| p + 2 <= remaining)
            .ok_or("catalog entry truncated")?;
        let stats_len = read_u16(stats_len_pos).ok_or("catalog entry stats length truncated")?;
        pos = stats_len_pos
            .checked_add(2 + stats_len)
            .filter(|&p| p <= remaining)
            .ok_or("catalog entry stats truncated")?;
    }

    // v4 trailing generation counters (two u64), present only when declared v4.
    if catalog_version >= 4 {
        pos = pos
            .checked_add(16)
            .filter(|&p| p <= remaining)
            .ok_or("catalog v4 trailer truncated")?;
    }

    let total_len = pos
        .checked_add(CHECKSUM_LEN)
        .filter(|&p| p <= remaining)
        .ok_or("catalog checksum truncated")?;

    let mut cur = Cursor::new(&mmap[offset..offset + total_len]);
    Ok(FullCatalog::read_from(&mut cur, total_len, true)?)
}

/// Format a Unix timestamp for display.
fn format_timestamp(ts: i64) -> String {
    use chrono::DateTime;

    DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| ts.to_string())
}

fn section_label(name: &str, section_type: &SectionType) -> String {
    match section_type {
        SectionType::ObsMetadata => "obs_metadata".to_string(),
        SectionType::ObsIndex => "obs_index".to_string(),
        SectionType::VarMetadata => "var_metadata".to_string(),
        SectionType::VarIndex => "var_index".to_string(),
        SectionType::CsrShard => "X/csr".to_string(),
        SectionType::CscShard => "X/csc".to_string(),
        SectionType::BitmapShard => "X/bitmap".to_string(),
        SectionType::LayerCsrShard => {
            // name format: "layer/{name}/shard_{idx}"
            if let Some(layer) = name.strip_prefix("layer/") {
                if let Some(lname) = layer.split('/').next() {
                    return format!("layer/{lname}");
                }
            }
            format!("layer ({})", name)
        }
        SectionType::ObsmEmbedding => {
            // name format: "obsm/{name}"
            name.to_string()
        }
        SectionType::ObspCsrShard => format!("obsp ({})", name),
        SectionType::UnsBlob => "uns".to_string(),
        SectionType::Provenance => "provenance".to_string(),
        SectionType::DeletionVectors => "deletion_vectors".to_string(),
        SectionType::ObsPredicateIndex => "obs_predicate_index".to_string(),
        SectionType::VarPredicateIndex => "var_predicate_index".to_string(),
        SectionType::ModalityTable => "modality_table".to_string(),
        SectionType::LayerCscShard => {
            // Mirrors the LayerCsrShard naming pattern.
            if let Some(layer) = name.strip_prefix("layer/") {
                if let Some(lname) = layer.split('/').next() {
                    return format!("layer/{lname}/csc");
                }
            }
            format!("layer-csc ({})", name)
        }
        SectionType::VarmEmbedding => name.to_string(),
        SectionType::ObspEmbedding => name.to_string(),
        SectionType::VarpEmbedding => name.to_string(),
        SectionType::ObsmEmbeddingShard
        | SectionType::VarmEmbeddingShard
        | SectionType::ObspEmbeddingShard
        | SectionType::VarpEmbeddingShard => name.to_string(),
        // Section names already include the axis prefix
        // (`obs_metadata/shard_N` / `var_metadata/shard_N`), so use them
        // verbatim — `scx info` displays them grouped by axis.
        SectionType::ObsMetadataShard
        | SectionType::VarMetadataShard
        | SectionType::DecodeMetadataShard => name.to_string(),
        SectionType::RawCsrShard => "raw/X".to_string(),
        SectionType::RawVarMetadata => "raw/var".to_string(),
        SectionType::GroupIndex => "group_index".to_string(),
    }
}

fn modality_type_name(t: ModalityType) -> &'static str {
    match t {
        ModalityType::Rna => "RNA",
        ModalityType::Protein => "Protein",
        ModalityType::Atac => "ATAC",
        ModalityType::Spatial => "Spatial",
        ModalityType::Methylation => "Methylation",
        ModalityType::Custom => "Custom",
    }
}

fn codec_id_name(id: u8) -> &'static str {
    CodecId::from_u8(id)
        .map(|c| c.display_name())
        .unwrap_or("unknown")
}

fn fmt_num(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test shims reproducing the per-shard codec / value-encoding summary
    /// the renderer prints (distinct sorted shard fields → `format_distinct`),
    /// so the summary assertions below stay focused.
    fn summarize_csr_codec(reader: &ScxReader) -> CliResult<String> {
        let csr = reader.catalog().shards(SectionType::CsrShard);
        let ids = if csr.is_empty() {
            Vec::new()
        } else {
            crate::shard_utils::distinct_sorted_shard_field(reader, &csr, |h| h.codec_id)?
        };
        Ok(format_distinct(&ids, codec_id_name))
    }

    fn summarize_csr_value_encoding(reader: &ScxReader) -> CliResult<String> {
        let csr = reader.catalog().shards(SectionType::CsrShard);
        let encs = if csr.is_empty() {
            Vec::new()
        } else {
            crate::shard_utils::distinct_sorted_shard_field(reader, &csr, |h| h.value_encoding)?
        };
        Ok(format_distinct(&encs, value_encoding_name))
    }
    use crate::test_utils::{sample_header, sample_obs, sample_var, write_test_file};
    use scx_format_io::writer::ScxWriter;

    /// Write a 2-shard file whose shards use different value encodings:
    /// shard 0 `Uint16`, shard 1 `Uint32` (holds 66279, > u16 max).
    fn write_mixed_encoding_file(dir: &tempfile::TempDir) -> std::path::PathBuf {
        let path = dir.path().join("mixed_enc.scx");
        let mut writer = ScxWriter::new(&path, sample_header(4, 5)).unwrap();
        writer.write_obs(&sample_obs(4)).unwrap();
        writer.write_var(&sample_var(5)).unwrap();

        let s0 = ValueEncoding::Uint16
            .encode_f32_batch(&[300.0, 400.0, 500.0, 600.0])
            .unwrap();
        writer
            .write_csr_shard(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &s0,
                CodecId::None,
                ValueEncoding::Uint16,
                0,
            )
            .unwrap();

        let s1 = ValueEncoding::Uint32
            .encode_f32_batch(&[66279.0, 5.0, 7.0, 8.0])
            .unwrap();
        writer
            .write_csr_shard(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &s1,
                CodecId::None,
                ValueEncoding::Uint32,
                2,
            )
            .unwrap();

        writer.finish().unwrap();
        path
    }

    #[test]
    fn test_summarize_value_encoding_mixed() {
        let dir = tempfile::tempdir().unwrap();
        let reader = ScxReader::open(write_mixed_encoding_file(&dir)).unwrap();
        assert_eq!(
            summarize_csr_value_encoding(&reader).unwrap(),
            "mixed (uint16, uint32)"
        );
    }

    #[test]
    fn test_summarize_value_encoding_uniform() {
        let dir = tempfile::tempdir().unwrap();
        // write_test_file writes a single uint8 shard (values < 256).
        let reader = ScxReader::open(write_test_file(&dir, 6, 5)).unwrap();
        assert_eq!(summarize_csr_value_encoding(&reader).unwrap(), "uint8");
    }

    /// Write a 2-shard file whose shards use different codecs: shard 0 `None`,
    /// shard 1 `Zstd`.
    fn write_mixed_codec_file(dir: &tempfile::TempDir) -> std::path::PathBuf {
        let path = dir.path().join("mixed_codec.scx");
        let mut writer = ScxWriter::new(&path, sample_header(4, 5)).unwrap();
        writer.write_obs(&sample_obs(4)).unwrap();
        writer.write_var(&sample_var(5)).unwrap();

        let vals = ValueEncoding::Uint16
            .encode_f32_batch(&[300.0, 400.0, 500.0, 600.0])
            .unwrap();
        writer
            .write_csr_shard(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &vals,
                CodecId::None,
                ValueEncoding::Uint16,
                0,
            )
            .unwrap();
        writer
            .write_csr_shard(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &vals,
                CodecId::Zstd,
                ValueEncoding::Uint16,
                2,
            )
            .unwrap();

        writer.finish().unwrap();
        path
    }

    #[test]
    fn test_summarize_codec_mixed() {
        let dir = tempfile::tempdir().unwrap();
        let reader = ScxReader::open(write_mixed_codec_file(&dir)).unwrap();
        // Names in codec-id order: None=0, Zstd=2.
        assert_eq!(summarize_csr_codec(&reader).unwrap(), "mixed (none, zstd)");
    }

    #[test]
    fn test_summarize_codec_uniform() {
        let dir = tempfile::tempdir().unwrap();
        // write_test_file writes a single shard with CodecId::None.
        let reader = ScxReader::open(write_test_file(&dir, 6, 5)).unwrap();
        assert_eq!(summarize_csr_codec(&reader).unwrap(), "none");
    }

    #[test]
    fn test_summarize_codec_no_shards() {
        // A file with obs/var but no CSR shards → "n/a" (no shard to inspect).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_shards.scx");
        let mut writer = ScxWriter::new(&path, sample_header(4, 5)).unwrap();
        writer.write_obs(&sample_obs(4)).unwrap();
        writer.write_var(&sample_var(5)).unwrap();
        writer.finish().unwrap();
        let reader = ScxReader::open(&path).unwrap();
        assert_eq!(summarize_csr_codec(&reader).unwrap(), "n/a");
    }

    #[test]
    fn try_read_catalog_at_rejects_out_of_bounds_offset() {
        // A corrupt `prev_catalog_offset` pointing at/past EOF must return Err,
        // not underflow `mmap.len() - offset` or panic on `&mmap[offset..]`.
        let dir = tempfile::tempdir().unwrap();
        let reader = ScxReader::open(write_test_file(&dir, 6, 5)).unwrap();
        let len = reader.mmap().len();

        // offset == len: zero remaining → "too short" error, no panic.
        assert!(try_read_catalog_at(reader.mmap(), len).is_err());
        // offset > len: would underflow the old `mmap.len() - offset`.
        assert!(try_read_catalog_at(reader.mmap(), len + 1).is_err());
        assert!(try_read_catalog_at(reader.mmap(), usize::MAX).is_err());
    }
}
