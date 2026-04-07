use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::Path;

use scx_codec::{CodecId, ValueEncoding};
use scx_format::catalog::FullCatalog;
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};

pub fn run_info(
    path: &Path,
    json_output: bool,
    history: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = ScxReader::open_unchecked(path)?;
    let header = reader.header();
    let catalog = reader.catalog();

    if json_output {
        return print_json(path, &reader);
    }

    // Line 1: overview
    println!(
        "SCX v{} | {} cells x {} genes | {} nnz",
        header.format_version,
        fmt_num(header.n_obs),
        fmt_num(header.n_vars),
        fmt_num(header.nnz),
    );

    // Codec name
    let codec_name = match CodecId::from_u8(header.codec_id) {
        Some(CodecId::None) => "none",
        Some(CodecId::Scx1) => "scx1",
        Some(CodecId::Zstd) => "zstd",
        Some(CodecId::Lz4Shuffle) => "lz4+shuffle",
        Some(CodecId::Pcodec) => "pcodec",
        None => "unknown",
    };

    // Index dtype
    let index_dtype = match header.index_dtype {
        0 => "u16",
        1 => "u32",
        _ => "unknown",
    };

    // Line 2: shards/codec/index
    println!(
        "Shards: {} CSR | Codec: {} | Index dtype: {}",
        header.n_csr_shards, codec_name, index_dtype,
    );

    // Value encoding: read from first CSR shard header
    let csr_shards = catalog.shards(SectionType::CsrShard);
    let value_enc_name = if let Some(first_shard) = csr_shards.first() {
        let bytes = reader.section_bytes(first_shard)?;
        if bytes.len() >= SHARD_HEADER_SIZE {
            let sh =
                ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE]))?;
            match ValueEncoding::from_u8(sh.value_encoding) {
                Some(ValueEncoding::Uint8) => "uint8",
                Some(ValueEncoding::Uint16) => "uint16",
                Some(ValueEncoding::Uint32) => "uint32",
                Some(ValueEncoding::Float32) => "float32",
                Some(ValueEncoding::Float16) => "float16",
                None => "unknown",
            }
        } else {
            "n/a"
        }
    } else {
        "n/a"
    };

    // Line 3: value encoding and shard target
    println!(
        "Value encoding: {} | Shard target: {} rows",
        value_enc_name,
        fmt_num(header.shard_target_rows as u64),
    );

    // Line 4: manifest sequence
    println!("Manifest: sequence {}", header.manifest_sequence);

    // Line 5: file size
    let file_size = std::fs::metadata(path)?.len();
    println!("File size: {}", human_size(file_size));

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

    // Deletion vector detail
    if header.has_deletion_vectors() {
        if let Ok(Some(dv)) = reader.read_deletion_vectors() {
            let n_shards_with_dv = dv.shards.len();
            let total_deleted = dv.total_deleted();
            println!(
                "Deletion vectors: {} shards, {} cells deleted",
                n_shards_with_dv,
                fmt_num(total_deleted),
            );
        }
    }

    // Provenance history
    if let Ok(prov) = reader.read_provenance() {
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
    if history {
        println!();
        print_history(path, &reader)?;
    }

    Ok(())
}

/// Print all info as JSON.
fn print_json(path: &Path, reader: &ScxReader) -> Result<(), Box<dyn std::error::Error>> {
    let header = reader.header();
    let catalog = reader.catalog();
    let file_size = std::fs::metadata(path)?.len();

    let codec_name = match CodecId::from_u8(header.codec_id) {
        Some(CodecId::None) => "none",
        Some(CodecId::Scx1) => "scx1",
        Some(CodecId::Zstd) => "zstd",
        Some(CodecId::Lz4Shuffle) => "lz4+shuffle",
        Some(CodecId::Pcodec) => "pcodec",
        None => "unknown",
    };

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
        "codec": codec_name,
        "index_dtype": if header.index_dtype == 0 { "u16" } else { "u32" },
        "shard_target_rows": header.shard_target_rows,
        "manifest_sequence": header.manifest_sequence,
        "file_size_bytes": file_size,
        "sections": sections,
    });

    // Deletion vectors
    if header.has_deletion_vectors() {
        if let Ok(Some(dv)) = reader.read_deletion_vectors() {
            obj["deletion_vectors"] = serde_json::json!({
                "n_shards": dv.shards.len(),
                "total_deleted": dv.total_deleted(),
            });
        }
    }

    // Provenance
    if let Ok(prov) = reader.read_provenance() {
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

/// Walk the prev_catalog_offset chain and display manifest history.
fn print_history(_path: &Path, reader: &ScxReader) -> Result<(), Box<dyn std::error::Error>> {
    let mmap = reader.mmap();
    let _header = reader.header();
    let catalog = reader.catalog();

    // Read provenance for action labels
    let prov_ops = reader
        .read_provenance()
        .map(|p| p.operations)
        .unwrap_or_default();

    println!("Manifest history:");

    // Current manifest
    let current_action = prov_ops
        .last()
        .map(|op| op.action.as_str())
        .unwrap_or("create");
    let current_ts = prov_ops
        .last()
        .map(|op| format_timestamp(op.timestamp))
        .unwrap_or_else(|| "unknown".to_string());
    println!(
        "  seq {}: n_obs={} ({}, {})",
        catalog.manifest_sequence,
        fmt_num(catalog.n_obs),
        current_action,
        current_ts,
    );

    // Walk the chain
    let mut prev_offset = catalog.prev_catalog_offset;
    let mut prov_idx = if prov_ops.len() >= 2 {
        prov_ops.len() - 2
    } else {
        0
    };

    while prev_offset != 0 {
        // Read the FullCatalog at prev_offset
        // We need to figure out the length. The catalog is self-describing:
        // we can scan until the BLAKE3 checksum validates. But the simpler
        // approach is to read until EOF from that offset and let the catalog
        // parser figure it out. However, FullCatalog::read_from needs a
        // total_len. We'll calculate as: the next catalog starts at the
        // offset of the catalog that pointed to this one. But we don't have
        // that info easily. Instead, scan forward from prev_offset to find
        // a valid catalog.
        //
        // Actually, the file header stores full_catalog_offset/length for the
        // _current_ manifest. Previous catalogs stop at the _next_ section/
        // catalog that was written after them. But we can try by reading
        // until EOF or until the next known offset.
        //
        // Simple approach: try increasing sizes until FullCatalog parses.
        let fc = try_read_catalog_at(mmap, prev_offset as usize)?;

        let action = if !prov_ops.is_empty() && prov_idx < prov_ops.len() {
            prov_ops[prov_idx].action.as_str()
        } else {
            "create"
        };
        let ts = if !prov_ops.is_empty() && prov_idx < prov_ops.len() {
            format_timestamp(prov_ops[prov_idx].timestamp)
        } else {
            "unknown".to_string()
        };

        println!(
            "  seq {}: n_obs={} ({}, {})",
            fc.manifest_sequence,
            fmt_num(fc.n_obs),
            action,
            ts,
        );

        prev_offset = fc.prev_catalog_offset;

        // Move provenance index back
        prov_idx = prov_idx.saturating_sub(1);
    }

    Ok(())
}

/// Try to read a FullCatalog at the given offset in the mmap.
/// Probes increasing sizes until the checksum validates.
fn try_read_catalog_at(
    mmap: &[u8],
    offset: usize,
) -> Result<FullCatalog, Box<dyn std::error::Error>> {
    let remaining = mmap.len() - offset;
    // The catalog has a trailing 32-byte BLAKE3 checksum.
    // Start with a reasonable guess and grow.
    let min_size = 64; // minimum catalog: header fields + checksum
    let max_size = remaining;

    // Try the full remaining size first — the catalog read_from validates checksum
    // so it will fail if we give it too many or too few bytes. But actually,
    // read_from reads exactly total_len bytes, so we need the exact size.
    //
    // Strategy: try sizes from min up to max, stepping by scanning for a valid
    // BLAKE3 checksum at each candidate size. But this is expensive.
    //
    // Better strategy: read the catalog header to determine n_entries, estimate
    // the size, then try.
    let slice = &mmap[offset..];

    // Read catalog header fields to estimate size
    let mut cur = Cursor::new(slice);
    use byteorder::{LittleEndian, ReadBytesExt};
    let _catalog_version = cur.read_u16::<LittleEndian>()?;
    let _manifest_sequence = cur.read_u64::<LittleEndian>()?;
    let _prev_catalog_offset = cur.read_u64::<LittleEndian>()?;
    let _n_obs = cur.read_u64::<LittleEndian>()?;
    let n_entries = cur.read_u32::<LittleEndian>()? as usize;

    // Each entry is approximately: 2 (name_len) + ~20 (name) + 8+8+1+32+2 = ~73 bytes
    // plus optional stats ~41 bytes. Estimate generously.
    let header_size = 2 + 8 + 8 + 8 + 4; // 30 bytes
    let estimated_entry_size = 120; // generous estimate
    let estimated_size = header_size + n_entries * estimated_entry_size + 32; // +32 for checksum

    // Try sizes from estimated down to min, then up from estimated
    for delta in 0..max_size {
        for candidate in [
            estimated_size.wrapping_add(delta),
            estimated_size.wrapping_sub(delta),
        ] {
            if candidate < min_size || candidate > max_size {
                continue;
            }
            let try_slice = &mmap[offset..offset + candidate];
            if let Ok(fc) = FullCatalog::read_from(&mut Cursor::new(try_slice), candidate, true) {
                return Ok(fc);
            }
        }
    }

    Err("could not read previous catalog: no valid checksum found".into())
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
    }
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

fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
