use std::collections::BTreeMap;
use std::path::Path;

use scx_codec::{CodecId, ValueEncoding};
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};

pub fn run_info(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let reader = ScxReader::open(path)?;
    let header = reader.header();
    let catalog = reader.catalog();

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
        let bytes = reader.section_bytes(first_shard);
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

    Ok(())
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
