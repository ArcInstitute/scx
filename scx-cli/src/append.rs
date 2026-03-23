// scx append — Append cells from another SCX file.

use std::path::Path;

use scx_codec::{CodecId, ValueEncoding};
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};

pub fn run_append(
    target: &Path,
    input: &Path,
    codec: &str,
    shard_size: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Open input file
    let input_reader = ScxReader::open(input)?;
    let input_header = input_reader.header();

    // Open target file to validate n_vars
    let target_reader = ScxReader::open(target)?;
    let target_header = target_reader.header();

    if target_header.n_vars != input_header.n_vars {
        return Err(format!(
            "n_vars mismatch: target has {} vars, input has {} vars",
            target_header.n_vars, input_header.n_vars
        )
        .into());
    }
    drop(target_reader);

    // Read CSR data from input (in-memory scipy types: i64/i32/f32)
    let csr = input_reader.read_all_csr_shards()?;

    if csr.n_rows() == 0 {
        println!("Input file has 0 cells, nothing to append.");
        return Ok(());
    }

    // Detect ValueEncoding from the input file's first CSR shard header
    let csr_entries = input_reader.catalog().shards(SectionType::CsrShard);
    let value_encoding = if let Some(first_shard) = csr_entries.first() {
        let bytes = input_reader.section_bytes(first_shard)?;
        if bytes.len() >= SHARD_HEADER_SIZE {
            let sh =
                ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE]))?;
            ValueEncoding::from_u8(sh.value_encoding)
                .ok_or_else(|| format!("unknown value encoding: {}", sh.value_encoding))?
        } else {
            return Err("input CSR shard too small to read header".into());
        }
    } else {
        return Err("input file has no CSR shards".into());
    };

    // Convert from scipy in-memory types back to on-disk types
    // indptr: i64 → u64
    let new_indptr: Vec<u64> = csr.indptr.iter().map(|&v| v as u64).collect();
    // indices: i32 → u32
    let new_indices: Vec<u32> = csr.indices.iter().map(|&v| v as u32).collect();
    // data: f32 → raw LE bytes matching ValueEncoding
    let new_values = f32_to_raw_values(&csr.data, value_encoding);

    // Read obs metadata from input
    let new_obs = input_reader.read_obs()?;

    // Resolve codec
    let codec_id = match codec {
        "auto" => scx_format::select_codec(&new_values, value_encoding),
        "none" => CodecId::None,
        "scx1" => CodecId::Scx1,
        "zstd" => CodecId::Zstd,
        other => {
            return Err(
                format!("unknown codec: '{}'. Use auto, none, scx1, or zstd.", other).into(),
            )
        }
    };

    // Call scx_ops::append
    let n_cells = new_indptr.len() - 1;
    scx_ops::append(
        target,
        &new_obs,
        &new_indptr,
        &new_indices,
        &new_values,
        value_encoding,
        codec_id,
        shard_size,
    )?;

    println!(
        "Appended {} cells from {} to {}",
        n_cells,
        input.display(),
        target.display()
    );

    Ok(())
}

/// Convert f32 data to raw LE bytes matching the given ValueEncoding.
fn f32_to_raw_values(data: &[f32], encoding: ValueEncoding) -> Vec<u8> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&v| v as u8).collect(),
        ValueEncoding::Uint16 => {
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &v in data {
                bytes.extend_from_slice(&(v as u16).to_le_bytes());
            }
            bytes
        }
        ValueEncoding::Uint32 => {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for &v in data {
                bytes.extend_from_slice(&(v as u32).to_le_bytes());
            }
            bytes
        }
        ValueEncoding::Float32 => {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for &v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes
        }
        ValueEncoding::Float16 => {
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &v in data {
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        }
    }
}
