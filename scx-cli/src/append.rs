// scx append — Append cells from another SCX file.

use std::path::Path;

use scx_codec::{CodecId, ValueEncoding};
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};

pub fn run_append(
    target: &Path,
    input: &Path,
    modality: Option<&str>,
    codec: &str,
    shard_size: u32,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Open input file
    let input_reader = ScxReader::open(input)?;

    // Open target file to validate n_vars
    let target_reader = ScxReader::open(target)?;
    let target_header = target_reader.header();

    // Phase F.3: resolve modality routing.
    //
    //   - On a multimodal target, `--modality NAME` is required so the
    //     append is unambiguous (cells are global, but X is
    //     per-modality).
    //   - On a single-modality target, `--modality` is optional and
    //     defaults to 0 (global / legacy).
    let target_modality_id: u8 = if target_reader.is_multimodal() {
        match modality {
            Some(name) => target_reader.modality_id(name).ok_or_else(|| {
                format!(
                    "target file does not have a modality named '{name}'; \
                     run `scx info {}` to list modalities",
                    target.display()
                )
            })?,
            None => {
                return Err(format!(
                    "target file is multimodal ({} modalities); pass `--modality NAME`. \
                     Run `scx info {}` to list modalities.",
                    target_reader.n_modalities(),
                    target.display()
                )
                .into());
            }
        }
    } else if let Some(name) = modality {
        return Err(format!(
            "target file is single-modality but `--modality {name}` was passed; \
             remove the flag (or use a multimodal target)"
        )
        .into());
    } else {
        0
    };

    // n_vars cross-check: per-modality append uses the target modality's
    // `n_vars`; legacy / single-modality append uses the file-wide
    // `header.n_vars`.
    let target_modality_n_vars: u64 = if target_modality_id == 0 {
        target_header.n_vars
    } else {
        target_reader
            .modality_info(target_modality_id)
            .map(|info| info.n_vars)
            .unwrap_or(target_header.n_vars)
    };

    // Resolve the matching input modality when both files are multimodal.
    // For a multimodal input → multimodal target append, the input must
    // expose the same modality name. For a single-modality input → any
    // target, just use the input's global shards.
    let input_modality_id: u8 = if input_reader.is_multimodal() {
        if target_modality_id == 0 {
            return Err("input file is multimodal but target is single-modality; \
                 use `scx subset --modality NAME` on the input first"
                .into());
        }
        let mname = modality.expect("multimodal target requires --modality");
        input_reader.modality_id(mname).ok_or_else(|| {
            format!(
                "input file is multimodal but does not have a modality named '{mname}'; \
                 the input must expose the same modality as the target"
            )
        })?
    } else {
        0
    };

    // n_vars equality between source and target on the matching axis.
    let input_n_vars: u64 = if input_modality_id == 0 {
        input_reader.header().n_vars
    } else {
        input_reader
            .modality_info(input_modality_id)
            .map(|info| info.n_vars)
            .unwrap_or(input_reader.header().n_vars)
    };
    if target_modality_n_vars != input_n_vars {
        return Err(format!(
            "n_vars mismatch: target has {target_modality_n_vars} vars, \
             input has {input_n_vars} vars"
        )
        .into());
    }

    // Read CSR data from the matching modality of the input.
    let csr = if input_modality_id == 0 {
        input_reader.read_all_csr_shards()?
    } else {
        input_reader.read_all_csr_shards_for(input_modality_id)?
    };

    if csr.n_rows() == 0 {
        println!("Input file has 0 cells, nothing to append.");
        return Ok(());
    }

    // Detect ValueEncoding from the matching modality's first CSR shard.
    let csr_entries: Vec<&scx_format::FullCatalogEntry> = input_reader
        .catalog()
        .shards(SectionType::CsrShard)
        .into_iter()
        .filter(|e| e.modality_id == input_modality_id)
        .collect();
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
        return Err("input file has no CSR shards for the requested modality".into());
    };

    // Convert from scipy in-memory types back to on-disk types
    // indptr: i64 → u64
    let new_indptr: Vec<u64> = csr.indptr.iter().map(|&v| v as u64).collect();
    // indices: i32 → u32
    let new_indices: Vec<u32> = csr.indices.iter().map(|&v| v as u32).collect();
    // data: f32 → raw LE bytes matching ValueEncoding
    let new_values = f32_to_raw_values(&csr.data, value_encoding);

    // Read obs metadata from input (cells are global across modalities,
    // so we always pull from the global obs table).
    let new_obs = input_reader.read_obs()?;

    // Resolve codec. Per-modality append uses the target modality's
    // biological type for auto-codec routing.
    let target_modality_type = if target_modality_id == 0 {
        scx_format::ModalityType::Rna
    } else {
        target_reader
            .modality_info(target_modality_id)
            .map(|info| info.modality_type)
            .unwrap_or(scx_format::ModalityType::Rna)
    };
    let codec_id = match codec {
        "auto" => {
            scx_format::select_codec_for_modality(&new_values, value_encoding, target_modality_type)
        }
        "none" => CodecId::None,
        "scx1" => CodecId::Scx1,
        "zstd" => CodecId::Zstd,
        "lz4" => CodecId::Lz4Shuffle,
        "pcodec" => CodecId::Pcodec,
        other => {
            return Err(format!(
                "unknown codec: '{}'. Use auto, none, scx1, zstd, lz4, or pcodec.",
                other
            )
            .into())
        }
    };

    // Drop the target reader before scx_ops::append acquires the file lock.
    drop(target_reader);

    // Call scx_ops::append (per-modality variant).
    let n_cells = new_indptr.len() - 1;
    scx_ops::append_for_modality(
        target,
        &new_obs,
        &new_indptr,
        &new_indices,
        &new_values,
        value_encoding,
        codec_id,
        shard_size,
        target_modality_id,
    )?;

    println!(
        "Appended {} cells from {} to {}",
        n_cells,
        input.display(),
        target.display()
    );

    // Re-emit the CSC sidecar that scx_ops::append dropped.
    // We always run when `--rebuild-csc` is set, even if the target
    // didn't have CSC before — the user opted in explicitly.
    if rebuild_csc {
        crate::rebuild_csc::rebuild_csc_inplace(target, csc_cols_per_shard, "4G")?;
        println!("Rebuilt CSC sidecar on {}", target.display());
    }

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
