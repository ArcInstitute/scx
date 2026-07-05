// scx append — Append cells from another SCX file.

use std::num::NonZeroU32;
use std::path::Path;

use scx_codec::{CodecId, CodecSelection};
use scx_engine::ConversionPredicateIndexOptions;
use scx_format_io::reader::ScxReader;

use crate::index_warnings::emit_index_summary;

#[allow(clippy::too_many_arguments)]
pub fn run_append(
    target: &Path,
    input: &Path,
    modality: Option<&str>,
    codec: &str,
    shard_size: NonZeroU32,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: &str,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
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

    // Count cells in the matching modality for the progress message.
    // Iterate shard headers rather than relying on `entry.stats`, which the
    // catalog format permits to be `None`. The streaming path also assumes
    // the source has at least one CSR shard for the requested modality.
    let csr_entries = input_reader
        .catalog()
        .csr_shards_for_modality(input_modality_id);

    if csr_entries.is_empty() {
        println!(
            "Input file has no CSR shards for modality {input_modality_id}, nothing to append."
        );
        return Ok(());
    }

    let mut n_cells: u64 = 0;
    for entry in &csr_entries {
        let sh = input_reader.read_shard_header(entry)?;
        n_cells += sh.n_major as u64;
    }

    if n_cells == 0 {
        println!("Input file has 0 cells, nothing to append.");
        return Ok(());
    }

    // Resolve codec.
    let codec_selection = match CodecId::parse_cli(codec)? {
        None => CodecSelection::Auto,
        Some(c) => CodecSelection::Explicit(c),
    };

    // Drop the target reader before scx_ops::append_from_reader acquires
    // the file lock.
    drop(target_reader);

    let append_options = scx_ops::AppendOptions {
        codec: codec_selection,
        shard_target_rows: shard_size,
        modality_id: target_modality_id,
    };

    // Route to the legacy entry point (which preserves pre-existing
    // predicate-index sections as stale) when the user passed no
    // `--index-*` flag, and to `*_with_index_options` (which rebuilds /
    // auto-detects) when at least one flag is set. `--index-auto-threshold N`
    // alone reaches the engine — this was the silent-no-op bug being
    // fixed.
    let any_index_flag = !index_obs.is_empty()
        || !index_var.is_empty()
        || index_preset.is_some()
        || index_auto_threshold.is_some();
    if any_index_flag {
        let index_options = ConversionPredicateIndexOptions {
            index_obs,
            index_var,
            index_preset,
            index_auto_threshold: index_auto_threshold.unwrap_or(1000),
        };
        let summary = scx_ops::append_from_reader_with_index_options(
            target,
            &input_reader,
            &append_options,
            input_modality_id,
            &index_options,
        )?;
        emit_index_summary("append", &summary);
    } else {
        scx_ops::append_from_reader(target, &input_reader, &append_options, input_modality_id)?;
    }

    println!(
        "Appended {} cells from {} to {}",
        n_cells,
        input.display(),
        target.display()
    );

    // Re-emit the CSC sidecar that scx_ops::append dropped, preserving the
    // target's framing (a v4 target must stay framed after the rebuild).
    if rebuild_csc {
        let framing = crate::cli_utils::framing_for_file(target);
        scx_ops::rebuild_csc_inplace(target, csc_cols_per_shard, csc_memory_limit, framing)?;
        println!("Rebuilt CSC sidecar on {}", target.display());
    }

    Ok(())
}
