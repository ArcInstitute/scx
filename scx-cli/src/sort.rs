// scx sort — global obs-axis row reorder for query locality.

use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use scx_engine::ConversionPredicateIndexOptions;

use crate::format::human_size;

#[allow(clippy::too_many_arguments)]
pub fn run_sort(
    input: &Path,
    output: &Path,
    by: Vec<String>,
    reverse: bool,
    // 1D: `Some(seed)` when `--shuffle` was passed. The dispatch layer resolves
    // `--seed`'s default, so the engine only ever sees an explicit seed.
    shuffle: Option<u64>,
    force: bool,
    shard_size: u32,
    codec: &str,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    memory_budget: Option<String>,
    temp_dir: Option<PathBuf>,
    bitmap: &str,
    csc: crate::cli_utils::CscArgs,
    group_by: Option<String>,
    reference: Option<String>,
    group_target_bytes: Option<String>,
    group_max_bytes: Option<String>,
    group_write_block_bytes: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    crate::cli_utils::validate_scx_file(input)?;
    // Three order sources: `--by`, `--group-by`, and 1D's `--shuffle`. The
    // engine rejects any combination of them; this only catches the case of
    // supplying none.
    if by.is_empty() && group_by.is_none() && shuffle.is_none() {
        return Err(
            "scx sort requires --by with at least one obs column (or --group-by, or --shuffle)"
                .into(),
        );
    }

    // sort always writes a fresh file, so same-path is never valid — writing
    // onto the input would destroy it before the engine reads it.
    crate::cli_utils::guard_destination(
        &[input],
        crate::cli_utils::Destination::File(output),
        crate::cli_utils::SamePath::Reject,
        force,
    )?;

    // Full intent axis (`auto`/`fast`/`compact`/`compact-trial` + explicit
    // codecs), not just `CodecId::parse_cli`'s explicit set. `auto` here now
    // means what it means in `scx convert`; it previously resolved to the
    // single-encode heuristic, which is what let `sort --by` roughly double X.
    let codec = scx_format_io::resolve_codec(Some(codec))?;
    let bitmap = scx_format_io::BitmapPolicy::parse(bitmap)?;
    let memory_budget = match memory_budget {
        Some(s) => Some(scx_format_io::MemoryBudget::parse(&s)?),
        None => None,
    };

    // F1: grouped-sharding options. `--reference` is a comma-separated label
    // list by default, or `col:<name>` (alias `column:<name>`) for the
    // boolean-column form. Shared parser with `scx convert` so the flag behaves
    // identically on both subcommands. A non-empty value that fails to parse is
    // a user error, not a silent "no reference".
    let reference = match reference.as_deref() {
        Some(spec) if !spec.trim().is_empty() => {
            let parsed = crate::parse_reference_spec_cli(spec);
            if parsed.is_none() {
                return Err(format!(
                    "--reference value {spec:?} is not a valid label set or `col:NAME` \
                     reference column"
                )
                .into());
            }
            parsed
        }
        _ => None,
    };
    let group_target_bytes = match group_target_bytes {
        Some(s) => Some(scx_format_io::MemoryBudget::parse(&s)?),
        None => None,
    };
    let group_max_bytes = match group_max_bytes {
        Some(s) => Some(scx_format_io::MemoryBudget::parse(&s)?),
        None => None,
    };
    let group_write_block_bytes = match group_write_block_bytes {
        Some(s) => Some(scx_format_io::MemoryBudget::parse(&s)?),
        None => None,
    };
    if reference.is_some() && group_by.is_none() {
        return Err("--reference requires --group-by".into());
    }
    if group_by.is_none()
        && (group_target_bytes.is_some()
            || group_max_bytes.is_some()
            || group_write_block_bytes.is_some())
    {
        log::warn!(
            "scx sort: --group-target-bytes / --group-max-bytes / --group-write-block-bytes are \
             ignored without --group-by"
        );
    }

    let before_size = std::fs::metadata(input)?.len();

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(match shuffle {
        Some(seed) => format!("Shuffling {} (seed {seed})...", input.display()),
        None => format!("Sorting {} by {}...", input.display(), by.join(",")),
    });
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // The sidecar builder spills where the sort's external partitions do.
    let csc = csc.options(temp_dir.clone());
    let opts = scx_ops::SortOptions {
        by,
        reverse,
        shuffle,
        shard_target_rows: shard_size,
        codec,
        index_options: ConversionPredicateIndexOptions {
            index_obs,
            index_var,
            index_preset,
            // The sort key is auto-added to the index regardless; this caps
            // auto-detection of *other* low-cardinality columns (0 = none).
            index_auto_threshold: index_auto_threshold.unwrap_or(0),
        },
        memory_budget,
        temp_dir,
        bitmap,
        group_by,
        reference,
        group_target_bytes,
        group_max_bytes,
        group_write_block_bytes,
        csc,
    };

    let summary = scx_ops::sort(input, output, &opts)?;
    pb.finish_and_clear();

    let after_size = std::fs::metadata(output)?.len();
    println!(
        "{} {} -> {} ({} -> {}, {} rows, {} shards, strategy {:?})",
        match shuffle {
            Some(seed) => format!("Shuffled (seed {seed})"),
            None => "Sorted".to_string(),
        },
        input.display(),
        output.display(),
        human_size(before_size),
        human_size(after_size),
        summary.n_obs,
        summary.n_output_shards,
        summary.strategy,
    );
    if summary.spill_bytes > 0 {
        println!(
            "  external partition sort: {} spilled across {} partitions",
            human_size(summary.spill_bytes),
            summary.partitions,
        );
    }
    if !summary.indexed_columns.is_empty() {
        println!(
            "  indexed obs columns: {}",
            summary.indexed_columns.join(", ")
        );
    }

    if scx_format_io::ScxReader::open(output)?.header().has_csc() {
        println!("  CSC sidecar built in the same pass");
    }

    Ok(())
}
