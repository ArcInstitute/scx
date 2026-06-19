// scx sort — global obs-axis row reorder for query locality (SCX-SORT-SPEC).

use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use scx_codec::{CodecId, CodecSelection};
use scx_engine::ConversionPredicateIndexOptions;

use crate::format::human_size;

#[allow(clippy::too_many_arguments)]
pub fn run_sort(
    input: &Path,
    output: &Path,
    by: Vec<String>,
    reverse: bool,
    force: bool,
    shard_size: u32,
    codec: &str,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    memory_budget: Option<String>,
    temp_dir: Option<PathBuf>,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    crate::cli_utils::validate_scx_file(input)?;
    if by.is_empty() {
        return Err("scx sort requires --by with at least one obs column".into());
    }

    if output.exists() && !force {
        return Err("output file already exists, use --force to overwrite".into());
    }
    if output.exists() && force {
        std::fs::remove_file(output)?;
    }

    let codec = match CodecId::parse_cli(codec)? {
        None => CodecSelection::Auto,
        Some(c) => CodecSelection::Explicit(c),
    };
    let memory_budget = match memory_budget {
        Some(s) => Some(scx_format_io::MemoryBudget::parse(&s)?),
        None => None,
    };

    let before_size = std::fs::metadata(input)?.len();

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!(
        "Sorting {} by {}...",
        input.display(),
        by.join(",")
    ));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let opts = scx_ops::SortOptions {
        by,
        reverse,
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
    };

    let summary = scx_ops::sort(input, output, &opts)?;
    pb.finish_and_clear();

    let after_size = std::fs::metadata(output)?.len();
    println!(
        "Sorted {} -> {} ({} -> {}, {} rows, {} shards, strategy {:?})",
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

    if rebuild_csc {
        scx_ops::rebuild_csc_inplace(output, csc_cols_per_shard, csc_memory_limit)?;
        println!("Rebuilt CSC sidecar on {}", output.display());
    }

    Ok(())
}
