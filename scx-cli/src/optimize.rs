// scx optimize — re-encode + canonicalize CSR shards, applying row-group
// framing and upgrading an existing single-modality file to the current format.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

use crate::format::human_size;

#[allow(clippy::too_many_arguments)]
pub fn run_optimize(
    input: &Path,
    output: &Path,
    force: bool,
    codec: &str,
    row_group_rows: Option<u32>,
    row_group_target_nnz: Option<u64>,
    shard_obs: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !input.exists() {
        return Err(format!("input file does not exist: {}", input.display()).into());
    }
    // Resolve the codec intent axis — shared with `scx convert` / pyscx via
    // `scx_format_io::resolve_codec`. `auto` (default) is cost-aware adaptive;
    // `fast` keeps the heuristic; `compact`/`compact-trial`/`shufdelta` require
    // row-group framing. clap's `value_parser` further restricts the CLI surface.
    let resolved = scx_format_io::resolve_codec(Some(codec))?;
    let codec_id = resolved.explicit_codec;
    let codec_trial = resolved.codec_trial;
    let decode_target = resolved.decode_target;

    // Framing is on by default; `--row-group-rows 0` is the explicit unframed
    // (v3) opt-out → normalize to None so the guard and framing agree.
    let row_group_rows = row_group_rows.filter(|&g| g > 0);
    if resolved.requires_framing && row_group_rows.is_none() {
        return Err(format!(
            "`--codec {codec}` requires `--row-group-rows N` (row-group-framed output); \
             drop `--row-group-rows 0`"
        )
        .into());
    }
    let framing = row_group_rows.map(|g| scx_format_io::FramingConfig {
        row_group_rows: g,
        target_nnz: row_group_target_nnz,
        trial: codec_trial,
        decode_target,
    });
    // Allow an explicit in-place upgrade (`--output` == input): `ScxWriter`
    // writes a sibling tempfile and atomically renames over the target on
    // `finish()`, so the input is read in full before it is replaced. Only
    // guard against clobbering a *different* pre-existing file.
    let same_file = match (std::fs::canonicalize(input), std::fs::canonicalize(output)) {
        (Ok(p1), Ok(p2)) => p1 == p2,
        _ => false,
    };
    if output.exists() && !force && !same_file {
        return Err("output file already exists, use --force to overwrite".into());
    }

    let obs_shard_policy = scx_format_io::ObsShardPolicy::parse(shard_obs)?;

    let before_size = std::fs::metadata(input)?.len();

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Optimizing {}...", input.display()));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let stats = scx_ops::optimize_with_framing(input, output, codec_id, obs_shard_policy, framing)?;

    pb.finish_and_clear();
    let after_size = std::fs::metadata(output)?.len();
    // Framed runs report the row-group layout + per-shard ShufDeltaZstd adoption;
    // unframed runs just report the shard count.
    let detail = if stats.shards_framed > 0 {
        format!(
            "row-group-framed {}/{} shards ({} stored as ShufDeltaZstd)",
            stats.shards_framed, stats.shards_total, stats.shards_shufdelta,
        )
    } else {
        format!("{} shards re-encoded (unframed)", stats.shards_total)
    };
    println!(
        "Optimized {} → {} ({} → {}); {}, format_version={}",
        input.display(),
        output.display(),
        human_size(before_size),
        human_size(after_size),
        detail,
        stats.format_version,
    );
    println!("  Verify with: scx validate --deep {}", output.display());
    Ok(())
}
