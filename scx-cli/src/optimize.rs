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
    // Explicit allow-set rather than `CodecId::parse_cli`. `auto` → None
    // (auto-codec, Scx1 for low-median integer shards), `scx1` → force Scx1 on
    // every integer shard. The framed codecs (`shufdelta` / `compact-trial`)
    // require `--row-group-rows` for the block-index random-access layout. clap's
    // `value_parser` restricts the surface; this match is the in-function source
    // of truth (and guards direct callers).
    let (codec_id, codec_trial) = match codec {
        "auto" => (None, false),
        "scx1" => (Some(scx_codec::CodecId::Scx1), false),
        "shufdelta" => (Some(scx_codec::CodecId::ShufDeltaZstd), false),
        // `compact-trial` is a framing *profile*, not a codec: per shard, keep
        // the smaller of {heuristic winner, ShufDeltaZstd}.
        "compact-trial" => (None, true),
        other => {
            return Err(format!(
                "--codec {other:?} is not supported by optimize \
                 (`auto` | `scx1` | `shufdelta` | `compact-trial`)"
            )
            .into())
        }
    };

    // Framing is on by default; `--row-group-rows 0` is the explicit unframed
    // (v3) opt-out → normalize to None so the guard and framing agree.
    let row_group_rows = row_group_rows.filter(|&g| g > 0);
    // The framed codecs only make sense with row-group framing on.
    if (codec_trial || codec_id == Some(scx_codec::CodecId::ShufDeltaZstd))
        && row_group_rows.is_none()
    {
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
    // unframed runs keep the historical "decode sidecars added" phrasing.
    let detail = if stats.shards_framed > 0 || stats.unframed_scx1_gpu > 0 {
        format!(
            "row-group-framed {}/{} shards ({} stored as ShufDeltaZstd, \
             {} kept unframed Scx1 for GPU/per-row)",
            stats.shards_framed,
            stats.shards_total,
            stats.shards_shufdelta,
            stats.unframed_scx1_gpu,
        )
    } else {
        "decode sidecars added where applicable".to_string()
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
