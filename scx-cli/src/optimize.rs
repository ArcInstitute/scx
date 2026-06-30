// scx optimize — re-encode + canonicalize CSR shards to add decode sidecars
// and upgrade an existing single-modality file to format_version 3.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

use crate::format::human_size;

pub fn run_optimize(
    input: &Path,
    output: &Path,
    force: bool,
    codec: &str,
    shard_obs: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !input.exists() {
        return Err(format!("input file does not exist: {}", input.display()).into());
    }
    // Explicit allow-set rather than `CodecId::parse_cli` (which also accepts
    // none/zstd/lz4/pcodec): every codec other than Scx1 drops the decode
    // sidecar, defeating the point of `optimize`. `auto` → None (auto-codec,
    // Scx1 for low-median integer shards), `scx1` → force Scx1 on every integer
    // shard. clap's `value_parser` already restricts the surface to these two;
    // this match is the in-function source of truth (and guards direct callers).
    let codec_id = match codec {
        "auto" => None,
        "scx1" => Some(scx_codec::CodecId::Scx1),
        other => {
            return Err(format!(
                "--codec {other:?} is not supported by optimize \
                 (only `auto` or `scx1`; other codecs drop decode sidecars)"
            )
            .into())
        }
    };
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

    scx_ops::optimize(input, output, codec_id, obs_shard_policy)?;

    pb.finish_and_clear();
    let after_size = std::fs::metadata(output)?.len();
    println!(
        "Optimized {} → {} ({} → {}); decode sidecars added where applicable, format_version=3",
        input.display(),
        output.display(),
        human_size(before_size),
        human_size(after_size),
    );
    println!("  Verify with: scx validate --deep {}", output.display());
    Ok(())
}
