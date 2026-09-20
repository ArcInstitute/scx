// MTX pipeline — always available, no hdf5 feature needed.

use std::path::Path;

/// Convert an MTX directory to an SCX file.
///
/// Returns the detected on-disk [`scx_mtx::MtxOrientation`] so the CLI can
/// warn the user when a square matrix forced an ambiguous orientation guess.
pub fn mtx_to_scx(
    input_dir: &Path,
    output: &Path,
    shard_target_rows: u32,
    codec_str: &str,
    obs_shard_policy: scx_format_io::ObsShardPolicy,
    allow_lossy: bool,
) -> Result<scx_mtx::MtxOrientation, Box<dyn std::error::Error>> {
    let orientation = scx_mtx::mtx_to_scx(
        input_dir,
        output,
        shard_target_rows,
        codec_str,
        "scx",
        obs_shard_policy,
        allow_lossy,
    )?;
    Ok(orientation)
}
