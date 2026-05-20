// MTX pipeline — always available, no hdf5 feature needed.

use std::path::Path;

/// Convert an MTX directory to an SCX file.
pub fn mtx_to_scx(
    input_dir: &Path,
    output: &Path,
    shard_target_rows: u32,
    codec_str: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    scx_mtx::mtx_to_scx(input_dir, output, shard_target_rows, codec_str, "scx")?;
    Ok(())
}
