// Shared helper used by `--rebuild-csc` on mutating CLI ops.
//
// The mutating ops (`append`, `compact`, `merge`, `subset`) drop CSC
// sidecars by default because their row layout no longer matches the
// pre-op CSC `indices` arrays. When the caller passes `--rebuild-csc`,
// we re-run `scx build-csc` against the post-op output via a
// temp-file + atomic rename, restoring the column-major sidecar.

use std::path::Path;

use crate::build_csc;

/// Rebuild the CSC sidecar on `target` in place via temp-file + rename.
///
/// `target` must exist and contain CSR shards. `csc_cols_per_shard`
/// and `memory_limit` mirror the `scx build-csc` CLI defaults
/// (5000 cols/shard, 4G memory budget).
pub(crate) fn rebuild_csc_inplace(
    target: &Path,
    csc_cols_per_shard: usize,
    memory_limit: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Stage the rebuilt file beside the target so the rename is atomic
    // on the same filesystem.
    let mut tmp_name = target.file_name().map(|s| s.to_owned()).unwrap_or_default();
    tmp_name.push(".rebuild_csc.tmp");
    let tmp_path = target.with_file_name(tmp_name);

    if tmp_path.exists() {
        std::fs::remove_file(&tmp_path)?;
    }

    // `run_build_csc` reads `target`, writes the CSR + new CSC sidecar
    // into `tmp_path`. Then we swap into place.
    build_csc::run_build_csc(target, &tmp_path, memory_limit, false, csc_cols_per_shard)?;
    std::fs::rename(&tmp_path, target)?;

    log::info!(
        "rebuild-csc: restored CSC sidecar on {target}",
        target = target.display()
    );
    Ok(())
}
