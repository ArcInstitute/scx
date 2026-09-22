// Shared helper used by `scx build-csc` with no `<OUTPUT>` and by
// `--rebuild-csc` on the mutating ops.
//
// The mutating ops (`append`, `compact`, `merge`, `subset`, `sort`, and the
// streaming convert) drop CSC sidecars by default because their row layout no
// longer matches the pre-op CSC `indices` arrays. When the caller passes
// `--rebuild-csc`, the sidecar is appended to the post-op output in place.

use std::path::Path;

use scx_format_io::FramingConfig;

use crate::build_csc::BuildCscOutcome;

/// Framing for a CSC-sidecar build on `path`: the file's existing layout, with
/// nothing re-selected.
///
/// `Some(default)` on a framed (v4) file, `None` on an unframed (≤v3) one —
/// exactly the values the in-place build admits, since an append frames the
/// sidecar to match the file and cannot change the file. `decode_target` is
/// `None` because it would authorise the encoder to re-pick the sidecar's codec,
/// which `pick_csc_encoding` has already decided.
///
/// One implementation for every surface — CLI, pyscx and convert — because this
/// rule was got wrong twice in two directions while `build-csc` still rewrote
/// the CSR: `subset` and `convert --csc` passed the rewrite framing (codec
/// override), and pyscx's `sort`/`shuffle`/`from_anndata` passed `None`, which
/// then stripped framing off the whole file. The build now refuses the first
/// kind of mismatch outright and cannot do the second. Returns `None` if `path`
/// cannot be opened; the build itself surfaces any real error.
pub fn framing_for_csc_rebuild(path: &Path) -> Option<FramingConfig> {
    scx_format_io::ScxReader::open(path)
        .ok()
        .filter(|r| r.header().format_version >= scx_format_io::CURRENT_FORMAT_VERSION)
        .map(|_| FramingConfig::default())
}

/// Build the CSC sidecar on `target` in place: append it at EOF and repoint the
/// catalog. See [`crate::build_csc`]'s in-place core for the contract — nothing
/// else in the file moves, and `scx rollback` undoes it.
///
/// `target` must exist and contain CSR shards. `csc_cols_per_shard`
/// and `memory_limit` mirror the `scx build-csc` CLI defaults
/// (5000 cols/shard, 4G memory budget).
///
/// `framing`: the sidecar's framing — admissible only on a v4 target. **Derive
/// it with [`framing_for_csc_rebuild`]**; the streaming-convert `--csc` path is
/// the one exception, passing `IngestOptions::framing_preserving_codec()` so a
/// custom `--row-group-rows` reaches the sidecar too.
pub fn rebuild_csc_inplace(
    target: &Path,
    csc_cols_per_shard: usize,
    memory_limit: &str,
    framing: Option<FramingConfig>,
    // `temp_dir`: root for the CSC builder's spill files; `None` uses the
    // target's own directory.
    temp_dir: Option<&Path>,
) -> Result<BuildCscOutcome, Box<dyn std::error::Error>> {
    let built = crate::build_csc::build_csc_in_place(
        target,
        memory_limit,
        csc_cols_per_shard,
        framing,
        temp_dir,
    )?;
    match built.outcome {
        BuildCscOutcome::Built => log::info!(
            "rebuild-csc: restored CSC sidecar on {target}",
            target = target.display()
        ),
        BuildCscOutcome::NoSidecar => log::info!(
            "rebuild-csc: {target} is an empty matrix; no CSC sidecar to restore",
            target = target.display()
        ),
    }
    Ok(built.outcome)
}
