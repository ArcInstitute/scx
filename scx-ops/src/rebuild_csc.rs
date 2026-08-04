// Shared helper used by `--rebuild-csc` on mutating CLI ops.
//
// The mutating ops (`append`, `compact`, `merge`, `subset`) drop CSC
// sidecars by default because their row layout no longer matches the
// pre-op CSC `indices` arrays. When the caller passes `--rebuild-csc`,
// we re-run `scx build-csc` against the post-op output via a
// temp-file + atomic rename, restoring the column-major sidecar.

use std::path::Path;

use scx_format_io::FramingConfig;

use crate::build_csc;

/// Framing for a CSC-sidecar rebuild on `path`: keep the file's existing layout
/// and re-select nothing.
///
/// A framed (v4) target keeps framing; an unframed (≤v3) one stays unframed.
/// `decode_target` is deliberately `None`, because `run_build_csc` re-writes each
/// CSR shard at the codec read off that shard's own header and
/// `decode_target: Some(_)` would let the writer override it.
///
/// One implementation for every surface — CLI, pyscx and convert — because this
/// rule has now been got wrong twice in two different directions: `subset` and
/// `convert --csc` passed the rewrite framing (codec override), and pyscx's
/// `sort`/`shuffle`/`from_anndata` passed `None` (framing downgrade). Returns
/// `None` if `path` cannot be opened; the rebuild itself surfaces any real error.
pub fn framing_for_csc_rebuild(path: &Path) -> Option<FramingConfig> {
    scx_format_io::ScxReader::open(path)
        .ok()
        .filter(|r| r.header().format_version >= scx_format_io::CURRENT_FORMAT_VERSION)
        .map(|_| FramingConfig::default())
}

/// Rebuild the CSC sidecar on `target` in place via temp-file + rename.
///
/// `target` must exist and contain CSR shards. `csc_cols_per_shard`
/// and `memory_limit` mirror the `scx build-csc` CLI defaults
/// (5000 cols/shard, 4G memory budget).
///
/// `framing`: forwarded to [`build_csc::run_build_csc`] — `Some` re-writes CSR +
/// CSC framed (v4), `None` unframed (v3).
///
/// **Derive it with [`framing_for_csc_rebuild`], not by hand.** Both ends of the
/// range are wrong: `None` on a v4 target strips row-group framing from the X
/// that was just written, and a framing carrying `decode_target: Some(_)`
/// authorises the writer to *re-select* each CSR shard's codec (see
/// `FramingConfig`'s contract) — the opposite of the preservation a sidecar
/// rebuild needs. The streaming-convert `--csc` path is the one exception: it
/// passes `ConvertOptions::framing_preserving_codec()` so a custom
/// `--row-group-rows` is honoured on the sidecar too, with `decode_target`
/// stripped for the same reason.
pub fn rebuild_csc_inplace(
    target: &Path,
    csc_cols_per_shard: usize,
    memory_limit: &str,
    framing: Option<FramingConfig>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Stage the rebuilt file beside the target so the rename is atomic
    // on the same filesystem.
    let mut tmp_name = target.file_name().map(|s| s.to_owned()).unwrap_or_default();
    tmp_name.push(".rebuild_csc.tmp");
    let tmp_path = target.with_file_name(tmp_name);

    // `run_build_csc` reads `target` and writes the CSR + new CSC sidecar into
    // `tmp_path` (via its own temp→fsync→rename, overwriting any stale temp),
    // then we swap into place. OE7: a cleanup guard removes the partial temp on
    // any failure so a failed rebuild never leaks a stray `*.rebuild_csc.tmp`;
    // this also drops the previous `exists()`-then-`remove_file` TOCTOU.
    let build_and_swap = || -> Result<(), Box<dyn std::error::Error>> {
        build_csc::run_build_csc(
            target,
            &tmp_path,
            memory_limit,
            false,
            csc_cols_per_shard,
            framing,
        )?;
        std::fs::rename(&tmp_path, target)?;
        Ok(())
    };
    if let Err(e) = build_and_swap() {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }

    log::info!(
        "rebuild-csc: restored CSC sidecar on {target}",
        target = target.display()
    );
    Ok(())
}
