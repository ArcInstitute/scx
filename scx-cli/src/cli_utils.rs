// Shared CLI helpers for opening and validating SCX input files.
//
// Several subcommands (`merge`, `compact`, …) repeat the same
// existence check and the "every input must share the same `n_vars`"
// guard. Defining them once keeps the error wording consistent.

use std::path::{Path, PathBuf};

use scx_format_io::reader::ScxReader;

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Error out if `path` does not exist on disk. Mirrors the message
/// previously inlined in `merge`/`compact`.
pub fn validate_scx_file(path: &Path) -> CliResult<()> {
    if !path.exists() {
        return Err(format!("input file does not exist: {}", path.display()).into());
    }
    Ok(())
}

/// Derive the framing config to use when rewriting `path`'s CSC sidecar so the
/// rebuild preserves the file's existing layout: a framed (v4) file keeps
/// framing (default `G`), an unframed (≤v3) file stays unframed. Without this,
/// `rebuild_csc_inplace(..., None)` re-encodes every CSR shard unframed and
/// silently downgrades a v4 file back to v3 — undoing the framing that
/// compact/sort/merge/append/subset just preserved. Returns `None` (unframed)
/// if the file can't be opened; the CSC rebuild surfaces any real error.
///
/// **Deliberately `FramingConfig::default()` — i.e. `decode_target: None`.**
/// This is the one framing site where the `fast` profile is correct rather than
/// a bug: a CSC rebuild re-writes each CSR shard at the codec read off the
/// source shard header, and `decode_target: Some(_)` would authorise the writer
/// to re-select it (see `FramingConfig`'s contract), silently defeating the
/// preservation. Do not "make this consistent" with the derived-file ops.
/// Pinned by `scx-ops/tests/codec_adaptive.rs::build_csc_preserves_per_shard_codec`.
///
/// **Every `rebuild_csc_inplace` caller must get its framing from here** (or, in
/// `scx-convert`, from `IngestOptions::framing_preserving_codec`) — never from
/// the framing used for the surrounding rewrite. `subset` and `convert --csc`
/// both passed the rewrite framing until this was caught in review: harmless
/// while `write_shard_inner` ignored `decode_target`, a silent codec override
/// once it honoured it.
pub fn framing_for_file(path: &Path) -> Option<scx_format_io::FramingConfig> {
    // Delegates so the CLI, pyscx and convert cannot drift apart on this rule
    // again -- see `scx_ops::framing_for_csc_rebuild`.
    scx_ops::framing_for_csc_rebuild(path)
}

/// Validate that every input file reports the same file-level `n_vars`,
/// returning that shared value. Opens each file once. The first input is
/// the reference; any mismatch reports both paths and counts.
pub fn validate_all_same_n_vars(paths: &[PathBuf]) -> CliResult<u64> {
    if paths.is_empty() {
        return Err("no input files provided".into());
    }
    let first_reader = ScxReader::open(&paths[0])?;
    let expected_n_vars = first_reader.header().n_vars;
    drop(first_reader);

    for p in &paths[1..] {
        let reader = ScxReader::open(p)?;
        let n_vars = reader.header().n_vars;
        if n_vars != expected_n_vars {
            return Err(format!(
                "n_vars mismatch: {} has {} vars, {} has {} vars",
                paths[0].display(),
                expected_n_vars,
                p.display(),
                n_vars
            )
            .into());
        }
    }
    Ok(expected_n_vars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_all_same_n_vars_empty_is_err_not_panic() {
        let err = validate_all_same_n_vars(&[]).unwrap_err();
        assert!(err.to_string().contains("no input files provided"));
    }
}
