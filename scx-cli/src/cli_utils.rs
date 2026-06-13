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

/// Validate that every input file reports the same file-level `n_vars`,
/// returning that shared value. Opens each file once. The first input is
/// the reference; any mismatch reports both paths and counts.
pub fn validate_all_same_n_vars(paths: &[PathBuf]) -> CliResult<u64> {
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
