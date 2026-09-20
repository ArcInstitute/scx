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

// ---------------------------------------------------------------------------
// Destination overwrite protection
// ---------------------------------------------------------------------------

/// What a destination *is*, so the guard asks the right existence question.
///
/// "Does the path exist?" is the wrong question for two of the destinations
/// `scx` writes. `scx convert --to mtx` writes *into* a directory that
/// `scx_mtx::write_scx_to_mtx` opens with `create_dir_all`, so an existing —
/// or empty — directory is the normal case, not a collision. `scx explode`
/// writes members into a `.scxd` directory for the same reason.
pub enum Destination<'a> {
    /// A single output file. Collides when that file exists.
    File(&'a Path),
    /// A Cell Ranger MTX directory. Collides on any member an export would
    /// replace — or, worse, leave behind: a stale `genes.tsv[.gz]` beside a
    /// fresh `features.tsv.gz` is a directory the MTX *reader* still accepts
    /// (it takes either spelling), so it would be read as the export's own
    /// feature table.
    MtxDir(&'a Path),
    /// An exploded `.scxd` directory. Collides when it exists and is
    /// non-empty; its member set is the catalog's, not a fixed list.
    ///
    /// Only `scx explode` writes one, and that subcommand is behind
    /// `--features cloud`, so the variant is genuinely unconstructed in a
    /// default build.
    #[cfg_attr(not(feature = "cloud"), allow(dead_code))]
    ScxdDir(&'a Path),
}

/// Every filename an MTX directory may carry that an export would replace or
/// shadow. The writer emits only the three `.gz` forms; the other five are
/// spellings [`scx_mtx::read_mtx_directory`] accepts, and leaving one next to
/// a fresh export is how a directory comes to describe two different matrices.
/// The three members `scx_mtx::write_scx_to_mtx_for` writes. Its own rename
/// replaces these, so they are never separately removed.
const MTX_WRITTEN_MEMBERS: &[&str] = &["matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"];

const MTX_MEMBERS: &[&str] = &[
    "matrix.mtx.gz",
    "matrix.mtx",
    "barcodes.tsv.gz",
    "barcodes.tsv",
    "features.tsv.gz",
    "features.tsv",
    "genes.tsv.gz",
    "genes.tsv",
];

impl<'a> Destination<'a> {
    fn path(&self) -> &'a Path {
        match *self {
            Destination::File(p) | Destination::MtxDir(p) | Destination::ScxdDir(p) => p,
        }
    }

    /// Paths that already exist and that writing this destination would
    /// replace or leave behind. Empty means there is nothing to clobber.
    fn collisions(&self) -> Vec<PathBuf> {
        match *self {
            // `symlink_metadata`, not `exists()`: `Path::exists` follows the
            // link and answers `false` for a dangling symlink, so the guard
            // would wave one through and `persist`'s rename would replace it
            // without `--force`. Any existing entry is a collision.
            Destination::File(p) => {
                if std::fs::symlink_metadata(p).is_ok() {
                    vec![p.to_path_buf()]
                } else {
                    Vec::new()
                }
            }
            // `symlink_metadata` here too: a dangling `matrix.mtx.gz`
            // symlink is not visible to `exists()`, so an unforced export
            // would silently replace it.
            Destination::MtxDir(dir) => MTX_MEMBERS
                .iter()
                .map(|name| dir.join(name))
                .filter(|p| std::fs::symlink_metadata(p).is_ok())
                .collect(),
            Destination::ScxdDir(dir) => match std::fs::read_dir(dir) {
                // Non-empty: the members we are about to write are not
                // enumerable ahead of time, so the directory itself is the
                // unit of collision.
                Ok(mut entries) => entries
                    .next()
                    .map_or(Vec::new(), |_| vec![dir.to_path_buf()]),
                // `read_dir` also fails when the path exists but is a file or
                // a dangling symlink, which is a collision, not an absence.
                Err(_) if std::fs::symlink_metadata(dir).is_ok() => {
                    vec![dir.to_path_buf()]
                }
                Err(_) => Vec::new(),
            },
        }
    }
}

/// Whether writing onto the command's own input is a legitimate in-place form.
pub enum SamePath {
    /// `optimize` / `cloud-optimize`: the writer stages a sibling tempfile and
    /// renames over the target on `finish()`, so the input is read in full
    /// before it is replaced.
    InPlaceOk,
    /// Everything else. Not a `--force` question: unlinking or renaming over
    /// the input before the op has read it is data loss whatever the flags say.
    Reject,
}

/// True when `a` and `b` name the same file. Canonicalization resolves `./a.scx`
/// against `a.scx` and follows symlinks; it fails on a path that does not exist
/// yet, which is exactly when a literal comparison is the right answer.
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// The one overwrite guard for every `scx` subcommand that writes a local
/// destination.
///
/// Before this existed the rule was hand-rolled four times — `compact.rs`,
/// `optimize.rs`, `sort.rs` and `scx_ops::run_build_csc` — in two different
/// wordings, with the same-path check present in two of them and the other six
/// destination-writing subcommands (`convert`, `merge`, `subset`,
/// `query --output`, `upgrade`, and the cloud ops) carrying no check at all.
/// `ScxWriter::finish` persists with `rename(2)`, which replaces
/// unconditionally, so "no check" meant "silently destroys the file".
///
/// `scx push` is deliberately **not** routed through here: its destination is a
/// remote `.scxd` URL, so an existence probe is a network round-trip with its
/// own failure modes rather than a `Path::exists`.
///
/// The same-path verdict is reached first, because the failure it prevents —
/// writing over the command's own input — is the one no `--force` can make
/// acceptable.
///
/// **This guard never deletes anything.** It answers one question — may this
/// command write here — and nothing else. An earlier version unlinked an
/// `MtxDir`'s members as soon as `--force` was seen, which ran *before* the
/// remaining flag validation and before the source file was even opened: a
/// forced invocation that then failed had already destroyed the previous
/// export. That is the data-loss class this helper exists to close, so the
/// rule is now structural rather than a matter of call ordering. Removal of
/// anything belongs after a successful write, at the call site that knows the
/// write succeeded.
///
/// Nothing is unlinked on `force` for a `File` destination either:
/// `ScxWriter::finish` renames over it atomically
/// (`scx-format-io/src/writer.rs`), so removing it first would only widen a
/// window in which neither the old nor the new file exists.
///
/// `inputs` is a slice because `scx merge` has several, and writing onto *any*
/// of them is the same data loss as writing onto the one input `compact` has.
/// Single-input callers pass `&[input]`; a command with no input file passes
/// `&[]`.
pub fn guard_destination(
    inputs: &[&Path],
    dest: Destination<'_>,
    same_path: SamePath,
    force: bool,
) -> CliResult<()> {
    for input in inputs {
        if same_file(input, dest.path()) {
            return match same_path {
                SamePath::InPlaceOk => Ok(()),
                SamePath::Reject => Err(format!(
                    "input and output must be different files ({} names an input)",
                    dest.path().display()
                )
                .into()),
            };
        }
    }

    let collisions = dest.collisions();
    if collisions.is_empty() || force {
        return Ok(());
    }
    Err(overwrite_refusal(&dest, &collisions))
}

/// Remove the MTX member spellings an export does **not** write, so a stale one
/// cannot survive beside the fresh files and describe a different matrix.
///
/// Call this only after `write_scx_to_mtx_for` has returned `Ok` and only when
/// `--force` authorised the overwrite. The three members the writer does
/// produce are replaced by its own rename, so they are not listed here; these
/// are the alternative spellings [`scx_mtx::read_mtx_directory`] also accepts
/// (`matrix.mtx`, the uncompressed `.tsv` forms, and the v2 `genes.tsv[.gz]`
/// that would otherwise be read as the export's own feature table).
///
/// `protect` names paths that must never be removed whatever their spelling.
/// The destination is a *directory*, so the same-path check in
/// [`guard_destination`] compares the input against the directory and cannot
/// see an input that happens to live inside it under one of these names.
pub fn clear_stale_mtx_aliases(dir: &Path, protect: &[&Path]) -> CliResult<()> {
    for name in MTX_MEMBERS {
        if MTX_WRITTEN_MEMBERS.contains(name) {
            continue;
        }
        let path = dir.join(name);
        if std::fs::symlink_metadata(&path).is_err() {
            continue;
        }
        if protect.iter().any(|p| same_file(p, &path)) {
            continue;
        }
        std::fs::remove_file(&path)?;
    }
    Ok(())
}

/// Refuse `--force` on a form that writes no destination.
///
/// `--force` is a question about a destination, so accepting it where there is
/// none would imply a guard that does not exist. That is the same failure as a
/// flag that is silently inert — `scx modify-metadata --index-*` without
/// `--obs`/`--var` used to exit 0, print "Updated metadata…", and build
/// nothing. `scx build-csc`'s in-place arm already states this reasoning; this
/// is the same rule for `query` without `--output`, `subset --dry-run`,
/// `upgrade --in-place` and `cloud-optimize` without `--output`.
///
/// `form` completes the sentence "…; " — e.g. `"--dry-run writes nothing"`.
pub fn reject_inert_force(force: bool, form: &str) -> CliResult<()> {
    if force {
        return Err(format!("--force applies only when writing to an output; {form}").into());
    }
    Ok(())
}

fn overwrite_refusal(dest: &Destination<'_>, collisions: &[PathBuf]) -> Box<dyn std::error::Error> {
    match dest {
        // The wording `scx_ops::run_build_csc` has always used. It names the
        // path, which the three CLI copies did not.
        Destination::File(p) => {
            format!("{} already exists (use --force to overwrite)", p.display()).into()
        }
        Destination::MtxDir(dir) => {
            let names: Vec<&str> = collisions
                .iter()
                .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
                .collect();
            format!(
                "{} already contains an MTX export ({}); use --force to overwrite",
                dir.display(),
                names.join(", ")
            )
            .into()
        }
        Destination::ScxdDir(dir) => format!(
            "{} already exists and is not empty (use --force to overwrite)",
            dir.display()
        )
        .into(),
    }
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
