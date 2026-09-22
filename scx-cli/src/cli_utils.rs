// Shared CLI helpers for opening and validating SCX input files.
//
// Several subcommands (`merge`, `compact`, …) repeat the same
// existence check and the "every input must share the same `n_vars`"
// guard. Defining them once keeps the error wording consistent.

use std::path::{Path, PathBuf};

use scx_format_io::reader::ScxReader;

type CliResult<T> = Result<T, Box<dyn std::error::Error>>;

/// `--csc` on a rewrite op.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CscMode {
    /// Build one iff the input had one.
    Carry,
    /// Build one whether or not the input had one.
    Always,
    /// Build none.
    Off,
}

/// The CSC-sidecar flags every rewrite op (`compact`, `merge`, `optimize`,
/// `sort`, `subset`) takes, in one place so their wording cannot drift.
#[derive(clap::Args, Clone, Debug)]
pub struct CscArgs {
    /// Whether the output carries a CSC (column-major) sidecar. `carry`
    /// (default): one iff the input had one — for merge, iff any input had one.
    /// `always`: build one regardless. `off`: none. The input's sidecar is
    /// never copied (its row indices are stale once rows move); a new one is
    /// built from the output's own X shards in the same pass that writes them,
    /// so there is no second read of the output. Not supported for a
    /// multimodal file: `carry` drops its sidecars with a warning and
    /// `always` is refused.
    #[arg(long, value_enum, default_value_t = CscMode::Carry)]
    pub csc: CscMode,
    /// Deprecated spelling of `--csc always`.
    #[arg(long, hide = true, conflicts_with = "csc")]
    pub rebuild_csc: bool,
    /// Maximum columns per emitted CSC shard (default: 5000).
    #[arg(long, default_value_t = 5000)]
    pub csc_cols_per_shard: usize,
    /// Budget for the sidecar build (default 4G), as for `scx build-csc
    /// --memory-limit`: it bounds the builder's resident column buckets and
    /// sizes the sidecar's shard widths. Accepts a binary-prefixed size
    /// (`K`/`M`/`G`/`T` or `KiB`..`TiB`); decimal `KB`/`MB`/`GB` is rejected.
    #[arg(long, default_value = "4G")]
    pub csc_memory_limit: String,
}

impl CscArgs {
    /// The ops-level options. `temp_dir` is where the builder spills (`None`:
    /// the output's own directory).
    pub fn options(&self, temp_dir: Option<PathBuf>) -> scx_ops::CscCarryOptions {
        let mode = if self.rebuild_csc {
            eprintln!("warning: --rebuild-csc is deprecated; it means --csc always");
            scx_ops::CscOutput::Always
        } else {
            match self.csc {
                CscMode::Carry => scx_ops::CscOutput::Carry,
                CscMode::Always => scx_ops::CscOutput::Always,
                CscMode::Off => scx_ops::CscOutput::Off,
            }
        };
        scx_ops::CscCarryOptions {
            mode,
            cols_per_shard: self.csc_cols_per_shard,
            memory_limit: self.csc_memory_limit.clone(),
            temp_dir,
        }
    }
}

/// Error out if `path` does not exist on disk. Mirrors the message
/// previously inlined in `merge`/`compact`.
pub fn validate_scx_file(path: &Path) -> CliResult<()> {
    if !path.exists() {
        return Err(format!("input file does not exist: {}", path.display()).into());
    }
    Ok(())
}

/// The framing argument for a CSC sidecar build on `path`: `Some(default)` on a
/// framed (v4) file, `None` on an unframed (≤v3) one — exactly the values the
/// build admits. The build appends the sidecar and never rewrites a CSR shard,
/// so it frames the sidecar to match the file and refuses `Some` on a ≤ v3 file
/// rather than re-framing it. Returns `None` if the file can't be opened; the
/// build surfaces any real error.
///
/// **Deliberately `FramingConfig::default()` — i.e. `decode_target: None`.**
/// `decode_target: Some(_)` would authorise the encoder to re-pick the
/// sidecar's codec, which `pick_csc_encoding` has already decided (the build
/// also clears it defensively). While build-csc still rewrote the CSR, the same
/// field would have re-selected every CSR shard's codec — `subset` and
/// `convert --csc` passed the rewrite framing until review caught it.
///
/// **Every `rebuild_csc_inplace` caller should get its framing from here** (or,
/// in `scx-convert`, from `IngestOptions::framing_preserving_codec`, which
/// carries a custom `--row-group-rows` through to the sidecar).
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

/// Resolve a source string to the local path it will actually be read from,
/// or `None` when it names a genuinely remote object.
///
/// `scx pull` and `scx query` take a URL *or* a path, and pass it to the guard
/// as a `Path`. For a `file://` URL that is a lie: `Path::new("file:///tmp/x")`
/// canonicalizes to nothing while `scx-cloud` later reads `/tmp/x`, so the
/// containment check compared the wrong string and
/// `scx pull file:///tmp/src.scxd /tmp/src.scxd/_catalog.bin --force` destroyed
/// the source it was pulling.
pub fn local_source_path(source: &str) -> Option<PathBuf> {
    if let Some(rest) = source.strip_prefix("file://") {
        // `file:///abs` → `/abs`; a host component (`file://host/p`) is not a
        // local path this process can compare against.
        // `file:///abs` → `/abs`. Anything else carries a host component
        // (`file://host/p`), which is not a path this process can compare
        // against.
        return rest
            .strip_prefix('/')
            .map(|abs| PathBuf::from(format!("/{abs}")));
    }
    if source.contains("://") {
        return None; // gs://, s3://, az://, …
    }
    Some(PathBuf::from(source))
}

/// Whether `path` lies strictly inside `dir`.
///
/// Tested on both the resolved and the literal path. Resolving alone is not
/// enough: a symlink sitting *inside* `dir` but pointing outside it
/// canonicalizes to a location outside, and would be judged safe even though
/// the entry that gets removed is the one inside.
fn contains(dir: &Path, path: &Path) -> bool {
    // Resolved location first: this is the answer for every ordinary case.
    if let (Ok(d), Some(p)) = (std::fs::canonicalize(dir), resolve_parent(path)) {
        if d != p && p.starts_with(&d) {
            return true;
        }
    }
    // The lexical arm exists only for one case the resolved one cannot see: a
    // symlink sitting *inside* `dir` but pointing outside it resolves to a
    // location outside, while the entry that actually gets removed is the one
    // inside.
    //
    // It is a string-prefix test, so it is only sound when both paths are made
    // of plain named components. `..` defeats a prefix comparison outright —
    // `/data/in.scxd/../out.scx` starts with `/data/in.scxd` and is not inside
    // it — and an *empty* normalized dir (`.`, `./`, `./.`) is a prefix of
    // literally every path, which refused every relative destination spelled
    // `.`. Both are cases where only the resolved answer means anything.
    let (d, p) = (normalize(dir), normalize(path));
    if d.as_os_str().is_empty() || has_parent_component(&d) || has_parent_component(&p) {
        return false;
    }
    d != p && p.starts_with(&d)
}

fn has_parent_component(p: &Path) -> bool {
    p.components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
}

/// Canonicalize `path`, falling back to canonicalizing its parent and
/// re-attaching the file name — a destination usually does not exist yet.
fn resolve_parent(path: &Path) -> Option<PathBuf> {
    if let Ok(c) = std::fs::canonicalize(path) {
        return Some(c);
    }
    let parent = path.parent()?;
    let name = path.file_name()?;
    std::fs::canonicalize(parent).ok().map(|c| c.join(name))
}

/// Strip `.` components so a literal comparison is not defeated by `./a`.
fn normalize(p: &Path) -> PathBuf {
    p.components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .collect()
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
        // Equality is not the whole invariant. A *directory* input and a file
        // destination inside it compare unequal and destroy the source anyway
        // — `scx convert --from mtx dir dir/matrix.mtx.gz` replaced its own
        // matrix — and the mirror case, an input inside a directory
        // destination, is deleted by the swap that replaces that directory.
        if matches!(same_path, SamePath::Reject) {
            if contains(input, dest.path()) {
                return Err(format!(
                    "{} is inside the input {}; writing there would destroy part of the source",
                    dest.path().display(),
                    input.display()
                )
                .into());
            }
            // Only for a destination this command replaces *wholesale*.
            // `explode --force` swaps the whole `.scxd` tree, so anything
            // inside it is deleted. An `MtxDir` is not that: the writer
            // replaces three member names and leaves every sibling alone, so
            // refusing here would reject `scx convert ./sample.scx . --to mtx`
            // — the ordinary "write the MTX files here" invocation. The case
            // that genuinely would destroy the source there (the source *is*
            // one of those three names) is refused by `write_scx_to_mtx_for`.
            if matches!(dest, Destination::ScxdDir(_)) && contains(dest.path(), input) {
                return Err(format!(
                    "{} is inside the output {}; writing there would destroy the input",
                    input.display(),
                    dest.path().display()
                )
                .into());
            }
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
/// Check, before anything is written, that every alias this command will
/// later remove *can* be removed.
///
/// `clear_stale_mtx_aliases` runs after the export has replaced the three
/// primary members, so a failure there — an alias that is a directory, say —
/// leaves the command exiting non-zero with the members already swapped,
/// breaking the "a failed command changes nothing" boundary the rest of this
/// guard maintains. Finding out first costs one `symlink_metadata` per name.
pub fn preflight_mtx_aliases(dir: &Path, protect: &[&Path]) -> CliResult<()> {
    for name in MTX_MEMBERS {
        if MTX_WRITTEN_MEMBERS.contains(name) {
            continue;
        }
        let path = dir.join(name);
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.is_dir() => {
                if protect.iter().any(|p| same_file(p, &path)) {
                    continue;
                }
                return Err(format!(
                    "{} is a directory, so it cannot be cleared after the export; \
                     remove it or choose another output directory",
                    path.display()
                )
                .into());
            }
            _ => {}
        }
    }
    Ok(())
}

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
