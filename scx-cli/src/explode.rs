use std::path::{Path, PathBuf};

pub fn run_explode(
    input: &Path,
    output_dir: &Path,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // An empty (or absent) `.scxd` is the normal case; a non-empty one holds
    // another file's sections, which this explode would interleave with.
    crate::cli_utils::guard_destination(
        &[input],
        crate::cli_utils::Destination::ScxdDir(output_dir),
        crate::cli_utils::SamePath::Reject,
        force,
    )?;

    // `--force` on a non-empty `.scxd` must *replace* it, not overlay it.
    // `scx_cloud::explode` writes the sections the current catalog names and
    // leaves every other file alone, so a previous explode's extra shards,
    // layers or obsm entries would ride along — invisible to `scx pack` /
    // `info` / `query`, which are catalog-authoritative, but not to a
    // directory sync of the `.scxd`.
    //
    // Staged through a sibling so a failure cannot destroy the directory it
    // was told to replace: explode into a temp dir, and only swap once it has
    // succeeded. `rename(2)` over a directory needs the target gone, so the
    // old one is moved aside first and removed after the swap.
    let staged = force && dir_is_non_empty(output_dir);

    // The swap below removes the whole retired tree, so an input living inside
    // the destination would be deleted along with it. The destination is a
    // directory, so the same-path check above compares a file against a
    // directory and cannot see this.
    if staged && path_contains(output_dir, input) {
        return Err(format!(
            "{} is inside the output directory {}; --force replaces that directory \
             wholesale, which would delete the input",
            input.display(),
            output_dir.display()
        )
        .into());
    }

    // Exclusive creation, never a guessed name: `remove_dir_all` on a
    // predictable `<dir>.<tag>-<pid>` path deletes whatever happens to be
    // there — a leftover from a killed run after PID reuse, or a directory the
    // user created — before anything has proved we own it.
    let work_dir = if staged {
        create_exclusive_sibling(output_dir, "exploding")?
    } else {
        output_dir.to_path_buf()
    };

    if let Err(e) = scx_cloud::explode(input, &work_dir) {
        if staged {
            let _ = std::fs::remove_dir_all(&work_dir);
        }
        return Err(e.into());
    }

    if staged {
        let retired = match create_exclusive_sibling(output_dir, "replaced") {
            Ok(p) => p,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&work_dir);
                return Err(e);
            }
        };
        // `rename` needs the target gone, and we only just created it.
        let _ = std::fs::remove_dir(&retired);
        if let Err(e) = std::fs::rename(output_dir, &retired) {
            let _ = std::fs::remove_dir_all(&work_dir);
            return Err(e.into());
        }
        if let Err(e) = std::fs::rename(&work_dir, output_dir) {
            // Put the original back rather than leaving the user with neither.
            let _ = std::fs::rename(&retired, output_dir);
            let _ = std::fs::remove_dir_all(&work_dir);
            return Err(e.into());
        }
        let _ = std::fs::remove_dir_all(&retired);
    }

    println!("Exploded {} → {}", input.display(), output_dir.display());
    Ok(())
}

fn dir_is_non_empty(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|mut it| it.next().is_some())
}

/// Create a fresh sibling directory of `dir` and return it.
///
/// A sibling so the later swap is a rename rather than a copy (same parent,
/// therefore same filesystem), and **created exclusively** so the path is ours
/// before anything is removed through it. `create_dir` fails on an existing
/// path, so a colliding name advances the counter instead of being deleted.
fn create_exclusive_sibling(dir: &Path, tag: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let parent = dir.parent().unwrap_or(Path::new("."));
    let stem = dir
        .file_name()
        .map_or_else(|| "scx".to_string(), |n| n.to_string_lossy().into_owned());
    for attempt in 0..1024 {
        let candidate = parent.join(format!(".{stem}.{tag}-{}-{attempt}", std::process::id()));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Err(format!(
        "could not create a staging directory beside {}",
        dir.display()
    )
    .into())
}

/// Whether `path` lies inside `dir`, by canonical path.
fn path_contains(dir: &Path, path: &Path) -> bool {
    match (std::fs::canonicalize(dir), std::fs::canonicalize(path)) {
        (Ok(d), Ok(p)) => p.starts_with(d),
        _ => false,
    }
}
