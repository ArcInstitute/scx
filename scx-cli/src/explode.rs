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
    let work_dir = if staged {
        sibling_temp(output_dir, "exploding")
    } else {
        output_dir.to_path_buf()
    };
    if staged {
        let _ = std::fs::remove_dir_all(&work_dir);
    }

    if let Err(e) = scx_cloud::explode(input, &work_dir) {
        if staged {
            let _ = std::fs::remove_dir_all(&work_dir);
        }
        return Err(e.into());
    }

    if staged {
        let retired = sibling_temp(output_dir, "replaced");
        let _ = std::fs::remove_dir_all(&retired);
        std::fs::rename(output_dir, &retired)?;
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

/// A sibling of `dir` (same parent, therefore the same filesystem, so the
/// swap is a rename rather than a copy).
fn sibling_temp(dir: &Path, tag: &str) -> PathBuf {
    let name = dir.file_name().map_or_else(
        || format!(".scx-{tag}-{}", std::process::id()),
        |n| format!(".{}.{tag}-{}", n.to_string_lossy(), std::process::id()),
    );
    dir.parent().unwrap_or(Path::new(".")).join(name)
}
