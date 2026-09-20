use std::path::Path;

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
    scx_cloud::explode(input, output_dir)?;
    println!("Exploded {} → {}", input.display(), output_dir.display());
    Ok(())
}
