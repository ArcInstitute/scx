use std::path::Path;

pub fn run_pack(
    input_dir: &Path,
    output: &Path,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    crate::cli_utils::guard_destination(
        &[input_dir],
        crate::cli_utils::Destination::File(output),
        crate::cli_utils::SamePath::Reject,
        force,
    )?;
    scx_cloud::pack(input_dir, output)?;
    println!("Packed {} → {}", input_dir.display(), output.display());
    Ok(())
}
