use std::path::Path;

pub fn run_pack(
    input_dir: &Path,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    scx_cloud::pack(input_dir, output)?;
    println!(
        "Packed {} → {}",
        input_dir.display(),
        output.display()
    );
    Ok(())
}
