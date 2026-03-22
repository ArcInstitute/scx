use std::path::Path;

pub fn run_explode(
    input: &Path,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    scx_cloud::explode(input, output_dir)?;
    println!(
        "Exploded {} → {}",
        input.display(),
        output_dir.display()
    );
    Ok(())
}
