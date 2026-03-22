use std::path::Path;

pub fn run_cloud_optimize(
    input: &Path,
    output: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = output.unwrap_or(input);
    scx_cloud::cloud_optimize(input, output_path)?;

    if output_path == input {
        println!("Cloud-optimized {} (in-place)", input.display());
    } else {
        println!(
            "Cloud-optimized {} → {}",
            input.display(),
            output_path.display()
        );
    }
    Ok(())
}
