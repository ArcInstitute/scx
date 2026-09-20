use std::path::Path;

pub fn run_cloud_optimize(
    input: &Path,
    output: Option<&Path>,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // `--output` omitted means an in-place rewrite via atomic rename, so an
    // explicit `--output <input>` is the same legitimate request.
    match output {
        Some(output) => crate::cli_utils::guard_destination(
            &[input],
            crate::cli_utils::Destination::File(output),
            crate::cli_utils::SamePath::InPlaceOk,
            force,
        )?,
        None => crate::cli_utils::reject_inert_force(
            force,
            "omitting --output rewrites <INPUT> in place",
        )?,
    }
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
