// scx optimize — re-encode + canonicalize CSR shards to add decode sidecars
// and upgrade an existing single-modality file to format_version 3.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

use crate::format::human_size;

pub fn run_optimize(
    input: &Path,
    output: &Path,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !input.exists() {
        return Err(format!("input file does not exist: {}", input.display()).into());
    }
    // Allow an explicit in-place upgrade (`--output` == input): `ScxWriter`
    // writes a sibling tempfile and atomically renames over the target on
    // `finish()`, so the input is read in full before it is replaced. Only
    // guard against clobbering a *different* pre-existing file.
    let same_file = match (std::fs::canonicalize(input), std::fs::canonicalize(output)) {
        (Ok(p1), Ok(p2)) => p1 == p2,
        _ => false,
    };
    if output.exists() && !force && !same_file {
        return Err("output file already exists, use --force to overwrite".into());
    }

    let before_size = std::fs::metadata(input)?.len();

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Optimizing {}...", input.display()));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    scx_ops::optimize(input, output)?;

    pb.finish_and_clear();
    let after_size = std::fs::metadata(output)?.len();
    println!(
        "Optimized {} → {} ({} → {}); decode sidecars added where applicable, format_version=3",
        input.display(),
        output.display(),
        human_size(before_size),
        human_size(after_size),
    );
    println!("  Verify with: scx validate --deep {}", output.display());
    Ok(())
}
