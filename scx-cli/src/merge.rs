// scx merge — Streaming merge of multiple SCX files.

use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use scx_format::reader::ScxReader;

pub fn run_merge(inputs: &[PathBuf], output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Validate at least 2 inputs
    if inputs.len() < 2 {
        return Err("at least 2 files required for merge".into());
    }

    // Validate all inputs exist
    for p in inputs {
        if !p.exists() {
            return Err(format!("input file does not exist: {}", p.display()).into());
        }
    }

    // Validate n_vars consistency
    let first_reader = ScxReader::open(&inputs[0])?;
    let expected_n_vars = first_reader.header().n_vars;
    drop(first_reader);

    for p in &inputs[1..] {
        let reader = ScxReader::open(p)?;
        let n_vars = reader.header().n_vars;
        if n_vars != expected_n_vars {
            return Err(format!(
                "n_vars mismatch: {} has {} vars, {} has {} vars",
                inputs[0].display(),
                expected_n_vars,
                p.display(),
                n_vars
            )
            .into());
        }
    }

    // Show progress spinner
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Merging {} files...", inputs.len()));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Build refs for the ops API
    let input_refs: Vec<&Path> = inputs.iter().map(|p| p.as_path()).collect();

    // Call scx_ops::merge
    scx_ops::merge(&input_refs, output)?;

    pb.finish_and_clear();

    // Report summary
    let out_reader = ScxReader::open(output)?;
    let total_obs = out_reader.header().n_obs;
    let total_shards = out_reader.header().n_csr_shards;

    println!(
        "Merged {} files into {} ({} cells, {} shards)",
        inputs.len(),
        output.display(),
        total_obs,
        total_shards,
    );

    Ok(())
}
