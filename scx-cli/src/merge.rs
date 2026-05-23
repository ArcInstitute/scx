// scx merge — Streaming merge of multiple SCX files.

use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use scx_engine::ConversionPredicateIndexOptions;
use scx_format::reader::ScxReader;

use crate::index_warnings::emit_index_summary;

#[allow(clippy::too_many_arguments)]
pub fn run_merge(
    inputs: &[PathBuf],
    output: &Path,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
) -> Result<(), Box<dyn std::error::Error>> {
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

    // Call scx_ops::merge with predicate-index options. Empty option
    // bag → behaves like the old no-index call.
    let index_options = ConversionPredicateIndexOptions {
        index_obs,
        index_var,
        index_preset,
        index_auto_threshold,
    };
    let summary = scx_ops::merge_with_index_options(&input_refs, output, &index_options)?;
    emit_index_summary("merge", &summary);

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
    drop(out_reader);

    // Re-emit the CSC sidecar against the merged output.
    if rebuild_csc {
        scx_ops::rebuild_csc_inplace(output, csc_cols_per_shard, "4G")?;
        println!("Rebuilt CSC sidecar on {}", output.display());
    }

    Ok(())
}
