// scx merge — Streaming merge of multiple SCX files.

use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use scx_engine::ConversionPredicateIndexOptions;
use scx_format::reader::ScxReader;
use scx_ops::{MergeOptions, UnsPolicy};

use crate::index_warnings::emit_index_summary;

#[allow(clippy::too_many_arguments)]
pub fn run_merge(
    inputs: &[PathBuf],
    output: &Path,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: &str,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    assume_identical_var: bool,
    assume_identical_obs: bool,
    uns_policy: Option<String>,
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

    // Parse the optional --uns-policy flag.
    let uns_policy = match uns_policy.as_deref() {
        Some(s) => UnsPolicy::parse(s).ok_or_else(|| {
            format!(
                "invalid --uns-policy value '{s}'; expected one of: \
                 first, require-equal, namespace, summary"
            )
        })?,
        None => UnsPolicy::First,
    };

    // Route to `merge_with_options` whenever any non-default
    // policy or index flag is set; fall back to the bare `merge`
    // wrapper otherwise. Both paths now enforce the default var- and
    // obs-identity checks; the routing only affects how policy
    // overrides + index work flow through. `--index-auto-threshold
    // N` alone reaches the engine.
    let any_index_flag = !index_obs.is_empty()
        || !index_var.is_empty()
        || index_preset.is_some()
        || index_auto_threshold.is_some();
    let any_policy_flag =
        assume_identical_var || assume_identical_obs || uns_policy != UnsPolicy::First;
    if any_index_flag || any_policy_flag {
        let index_options = ConversionPredicateIndexOptions {
            index_obs,
            index_var,
            index_preset,
            index_auto_threshold: index_auto_threshold.unwrap_or(1000),
        };
        let merge_opts = MergeOptions {
            index_options,
            assume_identical_var,
            assume_identical_obs,
            uns_policy,
            shard_target_rows: None,
        };
        let summary = scx_ops::merge_with_options(&input_refs, output, &merge_opts)?;
        emit_index_summary("merge", &summary);
    } else {
        scx_ops::merge(&input_refs, output)?;
    }

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
        scx_ops::rebuild_csc_inplace(output, csc_cols_per_shard, csc_memory_limit)?;
        println!("Rebuilt CSC sidecar on {}", output.display());
    }

    Ok(())
}
