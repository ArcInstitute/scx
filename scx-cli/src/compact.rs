// scx compact — Rewrite file reclaiming space from deletions.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use scx_engine::ConversionPredicateIndexOptions;

use crate::format::human_size;
use crate::index_warnings::emit_index_summary;

#[allow(clippy::too_many_arguments)]
pub fn run_compact(
    input: &Path,
    output: &Path,
    force: bool,
    csc: scx_ops::CscCarryOptions,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    reshape_obs: bool,
    codec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate input exists
    crate::cli_utils::validate_scx_file(input)?;

    // Resolve before any work: an unknown codec should fail immediately, not
    // after the rewrite has started.
    let resolved_codec = scx_format_io::resolve_codec(Some(codec))?;

    // Destination guard, shared with every other subcommand that writes a
    // local destination (see `cli_utils::guard_destination`). `Reject` rather
    // than `InPlaceOk` is a fix, not a translation: `scx compact f.scx f.scx
    // --force` used to unlink the output and *then* read `metadata(input)`,
    // destroying the input and failing. `sort` and `build-csc` already guarded
    // that; compact was the third instance.
    crate::cli_utils::guard_destination(
        &[input],
        crate::cli_utils::Destination::File(output),
        crate::cli_utils::SamePath::Reject,
        force,
    )?;

    let before_size = std::fs::metadata(input)?.len();

    // Show progress spinner
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!("Compacting {}...", input.display()));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Route to `compact` (drops predicate indexes — pre-fix default) when
    // no `--index-*` flag is set, and to `compact_with_index_options`
    // (rebuilds / auto-detects) when at least one flag is set.
    // `--index-auto-threshold N` alone reaches the engine.
    let any_index_flag = !index_obs.is_empty()
        || !index_var.is_empty()
        || index_preset.is_some()
        || index_auto_threshold.is_some();
    // `--reshape-obs` must reach the index-options path even with no
    // `--index-*` flag set. The `index_auto_threshold = 0` sentinel keeps
    // `user_wants_index()` false (no index built), matching bare
    // `compact()`, while still migrating obs to sharded sections.
    if any_index_flag || reshape_obs {
        let index_options = if any_index_flag {
            ConversionPredicateIndexOptions {
                index_obs,
                index_var,
                index_preset,
                index_auto_threshold: index_auto_threshold.unwrap_or(1000),
            }
        } else {
            ConversionPredicateIndexOptions {
                index_obs: Vec::new(),
                index_var: Vec::new(),
                index_preset: None,
                index_auto_threshold: 0,
            }
        };
        let summary = scx_ops::compact_with_options(
            input,
            output,
            &scx_ops::CompactOptions {
                index_options,
                reshape_obs,
                codec: resolved_codec,
                csc,
            },
        )?;
        emit_index_summary("compact", &summary);
    } else {
        // Still the options path, so a lone `--codec` is not silently dropped:
        // the default index options carry the `index_auto_threshold = 0`
        // sentinel, which reproduces bare `compact()` exactly.
        scx_ops::compact_with_options(
            input,
            output,
            &scx_ops::CompactOptions {
                codec: resolved_codec,
                csc,
                ..Default::default()
            },
        )?;
    }

    pb.finish_and_clear();

    let after_size = std::fs::metadata(output)?.len();
    let reduction = if before_size > 0 {
        ((before_size as f64 - after_size as f64) / before_size as f64) * 100.0
    } else {
        0.0
    };

    // Display-only: `reduction` is negative when the output grew, so label
    // the printed percentage by sign and report the absolute value.
    let (pct, label) = if after_size < before_size {
        (reduction, "reduction")
    } else if after_size > before_size {
        (-reduction, "increase")
    } else {
        (0.0, "unchanged")
    };
    println!(
        "Compacted {} -> {} ({} -> {}, {:.1}% {})",
        input.display(),
        output.display(),
        human_size(before_size),
        human_size(after_size),
        pct,
        label,
    );

    if scx_format_io::ScxReader::open(output)?.header().has_csc() {
        println!("  CSC sidecar built in the same pass");
    }

    Ok(())
}
