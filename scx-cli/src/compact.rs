// scx compact — Rewrite file reclaiming space from deletions.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

pub fn run_compact(
    input: &Path,
    output: &Path,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate input exists
    if !input.exists() {
        return Err(format!("input file does not exist: {}", input.display()).into());
    }

    // Check output
    if output.exists() && !force {
        return Err("output file already exists, use --force to overwrite".into());
    }

    // If force and output exists, remove it first
    if output.exists() && force {
        std::fs::remove_file(output)?;
    }

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

    // Call scx_ops::compact
    scx_ops::compact(input, output)?;

    pb.finish_and_clear();

    let after_size = std::fs::metadata(output)?.len();
    let reduction = if before_size > 0 {
        ((before_size as f64 - after_size as f64) / before_size as f64) * 100.0
    } else {
        0.0
    };

    println!(
        "Compacted {} -> {} ({} -> {}, {:.1}% reduction)",
        input.display(),
        output.display(),
        human_size(before_size),
        human_size(after_size),
        reduction,
    );

    Ok(())
}

fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
