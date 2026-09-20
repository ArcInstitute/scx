use std::path::Path;

pub fn run_pull(
    source: &str,
    dest: &Path,
    force: bool,
    parallelism: usize,
    cloud_ready: bool,
    filter: Option<&str>,
    filter_mode: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // `source` is a URL or a local `.scxd` path; when it resolves to a local
    // path, writing the pull onto it would destroy the thing being pulled.
    // Resolve `file://` first — comparing the raw string treats the URL as a
    // relative path named `file:` and the containment check never fires.
    let local_source = crate::cli_utils::local_source_path(source);
    let inputs: Vec<&Path> = local_source.as_deref().into_iter().collect();
    crate::cli_utils::guard_destination(
        &inputs,
        crate::cli_utils::Destination::File(dest),
        crate::cli_utils::SamePath::Reject,
        force,
    )?;
    let rt = tokio::runtime::Runtime::new()?;

    let mode = match filter_mode {
        "shard" => scx_cloud::FilterMode::Shard,
        "exact" => scx_cloud::FilterMode::Exact,
        other => {
            return Err(
                format!("invalid --filter-mode: '{other}'; expected 'shard' or 'exact'").into(),
            )
        }
    };

    let opts = scx_cloud::PullOptions {
        parallelism,
        cloud_ready,
        filter_mode: mode,
        retry_config: scx_cloud::RetryConfig::default(),
    };

    if let Some(filter_expr) = filter {
        let stats = rt.block_on(scx_cloud::pull_filtered(source, dest, filter_expr, opts))?;
        let mode_label = match stats.filter_mode {
            scx_cloud::FilterMode::Shard => "shard-granular",
            scx_cloud::FilterMode::Exact => "exact",
        };
        // Report the cells WRITTEN distinctly from the cells MATCHING the
        // filter. Shard-granular pull retains whole shards, so the output
        // file contains every row of each retained shard — typically far
        // more than the rows matching the predicate. Conflating the two
        // (the old "{matching_cells} cells") surprised users who got a
        // much larger file than the match count implied.
        let granularity_note = if matches!(stats.filter_mode, scx_cloud::FilterMode::Shard) {
            " — shard-granular, rows not cell-filtered"
        } else {
            ""
        };
        println!(
            "Selective pull ({}) {} → {} ({}/{} shards; {} cells written, {} match the filter{}; {:.1} MB downloaded, {:.1} MB saved)",
            mode_label,
            source,
            dest.display(),
            stats.downloaded_shards,
            stats.total_shards,
            stats.output_cells,
            stats.matching_cells,
            granularity_note,
            stats.bytes_downloaded as f64 / 1_000_000.0,
            stats.bytes_saved as f64 / 1_000_000.0,
        );
        if !stats.omitted_section_types.is_empty() {
            let names: Vec<String> = stats
                .omitted_section_types
                .iter()
                .map(|st| format!("{st:?}"))
                .collect();
            println!("Note: omitted section types: {}", names.join(", "));
        }
    } else {
        let stats = rt.block_on(scx_cloud::pull(source, dest, opts))?;
        println!(
            "Pulled {} → {} ({} sections, {:.1} MB, {:.1} MB/s)",
            source,
            dest.display(),
            stats.sections_downloaded,
            stats.bytes_downloaded as f64 / 1_000_000.0,
            stats.throughput_mbps,
        );
    }
    Ok(())
}
