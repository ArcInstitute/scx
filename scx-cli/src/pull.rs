use std::path::Path;

pub fn run_pull(
    source: &str,
    dest: &Path,
    parallelism: usize,
    cloud_ready: bool,
    filter: Option<&str>,
    filter_mode: &str,
) -> Result<(), Box<dyn std::error::Error>> {
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
        println!(
            "Selective pull ({}) {} → {} ({}/{} shards, {} cells, {:.1} MB downloaded, {:.1} MB saved)",
            mode_label,
            source,
            dest.display(),
            stats.downloaded_shards,
            stats.total_shards,
            stats.matching_cells,
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
