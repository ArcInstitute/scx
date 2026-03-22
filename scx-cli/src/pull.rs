use std::path::Path;

pub fn run_pull(
    source: &str,
    dest: &Path,
    parallelism: usize,
    cloud_ready: bool,
    filter: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    let opts = scx_cloud::PullOptions {
        parallelism,
        reorder_buffer: 4,
        cloud_ready,
    };

    if let Some(filter_expr) = filter {
        let stats = rt.block_on(scx_cloud::pull_filtered(source, dest, filter_expr, opts))?;
        println!(
            "Selective pull {} → {} ({}/{} shards, {} cells, {:.1} MB downloaded, {:.1} MB saved)",
            source,
            dest.display(),
            stats.downloaded_shards,
            stats.total_shards,
            stats.matching_cells,
            stats.bytes_downloaded as f64 / 1_000_000.0,
            stats.bytes_saved as f64 / 1_000_000.0,
        );
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
