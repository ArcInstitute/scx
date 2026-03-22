use std::path::Path;

pub fn run_pull(
    source: &str,
    dest: &Path,
    parallelism: usize,
    cloud_ready: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    let opts = scx_cloud::PullOptions {
        parallelism,
        reorder_buffer: 4,
        cloud_ready,
    };

    let stats = rt.block_on(scx_cloud::pull(source, dest, opts))?;

    println!(
        "Pulled {} → {} ({} sections, {:.1} MB, {:.1} MB/s)",
        source,
        dest.display(),
        stats.sections_downloaded,
        stats.bytes_downloaded as f64 / 1_000_000.0,
        stats.throughput_mbps,
    );
    Ok(())
}
