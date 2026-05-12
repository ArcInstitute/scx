use std::path::Path;

pub fn run_push(
    source: &Path,
    dest: &str,
    parallelism: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    let opts = scx_cloud::PushOptions { parallelism };

    let stats = rt.block_on(scx_cloud::push(source, dest, opts))?;

    println!(
        "Pushed {} → {} ({} objects, {:.1} MB, {:.1} MB/s)",
        source.display(),
        dest,
        stats.sections_uploaded,
        stats.bytes_uploaded as f64 / 1_000_000.0,
        stats.throughput_mbps,
    );
    Ok(())
}
