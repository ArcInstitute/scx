//! Shared CLI formatting helpers.

/// Render a byte count as a human-readable binary-prefixed size
/// (`B`/`KB`/`MB`/`GB`, powers of 1024). Single source for the byte-size
/// formatting used by `scx info`, `scx compact`, etc.
pub(crate) fn human_size(bytes: u64) -> String {
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
