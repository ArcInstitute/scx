// scx set-uns — Replace the `uns` block of an SCX file in place.
//
// Cheapest metadata mutation: appends one fresh `UnsBlob` section and
// repoints the catalog. The matrix (X / CSR / CSC) is never read or
// rewritten, so a pre-existing CSC sidecar stays valid. Replace semantics,
// not merge (read-modify-write in the caller for a shallow merge).

use std::path::Path;

pub fn run_set_uns(file: &Path, uns: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let f = std::fs::File::open(uns)
        .map_err(|e| format!("failed to open uns JSON '{}': {e}", uns.display()))?;
    let json: serde_json::Value = serde_json::from_reader(std::io::BufReader::new(f))
        .map_err(|e| format!("failed to parse uns JSON '{}': {e}", uns.display()))?;

    scx_ops::set_uns(file, &json)?;
    println!(
        "Updated uns on {} (O(uns), no matrix re-encode). \
         Run 'scx compact' to reclaim the orphaned section.",
        file.display()
    );
    Ok(())
}
