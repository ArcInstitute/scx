// scx set-uns — Replace, or with `--merge` shallow-merge into, the `uns`
// block of an SCX file in place.
//
// Cheapest metadata mutation: appends one fresh `UnsBlob` section and
// repoints the catalog. The matrix (X / CSR / CSC) is never read or
// rewritten, so a pre-existing CSC sidecar stays valid. Without `--merge` the
// JSON replaces the block (`scx_ops::set_uns`); with it the JSON object's
// top-level keys are laid over the existing block (`scx_ops::update_uns`) and
// every other key survives.

use std::path::Path;

pub fn run_set_uns(file: &Path, uns: &Path, merge: bool) -> Result<(), Box<dyn std::error::Error>> {
    let f = std::fs::File::open(uns)
        .map_err(|e| format!("failed to open uns JSON '{}': {e}", uns.display()))?;
    let json: serde_json::Value = serde_json::from_reader(std::io::BufReader::new(f))
        .map_err(|e| format!("failed to parse uns JSON '{}': {e}", uns.display()))?;

    if merge {
        scx_ops::update_uns(file, &json)?;
        println!(
            "Merged {} uns key(s) into {} (O(uns), no matrix re-encode). \
             Run 'scx compact' to reclaim the orphaned section.",
            json.as_object().map_or(0, |m| m.len()),
            file.display()
        );
    } else {
        scx_ops::set_uns(file, &json)?;
        println!(
            "Updated uns on {} (O(uns), no matrix re-encode). \
             Run 'scx compact' to reclaim the orphaned section.",
            file.display()
        );
    }
    Ok(())
}
