// scx delete — Logically delete cells matching a predicate.

use std::path::Path;

use scx_engine::{evaluate, parse_predicate};
use scx_format::reader::ScxReader;

pub fn run_delete(
    file: &Path,
    filter: &str,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Open file and read obs metadata
    let reader = ScxReader::open(file)?;
    let obs_schema = reader.read_obs_schema()?;
    let obs_batch = reader.read_obs()?;

    // Parse and evaluate predicate
    let predicate = parse_predicate(filter, &obs_schema, "obs")?;
    let mask = evaluate(&predicate, &obs_batch)?;

    // Collect matching global cell indices
    let cell_indices: Vec<u64> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            if v == Some(true) {
                Some(i as u64)
            } else {
                None
            }
        })
        .collect();

    let n_matching = cell_indices.len();

    if n_matching == 0 {
        println!("0 cells matched '{}', nothing to delete.", filter);
        return Ok(());
    }

    if dry_run {
        println!("Would delete {} cells matching '{}'", n_matching, filter);
        return Ok(());
    }

    // Drop reader before modifying the file
    drop(reader);

    // Call scx_ops::mark_deleted
    let total_deleted = scx_ops::mark_deleted(file, &cell_indices)?;

    println!(
        "Deleted {} cells matching '{}' ({} total deleted)",
        n_matching, filter, total_deleted
    );

    Ok(())
}
