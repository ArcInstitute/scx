use std::path::Path;

use scx_format::checksum::blake3_hash;
use scx_format::reader::ScxReader;

/// Validate all section checksums. Returns true if all pass.
///
/// `ScxReader::open()` already cross-checks `header.n_modalities`
/// against the parsed `ModalityTable`'s embedded count and verifies
/// the modality table's BLAKE3 truncated checksum during parse — a
/// validation failure on either surfaces as a parse error here. The
/// per-section catalog walk below is naturally modality-aware: every
/// per-modality CSR / CSC / var / obsm / layer / uns shard appears
/// in `catalog.entries` and gets BLAKE3-verified just like any
/// other section.
pub fn run_validate(path: &Path, verbose: bool) -> Result<bool, Box<dyn std::error::Error>> {
    let reader = ScxReader::open(path)?;
    let catalog = reader.catalog();
    let header = reader.header();

    let mut all_passed = true;

    for entry in &catalog.entries {
        let bytes = reader.section_bytes(entry)?;
        let computed = blake3_hash(bytes);
        let passed = computed == entry.checksum;

        let icon = if passed { "OK" } else { "FAIL" };
        println!("[{icon}] {}", entry.name);

        if verbose {
            println!("       expected: {}", hex_str(&entry.checksum));
            println!("       computed: {}", hex_str(&computed));
        }

        if !passed {
            all_passed = false;
        }
    }

    // Phase F.2: emit an explicit modality-table status line so users
    // can see at a glance that the cross-checks ran. The actual
    // checksum + n_modalities cross-check happened during open().
    if let Some(table) = reader.modality_table() {
        println!(
            "[OK] modality_table ({} modalities, header.n_modalities={})",
            table.len(),
            header.n_modalities,
        );
    }

    if all_passed {
        let n_sections = catalog.entries.len()
            + if reader.modality_table().is_some() {
                1
            } else {
                0
            };
        println!("\nAll {n_sections} sections passed.");
    } else {
        println!("\nValidation FAILED.");
    }

    Ok(all_passed)
}

fn hex_str(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
