use std::path::Path;

use scx_format::checksum::blake3_hash;
use scx_format::reader::ScxReader;

/// Validate all section checksums. Returns true if all pass.
pub fn run_validate(path: &Path, verbose: bool) -> Result<bool, Box<dyn std::error::Error>> {
    let reader = ScxReader::open(path)?;
    let catalog = reader.catalog();

    let mut all_passed = true;

    for entry in &catalog.entries {
        let bytes = reader.section_bytes(entry);
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

    if all_passed {
        println!("\nAll {} sections passed.", catalog.entries.len());
    } else {
        println!("\nValidation FAILED.");
    }

    Ok(all_passed)
}

fn hex_str(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
