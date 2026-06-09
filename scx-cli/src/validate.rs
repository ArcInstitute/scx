use std::path::Path;

use scx_format::checksum::blake3_hash;
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;

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
pub fn run_validate(
    path: &Path,
    verbose: bool,
    deep: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let reader = ScxReader::open(path)?;
    let catalog = reader.catalog();
    let header = reader.header();

    let mut all_passed = true;
    let mut n_checks = 0usize;

    for entry in &catalog.entries {
        let bytes = reader.section_bytes(entry)?;
        let computed = blake3_hash(bytes);
        let passed = computed == entry.checksum;

        n_checks += 1;
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
        n_checks += 1;
    }

    if deep {
        println!("\nDeep checks:");
        // The canonical-CSR invariant is a v3 guarantee only. Pre-v3 files may
        // legitimately carry unsorted / unsummed shards, so checking them would
        // be a false failure — skip the canonical-CSR loop below v3. (Decode
        // sidecars only exist in v3 files, so that loop is gated implicitly.)
        if header.format_version >= 3 {
            for entry in &catalog.entries {
                if matches!(
                    entry.section_type,
                    SectionType::CsrShard | SectionType::LayerCsrShard | SectionType::ObspCsrShard
                ) {
                    n_checks += 1;
                    let result = reader.validate_canonical_csr_entry(entry);
                    let passed = result.is_ok();
                    let icon = if passed { "OK" } else { "FAIL" };
                    println!("[{icon}] canonical-csr {}", entry.name);
                    if verbose {
                        if let Err(err) = &result {
                            println!("       error: {err}");
                        }
                    }
                    if !passed {
                        all_passed = false;
                    }
                }
            }
        } else {
            println!(
                "canonical-CSR checks skipped: file is format_version {} (< 3)",
                header.format_version
            );
        }
        for entry in &catalog.entries {
            if entry.section_type == SectionType::DecodeMetadataShard {
                n_checks += 1;
                let result = reader.validate_decode_sidecar_entry(entry);
                let passed = result.is_ok();
                let icon = if passed { "OK" } else { "FAIL" };
                println!("[{icon}] decode-sidecar {}", entry.name);
                if verbose {
                    if let Err(err) = &result {
                        println!("       error: {err}");
                    }
                }
                if !passed {
                    all_passed = false;
                }
            }
        }
    }

    if all_passed {
        println!("\nAll {n_checks} checks passed.");
    } else {
        println!("\nValidation FAILED.");
    }

    Ok(all_passed)
}

fn hex_str(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
