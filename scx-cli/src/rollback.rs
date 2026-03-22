// scx rollback — Revert to a previous manifest version.

use std::path::Path;

pub fn run_rollback(file: &Path, to_seq: Option<u64>) -> Result<(), Box<dyn std::error::Error>> {
    match to_seq {
        Some(n) => {
            scx_ops::rollback_to(file, n)?;
            println!("Rolled back {} to manifest sequence {}", file.display(), n);
        }
        None => {
            scx_ops::rollback(file)?;
            // Read back the new header to report the sequence number
            let reader = scx_format::reader::ScxReader::open(file)?;
            let seq = reader.header().manifest_sequence;
            println!(
                "Rolled back {} to manifest sequence {}",
                file.display(),
                seq
            );
        }
    }

    println!("Run 'scx compact' to permanently discard later versions.");

    Ok(())
}
