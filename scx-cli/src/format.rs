//! Shared CLI formatting helpers.

/// Render a byte count as a human-readable binary-prefixed size
/// (`B`/`KiB`/`MiB`/`GiB`, powers of 1024). Single source for the byte-size
/// formatting used by `scx info`, `scx compact`, etc.
///
/// The suffixes are deliberately **binary** (`MiB`, not `MB`): these are
/// powers of 1024, and printing `MB` made `scx info` report `863.7 MB` for a
/// 905,627,007-byte file — a 5% apparent discrepancy against `ls -l` or a
/// quota (dogfood F10). It also matches the parse side, where
/// `--memory-budget` / `--memory-limit` / `--csc-memory-limit` accept binary
/// prefixes only and *reject* `KB`/`MB`/`GB` as ambiguous.
///
/// Byte counts that genuinely are decimal — the `scx pull` / `scx push`
/// transfer totals, which divide by 1_000_000 — must not route through here.
pub(crate) fn human_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;

    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each branch boundary, so a future refactor cannot silently shift a
    /// threshold by one unit.
    #[test]
    fn human_size_branch_boundaries() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(1023), "1023 B");
        assert_eq!(human_size(1024), "1.0 KiB");
        assert_eq!(human_size(1024 * 1024 - 1), "1024.0 KiB");
        assert_eq!(human_size(1024 * 1024), "1.0 MiB");
        assert_eq!(human_size(1024 * 1024 * 1024), "1.0 GiB");
    }

    /// F10: the reported file size was MiB labelled `MB`. The unit is the
    /// whole point of this helper, so pin it directly — including the
    /// substring check, which is what catches a partial revert (e.g. `GB`
    /// fixed but `MB` missed).
    #[test]
    fn human_size_never_emits_decimal_suffixes() {
        // 905,627,007 B is the exact file from the dogfood report: 863.7 MiB,
        // which read as a 5% shortfall against `ls -l` when labelled "MB".
        assert_eq!(human_size(905_627_007), "863.7 MiB");
        for bytes in [
            0u64,
            512,
            1024,
            900_000,
            905_627_007,
            5 * 1024 * 1024 * 1024,
        ] {
            let rendered = human_size(bytes);
            for decimal in ["KB", "MB", "GB"] {
                assert!(
                    !rendered.contains(decimal),
                    "human_size({bytes}) = {rendered:?} must not use the decimal \
                     suffix {decimal:?}; these are powers of 1024"
                );
            }
        }
    }
}
