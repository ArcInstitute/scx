// Memory-budget / size-string parser shared across the workspace.
//
// One parser backs every size knob users type: `scx convert
// --memory-budget`, `scx build-csc --memory-limit`, and the pyscx
// `memory_budget=` kwargs. It accepts a bare byte count or a
// binary-prefixed size — `K`/`M`/`G`/`T` (powers of 1024 by
// convention, matching `dd` / `du -h`) or explicit `KiB`/`MiB`/`GiB`/
// `TiB`. Decimal suffixes (`KB`/`MB`/`GB`/`TB`) are rejected to avoid
// the usual 1000-vs-1024 ambiguity — pass the exact byte count for
// decimal magnitudes. It lives in `scx-format` (rather than
// `scx-convert`) so sibling crates like `scx-ops` can share it without
// a dependency cycle.

/// Namespacing struct; constructors live as associated functions.
pub struct MemoryBudget;

impl MemoryBudget {
    /// Parse a memory budget string into a byte count.
    ///
    /// Accepted forms (case-insensitive, optional whitespace
    /// between number and suffix):
    ///
    /// - bare bytes: `"1024"`, `"0"`
    /// - binary suffixes: `"4K"`, `"2M"`, `"8G"`, `"1T"` (treated as
    ///   `4 * 1024`, `2 * 1024^2`, etc. — matches `dd` and `du -h`)
    /// - explicit binary suffixes: `"4KiB"`, `"2MiB"`, `"8GiB"`,
    ///   `"1TiB"`
    ///
    /// Decimal suffixes (`KB`/`MB`/`GB`/`TB`) are rejected as
    /// ambiguous. Returns a descriptive error string for empty input,
    /// malformed numbers, unrecognised suffixes, or overflow. The error
    /// type is `String` (not a crate error enum) so the parser is
    /// usable from any caller; CLI callers wrap it via `?` /
    /// `Box::<dyn Error>::from`.
    pub fn parse(s: &str) -> Result<u64, String> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err("memory budget is empty; expected e.g. '2GiB'".into());
        }

        let lower = trimmed.to_ascii_lowercase();
        let (num_str, multiplier) = if let Some(rest) = lower.strip_suffix("tib") {
            (rest, 1u64 << 40)
        } else if let Some(rest) = lower.strip_suffix("gib") {
            (rest, 1u64 << 30)
        } else if let Some(rest) = lower.strip_suffix("mib") {
            (rest, 1u64 << 20)
        } else if let Some(rest) = lower.strip_suffix("kib") {
            (rest, 1u64 << 10)
        } else if let Some(rest) = lower.strip_suffix("kb") {
            return Err(format!(
                "decimal suffix in '{}' is ambiguous; use 'KiB' or a bare byte count",
                rest.trim()
            ));
        } else if let Some(rest) = lower.strip_suffix("mb") {
            return Err(format!(
                "decimal suffix in '{}' is ambiguous; use 'MiB' or a bare byte count",
                rest.trim()
            ));
        } else if let Some(rest) = lower.strip_suffix("gb") {
            return Err(format!(
                "decimal suffix in '{}' is ambiguous; use 'GiB' or a bare byte count",
                rest.trim()
            ));
        } else if let Some(rest) = lower.strip_suffix("tb") {
            return Err(format!(
                "decimal suffix in '{}' is ambiguous; use 'TiB' or a bare byte count",
                rest.trim()
            ));
        } else if let Some(rest) = lower.strip_suffix('t') {
            (rest, 1u64 << 40)
        } else if let Some(rest) = lower.strip_suffix('g') {
            (rest, 1u64 << 30)
        } else if let Some(rest) = lower.strip_suffix('m') {
            (rest, 1u64 << 20)
        } else if let Some(rest) = lower.strip_suffix('k') {
            (rest, 1u64 << 10)
        } else if let Some(rest) = lower.strip_suffix('b') {
            (rest, 1u64)
        } else {
            (lower.as_str(), 1u64)
        };

        let num_str = num_str.trim();
        if num_str.is_empty() {
            return Err(format!("no number in memory budget '{}'", trimmed));
        }

        let n: u64 = num_str.parse().map_err(|_| {
            format!(
                "invalid memory budget number '{}' in '{}'",
                num_str, trimmed
            )
        })?;

        n.checked_mul(multiplier)
            .ok_or_else(|| format!("memory budget '{}' overflows u64", trimmed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_bytes() {
        assert_eq!(MemoryBudget::parse("0").unwrap(), 0);
        assert_eq!(MemoryBudget::parse("1024").unwrap(), 1024);
        assert_eq!(MemoryBudget::parse(" 17 ").unwrap(), 17);
    }

    #[test]
    fn short_binary_suffixes() {
        assert_eq!(MemoryBudget::parse("4K").unwrap(), 4 * 1024);
        assert_eq!(MemoryBudget::parse("2m").unwrap(), 2 * 1024 * 1024);
        assert_eq!(
            MemoryBudget::parse("8G").unwrap(),
            8u64 * 1024 * 1024 * 1024
        );
        assert_eq!(MemoryBudget::parse("1t").unwrap(), 1u64 << 40);
    }

    #[test]
    fn explicit_binary_suffixes() {
        assert_eq!(MemoryBudget::parse("4KiB").unwrap(), 4 * 1024);
        assert_eq!(MemoryBudget::parse("2 MiB").unwrap(), 2 * 1024 * 1024);
        assert_eq!(MemoryBudget::parse("8 GiB").unwrap(), 8u64 << 30);
        assert_eq!(MemoryBudget::parse("1TiB").unwrap(), 1u64 << 40);
    }

    #[test]
    fn decimal_suffix_rejected() {
        assert!(MemoryBudget::parse("4KB").is_err());
        assert!(MemoryBudget::parse("2MB").is_err());
        assert!(MemoryBudget::parse("8GB").is_err());
        assert!(MemoryBudget::parse("1TB").is_err());
    }

    #[test]
    fn empty_or_malformed() {
        assert!(MemoryBudget::parse("").is_err());
        assert!(MemoryBudget::parse("  ").is_err());
        assert!(MemoryBudget::parse("MiB").is_err());
        assert!(MemoryBudget::parse("hello").is_err());
        assert!(MemoryBudget::parse("4.5GiB").is_err());
        assert!(MemoryBudget::parse("-1").is_err());
    }

    #[test]
    fn overflow_returns_err() {
        // u64::MAX TiB clearly overflows when scaled by 1<<40.
        let huge = format!("{}TiB", u64::MAX);
        assert!(MemoryBudget::parse(&huge).is_err());
    }
}
