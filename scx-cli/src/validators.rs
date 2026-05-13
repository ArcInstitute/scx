// Centralized CLI argument validators.
//
// Provides type-safe value-parser functions for clap, so that shared
// constraints (e.g., "must be a positive integer") are expressed once
// rather than repeated across subcommands.

/// Parse and validate that a string is a positive (> 0) `u32`.
///
/// Use with `#[arg(value_parser = positive_u32)]`.
pub fn positive_u32(s: &str) -> Result<u32, String> {
    let v: u32 = s
        .parse()
        .map_err(|e| format!("invalid u32 value '{s}': {e}"))?;
    if v == 0 {
        return Err("value must be greater than 0".to_string());
    }
    Ok(v)
}

/// Parse and validate that a string is a positive (> 0) `usize`.
///
/// Use with `#[arg(value_parser = positive_usize)]`.
pub fn positive_usize(s: &str) -> Result<usize, String> {
    let v: usize = s
        .parse()
        .map_err(|e| format!("invalid integer value '{s}': {e}"))?;
    if v == 0 {
        return Err("value must be greater than 0".to_string());
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_u32_accepts_valid() {
        assert_eq!(positive_u32("1").unwrap(), 1);
        assert_eq!(positive_u32("10000").unwrap(), 10000);
        assert_eq!(positive_u32("4294967295").unwrap(), u32::MAX);
    }

    #[test]
    fn positive_u32_rejects_zero() {
        assert!(positive_u32("0").is_err());
    }

    #[test]
    fn positive_u32_rejects_negative() {
        assert!(positive_u32("-1").is_err());
    }

    #[test]
    fn positive_u32_rejects_overflow() {
        assert!(positive_u32("4294967296").is_err());
    }

    #[test]
    fn positive_usize_accepts_valid() {
        assert_eq!(positive_usize("1").unwrap(), 1);
        assert_eq!(positive_usize("9999").unwrap(), 9999);
    }

    #[test]
    fn positive_usize_rejects_zero() {
        assert!(positive_usize("0").is_err());
    }
}
