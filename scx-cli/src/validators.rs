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

/// Parse and validate that a string is a finite, non-negative `f64`.
///
/// Zero is allowed (a `--min-counts 0` no-op is meaningful); NaN and the
/// infinities are not, because they silently produce an all-keep or all-drop
/// mask rather than an error.
///
/// Use with `#[arg(value_parser = non_negative_f64)]`. Note that clap still
/// treats a bare leading-hyphen token as a flag, so a user must write
/// `--min-counts=-1` to reach this validator; the runtime guard in
/// `run_convert` covers non-CLI callers.
pub fn non_negative_f64(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|e| format!("invalid number '{s}': {e}"))?;
    if !v.is_finite() || v < 0.0 {
        return Err(format!(
            "value must be a finite non-negative number; got '{s}'"
        ));
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_negative_f64_accepts_zero_and_positive() {
        assert_eq!(non_negative_f64("0").unwrap(), 0.0);
        assert_eq!(non_negative_f64("5").unwrap(), 5.0);
        assert_eq!(non_negative_f64("12.5").unwrap(), 12.5);
    }

    #[test]
    fn non_negative_f64_rejects_negative_nan_and_inf() {
        for bad in ["-1", "-0.5", "nan", "inf", "-inf"] {
            let err = non_negative_f64(bad).unwrap_err();
            assert!(
                err.contains("non-negative"),
                "unhelpful message for {bad}: {err}"
            );
        }
    }

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
