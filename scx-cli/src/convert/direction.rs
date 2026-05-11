use std::path::Path;

pub(crate) fn determine_convert_direction(
    from: Option<&str>,
    to: Option<&str>,
    input: &Path,
) -> Result<&'static str, String> {
    if let Some(f) = from {
        return match f {
            "mtx" => Ok("mtx_to_scx"),
            "h5ad" => Ok("h5ad_to_scx"),
            "h5mu" => Ok("h5mu_to_scx"),
            "10x" => Ok("tenx_to_scx"),
            other => Err(format!(
                "Unknown --from value: '{other}'. Use mtx, h5ad, h5mu, or 10x."
            )),
        };
    }
    if let Some(t) = to {
        return match t {
            "mtx" => Ok("scx_to_mtx"),
            "h5ad" => Ok("scx_to_h5ad"),
            "h5mu" => Ok("scx_to_h5mu"),
            other => Err(format!(
                "Unknown --to value: '{other}'. Use mtx, h5ad, or h5mu."
            )),
        };
    }
    if input.is_dir() {
        return Ok("mtx_to_scx");
    }
    match input.extension().and_then(|e| e.to_str()) {
        Some("h5ad") => Ok("h5ad_to_scx"),
        Some("h5mu") => Ok("h5mu_to_scx"),
        Some("h5") => Ok("tenx_to_scx"),
        Some("scx") => Ok("scx_to_h5ad"),
        _ => Err("Cannot determine conversion direction. Use --from/--to flags.".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_from_h5ad_with_scx_extension() {
        let p = Path::new("input.scx");
        assert_eq!(
            determine_convert_direction(Some("h5ad"), None, p).unwrap(),
            "h5ad_to_scx"
        );
    }

    #[test]
    fn explicit_from_tenx_with_scx_extension() {
        let p = Path::new("input.scx");
        assert_eq!(
            determine_convert_direction(Some("10x"), None, p).unwrap(),
            "tenx_to_scx"
        );
    }

    #[test]
    fn explicit_to_h5ad_with_scx_extension() {
        let p = Path::new("input.scx");
        assert_eq!(
            determine_convert_direction(None, Some("h5ad"), p).unwrap(),
            "scx_to_h5ad"
        );
    }

    #[test]
    fn explicit_to_h5ad_with_unrelated_extension() {
        let p = Path::new("input.bin");
        assert_eq!(
            determine_convert_direction(None, Some("h5ad"), p).unwrap(),
            "scx_to_h5ad"
        );
    }

    #[test]
    fn explicit_from_overrides_extension_sniff() {
        let p = Path::new("input.h5ad");
        assert_eq!(
            determine_convert_direction(Some("mtx"), None, p).unwrap(),
            "mtx_to_scx"
        );
    }

    #[test]
    fn from_takes_precedence_over_to() {
        let p = Path::new("input.bin");
        assert_eq!(
            determine_convert_direction(Some("h5ad"), Some("mtx"), p).unwrap(),
            "h5ad_to_scx"
        );
    }

    #[test]
    fn explicit_from_h5mu() {
        let p = Path::new("input.bin");
        assert_eq!(
            determine_convert_direction(Some("h5mu"), None, p).unwrap(),
            "h5mu_to_scx"
        );
    }

    #[test]
    fn explicit_from_mtx() {
        let p = Path::new("input.bin");
        assert_eq!(
            determine_convert_direction(Some("mtx"), None, p).unwrap(),
            "mtx_to_scx"
        );
    }

    #[test]
    fn explicit_to_mtx() {
        let p = Path::new("input.scx");
        assert_eq!(
            determine_convert_direction(None, Some("mtx"), p).unwrap(),
            "scx_to_mtx"
        );
    }

    #[test]
    fn explicit_to_h5mu() {
        let p = Path::new("input.scx");
        assert_eq!(
            determine_convert_direction(None, Some("h5mu"), p).unwrap(),
            "scx_to_h5mu"
        );
    }

    #[test]
    fn auto_detect_h5ad_extension() {
        let p = Path::new("dataset.h5ad");
        assert_eq!(
            determine_convert_direction(None, None, p).unwrap(),
            "h5ad_to_scx"
        );
    }

    #[test]
    fn auto_detect_h5mu_extension() {
        let p = Path::new("dataset.h5mu");
        assert_eq!(
            determine_convert_direction(None, None, p).unwrap(),
            "h5mu_to_scx"
        );
    }

    #[test]
    fn auto_detect_tenx_extension() {
        let p = Path::new("dataset.h5");
        assert_eq!(
            determine_convert_direction(None, None, p).unwrap(),
            "tenx_to_scx"
        );
    }

    #[test]
    fn auto_detect_scx_extension() {
        let p = Path::new("dataset.scx");
        assert_eq!(
            determine_convert_direction(None, None, p).unwrap(),
            "scx_to_h5ad"
        );
    }

    #[test]
    fn auto_detect_directory_is_mtx() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            determine_convert_direction(None, None, dir.path()).unwrap(),
            "mtx_to_scx"
        );
    }

    #[test]
    fn unknown_extension_errors() {
        let p = Path::new("dataset.txt");
        let err = determine_convert_direction(None, None, p).unwrap_err();
        assert!(err.contains("Cannot determine conversion direction"));
    }

    #[test]
    fn missing_extension_errors() {
        let p = Path::new("dataset");
        let err = determine_convert_direction(None, None, p).unwrap_err();
        assert!(err.contains("Cannot determine conversion direction"));
    }

    #[test]
    fn unknown_from_value_errors() {
        let p = Path::new("input.scx");
        let err = determine_convert_direction(Some("zarr"), None, p).unwrap_err();
        assert!(err.contains("Unknown --from value"));
    }

    #[test]
    fn unknown_to_value_errors() {
        let p = Path::new("input.scx");
        let err = determine_convert_direction(None, Some("parquet"), p).unwrap_err();
        assert!(err.contains("Unknown --to value"));
    }
}
