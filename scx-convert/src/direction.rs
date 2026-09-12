use std::path::Path;

pub fn determine_convert_direction(
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

/// Every direction [`determine_convert_direction`] can return, paired with
/// whether it has a streaming (bounded peak-RSS) implementation.
///
/// Deliberately **one** table rather than a list of names plus a separate
/// `match`: that way registering a direction *structurally* forces the
/// streaming decision, because there is no way to add a row without writing
/// its `bool`. An earlier shape — a names-only array plus a `match` with a
/// `_ => false` arm — could not do that: a new name simply fell through the
/// catch-all and was silently classified as non-streaming, and no test could
/// tell that apart from a deliberate `false`.
///
/// `true` = h5ad → scx (Phase 0/1/2), h5mu → scx (Phase 3), 10x → scx
/// (OPT-CONVERT-9) and scx → h5ad / h5mu (Phase 8), which each have a
/// shard-at-a-time path alongside the legacy materializing one, selected by the
/// resolved `--stream`. `false` = a single materializing path with nothing to
/// select: `scx-mtx` reads the whole MTX directory into memory
/// (`read_mtx_directory`) and `write_scx_to_mtx` assembles the whole CSR.
///
/// **Limitation, stated precisely.** Rust cannot reflect over the match arms
/// of [`determine_convert_direction`], so a new arm there that is never
/// registered *here* is invisible to this table and to the tests below — they
/// keep this table and the flag spellings mutually consistent, they do not
/// detect an unregistered arm. Closing that last gap needs a
/// `ConvertDirection` enum, which would make every direction guard a
/// compiler-checked `match`.
pub const CONVERT_DIRECTIONS: [(&str, bool); 7] = [
    ("mtx_to_scx", false),
    ("h5ad_to_scx", true),
    ("h5mu_to_scx", true),
    ("tenx_to_scx", true),
    ("scx_to_mtx", false),
    ("scx_to_h5ad", true),
    ("scx_to_h5mu", true),
];

/// Whether `direction` has a streaming (bounded peak-RSS) implementation.
pub fn direction_supports_streaming(direction: &str) -> bool {
    CONVERT_DIRECTIONS
        .iter()
        .find_map(|&(name, streams)| (name == direction).then_some(streams))
        // An unregistered direction is conservatively materializing. This is
        // reachable only via the limitation documented on CONVERT_DIRECTIONS;
        // materializing is the safe answer because every direction has a
        // materializing path, whereas claiming a streaming path that does not
        // exist would route into a `None` branch downstream.
        .unwrap_or(false)
}

/// Resolve the effective streaming mode for `direction`.
///
/// `requested` is the `--stream` flag as the user gave it: `None` when the flag
/// was absent, `Some(v)` when it was passed. Absent means "do whatever this
/// direction does natively", which is why the flag must not carry a clap
/// default — a default `true` cannot be told apart from an explicit one, and
/// rejecting on it makes every non-streaming direction unreachable.
///
/// `Some(false)` is always honoured: on a non-streaming direction the request
/// is already satisfied, so it resolves to `false` rather than erroring. Only
/// `Some(true)` on a direction with no streaming path is an error, because that
/// is the one request the pipeline cannot fulfil.
pub fn resolve_stream(requested: Option<bool>, direction: &str) -> Result<bool, String> {
    let supported = direction_supports_streaming(direction);
    match requested {
        None => Ok(supported),
        Some(false) => Ok(false),
        Some(true) if supported => Ok(true),
        Some(true) => Err(format!(
            "--stream is not supported for direction '{direction}'. Streaming is implemented \
             only for h5ad → scx, h5mu → scx, 10x → scx, scx → h5ad and scx → h5mu; mtx ↔ scx \
             has a single materializing path. Re-run without --stream — the flag is not needed \
             for this direction."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The canonical `--from` / `--to` spelling for each direction.
    const DIRECTION_FLAGS: [(Option<&str>, Option<&str>, &str); 7] = [
        (Some("mtx"), None, "mtx_to_scx"),
        (Some("h5ad"), None, "h5ad_to_scx"),
        (Some("h5mu"), None, "h5mu_to_scx"),
        (Some("10x"), None, "tenx_to_scx"),
        (None, Some("mtx"), "scx_to_mtx"),
        (None, Some("h5ad"), "scx_to_h5ad"),
        (None, Some("h5mu"), "scx_to_h5mu"),
    ];

    const STREAMING_DIRECTIONS: [&str; 5] = [
        "h5ad_to_scx",
        "h5mu_to_scx",
        "tenx_to_scx",
        "scx_to_h5ad",
        "scx_to_h5mu",
    ];

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

    // --- `--stream` resolution -------------------------------------------
    //
    // What is enforced where, so the guarantee is not overstated:
    //
    //   * That every registered direction has a streaming decision is
    //     *structural* — `CONVERT_DIRECTIONS` pairs each name with its bool,
    //     so a row cannot be added without making the call. No test needed.
    //   * That the table matches the flags users actually type is enforced by
    //     `every_direction_is_reachable_from_flags` below.
    //   * That the bools are the *right* bools is pinned by
    //     `streaming_classification_matches_the_implementations`.
    //
    // Not enforced by any of this: a new arm in `determine_convert_direction`
    // that is never registered in `CONVERT_DIRECTIONS`. Rust cannot reflect
    // over match arms; that gap needs a `ConvertDirection` enum.

    #[test]
    fn every_direction_is_reachable_from_flags() {
        let p = Path::new("input.bin");
        let mut reached = BTreeSet::new();
        for (from, to, expected) in DIRECTION_FLAGS {
            let got = determine_convert_direction(from, to, p).unwrap();
            assert_eq!(got, expected, "--from {from:?} --to {to:?}");
            reached.insert(got);
        }
        let registered: BTreeSet<&str> = CONVERT_DIRECTIONS.iter().map(|&(n, _)| n).collect();
        assert_eq!(
            reached, registered,
            "CONVERT_DIRECTIONS must be exactly the set determine_convert_direction returns \
             for the canonical --from/--to spellings"
        );
    }

    #[test]
    fn streaming_classification_matches_the_implementations() {
        // A value pin, not a totality lock: totality is structural (see the
        // note above). This catches a bool flipped in the table by mistake.
        let streaming: BTreeSet<&str> = CONVERT_DIRECTIONS
            .iter()
            .filter(|&&(_, streams)| streams)
            .map(|&(n, _)| n)
            .collect();
        let expected: BTreeSet<&str> = STREAMING_DIRECTIONS.into_iter().collect();
        assert_eq!(
            streaming, expected,
            "h5ad/h5mu ↔ scx and 10x → scx have a streaming path; if that changed, \
             update STREAMING_DIRECTIONS deliberately"
        );
    }

    #[test]
    fn direction_supports_streaming_reads_the_table() {
        for (name, streams) in CONVERT_DIRECTIONS {
            assert_eq!(direction_supports_streaming(name), streams, "{name}");
        }
        // An unregistered direction is conservatively non-streaming rather
        // than a panic — `direction_supports_streaming` is `pub` and takes an
        // arbitrary `&str`.
        assert!(!direction_supports_streaming("zarr_to_scx"));
    }

    #[test]
    fn resolve_stream_absent_follows_direction() {
        for (direction, _) in CONVERT_DIRECTIONS {
            let expected = STREAMING_DIRECTIONS.contains(&direction);
            assert_eq!(
                resolve_stream(None, direction).unwrap(),
                expected,
                "absent --stream on '{direction}'"
            );
        }
    }

    #[test]
    fn resolve_stream_explicit_true_rejected_on_non_streaming() {
        for (direction, _) in CONVERT_DIRECTIONS {
            let result = resolve_stream(Some(true), direction);
            if STREAMING_DIRECTIONS.contains(&direction) {
                assert!(result.unwrap(), "--stream on '{direction}' should stream");
            } else {
                let err = result.unwrap_err();
                assert!(err.contains("--stream"), "{err}");
                assert!(err.contains(direction), "{err}");
                assert!(
                    err.contains("Re-run without"),
                    "the message must name the remedy: {err}"
                );
            }
        }
    }

    #[test]
    fn resolve_stream_explicit_false_is_always_accepted() {
        // Deliberate: on a non-streaming direction `--stream=false` is already
        // satisfied, so it is a no-op rather than an error. Scripts written
        // against the older CLI (which rejected the *default* `true` and so
        // forced users to pass `--stream=false`) keep working.
        for (direction, _) in CONVERT_DIRECTIONS {
            assert!(
                !resolve_stream(Some(false), direction).unwrap(),
                "--stream=false on '{direction}'"
            );
        }
    }
}
