//! Explicit `prefer_format` dispatch for CSC consumers.
//!
//! Every CSC entry point in pyscx (and any future Rust caller) routes
//! through [`require_csc`] so the "CSC requested but not available"
//! error message is centralized here.

use crate::error::AccelError;
use scx_format_io::ColumnShardSource;

/// Caller-requested column format. There is intentionally no `Auto`
/// variant — CSC dispatch is explicit-opt-in by design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreferFormat {
    Csr,
    Csc,
}

impl PreferFormat {
    /// Parse the user-facing string form.
    ///
    /// `"csr"` and `"csc"` are accepted (case-insensitive on the
    /// pyscx side; we only accept lowercase here — pyscx normalises
    /// before calling). Any other value returns
    /// `AccelError::InvalidInput` so the user gets the same error
    /// regardless of which entry point they passed it to.
    pub fn parse(s: &str) -> Result<Self, AccelError> {
        match s {
            "csr" => Ok(PreferFormat::Csr),
            "csc" => Ok(PreferFormat::Csc),
            other => Err(AccelError::InvalidInput(format!(
                "invalid prefer_format {other:?}; expected 'csr' or 'csc'"
            ))),
        }
    }
}

/// Resolve `prefer_format` against the dataset's CSC capability.
///
/// - `prefer == Csc` and `source.is_some()` → returns the source.
/// - `prefer == Csc` and `source.is_none()` → returns
///   [`AccelError::CscRequestedNotAvailable`] with `reason` describing
///   the missing capability (caller's responsibility to populate this
///   string with the most specific cause they know — typical messages
///   name the missing CSC sidecar, a non-column-local transform in the
///   chain, or an active row deletion vector).
/// - `prefer == Csr` → returns [`AccelError::CscNotRequested`]
///   (sentinel; the caller's match arm should immediately fall through
///   to the existing CSR path).
pub fn require_csc<'a>(
    prefer: PreferFormat,
    source: Option<&'a dyn ColumnShardSource>,
    reason: &str,
) -> Result<&'a dyn ColumnShardSource, AccelError> {
    match prefer {
        PreferFormat::Csr => Err(AccelError::CscNotRequested),
        PreferFormat::Csc => {
            source.ok_or_else(|| AccelError::CscRequestedNotAvailable(reason.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_format_io::Result as ScxResult;
    use scx_sparse::ScxCsc;
    use std::ops::Range;

    /// Minimal stub used to exercise the dispatch helper without
    /// constructing a real `BackedCscReader`. The trait methods just
    /// return canned `unimplemented!()` because the dispatch test only
    /// looks at the `Option<&dyn ColumnShardSource>` discriminator.
    struct StubColumnSource;

    impl ColumnShardSource for StubColumnSource {
        fn n_csc_shards(&self) -> usize {
            0
        }
        fn n_obs(&self) -> usize {
            0
        }
        fn n_vars(&self) -> usize {
            0
        }
        fn read_csc_shard(&self, _idx: usize) -> ScxResult<ScxCsc> {
            unimplemented!("stub")
        }
        fn read_csc_columns(&self, _r: Range<u32>) -> ScxResult<ScxCsc> {
            unimplemented!("stub")
        }
        fn csc_shard_col_range(&self, _idx: usize) -> Option<(u32, u32)> {
            None
        }
    }

    #[test]
    fn parse_accepts_csr_and_csc() {
        assert_eq!(PreferFormat::parse("csr").unwrap(), PreferFormat::Csr);
        assert_eq!(PreferFormat::parse("csc").unwrap(), PreferFormat::Csc);
    }

    #[test]
    fn parse_rejects_other_values() {
        let err = PreferFormat::parse("auto").unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)));
        let err = PreferFormat::parse("CSC").unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)));
    }

    #[test]
    fn require_csc_passes_through_source_when_csc_requested() {
        let stub = StubColumnSource;
        let source: Option<&dyn ColumnShardSource> = Some(&stub);
        // Result<&dyn Trait, _> doesn't implement Debug (the trait
        // object can't be formatted). Discriminate on Ok/Err manually.
        match require_csc(PreferFormat::Csc, source, "stub") {
            Ok(got) => assert_eq!(got.n_csc_shards(), 0),
            Err(_) => panic!("expected Ok, got Err"),
        }
    }

    #[test]
    fn require_csc_errors_when_source_missing() {
        match require_csc(PreferFormat::Csc, None, "no CSC sidecar") {
            Err(AccelError::CscRequestedNotAvailable(msg)) => {
                assert!(msg.contains("no CSC sidecar"))
            }
            Err(other) => panic!("expected CscRequestedNotAvailable, got {other:?}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    #[test]
    fn require_csc_returns_sentinel_when_csr_requested() {
        let stub = StubColumnSource;
        let source: Option<&dyn ColumnShardSource> = Some(&stub);
        match require_csc(PreferFormat::Csr, source, "stub") {
            Err(AccelError::CscNotRequested) => {}
            Err(other) => panic!("expected CscNotRequested, got {other:?}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
        // Also when source is None — same sentinel.
        match require_csc(PreferFormat::Csr, None, "stub") {
            Err(AccelError::CscNotRequested) => {}
            Err(other) => panic!("expected CscNotRequested, got {other:?}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }
}
