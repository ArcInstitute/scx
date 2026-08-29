//! Fail-loud guard for the silent `u32 → f32` decode loss above 2²⁴.
//!
//! scx decodes on-disk integer counts to an `f32` CSR before materialization,
//! so counts above 2²⁴ (16,777,216 — routine in pseudobulk / aggregated counts)
//! are silently rounded. The R surface fails loud before an eager decode would
//! corrupt data, unless the caller passes `allow_lossy = TRUE`.
//!
//! The `value_max` catalog folds live on `FullCatalog` in scx-format
//! (`csr_max_value` / `layer_csr_max_value`, shared with pyscx); this module
//! keeps only the R-error mapping over `scx_codec::guard_f32_decode_loss`.

use extendr_api::prelude::*;

/// Fail loud (as an R error) when decoding an integer shard whose `value_max`
/// exceeds `f32`'s exact-integer range would silently round and the caller has
/// not opted into lossy narrowing.
///
/// Conservative by design: the catalog carries only the per-shard maximum, so a
/// shard whose `value_max` exceeds 2²⁴ trips the guard even if that particular
/// value is itself f32-exact (e.g. 2²⁵). False positives (recoverable via
/// `allow_lossy`) are preferable to silently missing a real rounding.
pub(crate) fn guard_decode_loss(max_value: u32, allow_lossy: bool) -> Result<()> {
    scx_codec::guard_f32_decode_loss(max_value, allow_lossy)
        .map_err(|e| Error::Other(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_codec::F32_MAX_EXACT_INT;

    #[test]
    fn guard_thresholds() {
        // At/below the exact-int limit: never errors.
        assert!(guard_decode_loss(0, false).is_ok());
        assert!(guard_decode_loss(F32_MAX_EXACT_INT, false).is_ok());
        // Above it without allow_lossy: fail loud.
        assert!(guard_decode_loss(F32_MAX_EXACT_INT + 1, false).is_err());
        assert!(guard_decode_loss(20_000_000, false).is_err());
        // allow_lossy escapes.
        assert!(guard_decode_loss(20_000_000, true).is_ok());
    }

    #[test]
    fn csr_max_value_reads_real_catalog_without_false_positive() {
        // The checked-in tiny fixture holds small counts, so the fold returns a
        // value within f32's exact range and the guard passes — proving the
        // catalog fold works end-to-end on a real file with no spurious error.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/testthat/fixtures/tiny.scx"
        );
        let reader = scx_format_io::ScxReader::open(path).expect("open tiny.scx fixture");
        let max = reader.catalog().csr_max_value(None);
        assert!(
            max <= F32_MAX_EXACT_INT,
            "tiny.scx max {max} unexpectedly > 2^24"
        );
        assert!(guard_decode_loss(max, false).is_ok());
    }
}
