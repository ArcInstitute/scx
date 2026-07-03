//! Fail-loud guard for the silent `u32 → f32` decode loss above 2²⁴.
//!
//! scx decodes on-disk integer counts to an `f32` CSR before materialization,
//! so counts above 2²⁴ (16,777,216 — routine in pseudobulk / aggregated counts)
//! are silently rounded. These helpers fold the catalog's per-shard `value_max`
//! and fail loud (as an R error) before an eager decode would corrupt data,
//! unless the caller passes `allow_lossy = TRUE`. Mirrors the pyscx-side guard.
//!
//! The fold reads `ShardStats::value_max`; integer encodings record the true
//! max, float encodings record 0 (so continuous data never trips the guard). An
//! entry missing `ShardStats` contributes 0 — a `> 2²⁴` shard written without
//! stats would slip the guard, acceptable pre-1.0 since every current writer
//! path emits stats, but the guard is only as strong as the catalog it reads.

use extendr_api::prelude::*;
use scx_format_io::ScxReader;

/// Maximum `ShardStats::value_max` over the CSR X shards in scope — all
/// modalities when `modality_id` is `None`, else just that modality.
pub(crate) fn csr_max_value(reader: &ScxReader, modality_id: Option<u8>) -> u32 {
    let cat = reader.catalog();
    let shards = match modality_id {
        Some(m) => cat.csr_shards_for_modality(m),
        None => cat.csr_shards_sorted(),
    };
    shards
        .iter()
        .filter_map(|e| e.stats.as_ref())
        .map(|s| s.value_max)
        .max()
        .unwrap_or(0)
}

/// Maximum `value_max` over the shards of a single named layer (modality 0).
pub(crate) fn layer_max_value(reader: &ScxReader, name: &str) -> u32 {
    reader
        .catalog()
        .layer_csr_shards_for_modality(0, name)
        .iter()
        .filter_map(|e| e.stats.as_ref())
        .map(|s| s.value_max)
        .max()
        .unwrap_or(0)
}

/// Fail loud (as an R error) when decoding an integer shard whose `value_max`
/// exceeds `f32`'s exact-integer range would silently round and the caller has
/// not opted into lossy narrowing.
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
        let reader = ScxReader::open(path).expect("open tiny.scx fixture");
        let max = csr_max_value(&reader, None);
        assert!(
            max <= F32_MAX_EXACT_INT,
            "tiny.scx max {max} unexpectedly > 2^24"
        );
        assert!(guard_decode_loss(max, false).is_ok());
    }
}
