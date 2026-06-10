// Shared CSR→CSC sidecar writer (I-ORG-1 / Task T4.9).
//
// The streaming CSR→CSC transpose-and-write loop used to be reimplemented at
// four call sites across three crates (`pyscx`, `scx-convert` ×2, `rscx`),
// each wrapping the shared `scx_sparse::streaming_csr_to_csc_iter_with_cap`
// iterator with the same encode + `write_csc_shard` loop. This is the single
// definition; all four delegate here.
//
// Callers build their canonical `ScxCsr` shard(s) (the casting /
// canonicalization preamble differs per source — e.g. the eager convert path
// canonicalizes, the multimodal path trusts already-canonical modality data —
// so it stays caller-side) and call `write_csc_sidecar`.

use scx_codec::value_encoding::values_to_raw_bytes;
use scx_codec::{CodecId, ValueEncoding};
use scx_sparse::{streaming_csr_to_csc_iter_with_cap, ScxCsr};

use crate::error::ScxError;
use crate::writer::ScxWriter;

/// Default per-chunk memory budget for the streaming CSC transpose (4 GiB).
/// Bounds the number of columns materialized at once independent of
/// `cols_per_shard`.
pub const DEFAULT_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Stream a CSR→CSC transpose over `csr_shards` and write the result as a CSC
/// sidecar, one shard per chunk, via the writer.
///
/// Each emitted shard's column count is bounded by `cols_per_shard` (or
/// `memory_budget_bytes`, whichever is smaller). `csr_shards` must already be
/// canonical — this helper does not sort/dedup/drop-zeros (callers do so
/// upstream when their source requires it).
///
/// `modality_id`:
/// - `None` → single-modality file; shards written via
///   [`ScxWriter::write_csc_shard`] (`X_csc_shard_*`).
/// - `Some(id)` → multimodal file; shards written via
///   [`ScxWriter::write_csc_shard_for`] under modality `id`.
#[allow(clippy::too_many_arguments)]
pub fn write_csc_sidecar(
    writer: &mut ScxWriter,
    csr_shards: &[ScxCsr],
    n_obs: usize,
    n_vars: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    cols_per_shard: usize,
    memory_budget_bytes: usize,
    modality_id: Option<u8>,
) -> Result<(), ScxError> {
    let mut iter = streaming_csr_to_csc_iter_with_cap(
        csr_shards,
        n_obs,
        n_vars,
        memory_budget_bytes,
        cols_per_shard,
    )
    .map_err(|e| ScxError::CscTranspose(e.to_string()))?;

    loop {
        let col_start = iter.current_col_start() as u64;
        let chunk = match iter.next() {
            Some(c) => c.map_err(|e| ScxError::CscTranspose(e.to_string()))?,
            None => break,
        };

        let csc_indptr_u64: Vec<u64> = chunk.indptr.iter().map(|&v| v as u64).collect();
        let csc_indices_u32: Vec<u32> = chunk.indices.iter().map(|&i| i as u32).collect();
        let raw_values = values_to_raw_bytes(&chunk.data, value_encoding)?;

        match modality_id {
            Some(mid) => writer.write_csc_shard_for(
                mid,
                &csc_indptr_u64,
                &csc_indices_u32,
                &raw_values,
                codec_id,
                value_encoding,
                col_start,
            )?,
            None => writer.write_csc_shard(
                &csc_indptr_u64,
                &csc_indices_u32,
                &raw_values,
                codec_id,
                value_encoding,
                col_start,
            )?,
        }
    }
    Ok(())
}
