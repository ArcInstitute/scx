// Shared CSR→CSC sidecar writer.
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

        // The transpose yields non-negative offsets/row-indices for valid CSR;
        // assert it in debug builds before the unchecked widening casts (a
        // negative would silently wrap). Release builds trust the contract.
        debug_assert!(
            chunk.indptr.iter().all(|&v| v >= 0),
            "CSC chunk indptr must be non-negative before u64 cast"
        );
        debug_assert!(
            chunk.indices.iter().all(|&i| i >= 0),
            "CSC chunk indices must be non-negative before u32 cast"
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::FileHeader;
    use crate::reader::ScxReader;
    use crate::writer::ScxWriter;

    /// Round-trip a small 3×3 canonical CSR through `write_csc_sidecar` and
    /// read the CSC sidecar back, checking the column-major transpose.
    #[test]
    fn write_csc_sidecar_round_trips_single_modality() {
        // CSR (3 rows × 3 cols):
        //   row0: (c0=1, c2=3)
        //   row1: (c1=2)
        //   row2: (c2=4)
        let csr = ScxCsr::new(
            (3, 3),
            vec![0, 2, 3, 4],
            vec![0, 2, 1, 2],
            vec![1.0, 3.0, 2.0, 4.0],
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("csc.scx");
        let header = FileHeader::new_single_modality(3, 3, 4, 16384, CodecId::None as u8, 0);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        let raw = values_to_raw_bytes(&csr.data, ValueEncoding::Uint8).unwrap();
        writer
            .write_csr_shard(
                &csr.indptr.iter().map(|&v| v as u64).collect::<Vec<_>>(),
                &csr.indices.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                &raw,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        write_csc_sidecar(
            &mut writer,
            std::slice::from_ref(&csr),
            3,
            3,
            ValueEncoding::Uint8,
            CodecId::None,
            8, // cols_per_shard ≥ n_cols → one CSC shard
            DEFAULT_CSC_MEMORY_BYTES,
            None,
        )
        .unwrap();
        let final_path = writer.finish().unwrap();

        let reader = ScxReader::open(&final_path).unwrap();
        assert!(reader.header().n_csc_shards >= 1);
        let csc = reader.read_all_csc_shards_for(0).unwrap();
        // Reconstruct dense from the CSC and compare to the CSR's dense form
        // (both row-major n_obs × n_vars).
        let dense_from_csc = csc.to_dense().unwrap();
        let dense_from_csr = csr.to_dense().unwrap();
        assert_eq!(dense_from_csc, dense_from_csr);
    }
}
