//! GPU-resident assembly of a multi-shard CSR matrix.
//!
//! [`decode_csr_shards_to_device`] decodes every CSR shard of a modality
//! directly onto the device and concatenates them (row-stacked) into a single
//! [`GpuCsr`] — the `data`/`indices` arrays (sized by total nnz) never touch
//! the host. Only the small `indptr` (`n_rows + 1` elements) is host-assembled.
//!
//! This is the device-resident path behind `pyscx.open(...).to_gpu_anndata()`:
//! it avoids the decode → host scipy CSR → re-upload double-trip that Phase 0.2
//! measured as ~92% of the census_1m PCA wall. It is the durable building block
//! the Phase 4 format work (decode-metadata sidecar, parallel/random-access
//! decode) plugs into — Phase 4 changes *how a shard's bytes are decoded*, not
//! how the per-shard results are concatenated.
//!
//! Peak device memory is `combined (data + indices + indptr) + one shard`: each
//! shard is decoded into its own buffers, `memcpy_dtod`-copied into the combined
//! buffer at its running offset, then dropped before the next shard.

use std::io::Cursor;

use scx_format_io::shard::ShardHeader;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::shard_decode::{decode_shard_gpu_with_stats, DeviceDecodeStats, GpuCsr};

/// Decode all CSR shards of a modality straight onto the device and concatenate
/// them (in the order given, row-stacked) into a single [`GpuCsr`].
///
/// `shards` are the raw shard byte buffers (header + encoded payload), e.g. from
/// `ScxReader::read_raw_csr_shard_bytes_for`. All shards must share the same
/// column count (`n_minor`); the assembled matrix has
/// `(sum of shard rows, n_cols)`.
///
/// The output is byte-identical to decoding each shard with
/// `scx_codec::decode_shard_scipy` and concatenating on the host — only the
/// nnz-sized arrays stay device-resident.
///
/// # Errors
/// Returns [`GpuError::InvalidShard`] if `shards` is empty or shard column
/// counts disagree, and propagates any per-shard decode / CUDA error.
pub fn decode_csr_shards_to_device(dev: &GpuDevice, shards: &[&[u8]]) -> Result<GpuCsr, GpuError> {
    decode_csr_shards_to_device_with_stats(dev, shards).map(|(csr, _stats)| csr)
}

/// Like [`decode_csr_shards_to_device`], but also returns an aggregate
/// [`DeviceDecodeStats`] (host↔device transfer accounting) the caller stamps
/// onto the §4.4 `transfer_mode` / `bytes_uploaded` route metadata. Framed Scx1
/// shards decode in VRAM group-by-group; all other codecs host-decode + bounce.
pub fn decode_csr_shards_to_device_with_stats(
    dev: &GpuDevice,
    shards: &[&[u8]],
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    if shards.is_empty() {
        return Err(GpuError::InvalidShard(
            "decode_csr_shards_to_device: no CSR shards".into(),
        ));
    }

    // 1. Pre-scan headers (cheap, no decode) → total rows / nnz + per-shard nnz.
    let mut total_rows: usize = 0;
    let mut total_nnz: usize = 0;
    let mut n_cols: usize = 0;
    let mut per_shard: Vec<(usize, usize)> = Vec::with_capacity(shards.len()); // (rows, nnz)
    for (i, bytes) in shards.iter().enumerate() {
        let header = ShardHeader::read_from(&mut Cursor::new(bytes))
            .map_err(|e| GpuError::InvalidShard(format!("shard {i} header: {e}")))?;
        let rows = header.n_major as usize;
        let nnz = header.nnz as usize;
        let cols = header.n_minor as usize;
        if i == 0 {
            n_cols = cols;
        } else if cols != n_cols {
            return Err(GpuError::InvalidShard(format!(
                "shard {i} column count {cols} != {n_cols} (shards must agree)"
            )));
        }
        total_rows += rows;
        total_nnz += nnz;
        per_shard.push((rows, nnz));
    }

    // 2. Allocate the combined nnz-sized device buffers once.
    let mut combined_data = dev.alloc_zeros::<f32>(total_nnz)?;
    let mut combined_indices = dev.alloc_zeros::<i32>(total_nnz)?;

    // Host-assembled indptr (tiny: total_rows + 1 elements).
    let mut combined_indptr: Vec<i64> = Vec::with_capacity(total_rows + 1);
    combined_indptr.push(0);

    // 3. Decode each shard onto the device, dtod-copy into the combined buffer
    //    at the running nnz offset, and fold its indptr into the host array.
    let mut nnz_base: usize = 0;
    let mut stats = DeviceDecodeStats::default();
    for (i, bytes) in shards.iter().enumerate() {
        let (rows, nnz) = per_shard[i];
        let (shard, shard_stats) = decode_shard_gpu_with_stats(dev, bytes)?;
        stats.merge(&shard_stats);
        debug_assert_eq!(shard.shape.0, rows);
        debug_assert_eq!(shard.indices.len(), nnz);

        if nnz > 0 {
            // memcpy_dtod into the [nnz_base, nnz_base + nnz) sub-range — the
            // slice / slice_mut + memcpy_dtod idiom mirrors cusparse.rs:520.
            let mut data_dst = combined_data.slice_mut(nnz_base..nnz_base + nnz);
            dev.stream()
                .memcpy_dtod(&shard.data, &mut data_dst)
                .map_err(|e| GpuError::CudaError(format!("dtod data (shard {i}): {e}")))?;
            let mut idx_dst = combined_indices.slice_mut(nnz_base..nnz_base + nnz);
            dev.stream()
                .memcpy_dtod(&shard.indices, &mut idx_dst)
                .map_err(|e| GpuError::CudaError(format!("dtod indices (shard {i}): {e}")))?;
        }

        // Fold the shard's indptr (shard-local, starts at 0) into the global
        // array, offset by the running nnz base. The small indptr is the only
        // array that round-trips to the host.
        let shard_indptr = dev.dtoh_copy(&shard.indptr)?;
        debug_assert_eq!(shard_indptr.len(), rows + 1);
        for &v in &shard_indptr[1..=rows] {
            combined_indptr.push(nnz_base as i64 + v);
        }

        nnz_base += nnz;
        // `shard` (its device buffers) is dropped here, before the next shard.
    }
    debug_assert_eq!(nnz_base, total_nnz);
    debug_assert_eq!(combined_indptr.len(), total_rows + 1);

    let combined_indptr = dev.htod_copy(&combined_indptr)?;

    // The per-shard Scx1 path launches async Rice/FOR-BP/cast kernels; the dtod
    // copies are queued on the same stream. Synchronize so a downstream consumer
    // (a cuPy CSR adopting these buffers) cannot race the still-running work.
    dev.synchronize()?;

    Ok((
        GpuCsr {
            indptr: combined_indptr,
            indices: combined_indices,
            data: combined_data,
            shape: (total_rows, n_cols),
        },
        stats,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard_decode::decode_shard_gpu;
    use crate::test_utils::build_test_shard;
    use scx_codec::{decode_shard_scipy, CodecId, EncodedShardRef, ValueEncoding};

    /// Build one Scx1 shard from explicit CSR arrays (u16 values).
    fn build_scx1_shard(
        indptr: &[u64],
        indices: &[u32],
        values_u16: &[u16],
        n_cols: u32,
    ) -> Vec<u8> {
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
        build_test_shard(
            indptr,
            indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        )
    }

    /// Host reference: decode a shard's CSR arrays via the canonical codec path.
    fn host_decode(shard_bytes: &[u8], n_cols: u32) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
        use scx_format_io::shard::ShardHeader;
        use std::io::Cursor;
        let header = ShardHeader::read_from(&mut Cursor::new(shard_bytes)).unwrap();
        let n_rows = header.n_major as usize;
        let nnz = header.nnz as usize;
        let enc = EncodedShardRef {
            indptr_bytes: &shard_bytes[header.indptr_rel_offset as usize..]
                [..header.indptr_length as usize],
            indices_bytes: &shard_bytes[header.indices_rel_offset as usize..]
                [..header.indices_length as usize],
            values_bytes: &shard_bytes[header.values_rel_offset as usize..]
                [..header.values_length as usize],
        };
        let _ = n_cols;
        decode_shard_scipy(
            &enc,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_rows,
            nnz,
            header.index_dtype == 0,
        )
        .unwrap()
    }

    /// Concatenate three uneven row-shards on device == host decode-and-concat.
    #[test]
    fn test_decode_csr_shards_to_device_matches_host_concat() {
        let dev = require_gpu!();
        let n_cols: u32 = 500;

        // Three shards with different row counts and densities.
        let shard_specs: [&[usize]; 3] = [
            &[3, 0, 5, 1, 4],          // 5 rows
            &[2, 2, 2],                // 3 rows
            &[7, 0, 0, 9, 1, 3, 6, 2], // 8 rows
        ];

        let mut shard_bytes_vec: Vec<Vec<u8>> = Vec::new();
        let mut exp_indptr: Vec<i64> = vec![0];
        let mut exp_indices: Vec<i32> = Vec::new();
        let mut exp_data: Vec<f32> = Vec::new();
        let mut nnz_base: i64 = 0;
        let mut state: u64 = 0x51A1_7E55_CAFE_0001; // arbitrary seed

        for rows in shard_specs {
            let mut indptr = vec![0u64];
            let mut indices: Vec<u32> = Vec::new();
            let mut values: Vec<u16> = Vec::new();
            for &nnz_row in rows {
                let mut col = 0u32;
                for _ in 0..nnz_row {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    col += (state % 11 + 1) as u32;
                    if col >= n_cols {
                        break;
                    }
                    indices.push(col);
                    values.push(((state % 5) + 1) as u16);
                }
                indptr.push(indices.len() as u64);
            }
            let bytes = build_scx1_shard(&indptr, &indices, &values, n_cols);
            // Host reference for this shard, offset into the global arrays.
            let (h_indptr, h_indices, h_data) = host_decode(&bytes, n_cols);
            let rows_n = h_indptr.len() - 1;
            for &v in &h_indptr[1..=rows_n] {
                exp_indptr.push(nnz_base + v);
            }
            exp_indices.extend_from_slice(&h_indices);
            exp_data.extend_from_slice(&h_data);
            nnz_base += *h_indptr.last().unwrap();
            shard_bytes_vec.push(bytes);
        }

        let refs: Vec<&[u8]> = shard_bytes_vec.iter().map(|v| v.as_slice()).collect();
        let gpu_csr = decode_csr_shards_to_device(&dev, &refs).unwrap();

        let total_rows: usize = shard_specs.iter().map(|s| s.len()).sum();
        assert_eq!(gpu_csr.shape, (total_rows, n_cols as usize));
        assert_eq!(
            dev.dtoh_copy(&gpu_csr.indptr).unwrap(),
            exp_indptr,
            "indptr"
        );
        assert_eq!(
            dev.dtoh_copy(&gpu_csr.indices).unwrap(),
            exp_indices,
            "indices"
        );
        assert_eq!(dev.dtoh_copy(&gpu_csr.data).unwrap(), exp_data, "data");
    }

    /// A single shard round-trips identically to a direct `decode_shard_gpu`.
    #[test]
    fn test_decode_csr_shards_to_device_single_shard() {
        let dev = require_gpu!();
        let n_cols: u32 = 300;
        let indptr = vec![0u64, 2, 2, 5];
        let indices = vec![1u32, 7, 3, 50, 120];
        let values = vec![1u16, 4, 2, 3, 1];
        let bytes = build_scx1_shard(&indptr, &indices, &values, n_cols);

        let direct = decode_shard_gpu(&dev, &bytes).unwrap();
        let assembled = decode_csr_shards_to_device(&dev, &[bytes.as_slice()]).unwrap();
        assert_eq!(direct.shape, assembled.shape);
        assert_eq!(
            dev.dtoh_copy(&direct.indptr).unwrap(),
            dev.dtoh_copy(&assembled.indptr).unwrap()
        );
        assert_eq!(
            dev.dtoh_copy(&direct.indices).unwrap(),
            dev.dtoh_copy(&assembled.indices).unwrap()
        );
        assert_eq!(
            dev.dtoh_copy(&direct.data).unwrap(),
            dev.dtoh_copy(&assembled.data).unwrap()
        );
    }

    #[test]
    fn test_decode_csr_shards_to_device_empty_errs() {
        // No GPU needed: the empty check fires before any device work.
        let dev = match crate::device::GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU — skip
        };
        assert!(decode_csr_shards_to_device(&dev, &[]).is_err());
    }
}
