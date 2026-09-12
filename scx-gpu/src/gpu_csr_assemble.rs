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

use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::shard::{ShardHeader, DEFAULT_WRITE_SHARD_FORMAT_VERSION};

use crate::combined_csr::CombinedCsr;
use crate::csr_placement::Placement;
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::shard_decode::{
    check_device_len, decode_shard_gpu_with_indptr, DeviceDecodeStats, GpuCsr,
};

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
    //    Also detect whether *every* shard is a framed (shard-v2) ShufDeltaZstd
    //    shard with an integer value encoding — the eligibility condition for the
    //    Phase-2.x cross-shard nvcomp batched decode.
    let mut total_rows: usize = 0;
    let mut total_nnz: usize = 0;
    let mut n_cols: usize = 0;
    let mut per_shard: Vec<(usize, usize)> = Vec::with_capacity(shards.len()); // (rows, nnz)
    let mut all_framed_shufdelta_int = true;
    // What `clamped_reserve` measures the untrusted `total_rows` against.
    let mut indptr_encoded_bytes: usize = 0;
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
        let framed = header.shard_format_version > DEFAULT_WRITE_SHARD_FORMAT_VERSION;
        let is_shufdelta = matches!(
            CodecId::from_u8(header.codec_id),
            Some(CodecId::ShufDeltaZstd)
        );
        let is_integer = ValueEncoding::from_u8(header.value_encoding)
            .map(|v| v.is_integer())
            .unwrap_or(false);
        all_framed_shufdelta_int &= framed && is_shufdelta && is_integer;
        indptr_encoded_bytes += header.indptr_length as usize;
        total_rows += rows;
        total_nnz += nnz;
        per_shard.push((rows, nnz));
    }

    // Phase-2.x batched path (2x-d): when every CSR shard is framed
    // ShufDeltaZstd-integer and the opt-in nvcomp decode is enabled, decode all
    // shards' compressed frames in **2** batched nvcomp calls (idx, val) instead
    // of ~2 per shard. `SCX_NVCOMP_NO_BATCH=1` forces the per-shard loop below
    // (for the batched-vs-per-shard A/B). Mixed-codec / Scx1 / float modalities
    // fall through to the per-shard loop, which still nvcomp-decodes each
    // shufdelta shard individually.
    let no_batch = std::env::var("SCX_NVCOMP_NO_BATCH")
        .map(|v| v == "1")
        .unwrap_or(false);
    if all_framed_shufdelta_int && !no_batch && crate::nvcomp::nvcomp_enabled() {
        return crate::shufdelta_gpu::decode_shufdelta_shards_nvcomp_batched(dev, shards);
    }

    // 2. Allocate the combined nnz-sized device buffers and the host indptr.
    //    `total_rows` is summed out of unauthenticated shard headers, so the
    //    reservation goes through `clamped_reserve` — `Vec::with_capacity`
    //    calls `handle_alloc_error`, which aborts rather than returning. This
    //    was the last production reservation in the crate not doing so.
    let mut combined = CombinedCsr::new(dev, total_nnz, total_rows, indptr_encoded_bytes)?;

    // 3. Decode each shard onto the device, place it into the combined buffer
    //    at the running nnz offset, and fold its indptr into the host array.
    let mut nnz_base: usize = 0;
    let mut stats = DeviceDecodeStats::default();
    for (i, bytes) in shards.iter().enumerate() {
        let (rows, nnz) = per_shard[i];
        let (shard, shard_stats, shard_indptr) = decode_shard_gpu_with_indptr(dev, bytes)?;
        stats.merge(&shard_stats);
        // Catalog stats vs. what the shard actually decoded to. This is one of
        // only two places the length half of the placement check has content —
        // the other is framed Scx1's per-row varint sizing; everywhere else the
        // decoded buffer was sized by `alloc_zeros` from the declared length.
        check_device_len(shard.shape().0, rows, &format!("shard {i} rows"))?;

        // Unconditional, including `nnz == 0`: the check is the point, and the
        // old code ran it for every shard. Gating the whole call on `nnz > 0`
        // narrowed it to non-empty shards, which is a quieter gate than the
        // comment above claims. `place` skips the memcpy itself when the unit is
        // empty, so nothing issues a zero-length copy.
        let at = Placement {
            base: nnz_base,
            len: nnz,
            op: "shard",
            index: i,
        };
        combined.place(dev, at, shard.indices(), shard.data())?;

        // Fold the shard's indptr (shard-local, starts at 0) into the global
        // array, offset by the running nnz base.
        //
        // The decoder hands this back rather than us reading it off the device.
        // Every decode path builds the vector on the host and then uploads it,
        // so the `dev.dtoh_copy(shard.indptr())` that used to stand here was
        // reading back something the host had just produced — and on pageable
        // host memory `cuMemcpyDtoHAsync` is host-synchronous, so it drained the
        // whole stream once per shard.
        //
        // What this removes is the redundant COPY, not the barrier: the framed
        // paths and unframed ShufDeltaZstd still synchronize before returning,
        // because a `decode_shard_gpu` caller may adopt the buffers at once. On
        // the canonical framed layout the loop therefore still costs one barrier
        // per shard plus the outer one — which is why this measured flat.
        // Unframed Scx1 is the exception: it synchronizes only under profiling,
        // so that path does lose a barrier here and leans on the outer `finish`.
        //
        // The length check stays, and means more than it did: it now compares
        // the decoder's own output against the catalog rather than checking a
        // round-trip against itself.
        check_device_len(shard_indptr.len(), rows + 1, &format!("shard {i} indptr"))?;
        for &v in &shard_indptr[1..=rows] {
            combined.indptr.push(nnz_base as i64 + v);
        }

        nnz_base += nnz;
        // `shard` (its device buffers) is dropped here, before the next shard.
    }
    check_device_len(nnz_base, total_nnz, "assembled CSR nnz")?;

    // `finish` synchronizes: the per-shard Scx1 path launches async
    // Rice/FOR-BP/cast kernels and the placements are queued on the same
    // stream, so a downstream cuPy consumer adopting these buffers must not
    // race the still-running work.
    Ok((combined.finish(dev, n_cols, "assembled CSR")?, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard_decode::decode_shard_gpu;
    use crate::test_utils::{build_dense_csr, build_framed_test_shard, build_test_shard};
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
        decode_shard_scipy(
            &enc,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_rows,
            nnz,
            header.index_dtype == 0,
            scx_codec::clamp_index_bound(n_cols),
        )
        .unwrap()
    }

    /// Concatenate three uneven row-shards on device == host decode-and-concat.
    #[test]
    #[ignore = "requires a CUDA GPU"]
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
        assert_eq!(gpu_csr.shape(), (total_rows, n_cols as usize));
        assert_eq!(
            dev.dtoh_copy(gpu_csr.indptr()).unwrap(),
            exp_indptr,
            "indptr"
        );
        assert_eq!(
            dev.dtoh_copy(gpu_csr.indices()).unwrap(),
            exp_indices,
            "indices"
        );
        assert_eq!(dev.dtoh_copy(gpu_csr.data()).unwrap(), exp_data, "data");
    }

    /// The **framed** (shard v2) multi-shard fold matches a host decode-and-concat.
    ///
    /// `test_decode_csr_shards_to_device_matches_host_concat` above covers
    /// unframed Scx1 only, and unframed is the one path whose host indptr was
    /// already a plain live `Vec<i64>`. The framed paths build theirs inside
    /// `CombinedCsr` and used to *destroy* it in `finish`, which is why the loop
    /// recovered it with a per-shard `dtoh_copy` — so framed is the path the
    /// indptr plumbing actually rewires, and it is what
    /// `accel_to_gpu_anndata__scx1_gpu` runs.
    ///
    /// Shard 0 is deliberately multi-group (8 rows at `row_group_rows = 3`): a
    /// single-group framed shard would exercise the fold with one span and
    /// could not see a per-group offset error. `build_framed_test_shard`
    /// asserts `shard_format_version == 2`, so this cannot silently degrade
    /// into a second unframed test.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_decode_csr_shards_to_device_matches_host_concat_framed() {
        let dev = require_gpu!();
        let n_cols: u32 = 4_000;
        let row_group_rows = 3u32;

        // Three shards, uneven rows; the first spans three row groups with a
        // two-row tail, and carries a fully-empty row.
        let shard_specs: [&[usize]; 3] = [
            &[4, 0, 6, 2, 5, 1, 3, 7], // 8 rows -> groups 0..3, 3..6, 6..8
            &[2, 2, 2],                // 3 rows -> one full group
            &[9, 0, 0, 1],             // 4 rows -> groups 0..3, 3..4
        ];

        let mut shard_bytes_vec: Vec<Vec<u8>> = Vec::new();
        let mut exp_indptr: Vec<i64> = vec![0];
        let mut exp_indices: Vec<i32> = Vec::new();
        let mut exp_data: Vec<f32> = Vec::new();
        let mut nnz_base: i64 = 0;

        for rows in shard_specs {
            let (indptr, indices, values_u16) = build_dense_csr(rows, n_cols);
            let values_f32: Vec<f32> = values_u16.iter().map(|&v| v as f32).collect();
            let bytes = build_framed_test_shard(
                &indptr,
                &indices,
                &values_f32,
                CodecId::Scx1,
                n_cols,
                row_group_rows,
            );
            // Host reference: the CSR we encoded, offset into the global arrays.
            for &v in &indptr[1..] {
                exp_indptr.push(nnz_base + v as i64);
            }
            exp_indices.extend(indices.iter().map(|&c| c as i32));
            exp_data.extend_from_slice(&values_f32);
            nnz_base += *indptr.last().unwrap() as i64;
            shard_bytes_vec.push(bytes);
        }

        let refs: Vec<&[u8]> = shard_bytes_vec.iter().map(|v| v.as_slice()).collect();
        let (gpu_csr, stats) = decode_csr_shards_to_device_with_stats(&dev, &refs).unwrap();

        // Premise: this really took the in-VRAM framed Scx1 route, not a
        // host bounce that would make the assertions below vacuous about the
        // path they claim to cover.
        assert_eq!(stats.n_shards_scx1_gpu, 3, "expected 3 framed Scx1 decodes");
        assert_eq!(stats.n_shards_host_bounced, 0, "unexpected host bounce");

        let total_rows: usize = shard_specs.iter().map(|s| s.len()).sum();
        assert_eq!(gpu_csr.shape(), (total_rows, n_cols as usize));
        assert_eq!(
            dev.dtoh_copy(gpu_csr.indptr()).unwrap(),
            exp_indptr,
            "indptr"
        );
        assert_eq!(
            dev.dtoh_copy(gpu_csr.indices()).unwrap(),
            exp_indices,
            "indices"
        );
        assert_eq!(dev.dtoh_copy(gpu_csr.data()).unwrap(), exp_data, "data");
    }

    /// A single shard round-trips identically to a direct `decode_shard_gpu`.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_decode_csr_shards_to_device_single_shard() {
        let dev = require_gpu!();
        let n_cols: u32 = 300;
        let indptr = vec![0u64, 2, 2, 5];
        let indices = vec![1u32, 7, 3, 50, 120];
        let values = vec![1u16, 4, 2, 3, 1];
        let bytes = build_scx1_shard(&indptr, &indices, &values, n_cols);

        let direct = decode_shard_gpu(&dev, &bytes).unwrap();
        let assembled = decode_csr_shards_to_device(&dev, &[bytes.as_slice()]).unwrap();
        assert_eq!(direct.shape(), assembled.shape());
        assert_eq!(
            dev.dtoh_copy(direct.indptr()).unwrap(),
            dev.dtoh_copy(assembled.indptr()).unwrap()
        );
        assert_eq!(
            dev.dtoh_copy(direct.indices()).unwrap(),
            dev.dtoh_copy(assembled.indices()).unwrap()
        );
        assert_eq!(
            dev.dtoh_copy(direct.data()).unwrap(),
            dev.dtoh_copy(assembled.data()).unwrap()
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_decode_csr_shards_to_device_empty_errs() {
        // The empty check fires before any device work, but constructing the
        // device is still what makes this a GPU test.
        let dev = require_gpu!();
        assert!(decode_csr_shards_to_device(&dev, &[]).is_err());
    }
}
