//! Full shard GPU decode pipeline.
//!
//! Parses a raw shard byte buffer (header + encoded data), dispatches to
//! the appropriate codec path (GPU-accelerated for Scx1, CPU fallback for
//! None/Zstd), and returns a GPU-resident CSR matrix.

use std::io::Cursor;

use cudarc::driver::safe::CudaSlice;

use scx_codec::delta_golomb::delta_golomb_decode;
use scx_codec::rice::B_VAL;
use scx_codec::{CodecId, EncodedShardRef, Scx1DecodeMetadata, ValueEncoding};
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};

use crate::cast_gpu::{cast_u32_to_f32_gpu, cast_u32_to_i32_gpu};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::forbp_gpu::{forbp_decode_gpu, forbp_decode_gpu_with_metadata};
use crate::profile::{self, CodecClass};
use crate::rice_gpu::{rice_decode_gpu, rice_decode_gpu_with_metadata};

/// GPU-resident CSR matrix.
///
/// Type layout matches scipy CSR conventions: i64 indptr, i32 indices, f32 data.
/// Binary-compatible with cuSPARSE and cupy `__cuda_array_interface__`.
pub struct GpuCsr {
    /// Compressed row pointer array (`n_rows + 1` elements, i64).
    pub indptr: CudaSlice<i64>,
    /// Column indices array (`nnz` elements, i32).
    pub indices: CudaSlice<i32>,
    /// Non-zero values array (`nnz` elements, f32).
    pub data: CudaSlice<f32>,
    /// Matrix dimensions `(n_rows, n_cols)`.
    pub shape: (usize, usize),
}

/// Host→device transfer accounting for a device decode, surfaced for the
/// ACC-RUST-OPT-V4 §4.4 `transfer_mode` / `bytes_uploaded` route metadata.
///
/// Lets a caller distinguish a genuine in-VRAM Scx1 decode (only the tiny indptr
/// uploaded) from a host-decode+HtoD bounce. All Scx1 indices+values (including
/// the >= 128-nnz BitPacker4x rows, Task 4.4b) decode on the device; only a
/// non-Scx1 codec shard decodes wholly on the host.
#[derive(Debug, Clone, Copy)]
pub struct DeviceDecodeStats {
    /// Total host→device bytes: the Scx1 indptr (always uploaded) and the full
    /// payload of any non-Scx1 shard.
    pub host_uploaded_bytes: u64,
    /// Bytes decoded directly on the device (Scx1 indices/values that took the
    /// GPU kernel path).
    pub device_decoded_bytes: u64,
    /// True iff every shard decoded its indices+values on the device (all Scx1)
    /// — i.e. only indptr was uploaded. Drives the `scx_device_decode_gpu`
    /// transfer-mode stamp.
    pub fully_device_decoded: bool,
}

impl Default for DeviceDecodeStats {
    fn default() -> Self {
        // `fully_device_decoded` starts `true` so aggregation can AND it down as
        // shards are merged; a fresh single-shard stat sets it explicitly.
        Self {
            host_uploaded_bytes: 0,
            device_decoded_bytes: 0,
            fully_device_decoded: true,
        }
    }
}

impl DeviceDecodeStats {
    /// Fold a per-shard stat into a running aggregate.
    pub fn merge(&mut self, other: &DeviceDecodeStats) {
        self.host_uploaded_bytes += other.host_uploaded_bytes;
        self.device_decoded_bytes += other.device_decoded_bytes;
        self.fully_device_decoded &= other.fully_device_decoded;
    }
}

/// Decode an entire SCX shard on GPU, producing a [`GpuCsr`].
///
/// Pipeline:
/// 1. Parse shard header (76 bytes, CPU)
/// 2. Extract encoded indptr/indices/values byte slices
/// 3. Dispatch on codec_id:
///    - **Scx1**: Delta-Golomb (CPU) + FOR-BP (GPU) + Rice (GPU)
///    - **None/Zstd**: CPU decode via `decode_shard_scipy` + upload
/// 4. Type-convert to i64/i32/f32 and return GpuCsr
///
/// Produces **bit-identical** output to `scx_codec::decode_shard_scipy`.
pub fn decode_shard_gpu(dev: &GpuDevice, shard_bytes: &[u8]) -> Result<GpuCsr, GpuError> {
    decode_shard_gpu_with_metadata(dev, shard_bytes, None).map(|(csr, _stats)| csr)
}

/// Decode an entire SCX shard on GPU, optionally driven by a decode-metadata
/// sidecar ([`Scx1DecodeMetadata`]) so the per-row FOR-BP / per-block Rice
/// offsets feed the kernels directly instead of being re-derived by a CPU
/// prescan (ACC-RUST-OPT-V4 Task 4.4a). `metadata` applies only to the Scx1
/// path; pass `None` (or a non-Scx1 shard) to use the prescan path. Returns the
/// [`GpuCsr`] plus [`DeviceDecodeStats`] describing the host↔device transfer.
pub fn decode_shard_gpu_with_metadata(
    dev: &GpuDevice,
    shard_bytes: &[u8],
    metadata: Option<&Scx1DecodeMetadata>,
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    // 1. Parse 76-byte shard header
    if shard_bytes.len() < SHARD_HEADER_SIZE {
        return Err(GpuError::InvalidShard(format!(
            "shard too small: {} bytes < {} header",
            shard_bytes.len(),
            SHARD_HEADER_SIZE
        )));
    }
    let header = ShardHeader::read_from(&mut Cursor::new(shard_bytes))
        .map_err(|e| GpuError::InvalidShard(format!("shard header: {e}")))?;

    let codec_id = CodecId::from_u8(header.codec_id)
        .ok_or_else(|| GpuError::InvalidShard(format!("unknown codec_id: {}", header.codec_id)))?;
    let value_encoding = ValueEncoding::from_u8(header.value_encoding).ok_or_else(|| {
        GpuError::InvalidShard(format!("unknown value_encoding: {}", header.value_encoding))
    })?;
    let index_dtype_u16 = header.index_dtype == 0;
    let n_rows = header.n_major as usize;
    let n_cols = header.n_minor as usize;
    let nnz = header.nnz as usize;

    // 2. Extract encoded byte slices
    let indptr_bytes = extract_slice(
        shard_bytes,
        header.indptr_rel_offset,
        header.indptr_length,
        "indptr",
    )?;
    let indices_bytes = extract_slice(
        shard_bytes,
        header.indices_rel_offset,
        header.indices_length,
        "indices",
    )?;
    let values_bytes = extract_slice(
        shard_bytes,
        header.values_rel_offset,
        header.values_length,
        "values",
    )?;

    // 3. Dispatch on codec
    match codec_id {
        CodecId::Scx1 => decode_scx1_gpu(
            dev,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            value_encoding,
            index_dtype_u16,
            n_rows,
            n_cols,
            nnz,
            metadata,
        ),
        CodecId::None | CodecId::Zstd | CodecId::Lz4Shuffle | CodecId::Pcodec => {
            decode_cpu_fallback(
                dev,
                indptr_bytes,
                indices_bytes,
                values_bytes,
                codec_id,
                value_encoding,
                index_dtype_u16,
                n_rows,
                n_cols,
                nnz,
            )
        }
    }
}

/// Extract a byte slice from shard data using relative offset and length.
fn extract_slice<'a>(
    shard_bytes: &'a [u8],
    rel_offset: u32,
    length: u32,
    name: &str,
) -> Result<&'a [u8], GpuError> {
    let start = rel_offset as usize;
    let end = start + length as usize;
    if end > shard_bytes.len() {
        return Err(GpuError::InvalidShard(format!(
            "{name} slice [{start}..{end}] exceeds shard size {}",
            shard_bytes.len()
        )));
    }
    Ok(&shard_bytes[start..end])
}

/// GPU-accelerated Scx1 decode path.
///
/// - indptr: Delta-Golomb on CPU (small array, not worth GPU overhead)
/// - indices: FOR-BP on GPU → on-device u32→i32 cast
/// - values: Rice on GPU → on-device u32→f32 cast
#[allow(clippy::too_many_arguments)]
fn decode_scx1_gpu(
    dev: &GpuDevice,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    n_rows: usize,
    n_cols: usize,
    nnz: usize,
    metadata: Option<&Scx1DecodeMetadata>,
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    if !value_encoding.is_integer() {
        return Err(GpuError::InvalidShard(
            "Scx1 codec does not support float value encodings".into(),
        ));
    }

    let mut stats = DeviceDecodeStats::default();

    // indptr: Delta-Golomb on CPU → Vec<u64> → Vec<i64> → upload
    let t_host = profile::start();
    let indptr_u64 =
        delta_golomb_decode(indptr_bytes, n_rows + 1).map_err(scx_codec::CodecError::from)?;
    let indptr_i64: Vec<i64> = indptr_u64.into_iter().map(|v| v as i64).collect();
    profile::record_host_decode_since(CodecClass::Scx1, t_host);

    let t_htod = profile::start();
    let d_indptr = dev.htod_copy(&indptr_i64)?;
    let indptr_bytes_uploaded = (indptr_i64.len() * 8) as u64;
    profile::record_htod_since(CodecClass::Scx1, t_htod, indptr_i64.len() * 8);
    // indptr always round-trips host→device; it is the residual upload of a
    // fully-GPU-decoded Scx1 shard (the "only the tiny indptr" of §2.5).
    stats.host_uploaded_bytes += indptr_bytes_uploaded;

    // indices: FOR-BP on GPU → on-device u32→i32 cast (no host round-trip)
    // values:  Rice on GPU → on-device u32→f32 cast (no host round-trip)
    // Both decode GPU-side; the bucket covers the bitstream upload + kernels.
    // With a sidecar, the per-row/per-block offsets feed the kernels directly
    // and the CPU prescan is skipped (Task 4.4a).
    let t_gpu = profile::start();
    let (d_indices_u32, _row_lengths) = match metadata {
        Some(meta) => forbp_decode_gpu_with_metadata(dev, indices_bytes, &meta.rows, n_rows)?,
        None => forbp_decode_gpu(dev, indices_bytes, n_rows, index_dtype_u16)?,
    };
    // FOR-BP indices (scalar + BitPacker4x rows, Task 4.4b) and Rice values both
    // decode on the device — the gpu_decode bucket covers the bitstream upload +
    // kernels; only the indptr round-trips through the host.
    stats.device_decoded_bytes += (nnz as u64) * 4;
    let d_indices = cast_u32_to_i32_gpu(dev, &d_indices_u32)?;

    let d_values_u32 = match metadata {
        Some(meta) => {
            rice_decode_gpu_with_metadata(dev, values_bytes, &meta.rice_blocks, nnz, B_VAL)?
        }
        None => rice_decode_gpu(dev, values_bytes, nnz, B_VAL)?,
    };
    // Rice values always decode on the device.
    stats.device_decoded_bytes += (nnz as u64) * 4;
    let d_data = cast_u32_to_f32_gpu(dev, &d_values_u32)?;
    if t_gpu.is_some() {
        // Synchronize so the elapsed time reflects completed device work, not
        // just async launch latency. Only paid when profiling is enabled.
        dev.synchronize()?;
    }
    profile::record_gpu_decode_since(t_gpu);

    Ok((
        GpuCsr {
            indptr: d_indptr,
            indices: d_indices,
            data: d_data,
            shape: (n_rows, n_cols),
        },
        stats,
    ))
}

/// CPU fallback decode for None and Zstd codecs, then upload to GPU.
#[allow(clippy::too_many_arguments)]
fn decode_cpu_fallback(
    dev: &GpuDevice,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    n_rows: usize,
    n_cols: usize,
    nnz: usize,
) -> Result<(GpuCsr, DeviceDecodeStats), GpuError> {
    let encoded = EncodedShardRef {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    };
    let t_host = profile::start();
    let (indptr, indices, data) = scx_codec::decode_shard_scipy(
        &encoded,
        codec_id,
        value_encoding,
        n_rows,
        nnz,
        index_dtype_u16,
    )?;
    profile::record_host_decode_since(CodecClass::Generic, t_host);

    let t_htod = profile::start();
    let d_indptr = dev.htod_copy(&indptr)?;
    let d_indices = dev.htod_copy(&indices)?;
    let d_data = dev.htod_copy(&data)?;
    let htod_bytes = indptr.len() * 8 + indices.len() * 4 + data.len() * 4;
    profile::record_htod_since(CodecClass::Generic, t_htod, htod_bytes);

    // Non-Scx1 codecs decode wholly on the host then upload the full CSR — never
    // a device decode, so this never counts as `fully_device_decoded`.
    let stats = DeviceDecodeStats {
        host_uploaded_bytes: htod_bytes as u64,
        device_decoded_bytes: 0,
        fully_device_decoded: false,
    };

    Ok((
        GpuCsr {
            indptr: d_indptr,
            indices: d_indices,
            data: d_data,
            shape: (n_rows, n_cols),
        },
        stats,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::build_test_shard;
    use scx_codec::decode_shard_scipy;
    use scx_format::shard::ShardHeader;

    /// CPU reference decode for comparison.
    fn cpu_decode(
        indptr_bytes: &[u8],
        indices_bytes: &[u8],
        values_bytes: &[u8],
        codec_id: CodecId,
        value_encoding: ValueEncoding,
        n_rows: usize,
        nnz: usize,
        index_dtype_u16: bool,
    ) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
        let encoded = EncodedShardRef {
            indptr_bytes,
            indices_bytes,
            values_bytes,
        };
        decode_shard_scipy(
            &encoded,
            codec_id,
            value_encoding,
            n_rows,
            nnz,
            index_dtype_u16,
        )
        .expect("CPU decode failed")
    }

    #[test]
    fn test_shard_decode_gpu_scx1() {
        let dev = require_gpu!();

        // Build test CSR: 200 rows, ~1000 columns, u16 values
        let n_rows = 200;
        let n_cols: u32 = 1000;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values_u16 = Vec::new();

        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for _ in 0..n_rows {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let nnz_row = (state % 20 + 1) as usize;
            let mut col = 0u32;
            for _ in 0..nnz_row {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                col += (state % 10 + 1) as u32;
                if col >= n_cols {
                    break;
                }
                indices.push(col);
                // UMI-like values: mostly 1-3, occasional larger
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let v = ((state % 5) + 1) as u16;
                values_u16.push(v);
            }
            indptr.push(indices.len() as u64);
        }

        // Convert u16 values to raw LE bytes
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let nnz = indices.len();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );

        // GPU decode
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();
        let gpu_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let gpu_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let gpu_data = dev.dtoh_copy(&gpu_csr.data).unwrap();

        // CPU reference decode
        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_rows,
            nnz,
            true,
        );

        assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize));
        assert_eq!(gpu_indptr, cpu_indptr, "indptr mismatch");
        assert_eq!(gpu_indices, cpu_indices, "indices mismatch");
        assert_eq!(gpu_data, cpu_data, "data mismatch");
    }

    /// Build a deterministic CSR with the given per-row nnz counts. Columns are
    /// strictly increasing within each row (FOR-BP requires sorted indices).
    /// Returns `(indptr, indices, values_u16)`.
    fn build_dense_csr(row_nnzs: &[usize], n_cols: u32) -> (Vec<u64>, Vec<u32>, Vec<u16>) {
        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values_u16: Vec<u16> = Vec::new();
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        for &nnz_row in row_nnzs {
            let mut col = 0u32;
            for _ in 0..nnz_row {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                col += (state % 4 + 1) as u32; // strictly increasing
                assert!(col < n_cols, "test column overflow — widen n_cols");
                indices.push(col);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                values_u16.push((state % 7 + 1) as u16);
            }
            indptr.push(indices.len() as u64);
        }
        (indptr, indices, values_u16)
    }

    fn assert_gpu_cpu_decode_match(
        dev: &GpuDevice,
        indptr: &[u64],
        indices: &[u32],
        values_u16: &[u16],
        n_cols: u32,
    ) {
        let n_rows = indptr.len() - 1;
        let nnz = indices.len();
        let index_dtype_u16 = n_cols <= 65535;
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shard_bytes = build_test_shard(
            indptr,
            indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );

        let gpu_csr = decode_shard_gpu(dev, &shard_bytes).unwrap();
        let gpu_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let gpu_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let gpu_data = dev.dtoh_copy(&gpu_csr.data).unwrap();

        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_rows,
            nnz,
            index_dtype_u16,
        );

        assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize));
        assert_eq!(gpu_indptr, cpu_indptr, "indptr mismatch");
        assert_eq!(gpu_indices, cpu_indices, "indices mismatch");
        assert_eq!(gpu_data, cpu_data, "data mismatch");
    }

    /// FOR-BP SIMD-layout (BitPacker4x) GPU-decode parity (Task 4.4b).
    ///
    /// Rows with nnz >= `scx_codec::forbp::SIMD_THRESHOLD` (128) are bit-packed
    /// by the encoder with BitPacker4x's SIMD layout; `forbp_decode_gpu` decodes
    /// them on-device via the BitPacker4x kernel. The synthetic
    /// `test_shard_decode_gpu_scx1` uses only small rows (<=20 nnz) and never
    /// exercises this path; real data (e.g. pbmc3k cells expressing >=128 genes)
    /// does.
    #[test]
    fn test_shard_decode_gpu_scx1_dense_rows_u16() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        // Mix of dense (>=128 nnz -> SIMD path), exactly-threshold, sparse, empty.
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        assert_gpu_cpu_decode_match(&dev, &indptr, &indices, &values_u16, n_cols);
    }

    /// Same SIMD-layout parity for u32 column indices (n_cols > 65535), which
    /// exercises the wider `frame_min` / `frame_bits` path of the BitPacker4x kernel.
    #[test]
    fn test_shard_decode_gpu_scx1_dense_rows_u32() {
        let dev = require_gpu!();
        let n_cols: u32 = 70_000;
        let row_nnzs = [2usize, 130, 300, 0, 512, 9, 256];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        assert_gpu_cpu_decode_match(&dev, &indptr, &indices, &values_u16, n_cols);
    }

    /// Decode an Scx1 shard three ways — host reference, GPU no-sidecar (CPU
    /// prescan), and GPU sidecar-driven (Task 4.4a) — and assert all three are
    /// byte-identical. Returns the sidecar-driven [`DeviceDecodeStats`].
    fn assert_sidecar_decode_match(
        dev: &GpuDevice,
        indptr: &[u64],
        indices: &[u32],
        values_u16: &[u16],
        n_cols: u32,
    ) -> DeviceDecodeStats {
        let n_rows = indptr.len() - 1;
        let nnz = indices.len();
        let index_dtype_u16 = n_cols <= 65535;
        let values_raw: Vec<u8> = values_u16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let (shard_bytes, meta) = crate::test_utils::build_test_shard_with_metadata(
            indptr,
            indices,
            &values_raw,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_cols,
        );
        let meta = meta.expect("Scx1 shard must emit decode metadata");

        // Host reference.
        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_rows,
            nnz,
            index_dtype_u16,
        );

        // GPU, no sidecar (CPU prescan path) and GPU, sidecar-driven.
        let (plain, _plain_stats) =
            decode_shard_gpu_with_metadata(dev, &shard_bytes, None).unwrap();
        let (sided, stats) =
            decode_shard_gpu_with_metadata(dev, &shard_bytes, Some(&meta)).unwrap();

        for (label, csr) in [("no-sidecar", &plain), ("sidecar", &sided)] {
            let g_indptr = dev.dtoh_copy(&csr.indptr).unwrap();
            let g_indices = dev.dtoh_copy(&csr.indices).unwrap();
            let g_data = dev.dtoh_copy(&csr.data).unwrap();
            assert_eq!(csr.shape, (n_rows, n_cols as usize), "{label} shape");
            assert_eq!(g_indptr, cpu_indptr, "{label} indptr mismatch");
            assert_eq!(g_indices, cpu_indices, "{label} indices mismatch");
            assert_eq!(g_data, cpu_data, "{label} data mismatch");
        }
        stats
    }

    /// Sidecar-driven GPU decode of an all-sparse Scx1 shard (<128 nnz/row,
    /// incl. empty rows and a partial last Rice block) is byte-identical to the
    /// prescan path and host, and decodes entirely on the device.
    #[test]
    fn test_sidecar_decode_all_sparse_fully_device() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs = [0usize, 1, 5, 64, 0, 100, 7, 33, 0, 120, 2, 90];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let stats = assert_sidecar_decode_match(&dev, &indptr, &indices, &values_u16, n_cols);
        assert!(
            stats.fully_device_decoded,
            "all-sparse must fully device-decode"
        );
        let nnz = indices.len() as u64;
        // Only the tiny indptr is uploaded; indices+values decode on the device.
        assert_eq!(stats.host_uploaded_bytes, (indptr.len() as u64) * 8);
        assert_eq!(stats.device_decoded_bytes, nnz * 8);
    }

    /// Sidecar-driven GPU decode of a mixed shard (some rows >= 128 nnz) stays
    /// byte-identical AND, as of Task 4.4b, decodes entirely on the device — the
    /// BitPacker4x kernel handles the >= 128-nnz rows, so FOR-BP indices no longer
    /// host-fall-back (only the tiny indptr uploads).
    #[test]
    fn test_sidecar_decode_mixed_dense_fully_device() {
        let dev = require_gpu!();
        let n_cols: u32 = 4000;
        let row_nnzs = [0usize, 1, 5, 130, 256, 7, 0, 384, 200, 3, 128, 129, 512, 50];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let stats = assert_sidecar_decode_match(&dev, &indptr, &indices, &values_u16, n_cols);
        assert!(
            stats.fully_device_decoded,
            "dense rows must decode on the device (BitPacker4x kernel, Task 4.4b)"
        );
        let nnz = indices.len() as u64;
        // Indices + values both decode on device; only the indptr uploads.
        assert_eq!(
            stats.device_decoded_bytes,
            nnz * 8,
            "indices + values on device"
        );
        assert_eq!(stats.host_uploaded_bytes, (indptr.len() as u64) * 8);
    }

    /// u32-column (n_cols > 65535) sidecar-driven parity.
    #[test]
    fn test_sidecar_decode_u32_indices() {
        let dev = require_gpu!();
        let n_cols: u32 = 70_000;
        let row_nnzs = [2usize, 130, 300, 0, 512, 9, 256, 1, 64];
        let (indptr, indices, values_u16) = build_dense_csr(&row_nnzs, n_cols);
        let _ = assert_sidecar_decode_match(&dev, &indptr, &indices, &values_u16, n_cols);
    }

    #[test]
    fn test_shard_decode_gpu_none() {
        let dev = require_gpu!();

        // Small CSR: 10 rows, u8 values, no compression
        let n_cols: u32 = 50;
        let indptr = vec![0u64, 3, 5, 5, 8, 10, 12, 15, 17, 20, 22];
        let indices = vec![
            0u32, 10, 20, // row 0
            5, 15, // row 1
            // row 2 empty
            1, 2, 3, // row 3
            30, 40, // row 4
            0, 49, // row 5
            10, 20, 30, // row 6
            25, 35, // row 7
            0, 1, 2, // row 8
            48, 49, // row 9
        ];
        let values_u8: Vec<u8> = (1..=22).collect();
        let nnz = indices.len();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_u8,
            CodecId::None,
            ValueEncoding::Uint8,
            n_cols,
        );

        // GPU decode
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();
        let gpu_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let gpu_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let gpu_data = dev.dtoh_copy(&gpu_csr.data).unwrap();

        // CPU reference
        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::None,
            ValueEncoding::Uint8,
            10,
            nnz,
            true,
        );

        assert_eq!(gpu_csr.shape, (10, n_cols as usize));
        assert_eq!(gpu_indptr, cpu_indptr, "indptr mismatch");
        assert_eq!(gpu_indices, cpu_indices, "indices mismatch");
        assert_eq!(gpu_data, cpu_data, "data mismatch");
    }

    #[test]
    fn test_shard_decode_gpu_zstd() {
        let dev = require_gpu!();

        // 50 rows, float32 values (Zstd is the only codec for floats)
        let n_rows = 50;
        let n_cols: u32 = 200;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values_f32 = Vec::new();

        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        for _ in 0..n_rows {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let nnz_row = (state % 10 + 1) as usize;
            let mut col = 0u32;
            for _ in 0..nnz_row {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                col += (state % 20 + 1) as u32;
                if col >= n_cols {
                    break;
                }
                indices.push(col);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                values_f32.push((state % 1000) as f32 / 100.0);
            }
            indptr.push(indices.len() as u64);
        }

        let values_raw: Vec<u8> = values_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let nnz = indices.len();

        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values_raw,
            CodecId::Zstd,
            ValueEncoding::Float32,
            n_cols,
        );

        // GPU decode
        let gpu_csr = decode_shard_gpu(&dev, &shard_bytes).unwrap();
        let gpu_indptr = dev.dtoh_copy(&gpu_csr.indptr).unwrap();
        let gpu_indices = dev.dtoh_copy(&gpu_csr.indices).unwrap();
        let gpu_data = dev.dtoh_copy(&gpu_csr.data).unwrap();

        // CPU reference
        let header = ShardHeader::read_from(&mut Cursor::new(&shard_bytes)).unwrap();
        let indptr_enc =
            &shard_bytes[header.indptr_rel_offset as usize..][..header.indptr_length as usize];
        let indices_enc =
            &shard_bytes[header.indices_rel_offset as usize..][..header.indices_length as usize];
        let values_enc =
            &shard_bytes[header.values_rel_offset as usize..][..header.values_length as usize];
        let (cpu_indptr, cpu_indices, cpu_data) = cpu_decode(
            indptr_enc,
            indices_enc,
            values_enc,
            CodecId::Zstd,
            ValueEncoding::Float32,
            n_rows,
            nnz,
            true,
        );

        assert_eq!(gpu_csr.shape, (n_rows, n_cols as usize));
        assert_eq!(gpu_indptr, cpu_indptr, "indptr mismatch");
        assert_eq!(gpu_indices, cpu_indices, "indices mismatch");
        assert_eq!(gpu_data, cpu_data, "data mismatch");
    }
}
