//! Full shard GPU decode pipeline.
//!
//! Parses a raw shard byte buffer (header + encoded data), dispatches to
//! the appropriate codec path (GPU-accelerated for Scx1, CPU fallback for
//! None/Zstd), and returns a GPU-resident CSR matrix.

use std::io::Cursor;

use cudarc::driver::safe::CudaSlice;

use scx_codec::delta_golomb::delta_golomb_decode;
use scx_codec::rice::B_VAL;
use scx_codec::{CodecId, EncodedShardRef, ValueEncoding};
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};

use crate::cast_gpu::{cast_u32_to_f32_gpu, cast_u32_to_i32_gpu};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::forbp_gpu::forbp_decode_gpu;
use crate::rice_gpu::rice_decode_gpu;

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
        ),
        CodecId::None | CodecId::Zstd => decode_cpu_fallback(
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
        ),
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
) -> Result<GpuCsr, GpuError> {
    if !value_encoding.is_integer() {
        return Err(GpuError::InvalidShard(
            "Scx1 codec does not support float value encodings".into(),
        ));
    }

    // indptr: Delta-Golomb on CPU → Vec<u64> → Vec<i64> → upload
    let indptr_u64 =
        delta_golomb_decode(indptr_bytes, n_rows + 1).map_err(scx_codec::CodecError::from)?;
    let indptr_i64: Vec<i64> = indptr_u64.into_iter().map(|v| v as i64).collect();
    let d_indptr = dev.htod_copy(&indptr_i64)?;

    // indices: FOR-BP on GPU → on-device u32→i32 cast (no host round-trip)
    let (d_indices_u32, _row_lengths) =
        forbp_decode_gpu(dev, indices_bytes, n_rows, index_dtype_u16)?;
    let d_indices = cast_u32_to_i32_gpu(dev, &d_indices_u32)?;

    // values: Rice on GPU → on-device u32→f32 cast (no host round-trip)
    let d_values_u32 = rice_decode_gpu(dev, values_bytes, nnz, B_VAL)?;
    let d_data = cast_u32_to_f32_gpu(dev, &d_values_u32)?;

    Ok(GpuCsr {
        indptr: d_indptr,
        indices: d_indices,
        data: d_data,
        shape: (n_rows, n_cols),
    })
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
) -> Result<GpuCsr, GpuError> {
    let encoded = EncodedShardRef {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    };
    let (indptr, indices, data) = scx_codec::decode_shard_scipy(
        &encoded,
        codec_id,
        value_encoding,
        n_rows,
        nnz,
        index_dtype_u16,
    )?;

    let d_indptr = dev.htod_copy(&indptr)?;
    let d_indices = dev.htod_copy(&indices)?;
    let d_data = dev.htod_copy(&data)?;

    Ok(GpuCsr {
        indptr: d_indptr,
        indices: d_indices,
        data: d_data,
        shape: (n_rows, n_cols),
    })
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
