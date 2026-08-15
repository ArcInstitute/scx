//! Little-endian raw conversion helpers.
//!
//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::codec_id::{CodecError, ValueEncoding};

/// Convert raw LE value bytes to f32 according to ValueEncoding.
pub(crate) fn values_raw_to_f32(raw: &[u8], encoding: ValueEncoding) -> Vec<f32> {
    match encoding {
        ValueEncoding::Uint8 => {
            let mut out = Vec::with_capacity(raw.len());
            out.extend(raw.iter().map(|&b| b as f32));
            out
        }
        ValueEncoding::Uint16 => raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        ValueEncoding::Uint32 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        ValueEncoding::Float32 => {
            // `bytemuck::cast_slice::<u8, f32>` panics if the source bytes
            // aren't 4-byte aligned. Mmap'd payloads are usually aligned, but
            // we can't rely on it — decompressed buffers from Zstd/LZ4 land at
            // whatever alignment the allocator picked. Branch on alignment +
            // length; fall back to a scalar byteswap-free decode otherwise.
            #[cfg(target_endian = "little")]
            {
                if (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
                    && raw.len().is_multiple_of(std::mem::size_of::<f32>())
                {
                    bytemuck::cast_slice::<u8, f32>(raw).to_vec()
                } else {
                    raw.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                }
            }
            #[cfg(not(target_endian = "little"))]
            {
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        }
        ValueEncoding::Float16 => {
            // `half::slice::HalfFloatSliceExt::convert_to_f32_slice` uses a
            // vectorized path when the input is aligned. Same alignment
            // guard as Float32 above.
            #[cfg(target_endian = "little")]
            {
                if (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<half::f16>())
                    && raw.len().is_multiple_of(std::mem::size_of::<half::f16>())
                {
                    use half::slice::HalfFloatSliceExt;
                    // SAFETY: alignment + length checked immediately above,
                    // `half::f16` is `#[repr(transparent)]` over `u16`, so any
                    // aligned 2-byte little-endian group is a valid `f16` bit
                    // pattern.
                    let src: &[half::f16] = unsafe {
                        std::slice::from_raw_parts(
                            raw.as_ptr() as *const half::f16,
                            raw.len() / std::mem::size_of::<half::f16>(),
                        )
                    };
                    let mut out = vec![0.0f32; src.len()];
                    src.convert_to_f32_slice(&mut out);
                    out
                } else {
                    raw.chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect()
                }
            }
            #[cfg(not(target_endian = "little"))]
            {
                raw.chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CodecId::None
// ---------------------------------------------------------------------------

pub(crate) fn u64_slice_to_le_bytes(data: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(data.len() * 8);
    for &v in data {
        buf.write_u64::<LittleEndian>(v).unwrap();
    }
    buf
}

pub(crate) fn le_bytes_to_u64(data: &[u8], count: usize) -> Result<Vec<u64>, CodecError> {
    if data.len() != count * 8 {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "indptr byte length {} != expected {}",
                data.len(),
                count * 8
            ),
        )));
    }
    let mut cursor = Cursor::new(data);
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(cursor.read_u64::<LittleEndian>()?);
    }
    Ok(result)
}

pub(crate) fn indices_to_le_bytes(
    indices: &[u32],
    index_dtype_u16: bool,
) -> Result<Vec<u8>, CodecError> {
    if index_dtype_u16 {
        let mut buf = Vec::with_capacity(indices.len() * 2);
        for &v in indices {
            if v > u16::MAX as u32 {
                return Err(CodecError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("index {} exceeds u16 range", v),
                )));
            }
            buf.write_u16::<LittleEndian>(v as u16).unwrap();
        }
        Ok(buf)
    } else {
        let mut buf = Vec::with_capacity(indices.len() * 4);
        for &v in indices {
            buf.write_u32::<LittleEndian>(v).unwrap();
        }
        Ok(buf)
    }
}

pub(crate) fn le_bytes_to_indices(
    data: &[u8],
    count: usize,
    index_dtype_u16: bool,
) -> Result<Vec<u32>, CodecError> {
    let elem_size = if index_dtype_u16 { 2 } else { 4 };
    if data.len() != count * elem_size {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "indices byte length {} != expected {}",
                data.len(),
                count * elem_size
            ),
        )));
    }
    let mut cursor = Cursor::new(data);
    let mut result = Vec::with_capacity(count);
    if index_dtype_u16 {
        for _ in 0..count {
            result.push(cursor.read_u16::<LittleEndian>()? as u32);
        }
    } else {
        for _ in 0..count {
            result.push(cursor.read_u32::<LittleEndian>()?);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers: value type conversion for Rice codec
// ---------------------------------------------------------------------------

/// Reinterpret raw LE value bytes as `Vec<u32>` according to `ValueEncoding`.
///
/// `data` is a writer-side buffer sized `n_values × width`, so a ragged tail
/// (length not a multiple of the element width) is an invariant violation, not
/// expected input. Use `chunks_exact` and reject the remainder with
/// [`CodecError::MalformedInput`] rather than silently dropping the partial
/// element the way a `while let Ok(read_…)` loop did (finding F8 — the
/// always-on form of the raggedness guard).
pub(crate) fn raw_bytes_to_u32(
    data: &[u8],
    encoding: ValueEncoding,
) -> Result<Vec<u32>, CodecError> {
    match encoding {
        ValueEncoding::Uint8 => Ok(data.iter().map(|&b| b as u32).collect()),
        ValueEncoding::Uint16 => {
            let chunks = data.chunks_exact(2);
            if !chunks.remainder().is_empty() {
                return Err(CodecError::MalformedInput(format!(
                    "Uint16 value buffer length {} is not a multiple of 2",
                    data.len()
                )));
            }
            Ok(chunks
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                .collect())
        }
        ValueEncoding::Uint32 => {
            let chunks = data.chunks_exact(4);
            if !chunks.remainder().is_empty() {
                return Err(CodecError::MalformedInput(format!(
                    "Uint32 value buffer length {} is not a multiple of 4",
                    data.len()
                )));
            }
            Ok(chunks
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        ValueEncoding::Float32 | ValueEncoding::Float16 => {
            unreachable!("raw_bytes_to_u32 called with float encoding")
        }
    }
}

/// Convert `Vec<u32>` back to raw LE bytes according to `ValueEncoding`.
pub(crate) fn u32_to_raw_bytes(
    data: &[u32],
    encoding: ValueEncoding,
) -> Result<Vec<u8>, CodecError> {
    match encoding {
        ValueEncoding::Uint8 => {
            let mut out = Vec::with_capacity(data.len());
            for &v in data {
                if v > u8::MAX as u32 {
                    return Err(CodecError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("decoded value {} exceeds u8 range", v),
                    )));
                }
                out.push(v as u8);
            }
            Ok(out)
        }
        ValueEncoding::Uint16 => {
            let mut buf = Vec::with_capacity(data.len() * 2);
            for &v in data {
                if v > u16::MAX as u32 {
                    return Err(CodecError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("decoded value {} exceeds u16 range", v),
                    )));
                }
                buf.write_u16::<LittleEndian>(v as u16).unwrap();
            }
            Ok(buf)
        }
        ValueEncoding::Uint32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_u32::<LittleEndian>(v).unwrap();
            }
            Ok(buf)
        }
        ValueEncoding::Float32 | ValueEncoding::Float16 => {
            unreachable!("u32_to_raw_bytes called with float encoding")
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
