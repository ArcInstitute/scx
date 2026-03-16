// Codec ID dispatch + zstd fallback (SPEC §4.5)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::bitstream::BitStreamError;
use crate::delta_golomb::{delta_golomb_decode, delta_golomb_encode};
use crate::forbp::{forbp_decode, forbp_encode};
use crate::rice::{rice_decode, rice_encode, B_VAL};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Identifies the compression codec used for a shard (SPEC §3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecId {
    /// Raw little-endian arrays, no compression.
    None = 0,
    /// Domain-specific: Delta-Golomb (indptr) + FOR-BP (indices) + Rice (values).
    /// Integer value encodings only.
    Scx1 = 1,
    /// Zstd compression per array. Works with any value encoding.
    Zstd = 2,
}

impl CodecId {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Scx1),
            2 => Some(Self::Zstd),
            _ => None,
        }
    }
}

/// Value encoding for the data array in a CSR shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueEncoding {
    Uint8 = 0,
    Uint16 = 1,
    Uint32 = 2,
    Float32 = 3,
    Float16 = 4,
}

impl ValueEncoding {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Uint16),
            2 => Some(Self::Uint32),
            3 => Some(Self::Float32),
            4 => Some(Self::Float16),
            _ => None,
        }
    }

    /// Number of bytes per value element.
    pub fn byte_width(&self) -> usize {
        match self {
            Self::Uint8 => 1,
            Self::Uint16 | Self::Float16 => 2,
            Self::Uint32 | Self::Float32 => 4,
        }
    }

    /// Returns `true` for integer encodings that can use the Scx1 codec.
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::Uint8 | Self::Uint16 | Self::Uint32)
    }
}

/// The encoded byte arrays for a single CSR shard.
#[derive(Debug)]
pub struct EncodedShard {
    pub indptr_bytes: Vec<u8>,
    pub indices_bytes: Vec<u8>,
    pub values_bytes: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("bitstream error: {0}")]
    BitStream(#[from] BitStreamError),

    #[error("unsupported codec id: {0}")]
    UnsupportedCodec(u8),

    #[error("unsupported value encoding: {0}")]
    UnsupportedValueEncoding(u8),

    #[error("Scx1 codec does not support float value encodings")]
    FloatWithScx1,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Decoded shard: `(indptr, indices, values_raw_bytes)`.
pub type DecodedShard = (Vec<u64>, Vec<u32>, Vec<u8>);

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode a CSR shard's three arrays using the specified codec.
///
/// - `indptr`: the indptr array (length = n_rows + 1).
/// - `indices`: the column indices (length = nnz), stored as u32.
/// - `values`: raw little-endian bytes of the value array (length = nnz × value_encoding.byte_width()).
/// - `index_dtype_u16`: if true, indices fit in u16 (n_vars <= 65535).
pub fn encode_shard(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    match codec_id {
        CodecId::None => encode_none(indptr, indices, values, index_dtype_u16),
        CodecId::Scx1 => encode_scx1(indptr, indices, values, value_encoding, index_dtype_u16),
        CodecId::Zstd => encode_zstd(indptr, indices, values, index_dtype_u16),
    }
}

/// Decode an `EncodedShard` back to `(indptr, indices, values_bytes)`.
///
/// - `n_rows`: number of rows (indptr has n_rows + 1 entries).
/// - `nnz`: number of non-zero values.
pub fn decode_shard(
    encoded: &EncodedShard,
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    n_rows: usize,
    nnz: usize,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    match codec_id {
        CodecId::None => decode_none(encoded, n_rows, nnz, value_encoding, index_dtype_u16),
        CodecId::Scx1 => decode_scx1(encoded, value_encoding, n_rows, nnz, index_dtype_u16),
        CodecId::Zstd => decode_zstd(encoded, n_rows, nnz, value_encoding, index_dtype_u16),
    }
}

// ---------------------------------------------------------------------------
// CodecId::None
// ---------------------------------------------------------------------------

fn encode_none(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    let indptr_bytes = u64_slice_to_le_bytes(indptr);
    let indices_bytes = indices_to_le_bytes(indices, index_dtype_u16);
    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes: values.to_vec(),
    })
}

fn decode_none(
    encoded: &EncodedShard,
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    let indptr = le_bytes_to_u64(&encoded.indptr_bytes, n_rows + 1)?;
    let indices = le_bytes_to_indices(&encoded.indices_bytes, nnz, index_dtype_u16)?;
    // Validate values length
    let expected_len = nnz * value_encoding.byte_width();
    if encoded.values_bytes.len() != expected_len {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "values byte length {} != expected {}",
                encoded.values_bytes.len(),
                expected_len
            ),
        )));
    }
    Ok((indptr, indices, encoded.values_bytes.clone()))
}

// ---------------------------------------------------------------------------
// CodecId::Scx1
// ---------------------------------------------------------------------------

fn encode_scx1(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    if !value_encoding.is_integer() {
        return Err(CodecError::FloatWithScx1);
    }

    // indptr → Delta-Golomb
    let indptr_bytes = delta_golomb_encode(indptr);

    // indices → FOR-BP (needs row_lengths from indptr)
    let row_lengths: Vec<usize> = indptr.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let indices_bytes = forbp_encode(indices, &row_lengths, index_dtype_u16);

    // values → reinterpret to u32, then Rice encode
    let values_u32 = raw_bytes_to_u32(values, value_encoding);
    let values_bytes = rice_encode(&values_u32, B_VAL);

    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    })
}

fn decode_scx1(
    encoded: &EncodedShard,
    value_encoding: ValueEncoding,
    n_rows: usize,
    nnz: usize,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    if !value_encoding.is_integer() {
        return Err(CodecError::FloatWithScx1);
    }

    // indptr ← Delta-Golomb
    let indptr = delta_golomb_decode(&encoded.indptr_bytes, n_rows + 1)?;

    // indices ← FOR-BP
    let (indices, _row_lengths) = forbp_decode(&encoded.indices_bytes, n_rows, index_dtype_u16)?;

    // values ← Rice decode, then convert u32 back to raw bytes
    let values_u32 = rice_decode(&encoded.values_bytes, nnz, B_VAL)?;
    let values_bytes = u32_to_raw_bytes(&values_u32, value_encoding);

    Ok((indptr, indices, values_bytes))
}

// ---------------------------------------------------------------------------
// CodecId::Zstd
// ---------------------------------------------------------------------------

fn encode_zstd(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    let indptr_raw = u64_slice_to_le_bytes(indptr);
    let indices_raw = indices_to_le_bytes(indices, index_dtype_u16);

    let indptr_bytes = zstd::encode_all(indptr_raw.as_slice(), 3)?;
    let indices_bytes = zstd::encode_all(indices_raw.as_slice(), 3)?;
    let values_bytes = zstd::encode_all(values, 3)?;

    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    })
}

fn decode_zstd(
    encoded: &EncodedShard,
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    let indptr_raw = zstd::decode_all(encoded.indptr_bytes.as_slice())?;
    let indices_raw = zstd::decode_all(encoded.indices_bytes.as_slice())?;
    let values_raw = zstd::decode_all(encoded.values_bytes.as_slice())?;

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    let expected_len = nnz * value_encoding.byte_width();
    if values_raw.len() != expected_len {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "decompressed values byte length {} != expected {}",
                values_raw.len(),
                expected_len
            ),
        )));
    }

    Ok((indptr, indices, values_raw))
}

// ---------------------------------------------------------------------------
// Helpers: serialization
// ---------------------------------------------------------------------------

fn u64_slice_to_le_bytes(data: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(data.len() * 8);
    for &v in data {
        buf.write_u64::<LittleEndian>(v).unwrap();
    }
    buf
}

fn le_bytes_to_u64(data: &[u8], count: usize) -> Result<Vec<u64>, CodecError> {
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

fn indices_to_le_bytes(indices: &[u32], index_dtype_u16: bool) -> Vec<u8> {
    if index_dtype_u16 {
        let mut buf = Vec::with_capacity(indices.len() * 2);
        for &v in indices {
            buf.write_u16::<LittleEndian>(v as u16).unwrap();
        }
        buf
    } else {
        let mut buf = Vec::with_capacity(indices.len() * 4);
        for &v in indices {
            buf.write_u32::<LittleEndian>(v).unwrap();
        }
        buf
    }
}

fn le_bytes_to_indices(
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
fn raw_bytes_to_u32(data: &[u8], encoding: ValueEncoding) -> Vec<u32> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&b| b as u32).collect(),
        ValueEncoding::Uint16 => {
            let mut cursor = Cursor::new(data);
            let mut out = Vec::with_capacity(data.len() / 2);
            while let Ok(v) = cursor.read_u16::<LittleEndian>() {
                out.push(v as u32);
            }
            out
        }
        ValueEncoding::Uint32 => {
            let mut cursor = Cursor::new(data);
            let mut out = Vec::with_capacity(data.len() / 4);
            while let Ok(v) = cursor.read_u32::<LittleEndian>() {
                out.push(v);
            }
            out
        }
        ValueEncoding::Float32 | ValueEncoding::Float16 => {
            unreachable!("raw_bytes_to_u32 called with float encoding")
        }
    }
}

/// Convert `Vec<u32>` back to raw LE bytes according to `ValueEncoding`.
fn u32_to_raw_bytes(data: &[u32], encoding: ValueEncoding) -> Vec<u8> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&v| v as u8).collect(),
        ValueEncoding::Uint16 => {
            let mut buf = Vec::with_capacity(data.len() * 2);
            for &v in data {
                buf.write_u16::<LittleEndian>(v as u16).unwrap();
            }
            buf
        }
        ValueEncoding::Uint32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_u32::<LittleEndian>(v).unwrap();
            }
            buf
        }
        ValueEncoding::Float32 | ValueEncoding::Float16 => {
            unreachable!("u32_to_raw_bytes called with float encoding")
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small CSR matrix for testing.
    /// 3 rows, varying nnz:
    ///   row 0: cols [1, 3]       vals [5, 10]
    ///   row 1: cols [0, 2, 4]    vals [1, 3, 7]
    ///   row 2: cols [2]          vals [2]
    fn make_test_csr(value_encoding: ValueEncoding) -> (Vec<u64>, Vec<u32>, Vec<u8>, usize, usize) {
        let indptr: Vec<u64> = vec![0, 2, 5, 6];
        let indices: Vec<u32> = vec![1, 3, 0, 2, 4, 2];
        let values_u32: Vec<u32> = vec![5, 10, 1, 3, 7, 2];
        let n_rows = 3;
        let nnz = 6;

        let values_bytes = match value_encoding {
            ValueEncoding::Uint8 => values_u32.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
            ValueEncoding::Uint16 => {
                let mut buf = Vec::new();
                for &v in &values_u32 {
                    buf.write_u16::<LittleEndian>(v as u16).unwrap();
                }
                buf
            }
            ValueEncoding::Uint32 => {
                let mut buf = Vec::new();
                for &v in &values_u32 {
                    buf.write_u32::<LittleEndian>(v).unwrap();
                }
                buf
            }
            ValueEncoding::Float32 => {
                let mut buf = Vec::new();
                for &v in &values_u32 {
                    buf.write_f32::<LittleEndian>(v as f32).unwrap();
                }
                buf
            }
            ValueEncoding::Float16 => {
                // For testing purposes, just use 2 bytes per value
                let mut buf = Vec::new();
                for &v in &values_u32 {
                    buf.write_u16::<LittleEndian>(v as u16).unwrap();
                }
                buf
            }
        };

        (indptr, indices, values_bytes, n_rows, nnz)
    }

    /// Task 6.7: Round-trip through each CodecId × integer ValueEncoding.
    #[test]
    fn test_roundtrip_all_integer_codecs() {
        let codecs = [CodecId::None, CodecId::Scx1, CodecId::Zstd];
        let encodings = [
            ValueEncoding::Uint8,
            ValueEncoding::Uint16,
            ValueEncoding::Uint32,
        ];

        for &codec in &codecs {
            for &enc in &encodings {
                for &u16_idx in &[true, false] {
                    let (indptr, indices, values, n_rows, nnz) = make_test_csr(enc);

                    let encoded =
                        encode_shard(&indptr, &indices, &values, codec, enc, u16_idx).unwrap();

                    let (dec_indptr, dec_indices, dec_values) =
                        decode_shard(&encoded, codec, enc, n_rows, nnz, u16_idx).unwrap();

                    assert_eq!(
                        indptr, dec_indptr,
                        "indptr mismatch: codec={codec:?} enc={enc:?} u16={u16_idx}"
                    );
                    assert_eq!(
                        indices, dec_indices,
                        "indices mismatch: codec={codec:?} enc={enc:?} u16={u16_idx}"
                    );
                    assert_eq!(
                        values, dec_values,
                        "values mismatch: codec={codec:?} enc={enc:?} u16={u16_idx}"
                    );
                }
            }
        }
    }

    /// Task 6.8: None codec produces raw LE bytes.
    #[test]
    fn test_none_produces_raw_bytes() {
        let (indptr, indices, values, _n_rows, _nnz) = make_test_csr(ValueEncoding::Uint32);
        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint32,
            false,
        )
        .unwrap();

        // indptr: 4 u64 values = 32 bytes
        assert_eq!(encoded.indptr_bytes.len(), 4 * 8);
        // First u64 should be 0
        let mut cursor = Cursor::new(&encoded.indptr_bytes);
        assert_eq!(cursor.read_u64::<LittleEndian>().unwrap(), 0);
        assert_eq!(cursor.read_u64::<LittleEndian>().unwrap(), 2);
        assert_eq!(cursor.read_u64::<LittleEndian>().unwrap(), 5);
        assert_eq!(cursor.read_u64::<LittleEndian>().unwrap(), 6);

        // indices: 6 u32 values = 24 bytes (index_dtype_u16=false)
        assert_eq!(encoded.indices_bytes.len(), 6 * 4);
        let mut cursor = Cursor::new(&encoded.indices_bytes);
        assert_eq!(cursor.read_u32::<LittleEndian>().unwrap(), 1);
        assert_eq!(cursor.read_u32::<LittleEndian>().unwrap(), 3);

        // values: pass-through
        assert_eq!(encoded.values_bytes, values);
    }

    /// Task 6.9: Scx1 + Float32 returns error.
    #[test]
    fn test_scx1_float32_error() {
        let (indptr, indices, values, _n_rows, _nnz) = make_test_csr(ValueEncoding::Float32);
        let result = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Scx1,
            ValueEncoding::Float32,
            false,
        );
        assert!(matches!(result, Err(CodecError::FloatWithScx1)));

        // Also test decode path
        let encoded = EncodedShard {
            indptr_bytes: vec![],
            indices_bytes: vec![],
            values_bytes: vec![],
        };
        let result = decode_shard(&encoded, CodecId::Scx1, ValueEncoding::Float32, 3, 6, false);
        assert!(matches!(result, Err(CodecError::FloatWithScx1)));
    }

    /// Task 6.10: Zstd + Float32 round-trips correctly.
    #[test]
    fn test_zstd_float32_roundtrip() {
        let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Float32);
        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Zstd,
            ValueEncoding::Float32,
            false,
        )
        .unwrap();
        let (dec_indptr, dec_indices, dec_values) = decode_shard(
            &encoded,
            CodecId::Zstd,
            ValueEncoding::Float32,
            n_rows,
            nnz,
            false,
        )
        .unwrap();

        assert_eq!(indptr, dec_indptr);
        assert_eq!(indices, dec_indices);
        assert_eq!(values, dec_values);
    }

    /// Test None with u16 indices.
    #[test]
    fn test_none_u16_indices() {
        let (indptr, indices, values, _n_rows, _nnz) = make_test_csr(ValueEncoding::Uint16);
        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint16,
            true,
        )
        .unwrap();

        // indices: 6 u16 values = 12 bytes
        assert_eq!(encoded.indices_bytes.len(), 6 * 2);
    }

    /// Test CodecId and ValueEncoding from_u8 helpers.
    #[test]
    fn test_from_u8_helpers() {
        assert_eq!(CodecId::from_u8(0), Some(CodecId::None));
        assert_eq!(CodecId::from_u8(1), Some(CodecId::Scx1));
        assert_eq!(CodecId::from_u8(2), Some(CodecId::Zstd));
        assert_eq!(CodecId::from_u8(3), None);

        assert_eq!(ValueEncoding::from_u8(0), Some(ValueEncoding::Uint8));
        assert_eq!(ValueEncoding::from_u8(4), Some(ValueEncoding::Float16));
        assert_eq!(ValueEncoding::from_u8(5), None);

        assert_eq!(ValueEncoding::Uint8.byte_width(), 1);
        assert_eq!(ValueEncoding::Uint16.byte_width(), 2);
        assert_eq!(ValueEncoding::Float32.byte_width(), 4);
        assert!(ValueEncoding::Uint32.is_integer());
        assert!(!ValueEncoding::Float32.is_integer());
    }
}
