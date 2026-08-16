//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.

use crate::codec_id::{CodecError, DecodedShard, EncodedShard, EncodedShardRef, ValueEncoding};
use crate::codecs::zstd_codec::zstd_decode_bounded;
use crate::guards::checked_len;
use crate::shard_codec::{DecodeBounds, ShardCodec, ShardShape};

/// `CodecId::Pcodec` — pcodec for float values, zstd for indptr/indices.
pub struct PcodecCodec;

impl ShardCodec for PcodecCodec {
    fn encode(
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
    ) -> Result<EncodedShard, CodecError> {
        encode_pcodec(indptr, indices, values, value_encoding, index_dtype_u16)
    }

    fn decode(
        encoded: &EncodedShardRef,
        shape: ShardShape,
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
        bounds: &DecodeBounds,
    ) -> Result<DecodedShard, CodecError> {
        decode_pcodec_ref(encoded, shape, value_encoding, index_dtype_u16, bounds)
    }

    fn decode_indptr_only(
        indptr_bytes: &[u8],
        n_rows: usize,
        indptr_max: usize,
    ) -> Result<Vec<u64>, CodecError> {
        let raw = zstd_decode_bounded(indptr_bytes, indptr_max)?;
        le_bytes_to_u64(&raw, n_rows + 1)
    }
}

use crate::raw::{
    indices_to_le_bytes, le_bytes_to_indices, le_bytes_to_u64, u64_slice_to_le_bytes,
};

fn encode_pcodec(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    let indptr_raw = u64_slice_to_le_bytes(indptr);
    let indices_raw = indices_to_le_bytes(indices, index_dtype_u16)?;

    // indptr and indices: Zstd (already well-compressed by generic codecs)
    let indptr_bytes = zstd::encode_all(indptr_raw.as_slice(), 3)?;
    let indices_bytes = zstd::encode_all(indices_raw.as_slice(), 3)?;

    // values: Pcodec for float encodings, Zstd for integer encodings
    let values_bytes = match value_encoding {
        ValueEncoding::Float32 => {
            let floats: Vec<f32> = values
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            pco::standalone::simple_compress(&floats, &pco::ChunkConfig::default())
                .map_err(|e| CodecError::Io(std::io::Error::other(e.to_string())))?
        }
        ValueEncoding::Float16 => {
            // Widen f16 to f32, then compress as f32
            let floats: Vec<f32> = values
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            pco::standalone::simple_compress(&floats, &pco::ChunkConfig::default())
                .map_err(|e| CodecError::Io(std::io::Error::other(e.to_string())))?
        }
        _ => {
            // Integer encodings: Zstd (Pcodec advantage is on floats)
            zstd::encode_all(values, 3)?
        }
    };

    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    })
}

/// Decompress exactly `n_values` f32s from a pcodec stream.
///
/// Two opposite hostile shapes meet here, and a guard for one is not a guard
/// for the other:
///
/// - **Stream larger than declared.** `pco::standalone::simple_decompress`
///   sizes its output from the *stream*, so a shard declaring two values could
///   decompress a million and the length check downstream would fire one full
///   allocation too late. Bounded below by refusing to append past `n_values`.
/// - **Declared larger than stream.** Sizing the destination from `n_values`
///   instead inverts the problem — see the `n_values` note below. Bounded by
///   growing `out` only as pco actually produces numbers.
///
/// **Do not add a compression-ratio plausibility bound to this function.**
/// [`bound_capacity`] is sound only for the Scx1 primitives, whose Golomb/Rice
/// codes spend at least one input bit per output value. pcodec is an entropy
/// coder with no such floor: a constant run costs ~0 bits/value, and ordinary
/// single-cell payloads sit under it too — raw counts stored as `f32` (the
/// standard AnnData layout, ~80% ones) measure ≈0.96 bits/value. Applying the
/// 1-bit floor here rejected shards this crate's *own encoder* had just
/// produced, at every realistic shard size. See
/// `test_pcodec_low_entropy_float_roundtrip`.
///
/// **`n_values` is not trustworthy, and must not size the destination.** It is
/// a shard-header field. It is tempting to argue it has already been
/// corroborated, because `decode_pcodec_ref` decodes the indices sub-stream
/// first and `le_bytes_to_indices` demands an *exact* `n_values * index_width`
/// bytes — but those are *decompressed* bytes, and Zstd will expand a ~2 KiB
/// run of zero indices into exactly the `n_values * width` the check wants. A
/// `vec![0f32; n_values]` sized from the header therefore hands a few kilobytes
/// of input a second allocation of the same magnitude as the indices bomb that
/// let it through: measured at 165 MB of peak RSS from a 2.5 KiB shard before
/// this loop replaced it.
///
/// So the destination grows with numbers pco has *actually produced*, capped at
/// `n_values`, which is the same shape `zstd_decode_bounded` uses — eager
/// capacity clamped, `read_to_end` growing against a `take` limiter. `read`
/// wants a `dst` whose length is a multiple of 256 (or ≥ the chunk remainder),
/// hence the fixed slab.
pub(crate) fn pcodec_decompress_bounded(
    data: &[u8],
    n_values: usize,
) -> Result<Vec<f32>, CodecError> {
    use pco::standalone::{DecompressorItem, FileDecompressor};

    // 65536 = 256 × 256, satisfying `ChunkDecompressor::read`'s stride rule.
    const SLAB: usize = 1 << 16;

    let pco_err = |e: pco::errors::PcoError| CodecError::Io(std::io::Error::other(e.to_string()));
    let too_many = || {
        CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("decompressed pcodec values exceeds limit {n_values}"),
        ))
    };

    // Overflow guard only — `n_values` still bounds the *total*, it just never
    // gets to be an up-front allocation.
    checked_len(n_values, std::mem::size_of::<f32>(), "pcodec values")?;

    let mut out: Vec<f32> = Vec::with_capacity(n_values.min(SLAB));
    // Round up to `read`'s 256 stride, but never overshoot a small shard: a
    // fixed 256 KiB slab made a 5k-value shard decode ~1.5× slower than the
    // eager path it replaces. `checked_len` above keeps `next_multiple_of`
    // clear of overflow.
    //
    // `read` accepts a `dst` that is a multiple of 256 *or* at least the
    // chunk's remaining count. Only the **first** clause is what makes this
    // sound, and it always holds: `.max(256)` keeps the slab non-zero and both
    // `next_multiple_of(256)` and `SLAB` are multiples of 256. The second
    // clause is *not* generally true here — a default pco page is 1<<18
    // values, which exceeds the 1<<16 slab — so do not lean on it.
    let slab_len = SLAB.min(n_values.next_multiple_of(256).max(256));
    let mut slab = vec![0f32; slab_len];
    let (fd, mut src) = FileDecompressor::new(data).map_err(pco_err)?;

    loop {
        match fd.chunk_decompressor::<f32, _>(src).map_err(pco_err)? {
            DecompressorItem::EndOfData(_) => break,
            DecompressorItem::Chunk(mut cd) => {
                loop {
                    let progress = cd.read(&mut slab).map_err(pco_err)?;
                    // Refuse *before* appending, so an over-long stream can
                    // never grow `out` past the declared shape.
                    if out.len() + progress.n_processed > n_values {
                        return Err(too_many());
                    }
                    out.extend_from_slice(&slab[..progress.n_processed]);
                    if progress.finished {
                        break;
                    }
                    if progress.n_processed == 0 {
                        // Defensive: a chunk that reports neither progress nor
                        // completion would otherwise spin forever.
                        return Err(CodecError::MalformedInput(
                            "pcodec chunk made no progress".to_string(),
                        ));
                    }
                }
                src = cd.into_src();
            }
        }
    }

    if out.len() != n_values {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "decompressed pcodec value count {} != expected {n_values}",
                out.len()
            ),
        )));
    }
    Ok(out)
}

fn decode_pcodec_ref(
    encoded: &EncodedShardRef,
    shape: ShardShape,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    bounds: &DecodeBounds,
) -> Result<DecodedShard, CodecError> {
    let (n_rows, nnz) = (shape.n_rows, shape.nnz);
    // indptr and indices: Zstd decompress, capped by the driver's bounds.
    let DecodeBounds {
        indptr_max,
        indices_max,
        ..
    } = *bounds;

    let indptr_raw = zstd_decode_bounded(encoded.indptr_bytes, indptr_max)?;
    let indices_raw = zstd_decode_bounded(encoded.indices_bytes, indices_max)?;

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    // values: Pcodec for float encodings, Zstd for integer encodings
    let values_raw = match value_encoding {
        ValueEncoding::Float32 => {
            let floats = pcodec_decompress_bounded(encoded.values_bytes, nnz)?;
            let mut buf = Vec::with_capacity(floats.len() * 4);
            for &f in &floats {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            buf
        }
        ValueEncoding::Float16 => {
            // Decompress as f32, narrow back to f16
            let floats = pcodec_decompress_bounded(encoded.values_bytes, nnz)?;
            let mut buf = Vec::with_capacity(floats.len() * 2);
            for &f in &floats {
                buf.extend_from_slice(&half::f16::from_f32(f).to_le_bytes());
            }
            buf
        }
        _ => {
            // Integer encodings: Zstd decompress
            let values_max = checked_len(nnz, value_encoding.byte_width(), "values")?;
            zstd_decode_bounded(encoded.values_bytes, values_max)?
        }
    };

    let expected_len = checked_len(nnz, value_encoding.byte_width(), "values")?;
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
