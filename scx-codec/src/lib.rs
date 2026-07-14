pub mod bitstream;
pub mod byte_delta;
pub mod cast;
pub mod delta_golomb;
pub mod dispatch;
pub mod forbp;
pub mod median;
pub mod rice;
pub mod shuffle;
#[cfg(target_arch = "x86_64")]
pub(crate) mod simd;
pub mod value_encoding;

pub use byte_delta::{byte_delta_planes, byte_undelta_planes};
pub use cast::{
    checked_cast_indices, checked_cast_values, guard_f32_decode_loss, CastFromF32, CastFromI32,
    F32_MAX_EXACT_INT,
};
pub use dispatch::{
    decode_indptr_only, decode_row_group, decode_row_group_indptr_only, decode_shard,
    decode_shard_ref, decode_shard_scipy, decoded_shard_to_scipy, encode_shard,
    zstd_decode_bounded as zstd_decompress_bounded, CodecError, CodecId, CodecSelection,
    DecodedShard, EncodedShard, EncodedShardRef, RowGroupSpan, ScipyShard, ValueEncoding,
};
pub use median::{
    floor_median_u32, floor_median_u32_inplace, floor_median_u64, floor_median_u64_inplace,
};
pub use value_encoding::{
    detect_value_encoding, detect_value_encoding_f64, is_integer_data, values_to_raw_bytes,
};
