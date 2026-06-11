pub mod bitstream;
pub mod delta_golomb;
pub mod dispatch;
pub mod forbp;
pub mod median;
pub mod rice;
pub mod shuffle;
pub mod value_encoding;

pub use dispatch::{
    decode_indptr_only, decode_scx1_row_range, decode_scx1_with_metadata, decode_shard,
    decode_shard_ref, decode_shard_scipy, decoded_shard_to_scipy, encode_shard, CodecError,
    CodecId, CodecSelection, DecodedShard, EncodedShard, EncodedShardRef, ScipyShard,
    Scx1DecodeMetadata, ValueEncoding,
};
pub use median::{
    floor_median_u32, floor_median_u32_inplace, floor_median_u64, floor_median_u64_inplace,
};
pub use value_encoding::{
    detect_value_encoding, detect_value_encoding_f64, is_integer_data, values_to_raw_bytes,
};
