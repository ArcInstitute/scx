pub mod bitstream;
pub mod delta_golomb;
pub mod dispatch;
pub mod forbp;
pub mod rice;
pub mod shuffle;
pub mod value_encoding;

pub use dispatch::{
    decode_shard, decode_shard_ref, decode_shard_scipy, encode_shard, CodecError, CodecId,
    DecodedShard, EncodedShard, EncodedShardRef, ScipyShard, ValueEncoding,
};
pub use value_encoding::{detect_value_encoding, is_integer_data, values_to_raw_bytes};
