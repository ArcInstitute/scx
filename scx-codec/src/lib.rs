pub mod bitstream;
pub mod delta_golomb;
pub mod dispatch;
pub mod forbp;
pub mod rice;
pub mod shuffle;

pub use dispatch::{
    decode_shard, decode_shard_ref, decode_shard_scipy, encode_shard, CodecError, CodecId,
    DecodedShard, EncodedShard, EncodedShardRef, ScipyShard, ValueEncoding,
};
