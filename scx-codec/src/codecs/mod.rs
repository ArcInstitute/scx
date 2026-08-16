//! One module per codec, split out of `dispatch.rs` (ORG-3.7-2).
//!
//! Each exposes an `encode_*` / `decode_*_ref` pair with the same shape; the
//! dispatch driver in [`crate::dispatch`] selects between them.

pub(crate) mod lz4_shuffle;
pub(crate) mod none;
pub(crate) mod pcodec;
pub(crate) mod scx1;
pub(crate) mod shufdelta;
pub(crate) mod zstd_codec;
