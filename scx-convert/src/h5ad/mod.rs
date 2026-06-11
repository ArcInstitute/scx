// h5ad (AnnData) ingest/export: the reader, writer, and their streaming
// variants, plus the dense/CSC stream helpers and the CSC→CSR transpose used
// only by the h5ad reader. Crate-root `lib.rs` re-exports the public surface
// from here, so `scx_convert::*` paths are unchanged.

pub mod csc_stream;
pub(crate) mod csc_transpose;
pub mod dense_stream;
pub mod read;
pub mod stream;
pub mod stream_write;
pub mod write;
