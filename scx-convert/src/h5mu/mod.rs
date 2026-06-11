// h5mu (MuData) ingest/export: the multimodal pipeline and writer. Builds on
// the h5ad reader/writer/stream modules under `crate::h5ad`. Crate-root
// `lib.rs` re-exports the public surface from here.

pub mod pipeline;
pub mod write;
