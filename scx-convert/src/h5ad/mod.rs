// h5ad (AnnData) ingest/export: the reader, writer, and their streaming
// variants, plus the dense/CSC stream helpers and the CSC→CSR transpose used
// only by the h5ad reader. Crate-root `lib.rs` re-exports the public surface
// from here, so `scx_convert::*` paths are unchanged.
//
// The export side was one 2908-line `write.rs` until ORG-11.16-6. It is now
// five files, because 64 % of it was the dataframe column writer and none of
// that concern was reachable from -- or reached into -- the matrix and `uns`
// writers around it:
//
// | Module          | Holds                                                    |
// |-----------------|----------------------------------------------------------|
// | `write`         | the SCX->h5ad drivers, `/X`, `/raw`, `/obsm`, `/obsp`    |
// | `columns`       | the dataframe layout / schema pre-pass                   |
// | `column_stream` | the nine column encodings and their two drivers          |
// | `categorical`   | Arrow dictionary -> h5ad categorical, shared by both     |
// | `uns`           | `/uns`: JSON -> HDF5                                     |
//
// On the ingest side `strings` is the one decoder for 1-D HDF5 string
// *datasets*: every flavour, one const-generic width ladder, one HDF5 read, and
// a sink that decides the output shape. Two readers had already hand-rolled
// that ladder and lost the fixed-width half of it. Not every string read routes
// through it -- rank-0 (scalar) reads, `uns`'s fixed-width 1-D arm and
// attribute reads decode themselves; that module's docs list all three,
// including the same fixed-width drift still live on the scalar path.

pub(crate) mod categorical;
pub(crate) mod column_stream;
pub(crate) mod columns;
pub(crate) mod csc_stream;
pub(crate) mod csc_transpose;
pub(crate) mod dense_stream;
pub(crate) mod read;
pub(crate) mod stream;
pub(crate) mod stream_write;
pub(crate) mod strings;
pub(crate) mod uns;
pub(crate) mod uns_dataframe;
pub(crate) mod write;
