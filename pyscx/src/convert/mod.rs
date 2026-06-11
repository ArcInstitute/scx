// AnnData <-> SCX conversion bindings.
//
// This directory was split out of the former monolithic pyscx/src/anndata.rs
// (T5.7) along responsibility lines, mirroring the well-factored
// pyscx/src/accel/ layout:
//
// - `interop`      Arrow / pyarrow / pandas / scipy interchange helpers.
// - `to_anndata`   SCX -> AnnData assembly (eager and backed).
// - `dtype`        Codec / value-encoding / dtype conversion helpers.
// - `warnings`     Conversion-warning bridging to Python warnings.
// - `uns`          uns serialize/deserialize (Python <-> JSON, plain & tagged).
// - `from_anndata` AnnData -> SCX rewrite: inline write path + from_anndata_impl.
// - `scx_to_scx`   SCX -> SCX streaming rewrite (backed / lazy sources).
// - `h5ad`         Backed / HDF5 AnnData routing to the streaming converter.
// - `multimodal`   Per-modality backed AnnData assembly.
//
// All items are crate-internal; submodules reach their siblings through the
// glob re-exports below (`use crate::convert::*`).

pub(crate) mod dtype;
pub(crate) mod from_anndata;
pub(crate) mod h5ad;
pub(crate) mod interop;
pub(crate) mod multimodal;
pub(crate) mod scx_to_scx;
pub(crate) mod to_anndata;
pub(crate) mod uns;
pub(crate) mod warnings;

pub(crate) use dtype::*;
pub(crate) use from_anndata::*;
pub(crate) use h5ad::*;
pub(crate) use interop::*;
pub(crate) use multimodal::*;
pub(crate) use scx_to_scx::*;
pub(crate) use to_anndata::*;
pub(crate) use uns::*;
pub(crate) use warnings::*;
