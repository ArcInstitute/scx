// External format <-> SCX conversion.
// Shared by scx-cli and pyscx; the `hdf5` feature gates the h5ad /
// h5mu / 10x readers and writers so consumers that don't need them
// can build without libhdf5.

mod direction;
pub use direction::determine_convert_direction;

#[cfg(feature = "hdf5")]
mod csc_stream;
#[cfg(feature = "hdf5")]
mod csc_transpose;
#[cfg(feature = "hdf5")]
mod dense_stream;
#[cfg(feature = "hdf5")]
mod detect;
#[cfg(feature = "hdf5")]
mod dtype;
#[cfg(feature = "hdf5")]
mod h5ad_read;
#[cfg(feature = "hdf5")]
mod h5ad_stream;
#[cfg(feature = "hdf5")]
mod h5ad_stream_write;
#[cfg(feature = "hdf5")]
mod h5ad_write;
#[cfg(feature = "hdf5")]
mod hdf_dtype;
#[cfg(feature = "hdf5")]
mod mudata_pipeline;
#[cfg(feature = "hdf5")]
mod mudata_write;
#[cfg(feature = "hdf5")]
mod tenx_read;

#[cfg(feature = "hdf5")]
mod hdf5_threadsafe;
#[cfg(feature = "hdf5")]
mod stream;
mod warnings;

#[cfg(feature = "hdf5")]
pub use csc_stream::{open_csc_layer_streaming, open_csc_streaming};
#[cfg(feature = "hdf5")]
pub use dense_stream::{open_dense_layer_streaming, open_dense_streaming, DenseXStreamReader};
#[cfg(feature = "hdf5")]
pub use h5ad_stream::{open_layer_streaming, open_x_streaming, CsrShardSlice, XStreamReader};

#[cfg(feature = "hdf5")]
pub use h5ad_read::{
    read_dataframe_group, read_h5ad_metadata_from_path, read_h5ad_x_shape,
    read_h5ad_x_shape_from_path, read_uns, H5adMetadataParts,
};

/// Arrow `Field::metadata` key carrying a categorical column's pandas
/// `ordered` bit. Arrow's `DictionaryArray` has no `ordered` flag, so the
/// h5ad reader stamps it here and the h5ad writer / `to_anndata` re-apply it.
///
/// Defined at the crate root (not in the `hdf5`-gated `h5ad_read` module) so
/// `pyscx` can reference it from the always-compiled `to_anndata` path
/// regardless of which scx-convert features are enabled.
pub const CATEGORICAL_ORDERED_KEY: &str = "scx.categorical.ordered";

// Re-exported from scx-format so existing `scx_convert::MemoryBudget`
// call sites keep working; the parser lives in scx-format so sibling
// crates (scx-ops) can share it without a dependency cycle.
pub use scx_format::MemoryBudget;
#[cfg(feature = "hdf5")]
pub use stream::{CsrShardStream, MajorAxis, StreamedCsrShard};
pub use warnings::{ConvertWarning, WarningSink};

// CSC policy re-export is ungated: the always-available MTX → SCX path
// (no `hdf5` feature) drives it too, so it must not live behind the
// hdf5-gated `pipeline` module.
pub use scx_format::CscPolicy;

#[cfg(feature = "hdf5")]
pub mod pipeline;

#[cfg(feature = "hdf5")]
pub use pipeline::{
    h5ad_to_scx, h5ad_to_scx_streaming, run_streaming_writer_coordinator, scx_to_h5ad,
    scx_to_h5ad_streaming, streaming_writer_coordinator, tenx_to_scx, BitmapPolicy, ConvertError,
    ConvertOptions, StreamingOverrides,
};

#[cfg(feature = "hdf5")]
pub use mudata_pipeline::{h5mu_to_scx, h5mu_to_scx_streaming, is_h5mu_file};

#[cfg(feature = "hdf5")]
pub use mudata_write::{
    scx_modality_to_h5ad, scx_modality_to_h5ad_streaming, scx_to_h5mu, scx_to_h5mu_streaming,
};

pub mod mtx_pipeline;
pub use scx_mtx::MtxOrientation;

#[cfg(all(test, feature = "hdf5"))]
mod tests;

#[cfg(test)]
mod mtx_tests;
