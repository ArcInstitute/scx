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
mod mem;
#[cfg(feature = "hdf5")]
mod stream;
#[cfg(feature = "hdf5")]
mod warnings;

#[cfg(feature = "hdf5")]
pub use csc_stream::{open_csc_layer_streaming, open_csc_streaming};
#[cfg(feature = "hdf5")]
pub use dense_stream::{open_dense_layer_streaming, open_dense_streaming, DenseXStreamReader};
#[cfg(feature = "hdf5")]
pub use h5ad_stream::{open_layer_streaming, open_x_streaming, CsrShardSlice, XStreamReader};

#[cfg(feature = "hdf5")]
pub use mem::MemoryBudget;
#[cfg(feature = "hdf5")]
pub use stream::{CsrShardStream, MajorAxis, StreamedCsrShard};
#[cfg(feature = "hdf5")]
pub use warnings::{ConvertWarning, WarningSink};

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

#[cfg(all(test, feature = "hdf5"))]
mod tests;

#[cfg(test)]
mod mtx_tests;
