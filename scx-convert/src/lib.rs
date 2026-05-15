// External format <-> SCX conversion.
// Extracted from scx-cli so pyscx can call into the same code (see
// STREAMING-CONVERSION.md). Behaviour is unchanged from the prior
// scx-cli/src/convert/ module.

mod direction;
pub use direction::determine_convert_direction;

#[cfg(feature = "hdf5")]
mod csc_transpose;
#[cfg(feature = "hdf5")]
mod detect;
#[cfg(feature = "hdf5")]
mod dtype;
#[cfg(feature = "hdf5")]
mod h5ad_read;
#[cfg(feature = "hdf5")]
mod h5ad_stream;
#[cfg(feature = "hdf5")]
mod h5ad_write;
#[cfg(feature = "hdf5")]
mod mudata_pipeline;
#[cfg(feature = "hdf5")]
mod mudata_write;
#[cfg(feature = "hdf5")]
mod tenx_read;

#[cfg(feature = "hdf5")]
pub use h5ad_stream::{open_layer_streaming, open_x_streaming, CsrShardSlice, XStreamReader};

#[cfg(feature = "hdf5")]
pub mod pipeline;

#[cfg(feature = "hdf5")]
pub use pipeline::{
    h5ad_to_scx, h5ad_to_scx_streaming, scx_to_h5ad, tenx_to_scx, ConvertError, ConvertOptions,
    StreamingOverrides,
};

#[cfg(feature = "hdf5")]
pub use mudata_pipeline::{h5mu_to_scx, is_h5mu_file};

#[cfg(feature = "hdf5")]
pub use mudata_write::{scx_modality_to_h5ad, scx_to_h5mu};

pub mod mtx_pipeline;

#[cfg(all(test, feature = "hdf5"))]
mod tests;

#[cfg(test)]
mod mtx_tests;
