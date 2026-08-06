// External format <-> SCX conversion.
// Shared by scx-cli and pyscx; the `hdf5` feature gates the h5ad /
// h5mu / 10x readers and writers so consumers that don't need them
// can build without libhdf5.

mod direction;
pub use direction::{
    determine_convert_direction, direction_supports_streaming, resolve_stream, CONVERT_DIRECTIONS,
};

// Per-format ingest/export submodule directories (T5.9): `h5ad/` owns the
// AnnData reader/writer/stream files plus the dense/CSC stream helpers and the
// CSC→CSR transpose; `h5mu/` owns the MuData pipeline + writer. The public
// surface is re-exported below at the unchanged `scx_convert::*` paths.
#[cfg(feature = "hdf5")]
mod detect;
#[cfg(feature = "hdf5")]
mod dtype;
#[cfg(feature = "hdf5")]
mod h5_write_util;
#[cfg(feature = "hdf5")]
mod h5ad;
#[cfg(feature = "hdf5")]
mod h5mu;
#[cfg(feature = "hdf5")]
mod hdf_dtype;
#[cfg(feature = "hdf5")]
mod tenx_read;

// Ungated: a delimited-table reader must not require libhdf5, or `scx
// obs-import` would be unavailable in a no-hdf5 build.
mod annotation_table;
pub use annotation_table::{
    read_annotation_table, read_obs_source, sniff_obs_source, AnnotationTableInfo,
    AnnotationTableOptions, ObsSourceFormat,
};

// The h5ad `/obs` reader behind `read_obs_source`'s H5ad arm. Gated like every
// other h5ad path; the dispatcher keeps a clear error without it.
#[cfg(feature = "hdf5")]
mod h5ad_obs;
#[cfg(feature = "hdf5")]
pub use h5ad_obs::read_h5ad_obs;

mod doublet;
pub use doublet::{
    doublet_profile, profile_has_call_column, read_doublet_table, CallTokens, DoubletImportOptions,
    DoubletProfile, DoubletTableInfo, MissingCallColumn, DOUBLET_PROFILE_NAMES,
};

// Shared by the gated cellbender reader and the ungated table reader.
mod file_checksum;

#[cfg(feature = "hdf5")]
mod cellbender;
#[cfg(feature = "hdf5")]
pub use cellbender::{
    is_cellbender_h5, read_cellbender_h5, CellBenderInfo, CellBenderOutput, CellBenderOutputKind,
    CellBenderReadOptions, FeatureKey, LatentAlignment,
};

#[cfg(feature = "hdf5")]
mod export_filter;
#[cfg(feature = "hdf5")]
pub use export_filter::min_counts_obs_mask;

#[cfg(feature = "hdf5")]
mod hdf5_threadsafe;
#[cfg(feature = "hdf5")]
mod permuted_reader;
#[cfg(feature = "hdf5")]
mod stream;
mod warnings;

#[cfg(feature = "hdf5")]
pub use h5ad::csc_stream::{open_csc_layer_streaming, open_csc_streaming};
#[cfg(feature = "hdf5")]
pub use h5ad::dense_stream::{
    open_dense_layer_streaming, open_dense_streaming, DenseXStreamReader,
};
#[cfg(feature = "hdf5")]
pub use h5ad::stream::{open_layer_streaming, open_x_streaming, CsrShardSlice, XStreamReader};

#[cfg(feature = "hdf5")]
pub use h5ad::read::{
    read_dataframe_group, read_h5ad_metadata_from_path, read_h5ad_x_shape,
    read_h5ad_x_shape_from_path, read_uns, H5adMetadataParts,
};

/// Arrow `Field::metadata` key carrying a categorical column's pandas
/// `ordered` bit. Arrow's `DictionaryArray` has no `ordered` flag, so the
/// h5ad reader stamps it here and the h5ad writer / `to_anndata` re-apply it.
///
/// Canonically defined in `scx-format` (the shared dep of every binding) and
/// re-exported here so existing `scx_convert::CATEGORICAL_ORDERED_KEY`
/// references — including `pyscx`'s always-compiled `to_anndata` path — keep
/// working regardless of which scx-convert features are enabled.
pub use scx_format_io::CATEGORICAL_ORDERED_KEY;

// Re-exported from scx-format so existing `scx_convert::MemoryBudget`
// call sites keep working; the parser lives in scx-format so sibling
// crates (scx-ops) can share it without a dependency cycle.
pub use scx_format_io::MemoryBudget;
#[cfg(feature = "hdf5")]
pub use stream::{CsrShardStream, MajorAxis, StreamedCsrShard};
pub use warnings::{ConvertWarning, WarningSink};

// CSC policy re-export is ungated: the always-available MTX → SCX path
// (no `hdf5` feature) drives it too, so it must not live behind the
// hdf5-gated `pipeline` module.
pub use scx_format_io::CscPolicy;

/// Strategy for realizing convert-time grouping (`--group-pass`). See
/// [`pipeline::ConvertOptions::group_pass`]. Defined at the crate root (not in
/// the hdf5-gated `pipeline` module) because the CLI's non-hdf5 build path
/// parses and threads it before dispatch, like [`CscPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupPass {
    /// Route by source density: CSR → one-pass streaming, dense → two-pass.
    #[default]
    Auto,
    /// Always stream the grouped layout in one pass (the random-row gather).
    One,
    /// Always plain-convert then `scx sort --group-by` (two writes).
    Two,
}

impl GroupPass {
    /// Parse the CLI / kwarg string. Unknown values error.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(GroupPass::Auto),
            "one" | "1" | "one-pass" => Ok(GroupPass::One),
            "two" | "2" | "two-pass" => Ok(GroupPass::Two),
            other => Err(format!(
                "invalid group-pass '{other}'; expected auto, one, or two"
            )),
        }
    }
}

#[cfg(feature = "hdf5")]
pub mod pipeline;

#[cfg(feature = "hdf5")]
pub use pipeline::{
    codec_selection_json, h5ad_to_scx, h5ad_to_scx_streaming, run_streaming_writer_coordinator,
    scx_to_h5ad, scx_to_h5ad_streaming, streaming_writer_coordinator, tenx_to_scx, BitmapPolicy,
    ConvertError, ConvertOptions, StreamingOverrides,
};

#[cfg(feature = "hdf5")]
pub use h5mu::pipeline::{h5mu_to_scx, h5mu_to_scx_streaming, is_h5mu_file};

#[cfg(feature = "hdf5")]
pub use h5mu::write::{
    scx_modality_to_h5ad, scx_modality_to_h5ad_streaming, scx_to_h5mu, scx_to_h5mu_streaming,
};

pub mod mtx_pipeline;
pub use scx_mtx::MtxOrientation;

// Integration tests, split by subject (T5.6). All are gated on `hdf5` and stay
// crate-root submodules (white-box access to crate internals via `super::`);
// `convert_tests_common` holds shared fixtures/helpers + re-exported imports.
#[cfg(all(test, feature = "hdf5"))]
mod convert_tests_common;
#[cfg(all(test, feature = "hdf5"))]
mod convert_tests_dataframe;
#[cfg(all(test, feature = "hdf5"))]
mod convert_tests_export_filter;
#[cfg(all(test, feature = "hdf5"))]
mod convert_tests_group;
#[cfg(all(test, feature = "hdf5"))]
mod convert_tests_h5ad;
#[cfg(all(test, feature = "hdf5"))]
mod convert_tests_index_export;
#[cfg(all(test, feature = "hdf5"))]
mod convert_tests_parallel;
#[cfg(all(test, feature = "hdf5"))]
mod convert_tests_sort;
#[cfg(all(test, feature = "hdf5"))]
mod convert_tests_streaming;

#[cfg(test)]
mod mtx_tests;
