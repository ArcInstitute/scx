// Backed (on-demand, mmap-backed) PyO3 dataset classes.
//
// This directory was split out of the former monolithic pyscx/src/backed.rs
// (T5.7), one cohesive PyO3 class (plus its helpers) per file:
//
// - `sparse_dataset` ScxBackedSparseDataset — on-demand sparse X access.
// - `layer`          ScxBackedLayerDataset — backed access to an AnnData layer.
// - `comparison`     ScxComparisonResult + row-factor scaling helpers.
// - `multimodal`     ScxBackedMuDataset / ScxBackedMuModality.
// - `obsm`           ScxBackedObsmDataset — backed obsm/varm embeddings.
// - `row_selector`   the one bool-mask / int-array row resolver every handle uses.
//
// Submodules reach their siblings through the glob re-exports below
// (`use super::*`); the public class names are re-exported at `crate::backed`
// so existing call sites and the `add_class` registrations in lib.rs are
// unchanged.

pub(crate) mod comparison;
pub(crate) mod layer;
pub(crate) mod multimodal;
pub(crate) mod obsm;
pub(crate) mod row_selector;
pub(crate) mod sparse_dataset;

pub(crate) use comparison::*;
pub(crate) use layer::*;
pub(crate) use multimodal::*;
pub(crate) use obsm::*;
pub(crate) use row_selector::*;
pub(crate) use sparse_dataset::*;
