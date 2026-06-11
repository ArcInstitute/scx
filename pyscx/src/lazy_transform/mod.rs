// Lazy per-row transform datasets and streaming sources.
//
// This directory was split out of the former monolithic
// pyscx/src/lazy_transform.rs (T5.7):
//
// - `transform`     Transform enum — NormalizeTotal / Log1p / RowScale.
// - `dataset`       ScxLazyTransformedDataset — the PyO3 lazy-transform class.
// - `transforms`    Free transform-application helpers over CSR shards.
// - `shard_source`  LazyShardSource — ShardSource / ColumnShardSource impls.
//
// Submodules reach their siblings through the glob re-exports below
// (`use super::*`); `Transform`, `ScxLazyTransformedDataset`, and
// `LazyShardSource` are re-exported at `crate::lazy_transform` so existing
// call sites and the `add_class` registration in lib.rs are unchanged.

pub(crate) mod dataset;
pub(crate) mod shard_source;
pub(crate) mod transform;
pub(crate) mod transforms;

pub(crate) use dataset::*;
pub(crate) use shard_source::*;
pub(crate) use transform::*;
pub(crate) use transforms::*;
