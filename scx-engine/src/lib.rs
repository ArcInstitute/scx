pub mod collect;
pub mod error;
pub mod fused_ops;
pub mod index;
pub mod pipeline;
pub mod predicate;
pub mod projection;
pub mod pushdown;
pub mod reader;

pub use collect::filter_csr_rows;
pub use error::{EngineError, Result};
pub use fused_ops::{
    apply_fused_ops, fused_normalize_log1p, log1p_row, normalize_row, streaming_preprocess,
    streaming_save_layer, PreprocessConfig,
};
pub use index::{
    apply_obs_shard_column_stats, build_and_write_conversion_predicate_indexes,
    build_and_write_conversion_predicate_indexes_streaming, build_indexes,
    build_obs_predicate_index_bytes, build_var_predicate_index_bytes, derive_shard_column_stats,
    index_preset_columns, BuildOutcome, ConversionPredicateIndexOptions,
    ConversionPredicateIndexResult, IndexPreset, ObsPredicateIndexBuilder, PredicateIndex,
    PredicateIndexBuildOptions, SkipReason,
};
pub use pipeline::{CountResult, NormalizeConfig, QueryPipeline, QueryResult};
pub use predicate::{evaluate, parse_predicate, Predicate, ScalarValue};
pub use projection::{decode_shard_projected, project_csr, project_csr_row, project_var};
pub use pushdown::{
    prune_shards_by_catalog, prune_shards_by_catalog_with_dict, CategoryDictionaries,
    ShardCandidate,
};
pub use reader::{BoxedSectionReader, SectionReader};
