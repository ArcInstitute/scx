pub mod error;
pub mod index;
pub mod pipeline;
pub mod predicate;
pub mod projection;
pub mod pushdown;

pub use error::{EngineError, Result};
pub use index::{build_indexes, PredicateIndex};
pub use pipeline::{NormalizeConfig, QueryPipeline, QueryResult};
pub use predicate::{evaluate, parse_predicate, Predicate, ScalarValue};
pub use projection::{decode_shard_projected, project_csr, project_csr_row, project_var};
pub use pushdown::{prune_rows_by_index, prune_shards_by_catalog, ShardCandidate};
