pub mod error;
pub mod pipeline;
pub mod predicate;
pub mod pushdown;

pub use error::{EngineError, Result};
pub use pipeline::{NormalizeConfig, QueryPipeline, QueryResult};
pub use predicate::{evaluate, parse_predicate, Predicate, ScalarValue};
pub use pushdown::{prune_shards_by_catalog, ShardCandidate};
