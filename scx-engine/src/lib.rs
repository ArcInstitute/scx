pub mod error;
pub mod pipeline;
pub mod predicate;

pub use error::{EngineError, Result};
pub use pipeline::{NormalizeConfig, QueryPipeline, QueryResult};
pub use predicate::{evaluate, parse_predicate, Predicate, ScalarValue};
