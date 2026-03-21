pub mod batch;
pub mod error;
pub mod normalize;
pub mod pipeline;
pub mod projection;
pub mod shuffle;

pub use batch::{Batch, ObsColumn};
pub use error::{LoaderError, Result};
pub use normalize::{fused_normalize_log1p_dense, log1p_dense_row, normalize_dense_row};
pub use pipeline::{compute_memory_budget, LoaderConfig, MemoryBudget};
pub use projection::{scatter_row_full, HvgProjection};
pub use shuffle::{RowShuffler, ShardShuffler};
