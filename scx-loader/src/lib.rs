pub mod batch;
pub mod decode_stage;
pub mod error;
pub mod index_plan;
pub mod io_stage;
pub mod normalize;
pub mod pipeline;
pub mod projection;
#[cfg(feature = "python")]
pub mod python;
pub mod shuffle;

pub use batch::{Batch, ObsColumn};
pub use decode_stage::{decode_stage, extract_obs_columns};
pub use error::{LoaderError, Result};
pub use index_plan::{IndexPlanBatch, IndexPlanLoader};
pub use io_stage::{io_stage, ShardData, ShardGroup};
pub use normalize::{fused_normalize_log1p_dense, log1p_dense_row, normalize_dense_row};
pub use pipeline::{compute_memory_budget, LoaderConfig, MemoryBudget, TrainingPipeline};
pub use projection::{scatter_row_full, HvgProjection};
pub use shuffle::{RowShuffler, ShardShuffler};

#[cfg(feature = "python")]
pub use python::{IndexPlanDataset, TrainingDataset};
