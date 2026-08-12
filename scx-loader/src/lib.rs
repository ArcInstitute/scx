pub mod batch;
pub mod budget;
pub mod decode_stage;
pub mod downsample;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod index_plan;
pub mod io_stage;
pub mod normalize;
pub mod pipeline;
pub mod plan_engine;
pub mod pool;
pub mod projection;
#[cfg(feature = "python")]
pub mod python;
pub(crate) mod runtime;
pub mod shuffle;
pub mod sparse_cellset;
pub mod sparse_cellset_collate;

pub use batch::{Batch, ObsColumn};
pub use budget::{BudgetBreakdown, PYTHON_OVERHEAD_BYTES};
pub use decode_stage::{build_category_dicts, decode_stage, extract_obs_columns, CategoryDict};
pub use downsample::{
    clip_negatives, downsample_row, file_identity, DownsampleConfig, DownsampleMethod,
};
pub use error::{LoaderError, Result};
pub use index_plan::{IndexPlanBatch, IndexPlanIter, IndexPlanLoader};
pub use io_stage::{io_stage, ShardData, ShardGroup};
pub use normalize::{
    apply_dense_transforms, fused_normalize_log1p_dense, fused_normalize_log1p_dense_with_depth,
    log1p_dense_row, normalize_dense_row, normalize_dense_row_with_depth,
};
pub use pipeline::{compute_memory_budget, LoaderConfig, MemoryBudget, TrainingPipeline};
pub use plan_engine::{PlanPrefetchIter, PrefetchEngine};
pub use pool::{cpu_pool, resolve_pool_threads, DEFAULT_DECODE_POOL_MAX_THREADS};
pub use projection::{scatter_row_full, HvgProjection};
pub use shuffle::{RowShuffler, ShardShuffler};
pub use sparse_cellset::{
    collate_gathered, CollateScalars, CollatedCellSetBatch, SparseCellSetBatch,
    SparseCellSetLoader, SparseCellSetPlan,
};
pub use sparse_cellset_collate::{collate_cell, CellIn, CellOut, CollateConfig, PreprocessMode};

#[cfg(feature = "python")]
pub use python::{
    collate_cellset_gathered, downsample_counts_csr, downsample_file_identity, IndexPlanDataset,
    MultimodalTrainingDataset, SparseCellSetBatchIter, SparseCellSetDataset, TrainingDataset,
};
