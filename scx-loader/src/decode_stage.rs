//! Decode Stage (Stage 2) — shuffle + decode + densify pipeline stage.
//!
//! Receives shard groups from the I/O stage, shuffles rows, projects HVG
//! genes, scatters sparse→dense into batch buffers, applies fused
//! normalize+log1p, extracts obs metadata columns, and sends completed
//! `Batch`es downstream via bounded channel.
//!
//! See [docs/architecture.md §Training Data Loader](../../docs/architecture.md)
//! and [docs/multithreading.md §Training data loader](../../docs/multithreading.md#training-data-loader-triple-buffered-pipeline).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, AsArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    StringArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::batch::{Batch, ObsColumn};
use crate::budget::profiling_enabled;
use crate::error::{LoaderError, Result};
use crate::io_stage::ShardGroup;
use crate::normalize::apply_dense_transforms;
use crate::pipeline::LoaderConfig;
use crate::projection::{pflog1ppf_row_full, scatter_row_full, HvgProjection};
use crate::shuffle::RowShuffler;

// ---------------------------------------------------------------------------
// D2: ShardGroupIndex — CSR row lookup index
// ---------------------------------------------------------------------------

/// Index for looking up a cell's CSR data from a pooled shard group.
///
/// Maps global cell indices to `(shard_index_in_group, local_row_index)` for
/// efficient row extraction after row-level shuffling.
struct ShardGroupIndex {
    /// Maps global_cell_index → (shard_index_in_group, local_row_index).
    cell_to_shard: HashMap<u64, (usize, usize)>,
}

impl ShardGroupIndex {
    /// Build the index from all non-deleted cells in the shard group.
    fn build(group: &ShardGroup) -> Self {
        let mut cell_to_shard = HashMap::new();
        for (shard_idx, shard) in group.shards.iter().enumerate() {
            for local_row in 0..shard.n_rows as usize {
                // Skip deleted rows
                if let Some(ref deleted) = shard.deleted_rows {
                    if deleted.contains(local_row as u32) {
                        continue;
                    }
                }
                let global_idx = shard.global_row_offset + local_row as u64;
                cell_to_shard.insert(global_idx, (shard_idx, local_row));
            }
        }
        ShardGroupIndex { cell_to_shard }
    }

    /// Look up a cell's CSR row data from the shard group.
    ///
    /// Returns `(csr_indices_slice, csr_data_slice)` for the given global
    /// cell index.
    fn get_row<'a>(
        &self,
        global_idx: u64,
        group: &'a ShardGroup,
    ) -> Result<(&'a [i32], &'a [f32])> {
        let &(shard_idx, local_row) = self.cell_to_shard.get(&global_idx).ok_or_else(|| {
            LoaderError::ShutdownError(format!(
                "global cell index {global_idx} not found in ShardGroupIndex"
            ))
        })?;

        let shard = &group.shards[shard_idx];
        let start = shard.indptr[local_row] as usize;
        let end = shard.indptr[local_row + 1] as usize;

        Ok((&shard.indices[start..end], &shard.data[start..end]))
    }

    /// Collect all non-deleted global cell indices from the shard group.
    ///
    /// Returns indices in sorted order to ensure deterministic pre-shuffle
    /// ordering (HashMap iteration order is non-deterministic).
    fn cell_indices(&self) -> Vec<u64> {
        let mut indices: Vec<u64> = self.cell_to_shard.keys().copied().collect();
        indices.sort_unstable();
        indices
    }
}

// ---------------------------------------------------------------------------
// D3: Parallel row scatter with rayon
//
// **Pool ownership.** All parallel-iterator dispatch in this module runs
// inside `pool.install(|| { ... })` against a per-`TrainingPipeline`
// `rayon::ThreadPool` passed in from `start_epoch`. Do *not* call
// `rayon::par_*` against the global registry from any code reachable on
// the worker hot path — under fork-mode DataLoader workers the global
// pool's worker threads do not survive `fork()` and the dispatch hangs
// forever. See `fill_batch_parallel` doc.
// ---------------------------------------------------------------------------

/// Fill a batch by scattering CSR rows into a dense matrix in parallel.
///
/// Each row in the batch is filled by an independent rayon thread via
/// `par_chunks_mut`, providing safe non-overlapping mutable slices without
/// `unsafe` code.
///
/// **Fork-safety contract**. `pool` MUST be a per-`TrainingPipeline`
/// `rayon::ThreadPool` constructed *after* any fork (i.e. inside the
/// worker process, in `start_epoch`). The function dispatches its
/// `par_chunks_mut().zip(par_iter()).for_each(...)` via `pool.install`
/// so rayon routes the work to *that* pool's worker queue rather than
/// the process-global registry. If this function is ever called against
/// rayon's global pool from a forked child whose parent had already
/// initialised that pool (which is the common case under PyTorch
/// DataLoader workers), the inherited pool's worker threads do not
/// exist post-fork and the dispatch hangs forever in
/// `LockLatch::wait_and_reset`.
#[allow(clippy::too_many_arguments)]
fn fill_batch_parallel(
    batch_cell_indices: &[u64],
    group_index: &ShardGroupIndex,
    group: &ShardGroup,
    projection: Option<&HvgProjection>,
    normalize: bool,
    log1p: bool,
    target_sum: f64,
    pflog1ppf: bool,
    pflog1ppf_c: f64,
    n_vars: usize,
    n_output_genes: usize,
    pool: &rayon::ThreadPool,
) -> Result<Vec<f32>> {
    let n_rows = batch_cell_indices.len();
    let mut x = vec![0.0f32; n_rows * n_output_genes];

    if n_output_genes == 0 || n_rows == 0 {
        return Ok(x);
    }

    // Split the output buffer into per-row chunks and process in parallel.
    // Pair each row chunk with its corresponding cell index. `pool.install`
    // routes the parallel work to the per-pipeline pool — *not* the global
    // registry, which would deadlock under fork (see doc comment).
    //
    // `try_for_each` short-circuits on the first `Err` and propagates it as
    // the closure's return value, so we avoid a per-row mutex check on the
    // hot path.
    pool.install(|| {
        x.par_chunks_mut(n_output_genes)
            .zip(batch_cell_indices.par_iter())
            .try_for_each(|(output_row, &global_cell_idx)| -> Result<()> {
                let (csr_indices, csr_data) = group_index.get_row(global_cell_idx, group)?;

                if pflog1ppf {
                    // PFlog1pPF needs the FULL pre-projection row to compute
                    // depth s_i and the centering denominator D = n_vars, so it
                    // dispatches here (at scatter time) rather than via the
                    // post-scatter `apply_dense_transforms`. It IS the
                    // normalization — normalize/log1p are not applied.
                    match projection {
                        Some(proj) => proj.scatter_pflog1ppf_row(
                            csr_indices,
                            csr_data,
                            pflog1ppf_c,
                            n_vars,
                            output_row,
                        ),
                        None => {
                            pflog1ppf_row_full(
                                csr_indices,
                                csr_data,
                                pflog1ppf_c,
                                n_vars,
                                output_row,
                            )?;
                        }
                    }
                    return Ok(());
                }

                // Normalization depth is the cell's FULL pre-projection total
                // count — computed from `csr_data` (all stored nonzeros of the
                // row) before any HVG projection. With a projection, the dense
                // `output_row` holds only the panel genes, so its own sum would
                // be a panel-local depth that silently diverges from scanpy's
                // normalize-then-subset and from the pflog1ppf path above. For
                // the no-projection case this equals `output_row.iter().sum()`.
                let depth: f64 = csr_data.iter().map(|&v| v as f64).sum();

                // Scatter CSR row into dense output row (with or without projection)
                match projection {
                    Some(proj) => proj.scatter_row(csr_indices, csr_data, output_row),
                    None => scatter_row_full(csr_indices, csr_data, output_row)?,
                }

                // Apply configured dense-row transforms
                apply_dense_transforms(output_row, normalize, log1p, target_sum, depth);
                Ok(())
            })
    })?;

    Ok(x)
}

// ---------------------------------------------------------------------------
// D4: Obs metadata column extraction
// ---------------------------------------------------------------------------

/// Label of the reserved category that null/missing cells map to.
const MISSING_CATEGORY_LABEL: &str = "NaN";

/// Stable, file-global category dictionary for one categorical obs column.
///
/// Built once over the full obs `RecordBatch` (see [`build_category_dicts`]) so
/// that every batch and every epoch encode the same category string to the
/// same integer code — and so the `TrainingDataset` and `IndexPlanDataset`
/// paths agree. This replaces the previous per-batch, first-seen-in-batch
/// encoding whose codes drifted with batch composition (silently wrong ML
/// labels). Codes index [`categories`](Self::categories) directly.
///
/// Code ordering is internally stable but *not* guaranteed to match
/// `pandas.Categorical.codes`: the Utf8 path assigns codes in first-seen order
/// over obs, while the Dictionary path uses the dictionary's declared order. So
/// loader codes are training-internal labels and should not be cross-referenced
/// against codes computed elsewhere.
#[derive(Debug, Clone)]
pub struct CategoryDict {
    /// Category string → stable code (used by the Utf8/LargeUtf8 path).
    map: HashMap<String, u32>,
    /// Code → category string; the slice index *is* the code. Held behind an
    /// `Arc` so each emitted batch shares one allocation (refcount bump) rather
    /// than re-cloning the full (possibly `O(n_obs)`) string list per batch.
    categories: Arc<[String]>,
    /// Code of the reserved trailing missing/`"NaN"` category, present iff the
    /// column contains any null cell.
    missing_code: Option<u32>,
}

/// Decode an Arrow dictionary value array (`Utf8` / `LargeUtf8`) to owned strings.
fn dict_value_strings(values: &dyn Array, col_name: &str) -> Result<Vec<String>> {
    // `iter()` yields `Option<&str>`; a null value slot maps to "" rather than
    // returning garbage / panicking via positional `value(i)`.
    if let Some(v) = values.as_any().downcast_ref::<StringArray>() {
        Ok(v.iter()
            .map(|opt| opt.unwrap_or_default().to_string())
            .collect())
    } else if let Some(v) = values.as_any().downcast_ref::<LargeStringArray>() {
        Ok(v.iter()
            .map(|opt| opt.unwrap_or_default().to_string())
            .collect())
    } else {
        Err(LoaderError::ConfigError {
            reason: format!("obs column '{col_name}': dictionary values are not Utf8/LargeUtf8"),
        })
    }
}

/// Append the reserved missing/`"NaN"` category when the column has nulls,
/// returning its code. If a literal `"NaN"` level already exists it is reused
/// (a column carrying both real nulls and a literal `"NaN"` string conflates
/// the two — an accepted minor limitation).
fn maybe_add_missing(
    categories: &mut Vec<String>,
    map: &mut HashMap<String, u32>,
    has_null: bool,
) -> Option<u32> {
    if !has_null {
        return None;
    }
    if let Some(&code) = map.get(MISSING_CATEGORY_LABEL) {
        return Some(code);
    }
    let code = categories.len() as u32;
    categories.push(MISSING_CATEGORY_LABEL.to_string());
    map.insert(MISSING_CATEGORY_LABEL.to_string(), code);
    Some(code)
}

/// Build the stable global category dictionary for one obs column, or `None`
/// when the column is not categorical (numeric / unsupported types are decoded
/// directly by `extract_single_column`).
fn build_single_category_dict(array: &dyn Array, col_name: &str) -> Result<Option<CategoryDict>> {
    match array.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 => {
            let mut categories: Vec<String> = Vec::new();
            let mut map: HashMap<String, u32> = HashMap::new();
            let mut has_null = false;
            macro_rules! scan {
                ($arr:expr) => {{
                    let arr = $arr;
                    for val_opt in arr.iter() {
                        match val_opt {
                            Some(val) => {
                                if !map.contains_key(val) {
                                    let code = categories.len() as u32;
                                    let owned = val.to_string();
                                    categories.push(owned.clone());
                                    map.insert(owned, code);
                                }
                            }
                            None => has_null = true,
                        }
                    }
                }};
            }
            if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
                scan!(arr);
            } else if let Some(arr) = array.as_any().downcast_ref::<LargeStringArray>() {
                scan!(arr);
            } else {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "obs column '{col_name}': expected StringArray/LargeStringArray"
                    ),
                });
            }
            let missing_code = maybe_add_missing(&mut categories, &mut map, has_null);
            Ok(Some(CategoryDict {
                map,
                categories: categories.into(),
                missing_code,
            }))
        }
        DataType::Dictionary(_, _) => {
            // Preserve the full declared value set (incl. unused levels) to
            // match pandas.Categorical semantics. Codes come straight from the
            // dictionary keys (already file-global), so `categories[key] == str`.
            let dict = array.as_any_dictionary();
            let mut categories = dict_value_strings(dict.values().as_ref(), col_name)?;
            let mut map: HashMap<String, u32> = HashMap::with_capacity(categories.len());
            for (i, c) in categories.iter().enumerate() {
                map.entry(c.clone()).or_insert(i as u32);
            }
            let has_null = dict.keys().null_count() > 0;
            let missing_code = maybe_add_missing(&mut categories, &mut map, has_null);
            Ok(Some(CategoryDict {
                map,
                categories: categories.into(),
                missing_code,
            }))
        }
        _ => Ok(None),
    }
}

/// Build stable global category dictionaries for the categorical columns among
/// `obs_columns`, computed ONCE over the full obs `RecordBatch`.
///
/// Numeric columns produce no entry (they are decoded directly). Returns an
/// error if a requested column is missing or a string/dictionary column has an
/// unsupported value layout. The result is reused for every batch and epoch so
/// categorical codes are stable and identical across the `TrainingDataset` and
/// `IndexPlanDataset` paths.
pub fn build_category_dicts(
    obs: &RecordBatch,
    obs_columns: &[String],
) -> Result<HashMap<String, CategoryDict>> {
    let mut dicts = HashMap::new();
    for col_name in obs_columns {
        let col_idx =
            obs.schema()
                .index_of(col_name)
                .map_err(|_| LoaderError::ObsColumnNotFound {
                    name: col_name.clone(),
                    available: obs
                        .schema()
                        .fields()
                        .iter()
                        .map(|f| f.name().clone())
                        .collect(),
                })?;
        if let Some(dict) = build_single_category_dict(obs.column(col_idx), col_name)? {
            dicts.insert(col_name.clone(), dict);
        }
    }
    Ok(dicts)
}

/// Extract the requested observation metadata columns for a batch.
///
/// For each requested column name, looks up the column in the `RecordBatch`
/// and extracts values at the positions given by `cell_indices` (global row
/// indices). Converts to the appropriate `ObsColumn` variant.
///
/// `cat_dicts` holds the precomputed stable global category dictionaries (from
/// [`build_category_dicts`]) keyed by column name; categorical columns require
/// an entry, numeric columns are unaffected.
///
/// Returns an error if a requested column is not found in the RecordBatch.
pub fn extract_obs_columns(
    obs: &RecordBatch,
    cell_indices: &[u64],
    obs_columns: &[String],
    cat_dicts: &HashMap<String, CategoryDict>,
) -> Result<HashMap<String, ObsColumn>> {
    let mut result = HashMap::with_capacity(obs_columns.len());

    for col_name in obs_columns {
        let col_idx =
            obs.schema()
                .index_of(col_name)
                .map_err(|_| LoaderError::ObsColumnNotFound {
                    name: col_name.clone(),
                    available: obs
                        .schema()
                        .fields()
                        .iter()
                        .map(|f| f.name().clone())
                        .collect(),
                })?;

        let array = obs.column(col_idx);
        let obs_col =
            extract_single_column(array, cell_indices, col_name, cat_dicts.get(col_name))?;
        result.insert(col_name.clone(), obs_col);
    }

    Ok(result)
}

/// Extract a single Arrow column into an `ObsColumn` for the given cell indices.
///
/// `cat_dict` is the precomputed stable global category dictionary for this
/// column (required for categorical Utf8/LargeUtf8/Dictionary columns; ignored
/// for numeric columns).
fn extract_single_column(
    array: &dyn Array,
    cell_indices: &[u64],
    col_name: &str,
    cat_dict: Option<&CategoryDict>,
) -> Result<ObsColumn> {
    let dt = array.data_type();

    match dt {
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                LoaderError::ConfigError {
                    reason: format!("obs column '{col_name}': expected Int64Array"),
                }
            })?;
            let values: Vec<i64> = cell_indices
                .iter()
                .map(|&idx| arr.value(idx as usize))
                .collect();
            Ok(ObsColumn::Int64(values))
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                LoaderError::ConfigError {
                    reason: format!("obs column '{col_name}': expected Int32Array"),
                }
            })?;
            let values: Vec<i64> = cell_indices
                .iter()
                .map(|&idx| arr.value(idx as usize) as i64)
                .collect();
            Ok(ObsColumn::Int64(values))
        }
        DataType::UInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| LoaderError::ConfigError {
                    reason: format!("obs column '{col_name}': expected UInt32Array"),
                })?;
            let values: Vec<i64> = cell_indices
                .iter()
                .map(|&idx| arr.value(idx as usize) as i64)
                .collect();
            Ok(ObsColumn::Int64(values))
        }
        DataType::UInt64 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| LoaderError::ConfigError {
                    reason: format!("obs column '{col_name}': expected UInt64Array"),
                })?;
            let values: Vec<i64> = cell_indices
                .iter()
                .map(|&idx| arr.value(idx as usize) as i64)
                .collect();
            Ok(ObsColumn::Int64(values))
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| LoaderError::ConfigError {
                    reason: format!("obs column '{col_name}': expected Float32Array"),
                })?;
            let values: Vec<f64> = cell_indices
                .iter()
                .map(|&idx| arr.value(idx as usize) as f64)
                .collect();
            Ok(ObsColumn::Float64(values))
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| LoaderError::ConfigError {
                    reason: format!("obs column '{col_name}': expected Float64Array"),
                })?;
            let values: Vec<f64> = cell_indices
                .iter()
                .map(|&idx| arr.value(idx as usize))
                .collect();
            Ok(ObsColumn::Float64(values))
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            // String → categorical via the precomputed stable global dictionary
            // (codes are file-global, not batch-local). The opportunistic
            // downcast in `scx_format_io::arrow_compat` may surface obs as
            // either `Utf8` (StringArray, i32 offsets) or `LargeUtf8`
            // (LargeStringArray, i64 offsets) on >2 GB single-column metadata.
            // Both array types share the same `.value(idx) -> &str` API.
            let dict = cat_dict.ok_or_else(|| LoaderError::ConfigError {
                reason: format!(
                    "obs column '{col_name}': category dictionary not precomputed (internal error)"
                ),
            })?;
            macro_rules! decode_strings {
                ($arr:expr) => {{
                    let arr = $arr;
                    cell_indices
                        .iter()
                        .map(|&idx| -> Result<u32> {
                            let i = idx as usize;
                            if arr.is_null(i) {
                                dict.missing_code.ok_or_else(|| LoaderError::ConfigError {
                                    reason: format!(
                                        "obs column '{col_name}': null cell but no missing \
                                         category reserved (internal error)"
                                    ),
                                })
                            } else {
                                dict.map.get(arr.value(i)).copied().ok_or_else(|| {
                                    LoaderError::ConfigError {
                                        reason: format!(
                                            "obs column '{col_name}': value not in precomputed \
                                             category dictionary (internal error)"
                                        ),
                                    }
                                })
                            }
                        })
                        .collect::<Result<Vec<u32>>>()
                }};
            }
            let codes = if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
                decode_strings!(arr)?
            } else if let Some(arr) = array.as_any().downcast_ref::<LargeStringArray>() {
                decode_strings!(arr)?
            } else {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "obs column '{col_name}': expected StringArray/LargeStringArray"
                    ),
                });
            };
            Ok(ObsColumn::Categorical(codes, dict.categories.clone()))
        }
        DataType::Dictionary(key_type, _value_type) => {
            // Arrow dictionary encoding → Categorical. Keys are already
            // file-global codes into the dictionary's value set; the categories
            // come from the precomputed global dict (full declared level set).
            let dict = cat_dict.ok_or_else(|| LoaderError::ConfigError {
                reason: format!(
                    "obs column '{col_name}': category dictionary not precomputed (internal error)"
                ),
            })?;
            macro_rules! decode_dict {
                ($key_ty:ty) => {{
                    let dict_arr = array.as_dictionary::<$key_ty>();
                    let keys = dict_arr.keys();
                    let codes: Result<Vec<u32>> = cell_indices
                        .iter()
                        .map(|&idx| -> Result<u32> {
                            let i = idx as usize;
                            if keys.is_null(i) {
                                dict.missing_code.ok_or_else(|| LoaderError::ConfigError {
                                    reason: format!(
                                        "obs column '{col_name}': null cell but no missing \
                                         category reserved (internal error)"
                                    ),
                                })
                            } else {
                                Ok(keys.value(i) as u32)
                            }
                        })
                        .collect();
                    codes.map(|codes| ObsColumn::Categorical(codes, dict.categories.clone()))
                }};
            }
            match key_type.as_ref() {
                DataType::Int8 => decode_dict!(arrow::datatypes::Int8Type),
                DataType::Int16 => decode_dict!(arrow::datatypes::Int16Type),
                DataType::Int32 => decode_dict!(arrow::datatypes::Int32Type),
                DataType::UInt8 => decode_dict!(arrow::datatypes::UInt8Type),
                DataType::UInt16 => decode_dict!(arrow::datatypes::UInt16Type),
                DataType::UInt32 => decode_dict!(arrow::datatypes::UInt32Type),
                _ => Err(LoaderError::ConfigError {
                    reason: format!(
                        "obs column '{col_name}': unsupported dictionary key type {:?}",
                        key_type
                    ),
                }),
            }
        }
        _ => Err(LoaderError::ConfigError {
            reason: format!("obs column '{col_name}': unsupported data type {:?}", dt),
        }),
    }
}

// ---------------------------------------------------------------------------
// D1: Decode stage coordinator
// ---------------------------------------------------------------------------

/// Run the decode stage (Stage 2).
///
/// Receives shard groups from the I/O stage via `rx`, processes each group by:
/// 1. Building a `ShardGroupIndex` for efficient row lookup
/// 2. Pooling all non-deleted cell indices
/// 3. Shuffling rows (Fisher-Yates via seeded RNG)
/// 4. Drawing mini-batches of `batch_size`
/// 5. Parallel sparse→dense scatter with HVG projection
/// 6. Fused normalize+log1p if configured
/// 7. Obs metadata column extraction
/// 8. Sending completed `Batch` via bounded channel
///
/// The last batch in a shard group may be shorter than `batch_size`.
///
/// `pool` is the per-pipeline rayon pool that drives `fill_batch_parallel`'s
/// parallel scatter — see `fill_batch_parallel`'s doc comment for the
/// fork-safety rationale. The pool is owned by the `TrainingPipeline` and
/// lives across epochs; the decode thread only borrows it.
#[allow(clippy::too_many_arguments)]
pub fn decode_stage(
    mut rx: tokio::sync::mpsc::Receiver<ShardGroup>,
    tx: crossbeam_channel::Sender<Batch>,
    config: &LoaderConfig,
    n_vars: u64,
    projection: Option<HvgProjection>,
    obs_metadata: &RecordBatch,
    cat_dicts: &HashMap<String, CategoryDict>,
    epoch: u64,
    pool: &rayon::ThreadPool,
) -> Result<()> {
    let n_output_genes = match &projection {
        Some(proj) => proj.n_output_cols(),
        None => n_vars as usize,
    };

    // Create a seeded RNG for row-level shuffle (Level 2).
    // The shard-level shuffle (Level 1) already happened in shuffle_epoch().
    // Incorporate the epoch number so different epochs produce different row orderings.
    let mut rng = ChaCha8Rng::seed_from_u64(
        config
            .seed
            .wrapping_add(0xDEADBEEF)
            .wrapping_add(epoch.wrapping_mul(0x9E3779B97F4A7C15)),
    );

    let profile = profiling_enabled();
    let decode_start = Instant::now();
    let mut group_count = 0usize;
    let mut total_batches = 0usize;
    let mut total_scatter_us = 0u128;

    // Receive shard groups from I/O stage via blocking_recv on the tokio channel.
    // This function runs on a standard thread, not inside the tokio runtime.
    while let Some(group) = rx.blocking_recv() {
        let t_group = Instant::now();

        // Step 1: Build index for efficient row lookup
        let t0 = Instant::now();
        let group_index = ShardGroupIndex::build(&group);
        let index_time = t0.elapsed();

        // Step 2: Pool all non-deleted cell indices
        let mut cell_indices = group_index.cell_indices();
        let n_cells = cell_indices.len();

        // Step 3: Shuffle rows (Level 2 — Fisher-Yates)
        RowShuffler::shuffle_rows(&mut cell_indices, &mut rng);

        // Step 4: Draw mini-batches
        let mut group_batches = 0;
        for batch_cells in cell_indices.chunks(config.batch_size) {
            let n_rows = batch_cells.len();

            // Step 5+6: Parallel scatter + normalize
            let t_scatter = Instant::now();
            let x = fill_batch_parallel(
                batch_cells,
                &group_index,
                &group,
                projection.as_ref(),
                config.normalize,
                config.log1p,
                config.target_sum,
                config.pflog1ppf,
                config.pflog1ppf_c,
                n_vars as usize,
                n_output_genes,
                pool,
            )?;
            total_scatter_us += t_scatter.elapsed().as_micros();

            // Step 7: Extract obs metadata columns
            let obs = if config.obs_columns.is_empty() {
                HashMap::new()
            } else {
                extract_obs_columns(obs_metadata, batch_cells, &config.obs_columns, cat_dicts)?
            };

            // Step 8: Build and send batch
            let batch = Batch {
                x,
                x_shape: (n_rows, n_output_genes),
                obs,
                cell_indices: batch_cells.to_vec(),
            };

            tx.send(batch).map_err(|e| {
                LoaderError::ChannelError(format!("decode stage: failed to send batch: {e}"))
            })?;
            group_batches += 1;
        }
        total_batches += group_batches;
        if profile {
            eprintln!("[scx-loader profile] decode_stage group {group_count}: {:?} ({n_cells} cells, {group_batches} batches, index={:?})",
                t_group.elapsed(), index_time);
        }
        group_count += 1;
    }

    if profile {
        eprintln!("[scx-loader profile] decode_stage total: {:?} ({total_batches} batches, {group_count} groups, scatter_total={total_scatter_us}µs)",
            decode_start.elapsed());
    }

    // Channel closed — I/O stage is done. Drop tx to signal end-of-epoch.
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io_stage::ShardData;
    use arrow::array::{DictionaryArray, Int64Array, LargeStringArray, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashSet;
    use std::sync::Arc;

    /// Build a small per-test rayon `ThreadPool` for `decode_stage` /
    /// `fill_batch_parallel` calls. Mirrors the fork-safe per-pipeline pool
    /// construction in `TrainingPipeline::start_epoch` so tests exercise
    /// the same dispatch path as production.
    fn test_pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
    }

    /// Helper: create a simple ShardData with known CSR data.
    /// Each row has 2 nonzeros at known column positions.
    fn make_shard_data(
        n_rows: usize,
        n_vars: usize,
        global_offset: u64,
        deleted: Option<roaring::RoaringBitmap>,
    ) -> ShardData {
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in 0..n_rows {
            let col0 = ((row * 2) % n_vars) as i32;
            let col1 = ((row * 2 + 1) % n_vars) as i32;
            indices.push(col0);
            indices.push(col1);
            data.push((row + 1) as f32);
            data.push((row + 2) as f32);
            indptr.push(*indptr.last().unwrap() + 2);
        }
        ShardData {
            indptr,
            indices,
            data,
            global_row_offset: global_offset,
            n_rows: n_rows as u32,
            deleted_rows: deleted,
        }
    }

    /// Helper: create a simple obs RecordBatch with known columns.
    fn make_obs(n: usize) -> RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let counts: Vec<i64> = (0..n).map(|i| (i * 10) as i64).collect();
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("count", DataType::Int64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(counts)),
            ],
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // D2 Tests: ShardGroupIndex
    // -----------------------------------------------------------------------

    #[test]
    fn test_index_all_cells_present() {
        let shard0 = make_shard_data(5, 10, 0, None);
        let shard1 = make_shard_data(3, 10, 5, None);
        let group = ShardGroup {
            shards: vec![shard0, shard1],
        };

        let index = ShardGroupIndex::build(&group);

        // All 8 cells should be present
        assert_eq!(index.cell_to_shard.len(), 8);
        for i in 0..8u64 {
            assert!(
                index.cell_to_shard.contains_key(&i),
                "cell {i} missing from index"
            );
        }
    }

    #[test]
    fn test_index_deleted_cells_excluded() {
        let mut deleted = roaring::RoaringBitmap::new();
        deleted.insert(1);
        deleted.insert(3);

        let shard0 = make_shard_data(5, 10, 0, Some(deleted));
        let shard1 = make_shard_data(3, 10, 5, None);
        let group = ShardGroup {
            shards: vec![shard0, shard1],
        };

        let index = ShardGroupIndex::build(&group);

        // 5 - 2 + 3 = 6 cells should be present
        assert_eq!(index.cell_to_shard.len(), 6);
        assert!(!index.cell_to_shard.contains_key(&1));
        assert!(!index.cell_to_shard.contains_key(&3));
        assert!(index.cell_to_shard.contains_key(&0));
        assert!(index.cell_to_shard.contains_key(&2));
        assert!(index.cell_to_shard.contains_key(&4));
    }

    #[test]
    fn test_index_row_data_correct() {
        let shard = make_shard_data(3, 10, 100, None);
        let group = ShardGroup {
            shards: vec![shard],
        };

        let index = ShardGroupIndex::build(&group);

        // Row 0 (global=100): indices=[0, 1], data=[1.0, 2.0]
        let (idx, data) = index.get_row(100, &group).unwrap();
        assert_eq!(idx, &[0, 1]);
        assert_eq!(data, &[1.0, 2.0]);

        // Row 1 (global=101): indices=[2, 3], data=[2.0, 3.0]
        let (idx, data) = index.get_row(101, &group).unwrap();
        assert_eq!(idx, &[2, 3]);
        assert_eq!(data, &[2.0, 3.0]);

        // Row 2 (global=102): indices=[4, 5], data=[3.0, 4.0]
        let (idx, data) = index.get_row(102, &group).unwrap();
        assert_eq!(idx, &[4, 5]);
        assert_eq!(data, &[3.0, 4.0]);
    }

    // -----------------------------------------------------------------------
    // D3 Tests: Parallel row scatter
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_fill_matches_sequential() {
        let shard = make_shard_data(4, 10, 0, None);
        let group = ShardGroup {
            shards: vec![shard],
        };
        let index = ShardGroupIndex::build(&group);
        let cell_indices: Vec<u64> = vec![0, 1, 2, 3];
        let n_genes = 10;

        // Parallel fill
        let parallel_x = fill_batch_parallel(
            &cell_indices,
            &index,
            &group,
            None,
            false,
            false,
            0.0,
            false,
            1.0,
            n_genes,
            n_genes,
            &test_pool(),
        )
        .unwrap();

        // Sequential fill (manual)
        let mut sequential_x = vec![0.0f32; 4 * n_genes];
        for (row_idx, &global_idx) in cell_indices.iter().enumerate() {
            let (csr_idx, csr_data) = index.get_row(global_idx, &group).unwrap();
            let row_slice = &mut sequential_x[row_idx * n_genes..(row_idx + 1) * n_genes];
            scatter_row_full(csr_idx, csr_data, row_slice).unwrap();
        }

        assert_eq!(parallel_x, sequential_x);
    }

    #[test]
    fn test_parallel_fill_with_projection() {
        let shard = make_shard_data(4, 10, 0, None);
        let group = ShardGroup {
            shards: vec![shard],
        };
        let index = ShardGroupIndex::build(&group);
        let cell_indices: Vec<u64> = vec![0, 1, 2, 3];

        // Project to genes [0, 1, 4, 5]
        let proj = HvgProjection::new(vec![0, 1, 4, 5]);
        let n_output = proj.n_output_cols();

        let x = fill_batch_parallel(
            &cell_indices,
            &index,
            &group,
            Some(&proj),
            false,
            false,
            0.0,
            false,
            1.0,
            10,
            n_output,
            &test_pool(),
        )
        .unwrap();

        assert_eq!(x.len(), 4 * n_output);
        // Row 0: CSR indices=[0,1], values=[1.0, 2.0]
        // HVG [0,1,4,5] → output [0,1,2,3], gene 0→pos 0, gene 1→pos 1
        assert_eq!(x[0], 1.0); // gene 0
        assert_eq!(x[1], 2.0); // gene 1
        assert_eq!(x[2], 0.0); // gene 4 (not in row)
        assert_eq!(x[3], 0.0); // gene 5 (not in row)
    }

    #[test]
    fn test_parallel_fill_zero_nnz_rows_stay_zero() {
        // Create a shard where row 1 has 0 nnz
        let shard = ShardData {
            indptr: vec![0, 2, 2, 4], // row 1 has 0 nnz
            indices: vec![0, 1, 3, 4],
            data: vec![1.0, 2.0, 3.0, 4.0],
            global_row_offset: 0,
            n_rows: 3,
            deleted_rows: None,
        };
        let group = ShardGroup {
            shards: vec![shard],
        };
        let index = ShardGroupIndex::build(&group);
        let cell_indices: Vec<u64> = vec![0, 1, 2];
        let n_genes = 10;

        let x = fill_batch_parallel(
            &cell_indices,
            &index,
            &group,
            None,
            false,
            false,
            0.0,
            false,
            1.0,
            n_genes,
            n_genes,
            &test_pool(),
        )
        .unwrap();

        // Row 1 (offset 10..20) should be all zeros
        let row1 = &x[n_genes..2 * n_genes];
        assert!(
            row1.iter().all(|&v| v == 0.0),
            "zero-nnz row should be all zeros"
        );

        // Row 0 has values at [0, 1]
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
    }

    #[test]
    fn test_parallel_fill_with_normalize_log1p() {
        let shard = make_shard_data(2, 10, 0, None);
        let group = ShardGroup {
            shards: vec![shard],
        };
        let index = ShardGroupIndex::build(&group);
        let cell_indices: Vec<u64> = vec![0, 1];
        let n_genes = 10;
        let target_sum = 1e4;

        let x = fill_batch_parallel(
            &cell_indices,
            &index,
            &group,
            None,
            true,
            true,
            target_sum,
            false,
            1.0,
            n_genes,
            n_genes,
            &test_pool(),
        )
        .unwrap();

        // Row 0: values [1.0, 2.0] at cols [0, 1], sum=3.0
        // After normalize: [1/3*1e4, 2/3*1e4]
        // After log1p: [ln(1/3*1e4+1), ln(2/3*1e4+1)]
        let expected_0 = ((1.0_f64 / 3.0 * target_sum) + 1.0).ln() as f32;
        let expected_1 = ((2.0_f64 / 3.0 * target_sum) + 1.0).ln() as f32;

        assert!(
            (x[0] - expected_0).abs() < 1e-3,
            "x[0]={} expected={}",
            x[0],
            expected_0
        );
        assert!(
            (x[1] - expected_1).abs() < 1e-3,
            "x[1]={} expected={}",
            x[1],
            expected_1
        );
        // Other columns should be ~0 (log1p(0) = 0)
        assert!((x[2] - 0.0).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // D4 Tests: Obs metadata extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_int_column() {
        let obs = make_obs(10);
        let cell_indices: Vec<u64> = vec![2, 5, 8];

        let cols = ["count".to_string()];
        let dicts = build_category_dicts(&obs, &cols).unwrap();
        let result = extract_obs_columns(&obs, &cell_indices, &cols, &dicts).unwrap();

        let col = result.get("count").unwrap();
        if let ObsColumn::Int64(values) = col {
            assert_eq!(*values, vec![20, 50, 80]); // i * 10 for i in [2, 5, 8]
        } else {
            panic!("expected Int64 variant");
        }
    }

    #[test]
    fn test_extract_categorical_column() {
        let obs = make_obs(5);
        let cell_indices: Vec<u64> = vec![0, 1, 2];

        let cols = ["cell_id".to_string()];
        let dicts = build_category_dicts(&obs, &cols).unwrap();
        let result = extract_obs_columns(&obs, &cell_indices, &cols, &dicts).unwrap();

        let col = result.get("cell_id").unwrap();
        if let ObsColumn::Categorical(codes, categories) = col {
            // Codes index the stable GLOBAL category set (all 5 unique cell_ids
            // in the full obs table), not a batch-local subset.
            assert_eq!(codes.len(), 3);
            assert_eq!(categories.len(), 5);
            assert_eq!(&categories[codes[0] as usize], "cell_0");
            assert_eq!(&categories[codes[1] as usize], "cell_1");
            assert_eq!(&categories[codes[2] as usize], "cell_2");
        } else {
            panic!("expected Categorical variant");
        }
    }

    #[test]
    fn test_extract_large_utf8_column_decodes_as_categorical() {
        // Mirrors `test_extract_categorical_column` but the obs column is
        // `LargeUtf8` (i64 offsets) — the in-memory shape that
        // `scx_format_io::arrow_compat::downcast_large_types` surfaces when
        // a >2 GB obs column does not fit back in i32 offsets. Without
        // the LargeUtf8 arm in `extract_single_column`, this would error
        // out as "unsupported data type".
        let n = 4;
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let schema = Schema::new(vec![Field::new("cell_id", DataType::LargeUtf8, false)]);
        let obs = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(LargeStringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        let cell_indices: Vec<u64> = vec![0, 2, 3];
        let cols = ["cell_id".to_string()];
        let dicts = build_category_dicts(&obs, &cols).unwrap();
        let result = extract_obs_columns(&obs, &cell_indices, &cols, &dicts).unwrap();
        let col = result.get("cell_id").unwrap();
        if let ObsColumn::Categorical(codes, categories) = col {
            assert_eq!(codes.len(), 3);
            assert_eq!(&categories[codes[0] as usize], "cell_0");
            assert_eq!(&categories[codes[1] as usize], "cell_2");
            assert_eq!(&categories[codes[2] as usize], "cell_3");
        } else {
            panic!("expected Categorical variant from LargeUtf8 obs column");
        }
    }

    #[test]
    fn test_extract_dictionary_largeutf8_values_decodes_as_categorical() {
        // `Dictionary(Int8, LargeUtf8)` is the post-opportunistic-downcast
        // shape for a clustered/categorical obs column whose value
        // dictionary's offsets overflow i32. Should decode into the same
        // ObsColumn::Categorical as `Dictionary(Int8, Utf8)`.
        use arrow::datatypes::Int8Type;
        let values = LargeStringArray::from(vec!["A", "B", "C"]);
        let keys = arrow::array::Int8Array::from(vec![0_i8, 1, 2, 1, 0]);
        let dict = DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap();
        let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::LargeUtf8));
        let schema = Schema::new(vec![Field::new("cluster", dict_dt, false)]);
        let obs =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(dict) as Arc<dyn Array>]).unwrap();

        let cell_indices: Vec<u64> = vec![0, 1, 2, 3, 4];
        let cols = ["cluster".to_string()];
        let dicts = build_category_dicts(&obs, &cols).unwrap();
        let result = extract_obs_columns(&obs, &cell_indices, &cols, &dicts).unwrap();
        let col = result.get("cluster").unwrap();
        if let ObsColumn::Categorical(codes, categories) = col {
            assert_eq!(codes, &vec![0_u32, 1, 2, 1, 0]);
            assert_eq!(
                &categories[..],
                ["A".to_string(), "B".to_string(), "C".to_string()].as_slice()
            );
        } else {
            panic!("expected Categorical variant from Dictionary(_, LargeUtf8) obs column");
        }
    }

    #[test]
    fn test_extract_unknown_column_errors() {
        let obs = make_obs(5);
        let cell_indices: Vec<u64> = vec![0];

        let result = extract_obs_columns(
            &obs,
            &cell_indices,
            &["nonexistent".to_string()],
            &HashMap::new(),
        );

        match result {
            Err(LoaderError::ObsColumnNotFound { name, available }) => {
                assert_eq!(name, "nonexistent");
                assert!(
                    !available.is_empty(),
                    "available list should be non-empty: {available:?}"
                );
            }
            Err(other) => panic!("expected ObsColumnNotFound, got {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn test_extract_values_match_original() {
        let obs = make_obs(10);
        let cell_indices: Vec<u64> = vec![0, 3, 7];

        let cols = ["count".to_string(), "cell_id".to_string()];
        let dicts = build_category_dicts(&obs, &cols).unwrap();
        let result = extract_obs_columns(&obs, &cell_indices, &cols, &dicts).unwrap();

        // Verify count column
        if let ObsColumn::Int64(values) = result.get("count").unwrap() {
            assert_eq!(*values, vec![0, 30, 70]);
        } else {
            panic!("expected Int64");
        }

        // Verify cell_id column
        if let ObsColumn::Categorical(codes, cats) = result.get("cell_id").unwrap() {
            assert_eq!(&cats[codes[0] as usize], "cell_0");
            assert_eq!(&cats[codes[1] as usize], "cell_3");
            assert_eq!(&cats[codes[2] as usize], "cell_7");
        } else {
            panic!("expected Categorical");
        }
    }

    /// Regression for the per-batch categorical-code instability bug: the same
    /// category string MUST map to the same code regardless of which batch
    /// (which `cell_indices` subset) it is decoded in, and the `categories`
    /// list must be identical across batches. Before the fix, the Utf8 path
    /// built a fresh batch-local dictionary, so codes drifted with batch
    /// composition.
    #[test]
    fn test_categorical_codes_stable_across_batches() {
        // Repeated categories so the two disjoint batches see them in a
        // different first-seen order — exactly what made codes drift before.
        let labels = ["X", "Y", "X", "Z", "Y", "Z"];
        let schema = Schema::new(vec![Field::new("grp", DataType::Utf8, false)]);
        let obs = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(labels.to_vec()))],
        )
        .unwrap();

        let cols = ["grp".to_string()];
        let dicts = build_category_dicts(&obs, &cols).unwrap();

        // Batch A sees rows [0,1,2] → X,Y,X ; Batch B sees [3,4,5] → Z,Y,Z.
        let batch_a = extract_obs_columns(&obs, &[0u64, 1, 2], &cols, &dicts).unwrap();
        let batch_b = extract_obs_columns(&obs, &[3u64, 4, 5], &cols, &dicts).unwrap();

        let (codes_a, cats_a) = match batch_a.get("grp").unwrap() {
            ObsColumn::Categorical(c, cats) => (c, cats),
            _ => panic!("expected Categorical"),
        };
        let (codes_b, cats_b) = match batch_b.get("grp").unwrap() {
            ObsColumn::Categorical(c, cats) => (c, cats),
            _ => panic!("expected Categorical"),
        };

        // Identical global category list across batches.
        assert_eq!(cats_a, cats_b);
        assert_eq!(
            &cats_a[..],
            ["X".to_string(), "Y".to_string(), "Z".to_string()].as_slice()
        );

        // Build a string→code map from each batch and assert no string is
        // assigned two different codes across the two batches.
        let mut seen: HashMap<String, u32> = HashMap::new();
        for (codes, cats) in [(codes_a, cats_a), (codes_b, cats_b)] {
            for &code in codes {
                let label = cats[code as usize].clone();
                if let Some(&prev) = seen.get(&label) {
                    assert_eq!(prev, code, "category '{label}' got two different codes");
                } else {
                    seen.insert(label, code);
                }
            }
        }
        // Concretely: "Y" is row 1 in batch A and row 4 in batch B — same code.
        assert_eq!(codes_a[1], codes_b[1]);
        assert_eq!(cats_a[codes_a[1] as usize], "Y");
    }

    /// Null/missing cells in a categorical obs column map to the reserved
    /// trailing `"NaN"` category (and no phantom real category is created).
    #[test]
    fn test_categorical_null_maps_to_reserved_missing() {
        let arr = StringArray::from(vec![Some("a"), None, Some("b"), None, Some("a")]);
        let schema = Schema::new(vec![Field::new("grp", DataType::Utf8, true)]);
        let obs = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(arr)]).unwrap();

        let cols = ["grp".to_string()];
        let dicts = build_category_dicts(&obs, &cols).unwrap();
        let result = extract_obs_columns(&obs, &[0u64, 1, 2, 3, 4], &cols, &dicts).unwrap();

        if let ObsColumn::Categorical(codes, cats) = result.get("grp").unwrap() {
            // Real levels first (first-seen), reserved "NaN" appended last.
            assert_eq!(
                &cats[..],
                ["a".to_string(), "b".to_string(), "NaN".to_string()].as_slice()
            );
            let nan_code = 2u32;
            assert_eq!(codes[1], nan_code, "null cell should map to NaN code");
            assert_eq!(codes[3], nan_code, "null cell should map to NaN code");
            assert_eq!(&cats[codes[0] as usize], "a");
            assert_eq!(&cats[codes[2] as usize], "b");
            assert_eq!(codes[0], codes[4], "'a' should be stable");
        } else {
            panic!("expected Categorical");
        }
    }

    // -----------------------------------------------------------------------
    // D1 Tests: Full decode stage coordinator
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_all_cells_once() {
        let n_vars = 10;
        let shard0 = make_shard_data(5, n_vars, 0, None);
        let shard1 = make_shard_data(5, n_vars, 5, None);
        let group = ShardGroup {
            shards: vec![shard0, shard1],
        };

        let config = LoaderConfig {
            batch_size: 4,
            normalize: false,
            log1p: false,
            obs_columns: Vec::new(),
            ..LoaderConfig::default()
        };

        let (io_tx, io_rx) = tokio::sync::mpsc::channel(4);
        let (batch_tx, batch_rx) = crossbeam_channel::bounded(4);
        let obs = make_obs(10);

        // Send one group then close
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            io_tx.send(group).await.unwrap();
            // Drop sender to close channel
        });

        let handle = std::thread::spawn(move || {
            decode_stage(
                io_rx,
                batch_tx,
                &config,
                n_vars as u64,
                None,
                &obs,
                &HashMap::new(),
                0,
                &test_pool(),
            )
        });

        // Collect all batches
        let mut all_cells: Vec<u64> = Vec::new();
        while let Ok(batch) = batch_rx.recv() {
            assert_eq!(batch.x_shape.1, n_vars);
            assert_eq!(batch.x.len(), batch.x_shape.0 * batch.x_shape.1);
            all_cells.extend_from_slice(&batch.cell_indices);
        }

        handle.join().unwrap().unwrap();

        // All 10 cells should appear exactly once
        let cell_set: HashSet<u64> = all_cells.iter().copied().collect();
        assert_eq!(cell_set.len(), 10, "all cells should be unique");
        assert_eq!(all_cells.len(), 10, "all cells should appear exactly once");
        for i in 0..10u64 {
            assert!(cell_set.contains(&i), "cell {i} missing");
        }
    }

    #[test]
    fn test_decode_deleted_cells_excluded() {
        let n_vars = 10;
        let mut deleted = roaring::RoaringBitmap::new();
        deleted.insert(1);
        deleted.insert(3);

        let shard0 = make_shard_data(5, n_vars, 0, Some(deleted));
        let shard1 = make_shard_data(5, n_vars, 5, None);
        let group = ShardGroup {
            shards: vec![shard0, shard1],
        };

        let config = LoaderConfig {
            batch_size: 100,
            normalize: false,
            log1p: false,
            obs_columns: Vec::new(),
            ..LoaderConfig::default()
        };

        let (io_tx, io_rx) = tokio::sync::mpsc::channel(4);
        let (batch_tx, batch_rx) = crossbeam_channel::bounded(4);
        let obs = make_obs(10);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            io_tx.send(group).await.unwrap();
        });

        let handle = std::thread::spawn(move || {
            decode_stage(
                io_rx,
                batch_tx,
                &config,
                n_vars as u64,
                None,
                &obs,
                &HashMap::new(),
                0,
                &test_pool(),
            )
        });

        let mut all_cells: Vec<u64> = Vec::new();
        while let Ok(batch) = batch_rx.recv() {
            all_cells.extend_from_slice(&batch.cell_indices);
        }

        handle.join().unwrap().unwrap();

        // 10 - 2 deleted = 8 cells
        assert_eq!(all_cells.len(), 8);
        let cell_set: HashSet<u64> = all_cells.iter().copied().collect();
        assert!(!cell_set.contains(&1), "deleted cell 1 should be excluded");
        assert!(!cell_set.contains(&3), "deleted cell 3 should be excluded");
    }

    #[test]
    fn test_decode_batch_shapes() {
        let n_vars = 10;
        // 7 cells, batch_size=3 → batches of [3, 3, 1]
        let shard = make_shard_data(7, n_vars, 0, None);
        let group = ShardGroup {
            shards: vec![shard],
        };

        let config = LoaderConfig {
            batch_size: 3,
            normalize: false,
            log1p: false,
            obs_columns: Vec::new(),
            ..LoaderConfig::default()
        };

        let (io_tx, io_rx) = tokio::sync::mpsc::channel(4);
        let (batch_tx, batch_rx) = crossbeam_channel::bounded(4);
        let obs = make_obs(7);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            io_tx.send(group).await.unwrap();
        });

        let handle = std::thread::spawn(move || {
            decode_stage(
                io_rx,
                batch_tx,
                &config,
                n_vars as u64,
                None,
                &obs,
                &HashMap::new(),
                0,
                &test_pool(),
            )
        });

        let mut batch_sizes: Vec<usize> = Vec::new();
        while let Ok(batch) = batch_rx.recv() {
            assert_eq!(batch.x_shape.1, n_vars);
            assert_eq!(batch.x.len(), batch.x_shape.0 * batch.x_shape.1);
            batch_sizes.push(batch.x_shape.0);
        }

        handle.join().unwrap().unwrap();

        // Should have 3 batches: [3, 3, 1]
        assert_eq!(batch_sizes.len(), 3);
        batch_sizes.sort();
        assert_eq!(batch_sizes, vec![1, 3, 3]);
    }

    #[test]
    fn test_decode_hvg_projection() {
        let n_vars = 10;
        let shard = make_shard_data(3, n_vars, 0, None);
        let group = ShardGroup {
            shards: vec![shard],
        };

        let proj = HvgProjection::new(vec![0, 1, 4]);
        let n_output = proj.n_output_cols();

        let config = LoaderConfig {
            batch_size: 10,
            normalize: false,
            log1p: false,
            obs_columns: Vec::new(),
            ..LoaderConfig::default()
        };

        let (io_tx, io_rx) = tokio::sync::mpsc::channel(4);
        let (batch_tx, batch_rx) = crossbeam_channel::bounded(4);
        let obs = make_obs(3);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            io_tx.send(group).await.unwrap();
        });

        let handle = std::thread::spawn(move || {
            decode_stage(
                io_rx,
                batch_tx,
                &config,
                n_vars as u64,
                Some(proj),
                &obs,
                &HashMap::new(),
                0,
                &test_pool(),
            )
        });

        let batch = batch_rx.recv().unwrap();
        handle.join().unwrap().unwrap();

        assert_eq!(batch.x_shape.1, n_output);
        assert_eq!(batch.x_shape.1, 3);
        assert_eq!(batch.x.len(), batch.x_shape.0 * 3);
    }

    #[test]
    fn test_decode_normalize_log1p() {
        let n_vars = 10;
        let shard = make_shard_data(2, n_vars, 0, None);
        let group = ShardGroup {
            shards: vec![shard],
        };

        let config = LoaderConfig {
            batch_size: 10,
            normalize: true,
            log1p: true,
            target_sum: 1e4,
            obs_columns: Vec::new(),
            ..LoaderConfig::default()
        };

        let (io_tx, io_rx) = tokio::sync::mpsc::channel(4);
        let (batch_tx, batch_rx) = crossbeam_channel::bounded(4);
        let obs = make_obs(2);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            io_tx.send(group).await.unwrap();
        });

        let handle = std::thread::spawn(move || {
            decode_stage(
                io_rx,
                batch_tx,
                &config,
                n_vars as u64,
                None,
                &obs,
                &HashMap::new(),
                0,
                &test_pool(),
            )
        });

        let batch = batch_rx.recv().unwrap();
        handle.join().unwrap().unwrap();

        // All values should have been normalized and log1p'd.
        // Rows are shuffled, so use cell_indices to determine which global
        // cell each output row corresponds to.
        for (row_pos, &global_idx) in batch.cell_indices.iter().enumerate() {
            let original_row = global_idx as usize;
            let col0 = (original_row * 2) % n_vars;
            let col1 = (original_row * 2 + 1) % n_vars;

            for col in 0..n_vars {
                let v = batch.x[row_pos * n_vars + col];
                if col == col0 || col == col1 {
                    assert!(
                        v > 0.0,
                        "global_cell={global_idx} col={col}: non-zero value should be positive after log1p, got {v}"
                    );
                } else {
                    assert!(
                        v.abs() < 1e-7,
                        "global_cell={global_idx} col={col}: zero value should stay ~0 after fused, got {v}"
                    );
                }
            }
        }
    }

    // ---- PFlog1pPF loader mode -------------------------------------------

    /// Reconstruct the full dense row produced by `make_shard_data` for a given
    /// global cell index: two nonzeros at cols `(g*2)%n_vars` / `(g*2+1)%n_vars`
    /// with values `g+1` / `g+2`.
    fn make_shard_full_row(global: usize, n_vars: usize) -> Vec<f64> {
        let mut row = vec![0.0f64; n_vars];
        row[(global * 2) % n_vars] = (global + 1) as f64;
        row[(global * 2 + 1) % n_vars] = (global + 2) as f64;
        row
    }

    /// Spec §11.1 exact reference for one row.
    fn pflog1ppf_reference_row(full: &[f64], c: f64) -> Vec<f64> {
        let depth: f64 = full.iter().sum();
        let logs: Vec<f64> = full.iter().map(|&v| (v / depth + c).ln()).collect();
        let mean = logs.iter().sum::<f64>() / logs.len() as f64;
        logs.iter().map(|&l| l - mean).collect()
    }

    #[test]
    fn test_decode_pflog1ppf_no_projection() {
        let n_vars = 10;
        let shard = make_shard_data(4, n_vars, 0, None);
        let group = ShardGroup {
            shards: vec![shard],
        };
        let c = 1.0;
        let config = LoaderConfig {
            batch_size: 10,
            pflog1ppf: true,
            pflog1ppf_c: c,
            obs_columns: Vec::new(),
            ..LoaderConfig::default()
        };

        let (io_tx, io_rx) = tokio::sync::mpsc::channel(4);
        let (batch_tx, batch_rx) = crossbeam_channel::bounded(4);
        let obs = make_obs(4);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            io_tx.send(group).await.unwrap();
        });
        let handle = std::thread::spawn(move || {
            decode_stage(
                io_rx,
                batch_tx,
                &config,
                n_vars as u64,
                None,
                &obs,
                &HashMap::new(),
                0,
                &test_pool(),
            )
        });
        let batch = batch_rx.recv().unwrap();
        handle.join().unwrap().unwrap();

        for (row_pos, &global_idx) in batch.cell_indices.iter().enumerate() {
            let full = make_shard_full_row(global_idx as usize, n_vars);
            let expected = pflog1ppf_reference_row(&full, c);
            let row = &batch.x[row_pos * n_vars..(row_pos + 1) * n_vars];
            // Exact-transform match.
            for (col, (&got, &want)) in row.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got as f64 - want).abs() <= 1e-5,
                    "cell {global_idx} col {col}: got {got}, want {want}"
                );
            }
            // Centering property: row sums to ~0.
            let s: f64 = row.iter().map(|&v| v as f64).sum();
            assert!(s.abs() <= 1e-5, "cell {global_idx} row sum {s} not ~0");
            // Original-zero columns share the per-row baseline.
            let col0 = (global_idx as usize * 2) % n_vars;
            let col1 = (global_idx as usize * 2 + 1) % n_vars;
            let baseline = row[(0..n_vars).find(|c| *c != col0 && *c != col1).unwrap()];
            for (col, &val) in row.iter().enumerate() {
                if col != col0 && col != col1 {
                    assert!((val - baseline).abs() <= 1e-6, "zero col {col} != baseline");
                }
            }
        }
    }

    /// 🔴 End-to-end projection regression (review item A): with an HVG panel,
    /// the loader's PFlog1pPF output must equal the corresponding columns of
    /// FULL-transcriptome PFlog1pPF (depth & D over all genes), NOT panel-local.
    #[test]
    fn test_decode_pflog1ppf_projection_uses_full_transcriptome() {
        let n_vars = 10;
        let shard = make_shard_data(5, n_vars, 0, None);
        let group = ShardGroup {
            shards: vec![shard],
        };
        let c = 1.0;
        let panel: Vec<u32> = vec![0, 1, 2, 3]; // strict subset of 10 genes
        let proj = HvgProjection::new(panel.clone());
        let n_output = proj.n_output_cols();

        let config = LoaderConfig {
            batch_size: 10,
            pflog1ppf: true,
            pflog1ppf_c: c,
            obs_columns: Vec::new(),
            ..LoaderConfig::default()
        };

        let (io_tx, io_rx) = tokio::sync::mpsc::channel(4);
        let (batch_tx, batch_rx) = crossbeam_channel::bounded(4);
        let obs = make_obs(5);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            io_tx.send(group).await.unwrap();
        });
        let handle = std::thread::spawn(move || {
            decode_stage(
                io_rx,
                batch_tx,
                &config,
                n_vars as u64,
                Some(proj),
                &obs,
                &HashMap::new(),
                0,
                &test_pool(),
            )
        });
        let batch = batch_rx.recv().unwrap();
        handle.join().unwrap().unwrap();

        assert_eq!(batch.x_shape.1, n_output);
        for (row_pos, &global_idx) in batch.cell_indices.iter().enumerate() {
            let full = make_shard_full_row(global_idx as usize, n_vars);
            let full_ref = pflog1ppf_reference_row(&full, c); // depth & D over ALL 10 genes
            let row = &batch.x[row_pos * n_output..(row_pos + 1) * n_output];
            for (pos, &g) in panel.iter().enumerate() {
                assert!(
                    (row[pos] as f64 - full_ref[g as usize]).abs() <= 1e-5,
                    "cell {global_idx} panel pos {pos} (gene {g}): got {}, want {} (full-transcriptome)",
                    row[pos],
                    full_ref[g as usize]
                );
            }
        }
    }
}
