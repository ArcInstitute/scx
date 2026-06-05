// Pipeline execution engine — collect.rs
//
// Ties together pushdown, decode, projection, filtering, and fused operations
// to execute a QueryPipeline. Called by `QueryPipeline::collect()`.

use std::io::Cursor;

use arrow::array::{Array, RecordBatch, UInt32Array};
use arrow::compute;
use rayon::prelude::*;
use scx_sparse::ScxCsr;

use crate::error::Result;
use crate::fused_ops::apply_fused_ops;
use crate::index::{IndexedColumn, PredicateIndex};
use crate::pipeline::{QueryPipeline, QueryResult};
use crate::predicate::{evaluate, Predicate};
use crate::projection::{decode_shard_projected, project_var};
use crate::pushdown::{prune_shards_by_catalog_with_dict, CategoryDictionaries, ShardCandidate};

use scx_format::DeletionVectors;

// ============================================================================
// F1. Execution plan
// ============================================================================

/// Internal execution plan built from a QueryPipeline.
struct ExecutionPlan {
    candidate_shards: Vec<ShardCandidate>,
    obs_predicates: Vec<Predicate>,
    /// Used for var-level filtering during execution (see `execute()`).
    var_predicates: Vec<Predicate>,
    gene_indices: Option<Vec<u32>>,
    normalize: Option<f64>,
    log1p: bool,
    limit: Option<usize>,
    deletion_vectors: Option<DeletionVectors>,
}

/// Build an execution plan from a QueryPipeline.
///
/// Runs catalog-level shard pruning and loads predicate indexes.
fn build_plan(pipeline: &QueryPipeline) -> Result<ExecutionPlan> {
    let catalog = pipeline.reader().catalog();

    // Load the obs predicate index (C5) — needed for category dictionaries.
    // The var predicate index is not consumed by execution (no var-level
    // pushdown is wired into the query path), so it is not read here.
    let obs_predicate_index = match pipeline.reader().read_obs_predicate_index_bytes()? {
        Some(bytes) => Some(PredicateIndex::read_from(&mut Cursor::new(bytes))?),
        None => None,
    };

    // Build category dictionaries from predicate index for catalog-level pruning.
    // Maps column_name_hash → sorted list of category values, so Utf8 predicate
    // values can be resolved to CategoryBitset bit positions.
    let category_dicts = build_category_dicts(&obs_predicate_index);
    let dicts_ref = if category_dicts.is_empty() {
        None
    } else {
        Some(&category_dicts)
    };

    // Catalog-level shard pruning (B1), now with category dictionary support
    let candidate_shards = prune_shards_by_catalog_with_dict(
        catalog,
        pipeline.obs_predicates(),
        pipeline.deletion_vectors().as_ref(),
        dicts_ref,
    );

    Ok(ExecutionPlan {
        candidate_shards,
        obs_predicates: pipeline.obs_predicates().to_vec(),
        var_predicates: pipeline.var_predicates().to_vec(),
        gene_indices: pipeline.gene_indices().cloned(),
        normalize: pipeline.normalize_target_sum(),
        log1p: pipeline.log1p(),
        limit: pipeline.limit_value(),
        deletion_vectors: pipeline.deletion_vectors().clone(),
    })
}

/// Build category dictionaries from a predicate index.
///
/// For each categorical column in the index, creates a mapping from
/// `column_name_hash` to the sorted list of category values. The position
/// in this list corresponds to the bit position in `CategoryBitset`.
fn build_category_dicts(index: &Option<PredicateIndex>) -> CategoryDictionaries {
    let mut dicts = CategoryDictionaries::new();
    let index = match index {
        Some(idx) => idx,
        None => return dicts,
    };
    for col in &index.columns {
        if let IndexedColumn::Categorical(cat) = col {
            let hash = scx_format::column_name_hash(&cat.column_name);
            let values: Vec<String> = cat.entries.iter().map(|e| e.value.clone()).collect();
            // CategoricalIndex entries are already sorted by BTreeMap in build_indexes
            dicts.insert(hash, values);
        }
    }
    dicts
}

// ============================================================================
// F3. Row filtering within shards
// ============================================================================

/// Extract only the rows where `keep_mask[row]` is true from decoded CSR data.
///
/// Returns new `(indptr, indices, data)` arrays for the filtered subset.
/// Operates on pre-decoded scipy-compatible types (i64/i32/f32).
pub fn filter_csr_rows(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    keep_mask: &[bool],
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let n_rows = indptr.len().saturating_sub(1);
    debug_assert_eq!(
        keep_mask.len(),
        n_rows,
        "filter_csr_rows: keep_mask length ({}) != CSR row count ({})",
        keep_mask.len(),
        n_rows,
    );
    let mask_len = keep_mask.len().min(n_rows);

    let mut new_indptr = Vec::with_capacity(mask_len + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    for row in 0..mask_len {
        if !keep_mask[row] {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        new_indices.extend_from_slice(&indices[start..end]);
        new_data.extend_from_slice(&data[start..end]);
        let prev = *new_indptr.last().unwrap();
        new_indptr.push(prev + (end - start) as i64);
    }

    (new_indptr, new_indices, new_data)
}

// ============================================================================
// F2. Pipeline execution
// ============================================================================

// Query execution is split into two halves (CLI2): `plan_and_mask` does the
// cheap work — (1) build plan / catalog-level pruning, (2) read obs metadata,
// (3) evaluate obs predicates → boolean mask, (4) apply deletion vectors,
// (5) map cells to shards / build keep masks, (6) var predicates + gene
// projection — without decoding any X shard. `materialize` does the expensive
// rest — (7) parallel shard decode, (8) assemble CSR, (9) fused normalize+log1p,
// (10) apply limit, (11) filter obs/var, (12) return QueryResult. `execute`
// runs both; `count` runs only the first and returns the match count.

/// Per-shard row range + local keep mask produced by `plan_and_mask`.
struct ShardInfo {
    shard_idx: usize,
    row_start: u64,
    #[allow(dead_code)] // retained for debugging and future use
    row_end: u64,
    local_keep_mask: Vec<bool>,
}

/// Result of the cheap planning + masking half of a query (CLI2): catalog
/// shard elimination, obs/var predicate evaluation, per-shard row-keep masks,
/// and gene projection — everything computable **without decoding any X
/// shard**. Consumed by [`materialize`] (full result) or summarised by
/// [`count`] (matched-row count only).
struct PlanAndMask {
    plan: ExecutionPlan,
    obs_batch: RecordBatch,
    var_batch: RecordBatch,
    shard_infos: Vec<ShardInfo>,
    effective_gene_indices: Option<Vec<u32>>,
    n_output_cols: usize,
    skipped_shards: usize,
    total_shards: usize,
    candidate_shard_rows: usize,
    /// Pre-limit Level-2 match count = Σ `local_keep_mask` trues. Equals
    /// `csr.n_rows()` after materialisation, but needs no decode.
    matched_rows: usize,
}

/// Cheap half of query execution: catalog pruning + obs/var predicate
/// evaluation + per-shard row-keep masks + gene projection. No X-shard decode.
fn plan_and_mask(pipeline: &QueryPipeline) -> Result<PlanAndMask> {
    let plan = build_plan(pipeline)?;
    let reader = pipeline.reader();
    let n_vars = reader.header().n_vars as usize;

    // Sorted CSR-shard view, computed once and reused across this function
    // (was recomputed — filter+sort+alloc — ≥4× per query). OE6.
    let sorted_shards = reader.catalog().shards_sorted();
    let total_shards = sorted_shards.len();
    let skipped_shards = total_shards - plan.candidate_shards.len();

    // Sum of rows in candidate shards (post Level 1
    // catalog-stats pruning, pre Level 2 PredicateIndex / row-evaluator
    // narrowing). Lets the CLI report "{matched_rows} of {candidate_rows}
    // candidate-shard rows matched" so users can see Level 2 is doing
    // work even when Level 1 skipped zero shards.
    let candidate_shard_rows: usize = plan
        .candidate_shards
        .iter()
        .filter_map(|sc| {
            sorted_shards[sc.shard_idx]
                .stats
                .as_ref()
                .map(|s| (s.row_end - s.row_start) as usize)
        })
        .sum();

    // Step 2: Read obs metadata. `read_obs` transparently handles
    // both legacy single-section and Phase 2 row-sharded layouts.
    let obs_batch = reader.read_obs()?;
    let n_obs = obs_batch.num_rows();

    // Step 3: Evaluate obs predicates to get boolean mask
    let mut obs_mask = vec![true; n_obs];
    for pred in &plan.obs_predicates {
        let mask_array = evaluate(pred, &obs_batch)?;
        for (i, m) in obs_mask.iter_mut().enumerate() {
            if *m {
                *m = mask_array.is_valid(i) && mask_array.value(i);
            }
        }
    }

    // Step 4: Apply deletion vectors — exclude deleted cells
    if let Some(ref dv) = plan.deletion_vectors {
        for (shard_idx, shard_entry) in sorted_shards.iter().enumerate() {
            if let Some(ref stats) = shard_entry.stats {
                if let Some(bitmap) = dv.shards.get(&(shard_idx as u32)) {
                    for local_row in bitmap.iter() {
                        let global_row = stats.row_start + local_row as u64;
                        if (global_row as usize) < n_obs {
                            obs_mask[global_row as usize] = false;
                        }
                    }
                }
            }
        }
    }

    // Step 5: Map matching cells to shards
    // Build per-shard keep masks (local row indices)
    let mut shard_infos: Vec<ShardInfo> = Vec::new();
    for sc in &plan.candidate_shards {
        let entry = sorted_shards[sc.shard_idx];
        if let Some(ref stats) = entry.stats {
            let n_shard_rows = (stats.row_end - stats.row_start) as usize;
            let mut local_mask = Vec::with_capacity(n_shard_rows);
            let mut any_match = false;
            for local_row in 0..n_shard_rows {
                let global_row = stats.row_start as usize + local_row;
                let keep = global_row < n_obs && obs_mask[global_row];
                if keep {
                    any_match = true;
                }
                local_mask.push(keep);
            }
            if any_match {
                shard_infos.push(ShardInfo {
                    shard_idx: sc.shard_idx,
                    row_start: stats.row_start,
                    row_end: stats.row_end,
                    local_keep_mask: local_mask,
                });
            }
        }
    }

    // Step 6: Handle var predicates and gene projection (assembled via trait impl)
    let var_batch = reader.read_var()?;
    let mut effective_gene_indices = plan.gene_indices.clone();

    if !plan.var_predicates.is_empty() {
        let var_mask = {
            let mut mask = vec![true; var_batch.num_rows()];
            for pred in &plan.var_predicates {
                let mask_array = evaluate(pred, &var_batch)?;
                for (i, m) in mask.iter_mut().enumerate() {
                    if *m {
                        *m = mask_array.is_valid(i) && mask_array.value(i);
                    }
                }
            }
            mask
        };

        // Convert var mask to gene indices
        let var_gene_indices: Vec<u32> = var_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i as u32) } else { None })
            .collect();

        // Intersect with explicit gene_indices if both present
        effective_gene_indices = Some(match effective_gene_indices {
            Some(explicit) => {
                let var_set: std::collections::HashSet<u32> =
                    var_gene_indices.iter().copied().collect();
                explicit
                    .into_iter()
                    .filter(|g| var_set.contains(g))
                    .collect()
            }
            None => var_gene_indices,
        });
    }

    // Sort gene indices for consistent projection
    if let Some(ref mut gi) = effective_gene_indices {
        gi.sort_unstable();
        gi.dedup();
    }

    let n_output_cols = effective_gene_indices
        .as_ref()
        .map_or(n_vars, |gi| gi.len());

    // Pre-limit Level-2 match count, computed from the keep masks alone — no
    // X decode required (CLI2/CLI6). Equals `csr.n_rows()` post-materialise.
    let matched_rows: usize = shard_infos
        .iter()
        .map(|si| si.local_keep_mask.iter().filter(|&&k| k).count())
        .sum();

    Ok(PlanAndMask {
        plan,
        obs_batch,
        var_batch,
        shard_infos,
        effective_gene_indices,
        n_output_cols,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
    })
}

/// Execute a query, materialising the full [`QueryResult`].
pub fn execute(pipeline: QueryPipeline) -> Result<QueryResult> {
    let pm = plan_and_mask(&pipeline)?;
    materialize(&pipeline, pm)
}

/// Count matching rows without decoding `X` (CLI2). Ignores `limit` entirely,
/// so `--count --limit` cannot misreport (CLI6).
pub fn count(pipeline: &QueryPipeline) -> Result<crate::pipeline::CountResult> {
    let pm = plan_and_mask(pipeline)?;
    Ok(crate::pipeline::CountResult {
        matched_rows: pm.matched_rows,
        skipped_shards: pm.skipped_shards,
        total_shards: pm.total_shards,
        candidate_shard_rows: pm.candidate_shard_rows,
    })
}

/// Expensive half of query execution: decode matching shards, assemble the CSR
/// matrix, apply fused ops + limit, and filter obs/var metadata to the result.
fn materialize(pipeline: &QueryPipeline, pm: PlanAndMask) -> Result<QueryResult> {
    let PlanAndMask {
        plan,
        obs_batch,
        var_batch,
        shard_infos,
        effective_gene_indices,
        n_output_cols,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
    } = pm;

    let reader = pipeline.reader();
    let sorted_shards = reader.catalog().shards_sorted();

    // Step 7: Parallel shard decode with optional projection
    // Each shard produces (indptr, indices, data) filtered to matching rows
    let shard_results: Vec<(Vec<i64>, Vec<i32>, Vec<f32>)> = shard_infos
        .par_iter()
        .map(|si| {
            let entry = sorted_shards[si.shard_idx];

            // Decode shard (with or without projection)
            let (indptr, indices, data) = if let Some(ref gi) = effective_gene_indices {
                decode_shard_projected(reader, entry, gi)?
            } else {
                reader.read_shard_from_entry(entry)?
            };

            // Filter to matching rows within the shard
            let (filtered_indptr, filtered_indices, filtered_data) =
                filter_csr_rows(&indptr, &indices, &data, &si.local_keep_mask);

            Ok((filtered_indptr, filtered_indices, filtered_data))
        })
        .collect::<Result<Vec<_>>>()?;

    // Step 8: Assemble CSR from per-shard results
    let mut merged_indptr: Vec<i64> = Vec::new();
    let mut merged_indices: Vec<i32> = Vec::new();
    let mut merged_data: Vec<f32> = Vec::new();
    let mut cumulative_nnz: i64 = 0;

    for (i, (indptr, indices, data)) in shard_results.iter().enumerate() {
        if i == 0 {
            merged_indptr.extend_from_slice(indptr);
        } else {
            // Skip leading 0 and offset by cumulative nnz
            for &v in &indptr[1..] {
                merged_indptr.push(v + cumulative_nnz);
            }
        }
        cumulative_nnz += indptr.last().copied().unwrap_or(0);
        merged_indices.extend_from_slice(indices);
        merged_data.extend_from_slice(data);
    }

    // Handle empty result case
    if merged_indptr.is_empty() {
        merged_indptr.push(0);
    }

    let n_rows = merged_indptr.len() - 1;
    let mut csr = ScxCsr::new_unchecked(
        (n_rows, n_output_cols),
        merged_indptr,
        merged_indices,
        merged_data,
    );

    // Step 9: Apply fused normalize+log1p
    apply_fused_ops(&mut csr, plan.normalize, plan.log1p);

    // `matched_rows` (pre-limit Level-2 count) was computed in `plan_and_mask`
    // from the keep masks and equals `csr.n_rows()` here. Asserting their
    // equality keeps the mask-only `count()` path honest against the decode
    // path without re-deriving it.
    debug_assert_eq!(matched_rows, csr.n_rows());

    // Step 10: Apply limit
    if let Some(limit) = plan.limit {
        if limit < csr.n_rows() {
            csr = csr.row_slice(0, limit)?;
        }
    }

    // Step 11: Filter obs metadata to matching rows
    // Build the list of global row indices that made it into the output
    let mut matching_global_rows: Vec<u32> = Vec::new();
    for si in &shard_infos {
        for (local_row, &keep) in si.local_keep_mask.iter().enumerate() {
            if keep {
                matching_global_rows.push((si.row_start as usize + local_row) as u32);
            }
        }
    }

    // Apply limit to matching rows
    if let Some(limit) = plan.limit {
        matching_global_rows.truncate(limit);
    }

    let filtered_obs = {
        let take_indices = UInt32Array::from(matching_global_rows);
        let columns: Vec<_> = obs_batch
            .columns()
            .iter()
            .map(|col| compute::take(col.as_ref(), &take_indices, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        RecordBatch::try_new(obs_batch.schema(), columns)?
    };

    // Step 11b: Filter var metadata to projected genes
    let filtered_var = if let Some(ref gi) = effective_gene_indices {
        project_var(&var_batch, gi)?
    } else {
        var_batch
    };

    // Step 12: Return QueryResult
    Ok(QueryResult {
        x: csr,
        obs: filtered_obs,
        var: filtered_var,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format::header::FileHeader;
    use scx_format::writer::ScxWriter;
    use std::sync::Arc;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader {
            magic: scx_format::MAGIC,
            format_version: scx_format::CURRENT_FORMAT_VERSION,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 0,
            root_catalog_length: 0,
            full_catalog_offset: 0,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            n_modalities: 0,
            modality_table_offset: 0,
            modality_table_length: 0,
            reserved: [0u8; 112],
        }
    }

    fn sample_obs(n: usize) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
        ]);
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let types: Vec<&str> = (0..n)
            .map(|i| match i % 3 {
                0 => "T cell",
                1 => "B cell",
                _ => "NK cell",
            })
            .collect();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(types)),
            ],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> RecordBatch {
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
        let path = dir.path().join("test.scx");
        let header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_obs {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            let (c0, c1) = if col0 < col1 {
                (col0, col1)
            } else {
                (col1, col0)
            };
            indices.push(c0 as u32);
            indices.push(c1 as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }

        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer.finish().unwrap();
        path
    }

    // -----------------------------------------------------------------------
    // F3 Tests: filter_csr_rows
    // -----------------------------------------------------------------------

    #[test]
    fn filter_alternating_rows() {
        // 4-row CSR: keep rows 0 and 2
        let indptr = vec![0i64, 2, 5, 7, 10];
        let indices = vec![0i32, 1, 0, 1, 2, 1, 3, 0, 2, 3];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let keep = vec![true, false, true, false];

        let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep);
        assert_eq!(new_ip, vec![0, 2, 4]);
        assert_eq!(new_idx, vec![0, 1, 1, 3]);
        assert_eq!(new_data, vec![1.0, 2.0, 6.0, 7.0]);
    }

    #[test]
    fn filter_keep_all() {
        let indptr = vec![0i64, 2, 5];
        let indices = vec![0i32, 1, 0, 1, 2];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let keep = vec![true, true];

        let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep);
        assert_eq!(new_ip, vec![0, 2, 5]);
        assert_eq!(new_idx, indices);
        assert_eq!(new_data, data);
    }

    #[test]
    fn filter_keep_none() {
        let indptr = vec![0i64, 2, 5];
        let indices = vec![0i32, 1, 0, 1, 2];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let keep = vec![false, false];

        let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep);
        assert_eq!(new_ip, vec![0]);
        assert!(new_idx.is_empty());
        assert!(new_data.is_empty());
    }

    // -----------------------------------------------------------------------
    // F2 Tests: Pipeline execution via QueryPipeline::collect()
    // -----------------------------------------------------------------------

    #[test]
    fn collect_no_predicates_returns_all() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
        assert_eq!(result.x.n_rows(), 12);
        assert_eq!(result.x.n_cols(), 5);
        assert_eq!(result.obs.num_rows(), 12);
        assert_eq!(result.var.num_rows(), 5);
        assert_eq!(result.total_shards, 1);
    }

    #[test]
    fn collect_filter_obs_returns_subset() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .collect()
            .unwrap();
        // cells 0, 3, 6, 9 are "T cell" (i % 3 == 0)
        assert_eq!(result.x.n_rows(), 4);
        assert_eq!(result.obs.num_rows(), 4);
        // Check that filtered obs matches
        let cell_types = result
            .obs
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..cell_types.len() {
            assert_eq!(cell_types.value(i), "T cell");
        }
    }

    #[test]
    fn collect_gene_projection() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 10);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(vec![0, 3, 7])
            .collect()
            .unwrap();
        assert_eq!(result.x.n_cols(), 3);
        assert_eq!(result.var.num_rows(), 3);
        // Check var metadata has correct gene IDs
        let gene_ids = result
            .var
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(gene_ids.value(0), "gene_0");
        assert_eq!(gene_ids.value(1), "gene_3");
        assert_eq!(gene_ids.value(2), "gene_7");
    }

    #[test]
    fn collect_with_normalize_log1p() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 6, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .with_normalize(1e4)
            .with_log1p()
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 6);
        // Values should be transformed (no longer raw integers)
        // Each row had 2 non-zero values, after normalize+log1p they should be ln(v/sum*1e4 + 1)
        for row in 0..result.x.n_rows() {
            let start = result.x.indptr[row] as usize;
            let end = result.x.indptr[row + 1] as usize;
            for i in start..end {
                assert!(
                    result.x.data[i] > 0.0,
                    "fused ops should produce positive values"
                );
                assert!(
                    result.x.data[i] < 20.0,
                    "ln(10001) ≈ 9.21, values should be reasonable"
                );
            }
        }
    }

    #[test]
    fn collect_empty_result() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        // Filter for a cell_type that doesn't exist — but the column exists
        // so the predicate is valid. All cells are T cell, B cell, or NK cell.
        // Use cell_id which is unique to force empty result.
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_id == 'nonexistent'")
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 0);
        assert_eq!(result.obs.num_rows(), 0);
    }

    #[test]
    fn collect_with_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .limit(3)
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 3);
        assert_eq!(result.obs.num_rows(), 3);
        // matched_rows must reflect the
        // pre-limit Level-2 match count (12 rows match the no-predicate
        // pipeline), not the post-limit returned count.
        assert_eq!(result.matched_rows, 12);
    }

    #[test]
    fn collect_limit_exceeds_total() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 6, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .limit(100)
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 6);
        assert_eq!(result.obs.num_rows(), 6);
        // No truncation occurred — matched_rows must equal returned rows.
        assert_eq!(result.matched_rows, result.x.n_rows());
    }

    #[test]
    fn count_matches_collect_matched_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        // No predicate: every cell matches.
        let c = QueryPipeline::open(&path).unwrap().count().unwrap();
        assert_eq!(c.matched_rows, 12);
        let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
        assert_eq!(c.matched_rows, result.matched_rows);
        assert_eq!(c.total_shards, result.total_shards);
        assert_eq!(c.skipped_shards, result.skipped_shards);
        assert_eq!(c.candidate_shard_rows, result.candidate_shard_rows);
    }

    #[test]
    fn count_with_predicate_matches_collect() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        // count() and collect().matched_rows must agree for the same predicate,
        // even though count() never decodes X.
        let pipeline = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap();
        let c = pipeline.count().unwrap();
        let result = pipeline.collect().unwrap();
        assert_eq!(c.matched_rows, result.matched_rows);
        assert_eq!(c.matched_rows, result.x.n_rows());
    }

    #[test]
    fn count_ignores_limit() {
        // CLI6: count() reports the true match count regardless of `limit`.
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let no_limit = QueryPipeline::open(&path).unwrap().count().unwrap();
        let with_limit = QueryPipeline::open(&path)
            .unwrap()
            .limit(3)
            .count()
            .unwrap();
        assert_eq!(no_limit.matched_rows, 12);
        assert_eq!(
            with_limit.matched_rows, no_limit.matched_rows,
            "limit must not change the counted match total"
        );
    }

    #[test]
    fn count_empty_result() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let c = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_id == 'nonexistent'")
            .unwrap()
            .count()
            .unwrap();
        assert_eq!(c.matched_rows, 0);
    }

    #[test]
    fn collect_multiple_filter_obs_and_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        // cell_type == 'T cell' AND cell_id == 'cell_0' → only cell_0
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .filter_obs("cell_id == 'cell_0'")
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 1);
        assert_eq!(result.obs.num_rows(), 1);
        let cell_ids = result
            .obs
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(cell_ids.value(0), "cell_0");
    }

    #[test]
    fn collect_obs_row_count_matches_x() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'B cell'")
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(result.obs.num_rows(), result.x.n_rows());
    }

    #[test]
    fn collect_var_row_count_matches_x_cols() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 10);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(vec![1, 5, 9])
            .collect()
            .unwrap();
        assert_eq!(result.var.num_rows(), result.x.n_cols());
    }

    #[test]
    fn collect_full_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 10);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .select_genes(vec![0, 2, 4, 6, 8])
            .with_normalize(1e4)
            .with_log1p()
            .collect()
            .unwrap();
        // T cell: indices 0, 3, 6, 9 → 4 cells
        assert_eq!(result.x.n_rows(), 4);
        assert_eq!(result.x.n_cols(), 5); // 5 projected genes
        assert_eq!(result.obs.num_rows(), 4);
        assert_eq!(result.var.num_rows(), 5);
    }
}
