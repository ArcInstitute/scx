//! Query execution — the public entry points and the X-shard decode.
//!
//! Split into two halves (CLI2): `plan_and_mask` does the cheap work — build
//! plan / catalog-level pruning, read obs metadata, evaluate obs predicates,
//! apply deletion vectors, build per-shard keep masks, resolve the gene
//! projection — without decoding any X shard. `materialize` does the expensive
//! rest — parallel shard decode, assemble CSR, fused normalize+log1p, apply
//! limit, filter obs/var. [`execute`] runs both; [`count`] runs only the first.
//!
//! Which masking arm ran is [`super::mask`]'s business, not this module's:
//! everything here consumes a `MaskResult`.

use arrow::array::{Array, RecordBatch, UInt32Array};
use arrow::compute;

use scx_sparse::{IndexBuffer, MaterializePlan, ScxCsr, ValueBuffer};

use crate::error::{EngineError, Result};
use crate::fused_ops::apply_fused_ops;
use crate::pipeline::{QueryPipeline, QueryResult, TypedQueryResult};
use crate::predicate::evaluate;
use crate::projection::{decode_shard_projected_presorted, project_var};
use crate::reader::SectionReader;

use scx_format_io::assemble_filtered_metadata;
use scx_format_io::catalog::FullCatalogEntry;

use super::mask::{compute_mask, MaskResult, ShardInfo};
use super::native::{decode_shard_native_filtered, truncate_to_limit, NativeShardRows};
use super::plan::{build_plan, scan_shards, ExecutionPlan};
use super::retry::par_map_with_shard_retry;
use super::rows::filter_csr_rows_owned;

/// Result of the cheap planning + masking half of a query (CLI2): catalog
/// shard elimination, obs/var predicate evaluation, per-shard row-keep masks,
/// and gene projection — everything computable **without decoding any X
/// shard**. Consumed by [`materialize`] (full result) or summarised by
/// [`count`] (matched-row count only).
pub(crate) struct PlanAndMask {
    pub(crate) plan: ExecutionPlan,
    /// Full obs batch — populated ONLY on the legacy single-section path.
    /// On the row-sharded (atlas-scale) path obs is never materialised in
    /// full; `materialize` rebuilds just the matching rows from
    /// `obs_shard_ranges`. See [`plan_and_mask`].
    pub(crate) legacy_obs: Option<RecordBatch>,
    /// `(shard_idx, row_start, row_end)` for every obs metadata shard, in
    /// `shard_idx` order. Empty on the legacy single-section path. Lets
    /// `materialize` map matching global rows to the few obs shards that
    /// contain them and read only those.
    pub(crate) obs_shard_ranges: Vec<(u32, u64, u64)>,
    pub(crate) var_batch: RecordBatch,
    pub(crate) shard_infos: Vec<ShardInfo>,
    pub(crate) effective_gene_indices: Option<Vec<u32>>,
    pub(crate) n_output_cols: usize,
    /// Restores the caller's requested gene order on the materialized output
    /// (F4). `None` when the request was already ascending-unique.
    pub(crate) reorder: Option<ColumnReorder>,
    pub(crate) skipped_shards: usize,
    pub(crate) total_shards: usize,
    pub(crate) candidate_shard_rows: usize,
    /// Pre-limit Level-2 match count = Σ `local_keep_mask` trues. Equals
    /// `csr.n_rows()` after materialisation, but needs no decode.
    pub(crate) matched_rows: usize,
}

/// Maps the engine's sorted-projection column layout to the caller's requested
/// gene order (F4). The decode merge-scan requires an ascending gene set, but
/// the *output* columns should follow the order the caller passed to
/// `select_genes`. Present only when those differ.
pub(crate) struct ColumnReorder {
    /// `sorted_to_output[k]` = output column index for the gene at sorted
    /// position `k`. Used to relabel decoded CSR column indices in place.
    pub(crate) sorted_to_output: Vec<u32>,
    /// `output_to_sorted[j]` = sorted position of output column `j`. Used as
    /// arrow `take` indices to reorder the projected `var` rows.
    pub(crate) output_to_sorted: Vec<u32>,
}

/// Resolved gene-column projection: the ascending index set fed to the decode
/// path, the output column count, and the optional reorder back to the
/// caller's requested order.
struct GeneProjection {
    /// Sorted, unique gene indices for `decode_shard_projected_presorted` /
    /// `project_var`
    /// (unchanged fast path). `None` = no projection (all genes).
    decode_indices: Option<Vec<u32>>,
    n_output_cols: usize,
    /// `None` when the request was already ascending-unique (output == decode
    /// order) — the common case, byte-identical to the pre-F4 behavior.
    reorder: Option<ColumnReorder>,
}

/// Resolve the effective gene-column projection (explicit gene indices
/// intersected with any var predicates) and the resulting output column count.
/// Operates on already-read var metadata — no obs/X shard decode. Shared by the
/// normal `plan_and_mask` path and its empty-candidate short-circuit.
fn resolve_gene_projection(
    plan: &ExecutionPlan,
    var_batch: &RecordBatch,
    n_vars: usize,
) -> Result<GeneProjection> {
    let mut effective_gene_indices = plan.gene_indices.clone();

    if !plan.var_predicates.is_empty() {
        let var_mask = {
            let mut mask = vec![true; var_batch.num_rows()];
            for pred in &plan.var_predicates {
                let mask_array = evaluate(pred, var_batch)?;
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

    // F4: preserve the caller's requested gene order. The decode merge-scan
    // needs an ascending unique set, but output columns should follow the
    // requested order. Build both and a reorder that maps sorted-column
    // positions to output positions; the reorder is `None` (and the path is
    // byte-identical to pre-F4) when the request is already ascending-unique.
    let Some(output_genes) = effective_gene_indices else {
        return Ok(GeneProjection {
            decode_indices: None,
            n_output_cols: n_vars,
            reorder: None,
        });
    };

    // Dedup preserving first-occurrence order (duplicates collapse, matching
    // the documented limitation).
    let mut seen = std::collections::HashSet::new();
    let output_genes: Vec<u32> = output_genes
        .into_iter()
        .filter(|g| seen.insert(*g))
        .collect();

    let mut sorted = output_genes.clone();
    sorted.sort_unstable(); // already unique (deduped above)
    let n_output_cols = output_genes.len();

    let reorder = if output_genes == sorted {
        None
    } else {
        let mut sorted_to_output = vec![0u32; sorted.len()];
        let mut output_to_sorted = vec![0u32; output_genes.len()];
        for (j, &g) in output_genes.iter().enumerate() {
            let k = sorted
                .binary_search(&g)
                .expect("output gene must be present in its own sorted set");
            sorted_to_output[k] = j as u32;
            output_to_sorted[j] = k as u32;
        }
        Some(ColumnReorder {
            sorted_to_output,
            output_to_sorted,
        })
    };

    Ok(GeneProjection {
        decode_indices: Some(sorted),
        n_output_cols,
        reorder,
    })
}

/// Cheap half of query execution: catalog pruning + obs/var predicate
/// evaluation + per-shard row-keep masks + gene projection. No X-shard decode.
pub(crate) fn plan_and_mask(pipeline: &QueryPipeline) -> Result<PlanAndMask> {
    let reader = pipeline.reader();
    let modality_id = pipeline.modality_id();
    // Per-modality X width (header().n_vars is the file-wide max, not the
    // modality's — docs/format.md § 13.4).
    let n_vars = pipeline.n_vars();

    // Sorted CSR-shard view, resolved from the pipeline's cached positions and
    // reused across this function *and* handed to `build_plan`, so the
    // `shard_idx` positions in its candidates index this very list. Scoped to
    // the pipeline's modality (== shards_sorted() for the single-modality
    // default).
    let sorted_shards = scan_shards(reader.catalog(), pipeline.csr_shard_positions());
    let plan = build_plan(pipeline, &sorted_shards)?;
    debug_assert!(
        modality_id == 0
            || reader
                .catalog()
                .modality_csr_ranges_tile_obs(modality_id, reader.header().n_obs),
        "modality {modality_id} CSR shards must tile [0, n_obs)"
    );
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

    // No candidate shards survived catalog-level pruning — e.g. an indexed
    // equality whose value is absent from every shard (typo / stale id), or a
    // numeric equality outside every shard's [min, max]. The result is empty,
    // so short-circuit: skip ALL obs/X shard reads. Peak RSS stays at the
    // (MB-scale) predicate-index read regardless of file size, and we report
    // every shard skipped. Var is still read (cheap) for the output schema.
    if plan.candidate_shards.is_empty() {
        let var_batch = reader.read_var_for(modality_id)?;
        let GeneProjection {
            decode_indices: effective_gene_indices,
            n_output_cols,
            reorder,
        } = resolve_gene_projection(&plan, &var_batch, n_vars)?;
        // The empty result still needs the obs schema. On the row-sharded path
        // `materialize_filtered_obs` derives it from a single shard (bounded);
        // on the legacy single-section path there are NO obs shards, so we must
        // hand `materialize` the full obs batch to take the (legacy) empty
        // branch. Legacy files are not atlas-scale, so this read is cheap.
        let legacy_obs = if reader.obs_metadata_shard_count() == 0 {
            Some(reader.read_obs()?)
        } else {
            None
        };
        return Ok(PlanAndMask {
            plan,
            legacy_obs,
            obs_shard_ranges: Vec::new(),
            var_batch,
            shard_infos: Vec::new(),
            effective_gene_indices,
            n_output_cols,
            reorder,
            skipped_shards, // == total_shards (no shard survived)
            total_shards,
            candidate_shard_rows, // == 0
            matched_rows: 0,
        });
    }

    // Steps 2-5: produce the per-shard keep masks. `compute_mask` prefers the
    // row-set fast path — resolving indexed obs predicates directly from the
    // predicate index without decoding any obs shard — and falls back to the
    // legacy full-decode path otherwise (non-indexed / residual-only predicates,
    // legacy single-section obs, a stale index, or the pushdown kill-switch).
    let n_obs = reader.header().n_obs as usize;
    let MaskResult {
        legacy_obs,
        obs_shard_ranges,
        shard_infos,
        matched_rows,
    } = compute_mask(pipeline, &plan, &sorted_shards, n_obs)?;

    // Step 6: Handle var predicates and gene projection (assembled via trait impl)
    let var_batch = reader.read_var_for(modality_id)?;
    let GeneProjection {
        decode_indices: effective_gene_indices,
        n_output_cols,
        reorder,
    } = resolve_gene_projection(&plan, &var_batch, n_vars)?;

    Ok(PlanAndMask {
        plan,
        legacy_obs,
        obs_shard_ranges,
        var_batch,
        shard_infos,
        effective_gene_indices,
        n_output_cols,
        reorder,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
    })
}

/// Execute a query, materialising the full [`QueryResult`].
///
/// Borrows the pipeline — execution is read-only, so it is repeatable and a
/// failure leaves the pipeline intact. See
/// [`QueryPipeline::collect_ref`](crate::pipeline::QueryPipeline::collect_ref).
pub fn execute(pipeline: &QueryPipeline) -> Result<QueryResult> {
    let pm = plan_and_mask(pipeline)?;
    materialize(pipeline, pm)
}

/// Execute a query, decoding `X` **at `mplan`'s dtype** rather than `f32`.
///
/// The default [`execute`] decodes every shard to scipy types, so a caller who
/// names a wider dtype afterwards receives values that already rounded through
/// `f32`. This path decodes each shard to its native stream instead and narrows
/// once, into the requested buffer — which is what makes an integer count above
/// 2²⁴ readable exactly.
///
/// Refuses a pipeline carrying `with_normalize` / `with_log1p`: those transform
/// counts into floats, so there is no stored value left to reproduce exactly,
/// and the honest route is the `f32` one with its own guard. Callers should ask
/// [`QueryPipeline::typed_collect_supported`] first rather than handle the
/// error.
pub fn execute_typed(
    pipeline: &QueryPipeline,
    mplan: &MaterializePlan,
) -> Result<TypedQueryResult> {
    let pm = plan_and_mask(pipeline)?;
    materialize_typed(pipeline, pm, mplan)
}

/// The typed twin of [`materialize`]. Shares the prefix choice, the `max_value`
/// fold and the whole metadata half with it; differs only in how `X` is decoded
/// and assembled.
pub(crate) fn materialize_typed(
    pipeline: &QueryPipeline,
    pm: PlanAndMask,
    mplan: &MaterializePlan,
) -> Result<TypedQueryResult> {
    let PlanAndMask {
        plan,
        legacy_obs,
        obs_shard_ranges,
        var_batch,
        shard_infos,
        effective_gene_indices,
        n_output_cols,
        reorder,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
    } = pm;

    if plan.normalize.is_some() || plan.log1p {
        // Not `Generic`: that variant is classified retryable, and this is a
        // usage error the caller must act on, not a transient one.
        return Err(EngineError::UnsupportedRewrite {
            op: "a dtype-selected collect".to_string(),
            feature: "with_normalize() / with_log1p()".to_string(),
            remedy: "those transforms replace the stored counts with floating-point values, \
                     so no requested dtype can reproduce the stored data exactly. Collect \
                     without a dtype (the values are transformed anyway), or drop the \
                     transform."
                .to_string(),
        });
    }

    let reader = pipeline.reader();
    let sorted_shards = scan_shards(reader.catalog(), pipeline.csr_shard_positions());

    let decode_count = decode_prefix_len(&shard_infos, plan.limit);
    let max_value = max_value_over_prefix(&sorted_shards, &shard_infos[..decode_count]);

    // O(1) pre-decode guard, as the whole-matrix typed reader does: fail loud
    // before the decode and the big allocation when the target dtype cannot hold
    // `max_value`. Routed through `ScxError` rather than `EngineError::Generic`
    // so a refused cast is not classified as a retryable shard fault and
    // re-decoded before failing.
    scx_format_io::guard_decode_loss_dtype(max_value, mplan.data_dtype, mplan.allow_lossy)
        .map_err(scx_format_io::ScxError::from)?;

    // Decode natively, keeping only matching rows and projected columns in one
    // pass, then relabel to the caller's requested gene order while the indices
    // are still plain `u32` — before they narrow into the target index buffer,
    // whose own range gate then covers them.
    let mut shard_results: Vec<NativeShardRows> =
        par_map_with_shard_retry(&shard_infos[..decode_count], |si| {
            let entry = sorted_shards[si.shard_idx];
            let mut rows = decode_shard_native_filtered(
                reader,
                entry,
                &si.local_keep_mask,
                effective_gene_indices.as_deref(),
            )?;
            if let Some(ref reorder) = reorder {
                for idx in rows.indices.iter_mut() {
                    *idx = reorder.sorted_to_output[*idx as usize];
                }
            }
            Ok(rows)
        })?;

    if let Some(limit) = plan.limit {
        truncate_to_limit(&mut shard_results, limit);
    }

    // Assemble at the target width. The exact totals are known here — the query
    // path materialises per-shard results before merging — so the buffers are
    // sized once and filled, with no growable typed buffer needed.
    let total_rows: usize = shard_results.iter().map(NativeShardRows::n_rows).sum();
    let total_nnz: usize = shard_results.iter().map(NativeShardRows::nnz).sum();

    let mut indptr = vec![0i64; total_rows + 1];
    let mut indices = IndexBuffer::zeroed(mplan.index_dtype, total_nnz);
    let mut values = ValueBuffer::zeroed(mplan.data_dtype, total_nnz);

    let mut cum_rows = 0usize;
    let mut cum_nnz = 0usize;
    for rows in &shard_results {
        let (n, nnz) = (rows.n_rows(), rows.nnz());
        let range = cum_nnz..cum_nnz + nnz;
        scx_format_io::cast_native_indices_into(
            &mut indices,
            range.clone(),
            &rows.indices,
            mplan.allow_lossy,
        )?;
        scx_format_io::cast_native_values_into(
            &mut values,
            range,
            &rows.values,
            mplan.allow_lossy,
        )?;
        let nnz_off = cum_nnz as i64;
        for j in 0..n {
            indptr[cum_rows + 1 + j] = rows.indptr[j + 1] + nnz_off;
        }
        cum_rows += n;
        cum_nnz += nnz;
    }

    let csr =
        scx_sparse::TypedCsr::new_unchecked((total_rows, n_output_cols), indptr, indices, values);

    let (filtered_obs, filtered_var) = materialize_metadata(
        reader,
        &plan,
        legacy_obs.as_ref(),
        &obs_shard_ranges,
        var_batch,
        &shard_infos[..decode_count],
        effective_gene_indices.as_deref(),
        reorder.as_ref(),
    )?;

    // Same invariant the f32 path asserts: with no limit every candidate shard
    // is decoded, so the mask-only count and the assembled row count must agree.
    debug_assert!(
        plan.limit.is_some() || matched_rows == csr.n_rows(),
        "unlimited typed query: matched_rows ({matched_rows}) must equal assembled rows ({})",
        csr.n_rows(),
    );

    Ok(QueryResult {
        x: csr,
        obs: filtered_obs,
        var: filtered_var,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
        max_value,
    })
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

/// Whether any row matches, without decoding `X` and ignoring `limit`. On the
/// indexed-only row-set path this needs no obs/X shard decode (the answer comes
/// from the row-set cardinality); with residual predicates it decodes only the
/// narrowed obs shards, like [`count`]. Computes the full match count rather
/// than stopping at the first match — cheap on the indexed path, and a true
/// early-exit is a deferred refinement.
pub fn exists(pipeline: &QueryPipeline) -> Result<bool> {
    Ok(plan_and_mask(pipeline)?.matched_rows > 0)
}

/// Rebuild the filtered obs metadata for a row-sharded file by reading only
/// the obs shards that contain matching rows (the whole point of the OOM
/// fix). `obs_shard_ranges` is `(shard_idx, row_start, row_end)` and
/// `matching_global_rows` is ascending. Peak memory is bounded by the
/// result size, not the file size.
pub(crate) fn materialize_filtered_obs(
    reader: &dyn crate::reader::SectionReader,
    obs_shard_ranges: &[(u32, u64, u64)],
    matching_global_rows: &[u32],
) -> Result<RecordBatch> {
    // View sorted by row_start so we can walk it with a single cursor as we
    // scan the (ascending) matching rows. For a contiguous cover shard_idx
    // order already equals row_start order, but sort defensively.
    let mut by_start: Vec<(u32, u64, u64)> = obs_shard_ranges.to_vec();
    by_start.sort_by_key(|(_, rs, _)| *rs);

    // Group matching rows into (position-in-`by_start`, local row indices),
    // preserving ascending order so the concatenation matches the CSR order.
    let mut groups: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut cur = 0usize;
    for &g in matching_global_rows {
        let g = g as u64;
        while cur < by_start.len() && g >= by_start[cur].2 {
            cur += 1;
        }
        let Some(&(_, rs, _)) = by_start.get(cur) else {
            break; // out of range (shouldn't happen for a valid cover)
        };
        let local = (g - rs) as u32;
        match groups.last_mut() {
            Some((bi, locals)) if *bi == cur => locals.push(local),
            _ => groups.push((cur, vec![local])),
        }
    }

    // Read each needed obs shard and take its matching local rows. For an
    // empty result, feed a single 0-row shard so the assembler produces the
    // canonical schema (avoids the cloud `read_obs_schema` full-assembly).
    //
    // `read_obs_shard(0)` is safe here: `materialize_filtered_obs` is only
    // reached on the row-sharded path, where `obs_shard_ranges` is
    // non-empty — which requires `obs_metadata_shard_count() > 0`, so shard
    // 0 always exists. A future caller that reaches this with empty
    // `obs_shard_ranges` would get a "section not found" error instead.
    let filtered_batches: Vec<RecordBatch> = if groups.is_empty() {
        vec![reader.read_obs_shard(0)?.slice(0, 0)]
    } else {
        par_map_with_shard_retry(&groups, |(bi, locals)| {
            let (shard_idx, _, _) = by_start[*bi];
            let batch = reader.read_obs_shard(shard_idx)?;
            // `par_map_with_shard_retry` may invoke this closure twice (on a
            // transient-failure retry), so it borrows `locals` and clones per
            // call rather than consuming the owned Vec. `UInt32Array::from`
            // still takes the clone's allocation via `Buffer::from_vec`.
            let take_indices = UInt32Array::from(locals.clone());
            let columns: Vec<_> = batch
                .columns()
                .iter()
                .map(|col| compute::take(col.as_ref(), &take_indices, None))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(RecordBatch::try_new(batch.schema(), columns)?)
        })?
    };

    // `template_schema` is only consulted when `batches` is empty; we always
    // pass at least one batch, so the first batch's schema is a safe filler.
    let template_schema = filtered_batches[0].schema();
    Ok(assemble_filtered_metadata(
        &template_schema,
        filtered_batches,
    )?)
}

/// Expensive half of query execution: decode matching shards, assemble the CSR
/// matrix, apply fused ops + limit, and filter obs/var metadata to the result.
pub(crate) fn materialize(pipeline: &QueryPipeline, pm: PlanAndMask) -> Result<QueryResult> {
    let PlanAndMask {
        plan,
        legacy_obs,
        obs_shard_ranges,
        var_batch,
        shard_infos,
        effective_gene_indices,
        n_output_cols,
        reorder,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
    } = pm;

    let reader = pipeline.reader();
    let sorted_shards = scan_shards(reader.catalog(), pipeline.csr_shard_positions());

    // Step 7: Parallel shard decode with optional projection, over the prefix of
    // candidate shards `limit` can reach (see `decode_prefix_len`).
    let decode_count = decode_prefix_len(&shard_infos, plan.limit);

    // Folded before the decode, not after: it comes from catalog stats and
    // `decode_count` alone, so a caller whose guard trips pays no decode.
    let max_value = max_value_over_prefix(&sorted_shards, &shard_infos[..decode_count]);

    // Each shard produces (indptr, indices, data) filtered to matching rows
    let shard_results: Vec<(Vec<i64>, Vec<i32>, Vec<f32>)> =
        par_map_with_shard_retry(&shard_infos[..decode_count], |si| {
            let entry = sorted_shards[si.shard_idx];

            // Decode shard (with or without projection)
            let (indptr, indices, data) = if let Some(ref gi) = effective_gene_indices {
                decode_shard_projected_presorted(reader, entry, gi)?
            } else {
                reader.read_shard_from_entry(entry)?
            };

            // Filter to matching rows within the shard. Owned, so a shard the
            // mask keeps whole is handed back rather than copied — the case for
            // every shard of an unfiltered query, and for any shard a row-set
            // fully covers.
            filter_csr_rows_owned(indptr, indices, data, &si.local_keep_mask)
        })?;

    // Step 8: Assemble CSR from per-shard results.
    //
    // The first shard's buffers *become* the merge buffers and the rest are
    // appended into them, then dropped as the loop advances — so the assembled
    // matrix costs one copy of every shard but the first, where it used to cost
    // one copy of all of them into three `Vec::new()`s (doubling as they grew)
    // while the whole per-shard copy stayed resident through fused ops, the
    // limit slice and the entire obs/var metadata phase. A single-shard
    // unfiltered `collect()` is now copy-free end to end.
    let total_rows: usize = shard_results
        .iter()
        .map(|(indptr, _, _)| indptr.len().saturating_sub(1))
        .sum();
    let total_nnz: usize = shard_results
        .iter()
        .map(|(_, indices, _)| indices.len())
        .sum();

    let mut shards = shard_results.into_iter();
    // `[0]` for the no-shard case, which is what the old `is_empty` guard
    // repaired after the fact.
    let (mut merged_indptr, mut merged_indices, mut merged_data) = shards
        .next()
        .unwrap_or_else(|| (vec![0i64], Vec::new(), Vec::new()));
    merged_indptr.reserve((total_rows + 1).saturating_sub(merged_indptr.len()));
    merged_indices.reserve(total_nnz.saturating_sub(merged_indices.len()));
    merged_data.reserve(total_nnz.saturating_sub(merged_data.len()));
    // Each shard's indptr is rebased to 0 by the row filter, so the running
    // offset is the accumulator's own last entry — for the first shard that is
    // its nnz, exactly what the old loop added on its `i == 0` pass.
    let mut cumulative_nnz: i64 = merged_indptr.last().copied().unwrap_or(0);

    for (indptr, indices, data) in shards {
        // Skip leading 0 and offset by cumulative nnz
        for &v in &indptr[1..] {
            merged_indptr.push(v + cumulative_nnz);
        }
        cumulative_nnz += indptr.last().copied().unwrap_or(0);
        merged_indices.extend_from_slice(&indices);
        merged_data.extend_from_slice(&data);
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
    // from the keep masks. With no limit we decode every candidate shard, so it
    // must equal `csr.n_rows()` here — asserting that keeps the mask-only
    // `count()` path honest against the decode path without re-deriving it. With
    // a limit we decode only the prefix of shards needed to fill it (Step 7), so
    // the assembled CSR holds only those rows (>= limit, <= matched_rows) until
    // Step 10 truncates.
    debug_assert!(
        plan.limit.is_some() || matched_rows == csr.n_rows(),
        "unlimited query: matched_rows ({matched_rows}) must equal assembled rows ({})",
        csr.n_rows(),
    );

    // Step 10: Apply limit
    if let Some(limit) = plan.limit {
        if limit < csr.n_rows() {
            csr = csr.row_slice(0, limit)?;
        }
    }

    // Step 11: Filter obs metadata to matching rows
    // Build the list of global row indices that made it into the output.
    // `shard_infos` come from `candidate_shards`, which derive from
    // `catalog.shards_sorted()` (ascending `row_start`), so this list is
    // globally ascending and aligns row-for-row with the assembled CSR.
    //
    // Only the decoded prefix (`shard_infos[..decode_count]`) contributes rows
    // to the assembled CSR; for a `limit`ed query the trailing shards were
    // never decoded. The prefix already holds >= `limit` matches, so building
    // over it (then truncating below) yields the same first-`limit` rows while
    // avoiding an all-shards alloc on the low-selectivity full-scan path. With
    // no limit `decode_count == shard_infos.len()`, so this is unchanged.
    // Steps 11–11c: obs rows for the same shard prefix, var rows for the
    // projection, both restored to the caller's requested gene order.
    let (filtered_obs, filtered_var) = materialize_metadata(
        reader,
        &plan,
        legacy_obs.as_ref(),
        &obs_shard_ranges,
        var_batch,
        &shard_infos[..decode_count],
        effective_gene_indices.as_deref(),
        reorder.as_ref(),
    )?;

    // The column relabel that goes with that reorder is container-specific, so
    // it stays out here — `materialize_metadata` moves the var rows, the caller
    // moves the matching CSR column indices.
    if let Some(ref reorder) = reorder {
        for idx in csr.indices.iter_mut() {
            *idx = reorder.sorted_to_output[*idx as usize] as i32;
        }
    }

    // Step 12: Return QueryResult
    Ok(QueryResult {
        x: csr,
        obs: filtered_obs,
        var: filtered_var,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
        max_value,
    })
}

/// How many of the (ascending, mask-carrying) candidate shards a `limit` can
/// reach.
///
/// `shard_infos` are ascending by `row_start` and each carries its keep mask, so
/// the per-shard matched-row count is known without decoding X. Decoding only
/// the first K shards whose cumulative matches reach the limit bounds peak
/// decode + assembled memory to those K, even when a low-selectivity predicate
/// matches rows in every candidate shard — otherwise the entire matched X would
/// be materialised before the limit truncated it.
///
/// Shared by both collect paths: a divergence here would silently misalign the
/// obs half (built over the same prefix) against the assembled matrix.
fn decode_prefix_len(shard_infos: &[ShardInfo], limit: Option<usize>) -> usize {
    match limit {
        Some(limit) => {
            let mut cum = 0usize;
            let mut k = 0usize;
            for si in shard_infos {
                if cum >= limit {
                    break;
                }
                cum += si.local_keep_mask.iter().filter(|&&keep| keep).count();
                k += 1;
            }
            k
        }
        None => shard_infos.len(),
    }
}

/// Max `ShardStats::value_max` over the shards that will actually be decoded.
///
/// Integer encodings record the true max; float encodings record 0. A reader
/// compares this against the largest integer its target dtype represents exactly
/// to fail loud on a silent decode loss before returning X. Deliberately
/// **narrower than a catalog-wide fold**: a large count in a shard the predicate
/// skipped, or one past the `limit` cutoff, is never decoded and so must not
/// refuse the read.
fn max_value_over_prefix(sorted_shards: &[&FullCatalogEntry], prefix: &[ShardInfo]) -> u32 {
    prefix
        .iter()
        .filter_map(|si| sorted_shards[si.shard_idx].stats.as_ref())
        .map(|s| s.value_max)
        .max()
        .unwrap_or(0)
}

/// Build the obs and var halves of a query result: the obs rows for the decoded
/// shard prefix (limit-truncated, categories pruned per the documented
/// `collect()` contract), and the var rows for the gene projection, restored to
/// the caller's requested order.
///
/// Container-independent, so both collect paths share it. What it deliberately
/// does **not** do is relabel the matrix's column indices — that half of the
/// reorder depends on how the values are stored, and lives with each caller.
#[allow(clippy::too_many_arguments)]
fn materialize_metadata(
    reader: &dyn SectionReader,
    plan: &ExecutionPlan,
    legacy_obs: Option<&RecordBatch>,
    obs_shard_ranges: &[(u32, u64, u64)],
    var_batch: RecordBatch,
    prefix: &[ShardInfo],
    effective_gene_indices: Option<&[u32]>,
    reorder: Option<&ColumnReorder>,
) -> Result<(RecordBatch, RecordBatch)> {
    let mut matching_global_rows: Vec<u32> = Vec::new();
    for si in prefix {
        for (local_row, &keep) in si.local_keep_mask.iter().enumerate() {
            if keep {
                matching_global_rows.push((si.row_start as usize + local_row) as u32);
            }
        }
    }

    // Apply limit to matching rows BEFORE choosing obs shards, so a small
    // `limit` reads only the obs shard(s) covering the first N matches.
    if let Some(limit) = plan.limit {
        matching_global_rows.truncate(limit);
    }
    debug_assert!(
        matching_global_rows.windows(2).all(|w| w[0] <= w[1]),
        "matching_global_rows must be ascending to align with the assembled CSR"
    );

    let filtered_obs = if let Some(obs_batch) = legacy_obs {
        // Legacy single-section obs: take directly from the full batch.
        let take_indices = UInt32Array::from(matching_global_rows);
        let columns: Vec<_> = obs_batch
            .columns()
            .iter()
            .map(|col| compute::take(col.as_ref(), &take_indices, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        RecordBatch::try_new(obs_batch.schema(), columns)?
    } else {
        // Row-sharded obs: read only the shards that contain matching rows.
        materialize_filtered_obs(reader, obs_shard_ranges, &matching_global_rows)?
    };
    // The documented `collect()` categorical contract, decided here so it does
    // not depend on the obs layout: a result whose rows the caller narrowed —
    // an obs predicate or a limit — carries only the categories its surviving
    // rows use, as an AnnData subset does (`remove_unused_categories`). Neither
    // `take` above nor the sharded assembler prunes (both keep the parent
    // vocabulary, deterministically), and an unfiltered `collect()` keeps the
    // declared list like `read_obs()`. Deletion vectors alone do not make a
    // subset: `to_anndata()` skips deleted rows and keeps the declared list too.
    let filtered_obs = if !plan.obs_predicates.is_empty() || plan.limit.is_some() {
        scx_format_io::prune_unused_dictionary_values(&filtered_obs)?
    } else {
        filtered_obs
    };

    // Step 11b: Filter var metadata to projected genes (in ascending order;
    // Step 11c restores the requested order).
    let filtered_var = if let Some(gi) = effective_gene_indices {
        project_var(&var_batch, gi)?
    } else {
        var_batch
    };

    // Step 11c: Restore the caller's requested gene order (F4). Decode +
    // `project_var` produce columns in ascending gene-index order; relabel the
    // CSR column indices and reorder the var rows to the requested order. The
    // reorder is `None` (skipped) when the request was already ascending-unique,
    // so the common path is unchanged. Per-row CSR indices may become
    // non-ascending here — correct for terminal output and matches anndata's
    // `adata[:, names]`; scipy tolerates unsorted indices.
    let filtered_var = if let Some(reorder) = reorder {
        let take_idx = UInt32Array::from(reorder.output_to_sorted.clone());
        let columns: Vec<_> = filtered_var
            .columns()
            .iter()
            .map(|col| compute::take(col.as_ref(), &take_idx, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        RecordBatch::try_new(filtered_var.schema(), columns)?
    } else {
        filtered_var
    };

    Ok((filtered_obs, filtered_var))
}
