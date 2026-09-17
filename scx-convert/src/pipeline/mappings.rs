//! `obsm` / `varm` / `obsp` / `varp`: the row-sharded metadata matrices.
//!
//! Both representations live here because they share the row-range sharding
//! decision and differ only below it: dense mappings are gathered and written
//! as 2-D slabs, sparse ones are partitioned from COO into per-shard CSR.

use scx_format_io::writer::ScxWriter;

use super::error::ConvertError;
use crate::h5ad::read::{
    list_dense_mapping_shapes, list_sparse_mapping_shapes, read_dense_mapping_shard,
    read_sparse_mapping_shard,
};
use crate::warnings::{ConvertWarning, WarningSink};
use arrow::record_batch::RecordBatch;

/// Dispatch tag for [`write_dense_mapping_section`] so the shard
/// writer can pick the right `ScxWriter` method without duplicating
/// the obsm/varm loops.
#[derive(Debug, Clone, Copy)]
pub(super) enum DenseMappingKind {
    Obsm,
    Varm,
}

/// Same for [`write_sparse_mapping_section`] over obsp/varp.
#[derive(Debug, Clone, Copy)]
pub(super) enum SparseMappingKind {
    Obsp,
    Varp,
}

/// Gather the obsm/varm rows for one output shard in permuted (sorted) order
/// while reading only contiguous source runs from disk — peak memory stays
/// ~one output shard, mirroring [`crate::permuted_reader::PermutedCsrReader::gather`]
/// for the dense case. `want[i]` is the source row index for output-local row
/// `i`. Used by the sort-on-convert disk-streaming path so it preserves the
/// same per-shard RSS bound as the non-sort path (the previous code
/// materialized the whole `n_obs × k` mapping before permuting).
///
/// No `max_slab_rows` cap is needed here (unlike `PermutedCsrReader::gather`):
/// every coalesced run lives inside a single output shard, so a run is bounded
/// by `shard_target_rows` — exactly the hyperslab size the non-sort path
/// already issues.
fn gather_dense_mapping_shard(
    file: &hdf5::File,
    group_path: &str,
    name: &str,
    want: &[u64],
) -> Result<RecordBatch, ConvertError> {
    // The sole caller passes a non-empty shard slice (the `info.n_rows == 0`
    // case is handled in a separate branch), but guard the `runs[0]`
    // precondition explicitly: an empty gather is an empty mapping batch with
    // the correct schema.
    if want.is_empty() {
        return read_dense_mapping_shard(file, group_path, name, 0, 0);
    }

    // (output-local index, source id) sorted by source id so consecutive
    // source rows coalesce into a single contiguous hyperslab read.
    let mut order: Vec<(usize, u64)> = want.iter().copied().enumerate().collect();
    order.sort_unstable_by_key(|&(_, src)| src);

    let mut runs: Vec<RecordBatch> = Vec::new();
    // `take_idx[out_local]` = row position of that output row within the
    // run-order concatenation below.
    let mut take_idx = vec![0u64; want.len()];
    let mut concat_pos = 0u64;
    let mut i = 0;
    while i < order.len() {
        let run_start = order[i].1 as usize;
        let mut j = i + 1;
        while j < order.len() && order[j].1 == order[j - 1].1 + 1 {
            j += 1;
        }
        // `read_dense_mapping_shard` takes a half-open [start, end) range.
        let run_end = order[j - 1].1 as usize + 1;
        runs.push(read_dense_mapping_shard(
            file, group_path, name, run_start, run_end,
        )?);
        for entry in &order[i..j] {
            take_idx[entry.0] = concat_pos;
            concat_pos += 1;
        }
        i = j;
    }

    // Runs are read in source-sorted order; concatenate then permute into
    // output (sorted-by-obs-key) order.
    let schema = runs[0].schema();
    let concatenated = arrow::compute::concat_batches(&schema, runs.iter())?;
    crate::permuted_reader::take_record_batch(&concatenated, &take_idx)
}

/// Emit one logical obsm/varm matrix as a sequence of row-shards. Used
/// by [`super::entry_streaming::h5ad_to_scx_streaming`] for both the override
/// path (in-memory
/// `RecordBatch` from pyscx) and the disk-streaming path (h5py
/// hyperslab reads per shard).
#[allow(clippy::too_many_arguments)]
pub(super) fn write_dense_mapping_section(
    file: &hdf5::File,
    writer: &mut ScxWriter,
    override_entries: Option<&Vec<(String, RecordBatch)>>,
    group_path: &str,
    shard_target_rows: u32,
    kind: DenseMappingKind,
    // Sort-on-convert (Phase 2): when `Some`, reorder this section's rows by
    // `row_perm[output_row] = source_row]` before sharding. Used for obsm
    // (obs-axis); always `None` for varm (var-axis is untouched by an obs sort).
    row_perm: Option<&[u64]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    // One dispatch for both shard writers, at `ScxError` — the error type the
    // shared emitters' callback takes. Every caller here is inside a
    // `Result<(), ConvertError>` fn, so `?` widens it via the existing
    // `ConvertError: From<ScxError>`; a second `ConvertError`-typed copy of
    // this match would be the same four arms for no decision.
    let emit_io = |w: &mut ScxWriter,
                   name: &str,
                   shard_idx: u32,
                   row_start: u64,
                   n_shard_rows: u64,
                   n_total: u64,
                   batch: &RecordBatch|
     -> scx_format_io::Result<()> {
        match kind {
            DenseMappingKind::Obsm => {
                w.write_obsm_shard(name, shard_idx, row_start, n_shard_rows, n_total, batch)
            }
            DenseMappingKind::Varm => {
                w.write_varm_shard(name, shard_idx, row_start, n_shard_rows, n_total, batch)
            }
        }
    };

    if let Some(entries) = override_entries {
        for (name, batch) in entries {
            // Sort-on-convert: permute rows before sharding (obsm only).
            let permuted;
            let batch: &RecordBatch = match row_perm {
                Some(p) => {
                    permuted = crate::permuted_reader::take_record_batch(batch, p)?;
                    &permuted
                }
                None => batch,
            };
            scx_format_io::for_each_dense_mapping_shard(batch, shard_target_rows, |m, shard| {
                emit_io(
                    writer,
                    name,
                    m.shard_idx,
                    m.row_start,
                    m.n_shard_rows,
                    m.n_rows_total,
                    shard,
                )
            })?;
        }
        return Ok(());
    }

    // Disk-streaming path. Missing group → nothing to do.
    let infos = list_dense_mapping_shapes(file, group_path)?;

    // Surface every member `list_dense_mapping_shapes` could not handle as
    // a `SkippedObsm` warning so the loss is visible from Python and counted
    // in provenance instead of being silently swallowed (B4) or aborting the
    // whole conversion on an unreadable dtype (B5 residual). Handled members
    // are exactly the readable 2D dense datasets returned above; anything
    // else is a DataFrame subgroup, a sparse-matrix subgroup, a non-2D
    // dataset, or an unsupported-dtype dataset. Mirrors the `DroppedObsp`
    // classification in `write_sparse_mapping_section`.
    if let Ok(group) = file.group(group_path) {
        use std::collections::HashSet;
        let handled: HashSet<&str> = infos.iter().map(|i| i.name.as_str()).collect();
        for name in group.member_names()? {
            if name.starts_with("__") || handled.contains(name.as_str()) {
                continue;
            }
            sink.emit(ConvertWarning::SkippedObsm {
                name: format!("{group_path}/{name}"),
                reason: "not a 2D dense numeric dataset (DataFrame-valued, \
                         sparse-matrix-valued, non-2D, or unsupported dtype) \
                         — dense obsm/varm only"
                    .to_string(),
            });
        }
    }

    for info in &infos {
        let n_total = info.n_rows as u64;
        // Zero-row dense mappings: emit a single empty shard so the key
        // survives round-trip (mirrors the override-path special case).
        if info.n_rows == 0 {
            let batch = read_dense_mapping_shard(file, group_path, &info.name, 0, 0)?;
            emit_io(writer, &info.name, 0, 0, 0, 0, &batch)?;
            continue;
        }
        // Sort-on-convert (obsm): gather each output shard in permuted order
        // directly, reading only the contiguous source runs that shard needs.
        // Peak memory stays ~one output shard (independent of n_obs), matching
        // the non-sort path's per-shard RSS bound rather than materializing the
        // whole n_obs × k mapping. Mirrors `PermutedCsrReader::gather` (X path).
        if let Some(perm) = row_perm {
            // The obs sort permutation is indexed per output shard below; a
            // malformed file whose mapping row count differs from n_obs would
            // otherwise slice `perm` out of bounds. Reject rather than panic.
            if info.n_rows != perm.len() {
                return Err(ConvertError::Other(format!(
                    "obsm/varm '{}/{}' has {} rows but the obs sort permutation has {} \
                     (mapping row count must equal n_obs)",
                    group_path,
                    info.name,
                    info.n_rows,
                    perm.len()
                )));
            }
            let step = shard_target_rows.max(1) as usize;
            let mut shard_idx = 0u32;
            let mut row_start = 0usize;
            while row_start < info.n_rows {
                let n = (info.n_rows - row_start).min(step);
                let shard = gather_dense_mapping_shard(
                    file,
                    group_path,
                    &info.name,
                    &perm[row_start..row_start + n],
                )?;
                emit_io(
                    writer,
                    &info.name,
                    shard_idx,
                    row_start as u64,
                    n as u64,
                    n_total,
                    &shard,
                )?;
                row_start += n;
                shard_idx += 1;
            }
            continue;
        }
        let step = shard_target_rows.max(1) as usize;
        let mut shard_idx = 0u32;
        let mut row_start = 0usize;
        while row_start < info.n_rows {
            let row_end = (row_start + step).min(info.n_rows);
            let batch = read_dense_mapping_shard(file, group_path, &info.name, row_start, row_end)?;
            let n_shard_rows = (row_end - row_start) as u64;
            emit_io(
                writer,
                &info.name,
                shard_idx,
                row_start as u64,
                n_shard_rows,
                n_total,
                &batch,
            )?;
            row_start = row_end;
            shard_idx += 1;
        }
    }
    Ok(())
}

/// Emit one logical obsp/varp matrix as a sequence of row-shards. The
/// override path takes a single materialised COO `RecordBatch` per key
/// and partitions its triples by `row` into shard ranges; the
/// disk-streaming path reads h5py CSR slices per row range.
pub(super) fn write_sparse_mapping_section(
    file: &hdf5::File,
    writer: &mut ScxWriter,
    override_entries: Option<&Vec<(String, RecordBatch)>>,
    group_path: &str,
    shard_target_rows: u32,
    kind: SparseMappingKind,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    // As in `write_dense_mapping_section`: one dispatch, at the error type the
    // shared emitter's callback takes; `?` widens it at each call site.
    let emit_io = |w: &mut ScxWriter,
                   name: &str,
                   shard_idx: u32,
                   row_start: u64,
                   n_shard_rows: u64,
                   n_total: u64,
                   batch: &RecordBatch|
     -> scx_format_io::Result<()> {
        match kind {
            SparseMappingKind::Obsp => {
                w.write_obsp_shard_coo(name, shard_idx, row_start, n_shard_rows, n_total, batch)
            }
            SparseMappingKind::Varp => {
                w.write_varp_shard_coo(name, shard_idx, row_start, n_shard_rows, n_total, batch)
            }
        }
    };

    if let Some(entries) = override_entries {
        for (name, batch) in entries {
            let logical = match kind {
                SparseMappingKind::Obsp => format!("obsp/{name}"),
                SparseMappingKind::Varp => format!("varp/{name}"),
            };
            scx_format_io::for_each_coo_mapping_shard(
                &logical,
                batch,
                shard_target_rows,
                |m, sub| {
                    emit_io(
                        writer,
                        name,
                        m.shard_idx,
                        m.row_start,
                        m.n_shard_rows,
                        m.n_rows_total,
                        sub,
                    )
                },
            )?;
        }
        return Ok(());
    }

    // Disk-streaming path. Classify every member of the obsp/varp group:
    //   * valid CSR sparse subgroup → existing COO shard path,
    //   * valid 2D dense dataset    → dense→COO shard path (reuse of the
    //     dense obsm/varm reader; only nonzeros are stored, so a mostly-
    //     zero pairwise matrix never costs the full n_obs² on disk),
    //   * anything else (CSC subgroup, non-2D, malformed CSR) → dropped
    //     with a `DroppedObsp` warning so the loss is visible from Python
    //     and counted in provenance instead of silently swallowed.
    let sparse_infos = list_sparse_mapping_shapes(file, group_path)?;
    let dense_infos = list_dense_mapping_shapes(file, group_path)?;

    // Surface the drop for any member handled by neither reader. The
    // group may be absent entirely (no obsp/varp) — that is not a drop.
    if let Ok(group) = file.group(group_path) {
        use std::collections::HashSet;
        let handled: HashSet<&str> = sparse_infos
            .iter()
            .map(|i| i.name.as_str())
            .chain(dense_infos.iter().map(|i| i.name.as_str()))
            .collect();
        for name in group.member_names()? {
            if name.starts_with("__") || handled.contains(name.as_str()) {
                continue;
            }
            sink.emit(ConvertWarning::DroppedObsp {
                name: format!("{group_path}/{name}"),
                reason: "not a CSR sparse group or 2D dense dataset \
                         (CSC or unsupported pairwise layout)"
                    .to_string(),
            });
        }
    }

    // CSR sparse members.
    for info in &sparse_infos {
        let n_total = info.n_rows as u64;
        // Zero-row sparse mappings: emit a single empty shard so the
        // key survives round-trip (mirrors the override path, which gets the
        // same arm from `scx_format_io::for_each_coo_mapping_shard`).
        if info.n_rows == 0 {
            let batch = read_sparse_mapping_shard(file, group_path, info, 0, 0)?;
            emit_io(writer, &info.name, 0, 0, 0, 0, &batch)?;
            continue;
        }
        let step = shard_target_rows.max(1) as usize;
        let mut shard_idx = 0u32;
        let mut row_start = 0usize;
        while row_start < info.n_rows {
            let row_end = (row_start + step).min(info.n_rows);
            let batch = read_sparse_mapping_shard(file, group_path, info, row_start, row_end)?;
            let n_shard_rows = (row_end - row_start) as u64;
            emit_io(
                writer,
                &info.name,
                shard_idx,
                row_start as u64,
                n_shard_rows,
                n_total,
                &batch,
            )?;
            row_start = row_end;
            shard_idx += 1;
        }
    }

    // Dense members → COO via the existing dense row-shard reader. Each
    // shard's nonzeros are emitted as the same COO `RecordBatch` shape
    // the CSR path produces, so the SCX `ObspEmbeddingShard` /
    // `VarpEmbeddingShard` reader and the h5ad exporter need no changes.
    // A dense `/obsp` therefore re-exports as a sparse matrix (values
    // preserved). Per-shard memory stays bounded to one row-range.
    for info in &dense_infos {
        let n_total = info.n_rows as u64;
        if info.n_rows == 0 {
            let batch = read_dense_mapping_shard(file, group_path, &info.name, 0, 0)?;
            let coo = dense_shard_to_coo(&batch, 0, 0)?;
            emit_io(writer, &info.name, 0, 0, 0, 0, &coo)?;
            continue;
        }
        let step = shard_target_rows.max(1) as usize;
        let mut shard_idx = 0u32;
        let mut row_start = 0usize;
        while row_start < info.n_rows {
            let row_end = (row_start + step).min(info.n_rows);
            let batch = read_dense_mapping_shard(file, group_path, &info.name, row_start, row_end)?;
            let coo = dense_shard_to_coo(&batch, row_start as u64, info.n_rows)?;
            let n_shard_rows = (row_end - row_start) as u64;
            emit_io(
                writer,
                &info.name,
                shard_idx,
                row_start as u64,
                n_shard_rows,
                n_total,
                &coo,
            )?;
            row_start = row_end;
            shard_idx += 1;
        }
    }
    Ok(())
}

/// Convert one dense mapping row-shard (columns `"0".."{k-1}"`, Float32,
/// as produced by [`read_dense_mapping_shard`]) into the COO
/// `RecordBatch` shape the obsp/varp shard writers expect (`row: Int32`,
/// `col: Int32`, `data: Float32` + `n_rows` / `n_cols` schema metadata).
///
/// Only nonzero entries are emitted. `row_start` is the global row offset
/// of this shard; COO `row` values are global (not shard-local), matching
/// [`read_sparse_mapping_shard`]. `n_rows_total` is the logical row count
/// of the full pairwise matrix; `n_cols` is taken from the shard's column
/// count.
fn dense_shard_to_coo(
    batch: &RecordBatch,
    row_start: u64,
    n_rows_total: usize,
) -> Result<RecordBatch, ConvertError> {
    use arrow::array::{Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    let n_cols = batch.num_columns();
    let n_local = batch.num_rows();

    let cols: Vec<&Float32Array> = (0..n_cols)
        .map(|c| {
            batch
                .column(c)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    ConvertError::Other("dense mapping shard column is not Float32".to_string())
                })
        })
        .collect::<Result<_, _>>()?;

    let mut rows: Vec<i32> = Vec::new();
    let mut col_idx: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for local in 0..n_local {
        let global_row = row_start + local as u64;
        let r_i32 = i32::try_from(global_row).map_err(|_| {
            ConvertError::Other(format!(
                "dense pairwise row index {global_row} exceeds i32::MAX \
                 (Arrow COO uses i32 row indices)"
            ))
        })?;
        for (c, arr) in cols.iter().enumerate() {
            let v = arr.value(local);
            if v != 0.0 {
                rows.push(r_i32);
                col_idx.push(c as i32);
                data.push(v);
            }
        }
    }

    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n_rows_total.to_string()),
            ("n_cols".to_string(), n_cols.to_string()),
        ]),
    ));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(rows)),
            Arc::new(Int32Array::from(col_idx)),
            Arc::new(Float32Array::from(data)),
        ],
    )?)
}
