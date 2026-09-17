//! Row-shard emitters for the four auxiliary mapping families — `obsm`,
//! `varm` (dense) and `obsp`, `varp` (pairwise COO).
//!
//! # Why this is one module and not four copies
//!
//! Before this existed there were **three** bucketers, in three crates, and
//! each hardcoded something different:
//!
//! * `scx-convert`'s `partition_coo_to_shards` accepted `Int32` coordinates
//!   and `Float32` values only, by positional downcast;
//! * `pyscx`'s `for_each_coo_shard` was width-generic over `Int32`/`Int64`
//!   coordinates but still required `Float32` values and rejected a nullable
//!   value column;
//! * `pyscx`'s `for_each_dense_shard` was the dense twin of the same loop,
//!   duplicated a fourth time in [`crate::writer::write_obs_section`] and a
//!   fifth in `scx-ops`' `sort_engine::write_obs_sharded`.
//!
//! Two bucketers is how two families come to disagree about a boundary; three
//! is how a caller finds that the one reachable from its crate is the one that
//! cannot represent its data. `scx-ops` cannot depend on `scx-convert` or
//! `pyscx` (the graph is `scx-format-io → {scx-ops, scx-convert, pyscx}`), so
//! the emitters live here, below everyone who needs them.
//!
//! # Why the COO emitter gathers with `take` rather than rebuilding arrays
//!
//! The hardcoded dtypes above are not a stylistic matter. `scx-ops`'
//! `compact::remap_obsp_coo_to_dim` **preserves** its input's `data` column
//! dtype (`Float32` *or* `Float64`) and its nullability, and chooses the
//! coordinate width (`Int32` or `Int64`) from the surviving axis extent. A
//! bucketer that rebuilds the output arrays by matching on dtype has to
//! enumerate that product, and every branch it forgets is an op that starts
//! refusing a file it used to write.
//!
//! So the COO emitter never looks at the value column at all. It computes,
//! per shard, the list of *triple positions* that belong to it and hands the
//! whole batch to [`arrow::compute::take_record_batch`], which builds the
//! output from `batch.schema()` — preserving every field's dtype and
//! nullability, and the schema metadata (`n_rows` / `n_cols`) with it.
//! Adding a coordinate or value width to the format needs no change here.
//!
//! # The cover contract
//!
//! Both emitters produce a **contiguous, ordered, gap-free** cover of
//! `[0, n_rows)`: `shard_idx` equals the emission position, `row_start` equals
//! the previous shard's `row_start + n_shard_rows`, and the final
//! `row_start + n_shard_rows` equals the stamped `n_rows_total`. The reader
//! validates exactly that, twice — in
//! `reader::metadata::assemble_sharded_metadata` for a whole-matrix read and
//! again in `reader::metadata::row_sharded_mapping_layout` for a bounded one.
//!
//! The consequence worth stating: the COO emitter emits **every** bucket in
//! `0..n_shards`, including buckets with no triples in them. A sparse graph
//! with an empty row band still gets a shard for that band, because skipping
//! it would put a gap in the cover and the reader would reject the file.

use crate::writer::DenseShardMetadata;
use arrow::array::{RecordBatch, UInt64Array};
use scx_format::{Result, ScxError};

/// Bucket-count ceiling for the COO emitter.
///
/// The bucket table is sized by the logical row axis, not by nnz, so a v2
/// wide-axis `obsp` (axis ≥ 2^31) at a small shard target would allocate
/// hundreds of thousands of empty `Vec`s before looking at a single triple.
/// Fail with a message that names the two ways out rather than OOM.
const MAX_BUCKETS: usize = 1_000_000;

/// Read the logical `n_rows` / `n_cols` a pairwise COO batch declares.
///
/// These are not derivable from the batch: its Arrow row count is the number
/// of `(row, col, data)` triples, so a graph with an empty trailing row band
/// cannot be distinguished from a smaller matrix without the declaration.
fn coo_dims(logical: &str, batch: &RecordBatch) -> Result<(usize, usize)> {
    let md = batch.schema_ref().metadata().clone();
    let get = |key: &str| -> Result<usize> {
        md.get(key)
            .ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "{logical}: pairwise COO batch has no '{key}' schema metadata, so its \
                     logical extent is unknown"
                ))
            })?
            .parse::<usize>()
            .map_err(|_| {
                ScxError::InvalidCatalog(format!(
                    "{logical}: pairwise COO batch has '{key}'='{}', which is not a row count",
                    md.get(key).map(String::as_str).unwrap_or("")
                ))
            })
    };
    Ok((get("n_rows")?, get("n_cols")?))
}

/// Slice a dense `obsm` / `varm` batch into row-aligned shards and hand each
/// to `f` with the metadata a `write_*_shard` call needs.
///
/// One Arrow row is one logical row here, so the split is
/// [`RecordBatch::slice`] — zero-copy, and dtype-agnostic without trying.
///
/// A zero-row batch emits **one** empty shard stamped `n_rows_total = 0`
/// rather than nothing at all, so a key that exists but covers no rows
/// survives a round trip instead of vanishing from the catalog.
pub fn for_each_dense_mapping_shard<F>(
    batch: &RecordBatch,
    shard_target_rows: u32,
    mut f: F,
) -> Result<()>
where
    F: FnMut(DenseShardMetadata, &RecordBatch) -> Result<()>,
{
    let n_rows = batch.num_rows();
    if n_rows == 0 {
        return f(DenseShardMetadata::new(0, 0, 0, 0), batch);
    }
    let step = shard_target_rows.max(1) as usize;
    let total = n_rows as u64;
    let (mut shard_idx, mut cursor) = (0u32, 0usize);
    while cursor < n_rows {
        let take = step.min(n_rows - cursor);
        let slice = batch.slice(cursor, take);
        f(
            DenseShardMetadata::new(shard_idx, cursor as u64, take as u64, total),
            &slice,
        )?;
        shard_idx += 1;
        cursor += take;
    }
    Ok(())
}

/// Bucket a pairwise COO `obsp` / `varp` batch into row-aligned shards by
/// `row / shard_target_rows` and hand each to `f`.
///
/// `logical` is the label used in error messages (e.g. `"obsp/connectivities"`).
///
/// Order: triples are grouped by shard, and within a shard they keep their
/// relative input order — a stable partition, not a sort. Concatenating the
/// shards therefore does **not** reproduce the input batch row for row unless
/// the input was already ordered by `row / step`; it reproduces the same
/// *multiset* of triples. Every reader of a COO mapping establishes its own
/// order (`BackedPairwiseReader::read_rows_range` counting-sorts and then
/// sorts columns within each row; the h5ad exporter sorts by `(row, col)` to
/// build a canonical CSR; scipy's `coo_matrix` does not care), so the
/// regrouping is not observable through any of them.
pub fn for_each_coo_mapping_shard<F>(
    logical: &str,
    batch: &RecordBatch,
    shard_target_rows: u32,
    mut f: F,
) -> Result<()>
where
    F: FnMut(DenseShardMetadata, &RecordBatch) -> Result<()>,
{
    if batch.num_columns() < 3 {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: pairwise COO batch has {} columns, expected at least 3 (row/col/data)",
            batch.num_columns()
        )));
    }
    let (n_rows, _n_cols) = coo_dims(logical, batch)?;
    let nnz = batch.num_rows();

    if n_rows == 0 {
        // A zero-row axis with triples on it is not an empty graph, it is a
        // batch whose declared extent contradicts its payload. Writing it as
        // "one shard covering nothing" would bury that: the cover validates,
        // and the triples are then unreachable through every bounded read.
        if nnz != 0 {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: pairwise COO batch declares n_rows=0 but carries {nnz} triples"
            )));
        }
        return f(DenseShardMetadata::new(0, 0, 0, 0), batch);
    }

    let step = shard_target_rows.max(1) as usize;
    let n_shards = n_rows.div_ceil(step);
    if n_shards > MAX_BUCKETS {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: n_rows={n_rows} at a shard target of {step} rows would need {n_shards} \
             shard buckets (cap {MAX_BUCKETS}); raise the shard target or split the mapping \
             into row bands first"
        )));
    }

    // Row coordinates as i64 regardless of the on-disk width. This is the one
    // column the emitter has to interpret; `take` handles the rest.
    let rows = crate::backed::coo_coord_column(batch, 0, logical)?;

    let mut buckets: Vec<Vec<u64>> = vec![Vec::new(); n_shards];
    for (i, &r) in rows.iter().enumerate() {
        if r < 0 {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: pairwise COO row index {r} is negative"
            )));
        }
        let shard = (r as usize) / step;
        if shard >= n_shards {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: pairwise COO row index {r} is outside the declared n_rows={n_rows}"
            )));
        }
        buckets[shard].push(i as u64);
    }

    let total = n_rows as u64;
    for (shard_idx, bucket) in buckets.into_iter().enumerate() {
        let row_start = shard_idx * step;
        let n_shard_rows = step.min(n_rows - row_start);
        let indices = UInt64Array::from(bucket);
        // Reuses `batch.schema()`, so field dtypes, field nullability and the
        // `n_rows` / `n_cols` metadata all survive verbatim.
        let shard = arrow::compute::take_record_batch(batch, &indices)?;
        f(
            DenseShardMetadata::new(
                shard_idx as u32,
                row_start as u64,
                n_shard_rows as u64,
                total,
            ),
            &shard,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "mapping_shards_tests.rs"]
mod tests;
