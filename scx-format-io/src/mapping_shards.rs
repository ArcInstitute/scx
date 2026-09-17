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
use arrow::array::{RecordBatch, UInt32Array};
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
    let md = batch.schema_ref().metadata();
    let get = |key: &str| -> Result<usize> {
        let raw = md.get(key).ok_or_else(|| {
            ScxError::InvalidCatalog(format!(
                "{logical}: pairwise COO batch has no '{key}' schema metadata, so its \
                 logical extent is unknown"
            ))
        })?;
        raw.parse::<usize>().map_err(|_| {
            ScxError::InvalidCatalog(format!(
                "{logical}: pairwise COO batch has '{key}'='{raw}', which is not a \
                 non-negative integer extent"
            ))
        })
    };
    Ok((get("n_rows")?, get("n_cols")?))
}

/// The **whole** COO wire contract this crate can read back: exactly three
/// columns named `row` / `col` / `data`; both coordinates `Int32` or `Int64`
/// and at the same width; `data` `Float32` or `Float64`; and no nulls in any of
/// the three.
///
/// Checked at **write** time because the emitter is `pub`, and because
/// `BackedPairwiseReader::from_layout` only inspects field *names* — so
/// everything else here is a payload that opens fine and then fails on first
/// decode, in `coo_coord_column` or `coo_data_column`. A write that succeeds
/// into an unreadable file is the worst of the available failures: it is
/// discovered later, by someone else, on a different machine.
///
/// The null rule is the one that narrows what callers may pass.
/// `scx-ops::compact::remap_obsp_coo_to_dim` *preserves* its input's `data`
/// nullability, so a nullable field reaches here routinely — that is fine, and
/// the field's bit is preserved. An actual null is not: `coo_data_column`
/// refuses one, so emitting it would write a graph no bounded read can decode.
/// Refusing at the writer converts that into an error naming the column.
fn validate_coo_schema(logical: &str, batch: &RecordBatch) -> Result<()> {
    use arrow::datatypes::DataType;
    let schema = batch.schema_ref();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    if names != ["row", "col", "data"] {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: pairwise COO batch has columns {names:?}, expected exactly \
             [\"row\", \"col\", \"data\"]"
        )));
    }
    let (r, c) = (schema.field(0).data_type(), schema.field(1).data_type());
    if !matches!(r, DataType::Int32 | DataType::Int64) {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: pairwise COO coordinate columns have dtype {r:?}, expected Int32 \
             or Int64"
        )));
    }
    if r != c {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: pairwise COO batch has row dtype {r:?} and col dtype {c:?}; both \
             coordinate columns must be the same width"
        )));
    }
    let d = schema.field(2).data_type();
    if !matches!(d, DataType::Float32 | DataType::Float64) {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: pairwise COO data column has dtype {d:?}, expected Float32 or Float64"
        )));
    }
    for (idx, what) in [
        (0usize, "coordinate column 0"),
        (1, "coordinate column 1"),
        (2, "data column"),
    ] {
        let n = batch.column(idx).null_count();
        if n > 0 {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: pairwise COO {what} has {n} null entries; the bounded reader \
                 refuses a null coordinate or value, so emitting one would write a graph \
                 it cannot decode"
            )));
        }
    }
    Ok(())
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
/// *multiset* of triples. No reader's **matrix semantics** depend on that
/// order (`BackedPairwiseReader::read_rows_range` counting-sorts and then sorts
/// columns within each row; the h5ad exporter sorts by `(row, col)` to build a
/// canonical CSR; scipy's `coo_matrix` does not care), but a caller reading the
/// raw triples through `ScxReader::read_obsp` and relying on their sequence
/// does see the regrouping.
pub fn for_each_coo_mapping_shard<F>(
    logical: &str,
    batch: &RecordBatch,
    shard_target_rows: u32,
    mut f: F,
) -> Result<()>
where
    F: FnMut(DenseShardMetadata, &RecordBatch) -> Result<()>,
{
    validate_coo_schema(logical, batch)?;
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

    // A gather needs one index per triple, so the partition costs `nnz`
    // positions however it is written. `u32` rather than `u64` halves that, and
    // the ceiling it imposes is not a real restriction: a batch with more than
    // `u32::MAX` triples is one every reader in this crate already cannot
    // handle, since decoding a COO shard materialises `Vec<i64>` coordinates
    // (34 GB at that size) before it does anything else.
    if nnz > u32::MAX as usize {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: pairwise COO batch has {nnz} triples, above the {} this emitter \
             partitions; split the mapping into row bands first",
            u32::MAX
        )));
    }

    // Row coordinates are **borrowed**, not materialised: this is a single
    // scan, and a copy would cost 8 B per triple for the duration.
    let rows = crate::backed::coo_coord_borrow(batch, 0, logical)?;

    // Counting sort into one contiguous position array, rather than a
    // `Vec<Vec<_>>` per bucket. Same O(nnz), but 4 B per triple with no
    // per-bucket allocation or growth slack — `sort` and `compact` run this on
    // a graph they are already holding a full remapped copy of, so the
    // partition's own footprint is the part worth not doubling.
    //
    // Pass 1 counts, and is where a row coordinate is validated.
    let mut counts = vec![0u32; n_shards + 1];
    for i in 0..rows.len() {
        let r = rows.at(i);
        if r < 0 {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: pairwise COO row index {r} is negative"
            )));
        }
        // Against `n_rows`, NOT against the bucket it divides into. The last
        // bucket's arithmetic range runs to `n_shards * step` while its stamp
        // stops at `n_rows`, so whenever `n_rows` is not a multiple of the
        // target there is slack in between that a `shard < n_shards` test
        // admits — and the triple then lands in a shard whose stamped span
        // excludes it, producing a file `BackedPairwiseReader::decode_shard`
        // refuses to read. Checking the coordinate makes `shard < n_shards`
        // hold by construction.
        if r as usize >= n_rows {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: pairwise COO row index {r} is outside the declared n_rows={n_rows}"
            )));
        }
        counts[(r as usize) / step + 1] += 1;
    }
    // Prefix-sum the counts into bucket start offsets.
    for i in 0..n_shards {
        counts[i + 1] += counts[i];
    }
    let starts = counts.clone();
    // Pass 2 scatters each triple's position into its bucket's run, preserving
    // input order within a bucket (a stable partition, as documented above).
    let mut positions = vec![0u32; nnz];
    let mut cursor = counts;
    for i in 0..rows.len() {
        let shard = (rows.at(i) as usize) / step;
        positions[cursor[shard] as usize] = i as u32;
        cursor[shard] += 1;
    }

    // One index array over the whole partition, sliced per shard. A
    // `positions[lo..hi].to_vec()` per shard would hold the full 4 B/nnz buffer
    // *and* a copy of the current bucket — on a graph concentrated in one row
    // band that doubles the partition's peak, which is the cost this counting
    // sort exists to avoid. Arrow slicing shares the buffer.
    let all_indices = UInt32Array::from(positions);
    let total = n_rows as u64;
    for shard_idx in 0..n_shards {
        let row_start = shard_idx * step;
        let n_shard_rows = step.min(n_rows - row_start);
        let (lo, hi) = (starts[shard_idx] as usize, starts[shard_idx + 1] as usize);
        let indices = all_indices.slice(lo, hi - lo);
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
