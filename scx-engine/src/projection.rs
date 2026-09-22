// Gene/column projection during CSR decode.
//
// Reduces the number of columns extracted from each shard by only keeping
// entries whose column index appears in the requested gene set.

use arrow::array::{RecordBatch, UInt32Array};
use arrow::compute;
use scx_format_io::catalog::FullCatalogEntry;
use scx_sparse::{ScxCsc, ScxCsr};

use crate::error::Result;
use crate::reader::SectionReader;

/// A CSR column-index element the projection can compare and remap.
///
/// Exists so [`project_csr_row`] serves both shapes a decoded shard comes in:
/// the scipy-facing `i32` of [`SectionReader::read_shard_from_entry`] and the
/// `u32` of its native twin, which the typed (dtype-selected) collect decodes to
/// because integer values must not round through `f32`. Two implementations, two
/// instantiations — the alternative was a second copy of the merge scan, and a
/// projection bug that lands in one copy is exactly the failure this avoids.
pub trait CsrIndex: Copy {
    /// The index as a `u32`, for comparison against the gene set.
    fn to_u32(self) -> u32;
    /// A position within the gene set (`< gene_set.len()`, so always in range
    /// for both implementations) as an index element.
    fn from_gene_position(pos: usize) -> Self;
}

impl CsrIndex for i32 {
    fn to_u32(self) -> u32 {
        self as u32
    }
    fn from_gene_position(pos: usize) -> Self {
        pos as i32
    }
}

impl CsrIndex for u32 {
    fn to_u32(self) -> u32 {
        self
    }
    fn from_gene_position(pos: usize) -> Self {
        pos as u32
    }
}

/// Project a single CSR row, keeping only entries whose column index
/// appears in `gene_set`.
///
/// # Precondition
///
/// Both `indices` (column indices for one row) and `gene_set` MUST be
/// sorted ascending. The merge scan uses a monotonic `gi` pointer that
/// never decreases; on unsorted `indices` it silently drops every
/// column that follows a larger predecessor in the row, producing a
/// wrong-but-not-noisy result. Callers passing a scipy `csr_matrix`
/// must either check `has_sorted_indices` or call `.sort_indices()` /
/// `.sorted_indices()` before extracting `indices`. The canonical pyscx
/// boundary helper is `pyscx::convert::ensure_csr`, which already
/// enforces this. Debug builds catch violations via `debug_assert!`;
/// release builds skip the check, so the boundary fix is what holds
/// correctness in production.
///
/// Uses a sorted merge scan — O(nnz_row + n_genes).
///
/// Returns `(projected_indices, projected_data)` where indices are
/// remapped to `0..gene_set.len()` (position in gene_set).
pub fn project_csr_row<I: CsrIndex, V: Copy>(
    indices: &[I],
    data: &[V],
    gene_set: &[u32],
) -> (Vec<I>, Vec<V>) {
    let mut out_indices = Vec::new();
    let mut out_data = Vec::new();
    project_csr_row_into(&mut out_indices, &mut out_data, indices, data, gene_set);
    (out_indices, out_data)
}

/// [`project_csr_row`] appending into caller-owned buffers.
///
/// The merge scan itself — the part with the monotonic-pointer precondition —
/// lives here and nowhere else; [`project_csr_row`] is a thin wrapper that
/// allocates. Shard-level callers project row after row, so returning a fresh
/// pair of `Vec`s per row cost two allocations and two frees per kept row, on
/// both the f32 and the native decode paths. Appending into one pair of buffers
/// per shard removes all of it.
pub fn project_csr_row_into<I: CsrIndex, V: Copy>(
    out_indices: &mut Vec<I>,
    out_data: &mut Vec<V>,
    indices: &[I],
    data: &[V],
    gene_set: &[u32],
) {
    // Compared as `u32`, which is how the merge scan below reads them too — so
    // the assertion checks the order the scan actually depends on rather than the
    // element type's own order.
    debug_assert!(
        indices.windows(2).all(|w| w[0].to_u32() <= w[1].to_u32()),
        "project_csr_row: row indices are not sorted ascending — unsorted input \
         silently produces wrong results because the merge scan uses a monotonic \
         pointer. Sort indices at the caller (e.g. via scipy `sort_indices()` or \
         `pyscx::convert::ensure_csr`) before invoking."
    );
    // **Strictly** ascending, not merely non-decreasing: `gi` only advances
    // while `gene_set[gi] < col`, so it parks on the first of a run of equal
    // values and every later copy is an output column that can never fire.
    // Duplicates are therefore as wrong as disorder, and were not covered.
    debug_assert!(
        gene_set.windows(2).all(|w| w[0] < w[1]),
        "project_csr_row: gene_set is not strictly ascending — the merge scan \
         requires ascending *and* unique. `project_csr` sorts and dedups \
         internally; direct callers must too."
    );

    let mut gi = 0; // pointer into gene_set

    for (&col_idx, &val) in indices.iter().zip(data.iter()) {
        let col = col_idx.to_u32();
        // Advance gene_set pointer past values smaller than current column
        while gi < gene_set.len() && gene_set[gi] < col {
            gi += 1;
        }
        if gi >= gene_set.len() {
            break; // remaining row indices are all beyond gene_set
        }
        if gene_set[gi] == col {
            // Remap to position within gene_set
            out_indices.push(I::from_gene_position(gi));
            out_data.push(val);
        }
        // If gene_set[gi] > col, this column is not in the gene set — skip it
    }
}

/// Project an entire `ScxCsr` matrix to keep only the specified gene columns.
///
/// `gene_indices` are the original column indices to retain. They are sorted
/// internally if not already sorted. The output CSR has
/// `n_cols = gene_indices.len()` with column indices remapped to `0..n_cols`.
pub fn project_csr(csr: &ScxCsr, gene_indices: &[u32]) -> ScxCsr {
    // Sort gene indices (stable sort to preserve order of equal elements)
    let mut sorted_genes: Vec<u32> = gene_indices.to_vec();
    sorted_genes.sort_unstable();
    sorted_genes.dedup();

    let n_rows = csr.n_rows();
    let n_cols = sorted_genes.len();

    let mut new_indptr = Vec::with_capacity(n_rows + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    for row in 0..n_rows {
        let start = csr.indptr[row] as usize;
        let end = csr.indptr[row + 1] as usize;

        let (row_indices, row_data) = project_csr_row(
            &csr.indices[start..end],
            &csr.data[start..end],
            &sorted_genes,
        );

        new_indices.extend_from_slice(&row_indices);
        new_data.extend_from_slice(&row_data);
        let prev = *new_indptr.last().unwrap();
        new_indptr.push(prev + row_indices.len() as i64);
    }

    ScxCsr::new_unchecked((n_rows, n_cols), new_indptr, new_indices, new_data)
}

/// Reorder the columns of a projected CSR into a caller-requested
/// presentation order.
///
/// `csr` is expected to be the output of [`project_csr`] — i.e. its
/// columns are `0..n` in **sorted** gene order. `out_to_sorted` maps each
/// output column position to the sorted column it should take its values
/// from: output column `k` is the projected (sorted) column
/// `out_to_sorted[k]`. `out_to_sorted` must be a permutation of `0..n`
/// where `n == csr.n_cols()`.
///
/// This is the final step that lets `var_names` honour the caller's
/// request order while the gather itself stays monotonic (the merge scan
/// in [`project_csr_row`] requires sorted columns). Indices within each
/// row are re-sorted so the result is canonical CSR (sorted row indices),
/// keeping scipy interop and any later [`project_csr`] composition safe.
pub fn reorder_csr_columns(csr: &ScxCsr, out_to_sorted: &[u32]) -> ScxCsr {
    let n_rows = csr.n_rows();
    let n_cols = csr.n_cols();
    debug_assert_eq!(
        out_to_sorted.len(),
        n_cols,
        "reorder_csr_columns: out_to_sorted length must equal csr.n_cols()"
    );

    // Inverse map: sorted column -> output column position.
    let mut sorted_to_out = vec![0u32; n_cols];
    for (out_pos, &sorted_col) in out_to_sorted.iter().enumerate() {
        sorted_to_out[sorted_col as usize] = out_pos as u32;
    }

    let mut new_indptr = Vec::with_capacity(n_rows + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::with_capacity(csr.indices.len());
    let mut new_data = Vec::with_capacity(csr.data.len());

    // Scratch reused per row to sort (remapped_index, value) pairs.
    let mut row: Vec<(i32, f32)> = Vec::new();
    for r in 0..n_rows {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        row.clear();
        row.reserve(end - start);
        for (&old_idx, &val) in csr.indices[start..end].iter().zip(&csr.data[start..end]) {
            row.push((sorted_to_out[old_idx as usize] as i32, val));
        }
        row.sort_unstable_by_key(|&(idx, _)| idx);
        for &(idx, val) in &row {
            new_indices.push(idx);
            new_data.push(val);
        }
        let prev = *new_indptr.last().unwrap();
        new_indptr.push(prev + (end - start) as i64);
    }

    ScxCsr::new_unchecked((n_rows, n_cols), new_indptr, new_indices, new_data)
}

/// Project the var (gene) RecordBatch to keep only the specified gene rows.
///
/// Uses `arrow::compute::take()` for efficient row selection.
pub fn project_var(var: &RecordBatch, gene_indices: &[u32]) -> Result<RecordBatch> {
    let mut sorted_indices: Vec<u32> = gene_indices.to_vec();
    sorted_indices.sort_unstable();
    sorted_indices.dedup();
    let take_indices = UInt32Array::from(sorted_indices);
    let columns: Vec<_> = var
        .columns()
        .iter()
        .map(|col| compute::take(col.as_ref(), &take_indices, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(RecordBatch::try_new(var.schema(), columns)?)
}

/// Decode a shard and apply gene projection in one step.
///
/// Decodes the full shard via `reader.read_shard_from_entry()`, then applies
/// per-row projection to extract only the requested gene columns. Returns
/// scipy-compatible `(indptr, indices, data)` with remapped column indices.
///
/// **Note**: True decode-time projection (filtering during FOR-BP decode)
/// would be more efficient but requires modifying scx-codec. Start with
/// post-decode projection; optimize into the codec layer if benchmarks
/// show it matters.
pub fn decode_shard_projected(
    reader: &dyn SectionReader,
    entry: &FullCatalogEntry,
    gene_indices: &[u32],
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    // Sort gene indices for merge scan
    let mut sorted_genes: Vec<u32> = gene_indices.to_vec();
    sorted_genes.sort_unstable();
    sorted_genes.dedup();
    decode_shard_projected_presorted(reader, entry, &sorted_genes)
}

/// [`decode_shard_projected`] for a caller that has already established the
/// precondition: `gene_set` **strictly** ascending.
///
/// The query path resolves its projection once per query
/// (`collect::execute::resolve_gene_projection` dedups through a `HashSet`, then
/// sorts) and then decodes shard by shard, so re-establishing it inside the
/// per-shard rayon closure was one `Vec<u32>` clone plus one sort per decoded
/// shard, discarded on return. The typed decode
/// (`collect::native::decode_shard_native_filtered`) already states and relies
/// on this same precondition; `project_csr_row_into`'s own `debug_assert!` is
/// what checks it.
pub(crate) fn decode_shard_projected_presorted(
    reader: &dyn SectionReader,
    entry: &FullCatalogEntry,
    sorted_genes: &[u32],
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    let (indptr, indices, data) = reader.read_shard_from_entry(entry)?;

    let n_rows = indptr.len() - 1;
    let mut new_indptr = Vec::with_capacity(n_rows + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;

        // Appends rather than returning a pair of `Vec`s per row.
        project_csr_row_into(
            &mut new_indices,
            &mut new_data,
            &indices[start..end],
            &data[start..end],
            sorted_genes,
        );
        new_indptr.push(new_indices.len() as i64);
    }

    Ok((new_indptr, new_indices, new_data))
}

// ---------------------------------------------------------------------------
// CSC column projection — column-major counterpart to project_csr
// ---------------------------------------------------------------------------

/// Borrowed-slice view of a single CSC column.
///
/// Returns `(indices, data)` for column `col` — the row indices and
/// values stored under that column. Zero-copy hot path; useful when a
/// caller iterates one column at a time without materializing a new
/// `ScxCsc`.
///
/// Panics if `col >= indptr.len() - 1`. Callers are expected to bound
/// `col < n_cols` from a known-correct `ScxCsc`.
pub fn project_csc_column<'a>(
    indptr: &[i64],
    indices: &'a [i32],
    data: &'a [f32],
    col: usize,
) -> (&'a [i32], &'a [f32]) {
    let start = indptr[col] as usize;
    let end = indptr[col + 1] as usize;
    (&indices[start..end], &data[start..end])
}

/// Project an entire `ScxCsc` matrix to keep only the specified gene
/// (column) indices.
///
/// Mirrors [`project_csr`] for the column axis. `gene_indices` are the
/// original column indices to retain; sorted internally if not already
/// sorted, deduplicated. Output column indices are remapped to
/// `0..gene_indices.len()` (position in the deduplicated/sorted set).
/// Row indices in the output are unchanged from the input (CSC indices
/// are already global row IDs).
pub fn project_csc(csc: &ScxCsc, gene_indices: &[u32]) -> ScxCsc {
    let mut sorted_genes: Vec<u32> = gene_indices.to_vec();
    sorted_genes.sort_unstable();
    sorted_genes.dedup();

    let n_rows = csc.n_rows();
    let n_cols_out = sorted_genes.len();

    // Pre-compute total nnz to size buffers without growth churn.
    let mut total_nnz: usize = 0;
    for &g in &sorted_genes {
        let g = g as usize;
        if g >= csc.n_cols() {
            // Out-of-range gene index contributes nothing; skip.
            continue;
        }
        total_nnz += (csc.indptr[g + 1] - csc.indptr[g]) as usize;
    }

    let mut new_indptr = Vec::with_capacity(n_cols_out + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::with_capacity(total_nnz);
    let mut new_data = Vec::with_capacity(total_nnz);

    for &g in &sorted_genes {
        let g = g as usize;
        if g < csc.n_cols() {
            let (col_idx, col_data) = project_csc_column(&csc.indptr, &csc.indices, &csc.data, g);
            new_indices.extend_from_slice(col_idx);
            new_data.extend_from_slice(col_data);
        }
        new_indptr.push(new_data.len() as i64);
    }

    ScxCsc::new_unchecked((n_rows, n_cols_out), new_indptr, new_indices, new_data)
}

/// Compact a CSC slab's rows onto a filtered (live) row space, in place.
///
/// A CSC sidecar stores each nonzero's **global physical** row, and a row
/// filter — `filter_cells`, a subset, or a deletion vector — renumbers the rows
/// the caller can see. Every CSC consumer reads its row space off the source
/// (`ColumnShardSource::n_obs`) and then indexes a *visible-length* per-row
/// array by the slab's stored row: a group label, a `cell_to_group` entry, a
/// dense `[n_obs x chunk]` scatter slot. So a filtered handle needs the slab's
/// rows translated before it leaves the reader, which is what this does.
///
/// `global_to_live[g]` is the live index of global row `g`, or a negative value
/// when `g` is filtered out. Nonzeros on dropped rows are removed; survivors
/// keep their value and take their live row. `indptr` is rewritten, `indices`
/// and `data` are truncated, and `shape.0` becomes `n_live`.
///
/// Two invariants carry over rather than needing re-establishing, because a
/// filter is monotone: row order **within** a column is preserved (which
/// `scx_accel::csc::pseudobulk` relies on in practice), and no column gains a
/// duplicate row, since distinct global rows have distinct live indices.
///
/// A row outside `global_to_live` is treated as dropped, matching the `row >=
/// n_obs` skip every CSC kernel already performs — a reader must not panic on a
/// malformed slab. Debug builds assert instead, so a real bug is loud in tests.
///
/// `O(nnz)`, one pass, no allocation.
pub fn compact_csc_rows_in_place(csc: &mut ScxCsc, global_to_live: &[i32], n_live: usize) {
    debug_assert!(
        csc.indices
            .iter()
            .all(|&r| r >= 0 && (r as usize) < global_to_live.len()),
        "compact_csc_rows_in_place: a slab row index is outside global_to_live ({}); \
         a CSC shard's indices must be global physical rows",
        global_to_live.len(),
    );

    let n_cols = csc.n_cols();
    // Write cursor into `indices`/`data`, always <= the read position, so the
    // compaction is a single in-place pass rather than a copy into new buffers.
    let mut w = 0usize;
    // The read cursor has to be carried rather than re-read from
    // `indptr[col]`: the previous iteration has already overwritten that entry
    // with its *write* position, so reading it here would restart each column
    // at the compacted offset and duplicate entries.
    let mut read_start = csc.indptr.first().copied().unwrap_or(0) as usize;
    for col in 0..n_cols {
        let read_end = csc.indptr[col + 1] as usize;
        for r in read_start..read_end {
            // `usize::try_from` rather than `as usize`: a negative row index
            // in a malformed slab would otherwise wrap to `usize::MAX` and be
            // dropped by `get`, which is the right *behaviour* reached by an
            // implicit two's-complement cast. Say it instead.
            let live = usize::try_from(csc.indices[r])
                .ok()
                .and_then(|row| global_to_live.get(row).copied())
                .unwrap_or(-1);
            if live >= 0 {
                csc.indices[w] = live;
                csc.data[w] = csc.data[r];
                w += 1;
            }
        }
        // Rewrite this column's end, having consumed its old value above.
        csc.indptr[col + 1] = w as i64;
        read_start = read_end;
    }
    csc.indices.truncate(w);
    csc.data.truncate(w);
    csc.shape.0 = n_live;

    debug_assert_eq!(csc.shape.0, n_live);
    debug_assert_eq!(csc.indices.len(), w);
    debug_assert_eq!(csc.indptr.last().copied(), Some(w as i64));
}
#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;
    use scx_format_io::ScxReader;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // D1: project_csr_row
    // -----------------------------------------------------------------------

    #[test]
    fn project_row_basic() {
        // Row with columns [0, 3, 5, 7], select genes {3, 5}
        let indices: Vec<i32> = vec![0, 3, 5, 7];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gene_set = vec![3, 5];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert_eq!(out_idx, vec![0, 1]); // remapped: 3→0, 5→1
        assert_eq!(out_data, vec![2.0, 3.0]);
    }

    #[test]
    fn project_row_no_match() {
        // Row has columns [0, 1, 2], gene_set is {5, 6}
        let indices: Vec<i32> = vec![0, 1, 2];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let gene_set = vec![5, 6];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert!(out_idx.is_empty());
        assert!(out_data.is_empty());
    }

    #[test]
    fn project_row_all_match() {
        let indices: Vec<i32> = vec![2, 4, 6];
        let data: Vec<f32> = vec![10.0, 20.0, 30.0];
        let gene_set = vec![2, 4, 6];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert_eq!(out_idx, vec![0, 1, 2]);
        assert_eq!(out_data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn project_row_empty_gene_set() {
        let indices: Vec<i32> = vec![0, 1, 2];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let gene_set: Vec<u32> = vec![];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert!(out_idx.is_empty());
        assert!(out_data.is_empty());
    }

    /// The native instantiation the typed (dtype-selected) collect runs on:
    /// `u32` indices and `u32` values. The f32 tests above cannot reach it, and
    /// an integer value domain is the whole point of the typed path — a count
    /// above 2²⁴ must survive the projection without touching `f32`.
    #[test]
    fn project_row_native_u32_domain() {
        let indices: Vec<u32> = vec![0, 3, 5, 7];
        let big = (1u32 << 24) + 7; // odd, and not representable exactly in f32
        let data: Vec<u32> = vec![1, big, 3, 4];
        let gene_set = vec![3, 5];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert_eq!(out_idx, vec![0u32, 1]); // remapped: 3→0, 5→1
        assert_eq!(out_data, vec![big, 3]);
        // The value that motivates the typed path did not round on the way.
        assert_ne!(out_data[0] as f32 as u32, out_data[0]);
    }

    #[test]
    fn project_row_empty_row() {
        let indices: Vec<i32> = vec![];
        let data: Vec<f32> = vec![];
        let gene_set = vec![0, 1, 2];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert!(out_idx.is_empty());
        assert!(out_data.is_empty());
    }

    // -----------------------------------------------------------------------
    // D1: project_csr
    // -----------------------------------------------------------------------

    fn sample_csr() -> ScxCsr {
        // 3×10 matrix:
        // row 0: cols [1, 3] = [5.0, 10.0]
        // row 1: cols [0, 2, 4] = [1.0, 3.0, 7.0]
        // row 2: cols [2] = [2.0]
        ScxCsr::new(
            (3, 10),
            vec![0, 2, 5, 6],
            vec![1, 3, 0, 2, 4, 2],
            vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn project_3_genes_from_10() {
        let csr = sample_csr();
        // Select genes {0, 2, 4} → 3 columns
        let projected = project_csr(&csr, &[0, 2, 4]);
        assert_eq!(projected.shape, (3, 3));
        assert_eq!(projected.n_rows(), 3);
        assert_eq!(projected.n_cols(), 3);

        // row 0: original [1, 3] → none of {0,2,4} match 1 or 3 → empty
        assert_eq!(projected.indptr[0], 0);
        assert_eq!(projected.indptr[1], 0);

        // row 1: original [0, 2, 4] → all match → remapped [0, 1, 2]
        assert_eq!(projected.indptr[2], 3);
        assert_eq!(&projected.indices[0..3], &[0, 1, 2]);
        assert_eq!(&projected.data[0..3], &[1.0, 3.0, 7.0]);

        // row 2: original [2] → matches gene_set[1]=2 → remapped [1]
        assert_eq!(projected.indptr[3], 4);
        assert_eq!(projected.indices[3], 1);
        assert_eq!(projected.data[3], 2.0);
    }

    #[test]
    fn project_gene_not_present_in_any_row() {
        let csr = sample_csr();
        // Gene index 9 doesn't appear in any row
        let projected = project_csr(&csr, &[9]);
        assert_eq!(projected.shape, (3, 1));
        assert_eq!(projected.nnz(), 0);
        assert_eq!(projected.indptr, vec![0, 0, 0, 0]);
    }

    #[test]
    fn project_preserves_values_exactly() {
        let csr = sample_csr();
        let projected = project_csr(&csr, &[2]);
        // Only gene 2: row 0 has nothing, row 1 has 3.0, row 2 has 2.0
        assert_eq!(projected.shape, (3, 1));
        assert_eq!(projected.nnz(), 2);
        assert_eq!(projected.data, vec![3.0, 2.0]);
    }

    #[test]
    fn project_column_indices_remapped() {
        let csr = sample_csr();
        // Select genes {1, 4} → remapped to {0, 1}
        let projected = project_csr(&csr, &[1, 4]);
        assert_eq!(projected.shape, (3, 2));
        // row 0: col 1 → remapped 0
        assert_eq!(projected.indices[0], 0);
        // row 1: col 4 → remapped 1
        assert_eq!(projected.indices[1], 1);
    }

    #[test]
    fn project_empty_gene_indices() {
        let csr = sample_csr();
        let projected = project_csr(&csr, &[]);
        assert_eq!(projected.shape, (3, 0));
        assert_eq!(projected.nnz(), 0);
    }

    #[test]
    fn project_unsorted_gene_indices() {
        let csr = sample_csr();
        // Unsorted input: {4, 0, 2} should be handled correctly
        let projected = project_csr(&csr, &[4, 0, 2]);
        // Should produce same result as sorted {0, 2, 4}
        let projected_sorted = project_csr(&csr, &[0, 2, 4]);
        assert_eq!(projected.shape, projected_sorted.shape);
        assert_eq!(projected.indptr, projected_sorted.indptr);
        assert_eq!(projected.indices, projected_sorted.indices);
        assert_eq!(projected.data, projected_sorted.data);
    }

    #[test]
    fn project_large_gene_count_to_small() {
        // Simulate "3 genes from 30K" scenario
        let n_vars = 30_000;
        let n_rows = 5;
        // Build a CSR where each row has indices [0, 100, 200] with values [1.0, 2.0, 3.0]
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for _ in 0..n_rows {
            indices.extend_from_slice(&[0, 100, 200]);
            data.extend_from_slice(&[1.0, 2.0, 3.0]);
            indptr.push(*indptr.last().unwrap() + 3);
        }
        let csr = ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data);

        // Project to 3 specific genes
        let projected = project_csr(&csr, &[0, 100, 200]);
        assert_eq!(projected.shape, (n_rows, 3));
        assert_eq!(projected.nnz(), n_rows * 3);
        // All projected indices should be 0, 1, 2 (remapped)
        for row in 0..n_rows {
            let start = projected.indptr[row] as usize;
            let end = projected.indptr[row + 1] as usize;
            assert_eq!(&projected.indices[start..end], &[0, 1, 2]);
            assert_eq!(&projected.data[start..end], &[1.0, 2.0, 3.0]);
        }
    }

    // -----------------------------------------------------------------------
    // reorder_csr_columns
    // -----------------------------------------------------------------------

    #[test]
    fn reorder_columns_basic() {
        // Sorted-order projection of sample_csr to genes {0, 2, 4} (cols 0,1,2).
        let csr = sample_csr();
        let sorted = project_csr(&csr, &[0, 2, 4]);
        // Request order was [4, 0, 2] -> sorted [0, 2, 4]; output col k takes
        // sorted col perm[k]: perm = [2, 0, 1] (4 is sorted-pos 2, 0 is 0, 2 is 1).
        let reordered = reorder_csr_columns(&sorted, &[2, 0, 1]);
        assert_eq!(reordered.shape, (3, 3));

        // Densify and compare: reordered col j == sorted col perm[j].
        let dense_sorted = sorted.to_dense().unwrap(); // 3x3 row-major
        let dense_re = reordered.to_dense().unwrap();
        let perm = [2usize, 0, 1];
        for r in 0..3 {
            for (j, &p) in perm.iter().enumerate() {
                assert_eq!(dense_re[r * 3 + j], dense_sorted[r * 3 + p]);
            }
        }
    }

    #[test]
    fn reorder_columns_identity() {
        let csr = sample_csr();
        let sorted = project_csr(&csr, &[0, 2, 4]);
        let reordered = reorder_csr_columns(&sorted, &[0, 1, 2]);
        assert_eq!(reordered.indptr, sorted.indptr);
        assert_eq!(reordered.indices, sorted.indices);
        assert_eq!(reordered.data, sorted.data);
    }

    #[test]
    fn reorder_columns_indices_stay_sorted_within_row() {
        let csr = sample_csr();
        let sorted = project_csr(&csr, &[0, 2, 4]);
        let reordered = reorder_csr_columns(&sorted, &[2, 0, 1]);
        for r in 0..reordered.n_rows() {
            let start = reordered.indptr[r] as usize;
            let end = reordered.indptr[r + 1] as usize;
            assert!(reordered.indices[start..end]
                .windows(2)
                .all(|w| w[0] < w[1]));
        }
    }

    // -----------------------------------------------------------------------
    // D1: project_var
    // -----------------------------------------------------------------------

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

    #[test]
    fn project_var_selects_correct_rows() {
        let var = sample_var(10);
        let projected = project_var(&var, &[0, 3, 7]).unwrap();
        assert_eq!(projected.num_rows(), 3);

        let gene_ids = projected
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(gene_ids.value(0), "gene_0");
        assert_eq!(gene_ids.value(1), "gene_3");
        assert_eq!(gene_ids.value(2), "gene_7");
    }

    #[test]
    fn project_var_empty_indices() {
        let var = sample_var(10);
        let projected = project_var(&var, &[]).unwrap();
        assert_eq!(projected.num_rows(), 0);
    }

    #[test]
    fn project_var_single_gene() {
        let var = sample_var(5);
        let projected = project_var(&var, &[2]).unwrap();
        assert_eq!(projected.num_rows(), 1);

        let gene_ids = projected
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(gene_ids.value(0), "gene_2");
    }

    // -----------------------------------------------------------------------
    // D2: decode_shard_projected
    // -----------------------------------------------------------------------

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
    }

    fn sample_obs(n: usize) -> RecordBatch {
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
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

        // Build shard data: each row has 2 entries at deterministic column positions
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_obs {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            // Ensure sorted column order within row
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

    #[test]
    fn decode_projected_matches_decode_then_project() {
        let dir = tempfile::tempdir().unwrap();
        let n_obs = 10;
        let n_vars = 20;
        let path = write_test_file(&dir, n_obs, n_vars);
        let reader = ScxReader::open(&path).unwrap();

        let gene_indices = vec![0u32, 2, 5, 10, 15];

        // Method 1: decode full shard, then project
        let shards = reader.catalog().shards_sorted();
        let entry = shards[0];
        let (indptr1, indices1, data1) = reader.read_shard_from_entry(entry).unwrap();
        let full_csr = ScxCsr::new_unchecked((n_obs, n_vars), indptr1, indices1, data1);
        let projected = project_csr(&full_csr, &gene_indices);

        // Method 2: decode_shard_projected
        let (indptr2, indices2, data2) =
            decode_shard_projected(&reader, entry, &gene_indices).unwrap();

        // They must produce the same result
        assert_eq!(projected.indptr, indptr2);
        assert_eq!(projected.indices, indices2);
        assert_eq!(projected.data, data2);
    }

    /// The public entry still normalises its gene set; only the internal
    /// `_presorted` form assumes the caller did.
    ///
    /// The sort and dedup moved out of the per-shard body into this wrapper, so
    /// the query path (which resolves a sorted, unique set once per query) stops
    /// re-establishing it per shard. Nothing in the tree passes an unsorted set
    /// today, which is exactly why the public contract needs a test rather than
    /// a reader's trust: a caller outside this crate can, and the merge scan
    /// silently drops columns when the order or the uniqueness is wrong.
    #[test]
    fn decode_projected_normalises_an_unsorted_duplicated_gene_set() {
        let dir = tempfile::tempdir().unwrap();
        let (n_obs, n_vars) = (10, 20);
        let path = write_test_file(&dir, n_obs, n_vars);
        let reader = ScxReader::open(&path).unwrap();
        let entry = reader.catalog().shards_sorted()[0];

        let sorted = vec![0u32, 2, 5, 10, 15];
        let scrambled = vec![15u32, 2, 0, 10, 2, 5, 15];
        assert_ne!(
            scrambled, sorted,
            "the input must not already be normalised"
        );

        let (ip_ref, idx_ref, data_ref) = decode_shard_projected(&reader, entry, &sorted).unwrap();
        let (ip, idx, data) = decode_shard_projected(&reader, entry, &scrambled).unwrap();

        assert_eq!(ip, ip_ref);
        assert_eq!(idx, idx_ref);
        assert_eq!(data, data_ref);
        // Not vacuous: the projection actually kept something.
        assert!(
            !idx.is_empty(),
            "the fixture must hit some projected column"
        );
    }

    // -----------------------------------------------------------------------
    // project_csc / project_csc_column
    // -----------------------------------------------------------------------

    fn sample_csc() -> ScxCsc {
        // 3×5 matrix (transpose of sample_csr's 3x10 — different fixture
        // for clarity here):
        //   col 0: row 1 -> 1.0
        //   col 1: row 0 -> 5.0
        //   col 2: row 1 -> 3.0, row 2 -> 2.0
        //   col 3: row 0 -> 10.0
        //   col 4: row 1 -> 7.0
        ScxCsc::new(
            (3, 5),
            vec![0, 1, 2, 4, 5, 6],
            vec![1, 0, 1, 2, 0, 1],
            vec![1.0, 5.0, 3.0, 2.0, 10.0, 7.0],
        )
        .unwrap()
    }

    #[test]
    fn project_csc_column_borrowed_slice() {
        let csc = sample_csc();
        let (col2_idx, col2_data) = project_csc_column(&csc.indptr, &csc.indices, &csc.data, 2);
        assert_eq!(col2_idx, &[1, 2]);
        assert_eq!(col2_data, &[3.0, 2.0]);

        let (col0_idx, col0_data) = project_csc_column(&csc.indptr, &csc.indices, &csc.data, 0);
        assert_eq!(col0_idx, &[1]);
        assert_eq!(col0_data, &[1.0]);
    }

    #[test]
    fn project_csc_basic() {
        let csc = sample_csc();
        // Select columns {0, 2, 4} → 3 columns of output, in remapped
        // 0..3 order.
        let projected = project_csc(&csc, &[0, 2, 4]);
        assert_eq!(projected.shape, (3, 3));
        assert_eq!(projected.nnz(), 4);
        // col 0 (orig 0): row 1 -> 1.0
        assert_eq!(&projected.indptr[0..2], &[0, 1]);
        assert_eq!(&projected.indices[0..1], &[1]);
        assert_eq!(&projected.data[0..1], &[1.0]);
        // col 1 (orig 2): row 1 -> 3.0, row 2 -> 2.0
        assert_eq!(projected.indptr[2], 3);
        assert_eq!(&projected.indices[1..3], &[1, 2]);
        assert_eq!(&projected.data[1..3], &[3.0, 2.0]);
        // col 2 (orig 4): row 1 -> 7.0
        assert_eq!(projected.indptr[3], 4);
        assert_eq!(projected.indices[3], 1);
        assert_eq!(projected.data[3], 7.0);
    }

    #[test]
    fn project_csc_unsorted_input() {
        let csc = sample_csc();
        let a = project_csc(&csc, &[4, 0, 2]);
        let b = project_csc(&csc, &[0, 2, 4]);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn project_csc_dedup() {
        let csc = sample_csc();
        let projected = project_csc(&csc, &[2, 2, 2]);
        // Three duplicates of column 2 should collapse to one.
        assert_eq!(projected.shape, (3, 1));
        assert_eq!(projected.nnz(), 2);
    }

    #[test]
    fn project_csc_empty_indices() {
        let csc = sample_csc();
        let projected = project_csc(&csc, &[]);
        assert_eq!(projected.shape, (3, 0));
        assert_eq!(projected.nnz(), 0);
    }

    #[test]
    fn project_csc_out_of_range_gene_skipped() {
        let csc = sample_csc(); // 5 columns
                                // Gene 99 is past the end; should be skipped, leaving an empty
                                // projected column.
        let projected = project_csc(&csc, &[2, 99]);
        assert_eq!(projected.shape, (3, 2));
        // Output col 0 (orig 2) has 2 nnz; col 1 (orig 99) has 0.
        assert_eq!(projected.indptr, vec![0, 2, 2]);
    }

    /// Densified equivalence: project then to_dense == to_dense then
    /// column-slice.
    #[test]
    fn project_csc_dense_equivalence() {
        let csc = sample_csc();
        let dense_full = csc.to_dense().unwrap(); // row-major, 3×5
        let projected = project_csc(&csc, &[1, 3]);
        let dense_proj = projected.to_dense().unwrap();
        // Reference: dense matrix's columns 1 and 3 in column-major sense
        let mut expected = vec![0.0f32; 3 * 2];
        for r in 0..3 {
            expected[r * 2] = dense_full[r * 5 + 1];
            expected[r * 2 + 1] = dense_full[r * 5 + 3];
        }
        assert_eq!(dense_proj, expected);
    }

    #[test]
    fn decode_projected_empty_genes() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 5, 10);
        let reader = ScxReader::open(&path).unwrap();

        let shards = reader.catalog().shards_sorted();
        let entry = shards[0];
        let (indptr, indices, data) = decode_shard_projected(&reader, entry, &[]).unwrap();

        assert_eq!(indptr.len(), 6); // 5 rows + 1
        assert!(indices.is_empty());
        assert!(data.is_empty());
        // All indptr values should be 0 (no nnz per row)
        assert!(indptr.iter().all(|&v| v == 0));
    }
    // -----------------------------------------------------------------------
    // compact_csc_rows_in_place
    //
    // Every case is checked against a dense oracle rather than against
    // hand-written indptr/indices, because the failure this function exists to
    // prevent is *silent*: a CSC kernel skips a row it cannot place
    // (`row >= n_obs`) and mislabels the rest, so a structural assertion can
    // pass while the values are wrong.
    // -----------------------------------------------------------------------

    /// Build a CSC from a row-major dense matrix, storing every nonzero with
    /// rows ascending within each column.
    fn csc_from_dense(n_rows: usize, n_cols: usize, dense: &[f32]) -> ScxCsc {
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for c in 0..n_cols {
            for r in 0..n_rows {
                let v = dense[r * n_cols + c];
                if v != 0.0 {
                    indices.push(r as i32);
                    data.push(v);
                }
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsc::new((n_rows, n_cols), indptr, indices, data).expect("fixture is canonical")
    }

    /// Read a CSC back as a row-major dense matrix.
    fn dense_from_csc(csc: &ScxCsc) -> Vec<f32> {
        let (n_rows, n_cols) = csc.shape;
        let mut dense = vec![0.0f32; n_rows * n_cols];
        for c in 0..n_cols {
            let (idx, vals) = project_csc_column(&csc.indptr, &csc.indices, &csc.data, c);
            for (&r, &v) in idx.iter().zip(vals.iter()) {
                dense[r as usize * n_cols + c] = v;
            }
        }
        dense
    }

    /// `global_to_live` for a sorted list of kept global rows.
    fn live_map(n_physical: usize, kept: &[usize]) -> Vec<i32> {
        let mut map = vec![-1i32; n_physical];
        for (live, &g) in kept.iter().enumerate() {
            map[g] = live as i32;
        }
        map
    }

    /// The oracle: the dense matrix restricted to `kept`, rows in kept order.
    fn dense_rows(n_cols: usize, dense: &[f32], kept: &[usize]) -> Vec<f32> {
        let mut out = Vec::with_capacity(kept.len() * n_cols);
        for &g in kept {
            out.extend_from_slice(&dense[g * n_cols..(g + 1) * n_cols]);
        }
        out
    }

    /// 6x4, deliberately ragged: column 1 lives only on rows the tests drop,
    /// column 3 only on rows they keep, row 2 is structurally empty, and
    /// there is a stored value on every other row.
    fn compaction_fixture() -> (usize, usize, Vec<f32>) {
        let (n_rows, n_cols) = (6usize, 4usize);
        #[rustfmt::skip]
        let dense: Vec<f32> = vec![
            1.0, 0.0, 0.0, 4.0,   // row 0
            0.0, 2.5, 3.0, 0.0,   // row 1
            0.0, 0.0, 0.0, 0.0,   // row 2 — structurally empty
            7.0, 0.0, 0.0, 8.0,   // row 3
            0.0, 9.5, 0.0, 0.0,   // row 4
            5.0, 0.0, 6.0, 1.5,   // row 5
        ];
        (n_rows, n_cols, dense)
    }

    /// Compacting against the identity map is a no-op on every array.
    #[test]
    fn compacting_with_an_identity_map_changes_nothing() {
        let (n_rows, n_cols, dense) = compaction_fixture();
        let original = csc_from_dense(n_rows, n_cols, &dense);
        let mut csc = original.clone();
        let kept: Vec<usize> = (0..n_rows).collect();
        compact_csc_rows_in_place(&mut csc, &live_map(n_rows, &kept), n_rows);
        assert_eq!(csc.shape, original.shape);
        assert_eq!(csc.indptr, original.indptr);
        assert_eq!(csc.indices, original.indices);
        assert_eq!(csc.data, original.data);
    }

    /// The interleaved case, against the dense oracle. This is the one that
    /// matters: rows 1 and 4 are dropped from the middle, so every surviving
    /// row above them shifts down by a different amount.
    #[test]
    fn compacting_renumbers_the_live_rows_and_drops_the_rest() {
        let (n_rows, n_cols, dense) = compaction_fixture();
        let mut csc = csc_from_dense(n_rows, n_cols, &dense);
        let kept = [0usize, 2, 3, 5];
        compact_csc_rows_in_place(&mut csc, &live_map(n_rows, &kept), kept.len());
        assert_eq!(csc.shape, (kept.len(), n_cols));
        assert_eq!(dense_from_csc(&csc), dense_rows(n_cols, &dense, &kept));
        // And the slab is still canonical in its new row space.
        ScxCsc::new(
            csc.shape,
            csc.indptr.clone(),
            csc.indices.clone(),
            csc.data.clone(),
        )
        .expect("compaction output must pass full validation");
    }

    /// Dropping the extremes is the boundary case for the write cursor.
    #[test]
    fn dropping_the_first_and_last_rows_keeps_the_middle_intact() {
        let (n_rows, n_cols, dense) = compaction_fixture();
        let mut csc = csc_from_dense(n_rows, n_cols, &dense);
        let kept = [1usize, 2, 3, 4];
        compact_csc_rows_in_place(&mut csc, &live_map(n_rows, &kept), kept.len());
        assert_eq!(dense_from_csc(&csc), dense_rows(n_cols, &dense, &kept));
    }

    /// A column all of whose rows are dropped comes back empty rather than
    /// carrying a stale `indptr` span. Column 1 lives only on rows 1 and 4.
    #[test]
    fn a_column_whose_rows_are_all_dropped_becomes_empty() {
        let (n_rows, n_cols, dense) = compaction_fixture();
        let mut csc = csc_from_dense(n_rows, n_cols, &dense);
        let kept = [0usize, 3, 5];
        compact_csc_rows_in_place(&mut csc, &live_map(n_rows, &kept), kept.len());
        assert_eq!(
            csc.indptr[2] - csc.indptr[1],
            0,
            "column 1's only rows were dropped"
        );
        assert_eq!(dense_from_csc(&csc), dense_rows(n_cols, &dense, &kept));
    }

    /// Keeping nothing empties the slab but leaves it structurally valid —
    /// `indptr` still has `n_cols + 1` entries, all zero.
    #[test]
    fn compacting_to_no_rows_leaves_a_valid_empty_slab() {
        let (n_rows, n_cols, dense) = compaction_fixture();
        let mut csc = csc_from_dense(n_rows, n_cols, &dense);
        compact_csc_rows_in_place(&mut csc, &live_map(n_rows, &[]), 0);
        assert_eq!(csc.shape, (0, n_cols));
        assert_eq!(csc.nnz(), 0);
        assert_eq!(csc.indptr.len(), n_cols + 1);
        assert!(csc.indptr.iter().all(|&v| v == 0));
        ScxCsc::new(
            csc.shape,
            csc.indptr.clone(),
            csc.indices.clone(),
            csc.data.clone(),
        )
        .expect("an empty slab is still a valid one");
    }

    /// `shape.0` is the slab's own account of its row space. Nothing on the DE
    /// path reads it — the kernels take `n_obs` from the source and the GPU
    /// validator from `source.shape()` — so this test is the only thing that
    /// catches a compaction that forgets to update it.
    #[test]
    fn compaction_reports_the_live_row_count_as_its_shape() {
        let (n_rows, n_cols, dense) = compaction_fixture();
        let mut csc = csc_from_dense(n_rows, n_cols, &dense);
        let kept = [2usize, 4];
        compact_csc_rows_in_place(&mut csc, &live_map(n_rows, &kept), kept.len());
        assert_eq!(csc.shape.0, 2);
        assert!(csc.indices.iter().all(|&r| (r as usize) < 2));
    }

    /// A row index the map cannot describe is a malformed slab. Debug builds
    /// say so; release builds drop the entry, which is what every CSC kernel
    /// already does with an out-of-range row (`row >= n_obs { continue }`),
    /// because a reader must not panic on bad input.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "outside global_to_live")]
    fn a_row_index_outside_the_map_is_a_debug_assertion() {
        let mut csc = ScxCsc::new_unchecked((9, 1), vec![0, 1], vec![8], vec![1.0]);
        compact_csc_rows_in_place(&mut csc, &live_map(4, &[0, 1]), 2);
    }
}
