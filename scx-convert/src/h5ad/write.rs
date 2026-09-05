//! SCX → h5ad export: the top-level drivers and the matrix/mapping group
//! writers.
//!
//! The dataframe (`/obs`, `/var`) side of the export does **not** live here —
//! it is 64 % of what this file used to be and splits three ways:
//! [`super::column_stream`] (the nine column encodings and their two drivers),
//! [`super::columns`] (the layout/schema pre-pass they both run) and
//! [`super::categorical`] (dictionary decode and the cross-shard vocabulary
//! accumulator). `/uns` is [`super::uns`].

use super::column_stream::write_dataframe_group_at;
use super::uns::write_uns_entries;
use crate::h5_write_util::vlu;
use crate::pipeline::ConvertError;
use crate::warnings::WarningSink;
use arrow::array::{Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use hdf5::types::VarLenUnicode;
use scx_format_io::reader::ScxReader;
use std::collections::HashMap;
use std::path::Path;

/// Write an SCX file to h5ad format.
pub fn write_scx_to_h5ad(
    scx_path: &Path,
    h5ad_path: &Path,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    let file = hdf5::File::create(h5ad_path)?;

    // Honor deletion vectors on every leg (X, obs, layers). The CSR
    // and layer readers filter by DV directly; obs goes through the
    // shared streaming-or-eager dispatcher which applies the same
    // global keep mask. Pre-fix, this path silently dropped DV
    // semantics — both /X and obs were written unfiltered.
    let keep_mask = super::stream_write::build_keep_mask(&reader)?;

    // Read the full CSR matrix (DV-filtered when active)
    let csr = reader.read_all_csr_shards_filtered()?;
    let n_obs = csr.shape.0;
    let n_vars = csr.shape.1;

    // Write X as CSR group
    write_sparse_group(
        &file,
        "X",
        &csr.indptr,
        &csr.indices,
        &csr.data,
        n_obs,
        n_vars,
    )?;

    // Write obs / var via the shared dispatcher so legacy + sharded
    // sources both flow through one code path and the DV keep mask
    // is honored.
    let root = file.as_group()?;
    super::stream_write::write_obs_streaming_or_eager(&root, &reader, keep_mask.as_deref(), sink)?;
    super::stream_write::write_var_streaming_or_eager(&root, &reader, sink)?;

    // Write obsm (DV-filtered when active — obs-axis rows must match
    // /X and /obs). `read_all_obsm` returns Ok(empty) when absent, so a
    // propagated error means genuine corruption — never swallow it (SCX-009).
    let obsm_map = reader.read_all_obsm()?;
    if !obsm_map.is_empty() {
        let obsm_group = file.create_group("obsm")?;
        for (name, batch) in &obsm_map {
            let filtered = match keep_mask.as_deref() {
                Some(mask) => super::stream_write::filter_record_batch_by_mask(batch, mask)?,
                None => batch.clone(),
            };
            write_obsm_entry(&obsm_group, name, &filtered)?;
        }
    }

    // Write uns. Absence is a clean `SectionNotFound`; any other error
    // (e.g. malformed JSON) is corruption and must abort (SCX-009).
    match reader.read_uns() {
        Ok(uns) => {
            let uns_group = file.create_group("uns")?;
            write_uns_entries(&uns_group, &uns, 1, sink)?;
        }
        Err(scx_format_io::error::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }

    // Write layers (DV-filtered when active — layers share X's
    // row count by AnnData invariant). `layer_names()` already enumerates the
    // layers present, so a read failure here is corruption, not absence — it
    // must abort rather than silently omit the layer (SCX-009).
    let layer_names = reader.layer_names();
    if !layer_names.is_empty() {
        let layers_group = file.create_group("layers")?;
        for layer_name in &layer_names {
            let layer_csr = reader.read_layer_filtered(layer_name)?;
            let lg = layers_group.create_group(layer_name)?;
            write_sparse_arrays(
                &lg,
                &layer_csr.indptr,
                &layer_csr.indices,
                &layer_csr.data,
                layer_csr.shape.0,
                layer_csr.shape.1,
            )?;
        }
    }

    // Write /raw (DV-filtered on the obs axis like /X).
    write_raw_to_h5ad(&root, &reader, keep_mask.as_deref(), sink)?;

    // Write obsp / varp pairwise matrices (COO → csr_matrix groups). obsp is
    // square on the obs axis, so deletion vectors filter BOTH axes; varp lives
    // on the var axis and is never obs-deleted. Both readers return Ok(empty)
    // on absence, so errors are corruption and propagate (SCX-009).
    let obsp = reader.read_all_obsp()?;
    write_pairwise_group(&root, "obsp", &obsp, keep_mask.as_deref())?;
    let varp = reader.read_all_varp()?;
    write_pairwise_group(&root, "varp", &varp, None)?;

    Ok(())
}

/// Write the `adata.raw` group (`raw/X` + `raw/var`) into an output
/// h5ad if the SCX file carries a raw matrix. Raw shares X's obs axis,
/// so the same deletion-vector keep mask is applied to its rows. Shared
/// by the eager and streaming SCX→h5ad export paths.
pub(crate) fn write_raw_to_h5ad(
    root: &hdf5::Group,
    reader: &ScxReader,
    keep_mask: Option<&[bool]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    if !reader.has_raw() {
        return Ok(());
    }
    let raw = reader.read_all_raw_csr_shards()?;
    let raw_n_vars = raw.shape.1;
    let (indptr, indices, data, n_obs) = match keep_mask {
        Some(mask) => filter_csr_rows(&raw.indptr, &raw.indices, &raw.data, mask)?,
        None => (raw.indptr, raw.indices, raw.data, raw.shape.0),
    };

    let raw_group = root.create_group("raw")?;
    write_sparse_group_at(&raw_group, "X", &indptr, &indices, &data, n_obs, raw_n_vars)?;
    let raw_var = reader.read_raw_var()?;
    write_dataframe_group_at(&raw_group, "var", &raw_var, sink)?;
    Ok(())
}

/// `(indptr, indices, data, n_kept_rows)` for a row-filtered CSR.
type FilteredCsr = (Vec<i64>, Vec<i32>, Vec<f32>, usize);

/// Subset CSR rows by a boolean obs keep-mask, returning new
/// `(indptr, indices, data, n_kept_rows)`. Used to apply deletion
/// vectors to the raw matrix on export (raw shares the obs axis).
fn filter_csr_rows(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    mask: &[bool],
) -> Result<FilteredCsr, ConvertError> {
    // `min(mask.len())` here silently dropped every row past the shorter of the
    // two, exporting a raw matrix with fewer rows than obs — the same
    // clamp-instead-of-check the reader-side deletion filters and the engine's
    // row filter had. The mask is obs-indexed and raw shares the obs axis, so a
    // mismatch means the file is inconsistent, not that the export should guess.
    let declared = indptr.len().saturating_sub(1);
    if mask.len() != declared {
        return Err(ConvertError::Scx(scx_format_io::ScxError::InvalidCatalog(
            format!(
                "raw export keep mask covers {} rows but raw X has {} \
                 (truncated or corrupt file)",
                mask.len(),
                declared,
            ),
        )));
    }
    let n_rows = declared;
    let mut out_indptr = vec![0i64];
    let mut out_indices = Vec::new();
    let mut out_data = Vec::new();
    for (row, &keep) in mask.iter().enumerate().take(n_rows) {
        if keep {
            let s = indptr[row] as usize;
            let e = indptr[row + 1] as usize;
            out_indices.extend_from_slice(&indices[s..e]);
            out_data.extend_from_slice(&data[s..e]);
            out_indptr.push(out_indices.len() as i64);
        }
    }
    let n = out_indptr.len() - 1;
    Ok((out_indptr, out_indices, out_data, n))
}

fn write_sparse_group(
    file: &hdf5::File,
    name: &str,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<(), ConvertError> {
    let group = file.create_group(name)?;
    write_sparse_arrays(&group, indptr, indices, data, n_obs, n_vars)
}

// Module-internal helpers for the h5mu writer (Phase D.2). All
// take a parent `hdf5::Group` instead of the root `hdf5::File` so
// per-modality blocks under `/mod/{name}/…` can reuse the same
// emitters as `/X`, `/obs`, `/var`, `/obsm/…`, `/uns/…`.

pub(crate) fn write_sparse_group_at(
    parent: &hdf5::Group,
    name: &str,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<(), ConvertError> {
    let group = parent.create_group(name)?;
    write_sparse_arrays(&group, indptr, indices, data, n_obs, n_vars)
}

pub(crate) fn write_obsm_entry_at(
    obsm_group: &hdf5::Group,
    name: &str,
    batch: &arrow::array::RecordBatch,
) -> Result<(), ConvertError> {
    write_obsm_entry(obsm_group, name, batch)
}

fn write_sparse_arrays(
    group: &hdf5::Group,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<(), ConvertError> {
    // Write arrays
    group
        .new_dataset::<i64>()
        .shape([indptr.len()])
        .create("indptr")?
        .write(indptr)?;
    group
        .new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")?
        .write(indices)?;
    group
        .new_dataset::<f32>()
        .shape([data.len()])
        .create("data")?
        .write(data)?;

    // Set attributes
    let encoding_type = vlu("csr_matrix");
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&encoding_type)?;

    let encoding_version = vlu("0.1.0");
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&encoding_version)?;

    let shape = [n_obs as i64, n_vars as i64];
    group
        .new_attr::<i64>()
        .shape([2])
        .create("shape")?
        .write(&shape)?;

    Ok(())
}

/// Read a COO column (`row` / `col`) as `i64`, accepting both the v1
/// (`Int32`) and v2 (`Int64`) coordinate widths the pairwise readers emit.
fn coo_coord_column(batch: &RecordBatch, name: &str) -> Result<Vec<i64>, ConvertError> {
    let col = batch.column_by_name(name).ok_or_else(|| {
        ConvertError::Other(format!("pairwise COO batch missing '{name}' column"))
    })?;
    if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        Ok(a.values().iter().map(|&v| v as i64).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        Ok(a.values().to_vec())
    } else {
        Err(ConvertError::Other(format!(
            "pairwise COO column '{name}' is neither Int32 nor Int64"
        )))
    }
}

/// CSR arrays (`indptr`, `indices`, `data`) plus the square dimension `n`.
type CooCsr = (Vec<i64>, Vec<i32>, Vec<f32>, usize);

/// Convert a pairwise COO `RecordBatch` (`row`, `col`, `data: Float32` +
/// `n_rows` / `n_cols` schema metadata) into CSR arrays for an h5ad
/// `csr_matrix` group. When `keep` is `Some`, the matrix is filtered on
/// **both** axes by the obs keep-mask (pairwise matrices are square on the
/// obs axis) and indices are remapped into the compacted space; `varp`
/// passes `None`. Output is sorted by `(row, col)` and is `n × n`.
fn coo_batch_to_csr(batch: &RecordBatch, keep: Option<&[bool]>) -> Result<CooCsr, ConvertError> {
    let meta = batch.schema_ref().metadata();
    let n_rows: usize = meta
        .get("n_rows")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| ConvertError::Other("pairwise COO batch missing n_rows metadata".into()))?;

    let rows = coo_coord_column(batch, "row")?;
    let cols = coo_coord_column(batch, "col")?;
    let data = batch
        .column_by_name("data")
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| ConvertError::Other("pairwise COO batch missing Float32 'data'".into()))?;

    // Build the old→new index remap (identity when no deletions). `remap[i]`
    // is the compacted index of kept row/col `i`, or -1 when dropped.
    let (remap, n_out): (Option<Vec<i64>>, usize) = match keep {
        Some(mask) => {
            let mut remap = vec![-1i64; n_rows];
            let mut next = 0i64;
            for (i, slot) in remap.iter_mut().enumerate().take(n_rows.min(mask.len())) {
                if mask[i] {
                    *slot = next;
                    next += 1;
                }
            }
            (Some(remap), next as usize)
        }
        None => (None, n_rows),
    };

    // CSR `indices` are i32 (scipy zero-copy). The largest column index emitted
    // is `n_out - 1`; reject axes too wide to represent rather than silently
    // wrapping the `c as i32` narrowing below. Checked here (before the
    // `indptr` allocation) so an oversized dimension fails fast.
    if n_out > i32::MAX as usize {
        return Err(ConvertError::Other(format!(
            "pairwise matrix dimension {n_out} exceeds the i32 CSR index limit ({})",
            i32::MAX
        )));
    }

    // Filter + remap into (row, col, value) triples.
    let mut triples: Vec<(i64, i64, f32)> = Vec::with_capacity(rows.len());
    for k in 0..rows.len() {
        let (r, c) = (rows[k], cols[k]);
        let (nr, nc) = match &remap {
            Some(remap) => {
                let (Some(&nr), Some(&nc)) = (remap.get(r as usize), remap.get(c as usize)) else {
                    continue;
                };
                if nr < 0 || nc < 0 {
                    continue;
                }
                (nr, nc)
            }
            // No remap (varp / no deletions): coordinates index `indptr`
            // directly, so an out-of-range coord from a malformed COO section
            // would panic. Reject it as a conversion error instead.
            None => {
                if r < 0 || r >= n_rows as i64 || c < 0 || c >= n_rows as i64 {
                    return Err(ConvertError::Other(format!(
                        "pairwise COO coordinate ({r}, {c}) out of bounds for \
                         {n_rows}x{n_rows} matrix"
                    )));
                }
                (r, c)
            }
        };
        triples.push((nr, nc, data.value(k)));
    }

    // Canonical CSR: sort by (row, col), then build indptr.
    triples.sort_by_key(|&(r, c, _)| (r, c));
    let mut indptr = vec![0i64; n_out + 1];
    let mut indices = Vec::with_capacity(triples.len());
    let mut values = Vec::with_capacity(triples.len());
    for &(r, c, v) in &triples {
        indptr[r as usize + 1] += 1;
        indices.push(c as i32);
        values.push(v);
    }
    for i in 0..n_out {
        indptr[i + 1] += indptr[i];
    }

    Ok((indptr, indices, values, n_out))
}

/// Write `obsp` / `varp` pairwise matrices into an h5ad group (one
/// `csr_matrix` subgroup per key). `keep` filters both axes by the obs
/// keep-mask (pass `Some` for `obsp` under deletion vectors, `None` for
/// `varp`, which lives on the var axis and is never obs-deleted).
pub(crate) fn write_pairwise_group(
    parent: &hdf5::Group,
    group_name: &str,
    entries: &HashMap<String, RecordBatch>,
    keep: Option<&[bool]>,
) -> Result<(), ConvertError> {
    if entries.is_empty() {
        return Ok(());
    }
    let group = parent.create_group(group_name)?;
    for (name, batch) in entries {
        let (indptr, indices, data, n) = coo_batch_to_csr(batch, keep)?;
        let sub = group.create_group(name)?;
        write_sparse_arrays(&sub, &indptr, &indices, &data, n, n)?;
    }
    Ok(())
}

fn write_obsm_entry(
    obsm_group: &hdf5::Group,
    name: &str,
    batch: &RecordBatch,
) -> Result<(), ConvertError> {
    let n_rows = batch.num_rows();
    let n_cols = batch.num_columns();

    // Flatten to 2D f32 array (finding 8.11: handle non-Float32 columns).
    let mut flat = vec![0.0f32; n_rows * n_cols];
    for col_idx in 0..n_cols {
        let col = batch.column(col_idx);
        if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
            for row_idx in 0..n_rows {
                flat[row_idx * n_cols + col_idx] = arr.value(row_idx);
            }
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            for row_idx in 0..n_rows {
                flat[row_idx * n_cols + col_idx] = arr.value(row_idx) as f32;
            }
        } else {
            return Err(ConvertError::Other(format!(
                "obsm '{name}' column {col_idx}: expected Float32 or Float64 array, got {:?}",
                col.data_type()
            )));
        }
    }

    let ds = obsm_group
        .new_dataset::<f32>()
        .shape([n_rows, n_cols])
        .create(name)?;

    // Write using ndarray
    let nd_array = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| ConvertError::Other(format!("ndarray shape error: {e}")))?;
    ds.write(&nd_array)?;

    Ok(())
}

#[cfg(test)]
mod pairwise_tests {
    use super::*;
    use arrow::array::{Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a COO pairwise `RecordBatch` (`row`/`col` Int32, `data` f32 +
    /// `n_rows`/`n_cols` metadata) like the obsp/varp readers emit.
    fn coo(n: usize, triples: &[(i32, i32, f32)]) -> RecordBatch {
        let rows: Vec<i32> = triples.iter().map(|t| t.0).collect();
        let cols: Vec<i32> = triples.iter().map(|t| t.1).collect();
        let data: Vec<f32> = triples.iter().map(|t| t.2).collect();
        let schema = Schema::new_with_metadata(
            vec![
                Field::new("row", DataType::Int32, false),
                Field::new("col", DataType::Int32, false),
                Field::new("data", DataType::Float32, false),
            ],
            HashMap::from([
                ("n_rows".to_string(), n.to_string()),
                ("n_cols".to_string(), n.to_string()),
            ]),
        );
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int32Array::from(rows)),
                Arc::new(Int32Array::from(cols)),
                Arc::new(Float32Array::from(data)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn coo_to_csr_identity_no_mask() {
        // 4×4: (0,0)=1 (0,2)=2 (2,1)=3 (3,3)=4 — deliberately unsorted input.
        let batch = coo(4, &[(0, 2, 2.0), (3, 3, 4.0), (0, 0, 1.0), (2, 1, 3.0)]);
        let (indptr, indices, data, n) = coo_batch_to_csr(&batch, None).unwrap();
        assert_eq!(n, 4);
        assert_eq!(indptr, vec![0, 2, 2, 3, 4]);
        assert_eq!(indices, vec![0, 2, 1, 3]); // row0 cols sorted, then row2, row3
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn coo_to_csr_keep_mask_filters_both_axes() {
        // Drop index 1 on both axes: keep = [T,F,T,T] → remap 0→0, 2→1, 3→2.
        let batch = coo(4, &[(0, 0, 1.0), (0, 2, 2.0), (2, 1, 3.0), (3, 3, 4.0)]);
        let keep = [true, false, true, true];
        let (indptr, indices, data, n) = coo_batch_to_csr(&batch, Some(&keep)).unwrap();
        // (2,1) is dropped (col 1 removed); the rest remap into a 3×3 matrix.
        assert_eq!(n, 3);
        assert_eq!(indptr, vec![0, 2, 2, 3]);
        assert_eq!(indices, vec![0, 1, 2]); // row0: (0,0)->0,(0,2)->1 ; row2: (3,3)->2
        assert_eq!(data, vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn coo_to_csr_out_of_bounds_coord_errors_no_mask() {
        // Malformed COO: a coordinate >= n_rows would index `indptr` out of
        // bounds. The no-mask path must reject it rather than panic.
        let batch = coo(3, &[(0, 0, 1.0), (5, 1, 2.0)]);
        let err = coo_batch_to_csr(&batch, None).unwrap_err();
        assert!(
            err.to_string().contains("out of bounds"),
            "expected out-of-bounds error, got: {err}"
        );
    }

    #[test]
    fn coo_to_csr_rejects_dimension_exceeding_i32() {
        // A pairwise axis wider than i32::MAX cannot be represented in the i32
        // CSR `indices`; reject rather than wrap. Empty triples so the guard
        // fires before the large `indptr` allocation (cheap test).
        let batch = coo(i32::MAX as usize + 1, &[]);
        let err = coo_batch_to_csr(&batch, None).unwrap_err();
        assert!(
            err.to_string().contains("i32"),
            "expected i32 index-limit error, got: {err}"
        );
    }
}

#[cfg(test)]
mod raw_export_filter_tests {
    use super::*;

    /// The raw-export keep mask and raw X share the obs axis, so a length
    /// disagreement means the file is inconsistent — not that the export should
    /// pick one and carry on.
    ///
    /// This helper used to open with `min(mask.len())`, which made the *shorter*
    /// case silent: it exported a raw matrix with fewer rows than obs and
    /// returned no error. That direction never panicked, which is why it needed
    /// a test rather than an assertion — a wrong file looks like a file.
    ///
    /// (`scx-engine::collect::filter_csr_rows` is a separate copy of the same
    /// helper with the same defect, tested separately; this one is private to
    /// the h5ad writer and easy to regress unnoticed.)
    #[test]
    fn raw_export_rejects_a_mask_that_disagrees_with_raw_x() {
        // 3-row raw X.
        let indptr = vec![0i64, 2, 3, 5];
        let indices = vec![0i32, 1, 2, 0, 3];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let short = vec![true];
        let err = filter_csr_rows(&indptr, &indices, &data, &short)
            .expect_err("a short mask must error rather than truncate the export");
        let msg = err.to_string();
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must report both counts: {msg}"
        );

        let long = vec![true, true, true, true];
        assert!(filter_csr_rows(&indptr, &indices, &data, &long).is_err());

        // Control: an exactly-matching mask still filters, so the guard is not
        // rejecting every export.
        let (ip, ix, d, n) = filter_csr_rows(&indptr, &indices, &data, &[true, false, true])
            .expect("a matching mask must still filter");
        assert_eq!(n, 2);
        assert_eq!(ip, vec![0, 2, 4]);
        assert_eq!(ix, vec![0, 1, 0, 3]);
        assert_eq!(d, vec![1.0, 2.0, 4.0, 5.0]);
    }
}
