//! The free `#[pyfunction]`s: cell-set collation and count downsampling.
//!
//! Split out of `python.rs` by ORG-9.10-2; a pure move.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use crate::sparse_cellset::{CollateScalars, CollatedCellSetBatch};
use crate::sparse_cellset_collate::PreprocessMode;

use super::*;

/// Convert a `CollatedCellSetBatch` into a dict of flat stacked tensors plus
/// shape scalars (`n_rows`, `k_enc`, `k_dec`); state3 reshapes to `[B, S, K]`.
fn collated_cellset_batch_to_dict<'py>(
    py: Python<'py>,
    batch: CollatedCellSetBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        "encoder_gene_ids",
        PyArray1::from_vec(py, batch.encoder_gene_ids),
    )?;
    dict.set_item(
        "encoder_counts",
        PyArray1::from_vec(py, batch.encoder_counts),
    )?;
    dict.set_item("encoder_mask", PyArray1::from_vec(py, batch.encoder_mask))?;
    dict.set_item(
        "encoder_pad_mask",
        PyArray1::from_vec(py, batch.encoder_pad_mask),
    )?;
    dict.set_item("target_counts", PyArray1::from_vec(py, batch.target_counts))?;
    dict.set_item("library_size", PyArray1::from_vec(py, batch.library_size))?;
    dict.set_item("cell_indices", PyArray1::from_vec(py, batch.cell_indices))?;
    dict.set_item("file_ids", PyArray1::from_vec(py, batch.file_ids))?;
    dict.set_item("set_offsets", PyArray1::from_vec(py, batch.set_offsets))?;
    dict.set_item("role_tags", PyArray1::from_vec(py, batch.role_tags))?;
    dict.set_item("n_rows", batch.n_rows)?;
    dict.set_item("k_enc", batch.k_enc)?;
    dict.set_item("k_dec", batch.k_dec)?;
    Ok(dict)
}

/// Collate an already-gathered, **global-vocab** CSR batch into stacked tensors
/// (state3 "3A hybrid"). Pure compute; releases the GIL. Python gathers (via
/// `iter_with_plans`) and samples the query, then collates here. `set_offsets`
/// delimits the sets; `enc_mask_positions` may be empty (perturbation path).
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, data, set_offsets, cell_indices, file_ids, role_tags,
    k_dec, query_gene_ids, enc_mask_positions, hide_readout, n_measured,
    k_enc, mode, n_genes_total, target_sum=None, lib_size_redef=None,
    pflog_alpha=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn collate_cellset_gathered<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f32>,
    set_offsets: PyReadonlyArray1<'py, i64>,
    cell_indices: PyReadonlyArray1<'py, u64>,
    file_ids: PyReadonlyArray1<'py, u32>,
    role_tags: PyReadonlyArray1<'py, i32>,
    k_dec: usize,
    query_gene_ids: PyReadonlyArray1<'py, i32>,
    enc_mask_positions: PyReadonlyArray1<'py, u8>,
    hide_readout: PyReadonlyArray1<'py, u8>,
    n_measured: PyReadonlyArray1<'py, u32>,
    k_enc: usize,
    mode: String,
    n_genes_total: i64,
    target_sum: Option<f64>,
    lib_size_redef: Option<bool>,
    pflog_alpha: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let mode = PreprocessMode::parse(&mode).map_err(loader_err_to_py)?;
    // v4 PFlog collate mode needs a pinned α (no dataset to estimate from here).
    if mode == PreprocessMode::PflogRaw {
        match pflog_alpha {
            None => {
                return Err(PyValueError::new_err(
                    "collate mode 'pflog_raw' requires pflog_alpha (estimate it once via \
                     pyscx.accel.pflog and pass it here)",
                ));
            }
            Some(a) if a <= 0.0 || !a.is_finite() => {
                return Err(PyValueError::new_err(format!(
                    "pflog_alpha must be positive and finite, got {a}"
                )));
            }
            _ => {}
        }
    }
    let scalars = CollateScalars {
        k_enc,
        mode,
        target_sum: target_sum.unwrap_or(1e4),
        pflog_alpha,
        n_genes_total,
        lib_size_redef: lib_size_redef.unwrap_or(false),
    };
    let err = |e| PyRuntimeError::new_err(format!("array not contiguous: {e}"));
    let indptr = indptr.as_slice().map_err(err)?;
    let indices = indices.as_slice().map_err(err)?;
    let data = data.as_slice().map_err(err)?;
    let set_offsets = set_offsets.as_slice().map_err(err)?;
    let cell_indices_v = cell_indices.as_slice().map_err(err)?.to_vec();
    let file_ids_v = file_ids.as_slice().map_err(err)?.to_vec();
    let role_tags_v = role_tags.as_slice().map_err(err)?.to_vec();
    let query = query_gene_ids.as_slice().map_err(err)?;
    let encmask = enc_mask_positions.as_slice().map_err(err)?;
    let hide = hide_readout.as_slice().map_err(err)?;
    let nmeas = n_measured.as_slice().map_err(err)?;
    let batch = py
        .detach(|| {
            crate::sparse_cellset::collate_gathered(
                indptr,
                indices,
                data,
                set_offsets,
                cell_indices_v,
                file_ids_v,
                role_tags_v,
                k_dec,
                query,
                encmask,
                hide,
                nmeas,
                &scalars,
            )
        })
        .map_err(loader_err_to_py)?;
    collated_cellset_batch_to_dict(py, batch)
}

/// Stable 64-bit RNG-key identity for an `.scx` path.
///
/// Exposed so a caller that gathers its own CSR (and therefore drives
/// [`downsample_counts_csr`] directly) can key on the same identity the dataset
/// path uses, and so a test can assert the two agree.
#[pyfunction]
pub fn downsample_file_identity(path: &str) -> u64 {
    crate::downsample::file_identity(path)
}

/// Seeded per-row count downsample over an already-gathered CSR batch.
///
/// A standalone counterpart to the `downsample_*` kwargs on
/// `SparseCellSetDataset`, for callers that gather their own CSR. Returns a new
/// `(indptr, indices, data)` triple — rows shrink, because counts that sample to
/// zero are pruned, so `indptr` is **not** preserved.
///
/// `rows` and `file_identities` are parallel to the batch's rows and supply the
/// RNG key. `file_identities` are the values [`downsample_file_identity`]
/// returns. Passing an **empty** array keys on `(seed, method, row)` alone —
/// correct for a single-file batch, but ambiguous across files, since two files'
/// row 5 would then share a draw.
///
/// Pure compute; releases the GIL.
#[pyfunction]
#[pyo3(signature = (
    indptr, indices, data, rows, file_identities,
    target_library_size, method=None, seed=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn downsample_counts_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f32>,
    rows: PyReadonlyArray1<'py, u64>,
    file_identities: PyReadonlyArray1<'py, u64>,
    target_library_size: u64,
    method: Option<String>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    if target_library_size == 0 {
        return Err(PyValueError::new_err("target_library_size must be > 0"));
    }
    // `ValueError`, matching the `target_library_size` check directly above and
    // the `SparseCellSetDataset` constructor: an unknown method name is a bad
    // argument value, and a caller catching malformed input should not have to
    // know which of the two downsample entry points it called.
    let method =
        crate::downsample::DownsampleMethod::parse(method.as_deref().unwrap_or("multinomial"))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let err = |e| PyRuntimeError::new_err(format!("array not contiguous: {e}"));
    let indptr = indptr.as_slice().map_err(err)?;
    let indices = indices.as_slice().map_err(err)?;
    let data = data.as_slice().map_err(err)?;
    let rows = rows.as_slice().map_err(err)?;
    let idents = file_identities.as_slice().map_err(err)?;

    let n_rows = indptr.len().saturating_sub(1);
    if rows.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "rows len {} != n_rows {n_rows}",
            rows.len()
        )));
    }
    if !idents.is_empty() && idents.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "file_identities len {} != n_rows {n_rows} (pass an empty array to key on \
             (seed, method, row) alone, which is correct for a single-file batch)",
            idents.len()
        )));
    }
    if indices.len() != data.len() {
        return Err(PyValueError::new_err(format!(
            "indices len {} != data len {}",
            indices.len(),
            data.len()
        )));
    }
    // Not just `last == nnz`: the rayon map below slices `indices[lo..hi]` from
    // these entries, so a non-monotonic or negative `indptr` panics across the FFI
    // boundary instead of erroring. This is a public entry point taking arbitrary
    // numpy arrays.
    //
    // Raised as `ValueError`, not the `loader_err_to_py` default of `RuntimeError`:
    // every other argument check in this function raises `ValueError`, and a caller
    // catching malformed input would otherwise miss exactly this one.
    crate::sparse_cellset::validate_indptr(indptr, data.len())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let cfg = crate::downsample::DownsampleConfig {
        target_library_size,
        method,
        seed: seed.unwrap_or(0),
        // Identities arrive per row here, not per file, so the config's own table
        // stays empty and each row's identity is passed explicitly below.
        file_identities: Vec::new(),
    };

    // Per-row work is independent and each row's key is derived from its own
    // identity, so this is safely parallel; `collect` restores row order before
    // flattening, so the output is byte-identical regardless of scheduling.
    //
    // On the loader's pool, never rayon's global registry: this is a bare
    // `#[pyfunction]`, so a forked DataLoader worker reaches it without ever
    // constructing a dataset and therefore without passing any PID check, and a
    // global-pool dispatch from a forked child hangs forever. See `crate::pool`.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let pool = crate::pool::cpu_pool();
    let out_rows: Vec<(Vec<i32>, Vec<f32>)> = py.detach(|| {
        pool.install(|| {
            (0..n_rows)
                .into_par_iter()
                .map(|r| {
                    let lo = indptr[r] as usize;
                    let hi = indptr[r + 1] as usize;
                    let mut i = indices[lo..hi].to_vec();
                    let mut d = data[lo..hi].to_vec();
                    // No identities supplied ⇒ key on `(seed, method, row)` alone.
                    // Falling back to `r` (the row's position in this batch) would be
                    // worse than useless: the same cell would draw differently
                    // depending on where it landed in the batch, which is exactly the
                    // scheduling dependence the per-row key exists to avoid.
                    let ident = if idents.is_empty() { 0 } else { idents[r] };
                    crate::downsample::downsample_row(&mut i, &mut d, &cfg, ident, rows[r]);
                    (i, d)
                })
                .collect()
        })
    });

    let mut out_indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
    let mut out_indices: Vec<i32> = Vec::new();
    let mut out_data: Vec<f32> = Vec::new();
    out_indptr.push(0);
    for (i, d) in out_rows {
        out_indices.extend_from_slice(&i);
        out_data.extend_from_slice(&d);
        out_indptr.push(out_indices.len() as i64);
    }

    let dict = PyDict::new(py);
    dict.set_item("indptr", PyArray1::from_vec(py, out_indptr))?;
    dict.set_item("indices", PyArray1::from_vec(py, out_indices))?;
    dict.set_item("data", PyArray1::from_vec(py, out_data))?;
    Ok(dict)
}
