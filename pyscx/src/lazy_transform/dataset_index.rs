//! Indexing / comparison helpers for `ScxLazyTransformedDataset` — the
//! private `__getitem__` machinery, split out of `dataset.rs` (ORG-10.16-6).

// ScxLazyTransformedDataset — PyO3 class for lazy per-row transforms.
//
// Extracted from the former pyscx/src/lazy_transform.rs (T5.7).

use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use scx_sparse::ScxCsr;

use crate::backed::detached;
use crate::backed::ScxComparisonResult;
use crate::backed::{is_all_rows_slice, resolve_col_request, scipy_column_gather};
use crate::convert::csr_to_scipy;

use super::*;

impl ScxLazyTransformedDataset {
    /// Handle 1D row indexing (slice, int, bool mask, fancy index).
    pub(crate) fn getitem_rows<'py>(
        &self,
        py: Python<'py>,
        row_idx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Integer index → single row
        if let Ok(i) = row_idx.extract::<i64>() {
            let row = self.normalize_row_index(i)?;
            let global_row = self.to_global_row(row)?;
            // Decode + transform + project off the GIL; build scipy on-GIL.
            let csr = detached(py, || {
                self.backed
                    .read_rows(global_row as u64, global_row as u64 + 1)
                    .map(|mut csr| {
                        self.apply_transforms(&mut csr, global_row);
                        self.apply_col_projection(csr)
                    })
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
            return csr_to_scipy(py, csr);
        }

        // Slice index
        if let Ok(slice) = row_idx.cast::<PySlice>() {
            let indices = slice.indices(self.shape_val.0 as isize)?;
            let start = indices.start.max(0) as u64;
            let stop = indices.stop.max(0) as u64;
            let step = indices.step;

            if step == 1 && self.kept_to_global.is_none() {
                // Contiguous slice, no deletions — direct range read + transform.
                // Decode + transform + project off the GIL; build scipy on-GIL.
                let csr = detached(py, || {
                    self.backed
                        .read_rows(start, stop)
                        .map(|mut csr| {
                            self.apply_transforms(&mut csr, start as usize);
                            self.apply_col_projection(csr)
                        })
                        .map_err(|e| e.to_string())
                })
                .map_err(PyRuntimeError::new_err)?;
                return csr_to_scipy(py, csr);
            }

            // With deletions or non-unit step — expand to individual global indices
            let mut rows = Vec::new();
            let mut i = indices.start;
            while (step > 0 && i < indices.stop) || (step < 0 && i > indices.stop) {
                if i >= 0 && (i as usize) < self.shape_val.0 {
                    rows.push(self.to_global_row(i as usize)? as u64);
                }
                i += step;
            }

            // Decode + per-row transform + project off the GIL; scipy on-GIL.
            let csr = detached(py, || {
                self.backed
                    .read_row_indices(&rows)
                    .map(|mut csr| {
                        // Apply transforms row-by-row with correct global offsets.
                        self.apply_transforms_per_row(&mut csr, &rows);
                        self.apply_col_projection(csr)
                    })
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
            return csr_to_scipy(py, csr);
        }

        // Numpy array or list — the shared resolver (boolean mask or integer
        // array-like), then the bounded gather. `apply_transforms_per_row`
        // looks each output row's parameters up by its *global* id, so it is
        // handed the same request-order `rows` the CSR was assembled from.
        let visible = crate::backed::resolve_row_selector(py, row_idx, self.shape_val.0)?;
        let rows: Vec<u64> = visible
            .iter()
            .map(|&v| self.to_global_row(v).map(|g| g as u64))
            .collect::<PyResult<Vec<u64>>>()?;
        // Decode + per-row transform + project off the GIL; scipy on-GIL.
        let csr = detached(py, || {
            self.backed
                .read_row_indices(&rows)
                .map(|mut csr| {
                    self.apply_transforms_per_row(&mut csr, &rows);
                    self.apply_col_projection(csr)
                })
                .map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
        csr_to_scipy(py, csr)
    }

    /// Handle 2D indexing (rows, cols).
    pub(crate) fn getitem_2d<'py>(
        &self,
        py: Python<'py>,
        row_idx: &Bound<'py, PyAny>,
        col_idx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Check for scalar (int, int) → return float
        let row_is_int = row_idx.extract::<i64>().is_ok();
        let col_is_int = col_idx.extract::<i64>().is_ok();

        if row_is_int && col_is_int {
            let row = self.normalize_row_index(row_idx.extract::<i64>()?)?;
            let col = col_idx.extract::<i64>()?;
            let col = if col < 0 {
                (self.shape_val.1 as i64 + col) as usize
            } else {
                col as usize
            };
            if col >= self.shape_val.1 {
                return Err(PyIndexError::new_err(format!(
                    "column index {} out of range for {} columns",
                    col, self.shape_val.1
                )));
            }
            let global_row = self.to_global_row(row)?;
            let mut csr = self
                .backed
                .read_rows(global_row as u64, global_row as u64 + 1)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms(&mut csr, global_row);

            // When col_projection is active, remap user-visible col to on-disk col
            let lookup_col = if let Some(ref proj) = self.col_projection {
                *proj.get(col).ok_or_else(|| {
                    PyIndexError::new_err(format!(
                        "column index {} out of range for {} projected columns",
                        col,
                        proj.len()
                    ))
                })? as usize
            } else {
                col
            };
            for (i, &idx) in csr.indices.iter().enumerate() {
                if idx as usize == lookup_col {
                    return Ok(csr.data[i].into_pyobject(py)?.into_any());
                }
            }
            return Ok(0.0f32.into_pyobject(py)?.into_any());
        }

        // ── Non-materializing column projection ────────────────────────
        // `X[:, cols]` on a lazy handle. The selector is resolved by the same
        // rules as the backed class, then composed through the current
        // projection. An ascending-unique composition is a projected handle
        // (no decode). This class stores its projection sorted and has no
        // presentation permutation (see `subset_clone`), so a reordered or
        // repeated request materialises the projected *unique* columns and
        // gathers them with scipy — peak is the result, never the whole
        // matrix. `X[:, :]` resolves to `None` and falls through.
        if is_all_rows_slice(row_idx, self.shape_val.0)? {
            if let Some(sel) = resolve_col_request(py, col_idx, self.shape_val.1)? {
                let sel_i64: Vec<i64> = sel.iter().map(|&c| c as i64).collect();
                let composed = crate::axis_align::compose_cols_positional(
                    self.col_projection.as_ref().map(|v| v.as_slice()),
                    &sel_i64,
                    self.shape_val.1,
                )?;
                let ascending = composed.windows(2).all(|w| w[0] < w[1]);
                let mut new_ds = self.clone_handle();
                new_ds.set_col_projection(composed.clone());
                if ascending {
                    return Ok(new_ds.into_pyobject(py)?.into_any());
                }
                let remap: Vec<usize> = {
                    let sorted_unique = new_ds
                        .col_projection()
                        .expect("set_col_projection installed a projection");
                    composed
                        .iter()
                        .map(|c| {
                            sorted_unique
                                .binary_search(c)
                                .expect("every composed column is in its own sorted set")
                        })
                        .collect()
                };
                let mat = new_ds.to_memory(py)?;
                return scipy_column_gather(py, &mat, &remap);
            }
        }

        // Get the full row selection first
        let row_csr = self.getitem_rows(py, row_idx)?;

        // Check if col_idx is a full slice (`:`)
        if let Ok(slice) = col_idx.cast::<PySlice>() {
            let indices = slice.indices(self.shape_val.1 as isize)?;
            if indices.start == 0 && indices.stop == self.shape_val.1 as isize && indices.step == 1
            {
                return Ok(row_csr);
            }
        }

        // Apply column selection
        let builtins = crate::pyimport::import_module(py, "builtins")?;
        let slice_none = builtins.call_method1("slice", (py.None(),))?;
        let col_tuple = PyTuple::new(py, &[slice_none.unbind(), col_idx.clone().unbind()])?;
        row_csr.get_item(col_tuple)
    }

    /// Apply transforms row-by-row for non-contiguous access.
    ///
    /// When rows are fetched via `read_row_indices` (fancy indexing), the resulting
    /// CSR has rows at positions 0..N but they correspond to global rows in `global_rows`.
    /// We need to use the correct global row offset for each row's transform lookup.
    pub(crate) fn apply_transforms_per_row(&self, csr: &mut ScxCsr, global_rows: &[u64]) {
        for (local_row, &global_row) in global_rows.iter().enumerate() {
            let start = csr.indptr[local_row] as usize;
            let end = csr.indptr[local_row + 1] as usize;

            // Detect fused NormalizeTotal + Log1p
            if self.transforms.len() >= 2 {
                if let (
                    Transform::NormalizeTotal {
                        row_sums,
                        target_sum,
                    },
                    Transform::Log1p,
                ) = (&self.transforms[0], &self.transforms[1])
                {
                    let g = global_row as usize;
                    let sum = row_sums[g];
                    if sum > 0.0 {
                        let factor = *target_sum / sum;
                        for v in &mut csr.data[start..end] {
                            *v = ((*v as f64 * factor) as f32).ln_1p();
                        }
                    } else {
                        // Zero-sum row: all stored values must be zero for CSR
                        // from count data. In the unfused path, NormalizeTotal
                        // skips the row and Log1p applies ln(0+1)=0, so both
                        // paths produce identical results when this invariant
                        // holds. Assert to catch upstream data corruption.
                        debug_assert!(
                            csr.data[start..end].iter().all(|&v| v == 0.0),
                            "Fused NormalizeTotal+Log1p: zero-sum row {} has non-zero values",
                            g
                        );
                    }
                    // Apply remaining transforms (index 2+)
                    for transform in &self.transforms[2..] {
                        self.apply_single_row_transform(csr, transform, local_row, g);
                    }
                    continue;
                }
            }

            // General path
            for transform in &self.transforms {
                let g = global_row as usize;
                self.apply_single_row_transform(csr, transform, local_row, g);
            }
        }
    }

    /// Apply a single transform to a single row.
    pub(crate) fn apply_single_row_transform(
        &self,
        csr: &mut ScxCsr,
        transform: &Transform,
        local_row: usize,
        global_row: usize,
    ) {
        let start = csr.indptr[local_row] as usize;
        let end = csr.indptr[local_row + 1] as usize;

        match transform {
            Transform::NormalizeTotal {
                row_sums,
                target_sum,
            } => {
                let sum = row_sums[global_row];
                if sum > 0.0 {
                    let factor = *target_sum / sum;
                    for v in &mut csr.data[start..end] {
                        *v = (*v as f64 * factor) as f32;
                    }
                }
            }
            Transform::Log1p => {
                for v in &mut csr.data[start..end] {
                    *v = v.ln_1p();
                }
            }
            Transform::RowScale { factors } => {
                let factor = factors[global_row] as f32;
                for v in &mut csr.data[start..end] {
                    *v *= factor;
                }
            }
            Transform::Scale { factor } => {
                for v in &mut csr.data[start..end] {
                    *v = (*v as f64 * *factor) as f32;
                }
            }
        }
    }

    /// Create a lazy comparison result wrapper.
    pub(crate) fn make_comparison_result<'py>(
        &self,
        py: Python<'py>,
        op: &str,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // For lazy comparison, we need a materialized view since transforms
        // change values. However, for the (X > 0).sum() pattern, NNZ is
        // preserved so we can still short-circuit.
        if let Ok(threshold) = other.extract::<f64>() {
            let result = ScxComparisonResult::new_for_lazy(
                Arc::clone(&self.backed),
                self.shape_val,
                op.to_string(),
                threshold,
                self.kept_to_global.clone(),
                self.col_projection.clone(),
                self.non_negative,
                self.transforms.clone(),
            );
            Ok(Bound::new(py, result)?.into_any())
        } else {
            // Non-numeric comparison — materialize immediately
            let mat = self.to_memory(py)?;
            let method = match op {
                "gt" => "__gt__",
                "ge" => "__ge__",
                "lt" => "__lt__",
                "le" => "__le__",
                "eq" => "__eq__",
                "ne" => "__ne__",
                _ => "__gt__",
            };
            mat.call_method1(method, (other,))
        }
    }
}
