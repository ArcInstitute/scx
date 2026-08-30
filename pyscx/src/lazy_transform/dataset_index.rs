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

        // Numpy array or list
        let np = crate::pyimport::import_module(py, "numpy")?;
        let arr = np.call_method1("asarray", (row_idx,))?;
        // `dtype.kind` — never `str(dtype)`/`dtype.name`: numpy's C code imports
        // `numpy._core._dtype` on every dtype stringification via the
        // frame-sensitive `PyImport_Import`, which detonates when this
        // __getitem__ is called from restricted-exec globals (no `__import__`).
        // `kind` is a plain C descriptor char. Matches the column-index helper.
        let dtype_kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;

        if dtype_kind == "b" {
            // Boolean mask → extract True indices
            let nonzero = arr.call_method0("nonzero")?;
            let idx_tuple = nonzero.cast::<PyTuple>()?;
            let idx_arr = idx_tuple.get_item(0)?;
            let flat = idx_arr.call_method1("astype", (np.getattr("int64")?,))?;
            let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
            let slice = readonly
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows: Vec<u64> = slice
                .iter()
                .map(|&v| self.to_global_row(v as usize).map(|g| g as u64))
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
            return csr_to_scipy(py, csr);
        }

        // Integer array / list → fancy indexing
        let flat = arr.call_method1("astype", (np.getattr("int64")?,))?;
        let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
        let slice = readonly
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n = self.shape_val.0 as i64;
        let rows: Vec<u64> = slice
            .iter()
            .map(|&v| {
                let normalized = if v < 0 { n + v } else { v };
                if normalized < 0 || normalized >= n {
                    return Err(PyIndexError::new_err(format!(
                        "row index {} out of range for {} rows",
                        v, self.shape_val.0
                    )));
                }
                self.to_global_row(normalized as usize).map(|g| g as u64)
            })
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
        // When row_idx selects ALL rows (`:` or `slice(None)`) and col_idx
        // is an array or boolean mask, return a new ScxLazyTransformedDataset
        // with col_projection set instead of materializing to scipy.
        if self.is_all_rows_slice(py, row_idx)? {
            if let Some(col_indices) = self.extract_col_indices(py, col_idx)? {
                let composed = self.compose_col_projection(&col_indices);
                let new_ds = ScxLazyTransformedDataset::new(
                    Arc::clone(&self.backed),
                    (self.shape_val.0, composed.len()),
                    self.kept_to_global.clone(),
                    Some(Arc::new(composed)),
                    self.transforms.clone(),
                    self.non_negative,
                )
                .with_csc_reader(self.backed_csc.clone())
                .with_source_path(self.source_path.clone());
                return Ok(new_ds.into_pyobject(py)?.into_any().unbind().into_bound(py));
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

    /// Check whether `row_idx` selects all rows (is `slice(None)` / `:`).
    pub(crate) fn is_all_rows_slice(
        &self,
        _py: Python<'_>,
        row_idx: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let Ok(slice) = row_idx.cast::<PySlice>() {
            let indices = slice.indices(self.shape_val.0 as isize)?;
            Ok(
                indices.start == 0
                    && indices.stop == self.shape_val.0 as isize
                    && indices.step == 1,
            )
        } else {
            Ok(false)
        }
    }

    /// Try to extract integer column indices from `col_idx`.
    /// Returns `Some(Vec<u32>)` for ndarray (int or bool), `None` otherwise.
    pub(crate) fn extract_col_indices(
        &self,
        py: Python<'_>,
        col_idx: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let numpy = crate::pyimport::import_module(py, "numpy")?;
        let is_ndarray = col_idx.is_instance(&numpy.getattr("ndarray")?)?;
        if !is_ndarray {
            return Ok(None);
        }

        let dtype_str: String = col_idx.getattr("dtype")?.getattr("kind")?.extract()?;

        match dtype_str.as_str() {
            "b" => {
                let mask: Vec<bool> = col_idx.extract()?;
                if mask.len() != self.shape_val.1 {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "boolean index length {} doesn't match axis 1 size {}",
                        mask.len(),
                        self.shape_val.1,
                    )));
                }
                let indices: Vec<u32> = mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &b)| if b { Some(i as u32) } else { None })
                    .collect();
                Ok(Some(indices))
            }
            // Integer array (signed or unsigned).
            // Only use non-materializing projection for sorted, unique indices.
            // Unsorted or duplicate indices need materialization to preserve
            // user-specified column order/repetition (numpy __getitem__ semantics).
            "i" | "u" => {
                let indices: Vec<i64> = col_idx.extract()?;
                let n = self.shape_val.1 as i64;
                let resolved: Vec<u32> = indices
                    .iter()
                    .map(|&i| {
                        let i = if i < 0 { n + i } else { i };
                        if i < 0 || i >= n {
                            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                                "column index {} out of range for axis of size {}",
                                i, n,
                            )))
                        } else {
                            Ok(i as u32)
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                if !resolved.windows(2).all(|w| w[0] < w[1]) {
                    return Ok(None); // unsorted or duplicates → materialize
                }
                Ok(Some(resolved))
            }
            _ => Ok(None),
        }
    }

    /// Compose new column indices with an existing col_projection.
    ///
    /// `new_indices` are in the user-visible column space (`0..shape_val.1`).
    /// Returns sorted, deduplicated indices in the original on-disk column space
    /// (matching the convention that `col_projection` is always sorted).
    ///
    /// The output is always in on-disk column order regardless of input order,
    /// and duplicate indices in `new_indices` are silently collapsed.
    pub(crate) fn compose_col_projection(&self, new_indices: &[u32]) -> Vec<u32> {
        let mut composed = match &self.col_projection {
            Some(existing) => new_indices.iter().map(|&i| existing[i as usize]).collect(),
            None => new_indices.to_vec(),
        };
        composed.sort_unstable();
        composed.dedup();
        composed
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
