//! The `pyscx.neighborhood_plans_*` entry points: R6's missing plan builders.
//!
//! Each returns a **list of plans** in the exact
//! `(file_ids, rows, role_tags, set_offsets)` shape
//! `SparseCellSetDataset.gather` and `.iter_with_plans` already take, with every
//! array **moved** into numpy (`PyArray1::from_vec` adopts the allocation)
//! rather than copied.
//!
//! These take a path rather than an `Experiment` because `scx-loader` cannot
//! name pyscx's `PyExperiment`. The Python wrappers in `pyscx/__init__.py`
//! coerce a handle to its `.path` through `_coerce_path`, which is the blessed
//! spelling everywhere else (`pyscx.obs_import` and friends).
//!
//! There is no rayon dispatch here — the builders are serial, and the benchmark
//! arms report `plan_build_s` apart from the gather so a measurement, not a
//! guess, is what would change that. That also means no `install(` guard is
//! needed; if a `par_*` is ever added, it needs one within 12 lines, lexically
//! inside the `py.detach` closure.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyModule, PyTuple};
use pyo3::wrap_pyfunction;

use crate::neighborhood::{
    batch_plans as batch_plans_impl, build_coord_plans, build_graph_plans, CoordQuery,
    NeighborhoodConfig, NeighborhoodPlans, DEFAULT_GRAPH_CHUNK_ROWS,
};
use crate::sparse_cellset::SparseCellSetPlan;

use super::*;

/// Physical rows the file's deletion vectors retain, or `None` when the file
/// has none.
///
/// Read here rather than passed in from Python: the keep mask is a property of
/// the file the plans are being built from, and a caller-supplied one that
/// disagreed with it would produce plans naming deleted rows with nothing to
/// catch it.
fn keep_mask(path: &std::path::Path) -> PyResult<Option<Vec<bool>>> {
    let reader = scx_format_io::ScxReader::open(path)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to open {}: {e}", path.display())))?;
    reader
        .deletion_keep_mask()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

fn plan_to_py<'py>(py: Python<'py>, plan: &SparseCellSetPlan) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, plan.file_ids.clone()).into_any(),
            PyArray1::from_vec(py, plan.rows.clone()).into_any(),
            PyArray1::from_vec(py, plan.role_tags.clone()).into_any(),
            PyArray1::from_vec(py, plan.set_offsets.clone()).into_any(),
        ],
    )
}

fn plans_to_py<'py>(py: Python<'py>, plans: &[SparseCellSetPlan]) -> PyResult<Bound<'py, PyList>> {
    let items: PyResult<Vec<_>> = plans.iter().map(|p| plan_to_py(py, p)).collect();
    PyList::new(py, items?)
}

/// Owned plans plus their centres, returned as `(plans, centers)`.
fn built_to_py<'py>(py: Python<'py>, built: NeighborhoodPlans) -> PyResult<Bound<'py, PyTuple>> {
    let plans = plans_to_py(py, &built.plans)?;
    let centers = built.centers.into_pyarray(py);
    PyTuple::new(py, [plans.into_any(), centers.into_any()])
}

/// Graph-driven neighbourhood plans from `obsp/<key>`.
#[pyfunction]
#[pyo3(signature = (
    path, key="connectivities", *, k=None, include_center=true, file_id=0,
    drop_deleted=true, chunk_rows=DEFAULT_GRAPH_CHUNK_ROWS
))]
#[allow(clippy::too_many_arguments)]
fn _neighborhood_plans_from_graph<'py>(
    py: Python<'py>,
    path: &str,
    key: &str,
    k: Option<usize>,
    include_center: bool,
    file_id: u32,
    drop_deleted: bool,
    chunk_rows: u64,
) -> PyResult<Bound<'py, PyTuple>> {
    if let Some(0) = k {
        return Err(PyValueError::new_err("k must be >= 1 or None"));
    }
    let p = std::path::Path::new(path);
    let keep = if drop_deleted { keep_mask(p)? } else { None };
    let cfg = NeighborhoodConfig {
        include_center,
        file_id,
    };
    let built = py
        .detach(|| build_graph_plans(p, key, keep.as_deref(), k, cfg, chunk_rows))
        .map_err(loader_err_to_py)?;
    built_to_py(py, built)
}

/// Coordinate-driven neighbourhood plans from `obsm/<obsm_key>`.
#[pyfunction]
#[pyo3(signature = (
    path, obsm_key="spatial", *, k=None, radius=None, include_center=true,
    file_id=0, drop_deleted=true
))]
#[allow(clippy::too_many_arguments)]
fn _neighborhood_plans_from_coords<'py>(
    py: Python<'py>,
    path: &str,
    obsm_key: &str,
    k: Option<usize>,
    radius: Option<f32>,
    include_center: bool,
    file_id: u32,
    drop_deleted: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    // Exactly one of the two, refused rather than defaulted: "k or radius,
    // whichever you gave" is the kind of resolution that silently answers a
    // different question from the one asked.
    let query =
        match (k, radius) {
            (Some(k), None) => CoordQuery::Knn(k),
            (None, Some(r)) => CoordQuery::Radius(r),
            (None, None) => return Err(PyValueError::new_err(
                "neighborhood_plans_from_coords needs exactly one of k= or radius=; got neither",
            )),
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "neighborhood_plans_from_coords needs exactly one of k= or radius=; got both",
                ))
            }
        };
    let p = std::path::Path::new(path);
    let keep = if drop_deleted { keep_mask(p)? } else { None };
    let cfg = NeighborhoodConfig {
        include_center,
        file_id,
    };
    let built = py
        .detach(|| build_coord_plans(p, obsm_key, keep.as_deref(), query, cfg))
        .map_err(loader_err_to_py)?;
    built_to_py(py, built)
}

/// Concatenate single-set plans into batch plans of `sets_per_batch` sets.
#[pyfunction]
#[pyo3(signature = (plans, sets_per_batch, *, shuffle_seed=None))]
fn batch_plans<'py>(
    py: Python<'py>,
    plans: Bound<'py, PyAny>,
    sets_per_batch: usize,
    shuffle_seed: Option<u64>,
) -> PyResult<Bound<'py, PyList>> {
    let mut owned: Vec<SparseCellSetPlan> = Vec::new();
    for item in plans.try_iter()? {
        let (file_ids, rows, role_tags, set_offsets) = item?
            .extract::<(Vec<u32>, Vec<u64>, Vec<i32>, Vec<i64>)>()
            .map_err(|e| {
                PyValueError::new_err(format!(
                    "batch_plans: each plan must be a tuple \
                     (file_ids:u32[], rows:u64[], role_tags:i32[], set_offsets:i64[]): {e}"
                ))
            })?;
        owned.push(SparseCellSetPlan {
            file_ids,
            rows,
            role_tags,
            set_offsets,
        });
    }
    let batched = py
        .detach(|| batch_plans_impl(&owned, sets_per_batch, shuffle_seed))
        .map_err(loader_err_to_py)?;
    plans_to_py(py, &batched)
}

/// Register the three plan-builder functions flat on the `pyscx` module.
pub fn register_neighborhood(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_neighborhood_plans_from_graph, m)?)?;
    m.add_function(wrap_pyfunction!(_neighborhood_plans_from_coords, m)?)?;
    m.add_function(wrap_pyfunction!(batch_plans, m)?)?;
    Ok(())
}
