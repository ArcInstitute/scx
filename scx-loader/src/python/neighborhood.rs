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

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyModule, PyTuple};
use pyo3::wrap_pyfunction;

use crate::neighborhood::{
    batch_plans as batch_plans_impl, build_coord_plans, build_graph_plans, CoordQuery,
    NeighborhoodConfig, NeighborhoodPlans, WeightOrder, DEFAULT_GRAPH_CHUNK_ROWS,
};
use crate::sparse_cellset::SparseCellSetPlan;

use super::*;

fn plan_to_py(py: Python<'_>, plan: SparseCellSetPlan) -> PyResult<Bound<'_, PyTuple>> {
    let SparseCellSetPlan {
        file_ids,
        rows,
        role_tags,
        set_offsets,
    } = plan;
    PyTuple::new(
        py,
        [
            PyArray1::from_vec(py, file_ids).into_any(),
            PyArray1::from_vec(py, rows).into_any(),
            PyArray1::from_vec(py, role_tags).into_any(),
            PyArray1::from_vec(py, set_offsets).into_any(),
        ],
    )
}

fn plans_to_py(py: Python<'_>, plans: Vec<SparseCellSetPlan>) -> PyResult<Bound<'_, PyList>> {
    let items: PyResult<Vec<_>> = plans.into_iter().map(|p| plan_to_py(py, p)).collect();
    PyList::new(py, items?)
}

/// Owned plans plus their centres, returned as `(plans, centers)`.
fn built_to_py(py: Python<'_>, built: NeighborhoodPlans) -> PyResult<Bound<'_, PyTuple>> {
    let plans = plans_to_py(py, built.plans)?;
    let centers = built.centers.into_pyarray(py);
    PyTuple::new(py, [plans.into_any(), centers.into_any()])
}

/// `"desc"` / `"asc"` → [`WeightOrder`], **required whenever `k` is given**.
///
/// A default here is the whole defect: `weight_order="desc"` is right for the
/// default key and silently wrong the moment a caller changes only the key, and
/// `neighborhood_plans_from_graph(exp, "distances", file_id=0, k=8)` then
/// returns each cell's *farthest* stored neighbours. Documenting that is weaker
/// than refusing it, so with `k` the caller states the direction and without
/// `k` there is no ranking for it to mean anything about.
fn resolve_weight_order(k: Option<usize>, order: Option<&str>) -> PyResult<WeightOrder> {
    match (k, order) {
        (_, Some("desc")) => Ok(WeightOrder::Desc),
        (_, Some("asc")) => Ok(WeightOrder::Asc),
        (_, Some(other)) => Err(PyValueError::new_err(format!(
            "weight_order must be 'desc' (an affinity graph such as obsp['connectivities'], \
             where larger means closer) or 'asc' (a distance graph such as obsp['distances'], \
             where larger means farther); got {other:?}"
        ))),
        // No ranking happens, so any answer would be a lie about what ran.
        (None, None) => Ok(WeightOrder::Desc),
        (Some(_), None) => Err(PyValueError::new_err(
            "weight_order is required when k is given: it decides which end of the graph's \
             weights k keeps, and the key's name does not say. Pass weight_order='desc' for an \
             affinity graph (obsp['connectivities'], larger = closer) or 'asc' for a distance \
             graph (obsp['distances'], larger = farther). Without it, k on a distance graph \
             would silently return each cell's FARTHEST neighbours.",
        )),
    }
}

/// Graph-driven neighbourhood plans from `obsp/<key>`.
///
/// `file_id` has **no default** — see the module docs. Nor does `weight_order`
/// when `k` is given: the right answer depends on whether the graph's weights
/// are affinities or distances, and the key's name is the caller's.
#[pyfunction]
#[pyo3(signature = (
    path, key="connectivities", *, file_id, k=None, weight_order=None,
    include_center=true, drop_deleted=true, chunk_rows=DEFAULT_GRAPH_CHUNK_ROWS
))]
#[allow(clippy::too_many_arguments)]
fn _neighborhood_plans_from_graph<'py>(
    py: Python<'py>,
    path: &str,
    key: &str,
    file_id: u32,
    k: Option<usize>,
    weight_order: Option<&str>,
    include_center: bool,
    drop_deleted: bool,
    chunk_rows: u64,
) -> PyResult<Bound<'py, PyTuple>> {
    if let Some(0) = k {
        return Err(PyValueError::new_err("k must be >= 1 or None"));
    }
    let order = resolve_weight_order(k, weight_order)?;
    let p = std::path::Path::new(path);
    let cfg = NeighborhoodConfig {
        include_center,
        file_id,
    };
    // `drop_deleted` goes IN rather than a keep mask coming out: reading the
    // mask here meant a second `ScxReader::open` of the same path, so the mask
    // and the graph were two snapshots that a file replaced between them could
    // put out of step — with the disagreement showing up as a plan naming rows
    // that no longer exist and nothing to catch it.
    let built = py
        .detach(|| build_graph_plans(p, key, drop_deleted, k, order, cfg, chunk_rows))
        .map_err(loader_err_to_py)?;
    built_to_py(py, built)
}

/// Coordinate-driven neighbourhood plans from `obsm/<obsm_key>`.
#[pyfunction]
#[pyo3(signature = (
    path, obsm_key="spatial", *, file_id, k=None, radius=None,
    include_center=true, drop_deleted=true
))]
#[allow(clippy::too_many_arguments)]
fn _neighborhood_plans_from_coords<'py>(
    py: Python<'py>,
    path: &str,
    obsm_key: &str,
    file_id: u32,
    k: Option<usize>,
    radius: Option<f32>,
    include_center: bool,
    drop_deleted: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    // Same refusal, same exception type, as the graph builder's. Left to
    // `plans_from_coords` this came back as a RuntimeError there and a
    // ValueError here for the identical mistake.
    if let Some(0) = k {
        return Err(PyValueError::new_err("k must be >= 1 or None"));
    }
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
    let cfg = NeighborhoodConfig {
        include_center,
        file_id,
    };
    let built = py
        .detach(|| build_coord_plans(p, obsm_key, drop_deleted, query, cfg))
        // Every `ConfigError` out of this entry point is a bad argument or
        // bad input data — the dimensionality cap, a non-finite coordinate, a
        // malformed obsm — so all of them are `ValueError`, matching the
        // `k = 0` refusal above. An earlier version matched on the message
        // text to pick out the dimensionality case, which meant rewording
        // either of the two messages silently reverted the exception type and
        // no test would have noticed.
        .map_err(|e| match e {
            crate::error::LoaderError::ConfigError { reason } => PyValueError::new_err(reason),
            other => loader_err_to_py(other),
        })?;
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
        // Typed numpy views first, falling back to the generic sequence
        // protocol. `extract::<Vec<u32>>` boxes and unboxes one Python scalar
        // per element, which on a whole file's worth of plans is millions of
        // object round-trips under the GIL — and the builders hand back numpy
        // arrays, so the fast path is the normal one.
        let item = item?;
        let plan = extract_plan_fast(&item).or_else(|| extract_plan_generic(&item));
        owned.push(plan.ok_or_else(|| {
            PyValueError::new_err(
                "batch_plans: each plan must be a tuple \
                 (file_ids:u32[], rows:u64[], role_tags:i32[], set_offsets:i64[])",
            )
        })?);
    }
    let batched = py
        .detach(|| batch_plans_impl(&owned, sets_per_batch, shuffle_seed))
        .map_err(loader_err_to_py)?;
    plans_to_py(py, batched)
}

/// Zero-copy numpy views, for the arrays these builders themselves return.
fn extract_plan_fast(item: &Bound<'_, PyAny>) -> Option<SparseCellSetPlan> {
    let t: (
        PyReadonlyArray1<u32>,
        PyReadonlyArray1<u64>,
        PyReadonlyArray1<i32>,
        PyReadonlyArray1<i64>,
    ) = item.extract().ok()?;
    Some(SparseCellSetPlan {
        file_ids: t.0.as_slice().ok()?.to_vec(),
        rows: t.1.as_slice().ok()?.to_vec(),
        role_tags: t.2.as_slice().ok()?.to_vec(),
        set_offsets: t.3.as_slice().ok()?.to_vec(),
    })
}

/// Anything else a caller has: lists, tuples, arrays of another dtype.
fn extract_plan_generic(item: &Bound<'_, PyAny>) -> Option<SparseCellSetPlan> {
    let (file_ids, rows, role_tags, set_offsets) = item
        .extract::<(Vec<u32>, Vec<u64>, Vec<i32>, Vec<i64>)>()
        .ok()?;
    Some(SparseCellSetPlan {
        file_ids,
        rows,
        role_tags,
        set_offsets,
    })
}

/// Register the three plan-builder functions flat on the `pyscx` module.
pub fn register_neighborhood(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_neighborhood_plans_from_graph, m)?)?;
    m.add_function(wrap_pyfunction!(_neighborhood_plans_from_coords, m)?)?;
    m.add_function(wrap_pyfunction!(batch_plans, m)?)?;
    Ok(())
}
