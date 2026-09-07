//! Preprocessing bindings — normalize_total, log1p, calculate_qc_metrics.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::{detached, ScxBackedLayerDataset, ScxBackedSparseDataset};
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};
use crate::optional_deps::{import_optional_with_hint, BACKED_ESCAPE_HATCH, EXTRA_SCANPY};
use crate::projected_agg;

// Fusion marker key stashed on `adata.uns` by `normalize_total(device="gpu")`
// so that a subsequent `log1p(device="gpu")` can re-run a fused normalize+log1p
// pass over the ORIGINAL backed source instead of reading the already-
// materialized `adata.X`. See Phase 5 Task 5.4.
//
// Lifecycle contract: the marker is **valid only while X and its row/col
// dimensions are unchanged**. It MUST be cleared by any op that mutates X or
// changes obs/var shape (normalize_total, CPU log1p, filter_*, subset_obs).
// Ops that don't mutate X (PCA, kNN, UMAP, Leiden, Harmony, HVG,
// calculate_qc_metrics) leave it intact. New call sites that mutate X must
// call `clear_gpu_normalize_marker`.
pub(crate) const GPU_NORMALIZE_MARKER_KEY: &str = "__scx_gpu_pending_normalize__";

/// Pop the GPU `normalize_total → log1p` fusion marker off `adata.uns` if
/// present. No-op if absent. Always available (independent of `gpu` feature)
/// so non-GPU builds can clear stale markers left by a prior GPU build.
pub(crate) fn clear_gpu_normalize_marker(adata: &Bound<'_, PyAny>) -> PyResult<()> {
    let uns = adata.getattr("uns")?;
    // dict.pop(key, None) returns None on absence rather than raising KeyError.
    let py = adata.py();
    uns.call_method1("pop", (GPU_NORMALIZE_MARKER_KEY, py.None()))?;
    Ok(())
}

/// Opaque marker stored on `adata.uns` after `normalize_total(device="gpu")`.
///
/// Carries everything needed to rebuild a `LazyShardSource` over the original
/// backed reader and re-run a fused normalize+log1p in a single GPU pass from
/// a later `log1p(device="gpu")` call.
///
/// The marker is consumed by `log1p(device="gpu")` on first hit; it is **not**
/// inspected by any other operation, so users who call `pca()` (or anything
/// else) between the two will get the non-fused (already-materialized) result.
/// Correct, just 1× redundant pass.
#[cfg(feature = "gpu")]
#[pyclass(name = "ScxGpuNormalizeMarker", module = "pyscx")]
pub struct ScxGpuNormalizeMarker {
    backed: Arc<scx_format_io::BackedCsrReader>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    target_sum: f64,
    n_obs: usize,
    n_vars: usize,
}

/// Resolve scanpy's `target_sum=None` semantics: the median of positive
/// per-cell totals over the **visible** cells (matching `sc.pp.normalize_total`,
/// which medians `counts_per_cell[counts_per_cell > 0]`). `row_sums` is in
/// physical-row space; `kept`, when present, maps visible → physical row so the
/// median is taken over user-visible cells only (deletion-correct).
///
/// For an all-zero matrix (no positive totals) scanpy yields NaN; we fall back
/// to `1.0` to avoid a NaN denominator — no cell has counts to scale anyway.
fn median_positive_visible(row_sums: &[f64], kept: Option<&[u64]>) -> f64 {
    let mut vals: Vec<f64> = match kept {
        Some(map) => map
            .iter()
            .map(|&g| row_sums[g as usize])
            .filter(|&s| s > 0.0)
            .collect(),
        None => row_sums.iter().copied().filter(|&s| s > 0.0).collect(),
    };
    if vals.is_empty() {
        return 1.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        0.5 * (vals[n / 2 - 1] + vals[n / 2])
    }
}

/// Resolve a caller `target_sum` (`None` → scanpy median of positive per-cell
/// totals) to a concrete scalar, given the per-cell `row_sums` and the optional
/// visible→physical row map.
fn resolve_target_sum(target_sum: Option<f64>, row_sums: &[f64], kept: Option<&[u64]>) -> f64 {
    target_sum.unwrap_or_else(|| median_positive_visible(row_sums, kept))
}

/// Normalize total counts per cell.
///
/// By default (`device="auto"` or `device="cpu"`), replaces `adata.X` with a
/// lazy wrapper that applies row normalization during `__getitem__`. The row
/// sums are precomputed via streaming and cached in the wrapper.
///
/// With `device="gpu"`, the operation is **eager**: the normalized CSR is
/// streamed through GPU kernels shard-by-shard and materialized into a scipy
/// sparse CSR matrix that replaces `adata.X`. This breaks the lazy chain,
/// which is reported via a `UserWarning`.
///
/// Four cases (CPU path):
/// 1. X is `ScxBackedSparseDataset` → create new `ScxLazyTransformedDataset`
/// 2. X is `ScxLazyTransformedDataset` → append NormalizeTotal transform
/// 3. X is scipy sparse/dense → delegate to `sc.pp.normalize_total()`
///
/// GPU path (requires `pyscx` built with `--features gpu`):
/// * Backed/lazy X → stream through `scx_accel::gpu_preprocess_to_csr`, replace
///   `adata.X` with the materialized scipy CSR, and stash a fusion marker on
///   `adata.uns` so a subsequent `log1p(device="gpu")` can re-run a fused
///   pass over the ORIGINAL source.
/// * Scipy/dense X → defer to the CPU path (delegates to `sc.pp.normalize_total`).
///
/// Args:
///     adata: AnnData object
///     target_sum: Target total counts per cell. `None` (default) uses the
///         **median of positive per-cell totals** — scanpy's `target_sum=None`
///         semantics. A float pins an explicit target.
///         COMPATIBILITY BREAK: the previous default was a hard `1e4`; it is now
///         `None` (scanpy-parity). Pass `target_sum=1e4` to restore the old
///         behavior.
///     device: Device selection — "auto" (default), "cpu", "gpu", or
///         "gpu:N" to target CUDA device N on multi-GPU systems.
#[pyfunction]
#[pyo3(signature = (adata, target_sum=None, device="auto"))]
pub fn normalize_total(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    target_sum: Option<f64>,
    device: &str,
) -> PyResult<()> {
    let _device = validate_device_or_default(device)?;

    // A presentation-ordered backed X (preserve_var_order=True) would have its
    // request-ordered var misaligned against the sorted-projection lazy X this
    // op produces — reject rather than silently mis-order.
    super::prepare_target(py, adata, "normalize_total")?;

    // Any prior fusion marker is now stale: this normalize call supersedes it.
    // The GPU success path will stash a fresh marker; CPU / scipy-fallback
    // paths leave it cleared.
    clear_gpu_normalize_marker(adata)?;

    // Record the planned route. The GPU shard-streaming kernel runs only for
    // backed / lazy X; a scipy/dense X has no GPU kernel and falls back to the
    // CPU path (UnsupportedInputLayout on a GPU host). The CPU path itself is
    // lazy (it wraps X), recorded as cpu_csr.
    let np_eligible = {
        let xp = adata.getattr("X")?;
        xp.cast::<ScxBackedSparseDataset>().is_ok()
            || xp.cast::<ScxLazyTransformedDataset>().is_ok()
    };

    // In-VRAM `device="gpu"` preprocess on an in-memory X hands off to
    // rapids-singlecell. backed/lazy X keeps the native streaming path.
    #[cfg(feature = "gpu")]
    if !np_eligible {
        match super::rapids::decide(py, _device, "normalize_total") {
            super::rapids::RapidsDecision::Rapids(gid) => {
                super::rapids::run(py, adata, "normalize_total", gid, |py, adata| {
                    let kw = super::rapids::kwargs(py);
                    kw.set_item("target_sum", target_sum)?;
                    super::rapids::rsc_fn(py, "pp", "normalize_total")?
                        .call((adata,), Some(&kw))?;
                    Ok(())
                })?;
                return Ok(());
            }
            super::rapids::RapidsDecision::NoRapidsCpu => {
                normalize_total(py, adata, target_sum, "cpu")?;
                return super::rapids::stamp_no_rapids(
                    py,
                    adata,
                    "normalize_total",
                    scx_accel::AccelRoute::CpuCsr,
                );
            }
            super::rapids::RapidsDecision::Native => {}
        }
    }

    // The stamp is rolled back if anything below raises, so a present
    // `uns["scx_accel"]["normalize_total"]` always means the op completed.
    let route = super::route::RouteStamp::write(
        py,
        adata,
        "normalize_total",
        &super::route::simple_exec_info(
            device,
            np_eligible,
            scx_accel::AccelRoute::GpuCsr,
            scx_accel::AccelRoute::CpuCsr,
        ),
    )?;

    #[cfg(feature = "gpu")]
    if let Some(device_id) = _device.gpu_id() {
        return route.settle(gpu_normalize_total(
            py, adata, target_sum, device_id, device,
        ));
    }

    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset — create new lazy wrapper
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();

        // Scanpy compat: sum only over projected (user-visible) genes.
        // After filter_genes(), col_projection restricts to kept genes.
        // Without col_projection, this sums all columns (same as before).
        // Must be in physical-row space because transforms are applied
        // per-shard before deletion vector filtering.
        // Heavy row-sum scan runs off the GIL (`detached`); rebind to a plain
        // `&Self` (Send) so the closure captures `b`, not the `!Send` PyRef.
        let b: &ScxBackedSparseDataset = &backed_ref;
        let all_row_sums = detached(py, || {
            if let Some(cols) = b.col_projection() {
                projected_agg::row_sums_projected(&b.backed, cols).map_err(|e| e.to_string())
            } else {
                b.backed.row_sums().map_err(|e| e.to_string())
            }
        })
        .map_err(PyRuntimeError::new_err)?;

        // Resolve target_sum=None → median of positive visible-cell totals
        // before moving all_row_sums into the transform.
        let target = resolve_target_sum(
            target_sum,
            &all_row_sums,
            backed_ref.kept_to_global.as_deref().map(|v| v.as_slice()),
        );
        let non_negative = backed_ref.non_negative;
        let lazy = ScxLazyTransformedDataset::new(
            Arc::clone(&backed_ref.backed),
            backed_ref.shape_val,
            backed_ref.kept_to_global.clone(),
            // Inherit col_projection: transforms operate on full columns,
            // projection is applied per-shard after transforms
            backed_ref.col_projection_arc(),
            vec![Transform::NormalizeTotal {
                row_sums: Arc::new(all_row_sums),
                target_sum: target,
            }],
            non_negative,
        )
        // Forward CSC sidecar so a later log1p() preserves the
        // CSC-dispatch capability (NormalizeTotal itself is not
        // column-local, but a NormalizeTotal+Log1p chain is — the
        // gate at as_column_source() will reject it for now;
        // carrying the handle costs nothing).
        .with_csc_reader(backed_ref.backed_csc.clone())
        .with_source_path(backed_ref.source_path.clone());
        // Drop the borrow before setattr to avoid RefCell borrow conflict
        drop(backed_ref);
        adata.setattr("X", Bound::new(py, lazy)?)?;
        route.commit();
        return Ok(());
    }

    // Case 2: X is already ScxLazyTransformedDataset — append transform
    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let mut lazy_ref = lazy.borrow_mut();
        // Scanpy compat: sum only over projected (user-visible) genes.
        // row_sums_raw() applies transforms to the full-width shard first (so a
        // prior NormalizeTotal sees the correct denominator), then project_csr
        // restricts to projected genes before summing.
        // Returns a global-length vector (n_obs_global), which is what
        // apply_transforms_to_csr expects (indexes by global row).
        // Heavy streaming scan runs off the GIL (`detached`); rebind to a plain
        // `&Self` (Send) so the closure captures `l`, not the `!Send` PyRefMut.
        // `l`'s immutable borrow ends before the mutable `transforms.push` below.
        let sums = {
            let l: &ScxLazyTransformedDataset = &lazy_ref;
            detached(py, || l.row_sums_raw()).map_err(PyRuntimeError::new_err)?
        };
        // Resolve target_sum=None → median of positive visible-cell totals
        // (over the correct denominator, which already reflects prior transforms).
        let target = resolve_target_sum(
            target_sum,
            &sums,
            lazy_ref.kept_to_global.as_deref().map(|v| v.as_slice()),
        );
        lazy_ref.transforms.push(Transform::NormalizeTotal {
            row_sums: Arc::new(sums),
            target_sum: target,
        });
        route.commit();
        return Ok(());
    }

    // Case 3: X is a regular scipy sparse or dense — delegate to scanpy
    let sc = import_optional_with_hint(
        py,
        "scanpy",
        EXTRA_SCANPY,
        "pyscx.accel.normalize_total()",
        "scanpy",
        Some(BACKED_ESCAPE_HATCH),
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("target_sum", target_sum)?;
    sc.getattr("pp")?
        .call_method("normalize_total", (adata,), Some(&kwargs))?;
    route.commit();
    Ok(())
}

/// Apply log1p (ln(x + 1)) element-wise.
///
/// By default (`device="auto"` or `device="cpu"`), replaces `adata.X` with a
/// lazy wrapper that applies log1p during `__getitem__`. When chained after
/// `normalize_total`, the fused optimization in `ScxLazyTransformedDataset`
/// computes `ln(x * target_sum / row_sum + 1)` in a single pass.
///
/// With `device="gpu"`, the operation is **eager**. Three GPU sub-cases:
/// * If `adata.uns` carries a fusion marker from a preceding
///   `normalize_total(device="gpu")`, re-run a single fused normalize+log1p
///   pass over the original backed source and overwrite `adata.X` with the
///   fused result.
/// * Else if X is still backed/lazy, stream through `gpu_preprocess_to_csr`
///   with log1p only and materialize into scipy CSR.
/// * Else (X is already scipy/dense), GPU dispatch is slower than CPU
///   (H→D + D→H copies dominate log1p's trivial math), so `pyscx` emits a
///   `UserWarning` and falls back to `sc.pp.log1p`.
///
/// CPU path (three cases):
/// 1. X is `ScxBackedSparseDataset` → create new `ScxLazyTransformedDataset`
/// 2. X is `ScxLazyTransformedDataset` → append Log1p transform
/// 3. X is scipy sparse/dense → delegate to `sc.pp.log1p()`
///
/// Args:
///     adata: AnnData object
///     device: Device selection — "auto" (default), "cpu", "gpu", or
///         "gpu:N" to target CUDA device N on multi-GPU systems.
#[pyfunction]
#[pyo3(signature = (adata, device="auto"))]
pub fn log1p(py: Python<'_>, adata: &Bound<'_, PyAny>, device: &str) -> PyResult<()> {
    let _device = validate_device_or_default(device)?;

    // See normalize_total: a presentation-ordered backed X would misalign, and
    // a view must be rebuilt as actual before the first `uns` write below.
    super::prepare_target(py, adata, "log1p")?;

    // Record the planned route. The GPU kernel runs when a normalize+log1p
    // fusion marker is present (re-runs over the ORIGINAL source) or when X is
    // still backed / lazy; a materialized scipy/dense X falls back to CPU
    // (UnsupportedInputLayout on a GPU host). The CPU path is lazy (cpu_csr).
    let log1p_eligible = {
        let has_marker = !adata
            .getattr("uns")?
            .call_method1("get", (GPU_NORMALIZE_MARKER_KEY, py.None()))?
            .is_none();
        let xp = adata.getattr("X")?;
        has_marker
            || xp.cast::<ScxBackedSparseDataset>().is_ok()
            || xp.cast::<ScxLazyTransformedDataset>().is_ok()
    };

    // In-VRAM `device="gpu"` log1p on an in-memory X (no pending fusion marker,
    // not backed/lazy) hands off to rapids-singlecell.
    #[cfg(feature = "gpu")]
    if !log1p_eligible {
        match super::rapids::decide(py, _device, "log1p") {
            super::rapids::RapidsDecision::Rapids(gid) => {
                super::rapids::run(py, adata, "log1p", gid, |py, adata| {
                    super::rapids::rsc_fn(py, "pp", "log1p")?.call1((adata,))?;
                    Ok(())
                })?;
                return Ok(());
            }
            super::rapids::RapidsDecision::NoRapidsCpu => {
                log1p(py, adata, "cpu")?;
                return super::rapids::stamp_no_rapids(
                    py,
                    adata,
                    "log1p",
                    scx_accel::AccelRoute::CpuCsr,
                );
            }
            super::rapids::RapidsDecision::Native => {}
        }
    }

    // Rolled back if anything below raises — see RouteStamp.
    let route = super::route::RouteStamp::write(
        py,
        adata,
        "log1p",
        &super::route::simple_exec_info(
            device,
            log1p_eligible,
            scx_accel::AccelRoute::GpuCsr,
            scx_accel::AccelRoute::CpuCsr,
        ),
    )?;

    #[cfg(feature = "gpu")]
    if let Some(device_id) = _device.gpu_id() {
        return route.settle(gpu_log1p_dispatch(py, adata, device_id, device));
    }

    // CPU log1p invalidates any pending GPU fusion: a later log1p(device="gpu")
    // must NOT replay a fused normalize+log1p over the original source (it
    // would silently overwrite the result of this CPU log1p call).
    clear_gpu_normalize_marker(adata)?;

    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset — create new lazy wrapper with Log1p
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();

        let non_negative = backed_ref.non_negative;
        let lazy = ScxLazyTransformedDataset::new(
            Arc::clone(&backed_ref.backed),
            backed_ref.shape_val,
            backed_ref.kept_to_global.clone(),
            // Inherit col_projection so user-visible shape matches adata.var
            backed_ref.col_projection_arc(),
            vec![Transform::Log1p],
            non_negative,
        )
        // Forward CSC sidecar so the resulting log1p()-only chain
        // remains CSC-dispatchable via as_column_source().
        .with_csc_reader(backed_ref.backed_csc.clone())
        .with_source_path(backed_ref.source_path.clone());
        // Drop the borrow before setattr to avoid RefCell borrow conflict
        drop(backed_ref);
        adata.setattr("X", Bound::new(py, lazy)?)?;
        // Same annotation `sc.pp.log1p` writes (Case 3 below gets it for free
        // by delegating). Without it the SCX-native path is the only one whose
        // log1p is invisible to every downstream `uns["log1p"]` consumer.
        super::util::stamp_uns_log1p(py, adata)?;
        route.commit();
        return Ok(());
    }

    // Case 2: X is already ScxLazyTransformedDataset — append Log1p transform
    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let mut lazy_ref = lazy.borrow_mut();
        lazy_ref.transforms.push(Transform::Log1p);
        drop(lazy_ref);
        super::util::stamp_uns_log1p(py, adata)?;
        route.commit();
        return Ok(());
    }

    // Case 3: X is a regular scipy sparse or dense — delegate to scanpy
    let sc = import_optional_with_hint(
        py,
        "scanpy",
        EXTRA_SCANPY,
        "pyscx.accel.log1p()",
        "scanpy",
        Some(BACKED_ESCAPE_HATCH),
    )?;
    sc.getattr("pp")?.call_method1("log1p", (adata,))?;
    route.commit();
    Ok(())
}

/// Calculate QC metrics natively using streaming aggregation.
///
/// Replacement for `sc.pp.calculate_qc_metrics()` that works on backed and
/// lazy-transformed SCX data without materialization.
///
/// Computes per-cell (obs) and per-gene (var) metrics and writes them to
/// `adata.obs` / `adata.var`, matching scanpy's column names, column order and
/// values.
///
/// **One kernel, every matrix.** A scipy/dense `X` is copied into an owned CSR
/// and run through the same streaming accumulator as a backed or lazy `X`,
/// rather than delegated to `sc.pp.calculate_qc_metrics`. Two implementations
/// behind one name is how the op came to return different column sets depending
/// on what `X` was — the streaming route never wrote
/// `log1p_n_genes_by_counts`, `mean_counts`, `log1p_mean_counts` or
/// `pct_dropout_by_counts`, and only the scanpy route could produce
/// `pct_counts_in_top_<n>_genes`.
///
/// Consequences of that, for a caller who was on the scipy route:
///
/// * `scanpy` is no longer needed for this op on any kind of `X`.
/// * The in-memory matrix is copied (`nnz × 8` bytes; the shared >2 GB warning
///   applies), where handing it to scanpy copied nothing. HVG and `score_genes`
///   already worked this way.
/// * Values are accumulated in f64 rather than scanpy's f32, so they differ in
///   the last digits — and a float64 `X` is narrowed to f32 on the way in, as
///   everywhere else in pyscx.
/// * `pct_counts_<v>` for a cell with no counts is `0.0`, this module's
///   convention, where scanpy gives `NaN`.
/// * Explicitly-stored zeros are dropped from the copy before counting, so
///   `n_genes_by_counts` / `n_cells_by_counts` count real nonzeros as scanpy
///   does (it calls `eliminate_zeros()` on the caller's matrix; we do it on
///   ours, which reaches the same numbers without the side effect).
///
/// Args:
///     adata: AnnData object
///     qc_vars: list of boolean column names in `adata.var` identifying gene
///              subsets (e.g. `["mt"]` for mitochondrial genes)
///     log1p: if True, also add log1p-transformed versions of count metrics
///     inplace: if True (default), write metrics to adata.obs/var;
///              if False, return (obs_df, var_df)
///     layer: read `adata.layers[name]` instead of `adata.X`. Accepts a backed
///            layer handle, a lazy handle, or a scipy/dense matrix. Rejected
///            with `prefer_format="csc"` — the CSC sidecar belongs to `X`.
///     percent_top: 1-indexed positions for `pct_counts_in_top_<n>_genes`,
///            computed in the same shard pass as everything else. Defaults to
///            `None`, deliberately **not** scanpy's `(50, 100, 200, 500)`:
///            that default raises `IndexError` on any file with fewer than 500
///            genes, which is how the old scipy route inherited a crash on
///            narrow data. Positions outside `1..=n_visible` raise a
///            `ValueError` naming the visible width.
///     prefer_format: "csr" (default) or "csc". When "csc", the gene-axis
///         (`total_counts`, `n_cells_by_counts`) accumulators read the CSC
///         sidecar instead of streaming CSR shards. The cell-axis stays
///         on CSR — row aggregations have no CSC win. Capability gate
///         applies (no row deletion vector, no non-column-local
///         transforms; raises on missing CSC sidecar).
///
/// Subsets are packed into a 64-bit per-column bitmask, so up to 64 `qc_vars`
/// share one shard pass; beyond that the row pass repeats once per additional
/// 64 (`ceil(n / 64) + 1` passes total).
///
/// BEHAVIOR CHANGE (bug fix): column projections are now honored on every
/// route. Per-cell totals cover only the visible genes, the gene axis is
/// returned at the visible width, and `qc_vars` masks are read against
/// `adata.var`. Projections reach here from `filter_genes`,
/// `highly_variable_genes(subset=True)`, a gene-selected
/// `to_anndata(backed=True, var_names=[...])`, `X.set_col_projection(...)`, or
/// `adata.X = adata.X[:, cols]`. (`adata[:, mask]` is *not* among them — anndata
/// has no registered view type for a backed SCX `X`, so it raises before
/// reaching any accelerator.)
///
/// What was wrong before depends on the route, and only one of them was silent:
///
/// * `prefer_format="csr"` (default) — the gene axis came back at the
///   underlying (physical) width, so building the `var` frame raised
///   `ValueError: Length mismatch`. Loud; no wrong numbers were returned.
/// * `prefer_format="csc"` — the CSC gene axis was already projection-correct,
///   so the lengths matched and the call **returned silently wrong per-cell
///   numbers**: the row axis was projection-blind, and `qc_vars` masks (visible
///   positions) were applied to the on-disk axis, scoring different genes
///   entirely. Anyone who ran projected QC on the CSC route should re-check
///   published `total_counts` / `pct_counts_*`.
///
/// On a lazy `X` the `qc_vars` subset sums are now taken *through* the
/// transform chain, so `pct_counts_<v>` divides a transformed numerator by a
/// transformed denominator (the numerator used to be pre-transform). This is a
/// silent numeric change with no runtime signal — a pipeline that filters on
/// `pct_counts_mt` after a lazy `normalize_total` will keep and drop different
/// cells than before.
///
/// Results are unchanged for unprojected, untransformed input on the backed and
/// lazy routes; see the one-kernel note above for what moved on a scipy `X`.
///
/// SUPPORTED CONTRACT: `adata.var` row *i* must describe visible column *i*.
/// `set_col_projection` sorts and dedups its input, so passing an unsorted
/// column list and slicing `var` in that same unsorted order leaves the two
/// disagreeing — the gene axis is then labelled wrongly with no error, since
/// the lengths still match. Slice `var` in ascending column order.
#[pyfunction]
#[pyo3(signature = (
    adata,
    qc_vars=None,
    log1p=true,
    inplace=true,
    prefer_format="csr",
    *,
    layer=None,
    percent_top=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn calculate_qc_metrics<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    qc_vars: Option<Vec<String>>,
    log1p: bool,
    inplace: bool,
    prefer_format: &str,
    layer: Option<&str>,
    percent_top: Option<Vec<i64>>,
) -> PyResult<Bound<'py, PyAny>> {
    if !matches!(prefer_format, "csr" | "csc") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'csr' or 'csc'"
        )));
    }

    // Per-gene QC metrics are computed over the sorted projection and written to
    // adata.var; a presentation-ordered backed X would misalign them.
    // Deliberately the no-var-guard prologue: the
    // `reject_presentation_ordered_source` call below guards the matrix this
    // op will actually read, which is `adata.layers[layer]` when `layer=` was
    // given. Keeping the X-only check here made the documented remedy
    // impossible — materialise the layer, and the op still refused because
    // `adata.X` was presentation-ordered, while the matrix it was about to
    // read was fine.
    super::prepare_target_no_var_guard(py, adata, "calculate_qc_metrics")?;

    // Read the source matrix: `adata.layers[name]` when a layer is named,
    // `adata.X` otherwise. Modelled on `select_de_matrix` rather than HVG's
    // resolution, so a typo raises a `ValueError` naming the layer instead of
    // leaking a bare `KeyError` from the mapping.
    let x = match layer {
        Some(name) => adata.getattr("layers")?.get_item(name).map_err(|_| {
            PyValueError::new_err(format!("layer '{name}' not found in adata.layers"))
        })?,
        None => adata.getattr("X")?,
    };

    // The `prepare_target` guard above only saw `adata.X`; a named layer is a
    // different matrix and can carry its own presentation permutation.
    super::reject_presentation_ordered_source(&x, "calculate_qc_metrics")?;

    // Normalize `percent_top` before anything is stamped or computed, so a bad
    // value leaves `uns` exactly as it found it.
    let percent_top = normalize_percent_top(percent_top.as_deref(), source_n_vars(&x)?)?;

    // Record the planned route. QC metrics have no GPU kernel; the only route
    // choice is the gene-axis layout — cpu_csc when prefer_format="csc" (reads
    // the gene-major sidecar), cpu_csr otherwise. This is the route the CSC
    // dispatch gate asserts.
    let qc_route = if prefer_format == "csc" {
        scx_accel::AccelRoute::CpuCsc
    } else {
        scx_accel::AccelRoute::CpuCsr
    };
    // Rolled back if anything below raises — notably the `prefer_format="csc"`
    // rejection on a scipy X, which happens after this point.
    let route = super::route::RouteStamp::write(
        py,
        adata,
        "calculate_qc_metrics",
        &scx_accel::AccelExecutionInfo::new(qc_route, scx_accel::FallbackReason::None),
    )?;

    let qc_vars = qc_vars.unwrap_or_default();

    // Advisories are route-independent (they read `adata.var` only), and run
    // before any rejection below so a user who asked for the impossible still
    // learns their `qc_vars` mask is empty.
    emit_qc_advisories(py, adata, &qc_vars)?;

    // Which kind of matrix are we reading? A backed *layer* is its own pyclass
    // and does not answer to `cast::<ScxBackedSparseDataset>()`, so it needs its
    // own arm — miss it and the handle falls through to the in-memory arm and
    // dies inside `scipy.sparse.csr_matrix(<handle>)`.
    let kind = QcSourceKind::of(&x);

    if prefer_format == "csc" {
        // CSC on a non-SCX matrix has no meaningful interpretation — reject
        // explicitly so the user doesn't think it had any effect.
        if matches!(kind, QcSourceKind::InMemory) {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' requires adata.X to be ScxBackedSparseDataset \
                 or ScxLazyTransformedDataset (got scipy/dense)",
            ));
        }
        // The CSC sidecar belongs to `X`, not to an arbitrary layer, and a
        // layer handle is built with no column source at all — so this would
        // otherwise fail deeper down with a message about a missing sidecar,
        // blaming the file for the caller's kwarg combination. The
        // `BackedLayer` arm catches the same handle reached without the kwarg,
        // via `adata.X = adata.layers["counts"]`.
        if layer.is_some() {
            return Err(PyRuntimeError::new_err(
                "calculate_qc_metrics: prefer_format='csc' does not support layer= \
                 (the CSC sidecar lives on adata.X, not on layers); pass layer=None \
                 or use prefer_format='csr'",
            ));
        }
        // Reached via `adata.X = adata.layers["counts"]`, where `layer` is
        // already None — so the message above would advise passing the value
        // the caller is already using.
        if matches!(kind, QcSourceKind::BackedLayer) {
            return Err(PyRuntimeError::new_err(
                "calculate_qc_metrics: prefer_format='csc' does not support a layer \
                 source (adata.X is a layer handle; the CSC sidecar belongs to the \
                 file's X). Restore adata.X to the base backed dataset, or use \
                 prefer_format='csr'",
            ));
        }
    }

    // --- One streaming kernel, whatever the matrix is ---
    //
    // An in-memory scipy/dense `X` is copied into an owned Rust CSR and fed to
    // the same accumulator the backed and lazy routes use, rather than handed
    // to `sc.pp.calculate_qc_metrics`. Two implementations behind one name is
    // how this op came to return different column sets depending on what `X`
    // was; there is now one.
    let in_memory_csr = match kind {
        QcSourceKind::InMemory => {
            reject_unmaterialized_matrix(py, &x)?;
            Some(canonicalize_for_qc(crate::convert::owned_csr(
                py,
                &x,
                Some("calculate_qc_metrics"),
            )?))
        }
        _ => None,
    };

    let pd = crate::pyimport::import_module(py, "pandas")?;

    // Resolve every qc_var gene subset up front, as one bitmask per visible
    // column, so the row pass below can accumulate all of them alongside
    // total_counts in a single scan (see `qc_row_pass`).
    let qc_masks = resolve_qc_masks(py, adata, &x, &qc_vars)?;

    // Compute per-cell total_counts, n_genes_by_counts and every
    // total_counts_<qc_var> in ONE shard scan. Row-axis aggregations stay CSR
    // regardless of prefer_format — CSC offers no win for row sums (it would
    // require gathering per-shard column contributions back into a row index).
    //
    // Both branches go through the projection-aware kernels rather than the
    // underlying reader directly, so per-cell totals cover exactly the
    // **visible** gene set: reading `b.backed.row_sums()` here would silently
    // include genes hidden by a projection (e.g. after `filter_genes`).
    //
    // Heavy row-axis scans run off the GIL (`detached`); rebind to a plain
    // `&Self` (Send) so the closure captures the ref, not the `!Send` PyRef.
    let ns = percent_top.as_slice();
    let row_stats: RowQcOutputs = match kind {
        QcSourceKind::Backed => {
            let backed = x.extract::<PyRef<ScxBackedSparseDataset>>()?;
            let b: &ScxBackedSparseDataset = &backed;
            detached(py, || {
                RowQcOutputs::accumulate(
                    &qc_masks,
                    b.kept_to_global.as_ref().map(|k| k.as_slice()),
                    ns,
                    |bits, n, top| b.qc_row_pass_raw(bits, n, top),
                )
            })
            .map_err(PyRuntimeError::new_err)?
        }
        QcSourceKind::BackedLayer => {
            let layer_ds = x.extract::<PyRef<ScxBackedLayerDataset>>()?;
            let b: &ScxBackedSparseDataset = &layer_ds.inner;
            detached(py, || {
                RowQcOutputs::accumulate(
                    &qc_masks,
                    b.kept_to_global.as_ref().map(|k| k.as_slice()),
                    ns,
                    |bits, n, top| b.qc_row_pass_raw(bits, n, top),
                )
            })
            .map_err(PyRuntimeError::new_err)?
        }
        QcSourceKind::Lazy => {
            let lazy = x.extract::<PyRef<ScxLazyTransformedDataset>>()?;
            let l: &ScxLazyTransformedDataset = &lazy;
            detached(py, || {
                RowQcOutputs::accumulate(
                    &qc_masks,
                    l.kept_to_global.as_ref().map(|k| k.as_slice()),
                    ns,
                    |bits, n, top| l.streaming_qc_row_pass(bits, n, top),
                )
            })
            .map_err(PyRuntimeError::new_err)?
        }
        QcSourceKind::InMemory => {
            // One "shard": the whole matrix, through the same accumulator. No
            // deletion vector — an in-memory `X` is already the visible rows.
            let csr = in_memory_csr.as_ref().expect("in-memory arm builds a CSR");
            detached(py, || {
                RowQcOutputs::accumulate(&qc_masks, None, ns, |bits, n, top| {
                    let mut out = projected_agg::QcRowStats::zeroed(csr.n_rows(), n, top.len());
                    projected_agg::accumulate_qc_rows_into(csr, 0, bits, top, &mut out);
                    Ok(out)
                })
            })
            .map_err(PyRuntimeError::new_err)?
        }
    };
    let RowQcOutputs {
        total_counts,
        n_genes,
        qc_subset_sums,
        top_fractions,
    } = row_stats;

    // Compute per-gene total_counts and n_cells_by_counts.
    let (gene_total_counts, n_cells): (Vec<f64>, Vec<u32>) = if prefer_format == "csc" {
        // CSC dispatch: capability gate + projected_agg twins.
        compute_gene_axis_csc(&x)?
    } else {
        match kind {
            // The fused kernel resolves projection + keep-mask and returns
            // vectors whose length matches the visible var axis
            // (`shape_val.1`) — in one shard scan rather than one per statistic.
            QcSourceKind::Backed => {
                let backed = x.extract::<PyRef<ScxBackedSparseDataset>>()?;
                let b: &ScxBackedSparseDataset = &backed;
                detached(py, || b.col_sums_and_nnz_raw()).map_err(PyRuntimeError::new_err)?
            }
            QcSourceKind::BackedLayer => {
                let layer_ds = x.extract::<PyRef<ScxBackedLayerDataset>>()?;
                let b: &ScxBackedSparseDataset = &layer_ds.inner;
                detached(py, || b.col_sums_and_nnz_raw()).map_err(PyRuntimeError::new_err)?
            }
            QcSourceKind::Lazy => {
                let lazy = x.extract::<PyRef<ScxLazyTransformedDataset>>()?;
                let l: &ScxLazyTransformedDataset = &lazy;
                detached(py, || l.col_sums_and_nnz_raw()).map_err(PyRuntimeError::new_err)?
            }
            QcSourceKind::InMemory => {
                let csr = in_memory_csr.as_ref().expect("in-memory arm builds a CSR");
                // Same visit order and same left-to-right f64 accumulation as
                // `col_sums_and_nnz_projected`'s inner loop.
                csr.col_sums_and_nnz()
            }
        }
    };

    // The divisor for `mean_counts` / `pct_dropout_by_counts` is the number of
    // rows the gene axis was actually accumulated over — the *visible* count,
    // after any deletion vector — matching scanpy's `x.shape[0]` on the object
    // it was handed. Reading it off the file would be silently wrong on every
    // filtered dataset, and no assertion about the gene axis would notice.
    let n_obs_visible = total_counts.len();

    // Build obs DataFrame. Column order follows scanpy's `describe_obs`, so a
    // frame from `inplace=False` lines up with one from `sc.pp` without a sort.
    let obs_dict = PyDict::new(py);
    if log1p {
        let log_n_genes: Vec<f64> = n_genes.iter().map(|&v| (v as f64).ln_1p()).collect();
        obs_dict.set_item("n_genes_by_counts", numpy::PyArray::from_vec(py, n_genes))?;
        obs_dict.set_item(
            "log1p_n_genes_by_counts",
            numpy::PyArray::from_vec(py, log_n_genes),
        )?;
    } else {
        obs_dict.set_item("n_genes_by_counts", numpy::PyArray::from_vec(py, n_genes))?;
    }
    obs_dict.set_item(
        "total_counts",
        numpy::PyArray::from_vec(py, total_counts.clone()),
    )?;
    if log1p {
        let log_total: Vec<f64> = total_counts.iter().map(|&v| v.ln_1p()).collect();
        obs_dict.set_item(
            "log1p_total_counts",
            numpy::PyArray::from_vec(py, log_total),
        )?;
    }
    // `percent_top` is normalized ascending, so these emit in scanpy's order.
    for (&n, fractions) in percent_top.iter().zip(top_fractions) {
        let pct: Vec<f64> = fractions.into_iter().map(|f| f * 100.0).collect();
        obs_dict.set_item(
            format!("pct_counts_in_top_{n}_genes"),
            numpy::PyArray::from_vec(py, pct),
        )?;
    }

    // Build var DataFrame, in scanpy's `describe_var` column order.
    //
    // `mean_counts` and `pct_dropout_by_counts` cost nothing here — they are
    // the gene totals and cell counts the pass already produced, divided by the
    // visible row count — but they were absent for as long as this route
    // existed, which is most of what made the two schemas differ.
    let var_dict = PyDict::new(py);
    let n_obs_f = n_obs_visible as f64;
    let mean_counts: Vec<f64> = gene_total_counts
        .iter()
        .map(|&v| if n_obs_visible == 0 { 0.0 } else { v / n_obs_f })
        .collect();
    // Widened from the kernel's u32: scanpy publishes a signed integer, and an
    // unsigned column turns the natural `n_cells_by_counts - k` into a wrap to
    // ~4e9 rather than a negative number.
    let n_cells_i64: Vec<i64> = n_cells.iter().map(|&c| c as i64).collect();
    let pct_dropout: Vec<f64> = n_cells
        .iter()
        .map(|&c| {
            if n_obs_visible == 0 {
                0.0
            } else {
                (1.0 - c as f64 / n_obs_f) * 100.0
            }
        })
        .collect();

    var_dict.set_item(
        "n_cells_by_counts",
        numpy::PyArray::from_vec(py, n_cells_i64),
    )?;
    var_dict.set_item(
        "mean_counts",
        numpy::PyArray::from_vec(py, mean_counts.clone()),
    )?;
    if log1p {
        let log_mean: Vec<f64> = mean_counts.iter().map(|&v| v.ln_1p()).collect();
        var_dict.set_item("log1p_mean_counts", numpy::PyArray::from_vec(py, log_mean))?;
    }
    var_dict.set_item(
        "pct_dropout_by_counts",
        numpy::PyArray::from_vec(py, pct_dropout),
    )?;
    var_dict.set_item(
        "total_counts",
        numpy::PyArray::from_vec(py, gene_total_counts.clone()),
    )?;
    if log1p {
        let log_gene_total: Vec<f64> = gene_total_counts.iter().map(|&v| v.ln_1p()).collect();
        var_dict.set_item(
            "log1p_total_counts",
            numpy::PyArray::from_vec(py, log_gene_total),
        )?;
    }

    // Publish the qc_var metrics. The subset sums were accumulated by the row
    // pass above — no further shard scans here. An empty mask contributes an
    // all-zero row (its advisory was emitted upfront by `emit_qc_advisories`)
    // and needs no special case: it publishes the same three columns as any
    // other mask, all zero.
    //
    // It used to omit `log1p_total_counts_<v>` for an empty mask, so the column
    // set depended on whether any gene happened to match rather than on the
    // call. Column order is scanpy's: total, log1p_total, pct.
    for (qc_var, subset_sums) in qc_vars.iter().zip(qc_subset_sums) {
        // pct_counts = subset_sum / total * 100
        let pct_counts: Vec<f64> = subset_sums
            .iter()
            .zip(total_counts.iter())
            .map(|(&s, &t)| if t > 0.0 { s / t * 100.0 } else { 0.0 })
            .collect();

        // Compute log1p before moving subset_sums.
        let log1p_subset_sums: Option<Vec<f64>> = if log1p {
            Some(subset_sums.iter().map(|&v| v.ln_1p()).collect())
        } else {
            None
        };

        obs_dict.set_item(
            format!("total_counts_{qc_var}"),
            numpy::PyArray::from_vec(py, subset_sums),
        )?;
        if let Some(log1p_sums) = log1p_subset_sums {
            obs_dict.set_item(
                format!("log1p_total_counts_{qc_var}"),
                numpy::PyArray::from_vec(py, log1p_sums),
            )?;
        }
        obs_dict.set_item(
            format!("pct_counts_{qc_var}"),
            numpy::PyArray::from_vec(py, pct_counts),
        )?;
    }

    // Create DataFrames
    let obs_index = adata.getattr("obs")?.getattr("index")?;
    let var_index = adata.getattr("var")?.getattr("index")?;
    let obs_df = pd.call_method1("DataFrame", (obs_dict,))?;
    obs_df.setattr("index", &obs_index)?;
    let var_df = pd.call_method1("DataFrame", (var_dict,))?;
    var_df.setattr("index", &var_index)?;

    if inplace {
        // Write columns to adata.obs / adata.var
        let adata_obs = adata.getattr("obs")?;
        let adata_var = adata.getattr("var")?;
        // Iterate obs_df columns
        let obs_columns: Vec<String> = obs_df
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        for col in &obs_columns {
            let values = obs_df.get_item(col.as_str())?;
            adata_obs.set_item(col.as_str(), &values)?;
        }
        let var_columns: Vec<String> = var_df
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        for col in &var_columns {
            let values = var_df.get_item(col.as_str())?;
            adata_var.set_item(col.as_str(), &values)?;
        }
        route.commit();
        Ok(py.None().into_bound(py))
    } else {
        // Return (obs_df, var_df) tuple
        let tuple = pyo3::types::PyTuple::new(py, &[obs_df, var_df])?;
        route.commit();
        Ok(tuple.into_any())
    }
}

// ---------------------------------------------------------------------------
// Device resolution + GPU eager dispatch (Phase 5)
// ---------------------------------------------------------------------------

/// Parse `device` into a [`ResolvedDevice`]. Delegates to [`resolve_device`]
/// in both feature configurations — the CPU-only build's parser already
/// rejects `gpu*` requests with a `RuntimeError`, so no separate
/// implementation is needed.
fn validate_device_or_default(device: &str) -> PyResult<super::gpu::ResolvedDevice> {
    super::gpu::resolve_device(device)
}

#[cfg(feature = "gpu")]
fn emit_laziness_break_warning(py: Python<'_>) -> PyResult<()> {
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    warnings.call_method1(
        "warn",
        (
            "pyscx preprocessing device='gpu' is eager: adata.X will be \
             materialized as scipy.sparse.csr_matrix, breaking the lazy chain.",
            py.get_type::<pyo3::exceptions::PyUserWarning>(),
        ),
    )?;
    Ok(())
}

#[cfg(feature = "gpu")]
fn emit_gpu_log1p_fallback_warning(py: Python<'_>, device: &str) -> PyResult<()> {
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    warnings.call_method1(
        "warn",
        (
            format!(
                "pyscx.accel.log1p(device={device:?}) on a materialized scipy/dense \
                 X falls back to CPU: H→D and D→H copies dominate log1p's trivial \
                 math, making GPU dispatch 50–100× slower than scanpy.pp.log1p. The \
                 GPU fast path requires X to be ScxBackedSparseDataset or \
                 ScxLazyTransformedDataset at the time log1p runs — once an \
                 intermediate step (.copy(), adata[:, mask].copy(), etc.) materialises \
                 X to scipy CSR, the fast path is unreachable for the rest of the \
                 pipeline. To enable it, open via pyscx.open(...).to_anndata(backed=True) \
                 and avoid materialising between normalize_total and log1p."
            ),
            py.get_type::<pyo3::exceptions::PyUserWarning>(),
        ),
    )?;
    Ok(())
}

/// Symmetric helper for `gpu_normalize_total`'s scipy/dense fallback. Mirrors
/// `emit_gpu_log1p_fallback_warning`'s shape so a docs-skimmer reading the two
/// warnings side-by-side sees the same "open backed" recommendation.
#[cfg(feature = "gpu")]
fn emit_gpu_normalize_fallback_warning(py: Python<'_>, device: &str) -> PyResult<()> {
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    warnings.call_method1(
        "warn",
        (
            format!(
                "pyscx.accel.normalize_total(device={device:?}) on a materialized \
                 scipy/dense X falls back to CPU (scanpy.pp.normalize_total); the GPU \
                 shard-streaming path requires an ScxBackedSparseDataset or \
                 ScxLazyTransformedDataset. To get the GPU fast path (including the \
                 normalize+log1p fusion marker on adata.uns), open via \
                 pyscx.open(...).to_anndata(backed=True) or keep the source as \
                 ScxLazyTransformedDataset until after log1p."
            ),
            py.get_type::<pyo3::exceptions::PyUserWarning>(),
        ),
    )?;
    Ok(())
}

/// Build a scipy `csr_matrix((data, indices, indptr), shape=...)` on the Python side.
#[cfg(feature = "gpu")]
fn scx_csr_to_scipy<'py>(py: Python<'py>, csr: scx_sparse::ScxCsr) -> PyResult<Bound<'py, PyAny>> {
    let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
    let data = numpy::PyArray::from_vec(py, csr.data);
    let indices = numpy::PyArray::from_vec(py, csr.indices);
    let indptr = numpy::PyArray::from_vec(py, csr.indptr);
    let args = pyo3::types::PyTuple::new(
        py,
        &[data.into_any(), indices.into_any(), indptr.into_any()],
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("shape", csr.shape)?;
    scipy_sparse.call_method("csr_matrix", (args,), Some(&kwargs))
}

/// Everything `gpu_normalize_total` needs from an AnnData's `X`: a live
/// `LazyShardSource` plus the raw handles required to rebuild an equivalent
/// source later (fusion marker). Returned by [`source_from_x`].
#[cfg(feature = "gpu")]
struct GpuShardSource {
    source: crate::lazy_transform::LazyShardSource,
    backed: Arc<scx_format_io::BackedCsrReader>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    n_obs: usize,
    n_vars: usize,
}

/// Extract a [`GpuShardSource`] from either a backed or lazy X. Returns
/// `None` when `x` is neither (i.e. scipy/dense), signalling that the caller
/// should fall back to the CPU path.
#[cfg(feature = "gpu")]
fn source_from_x(x: &Bound<'_, PyAny>) -> PyResult<Option<GpuShardSource>> {
    use crate::lazy_transform::LazyShardSource;

    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let r = backed.borrow();
        let backed_arc = Arc::clone(&r.backed);
        let kept = r.kept_to_global.clone();
        let col_proj = r.col_projection_arc();
        let (n_obs, n_vars) = r.shape_val;
        let source = LazyShardSource::new(
            Arc::clone(&backed_arc),
            vec![],
            kept.clone(),
            col_proj.clone(),
            n_obs,
            n_vars,
        );
        return Ok(Some(GpuShardSource {
            source,
            backed: backed_arc,
            kept_to_global: kept,
            col_projection: col_proj,
            n_obs,
            n_vars,
        }));
    }

    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let r = lazy.borrow();
        let source = r.as_shard_source();
        let backed_arc = Arc::clone(&r.backed);
        let kept = r.kept_to_global.clone();
        let col_proj = r.col_projection.clone();
        let (n_obs, n_vars) = r.shape_val;
        return Ok(Some(GpuShardSource {
            source,
            backed: backed_arc,
            kept_to_global: kept,
            col_projection: col_proj,
            n_obs,
            n_vars,
        }));
    }

    Ok(None)
}

/// GPU-eager `normalize_total` dispatch.
#[cfg(feature = "gpu")]
fn gpu_normalize_total(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    target_sum: Option<f64>,
    device_id: usize,
    device: &str,
) -> PyResult<()> {
    let x = adata.getattr("X")?;

    let Some(gs) = source_from_x(&x)? else {
        // Scipy/dense X: fall back to CPU path (scanpy's normalize_total).
        // Emit a UserWarning symmetric with `gpu_log1p_dispatch`'s scipy/dense
        // fallback so the silent-fallback regression that the 2026-05-21
        // dogfood report flagged (B4) cannot recur.
        emit_gpu_normalize_fallback_warning(py, device)?;
        let sc = import_optional_with_hint(
            py,
            "scanpy",
            EXTRA_SCANPY,
            "pyscx.accel.normalize_total()",
            "scanpy",
            Some(BACKED_ESCAPE_HATCH),
        )?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("target_sum", target_sum)?;
        sc.getattr("pp")?
            .call_method("normalize_total", (adata,), Some(&kwargs))?;
        return Ok(());
    };

    let was_lazy = x.cast::<ScxLazyTransformedDataset>().is_ok();
    let GpuShardSource {
        source,
        backed,
        kept_to_global,
        col_projection,
        n_obs,
        n_vars,
    } = gs;

    // Resolve target_sum=None → median of positive visible-cell totals. The GPU
    // kernel needs a concrete scalar, so when the caller left it to the default
    // we compute host-side row sums off the GIL (mirrors the CPU backed path).
    // Clones move into the closure so the originals stay live for the marker.
    let resolved_target = match target_sum {
        Some(t) => t,
        None => {
            let cols_opt = col_projection.clone();
            let backed_for_sums = Arc::clone(&backed);
            let row_sums = detached(py, move || match cols_opt.as_deref() {
                Some(cols) => projected_agg::row_sums_projected(&backed_for_sums, cols)
                    .map_err(|e| e.to_string()),
                None => backed_for_sums.row_sums().map_err(|e| e.to_string()),
            })
            .map_err(PyRuntimeError::new_err)?;
            median_positive_visible(&row_sums, kept_to_global.as_deref().map(|v| v.as_slice()))
        }
    };

    let dev = scx_accel::GpuDevice::new(device_id)
        .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
    let csr =
        scx_accel::gpu_preprocess_to_csr(&dev, &source, Some(resolved_target as f32), false, None)
            .map_err(|e| PyRuntimeError::new_err(format!("gpu_preprocess_to_csr: {e}")))?;
    drop(source);

    // Replace adata.X with materialized scipy CSR.
    let scipy_csr = scx_csr_to_scipy(py, csr)?;
    adata.setattr("X", scipy_csr)?;

    // Stash fusion marker on adata.uns so a subsequent log1p(device="gpu")
    // can re-run a fused normalize+log1p pass over the original source.
    let marker = ScxGpuNormalizeMarker {
        backed,
        kept_to_global,
        col_projection,
        target_sum: resolved_target,
        n_obs,
        n_vars,
    };
    let uns = adata.getattr("uns")?;
    uns.set_item(GPU_NORMALIZE_MARKER_KEY, Bound::new(py, marker)?)?;

    if was_lazy {
        emit_laziness_break_warning(py)?;
    }
    Ok(())
}

/// GPU-eager `log1p` dispatch — handles the fusion-marker path and backed/lazy
/// materialization. Materialised scipy/dense X warns and falls back to
/// `sc.pp.log1p` (the H→D / D→H round-trip dominates the kernel cost).
#[cfg(feature = "gpu")]
fn gpu_log1p_dispatch(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    device_id: usize,
    device: &str,
) -> PyResult<()> {
    let uns = adata.getattr("uns")?;

    // 1) Fusion marker path: re-run fused normalize+log1p over ORIGINAL source.
    let marker_obj = uns.call_method1("get", (GPU_NORMALIZE_MARKER_KEY,))?;
    if !marker_obj.is_none() {
        if let Ok(marker_ref) = marker_obj.cast::<ScxGpuNormalizeMarker>() {
            let m = marker_ref.borrow();
            let source = crate::lazy_transform::LazyShardSource::new(
                Arc::clone(&m.backed),
                vec![],
                m.kept_to_global.clone(),
                m.col_projection.clone(),
                m.n_obs,
                m.n_vars,
            );
            let target_sum = m.target_sum;
            drop(m);

            let dev = scx_accel::GpuDevice::new(device_id)
                .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
            let csr = scx_accel::gpu_preprocess_to_csr(
                &dev,
                &source,
                Some(target_sum as f32),
                true,
                None,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("gpu_preprocess_to_csr: {e}")))?;
            drop(source);

            let scipy_csr = scx_csr_to_scipy(py, csr)?;
            adata.setattr("X", scipy_csr)?;
            super::util::stamp_uns_log1p(py, adata)?;

            // Invalidate marker after consuming it.
            uns.call_method1("pop", (GPU_NORMALIZE_MARKER_KEY,))?;
            return Ok(());
        }
    }

    // Reaching here means Path 1 was not consumed (marker absent or downcast
    // failed). Any leftover marker is now stale — defensively clear before the
    // remaining paths mutate X.
    clear_gpu_normalize_marker(adata)?;

    // 2) Backed/lazy X: stream through gpu_preprocess_to_csr with log1p only.
    let x = adata.getattr("X")?;
    if let Some(gs) = source_from_x(&x)? {
        let was_lazy = x.cast::<ScxLazyTransformedDataset>().is_ok();
        let dev = scx_accel::GpuDevice::new(device_id)
            .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;
        let csr = scx_accel::gpu_preprocess_to_csr(&dev, &gs.source, None, true, None)
            .map_err(|e| PyRuntimeError::new_err(format!("gpu_preprocess_to_csr: {e}")))?;
        drop(gs);
        let scipy_csr = scx_csr_to_scipy(py, csr)?;
        adata.setattr("X", scipy_csr)?;
        super::util::stamp_uns_log1p(py, adata)?;
        if was_lazy {
            emit_laziness_break_warning(py)?;
        }
        return Ok(());
    }

    // 3) Scipy/dense X: GPU dispatch is slower than CPU here (H→D + D→H
    //    copies dominate log1p's trivial math). Warn and fall back to scanpy.
    emit_gpu_log1p_fallback_warning(py, device)?;
    let sc = import_optional_with_hint(
        py,
        "scanpy",
        EXTRA_SCANPY,
        "pyscx.accel.log1p()",
        "scanpy",
        Some(BACKED_ESCAPE_HATCH),
    )?;
    sc.getattr("pp")?.call_method1("log1p", (adata,))?;
    Ok(())
}

/// Return whether `adata.var[qc_var]` is present and has any True entries.
/// `Some(true)` = mask has ≥1 True; `Some(false)` = mask is all False;
/// `None` = column absent or values not coercible to a Python bool array.
fn qc_var_mask_has_true(adata: &Bound<'_, PyAny>, qc_var: &str) -> Option<bool> {
    let var = adata.getattr("var").ok()?;
    let has_col = var
        .call_method1("__contains__", (qc_var,))
        .ok()
        .and_then(|v| v.is_truthy().ok())
        .unwrap_or(false);
    if !has_col {
        return None;
    }
    let series = var.get_item(qc_var).ok()?;
    // `Series.any()` returns a numpy.bool_ scalar. Extract via Python truthiness.
    let any_result = series.call_method0("any").ok()?;
    any_result.is_truthy().ok()
}

/// Emit qc-vars-related advisories before either the SCX-backed loop or
/// the scanpy fallback runs. Covers three cases:
///   * **F2-2026-05-19**: `qc_vars` is None AND adata.var_names contains
///     5+ MT-/mt- prefixed symbols. Without `qc_vars`, `pct_counts_mt`
///     is never computed and a downstream scanpy filter raises KeyError.
///   * **N1-2026-05-21-Tier2**: `qc_vars` is None AND var_names lacks MT
///     prefixes (e.g. CELLxGENE Census integer-string indexes) but
///     `var['feature_name']` has 5+ MT-/mt- prefixed symbols. Same
///     downstream symptom; the message points at feature_name as the
///     correct source. The two qc_vars=None branches are mutually
///     exclusive (else-arm) so var_names wins when both axes have MT.
///   * **B1-2026-05-20-Tier2 options 1+2**: a user-supplied qc_var mask
///     matches zero genes — `pct_counts_{qc_var}` is silently filled with
///     zeros. For `qc_var == "mt"`, when adata.var['feature_name'] has
///     MT-prefixed symbols (the CELLxGENE Census layout signature), emit
///     a CELLxGENE-specific message pointing at `feature_name` as the
///     right source; otherwise emit a generic empty-mask message.
fn emit_qc_advisories(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    qc_vars: &[String],
) -> PyResult<()> {
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let builtins = crate::pyimport::import_module(py, "builtins")?;
    let user_warning = builtins.getattr("UserWarning")?;

    // The MT-prefix counts cost a pandas
    // `.str.startswith(...).sum()` scan each — small but non-trivial on
    // 60k+ genes. Compute lazily and memoize so the common path (valid
    // non-empty `qc_vars=["mt"]` mask) does zero scans, and the warning
    // paths do at most one scan per axis.
    let mut n_mt_var_names: Option<usize> = None;
    let mut n_mt_feature_name: Option<usize> = None;

    // F2-2026-05-19: qc_vars=None + MT genes visible in var_names.
    // N1-2026-05-21-Tier2: extend to CELLxGENE Census layout (integer-
    // string var_names, MT symbols in var['feature_name']) via an
    // `else if` arm — var_names wins when both axes have MT prefixes so
    // the existing test_warns_when_mt_genes_present_and_qc_vars_none
    // fixture (no feature_name column) keeps its message verbatim.
    if qc_vars.is_empty() {
        let n_vn = *n_mt_var_names.get_or_insert_with(|| count_mt_prefix_in_var_names(adata));
        if n_vn >= 5 {
            let msg = format!(
                "calculate_qc_metrics: found {n_vn} MT-/mt- prefixed gene \
                 symbols in adata.var_names but `qc_vars` is None, so \
                 `pct_counts_mt` will NOT be computed. To compute it, run:\n  \
                 adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')\n  \
                 pyscx.accel.calculate_qc_metrics(adata, qc_vars=['mt'])"
            );
            warnings.call_method1("warn", (msg, &user_warning))?;
        } else {
            let n_fn =
                *n_mt_feature_name.get_or_insert_with(|| count_mt_prefix_in_feature_name(adata));
            if n_fn >= 5 {
                let msg = format!(
                    "calculate_qc_metrics: `qc_vars` is None but \
                     adata.var['feature_name'] has {n_fn} MT-/mt- prefixed \
                     gene symbols (CELLxGENE Census layout: integer-string \
                     var_names, gene symbols in var['feature_name']). \
                     pct_counts_mt will NOT be computed. To compute it, run:\n  \
                     adata.var['mt'] = adata.var['feature_name'].str.upper().str.startswith('MT-')\n  \
                     pyscx.accel.calculate_qc_metrics(adata, qc_vars=['mt'])"
                );
                warnings.call_method1("warn", (msg, &user_warning))?;
            }
        }
    }

    // `qc_var_mask_has_true` returns None when the column is absent or
    // not introspectable — in either case we let the downstream code
    // raise its own error rather than guessing.
    for qc_var in qc_vars {
        if qc_var_mask_has_true(adata, qc_var) != Some(false) {
            continue;
        }
        let is_mt = qc_var.eq_ignore_ascii_case("mt");
        // CELLxGENE-shape branch only reachable when `is_mt` — gate the
        // count materialisation on that so non-mt qc_vars don't pay the
        // scan cost.
        let msg = if is_mt {
            let n_vn = *n_mt_var_names.get_or_insert_with(|| count_mt_prefix_in_var_names(adata));
            let n_fn =
                *n_mt_feature_name.get_or_insert_with(|| count_mt_prefix_in_feature_name(adata));
            if n_vn < 5 && n_fn >= 5 {
                format!(
                    "calculate_qc_metrics: adata.var['mt'] mask matches 0 \
                     genes, but adata.var['feature_name'] has \
                     {n_fn} MT-/mt- prefixed symbols — the \
                     CELLxGENE Census layout (integer-string var_names, \
                     gene symbols in var['feature_name']). \
                     pct_counts_mt will be 0 for every cell. Tag mt from \
                     feature_name instead:\n  \
                     adata.var['mt'] = adata.var['feature_name'].str.upper().str.startswith('MT-')\n  \
                     pyscx.accel.calculate_qc_metrics(adata, qc_vars=['mt'])"
                )
            } else {
                format!(
                    "calculate_qc_metrics: qc_var '{qc_var}' mask matches 0 \
                     genes in adata.var['{qc_var}']; total_counts_{qc_var} \
                     and pct_counts_{qc_var} will be 0 for every cell. \
                     Check that adata.var['{qc_var}'] is populated correctly \
                     — e.g. for CELLxGENE Census data, gene symbols live in \
                     adata.var['feature_name'], not in adata.var_names, so \
                     tag as `adata.var['{qc_var}'] = \
                     adata.var['feature_name'].str.upper().str.startswith('MT-')`."
                )
            }
        } else {
            format!(
                "calculate_qc_metrics: qc_var '{qc_var}' mask matches 0 \
                 genes in adata.var['{qc_var}']; total_counts_{qc_var} \
                 and pct_counts_{qc_var} will be 0 for every cell. \
                 Check that adata.var['{qc_var}'] is populated correctly."
            )
        };
        warnings.call_method1("warn", (msg, &user_warning))?;
    }

    Ok(())
}

/// Count gene symbols starting with `MT-` or `mt-` in `adata.var_names`.
/// Returns 0 on any access error (var_names missing, accessor failure,
/// etc.) so the caller can treat 0 as "no MT-tagged genes".
///
/// Uses the pandas `.str.startswith(...).sum()` vectorised accessor so
/// no per-name Python→Rust string copy is needed — the prefix scan runs
/// in C inside pandas, returning a `numpy.int64` scalar.
fn count_mt_prefix_in_var_names(adata: &Bound<'_, PyAny>) -> usize {
    let Ok(var_names) = adata.getattr("var_names") else {
        return 0;
    };
    var_names
        .getattr("str")
        .and_then(|s| s.call_method1("startswith", (("MT-", "mt-"),)))
        .and_then(|m| m.call_method0("sum"))
        .and_then(|v| v.extract::<usize>())
        .unwrap_or(0)
}

/// Count gene symbols starting with `MT-` or `mt-` in
/// `adata.var['feature_name']`. Used to detect the CELLxGENE Census layout
/// (integer-string var_names, real symbols under `feature_name`). Returns
/// 0 when `feature_name` is absent or the accessor chain fails.
///
/// pandas `.str.startswith` works directly on Categorical columns
/// (Census stores `feature_name` as Categorical) — no `.astype(str)` /
/// `Vec<String>` materialisation needed.
fn count_mt_prefix_in_feature_name(adata: &Bound<'_, PyAny>) -> usize {
    let Ok(var) = adata.getattr("var") else {
        return 0;
    };
    // pandas `feature_name in var.columns` check via __contains__.
    let has_col = var
        .call_method1("__contains__", ("feature_name",))
        .ok()
        .and_then(|v| v.is_truthy().ok())
        .unwrap_or(false);
    if !has_col {
        return 0;
    }
    let Ok(col) = var.get_item("feature_name") else {
        return 0;
    };
    col.getattr("str")
        .and_then(|s| s.call_method1("startswith", (("MT-", "mt-"),)))
        .and_then(|m| m.call_method0("sum"))
        .and_then(|v| v.extract::<usize>())
        .unwrap_or(0)
}

/// Compute per-gene `(total_counts, n_cells_by_counts)` via the CSC
/// path. Used by `calculate_qc_metrics(prefer_format="csc")`.
///
/// Capability gate (raised on missing CSC sidecar / non-column-local
/// transforms / row deletion vector active) is delegated to
/// `as_column_source()`. Honors `col_projection` if set on the
/// dataset.
/// Width of the per-column subset bitmask; `qc_vars` beyond this are handled by
/// running additional row passes (see [`resolve_qc_masks`]).
const QC_MASK_BITS: usize = 64;

/// One row pass' worth of `qc_var` subsets.
struct QcMaskChunk {
    /// One bitmask per **visible** column: bit *k* set ⇒ that column belongs to
    /// this chunk's `k`-th subset.
    bits: Vec<u64>,
    /// Number of subsets in this chunk (≤ [`QC_MASK_BITS`]), carried separately
    /// so an all-false mask still gets its own all-zero output row.
    n_qc: usize,
}

/// The `qc_var` gene subsets, encoded for the fused row pass.
struct QcMasks {
    /// Subsets split into ≤64-wide chunks, in `qc_vars` order. Empty when no
    /// `qc_var` was requested (the row pass then runs once with no subsets).
    chunks: Vec<QcMaskChunk>,
}

/// Read each `qc_var` boolean column off `adata.var` and pack the subsets into
/// bitmasks over the visible columns, **chunked** at the 64-bit mask width.
///
/// The masks index `adata.var`, i.e. the **visible** gene axis, which is also
/// the space the row pass accumulates in (it projects each shard before
/// accumulating). So — unlike the previous implementation, which handed visible
/// positions straight to the underlying reader — no index composition is needed
/// here; the projection is applied once, by the kernel.
///
/// More than 64 `qc_var`s is **supported**, not rejected: the subsets are split
/// into `ceil(n / 64)` chunks and the row pass runs once per chunk, so the total
/// is `ceil(n / 64) + 1` shard passes (the common `["mt", "ribo", "hb"]` case is
/// one chunk → 2 passes). Rejecting instead would have been a capability
/// regression — the pre-fusion implementation handled any number of `qc_var`s,
/// at one full pass each.
fn resolve_qc_masks(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    qc_vars: &[String],
) -> PyResult<QcMasks> {
    if qc_vars.is_empty() {
        return Ok(QcMasks { chunks: Vec::new() });
    }

    let n_visible = source_n_vars(x)?;
    let np = crate::pyimport::import_module(py, "numpy")?;
    let var_df = adata.getattr("var")?;
    let mut chunks: Vec<QcMaskChunk> = Vec::new();

    for group in qc_vars.chunks(QC_MASK_BITS) {
        let mut bits = vec![0u64; n_visible];
        for (k, qc_var) in group.iter().enumerate() {
            let mask_values = var_df.get_item(qc_var.as_str())?.getattr("values")?;
            let idx: Vec<u32> = np
                .call_method1("where", (&mask_values,))?
                .get_item(0)?
                .call_method1("astype", ("uint32",))?
                .extract()?;
            for &i in &idx {
                let slot = bits.get_mut(i as usize).ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "qc_var {qc_var:?} mask index {i} is out of range for the {n_visible} \
                         visible columns; adata.var and adata.X disagree on the gene axis"
                    ))
                })?;
                *slot |= 1u64 << k;
            }
        }
        chunks.push(QcMaskChunk {
            bits,
            n_qc: group.len(),
        });
    }

    Ok(QcMasks { chunks })
}

/// Refuse a matrix `owned_csr` would silently materialize in full.
///
/// The in-memory arm accepts what it can copy cheaply and predictably: a scipy
/// sparse matrix or a numpy array. Anything else reaches
/// `scipy.sparse.csr_matrix(x)`, which asks for the array protocol — and for a
/// `dask.array.Array` that is `compute()`, i.e. the **whole** matrix densified
/// before a single QC number is produced. Measured: pyscx computed the entire
/// array where `sc.pp.calculate_qc_metrics` had done bounded per-axis
/// reductions. The `nnz × 8` large-copy warning cannot help, since it is sized
/// from the CSR that materialization produces — after the peak.
///
/// This used to be scanpy's problem: the delegation covered "neither of our two
/// SCX handles", so dask went to scanpy's own dask arm. Unifying the kernel took
/// that away, so the honest contract is to refuse rather than to quietly spend
/// the memory. An h5ad-backed `CSRDataset` lands here too, where it previously
/// failed deeper down with `scipy.sparse does not support dtype object`.
fn reject_unmaterialized_matrix(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<()> {
    let np = crate::pyimport::import_module(py, "numpy")?;
    let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
    let is_sparse: bool = scipy_sparse
        .call_method1("issparse", (x,))?
        .extract()
        .unwrap_or(false);
    if is_sparse || x.is_instance(&np.getattr("ndarray")?)? {
        return Ok(());
    }
    let name = x
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    Err(PyRuntimeError::new_err(format!(
        "calculate_qc_metrics does not accept a {name} matrix: it would have to \
         materialize the whole thing before computing anything (for a dask array \
         that is a full `compute()`), which is neither bounded nor announced. \
         Pass a scipy sparse or numpy `X`, an SCX handle from \
         `pyscx.open(...).to_anndata(backed=True)`, or materialize deliberately \
         first (e.g. `adata.X = adata.X.compute()`)."
    )))
}

/// Which kind of matrix `calculate_qc_metrics` was pointed at.
///
/// `BackedLayer` exists because `adata.layers[name]` on a backed file is
/// `ScxBackedLayerDataset` — a distinct `#[pyclass]` wrapping an
/// `ScxBackedSparseDataset` — so a plain `cast::<ScxBackedSparseDataset>()`
/// silently misses it and the handle ends up in the in-memory arm, where
/// `scipy.sparse.csr_matrix(<handle>)` raises.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum QcSourceKind {
    Backed,
    BackedLayer,
    Lazy,
    InMemory,
}

impl QcSourceKind {
    fn of(x: &Bound<'_, PyAny>) -> Self {
        if x.cast::<ScxBackedSparseDataset>().is_ok() {
            Self::Backed
        } else if x.cast::<ScxBackedLayerDataset>().is_ok() {
            Self::BackedLayer
        } else if x.cast::<ScxLazyTransformedDataset>().is_ok() {
            Self::Lazy
        } else {
            Self::InMemory
        }
    }
}

/// Visible column count for any matrix `calculate_qc_metrics` accepts.
///
/// One pass over the four kinds, rather than a classify-then-re-extract pair:
/// `percent_top` has to be range-checked against the **visible** width before a
/// route is chosen, and under a column projection that is not the file's
/// `n_vars`.
fn source_n_vars(x: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        return Ok(backed.shape_val.1);
    }
    if let Ok(layer) = x.extract::<PyRef<ScxBackedLayerDataset>>() {
        return Ok(layer.inner.shape_val.1);
    }
    if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        return Ok(lazy.shape_val.1);
    }
    let shape: (usize, usize) = x.getattr("shape")?.extract()?;
    Ok(shape.1)
}

/// Validate and canonicalize `percent_top` into ascending, de-duplicated,
/// 1-indexed positions.
///
/// scanpy raises `IndexError("Positions outside range of features.")` here; we
/// raise `ValueError` naming the offending value and the width it has to fit,
/// because this is a bad argument rather than a bad subscript, and because the
/// width that matters is the *visible* one — under a column projection it is
/// not the file's `n_vars`, and a message quoting the file would send the
/// reader looking in the wrong place.
///
/// An empty collection means "not requested", matching scanpy's `if
/// percent_top:` — it is skipped, not rejected.
fn normalize_percent_top(percent_top: Option<&[i64]>, n_visible: usize) -> PyResult<Vec<usize>> {
    let Some(requested) = percent_top else {
        return Ok(Vec::new());
    };
    let mut ns: Vec<usize> = Vec::with_capacity(requested.len());
    for &n in requested {
        if n <= 0 {
            return Err(PyValueError::new_err(format!(
                "percent_top entries are 1-indexed gene positions and must be >= 1, got {n}"
            )));
        }
        let n = n as usize;
        if n > n_visible {
            return Err(PyValueError::new_err(format!(
                "percent_top entry {n} exceeds the {n_visible} visible genes; \
                 percent_top positions must fall in 1..={n_visible}"
            )));
        }
        ns.push(n);
    }
    ns.sort_unstable();
    ns.dedup();
    Ok(ns)
}

/// Drop explicitly-stored zeros from an in-memory CSR.
///
/// The QC kernel takes each row's nnz from `indptr`, which is right for SCX
/// shards — they never store a zero. A user's scipy matrix can (`X[mask] = 0`
/// leaves the entries in place), and counting those would report more expressed
/// genes and fewer dropouts than there are. scanpy avoids it by calling
/// `x.eliminate_zeros()` on the caller's matrix; doing it on our own copy
/// reaches the same numbers without mutating the caller's data.
///
/// Returns the input untouched when there is nothing to drop, which is the
/// common case and costs one scan.
fn canonicalize_for_qc(mut csr: scx_sparse::ScxCsr) -> scx_sparse::ScxCsr {
    if !csr.data.contains(&0.0) {
        return csr;
    }
    // Single forward pass, compacting in place: read `indptr[row + 1]` into
    // `next_start` before the write that overwrites it. Mirrors
    // `scx_sparse::drop_explicit_zeros_inplace`, which cannot be called here —
    // it takes the writer-side `u64`/`u32` shard widths, while `ScxCsr` is
    // scipy-compatible `i64`/`i32`.
    let n_rows = csr.n_rows();
    let mut write = 0usize;
    let mut next_start = csr.indptr[0] as usize;
    for row in 0..n_rows {
        let start = next_start;
        let end = csr.indptr[row + 1] as usize;
        next_start = end;
        for k in start..end {
            if csr.data[k] != 0.0 {
                csr.indices[write] = csr.indices[k];
                csr.data[write] = csr.data[k];
                write += 1;
            }
        }
        csr.indptr[row + 1] = write as i64;
    }
    csr.indices.truncate(write);
    csr.data.truncate(write);
    csr
}

/// Deletion-filtered outputs of the fused row pass, in publish order.
struct RowQcOutputs {
    total_counts: Vec<f64>,
    n_genes: Vec<i64>,
    qc_subset_sums: Vec<Vec<f64>>,
    top_fractions: Vec<Vec<f64>>,
}

impl RowQcOutputs {
    /// Run `pass` once per `qc_var` chunk and assemble the visible-row outputs.
    ///
    /// With ≤64 `qc_var`s (every realistic call) this is a single pass. Beyond
    /// that, each extra chunk costs one more shard scan; `nnz`/`sums` are taken
    /// from the first chunk and the later chunks contribute only their subset
    /// sums. A `qc_vars`-free call still runs exactly one pass.
    ///
    /// `kept` is the visible→global row map (`None` = no deletions), matching
    /// `filter_row_results` on both dataset types.
    fn accumulate<F>(
        masks: &QcMasks,
        kept: Option<&[u64]>,
        percent_top: &[usize],
        mut pass: F,
    ) -> Result<Self, String>
    where
        F: FnMut(&[u64], usize, &[usize]) -> Result<projected_agg::QcRowStats, String>,
    {
        // Consumes `all`: with no deletion vector the global-length vector is
        // already the answer, so the common path moves instead of copying.
        fn take<T: Copy>(all: Vec<T>, kept: Option<&[u64]>) -> Vec<T> {
            match kept {
                Some(map) => map.iter().map(|&g| all[g as usize]).collect(),
                None => all,
            }
        }

        let mut total_counts: Option<Vec<f64>> = None;
        let mut n_genes: Option<Vec<i64>> = None;
        let mut top_fractions: Vec<Vec<f64>> = Vec::new();
        let mut qc_subset_sums: Vec<Vec<f64>> =
            Vec::with_capacity(masks.chunks.iter().map(|c| c.n_qc).sum());

        // `chunks` is empty only when no qc_var was requested — still run one
        // pass, for total_counts / n_genes_by_counts.
        let empty_chunk = [QcMaskChunk {
            bits: Vec::new(),
            n_qc: 0,
        }];
        let chunks = if masks.chunks.is_empty() {
            &empty_chunk[..]
        } else {
            &masks.chunks[..]
        };

        for (i, chunk) in chunks.iter().enumerate() {
            // Only the first chunk's row-axis outputs are kept (below), so the
            // later chunks must not pay for a per-row top-N selection whose
            // result is discarded. The "first chunk only" decision lives here,
            // not in each caller's closure.
            let ns = if i == 0 { percent_top } else { &[] };
            let stats = pass(&chunk.bits, chunk.n_qc, ns)?;
            if total_counts.is_none() {
                total_counts = Some(take(stats.sums, kept));
                n_genes = Some(take(stats.nnz, kept));
                // Like `sums`/`nnz`, these do not depend on the qc_var chunk;
                // taking them from every chunk would recompute an identical
                // answer once per additional 64 subsets.
                top_fractions = stats
                    .top_fractions
                    .into_iter()
                    .map(|v| take(v, kept))
                    .collect();
            }
            qc_subset_sums.extend(stats.qc_sums.into_iter().map(|v| take(v, kept)));
        }

        Ok(Self {
            total_counts: total_counts.unwrap_or_default(),
            n_genes: n_genes.unwrap_or_default(),
            qc_subset_sums,
            top_fractions,
        })
    }
}

fn compute_gene_axis_csc(x: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<u32>)> {
    if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        let source = backed.as_column_source().ok_or_else(|| {
            PyRuntimeError::new_err(
                "CSC requested but unavailable: file has no CSC sidecar, \
                 or a row deletion vector is active",
            )
        })?;
        let cols_owned: Vec<u32> = match backed.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..backed.shape_val.1 as u32).collect(),
        };
        return projected_agg::col_sums_and_nnz_projected_csc(source, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()));
    }
    if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let lazy_src = lazy.as_column_source().ok_or_else(|| {
            PyRuntimeError::new_err(
                "CSC requested but unavailable: file has no CSC sidecar, \
                 the transform chain contains a non-column-local op, or a \
                 row deletion vector is active",
            )
        })?;
        let cols_owned: Vec<u32> = match lazy.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..lazy.shape_val.1 as u32).collect(),
        };
        return projected_agg::col_sums_and_nnz_projected_csc(&lazy_src, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()));
    }
    Err(PyRuntimeError::new_err(
        "prefer_format='csc' requires backed or lazy SCX dataset",
    ))
}
