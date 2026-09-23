//! Highly variable gene selection — seurat_v3 and seurat flavors.
//!
//! Split into flavor submodules (ORG-10.16-6): `seurat_v3` and `seurat`
//! (the two streaming kernels), `csc` (the column-major seurat_v3 path),
//! `loess_diag` (loess-failure diagnostics + their tests). This module keeps
//! the entry point / dispatch, the shard-source plumbing and the shared
//! post-processing, and re-exports the `#[pyfunction]` so lib.rs's
//! `accel::hvg::highly_variable_genes` registration path is unchanged.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::{ScxBackedLayerDataset, ScxBackedSparseDataset};
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};
use crate::optional_deps::{import_optional_with_hint, EXTRA_SCANPY};

pub(crate) mod csc;
pub(crate) mod loess_diag;
pub(crate) mod seurat;
pub(crate) mod seurat_v3;

pub(crate) use csc::*;
pub(crate) use loess_diag::*;
pub(crate) use seurat::*;
pub(crate) use seurat_v3::*;

/// Single-shard [`scx_format_io::ShardSource`] adapter over a borrowed in-memory
/// [`scx_sparse::ScxCsr`].
///
/// Lets a materialized scipy/dense `adata.X` flow through the same native HVG
/// kernels as the backed/lazy datasets — so in-memory `seurat_v3` / `seurat`
/// gets identical numerics and the same per-batch LOESS-singularity tolerance
/// instead of delegating to `scanpy.pp.highly_variable_genes`. Mirrors the
/// GPU-only `ScxCsrSource` in `pca.rs`, but is non-feature-gated because the
/// CPU HVG path needs it too. `ScxCsr` is `Send + Sync`, so this satisfies the
/// `+ Sync` bound the GPU kernel variants require.
struct InMemoryCsrSource<'a> {
    csr: &'a scx_sparse::ScxCsr,
}

impl scx_format_io::ShardSource for InMemoryCsrSource<'_> {
    fn n_shards(&self) -> usize {
        1
    }
    fn n_obs(&self) -> usize {
        self.csr.n_rows()
    }
    fn n_vars(&self) -> usize {
        self.csr.n_cols()
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<scx_sparse::ScxCsr> {
        if shard_idx != 0 {
            return Err(scx_format_io::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: 1,
            });
        }
        // One clone per pass (mean/var, then clipped-sum). Acceptable for a
        // first cut; a borrowed-slice variant (cf. `BorrowedCsrSource`) can
        // remove it later if in-memory HVG RSS becomes a concern.
        Ok(self.csr.clone())
    }
    fn max_shard_rows(&self) -> scx_format_io::Result<usize> {
        Ok(self.csr.n_rows())
    }
}

/// Whether `x` is a backed SCX dataset this op should **auto-route** to the
/// column-major GPU reduce: a CSC sidecar is present *and* no row filter is
/// active.
///
/// The row-filter clause is a policy choice, not a capability one — the reader
/// serves a filtered handle's CSC reads perfectly well, and an explicit
/// `prefer_format="csc"` still gets them. But HVG is the one CSC consumer
/// whose column-major walk is *slower* than the row-major sweep it replaces
/// (see the `highly_variable_genes` row in `docs/api.md`), so this auto-route
/// exists only for the case where the sidecar is otherwise free. A row filter
/// adds a compaction pass over every column shard on top of a walk that was
/// already losing, and nobody asked for CSC: `filter_cells` →
/// `highly_variable_genes(device="gpu")` would silently record `gpu_csc_v3`.
fn backed_x_has_csc_sidecar(x: &Bound<'_, PyAny>) -> bool {
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let b = backed.borrow();
        b.kept_to_global.is_none() && b.as_column_source().is_some()
    } else {
        false
    }
}

/// Streaming highly-variable gene selection without materialization.
///
/// Computes HVG statistics shard-by-shard via the `ShardSource` abstraction,
/// then selects the top `n_top_genes` by normalized variance (seurat_v3) or
/// normalized dispersion (seurat).
///
/// With `device="gpu"`, the per-column mean/variance and clipped-sum kernels
/// run on GPU (see `scx_gpu::gpu_streaming_mean_var` /
/// `gpu_streaming_clip_square_sum`, plus the batched variants used when
/// `batch_key` is set). The loess fit, ranking, and result writing stay on
/// CPU. The GPU path runs for any `seurat_v3` configuration regardless of
/// `batch_key`; `flavor="seurat"` still falls back to CPU (a single warning
/// is emitted in that case).
///
/// Per-batch loess fits are run via `skmisc.loess`. If a batch's variance
/// structure is too degenerate for loess (small batch sizes, near-singular
/// log-mean / log-variance regression), the fit is caught, a UserWarning is
/// emitted naming the batch index and size, and that batch is excluded from
/// the per-batch normalised-variance ranking. Other batches proceed normally.
///
/// `X` may be an `ScxBackedSparseDataset`, an `ScxLazyTransformedDataset`, or a
/// materialized scipy/dense matrix: for `flavor` in `seurat_v3` /
/// `seurat_v3_paper` / `seurat`, all three run the native streaming kernel (a
/// materialized `X` is wrapped in a single-shard `ShardSource`), so in-memory
/// `X` gets the same numerics and the same per-batch LOESS-singularity
/// tolerance as the backed path. Only flavors the native kernel does not
/// implement (e.g. `cell_ranger`) delegate to `scanpy.pp.highly_variable_genes`
/// (with a one-shot UserWarning).
///
/// Args:
///     adata: AnnData with X as ScxBackedSparseDataset, ScxLazyTransformedDataset,
///         or a materialized scipy/dense matrix
///     n_top_genes: Number of highly variable genes to select (default: 2000)
///     flavor: "seurat_v3" (raw counts) or "seurat" (log-normalized) (default: "seurat_v3")
///     batch_key: Column in adata.obs for batch-aware HVG (default: None)
///     span: Loess span for seurat_v3 (default: 0.3)
///     subset: If True, subset adata to HVG via column projection (default: False)
///     n_bins: Number of bins for seurat flavor (default: 20)
///     device: Device selection — "auto" (default), "cpu", "gpu", or
///         "gpu:N" to target CUDA device N on multi-GPU systems.
///     prefer_format: "csr" (default) or "csc". When "csc", the streaming
///         mean/var and clipped-sum passes use the column-major sidecar
///         instead of the row-major shards. Requires the file to have a
///         CSC sidecar (`from_anndata(csc="always")`). Single-batch
///         seurat_v3 only — multi-batch and seurat flavor raise on CSC.
///         With `device="gpu"` the CSC sidecar runs the column-major
///         reduce kernel (route `gpu_csc_v3`, no `atomicAdd` contention);
///         on CPU it runs the CSC reduce (route `cpu_csc`), which is
///         **slower than the CSR default** on every file measured (4.4x at
///         tabula_sapiens_100k, 6.4x at census_1m) and is kept for parity
///         testing, not speed. Note: even
///         under the default `prefer_format="csr"`, a single-batch
///         seurat_v3 GPU run on a backed dataset that has a CSC sidecar
///         auto-routes to `gpu_csc_v3` — but **only when no row filter is
///         active**. The column-major walk is slower than the row-major
///         sweep here, so that auto-route exists for the case where the
///         sidecar costs nothing; a filtered handle would add a row
///         compaction on top and is left on `gpu_csr`. An explicit
///         `prefer_format="csc"` is still honoured on any window.
///     layer: Read counts from `adata.layers[layer]` instead of
///         `adata.X`. Mirrors `scanpy.pp.highly_variable_genes(layer=)`
///         and is the canonical way to compute `flavor="seurat_v3"`
///         on a raw-counts layer after the main X has been
///         log-normalized.
#[pyfunction]
#[pyo3(signature = (adata, n_top_genes=2000, flavor="seurat_v3", batch_key=None, span=0.3, subset=false, n_bins=20, device="auto", prefer_format="csr", layer=None))]
#[allow(clippy::too_many_arguments)]
pub fn highly_variable_genes<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    n_top_genes: usize,
    flavor: &str,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    n_bins: usize,
    device: &str,
    prefer_format: &str,
    layer: Option<&str>,
) -> PyResult<()> {
    // Validate prefer_format up front (matches the constraint applied
    // across all `pyscx.accel.*` entry points).
    if !matches!(prefer_format, "csr" | "csc") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'csr' or 'csc'"
        )));
    }
    let seurat_v3_family = matches!(flavor, "seurat_v3" | "seurat_v3_paper");
    let single_batch = batch_key.is_none();

    // HVG computes per-gene stats and the selection mask over the sorted
    // column projection (via build_shard_source), then writes them to
    // adata.var. A presentation-ordered backed X (preserve_var_order=True)
    // would misalign those stats against the request-ordered var — reject.
    // Deliberately the no-var-guard prologue: the
    // `reject_presentation_ordered_source` call below guards the matrix this
    // op will actually read, which is `adata.layers[layer]` when `layer=` was
    // given. Keeping the X-only check here made the documented remedy
    // impossible — materialise the layer, and the op still refused because
    // `adata.X` was presentation-ordered, while the matrix it was about to
    // read was fine.
    super::prepare_target_no_var_guard(py, adata, "highly_variable_genes")?;

    // Resolve the source matrix and guard it *before* the route stamp: a guard
    // has to refuse before `adata` is touched (see `accel::prepare_target`'s
    // ordering invariant), and `prepare_target` itself only inspected
    // `adata.X`, which says nothing about a named layer.
    let x = match layer {
        // Typed error naming the layer, as `calculate_qc_metrics` /
        // `score_genes` / `select_de_matrix` do, rather than the mapping's
        // bare `KeyError`.
        Some(name) => adata.getattr("layers")?.get_item(name).map_err(|_| {
            PyValueError::new_err(format!("layer '{name}' not found in adata.layers"))
        })?,
        None => adata.getattr("X")?,
    };
    super::reject_presentation_ordered_source(&x, "highly_variable_genes")?;

    if prefer_format == "csc" {
        // Explicit CSC: single-batch seurat_v3 only. Reject mismatched
        // configurations with a clear message rather than silently falling
        // back, since the user has explicitly opted in.
        if !single_batch {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' for HVG only supports single-batch mode; \
                 pass batch_key=None or use prefer_format='csr'",
            ));
        }
        if !seurat_v3_family {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' for HVG only supports flavor='seurat_v3' \
                 (or 'seurat_v3_paper'); use prefer_format='csr' for 'seurat'",
            ));
        }
        // The CSC sidecar lives on `adata.X`, not on arbitrary layers — the CSC
        // dispatch (`hvg_seurat_v3_csc`) reads `adata.X` unconditionally. Reject
        // `layer=` rather than silently computing on X (mirrors the `layer.is_none()`
        // gate on the CSR auto-detect path).
        if layer.is_some() {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' for HVG reads adata.X only and does not support \
                 layer=; pass layer=None or use prefer_format='csr'",
            ));
        }
        // Reached via `adata.X = adata.layers["counts"]`, where `layer` is
        // already None, so the advice above would name a kwarg the caller is
        // not using. A layer handle carries no CSC sidecar of its own.
        if x.cast::<ScxBackedLayerDataset>().is_ok() {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' for HVG does not support a layer source \
                 (adata.X is a layer handle; the CSC sidecar belongs to the file's \
                 X). Restore adata.X to the base backed dataset, or use \
                 prefer_format='csr'",
            ));
        }
        // GPU CSC reduce when a GPU is requested and available, else CPU CSC.
        // The GPU CSC reduce kernel lifts the previous "csc is cpu-only"
        // restriction; an explicit device="gpu" with no GPU still errors via
        // resolve_device (consistent with the CSR path).
        #[cfg(feature = "gpu")]
        let csc_device_id: Option<usize> = super::gpu::resolve_device(device)?.gpu_id();
        #[cfg(not(feature = "gpu"))]
        let csc_device_id: Option<usize> = {
            // Validate the device string even without GPU support.
            let _ = super::gpu::resolve_device(device)?;
            None
        };
        // Route: gpu_csc_v3 (GPU reduce) or cpu_csc — the planner reads
        // gpu_available() internally, so a no-GPU host records cpu_csc.
        // Pre-dispatch stamp is honest because the CSC GPU wrappers propagate
        // `AccelError::GpuInitFailed` (§4.1) instead of silently CPU-ing under
        // this stamp — a failed GPU init fails the op, never mislabels CPU.
        let info = super::route::hvg_exec_info(device, seurat_v3_family, true);
        super::route::announce_route(py, "highly_variable_genes", device, &info);
        let route = super::route::RouteStamp::write(py, adata, "highly_variable_genes", &info)?;
        return route.settle(hvg_seurat_v3_csc(
            py,
            adata,
            n_top_genes,
            span,
            subset,
            flavor,
            csc_device_id,
        ));
    }
    let resolved = super::gpu::resolve_device(device)?;

    // The HVG flavors SCX has no native GPU kernel for (`seurat` / `cell_ranger` /
    // `pearson_residuals` / `poisson_gene_selection`) route to rapids-singlecell in
    // the in-VRAM regime. `seurat_v3` / `seurat_v3_paper` stay native (SCX wins).
    // Restricted to an in-memory X read from adata.X (no `layer=`).
    #[cfg(feature = "gpu")]
    if !seurat_v3_family && layer.is_none() {
        let x_in_memory = {
            let xp = adata.getattr("X")?;
            xp.cast::<ScxBackedSparseDataset>().is_err()
                && xp.cast::<ScxLazyTransformedDataset>().is_err()
        };
        if x_in_memory {
            match super::rapids::decide(py, resolved, "highly_variable_genes") {
                super::rapids::RapidsDecision::Rapids(gid) => {
                    super::rapids::run(py, adata, "highly_variable_genes", gid, |py, adata| {
                        let kw = super::rapids::kwargs(py);
                        kw.set_item("n_top_genes", n_top_genes)?;
                        kw.set_item("flavor", flavor)?;
                        if let Some(bk) = batch_key {
                            kw.set_item("batch_key", bk)?;
                        }
                        kw.set_item("span", span)?;
                        kw.set_item("n_bins", n_bins)?;
                        // NB: rsc.pp.highly_variable_genes has no `subset` kwarg
                        // (unlike scanpy); we apply the subset post-hoc below,
                        // exactly as the CPU/native path does.
                        super::rapids::rsc_fn(py, "pp", "highly_variable_genes")?
                            .call((adata,), Some(&kw))?;
                        Ok(())
                    })?;
                    // `run` restored X to host (host-in → host-out); honor
                    // `subset=True` the same way the CPU/native path does, since
                    // rsc.pp.highly_variable_genes does not subset itself.
                    if subset {
                        let mask: Vec<bool> = adata
                            .getattr("var")?
                            .get_item("highly_variable")?
                            .call_method0("to_numpy")?
                            .extract::<numpy::PyReadonlyArray1<bool>>()?
                            .as_slice()?
                            .to_vec();
                        apply_hvg_subset(py, adata, &mask)?;
                    }
                    return Ok(());
                }
                super::rapids::RapidsDecision::NoRapidsCpu => {
                    highly_variable_genes(
                        py,
                        adata,
                        n_top_genes,
                        flavor,
                        batch_key,
                        span,
                        subset,
                        n_bins,
                        "cpu",
                        prefer_format,
                        layer,
                    )?;
                    return super::rapids::stamp_no_rapids(
                        py,
                        adata,
                        "highly_variable_genes",
                        scx_accel::AccelRoute::CpuCsr,
                    );
                }
                super::rapids::RapidsDecision::Native => {}
            }
        }
    }
    // GPU path is supported for any seurat_v3 / seurat_v3_paper config
    // (including multi-batch via per-batch kernels). For flavor="seurat" the
    // GPU kernels don't apply — fall back to CPU with a single warning.
    #[cfg(feature = "gpu")]
    let effective_gpu_id: Option<usize> = if let Some(gid) = resolved.gpu_id() {
        let seurat_v3 = matches!(flavor, "seurat_v3" | "seurat_v3_paper");
        if !seurat_v3 {
            let warnings = crate::pyimport::import_module(py, "warnings")?;
            warnings.call_method1(
                "warn",
                (
                    format!(
                        "highly_variable_genes(device={device:?}) GPU path is only \
                         implemented for flavor=\"seurat_v3\" (or \"seurat_v3_paper\"); \
                         falling back to CPU for flavor={flavor:?}."
                    ),
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
            None
        } else {
            Some(gid)
        }
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    let effective_gpu_id: Option<usize> = {
        let _ = resolved;
        None
    };

    // F3: read the source matrix from `adata.layers[layer]` when a
    // layer is named (scanpy parity); otherwise from `adata.X`. A backed
    // file's layer arrives as `ScxBackedLayerDataset`, which has its own
    // dispatch arm below (it is not an `ScxBackedSparseDataset`); a scipy
    // layer falls through to the in-memory path, and a flavor scx has no
    // native kernel for falls through to scanpy with `layer=`.
    // Auto-detect CSC, mirroring DE's gpu_csc_v3 default: a single-batch
    // seurat_v3 GPU run on a backed dataset that exposes a CSC sidecar uses
    // the column-major reduce (route gpu_csc_v3) even under the default
    // prefer_format="csr". Restricted to `layer is None` (the sidecar lives
    // on adata.X, not on arbitrary layers) and to a GPU run
    // (`effective_gpu_id.is_some()`) — we do not silently switch the CPU
    // default from CSR to CSC. `backed_x_has_csc_sidecar` is the
    // `as_column_source` capability gate **plus** "no row filter" — it is a
    // policy predicate, not a mirror of the capability one, because this op's
    // CSC walk is the slower of the two.
    if effective_gpu_id.is_some()
        && single_batch
        && seurat_v3_family
        && layer.is_none()
        && backed_x_has_csc_sidecar(&x)
    {
        let info = super::route::hvg_exec_info(device, true, true);
        super::route::announce_route(py, "highly_variable_genes", device, &info);
        let route = super::route::RouteStamp::write(py, adata, "highly_variable_genes", &info)?;
        return route.settle(hvg_seurat_v3_csc(
            py,
            adata,
            n_top_genes,
            span,
            subset,
            flavor,
            effective_gpu_id,
        ));
    }

    // Record the planned CSR route on
    // adata.uns["scx_accel"]["highly_variable_genes"] before dispatch. Only
    // the seurat_v3 family has a GPU kernel; "seurat" and "cell_ranger" are
    // CPU-only (gpu_eligible=false records UnsupportedInputLayout on a GPU
    // host rather than implying CUDA was absent). This single stamp covers the
    // backed / lazy / in-memory native paths and the scanpy fallback — they
    // share `device` + `flavor`.
    //
    // INVARIANT (pre-dispatch stamp): safe to stamp *before* dispatch only
    // because (a) `seurat_v3_family` mirrors the same flavor gate that derives
    // `effective_gpu_id` above (GPU runs iff the flavor is seurat_v3 family and
    // CUDA is present), and (b) the streaming seurat_v3 GPU kernel propagates
    // errors via `.map_err(..)?` rather than silently falling back to CPU — so
    // the recorded `gpu_csr` route always reflects the code that ran. If a
    // silent GPU→CPU runtime fallback is ever added, stamp *after* dispatch on
    // the branch that ran (see umap.rs) or this gate will false-pass.
    //
    // The invariant is about which *branch* the recorded route names; it is
    // orthogonal to whether the op finished. `RouteStamp` covers the latter —
    // the stamp is rolled back if any branch below raises, so a present entry
    // means the route ran *and* completed.
    // Plan with `false`, then correct only the capability field.
    //
    // `hvg_exec_info`'s third argument is a **route** input, not a metadata
    // one: `plan_hvg_route` picks `CpuCsc` / `GpuCscV3` from it, and derives
    // `reduction` from the route it picked. This stamp belongs to the CSR
    // fall-through — the branch reached *after* the CSC auto-route has been
    // declined — so passing the real capability here named a CSC kernel over a
    // CSR one: `cpu_csc` for any sidecar file on CPU, and `gpu_csc_v3` with
    // `reduction="deterministic"` for a row-filtered GPU run whose reduce is
    // the atomic CSR one. That is the shape DE avoids by planning the CSR
    // layout and overwriting the field afterwards, and it is what this now
    // does.
    //
    // Capability comes from `x`, the matrix this op will actually read, not
    // from `adata.X`: under `layer=` those differ, the sidecar lives on `X`,
    // and the layer's own is not usable here.
    let mut info = super::route::hvg_exec_info(device, seurat_v3_family, false);
    info.csc_available = Some(crate::accel::csc_source_for(&x).is_some());
    super::route::announce_route(py, "highly_variable_genes", device, &info);
    // Rolled back if any branch below raises (a loess singularity across every
    // batch, a missing `batch_key`, …) — see RouteStamp.
    let route = super::route::RouteStamp::write(py, adata, "highly_variable_genes", &info)?;

    // ── Try a backed SCX dataset: `adata.X` or a backed layer handle ────
    if let Some(parts) = super::backed_shard_parts(&x) {
        let source = build_shard_source(
            &parts.reader,
            &[],
            &parts.kept,
            &parts.col_proj,
            parts.n_vars,
        );
        return route.settle(hvg_on_source(
            py,
            adata,
            &source,
            parts.n_obs,
            parts.n_vars,
            n_top_genes,
            flavor,
            batch_key,
            span,
            subset,
            n_bins,
            effective_gpu_id,
        ));
    }

    // ── Try SCX lazy-transformed dataset ────────────────────────────────
    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let reader = Arc::clone(&lazy_ref.backed);
        let transforms = lazy_ref.transforms.clone();
        let n_vars = lazy_ref.shape_val.1;
        let n_obs = lazy_ref.shape_val.0;
        let kept = lazy_ref.kept_to_global.clone();
        let col_proj = lazy_ref.col_projection.clone();
        drop(lazy_ref);

        let source = build_shard_source(&reader, &transforms, &kept, &col_proj, n_vars);
        return route.settle(hvg_on_source(
            py,
            adata,
            &source,
            n_obs,
            n_vars,
            n_top_genes,
            flavor,
            batch_key,
            span,
            subset,
            n_bins,
            effective_gpu_id,
        ));
    }

    // ── In-memory scipy/dense X: run the native kernel ──────────────────
    // seurat_v3 / seurat_v3_paper / seurat are implemented natively, so wrap
    // the materialized matrix in a single-shard `ShardSource` and run the same
    // streaming kernel as the backed path. This gives in-memory X identical
    // numerics AND the per-batch LOESS-singularity tolerance — the scanpy
    // delegation below has none, which was the original Tier-2 crash
    // (per-batch loess on a high-cardinality `batch_key`). Only flavors the
    // native kernel doesn't implement (e.g. `cell_ranger`) fall through.
    if matches!(flavor, "seurat_v3" | "seurat_v3_paper" | "seurat") {
        let csr = crate::convert::owned_csr(py, &x, None)?;
        let n_obs = csr.n_rows();
        let n_vars = csr.n_cols();
        let source = InMemoryCsrSource { csr: &csr };
        return route.settle(hvg_on_source(
            py,
            adata,
            &source,
            n_obs,
            n_vars,
            n_top_genes,
            flavor,
            batch_key,
            span,
            subset,
            n_bins,
            effective_gpu_id,
        ));
    }

    // ── Fallback to scanpy (cell_ranger / unsupported flavors only) ─────
    // seurat_v3 / seurat_v3_paper / seurat now run the scx-native kernel even
    // on materialized scipy/dense X (above), so this path is reached only for
    // flavors scx does not implement natively — today `cell_ranger`. Surface
    // the delegation so the user knows scx handed off to scanpy (and inherits
    // scanpy's pd.cut bin-edge fragility on sparse Census data). Python's
    // default warning filter dedupes by (message, category, location), so
    // repeated calls only emit once per site.
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    let core = format!(
        "highly_variable_genes(flavor={flavor:?}) is not implemented natively in \
         scx and is delegated to scanpy.pp.highly_variable_genes. flavor=\"seurat_v3\", \
         \"seurat_v3_paper\" and \"seurat\" run the scx-native streaming kernel — \
         including on materialized scipy/dense X — and gain per-batch LOESS-singularity \
         tolerance there."
    );
    let fragility_tail = if matches!(flavor, "cell_ranger") {
        " On the scanpy path, `cell_ranger` (pd.cut) can fail with bin-edge \
         collisions on data with many low-expression genes — pre-filter via \
         pyscx.accel.filter_genes(min_cells=10), or use flavor=\"seurat_v3\" / \
         \"seurat\" to stay on the scx-native path."
    } else {
        ""
    };
    let msg = format!("{core}{fragility_tail}");
    warnings.call_method1("warn", (msg, user_warning))?;

    let sc = import_optional_with_hint(
        py,
        "scanpy",
        EXTRA_SCANPY,
        &format!("pyscx.accel.highly_variable_genes(flavor={flavor:?})"),
        "scanpy",
        // Not the backed hatch: this flavor has no scx-native kernel at all, so
        // backing X would not help. Point at the flavors that do.
        Some(
            "flavor=\"seurat_v3\", \"seurat_v3_paper\" and \"seurat\" run the \
             scx-native kernel and need no scanpy (\"seurat_v3\" needs \
             `pip install 'pyscx[hvg]'` for its loess).",
        ),
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("n_top_genes", n_top_genes)?;
    kwargs.set_item("flavor", flavor)?;
    kwargs.set_item("span", span)?;
    kwargs.set_item("subset", subset)?;
    kwargs.set_item("n_bins", n_bins)?;
    if let Some(bk) = batch_key {
        kwargs.set_item("batch_key", bk)?;
    }
    if let Some(layer_name) = layer {
        kwargs.set_item("layer", layer_name)?;
    }
    sc.getattr("pp")?
        .call_method("highly_variable_genes", (adata,), Some(&kwargs))?;
    route.commit();
    Ok(())
}

/// Dispatch to seurat_v3 or seurat HVG implementation.
///
/// Generic over any `ShardSource` so the same kernels serve the backed/lazy
/// datasets (`LazyShardSource`) and a materialized scipy/dense `X`
/// (`InMemoryCsrSource`). `+ Sync` is required by the GPU kernel variants;
/// both source types satisfy it.
#[allow(clippy::too_many_arguments)]
fn hvg_on_source<'py, S: scx_format_io::ShardSource + Sync>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    source: &S,
    n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    flavor: &str,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    n_bins: usize,
    device_id: Option<usize>,
) -> PyResult<()> {
    match flavor {
        "seurat_v3" | "seurat_v3_paper" => hvg_seurat_v3(
            py,
            adata,
            source,
            n_obs,
            n_vars,
            n_top_genes,
            batch_key,
            span,
            subset,
            flavor,
            device_id,
        ),
        "seurat" => hvg_seurat(
            py,
            adata,
            source,
            n_obs,
            n_vars,
            n_top_genes,
            batch_key,
            subset,
            n_bins,
        ),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported HVG flavor '{flavor}'. Use 'seurat_v3' or 'seurat'."
        ))),
    }
}

/// Build a full-dataset `LazyShardSource` for the backed / lazy dispatch.
///
/// Per-batch filtering is handled downstream via the `cell_batch` array passed
/// to the batched streaming kernels, so this always builds the whole-dataset
/// source (the previous `batch_indices` branch was unused).
pub(super) fn build_shard_source(
    reader: &Arc<scx_format_io::BackedCsrReader>,
    transforms: &[Transform],
    kept_to_global: &Option<Arc<Vec<u64>>>,
    col_projection: &Option<Arc<Vec<u32>>>,
    n_vars: usize,
) -> crate::lazy_transform::LazyShardSource {
    use crate::lazy_transform::LazyShardSource;

    // Pass None for kept rows when no filtering is needed — avoids allocating a
    // full identity range and skips the deletion-vector path in read_shard.
    let n_obs = match kept_to_global {
        Some(k) => k.len(),
        None => reader.shape().0,
    };
    LazyShardSource::new(
        Arc::clone(reader),
        transforms.to_vec(),
        kept_to_global.as_ref().map(Arc::clone),
        col_projection.clone(),
        n_obs,
        n_vars,
    )
}

/// Resolve the `expm1` scale for the seurat flavor from
/// `adata.uns["log1p"]["base"]`.
///
/// scanpy un-`log1p`s with `x *= np.log(base)` only when a numeric `base` is
/// recorded; natural-log / no recorded base is `scale = 1.0`. Mirrors
/// `uns.get("log1p", {}).get("base")` and tolerates a missing/none/odd `uns`.
fn log1p_base_scale(py: Python<'_>, adata: &Bound<'_, PyAny>) -> PyResult<f64> {
    let _ = py;
    let uns = match adata.getattr("uns") {
        Ok(u) => u,
        Err(_) => return Ok(1.0),
    };
    let log1p = match uns.call_method1("get", ("log1p",)) {
        Ok(v) if !v.is_none() => v,
        _ => return Ok(1.0),
    };
    let base = match log1p.call_method1("get", ("base",)) {
        Ok(v) if !v.is_none() => v,
        _ => return Ok(1.0),
    };
    // Defensive: an unexpected `base` type (not a number) must not crash HVG —
    // fall back to natural-log scale (1.0), matching the "no recorded base" case.
    let base: f64 = match base.extract() {
        Ok(b) => b,
        Err(_) => return Ok(1.0),
    };
    if base <= 0.0 || base == 1.0 {
        return Ok(1.0);
    }
    Ok(base.ln())
}

/// Apply HVG subset: the same var-axis mutation `filter_genes` performs.
///
/// `subset_var_axis` composes the new `col_projection` for a backed / lazy X
/// and hands a plain in-memory X to anndata's `_inplace_subset_var`; either way
/// it slices `_var` and every var-aligned member. The boolean mask selects rows
/// of the *already-updated* var frame, so the HVG result columns written above
/// survive.
fn apply_hvg_subset(py: Python<'_>, adata: &Bound<'_, PyAny>, hvg_mask: &[bool]) -> PyResult<()> {
    crate::axis_align::subset_var_axis(py, adata, hvg_mask)
}
