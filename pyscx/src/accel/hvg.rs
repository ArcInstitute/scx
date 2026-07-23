//! Highly variable gene selection — seurat_v3 and seurat flavors.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};

use super::filtering::update_layers_col_projection;
use super::util::extract_materialized_csr;

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

/// Whether `x` is a backed SCX dataset that exposes a usable CSC sidecar
/// (the same capability gate as `ScxBackedSparseDataset::as_column_source`:
/// a CSC sidecar is present and no row deletion vector is active). Used to
/// auto-route single-batch seurat_v3 GPU HVG to the column-major reduce.
fn backed_x_has_csc_sidecar(x: &Bound<'_, PyAny>) -> bool {
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        backed.borrow().as_column_source().is_some()
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
///         on CPU it runs the CSC reduce (route `cpu_csc`). Note: even
///         under the default `prefer_format="csr"`, a single-batch
///         seurat_v3 GPU run on a backed dataset that has a CSC sidecar
///         auto-routes to `gpu_csc_v3` (mirrors GPU DE).
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
    super::reject_preserve_var_order(adata, "highly_variable_genes")?;

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
        super::route::write_accel_route(py, adata, "highly_variable_genes", &info)?;
        return hvg_seurat_v3_csc(py, adata, n_top_genes, span, subset, flavor, csc_device_id);
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
                        let x_obj = adata.getattr("X")?;
                        apply_hvg_subset(py, adata, &x_obj, &mask)?;
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
            let warnings = py.import("warnings")?;
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
    // layer is named (scanpy parity); otherwise from `adata.X`. The
    // downstream dispatch on `ScxBackedSparseDataset` /
    // `ScxLazyTransformedDataset` works identically — if the layer is
    // itself an SCX-backed dataset (e.g. set via
    // `adata.layers["counts"] = adata.X.copy()`), the streaming path
    // applies; otherwise we fall through to scanpy with `layer=`.
    let x = match layer {
        Some(name) => adata.getattr("layers")?.get_item(name)?,
        None => adata.getattr("X")?,
    };

    // Auto-detect CSC, mirroring DE's gpu_csc_v3 default: a single-batch
    // seurat_v3 GPU run on a backed dataset that exposes a CSC sidecar uses
    // the column-major reduce (route gpu_csc_v3) even under the default
    // prefer_format="csr". Restricted to `layer is None` (the sidecar lives
    // on adata.X, not on arbitrary layers) and to a GPU run
    // (`effective_gpu_id.is_some()`) — we do not silently switch the CPU
    // default from CSR to CSC. `backed_x_has_csc_sidecar` mirrors the
    // `as_column_source` capability gate.
    if effective_gpu_id.is_some()
        && single_batch
        && seurat_v3_family
        && layer.is_none()
        && backed_x_has_csc_sidecar(&x)
    {
        let info = super::route::hvg_exec_info(device, true, true);
        super::route::announce_route(py, "highly_variable_genes", device, &info);
        super::route::write_accel_route(py, adata, "highly_variable_genes", &info)?;
        return hvg_seurat_v3_csc(
            py,
            adata,
            n_top_genes,
            span,
            subset,
            flavor,
            effective_gpu_id,
        );
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
    let info = super::route::hvg_exec_info(device, seurat_v3_family, false);
    super::route::announce_route(py, "highly_variable_genes", device, &info);
    super::route::write_accel_route(py, adata, "highly_variable_genes", &info)?;

    // ── Try SCX backed dataset ──────────────────────────────────────────
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();
        let reader = Arc::clone(&backed_ref.backed);
        let n_vars = backed_ref.shape_val.1;
        let n_obs = backed_ref.shape_val.0;
        let kept = backed_ref.kept_to_global.clone();
        let col_proj = backed_ref.col_projection_arc();
        drop(backed_ref);

        let source = build_shard_source(&reader, &[], &kept, &col_proj, n_vars);
        return hvg_on_source(
            py,
            adata,
            &x,
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
        );
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
        return hvg_on_source(
            py,
            adata,
            &x,
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
        );
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
        let csr = extract_materialized_csr(py, &x)?;
        let n_obs = csr.n_rows();
        let n_vars = csr.n_cols();
        let source = InMemoryCsrSource { csr: &csr };
        return hvg_on_source(
            py,
            adata,
            &x,
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
        );
    }

    // ── Fallback to scanpy (cell_ranger / unsupported flavors only) ─────
    // seurat_v3 / seurat_v3_paper / seurat now run the scx-native kernel even
    // on materialized scipy/dense X (above), so this path is reached only for
    // flavors scx does not implement natively — today `cell_ranger`. Surface
    // the delegation so the user knows scx handed off to scanpy (and inherits
    // scanpy's pd.cut bin-edge fragility on sparse Census data). Python's
    // default warning filter dedupes by (message, category, location), so
    // repeated calls only emit once per site.
    let warnings = py.import("warnings")?;
    let user_warning = py.import("builtins")?.getattr("UserWarning")?;
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

    let sc = py.import("scanpy")?;
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
    x_obj: &Bound<'py, PyAny>,
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
            x_obj,
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
            x_obj,
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

/// Emit a single summary UserWarning for all batches whose `skmisc.loess.fit()`
/// raised a singularity `ValueError`.
///
/// `failed` is `(batch_idx, batch_n, error_string)` per failing batch, in
/// iteration order; `n_total` is the total batch count. A high-cardinality
/// `batch_key` (e.g. a per-dataset id) can produce dozens of tiny, singular
/// batches — emitting one warning each buries the signal under verbatim
/// copies (user-report F10). Instead we coalesce into one warning that reports
/// the failed/total count and a representative first failure (index + cell
/// count + the upstream error string), then lists the remedies. The full
/// per-batch index/size detail is preserved on `adata.uns` by the caller.
///
/// Each failing batch is excluded from the per-batch normalised-variance
/// ranking — semantics identical to a batch with too few non-constant genes.
/// Remedies are ordered by what actually helps the per-batch case: the
/// singularity is driven by small / near-collinear batches, so dropping or
/// coarsening `batch_key` is the effective fix. `filter_genes(min_cells=10)`
/// only helps the single global fit (no `batch_key`); it cannot make a tiny
/// batch's log-mean / log-variance regression well-conditioned.
fn emit_hvg_loess_singularity_warning(
    py: Python<'_>,
    failed: &[(usize, usize, String)],
    n_total: usize,
) -> PyResult<()> {
    if failed.is_empty() {
        return Ok(());
    }
    let warnings = py.import("warnings")?;
    let (first_idx, first_n, first_err) = &failed[0];
    let msg = format!(
        "highly_variable_genes(flavor=\"seurat_v3\"): skmisc.loess fit failed on \
         {n_failed} of {n_total} batches — these batches are excluded from the \
         per-batch HVG ranking; the remaining {n_valid} proceed normally. First \
         failure: batch index {first_idx} (n={first_n} cells) — {first_err}. \
         Common causes: very small batches, near-collinear log-mean / log-variance, \
         or many zero-variance genes within a batch. To avoid this, prefer \
         dropping or coarsening batch_key (a high-cardinality key such as a \
         per-dataset id produces many tiny, singular batches); or switch to \
         flavor=\"seurat\" post-normalize. Note pyscx.accel.filter_genes(min_cells=10) \
         only helps the no-batch_key global fit, not the per-batch singularity. \
         (Full per-batch detail is recorded in adata.uns[\"hvg\"][\"loess_failed_batches\"].)",
        n_failed = failed.len(),
        n_valid = n_total.saturating_sub(failed.len()),
    );
    warnings.call_method1(
        "warn",
        (msg, py.get_type::<pyo3::exceptions::PyUserWarning>()),
    )?;
    Ok(())
}

/// Record the per-batch loess-failure detail on
/// `adata.uns["hvg"]["loess_failed_batches"]` as a list of `[batch_idx, n_cells]`
/// pairs (the summary UserWarning only names the first failure). Mirrors scanpy
/// storing HVG metadata under `uns["hvg"]`.
///
/// Called **once per batched seurat_v3 run**, with `failed` possibly empty.
/// Two invariants this guarantees:
/// - **Merge, don't clobber:** reuse any pre-existing `uns["hvg"]` dict (e.g.
///   metadata a prior `scanpy.pp.highly_variable_genes` wrote) and only set the
///   `loess_failed_batches` key, so sibling metadata survives.
/// - **No stale failures:** the key is rewritten every run to the *current*
///   list, so a clean rerun of the same `AnnData` overwrites a stale list from
///   an earlier failed run (writing `[]` when no batch failed) rather than
///   leaving downstream diagnostics reporting phantom failures.
fn record_hvg_loess_failed_batches(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    failed: &[(usize, usize, String)],
) -> PyResult<()> {
    let uns = adata.getattr("uns")?;
    let hvg_dict = match uns.get_item("hvg") {
        // Present and a dict → merge into it; present but not a dict (unexpected)
        // → replace with a fresh dict; absent → fresh dict.
        Ok(existing) => existing
            .cast_into::<PyDict>()
            .unwrap_or_else(|_| PyDict::new(py)),
        Err(_) => PyDict::new(py),
    };
    let failed_list = pyo3::types::PyList::empty(py);
    for (idx, n, _err) in failed {
        failed_list.append(pyo3::types::PyList::new(py, [*idx, *n])?)?;
    }
    hvg_dict.set_item("loess_failed_batches", failed_list)?;
    uns.set_item("hvg", hvg_dict)?;
    Ok(())
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

/// seurat_v3 flavor: raw count data, loess fit, clipped variance.
#[allow(clippy::too_many_arguments)]
fn hvg_seurat_v3<'py, S: scx_format_io::ShardSource + Sync>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    x_obj: &Bound<'py, PyAny>,
    source: &S,
    n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    flavor: &str,
    _device_id: Option<usize>,
) -> PyResult<()> {
    // Route-at-start visibility (report P1): the streaming mean/var passes below
    // run inside `py.detach`, and on backed / atlas-scale input the CPU-bound
    // shard decode dominates while the GPU waits — so a multi-minute run shows
    // no output and ~0% GPU util, which reads as a hang. Log the shape + device
    // at op start (INFO; enable via `logging.basicConfig(level=logging.INFO)`)
    // so the user can tell "GPU route, decode-bound" from "hung".
    #[cfg(feature = "gpu")]
    let on_gpu = _device_id.is_some();
    #[cfg(not(feature = "gpu"))]
    let on_gpu = false;
    log::info!(
        target: "pyscx.accel",
        "highly_variable_genes: {flavor} over {n_obs}\u{00d7}{n_vars} on {} — streaming \
         mean/var; large/backed input is CPU-decode-bound (GPU may show low utilization), \
         not hung",
        if on_gpu { "gpu" } else { "cpu" },
    );

    // ── 1. Determine batches ────────────────────────────────────────────
    let batches: Vec<Vec<usize>> = match batch_key {
        Some(bk) => {
            let obs = adata.getattr("obs")?;
            let batch_col = obs.get_item(bk)?;
            // Handle both categorical and non-categorical columns:
            // wrap in pd.Categorical() which is a no-op for already-categorical data.
            let pd = py.import("pandas")?;
            let np = py.import("numpy")?;
            let cat = pd.call_method1("Categorical", (&batch_col,))?;
            let codes = cat.getattr("codes")?;
            let cat_codes: Vec<i64> = np
                .call_method1("asarray", (&codes,))?
                .call_method1("astype", ("int64",))?
                .extract()?;
            let n_batches = *cat_codes.iter().max().unwrap_or(&0) as usize + 1;
            let mut groups = vec![vec![]; n_batches];
            for (i, &code) in cat_codes.iter().enumerate() {
                if code >= 0 {
                    groups[code as usize].push(i);
                }
            }
            groups.into_iter().filter(|g| !g.is_empty()).collect()
        }
        None => vec![(0..n_obs).collect()],
    };

    let n_batches_actual = batches.len();

    // Build cell-to-batch mapping for batched streaming
    let mut cell_batch = vec![-1i32; n_obs];
    for (batch_id, batch_cells) in batches.iter().enumerate() {
        for &cell_idx in batch_cells {
            cell_batch[cell_idx] = batch_id as i32;
        }
    }

    // ── 2. Batched streaming mean/var (single pass for ALL batches + global) ──
    // GPU path: per-batch streaming mean/var via the batched device wrapper.
    // The wrapper finalises Bessel-corrected means/variances and the global
    // accumulator on host, matching the CPU formula byte-for-byte.
    #[cfg(feature = "gpu")]
    let batched_stats = if let Some(dev_id) = _device_id {
        py.detach(|| {
            scx_accel::streaming_mean_var_batched_with_device(
                source,
                &cell_batch,
                n_batches_actual,
                "gpu",
                dev_id,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("gpu streaming_mean_var_batched: {e}")))?
    } else {
        py.detach(|| scx_accel::streaming_mean_var_batched(source, &cell_batch, n_batches_actual))
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_batched: {e}")))?
    };
    #[cfg(not(feature = "gpu"))]
    let batched_stats = py
        .detach(|| scx_accel::streaming_mean_var_batched(source, &cell_batch, n_batches_actual))
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_batched: {e}")))?;

    let global_stats = batched_stats.global.clone();

    // ── 3. Per-batch: loess fit → clip_val (in-memory, no I/O) ───────────
    let mut all_clip_vals: Vec<Vec<f64>> = Vec::new();
    let mut batch_estimat_vars: Vec<Vec<f64>> = Vec::new();
    // Tracks which batches had their per-batch loess fit fail (singular /
    // under-determined). Failed batches are skipped in step 5's
    // per-batch normalised-variance computation and excluded from the
    // cross-batch mean and median-rank aggregations below — the contract
    // the UserWarning text promises.
    let mut batch_failed = vec![false; n_batches_actual];
    // Collected `(batch_idx, batch_n, error_string)` for batches whose loess
    // fit raised a singularity `ValueError`. Coalesced into a single summary
    // UserWarning after the loop (user-report F10) instead of one warning per
    // failing batch — a high-cardinality batch_key can fail dozens of batches.
    let mut loess_failed: Vec<(usize, usize, String)> = Vec::new();

    // Hoist the loess import out of the per-batch closure so a missing
    // or broken `skmisc.loess` fails fast with its real type
    // (`ModuleNotFoundError` / `AttributeError`) instead of getting
    // swallowed by the narrow `PyValueError` catch in the closure below.
    let loess_mod = py.import("skmisc.loess")?;
    let loess_cls = loess_mod.getattr("loess")?;

    for (b, batch_cells) in batches.iter().enumerate() {
        let batch_n = batch_cells.len();
        if batch_n < 2 {
            all_clip_vals.push(vec![0.0; n_vars]);
            batch_estimat_vars.push(vec![0.0; n_vars]);
            continue;
        }

        let batch_stats = &batched_stats.per_batch[b];

        // Loess fit via Python (on non-constant genes)
        let mut estimat_var = vec![0.0f64; n_vars];
        let not_const: Vec<bool> = batch_stats.variances.iter().map(|&v| v > 0.0).collect();
        let x_vals: Vec<f64> = batch_stats
            .means
            .iter()
            .zip(not_const.iter())
            .filter(|(_, &nc)| nc)
            .map(|(&m, _)| m.max(1e-300).log10())
            .collect();
        let y_vals: Vec<f64> = batch_stats
            .variances
            .iter()
            .zip(not_const.iter())
            .filter(|(_, &nc)| nc)
            .map(|(&v, _)| v.max(1e-300).log10())
            .collect();

        if x_vals.len() >= 3 {
            let x_arr = numpy::PyArray::from_vec(py, x_vals);
            let y_arr = numpy::PyArray::from_vec(py, y_vals);

            let fit_result: PyResult<Vec<f64>> = (|| -> PyResult<Vec<f64>> {
                let kwargs = PyDict::new(py);
                kwargs.set_item("span", span)?;
                kwargs.set_item("degree", 2)?;
                let model = loess_cls.call((x_arr, y_arr), Some(&kwargs))?;
                model.call_method0("fit")?;
                model
                    .getattr("outputs")?
                    .getattr("fitted_values")?
                    .extract::<Vec<f64>>()
            })();

            match fit_result {
                Ok(fitted) => {
                    let mut fi = 0;
                    for (j, &nc) in not_const.iter().enumerate() {
                        if nc {
                            estimat_var[j] = fitted[fi];
                            fi += 1;
                        }
                    }
                }
                Err(e) if e.is_instance_of::<PyValueError>(py) => {
                    // Singular / under-determined LOESS — `skmisc.loess`
                    // raises `ValueError` ("There are other near
                    // singularities…") on Census-style degenerate
                    // batches. Record the batch and mark it failed so
                    // step 5 and the rank step exclude it from per-batch
                    // and cross-batch aggregation; a single coalesced
                    // summary warning is emitted after the loop. We
                    // cannot rely on `estimat_var == 0` to opt out — a
                    // successful loess fit can legitimately produce zero
                    // entries, and downstream `reg_std_sq = 10^0 = 1` so
                    // a zero `estimat_var` would still pass the
                    // `reg_std_sq > 0` guard.
                    loess_failed.push((b, batch_n, e.to_string()));
                    batch_failed[b] = true;
                }
                Err(e) => {
                    // Any other PyErr (TypeError, RuntimeError,
                    // MemoryError, etc.) is a real environmental issue,
                    // not a benign singularity — propagate so the user
                    // sees the real cause instead of a misleading
                    // "this batch had a singular loess fit" warning.
                    return Err(e);
                }
            }
        }

        // reg_std and clip_val
        let mut clip_val = vec![0.0f64; n_vars];
        let batch_n_f = batch_n as f64;
        let sqrt_n = batch_n_f.sqrt();
        for j in 0..n_vars {
            let reg_std = 10.0f64.powf(estimat_var[j]).sqrt();
            clip_val[j] = reg_std * sqrt_n + batch_stats.means[j];
        }

        all_clip_vals.push(clip_val);
        batch_estimat_vars.push(estimat_var);
    }

    // Coalesce per-batch loess singularities into ONE summary UserWarning
    // (user-report F10) — a high-cardinality batch_key can fail dozens of
    // batches, and one verbatim warning each buries the signal. The full
    // per-batch list is recorded on adata.uns["hvg"]["loess_failed_batches"]
    // for callers who want the complete detail.
    //
    // Record unconditionally: writing the current list every
    // run — `[]` when nothing failed — clears any stale `loess_failed_batches`
    // left on a reused AnnData by an earlier failed run, and merges into (rather
    // than clobbers) any pre-existing uns["hvg"]. The summary warning stays
    // guarded by a non-empty list, and is emitted before the all-failed check
    // below so the diagnostic survives even that error.
    record_hvg_loess_failed_batches(py, adata, &loess_failed)?;
    if !loess_failed.is_empty() {
        emit_hvg_loess_singularity_warning(py, &loess_failed, n_batches_actual)?;
    }

    // Surviving batches (per-batch loess fit succeeded). If all batches
    // failed there's nothing to rank against and dividing by zero in the
    // cross-batch average below would silently produce NaN HVGs — raise
    // so the user sees the summary UserWarning as the cause.
    let n_valid_batches = batch_failed.iter().filter(|&&f| !f).count();
    if n_valid_batches == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "highly_variable_genes(flavor=\"seurat_v3\"): all {n_batches_actual} \
             batches failed skmisc.loess fitting; see the preceding summary \
             UserWarning (and adata.uns[\"hvg\"][\"loess_failed_batches\"]) for \
             per-batch causes."
        )));
    }

    // ── 4. Batched streaming clipped sums (single pass for ALL batches) ──
    #[cfg(feature = "gpu")]
    let all_clipped = if let Some(dev_id) = _device_id {
        py.detach(|| {
            scx_accel::streaming_clip_square_sum_batched_with_device(
                source,
                &cell_batch,
                n_batches_actual,
                &all_clip_vals,
                "gpu",
                dev_id,
            )
        })
        .map_err(|e| {
            PyRuntimeError::new_err(format!("gpu streaming_clip_square_sum_batched: {e}"))
        })?
    } else {
        py.detach(|| {
            scx_accel::streaming_clip_square_sum_batched(
                source,
                &cell_batch,
                n_batches_actual,
                &all_clip_vals,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_batched: {e}")))?
    };
    #[cfg(not(feature = "gpu"))]
    let all_clipped = py
        .detach(|| {
            scx_accel::streaming_clip_square_sum_batched(
                source,
                &cell_batch,
                n_batches_actual,
                &all_clip_vals,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_batched: {e}")))?;

    // ── 5. Compute normalized variance per batch (in-memory) ─────────────
    let mut all_norm_vars: Vec<Vec<f64>> = Vec::new();
    for (b, batch_cells) in batches.iter().enumerate() {
        let batch_n = batch_cells.len();
        if batch_n < 2 || batch_failed[b] {
            // Push a zero row so `all_norm_vars` stays indexable by batch
            // id; the rank step skips these by index via `batch_failed`.
            all_norm_vars.push(vec![0.0; n_vars]);
            continue;
        }

        let batch_stats = &batched_stats.per_batch[b];
        let (ref bcs, ref sbcs) = all_clipped[b];
        let estimat_var = &batch_estimat_vars[b];
        let batch_n_f = batch_n as f64;
        let denom_n = (batch_n_f - 1.0).max(1.0);

        let mut norm_gene_var = vec![0.0f64; n_vars];
        for j in 0..n_vars {
            let reg_std_sq = 10.0f64.powf(estimat_var[j]);
            if reg_std_sq > 0.0 {
                norm_gene_var[j] = (1.0 / (denom_n * reg_std_sq))
                    * (batch_n_f * batch_stats.means[j] * batch_stats.means[j] + sbcs[j]
                        - 2.0 * bcs[j] * batch_stats.means[j]);
            }
        }
        all_norm_vars.push(norm_gene_var);
    }

    // ── 4. Rank genes and select top N ──────────────────────────────────
    // Mean normalized variance across **surviving** batches only. Failed
    // batches contributed a zero row to `all_norm_vars` (see step 5) but
    // must not enter the average — we divide by `n_valid_batches`, not
    // `all_norm_vars.len()`.
    let mut mean_norm_var = vec![0.0f64; n_vars];
    for (b, nv) in all_norm_vars.iter().enumerate() {
        if batch_failed[b] {
            continue;
        }
        for (j, &v) in nv.iter().enumerate() {
            mean_norm_var[j] += v;
        }
    }
    for v in &mut mean_norm_var {
        *v /= n_valid_batches as f64;
    }

    // Multi-batch ranking when more than one batch survived. When only
    // one batch survives (`n_valid_batches == 1`), `mean_norm_var` equals
    // that batch's `norm_gene_var`, so the single-batch ranking branch
    // produces the same HVG mask as a direct 1-batch run on that batch
    // alone — that's the contract the parity test in
    // tests/test_hvg_loess_singularity.py asserts.
    // Per-batch dense ranks (0 = most variable), matching scanpy's
    // `argsort(argsort(-norm_gene_vars))`. Failed batches cast no rank vote and
    // don't count toward `nbatches_hv` / `median_ranks`. A single surviving
    // batch is just the 1-row case of the same algorithm (its `mean_norm_var`
    // equals that batch's `norm_gene_var`), so we use one code path for both.
    let mut batch_ranks: Vec<Vec<usize>> = Vec::new();
    for (b, nv) in all_norm_vars.iter().enumerate() {
        if batch_failed[b] {
            continue;
        }
        let mut indices: Vec<usize> = (0..n_vars).collect();
        indices.sort_by(|&a, &c| {
            nv[c]
                .partial_cmp(&nv[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut rank = vec![0usize; n_vars];
        for (r, &idx) in indices.iter().enumerate() {
            rank[idx] = r;
        }
        batch_ranks.push(rank);
    }

    // scanpy: `num_batches_high_var = sum(rank < n_top_genes, axis=0)`;
    // ranks >= n_top_genes are masked to NaN, then `highly_variable_rank` is
    // the per-gene `np.ma.median` over surviving batches (NaN when the gene is
    // never in any batch's top-N). `np.ma.median` AVERAGES the two middle
    // values for an even count — not the upper-middle element.
    let mut nbatches_hv = vec![0u32; n_vars];
    let mut median_ranks = vec![f64::NAN; n_vars];
    for j in 0..n_vars {
        let mut valid: Vec<f64> = batch_ranks
            .iter()
            .map(|br| br[j])
            .filter(|&r| r < n_top_genes)
            .map(|r| r as f64)
            .collect();
        nbatches_hv[j] = valid.len() as u32;
        if !valid.is_empty() {
            valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = valid.len();
            median_ranks[j] = if m % 2 == 1 {
                valid[m / 2]
            } else {
                0.5 * (valid[m / 2 - 1] + valid[m / 2])
            };
        }
    }

    // Sort with scanpy's key + `na_position="last"`: NaN median-ranks sort
    // after all finite ranks (mapped to +inf here). seurat_v3 sorts by
    // (rank asc, nbatches desc); seurat_v3_paper by (nbatches desc, rank asc).
    let rank_key = |g: usize| {
        if median_ranks[g].is_nan() {
            f64::INFINITY
        } else {
            median_ranks[g]
        }
    };
    let mut gene_order: Vec<usize> = (0..n_vars).collect();
    if flavor == "seurat_v3_paper" {
        gene_order.sort_by(|&a, &b| {
            nbatches_hv[b].cmp(&nbatches_hv[a]).then(
                rank_key(a)
                    .partial_cmp(&rank_key(b))
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
    } else {
        gene_order.sort_by(|&a, &b| {
            rank_key(a)
                .partial_cmp(&rank_key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(nbatches_hv[b].cmp(&nbatches_hv[a]))
        });
    }

    // scanpy: `highly_variable = sorted_index[:n_top_genes]`. The published
    // `highly_variable_rank` is the per-gene median rank (NaN-preserving), NOT
    // the final selection order.
    let mut hvg_mask = vec![false; n_vars];
    for &g in gene_order.iter().take(n_top_genes.min(n_vars)) {
        hvg_mask[g] = true;
    }
    let ranks = median_ranks;

    // ── 5. Write results to adata.var ───────────────────────────────────
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, hvg_mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, global_stats.means))?;
    var.set_item(
        "variances",
        numpy::PyArray::from_vec(py, global_stats.variances),
    )?;
    var.set_item(
        "variances_norm",
        numpy::PyArray::from_vec(py, mean_norm_var),
    )?;
    var.set_item("highly_variable_rank", numpy::PyArray::from_vec(py, ranks))?;
    // scanpy writes `highly_variable_nbatches` only when a batch_key was given.
    if batch_key.is_some() {
        var.set_item(
            "highly_variable_nbatches",
            numpy::PyArray::from_vec(py, nbatches_hv),
        )?;
    }

    // ── 6. Subset if requested ──────────────────────────────────────────
    if subset {
        apply_hvg_subset(py, adata, x_obj, &hvg_mask)?;
    }

    Ok(())
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

/// seurat flavor: log-normalized data, binned dispersion normalization.
#[allow(clippy::too_many_arguments)]
fn hvg_seurat<'py, S: scx_format_io::ShardSource + Sync>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    x_obj: &Bound<'py, PyAny>,
    source: &S,
    _n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    batch_key: Option<&str>,
    subset: bool,
    n_bins: usize,
) -> PyResult<()> {
    // For batched seurat, fall back to scanpy (complex aggregation logic)
    if batch_key.is_some() {
        let sc = py.import("scanpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("n_top_genes", n_top_genes)?;
        kwargs.set_item("flavor", "seurat")?;
        kwargs.set_item("subset", subset)?;
        kwargs.set_item("n_bins", n_bins)?;
        kwargs.set_item("batch_key", batch_key)?;
        sc.getattr("pp")?
            .call_method("highly_variable_genes", (adata,), Some(&kwargs))?;
        return Ok(());
    }

    // ── 1. Streaming mean/var in COUNT space ───────────────────────────
    // scanpy's seurat flavor un-`log1p`s the matrix before computing moments:
    // `x *= ln(base)` (identity for natural-log / no recorded base), then
    // `expm1`. We stream the same count-space moments directly from the
    // (log-transformed) shards. `scale = ln(base)`, or 1.0 when no base is
    // recorded (matches scanpy's `uns.get("log1p", {}).get("base")`).
    let scale = log1p_base_scale(py, adata)?;
    let stats = py
        .detach(|| scx_accel::streaming_mean_var_expm1(source, scale))
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_expm1: {e}")))?;

    // ── 2. Compute dispersion (matching scanpy's seurat flavor) ────────
    // scanpy publishes the LOG dispersion and the LOG1P count-space mean:
    //   mean[mean==0] = 1e-12; dispersion = var/mean;
    //   dispersion[dispersion==0] = NaN; dispersion = log(dispersion);
    //   mean = log1p(mean).
    let mut log_dispersions = vec![f64::NAN; n_vars];
    let mut log_means = vec![0.0f64; n_vars];
    let mut means_for_disp = stats.means.clone();

    for j in 0..n_vars {
        // scanpy: mean[mean == 0] = 1e-12 (before dispersion computation)
        if means_for_disp[j] == 0.0 {
            means_for_disp[j] = 1e-12;
        }
        let disp = stats.variances[j] / means_for_disp[j];
        // scanpy: dispersion[dispersion == 0] = NaN, then log(dispersion)
        if disp > 0.0 {
            log_dispersions[j] = disp.ln();
        }
        // scanpy: mean = log1p(mean) — count-space mean, logged for binning
        // and for the published `means` column.
        log_means[j] = means_for_disp[j].ln_1p();
    }

    // ── 3. Bin by mean, z-score dispersion within bins (via Python) ────
    // Clone: `log_means` / `log_dispersions` are also published to `var` below.
    let log_means_np = numpy::PyArray::from_vec(py, log_means.clone());
    let log_disp_np = numpy::PyArray::from_vec(py, log_dispersions.clone());

    let helpers = py.import("pyscx._hvg_helpers")?;
    let dispersions_norm: Vec<f64> = helpers
        .call_method1(
            "binned_dispersion_norm",
            (log_means_np, log_disp_np, n_bins),
        )?
        .extract()?;

    // ── 4. Select top genes by normalized dispersion ────────────────────
    // scanpy selects via `nan_to_num(dispersion_norm, nan=-inf) >= cutoff`, so
    // NaN dispersions (zero-dispersion genes) must sort LAST and never be
    // selected. Order NaN as -inf.
    let mut indices: Vec<usize> = (0..n_vars).collect();
    indices.sort_by(|&a, &b| {
        let va = if dispersions_norm[a].is_nan() {
            f64::NEG_INFINITY
        } else {
            dispersions_norm[a]
        };
        let vb = if dispersions_norm[b].is_nan() {
            f64::NEG_INFINITY
        } else {
            dispersions_norm[b]
        };
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mask = vec![false; n_vars];
    for &g in indices.iter().take(n_top_genes.min(n_vars)) {
        // Never select a NaN-dispersion gene (scanpy's -inf floor).
        if !dispersions_norm[g].is_nan() {
            mask[g] = true;
        }
    }

    // ── 5. Write results to adata.var ───────────────────────────────────
    // scanpy publishes `means = log1p(count-space mean)` and
    // `dispersions = log(var/mean)` (both may be NaN for zero-dispersion genes).
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, log_means))?;
    var.set_item("dispersions", numpy::PyArray::from_vec(py, log_dispersions))?;
    var.set_item(
        "dispersions_norm",
        numpy::PyArray::from_vec(
            py,
            dispersions_norm
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>(),
        ),
    )?;

    // ── 6. Subset if requested ──────────────────────────────────────────
    if subset {
        apply_hvg_subset(py, adata, x_obj, &mask)?;
    }

    Ok(())
}

/// Apply HVG subset: set column projection on X (and layers), slice var.
///
/// Order: update X col_projection + layers FIRST so shapes match,
/// then set `_var` (bypassing AnnData shape validation).
fn apply_hvg_subset(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    x_obj: &Bound<'_, PyAny>,
    hvg_mask: &[bool],
) -> PyResult<()> {
    let mask_arr = numpy::PyArray::from_vec(py, hvg_mask.to_vec());

    if let Ok(backed) = x_obj.cast::<ScxBackedSparseDataset>() {
        let new_col_indices: Vec<u32> = match backed.borrow().col_projection() {
            Some(existing) => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        // Update X and layers FIRST so shapes are consistent
        backed
            .borrow_mut()
            .set_col_projection(new_col_indices.clone());
        update_layers_col_projection(adata, &new_col_indices, false)?;

        // Slice var via _var: AnnData's public var setter validates
        // len(value) == self.n_vars, where n_vars is derived from the current
        // _var DataFrame. Since we're changing the column count, the public
        // setter would reject the new (shorter) DataFrame. Setting _var
        // directly is the same approach used by anndata's own _inplace_subset_var.
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        adata.setattr("_var", filtered_var)?;
    } else if let Ok(lazy) = x_obj.cast::<ScxLazyTransformedDataset>() {
        let new_col_indices: Vec<u32> = match lazy.borrow().col_projection() {
            Some(existing) => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        lazy.borrow_mut()
            .set_col_projection(new_col_indices.clone());
        update_layers_col_projection(adata, &new_col_indices, false)?;

        // See comment above in backed branch for why _var is used.
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        adata.setattr("_var", filtered_var)?;
    } else {
        // In-memory scipy/dense X (the native in-memory HVG path). There is no
        // col_projection to update — slice the AnnData in place. anndata's own
        // `_inplace_subset_var` handles X, var, varm, and layers consistently,
        // and preserves the result columns we just wrote to `var` (the boolean
        // mask selects the kept rows of the already-updated frame).
        adata.call_method1("_inplace_subset_var", (mask_arr,))?;
    }

    Ok(())
}

/// Loess fit on (log10 mean, log10 var) for non-constant genes — the
/// seurat_v3 dispersion regression. Returns per-gene `estimat_var`
/// (log10 fitted variance); constant genes and the too-few-points case
/// stay 0.0. Shared by the backed and lazy CSC pipelines.
fn csc_loess_estimat_var(
    py: Python<'_>,
    means: &[f64],
    variances: &[f64],
    span: f64,
) -> PyResult<Vec<f64>> {
    let n_vars = means.len();
    let mut estimat_var = vec![0.0f64; n_vars];
    let not_const: Vec<bool> = variances.iter().map(|&v| v > 0.0).collect();
    let x_vals: Vec<f64> = means
        .iter()
        .zip(not_const.iter())
        .filter(|(_, &nc)| nc)
        .map(|(&m, _)| m.max(1e-300).log10())
        .collect();
    let y_vals: Vec<f64> = variances
        .iter()
        .zip(not_const.iter())
        .filter(|(_, &nc)| nc)
        .map(|(&v, _)| v.max(1e-300).log10())
        .collect();

    if x_vals.len() >= 3 {
        let x_arr = numpy::PyArray::from_vec(py, x_vals);
        let y_arr = numpy::PyArray::from_vec(py, y_vals);
        let loess_mod = py.import("skmisc.loess")?;
        let loess_cls = loess_mod.getattr("loess")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("span", span)?;
        kwargs.set_item("degree", 2)?;
        let model = loess_cls.call((x_arr, y_arr), Some(&kwargs))?;
        model.call_method0("fit")?;
        let fitted: Vec<f64> = model
            .getattr("outputs")?
            .getattr("fitted_values")?
            .extract()?;
        let mut fi = 0;
        for (j, &nc) in not_const.iter().enumerate() {
            if nc {
                estimat_var[j] = fitted[fi];
                fi += 1;
            }
        }
    }
    Ok(estimat_var)
}

/// clip_val per gene = `reg_std * sqrt(n) + mean` (seurat_v3).
fn csc_clip_val(means: &[f64], estimat_var: &[f64], n_obs: usize) -> Vec<f64> {
    let sqrt_n = (n_obs as f64).sqrt();
    means
        .iter()
        .zip(estimat_var.iter())
        .map(|(&mean, &ev)| 10.0f64.powf(ev).sqrt() * sqrt_n + mean)
        .collect()
}

/// Steps 5–8 of single-batch seurat_v3: normalized variance, rank, write
/// `adata.var`, optional subset. Shared by the backed and lazy CSC paths.
#[allow(clippy::too_many_arguments)]
fn csc_finish_seurat_v3(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    x_obj: &Bound<'_, PyAny>,
    n_obs: usize,
    n_vars: usize,
    means: Vec<f64>,
    variances: Vec<f64>,
    estimat_var: Vec<f64>,
    bcs: Vec<f64>,
    sbcs: Vec<f64>,
    n_top_genes: usize,
    subset: bool,
) -> PyResult<()> {
    let n_f = n_obs as f64;
    let denom_n = (n_f - 1.0).max(1.0);
    let mut norm_gene_var = vec![0.0f64; n_vars];
    for j in 0..n_vars {
        let reg_std_sq = 10.0f64.powf(estimat_var[j]);
        if reg_std_sq > 0.0 {
            norm_gene_var[j] = (1.0 / (denom_n * reg_std_sq))
                * (n_f * means[j] * means[j] + sbcs[j] - 2.0 * bcs[j] * means[j]);
        }
    }

    // Single-batch ranking: sort by normalized variance desc.
    let mut indices: Vec<usize> = (0..n_vars).collect();
    indices.sort_by(|&a, &b| {
        norm_gene_var[b]
            .partial_cmp(&norm_gene_var[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut hvg_mask = vec![false; n_vars];
    let mut ranks = vec![f64::NAN; n_vars];
    for (r, &g) in indices.iter().enumerate().take(n_top_genes.min(n_vars)) {
        hvg_mask[g] = true;
        ranks[g] = r as f64;
    }

    // Write to adata.var (matches CSR seurat_v3 schema).
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, hvg_mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, means))?;
    var.set_item("variances", numpy::PyArray::from_vec(py, variances))?;
    var.set_item(
        "variances_norm",
        numpy::PyArray::from_vec(py, norm_gene_var),
    )?;
    var.set_item("highly_variable_rank", numpy::PyArray::from_vec(py, ranks))?;

    if subset {
        apply_hvg_subset(py, adata, x_obj, &hvg_mask)?;
    }
    Ok(())
}

/// Device-dispatched per-column mean/var on a `Sync` CSC source: GPU CSC
/// reduce when `device_id` is `Some` (no `atomicAdd`), else CPU CSC.
#[cfg(feature = "gpu")]
fn csc_mean_var_dispatch<S: scx_format_io::ColumnShardSource + Sync>(
    py: Python<'_>,
    source: &S,
    device_id: Option<usize>,
) -> PyResult<scx_accel::HvgStats> {
    match device_id {
        Some(id) => py
            .detach(|| scx_accel::streaming_mean_var_csc_with_device(source, "gpu", id))
            .map_err(|e| PyRuntimeError::new_err(format!("gpu streaming_mean_var_csc: {e}"))),
        None => scx_accel::streaming_mean_var_csc(source)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_csc: {e}"))),
    }
}

/// Device-dispatched per-column clipped sums on a `Sync` CSC source.
#[cfg(feature = "gpu")]
fn csc_clip_dispatch<S: scx_format_io::ColumnShardSource + Sync>(
    py: Python<'_>,
    source: &S,
    clip_val: &[f64],
    device_id: Option<usize>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    match device_id {
        Some(id) => {
            py.detach(|| {
                scx_accel::streaming_clip_square_sum_csc_with_device(source, clip_val, "gpu", id)
            })
            .map_err(|e| PyRuntimeError::new_err(format!("gpu streaming_clip_square_sum_csc: {e}")))
        }
        None => scx_accel::streaming_clip_square_sum_csc(source, clip_val)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_csc: {e}"))),
    }
}

/// Single-batch seurat_v3 HVG via the CSC sidecar.
///
/// Same numerics as `hvg_seurat_v3` for the single-batch case, but pulls
/// per-column mean/var and clipped sums from the gene-major
/// `ColumnShardSource`. With `device_id = Some(gid)` the two reduction
/// passes run the GPU CSC reduce kernels (one block per gene, no
/// `atomicAdd`) — route `gpu_csc_v3`; `None` runs the CPU CSC kernels.
/// GPU CSC reduce is **backed-only** (raw counts): the lazy-transformed
/// branch always runs CPU since its `&dyn` source is not `Sync`.
fn hvg_seurat_v3_csc(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_top_genes: usize,
    span: f64,
    subset: bool,
    flavor: &str,
    device_id: Option<usize>,
) -> PyResult<()> {
    use scx_format_io::ColumnShardSource;
    let _ = flavor; // single-batch: seurat_v3 / seurat_v3_paper share ordering.

    let x = adata.getattr("X")?;

    // ── Backed dataset: concrete `Arc<BackedCscReader>` (Send + Sync), so
    //    the GPU CSC reduce path (which decodes on a worker thread) is
    //    reachable. Mirror the same capability gate as `as_column_source`. ──
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();
        if backed_ref.kept_to_global.is_some() {
            return Err(PyRuntimeError::new_err(
                "CSC requested but unavailable: a row deletion vector is active. \
                 Pass `prefer_format='csr'`.",
            ));
        }
        let csc_reader = backed_ref
            .backed_csc
            .as_ref()
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "CSC requested but unavailable: file has no CSC sidecar. \
                     Re-import with `csc=\"always\"` or pass `prefer_format='csr'`.",
                )
            })?
            .clone();
        drop(backed_ref);

        let n_obs = csc_reader.n_obs();
        let n_vars = csc_reader.n_vars();

        #[cfg(feature = "gpu")]
        let stats = csc_mean_var_dispatch(py, csc_reader.as_ref(), device_id)?;
        #[cfg(not(feature = "gpu"))]
        let stats = {
            let _ = device_id;
            scx_accel::streaming_mean_var_csc(csc_reader.as_ref())
                .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_csc: {e}")))?
        };

        let estimat_var = csc_loess_estimat_var(py, &stats.means, &stats.variances, span)?;
        let clip_val = csc_clip_val(&stats.means, &estimat_var, n_obs);

        #[cfg(feature = "gpu")]
        let (bcs, sbcs) = csc_clip_dispatch(py, csc_reader.as_ref(), &clip_val, device_id)?;
        #[cfg(not(feature = "gpu"))]
        let (bcs, sbcs) = scx_accel::streaming_clip_square_sum_csc(csc_reader.as_ref(), &clip_val)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_csc: {e}")))?;

        return csc_finish_seurat_v3(
            py,
            adata,
            &x,
            n_obs,
            n_vars,
            stats.means,
            stats.variances,
            estimat_var,
            bcs,
            sbcs,
            n_top_genes,
            subset,
        );
    }

    // ── Lazy-transformed dataset: CPU CSC only (the `&dyn` column source is
    //    not `Sync`, and GPU CSC reduce is a raw-counts / backed feature). ──
    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let lazy_src = lazy_ref.as_column_source().ok_or_else(|| {
            PyRuntimeError::new_err(
                "CSC requested but unavailable: file has no CSC sidecar, \
                 the transform chain contains a non-column-local op \
                 (NormalizeTotal or RowScale), or a row deletion vector \
                 is active. Pass `prefer_format='csr'` to use the CSR path.",
            )
        })?;
        let n_obs = lazy_src.n_obs();
        let n_vars = lazy_src.n_vars();
        let stats = scx_accel::streaming_mean_var_csc(&lazy_src)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_csc: {e}")))?;
        let estimat_var = csc_loess_estimat_var(py, &stats.means, &stats.variances, span)?;
        let clip_val = csc_clip_val(&stats.means, &estimat_var, n_obs);
        let (bcs, sbcs) = scx_accel::streaming_clip_square_sum_csc(&lazy_src, &clip_val)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_csc: {e}")))?;
        return csc_finish_seurat_v3(
            py,
            adata,
            &x,
            n_obs,
            n_vars,
            stats.means,
            stats.variances,
            estimat_var,
            bcs,
            sbcs,
            n_top_genes,
            subset,
        );
    }

    Err(PyRuntimeError::new_err(
        "prefer_format='csc' requires adata.X to be ScxBackedSparseDataset \
         or ScxLazyTransformedDataset (got a regular scipy/dense matrix)",
    ))
}
