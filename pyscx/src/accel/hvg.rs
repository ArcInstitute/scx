//! Highly variable gene selection — seurat_v3 and seurat flavors.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};

use super::filtering::update_layers_col_projection;
use super::util::extract_materialized_csr;

/// Single-shard [`scx_format::ShardSource`] adapter over a borrowed in-memory
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

impl scx_format::ShardSource for InMemoryCsrSource<'_> {
    fn n_shards(&self) -> usize {
        1
    }
    fn n_obs(&self) -> usize {
        self.csr.n_rows()
    }
    fn n_vars(&self) -> usize {
        self.csr.n_cols()
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format::Result<scx_sparse::ScxCsr> {
        if shard_idx != 0 {
            return Err(scx_format::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: 1,
            });
        }
        // One clone per pass (mean/var, then clipped-sum). Acceptable for a
        // first cut; a borrowed-slice variant (cf. `BorrowedCsrSource`) can
        // remove it later if in-memory HVG RSS becomes a concern.
        Ok(self.csr.clone())
    }
    fn max_shard_rows(&self) -> scx_format::Result<usize> {
        Ok(self.csr.n_rows())
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
///         seurat_v3 only — multi-batch and seurat flavor raise on
///         CSC. Mutually exclusive with `device != "cpu"`.
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
    if prefer_format == "csc" {
        // CSC HVG only handles single-batch seurat_v3 on CPU. Reject
        // mismatched configurations with a clear message rather than
        // silently falling back, since the user has explicitly opted in.
        if batch_key.is_some() {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' for HVG only supports single-batch mode; \
                 pass batch_key=None or use prefer_format='csr'",
            ));
        }
        if !matches!(flavor, "seurat_v3" | "seurat_v3_paper") {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' for HVG only supports flavor='seurat_v3' \
                 (or 'seurat_v3_paper'); use prefer_format='csr' for 'seurat'",
            ));
        }
        if device != "cpu" && device != "auto" {
            return Err(PyRuntimeError::new_err(format!(
                "prefer_format='csc' for HVG is mutually exclusive with device={device:?}; \
                 set device='cpu' (or 'auto') or use prefer_format='csr'"
            )));
        }
        // CSC has no GPU kernel and runs CPU-only: record the cpu_csc route.
        super::route::write_accel_route(
            py,
            adata,
            "highly_variable_genes",
            &super::route::hvg_exec_info(device, false, true),
        )?;
        return hvg_seurat_v3_csc(py, adata, n_top_genes, span, subset, flavor);
    }
    let resolved = super::gpu::resolve_device(device)?;
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

    // Record the planned route on adata.uns["scx_accel"]["highly_variable_genes"]
    // before dispatch. Only the seurat_v3 family has a GPU kernel; "seurat" and
    // "cell_ranger" are CPU-only (gpu_eligible=false records
    // UnsupportedInputLayout on a GPU host rather than implying CUDA was
    // absent). This single stamp covers the backed / lazy / in-memory native
    // paths and the scanpy fallback — they share `device` + `flavor`.
    //
    // INVARIANT (pre-dispatch stamp): safe to stamp *before* dispatch only
    // because (a) `hvg_gpu_eligible` mirrors the same flavor gate that derives
    // `effective_gpu_id` above (GPU runs iff the flavor is seurat_v3 family and
    // CUDA is present), and (b) the streaming seurat_v3 GPU kernel propagates
    // errors via `.map_err(..)?` rather than silently falling back to CPU — so
    // the recorded `gpu_csr_v1` route always reflects the code that ran. If a
    // silent GPU→CPU runtime fallback is ever added, stamp *after* dispatch on
    // the branch that ran (see umap.rs) or this gate will false-pass.
    let hvg_gpu_eligible = matches!(flavor, "seurat_v3" | "seurat_v3_paper");
    super::route::write_accel_route(
        py,
        adata,
        "highly_variable_genes",
        &super::route::hvg_exec_info(device, hvg_gpu_eligible, false),
    )?;

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
fn hvg_on_source<'py, S: scx_format::ShardSource + Sync>(
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

/// Emit a UserWarning when `skmisc.loess.fit()` raises on a single batch.
///
/// Names the failing batch (index + cell count) and the upstream error string.
/// The failing batch is then excluded from the per-batch normalised-variance
/// ranking — semantics identical to a batch with too few non-constant genes.
///
/// Remedies are ordered by what actually helps the per-batch case: the
/// singularity is driven by small / near-collinear batches, so dropping or
/// coarsening `batch_key` is the effective fix. `filter_genes(min_cells=10)`
/// only helps the single global fit (no `batch_key`); it cannot make a tiny
/// batch's log-mean / log-variance regression well-conditioned.
fn emit_hvg_loess_singularity_warning(
    py: Python<'_>,
    batch_idx: usize,
    batch_n: usize,
    err: &PyErr,
) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    let msg = format!(
        "highly_variable_genes(flavor=\"seurat_v3\"): skmisc.loess fit failed on \
         batch index {batch_idx} (n={batch_n} cells) — {err}. This batch will be \
         excluded from the per-batch HVG ranking; other batches proceed normally. \
         Common causes: very small batches, near-collinear log-mean / log-variance, \
         or many zero-variance genes within this batch. To avoid this, prefer \
         dropping or coarsening batch_key (a high-cardinality key such as a \
         per-dataset id produces many tiny, singular batches); or switch to \
         flavor=\"seurat\" post-normalize. Note pyscx.accel.filter_genes(min_cells=10) \
         only helps the no-batch_key global fit, not the per-batch singularity."
    );
    warnings.call_method1(
        "warn",
        (msg, py.get_type::<pyo3::exceptions::PyUserWarning>()),
    )?;
    Ok(())
}

/// Build a full-dataset `LazyShardSource` for the backed / lazy dispatch.
///
/// Per-batch filtering is handled downstream via the `cell_batch` array passed
/// to the batched streaming kernels, so this always builds the whole-dataset
/// source (the previous `batch_indices` branch was unused).
fn build_shard_source(
    reader: &Arc<scx_format::BackedCsrReader>,
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
fn hvg_seurat_v3<'py, S: scx_format::ShardSource + Sync>(
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
                    // batches. Warn naming the batch and mark it failed
                    // so step 5 and the rank step exclude it from
                    // per-batch and cross-batch aggregation. We cannot
                    // rely on `estimat_var == 0` to opt out — a
                    // successful loess fit can legitimately produce zero
                    // entries, and downstream `reg_std_sq = 10^0 = 1` so
                    // a zero `estimat_var` would still pass the
                    // `reg_std_sq > 0` guard.
                    emit_hvg_loess_singularity_warning(py, b, batch_n, &e)?;
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

    // Surviving batches (per-batch loess fit succeeded). If all batches
    // failed there's nothing to rank against and dividing by zero in the
    // cross-batch average below would silently produce NaN HVGs — raise
    // so the user sees the per-batch UserWarnings as the cause.
    let n_valid_batches = batch_failed.iter().filter(|&&f| !f).count();
    if n_valid_batches == 0 {
        return Err(PyRuntimeError::new_err(format!(
            "highly_variable_genes(flavor=\"seurat_v3\"): all {n_batches_actual} \
             batches failed skmisc.loess fitting; see prior UserWarnings for \
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
    let (hvg_mask, ranks) = if n_valid_batches > 1 {
        // Per-batch ranks: for each surviving batch, rank genes by
        // normalized variance (descending). Failed batches are skipped
        // so they neither cast a rank vote nor count toward
        // `nbatches_hv` / `median_ranks`.
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

        // Count in how many batches each gene is in top n_top_genes
        let mut nbatches_hv = vec![0usize; n_vars];
        let mut median_ranks = vec![f64::NAN; n_vars];
        for j in 0..n_vars {
            let ranks_j: Vec<usize> = batch_ranks.iter().map(|br| br[j]).collect();
            nbatches_hv[j] = ranks_j.iter().filter(|&&r| r < n_top_genes).count();
            // Median of ranks where gene is in top n_top_genes
            let mut valid: Vec<f64> = ranks_j
                .iter()
                .filter(|&&r| r < n_top_genes)
                .map(|&r| r as f64)
                .collect();
            if !valid.is_empty() {
                valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
                median_ranks[j] = valid[valid.len() / 2];
            }
        }

        // Sort genes: by nbatches (desc), then median_rank (asc)
        let mut gene_order: Vec<usize> = (0..n_vars).collect();
        if flavor == "seurat_v3_paper" {
            gene_order.sort_by(|&a, &b| {
                nbatches_hv[b].cmp(&nbatches_hv[a]).then(
                    median_ranks[a]
                        .partial_cmp(&median_ranks[b])
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            });
        } else {
            gene_order.sort_by(|&a, &b| {
                median_ranks[a]
                    .partial_cmp(&median_ranks[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(nbatches_hv[b].cmp(&nbatches_hv[a]))
            });
        }

        let mut mask = vec![false; n_vars];
        let mut rank_out = vec![f64::NAN; n_vars];
        for (r, &g) in gene_order.iter().enumerate().take(n_top_genes.min(n_vars)) {
            mask[g] = true;
            rank_out[g] = r as f64;
        }
        (mask, rank_out)
    } else {
        // Single batch: simple rank by normalized variance descending
        let mut indices: Vec<usize> = (0..n_vars).collect();
        indices.sort_by(|&a, &b| {
            mean_norm_var[b]
                .partial_cmp(&mean_norm_var[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut mask = vec![false; n_vars];
        let mut rank_out = vec![f64::NAN; n_vars];
        for (r, &g) in indices.iter().enumerate().take(n_top_genes.min(n_vars)) {
            mask[g] = true;
            rank_out[g] = r as f64;
        }
        (mask, rank_out)
    };

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

    // ── 6. Subset if requested ──────────────────────────────────────────
    if subset {
        apply_hvg_subset(py, adata, x_obj, &hvg_mask)?;
    }

    Ok(())
}

/// seurat flavor: log-normalized data, binned dispersion normalization.
#[allow(clippy::too_many_arguments)]
fn hvg_seurat<'py, S: scx_format::ShardSource + Sync>(
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

    // ── 1. Streaming mean/var ───────────────────────────────────────────
    let stats = py
        .detach(|| scx_accel::streaming_mean_var(source))
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var: {e}")))?;

    // ── 2. Compute dispersion (matching scanpy's seurat flavor) ────────
    let mut dispersions = vec![0.0f64; n_vars];
    let mut log_dispersions = vec![f64::NAN; n_vars];
    let mut log_means = vec![0.0f64; n_vars];
    let mut means_for_disp = stats.means.clone();

    for j in 0..n_vars {
        // scanpy: mean[mean == 0] = 1e-12 (before dispersion computation)
        if means_for_disp[j] == 0.0 {
            means_for_disp[j] = 1e-12;
        }
        dispersions[j] = stats.variances[j] / means_for_disp[j];
        // scanpy: dispersion[dispersion == 0] = NaN, then log(dispersion)
        if dispersions[j] > 0.0 {
            log_dispersions[j] = dispersions[j].ln();
        } else {
            dispersions[j] = f64::NAN;
        }
        // scanpy: mean = log1p(mean) — overwrite mean with log1p for binning
        log_means[j] = (means_for_disp[j] + 1.0).ln();
    }

    // ── 3. Bin by mean, z-score dispersion within bins (via Python) ────
    let log_means_np = numpy::PyArray::from_vec(py, log_means);
    let log_disp_np = numpy::PyArray::from_vec(py, log_dispersions);

    let helpers = py.import("pyscx._hvg_helpers")?;
    let dispersions_norm: Vec<f64> = helpers
        .call_method1(
            "binned_dispersion_norm",
            (log_means_np, log_disp_np, n_bins),
        )?
        .extract()?;

    // ── 4. Select top genes by normalized dispersion ────────────────────
    let mut indices: Vec<usize> = (0..n_vars).collect();
    indices.sort_by(|&a, &b| {
        dispersions_norm[b]
            .partial_cmp(&dispersions_norm[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mask = vec![false; n_vars];
    for &g in indices.iter().take(n_top_genes.min(n_vars)) {
        mask[g] = true;
    }

    // ── 5. Write results to adata.var ───────────────────────────────────
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, stats.means))?;
    var.set_item("dispersions", numpy::PyArray::from_vec(py, dispersions))?;
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
        update_layers_col_projection(adata, &new_col_indices)?;

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
        update_layers_col_projection(adata, &new_col_indices)?;

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

/// Single-batch seurat_v3 HVG via the CSC dispatch.
///
/// Same numerics as `hvg_seurat_v3` for the single-batch case, but
/// pulls per-column mean/var and clipped sums from `ColumnShardSource`
/// instead of the CSR-side `streaming_mean_var_batched`. The loess fit
/// (Python `skmisc.loess`), ranking, and result-writing logic are
/// identical to the CSR path — duplicated rather than abstracted to
/// keep the CSR side untouched.
fn hvg_seurat_v3_csc(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_top_genes: usize,
    span: f64,
    subset: bool,
    flavor: &str,
) -> PyResult<()> {
    use scx_format::ColumnShardSource;

    let x = adata.getattr("X")?;

    // Helper closure that runs the entire CSC seurat_v3 pipeline against
    // a `&dyn ColumnShardSource`. Used by both the backed and lazy
    // branches below to share the kernel invocations and the post-
    // processing (loess, ranking, var-writing).
    let run_pipeline = |source: &dyn ColumnShardSource, x_obj: &Bound<'_, PyAny>| -> PyResult<()> {
        let n_obs = source.n_obs();
        let n_vars = source.n_vars();

        // ── 1. Single-pass per-column mean / var ────────────────────
        // Note: `&dyn ColumnShardSource` is not `Send`, so this Rust call
        // runs with the GIL held. Wrapping requires monomorphizing on the
        // concrete reader type (BackedCscReader / LazyShardSource).
        let stats = scx_accel::streaming_mean_var_csc(source)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_csc: {e}")))?;

        // ── 2. Loess fit on (log10 mean, log10 var) for non-constant
        //       genes — identical to the CSR seurat_v3 fit.
        let mut estimat_var = vec![0.0f64; n_vars];
        let not_const: Vec<bool> = stats.variances.iter().map(|&v| v > 0.0).collect();
        let x_vals: Vec<f64> = stats
            .means
            .iter()
            .zip(not_const.iter())
            .filter(|(_, &nc)| nc)
            .map(|(&m, _)| m.max(1e-300).log10())
            .collect();
        let y_vals: Vec<f64> = stats
            .variances
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

        // ── 3. clip_val per gene — `reg_std * sqrt(n) + mean`.
        let mut clip_val = vec![0.0f64; n_vars];
        let n_f = n_obs as f64;
        let sqrt_n = n_f.sqrt();
        for j in 0..n_vars {
            let reg_std = 10.0f64.powf(estimat_var[j]).sqrt();
            clip_val[j] = reg_std * sqrt_n + stats.means[j];
        }

        // ── 4. Single-pass per-column clipped sum / sum_sq.
        let (bcs, sbcs) = scx_accel::streaming_clip_square_sum_csc(source, &clip_val)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_csc: {e}")))?;

        // ── 5. Compute normalized variance per gene.
        let denom_n = (n_f - 1.0).max(1.0);
        let mut norm_gene_var = vec![0.0f64; n_vars];
        for j in 0..n_vars {
            let reg_std_sq = 10.0f64.powf(estimat_var[j]);
            if reg_std_sq > 0.0 {
                norm_gene_var[j] = (1.0 / (denom_n * reg_std_sq))
                    * (n_f * stats.means[j] * stats.means[j] + sbcs[j]
                        - 2.0 * bcs[j] * stats.means[j]);
            }
        }

        // ── 6. Single-batch ranking: sort by normalized variance desc.
        //       (Multi-batch logic is unreachable here — gated upstream.)
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
        let _ = flavor; // single-batch path: seurat_v3 / seurat_v3_paper share ordering.

        // ── 7. Write to adata.var (matches CSR seurat_v3 schema).
        let var = adata.getattr("var")?;
        var.set_item(
            "highly_variable",
            numpy::PyArray::from_vec(py, hvg_mask.clone()),
        )?;
        var.set_item("means", numpy::PyArray::from_vec(py, stats.means))?;
        var.set_item("variances", numpy::PyArray::from_vec(py, stats.variances))?;
        var.set_item(
            "variances_norm",
            numpy::PyArray::from_vec(py, norm_gene_var),
        )?;
        var.set_item("highly_variable_rank", numpy::PyArray::from_vec(py, ranks))?;

        // ── 8. Optionally subset adata to HVG.
        if subset {
            apply_hvg_subset(py, adata, x_obj, &hvg_mask)?;
        }
        Ok(())
    };

    // Dispatch: backed yields a borrowed `&dyn`, lazy yields an owned
    // `LazyShardSource` (we then borrow from it).
    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();
        let source = backed_ref.as_column_source().ok_or_else(|| {
            PyRuntimeError::new_err(
                "CSC requested but unavailable: file has no CSC sidecar, \
                 or a row deletion vector is active. Re-import with \
                 `csc=\"always\"` or pass `prefer_format='csr'`.",
            )
        })?;
        return run_pipeline(source, &x);
    }

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
        return run_pipeline(&lazy_src, &x);
    }

    Err(PyRuntimeError::new_err(
        "prefer_format='csc' requires adata.X to be ScxBackedSparseDataset \
         or ScxLazyTransformedDataset (got a regular scipy/dense matrix)",
    ))
}
