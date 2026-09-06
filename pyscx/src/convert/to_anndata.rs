// SCX -> AnnData assembly (eager and backed).
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use numpy::PyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use scx_format_io::section::SectionType;
use scx_format_io::ScxReader;

use crate::to_pyerr;

use super::*;

/// Default memory budget for the eager [`to_anndata`] full-assembly
/// path (Phase 4d). Estimated bytes above this threshold trigger a
/// `UserWarning` that recommends `to_anndata(backed=True)` or
/// `pyscx.open(path).query()`. Assembly still proceeds — the warning
/// is advisory.
const DEFAULT_EAGER_MEMORY_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Per-slot key filters for `obsp` / `varp` / `varm`, plus the `raw` opt-out.
///
/// `layers=` and `obsm=` have carried their own filter parameters since they
/// were added; these three had none, so the only way to avoid decoding an
/// `n_obs × n_obs` kNN graph was not to touch `adata.obsp` at all — and
/// anndata touches it for you, since `AlignedMappingProperty.__get__` builds
/// an `AlignedActual` that validates (and therefore decodes) every entry of
/// the slot on first property access.
///
/// Each field follows `obsm=`'s contract exactly: `None` = every key on disk,
/// `Some([])` = none, `Some(keys)` = that subset, unknown key → `KeyError`.
/// `raw` gates the three `has_raw()` sites; `false` skips both the rebuild and
/// the `DroppedRaw` notice.
///
/// Grouped rather than threaded as four more positional parameters: the four
/// functions this passes through already carry 11–16 of them under
/// `#[allow(clippy::too_many_arguments)]`.
#[derive(Clone, Copy)]
pub(crate) struct SlotFilters<'a> {
    pub obsp: Option<&'a [String]>,
    pub varp: Option<&'a [String]>,
    pub varm: Option<&'a [String]>,
    pub raw: bool,
}

impl Default for SlotFilters<'_> {
    /// Today's behaviour: every key of every slot, and raw reconstructed (or
    /// dropped with its notice) as before. Hand-written rather than derived
    /// because `raw` must default to `true` — a derived `false` would silently
    /// disable raw at every call site that has not opted in.
    fn default() -> Self {
        Self {
            obsp: None,
            varp: None,
            varm: None,
            raw: true,
        }
    }
}

/// Which matrices [`to_anndata_with_layers`] builds into the `AnnData` it
/// returns, and which it leaves to its caller.
///
/// Three named states rather than a pair of booleans, because each one changes
/// four things at once (whether `X` is decoded, whether eager `layers` are, and
/// therefore whether a `shape=` has to be supplied and where `.raw` is
/// attached) and a boolean pair cannot say which combinations are meaningful.
#[derive(Clone, Copy)]
pub(crate) enum MatrixMode {
    /// Everything inline: `X`, eager `layers`, and `.raw`.
    Eager,
    /// No `X`; `layers` and `.raw` unchanged. `to_gpu_anndata` decodes X
    /// straight onto the device and assigns it afterwards.
    SkipX,
    /// No `X` and no `layers`, an explicit `shape=`, and `.raw` deferred to the
    /// caller. The gene-projection path builds the metadata half here, lets
    /// anndata slice it, and then assigns matrices that were assembled already
    /// projected. Carries the projected gene count so the memory-budget
    /// estimate describes the assembly that will actually happen.
    Skeleton { n_selected_vars: usize },
}

/// Whether a `u32 → f32 → target` round trip can lose a value in this file.
///
/// The projected assembler is f32 (it reuses the backed handle's shard-by-shard
/// materialisation), while `read_all_csr_shards_typed` narrows from the native
/// `u32` stream and is therefore exact for an integer target of any width. The
/// two only differ above 2²⁴, so a non-default plan may take the projected path
/// exactly when the file's values fit f32 without rounding.
///
/// Float-encoded shards record `value_max == 0`, so they qualify: their values
/// were f32 on the way in, and a lossy *narrow* of one is still rejected
/// per element by `checked_cast_values`, the same gate the typed path uses.
pub(crate) fn f32_roundtrip_is_exact(csr_max_value: u32) -> bool {
    csr_max_value <= scx_codec::F32_MAX_EXACT_INT
}

/// Map a typed-read error to a Python exception. The in-assembly narrow reader
/// surfaces the fail-loud cast gate (lossy narrow / decode-loss) as
/// `ScxError::Codec(..)`; that is a bad-request condition — map it to
/// `ValueError` to match the F4 contract (`to_anndata(data_dtype=…)` raises
/// `ValueError` on a lossy narrow). Genuine I/O / catalog errors keep the
/// default `to_pyerr` mapping (`RuntimeError`).
pub(crate) fn typed_read_to_pyerr(e: scx_format_io::ScxError) -> PyErr {
    match e {
        scx_format_io::ScxError::Codec(ce) => {
            pyo3::exceptions::PyValueError::new_err(ce.to_string())
        }
        other => to_pyerr(other),
    }
}

/// Catalog-only estimate of the bytes required to assemble the full X
/// matrix plus obs / var metadata into an in-memory AnnData. Sums
/// `nnz × 16` for CSR shards (i32 indices + f32 data), `n_rows × 8`
/// for the assembled CSR indptr (i64), and the on-disk size of every
/// obs / var section (sharded or single). Walks `reader.catalog()`
/// only — no payload reads.
pub(crate) fn estimate_eager_assembly_bytes(
    reader: &ScxReader,
    selected_vars: Option<usize>,
) -> u64 {
    let entries = &reader.catalog().entries;
    let mut nnz: u64 = 0;
    let mut x_rows: u64 = 0;
    for entry in entries {
        if entry.section_type != SectionType::CsrShard || entry.modality_id != 0 {
            continue;
        }
        if let Some(stats) = &entry.stats {
            nnz = nnz.saturating_add(stats.nnz);
            x_rows = x_rows.saturating_add(stats.row_end.saturating_sub(stats.row_start));
        }
    }
    let mut meta_bytes: u64 = 0;
    for entry in entries {
        match entry.section_type {
            SectionType::ObsMetadata
            | SectionType::VarMetadata
            | SectionType::ObsMetadataShard
            | SectionType::VarMetadataShard => {
                meta_bytes = meta_bytes.saturating_add(entry.length);
            }
            _ => {}
        }
    }
    // A gene projection assembles only the selected columns, so the whole-file
    // nnz describes a matrix that is never built. The catalog has no per-column
    // nnz, so scale proportionally and say so: this assumes an even spread of
    // nonzeros across genes, which is an approximation on a warning that was
    // always advisory. Without it, `to_anndata(var_names=…)` on a large file
    // would keep recommending a smaller `var_names` — advice the caller has
    // already taken.
    let n_vars = reader.n_vars();
    if let Some(selected) = selected_vars {
        if n_vars > 0 && (selected as u64) < n_vars {
            nnz = (nnz as u128 * selected as u128 / n_vars as u128) as u64;
        }
    }
    nnz.saturating_mul(16)
        .saturating_add(x_rows.saturating_mul(8))
        .saturating_add(meta_bytes)
}

// The decode-loss guard folds `ShardStats::value_max` over the shards in scope
// via `FullCatalog::{csr_max_value, raw_csr_max_value, layer_csr_max_value}`
// (scx-format), the shared implementation consumed by both bindings; the fold
// semantics (float shards record 0, a stats-less entry contributes 0) are
// documented there.

/// Build an AnnData object from an ScxReader with optional layer filtering.
///
/// `eager` controls how `obsp` / `varp` / `varm` / `layers` are
/// populated. When `false` (default for `pyscx.open(...).to_anndata()`),
/// these slots are wrapped in `ScxLazyPairwiseMapping` /
/// `ScxLazyVarmMapping` / `ScxLazyLayersMapping` and attached to the
/// AnnData's private `_obsp` / `_varp` / `_varm` / `_layers` storage,
/// deferring each section's decode until the consumer first accesses
/// `ad.obsp[…]` etc. Keeps peak RSS of `to_anndata()` bounded for
/// files that carry large kNN graphs / embeddings. When `true`, every
/// section is decoded up front and a plain `dict` is passed through
/// the AnnData constructor — matches pre-fix behaviour and detaches
/// the returned AnnData from the SCX file handle. See
/// [`crate::lazy_mapping`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn to_anndata_with_layers<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    reader: &ScxReader,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    filters: SlotFilters<'_>,
    eager: bool,
    memory_budget: Option<u64>,
    mode: MatrixMode,
    plan: &scx_sparse::MaterializePlan,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::lazy_mapping::{
        PairwiseAxis, ScxLazyLayersMapping, ScxLazyObsmMapping, ScxLazyPairwiseMapping,
        ScxLazyVarmMapping,
    };
    use std::sync::Arc;

    let anndata_mod = crate::pyimport::import_module(py, "anndata")?;

    // Phase 4d: catalog-only estimate of the full-assembly bytes. If
    // the estimate exceeds the budget (caller's `memory_budget` kwarg,
    // or `DEFAULT_EAGER_MEMORY_BUDGET_BYTES` = 8 GiB when unset) emit a
    // `UserWarning` recommending the backed / query alternatives.
    // Assembly proceeds regardless — the warning is advisory.
    let budget = memory_budget.unwrap_or(DEFAULT_EAGER_MEMORY_BUDGET_BYTES);
    let est_bytes = estimate_eager_assembly_bytes(
        reader,
        match mode {
            MatrixMode::Skeleton { n_selected_vars } => Some(n_selected_vars),
            _ => None,
        },
    );
    if est_bytes > budget {
        warn_python_convert(
            py,
            &scx_convert::ConvertWarning::EagerAssemblyMemoryHigh {
                estimated_bytes: est_bytes,
                budget_bytes: budget,
            },
        )?;
    }

    // Fail loud on the silent u32→f32 decode loss before any decode. Covers X
    // (this function is the eager assembler shared by the no-filter, var_names,
    // preserve_slots, and gpu-skeleton paths), so all of them are guarded here
    // rather than at each call site. The check is catalog-only, so it applies
    // even when the caller defers the host X decode to `to_gpu_anndata`'s
    // f32-native device path.
    guard_decode_loss_dtype(
        reader.catalog().csr_max_value(None),
        plan.data_dtype,
        plan.allow_lossy,
    )?;

    // X — assemble all CSR shards (with deletion vector filtering).
    //
    // `SkipX` and `Skeleton` both build an X-less object and let the caller
    // assign `adata.X`: `to_gpu_anndata`'s device-resident streamed path
    // decodes X straight onto the GPU, and the gene-projection path assembles
    // it already projected. obs/var/obsm/uns are assembled identically in every
    // mode; `Skeleton` additionally omits eager layers, since those are
    // projected too.
    //
    // Non-default plans narrow **in-decode**: the typed reader assembles X
    // directly at the target dtype (never building the full-matrix f32 CSR), so
    // a narrow read lowers peak RSS and integer→integer narrows are exact for
    // any value (including > 2²⁴). The default (csr/f32/i32) plan stays on the
    // untouched zero-copy path.
    let x = if !matches!(mode, MatrixMode::Eager) {
        None
    } else if plan.is_default_csr_f32() {
        let csr = reader.read_all_csr_shards_filtered().map_err(to_pyerr)?;
        Some(csr_to_scipy(py, csr)?)
    } else if plan.container == scx_sparse::Container::Dense {
        let dense = reader
            .read_all_csr_shards_dense_typed(plan)
            .map_err(typed_read_to_pyerr)?;
        Some(typed_dense_to_numpy(py, dense)?)
    } else {
        let csr = reader
            .read_all_csr_shards_typed(plan)
            .map_err(typed_read_to_pyerr)?;
        Some(typed_csr_to_scipy(py, csr)?)
    };

    // obs metadata — filter by deletion vectors if present
    let obs = match reader.read_obs() {
        Ok(batch) => {
            let filtered_batch = filter_obs_by_deletion_vectors(reader, batch)?;
            let table = record_batch_to_pyarrow(py, &filtered_batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // var metadata
    let var = match reader.read_var() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // obsm embeddings.
    //
    // `obsm` is eager by default (it tends to be small relative to
    // obsp/varp/varm). It only becomes a lazy bridge when the caller has
    // explicitly opted into selective loading via `obsm=[...]` AND not
    // forced `eager=True` — keeping default semantics byte-identical
    // while letting the random-access dataloader path defer (and skip)
    // per-key materialisation. See `ScxLazyObsmMapping`.
    //
    // `obsm_filter` restricts the loaded keys in either mode
    // (`to_anndata(obsm=[...])`).
    let lazy_obsm_requested = !eager && obsm_filter.is_some();
    let obsm_dict = pyo3::types::PyDict::new(py);
    // Lazy mode reads nothing here: keys were validated by `to_anndata_filtered`
    // before it chose a branch, and the bridge built below reads each on first
    // access.
    if !lazy_obsm_requested {
        let obsm_map = read_obsm_selected(reader, obsm_filter)?;
        for (name, batch) in &obsm_map {
            let filtered = filter_obs_by_deletion_vectors(reader, batch.clone())?;
            let np_arr = obsm_batch_to_numpy(py, &filtered)?;
            obsm_dict.set_item(name, np_arr)?;
        }
    }

    // uns — reconstruct any `__scx_type__` envelopes back into NumPy
    // ndarrays / scalars / tuples / pandas Index/Series/Categorical /
    // structured recarrays. Plain JSON passes through unchanged.
    let uns_dict = read_uns_as_pyobject(py, reader, 0)?;

    // Sibling reader for the lazy bridges (`_obsp`/`_varp`/`_varm`/
    // `_layers`). Independent mmap so the returned AnnData stays valid
    // after the caller's `ScxReader` drops. Skipped when no lazy slot
    // is needed.
    //
    // `obsp` / `varp` / `varm` narrow the same way `layers` always has: the
    // filter decides whether the slot exists for this call at all, so an empty
    // list builds no bridge rather than an empty one. Keys were validated by
    // the entry point (`to_anndata_filtered`), which reaches the query branch
    // this function does not.
    let has_obsp = slot_has_selected(&reader.list_obsp(), filters.obsp);
    let has_varp = slot_has_selected(&reader.list_varp(), filters.varp);
    let has_varm = slot_has_selected(&reader.list_varm(), filters.varm);
    let layer_names = reader.layer_names();
    // `Skeleton` never builds a layer bridge — the caller assembles each
    // selected layer projected and assigns it after the slice. Leaving
    // `has_layers` true here would open a sibling `ScxReader` mmap for a bridge
    // nothing reads.
    let has_layers = !matches!(mode, MatrixMode::Skeleton { .. })
        && if let Some(filter) = layer_filter {
            layer_names.iter().any(|n| filter.iter().any(|f| f == n))
        } else {
            !layer_names.is_empty()
        };
    let need_lazy = has_obsp || has_varp || has_varm || has_layers || lazy_obsm_requested;

    let obsp_kept = if has_obsp {
        compute_kept_to_global(reader)?.map(Arc::new)
    } else {
        None
    };

    let lazy_reader: Option<Arc<ScxReader>> = if need_lazy {
        Some(Arc::new(
            crate::open_handle_reader_shared(path, reader.catalog_arc()).map_err(to_pyerr)?,
        ))
    } else {
        None
    };
    let lazy_obsp = lazy_reader.as_ref().filter(|_| has_obsp).map(|r| {
        ScxLazyPairwiseMapping::new(
            Arc::clone(r),
            PairwiseAxis::Obsp,
            obsp_kept.clone(),
            filters.obsp,
        )
    });
    let lazy_varp = lazy_reader.as_ref().filter(|_| has_varp).map(|r| {
        ScxLazyPairwiseMapping::new(Arc::clone(r), PairwiseAxis::Varp, None, filters.varp)
    });
    let lazy_varm = lazy_reader
        .as_ref()
        .filter(|_| has_varm)
        .map(|r| ScxLazyVarmMapping::new(Arc::clone(r), filters.varm));
    let lazy_layers = lazy_reader
        .as_ref()
        .filter(|_| has_layers)
        .map(|r| ScxLazyLayersMapping::new(Arc::clone(r), layer_filter));
    let lazy_obsm = lazy_reader
        .as_ref()
        .filter(|_| lazy_obsm_requested)
        .map(|r| ScxLazyObsmMapping::new(Arc::clone(r), obsm_filter));

    // Build AnnData kwargs.
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(x) = x {
        kwargs.set_item("X", x)?;
    }
    if let Some(obs) = &obs {
        kwargs.set_item("obs", obs)?;
    }
    if let Some(var) = var {
        kwargs.set_item("var", var)?;
    }
    if !obsm_dict.is_empty() {
        kwargs.set_item("obsm", obsm_dict)?;
    }
    if let Some(uns) = uns_dict {
        kwargs.set_item("uns", uns)?;
    }
    if matches!(mode, MatrixMode::Skeleton { .. }) {
        // Without X *and* without layers, AnnData infers `n_obs` from obs /
        // obsm / obsp — and a file with no obs section has none of them, so the
        // skeleton would silently come out `(0, n_vars)` and the caller's
        // `adata.X = …` would die on a shape mismatch. Stating the shape also
        // gets a free cross-check against `obs` when obs *is* present.
        let n_obs_visible = match &obs {
            Some(df) => df.getattr("shape")?.get_item(0)?.extract::<usize>()?,
            None => match compute_kept_to_global(reader)? {
                Some(kept) => kept.len(),
                None => reader.n_obs() as usize,
            },
        };
        kwargs.set_item("shape", (n_obs_visible, reader.n_vars() as usize))?;
    }
    if eager {
        // Materialize each lazy bridge up front; AnnData's __init__
        // receives plain dicts (same shape as pre-fix). Returned AnnData
        // is fully detached from the SCX file handle.
        if let Some(m) = &lazy_obsp {
            kwargs.set_item("obsp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varp {
            kwargs.set_item("varp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varm {
            kwargs.set_item("varm", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_layers {
            // Layers still materialize to f32 (the in-decode typed layer reader is
            // a Phase-5 follow-up), then a non-default plan narrows them via the
            // post-assembly `retype_matrix` pass in `experiment.rs`. Because that
            // pass casts the *already-f32* values, a layer count > 2²⁴ cannot be
            // delivered losslessly on any path — so the f32 decode-loss guard is
            // the honest gate here (fail loud rather than silently round). The
            // non-eager narrow path is guarded symmetrically in `experiment.rs`
            // before the retype loop; X/raw, by contrast, narrow in-decode and are
            // exact for `>2²⁴` integer targets.
            guard_decode_loss(
                reader.catalog().layer_csr_max_value(0, None),
                plan.allow_lossy,
            )?;
            kwargs.set_item("layers", m.materialize_all(py)?)?;
        }
    }

    let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;

    // `Skeleton` defers raw to the caller, which attaches it *after* anndata's
    // slice + copy. Attaching it here would put it through both — a var slice
    // does not touch raw, so the view and `_mutated_copy` would each take a
    // full-width copy of a matrix nothing narrowed.
    if !matches!(mode, MatrixMode::Skeleton { .. }) {
        attach_raw(py, &adata, reader, filters, plan)?;
    }

    if !eager {
        // Lazy mode: attach each bridge to AnnData's private `_obsp` /
        // `_varp` / `_varm` / `_layers` storage. AnnData's
        // `AlignedMappingProperty` descriptor reads from these on
        // every public `.obsp` (etc.) access — the first access drives
        // the bridge's per-key materialization through AnnData's
        // validation loop; subsequent accesses hit the bridge's cache.
        // We bypass the property setter (which would otherwise iterate
        // and validate every entry up front, defeating the lazy point).
        if let Some(m) = lazy_obsp {
            adata.setattr("_obsp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varp {
            adata.setattr("_varp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varm {
            adata.setattr("_varm", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_layers {
            adata.setattr("_layers", m.into_pyobject(py)?)?;
        }
        // Lazy obsm is only built when the caller opted into selective
        // loading (`obsm=[...]`, eager=False); default behaviour keeps
        // obsm eager. Attach via `_obsm` to bypass AnnData's axis-length
        // validation, same as the other bridges.
        if let Some(m) = lazy_obsm {
            adata.setattr("_obsm", m.into_pyobject(py)?)?;
        }
    }

    Ok(adata)
}

/// Whether the projected assembler may serve this read.
///
/// It is f32 (it reuses the backed handle's shard-by-shard materialisation),
/// while `read_all_csr_shards_typed` narrows from the native `u32` stream and is
/// exact for an integer target of any width. The two can only differ above 2²⁴,
/// so a non-default plan takes the projected path exactly when the file's values
/// survive an f32 round trip — otherwise a `data_dtype="uint32"` read of a
/// `> 2²⁴` count would round with no guard firing, since
/// `guard_decode_loss_for::<u32>` permits that target by design.
fn projection_is_safe(reader: &ScxReader, plan: &scx_sparse::MaterializePlan) -> bool {
    plan.is_default_csr_f32() || f32_roundtrip_is_exact(reader.catalog().csr_max_value(None))
}

/// Evaluate a `preserve_slots=True` `obs_filter` against an assembled AnnData's
/// obs frame, rejecting a non-boolean result and warning about the grammar shift.
///
/// Split out so the projected and unprojected `preserve_slots` paths cannot
/// drift on the validation or on the warning text.
fn eval_preserve_slots_mask<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    expr: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let obs_attr = adata.getattr("obs")?;
    let mask = obs_attr.call_method1("eval", (expr,)).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "preserve_slots=True parses obs_filter via pandas.eval; \
             failed to evaluate {expr:?}: {e}"
        ))
    })?;

    // Reject non-boolean results: AnnData treats numeric arrays as positional
    // indices, which would silently reorder rows instead of failing on a
    // malformed predicate.
    let dtype_kind: String = mask.getattr("dtype")?.getattr("kind")?.extract()?;
    if dtype_kind != "b" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "preserve_slots=True requires obs_filter to evaluate to a \
             boolean mask (e.g. \"cell_type == 'T cell'\"); expression \
             {expr:?} produced dtype kind {dtype_kind:?}"
        )));
    }

    // Surface the grammar shift: this path evaluates obs_filter via pandas.eval,
    // which does not match the SCX predicate engine (e.g. pandas accepts `&` /
    // `|` / `~`; SCX accepts only `and` / `or` / `not`). Users opted into
    // preserve_slots=True, so one warning per call is appropriate.
    crate::pyimport::import_module(py, "warnings")?.call_method1(
        "warn",
        (format!(
            "preserve_slots=True evaluated obs_filter {expr:?} via pandas.eval; \
             grammar differs from the SCX predicate engine used by \
             preserve_slots=False (see docs/scanpy.md \"Filter Expression Compatibility\")."
        ),),
    )?;
    Ok(mask)
}

/// Positional indices of the `True` entries of a boolean mask, in order.
fn mask_true_positions(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let np = crate::pyimport::import_module(py, "numpy")?;
    let where_result = np.call_method1("where", (mask,))?;
    let arr: numpy::PyReadonlyArray1<'_, i64> = where_result
        .get_item(0)?
        .call_method1("astype", (np.getattr("int64")?,))?
        .extract()?;
    Ok(arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .iter()
        .map(|&i| i as usize)
        .collect())
}

/// Narrow a visible-row → global-row map by a set of visible positions.
///
/// `None` means no deletion vectors, so a visible row *is* a global row and the
/// positions are already the answer. Shared by the backed `obs_filter` path and
/// the projected `preserve_slots` one, which compose the same two things.
fn compose_kept_to_global(kept_to_global: Option<&Vec<u64>>, positions: &[usize]) -> Vec<u64> {
    match kept_to_global {
        Some(existing) => positions.iter().map(|&i| existing[i]).collect(),
        None => positions.iter().map(|&i| i as u64).collect(),
    }
}

/// Build the eager AnnData for a gene-projected read.
///
/// The metadata half is built full width and handed to anndata to slice, which
/// is not laziness: `adata[:, idx]` prunes unused **var** categories and
/// reindexes or deletes `uns["<col>_colors"]` (`AnnData._init_as_view`).
/// Reproducing that by hand is how a rewrite silently changes output, so the
/// slice stays and only the matrices are taken away from it — X and each
/// selected layer are assembled already projected, and `.raw` (which anndata
/// never var-slices) is attached afterwards so the view and the copy do not each
/// take a full-width copy of it.
///
/// `obs_filter_expr` is the `preserve_slots=True` case: the mask is evaluated on
/// the skeleton's obs, folded into the row set the projected matrices are built
/// with, and passed to the same slice.
#[allow(clippy::too_many_arguments)]
fn projected_eager_anndata<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    reader: &ScxReader,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    filters: SlotFilters<'_>,
    memory_budget: Option<u64>,
    plan: &scx_sparse::MaterializePlan,
    col_indices: &Vec<u32>,
    preserve_var_order: bool,
    obs_filter_expr: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let skeleton = to_anndata_with_layers(
        py,
        path,
        reader,
        layer_filter,
        obsm_filter,
        filters,
        true,
        memory_budget,
        MatrixMode::Skeleton {
            n_selected_vars: col_indices.len(),
        },
        plan,
    )?;

    // Row set the projected matrices must be built over: the file's deletion
    // vectors, further narrowed by `preserve_slots`' pandas mask.
    let mut kept_to_global = compute_kept_to_global(reader)?;
    let builtins = crate::pyimport::import_module(py, "builtins")?;
    let row_idx = match obs_filter_expr {
        None => builtins.call_method1("slice", (py.None(),))?,
        Some(expr) => {
            let mask = eval_preserve_slots_mask(py, &skeleton, expr)?;
            let positions = mask_true_positions(py, &mask)?;
            kept_to_global = Some(compose_kept_to_global(kept_to_global.as_ref(), &positions));
            mask
        }
    };

    let np_indices = PyArray1::from_slice(py, col_indices);
    let idx = pyo3::types::PyTuple::new(py, &[row_idx.unbind(), np_indices.into_any().unbind()])?;
    let adata = skeleton.get_item(idx)?.call_method0("copy")?;

    let catalog = reader.catalog_arc();
    let window = decode_window(reader);
    let x = projected_matrix(
        py,
        path,
        std::sync::Arc::clone(&catalog),
        None,
        kept_to_global.as_ref(),
        col_indices,
        preserve_var_order,
        window,
        plan,
    )?;
    adata.setattr("X", x)?;

    // Layers, in catalog order (the pre-projection path materialised them from
    // a `HashMap`, so `list(adata.layers)` used to vary between runs).
    let selected: Vec<String> = reader
        .layer_names()
        .into_iter()
        .filter(|n| layer_filter.is_none_or(|f| f.iter().any(|x| x == n)))
        .collect();
    if !selected.is_empty() {
        // The eager-layers decode-loss guard lived inside the bridge branch this
        // path replaces. Layers materialise as f32 on every path and are
        // narrowed post-assembly, so a `> 2²⁴` layer count cannot be delivered
        // losslessly by any route — fail loud rather than round silently.
        guard_decode_loss(
            reader.catalog().layer_csr_max_value(0, None),
            plan.allow_lossy,
        )?;
        let layers_attr = adata.getattr("layers")?;
        // f32, like every other layer path: `experiment::to_anndata_impl`
        // applies a non-default plan to layers after this returns.
        let layer_plan = scx_sparse::MaterializePlan::default_csr_f32();
        for name in &selected {
            let mat = projected_matrix(
                py,
                path,
                std::sync::Arc::clone(&catalog),
                Some(name),
                kept_to_global.as_ref(),
                col_indices,
                preserve_var_order,
                window,
                &layer_plan,
            )?;
            layers_attr.set_item(name, mat)?;
        }
    }

    attach_raw(py, &adata, reader, filters, plan)?;
    Ok(adata)
}

/// How many shards the projected assembly may decode at once.
///
/// The path this replaces ran its shard loop through rayon, so a purely
/// sequential projected assembly is a several-fold wall-clock regression on the
/// call it is supposed to make cheaper — measured at ~6x on a 12-core box.
/// A window restores the parallelism while keeping the memory claim honest:
/// peak holds `window` full-width shards, not the whole matrix.
///
/// Sized from the catalog's average X shard nnz against a fixed budget, so a
/// file with very large shards narrows the window rather than the in-flight
/// bytes growing with core count. The same window serves each layer, whose
/// shards are assumed to be of comparable density — an approximation, and the
/// reason the budget is set well below anything a caller would notice.
///
/// Deliberately independent of the `memory_budget` kwarg: that one is advisory
/// (it warns, it does not block), and making it govern decode concurrency would
/// quietly change what it means.
fn decode_window(reader: &ScxReader) -> usize {
    /// Full-width decoded shards may occupy this much at once. Well under the
    /// 8 GiB default eager budget, and far above any real shard.
    const IN_FLIGHT_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

    let threads = rayon::current_num_threads().max(1);
    let n_shards = reader.header().n_csr_shards as u64;
    if n_shards == 0 {
        return 1;
    }
    // A decoded shard is roughly `nnz * 8` (i32 index + f32 value per nonzero)
    // plus its indptr; the indptr term is noise beside the values.
    let shard_bytes = (reader.nnz() / n_shards).saturating_mul(8).max(1);
    let by_bytes = (IN_FLIGHT_BUDGET_BYTES / shard_bytes).max(1) as usize;
    threads.min(by_bytes)
}

/// Open a `BackedCsrReader` over X (`layer = None`) or one named layer, reusing
/// an already-parsed catalog.
///
/// `cache_shards = 0` is the right value for a one-shot projected assembly, and
/// not merely a small one: `ScxBackedSparseDataset::as_shard_source` does not
/// opt into cached reads, so every shard is decoded through
/// `read_shard_uncached` and the LRU is never consulted. Any other value would
/// allocate a cache with a guaranteed zero hit rate.
pub(crate) fn open_backed_matrix_reader(
    path: &std::path::Path,
    catalog: std::sync::Arc<scx_format_io::FullCatalog>,
    layer: Option<&str>,
    cache_shards: usize,
) -> PyResult<std::sync::Arc<scx_format_io::BackedCsrReader>> {
    let reader = crate::open_handle_reader_shared(path, catalog).map_err(to_pyerr)?;
    Ok(std::sync::Arc::new(match layer {
        Some(name) => scx_format_io::BackedCsrReader::new_for_layer(reader, name, cache_shards),
        None => scx_format_io::BackedCsrReader::new(reader, cache_shards),
    }))
}

/// Install a resolved gene projection on a backed handle.
///
/// Under `preserve_var_order` the visible axis is in request order, which the
/// handle expresses as a sorted projection plus a presentation permutation;
/// using the sorted setter there leaves the handle transposed relative to
/// `adata.var`. Shared so X and every layer cannot pick different setters.
pub(crate) fn install_projection(
    ds: &mut crate::backed::ScxBackedSparseDataset,
    col_indices: Option<&Vec<u32>>,
    preserve_var_order: bool,
) {
    if let Some(indices) = col_indices {
        if preserve_var_order {
            ds.set_col_projection_ordered(indices.clone());
        } else {
            ds.set_col_projection(indices.clone());
        }
    }
}

/// Assemble X (or one layer), projected to `col_indices`, as a scipy matrix.
///
/// Streams shard by shard, so peak is twice the projected result plus `window`
/// full-width shards in flight rather than the whole matrix twice. Deletion
/// vectors ride along through `kept_to_global`.
#[allow(clippy::too_many_arguments)]
fn projected_matrix<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    catalog: std::sync::Arc<scx_format_io::FullCatalog>,
    layer: Option<&str>,
    kept_to_global: Option<&Vec<u64>>,
    col_indices: &Vec<u32>,
    preserve_var_order: bool,
    window: usize,
    plan: &scx_sparse::MaterializePlan,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::backed::ScxBackedSparseDataset;

    let backed = open_backed_matrix_reader(path, catalog, layer, 0)?;
    let mut ds = match kept_to_global {
        Some(mapping) => {
            ScxBackedSparseDataset::from_reader_with_deletions(backed, mapping.clone())
        }
        None => ScxBackedSparseDataset::from_reader(backed),
    };
    install_projection(&mut ds, Some(col_indices), preserve_var_order);
    let csr = crate::backed::detached(py, || {
        crate::backed::materialize_projected(&ds, window).map_err(|e| e.to_string())
    })
    .map_err(PyRuntimeError::new_err)?;
    let csr = csr_to_scipy(py, csr)?;
    // The projected assembler is f32; a non-default plan narrows the (already
    // small) projected result afterwards. Reaching here at all means
    // `f32_roundtrip_is_exact` said the detour cannot round a value, and the
    // narrow itself is still gated per element by `retype_matrix`.
    if plan.is_default_csr_f32() {
        Ok(csr)
    } else {
        retype_matrix(py, csr, plan)
    }
}

/// Reconstruct `adata.raw` from the file's raw sections, or say why not.
///
/// Raw shares X's obs axis; when deletion vectors are active the raw rows would
/// need the same filtering as X, which this path does not apply — warn and drop
/// rather than emit a misaligned raw. `raw=False` opts out of both branches: no
/// rebuild, and no notice about a matrix the caller has said they do not want.
///
/// A **gene** projection is deliberately not passed here: anndata does not
/// var-slice `.raw` (`AnnData._init_as_view` hands it only the obs index), so
/// raw stays on its own — usually wider — gene axis on every path.
pub(crate) fn attach_raw(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    reader: &ScxReader,
    filters: SlotFilters<'_>,
    plan: &scx_sparse::MaterializePlan,
) -> PyResult<()> {
    if !(filters.raw && reader.has_raw()) {
        return Ok(());
    }
    if reader.header().has_deletion_vectors() {
        warn_python_convert(
            py,
            &scx_convert::ConvertWarning::DroppedRaw {
                raw_n_vars: reader.raw_n_vars().unwrap_or(0),
            },
        )?;
        return Ok(());
    }
    // adata.raw holds pre-normalization counts — the most likely place a > 2²⁴
    // integer lives. Guard before decode (dtype-aware).
    guard_decode_loss_dtype(
        reader.catalog().raw_csr_max_value(),
        plan.data_dtype,
        plan.allow_lossy,
    )?;
    // Non-default plans narrow raw in-decode too (raw stays CSR even for
    // `container="dense"` — the conventional raw representation). This also
    // fixes the pre-existing gap where raw stayed f32 under a non-default plan
    // (the old post-assembly retype only touched X/layers).
    let raw_x = if plan.is_default_csr_f32() {
        let raw_csr = reader.read_all_raw_csr_shards().map_err(to_pyerr)?;
        csr_to_scipy(py, raw_csr)?
    } else {
        let raw_csr = reader
            .read_all_raw_csr_shards_typed(plan)
            .map_err(typed_read_to_pyerr)?;
        typed_csr_to_scipy(py, raw_csr)?
    };
    let raw_var_batch = reader.read_raw_var().map_err(to_pyerr)?;
    let raw_var_table = record_batch_to_pyarrow(py, &raw_var_batch)?;
    let raw_var = pyarrow_table_to_pandas(&raw_var_table)?;
    let anndata_mod = crate::pyimport::import_module(py, "anndata")?;
    let raw_kwargs = pyo3::types::PyDict::new(py);
    raw_kwargs.set_item("X", raw_x)?;
    raw_kwargs.set_item("var", raw_var)?;
    let raw_adata = anndata_mod.call_method("AnnData", (), Some(&raw_kwargs))?;
    // `adata.raw = AnnData(X=..., var=...)` stores it as a Raw — the canonical
    // scanpy idiom.
    adata.setattr("raw", raw_adata)?;
    Ok(())
}

/// Build an AnnData with optional var_names projection, obs_filter, and layers selection.
///
/// For obs_filter: delegates to the QueryPipeline for predicate pushdown.
/// For var_names: resolves gene names to column indices and applies column slicing.
/// For layers: filters which layers are loaded.
///
/// When `preserve_slots=true` and `obs_filter` is set, the eager path
/// `to_anndata_with_layers()` is used (loading X / obs / var / obsm /
/// layers with deletion vectors applied) and then sliced by a pandas.eval
/// boolean mask. This preserves obsm and layers at the cost of the query
/// engine's predicate-pushdown shard skipping. When `preserve_slots=false`
/// (default), the query-engine path runs and emits a warning if obsm or
/// layers exist on disk (since they are dropped from the result).
#[allow(clippy::too_many_arguments)]
pub fn to_anndata_filtered<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    reader: &ScxReader,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    filters: SlotFilters<'_>,
    preserve_slots: bool,
    eager: bool,
    memory_budget: Option<u64>,
    skip_x: bool,
    preserve_var_order: bool,
    strict_var_names: bool,
    plan: &scx_sparse::MaterializePlan,
) -> PyResult<Bound<'py, PyAny>> {
    // `skip_x` (the X-less object `to_gpu_anndata` fills in) is only meaningful
    // on the no-filter fast path — the caller guarantees no var_names /
    // obs_filter / layer projection when it sets it (those paths reshape X and
    // must build it).
    debug_assert!(
        !skip_x || (var_names.is_none() && obs_filter.is_none() && layer_filter.is_none()),
        "skip_x requires no var_names / obs_filter / layer_filter"
    );
    let base_mode = if skip_x {
        MatrixMode::SkipX
    } else {
        MatrixMode::Eager
    };

    // Validate the slot filters here rather than in `to_anndata_with_layers`:
    // this function has four branches and one of them (the obs-filtered query
    // path) never calls it, so validating downstream would let a typo through
    // on exactly the path that already discards those slots. Catalog-only
    // listings — no shard bytes, so the lazy bridges' deferral is intact.
    //
    // All FIVE slots, not just the three new ones. The query branch decides its
    // "does not load {slots}" warning with `slot_has_selected`, which cannot
    // tell "the caller excluded this slot" from "the caller misspelled a key" —
    // both select nothing. So an unvalidated slot with a typo would produce no
    // `KeyError` *and* no warning naming it: on a file whose only aligned slot
    // is that one, `to_anndata(obs_filter=…, obsm=["typo"])` returned in silence.
    // `obsm=` already documents `KeyError`; the check simply lived in
    // `to_anndata_with_layers`, which this branch never reaches. `layers=` had
    // no such contract at all and silently yielded an empty slot on any path —
    // it gets one here, so every slot filter fails the same way.
    validate_slot_keys(&reader.layer_names(), layer_filter, "layers")?;
    validate_slot_keys(&reader.list_obsm(), obsm_filter, "obsm")?;
    validate_slot_keys(&reader.list_obsp(), filters.obsp, "obsp")?;
    validate_slot_keys(&reader.list_varp(), filters.varp, "varp")?;
    validate_slot_keys(&reader.list_varm(), filters.varm, "varm")?;

    // Fast path: no filtering → use existing implementation. The decode-loss
    // guard (X / raw / eager layers) runs inside `to_anndata_with_layers`, so
    // every caller of it — this branch, preserve_slots, and the var_names-only
    // path below — is covered uniformly.
    if var_names.is_none() && obs_filter.is_none() && layer_filter.is_none() {
        return to_anndata_with_layers(
            py,
            path,
            reader,
            None,
            obsm_filter,
            filters,
            eager,
            memory_budget,
            base_mode,
            plan,
        );
    }

    // preserve_slots=true with obs_filter: load full AnnData, then filter
    // rows via pandas.eval. Keeps obsm / layers / uns intact at the cost
    // of skipping query-engine predicate pushdown. Force eager so the
    // pandas-side __getitem__ slicing operates on real arrays rather
    // than lazy bridges (which AnnData iterates / validates during
    // `.copy()` anyway).
    if let (Some(expr), true) = (obs_filter, preserve_slots) {
        // A gene projection takes the same route as the unfiltered one: the
        // metadata half is sliced by anndata (rows *and* columns at once), and
        // X / layers are assembled already projected over the surviving rows.
        // This branch is the highest-peak path in the API — it exists to keep
        // every slot — so leaving it on assemble-then-slice would be an
        // incoherent contract.
        if let Some(names) = var_names {
            let indices =
                resolve_var_names_to_indices(reader, names, preserve_var_order, strict_var_names)?;
            if projection_is_safe(reader, plan) {
                return projected_eager_anndata(
                    py,
                    path,
                    reader,
                    layer_filter,
                    obsm_filter,
                    filters,
                    memory_budget,
                    plan,
                    &indices,
                    preserve_var_order,
                    Some(expr),
                );
            }
        }

        // Decode-loss guard runs inside to_anndata_with_layers.
        let full = to_anndata_with_layers(
            py,
            path,
            reader,
            layer_filter,
            obsm_filter,
            filters,
            true,
            memory_budget,
            MatrixMode::Eager,
            plan,
        )?;

        let mask = eval_preserve_slots_mask(py, &full, expr)?;
        let builtins = crate::pyimport::import_module(py, "builtins")?;
        let slice_all = builtins.call_method1("slice", (py.None(),))?;
        let col_idx = if let Some(names) = var_names {
            // Fancy column indexing honours order, so request order is preserved
            // automatically when preserve_var_order is set.
            let indices =
                resolve_var_names_to_indices(reader, names, preserve_var_order, strict_var_names)?;
            PyArray1::from_vec(py, indices).into_any().unbind()
        } else {
            slice_all.unbind()
        };
        let idx = pyo3::types::PyTuple::new(py, &[mask.unbind(), col_idx])?;
        return full.get_item(idx)?.call_method0("copy");
    }

    // If obs_filter is specified, use the query engine for predicate pushdown
    if let Some(expr) = obs_filter {
        use scx_engine::QueryPipeline;

        let mut pipeline =
            QueryPipeline::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        pipeline = pipeline
            .filter_obs(expr)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // If var_names is also specified, resolve to gene indices. The query
        // engine (scx-engine collect's F4 ColumnReorder) already presents the
        // output columns in the order passed to select_genes — so resolving in
        // request order (preserve_var_order=True) vs sorted order is sufficient;
        // no post-collect reorder is needed here.
        if let Some(names) = var_names {
            let gene_indices =
                resolve_var_names_to_indices(reader, names, preserve_var_order, strict_var_names)?;
            pipeline = pipeline.select_genes(gene_indices);
        }

        let result = pipeline
            .collect()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Guard against the silent u32→f32 decode loss over exactly the shards
        // that survived predicate pushdown (`result.max_value`). The query path
        // assembles X as f32 first (Option A: post-assembly cast), so a `>2²⁴`
        // integer request is still gated here rather than decoded losslessly —
        // the in-decode narrow (G2) lands on the eager path only.
        guard_decode_loss(result.max_value, plan.allow_lossy)?;

        let anndata_mod = crate::pyimport::import_module(py, "anndata")?;
        // Narrow to the requested container/dtype (no-op for the default plan).
        let x = csr_to_scipy_typed(py, result.x, plan)?;
        let obs_table = record_batch_to_pyarrow(py, &result.obs)?;
        let obs_df = pyarrow_table_to_pandas(&obs_table)?;
        let var_table = record_batch_to_pyarrow(py, &result.var)?;
        let var_df = pyarrow_table_to_pandas(&var_table)?;

        // uns (still loaded from reader; see read_uns_as_pyobject for the
        // tagged-envelope reconstruction).
        let uns_dict = read_uns_as_pyobject(py, reader, 0)?;

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("X", x)?;
        kwargs.set_item("obs", obs_df)?;
        kwargs.set_item("var", var_df)?;
        if let Some(uns) = uns_dict {
            kwargs.set_item("uns", uns)?;
        }
        // obsm, varm, obsp, varp, and layers are not available via QueryResult.
        // Warn if the source file contains them so users know they're being
        // dropped — but only for slots the caller has not already excluded. A
        // caller who passed `obsp=[]` has said they do not want obsp; naming it
        // in a "you are losing these" notice would be telling them about a loss
        // they asked for.
        let has_obsm = slot_has_selected(&reader.list_obsm(), obsm_filter);
        let has_varm = slot_has_selected(&reader.list_varm(), filters.varm);
        let has_obsp = slot_has_selected(&reader.list_obsp(), filters.obsp);
        let has_varp = slot_has_selected(&reader.list_varp(), filters.varp);
        let has_layers = slot_has_selected(&reader.layer_names(), layer_filter);
        if has_obsm || has_varm || has_obsp || has_varp || has_layers {
            let warnings = crate::pyimport::import_module(py, "warnings")?;
            let mut parts = Vec::new();
            if has_obsm {
                parts.push("obsm");
            }
            if has_varm {
                parts.push("varm");
            }
            if has_obsp {
                parts.push("obsp");
            }
            if has_varp {
                parts.push("varp");
            }
            if has_layers {
                parts.push("layers");
            }
            warnings.call_method1(
                "warn",
                (format!(
                    "obs_filter with non-backed mode uses the query engine, which does not \
                     load {}. Pass preserve_slots=True to materialize them (skips predicate \
                     pushdown), use backed=True, or load the full dataset and filter in Python.",
                    parts.join(", ")
                ),),
            )?;
        }

        // The obs-filtered query path does not subset the raw matrix's
        // obs axis — warn + drop rather than emit a misaligned raw. `raw=False`
        // opts out of the notice.
        if filters.raw && reader.has_raw() {
            warn_python_convert(
                py,
                &scx_convert::ConvertWarning::DroppedRaw {
                    raw_n_vars: reader.raw_n_vars().unwrap_or(0),
                },
            )?;
        }

        let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;
        return Ok(adata);
    }

    // No obs_filter, but var_names and/or layers. `var_names` projects while
    // assembling; `layers=` alone still forces eager assembly, because slicing
    // the AnnData drags every aligned bridge through anndata's validation
    // anyway and fragmenting that cost across implicit slicing helps nobody.
    if let Some(names) = var_names {
        // Resolve via the same path as backed / query-engine: scans all string
        // columns (so gene symbols in non-index columns work). Returns sorted
        // positional indices by default, or request-order (deduped first-wins)
        // when preserve_var_order is set.
        let indices =
            resolve_var_names_to_indices(reader, names, preserve_var_order, strict_var_names)?;
        if projection_is_safe(reader, plan) {
            return projected_eager_anndata(
                py,
                path,
                reader,
                layer_filter,
                obsm_filter,
                filters,
                memory_budget,
                plan,
                &indices,
                preserve_var_order,
                None,
            );
        }
        // Fallback: assemble full width and let anndata slice. Reached only for
        // a narrow-dtype request on a file whose values exceed 2²⁴, where the
        // projected assembler's f32 stage could round a count the typed reader
        // delivers exactly. Same output, same peak, as before this path existed.
        let adata = to_anndata_with_layers(
            py,
            path,
            reader,
            layer_filter,
            obsm_filter,
            filters,
            true,
            memory_budget,
            MatrixMode::Eager,
            plan,
        )?;
        let np_indices = PyArray1::from_vec(py, indices);
        let builtins = crate::pyimport::import_module(py, "builtins")?;
        let slice_all = builtins.call_method1("slice", (py.None(),))?;
        let idx =
            pyo3::types::PyTuple::new(py, &[slice_all.unbind(), np_indices.into_any().unbind()])?;
        return adata.get_item(idx)?.call_method0("copy");
    }

    to_anndata_with_layers(
        py,
        path,
        reader,
        layer_filter,
        obsm_filter,
        filters,
        true,
        memory_budget,
        MatrixMode::Eager,
        plan,
    )
}

/// Resolve gene names to column indices using the var metadata.
///
/// `preserve_order`: when `true`, the returned indices follow the request
/// order (deduplicated, first occurrence wins); when `false` they are
/// sorted ascending and deduplicated (matching `scx-engine::project_var`).
/// `strict`: when `true`, any name absent from the var metadata raises a
/// `KeyError`; when `false`, unknown names are dropped (and only an
/// all-unknown request errors).
pub(crate) fn resolve_var_names_to_indices(
    reader: &ScxReader,
    names: &[String],
    preserve_order: bool,
    strict: bool,
) -> PyResult<Vec<u32>> {
    let var_batch = match reader.read_var() {
        Ok(batch) => batch,
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {
            return Err(PyRuntimeError::new_err(
                "Cannot resolve var_names: this SCX file has no var metadata. \
                 Open without var_names to load all genes."
                    .to_string(),
            ));
        }
        Err(e) => return Err(to_pyerr(e)),
    };

    // Try to find gene names in the var DataFrame index.
    // The index column is typically the first column (or named "gene_id").
    // We check all string columns.
    let mut name_to_idx: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();

    for col_idx in 0..var_batch.num_columns() {
        let col = var_batch.column(col_idx);
        if let Some(str_arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
            for (row, val) in str_arr.iter().enumerate() {
                if let Some(v) = val {
                    name_to_idx.entry(v).or_insert(row as u32);
                }
            }
        }
    }

    let mut indices = Vec::with_capacity(names.len());
    let mut not_found = Vec::new();
    for name in names {
        match name_to_idx.get(name.as_str()) {
            Some(&idx) => indices.push(idx),
            None => not_found.push(name.as_str()),
        }
    }

    if strict && !not_found.is_empty() {
        return Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "var_names not found in the var metadata: {:?}. \
             Pass strict_var_names=False to silently drop unknown names.",
            not_found
        )));
    }

    if indices.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "None of the requested var_names were found in the var metadata: {:?}",
            not_found
        )));
    }

    if preserve_order {
        // Dedup preserving first occurrence so columns follow request order.
        let mut seen = std::collections::HashSet::new();
        indices.retain(|&i| seen.insert(i));
    } else {
        // Sort + dedup so all callers produce var rows in sorted column-position
        // order, matching scx-engine::project_var(). Keeps eager / backed /
        // query-engine paths consistent under reordered or duplicated requests.
        indices.sort_unstable();
        indices.dedup();
    }

    Ok(indices)
}

/// Build an AnnData object with backed (on-demand) X and layers.
///
/// Opens a new ScxReader (independent mmap) so the backed dataset can
/// outlive the PyExperiment that created it. obs/var/obsm/uns are loaded
/// eagerly (same as non-backed mode).
///
/// When deletion vectors are present, a `kept_to_global` mapping is
/// computed and passed to `ScxBackedSparseDataset` so that user-visible
/// row indices exclude deleted rows (matching non-backed behavior).
#[allow(clippy::too_many_arguments)]
pub fn to_anndata_backed<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    cache_shards: usize,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    filters: SlotFilters<'_>,
    eager: bool,
    preserve_var_order: bool,
    strict_var_names: bool,
) -> PyResult<Bound<'py, PyAny>> {
    to_anndata_backed_with_options(
        py,
        path,
        cache_shards,
        var_names,
        obs_filter,
        layer_filter,
        obsm_filter,
        filters,
        true,
        eager,
        preserve_var_order,
        strict_var_names,
    )
}

/// Internal entrypoint for the backed AnnData builder.
///
/// `apply_deletion_vectors`: when `true` (the default for the public
/// `to_anndata_backed`), the X / obs / obsm / obsp paths are filtered through
/// the file's global deletion vectors. When `false`, the function returns the
/// unfiltered axes — used by `mudata::to_mudata_backed` so that the inner
/// AnnData's `obs` row count matches the outer MuData's global `obs` (which
/// is also unfiltered, matching the eager `to_mudata` path's behaviour).
#[allow(clippy::too_many_arguments)]
pub(crate) fn to_anndata_backed_with_options<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    cache_shards: usize,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    filters: SlotFilters<'_>,
    apply_deletion_vectors: bool,
    eager: bool,
    preserve_var_order: bool,
    strict_var_names: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::backed::{ScxBackedLayerDataset, ScxBackedSparseDataset};
    use crate::lazy_mapping::{
        PairwiseAxis, ScxLazyObsmMapping, ScxLazyPairwiseMapping, ScxLazyVarmMapping,
    };
    use std::sync::Arc;

    let anndata_mod = crate::pyimport::import_module(py, "anndata")?;
    let reader = crate::open_handle_reader(path).map_err(to_pyerr)?;
    // Share one parsed `FullCatalog` across the N+3 `ScxReader`
    // instances this function constructs (main reader + X CSR + CSC
    // sidecar + one per backed layer). The catalog is bytes-identical
    // across all opens of the same file, so re-parsing it N+3 times
    // per worker is pure overhead — the worker-amplification path that
    // motivated the Arc-sharing change. The shard cache and
    // singleflight table stay per-instance; only the immutable
    // catalog is reused. See docs/multithreading.md for the
    // fork-safety contract.
    let shared_catalog = reader.catalog_arc();

    // --- Validate the slot filters FIRST ---
    //
    // Before obs / var are read, before the backed X and CSC readers open, and
    // before `build_eager_obsm_dict` decodes every obsm entry. A typo is a
    // bad request, and answering it should not first cost a full obsm decode on
    // an atlas-scale file. All five listings are catalog-only scans.
    let obsp_names = reader.list_obsp();
    let varp_names = reader.list_varp();
    let varm_names = reader.list_varm();
    validate_slot_keys(&reader.layer_names(), layer_filter, "layers")?;
    validate_slot_keys(&reader.list_obsm(), obsm_filter, "obsm")?;
    validate_slot_keys(&obsp_names, filters.obsp, "obsp")?;
    validate_slot_keys(&varp_names, filters.varp, "varp")?;
    validate_slot_keys(&varm_names, filters.varm, "varm")?;

    // --- Compute kept_to_global from deletion vectors (if present) ---
    // Cache the deletion-vector-only mapping; obs_filter may mutate kept_to_global
    // further, but obsm filtering needs the original DV-only version.
    // Skipped when `apply_deletion_vectors` is false (e.g. `to_mudata_backed`
    // single-modality wrap, where DVs are intentionally not applied to keep
    // inner-AnnData obs in lockstep with the unfiltered outer MuData obs).
    let dv_kept_to_global = if apply_deletion_vectors {
        compute_kept_to_global(&reader)?
    } else {
        None
    };
    let mut kept_to_global = dv_kept_to_global.clone();

    // --- obs (eager, optionally filtered by deletion vectors) ---
    let obs = match reader.read_obs() {
        Ok(batch) => {
            let filtered_batch = if apply_deletion_vectors {
                filter_obs_by_deletion_vectors(&reader, batch)?
            } else {
                batch
            };
            let table = record_batch_to_pyarrow(py, &filtered_batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // --- Apply obs_filter if specified ---
    // Evaluate on the pandas DataFrame rather than QueryPipeline. In backed
    // mode, shard-level pushdown has negligible benefit since X is lazy (only
    // accessed shards are decoded). Pandas .query() is simpler and supports
    // richer expressions.
    let obs = if let Some(expr) = obs_filter {
        if let Some(obs_df) = obs {
            // Use pandas query to filter
            let filtered = obs_df.call_method1("query", (expr,))?;
            let original_idx = obs_df.getattr("index")?;
            let filtered_idx = filtered.getattr("index")?;

            // Positional indices of the kept rows in the (already
            // deletion-filtered) obs, composed onto the visible → global map.
            let isin_mask = original_idx.call_method1("isin", (&filtered_idx,))?;
            let positions = mask_true_positions(py, &isin_mask)?;
            kept_to_global = Some(compose_kept_to_global(kept_to_global.as_ref(), &positions));

            Some(filtered)
        } else {
            None
        }
    } else {
        obs
    };

    // --- Resolve var_names to column indices ---
    // `col_indices` follows request order when preserve_var_order is set
    // (the X projection then presents columns in that order via
    // set_col_projection_ordered, and var is sliced with the same iloc).
    let col_indices = if let Some(names) = var_names {
        Some(resolve_var_names_to_indices(
            &reader,
            names,
            preserve_var_order,
            strict_var_names,
        )?)
    } else {
        None
    };

    // --- X: backed ---
    let has_csc = reader.header().has_csc();
    let x_backed =
        open_backed_matrix_reader(path, Arc::clone(&shared_catalog), None, cache_shards)?;
    let x_backed_csc: Option<Arc<scx_format_io::BackedCscReader>> = if has_csc {
        // Open a separate ScxReader for the CSC sidecar (BackedCscReader
        // takes ownership). Header check is cheap; the reader holds a
        // mmap and per-shard catalog, but no shards decode until we
        // actually call read_csc_shard(). Catalog parse is skipped via
        // the shared `Arc<FullCatalog>`.
        let csc_reader = crate::open_handle_reader_shared(path, Arc::clone(&shared_catalog))
            .map_err(to_pyerr)?;
        Some(Arc::new(
            scx_format_io::BackedCscReader::new(csc_reader, cache_shards).map_err(to_pyerr)?,
        ))
    } else {
        None
    };
    let mut x_dataset = match &kept_to_global {
        Some(mapping) => ScxBackedSparseDataset::from_reader_with_deletions(
            Arc::clone(&x_backed),
            mapping.clone(),
        ),
        None => ScxBackedSparseDataset::from_reader(Arc::clone(&x_backed)),
    };
    x_dataset.with_csc_reader(x_backed_csc);
    x_dataset.with_source_path(path);
    install_projection(&mut x_dataset, col_indices.as_ref(), preserve_var_order);

    // --- var (eager, optionally filtered by var_names) ---
    let var = match reader.read_var() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            let df = pyarrow_table_to_pandas(&table)?;
            if let Some(ref indices) = col_indices {
                // Slice var positionally with the same indices used to project
                // X (set_col_projection above). Using df.iloc keeps var aligned
                // with X when names match a non-index column like gene_symbol;
                // the prior var.index.isin(names) approach produced an empty
                // var when symbols were resolved from non-index columns.
                let np_indices = PyArray1::from_slice(py, indices);
                let iloc = df.getattr("iloc")?;
                let filtered = iloc.get_item(np_indices)?;
                Some(filtered)
            } else {
                Some(df)
            }
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // --- obsm ---
    //
    // Backed dense row-gather (`ScxBackedObsmDataset`) kicks in when the
    // caller selected obsm keys (`obsm=[...]`), did not force `eager`,
    // and did not pass `obs_filter`. Under `obs_filter` we fall back to
    // the eager path (composing a pandas-query row mask with shard
    // gather is deferred), and `obsm=None` keeps the historical
    // eager-all behaviour.
    let use_backed_obsm = obsm_filter.is_some() && !eager && obs_filter.is_none();
    // Under backed mode nothing is read here — the `ScxBackedObsmDataset` bridge
    // built below gathers each key on demand.
    let obsm_dict = pyo3::types::PyDict::new(py);
    if !use_backed_obsm {
        build_eager_obsm_dict(
            py,
            &reader,
            obsm_filter,
            obs_filter,
            apply_deletion_vectors,
            &kept_to_global,
            &dv_kept_to_global,
            &obsm_dict,
        )?;
    }

    let lazy_obsm = if use_backed_obsm {
        let obsm_reader = Arc::new(
            crate::open_handle_reader_shared(path, Arc::clone(&shared_catalog))
                .map_err(to_pyerr)?,
        );
        // obs_filter is None here, so kept_to_global == dv_kept_to_global
        // (deletion vectors only).
        let kept_arc = kept_to_global.as_ref().map(|k| Arc::new(k.clone()));
        let config = crate::lazy_mapping::BackedObsmConfig {
            path: path.to_path_buf(),
            cache_shards,
            shared_catalog: Arc::clone(&shared_catalog),
            kept_to_global: kept_arc,
        };
        Some(ScxLazyObsmMapping::new_backed(
            obsm_reader,
            obsm_filter,
            config,
        ))
    } else {
        None
    };

    // --- obsp / varp / varm — lazy bridges by default ---
    //
    // Same `Arc<ScxReader>` (sibling of the main one, sharing the
    // parsed catalog) backs all three bridges; refcount-only clones
    // when handing it to each `ScxLazyPairwiseMapping` /
    // `ScxLazyVarmMapping`. Each bridge decodes its sections on the
    // consumer's first `ad.obsp[…]` / `.varp[…]` / `.varm[…]` access.
    // See `to_anndata_with_layers` for the contract between
    // `eager=true/false` and AnnData's private `_obsp` / `_varp` /
    // `_varm` storage.
    // Filter-aware, exactly as on the eager path: an empty list builds no
    // bridge. Keys were validated at the top of this function, before any
    // payload read.
    let has_obsp = slot_has_selected(&obsp_names, filters.obsp);
    let has_varp = slot_has_selected(&varp_names, filters.varp);
    let has_varm = slot_has_selected(&varm_names, filters.varm);
    let need_lazy_aligned = has_obsp || has_varp || has_varm;
    let lazy_reader: Option<Arc<ScxReader>> = if need_lazy_aligned {
        Some(Arc::new(
            crate::open_handle_reader_shared(path, Arc::clone(&shared_catalog))
                .map_err(to_pyerr)?,
        ))
    } else {
        None
    };
    // Backed-path obsp filter mirrors the eager pre-fix logic in this
    // module (kept_to_global composes deletion vectors with
    // obs_filter); shared by Arc so the bridge holds its own ref.
    let kept_to_global_arc = kept_to_global.as_ref().map(|k| Arc::new(k.clone()));
    let lazy_obsp = lazy_reader.as_ref().filter(|_| has_obsp).map(|r| {
        ScxLazyPairwiseMapping::new(
            Arc::clone(r),
            PairwiseAxis::Obsp,
            kept_to_global_arc.clone(),
            filters.obsp,
        )
    });
    // varp / varm decode at physical var width, so an open-time `var_names=`
    // projection has to be seeded into the bridge — otherwise a later
    // `filter_genes` replays visible-space indices against a physical-width
    // value and silently returns the wrong genes' rows.
    let lazy_varp = lazy_reader.as_ref().filter(|_| has_varp).map(|r| {
        ScxLazyPairwiseMapping::new(Arc::clone(r), PairwiseAxis::Varp, None, filters.varp)
            .with_var_projection(col_indices.as_deref())
    });
    let lazy_varm = lazy_reader.as_ref().filter(|_| has_varm).map(|r| {
        ScxLazyVarmMapping::new(Arc::clone(r), filters.varm)
            .with_var_projection(col_indices.as_deref())
    });

    // --- uns (eager; tagged envelopes reconstructed) ---
    let uns_dict = read_uns_as_pyobject(py, &reader, 0)?;

    // --- layers (backed, with optional filtering) ---
    let all_layer_names = reader.layer_names();
    let layers_dict = pyo3::types::PyDict::new(py);
    for name in &all_layer_names {
        // Skip layers not in the filter list (if specified)
        if let Some(filter) = layer_filter {
            if !filter.iter().any(|f| f == name) {
                continue;
            }
        }
        let l_backed =
            open_backed_matrix_reader(path, Arc::clone(&shared_catalog), Some(name), cache_shards)?;
        let mut l_dataset = match &kept_to_global {
            Some(mapping) => ScxBackedLayerDataset::from_reader_with_deletions(
                l_backed,
                name.clone(),
                mapping.clone(),
            ),
            None => ScxBackedLayerDataset::from_reader(l_backed, name.clone()),
        };
        install_projection(
            &mut l_dataset.inner,
            col_indices.as_ref(),
            preserve_var_order,
        );
        let l_py = l_dataset.into_pyobject(py)?;
        layers_dict.set_item(name, l_py)?;
    }

    // Build AnnData kwargs
    let kwargs = pyo3::types::PyDict::new(py);
    let x_py = x_dataset.into_pyobject(py)?;
    kwargs.set_item("X", x_py)?;
    if let Some(obs) = obs {
        kwargs.set_item("obs", obs)?;
    }
    if let Some(var) = var {
        kwargs.set_item("var", var)?;
    }
    if !obsm_dict.is_empty() {
        kwargs.set_item("obsm", obsm_dict)?;
    }
    if let Some(uns) = uns_dict {
        kwargs.set_item("uns", uns)?;
    }
    if !layers_dict.is_empty() {
        kwargs.set_item("layers", layers_dict)?;
    }
    if eager {
        // Eager mode: materialize lazy bridges up front and pass
        // through normal kwargs path. Caller receives an AnnData
        // detached from the SCX file handle.
        if let Some(m) = &lazy_obsp {
            kwargs.set_item("obsp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varp {
            kwargs.set_item("varp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varm {
            kwargs.set_item("varm", m.materialize_all(py)?)?;
        }
    }

    let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;

    // Backed mode does not reconstruct the raw matrix — warn + drop.
    // `raw=False` opts out of the notice.
    if filters.raw && reader.has_raw() {
        warn_python_convert(
            py,
            &scx_convert::ConvertWarning::DroppedRaw {
                raw_n_vars: reader.raw_n_vars().unwrap_or(0),
            },
        )?;
    }

    if !eager {
        // Lazy mode: attach bridges directly to AnnData's private
        // storage to bypass `AlignedMappingProperty.__set__`'s eager
        // validation. See [`to_anndata_with_layers`].
        if let Some(m) = lazy_obsp {
            adata.setattr("_obsp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varp {
            adata.setattr("_varp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varm {
            adata.setattr("_varm", m.into_pyobject(py)?)?;
        }
        // Backed dense row-gather obsm (only built when obsm was
        // selected, not eager, no obs_filter). Attach via `_obsm` to
        // bypass AnnData's axis-length validation, like the other
        // bridges.
        if let Some(m) = lazy_obsm {
            adata.setattr("_obsm", m.into_pyobject(py)?)?;
        }
    }

    Ok(adata)
}

/// Compute `kept_to_global` mapping from deletion vectors.
///
/// Returns `None` if there are no deletions. Otherwise returns a Vec
/// where `kept_to_global[i]` is the global (file-level) row index for
/// user-visible row `i`.
pub(crate) fn compute_kept_to_global(reader: &ScxReader) -> PyResult<Option<Vec<u64>>> {
    // Reuse the shared obs-indexed keep mask, then compress to the global row
    // indices of the surviving rows (same source of truth as the reader CSR
    // filter, `scx compact`, and the streaming export).
    let keep = match reader.deletion_keep_mask().map_err(to_pyerr)? {
        Some(keep) => keep,
        None => return Ok(None),
    };

    let kept: Vec<u64> = (0..keep.len())
        .filter(|&i| keep[i])
        .map(|i| i as u64)
        .collect();

    Ok(Some(kept))
}

/// Accept a boolean row mask in either obs row space and return it
/// physical-length.
///
/// The length rule is the one the in-place obs writers use for a positional
/// frame (`scx_ops::classify_obs_frame_length`): `n_obs_physical` entries
/// (`header.n_obs`; what `read_obs(logical=False)` describes) are returned as
/// they are; the live count (what `read_obs()` describes since 0.17) is
/// expanded through the keep mask — a live entry lands on its physical row, an
/// already-deleted row gets `false`. Any other length is a `ValueError` naming
/// both counts and the read that yields each. `what` names the argument.
pub(crate) fn physical_row_mask(
    reader: &ScxReader,
    mask: &[bool],
    what: &str,
) -> PyResult<Vec<bool>> {
    let n_physical = reader.n_obs();
    let keep = if mask.len() as u64 == n_physical {
        None
    } else {
        reader.deletion_keep_mask().map_err(to_pyerr)?
    };
    let space =
        scx_ops::classify_obs_frame_length(what, mask.len() as u64, n_physical, keep.as_deref())
            .map_err(|e| match e {
                // The ops error's Display prefixes "CSR shape mismatch on append:",
                // which is about matrices; a mask wants the bare sentence.
                scx_ops::OpsError::ShapeMismatch { detail } => {
                    pyo3::exceptions::PyValueError::new_err(detail)
                }
                other => crate::ops::ops_to_pyerr(other),
            })?;
    match (space, keep) {
        (scx_ops::ObsFrameRowSpace::Live, Some(keep)) => {
            let mut live = mask.iter();
            Ok(keep
                .iter()
                .map(|&k| {
                    if k {
                        *live.next().expect("n_live entries")
                    } else {
                        false
                    }
                })
                .collect())
        }
        _ => Ok(mask.to_vec()),
    }
}

/// Refuse a row mask whose pandas index says its rows were reordered.
///
/// Dispatch by length cannot see a mask built from a frame that was sorted or
/// reindexed after `read_obs()` — right length, every entry on the wrong cell,
/// and the sink here is a deletion or an export. So when the caller hands a
/// pandas Series with a labelled index, `labels` (one per mask entry) are
/// compared with the file's obs-index barcodes in the row space the mask's
/// length names: the same barcodes in a different order are refused naming the
/// first misplaced row; labels that are not the file's barcodes are ignored
/// (the mask may have been built from another source), and a RangeIndex Series
/// or a bare array hands no labels and is not checked. A multi-level index
/// compares as the composite key the key join builds.
pub(crate) fn check_row_mask_order(
    reader: &ScxReader,
    labels: &[String],
    what: &str,
) -> PyResult<()> {
    let n_physical = reader.n_obs() as usize;
    let schema = reader.read_obs_schema_physical().map_err(to_pyerr)?;
    let index_cols: Vec<String> = scx_format_io::resolve_index_columns(&schema)
        .into_iter()
        .filter(|c| schema.index_of(c).is_ok())
        .collect();
    if index_cols.is_empty() {
        return Ok(());
    }
    let batch = if labels.len() == n_physical {
        reader.read_obs_keys(&index_cols)
    } else {
        reader.read_obs_keys_filtered(&index_cols)
    }
    .map_err(to_pyerr)?;
    if batch.num_rows() != labels.len() {
        // Neither row space — `physical_row_mask` reports that with both counts.
        return Ok(());
    }
    let file_keys = if index_cols.len() == 1 {
        scx_ops::obs_key_values(&batch, &index_cols[0])
    } else {
        scx_ops::build_composite_key(&batch, &index_cols)
    }
    .map_err(crate::ops::ops_to_pyerr)?;
    if file_keys == labels {
        return Ok(());
    }
    let mut a: Vec<&str> = file_keys.iter().map(String::as_str).collect();
    let mut b: Vec<&str> = labels.iter().map(String::as_str).collect();
    a.sort_unstable();
    b.sort_unstable();
    if a != b {
        return Ok(());
    }
    let row = file_keys
        .iter()
        .zip(labels.iter())
        .position(|(f, l)| f != l)
        .expect("vectors differ");
    let sep = scx_ops::COMPOSITE_KEY_SEPARATOR;
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "{what}: the mask's index holds the file's own barcodes in a different order — entry \
         {row} is '{}' but obs row {row} ('{}') is '{}'. A mask built from a frame sorted or \
         reindexed after read_obs() would address the wrong cells; restore the original order \
         (or pass a plain array in the file's row order)",
        labels[row].replace(sep, " / "),
        index_cols.join("+"),
        file_keys[row].replace(sep, " / ")
    )))
}

/// Filter an obs RecordBatch to exclude deleted rows.
///
/// Builds a boolean keep-mask from the deletion vectors (same logic
/// as `read_all_csr_shards_filtered`) and applies
/// `arrow::compute::filter_record_batch`.
pub(crate) fn filter_obs_by_deletion_vectors(
    reader: &ScxReader,
    obs: arrow::array::RecordBatch,
) -> PyResult<arrow::array::RecordBatch> {
    // Shared obs-indexed keep mask (same logic as the reader CSR filter,
    // `scx compact`, and the streaming export); see
    // `ScxReader::deletion_keep_mask`.
    let keep = match reader.deletion_keep_mask().map_err(to_pyerr)? {
        Some(keep) => keep,
        None => return Ok(obs), // No deletions — return as-is
    };

    let bool_array = arrow::array::BooleanArray::from(keep);
    arrow::compute::filter_record_batch(&obs, &bool_array)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to filter obs: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate that decides whether a narrow-dtype read may take the projected
    /// (f32) assembler. Its falsifying case is unreachable from Python — every
    /// pyscx write door routes X through f32, so an *odd* count above 2²⁴
    /// cannot be put in a file from that side and the Python suite can only
    /// pin the accept half. Pin both halves here instead.
    #[test]
    fn f32_roundtrip_exactness_matches_the_decode_loss_guard() {
        // Float-encoded shards record no integer maximum.
        assert!(f32_roundtrip_is_exact(0));
        assert!(f32_roundtrip_is_exact(1));
        // 2²⁴ itself is representable; the guard uses the same inclusive bound,
        // so the two cannot disagree about which files are exact.
        assert!(f32_roundtrip_is_exact(scx_codec::F32_MAX_EXACT_INT));
        assert!(scx_codec::guard_f32_decode_loss(scx_codec::F32_MAX_EXACT_INT, false).is_ok());
        // One above, and the projection must stand down.
        assert!(!f32_roundtrip_is_exact(scx_codec::F32_MAX_EXACT_INT + 1));
        assert!(scx_codec::guard_f32_decode_loss(scx_codec::F32_MAX_EXACT_INT + 1, false).is_err());
        assert!(!f32_roundtrip_is_exact(u32::MAX));
    }
}
