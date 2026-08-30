//! `MultimodalTrainingDataset` and the per-modality budget split.
//!
//! Split out of `python.rs` by ORG-9.10-2; a pure move.

use std::collections::HashMap;

use numpy::PyArrayMethods;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::batch::{Batch, ObsColumn};
use crate::error::LoaderError;
use crate::pipeline::{
    check_uniform_modality_layouts, resolve_loader_config, LoaderConfig, TrainingPipeline,
};

use super::*;

/// Split a total `max_memory_mb` across modalities in proportion to their nnz,
/// with a per-modality floor.
///
/// The floor is load-bearing, not slack: `LoaderConfig::validate` rejects
/// anything below `MIN_MODALITY_BUDGET_MB`, so a modality whose proportional
/// share falls under it would fail construction outright. The consequence is
/// that **the shares can sum to more than `total_mb`** — three modalities under
/// a 64 MB request budget 192 MB between them. That is a real over-budget, it
/// is the price of not refusing the file, and
/// `MultimodalTrainingDataset.memory_budget()` reports it rather than leaving
/// it silent.
///
/// Pure (no Python), so it is unit-testable without a Python interpreter.
fn split_budget_across_modalities(total_mb: usize, per_modality_nnz: &[u64]) -> Vec<usize> {
    let total_nnz: u64 = per_modality_nnz.iter().sum::<u64>().max(1);
    per_modality_nnz
        .iter()
        .map(|nnz| {
            let share = (total_mb as f64) * (*nnz as f64) / (total_nnz as f64);
            (share as usize).max(MIN_MODALITY_BUDGET_MB)
        })
        .collect()
}

/// Per-modality budget floor. Equal to the minimum `LoaderConfig::validate`
/// accepts, which is what makes it a floor rather than a preference.
pub(super) const MIN_MODALITY_BUDGET_MB: usize = 64;

/// The smallest `max_memory_mb` whose *proportional* shares all clear the floor,
/// i.e. the smallest request `split_budget_across_modalities` returns unchanged.
///
/// `None` when no such total exists: a modality with zero nnz gets a zero share
/// at every budget, so raising the request can never lift it off the floor.
///
/// This is what the warning must quote. Advising the current *effective sum*
/// instead does not converge — the floor is re-applied after the new split, so
/// with a 1:99 nnz ratio a 64 MB request reports 128, and passing 128 reports
/// 190, and so on. The fixed point is `max_i(ceil(floor × total_nnz / nnz_i))`.
pub(super) fn min_total_mb_clearing_the_floor(per_modality_nnz: &[u64]) -> Option<usize> {
    let total_nnz: u64 = per_modality_nnz.iter().sum();
    if total_nnz == 0 {
        return None;
    }
    let mut required = 0usize;
    for &nnz in per_modality_nnz {
        if nnz == 0 {
            return None;
        }
        let need = (MIN_MODALITY_BUDGET_MB as u128)
            .saturating_mul(total_nnz as u128)
            .div_ceil(nnz as u128);
        required = required.max(need.min(usize::MAX as u128) as usize);
    }
    Some(required)
}

/// Multimodal training dataset.
///
/// Wraps N independent `TrainingPipeline` instances — one per requested
/// modality — and yields per-batch dicts whose cell axes align across
/// modalities. Cells (obs) are global across modalities, so all
/// pipelines see the same `n_obs` and the same shuffler seed produces
/// the same row ordering when their per-modality shard layouts agree
/// (the standard CITE-seq / multiome writer guarantees this).
///
/// On `__next__`, returns
/// `{"X": {modality_name: ndarray, ...}, "obs": {...}, "cell_indices": ndarray}`
/// when constructed with `return_dict=True` (default), or a tuple
/// `(X_modality_0, X_modality_1, ...)` when `return_dict=False`. The
/// per-modality X arrays share the same `cell_indices` row ordering;
/// the wrapper validates this on each batch and raises
/// `RuntimeError` if the per-modality shufflers diverge (e.g. because
/// the modalities have different shard layouts on disk — typically a
/// writer / file-construction bug).
#[pyclass]
pub struct MultimodalTrainingDataset {
    /// One pipeline per requested modality, in the order the user
    /// supplied them.
    pipelines: Vec<TrainingPipeline>,
    /// Modality names parallel to `pipelines`.
    modality_names: Vec<String>,
    /// True → batches are dicts. False → batches are tuples of X arrays.
    return_dict: bool,
    epoch_started: bool,
    /// `close()` was the last lifecycle action — see the `closed` getter.
    /// Not terminal here either: `__iter__` clears it and rebuilds.
    closed: bool,
    /// `max_memory_mb` **as requested**, before the per-modality split. Kept so
    /// `memory_budget()` can report the request beside what the floors actually
    /// budgeted — the two are not the same number and nothing used to say so.
    requested_max_memory_mb: usize,
    creation_pid: u32,
}

#[pymethods]
impl MultimodalTrainingDataset {
    /// Create a new `MultimodalTrainingDataset` from a multimodal SCX file.
    ///
    /// Args:
    ///     path: Path to the .scx file.
    ///     modalities: List of modality names to load (e.g.
    ///         `["rna", "adt"]`). All listed modalities must exist
    ///         in the file's modality table.
    ///     batch_size: Mini-batch size shared across modalities.
    ///     hvg_indices: Optional HVG projection. Currently applied
    ///         to every modality identically; for per-modality
    ///         projections, instantiate separate
    ///         `TrainingDataset(modality=…, hvg_indices=…)` instances
    ///         and zip in Python. Because one panel spans modalities of
    ///         differing widths, this is the ONE loader that does not
    ///         range-check the panel: an index past a given modality's
    ///         n_vars yields a silently always-zero column there. The
    ///         per-modality `TrainingDataset` route above is checked.
    ///         Sorted and deduplicated as everywhere else, so batch columns
    ///         are in ascending gene-index order regardless of the order
    ///         passed; a panel that is not already ascending-unique gets a
    ///         UserWarning (once, since the panel is shared).
    ///     obs_columns: Obs columns to include in each batch (read
    ///         once from the global obs table).
    ///     return_dict: If True (default), yield
    ///         `{"X": {name: ndarray}, "obs": {...}, "cell_indices": ...}`.
    ///         If False, yield a tuple `(X_0, X_1, …)` aligned with
    ///         `modalities` order.
    ///     normalize, log1p, target_sum, pflog, pflog_alpha,
    ///     shard_group_size, prefetch_batches, seed, max_memory_mb: see
    ///     `TrainingDataset` for semantics — applied to every modality
    ///     uniformly. `pflog` (RNA-appropriate) replaces normalize/log1p
    ///     when set (a modality-scoped loader must pin `pflog_alpha`). An explicit `max_memory_mb` is divided across modalities
    ///     proportionally to per-modality nnz (modalities with denser X get a
    ///     larger share of the memory budget). When omitted, each modality's
    ///     per-modality share becomes an adaptive floor (raised to fit that
    ///     modality's full-width configuration), matching `TrainingDataset`'s
    ///     default behaviour.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        path,
        modalities,
        batch_size=None,
        hvg_indices=None,
        obs_columns=None,
        return_dict=None,
        normalize=None,
        log1p=None,
        target_sum=None,
        pflog=None,
        pflog_alpha=None,
        shard_group_size=None,
        prefetch_batches=None,
        seed=None,
        max_memory_mb=None,
    ))]
    fn new(
        py: Python<'_>,
        path: &str,
        modalities: Vec<String>,
        batch_size: Option<usize>,
        hvg_indices: Option<Vec<u32>>,
        obs_columns: Option<Vec<String>>,
        return_dict: Option<bool>,
        normalize: Option<bool>,
        log1p: Option<bool>,
        target_sum: Option<f64>,
        pflog: Option<bool>,
        pflog_alpha: Option<f64>,
        shard_group_size: Option<usize>,
        prefetch_batches: Option<usize>,
        seed: Option<u64>,
        max_memory_mb: Option<usize>,
    ) -> PyResult<Self> {
        use scx_format_io::ScxReader;
        if modalities.is_empty() {
            return Err(PyRuntimeError::new_err(
                "MultimodalTrainingDataset: `modalities` must contain at least one name",
            ));
        }

        // Open the file once to resolve modality_id + per-modality nnz
        // for the memory-budget split. The inner `TrainingPipeline::new`
        // re-opens the file per modality.
        let reader = ScxReader::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if !reader.is_multimodal() {
            return Err(PyRuntimeError::new_err(format!(
                "MultimodalTrainingDataset: file '{path}' is single-modality. \
                 Use TrainingDataset(path) instead."
            )));
        }
        let mut resolved: Vec<(String, u8, u64)> = Vec::with_capacity(modalities.len());
        for name in &modalities {
            let mid = reader.modality_id(name).ok_or_else(|| {
                let registered: Vec<&str> = reader.modality_names();
                PyRuntimeError::new_err(format!(
                    "MultimodalTrainingDataset: file '{path}' does not have a \
                     modality named '{name}'. Registered modalities: {registered:?}"
                ))
            })?;
            let info = reader.modality_info(mid).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "MultimodalTrainingDataset: failed to resolve modality_info for '{name}'"
                ))
            })?;
            resolved.push((name.clone(), mid, info.nnz));
        }

        // Memory budget split. The user-supplied `max_memory_mb` is
        // total across all modalities; allocate proportionally to
        // per-modality nnz. Each modality gets at least 64 MB so the
        // budget tuner has room to settle on viable shard_group_size /
        // prefetch_batches values.
        let defaults = LoaderConfig::default();
        let total_mb = max_memory_mb.unwrap_or(defaults.max_memory_mb);
        let per_modality_nnz: Vec<u64> = resolved.iter().map(|(_, _, n)| *n).collect();
        let per_modality_mb = split_budget_across_modalities(total_mb, &per_modality_nnz);

        // Per-modality shard-layout alignment check (fail loud at
        // construction rather than mid-`__next__`). The per-modality
        // shufflers can only agree on row ordering when the shard
        // layouts match; a genuine layout mismatch (different shard
        // counts or row ranges) can never produce aligned batches.
        check_uniform_modality_layouts(reader.catalog(), &resolved)
            .map_err(|e| PyRuntimeError::new_err(format!("MultimodalTrainingDataset: {e}")))?;

        // Build one pipeline per modality. All pipelines share the same
        // seed, and the shard layouts are validated identical above, so
        // the per-modality Level-1/Level-2 shufflers agree on row
        // ordering. Batching, however, is chunked by each pipeline's
        // *effective* `batch_size`, which the per-modality memory-budget
        // auto-tuner (`compute_memory_budget`) can shrink independently:
        // a wide modality (e.g. ATAC, ~144k genes) can be forced below
        // the `ADAPTIVE_BUDGET_CAP_MB` ceiling while a narrow one (e.g.
        // RNA) keeps the requested batch. Different effective batch sizes
        // desync the per-batch `cell_indices` and trip the alignment
        // check in `__next__`. To prevent that, build once to discover
        // each modality's effective batch/shard_group_size, then pin all
        // pipelines to the minimum across modalities (always feasible —
        // each modality already fit its own larger effective config).
        drop(reader);
        let names: Vec<String> = resolved.iter().map(|(n, _, _)| n.clone()).collect();
        // Resolve `obs_columns` once (not per modality inside the map).
        let obs_columns = obs_columns.unwrap_or_default();
        let base_configs: Vec<LoaderConfig> = resolved
            .iter()
            .zip(per_modality_mb)
            // `max_memory_mb: modality_mb` — no explicit total budget → adaptive
            // per-modality floor: each modality's pipeline raises to fit its own
            // full-width configuration rather than shrinking the batch.
            .map(|((_, mid, _), modality_mb)| {
                resolve_loader_config(
                    batch_size,
                    shard_group_size,
                    prefetch_batches,
                    normalize,
                    log1p,
                    target_sum,
                    pflog,
                    pflog_alpha,
                    seed,
                    hvg_indices.clone(),
                    obs_columns.clone(),
                    modality_mb,
                    max_memory_mb.is_none(),
                    Some(*mid),
                    // The one caller that shares a single panel across
                    // modalities of differing widths.
                    /*shared_hvg_panel=*/
                    true,
                )
            })
            .collect();

        // Consumed, not borrowed-and-cloned: nothing rebuilds from these any
        // more now that the uniform pin happens in place.
        let mut pipelines = Vec::with_capacity(base_configs.len());
        for config in base_configs {
            let pipeline = TrainingPipeline::new(path, config)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            pipelines.push(pipeline);
        }

        // One shared panel, so one warning — emit it off the first pipeline
        // rather than once per modality.
        if let Some(v) = pipelines.first().and_then(|p| p.hvg_panel()) {
            warn_hvg_panel(py, "MultimodalTrainingDataset", &v)?;
        }

        // Pin uniform effective batch_size + shard_group_size (min across
        // modalities), in place. Each modality already fit its own effective
        // (>=) config and the model is monotone, so the minimum always still
        // fits — `TrainingPipeline::pin_effective_config` enforces the "only
        // shrink" half of that as a real error rather than the release-stripped
        // `debug_assert` this block used to end with, and the reduction chain is
        // asserted monotone per step inside `budget::tune`.
        let common_batch = pipelines
            .iter()
            .map(|p| p.effective_batch_size())
            .min()
            .unwrap_or_else(|| batch_size.unwrap_or(defaults.batch_size));
        let common_sgs = pipelines
            .iter()
            .map(|p| p.memory_budget_info().shard_group_size)
            .min()
            .unwrap_or_else(|| shard_group_size.unwrap_or(defaults.shard_group_size));
        let needs_repin = pipelines.iter().any(|p| {
            p.effective_batch_size() != common_batch
                || p.memory_budget_info().shard_group_size != common_sgs
        });
        if needs_repin {
            let requested_batch = batch_size.unwrap_or(defaults.batch_size);
            let requested_sgs = shard_group_size.unwrap_or(defaults.shard_group_size);
            if common_batch < requested_batch {
                log::info!(
                    "MultimodalTrainingDataset: pinning uniform effective batch_size \
                     {common_batch} (requested {requested_batch}) across modalities so a wide \
                     modality's memory-budget auto-tune does not desync per-modality batching. \
                     Pass a larger max_memory_mb to keep the requested batch.",
                );
            }
            if common_sgs < requested_sgs {
                // A pinned shard_group_size below what a narrow modality would
                // pick lowers its within-group shuffle entropy (the per-pipeline
                // `shuffle_quality_degraded` warn only fires for modalities the
                // tuner itself shrank, so surface the cross-modality pin here).
                log::info!(
                    "MultimodalTrainingDataset: pinning uniform effective shard_group_size \
                     {common_sgs} (requested {requested_sgs}) across modalities; narrow \
                     modalities see lower within-group shuffle entropy as a result. Pass a \
                     larger max_memory_mb to keep a larger shard_group_size.",
                );
            }
            for p in pipelines.iter_mut() {
                p.pin_effective_config(common_batch, common_sgs)
                    .map_err(loader_err_to_py)?;
            }
        }

        let effective_total_mb: usize = pipelines
            .iter()
            .map(|p| p.max_memory_mb())
            .fold(0usize, |a, b| a.saturating_add(b));
        // Only on an *explicit* request. With `max_memory_mb=None` every
        // modality's pipeline resolves its own adaptive budget and the sum
        // routinely exceeds the 512 MB default — that is the adaptive policy
        // working, and warning there would fire on every default construction.
        // Same don't-cry-wolf rule `assess_cache_sizing` applies.
        if max_memory_mb.is_some() && effective_total_mb > total_mb {
            warn_modality_budget_floors(py, total_mb, effective_total_mb, &per_modality_nnz)?;
        }

        // Per-modality `budget_exceeded`, for the same reason `TrainingDataset`
        // warns: a modality that cannot fit even at its minimums will exceed the
        // budget it was given, and a `log::warn!` is invisible in a notebook.
        // **After** the pin, never before — the pin shrinks knobs further and can
        // clear the flag on a modality the tuner had already flagged.
        for (name, p) in names.iter().zip(pipelines.iter()) {
            if p.memory_budget_info().budget_exceeded {
                warn_budget_exceeded(
                    py,
                    &format!("MultimodalTrainingDataset modality '{name}'"),
                    p.max_memory_mb(),
                    p.memory_budget_info(),
                )?;
            }
        }

        Ok(MultimodalTrainingDataset {
            pipelines,
            modality_names: names,
            return_dict: return_dict.unwrap_or(true),
            epoch_started: false,
            closed: false,
            requested_max_memory_mb: total_mb,
            creation_pid: std::process::id(),
        })
    }

    /// Start a new epoch across every modality. Legal after `close()` — see
    /// [`TrainingDataset::__iter__`].
    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        for p in slf.pipelines.iter_mut() {
            p.start_epoch()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        slf.epoch_started = true;
        slf.closed = false;
        Ok(slf)
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.MultimodalTrainingDataset requires num_workers=0. The Rust pipeline manages its \
                 own threads.",
            ));
        }
        if !self.epoch_started {
            return Err(PyRuntimeError::new_err(
                "Must call __iter__ before __next__",
            ));
        }

        // Pull one batch from each modality in lockstep. Releasing the
        // GIL once around all pulls is the simplest correct
        // implementation; with the same seed the pipelines should
        // produce in roughly aligned cadence.
        let batches_res: std::result::Result<Option<Vec<Batch>>, LoaderError> = py.detach(|| {
            let mut out = Vec::with_capacity(self.pipelines.len());
            for p in self.pipelines.iter_mut() {
                match p.next_batch()? {
                    Some(b) => out.push(b),
                    None => return Ok(None),
                }
            }
            Ok(Some(out))
        });

        let batches = match batches_res {
            Ok(Some(b)) => b,
            Ok(None) => {
                self.epoch_started = false;
                return Ok(None);
            }
            Err(e) => {
                // A mid-epoch fault in any modality's pipeline: end iteration
                // and surface the error rather than silently truncating.
                self.epoch_started = false;
                return Err(loader_err_to_py(e));
            }
        };

        // Cross-modality cell-axis alignment check. The per-modality
        // shufflers should produce identical row orderings when their
        // shard layouts match; if they don't, surface a clear error
        // rather than silently emit mis-aligned batches.
        let first_indices = &batches[0].cell_indices;
        for (i, b) in batches.iter().enumerate().skip(1) {
            if b.cell_indices != *first_indices {
                return Err(PyRuntimeError::new_err(format!(
                    "MultimodalTrainingDataset: modalities '{}' and '{}' produced \
                     different cell_indices on the same batch — this typically means \
                     the per-modality shard layouts disagree (different shard counts \
                     or row ranges). Reshard the file with a uniform shard_target_rows \
                     before training.",
                    self.modality_names[0], self.modality_names[i],
                )));
            }
        }

        // Build the output. Both the dict and tuple paths share the
        // first batch's obs / cell_indices (cells are global, so the
        // obs record is identical across modalities).
        let dict = build_multimodal_batch_dict(py, &batches, &self.modality_names)?;
        if self.return_dict {
            Ok(Some(dict.into_any()))
        } else {
            // Tuple of X arrays in modality order.
            let x_dict = dict.get_item("X")?.expect("X key always present");
            let x_dict = x_dict.cast::<PyDict>()?;
            let mut tuple_items: Vec<Bound<'_, PyAny>> =
                Vec::with_capacity(self.modality_names.len());
            for name in &self.modality_names {
                let v = x_dict
                    .get_item(name)?
                    .expect("modality entry always present");
                tuple_items.push(v);
            }
            Ok(Some(PyTuple::new(py, &tuple_items)?.into_any()))
        }
    }

    /// Total observations (cells), shared across modalities (global obs axis).
    #[getter]
    fn n_obs(&self) -> u64 {
        self.pipelines.first().map(|p| p.n_obs()).unwrap_or(0)
    }

    /// List of modality names, in the same order as the constructor's
    /// `modalities` argument.
    #[getter]
    fn modality_names(&self) -> Vec<String> {
        self.modality_names.clone()
    }

    /// Per-modality `n_vars` as a dict.
    #[getter]
    fn n_vars(&self) -> HashMap<String, u64> {
        self.modality_names
            .iter()
            .zip(self.pipelines.iter())
            .map(|(n, p)| (n.clone(), p.n_vars()))
            .collect()
    }

    /// Memory budget diagnostics as a dict.
    ///
    /// ```text
    /// batch_size          - pinned, uniform across modalities
    /// shard_group_size    - pinned, uniform across modalities
    /// max_memory_mb       - the budget as REQUESTED
    /// effective_total_mb  - the sum the per-modality floors actually budgeted
    /// modalities          - {name: <TrainingDataset-shaped envelope>}
    /// ```
    ///
    /// ORG-9.10-4 deferred this class's accessor to ORG-9.10-5 because its
    /// budget is split per modality and there was no surface to report a split
    /// through. This is that surface, and it says two things nothing else did:
    ///
    /// * **`batch_size` / `shard_group_size` are uniform by construction.** The
    ///   loader pins every modality to the cross-modality minimum so their
    ///   batches stay row-aligned; that used to be guarded by a `debug_assert`,
    ///   i.e. by nothing at all in the release wheels users run. It is now a
    ///   checked property of `TrainingPipeline::pin_effective_config`, and
    ///   readable from Python.
    /// * **`effective_total_mb` can exceed `max_memory_mb`.** Two causes, both
    ///   real: the per-modality floor (the split is nnz-proportional, and a
    ///   share below the floor fails `LoaderConfig::validate`, so two
    ///   modalities under a 64 MB request budget 128 MB between them), and the
    ///   adaptive resolution when no budget was passed at all. Only the first
    ///   warns — an adaptive raise is the policy working.
    ///
    /// Each per-modality value carries the same keys as
    /// `TrainingDataset.memory_budget()`, including the six-key `breakdown`
    /// every class that reports a budget shares.
    fn memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let modalities = PyDict::new(py);
        let mut effective_total_mb = 0usize;
        for (name, p) in self.modality_names.iter().zip(self.pipelines.iter()) {
            let mb = p.memory_budget_info();
            let per = PyDict::new(py);
            per.set_item("shard_group_size", mb.shard_group_size)?;
            per.set_item("prefetch_batches", mb.prefetch_batches)?;
            per.set_item("batch_size", mb.batch_size)?;
            per.set_item("max_memory_mb", p.max_memory_mb())?;
            per.set_item("estimated_mb", mb.estimated_bytes / (1024 * 1024))?;
            per.set_item("mmap_mb", mb.mmap_bytes / (1024 * 1024))?;
            per.set_item("budget_exceeded", mb.budget_exceeded)?;
            per.set_item("breakdown", mb.breakdown.to_pydict(py)?)?;
            modalities.set_item(name, per)?;
            effective_total_mb = effective_total_mb.saturating_add(p.max_memory_mb());
        }
        // Uniform by construction; reading the first is reading all of them.
        let first = self.pipelines.first().map(|p| p.memory_budget_info());
        dict.set_item("batch_size", first.map(|b| b.batch_size).unwrap_or(0))?;
        dict.set_item(
            "shard_group_size",
            first.map(|b| b.shard_group_size).unwrap_or(0),
        )?;
        dict.set_item("max_memory_mb", self.requested_max_memory_mb)?;
        dict.set_item("effective_total_mb", effective_total_mb)?;
        dict.set_item("modalities", modalities)?;
        Ok(dict)
    }

    /// Shut every modality's pipeline down, GIL detached. Idempotent, and — as
    /// on [`TrainingDataset`] — **not** terminal: the next `__iter__` rebuilds.
    fn close(&mut self, py: Python<'_>) {
        py.detach(|| {
            for p in self.pipelines.iter_mut() {
                p.shutdown();
            }
        });
        self.closed = true;
    }

    /// True from [`Self::close`] until the next `__iter__` rebuilds. Never raises.
    ///
    /// Deliberately **not** the terminal flag `IndexPlanDataset.closed` is.
    /// `close()` on this class releases the rayon pool and joins the epoch
    /// threads, and the next `__iter__` builds them again — so the honest
    /// reading is "torn down right now", not "unusable from here on". It cannot
    /// be derived from pipeline state either: a freshly constructed dataset has
    /// no pool yet and would report `True` before it had ever been closed.
    #[getter]
    fn closed(&self) -> bool {
        self.closed
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.modality_names.iter().map(|s| s.as_str()).collect();
        format!(
            "MultimodalTrainingDataset(n_obs={}, modalities={names:?})",
            self.n_obs(),
        )
    }
}

impl Drop for MultimodalTrainingDataset {
    /// Release the GIL around the per-modality pipeline shutdowns on drop.
    /// See [`TrainingDataset`]'s `Drop` for the full rationale (pyclass drop
    /// holds the GIL; the bounded joins would otherwise freeze other Python
    /// threads; `py.detach` never calls into Python; `Py_IsInitialized` guards
    /// the post-finalization case). Mirrors this type's `close()`.
    fn drop(&mut self) {
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            Python::attach(|py| {
                py.detach(|| {
                    for p in self.pipelines.iter_mut() {
                        p.shutdown();
                    }
                })
            });
        } else {
            for p in self.pipelines.iter_mut() {
                p.shutdown();
            }
        }
    }
}

/// Phase H.2 helper: build a `{"X": {name: ndarray}, "obs": {...},
/// "cell_indices": ndarray}` dict from a slice of per-modality
/// `Batch`es. Uses the first batch's `obs` and `cell_indices` (cells
/// are global across modalities).
fn build_multimodal_batch_dict<'py>(
    py: Python<'py>,
    batches: &[Batch],
    modality_names: &[String],
) -> PyResult<Bound<'py, PyDict>> {
    use numpy::IntoPyArray;
    let dict = PyDict::new(py);

    let x_dict = PyDict::new(py);
    for (name, batch) in modality_names.iter().zip(batches.iter()) {
        let n_genes = if batch.x_shape.0 > 0 {
            batch.x_shape.1
        } else {
            0
        };
        let x_owned: Vec<f32> = batch.x.clone();
        let x_arr = x_owned.into_pyarray(py);
        let x_2d = x_arr
            .reshape([batch.x_shape.0, n_genes])
            .map_err(|e| PyRuntimeError::new_err(format!("X reshape failed: {e}")))?;
        x_dict.set_item(name, x_2d)?;
    }
    dict.set_item("X", x_dict)?;

    // obs and cell_indices come from the first modality's batch
    // (all batches share the global obs table). Skip if obs_columns
    // were not requested — empty dict.
    let obs_dict = PyDict::new(py);
    for (col_name, col) in &batches[0].obs {
        let arr_obj = match col {
            ObsColumn::Float64(v) => v.clone().into_pyarray(py).into_any(),
            ObsColumn::Int64(v) => v.clone().into_pyarray(py).into_any(),
            ObsColumn::Categorical(codes, cats) => {
                // Decode codes to category strings; emit as a Python
                // list (numpy lacks a native variable-width string
                // dtype). Consumer can wrap in pandas.Categorical.
                let decoded: Vec<&str> = codes
                    .iter()
                    .map(|&c| cats.get(c as usize).map(|s| s.as_str()).unwrap_or(""))
                    .collect();
                let list = pyo3::types::PyList::new(py, decoded)?;
                list.into_any()
            }
        };
        obs_dict.set_item(col_name, arr_obj)?;
    }
    dict.set_item("obs", obs_dict)?;
    dict.set_item(
        "cell_indices",
        batches[0].cell_indices.clone().into_pyarray(py),
    )?;
    Ok(dict)
}

/// ORG-9.10-5 pre-refactor pins for the multimodal budget split
/// (`split_budget_across_modalities`), which was inline in the constructor and
/// therefore asserted nowhere.
#[cfg(test)]
mod modality_budget_split_tests {
    use super::{split_budget_across_modalities, MIN_MODALITY_BUDGET_MB};

    #[test]
    fn shares_are_proportional_to_nnz() {
        // 1:3 nnz over a budget far above the floor.
        let split = split_budget_across_modalities(4000, &[1_000, 3_000]);
        assert_eq!(split, vec![1000, 3000]);
    }

    #[test]
    fn a_single_modality_receives_the_whole_budget() {
        assert_eq!(split_budget_across_modalities(4000, &[7]), vec![4000]);
    }

    /// The floor is what keeps a starved modality from failing
    /// `LoaderConfig::validate`, so it must survive any refactor of the split.
    #[test]
    fn a_starved_modality_is_raised_to_the_floor() {
        // 1:999 — the small modality's proportional share is 4 MB.
        let split = split_budget_across_modalities(4000, &[4, 3_996]);
        assert_eq!(split[0], MIN_MODALITY_BUDGET_MB);
        assert_eq!(split[1], 3996, "the large modality keeps its own share");
    }

    /// **The over-budget the floor buys.** Two modalities under a 64 MB request
    /// budget 128 MB between them; nothing reports this today. Pinned as
    /// observed fact so `memory_budget()` can surface it without the arithmetic
    /// shifting underneath.
    #[test]
    fn the_floor_lets_the_shares_exceed_the_request() {
        let total_mb = 64;
        let split = split_budget_across_modalities(total_mb, &[1_000, 1_000]);
        assert_eq!(split, vec![MIN_MODALITY_BUDGET_MB, MIN_MODALITY_BUDGET_MB]);
        assert_eq!(
            split.iter().sum::<usize>(),
            2 * MIN_MODALITY_BUDGET_MB,
            "two modalities at a {total_mb} MB request budget {} MB",
            2 * MIN_MODALITY_BUDGET_MB
        );
        assert!(
            split.iter().sum::<usize>() > total_mb,
            "premise of the whole pin: the floors exceed the request"
        );
    }

    /// The degenerate inputs the constructor can hand it: an all-zero nnz
    /// census (a file whose catalog carries no stats) must not divide by zero.
    /// The warning has to name a budget that actually stops it firing.
    ///
    /// Echoing the current effective sum back does **not** converge: the floor
    /// is re-applied after the new split, so with a 1:99 ratio 64 MB reports
    /// 128, 128 reports 190, 190 reports 252 … Found by review; this pins the
    /// fixed point instead.
    #[test]
    fn the_recommended_total_is_a_fixed_point() {
        let nnz = [1u64, 99];
        let mb = super::min_total_mb_clearing_the_floor(&nnz)
            .expect("a positive-nnz split always has a fixed point");
        let split = split_budget_across_modalities(mb, &nnz);
        assert!(
            split.iter().all(|&s| s >= MIN_MODALITY_BUDGET_MB),
            "every share must clear the floor: {split:?}"
        );
        assert_eq!(
            split.iter().sum::<usize>(),
            mb,
            "at the fixed point the shares sum to the request, so the warning stops"
        );
        // And the value the old message quoted does NOT converge.
        let naive: usize = split_budget_across_modalities(64, &nnz).iter().sum();
        assert!(
            split_budget_across_modalities(naive, &nnz)
                .iter()
                .sum::<usize>()
                > naive,
            "premise: echoing the effective sum back re-fires the warning"
        );
    }

    /// A modality with no recorded nnz gets a zero share at every budget, so no
    /// total lifts it off the floor — the message must say that rather than
    /// quoting a number that cannot work.
    #[test]
    fn a_zero_nnz_modality_has_no_fixed_point() {
        assert!(super::min_total_mb_clearing_the_floor(&[0, 100]).is_none());
        assert!(super::min_total_mb_clearing_the_floor(&[0, 0]).is_none());
        assert!(super::min_total_mb_clearing_the_floor(&[]).is_none());
    }

    #[test]
    fn zero_nnz_does_not_divide_by_zero() {
        let split = split_budget_across_modalities(4000, &[0, 0]);
        assert_eq!(split, vec![MIN_MODALITY_BUDGET_MB; 2]);
        assert!(split_budget_across_modalities(4000, &[]).is_empty());
    }
}
