// AnnData -> SCX rewrite: inline write path and the from_anndata_impl router.
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use arrow::array::RecordBatch;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

use rayon::prelude::*;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::section::SectionType;
use scx_format_io::{
    select_codec_for_modality, FileHeader, ModalityType, PreEncodedSection, ProvenanceEntry,
    ScxWriter,
};
use scx_sparse::canonicalize_csr;

use crate::to_pyerr;

use super::*;

/// Phase 5b: build and (conditionally) write a detection-bitmap shard
/// for the in-memory `from_anndata` write path. Mirrors
/// `scx_convert::pipeline::build_and_write_bitmap_for_shard` but emits
/// a Python `UserWarning` instead of `ConvertWarning::BitmapSkipped`.
///
/// Only the unimodal RNA case is exercised here (multimodal MuData
/// converts go through `scx-convert::h5mu_to_scx[_streaming]`); the
/// auto policy is therefore conservative — no ATAC eagerness branch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_and_write_bitmap_for_shard_python(
    py: Python<'_>,
    writer: &mut ScxWriter,
    indptr: &[u64],
    indices: &[u32],
    row_start: u64,
    n_rows: u32,
    n_vars: u32,
    encoded_csr_size: usize,
    policy: scx_format_io::BitmapPolicy,
) -> PyResult<()> {
    use scx_format_io::bitmap::BitmapShard;
    use scx_format_io::BitmapPolicy;
    const DENSITY_THRESHOLD: f32 = 0.30;
    const N_VARS_CAP: u32 = 1_000_000;
    const SIZE_PERCENT: usize = 15;

    let emit_warning = |msg: String| -> PyResult<()> {
        py.import("warnings")?.call_method1("warn", (msg,))?;
        Ok(())
    };

    if matches!(policy, BitmapPolicy::Off) {
        return Ok(());
    }
    if n_vars > N_VARS_CAP && !matches!(policy, BitmapPolicy::Always) {
        return emit_warning(format!(
            "bitmap skipped: n_vars {n_vars} exceeds auto cap {N_VARS_CAP}"
        ));
    }
    let nnz = *indptr.last().unwrap_or(&0);
    let cells = n_rows as u64;
    let density = if cells == 0 || n_vars == 0 {
        0.0_f32
    } else {
        nnz as f32 / (cells as f32 * n_vars as f32)
    };
    if matches!(policy, BitmapPolicy::Auto) && density > DENSITY_THRESHOLD {
        return emit_warning(format!(
            "bitmap skipped: density {density:.3} above auto threshold {DENSITY_THRESHOLD}"
        ));
    }
    let shard = BitmapShard::build_from_csr(row_start, n_rows, n_vars, indptr, indices);
    if matches!(policy, BitmapPolicy::Auto) {
        let est = shard.estimated_encoded_size();
        if encoded_csr_size > 0
            && est.saturating_mul(100) > encoded_csr_size.saturating_mul(SIZE_PERCENT)
        {
            return emit_warning(format!(
                "bitmap skipped: estimated {est} bytes > {SIZE_PERCENT}% of CSR shard ({encoded_csr_size})"
            ));
        }
    }
    writer.write_bitmap_shard(&shard).map_err(to_pyerr)?;
    Ok(())
}

/// Phase 5a: build and write obs / var predicate indexes from a Python
/// in-memory AnnData write path. Thin wrapper over
/// `scx_engine::build_and_write_conversion_predicate_indexes`; only the
/// outcome-to-Python mapping differs from the convert-side wrapper in
/// `scx_convert::pipeline::build_and_write_predicate_indexes`
/// (`PyValueError` for forced errors, `warnings.warn(...)` for preset
/// skips). Keep those two outcome maps in sync.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_and_write_predicate_indexes_inline(
    py: Python<'_>,
    writer: &mut ScxWriter,
    obs: &RecordBatch,
    var: &RecordBatch,
    csr_row_ranges: &[(u64, u64)],
    n_vars: usize,
    index_obs: &[String],
    index_var: &[String],
    index_preset: Option<&str>,
    index_auto_threshold: usize,
) -> PyResult<()> {
    use scx_engine::{
        build_and_write_conversion_predicate_indexes, BuildOutcome,
        ConversionPredicateIndexOptions, EngineError,
    };

    let engine_opts = ConversionPredicateIndexOptions {
        index_obs: index_obs.to_vec(),
        index_var: index_var.to_vec(),
        index_preset: index_preset.map(|s| s.to_string()),
        index_auto_threshold,
    };
    let result = build_and_write_conversion_predicate_indexes(
        writer,
        obs,
        var,
        csr_row_ranges,
        n_vars,
        &engine_opts,
    )
    .map_err(|e| match e {
        // Unknown preset is user-facing — surface as PyValueError so it
        // shows up as a clean `ValueError` in Python.
        EngineError::UnknownIndexPreset(_) => PyValueError::new_err(e.to_string()),
        other => PyRuntimeError::new_err(format!("build predicate index: {other}")),
    })?;

    let emit_warning = |msg: String| -> PyResult<()> {
        py.import("warnings")?.call_method1("warn", (msg,))?;
        Ok(())
    };
    // Drop pyarrow-internal `__*` columns
    // (notably `__index_level_0__`) before passing to the error
    // renderer — they live in the arrow schema for round-trip but
    // are never the right user-facing suggestion.
    let obs_available: Vec<String> = obs
        .schema()
        .fields()
        .iter()
        .map(|f| f.name())
        .filter(|n| !n.starts_with("__"))
        .cloned()
        .collect();
    let var_available: Vec<String> = var
        .schema()
        .fields()
        .iter()
        .map(|f| f.name())
        .filter(|n| !n.starts_with("__"))
        .cloned()
        .collect();
    let process = |outcomes: Vec<BuildOutcome>, axis: &str, available: &[String]| -> PyResult<()> {
        // Aggregate forced missing-column errors
        // into a single error so users see ALL typos in one shot,
        // matching the CLI's `process_predicate_index_outcomes` policy.
        // Non-missing forced errors (unsupported dtype, high
        // cardinality) stay fail-fast — the column exists, the message
        // is per-column.
        let mut forced_missing: Vec<String> = Vec::new();
        for outcome in outcomes {
            match outcome {
                BuildOutcome::ForcedColumnError { column, reason } => {
                    if matches!(reason, scx_engine::index::SkipReason::MissingColumn) {
                        forced_missing.push(column);
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "forced {axis} index column '{column}': {reason}"
                        )));
                    }
                }
                BuildOutcome::PresetSkipped { column, reason } => {
                    emit_warning(format!(
                        "predicate index skipped for {axis} column '{column}': {reason}"
                    ))?;
                }
            }
        }
        if !forced_missing.is_empty() {
            let msg =
                scx_engine::index::forced_columns_missing_message(axis, &forced_missing, available);
            return Err(PyValueError::new_err(msg));
        }
        Ok(())
    };
    process(result.obs_outcomes, "obs", &obs_available)?;
    process(result.var_outcomes, "var", &var_available)?;

    Ok(())
}

/// Owned, width-converted CSR buffers on their way to the CSC transpose.
///
/// Exists so the numpy-borrowed half of the work can be separated from the
/// long half: [`csc_input_from_csr_slices`] fills these under the GIL,
/// [`write_csc_shards_from_owned`] canonicalizes and writes them detached.
pub(crate) struct CscInput {
    indptr: Vec<u64>,
    indices: Vec<u32>,
    values: Vec<f32>,
}

/// Copy borrowed CSR arrays into the owned buffers the CSC transpose needs.
///
/// **Runs under the GIL, and does no more than it has to.** The three input
/// slices may be borrowed from live numpy buffers, so they are consumed here —
/// before any `py.detach` — and never escape. These are the same three
/// allocations the transpose always made; they are hoisted out of the detached
/// region, not added to it. The row-sorting pass that used to follow them stays
/// on the detached side, where it only touches owned memory.
///
/// All three fills run on rayon. This is the one copy in the crate that holds
/// the GIL for its whole duration, so its cost is a stall for every other
/// Python thread — and a fresh-allocation copy is page-fault bound, which
/// parallelizes (see [`crate::convert::par_to_vec`]). Rayon is safe under the
/// GIL here: nothing in these closures re-enters the interpreter.
pub(crate) fn csc_input_from_csr_slices(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
) -> Result<CscInput, scx_format_io::ScxError> {
    // A plain parallel map-collect over `Result` reads better than the
    // `try_fold`/`try_reduce` spelling. Two differences from the sequential
    // version, both harmless here: rayon may do more work before reducing to
    // `Err` (it does not abandon in-flight chunks the way `?` abandons the rest
    // of an iterator), and *which* negative entry is reported is no longer
    // first-in-index-order. Any negative entry is an equally valid diagnostic
    // and no test pins the message.
    let indptr_u64: Vec<u64> = indptr
        .par_iter()
        .map(|&v| {
            if v < 0 {
                Err(scx_format_io::ScxError::InvalidCatalog(format!(
                    "negative CSR indptr value {v} before CSC transpose"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let indices_u32: Vec<u32> = indices
        .par_iter()
        .map(|&v| {
            if v < 0 {
                Err(scx_format_io::ScxError::InvalidCatalog(format!(
                    "negative CSR index value {v} before CSC transpose"
                )))
            } else {
                Ok(v as u32)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CscInput {
        indptr: indptr_u64,
        indices: indices_u32,
        values: super::par_to_vec(data),
    })
}

/// Streaming CSR → CSC transpose over **owned** buffers, writing each emitted
/// chunk as one CSC shard.
///
/// Safe to call with the GIL released: every buffer it reads is Rust-owned.
/// There is deliberately no borrowed-slice entry point next to this one — the
/// pair `csc_input_from_csr_slices` + `write_csc_shards_from_owned` exists so
/// the copy cannot end up on the wrong side of a `py.detach`.
///
/// Together the pair mirrors `scx-cli::convert::write_csc_shards_from_csr`, so
/// the two import paths produce structurally identical CSC sidecars (same
/// `csc_cols_per_shard`, same encoder).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_csc_shards_from_owned(
    writer: &mut ScxWriter,
    input: CscInput,
    n_obs: usize,
    n_vars: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    csc_cols_per_shard: usize,
    // Capped at the sidecar builder's own default by
    // `scx_convert::csc_sidecar_bytes`. This was the fifth call site still
    // passing that default unconditionally; it sits outside `scx-convert/src`,
    // so the CI guard added for the other four did not see it.
    memory_budget: Option<u64>,
    framing: Option<scx_format_io::FramingConfig>,
) -> Result<(), scx_format_io::ScxError> {
    let CscInput {
        mut indptr,
        mut indices,
        mut values,
    } = input;
    canonicalize_csr(&mut indptr, &mut indices, &mut values);
    let csr = scx_sparse::ScxCsr::new_unchecked(
        (n_obs, n_vars),
        indptr.iter().map(|&v| v as i64).collect(),
        indices.iter().map(|&v| v as i32).collect(),
        values,
    );

    // Shared transpose-and-write loop (single-modality → modality_id None).
    scx_format_io::csc_sidecar::write_csc_sidecar(
        writer,
        std::slice::from_ref(&csr),
        n_obs,
        n_vars,
        value_encoding,
        codec_id,
        csc_cols_per_shard,
        scx_convert::csc_sidecar_bytes(memory_budget) as usize,
        None,
        framing,
    )
}

// ---------------------------------------------------------------------------
// to_anndata: SCX → AnnData
// ---------------------------------------------------------------------------

/// Shard boundary computed sequentially before parallel encoding.
pub(crate) struct ShardBoundary {
    row_start: usize,
    row_end: usize,
    nnz_start: usize,
    nnz_end: usize,
    indptr_base: i64,
    shard_idx: u32,
}

/// Parallel-encode CSR shards using rayon.
///
/// Clones the numpy-borrowed arrays into Rust-owned `Arc` slices for thread
/// safety, then encodes all shards in parallel under `py.detach()`.
/// Returns `PreEncodedSection`s in shard order, ready for sequential write.
#[allow(clippy::too_many_arguments)]
pub(crate) fn parallel_encode_csr_shards(
    py: Python<'_>,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    boundaries: &[ShardBoundary],
    csr_validated: bool,
    explicit_codec: Option<CodecId>,
    index_dtype: u8,
    n_vars: u32,
    section_type: SectionType,
    name_prefix: &str,
    framing: Option<scx_format_io::FramingConfig>,
) -> PyResult<Vec<PreEncodedSection>> {
    if boundaries.is_empty() {
        return Ok(Vec::new());
    }

    // Clone into Rust-owned Arc slices for Send + Sync across rayon threads.
    let indptr_owned: Arc<[i64]> = indptr.to_vec().into();
    let indices_owned: Arc<[i32]> = indices.to_vec().into();
    let data_owned: Arc<[f32]> = data.to_vec().into();
    let name_prefix = name_prefix.to_string();

    let result: Result<Vec<PreEncodedSection>, String> = py.detach(|| {
        boundaries
            .par_iter()
            .map(|b| {
                // 1. Rebase indptr for this shard
                let mut shard_indptr: Vec<u64> = if csr_validated {
                    indptr_owned[b.row_start..=b.row_end]
                        .iter()
                        .map(|&v| (v - b.indptr_base) as u64)
                        .collect()
                } else {
                    indptr_owned[b.row_start..=b.row_end]
                        .iter()
                        .map(|&v| {
                            if v < b.indptr_base {
                                Err(format!(
                                    "indptr value {v} < base {} (non-monotonic)",
                                    b.indptr_base
                                ))
                            } else {
                                Ok((v - b.indptr_base) as u64)
                            }
                        })
                        .collect::<Result<Vec<u64>, String>>()?
                };

                // 2. Convert indices i32 → u32
                let mut shard_indices: Vec<u32> = if csr_validated {
                    indices_owned[b.nnz_start..b.nnz_end]
                        .iter()
                        .map(|&v| v as u32)
                        .collect()
                } else {
                    indices_owned[b.nnz_start..b.nnz_end]
                        .iter()
                        .map(|&v| {
                            if v < 0 {
                                Err(format!("negative CSR index {v}"))
                            } else {
                                Ok(v as u32)
                            }
                        })
                        .collect::<Result<Vec<u32>, String>>()?
                };

                let data_borrow = &data_owned[b.nnz_start..b.nnz_end];
                let name = format!("{name_prefix}_shard_{}", b.shard_idx);

                // Skip the per-shard f32 copy when the source is already
                // canonical (the common case); only materialize + canonicalize
                // a genuinely non-canonical shard.
                let encode = |indptr: &[u64], indices: &[u32], data: &[f32]| {
                    scx_format_io::encode_one_shard(
                        indptr,
                        indices,
                        data,
                        explicit_codec,
                        index_dtype,
                        n_vars,
                        b.row_start as u64,
                        section_type,
                        ModalityType::Rna,
                        name.clone(),
                        framing,
                    )
                    .map_err(|e| e.to_string())
                };
                if scx_sparse::is_canonical_csr(&shard_indptr, &shard_indices, data_borrow) {
                    encode(&shard_indptr, &shard_indices, data_borrow)
                } else {
                    let mut shard_data = data_borrow.to_vec();
                    canonicalize_csr(&mut shard_indptr, &mut shard_indices, &mut shard_data);
                    encode(&shard_indptr, &shard_indices, &shard_data)
                }
            })
            .collect()
    });

    result.map_err(PyRuntimeError::new_err)
}

/// Decompose a scipy CSR matrix, canonicalize it, and invoke `f` with
/// slices suitable for `encode_one_shard`. `indptr`, `indices`, and
/// `data` are owned because v3 writers must sort rows, sum duplicate
/// coordinates, and drop explicit zeros before encoding.
pub(crate) fn decompose_scipy_csr_with<F, R>(
    py: Python<'_>,
    csr: &Bound<'_, PyAny>,
    f: F,
) -> PyResult<R>
where
    F: FnOnce(&[u64], &[u32], &[f32]) -> PyResult<R>,
{
    let np = py.import("numpy")?;

    let indptr_obj = csr.getattr("indptr")?;
    let indptr_arr = astype_if_needed(&indptr_obj, &np, "int64")?;
    let indptr: PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
    let indptr_slice = indptr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let mut indptr_u64: Vec<u64> = indptr_slice
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "negative indptr value {v} from wrapper.__getitem__"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<PyResult<Vec<u64>>>()?;

    let indices_obj = csr.getattr("indices")?;
    let indices_arr = astype_if_needed(&indices_obj, &np, "int32")?;
    let indices: PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
    let indices_slice = indices
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let mut indices_u32: Vec<u32> = indices_slice
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "negative column index {v} from wrapper.__getitem__"
                )))
            } else {
                Ok(v as u32)
            }
        })
        .collect::<PyResult<Vec<u32>>>()?;

    let data_obj = csr.getattr("data")?;
    let data_arr = astype_if_needed(&data_obj, &np, "float32")?;
    let data: PyReadonlyArray1<'_, f32> = data_arr.extract()?;
    let data_slice = data
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // `data` is copied like `indptr` / `indices` already are, rather than
    // handed through as a borrow on the canonical fast path.
    //
    // Every caller wraps `f` in `py.detach` (`convert/scx_to_scx.rs`, three
    // sites), so a borrowed numpy slice reaching `f` is finding §10.1 all over
    // again — the callers happen to pass per-shard `__getitem__` temps today,
    // which is a property of the callers, not of this `pub(crate)` signature.
    // The non-canonical branch always paid this copy; now the contract is
    // uniform, and `par_to_vec` keeps the added cost off the critical path.
    let mut data_vec = super::par_to_vec(data_slice);
    if !scx_sparse::is_canonical_csr(&indptr_u64, &indices_u32, &data_vec) {
        canonicalize_csr(&mut indptr_u64, &mut indices_u32, &mut data_vec);
    }
    f(&indptr_u64, &indices_u32, &data_vec)
}

/// Build a fresh `FileHeader` template for an SCX → SCX rewrite.
/// Catalog offsets, shard counts, and `nnz` are written by
/// `ScxWriter::finish()`.
///
/// `source_format_version` is the source SCX's `format_version`. Because
/// passthrough / per-shard re-encode does **not** re-canonicalize, the v3
/// canonical-CSR invariant may only be claimed when the source already
/// guarantees it (is itself v3+). `rewrite_output_format_version` gates the
/// stamp so a pre-v3 source is never silently upgraded to a false v3 claim.
pub(crate) fn build_output_header(
    n_obs: u64,
    n_vars: u64,
    shard_target_rows: u32,
    codec: CodecId,
    index_dtype: u8,
    source_format_version: u16,
) -> FileHeader {
    FileHeader {
        // Single-modality output → feature floor 1.
        format_version: scx_format_io::rewrite_output_format_version(&[source_format_version], 1),
        n_obs,
        n_vars,
        shard_target_rows,
        codec_id: codec as u8,
        index_dtype,
        manifest_sequence: 1,
        ..Default::default()
    }
}

/// Implementation of from_anndata: extract data from AnnData and write SCX.
///
/// `in_place`: when true, allow [`ensure_csr`] to sort caller-owned CSR
/// indices in place (mutates `adata.X`, `adata.layers[*]`, and
/// `adata.raw.X`). When false
/// (default), unsorted CSR inputs are copied via `.sorted_indices()` so
/// the caller's matrices are untouched.
#[allow(clippy::too_many_arguments)]
pub fn from_anndata_impl(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    in_place: bool,
    csc: &str,
    csc_cols_per_shard: usize,
    uns_format: &str,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    memory_budget: Option<u64>,
    force_legacy_metadata: bool,
    sort_by: Vec<String>,
    sort_reverse: bool,
    row_group_rows: Option<u32>,
    row_group_target_nnz: Option<u64>,
) -> PyResult<()> {
    // Resolve the codec intent axis (`auto`/`fast`/`compact` + explicit forces).
    // `compact`/`compact-trial`/explicit-`shufdelta` require row-group framing;
    // `auto` silently falls back to the heuristic single-encode when unframed.
    let resolved = scx_format_io::resolve_codec(codec).map_err(PyValueError::new_err)?;
    let explicit_codec = resolved.explicit_codec;
    let codec_trial = resolved.codec_trial;
    let decode_target = resolved.decode_target;
    if resolved.requires_framing && !matches!(row_group_rows, Some(g) if g > 0) {
        return Err(PyValueError::new_err(format!(
            "codec='{}' requires row_group_rows=N with N > 0 \
             (row-group-framed output for random-access-safe reads)",
            resolved.profile
        )));
    }
    // Framing is on by default (row_group_rows default = 256); `Some(0)` is the
    // explicit unframed (v3) opt-out — normalize to None so it threads through as
    // the legacy layout rather than a confusing v4-header-with-v1-shards no-op.
    let row_group_rows = row_group_rows.filter(|&g| g > 0);
    let framing = row_group_rows.map(|g| scx_format_io::FramingConfig {
        row_group_rows: g,
        target_nnz: row_group_target_nnz,
        trial: codec_trial,
        decode_target,
    });
    let shard_target_rows = crate::resolve_shard_size(shard_size, 16384)?;
    let csc_policy =
        scx_format_io::CscPolicy::parse(csc).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let uns_format_parsed = parse_uns_format(uns_format)?;

    // Backed AnnData → route through the streaming converter
    // (`scx_convert::h5ad_to_scx_streaming`) instead of the in-memory
    // path, which would fail at the `ensure_csr` step (backed `X` is
    // an `_CSRDataset`, not a scipy sparse matrix). In-memory
    // mutations on `obs` / `var` / `uns` / `obsm` / `varm` / `obsp` /
    // `varp` are extracted to Rust and passed as `StreamingOverrides`
    // so user edits aren't silently overwritten by the on-disk
    // version. Available only when pyscx was built with the `hdf5`
    // feature; without it the call falls through to the in-memory
    // path which raises a clear error on the backed `_CSRDataset`.
    let is_backed: bool = adata
        .getattr("isbacked")
        .ok()
        .and_then(|v| v.extract::<bool>().ok())
        .unwrap_or(false);
    if is_backed {
        #[cfg(feature = "hdf5")]
        {
            // Backed AnnData routes through the streaming converter, which
            // supports sort-on-convert (Phase 2). Other Phase-1 kwargs still
            // live on `pyscx.from_h5ad(path, ...)`.
            return route_backed_anndata_to_streaming(
                py,
                adata,
                path,
                explicit_codec,
                shard_target_rows,
                csc_policy,
                csc_cols_per_shard,
                uns_format_parsed,
                true,  // stream
                false, // strict_uns
                0.0,   // dense_zero_epsilon
                memory_budget,
                None, // temp_dir
                index_obs,
                index_var,
                index_preset,
                index_auto_threshold,
                bitmap,
                None, // reader_threads (auto)
                4,    // writer_queue_depth (default)
                sort_by,
                sort_reverse,
                row_group_rows,
                row_group_target_nnz,
                codec_trial,
                decode_target,
                force_legacy_metadata,
            );
        }
        #[cfg(not(feature = "hdf5"))]
        {
            let _ = (
                explicit_codec,
                csc_policy,
                csc_cols_per_shard,
                uns_format_parsed,
                &index_obs,
                &index_var,
                &index_preset,
                index_auto_threshold,
                bitmap,
                sort_reverse,
            );
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "pyscx was built without the `hdf5` feature; backed AnnData \
                 routing requires libhdf5. Rebuild with \
                 `maturin develop --features hdf5` or convert the AnnData \
                 to a non-backed form first.",
            ));
        }
    }

    // Sort-on-convert (Phase 2) currently rides the streaming converter,
    // which only the backed-AnnData path uses. For in-memory / SCX-backed
    // inputs, fail clearly rather than silently ignore the request.
    if !sort_by.is_empty() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "from_anndata(sort_by=...) is currently supported only for backed AnnData \
             (read with backed='r'). For an in-memory AnnData, write it to h5ad and use \
             `pyscx.from_h5ad(..., sort_by=...)`, or `scx convert --sort-by`.",
        ));
    }

    // Extract X as CSR. By default we do not mutate caller-owned CSR
    // matrices; pass `in_place=true` to opt into the original in-place
    // sort behavior for speed/memory.
    let x = adata.getattr("X")?;

    // Phase 8b: SCX-backed or lazy `X` → stream from the source SCX
    // file without materialising X into a scipy CSR. Falls through
    // to the existing in-memory path for scipy / numpy input. The
    // `extract::<PyRef<…>>()` calls are no-ops on non-matching
    // types (fail-fast, no Python call overhead).
    if let Ok(backed) = x.extract::<PyRef<crate::backed::ScxBackedSparseDataset>>() {
        return route_scx_backed_to_scx(
            py,
            adata,
            &backed,
            path,
            explicit_codec,
            shard_target_rows,
            csc_policy,
            csc_cols_per_shard,
            uns_format_parsed,
            shard_size.is_some(),
        );
    }
    if let Ok(lazy) = x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>() {
        return route_scx_lazy_to_scx(
            py,
            adata,
            &lazy,
            path,
            explicit_codec,
            shard_target_rows,
            csc_policy,
            csc_cols_per_shard,
            uns_format_parsed,
        );
    }

    let (x_csr, csr_validated) = ensure_csr(py, &x, in_place)?;

    // Get shape
    let shape: (u64, u64) = x_csr.getattr("shape")?.extract()?;
    let n_obs = shape.0;
    let n_vars = shape.1;

    // Resolve the CSC policy now that the in-memory shape is known
    // (`Auto` compares against the size thresholds).
    let csc_build = csc_policy.should_build_csc(n_obs, n_vars);

    if n_vars > u32::MAX as u64 {
        return Err(PyRuntimeError::new_err(format!(
            "n_vars ({n_vars}) exceeds u32::MAX; SCX format requires n_vars <= {}",
            u32::MAX
        )));
    }

    // Extract CSR arrays — skip .astype() when dtypes already match (1C.1)
    let np = py.import("numpy")?;

    let indptr_obj = x_csr.getattr("indptr")?;
    let indptr_arr = astype_if_needed(&indptr_obj, &np, "int64")?;
    let indptr: PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
    let indptr_slice = indptr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let expected_indptr_len = (n_obs as usize) + 1;
    if indptr_slice.len() != expected_indptr_len {
        return Err(PyValueError::new_err(format!(
            "X indptr has length {}, expected n_obs + 1 = {} (X.shape = ({}, {}))",
            indptr_slice.len(),
            expected_indptr_len,
            n_obs,
            n_vars
        )));
    }

    let indices_obj = x_csr.getattr("indices")?;
    let indices_arr = astype_if_needed(&indices_obj, &np, "int32")?;
    let indices: PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
    let indices_slice = indices
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let data_obj = x_csr.getattr("data")?;
    // T3.7: warn on lossy float64 → float32 value downcast at the write
    // boundary so the precision loss recorded in the round-trip fidelity
    // table is also visible at runtime, not just in docs.
    if let Ok(name) = data_obj
        .getattr("dtype")
        .and_then(|d| d.getattr("name"))
        .and_then(|n| n.extract::<String>())
    {
        if name == "float64" || name == "float128" {
            // stacklevel=2 so `-W error` points at the user's
            // `write()` / `from_anndata()` call, not the PyO3 bridge frame.
            let warn_fn = py.import("warnings")?.getattr("warn")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("stacklevel", 2)?;
            warn_fn.call(
                (format!(
                    "X values are stored as float32 in SCX; the source matrix is \
                     {name}, so values are downcast and precision is reduced. \
                     This is expected — see the round-trip fidelity table in the docs."
                ),),
                Some(&kwargs),
            )?;
        }
    }
    let data_arr = astype_if_needed(&data_obj, &np, "float32")?;
    let data: PyReadonlyArray1<'_, f32> = data_arr.extract()?;
    let data_slice = data
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let nnz = data_slice.len() as u64;

    // 1C.2: Fast upfront validation when CSR bypass is active.
    // After this, the shard loop can skip per-element checks.
    if csr_validated {
        scx_sparse::validate_csr_arrays(indptr_slice, indices_slice, n_vars)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }

    // Determine index dtype.
    //
    // CSR shards encode column indices (bounded by `n_vars`); CSC
    // sidecars encode global row indices (bounded by `n_obs`). The
    // file header carries one shared `index_dtype` that drives the
    // u16/u32 encoding choice in `write_shard_inner`. When CSC is
    // requested, fall back to u32 if EITHER axis exceeds u16. This
    // costs CSR a few bytes per index when n_obs > 65535 but
    // unblocks CSC writes on large-cell datasets (`census_1m`+).
    let index_dtype: u8 = {
        let max_axis = if csc_build { n_obs.max(n_vars) } else { n_vars };
        if max_axis <= 65535 {
            0
        } else {
            1
        }
    };

    // Peek at first shard's data to set file header codec_id (informational only;
    // readers use the per-shard header). Per-shard encoding/codec selection
    // happens inside the shard loop below.
    let first_shard_nnz_end = if n_obs as usize > 0 {
        indptr_slice[(shard_target_rows as usize).min(n_obs as usize)] as usize
    } else {
        0
    };
    let first_shard_data = &data_slice[..first_shard_nnz_end];
    let first_encoding = detect_value_encoding(first_shard_data);
    let first_values = encode_values(first_shard_data, first_encoding)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let header_codec = match explicit_codec {
        Some(codec_id) => {
            if codec_id == CodecId::Scx1 && !first_encoding.is_integer() {
                CodecId::Zstd
            } else {
                codec_id
            }
        }
        None => select_codec_for_modality(&first_values, first_encoding, ModalityType::Rna),
    };

    // Build FileHeader
    let mut header = FileHeader::new_single_modality(
        n_obs,
        n_vars,
        nnz,
        shard_target_rows,
        header_codec as u8,
        index_dtype,
    );
    // Row-group framing produces v4/shard-v2 shards; stamp the file v4 so old
    // readers reject it and the v4 write-guard admits the framed (and, under the
    // §4.3 cost model, unframed-Scx1-with-sidecar) shards. `new_single_modality`
    // stamps the default (v3), so bump it here (mirrors the streaming pipeline).
    if framing.is_some() {
        header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION;
    }

    let mut writer = ScxWriter::new(path, header).map_err(to_pyerr)?;

    // Write obs — always write even for 0-cell datasets to preserve column schema (finding 9.7).
    let obs_df = adata.getattr("obs")?;
    let obs_batch = pandas_to_record_batch(py, &obs_df)?;

    // Write var — always write even for 0-gene datasets to preserve column schema (finding 9.7).
    let var_df = adata.getattr("var")?;
    let var_batch = pandas_to_record_batch(py, &var_df)?;

    // Phase 4c: shard obs/var when row counts exceed shard_target_rows so
    // `from_anndata` emits the same `ObsMetadataShard` / `VarMetadataShard`
    // layout that merge / append / streaming ingest produce at atlas scale.
    // `force_legacy_metadata=true` opts back into single-section writes.
    let step = shard_target_rows as usize;
    let obs_rows = obs_batch.num_rows();
    let var_rows = var_batch.num_rows();
    let shard_obs = !force_legacy_metadata && obs_rows > step;
    let shard_var = !force_legacy_metadata && var_rows > step;
    py.detach(|| -> std::result::Result<(), scx_format_io::ScxError> {
        // Obs goes through the shared boundary loop, so this producer cannot
        // drift from `scx convert` / `scx-mtx` / `compact --reshape-obs` /
        // `optimize --shard-obs`. (Var keeps its own loop below: it is the one
        // axis this entry point shards and the shared helper is obs-scoped,
        // matching `ObsShardPolicy`.)
        scx_format_io::write_obs_section(&mut writer, &obs_batch, shard_obs, shard_target_rows)?;
        if shard_var {
            let n_total = var_rows as u64;
            let mut shard_idx: u32 = 0;
            let mut row_start: usize = 0;
            while row_start < var_rows {
                let lo = row_start;
                let hi = (lo + step).min(var_rows);
                let shard = var_batch.slice(lo, hi - lo);
                writer.write_var_shard(shard_idx, lo as u64, (hi - lo) as u64, n_total, &shard)?;
                shard_idx += 1;
                row_start = hi;
            }
        } else {
            writer.write_var(&var_batch)?;
        }
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Write CSR shards (1D: parallel shard encoding)
    let n_obs_usize = n_obs as usize;
    let shard_rows = shard_target_rows as usize;

    // 1D.1: Compute shard boundaries sequentially
    let mut boundaries = Vec::new();
    {
        let mut row_start: usize = 0;
        let mut shard_idx: u32 = 0;
        while row_start < n_obs_usize {
            let row_end = (row_start + shard_rows).min(n_obs_usize);
            let base = indptr_slice[row_start];
            if !csr_validated && base < 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "negative indptr value {base} at row {row_start}"
                )));
            }
            boundaries.push(ShardBoundary {
                row_start,
                row_end,
                nnz_start: base as usize,
                nnz_end: indptr_slice[row_end] as usize,
                indptr_base: base,
                shard_idx,
            });
            row_start = row_end;
            shard_idx += 1;
        }
    }

    // 1D.2+1D.3: Parallel encode + sequential write
    let pre_encoded = parallel_encode_csr_shards(
        py,
        indptr_slice,
        indices_slice,
        data_slice,
        &boundaries,
        csr_validated,
        explicit_codec,
        index_dtype,
        n_vars as u32,
        SectionType::CsrShard,
        "X",
        framing,
    )?;
    // Phase 5b: parse bitmap policy once.
    let bitmap_policy = scx_format_io::BitmapPolicy::parse(bitmap)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    for (boundary, section) in boundaries.iter().zip(pre_encoded) {
        let encoded_csr_size = section.section_length as usize;
        writer.write_preencoded_shard(section).map_err(to_pyerr)?;
        if !matches!(bitmap_policy, scx_format_io::BitmapPolicy::Off) {
            // Build the bitmap from the same canonical local CSR
            // representation used by the encoded shard.
            let lo = boundary.row_start;
            let hi = boundary.row_end;
            let nnz_lo = boundary.nnz_start;
            let nnz_hi = boundary.nnz_end;
            // Rebase indptr to shard-local. Mirror `parallel_encode_csr_shards`'s
            // guard: on non-canonical input, `v < base` would make `(v - base)`
            // go negative and wrap to a huge u64, corrupting the bitmap. (The
            // upstream encode call already rejects such shards, so this is
            // defense-in-depth.)
            let mut local_indptr: Vec<u64> = if csr_validated {
                indptr_slice[lo..=hi]
                    .iter()
                    .map(|&v| (v - boundary.indptr_base) as u64)
                    .collect()
            } else {
                indptr_slice[lo..=hi]
                    .iter()
                    .map(|&v| {
                        if v < boundary.indptr_base {
                            Err(PyRuntimeError::new_err(format!(
                                "indptr value {v} < base {} (non-monotonic)",
                                boundary.indptr_base
                            )))
                        } else {
                            Ok((v - boundary.indptr_base) as u64)
                        }
                    })
                    .collect::<PyResult<Vec<u64>>>()?
            };
            // Mirror `parallel_encode_csr_shards`'s index guard: on non-canonical
            // input a negative `i32` index would wrap via `as u32` to a huge
            // column, silently corrupting the bitmap. (Defense-in-depth — the
            // upstream encode call already rejects such shards.)
            let mut local_indices: Vec<u32> = if csr_validated {
                indices_slice[nnz_lo..nnz_hi]
                    .iter()
                    .map(|&v| v as u32)
                    .collect()
            } else {
                indices_slice[nnz_lo..nnz_hi]
                    .iter()
                    .map(|&v| {
                        if v < 0 {
                            Err(PyRuntimeError::new_err(format!("negative CSR index {v}")))
                        } else {
                            Ok(v as u32)
                        }
                    })
                    .collect::<PyResult<Vec<u32>>>()?
            };
            let mut local_data = data_slice[nnz_lo..nnz_hi].to_vec();
            canonicalize_csr(&mut local_indptr, &mut local_indices, &mut local_data);
            let n_rows = (hi - lo) as u32;
            build_and_write_bitmap_for_shard_python(
                py,
                &mut writer,
                &local_indptr,
                &local_indices,
                lo as u64,
                n_rows,
                n_vars as u32,
                encoded_csr_size,
                bitmap_policy,
            )?;
        }
    }

    // Phase 4a/4b: stream obsm/varm/obsp/varp one key at a time.
    // Each iteration extracts a single key's value under the GIL,
    // builds one RecordBatch (numpy fast-path for plain numeric ndarrays
    // skips the `pd.DataFrame(arr)` roundtrip), optionally warns when
    // the estimated peak footprint exceeds `memory_budget`, then writes
    // shards in `py.detach` and drops the batch before moving on.
    // This bounds peak RSS to one key's payload at a time instead of
    // the full sum of all mappings.
    let obsm = adata.getattr("obsm")?;
    let obsm_keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (obsm.call_method0("keys")?,))?
        .extract()?;
    for key in &obsm_keys {
        let arr = obsm.call_method1("__getitem__", (key,))?;
        let batch = numpy_or_pandas_to_record_batch(py, &arr)?;
        let est = estimate_dense_bytes(&batch);
        if let Some(budget) = memory_budget {
            if est > budget {
                warn_python_convert(
                    py,
                    &scx_convert::ConvertWarning::MappingPeakFootprintHigh {
                        key: key.clone(),
                        axis: "obsm",
                        estimated_bytes: est,
                        budget_bytes: budget,
                    },
                )?;
            }
        }
        py.detach(|| -> Result<(), scx_format_io::ScxError> {
            for_each_dense_shard(
                &batch,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsm_shard(key, idx, row_start, n_shard_rows, n_total, shard)
                },
            )
        })
        .map_err(to_pyerr)?;
    }

    // varm — same pattern. Duck-typed AnnData-likes may omit
    // `varm`/`obsp`/`varp` entirely; missing attrs are treated as empty.
    if let Ok(varm) = adata.getattr("varm") {
        let varm_keys: Vec<String> = py
            .import("builtins")?
            .call_method1("list", (varm.call_method0("keys")?,))?
            .extract()?;
        for key in &varm_keys {
            let arr = varm.call_method1("__getitem__", (key,))?;
            let batch = numpy_or_pandas_to_record_batch(py, &arr)?;
            let est = estimate_dense_bytes(&batch);
            if let Some(budget) = memory_budget {
                if est > budget {
                    warn_python_convert(
                        py,
                        &scx_convert::ConvertWarning::MappingPeakFootprintHigh {
                            key: key.clone(),
                            axis: "varm",
                            estimated_bytes: est,
                            budget_bytes: budget,
                        },
                    )?;
                }
            }
            py.detach(|| -> Result<(), scx_format_io::ScxError> {
                for_each_dense_shard(
                    &batch,
                    shard_target_rows,
                    |idx, row_start, n_shard_rows, n_total, shard| {
                        writer.write_varm_shard(key, idx, row_start, n_shard_rows, n_total, shard)
                    },
                )
            })
            .map_err(to_pyerr)?;
        }
    }

    // obsp — sparse COO, one key at a time.
    if let Ok(obsp) = adata.getattr("obsp") {
        let obsp_keys: Vec<String> = py
            .import("builtins")?
            .call_method1("list", (obsp.call_method0("keys")?,))?
            .extract()?;
        for key in &obsp_keys {
            let mat = obsp.call_method1("__getitem__", (key,))?;
            let batch = sparse_to_coo_record_batch(py, &mat)?;
            let est = estimate_coo_bytes(&batch);
            if let Some(budget) = memory_budget {
                if est > budget {
                    warn_python_convert(
                        py,
                        &scx_convert::ConvertWarning::MappingPeakFootprintHigh {
                            key: key.clone(),
                            axis: "obsp",
                            estimated_bytes: est,
                            budget_bytes: budget,
                        },
                    )?;
                }
            }
            py.detach(|| -> Result<(), scx_format_io::ScxError> {
                for_each_coo_shard(
                    &batch,
                    shard_target_rows,
                    |idx, row_start, n_shard_rows, n_total, shard| {
                        writer.write_obsp_shard_coo(
                            key,
                            idx,
                            row_start,
                            n_shard_rows,
                            n_total,
                            shard,
                        )
                    },
                )
            })
            .map_err(to_pyerr)?;
        }
    }

    // varp — sparse COO, one key at a time.
    if let Ok(varp) = adata.getattr("varp") {
        let varp_keys: Vec<String> = py
            .import("builtins")?
            .call_method1("list", (varp.call_method0("keys")?,))?
            .extract()?;
        for key in &varp_keys {
            let mat = varp.call_method1("__getitem__", (key,))?;
            let batch = sparse_to_coo_record_batch(py, &mat)?;
            let est = estimate_coo_bytes(&batch);
            if let Some(budget) = memory_budget {
                if est > budget {
                    warn_python_convert(
                        py,
                        &scx_convert::ConvertWarning::MappingPeakFootprintHigh {
                            key: key.clone(),
                            axis: "varp",
                            estimated_bytes: est,
                            budget_bytes: budget,
                        },
                    )?;
                }
            }
            py.detach(|| -> Result<(), scx_format_io::ScxError> {
                for_each_coo_shard(
                    &batch,
                    shard_target_rows,
                    |idx, row_start, n_shard_rows, n_total, shard| {
                        writer.write_varp_shard_coo(
                            key,
                            idx,
                            row_start,
                            n_shard_rows,
                            n_total,
                            shard,
                        )
                    },
                )
            })
            .map_err(to_pyerr)?;
        }
    }

    // 1E.2: Collect uns JSON under GIL.
    // Use a recursive Python-side normalizer so common AnnData payloads
    // (NumPy arrays/scalars, pandas Index/Series/Categorical) survive the
    // JSON boundary instead of erroring out of `json.dumps`. Under
    // `UnsFormat::Tagged` (default), payloads are wrapped in `__scx_type__`
    // envelopes so dtype/shape/NaN/Inf round-trip losslessly.
    let uns = adata.getattr("uns")?;
    // Drop the transient F10 provenance hint (`scx_source_has_csc_sidecar`,
    // stamped by `Experiment.to_anndata` on a materialized CSC file) so it never
    // persists into the on-disk file — it describes the in-memory materialization,
    // not the data. Shallow-copy so the caller's `adata.uns` is left untouched.
    let uns = if uns.contains("scx_source_has_csc_sidecar").unwrap_or(false) {
        let copy = py.import("builtins")?.call_method1("dict", (&uns,))?;
        copy.del_item("scx_source_has_csc_sidecar")?;
        copy
    } else {
        uns
    };
    let uns_len: usize = uns.call_method0("__len__")?.extract()?;
    let uns_json: Option<serde_json::Value> = if uns_len > 0 {
        let np_generic = np.getattr("generic")?;
        let np_ndarray = np.getattr("ndarray")?;
        let mut ctx = UnsWriteCtx::new(uns_format_parsed, &np_generic, &np_ndarray);
        Some(normalize_uns_value(&uns, "uns", &mut ctx)?)
    } else {
        None
    };

    // Write uns outside GIL.
    if let Some(ref json_val) = uns_json {
        py.detach(|| writer.write_uns(json_val)).map_err(to_pyerr)?;
    }

    // Write layers
    let layers = adata.getattr("layers")?;
    let layer_keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (layers.call_method0("keys")?,))?
        .extract()?;
    for layer_name in &layer_keys {
        let layer_x = layers.call_method1("__getitem__", (layer_name,))?;
        let (layer_csr, l_csr_validated) = ensure_csr(py, &layer_x, in_place)?;

        let l_shape: (u64, u64) = layer_csr.getattr("shape")?.extract()?;
        if l_shape != (n_obs, n_vars) {
            return Err(PyValueError::new_err(format!(
                "Layer '{layer_name}' has shape ({}, {}), expected ({}, {})",
                l_shape.0, l_shape.1, n_obs, n_vars
            )));
        }

        let l_indptr_obj = layer_csr.getattr("indptr")?;
        let l_indptr_arr = astype_if_needed(&l_indptr_obj, &np, "int64")?;
        let l_indptr: PyReadonlyArray1<'_, i64> = l_indptr_arr.extract()?;
        let l_indptr_slice = l_indptr
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if l_indptr_slice.len() != expected_indptr_len {
            return Err(PyValueError::new_err(format!(
                "Layer '{layer_name}' indptr has length {}, expected n_obs + 1 = {}",
                l_indptr_slice.len(),
                expected_indptr_len
            )));
        }

        let l_indices_obj = layer_csr.getattr("indices")?;
        let l_indices_arr = astype_if_needed(&l_indices_obj, &np, "int32")?;
        let l_indices: PyReadonlyArray1<'_, i32> = l_indices_arr.extract()?;
        let l_indices_slice = l_indices
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let l_data_obj = layer_csr.getattr("data")?;
        let l_data_arr = astype_if_needed(&l_data_obj, &np, "float32")?;
        let l_data: PyReadonlyArray1<'_, f32> = l_data_arr.extract()?;
        let l_data_slice = l_data
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // 1C.2: Upfront validation for layer bypass
        if l_csr_validated {
            scx_sparse::validate_csr_arrays(l_indptr_slice, l_indices_slice, n_vars)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        // 1D: Parallel shard encoding for layers
        let mut l_boundaries = Vec::new();
        {
            let mut l_row_start: usize = 0;
            let mut shard_idx: u32 = 0;
            while l_row_start < n_obs_usize {
                let l_row_end = (l_row_start + shard_rows).min(n_obs_usize);
                let l_base = l_indptr_slice[l_row_start];
                if !l_csr_validated && l_base < 0 {
                    return Err(PyRuntimeError::new_err(format!(
                        "layer '{layer_name}': negative indptr value {l_base} at row {l_row_start}"
                    )));
                }
                l_boundaries.push(ShardBoundary {
                    row_start: l_row_start,
                    row_end: l_row_end,
                    nnz_start: l_base as usize,
                    nnz_end: l_indptr_slice[l_row_end] as usize,
                    indptr_base: l_base,
                    shard_idx,
                });
                l_row_start = l_row_end;
                shard_idx += 1;
            }
        }

        let l_pre_encoded = parallel_encode_csr_shards(
            py,
            l_indptr_slice,
            l_indices_slice,
            l_data_slice,
            &l_boundaries,
            l_csr_validated,
            explicit_codec,
            index_dtype,
            n_vars as u32,
            SectionType::LayerCsrShard,
            layer_name,
            framing,
        )?;
        for section in l_pre_encoded {
            writer.write_preencoded_shard(section).map_err(to_pyerr)?;
        }
    }

    // Optional `adata.raw` count matrix → the raw section family
    // (`RawCsrShard` + `raw/var`), mirroring what the h5ad ingest path
    // writes so both doors into SCX preserve raw identically.
    write_raw_from_anndata(
        py,
        adata,
        &mut writer,
        &np,
        n_obs,
        in_place,
        explicit_codec,
        shard_rows,
        framing,
    )?;

    // Optional CSC sidecar — streaming transpose over the in-memory
    // CSR view of X. Layers are CSR-only (no layer-CSC support yet —
    // a `LayerCscShard` section type would need to land first).
    //
    // The `*_slice` bindings are borrowed from `adata.X`'s numpy buffers
    // (`ensure_csr` returns the input unchanged when it is already sorted
    // CSR), so they are copied into owned buffers **under the GIL**; only
    // those cross `detach`. The transpose always made these three
    // allocations — they were simply made on the wrong side of the GIL
    // release, which is why this is RSS-neutral.
    if csc_build {
        let csc_input =
            csc_input_from_csr_slices(indptr_slice, indices_slice, data_slice).map_err(to_pyerr)?;
        py.detach(|| -> Result<(), scx_format_io::ScxError> {
            write_csc_shards_from_owned(
                &mut writer,
                csc_input,
                n_obs as usize,
                n_vars as usize,
                first_encoding,
                header_codec,
                csc_cols_per_shard,
                memory_budget,
                framing,
            )
        })
        .map_err(to_pyerr)?;
    }

    // Phase 5a: predicate indexes. Row ranges come from the boundaries
    // we already computed for the CSR shards — guaranteed to match
    // what's on disk because they drove the write itself.
    let csr_row_ranges: Vec<(u64, u64)> = boundaries
        .iter()
        .map(|b| (b.row_start as u64, b.row_end as u64))
        .collect();
    build_and_write_predicate_indexes_inline(
        py,
        &mut writer,
        &obs_batch,
        &var_batch,
        &csr_row_ranges,
        n_vars as usize,
        &index_obs,
        &index_var,
        index_preset.as_deref(),
        index_auto_threshold,
    )?;

    // Write provenance, stamping the codec-selection profile (intent) so a
    // downstream reader / gate can verify it — `scx info` prints non-empty
    // `params_json`. Plain `auto`/explicit writes keep the empty `{}`.
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            action: "from_anndata".to_string(),
            tool: format!("pyscx {}", env!("CARGO_PKG_VERSION")),
            params_json: codec_selection_params_json(resolved.profile),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;
    Ok(())
}

/// Write `adata.raw` (if present) as the raw section family:
/// `RawCsrShard` row shards named `raw/X_shard_<idx>` plus the `raw/var`
/// Arrow IPC section. The in-memory counterpart of
/// `scx_convert::pipeline`'s h5ad raw ingest, so `from_anndata` and
/// `from_h5ad` produce the same layout from the same dataset.
///
/// Raw shares X's obs axis but has its OWN, usually wider, column count.
/// Three things therefore key off `raw_n_vars` rather than X's `n_vars`:
/// the shard headers' minor-axis extent, the CSR index validation bound,
/// and `index_dtype` (per-shard, so a raw axis crossing 65535 widens raw's
/// indices without touching X's). Codec and value encoding are likewise
/// selected independently — raw holds counts while X may hold normalized
/// floats.
///
/// `ScxWriter::set_raw_n_vars` is deliberately NOT called: it feeds only
/// the eager `write_raw_csr_shard` path's header stamping, whereas
/// pre-encoded shards carry the extent passed to `encode_one_shard`.
/// `has_raw` needs no call either — `FileHeader::sync_from_catalog`
/// derives it at `finish()` from the presence of a `RawCsrShard` entry.
#[allow(clippy::too_many_arguments)]
fn write_raw_from_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    writer: &mut ScxWriter,
    np: &Bound<'_, pyo3::types::PyModule>,
    n_obs: u64,
    in_place: bool,
    explicit_codec: Option<CodecId>,
    shard_rows: usize,
    framing: Option<scx_format_io::FramingConfig>,
) -> PyResult<()> {
    // Duck-typed AnnData-likes may omit `raw` entirely; a missing attr is
    // treated the same as `raw = None`.
    let Some(raw) = adata.getattr("raw").ok().filter(|r| !r.is_none()) else {
        return Ok(());
    };

    let raw_x = raw.getattr("X")?;
    // Read the shape off `raw.X`, NEVER off `raw`: anndata's `Raw.shape`
    // reports the parent's `n_obs`, so a raw whose rows do not line up with
    // X looks correct from Python and would be written silently misaligned.
    let (raw_n_obs, raw_n_vars): (u64, u64) = raw_x.getattr("shape")?.extract()?;
    if raw_n_obs != n_obs {
        return Err(PyValueError::new_err(format!(
            "adata.raw.X has {raw_n_obs} rows but X has {n_obs}; adata.raw must share \
             the obs axis. (Note that adata.raw.shape reports X's row count, not \
             adata.raw.X's — compare adata.raw.X.shape[0].)"
        )));
    }
    if raw_n_vars > u32::MAX as u64 {
        return Err(PyRuntimeError::new_err(format!(
            "adata.raw n_vars ({raw_n_vars}) exceeds u32::MAX; SCX format requires \
             n_vars <= {}",
            u32::MAX
        )));
    }

    // SCX's raw section family stores `raw/X` and `raw/var` only.
    if let Ok(varm) = raw.getattr("varm") {
        if let Ok(keys) = varm.len() {
            if keys > 0 {
                warn_python_convert(py, &scx_convert::ConvertWarning::DroppedRawVarm { keys })?;
            }
        }
    }

    let (raw_csr, raw_validated) = ensure_csr(py, &raw_x, in_place)?;

    let r_indptr_obj = raw_csr.getattr("indptr")?;
    let r_indptr_arr = astype_if_needed(&r_indptr_obj, np, "int64")?;
    let r_indptr: PyReadonlyArray1<'_, i64> = r_indptr_arr.extract()?;
    let r_indptr_slice = r_indptr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let expected_indptr_len = (n_obs as usize) + 1;
    if r_indptr_slice.len() != expected_indptr_len {
        return Err(PyValueError::new_err(format!(
            "adata.raw.X indptr has length {}, expected n_obs + 1 = {}",
            r_indptr_slice.len(),
            expected_indptr_len
        )));
    }

    let r_indices_obj = raw_csr.getattr("indices")?;
    let r_indices_arr = astype_if_needed(&r_indices_obj, np, "int32")?;
    let r_indices: PyReadonlyArray1<'_, i32> = r_indices_arr.extract()?;
    let r_indices_slice = r_indices
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let r_data_obj = raw_csr.getattr("data")?;
    let r_data_arr = astype_if_needed(&r_data_obj, np, "float32")?;
    let r_data: PyReadonlyArray1<'_, f32> = r_data_arr.extract()?;
    let r_data_slice = r_data
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    if raw_validated {
        scx_sparse::validate_csr_arrays(r_indptr_slice, r_indices_slice, raw_n_vars)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }

    // Per-shard index width, resolved against raw's own column count.
    let raw_index_dtype: u8 = if raw_n_vars <= 65535 { 0 } else { 1 };

    let n_obs_usize = n_obs as usize;
    let mut boundaries = Vec::new();
    let mut row_start: usize = 0;
    let mut shard_idx: u32 = 0;
    while row_start < n_obs_usize {
        let row_end = (row_start + shard_rows).min(n_obs_usize);
        let base = r_indptr_slice[row_start];
        if !raw_validated && base < 0 {
            return Err(PyRuntimeError::new_err(format!(
                "adata.raw.X: negative indptr value {base} at row {row_start}"
            )));
        }
        boundaries.push(ShardBoundary {
            row_start,
            row_end,
            nnz_start: base as usize,
            nnz_end: r_indptr_slice[row_end] as usize,
            indptr_base: base,
            shard_idx,
        });
        row_start = row_end;
        shard_idx += 1;
    }

    let pre_encoded = parallel_encode_csr_shards(
        py,
        r_indptr_slice,
        r_indices_slice,
        r_data_slice,
        &boundaries,
        raw_validated,
        explicit_codec,
        raw_index_dtype,
        raw_n_vars as u32,
        SectionType::RawCsrShard,
        "raw/X",
        framing,
    )?;
    for section in pre_encoded {
        writer.write_preencoded_shard(section).map_err(to_pyerr)?;
    }

    let raw_var_df = raw.getattr("var")?;
    let raw_var_batch = pandas_to_record_batch(py, &raw_var_df)?;
    py.detach(|| writer.write_raw_var(&raw_var_batch))
        .map_err(to_pyerr)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Per-modality backed AnnData assembly.
//
// Builds a backed `AnnData` scoped to a single modality of a multimodal SCX
// file. X is wrapped in `ScxBackedSparseDataset` over a
// `BackedCsrReader::for_modality` (+ optional `BackedCscReader::for_modality`
// when the modality has a CSC sidecar). `obs` is the global obs DataFrame
// shared across modalities; `var` and `obsm` come from the per-modality
// reader helpers (`read_var_for`, `read_obsm_for`).
//
// Deletion vectors are NOT applied on this path. They are global (operate on
// the global obs axis) and would apply identically across modalities; lifting
// `compute_kept_to_global` / `filter_obs_by_deletion_vectors` into the
// per-modality helper is a follow-on. The existing `to_mudata` (eager) path
// also skips deletion vectors, so this matches today's eager behaviour.
// ---------------------------------------------------------------------------
