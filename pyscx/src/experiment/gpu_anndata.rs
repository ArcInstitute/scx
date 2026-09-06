//! The cfg(gpu) `to_gpu_anndata` device path — minimal-copy shard decode
//! straight onto the device — plus its decode-fallback diagnostics.

#[cfg(feature = "gpu")]
use super::PyExperiment;
#[cfg(feature = "gpu")]
use crate::convert;
#[cfg(feature = "gpu")]
use numpy::PyReadonlyArray1;
#[cfg(feature = "gpu")]
use pyo3::prelude::*;

/// Fault-injection hook for the host-assemble fallback below.
///
/// `SCX_FORCE_DEVICE_DECODE_FAILURE=1` makes the in-VRAM decode report a
/// module-load failure — the realistic trigger, since a build whose PTX did not
/// compile bakes empty stubs that fail exactly here. Without it the fallback
/// can only be exercised by breaking a build on purpose, which is not something
/// a test can do. Same role `SCX_DISABLE_RAPIDS=1` already plays for the
/// rapids-absent path.
///
/// Returns `None` in the normal case, leaving the real decode to run.
///
/// Read per call, **not** cached in a `OnceLock` like the workspace's other
/// `SCX_*` knobs. Those are cached because they sit on hot paths and are meant
/// to be process-stable; this one is read once per `to_gpu_anndata` (never in a
/// loop), and caching it would silently pin whichever value the first call in
/// the process happened to see — making the two tests that exercise the on and
/// off states order-dependent, and `monkeypatch.setenv` a no-op.
#[cfg(feature = "gpu")]
fn force_device_decode_failure(
) -> Option<Result<(scx_accel::GpuCsr, scx_accel::DeviceDecodeStats), scx_accel::GpuError>> {
    match std::env::var("SCX_FORCE_DEVICE_DECODE_FAILURE").as_deref() {
        Ok("1") => Some(Err(scx_accel::GpuError::ModuleLoadError(
            "forced by SCX_FORCE_DEVICE_DECODE_FAILURE=1".to_string(),
        ))),
        _ => None,
    }
}

/// Announce that `to_gpu_anndata` reached the device the slow way because the
/// fast way failed.
///
/// Always visible, and not one-shot-suppressed: a device decode that stopped
/// working is a real problem — a broken build, a driver mismatch — and the only
/// other trace of it is a `fallback_reason` nobody thinks to read when the call
/// appeared to succeed. Once per call is the right frequency for an op a user
/// invokes deliberately, not in a loop.
///
/// **Called only after the host route has actually produced the result.** The
/// message is in the past tense and asserts the result is correct; emitting it
/// at the point of failure would hand that reassurance to a caller who is about
/// to receive an exception instead. Takes the error's text rather than the error
/// itself, since by then the `GpuError` is long out of scope.
#[cfg(feature = "gpu")]
fn warn_device_decode_fallback(py: Python<'_>, e: &str) {
    let msg = format!(
        "to_gpu_anndata: the in-VRAM shard decode failed ({e}); assembled X on the host and \
         uploaded it instead. The result is correct but the fast path did not run — \
         uns[\"scx_accel\"][\"to_gpu_anndata\"] records transfer_mode=\"scx_device_handoff\" \
         with fallback_reason=\"gpu_runtime_error\"."
    );
    if let Ok(warnings) = crate::pyimport::import_module(py, "warnings") {
        let _ = warnings.call_method1(
            "warn",
            (msg, py.get_type::<pyo3::exceptions::PyUserWarning>()),
        );
    }
}

/// The gpu-feature body of `Experiment.to_gpu_anndata`, extracted so the
/// device path lives beside its decode-fallback diagnostics. `exp` is the
/// handle the #[pymethods] wrapper was called on; the wrapper has already
/// run the staleness check and rejected non-default materialization plans.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub(super) fn to_gpu_anndata_impl<'py>(
    exp: &PyExperiment,
    py: Python<'py>,
    var_names: Option<Vec<String>>,
    obs_filter: Option<&str>,
    layers: Option<Vec<String>>,
    obsm: Option<Vec<String>>,
    device: &str,
    memory_budget: Option<Bound<'_, PyAny>>,
    preserve_var_order: bool,
    strict_var_names: bool,
    plan: &scx_sparse::MaterializePlan,
    filters: crate::convert::SlotFilters<'_>,
) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};

    // cuPy is the hard requirement — the returned X is cupyx-sparse.
    let cupy_version = crate::accel::gpu::cupy_info(py).ok_or_else(|| {
        PyRuntimeError::new_err(
            "to_gpu_anndata requires cuPy (cupyx.scipy.sparse). Install the rapids \
                     analysis backend — see docs/gpu-setup.md.",
        )
    })?;
    // Resolve to a concrete GPU ordinal; reject device="cpu".
    let resolved = crate::accel::gpu::resolve_device(device)?;
    let gpu_id = resolved.gpu_id().ok_or_else(|| {
        PyValueError::new_err(
            "to_gpu_anndata requires a GPU device ('gpu', 'gpu:N', or 'auto' on a GPU host)",
        )
    })?;

    let dev = scx_accel::GpuDevice::new(gpu_id)
        .map_err(|e| PyRuntimeError::new_err(format!("GPU device {gpu_id}: {e}")))?;
    const HEADROOM: f64 = 1.2;
    let memory_budget_bytes = convert::parse_memory_budget(memory_budget.as_ref())?;

    // Fast path: a full-matrix handoff with no row/column reshaping decodes
    // X straight onto the device (no host scipy CSR, no re-upload — the
    // decode→host→re-upload trip Phase 0.2 measured as ~92% of census_1m
    // PCA wall). Any var_names / obs_filter / layer projection, a deletion
    // vector, or a multimodal source falls back to the host-assemble path
    // below — on-device filtered decode is Phase 4 format work.
    let n_csr_shards = exp.reader()?.csr_shard_count_for(0) as usize;
    let fast_path = var_names.is_none()
        && obs_filter.is_none()
        && layers.is_none()
        && !exp.reader()?.is_multimodal()
        && !exp.reader()?.header().has_deletion_vectors()
        && n_csr_shards > 0;

    // Set when the in-VRAM decode failed on the device and the
    // host-assemble path below is being asked to produce the result
    // instead — recorded as `fallback_reason` so a degraded handoff is
    // distinguishable from host-assemble chosen up front for a filtered
    // request. Holds the device error's text so the warning, which
    // fires only once the fallback has actually worked, can still name
    // what failed.
    let mut device_decode_error: Option<String> = None;

    // Yields the decoded CSR rather than a finished handoff: adopting
    // it consumes `dev`, and `dev` must survive for the host-assemble
    // arm below. Moving the adopt into the `match` puts the move in one
    // arm of two exclusive branches, which is what lets the borrow
    // checker see that only one of them takes the device.
    #[allow(clippy::type_complexity)]
    let fast: Option<(
        Bound<'py, PyAny>,
        usize,
        scx_accel::GpuCsr,
        scx_accel::DeviceDecodeStats,
    )> = if fast_path {
        'fast: {
            // X-less skeleton (obs / var / obsm / uns / layers assembled eagerly;
            // X is assigned after the device decode below).
            let adata = convert::to_anndata_filtered(
                py,
                &exp.path,
                exp.reader()?,
                None,
                None,
                None,
                obsm.as_deref(),
                filters,
                false, // preserve_slots
                true,  // eager
                memory_budget_bytes,
                true,  // skip_x
                false, // preserve_var_order (fast path: var_names is None)
                false, // strict_var_names (no names to check)
                plan,  // default plan (non-default rejected above); GPU X is f32-native
            )?;

            // Raw shard bytes (borrow the reader's mmap) + a cheap header
            // pre-scan for the VRAM gate and the honest HtoD byte count.
            let mut shard_refs: Vec<&[u8]> = Vec::with_capacity(n_csr_shards);
            let mut total_rows: usize = 0;
            let mut total_nnz: usize = 0;
            // Phase-2.x batched-nvcomp VRAM accounting: whether every shard is
            // framed ShufDeltaZstd-integer (batched path eligible) + the total
            // compressed idx/val bytes and the max index/value width — used to
            // size the all-shards-decompressed transient below.
            let mut all_framed_shufdelta_int = true;
            let mut total_compressed_bytes: u64 = 0;
            let mut max_index_width: u64 = 2;
            let mut max_value_width: u64 = 1;
            for i in 0..n_csr_shards {
                let bytes = exp
                    .reader()?
                    .read_raw_csr_shard_bytes_for(0, i)
                    .map_err(|e| PyRuntimeError::new_err(format!("read CSR shard {i}: {e}")))?;
                let header =
                    scx_format_io::shard::ShardHeader::read_from(&mut std::io::Cursor::new(bytes))
                        .map_err(|e| PyRuntimeError::new_err(format!("shard {i} header: {e}")))?;
                total_rows += header.n_major as usize;
                total_nnz += header.nnz as usize;
                let framed = header.shard_format_version
                    > scx_format_io::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION;
                let is_shufdelta = matches!(
                    scx_codec::CodecId::from_u8(header.codec_id),
                    Some(scx_codec::CodecId::ShufDeltaZstd)
                );
                let is_integer = scx_codec::ValueEncoding::from_u8(header.value_encoding)
                    .map(|v| v.is_integer())
                    .unwrap_or(false);
                all_framed_shufdelta_int &= framed && is_shufdelta && is_integer;
                total_compressed_bytes +=
                    header.indices_length as u64 + header.values_length as u64;
                let iw = if header.index_dtype == 0 { 2u64 } else { 4 };
                max_index_width = max_index_width.max(iw);
                if let Some(ve) = scx_codec::ValueEncoding::from_u8(header.value_encoding) {
                    max_value_width = max_value_width.max(ve.byte_width() as u64);
                }
                shard_refs.push(bytes);
            }

            let n_cols: usize = adata.getattr("n_vars")?.extract()?;
            if n_cols > i32::MAX as usize {
                return Err(PyValueError::new_err(format!(
                    "to_gpu_anndata: n_cols ({n_cols}) exceeds the i32 column-index range; \
                         the cupyx CSR handoff requires i32 indices."
                )));
            }

            // ≤VRAM pre-flight on the device-resident CSR size (HEADROOM covers
            // the transient single-shard decode buffer during concat).
            let csr_bytes = (total_nnz as u64) * 8 + (total_rows as u64 + 1) * 8;
            // The Phase-2.x batched nvcomp path (2x-e) holds three buffers at
            // peak: the compressed blob (all shards' idx+val frames), the
            // decompressed plane buffers (nnz × index_width + nnz × value_width),
            // and the final CSR — vs the per-shard path's CSR + one shard's
            // transient. Size the gate on that transient when the batched path
            // will actually run so a card that fits the CSR but not the transient
            // is rejected up front rather than OOMing mid-decode.
            let no_batch = std::env::var("SCX_NVCOMP_NO_BATCH")
                .map(|v| v == "1")
                .unwrap_or(false);
            let device_bytes =
                if all_framed_shufdelta_int && !no_batch && scx_accel::nvcomp_enabled() {
                    let plane_bytes =
                        (total_nnz as u64) * max_index_width + (total_nnz as u64) * max_value_width;
                    csr_bytes + total_compressed_bytes + plane_bytes
                } else {
                    // Per-shard fallback (mixed-codec / Scx1 / float, or the
                    // batched path forced off): the nvcomp/pipeline transient is
                    // bounded by a *single* shard's compressed + plane buffers
                    // (one shard is decoded then dropped before the next), which
                    // HEADROOM's 20% slack on the full CSR comfortably absorbs.
                    csr_bytes
                };
            let (free, total) = dev
                .free_memory()
                .map_err(|e| PyRuntimeError::new_err(format!("query free VRAM: {e}")))?;
            if (device_bytes as f64) * HEADROOM > free as f64 {
                return Err(PyValueError::new_err(format!(
                    "to_gpu_anndata needs ~{:.1} GB device memory for X ({} nnz) but only \
                         {:.1} GB of {:.1} GB is free on GPU {}. This is the >VRAM regime: use a \
                         backed/streaming workflow (open(...).to_anndata(backed=True) + \
                         pyscx.accel.*), not to_gpu_anndata.",
                    device_bytes as f64 / 1e9,
                    total_nnz,
                    free as f64 / 1e9,
                    total as f64 / 1e9,
                    gpu_id,
                )));
            }

            let decoded = force_device_decode_failure().unwrap_or_else(|| {
                scx_accel::decode_csr_shards_to_device_with_stats(&dev, &shard_refs)
            });
            let (gpu_csr, decode_stats) = match decoded {
                Ok(v) => v,
                // §8.10: the device could not assemble the CSR, but the
                // host-assemble path below reaches the same cupy `X` by a
                // different road. Take it rather than failing the call —
                // the realistic trigger is a module that will not load
                // (a stub-PTX build), which host-assemble does not touch.
                //
                // `alternate_route_may_succeed` is what excludes an
                // out-of-memory failure (both roads end with the same
                // CSR in the same VRAM) and an input defect (the host
                // decode rejects the same shard). See its doc.
                Err(e) if e.alternate_route_may_succeed() => {
                    // Deliberately does NOT warn here. The warning says
                    // the result was assembled on the host and is
                    // correct; the host route has not run yet, and if it
                    // fails the caller would get that reassurance
                    // followed by an exception. Recorded now, announced
                    // after it is true.
                    device_decode_error = Some(e.to_string());
                    // The partial device buffers are dropped by now, but
                    // cudarc frees into a memory pool that keeps them
                    // charged to the process until trimmed — and the
                    // host-assemble arm's first act is a free-VRAM
                    // pre-flight, which would otherwise pay for memory
                    // nothing holds. Best-effort: a failing trim must not
                    // replace the fallback with an error.
                    let _ = dev.reclaim_memory_pool();
                    break 'fast None;
                }
                // An out-of-memory failure is *not* worth retrying: both
                // arms end with the same CSR resident on the device, so
                // host-assemble cannot conjure the VRAM — it would pay a
                // full host materialization to fail again, and with a worse
                // message than this one.
                //
                // "With a worse message" is only true if this one carries
                // the remedy, so an OOM here gets the same `>VRAM`
                // guidance the up-front gate gives. It is appended only
                // for an OOM: the same arm also catches malformed shards
                // and unsupported layouts, where "use a backed workflow"
                // would be advice for a problem the caller does not have.
                Err(e) => {
                    let remedy = if matches!(e, scx_accel::GpuError::OutOfMemory(_)) {
                        " — this is the >VRAM regime: use a backed/streaming \
                                 workflow (open(...).to_anndata(backed=True) + \
                                 pyscx.accel.*), not to_gpu_anndata"
                    } else {
                        ""
                    };
                    return Err(PyRuntimeError::new_err(format!(
                        "GPU shard assembly failed: {e}{remedy}"
                    )));
                }
            };
            Some((adata, n_cols, gpu_csr, decode_stats))
        }
    } else {
        None
    };

    let (adata, holder, n_rows, n_cols, bytes_uploaded, transfer_mode, n_shufdelta_gpu) = match fast
    {
        Some((adata, n_cols, gpu_csr, decode_stats)) => {
            let n_rows = gpu_csr.shape().0;
            let holder = crate::accel::gpu_handoff::adopt_device_csr(dev, gpu_csr)?;
            // Honest transfer mode: a genuine fully-in-VRAM Scx1 decode (only the
            // tiny indptr uploaded — framed Scx1 shards decode group-by-group in
            // VRAM) vs a path where some shard bounced through the host because it
            // is a non-Scx1 codec. `bytes_uploaded` is the real HtoD total from
            // the decode, not a header estimate.
            let transfer_mode = if decode_stats.fully_device_decoded {
                "scx_device_decode_gpu"
            } else {
                "scx_device_handoff_streamed"
            };
            (
                adata,
                holder,
                n_rows,
                n_cols,
                decode_stats.host_uploaded_bytes,
                transfer_mode,
                Some(decode_stats.n_shards_shufdelta_gpu),
            )
        }
        None => {
            // Host-assemble fallback (filtered / projected / multimodal inputs):
            // the full option surface via the eager path, then a single HtoD.
            let adata = convert::to_anndata_filtered(
                py,
                &exp.path,
                exp.reader()?,
                var_names.as_deref(),
                obs_filter,
                layers.as_deref(),
                obsm.as_deref(),
                filters,
                false, // preserve_slots
                true,  // eager
                memory_budget_bytes,
                false, // skip_x
                preserve_var_order,
                strict_var_names,
                plan, // default plan (non-default rejected above); GPU X is f32-native
            )?;

            // Pull X's CSR arrays. scipy may store indptr/indices as int32 when
            // they fit, so coerce to the GpuCsr layout (f32 data / i32 indices /
            // i64 indptr) via astype(copy=False) — a no-op when already correct.
            let np = crate::pyimport::import_module(py, "numpy")?;
            let f32_ty = np.getattr("float32")?;
            let i32_ty = np.getattr("int32")?;
            let i64_ty = np.getattr("int64")?;
            let x = adata.getattr("X")?;
            let (n_rows, n_cols): (usize, usize) = x.getattr("shape")?.extract()?;
            if n_cols > i32::MAX as usize {
                return Err(PyValueError::new_err(format!(
                    "to_gpu_anndata: n_cols ({n_cols}) exceeds the i32 column-index range; \
                         the cupyx CSR handoff requires i32 indices."
                )));
            }
            let astype =
                |arr: Bound<'py, PyAny>, ty: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
                    let kw = pyo3::types::PyDict::new(py);
                    kw.set_item("copy", false)?;
                    arr.call_method("astype", (ty,), Some(&kw))
                };
            let data_arr = astype(x.getattr("data")?, &f32_ty)?;
            let indices_arr = astype(x.getattr("indices")?, &i32_ty)?;
            let indptr_arr = astype(x.getattr("indptr")?, &i64_ty)?;
            let data: Vec<f32> = data_arr
                .extract::<PyReadonlyArray1<f32>>()?
                .as_slice()?
                .to_vec();
            let indices: Vec<i32> = indices_arr
                .extract::<PyReadonlyArray1<i32>>()?
                .as_slice()?
                .to_vec();
            let indptr: Vec<i64> = indptr_arr
                .extract::<PyReadonlyArray1<i64>>()?
                .as_slice()?
                .to_vec();

            let bytes_uploaded =
                (data.len() as u64) * 4 + (indices.len() as u64) * 4 + (indptr.len() as u64) * 8;
            let (free, total) = dev
                .free_memory()
                .map_err(|e| PyRuntimeError::new_err(format!("query free VRAM: {e}")))?;
            if (bytes_uploaded as f64) * HEADROOM > free as f64 {
                return Err(PyValueError::new_err(format!(
                    "to_gpu_anndata needs ~{:.1} GB device memory for X ({} nnz) but only \
                         {:.1} GB of {:.1} GB is free on GPU {}. This is the >VRAM regime: use a \
                         backed/streaming workflow (open(...).to_anndata(backed=True) + \
                         pyscx.accel.*), not to_gpu_anndata.",
                    bytes_uploaded as f64 / 1e9,
                    indices.len(),
                    free as f64 / 1e9,
                    total as f64 / 1e9,
                    gpu_id,
                )));
            }

            let holder = crate::accel::gpu_handoff::upload_host_csr(
                dev, &indptr, &indices, &data, n_rows, n_cols,
            )?;
            (
                adata,
                holder,
                n_rows,
                n_cols,
                bytes_uploaded,
                "scx_device_handoff",
                None,
            )
        }
    };

    // Adopt the device buffers into a cupyx CSR and assign as X.
    let holder = Bound::new(py, holder)?;
    let cupy = crate::pyimport::import_module(py, "cupy")?;
    let cupyx_sparse = crate::pyimport::import_module(py, "cupyx.scipy.sparse")?;
    let adopt = |method: &str| -> PyResult<Bound<'py, PyAny>> {
        // cupy.asarray adopts the CAI view without copy and sets .base to
        // it, transitively keeping `holder` (and its device memory) alive.
        cupy.call_method1("asarray", (holder.call_method0(method)?,))
    };
    let data_cp = adopt("data")?;
    let indices_cp = adopt("indices")?;
    let indptr_cp = adopt("indptr")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("shape", (n_rows, n_cols))?;
    kwargs.set_item("copy", false)?;
    let gpu_x = cupyx_sparse.call_method(
        "csr_matrix",
        ((data_cp, indices_cp, indptr_cp),),
        Some(&kwargs),
    )?;
    adata.setattr("X", gpu_x)?;

    // Honest device-handoff metadata. `transfer_mode` alone cannot say
    // *why* a host-assembled handoff happened — the request needing a
    // filter and the device decode having failed produce the same
    // string — so the reason carries that distinction.
    let mut info = scx_accel::route::AccelExecutionInfo::new(
        scx_accel::route::AccelRoute::GpuCsr,
        if device_decode_error.is_some() {
            scx_accel::route::FallbackReason::GpuRuntimeError
        } else {
            scx_accel::route::FallbackReason::None
        },
    );
    info.transfer_mode = Some(transfer_mode);
    info.device_id = Some(gpu_id);
    info.bytes_uploaded = Some(bytes_uploaded);
    info.cupy_version = Some(cupy_version);
    info.n_shards_shufdelta_gpu = n_shufdelta_gpu;
    crate::accel::route::write_accel_route(py, &adata, "to_gpu_anndata", &info)?;

    // Announce the degraded path only now — every step it claims
    // succeeded (host assembly, its own VRAM gate, the upload, the cuPy
    // adoption, the stamp) is behind us, so the past tense is true and
    // the route it points the reader at exists to be read.
    if let Some(err) = device_decode_error {
        warn_device_decode_fallback(py, &err);
    }

    Ok(adata)
}
