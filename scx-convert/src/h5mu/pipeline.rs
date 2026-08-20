// h5mu (MuData) → SCX conversion (Phase D.1).
//
// MuData layout (scverse standard):
//   /obs                    — outer / shared global obs DataFrame
//   /obsm                   — outer / shared global obsm
//   /var                    — concatenated var (we ignore this; SCX
//                             keeps per-modality var)
//   /uns                    — outer uns
//   /mod/{name}/X           — per-modality count matrix
//   /mod/{name}/var         — per-modality var
//   /mod/{name}/obs         — per-modality obs (cell subset; ignored —
//                             SCX uses one shared global obs and
//                             expresses "no measurement" as empty
//                             rows in the modality's CSR shard)
//   /mod/{name}/obsm/{k}    — per-modality embeddings
//   /mod/{name}/layers/{l}  — per-modality layers
//   /mod/{name}/uns/...     — per-modality uns
//
// This pipeline currently assumes cell-aligned modalities — every
// modality has the same n_obs as the outer obs. CITE-seq and 10x
// Multiome (the primary CITE/multiome targets of Phase D) match that
// assumption. Misaligned modalities (e.g. some MAE inputs) will be
// rejected upstream (Phase I).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::error::ScxError;
use scx_format_io::header::FileHeader;
use scx_format_io::modality::ModalityType;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;

use crate::detect::{detect_matrix_format_at, MatrixFormat};
use crate::dtype::{detect_value_encoding_for_modality, index_dtype_for, values_to_raw_bytes};
use crate::h5ad::csc_stream::open_csc_streaming;
use crate::h5ad::dense_stream::{open_dense_streaming, read_dense_slab_f32, DenseDtype};
use crate::h5ad::read::{read_dataframe_group, read_layers_at, read_obsm_at, read_x_matrix_at};
use crate::h5ad::stream::{open_x_streaming, read_slice_f32};
use crate::pipeline::{ConvertError, CscPolicy, IngestOptions};
use crate::stream::CsrShardStream;
use crate::warnings::{ConvertWarning, WarningSink};

/// Detect whether an HDF5 file is an h5mu file (has `/mod` group).
pub fn is_h5mu_file(file: &hdf5::File) -> bool {
    file.group("mod").is_ok()
}

/// Phase 5a: emit a `PredicateIndexSkippedMultimodal` warning when the
/// caller passed predicate-index flags on multimodal input. The engine
/// read-side is unimodal-only today (`scx-format/src/reader.rs`
/// `read_obs_predicate_index_bytes` ignores `modality_id`), so writing
/// per-modality sections at conversion time would produce on-disk
/// artefacts that `QueryPipeline` cannot consume. Drop them with a
/// typed warning instead; users get a clear signal at convert time.
/// Convert-time grouping (`--group-by`) is single-modality
/// only. Reject it on h5mu input rather than silently emit an ungrouped file —
/// multimodal grouping (per-modality emission over one shared obs order).
fn reject_group_by_multimodal(opts: &IngestOptions) -> Result<(), ConvertError> {
    if opts.group_by.is_some() {
        return Err(ConvertError::Other(
            "convert --group-by is not supported on multimodal (h5mu) inputs (Phase 7.5); \
             convert a single modality or drop --group-by"
                .to_string(),
        ));
    }
    Ok(())
}

fn emit_multimodal_index_skip_warning(opts: &IngestOptions, sink: &mut WarningSink) {
    if opts.index_obs.is_empty() && opts.index_var.is_empty() && opts.index_preset.is_none() {
        return;
    }
    let mut columns: Vec<String> = Vec::new();
    columns.extend(opts.index_obs.iter().cloned());
    columns.extend(opts.index_var.iter().cloned());
    if let Some(name) = opts.index_preset.as_deref() {
        columns.push(format!("preset:{name}"));
    }
    sink.emit(ConvertWarning::PredicateIndexSkippedMultimodal { columns });
}

/// Resolve the modality type for `name`: prefer
/// `opts.modality_types` if the caller supplied an explicit
/// override; otherwise fall back to
/// [`infer_modality_type_from_name`] and emit a
/// [`crate::warnings::ConvertWarning::ModalityTypeInferred`] so the
/// inference is visible in provenance and the CLI summary.
pub(crate) fn resolve_modality_type(
    name: &str,
    opts: &IngestOptions,
    sink: &mut WarningSink,
) -> ModalityType {
    if let Some((_, t)) = opts.modality_types.iter().find(|(n, _)| n == name) {
        return *t;
    }
    let inferred = infer_modality_type_from_name(name);
    sink.emit(ConvertWarning::ModalityTypeInferred {
        name: name.to_string(),
        modality_type: inferred,
    });
    inferred
}

/// Heuristic to map a modality name to a `ModalityType`. Used when
/// the caller hasn't supplied an explicit `modality_types` override.
/// The names follow the conventions adopted by scverse / 10x for
/// CITE-seq and multiome files. Callers that fall through to this
/// helper (instead of consulting `opts.modality_types` first) should
/// emit a [`crate::ConvertWarning::ModalityTypeInferred`] so the
/// inference is visible in provenance.
pub(crate) fn infer_modality_type_from_name(name: &str) -> ModalityType {
    ModalityType::infer_from_name(name)
}

/// Per-modality header metadata from the shape-only pre-pass:
/// `(modality name, matrix format, n_vars)`.
type ModalityHeaderMeta = (String, MatrixFormat, u64);

/// Pre-pass: detect each modality's matrix format and read its column
/// count from the `/X` `/shape` attr (CSR/CSC) or dataset dims (dense),
/// reading no X data. Validates each modality's row count against the
/// shared outer `n_obs`. Returns per-modality `(name, format, n_vars)`
/// plus the widest `n_vars` seen (for the file-level `index_dtype`).
///
/// Shared by both `h5mu_to_scx` (non-streaming) and
/// `h5mu_to_scx_streaming`: neither needs to materialise X to populate
/// the file header — `n_vars` comes from the shape, and `nnz` is
/// reconstructed by `ScxWriter::finish()` from the per-shard catalog
/// stats, so it is left at `0` in the header by both callers.
fn read_modality_header_meta(
    file: &hdf5::File,
    modality_names: &[String],
    n_obs: usize,
    sink: &mut WarningSink,
) -> Result<(Vec<ModalityHeaderMeta>, u64), ConvertError> {
    let mut modality_meta: Vec<ModalityHeaderMeta> = Vec::with_capacity(modality_names.len());
    let mut max_n_vars: u64 = 0;
    for mname in modality_names {
        let x_path = format!("mod/{mname}/X");
        let fmt = detect_matrix_format_at(file, &x_path, sink)?;
        let mod_n_vars: u64 = match fmt {
            MatrixFormat::Csr | MatrixFormat::Csc => {
                let group = file.group(&x_path)?;
                let shape: Vec<i64> = group.attr("shape")?.read_1d()?.to_vec();
                if shape.len() != 2 {
                    return Err(ConvertError::Other(format!(
                        "expected 2D shape attr on '{x_path}', got {}-D",
                        shape.len()
                    )));
                }
                // The `/shape` attr is read as signed i64; guard against a
                // negative (corrupt) dimension silently wrapping to a huge
                // usize/u64 (review theme #1) instead of a bare `as` cast.
                let mod_n_obs = usize::try_from(shape[0]).map_err(|_| {
                    ConvertError::Other(format!(
                        "modality '{mname}' has negative n_obs dimension {} on '{x_path}'",
                        shape[0]
                    ))
                })?;
                if mod_n_obs != n_obs {
                    return Err(ConvertError::Other(format!(
                        "modality '{mname}' has n_obs={mod_n_obs} but outer obs has n_obs={n_obs}"
                    )));
                }
                u64::try_from(shape[1]).map_err(|_| {
                    ConvertError::Other(format!(
                        "modality '{mname}' has negative n_vars dimension {} on '{x_path}'",
                        shape[1]
                    ))
                })?
            }
            MatrixFormat::Dense => {
                let ds = file.dataset(&x_path)?;
                let shape = ds.shape();
                if shape.len() != 2 {
                    return Err(ConvertError::Other(format!(
                        "dense modality '{mname}' /X must be 2D, got {}-D",
                        shape.len()
                    )));
                }
                if shape[0] != n_obs {
                    return Err(ConvertError::Other(format!(
                        "modality '{mname}' has n_obs={} but outer obs has n_obs={n_obs}",
                        shape[0]
                    )));
                }
                shape[1] as u64
            }
        };
        if mod_n_vars > max_n_vars {
            max_n_vars = mod_n_vars;
        }
        modality_meta.push((mname.clone(), fmt, mod_n_vars));
    }
    Ok((modality_meta, max_n_vars))
}

/// Convert an h5mu file to a multimodal SCX v2 file.
pub fn h5mu_to_scx(
    input: &Path,
    output: &Path,
    opts: &IngestOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let file = hdf5::File::open(input)?;

    if !is_h5mu_file(&file) {
        return Err(ConvertError::FormatMismatch {
            expected: "h5mu (MuData with /mod group)".to_string(),
            got: "unknown HDF5 layout".to_string(),
        });
    }

    reject_group_by_multimodal(opts)?;
    emit_multimodal_index_skip_warning(opts, sink);

    // List modalities under /mod, in member-name order. h5mu stores
    // modalities under `/mod/{name}` — the names are case-sensitive
    // strings, treated as keys for round-tripping with MuData.
    let mod_group = file.group("mod")?;
    let modality_names = mod_group.member_names()?;
    if modality_names.is_empty() {
        return Err(ConvertError::Other(
            "h5mu file has /mod group but no modalities inside".to_string(),
        ));
    }

    // Outer obs is the shared global axis. Read it first to pin n_obs;
    // every modality's X must agree on n_obs.
    let outer_obs = read_dataframe_group(&file, "obs", sink)?;
    let n_obs = outer_obs.num_rows();
    if n_obs == 0 {
        return Err(ConvertError::Other(
            "h5mu outer /obs is empty — cannot determine global cell count".to_string(),
        ));
    }

    // Pre-pass: detect each modality's format and read its column count
    // from the `/X` shape attr / dataset dims — no X data is read here.
    // The full X is read exactly once, inside the per-modality write loop
    // below. The header `nnz` is reconstructed by `ScxWriter::finish()`
    // from the per-shard catalog stats, so we leave it at 0 (see the
    // shared `read_modality_header_meta` doc and the streaming path).
    let (modality_meta, max_n_vars) =
        read_modality_header_meta(&file, &modality_names, n_obs, sink)?;

    // index_dtype is shared across all CSR shards in the file (it's
    // a header-level setting). Pick the widest needed.
    let index_dtype: u8 = index_dtype_for(max_n_vars);

    // Build header. n_modalities + modality-table fields are set by
    // writer.finish() once add_modality has been called for each modality;
    // codec_id/value_encoding are per-modality at write time, so the header
    // value is a nominal default. `nnz` is left at 0 and reconstructed by
    // writer.finish() (sync_from_catalog) from the per-shard stats.
    let mut header = FileHeader::new_single_modality(
        n_obs as u64,
        max_n_vars,
        0,
        opts.shard_target_rows,
        0,
        index_dtype,
    );
    // Row-group framing (v4) is the default write layout. A framed shard is
    // only valid inside a v4 file, so bump the header alongside set_framing;
    // `row_group_rows=0` (framing() == None) keeps the legacy v3 layout.
    if opts.framing().is_some() {
        header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION;
    }

    let mut writer = ScxWriter::new(output, header)?;
    writer.set_framing(opts.framing());

    // Global obs goes first. Outer obsm and uns are also global.
    crate::pipeline::write_ingest_obs(&mut writer, &outer_obs, opts)?;

    // Outer obsm (global) → write as obsm/{key} with modality_id=0.
    // `read_obsm_at` returns Ok(empty) when absent and warns per-entry; a
    // propagated Err means the group is present but structurally unreadable
    // (member listing failed) — surface it rather than silently skipping.
    let global_obsm = read_obsm_at(&file, "obsm", sink)?;
    for (key, batch) in &global_obsm {
        writer.write_obsm(key, batch)?;
    }

    // Outer uns (global) → write as the global uns blob.
    if file.group("uns").is_ok() {
        let uns = crate::h5ad::read::read_uns(&file, opts.strict_uns, sink)?;
        writer.write_uns(&uns)?;
    }

    // Per-modality writes. Each iteration registers the modality,
    // writes its var, streams CSR shards, optionally emits CSC, then
    // writes per-modality obsm / layers / uns.
    for (mname, fmt, mod_n_vars) in &modality_meta {
        let x_path = format!("mod/{mname}/X");
        let var_path = format!("mod/{mname}/var");
        let obsm_path = format!("mod/{mname}/obsm");
        let layers_path = format!("mod/{mname}/layers");

        let (indptr, indices, data, _read_n_obs, _read_n_vars) =
            read_x_matrix_at(&file, &x_path, *fmt)?;
        let modality_type = resolve_modality_type(mname, opts, sink);
        let (value_encoding, codec_id) =
            detect_value_encoding_for_modality(&data, opts.codec, modality_type)
                .map_err(ScxError::from)?;

        // Per-modality CSC is emitted by the manual
        // `write_modality_csc_shards_from_csr` invocation below (gated on
        // the resolved `opts.csc` policy). The writer can alternatively
        // auto-emit per-modality CSC at finish() time once callers
        // migrate; auto-emit + manual would double-write, so only one is
        // wired at a time.
        let modality_id = writer
            .add_modality(mname, modality_type, codec_id, value_encoding, false)
            .map_err(ConvertError::from)?;
        writer
            .set_modality_n_vars(modality_id, *mod_n_vars)
            .map_err(ConvertError::from)?;

        let var = read_dataframe_group(&file, &var_path, sink)?;
        writer
            .write_var_for(modality_id, &var)
            .map_err(ConvertError::from)?;

        // Shard the per-modality X by row.
        write_modality_csr_shards(
            &mut writer,
            modality_id,
            &indptr,
            &indices,
            &data,
            n_obs,
            *mod_n_vars as usize,
            opts.shard_target_rows as usize,
            value_encoding,
            codec_id,
        )?;

        // Optional CSC sidecar — uses the same streaming-transpose
        // helper as the h5ad pipeline, but stamped on this modality.
        if opts.csc.should_build_csc(n_obs as u64, *mod_n_vars) {
            write_modality_csc_shards_from_csr(
                &mut writer,
                modality_id,
                &indptr,
                &indices,
                &data,
                n_obs,
                *mod_n_vars as usize,
                value_encoding,
                codec_id,
                opts.csc_cols_per_shard,
                opts.framing(),
            )?;
        }

        // Per-modality obsm. (Ok(empty) when absent / warned per-entry; a
        // propagated Err means a present-but-unreadable obsm group.)
        let obsm_map = read_obsm_at(&file, &obsm_path, sink)?;
        for (key, batch) in &obsm_map {
            writer
                .write_obsm_for(modality_id, key, batch)
                .map_err(ConvertError::from)?;
        }

        // Per-modality layers — written as layer-CSR shards stamped
        // with this modality_id. (Same absent-vs-unreadable contract as
        // obsm: Ok(empty)/warned per-entry, Err only on a corrupt group.)
        let layers = read_layers_at(&file, &layers_path, sink)?;
        for (layer_name, (l_indptr, l_indices, l_data, _l_nobs, l_nvars)) in &layers {
            let (l_enc, l_codec) =
                detect_value_encoding_for_modality(l_data, opts.codec, modality_type)
                    .map_err(ScxError::from)?;
            write_modality_layer_shards(
                &mut writer,
                modality_id,
                layer_name,
                l_indptr,
                l_indices,
                l_data,
                n_obs,
                *l_nvars,
                opts.shard_target_rows as usize,
                l_enc,
                l_codec,
            )?;
        }
    }

    // Provenance (global).
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: opts.tool.clone(),
        params_json: serde_json::json!({
            "input": input.display().to_string(),
            "format": "h5mu",
            "warnings": sink.summary_json(),
        })
        .to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

/// Streaming h5mu → SCX conversion. Phase 3 entry point.
///
/// Unlike [`h5mu_to_scx`], this path never materialises a full
/// modality's CSR in RAM. It composes the Phase 1/2 streaming
/// readers (CSR / dense / CSC) per modality and drives the shared
/// [`crate::pipeline::streaming_writer_coordinator`] inside a
/// [`ScxWriter::with_modality`] scope so per-shard catalog entries
/// are stamped with the right modality id.
///
/// Peak memory is bounded by
/// `shard_target_rows × max_n_vars × density × ~16 bytes` plus
/// per-modality non-streaming sections (var / obsm / uns) and the
/// always-resident outer obs.
pub fn h5mu_to_scx_streaming(
    input: &Path,
    output: &Path,
    opts: &IngestOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let file = hdf5::File::open(input)?;

    if !is_h5mu_file(&file) {
        return Err(ConvertError::FormatMismatch {
            expected: "h5mu (MuData with /mod group)".to_string(),
            got: "unknown HDF5 layout".to_string(),
        });
    }

    reject_group_by_multimodal(opts)?;
    emit_multimodal_index_skip_warning(opts, sink);

    let mod_group = file.group("mod")?;
    let modality_names_all = mod_group.member_names()?;
    if modality_names_all.is_empty() {
        return Err(ConvertError::Other(
            "h5mu file has /mod group but no modalities inside".to_string(),
        ));
    }

    // Apply opts.modalities filter (case-sensitive match against
    // /mod/{name}). Unknown names error with the full list of valid
    // modalities so the user can recover quickly.
    let modality_names: Vec<String> = match opts.modalities.as_ref() {
        None => modality_names_all.clone(),
        Some(wanted) => {
            let mut unknown: Vec<String> = Vec::new();
            for w in wanted {
                if !modality_names_all.iter().any(|n| n == w) {
                    unknown.push(w.clone());
                }
            }
            if !unknown.is_empty() {
                return Err(ConvertError::Other(format!(
                    "unknown modality name(s) {unknown:?}; available: {modality_names_all:?}"
                )));
            }
            wanted.clone()
        }
    };
    if modality_names.is_empty() {
        return Err(ConvertError::Other(
            "h5mu modality filter selected zero modalities".to_string(),
        ));
    }

    let outer_obs = read_dataframe_group(&file, "obs", sink)?;
    let n_obs = outer_obs.num_rows();
    if n_obs == 0 {
        return Err(ConvertError::Other(
            "h5mu outer /obs is empty — cannot determine global cell count".to_string(),
        ));
    }

    // Pre-pass: detect each modality's format and read the X /shape
    // attr (CSR/CSC) or dataset dims (dense). No data is read. Shared
    // with the non-streaming `h5mu_to_scx` path.
    let (modality_meta, max_n_vars) =
        read_modality_header_meta(&file, &modality_names, n_obs, sink)?;

    let index_dtype: u8 = index_dtype_for(max_n_vars);

    // CSC sidecar policy on the streaming multimodal path. The streaming
    // writer cannot build per-modality CSC: the non-streaming `h5mu_to_scx`
    // does so (it holds each modality's full CSR in memory), and
    // `rebuild_csc_inplace` is unimodal-only — running it over a multimodal
    // file would collapse every modality into one CSC transpose. So:
    //   - `Always` → reject; the user explicitly demanded CSC we can't honor.
    //   - `Auto`   → warn for each modality that would have cleared the size
    //                threshold, then proceed CSR-only (best-effort).
    //   - `Off`    → no-op.
    match opts.csc {
        CscPolicy::Always => {
            return Err(ConvertError::Other(
                "csc='always' is not supported on the streaming h5mu path \
                 (per-modality CSC cannot be built while streaming). Re-run with \
                 stream=False / --stream=false to build per-modality CSC sidecars."
                    .to_string(),
            ));
        }
        CscPolicy::Auto => {
            let would_build: Vec<String> = modality_meta
                .iter()
                .filter(|(_, _, mod_n_vars)| opts.csc.should_build_csc(n_obs as u64, *mod_n_vars))
                .map(|(name, _, _)| name.clone())
                .collect();
            if !would_build.is_empty() {
                sink.emit(ConvertWarning::CscSkippedStreamingMultimodal {
                    modalities: would_build,
                });
            }
        }
        CscPolicy::Off => {}
    }

    // Placeholder header. `nnz`, `n_csr_shards`, `n_modalities`, the
    // modality table offset, and codec_id are all overwritten by
    // `ScxWriter::finish()` from running accumulators.
    let mut header = FileHeader::new_single_modality(
        n_obs as u64,
        max_n_vars,
        0,
        opts.shard_target_rows,
        0,
        index_dtype,
    );
    // Row-group framing (v4) default — same as the non-streaming path. The
    // streaming coordinator already encodes X/layer shards with
    // `opts.framing()`; bumping the header + set_framing makes the file a
    // clean v4 (rather than v3-header/v2-shard) and frames any non-coordinator
    // shards. `row_group_rows=0` (framing() == None) keeps the v3 layout.
    if opts.framing().is_some() {
        header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION;
    }

    let mut writer = ScxWriter::new(output, header)?;
    writer.set_framing(opts.framing());

    // Outer obs / obsm / uns. Global obs goes first.
    crate::pipeline::write_ingest_obs(&mut writer, &outer_obs, opts)?;
    // Ok(empty) when absent / warned per-entry; Err only on a present-but-
    // unreadable obsm group — propagate rather than silently skip.
    let global_obsm = read_obsm_at(&file, "obsm", sink)?;
    for (key, batch) in &global_obsm {
        writer.write_obsm(key, batch)?;
    }
    if file.group("uns").is_ok() {
        let uns = crate::h5ad::read::read_uns(&file, opts.strict_uns, sink)?;
        writer.write_uns(&uns)?;
    }

    // Per-modality streaming writes. Each iteration:
    // 1. resolves modality_type (override or infer + warn);
    // 2. samples X values to pick a stable per-modality codec /
    //    value_encoding (one read of ≤ 64 KiB nnz, not the full data);
    // 3. registers the modality + writes var;
    // 4. wraps the streaming coordinator in with_modality so each
    //    emitted shard's catalog entry is stamped with this modality;
    // 5. writes per-modality layers (Csr/Dense/Csc dispatch), obsm,
    //    varm, obsp, varp, uns — small dense sections stay
    //    non-streaming as the spec calls out.
    let n_obs_u32: u32 = u32::try_from(n_obs)
        .map_err(|_| ConvertError::Other(format!("n_obs {n_obs} exceeds u32::MAX")))?;
    let _ = n_obs_u32; // currently unused; reserved for future per-shard validation.
    for (mname, fmt, mod_n_vars) in &modality_meta {
        let x_path = format!("mod/{mname}/X");
        let var_path = format!("mod/{mname}/var");
        let obsm_path = format!("mod/{mname}/obsm");
        let layers_path = format!("mod/{mname}/layers");

        let modality_type = resolve_modality_type(mname, opts, sink);

        let sample = sample_modality_values(&file, &x_path, *fmt, *mod_n_vars)?;
        let (value_encoding, codec_id) =
            detect_value_encoding_for_modality(&sample, opts.codec, modality_type)
                .map_err(ScxError::from)?;

        let modality_id = writer
            .add_modality(mname, modality_type, codec_id, value_encoding, false)
            .map_err(ConvertError::from)?;
        writer
            .set_modality_n_vars(modality_id, *mod_n_vars)
            .map_err(ConvertError::from)?;

        let var = read_dataframe_group(&file, &var_path, sink)?;
        writer
            .write_var_for(modality_id, &var)
            .map_err(ConvertError::from)?;

        let mod_n_vars_u32: u32 = u32::try_from(*mod_n_vars).map_err(|_| {
            ConvertError::Other(format!("modality '{mname}' n_vars exceeds u32::MAX"))
        })?;
        let mod_index_dtype: u8 = index_dtype_for(*mod_n_vars);

        // Open the per-modality X reader and validate its n_obs
        // matches the global axis. Phase 1/2 readers are reused
        // verbatim — the only new wrapper is `with_modality` around
        // the writer-coordinator call.
        let mut x_reader: Box<dyn CsrShardStream> = match fmt {
            MatrixFormat::Csr => Box::new(open_x_streaming(&file, &x_path, *fmt, sink)?),
            MatrixFormat::Dense => Box::new(open_dense_streaming(&file, &x_path, opts, sink)?),
            MatrixFormat::Csc => open_csc_streaming(&file, &x_path, opts, sink)?,
        };
        if x_reader.n_obs() as usize != n_obs {
            return Err(ConvertError::Other(format!(
                "modality '{mname}' streaming reader reports n_obs={} but outer obs has n_obs={n_obs}",
                x_reader.n_obs()
            )));
        }

        let modality_id_for_closure = modality_id;
        // Canonical per-modality CSR shard name is `X/{mname}/shard_{idx}`
        // (writer.rs::write_csr_shard_for and the explode/push name->path
        // mapper both expect this layout; the in-memory h5mu path below uses
        // it via write_csr_shard_for). The coordinator appends `_{idx}` to
        // this prefix. The old `{mname}_x_shard` prefix produced
        // `{mname}_x_shard_0`, which the reader tolerated (it resolves shards
        // by SectionType + modality_id, not name) but `scx explode`/`scx push`
        // rejected, since it matches neither the per-modality nor the legacy
        // single-modality name pattern.
        let section_prefix = format!("X/{mname}/shard");
        writer.with_modality::<_, _, ConvertError>(modality_id_for_closure, |w| {
            crate::pipeline::run_streaming_writer_coordinator(
                x_reader.as_mut(),
                w,
                opts,
                mod_index_dtype,
                mod_n_vars_u32,
                SectionType::CsrShard,
                modality_type,
                &section_prefix,
                sink,
                None,
            )?;
            Ok(())
        })?;
        drop(x_reader);

        // Per-modality layers. Mirror the h5ad layer dispatch: detect
        // per-layer format, open the appropriate streaming reader,
        // wrap in with_modality, drive coordinator. CSC layers go
        // through the Phase 2 dispatcher just like X.
        if let Ok(layers_group) = file.group(&layers_path) {
            let layer_names = layers_group.member_names()?;
            for layer_name in &layer_names {
                let layer_path = format!("{layers_path}/{layer_name}");
                let layer_fmt = match detect_matrix_format_at(&file, &layer_path, sink) {
                    Ok(f) => f,
                    Err(e) => {
                        sink.emit(ConvertWarning::LayerSkipped {
                            name: format!("{mname}/{layer_name}"),
                            reason: format!("{e}"),
                        });
                        continue;
                    }
                };
                let mut layer_reader: Box<dyn CsrShardStream> = match layer_fmt {
                    MatrixFormat::Csr => match open_layer_streaming_at(&file, &layer_path, sink) {
                        Ok(r) => Box::new(r),
                        Err(e) => {
                            sink.emit(ConvertWarning::LayerSkipped {
                                name: format!("{mname}/{layer_name}"),
                                reason: format!("{e}"),
                            });
                            continue;
                        }
                    },
                    MatrixFormat::Dense => {
                        match open_dense_streaming(&file, &layer_path, opts, sink) {
                            Ok(r) => Box::new(r),
                            Err(e) => {
                                sink.emit(ConvertWarning::LayerSkipped {
                                    name: format!("{mname}/{layer_name}"),
                                    reason: format!("{e}"),
                                });
                                continue;
                            }
                        }
                    }
                    MatrixFormat::Csc => match open_csc_streaming(&file, &layer_path, opts, sink) {
                        Ok(r) => r,
                        Err(e) => {
                            sink.emit(ConvertWarning::LayerSkipped {
                                name: format!("{mname}/{layer_name}"),
                                reason: format!("{e}"),
                            });
                            continue;
                        }
                    },
                };
                let layer_n_obs = layer_reader.n_obs() as usize;
                if layer_n_obs != n_obs {
                    sink.emit(ConvertWarning::LayerSkipped {
                        name: format!("{mname}/{layer_name}"),
                        reason: format!(
                            "n_obs {layer_n_obs} does not match outer obs n_obs {n_obs}"
                        ),
                    });
                    continue;
                }
                let layer_n_vars = layer_reader.n_vars();
                let layer_n_vars_u32 = match u32::try_from(layer_n_vars) {
                    Ok(v) => v,
                    Err(_) => {
                        sink.emit(ConvertWarning::LayerSkipped {
                            name: format!("{mname}/{layer_name}"),
                            reason: format!("n_vars {layer_n_vars} exceeds u32::MAX"),
                        });
                        continue;
                    }
                };
                let layer_index_dtype: u8 = index_dtype_for(layer_n_vars);
                // Must match `ScxWriter::write_layer_csr_shard_for`'s naming
                // (`layer/{mname}/{layer_name}/shard_{idx}`): the coordinator
                // appends `_{shard_idx}`, and both
                // `layer_csr_shards_for_modality` (needle `/{layer_name}/`) and
                // `layer_names_for` (prefix `layer/…/shard_`) parse that scheme.
                // A plain `{mname}_{layer_name}_shard` prefix made
                // streaming-written per-modality layers invisible to
                // `read_layer_for` / `layer_names_for`.
                let prefix = format!("layer/{mname}/{layer_name}/shard");
                writer.with_modality::<_, _, ConvertError>(modality_id_for_closure, |w| {
                    crate::pipeline::run_streaming_writer_coordinator(
                        layer_reader.as_mut(),
                        w,
                        opts,
                        layer_index_dtype,
                        layer_n_vars_u32,
                        SectionType::LayerCsrShard,
                        modality_type,
                        &prefix,
                        sink,
                        None,
                    )?;
                    Ok(())
                })?;
            }
        }

        // Per-modality obsm (small dense; non-streaming reuse of the
        // existing helper). Ok(empty) when absent / warned per-entry; Err
        // only on a present-but-unreadable obsm group — propagate it.
        let obsm_map = read_obsm_at(&file, &obsm_path, sink)?;
        for (key, batch) in &obsm_map {
            writer
                .write_obsm_for(modality_id, key, batch)
                .map_err(ConvertError::from)?;
        }
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let modality_names_for_json: Vec<&String> =
        modality_meta.iter().map(|(name, _, _)| name).collect();
    let resolved_reader_threads = crate::pipeline::resolve_reader_threads(opts.reader_threads);
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: opts.tool.clone(),
        params_json: serde_json::json!({
            "input": input.display().to_string(),
            "format": "h5mu",
            "stream": true,
            "modalities": modality_names_for_json,
            "warnings": sink.summary_json(),
            "reader_threads": resolved_reader_threads,
            "writer_queue_depth": opts.writer_queue_depth,
        })
        .to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

/// Layer-streaming helper using an absolute h5ad-style path
/// (`mod/{name}/layers/{layer}` or `layers/{layer}`). `open_layer_streaming`
/// only takes a bare layer name and assumes the `layers/` prefix —
/// the h5mu modality variant needs a full path so we go through
/// `open_x_streaming` directly.
fn open_layer_streaming_at(
    file: &hdf5::File,
    group_path: &str,
    sink: &mut WarningSink,
) -> Result<crate::h5ad::stream::XStreamReader, ConvertError> {
    open_x_streaming(file, group_path, MatrixFormat::Csr, sink)
}

/// Read up to 64 KiB worth of values from a modality's X to feed
/// `detect_value_encoding_for_modality`. The codec selector only
/// needs a representative sample — running it on the full data
/// would defeat the streaming pipeline's memory bound.
fn sample_modality_values(
    file: &hdf5::File,
    x_path: &str,
    fmt: MatrixFormat,
    n_vars: u64,
) -> Result<Vec<f32>, ConvertError> {
    const SAMPLE_VALUES: usize = 16 * 1024;
    match fmt {
        MatrixFormat::Csr | MatrixFormat::Csc => {
            let group = file.group(x_path)?;
            let data_ds = group.dataset("data")?;
            let n = data_ds.shape().first().copied().unwrap_or(0);
            let take = n.min(SAMPLE_VALUES);
            if take == 0 {
                return Ok(Vec::new());
            }
            read_slice_f32(&data_ds, 0, take)
        }
        MatrixFormat::Dense => {
            // Read the first slab worth of values (up to SAMPLE_VALUES
            // total elements). For a thin n_vars this samples many
            // rows; for wide matrices it samples one partial row.
            let ds = file.dataset(x_path)?;
            let shape = ds.shape();
            if shape.len() != 2 {
                return Err(ConvertError::Other(format!(
                    "dense /X at '{x_path}' must be 2D, got {}-D",
                    shape.len()
                )));
            }
            let n_obs_total = shape[0];
            if n_obs_total == 0 || n_vars == 0 {
                return Ok(Vec::new());
            }
            let rows = (SAMPLE_VALUES / (n_vars as usize).max(1))
                .max(1)
                .min(n_obs_total);
            let dtype = DenseDtype::from_descriptor(&ds.dtype()?.to_descriptor()?)?;
            read_dense_slab_f32(&ds, dtype, 0, rows)
        }
    }
}

/// Shard a modality's in-memory CSR (`indptr`/`indices`/`data`) by row and
/// emit each shard via `write_shard`, which receives the shard's 0-based index,
/// global `row_start`, and the rebased shard arrays (indptr `u64`, indices
/// `u32`, raw value bytes).
///
/// Per-shard validate + rebase + index-narrow go through the shared
/// `scx_sparse` helpers (`shard_nnz_bounds` + `rebase_csr_shard`), matching the
/// eager `pipeline.rs` and streaming `h5ad/stream.rs` ingest paths. This also
/// runs the `indices < n_vars` column-bound check (via `validate_csr_arrays`)
/// that the previously hand-rolled h5mu loops skipped.
#[allow(clippy::too_many_arguments)]
fn shard_modality_csr<F>(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    shard_target_rows: usize,
    value_encoding: ValueEncoding,
    mut write_shard: F,
) -> Result<(), ConvertError>
where
    F: FnMut(u32, u64, &[u64], &[u32], &[u8]) -> Result<(), ConvertError>,
{
    // Guard against a zero shard size: `row_end == row_start` would never
    // advance the cursor, hanging the `while row_start < n_obs` loop.
    if shard_target_rows == 0 {
        return Err(ConvertError::Other(
            "shard_target_rows must be greater than 0".to_string(),
        ));
    }
    let backing_len = indices.len().min(data.len());
    let mut row_start: usize = 0;
    let mut shard_idx: u32 = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        let shard_indptr_slice = &indptr[row_start..=row_end];
        let (nnz_start, nnz_end) = scx_sparse::shard_nnz_bounds(shard_indptr_slice, backing_len)
            .map_err(|e| {
                ConvertError::Other(format!("modality CSR shard validation failed: {e}"))
            })?;
        let (shard_indptr, shard_indices) = scx_sparse::rebase_csr_shard(
            shard_indptr_slice,
            &indices[nnz_start..nnz_end],
            n_vars as u64,
        )
        .map_err(|e| ConvertError::Other(format!("modality CSR shard validation failed: {e}")))?;
        let raw_values = values_to_raw_bytes(&data[nnz_start..nnz_end], value_encoding)
            .map_err(ScxError::from)?;

        write_shard(
            shard_idx,
            row_start as u64,
            &shard_indptr,
            &shard_indices,
            &raw_values,
        )?;

        row_start = row_end;
        shard_idx += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_modality_csr_shards(
    writer: &mut ScxWriter,
    modality_id: u8,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    shard_target_rows: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
) -> Result<(), ConvertError> {
    shard_modality_csr(
        indptr,
        indices,
        data,
        n_obs,
        n_vars,
        shard_target_rows,
        value_encoding,
        |_shard_idx, row_start, shard_indptr, shard_indices, raw_values| {
            writer
                .write_csr_shard_for(
                    modality_id,
                    shard_indptr,
                    shard_indices,
                    raw_values,
                    codec_id,
                    value_encoding,
                    row_start,
                )
                .map_err(ConvertError::from)
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn write_modality_csc_shards_from_csr(
    writer: &mut ScxWriter,
    modality_id: u8,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    csc_cols_per_shard: usize,
    framing: Option<scx_format_io::FramingConfig>,
) -> Result<(), ConvertError> {
    // Multimodal data arrives already canonical from upstream, so wrap it
    // as-is (new_unchecked) and route through the shared transpose-and-write
    // loop with this modality's id.
    let csr = scx_sparse::ScxCsr::new_unchecked(
        (n_obs, n_vars),
        indptr.to_vec(),
        indices.to_vec(),
        data.to_vec(),
    );
    scx_format_io::csc_sidecar::write_csc_sidecar(
        writer,
        std::slice::from_ref(&csr),
        n_obs,
        n_vars,
        value_encoding,
        codec_id,
        csc_cols_per_shard,
        scx_format_io::csc_sidecar::DEFAULT_CSC_MEMORY_BYTES,
        Some(modality_id),
        framing, // frame the CSC sidecar to match the v4 default (None = v3)
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_modality_layer_shards(
    writer: &mut ScxWriter,
    modality_id: u8,
    layer_name: &str,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    shard_target_rows: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
) -> Result<(), ConvertError> {
    shard_modality_csr(
        indptr,
        indices,
        data,
        n_obs,
        n_vars,
        shard_target_rows,
        value_encoding,
        |shard_idx, row_start, shard_indptr, shard_indices, raw_values| {
            writer
                .write_layer_csr_shard_for(
                    modality_id,
                    layer_name,
                    shard_idx,
                    shard_indptr,
                    shard_indices,
                    raw_values,
                    codec_id,
                    value_encoding,
                    row_start,
                )
                .map_err(ConvertError::from)
        },
    )
}
