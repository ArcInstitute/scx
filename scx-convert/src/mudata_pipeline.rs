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
use scx_format::error::ScxError;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::modality::ModalityType;
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;

use super::csc_stream::open_csc_streaming;
use super::dense_stream::{open_dense_streaming, read_dense_slab_f32, DenseDtype};
use super::detect::{detect_matrix_format_at, MatrixFormat};
use super::dtype::{detect_value_encoding_for_modality, values_to_raw_bytes};
use super::h5ad_read::{read_dataframe_group, read_layers_at, read_obsm_at, read_x_matrix_at};
use super::h5ad_stream::{open_x_streaming, read_slice_f32};
use super::pipeline::{ConvertError, ConvertOptions};
use super::stream::CsrShardStream;
use super::warnings::{ConvertWarning, WarningSink};

/// Detect whether an HDF5 file is an h5mu file (has `/mod` group).
pub fn is_h5mu_file(file: &hdf5::File) -> bool {
    file.group("mod").is_ok()
}

/// Resolve the modality type for `name`: prefer
/// `opts.modality_types` if the caller supplied an explicit
/// override; otherwise fall back to
/// [`infer_modality_type_from_name`] and emit a
/// [`super::warnings::ConvertWarning::ModalityTypeInferred`] so the
/// inference is visible in provenance and the CLI summary.
pub(crate) fn resolve_modality_type(
    name: &str,
    opts: &ConvertOptions,
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
    let lower = name.to_ascii_lowercase();
    if lower.contains("atac") || lower.contains("peak") || lower.contains("accessibility") {
        ModalityType::Atac
    } else if lower.contains("adt")
        || lower.contains("protein")
        || lower.contains("antibody")
        || lower.contains("prot")
    {
        ModalityType::Protein
    } else if lower.contains("spatial") {
        ModalityType::Spatial
    } else if lower.contains("methyl") {
        ModalityType::Methylation
    } else if lower == "rna" || lower == "gex" || lower.contains("expression") {
        ModalityType::Rna
    } else {
        ModalityType::Custom
    }
}

/// Convert an h5mu file to a multimodal SCX v2 file.
pub fn h5mu_to_scx(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let file = hdf5::File::open(input)?;

    if !is_h5mu_file(&file) {
        return Err(ConvertError::FormatMismatch {
            expected: "h5mu (MuData with /mod group)".to_string(),
            got: "unknown HDF5 layout".to_string(),
        });
    }

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
    let outer_obs = read_dataframe_group(&file, "obs")?;
    let n_obs = outer_obs.num_rows();
    if n_obs == 0 {
        return Err(ConvertError::Other(
            "h5mu outer /obs is empty — cannot determine global cell count".to_string(),
        ));
    }

    // Pre-read each modality's X to learn shapes + nnz so we can
    // populate the file header before opening the writer. We discard
    // the read data afterwards and re-read inside the per-modality
    // write loop — h5mu files are typically small enough that this
    // double-read is acceptable for an MVP. A streaming-only path
    // (Phase D follow-up) can compute n_vars / nnz from group attrs
    // alone without materialising X.
    let mut modality_meta: Vec<(String, MatrixFormat, usize, u64)> = Vec::new();
    let mut total_nnz: u64 = 0;
    let mut max_n_vars: u64 = 0;
    for mname in &modality_names {
        let x_path = format!("mod/{mname}/X");
        let fmt = detect_matrix_format_at(&file, &x_path, sink)?;
        let (indptr, _indices, _data, mod_n_obs, mod_n_vars) =
            read_x_matrix_at(&file, &x_path, fmt)?;
        if mod_n_obs != n_obs {
            return Err(ConvertError::Other(format!(
                "modality '{mname}' has n_obs={mod_n_obs} but outer obs has n_obs={n_obs} — \
                 Phase D currently requires cell-aligned modalities"
            )));
        }
        let nnz = *indptr.last().unwrap_or(&0) as u64;
        total_nnz += nnz;
        if (mod_n_vars as u64) > max_n_vars {
            max_n_vars = mod_n_vars as u64;
        }
        modality_meta.push((mname.clone(), fmt, mod_n_vars, nnz));
    }

    // index_dtype is shared across all CSR shards in the file (it's
    // a header-level setting). Pick the widest needed.
    let index_dtype: u8 = if max_n_vars <= 65535 { 0 } else { 1 };

    // Build header. n_modalities is set by writer.finish() once
    // add_modality has been called for each modality; we leave the
    // template at 0/0/0 here.
    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: n_obs as u64,
        n_vars: max_n_vars,
        nnz: total_nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: opts.shard_target_rows,
        // codec_id and value_encoding are per-modality at write time;
        // the header values are nominal defaults.
        codec_id: 0,
        index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    let mut writer = ScxWriter::new(output, header)?;

    // Global obs goes first. Outer obsm and uns are also global.
    writer.write_obs(&outer_obs)?;

    // Outer obsm (global) → write as obsm/{key} with modality_id=0.
    if let Ok(global_obsm) = read_obsm_at(&file, "obsm") {
        for (key, batch) in &global_obsm {
            writer.write_obsm(key, batch)?;
        }
    }

    // Outer uns (global) → write as the global uns blob.
    if file.group("uns").is_ok() {
        let uns = super::h5ad_read::read_uns(&file, opts.strict_uns, sink)?;
        writer.write_uns(&uns)?;
    }

    // Per-modality writes. Each iteration registers the modality,
    // writes its var, streams CSR shards, optionally emits CSC, then
    // writes per-modality obsm / layers / uns.
    for (mname, fmt, mod_n_vars, _expected_nnz) in &modality_meta {
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

        // build_csc=opts.csc lets the writer emit per-modality CSC
        // sidecars at finish() time so callers can drop the manual
        // `write_modality_csc_shards_from_csr` invocation below once
        // they migrate. The legacy manual path still works (and is
        // still wired below) — auto-emit + manual would double-write,
        // so flip one or the other.
        let modality_id = writer
            .add_modality(mname, modality_type, codec_id, value_encoding, false)
            .map_err(ConvertError::from)?;
        writer
            .set_modality_n_vars(modality_id, *mod_n_vars as u64)
            .map_err(ConvertError::from)?;

        let var = read_dataframe_group(&file, &var_path)?;
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
            *mod_n_vars,
            opts.shard_target_rows as usize,
            value_encoding,
            codec_id,
        )?;

        // Optional CSC sidecar — uses the same streaming-transpose
        // helper as the h5ad pipeline, but stamped on this modality.
        if opts.csc {
            write_modality_csc_shards_from_csr(
                &mut writer,
                modality_id,
                &indptr,
                &indices,
                &data,
                n_obs,
                *mod_n_vars,
                value_encoding,
                codec_id,
                opts.csc_cols_per_shard,
            )?;
        }

        // Per-modality obsm.
        if let Ok(obsm_map) = read_obsm_at(&file, &obsm_path) {
            for (key, batch) in &obsm_map {
                writer
                    .write_obsm_for(modality_id, key, batch)
                    .map_err(ConvertError::from)?;
            }
        }

        // Per-modality layers — written as layer-CSR shards stamped
        // with this modality_id.
        if let Ok(layers) = read_layers_at(&file, &layers_path) {
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
/// [`super::pipeline::streaming_writer_coordinator`] inside a
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
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let file = hdf5::File::open(input)?;

    if !is_h5mu_file(&file) {
        return Err(ConvertError::FormatMismatch {
            expected: "h5mu (MuData with /mod group)".to_string(),
            got: "unknown HDF5 layout".to_string(),
        });
    }

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

    let outer_obs = read_dataframe_group(&file, "obs")?;
    let n_obs = outer_obs.num_rows();
    if n_obs == 0 {
        return Err(ConvertError::Other(
            "h5mu outer /obs is empty — cannot determine global cell count".to_string(),
        ));
    }

    // Pre-pass: detect each modality's format and read the X /shape
    // attr (CSR/CSC) or dataset dims (dense). No data is read.
    let mut modality_meta: Vec<(String, MatrixFormat, u64)> =
        Vec::with_capacity(modality_names.len());
    let mut max_n_vars: u64 = 0;
    for mname in &modality_names {
        let x_path = format!("mod/{mname}/X");
        let fmt = detect_matrix_format_at(&file, &x_path, sink)?;
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
                let mod_n_obs = shape[0] as usize;
                if mod_n_obs != n_obs {
                    return Err(ConvertError::Other(format!(
                        "modality '{mname}' has n_obs={mod_n_obs} but outer obs has n_obs={n_obs}"
                    )));
                }
                shape[1] as u64
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

    let index_dtype: u8 = if max_n_vars <= 65535 { 0 } else { 1 };

    // Placeholder header. `nnz`, `n_csr_shards`, `n_modalities`, the
    // modality table offset, and codec_id are all overwritten by
    // `ScxWriter::finish()` from running accumulators.
    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: n_obs as u64,
        n_vars: max_n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: opts.shard_target_rows,
        codec_id: 0,
        index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    let mut writer = ScxWriter::new(output, header)?;

    // Outer obs / obsm / uns. Global obs goes first.
    writer.write_obs(&outer_obs)?;
    if let Ok(global_obsm) = read_obsm_at(&file, "obsm") {
        for (key, batch) in &global_obsm {
            writer.write_obsm(key, batch)?;
        }
    }
    if file.group("uns").is_ok() {
        let uns = super::h5ad_read::read_uns(&file, opts.strict_uns, sink)?;
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

        let var = read_dataframe_group(&file, &var_path)?;
        writer
            .write_var_for(modality_id, &var)
            .map_err(ConvertError::from)?;

        let mod_n_vars_u32: u32 = u32::try_from(*mod_n_vars).map_err(|_| {
            ConvertError::Other(format!("modality '{mname}' n_vars exceeds u32::MAX"))
        })?;
        let mod_index_dtype: u8 = if *mod_n_vars <= 65535 { 0 } else { 1 };

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
        let section_prefix = format!("{mname}_x_shard");
        writer.with_modality::<_, _, ConvertError>(modality_id_for_closure, |w| {
            super::pipeline::streaming_writer_coordinator(
                x_reader.as_mut(),
                w,
                opts,
                mod_index_dtype,
                mod_n_vars_u32,
                SectionType::CsrShard,
                modality_type,
                &section_prefix,
                sink,
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
                let layer_index_dtype: u8 = if layer_n_vars <= 65535 { 0 } else { 1 };
                let prefix = format!("{mname}_{layer_name}_shard");
                writer.with_modality::<_, _, ConvertError>(modality_id_for_closure, |w| {
                    super::pipeline::streaming_writer_coordinator(
                        layer_reader.as_mut(),
                        w,
                        opts,
                        layer_index_dtype,
                        layer_n_vars_u32,
                        SectionType::LayerCsrShard,
                        modality_type,
                        &prefix,
                        sink,
                    )?;
                    Ok(())
                })?;
            }
        }

        // Per-modality obsm (small dense; non-streaming reuse of the
        // existing helper).
        if let Ok(obsm_map) = read_obsm_at(&file, &obsm_path) {
            for (key, batch) in &obsm_map {
                writer
                    .write_obsm_for(modality_id, key, batch)
                    .map_err(ConvertError::from)?;
            }
        }
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let modality_names_for_json: Vec<&String> =
        modality_meta.iter().map(|(name, _, _)| name).collect();
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
) -> Result<super::h5ad_stream::XStreamReader, ConvertError> {
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

/// Memory budget for the streaming CSR→CSC transpose during h5mu
/// conversion. Mirrors the `CONVERT_CSC_MEMORY_BYTES` constant in the
/// h5ad path (4 GiB).
const CONVERT_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

#[allow(clippy::too_many_arguments)]
fn write_modality_csr_shards(
    writer: &mut ScxWriter,
    modality_id: u8,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    _n_vars: usize,
    shard_target_rows: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
) -> Result<(), ConvertError> {
    let mut row_start: usize = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        let shard_indptr_slice = &indptr[row_start..=row_end];
        let base = shard_indptr_slice[0];
        let shard_indptr: Vec<u64> = shard_indptr_slice
            .iter()
            .map(|&v| {
                if v < base {
                    Err(ConvertError::Other(format!(
                        "indptr value {v} less than base {base}"
                    )))
                } else {
                    Ok((v - base) as u64)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let nnz_start = usize::try_from(base)
            .map_err(|_| ConvertError::Other(format!("negative indptr base {base}")))?;
        let nnz_end = usize::try_from(*shard_indptr_slice.last().unwrap()).map_err(|_| {
            ConvertError::Other(format!(
                "negative indptr value {}",
                shard_indptr_slice.last().unwrap()
            ))
        })?;
        let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
            .iter()
            .map(|&v| {
                u32::try_from(v)
                    .map_err(|_| ConvertError::Other(format!("negative column index {v}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shard_data = &data[nnz_start..nnz_end];
        let raw_values = values_to_raw_bytes(shard_data, value_encoding).map_err(ScxError::from)?;

        writer
            .write_csr_shard_for(
                modality_id,
                &shard_indptr,
                &shard_indices,
                &raw_values,
                codec_id,
                value_encoding,
                row_start as u64,
            )
            .map_err(ConvertError::from)?;

        row_start = row_end;
    }
    Ok(())
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
) -> Result<(), ConvertError> {
    let csr = scx_sparse::ScxCsr::new_unchecked(
        (n_obs, n_vars),
        indptr.to_vec(),
        indices.to_vec(),
        data.to_vec(),
    );
    let shards = std::slice::from_ref(&csr);

    let mut iter = scx_sparse::streaming_csr_to_csc_iter_with_cap(
        shards,
        n_obs,
        n_vars,
        CONVERT_CSC_MEMORY_BYTES,
        csc_cols_per_shard,
    )
    .map_err(|e| ConvertError::Other(format!("CSC transpose failed: {e}")))?;

    loop {
        let col_start = iter.current_col_start() as u64;
        let chunk = match iter.next() {
            Some(c) => c.map_err(|e| ConvertError::Other(format!("CSC chunk failed: {e}")))?,
            None => break,
        };

        let csc_indptr_u64: Vec<u64> = chunk.indptr.iter().map(|&v| v as u64).collect();
        let csc_indices_u32: Vec<u32> = chunk.indices.iter().map(|&i| i as u32).collect();
        let raw_values =
            values_to_raw_bytes(&chunk.data, value_encoding).map_err(ScxError::from)?;

        writer
            .write_csc_shard_for(
                modality_id,
                &csc_indptr_u64,
                &csc_indices_u32,
                &raw_values,
                codec_id,
                value_encoding,
                col_start,
            )
            .map_err(ConvertError::from)?;
    }
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
    _n_vars: usize,
    shard_target_rows: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
) -> Result<(), ConvertError> {
    let mut row_start: usize = 0;
    let mut shard_idx: u32 = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        let shard_indptr_slice = &indptr[row_start..=row_end];
        let base = shard_indptr_slice[0];
        let shard_indptr: Vec<u64> = shard_indptr_slice
            .iter()
            .map(|&v| {
                if v < base {
                    Err(ConvertError::Other(format!(
                        "indptr value {v} less than base {base}"
                    )))
                } else {
                    Ok((v - base) as u64)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let nnz_start = usize::try_from(base)
            .map_err(|_| ConvertError::Other(format!("negative indptr base {base}")))?;
        let nnz_end = usize::try_from(*shard_indptr_slice.last().unwrap()).map_err(|_| {
            ConvertError::Other(format!(
                "negative indptr value {}",
                shard_indptr_slice.last().unwrap()
            ))
        })?;
        let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
            .iter()
            .map(|&v| {
                u32::try_from(v)
                    .map_err(|_| ConvertError::Other(format!("negative column index {v}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shard_data = &data[nnz_start..nnz_end];
        let raw_values = values_to_raw_bytes(shard_data, value_encoding).map_err(ScxError::from)?;

        writer
            .write_layer_csr_shard_for(
                modality_id,
                layer_name,
                shard_idx,
                &shard_indptr,
                &shard_indices,
                &raw_values,
                codec_id,
                value_encoding,
                row_start as u64,
            )
            .map_err(ConvertError::from)?;

        row_start = row_end;
        shard_idx += 1;
    }
    Ok(())
}
