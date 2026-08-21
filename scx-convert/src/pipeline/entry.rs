//! The eager entry points: whole-matrix h5ad ingest, 10x HDF5 ingest, and the
//! SCX -> h5ad export front doors.
//!
//! "Eager" means the matrix is materialised in memory. The streaming siblings
//! are [`super::entry_streaming`]; the export front doors here are thin
//! forwards into `crate::h5ad::stream_write`.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use scx_format_io::error::ScxError;
use scx_format_io::header::FileHeader;
use scx_format_io::modality::ModalityType;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::writer::ScxWriter;

use super::error::ConvertError;
use super::index::build_and_write_predicate_indexes;
use super::mappings::{
    write_dense_mapping_section, write_sparse_mapping_section, DenseMappingKind, SparseMappingKind,
};
use super::shards::{
    ingest_raw_if_present, write_csc_shards_from_csr, write_csr_shards, write_layer_shards,
};
use super::write_ingest_obs;
use crate::detect::{detect_input_format, detect_matrix_format, InputFormat};
use crate::dtype::{detect_value_encoding, index_dtype_for};
use crate::h5ad::read::{read_dataframe_group, read_layers, read_uns, read_x_matrix};
use crate::h5ad::write::write_scx_to_h5ad;
use crate::options::IngestOptions;
use crate::tenx_read::read_tenx_h5;
use crate::warnings::{ConvertWarning, WarningSink};

pub fn h5ad_to_scx(
    input: &Path,
    output: &Path,
    opts: &IngestOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    // Reorder-on-convert (`--sort-by` / `--group-by`) runs only on the streaming
    // path (it needs the random-access gather). Callers route these to streaming;
    // guard the eager path defensively.
    if !opts.sort_by.is_empty() || opts.group_by.is_some() {
        return Err(ConvertError::Other(
            "reorder-on-convert (--sort-by / --group-by) requires the streaming conversion \
             path; enable streaming (the default) and retry."
                .to_string(),
        ));
    }

    let file = hdf5::File::open(input)?;

    // Validate format
    let format = detect_input_format(&file)?;
    if matches!(format, InputFormat::TenX) {
        return Err(ConvertError::FormatMismatch {
            expected: "h5ad".to_string(),
            got: "10x".to_string(),
        });
    }

    // Read X matrix
    let matrix_format = detect_matrix_format(&file, sink)?;
    let (indptr, indices, data, n_obs, n_vars) = read_x_matrix(&file, matrix_format)?;
    let nnz = *indptr.last().unwrap_or(&0) as u64;

    // Detect encoding and codec
    let (value_encoding, codec_id) =
        detect_value_encoding(&data, opts.codec).map_err(ScxError::from)?;
    let index_dtype: u8 = index_dtype_for(n_vars as u64);

    // Build header
    let mut header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        nnz,
        opts.shard_target_rows,
        codec_id as u8,
        index_dtype,
    );
    // F5 Phase 1: row-group framing produces a v4 file (its shards are v2).
    if opts.framing().is_some() {
        header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION;
    }

    let mut writer = ScxWriter::new(output, header)?;
    // F5-b: frame CSC sidecars / layers / obsp shards written through this writer
    // (CSR X shards frame via encode_one_shard). No-op unless framing is on.
    writer.set_framing(opts.framing());

    // Write obs/var
    let obs = read_dataframe_group(&file, "obs", sink)?;
    let var = read_dataframe_group(&file, "var", sink)?;
    write_ingest_obs(&mut writer, &obs, opts)?;
    writer.write_var(&var)?;

    // Write CSR shards
    let csr_row_ranges = write_csr_shards(
        &mut writer,
        &indptr,
        &indices,
        &data,
        n_obs,
        n_vars,
        opts.shard_target_rows as usize,
        codec_id,
        index_dtype,
        opts.bitmap,
        ModalityType::Rna,
        opts.framing(),
        sink,
    )?;

    // Optional CSC sidecar — streaming transpose over the in-memory
    // CSR data, one shard per chunk.
    if opts.csc.should_build_csc(n_obs as u64, n_vars as u64) {
        write_csc_shards_from_csr(
            &mut writer,
            &indptr,
            &indices,
            &data,
            n_obs,
            n_vars,
            value_encoding,
            codec_id,
            opts.csc_cols_per_shard,
            opts.memory_budget,
            opts.framing(),
        )?;
    }

    // Optional `adata.raw` count matrix (its own var axis) → raw section
    // family. Must run while `file` is still open.
    ingest_raw_if_present(&file, &mut writer, n_obs, opts, sink)?;

    // Write optional sections. Even on this non-streaming path we emit
    // obsm/varm/obsp/varp as sharded sections so the on-disk layout is
    // uniform with `h5ad_to_scx_streaming`.
    write_dense_mapping_section(
        &file,
        &mut writer,
        None,
        "obsm",
        opts.shard_target_rows,
        DenseMappingKind::Obsm,
        None,
        sink,
    )?;
    write_dense_mapping_section(
        &file,
        &mut writer,
        None,
        "varm",
        opts.shard_target_rows,
        DenseMappingKind::Varm,
        None,
        sink,
    )?;
    write_sparse_mapping_section(
        &file,
        &mut writer,
        None,
        "obsp",
        opts.shard_target_rows,
        SparseMappingKind::Obsp,
        sink,
    )?;
    write_sparse_mapping_section(
        &file,
        &mut writer,
        None,
        "varp",
        opts.shard_target_rows,
        SparseMappingKind::Varp,
        sink,
    )?;

    // Read /uns only when the group exists; key-level failures route
    // through the sink (lenient) or propagate (strict).
    if file.group("uns").is_ok() {
        let uns = read_uns(&file, opts.strict_uns, sink)?;
        writer.write_uns(&uns)?;
    }

    if let Ok(layers) = read_layers(&file, sink) {
        for (layer_name, (l_indptr, l_indices, l_data, l_nobs, l_nvars)) in &layers {
            // C2: validate the layer's shape against X, matching the streaming
            // path. A layer whose row/column count disagrees with /X would
            // otherwise produce an SCX file whose layer dimensions silently
            // diverge from the primary matrix.
            if *l_nobs != n_obs {
                sink.emit(ConvertWarning::LayerSkipped {
                    name: layer_name.clone(),
                    reason: format!("n_obs {l_nobs} does not match X n_obs {n_obs}"),
                });
                continue;
            }
            if *l_nvars != n_vars {
                sink.emit(ConvertWarning::LayerSkipped {
                    name: layer_name.clone(),
                    reason: format!("n_vars {l_nvars} does not match X n_vars {n_vars}"),
                });
                continue;
            }
            let (l_enc, l_codec) =
                detect_value_encoding(l_data, opts.codec).map_err(ScxError::from)?;
            let l_index_dtype: u8 = index_dtype_for(*l_nvars as u64);
            write_layer_shards(
                &mut writer,
                l_indptr,
                l_indices,
                l_data,
                *l_nobs,
                *l_nvars,
                opts.shard_target_rows as usize,
                l_enc,
                l_codec,
                l_index_dtype,
                layer_name,
            )?;
        }
    }

    // Phase 5a: predicate indexes built from the obs/var we just wrote,
    // using the actual on-disk shard boundaries.
    let (obs_indexed, var_indexed) = build_and_write_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &csr_row_ranges,
        n_vars,
        opts,
        &[],
        sink,
    )?;

    // Write provenance
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
            "format": "h5ad",
            "codec_selection": opts.codec_selection_value(),
            "warnings": sink.summary_json(),
            "predicate_index": {
                "obs_columns": obs_indexed,
                "var_columns": var_indexed,
                "preset": opts.index_preset,
            },
        })
        .to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

pub fn tenx_to_scx(
    input: &Path,
    output: &Path,
    opts: &IngestOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let file = hdf5::File::open(input)?;

    let format = detect_input_format(&file)?;
    if matches!(format, InputFormat::H5ad) {
        return Err(ConvertError::FormatMismatch {
            expected: "10x".to_string(),
            got: "h5ad".to_string(),
        });
    }

    // A CellBender output also has `/matrix/barcodes`, so it lands here by
    // extension. Redirect rather than failing somewhere deep inside the 10x
    // reader — and note this is a *different* operation, not a conversion:
    // the corrected counts belong on an existing file's obs axis.
    if file.group("droplet_latents").is_ok() {
        return Err(ConvertError::Other(format!(
            "'{}' looks like a CellBender remove-background output (it has a \
             /droplet_latents group), not a 10x CellRanger matrix. Attach it to \
             an existing SCX file with: scx cellbender-import <target.scx> {}",
            input.display(),
            input.display()
        )));
    }

    let tenx = read_tenx_h5(&file)?;
    let nnz = *tenx.indptr.last().unwrap_or(&0) as u64;
    let (value_encoding, codec_id) =
        detect_value_encoding(&tenx.data, opts.codec).map_err(ScxError::from)?;
    let index_dtype: u8 = index_dtype_for(tenx.n_genes as u64);

    let mut header = FileHeader::new_single_modality(
        tenx.n_cells as u64,
        tenx.n_genes as u64,
        nnz,
        opts.shard_target_rows,
        codec_id as u8,
        index_dtype,
    );
    // F5 Phase 1: row-group framing produces a v4 file (its shards are v2).
    if opts.framing().is_some() {
        header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION;
    }

    let mut writer = ScxWriter::new(output, header)?;
    // F5-b: frame CSC sidecars / layers / obsp shards written through this writer
    // (CSR X shards frame via encode_one_shard). No-op unless framing is on.
    writer.set_framing(opts.framing());
    write_ingest_obs(&mut writer, &tenx.obs, opts)?;
    writer.write_var(&tenx.var)?;

    let csr_row_ranges = write_csr_shards(
        &mut writer,
        &tenx.indptr,
        &tenx.indices,
        &tenx.data,
        tenx.n_cells,
        tenx.n_genes,
        opts.shard_target_rows as usize,
        codec_id,
        index_dtype,
        opts.bitmap,
        ModalityType::Rna,
        opts.framing(),
        sink,
    )?;

    // Optional CSC sidecar — same streaming transpose as h5ad.
    if opts
        .csc
        .should_build_csc(tenx.n_cells as u64, tenx.n_genes as u64)
    {
        write_csc_shards_from_csr(
            &mut writer,
            &tenx.indptr,
            &tenx.indices,
            &tenx.data,
            tenx.n_cells,
            tenx.n_genes,
            value_encoding,
            codec_id,
            opts.csc_cols_per_shard,
            opts.memory_budget,
            opts.framing(),
        )?;
    }

    // Phase 5a: predicate indexes from 10x obs/var. 10x obs is usually
    // just barcodes; var has gene_name / feature_type. Auto-detection
    // is the common path here.
    let (obs_indexed, var_indexed) = build_and_write_predicate_indexes(
        &mut writer,
        &tenx.obs,
        &tenx.var,
        &csr_row_ranges,
        tenx.n_genes,
        opts,
        &[],
        sink,
    )?;

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
            "format": "10x",
            "codec_selection": opts.codec_selection_value(),
            "warnings": sink.summary_json(),
            "predicate_index": {
                "obs_columns": obs_indexed,
                "var_columns": var_indexed,
                "preset": opts.index_preset,
            },
        })
        .to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

pub fn scx_to_h5ad(
    scx_path: &Path,
    h5ad_path: &Path,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    write_scx_to_h5ad(scx_path, h5ad_path, sink)
}

/// Streaming SCX → h5ad. Walks SCX CSR shards in row order and writes
/// `/X/{indptr,indices,data}` (and `/layers/{name}/…`) via pre-
/// allocated HDF5 hyperslab slices, so peak RSS is bounded by one
/// shard's worth of CSR plus encode buffers regardless of file size.
/// Single-modality only; multimodal SCX files must use
/// [`scx_to_h5mu_streaming`] or [`scx_modality_to_h5ad_streaming`].
pub fn scx_to_h5ad_streaming(
    scx_path: &Path,
    h5ad_path: &Path,
    opts: &crate::ExportOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    crate::h5ad::stream_write::write_scx_to_h5ad_streaming(scx_path, h5ad_path, opts, sink)
}
