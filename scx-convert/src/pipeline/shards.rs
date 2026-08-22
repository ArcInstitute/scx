//! Shard writers for the matrix sections: X, `raw/X`, layers, and the
//! gene-major CSC sidecar.
//!
//! Each takes an already-resident matrix and emits framed shards through
//! `ScxWriter`; the streaming ingest path writes X through
//! [`super::coordinator`] instead and reaches here only for `raw` and layers.

use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::error::ScxError;
use scx_format_io::modality::ModalityType;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{encode_one_shard, FramingConfig};
use scx_sparse::canonicalize_csr;

use super::bitmap::build_and_write_bitmap_for_shard;
use super::coordinator::run_streaming_writer_coordinator;
use super::error::ConvertError;
use crate::detect::MatrixFormat;
use crate::dtype::{detect_value_encoding, index_dtype_for, values_to_raw_bytes};
use crate::h5ad::csc_stream::open_csc_streaming;
use crate::h5ad::dense_stream::open_dense_streaming;
use crate::h5ad::read::read_dataframe_group;
use crate::h5ad::stream::open_x_streaming;
use crate::options::IngestOptions;
use crate::stream::CsrShardStream;
use crate::warnings::{ConvertWarning, WarningSink};
use arrow::record_batch::RecordBatch;
use scx_format_io::BitmapPolicy;

/// Streaming CSR → CSC transpose over the in-memory matrix, with
/// the result written shard-by-shard via `writer.write_csc_shard`.
///
/// Each emitted shard's column count is bounded by
/// `csc_cols_per_shard` (or the memory budget, whichever is
/// smaller). The CSR data already lives in `(indptr, indices, data)`
/// at this point in the pipeline — passed straight to the streaming
/// iterator without re-reading from disk.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_csc_shards_from_csr(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    csc_cols_per_shard: usize,
    // Caller's `memory_budget`; capped at the sidecar builder's own default by
    // `budget::csc_sidecar_bytes`. Threaded rather than defaulted because the
    // bound is on the emitted shard's column count, so ignoring it silently let
    // a `--memory-budget 512M --csc always` convert claim 4 GiB.
    memory_budget: Option<u64>,
    framing: Option<scx_format_io::FramingConfig>,
) -> Result<(), ConvertError> {
    // Wrap the canonical in-memory CSR as a single ScxCsr "shard"
    // for the transpose iterator so the optional CSC sidecar mirrors
    // the row-major shards emitted by `write_csr_shards`.
    let mut indptr_u64: Vec<u64> = indptr
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(ConvertError::Other(format!(
                    "negative CSR indptr value {v} before CSC transpose"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut indices_u32: Vec<u32> = indices
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(ConvertError::Other(format!(
                    "negative CSR index value {v} before CSC transpose"
                )))
            } else {
                Ok(v as u32)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut values = data.to_vec();
    canonicalize_csr(&mut indptr_u64, &mut indices_u32, &mut values);
    let csr = scx_sparse::ScxCsr::new_unchecked(
        (n_obs, n_vars),
        indptr_u64.iter().map(|&v| v as i64).collect(),
        indices_u32.iter().map(|&v| v as i32).collect(),
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
        crate::budget::csc_sidecar_bytes(memory_budget) as usize,
        None,
        framing,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_csr_shards(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    shard_target_rows: usize,
    codec_id: CodecId,
    index_dtype: u8,
    bitmap_policy: BitmapPolicy,
    modality_type: ModalityType,
    framing: Option<FramingConfig>,
    sink: &mut WarningSink,
) -> Result<Vec<(u64, u64)>, ConvertError> {
    let n_vars_u32 = u32::try_from(n_vars)
        .map_err(|_| ConvertError::Other(format!("n_vars {n_vars} exceeds u32::MAX")))?;

    let mut row_ranges: Vec<(u64, u64)> = Vec::new();
    let mut row_start: usize = 0;
    let mut shard_idx: u32 = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        // Slice indptr for this shard, then validate + rebase + cast
        // through the shared scx-sparse helper. C6: this eager site
        // previously did a manual `v >= base` check that skipped the
        // column-bound check (`indices < n_vars`); `rebase_csr_shard`
        // runs the full `validate_csr_arrays`.
        let shard_indptr_slice = &indptr[row_start..=row_end];
        let (nnz_start, nnz_end) =
            scx_sparse::shard_nnz_bounds(shard_indptr_slice, indices.len().min(data.len()))
                .map_err(|e| ConvertError::Other(format!("X shard validation failed: {e}")))?;
        let (shard_indptr, shard_indices) = scx_sparse::rebase_csr_shard(
            shard_indptr_slice,
            &indices[nnz_start..nnz_end],
            n_vars as u64,
        )
        .map_err(|e| ConvertError::Other(format!("X shard validation failed: {e}")))?;
        let mut shard_indptr = shard_indptr;
        let mut shard_indices = shard_indices;
        let mut shard_data = data[nnz_start..nnz_end].to_vec();
        canonicalize_csr(&mut shard_indptr, &mut shard_indices, &mut shard_data);

        // Pre-encode so the bitmap auto-policy can compare against the
        // post-codec section length (matches streaming + python in-memory
        // paths). Section name `X_shard_{idx}` mirrors the name that
        // `ScxWriter::write_csr_shard` constructs internally.
        let pre = encode_one_shard(
            &shard_indptr,
            &shard_indices,
            &shard_data,
            Some(codec_id),
            index_dtype,
            n_vars_u32,
            row_start as u64,
            SectionType::CsrShard,
            modality_type,
            format!("X_shard_{shard_idx}"),
            framing,
        )?;
        let encoded_csr_size = pre.section_length as usize;
        writer.write_preencoded_shard(pre)?;

        // Phase 5b: detection bitmap, post-CSR-write so a failed bitmap
        // never strands a half-written file.
        let n_rows_u32 = u32::try_from(row_end - row_start).map_err(|_| {
            ConvertError::Other(format!(
                "shard rows {} exceeds u32::MAX",
                row_end - row_start
            ))
        })?;
        build_and_write_bitmap_for_shard(
            writer,
            &shard_indptr,
            &shard_indices,
            row_start as u64,
            n_rows_u32,
            n_vars_u32,
            encoded_csr_size,
            bitmap_policy,
            modality_type,
            None,
            sink,
        )?;

        row_ranges.push((row_start as u64, row_end as u64));
        row_start = row_end;
        shard_idx += 1;
    }
    Ok(row_ranges)
}

/// Write the `adata.raw` count matrix as `RawCsrShard` sections plus the
/// `raw/var` metadata section. Raw shares X's obs axis but has its OWN
/// column count (`raw_n_vars`), so `writer.set_raw_n_vars` is called
/// first and a per-shard `index_dtype` is resolved against `raw_n_vars`.
/// Eager (the whole raw CSR is materialized) — mirrors the non-streaming
/// X path; streaming raw is a deferred optimization.
#[allow(clippy::too_many_arguments)]
fn write_raw_csr_shards(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    raw_var: &RecordBatch,
    n_obs: usize,
    raw_n_vars: usize,
    shard_target_rows: usize,
    codec_id: CodecId,
    value_encoding: ValueEncoding,
) -> Result<(), ConvertError> {
    writer.set_raw_n_vars(raw_n_vars as u64);

    let mut row_start: usize = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);
        let shard_indptr_slice = &indptr[row_start..=row_end];
        let (nnz_start, nnz_end) =
            scx_sparse::shard_nnz_bounds(shard_indptr_slice, indices.len().min(data.len()))
                .map_err(|e| ConvertError::Other(format!("raw shard validation failed: {e}")))?;
        let (shard_indptr, shard_indices) = scx_sparse::rebase_csr_shard(
            shard_indptr_slice,
            &indices[nnz_start..nnz_end],
            raw_n_vars as u64,
        )
        .map_err(|e| ConvertError::Other(format!("raw shard validation failed: {e}")))?;
        let shard_data = &data[nnz_start..nnz_end];
        let raw_values = values_to_raw_bytes(shard_data, value_encoding).map_err(ScxError::from)?;

        writer.write_raw_csr_shard(
            &shard_indptr,
            &shard_indices,
            &raw_values,
            codec_id,
            value_encoding,
            row_start as u64,
        )?;
        row_start = row_end;
    }

    writer.write_raw_var(raw_var)?;
    Ok(())
}

/// Read the optional `/raw` group from an open h5ad and, if present,
/// write it as the raw section family. Shared by the eager and streaming
/// ingest paths. Asserts `raw.n_obs == n_obs` (raw shares the obs axis).
pub(super) fn ingest_raw_if_present(
    file: &hdf5::File,
    writer: &mut ScxWriter,
    n_obs: usize,
    opts: &IngestOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let Some(((indptr, indices, data, raw_n_obs, raw_n_vars), raw_var)) =
        crate::h5ad::read::read_raw_group(file, sink)?
    else {
        return Ok(());
    };
    if raw_n_obs != n_obs {
        return Err(ConvertError::Other(format!(
            "raw/X has {raw_n_obs} rows but X has {n_obs}; \
             adata.raw must share the obs axis"
        )));
    }
    let (value_encoding, codec_id) =
        detect_value_encoding(&data, opts.codec).map_err(ScxError::from)?;
    write_raw_csr_shards(
        writer,
        &indptr,
        &indices,
        &data,
        &raw_var,
        n_obs,
        raw_n_vars,
        opts.shard_target_rows as usize,
        codec_id,
        value_encoding,
    )
}

/// Streaming variant of [`ingest_raw_if_present`]: open `raw/X` as a
/// streaming reader and drive it through the shared writer coordinator
/// with the `RawCsrShard` section type, so peak RSS stays bounded to one
/// raw shard at a time (mirrors the streaming `/X` path). `raw/var` is a
/// small DataFrame and is read eagerly. Used by `h5ad_to_scx_streaming`.
///
/// The coordinator gates detection bitmaps on `section_type == CsrShard`
/// and `write_preencoded_shard` only bumps `csr`/`csc` counters, so raw
/// shards add no bitmap and do not perturb the main matrix's header
/// counts; presence is recorded via the `has_raw` flag at `finish()`.
pub(super) fn ingest_raw_streaming(
    file: &hdf5::File,
    writer: &mut ScxWriter,
    n_obs: usize,
    opts: &IngestOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    if file.group("raw").is_err() {
        return Ok(());
    }
    if file.group("raw/X").is_err() && file.dataset("raw/X").is_err() {
        return Ok(());
    }

    crate::h5ad::read::warn_raw_varm_if_present(file, sink);

    let raw_format = crate::detect::detect_matrix_format_at(file, "raw/X", sink)?;
    let mut raw_reader: Box<dyn CsrShardStream> = match raw_format {
        MatrixFormat::Csr => Box::new(open_x_streaming(file, "raw/X", raw_format, sink)?),
        MatrixFormat::Dense => Box::new(open_dense_streaming(file, "raw/X", opts, sink)?),
        MatrixFormat::Csc => open_csc_streaming(file, "raw/X", opts, sink)?,
    };

    let raw_n_obs = raw_reader.n_obs() as usize;
    if raw_n_obs != n_obs {
        return Err(ConvertError::Other(format!(
            "raw/X has {raw_n_obs} rows but X has {n_obs}; \
             adata.raw must share the obs axis"
        )));
    }
    let raw_n_vars = raw_reader.n_vars() as usize;

    // `adata.raw` shares the obs axis, but the reorder permutation is applied
    // only to X/obs/obsm/layers — raw is streamed in source order. Rather than
    // emit a raw matrix whose rows no longer line up with the reordered cells,
    // drop it (with a visible warning) under any obs-axis reorder (--sort-by or
    // --group-by), mirroring the standalone `scx sort` engine which also drops
    // raw.
    //
    // `DroppedRawOnWrite`, not `DroppedRaw`: this is a conversion producing an
    // output file with no raw, so the read-side variant's "the on-disk raw
    // sections are preserved" would describe the h5ad input while the user is
    // asking about the SCX file being written. The reason is supplied here
    // rather than baked into the variant — this caller is already converting
    // *from* the h5ad, so the SCX → SCX door's "convert from the h5ad" remedy
    // would be nonsense.
    if !opts.sort_by.is_empty() || opts.group_by.is_some() {
        sink.emit(ConvertWarning::DroppedRawOnWrite {
            raw_n_vars,
            reason: "reorder-on-convert (--sort-by / --group-by) permutes X, obs, obsm and \
                     layers, but raw is streamed in source order, so carrying it would leave \
                     raw's rows attached to the wrong cells. Convert without the reorder to \
                     keep raw",
        });
        return Ok(());
    }

    let raw_n_vars_u32 = u32::try_from(raw_n_vars)
        .map_err(|_| ConvertError::Other(format!("raw n_vars {raw_n_vars} exceeds u32::MAX")))?;
    let raw_index_dtype: u8 = index_dtype_for(raw_n_vars as u64);

    run_streaming_writer_coordinator(
        raw_reader.as_mut(),
        writer,
        opts,
        raw_index_dtype,
        raw_n_vars_u32,
        SectionType::RawCsrShard,
        ModalityType::Rna,
        "raw/X_shard",
        sink,
        None,
    )?;

    let raw_var = read_dataframe_group(file, "raw/var", sink)?;
    writer.write_raw_var(&raw_var)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_layer_shards(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    shard_target_rows: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    index_dtype: u8,
    layer_name: &str,
) -> Result<(), ConvertError> {
    let _ = index_dtype;
    let mut row_start: usize = 0;
    let mut shard_idx: u32 = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        // Validate + rebase + cast through the shared scx-sparse helper
        // (C6: adds the column-bound check this eager site previously
        // skipped).
        let shard_indptr_slice = &indptr[row_start..=row_end];
        let (nnz_start, nnz_end) =
            scx_sparse::shard_nnz_bounds(shard_indptr_slice, indices.len().min(data.len()))
                .map_err(|e| {
                    ConvertError::Other(format!(
                        "layer '{layer_name}' shard validation failed: {e}"
                    ))
                })?;
        let (shard_indptr, shard_indices) = scx_sparse::rebase_csr_shard(
            shard_indptr_slice,
            &indices[nnz_start..nnz_end],
            n_vars as u64,
        )
        .map_err(|e| {
            ConvertError::Other(format!("layer '{layer_name}' shard validation failed: {e}"))
        })?;
        let shard_data = &data[nnz_start..nnz_end];
        let raw_values = values_to_raw_bytes(shard_data, value_encoding).map_err(ScxError::from)?;

        writer.write_layer_csr_shard(
            &shard_indptr,
            &shard_indices,
            &raw_values,
            codec_id,
            value_encoding,
            row_start as u64,
            layer_name,
            shard_idx,
        )?;

        row_start = row_end;
        shard_idx += 1;
    }
    Ok(())
}
