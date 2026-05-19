use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use scx_codec::{CodecId, ValueEncoding};
use scx_format::encode_one_shard;
use scx_format::error::ScxError;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::modality::ModalityType;
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::{PreEncodedSection, ScxWriter};
use scx_sparse::{drop_explicit_zeros_inplace, sort_csr_rows_in_place};

use super::csc_stream::{open_csc_layer_streaming, open_csc_streaming};
use super::dense_stream::{open_dense_layer_streaming, open_dense_streaming};
use super::detect::{detect_input_format, detect_matrix_format, InputFormat, MatrixFormat};
use super::dtype::{detect_value_encoding, values_to_raw_bytes};
use super::h5ad_read::{
    read_dataframe_group, read_layers, read_obsm, read_uns, read_varm, read_x_matrix,
};
use super::h5ad_stream::{open_layer_streaming, open_x_streaming};
use super::h5ad_write::write_scx_to_h5ad;
use super::stream::CsrShardStream;
use super::tenx_read::read_tenx_h5;
use super::warnings::{ConvertWarning, WarningSink};
use arrow::record_batch::RecordBatch;
use scx_engine::{
    build_and_write_conversion_predicate_indexes, BuildOutcome, ConversionPredicateIndexOptions,
    SkipReason,
};
use scx_format::bitmap::BitmapShard;

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),

    #[error("SCX error: {0}")]
    Scx(#[from] ScxError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),

    /// Narrowing an HDF5 source value to the target Rust type would
    /// truncate. Used by `read_*_dataset` / `read_slice_*` when a
    /// source `i64` / `u32` / `u64` value falls outside the target
    /// range. Silent truncation of CSR indptr / indices would corrupt
    /// the on-disk sparse layout, so the conversion fails loudly.
    #[error("value {value} from {source_dtype} dataset '{path}' overflows {target} target range")]
    IndexOverflow {
        path: String,
        source_dtype: &'static str,
        target: &'static str,
        value: String,
    },

    #[error("format mismatch: expected {expected}, got {got}")]
    FormatMismatch { expected: String, got: String },

    #[error("streaming unsupported: {0}")]
    StreamingUnsupported(String),

    /// A per-shard read or encode failed on one of the
    /// parallel reader workers. The wrapper carries the row range and
    /// source-matrix name so the user knows exactly which shard
    /// produced the error.
    #[error(
        "shard read failed at rows [{row_start}, {}) of '{source}': {inner}",
        row_start + *n_rows as u64
    )]
    ShardRead {
        row_start: u64,
        n_rows: u32,
        source: String,
        #[source]
        inner: Box<ConvertError>,
    },

    #[error("{0}")]
    Other(String),
}

pub struct ConvertOptions {
    pub shard_target_rows: u32,
    /// Explicit codec override. None = auto-select based on value distribution.
    pub codec: Option<CodecId>,
    /// When `true`, also emit a CSC sidecar at write time (multi-shard
    /// column-major layout). The CSR shards are still written first;
    /// CSC chunks are produced via streaming transpose over the
    /// in-memory CSR data.
    pub csc: bool,
    /// Columns per CSC shard when `csc == true`. `0` disables the
    /// cap (single CSC shard, memory permitting).
    pub csc_cols_per_shard: usize,
    /// Tool name recorded in the provenance entry. Defaults to
    /// `"scx-cli"`; `pyscx` overrides this to `"pyscx"` so the
    /// recorded provenance reflects the actual caller.
    pub tool: String,
    /// Phase-0.4 budget shared by dense slab sizing (Phase 1), CSC
    /// transpose buffers (Phase 2), cloud in-flight bytes (Phase 7),
    /// and worker derate (Phase 8c). `None` keeps each phase's own
    /// sizing heuristic. Parse user-facing strings with
    /// [`crate::MemoryBudget::parse`].
    pub memory_budget: Option<u64>,
    /// Prefer streaming I/O over full materialisation when the input
    /// supports it (CSR and dense `/X`). When `false`, `h5ad_to_scx`
    /// keeps the legacy in-memory path. When `true` (default), CSR
    /// and dense routes go through `h5ad_to_scx_streaming`; CSC-on-
    /// disk still errors with the Phase 2 message.
    pub stream: bool,
    /// Fail conversion on the first unsupported `uns` key instead of
    /// skipping it with a warning. Default `false` keeps the existing
    /// lenient behaviour.
    pub strict_uns: bool,
    /// Treat dense values with absolute magnitude `<= dense_zero_epsilon`
    /// as zeros during sparsification. Default `0.0` keeps the
    /// equality-to-zero filtering that `scx_sparse::dense_to_csr`
    /// already does (matches scipy `csr_matrix(dense)`).
    pub dense_zero_epsilon: f32,
    /// Directory under which the Phase 2 external CSC → CSR transpose
    /// writes its session temp directory
    /// (`<temp_dir>/scx-transpose-<pid>-<random>/`). `None` falls back
    /// to [`std::env::temp_dir`]. Used only when the budget arithmetic
    /// forces the external path; the in-memory CSC route never
    /// touches disk.
    pub temp_dir: Option<std::path::PathBuf>,
    /// Phase 3: Filter h5mu input to only the named modalities.
    /// `None` (default) keeps every modality. Unknown names error
    /// with the full list of available modalities.
    pub modalities: Option<Vec<String>>,
    /// Phase 3: Explicit modality-type overrides keyed by modality
    /// name. Modalities not listed get
    /// [`crate::infer_modality_type_from_name`] and emit
    /// [`crate::ConvertWarning::ModalityTypeInferred`].
    pub modality_types: Vec<(String, ModalityType)>,
    /// Phase 5a: force-index these obs columns at conversion time.
    /// Missing or unsupported columns fail the convert.
    pub index_obs: Vec<String>,
    /// Phase 5a: force-index these var columns at conversion time.
    /// Missing or unsupported columns fail the convert.
    pub index_var: Vec<String>,
    /// Phase 5a: named column preset
    /// (`cellxgene` / `perturbseq` / `training`). Missing preset
    /// columns warn but don't fail.
    pub index_preset: Option<String>,
    /// Phase 5a: cardinality cap for auto-detected index columns
    /// when neither `index_obs`/`index_var` nor `index_preset` is set.
    /// Default 1000.
    pub index_auto_threshold: usize,
    /// Phase 5b: detection-bitmap shard generation policy. Default
    /// `Off` (explicit opt-in, matches `--csc` ergonomics).
    pub bitmap: BitmapPolicy,
    /// Streaming reader worker thread count.
    /// `None` (default) = auto: use `RAYON_NUM_THREADS` if set, else
    /// [`std::thread::available_parallelism`]. `Some(1)` forces the
    /// sequential coordinator. `Some(N>1)` requests N rayon workers;
    /// the coordinator falls back to sequential when (a) libhdf5
    /// isn't built threadsafe, (b) the reader doesn't implement
    /// [`crate::stream::IndexedCsrShardStream`], or (c) the
    /// `memory_budget` derate forces it. The parallel path is
    /// byte-identical to the sequential path.
    pub reader_threads: Option<usize>,
    /// Backpressure window between the parallel encoder pool and the
    /// ordered writer. Default 4. The parallel coordinator caps
    /// outstanding shards (encoding + in channel + in reorder buffer)
    /// at `reader_threads + writer_queue_depth` via a rolling-window
    /// spawn, so peak RSS scales with that sum, not with the total
    /// shard count. Larger values give the slow shard a deeper
    /// look-ahead buffer; smaller values risk starving encoders when
    /// one shard takes much longer than its siblings.
    pub writer_queue_depth: usize,
}

/// Phase 5b: density threshold below which `--bitmap=auto` considers a
/// shard "sparse enough" for bitmaps. Above this, the CSR storage is
/// already dense-ish (>30% nonzero) and bitmaps offer little win.
const BITMAP_AUTO_DENSITY_THRESHOLD: f32 = 0.30;
/// Phase 5b: `n_vars` cap for `--bitmap=auto`. Tied to the per-row
/// allocator cost on extremely wide matrices.
const BITMAP_AUTO_N_VARS_CAP: u32 = 1_000_000;
/// Phase 5b: bitmap size budget under `--bitmap=auto`, expressed as a
/// percentage of the encoded CSR shard size. Roaring sizes vary enough
/// that this is checked *after* the build, not before.
const BITMAP_AUTO_SIZE_PERCENT: usize = 15;

/// Per-modality density assumptions used by the
/// `IndexedCsrShardStream::per_worker_bytes` default impl. The
/// dispatcher uses this estimate to derate workers under
/// `memory_budget`. Over-estimating routes the convert to the
/// sequential coordinator (safe failure mode), so values err
/// conservative. The dense reader overrides `per_worker_bytes`
/// entirely; these constants only affect sparse readers.
pub(crate) const PARALLEL_DENSITY_DEFAULT_DEN: u64 = 20; // ≈ 5 % RNA/general
pub(crate) const PARALLEL_DENSITY_ATAC_DEN: u64 = 10; // ≈ 10 % ATAC peak matrices

/// Outcome of [`maybe_build_bitmap_shard`]. Either a built shard
/// (ready to write) or a structured reason for skipping that the
/// caller forwards to the warning sink.
pub(crate) enum BitmapBuildOutcome {
    Skip { reason: String },
    Built(BitmapShard),
}

/// Pure (no I/O, no sink) bitmap-build helper. Applies
/// the same `--bitmap=auto` density / size gates as the sequential
/// path but returns an outcome instead of writing. The parallel
/// coordinator runs this in a worker thread and the sequential
/// wrapper (`build_and_write_bitmap_for_shard`) routes the outcome
/// through the writer + sink.
#[allow(clippy::too_many_arguments)]
pub(crate) fn maybe_build_bitmap_shard(
    indptr: &[u64],
    indices: &[u32],
    row_start: u64,
    n_rows: u32,
    n_vars: u32,
    encoded_csr_size: usize,
    policy: BitmapPolicy,
    modality_type: ModalityType,
) -> Option<BitmapBuildOutcome> {
    if matches!(policy, BitmapPolicy::Off) {
        return None;
    }
    if n_vars > BITMAP_AUTO_N_VARS_CAP && !matches!(policy, BitmapPolicy::Always) {
        return Some(BitmapBuildOutcome::Skip {
            reason: format!("n_vars {n_vars} exceeds auto cap {BITMAP_AUTO_N_VARS_CAP}"),
        });
    }

    let nnz = *indptr.last().unwrap_or(&0);
    let cells = n_rows as u64;
    let density = if cells == 0 || n_vars == 0 {
        0.0_f32
    } else {
        nnz as f32 / (cells as f32 * n_vars as f32)
    };
    if matches!(policy, BitmapPolicy::Auto)
        && density > BITMAP_AUTO_DENSITY_THRESHOLD
        && !matches!(modality_type, ModalityType::Atac)
    {
        return Some(BitmapBuildOutcome::Skip {
            reason: format!(
                "density {density:.3} above auto threshold {BITMAP_AUTO_DENSITY_THRESHOLD}"
            ),
        });
    }

    let shard = BitmapShard::build_from_csr(row_start, n_rows, n_vars, indptr, indices);

    if matches!(policy, BitmapPolicy::Auto) && !matches!(modality_type, ModalityType::Atac) {
        let est = shard.estimated_encoded_size();
        if encoded_csr_size > 0
            && est.saturating_mul(100) > encoded_csr_size.saturating_mul(BITMAP_AUTO_SIZE_PERCENT)
        {
            return Some(BitmapBuildOutcome::Skip {
                reason: format!(
                    "estimated {est} bytes > {BITMAP_AUTO_SIZE_PERCENT}% of CSR shard ({encoded_csr_size})"
                ),
            });
        }
    }

    Some(BitmapBuildOutcome::Built(shard))
}

/// Build (and conditionally write) a detection bitmap for
/// one CSR shard.
///
/// `modality_type` and `modality_name` drive the auto policy
/// (ATAC modalities are eager; everything else compares estimated
/// bitmap size against `encoded_csr_size`).
///
/// Returns whether a bitmap section was actually written so callers
/// can stamp provenance.
#[allow(clippy::too_many_arguments)]
fn build_and_write_bitmap_for_shard(
    writer: &mut ScxWriter,
    indptr: &[u64],
    indices: &[u32],
    row_start: u64,
    n_rows: u32,
    n_vars: u32,
    encoded_csr_size: usize,
    policy: BitmapPolicy,
    modality_type: ModalityType,
    modality_name: Option<&str>,
    sink: &mut WarningSink,
) -> Result<bool, ConvertError> {
    let outcome = maybe_build_bitmap_shard(
        indptr,
        indices,
        row_start,
        n_rows,
        n_vars,
        encoded_csr_size,
        policy,
        modality_type,
    );
    match outcome {
        None => Ok(false),
        Some(BitmapBuildOutcome::Skip { reason }) => {
            sink.emit(ConvertWarning::BitmapSkipped {
                modality: modality_name.map(String::from),
                reason,
            });
            Ok(false)
        }
        Some(BitmapBuildOutcome::Built(shard)) => {
            writer
                .write_bitmap_shard(&shard)
                .map_err(ConvertError::from)?;
            Ok(true)
        }
    }
}

/// Detection-bitmap generation policy.
///
/// Re-exported from [`scx_format::BitmapPolicy`] so callers that depend
/// on `scx-convert` (CLI, pyscx with hdf5) can name it without an
/// extra `scx_format` import. The actual definition lives in
/// `scx-format` so the CPU-only pyscx build (which doesn't pull in
/// `scx-convert`) can still drive bitmap generation from its in-memory
/// write path.
pub use scx_format::BitmapPolicy;

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions {
            shard_target_rows: 16384,
            codec: None,
            csc: false,
            csc_cols_per_shard: 5000,
            tool: "scx-cli".into(),
            memory_budget: None,
            stream: true,
            strict_uns: false,
            dense_zero_epsilon: 0.0,
            temp_dir: None,
            modalities: None,
            modality_types: Vec::new(),
            index_obs: Vec::new(),
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 1000,
            bitmap: BitmapPolicy::Off,
            reader_threads: None,
            writer_queue_depth: 4,
        }
    }
}

/// Resolve [`ConvertOptions::reader_threads`] to a concrete
/// worker count. `None` (auto) reads `RAYON_NUM_THREADS` if set, else
/// falls back to [`std::thread::available_parallelism`].
pub(crate) fn resolve_reader_threads(opts: &ConvertOptions) -> usize {
    if let Some(n) = opts.reader_threads {
        return n.max(1);
    }
    if let Ok(s) = std::env::var("RAYON_NUM_THREADS") {
        if let Ok(n) = s.parse::<usize>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Build and write obs/var predicate indexes from the
/// currently configured conversion options, then return the list of
/// columns that ended up indexed so the caller can stamp provenance.
///
/// Three sources of column names are combined:
///   1. `opts.index_obs` / `opts.index_var` (forced; missing/unsupported
///      columns produce a hard `ConvertError`),
///   2. `opts.index_preset` (skipped + warned via the sink on
///      missing/unsupported columns),
///   3. auto-detection on cardinality `< opts.index_auto_threshold` when
///      neither forced nor preset columns are supplied.
///
/// `csr_row_ranges` must reflect the actual on-disk shard boundaries
/// produced by the writer (the engine uses local row indices within
/// each shard, so any drift between assumed and actual ranges produces
/// silently wrong pruning).
///
/// All of the build orchestration (preset resolution, encoding, writer
/// calls) lives in
/// [`scx_engine::build_and_write_conversion_predicate_indexes`]; this
/// wrapper only maps the engine's typed outcomes into `ConvertError` /
/// `ConvertWarning::{MissingPresetIndexColumn, UnsupportedIndexColumn}`.
/// `pyscx::anndata::build_and_write_predicate_indexes_inline` is the
/// Python-side mirror — keep their outcome handling shapes in sync.
fn build_and_write_predicate_indexes(
    writer: &mut ScxWriter,
    obs: &RecordBatch,
    var: &RecordBatch,
    csr_row_ranges: &[(u64, u64)],
    n_vars: usize,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(Vec<String>, Vec<String>), ConvertError> {
    let engine_opts = ConversionPredicateIndexOptions {
        index_obs: opts.index_obs.clone(),
        index_var: opts.index_var.clone(),
        index_preset: opts.index_preset.clone(),
        index_auto_threshold: opts.index_auto_threshold,
    };
    let result = build_and_write_conversion_predicate_indexes(
        writer,
        obs,
        var,
        csr_row_ranges,
        n_vars,
        &engine_opts,
    )
    .map_err(|e| ConvertError::Other(format!("build predicate index: {e}")))?;

    process_predicate_index_outcomes(result.obs_outcomes, "obs", sink)?;
    process_predicate_index_outcomes(result.var_outcomes, "var", sink)?;

    Ok((result.obs_indexed_columns, result.var_indexed_columns))
}

/// Demote per-column outcomes from
/// `scx_engine::build_and_write_conversion_predicate_indexes` into the
/// convert layer's policy: forced errors abort the convert; preset
/// skips emit a typed warning whose variant is chosen by the
/// `SkipReason` discriminant.
fn process_predicate_index_outcomes(
    outcomes: Vec<BuildOutcome>,
    axis: &str,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    for outcome in outcomes {
        match outcome {
            BuildOutcome::ForcedColumnError { column, reason } => {
                return Err(ConvertError::Other(format!(
                    "forced {axis} index column '{column}': {reason}"
                )));
            }
            BuildOutcome::PresetSkipped { column, reason } => match reason {
                SkipReason::MissingColumn => {
                    sink.emit(ConvertWarning::MissingPresetIndexColumn { column });
                }
                other => sink.emit(ConvertWarning::UnsupportedIndexColumn {
                    column,
                    reason: other.to_string(),
                }),
            },
        }
    }
    Ok(())
}

pub fn h5ad_to_scx(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
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
    let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };

    // Build header
    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: n_obs as u64,
        n_vars: n_vars as u64,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: opts.shard_target_rows,
        codec_id: codec_id as u8,
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

    // Write obs/var
    let obs = read_dataframe_group(&file, "obs")?;
    let var = read_dataframe_group(&file, "var")?;
    writer.write_obs(&obs)?;
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
        sink,
    )?;

    // Optional CSC sidecar — streaming transpose over the in-memory
    // CSR data, one shard per chunk.
    if opts.csc {
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
        )?;
    }

    // Write optional sections
    if let Ok(obsm_map) = read_obsm(&file) {
        for (name, batch) in &obsm_map {
            writer.write_obsm(name, batch)?;
        }
    }

    // Read /uns only when the group exists; key-level failures route
    // through the sink (lenient) or propagate (strict).
    if file.group("uns").is_ok() {
        let uns = read_uns(&file, opts.strict_uns, sink)?;
        writer.write_uns(&uns)?;
    }

    if let Ok(layers) = read_layers(&file) {
        for (layer_name, (l_indptr, l_indices, l_data, l_nobs, l_nvars)) in &layers {
            let (l_enc, l_codec) =
                detect_value_encoding(l_data, opts.codec).map_err(ScxError::from)?;
            let l_index_dtype: u8 = if *l_nvars <= 65535 { 0 } else { 1 };
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
    opts: &ConvertOptions,
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

    let tenx = read_tenx_h5(&file)?;
    let nnz = *tenx.indptr.last().unwrap_or(&0) as u64;
    let (value_encoding, codec_id) =
        detect_value_encoding(&tenx.data, opts.codec).map_err(ScxError::from)?;
    let index_dtype: u8 = if tenx.n_genes <= 65535 { 0 } else { 1 };

    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: tenx.n_cells as u64,
        n_vars: tenx.n_genes as u64,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: opts.shard_target_rows,
        codec_id: codec_id as u8,
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
    writer.write_obs(&tenx.obs)?;
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
        sink,
    )?;

    // Optional CSC sidecar — same streaming transpose as h5ad.
    if opts.csc {
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
    let _ = sink; // Phase 8 reverse-conversion will emit through this sink
                  // (e.g. unsupported uns shape, dropped predicate indexes);
                  // Phase 0 only threads the parameter.
    write_scx_to_h5ad(scx_path, h5ad_path)
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
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    crate::h5ad_stream_write::write_scx_to_h5ad_streaming(scx_path, h5ad_path, opts, sink)
}

/// Override hooks for [`h5ad_to_scx_streaming`]. Each `Some(...)`
/// field skips the corresponding on-disk read and uses the provided
/// value instead.
///
/// The pyscx backed-AnnData routing path in `from_anndata` uses this
/// to preserve in-Python mutations to `obs` / `var` / `uns` / `obsm` /
/// `varm` / `obsp` / `varp` that would otherwise be silently lost
/// when the streaming pipeline re-reads them from disk.
///
/// Layers are intentionally not overridable — they're streamed
/// directly from disk per shard, and the pyscx backed-mode path
/// emits a `UserWarning` if the in-memory AnnData has layers (where
/// any in-memory mutations would be dropped).
#[derive(Default)]
pub struct StreamingOverrides {
    pub obs: Option<arrow::record_batch::RecordBatch>,
    pub var: Option<arrow::record_batch::RecordBatch>,
    pub uns: Option<serde_json::Value>,
    pub obsm: Option<Vec<(String, arrow::record_batch::RecordBatch)>>,
    pub varm: Option<Vec<(String, arrow::record_batch::RecordBatch)>>,
    pub obsp: Option<Vec<(String, arrow::record_batch::RecordBatch)>>,
    pub varp: Option<Vec<(String, arrow::record_batch::RecordBatch)>>,
}

/// Streaming h5ad → SCX conversion. Reads the input one shard's worth
/// of rows at a time via [`super::h5ad_stream::XStreamReader`] so peak
/// memory is bounded by `shard_target_rows × n_vars × density × ~16
/// bytes` plus the always-resident indptr (`(n_obs + 1) × 8 bytes`).
///
/// The pipeline is currently sequential: one shard read → sort →
/// drop-zeros → encode → write per iteration. A concurrent encoder
/// pool is a planned follow-on if benchmarks show I/O starvation.
///
/// `opts.csc == true` runs a post-`finish()`
/// [`scx_ops::rebuild_csc_inplace`] pass over the just-written file;
/// peak disk briefly reaches ~2× the output size during the rebuild.
///
/// `varm` is read and written; `obsp` / `varp` are silently skipped
/// unless caller-supplied via [`StreamingOverrides`] (same gap the
/// non-streaming CLI converter has). CSC-on-disk and dense `X` are
/// rejected up front with [`ConvertError::StreamingUnsupported`].
pub fn h5ad_to_scx_streaming(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
    overrides: &StreamingOverrides,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let file = hdf5::File::open(input)?;

    // Format gating. `open_x_streaming` re-checks CSC/dense and the
    // encoding-type attribute; this branch only catches the 10x case
    // (which has no `X` group at all).
    let input_format = detect_input_format(&file)?;
    if matches!(input_format, InputFormat::TenX) {
        return Err(ConvertError::FormatMismatch {
            expected: "h5ad".to_string(),
            got: "10x".to_string(),
        });
    }
    let matrix_format = detect_matrix_format(&file, sink)?;

    // Open the X reader first — it surfaces shape via `n_obs` /
    // `n_vars`, both of which the file header needs before any section
    // write. CSR uses the indptr-eager reader; Dense slabs rows on
    // demand; CSC routes through the Phase 2 dispatcher which picks
    // in-memory vs. external-memory transpose based on the budget.
    let mut x_reader: Box<dyn CsrShardStream> = match matrix_format {
        MatrixFormat::Csr => Box::new(open_x_streaming(&file, "X", matrix_format, sink)?),
        MatrixFormat::Dense => Box::new(open_dense_streaming(&file, "X", opts, sink)?),
        MatrixFormat::Csc => open_csc_streaming(&file, "X", opts, sink)?,
    };
    let n_obs = x_reader.n_obs() as usize;
    let n_vars = x_reader.n_vars() as usize;
    let n_vars_u32: u32 = u32::try_from(n_vars)
        .map_err(|_| ConvertError::Other(format!("n_vars {n_vars} exceeds u32::MAX")))?;
    let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };

    // Placeholder header. `nnz`, `n_csr_shards`, `n_csc_shards`, and
    // `codec_id` are overwritten by `ScxWriter::finish()` from
    // running accumulators (see scx-format/src/writer.rs).
    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: n_obs as u64,
        n_vars: n_vars as u64,
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

    // obs / var. Override-or-disk per section: any `Some(...)` field
    // wins over the on-disk read so the backed-AnnData routing path
    // can preserve in-memory mutations.
    let obs = match overrides.obs.as_ref() {
        Some(batch) => batch.clone(),
        None => read_dataframe_group(&file, "obs")?,
    };
    let var = match overrides.var.as_ref() {
        Some(batch) => batch.clone(),
        None => read_dataframe_group(&file, "var")?,
    };
    writer.write_obs(&obs)?;
    writer.write_var(&var)?;

    // X shards (streaming). `run_streaming_writer_coordinator`
    // dispatches to the parallel encoder pool when the reader is
    // indexable + libhdf5 is thread-safe + the resolved thread count
    // is > 1; otherwise it falls back to the sequential path. Output
    // is byte-identical regardless of route.
    let (_csr_shard_count, csr_row_ranges) = run_streaming_writer_coordinator(
        x_reader.as_mut(),
        &mut writer,
        opts,
        index_dtype,
        n_vars_u32,
        SectionType::CsrShard,
        ModalityType::Rna,
        "x_shard",
        sink,
    )?;
    drop(x_reader);

    // obsm / varm / uns. Small dense sections — non-streaming reads,
    // override-or-disk per section. obsp / varp have no on-disk
    // readers yet, so they're only written when an override supplies
    // them (matches the non-streaming converter's gap for now).
    match overrides.obsm.as_ref() {
        Some(entries) => {
            for (name, batch) in entries {
                writer.write_obsm(name, batch)?;
            }
        }
        None => {
            if let Ok(obsm_map) = read_obsm(&file) {
                for (name, batch) in &obsm_map {
                    writer.write_obsm(name, batch)?;
                }
            }
        }
    }
    match overrides.varm.as_ref() {
        Some(entries) => {
            for (name, batch) in entries {
                writer.write_varm(name, batch)?;
            }
        }
        None => {
            if let Ok(varm_map) = read_varm(&file) {
                for (name, batch) in &varm_map {
                    writer.write_varm(name, batch)?;
                }
            }
        }
    }
    if let Some(entries) = overrides.obsp.as_ref() {
        for (name, batch) in entries {
            writer.write_obsp(name, batch)?;
        }
    }
    if let Some(entries) = overrides.varp.as_ref() {
        for (name, batch) in entries {
            writer.write_varp(name, batch)?;
        }
    }
    match overrides.uns.as_ref() {
        Some(json) => writer.write_uns(json)?,
        None => {
            if file.group("uns").is_ok() {
                let uns = read_uns(&file, opts.strict_uns, sink)?;
                writer.write_uns(&uns)?;
            }
        }
    }

    // Layers (one streaming pass per layer). Best-effort per layer:
    // an open failure (dense/CSC layer, malformed encoding, shape
    // mismatch with X) is logged and skipped, mirroring the
    // non-streaming `read_layers` warn-and-continue behaviour
    // (scx-convert/src/h5ad_read.rs). Once a layer's shards start
    // writing, a mid-stream shard error aborts — leaving a
    // half-written layer in the SCX file would be worse than failing
    // loudly. Width-dependent encoding values are recomputed per
    // layer instead of inheriting `X`'s.
    if let Ok(layers_group) = file.group("layers") {
        let layer_names = layers_group.member_names()?;
        for layer_name in &layer_names {
            let layer_path = format!("layers/{layer_name}");
            let layer_format =
                match super::detect::detect_matrix_format_at(&file, &layer_path, sink) {
                    Ok(f) => f,
                    Err(e) => {
                        sink.emit(ConvertWarning::LayerSkipped {
                            name: layer_name.clone(),
                            reason: format!("{e}"),
                        });
                        continue;
                    }
                };
            let mut layer_reader: Box<dyn CsrShardStream> = match layer_format {
                MatrixFormat::Csr => match open_layer_streaming(&file, layer_name, sink) {
                    Ok(r) => Box::new(r),
                    Err(e) => {
                        sink.emit(ConvertWarning::LayerSkipped {
                            name: layer_name.clone(),
                            reason: format!("{e}"),
                        });
                        continue;
                    }
                },
                MatrixFormat::Dense => {
                    match open_dense_layer_streaming(&file, layer_name, opts, sink) {
                        Ok(r) => Box::new(r),
                        Err(e) => {
                            sink.emit(ConvertWarning::LayerSkipped {
                                name: layer_name.clone(),
                                reason: format!("{e}"),
                            });
                            continue;
                        }
                    }
                }
                MatrixFormat::Csc => {
                    match open_csc_layer_streaming(&file, layer_name, opts, sink) {
                        Ok(r) => r,
                        Err(e) => {
                            sink.emit(ConvertWarning::LayerSkipped {
                                name: layer_name.clone(),
                                reason: format!("{e}"),
                            });
                            continue;
                        }
                    }
                }
            };
            let l_n_obs = layer_reader.n_obs() as usize;
            if l_n_obs != n_obs {
                sink.emit(ConvertWarning::LayerSkipped {
                    name: layer_name.clone(),
                    reason: format!("n_obs {l_n_obs} does not match X n_obs {n_obs}"),
                });
                continue;
            }
            let l_n_vars = layer_reader.n_vars() as usize;
            let l_n_vars_u32 = match u32::try_from(l_n_vars) {
                Ok(v) => v,
                Err(_) => {
                    sink.emit(ConvertWarning::LayerSkipped {
                        name: layer_name.clone(),
                        reason: format!("n_vars {l_n_vars} exceeds u32::MAX"),
                    });
                    continue;
                }
            };
            let l_index_dtype: u8 = if l_n_vars <= 65535 { 0 } else { 1 };
            let (_, _) = run_streaming_writer_coordinator(
                layer_reader.as_mut(),
                &mut writer,
                opts,
                l_index_dtype,
                l_n_vars_u32,
                SectionType::LayerCsrShard,
                ModalityType::Rna,
                &format!("{layer_name}_shard"),
                sink,
            )?;
        }
    }

    // Phase 5a: predicate indexes from the obs/var we just wrote and
    // the actual shard boundaries reported by `streaming_writer_coordinator`.
    let (obs_indexed, var_indexed) = build_and_write_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &csr_row_ranges,
        n_vars,
        opts,
        sink,
    )?;

    // Provenance carries the streaming flag so consumers can tell at
    // a glance how the file was produced.
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let source_format_str = match matrix_format {
        MatrixFormat::Csr => "csr_matrix",
        MatrixFormat::Csc => "csc_matrix",
        MatrixFormat::Dense => "array",
    };
    let resolved_reader_threads = resolve_reader_threads(opts);
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: opts.tool.clone(),
        params_json: serde_json::json!({
            "input": input.display().to_string(),
            "format": "h5ad",
            "stream": true,
            "source_matrix_format": source_format_str,
            "warnings": sink.summary_json(),
            "predicate_index": {
                "obs_columns": obs_indexed,
                "var_columns": var_indexed,
                "preset": opts.index_preset,
            },
            "reader_threads": resolved_reader_threads,
            "writer_queue_depth": opts.writer_queue_depth,
        })
        .to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;

    // CSC sidecar (opt-in). Two-pass: streaming write produces CSR
    // shards only; if requested, rebuild the CSC sidecar in place
    // over the just-finished file. Peak disk briefly reaches ~2×
    // output size for the duration of the rebuild (writes to a
    // sibling `.rebuild_csc.tmp` and renames).
    if opts.csc {
        scx_ops::rebuild_csc_inplace(output, opts.csc_cols_per_shard, "4G")
            .map_err(|e| ConvertError::Other(format!("rebuild_csc_inplace failed: {e}")))?;
    }

    Ok(())
}

/// Drain a [`CsrShardStream`] into an [`ScxWriter`] one shard at a
/// time, encoding each shard through [`encode_one_shard`].
///
/// Phase 0.2 seam. The initial implementation is sequential — one
/// shard read → drop-zeros → sort → encode → write per iteration —
/// matching the behaviour of the original inline loop in
/// `h5ad_to_scx_streaming`. Phase 8c will replace the body with a
/// bounded shard queue plus an ordered writer stage, but the
/// signature is the same: every entry point that drives a
/// `CsrShardStream` (X, layers, h5mu modalities, Zarr) goes through
/// this helper.
///
/// `section_name_prefix` is appended with `_{shard_idx}` to produce
/// the per-shard section name. Returns the number of shards
/// written, useful for callers that need to track per-source shard
/// counts (e.g. the layer loop).
#[allow(clippy::too_many_arguments)]
pub fn streaming_writer_coordinator(
    reader: &mut dyn CsrShardStream,
    writer: &mut ScxWriter,
    opts: &ConvertOptions,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    section_name_prefix: &str,
    sink: &mut WarningSink,
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    let target_rows = opts.shard_target_rows as usize;
    let mut shard_idx: u32 = 0;
    let mut row_ranges: Vec<(u64, u64)> = Vec::new();
    while let Some(mut shard) = reader.next_csr_shard(target_rows)? {
        // Surface upstream duplicate-coordinate canonicalisation
        // (Phase 2 CSC external transpose) as a typed warning. Other
        // readers always set `duplicates_merged = 0` so this is a
        // no-op for them.
        if shard.duplicates_merged > 0 {
            sink.emit(ConvertWarning::DuplicateCoordinatesMerged {
                count: shard.duplicates_merged,
                policy: "sum".to_string(),
            });
        }
        drop_explicit_zeros_inplace(&mut shard.indptr, &mut shard.indices, &mut shard.values);
        sort_csr_rows_in_place(&shard.indptr, &mut shard.indices, &mut shard.values);
        let row_start = shard.row_start;
        let n_rows = shard.n_rows as u64;
        let pre = encode_one_shard(
            &shard.indptr,
            &shard.indices,
            &shard.values,
            opts.codec,
            index_dtype,
            n_vars_u32,
            row_start,
            section_type,
            modality_type,
            format!("{section_name_prefix}_{shard_idx}"),
        )?;
        let encoded_csr_size = pre.section_length as usize;
        writer.write_preencoded_shard(pre)?;
        // Detection bitmap (only for primary X shards; layer
        // shards are skipped — bitmaps are per X-axis presence today).
        if section_type == SectionType::CsrShard {
            build_and_write_bitmap_for_shard(
                writer,
                &shard.indptr,
                &shard.indices,
                row_start,
                shard.n_rows,
                n_vars_u32,
                encoded_csr_size,
                opts.bitmap,
                modality_type,
                None,
                sink,
            )?;
        }
        row_ranges.push((row_start, row_start + n_rows));
        shard_idx += 1;
    }
    Ok((shard_idx, row_ranges))
}

/// Split the matrix into `[(row_start, n_rows)]` shard
/// ranges. The sequential coordinator's `while next_csr_shard(...)`
/// loop produces the same partition implicitly; the parallel
/// coordinator hoists it so workers can claim ranges without
/// coordination.
pub(crate) fn compute_shard_row_ranges(n_obs: u64, target_rows: u32) -> Vec<(u64, u32)> {
    if target_rows == 0 || n_obs == 0 {
        return Vec::new();
    }
    let target = target_rows as u64;
    let cap = (n_obs / target + 1) as usize;
    let mut out = Vec::with_capacity(cap);
    let mut row = 0u64;
    while row < n_obs {
        let n = (n_obs - row).min(target) as u32;
        out.push((row, n));
        row += n as u64;
    }
    out
}

/// Payload from a worker thread to the ordered writer.
struct EncodedShardOutput {
    pre: PreEncodedSection,
    /// For sequential `BitmapShard` write after the CSR shard.
    /// Built by the worker; the writer just calls `write_bitmap_shard`.
    bitmap: Option<BitmapBuildOutcome>,
    duplicates_merged: u64,
}

/// Parallel sibling of [`streaming_writer_coordinator`].
///
/// Partitions the reader into `[(row_start, n_rows)]` ranges, fans
/// them out across a rayon worker pool, and writes encoded shards in
/// shard-index order via a bounded reorder buffer. Output is
/// byte-identical to the sequential coordinator.
///
/// Requires `reader.as_indexed()` to return `Some(_)`; callers must
/// verify that and the libhdf5 thread-safety check before routing
/// here. The top-level [`run_streaming_writer_coordinator`] handles
/// the fallback to sequential when those preconditions fail.
#[allow(clippy::too_many_arguments)]
fn streaming_writer_coordinator_parallel(
    reader: &dyn crate::stream::IndexedCsrShardStream,
    writer: &mut ScxWriter,
    opts: &ConvertOptions,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    section_name_prefix: &str,
    sink: &mut WarningSink,
    reader_threads: usize,
    queue_depth: usize,
    effective_target_rows: u32,
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    use crossbeam_channel::bounded;
    use rayon::ThreadPoolBuilder;

    let n_obs = reader.n_obs();
    let ranges = compute_shard_row_ranges(n_obs, effective_target_rows);
    if ranges.is_empty() {
        return Ok((0, Vec::new()));
    }

    let row_ranges: Vec<(u64, u64)> = ranges
        .iter()
        .map(|&(rs, nr)| (rs, rs + nr as u64))
        .collect();

    let n_ranges = ranges.len();
    let pool = ThreadPoolBuilder::new()
        .num_threads(reader_threads)
        .thread_name(|i| format!("scx-stream-{i}"))
        .build()
        .map_err(|e| {
            ConvertError::Other(format!(
                "failed to build rayon pool with {reader_threads} threads: {e}"
            ))
        })?;

    let queue_depth = queue_depth.max(1);
    let (tx, rx) = bounded::<(usize, Result<EncodedShardOutput, ConvertError>)>(queue_depth);

    let source_name: String = reader.source_matrix_name().to_string();
    let opts_codec = opts.codec;
    let opts_bitmap = opts.bitmap;
    let want_bitmap = section_type == SectionType::CsrShard;
    let name_prefix = section_name_prefix.to_string();

    // Cap outstanding shards (encoding + in channel + in BTreeMap) at
    // `reader_threads + queue_depth`. Rolling-window spawn: prime the
    // pool with `in_flight_cap` tasks, then spawn one new task each
    // time a shard is received. This bounds the reorder buffer; the
    // previous up-front spawn loop let the BTreeMap grow to ~n_ranges
    // when shard 0 was slow (Gemini code review feedback).
    let in_flight_cap = reader_threads.saturating_add(queue_depth);

    // Serialize parallel coordinator runs across the test binary so
    // the cfg(test) in-flight counter is observable race-free. Held
    // for the entire parallel scope; no effect in production.
    #[cfg(test)]
    let _serial = {
        let lock = test_hooks::SERIALIZE
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        test_hooks::reset_in_flight();
        lock
    };

    // Capture the fault-injection setting on the calling thread (the
    // caller's `FailIngestShardGuard` lives in this thread's
    // thread-local). The captured `Option<usize>` is `Copy` and
    // propagates into rayon workers via the spawn closure capture, so
    // each coordinator invocation carries its own fault config —
    // concurrent non-fault tests on other threads see `None`.
    #[cfg(test)]
    let captured_fault_shard = test_hooks::current_ingest_fault_shard();

    // Macro-style local spawn: must inline because extracting a
    // closure would re-borrow `reader` from a nested closure scope
    // and rayon's `'scope` lifetime can't be reconciled with that
    // shape. Each spawn clones `tx` + `source_name` for the worker.
    //
    // `move` is load-bearing: it moves `rx` into the closure so an
    // early `return Err(...)` from the drain loop drops `rx` on
    // unwind, unblocking workers parked in `tx.send(...)` on the
    // bounded channel. Without `move` `rx` lives in the parent frame
    // and the scope can never join those workers.
    pool.in_place_scope(move |s| -> Result<(), ConvertError> {
        macro_rules! spawn_shard {
            ($scope:expr, $idx:expr) => {{
                let idx_ = $idx;
                let (row_start, n_rows) = ranges[idx_];
                let tx = tx.clone();
                let source_name = source_name.clone();
                let name = format!("{name_prefix}_{idx_}");
                $scope.spawn(move |_| {
                    #[cfg(test)]
                    let _guard = crate::pipeline::test_hooks::InFlightGuard::new();
                    #[cfg(test)]
                    if Some(idx_) == captured_fault_shard {
                        let inner = ConvertError::Other(format!(
                            "test_hooks: injected failure at shard {idx_}"
                        ));
                        let wrapped = ConvertError::ShardRead {
                            row_start,
                            n_rows,
                            source: source_name,
                            inner: Box::new(inner),
                        };
                        let _ = tx.send((idx_, Err(wrapped)));
                        return;
                    }
                    let result = encode_one_shard_worker(
                        reader,
                        row_start,
                        n_rows,
                        opts_codec,
                        index_dtype,
                        n_vars_u32,
                        section_type,
                        modality_type,
                        name,
                        opts_bitmap,
                        want_bitmap,
                    );
                    let wrapped = result.map_err(|inner| ConvertError::ShardRead {
                        row_start,
                        n_rows,
                        source: source_name,
                        inner: Box::new(inner),
                    });
                    let _ = tx.send((idx_, wrapped));
                });
            }};
        }

        // Prime the pump with up to `in_flight_cap` tasks.
        let mut next_to_spawn: usize = 0;
        let prime = in_flight_cap.min(n_ranges);
        while next_to_spawn < prime {
            spawn_shard!(s, next_to_spawn);
            next_to_spawn += 1;
        }

        // Drain in the calling thread, reordering by shard_idx and
        // spawning one new task per received shard. The BTreeMap can
        // hold at most `in_flight_cap - 1` out-of-order shards.
        let mut buffer: std::collections::BTreeMap<usize, EncodedShardOutput> =
            std::collections::BTreeMap::new();
        let mut next_idx: usize = 0;
        let mut received: usize = 0;
        while received < n_ranges {
            let (idx, r) = rx.recv().map_err(|_| {
                ConvertError::Other(
                    "parallel streaming worker channel closed before all shards arrived".into(),
                )
            })?;
            received += 1;
            if next_to_spawn < n_ranges {
                spawn_shard!(s, next_to_spawn);
                next_to_spawn += 1;
            }
            match r {
                Err(e) => return Err(e),
                Ok(out) => {
                    buffer.insert(idx, out);
                }
            }
            while let Some(out) = buffer.remove(&next_idx) {
                if out.duplicates_merged > 0 {
                    sink.emit(ConvertWarning::DuplicateCoordinatesMerged {
                        count: out.duplicates_merged,
                        policy: "sum".to_string(),
                    });
                }
                let bitmap = out.bitmap;
                writer.write_preencoded_shard(out.pre)?;
                if want_bitmap {
                    match bitmap {
                        None => { /* policy was Off — nothing to do */ }
                        Some(BitmapBuildOutcome::Skip { reason }) => {
                            sink.emit(ConvertWarning::BitmapSkipped {
                                modality: None,
                                reason,
                            });
                        }
                        Some(BitmapBuildOutcome::Built(shard)) => {
                            writer
                                .write_bitmap_shard(&shard)
                                .map_err(ConvertError::from)?;
                        }
                    }
                }
                next_idx += 1;
            }
        }
        Ok(())
    })?;

    // Capture the in-flight peak into the calling thread's
    // thread-local *before* releasing `_serial`, so tests reading
    // `LAST_RUN_PEAK` after `h5ad_to_scx_streaming` returns see the
    // peak from this run without interference from any subsequent
    // parallel coordinator invocation. No-op in production.
    #[cfg(test)]
    {
        let peak = test_hooks::IN_FLIGHT_PEAK.load(std::sync::atomic::Ordering::SeqCst);
        test_hooks::set_last_run_peak(peak);
        drop(_serial);
    }

    Ok((n_ranges as u32, row_ranges))
}

/// Worker body: read + canonicalise + encode one shard and (if
/// requested) build the bitmap for it. Pure with respect to the
/// writer — output is funnelled back through a channel to the
/// ordered writer thread.
#[allow(clippy::too_many_arguments)]
fn encode_one_shard_worker(
    reader: &dyn crate::stream::IndexedCsrShardStream,
    row_start: u64,
    n_rows: u32,
    codec: Option<CodecId>,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    name: String,
    bitmap_policy: BitmapPolicy,
    want_bitmap: bool,
) -> Result<EncodedShardOutput, ConvertError> {
    let mut shard = reader.read_range(row_start, n_rows)?;
    let duplicates_merged = shard.duplicates_merged;
    drop_explicit_zeros_inplace(&mut shard.indptr, &mut shard.indices, &mut shard.values);
    sort_csr_rows_in_place(&shard.indptr, &mut shard.indices, &mut shard.values);

    let pre = encode_one_shard(
        &shard.indptr,
        &shard.indices,
        &shard.values,
        codec,
        index_dtype,
        n_vars_u32,
        shard.row_start,
        section_type,
        modality_type,
        name,
    )?;
    let encoded_csr_size = pre.section_length as usize;

    let bitmap = if want_bitmap {
        maybe_build_bitmap_shard(
            &shard.indptr,
            &shard.indices,
            shard.row_start,
            shard.n_rows,
            n_vars_u32,
            encoded_csr_size,
            bitmap_policy,
            modality_type,
        )
    } else {
        None
    };

    Ok(EncodedShardOutput {
        pre,
        bitmap,
        duplicates_merged,
    })
}

/// Top-level dispatcher: pick parallel vs sequential
/// coordinator based on `opts.reader_threads`, libhdf5 thread-safety,
/// reader trait support, and memory-budget derate.
///
/// Always returns `(shard_count, row_ranges)` matching the sequential
/// coordinator's contract. Output is byte-identical regardless of
/// path.
#[allow(clippy::too_many_arguments)]
pub fn run_streaming_writer_coordinator(
    reader: &mut dyn CsrShardStream,
    writer: &mut ScxWriter,
    opts: &ConvertOptions,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    section_name_prefix: &str,
    sink: &mut WarningSink,
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    let requested = resolve_reader_threads(opts);
    if requested <= 1 {
        return streaming_writer_coordinator(
            reader,
            writer,
            opts,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            section_name_prefix,
            sink,
        );
    }

    // Reader must expose row-range reads.
    let Some(indexed) = reader.as_indexed() else {
        return streaming_writer_coordinator(
            reader,
            writer,
            opts,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            section_name_prefix,
            sink,
        );
    };

    // libhdf5 must be threadsafe (only when we hit the HDF5-backed
    // readers — for in-memory MaterializedCsrStream the check is
    // harmless but we still gate to keep behaviour uniform).
    if !crate::hdf5_threadsafe::hdf5_is_threadsafe() {
        crate::hdf5_threadsafe::try_emit_not_threadsafe_warning(sink);
        return streaming_writer_coordinator(
            reader,
            writer,
            opts,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            section_name_prefix,
            sink,
        );
    }

    // Clamp the partition's shard size by the reader's hard slab cap.
    // Only `DenseXStreamReader` returns `Some(_)` today (when
    // `memory_budget` shrinks `max_slab_rows` below
    // `shard_target_rows`). Sequential `DenseXStreamReader::
    // next_csr_shard` already clamps the same way, so byte-identity
    // with the sequential path holds.
    let effective_target = indexed
        .max_slab_rows()
        .map_or(opts.shard_target_rows, |cap| {
            opts.shard_target_rows.min(cap)
        });

    // Memory-budget derate: cap workers so per-worker working set
    // fits under `memory_budget`. Estimate is delegated to the reader
    // via `IndexedCsrShardStream::per_worker_bytes`: the default impl
    // assumes the sparsified output is the binding bound and picks
    // density by modality (RNA/default 5 %, ATAC 10 %); the dense
    // reader override sizes the dense slab buffer instead. Pass
    // `effective_target` so the estimate matches the shard size we
    // are actually about to drive workers at.
    let per_worker_bytes = indexed
        .per_worker_bytes(effective_target, modality_type)
        .max(1);
    let granted = match opts.memory_budget {
        Some(b) if per_worker_bytes > b => {
            return Err(ConvertError::Other(format!(
                "single parallel-streaming shard requires \u{2248} {per_worker_bytes} bytes \
                 but memory_budget is {b}; lower --shard-size or raise --memory-budget"
            )));
        }
        Some(b) => {
            let max_workers = (b / per_worker_bytes).max(1) as usize;
            max_workers.min(requested)
        }
        None => requested,
    };
    if granted < requested {
        sink.emit(ConvertWarning::ReaderThreadsDerated {
            requested,
            granted,
            reason: format!(
                "memory_budget caps per-worker working set to {per_worker_bytes} bytes"
            ),
        });
    }

    if granted <= 1 {
        return streaming_writer_coordinator(
            reader,
            writer,
            opts,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            section_name_prefix,
            sink,
        );
    }

    streaming_writer_coordinator_parallel(
        indexed,
        writer,
        opts,
        index_dtype,
        n_vars_u32,
        section_type,
        modality_type,
        section_name_prefix,
        sink,
        granted,
        opts.writer_queue_depth.max(1),
        effective_target,
    )
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
pub(crate) mod test_hooks {
    //! Test-only instrumentation for the parallel coordinator.
    //!
    //! `IN_FLIGHT_NOW` tracks worker tasks currently executing (one
    //! entry per running `encode_one_shard_worker`); `IN_FLIGHT_PEAK`
    //! is the running maximum across the most recent run. The
    //! coordinator acquires `SERIALIZE` for the duration of the
    //! parallel scope to ensure exactly one parallel coordinator run
    //! is in flight at a time across all tests in the binary, then
    //! captures `IN_FLIGHT_PEAK` into the calling thread's
    //! `LAST_RUN_PEAK` before releasing the lock. Tests read
    //! `LAST_RUN_PEAK` after the streaming call returns; the
    //! thread-local pin makes the read race-free without requiring
    //! tests themselves to hold the global lock.
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    pub static IN_FLIGHT_NOW: AtomicUsize = AtomicUsize::new(0);
    pub static IN_FLIGHT_PEAK: AtomicUsize = AtomicUsize::new(0);
    pub static SERIALIZE: Mutex<()> = Mutex::new(());

    thread_local! {
        /// Per-thread fault-injection switch read by the parallel
        /// ingest coordinator (`streaming_writer_coordinator_parallel`)
        /// at entry. Workers fail synthetically on the configured shard
        /// index instead of running the real shard worker. Used by the
        /// deadlock regression test to force a mid-stream worker failure
        /// without corrupting the source h5ad on disk.
        ///
        /// Thread-local (not a global atomic) so a concurrently
        /// scheduled non-fault test running on a different thread never
        /// observes another test's fault config. The coordinator copies
        /// the value at entry on its calling thread and propagates it
        /// to the spawned rayon workers; each coordinator invocation
        /// captures its own value.
        ///
        /// Mutate only via [`FailIngestShardGuard`]; the guard restores
        /// the previous value on drop so the hook can't leak across
        /// tests, even on panic mid-flight.
        pub static FAIL_INGEST_SHARD_AT: Cell<Option<usize>> = const { Cell::new(None) };
    }

    /// Read the current thread's ingest fault-injection setting.
    /// Coordinator captures this at entry on its calling thread, then
    /// propagates to spawned workers via the closure capture.
    pub fn current_ingest_fault_shard() -> Option<usize> {
        FAIL_INGEST_SHARD_AT.with(|c| c.get())
    }

    /// RAII guard that arms the ingest fault injector for the
    /// current thread. Drop restores the previous value — keeps the
    /// hook from leaking when the test panics mid-flight. The fault is
    /// thread-scoped, so tests that spawn the convert call onto a
    /// helper thread must create the guard *on that helper thread*
    /// (typically inside the `std::thread::spawn` closure body).
    pub struct FailIngestShardGuard {
        prev: Option<usize>,
    }

    impl FailIngestShardGuard {
        pub fn new(shard_idx: usize) -> Self {
            let prev = FAIL_INGEST_SHARD_AT.with(|c| c.replace(Some(shard_idx)));
            Self { prev }
        }
    }

    impl Drop for FailIngestShardGuard {
        fn drop(&mut self) {
            let prev = self.prev;
            FAIL_INGEST_SHARD_AT.with(|c| c.set(prev));
        }
    }

    thread_local! {
        pub static LAST_RUN_PEAK: Cell<usize> = const { Cell::new(0) };
    }

    pub fn reset_in_flight() {
        IN_FLIGHT_NOW.store(0, Ordering::SeqCst);
        IN_FLIGHT_PEAK.store(0, Ordering::SeqCst);
    }

    pub fn last_run_peak() -> usize {
        LAST_RUN_PEAK.with(|c| c.get())
    }

    pub fn set_last_run_peak(v: usize) {
        LAST_RUN_PEAK.with(|c| c.set(v));
    }

    pub struct InFlightGuard;

    impl InFlightGuard {
        pub fn new() -> Self {
            let now = IN_FLIGHT_NOW.fetch_add(1, Ordering::SeqCst) + 1;
            IN_FLIGHT_PEAK.fetch_max(now, Ordering::SeqCst);
            Self
        }
    }

    impl Drop for InFlightGuard {
        fn drop(&mut self) {
            IN_FLIGHT_NOW.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// Memory budget for the streaming CSR→CSC transpose at convert time.
///
/// 4 GiB matches the `scx build-csc` default. The convert pipeline
/// already holds the full CSR matrix in RAM, so this only bounds
/// the per-chunk transpose working set. Large enough for typical
/// inputs; the user-facing knob is `csc_cols_per_shard`.
const CONVERT_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Streaming CSR → CSC transpose over the in-memory matrix, with
/// the result written shard-by-shard via `writer.write_csc_shard`.
///
/// Each emitted shard's column count is bounded by
/// `csc_cols_per_shard` (or the memory budget, whichever is
/// smaller). The CSR data already lives in `(indptr, indices, data)`
/// at this point in the pipeline — passed straight to the streaming
/// iterator without re-reading from disk.
#[allow(clippy::too_many_arguments)]
fn write_csc_shards_from_csr(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    csc_cols_per_shard: usize,
) -> Result<(), ConvertError> {
    // Wrap the in-memory CSR as a single ScxCsr "shard" for the
    // transpose iterator. Use the unchecked constructor — these
    // arrays were just produced by validated readers, no need to
    // re-validate.
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

        writer.write_csc_shard(
            &csc_indptr_u64,
            &csc_indices_u32,
            &raw_values,
            codec_id,
            value_encoding,
            col_start,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_csr_shards(
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
    sink: &mut WarningSink,
) -> Result<Vec<(u64, u64)>, ConvertError> {
    let n_vars_u32 = u32::try_from(n_vars)
        .map_err(|_| ConvertError::Other(format!("n_vars {n_vars} exceeds u32::MAX")))?;

    let mut row_ranges: Vec<(u64, u64)> = Vec::new();
    let mut row_start: usize = 0;
    let mut shard_idx: u32 = 0;
    while row_start < n_obs {
        let row_end = (row_start + shard_target_rows).min(n_obs);

        // Slice indptr for this shard
        let shard_indptr_slice = &indptr[row_start..=row_end];
        let base = shard_indptr_slice[0];
        // Validate indptr values are non-negative and >= base (finding 8.6).
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

        // Slice indices and data
        let nnz_start = usize::try_from(base)
            .map_err(|_| ConvertError::Other(format!("negative indptr base {base}")))?;
        let nnz_end = usize::try_from(*shard_indptr_slice.last().unwrap()).map_err(|_| {
            ConvertError::Other(format!(
                "negative indptr value {}",
                shard_indptr_slice.last().unwrap()
            ))
        })?;
        // Validate indices are non-negative before casting to u32 (finding 8.5).
        let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
            .iter()
            .map(|&v| {
                u32::try_from(v)
                    .map_err(|_| ConvertError::Other(format!("negative column index {v}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shard_data = &data[nnz_start..nnz_end];

        // Pre-encode so the bitmap auto-policy can compare against the
        // post-codec section length (matches streaming + python in-memory
        // paths). Section name `X_shard_{idx}` mirrors the name that
        // `ScxWriter::write_csr_shard` constructs internally.
        let pre = encode_one_shard(
            &shard_indptr,
            &shard_indices,
            shard_data,
            Some(codec_id),
            index_dtype,
            n_vars_u32,
            row_start as u64,
            SectionType::CsrShard,
            modality_type,
            format!("X_shard_{shard_idx}"),
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

#[allow(clippy::too_many_arguments)]
fn write_layer_shards(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    _n_vars: usize,
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

        let shard_indptr_slice = &indptr[row_start..=row_end];
        let base = shard_indptr_slice[0];
        // Validate indptr values are non-negative and >= base (finding 8.6).
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
        // Validate indices are non-negative before casting to u32 (finding 8.5).
        let shard_indices: Vec<u32> = indices[nnz_start..nnz_end]
            .iter()
            .map(|&v| {
                u32::try_from(v)
                    .map_err(|_| ConvertError::Other(format!("negative column index {v}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
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
