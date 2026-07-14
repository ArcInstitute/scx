use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::GroupPass;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::error::ScxError;
use scx_format_io::header::FileHeader;
use scx_format_io::modality::ModalityType;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::{PreEncodedSection, ScxWriter};
use scx_format_io::{encode_one_shard, FramingConfig};
use scx_sparse::canonicalize_csr;

use super::detect::{detect_input_format, detect_matrix_format, InputFormat, MatrixFormat};
use super::dtype::{detect_value_encoding, index_dtype_for, values_to_raw_bytes};
use super::stream::CsrShardStream;
use super::tenx_read::read_tenx_h5;
use super::warnings::{ConvertWarning, WarningSink};
use crate::h5ad::csc_stream::{open_csc_layer_streaming, open_csc_streaming};
use crate::h5ad::dense_stream::{open_dense_layer_streaming, open_dense_streaming};
use crate::h5ad::read::{
    list_dense_mapping_shapes, list_sparse_mapping_shapes, read_dataframe_group,
    read_dense_mapping_shard, read_layers, read_sparse_mapping_shard, read_uns, read_x_matrix,
};
use crate::h5ad::stream::{open_layer_streaming, open_x_streaming};
use crate::h5ad::write::write_scx_to_h5ad;
use arrow::record_batch::RecordBatch;
use scx_engine::{
    build_and_write_conversion_predicate_indexes, BuildOutcome, ConversionPredicateIndexOptions,
    SkipReason,
};
use scx_format_io::bitmap::BitmapShard;

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

#[derive(Clone)]
pub struct ConvertOptions {
    pub shard_target_rows: u32,
    /// Explicit codec override. None = auto-select based on value distribution.
    pub codec: Option<CodecId>,
    /// CSC-sidecar generation policy (`Off` / `Auto` / `Always`). When the
    /// policy resolves to build (always, or auto + dataset over the size
    /// thresholds), a multi-shard column-major CSC sidecar is emitted at
    /// write time. The CSR shards are still written first; CSC chunks are
    /// produced via streaming transpose over the in-memory CSR data.
    pub csc: CscPolicy,
    /// Columns per CSC shard when a CSC sidecar is emitted. `0` disables the
    /// cap (single CSC shard, memory permitting).
    pub csc_cols_per_shard: usize,
    /// Experimental (F5): row-group-frame each shard into groups of at most this
    /// many rows, emitting a v4 file / v2 shards with a multi-entry `BlockIndex`
    /// for codec-agnostic sub-shard random access. `None` (default) writes the
    /// ordinary unframed (v3) layout. Works for any codec (None/ShufDeltaZstd/
    /// Zstd/Lz4/Pcodec).
    pub row_group_rows: Option<u32>,
    /// Byte/nnz-aware group cap (F5 §4.3): additionally close a row-group once it
    /// reaches this many non-zeros. `None` = row-count-only grouping. Ignored
    /// unless `row_group_rows` is set.
    pub row_group_target_nnz: Option<u64>,
    /// Trial-encode (`--codec compact-trial`): per framed shard, keep the smaller
    /// of {heuristic codec, ShufDeltaZstd}. Ignored unless `row_group_rows` is set.
    pub codec_trial: bool,
    /// Internal adaptive-profile mechanism behind `codec="auto"`/`"compact"`
    /// (set by `scx_format::resolve_codec`, not a user-facing knob): per framed
    /// integer shard, pick the codec by [`scx_format_io::pick_codec_v2`] biased by
    /// this target — `Auto` (auto, cost-aware margin) or `Storage` (compact, tie
    /// -adopt). `None` = heuristic single-encode (`fast` / explicit codec). Takes
    /// precedence over `codec_trial`; ignored unless `row_group_rows` is set.
    pub decode_target: Option<scx_format_io::DecodeTarget>,
    /// Tool name recorded in the provenance entry. Defaults to
    /// `"scx"`; `pyscx` overrides this to `"pyscx"` so the
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
    /// Sort-on-convert: obs columns to globally
    /// reorder the cell axis by, lexicographic in order (leading key first).
    /// Empty (default) = no reorder. Requires a CSR or dense `/X`; CSC-on-disk
    /// X errors. The reorder is applied to X, layers, obs, and obsm; obsp is
    /// dropped with a warning (obsp remap is Phase 5).
    pub sort_by: Vec<String>,
    /// Descending sort when `sort_by` is set.
    pub sort_reverse: bool,
    /// Phase 7.4 convert-time grouping: obs column whose label clusters cells
    /// into contiguous, never-split shards (reference-first), writing a grouped
    /// layout directly during conversion (byte-equivalent to convert-then-`scx
    /// sort --group-by`). `None` (default) = no grouping. Implies an obs-axis
    /// reorder, so it requires a CSR or dense `/X` (CSC-on-disk errors) and a
    /// single-modality input; `sort_by`, when also set, supplies secondary sort
    /// keys after the group key.
    pub group_by: Option<String>,
    /// Which cells are reference (e.g. non-targeting controls); packed first and
    /// isolated in shard 0. Requires `group_by`. `None` = no reference shard.
    pub reference: Option<scx_ops::ReferenceSpec>,
    /// Target shard size in bytes for the group planner (group edges only). When
    /// set, a per-row nnz pre-scan sizes shards by encoded width instead of row
    /// count. CSR inputs only; dense/CSC fall back to row-count mode with a
    /// warning. Only meaningful with `group_by`.
    pub group_target_bytes: Option<u64>,
    /// Oversize threshold: a single group exceeding this becomes its own shard
    /// with a warning. Defaults to a multiple of `group_target_bytes`. Only
    /// meaningful with `group_by`.
    pub group_max_bytes: Option<u64>,
    /// How to realize convert-time grouping (only meaningful with `group_by`).
    /// `Auto` (default) routes by source density: a CSR source uses the
    /// one-pass streaming grouped gather (cheaper — reads only nnz per row); a
    /// dense source falls back to a two-pass plain-convert-then-`scx sort`
    /// (the grouped random-row gather over a dense matrix reads full rows and
    /// is ~4–5× slower / ~2× the memory). `One` / `Two` force the choice.
    pub group_pass: GroupPass,
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
/// Re-exported from [`scx_format_io::BitmapPolicy`] so callers that depend
/// on `scx-convert` (CLI, pyscx with hdf5) can name it without an
/// extra `scx_format` import. The actual definition lives in
/// `scx-format` so the CPU-only pyscx build (which doesn't pull in
/// `scx-convert`) can still drive bitmap generation from its in-memory
/// write path.
pub use scx_format_io::BitmapPolicy;
/// Re-exported from [`scx_format_io::CscPolicy`] so callers depending only on
/// `scx-convert` get the CSC policy type without an explicit `scx-format` dep.
pub use scx_format_io::CscPolicy;

impl ConvertOptions {
    /// Build the row-group [`FramingConfig`] for the shard emitters, or `None`
    /// for the unframed (v3) layout. Framing is active iff `row_group_rows` is set
    /// to a value > 0; `Some(0)` is the explicit unframed opt-out (v3 output).
    pub fn framing(&self) -> Option<scx_format_io::FramingConfig> {
        self.row_group_rows
            .filter(|&g| g > 0)
            .map(|g| scx_format_io::FramingConfig {
                row_group_rows: g,
                target_nnz: self.row_group_target_nnz,
                trial: self.codec_trial,
                decode_target: self.decode_target,
            })
    }

    /// Codec-selection intent for the provenance `params_json` (`codec_selection`
    /// key): the resolved profile (`auto`/`fast`/`compact`/`compact-trial` or an
    /// explicit codec name). The realized per-shard codecs are reported read-side
    /// by `scx info`; this records only the intent so a gate can verify it.
    pub fn codec_selection_value(&self) -> serde_json::Value {
        codec_selection_json(self.codec, self.codec_trial, self.decode_target)
    }
}

/// Build the `codec_selection` provenance value from a write's codec choice.
/// Shared by the streaming coordinators and the pyscx in-memory writer so the
/// stamp is identical across paths. `decode_target` is the internal mechanism
/// behind the adaptive intent profiles: `Auto` → `auto`, `Storage` → `compact`.
/// `None` (no adaptive bias) stamps `fast` (or the explicit codec name).
pub fn codec_selection_json(
    codec: Option<scx_codec::CodecId>,
    codec_trial: bool,
    decode_target: Option<scx_format_io::DecodeTarget>,
) -> serde_json::Value {
    use scx_format_io::DecodeTarget;
    let profile = if let Some(dt) = decode_target {
        match dt {
            DecodeTarget::Auto => "auto",
            DecodeTarget::Storage => "compact",
        }
    } else if codec_trial {
        "compact-trial"
    } else {
        // No adaptive bias and no explicit codec → heuristic single-encode
        // (the `fast` profile). An explicit codec stamps its own name.
        codec.map(|c| c.display_name()).unwrap_or("fast")
    };
    serde_json::json!({ "profile": profile })
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions {
            shard_target_rows: 16384,
            codec: None,
            csc: CscPolicy::Off,
            csc_cols_per_shard: 5000,
            // Framing on by default (Phase C): a plain `codec="auto"` write frames
            // at G=256 (codec-agnostic; no extra encode cost). `Some(0)` opts out
            // to unframed v3. `codec_trial` stays off — compact-trial's per-shard
            // trial encode remains opt-in.
            row_group_rows: Some(scx_format_io::DEFAULT_ROW_GROUP_ROWS),
            row_group_target_nnz: None,
            codec_trial: false,
            decode_target: None,
            tool: "scx".into(),
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
            sort_by: Vec::new(),
            sort_reverse: false,
            group_by: None,
            reference: None,
            group_target_bytes: None,
            group_max_bytes: None,
            group_pass: GroupPass::default(),
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

/// Apply `memory_budget` to the requested `(reader_threads,
/// writer_queue_depth)` so the peak outstanding shards stay within the
/// budget. Returns `(granted_threads, granted_depth)`.
///
/// Peak outstanding shards in the parallel coordinators is
/// `granted_threads + granted_depth` (workers in-flight + reorder
/// buffer + bounded channel), each holding roughly `per_shard_bytes`.
/// The constraint is therefore `(granted_threads + granted_depth) ×
/// per_shard_bytes ≤ budget`. The derate prefers shrinking
/// `granted_depth` over `granted_threads` so the dispatcher stays on
/// the parallel route under tight budgets — collapsing
/// `granted_threads` to 1 would route through the sequential
/// coordinator and lose parallelism entirely. Both have a floor of 1
/// (a queue depth of zero would starve the writer).
///
/// Shared by the streaming **ingest** coordinator
/// (`run_streaming_writer_coordinator`) and the streaming **export**
/// coordinator (`h5ad::stream_write`); `shard_noun` / `remedy`
/// customise the refusal error for each caller.
/// Refuse (T4.7 — no silent cap) when a single shard's working set cannot fit
/// `memory_budget`. `per_shard_bytes == 0` (empty / stats-less shard) and an
/// unset budget are both no-ops. Shared by the parallel derate
/// ([`derate_threads_and_depth`]) and the sequential grouped-ranges dispatch
/// (M2) so every route rejects an oversized shard with the same message.
pub(crate) fn ensure_shard_fits_budget(
    memory_budget: Option<u64>,
    per_shard_bytes: u64,
    shard_noun: &str,
    remedy: &str,
) -> Result<(), ConvertError> {
    if let Some(budget) = memory_budget {
        if per_shard_bytes > 0 && per_shard_bytes > budget {
            return Err(ConvertError::Other(format!(
                "single {shard_noun} requires \u{2248} {per_shard_bytes} bytes \
                 but memory_budget is {budget}; {remedy}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn derate_threads_and_depth(
    memory_budget: Option<u64>,
    per_shard_bytes: u64,
    requested_threads: usize,
    requested_depth: usize,
    shard_noun: &str,
    remedy: &str,
    sink: &mut WarningSink,
) -> Result<(usize, usize), ConvertError> {
    let Some(budget) = memory_budget else {
        return Ok((requested_threads, requested_depth));
    };
    if per_shard_bytes == 0 {
        // Empty shards (no rows / no stats) — nothing to cap.
        return Ok((requested_threads, requested_depth));
    }
    // Refuse when even a single shard cannot fit the budget (T4.7 — no silent
    // cap). Shared with the sequential grouped path (M2) so both routes reject
    // an oversized shard identically.
    ensure_shard_fits_budget(memory_budget, per_shard_bytes, shard_noun, remedy)?;
    // peak outstanding = threads + depth. Solve for the largest
    // outstanding ≤ budget / per_shard_bytes.
    let outstanding_max = (budget / per_shard_bytes).max(1) as usize;
    let requested_outstanding = requested_threads.saturating_add(requested_depth);
    if requested_outstanding <= outstanding_max {
        return Ok((requested_threads, requested_depth));
    }
    // Preserve parallelism: shrink depth first (floor 1), then shrink
    // threads only if necessary (floor 1). Reserving one slot for depth
    // and giving the rest to threads keeps `granted_threads > 1`
    // whenever `outstanding_max >= 3` (at `outstanding_max == 2`,
    // `granted_threads == 1` routes to the sequential coordinator), so the
    // dispatcher stays on the parallel route under tight budgets instead of
    // falling back to sequential.
    let granted_threads = outstanding_max
        .saturating_sub(1)
        .min(requested_threads)
        .max(1);
    let granted_depth = outstanding_max
        .saturating_sub(granted_threads)
        .min(requested_depth)
        .max(1);
    sink.emit(ConvertWarning::ReaderThreadsDerated {
        requested: requested_threads,
        granted: granted_threads,
        reason: format!(
            "memory_budget {budget} caps outstanding shards to {outstanding_max} \
             (per shard \u{2248} {per_shard_bytes} bytes); writer_queue_depth granted = \
             {granted_depth}"
        ),
    });
    Ok((granted_threads, granted_depth))
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
/// `pyscx::convert::build_and_write_predicate_indexes_inline` is the
/// Python-side mirror — keep their outcome handling shapes in sync.
#[allow(clippy::too_many_arguments)]
fn build_and_write_predicate_indexes(
    writer: &mut ScxWriter,
    obs: &RecordBatch,
    var: &RecordBatch,
    csr_row_ranges: &[(u64, u64)],
    n_vars: usize,
    opts: &ConvertOptions,
    // Sort-on-convert: extra obs columns to force-index (the sort key) so its
    // now-contiguous `shard_ranges` are emitted. Merged into `index_obs`,
    // deduped, order preserved.
    extra_index_obs: &[String],
    sink: &mut WarningSink,
) -> Result<(Vec<String>, Vec<String>), ConvertError> {
    let mut index_obs = opts.index_obs.clone();
    for col in extra_index_obs {
        if !index_obs.iter().any(|c| c == col) {
            index_obs.push(col.clone());
        }
    }
    let engine_opts = ConversionPredicateIndexOptions {
        index_obs,
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

    // Preset name + per-axis expected count flow into the outcome
    // processor so it can batch a fully-missing preset into a single
    // actionable warning. Unknown preset names resolve
    // to (0, 0); engine surfaces those as `ConvertError`.
    let (preset_obs_expected, preset_var_expected) = opts
        .index_preset
        .as_deref()
        .and_then(scx_engine::index::index_preset_columns)
        .map(|p| (p.obs_columns.len(), p.var_columns.len()))
        .unwrap_or((0, 0));
    // `__index_level_0__` (pyarrow's canonical
    // name for an unnamed pandas index) survives into the arrow schema
    // for round-trip purposes but should never be suggested as a
    // user-facing column name in a "did you mean" / "available columns"
    // preview. Drop any `__`-prefixed entries.
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
    process_predicate_index_outcomes(
        result.obs_outcomes,
        "obs",
        opts.index_preset.as_deref(),
        preset_obs_expected,
        &obs_available,
        sink,
    )?;
    process_predicate_index_outcomes(
        result.var_outcomes,
        "var",
        opts.index_preset.as_deref(),
        preset_var_expected,
        &var_available,
        sink,
    )?;

    Ok((result.obs_indexed_columns, result.var_indexed_columns))
}

/// Demote per-column outcomes from
/// `scx_engine::build_and_write_conversion_predicate_indexes` into the
/// convert layer's policy: forced errors abort the convert; preset
/// skips emit a typed warning whose variant is chosen by the
/// `SkipReason` discriminant.
///
/// When `preset` is `Some` and EVERY preset column on this axis came
/// back `SkipReason::MissingColumn` (preset/file format mismatch),
/// collapse the burst into a single `PresetNoColumnsMatched` warning
/// pointing the user at the fix instead of emitting one
/// `MissingPresetIndexColumn` per column. Partial mismatch keeps the
/// per-column shape — that's a real schema drift worth surfacing.
pub(crate) fn process_predicate_index_outcomes(
    outcomes: Vec<BuildOutcome>,
    axis: &str,
    preset: Option<&str>,
    preset_expected: usize,
    available_columns: &[String],
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let mut missing: Vec<String> = Vec::new();
    let mut deferred: Vec<ConvertWarning> = Vec::new();
    // Forced-column errors used to be fail-fast on
    // the first miss, so users had to iterate one typo per run. Collect
    // them all and surface as a single aggregated error after the loop,
    // matching the `missing` / `deferred` pattern below for PresetSkipped.
    let mut forced_missing: Vec<String> = Vec::new();
    // Non-missing forced errors (unsupported dtype, high cardinality)
    // can't reuse the strsim renderer — the column DOES exist; the
    // engine's text already describes the real reason. Preserve fail-
    // fast on those: they're per-column type/cardinality problems that
    // aren't related to typo'd column names.
    for outcome in outcomes {
        match outcome {
            BuildOutcome::ForcedColumnError { column, reason } => {
                if matches!(reason, SkipReason::MissingColumn) {
                    forced_missing.push(column);
                } else {
                    return Err(ConvertError::Other(format!(
                        "forced {axis} index column '{column}': {reason}"
                    )));
                }
            }
            BuildOutcome::PresetSkipped { column, reason } => match reason {
                SkipReason::MissingColumn => missing.push(column),
                other => deferred.push(ConvertWarning::UnsupportedIndexColumn {
                    column,
                    reason: other.to_string(),
                }),
            },
        }
    }

    if !forced_missing.is_empty() {
        let msg = scx_engine::index::forced_columns_missing_message(
            axis,
            &forced_missing,
            available_columns,
        );
        return Err(ConvertError::Other(msg));
    }

    let aggregate = preset.is_some_and(|_| {
        !missing.is_empty() && preset_expected > 0 && missing.len() == preset_expected
    });

    if aggregate {
        sink.emit(ConvertWarning::PresetNoColumnsMatched {
            preset: preset.expect("aggregate => preset.is_some()").to_string(),
            axis: axis.to_string(),
            missing,
        });
    } else {
        for column in missing {
            sink.emit(ConvertWarning::MissingPresetIndexColumn { column });
        }
    }
    for w in deferred {
        sink.emit(w);
    }
    Ok(())
}

pub fn h5ad_to_scx(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
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
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    crate::h5ad::stream_write::write_scx_to_h5ad_streaming(scx_path, h5ad_path, opts, sink)
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
/// of rows at a time via [`crate::h5ad::stream::XStreamReader`] so peak
/// memory is bounded by `shard_target_rows × n_vars × density × ~16
/// bytes` plus the always-resident indptr (`(n_obs + 1) × 8 bytes`).
///
/// Shard processing dispatches through
/// [`run_streaming_writer_coordinator`], which fans the per-shard read →
/// sort → drop-zeros → encode work out across a rayon worker pool and
/// reassembles output in shard order via a bounded crossbeam reorder
/// buffer (output is byte-identical to the sequential path). It falls
/// back to the sequential coordinator when libhdf5 is not built
/// threadsafe (gated by an `H5is_library_threadsafe` probe), when the
/// reader can't do row-range reads, or when `reader_threads <= 1`.
///
/// `opts.csc == true` runs a post-`finish()`
/// [`scx_ops::rebuild_csc_inplace`] pass over the just-written file;
/// peak disk briefly reaches ~2× the output size during the rebuild.
///
/// `obsm` / `varm` / `obsp` / `varp` are hyperslab-read one row-range
/// at a time and emitted as row-sharded sections
/// (`<section>/<name>_shard_<idx>`). Peak memory per matrix is bounded
/// by `shard_target_rows × k × 4 B` (dense) or
/// `shard_target_rows × density × n_cols × 16 B` (sparse). Caller
/// overrides supplied via [`StreamingOverrides`] are partitioned
/// into the same row-shards on the way out. `uns` is read in full
/// from disk when not overridden (typically KB–MB; no shard format
/// makes sense for a JSON tree). CSC-on-disk and dense `X` are
/// rejected up front with [`ConvertError::StreamingUnsupported`].
/// True when the h5ad has a non-empty `obsp` group (ignoring `__`-prefixed
/// internal members). Used by the grouped-convert router (M3) to route
/// obsp-carrying inputs through the obsp-preserving two-pass path.
fn h5ad_has_obsp_members(file: &hdf5::File) -> bool {
    match file.group("obsp") {
        Ok(group) => group
            .member_names()
            .unwrap_or_default()
            .iter()
            .any(|n| !n.starts_with("__")),
        Err(_) => false,
    }
}

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
    // Reorder-on-convert: `--sort-by` (Phase 2) and `--group-by` (Phase 7.4)
    // both permute the obs axis during ingest. A CSC-on-disk X cannot be
    // reordered (the CSC→CSR transposer is sequential and cannot serve the
    // permuted gather). Fail fast rather than silently emit a misordered file.
    if opts.reference.is_some() && opts.group_by.is_none() {
        return Err(ConvertError::Other(
            "convert --reference requires --group-by".to_string(),
        ));
    }
    let want_sort = !opts.sort_by.is_empty();
    let want_group = opts.group_by.is_some();
    let want_reorder = want_sort || want_group;
    if want_reorder && matches!(matrix_format, MatrixFormat::Csc) {
        return Err(ConvertError::Other(
            "reorder-on-convert (--sort-by / --group-by) requires a CSR or dense h5ad X; the \
             on-disk CSC X cannot be reordered during conversion. Re-export X as CSR/dense, or \
             reorder after conversion with `scx sort`."
                .to_string(),
        ));
    }

    // Phase 7.4 group-pass routing. The one-pass streaming grouped gather is a
    // win for CSR (reads only nnz/row) but a ~4–5× loss for dense (full-width
    // random row reads), so `Auto` routes dense → two-pass (plain convert then
    // `scx sort --group-by`), which is faster and lighter and produces the same
    // grouped layout. `One`/`Two` force the choice.
    if want_group {
        // M3: the one-pass streaming grouped route drops obsp (obsp remap in the
        // streaming writer is unsupported), while two-pass preserves it via
        // `scx sort`. Route any obsp-carrying grouped input through two-pass so
        // obsp survives regardless of X density — matching the byte-equivalent
        // convert-then-sort guarantee. `overrides.obsp` (in-memory, forwarded to
        // the two-pass plain pass) counts as obsp too.
        let has_obsp =
            overrides.obsp.as_ref().is_some_and(|v| !v.is_empty()) || h5ad_has_obsp_members(&file);
        let two_pass = match opts.group_pass {
            GroupPass::One => false,
            GroupPass::Two => true,
            GroupPass::Auto => matches!(matrix_format, MatrixFormat::Dense) || has_obsp,
        };
        // Byte-mode grouping needs a cheap per-row nnz source, which the one-pass
        // path only has for CSR. A forced one-pass over a non-CSR source with a
        // byte budget would silently degrade to row-count sizing, producing a
        // *different* layout than the `Auto`/`Two` route for the same flags —
        // error instead of diverging. (`Auto` already routes dense → two-pass.)
        if !two_pass
            && opts.group_target_bytes.is_some()
            && !matches!(matrix_format, MatrixFormat::Csr)
        {
            return Err(ConvertError::Other(format!(
                "convert --group-by --group-target-bytes with --group-pass one is unsupported for \
                 a {matrix_format:?} source (byte-budget sizing needs per-row nnz, available only \
                 for CSR in one pass); use --group-pass two (or auto) for byte-mode grouping"
            )));
        }
        // M3: a forced one-pass with obsp present would silently drop obsp. Refuse
        // rather than lose the graph; two-pass (or auto) preserves it.
        if !two_pass && has_obsp {
            return Err(ConvertError::Other(
                "convert --group-by with --group-pass one drops obsp (obsp remap in the one-pass \
                 streaming route is unsupported); use --group-pass two (or auto) to preserve obsp, \
                 or drop obsp before converting"
                    .to_string(),
            ));
        }
        if two_pass {
            log::info!(
                "convert --group-by: routing to two-pass (plain convert + scx sort) for \
                 {matrix_format:?} source"
            );
            return convert_then_sort_grouped(input, output, opts, overrides, sink);
        }
    }

    let mut x_reader: Box<dyn CsrShardStream> = match matrix_format {
        MatrixFormat::Csr => Box::new(open_x_streaming(&file, "X", matrix_format, sink)?),
        MatrixFormat::Dense => Box::new(open_dense_streaming(&file, "X", opts, sink)?),
        MatrixFormat::Csc => open_csc_streaming(&file, "X", opts, sink)?,
    };
    let n_obs = x_reader.n_obs() as usize;
    let n_vars = x_reader.n_vars() as usize;
    let n_vars_u32: u32 = u32::try_from(n_vars)
        .map_err(|_| ConvertError::Other(format!("n_vars {n_vars} exceeds u32::MAX")))?;
    let index_dtype: u8 = index_dtype_for(n_vars as u64);

    // Placeholder header. `nnz`, `n_csr_shards`, `n_csc_shards`, and
    // `codec_id` are overwritten by `ScxWriter::finish()` from
    // running accumulators (see scx-format/src/writer.rs).
    let mut header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        0,
        opts.shard_target_rows,
        0,
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

    // obs / var. Override-or-disk per section: any `Some(...)` field
    // wins over the on-disk read so the backed-AnnData routing path
    // can preserve in-memory mutations.
    let obs = match overrides.obs.as_ref() {
        Some(batch) => batch.clone(),
        None => read_dataframe_group(&file, "obs", sink)?,
    };
    let var = match overrides.var.as_ref() {
        Some(batch) => batch.clone(),
        None => read_dataframe_group(&file, "var", sink)?,
    };

    // Reorder-on-convert: compute the obs-axis permutation, reorder obs to
    // match, and wrap the X reader so the coordinator gathers source rows in the
    // new order. `--group-by` (Phase 7.4) takes precedence over `--sort-by`: it
    // computes a reference-first / group-by permutation via the shared
    // `scx_ops::compute_grouped_order` (the same routine `scx sort --group-by`
    // uses, so the output is byte-equivalent to convert-then-sort), plans
    // group-aligned shard breaks, and stages the `group_index` sidecar (written
    // after X). The non-reorder path keeps `obs` / `x_reader` untouched.
    let mut group_ranges: Option<Vec<(u64, u32)>> = None;
    let mut group_index_bytes: Option<Vec<u8>> = None;
    let sort_perm: Option<std::sync::Arc<Vec<u64>>> = if let Some(group_col) = &opts.group_by {
        let go =
            scx_ops::compute_grouped_order(&obs, group_col, &opts.sort_by, opts.reference.as_ref())
                .map_err(|e| ConvertError::Other(format!("convert --group-by: {e}")))?;

        // Per-row nnz in emission order (byte-budget mode, CSR only): the h5ad
        // CSR indptr is the cheap per-row nnz source. Dense/CSC have no cheap
        // per-row nnz, so byte mode falls back to row-count with a warning.
        let (per_row_nnz, target_units, bytes_per_nnz) = match opts.group_target_bytes {
            Some(tb) if matches!(matrix_format, MatrixFormat::Csr) => {
                let indptr_ds = file.group("X")?.dataset("indptr")?;
                let indptr = crate::h5ad::read::read_i64_dataset(&indptr_ds)?;
                let prn: Vec<u64> = go
                    .perm
                    .iter()
                    .map(|&src| {
                        let s = src as usize;
                        (indptr[s + 1] - indptr[s]) as u64
                    })
                    .collect();
                (prn, tb.max(1), scx_ops::GROUP_BYTES_PER_NNZ)
            }
            Some(_) => {
                sink.emit(ConvertWarning::GroupByteModeUnsupported {
                    source_format: match matrix_format {
                        MatrixFormat::Dense => "dense".to_string(),
                        MatrixFormat::Csc => "csc".to_string(),
                        MatrixFormat::Csr => "csr".to_string(),
                    },
                });
                (Vec::new(), opts.shard_target_rows.max(1) as u64, 0u64)
            }
            None => (Vec::new(), opts.shard_target_rows.max(1) as u64, 0u64),
        };
        let max_units = opts
            .group_max_bytes
            .unwrap_or_else(|| target_units.saturating_mul(scx_ops::GROUP_MAX_BYTES_MULTIPLE));
        let plan = scx_ops::plan_group_shards(
            &go.group_of_new,
            &go.ref_of_new,
            &go.labels,
            &per_row_nnz,
            target_units,
            bytes_per_nnz,
            max_units,
        );
        group_ranges = Some(group_shard_starts_to_ranges(
            &plan.shard_starts,
            n_obs as u64,
        ));
        let payload = plan.to_sidecar_json(group_col, &go.reference_labels);
        group_index_bytes = Some(serde_json::to_vec(&payload).map_err(|e| {
            ConvertError::Other(format!(
                "convert --group-by: failed to serialize group index: {e}"
            ))
        })?);
        Some(std::sync::Arc::new(go.perm))
    } else if want_sort {
        let perm =
            crate::permuted_reader::compute_sort_perm(&obs, &opts.sort_by, opts.sort_reverse)?;
        Some(std::sync::Arc::new(perm))
    } else {
        None
    };
    let obs = match &sort_perm {
        Some(perm) => crate::permuted_reader::take_record_batch(&obs, perm)?,
        None => obs,
    };
    if let Some(perm) = &sort_perm {
        // Re-open X as an indexed reader and wrap it in the permuted gather.
        drop(x_reader);
        let inner: Box<dyn crate::stream::IndexedCsrShardStream> = match matrix_format {
            MatrixFormat::Csr => Box::new(open_x_streaming(&file, "X", matrix_format, sink)?),
            MatrixFormat::Dense => Box::new(open_dense_streaming(&file, "X", opts, sink)?),
            MatrixFormat::Csc => unreachable!("CSC + sort guarded above"),
        };
        x_reader = Box::new(crate::permuted_reader::PermutedCsrReader::new(
            inner,
            perm.clone(),
        ));
    }

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
        // Canonical single-modality CSR shard name is `X_shard_{idx}`
        // (writer.rs::write_csr_shard, catalog_view, and the explode/push
        // name->path mapper all expect uppercase). The coordinator appends
        // `_{idx}` to this prefix. A lowercase prefix produced `x_shard_0`,
        // which the reader tolerated (it resolves shards by SectionType, not
        // name) but `scx explode`/`scx push` rejected.
        "X_shard",
        sink,
        group_ranges.as_deref(),
    )?;
    drop(x_reader);

    // Phase 7.4: write the `group_index` sidecar (same bytes `scx sort` emits).
    if let Some(bytes) = &group_index_bytes {
        writer.write_group_index(bytes)?;
    }

    // obsm / varm / obsp / varp / uns. The dense + sparse mappings are
    // emitted as row-sharded sections — one Arrow IPC section per
    // shard — so peak memory is bounded by `shard_target_rows` worth
    // of rows per matrix. Override path: an in-memory `RecordBatch`
    // supplied by the caller (typically pyscx's backed-routing path
    // for sections the user mutated in Python) is sliced into shards
    // on the way out, also keeping peak memory to one shard at a time.
    // Disk-streaming path: the source h5ad is hyperslab-read one
    // row-range at a time per key.
    // obsm is obs-axis → reorder it by the same permutation when sorting.
    let obsm_perm: Option<&[u64]> = sort_perm.as_ref().map(|p| p.as_slice());
    write_dense_mapping_section(
        &file,
        &mut writer,
        overrides.obsm.as_ref(),
        "obsm",
        opts.shard_target_rows,
        DenseMappingKind::Obsm,
        obsm_perm,
        sink,
    )?;
    // varm is var-axis → never reordered by an obs sort.
    write_dense_mapping_section(
        &file,
        &mut writer,
        overrides.varm.as_ref(),
        "varm",
        opts.shard_target_rows,
        DenseMappingKind::Varm,
        None,
        sink,
    )?;
    // obsp (obs×obs) needs both axes remapped through the permutation. The
    // standalone `scx sort` engine does this remap; the convert-on-sort path
    // does not yet, so drop obsp with a warning here rather than emit a
    // misaligned graph. varp (var×var) is untouched by an obs sort.
    if sort_perm.is_some() {
        if let Ok(group) = file.group("obsp") {
            for name in group.member_names().unwrap_or_default() {
                if name.starts_with("__") {
                    continue;
                }
                sink.emit(ConvertWarning::DroppedObsp {
                    name: format!("obsp/{name}"),
                    reason: "obsp remap under sort-on-convert is not yet supported \
                             (Phase 5); dropped to avoid an axis-misaligned graph"
                        .to_string(),
                });
            }
        }
    } else {
        write_sparse_mapping_section(
            &file,
            &mut writer,
            overrides.obsp.as_ref(),
            "obsp",
            opts.shard_target_rows,
            SparseMappingKind::Obsp,
            sink,
        )?;
    }
    write_sparse_mapping_section(
        &file,
        &mut writer,
        overrides.varp.as_ref(),
        "varp",
        opts.shard_target_rows,
        SparseMappingKind::Varp,
        sink,
    )?;

    // Optional `adata.raw` count matrix (its own var axis) → raw section
    // family, streamed shard-by-shard so peak RSS stays bounded (mirrors
    // the streaming `/X` path).
    ingest_raw_streaming(&file, &mut writer, n_obs, opts, sink)?;

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
    // (scx-convert/src/h5ad/read.rs). Once a layer's shards start
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
                    Ok(r) => match &sort_perm {
                        Some(p) => Box::new(crate::permuted_reader::PermutedCsrReader::new(
                            Box::new(r),
                            p.clone(),
                        )),
                        None => Box::new(r),
                    },
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
                        Ok(r) => match &sort_perm {
                            Some(p) => Box::new(crate::permuted_reader::PermutedCsrReader::new(
                                Box::new(r),
                                p.clone(),
                            )),
                            None => Box::new(r),
                        },
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
                    // A CSC-on-disk layer can't be reordered (sequential
                    // transposer); under sort-on-convert, drop it rather than
                    // emit an axis-misaligned layer.
                    if sort_perm.is_some() {
                        sink.emit(ConvertWarning::LayerSkipped {
                            name: layer_name.clone(),
                            reason: "cannot reorder a CSC-on-disk layer during \
                                     sort-on-convert; layer dropped"
                                .to_string(),
                        });
                        continue;
                    }
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
            let l_index_dtype: u8 = index_dtype_for(l_n_vars as u64);
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
                // Layers keep fixed-size shard breaks even under a grouped
                // convert — matching `scx sort`, which group-breaks only X.
                None,
            )?;
        }
    }

    // Phase 5a: predicate indexes from the obs/var we just wrote and
    // the actual shard boundaries reported by `streaming_writer_coordinator`.
    // Reorder-on-convert auto-indexes its keys so the contiguous shard ranges
    // are emitted: the `--group-by` column leads (Phase 7.4), then `--sort-by`.
    let reorder_keys: Vec<String> = match &opts.group_by {
        Some(group_col) => {
            let mut keys = Vec::with_capacity(opts.sort_by.len() + 1);
            keys.push(group_col.clone());
            keys.extend(opts.sort_by.iter().filter(|c| *c != group_col).cloned());
            keys
        }
        None => opts.sort_by.clone(),
    };
    let (obs_indexed, var_indexed) = build_and_write_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &csr_row_ranges,
        n_vars,
        opts,
        &reorder_keys,
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
    // Phase 7.4: record the grouping config when `--group-by` was used (mirrors
    // `scx sort`'s `grouping_provenance`), else `null`.
    let grouping_json = match &opts.group_by {
        Some(group_by) => {
            let reference = match &opts.reference {
                Some(scx_ops::ReferenceSpec::Labels(l)) => serde_json::json!({ "labels": l }),
                Some(scx_ops::ReferenceSpec::Column(c)) => serde_json::json!({ "column": c }),
                None => serde_json::Value::Null,
            };
            serde_json::json!({
                "group_by": group_by,
                "reference": reference,
                "group_target_bytes": opts.group_target_bytes,
                "group_max_bytes": opts.group_max_bytes,
            })
        }
        None => serde_json::Value::Null,
    };
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: opts.tool.clone(),
        params_json: serde_json::json!({
            "input": input.display().to_string(),
            "format": "h5ad",
            "stream": true,
            "codec_selection": opts.codec_selection_value(),
            "source_matrix_format": source_format_str,
            "warnings": sink.summary_json(),
            "predicate_index": {
                "obs_columns": obs_indexed,
                "var_columns": var_indexed,
                "preset": opts.index_preset,
            },
            "sort_by": opts.sort_by,
            "sort_reverse": opts.sort_reverse,
            "grouping": grouping_json,
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
    if opts.csc.should_build_csc(n_obs as u64, n_vars as u64) {
        // Pass framing so `--csc <policy> --row-group-rows N` produces a framed
        // CSC sidecar (and keeps X framed) instead of silently downgrading to v3.
        scx_ops::rebuild_csc_inplace(output, opts.csc_cols_per_shard, "4G", opts.framing())
            .map_err(|e| ConvertError::Other(format!("rebuild_csc_inplace failed: {e}")))?;
    }

    Ok(())
}

/// Phase 7.4 two-pass grouped convert: plain convert to a temp SCX, then
/// `scx sort --group-by` into `output`. The `Auto` group-pass routes dense
/// sources here because the one-pass streaming gather reads full rows for a
/// dense matrix (~4–5× slower / ~2× memory than this). Produces the same
/// grouped layout as a manual convert-then-sort; the temp is removed on exit.
fn convert_then_sort_grouped(
    input: &Path,
    output: &Path,
    opts: &ConvertOptions,
    overrides: &StreamingOverrides,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let group_by = opts
        .group_by
        .clone()
        .expect("convert_then_sort_grouped requires group_by");

    // Temp plain SCX alongside the output (same filesystem). Grouping / sort /
    // index / bitmap / csc are stripped from the plain pass — `scx sort` owns
    // the final grouped layout, predicate index, bitmaps, and (rebuilt) CSC.
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output.file_name().and_then(|s| s.to_str()).unwrap_or("out");
    let tmp = parent.join(format!(".{stem}.grouptmp.scx"));

    let plain_opts = ConvertOptions {
        group_by: None,
        reference: None,
        group_target_bytes: None,
        group_max_bytes: None,
        group_pass: GroupPass::One,
        sort_by: Vec::new(),
        sort_reverse: false,
        csc: CscPolicy::Off,
        bitmap: BitmapPolicy::Off,
        index_obs: Vec::new(),
        index_var: Vec::new(),
        index_preset: None,
        ..opts.clone()
    };
    h5ad_to_scx_streaming(input, &tmp, &plain_opts, overrides, sink)?;

    let mut by = Vec::with_capacity(opts.sort_by.len() + 1);
    by.push(group_by.clone());
    by.extend(opts.sort_by.iter().filter(|c| **c != group_by).cloned());
    let sort_opts = scx_ops::SortOptions {
        by,
        reverse: false,
        shard_target_rows: opts.shard_target_rows,
        codec: match opts.codec {
            Some(c) => scx_codec::CodecSelection::Explicit(c),
            None => scx_codec::CodecSelection::Auto,
        },
        index_options: ConversionPredicateIndexOptions {
            index_obs: opts.index_obs.clone(),
            index_var: opts.index_var.clone(),
            index_preset: opts.index_preset.clone(),
            index_auto_threshold: opts.index_auto_threshold,
        },
        memory_budget: opts.memory_budget,
        temp_dir: opts.temp_dir.clone(),
        bitmap: opts.bitmap,
        group_by: Some(group_by),
        reference: opts.reference.clone(),
        group_target_bytes: opts.group_target_bytes,
        group_max_bytes: opts.group_max_bytes,
        // None => the sort engine's default block cap (256 MB), giving the
        // convert two-pass sort the F6 grouped-write OOM fix for free.
        group_write_block_bytes: None,
    };
    let sort_result = scx_ops::sort(&tmp, output, &sort_opts)
        .map_err(|e| ConvertError::Other(format!("convert --group-by (two-pass sort): {e}")));
    // Always clean up the temp, even on sort failure.
    let _ = std::fs::remove_file(&tmp);
    sort_result?;

    // `scx sort` drops any CSC sidecar; rebuild it on the output to honour the
    // requested CSC policy (mirrors the one-pass path's end-of-convert rebuild).
    if let Ok(reader) = scx_format_io::reader::ScxReader::open(output) {
        let (n_obs, n_vars) = (reader.n_obs(), reader.n_vars());
        drop(reader);
        if opts.csc.should_build_csc(n_obs, n_vars) {
            scx_ops::rebuild_csc_inplace(output, opts.csc_cols_per_shard, "4G", opts.framing())
                .map_err(|e| ConvertError::Other(format!("rebuild_csc_inplace failed: {e}")))?;
        }
    }
    Ok(())
}

/// Drain a [`CsrShardStream`] into an [`ScxWriter`] one shard at a
/// time, encoding each shard through [`encode_one_shard`].
///
/// Phase 0.2 seam. The initial implementation is sequential — one
/// shard read → canonicalize → encode → write per iteration —
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
        canonicalize_csr(&mut shard.indptr, &mut shard.indices, &mut shard.values);
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
            opts.framing(),
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

/// Phase 7.4: expand a group planner's `shard_starts` (emit-row indices at
/// which a new shard begins, excluding 0 and EOF) into the explicit
/// `[(row_start, n_rows)]` shard ranges covering `[0, n_obs)` that the
/// coordinator drives. Groups are never split, so ranges are contiguous and
/// gap-free.
fn group_shard_starts_to_ranges(shard_starts: &[u64], n_obs: u64) -> Vec<(u64, u32)> {
    let mut ranges: Vec<(u64, u32)> = Vec::with_capacity(shard_starts.len() + 1);
    let mut start = 0u64;
    for &brk in shard_starts {
        if brk > start {
            ranges.push((start, (brk - start) as u32));
            start = brk;
        }
    }
    if start < n_obs {
        ranges.push((start, (n_obs - start) as u32));
    }
    ranges
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
    ranges: Vec<(u64, u32)>,
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    use crossbeam_channel::bounded;
    use rayon::ThreadPoolBuilder;

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
    let opts_framing = opts.framing();
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
                        opts_framing,
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
            let out = r?;
            buffer.insert(idx, out);
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
    framing: Option<FramingConfig>,
) -> Result<EncodedShardOutput, ConvertError> {
    let mut shard = reader.read_range(row_start, n_rows)?;
    let duplicates_merged = shard.duplicates_merged;
    canonicalize_csr(&mut shard.indptr, &mut shard.indices, &mut shard.values);

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
        framing,
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
///
/// `explicit_ranges` (Phase 7.4 convert-time grouping): when `Some`, the
/// caller supplies group-aligned `[(row_start, n_rows)]` shard ranges instead
/// of the fixed-`shard_target_rows` partition. Groups are never split, so a
/// single range may exceed `shard_target_rows`; the per-worker budget is sized
/// by the largest range. The grouped X reader is always indexed, so the ranges
/// are honored in both the parallel pool and the sequential fallback. All
/// non-grouped callers pass `None` and are byte-identical to before.
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
    explicit_ranges: Option<&[(u64, u32)]>,
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    let requested = resolve_reader_threads(opts);

    // ----- Phase 7.4: group-aligned ranges -----
    if let Some(ranges) = explicit_ranges {
        if ranges.is_empty() {
            return Ok((0, Vec::new()));
        }
        // The grouped X reader is always an indexed (row-range) reader
        // (PermutedCsrReader). Without it we cannot honor variable breaks.
        let Some(indexed) = reader.as_indexed() else {
            return Err(ConvertError::Other(
                "grouped convert requires an indexed (row-range) X reader".to_string(),
            ));
        };
        // The largest group bounds the per-worker working set (groups are never
        // split), so size the derate by it rather than by `--shard-size`.
        let max_range_rows = ranges.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let per_worker_bytes = indexed
            .per_worker_bytes(max_range_rows, modality_type)
            .max(1);
        // M2: the sequential fallback below buffers the largest group whole, so
        // it must honor `memory_budget` too — the parallel derate already
        // refuses an oversized shard, but the sequential branch previously did
        // not. Check before either dispatch so both routes reject identically.
        ensure_shard_fits_budget(
            opts.memory_budget,
            per_worker_bytes,
            "grouped shard",
            "raise --group-target-bytes/--memory-budget or accept a larger group shard",
        )?;
        let threadsafe = crate::hdf5_threadsafe::hdf5_is_threadsafe();
        if requested <= 1 || !threadsafe {
            if requested > 1 && !threadsafe {
                crate::hdf5_threadsafe::try_emit_not_threadsafe_warning(sink);
            }
            return streaming_writer_coordinator_ranges(
                indexed,
                writer,
                opts,
                index_dtype,
                n_vars_u32,
                section_type,
                modality_type,
                section_name_prefix,
                sink,
                ranges,
            );
        }
        let requested_depth = opts.writer_queue_depth.max(1);
        let (granted, granted_depth) = derate_threads_and_depth(
            opts.memory_budget,
            per_worker_bytes,
            requested,
            requested_depth,
            "grouped parallel-streaming shard",
            "raise --group-target-bytes/--memory-budget or accept a larger group shard",
            sink,
        )?;
        if granted <= 1 {
            return streaming_writer_coordinator_ranges(
                indexed,
                writer,
                opts,
                index_dtype,
                n_vars_u32,
                section_type,
                modality_type,
                section_name_prefix,
                sink,
                ranges,
            );
        }
        return streaming_writer_coordinator_parallel(
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
            granted_depth,
            ranges.to_vec(),
        );
    }

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
    // Derate threads AND depth together: peak outstanding shards is
    // `granted + granted_depth` (see `streaming_writer_coordinator_parallel`'s
    // `in_flight_cap`), so the budget must cover both — sizing threads alone
    // overshot by up to `writer_queue_depth × per_worker_bytes`.
    let requested_depth = opts.writer_queue_depth.max(1);
    let (granted, granted_depth) = derate_threads_and_depth(
        opts.memory_budget,
        per_worker_bytes,
        requested,
        requested_depth,
        "parallel-streaming shard",
        "lower --shard-size or raise --memory-budget",
        sink,
    )?;

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

    let ranges = compute_shard_row_ranges(indexed.n_obs(), effective_target);
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
        granted_depth,
        ranges,
    )
}

/// Sequential sibling of [`streaming_writer_coordinator`] that honors an
/// explicit list of (possibly variable-size) shard ranges — the group-aligned
/// breaks from the Phase 7.4 grouped-convert planner. Reuses
/// [`encode_one_shard_worker`] per range so the encoded bytes are identical to
/// the parallel path for the same ranges (which is in turn byte-identical to
/// the fixed-size sequential coordinator). Requires an indexed reader.
#[allow(clippy::too_many_arguments)]
fn streaming_writer_coordinator_ranges(
    reader: &dyn crate::stream::IndexedCsrShardStream,
    writer: &mut ScxWriter,
    opts: &ConvertOptions,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    section_name_prefix: &str,
    sink: &mut WarningSink,
    ranges: &[(u64, u32)],
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    let want_bitmap = section_type == SectionType::CsrShard;
    let mut row_ranges: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (shard_idx, &(row_start, n_rows)) in ranges.iter().enumerate() {
        let out = encode_one_shard_worker(
            reader,
            row_start,
            n_rows,
            opts.codec,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            format!("{section_name_prefix}_{shard_idx}"),
            opts.bitmap,
            want_bitmap,
            opts.framing(),
        )?;
        if out.duplicates_merged > 0 {
            sink.emit(ConvertWarning::DuplicateCoordinatesMerged {
                count: out.duplicates_merged,
                policy: "sum".to_string(),
            });
        }
        writer.write_preencoded_shard(out.pre)?;
        if want_bitmap {
            match out.bitmap {
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
        row_ranges.push((row_start, row_start + n_rows as u64));
    }
    Ok((ranges.len() as u32, row_ranges))
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
#[path = "pipeline_test_hooks.rs"]
pub(crate) mod test_hooks;

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
        scx_format_io::csc_sidecar::DEFAULT_CSC_MEMORY_BYTES,
        None,
        framing,
    )?;
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
fn ingest_raw_if_present(
    file: &hdf5::File,
    writer: &mut ScxWriter,
    n_obs: usize,
    opts: &ConvertOptions,
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
fn ingest_raw_streaming(
    file: &hdf5::File,
    writer: &mut ScxWriter,
    n_obs: usize,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    if file.group("raw").is_err() {
        return Ok(());
    }
    if file.group("raw/X").is_err() && file.dataset("raw/X").is_err() {
        return Ok(());
    }

    let raw_format = super::detect::detect_matrix_format_at(file, "raw/X", sink)?;
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
    if !opts.sort_by.is_empty() || opts.group_by.is_some() {
        sink.emit(ConvertWarning::DroppedRaw { raw_n_vars });
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

/// Dispatch tag for [`write_dense_mapping_section`] so the shard
/// writer can pick the right `ScxWriter` method without duplicating
/// the obsm/varm loops.
#[derive(Debug, Clone, Copy)]
enum DenseMappingKind {
    Obsm,
    Varm,
}

/// Same for [`write_sparse_mapping_section`] over obsp/varp.
#[derive(Debug, Clone, Copy)]
enum SparseMappingKind {
    Obsp,
    Varp,
}

/// Gather the obsm/varm rows for one output shard in permuted (sorted) order
/// while reading only contiguous source runs from disk — peak memory stays
/// ~one output shard, mirroring [`crate::permuted_reader::PermutedCsrReader::gather`]
/// for the dense case. `want[i]` is the source row index for output-local row
/// `i`. Used by the sort-on-convert disk-streaming path so it preserves the
/// same per-shard RSS bound as the non-sort path (the previous code
/// materialized the whole `n_obs × k` mapping before permuting).
///
/// No `max_slab_rows` cap is needed here (unlike `PermutedCsrReader::gather`):
/// every coalesced run lives inside a single output shard, so a run is bounded
/// by `shard_target_rows` — exactly the hyperslab size the non-sort path
/// already issues.
fn gather_dense_mapping_shard(
    file: &hdf5::File,
    group_path: &str,
    name: &str,
    want: &[u64],
) -> Result<RecordBatch, ConvertError> {
    // The sole caller passes a non-empty shard slice (the `info.n_rows == 0`
    // case is handled in a separate branch), but guard the `runs[0]`
    // precondition explicitly: an empty gather is an empty mapping batch with
    // the correct schema.
    if want.is_empty() {
        return read_dense_mapping_shard(file, group_path, name, 0, 0);
    }

    // (output-local index, source id) sorted by source id so consecutive
    // source rows coalesce into a single contiguous hyperslab read.
    let mut order: Vec<(usize, u64)> = want.iter().copied().enumerate().collect();
    order.sort_unstable_by_key(|&(_, src)| src);

    let mut runs: Vec<RecordBatch> = Vec::new();
    // `take_idx[out_local]` = row position of that output row within the
    // run-order concatenation below.
    let mut take_idx = vec![0u64; want.len()];
    let mut concat_pos = 0u64;
    let mut i = 0;
    while i < order.len() {
        let run_start = order[i].1 as usize;
        let mut j = i + 1;
        while j < order.len() && order[j].1 == order[j - 1].1 + 1 {
            j += 1;
        }
        // `read_dense_mapping_shard` takes a half-open [start, end) range.
        let run_end = order[j - 1].1 as usize + 1;
        runs.push(read_dense_mapping_shard(
            file, group_path, name, run_start, run_end,
        )?);
        for entry in &order[i..j] {
            take_idx[entry.0] = concat_pos;
            concat_pos += 1;
        }
        i = j;
    }

    // Runs are read in source-sorted order; concatenate then permute into
    // output (sorted-by-obs-key) order.
    let schema = runs[0].schema();
    let concatenated = arrow::compute::concat_batches(&schema, runs.iter())?;
    crate::permuted_reader::take_record_batch(&concatenated, &take_idx)
}

/// Emit one logical obsm/varm matrix as a sequence of row-shards. Used
/// by [`h5ad_to_scx_streaming`] for both the override path (in-memory
/// `RecordBatch` from pyscx) and the disk-streaming path (h5py
/// hyperslab reads per shard).
#[allow(clippy::too_many_arguments)]
fn write_dense_mapping_section(
    file: &hdf5::File,
    writer: &mut ScxWriter,
    override_entries: Option<&Vec<(String, RecordBatch)>>,
    group_path: &str,
    shard_target_rows: u32,
    kind: DenseMappingKind,
    // Sort-on-convert (Phase 2): when `Some`, reorder this section's rows by
    // `row_perm[output_row] = source_row]` before sharding. Used for obsm
    // (obs-axis); always `None` for varm (var-axis is untouched by an obs sort).
    row_perm: Option<&[u64]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let emit = |w: &mut ScxWriter,
                name: &str,
                shard_idx: u32,
                row_start: u64,
                n_shard_rows: u64,
                n_total: u64,
                batch: &RecordBatch|
     -> Result<(), ConvertError> {
        match kind {
            DenseMappingKind::Obsm => {
                w.write_obsm_shard(name, shard_idx, row_start, n_shard_rows, n_total, batch)
            }
            DenseMappingKind::Varm => {
                w.write_varm_shard(name, shard_idx, row_start, n_shard_rows, n_total, batch)
            }
        }
        .map_err(ConvertError::from)
    };

    if let Some(entries) = override_entries {
        for (name, batch) in entries {
            // Sort-on-convert: permute rows before sharding (obsm only).
            let permuted;
            let batch: &RecordBatch = match row_perm {
                Some(p) => {
                    permuted = crate::permuted_reader::take_record_batch(batch, p)?;
                    &permuted
                }
                None => batch,
            };
            let n_rows = batch.num_rows();
            let n_total = n_rows as u64;
            if n_rows == 0 {
                emit(writer, name, 0, 0, 0, 0, batch)?;
                continue;
            }
            let step = shard_target_rows.max(1) as usize;
            let mut shard_idx = 0u32;
            let mut row_start = 0usize;
            while row_start < n_rows {
                let n = (n_rows - row_start).min(step);
                let shard = batch.slice(row_start, n);
                emit(
                    writer,
                    name,
                    shard_idx,
                    row_start as u64,
                    n as u64,
                    n_total,
                    &shard,
                )?;
                row_start += n;
                shard_idx += 1;
            }
        }
        return Ok(());
    }

    // Disk-streaming path. Missing group → nothing to do.
    let infos = list_dense_mapping_shapes(file, group_path)?;

    // Surface every member `list_dense_mapping_shapes` could not handle as
    // a `SkippedObsm` warning so the loss is visible from Python and counted
    // in provenance instead of being silently swallowed (B4) or aborting the
    // whole conversion on an unreadable dtype (B5 residual). Handled members
    // are exactly the readable 2D dense datasets returned above; anything
    // else is a DataFrame subgroup, a sparse-matrix subgroup, a non-2D
    // dataset, or an unsupported-dtype dataset. Mirrors the `DroppedObsp`
    // classification in `write_sparse_mapping_section`.
    if let Ok(group) = file.group(group_path) {
        use std::collections::HashSet;
        let handled: HashSet<&str> = infos.iter().map(|i| i.name.as_str()).collect();
        for name in group.member_names()? {
            if name.starts_with("__") || handled.contains(name.as_str()) {
                continue;
            }
            sink.emit(ConvertWarning::SkippedObsm {
                name: format!("{group_path}/{name}"),
                reason: "not a 2D dense numeric dataset (DataFrame-valued, \
                         sparse-matrix-valued, non-2D, or unsupported dtype) \
                         — dense obsm/varm only"
                    .to_string(),
            });
        }
    }

    for info in &infos {
        let n_total = info.n_rows as u64;
        // Zero-row dense mappings: emit a single empty shard so the key
        // survives round-trip (mirrors the override-path special case).
        if info.n_rows == 0 {
            let batch = read_dense_mapping_shard(file, group_path, &info.name, 0, 0)?;
            emit(writer, &info.name, 0, 0, 0, 0, &batch)?;
            continue;
        }
        // Sort-on-convert (obsm): gather each output shard in permuted order
        // directly, reading only the contiguous source runs that shard needs.
        // Peak memory stays ~one output shard (independent of n_obs), matching
        // the non-sort path's per-shard RSS bound rather than materializing the
        // whole n_obs × k mapping. Mirrors `PermutedCsrReader::gather` (X path).
        if let Some(perm) = row_perm {
            // The obs sort permutation is indexed per output shard below; a
            // malformed file whose mapping row count differs from n_obs would
            // otherwise slice `perm` out of bounds. Reject rather than panic.
            if info.n_rows != perm.len() {
                return Err(ConvertError::Other(format!(
                    "obsm/varm '{}/{}' has {} rows but the obs sort permutation has {} \
                     (mapping row count must equal n_obs)",
                    group_path,
                    info.name,
                    info.n_rows,
                    perm.len()
                )));
            }
            let step = shard_target_rows.max(1) as usize;
            let mut shard_idx = 0u32;
            let mut row_start = 0usize;
            while row_start < info.n_rows {
                let n = (info.n_rows - row_start).min(step);
                let shard = gather_dense_mapping_shard(
                    file,
                    group_path,
                    &info.name,
                    &perm[row_start..row_start + n],
                )?;
                emit(
                    writer,
                    &info.name,
                    shard_idx,
                    row_start as u64,
                    n as u64,
                    n_total,
                    &shard,
                )?;
                row_start += n;
                shard_idx += 1;
            }
            continue;
        }
        let step = shard_target_rows.max(1) as usize;
        let mut shard_idx = 0u32;
        let mut row_start = 0usize;
        while row_start < info.n_rows {
            let row_end = (row_start + step).min(info.n_rows);
            let batch = read_dense_mapping_shard(file, group_path, &info.name, row_start, row_end)?;
            let n_shard_rows = (row_end - row_start) as u64;
            emit(
                writer,
                &info.name,
                shard_idx,
                row_start as u64,
                n_shard_rows,
                n_total,
                &batch,
            )?;
            row_start = row_end;
            shard_idx += 1;
        }
    }
    Ok(())
}

/// Emit one logical obsp/varp matrix as a sequence of row-shards. The
/// override path takes a single materialised COO `RecordBatch` per key
/// and partitions its triples by `row` into shard ranges; the
/// disk-streaming path reads h5py CSR slices per row range.
fn write_sparse_mapping_section(
    file: &hdf5::File,
    writer: &mut ScxWriter,
    override_entries: Option<&Vec<(String, RecordBatch)>>,
    group_path: &str,
    shard_target_rows: u32,
    kind: SparseMappingKind,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let emit = |w: &mut ScxWriter,
                name: &str,
                shard_idx: u32,
                row_start: u64,
                n_shard_rows: u64,
                n_total: u64,
                batch: &RecordBatch|
     -> Result<(), ConvertError> {
        match kind {
            SparseMappingKind::Obsp => {
                w.write_obsp_shard_coo(name, shard_idx, row_start, n_shard_rows, n_total, batch)
            }
            SparseMappingKind::Varp => {
                w.write_varp_shard_coo(name, shard_idx, row_start, n_shard_rows, n_total, batch)
            }
        }
        .map_err(ConvertError::from)
    };

    if let Some(entries) = override_entries {
        for (name, batch) in entries {
            partition_coo_to_shards(
                batch,
                shard_target_rows,
                |shard_idx, row_start, n_shard_rows, n_total, sub| {
                    emit(
                        writer,
                        name,
                        shard_idx,
                        row_start,
                        n_shard_rows,
                        n_total,
                        sub,
                    )
                },
            )?;
        }
        return Ok(());
    }

    // Disk-streaming path. Classify every member of the obsp/varp group:
    //   * valid CSR sparse subgroup → existing COO shard path,
    //   * valid 2D dense dataset    → dense→COO shard path (reuse of the
    //     dense obsm/varm reader; only nonzeros are stored, so a mostly-
    //     zero pairwise matrix never costs the full n_obs² on disk),
    //   * anything else (CSC subgroup, non-2D, malformed CSR) → dropped
    //     with a `DroppedObsp` warning so the loss is visible from Python
    //     and counted in provenance instead of silently swallowed.
    let sparse_infos = list_sparse_mapping_shapes(file, group_path)?;
    let dense_infos = list_dense_mapping_shapes(file, group_path)?;

    // Surface the drop for any member handled by neither reader. The
    // group may be absent entirely (no obsp/varp) — that is not a drop.
    if let Ok(group) = file.group(group_path) {
        use std::collections::HashSet;
        let handled: HashSet<&str> = sparse_infos
            .iter()
            .map(|i| i.name.as_str())
            .chain(dense_infos.iter().map(|i| i.name.as_str()))
            .collect();
        for name in group.member_names()? {
            if name.starts_with("__") || handled.contains(name.as_str()) {
                continue;
            }
            sink.emit(ConvertWarning::DroppedObsp {
                name: format!("{group_path}/{name}"),
                reason: "not a CSR sparse group or 2D dense dataset \
                         (CSC or unsupported pairwise layout)"
                    .to_string(),
            });
        }
    }

    // CSR sparse members.
    for info in &sparse_infos {
        let n_total = info.n_rows as u64;
        // Zero-row sparse mappings: emit a single empty shard so the
        // key survives round-trip (mirrors the override-path special
        // case in `partition_coo_to_shards`).
        if info.n_rows == 0 {
            let batch = read_sparse_mapping_shard(file, group_path, info, 0, 0)?;
            emit(writer, &info.name, 0, 0, 0, 0, &batch)?;
            continue;
        }
        let step = shard_target_rows.max(1) as usize;
        let mut shard_idx = 0u32;
        let mut row_start = 0usize;
        while row_start < info.n_rows {
            let row_end = (row_start + step).min(info.n_rows);
            let batch = read_sparse_mapping_shard(file, group_path, info, row_start, row_end)?;
            let n_shard_rows = (row_end - row_start) as u64;
            emit(
                writer,
                &info.name,
                shard_idx,
                row_start as u64,
                n_shard_rows,
                n_total,
                &batch,
            )?;
            row_start = row_end;
            shard_idx += 1;
        }
    }

    // Dense members → COO via the existing dense row-shard reader. Each
    // shard's nonzeros are emitted as the same COO `RecordBatch` shape
    // the CSR path produces, so the SCX `ObspEmbeddingShard` /
    // `VarpEmbeddingShard` reader and the h5ad exporter need no changes.
    // A dense `/obsp` therefore re-exports as a sparse matrix (values
    // preserved). Per-shard memory stays bounded to one row-range.
    for info in &dense_infos {
        let n_total = info.n_rows as u64;
        if info.n_rows == 0 {
            let batch = read_dense_mapping_shard(file, group_path, &info.name, 0, 0)?;
            let coo = dense_shard_to_coo(&batch, 0, 0)?;
            emit(writer, &info.name, 0, 0, 0, 0, &coo)?;
            continue;
        }
        let step = shard_target_rows.max(1) as usize;
        let mut shard_idx = 0u32;
        let mut row_start = 0usize;
        while row_start < info.n_rows {
            let row_end = (row_start + step).min(info.n_rows);
            let batch = read_dense_mapping_shard(file, group_path, &info.name, row_start, row_end)?;
            let coo = dense_shard_to_coo(&batch, row_start as u64, info.n_rows)?;
            let n_shard_rows = (row_end - row_start) as u64;
            emit(
                writer,
                &info.name,
                shard_idx,
                row_start as u64,
                n_shard_rows,
                n_total,
                &coo,
            )?;
            row_start = row_end;
            shard_idx += 1;
        }
    }
    Ok(())
}

/// Convert one dense mapping row-shard (columns `"0".."{k-1}"`, Float32,
/// as produced by [`read_dense_mapping_shard`]) into the COO
/// `RecordBatch` shape the obsp/varp shard writers expect (`row: Int32`,
/// `col: Int32`, `data: Float32` + `n_rows` / `n_cols` schema metadata).
///
/// Only nonzero entries are emitted. `row_start` is the global row offset
/// of this shard; COO `row` values are global (not shard-local), matching
/// [`read_sparse_mapping_shard`]. `n_rows_total` is the logical row count
/// of the full pairwise matrix; `n_cols` is taken from the shard's column
/// count.
fn dense_shard_to_coo(
    batch: &RecordBatch,
    row_start: u64,
    n_rows_total: usize,
) -> Result<RecordBatch, ConvertError> {
    use arrow::array::{Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    let n_cols = batch.num_columns();
    let n_local = batch.num_rows();

    let cols: Vec<&Float32Array> = (0..n_cols)
        .map(|c| {
            batch
                .column(c)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    ConvertError::Other("dense mapping shard column is not Float32".to_string())
                })
        })
        .collect::<Result<_, _>>()?;

    let mut rows: Vec<i32> = Vec::new();
    let mut col_idx: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for local in 0..n_local {
        let global_row = row_start + local as u64;
        let r_i32 = i32::try_from(global_row).map_err(|_| {
            ConvertError::Other(format!(
                "dense pairwise row index {global_row} exceeds i32::MAX \
                 (Arrow COO uses i32 row indices)"
            ))
        })?;
        for (c, arr) in cols.iter().enumerate() {
            let v = arr.value(local);
            if v != 0.0 {
                rows.push(r_i32);
                col_idx.push(c as i32);
                data.push(v);
            }
        }
    }

    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n_rows_total.to_string()),
            ("n_cols".to_string(), n_cols.to_string()),
        ]),
    ));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(rows)),
            Arc::new(Int32Array::from(col_idx)),
            Arc::new(Float32Array::from(data)),
        ],
    )?)
}

/// Slice an in-memory COO `RecordBatch` into row-aligned shards and
/// invoke `f` once per shard. Used only on the override path — the
/// disk-streaming path reads pre-sliced shards directly.
fn partition_coo_to_shards<F>(
    batch: &RecordBatch,
    shard_target_rows: u32,
    mut f: F,
) -> Result<(), ConvertError>
where
    F: FnMut(u32, u64, u64, u64, &RecordBatch) -> Result<(), ConvertError>,
{
    use arrow::array::{Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    let row_arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| ConvertError::Other("sparse override: column 0 must be Int32".into()))?;
    let col_arr = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| ConvertError::Other("sparse override: column 1 must be Int32".into()))?;
    let data_arr = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| ConvertError::Other("sparse override: column 2 must be Float32".into()))?;

    let metadata = batch.schema_ref().metadata().clone();
    let n_rows: usize = metadata
        .get("n_rows")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            ConvertError::Other("sparse override: schema metadata missing 'n_rows'".into())
        })?;
    let n_cols: usize = metadata
        .get("n_cols")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            ConvertError::Other("sparse override: schema metadata missing 'n_cols'".into())
        })?;

    let step = shard_target_rows.max(1) as usize;
    if n_rows == 0 {
        let sub = batch.slice(0, 0);
        f(0, 0, 0, 0, &sub)?;
        return Ok(());
    }

    let nnz = row_arr.len();
    let row_values = row_arr.values();
    let col_values = col_arr.values();
    let data_values = data_arr.values();

    // Group COO triples by shard via a single linear pass; the override
    // batch may be unsorted by row, so we bucket into per-shard Vecs.
    let n_shards = n_rows.div_ceil(step);
    let mut buckets_row: Vec<Vec<i32>> = (0..n_shards).map(|_| Vec::new()).collect();
    let mut buckets_col: Vec<Vec<i32>> = (0..n_shards).map(|_| Vec::new()).collect();
    let mut buckets_data: Vec<Vec<f32>> = (0..n_shards).map(|_| Vec::new()).collect();
    for i in 0..nnz {
        let r = row_values[i];
        if r < 0 {
            return Err(ConvertError::Other(format!(
                "sparse override: negative row index {r}"
            )));
        }
        let shard = (r as usize) / step;
        if shard >= n_shards {
            return Err(ConvertError::Other(format!(
                "sparse override: row {r} exceeds n_rows={n_rows}"
            )));
        }
        buckets_row[shard].push(r);
        buckets_col[shard].push(col_values[i]);
        buckets_data[shard].push(data_values[i]);
    }

    let n_total = n_rows as u64;
    for shard_idx in 0..n_shards {
        let row_start = shard_idx * step;
        let n_shard_rows = step.min(n_rows - row_start);
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("row", DataType::Int32, false),
                Field::new("col", DataType::Int32, false),
                Field::new("data", DataType::Float32, false),
            ],
            HashMap::from([
                ("n_rows".to_string(), n_rows.to_string()),
                ("n_cols".to_string(), n_cols.to_string()),
            ]),
        ));
        let shard_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(std::mem::take(
                    &mut buckets_row[shard_idx],
                ))),
                Arc::new(Int32Array::from(std::mem::take(
                    &mut buckets_col[shard_idx],
                ))),
                Arc::new(Float32Array::from(std::mem::take(
                    &mut buckets_data[shard_idx],
                ))),
            ],
        )?;
        f(
            shard_idx as u32,
            row_start as u64,
            n_shard_rows as u64,
            n_total,
            &shard_batch,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_layer_shards(
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

#[cfg(test)]
mod default_framing_tests {
    use super::*;

    /// Phase C: the convert default frames at `DEFAULT_ROW_GROUP_ROWS` (256), so
    /// a plain `ConvertOptions::default()` yields an active framing config.
    #[test]
    fn default_convert_options_frame_at_256() {
        let opts = ConvertOptions::default();
        assert_eq!(
            opts.row_group_rows,
            Some(scx_format_io::DEFAULT_ROW_GROUP_ROWS)
        );
        assert_eq!(scx_format_io::DEFAULT_ROW_GROUP_ROWS, 256);
        let fc = opts.framing().expect("default must be framed");
        assert_eq!(fc.row_group_rows, 256);
        assert!(!fc.trial, "default codec stays auto, not compact-trial");
    }

    /// `row_group_rows = Some(0)` is the explicit unframed (v3) opt-out:
    /// `framing()` returns None so the pipeline keeps the legacy layout.
    #[test]
    fn zero_row_group_rows_opts_out_of_framing() {
        let opts = ConvertOptions {
            row_group_rows: Some(0),
            ..Default::default()
        };
        assert!(
            opts.framing().is_none(),
            "row_group_rows=0 must disable framing (unframed v3 opt-out)"
        );
    }
}
