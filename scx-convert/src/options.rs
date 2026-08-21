//! Direction-specific conversion options (ORG-11.16-3).
//!
//! `IngestOptions` used to carry both directions' knobs, which meant an
//! export-only field could be *set* on an ingest call. Nothing in the type
//! could say otherwise, so a runtime helper policed it instead:
//! `reject_export_row_filter_on_import`, called at the top of all five ingest
//! entry points, existing only to turn a nonsensical combination into an error
//! string. Its own doc comment named the contract it was defending — "an option
//! you set always did something" — which is the contract §1.1 broke and §11.12
//! still breaks elsewhere in the tree.
//!
//! [`ExportOptions`] makes the distinction a type. The helper is gone: the
//! combination it rejected can no longer be spelled.
//!
//! **Two flat structs, not a shared `CommonOptions`.** The four fields both
//! directions read are declared twice, which is a real cost and a deliberate
//! one. Nesting them would have added ~83 edit sites, 39 of which are struct
//! literals ending in `..Default::default()` — and an initializer dropped at
//! one of those still *compiles*, silently reverting a field to its default.
//! Nesting would have created 39 chances for that failure; flat creates none.
//! The drift nesting actually prevents — the shared defaults disagreeing — is
//! bought instead by
//! `convert_tests_options_split::shared_option_defaults_agree_across_directions`,
//! for six lines.

use crate::pipeline::codec_selection_json;
use crate::GroupPass;
use scx_codec::CodecId;
use scx_format_io::modality::ModalityType;
use scx_format_io::{BitmapPolicy, CscPolicy, ObsShardPolicy};

/// Options for the SCX → h5ad / h5mu export directions.
///
/// The split's central claim is that an export-only option can no longer be
/// *set* on an ingest call. That is a compile-time property, so no runtime test
/// can assert it — only a `compile_fail` doctest can:
///
/// ```compile_fail
/// # use scx_convert::IngestOptions;
/// // `export_min_counts` is not a field of the ingest options.
/// let _ = IngestOptions {
///     export_min_counts: Some(5.0),
///     ..Default::default()
/// };
/// ```
///
/// ⚠️ A lone `compile_fail` block is an assertion that cannot fail for the
/// right reason: it passes on *any* compile error, including a typo in the
/// field name or a wrong import. The positive half is what makes it mean
/// something — same field, same spelling, on the type that does have it:
///
/// ```
/// # use scx_convert::ExportOptions;
/// let opts = ExportOptions {
///     export_min_counts: Some(5.0),
///     ..Default::default()
/// };
/// assert!(opts.has_export_row_filter());
/// ```
///
///
/// Deliberately **no** `Debug` derive: `export_obs_keep_mask` holds one bool
/// per cell in the global obs row space, and a `{:?}` of an atlas-scale mask on
/// an error path is a denial of service. `IngestOptions` does not derive it
/// either.
#[derive(Clone)]
pub struct ExportOptions {
    /// Provenance tool name recorded in the exported `uns["scx_export"]`.
    /// Shared with [`crate::IngestOptions::tool`].
    pub tool: String,
    /// Byte budget for the export reader pool. Shared with
    /// [`crate::IngestOptions::memory_budget`].
    pub memory_budget: Option<u64>,
    /// Export-side shard-decode worker count; `None` resolves to
    /// `RAYON_NUM_THREADS` or `available_parallelism`. Shared with
    /// [`crate::IngestOptions::reader_threads`].
    pub reader_threads: Option<usize>,
    /// Backpressure window between the decode pool and the HDF5 writer.
    /// Shared with [`crate::IngestOptions::writer_queue_depth`].
    pub writer_queue_depth: usize,
    /// Caller-supplied obs keep mask, in the **global / physical** obs row
    /// space — its length must equal the file header's `n_obs`, not the
    /// post-deletion live count. Same coordinate system as
    /// [`scx_format_io::ScxReader::deletion_keep_mask`] and
    /// [`crate::min_counts_obs_mask`].
    ///
    /// Intersected (AND) with the deletion-vector mask, never substituted for
    /// it: a logically deleted row stays dropped regardless of its entry here.
    ///
    /// `Arc<[bool]>` so `Clone` stays O(1) on atlas-scale masks.
    pub export_obs_keep_mask: Option<std::sync::Arc<[bool]>>,
    /// Keep only obs rows whose total X counts are `>= n`.
    pub export_min_counts: Option<f64>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions {
            tool: "scx".into(),
            memory_budget: None,
            reader_threads: None,
            writer_queue_depth: 4,
            export_obs_keep_mask: None,
            export_min_counts: None,
        }
    }
}

impl ExportOptions {
    /// True when any caller-supplied export row filter is set.
    ///
    /// The export path uses it to decide whether to record filter provenance.
    /// It no longer has a second job: on `IngestOptions` this also answered
    /// "should an ingest entry point refuse this?", a question the split
    /// deletes rather than answers.
    pub fn has_export_row_filter(&self) -> bool {
        self.export_obs_keep_mask.is_some() || self.export_min_counts.is_some()
    }
}

#[derive(Clone)]
pub struct IngestOptions {
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
    /// Whether ingest emits the obs table as row-sharded
    /// [`scx_format_io::section::SectionType::ObsMetadataShard`] sections
    /// rather than one legacy `ObsMetadata` section.
    ///
    /// The same tri-state `scx optimize --shard-obs` and
    /// `pyscx.optimize(shard_obs=)` take, on the same threshold
    /// `pyscx.from_anndata` uses: `Auto` (default) shards when
    /// `n_obs > shard_target_rows`, `Always` shards unconditionally, `Off`
    /// keeps the single section. Applies to the **obs** axis only — var stays
    /// a single section on every ingest path.
    pub obs_shard_policy: ObsShardPolicy,
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

impl IngestOptions {
    /// Build the row-group [`scx_format_io::FramingConfig`] for the shard
    /// emitters, or `None`
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

    /// [`Self::framing`] with codec re-selection switched **off**, for a pass
    /// that must preserve each shard's existing codec rather than pick one.
    ///
    /// The CSC sidecar rebuild is the case: `scx_ops::rebuild_csc_inplace`
    /// re-writes every CSR shard at the codec read off that shard's own header,
    /// and `decode_target: Some(_)` authorises `write_shard_inner` to override
    /// it (see `FramingConfig`'s contract). Passing `framing()` there was
    /// harmless only while `write_shard_inner` ignored the field; now that it
    /// honours it, the sidecar rebuild would re-decide a codec the just-written
    /// X had already settled. Same rule as `scx-cli`'s
    /// `cli_utils::framing_for_file`, expressed locally because `scx-convert`
    /// does not depend on the CLI.
    pub fn framing_preserving_codec(&self) -> Option<scx_format_io::FramingConfig> {
        self.framing().map(|f| scx_format_io::FramingConfig {
            trial: false,
            decode_target: None,
            ..f
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

impl Default for IngestOptions {
    fn default() -> Self {
        IngestOptions {
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
            obs_shard_policy: ObsShardPolicy::default(),
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

#[cfg(test)]
mod default_framing_tests {
    use super::*;

    /// Phase C: the convert default frames at `DEFAULT_ROW_GROUP_ROWS` (256), so
    /// a plain `IngestOptions::default()` yields an active framing config.
    #[test]
    fn default_convert_options_frame_at_256() {
        let opts = IngestOptions::default();
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
        let opts = IngestOptions {
            row_group_rows: Some(0),
            ..Default::default()
        };
        assert!(
            opts.framing().is_none(),
            "row_group_rows=0 must disable framing (unframed v3 opt-out)"
        );
    }
}
