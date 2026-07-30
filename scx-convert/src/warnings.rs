// Structured warning channel for conversion pipelines.
//
// Replaces ad-hoc `eprintln!` / `log::warn!` calls across
// `h5ad_to_scx`, `h5ad_to_scx_streaming`, `h5mu_to_scx`, and
// `scx_to_h5ad` with a typed channel. CLI summarises per-category
// counts at the end of the command; pyscx forwards each warning to
// `warnings.warn(...)` as `UserWarning`. A per-category count summary
// is folded into the provenance entry so the recipient of the .scx
// file can see which classes of issue were tolerated during
// conversion.

use std::collections::BTreeMap;
use std::fmt;

use scx_format_io::modality::ModalityType;

/// One conversion-time warning. New variants land alongside the
/// phase that emits them; the default `log::warn!` backend prints
/// the `Debug` form so adding a variant is non-breaking.
#[derive(Debug)]
pub enum ConvertWarning {
    /// h5ad layout was inferred from group children because the
    /// `encoding-type` attribute was absent or unknown.
    InferredEncoding { path: String, inferred: String },
    /// An `uns` entry could not be represented and was skipped.
    SkippedUnsKey { key: String, reason: String },
    /// A `uns` entry stored as a pandas DataFrame (`encoding-type ==
    /// "dataframe"`) was preserved as a nested dict (per-column values +
    /// `_index`) rather than reconstructed as a DataFrame. The column data
    /// survives, but column order and per-column categorical dtypes are not
    /// restored on read. Surfaces the structure loss so it is never silent.
    FlattenedUnsDataframe { key: String },
    /// A `uns` entry stored as a scipy-sparse matrix (`encoding-type` ==
    /// `"csr_matrix"` / `"csc_matrix"` / `"coo_matrix"`) was preserved as a
    /// nested dict of `data` / `indices` / `indptr` arrays rather than
    /// reconstructed as a sparse matrix — the sparse type tag is not
    /// restored on read. Surfaces the type loss so it is never silent.
    FlattenedUnsSparse { key: String, format: String },
    /// A column requested via `--index-preset` was missing from the
    /// source DataFrame. Emitted for partial preset/file mismatch only —
    /// when EVERY preset column is missing, the convert layer batches
    /// the burst into a single `PresetNoColumnsMatched` warning with
    /// actionable copy.
    MissingPresetIndexColumn { column: String },
    /// `--index-preset <preset>` matched **none** of the source obs/var
    /// columns. The user almost certainly picked the wrong preset for
    /// the input format, so we collapse the 10+ per-column warnings
    /// into a single actionable diagnosis pointing at the fix.
    PresetNoColumnsMatched {
        preset: String,
        axis: String,
        missing: Vec<String>,
    },
    /// A column was eligible for indexing by name but unsupported by
    /// dtype, cardinality, or null density.
    UnsupportedIndexColumn { column: String, reason: String },
    /// An `obsp` / `varp` entry was dropped (e.g. unsupported dtype).
    DroppedObsp { name: String, reason: String },
    /// An obs/var DataFrame column could not be read and was skipped
    /// (unsupported encoding-type, read error, or malformed group). The
    /// column is absent from the converted output. Replaces the prior
    /// `eprintln!` so Python callers can intercept via `warnings.warn`
    /// and CLI callers get a machine-readable per-category count.
    SkippedColumn {
        group: String,
        name: String,
        reason: String,
    },
    /// An `obsm` / `varm` embedding could not be read and was skipped.
    /// Replaces the prior `eprintln!` for the same reasons as
    /// [`Self::SkippedColumn`].
    SkippedObsm { name: String, reason: String },
    /// A modality's type was inferred (from var/obs schema or layer
    /// presence) rather than being declared in the source file.
    ModalityTypeInferred {
        name: String,
        modality_type: ModalityType,
    },
    /// A dense matrix was sparsified during streaming. `density` is
    /// the observed fraction of nonzero values.
    DenseSparsified { path: String, density: f32 },
    /// Duplicate (row, col) coordinates were merged.
    DuplicateCoordinatesMerged { count: u64, policy: String },
    /// A layer was skipped during streaming (open failed, shape
    /// mismatch, or width exceeds u32::MAX).
    LayerSkipped { name: String, reason: String },
    /// Predicate index flags were passed on a multimodal input but the
    /// engine read-side is unimodal-only today — the indexes were
    /// skipped. Lifted once `QueryPipeline` grows per-modality
    /// predicate-index lookup (Phase 6 / follow-on).
    PredicateIndexSkippedMultimodal { columns: Vec<String> },
    /// `csc='auto'` resolved to "build" for one or more modalities, but
    /// the streaming h5mu path cannot emit per-modality CSC sidecars (only
    /// the non-streaming `h5mu_to_scx` can, having each modality's full CSR
    /// in memory; `rebuild_csc_inplace` is unimodal-only and would corrupt
    /// a multimodal file). The sidecar was skipped — re-run with
    /// `stream=False` to build it. Explicit `csc='always'` is rejected with
    /// an error instead of being downgraded to this warning.
    CscSkippedStreamingMultimodal { modalities: Vec<String> },
    /// Detection bitmap was skipped on a shard because the
    /// `--bitmap=auto` policy rejected it (density not sparse,
    /// `n_vars` exceeds the cap, or estimated bitmap size > 15 % of
    /// the CSR shard).
    BitmapSkipped {
        modality: Option<String>,
        reason: String,
    },
    /// libhdf5 was built without `--enable-threadsafe`, so
    /// the parallel streaming reader fell back to the sequential
    /// coordinator. Functional output is unchanged. Emitted at most
    /// once per process (see
    /// [`crate::hdf5_threadsafe::try_emit_not_threadsafe_warning`]) —
    /// the threadsafe flag is a build-time property of libhdf5 and
    /// cannot change within a process, so repeating the warning per
    /// matrix / per modality is pure noise.
    Hdf5NotThreadsafe,
    /// The requested `reader_threads` was derated to fit
    /// within `memory_budget`. Functional output is unchanged.
    ReaderThreadsDerated {
        requested: usize,
        granted: usize,
        reason: String,
    },
    /// A caller-supplied export row filter (`export_obs_keep_mask` /
    /// `export_min_counts`) is active, but this section is read eagerly and
    /// filtered afterwards rather than streamed. The output is correct; the
    /// filter simply does not bound peak RSS for this section the way it does
    /// for `/X`, `obs`, and `/layers`. Matters most on raw-droplet files,
    /// which are exactly the ones a row filter targets.
    ExportFilterSectionEager { section: &'static str },
    /// In-memory `from_anndata` ingest detected an obsm / varm / obsp /
    /// varp key whose estimated peak footprint exceeds `memory_budget`.
    /// The shard is still written (no derating); the warning surfaces so
    /// the caller can shrink `shard_size` or route through the streaming
    /// path for files of this scale.
    MappingPeakFootprintHigh {
        key: String,
        axis: &'static str,
        estimated_bytes: u64,
        budget_bytes: u64,
    },
    /// Eager `to_anndata()` estimated that assembling the full X matrix
    /// will exceed `memory_budget`. The assembly still proceeds; the
    /// warning recommends `to_anndata(backed=True)` or
    /// `pyscx.open(path).query()` for atlas-scale files.
    EagerAssemblyMemoryHigh {
        estimated_bytes: u64,
        budget_bytes: u64,
    },
    /// SCX → h5ad export coerced null entries in an obs/var column to a
    /// sentinel value (`0` / `""`) because the column cannot carry a null
    /// mask. Regular integer / string columns now round-trip losslessly
    /// via anndata's `nullable-integer` / `nullable-string-array` group
    /// encodings, and floats use `NaN`; `nullable-boolean` and
    /// `categorical` columns preserve null state too. This warning is
    /// therefore limited to the pandas index column (`_index`), which
    /// anndata requires to be a plain dataset and so can never use a
    /// nullable group — its nulls (virtually never present) coerce to
    /// `0` / `""`. The column's non-null values are written faithfully.
    CoercedNulls {
        column: String,
        dtype: String,
        count: u64,
    },
    /// A dataframe column type is not supported by the SCX → h5ad
    /// writer and was skipped (no dataset written, name excluded from
    /// `column-order`). Replaces the prior `eprintln!` so Python
    /// callers can intercept via `warnings.warn` and CLI callers get a
    /// machine-readable per-category count.
    UnsupportedExportColumn { column: String, dtype: String },
    /// The `adata.raw` matrix was present but dropped from the
    /// reconstructed AnnData because the current mode cannot reproduce
    /// its obs-axis filtering (deletion vectors active, obs-filtered
    /// query, or backed mode). The on-disk raw sections are preserved;
    /// only this particular reconstruction omits raw.
    DroppedRaw { raw_n_vars: usize },
    /// `--group-target-bytes` (byte-budget grouped sharding) was
    /// requested on a dense or CSC-on-disk `/X`, which has no cheap per-row nnz
    /// to size shards by encoded width. The convert fell back to row-count
    /// grouping (`--shard-size` rows per shard). Re-export X as CSR for
    /// byte-budget grouping.
    GroupByteModeUnsupported { source_format: String },
}

impl ConvertWarning {
    /// Stable category key used for per-category count aggregation
    /// and for the JSON summary written into provenance. The
    /// `static` lifetime keeps the map keys cheap.
    pub fn category(&self) -> &'static str {
        match self {
            Self::InferredEncoding { .. } => "inferred_encoding",
            Self::SkippedUnsKey { .. } => "skipped_uns_key",
            Self::FlattenedUnsDataframe { .. } => "flattened_uns_dataframe",
            Self::FlattenedUnsSparse { .. } => "flattened_uns_sparse",
            Self::MissingPresetIndexColumn { .. } => "missing_preset_index_column",
            Self::PresetNoColumnsMatched { .. } => "preset_no_columns_matched",
            Self::UnsupportedIndexColumn { .. } => "unsupported_index_column",
            Self::DroppedObsp { .. } => "dropped_obsp",
            Self::SkippedColumn { .. } => "skipped_column",
            Self::SkippedObsm { .. } => "skipped_obsm",
            Self::ModalityTypeInferred { .. } => "modality_type_inferred",
            Self::DenseSparsified { .. } => "dense_sparsified",
            Self::DuplicateCoordinatesMerged { .. } => "duplicate_coordinates_merged",
            Self::LayerSkipped { .. } => "layer_skipped",
            Self::PredicateIndexSkippedMultimodal { .. } => "predicate_index_skipped_multimodal",
            Self::CscSkippedStreamingMultimodal { .. } => "csc_skipped_streaming_multimodal",
            Self::BitmapSkipped { .. } => "bitmap_skipped",
            Self::Hdf5NotThreadsafe => "hdf5_not_threadsafe",
            Self::ReaderThreadsDerated { .. } => "reader_threads_derated",
            Self::ExportFilterSectionEager { .. } => "export_filter_section_eager",
            Self::MappingPeakFootprintHigh { .. } => "mapping_peak_footprint_high",
            Self::EagerAssemblyMemoryHigh { .. } => "eager_assembly_memory_high",
            Self::CoercedNulls { .. } => "coerced_nulls",
            Self::UnsupportedExportColumn { .. } => "unsupported_export_column",
            Self::DroppedRaw { .. } => "dropped_raw",
            Self::GroupByteModeUnsupported { .. } => "group_byte_mode_unsupported",
        }
    }
}

impl fmt::Display for ConvertWarning {
    /// Human-readable rendering used by `WarningSink::log()`. The
    /// special-cased variant is `PresetNoColumnsMatched`, which earns a
    /// single actionable sentence; every other variant falls back to
    /// the derived `Debug` shape (preserves the prior log format).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PresetNoColumnsMatched {
                preset,
                axis,
                missing,
            } => {
                let preview_count = missing.len().min(3);
                let preview = missing[..preview_count].join(", ");
                let suffix = if missing.len() > preview_count {
                    ", ..."
                } else {
                    ""
                };
                write!(
                    f,
                    "--index-preset {preset} expects {n} {axis} columns ({preview}{suffix}) \
                     but the input file has none of them. Drop --index-preset, or use \
                     --index-{axis} <col>,... to pick existing columns.",
                    n = missing.len(),
                )
            }
            Self::FlattenedUnsDataframe { key } => write!(
                f,
                "uns['{key}'] is a pandas DataFrame; it was preserved as a nested \
                 dict (per-column values + `_index`) but not reconstructed as a \
                 DataFrame — column order and per-column categorical dtypes are not \
                 restored on read."
            ),
            Self::FlattenedUnsSparse { key, format } => write!(
                f,
                "uns['{key}'] is a scipy-sparse {format}; its data/indices/indptr \
                 arrays were preserved as a nested dict but the sparse type is not \
                 reconstructed on read — it comes back as a dict. Rebuild with e.g. \
                 scipy.sparse.{format}((data, indices, indptr), shape=shape)."
            ),
            Self::DroppedRaw { raw_n_vars } => write!(
                f,
                "adata.raw ({raw_n_vars} genes) was present but dropped from this \
                 reconstruction; the mode in use (obs-filtered query, backed mode, or \
                 deletion-vectors-active file) cannot reproduce raw's obs-axis filtering. \
                 The on-disk raw sections are preserved."
            ),
            Self::EagerAssemblyMemoryHigh {
                estimated_bytes,
                budget_bytes,
            } => {
                let gib = 1024.0 * 1024.0 * 1024.0;
                let est_gb = *estimated_bytes as f64 / gib;
                let budget_gb = *budget_bytes as f64 / gib;
                write!(
                    f,
                    "estimated host assembly ~{est_gb:.1} GB exceeds the {budget_gb:.1} GB \
                     budget; proceeding (peak host RSS may be high). Pass a smaller \
                     var_names / obs_filter subset, open with backed=True, or raise \
                     memory_budget to reduce it."
                )
            }
            other => write!(f, "{other:?}"),
        }
    }
}

/// Sink for `ConvertWarning`s. Each emission updates the per-
/// category count and invokes the configured backend.
///
/// Default backend is `log::warn!`; CLI and pyscx swap in their own
/// closures (eprintln summary / `warnings.warn` respectively).
pub struct WarningSink {
    on_warning: Box<dyn FnMut(&ConvertWarning) + Send>,
    counts: BTreeMap<&'static str, u64>,
}

impl WarningSink {
    /// Build a sink that forwards each warning to `log::warn!`.
    /// Uses `Display`, which gives `PresetNoColumnsMatched` its
    /// actionable single-sentence rendering; every other variant
    /// falls through to the derived `Debug` shape via
    /// [`ConvertWarning`]'s `Display` impl.
    pub fn log() -> Self {
        Self {
            on_warning: Box::new(|w| log::warn!("{w}")),
            counts: BTreeMap::new(),
        }
    }

    /// Build a sink that forwards each warning to the supplied
    /// closure. The closure also gets the running count for that
    /// category (post-increment) so per-category rate-limiting is
    /// possible without keeping external state.
    pub fn with_handler<F>(handler: F) -> Self
    where
        F: FnMut(&ConvertWarning) + Send + 'static,
    {
        Self {
            on_warning: Box::new(handler),
            counts: BTreeMap::new(),
        }
    }

    /// Record a warning. Increments the per-category count and
    /// invokes the backend exactly once.
    pub fn emit(&mut self, w: ConvertWarning) {
        let key = w.category();
        *self.counts.entry(key).or_insert(0) += 1;
        (self.on_warning)(&w);
    }

    /// Per-category counts collected so far.
    pub fn counts(&self) -> &BTreeMap<&'static str, u64> {
        &self.counts
    }

    /// Total number of warnings emitted.
    pub fn total(&self) -> u64 {
        self.counts.values().copied().sum()
    }

    /// JSON summary suitable for embedding in `ProvenanceEntry`.
    /// Empty when nothing was emitted.
    pub fn summary_json(&self) -> serde_json::Value {
        let map: serde_json::Map<String, serde_json::Value> = self
            .counts
            .iter()
            .map(|(k, v)| ((*k).to_string(), serde_json::Value::from(*v)))
            .collect();
        serde_json::Value::Object(map)
    }
}

impl Default for WarningSink {
    fn default() -> Self {
        Self::log()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_increment_per_category() {
        let mut sink = WarningSink::log();
        sink.emit(ConvertWarning::SkippedUnsKey {
            key: "a".into(),
            reason: "bad dtype".into(),
        });
        sink.emit(ConvertWarning::SkippedUnsKey {
            key: "b".into(),
            reason: "bad dtype".into(),
        });
        sink.emit(ConvertWarning::LayerSkipped {
            name: "spliced".into(),
            reason: "csc".into(),
        });
        assert_eq!(sink.counts().get("skipped_uns_key"), Some(&2));
        assert_eq!(sink.counts().get("layer_skipped"), Some(&1));
        assert_eq!(sink.total(), 3);
    }

    #[test]
    fn summary_json_shape() {
        let mut sink = WarningSink::log();
        sink.emit(ConvertWarning::InferredEncoding {
            path: "/X".into(),
            inferred: "csr_matrix".into(),
        });
        let s = sink.summary_json();
        assert_eq!(s["inferred_encoding"], serde_json::Value::from(1u64));
    }

    #[test]
    fn preset_no_columns_matched_display_includes_actionable_text() {
        let w = ConvertWarning::PresetNoColumnsMatched {
            preset: "cellxgene".into(),
            axis: "obs".into(),
            missing: vec![
                "cell_type".into(),
                "cell_type_ontology_term_id".into(),
                "tissue".into(),
                "tissue_ontology_term_id".into(),
            ],
        };
        let rendered = format!("{w}");
        assert!(rendered.contains("--index-preset cellxgene"), "{rendered}");
        assert!(rendered.contains("4 obs columns"), "{rendered}");
        assert!(
            rendered.contains("cell_type, cell_type_ontology_term_id, tissue"),
            "{rendered}"
        );
        assert!(rendered.contains(", ..."), "{rendered}");
        assert!(rendered.contains("Drop --index-preset"), "{rendered}");
        assert!(rendered.contains("--index-obs"), "{rendered}");
    }

    #[test]
    fn eager_assembly_memory_high_display_is_human_readable() {
        // Report E4: the warning must read as a sentence with GiB figures and
        // actionable advice, not a `{:?}`-formatted struct with raw byte counts.
        let w = ConvertWarning::EagerAssemblyMemoryHigh {
            estimated_bytes: 25_239_799_332,
            budget_bytes: 8_589_934_592,
        };
        let rendered = format!("{w}");
        assert!(rendered.contains("23.5 GB"), "{rendered}");
        assert!(rendered.contains("8.0 GB"), "{rendered}");
        assert!(rendered.contains("proceeding"), "{rendered}");
        assert!(rendered.contains("backed=True"), "{rendered}");
        // Must NOT leak the debug struct shape.
        assert!(!rendered.contains("EagerAssemblyMemoryHigh"), "{rendered}");
        assert!(!rendered.contains("estimated_bytes"), "{rendered}");
    }

    #[test]
    fn other_variants_display_falls_back_to_debug() {
        let w = ConvertWarning::MissingPresetIndexColumn {
            column: "feature_name".into(),
        };
        assert_eq!(format!("{w}"), format!("{w:?}"));
    }

    #[test]
    fn handler_receives_each_emission() {
        use std::sync::{Arc, Mutex};
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = Arc::clone(&log);
        let mut sink = WarningSink::with_handler(move |w| {
            log_clone.lock().unwrap().push(format!("{:?}", w));
        });
        sink.emit(ConvertWarning::LayerSkipped {
            name: "x".into(),
            reason: "y".into(),
        });
        assert_eq!(log.lock().unwrap().len(), 1);
    }
}
