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

use scx_format::modality::ModalityType;

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
    /// Phase 5b: detection bitmap was skipped on a shard because the
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
}

impl ConvertWarning {
    /// Stable category key used for per-category count aggregation
    /// and for the JSON summary written into provenance. The
    /// `static` lifetime keeps the map keys cheap.
    pub fn category(&self) -> &'static str {
        match self {
            Self::InferredEncoding { .. } => "inferred_encoding",
            Self::SkippedUnsKey { .. } => "skipped_uns_key",
            Self::MissingPresetIndexColumn { .. } => "missing_preset_index_column",
            Self::PresetNoColumnsMatched { .. } => "preset_no_columns_matched",
            Self::UnsupportedIndexColumn { .. } => "unsupported_index_column",
            Self::DroppedObsp { .. } => "dropped_obsp",
            Self::ModalityTypeInferred { .. } => "modality_type_inferred",
            Self::DenseSparsified { .. } => "dense_sparsified",
            Self::DuplicateCoordinatesMerged { .. } => "duplicate_coordinates_merged",
            Self::LayerSkipped { .. } => "layer_skipped",
            Self::PredicateIndexSkippedMultimodal { .. } => "predicate_index_skipped_multimodal",
            Self::BitmapSkipped { .. } => "bitmap_skipped",
            Self::Hdf5NotThreadsafe => "hdf5_not_threadsafe",
            Self::ReaderThreadsDerated { .. } => "reader_threads_derated",
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
