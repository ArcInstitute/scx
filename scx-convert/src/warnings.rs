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
    /// source DataFrame.
    MissingPresetIndexColumn { column: String },
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
            Self::UnsupportedIndexColumn { .. } => "unsupported_index_column",
            Self::DroppedObsp { .. } => "dropped_obsp",
            Self::ModalityTypeInferred { .. } => "modality_type_inferred",
            Self::DenseSparsified { .. } => "dense_sparsified",
            Self::DuplicateCoordinatesMerged { .. } => "duplicate_coordinates_merged",
            Self::LayerSkipped { .. } => "layer_skipped",
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
    pub fn log() -> Self {
        Self {
            on_warning: Box::new(|w| log::warn!("{:?}", w)),
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
