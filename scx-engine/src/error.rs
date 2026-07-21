use thiserror::Error;

/// Errors produced by the SCX query engine.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("schema error on column '{column}': {reason}")]
    SchemaError { column: String, reason: String },

    #[error("predicate parse error in '{expr}': {reason}")]
    PredicateParseError { expr: String, reason: String },

    #[error("collect() called on empty pipeline (no file opened)")]
    EmptyPipeline,

    #[error("unknown index preset '{0}'; expected one of cellxgene, perturbseq, training")]
    UnknownIndexPreset(String),

    /// A grouped-read API (`read_group` / `read_reference` / `group_labels` /
    /// `iter_group_shards`) was called on an archive that has no `group_index`
    /// sidecar (not written with `--group-by`).
    #[error(
        "archive is not grouped: write it with `scx sort --group-by <col>` to enable grouped reads"
    )]
    NotGrouped,

    /// `read_group` was asked for a label not present in the group index.
    #[error("unknown group label '{label}'{}", suggestions_hint(.suggestions))]
    UnknownGroupLabel {
        label: String,
        suggestions: Vec<String>,
    },

    /// A modality-scoped query named a modality that is not registered in
    /// the file's modality table (or named one on a single-modality file).
    #[error("unknown modality '{requested}'{}", modality_hint(.available))]
    UnknownModality {
        requested: String,
        available: Vec<String>,
    },

    /// `query()` was called without a modality on a multimodal file, where
    /// there is no unambiguous default axis (`modality_id = 0` carries no
    /// CSR shards on a multimodal file).
    #[error(
        "file is multimodal; specify a modality (one of: {})",
        .available.join(", ")
    )]
    ModalityRequired { available: Vec<String> },

    /// A streaming rewrite (`streaming_preprocess` / `streaming_save_layer`)
    /// was called on an input whose layout it cannot faithfully reproduce
    /// (multimodal files, or files carrying `adata.raw`). Rather than silently
    /// drop or corrupt those sections, the operation refuses. See SCX-002.
    #[error("{op} does not support {feature} inputs: {remedy}")]
    UnsupportedRewrite {
        op: String,
        feature: String,
        remedy: String,
    },

    #[error(transparent)]
    FormatError(#[from] scx_format_io::ScxError),

    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error(transparent)]
    CsrError(#[from] scx_sparse::CsrError),

    /// Free-form error for builder/contract violations that don't
    /// warrant a dedicated variant. Used by
    /// [`crate::ObsPredicateIndexBuilder`] for shard-order /
    /// schema-shape checks where the caller is the offender.
    #[error("{0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;

/// Render `; did you mean: a, b, c?` for unknown-label errors (empty when there
/// are no close matches).
fn suggestions_hint(suggestions: &[String]) -> String {
    if suggestions.is_empty() {
        String::new()
    } else {
        format!("; did you mean: {}?", suggestions.join(", "))
    }
}

/// Render `; available: a, b` for unknown-modality errors (empty when the
/// modality table is absent — e.g. a modality named on a single-modality file).
fn modality_hint(available: &[String]) -> String {
    if available.is_empty() {
        String::new()
    } else {
        format!("; available: {}", available.join(", "))
    }
}
