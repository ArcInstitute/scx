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
