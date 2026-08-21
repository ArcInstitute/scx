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
