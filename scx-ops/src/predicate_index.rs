//! Shared scaffolding for wiring predicate-index sections into the
//! `merge`, `append`, and `compact` rewrites.
//!
//! [`ObsVarIndexPass`] at the bottom is the rebuild itself — one resolution of
//! the user's `--index-*` request, one obs builder, one var write, and one
//! decision about where the per-shard `column_stats` go. The rest of this module
//! is the surrounding policy: what to warn about, what to validate before
//! committing, and how a caller learns what happened.
//!
//! These ops historically dropped `ObsPredicateIndex` / `VarPredicateIndex`
//! sections from their output, so query-time pushdown silently regressed
//! to full obs scans after every rewrite. The new `*_with_index_options`
//! entry points accept the same `ConversionPredicateIndexOptions` that
//! `scx convert` / `pyscx.from_anndata` already expose and produce a
//! [`PredicateIndexBuildSummary`] the caller can map onto its preferred
//! warning channel.
//!
//! Cross-crate warning routing is intentionally left to the caller:
//! `scx-ops` has no warning type of its own. It returns the raw outcomes
//! from [`scx_engine::build_and_write_conversion_predicate_indexes`] plus a
//! `multimodal_skip` slot on [`PredicateIndexBuildSummary`], and each caller
//! maps those onto its own channel — e.g. `scx-cli` / `pyscx` translate them
//! into `scx-convert`'s `ConvertWarning` variants. This keeps `scx-ops` free
//! of any dep on `scx-convert`.

use arrow::datatypes::Schema;
use scx_engine::{ConversionPredicateIndexOptions, ConversionPredicateIndexResult};

use crate::error::{OpsError, Result};

/// Outcome of attempting to build predicate indexes during a `merge`,
/// `append`, or `compact` rewrite. The caller decides how to surface
/// the per-axis outcomes to its user — e.g. `scx-convert`'s typed
/// `ConvertWarning`s, Python `warnings.warn(...)`, or CLI stderr.
#[derive(Debug, Default)]
pub struct PredicateIndexBuildSummary {
    /// Populated when the op actually invoked the engine builder. The
    /// `obs_outcomes` / `var_outcomes` fields carry per-column results
    /// that the caller maps to user-facing warnings (mirroring
    /// `scx-convert/src/pipeline/index.rs::process_predicate_index_outcomes`).
    pub result: Option<ConversionPredicateIndexResult>,
    /// Populated when the op was multimodal and the caller requested an
    /// index. Predicate-index sections are unimodal-only today (the
    /// `scx-format::writer` emitters take no `modality_id` and the
    /// `scx-engine` read-side ignores it). Multimodal merge / compact /
    /// append skip the write and hand back this column list so the caller
    /// can surface it as a single warning (e.g. `scx-convert`'s
    /// `PredicateIndexSkippedMultimodal { columns }`).
    pub multimodal_skip: Option<Vec<String>>,
}

impl PredicateIndexBuildSummary {
    /// Empty summary — the rewrite path was invoked without any
    /// `--index-*` kwarg and produced no predicate-index sections.
    pub fn skipped() -> Self {
        Self::default()
    }

    /// True when the request was non-trivial but no index was written
    /// because the output is multimodal. Callers should emit a single
    /// warning naming the skipped columns in that case (e.g.
    /// `scx-convert`'s `PredicateIndexSkippedMultimodal { columns }`).
    pub fn was_multimodal_skip(&self) -> bool {
        self.multimodal_skip.is_some()
    }
}

/// True when the user explicitly requested any predicate-index work —
/// a forced obs / var column, a preset, OR a non-zero auto-threshold.
/// The legacy `merge` / `compact` / `append` / `append_from_reader`
/// wrappers delegate with `index_auto_threshold = 0` as a sentinel that
/// disables both auto-detect and any multimodal-skip warning (preserves
/// pre-fix behaviour where the legacy entry points emit no predicate
/// index and no warning).
///
/// The single-modality rewrite path calls the engine unconditionally;
/// this helper only gates the multimodal-skip surface so callers that
/// passed the legacy `Default::default()` (auto_threshold = 1000) but
/// hit a multimodal target don't silently see a skip warning they
/// didn't ask for.
pub fn user_wants_index(options: &ConversionPredicateIndexOptions) -> bool {
    !options.index_obs.is_empty()
        || !options.index_var.is_empty()
        || options.index_preset.is_some()
        || options.index_auto_threshold > 0
}

/// Flatten the requested columns / preset into a single `Vec<String>`
/// for the multimodal-skip warning. Mirrors
/// `scx_convert::h5mu::pipeline::emit_multimodal_index_skip_warning`
/// so the user sees the same payload regardless of the rewrite op that
/// triggered the skip.
pub fn requested_columns(options: &ConversionPredicateIndexOptions) -> Vec<String> {
    let mut columns: Vec<String> = Vec::new();
    columns.extend(options.index_obs.iter().cloned());
    columns.extend(options.index_var.iter().cloned());
    if let Some(name) = options.index_preset.as_deref() {
        columns.push(format!("preset:{name}"));
    }
    columns
}

/// Validate that every `index_obs` / `index_var` column the caller forced
/// is present in the corresponding output schema. Called by each
/// `*_with_index_options` rewrite op BEFORE any committing I/O so that
/// missing-forced-column errors fail loudly without leaving a half-built
/// output on disk.
///
/// Engine outcome flow (after writes) still emits
/// `BuildOutcome::ForcedColumnError` in defensive paths — the rewrite ops
/// rely on this upfront check to make that path unreachable for
/// rewrites. See `pyscx::convert::build_and_write_predicate_indexes_inline`
/// and `scx-convert/src/pipeline/index.rs::process_predicate_index_outcomes` for the
/// equivalent fail-late paths that this duplicates as fail-fast.
pub fn validate_forced_columns(
    options: &ConversionPredicateIndexOptions,
    obs_schema: &Schema,
    var_schema: &Schema,
) -> Result<()> {
    let obs_missing: Vec<String> = options
        .index_obs
        .iter()
        .filter(|c| obs_schema.column_with_name(c).is_none())
        .cloned()
        .collect();
    let var_missing: Vec<String> = options
        .index_var
        .iter()
        .filter(|c| var_schema.column_with_name(c).is_none())
        .cloned()
        .collect();
    if obs_missing.is_empty() && var_missing.is_empty() {
        return Ok(());
    }
    let mut parts: Vec<String> = Vec::new();
    if !obs_missing.is_empty() {
        let available: Vec<String> = obs_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        parts.push(scx_engine::index::forced_columns_missing_message(
            "obs",
            &obs_missing,
            &available,
        ));
    }
    if !var_missing.is_empty() {
        let available: Vec<String> = var_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        parts.push(scx_engine::index::forced_columns_missing_message(
            "var",
            &var_missing,
            &available,
        ));
    }
    Err(OpsError::InvalidInput(parts.join("\n")))
}

// ---------------------------------------------------------------------------
// ObsVarIndexPass — the one predicate-index rebuild (ORG-6.14-2)
// ---------------------------------------------------------------------------

/// Where the per-shard `column_stats` derived from a freshly built obs index
/// should go.
///
/// **This is the one structural difference between the rewrite ops**, which is
/// why it is a parameter of [`ObsVarIndexPass::finish_obs`] rather than
/// something two callers work around. `scx_engine::apply_obs_shard_column_stats`
/// mutates the *writer's* pending catalog entries, and that is right for an op
/// that lets `ScxWriter::finish` assemble the catalog. `append` and
/// `modify_metadata` are in-place: they adopt a writer for the new sections,
/// take `into_in_place_parts()`, and then assemble a catalog by hand out of
/// carried-over old entries plus the new ones. The stats have to reach *that*
/// list, via `scx_format_io::assign_csr_shard_column_stats`, after this pass has
/// returned.
pub enum StatsSink<'a> {
    /// Apply straight to the writer's pending CSR shard entries. For ops whose
    /// output catalog is the writer's: `merge`, `compact`, `sort`, `optimize`,
    /// and the conversion pipelines.
    Writer,
    /// Hand the derived stats back instead of applying them, for the in-place
    /// ops that assemble their own catalog. `None` on return means no index
    /// bytes were produced — see [`ObsVarIndexPass::finish_obs`].
    Deferred(&'a mut Option<Vec<Vec<scx_format_io::catalog::ColumnStat>>>),
    /// Write the index but derive **no** stats, because the shard space it was
    /// finished over is not the CSR-shard space `column_stats` live in.
    ///
    /// The one caller is `modify_metadata` on a file whose CSR shards do not
    /// tile `[0, n_obs)`: it falls back to finishing over the *obs*-shard ranges
    /// so the index is still usable at Level 2, and deriving Level-1 stats from
    /// that would attribute one shard's values to another. It pairs this with
    /// clearing the stale stats and a `NoStatsReason` telling the user which of
    /// the three remedies applies, which is why the decision stays there rather
    /// than being inferred here.
    Skip,
}

/// One predicate-index rebuild, resolved once and reusable across both axes.
///
/// Replaces the four hand-rolled rebuild bodies in `append` (1), `merge` (2) and
/// `modify_metadata` (1), each of which independently resolved the preset,
/// re-spelled the `100_000` high-cardinality cap, and separately decided whether
/// to derive per-shard column stats — which is why only some of them did. §6.1
/// (stale `column_stats`) was the review's most dangerous finding and this is
/// its structural cause.
///
/// `compact`, `sort`, `optimize` and `scx-convert` reach the same resolution
/// through `scx_engine::build_and_write_conversion_predicate_indexes[_streaming]`,
/// which owns the whole-batch and shard-stream drivers. This type exists for the
/// ops that cannot use those drivers — because they interleave the obs shard
/// pushes with their own shard writes, or need the deferred stats sink above —
/// and it shares the resolution step with them
/// (`scx_engine::resolve_predicate_index_build_options`) so the two cannot drift.
pub struct ObsVarIndexPass {
    resolved: scx_engine::ResolvedIndexBuildOptions,
    request: ConversionPredicateIndexOptions,
}

impl ObsVarIndexPass {
    /// Resolve a user request. **Errors on an unknown `index_preset`**, before
    /// the caller has written anything — which is the point of resolving up
    /// front rather than per axis.
    pub fn resolve(options: &ConversionPredicateIndexOptions) -> Result<Self> {
        Ok(Self {
            resolved: scx_engine::resolve_predicate_index_build_options(options)
                .map_err(OpsError::Engine)?,
            request: options.clone(),
        })
    }

    /// The carry case: rebuild the index the file already had, over whatever
    /// the new metadata now contains.
    ///
    /// Carried columns enter as **preset**, never **forced**. A preset column
    /// the new frame no longer has is a `PresetSkipped` warning where a forced
    /// one is a hard error, and an ordinary `doublet_consensus` must not start
    /// raising on a file whose indexed column an earlier edit dropped.
    /// `auto_threshold` is pinned to 0 — belt and braces, since a non-empty
    /// `preset_columns` already disables auto-detection — so a future refactor
    /// cannot silently index columns the file never had.
    pub fn carried(obs_columns: &[String], var_columns: &[String]) -> Self {
        let opts = |columns: &[String]| scx_engine::PredicateIndexBuildOptions {
            forced_columns: Vec::new(),
            preset_columns: columns.to_vec(),
            auto_threshold: 0,
            high_cardinality_threshold: scx_engine::HIGH_CARDINALITY_THRESHOLD,
        };
        Self {
            resolved: scx_engine::ResolvedIndexBuildOptions {
                obs: opts(obs_columns),
                var: opts(var_columns),
            },
            // The carried case has no user request — the columns came off the
            // file — and `ConversionPredicateIndexOptions::default()` says
            // exactly that: nothing forced, no preset, auto-detect off.
            request: ConversionPredicateIndexOptions::default(),
        }
    }

    /// The request this pass was resolved from, for the callers that also have
    /// to report it (`user_wants_index`, `validate_forced_columns`, the
    /// provenance entry). Carrying it here is what keeps a rewrite op from
    /// having to thread the options alongside the pass and risk the two
    /// describing different requests.
    pub fn request(&self) -> &ConversionPredicateIndexOptions {
        &self.request
    }

    /// The resolved obs-axis options, for a caller that has to drive the
    /// builder itself.
    pub fn obs_options(&self) -> &scx_engine::PredicateIndexBuildOptions {
        &self.resolved.obs
    }

    /// A streaming obs builder over `schema`. Push shards into it with
    /// `push_shard_split`, then hand it to [`Self::finish_obs`].
    pub fn obs_builder(
        &self,
        schema: arrow::datatypes::SchemaRef,
    ) -> Result<scx_engine::ObsPredicateIndexBuilder> {
        scx_engine::ObsPredicateIndexBuilder::new(schema, &self.resolved.obs)
            .map_err(OpsError::Engine)
    }

    /// Finish the obs index over `shard_row_ranges`, write it, and route the
    /// per-shard column stats it implies to `sink`.
    ///
    /// Returns whether index bytes were produced. `false` means no column was
    /// indexable — and for [`StatsSink::Deferred`] it means the slot is left
    /// `None`, which is the signal the in-place ops use to decide whether to
    /// *clear* the stale stats instead. Building the write and the stats into
    /// one call is the point: a rebuild that writes the index and forgets the
    /// stats leaves Level-1 pruning excluding shards on values the new obs
    /// satisfies.
    pub fn finish_obs(
        &self,
        builder: scx_engine::ObsPredicateIndexBuilder,
        shard_row_ranges: &[(u64, u64)],
        writer: &mut scx_format_io::ScxWriter,
        result: &mut ConversionPredicateIndexResult,
        sink: StatsSink<'_>,
    ) -> Result<bool> {
        let obs_bytes = builder
            .finish(
                shard_row_ranges,
                &mut result.obs_outcomes,
                &mut result.obs_indexed_columns,
            )
            .map_err(OpsError::Engine)?;
        let Some(bytes) = obs_bytes else {
            return Ok(false);
        };
        writer.write_obs_predicate_index(&bytes)?;
        match sink {
            StatsSink::Writer => {
                scx_engine::apply_obs_shard_column_stats(writer, &bytes, shard_row_ranges.len())
                    .map_err(OpsError::Engine)?;
            }
            StatsSink::Deferred(slot) => {
                let index =
                    scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(&bytes))
                        .map_err(OpsError::Engine)?;
                *slot = Some(scx_engine::derive_shard_column_stats(
                    &index,
                    shard_row_ranges.len(),
                ));
            }
            StatsSink::Skip => {}
        }
        Ok(true)
    }

    /// Build the var index over the single `[(0, n_vars)]` range and write it.
    ///
    /// var stays on the batch-mode builder in every caller: gene metadata is
    /// small enough that shard streaming buys nothing, and there is no var
    /// equivalent of the per-shard `column_stats` Level-1 pruning reads.
    pub fn write_var(
        &self,
        var: &arrow::array::RecordBatch,
        n_vars: u64,
        writer: &mut scx_format_io::ScxWriter,
        result: &mut ConversionPredicateIndexResult,
    ) -> Result<()> {
        let var_row_ranges: [(u64, u64); 1] = [(0, n_vars)];
        let var_bytes = scx_engine::build_var_predicate_index_bytes(
            var,
            &var_row_ranges,
            &self.resolved.var,
            &mut result.var_outcomes,
            &mut result.var_indexed_columns,
        )?;
        if let Some(bytes) = var_bytes {
            writer.write_var_predicate_index(&bytes)?;
        }
        Ok(())
    }

    /// [`Self::write_var`] without a writer: the serialised index and the
    /// columns it actually covers.
    ///
    /// The in-place attaches need the *decision* before they open a writer,
    /// because a `dry_run` has to report the same index outcome the real
    /// import would produce — and which covered columns survive is only
    /// knowable from the new values (a covered string column overwritten by a
    /// float cannot be indexed). They then write these bytes rather than
    /// building a second time.
    ///
    /// `None` bytes mean nothing could be indexed, so the caller's stale
    /// section is *retired* rather than replaced — a different outcome from a
    /// rebuild, and one the planning boolean cannot predict.
    pub fn build_var_bytes(
        &self,
        var: &arrow::array::RecordBatch,
        n_vars: u64,
    ) -> Result<(Option<Vec<u8>>, Vec<String>)> {
        let var_row_ranges: [(u64, u64); 1] = [(0, n_vars)];
        let mut outcomes = Vec::new();
        let mut covered = Vec::new();
        let bytes = scx_engine::build_var_predicate_index_bytes(
            var,
            &var_row_ranges,
            &self.resolved.var,
            &mut outcomes,
            &mut covered,
        )?;
        Ok((bytes, covered))
    }
}
