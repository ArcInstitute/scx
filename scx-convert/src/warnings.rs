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
    /// One column of a `uns` pandas DataFrame had no lossless representation
    /// and was left out of the reconstructed frame. The rest of the frame —
    /// index, column order, every other column — is intact.
    ///
    /// Reached for anndata's nullable encodings (`nullable-integer`,
    /// `nullable-boolean`, `nullable-string-array`) and for any column dtype
    /// the `uns` envelope cannot spell. Ingest drops per column because there
    /// is no fallback that keeps the column *and* the frame — unlike export,
    /// which declines the whole frame precisely because its raw-envelope
    /// fallback keeps everything (see [`Self::UnsExportedAsRawEnvelope`]).
    ///
    /// Under `strict_uns=true` the column's error is returned instead, so a
    /// strict conversion cannot succeed with a truncated frame.
    UnsupportedUnsDataframeColumn {
        key: String,
        column: String,
        reason: String,
    },

    /// A `uns` pandas DataFrame could not be exported to h5ad as an anndata
    /// dataframe group and was written as a raw `__scx_type__` envelope
    /// subgroup instead — anndata reads that subgroup back as a nested dict, so
    /// DataFrame consumers (`sc.tl.filter_rank_genes_groups`,
    /// `sc.get.rank_genes_groups_df`) will not recognise it.
    ///
    /// The fallback keeps every value **whose key HDF5 can carry**. A column
    /// named `"/evil"` or `""` has no HDF5 member spelling anywhere, so the
    /// fallback drops it too, with its own [`Self::SkippedUnsKey`].
    ///
    /// Declining the whole frame rather than dropping the offending column is
    /// deliberate: on export the fallback preserves everything, so dropping a
    /// column would lose data the fallback would have kept. (Ingest has no such
    /// fallback, which is why it drops per column instead — see
    /// [`Self::UnsupportedUnsDataframeColumn`].)
    UnsExportedAsRawEnvelope { key: String, reason: String },
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
    /// for `/X`, `obs`, `/layers` and `/raw`.
    ///
    /// The remaining eager sections are `obsm` and `obsp`. `/raw` used to be
    /// the third and by far the largest — the reason this variant said it
    /// "matters most on raw-droplet files" — but it streams now, so a filtered
    /// export of one no longer emits this for raw.
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
    ///
    /// `estimated_bytes` covers **assembly** — `X`, `adata.raw` when it is
    /// assembled too, and the obs / var metadata — and nothing that reads the
    /// matrix afterwards. It is a floor on process RSS, not a job size; the
    /// message says so, because a figure an operator can size a node from is
    /// the one thing this warning cannot supply.
    EagerAssemblyMemoryHigh {
        estimated_bytes: u64,
        budget_bytes: u64,
        /// Stored nonzeros in `X` (the estimate's dominant term).
        nnz: u64,
        /// Bytes per value in the assembled matrix, from the caller's
        /// `data_dtype`. The message derives its per-nonzero figure from this
        /// and `index_bytes` rather than assuming `f32`: a `float64` read is
        /// 12 B/nnz below the int64 line and 16 above it, not 8 and 12.
        value_bytes: u64,
        /// Bytes per column index: `Some(4)` or `Some(8)` for a CSR read —
        /// scipy's choice, from `max(nnz, n_rows)`, not the caller's.
        ///
        /// `None` means the message must not claim a width. Two cases: a
        /// `container="dense"` request, which has no index array at all; and a
        /// file with **deletion vectors**, where `nnz` is physical and scipy
        /// decides from the live count, which the catalog cannot supply.
        index_bytes: Option<u64>,
        /// `true` when the caller asked for `container="dense"`, so the estimate
        /// is `n_obs × n_vars × value_bytes` and the nonzero count does not
        /// bound it.
        dense_request: bool,
        /// `n_obs × n_vars × value_bytes`, when `container="dense"` would be
        /// smaller than the **sparse `X` term alone** — not than
        /// `estimated_bytes`, which also carries `adata.raw` and the metadata,
        /// and raw is assembled either way. `None` when it would not be, when
        /// this call assembles no host CSR `X`, or when the product saturates.
        dense_bytes: Option<u64>,
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
    /// The source carries an `adata.raw` matrix that the file being written
    /// will NOT carry. Distinct from [`Self::DroppedRaw`], which is a
    /// read-side notice whose "the on-disk raw sections are preserved"
    /// reassurance is true of the source and says nothing about an output
    /// file — on this path raw is gone from the new file for good.
    ///
    /// `reason` is supplied per call site and carries the cause **and** the
    /// remedy, because the doors differ: the SCX → SCX rewrite loses raw
    /// because the in-memory AnnData does not hold it, while a
    /// reorder-on-convert loses it because raw is streamed unpermuted. A
    /// single hard-coded remedy would be wrong on one of them — telling a
    /// `from_h5ad(..., sort_by=…)` caller to "convert from the h5ad" is the
    /// same class of misdirection this variant exists to end.
    DroppedRawOnWrite {
        raw_n_vars: usize,
        reason: &'static str,
    },
    /// `adata.raw.varm` was present but is not representable: SCX's raw
    /// section family stores `raw/X` and `raw/var` only. Raw's own var-axis
    /// mappings are dropped on both h5ad ingest and the in-memory write.
    DroppedRawVarm { keys: usize },
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
            Self::UnsupportedUnsDataframeColumn { .. } => "unsupported_uns_dataframe_column",
            Self::UnsExportedAsRawEnvelope { .. } => "uns_exported_as_raw_envelope",
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
            Self::DroppedRawOnWrite { .. } => "dropped_raw_on_write",
            Self::DroppedRawVarm { .. } => "dropped_raw_varm",
            Self::GroupByteModeUnsupported { .. } => "group_byte_mode_unsupported",
        }
    }
}

/// `2650704199` -> `2,650,704,199`.
///
/// A ten-digit run is not a number a reader parses at a glance, and this one is
/// the term that explains the whole estimate.
fn group_thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
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
            Self::UnsupportedUnsDataframeColumn {
                key,
                column,
                reason,
            } => write!(
                f,
                "uns['{key}'] column '{column}' has no lossless uns encoding \
                 ({reason}); it was left out of the reconstructed DataFrame. The \
                 index, the column order and every other column are intact."
            ),
            Self::UnsExportedAsRawEnvelope { key, reason } => write!(
                f,
                "uns['{key}'] could not be written as an h5ad DataFrame ({reason}); \
                 it was exported as a raw envelope subgroup instead, which anndata \
                 reads back as a nested dict. Every value whose key HDF5 can carry \
                 is preserved; a key it cannot is dropped with its own \
                 skipped_uns_key warning."
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
            Self::DroppedRawOnWrite { raw_n_vars, reason } => write!(
                f,
                "the source carries an adata.raw matrix ({raw_n_vars} genes) that this write \
                 does not carry forward, so the output file will have no raw: {reason}"
            ),
            Self::DroppedRawVarm { keys } => write!(
                f,
                "adata.raw.varm carries {keys} key(s), which SCX's raw section family \
                 cannot store (it holds raw/X and raw/var only) — they are dropped. \
                 raw.X and raw.var are unaffected; move anything you need onto \
                 raw.var columns, or keep it in adata.varm."
            ),
            Self::EagerAssemblyMemoryHigh {
                estimated_bytes,
                budget_bytes,
                nnz,
                value_bytes,
                index_bytes,
                dense_request,
                dense_bytes,
            } => {
                let gib = 1024.0 * 1024.0 * 1024.0;
                let est = *estimated_bytes as f64 / gib;
                let budget = *budget_bytes as f64 / gib;
                // Why it is that big: the index array is two thirds of a wide
                // matrix's footprint and nothing else surfaces that.
                let n = group_thousands(*nnz);
                // Every figure here is derived from the plan the estimate was
                // priced with. Hard-coding f32/i32 widths made the message
                // contradict its own total on any other request: a float64 CSR
                // is 12 B/nnz below the int64 line and 16 above, not 8 and 12.
                let why = match (*dense_request, *index_bytes) {
                    (true, _) => format!(
                        " A dense container is n_obs x n_vars x {value_bytes} B, so the {n} \
                         stored nonzeros do not bound it."
                    ),
                    (false, Some(ix)) => {
                        let per = value_bytes.saturating_add(ix);
                        let wide = if ix == 8 {
                            ", the larger half, because above 2^31 nonzeros scipy holds int64 \
                             column indices"
                        } else {
                            ""
                        };
                        format!(
                            " {n} nonzeros at {per} B each: {value_bytes} B of value and {ix} B \
                             of column index{wide}."
                        )
                    }
                    (false, None) => format!(
                        " {n} stored nonzeros at {value_bytes} B of value each, plus a column \
                         index whose width scipy picks from the count after deletions — a \
                         count the catalog cannot supply, so this figure assumes the wider one."
                    ),
                };
                let dense = match dense_bytes {
                    Some(b) => format!(
                        " container=\"dense\" would be {:.1} GiB here;",
                        *b as f64 / gib
                    ),
                    None => String::new(),
                };
                write!(
                    f,
                    "estimated host assembly ~{est:.1} GiB exceeds the {budget:.1} GiB budget; \
                     proceeding (peak host RSS may be high). That estimate covers assembly \
                     only — X, adata.raw when the read includes it, and obs/var — and excludes \
                     everything that reads the matrix afterwards, so it is a floor on process \
                     RSS, not a job size.{why} \
                     Cheaper routes: open with backed=True, which streams and assembles \
                     nothing;{dense} or pass a smaller var_names / obs_filter subset. Raising \
                     memory_budget silences this without changing what it costs."
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
            nnz: 1_000_000_000,
            value_bytes: 4,
            index_bytes: Some(4),
            dense_request: false,
            dense_bytes: None,
        };
        let rendered = format!("{w}");
        // The figures are GiB and are labelled GiB. They were computed by
        // dividing by 1024^3 and printed as "GB", which is how an operator
        // reads ~39.7 and sizes a node for a job that peaked at 84.76 GiB.
        assert!(rendered.contains("23.5 GiB"), "{rendered}");
        assert!(rendered.contains("8.0 GiB"), "{rendered}");
        assert!(!rendered.contains(" GB"), "{rendered}");
        assert!(rendered.contains("proceeding"), "{rendered}");
        assert!(rendered.contains("backed=True"), "{rendered}");
        // The scope caveat is the point of the message: this number cannot
        // size a job, and saying so is the only honest thing it can do.
        assert!(rendered.contains("assembly only"), "{rendered}");
        assert!(rendered.contains("not a job size"), "{rendered}");
        // Must NOT leak the debug struct shape.
        assert!(!rendered.contains("EagerAssemblyMemoryHigh"), "{rendered}");
        assert!(!rendered.contains("estimated_bytes"), "{rendered}");
    }

    /// A `container="dense"` request is `n_obs x n_vars x value_width`; the
    /// nonzero count does not bound it, and there is no column-index array for
    /// the int64 story to be about. Saying "N nonzeros at 8 B each (int32
    /// column indices)" there describes a buffer the read is not building.
    #[test]
    fn a_dense_request_is_not_explained_in_nonzeros() {
        let w = ConvertWarning::EagerAssemblyMemoryHigh {
            estimated_bytes: 22_000_000_000,
            budget_bytes: 8_589_934_592,
            nnz: 1_000_000_000,
            value_bytes: 4,
            index_bytes: None,
            dense_request: true,
            dense_bytes: None,
        };
        let rendered = format!("{w}");
        assert!(rendered.contains("do not bound it"), "{rendered}");
        assert!(rendered.contains("1,000,000,000"), "{rendered}");
        assert!(!rendered.contains("column indices"), "{rendered}");
        assert!(!rendered.contains("B each"), "{rendered}");
    }

    /// Every figure in the message is derived from the plan the estimate was
    /// priced with.
    ///
    /// The message hard-coded 8 and 12 B/nnz while the estimate had moved to the
    /// caller's value width — so a `data_dtype="float64"` read reported a total
    /// computed at 12 B/nnz and explained it at 8, and called an int64 index
    /// "two thirds" of a footprint where it is half. A regression introduced by
    /// the fix that made the estimate plan-aware, and the reason this asserts
    /// arithmetic rather than a phrase.
    #[test]
    fn the_per_nonzero_explanation_follows_the_plans_widths() {
        let f64_wide = ConvertWarning::EagerAssemblyMemoryHigh {
            estimated_bytes: 40_000_000_000,
            budget_bytes: 8_589_934_592,
            nnz: 2_500_000_000,
            value_bytes: 8,
            index_bytes: Some(8),
            dense_request: false,
            dense_bytes: None,
        };
        let rendered = format!("{f64_wide}");
        assert!(rendered.contains("at 16 B each"), "{rendered}");
        assert!(
            rendered.contains("8 B of value and 8 B of column index"),
            "{rendered}"
        );
        assert!(!rendered.contains("two thirds"), "{rendered}");

        let u8_narrow = ConvertWarning::EagerAssemblyMemoryHigh {
            estimated_bytes: 12_000_000_000,
            budget_bytes: 8_589_934_592,
            nnz: 2_000_000_000,
            value_bytes: 1,
            index_bytes: Some(4),
            dense_request: false,
            dense_bytes: None,
        };
        let rendered = format!("{u8_narrow}");
        assert!(rendered.contains("at 5 B each"), "{rendered}");
        assert!(!rendered.contains("int64"), "{rendered}");
    }

    /// On a file with deletion vectors the nonzero count is physical and scipy
    /// picks the index width from the live count, which the catalog cannot
    /// supply. The widen refuses to guess it; so must the message.
    #[test]
    fn an_unknown_index_width_is_not_asserted() {
        let w = ConvertWarning::EagerAssemblyMemoryHigh {
            estimated_bytes: 40_000_000_000,
            budget_bytes: 8_589_934_592,
            nnz: 2_650_704_199,
            value_bytes: 4,
            index_bytes: None,
            dense_request: false,
            dense_bytes: None,
        };
        let rendered = format!("{w}");
        assert!(rendered.contains("after deletions"), "{rendered}");
        assert!(rendered.contains("assumes the wider one"), "{rendered}");
        // No per-nonzero total, because the index half of it is unknown.
        assert!(!rendered.contains(" B each:"), "{rendered}");
        assert!(!rendered.contains("int64"), "{rendered}");
    }

    #[test]
    fn group_thousands_groups_from_the_right() {
        assert_eq!(group_thousands(0), "0");
        assert_eq!(group_thousands(7), "7");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1_000), "1,000");
        assert_eq!(group_thousands(12_345), "12,345");
        assert_eq!(group_thousands(2_650_704_199), "2,650,704,199");
        assert_eq!(group_thousands(u64::MAX), "18,446,744,073,709,551,615");
    }

    /// Above `i32::MAX` nonzeros the message names the int64 promotion and its
    /// share of the footprint, and offers `container="dense"` when dense is
    /// genuinely smaller. Below it, neither claim appears — a narrow matrix
    /// told to go dense would be advised into more memory, not less.
    #[test]
    fn eager_assembly_memory_high_names_int64_and_dense_only_when_true() {
        let wide = ConvertWarning::EagerAssemblyMemoryHigh {
            estimated_bytes: 31_782_000_000,
            budget_bytes: 8_589_934_592,
            nnz: 2_650_704_199,
            value_bytes: 4,
            index_bytes: Some(8),
            dense_request: false,
            dense_bytes: Some(23_600_000_000),
        };
        let rendered = format!("{wide}");
        assert!(rendered.contains("2,650,704,199 nonzeros"), "{rendered}");
        assert!(rendered.contains("int64"), "{rendered}");
        assert!(rendered.contains("12 B each"), "{rendered}");
        assert!(rendered.contains("4 B of value and 8 B"), "{rendered}");
        assert!(rendered.contains("container=\"dense\""), "{rendered}");
        assert!(rendered.contains("22.0 GiB"), "{rendered}");

        let narrow = ConvertWarning::EagerAssemblyMemoryHigh {
            estimated_bytes: 25_239_799_332,
            budget_bytes: 8_589_934_592,
            nnz: 1_000_000_000,
            value_bytes: 4,
            index_bytes: Some(4),
            dense_request: false,
            dense_bytes: None,
        };
        let rendered = format!("{narrow}");
        assert!(rendered.contains("8 B each"), "{rendered}");
        assert!(rendered.contains("4 B of value and 4 B"), "{rendered}");
        assert!(!rendered.contains("int64"), "{rendered}");
        assert!(!rendered.contains("dense"), "{rendered}");
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
