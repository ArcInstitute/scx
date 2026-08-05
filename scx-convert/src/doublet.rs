//! Doublet-caller wrapper over [`crate::read_obs_source`].
//!
//! Every popular doublet caller emits a continuous score and a binary call, and
//! every one of them names those two things differently. This module is the one
//! place that knows the six spellings, so downstream consensus code — and the
//! native detector, when it lands — can read `<K>_score` / `<K>_predicted`
//! without branching on which tool produced them.
//!
//! # Why this is a reader concern and not an op
//!
//! `scx_ops::attach_external_obs` knows nothing about doublets, exactly as
//! `attach_external_layer` knows nothing about CellBender. A new tool is a new
//! *reader*, not a new op. This module lives beside [`crate::cellbender`] for
//! the same reason, and is ungated for the same reason
//! [`crate::annotation_table`] is: a CSV importer must work in a build with no
//! libhdf5.
//!
//! # The value derivation is the new part
//!
//! [`crate::annotation_table`] selects, renames and prefixes columns but never
//! transforms *values* — it clones arrays verbatim. Renaming therefore gets the
//! canonical column *names* for free but not the canonical call, which needs
//! `scDblFinder.class == "doublet"` → `bool`. [`coerce_call`] is that step, and
//! it dispatches on the array's **runtime** type rather than on the profile, so
//! it does not depend on which type arrow's inference happened to pick. That
//! matters: `True`/`False` from `obs.to_csv()` infers as `Boolean`, but the same
//! column with one `NA` in it may not — and from an h5ad the same values arrive
//! as an integer or, for a pandas Categorical, dictionary-encoded.
//!
//! # Two source formats, one derivation
//!
//! The source is read through [`crate::read_obs_source`], so a delimited table
//! and an h5ad's `/obs` reach the code below identically. That is the point: a
//! tool that writes its calls back into the h5ad needs no CSV detour, and the
//! two routes cannot drift into producing different answers from the same
//! values.
//!
//! # What it refuses to guess
//!
//! * A profile whose expected column is missing is an error naming the columns
//!   actually present — never a silent skip that yields an all-null score.
//! * A call value that is neither of the profile's two declared tokens is an
//!   error naming the value. Folding a future `ambiguous` class into `false`
//!   would be a scientific claim the tool never made.
//! * `pANN_*` matching by prefix requires **exactly one** match. DoubletFinder
//!   run twice with different `pK` leaves both columns behind, and picking one
//!   silently is a coin flip.
//! * A tool that emits no call column at all (scds) gets no `<K>_predicted`.
//!   Thresholding a score is a scientific decision this importer does not own.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanBuilder, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use serde_json::json;

use scx_ops::{ExternalObsData, OpsError, Result};

use crate::annotation_table::{read_obs_source, AnnotationTableInfo, AnnotationTableOptions};

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

/// The two spellings a text call column uses, compared case-insensitively.
///
/// Both are declared rather than just the positive one so an unrecognised value
/// can be rejected. Knowing only "doublet" would force every other token —
/// including a legitimate one — into the same bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallTokens {
    pub doublet: &'static str,
    pub singlet: &'static str,
}

/// Where one tool keeps its score and its call.
#[derive(Debug, Clone, Copy)]
pub struct DoubletProfile {
    pub name: &'static str,
    /// Ordered aliases; the first one present wins. These are alternate
    /// spellings across tool versions, not parameterisations, so first-wins is
    /// the right rule — unlike [`Self::score_prefix`].
    pub score_columns: &'static [&'static str],
    /// Prefix match requiring **exactly one** hit.
    pub score_prefix: Option<&'static str>,
    pub call_columns: &'static [&'static str],
    pub call_prefix: Option<&'static str>,
    /// Vocabulary for a text call column. `None` where the call is natively
    /// boolean or numeric.
    pub call_tokens: Option<CallTokens>,
}

const P_SCDBLFINDER: DoubletProfile = DoubletProfile {
    name: "scdblfinder",
    score_columns: &["scDblFinder.score"],
    score_prefix: None,
    call_columns: &["scDblFinder.class"],
    call_prefix: None,
    call_tokens: Some(CallTokens {
        doublet: "doublet",
        singlet: "singlet",
    }),
};

const P_SCRUBLET: DoubletProfile = DoubletProfile {
    name: "scrublet",
    score_columns: &["doublet_score"],
    score_prefix: None,
    call_columns: &["predicted_doublet"],
    call_prefix: None,
    // `predicted_doublet` is a numpy bool, so it normally arrives as a
    // `BooleanArray`. The tokens are the fallback for the `NA`-bearing case
    // where inference lands on `Utf8` instead.
    call_tokens: Some(CallTokens {
        doublet: "true",
        singlet: "false",
    }),
};

const P_DOUBLETFINDER: DoubletProfile = DoubletProfile {
    name: "doubletfinder",
    score_columns: &[],
    score_prefix: Some("pANN_"),
    call_columns: &[],
    call_prefix: Some("DF.classifications_"),
    call_tokens: Some(CallTokens {
        doublet: "Doublet",
        singlet: "Singlet",
    }),
};

const P_DOUBLETDETECTION: DoubletProfile = DoubletProfile {
    name: "doubletdetection",
    score_columns: &["doublet_score"],
    score_prefix: None,
    // Numeric 0/1, and NaN for cells the classifier never converged on — which
    // `NULL_TOKENS` turns into a null, and a null stays a null.
    call_columns: &["doublet_label"],
    call_prefix: None,
    call_tokens: None,
};

const P_SOLO: DoubletProfile = DoubletProfile {
    name: "solo",
    score_columns: &["softmax_score", "score"],
    score_prefix: None,
    call_columns: &["prediction"],
    call_prefix: None,
    call_tokens: Some(CallTokens {
        doublet: "doublet",
        singlet: "singlet",
    }),
};

const P_SCDS: DoubletProfile = DoubletProfile {
    name: "scds",
    score_columns: &["hybrid_score", "cxds_score", "bcds_score"],
    score_prefix: None,
    // Deliberately empty: scds emits no call. `<K>_predicted` is omitted rather
    // than invented by thresholding.
    call_columns: &[],
    call_prefix: None,
    call_tokens: None,
};

const P_GENERIC: DoubletProfile = DoubletProfile {
    name: "generic",
    score_columns: &[],
    score_prefix: None,
    call_columns: &[],
    call_prefix: None,
    call_tokens: None,
};

const PROFILES: &[&DoubletProfile] = &[
    &P_SCDBLFINDER,
    &P_SCRUBLET,
    &P_DOUBLETFINDER,
    &P_DOUBLETDETECTION,
    &P_SOLO,
    &P_SCDS,
    &P_GENERIC,
];

/// Valid `--tool` / `tool=` values, in table order.
///
/// Shared so clap's value list and the Python binding's error message cannot
/// disagree about what is accepted.
pub const DOUBLET_PROFILE_NAMES: &[&str] = &[
    "scdblfinder",
    "scrublet",
    "doubletfinder",
    "doubletdetection",
    "solo",
    "scds",
    "generic",
];

/// Look up a tool profile by name.
pub fn doublet_profile(name: &str) -> Result<&'static DoubletProfile> {
    PROFILES
        .iter()
        .copied()
        .find(|p| p.name == name)
        .ok_or_else(|| {
            OpsError::InvalidInput(format!(
                "unknown doublet tool '{name}'; expected one of {DOUBLET_PROFILE_NAMES:?}"
            ))
        })
}

// ---------------------------------------------------------------------------
// Options and report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DoubletImportOptions {
    /// Tool profile name; see [`DOUBLET_PROFILE_NAMES`].
    pub tool: String,
    /// Canonical prefix `<K>`. Empty defaults to the tool name.
    pub key_added: String,
    /// Overrides the profile's score column. Required for `generic`.
    pub score_column: Option<String>,
    /// Overrides the profile's call column. Supplying one for a profile that
    /// has none (scds) is how you opt into `<K>_predicted`.
    pub call_column: Option<String>,
    /// Overrides the text token meaning "doublet".
    pub call_true: Option<String>,
    /// Overrides the text token meaning "singlet".
    pub call_false: Option<String>,
    /// Keep every non-score, non-call column as `<K>_<native>`.
    pub keep_native_columns: bool,
    /// Join key column(s) in the source table. Empty auto-resolves.
    pub key_columns: Vec<String>,
    pub delimiter: Option<u8>,
    /// `/uns` keys to carry across from an h5ad source, nested under the
    /// wrapper's own record as `uns["<K>"]["source_uns"]`. Empty imports none;
    /// meaningless for a delimited table, which carries no uns.
    pub uns_keys: Vec<String>,
}

impl Default for DoubletImportOptions {
    fn default() -> Self {
        Self {
            tool: "generic".to_string(),
            key_added: String::new(),
            score_column: None,
            call_column: None,
            call_true: None,
            call_false: None,
            keep_native_columns: true,
            key_columns: Vec::new(),
            delimiter: None,
            uns_keys: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MissingCallColumn {
    /// Rendered expectation, ready to print: `one of ["doublet_label"]`.
    pub expected: String,
    /// The profile's declared call spellings, machine-readable. Empty for a
    /// prefix-only profile (doubletfinder).
    pub expected_columns: Vec<String>,
    /// The profile's declared call prefix, if any (`DF.classifications_`).
    pub expected_prefix: Option<String>,
    /// The table's schema — the "but what did I have?" half of the diagnostic.
    pub present_columns: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DoubletTableInfo {
    pub tool: String,
    pub key_added: String,
    /// Source column the canonical score came from.
    pub score_source_column: String,
    /// Source column the canonical call came from, if any.
    pub call_source_column: Option<String>,
    /// Set when the tool's profile declares a call column and the table carried
    /// none of its spellings — so `<K>_predicted` was NOT written even though
    /// this tool does emit a call. `None` both when the call resolved and when
    /// the profile declares none (scds), which are the two cases a bare
    /// `call_source_column == None` cannot tell apart.
    pub call_column_missing: Option<MissingCallColumn>,
    /// `<K>_score`, and `<K>_predicted` when a call column was resolved.
    pub canonical_columns: Vec<String>,
    /// Passed-through source columns, under their final `<K>_<native>` names.
    pub native_columns: Vec<String>,
    /// Source columns dropped because they are another declared spelling of the
    /// score or call that was chosen, and would otherwise collide with the
    /// canonical name. Reported rather than silently discarded.
    pub dropped_alias_columns: Vec<String>,
    /// The underlying delimited-table read.
    pub table: AnnotationTableInfo,
}

// ---------------------------------------------------------------------------
// Column resolution
// ---------------------------------------------------------------------------

fn present_columns(batch: &RecordBatch) -> Vec<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// Outcome of resolving one of a profile's columns.
///
/// This used to be `Option<String>`, which conflated two different facts: "this
/// profile declares no such column" (scds, generic) and "it declares one and the
/// table carries none of the spellings". Worse, the sentence explaining the
/// second case — the one thing a user needs — was built only on the error path
/// and thrown away when the column was optional. So a `doubletdetection` import
/// of a table spelling its call `predicted_doublet` (scrublet's name) silently
/// produced a score-only import, and the failure surfaced much later in
/// `doublet_consensus` under the false claim "That tool emits no call column".
///
/// Both facts, and the sentence, now reach the caller, which decides whether
/// either is fatal.
enum ColumnMatch {
    Found(String),
    /// Nothing to look for: no aliases, no prefix, no explicit override. This is
    /// `!profile_has_call_column(profile)` for the call role.
    NotDeclared {
        expected: String,
        present: Vec<String>,
    },
    /// Declared, and absent from this table.
    DeclaredButAbsent {
        expected: String,
        present: Vec<String>,
    },
}

/// Resolve one of a profile's columns: explicit override, then aliases, then a
/// prefix search that must land on exactly one column.
///
/// Returns [`ColumnMatch`] rather than `Option`, so a caller can tell a
/// by-design absence from a name mismatch. An explicit override that is absent
/// is still a hard error for every role — that is unambiguous user error and
/// must not degrade to a warning.
///
/// Deliberately takes `aliases`/`prefix` rather than a `&DoubletProfile`, so it
/// stays profile-agnostic and a caller-supplied `call_column=` correctly flips
/// the answer from `NotDeclared` to `Found`/an error.
fn resolve_column(
    batch: &RecordBatch,
    role: &str,
    explicit: Option<&str>,
    aliases: &[&str],
    prefix: Option<&str>,
) -> Result<ColumnMatch> {
    let present = present_columns(batch);

    if let Some(name) = explicit {
        if !present.iter().any(|p| p == name) {
            return Err(OpsError::InvalidInput(format!(
                "{role} column '{name}' is not in the table; columns present are {present:?}"
            )));
        }
        return Ok(ColumnMatch::Found(name.to_string()));
    }

    for a in aliases {
        if let Some(hit) = present.iter().find(|p| p.as_str() == *a) {
            return Ok(ColumnMatch::Found(hit.clone()));
        }
    }

    if let Some(pre) = prefix {
        let hits: Vec<&String> = present.iter().filter(|p| p.starts_with(pre)).collect();
        match hits.len() {
            1 => return Ok(ColumnMatch::Found(hits[0].clone())),
            0 => {}
            // Two `pANN_*` columns means the tool was run twice with different
            // parameters. Either is defensible; choosing for the user is not.
            _ => {
                return Err(OpsError::InvalidInput(format!(
                    "{} columns start with '{pre}': {hits:?}. Pass the one you want \
                     explicitly, because picking one would silently choose between two \
                     different parameterisations of the same tool.",
                    hits.len()
                )))
            }
        }
    }

    // Built unconditionally now: it is the diagnostic for BOTH the hard-error
    // (score) and the warn-only (call) cases, and discarding it on the latter is
    // what made the score-only degradation silent.
    let expected = match (aliases.is_empty(), prefix) {
        (false, Some(p)) => format!("one of {aliases:?} or a column starting with '{p}'"),
        (false, None) => format!("one of {aliases:?}"),
        (true, Some(p)) => format!("a column starting with '{p}'"),
        (true, None) => "an explicitly named column".to_string(),
    };
    if aliases.is_empty() && prefix.is_none() {
        Ok(ColumnMatch::NotDeclared { expected, present })
    } else {
        Ok(ColumnMatch::DeclaredButAbsent { expected, present })
    }
}

/// The error a *required* column's non-`Found` outcome becomes.
///
/// One place, so the score role keeps its exact pre-existing message shape
/// (pinned by `missing_expected_column_errors_naming_present_columns` and
/// `generic_requires_a_score_column`).
fn missing_required_column(role: &str, m: &ColumnMatch) -> OpsError {
    let (expected, present) = match m {
        ColumnMatch::Found(_) => unreachable!("caller checked for Found"),
        ColumnMatch::NotDeclared { expected, present }
        | ColumnMatch::DeclaredButAbsent { expected, present } => {
            (expected.clone(), present.clone())
        }
    };
    OpsError::InvalidInput(format!(
        "no {role} column found: expected {expected}, but the columns present are {present:?}"
    ))
}

// ---------------------------------------------------------------------------
// Value derivation
// ---------------------------------------------------------------------------

/// Decode a dictionary-encoded column to its plain value array.
///
/// A pandas Categorical arrives from an h5ad's `obs` as
/// `Dictionary(Int32, Utf8)` — and a class / prediction column is exactly the
/// kind pandas stores that way. The derivation has to see through the encoding
/// rather than reject it, or the same values would import from a CSV and fail
/// from an h5ad. Non-dictionary columns pass through untouched.
fn undictionary(col: &ArrayRef, name: &str, role: &str) -> Result<ArrayRef> {
    match col.data_type() {
        DataType::Dictionary(_, value) => arrow::compute::cast(col.as_ref(), value).map_err(|e| {
            OpsError::InvalidInput(format!(
                "could not decode categorical {role} column '{name}' to {value:?}: {e}"
            ))
        }),
        _ => Ok(Arc::clone(col)),
    }
}

/// Narrow a score column to the canonical `f32`.
///
/// A non-numeric column is rejected rather than cast: `Utf8` → `Float32` would
/// succeed and null every row, which reads downstream as "the tool covered no
/// cells" rather than "you pointed at the wrong column".
fn coerce_score(col: &ArrayRef, name: &str) -> Result<ArrayRef> {
    let col = &undictionary(col, name, "score")?;
    if !col.data_type().is_numeric() {
        return Err(OpsError::InvalidInput(format!(
            "score column '{name}' has type {:?}, which is not numeric. A text score \
             usually means the wrong column was named, or that the file uses a missing \
             value token the reader does not recognise.",
            col.data_type()
        )));
    }
    arrow::compute::cast(col.as_ref(), &DataType::Float32).map_err(|e| {
        OpsError::InvalidInput(format!("could not read score column '{name}' as f32: {e}"))
    })
}

/// Derive the canonical boolean call.
///
/// Dispatches on the array's runtime type, not on the profile, so the same
/// profile works whether arrow inferred `Boolean`, an integer, or `Utf8` for
/// the column — and, after [`undictionary`], whether the source stored it as a
/// pandas Categorical. Nulls stay null throughout: a cell the tool did not call
/// is not a singlet.
fn coerce_call(
    col: &ArrayRef,
    name: &str,
    tokens: Option<CallTokens>,
    true_override: Option<&str>,
    false_override: Option<&str>,
) -> Result<ArrayRef> {
    let col = &undictionary(col, name, "call")?;
    match col.data_type() {
        DataType::Boolean => Ok(Arc::clone(col)),

        dt if dt.is_numeric() => {
            // Cast through f64 so one arm covers every integer and float width.
            let vals = arrow::compute::cast(col.as_ref(), &DataType::Float64).map_err(|e| {
                OpsError::InvalidInput(format!("could not read call column '{name}': {e}"))
            })?;
            let vals = vals
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("cast to Float64 yields a Float64Array");
            let mut out = BooleanBuilder::with_capacity(vals.len());
            for i in 0..vals.len() {
                if vals.is_null(i) {
                    out.append_null();
                    continue;
                }
                match vals.value(i) {
                    1.0 => out.append_value(true),
                    0.0 => out.append_value(false),
                    other => {
                        return Err(OpsError::InvalidInput(format!(
                            "call column '{name}' contains {other}, but a numeric call must \
                             be 0 or 1. Mapping anything else to a singlet would be a claim \
                             the tool never made."
                        )))
                    }
                }
            }
            Ok(Arc::new(out.finish()) as ArrayRef)
        }

        DataType::Utf8 | DataType::LargeUtf8 => {
            let (yes, no) = match (true_override, false_override, tokens) {
                // An explicit pair, or the profile's declared vocabulary: strict,
                // because we know both spellings and an unknown third is a
                // genuine surprise.
                (Some(t), Some(f), _) => (t.to_string(), Some(f.to_string())),
                (Some(t), None, Some(tk)) => (t.to_string(), Some(tk.singlet.to_string())),
                (None, Some(f), Some(tk)) => (tk.doublet.to_string(), Some(f.to_string())),
                (None, None, Some(tk)) => (tk.doublet.to_string(), Some(tk.singlet.to_string())),
                // Only the positive token, and no profile vocabulary to fill in
                // the negative: the caller has asked for a partition, so honour
                // it. Their explicit choice, not our inference.
                (Some(t), None, None) => (t.to_string(), None),
                (None, _, None) => {
                    return Err(OpsError::InvalidInput(format!(
                        "call column '{name}' is text and this tool profile declares no \
                         vocabulary for it. Pass call_true (and optionally call_false) to \
                         say which value means doublet."
                    )))
                }
            };

            let text = arrow::compute::cast(col.as_ref(), &DataType::Utf8).map_err(|e| {
                OpsError::InvalidInput(format!("could not read call column '{name}': {e}"))
            })?;
            let text = text
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("cast to Utf8 yields a StringArray");

            let mut out = BooleanBuilder::with_capacity(text.len());
            for i in 0..text.len() {
                if text.is_null(i) {
                    out.append_null();
                    continue;
                }
                let v = text.value(i);
                if v.eq_ignore_ascii_case(&yes) {
                    out.append_value(true);
                } else if let Some(no) = &no {
                    if v.eq_ignore_ascii_case(no) {
                        out.append_value(false);
                    } else {
                        return Err(OpsError::InvalidInput(format!(
                            "call column '{name}' contains '{v}', which is neither '{yes}' \
                             nor '{no}'. Pass call_true / call_false if this tool version \
                             spells its classes differently."
                        )));
                    }
                } else {
                    out.append_value(false);
                }
            }
            Ok(Arc::new(out.finish()) as ArrayRef)
        }

        other => Err(OpsError::InvalidInput(format!(
            "call column '{name}' has type {other:?}, which cannot be read as a \
             doublet call"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Read a doublet caller's output table into an [`ExternalObsData`] carrying
/// the canonical `<K>_score` / `<K>_predicted` columns.
pub fn read_doublet_table(
    path: &Path,
    opts: &DoubletImportOptions,
) -> Result<(ExternalObsData, DoubletTableInfo)> {
    let profile = doublet_profile(&opts.tool)?;
    let key_added = if opts.key_added.is_empty() {
        profile.name.to_string()
    } else {
        opts.key_added.clone()
    };

    // Read everything raw: no selection, no rename, no prefix. Resolving the
    // profile's columns against the parsed batch (rather than teaching the
    // reader about profiles) means prefix matching sees the real schema.
    //
    // Via `read_obs_source`, so a delimited table and an h5ad's /obs both reach
    // the same derivation — a tool that writes its calls back into the h5ad
    // needs no CSV detour.
    let (mut data, table_info) = read_obs_source(
        path,
        &AnnotationTableOptions {
            key_columns: opts.key_columns.clone(),
            delimiter: opts.delimiter,
            columns: None,
            rename: HashMap::new(),
            prefix: String::new(),
            keep_key_columns: false,
            infer_max_records: None,
        },
        &opts.uns_keys,
    )?;
    // Whatever the source reader put in `uns` is the tool's own metadata; the
    // wrapper's record is built below and must not silently replace it.
    let source_uns = data.uns.take();

    let batch = &data.row_annotations;

    // The score is required: both non-`Found` outcomes are a hard error, with
    // the pre-existing message shape.
    let score_source = match resolve_column(
        batch,
        "score",
        opts.score_column.as_deref(),
        profile.score_columns,
        profile.score_prefix,
    )? {
        ColumnMatch::Found(name) => name,
        other => return Err(missing_required_column("score", &other)),
    };

    // The call is optional, but the two ways it can be absent mean very
    // different things and only one of them is by design.
    let (call_source, call_column_missing) = match resolve_column(
        batch,
        "call",
        opts.call_column.as_deref(),
        profile.call_columns,
        profile.call_prefix,
    )? {
        ColumnMatch::Found(name) => (Some(name), None),
        // scds / generic: no call to look for. Omitting `<K>_predicted` is the
        // documented behaviour, so say nothing.
        ColumnMatch::NotDeclared { .. } => (None, None),
        // This profile DOES emit a call and the table did not carry it under any
        // spelling the profile knows. Carry the diagnostic out so the binding can
        // warn and `uns` can record why there is no `<K>_predicted`.
        ColumnMatch::DeclaredButAbsent { expected, present } => (
            None,
            Some(MissingCallColumn {
                expected,
                expected_columns: profile.call_columns.iter().map(|s| s.to_string()).collect(),
                expected_prefix: profile.call_prefix.map(|s| s.to_string()),
                present_columns: present,
            }),
        ),
    };

    let score_name = format!("{key_added}_score");
    let call_name = format!("{key_added}_predicted");

    let mut fields = vec![Field::new(&score_name, DataType::Float32, true)];
    let mut columns = vec![coerce_score(
        batch
            .column_by_name(&score_source)
            .expect("resolved column exists"),
        &score_source,
    )?];
    let mut canonical_columns = vec![score_name.clone()];

    if let Some(src) = &call_source {
        fields.push(Field::new(&call_name, DataType::Boolean, true));
        columns.push(coerce_call(
            batch.column_by_name(src).expect("resolved column exists"),
            src,
            profile.call_tokens,
            opts.call_true.as_deref(),
            opts.call_false.as_deref(),
        )?);
        canonical_columns.push(call_name.clone());
    }

    // Everything else, prefixed and unchanged. The score and call sources are
    // consumed into the canonical pair rather than re-emitted — the same numbers
    // under two names is an invitation for the two to drift.
    let mut native_columns = Vec::new();
    let mut dropped_alias_columns = Vec::new();
    if opts.keep_native_columns {
        for (i, f) in batch.schema().fields().iter().enumerate() {
            let name = f.name();
            if name == &score_source || Some(name) == call_source.as_ref() {
                continue;
            }
            let final_name = format!("{key_added}_{name}");
            if final_name == score_name || final_name == call_name {
                // A profile alias that lost the first-present-wins race is a
                // declared alternate *spelling* of the column already emitted,
                // so dropping it loses nothing — solo writing both
                // `softmax_score` and `score` is the case. Any other column
                // landing on a canonical name is a genuine clash, and silently
                // dropping it would lose data.
                let is_losing_alias = profile.score_columns.contains(&name.as_str())
                    || profile.call_columns.contains(&name.as_str());
                if is_losing_alias {
                    dropped_alias_columns.push(name.clone());
                    continue;
                }
                return Err(OpsError::InvalidInput(format!(
                    "source column '{name}' would be imported as '{final_name}', which \
                     collides with the canonical column of the same name. Rename it in \
                     the source table, or pass keep_native_columns=false."
                )));
            }
            fields.push(Field::new(&final_name, f.data_type().clone(), true));
            columns.push(Arc::clone(batch.column(i)));
            native_columns.push(final_name);
        }
    }

    data.row_annotations = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| OpsError::InvalidInput(format!("failed to build doublet columns: {e}")))?;

    let mut record = json!({
        "tool": profile.name,
        "key_added": key_added,
        "source_file": data.source_name,
        "source_format": table_info.format.as_str(),
        "source_score_column": score_source,
        "source_call_column": call_source,
        // Three-way, because a bare `source_call_column: null` cannot say WHY.
        // `not_declared` is a positive statement ("this tool emits scores only")
        // that the consensus error needs in order to stop asserting it for every
        // tool. Read by `_resolve_consensus_keys` in pyscx.
        "call_column_status": match (&call_source, &call_column_missing) {
            (Some(_), _) => "resolved",
            (None, None) => "not_declared",
            (None, Some(_)) => "declared_but_absent",
        },
        "expected_call_columns": call_column_missing
            .as_ref()
            .map(|m| m.expected_columns.clone())
            .unwrap_or_default(),
        "expected_call_prefix": call_column_missing
            .as_ref()
            .and_then(|m| m.expected_prefix.clone()),
        "n_rows_in_source": table_info.n_rows,
        "join_key_columns": table_info.key_columns,
    });
    // Nested rather than merged: the tool's own uns keys keep their source
    // names, and none of them can collide with the fields above however the
    // caller spelled `key_added`.
    if let Some(src) = source_uns {
        record["source_uns"] = src;
    }
    data.uns = Some(record);

    let info = DoubletTableInfo {
        tool: profile.name.to_string(),
        key_added,
        score_source_column: score_source,
        call_source_column: call_source,
        call_column_missing,
        canonical_columns,
        native_columns,
        dropped_alias_columns,
        table: table_info,
    };
    Ok((data, info))
}

/// `true` when this tool profile can produce a `<K>_predicted` at all.
///
/// scds emits scores only, so a caller printing a report should not promise a
/// call column that will never appear.
pub fn profile_has_call_column(profile: &DoubletProfile) -> bool {
    !profile.call_columns.is_empty() || profile.call_prefix.is_some()
}

#[cfg(test)]
#[path = "doublet_tests.rs"]
mod tests;
