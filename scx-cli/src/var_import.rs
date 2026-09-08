//! `scx var-import` — land a delimited annotation table as `var` columns on an
//! existing SCX file.
//!
//! The var-axis twin of [`crate::obs_import`]: per-**gene** annotations
//! computed elsewhere — a normalised symbol from a reference release, an ATAC
//! peak annotation, a curated flag — landed in place without rewriting the
//! file. Ungated for the same reason: a CSV importer has no business requiring
//! libhdf5, and this command must exist in a no-default-features build.
//!
//! The real risk is the join, not the write: if the keys do not line up the
//! result is a plausible-looking but empty column. Hence `--dry-run`, which
//! runs the join and every validation, prints the match counts *and* a key
//! diagnosis, and touches nothing.

use std::path::Path;

use scx_convert::AnnotationTableOptions;
use scx_ops::{AttachVarOptions, AxisJoinKey, ExtraRowPolicy, MissingRowPolicy};

use crate::obs_import::{parse_key_components, parse_rename_pairs};

type CmdResult = Result<(), Box<dyn std::error::Error>>;

/// Split `--key` / `--source-key` into the source-side column list and the
/// target-side join spec.
///
/// The obs twin's `resolve_key_pair`, with `var_names` as the index alias the
/// remedy names — telling a var caller to pass `obs_names` would send them
/// looking for a column their file does not have.
fn resolve_key_pair(
    key: Option<&str>,
    source_key: Option<&str>,
) -> Result<(Vec<String>, AxisJoinKey), Box<dyn std::error::Error>> {
    let target = parse_key_components(key);
    let source = parse_key_components(source_key);
    if !source.is_empty() && target.is_empty() {
        return Err(
            "--source-key needs --key: it names the source-side column for each \
                    target-side key component, positionally. To key on the target's var \
                    index, pass --key var_names."
                .into(),
        );
    }
    if !source.is_empty() && source.len() != target.len() {
        return Err(format!(
            "--key has {} component(s) but --source-key has {}; they pair up \
             positionally, so the counts must match",
            target.len(),
            source.len()
        )
        .into());
    }
    let join_key = match target.len() {
        0 => AxisJoinKey::Auto,
        1 => AxisJoinKey::Column(target[0].clone()),
        _ => AxisJoinKey::Composite {
            columns: target.clone(),
        },
    };
    let source_columns = if source.is_empty() { target } else { source };
    Ok((source_columns, join_key))
}

#[allow(clippy::too_many_arguments)]
pub fn run_var_import(
    input: &Path,
    table: &Path,
    key: Option<&str>,
    source_key: Option<&str>,
    columns: Option<&str>,
    rename: &[String],
    prefix: &str,
    keep_key_columns: bool,
    delimiter: Option<&str>,
    status_column: Option<&str>,
    uns_key: Option<&str>,
    uns_keys: &[String],
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    dry_run: bool,
) -> CmdResult {
    let missing = match on_missing_rows {
        // "null" is what actually happens on the annotation paths (Arrow
        // nulls, not zeros); "zero" stays accepted for symmetry with obs.
        "null" | "zero" => MissingRowPolicy::ZeroFill,
        "error" => MissingRowPolicy::Error,
        other => {
            return Err(format!("--on-missing-rows must be null|zero|error; got '{other}'").into())
        }
    };
    let extra = match on_extra_rows {
        "warn" => ExtraRowPolicy::WarnSkip,
        "error" => ExtraRowPolicy::Error,
        other => return Err(format!("--on-extra-rows must be warn|error; got '{other}'").into()),
    };
    let delimiter_byte = match delimiter {
        None => None,
        Some(s) => {
            let b = s.as_bytes();
            if b.len() != 1 {
                return Err(format!("--delimiter must be one byte; got '{s}'").into());
            }
            Some(b[0])
        }
    };

    let (key_columns, join_key) = resolve_key_pair(key, source_key)?;

    let read_opts = AnnotationTableOptions {
        key_columns,
        delimiter: delimiter_byte,
        columns: columns.map(|c| {
            c.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        }),
        rename: parse_rename_pairs(rename)?,
        prefix: prefix.to_string(),
        keep_key_columns,
        infer_max_records: None,
    };

    let attach_opts = AttachVarOptions {
        join_key,
        missing_row_policy: missing,
        extra_row_policy: extra,
        status_column: status_column.map(str::to_string),
        overwrite,
        provenance_action: "var_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let (mut data, info) = scx_convert::read_var_source(table, &read_opts, uns_keys)?;
    if let Some(k) = uns_key {
        data.nest_uns_under(k);
    }
    println!(
        "Source: {} ({}, {} rows{}, key {:?}{})",
        table.display(),
        info.format.as_str(),
        info.n_rows,
        match info.delimiter {
            Some(d) => format!(", delimiter {:?}", d as char),
            None => String::new(),
        },
        info.key_columns,
        if info.renamed_index_column {
            ", unnamed index column renamed to '_index'"
        } else {
            ""
        }
    );
    println!("Columns to import: {:?}", info.columns_imported);
    if !info.uns_keys_imported.is_empty() {
        println!("uns keys to import: {:?}", info.uns_keys_imported);
    }

    let s = scx_ops::attach_external_var(input, &data, &attach_opts)?;
    println!(
        "Join: {}/{} genes matched on {} ({} genes absent, {} source rows skipped)",
        s.n_matched,
        s.n_vars,
        scx_ops::display_key_name("var", &s.var_key_column),
        s.n_target_rows_absent,
        s.n_source_rows_absent,
    );
    // Printed before the dry-run return: which layout the file has decides how
    // var is rewritten, and it is what a preview is being asked about.
    println!(
        "Var rewrite: {}",
        if s.var_streamed {
            "streamed shard-by-shard, boundaries preserved"
        } else {
            "single section rewritten whole"
        },
    );
    // Printed before the dry-run return: the index outcome is decided from the
    // new values before anything is written, so a preview can and must report
    // it. A caller who indexed a var column deliberately needs to know it is
    // about to lose pushdown, not to discover it afterwards.
    // Past tense only once something has happened: on a preview these are
    // still predictions, and "was DROPPED" one line above "nothing written"
    // reads as an accomplished fact.
    let (verb_rebuilt, verb_dropped, verb_covers) = if dry_run {
        (
            "would be rebuilt",
            "would be DROPPED",
            "would no longer cover",
        )
    } else {
        ("was rebuilt", "was DROPPED", "no longer covers")
    };
    if s.var_index_rebuilt {
        println!(
            "Note: the var predicate index {verb_rebuilt} — this import overwrites \
             a column it covered, so its entries describe the new values."
        );
    }
    if s.var_index_dropped {
        println!(
            "Note: the var predicate index {verb_dropped} — this import overwrites \
             every column it covered with values that cannot be indexed, so no \
             replacement is written."
        );
    }
    if !s.var_columns_not_carried.is_empty() {
        println!(
            "Note: the var index {verb_covers} {:?} — the new values cannot be \
             indexed.",
            s.var_columns_not_carried
        );
    }
    if dry_run {
        // The join succeeded, but "succeeded" is not the same as "is the key
        // you wanted" — report the alternatives while nothing is committed.
        if let Ok(diag) = scx_ops::diagnose_var_key(input, Some(&attach_opts.join_key)) {
            println!("Key diagnosis: {}", diag.describe());
        }
        println!("Dry run: nothing written.");
        return Ok(());
    }

    println!("Wrote var columns {:?}", s.var_columns_added);
    println!("Undo with: scx rollback {}", input.display());
    Ok(())
}
