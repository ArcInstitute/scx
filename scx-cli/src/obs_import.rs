//! `scx obs-import` — land a delimited annotation table as `obs` columns on an
//! existing SCX file.
//!
//! The generic half of the doublet-caller workflow: run a tool externally, have
//! it write `barcode,score,call`, import. Nothing here is doublet-specific.
//!
//! Ungated, unlike `cellbender-import` — a CSV importer has no business
//! requiring libhdf5, and this command must exist in a no-default-features
//! build.
//!
//! As with the CellBender importer, the real risk is the join, not the write:
//! if the keys don't line up the result is a plausible-looking but empty
//! column. Hence `--dry-run`, which runs the join and every validation that
//! does not need a non-key obs column decoded (see `attach_external_obs`), prints
//! the match counts *and* a key diagnosis, and touches nothing.

use std::path::Path;

use scx_convert::AnnotationTableOptions;
use scx_ops::{AttachObsOptions, ExtraRowPolicy, MissingRowPolicy, ObsJoinKey};

type CmdResult = Result<(), Box<dyn std::error::Error>>;

/// Split `--key a,b` into components. Empty (or absent) means auto-resolve.
///
/// Shared with `var-import`: both commands accept the same spelling, and two
/// copies would be two places for the trimming rules to drift.
pub(crate) fn parse_key_components(key: Option<&str>) -> Vec<String> {
    key.map(|k| {
        k.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

/// Split `--key` / `--source-key` into the source-side column list and the
/// target-side join spec.
///
/// `--source-key` is how a join whose two sides name the key differently is
/// expressed — `--key sample_id,obs_names --source-key sample_id,barcode`.
/// Without it both sides use the same names, which stays the common case.
pub(crate) fn resolve_key_pair(
    key: Option<&str>,
    source_key: Option<&str>,
) -> Result<(Vec<String>, ObsJoinKey), Box<dyn std::error::Error>> {
    let target = parse_key_components(key);
    let source = parse_key_components(source_key);
    if !source.is_empty() && target.is_empty() {
        return Err(
            "--source-key needs --key: it names the source-side column for each \
                    target-side key component, positionally. To key on the target's obs \
                    index, pass --key obs_names."
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
        0 => ObsJoinKey::Auto,
        1 => ObsJoinKey::Column(target[0].clone()),
        _ => ObsJoinKey::Composite {
            columns: target.clone(),
        },
    };
    let source_columns = if source.is_empty() { target } else { source };
    Ok((source_columns, join_key))
}

/// Parse `--rename SRC=DST` pairs. Shared with `var-import`.
pub(crate) fn parse_rename_pairs(
    pairs: &[String],
) -> Result<std::collections::HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut out = std::collections::HashMap::new();
    for p in pairs {
        let (src, dst) = p
            .split_once('=')
            .ok_or_else(|| format!("--rename expects SRC=DST; got '{p}'"))?;
        if src.is_empty() || dst.is_empty() {
            return Err(format!("--rename expects a non-empty SRC and DST; got '{p}'").into());
        }
        out.insert(src.to_string(), dst.to_string());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn run_obs_import(
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
        // "null" is what actually happens on the obs paths (Arrow
        // nulls, not zeros); "zero" stays accepted for back-compat.
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

    // Each side resolves its own names, so a source table that spells the key
    // differently joins without a rename — but by default both sides use the
    // same names and cannot disagree about what the key is.
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

    let attach_opts = AttachObsOptions {
        join_key,
        missing_row_policy: missing,
        extra_row_policy: extra,
        status_column: status_column.map(str::to_string),
        overwrite,
        provenance_action: "obs_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let (mut data, info) = scx_convert::read_obs_source(table, &read_opts, uns_keys)?;
    // Without --uns-key the selected source keys land at top level under
    // their own names; with it they nest under the one key.
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

    let s = scx_ops::attach_external_obs(input, &data, &attach_opts)?;
    println!(
        "Join: {}/{} target rows matched on {} ({} target rows absent, \
         {} source rows skipped)",
        s.n_matched,
        s.n_obs,
        scx_ops::display_key_name("obs", &s.obs_key_column),
        s.n_target_rows_absent,
        s.n_source_rows_absent,
    );

    // Printed before the dry-run return on purpose: a preview whose whole job
    // is "should I run this on the atlas" must say which memory path it would
    // take. (Round-2 finding: Grok, Gemini, codex.)
    println!(
        "Obs rewrite: {} (peak memory {})",
        if s.obs_streamed {
            "streamed shard-by-shard"
        } else {
            "whole table materialized"
        },
        if s.obs_streamed {
            "one obs shard"
        } else {
            "the entire obs table \u{2014} run `scx optimize` to shard it"
        },
    );
    if dry_run {
        // The join succeeded, but "succeeded" is not the same as "is the key
        // you wanted" — report the alternatives while nothing is committed.
        if let Ok(diag) = scx_ops::diagnose_obs_key(input, Some(&attach_opts.join_key)) {
            println!("Key diagnosis: {}", diag.describe());
        }
        println!("Dry run: nothing written.");
        return Ok(());
    }

    println!("Wrote obs columns {:?}", s.obs_columns_added);
    if s.obs_index_dropped {
        println!(
            "Note: the obs predicate index was dropped — this import overwrote a \
             column it covered, so query pushdown on that column is gone until \
             the index is rebuilt."
        );
    }
    println!("Undo with: scx rollback {}", input.display());
    Ok(())
}
