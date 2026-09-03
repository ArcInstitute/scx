//! `scx doublet-import` — land a doublet caller's output as canonical `obs`
//! columns on an existing SCX file.
//!
//! The doublet-specific wrapper over `scx obs-import`. Same in-place,
//! key-joined, rollback-able import; what this adds is the per-tool column
//! mapping, so `scDblFinder.score` and `doublet_score` both arrive as
//! `<K>_score` and downstream code never branches on which tool ran.
//!
//! Ungated for the same reason `obs-import` is: reading a CSV has no business
//! requiring libhdf5, and this command must exist in a no-default-features
//! build.
//!
//! As always, the risk is the join rather than the write — a key that fails to
//! line up yields a plausible-looking but empty column. Hence `--dry-run`,
//! which runs every validation and the join, prints the counts *and* a key
//! diagnosis, and touches nothing.

use std::path::Path;

use scx_convert::DoubletImportOptions;
use scx_ops::{AttachObsOptions, ExtraRowPolicy, MissingRowPolicy};

type CmdResult = Result<(), Box<dyn std::error::Error>>;

#[allow(clippy::too_many_arguments)]
pub fn run_doublet_import(
    input: &Path,
    table: &Path,
    tool: &str,
    key: Option<&str>,
    source_key: Option<&str>,
    key_added: Option<&str>,
    score_column: Option<&str>,
    call_column: Option<&str>,
    call_true: Option<&str>,
    call_false: Option<&str>,
    drop_native_columns: bool,
    delimiter: Option<&str>,
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

    // Resolve the profile before any I/O so a typo'd --tool fails immediately,
    // and so --key-added can default to the profile's own name.
    let profile = scx_convert::doublet_profile(tool)?;
    let key_added = key_added
        .filter(|s| !s.is_empty())
        .unwrap_or(profile.name)
        .to_string();

    // Each side resolves its own names, so a caller output that spells the key
    // differently joins without a rename — but by default both sides use the
    // same names and cannot disagree about what the key is.
    let (key_columns, join_key) = crate::obs_import::resolve_key_pair(key, source_key)?;

    let read_opts = DoubletImportOptions {
        tool: tool.to_string(),
        key_added: key_added.clone(),
        score_column: score_column.map(str::to_string),
        call_column: call_column.map(str::to_string),
        call_true: call_true.map(str::to_string),
        call_false: call_false.map(str::to_string),
        keep_native_columns: !drop_native_columns,
        key_columns,
        delimiter: delimiter_byte,
        uns_keys: uns_keys.to_vec(),
    };

    let attach_opts = AttachObsOptions {
        join_key,
        missing_row_policy: missing,
        extra_row_policy: extra,
        status_column: Some(format!("{key_added}_status")),
        overwrite,
        provenance_action: "doublet_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let (data, info) = scx_convert::read_doublet_table(table, &read_opts)?;
    println!(
        "Source: {} ({}, {} rows{}, key {:?}{})",
        table.display(),
        info.table.format.as_str(),
        info.table.n_rows,
        match info.table.delimiter {
            Some(d) => format!(", delimiter {:?}", d as char),
            None => String::new(),
        },
        info.table.key_columns,
        if info.table.renamed_index_column {
            ", unnamed index column renamed to '_index'"
        } else {
            ""
        }
    );
    println!(
        "Tool: {} (score from {:?}, call from {:?})",
        info.tool, info.score_source_column, info.call_source_column
    );
    // Two very different reasons there is no call column, and the note used to
    // treat them identically — asserting "by design" even when the real cause
    // was that the table spelled the call something the profile does not know.
    match (&info.call_source_column, &info.call_column_missing) {
        (None, None) => {
            // scds / generic: by design, not a failed resolution.
            println!(
                "Note: no call column, so '{}_predicted' is not written. Threshold \
                 '{}_score' yourself — the importer will not choose a cutoff for you.",
                info.key_added, info.key_added
            );
        }
        (None, Some(m)) => {
            // `m.expected` already renders aliases AND prefix in one phrase.
            let expected = &m.expected;
            println!(
                "Note: '{}_predicted' was NOT written — see the warning below.",
                info.key_added
            );
            // stderr, matching the CLI's established warning channel, so it
            // stays visible when stdout is piped to a log.
            eprintln!(
                "warning: --tool {} declares a call column ({expected}) but the table has none \
                 of those names — columns present are {:?}. Imported score only, so this tool \
                 cannot vote on a call in a later consensus. Re-run with \
                 --call-column <your column>, or pass the --tool whose profile matches this \
                 table.",
                info.tool, m.present_columns
            );
        }
        _ => {}
    }
    if !info.table.uns_keys_imported.is_empty() {
        println!(
            "uns keys carried across: {:?} (nested under uns['{}']['source_uns'])",
            info.table.uns_keys_imported, info.key_added
        );
    }
    if !info.dropped_alias_columns.is_empty() {
        println!(
            "Note: dropped {:?} — another spelling of a column already imported.",
            info.dropped_alias_columns
        );
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
