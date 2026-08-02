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
use scx_ops::{AttachObsOptions, ExtraRowPolicy, MissingRowPolicy, ObsJoinKey};

type CmdResult = Result<(), Box<dyn std::error::Error>>;

/// Split `--key a,b` into components. Empty (or absent) means auto-resolve.
fn parse_key(key: Option<&str>) -> Vec<String> {
    key.map(|k| {
        k.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
pub fn run_doublet_import(
    input: &Path,
    table: &Path,
    tool: &str,
    key: Option<&str>,
    key_added: Option<&str>,
    score_column: Option<&str>,
    call_column: Option<&str>,
    call_true: Option<&str>,
    call_false: Option<&str>,
    drop_native_columns: bool,
    delimiter: Option<&str>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    dry_run: bool,
) -> CmdResult {
    let missing = match on_missing_rows {
        "zero" => MissingRowPolicy::ZeroFill,
        "error" => MissingRowPolicy::Error,
        other => return Err(format!("--on-missing-rows must be zero|error; got '{other}'").into()),
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

    // Both sides of the join are built from the same names, so the reader and
    // the op cannot disagree about what the key is.
    let key_columns = parse_key(key);
    let join_key = match key_columns.len() {
        0 => ObsJoinKey::Auto,
        1 => ObsJoinKey::Column(key_columns[0].clone()),
        _ => ObsJoinKey::Composite {
            columns: key_columns.clone(),
        },
    };

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
    };

    let attach_opts = AttachObsOptions {
        join_key,
        missing_row_policy: missing,
        extra_row_policy: extra,
        status_column: Some(format!("{key_added}_status")),
        uns_key: Some(key_added.clone()),
        overwrite,
        provenance_action: "doublet_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let (data, info) = scx_convert::read_doublet_table(table, &read_opts)?;
    println!(
        "Table: {} ({} rows, delimiter {:?}, key {:?}{})",
        table.display(),
        info.table.n_rows,
        info.table.delimiter as char,
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
    if info.call_source_column.is_none() {
        // Say so explicitly rather than let the missing column be discovered
        // later: for scds this is by design, not a failed resolution.
        println!(
            "Note: no call column, so '{}_predicted' is not written. Threshold \
             '{}_score' yourself — the importer will not choose a cutoff for you.",
            info.key_added, info.key_added
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
        s.n_matched, s.n_obs, s.obs_key_column, s.n_target_rows_absent, s.n_source_rows_absent,
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
