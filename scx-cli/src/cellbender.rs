//! `scx cellbender-import` — land a CellBender `remove-background` output as a
//! layer on an existing SCX file.
//!
//! The real risk in this operation is the join, not the write: if the barcodes
//! don't line up, the result is a plausible-looking but empty (or wrong) layer.
//! Hence `--dry-run`, which runs every validation and the join and prints the
//! match counts without touching the file.

use std::path::Path;

type CmdResult = Result<(), Box<dyn std::error::Error>>;

#[cfg(feature = "hdf5")]
#[allow(clippy::too_many_arguments)]
pub fn run_cellbender_import(
    input: &Path,
    cellbender_h5: &Path,
    layer: &str,
    obs_key: Option<&str>,
    var_key: Option<&str>,
    prefix: &str,
    uns_key: &str,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    gene_axis: &str,
    latent_embedding: bool,
    dry_run: bool,
) -> CmdResult {
    let missing = match on_missing_rows {
        "zero" => scx_ops::MissingRowPolicy::ZeroFill,
        "error" => scx_ops::MissingRowPolicy::Error,
        other => return Err(format!("--on-missing-rows must be zero|error; got '{other}'").into()),
    };
    let extra = match on_extra_rows {
        "warn" => scx_ops::ExtraRowPolicy::WarnSkip,
        "error" => scx_ops::ExtraRowPolicy::Error,
        other => return Err(format!("--on-extra-rows must be warn|error; got '{other}'").into()),
    };
    let axis = match gene_axis {
        "identical" => scx_ops::ColumnAxisPolicy::RequireIdentical,
        "reorder" => scx_ops::ColumnAxisPolicy::AllowReorder,
        "subset" => scx_ops::ColumnAxisPolicy::AllowSubset,
        other => {
            return Err(
                format!("--gene-axis must be identical|reorder|subset; got '{other}'").into(),
            )
        }
    };

    let read_opts = scx_convert::CellBenderReadOptions {
        column_prefix: prefix.to_string(),
        latent_embedding,
        ..Default::default()
    };
    let mut sink = scx_convert::WarningSink::log();
    let out = scx_convert::read_cellbender_h5(cellbender_h5, &read_opts, &mut sink)?;

    let opts = scx_ops::AttachLayerOptions {
        layer_name: layer.to_string(),
        obs_key_column: obs_key.map(str::to_string),
        var_key_column: var_key.map(str::to_string),
        missing_row_policy: missing,
        extra_row_policy: extra,
        column_axis_policy: axis,
        status_column: Some(format!("{prefix}status")),
        row_sum_column: Some(format!("{prefix}total_counts")),
        uns_key: Some(uns_key.to_string()),
        overwrite,
        provenance_action: "cellbender_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let s = scx_ops::attach_external_layer(input, &out.data, &opts)?;

    println!(
        "CellBender output: {} ({:?}, latents {:?}, {} rows x {} features, estimator {})",
        cellbender_h5.display(),
        out.info.output_kind,
        out.info.latent_alignment,
        out.info.n_rows,
        out.info.n_features,
        out.info.estimator.as_deref().unwrap_or("unknown"),
    );
    println!(
        "Join: {}/{} target rows matched on {} <-> barcodes ({} target rows absent, \
         {} source rows skipped, {} of those carried counts)",
        s.n_matched,
        s.n_obs,
        s.obs_key_column,
        s.n_target_rows_absent,
        s.n_source_rows_absent,
        s.n_source_rows_absent_nonzero,
    );

    if dry_run {
        println!("Dry run: nothing written.");
        return Ok(());
    }

    println!(
        "Wrote layer '{layer}' ({} nnz, {:?}) + obs {:?} + var {:?}",
        s.layer_nnz, s.value_encoding, s.obs_columns_added, s.var_columns_added
    );
    println!("Undo with: scx rollback {}", input.display());
    // Worth saying out loud: a later subset would silently discard the layer.
    println!("Note: `scx subset` currently drops layers.");
    Ok(())
}

#[cfg(not(feature = "hdf5"))]
#[allow(clippy::too_many_arguments)]
pub fn run_cellbender_import(
    _input: &Path,
    _cellbender_h5: &Path,
    _layer: &str,
    _obs_key: Option<&str>,
    _var_key: Option<&str>,
    _prefix: &str,
    _uns_key: &str,
    _overwrite: bool,
    _on_missing_rows: &str,
    _on_extra_rows: &str,
    _gene_axis: &str,
    _latent_embedding: bool,
    _dry_run: bool,
) -> CmdResult {
    Err(
        "cellbender-import requires the 'hdf5' feature. Rebuild with: \
         cargo build -p scx-cli --features hdf5"
            .into(),
    )
}
