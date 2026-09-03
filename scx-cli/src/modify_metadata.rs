// scx modify-metadata — Replace metadata sections (uns/obs/var/obsm/varm)
// of an SCX file in place, without re-encoding X.
//
// obs/var are read from Parquet; obsm/varm from 2D `.npy` (float32 or
// float64, cast to f32). A replaced obs/var keeps the predicate index it had
// (rebuilt over the same columns) unless --index-* names a different set.
// Multimodal is deferred.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use scx_engine::ConversionPredicateIndexOptions;
use scx_ops::MetadataPatch;

use crate::index_warnings::emit_index_summary;

#[allow(clippy::too_many_arguments)]
pub fn run_modify_metadata(
    file: &Path,
    uns: Option<&Path>,
    obs: Option<&Path>,
    var: Option<&Path>,
    obsm: &[String],
    varm: &[String],
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    modality: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(m) = modality.as_deref() {
        let t = m.trim();
        // `0` is the supported global modality; only non-zero is unsupported.
        if !t.is_empty() && t != "0" {
            return Err("multimodal modify-metadata is not yet supported; \
                        only the global modality (0) is available today"
                .into());
        }
    }

    let uns_json = match uns {
        Some(p) => Some(read_json(p)?),
        None => None,
    };
    let obs_batch = match obs {
        Some(p) => Some(read_parquet(p)?),
        None => None,
    };
    let var_batch = match var {
        Some(p) => Some(read_parquet(p)?),
        None => None,
    };
    let obsm_batches = read_named_npy(obsm, "obsm")?;
    let varm_batches = read_named_npy(varm, "varm")?;

    // `index_auto_threshold` is NOT defaulted to 1000 when some other index flag
    // is set, unlike the conversion commands. This op reads a non-zero threshold
    // as "auto-detect on both axes", so defaulting it would make
    // `--index-var gene_id --obs obs.parquet` silently take the obs axis off
    // carry-forward and onto auto-detect. `0` means "no auto unless asked", which
    // is what an omitted flag means.
    let index = ConversionPredicateIndexOptions {
        index_obs,
        index_var,
        index_preset,
        index_auto_threshold: index_auto_threshold.unwrap_or(0),
    };

    let patch = MetadataPatch {
        uns: uns_json,
        uns_merge: false,
        obs: obs_batch,
        var: var_batch,
        obsm: (!obsm_batches.is_empty()).then_some(obsm_batches),
        varm: (!varm_batches.is_empty()).then_some(varm_batches),
        index,
        modality_id: 0,
    };

    let summary = scx_ops::modify_metadata(file, &patch)?;
    println!(
        "Updated metadata on {} (O(replaced sections), no matrix re-encode). \
         Run 'scx compact' to reclaim orphaned sections.",
        file.display()
    );
    report_index_outcome(&summary);
    Ok(())
}

/// Say what happened to the file's predicate indexes. Silent when there were
/// none — the common `--uns`-only case — and explicit otherwise, because the
/// alternative is a user discovering months later that their queries went back
/// to a full obs scan.
fn report_index_outcome(summary: &scx_ops::ModifyMetadataSummary) {
    // Per-column build outcomes go through the shared emitter every other
    // index-writing command uses, so a `PresetSkipped` / `ForcedColumnError`
    // reads the same here as under `merge` / `compact` / `append`.
    emit_index_summary("modify-metadata", &summary.index);

    if let Some(result) = summary.index.result.as_ref() {
        // Per axis: with "explicit" decided per axis, one side can be carried
        // while the other is rebuilt from the caller's own column list.
        for (axis, columns, carried) in [
            (
                "obs",
                &result.obs_indexed_columns,
                summary.obs_carried_forward,
            ),
            (
                "var",
                &result.var_indexed_columns,
                summary.var_carried_forward,
            ),
        ] {
            if columns.is_empty() {
                continue;
            }
            let how = if carried {
                "carried forward"
            } else {
                "rebuilt"
            };
            println!("  {axis} predicate index {how} over {columns:?}");
        }
    }
    // `*_not_carried_unreported`, not the raw lists: on the carry path
    // `emit_index_summary` above has already named every column that could not
    // be carried, and printing the raw list too warns twice about one column.
    for (axis, columns) in [
        ("obs", summary.obs_not_carried_unreported()),
        ("var", summary.var_not_carried_unreported()),
    ] {
        if !columns.is_empty() {
            eprintln!(
                "  warning: the {axis} predicate index no longer covers {columns:?}; \
                 queries on those columns fall back to a full scan"
            );
        }
    }
}

fn read_json(p: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let f = std::fs::File::open(p).map_err(|e| format!("failed to open '{}': {e}", p.display()))?;
    Ok(serde_json::from_reader(std::io::BufReader::new(f))
        .map_err(|e| format!("failed to parse JSON '{}': {e}", p.display()))?)
}

fn read_parquet(p: &Path) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let f = std::fs::File::open(p)
        .map_err(|e| format!("failed to open Parquet '{}': {e}", p.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(f)
        .map_err(|e| format!("failed to read Parquet '{}': {e}", p.display()))?;
    let schema = builder.schema().clone();
    let mut batches = Vec::new();
    for b in builder.build()? {
        batches.push(b?);
    }
    if batches.is_empty() {
        return Err(format!("Parquet file '{}' has no rows", p.display()).into());
    }
    Ok(arrow::compute::concat_batches(&schema, &batches)?)
}

/// Parse `name=path.npy` items and load each as a dense f32 RecordBatch.
fn read_named_npy(
    items: &[String],
    axis: &str,
) -> Result<Vec<(String, RecordBatch)>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let (name, path) = item
            .split_once('=')
            .ok_or_else(|| format!("--{axis} expects NAME=PATH.npy (got '{item}')"))?;
        out.push((name.to_string(), npy_to_batch(Path::new(path))?));
    }
    Ok(out)
}

/// Read a 2D `.npy` (float32, or float64 cast to f32) into a dense
/// RecordBatch with one Float32 column per array column (names "0".."k-1"),
/// matching the ingest convention.
fn npy_to_batch(p: &Path) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    use ndarray_npy::ReadNpyExt;
    let bytes =
        std::fs::read(p).map_err(|e| format!("failed to open .npy '{}': {e}", p.display()))?;

    let arr = match ndarray::Array2::<f32>::read_npy(Cursor::new(&bytes)) {
        Ok(a) => a,
        Err(_) => ndarray::Array2::<f64>::read_npy(Cursor::new(&bytes))
            .map_err(|e| {
                format!(
                    "'{}' must be a 2D float32/float64 .npy array: {e}",
                    p.display()
                )
            })?
            .mapv(|v| v as f32),
    };

    let (n_rows, n_cols) = arr.dim();
    let mut fields = Vec::with_capacity(n_cols);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(n_cols);
    for c in 0..n_cols {
        let col: Vec<f32> = (0..n_rows).map(|r| arr[[r, c]]).collect();
        fields.push(Field::new(c.to_string(), DataType::Float32, false));
        columns.push(Arc::new(Float32Array::from(col)));
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}
