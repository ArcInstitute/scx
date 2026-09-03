//! Reader for CellBender `remove-background` output HDF5 files.
//!
//! Produces a [`scx_ops::ExternalLayerData`] ready for
//! `scx_ops::attach_external_layer`. All CellBender-specific knowledge lives
//! here; the write side is generic.
//!
//! # On-disk layout (CellRanger v3, written by `write_matrix_to_cellranger_h5`)
//!
//! ```text
//! /matrix/features/{name,id,feature_type,genome}   G-length strings
//! /matrix/barcodes                                 N-length strings
//! /matrix/{data,indices,indptr}                    CSC of [G x N]
//! /matrix/shape                                    i32 DATASET = [G, N]
//! /droplet_latents/{barcode_indices_for_latents,gene_expression_encoding,
//!                   cell_size,cell_probability,droplet_efficiency,
//!                   background_fraction}
//! /global_latents/{ambient_expression, ...}
//! /metadata/{...}
//! ```
//!
//! Four things about that layout are easy to get wrong:
//!
//! * **The matrix needs no transpose.** CellBender transposes before writing,
//!   so on-disk CSC of `[G x N]` *is* CSR of `[N x G]` — a free reinterpretation.
//! * **`shape` is a dataset, not an attribute**, unlike the synthetic fixtures
//!   the 10x reader was originally written against.
//! * **Strings are fixed-length ASCII** (PyTables `create_carray`), not the
//!   variable-length UTF-8 h5py emits.
//! * **The droplet latents are not positionally aligned to the matrix rows** in
//!   general — see [`LatentAlignment`].

use std::collections::BTreeMap;
use std::path::Path;

use arrow::array::{ArrayRef, BooleanArray, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

use scx_ops::ExternalLayerData;

use crate::file_checksum::blake3_of_file;
use crate::h5ad::read::{read_f32_dataset, read_i64_dataset, read_string_dataset};
use crate::pipeline::ConvertError;
use crate::warnings::{ConvertWarning, WarningSink};

/// Which of CellBender's two output files this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellBenderOutputKind {
    /// `<name>.h5` — every input barcode, in the input's row order.
    Full,
    /// `<name>_filtered.h5` — cells only, in **descending-UMI** order.
    Filtered,
    /// Could not be determined.
    Unknown,
}

/// How `droplet_latents/*` maps onto the matrix rows.
///
/// CellBender does not record this, and it differs between its two outputs, so
/// it has to be inferred from array lengths and then validated:
///
/// * **Scatter** (the full file): latents are length `n_analyzed`, and
///   `barcode_indices_for_latents` maps each to a matrix row.
/// * **Positional** (the filtered file): latents are length `n_rows`, one per
///   matrix row — but `barcode_indices_for_latents` is *still* the length-
///   `n_analyzed` array, so it must not be trusted there.
///
/// Neither is inferred from the filename: `--fpr a b c` renames the outputs,
/// and users rename files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LatentAlignment {
    Scatter,
    Positional,
    /// Ambiguous or contradictory; latents are skipped rather than guessed.
    #[default]
    None,
}

/// Which feature column supplied the join key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureKey {
    /// `features/id`, preferring Ensembl IDs.
    #[default]
    Auto,
    Id,
    Name,
}

#[derive(Clone)]
pub struct CellBenderReadOptions {
    /// Prefix for emitted obs/var column names.
    pub column_prefix: String,
    /// Import `droplet_latents/gene_expression_encoding` as an obsm matrix.
    ///
    /// Off by default: it is `n_obs x z_dim` dense f32 written as a whole-
    /// section replace, and `null` for every unmatched row.
    pub latent_embedding: bool,
    /// Skip `/metadata` and `/global_latents` arrays longer than this when
    /// building `uns`, so per-barcode arrays never bloat the JSON blob.
    pub uns_array_max_len: usize,
    /// The `uns` key the run's diagnostics record lands under. `None` omits
    /// the record altogether.
    pub uns_key: Option<String>,
    pub feature_key: FeatureKey,
}

impl Default for CellBenderReadOptions {
    fn default() -> Self {
        Self {
            column_prefix: "cellbender_".to_string(),
            latent_embedding: false,
            uns_array_max_len: 10_000,
            uns_key: Some("cellbender".to_string()),
            feature_key: FeatureKey::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CellBenderInfo {
    pub output_kind: CellBenderOutputKind,
    pub latent_alignment: LatentAlignment,
    pub n_rows: usize,
    pub n_features: usize,
    pub n_barcodes_analyzed: Option<usize>,
    pub n_features_analyzed: Option<usize>,
    pub estimator: Option<String>,
    pub target_false_positive_rate: Option<f64>,
    pub feature_key_used: &'static str,
    pub all_values_integer: bool,
    pub z_dim: Option<usize>,
}

pub struct CellBenderOutput {
    pub data: ExternalLayerData,
    pub info: CellBenderInfo,
}

/// Cheap probe: does this look like a `remove-background` output rather than a
/// plain CellRanger matrix?
///
/// `/droplet_latents` is the discriminator — no 10x file has one.
pub fn is_cellbender_h5(path: &Path) -> bool {
    let Ok(file) = hdf5::File::open(path) else {
        return false;
    };
    file.group("droplet_latents").is_ok() && file.group("matrix").is_ok()
}

/// Read a CellBender `remove-background` output into an [`ExternalLayerData`].
///
/// **Not streaming**, unlike the h5ad/h5mu ingest pipelines, and it does not
/// honour `memory_budget`: the barcode-keyed join reorders rows arbitrarily, so
/// the matrix has to be resident. That is bounded in practice — CellBender
/// analyses at most `total_droplets_included` barcodes (default 25k, heuristic
/// cap 70k) — but a full all-droplet output on a very wide feature axis will
/// still cost several GB.
pub fn read_cellbender_h5(
    path: &Path,
    opts: &CellBenderReadOptions,
    sink: &mut WarningSink,
) -> Result<CellBenderOutput, ConvertError> {
    let file = hdf5::File::open(path)?;
    let matrix = file.group("matrix").map_err(|_| {
        ConvertError::Other(format!(
            "'{}' has no /matrix group; this does not look like a CellBender \
             remove-background output",
            path.display()
        ))
    })?;
    if file.group("droplet_latents").is_err() && file.group("global_latents").is_err() {
        return Err(ConvertError::Other(format!(
            "'{}' has a /matrix group but neither /droplet_latents nor \
             /global_latents; it looks like a plain 10x CellRanger file. Use \
             `scx convert --from 10x` to ingest it instead.",
            path.display()
        )));
    }

    // --- Matrix ------------------------------------------------------------
    // shape = [n_genes, n_cells]; CSC of [G x N] == CSR of [N x G].
    let shape = read_i64_dataset(&matrix.dataset("shape")?)?;
    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "/matrix/shape must have 2 entries, found {}",
            shape.len()
        )));
    }
    let n_features = shape[0] as usize;
    let n_rows = shape[1] as usize;

    let indptr_i64 = read_i64_dataset(&matrix.dataset("indptr")?)?;
    if indptr_i64.len() != n_rows + 1 {
        return Err(ConvertError::Other(format!(
            "/matrix/indptr has {} entries but /matrix/shape implies {} rows + 1. \
             A CellRanger-v2 layout (/matrix_v2) is not supported.",
            indptr_i64.len(),
            n_rows
        )));
    }
    // A damaged file can carry negative offsets/indices; `as u64` would wrap
    // them into enormous values and panic somewhere downstream. Reject with a
    // message that names the file instead.
    if let Some(bad) = indptr_i64.iter().find(|v| **v < 0) {
        return Err(ConvertError::Other(format!(
            "/matrix/indptr contains a negative offset ({bad}); '{}' is corrupt",
            path.display()
        )));
    }
    let indptr: Vec<u64> = indptr_i64.iter().map(|v| *v as u64).collect();
    let indices_i32 = crate::h5ad::read::read_i32_dataset(&matrix.dataset("indices")?)?;
    if let Some(bad) = indices_i32.iter().find(|v| **v < 0) {
        return Err(ConvertError::Other(format!(
            "/matrix/indices contains a negative index ({bad}); '{}' is corrupt",
            path.display()
        )));
    }
    let indices: Vec<u32> = indices_i32.into_iter().map(|v| v as u32).collect();
    let values = read_f32_dataset(&matrix.dataset("data")?)?;

    let row_keys = read_string_dataset(&matrix.dataset("barcodes")?)?;
    if row_keys.len() != n_rows {
        return Err(ConvertError::Other(format!(
            "/matrix/barcodes has {} entries but /matrix/shape implies {n_rows} rows",
            row_keys.len()
        )));
    }

    let (col_keys, feature_key_used) = read_feature_keys(&matrix, n_features, opts)?;

    let all_values_integer = values
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && v.fract() == 0.0);

    // --- Latents ------------------------------------------------------------
    let latents = read_droplet_latents(&file, n_rows, sink)?;
    let output_kind = match latents.alignment {
        LatentAlignment::Scatter => CellBenderOutputKind::Full,
        LatentAlignment::Positional => CellBenderOutputKind::Filtered,
        LatentAlignment::None => CellBenderOutputKind::Unknown,
    };
    warn_on_filename_disagreement(path, output_kind, sink);

    let metadata = read_group_scalars(&file, "metadata", opts.uns_array_max_len);
    let global = read_group_scalars(&file, "global_latents", opts.uns_array_max_len);

    let analyzed_barcodes = read_usize_array(&file, "metadata/barcodes_analyzed_inds");
    let analyzed_features = read_usize_array(&file, "metadata/features_analyzed_inds");

    let row_annotations =
        build_row_annotations(&latents, &analyzed_barcodes, n_rows, &opts.column_prefix);
    let col_annotations = build_col_annotations(
        &file,
        &analyzed_features,
        n_features,
        &opts.column_prefix,
        sink,
    )?;

    let row_embeddings = if opts.latent_embedding {
        build_latent_embedding(&latents, n_rows, sink)
    } else {
        Vec::new()
    };

    let info = CellBenderInfo {
        output_kind,
        latent_alignment: latents.alignment,
        n_rows,
        n_features,
        n_barcodes_analyzed: analyzed_barcodes.as_ref().map(|v| v.len()),
        n_features_analyzed: analyzed_features.as_ref().map(|v| v.len()),
        estimator: metadata
            .get("estimator")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        target_false_positive_rate: metadata
            .get("target_false_positive_rate")
            .and_then(|v| v.as_f64()),
        feature_key_used,
        all_values_integer,
        z_dim: latents.z_dim,
    };

    let mut uns = serde_json::Map::new();
    if let Some(key) = &opts.uns_key {
        uns.insert(key.clone(), build_uns(path, &file, &info, metadata, global));
    }

    Ok(CellBenderOutput {
        data: ExternalLayerData {
            row_keys,
            col_keys,
            indptr,
            indices,
            values,
            row_annotations,
            row_embeddings,
            col_annotations,
            uns,
            source_checksum: blake3_of_file(path).ok(),
            source_name: path.file_name().map(|s| s.to_string_lossy().to_string()),
        },
        info,
    })
}

// ---------------------------------------------------------------------------
// Feature keys
// ---------------------------------------------------------------------------

/// CellBender fills a missing `features/id` with `NA_{i}` placeholders
/// (`io.py`), which would join against nothing. Detect that and fall back to
/// `features/name`.
fn all_placeholder_ids(ids: &[String]) -> bool {
    !ids.is_empty() && ids.iter().enumerate().all(|(i, s)| s == &format!("NA_{i}"))
}

fn read_feature_keys(
    matrix: &hdf5::Group,
    n_features: usize,
    opts: &CellBenderReadOptions,
) -> Result<(Vec<String>, &'static str), ConvertError> {
    let features = matrix
        .group("features")
        .map_err(|_| ConvertError::Other("/matrix/features group is missing".to_string()))?;
    let read = |name: &str| -> Option<Vec<String>> {
        features
            .dataset(name)
            .ok()
            .and_then(|ds| read_string_dataset(&ds).ok())
            .filter(|v| v.len() == n_features)
    };
    let ids = read("id");
    let names = read("name");

    let (keys, used) = match opts.feature_key {
        FeatureKey::Id => (ids, "id"),
        FeatureKey::Name => (names, "name"),
        FeatureKey::Auto => match (&ids, &names) {
            (Some(i), Some(_)) if all_placeholder_ids(i) => (names.clone(), "name"),
            (Some(_), _) => (ids.clone(), "id"),
            (None, Some(_)) => (names.clone(), "name"),
            _ => (None, "id"),
        },
    };

    keys.map(|k| (k, used)).ok_or_else(|| {
        ConvertError::Other(format!(
            "/matrix/features has no usable '{used}' dataset of length {n_features}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Droplet latents
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Latents {
    alignment: LatentAlignment,
    /// For each matrix row, the latent index that describes it.
    latent_of_row: Vec<Option<usize>>,
    cell_probability: Option<Vec<f32>>,
    cell_size: Option<Vec<f32>>,
    droplet_efficiency: Option<Vec<f32>>,
    background_fraction: Option<Vec<f32>>,
    embedding: Option<(Vec<f32>, usize)>,
    z_dim: Option<usize>,
}

fn read_droplet_latents(
    file: &hdf5::File,
    n_rows: usize,
    sink: &mut WarningSink,
) -> Result<Latents, ConvertError> {
    let Ok(group) = file.group("droplet_latents") else {
        return Ok(Latents::default());
    };
    let f32_vec = |name: &str| -> Option<Vec<f32>> {
        group
            .dataset(name)
            .ok()
            .and_then(|ds| read_f32_dataset(&ds).ok())
    };

    let cell_probability = f32_vec("cell_probability");
    let Some(probs) = &cell_probability else {
        return Ok(Latents::default());
    };
    let n_latents = probs.len();

    let indices: Option<Vec<usize>> = group
        .dataset("barcode_indices_for_latents")
        .ok()
        .and_then(|ds| read_i64_dataset(&ds).ok())
        .map(|v| v.into_iter().map(|x| x as usize).collect());
    let barcodes_analyzed = read_string_vec(file, "metadata/barcodes_analyzed");
    let barcodes = read_string_vec(file, "matrix/barcodes");

    // Scatter is valid when the index array matches the latents in length, is
    // in range and unique, and — when checkable — agrees with the analyzed
    // barcode list.
    let scatter_ok = indices.as_ref().is_some_and(|idx| {
        idx.len() == n_latents
            && idx.iter().all(|&i| i < n_rows)
            && {
                let mut seen = std::collections::HashSet::new();
                idx.iter().all(|i| seen.insert(*i))
            }
            && match (&barcodes_analyzed, &barcodes) {
                (Some(ba), Some(bc)) if ba.len() == n_latents => {
                    idx.iter().enumerate().all(|(k, &row)| bc[row] == ba[k])
                }
                _ => true,
            }
    });

    // Positional is valid when there is exactly one latent per matrix row.
    let positional_ok = n_latents == n_rows
        && match (&barcodes_analyzed, &barcodes) {
            (Some(ba), Some(bc)) if ba.len() == n_latents => ba == bc,
            _ => true,
        };

    let (alignment, latent_of_row) = match (scatter_ok, positional_ok) {
        (true, _) => {
            let idx = indices.as_ref().expect("scatter_ok implies Some");
            let mut map = vec![None; n_rows];
            for (k, &row) in idx.iter().enumerate() {
                map[row] = Some(k);
            }
            (LatentAlignment::Scatter, map)
        }
        (false, true) => (LatentAlignment::Positional, (0..n_rows).map(Some).collect()),
        (false, false) => {
            sink.emit(ConvertWarning::SkippedUnsKey {
                key: "droplet_latents".to_string(),
                reason: format!(
                    "cannot align {n_latents} latents to {n_rows} matrix rows: \
                     barcode_indices_for_latents is {} and the lengths do not \
                     imply a positional match. Per-cell CellBender columns are \
                     omitted rather than guessed.",
                    indices
                        .as_ref()
                        .map(|v| format!("length {}", v.len()))
                        .unwrap_or_else(|| "absent".to_string())
                ),
            });
            // Return early rather than falling through with an all-`None` map:
            // that would still emit the obs columns, just entirely null, which
            // reads as "CellBender analysed nothing" instead of "we could not
            // tell how these line up".
            return Ok(Latents {
                alignment: LatentAlignment::None,
                latent_of_row: vec![None; n_rows],
                ..Default::default()
            });
        }
    };

    let embedding = group
        .dataset("gene_expression_encoding")
        .ok()
        .and_then(|ds| {
            let shape = ds.shape();
            if shape.len() != 2 {
                return None;
            }
            let flat: Vec<f32> = ds.read_raw().ok()?;
            Some((flat, shape[1]))
        });
    let z_dim = embedding.as_ref().map(|(_, d)| *d);

    Ok(Latents {
        alignment,
        latent_of_row,
        cell_probability,
        cell_size: f32_vec("cell_size"),
        droplet_efficiency: f32_vec("droplet_efficiency"),
        background_fraction: f32_vec("background_fraction"),
        embedding,
        z_dim,
    })
}

fn warn_on_filename_disagreement(path: &Path, kind: CellBenderOutputKind, sink: &mut WarningSink) {
    let looks_filtered = path
        .file_stem()
        .map(|s| s.to_string_lossy().ends_with("_filtered"))
        .unwrap_or(false);
    let disagrees = match kind {
        CellBenderOutputKind::Full => looks_filtered,
        CellBenderOutputKind::Filtered => !looks_filtered,
        CellBenderOutputKind::Unknown => false,
    };
    if disagrees {
        sink.emit(ConvertWarning::SkippedUnsKey {
            key: "cellbender_output_kind".to_string(),
            reason: format!(
                "latent alignment implies a {kind:?} output but the filename says \
                 otherwise; trusting the file contents. (`--fpr` renames outputs, \
                 so the filename is only a hint.)"
            ),
        });
    }
}

// ---------------------------------------------------------------------------
// Annotations
// ---------------------------------------------------------------------------

/// Gather a latent array onto the matrix row axis, `null` where a row has no
/// latent (never analysed).
fn gather_latent(values: &Option<Vec<f32>>, latent_of_row: &[Option<usize>]) -> Option<ArrayRef> {
    let v = values.as_ref()?;
    let arr: Float32Array = latent_of_row
        .iter()
        .map(|k| k.and_then(|i| v.get(i).copied()))
        .collect();
    Some(Arc::new(arr) as ArrayRef)
}

fn build_row_annotations(
    latents: &Latents,
    analyzed_barcodes: &Option<Vec<usize>>,
    n_rows: usize,
    prefix: &str,
) -> Option<RecordBatch> {
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    for (suffix, values) in [
        ("cell_probability", &latents.cell_probability),
        ("cell_size", &latents.cell_size),
        ("droplet_efficiency", &latents.droplet_efficiency),
        ("background_fraction", &latents.background_fraction),
    ] {
        if let Some(arr) = gather_latent(values, &latents.latent_of_row) {
            fields.push(Field::new(
                format!("{prefix}{suffix}"),
                DataType::Float32,
                true,
            ));
            columns.push(arr);
        }
    }

    if let Some(inds) = analyzed_barcodes {
        let set: std::collections::HashSet<usize> = inds.iter().copied().collect();
        let arr: BooleanArray = (0..n_rows).map(|i| Some(set.contains(&i))).collect();
        fields.push(Field::new(
            format!("{prefix}analyzed"),
            DataType::Boolean,
            true,
        ));
        columns.push(Arc::new(arr) as ArrayRef);
    }

    if fields.is_empty() {
        return None;
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).ok()
}

fn build_col_annotations(
    file: &hdf5::File,
    analyzed_features: &Option<Vec<usize>>,
    n_features: usize,
    prefix: &str,
    sink: &mut WarningSink,
) -> Result<Option<RecordBatch>, ConvertError> {
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    // Full-G, zero-filled by CellBender for genes it excluded from inference.
    // Goes to var rather than uns: it is a length-G vector.
    if let Ok(ds) = file.dataset("global_latents/ambient_expression") {
        match read_f32_dataset(&ds) {
            Ok(v) if v.len() == n_features => {
                fields.push(Field::new(
                    format!("{prefix}ambient_expression"),
                    DataType::Float32,
                    true,
                ));
                columns.push(Arc::new(Float32Array::from(v)) as ArrayRef);
            }
            Ok(v) => sink.emit(ConvertWarning::SkippedColumn {
                group: "global_latents".to_string(),
                name: "ambient_expression".to_string(),
                reason: format!("length {} != n_features {n_features}", v.len()),
            }),
            Err(e) => sink.emit(ConvertWarning::SkippedColumn {
                group: "global_latents".to_string(),
                name: "ambient_expression".to_string(),
                reason: e.to_string(),
            }),
        }
    }

    if let Some(inds) = analyzed_features {
        let set: std::collections::HashSet<usize> = inds.iter().copied().collect();
        let arr: BooleanArray = (0..n_features).map(|i| Some(set.contains(&i))).collect();
        fields.push(Field::new(
            format!("{prefix}analyzed"),
            DataType::Boolean,
            true,
        ));
        columns.push(Arc::new(arr) as ArrayRef);
    }

    if fields.is_empty() {
        return Ok(None);
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).ok())
}

fn build_latent_embedding(
    latents: &Latents,
    n_rows: usize,
    sink: &mut WarningSink,
) -> Vec<(String, RecordBatch)> {
    let Some((flat, z_dim)) = &latents.embedding else {
        return Vec::new();
    };
    let n_latents = if *z_dim == 0 { 0 } else { flat.len() / z_dim };
    // The alignment step validated every index against the *latent* arrays, so
    // an index past the embedding's own row count means the two disagree —
    // corruption, not an un-analysed droplet. Say so instead of emitting a
    // column of nulls that reads as "CellBender analysed nothing".
    if latents
        .latent_of_row
        .iter()
        .any(|k| k.is_some_and(|i| i >= n_latents))
    {
        sink.emit(ConvertWarning::SkippedObsm {
            name: "X_cellbender_latent".to_string(),
            reason: format!(
                "gene_expression_encoding has {n_latents} rows but the droplet \
                 latents index beyond it; the two are inconsistent"
            ),
        });
        return Vec::new();
    }
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    for d in 0..*z_dim {
        let arr: Float32Array = latents
            .latent_of_row
            .iter()
            .map(|k| k.map(|i| flat[i * z_dim + d]))
            .collect();
        fields.push(Field::new(format!("z{d}"), DataType::Float32, true));
        columns.push(Arc::new(arr) as ArrayRef);
    }
    match RecordBatch::try_new(Arc::new(Schema::new(fields)), columns) {
        Ok(b) => vec![("X_cellbender_latent".to_string(), b)],
        Err(e) => {
            sink.emit(ConvertWarning::SkippedObsm {
                name: "X_cellbender_latent".to_string(),
                reason: e.to_string(),
            });
            let _ = n_rows;
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// uns
// ---------------------------------------------------------------------------

/// Read every member of an HDF5 group into JSON.
///
/// Enumerated rather than probed against a fixed key list: CellBender flattens
/// nested dicts, so `learning_curve` arrives as `learning_curve_train_elbo`
/// and friends, and the exact key set depends on the run's options. Rank-0
/// datasets and length-1 arrays are unwrapped to scalars.
fn read_group_scalars(
    file: &hdf5::File,
    group_name: &str,
    max_len: usize,
) -> BTreeMap<String, serde_json::Value> {
    let mut out = BTreeMap::new();
    let Ok(group) = file.group(group_name) else {
        return out;
    };
    let Ok(names) = group.member_names() else {
        return out;
    };
    for name in names {
        let Ok(ds) = group.dataset(&name) else {
            continue;
        };
        if let Some(v) = dataset_to_json(&ds, max_len) {
            out.insert(name, v);
        }
    }
    out
}

fn dataset_to_json(ds: &hdf5::Dataset, max_len: usize) -> Option<serde_json::Value> {
    let rank0 = ds.shape().is_empty();

    // Strings first — a rank-0 string is a common metadata shape.
    if let Ok(desc) = ds.dtype().and_then(|d| d.to_descriptor()) {
        use hdf5::types::TypeDescriptor::*;
        if matches!(
            desc,
            VarLenUnicode | VarLenAscii | FixedUnicode(_) | FixedAscii(_)
        ) {
            if rank0 {
                let s: hdf5::types::VarLenUnicode = ds.read_scalar().ok()?;
                return Some(serde_json::json!(s.to_string()));
            }
            let v = read_string_dataset(ds).ok()?;
            if v.len() > max_len {
                return None;
            }
            // CellBender wraps scalars in a length-1 array to work around a
            // scanpy loading bug; unwrap them back.
            return Some(if v.len() == 1 {
                serde_json::json!(v[0])
            } else {
                serde_json::json!(v)
            });
        }
    }

    if rank0 {
        let v: f64 = ds.read_scalar().ok()?;
        return Some(serde_json::json!(v));
    }
    let v: Vec<f64> = ds.read_raw().ok()?;
    if v.len() > max_len {
        return None;
    }
    Some(if v.len() == 1 {
        serde_json::json!(v[0])
    } else {
        serde_json::json!(v)
    })
}

fn read_string_vec(file: &hdf5::File, path: &str) -> Option<Vec<String>> {
    file.dataset(path)
        .ok()
        .and_then(|ds| read_string_dataset(&ds).ok())
}

fn read_usize_array(file: &hdf5::File, path: &str) -> Option<Vec<usize>> {
    file.dataset(path)
        .ok()
        .and_then(|ds| read_i64_dataset(&ds).ok())
        .map(|v| v.into_iter().map(|x| x.max(0) as usize).collect())
}

fn build_uns(
    path: &Path,
    _file: &hdf5::File,
    info: &CellBenderInfo,
    mut metadata: BTreeMap<String, serde_json::Value>,
    mut global: BTreeMap<String, serde_json::Value>,
) -> serde_json::Value {
    // `ambient_expression` is a length-G vector that belongs in var.
    global.remove("ambient_expression");
    // Per-barcode arrays belong nowhere near uns.
    metadata.remove("barcodes_analyzed");
    metadata.remove("barcodes_analyzed_inds");
    metadata.remove("features_analyzed_inds");

    serde_json::json!({
        "version": 1,
        "source_file": path.file_name().map(|s| s.to_string_lossy().to_string()),
        "output_kind": format!("{:?}", info.output_kind).to_lowercase(),
        "latent_alignment": format!("{:?}", info.latent_alignment).to_lowercase(),
        "n_rows_in_source": info.n_rows,
        "n_features_in_source": info.n_features,
        "n_barcodes_analyzed": info.n_barcodes_analyzed,
        "n_features_analyzed": info.n_features_analyzed,
        "feature_key_used": info.feature_key_used,
        "all_values_integer": info.all_values_integer,
        "z_dim": info.z_dim,
        "estimator": info.estimator,
        "target_false_positive_rate": info.target_false_positive_rate,
        "global_latents": global,
        "metadata": metadata,
    })
}

#[cfg(test)]
#[path = "cellbender_tests.rs"]
mod tests;
