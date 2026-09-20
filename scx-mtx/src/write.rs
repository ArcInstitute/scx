//! Write SCX data to Cell Ranger–style MTX directories.
//!
//! Output is always gzip-compressed:
//! - `matrix.mtx.gz`
//! - `barcodes.tsv.gz`
//! - `features.tsv.gz`

use std::io::{BufWriter, Write};
use std::path::Path;

use arrow::array::{Array, AsArray};
use flate2::write::GzEncoder;
use flate2::Compression;

use scx_codec::{ShardValuesNative, ValueEncoding};
use scx_format_io::reader::ScxReader;
use scx_format_io::{FullCatalogEntry, ShardStats};

use crate::error::MtxError;

/// Write an SCX file to a Cell Ranger–style MTX directory.
///
/// Creates `output_dir` if it doesn't exist, then writes:
/// - `matrix.mtx.gz` — sparse matrix in COO format
/// - `barcodes.tsv.gz` — cell barcodes
/// - `features.tsv.gz` — gene/feature metadata
///
/// Exports the file's only modality; a multimodal file is refused. Use
/// [`write_scx_to_mtx_for`] to name one.
///
/// Logically deleted cells are **excluded**. MatrixMarket has no notion of a
/// deletion vector, so unlike an SCX→SCX rewrite there is nothing to carry the
/// deletion in — the only faithful export is one that leaves the rows out. X and
/// obs are filtered by the same mask and must stay that way: `barcodes.tsv.gz`
/// with more lines than the matrix has columns is a directory Cell Ranger and
/// Scanpy both reject.
///
/// Peak memory is one decoded shard, not the whole matrix.
pub fn write_scx_to_mtx(scx_path: &Path, output_dir: &Path) -> Result<(), MtxError> {
    write_scx_to_mtx_for(scx_path, output_dir, None)
}

/// Modality-scoped [`write_scx_to_mtx`]. `modality` names one modality of a
/// multimodal file, and must be `None` for a single-modality one.
///
/// A MatrixMarket directory describes exactly one matrix over one feature
/// space, and a multimodal SCX file is not that. Each modality independently
/// tiles the shared obs axis, so the flattened CSR shard list carries
/// overlapping row ranges: exporting it unscoped stacks the modalities into an
/// `n_obs × n_modalities` matrix over mixed column spaces. It also mislabels
/// the result — a multimodal file has no global `var` section (each modality
/// owns `var/<name>`), so the features table falls back to synthetic `gene_i`
/// names. Both are refused here rather than written.
pub fn write_scx_to_mtx_for(
    scx_path: &Path,
    output_dir: &Path,
    modality: Option<&str>,
) -> Result<(), MtxError> {
    // Everything fallible that does not touch the destination happens first.
    // A refused export — an unreadable source, a multimodal file with no
    // `modality`, an unknown name — must leave the filesystem exactly as it
    // found it, including not creating `output_dir`.
    let reader = ScxReader::open(scx_path)?;
    let modality_id = resolve_modality(&reader, modality)?;

    // One mask for X and obs. They must move together (see above), so both
    // halves take it from the same call rather than from the global-only
    // `read_obs_filtered` / `read_all_csr_shards_filtered` pair.
    let keep = reader.deletion_keep_mask_for(modality_id)?;
    let shards = reader.catalog().csr_shards_for_modality(modality_id);

    let n_obs = match keep.as_deref() {
        Some(mask) => mask.iter().filter(|&&k| k).count(),
        None => reader.n_obs() as usize,
    };
    // Prefer the modality table's `n_vars`: a multimodal file's header carries
    // the file-wide maximum, not this modality's width.
    let n_vars = match reader.modality_info(modality_id) {
        Some(info) => info.n_vars as usize,
        None => reader.n_vars() as usize,
    };

    std::fs::create_dir_all(output_dir)?;

    // Every member is written to a temp name and renamed into place only once
    // all three have succeeded, so a failure part-way through leaves a
    // previous export byte-identical rather than half-replaced. `Staged`
    // removes anything left over on drop, including on the error paths below.
    let mut staged = Staged::new(output_dir);

    write_matrix_mtx(
        staged.temp_for(MATRIX_MTX),
        &reader,
        &shards,
        keep.as_deref(),
        n_obs,
        n_vars,
    )?;

    // Synthetic barcodes are a fallback for *genuinely absent* obs only; a
    // decode/checksum failure is corruption and must abort rather than
    // silently emit fabricated `cell_i` IDs (SCX-009).
    match reader.read_obs() {
        Ok(obs) => {
            let obs = match keep.as_deref() {
                Some(mask) => scx_format_io::filter_batch_by_keep_mask(&obs, mask)?,
                None => obs,
            };
            write_barcodes_tsv(staged.temp_for(BARCODES_TSV), &obs)?
        }
        Err(scx_format_io::error::ScxError::SectionNotFound(_)) => {
            write_synthetic_barcodes(staged.temp_for(BARCODES_TSV), n_obs)?;
        }
        Err(e) => return Err(e.into()),
    }

    // Same absence-vs-corruption rule as obs. `read_var_for` routes
    // `modality_id == 0` to `read_var()`, so this one call covers both the
    // multimodal per-modality section and the global one.
    match reader.read_var_for(modality_id) {
        Ok(var) => write_features_tsv(staged.temp_for(FEATURES_TSV), &var)?,
        Err(scx_format_io::error::ScxError::SectionNotFound(_)) => {
            write_synthetic_features(staged.temp_for(FEATURES_TSV), n_vars)?;
        }
        Err(e) => return Err(e.into()),
    }

    staged.commit()
}

const MATRIX_MTX: &str = "matrix.mtx.gz";
const BARCODES_TSV: &str = "barcodes.tsv.gz";
const FEATURES_TSV: &str = "features.tsv.gz";

/// Members written under a temp name and renamed into place together.
///
/// The point is the failure path, not the happy one: before this existed a
/// half-finished export left the destination with a fresh `matrix.mtx.gz`
/// beside a previous run's `barcodes.tsv.gz`, describing two different
/// matrices. Anything still staged when this drops is removed, so an early
/// `?` cannot leave debris either.
struct Staged {
    dir: std::path::PathBuf,
    members: Vec<(std::path::PathBuf, std::path::PathBuf)>,
}

impl Staged {
    fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            members: Vec::new(),
        }
    }

    /// Reserve `name` and hand back the temp path to write to. The temp lives
    /// in the destination directory so the commit is a rename, not a copy.
    fn temp_for(&mut self, name: &str) -> &Path {
        let temp = self
            .dir
            .join(format!(".{name}.scx-tmp-{}", std::process::id()));
        self.members.push((temp, self.dir.join(name)));
        &self.members.last().expect("just pushed").0
    }

    fn commit(mut self) -> Result<(), MtxError> {
        for (temp, final_path) in std::mem::take(&mut self.members) {
            std::fs::rename(&temp, &final_path)?;
        }
        Ok(())
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        for (temp, _) in &self.members {
            let _ = std::fs::remove_file(temp);
        }
    }
}

/// Resolve `--modality` against the file. Mirrors the refusal the streaming
/// h5ad export makes and the `is_multimodal()` / `modality_id()` idiom every
/// other modality-aware entry point uses.
fn resolve_modality(reader: &ScxReader, modality: Option<&str>) -> Result<u8, MtxError> {
    let available = || {
        reader
            .modality_names()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    };
    match (reader.is_multimodal(), modality) {
        (true, None) => Err(MtxError::ModalityRequired {
            available: available(),
        }),
        (true, Some(name)) => reader
            .modality_id(name)
            .ok_or_else(|| MtxError::UnknownModality {
                requested: name.to_string(),
                available: available(),
            }),
        (false, Some(name)) => Err(MtxError::ModalityNotApplicable {
            requested: name.to_string(),
        }),
        (false, None) => Ok(0),
    }
}

/// Everything the MatrixMarket banner and size line must state before the
/// first body line can be written.
struct MatrixHeaderFacts {
    /// Non-zeros the body will emit, after the keep mask.
    nnz: u64,
    /// Whether every kept value is a finite non-negative integer, which
    /// selects both the declared type and the body's per-value format.
    all_integer: bool,
}

/// Write `matrix.mtx.gz` in COO format, one shard at a time.
///
/// Peak memory is one decoded shard plus flate2's window, not the whole CSR.
/// The two passes are forced by the format, not chosen: MatrixMarket's size
/// line is the first data line, and the declared type also formats every
/// value below it, so neither can be deferred until the body is known.
fn write_matrix_mtx(
    path: &Path,
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
    keep: Option<&[bool]>,
    n_obs: usize,
    n_vars: usize,
) -> Result<(), MtxError> {
    let facts = scan_header_facts(reader, shards, keep)?;

    let file = std::fs::File::create(path)?;
    let gz = GzEncoder::new(file, Compression::default());
    let mut w = BufWriter::new(gz);

    let type_str = if facts.all_integer { "integer" } else { "real" };
    writeln!(w, "%%MatrixMarket matrix coordinate {} general", type_str)?;
    writeln!(w, "% Generated by scx-mtx from SCX format")?;
    // SCX stores cells × genes (obs × var), but 10x/Cell Ranger Matrix Market
    // is features × barcodes (genes × cells). Transpose on export (SCX-010):
    // the size line is `n_vars n_obs nnz` and each triplet is emitted as
    // `col(gene) row(cell) value`. Iterating the CSR in cell (obs) order means
    // entries come out grouped by column (barcode) and ascending within a
    // column by row (feature) — exactly Cell Ranger's column-major layout, and
    // a shard-at-a-time walk in `row_start` order preserves it.
    writeln!(w, "{} {} {}", n_vars, n_obs, facts.nnz)?;

    // Ordinal among *kept* rows, so a triplet's barcode index and the line it
    // names in `barcodes.tsv.gz` stay the same number.
    let mut out_row: usize = 0;
    let mut emitted: u64 = 0;

    for entry in shards {
        let row_start = require_stats(entry)?.row_start as usize;
        let (indptr, indices, values) = reader.read_shard_from_entry_native(entry)?;
        let n_rows = indptr.len().saturating_sub(1);
        for r in 0..n_rows {
            if !row_is_kept(keep, row_start + r) {
                continue;
            }
            for idx in indptr[r] as usize..indptr[r + 1] as usize {
                let col = indices[idx] + 1;
                let row = out_row + 1;
                match &values {
                    // Integer-encoded values stay `u32` the whole way out. The
                    // assembled export decoded them to `f32` first and wrote
                    // `val as i64`, which silently rounds any count above 2²⁴ —
                    // the same defect MTX *ingest* had, in the other direction.
                    ShardValuesNative::U32(v) => writeln!(w, "{} {} {}", col, row, v[idx])?,
                    ShardValuesNative::F32(v) if facts.all_integer => {
                        // Write as integer to avoid unnecessary decimal points.
                        writeln!(w, "{} {} {}", col, row, v[idx] as i64)?
                    }
                    ShardValuesNative::F32(v) => writeln!(w, "{} {} {}", col, row, v[idx])?,
                }
                emitted += 1;
            }
            out_row += 1;
        }
    }

    w.flush()?;

    // CLI8: the declared counts must equal what the body emitted. This used to
    // be a `debug_assert_eq!`, which in a release build lets a disagreement
    // ship as a file strict readers reject part-way through.
    if emitted != facts.nnz {
        return Err(MtxError::Other(format!(
            "MTX writer: declared {} non-zeros but emitted {emitted}",
            facts.nnz
        )));
    }
    if out_row != n_obs {
        return Err(MtxError::Other(format!(
            "MTX writer: declared {n_obs} columns (barcodes) but the CSR shards \
             covered {out_row} kept rows"
        )));
    }
    Ok(())
}

/// Compute the header facts, taking each from the cheapest source that can
/// answer it.
///
/// `nnz` mirrors the streaming SCX → h5ad export's `precompute_total_nnz`:
/// catalog `stats.nnz` with zero decode when nothing is deleted, and an
/// indptr-only decode when a keep mask is active.
///
/// `all_integer` is derived from the **values**, never from the shard's
/// `ValueEncoding`, and that is load-bearing rather than fussy: an h5ad-sourced
/// file routinely holds integral counts in a `Float32`-encoded shard, so
/// reading the flag off the encoding would flip the declared type of every such
/// export from `integer` to `real`. (`benchmarks/comprehensive/thresholds.yaml`
/// floors `mtx_header_integer` and `mtx_values_all_integral` on exactly that
/// file shape.) An integer-encoded shard *is* provably finite, non-negative and
/// integral, so it answers from its 76-byte shard header with no decode; only
/// float-encoded shards are decoded, and the scan stops as soon as one value
/// settles the question for the whole matrix. Cost: nothing on an
/// integer-encoded file, one early-exiting decode on a genuinely real one, and
/// one extra value pass on a float-encoded all-integral one.
///
/// The scan is restricted to **kept** rows, because the flag describes the
/// matrix that gets written: a value in a deleted row can no more make the
/// export `real` than it can contribute to `nnz`.
fn scan_header_facts(
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
    keep: Option<&[bool]>,
) -> Result<MatrixHeaderFacts, MtxError> {
    let mut nnz: u64 = 0;
    let mut all_integer = true;

    for entry in shards {
        let stats = require_stats(entry)?;
        let row_start = stats.row_start as usize;
        let encoding = ValueEncoding::from_u8(reader.read_shard_header(entry)?.value_encoding)
            .ok_or_else(|| {
                MtxError::Other(format!(
                    "CSR shard '{}' declares an unknown value encoding",
                    entry.name
                ))
            })?;
        // Once one value has proved the matrix is not all-integral, no later
        // shard can change that back, so the decode is skipped from there on.
        let needs_values = all_integer && !encoding.is_integer();

        match (keep, needs_values) {
            (None, false) => nnz = nnz.saturating_add(stats.nnz),
            (Some(mask), false) => {
                let indptr = reader.read_shard_indptr_from_entry(entry)?;
                nnz = nnz.saturating_add(kept_nnz(&indptr, row_start, mask));
            }
            // The values are needed anyway, so nnz comes out of the same
            // decode rather than reading the shard a second time.
            (_, true) => {
                let (indptr, _indices, values) = reader.read_shard_from_entry_native(entry)?;
                nnz = nnz.saturating_add(match keep {
                    Some(mask) => kept_nnz(&indptr, row_start, mask),
                    None => stats.nnz,
                });
                if let ShardValuesNative::F32(v) = &values {
                    all_integer = kept_values_are_integral(&indptr, row_start, keep, v);
                }
            }
        }
    }

    Ok(MatrixHeaderFacts { nnz, all_integer })
}

fn require_stats(entry: &FullCatalogEntry) -> Result<&ShardStats, MtxError> {
    entry.stats.as_ref().ok_or_else(|| {
        MtxError::Other(format!(
            "CSR shard '{}' carries no catalog stats, so its row range is unknown",
            entry.name
        ))
    })
}

/// Whether the global obs row survives the deletion vector. A row past the end
/// of the mask reads as kept on purpose: the mask is obs-indexed and covers the
/// whole axis, so a shard running past it is a shape error, and
/// `write_matrix_mtx`'s closing row-count check reports it rather than this
/// dropping the rows quietly.
fn row_is_kept(keep: Option<&[bool]>, global_row: usize) -> bool {
    match keep {
        None => true,
        Some(mask) => mask.get(global_row).copied().unwrap_or(true),
    }
}

fn kept_nnz(indptr: &[i64], row_start: usize, mask: &[bool]) -> u64 {
    let n_rows = indptr.len().saturating_sub(1);
    let mut nnz: u64 = 0;
    for r in 0..n_rows {
        if row_is_kept(Some(mask), row_start + r) {
            nnz = nnz.saturating_add((indptr[r + 1] - indptr[r]).max(0) as u64);
        }
    }
    nnz
}

fn kept_values_are_integral(
    indptr: &[i64],
    row_start: usize,
    keep: Option<&[bool]>,
    values: &[f32],
) -> bool {
    let n_rows = indptr.len().saturating_sub(1);
    for r in 0..n_rows {
        if !row_is_kept(keep, row_start + r) {
            continue;
        }
        for &v in &values[indptr[r] as usize..indptr[r + 1] as usize] {
            if !(v.is_finite() && v >= 0.0 && v == v.floor()) {
                return false;
            }
        }
    }
    true
}

/// Write `barcodes.tsv.gz` from obs RecordBatch.
fn write_barcodes_tsv(path: &Path, obs: &arrow::array::RecordBatch) -> Result<(), MtxError> {
    let file = std::fs::File::create(path)?;
    let gz = GzEncoder::new(file, Compression::default());
    let mut w = BufWriter::new(gz);

    // Use the first column as barcode (typically "_index" or "barcode")
    if obs.num_columns() == 0 || obs.num_rows() == 0 {
        return Ok(());
    }

    let col = obs.column(0);
    if let Some(string_arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
        for i in 0..string_arr.len() {
            writeln!(w, "{}", string_arr.value(i))?;
        }
    } else if let Some(dict_arr) = col
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>()
    {
        let values = dict_arr.values().as_string::<i32>();
        for i in 0..dict_arr.len() {
            // CLI5: a null dictionary key must still emit a line, or every
            // subsequent positional barcode shifts up by one row. Use a
            // synthesized placeholder matching the non-dictionary fallback.
            match dict_arr.key(i) {
                Some(key) => writeln!(w, "{}", values.value(key))?,
                None => writeln!(w, "cell_{}", i)?,
            }
        }
    } else {
        // Fallback: use Debug representation
        for i in 0..col.len() {
            writeln!(w, "cell_{}", i)?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Write `features.tsv.gz` from var RecordBatch.
fn write_features_tsv(path: &Path, var: &arrow::array::RecordBatch) -> Result<(), MtxError> {
    let file = std::fs::File::create(path)?;
    let gz = GzEncoder::new(file, Compression::default());
    let mut w = BufWriter::new(gz);

    if var.num_rows() == 0 {
        return Ok(());
    }

    let schema = var.schema();
    let n_rows = var.num_rows();

    // Try to find standard columns by name
    let id_col = find_string_column(var, &["gene_id", "id"]);
    let name_col = find_string_column(var, &["gene_name", "name"]);
    let type_col = find_string_column(var, &["feature_type"]);

    for i in 0..n_rows {
        let id = column_value_at(&id_col, var, i, &schema, 0);
        let name = column_value_at(&name_col, var, i, &schema, 1);

        if let Some(tc) = type_col {
            let ft = string_value(tc, i);
            writeln!(w, "{}\t{}\t{}", id, name, ft)?;
        } else {
            writeln!(w, "{}\t{}", id, name)?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Find a StringArray column by trying candidate names.
fn find_string_column<'a>(
    batch: &'a arrow::array::RecordBatch,
    candidates: &[&str],
) -> Option<&'a arrow::array::StringArray> {
    let schema = batch.schema();
    for name in candidates {
        if let Ok(idx) = schema.index_of(name) {
            if let Some(arr) = batch
                .column(idx)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
            {
                return Some(arr);
            }
        }
    }
    None
}

/// Get a string value from a column, with fallback to positional column or generated value.
fn column_value_at(
    col: &Option<&arrow::array::StringArray>,
    batch: &arrow::array::RecordBatch,
    row: usize,
    schema: &arrow::datatypes::SchemaRef,
    fallback_col_idx: usize,
) -> String {
    if let Some(arr) = col {
        return arr.value(row).to_string();
    }
    // Fallback: try positional column
    if fallback_col_idx < batch.num_columns() {
        if let Some(arr) = batch
            .column(fallback_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
        {
            return arr.value(row).to_string();
        }
    }
    // Last resort
    let _ = schema; // suppress unused warning
    format!("feature_{}", row)
}

/// Get string value from a StringArray.
fn string_value(arr: &arrow::array::StringArray, idx: usize) -> &str {
    arr.value(idx)
}

/// Write synthetic barcodes when obs is not available.
fn write_synthetic_barcodes(path: &Path, n_obs: usize) -> Result<(), MtxError> {
    let file = std::fs::File::create(path)?;
    let gz = GzEncoder::new(file, Compression::default());
    let mut w = BufWriter::new(gz);

    for i in 0..n_obs {
        writeln!(w, "cell_{}", i)?;
    }

    w.flush()?;
    Ok(())
}

/// Write synthetic features when var is not available.
fn write_synthetic_features(path: &Path, n_vars: usize) -> Result<(), MtxError> {
    let file = std::fs::File::create(path)?;
    let gz = GzEncoder::new(file, Compression::default());
    let mut w = BufWriter::new(gz);

    for i in 0..n_vars {
        writeln!(w, "gene_{}\tGene{}\tGene Expression", i, i)?;
    }

    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format_io::deletion_vectors::DeletionVectors;
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;
    use std::io::Read;
    use std::sync::Arc;

    /// A 6×4 file, one nnz per row, with rows 1 and 4 logically deleted.
    fn write_fixture(dir: &tempfile::TempDir, deleted: &[u32]) -> std::path::PathBuf {
        let (n_obs, n_vars) = (6usize, 4usize);
        let path = dir.path().join("in.scx");
        let mut w = ScxWriter::new(
            &path,
            FileHeader {
                n_obs: n_obs as u64,
                n_vars: n_vars as u64,
                ..Default::default()
            },
        )
        .unwrap();

        let obs = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "cell_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                (0..n_obs).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        w.write_obs(&obs).unwrap();
        let var = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("gene_id", DataType::Utf8, false),
                Field::new("gene_name", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(StringArray::from(
                    (0..n_vars).map(|i| format!("ENSG{i}")).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    (0..n_vars).map(|i| format!("Gene{i}")).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        w.write_var(&var).unwrap();

        let mut indptr = vec![0u64];
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for row in 0..n_obs {
            indices.push((row % n_vars) as u32);
            values.push((row + 1) as u8);
            indptr.push(indptr.last().unwrap() + 1);
        }
        w.write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

        if !deleted.is_empty() {
            let mut dv = DeletionVectors::new();
            dv.insert_global(deleted.iter().copied());
            w.write_deletion_vectors(&dv).unwrap();
        }
        w.finish().unwrap();
        path
    }

    fn read_gz(path: &Path) -> String {
        let f = std::fs::File::open(path).unwrap();
        let mut s = String::new();
        flate2::read::GzDecoder::new(f)
            .read_to_string(&mut s)
            .unwrap();
        s
    }

    /// MTX cannot represent a logical deletion, so the export has to apply it.
    ///
    /// The barcode count is asserted against the matrix size line rather than
    /// against a constant: a version of this bug that filtered X but not obs
    /// would still produce "6 barcodes", and would be the worse failure — an
    /// internally inconsistent directory rather than a merely stale one.
    #[test]
    fn mtx_export_omits_deleted_cells() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_fixture(&dir, &[1, 4]);
        let out = dir.path().join("mtx");
        write_scx_to_mtx(&src, &out).unwrap();

        let barcodes: Vec<String> = read_gz(&out.join("barcodes.tsv.gz"))
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        assert_eq!(
            barcodes,
            vec!["cell_0", "cell_2", "cell_3", "cell_5"],
            "the deleted cells must not appear, and the survivors keep their order"
        );

        // Size line is "n_vars n_obs nnz" (features × barcodes, Cell Ranger's
        // orientation — SCX-010).
        let mtx = read_gz(&out.join("matrix.mtx.gz"));
        let size_line = mtx
            .lines()
            .find(|l| !l.starts_with('%'))
            .expect("size line");
        let parts: Vec<usize> = size_line
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        assert_eq!(parts[0], 4, "n_vars unchanged");
        assert_eq!(
            parts[1],
            barcodes.len(),
            "matrix column count must equal the barcode count"
        );
        assert_eq!(parts[2], 4, "one nnz per surviving row");
    }

    /// The no-deletions path must be untouched by the filter.
    #[test]
    fn mtx_export_without_deletions_writes_every_cell() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_fixture(&dir, &[]);
        let out = dir.path().join("mtx");
        write_scx_to_mtx(&src, &out).unwrap();
        assert_eq!(
            read_gz(&out.join("barcodes.tsv.gz")).lines().count(),
            6,
            "no deletion vector: every cell is exported"
        );
    }

    // -----------------------------------------------------------------
    // Streaming export
    // -----------------------------------------------------------------

    /// A single-modality file whose one CSR shard carries `values` under
    /// `encoding`, one entry per row at column `row % n_vars`.
    ///
    /// The encoding is chosen by the caller rather than detected, which is the
    /// whole point: the two shapes that matter here — integral counts stored
    /// as `Float32`, and a `Uint32` count above 2²⁴ — are exactly the ones a
    /// value-detecting writer would not produce on demand.
    fn write_encoded_fixture(
        dir: &tempfile::TempDir,
        file_name: &str,
        encoding: ValueEncoding,
        values: &[f64],
        n_vars: usize,
    ) -> std::path::PathBuf {
        let n_obs = values.len();
        let path = dir.path().join(file_name);
        let mut w = ScxWriter::new(
            &path,
            FileHeader {
                n_obs: n_obs as u64,
                n_vars: n_vars as u64,
                ..Default::default()
            },
        )
        .unwrap();

        let obs = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "cell_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                (0..n_obs).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        w.write_obs(&obs).unwrap();
        let var = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("gene_id", DataType::Utf8, false),
                Field::new("gene_name", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(StringArray::from(
                    (0..n_vars).map(|i| format!("ENSG{i}")).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    (0..n_vars).map(|i| format!("Gene{i}")).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        w.write_var(&var).unwrap();

        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut raw: Vec<u8> = Vec::new();
        for (row, &v) in values.iter().enumerate() {
            indices.push((row % n_vars) as u32);
            match encoding {
                ValueEncoding::Uint32 => raw.extend_from_slice(&(v as u32).to_le_bytes()),
                ValueEncoding::Float32 => raw.extend_from_slice(&(v as f32).to_le_bytes()),
                other => panic!("fixture does not encode {other:?}"),
            }
            indptr.push(indptr.last().unwrap() + 1);
        }
        w.write_csr_shard(&indptr, &indices, &raw, CodecId::None, encoding, 0)
            .unwrap();
        w.finish().unwrap();
        path
    }

    fn body_lines(mtx: &str) -> Vec<String> {
        mtx.lines()
            .filter(|l| !l.starts_with('%'))
            .skip(1) // size line
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().to_string())
            .collect()
    }

    fn banner(mtx: &str) -> String {
        mtx.lines().next().unwrap().to_string()
    }

    /// The shard-at-a-time walk must emit exactly the COO the assembled matrix
    /// implies — same triplets, same order, same declared counts.
    ///
    /// The oracle is built here from `read_all_csr_shards_filtered`, the
    /// materialising read the export used to be written on top of, so this
    /// compares the streaming implementation against an independent
    /// construction rather than against a frozen blob that would have to be
    /// regenerated with the code it is supposed to check.
    #[test]
    fn the_streamed_body_matches_the_assembled_matrix() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_fixture(&dir, &[1, 4]);
        let out = dir.path().join("mtx");
        write_scx_to_mtx(&src, &out).unwrap();
        let mtx = read_gz(&out.join("matrix.mtx.gz"));

        let reader = ScxReader::open(&src).unwrap();
        let csr = reader.read_all_csr_shards_filtered().unwrap();
        let mut expected = Vec::new();
        for row in 0..csr.shape.0 {
            for idx in csr.indptr[row] as usize..csr.indptr[row + 1] as usize {
                expected.push(format!(
                    "{} {} {}",
                    csr.indices[idx] + 1,
                    row + 1,
                    csr.data[idx] as i64
                ));
            }
        }

        assert_eq!(body_lines(&mtx), expected, "streamed body");
        let size_line = mtx.lines().find(|l| !l.starts_with('%')).unwrap();
        assert_eq!(
            size_line,
            format!("{} {} {}", csr.shape.1, csr.shape.0, csr.nnz()),
            "size line"
        );
    }

    /// The declared type must come from the **values**, not from the shard's
    /// `ValueEncoding`.
    ///
    /// An h5ad-sourced count matrix routinely lands as integral values in a
    /// `Float32`-encoded shard, so reading the flag off the encoding would flip
    /// every such export from `integer` to `real`. That is the obvious
    /// shortcut when making the export streaming — the values are exactly what
    /// a streaming writer no longer holds all of at once — and
    /// `benchmarks/comprehensive/thresholds.yaml` floors `mtx_header_integer`
    /// on a file of this shape for the same reason. This is the CI-side twin,
    /// so the regression does not have to wait for a benchmark capture.
    #[test]
    fn a_float_encoded_all_integral_shard_still_declares_integer() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_encoded_fixture(
            &dir,
            "float_integral.scx",
            ValueEncoding::Float32,
            &[1.0, 2.0, 3.0, 419.0],
            4,
        );
        let out = dir.path().join("mtx");
        write_scx_to_mtx(&src, &out).unwrap();
        let mtx = read_gz(&out.join("matrix.mtx.gz"));

        assert_eq!(
            banner(&mtx),
            "%%MatrixMarket matrix coordinate integer general",
            "integral values in a Float32-encoded shard still declare `integer`"
        );
        assert_eq!(
            body_lines(&mtx).last().unwrap(),
            "4 4 419",
            "and the body is formatted as integers, not `419.0`"
        );
    }

    /// The accept-side twin: one fractional value and the same file declares
    /// `real`. Without this the test above would pass against a writer that
    /// hard-coded `integer`.
    #[test]
    fn one_fractional_value_makes_the_export_real() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_encoded_fixture(
            &dir,
            "float_fraction.scx",
            ValueEncoding::Float32,
            &[1.0, 2.0, 3.0, 2.5],
            4,
        );
        let out = dir.path().join("mtx");
        write_scx_to_mtx(&src, &out).unwrap();
        let mtx = read_gz(&out.join("matrix.mtx.gz"));

        assert_eq!(
            banner(&mtx),
            "%%MatrixMarket matrix coordinate real general"
        );
        assert_eq!(body_lines(&mtx).last().unwrap(), "4 4 2.5");
    }

    /// An integer-encoded count above 2²⁴ must export exactly.
    ///
    /// The assembled export decoded every shard to `f32` first and wrote
    /// `val as i64`, so `16_777_217` came out as `16_777_216` — the same
    /// rounding MTX *ingest* did, in the other direction. Streaming through
    /// `read_shard_from_entry_native` keeps integer-encoded values as `u32`.
    #[test]
    fn an_integer_shard_above_2_24_exports_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_encoded_fixture(
            &dir,
            "big_counts.scx",
            ValueEncoding::Uint32,
            &[1.0, 16_777_217.0],
            2,
        );
        let out = dir.path().join("mtx");
        write_scx_to_mtx(&src, &out).unwrap();
        let mtx = read_gz(&out.join("matrix.mtx.gz"));

        assert_eq!(
            body_lines(&mtx),
            vec!["1 1 1".to_string(), "2 2 16777217".to_string()],
            "the count must survive the export, not round to 16777216"
        );
    }

    // -----------------------------------------------------------------
    // Multimodal
    // -----------------------------------------------------------------

    /// A 3-cell file with two modalities: `rna` over 2 genes and `adt` over 3.
    /// Each modality independently tiles the shared obs axis, which is what
    /// makes the flattened shard list overlap.
    fn write_multimodal_fixture(dir: &tempfile::TempDir) -> std::path::PathBuf {
        use scx_format_io::modality::ModalityType;
        use scx_format_io::writer::ShardBuffers;

        let n_obs = 3usize;
        let path = dir.path().join("multi.scx");
        let mut w = ScxWriter::new(
            &path,
            FileHeader {
                n_obs: n_obs as u64,
                n_vars: 3,
                ..Default::default()
            },
        )
        .unwrap();

        let obs = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "cell_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                (0..n_obs).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        w.write_obs(&obs).unwrap();

        for (name, mtype, genes) in [
            ("rna", ModalityType::Rna, vec!["RNA0", "RNA1"]),
            ("adt", ModalityType::Protein, vec!["ADT0", "ADT1", "ADT2"]),
        ] {
            let mid = w
                .add_modality(name, mtype, CodecId::None, ValueEncoding::Uint8, false)
                .unwrap();
            w.set_modality_n_vars(mid, genes.len() as u64).unwrap();

            let var = RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("gene_id", DataType::Utf8, false),
                    Field::new("gene_name", DataType::Utf8, false),
                ])),
                vec![
                    Arc::new(StringArray::from(genes.clone())),
                    Arc::new(StringArray::from(genes.clone())),
                ],
            )
            .unwrap();
            w.write_var_for(mid, &var).unwrap();

            // One entry per cell, so every modality covers every obs row.
            let mut indptr = vec![0u64];
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for row in 0..n_obs {
                indices.push((row % genes.len()) as u32);
                values.push((row + 1) as u8);
                indptr.push(indptr.last().unwrap() + 1);
            }
            w.write_csr_shard_for(
                mid,
                0,
                ShardBuffers::new(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                ),
            )
            .unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// A MatrixMarket directory describes one matrix over one feature space.
    ///
    /// Unscoped, the export used to stack both modalities into a 6-row matrix
    /// over mixed column spaces — and label it with synthetic `gene_i` names,
    /// because a multimodal file has no global `var` section. The assertion is
    /// on the message rather than on `is_err()` so the test cannot pass via
    /// some unrelated failure.
    #[test]
    fn mtx_export_refuses_a_multimodal_file_without_a_modality() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_multimodal_fixture(&dir);
        let out = dir.path().join("mtx");
        let err = write_scx_to_mtx(&src, &out).unwrap_err().to_string();
        assert!(
            err.contains("--modality") && err.contains("rna") && err.contains("adt"),
            "the refusal must name the flag and the available modalities: {err}"
        );
    }

    /// Naming a modality scopes the matrix **and** the feature table to it.
    ///
    /// The features half is the one a shape-only assertion would miss: an
    /// export that scoped X but kept reaching for the global `var` would fall
    /// through to synthetic `gene_0` names and still produce a
    /// correctly-shaped directory.
    #[test]
    fn mtx_export_of_one_modality_uses_that_modalitys_var() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_multimodal_fixture(&dir);

        let out = dir.path().join("rna_mtx");
        write_scx_to_mtx_for(&src, &out, Some("rna")).unwrap();
        let features: Vec<String> = read_gz(&out.join("features.tsv.gz"))
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.split('\t').next().unwrap().to_string())
            .collect();
        assert_eq!(features, vec!["RNA0", "RNA1"], "rna's own var, not gene_i");

        let mtx = read_gz(&out.join("matrix.mtx.gz"));
        let size_line = mtx.lines().find(|l| !l.starts_with('%')).unwrap();
        assert_eq!(size_line, "2 3 3", "2 rna genes x 3 cells, 3 non-zeros");

        // And the sibling modality is a different matrix over a different
        // feature space, not a slice of the same one.
        let out_adt = dir.path().join("adt_mtx");
        write_scx_to_mtx_for(&src, &out_adt, Some("adt")).unwrap();
        let adt_features: Vec<String> = read_gz(&out_adt.join("features.tsv.gz"))
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.split('\t').next().unwrap().to_string())
            .collect();
        assert_eq!(adt_features, vec!["ADT0", "ADT1", "ADT2"]);
    }

    #[test]
    fn mtx_export_rejects_an_unknown_modality() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_multimodal_fixture(&dir);
        let out = dir.path().join("mtx");
        let err = write_scx_to_mtx_for(&src, &out, Some("atac"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown modality 'atac'") && err.contains("rna"),
            "{err}"
        );
    }

    #[test]
    fn mtx_export_rejects_a_modality_on_a_single_modality_file() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_fixture(&dir, &[]);
        let out = dir.path().join("mtx");
        let err = write_scx_to_mtx_for(&src, &out, Some("rna"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("single-modality"), "{err}");
    }
}
