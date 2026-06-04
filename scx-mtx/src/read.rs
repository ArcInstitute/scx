//! Read Cell Ranger–style MTX directories.
//!
//! Expects a directory containing:
//! - `matrix.mtx[.gz]`  — sparse matrix in COO (coordinate) format
//! - `barcodes.tsv[.gz]` — one barcode per line
//! - `features.tsv[.gz]` or `genes.tsv[.gz]` — tab-separated gene metadata

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use flate2::read::GzDecoder;

use crate::error::MtxError;

/// Parsed MTX directory data, ready for SCX conversion.
pub struct MtxData {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
    pub n_obs: usize,
    pub n_vars: usize,
    pub obs: RecordBatch,
    pub var: RecordBatch,
}

/// Read a Cell Ranger–style MTX directory.
///
/// Locates `matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, and
/// `features.tsv[.gz]`/`genes.tsv[.gz]` inside `dir`. All three are required.
pub fn read_mtx_directory(dir: &Path) -> Result<MtxData, MtxError> {
    // Locate matrix file
    let mtx_path = find_file(dir, &["matrix.mtx.gz", "matrix.mtx"])?;

    // Locate barcodes file
    let barcodes_path = find_file(dir, &["barcodes.tsv.gz", "barcodes.tsv"])?;

    // Locate features file (Cell Ranger v3+ uses features.tsv, older uses genes.tsv)
    let features_path = find_file(
        dir,
        &[
            "features.tsv.gz",
            "features.tsv",
            "genes.tsv.gz",
            "genes.tsv",
        ],
    )?;

    // Parse matrix. Derive an `nnz` upper bound from the on-disk file
    // size: the minimum ASCII triplet is "1 1 1\n" = 6 bytes, so for
    // an uncompressed file nnz <= file_size / 6. For gzipped MTX we
    // don't know the decompressed size cheaply; sparse-integer MTX
    // text compresses 5–10×, so a 20× factor is a safe upper bound.
    let mtx_file_size = std::fs::metadata(&mtx_path)?.len() as usize;
    let is_gz = mtx_path.extension().is_some_and(|e| e == "gz");
    let gz_factor: usize = if is_gz { 20 } else { 1 };
    let max_nnz_bound = mtx_file_size.saturating_mul(gz_factor) / 6;
    let mtx_reader = open_maybe_gzipped(&mtx_path)?;
    let (indptr, indices, data, n_rows, n_cols) = parse_mtx_file(mtx_reader, max_nnz_bound)?;

    // Parse barcodes (obs)
    let barcodes_reader = open_maybe_gzipped(&barcodes_path)?;
    let obs = parse_barcodes_tsv(barcodes_reader)?;

    // Parse features (var)
    let features_reader = open_maybe_gzipped(&features_path)?;
    let var = parse_features_tsv(features_reader)?;

    // Validate dimensions
    if obs.num_rows() != n_rows {
        return Err(MtxError::Parse(format!(
            "barcodes count ({}) != matrix rows ({})",
            obs.num_rows(),
            n_rows
        )));
    }
    if var.num_rows() != n_cols {
        return Err(MtxError::Parse(format!(
            "features count ({}) != matrix columns ({})",
            var.num_rows(),
            n_cols
        )));
    }

    Ok(MtxData {
        indptr,
        indices,
        data,
        n_obs: n_rows,
        n_vars: n_cols,
        obs,
        var,
    })
}

/// Find the first existing file from `candidates` inside `dir`.
fn find_file(dir: &Path, candidates: &[&str]) -> Result<std::path::PathBuf, MtxError> {
    for name in candidates {
        let p = dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(MtxError::MissingFile(format!(
        "none of [{}] found in {}",
        candidates.join(", "),
        dir.display()
    )))
}

/// Open a file, transparently decompressing if it ends in `.gz`.
fn open_maybe_gzipped(path: &Path) -> Result<Box<dyn BufRead>, MtxError> {
    let file = std::fs::File::open(path)?;
    if path.extension().is_some_and(|e| e == "gz") {
        Ok(Box::new(BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Parse a Matrix Market file from a buffered reader.
///
/// Returns `(indptr, indices, data, n_rows, n_cols)` in CSR format.
/// The MTX file is COO (coordinate) format with 1-indexed row/col.
///
/// `max_nnz_bound` is an upper bound on the number of triplets we
/// accept, derived by the caller from the on-disk file size. The
/// minimum ASCII triplet is "1 1 1\n" = 6 bytes, so for an
/// uncompressed file the tight bound is `file_size / 6`. The bound only
/// guards the up-front allocation reservation; it does not lift the
/// practical ceiling on `nnz`, which is dominated by the in-memory
/// model: the parser holds the full COO triplet buffer
/// (`Vec<(usize, usize, f32)>`) and then the full CSR arrays
/// simultaneously, so resident memory is roughly 32 B per nnz. A
/// 2B-nnz matrix would need ~64 GB resident regardless of any cap.
#[allow(clippy::type_complexity)]
fn parse_mtx_file(
    reader: Box<dyn BufRead>,
    max_nnz_bound: usize,
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>, usize, usize), MtxError> {
    let mut lines = reader.lines();

    // Parse header line
    let header = lines
        .next()
        .ok_or_else(|| MtxError::Parse("empty MTX file".into()))??;
    let header_lower = header.to_lowercase();
    if !header_lower.starts_with("%%matrixmarket") {
        return Err(MtxError::Parse(format!(
            "invalid MTX header: expected %%MatrixMarket, got: {}",
            header
        )));
    }

    // Validate header tokens: %%MatrixMarket matrix coordinate real/integer general
    let tokens: Vec<&str> = header_lower.split_whitespace().collect();
    if tokens.len() < 4 {
        return Err(MtxError::Parse(format!(
            "MTX header has too few fields: {}",
            header
        )));
    }
    if tokens[1] != "matrix" {
        return Err(MtxError::Parse(format!(
            "MTX object type must be 'matrix', got: {}",
            tokens[1]
        )));
    }
    if tokens[2] != "coordinate" {
        return Err(MtxError::Parse(format!(
            "only coordinate (sparse) format is supported, got: {}",
            tokens[2]
        )));
    }
    let is_integer = tokens[3] == "integer";
    let is_real = tokens[3] == "real";
    if !is_integer && !is_real {
        return Err(MtxError::Parse(format!(
            "MTX data type must be 'real' or 'integer', got: {}",
            tokens[3]
        )));
    }

    // Skip comment lines (start with %)
    let mut size_line = String::new();
    for line_result in lines.by_ref() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        size_line = line;
        break;
    }

    // Parse size line: rows cols nnz
    let size_parts: Vec<&str> = size_line.split_whitespace().collect();
    if size_parts.len() < 3 {
        return Err(MtxError::Parse(format!("invalid size line: {}", size_line)));
    }
    let n_rows: usize = size_parts[0]
        .parse()
        .map_err(|_| MtxError::Parse(format!("invalid row count: {}", size_parts[0])))?;
    let n_cols: usize = size_parts[1]
        .parse()
        .map_err(|_| MtxError::Parse(format!("invalid col count: {}", size_parts[1])))?;
    let nnz: usize = size_parts[2]
        .parse()
        .map_err(|_| MtxError::Parse(format!("invalid nnz count: {}", size_parts[2])))?;

    // Reject a malformed size line whose nnz claim exceeds what the
    // file could physically encode (at 6 bytes minimum per triplet).
    // This bounds the up-front `Vec::with_capacity(nnz)` reservation
    // to the file's actual byte budget rather than a hand-picked
    // constant, so atlas-scale inputs (>2B nnz) aren't blocked by
    // the defense.
    if nnz > max_nnz_bound {
        return Err(MtxError::Parse(format!(
            "nnz {nnz} exceeds upper bound {max_nnz_bound} derived from file size"
        )));
    }

    // Read COO triplets into a single vec for cache-friendly sorting
    let mut entries: Vec<(usize, usize, f32)> = Vec::with_capacity(nnz);

    for line_result in lines {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(MtxError::Parse(format!("invalid COO triplet: {}", trimmed)));
        }
        // MTX is 1-indexed, convert to 0-indexed
        let row: usize = parts[0]
            .parse::<usize>()
            .map_err(|_| MtxError::Parse(format!("invalid row index: {}", parts[0])))?
            .checked_sub(1)
            .ok_or_else(|| MtxError::Parse("row index 0 in 1-indexed format".into()))?;
        let col: usize = parts[1]
            .parse::<usize>()
            .map_err(|_| MtxError::Parse(format!("invalid col index: {}", parts[1])))?
            .checked_sub(1)
            .ok_or_else(|| MtxError::Parse("col index 0 in 1-indexed format".into()))?;
        let val: f32 = parts[2]
            .parse()
            .map_err(|_| MtxError::Parse(format!("invalid value: {}", parts[2])))?;

        entries.push((row, col, val));
    }

    if entries.len() != nnz {
        return Err(MtxError::Parse(format!(
            "expected {} entries, read {}",
            nnz,
            entries.len()
        )));
    }

    // Sort COO entries in-place by (row, col), then build CSR directly
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut indptr = vec![0i64; n_rows + 1];
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    for &(row, col, val) in &entries {
        indptr[row + 1] += 1;
        indices.push(col as i32);
        data.push(val);
    }

    // Cumulative sum for indptr
    for i in 1..=n_rows {
        indptr[i] += indptr[i - 1];
    }

    Ok((indptr, indices, data, n_rows, n_cols))
}

/// Parse `barcodes.tsv[.gz]` — one barcode per line.
fn parse_barcodes_tsv(reader: Box<dyn BufRead>) -> Result<RecordBatch, MtxError> {
    let mut barcodes = Vec::new();
    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            // Take first field (in case of multi-column barcodes files)
            let barcode = trimmed.split('\t').next().unwrap_or(trimmed);
            barcodes.push(barcode.to_string());
        }
    }
    let schema = Schema::new(vec![Field::new("barcode", DataType::Utf8, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(barcodes)) as ArrayRef],
    )?;
    Ok(batch)
}

/// Parse `features.tsv[.gz]` or `genes.tsv[.gz]`.
///
/// Expected tab-separated columns:
/// - Column 0: gene/feature ID (e.g. ENSMUSG00000051951)
/// - Column 1: gene/feature name (e.g. Xkr4)
/// - Column 2 (optional, Cell Ranger v3+): feature type (e.g. Gene Expression)
fn parse_features_tsv(reader: Box<dyn BufRead>) -> Result<RecordBatch, MtxError> {
    let mut ids = Vec::new();
    let mut names = Vec::new();
    let mut feature_types = Vec::new();
    let mut has_feature_type = false;

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split('\t').collect();
        if parts.is_empty() {
            continue;
        }
        ids.push(parts[0].to_string());
        names.push(if parts.len() > 1 {
            parts[1].to_string()
        } else {
            parts[0].to_string()
        });
        if parts.len() > 2 {
            feature_types.push(parts[2].to_string());
            has_feature_type = true;
        } else {
            feature_types.push(String::new());
        }
    }

    let mut fields = vec![
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("gene_name", DataType::Utf8, false),
    ];
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(ids)),
        Arc::new(StringArray::from(names)),
    ];

    if has_feature_type {
        fields.push(Field::new("feature_type", DataType::Utf8, false));
        arrays.push(Arc::new(StringArray::from(feature_types)));
    }

    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_mtx_basic() {
        let mtx_content = "\
%%MatrixMarket matrix coordinate integer general
% comment line
3 4 5
1 2 1
1 4 2
2 1 3
3 2 4
3 3 5
";
        let bound = mtx_content.len();
        let reader: Box<dyn BufRead> = Box::new(BufReader::new(Cursor::new(mtx_content)));
        let (indptr, indices, data, n_rows, n_cols) = parse_mtx_file(reader, bound).unwrap();

        assert_eq!(n_rows, 3);
        assert_eq!(n_cols, 4);
        assert_eq!(indptr, vec![0, 2, 3, 5]);
        assert_eq!(indices, vec![1, 3, 0, 1, 2]);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_parse_mtx_real() {
        let mtx_content = "\
%%MatrixMarket matrix coordinate real general
2 2 2
1 1 1.5
2 2 2.5
";
        let bound = mtx_content.len();
        let reader: Box<dyn BufRead> = Box::new(BufReader::new(Cursor::new(mtx_content)));
        let (indptr, indices, data, n_rows, n_cols) = parse_mtx_file(reader, bound).unwrap();

        assert_eq!(n_rows, 2);
        assert_eq!(n_cols, 2);
        assert_eq!(indptr, vec![0, 1, 2]);
        assert_eq!(indices, vec![0, 1]);
        assert_eq!(data, vec![1.5, 2.5]);
    }

    #[test]
    fn test_parse_barcodes() {
        let content = "AAACCCAA-1\nAAACCCAB-1\nAAACCCAC-1\n";
        let reader: Box<dyn BufRead> = Box::new(BufReader::new(Cursor::new(content)));
        let batch = parse_barcodes_tsv(reader).unwrap();

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 1);
    }

    #[test]
    fn test_parse_features_with_type() {
        let content = "ENSG001\tGeneA\tGene Expression\nENSG002\tGeneB\tGene Expression\n";
        let reader: Box<dyn BufRead> = Box::new(BufReader::new(Cursor::new(content)));
        let batch = parse_features_tsv(reader).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3); // id, name, feature_type
    }

    #[test]
    fn test_parse_features_without_type() {
        let content = "ENSG001\tGeneA\nENSG002\tGeneB\n";
        let reader: Box<dyn BufRead> = Box::new(BufReader::new(Cursor::new(content)));
        let batch = parse_features_tsv(reader).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2); // id, name only
    }

    #[test]
    fn test_invalid_mtx_header() {
        let content = "not a valid header\n";
        let reader: Box<dyn BufRead> = Box::new(BufReader::new(Cursor::new(content)));
        let result = parse_mtx_file(reader, content.len());
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Defensive allocation cap tests (Patch 9)
    // -----------------------------------------------------------------------

    #[test]
    fn test_mtx_rejects_oversized_nnz() {
        // The MTX body below is only ~80 bytes, so any claim of 3B
        // nnz vastly exceeds the file's physical encoding budget
        // (6 bytes minimum per triplet). The file-size-derived bound
        // must reject it.
        let mtx_content = "\
%%MatrixMarket matrix coordinate integer general
2 2 3000000000
";
        let bound = mtx_content.len() / 6;
        let reader: Box<dyn BufRead> = Box::new(BufReader::new(Cursor::new(mtx_content)));
        let result = parse_mtx_file(reader, bound);
        assert!(result.is_err(), "should reject oversized nnz");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("exceeds upper bound"),
            "error should mention exceeds upper bound: {err_msg}"
        );
    }
}
