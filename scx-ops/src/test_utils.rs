// Shared test helpers for scx-ops tests.
//
// Copy of scx-cli/src/test_utils.rs — moved CSC modules
// (build_csc, rebuild_csc, rewrite_helpers) brought their tests
// along, and those tests reach for `sample_header` / `sample_obs` /
// `sample_var` / `write_test_file`. scx-cli keeps its own copy for
// subset/upgrade test modules that haven't moved.

#![cfg(test)]

use arrow::array::{Float32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::modality::ModalityType;
use scx_format_io::writer::ScxWriter;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Create a sample FileHeader for testing.
pub fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 10000, 0, 0)
}

/// Create a sample obs RecordBatch with cell_id and cell_type columns.
pub fn sample_obs(n: usize) -> arrow::array::RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
    ]);
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n)
        .map(|i| match i % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        })
        .collect();
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
        ],
    )
    .unwrap()
}

/// Create a sample var RecordBatch with a gene_id column.
pub fn sample_var(n: usize) -> arrow::array::RecordBatch {
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// Write a test SCX file with CSR shards and obs metadata for filtering.
#[allow(dead_code)]
pub fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
    let path = dir.path().join("test.scx");
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Build CSR data: each row has 2 nnz
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }

    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    writer.finish().unwrap();
    path
}

// ---------------------------------------------------------------------------
// Sort test fixtures (SCX-SORT-SPEC §13 Phase 0 / T0.3)
//
// Seven small self-contained `.scx` fixtures that back the §10 sort tests
// across every later phase. Each writes one file into `dir` and returns its
// path. They are deliberately tiny (unit-test small); later phases scale a
// specific one up where a test needs multiple shards or to force spill.
// ---------------------------------------------------------------------------

/// Write `n_obs` rows of deterministic 2-nnz-per-row CSR as a single shard.
#[allow(dead_code)]
fn write_csr_single_shard(writer: &mut ScxWriter, n_obs: usize, n_vars: usize) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
}

/// Build an obs `RecordBatch` from a `cell_id` column plus a set of named
/// `Utf8` (categorical-by-string) columns of length `n_obs`.
#[allow(dead_code)]
fn obs_with_string_cols(n_obs: usize, cols: &[(&str, Vec<String>)]) -> RecordBatch {
    let mut fields = vec![Field::new("cell_id", DataType::Utf8, false)];
    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let mut arrays: Vec<arrow::array::ArrayRef> = vec![Arc::new(StringArray::from(
        ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    ))];
    for (name, vals) in cols {
        assert_eq!(vals.len(), n_obs, "column {name} length must equal n_obs");
        fields.push(Field::new(*name, DataType::Utf8, true));
        arrays.push(Arc::new(StringArray::from(
            vals.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        )));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

/// (a) Plain single-modality categorical-key fixture (`cell_type`).
#[allow(dead_code)]
pub fn fixture_plain(dir: &tempfile::TempDir) -> PathBuf {
    let (n_obs, n_vars) = (12usize, 8usize);
    let path = dir.path().join("sort_plain.scx");
    let mut writer = ScxWriter::new(&path, sample_header(n_obs as u64, n_vars as u64)).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    write_csr_single_shard(&mut writer, n_obs, n_vars);
    writer.finish().unwrap();
    path
}

/// (b) Skewed fixture: one dominant category (~90% `T cell`) to drive the
/// pass-2 partition sub-split test.
#[allow(dead_code)]
pub fn fixture_skewed(dir: &tempfile::TempDir) -> PathBuf {
    let (n_obs, n_vars) = (20usize, 8usize);
    let path = dir.path().join("sort_skewed.scx");
    let mut writer = ScxWriter::new(&path, sample_header(n_obs as u64, n_vars as u64)).unwrap();
    // 18/20 dominant -> 90%; the remaining two are distinct minorities.
    let types: Vec<String> = (0..n_obs)
        .map(|i| match i {
            0 => "B cell".to_string(),
            1 => "NK cell".to_string(),
            _ => "T cell".to_string(),
        })
        .collect();
    writer
        .write_obs(&obs_with_string_cols(n_obs, &[("cell_type", types)]))
        .unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    write_csr_single_shard(&mut writer, n_obs, n_vars);
    writer.finish().unwrap();
    path
}

/// (c) Composite-key fixture: two categoricals (`cell_type` + `donor`).
#[allow(dead_code)]
pub fn fixture_composite(dir: &tempfile::TempDir) -> PathBuf {
    let (n_obs, n_vars) = (12usize, 8usize);
    let path = dir.path().join("sort_composite.scx");
    let mut writer = ScxWriter::new(&path, sample_header(n_obs as u64, n_vars as u64)).unwrap();
    let cell_type: Vec<String> = (0..n_obs)
        .map(|i| if i % 2 == 0 { "T cell" } else { "B cell" }.to_string())
        .collect();
    let donor: Vec<String> = (0..n_obs).map(|i| format!("donor_{}", i % 3)).collect();
    writer
        .write_obs(&obs_with_string_cols(
            n_obs,
            &[("cell_type", cell_type), ("donor", donor)],
        ))
        .unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    write_csr_single_shard(&mut writer, n_obs, n_vars);
    writer.finish().unwrap();
    path
}

/// (d) Numeric-key fixture: an `Int64` `n_genes` obs column.
#[allow(dead_code)]
pub fn fixture_numeric(dir: &tempfile::TempDir) -> PathBuf {
    let (n_obs, n_vars) = (12usize, 8usize);
    let path = dir.path().join("sort_numeric.scx");
    let mut writer = ScxWriter::new(&path, sample_header(n_obs as u64, n_vars as u64)).unwrap();
    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    // Deliberately unsorted numeric key so a sort has work to do.
    let n_genes: Vec<i64> = (0..n_obs).map(|i| ((i * 7 + 3) % 17) as i64).collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("n_genes", DataType::Int64, true),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(n_genes)),
        ],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    write_csr_single_shard(&mut writer, n_obs, n_vars);
    writer.finish().unwrap();
    path
}

/// (e) Multimodal fixture: shared obs axis with `rna` + `adt` modalities.
#[allow(dead_code)]
pub fn fixture_multimodal(dir: &tempfile::TempDir) -> PathBuf {
    let n_obs = 12usize;
    let (rna_vars, adt_vars) = (8usize, 4usize);
    let path = dir.path().join("sort_multimodal.scx");
    let mut writer = ScxWriter::new(&path, sample_header(n_obs as u64, rna_vars as u64)).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();

    let rna_id = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    let adt_id = writer
        .add_modality(
            "adt",
            ModalityType::Protein,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();

    writer.write_var_for(rna_id, &sample_var(rna_vars)).unwrap();
    writer.set_modality_n_vars(rna_id, rna_vars as u64).unwrap();
    writer.write_var_for(adt_id, &sample_var(adt_vars)).unwrap();
    writer.set_modality_n_vars(adt_id, adt_vars as u64).unwrap();

    for (mod_id, n_vars) in [(rna_id, rna_vars), (adt_id, adt_vars)] {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_obs {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            indices.push(col0 as u32);
            indices.push(col1 as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        writer
            .write_csr_shard_for(
                mod_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    path
}

/// (f) obsp + layers fixture: primary X, a `raw` layer, and a cell×cell
/// `connectivities` obsp (v2 Int64 COO).
#[allow(dead_code)]
pub fn fixture_obsp_layers(dir: &tempfile::TempDir) -> PathBuf {
    let (n_obs, n_vars) = (8usize, 6usize);
    let path = dir.path().join("sort_obsp_layers.scx");
    let mut writer = ScxWriter::new(&path, sample_header(n_obs as u64, n_vars as u64)).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    write_csr_single_shard(&mut writer, n_obs, n_vars);

    // A second layer ("raw") mirroring the X shape.
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 3) % 256) as u8);
        values.push(((row + 5) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    writer
        .write_layer_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
            "raw",
            0,
        )
        .unwrap();

    // A small cell×cell obsp as a v2 Int64 COO: each cell linked to its
    // successor (mod n_obs).
    let rows: Vec<i64> = (0..n_obs as i64).collect();
    let cols: Vec<i64> = (0..n_obs as i64).map(|r| (r + 1) % n_obs as i64).collect();
    let data: Vec<f32> = (0..n_obs).map(|i| (i + 1) as f32).collect();
    let obsp_schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int64, false),
            Field::new("col", DataType::Int64, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n_obs.to_string()),
            ("n_cols".to_string(), n_obs.to_string()),
        ]),
    ));
    let obsp_batch = RecordBatch::try_new(
        obsp_schema,
        vec![
            Arc::new(Int64Array::from(rows)),
            Arc::new(Int64Array::from(cols)),
            Arc::new(Float32Array::from(data)),
        ],
    )
    .unwrap();
    writer
        .write_obsp_shard_coo(
            "connectivities",
            0,
            0,
            n_obs as u64,
            n_obs as u64,
            &obsp_batch,
        )
        .unwrap();

    writer.finish().unwrap();
    path
}

/// (g) Deletion-vector fixture: a plain file with three rows logically
/// deleted (live count < total count).
#[allow(dead_code)]
pub fn fixture_deletion(dir: &tempfile::TempDir) -> (PathBuf, usize) {
    let path = fixture_plain(dir);
    // Rename so it doesn't collide with a standalone plain fixture.
    let dv_path = dir.path().join("sort_deletion.scx");
    std::fs::rename(&path, &dv_path).unwrap();
    let deleted = vec![1u64, 3, 5];
    crate::delete::mark_deleted(&dv_path, &deleted).unwrap();
    (dv_path, deleted.len())
}

#[cfg(test)]
mod fixture_smoke {
    //! Phase 0 gate (SCX-SORT-SPEC §13): every sort fixture builds and
    //! reopens with its defining structure intact.
    use super::*;
    use arrow::array::Array;
    use scx_format_io::ScxReader;

    fn has_col(batch: &RecordBatch, name: &str) -> bool {
        batch.schema().column_with_name(name).is_some()
    }

    #[test]
    fn plain_fixture_loads() {
        let dir = tempfile::tempdir().unwrap();
        let r = ScxReader::open(fixture_plain(&dir)).unwrap();
        assert_eq!(r.n_obs(), 12);
        assert!(has_col(&r.read_obs().unwrap(), "cell_type"));
        assert_eq!(r.read_all_csr_shards().unwrap().shape.0, 12);
    }

    #[test]
    fn skewed_fixture_loads() {
        let dir = tempfile::tempdir().unwrap();
        let r = ScxReader::open(fixture_skewed(&dir)).unwrap();
        assert_eq!(r.n_obs(), 20);
        let obs = r.read_obs().unwrap();
        assert!(has_col(&obs, "cell_type"));
        // Dominant category really dominates.
        let ct = obs
            .column_by_name("cell_type")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let dominant = (0..ct.len()).filter(|&i| ct.value(i) == "T cell").count();
        assert_eq!(dominant, 18);
    }

    #[test]
    fn composite_fixture_loads() {
        let dir = tempfile::tempdir().unwrap();
        let r = ScxReader::open(fixture_composite(&dir)).unwrap();
        let obs = r.read_obs().unwrap();
        assert!(has_col(&obs, "cell_type") && has_col(&obs, "donor"));
    }

    #[test]
    fn numeric_fixture_loads() {
        let dir = tempfile::tempdir().unwrap();
        let r = ScxReader::open(fixture_numeric(&dir)).unwrap();
        let obs = r.read_obs().unwrap();
        assert_eq!(
            obs.schema()
                .column_with_name("n_genes")
                .unwrap()
                .1
                .data_type(),
            &DataType::Int64
        );
    }

    #[test]
    fn multimodal_fixture_loads() {
        let dir = tempfile::tempdir().unwrap();
        let r = ScxReader::open(fixture_multimodal(&dir)).unwrap();
        assert!(r.is_multimodal());
        assert_eq!(r.n_modalities(), 2);
        let names = r.modality_names();
        assert!(names.contains(&"rna") && names.contains(&"adt"));
    }

    #[test]
    fn obsp_layers_fixture_loads() {
        let dir = tempfile::tempdir().unwrap();
        let r = ScxReader::open(fixture_obsp_layers(&dir)).unwrap();
        assert!(r.layer_names().iter().any(|n| n == "raw"));
        let obsp = r.read_obsp("connectivities").unwrap();
        assert_eq!(obsp.num_rows(), 8);
    }

    #[test]
    fn deletion_fixture_loads() {
        let dir = tempfile::tempdir().unwrap();
        let (path, n_deleted) = fixture_deletion(&dir);
        let r = ScxReader::open(&path).unwrap();
        let mask = r
            .deletion_keep_mask()
            .unwrap()
            .expect("deletion fixture must carry a deletion vector");
        let dropped = mask.iter().filter(|&&keep| !keep).count();
        assert_eq!(dropped, n_deleted);
    }
}
