//! scx-convert integration tests — dataframe (T5.6 split).

use super::convert_tests_common::*;

#[test]
fn write_dataframe_group_honors_pandas_index_metadata_unnamed() {
    // pyarrow.Table.from_pandas on a var DataFrame with var.index.name = None
    // emits the index as `__index_level_0__`. Writer must rename it to
    // `_index` on disk and exclude it from `column-order`.
    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let root = file.as_group().unwrap();

    let batch = build_var_batch_with_pandas_metadata(
        &["__index_level_0__"],
        "__index_level_0__",
        &["MIR1302-2HG", "FAM138A"],
    );
    crate::h5ad::write::write_dataframe_group_at(&root, "var", &batch, &mut WarningSink::log())
        .unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let var = file.group("var").unwrap();

    let idx_name: VarLenUnicode = var.attr("_index").unwrap().read_scalar().unwrap();
    assert_eq!(idx_name.as_str(), "_index");

    let col_order: Vec<VarLenUnicode> = var
        .attr("column-order")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let names: Vec<String> = col_order.iter().map(|s| s.to_string()).collect();
    assert_eq!(names, vec!["gene_ids".to_string()]);

    let symbols: Vec<VarLenUnicode> = var.dataset("_index").unwrap().read_1d().unwrap().to_vec();
    let symbols: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
    assert_eq!(symbols, vec!["MIR1302-2HG", "FAM138A"]);

    assert!(
        var.dataset("__index_level_0__").is_err(),
        "phantom __index_level_0__ dataset must not exist on disk"
    );
}

#[test]
fn write_dataframe_group_honors_pandas_index_metadata_named() {
    // Named pandas index (`var.index.name = "gene_symbols"`): the column
    // is named gene_symbols in the schema; pyarrow lists it under
    // `index_columns`. Writer must use "gene_symbols" as both the
    // on-disk dataset name AND the `_index` attribute, and exclude it
    // from `column-order`.
    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let root = file.as_group().unwrap();

    let batch = build_var_batch_with_pandas_metadata(
        &["gene_symbols"],
        "gene_symbols",
        &["MIR1302-2HG", "FAM138A"],
    );
    crate::h5ad::write::write_dataframe_group_at(&root, "var", &batch, &mut WarningSink::log())
        .unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let var = file.group("var").unwrap();

    let idx_name: VarLenUnicode = var.attr("_index").unwrap().read_scalar().unwrap();
    assert_eq!(idx_name.as_str(), "gene_symbols");

    let col_order: Vec<VarLenUnicode> = var
        .attr("column-order")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let names: Vec<String> = col_order.iter().map(|s| s.to_string()).collect();
    assert_eq!(names, vec!["gene_ids".to_string()]);

    let symbols: Vec<VarLenUnicode> = var
        .dataset("gene_symbols")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let symbols: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
    assert_eq!(symbols, vec!["MIR1302-2HG", "FAM138A"]);

    assert!(
        var.dataset("_index").is_err(),
        "for a named pandas index, the on-disk dataset must be the named one, \
         not a renamed `_index`"
    );
}

#[test]
fn write_dataframe_group_no_pandas_metadata_fallback() {
    // CLI path: obs/var came from `read_dataframe_group` which doesn't
    // carry pandas metadata. The first schema field is already the
    // index dataset name (e.g. "_index" or "gene_symbols" from the
    // source h5ad). Writer must use field(0) as the index AND exclude
    // it from `column-order` — matches anndata's convention.
    use arrow::array::StringArray;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let root = file.as_group().unwrap();

    // No pandas metadata; first field is "_index" (anndata default).
    let schema = Arc::new(Schema::new(vec![
        Field::new("_index", DataType::Utf8, false),
        Field::new("gene_ids", DataType::Utf8, false),
    ]));
    let symbols = Arc::new(StringArray::from(vec!["GENE_A", "GENE_B"]));
    let gene_ids = Arc::new(StringArray::from(vec!["ENSG1", "ENSG2"]));
    let batch = arrow::record_batch::RecordBatch::try_new(schema, vec![symbols, gene_ids]).unwrap();
    crate::h5ad::write::write_dataframe_group_at(&root, "var", &batch, &mut WarningSink::log())
        .unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let var = file.group("var").unwrap();

    let idx_name: VarLenUnicode = var.attr("_index").unwrap().read_scalar().unwrap();
    assert_eq!(idx_name.as_str(), "_index");

    let col_order: Vec<VarLenUnicode> = var
        .attr("column-order")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let names: Vec<String> = col_order.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        names,
        vec!["gene_ids".to_string()],
        "the index column must be excluded from column-order"
    );
}

#[test]
fn read_dataframe_group_index_only_recovers_values() {
    // When a dataframe group has an empty `column-order` attribute (the
    // canonical anndata emission for an index-only frame, and what
    // `write_dataframe_body` now emits when all schema fields are the
    // pandas index), `read_dataframe_group` must still read the real
    // values from the `_index` dataset — not synthesise blank
    // strings, which is what the pre-fix fallback did.
    use crate::h5ad::read::read_dataframe_group;

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("idx_only.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let var = file.create_group("var").unwrap();

    // Encoding metadata that anndata expects.
    var.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("dataframe"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("encoding-version")
        .unwrap()
        .write_scalar(&vlu("0.2.0"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();
    // Length-0 column-order — matches our writer's emission for
    // index-only frames.
    let empty: Vec<VarLenUnicode> = Vec::new();
    var.new_attr::<VarLenUnicode>()
        .shape(0_usize)
        .create("column-order")
        .unwrap()
        .write_raw(&empty)
        .unwrap();

    // Actual index dataset — gene symbols on disk.
    let symbols: Vec<VarLenUnicode> = ["MIR1302-2HG", "FAM138A", "OR4F5"]
        .iter()
        .map(|s| vlu(s))
        .collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([symbols.len()])
        .create("_index")
        .unwrap()
        .write(&symbols)
        .unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let batch = read_dataframe_group(&file, "var", &mut WarningSink::log()).unwrap();

    // anndata's `_index = "_index"` sentinel
    // (unnamed pandas index) is renamed to pyarrow's canonical
    // `__index_level_0__` in the Arrow schema, and the schema gains a
    // `pandas` metadata envelope so consumers like
    // `pyscx.open(...).to_anndata()` and
    // `scx-convert/src/h5ad/write.rs::write_dataframe_body` identify
    // the index automatically.
    assert_eq!(batch.num_columns(), 1, "expected single index column");
    assert_eq!(batch.schema().field(0).name(), "__index_level_0__");
    assert_eq!(
        scx_format_io::pandas_index_columns(batch.schema_ref()),
        vec!["__index_level_0__".to_string()],
        "schema must carry pandas metadata pointing at the index column"
    );
    assert_eq!(batch.num_rows(), 3);

    let col = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("index column must be Utf8/StringArray");
    let values: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
    assert_eq!(
        values,
        vec!["MIR1302-2HG", "FAM138A", "OR4F5"],
        "fallback must read real values from the _index dataset, not blanks"
    );
}

#[test]
fn read_dataframe_group_attaches_pandas_index_metadata_unnamed() {
    // var with `_index = "_index"`
    // (unnamed pandas index) PLUS non-empty `column-order`. Pre-fix,
    // `read_dataframe_group` silently dropped the index. Post-fix, the
    // schema must include `__index_level_0__` AND stamp the pandas
    // metadata envelope so consumers find the index.
    use crate::h5ad::read::read_dataframe_group;

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var_unnamed.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let var = file.create_group("var").unwrap();

    var.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("dataframe"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("encoding-version")
        .unwrap()
        .write_scalar(&vlu("0.2.0"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();
    // column-order = ["gene_ids", "feature_types"] — the index is
    // excluded by anndata convention.
    let col_order: Vec<VarLenUnicode> = ["gene_ids", "feature_types"]
        .iter()
        .map(|s| vlu(s))
        .collect();
    var.new_attr::<VarLenUnicode>()
        .shape([col_order.len()])
        .create("column-order")
        .unwrap()
        .write(&col_order)
        .unwrap();

    let symbols: Vec<VarLenUnicode> = ["MIR1302-2HG", "FAM138A", "OR4F5"]
        .iter()
        .map(|s| vlu(s))
        .collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([symbols.len()])
        .create("_index")
        .unwrap()
        .write(&symbols)
        .unwrap();
    let gene_ids: Vec<VarLenUnicode> = ["ENSG1", "ENSG2", "ENSG3"].iter().map(|s| vlu(s)).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([gene_ids.len()])
        .create("gene_ids")
        .unwrap()
        .write(&gene_ids)
        .unwrap();
    let feature_types: Vec<VarLenUnicode> = ["Gene Expression"; 3].iter().map(|s| vlu(s)).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([feature_types.len()])
        .create("feature_types")
        .unwrap()
        .write(&feature_types)
        .unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let batch = read_dataframe_group(&file, "var", &mut WarningSink::log()).unwrap();

    let schema = batch.schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(field_names.contains(&"gene_ids"), "fields={field_names:?}");
    assert!(
        field_names.contains(&"feature_types"),
        "fields={field_names:?}"
    );
    assert!(
        field_names.contains(&"__index_level_0__"),
        "fields={field_names:?} — B1 reader must inject the index column"
    );
    assert_eq!(
        scx_format_io::pandas_index_columns(batch.schema_ref()),
        vec!["__index_level_0__".to_string()],
        "schema must carry pandas metadata pointing at the index column"
    );

    // Values in the index column round-trip.
    let idx_pos = field_names
        .iter()
        .position(|n| *n == "__index_level_0__")
        .unwrap();
    let col = batch
        .column(idx_pos)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("index column must be Utf8/StringArray");
    let values: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
    assert_eq!(values, vec!["MIR1302-2HG", "FAM138A", "OR4F5"]);
}

#[test]
fn read_dataframe_group_attaches_pandas_index_metadata_named() {
    // B1-2026-05-20 named-index shape: `_index = "gene_symbols"`. The
    // reader must NOT rename to `__index_level_0__`; the field keeps
    // its original name and the pandas metadata points at it.
    use crate::h5ad::read::read_dataframe_group;

    let dir = tempfile::tempdir().unwrap();
    let h5_path = dir.path().join("var_named.h5");
    let file = hdf5::File::create(&h5_path).unwrap();
    let var = file.create_group("var").unwrap();

    var.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("dataframe"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("encoding-version")
        .unwrap()
        .write_scalar(&vlu("0.2.0"))
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("gene_symbols"))
        .unwrap();
    let col_order: Vec<VarLenUnicode> = ["gene_ids"].iter().map(|s| vlu(s)).collect();
    var.new_attr::<VarLenUnicode>()
        .shape([col_order.len()])
        .create("column-order")
        .unwrap()
        .write(&col_order)
        .unwrap();

    let symbols: Vec<VarLenUnicode> = ["MIR1302-2HG", "FAM138A", "OR4F5"]
        .iter()
        .map(|s| vlu(s))
        .collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([symbols.len()])
        .create("gene_symbols")
        .unwrap()
        .write(&symbols)
        .unwrap();
    let gene_ids: Vec<VarLenUnicode> = ["ENSG1", "ENSG2", "ENSG3"].iter().map(|s| vlu(s)).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([gene_ids.len()])
        .create("gene_ids")
        .unwrap()
        .write(&gene_ids)
        .unwrap();
    drop(file);

    let file = hdf5::File::open(&h5_path).unwrap();
    let batch = read_dataframe_group(&file, "var", &mut WarningSink::log()).unwrap();

    let schema = batch.schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(field_names.contains(&"gene_ids"), "fields={field_names:?}");
    assert!(
        field_names.contains(&"gene_symbols"),
        "named index keeps its source name: {field_names:?}"
    );
    assert!(
        !field_names.contains(&"__index_level_0__"),
        "named index must NOT be renamed: {field_names:?}"
    );
    assert_eq!(
        scx_format_io::pandas_index_columns(batch.schema_ref()),
        vec!["gene_symbols".to_string()],
        "pandas metadata must point at the named index"
    );
}

#[test]
fn h5ad_to_scx_streaming_preserves_obs_var_names() {
    // B1-2026-05-20 end-to-end: the streaming path that `scx convert`
    // uses by default must produce an SCX whose obs/var carry the
    // pandas index metadata, so `pyscx.open(...).to_anndata()` sees
    // `cell_<i>` / `gene_<i>` as obs_names / var_names and NOT as
    // integer-positional defaults.
    use super::pipeline::{h5ad_to_scx_streaming, StreamingOverrides};

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let n_obs = 6;
    let n_vars = 4;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);

    let scx_path = dir.path().join("out.scx");
    h5ad_to_scx_streaming(
        &h5ad_path,
        &scx_path,
        &ConvertOptions::default(),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("streaming convert must succeed");

    let reader = ScxReader::open(&scx_path).unwrap();
    let obs = reader.read_obs().unwrap();
    assert_eq!(
        scx_format_io::pandas_index_columns(obs.schema_ref()),
        vec!["__index_level_0__".to_string()],
        "obs schema must identify the index column"
    );
    let obs_idx = obs.schema().index_of("__index_level_0__").unwrap();
    let obs_col = obs
        .column(obs_idx)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("obs index must be Utf8/StringArray");
    let obs_values: Vec<String> = (0..obs_col.len())
        .map(|i| obs_col.value(i).to_string())
        .collect();
    let expected_obs: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    assert_eq!(obs_values, expected_obs, "obs_names must round-trip");

    let var = reader.read_var().unwrap();
    assert_eq!(
        scx_format_io::pandas_index_columns(var.schema_ref()),
        vec!["__index_level_0__".to_string()],
        "var schema must identify the index column"
    );
    let var_idx = var.schema().index_of("__index_level_0__").unwrap();
    let var_col = var
        .column(var_idx)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("var index must be Utf8/StringArray");
    let var_values: Vec<String> = (0..var_col.len())
        .map(|i| var_col.value(i).to_string())
        .collect();
    let expected_var: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    assert_eq!(var_values, expected_var, "var_names must round-trip");
}

#[test]
fn process_outcomes_emits_aggregate_when_preset_fully_missing() {
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};
    use std::sync::{Arc, Mutex};

    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);
    let mut sink = WarningSink::with_handler(move |w| {
        captured_clone
            .lock()
            .unwrap()
            .push(w.category().to_string());
    });

    let outcomes = vec![
        BuildOutcome::PresetSkipped {
            column: "cell_type".into(),
            reason: SkipReason::MissingColumn,
        },
        BuildOutcome::PresetSkipped {
            column: "tissue".into(),
            reason: SkipReason::MissingColumn,
        },
        BuildOutcome::PresetSkipped {
            column: "disease".into(),
            reason: SkipReason::MissingColumn,
        },
    ];
    process_predicate_index_outcomes(outcomes, "obs", Some("cellxgene"), 3, &[], &mut sink)
        .unwrap();

    let cats = captured.lock().unwrap();
    assert_eq!(*cats, vec!["preset_no_columns_matched".to_string()]);
    assert_eq!(
        sink.counts().get("missing_preset_index_column"),
        None,
        "no per-column warnings expected when preset fully missing"
    );
    assert_eq!(sink.counts().get("preset_no_columns_matched"), Some(&1));
}

#[test]
fn process_outcomes_falls_back_to_per_column_on_partial_mismatch() {
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    let mut sink = WarningSink::log();
    let outcomes = vec![
        BuildOutcome::PresetSkipped {
            column: "cell_type".into(),
            reason: SkipReason::MissingColumn,
        },
        BuildOutcome::PresetSkipped {
            column: "tissue".into(),
            reason: SkipReason::MissingColumn,
        },
        // 1 of 3 expected columns is *not* missing → preset is real,
        // and the per-column warnings remain useful signal.
    ];
    process_predicate_index_outcomes(outcomes, "obs", Some("cellxgene"), 3, &[], &mut sink)
        .unwrap();

    assert_eq!(
        sink.counts().get("missing_preset_index_column"),
        Some(&2),
        "partial mismatch must surface per-column warnings"
    );
    assert_eq!(sink.counts().get("preset_no_columns_matched"), None);
}

#[test]
fn process_outcomes_keeps_per_column_when_no_preset() {
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    // Caller passed user-explicit `--index-obs <col>` (no preset). The
    // engine still surfaces these as `PresetSkipped` because they were
    // resolved via the preset code path — but `preset = None` means
    // there's no aggregate to collapse to.
    let mut sink = WarningSink::log();
    let outcomes = vec![BuildOutcome::PresetSkipped {
        column: "ghost_column".into(),
        reason: SkipReason::MissingColumn,
    }];
    process_predicate_index_outcomes(outcomes, "obs", None, 0, &[], &mut sink).unwrap();
    assert_eq!(sink.counts().get("missing_preset_index_column"), Some(&1));
    assert_eq!(sink.counts().get("preset_no_columns_matched"), None);
}

#[test]
fn process_outcomes_aggregates_forced_missing_columns_into_single_error() {
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    let mut sink = WarningSink::log();
    let outcomes = vec![
        BuildOutcome::ForcedColumnError {
            column: "raw_summ".into(),
            reason: SkipReason::MissingColumn,
        },
        BuildOutcome::ForcedColumnError {
            column: "cell_typ".into(),
            reason: SkipReason::MissingColumn,
        },
    ];
    let available: Vec<String> = vec![
        "soma_joinid".into(),
        "dataset_id".into(),
        "cell_type".into(),
        "raw_sum".into(),
    ];
    let err = process_predicate_index_outcomes(outcomes, "obs", None, 0, &available, &mut sink)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("2 forced obs index columns are missing"),
        "should aggregate: {msg}"
    );
    assert!(
        msg.contains("'raw_summ': did you mean 'raw_sum'?"),
        "first typo + suggestion: {msg}"
    );
    assert!(
        msg.contains("'cell_typ': did you mean 'cell_type'?"),
        "second typo + suggestion: {msg}"
    );
}

#[test]
fn process_outcomes_single_forced_miss_uses_singular_wording() {
    // Sanity: the single-miss path must keep the existing singular
    // wording so PR #113's user-visible message is byte-identical when
    // only one column is typo'd (the common case).
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    let mut sink = WarningSink::log();
    let outcomes = vec![BuildOutcome::ForcedColumnError {
        column: "raw_summ".into(),
        reason: SkipReason::MissingColumn,
    }];
    let available: Vec<String> = vec!["soma_joinid".into(), "raw_sum".into()];
    let err = process_predicate_index_outcomes(outcomes, "obs", None, 0, &available, &mut sink)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("forced obs index column 'raw_summ': missing column."),
        "should use singular wording: {msg}"
    );
    assert!(msg.contains("Did you mean 'raw_sum'?"), "{msg}");
    assert!(
        !msg.contains("forced obs index columns are missing"),
        "should NOT emit plural header for single miss: {msg}"
    );
}

#[test]
fn process_outcomes_non_missing_forced_error_stays_fail_fast() {
    // Forced errors with non-MissingColumn reasons (unsupported dtype,
    // high cardinality) describe a real per-column condition — they
    // should still abort on the first hit rather than aggregating.
    use super::pipeline::process_predicate_index_outcomes;
    use super::warnings::WarningSink;
    use scx_engine::index::{BuildOutcome, SkipReason};

    let mut sink = WarningSink::log();
    let outcomes = vec![BuildOutcome::ForcedColumnError {
        column: "donor_id".into(),
        reason: SkipReason::HighCardinality {
            n_unique: 10_000,
            threshold: 1_024,
        },
    }];
    let err =
        process_predicate_index_outcomes(outcomes, "obs", None, 0, &[], &mut sink).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("forced obs index column 'donor_id'"),
        "should name the column: {msg}"
    );
    // The fail-fast (non-aggregated) path should NOT emit the plural
    // header or the strsim treatment — the column exists.
    assert!(
        !msg.contains("forced obs index columns are missing"),
        "non-missing reason should not aggregate: {msg}"
    );
    assert!(
        !msg.contains("Did you mean"),
        "non-missing reason should not strsim: {msg}"
    );
}

mod streaming_obs_hdf5 {
    //! Round-trip and edge-case tests for the
    //! [`write_dataframe_group_streaming`] obs/var writer wired into
    //! `scx_to_h5ad_streaming` / `scx_to_h5mu_streaming`.
    //!
    //! Fixtures construct sharded-obs SCX files directly via
    //! `ScxWriter::write_obs_shard` so test inputs span numeric, string,
    //! disjoint-dictionary, and nullable-boolean columns. Outputs are
    //! validated by re-reading the h5ad on disk via the SCX writer's own
    //! `read_dataframe_group` helper (the same code AnnData uses) and
    //! asserting column equality with the assembled-eager baseline.
    //!
    //! Pre-existing tests in `scx-format/tests/large_obs.rs` cover the
    //! format-level sharded round-trip; this module covers the export
    //! direction (sharded obs → h5ad/h5mu via hyperslab writes).

    use std::sync::Arc;

    use arrow::array::{
        Array, ArrayRef, AsArray, BooleanArray, DictionaryArray, Float32Array, Float64Array,
        Int32Array, Int64Array, Int8Array, LargeStringArray, RecordBatch, StringArray,
    };
    use arrow::datatypes::{DataType, Field, Float64Type, Int32Type, Int64Type, Int8Type, Schema};

    use scx_codec::{CodecId, ValueEncoding};
    use scx_format_io::{FileHeader, ScxReader, ScxWriter};

    use crate::h5ad::read::read_dataframe_group;
    use crate::h5ad::stream_write::write_scx_to_h5ad_streaming;
    use crate::h5ad::write::{
        write_dataframe_group_at, write_dataframe_group_streaming, write_scx_to_h5ad,
    };
    use crate::pipeline::ConvertOptions;
    use crate::warnings::WarningSink;

    fn header(n_obs: u64, n_vars: u64) -> FileHeader {
        FileHeader::new_single_modality(
            n_obs,
            n_vars,
            0,
            16384,
            0,
            if n_vars <= 65535 { 0 } else { 1 },
        )
    }

    /// Build an obs `RecordBatch` for a shard. Columns:
    ///   - `_index`: Utf8 "cell_{i}".
    ///   - `n_genes`: Int32 (i % 100).
    ///   - `is_doublet`: Boolean, alternating; nulls at every 7th row.
    ///   - `cell_type`: Dictionary<Int8, Utf8> with a per-shard subset
    ///     of category strings so streaming must unify across shards.
    fn obs_shard_batch(start_row: usize, n: usize, shard_idx: u32) -> RecordBatch {
        let cell_ids: Vec<String> = (start_row..start_row + n)
            .map(|i| format!("cell_{i:06}"))
            .collect();
        let n_genes: Vec<i32> = (0..n).map(|i| ((start_row + i) % 100) as i32).collect();
        let is_doublet_values: Vec<Option<bool>> = (0..n)
            .map(|i| {
                if (start_row + i).is_multiple_of(7) {
                    None
                } else {
                    Some(!(start_row + i).is_multiple_of(2))
                }
            })
            .collect();
        // Per-shard categorical vocabulary (disjoint across shards):
        //   shard 0 → {"T cell", "B cell"}
        //   shard 1 → {"NK cell", "Monocyte"}
        //   shard 2 → {"T cell", "Macrophage"}
        //   shard 3 → {"B cell"}
        let vocab: Vec<&'static str> = match shard_idx {
            0 => vec!["T cell", "B cell"],
            1 => vec!["NK cell", "Monocyte"],
            2 => vec!["T cell", "Macrophage"],
            _ => vec!["B cell"],
        };
        let cat_array: Vec<&str> = (0..n).map(|i| vocab[i % vocab.len()]).collect();

        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("n_genes", DataType::Int32, false),
            Field::new("is_doublet", DataType::Boolean, true),
            Field::new(
                "cell_type",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                false,
            ),
        ]);
        let cell_ids_arr: ArrayRef = Arc::new(StringArray::from(cell_ids));
        let n_genes_arr: ArrayRef = Arc::new(Int32Array::from(n_genes));
        let is_doublet_arr: ArrayRef = Arc::new(BooleanArray::from(is_doublet_values));
        let cat_arr: ArrayRef = Arc::new(DictionaryArray::<Int8Type>::from_iter(
            cat_array.into_iter().map(Some),
        ));
        RecordBatch::try_new(
            Arc::new(schema),
            vec![cell_ids_arr, n_genes_arr, is_doublet_arr, cat_arr],
        )
        .unwrap()
    }

    fn small_var_batch() -> RecordBatch {
        let gene_ids = StringArray::from(vec!["ENSG0", "ENSG1", "ENSG2", "ENSG3"]);
        let schema = Schema::new(vec![Field::new("_index", DataType::Utf8, false)]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(gene_ids)]).unwrap()
    }

    /// One empty CSR shard per obs shard so CSR row counts match obs.
    fn write_zero_csr_shard(writer: &mut ScxWriter, row_start: u64, n_rows: u64) {
        let indptr: Vec<u64> = vec![0u64; (n_rows + 1) as usize];
        writer
            .write_csr_shard(
                &indptr,
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                row_start,
            )
            .unwrap();
    }

    /// Build a sharded-obs SCX file with the per-shard fixture above.
    fn build_sharded_obs_scx(path: &std::path::Path, n_shards: u32, rows_per_shard: u64) {
        let n_obs = u64::from(n_shards) * rows_per_shard;
        let mut writer = ScxWriter::new(path, header(n_obs, 4)).unwrap();
        for shard_idx in 0..n_shards {
            let row_start = u64::from(shard_idx) * rows_per_shard;
            let batch = obs_shard_batch(row_start as usize, rows_per_shard as usize, shard_idx);
            writer
                .write_obs_shard(shard_idx, row_start, rows_per_shard, n_obs, &batch)
                .unwrap();
            write_zero_csr_shard(&mut writer, row_start, rows_per_shard);
        }
        writer.write_var(&small_var_batch()).unwrap();
        writer.finish().unwrap();
    }

    /// Sharded-obs SCX → streaming h5ad export. Confirms that each
    /// column round-trips with the same values as the eager (assembled)
    /// baseline — and that the categorical column's running global
    /// dictionary correctly unifies disjoint per-shard vocabularies.
    #[test]
    fn test_streaming_obs_round_trip_sharded_to_h5ad() {
        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("sharded.scx");
        let h5ad_stream = dir.path().join("stream.h5ad");
        let h5ad_eager = dir.path().join("eager.h5ad");

        // 4 shards × 50 rows = 200 obs. Disjoint dict per shard exposes
        // the running-global-dictionary code path.
        build_sharded_obs_scx(&scx_path, 4, 50);

        // Sanity: the fixture is actually sharded.
        let reader = ScxReader::open(&scx_path).unwrap();
        assert_eq!(reader.obs_metadata_shard_count(), 4);
        assert_eq!(reader.n_obs(), 200);

        // Streaming export.
        let opts = ConvertOptions::default();
        write_scx_to_h5ad_streaming(&scx_path, &h5ad_stream, &opts, &mut WarningSink::log())
            .unwrap();
        // Eager baseline (the existing materialising writer reads
        // assembled obs and writes via `write_dataframe_group_at`).
        write_scx_to_h5ad(&scx_path, &h5ad_eager, &mut WarningSink::log()).unwrap();

        // Re-read both via the SCX writer's own h5ad reader and
        // compare obs column-by-column. The eager baseline is the
        // source of truth (it goes through `reader.read_obs()` and
        // the existing eager column writer — same code path as before
        // task 6a).
        let stream_file = hdf5::File::open(&h5ad_stream).unwrap();
        let eager_file = hdf5::File::open(&h5ad_eager).unwrap();
        let stream_obs =
            read_dataframe_group(&stream_file, "obs", &mut WarningSink::log()).unwrap();
        let eager_obs = read_dataframe_group(&eager_file, "obs", &mut WarningSink::log()).unwrap();

        assert_eq!(stream_obs.num_rows(), 200);
        assert_eq!(stream_obs.num_rows(), eager_obs.num_rows());
        assert_eq!(stream_obs.num_columns(), eager_obs.num_columns());

        for col_name in ["cell_id", "n_genes", "is_doublet", "cell_type"] {
            let s_idx = stream_obs
                .schema()
                .index_of(col_name)
                .unwrap_or_else(|_| panic!("streaming output missing column {col_name}"));
            let e_idx = eager_obs.schema().index_of(col_name).unwrap();
            let s = stream_obs.column(s_idx);
            let e = eager_obs.column(e_idx);
            // Categorical reassembly may use different dictionary
            // key widths (eager uses i32 promoted; streaming reads
            // back as whatever AnnData's reader produces). Compare
            // logical values column-by-column, not dictionary codes.
            compare_columns_logical(s, e, col_name);
        }
    }

    /// Build an obs shard with two **numeric** categorical columns whose
    /// per-shard vocabularies are disjoint (so the streaming exporter's running
    /// global dictionary must unify them): `cluster` (`Dictionary<Int8, Int64>`,
    /// integer cluster labels) and `dose` (`Dictionary<Int8, Float64>`, float
    /// dose levels). Returns the batch plus the expected per-row logical values
    /// for cross-checking the round-trip.
    fn obs_shard_batch_numcat(
        start_row: usize,
        n: usize,
        shard_idx: u32,
    ) -> (RecordBatch, Vec<i64>, Vec<f64>) {
        let cell_ids: Vec<String> = (start_row..start_row + n)
            .map(|i| format!("cell_{i:06}"))
            .collect();
        let int_vocab: Vec<i64> = match shard_idx {
            0 => vec![10, 20],
            1 => vec![30, 40],
            2 => vec![10, 50],
            _ => vec![20],
        };
        let float_vocab: Vec<f64> = match shard_idx {
            0 => vec![0.1, 0.5],
            1 => vec![0.9, 0.25],
            2 => vec![0.1, 0.75],
            _ => vec![0.5],
        };
        let int_keys: Vec<i8> = (0..n).map(|i| (i % int_vocab.len()) as i8).collect();
        let float_keys: Vec<i8> = (0..n).map(|i| (i % float_vocab.len()) as i8).collect();
        let expected_int: Vec<i64> = int_keys.iter().map(|&k| int_vocab[k as usize]).collect();
        let expected_float: Vec<f64> = float_keys
            .iter()
            .map(|&k| float_vocab[k as usize])
            .collect();

        let cluster = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(int_keys),
            Arc::new(Int64Array::from(int_vocab)),
        )
        .unwrap();
        let dose = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(float_keys),
            Arc::new(Float64Array::from(float_vocab)),
        )
        .unwrap();

        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new(
                "cluster",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int64)),
                false,
            ),
            Field::new(
                "dose",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Float64)),
                false,
            ),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(cell_ids)) as ArrayRef,
                Arc::new(cluster) as ArrayRef,
                Arc::new(dose) as ArrayRef,
            ],
        )
        .unwrap();
        (batch, expected_int, expected_float)
    }

    /// Decode a `Dictionary<_, Int64>` column to its per-row `i64` values
    /// (no nulls in the fixture).
    fn decode_dict_i64(arr: &ArrayRef) -> Vec<i64> {
        let d = arr.as_any_dictionary();
        let vals = d.values().as_primitive::<Int64Type>();
        d.normalized_keys().iter().map(|&k| vals.value(k)).collect()
    }

    /// Decode a `Dictionary<_, Float64>` column to its per-row `f64` values.
    fn decode_dict_f64(arr: &ArrayRef) -> Vec<f64> {
        let d = arr.as_any_dictionary();
        let vals = d.values().as_primitive::<Float64Type>();
        d.normalized_keys().iter().map(|&k| vals.value(k)).collect()
    }

    /// Report PR #249 (Codex review): numeric (int/float) categoricals were
    /// preserved by the *eager* exporter but still dropped on the **streaming**
    /// (sharded obs/var) path. This asserts they now round-trip there too:
    /// the streamed `categories` datasets are numeric (not strings), the global
    /// vocabulary unifies the disjoint per-shard dicts, and per-row logical
    /// values match the source and the eager baseline.
    #[test]
    fn test_streaming_obs_numeric_categorical_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("numcat_sharded.scx");
        let h5ad_stream = dir.path().join("stream.h5ad");
        let h5ad_eager = dir.path().join("eager.h5ad");

        // 4 shards × 50 rows, disjoint numeric vocabularies per shard.
        let n_shards = 4u32;
        let rows_per_shard = 50u64;
        let n_obs = u64::from(n_shards) * rows_per_shard;
        let mut expected_int: Vec<i64> = Vec::new();
        let mut expected_float: Vec<f64> = Vec::new();
        {
            let mut writer = ScxWriter::new(&scx_path, header(n_obs, 4)).unwrap();
            for shard_idx in 0..n_shards {
                let row_start = u64::from(shard_idx) * rows_per_shard;
                let (batch, exp_i, exp_f) =
                    obs_shard_batch_numcat(row_start as usize, rows_per_shard as usize, shard_idx);
                writer
                    .write_obs_shard(shard_idx, row_start, rows_per_shard, n_obs, &batch)
                    .unwrap();
                write_zero_csr_shard(&mut writer, row_start, rows_per_shard);
                expected_int.extend(exp_i);
                expected_float.extend(exp_f);
            }
            writer.write_var(&small_var_batch()).unwrap();
            writer.finish().unwrap();
        }

        let reader = ScxReader::open(&scx_path).unwrap();
        assert_eq!(
            reader.obs_metadata_shard_count(),
            4,
            "fixture must be sharded"
        );
        assert_eq!(reader.n_obs(), n_obs);

        let opts = ConvertOptions::default();
        write_scx_to_h5ad_streaming(&scx_path, &h5ad_stream, &opts, &mut WarningSink::log())
            .unwrap();
        write_scx_to_h5ad(&scx_path, &h5ad_eager, &mut WarningSink::log()).unwrap();

        // On-disk: the streamed `categories` datasets must be numeric — reading
        // them as i64/f64 fails if the streaming path fell back to strings (or
        // dropped the column, in which case the group is absent).
        let sf = hdf5::File::open(&h5ad_stream).unwrap();
        let obs = sf.group("obs").unwrap();
        let int_cats: Vec<i64> = obs
            .group("cluster")
            .unwrap()
            .dataset("categories")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        let float_cats: Vec<f64> = obs
            .group("dose")
            .unwrap()
            .dataset("categories")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        // Global union across the disjoint per-shard vocabularies.
        let int_set: std::collections::HashSet<i64> = int_cats.iter().copied().collect();
        assert_eq!(
            int_set,
            std::collections::HashSet::from([10, 20, 30, 40, 50]),
            "cluster categories must unify across shards"
        );
        let float_set: std::collections::HashSet<u64> =
            float_cats.iter().map(|f| f.to_bits()).collect();
        let want_float: std::collections::HashSet<u64> = [0.1f64, 0.5, 0.9, 0.25, 0.75]
            .iter()
            .map(|f| f.to_bits())
            .collect();
        assert_eq!(
            float_set, want_float,
            "dose categories must unify across shards"
        );

        // Per-row logical values match the source and the eager baseline.
        let stream_obs = read_dataframe_group(&sf, "obs", &mut WarningSink::log()).unwrap();
        let ef = hdf5::File::open(&h5ad_eager).unwrap();
        let eager_obs = read_dataframe_group(&ef, "obs", &mut WarningSink::log()).unwrap();

        let s_cluster = stream_obs.column(stream_obs.schema().index_of("cluster").unwrap());
        let s_dose = stream_obs.column(stream_obs.schema().index_of("dose").unwrap());
        assert!(
            matches!(s_cluster.data_type(), DataType::Dictionary(_, v) if **v == DataType::Int64),
            "cluster should round-trip as Dictionary(_, Int64), got {:?}",
            s_cluster.data_type()
        );
        assert!(
            matches!(s_dose.data_type(), DataType::Dictionary(_, v) if **v == DataType::Float64),
            "dose should round-trip as Dictionary(_, Float64), got {:?}",
            s_dose.data_type()
        );
        assert_eq!(
            decode_dict_i64(s_cluster),
            expected_int,
            "cluster values (streaming)"
        );
        assert_eq!(
            decode_dict_f64(s_dose),
            expected_float,
            "dose values (streaming)"
        );

        let e_cluster = eager_obs.column(eager_obs.schema().index_of("cluster").unwrap());
        let e_dose = eager_obs.column(eager_obs.schema().index_of("dose").unwrap());
        assert_eq!(
            decode_dict_i64(e_cluster),
            expected_int,
            "cluster values (eager parity)"
        );
        assert_eq!(
            decode_dict_f64(e_dose),
            expected_float,
            "dose values (eager parity)"
        );
    }

    /// Streaming export when the entry point is called on a legacy
    /// single-section SCX file. The dispatcher must fall back to the
    /// eager path; output is byte-identical to the prior behavior.
    #[test]
    fn test_streaming_obs_legacy_single_section_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("legacy.scx");
        let h5ad_stream = dir.path().join("stream.h5ad");
        let h5ad_eager = dir.path().join("eager.h5ad");

        // Legacy single-section obs.
        {
            let n_obs: u64 = 80;
            let mut writer = ScxWriter::new(&scx_path, header(n_obs, 4)).unwrap();
            let batch = obs_shard_batch(0, n_obs as usize, 0);
            writer.write_obs(&batch).unwrap();
            write_zero_csr_shard(&mut writer, 0, n_obs);
            writer.write_var(&small_var_batch()).unwrap();
            writer.finish().unwrap();
        }

        let reader = ScxReader::open(&scx_path).unwrap();
        assert_eq!(
            reader.obs_metadata_shard_count(),
            0,
            "fixture must be legacy"
        );

        let opts = ConvertOptions::default();
        write_scx_to_h5ad_streaming(&scx_path, &h5ad_stream, &opts, &mut WarningSink::log())
            .unwrap();
        write_scx_to_h5ad(&scx_path, &h5ad_eager, &mut WarningSink::log()).unwrap();

        // Logical column equality is the user-visible contract.
        let stream_file = hdf5::File::open(&h5ad_stream).unwrap();
        let eager_file = hdf5::File::open(&h5ad_eager).unwrap();
        let stream_obs =
            read_dataframe_group(&stream_file, "obs", &mut WarningSink::log()).unwrap();
        let eager_obs = read_dataframe_group(&eager_file, "obs", &mut WarningSink::log()).unwrap();

        assert_eq!(stream_obs.num_rows(), eager_obs.num_rows());
        assert_eq!(stream_obs.num_columns(), eager_obs.num_columns());
        for col_name in ["cell_id", "n_genes", "is_doublet", "cell_type"] {
            let s = stream_obs.column(stream_obs.schema().index_of(col_name).unwrap());
            let e = eager_obs.column(eager_obs.schema().index_of(col_name).unwrap());
            compare_columns_logical(s, e, col_name);
        }
    }

    /// Sharded obs + active deletion vectors: only kept-row obs values
    /// must appear in the h5ad output, and the row count must match
    /// `/X/shape[0]` (which the streaming `/X` writer already filters
    /// by the same keep mask).
    #[test]
    fn test_streaming_obs_with_deletion_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("dv.scx");
        let h5ad_out = dir.path().join("out.h5ad");

        // 4 shards × 20 rows = 80 obs. Delete rows {3, 15, 27, 60} —
        // 76 kept rows, spread across shards (1st in shard 0, 2nd in
        // shard 0, 3rd in shard 1, 4th in shard 3).
        build_sharded_obs_scx(&scx_path, 4, 20);
        let deleted: Vec<u64> = vec![3, 15, 27, 60];
        scx_ops::mark_deleted(&scx_path, &deleted).unwrap();

        let opts = ConvertOptions::default();
        write_scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

        let file = hdf5::File::open(&h5ad_out).unwrap();
        let shape: Vec<i64> = file
            .group("X")
            .unwrap()
            .attr("shape")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        assert_eq!(shape[0], 76, "X kept-row count");

        // Read obs back and verify row count + that deleted cell_ids
        // do not appear.
        let obs = read_dataframe_group(&file, "obs", &mut WarningSink::log()).unwrap();
        assert_eq!(obs.num_rows(), 76, "obs kept-row count must match /X");
        let cell_id_col = obs.column(obs.schema().index_of("cell_id").unwrap());
        let s = cell_id_col.as_any().downcast_ref::<StringArray>().unwrap();
        let deleted_strings: std::collections::HashSet<String> =
            deleted.iter().map(|&i| format!("cell_{i:06}")).collect();
        for i in 0..s.len() {
            assert!(
                !deleted_strings.contains(s.value(i)),
                "obs row {i} ({}) was supposed to be deleted",
                s.value(i)
            );
        }
        // First-non-deleted row check: row 0 must be "cell_000000"
        // (row 0 was kept).
        assert_eq!(s.value(0), "cell_000000");
    }

    /// Non-streaming `write_scx_to_h5ad` on a sharded source with
    /// active deletion vectors. Pre-fix, this path passed `None` as
    /// the keep mask and dropped DVs silently on both /X (via the
    /// unfiltered `read_all_csr_shards`) and obs (via unfiltered
    /// `read_obs`). The fix routes both legs through the
    /// `_filtered` reader and the shared streaming-or-eager obs
    /// dispatcher, so the kept-row count is honored symmetrically.
    #[test]
    fn test_eager_obs_with_deletion_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("dv_eager.scx");
        let h5ad_out = dir.path().join("eager_out.h5ad");

        // Same fixture as the streaming DV test: 4×20 rows, delete 4.
        build_sharded_obs_scx(&scx_path, 4, 20);
        let deleted: Vec<u64> = vec![3, 15, 27, 60];
        scx_ops::mark_deleted(&scx_path, &deleted).unwrap();

        write_scx_to_h5ad(&scx_path, &h5ad_out, &mut WarningSink::log()).unwrap();

        let file = hdf5::File::open(&h5ad_out).unwrap();
        let shape: Vec<i64> = file
            .group("X")
            .unwrap()
            .attr("shape")
            .unwrap()
            .read_1d()
            .unwrap()
            .to_vec();
        assert_eq!(shape[0], 76, "non-streaming /X must filter DVs");

        let obs = read_dataframe_group(&file, "obs", &mut WarningSink::log()).unwrap();
        assert_eq!(
            obs.num_rows(),
            76,
            "non-streaming obs row count must match /X (was unfiltered pre-fix)"
        );
        let cell_id_col = obs.column(obs.schema().index_of("cell_id").unwrap());
        let s = cell_id_col.as_any().downcast_ref::<StringArray>().unwrap();
        let deleted_strings: std::collections::HashSet<String> =
            deleted.iter().map(|&i| format!("cell_{i:06}")).collect();
        for i in 0..s.len() {
            assert!(
                !deleted_strings.contains(s.value(i)),
                "obs row {i} ({}) was supposed to be deleted",
                s.value(i)
            );
        }
        assert_eq!(s.value(0), "cell_000000");
    }

    /// Obsm DV regression: after the eager / streaming paths started
    /// filtering /X and obs by the keep mask, a latent row-count
    /// mismatch on obsm became user-visible (obs.n_obs == X.shape[0]
    /// but obsm[key].shape[0] still equalled pre-deletion n_obs).
    /// This test locks in obsm filtering on both entry points.
    #[test]
    fn test_obsm_with_deletion_vectors() {
        use arrow::array::Float32Array;

        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("dv_obsm.scx");
        let h5ad_stream_out = dir.path().join("stream.h5ad");
        let h5ad_eager_out = dir.path().join("eager.h5ad");

        // 4×20 = 80 obs; add an X_pca-like obsm with 3 components.
        let n_shards: u32 = 4;
        let rows_per_shard: u64 = 20;
        let n_obs = u64::from(n_shards) * rows_per_shard;
        {
            let mut writer = ScxWriter::new(&scx_path, header(n_obs, 4)).unwrap();
            for shard_idx in 0..n_shards {
                let row_start = u64::from(shard_idx) * rows_per_shard;
                let batch = obs_shard_batch(row_start as usize, rows_per_shard as usize, shard_idx);
                writer
                    .write_obs_shard(shard_idx, row_start, rows_per_shard, n_obs, &batch)
                    .unwrap();
                write_zero_csr_shard(&mut writer, row_start, rows_per_shard);
            }
            writer.write_var(&small_var_batch()).unwrap();
            // Obsm: 80 × 3 dense Float32. Values encode (row, comp)
            // so we can verify post-filter alignment in the assert.
            let n = n_obs as usize;
            let c0: ArrayRef = Arc::new(Float32Array::from(
                (0..n).map(|i| i as f32).collect::<Vec<f32>>(),
            ));
            let c1: ArrayRef = Arc::new(Float32Array::from(
                (0..n).map(|i| (i as f32) + 0.5).collect::<Vec<f32>>(),
            ));
            let c2: ArrayRef = Arc::new(Float32Array::from(
                (0..n).map(|i| -(i as f32)).collect::<Vec<f32>>(),
            ));
            let obsm_schema = Schema::new(vec![
                Field::new("c0", DataType::Float32, false),
                Field::new("c1", DataType::Float32, false),
                Field::new("c2", DataType::Float32, false),
            ]);
            let obsm_batch = RecordBatch::try_new(Arc::new(obsm_schema), vec![c0, c1, c2]).unwrap();
            writer.write_obsm("X_pca", &obsm_batch).unwrap();
            writer.finish().unwrap();
        }

        let deleted: Vec<u64> = vec![3, 15, 27, 60];
        scx_ops::mark_deleted(&scx_path, &deleted).unwrap();

        let opts = ConvertOptions::default();
        write_scx_to_h5ad_streaming(&scx_path, &h5ad_stream_out, &opts, &mut WarningSink::log())
            .unwrap();
        write_scx_to_h5ad(&scx_path, &h5ad_eager_out, &mut WarningSink::log()).unwrap();

        let kept_rows: Vec<usize> = (0..n_obs as usize)
            .filter(|i| !deleted.contains(&(*i as u64)))
            .collect();
        assert_eq!(kept_rows.len(), 76);

        for path in [&h5ad_stream_out, &h5ad_eager_out] {
            let file = hdf5::File::open(path).unwrap();
            let obsm_pca = file.dataset("obsm/X_pca").unwrap();
            let shape = obsm_pca.shape();
            assert_eq!(
                shape,
                vec![76, 3],
                "obsm/X_pca shape mismatch at {path:?} — must equal kept-row count after DV filter",
            );
            // First-row spot-check: kept_rows[0] == 0 (row 0 is kept),
            // so the first surviving row's c0 should be `0.0`.
            let arr: ndarray::Array2<f32> = obsm_pca.read_2d().unwrap();
            assert_eq!(arr[[0, 0]], 0.0, "first kept row's c0 at {path:?}");
            assert_eq!(arr[[0, 1]], 0.5, "first kept row's c1 at {path:?}");
            // Last surviving row's c0 should equal kept_rows.last().
            let last_kept = *kept_rows.last().unwrap() as f32;
            assert_eq!(arr[[75, 0]], last_kept, "last kept row's c0 at {path:?}",);
        }
    }

    /// Build an obs shard fixture with an Arrow type the writer
    /// cannot encode (`Date32`) alongside the standard columns. Used
    /// by the issue-5 regression test.
    fn obs_shard_batch_with_unsupported(start_row: usize, n: usize, shard_idx: u32) -> RecordBatch {
        use arrow::array::Date32Array;
        let base = obs_shard_batch(start_row, n, shard_idx);
        // Append a Date32 column. Days-since-epoch values are arbitrary.
        let dates: Vec<i32> = (0..n).map(|i| (start_row + i) as i32).collect();
        let dates_arr: ArrayRef = Arc::new(Date32Array::from(dates));

        let mut fields: Vec<Field> = base
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new("captured_on", DataType::Date32, false));
        let new_schema = Arc::new(Schema::new(fields));
        let mut cols: Vec<ArrayRef> = (0..base.num_columns())
            .map(|i| base.column(i).clone())
            .collect();
        cols.push(dates_arr);
        RecordBatch::try_new(new_schema, cols).unwrap()
    }

    /// Issue 5: when a shard carries a column with an Arrow type the
    /// writer cannot encode (e.g. `Date32`), the streaming + eager
    /// paths warn-and-skip the column. The column must NOT appear in
    /// the `column-order` HDF5 attribute — otherwise
    /// `anndata.read_h5ad` raises a `KeyError` looking up a missing
    /// dataset.
    #[test]
    fn test_streaming_obs_unsupported_column_excluded_from_column_order() {
        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("unsupported.scx");
        let h5ad_stream = dir.path().join("stream.h5ad");
        let h5ad_eager = dir.path().join("eager.h5ad");

        // Build a sharded obs SCX with the unsupported `captured_on`
        // (Date32) column tacked onto each shard's batch.
        let n_shards: u32 = 2;
        let rows_per_shard: u64 = 20;
        let n_obs = u64::from(n_shards) * rows_per_shard;
        {
            let mut writer = ScxWriter::new(&scx_path, header(n_obs, 4)).unwrap();
            for shard_idx in 0..n_shards {
                let row_start = u64::from(shard_idx) * rows_per_shard;
                let batch = obs_shard_batch_with_unsupported(
                    row_start as usize,
                    rows_per_shard as usize,
                    shard_idx,
                );
                writer
                    .write_obs_shard(shard_idx, row_start, rows_per_shard, n_obs, &batch)
                    .unwrap();
                write_zero_csr_shard(&mut writer, row_start, rows_per_shard);
            }
            writer.write_var(&small_var_batch()).unwrap();
            writer.finish().unwrap();
        }

        let opts = ConvertOptions::default();
        write_scx_to_h5ad_streaming(&scx_path, &h5ad_stream, &opts, &mut WarningSink::log())
            .unwrap();
        // Eager path also routes through the streaming-or-eager
        // dispatcher post-issue-2 fix, but the assertion is the same:
        // the unsupported column must not leak into `column-order`.
        write_scx_to_h5ad(&scx_path, &h5ad_eager, &mut WarningSink::log()).unwrap();

        for path in [&h5ad_stream, &h5ad_eager] {
            let file = hdf5::File::open(path).unwrap();
            let obs = file.group("obs").unwrap();
            let column_order: Vec<hdf5::types::VarLenUnicode> = obs
                .attr("column-order")
                .unwrap()
                .read_1d()
                .unwrap()
                .to_vec();
            let names: Vec<String> = column_order.iter().map(|v| v.to_string()).collect();
            assert!(
                !names.iter().any(|n| n == "captured_on"),
                "unsupported `captured_on` (Date32) leaked into column-order at {path:?}: {names:?}",
            );
            // The supported columns must still be present.
            for expected in ["n_genes", "is_doublet", "cell_type"] {
                assert!(
                    names.iter().any(|n| n == expected),
                    "supported column `{expected}` missing from column-order at {path:?}: {names:?}",
                );
            }
            // No HDF5 dataset for the skipped column either.
            assert!(
                obs.dataset("captured_on").is_err(),
                "skipped column should not have a backing dataset at {path:?}",
            );
        }
    }

    /// Sharded **var** export: var has its own `VarMetadataShard`
    /// section type, and the streaming dispatcher must take the
    /// sharded var path when shards are present. var-axis deletion
    /// vectors don't exist, so this only validates the schema-only
    /// shard count plus pre-allocate + hyperslab round-trip.
    #[test]
    fn test_streaming_var_round_trip_sharded_to_h5ad() {
        use arrow::array::ArrayRef;

        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("sharded_var.scx");
        let h5ad_out = dir.path().join("out.h5ad");

        // 3 var shards × 5 rows = 15 genes. Two columns:
        // `gene_id: Utf8` (the index), `gene_type: Dictionary<Int8, Utf8>`.
        // The single-section obs path runs in parallel to keep the
        // fixture small.
        let n_obs: u64 = 10;
        let n_vars: u64 = 15;
        let var_shards: u32 = 3;
        let rows_per_var_shard: u64 = 5;
        {
            let mut writer = ScxWriter::new(&scx_path, header(n_obs, n_vars)).unwrap();
            // Tiny legacy obs.
            let obs_batch = obs_shard_batch(0, n_obs as usize, 0);
            writer.write_obs(&obs_batch).unwrap();
            // Sharded var.
            for s in 0..var_shards {
                let row_start = u64::from(s) * rows_per_var_shard;
                let gene_ids: Vec<String> = (row_start..row_start + rows_per_var_shard)
                    .map(|i| format!("ENSG{i:05}"))
                    .collect();
                let gene_types: Vec<&'static str> = (0..rows_per_var_shard)
                    .map(|i| {
                        if i.is_multiple_of(2) {
                            "protein_coding"
                        } else {
                            "lncRNA"
                        }
                    })
                    .collect();
                let var_schema = Schema::new(vec![
                    Field::new("gene_id", DataType::Utf8, false),
                    Field::new(
                        "gene_type",
                        DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                        false,
                    ),
                ]);
                let gene_id_arr: ArrayRef = Arc::new(StringArray::from(gene_ids));
                let gene_type_arr: ArrayRef = Arc::new(DictionaryArray::<Int8Type>::from_iter(
                    gene_types.into_iter().map(Some),
                ));
                let batch =
                    RecordBatch::try_new(Arc::new(var_schema), vec![gene_id_arr, gene_type_arr])
                        .unwrap();
                writer
                    .write_var_shard(s, row_start, rows_per_var_shard, n_vars, &batch)
                    .unwrap();
            }
            // One CSR shard matching obs rows.
            write_zero_csr_shard(&mut writer, 0, n_obs);
            writer.finish().unwrap();
        }

        // Sanity: var actually sharded.
        let reader = ScxReader::open(&scx_path).unwrap();
        assert_eq!(reader.var_metadata_shard_count(), var_shards as usize);

        let opts = ConvertOptions::default();
        write_scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

        let file = hdf5::File::open(&h5ad_out).unwrap();
        let var = read_dataframe_group(&file, "var", &mut WarningSink::log()).unwrap();
        assert_eq!(var.num_rows(), n_vars as usize);
        let gene_id_col = var.column(var.schema().index_of("gene_id").unwrap());
        let s = gene_id_col.as_any().downcast_ref::<StringArray>().unwrap();
        // Sharded contents arrived in order under the streaming write.
        assert_eq!(s.value(0), "ENSG00000");
        assert_eq!(
            s.value((n_vars - 1) as usize),
            format!("ENSG{:05}", n_vars - 1)
        );
    }

    /// h5mu equivalent: sharded global obs + multimodal CSR. The
    /// streaming h5mu writer must emit obs once at root and again
    /// per-modality (mudata convention).
    #[test]
    fn test_streaming_obs_round_trip_sharded_to_h5mu() {
        use crate::h5mu::write::scx_to_h5mu_streaming;
        use scx_format_io::modality::ModalityType;

        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("sharded_mm.scx");
        let h5mu_out = dir.path().join("out.h5mu");

        // Build a minimal multimodal SCX with sharded global obs.
        // Two modalities, one CSR shard each. 3 obs shards × 30 rows.
        let n_shards: u32 = 3;
        let rows_per_shard: u64 = 30;
        let n_obs = u64::from(n_shards) * rows_per_shard;
        {
            let mut writer = ScxWriter::new(&scx_path, header(n_obs, 0)).unwrap();
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
            // Per-modality var (small, not sharded). Must be written
            // before write_csr_shard_for picks up the n_vars from the
            // modality table.
            writer.write_var_for(rna_id, &small_var_batch()).unwrap();
            let adt_var = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "_index",
                    DataType::Utf8,
                    false,
                )])),
                vec![Arc::new(StringArray::from(vec!["CD4", "CD8"]))],
            )
            .unwrap();
            writer.write_var_for(adt_id, &adt_var).unwrap();
            for shard_idx in 0..n_shards {
                let row_start = u64::from(shard_idx) * rows_per_shard;
                let batch = obs_shard_batch(row_start as usize, rows_per_shard as usize, shard_idx);
                writer
                    .write_obs_shard(shard_idx, row_start, rows_per_shard, n_obs, &batch)
                    .unwrap();
            }
            // One zero-nnz CSR shard per modality covering all rows.
            let indptr: Vec<u64> = vec![0u64; (n_obs + 1) as usize];
            writer
                .write_csr_shard_for(
                    rna_id,
                    &indptr,
                    &[],
                    &[],
                    CodecId::None,
                    ValueEncoding::Uint8,
                    0,
                )
                .unwrap();
            writer
                .write_csr_shard_for(
                    adt_id,
                    &indptr,
                    &[],
                    &[],
                    CodecId::None,
                    ValueEncoding::Uint8,
                    0,
                )
                .unwrap();
            writer.finish().unwrap();
        }

        let opts = ConvertOptions::default();
        scx_to_h5mu_streaming(&scx_path, &h5mu_out, &opts, &mut WarningSink::log()).unwrap();

        let file = hdf5::File::open(&h5mu_out).unwrap();
        // Global obs present.
        let root_obs = read_dataframe_group(&file, "obs", &mut WarningSink::log()).unwrap();
        assert_eq!(root_obs.num_rows(), n_obs as usize);
        // Each column survived the streaming export.
        for col_name in ["cell_id", "n_genes", "is_doublet", "cell_type"] {
            assert!(
                root_obs.schema().index_of(col_name).is_ok(),
                "root obs missing column '{col_name}'"
            );
        }
        // Per-modality obs is also written (mudata convention) and
        // carries the same row count.
        let rna_obs = read_dataframe_group(&file, "mod/rna/obs", &mut WarningSink::log()).unwrap();
        assert_eq!(rna_obs.num_rows(), n_obs as usize);
        let adt_obs = read_dataframe_group(&file, "mod/adt/obs", &mut WarningSink::log()).unwrap();
        assert_eq!(adt_obs.num_rows(), n_obs as usize);
    }

    /// Helper: compare two h5ad-read columns by logical values.
    /// Handles the case where streaming and eager may pick different
    /// dictionary key widths or coerce strings differently — what
    /// matters is the user-visible string / int / bool.
    fn compare_columns_logical(s: &ArrayRef, e: &ArrayRef, col_name: &str) {
        let n = s.len();
        assert_eq!(e.len(), n, "{col_name}: length mismatch");
        match (s.data_type(), e.data_type()) {
            (DataType::Int32, DataType::Int32) => {
                let s = s.as_any().downcast_ref::<Int32Array>().unwrap();
                let e = e.as_any().downcast_ref::<Int32Array>().unwrap();
                for i in 0..n {
                    assert_eq!(
                        s.value(i),
                        e.value(i),
                        "{col_name} row {i}: streaming={} eager={}",
                        s.value(i),
                        e.value(i)
                    );
                }
            }
            (DataType::Utf8, DataType::Utf8) => {
                let s = s.as_any().downcast_ref::<StringArray>().unwrap();
                let e = e.as_any().downcast_ref::<StringArray>().unwrap();
                for i in 0..n {
                    assert_eq!(s.value(i), e.value(i), "{col_name} row {i}");
                }
            }
            (DataType::Boolean, DataType::Boolean) => {
                let s = s.as_any().downcast_ref::<BooleanArray>().unwrap();
                let e = e.as_any().downcast_ref::<BooleanArray>().unwrap();
                for i in 0..n {
                    assert_eq!(s.is_valid(i), e.is_valid(i), "{col_name} mask row {i}");
                    if s.is_valid(i) {
                        assert_eq!(s.value(i), e.value(i), "{col_name} value row {i}");
                    }
                }
            }
            // Dictionary <K, Utf8> — compare by resolved strings.
            (DataType::Dictionary(_, _), DataType::Dictionary(_, _)) => {
                let s_strings = dict_to_strings(s, col_name);
                let e_strings = dict_to_strings(e, col_name);
                assert_eq!(s_strings, e_strings, "{col_name}: categorical mismatch");
            }
            (lhs, rhs) => panic!("{col_name}: dtype mismatch (streaming={lhs:?}, eager={rhs:?})"),
        }
    }

    fn dict_to_strings(arr: &ArrayRef, col_name: &str) -> Vec<Option<String>> {
        // The h5ad reader produces Dictionary<Int32, Utf8> regardless
        // of the input key width (`dict_codes_and_categories_i32`
        // promotes everything to i32 on the eager path; the streaming
        // path does the same via the running global dict).
        let dict = arr
            .as_any()
            .downcast_ref::<DictionaryArray<arrow::datatypes::Int32Type>>()
            .unwrap_or_else(|| panic!("{col_name}: expected Dictionary<Int32, Utf8>"));
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("{col_name}: expected Utf8 dict values"));
        (0..dict.len())
            .map(|i| {
                if dict.is_valid(i) {
                    let code = dict.keys().value(i) as usize;
                    Some(values.value(code).to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    // ---- Nullable-encoding round-trip coverage (Patch 2) ----------------

    /// obs shard batch with explicitly nullable numeric / string columns.
    /// Besides the `cell_id` index (non-null):
    ///   - `ncount` Int32,   null when global row % 5 == 0
    ///   - `umi`    Int64,   null when global row % 3 == 0
    ///   - `pct`    Float32, null when global row % 4 == 0
    ///   - `score`  Float64, null when global row % 6 == 0
    ///   - `batch`  Utf8,    null when global row % 7 == 0
    fn nullable_obs_shard_batch(start_row: usize, n: usize) -> RecordBatch {
        let cell_ids: Vec<String> = (start_row..start_row + n)
            .map(|i| format!("cell_{i:06}"))
            .collect();
        let ncount: Vec<Option<i32>> = (0..n)
            .map(|i| {
                let g = start_row + i;
                (!g.is_multiple_of(5)).then_some(g as i32)
            })
            .collect();
        let umi: Vec<Option<i64>> = (0..n)
            .map(|i| {
                let g = start_row + i;
                (!g.is_multiple_of(3)).then_some(g as i64 * 1000)
            })
            .collect();
        let pct: Vec<Option<f32>> = (0..n)
            .map(|i| {
                let g = start_row + i;
                (!g.is_multiple_of(4)).then_some(g as f32 * 0.5)
            })
            .collect();
        let score: Vec<Option<f64>> = (0..n)
            .map(|i| {
                let g = start_row + i;
                (!g.is_multiple_of(6)).then_some(g as f64 * 1.5)
            })
            .collect();
        let batch: Vec<Option<String>> = (0..n)
            .map(|i| {
                let g = start_row + i;
                (!g.is_multiple_of(7)).then(|| format!("batch_{}", g % 3))
            })
            .collect();

        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("ncount", DataType::Int32, true),
            Field::new("umi", DataType::Int64, true),
            Field::new("pct", DataType::Float32, true),
            Field::new("score", DataType::Float64, true),
            Field::new("batch", DataType::Utf8, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(cell_ids)),
                Arc::new(Int32Array::from(ncount)),
                Arc::new(Int64Array::from(umi)),
                Arc::new(Float32Array::from(pct)),
                Arc::new(Float64Array::from(score)),
                Arc::new(StringArray::from(batch)),
            ],
        )
        .unwrap()
    }

    fn build_nullable_sharded_scx(path: &std::path::Path, n_shards: u32, rows_per_shard: u64) {
        let n_obs = u64::from(n_shards) * rows_per_shard;
        let mut writer = ScxWriter::new(path, header(n_obs, 4)).unwrap();
        for shard_idx in 0..n_shards {
            let row_start = u64::from(shard_idx) * rows_per_shard;
            let batch = nullable_obs_shard_batch(row_start as usize, rows_per_shard as usize);
            writer
                .write_obs_shard(shard_idx, row_start, rows_per_shard, n_obs, &batch)
                .unwrap();
            write_zero_csr_shard(&mut writer, row_start, rows_per_shard);
        }
        writer.write_var(&small_var_batch()).unwrap();
        writer.finish().unwrap();
    }

    /// On-disk `encoding-type` of an obs column when it is a group
    /// (categorical / nullable-*); `None` when the column is a plain
    /// dataset.
    fn obs_col_encoding(file: &hdf5::File, col: &str) -> Option<String> {
        file.group("obs")
            .unwrap()
            .group(col)
            .ok()
            .and_then(|g| g.attr("encoding-type").ok())
            .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
            .map(|v| v.to_string())
    }

    /// Assert the per-row null state + values of the nullable fixture
    /// survive the round trip through `read_dataframe_group`.
    fn assert_nullable_values(obs: &RecordBatch, n_obs: usize) {
        let ncount = obs.column(obs.schema().index_of("ncount").unwrap());
        let ncount = ncount.as_any().downcast_ref::<Int32Array>().unwrap();
        let umi = obs.column(obs.schema().index_of("umi").unwrap());
        let umi = umi.as_any().downcast_ref::<Int64Array>().unwrap();
        let pct = obs.column(obs.schema().index_of("pct").unwrap());
        let pct = pct.as_any().downcast_ref::<Float32Array>().unwrap();
        let score = obs.column(obs.schema().index_of("score").unwrap());
        let score = score.as_any().downcast_ref::<Float64Array>().unwrap();
        let batch = obs.column(obs.schema().index_of("batch").unwrap());
        let batch = batch.as_any().downcast_ref::<StringArray>().unwrap();

        for g in 0..n_obs {
            // Integer / string nulls preserve validity (mask).
            if g.is_multiple_of(5) {
                assert!(ncount.is_null(g), "ncount row {g} should be null");
            } else {
                assert_eq!(ncount.value(g), g as i32, "ncount row {g}");
            }
            if g.is_multiple_of(3) {
                assert!(umi.is_null(g), "umi row {g} should be null");
            } else {
                assert_eq!(umi.value(g), g as i64 * 1000, "umi row {g}");
            }
            if g.is_multiple_of(7) {
                assert!(batch.is_null(g), "batch row {g} should be null");
            } else {
                assert_eq!(batch.value(g), format!("batch_{}", g % 3), "batch row {g}");
            }
            // Float nulls become NaN (plain dataset, no validity).
            if g.is_multiple_of(4) {
                assert!(pct.value(g).is_nan(), "pct row {g} should be NaN");
            } else {
                assert_eq!(pct.value(g), g as f32 * 0.5, "pct row {g}");
            }
            if g.is_multiple_of(6) {
                assert!(score.value(g).is_nan(), "score row {g} should be NaN");
            } else {
                assert_eq!(score.value(g), g as f64 * 1.5, "score row {g}");
            }
        }
    }

    /// Assert the on-disk encodings: int/string → nullable group; float →
    /// plain dataset (anndata has no nullable-float spec).
    fn assert_nullable_encodings(file: &hdf5::File) {
        assert_eq!(
            obs_col_encoding(file, "ncount").as_deref(),
            Some("nullable-integer")
        );
        assert_eq!(
            obs_col_encoding(file, "umi").as_deref(),
            Some("nullable-integer")
        );
        assert_eq!(
            obs_col_encoding(file, "batch").as_deref(),
            Some("nullable-string-array")
        );
        // Floats stay plain datasets.
        assert_eq!(obs_col_encoding(file, "pct"), None, "pct must be plain");
        assert_eq!(obs_col_encoding(file, "score"), None, "score must be plain");
    }

    /// Eager export (legacy single-section obs): null int/string columns
    /// round-trip via nullable groups, floats via NaN.
    #[test]
    fn test_nullable_round_trip_eager() {
        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("nullable_legacy.scx");
        let h5ad = dir.path().join("out.h5ad");

        let n_obs: u64 = 60;
        {
            let mut writer = ScxWriter::new(&scx_path, header(n_obs, 4)).unwrap();
            writer
                .write_obs(&nullable_obs_shard_batch(0, n_obs as usize))
                .unwrap();
            write_zero_csr_shard(&mut writer, 0, n_obs);
            writer.write_var(&small_var_batch()).unwrap();
            writer.finish().unwrap();
        }
        assert_eq!(
            ScxReader::open(&scx_path)
                .unwrap()
                .obs_metadata_shard_count(),
            0,
            "fixture must be legacy single-section"
        );

        write_scx_to_h5ad(&scx_path, &h5ad, &mut WarningSink::log()).unwrap();

        let file = hdf5::File::open(&h5ad).unwrap();
        assert_nullable_encodings(&file);
        let obs = read_dataframe_group(&file, "obs", &mut WarningSink::log()).unwrap();
        assert_eq!(obs.num_rows(), n_obs as usize);
        assert_nullable_values(&obs, n_obs as usize);
    }

    /// Streaming export (sharded obs): same nullable contract, exercised
    /// through the pre-scan + per-shard nullable group writers.
    #[test]
    fn test_nullable_round_trip_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("nullable_sharded.scx");
        let h5ad = dir.path().join("out.h5ad");

        // 3 shards × 20 rows = 60 obs.
        build_nullable_sharded_scx(&scx_path, 3, 20);
        assert_eq!(
            ScxReader::open(&scx_path)
                .unwrap()
                .obs_metadata_shard_count(),
            3
        );

        let opts = ConvertOptions::default();
        write_scx_to_h5ad_streaming(&scx_path, &h5ad, &opts, &mut WarningSink::log()).unwrap();

        let file = hdf5::File::open(&h5ad).unwrap();
        assert_nullable_encodings(&file);
        let obs = read_dataframe_group(&file, "obs", &mut WarningSink::log()).unwrap();
        assert_eq!(obs.num_rows(), 60);
        assert_nullable_values(&obs, 60);
    }

    /// A null-free integer column must still be written as a plain
    /// dataset (no behaviour change for the common case). Uses the
    /// existing all-valid `n_genes` Int32 fixture column.
    #[test]
    fn test_null_free_int_column_stays_plain() {
        let dir = tempfile::tempdir().unwrap();
        let scx_path = dir.path().join("nullfree.scx");
        let h5ad = dir.path().join("out.h5ad");

        build_sharded_obs_scx(&scx_path, 2, 25);
        let opts = ConvertOptions::default();
        write_scx_to_h5ad_streaming(&scx_path, &h5ad, &opts, &mut WarningSink::log()).unwrap();

        let file = hdf5::File::open(&h5ad).unwrap();
        // `n_genes` has no nulls → plain dataset, not a nullable group.
        assert_eq!(
            obs_col_encoding(&file, "n_genes"),
            None,
            "null-free int column must remain a plain dataset"
        );
    }

    /// The streaming writer must reject a shard whose columns are
    /// reordered relative to the declared schema (same count) rather than
    /// writing values into the wrong HDF5 column.
    #[test]
    fn test_streaming_rejects_reordered_shard_schema() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("x.h5ad")).unwrap();
        let root = file.as_group().unwrap();

        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let good = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(StringArray::from(vec!["c0", "c1"])),
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![3, 4])),
            ],
        )
        .unwrap();
        // Same column count + types but `a`/`b` names swapped.
        let bad_schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("b", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
        ]);
        let bad = RecordBatch::try_new(
            Arc::new(bad_schema),
            vec![
                Arc::new(StringArray::from(vec!["c2", "c3"])),
                Arc::new(Int32Array::from(vec![5, 6])),
                Arc::new(Int32Array::from(vec![7, 8])),
            ],
        )
        .unwrap();

        let needs_nullable = vec![false; schema.fields().len()];
        let shards: Vec<Result<RecordBatch, scx_format_io::error::ScxError>> =
            vec![Ok(good), Ok(bad)];
        let res = write_dataframe_group_streaming(
            &root,
            "obs",
            &schema,
            shards,
            4,
            None,
            &needs_nullable,
            &mut WarningSink::log(),
        );
        let err = res.expect_err("reordered shard must be rejected");
        assert!(
            format!("{err}").contains("shard schema mismatch"),
            "unexpected error: {err}"
        );
    }

    /// The eager dataframe writer must handle wide string columns
    /// (`LargeUtf8`) and `Dictionary(_, LargeUtf8)` categoricals the same
    /// way the streaming writer does, rather than dropping them to
    /// `UnsupportedExportColumn`. A null-bearing `LargeUtf8` column must
    /// round-trip via a `nullable-string-array` group; a
    /// `Dictionary(Int32, LargeUtf8)` categorical (with a null code) must
    /// round-trip via a `categorical` group.
    #[test]
    fn test_eager_writes_largeutf8_and_largeutf8_categorical() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("largeutf8.h5ad")).unwrap();
        let root = file.as_group().unwrap();

        // First field is the index (no pandas metadata → field 0).
        // `note` is a null-bearing wide-string column; `ct` is a
        // categorical whose dictionary values are LargeUtf8 with a null
        // code at row 2.
        let keys = Int32Array::from(vec![Some(0), Some(1), None, Some(0)]);
        let cat_values: ArrayRef = Arc::new(LargeStringArray::from(vec!["typeA", "typeB"]));
        let ct = DictionaryArray::<Int32Type>::new(keys, cat_values);

        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("note", DataType::LargeUtf8, true),
            Field::new("ct", ct.data_type().clone(), true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(vec!["c0", "c1", "c2", "c3"])),
                Arc::new(LargeStringArray::from(vec![
                    Some("a"),
                    None,
                    Some("c"),
                    Some("d"),
                ])),
                Arc::new(ct),
            ],
        )
        .unwrap();

        write_dataframe_group_at(&root, "obs", &batch, &mut WarningSink::log()).unwrap();
        drop(file);

        let file = hdf5::File::open(dir.path().join("largeutf8.h5ad")).unwrap();
        // Neither column was dropped to UnsupportedExportColumn.
        assert_eq!(
            obs_col_encoding(&file, "note").as_deref(),
            Some("nullable-string-array"),
            "null-bearing LargeUtf8 column must use a nullable-string-array group"
        );
        assert_eq!(
            obs_col_encoding(&file, "ct").as_deref(),
            Some("categorical"),
            "Dictionary(_, LargeUtf8) column must use a categorical group"
        );

        let obs = read_dataframe_group(&file, "obs", &mut WarningSink::log()).unwrap();
        assert_eq!(obs.num_rows(), 4);

        // `note`: null mask + values survive (reader yields Utf8).
        let note = obs.column(obs.schema().index_of("note").unwrap());
        let note = note.as_any().downcast_ref::<StringArray>().unwrap();
        assert!(note.is_null(1), "note row 1 should be null");
        assert_eq!(note.value(0), "a");
        assert_eq!(note.value(2), "c");
        assert_eq!(note.value(3), "d");

        // `ct`: categorical round-trips (reader yields Dictionary<Int32, Utf8>).
        let ct_out = obs.column(obs.schema().index_of("ct").unwrap());
        let ct_out = ct_out
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();
        let cats = ct_out
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(ct_out.keys().is_null(2), "ct row 2 should be null");
        assert_eq!(cats.value(ct_out.keys().value(0) as usize), "typeA");
        assert_eq!(cats.value(ct_out.keys().value(1) as usize), "typeB");
        assert_eq!(cats.value(ct_out.keys().value(3) as usize), "typeA");
    }
}
