//! scx-convert integration tests — index_export (T5.6 split).

use super::convert_tests_common::*;

#[test]
fn convert_with_index_obs_writes_predicate_index() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 10, 5);

    let opts = ConvertOptions {
        index_obs: vec!["cell_type".to_string()],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    let bytes = reader
        .read_obs_predicate_index_bytes()
        .unwrap()
        .expect("obs predicate index should be present");
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
    assert_eq!(index.columns.len(), 1);
    match &index.columns[0] {
        scx_engine::index::IndexedColumn::Categorical(c) => {
            assert_eq!(c.column_name, "cell_type");
            assert_eq!(c.entries.len(), 2);
        }
        _ => panic!("expected categorical index for cell_type"),
    }
}

#[test]
fn convert_with_unknown_forced_index_column_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 6, 4);

    let opts = ConvertOptions {
        index_obs: vec!["nonexistent_column".to_string()],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let err = h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("nonexistent_column"),
        "expected error to mention the column name; got: {msg}"
    );
}

#[test]
fn convert_with_preset_missing_column_warns() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 6, 4);

    let opts = ConvertOptions {
        index_preset: Some("cellxgene".to_string()),
        ..ConvertOptions::default()
    };
    let counter = std::sync::Arc::new(std::sync::Mutex::new(0u64));
    let counter_clone = counter.clone();
    let mut sink = WarningSink::with_handler(move |w| {
        if matches!(
            w,
            super::warnings::ConvertWarning::MissingPresetIndexColumn { .. }
        ) {
            *counter_clone.lock().unwrap() += 1;
        }
    });
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();
    let n_missing = *counter.lock().unwrap();
    assert!(
        n_missing > 0,
        "expected at least one MissingPresetIndexColumn warning"
    );

    let reader = ScxReader::open(&scx_path).unwrap();
    let bytes = reader
        .read_obs_predicate_index_bytes()
        .unwrap()
        .expect("obs predicate index should be present");
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
    let names: Vec<&str> = index
        .columns
        .iter()
        .map(|c| match c {
            scx_engine::index::IndexedColumn::Categorical(c) => c.column_name.as_str(),
            scx_engine::index::IndexedColumn::Numeric(n) => n.column_name.as_str(),
        })
        .collect();
    assert!(
        names.contains(&"cell_type"),
        "expected cell_type to be indexed; got {names:?}"
    );
}

#[test]
fn unknown_index_preset_name_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 4, 3);

    let opts = ConvertOptions {
        index_preset: Some("does_not_exist".to_string()),
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let err = h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap_err();
    assert!(format!("{err}").contains("does_not_exist"));
}

/// Symmetry with `convert_with_index_obs_writes_predicate_index`: a
/// forced var column should round-trip into the var predicate index
/// section. Without this test the var branch of the new engine helper
/// is exercised only via auto-detect.
#[test]
fn convert_with_index_var_writes_predicate_index() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 8, 6);

    let opts = ConvertOptions {
        index_var: vec!["feature_type".to_string()],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    let bytes = reader
        .read_var_predicate_index_bytes()
        .unwrap()
        .expect("var predicate index should be present");
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
    assert!(
        index.columns.iter().any(|c| matches!(
            c,
            scx_engine::index::IndexedColumn::Categorical(cat)
                if cat.column_name == "feature_type"
        )),
        "expected feature_type to be indexed in var; got {:?}",
        index
            .columns
            .iter()
            .map(|c| match c {
                scx_engine::index::IndexedColumn::Categorical(c) => c.column_name.clone(),
                scx_engine::index::IndexedColumn::Numeric(n) => n.column_name.clone(),
            })
            .collect::<Vec<_>>()
    );
}

/// Forced var column that doesn't exist in the schema must surface
/// as a `ConvertError` (mirrors
/// `convert_with_unknown_forced_index_column_errors` for the var
/// axis). The high-cardinality rejection path is covered by the
/// engine unit test
/// `build_obs_predicate_index_bytes_forced_high_cardinality_errors`
/// — `high_cardinality_threshold` is not user-tunable from the
/// convert layer today (Phase 5a), so the missing-column branch is
/// the only forced-error shape reachable through the CLI flag
/// surface here.
#[test]
fn convert_with_forced_missing_var_column_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad_with_cell_type(&h5ad_path, 4, 3);

    let opts = ConvertOptions {
        index_var: vec!["no_such_var_column".to_string()],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let err = h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("no_such_var_column"),
        "expected error to mention forced var column name; got: {msg}"
    );
    assert!(
        msg.contains("missing column"),
        "expected error to mention the typed SkipReason; got: {msg}"
    );
}

/// Phase 5a multimodal: predicate index flags on an h5mu input must
/// emit `PredicateIndexSkippedMultimodal` (engine read-side is
/// unimodal-only today). This pins down the typed warning so a
/// future read-side per-modality lookup change can flip the
/// behaviour without breaking expectations silently.
#[test]
fn convert_h5mu_with_index_obs_emits_skip_warning() {
    use crate::h5mu::pipeline::h5mu_to_scx;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_path = dir.path().join("input.h5mu");
    let scx_path = dir.path().join("output.scx");
    create_test_h5mu(&h5mu_path, 6, 4, 3);

    let opts = ConvertOptions {
        index_obs: vec!["cell_type".to_string()],
        ..ConvertOptions::default()
    };
    let saw_skip = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let saw_skip_clone = saw_skip.clone();
    let mut sink = WarningSink::with_handler(move |w| {
        if matches!(
            w,
            super::warnings::ConvertWarning::PredicateIndexSkippedMultimodal { .. }
        ) {
            saw_skip_clone.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    h5mu_to_scx(&h5mu_path, &scx_path, &opts, &mut sink).unwrap();
    assert!(
        saw_skip.load(std::sync::atomic::Ordering::Relaxed),
        "expected PredicateIndexSkippedMultimodal warning when --index-obs is set on h5mu input"
    );

    // And the on-disk file must NOT have an obs predicate index section
    // — otherwise the engine read-side (unimodal-only) would silently
    // see an orphan.
    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(
        reader.read_obs_predicate_index_bytes().unwrap().is_none(),
        "h5mu output must not carry an obs predicate index until per-modality lookup lands"
    );
}

#[test]
fn convert_with_bitmap_always_emits_section() {
    use scx_format_io::section::SectionType;
    use scx_format_io::BitmapShard;
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad(&h5ad_path, 8, 4, "csr", false);

    let opts = ConvertOptions {
        bitmap: super::pipeline::BitmapPolicy::Always,
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(
        reader.header().has_bitmap(),
        "has_bitmap flag should be set"
    );
    let bm_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::BitmapShard)
        .collect();
    assert_eq!(
        bm_entries.len(),
        reader.header().n_csr_shards as usize,
        "one bitmap shard per CSR shard under always policy",
    );
    // Decode shard 0 and sanity-check against a CSR scan.
    let bm0 = reader.read_bitmap_shard(0).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    let dense = csr.to_dense().unwrap();
    let n_vars = bm0.n_vars as usize;
    // shard 0 spans rows [bm0.row_start, bm0.row_start + bm0.n_rows).
    let row_start = bm0.row_start as usize;
    let row_end = row_start + bm0.n_rows as usize;
    for col in 0..bm0.n_vars as usize {
        let mut expected = 0u64;
        for row in row_start..row_end {
            if dense[row * n_vars + col] != 0.0 {
                expected += 1;
            }
        }
        assert_eq!(
            bm0.gene_detection_count(col as u32),
            expected,
            "gene {col} count mismatch in shard 0",
        );
    }
    let _ = BitmapShard::build_from_csr(0, 0, 1, &[0], &[]); // sanity: type is callable from test crate
}

#[test]
fn convert_with_bitmap_auto_dense_skips() {
    // Fixture has density ~ 50% (2-3 nnz per row in 4-col matrix) which
    // is above the 30% auto-density threshold.
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad(&h5ad_path, 8, 4, "csr", false);

    let opts = ConvertOptions {
        bitmap: super::pipeline::BitmapPolicy::Auto,
        ..ConvertOptions::default()
    };
    let counter = std::sync::Arc::new(std::sync::Mutex::new(0u64));
    let counter_clone = counter.clone();
    let mut sink = WarningSink::with_handler(move |w| {
        if matches!(w, super::warnings::ConvertWarning::BitmapSkipped { .. }) {
            *counter_clone.lock().unwrap() += 1;
        }
    });
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();

    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(
        !reader.header().has_bitmap(),
        "auto+dense fixture should not write bitmaps"
    );
    assert!(
        *counter.lock().unwrap() > 0,
        "expected at least one BitmapSkipped warning under auto on a dense fixture"
    );
}

#[test]
fn bitmap_off_default_no_section() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("input.h5ad");
    let scx_path = dir.path().join("output.scx");
    create_test_h5ad(&h5ad_path, 6, 4, "csr", false);

    let opts = ConvertOptions::default();
    let mut sink = WarningSink::log();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut sink).unwrap();
    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(!reader.header().has_bitmap());
}

/// Streaming variant of `test_h5ad_csr_to_scx_to_h5ad_round_trip`.
/// Goes through `scx_to_h5ad_streaming` and checks that `/X/{data,
/// indices, indptr}` and the auxiliary `/layers/raw/*` group match
/// the source h5ad bit-for-bit.
#[test]
fn test_h5ad_csr_to_scx_to_h5ad_streaming_round_trip() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("test.h5ad");
    let scx_path = dir.path().join("test.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    let n_obs = 20;
    let n_vars = 15;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", true);

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    let orig_file = hdf5::File::open(&h5ad_path).unwrap();
    let out_file = hdf5::File::open(&h5ad_out).unwrap();

    let orig_data: Vec<f32> = orig_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let orig_indices: Vec<i32> = orig_file
        .dataset("X/indices")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let orig_indptr: Vec<i64> = orig_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();

    let out_data: Vec<f32> = out_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_indices: Vec<i32> = out_file
        .dataset("X/indices")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_indptr: Vec<i64> = out_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();

    assert_eq!(orig_data, out_data, "/X/data mismatch after streaming");
    assert_eq!(orig_indices, out_indices, "/X/indices mismatch");
    assert_eq!(orig_indptr, out_indptr, "/X/indptr mismatch");

    // Layer round-trip (the fixture writes `layers/raw` mirroring X).
    let orig_layer_data: Vec<f32> = orig_file
        .dataset("layers/raw/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_layer_data: Vec<f32> = out_file
        .dataset("layers/raw/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(
        orig_layer_data, out_layer_data,
        "streaming layer round-trip mismatch"
    );

    // Streaming and materialising writers must produce equivalent
    // CSR triplets; reuse `scx_to_h5ad` for the cross-check.
    let h5ad_mat = dir.path().join("out_mat.h5ad");
    scx_to_h5ad(&scx_path, &h5ad_mat, &mut WarningSink::log()).unwrap();
    let mat_file = hdf5::File::open(&h5ad_mat).unwrap();
    let mat_data: Vec<f32> = mat_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(
        mat_data, out_data,
        "streaming and materialising writers diverged on /X/data"
    );
}

/// `scx_to_h5mu_streaming` round-trip: ensures the multimodal h5mu
/// export streams per-modality `/X` and produces a valid output.
#[test]
fn test_scx_to_h5mu_streaming_round_trip() {
    use crate::h5mu::pipeline::h5mu_to_scx;
    use crate::h5mu::write::scx_to_h5mu_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_path = dir.path().join("mid.scx");
    let h5mu_out = dir.path().join("out.h5mu");
    create_test_h5mu(&h5mu_in, 8, 30, 5);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_to_h5mu_streaming(&scx_path, &h5mu_out, &opts, &mut WarningSink::log()).unwrap();

    let file = hdf5::File::open(&h5mu_out).unwrap();
    assert!(file.group("mod").is_ok());
    assert!(file.group("mod/rna").is_ok());
    assert!(file.group("mod/adt").is_ok());
    assert!(file.group("mod/rna/X").is_ok());
    assert!(file.group("mod/adt/X").is_ok());
    assert!(file.group("obs").is_ok());
    let rna_var = file.group("mod/rna/var").unwrap();
    assert!(rna_var.dataset("_index").is_ok());

    // Cross-check streaming vs. materialising writer on the same SCX.
    let h5mu_mat = dir.path().join("out_mat.h5mu");
    crate::h5mu::write::scx_to_h5mu(&scx_path, &h5mu_mat, &mut WarningSink::log()).unwrap();
    let mat_file = hdf5::File::open(&h5mu_mat).unwrap();
    let stream_rna_data: Vec<f32> = file
        .dataset("mod/rna/X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let mat_rna_data: Vec<f32> = mat_file
        .dataset("mod/rna/X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(
        stream_rna_data, mat_rna_data,
        "h5mu streaming and materialising writers diverged on /mod/rna/X/data"
    );
}

/// Multi-shard streaming export: force a small `shard_target_rows` so
/// the streaming writer must walk several shards and write multiple
/// hyperslab slices. Exercises the `nnz_offset` / `row_offset_kept`
/// accumulators and the indptr/indices/data write loop more than the
/// single-shard happy path.
#[test]
fn test_h5ad_streaming_multi_shard_round_trip() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("test.h5ad");
    let scx_path = dir.path().join("test.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    let n_obs = 32;
    let n_vars = 12;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);

    let opts = ConvertOptions {
        shard_target_rows: 7, // 5 shards for 32 rows
        ..Default::default()
    };
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    assert!(
        ScxReader::open(&scx_path)
            .unwrap()
            .catalog()
            .csr_shards_for_modality(0)
            .len()
            > 1,
        "fixture should produce multiple shards to exercise the streaming loop"
    );

    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    let orig_file = hdf5::File::open(&h5ad_path).unwrap();
    let out_file = hdf5::File::open(&h5ad_out).unwrap();
    let orig_data: Vec<f32> = orig_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_data: Vec<f32> = out_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let orig_indptr: Vec<i64> = orig_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_indptr: Vec<i64> = out_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(
        orig_data, out_data,
        "multi-shard streaming /X/data mismatch"
    );
    assert_eq!(
        orig_indptr, out_indptr,
        "multi-shard streaming /X/indptr mismatch"
    );
}

/// Streaming export with active deletion vectors: only kept rows
/// must appear in the output `/X/{indptr,indices,data}` and the
/// `shape[0]` attribute must reflect `n_obs - n_deleted`.
#[test]
fn test_h5ad_streaming_with_deletion_vectors() {
    use super::pipeline::scx_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_path = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("in.scx");
    let h5ad_out = dir.path().join("out.h5ad");

    let n_obs: usize = 10;
    let n_vars: usize = 8;
    create_test_h5ad(&h5ad_path, n_obs, n_vars, "csr", false);

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_path, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    // Mark rows 1, 3, 7 deleted; 7 kept rows remain.
    let deleted: Vec<u64> = vec![1, 3, 7];
    scx_ops::mark_deleted(&scx_path, &deleted).unwrap();

    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    let out_file = hdf5::File::open(&h5ad_out).unwrap();
    let shape: Vec<i64> = out_file
        .group("X")
        .unwrap()
        .attr("shape")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let n_obs_kept = (n_obs - deleted.len()) as i64;
    assert_eq!(shape[0], n_obs_kept, "kept-row count in shape attr");
    assert_eq!(shape[1], n_vars as i64);

    let out_indptr: Vec<i64> = out_file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(out_indptr.len() as i64, n_obs_kept + 1, "indptr length");

    // Materialising path applied by `read_all_csr_shards_filtered`
    // must produce the same indices/data as the streamed export.
    let reader = ScxReader::open(&scx_path).unwrap();
    let expected = reader.read_all_csr_shards_filtered().unwrap();
    let out_indices: Vec<i32> = out_file
        .dataset("X/indices")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    let out_data: Vec<f32> = out_file
        .dataset("X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(expected.indices, out_indices, "indices after DV streaming");
    assert_eq!(expected.data, out_data, "data after DV streaming");
}

/// Streaming variant of `test_modality_extract_to_h5ad`.
#[test]
fn test_modality_extract_to_h5ad_streaming() {
    use crate::h5mu::pipeline::h5mu_to_scx;
    use crate::h5mu::write::scx_modality_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_path = dir.path().join("mid.scx");
    let h5ad_out = dir.path().join("rna.h5ad");
    create_test_h5mu(&h5mu_in, 6, 40, 7);

    let opts = ConvertOptions::default();
    h5mu_to_scx(&h5mu_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_modality_to_h5ad_streaming(&scx_path, &h5ad_out, "rna", &opts, &mut WarningSink::log())
        .unwrap();

    let file = hdf5::File::open(&h5ad_out).unwrap();
    assert!(file.group("X").is_ok());
    assert!(file.group("obs").is_ok());
    assert!(file.group("var").is_ok());
    let var_idx = file.dataset("var/_index").unwrap();
    assert_eq!(var_idx.shape()[0], 40);
}

/// Regression test for the boolean encoding mismatch surfaced via
/// the export_streaming benchmark: the legacy
/// flat-u8-with-encoding-type-boolean shape was rejected by
/// `anndata.read_h5ad` (no registered IOSpec). The writer now emits
/// the canonical `nullable-boolean` group form (values + mask
/// datasets) which anndata reads natively.
#[test]
fn test_boolean_round_trip_via_streaming_export() {
    use super::pipeline::scx_to_h5ad_streaming;
    use crate::h5ad::read::read_dataframe_group;
    use arrow::array::BooleanArray;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_in = dir.path().join("bool.h5ad");
    let scx_path = dir.path().join("bool.scx");
    let h5ad_out = dir.path().join("bool_out.h5ad");

    let n_obs: usize = 20;
    let n_vars: usize = 5;
    create_test_h5ad(&h5ad_in, n_obs, n_vars, "csr", false);

    // Inject a legacy attribute-form boolean column at the input;
    // the SCX → h5ad writer must emit the modern group form
    // regardless of what came in.
    {
        let file = hdf5::File::open_rw(&h5ad_in).unwrap();
        let obs = file.group("obs").unwrap();
        let codes: Vec<u8> = (0..n_obs).map(|i| (i % 2) as u8).collect();
        let ds = obs
            .new_dataset::<u8>()
            .shape([n_obs])
            .create("is_doublet")
            .unwrap();
        ds.write(&codes).unwrap();
        ds.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("boolean"))
            .unwrap();
    }

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();
    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    // Output shape: /obs/is_doublet is a group with values + mask.
    let file = hdf5::File::open(&h5ad_out).unwrap();
    let g = file.group("obs/is_doublet").unwrap();
    assert!(g.dataset("values").is_ok(), "expected values dataset");
    assert!(g.dataset("mask").is_ok(), "expected mask dataset");
    let enc = g
        .attr("encoding-type")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap();
    assert_eq!(enc.as_str(), "nullable-boolean");
    let enc_v = g
        .attr("encoding-version")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap();
    assert_eq!(enc_v.as_str(), "0.1.0");

    // Reader round-trip via read_dataframe_group.
    let obs = read_dataframe_group(&file, "obs", &mut WarningSink::log()).unwrap();
    let idx = obs.schema().index_of("is_doublet").unwrap();
    let col = obs.column(idx);
    assert!(matches!(col.data_type(), DataType::Boolean));
    let bool_arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
    assert_eq!(bool_arr.len(), n_obs);
    for i in 0..n_obs {
        assert!(bool_arr.is_valid(i), "no nulls expected");
        assert_eq!(bool_arr.value(i), i % 2 == 1, "row {i}");
    }
}

/// Regression test for the categorical attribute-form bug that
/// surfaced via the export_streaming benchmark on census_500k:
/// `H5Acreate2(): object header message is too large` when a
/// categorical column has too many categories to fit in HDF5's
/// 64 KB attribute payload limit. The fix writes categoricals as a
/// group (`codes` + `categories` datasets), which has no such cap.
#[test]
fn test_categorical_wide_round_trip() {
    use super::pipeline::scx_to_h5ad_streaming;
    use crate::h5ad::read::read_dataframe_group;
    use arrow::array::DictionaryArray;
    use arrow::datatypes::Int32Type;

    let dir = tempfile::tempdir().unwrap();
    let h5ad_in = dir.path().join("wide_cat_in.h5ad");
    let scx_path = dir.path().join("wide_cat.scx");
    let h5ad_out = dir.path().join("wide_cat_out.h5ad");

    // 2048 categories × ~44 chars each ≈ 90 KB raw — well past the
    // ~64 KB attribute ceiling. The legacy writer fails with
    // `H5Acreate2: object header too large` on this fixture.
    let n_obs: usize = 4096;
    let n_cats: usize = 2048;
    let cats: Vec<VarLenUnicode> = (0..n_cats)
        .map(|i| vlu(&format!("category_with_long_descriptive_name_{i:08x}")))
        .collect();
    let codes: Vec<i32> = (0..n_obs as i32).map(|i| i % n_cats as i32).collect();

    let n_vars: usize = 5;
    create_test_h5ad(&h5ad_in, n_obs, n_vars, "csr", false);
    {
        let file = hdf5::File::open_rw(&h5ad_in).unwrap();
        let obs = file.group("obs").unwrap();
        // Write the wide categorical using the legacy attribute form
        // — anndata still emits this in some pipelines, and the SCX
        // ingest path handles it via `read_categorical_column`.
        let ds = obs
            .new_dataset::<i32>()
            .shape([n_obs])
            .create("wide_cat")
            .unwrap();
        ds.write(&codes).unwrap();
        ds.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("categorical"))
            .unwrap();
        // The input is intentionally written via the dataset form
        // for the *ingest* side; the export-side fix lives in the
        // writer. The legacy form fits at ingest because anndata
        // pipelines that produce such files use HDF5's compact-vs-
        // dense attribute storage transitions.
        ds.new_attr::<VarLenUnicode>()
            .shape([cats.len()])
            .create("categories")
            .unwrap()
            .write(&cats)
            .unwrap();
    }

    let opts = ConvertOptions::default();
    h5ad_to_scx(&h5ad_in, &scx_path, &opts, &mut WarningSink::log()).unwrap();

    // This is the call that previously failed at HDF5's attribute
    // limit. With the writer fix, it succeeds.
    scx_to_h5ad_streaming(&scx_path, &h5ad_out, &opts, &mut WarningSink::log()).unwrap();

    // Output shape: /obs/wide_cat is a *group* (not a dataset)
    // containing codes + categories.
    let file = hdf5::File::open(&h5ad_out).unwrap();
    let wide = file.group("obs/wide_cat").unwrap();
    assert!(
        wide.dataset("codes").is_ok(),
        "expected /obs/wide_cat/codes dataset"
    );
    assert!(
        wide.dataset("categories").is_ok(),
        "expected /obs/wide_cat/categories dataset"
    );
    let enc = wide
        .attr("encoding-type")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap();
    assert_eq!(enc.as_str(), "categorical");
    let enc_v = wide
        .attr("encoding-version")
        .unwrap()
        .read_scalar::<VarLenUnicode>()
        .unwrap();
    assert_eq!(enc_v.as_str(), "0.2.0");

    // Reader side: the new `read_categorical_group` path picks it up.
    let obs = read_dataframe_group(&file, "obs", &mut WarningSink::log()).unwrap();
    let idx = obs.schema().index_of("wide_cat").unwrap();
    let col = obs.column(idx);
    let dict = col
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .expect("expected Dictionary<Int32, Utf8>");
    assert_eq!(dict.values().len(), n_cats);
    assert_eq!(dict.len(), n_obs);
}
