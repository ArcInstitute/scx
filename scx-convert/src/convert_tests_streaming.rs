//! scx-convert integration tests — streaming (T5.6 split).

use super::convert_tests_common::*;

#[test]
fn streaming_csr_round_trip_matches_bulk_reader() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_csr.h5ad");
    let n_obs = 37;
    let n_vars = 15;
    create_test_h5ad(&path, n_obs, n_vars, "csr", false);

    let (bulk_indptr, bulk_indices, bulk_data, bulk_n_obs, bulk_n_vars) = {
        let f = hdf5::File::open(&path).unwrap();
        read_x_matrix(&f, MatrixFormat::Csr).unwrap()
    };
    assert_eq!(bulk_n_obs, n_obs);
    assert_eq!(bulk_n_vars, n_vars);

    let file = hdf5::File::open(&path).unwrap();
    let mut reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
    assert_eq!(reader.n_obs, n_obs);
    assert_eq!(reader.n_vars, n_vars);

    let (stream_indptr, stream_indices, stream_values) = drain_streaming(&mut reader, 10);

    let bulk_indptr_u64: Vec<u64> = bulk_indptr.iter().map(|&v| v as u64).collect();
    let bulk_indices_u32: Vec<u32> = bulk_indices.iter().map(|&v| v as u32).collect();
    assert_eq!(stream_indptr, bulk_indptr_u64);
    assert_eq!(stream_indices, bulk_indices_u32);
    assert_eq!(stream_values, bulk_data);
}

#[test]
fn streaming_layer_matches_x_for_synthetic_fixture() {
    // create_test_h5ad with include_extras=true writes a "raw" layer
    // that mirrors X exactly. Streaming the layer must therefore
    // yield the same content as streaming X.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_layer.h5ad");
    let n_obs = 24;
    let n_vars = 10;
    create_test_h5ad(&path, n_obs, n_vars, "csr", true);

    let file = hdf5::File::open(&path).unwrap();
    let mut x_reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
    let (x_indptr, x_indices, x_values) = drain_streaming(&mut x_reader, 7);

    let mut layer_reader = open_layer_streaming(&file, "raw", &mut WarningSink::log()).unwrap();
    assert_eq!(layer_reader.n_obs, n_obs);
    assert_eq!(layer_reader.n_vars, n_vars);
    let (l_indptr, l_indices, l_values) = drain_streaming(&mut layer_reader, 7);

    assert_eq!(x_indptr, l_indptr);
    assert_eq!(x_indices, l_indices);
    assert_eq!(x_values, l_values);
}

#[test]
fn streaming_rejects_csc_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_csc.h5ad");
    create_test_h5ad(&path, 6, 5, "csc", false);

    let file = hdf5::File::open(&path).unwrap();
    let err = open_x_streaming(&file, "X", MatrixFormat::Csc, &mut WarningSink::log()).unwrap_err();
    match err {
        ConvertError::StreamingUnsupported(msg) => {
            assert!(
                msg.contains("CSC"),
                "error message should mention CSC; got: {msg}"
            );
        }
        other => panic!("expected StreamingUnsupported, got {other:?}"),
    }
}

#[test]
fn streaming_rejects_dense_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_dense.h5ad");
    create_test_h5ad(&path, 6, 5, "dense", false);

    let file = hdf5::File::open(&path).unwrap();
    let err =
        open_x_streaming(&file, "X", MatrixFormat::Dense, &mut WarningSink::log()).unwrap_err();
    match err {
        ConvertError::StreamingUnsupported(msg) => {
            assert!(
                msg.contains("dense"),
                "error message should mention dense; got: {msg}"
            );
        }
        other => panic!("expected StreamingUnsupported, got {other:?}"),
    }
}

#[test]
fn streaming_empty_matrix_yields_no_shards() {
    // Build a minimal h5ad with n_obs = 0 by hand — create_test_h5ad's
    // loop is unbounded but works at zero, producing indptr = [0] and
    // empty indices / data datasets.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_empty.h5ad");
    let n_vars = 5;
    {
        let file = hdf5::File::create(&path).unwrap();
        let x = file.create_group("X").unwrap();
        let indptr: Vec<i64> = vec![0];
        x.new_dataset::<i64>()
            .shape([1])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        // hdf5 doesn't allow zero-sized writes via the typed builder;
        // create empty datasets via shape=[0] and skip the .write().
        x.new_dataset::<i32>().shape([0]).create("indices").unwrap();
        x.new_dataset::<f32>().shape([0]).create("data").unwrap();
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[0i64, n_vars as i64])
            .unwrap();
    }

    let file = hdf5::File::open(&path).unwrap();
    let mut reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
    assert_eq!(reader.n_obs, 0);
    assert_eq!(reader.n_vars, n_vars);
    assert!(reader.next_shard(16).is_none());
}

#[test]
fn streaming_handles_empty_rows() {
    // Hand-build a CSR fixture where some rows have zero nnz. The
    // streaming reader must yield row counts unchanged and nnz == 4.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_empty_rows.h5ad");
    let n_vars: usize = 4;
    let indptr: Vec<i64> = vec![0, 2, 2, 2, 4, 4]; // 5 rows, rows 1/2/4 empty
    let indices: Vec<i32> = vec![0, 1, 2, 3];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    {
        let file = hdf5::File::create(&path).unwrap();
        let x = file.create_group("X").unwrap();
        x.new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        x.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        x.new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[5i64, n_vars as i64])
            .unwrap();
    }

    let file = hdf5::File::open(&path).unwrap();
    let mut reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
    let (full_indptr, full_indices, full_values) = drain_streaming(&mut reader, 2);
    assert_eq!(full_indptr, vec![0, 2, 2, 2, 4, 4]);
    assert_eq!(full_indices, vec![0u32, 1, 2, 3]);
    assert_eq!(full_values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn streaming_round_trip_matches_non_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt.h5ad");
    create_test_h5ad(&h5ad, 53, 17, "csr", false);

    let scx_stream = dir.path().join("stream.scx");
    let scx_bulk = dir.path().join("bulk.scx");

    let opts = streaming_opts(16);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_stream,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx(&h5ad, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.header().n_vars, b.header().n_vars);
    assert_eq!(a.header().nnz, b.header().nnz);
    assert_eq!(a.header().n_csr_shards, b.header().n_csr_shards);

    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.shape, csr_b.shape);
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn streaming_empty_n_obs_produces_valid_scx() {
    // Hand-build a 0-row CSR h5ad — `create_test_h5ad`'s loop assumes
    // n_obs > 0 so we skip it.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("empty.h5ad");
    let n_vars: usize = 7;
    {
        let file = hdf5::File::create(&h5ad).unwrap();
        let x = file.create_group("X").unwrap();
        x.new_dataset::<i64>()
            .shape([1])
            .create("indptr")
            .unwrap()
            .write(&[0i64])
            .unwrap();
        x.new_dataset::<i32>().shape([0]).create("indices").unwrap();
        x.new_dataset::<f32>().shape([0]).create("data").unwrap();
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[0i64, n_vars as i64])
            .unwrap();

        // Minimal obs / var so write_obs / write_var don't error.
        let obs = file.create_group("obs").unwrap();
        let obs_index: Vec<VarLenUnicode> = Vec::new();
        obs.new_dataset::<VarLenUnicode>()
            .shape([0])
            .create("_index")
            .unwrap();
        let _ = obs_index;
        obs.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
        let var = file.create_group("var").unwrap();
        let var_idx: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("g{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([n_vars])
            .create("_index")
            .unwrap()
            .write(&var_idx)
            .unwrap();
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
    }

    let scx = dir.path().join("empty.scx");
    let opts = streaming_opts(16);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.header().n_obs, 0);
    assert_eq!(reader.header().n_vars, n_vars as u64);
    assert_eq!(reader.header().n_csr_shards, 0);
    assert_eq!(reader.header().nnz, 0);
}

#[test]
fn streaming_sets_index_dtype_1_when_n_vars_above_u16() {
    // 70_000 vars > u16::MAX (65535) → index_dtype must be 1 (u32 indices).
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("wide.h5ad");
    let n_obs: usize = 8;
    let n_vars: usize = 70_000;
    {
        let file = hdf5::File::create(&h5ad).unwrap();
        let x = file.create_group("X").unwrap();

        // One nnz per row, evenly spread across columns up to and
        // beyond the u16 limit so the on-disk index dtype must be u32
        // (i32 in scipy's CSR layout).
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::with_capacity(n_obs);
        let mut data: Vec<f32> = Vec::with_capacity(n_obs);
        let stride = n_vars / n_obs;
        for row in 0..n_obs {
            indices.push((row * stride) as i32);
            data.push((row + 1) as f32);
            indptr.push(data.len() as i64);
        }

        x.new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        x.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        x.new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, n_vars as i64])
            .unwrap();

        let obs = file.create_group("obs").unwrap();
        let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("c{i}"))).collect();
        obs.new_dataset::<VarLenUnicode>()
            .shape([n_obs])
            .create("_index")
            .unwrap()
            .write(&obs_index)
            .unwrap();
        obs.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();

        let var = file.create_group("var").unwrap();
        // Wide var index: writing 70k VarLenUnicode strings is fast
        // enough for a test and exercises the read_dataframe_group
        // path on a large dimension.
        let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("g{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([n_vars])
            .create("_index")
            .unwrap()
            .write(&var_index)
            .unwrap();
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
    }

    let scx = dir.path().join("wide.scx");
    let opts = streaming_opts(16);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.header().n_vars, n_vars as u64);
    assert_eq!(
        reader.header().index_dtype,
        1,
        "n_vars > u16::MAX requires index_dtype = 1 (u32 indices)"
    );
}

#[test]
fn streaming_csc_on_disk_routes_through_phase2_dispatcher() {
    // Pre-Phase 2 this used to reject CSC at the pipeline level.
    // Phase 2 lights up `open_csc_streaming`, so the same fixture
    // should now convert successfully via the in-memory CSC route
    // (no `memory_budget` set → MaterializedCsrStream).
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc.h5ad");
    create_test_h5ad(&h5ad, 6, 5, "csc", false);

    let scx = dir.path().join("csc.scx");
    let opts = streaming_opts(16);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("Phase 2 should route CSC through open_csc_streaming");
    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.header().n_obs, 6);
    assert_eq!(reader.header().n_vars, 5);
}

#[test]
fn streaming_csc_always_emits_sidecar_matching_non_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt_csc.h5ad");
    create_test_h5ad(&h5ad, 41, 13, "csr", false);

    let scx_stream = dir.path().join("stream_csc.scx");
    let scx_bulk = dir.path().join("bulk_csc.scx");
    let opts = ConvertOptions {
        shard_target_rows: 16,
        codec: None,
        csc: super::pipeline::CscPolicy::Always,
        csc_cols_per_shard: 5,
        tool: "scx".into(),
        ..ConvertOptions::default()
    };
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_stream,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx(&h5ad, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();

    // CSC sidecars present on both files, same shape.
    assert!(
        a.header().n_csc_shards >= 1,
        "streaming run should emit at least one CSC shard, got {}",
        a.header().n_csc_shards
    );
    assert_eq!(a.header().n_csc_shards, b.header().n_csc_shards);

    // Catalog CSC entries align.
    let csc_a: Vec<_> = a
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::CscShard)
        .collect();
    let csc_b: Vec<_> = b
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::CscShard)
        .collect();
    assert_eq!(csc_a.len(), csc_b.len());
    // Section lengths should match — `rebuild_csc_inplace` deterministic
    // on the same CSR.
    for (ea, eb) in csc_a.iter().zip(csc_b.iter()) {
        assert_eq!(ea.length, eb.length, "CSC shard length parity");
    }
}

#[test]
fn streaming_two_layer_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("layers.h5ad");
    // `create_test_h5ad` with include_extras=true writes a single
    // "raw" layer. Add a second layer by hand so we exercise the
    // multi-layer streaming loop.
    create_test_h5ad(&h5ad, 33, 9, "csr", true);
    {
        let file = hdf5::File::open_rw(&h5ad).unwrap();
        let layers = file.group("layers").unwrap();
        let counts = layers.create_group("counts").unwrap();
        // Reuse the same CSR triplet as "raw" — same shape, different
        // catalog name is enough to verify the multi-layer path.
        let raw = layers.group("raw").unwrap();
        let raw_indptr: Vec<i64> = raw.dataset("indptr").unwrap().read_1d().unwrap().to_vec();
        let raw_indices: Vec<i32> = raw.dataset("indices").unwrap().read_1d().unwrap().to_vec();
        let raw_data: Vec<f32> = raw.dataset("data").unwrap().read_1d().unwrap().to_vec();
        counts
            .new_dataset::<i64>()
            .shape([raw_indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&raw_indptr)
            .unwrap();
        counts
            .new_dataset::<i32>()
            .shape([raw_indices.len()])
            .create("indices")
            .unwrap()
            .write(&raw_indices)
            .unwrap();
        counts
            .new_dataset::<f32>()
            .shape([raw_data.len()])
            .create("data")
            .unwrap()
            .write(&raw_data)
            .unwrap();
        counts
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        counts
            .new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[33i64, 9])
            .unwrap();
    }

    let scx = dir.path().join("layers.scx");
    let opts = streaming_opts(8);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    let layer_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::LayerCsrShard)
        .collect();
    let raw_count = layer_entries
        .iter()
        .filter(|e| e.name.starts_with("raw_shard_"))
        .count();
    let counts_count = layer_entries
        .iter()
        .filter(|e| e.name.starts_with("counts_shard_"))
        .count();
    assert!(raw_count >= 1, "expected at least one 'raw' layer shard");
    assert!(
        counts_count >= 1,
        "expected at least one 'counts' layer shard"
    );
}

#[test]
fn streaming_skips_unreadable_layer() {
    // A dense `/layers/{name}` group must not abort the streaming
    // convert — `open_layer_streaming` rejects dense layers, the
    // pipeline should warn and continue. Mirrors the non-streaming
    // `read_layers` best-effort behaviour.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("bad_layer.h5ad");
    let n_obs = 12;
    let n_vars = 5;
    create_test_h5ad(&h5ad, n_obs, n_vars, "csr", true);
    {
        let file = hdf5::File::open_rw(&h5ad).unwrap();
        let layers = file.group("layers").unwrap();
        let dense = layers.create_group("dense_bad").unwrap();
        // Mark as a dense matrix so `open_layer_streaming` rejects it.
        dense
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("array"))
            .unwrap();
        dense
            .new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, n_vars as i64])
            .unwrap();
    }

    let scx = dir.path().join("bad_layer.scx");
    let opts = streaming_opts(8);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("streaming convert must skip the bad layer, not abort");

    let reader = ScxReader::open(&scx).unwrap();
    let layer_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == FmtSectionType::LayerCsrShard)
        .collect();
    let raw_count = layer_entries
        .iter()
        .filter(|e| e.name.starts_with("raw_shard_"))
        .count();
    let dense_count = layer_entries
        .iter()
        .filter(|e| e.name.starts_with("dense_bad_shard_"))
        .count();
    assert!(
        raw_count >= 1,
        "expected the valid 'raw' layer to still be converted"
    );
    assert_eq!(
        dense_count, 0,
        "the dense 'dense_bad' layer must be silently skipped, not emit shards"
    );
}

#[test]
fn streaming_provenance_escapes_path_quotes() {
    // A path containing a `"` would break the previous
    // `format!`-built JSON; the `serde_json::json!` construction
    // must produce parseable JSON for any path.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("has\"quote.h5ad");
    create_test_h5ad(&h5ad, 8, 4, "csr", false);

    let scx = dir.path().join("out.scx");
    let opts = streaming_opts(4);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    let prov = reader.read_provenance().unwrap();
    assert_eq!(prov.operations.len(), 1);
    let entry = &prov.operations[0];
    assert_eq!(entry.action, "convert");
    assert_eq!(entry.tool, "scx");
    // `params_json` must round-trip through serde_json::from_str —
    // proves the path was escaped correctly.
    let parsed: serde_json::Value =
        serde_json::from_str(&entry.params_json).expect("params_json must be valid JSON");
    assert_eq!(parsed["format"], "h5ad");
    assert_eq!(parsed["stream"], true);
    let recorded = parsed["input"]
        .as_str()
        .expect("input field must be a string");
    assert!(
        recorded.ends_with("has\"quote.h5ad"),
        "input path must contain the literal quote, got {recorded:?}"
    );
}

#[test]
fn streaming_provenance_uses_configured_tool_name() {
    // `ConvertOptions::tool` must flow through to the provenance
    // entry verbatim — `pyscx` overrides it to "pyscx" so the
    // recorded provenance reflects the actual caller.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt.h5ad");
    create_test_h5ad(&h5ad, 6, 3, "csr", false);

    let scx = dir.path().join("out.scx");
    let mut opts = streaming_opts(4);
    opts.tool = "pyscx".into();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    let prov = reader.read_provenance().unwrap();
    assert_eq!(prov.operations.len(), 1);
    assert_eq!(prov.operations[0].tool, "pyscx");
}

#[test]
fn streaming_through_trait_object() {
    // Phase 0 acceptance criterion 5: the existing concrete reader
    // must drive the streaming pipeline equivalently when accessed
    // through `&mut dyn CsrShardStream`. Drains a fixture twice —
    // once through the inherent `next_shard` (which the in-tree
    // pipeline uses today) and once through `next_csr_shard` on the
    // trait — and asserts every emitted shard matches.
    use crate::h5ad::stream::open_x_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("rt.h5ad");
    create_test_h5ad(&h5ad, 53, 7, "csr", false);
    let file = hdf5::File::open(&h5ad).unwrap();

    let mut concrete =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
    let mut trait_reader =
        open_x_streaming(&file, "X", MatrixFormat::Csr, &mut WarningSink::log()).unwrap();
    let dyn_reader: &mut dyn CsrShardStream = &mut trait_reader;

    let target = 16usize;
    let mut total_concrete_rows: usize = 0;
    let mut total_trait_rows: u64 = 0;

    loop {
        let lhs = concrete.next_shard(target);
        let rhs: Option<StreamedCsrShard> = dyn_reader.next_csr_shard(target).unwrap();
        match (lhs, rhs) {
            (None, None) => break,
            (None, Some(_)) | (Some(_), None) => {
                panic!("inherent and trait drains disagreed on termination")
            }
            (Some(Ok(l)), Some(r)) => {
                assert_eq!(l.row_start as u64, r.row_start);
                assert_eq!(l.n_rows as u32, r.n_rows);
                assert_eq!(l.indptr, r.indptr);
                assert_eq!(l.indices, r.indices);
                assert_eq!(l.values, r.values);
                total_concrete_rows += l.n_rows;
                total_trait_rows += r.n_rows as u64;
            }
            (Some(Err(e)), _) => panic!("inherent reader returned error: {e}"),
        }
    }
    assert_eq!(total_concrete_rows, 53);
    assert_eq!(total_trait_rows, 53);

    // Trait-object exposes shape and source name.
    assert_eq!(dyn_reader.n_obs(), 53);
    assert_eq!(dyn_reader.n_vars(), 7);
    assert_eq!(dyn_reader.source_matrix_name(), "X");
}

#[test]
fn phase1_streaming_dense_int_matches_non_streaming() {
    // Dense /X (f32 with integer values 1-200) → both pipelines must
    // produce the same CSR header counts and the same per-shard CSR
    // arrays. Exercises `DenseXStreamReader` via the new dispatch.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense_int.h5ad");
    create_test_h5ad(&h5ad, 47, 11, "dense", false);

    let scx_stream = dir.path().join("dense_int_stream.scx");
    let scx_bulk = dir.path().join("dense_int_bulk.scx");
    let opts = streaming_opts(16);

    h5ad_to_scx_streaming(
        &h5ad,
        &scx_stream,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx(&h5ad, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.header().n_vars, b.header().n_vars);
    assert_eq!(a.header().nnz, b.header().nnz);
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.shape, csr_b.shape);
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn phase1_streaming_dense_nan_retained_with_epsilon() {
    // With `dense_zero_epsilon > 0`, NaN cells must be retained — scipy/anndata
    // keep explicit NaN, and the eps==0 branch already keeps it via `v != 0.0`.
    // Before the fix, `NaN.abs() > eps` was false so NaN was silently dropped.
    // Drive the dense reader directly to assert on the pre-encode shard.
    use crate::stream::CsrShardStream;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense_nan.h5ad");
    let n_obs = 1usize;
    let n_vars = 4usize;
    // [NaN, 0.05 (< eps → drop), 0.0 (drop), 3.0 (keep)]
    let dense = vec![f32::NAN, 0.05, 0.0, 3.0];
    write_dense_h5ad(&h5ad, n_obs, n_vars, &dense);

    let opts = ConvertOptions {
        dense_zero_epsilon: 0.1,
        ..ConvertOptions::default()
    };
    let file = hdf5::File::open(&h5ad).unwrap();
    let mut reader =
        crate::h5ad::dense_stream::open_dense_streaming(&file, "X", &opts, &mut WarningSink::log())
            .unwrap();
    let shard = reader.next_csr_shard(16).unwrap().unwrap();

    assert_eq!(
        shard.indptr,
        vec![0u64, 2],
        "indptr should count 2 kept cells"
    );
    assert_eq!(
        shard.indices,
        vec![0u32, 3],
        "sub-epsilon 0.05 and 0.0 dropped; NaN (col 0) and 3.0 (col 3) kept"
    );
    assert_eq!(shard.values.len(), 2);
    assert!(
        shard.values[0].is_nan(),
        "NaN must be retained, not dropped, under dense_zero_epsilon"
    );
    assert_eq!(shard.values[1], 3.0);
}

#[test]
fn phase1_streaming_dense_float_round_trip() {
    // Fractional values force `ValueEncoding::Float32` and exercise
    // the codec dispatch on a float dense fixture. Verify the
    // round-trip preserves every nonzero within f32 tolerance.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense_float.h5ad");
    let n_obs = 9usize;
    let n_vars = 5usize;
    // hand-built float dense matrix with fractional values
    let mut dense = vec![0.0f32; n_obs * n_vars];
    dense[0] = 0.5;
    dense[3] = 1.25;
    dense[n_vars + 1] = 2.75;
    dense[3 * n_vars + 4] = -3.5;
    dense[7 * n_vars + 2] = 100.125;
    write_dense_h5ad(&h5ad, n_obs, n_vars, &dense);

    let scx = dir.path().join("dense_float.scx");
    let opts = streaming_opts(4);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.header().n_obs, n_obs as u64);
    assert_eq!(reader.header().n_vars, n_vars as u64);
    assert_eq!(reader.header().nnz, 5);
    let csr = reader.read_all_csr_shards().unwrap();
    // Reconstruct dense and compare element-wise within tolerance.
    let mut rebuilt = vec![0.0f32; n_obs * n_vars];
    for row in 0..n_obs {
        let start = csr.indptr[row] as usize;
        let end = csr.indptr[row + 1] as usize;
        for k in start..end {
            let col = csr.indices[k] as usize;
            rebuilt[row * n_vars + col] = csr.data[k];
        }
    }
    for i in 0..(n_obs * n_vars) {
        let diff = (dense[i] - rebuilt[i]).abs();
        assert!(
            diff < 1e-5,
            "value at index {i}: dense={}, scx={}, diff={diff}",
            dense[i],
            rebuilt[i]
        );
    }
}

#[test]
fn phase1_streaming_dense_empty_rows_round_trip() {
    // Rows of all zeros must preserve scipy CSR invariants:
    // `indptr[i] == indptr[i+1]` per empty row, total length n_obs+1.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("empty_rows.h5ad");
    let n_obs = 6usize;
    let n_vars = 4usize;
    let mut dense = vec![0.0f32; n_obs * n_vars];
    // Row 0 has one nonzero, rows 1-3 are empty, row 4 has two, row 5 empty.
    dense[2] = 7.0;
    dense[4 * n_vars] = 3.0;
    dense[4 * n_vars + 3] = 4.0;
    write_dense_h5ad(&h5ad, n_obs, n_vars, &dense);

    let scx = dir.path().join("empty_rows.scx");
    let opts = streaming_opts(8);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.indptr.len(), n_obs + 1);
    // Empty rows: indptr[i] == indptr[i+1].
    for i in [1usize, 2, 3, 5] {
        assert_eq!(
            csr.indptr[i],
            csr.indptr[i + 1],
            "row {i} should be empty (indptr[{i}]={}, indptr[{}]={})",
            csr.indptr[i],
            i + 1,
            csr.indptr[i + 1]
        );
    }
    assert_eq!(csr.indptr[n_obs], 3); // total nnz
}

#[test]
fn phase1_streaming_inferred_encoding_emits_warning() {
    // Build an h5ad whose `/X` group is sparse but has NO
    // `encoding-type` attribute. The streaming open path infers CSR
    // from the indptr+indices children and must emit one
    // `InferredEncoding` warning.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("no_enc.h5ad");
    write_csr_h5ad_without_encoding_type(&h5ad, 4, 3);

    let scx = dir.path().join("no_enc.scx");
    let opts = streaming_opts(8);
    let mut sink = WarningSink::log();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .unwrap();
    // The Phase 1 inference is emitted by both `detect_matrix_format`
    // (children-based fallback) and `open_x_streaming` (attr-absent
    // fallback). Either way the category counter must be non-zero.
    let n = sink.counts().get("inferred_encoding").copied().unwrap_or(0);
    assert!(n >= 1, "expected ≥1 inferred_encoding warning, got {n}");
}

#[test]
fn phase1_streaming_strict_uns_errors_on_unsupported_key() {
    // Build an h5ad with an unsupported `uns/bad3d` entry. Numeric N-D
    // arrays now round-trip via the tagged envelope (B7), so the fixture is
    // a non-numeric (string) 3-D array, which is still unrepresentable.
    // Lenient: convert succeeds + SkippedUnsKey warning. Strict: ConvertError.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("bad_uns.h5ad");
    create_test_h5ad(&h5ad, 4, 3, "csr", false);
    {
        let file = hdf5::File::open_rw(&h5ad).unwrap();
        let uns = file.create_group("uns").unwrap();
        // 3-D *string* dataset → read_uns_entry's N-D arm rejects non-numeric
        // dtypes (numeric N-D is now preserved via the envelope).
        let nd = ndarray::Array3::<VarLenUnicode>::from_elem((2, 2, 2), vlu("x"));
        uns.new_dataset::<VarLenUnicode>()
            .shape([2, 2, 2])
            .create("bad3d")
            .unwrap()
            .write(&nd)
            .unwrap();
    }
    let opts_lenient = streaming_opts(8);
    let scx = dir.path().join("bad_uns_lenient.scx");
    let mut sink = WarningSink::log();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts_lenient,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .expect("lenient mode must accept unsupported uns key");
    assert!(sink.counts().get("skipped_uns_key").copied().unwrap_or(0) >= 1);

    let mut opts_strict = streaming_opts(8);
    opts_strict.strict_uns = true;
    let scx_strict = dir.path().join("bad_uns_strict.scx");
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx_strict,
        &opts_strict,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect_err("strict mode must error on first unsupported uns key");
    let msg = format!("{err}");
    assert!(
        msg.contains("uns") || msg.contains("dataset shape") || msg.contains("scalar"),
        "expected unsupported-uns error, got: {msg}"
    );
}

#[test]
fn deep_uns_group_chain_is_bounded_not_a_stack_overflow() {
    // `read_uns_entry` recurses over `/uns` subgroups, and an h5ad's group
    // tree can nest arbitrarily deep. Uncapped, this overflowed the stack and
    // aborted the process — verified before the fix: a 30000-deep chain
    // SIGSEGV'd `pyscx.read_h5ad_metadata`. Nothing here is a pyscx *write*:
    // the crash was on reading a file someone else produced.
    //
    // Observed red on unmodified source: 500 is already past what a Rust test
    // thread's 2 MB stack survives, so this test did not merely fail — it
    // aborted the whole `scx-convert` test binary with
    // "has overflowed its stack ... (signal: 6, SIGABRT)". That is the defect
    // stated as plainly as it can be: a depth no assertion ever got to judge.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("deep_uns.h5ad");
    create_test_h5ad(&h5ad, 4, 3, "csr", false);
    {
        let file = hdf5::File::open_rw(&h5ad).unwrap();
        let mut group = file
            .create_group("uns")
            .unwrap()
            .create_group("bomb")
            .unwrap();
        for i in 0..500 {
            group = group.create_group(&format!("g{i}")).unwrap();
        }
        group
            .new_dataset::<i64>()
            .shape(())
            .create("leaf")
            .unwrap()
            .write_scalar(&1i64)
            .unwrap();
    }

    // Lenient (the default): the depth error travels the same `strict_uns`
    // route as any other unrepresentable key, so the file still converts and
    // the truncated subtree is reported rather than dropped in silence.
    let scx = dir.path().join("deep_uns_lenient.scx");
    let mut sink = WarningSink::log();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &streaming_opts(8),
        &StreamingOverrides::default(),
        &mut sink,
    )
    .expect("lenient mode must convert a file with an over-deep uns key");
    assert!(sink.counts().get("skipped_uns_key").copied().unwrap_or(0) >= 1);

    // And what survived must be readable: the ingest cap is the same constant
    // the writer enforces, so a truncated tree can never be too deep to parse.
    let reader = ScxReader::open(&scx).unwrap();
    assert!(reader.read_uns().is_ok(), "truncated uns must parse back");

    let mut opts_strict = streaming_opts(8);
    opts_strict.strict_uns = true;
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &dir.path().join("deep_uns_strict.scx"),
        &opts_strict,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect_err("strict mode must refuse an over-deep uns key");
    let msg = format!("{err}");
    assert!(
        msg.contains("nesting is deeper than"),
        "expected an uns depth error, got: {msg}"
    );
    // The full key path, not just the leaf group. Reverting to the leaf name
    // would still satisfy the assertion above, so the top-level key — the only
    // part of `bomb/g0/.../g58` a user can act on — needs pinning separately.
    assert!(
        msg.contains("bomb/g"),
        "depth error must name the full key path, got: {msg}"
    );
}

#[test]
fn scx_uns_between_the_write_cap_and_the_read_ceiling_still_exports_to_h5ad() {
    // The producer/consumer asymmetry, exercised end to end rather than only
    // at the constant layer. `pyscx` writers stop at MAX_UNS_DEPTH (60), but a
    // file written by an older uncapped SCX — or by `scx-ops` merge, whose
    // `Namespace` policy wraps each input in one extra level — can legally sit
    // above it and still parse. Binding the *exporter* to the producer cap
    // would make such a file suddenly unexportable, so it stops at
    // SERDE_JSON_MAX_NESTING (127) instead. 100 is in that window.
    use scx_format_io::writer::ScxWriter;
    use scx_format_io::{FileHeader, MAX_UNS_DEPTH, SERDE_JSON_MAX_NESTING};

    let nest = |n: usize| {
        let mut v = serde_json::json!("leaf");
        for i in 0..n {
            v = serde_json::json!({ format!("g{i}"): v });
        }
        serde_json::json!({ "deep": v })
    };

    let dir = tempfile::tempdir().unwrap();
    let scx = dir.path().join("deep_uns.scx");
    let n_obs = 4usize;

    let depth = 100;
    assert!(depth > MAX_UNS_DEPTH && depth < SERDE_JSON_MAX_NESTING);

    let mut w = ScxWriter::new(&scx, FileHeader::new_single_modality(4, 2, 0, 4, 0, 0)).unwrap();
    w.write_uns(&nest(depth - 1)).unwrap();
    w.write_csr_shard(
        &vec![0u64; n_obs + 1],
        &[],
        &[],
        CodecId::None,
        ValueEncoding::Uint8,
        0,
    )
    .unwrap();
    w.finish().unwrap();

    // It reads back...
    let reader = ScxReader::open(&scx).unwrap();
    reader.read_uns().expect("a sub-127 uns must parse");
    drop(reader);

    // ...and it exports.
    let h5ad = dir.path().join("deep_uns.h5ad");
    crate::h5ad::write::write_scx_to_h5ad(&scx, &h5ad, &mut WarningSink::log())
        .expect("an uns above the write cap but below the read ceiling must still export");
    {
        let f = hdf5::File::open(&h5ad).unwrap();
        assert!(f.group("uns/deep/g98").is_ok(), "the chain must survive");
    }

    // And the writer refuses to store what no reader could take back, which is
    // what keeps the exporter's 127 a safe ceiling rather than a hopeful one.
    let too_deep = nest(SERDE_JSON_MAX_NESTING + 50);
    let mut w2 = ScxWriter::new(
        &dir.path().join("over.scx"),
        FileHeader::new_single_modality(4, 2, 0, 4, 0, 0),
    )
    .unwrap();
    let err = w2
        .write_uns(&too_deep)
        .expect_err("write_uns must reject an unreadable uns");
    assert!(
        format!("{err}").contains("nests deeper"),
        "expected an uns depth error, got: {err}"
    );
}

#[test]
fn phase1_streaming_dense_determinism() {
    // Re-running the streaming dense pipeline on the same fixture
    // must produce byte-identical output.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("det.h5ad");
    create_test_h5ad(&h5ad, 20, 6, "dense", false);
    let opts = streaming_opts(5);

    let scx_a = dir.path().join("a.scx");
    let scx_b = dir.path().join("b.scx");
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_a,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_b,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let bytes_a = std::fs::read(&scx_a).unwrap();
    let bytes_b = std::fs::read(&scx_b).unwrap();
    // SCX provenance carries a timestamp — strip the on-disk file
    // checksum from the comparison by comparing the CSR shards
    // directly instead of full file bytes.
    let ra = ScxReader::open(&scx_a).unwrap();
    let rb = ScxReader::open(&scx_b).unwrap();
    let csr_a = ra.read_all_csr_shards().unwrap();
    let csr_b = rb.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
    // Both writes should have produced the same number of bytes
    // even though provenance timestamps may differ.
    assert_eq!(
        bytes_a.len(),
        bytes_b.len(),
        "two streaming runs produced different output sizes"
    );
}

#[test]
fn phase1_streaming_dense_memory_budget_caps_slab() {
    // With `memory_budget` set so the per-row dense cost forces
    // `max_slab_rows < shard_target_rows`, the first emitted shard
    // must have `n_rows < shard_target_rows`.
    use super::stream::CsrShardStream;
    use crate::h5ad::dense_stream::open_dense_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("budget.h5ad");
    let n_obs = 32usize;
    let n_vars = 1000usize;
    let dense = vec![0.0f32; n_obs * n_vars];
    write_dense_h5ad(&h5ad, n_obs, n_vars, &dense);

    let file = hdf5::File::open(&h5ad).unwrap();
    // Budget = n_vars * 4 (f32) * shard_target_rows / 2 — half what
    // a full shard would need, so the slab cap activates.
    let shard_target_rows: usize = 16;
    let budget = (n_vars as u64) * 4 * (shard_target_rows as u64) / 2;

    let opts = ConvertOptions {
        memory_budget: Some(budget),
        shard_target_rows: shard_target_rows as u32,
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let mut reader = open_dense_streaming(&file, "X", &opts, &mut sink).unwrap();
    let shard = reader
        .next_csr_shard(shard_target_rows)
        .unwrap()
        .expect("expected at least one shard");
    assert!(
        (shard.n_rows as usize) < shard_target_rows,
        "memory_budget should cap slab to <{shard_target_rows} rows; got {}",
        shard.n_rows
    );
    assert!(shard.n_rows >= 1);
}

#[test]
fn phase1_streaming_dense_budget_too_small_actionable_error() {
    // `memory_budget` smaller than a single dense row must be
    // rejected with a clear error rather than silently disabling the
    // slab cap (and risking OOM). Mirrors
    // `phase2_streaming_csc_budget_too_small_actionable_error` but
    // exercises the dense path.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("dense_tiny_budget.h5ad");
    let n_obs = 4usize;
    let n_vars = 1000usize;
    let dense = vec![1.0f32; n_obs * n_vars];
    write_dense_h5ad(&h5ad, n_obs, n_vars, &dense);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 2,
        // `n_vars * 4` is 4000 bytes per row; budget = 1 byte cannot
        // fit anything.
        memory_budget: Some(1),
        ..ConvertOptions::default()
    };
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect_err("budget=1 byte must be rejected for a dense matrix");
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("memory_budget"),
        "error should mention memory_budget; got: {msg}"
    );
    assert!(
        msg.contains("dense"),
        "error should mention dense path; got: {msg}"
    );
}

#[test]
fn phase2_streaming_csc_matches_in_memory_csr() {
    // Default budget (None) → in-memory CSC route via
    // MaterializedCsrStream. Output must match the non-streaming
    // `h5ad_to_scx` path which uses the same csc_to_csr scatter.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_small.h5ad");
    create_test_h5ad(&h5ad, 41, 13, "csc", false);

    let scx_stream = dir.path().join("stream.scx");
    let scx_bulk = dir.path().join("bulk.scx");
    let opts = streaming_opts(16);

    h5ad_to_scx_streaming(
        &h5ad,
        &scx_stream,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx(&h5ad, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.header().n_vars, b.header().n_vars);
    assert_eq!(a.header().nnz, b.header().nnz);
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn phase2_streaming_csc_external_transpose_matches_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_ext.h5ad");
    create_test_h5ad(&h5ad, 41, 13, "csc", false);

    let scx_ext = dir.path().join("ext.scx");
    let scx_bulk = dir.path().join("bulk.scx");
    let bulk_opts = streaming_opts(16);

    // 2 KiB budget — well below the in-memory threshold of
    // `16 × nnz + 16 × n_obs`, so the external route is forced.
    let ext_opts = ConvertOptions {
        shard_target_rows: 16,
        memory_budget: Some(2048),
        ..ConvertOptions::default()
    };

    h5ad_to_scx_streaming(
        &h5ad,
        &scx_ext,
        &ext_opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    h5ad_to_scx(&h5ad, &scx_bulk, &bulk_opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_ext).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
}

#[test]
fn phase2_streaming_csc_unsorted_rows_per_col() {
    // CSC where per-column row indices are NOT sorted. scipy allows
    // this; the streaming pipeline's downstream `sort_csr_rows_in_place`
    // keeps the output canonical.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_unsorted.h5ad");
    // 3 rows, 2 cols. Col 0 has rows in order [2, 0] (unsorted)
    // with values 5.0, 3.0. Col 1 has row [1] with value 7.0.
    write_csc_h5ad(
        &h5ad,
        3,
        2,
        &[0i64, 2, 3],
        &[2i32, 0, 1],
        &[5.0f32, 3.0, 7.0],
    );

    let scx = dir.path().join("out.scx");
    let opts = streaming_opts(8);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    // Row 0: col 0, val 3.0; Row 1: col 1, val 7.0; Row 2: col 0, val 5.0
    assert_eq!(csr.indptr, vec![0, 1, 2, 3]);
    assert_eq!(csr.indices, vec![0, 1, 0]);
    assert_eq!(csr.data, vec![3.0, 7.0, 5.0]);
}

#[test]
fn phase2_streaming_csc_external_unsorted_rows_per_col() {
    // Same fixture, but force the external transposer to exercise
    // the sort+coalesce path.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_unsorted_ext.h5ad");
    write_csc_h5ad(
        &h5ad,
        3,
        2,
        &[0i64, 2, 3],
        &[2i32, 0, 1],
        &[5.0f32, 3.0, 7.0],
    );

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 8,
        memory_budget: Some(1024),
        ..ConvertOptions::default()
    };
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.indptr, vec![0, 1, 2, 3]);
    assert_eq!(csr.indices, vec![0, 1, 0]);
    assert_eq!(csr.data, vec![3.0, 7.0, 5.0]);
}

#[test]
fn phase2_streaming_csc_duplicate_coords_sum() {
    // CSC with two entries at the SAME (row, col). The external
    // transposer's coalesce step sums them; the writer coordinator
    // emits one `DuplicateCoordinatesMerged` warning per non-zero
    // shard.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_dup.h5ad");
    // 2 rows × 2 cols. Col 0 has TWO entries at row 0 (vals 1.0, 2.5),
    // plus one at row 1 (val 4.0). Col 1 has one entry at row 0 (val 0.5).
    write_csc_h5ad(
        &h5ad,
        2,
        2,
        &[0i64, 3, 4],
        &[0i32, 0, 1, 0],
        &[1.0f32, 2.5, 4.0, 0.5],
    );

    let scx = dir.path().join("out.scx");
    // Budget below the in-memory threshold (16*nnz + 16*n_obs = 96 B
    // here) but above the 4-record minimum (64 B) — forces the
    // external transposer which is the path that coalesces
    // duplicates.
    let opts = ConvertOptions {
        shard_target_rows: 8,
        memory_budget: Some(80),
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .unwrap();
    let dup = sink
        .counts()
        .get("duplicate_coordinates_merged")
        .copied()
        .unwrap_or(0);
    assert!(
        dup >= 1,
        "expected ≥1 DuplicateCoordinatesMerged warning, got {dup}"
    );

    let reader = ScxReader::open(&scx).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    // Row 0: col 0 → 1.0 + 2.5 = 3.5; col 1 → 0.5
    // Row 1: col 0 → 4.0
    assert_eq!(csr.indptr, vec![0, 2, 3]);
    assert_eq!(csr.indices, vec![0, 1, 0]);
    assert_eq!(csr.data, vec![3.5, 0.5, 4.0]);
}

#[test]
fn phase2_streaming_csc_explicit_zeros_dropped() {
    // CSC with explicit 0.0 entries in `data` must round-trip
    // without those zeros (`drop_explicit_zeros_inplace` applies
    // after each shard).
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_zero.h5ad");
    // 2 rows × 2 cols. Each column has one zero entry and one
    // nonzero entry.
    write_csc_h5ad(
        &h5ad,
        2,
        2,
        &[0i64, 2, 4],
        &[0i32, 1, 0, 1],
        &[0.0f32, 7.0, 3.0, 0.0],
    );

    let scx = dir.path().join("out.scx");
    let opts = streaming_opts(8);
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    // After zero-drop: row 0 has col 1 (3.0); row 1 has col 0 (7.0)
    assert_eq!(csr.indptr, vec![0, 1, 2]);
    assert_eq!(csr.indices, vec![1, 0]);
    assert_eq!(csr.data, vec![3.0, 7.0]);
}

#[test]
fn phase2_streaming_csc_external_temp_cleanup_on_success() {
    // Open + drive the external transposer to completion, then
    // assert no `scx-transpose-*` directories remain under the
    // configured temp_dir.
    let dir = tempfile::tempdir().unwrap();
    let scratch = dir.path().join("scratch");
    std::fs::create_dir(&scratch).unwrap();

    let h5ad = dir.path().join("csc.h5ad");
    create_test_h5ad(&h5ad, 12, 4, "csc", false);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        memory_budget: Some(1024),
        temp_dir: Some(scratch.clone()),
        ..ConvertOptions::default()
    };
    h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    // After the transposer's `TempDir` drops, the session
    // directory under `scratch` should be gone.
    let leftover: Vec<_> = std::fs::read_dir(&scratch).unwrap().collect();
    assert!(
        leftover.is_empty(),
        "expected scratch to be empty after success; found {} entries",
        leftover.len()
    );
}

#[test]
fn phase2_streaming_csc_budget_too_small_actionable_error() {
    // memory_budget so small the external transposer can't fit even
    // 4 temp records. Must return an actionable error mentioning
    // "memory_budget".
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("csc_tiny_budget.h5ad");
    create_test_h5ad(&h5ad, 6, 3, "csc", false);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        memory_budget: Some(1),
        ..ConvertOptions::default()
    };
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect_err("budget=1 byte must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("memory_budget"),
        "error message should mention memory_budget; got: {msg}"
    );
}

#[test]
fn phase3_streaming_h5mu_round_trip_matches_bulk() {
    use crate::h5mu::pipeline::{h5mu_to_scx, h5mu_to_scx_streaming};
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("phase3.h5mu");
    create_test_h5mu(&h5mu, 20, 8, 4);

    let scx_stream = dir.path().join("stream.scx");
    let scx_bulk = dir.path().join("bulk.scx");
    let opts = streaming_opts(8);

    h5mu_to_scx_streaming(&h5mu, &scx_stream, &opts, &mut WarningSink::log()).unwrap();
    h5mu_to_scx(&h5mu, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.n_modalities(), b.n_modalities());
    assert_eq!(a.n_modalities(), 2);
    // Per-modality CSR shard payloads must match exactly.
    let n = a.n_modalities() as u8;
    let mut total_nnz: usize = 0;
    for mid in 1u8..=n {
        let csr_a = a.read_all_csr_shards_for(mid).unwrap();
        let csr_b = b.read_all_csr_shards_for(mid).unwrap();
        assert_eq!(
            csr_a.indptr, csr_b.indptr,
            "modality {mid} indptr divergence"
        );
        assert_eq!(
            csr_a.indices, csr_b.indices,
            "modality {mid} indices divergence"
        );
        assert_eq!(csr_a.data, csr_b.data, "modality {mid} data divergence");
        total_nnz += csr_b.data.len();
    }
    // The header `nnz` is reconstructed by `ScxWriter::finish()` (not the
    // pre-pass), so the streaming and bulk paths must agree on it, and it must
    // equal the cross-modality total. Pre-PR the two paths could silently
    // disagree here with no test noticing.
    assert_eq!(
        a.nnz(),
        b.nnz(),
        "header nnz divergence between streaming and bulk h5mu paths"
    );
    assert_eq!(
        b.nnz() as usize,
        total_nnz,
        "header nnz must equal summed per-modality CSR nnz"
    );
}

#[test]
fn h5mu_per_modality_layer_round_trips_bulk_and_streaming() {
    // Per-modality layers live at `mod/<modality>/layers/<name>` and must
    // round-trip under the correct modality on BOTH ingest paths:
    //
    // - Bulk: `read_layers_at`'s `read_layer_entry` must resolve the full
    //   parent path, not a hardcoded root `layers/<name>` (PR #239 fix).
    // - Streaming: the per-modality layer shards must be named
    //   `layer/{mname}/{layer_name}/shard_{idx}` to match what
    //   `layer_csr_shards_for_modality` (needle `/{layer_name}/`) and
    //   `layer_names_for` parse — the plain `{mname}_{layer_name}_shard`
    //   prefix left streaming-written layers unreadable via `read_layer_for`.
    use crate::h5mu::pipeline::{h5mu_to_scx, h5mu_to_scx_streaming};
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("layer.h5mu");

    let n_obs = 5usize;
    let rna_n_vars = 3usize;
    create_test_h5mu(&h5mu, n_obs, rna_n_vars, 2);

    // Known CSR layer matrix for the rna modality (n_obs × rna_n_vars).
    #[rustfmt::skip]
    let dense: Vec<f32> = vec![
        1.0, 0.0, 2.0,
        0.0, 3.0, 0.0,
        0.0, 0.0, 4.0,
        5.0, 6.0, 0.0,
        0.0, 7.0, 0.0,
    ];
    let mut indptr = vec![0i64];
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for row in 0..n_obs {
        for col in 0..rna_n_vars {
            let v = dense[row * rna_n_vars + col];
            if v != 0.0 {
                indices.push(col as i32);
                data.push(v);
            }
        }
        indptr.push(data.len() as i64);
    }

    // Attach `mod/rna/layers/spliced` (CSR) to the existing modality.
    {
        let file = hdf5::File::open_rw(&h5mu).unwrap();
        let rna = file.group("mod/rna").unwrap();
        let layers = rna.create_group("layers").unwrap();
        let lyr = layers.create_group("spliced").unwrap();
        lyr.new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        lyr.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        lyr.new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();
        lyr.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        lyr.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, rna_n_vars as i64])
            .unwrap();
    }

    let opts = streaming_opts(8);
    let scx_bulk = dir.path().join("bulk.scx");
    let scx_stream = dir.path().join("stream.scx");
    let mut sink_bulk = WarningSink::log();
    h5mu_to_scx(&h5mu, &scx_bulk, &opts, &mut sink_bulk).unwrap();
    let mut sink_stream = WarningSink::log();
    h5mu_to_scx_streaming(&h5mu, &scx_stream, &opts, &mut sink_stream).unwrap();

    for (label, scx, sink) in [
        ("bulk", &scx_bulk, &sink_bulk),
        ("stream", &scx_stream, &sink_stream),
    ] {
        assert_eq!(
            sink.counts().get("layer_skipped").copied().unwrap_or(0),
            0,
            "{label}: per-modality layer must be ingested, not skipped"
        );
        let reader = ScxReader::open(scx).unwrap();
        let rna_id = reader.modality_id("rna").expect("rna modality present");
        assert!(
            reader
                .layer_names_for(rna_id)
                .contains(&"spliced".to_string()),
            "{label}: layer_names_for must list the per-modality layer"
        );
        let csr = reader.read_layer_for(rna_id, "spliced").unwrap();
        assert_eq!(csr.shape, (n_obs, rna_n_vars), "{label}: layer shape");
        assert_eq!(
            csr.to_dense().unwrap(),
            dense,
            "{label}: per-modality layer must decode to the original matrix"
        );
    }
}

#[test]
fn phase3_streaming_h5mu_per_modality_codec_routing() {
    // Per-modality codec choices made by `select_codec_for_modality`
    // must match between the bulk and streaming paths. The streaming
    // path samples up to 16 KiB of values per modality before picking
    // — enough for any fixture small enough to fit in the bulk path
    // for comparison.
    use crate::h5mu::pipeline::{h5mu_to_scx, h5mu_to_scx_streaming};
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("codec.h5mu");
    create_test_h5mu(&h5mu, 20, 8, 4);

    let scx_stream = dir.path().join("stream.scx");
    let scx_bulk = dir.path().join("bulk.scx");
    let opts = streaming_opts(8);
    h5mu_to_scx_streaming(&h5mu, &scx_stream, &opts, &mut WarningSink::log()).unwrap();
    h5mu_to_scx(&h5mu, &scx_bulk, &opts, &mut WarningSink::log()).unwrap();

    let a = ScxReader::open(&scx_stream).unwrap();
    let b = ScxReader::open(&scx_bulk).unwrap();
    let ta = a.modality_table().expect("modality table present");
    let tb = b.modality_table().expect("modality table present");
    assert_eq!(ta.entries.len(), tb.entries.len());
    for (sa, sb) in ta.entries.iter().zip(tb.entries.iter()) {
        assert_eq!(
            sa.name, sb.name,
            "modality name divergence between streaming and bulk"
        );
        assert_eq!(
            sa.default_codec_id, sb.default_codec_id,
            "modality '{}' codec divergence: streaming={}, bulk={}",
            sa.name, sa.default_codec_id, sb.default_codec_id
        );
        assert_eq!(
            sa.default_value_encoding, sb.default_value_encoding,
            "modality '{}' value-encoding divergence",
            sa.name
        );
    }
}

#[test]
fn phase3_streaming_h5mu_dense_modality_non_f32_dtype() {
    // The streaming h5mu sampler reads the leading slab of any dense
    // modality `/X` to feed codec auto-selection. Before the fix it
    // hardcoded `read_slice_2d::<f32>`, which rejects non-f32 source
    // dtypes that the actual streaming reader supports. Write a
    // two-modality h5mu where the second modality's X is dense f64
    // (`encoding-type=array`) and confirm the streaming pipeline
    // converts it without erroring on the sample step.
    use crate::h5mu::pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("dense_f64.h5mu");
    create_test_h5mu_with_dense_f64_modality(&h5mu, 6, 4, 3);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        ..ConvertOptions::default()
    };
    h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut WarningSink::log()).unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.n_modalities(), 2);
    let dense_id = reader
        .modality_id("dense_adt")
        .expect("dense modality registered");
    let info = reader.modality_info(dense_id).unwrap();
    assert_eq!(info.n_vars, 3);
    let csr = reader.read_all_csr_shards_for(dense_id).unwrap();
    assert_eq!(csr.shape, (6, 3));
}

#[test]
fn phase3_streaming_h5mu_modality_filter() {
    use crate::h5mu::pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("filter.h5mu");
    create_test_h5mu(&h5mu, 8, 5, 3);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        modalities: Some(vec!["rna".to_string()]),
        ..ConvertOptions::default()
    };
    h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut WarningSink::log()).unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.n_modalities(), 1, "expected only 'rna' modality");
    let table = reader.modality_table().expect("modality table present");
    let names: Vec<&str> = table.entries.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, vec!["rna"]);
}

#[test]
fn phase3_streaming_h5mu_modality_filter_unknown_errors() {
    use crate::h5mu::pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("unknown.h5mu");
    create_test_h5mu(&h5mu, 6, 3, 2);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        modalities: Some(vec!["zzz".to_string()]),
        ..ConvertOptions::default()
    };
    let err = h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut WarningSink::log())
        .expect_err("unknown modality must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("zzz") && msg.contains("available"),
        "expected message to name 'zzz' and 'available'; got: {msg}"
    );
}

/// PR #155 follow-on: the streaming h5mu path cannot build per-modality
/// CSC, so an explicit `csc='always'` must be rejected (not silently
/// dropped) and the error must point users to the non-streaming path.
#[test]
fn streaming_h5mu_csc_always_rejected() {
    use super::pipeline::CscPolicy;
    use crate::h5mu::pipeline::h5mu_to_scx_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("cite.h5mu");
    create_test_h5mu(&h5mu, 8, 30, 5);

    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        csc: CscPolicy::Always,
        ..ConvertOptions::default()
    };
    let err = h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut WarningSink::log())
        .expect_err("csc='always' must be rejected on the streaming h5mu path");
    let msg = format!("{err}");
    assert!(
        msg.contains("stream=False"),
        "error must point users to stream=False; got: {msg}"
    );
}

/// `csc='auto'` on the streaming h5mu path degrades to CSR-only (best
/// effort) but must emit a `CscSkippedStreamingMultimodal` warning so the
/// drop is not silent. Both thresholds are zeroed so the tiny fixture
/// qualifies. Env is process-global, so this test is self-contained
/// (set → run → restore) and must not share these vars with other tests.
#[test]
fn streaming_h5mu_csc_auto_warns_and_skips() {
    use super::pipeline::CscPolicy;
    use crate::h5mu::pipeline::h5mu_to_scx_streaming;

    std::env::set_var("SCX_CSC_AUTO_OBS_THRESHOLD", "0");
    std::env::set_var("SCX_CSC_AUTO_VARS_THRESHOLD", "0");

    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("cite.h5mu");
    create_test_h5mu(&h5mu, 8, 30, 5);
    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        csc: CscPolicy::Auto,
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    let result = h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut sink);

    std::env::remove_var("SCX_CSC_AUTO_OBS_THRESHOLD");
    std::env::remove_var("SCX_CSC_AUTO_VARS_THRESHOLD");

    result.expect("csc='auto' must succeed (degrades to CSR-only)");
    assert!(
        sink.counts()
            .get("csc_skipped_streaming_multimodal")
            .copied()
            .unwrap_or(0)
            >= 1,
        "expected a csc_skipped_streaming_multimodal warning; counts: {:?}",
        sink.counts()
    );

    // No CSC sidecar was actually written.
    let reader = ScxReader::open(&scx).unwrap();
    assert!(!reader.header().has_csc(), "auto must not build CSC here");
    assert_eq!(reader.header().n_csc_shards, 0);
}

#[test]
fn phase3_streaming_h5mu_modality_types_override() {
    use crate::h5mu::pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("types.h5mu");
    create_test_h5mu(&h5mu, 8, 4, 2);

    let scx = dir.path().join("out.scx");
    // Override adt → Atac (a deliberately surprising mapping so we
    // can tell override actually took effect). The default heuristic
    // would map "adt" → Protein. No override for rna → inference +
    // ModalityTypeInferred warning emitted for rna only.
    let opts = ConvertOptions {
        shard_target_rows: 4,
        modality_types: vec![(
            "adt".to_string(),
            scx_format_io::modality::ModalityType::Atac,
        )],
        ..ConvertOptions::default()
    };
    let mut sink = WarningSink::log();
    h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut sink).unwrap();
    let reader = ScxReader::open(&scx).unwrap();
    let table = reader.modality_table().expect("modality table present");
    let adt = table.entries.iter().find(|m| m.name == "adt").unwrap();
    assert_eq!(
        adt.modality_type,
        scx_format_io::modality::ModalityType::Atac
    );
    let rna = table.entries.iter().find(|m| m.name == "rna").unwrap();
    assert_eq!(
        rna.modality_type,
        scx_format_io::modality::ModalityType::Rna
    );
    // rna had no override → one inferred-type warning. adt was
    // explicitly overridden → no inferred warning for it.
    let inferred = sink
        .counts()
        .get("modality_type_inferred")
        .copied()
        .unwrap_or(0);
    assert_eq!(
        inferred, 1,
        "expected exactly one ModalityTypeInferred (rna), got {inferred}"
    );
}

#[test]
fn phase3_streaming_h5mu_non_aligned_obs_errors() {
    use crate::h5mu::pipeline::h5mu_to_scx_streaming;
    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("misaligned.h5mu");
    // Build an h5mu by hand where outer obs has n_obs=8 but
    // /mod/rna/X.shape[0] = 12.
    {
        let file = hdf5::File::create(&h5mu).unwrap();
        let obs = file.create_group("obs").unwrap();
        let obs_index: Vec<VarLenUnicode> = (0..8).map(|i| vlu(&format!("cell_{i}"))).collect();
        obs.new_dataset::<VarLenUnicode>()
            .shape([8])
            .create("_index")
            .unwrap()
            .write(&obs_index)
            .unwrap();
        obs.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
        let mod_group = file.create_group("mod").unwrap();
        let rna = mod_group.create_group("rna").unwrap();
        let n_obs = 12usize;
        let n_vars = 3usize;
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for row in 0..n_obs {
            indices.push((row % n_vars) as i32);
            data.push((row + 1) as f32);
            indptr.push(data.len() as i64);
        }
        let x = rna.create_group("X").unwrap();
        x.new_dataset::<i64>()
            .shape([indptr.len()])
            .create("indptr")
            .unwrap()
            .write(&indptr)
            .unwrap();
        x.new_dataset::<i32>()
            .shape([indices.len()])
            .create("indices")
            .unwrap()
            .write(&indices)
            .unwrap();
        x.new_dataset::<f32>()
            .shape([data.len()])
            .create("data")
            .unwrap()
            .write(&data)
            .unwrap();
        x.new_attr::<VarLenUnicode>()
            .create("encoding-type")
            .unwrap()
            .write_scalar(&vlu("csr_matrix"))
            .unwrap();
        x.new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_obs as i64, n_vars as i64])
            .unwrap();
        let var = rna.create_group("var").unwrap();
        let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("g_{i}"))).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([n_vars])
            .create("_index")
            .unwrap()
            .write(&var_index)
            .unwrap();
        var.new_attr::<VarLenUnicode>()
            .create("_index")
            .unwrap()
            .write_scalar(&vlu("_index"))
            .unwrap();
    }

    let scx = dir.path().join("out.scx");
    let opts = streaming_opts(4);
    let err = h5mu_to_scx_streaming(&h5mu, &scx, &opts, &mut WarningSink::log())
        .expect_err("non-aligned modality obs must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("rna") && msg.contains("12") && msg.contains("8"),
        "expected message to name 'rna' and the offending counts; got: {msg}"
    );
}
