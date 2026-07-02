//! scx-convert integration tests — convert-time grouping.
//!
//! Exercises `ConvertOptions::group_by` / `--group-by`: the grouped
//! (reference-first, group-clustered) CSR layout is written directly during
//! conversion. The headline guarantee is **byte-equivalence to
//! convert-then-`scx sort --group-by`** at the order / roles / ranges level —
//! identical `group_index` sidecar bytes, identical X shard row ranges, and an
//! identical decoded matrix in the same row order.

use super::convert_tests_common::*;
use arrow::array::{Int32Array, StringArray};
use std::path::Path;

/// CSR/dense h5ad with an unsorted `cell_type` column (palette `A,B,A,B,A,…`)
/// plus the standard unique-per-row `n_counts` id. `A` doubles as the reference
/// label in these tests.
fn make_grouped_h5ad(path: &Path, n_obs: usize, n_vars: usize, fmt: &str) {
    create_test_h5ad(path, n_obs, n_vars, fmt, false);
    let file = hdf5::File::append(path).unwrap();
    let obs = file.group("obs").unwrap();
    let labels = ["A", "B", "A", "B", "A"];
    let col: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(labels[i % labels.len()])).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("cell_type")
        .unwrap()
        .write(&col)
        .unwrap();
}

/// Convert directly with `--group-by`. Sequential (`reader_threads
/// = Some(1)`) for deterministic comparison.
fn convert_grouped(
    h5ad: &Path,
    scx: &Path,
    group_by: &str,
    reference: Option<scx_ops::ReferenceSpec>,
    shard_target_rows: u32,
    group_target_bytes: Option<u64>,
    sink: &mut WarningSink,
) {
    let opts = ConvertOptions {
        group_by: Some(group_by.to_string()),
        reference,
        group_target_bytes,
        shard_target_rows,
        reader_threads: Some(1),
        ..Default::default()
    };
    h5ad_to_scx_streaming(h5ad, scx, &opts, &StreamingOverrides::default(), sink).unwrap();
}

/// Baseline: plain convert, then standalone `scx sort --group-by`.
fn convert_then_sort_grouped(
    h5ad: &Path,
    tmp: &Path,
    out: &Path,
    group_by: &str,
    reference: Option<scx_ops::ReferenceSpec>,
    shard_target_rows: u32,
    group_target_bytes: Option<u64>,
) {
    h5ad_to_scx_streaming(
        h5ad,
        tmp,
        &ConvertOptions {
            shard_target_rows,
            ..Default::default()
        },
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    let opts = scx_ops::SortOptions {
        by: vec![group_by.to_string()],
        group_by: Some(group_by.to_string()),
        reference,
        group_target_bytes,
        shard_target_rows,
        ..Default::default()
    };
    scx_ops::sort(tmp, out, &opts).unwrap();
}

fn dense_rows(csr: &scx_sparse::ScxCsr) -> Vec<Vec<f32>> {
    let (n, c) = csr.shape;
    let flat = csr.to_dense().unwrap();
    (0..n).map(|r| flat[r * c..(r + 1) * c].to_vec()).collect()
}

fn col_i32(batch: &arrow::array::RecordBatch, name: &str) -> Vec<i32> {
    let a = batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    (0..a.len()).map(|i| a.value(i)).collect()
}

fn col_str(batch: &arrow::array::RecordBatch, name: &str) -> Vec<String> {
    let col = batch.column_by_name(name).unwrap();
    let utf8 = arrow::compute::cast(col, &DataType::Utf8).unwrap();
    let a = utf8.as_any().downcast_ref::<StringArray>().unwrap();
    (0..a.len()).map(|i| a.value(i).to_string()).collect()
}

/// Raw bytes of the `group_index` sidecar section.
fn group_index_bytes(r: &ScxReader) -> Vec<u8> {
    let entry = r
        .catalog()
        .entries
        .iter()
        .find(|e| e.section_type == FmtSectionType::GroupIndex)
        .expect("group_index section present");
    r.section_bytes(entry).unwrap().to_vec()
}

/// `(row_start, row_end)` for every X CSR shard, in shard order.
fn x_shard_ranges(r: &ScxReader) -> Vec<(u64, u64)> {
    r.catalog()
        .shards_sorted()
        .iter()
        .map(|e| {
            let s = e.stats.as_ref().expect("shard stats");
            (s.row_start, s.row_end)
        })
        .collect()
}

/// Core acceptance: `convert --group-by` is byte-equivalent (order / roles /
/// ranges) to convert-then-`scx sort --group-by` — row-count mode.
#[test]
fn group_convert_matches_convert_then_sort_row_count() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let a = dir.path().join("grouped_convert.scx");
    let tmp = dir.path().join("plain.scx");
    let b = dir.path().join("grouped_sort.scx");
    let (n_obs, n_vars, shard) = (10usize, 8usize, 16u32);
    make_grouped_h5ad(&h5ad, n_obs, n_vars, "csr");

    let reference = Some(scx_ops::ReferenceSpec::Labels(vec!["A".to_string()]));
    let mut sink = WarningSink::log();
    convert_grouped(
        &h5ad,
        &a,
        "cell_type",
        reference.clone(),
        shard,
        None,
        &mut sink,
    );
    convert_then_sort_grouped(&h5ad, &tmp, &b, "cell_type", reference, shard, None);

    let ra = ScxReader::open(&a).unwrap();
    let rb = ScxReader::open(&b).unwrap();

    // 1. group_index sidecar bytes identical (group_by / records / roles /
    //    ranges / reference_labels / reference_shard).
    assert_eq!(
        group_index_bytes(&ra),
        group_index_bytes(&rb),
        "group_index bytes differ"
    );
    // 2. X shard row ranges identical (reference isolated in shard 0).
    let ranges = x_shard_ranges(&ra);
    assert_eq!(ranges, x_shard_ranges(&rb), "X shard ranges differ");
    // Palette A,B,A,B,A over 10 rows → A=6 (reference, shard 0), B=4 (shard 1).
    assert_eq!(ranges, vec![(0, 6), (6, 10)], "ref A=[0,6), group B=[6,10)");
    // 3. Decoded X identical in the same row order.
    assert_eq!(
        dense_rows(&ra.read_all_csr_shards().unwrap()),
        dense_rows(&rb.read_all_csr_shards().unwrap()),
        "decoded X differs"
    );
    // 4. obs row order identical (unique n_counts id) + grouped layout.
    let obs_a = ra.read_obs().unwrap();
    assert_eq!(
        col_i32(&obs_a, "n_counts"),
        col_i32(&rb.read_obs().unwrap(), "n_counts"),
        "obs row order differs"
    );
    let ct = col_str(&obs_a, "cell_type");
    assert_eq!(ct, vec!["A", "A", "A", "A", "A", "A", "B", "B", "B", "B"]);
}

/// Same equivalence under byte-budget grouped sharding (`group_target_bytes`),
/// CSR input. Also covers the deferred 7.6a byte-mode coverage end-to-end.
#[test]
fn group_convert_matches_convert_then_sort_byte_mode() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let a = dir.path().join("grouped_convert.scx");
    let tmp = dir.path().join("plain.scx");
    let b = dir.path().join("grouped_sort.scx");
    let (n_obs, n_vars, shard) = (10usize, 8usize, 16u32);
    make_grouped_h5ad(&h5ad, n_obs, n_vars, "csr");

    let reference = Some(scx_ops::ReferenceSpec::Labels(vec!["A".to_string()]));
    let budget = Some(64u64);
    let mut sink = WarningSink::log();
    convert_grouped(
        &h5ad,
        &a,
        "cell_type",
        reference.clone(),
        shard,
        budget,
        &mut sink,
    );
    convert_then_sort_grouped(&h5ad, &tmp, &b, "cell_type", reference, shard, budget);

    let ra = ScxReader::open(&a).unwrap();
    let rb = ScxReader::open(&b).unwrap();
    assert_eq!(group_index_bytes(&ra), group_index_bytes(&rb));
    assert_eq!(x_shard_ranges(&ra), x_shard_ranges(&rb));
    assert_eq!(
        dense_rows(&ra.read_all_csr_shards().unwrap()),
        dense_rows(&rb.read_all_csr_shards().unwrap())
    );
}

/// The grouped file opens through the engine and `read_group` / `read_reference`
/// partition the obs axis by label (end-to-end → F2 read path).
#[test]
fn group_convert_reads_back_per_label() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let a = dir.path().join("grouped.scx");
    make_grouped_h5ad(&h5ad, 10, 8, "csr");
    let mut sink = WarningSink::log();
    convert_grouped(
        &h5ad,
        &a,
        "cell_type",
        Some(scx_ops::ReferenceSpec::Labels(vec!["A".to_string()])),
        16,
        None,
        &mut sink,
    );

    let pipeline = scx_engine::QueryPipeline::open(&a).unwrap();
    let mut labels = pipeline.group_labels().unwrap();
    labels.sort();
    assert_eq!(labels, vec!["A".to_string(), "B".to_string()]);

    // `A` is the reference (isolated, read via read_reference); `B` is a group.
    let reference = pipeline
        .read_reference()
        .unwrap()
        .expect("reference present");
    assert_eq!(reference.x.shape.0, 6);
    assert!(col_str(&reference.obs, "cell_type")
        .iter()
        .all(|v| v == "A"));

    let group_b = pipeline.read_group("B").unwrap();
    assert_eq!(group_b.x.shape.0, 4);
    assert!(col_str(&group_b.obs, "cell_type").iter().all(|v| v == "B"));
}

/// `--reference` without `--group-by` is rejected.
#[test]
fn group_reference_requires_group_by_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    make_grouped_h5ad(&h5ad, 10, 8, "csr");
    let opts = ConvertOptions {
        reference: Some(scx_ops::ReferenceSpec::Labels(vec!["A".to_string()])),
        ..Default::default()
    };
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    );
    assert!(err.is_err(), "--reference without --group-by must error");
}

/// A CSC-on-disk X cannot be reordered → `--group-by` hard-errors.
#[test]
fn group_csc_input_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    make_grouped_h5ad(&h5ad, 10, 8, "csc");
    let opts = ConvertOptions {
        group_by: Some("cell_type".to_string()),
        ..Default::default()
    };
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    );
    assert!(err.is_err(), "CSC + --group-by must hard-error");
}

/// M2: the sequential grouped route (`reader_threads = 1`) buffers the largest
/// group whole, so it must refuse when a single group shard cannot fit
/// `memory_budget` — symmetric with the parallel path's derate. A tiny budget
/// hard-errors naming the grouped shard; an ample budget succeeds.
#[test]
fn group_convert_sequential_tiny_budget_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    make_grouped_h5ad(&h5ad, 10, 8, "csr");

    let base = |budget: u64| ConvertOptions {
        group_by: Some("cell_type".to_string()),
        reference: Some(scx_ops::ReferenceSpec::Labels(vec!["A".to_string()])),
        shard_target_rows: 16,
        reader_threads: Some(1),
        memory_budget: Some(budget),
        ..Default::default()
    };

    // 1-byte budget: any non-empty grouped shard exceeds it -> refuse.
    let scx_err = dir.path().join("err.scx");
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx_err,
        &base(1),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    );
    let msg = err
        .expect_err("tiny memory_budget must refuse the sequential grouped shard")
        .to_string();
    assert!(
        msg.contains("memory_budget") && msg.contains("grouped shard"),
        "error must name the grouped shard and the budget: {msg}"
    );

    // Ample budget: same sequential grouped convert succeeds.
    let scx_ok = dir.path().join("ok.scx");
    h5ad_to_scx_streaming(
        &h5ad,
        &scx_ok,
        &base(1 << 30),
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .expect("ample budget must convert");
    assert_eq!(ScxReader::open(&scx_ok).unwrap().header().n_obs, 10);
}

/// Forcing the one-pass path (`GroupPass::One`) on a dense input with
/// `--group-target-bytes` has no cheap per-row nnz, so it cannot honor the byte
/// budget. Rather than silently falling back to row-count sizing — which would
/// produce a *different* layout than the `Auto`/`Two` route for the same flags
/// — it hard-errors and points at `--group-pass two`.
#[test]
fn group_dense_byte_mode_one_pass_errors() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    make_grouped_h5ad(&h5ad, 10, 8, "dense");

    let mut sink = WarningSink::log();
    let opts = ConvertOptions {
        group_by: Some("cell_type".to_string()),
        reference: Some(scx_ops::ReferenceSpec::Labels(vec!["A".to_string()])),
        group_target_bytes: Some(64),
        shard_target_rows: 16,
        reader_threads: Some(1),
        group_pass: GroupPass::One, // exercise the one-pass dense path directly
        ..Default::default()
    };
    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut sink,
    );
    let msg = err
        .expect_err("dense + byte budget + one-pass must error")
        .to_string();
    assert!(
        msg.contains("group-pass two") && msg.contains("Dense"),
        "error must explain the byte-mode limitation and the fix: {msg}"
    );
}

/// `GroupPass::Auto` routes a **dense** source to the two-pass path (plain
/// convert + `scx sort`), which must produce the same grouped layout
/// (group_index bytes + shard ranges + decoded X) as forcing the one-pass
/// path — both equal convert-then-sort.
#[test]
fn group_dense_auto_two_pass_matches_one_pass() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let auto = dir.path().join("auto_twopass.scx");
    let forced = dir.path().join("forced_onepass.scx");
    make_grouped_h5ad(&h5ad, 10, 8, "dense");
    let reference = Some(scx_ops::ReferenceSpec::Labels(vec!["A".to_string()]));

    // Default Auto on dense → internal two-pass.
    let mut sink = WarningSink::log();
    let opts_auto = ConvertOptions {
        group_by: Some("cell_type".to_string()),
        reference: reference.clone(),
        shard_target_rows: 16,
        reader_threads: Some(1),
        ..Default::default()
    };
    h5ad_to_scx_streaming(
        &h5ad,
        &auto,
        &opts_auto,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .unwrap();

    // Forced one-pass (the dense random-row gather) for the same inputs.
    let opts_one = ConvertOptions {
        group_pass: GroupPass::One,
        ..opts_auto.clone()
    };
    h5ad_to_scx_streaming(
        &h5ad,
        &forced,
        &opts_one,
        &StreamingOverrides::default(),
        &mut sink,
    )
    .unwrap();

    let ra = ScxReader::open(&auto).unwrap();
    let rf = ScxReader::open(&forced).unwrap();
    assert_eq!(group_index_bytes(&ra), group_index_bytes(&rf));
    assert_eq!(x_shard_ranges(&ra), x_shard_ranges(&rf));
    assert_eq!(x_shard_ranges(&ra), vec![(0, 6), (6, 10)]);
    assert_eq!(
        dense_rows(&ra.read_all_csr_shards().unwrap()),
        dense_rows(&rf.read_all_csr_shards().unwrap())
    );
}
