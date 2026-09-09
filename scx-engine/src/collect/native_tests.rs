//! Tests for the native (dtype-selected) collect.
//!
//! The fixture here writes value bytes directly through
//! `ScxWriter::write_csr_shard` with an explicit `ValueEncoding`, which is the
//! only way in the tree to store an **odd** integer above 2²⁴: every Python
//! write door casts `X` through `f32` first, and an even value like `20_000_000`
//! survives that cast exactly. That matters because an f32-exact fixture cannot
//! tell a native decode from an f32 detour — a guard change alone would pass it.
//! `(1 << 24) + 7` cannot, and `the_fixture_is_discriminating` proves the
//! fixture has that property rather than assuming it.

use super::{truncate_to_limit, NativeShardRows};
use crate::pipeline::QueryPipeline;
use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ShardValuesNative, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_sparse::{Container, IndexDtype, MaterializePlan, ValueBuffer, ValueDtype};
use std::sync::Arc;

const BIG_ODD: u32 = (1 << 24) + 7; // 16_777_223

/// One value per row, one column per row (`col = row % n_vars`), so every
/// assertion can name a row and read one number back.
fn write_fixture(
    dir: &tempfile::TempDir,
    shards: &[(&[u32], ValueEncoding)],
    n_vars: usize,
) -> std::path::PathBuf {
    let n_obs: usize = shards.iter().map(|(v, _)| v.len()).sum();
    let path = dir.path().join("native.scx");
    let header =
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, n_obs as u64, 8, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs_schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("batch", DataType::Utf8, false),
    ]);
    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let batches: Vec<&str> = (0..n_obs)
        .map(|i| if i % 2 == 0 { "a" } else { "b" })
        .collect();
    writer
        .write_obs(
            &RecordBatch::try_new(
                Arc::new(obs_schema),
                vec![
                    Arc::new(StringArray::from(
                        ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    )),
                    Arc::new(StringArray::from(batches)),
                ],
            )
            .unwrap(),
        )
        .unwrap();

    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    writer
        .write_var(
            &RecordBatch::try_new(
                Arc::new(var_schema),
                vec![Arc::new(StringArray::from(
                    gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();

    let mut row_start = 0u64;
    for (vals, encoding) in shards {
        let indptr: Vec<u64> = (0..=vals.len() as u64).collect();
        let indices: Vec<u32> = (0..vals.len())
            .map(|r| ((row_start as usize + r) % n_vars) as u32)
            .collect();
        let bytes: Vec<u8> = match encoding {
            ValueEncoding::Uint32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ValueEncoding::Uint16 => vals
                .iter()
                .flat_map(|v| (*v as u16).to_le_bytes())
                .collect(),
            ValueEncoding::Float32 => vals
                .iter()
                .flat_map(|v| (*v as f32).to_le_bytes())
                .collect(),
            other => panic!("fixture does not write {other:?}"),
        };
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &bytes,
                CodecId::None,
                *encoding,
                row_start,
            )
            .unwrap();
        row_start += vals.len() as u64;
    }
    writer.finish().unwrap();
    path
}

fn plan(dtype: ValueDtype) -> MaterializePlan {
    MaterializePlan {
        container: Container::Csr,
        data_dtype: dtype,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    }
}

fn u32_values(buf: &ValueBuffer) -> Vec<u32> {
    match buf {
        ValueBuffer::U32(v) => v.clone(),
        other => panic!("expected a U32 buffer, got {:?}", other.dtype()),
    }
}

fn f64_values(buf: &ValueBuffer) -> Vec<f64> {
    match buf {
        ValueBuffer::F64(v) => v.clone(),
        other => panic!("expected an F64 buffer, got {:?}", other.dtype()),
    }
}

/// The premise every other test here rests on: this value does **not** survive
/// an f32 round trip, so a typed read that returns it exactly cannot have gone
/// through `f32`. Without this, an implementation that only moved the guard
/// would pass the suite.
#[test]
fn the_fixture_is_discriminating() {
    assert_ne!(BIG_ODD as f32 as u32, BIG_ODD);
    // Exactly between two representable f32s, so ties-to-even rounds *up*.
    assert_eq!(BIG_ODD as f32 as u32, BIG_ODD + 1);
    // And an even neighbour — the value every Python fixture can write — does
    // survive, which is why one is not a substitute for the other.
    assert_eq!(20_000_000f32 as u32, 20_000_000);
}

#[test]
fn an_odd_count_above_2pow24_reads_exactly_at_uint32() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(&dir, &[(&[3, BIG_ODD, 5, 7], ValueEncoding::Uint32)], 4);

    let typed = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("batch == 'b'")
        .unwrap()
        .collect_typed(&plan(ValueDtype::U32))
        .unwrap();

    // Rows 1 and 3 (batch "b"): the big value and 7.
    assert_eq!(typed.x.n_rows(), 2);
    assert_eq!(u32_values(&typed.x.values), vec![BIG_ODD, 7]);

    // The same query on the f32 route rounds it — the divergence this PR closes.
    let f32_result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("batch == 'b'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(f32_result.x.data[0] as u32, BIG_ODD + 1);
}

#[test]
fn an_odd_count_above_2pow24_reads_exactly_at_float64() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(&dir, &[(&[BIG_ODD], ValueEncoding::Uint32)], 2);

    let typed = QueryPipeline::open(&path)
        .unwrap()
        .collect_typed(&plan(ValueDtype::F64))
        .unwrap();
    assert_eq!(f64_values(&typed.x.values), vec![f64::from(BIG_ODD)]);
}

#[test]
fn a_target_that_cannot_hold_the_max_is_refused_before_decoding() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(&dir, &[(&[1, BIG_ODD], ValueEncoding::Uint32)], 2);

    let err = QueryPipeline::open(&path)
        .unwrap()
        .collect_typed(&plan(ValueDtype::U16))
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("65535"), "should name uint16's limit: {msg}");

    // `allow_lossy` is still the documented escape.
    let mut lossy = plan(ValueDtype::U16);
    lossy.allow_lossy = true;
    assert!(QueryPipeline::open(&path)
        .unwrap()
        .collect_typed(&lossy)
        .is_ok());
}

#[test]
fn a_big_value_past_the_limit_cutoff_does_not_refuse_a_narrow_read() {
    let dir = tempfile::tempdir().unwrap();
    // Shard 0 holds small counts; shard 1 holds the big one.
    let path = write_fixture(
        &dir,
        &[
            (&[1, 2, 3, 4], ValueEncoding::Uint32),
            (&[BIG_ODD, 6, 7, 8], ValueEncoding::Uint32),
        ],
        4,
    );

    // A limit satisfied by shard 0 never decodes shard 1, so its `value_max`
    // must not gate the read — the query guard is legitimately less
    // conservative than the eager path's catalog-wide fold.
    let typed = QueryPipeline::open(&path)
        .unwrap()
        .limit(2)
        .collect_typed(&plan(ValueDtype::U16))
        .unwrap();
    assert_eq!(u32_values_as_u16(&typed.x.values), vec![1, 2]);

    // Without the limit the same request is refused.
    assert!(QueryPipeline::open(&path)
        .unwrap()
        .collect_typed(&plan(ValueDtype::U16))
        .is_err());
}

fn u32_values_as_u16(buf: &ValueBuffer) -> Vec<u16> {
    match buf {
        ValueBuffer::U16(v) => v.clone(),
        other => panic!("expected a U16 buffer, got {:?}", other.dtype()),
    }
}

#[test]
fn a_file_mixing_integer_and_float_shards_assembles_at_one_dtype() {
    let dir = tempfile::tempdir().unwrap();
    // Uniformity is guaranteed only *within* a shard, so the merge must pick
    // the source arm per shard rather than once for the file.
    let path = write_fixture(
        &dir,
        &[
            (&[10, 20], ValueEncoding::Uint16),
            (&[30, 40], ValueEncoding::Float32),
        ],
        2,
    );

    let typed = QueryPipeline::open(&path)
        .unwrap()
        .collect_typed(&plan(ValueDtype::F64))
        .unwrap();
    assert_eq!(f64_values(&typed.x.values), vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn a_typed_collect_matches_the_f32_collect_where_f32_is_exact() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(&dir, &[(&[1, 2, 3, 4, 5, 6], ValueEncoding::Uint16)], 3);

    for expr in ["batch == 'a'", "batch == 'b'"] {
        let f32_result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs(expr)
            .unwrap()
            .collect()
            .unwrap();
        let typed = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs(expr)
            .unwrap()
            .collect_typed(&plan(ValueDtype::F64))
            .unwrap();

        assert_eq!(typed.x.shape, f32_result.x.shape, "{expr}");
        assert_eq!(typed.x.indptr, f32_result.x.indptr, "{expr}");
        assert_eq!(typed.x.n_rows(), f32_result.x.n_rows(), "{expr}");
        assert_eq!(
            f64_values(&typed.x.values),
            f32_result
                .x
                .data
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
            "{expr}"
        );
        assert_eq!(typed.obs.num_rows(), f32_result.obs.num_rows(), "{expr}");
        assert_eq!(typed.matched_rows, f32_result.matched_rows, "{expr}");
    }
}

#[test]
fn a_limit_truncates_the_typed_result_like_the_f32_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(
        &dir,
        &[
            (&[1, 2, 3, 4], ValueEncoding::Uint16),
            (&[5, 6, 7, 8], ValueEncoding::Uint16),
        ],
        4,
    );

    for limit in [1usize, 3, 5, 8, 99] {
        let f32_result = QueryPipeline::open(&path)
            .unwrap()
            .limit(limit)
            .collect()
            .unwrap();
        let typed = QueryPipeline::open(&path)
            .unwrap()
            .limit(limit)
            .collect_typed(&plan(ValueDtype::F64))
            .unwrap();
        assert_eq!(typed.x.n_rows(), f32_result.x.n_rows(), "limit {limit}");
        assert_eq!(typed.x.indptr, f32_result.x.indptr, "limit {limit}");
        assert_eq!(
            f64_values(&typed.x.values),
            f32_result
                .x
                .data
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
            "limit {limit}"
        );
        assert_eq!(
            typed.obs.num_rows(),
            f32_result.obs.num_rows(),
            "limit {limit}"
        );
    }
}

#[test]
fn select_genes_projects_and_reorders_the_typed_result() {
    let dir = tempfile::tempdir().unwrap();
    // Row r has its single value in column r % 4.
    let path = write_fixture(&dir, &[(&[10, 20, 30, 40], ValueEncoding::Uint16)], 4);

    for genes in [vec![1u32, 3], vec![3u32, 1]] {
        let f32_result = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(genes.clone())
            .collect()
            .unwrap();
        let typed = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(genes.clone())
            .collect_typed(&plan(ValueDtype::F64))
            .unwrap();

        assert_eq!(typed.x.shape, f32_result.x.shape, "{genes:?}");
        assert_eq!(typed.x.indptr, f32_result.x.indptr, "{genes:?}");
        assert_eq!(
            index_values(&typed.x.indices),
            f32_result.x.indices,
            "column relabel must match the f32 path for {genes:?}"
        );
        assert_eq!(
            f64_values(&typed.x.values),
            f32_result
                .x
                .data
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<_>>(),
            "{genes:?}"
        );
        assert_eq!(typed.var.num_rows(), f32_result.var.num_rows(), "{genes:?}");
    }
}

fn index_values(buf: &scx_sparse::IndexBuffer) -> Vec<i32> {
    match buf {
        scx_sparse::IndexBuffer::I32(v) => v.clone(),
        other => panic!("expected an I32 index buffer, got {:?}", other.dtype()),
    }
}

#[test]
fn an_empty_typed_result_keeps_its_shape_and_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(&dir, &[(&[1, 2], ValueEncoding::Uint16)], 3);

    let typed = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("batch == 'nobody'")
        .unwrap()
        .collect_typed(&plan(ValueDtype::F64))
        .unwrap();

    assert_eq!(typed.x.shape, (0, 3));
    assert_eq!(typed.x.indptr, vec![0]);
    assert_eq!(typed.x.nnz(), 0);
    assert_eq!(typed.x.values.dtype(), ValueDtype::F64);
    assert_eq!(typed.obs.num_rows(), 0);
}

/// The typed all-kept skip returns the decoded buffers **by move**.
///
/// `filter_owned_keeps_all_rows_by_moving_them` pins that property for the f32
/// route. The typed route had no analog and could not have one while the skip
/// was inlined in `decode_shard_native_filtered`: the buffers it hands back are
/// allocated inside that function, so no caller can tell a move from a faithful
/// copy — and every other test in this file compares values, which a copy
/// preserves. Deleting the skip left them all green. The decision now lives in
/// an owned helper, which is callable with buffers whose addresses the test
/// knows.
#[test]
fn the_typed_skip_returns_the_decoded_buffers_by_moving_them() {
    use super::filter_project_rows_owned;

    let indptr = vec![0i64, 2, 3, 5];
    let indices = vec![0u32, 3, 1, 2, 3];
    let values = vec![10u32, 11, 12, 13, 14];
    let (ip_ptr, ix_ptr, v_ptr) = (indptr.as_ptr(), indices.as_ptr(), values.as_ptr());

    let (ip, ix, v) = filter_project_rows_owned(indptr, indices, values, &[true; 3], None).unwrap();
    assert_eq!(ip.as_ptr(), ip_ptr, "indptr was copied, not moved");
    assert_eq!(ix.as_ptr(), ix_ptr, "indices were copied, not moved");
    assert_eq!(v.as_ptr(), v_ptr, "values were copied, not moved");
    assert_eq!(ip, vec![0, 2, 3, 5]);
    assert_eq!(ix, vec![0, 3, 1, 2, 3]);
    assert_eq!(v, vec![10, 11, 12, 13, 14]);
}

/// The owned helper rejects a short all-true mask **itself**.
///
/// A `debug_assert` was not enough: `all()` over a mask shorter than the CSR is
/// vacuously true, so a release build would take the fast path and hand back
/// every row of a shard the caller believes it truncated. The production caller
/// checking first does not stop a later one from forgetting, which is why the
/// f32 analog puts `check_keep_mask_len` inside the owned function too.
#[test]
fn the_typed_owned_filter_rejects_a_short_all_true_mask() {
    use super::filter_project_rows_owned;

    let indptr = vec![0i64, 2, 3, 5];
    let indices = vec![0u32, 3, 1, 2, 3];
    let values = vec![10u32, 11, 12, 13, 14];

    let err = filter_project_rows_owned(
        indptr.clone(),
        indices.clone(),
        values.clone(),
        &[true, true],
        None,
    )
    .expect_err("a 2-entry mask must not be accepted for a 3-row shard")
    .to_string();
    assert!(err.contains("keep mask covers 2 rows"), "unexpected: {err}");

    assert!(
        filter_project_rows_owned(indptr, indices, values, &[true; 4], None).is_err(),
        "a 4-entry mask must not be accepted for a 3-row shard either"
    );
}

/// …and does not fire when either half of its precondition fails.
///
/// The premises the move rests on, separately: a projection is present, or a
/// row is dropped. Either way the buffers must be rebuilt — a skip that ignored
/// `gene_set` would return unprojected columns, and one that ignored the mask
/// would return rows the caller filtered out.
#[test]
fn the_typed_skip_does_not_fire_under_a_projection_or_a_partial_mask() {
    use super::filter_project_rows_owned;

    let indptr = vec![0i64, 2, 3, 5];
    let indices = vec![0u32, 3, 1, 2, 3];
    let values = vec![10u32, 11, 12, 13, 14];

    // A projection: columns are remapped to `0..gene_set.len()`.
    let (ip, ix, v) = filter_project_rows_owned(
        indptr.clone(),
        indices.clone(),
        values.clone(),
        &[true; 3],
        Some(&[1u32, 3]),
    )
    .unwrap();
    assert_eq!(ip, vec![0, 1, 2, 3], "one kept column per row");
    assert_eq!(ix, vec![1, 0, 1], "gene 3 -> col 1, gene 1 -> col 0");
    assert_eq!(v, vec![11, 12, 14]);

    // A dropped row.
    let (ip, ix, v) =
        filter_project_rows_owned(indptr, indices, values, &[true, false, true], None).unwrap();
    assert_eq!(ip, vec![0, 2, 4]);
    assert_eq!(ix, vec![0, 3, 2, 3]);
    assert_eq!(v, vec![10, 11, 13, 14]);
}

/// White-box: the limit trim drops whole shards past the cutoff rather than
/// keeping empty ones, so the merge's row/nnz totals match what it allocates.
#[test]
fn truncate_to_limit_drops_shards_past_the_cutoff() {
    let shard = |rows: usize| NativeShardRows {
        indptr: (0..=rows as i64).collect(),
        indices: vec![0u32; rows],
        values: ShardValuesNative::U32(vec![7u32; rows]),
    };

    for (limit, want_shards, want_rows) in [
        (0usize, 0usize, 0usize),
        (1, 1, 1),
        (2, 1, 2),
        (3, 2, 3),
        (4, 2, 4),
        (9, 2, 4),
    ] {
        let mut results = vec![shard(2), shard(2)];
        truncate_to_limit(&mut results, limit);
        assert_eq!(results.len(), want_shards, "limit {limit}");
        let rows: usize = results.iter().map(NativeShardRows::n_rows).sum();
        assert_eq!(rows, want_rows, "limit {limit}");
        // Every surviving shard stays self-consistent: nnz matches its indptr.
        for r in &results {
            assert_eq!(r.nnz(), *r.indptr.last().unwrap() as usize, "limit {limit}");
        }
    }
}

#[test]
fn fused_transforms_are_refused_rather_than_served_from_the_f32_route() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(&dir, &[(&[1, 2], ValueEncoding::Uint16)], 2);
    let mplan = plan(ValueDtype::F64);

    for pipeline in [
        QueryPipeline::open(&path).unwrap().with_normalize(1e4),
        QueryPipeline::open(&path).unwrap().with_log1p(),
    ] {
        assert!(!pipeline.typed_collect_supported());
        let err = pipeline.collect_typed(&mplan).unwrap_err().to_string();
        assert!(
            err.contains("with_normalize") && err.contains("with_log1p"),
            "the refusal should name both transforms: {err}"
        );
    }

    // A dense request, by contrast, does **not** disqualify the typed decode —
    // the predicate does not even take a plan any more. The container is
    // presentation applied after the decode, and gating the decode on it sent
    // `to_anndata(obs_filter=…, container="dense", data_dtype=…)` back to the
    // f32 route and refused the read this exists to serve.
    let mut dense = plan(ValueDtype::F64);
    dense.container = Container::Dense;
    assert!(QueryPipeline::open(&path)
        .unwrap()
        .typed_collect_supported());
    // And it assembles a CSR regardless, which is what makes the scatter the
    // caller's step.
    let typed = QueryPipeline::open(&path)
        .unwrap()
        .collect_typed(&dense)
        .unwrap();
    assert_eq!(typed.x.values.dtype(), ValueDtype::F64);
    assert_eq!(typed.x.n_cols(), 2);
}
