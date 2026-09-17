//! Tests for the shared mapping-shard emitters.
//!
//! The load-bearing one is [`coo_preserves_every_dtype_the_ops_can_produce`]:
//! it is the test that would have caught reusing either of the two bucketers
//! this module replaced, both of which hardcoded `Float32` values (and one of
//! which hardcoded `Int32` coordinates), against a `compact` that preserves
//! whatever its input had.

use super::*;
use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;

/// `(shard_idx, row_start, n_shard_rows, n_rows_total)` for each emitted shard.
type Stamps = Vec<(u32, u64, u64, u64)>;

fn dense(n_rows: usize, n_cols: usize) -> RecordBatch {
    let fields: Vec<Field> = (0..n_cols)
        .map(|c| Field::new(format!("c{c}"), DataType::Float32, false))
        .collect();
    let cols: Vec<Arc<dyn Array>> = (0..n_cols)
        .map(|c| {
            Arc::new(Float32Array::from(
                (0..n_rows)
                    .map(|r| (r * n_cols + c) as f32)
                    .collect::<Vec<_>>(),
            )) as Arc<dyn Array>
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
}

/// A COO batch with caller-chosen coordinate width, value width and value
/// nullability — the product `compact::remap_obsp_coo_to_dim` can produce.
fn coo(
    n_rows: usize,
    triples: &[(i64, i64, f64)],
    wide_coords: bool,
    wide_values: bool,
    nullable_values: bool,
) -> RecordBatch {
    let coord_dt = if wide_coords {
        DataType::Int64
    } else {
        DataType::Int32
    };
    let value_dt = if wide_values {
        DataType::Float64
    } else {
        DataType::Float32
    };
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", coord_dt.clone(), false),
            Field::new("col", coord_dt, false),
            Field::new("data", value_dt, nullable_values),
        ],
        HashMap::from([
            ("n_rows".to_string(), n_rows.to_string()),
            ("n_cols".to_string(), n_rows.to_string()),
        ]),
    ));
    let (rows, cols): (Vec<i64>, Vec<i64>) = triples.iter().map(|t| (t.0, t.1)).unzip();
    let (row_arr, col_arr): (Arc<dyn Array>, Arc<dyn Array>) = if wide_coords {
        (
            Arc::new(Int64Array::from(rows)),
            Arc::new(Int64Array::from(cols)),
        )
    } else {
        (
            Arc::new(Int32Array::from(
                rows.into_iter().map(|v| v as i32).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                cols.into_iter().map(|v| v as i32).collect::<Vec<_>>(),
            )),
        )
    };
    // On the nullable arm the LAST value is actually null. Flipping the field's
    // nullable bit alone proves nothing: `Float32Array::from(Vec<f32>)` has
    // `null_count == 0`, so schema equality would still hold if `take` dropped
    // the null bitmap. The bitmap has to be there to be preserved.
    let mut vals: Vec<Option<f64>> = triples.iter().map(|t| Some(t.2)).collect();
    if nullable_values {
        if let Some(last) = vals.last_mut() {
            *last = None;
        }
    }
    let val_arr: Arc<dyn Array> = if wide_values {
        Arc::new(Float64Array::from(vals))
    } else {
        Arc::new(Float32Array::from(
            vals.into_iter()
                .map(|v| v.map(|x| x as f32))
                .collect::<Vec<_>>(),
        ))
    };
    RecordBatch::try_new(schema, vec![row_arr, col_arr, val_arr]).unwrap()
}

/// Run the COO emitter, returning the stamps and the emitted shards.
fn run_coo(batch: &RecordBatch, target: u32) -> scx_format::Result<(Stamps, Vec<RecordBatch>)> {
    let (mut stamps, mut shards) = (Stamps::new(), Vec::new());
    for_each_coo_mapping_shard("obsp/g", batch, target, |m, s| {
        stamps.push((m.shard_idx, m.row_start, m.n_shard_rows, m.n_rows_total));
        shards.push(s.clone());
        Ok(())
    })?;
    Ok((stamps, shards))
}

fn run_dense(batch: &RecordBatch, target: u32) -> (Stamps, Vec<RecordBatch>) {
    let (mut stamps, mut shards) = (Stamps::new(), Vec::new());
    for_each_dense_mapping_shard(batch, target, |m, s| {
        stamps.push((m.shard_idx, m.row_start, m.n_shard_rows, m.n_rows_total));
        shards.push(s.clone());
        Ok(())
    })
    .unwrap();
    (stamps, shards)
}

/// The triples a shard carries, as a sortable multiset.
fn triples_of(b: &RecordBatch) -> Vec<(i64, i64, String)> {
    let rows = crate::backed::coo_coord_borrow(b, 0, "obsp/g").unwrap();
    let cols = crate::backed::coo_coord_borrow(b, 1, "obsp/g").unwrap();
    let data = b.column(2);
    (0..b.num_rows())
        .map(|i| {
            // Formatted rather than cast so an f64 value is compared at its
            // own width; a lossy narrow here would hide a lossy narrow there.
            let v = if data.is_null(i) {
                "null".to_string()
            } else if let Some(a) = data.as_any().downcast_ref::<Float64Array>() {
                format!("{:?}", a.value(i))
            } else {
                let a = data.as_any().downcast_ref::<Float32Array>().unwrap();
                format!("{:?}", a.value(i))
            };
            (rows.at(i), cols.at(i), v)
        })
        .collect()
}

fn sorted_triples(bs: &[RecordBatch]) -> Vec<(i64, i64, String)> {
    let mut all: Vec<_> = bs.iter().flat_map(triples_of).collect();
    all.sort();
    all
}

/// The cover contract, asserted directly rather than via a file round trip:
/// `shard_idx` is the position, spans are contiguous from 0, and the last
/// span ends exactly at `n_rows_total`.
fn assert_cover(stamps: &Stamps, expect_total: u64) {
    assert!(!stamps.is_empty(), "no shards emitted");
    let mut next = 0u64;
    for (pos, &(idx, row_start, n, total)) in stamps.iter().enumerate() {
        assert_eq!(idx as usize, pos, "shard_idx must equal emission position");
        assert_eq!(row_start, next, "shard {pos} leaves a gap in the cover");
        assert_eq!(total, expect_total, "shard {pos} stamps the wrong total");
        next += n;
    }
    assert_eq!(next, expect_total, "cover stops short of n_rows_total");
}

// ---------------------------------------------------------------- dense

#[test]
fn dense_splits_at_the_target_and_covers_every_row() {
    let b = dense(10, 3);
    let (stamps, shards) = run_dense(&b, 4);
    assert_eq!(
        stamps,
        vec![(0, 0, 4, 10), (1, 4, 4, 10), (2, 8, 2, 10)],
        "10 rows at a target of 4 is 4 + 4 + 2"
    );
    assert_cover(&stamps, 10);
    assert_eq!(
        shards.iter().map(|s| s.num_rows()).sum::<usize>(),
        10,
        "every row lands in exactly one shard"
    );
    // Dense slicing is order-preserving, so concatenation is row-for-row equal.
    let rejoined = arrow::compute::concat_batches(&b.schema(), &shards).unwrap();
    assert_eq!(rejoined, b, "concatenated dense shards equal the input");
}

#[test]
fn dense_below_the_target_is_one_shard() {
    let (stamps, shards) = run_dense(&dense(3, 2), 16384);
    assert_eq!(stamps, vec![(0, 0, 3, 3)]);
    assert_eq!(shards.len(), 1);
}

#[test]
fn dense_zero_rows_still_emits_one_shard() {
    // Otherwise a key that exists but covers no rows disappears from the
    // catalog on rewrite, which reads as "the op dropped it".
    let (stamps, shards) = run_dense(&dense(0, 2), 4);
    assert_eq!(stamps, vec![(0, 0, 0, 0)]);
    assert_eq!(shards[0].num_rows(), 0);
}

// ------------------------------------------------------------------ COO

#[test]
fn coo_buckets_by_row_over_the_target() {
    let t = [(0, 1, 1.0), (3, 0, 2.0), (4, 5, 3.0), (7, 7, 4.0)];
    let (stamps, shards) = run_coo(&coo(8, &t, false, false, false), 4).unwrap();
    assert_eq!(stamps, vec![(0, 0, 4, 8), (1, 4, 4, 8)]);
    assert_cover(&stamps, 8);
    assert_eq!(shards[0].num_rows(), 2, "rows 0 and 3 land in shard 0");
    assert_eq!(shards[1].num_rows(), 2, "rows 4 and 7 land in shard 1");
}

#[test]
fn coo_preserves_every_dtype_the_ops_can_produce() {
    // `compact::remap_obsp_coo_to_dim` preserves its input's value dtype and
    // nullability and picks the coordinate width from the surviving extent, so
    // the emitter sees this whole product. Both bucketers this module replaced
    // hardcoded Float32 values; one also hardcoded Int32 coordinates.
    let t = [(0, 1, 1.5), (5, 2, 2.5), (9, 9, 3.5)];
    for &wide_coords in &[false, true] {
        for &wide_values in &[false, true] {
            for &nullable in &[false, true] {
                let b = coo(12, &t, wide_coords, wide_values, nullable);
                let (stamps, shards) = run_coo(&b, 4).unwrap_or_else(|e| {
                    panic!("coords_i64={wide_coords} values_f64={wide_values} nullable={nullable}: {e}")
                });
                assert_cover(&stamps, 12);
                for (i, s) in shards.iter().enumerate() {
                    assert_eq!(
                        s.schema().fields(),
                        b.schema().fields(),
                        "shard {i} changed the field schema \
                         (coords_i64={wide_coords} values_f64={wide_values} nullable={nullable})"
                    );
                }
                assert_eq!(
                    sorted_triples(&shards),
                    sorted_triples(std::slice::from_ref(&b)),
                    "triples changed (coords_i64={wide_coords} values_f64={wide_values} nullable={nullable})"
                );
                // The null bitmap itself, not just the field's nullable bit.
                let nulls: usize = shards.iter().map(|s| s.column(2).null_count()).sum();
                assert_eq!(
                    nulls,
                    b.column(2).null_count(),
                    "null count changed (coords_i64={wide_coords} values_f64={wide_values} nullable={nullable})"
                );
                if nullable {
                    assert_eq!(nulls, 1, "the nullable arm must actually carry a null");
                }
            }
        }
    }
}

#[test]
fn coo_keeps_the_n_rows_and_n_cols_declaration_on_every_shard() {
    // The reader takes the matrix's column extent from shard 0's `n_cols` and
    // cross-checks the rest against it, so losing the key on any shard is a
    // file `BackedPairwiseReader::new_obsp` refuses to open.
    let (_, shards) =
        run_coo(&coo(8, &[(0, 1, 1.0), (6, 2, 2.0)], false, false, false), 4).unwrap();
    for (i, s) in shards.iter().enumerate() {
        let md = s.schema_ref().metadata().clone();
        assert_eq!(md.get("n_rows").map(String::as_str), Some("8"), "shard {i}");
        assert_eq!(md.get("n_cols").map(String::as_str), Some("8"), "shard {i}");
    }
}

#[test]
fn coo_input_order_does_not_decide_the_bucket() {
    // The emitter is a counting sort, not a scan: `scx-convert`'s override
    // path and `compact`'s remap both hand over triples in input order, which
    // is row-sorted only by accident.
    let ascending = [(0, 0, 1.0), (2, 0, 2.0), (5, 0, 3.0), (7, 0, 4.0)];
    let shuffled = [(7, 0, 4.0), (2, 0, 2.0), (0, 0, 1.0), (5, 0, 3.0)];
    let (sa, ba) = run_coo(&coo(8, &ascending, false, false, false), 4).unwrap();
    let (ss, bs) = run_coo(&coo(8, &shuffled, false, false, false), 4).unwrap();
    assert_eq!(sa, ss, "stamps must not depend on input order");
    assert_eq!(
        ba.iter().map(|b| b.num_rows()).collect::<Vec<_>>(),
        bs.iter().map(|b| b.num_rows()).collect::<Vec<_>>(),
        "a triple's bucket is decided by its row, not by where it appeared"
    );
    assert_eq!(sorted_triples(&ba), sorted_triples(&bs));
}

#[test]
fn coo_emits_empty_buckets_so_the_cover_has_no_gaps() {
    // Rows 4..8 carry nothing. Skipping that shard would make `row_start`
    // jump, and the reader's cover validator rejects the file — so the empty
    // shard is required, not merely harmless.
    let (stamps, shards) = run_coo(
        &coo(12, &[(0, 0, 1.0), (9, 9, 2.0)], false, false, false),
        4,
    )
    .unwrap();
    assert_eq!(stamps, vec![(0, 0, 4, 12), (1, 4, 4, 12), (2, 8, 4, 12)]);
    assert_cover(&stamps, 12);
    assert_eq!(shards[1].num_rows(), 0, "the empty band still gets a shard");
}

#[test]
fn coo_zero_rows_with_no_triples_is_one_empty_shard() {
    let (stamps, shards) = run_coo(&coo(0, &[], false, false, false), 4).unwrap();
    assert_eq!(stamps, vec![(0, 0, 0, 0)]);
    assert_eq!(shards[0].num_rows(), 0);
}

#[test]
fn coo_zero_rows_carrying_triples_is_rejected() {
    // Writing this as "one shard covering nothing" validates as a cover and
    // then hides the triples from every bounded read. Refuse it instead.
    let err = run_coo(&coo(0, &[(0, 0, 1.0)], false, false, false), 4).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("n_rows=0"), "{msg}");
    assert!(msg.contains("1 triples"), "{msg}");
}

#[test]
fn coo_rejects_a_row_outside_the_declared_extent() {
    let err = run_coo(&coo(4, &[(0, 0, 1.0), (9, 0, 2.0)], false, false, false), 4).unwrap_err();
    assert!(
        err.to_string().contains("outside the declared n_rows=4"),
        "{err}"
    );
}

#[test]
fn coo_rejects_a_row_in_the_last_shards_slack() {
    // The case the test above **cannot** see, because `n_rows == step` leaves no
    // slack. When `n_rows` is not a multiple of the target, the last bucket's
    // arithmetic range runs to `n_shards * step` while its stamp stops at
    // `n_rows`, so a bounds check written against the derived *bucket* rather
    // than the row admits every coordinate in between.
    //
    // n_rows=5, step=4 -> n_shards=2, so rows 5, 6 and 7 all divide into bucket
    // 1 and pass a `shard < n_shards` test, then land in a shard stamped [4, 5).
    for bad in [5i64, 6, 7] {
        let err = run_coo(
            &coo(5, &[(0, 0, 1.0), (bad, 0, 2.0)], false, false, false),
            4,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("outside the declared n_rows=5"),
            "row {bad} was accepted into the last shard's slack: {err}"
        );
    }
    // And the one-row-of-slack shape, where only the single value n_rows is bad.
    let err = run_coo(&coo(5, &[(4, 0, 1.0), (5, 0, 2.0)], false, false, false), 2).unwrap_err();
    assert!(
        err.to_string().contains("outside the declared n_rows=5"),
        "{err}"
    );
}

#[test]
fn coo_requires_the_extent_declaration() {
    // Without `n_rows` the logical extent is unknowable: the batch's own row
    // count is nnz. Name the missing key rather than guessing.
    let b = coo(8, &[(0, 0, 1.0)], false, false, false);
    let bare = RecordBatch::try_new(
        Arc::new(Schema::new(b.schema().fields().clone())),
        b.columns().to_vec(),
    )
    .unwrap();
    let err = run_coo(&bare, 4).unwrap_err();
    assert!(
        err.to_string().contains("no 'n_rows' schema metadata"),
        "{err}"
    );
}

#[test]
fn coo_refuses_a_bucket_table_it_cannot_afford() {
    // The table is sized by the row axis, not nnz, so a wide-axis v2 graph at
    // a small target would allocate before reading a triple.
    let b = coo(0, &[], false, false, false);
    let wide = RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            b.schema().fields().clone(),
            HashMap::from([
                ("n_rows".to_string(), (MAX_BUCKETS * 4).to_string()),
                ("n_cols".to_string(), (MAX_BUCKETS * 4).to_string()),
            ]),
        )),
        b.columns().to_vec(),
    )
    .unwrap();
    let err = run_coo(&wide, 1).unwrap_err();
    assert!(err.to_string().contains("shard buckets"), "{err}");
}

#[test]
fn coo_rejects_a_schema_its_own_reader_would_refuse() {
    // The emitter is `pub`, so a caller can hand it anything. Writing a file
    // that `BackedPairwiseReader::from_layout` then refuses to open is worse
    // than refusing the write: the failure surfaces later, elsewhere.
    let b = coo(8, &[(0, 1, 1.0)], false, false, false);

    // A fourth column.
    let mut fields: Vec<Field> = b.schema().fields().iter().map(|f| (**f).clone()).collect();
    fields.push(Field::new("extra", DataType::Float32, false));
    let mut cols = b.columns().to_vec();
    cols.push(Arc::new(Float32Array::from(vec![0.0f32])) as Arc<dyn Array>);
    let wide = RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            fields,
            b.schema_ref().metadata().clone(),
        )),
        cols,
    )
    .unwrap();
    let err = run_coo(&wide, 4).unwrap_err();
    assert!(err.to_string().contains("expected exactly"), "{err}");

    // Right count, wrong names.
    let renamed = RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("i", DataType::Int32, false),
                Field::new("j", DataType::Int32, false),
                Field::new("v", DataType::Float32, false),
            ],
            b.schema_ref().metadata().clone(),
        )),
        b.columns().to_vec(),
    )
    .unwrap();
    let err = run_coo(&renamed, 4).unwrap_err();
    assert!(err.to_string().contains("expected exactly"), "{err}");

    // Coordinate columns at different widths — accepted before, and a file
    // `coo_coord_borrow` would then read inconsistently.
    let mixed = RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("row", DataType::Int32, false),
                Field::new("col", DataType::Int64, false),
                Field::new("data", DataType::Float32, false),
            ],
            b.schema_ref().metadata().clone(),
        )),
        vec![
            b.column(0).clone(),
            Arc::new(Int64Array::from(vec![1i64])) as Arc<dyn Array>,
            b.column(2).clone(),
        ],
    )
    .unwrap();
    let err = run_coo(&mixed, 4).unwrap_err();
    assert!(err.to_string().contains("same width"), "{err}");
}

// The `nnz > u32::MAX` ceiling in `for_each_coo_mapping_shard` has no test:
// constructing a batch with 4.3e9 triples is not possible here, and asserting on
// the source text of the error would be a grep dressed up as coverage. The
// guard's justification lives with the guard.
