//! Phase 5a: v2 pairwise COO Int64 coordinate-width round-trip.
//!
//! Pairwise sections (`obsp/<name>`, `varp/<name>` and their `*_shard_*`
//! variants) are written as Arrow IPC RecordBatches with `row`/`col`/`data`
//! columns. The v1 wire format uses Int32 row/col coordinates; the v2
//! format uses Int64. The Arrow schema self-describes the width, so the
//! reader can transparently handle either.
//!
//! This test exercises the v2 layout end-to-end: build a synthetic obsp
//! shard whose row/col columns use Int64 with values that exceed
//! `i32::MAX`, write through `ScxWriter::write_obsp_shard_coo`, reopen
//! via `ScxReader::read_obsp`, and assert the on-disk schema and
//! coordinate values survive intact. Without v2 support these values
//! would have to be either truncated (silent corruption) or rejected
//! (a hard ceiling).

use std::sync::Arc;

use arrow::array::{Float32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::dispatch::{CodecId, ValueEncoding};
use scx_format_io::{FileHeader, ScxReader, ScxWriter};

fn header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
}

#[test]
fn v2_obsp_int64_coordinates_round_trip_through_writer_and_reader() {
    // Logical axes far past i32::MAX. The actual file has 5 obs and a
    // 5-entry obsp; only the recorded coordinates (and the schema
    // metadata `n_rows` / `n_cols`) carry the multi-billion-row claim.
    const HUGE_AXIS: i64 = (i32::MAX as i64) * 3; // ~6.44B

    // Build a v2 (Int64) obsp shard batch directly. row[2], col[2],
    // row[4], col[4] all exceed i32::MAX — under v1 (Int32) routing
    // these would silently wrap on cast. Under v2 they survive.
    let rows = vec![0i64, 1, HUGE_AXIS - 1, 2, HUGE_AXIS - 2];
    let cols = vec![1i64, 0, HUGE_AXIS - 2, 2, HUGE_AXIS - 1];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

    // Sanity: the test fixture must actually exercise the > i32::MAX
    // boundary, otherwise the v2 path isn't being tested.
    assert!(*rows.iter().max().unwrap() > i32::MAX as i64);
    assert!(*cols.iter().max().unwrap() > i32::MAX as i64);

    use std::collections::HashMap;
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int64, false),
            Field::new("col", DataType::Int64, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), HUGE_AXIS.to_string()),
            ("n_cols".to_string(), HUGE_AXIS.to_string()),
        ]),
    ));
    let obsp_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(rows.clone())),
            Arc::new(Int64Array::from(cols.clone())),
            Arc::new(Float32Array::from(data.clone())),
        ],
    )
    .unwrap();

    // Build a minimal file with valid obs / var / one empty X shard
    // plus the v2 obsp_shard. The X shard is required because the
    // file header records `n_obs` and the catalog needs at least one
    // CsrShard to satisfy the format invariants. Use n_obs = 5 (the
    // tiny test obs, not the logical HUGE_AXIS — the obsp's logical
    // shape is independent of the X's row count).
    const N_OBS: usize = 5;
    const N_VARS: usize = 3;
    let obs_ids = StringArray::from(vec!["c0", "c1", "c2", "c3", "c4"]);
    let obs_schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    let obs_batch = RecordBatch::try_new(Arc::new(obs_schema), vec![Arc::new(obs_ids)]).unwrap();
    let var_ids = StringArray::from(vec!["g0", "g1", "g2"]);
    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let var_batch = RecordBatch::try_new(Arc::new(var_schema), vec![Arc::new(var_ids)]).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2_obsp.scx");
    {
        let mut writer = ScxWriter::new(&path, header(N_OBS as u64, N_VARS as u64)).unwrap();
        writer.write_obs(&obs_batch).unwrap();
        writer.write_var(&var_batch).unwrap();

        // Empty CSR shard spanning all rows.
        let indptr: Vec<u64> = vec![0u64; N_OBS + 1];
        let empty_indices: Vec<u32> = Vec::new();
        let empty_values: Vec<u8> = Vec::new();
        writer
            .write_csr_shard(
                &indptr,
                &empty_indices,
                &empty_values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        // Single shard of the v2 obsp. `n_rows_total` is the obsp's
        // logical row count (HUGE_AXIS), `n_shard_rows` covers the
        // 4 distinct rows in the shard. Values are global indices
        // (no shard-local renumbering) per the shard-coo contract.
        writer
            .write_obsp_shard_coo(
                "W_huge",
                /* shard_idx */ 0,
                /* row_start */ 0,
                /* n_shard_rows */ HUGE_AXIS as u64,
                /* n_rows_total */ HUGE_AXIS as u64,
                &obsp_batch,
            )
            .unwrap();
        writer.finish().unwrap();
    }

    // Reopen and assert the v2 layout survives end-to-end.
    let reader = ScxReader::open(&path).unwrap();
    let obsp_round_trip = reader.read_obsp("W_huge").unwrap();

    // 1. Schema preserves Int64 coordinate dtype.
    assert_eq!(
        obsp_round_trip.schema().field(0).data_type(),
        &DataType::Int64,
        "v2 obsp must round-trip with Int64 row column"
    );
    assert_eq!(
        obsp_round_trip.schema().field(1).data_type(),
        &DataType::Int64,
        "v2 obsp must round-trip with Int64 col column"
    );

    // 2. n_rows / n_cols metadata records the HUGE_AXIS value.
    let meta = obsp_round_trip.schema().metadata().clone();
    assert_eq!(
        meta.get("n_rows").map(String::as_str),
        Some(HUGE_AXIS.to_string().as_str())
    );
    assert_eq!(
        meta.get("n_cols").map(String::as_str),
        Some(HUGE_AXIS.to_string().as_str())
    );

    // 3. Coordinate values themselves survive intact (no Int32 wrap).
    let rows_back = obsp_round_trip
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("v2 obsp row column must be Int64Array");
    let cols_back = obsp_round_trip
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("v2 obsp col column must be Int64Array");
    let data_back = obsp_round_trip
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("obsp data column must be Float32Array");

    assert_eq!(rows_back.values(), &rows[..]);
    assert_eq!(cols_back.values(), &cols[..]);
    assert_eq!(data_back.values(), &data[..]);
    // Spot-check the > i32::MAX entries — these are the ones that
    // would silently wrap on a v1 (Int32) cast.
    assert_eq!(rows_back.value(2), HUGE_AXIS - 1);
    assert_eq!(cols_back.value(4), HUGE_AXIS - 1);
}
