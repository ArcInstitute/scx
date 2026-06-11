//! End-to-end >2 GiB obs round-trip test for the Arrow IPC offset fix.
//!
//! Arrow IPC's `Utf8`/`Binary` offsets are i32, so any single string
//! buffer is capped at ~2.15 GB. `scx_format_io::arrow_compat`
//! widens to `LargeUtf8`/`LargeBinary` on write and **opportunistically**
//! narrows back on read — so columns whose offsets actually exceed
//! `i32::MAX` stay wide in memory rather than blowing up in
//! `arrow::compute::cast`.
//!
//! This test is the only thing that genuinely exercises that path: it
//! builds an obs column whose UTF-8 buffer is over 2 GiB, writes via
//! `ScxWriter`, reopens via `ScxReader`, and asserts both
//! `read_obs_schema()` and `read_obs()` return `LargeUtf8` (not `Utf8`)
//! for the overflowing column.
//!
//! Marked `#[ignore]` because it allocates ~3 GiB of strings and writes
//! ~2.5 GiB to a tempdir. Run locally with:
//!
//! ```bash
//! cargo test --release -p scx-format --test large_obs -- --ignored --nocapture
//! ```

use std::fmt::Write as _;
use std::sync::Arc;

use arrow::array::{LargeStringArray, LargeStringBuilder, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::dispatch::{CodecId, ValueEncoding};
use scx_format_io::header::{HEADER_SIZE, MAGIC};
use scx_format_io::{FileHeader, ScxReader, ScxWriter};

fn header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: scx_format_io::CURRENT_FORMAT_VERSION,
        header_length: HEADER_SIZE as u16,
        flags: 0,
        n_obs,
        n_vars,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: 16384,
        codec_id: 0,
        index_dtype: if n_vars <= 65535 { 0 } else { 1 },
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    }
}

#[test]
#[ignore]
fn test_obs_largeutf8_overflow_round_trip() {
    // 2.5 M cells × 900 B/string ≈ 2.25 GiB cumulative UTF-8 — just
    // over `i32::MAX` (2.147 GiB). Build directly into a
    // `LargeStringBuilder` so the in-memory array carries i64 offsets;
    // a `StringBuilder` would itself overflow during construction.
    const N: usize = 2_500_000;
    const PAYLOAD_LEN: usize = 880; // chosen so each string ≈ 900 B
    let payload: String = "x".repeat(PAYLOAD_LEN);

    let mut builder = LargeStringBuilder::with_capacity(N, N * 900);
    let mut scratch = String::with_capacity(900);
    for i in 0..N {
        scratch.clear();
        write!(&mut scratch, "cell_{i:07}{payload}").unwrap();
        builder.append_value(&scratch);
    }
    let cell_ids = builder.finish();

    // Sanity check: the array's last offset MUST exceed i32::MAX, or
    // the test isn't actually exercising the >2 GiB code path.
    let last_offset = *cell_ids.value_offsets().last().expect("non-empty offsets");
    assert!(
        last_offset > i32::MAX as i64,
        "fixture must overflow i32 offsets to exercise the fix; got {last_offset}"
    );

    let obs_schema = Schema::new(vec![Field::new("cell_id", DataType::LargeUtf8, false)]);
    let obs_batch = RecordBatch::try_new(Arc::new(obs_schema), vec![Arc::new(cell_ids)]).unwrap();

    // Tiny var (<2 GiB trivially) and a minimal CSR shard so the file
    // satisfies the format invariants without bloating the tempfile.
    let n_vars: usize = 4;
    let var_ids = arrow::array::StringArray::from(vec!["g0", "g1", "g2", "g3"]);
    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let var_batch = RecordBatch::try_new(Arc::new(var_schema), vec![Arc::new(var_ids)]).unwrap();

    // CSR shards each cap at u16::MAX rows (`BlockRowsOverflow`), so
    // split N across ~40 zero-nonzero shards. The actual matrix data
    // is irrelevant for this test — we just need the file to satisfy
    // the format invariants (n_obs == sum of shard rows).
    const ROWS_PER_SHARD: usize = 65_000;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_obs.scx");
    {
        let mut writer = ScxWriter::new(&path, header(N as u64, n_vars as u64, 0)).unwrap();
        writer.write_obs(&obs_batch).unwrap();
        writer.write_var(&var_batch).unwrap();

        let empty_indices: Vec<u32> = Vec::new();
        let empty_values: Vec<u8> = Vec::new();
        let mut row_start: u64 = 0;
        while (row_start as usize) < N {
            let shard_rows = std::cmp::min(ROWS_PER_SHARD, N - row_start as usize);
            let indptr: Vec<u64> = vec![0u64; shard_rows + 1];
            writer
                .write_csr_shard(
                    &indptr,
                    &empty_indices,
                    &empty_values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    row_start,
                )
                .unwrap();
            row_start += shard_rows as u64;
        }
        writer.finish().unwrap();
    }
    drop(obs_batch); // free ~2.25 GiB before reading back

    // Reopen and assert that the >2 GiB obs column round-trips with the
    // canonical wide type intact (opportunistic downcast must NOT have
    // tried to narrow it).
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), N as u64);

    let schema = reader.read_obs_schema().unwrap();
    assert_eq!(
        schema.field(0).data_type(),
        &DataType::LargeUtf8,
        "schema must surface LargeUtf8 for >2 GiB obs (opportunistic downcast \
         must not narrow when offsets overflow i32::MAX)"
    );

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), N);
    assert_eq!(
        obs.schema().field(0).data_type(),
        &DataType::LargeUtf8,
        "data path must surface LargeUtf8 for >2 GiB obs"
    );

    let arr = obs
        .column(0)
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .expect("obs column should be LargeStringArray");
    let last_offset = *arr.value_offsets().last().unwrap();
    assert!(
        last_offset > i32::MAX as i64,
        "round-tripped offsets should still overflow i32::MAX; got {last_offset}"
    );
    assert!(arr.value(0).starts_with("cell_0000000"));
    assert!(arr.value(N - 1).starts_with("cell_2499999"));
}

// ---------------------------------------------------------------------------
// Phase 1f: sharded obs/var metadata round-trips
// ---------------------------------------------------------------------------

/// Build an obs `RecordBatch` of `n` rows with `cell_id: Utf8` and
/// `donor: Utf8` columns. Deterministic — `(cell_id, donor)` are
/// derived from the row index, so callers can split into shards and
/// reassemble without bookkeeping.
fn obs_batch(start_row: usize, n: usize) -> RecordBatch {
    let cell_ids: Vec<String> = (start_row..start_row + n)
        .map(|i| format!("cell_{i:07}"))
        .collect();
    let donors: Vec<String> = (start_row..start_row + n)
        .map(|i| format!("donor_{}", i % 16))
        .collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("donor", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(cell_ids)),
            Arc::new(StringArray::from(donors)),
        ],
    )
    .unwrap()
}

/// Minimal CSR shard machinery: one zero-nonzero shard covering
/// `row_start..row_start + n_rows`. The format requires the CSR shard
/// row counts to sum to `n_obs`, so writers in these tests emit one
/// per metadata shard for symmetry.
fn write_zero_csr_shard(writer: &mut ScxWriter, row_start: u64, n_rows: u64) {
    let indptr: Vec<u64> = vec![0u64; (n_rows + 1) as usize];
    let indices: Vec<u32> = Vec::new();
    let values: Vec<u8> = Vec::new();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            row_start,
        )
        .unwrap();
}

fn small_var_batch() -> RecordBatch {
    let var_ids = StringArray::from(vec!["g0", "g1", "g2", "g3"]);
    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(var_schema), vec![Arc::new(var_ids)]).unwrap()
}

#[test]
fn test_obs_shard_round_trip() {
    // Four shards × 100 rows = 400 obs rows total. Each shard has
    // distinct content so we can verify per-shard reads and the
    // contiguous-cover assembly path.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("obs_shards.scx");
    let n_obs: u64 = 400;
    let shard_rows: u64 = 100;
    let n_shards: u32 = 4;
    {
        let mut writer = ScxWriter::new(&path, header(n_obs, 4, 0)).unwrap();
        for shard_idx in 0..n_shards {
            let row_start = u64::from(shard_idx) * shard_rows;
            let batch = obs_batch(row_start as usize, shard_rows as usize);
            writer
                .write_obs_shard(shard_idx, row_start, shard_rows, n_obs, &batch)
                .unwrap();
            write_zero_csr_shard(&mut writer, row_start, shard_rows);
        }
        writer.write_var(&small_var_batch()).unwrap();
        writer.finish().unwrap();
    }

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), n_shards as usize);
    assert_eq!(reader.var_metadata_shard_count(), 0);

    // Per-shard read returns stamped shard metadata.
    for shard_idx in 0..n_shards {
        let batch = reader.read_obs_shard(shard_idx).unwrap();
        assert_eq!(batch.num_rows(), shard_rows as usize);
        let md = batch.schema().metadata().clone();
        assert_eq!(md.get("shard_idx").unwrap(), &shard_idx.to_string());
        assert_eq!(
            md.get("row_start").unwrap(),
            &(u64::from(shard_idx) * shard_rows).to_string()
        );
        assert_eq!(md.get("n_shard_rows").unwrap(), &shard_rows.to_string());
        assert_eq!(md.get("n_rows_total").unwrap(), &n_obs.to_string());
    }

    // Iterator yields shards in order.
    let total: usize = reader.obs_shards().map(|b| b.unwrap().num_rows()).sum();
    assert_eq!(total as u64, n_obs);
}

#[test]
fn test_obs_shards_read_obs_matches_legacy() {
    // Same obs payload written two ways:
    //   (a) one legacy `ObsMetadata` section,
    //   (b) three `ObsMetadataShard` sections (rows 0..100, 100..250, 250..400).
    // `read_obs()` must transparently handle both layouts and return
    // identical column data. Schemas differ only by the absence of
    // per-shard metadata keys (shard_idx, row_start, n_shard_rows) on
    // the sharded side — `n_rows_total` survives the strip.
    let dir = tempfile::tempdir().unwrap();
    let legacy_path = dir.path().join("legacy.scx");
    let sharded_path = dir.path().join("sharded.scx");
    let full = obs_batch(0, 400);

    {
        let mut w = ScxWriter::new(&legacy_path, header(400, 4, 0)).unwrap();
        w.write_obs(&full).unwrap();
        w.write_var(&small_var_batch()).unwrap();
        write_zero_csr_shard(&mut w, 0, 400);
        w.finish().unwrap();
    }
    {
        let mut w = ScxWriter::new(&sharded_path, header(400, 4, 0)).unwrap();
        let splits = [(0u64, 100u64), (100, 150), (250, 150)];
        for (i, (row_start, n)) in splits.iter().enumerate() {
            let batch = full.slice(*row_start as usize, *n as usize);
            w.write_obs_shard(i as u32, *row_start, *n, 400, &batch)
                .unwrap();
            write_zero_csr_shard(&mut w, *row_start, *n);
        }
        w.write_var(&small_var_batch()).unwrap();
        w.finish().unwrap();
    }

    let legacy_obs = ScxReader::open(&legacy_path).unwrap().read_obs().unwrap();
    let sharded_obs = ScxReader::open(&sharded_path).unwrap().read_obs().unwrap();
    assert_eq!(legacy_obs.num_rows(), sharded_obs.num_rows());
    assert_eq!(legacy_obs.num_columns(), sharded_obs.num_columns());
    for col in 0..legacy_obs.num_columns() {
        assert_eq!(
            legacy_obs.column(col).as_ref(),
            sharded_obs.column(col).as_ref(),
            "column {col} differs between legacy and sharded layouts"
        );
    }
}

#[test]
fn test_mixed_obs_writes_rejected() {
    // Single → shard must error.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_single_then_shard.scx");
    {
        let mut w = ScxWriter::new(&path, header(100, 4, 0)).unwrap();
        w.write_obs(&obs_batch(0, 100)).unwrap();
        let err = w
            .write_obs_shard(0, 0, 100, 100, &obs_batch(0, 100))
            .unwrap_err();
        assert!(
            matches!(err, scx_format_io::ScxError::ObsLayoutConflict { .. }),
            "expected ObsLayoutConflict, got: {err:?}"
        );
    }
    // Shard → single must error.
    let path2 = dir.path().join("mixed_shard_then_single.scx");
    {
        let mut w = ScxWriter::new(&path2, header(100, 4, 0)).unwrap();
        w.write_obs_shard(0, 0, 100, 100, &obs_batch(0, 100))
            .unwrap();
        let err = w.write_obs(&obs_batch(0, 100)).unwrap_err();
        assert!(
            matches!(err, scx_format_io::ScxError::ObsLayoutConflict { .. }),
            "expected ObsLayoutConflict, got: {err:?}"
        );
    }
}

#[test]
fn test_read_obs_on_sharded_assembles_transparently() {
    // After the read_obs / read_obs_assembled collapse, `read_obs()`
    // transparently reassembles row-sharded obs into one RecordBatch.
    // The cover-verification is exercised by
    // `test_obs_shards_read_obs_matches_legacy`; this test just
    // confirms `read_obs()` no longer errors on a sharded file.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sharded.scx");
    {
        let mut w = ScxWriter::new(&path, header(200, 4, 0)).unwrap();
        w.write_obs_shard(0, 0, 100, 200, &obs_batch(0, 100))
            .unwrap();
        w.write_obs_shard(1, 100, 100, 200, &obs_batch(100, 100))
            .unwrap();
        write_zero_csr_shard(&mut w, 0, 200);
        w.write_var(&small_var_batch()).unwrap();
        w.finish().unwrap();
    }
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 2);
    let assembled = reader.read_obs().unwrap();
    assert_eq!(assembled.num_rows(), 200);
}

#[test]
fn test_schema_apis_no_payload_read() {
    // Build a file with non-trivial obs payload so we can demonstrate
    // that the physical/logical-lossy schema APIs are constant-cost in
    // payload size. We can't directly assert "no batch deserialised"
    // without instrumentation, but we can assert correctness of both
    // APIs on both layouts (legacy + sharded) and the cheap-path
    // return-type matches what `read_arrow_ipc_schema_physical`
    // produces.
    let dir = tempfile::tempdir().unwrap();
    let legacy = dir.path().join("legacy_schema.scx");
    let sharded = dir.path().join("sharded_schema.scx");

    // Schema with a `LargeUtf8` column to exercise the
    // logical-lossy narrowing path.
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::LargeUtf8, false),
        Field::new("count", DataType::UInt32, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(LargeStringArray::from(vec!["a", "b", "c"])),
            Arc::new(UInt32Array::from(vec![10u32, 20, 30])),
        ],
    )
    .unwrap();

    {
        let mut w = ScxWriter::new(&legacy, header(3, 4, 0)).unwrap();
        w.write_obs(&batch).unwrap();
        w.write_var(&small_var_batch()).unwrap();
        write_zero_csr_shard(&mut w, 0, 3);
        w.finish().unwrap();
    }
    {
        let mut w = ScxWriter::new(&sharded, header(3, 4, 0)).unwrap();
        w.write_obs_shard(0, 0, 3, 3, &batch).unwrap();
        w.write_var(&small_var_batch()).unwrap();
        write_zero_csr_shard(&mut w, 0, 3);
        w.finish().unwrap();
    }

    for path in [&legacy, &sharded] {
        let reader = ScxReader::open(path).unwrap();
        let phys = reader.read_obs_schema_physical().unwrap();
        let logical = reader.read_obs_schema_logical_lossy().unwrap();

        // Physical preserves on-disk types — LargeUtf8 stays LargeUtf8.
        assert_eq!(
            phys.field_with_name("cell_id").unwrap().data_type(),
            &DataType::LargeUtf8,
            "physical schema must surface LargeUtf8 as written for {path:?}"
        );
        // Logical-lossy unconditionally narrows.
        assert_eq!(
            logical.field_with_name("cell_id").unwrap().data_type(),
            &DataType::Utf8,
            "logical-lossy schema must narrow LargeUtf8 → Utf8 for {path:?}"
        );
        // Non-wide columns are untouched.
        assert_eq!(
            phys.field_with_name("count").unwrap().data_type(),
            &DataType::UInt32,
        );
        assert_eq!(
            logical.field_with_name("count").unwrap().data_type(),
            &DataType::UInt32,
        );
    }
}

#[test]
#[ignore]
fn test_obs_shards_largeutf8_overflow_round_trip() {
    // Phase 1f overflow check: three shards × ~800 MB cumulative
    // string-buffer per shard → summed > 2 GiB. Each individual
    // shard's offsets stay below `i32::MAX` (the writer's per-shard
    // upcast in `write_arrow_ipc` happens regardless), and the
    // assembly path must keep the cumulative column wide because the
    // concatenated offsets overflow.
    //
    // Marked `#[ignore]` — allocates ~3 GiB; run with:
    //
    // ```bash
    // cargo test --release -p scx-format --test large_obs -- --ignored --nocapture
    // ```
    const SHARDS: u32 = 3;
    const ROWS_PER_SHARD: usize = 900_000;
    const PAYLOAD_LEN: usize = 880;
    let n_obs = SHARDS as u64 * ROWS_PER_SHARD as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overflow_shards.scx");

    {
        let mut writer = ScxWriter::new(&path, header(n_obs, 4, 0)).unwrap();
        let payload: String = "x".repeat(PAYLOAD_LEN);
        for shard_idx in 0..SHARDS {
            let row_start = u64::from(shard_idx) * ROWS_PER_SHARD as u64;
            let mut builder = LargeStringBuilder::with_capacity(
                ROWS_PER_SHARD,
                ROWS_PER_SHARD * (PAYLOAD_LEN + 20),
            );
            let mut scratch = String::with_capacity(PAYLOAD_LEN + 20);
            for i in 0..ROWS_PER_SHARD {
                let global = row_start as usize + i;
                scratch.clear();
                write!(&mut scratch, "cell_{global:09}{payload}").unwrap();
                builder.append_value(&scratch);
            }
            let cell_ids = builder.finish();
            let schema = Schema::new(vec![Field::new("cell_id", DataType::LargeUtf8, false)]);
            let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(cell_ids)]).unwrap();
            writer
                .write_obs_shard(shard_idx, row_start, ROWS_PER_SHARD as u64, n_obs, &batch)
                .unwrap();
            write_zero_csr_shard(&mut writer, row_start, ROWS_PER_SHARD as u64);
        }
        writer.write_var(&small_var_batch()).unwrap();
        writer.finish().unwrap();
    }

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), SHARDS as usize);

    let mut total_rows: usize = 0;
    for shard_result in reader.obs_shards() {
        let shard = shard_result.unwrap();
        total_rows += shard.num_rows();
    }
    assert_eq!(total_rows as u64, n_obs);
}

#[test]
fn test_reader_accepts_mixed_n_rows_total_across_shards() {
    // Append-grown obs leaves older shards stamped with their original
    // (smaller) `n_rows_total` value while later-appended shards carry
    // the bumped total. The reader's cover-verification must accept
    // this monotonically-non-decreasing pattern; only the LAST shard's
    // `n_rows_total` is canonical.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_totals.scx");
    {
        let mut w = ScxWriter::new(&path, header(150, 4, 0)).unwrap();
        // Shard 0 stamped with the file's pre-append total (100).
        w.write_obs_shard(0, 0, 100, 100, &obs_batch(0, 100))
            .unwrap();
        // Shard 1 stamped with the post-append total (150). The
        // writer doesn't validate consistency between shards; that's
        // the reader's job.
        w.write_obs_shard(1, 100, 50, 150, &obs_batch(100, 50))
            .unwrap();
        write_zero_csr_shard(&mut w, 0, 150);
        w.write_var(&small_var_batch()).unwrap();
        w.finish().unwrap();
    }
    let reader = ScxReader::open(&path).unwrap();
    let obs = reader.read_obs().unwrap();
    assert_eq!(
        obs.num_rows(),
        150,
        "cover sum (100 + 50) becomes the total"
    );
}

#[test]
fn test_reader_rejects_contracting_n_rows_total() {
    // The inverse case: shard 1's `n_rows_total` is *smaller* than
    // shard 0's. That can't happen under correct append semantics
    // (totals only grow) and indicates either catalog corruption or
    // a writer bug. Reject it.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("contracting_totals.scx");
    {
        let mut w = ScxWriter::new(&path, header(150, 4, 0)).unwrap();
        w.write_obs_shard(0, 0, 100, 150, &obs_batch(0, 100))
            .unwrap();
        // Shard 1 contracts the stamp to 50 — invalid.
        w.write_obs_shard(1, 100, 50, 50, &obs_batch(100, 50))
            .unwrap();
        write_zero_csr_shard(&mut w, 0, 150);
        w.write_var(&small_var_batch()).unwrap();
        w.finish().unwrap();
    }
    let reader = ScxReader::open(&path).unwrap();
    let err = reader.read_obs().unwrap_err();
    assert!(
        matches!(err, scx_format_io::ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog for contracting n_rows_total, got: {err:?}"
    );
}
