//! End-to-end >2 GiB obs round-trip test for the Arrow IPC offset fix.
//!
//! Arrow IPC's `Utf8`/`Binary` offsets are i32, so any single string
//! buffer is capped at ~2.15 GB. `scx_format::arrow_compat`
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

use arrow::array::{LargeStringArray, LargeStringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::dispatch::{CodecId, ValueEncoding};
use scx_format::header::{HEADER_SIZE, MAGIC};
use scx_format::{FileHeader, ScxReader, ScxWriter};

fn header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: 1,
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
        reserved: [0u8; 132],
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
