use super::*;
use scx_format_io::reader::ScxReader;
use scx_format_io::writer::ScxWriter;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use std::sync::Arc;

fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
}

fn sample_obs(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_var(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_rows {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
    let path = dir.path().join("input.scx");
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = 50;
    let mut row_offset = 0;
    while row_offset < n_obs {
        let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_offset as u64,
            )
            .unwrap();
        row_offset += shard_rows;
    }
    writer.finish().unwrap();
    path
}

/// Helper: create a test file, explode it, then pull from the exploded dir.
async fn pull_from_exploded(
    dir: &tempfile::TempDir,
    n_obs: usize,
    n_vars: usize,
    parallelism: usize,
) -> std::path::PathBuf {
    let input = write_test_file(dir, n_obs, n_vars);
    let exploded_dir = dir.path().join("exploded.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let output = dir.path().join(format!("pulled_p{parallelism}.scx"));
    let opts = PullOptions {
        parallelism,
        cloud_ready: true,
        filter_mode: FilterMode::default(),
        retry_config: RetryConfig::default(),
    };

    let source = exploded_dir.to_string_lossy().to_string();
    pull(&source, &output, opts).await.unwrap();
    output
}

#[tokio::test]
async fn test_pull_produces_valid_scx() {
    let dir = tempfile::tempdir().unwrap();
    let output = pull_from_exploded(&dir, 100, 50, 4).await;

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 100);
    assert_eq!(reader.n_vars(), 50);

    // Read data
    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 100);
    let var = reader.read_var().unwrap();
    assert_eq!(var.num_rows(), 50);
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 100);
    assert_eq!(csr.shape.1, 50);
}

#[tokio::test]
async fn test_pull_validates_checksums() {
    let dir = tempfile::tempdir().unwrap();
    let output = pull_from_exploded(&dir, 100, 50, 4).await;

    let reader = ScxReader::open(&output).unwrap();
    let results = reader.validate().unwrap();
    for (name, passed) in &results {
        assert!(passed, "checksum failed for section: {name}");
    }
}

#[tokio::test]
async fn test_pull_parallelism_1_matches_8() {
    let dir = tempfile::tempdir().unwrap();

    // Create source file and explode
    let input = write_test_file(&dir, 100, 50);
    let exploded_dir = dir.path().join("exploded.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let source = exploded_dir.to_string_lossy().to_string();

    // Pull with parallelism=1
    let output1 = dir.path().join("pulled_p1.scx");
    let opts1 = PullOptions {
        parallelism: 1,
        cloud_ready: true,
        filter_mode: FilterMode::default(),
        retry_config: RetryConfig::default(),
    };
    pull(&source, &output1, opts1).await.unwrap();

    // Pull with parallelism=8
    let output8 = dir.path().join("pulled_p8.scx");
    let opts8 = PullOptions {
        parallelism: 8,
        cloud_ready: true,
        filter_mode: FilterMode::default(),
        retry_config: RetryConfig::default(),
    };
    pull(&source, &output8, opts8).await.unwrap();

    // Compare CSR data (not byte-identical due to different file checksums,
    // but data should be identical)
    let reader1 = ScxReader::open(&output1).unwrap();
    let reader8 = ScxReader::open(&output8).unwrap();

    assert_eq!(reader1.n_obs(), reader8.n_obs());
    assert_eq!(reader1.n_vars(), reader8.n_vars());
    assert_eq!(reader1.nnz(), reader8.nnz());

    let csr1 = reader1.read_all_csr_shards().unwrap();
    let csr8 = reader8.read_all_csr_shards().unwrap();
    assert_eq!(csr1.indptr, csr8.indptr);
    assert_eq!(csr1.indices, csr8.indices);
    assert_eq!(csr1.data, csr8.data);
}

#[tokio::test]
async fn test_pull_stats() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, 100, 50);
    let exploded_dir = dir.path().join("exploded.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let source = exploded_dir.to_string_lossy().to_string();
    let output = dir.path().join("pulled.scx");
    let opts = PullOptions::default();

    let stats = pull(&source, &output, opts).await.unwrap();

    assert!(stats.bytes_downloaded > 0);
    assert!(stats.sections_downloaded > 0);
    assert!(stats.elapsed.as_nanos() > 0);
}

#[tokio::test]
async fn test_pull_cloud_ready_has_front_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let output = pull_from_exploded(&dir, 100, 50, 4).await;

    let data = std::fs::read(&output).unwrap();
    let hdr = FileHeader::read_from(&mut Cursor::new(&data[..HEADER_SIZE])).unwrap();
    assert!(hdr.has_front_catalog(), "pull output should be cloud-ready");
    assert!(hdr.front_catalog_offset > 0);
    assert!(hdr.front_catalog_length > 0);
}

// ===== pull_filtered tests =====

fn write_test_file_with_cell_type(
    dir: &tempfile::TempDir,
    n_obs: usize,
    n_vars: usize,
) -> std::path::PathBuf {
    let path = dir.path().join("input_filter.scx");
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let types: Vec<String> = (0..n_obs)
        .map(|i| {
            if i < n_obs / 2 {
                "typeA".to_string()
            } else {
                "typeB".to_string()
            }
        })
        .collect();
    let obs_schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let obs_batch = arrow::array::RecordBatch::try_new(
        Arc::new(obs_schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                types.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    writer.write_obs(&obs_batch).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = 50;
    let mut row_offset = 0;
    while row_offset < n_obs {
        let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_offset as u64,
            )
            .unwrap();
        row_offset += shard_rows;
    }
    writer.finish().unwrap();
    path
}

#[tokio::test]
async fn test_selective_pull_downloads_only_matching_shards() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file_with_cell_type(&dir, 100, 50);
    let exploded_dir = dir.path().join("exploded_filter.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let output = dir.path().join("filtered.scx");
    let source = exploded_dir.to_string_lossy().to_string();

    let stats = pull_filtered(
        &source,
        &output,
        "cell_type == 'typeA'",
        PullOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(stats.total_shards, 2);
    assert_eq!(stats.downloaded_shards, 1);
    assert_eq!(stats.skipped_shards, 1);
    assert_eq!(stats.matching_cells, 50);
    assert!(stats.bytes_saved > 0);
}

#[tokio::test]
async fn test_selective_pull_correct_cell_count_and_data() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file_with_cell_type(&dir, 100, 50);
    let exploded_dir = dir.path().join("exploded_filter.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let output = dir.path().join("filtered.scx");
    let source = exploded_dir.to_string_lossy().to_string();

    pull_filtered(
        &source,
        &output,
        "cell_type == 'typeA'",
        PullOptions::default(),
    )
    .await
    .unwrap();

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 50);
    assert_eq!(reader.n_vars(), 50);

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 50);

    let cell_type_col = obs
        .column_by_name("cell_type")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..arrow::array::Array::len(cell_type_col) {
        assert_eq!(cell_type_col.value(i), "typeA");
    }

    let var = reader.read_var().unwrap();
    assert_eq!(var.num_rows(), 50);
}

#[tokio::test]
async fn test_selective_pull_no_matching_cells() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file_with_cell_type(&dir, 100, 50);
    let exploded_dir = dir.path().join("exploded_filter.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let output = dir.path().join("filtered_empty.scx");
    let source = exploded_dir.to_string_lossy().to_string();

    let stats = pull_filtered(
        &source,
        &output,
        "cell_type == 'typeC'",
        PullOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(stats.matching_cells, 0);
    assert_eq!(stats.downloaded_shards, 0);
    assert_eq!(stats.skipped_shards, 2);

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 0);
}

/// Force the mtime on `path` to `STALE_TMP_AGE_OLD + 60s` in the past
/// so the cleanup helper treats an old-format orphan as stale. Uses
/// `File::set_times` (stable since Rust 1.75) and
/// `FileTimes::set_modified`.
fn age_file(path: &std::path::Path) {
    age_file_by(path, STALE_TMP_AGE_OLD + std::time::Duration::from_secs(60));
}

/// Force the mtime on `path` to `age` in the past. Used by new-format
/// tests that need to age beyond `STALE_TMP_AGE_NEW` (24 h) or, for
/// the freshness test, sit within the threshold.
fn age_file_by(path: &std::path::Path, age: std::time::Duration) {
    let past = std::time::SystemTime::now() - age;
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open file for set_times");
    let times = std::fs::FileTimes::new().set_modified(past);
    f.set_times(times).expect("set_times failed");
}

#[test]
fn test_cleanup_stale_tmp_files_removes_dead_pid_and_old_mtime() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("output.scx");

    // u32::MAX exceeds the kernel's pid_max (default 2^22 on Linux)
    // so /proc/4294967295 cannot exist. Also age the files past
    // STALE_TMP_AGE so the mtime guard clears in both branches.
    let stale_dead_pid = dir.path().join("output.scx.tmp.4294967295");
    let stale_unparseable = dir.path().join("output.scx.tmp.abcd");
    let unrelated = dir.path().join("other.scx");
    for p in &[&stale_dead_pid, &stale_unparseable, &unrelated] {
        std::fs::write(p, b"stale").unwrap();
    }
    age_file(&stale_dead_pid);
    age_file(&stale_unparseable);

    cleanup_stale_tmp_files(&dest);

    assert!(
        !stale_dead_pid.exists(),
        "dead-pid orphan past age should be removed",
    );
    assert!(
        !stale_unparseable.exists(),
        "unparseable-suffix orphan past age should be removed",
    );
    assert!(unrelated.exists(), "unrelated file must not be touched");
}

#[test]
fn test_cleanup_stale_tmp_files_preserves_live_pid() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("output.scx");

    // Our own pid is provably alive; even with an aged mtime the
    // helper must leave it alone (concurrent-pull safety).
    let live_pid = std::process::id();
    let live = dir.path().join(format!("output.scx.tmp.{live_pid}"));
    std::fs::write(&live, b"in-flight").unwrap();
    age_file(&live);

    cleanup_stale_tmp_files(&dest);

    assert!(
        live.exists(),
        "tmp file owned by a live pid must never be unlinked",
    );
}

#[test]
fn test_cleanup_stale_tmp_files_preserves_fresh_mtime() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("output.scx");

    // Dead pid but fresh mtime: could be a sibling that spawned
    // moments ago on a system where /proc/{pid} hasn't materialized
    // yet, or a pid we can't confirm. Skip deletion to stay safe.
    let fresh_dead = dir.path().join("output.scx.tmp.4294967295");
    std::fs::write(&fresh_dead, b"very recent").unwrap();

    cleanup_stale_tmp_files(&dest);

    assert!(
        fresh_dead.exists(),
        "fresh orphan should not be unlinked; mtime guard must hold",
    );
}

#[test]
fn test_cleanup_stale_tmp_files_handles_missing_parent() {
    // Should not panic on a destination whose parent directory doesn't exist.
    let nonexistent = std::path::Path::new("/tmp/scx-cloud-nonexistent-xyz/foo.scx");
    cleanup_stale_tmp_files(nonexistent);
}

/// New-format orphan (`.{stem}_<rand>.tmp`) older than 24 h is reaped.
/// No pid check applies — `tempfile`'s random suffix guarantees no
/// collision with a live process, so the mtime guard is the sole
/// safety net.
#[test]
fn test_cleanup_stale_tmp_files_removes_new_format_orphan() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("output.scx");

    let orphan = dir.path().join(".output.scx_aBcDeF.tmp");
    std::fs::write(&orphan, b"stale new-format").unwrap();
    // Age past 24 h + slop.
    age_file_by(
        &orphan,
        STALE_TMP_AGE_NEW + std::time::Duration::from_secs(60),
    );

    cleanup_stale_tmp_files(&dest);

    assert!(
        !orphan.exists(),
        "new-format orphan past 24 h should be reaped",
    );
}

/// New-format file with fresh mtime (well within 24 h) must be
/// preserved — could belong to an in-flight pull from this or another
/// process.
#[test]
fn test_cleanup_stale_tmp_files_preserves_fresh_new_format() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("output.scx");

    let fresh = dir.path().join(".output.scx_qWeRtY.tmp");
    std::fs::write(&fresh, b"in-flight").unwrap();
    // No aging — mtime is "now".

    cleanup_stale_tmp_files(&dest);

    assert!(fresh.exists(), "fresh new-format file should not be reaped",);
}

/// New-format file at ~2 h old would be reaped under the old 1 h
/// threshold; assert the new 24 h threshold preserves it. Guards
/// against accidental regression to a too-short threshold for very
/// large pulls.
#[test]
fn test_cleanup_stale_tmp_files_preserves_new_format_within_24h() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("output.scx");

    let two_h_old = dir.path().join(".output.scx_zXcVbN.tmp");
    std::fs::write(&two_h_old, b"long pull").unwrap();
    age_file_by(&two_h_old, std::time::Duration::from_secs(2 * 60 * 60));

    cleanup_stale_tmp_files(&dest);

    assert!(
        two_h_old.exists(),
        "new-format file aged 2 h (< 24 h) must be preserved",
    );
}

#[tokio::test]
async fn test_pull_retry_after_orphan_tmp_is_idempotent() {
    // Simulates an interrupted pull by leaving an orphan `.tmp.*` file
    // before starting a fresh pull; the fresh pull must succeed and
    // produce a byte-identical output to a clean pull.
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, 100, 50);
    let exploded_dir = dir.path().join("exploded.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let source = exploded_dir.to_string_lossy().to_string();
    let output = dir.path().join("pulled.scx");

    // Plant an orphan `.tmp.*` as if a prior pull crashed. Use
    // u32::MAX (above pid_max) AND age the file beyond STALE_TMP_AGE
    // so the sweeper's pid+mtime guard both fire and the orphan is
    // unlinked. A fresh-mtime orphan is preserved by design — that
    // path is covered by ``test_cleanup_stale_tmp_files_preserves_fresh_mtime``.
    let orphan = dir.path().join("pulled.scx.tmp.4294967295");
    std::fs::write(&orphan, b"junk from an interrupted prior run").unwrap();
    age_file(&orphan);

    let opts = PullOptions::default();
    let stats = pull(&source, &output, opts).await.unwrap();

    assert!(stats.bytes_downloaded > 0);
    assert!(output.exists(), "pull output should exist after retry");
    assert!(
        !orphan.exists(),
        "aged orphan tmp file should have been swept"
    );

    // Verify the output is a valid SCX file.
    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 100);
    assert_eq!(reader.n_vars(), 50);
}

#[tokio::test]
async fn test_selective_pull_all_cells_matching() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file_with_cell_type(&dir, 100, 50);
    let exploded_dir = dir.path().join("exploded_filter.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let source = exploded_dir.to_string_lossy().to_string();

    let full_output = dir.path().join("full.scx");
    pull(&source, &full_output, PullOptions::default())
        .await
        .unwrap();

    let filtered_output = dir.path().join("filtered_all.scx");
    let stats = pull_filtered(
        &source,
        &filtered_output,
        "cell_type == 'typeA' or cell_type == 'typeB'",
        PullOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(stats.matching_cells, 100);
    assert_eq!(stats.downloaded_shards, 2);
    assert_eq!(stats.skipped_shards, 0);

    let reader_full = ScxReader::open(&full_output).unwrap();
    let reader_filt = ScxReader::open(&filtered_output).unwrap();

    assert_eq!(reader_full.n_obs(), reader_filt.n_obs());
    assert_eq!(reader_full.n_vars(), reader_filt.n_vars());

    let csr_full = reader_full.read_all_csr_shards().unwrap();
    let csr_filt = reader_filt.read_all_csr_shards().unwrap();
    assert_eq!(csr_full.indptr, csr_filt.indptr);
    assert_eq!(csr_full.indices, csr_filt.indices);
    assert_eq!(csr_full.data, csr_filt.data);
}

// ===== Patch 5 tests =====

/// Verify the ordered bounded download pipeline (LC2) produces
/// byte-identical output regardless of parallelism — `buffered` yields in
/// write order, so write offsets/padding are deterministic.
#[tokio::test]
async fn test_pull_buffered_matches_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, 200, 60);
    let exploded_dir = dir.path().join("exploded_stream.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let source = exploded_dir.to_string_lossy().to_string();

    // Sequential (parallelism=1)
    let out_seq = dir.path().join("seq.scx");
    let opts_seq = PullOptions {
        parallelism: 1,
        cloud_ready: true,
        filter_mode: FilterMode::default(),
        retry_config: RetryConfig::default(),
    };
    pull(&source, &out_seq, opts_seq).await.unwrap();

    // Concurrent (parallelism=16)
    let out_par = dir.path().join("par.scx");
    let opts_par = PullOptions {
        parallelism: 16,
        cloud_ready: true,
        filter_mode: FilterMode::default(),
        retry_config: RetryConfig::default(),
    };
    pull(&source, &out_par, opts_par).await.unwrap();

    let r_seq = ScxReader::open(&out_seq).unwrap();
    let r_par = ScxReader::open(&out_par).unwrap();

    assert_eq!(r_seq.n_obs(), r_par.n_obs());
    assert_eq!(r_seq.n_vars(), r_par.n_vars());
    assert_eq!(r_seq.nnz(), r_par.nnz());

    let csr_seq = r_seq.read_all_csr_shards().unwrap();
    let csr_par = r_par.read_all_csr_shards().unwrap();
    assert_eq!(csr_seq.indptr, csr_par.indptr);
    assert_eq!(csr_seq.indices, csr_par.indices);
    assert_eq!(csr_seq.data, csr_par.data);

    // LC2: ordered pipeline ⇒ the on-disk files are byte-identical, not
    // merely semantically equal.
    let bytes_seq = std::fs::read(&out_seq).unwrap();
    let bytes_par = std::fs::read(&out_par).unwrap();
    assert_eq!(
        bytes_seq, bytes_par,
        "parallelism must not change the on-disk byte layout"
    );
}

/// Verify that the streaming reorder window does NOT buffer all
/// sections simultaneously: with parallelism=1, pull still works.
#[tokio::test]
async fn test_pull_streaming_parallelism_1() {
    let dir = tempfile::tempdir().unwrap();
    let output = pull_from_exploded(&dir, 100, 50, 1).await;

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 100);
    assert_eq!(reader.n_vars(), 50);

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 100);
    assert_eq!(csr.shape.1, 50);
}

/// Selective pull should report omitted section types when the source
/// contains sections beyond the basic obs/var/csr set.
/// The simple test fixture only has CsrShard that didn't match, which
/// is tracked in `skipped_shards` rather than `omitted_section_types`.
/// No other section types (obsm, layers, etc.) exist in the simple
/// fixture, so `omitted_section_types` should be empty.
#[tokio::test]
async fn test_pull_filtered_reports_filter_mode() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file_with_cell_type(&dir, 100, 50);
    let exploded_dir = dir.path().join("exploded_filter.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let output = dir.path().join("filtered_mode.scx");
    let source = exploded_dir.to_string_lossy().to_string();

    let stats = pull_filtered(
        &source,
        &output,
        "cell_type == 'typeA'",
        PullOptions::default(),
    )
    .await
    .unwrap();

    // Default mode is shard
    assert_eq!(stats.filter_mode, FilterMode::Shard);
    // Simple fixture has only obs + var + csr shards. CsrShard is
    // excluded from omitted_section_types (tracked via skipped_shards),
    // and no obsm/layers/obsp sections exist, so the list is empty.
    assert!(
        !stats.omitted_section_types.contains(&SectionType::CsrShard),
        "CsrShard should not appear in omitted_section_types; got {:?}",
        stats.omitted_section_types,
    );
    assert!(
        stats.omitted_section_types.is_empty(),
        "expected empty omitted list for simple fixture, got {:?}",
        stats.omitted_section_types,
    );
}

/// Shard-granular mode includes ALL cells from matching shards,
/// not just predicate-matching ones. This test pins that behavior.
#[tokio::test]
async fn test_pull_filtered_shard_mode_includes_extra_cells() {
    let dir = tempfile::tempdir().unwrap();
    // 100 cells: 0..49 typeA, 50..99 typeB, shard_size=50
    // Both types fit cleanly in separate shards, so no extra cells.
    // Use a shard_size that straddles the type boundary instead.
    let path = dir.path().join("straddle.scx");
    let n_obs = 100;
    let n_vars = 20;
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    // Alternate types so every shard contains both typeA and typeB
    let types: Vec<String> = (0..n_obs)
        .map(|i| {
            if i % 2 == 0 {
                "typeA".to_string()
            } else {
                "typeB".to_string()
            }
        })
        .collect();
    let obs_schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let obs_batch = arrow::array::RecordBatch::try_new(
        Arc::new(obs_schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                types.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    writer.write_obs(&obs_batch).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = 50;
    let mut row_offset = 0;
    while row_offset < n_obs {
        let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_offset as u64,
            )
            .unwrap();
        row_offset += shard_rows;
    }
    writer.finish().unwrap();

    let exploded_dir = dir.path().join("straddle.scxd");
    crate::explode::explode(&path, &exploded_dir).unwrap();
    let source = exploded_dir.to_string_lossy().to_string();

    let output = dir.path().join("straddle_filtered.scx");
    let stats = pull_filtered(
        &source,
        &output,
        "cell_type == 'typeA'",
        PullOptions::default(),
    )
    .await
    .unwrap();

    // Both shards contain typeA cells, so both are downloaded.
    assert_eq!(stats.downloaded_shards, 2);
    // Predicate matches 50 cells, but shard-granular mode returns all 100
    // (both full shards).
    assert_eq!(stats.matching_cells, 50);
    // `output_cells` reports the cells actually WRITTEN (all rows of the
    // retained shards), distinct from `matching_cells` (F2). For a
    // shard-granular pull that straddles the type boundary this is the
    // full 100, strictly greater than the 50 matches.
    assert_eq!(stats.output_cells, 100);
    assert!(stats.output_cells > stats.matching_cells);

    let reader = ScxReader::open(&output).unwrap();
    // n_obs should be ALL cells from downloaded shards (100), not just
    // the 50 matching ones — that's the shard-granular contract.
    assert_eq!(reader.n_obs(), 100);
    // `output_cells` must equal the output file's actual row count.
    assert_eq!(stats.output_cells, reader.n_obs());
}

/// Regression: `pull_filtered` must recompute the output header `nnz` for
/// the retained shard subset. It previously carried the full source
/// dataset's `nnz` verbatim, so `scx info` on a pulled subset reported the
/// whole-dataset nnz. Uses a clean type→shard split so only ONE of two
/// shards is downloaded, making the correct subset nnz strictly less than
/// the full nnz (a stale full-nnz value would fail the assertions).
#[tokio::test]
async fn test_pull_filtered_recomputes_subset_nnz() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("split.scx");
    let n_obs = 100;
    let n_vars = 20;
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    // Clean split: cells 0..49 = typeA (shard 0), 50..99 = typeB (shard 1).
    let types: Vec<String> = (0..n_obs)
        .map(|i| if i < 50 { "typeA" } else { "typeB" }.to_string())
        .collect();
    let obs_schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let obs_batch = arrow::array::RecordBatch::try_new(
        Arc::new(obs_schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                types.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    writer.write_obs(&obs_batch).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = 50;
    let mut row_offset = 0;
    while row_offset < n_obs {
        let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_offset as u64,
            )
            .unwrap();
        row_offset += shard_rows;
    }
    writer.finish().unwrap();

    // Full-dataset nnz from the unfiltered source, for the < assertion.
    let full_nnz = ScxReader::open(&path).unwrap().header().nnz;
    assert!(full_nnz > 0);

    let exploded_dir = dir.path().join("split.scxd");
    crate::explode::explode(&path, &exploded_dir).unwrap();
    let source = exploded_dir.to_string_lossy().to_string();

    let output = dir.path().join("split_filtered.scx");
    let stats = pull_filtered(
        &source,
        &output,
        "cell_type == 'typeA'",
        PullOptions::default(),
    )
    .await
    .unwrap();
    // typeA lives only in shard 0 → exactly one shard downloaded.
    assert_eq!(stats.downloaded_shards, 1);

    let reader = ScxReader::open(&output).unwrap();
    let actual_nnz = reader.read_all_csr_shards().unwrap().data.len() as u64;
    assert_eq!(
        reader.header().nnz,
        actual_nnz,
        "header nnz must equal the materialized subset nnz"
    );
    assert!(
        reader.header().nnz < full_nnz,
        "subset nnz ({}) must be strictly less than full nnz ({full_nnz}); \
         a stale full-dataset nnz would equal it",
        reader.header().nnz
    );
}

/// Exact filter mode should return an error (not yet implemented).
#[tokio::test]
async fn test_pull_filtered_exact_mode_errors() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file_with_cell_type(&dir, 100, 50);
    let exploded_dir = dir.path().join("exploded_exact.scxd");
    crate::explode::explode(&input, &exploded_dir).unwrap();

    let output = dir.path().join("exact.scx");
    let source = exploded_dir.to_string_lossy().to_string();

    let opts = PullOptions {
        filter_mode: FilterMode::Exact,
        ..PullOptions::default()
    };

    let result = pull_filtered(&source, &output, "cell_type == 'typeA'", opts).await;

    assert!(result.is_err(), "exact mode should return an error");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not yet implemented"),
        "error message should mention 'not yet implemented', got: {err_msg}"
    );
}

// ─── Patch 11 § P2 #43 — retry + timeout layer ────────────────────
//
// `get_with_retry` is exercised end-to-end in every other pull test
// (default RetryConfig is enabled), so the cases below focus on
// behaviors the happy-path tests do not cover: retryable-error
// classification, eventual retry success, timeout enforcement, and
// retry-budget exhaustion. `FaultyStore` wraps an inner
// `LocalFileSystem` and intercepts `get_opts` to inject failures
// before each delegating call.

use crate::cloud_reader::read_range_chunked;
use crate::retry::{backoff_delay, contains_http_status, is_retryable, RetryingStore};
use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::local::LocalFileSystem;
use object_store::{
    GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions, PutOptions, PutPayload,
    PutResult,
};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

/// Fault injection mode applied to the next `get_opts` call.
#[derive(Clone, Copy, Debug)]
enum FaultMode {
    /// Return a `Generic` error whose message marks it retryable.
    TransientError,
    /// Sleep slightly longer than the request timeout, then
    /// delegate (the caller will hit `tokio::time::timeout` first).
    SleepBeyondTimeout(Duration),
}

/// Wraps a `LocalFileSystem` and injects a pre-canned schedule of
/// faults into `get_opts` to exercise retry classification, eventual
/// success after transients, and timeout enforcement.
#[derive(Debug)]
struct FaultyStore {
    inner: LocalFileSystem,
    /// Pop one fault per `get_opts` call; once exhausted, calls
    /// delegate to `inner` and succeed.
    faults: std::sync::Mutex<std::collections::VecDeque<FaultMode>>,
    get_attempts: AtomicUsize,
}

impl FaultyStore {
    fn new(inner: LocalFileSystem, faults: Vec<FaultMode>) -> Self {
        Self {
            inner,
            faults: std::sync::Mutex::new(faults.into()),
            get_attempts: AtomicUsize::new(0),
        }
    }
}

impl std::fmt::Display for FaultyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FaultyStore(inner={})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for FaultyStore {
    async fn put_opts(
        &self,
        location: &ObjPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjPath,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &ObjPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.get_attempts.fetch_add(1, AtomicOrdering::Relaxed);
        let next = self.faults.lock().unwrap().pop_front();
        match next {
            Some(FaultMode::TransientError) => Err(object_store::Error::Generic {
                store: "FaultyStore",
                source: "synthetic 503 Service Unavailable".into(),
            }),
            Some(FaultMode::SleepBeyondTimeout(d)) => {
                tokio::time::sleep(d).await;
                self.inner.get_opts(location, options).await
            }
            None => self.inner.get_opts(location, options).await,
        }
    }

    async fn delete(&self, location: &ObjPath) -> object_store::Result<()> {
        self.inner.delete(location).await
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjPath>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &ObjPath, to: &ObjPath) -> object_store::Result<()> {
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &ObjPath, to: &ObjPath) -> object_store::Result<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}

#[test]
fn is_retryable_distinguishes_permanent_from_transient() {
    // NotFound and AlreadyExists are permanent — no retry.
    let nf = object_store::Error::NotFound {
        path: "x".into(),
        source: "missing".into(),
    };
    assert!(!is_retryable(&nf));

    let ae = object_store::Error::AlreadyExists {
        path: "y".into(),
        source: "dup".into(),
    };
    assert!(!is_retryable(&ae));

    // Generic errors whose message mentions 503/throttle/connection
    // are retryable.
    let g503 = object_store::Error::Generic {
        store: "test",
        source: "503 service unavailable".into(),
    };
    assert!(is_retryable(&g503));

    let g_throttle = object_store::Error::Generic {
        store: "test",
        source: "request throttled".into(),
    };
    assert!(is_retryable(&g_throttle));
}

#[test]
fn is_retryable_treats_missing_object_as_permanent() {
    // Regression: a missing LOCAL file surfaces not as the canonical
    // `NotFound` but as a `Generic` wrapping a `std::io::Error` of kind
    // `NotFound`. Such a Generic must be classified permanent so the
    // open_cloud `_catalog.bin` probe fails fast (no full backoff budget) and
    // the actionable `CatalogNotFound` mapping survives.
    let g_notfound = object_store::Error::Generic {
        store: "test",
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no such file",
        )),
    };
    assert!(!is_retryable(&g_notfound));

    // Mirror LocalFileSystem's actual shape: the io NotFound sits one level
    // deeper, behind an `UnableToCanonicalize`-style wrapper whose `source()`
    // returns the io error. `is_missing_object` walks the chain, so this must
    // also be permanent.
    #[derive(Debug)]
    struct Wrapper(std::io::Error);
    impl std::fmt::Display for Wrapper {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "unable to canonicalize")
        }
    }
    impl std::error::Error for Wrapper {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }
    let g_nested = object_store::Error::Generic {
        store: "test",
        source: Box::new(Wrapper(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing",
        ))),
    };
    assert!(!is_retryable(&g_nested));

    // Guard against over-broadening: a transient Generic with no io NotFound in
    // its source chain is still retryable.
    let g_io_transient = object_store::Error::Generic {
        store: "test",
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "connection reset by peer",
        )),
    };
    assert!(is_retryable(&g_io_transient));
}

#[test]
fn is_retryable_treats_unknown_config_key_as_permanent() {
    // A bad object_store config key can never succeed on retry — classifying
    // it as transient would burn the full backoff budget (~31s) on a
    // guaranteed failure. Must be permanent.
    let e = object_store::Error::UnknownConfigurationKey {
        store: "test",
        key: "bogus_option".into(),
    };
    assert!(!is_retryable(&e));

    // Regression: the Display string embeds the key, so a key literally
    // containing a transient token (`timeout`, `500`, …) must still be
    // permanent — it has to be matched before the substring heuristics, not
    // by the final `matches!` arm.
    for key in [
        "connect_timeout",
        "retry_500",
        "server_error_mode",
        "throttle_limit",
    ] {
        let e = object_store::Error::UnknownConfigurationKey {
            store: "test",
            key: key.into(),
        };
        assert!(
            !is_retryable(&e),
            "config key {key:?} must be permanent despite its transient-looking token",
        );
    }
}

#[test]
fn contains_http_status_requires_digit_boundaries() {
    // Genuine status wording matches.
    assert!(contains_http_status(
        "status: 503 service unavailable",
        "503"
    ));
    assert!(contains_http_status("http error 500", "500"));
    assert!(contains_http_status("502", "502")); // whole-string
    assert!(contains_http_status("(504 gateway timeout)", "504"));

    // Digit runs that merely *contain* the code must NOT match — these are
    // the byte counts / offsets that the old bare-substring test misread as
    // 5xx statuses.
    assert!(!contains_http_status(
        "read 15000 bytes (offset 5030)",
        "500"
    ));
    assert!(!contains_http_status(
        "read 15000 bytes (offset 5030)",
        "503"
    ));
    assert!(!contains_http_status("transferred 25022 bytes", "502"));
    assert!(!contains_http_status("chunk 5041 of 9000", "504"));
}

#[test]
fn default_retry_budget_is_six() {
    // Guards the intentional bump from 3 → 6: the multi-shard query fan-out
    // needs headroom, but the value is kept modest because this is the shared
    // default for every read path.
    assert_eq!(RetryConfig::default().max_retries, 6);
}

#[tokio::test]
async fn read_range_chunked_reassembles_byte_identical() {
    // A section larger than the chunk size must reassemble byte-for-byte,
    // including a non-zero start offset and a final short chunk.
    let dir = tempfile::tempdir().unwrap();
    let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(dir.path().join("file.bin"), &data).unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    let store = RetryingStore::new(Arc::new(inner), RetryConfig::default());
    let path = ObjPath::from("file.bin");

    // Whole section, chunk=128 → 8 chunks (last short).
    let whole = read_range_chunked(&store, &path, 0, 1000, 128, 4)
        .await
        .unwrap();
    assert_eq!(whole, data, "chunked whole-section read must match source");

    // Sub-range with a non-zero start, chunk=100 → 9 chunks.
    let sub = read_range_chunked(&store, &path, 50, 950, 100, 4)
        .await
        .unwrap();
    assert_eq!(
        sub,
        data[50..950],
        "chunked sub-range read must match slice"
    );

    // Read smaller than the chunk takes the single-GET fast path.
    let small = read_range_chunked(&store, &path, 10, 30, 128, 4)
        .await
        .unwrap();
    assert_eq!(small, data[10..30]);
}

#[tokio::test]
async fn read_range_chunked_recovers_transient_chunk_failure() {
    // A transient body error on a chunk must be retried per-chunk; the full
    // section still reassembles correctly. This is the atlas-scale
    // predicate-index download made resilient: pre-fix the whole multi-GB GET
    // failed on one body reset.
    let dir = tempfile::tempdir().unwrap();
    let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(dir.path().join("file.bin"), &data).unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    // Two transient failures hit two of the chunk GETs; each is retried.
    let faulty = Arc::new(FaultyStore::new(
        inner,
        vec![FaultMode::TransientError, FaultMode::TransientError],
    ));
    let cfg = RetryConfig {
        max_retries: 3,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
        request_timeout: Duration::from_secs(10),
    };
    let store = RetryingStore::new(faulty.clone(), cfg);
    let path = ObjPath::from("file.bin");

    let got = read_range_chunked(&store, &path, 0, 1000, 100, 4)
        .await
        .unwrap();
    assert_eq!(
        got, data,
        "section must reassemble despite transient chunk failures"
    );
    // 10 chunks + 2 retries = 12 backend hits.
    assert_eq!(
        faulty.get_attempts.load(AtomicOrdering::Relaxed),
        12,
        "10 chunk GETs + 2 retried transients"
    );
}

#[test]
fn backoff_delay_grows_exponentially_and_clamps() {
    let cfg = RetryConfig {
        max_retries: 5,
        base_delay: Duration::from_millis(100),
        max_delay: Duration::from_millis(800),
        jitter_factor: 0.0, // deterministic
        request_timeout: Duration::from_secs(1),
    };
    // attempt 1 → ~100 ms, attempt 2 → ~200 ms, … attempt 4 → ~800 ms,
    // attempt 5 → still 800 ms (clamped).
    let d1 = backoff_delay(&cfg, 1);
    let d2 = backoff_delay(&cfg, 2);
    let d3 = backoff_delay(&cfg, 3);
    let d4 = backoff_delay(&cfg, 4);
    let d5 = backoff_delay(&cfg, 5);
    assert_eq!(d1, Duration::from_millis(100));
    assert_eq!(d2, Duration::from_millis(200));
    assert_eq!(d3, Duration::from_millis(400));
    assert_eq!(d4, Duration::from_millis(800));
    assert_eq!(d5, Duration::from_millis(800), "should clamp at max_delay");
}

#[tokio::test]
async fn get_with_retry_retries_transient_errors() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file.bin"), b"hello world").unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    // Two transient failures, then success on the third call.
    let faulty = Arc::new(FaultyStore::new(
        inner,
        vec![FaultMode::TransientError, FaultMode::TransientError],
    ));
    let cfg = RetryConfig {
        max_retries: 3,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
        request_timeout: Duration::from_secs(10),
    };
    let store = RetryingStore::new(faulty.clone(), cfg);
    let path = ObjPath::from("file.bin");
    let bytes: Bytes = get_with_retry(&store, &path).await.unwrap();
    assert_eq!(&*bytes, b"hello world");
    assert_eq!(
        faulty.get_attempts.load(AtomicOrdering::Relaxed),
        3,
        "should have hit the backend exactly 3 times (2 fail + 1 success)"
    );
}

#[tokio::test]
async fn get_with_retry_returns_download_failed_when_budget_exhausted() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file.bin"), b"hello").unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    // 4 transient failures, budget allows max_retries=2 → total 3
    // attempts, all fail.
    let faulty = Arc::new(FaultyStore::new(
        inner,
        vec![
            FaultMode::TransientError,
            FaultMode::TransientError,
            FaultMode::TransientError,
            FaultMode::TransientError,
        ],
    ));
    let cfg = RetryConfig {
        max_retries: 2,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
        request_timeout: Duration::from_secs(10),
    };
    let store = RetryingStore::new(faulty.clone(), cfg);
    let path = ObjPath::from("file.bin");
    let err = get_with_retry(&store, &path).await.unwrap_err();
    match err {
        CloudError::DownloadFailed { retries, message } => {
            assert_eq!(retries, 2);
            assert!(
                message.contains("file.bin"),
                "message should name the path: {message}"
            );
        }
        other => panic!("expected DownloadFailed, got {other}"),
    }
    // 1 initial + 2 retries = 3 backend hits.
    assert_eq!(faulty.get_attempts.load(AtomicOrdering::Relaxed), 3);
}

#[tokio::test]
async fn get_with_retry_enforces_request_timeout() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file.bin"), b"hello").unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    // Backend sleeps 200 ms; request_timeout is 50 ms → first
    // attempt times out, second sleep schedule is empty so the
    // retry succeeds.
    let faulty = Arc::new(FaultyStore::new(
        inner,
        vec![FaultMode::SleepBeyondTimeout(Duration::from_millis(200))],
    ));
    let cfg = RetryConfig {
        max_retries: 2,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
        request_timeout: Duration::from_millis(50),
    };
    let store = RetryingStore::new(faulty.clone(), cfg);
    let path = ObjPath::from("file.bin");
    let bytes = get_with_retry(&store, &path).await.unwrap();
    assert_eq!(&*bytes, b"hello");
}

// LC1: the cloud query path's range reads get the same retry budget as
// pull. A transient failure is retried to success.
#[tokio::test]
async fn get_range_with_retry_retries_transient_errors() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file.bin"), b"hello world").unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    let faulty = Arc::new(FaultyStore::new(
        inner,
        vec![FaultMode::TransientError, FaultMode::TransientError],
    ));
    let cfg = RetryConfig {
        max_retries: 3,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
        request_timeout: Duration::from_secs(10),
    };
    let store = RetryingStore::new(faulty.clone(), cfg);
    let path = ObjPath::from("file.bin");
    let bytes = get_range_with_retry(&store, &path, 0..5).await.unwrap();
    assert_eq!(&*bytes, b"hello");
    assert_eq!(faulty.get_attempts.load(AtomicOrdering::Relaxed), 3);
}

// LC1: a range read that always times out surfaces CloudError::Timeout
// rather than hanging the parallel decode without a deadline.
#[tokio::test]
async fn get_range_with_retry_enforces_timeout() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file.bin"), b"hello world").unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    // Sleep beyond the timeout on every attempt (1 initial + 1 retry).
    let faulty = Arc::new(FaultyStore::new(
        inner,
        vec![
            FaultMode::SleepBeyondTimeout(Duration::from_millis(200)),
            FaultMode::SleepBeyondTimeout(Duration::from_millis(200)),
        ],
    ));
    let cfg = RetryConfig {
        max_retries: 1,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
        request_timeout: Duration::from_millis(50),
    };
    let store = RetryingStore::new(faulty.clone(), cfg);
    let path = ObjPath::from("file.bin");
    let err = get_range_with_retry(&store, &path, 0..5).await.unwrap_err();
    assert!(
        matches!(err, CloudError::Timeout { .. }),
        "expected Timeout, got {err}"
    );
}

// ─── Patch 11 § P2 #44 — hash-while-write equivalence ────────────
//
// The new `pull()` writes through a `HashingWriter` and patches the
// file_checksum field at the end. The convention is "hash of the
// file with file_checksum=0 in the header" (matching writer.rs's
// pattern). This test verifies the stored checksum equals what a
// fresh re-read with the zeroed-checksum convention produces.

#[tokio::test]
async fn pull_checksum_matches_zeroed_header_rehash() {
    let dir = tempfile::tempdir().unwrap();
    let output = pull_from_exploded(&dir, 100, 50, 4).await;

    // Read the on-disk header and stash the stored checksum.
    let mut file = std::fs::File::open(&output).unwrap();
    let mut header_buf = [0u8; HEADER_SIZE];
    std::io::Read::read_exact(&mut file, &mut header_buf).unwrap();
    let on_disk_header = FileHeader::read_from(&mut Cursor::new(&header_buf[..])).unwrap();
    let stored = on_disk_header.file_checksum;
    assert_ne!(stored, 0, "file_checksum must be non-zero in a valid file");

    // Re-hash the file with the file_checksum field zeroed out (the
    // verification convention used by writer.rs).
    let mut zeroed_header = on_disk_header;
    zeroed_header.file_checksum = 0;
    let mut zeroed_bytes = Vec::with_capacity(HEADER_SIZE);
    zeroed_header.write_to(&mut zeroed_bytes).unwrap();

    let mut hasher = blake3::Hasher::new();
    hasher.update(&zeroed_bytes);
    let mut chunk = [0u8; 65536];
    let mut file = std::fs::File::open(&output).unwrap();
    std::io::Seek::seek(&mut file, SeekFrom::Start(HEADER_SIZE as u64)).unwrap();
    loop {
        let n = std::io::Read::read(&mut file, &mut chunk).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
    }
    let recomputed = scx_format_io::checksum::truncate_hash_to_u64(&hasher.finalize());
    assert_eq!(
        stored, recomputed,
        "hash-while-write checksum must match the zeroed-header rehash convention"
    );
}

#[tokio::test]
async fn get_with_retry_surfaces_timeout_when_all_attempts_hang() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file.bin"), b"hello").unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    // Every attempt sleeps longer than the per-request timeout.
    let faulty = Arc::new(FaultyStore::new(
        inner,
        vec![
            FaultMode::SleepBeyondTimeout(Duration::from_millis(200)),
            FaultMode::SleepBeyondTimeout(Duration::from_millis(200)),
        ],
    ));
    let cfg = RetryConfig {
        max_retries: 1,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
        request_timeout: Duration::from_millis(50),
    };
    let store = RetryingStore::new(faulty.clone(), cfg);
    let path = ObjPath::from("file.bin");
    let err = get_with_retry(&store, &path).await.unwrap_err();
    match err {
        CloudError::Timeout { path: p, .. } => assert!(p.contains("file.bin")),
        other => panic!("expected Timeout, got {other}"),
    }
}

/// Regression: when retries time out, the prior transient error
/// (here a synthetic 503) must propagate through
/// `CloudError::Timeout::last_error` so on-call can diagnose what
/// was actually failing — not just "timed out".
#[tokio::test]
async fn get_with_retry_timeout_surfaces_last_transient() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file.bin"), b"hello").unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    // First call: synthetic 503 (retryable). Second call: sleep
    // past the timeout. Budget allows only one retry, so the
    // second attempt's timeout exhausts it.
    let faulty = Arc::new(FaultyStore::new(
        inner,
        vec![
            FaultMode::TransientError,
            FaultMode::SleepBeyondTimeout(Duration::from_millis(200)),
        ],
    ));
    let cfg = RetryConfig {
        max_retries: 1,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
        request_timeout: Duration::from_millis(50),
    };
    let store = RetryingStore::new(faulty.clone(), cfg);
    let path = ObjPath::from("file.bin");
    let err = get_with_retry(&store, &path).await.unwrap_err();
    match err {
        CloudError::Timeout {
            last_error: Some(msg),
            ..
        } => {
            let lower = msg.to_ascii_lowercase();
            assert!(
                lower.contains("503") || lower.contains("service unavailable"),
                "last_error must carry the prior 503 message, got: {msg}"
            );
        }
        CloudError::Timeout {
            last_error: None, ..
        } => panic!("expected last_error to carry prior 503 message, got None"),
        other => panic!("expected Timeout, got {other}"),
    }
}

/// LC1: the raw `backend.get` calls in `open_cloud` (catalog/header
/// detection, packed front/EOF catalog reads) inherit the decorator's
/// retry budget without going through the `get_with_retry` helper —
/// proving resilience is installed at the backend boundary, not per call
/// site. Drives `ObjectStore::get` directly on the wrapped store.
#[tokio::test]
async fn retrying_store_inherited_by_raw_get() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file.bin"), b"hello world").unwrap();
    let inner = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    let faulty = Arc::new(FaultyStore::new(
        inner,
        vec![FaultMode::TransientError, FaultMode::TransientError],
    ));
    let cfg = RetryConfig {
        max_retries: 3,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
        request_timeout: Duration::from_secs(10),
    };
    let store = RetryingStore::new(faulty.clone(), cfg);
    let path = ObjPath::from("file.bin");
    // No `get_with_retry` helper — call the store API directly.
    let got = ObjectStore::get(&store, &path).await.unwrap();
    let bytes = got.bytes().await.unwrap();
    assert_eq!(&*bytes, b"hello world");
    assert_eq!(
        faulty.get_attempts.load(AtomicOrdering::Relaxed),
        3,
        "raw get must inherit the decorator's retry budget (2 fail + 1 success)"
    );
}
