// I/O Stage (Stage 1) — async shard group reader for the training pipeline.
//
// Reads shard groups from the SCX file using tokio async I/O and sends
// decoded shard data to the decode stage via bounded channel.
// See docs/multithreading.md §Training data loader (triple-buffered pipeline).

use std::sync::Arc;
use std::time::Instant;

use roaring::RoaringBitmap;
use scx_format::reader::ScxReader;

use crate::budget::profiling_enabled;
use crate::error::{LoaderError, Result};

// Compile-time assertion: ScxReader must be Send + Sync for Arc sharing
// across tokio tasks and rayon threads.
const _: () = {
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn _check() {
        assert_send_sync::<ScxReader>();
    }
};

/// A group of decoded CSR shards ready for the decode stage.
#[derive(Debug)]
pub struct ShardGroup {
    /// Per-shard data: decoded CSR arrays + metadata.
    pub shards: Vec<ShardData>,
}

/// Decoded CSR data for a single shard, with deletion metadata.
#[derive(Debug)]
pub struct ShardData {
    /// CSR indptr array (i64, scipy-compatible).
    pub indptr: Vec<i64>,
    /// CSR column indices (i32, scipy-compatible).
    pub indices: Vec<i32>,
    /// CSR values (f32, scipy-compatible).
    pub data: Vec<f32>,
    /// First global row index of this shard (from catalog entry `stats.row_start`).
    pub global_row_offset: u64,
    /// Number of rows in this shard (from shard header `n_major` /
    /// catalog entry `stats.row_end - row_start`).
    pub n_rows: u32,
    /// Deletion bitmap for this shard (None if no deletions).
    /// Local row indices that have been logically deleted.
    pub deleted_rows: Option<RoaringBitmap>,
}

/// Run the I/O stage: read shard groups and send them to the decode stage.
///
/// For each shard group in `shard_groups`:
/// - Resolves shard indices to catalog entries via `reader.catalog().shards_sorted()`
/// - Reads each shard sequentially within the group (disk-sequential I/O)
/// - Filters fully-deleted shards; attaches deletion bitmaps to partially-deleted shards
/// - Sends the `ShardGroup` via bounded channel (back-pressure blocks automatically)
///
/// After all groups are sent, drops the sender to signal end-of-epoch.
///
/// Uses `tokio::task::spawn_blocking()` for the blocking shard reads since
/// `ScxReader` uses mmap (blocking) I/O.
pub async fn io_stage(
    reader: Arc<ScxReader>,
    shard_groups: Vec<Vec<usize>>,
    deletion_vectors: Option<scx_format::deletion_vectors::DeletionVectors>,
    tx: tokio::sync::mpsc::Sender<ShardGroup>,
) -> Result<()> {
    // Pre-compute the sorted shard catalog entries. The shard indices in
    // `shard_groups` refer to positions in this sorted list.
    let sorted_entries = reader.catalog().shards_sorted();
    let n_shards = sorted_entries.len();

    // Pre-compute per-shard deletion bitmaps for O(1) lookup.
    // Map: shard_index (position in sorted list) → RoaringBitmap of deleted local rows.
    let deletion_map: Arc<std::collections::HashMap<usize, RoaringBitmap>> =
        Arc::new(match &deletion_vectors {
            Some(dv) => {
                let mut map = std::collections::HashMap::new();
                for (&shard_id, bitmap) in &dv.shards {
                    let shard_idx = shard_id as usize;
                    if shard_idx < n_shards && !bitmap.is_empty() {
                        map.insert(shard_idx, bitmap.clone());
                    }
                }
                map
            }
            None => std::collections::HashMap::new(),
        });

    // Pre-compute per-group byte ranges for coalesced MADV_WILLNEED prefetch.
    // Each entry is (min_offset, max_end) covering all shards in the group.
    #[cfg(unix)]
    let group_byte_ranges: Vec<(usize, usize)> = shard_groups
        .iter()
        .map(|group_indices| {
            let mut min_offset = usize::MAX;
            let mut max_end = 0usize;
            for &idx in group_indices {
                if let Some(entry) = sorted_entries.get(idx) {
                    let start = entry.offset as usize;
                    let end = start + entry.length as usize;
                    min_offset = min_offset.min(start);
                    max_end = max_end.max(end);
                }
            }
            (min_offset, max_end)
        })
        .collect();

    let profile = profiling_enabled();
    let io_start = Instant::now();
    let mut group_count = 0usize;
    let n_groups = shard_groups.len();

    for group_indices in shard_groups {
        // Prefetch upcoming groups with MADV_WILLNEED (look ahead 2 groups).
        #[cfg(unix)]
        {
            let lookahead = 2;
            for ahead in 1..=lookahead {
                let future_idx = group_count + ahead;
                if future_idx < n_groups {
                    let (start, end) = group_byte_ranges[future_idx];
                    if start < end {
                        reader.advise_willneed(start, end - start);
                    }
                }
            }
        }

        // Capture pre-computed byte range for this group's MADV_WILLNEED hint.
        #[cfg(unix)]
        let current_byte_range = group_byte_ranges[group_count];

        // Clone Arc handles for the spawn_blocking closure.
        let reader = Arc::clone(&reader);
        let deletion_map = Arc::clone(&deletion_map);
        let group_num = group_count;
        group_count += 1;

        // Perform blocking shard reads inside spawn_blocking.
        let group = tokio::task::spawn_blocking(move || -> Result<ShardGroup> {
            let t0 = Instant::now();
            let sorted = reader.catalog().shards_sorted();
            let mut shards = Vec::with_capacity(group_indices.len());

            // Issue a coalesced MADV_WILLNEED for this group's byte range.
            // Uses the pre-computed range to avoid reiterating over group_indices.
            #[cfg(unix)]
            {
                let (min_offset, max_end) = current_byte_range;
                if min_offset < max_end {
                    reader.advise_willneed(min_offset, max_end - min_offset);
                }
            }

            for &shard_idx in &group_indices {
                if shard_idx >= sorted.len() {
                    return Err(LoaderError::ConfigError {
                        reason: format!(
                            "shard index {} out of bounds (file has {} shards)",
                            shard_idx,
                            sorted.len()
                        ),
                    });
                }

                let entry = sorted[shard_idx];

                // Extract row metadata from catalog stats.
                let stats = entry
                    .stats
                    .as_ref()
                    .ok_or_else(|| LoaderError::ConfigError {
                        reason: format!(
                            "shard '{}' has no stats (row_start/row_end unavailable)",
                            entry.name
                        ),
                    })?;

                let global_row_offset = stats.row_start;
                let n_rows = if stats.row_end < stats.row_start {
                    return Err(LoaderError::ConfigError {
                        reason: format!(
                            "shard '{}' has row_end ({}) < row_start ({})",
                            entry.name, stats.row_end, stats.row_start
                        ),
                    });
                } else {
                    let n = stats.row_end - stats.row_start;
                    u32::try_from(n).map_err(|_| LoaderError::ConfigError {
                        reason: format!("shard '{}' row count {} exceeds u32::MAX", entry.name, n),
                    })?
                };

                // Check deletion status.
                let deleted_bitmap = deletion_map.get(&shard_idx).cloned();

                // Skip fully-deleted shards.
                if let Some(ref bm) = deleted_bitmap {
                    if bm.len() >= n_rows as u64 {
                        // All rows deleted — exclude this shard entirely.
                        continue;
                    }
                }

                // Decode the shard CSR data (skip checksum for throughput —
                // data integrity verified at file open or via explicit validate()).
                let (indptr, indices, data) = reader.read_shard_from_entry(entry)?;

                shards.push(ShardData {
                    indptr,
                    indices,
                    data,
                    global_row_offset,
                    n_rows,
                    deleted_rows: deleted_bitmap,
                });
            }

            let profile_inner = std::env::var("SCX_LOADER_PROFILE")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false);
            if profile_inner {
                let total_rows: u32 = shards.iter().map(|s| s.n_rows).sum();
                eprintln!(
                    "[scx-loader profile] io_stage group {group_num}: {:?} ({} shards, {} rows)",
                    t0.elapsed(),
                    shards.len(),
                    total_rows
                );
            }

            Ok(ShardGroup { shards })
        })
        .await
        .map_err(|e| LoaderError::ShutdownError(format!("I/O stage task panicked: {e}")))??;

        // Send the group via bounded channel (blocks if full = back-pressure).
        let t_send = Instant::now();
        tx.send(group).await.map_err(|e| {
            LoaderError::ChannelError(format!("I/O stage: failed to send shard group: {e}"))
        })?;
        if profile {
            eprintln!(
                "[scx-loader profile] io_stage group {group_num} send wait: {:?}",
                t_send.elapsed()
            );
        }
    }

    if profile {
        eprintln!(
            "[scx-loader profile] io_stage total: {:?} ({group_count} groups)",
            io_start.elapsed()
        );
    }

    // Drop sender implicitly when function returns → signals end-of-epoch.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_codec::CodecId;
    use scx_format::header::FileHeader;
    use scx_format::writer::ScxWriter;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc as StdArc;

    use scx_format::header::MAGIC;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0,
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

    fn sample_obs(n: usize) -> arrow::record_batch::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        arrow::record_batch::RecordBatch::try_new(
            StdArc::new(schema),
            vec![StdArc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> arrow::record_batch::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        arrow::record_batch::RecordBatch::try_new(
            StdArc::new(schema),
            vec![StdArc::new(StringArray::from(
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

    /// Write a multi-shard test file.
    fn write_test_file(
        dir: &tempfile::TempDir,
        filename: &str,
        n_obs: usize,
        n_vars: usize,
        n_shards: usize,
    ) -> std::path::PathBuf {
        let path = dir.path().join(filename);
        let total_nnz = n_obs * 2;
        let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();

        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let rows_per_shard = n_obs / n_shards;
        for s in 0..n_shards {
            let shard_rows = if s == n_shards - 1 {
                n_obs - rows_per_shard * s
            } else {
                rows_per_shard
            };
            let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    scx_codec::ValueEncoding::Uint8,
                    (s * rows_per_shard) as u64,
                )
                .unwrap();
        }

        writer.finish().unwrap();
        path
    }

    // -----------------------------------------------------------------
    // C1 Tests
    // -----------------------------------------------------------------

    #[test]
    fn test_io_stage_sends_all_groups() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let path = write_test_file(&dir, "test.scx", 30, 10, 3);
            let reader = Arc::new(ScxReader::open(&path).unwrap());

            // 3 shards, group_size=2 → groups: [0,1], [2]
            let shard_groups = vec![vec![0, 1], vec![2]];

            let (tx, mut rx) = tokio::sync::mpsc::channel(8);
            let handle = tokio::spawn(io_stage(reader, shard_groups, None, tx));

            let mut received_groups = Vec::new();
            while let Some(group) = rx.recv().await {
                received_groups.push(group);
            }

            handle.await.unwrap().unwrap();

            assert_eq!(received_groups.len(), 2);
            assert_eq!(received_groups[0].shards.len(), 2);
            assert_eq!(received_groups[1].shards.len(), 1);

            // Verify all shards' total rows match n_obs
            let total_rows: u32 = received_groups
                .iter()
                .flat_map(|g| g.shards.iter())
                .map(|s| s.n_rows)
                .sum();
            assert_eq!(total_rows, 30);
        });
    }

    #[test]
    fn test_shard_data_correct() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let path = write_test_file(&dir, "test.scx", 10, 5, 1);
            let reader = Arc::new(ScxReader::open(&path).unwrap());

            // Read via io_stage
            let shard_groups = vec![vec![0]];
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let r2 = Arc::clone(&reader);
            let handle = tokio::spawn(io_stage(r2, shard_groups, None, tx));

            let group = rx.recv().await.unwrap();
            handle.await.unwrap().unwrap();

            assert_eq!(group.shards.len(), 1);
            let shard = &group.shards[0];

            // Read directly via ScxReader for comparison
            let (expected_indptr, expected_indices, expected_data) =
                reader.read_csr_shard(0).unwrap();

            assert_eq!(shard.indptr, expected_indptr);
            assert_eq!(shard.indices, expected_indices);
            assert_eq!(shard.data, expected_data);
            assert_eq!(shard.global_row_offset, 0);
            assert_eq!(shard.n_rows, 10);
            assert!(shard.deleted_rows.is_none());
        });
    }

    #[test]
    fn test_backpressure() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Channel capacity of 1: I/O stage must block until consumed.
            let dir = tempfile::tempdir().unwrap();
            let path = write_test_file(&dir, "test.scx", 30, 10, 3);
            let reader = Arc::new(ScxReader::open(&path).unwrap());

            let shard_groups = vec![vec![0], vec![1], vec![2]];
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);

            let handle = tokio::spawn(io_stage(reader, shard_groups, None, tx));

            // Consume one at a time
            let mut count = 0;
            while let Some(_group) = rx.recv().await {
                count += 1;
            }

            handle.await.unwrap().unwrap();
            assert_eq!(count, 3);
        });
    }

    #[test]
    fn test_single_shard_file() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let path = write_test_file(&dir, "single.scx", 5, 3, 1);
            let reader = Arc::new(ScxReader::open(&path).unwrap());

            let shard_groups = vec![vec![0]];
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let handle = tokio::spawn(io_stage(reader, shard_groups, None, tx));

            let group = rx.recv().await.unwrap();
            assert_eq!(group.shards.len(), 1);
            assert_eq!(group.shards[0].n_rows, 5);

            // Channel should close after all groups sent
            assert!(rx.recv().await.is_none());
            handle.await.unwrap().unwrap();
        });
    }

    #[test]
    fn test_fully_deleted_shard_excluded() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            // 2 shards with 5 rows each
            let path = write_test_file(&dir, "del.scx", 10, 5, 2);
            let reader = Arc::new(ScxReader::open(&path).unwrap());

            // Delete ALL rows in shard 0 (local rows 0..5)
            let mut dv = scx_format::deletion_vectors::DeletionVectors::new();
            let mut bm = RoaringBitmap::new();
            for i in 0..5u32 {
                bm.insert(i);
            }
            dv.shards.insert(0, bm);

            let shard_groups = vec![vec![0, 1]];
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let handle = tokio::spawn(io_stage(reader, shard_groups, Some(dv), tx));

            let group = rx.recv().await.unwrap();
            handle.await.unwrap().unwrap();

            // Shard 0 should be excluded (fully deleted), only shard 1 remains
            assert_eq!(group.shards.len(), 1);
            assert_eq!(group.shards[0].global_row_offset, 5); // shard 1 starts at row 5
            assert_eq!(group.shards[0].n_rows, 5);
            assert!(group.shards[0].deleted_rows.is_none());
        });
    }

    #[test]
    fn test_partial_deletion_bitmap() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let path = write_test_file(&dir, "partial.scx", 10, 5, 2);
            let reader = Arc::new(ScxReader::open(&path).unwrap());

            // Delete rows 1 and 3 (local) from shard 0
            let mut dv = scx_format::deletion_vectors::DeletionVectors::new();
            let mut bm = RoaringBitmap::new();
            bm.insert(1);
            bm.insert(3);
            dv.shards.insert(0, bm);

            let shard_groups = vec![vec![0, 1]];
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let handle = tokio::spawn(io_stage(reader, shard_groups, Some(dv), tx));

            let group = rx.recv().await.unwrap();
            handle.await.unwrap().unwrap();

            // Both shards should be present
            assert_eq!(group.shards.len(), 2);

            // Shard 0 should have the deletion bitmap
            let s0 = &group.shards[0];
            assert_eq!(s0.global_row_offset, 0);
            assert!(s0.deleted_rows.is_some());
            let bm = s0.deleted_rows.as_ref().unwrap();
            assert!(bm.contains(1));
            assert!(bm.contains(3));
            assert!(!bm.contains(0));
            assert!(!bm.contains(2));
            assert_eq!(bm.len(), 2);

            // Shard 1 should have no deletions
            assert!(group.shards[1].deleted_rows.is_none());
        });
    }

    // -----------------------------------------------------------------
    // C2 Tests: ScxReader Arc sharing
    // -----------------------------------------------------------------

    #[test]
    fn test_reader_arc_concurrent() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, "concurrent.scx", 20, 10, 2);
        let reader = Arc::new(ScxReader::open(&path).unwrap());

        // Spawn multiple threads reading different shards concurrently
        let handles: Vec<_> = (0..2)
            .map(|shard_idx| {
                let r = Arc::clone(&reader);
                std::thread::spawn(move || r.read_csr_shard(shard_idx).unwrap())
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Both should return valid data
        assert_eq!(results.len(), 2);
        // Shard 0: 10 rows → indptr has 11 elements
        assert_eq!(results[0].0.len(), 11);
        // Shard 1: 10 rows → indptr has 11 elements
        assert_eq!(results[1].0.len(), 11);
    }

    // -----------------------------------------------------------------
    // 3F Tests: Prefetch scheduling
    // -----------------------------------------------------------------

    #[test]
    fn test_io_stage_offset_sorted_order() {
        // Verify that when shard groups are sorted by offset, shards arrive
        // in the expected order.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let path = write_test_file(&dir, "sorted.scx", 40, 10, 4);
            let reader = Arc::new(ScxReader::open(&path).unwrap());

            // Feed groups already sorted by offset: [0,1], [2,3]
            let shard_groups = vec![vec![0, 1], vec![2, 3]];
            let (tx, mut rx) = tokio::sync::mpsc::channel(8);
            let handle = tokio::spawn(io_stage(reader, shard_groups, None, tx));

            let mut received = Vec::new();
            while let Some(group) = rx.recv().await {
                received.push(group);
            }
            handle.await.unwrap().unwrap();

            assert_eq!(received.len(), 2);
            // First group: shards 0 and 1 (offsets ascending)
            assert_eq!(received[0].shards.len(), 2);
            assert!(
                received[0].shards[0].global_row_offset < received[0].shards[1].global_row_offset
            );
            // Second group: shards 2 and 3
            assert_eq!(received[1].shards.len(), 2);
            assert!(
                received[1].shards[0].global_row_offset < received[1].shards[1].global_row_offset
            );
            // Group 1 offsets should be after group 0
            assert!(
                received[0].shards[1].global_row_offset < received[1].shards[0].global_row_offset
            );
        });
    }

    #[test]
    fn test_io_stage_prefetch_does_not_panic() {
        // Smoke test: prefetch hints on a small file should not panic.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let path = write_test_file(&dir, "prefetch.scx", 5, 3, 1);
            let reader = Arc::new(ScxReader::open(&path).unwrap());

            let shard_groups = vec![vec![0]];
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let handle = tokio::spawn(io_stage(reader, shard_groups, None, tx));

            let group = rx.recv().await.unwrap();
            assert_eq!(group.shards.len(), 1);
            assert!(rx.recv().await.is_none());
            handle.await.unwrap().unwrap();
        });
    }
}
