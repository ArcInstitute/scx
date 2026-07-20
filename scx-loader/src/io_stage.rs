// I/O Stage (Stage 1) — async shard group reader for the training pipeline.
//
// Reads shard groups from the SCX file using tokio async I/O and sends
// decoded shard data to the decode stage via bounded channel.
// See docs/multithreading.md §Training data loader (triple-buffered pipeline).

use std::sync::Arc;
use std::time::Instant;

use roaring::RoaringBitmap;
use scx_format_io::reader::ScxReader;

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

/// Reconstruct per-output-position shard-local deletion bitmaps from the
/// (v2, global-obs) deletion vector.
///
/// The on-disk `DeletionVectors` (v2) stores a global obs-row bitmap keyed by
/// `modality_id` (`0` = whole-cell / all modalities). `global_deleted()` is
/// therefore the exact set of deleted *global cell* indices directly — no
/// per-shard remapping needed. We re-bucket that set into `target_ranges` — the
/// output shard list, which is one modality's shards on the per-modality loader
/// path and the global list otherwise.
///
/// - `target_ranges[p]` is the `(row_start, row_end)` of output shard position `p`
///   (`None` skips it). The returned map is keyed by `p`, matching the positions in
///   `shard_groups`, and holds shard-local (`global - row_start`) bitmaps.
/// - `global_row_starts` is retained for signature stability with the caller but
///   is unused under v2 (deletions are already global obs rows); it existed for
///   the legacy v1 per-shard `global_row_starts[gid] + local` remapping.
///
/// For the single-modality/global path (`target_ranges == global` ranges) this is
/// byte-identical to bucketing the global deletion bitmap directly.
fn reconstruct_deletion_map(
    dv: &scx_format_io::deletion_vectors::DeletionVectors,
    global_row_starts: &[Option<u64>],
    target_ranges: &[Option<(u64, u64)>],
) -> std::collections::HashMap<usize, RoaringBitmap> {
    let _ = global_row_starts; // unused under v2 (deletions are already global obs rows).
                               // Recover the set of deleted global cell indices.
    let mut deleted_global: Vec<u64> = match dv.global_deleted() {
        Some(bitmap) => bitmap.iter().map(|r| r as u64).collect(),
        None => Vec::new(),
    };
    if deleted_global.is_empty() {
        return std::collections::HashMap::new();
    }
    deleted_global.sort_unstable();
    deleted_global.dedup();

    // Re-bucket into each target shard's row range as shard-local bitmaps.
    let mut map = std::collections::HashMap::new();
    for (pos, range) in target_ranges.iter().enumerate() {
        let Some((rs, re)) = *range else { continue };
        let lo = deleted_global.partition_point(|&g| g < rs);
        let hi = deleted_global.partition_point(|&g| g < re);
        if hi > lo {
            let mut bm = RoaringBitmap::new();
            for &g in &deleted_global[lo..hi] {
                bm.insert((g - rs) as u32);
            }
            map.insert(pos, bm);
        }
    }
    map
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
    deletion_vectors: Option<scx_format_io::deletion_vectors::DeletionVectors>,
    modality_id: Option<u8>,
    tx: tokio::sync::mpsc::Sender<ShardGroup>,
) -> Result<()> {
    // Pre-compute the sorted shard catalog entries. The shard indices in
    // `shard_groups` refer to positions in this sorted list.
    //
    // Phase H.1: when `modality_id` is set, the entries are filtered to
    // that modality's CSR shards (matching the filter applied in
    // `pipeline.rs::start_epoch`). Per-modality and global runs use the
    // same code path; only the entry list differs.
    let sorted_entries: Vec<&scx_format_io::FullCatalogEntry> = match modality_id {
        Some(mid) => reader.catalog().csr_shards_for_modality(mid),
        None => reader.catalog().shards_sorted(),
    };

    // Pre-compute per-shard deletion bitmaps for O(1) lookup.
    // Map: output shard position (index into `sorted_entries`, matching the
    // positions in `shard_groups`) → RoaringBitmap of deleted *shard-local* rows.
    //
    // The on-disk deletion vector is keyed by GLOBAL shard index (position in
    // `shards_sorted()`, how `scx delete` writes it) with shard-local bitmaps.
    // That losslessly encodes a set of deleted *global cell* indices (each
    // deleted cell is assigned to exactly one global shard). We reconstruct that
    // global set and re-bucket it into `sorted_entries`' row ranges — so the
    // per-modality path (where `sorted_entries` is one modality's shards, whose
    // positions don't match the global DV keys) applies deletions correctly, and
    // all modalities skip the same global cells (alignment preserved). For the
    // single-modality/global path (`sorted_entries == shards_sorted()`) this is
    // byte-identical to a direct copy.
    let deletion_map: Arc<std::collections::HashMap<usize, RoaringBitmap>> =
        Arc::new(match &deletion_vectors {
            Some(dv) => {
                let global = reader.catalog().shards_sorted();
                let global_row_starts: Vec<Option<u64>> = global
                    .iter()
                    .map(|e| e.stats.as_ref().map(|s| s.row_start))
                    .collect();
                let target_ranges: Vec<Option<(u64, u64)>> = sorted_entries
                    .iter()
                    .map(|e| e.stats.as_ref().map(|s| (s.row_start, s.row_end)))
                    .collect();
                reconstruct_deletion_map(dv, &global_row_starts, &target_ranges)
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
                    let end = start.saturating_add(entry.length as usize);
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
    // D0 profiling accumulators (only meaningful when `profile`): the sum of
    // per-group whole-shard **decode** wall vs the sum of per-group `tx.send`
    // back-pressure wait. `decode_total` times the `read_shard_from_entry` call
    // — codec decode (zstd + byte-transforms) + CSR-vector assembly; the dense
    // scatter into batch buffers is a *separate* stage (`decode_stage`). Their
    // ratio is the critical-path verdict — decode ≈ wall & send-wait ≈ 0 ⇒
    // stage-1 decode is the bottleneck; large send-wait ⇒ decode is hidden
    // behind the downstream consumer.
    let mut total_decode_us = 0u128;
    let mut total_send_wait_us = 0u128;

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
        let group_modality_id = modality_id;
        let group =
            tokio::task::spawn_blocking(move || -> Result<(ShardGroup, std::time::Duration)> {
                let t0 = Instant::now();
                let sorted: Vec<&scx_format_io::FullCatalogEntry> = match group_modality_id {
                    Some(mid) => reader.catalog().csr_shards_for_modality(mid),
                    None => reader.catalog().shards_sorted(),
                };
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
                            reason: format!(
                                "shard '{}' row count {} exceeds u32::MAX",
                                entry.name, n
                            ),
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

                // `profile` (read once via `profiling_enabled()` at pipeline scope)
                // is captured by copy into this `move` closure — no per-group env read.
                let decode_elapsed = t0.elapsed();
                if profile {
                    let total_rows: u32 = shards.iter().map(|s| s.n_rows).sum();
                    eprintln!(
                    "[scx-loader profile] io_stage group {group_num}: {:?} ({} shards, {} rows)",
                    decode_elapsed,
                    shards.len(),
                    total_rows
                );
                }

                Ok((ShardGroup { shards }, decode_elapsed))
            })
            .await
            .map_err(|e| LoaderError::ShutdownError(format!("I/O stage task panicked: {e}")))??;
        let (group, decode_elapsed) = group;
        total_decode_us += decode_elapsed.as_micros();

        // Send the group via bounded channel (blocks if full = back-pressure).
        let t_send = Instant::now();
        tx.send(group).await.map_err(|e| {
            LoaderError::ChannelError(format!("I/O stage: failed to send shard group: {e}"))
        })?;
        let send_wait = t_send.elapsed();
        total_send_wait_us += send_wait.as_micros();
        if profile {
            eprintln!("[scx-loader profile] io_stage group {group_num} send wait: {send_wait:?}");
        }
    }

    if profile {
        eprintln!(
            "[scx-loader profile] io_stage total: {:?} ({group_count} groups, \
             decode_total={total_decode_us}µs, send_wait_total={total_send_wait_us}µs)",
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
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc as StdArc;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
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
            let handle = tokio::spawn(io_stage(reader, shard_groups, None, None, tx));

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
            let handle = tokio::spawn(io_stage(r2, shard_groups, None, None, tx));

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

            let handle = tokio::spawn(io_stage(reader, shard_groups, None, None, tx));

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
            let handle = tokio::spawn(io_stage(reader, shard_groups, None, None, tx));

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

            // Delete ALL rows in shard 0 (global rows 0..5; shard row_start 0).
            let mut dv = scx_format_io::deletion_vectors::DeletionVectors::new();
            dv.insert_global(0..5u32);

            let shard_groups = vec![vec![0, 1]];
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let handle = tokio::spawn(io_stage(reader, shard_groups, Some(dv), None, tx));

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

            // Delete global rows 1 and 3 (shard 0, row_start 0).
            let mut dv = scx_format_io::deletion_vectors::DeletionVectors::new();
            dv.insert_global([1u32, 3]);

            let shard_groups = vec![vec![0, 1]];
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let handle = tokio::spawn(io_stage(reader, shard_groups, Some(dv), None, tx));

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
    // Reconstruct_deletion_map — global-cell reconstruction + per-modality
    // re-bucketing (pure, no file fixture).
    // -----------------------------------------------------------------

    fn dv_global(rows: &[u32]) -> scx_format_io::deletion_vectors::DeletionVectors {
        let mut dv = scx_format_io::deletion_vectors::DeletionVectors::new();
        dv.insert_global(rows.iter().copied());
        dv
    }

    /// Single-modality / global: target ranges == global ranges → byte-identical
    /// to a direct copy of the DV bitmaps.
    #[test]
    fn reconstruct_deletion_map_single_modality_identity() {
        // Two shards: [0,5) and [5,10). Delete global rows 1 and 3.
        let global_row_starts = [Some(0u64), Some(5u64)];
        let ranges = [Some((0u64, 5u64)), Some((5u64, 10u64))];
        let dv = dv_global(&[1, 3]);

        let map = reconstruct_deletion_map(&dv, &global_row_starts, &ranges);
        assert_eq!(map.len(), 1);
        let bm = map.get(&0).unwrap();
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![1, 3]);
        assert!(!map.contains_key(&1));
    }

    /// Multimodal: the DV names only modality-A's global shards, but the deleted
    /// cells must still be applied to modality-B's shards (same global cells).
    #[test]
    fn reconstruct_deletion_map_cross_modality() {
        // Global deletes apply to every modality (v2 stores global obs rows).
        // Deleted global cells 1, 3, 5 must land in modality-B's shards too.
        let global_row_starts = [Some(0u64), Some(0u64), Some(5u64), Some(5u64)];
        let dv = dv_global(&[1, 3, 5]);

        // Target = modality B's shards: positions 0=[0,5), 1=[5,10).
        let target_b = [Some((0u64, 5u64)), Some((5u64, 10u64))];
        let map = reconstruct_deletion_map(&dv, &global_row_starts, &target_b);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&0).unwrap().iter().collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(map.get(&1).unwrap().iter().collect::<Vec<_>>(), vec![0]); // cell 5 → local 0
    }

    /// Deleted global rows outside every target range are dropped without
    /// panicking.
    #[test]
    fn reconstruct_deletion_map_skips_out_of_range() {
        let global_row_starts = [Some(0u64), Some(5u64)];
        let ranges = [Some((0u64, 5u64)), Some((5u64, 10u64))];
        // Global row 2 is in range; 99 is beyond all target ranges → dropped.
        let dv = dv_global(&[2, 99]);

        let map = reconstruct_deletion_map(&dv, &global_row_starts, &ranges);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&0).unwrap().iter().collect::<Vec<_>>(), vec![2]);
    }

    /// A fully-deleted shard yields a bitmap whose length equals the shard's row
    /// count, so io_stage's `bm.len() >= n_rows` skip still fires.
    #[test]
    fn reconstruct_deletion_map_full_shard() {
        let global_row_starts = [Some(0u64)];
        let ranges = [Some((0u64, 5u64))];
        let dv = dv_global(&[0, 1, 2, 3, 4]);

        let map = reconstruct_deletion_map(&dv, &global_row_starts, &ranges);
        assert_eq!(map.get(&0).unwrap().len(), 5);
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
            let handle = tokio::spawn(io_stage(reader, shard_groups, None, None, tx));

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
            let handle = tokio::spawn(io_stage(reader, shard_groups, None, None, tx));

            let group = rx.recv().await.unwrap();
            assert_eq!(group.shards.len(), 1);
            assert!(rx.recv().await.is_none());
            handle.await.unwrap().unwrap();
        });
    }
}
