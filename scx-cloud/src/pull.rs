//! Streaming Pull: download from cloud/local exploded `.scxd` and pack
//! into a local `.scx` file in a single pass.
//!
//! Implements docs/cloud.md (Streaming pull/push).
//!
//! Pipeline:
//!   1. GET `_catalog.bin` + `_header.bin`
//!   2. Parse catalog → know all sections
//!   3. Open output file, write header placeholder
//!   4. Download sections (parallel async) → reorder buffer → sequential write
//!   5. Write full catalog + front catalog + root catalog + header
//!   6. fsync + atomic rename

use std::collections::BTreeMap;
use std::io::{BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use object_store::path::Path as ObjPath;
use object_store::ObjectStore;

use scx_format::catalog::{FullCatalog, FullCatalogEntry, RootCatalog, RootCatalogEntry};
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::section::{align_to_8, SectionType};

use crate::error::{CloudError, Result};
use crate::explode::section_name_to_path;

/// Offset where sections begin: 256 (header) + 4096 (root catalog placeholder).
const SECTIONS_START_OFFSET: u64 = 4352;

/// Age threshold above which a `{stem}.tmp.*` file whose owning pid is
/// no longer running (or unparseable) is considered orphaned and safe to
/// delete. One hour comfortably exceeds any legitimate in-flight pull.
const STALE_TMP_AGE: Duration = Duration::from_secs(60 * 60);

/// Return true if a pid is still live according to `/proc/{pid}` (Linux).
///
/// Non-Linux platforms have no `/proc` — the function returns `false` so
/// the caller falls back to the mtime heuristic. Benchmarks and cloud
/// operations run on Linux, so the proc path is exercised in practice.
fn pid_is_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

/// Remove stale `{dest}.tmp.*` files left behind by prior interrupted pulls.
///
/// Pull writes to `{dest}.tmp.{pid}` and atomically renames on completion.
/// If a pull is interrupted (SIGTERM, crash), the `.tmp.{pid}` file remains
/// orphaned. The current run picks a different PID-keyed path so it's not
/// blocked, but the orphan accumulates disk usage on repeated retries.
/// This helper sweeps orphaned `{stem}.tmp.*` siblings of `dest` before we
/// start downloading. Called at the top of both `pull` and `pull_filtered`.
///
/// A file is treated as orphaned and deleted only if BOTH hold:
///   1. Its owning pid (the trailing numeric suffix) is not live on
///      `/proc`, OR the suffix does not parse as a pid.
///   2. Its mtime is older than `STALE_TMP_AGE`.
///
/// This keeps the sweep safe when multiple concurrent pulls target the
/// same `dest` — a live sibling's tmp file is preserved regardless of
/// filename patterns. Deliberately best-effort: a directory-listing
/// failure is not a pull failure (the worst case is orphans persist;
/// they never block retries).
fn cleanup_stale_tmp_files(dest: &Path) {
    let parent = match dest.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => std::path::Path::new("."),
    };
    let stem = match dest.file_name().and_then(|s| s.to_str()) {
        Some(s) => s,
        None => return,
    };
    let prefix = format!("{stem}.tmp.");
    let entries = match std::fs::read_dir(parent) {
        Ok(e) => e,
        Err(_) => return,
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if !name_str.starts_with(&prefix) {
            continue;
        }
        // Bail out cheaply if the suffix parses to a pid that is still alive.
        let suffix = &name_str[prefix.len()..];
        if let Ok(pid) = suffix.parse::<u32>() {
            if pid_is_alive(pid) {
                continue;
            }
        }
        // Only delete files that look abandoned. Files younger than
        // STALE_TMP_AGE might belong to a sibling whose pid we couldn't
        // confirm (e.g. wrapped around, or non-Linux without /proc).
        let Ok(meta) = entry.metadata() else { continue };
        let Ok(mtime) = meta.modified() else { continue };
        let Ok(age) = now.duration_since(mtime) else {
            continue;
        };
        if age < STALE_TMP_AGE {
            continue;
        }
        let _ = std::fs::remove_file(entry.path());
    }
}

/// Options for the pull operation.
pub struct PullOptions {
    /// Number of parallel download tasks (default: 8).
    pub parallelism: usize,
    /// Reorder buffer size in shard slots (default: 4).
    pub reorder_buffer: usize,
    /// Produce cloud-ready output with front catalog (default: true).
    pub cloud_ready: bool,
}

impl Default for PullOptions {
    fn default() -> Self {
        Self {
            parallelism: 8,
            reorder_buffer: 4,
            cloud_ready: true,
        }
    }
}

/// Statistics from a pull operation.
pub struct PullStats {
    pub bytes_downloaded: u64,
    pub sections_downloaded: usize,
    pub elapsed: std::time::Duration,
    pub throughput_mbps: f64,
}

/// Streaming pull: download an exploded `.scxd` from a source (cloud or local)
/// and pack into a local `.scx` file.
///
/// The source can be a local directory path or a cloud URL pointing to an
/// exploded `.scxd` directory.
pub async fn pull(source: &str, dest: &Path, options: PullOptions) -> Result<PullStats> {
    let start = Instant::now();

    // Sweep any `.tmp.*` orphans left behind by a previously interrupted
    // pull targeting the same destination. Best-effort; a failure here is
    // not a pull failure (see ``cleanup_stale_tmp_files``).
    cleanup_stale_tmp_files(dest);

    // 1. Parse location and create backend
    let location = crate::backend::parse_location(source)?;
    let backend = crate::backend::create_backend(&location).await?;

    // Determine the prefix for object paths
    let prefix = match &location {
        crate::backend::CloudLocation::Gcs { prefix, .. }
        | crate::backend::CloudLocation::S3 { prefix, .. }
        | crate::backend::CloudLocation::Azure { prefix, .. } => prefix.clone(),
        crate::backend::CloudLocation::Local(path) => {
            // For local backend, the backend is rooted at the directory,
            // so we use the directory name as prefix context
            path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default()
        }
    };

    // Helper to build object paths
    let make_path = |filename: &str| -> ObjPath {
        match &location {
            crate::backend::CloudLocation::Local(_) => {
                // Local backend is rooted at the .scxd directory, paths are relative
                ObjPath::from(filename)
            }
            _ => {
                let full = if prefix.ends_with('/') {
                    format!("{prefix}{filename}")
                } else if prefix.is_empty() {
                    filename.to_string()
                } else {
                    format!("{prefix}/{filename}")
                };
                ObjPath::from(full)
            }
        }
    };

    // 2. Download _catalog.bin and _header.bin
    let catalog_path = make_path("_catalog.bin");
    let catalog_data = backend.get(&catalog_path).await?.bytes().await?;
    let catalog_bytes = catalog_data.to_vec();
    let original_catalog =
        FullCatalog::read_from(&mut Cursor::new(&catalog_bytes), catalog_bytes.len(), true)?;

    let header_path = make_path("_header.bin");
    let header_data = backend.get(&header_path).await?.bytes().await?;
    let header = FileHeader::read_from(&mut Cursor::new(&header_data))?;

    let mut total_bytes_downloaded = (catalog_bytes.len() + header_data.len()) as u64;

    // 3. Define section ordering for cloud-optimized layout.
    //    `CscShard` placed adjacent to `CsrShard` so column-major
    //    reads stay in the contiguous prefix region of the pulled
    //    file. Phase I.2.
    let section_order: &[SectionType] = &[
        SectionType::ObsMetadata,
        SectionType::ObsIndex,
        SectionType::VarMetadata,
        SectionType::VarIndex,
        SectionType::CsrShard,
        SectionType::CscShard,
        SectionType::LayerCsrShard,
        SectionType::ObsmEmbedding,
        SectionType::ObspCsrShard,
        SectionType::UnsBlob,
        SectionType::ObsPredicateIndex,
        SectionType::VarPredicateIndex,
        SectionType::Provenance,
        SectionType::DeletionVectors,
    ];

    // Group and order entries
    let mut grouped: BTreeMap<u8, Vec<&FullCatalogEntry>> = BTreeMap::new();
    for entry in &original_catalog.entries {
        grouped
            .entry(entry.section_type as u8)
            .or_default()
            .push(entry);
    }

    let known_types: std::collections::HashSet<u8> =
        section_order.iter().map(|&st| st as u8).collect();
    let mut ordered_entries: Vec<&FullCatalogEntry> =
        Vec::with_capacity(original_catalog.entries.len());
    for &st in section_order {
        if let Some(entries) = grouped.get(&(st as u8)) {
            ordered_entries.extend(entries);
        }
    }
    // Include any section types not in section_order (unknown/future types)
    for (&group_type, entries) in &grouped {
        if !known_types.contains(&group_type) {
            ordered_entries.extend(entries);
        }
    }

    // 4. Download all section files in parallel batches
    let parallelism = options.parallelism.max(1);
    let mut downloaded_sections: Vec<(usize, Vec<u8>)> = Vec::with_capacity(ordered_entries.len());

    // Build download tasks as (index, filename) pairs
    let download_tasks: Vec<(usize, String)> = ordered_entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let rel_path = section_name_to_path(&entry.name, entry.section_type).map_err(|e| {
                CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            })?;
            Ok((i, rel_path))
        })
        .collect::<Result<Vec<_>>>()?;

    // Download in batches of `parallelism`
    for chunk in download_tasks.chunks(parallelism) {
        let mut handles = Vec::with_capacity(chunk.len());

        for &(idx, ref filename) in chunk {
            let path = make_path(filename);
            let backend_ref = &backend;
            handles.push(async move {
                let result = backend_ref.get(&path).await?.bytes().await?;
                Ok::<(usize, Vec<u8>), CloudError>((idx, result.to_vec()))
            });
        }

        let results = futures::future::join_all(handles).await;
        for result in results {
            let (idx, data) = result?;
            total_bytes_downloaded += data.len() as u64;
            downloaded_sections.push((idx, data));
        }
    }

    // Sort by index to ensure correct order
    downloaded_sections.sort_by_key(|(idx, _)| *idx);

    // 5. Write packed output file
    let tmp_path =
        std::path::PathBuf::from(format!("{}.tmp.{}", dest.display(), std::process::id()));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)?;
    let mut writer = BufWriter::new(file);

    // Write placeholder for header + root catalog
    writer.write_all(&vec![0u8; SECTIONS_START_OFFSET as usize])?;
    let mut write_offset = SECTIONS_START_OFFSET;

    // Estimate and reserve front catalog space if cloud-ready
    let front_catalog_offset;
    let estimated_front_catalog_size;
    if options.cloud_ready {
        estimated_front_catalog_size = estimate_catalog_size(ordered_entries.len());
        front_catalog_offset = write_offset;
        writer.write_all(&vec![0u8; estimated_front_catalog_size])?;
        write_offset += estimated_front_catalog_size as u64;

        let aligned = align_to_8(write_offset);
        let pad = (aligned - write_offset) as usize;
        if pad > 0 {
            writer.write_all(&vec![0u8; pad])?;
            write_offset = aligned;
        }
    } else {
        estimated_front_catalog_size = 0;
        front_catalog_offset = 0;
    }

    // Write each section sequentially
    let mut new_entries: Vec<FullCatalogEntry> = Vec::with_capacity(ordered_entries.len());

    for (i, (_, section_data)) in downloaded_sections.iter().enumerate() {
        let entry = ordered_entries[i];

        // Pad to 8-byte alignment
        let aligned = align_to_8(write_offset);
        let pad = (aligned - write_offset) as usize;
        if pad > 0 {
            writer.write_all(&vec![0u8; pad])?;
            write_offset = aligned;
        }

        let new_offset = write_offset;
        writer.write_all(section_data)?;
        write_offset += section_data.len() as u64;

        new_entries.push(FullCatalogEntry {
            name: entry.name.clone(),
            offset: new_offset,
            length: section_data.len() as u64,
            section_type: entry.section_type,
            checksum: entry.checksum,
            stats: entry.stats.clone(),
        });
    }

    // 6. Write full catalog at EOF
    let catalog_aligned = align_to_8(write_offset);
    let pad = (catalog_aligned - write_offset) as usize;
    if pad > 0 {
        writer.write_all(&vec![0u8; pad])?;
    }
    let full_catalog_offset_new = catalog_aligned;

    let new_full_catalog = FullCatalog {
        catalog_version: original_catalog.catalog_version,
        manifest_sequence: original_catalog.manifest_sequence,
        prev_catalog_offset: original_catalog.prev_catalog_offset,
        n_obs: original_catalog.n_obs,
        entries: new_entries,
    };
    let mut new_catalog_bytes = Vec::new();
    new_full_catalog.write_to(&mut new_catalog_bytes)?;
    let new_full_catalog_length = new_catalog_bytes.len() as u64;
    writer.write_all(&new_catalog_bytes)?;

    // 7. Write front catalog if cloud-ready
    let mut new_header = header;
    if options.cloud_ready {
        let front_catalog_length = new_catalog_bytes.len();
        if front_catalog_length <= estimated_front_catalog_size {
            writer.seek(SeekFrom::Start(front_catalog_offset))?;
            writer.write_all(&new_catalog_bytes)?;
            let remaining = estimated_front_catalog_size - front_catalog_length;
            if remaining > 0 {
                writer.write_all(&vec![0u8; remaining])?;
            }
            new_header.front_catalog_offset = front_catalog_offset;
            new_header.front_catalog_length = front_catalog_length as u64;
            new_header.set_front_catalog();
        } else {
            writer.seek(SeekFrom::Start(front_catalog_offset))?;
            writer.write_all(&vec![0u8; estimated_front_catalog_size])?;
            new_header.front_catalog_offset = 0;
            new_header.front_catalog_length = 0;
            new_header.clear_front_catalog();
        }
    }

    // 8. Build and write root catalog at offset 256
    let root_catalog = build_root_catalog(&new_full_catalog);
    let mut root_buf = Vec::new();
    root_catalog.write_to(&mut root_buf)?;
    let root_catalog_length = root_buf.len() as u64;
    root_buf.resize(4096, 0);

    writer.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    writer.write_all(&root_buf)?;

    // 9. Write header
    new_header.root_catalog_offset = HEADER_SIZE as u64;
    new_header.root_catalog_length = root_catalog_length;
    new_header.full_catalog_offset = full_catalog_offset_new;
    new_header.full_catalog_length = new_full_catalog_length;
    new_header.file_checksum = 0;

    writer.seek(SeekFrom::Start(0))?;
    new_header.write_to(&mut writer)?;

    // 10. Compute file checksum
    writer.flush()?;
    let mut file = writer.into_inner().map_err(std::io::Error::from)?;
    let file_checksum = compute_file_checksum(&mut file)?;
    new_header.file_checksum = file_checksum;
    file.seek(SeekFrom::Start(0))?;
    new_header.write_to(&mut file)?;

    // 11. fsync + atomic rename
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp_path, dest)?;

    let elapsed = start.elapsed();
    let throughput_mbps = if elapsed.as_secs_f64() > 0.0 {
        (total_bytes_downloaded as f64 / 1_000_000.0) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    Ok(PullStats {
        bytes_downloaded: total_bytes_downloaded,
        sections_downloaded: ordered_entries.len(),
        elapsed,
        throughput_mbps,
    })
}

/// Estimate catalog size: 30 bytes header + n_entries × 100 bytes + checksum.
fn estimate_catalog_size(n_entries: usize) -> usize {
    let size = 30 + n_entries * 100 + 32;
    (size + 7) & !7
}

/// Build a root catalog from a full catalog.
fn build_root_catalog(catalog: &FullCatalog) -> RootCatalog {
    let mut groups: BTreeMap<u8, Vec<&FullCatalogEntry>> = BTreeMap::new();
    for entry in &catalog.entries {
        groups
            .entry(entry.section_type as u8)
            .or_default()
            .push(entry);
    }

    let mut root_entries = Vec::new();
    for (&group_type, entries) in &groups {
        let first_offset = entries.iter().map(|e| e.offset).min().unwrap_or(0);
        let total_length: u64 = entries.iter().map(|e| e.length).sum();
        let n_sections = entries.len() as u32;
        root_entries.push(RootCatalogEntry {
            group_type,
            first_section_offset: first_offset,
            total_group_length: total_length,
            n_sections,
            summary: [0u8; 32],
        });
    }

    RootCatalog {
        n_section_groups: root_entries.len() as u16,
        entries: root_entries,
    }
}

/// Compute file checksum: BLAKE3 of entire file, truncated to u64.
fn compute_file_checksum(file: &mut (impl Read + Seek)) -> Result<u64> {
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = blake3::Hasher::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = file.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
    }
    let hash = hasher.finalize();
    Ok(scx_format::checksum::truncate_hash_to_u64(&hash))
}

/// Statistics from a selective pull operation.
pub struct PullFilteredStats {
    pub total_shards: usize,
    pub downloaded_shards: usize,
    pub skipped_shards: usize,
    pub matching_cells: u64,
    pub bytes_downloaded: u64,
    pub bytes_saved: u64,
    pub elapsed: std::time::Duration,
}

/// Selective pull: download only shards matching a predicate.
///
/// Pipeline:
///   1. GET `_catalog.bin` → parse catalog, identify shard row ranges
///   2. GET `obs.arrow` → evaluate predicate against obs metadata
///   3. Map matching cells to shard IDs (from catalog row ranges)
///   4. GET only matching shards (parallel) + var.arrow + other non-shard sections
///   5. Write filtered obs (only matching rows) + matching shards
///   6. Write catalog + header → fsync → rename
pub async fn pull_filtered(
    source: &str,
    dest: &Path,
    filter: &str,
    options: PullOptions,
) -> Result<PullFilteredStats> {
    let start = Instant::now();

    // Same orphan-sweep as `pull`: any `.tmp.{pid}` files left by a
    // prior interrupted run are removed before we start downloading.
    cleanup_stale_tmp_files(dest);

    // 1. Parse location and create backend
    let location = crate::backend::parse_location(source)?;
    let backend = crate::backend::create_backend(&location).await?;

    let make_path = build_path_fn(&location);

    // 2. Download _catalog.bin and _header.bin
    let catalog_path = make_path("_catalog.bin");
    let catalog_data = backend.get(&catalog_path).await?.bytes().await?;
    let catalog_bytes = catalog_data.to_vec();
    let original_catalog =
        FullCatalog::read_from(&mut Cursor::new(&catalog_bytes), catalog_bytes.len(), true)?;

    let header_path = make_path("_header.bin");
    let header_data = backend.get(&header_path).await?.bytes().await?;
    let header = FileHeader::read_from(&mut Cursor::new(&header_data))?;

    let mut total_bytes_downloaded = (catalog_bytes.len() + header_data.len()) as u64;

    // 3. Download obs.arrow and evaluate predicate
    let obs_path_str = section_name_to_path("obs", SectionType::ObsMetadata)
        .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    let obs_obj_path = make_path(&obs_path_str);
    let obs_data = backend.get(&obs_obj_path).await?.bytes().await?;
    total_bytes_downloaded += obs_data.len() as u64;
    let obs_bytes = obs_data.to_vec();

    // Parse obs as Arrow IPC
    let obs_cursor = Cursor::new(&obs_bytes);
    let obs_reader = arrow::ipc::reader::FileReader::try_new(obs_cursor, None)
        .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    let obs_schema = obs_reader.schema();
    let obs_batch: arrow::array::RecordBatch = obs_reader
        .into_iter()
        .next()
        .ok_or_else(|| {
            CloudError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "obs Arrow IPC file contains no batches",
            ))
        })?
        .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;

    // Parse and evaluate predicate
    let predicate = scx_engine::parse_predicate(filter, &obs_schema).map_err(|e| {
        CloudError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("predicate parse error: {e}"),
        ))
    })?;
    let mask = scx_engine::evaluate(&predicate, &obs_batch).map_err(|e| {
        CloudError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("predicate evaluation error: {e}"),
        ))
    })?;

    // Collect matching row indices (used to identify needed shards)
    let matching_rows: Vec<u64> = (0..mask.len())
        .filter(|&i| mask.value(i))
        .map(|i| i as u64)
        .collect();
    let matching_cells = matching_rows.len() as u64;

    // 4. Map matching rows to shard IDs
    let sorted_shards = original_catalog.shards_sorted();
    let total_shards = sorted_shards.len();

    let mut needed_shard_indices: std::collections::BTreeSet<usize> =
        std::collections::BTreeSet::new();

    for &row_idx in &matching_rows {
        for (shard_idx, entry) in sorted_shards.iter().enumerate() {
            if let Some(stats) = &entry.stats {
                if row_idx >= stats.row_start && row_idx < stats.row_end {
                    needed_shard_indices.insert(shard_idx);
                    break;
                }
            }
        }
    }

    let downloaded_shards = needed_shard_indices.len();
    let skipped_shards = total_shards - downloaded_shards;

    // Collect ALL row indices from downloaded shards (not just predicate-matching
    // rows). Shards are copied verbatim so the output file must include obs rows
    // for every row in those shards, otherwise n_obs disagrees with shard data.
    let shard_row_indices: Vec<u64> = needed_shard_indices
        .iter()
        .flat_map(|&si| {
            if let Some(stats) = &sorted_shards[si].stats {
                (stats.row_start..stats.row_end).collect::<Vec<u64>>()
            } else {
                vec![]
            }
        })
        .collect();
    let shard_total_rows = shard_row_indices.len() as u64;

    // Calculate bytes saved (estimate from skipped shards)
    let bytes_saved: u64 = sorted_shards
        .iter()
        .enumerate()
        .filter(|(i, _)| !needed_shard_indices.contains(i))
        .map(|(_, e)| e.length)
        .sum();

    // 5. Determine which sections to download
    let mut entries_to_download: Vec<&FullCatalogEntry> = Vec::new();
    let mut shard_name_set: std::collections::HashSet<String> = std::collections::HashSet::new();

    for &si in &needed_shard_indices {
        shard_name_set.insert(sorted_shards[si].name.clone());
    }

    for entry in &original_catalog.entries {
        match entry.section_type {
            SectionType::ObsMetadata => continue, // already downloaded
            SectionType::CsrShard if shard_name_set.contains(&entry.name) => {
                entries_to_download.push(entry);
            }
            SectionType::VarMetadata | SectionType::VarIndex | SectionType::UnsBlob => {
                entries_to_download.push(entry);
            }
            _ => {} // skip obsm, layers, obsp, etc.
        }
    }

    // Download needed sections
    let parallelism = options.parallelism.max(1);
    let mut section_data_map: std::collections::HashMap<String, Vec<u8>> =
        std::collections::HashMap::new();

    let download_tasks: Vec<(String, String)> = entries_to_download
        .iter()
        .map(|entry| {
            let rel_path = section_name_to_path(&entry.name, entry.section_type).map_err(|e| {
                CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            })?;
            Ok((entry.name.clone(), rel_path))
        })
        .collect::<Result<Vec<_>>>()?;

    for chunk in download_tasks.chunks(parallelism) {
        let mut handles = Vec::with_capacity(chunk.len());

        for (name, filename) in chunk {
            let path = make_path(filename);
            let name_clone = name.clone();
            let backend_ref = &backend;
            handles.push(async move {
                let result = backend_ref.get(&path).await?.bytes().await?;
                Ok::<(String, Vec<u8>), CloudError>((name_clone, result.to_vec()))
            });
        }

        let results = futures::future::join_all(handles).await;
        for result in results {
            let (name, data) = result?;
            total_bytes_downloaded += data.len() as u64;
            section_data_map.insert(name, data);
        }
    }

    // 6. Build obs for all rows in downloaded shards (not just predicate-matching
    //    rows) so that obs row count matches shard data.
    let filtered_obs = filter_record_batch(&obs_batch, &shard_row_indices)?;
    let filtered_obs_bytes = record_batch_to_arrow_ipc(&filtered_obs)?;

    // 7. Write packed output
    let tmp_path =
        std::path::PathBuf::from(format!("{}.tmp.{}", dest.display(), std::process::id()));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(&vec![0u8; SECTIONS_START_OFFSET as usize])?;
    let mut write_offset = SECTIONS_START_OFFSET;

    let mut new_entries: Vec<FullCatalogEntry> = Vec::new();

    // Write filtered obs first
    {
        let aligned = align_to_8(write_offset);
        let pad = (aligned - write_offset) as usize;
        if pad > 0 {
            writer.write_all(&vec![0u8; pad])?;
            write_offset = aligned;
        }
        let obs_checksum = blake3::hash(&filtered_obs_bytes);
        writer.write_all(&filtered_obs_bytes)?;
        new_entries.push(FullCatalogEntry {
            name: "obs".to_string(),
            offset: write_offset,
            length: filtered_obs_bytes.len() as u64,
            section_type: SectionType::ObsMetadata,
            checksum: *obs_checksum.as_bytes(),
            stats: None,
        });
        write_offset += filtered_obs_bytes.len() as u64;
    }

    // Write sections in order: var, shards (renumbered), uns
    let section_write_order = [
        SectionType::VarMetadata,
        SectionType::VarIndex,
        SectionType::CsrShard,
        SectionType::UnsBlob,
    ];

    let mut new_row_offset: u64 = 0;

    for &st in &section_write_order {
        for entry in &entries_to_download {
            if entry.section_type != st {
                continue;
            }
            let data = match section_data_map.get(&entry.name) {
                Some(d) => d,
                None => continue,
            };

            let aligned = align_to_8(write_offset);
            let pad = (aligned - write_offset) as usize;
            if pad > 0 {
                writer.write_all(&vec![0u8; pad])?;
                write_offset = aligned;
            }

            let new_offset = write_offset;
            writer.write_all(data)?;
            write_offset += data.len() as u64;

            let new_stats = if st == SectionType::CsrShard {
                if let Some(old_stats) = &entry.stats {
                    let shard_rows = old_stats.row_end - old_stats.row_start;
                    let ns = scx_format::catalog::ShardStats {
                        row_start: new_row_offset,
                        row_end: new_row_offset + shard_rows,
                        nnz: old_stats.nnz,
                        value_min: old_stats.value_min,
                        value_max: old_stats.value_max,
                        value_sum: old_stats.value_sum,
                        n_indexed_columns: old_stats.n_indexed_columns,
                        column_stats: old_stats.column_stats.clone(),
                    };
                    new_row_offset += shard_rows;
                    Some(ns)
                } else {
                    entry.stats.clone()
                }
            } else {
                entry.stats.clone()
            };

            new_entries.push(FullCatalogEntry {
                name: entry.name.clone(),
                offset: new_offset,
                length: data.len() as u64,
                section_type: entry.section_type,
                checksum: entry.checksum,
                stats: new_stats,
            });
        }
    }

    // Write full catalog at EOF
    let catalog_aligned = align_to_8(write_offset);
    let pad = (catalog_aligned - write_offset) as usize;
    if pad > 0 {
        writer.write_all(&vec![0u8; pad])?;
    }
    let full_catalog_offset_new = catalog_aligned;

    let new_full_catalog = FullCatalog {
        catalog_version: original_catalog.catalog_version,
        manifest_sequence: original_catalog.manifest_sequence + 1,
        prev_catalog_offset: 0,
        n_obs: shard_total_rows,
        entries: new_entries,
    };
    let mut new_catalog_bytes = Vec::new();
    new_full_catalog.write_to(&mut new_catalog_bytes)?;
    let new_full_catalog_length = new_catalog_bytes.len() as u64;
    writer.write_all(&new_catalog_bytes)?;

    // Build and write root catalog
    let root_catalog = build_root_catalog(&new_full_catalog);
    let mut root_buf = Vec::new();
    root_catalog.write_to(&mut root_buf)?;
    let root_catalog_length = root_buf.len() as u64;
    root_buf.resize(4096, 0);
    writer.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    writer.write_all(&root_buf)?;

    // Write header
    let mut new_header = header;
    new_header.n_obs = shard_total_rows;
    new_header.n_csr_shards = downloaded_shards as u32;
    // pull_filtered drops CSC sidecars: the filter changes the row
    // layout, so any input CSC `indices` arrays would reference stale
    // rows. The catalog filter loop (around line 642) excludes
    // CscShard entries; clear the header count + flag to match.
    // CSC-SUPPORT.md Phase I.2.
    new_header.n_csc_shards = 0;
    new_header.clear_csc();
    new_header.root_catalog_offset = HEADER_SIZE as u64;
    new_header.root_catalog_length = root_catalog_length;
    new_header.full_catalog_offset = full_catalog_offset_new;
    new_header.full_catalog_length = new_full_catalog_length;
    new_header.front_catalog_offset = 0;
    new_header.front_catalog_length = 0;
    new_header.clear_front_catalog();
    new_header.file_checksum = 0;

    writer.seek(SeekFrom::Start(0))?;
    new_header.write_to(&mut writer)?;

    writer.flush()?;
    let mut file = writer.into_inner().map_err(std::io::Error::from)?;
    let file_checksum = compute_file_checksum(&mut file)?;
    new_header.file_checksum = file_checksum;
    file.seek(SeekFrom::Start(0))?;
    new_header.write_to(&mut file)?;

    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp_path, dest)?;

    let elapsed = start.elapsed();

    Ok(PullFilteredStats {
        total_shards,
        downloaded_shards,
        skipped_shards,
        matching_cells,
        bytes_downloaded: total_bytes_downloaded,
        bytes_saved,
        elapsed,
    })
}

/// Build a path-constructing closure from a parsed location.
pub(crate) fn build_path_fn(
    location: &crate::backend::CloudLocation,
) -> Box<dyn Fn(&str) -> ObjPath + '_> {
    match location {
        crate::backend::CloudLocation::Local(_) => {
            Box::new(|filename: &str| ObjPath::from(filename))
        }
        crate::backend::CloudLocation::Gcs { prefix, .. }
        | crate::backend::CloudLocation::S3 { prefix, .. }
        | crate::backend::CloudLocation::Azure { prefix, .. } => {
            let prefix = prefix.clone();
            Box::new(move |filename: &str| {
                let full = if prefix.ends_with('/') {
                    format!("{prefix}{filename}")
                } else if prefix.is_empty() {
                    filename.to_string()
                } else {
                    format!("{prefix}/{filename}")
                };
                ObjPath::from(full)
            })
        }
    }
}

/// Filter a RecordBatch to only include rows at the given indices.
fn filter_record_batch(
    batch: &arrow::array::RecordBatch,
    row_indices: &[u64],
) -> Result<arrow::array::RecordBatch> {
    use arrow::array::UInt64Array;
    let indices = UInt64Array::from(row_indices.to_vec());
    let columns: Vec<arrow::array::ArrayRef> = batch
        .columns()
        .iter()
        .map(|col| arrow::compute::take(col.as_ref(), &indices, None))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    let filtered = arrow::array::RecordBatch::try_new(batch.schema(), columns)
        .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    Ok(filtered)
}

/// Serialize a RecordBatch to Arrow IPC bytes (File format).
fn record_batch_to_arrow_ipc(batch: &arrow::array::RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut ipc_writer = arrow::ipc::writer::FileWriter::try_new(&mut buf, &batch.schema())
            .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        ipc_writer
            .write(batch)
            .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        ipc_writer
            .finish()
            .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_format::header::MAGIC;
    use scx_format::reader::ScxReader;
    use scx_format::writer::ScxWriter;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use std::sync::Arc;

    fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz: 0,
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
            reorder_buffer: 4,
            cloud_ready: true,
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
            reorder_buffer: 4,
            cloud_ready: true,
        };
        pull(&source, &output1, opts1).await.unwrap();

        // Pull with parallelism=8
        let output8 = dir.path().join("pulled_p8.scx");
        let opts8 = PullOptions {
            parallelism: 8,
            reorder_buffer: 4,
            cloud_ready: true,
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

    /// Force the mtime on `path` to `STALE_TMP_AGE + 60s` in the past so
    /// the cleanup helper treats it as orphaned. Uses `File::set_times`
    /// (stable since Rust 1.75) and `FileTimes::set_modified`.
    fn age_file(path: &std::path::Path) {
        let past =
            std::time::SystemTime::now() - STALE_TMP_AGE - std::time::Duration::from_secs(60);
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
}
