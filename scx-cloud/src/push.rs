//! Streaming Push: upload a local `.scx` file to a cloud/local exploded
//! `.scxd` directory without creating an intermediate local directory.
//!
//! Implements docs/cloud.md (Streaming pull/push).
//!
//! Pipeline:
//!   1. Open local `.scx`, read header + full catalog
//!   2. Upload section files in parallel (N async tasks)
//!   3. Upload `_header.bin`
//!   4. Upload `_catalog.bin` LAST (atomic-publish semantics)

#[cfg(test)]
use std::io::Cursor;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use object_store::path::Path as ObjPath;
use object_store::{ObjectStore, WriteMultipart};

use scx_format_io::header::HEADER_SIZE;

use crate::error::{CloudError, Result};
use crate::explode::section_name_to_path;

/// Max concurrent in-flight parts per multipart upload. Bounds the per-upload
/// memory to `PART_CONCURRENCY × chunk` on top of the read buffer.
const PART_CONCURRENCY: usize = 4;

/// Options for the push operation.
pub struct PushOptions {
    /// Number of parallel upload tasks (default: 8).
    pub parallelism: usize,
}

impl Default for PushOptions {
    fn default() -> Self {
        Self { parallelism: 8 }
    }
}

/// Statistics from a push operation.
pub struct PushStats {
    pub bytes_uploaded: u64,
    pub sections_uploaded: usize,
    pub elapsed: std::time::Duration,
    pub throughput_mbps: f64,
}

/// Streaming push: read a local `.scx` file and upload as an exploded
/// `.scxd` directory to cloud or local storage.
///
/// The catalog object is uploaded last. Until it exists, the remote
/// `.scxd` directory is not openable — providing atomic-publish semantics.
pub async fn push(source: &Path, dest: &str, options: PushOptions) -> Result<PushStats> {
    push_inner(source, dest, options, crate::streaming::MULTIPART_THRESHOLD).await
}

/// Inner implementation with an injectable multipart threshold so tests can
/// exercise the chunked multipart path without a multi-MiB fixture.
async fn push_inner(
    source: &Path,
    dest: &str,
    options: PushOptions,
    multipart_threshold: u64,
) -> Result<PushStats> {
    let start = Instant::now();

    // 1. Read only the header + catalog (KB–MB), not the whole file. Section
    //    payloads are streamed from the file per upload below, so peak memory
    //    is O(parallelism × chunk), independent of the source-file size.
    let (_header, full_catalog) = crate::streaming::read_header_and_catalog(source)?;

    // 2. Parse destination and create backend
    let location = crate::backend::parse_location(dest)?;
    let backend = crate::backend::create_backend(&location).await?;
    let make_path = crate::pull::build_path_fn(&location);

    let parallelism = options.parallelism.max(1);
    let sections_uploaded = full_catalog.entries.len();
    let total_bytes_uploaded = AtomicU64::new(0);

    // 3. Resolve section → object paths up front (cheap; surfaces a malformed
    //    catalog before any upload starts). No section payloads are copied.
    let mut planned: Vec<(ObjPath, u64, u64)> = Vec::with_capacity(full_catalog.entries.len());
    for entry in &full_catalog.entries {
        let rel_path = section_name_to_path(&entry.name, entry.section_type)
            .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        planned.push((make_path(&rel_path), entry.offset, entry.length));
    }

    // 4. Upload sections in bounded windows. Each task streams its section
    //    from its own file handle — no whole-file buffer, no full list of
    //    per-section copies held at once.
    let backend_ref: &dyn ObjectStore = backend.as_ref();
    for window in planned.chunks(parallelism) {
        let mut handles = Vec::with_capacity(window.len());
        for (obj_path, offset, length) in window {
            let counter = &total_bytes_uploaded;
            handles.push(async move {
                upload_section(
                    backend_ref,
                    source,
                    *offset,
                    *length,
                    obj_path,
                    multipart_threshold,
                )
                .await?;
                counter.fetch_add(*length, Ordering::Relaxed);
                Ok::<(), CloudError>(())
            });
        }
        for result in futures::future::join_all(handles).await {
            result?;
        }
    }

    // 5. Upload _header.bin (raw 256-byte header, byte-identical to source).
    {
        let header_path = make_path("_header.bin");
        let header_bytes = crate::streaming::read_raw_header(source)?;
        backend
            .put(
                &header_path,
                bytes::Bytes::copy_from_slice(&header_bytes).into(),
            )
            .await?;
        total_bytes_uploaded.fetch_add(HEADER_SIZE as u64, Ordering::Relaxed);
    }

    // 6. Upload _catalog.bin LAST (atomic-publish semantics)
    {
        let catalog_path = make_path("_catalog.bin");
        let mut catalog_bytes = Vec::new();
        full_catalog.write_to(&mut catalog_bytes)?;
        let catalog_len = catalog_bytes.len() as u64;
        backend
            .put(&catalog_path, bytes::Bytes::from(catalog_bytes).into())
            .await?;
        total_bytes_uploaded.fetch_add(catalog_len, Ordering::Relaxed);
    }

    let total_bytes_uploaded = total_bytes_uploaded.into_inner();
    let elapsed = start.elapsed();
    let throughput_mbps = if elapsed.as_secs_f64() > 0.0 {
        (total_bytes_uploaded as f64 / 1_000_000.0) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    Ok(PushStats {
        bytes_uploaded: total_bytes_uploaded,
        sections_uploaded: sections_uploaded + 2, // +2 for _header.bin and _catalog.bin
        elapsed,
        throughput_mbps,
    })
}

/// Stream a single section's byte range from `source` to `obj_path`.
///
/// Small sections (< [`MULTIPART_THRESHOLD`](crate::streaming::MULTIPART_THRESHOLD))
/// upload via a single `put`; larger ones (X / CSC shards) stream through a
/// chunked multipart upload so neither path holds the whole section in memory.
async fn upload_section(
    backend: &dyn ObjectStore,
    source: &Path,
    offset: u64,
    length: u64,
    obj_path: &ObjPath,
    multipart_threshold: u64,
) -> Result<()> {
    let mut file = std::fs::File::open(source)?;
    file.seek(SeekFrom::Start(offset))?;

    if length < multipart_threshold {
        let mut buf = vec![0u8; length as usize];
        file.read_exact(&mut buf)?;
        backend
            .put(obj_path, bytes::Bytes::from(buf).into())
            .await?;
        return Ok(());
    }

    let upload = backend.put_multipart(obj_path).await?;
    let mut writer = WriteMultipart::new(upload);
    let mut remaining = length;
    let mut buf = vec![0u8; crate::streaming::CHUNK_SIZE];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        file.read_exact(&mut buf[..want])?;
        writer
            .wait_for_capacity(PART_CONCURRENCY)
            .await
            .map_err(CloudError::ObjectStore)?;
        writer.write(&buf[..want]);
        remaining -= want as u64;
    }
    writer.finish().await.map_err(CloudError::ObjectStore)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_format_io::catalog::FullCatalog;
    use scx_format_io::header::FileHeader;
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

    #[tokio::test]
    async fn test_push_creates_correct_directory_structure() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);

        let dest_dir = dir.path().join("pushed.scxd");
        std::fs::create_dir_all(&dest_dir).unwrap();
        let dest_str = dest_dir.to_string_lossy().to_string();

        let stats = push(&input, &dest_str, PushOptions::default())
            .await
            .unwrap();

        assert!(stats.bytes_uploaded > 0);
        assert!(stats.sections_uploaded > 0);

        // Verify directory structure
        assert!(dest_dir.join("_catalog.bin").exists());
        assert!(dest_dir.join("_header.bin").exists());
        assert!(dest_dir.join("obs.arrow").exists());
        assert!(dest_dir.join("var.arrow").exists());
        assert!(dest_dir.join("X").join("000000.shard").exists());
        assert!(dest_dir.join("X").join("000001.shard").exists());
    }

    #[tokio::test]
    async fn test_push_pull_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);

        // Push to exploded directory
        let dest_dir = dir.path().join("pushed.scxd");
        std::fs::create_dir_all(&dest_dir).unwrap();
        let dest_str = dest_dir.to_string_lossy().to_string();

        push(&input, &dest_str, PushOptions::default())
            .await
            .unwrap();

        // Pull back to a packed file
        let pulled_output = dir.path().join("pulled.scx");
        crate::pull::pull(
            &dest_str,
            &pulled_output,
            crate::pull::PullOptions::default(),
        )
        .await
        .unwrap();

        // Compare original and round-tripped data
        let reader_orig = ScxReader::open(&input).unwrap();
        let reader_pulled = ScxReader::open(&pulled_output).unwrap();

        assert_eq!(reader_orig.n_obs(), reader_pulled.n_obs());
        assert_eq!(reader_orig.n_vars(), reader_pulled.n_vars());

        let csr_orig = reader_orig.read_all_csr_shards().unwrap();
        let csr_pulled = reader_pulled.read_all_csr_shards().unwrap();
        assert_eq!(csr_orig.indptr, csr_pulled.indptr);
        assert_eq!(csr_orig.indices, csr_pulled.indices);
        assert_eq!(csr_orig.data, csr_pulled.data);
    }

    #[tokio::test]
    async fn test_push_uploads_catalog_last() {
        // We verify atomic-publish semantics by checking that _catalog.bin
        // exists and is valid after a successful push. The implementation
        // uploads it last, so if push succeeds, all other files must exist.
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);

        let dest_dir = dir.path().join("pushed_atomic.scxd");
        std::fs::create_dir_all(&dest_dir).unwrap();
        let dest_str = dest_dir.to_string_lossy().to_string();

        push(&input, &dest_str, PushOptions::default())
            .await
            .unwrap();

        // Read and validate _catalog.bin
        let catalog_data = std::fs::read(dest_dir.join("_catalog.bin")).unwrap();
        let catalog =
            FullCatalog::read_from(&mut Cursor::new(&catalog_data), catalog_data.len(), true)
                .unwrap();

        // All section files referenced by catalog should exist
        for entry in &catalog.entries {
            let rel_path = section_name_to_path(&entry.name, entry.section_type).unwrap();
            let file_path = dest_dir.join(&rel_path);
            assert!(
                file_path.exists(),
                "section file {rel_path} (name: {}) should exist after push",
                entry.name
            );
        }
    }

    /// Drive the chunked multipart streaming path (not just the single-`put`
    /// path) by forcing a tiny multipart threshold so every shard section goes
    /// through `WriteMultipart`, then assert the pulled-back data is identical.
    /// The local-filesystem `ObjectStore` backend implements `put_multipart`,
    /// so this exercises the real streaming code path against disk.
    #[tokio::test]
    async fn test_push_multipart_path_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 200, 60);

        let dest_dir = dir.path().join("pushed_mp.scxd");
        std::fs::create_dir_all(&dest_dir).unwrap();
        let dest_str = dest_dir.to_string_lossy().to_string();

        // Threshold of 1 byte → all non-empty sections take the multipart path.
        push_inner(&input, &dest_str, PushOptions::default(), 1)
            .await
            .unwrap();

        let pulled_output = dir.path().join("pulled_mp.scx");
        crate::pull::pull(
            &dest_str,
            &pulled_output,
            crate::pull::PullOptions::default(),
        )
        .await
        .unwrap();

        let reader_orig = ScxReader::open(&input).unwrap();
        let reader_pulled = ScxReader::open(&pulled_output).unwrap();
        assert_eq!(reader_orig.n_obs(), reader_pulled.n_obs());
        assert_eq!(reader_orig.n_vars(), reader_pulled.n_vars());

        let csr_orig = reader_orig.read_all_csr_shards().unwrap();
        let csr_pulled = reader_pulled.read_all_csr_shards().unwrap();
        assert_eq!(csr_orig.indptr, csr_pulled.indptr);
        assert_eq!(csr_orig.indices, csr_pulled.indices);
        assert_eq!(csr_orig.data, csr_pulled.data);

        // Round-tripped file passes checksum validation.
        for (name, passed) in reader_pulled.validate().unwrap() {
            assert!(passed, "checksum failed for section: {name}");
        }
    }
}
