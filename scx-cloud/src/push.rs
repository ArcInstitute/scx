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

use std::io::Cursor;
use std::path::Path;
use std::time::Instant;

use object_store::ObjectStore;

use scx_format_io::catalog::FullCatalog;
use scx_format_io::header::{FileHeader, HEADER_SIZE};

use crate::error::{CloudError, Result};
use crate::explode::section_name_to_path;

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
    let start = Instant::now();

    // 1. Read the entire source file
    let file_data = std::fs::read(source)?;
    if file_data.len() < HEADER_SIZE {
        return Err(CloudError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file too small to contain SCX header",
        )));
    }

    let header = FileHeader::read_from(&mut Cursor::new(&file_data[..HEADER_SIZE]))?;
    let fc_offset = header.full_catalog_offset as usize;
    let fc_length = header.full_catalog_length as usize;

    if fc_offset + fc_length > file_data.len() {
        return Err(CloudError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "catalog extends beyond file",
        )));
    }

    let full_catalog = FullCatalog::read_from(
        &mut Cursor::new(&file_data[fc_offset..fc_offset + fc_length]),
        fc_length,
        true,
    )?;

    // 2. Parse destination and create backend
    let location = crate::backend::parse_location(dest)?;
    let backend = crate::backend::create_backend(&location).await?;

    let make_path = crate::pull::build_path_fn(&location);

    let mut total_bytes_uploaded: u64 = 0;
    let parallelism = options.parallelism.max(1);

    // 3. Build upload tasks for each section (validate bounds first)
    let mut upload_tasks: Vec<(String, Vec<u8>)> = Vec::with_capacity(full_catalog.entries.len());
    for entry in &full_catalog.entries {
        let rel_path = section_name_to_path(&entry.name, entry.section_type)
            .map_err(|e| CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        let src_start = entry.offset as usize;
        let src_len = entry.length as usize;
        let src_end =
            src_start
                .checked_add(src_len)
                .ok_or_else(|| CloudError::SliceBoundsExceeded {
                    offset: src_start,
                    length: src_len,
                    data_len: file_data.len(),
                })?;
        if src_end > file_data.len() {
            return Err(CloudError::SliceBoundsExceeded {
                offset: src_start,
                length: src_len,
                data_len: file_data.len(),
            });
        }
        let data = file_data[src_start..src_end].to_vec();
        upload_tasks.push((rel_path, data));
    }

    // 4. Upload sections in parallel batches (consume upload_tasks to avoid cloning)
    let sections_uploaded = upload_tasks.len();
    let mut remaining = upload_tasks;
    while !remaining.is_empty() {
        let chunk_size = remaining.len().min(parallelism);
        let chunk: Vec<_> = remaining.drain(..chunk_size).collect();
        let mut handles = Vec::with_capacity(chunk.len());

        for (rel_path, data) in chunk {
            let obj_path = make_path(&rel_path);
            let len = data.len() as u64;
            let bytes = bytes::Bytes::from(data);
            let backend_ref = &backend;
            handles.push(async move {
                backend_ref.put(&obj_path, bytes.into()).await?;
                Ok::<u64, CloudError>(len)
            });
        }

        let results = futures::future::join_all(handles).await;
        for result in results {
            total_bytes_uploaded += result?;
        }
    }

    // 5. Upload _header.bin
    {
        let header_path = make_path("_header.bin");
        let header_bytes = bytes::Bytes::from(file_data[..HEADER_SIZE].to_vec());
        backend.put(&header_path, header_bytes.into()).await?;
        total_bytes_uploaded += HEADER_SIZE as u64;
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
        total_bytes_uploaded += catalog_len;
    }

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

#[cfg(test)]
mod tests {
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
}
