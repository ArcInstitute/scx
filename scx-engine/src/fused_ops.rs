//! Fused normalize + log1p operations for CSR data.
//!
//! Applies both transformations in a single pass over each CSR row, avoiding
//! a separate traversal for each operation.

use scx_sparse::ScxCsr;

/// Normalize a single CSR row to a target sum.
///
/// Equivalent to `scanpy.pp.normalize_total` for one row:
///   `data[i] = data[i] / row_sum * target_sum`
///
/// If the row sum is zero (or the row is empty), the data is left unchanged.
pub fn normalize_row(data: &mut [f32], indptr: &[i64], row_idx: usize, target_sum: f64) {
    let start = indptr[row_idx] as usize;
    let end = indptr[row_idx + 1] as usize;
    let row_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
    if row_sum > 0.0 {
        let factor = target_sum / row_sum;
        for v in &mut data[start..end] {
            *v = (*v as f64 * factor) as f32;
        }
    }
}

/// Apply `ln(x + 1)` to a single CSR row.
///
/// Equivalent to `numpy.log1p` for one row's non-zero values.
pub fn log1p_row(data: &mut [f32], indptr: &[i64], row_idx: usize) {
    let start = indptr[row_idx] as usize;
    let end = indptr[row_idx + 1] as usize;
    for v in &mut data[start..end] {
        *v = v.ln_1p();
    }
}

/// Fused normalize + log1p for a single CSR row.
///
/// Computes `ln(x / row_sum * target_sum + 1)` in a single pass, avoiding
/// an intermediate materialized array. Numerically equivalent to calling
/// `normalize_row` then `log1p_row` (within f32 rounding).
pub fn fused_normalize_log1p(data: &mut [f32], indptr: &[i64], row_idx: usize, target_sum: f64) {
    let start = indptr[row_idx] as usize;
    let end = indptr[row_idx + 1] as usize;
    let row_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
    if row_sum > 0.0 {
        let factor = target_sum / row_sum;
        for v in &mut data[start..end] {
            // Scale in f64 for precision, cast to f32, then f32 ln for speed.
            // Numerically matches sequential normalize→log1p path.
            *v = ((*v as f64 * factor) as f32).ln_1p();
        }
    }
}

/// Apply fused operations to an entire CSR matrix.
///
/// Dispatches to the optimal path based on which operations are requested:
/// - `(Some(target_sum), true)` → fused normalize+log1p (single pass)
/// - `(Some(target_sum), false)` → normalize only
/// - `(None, true)` → log1p only
/// - `(None, false)` → no-op
pub fn apply_fused_ops(csr: &mut ScxCsr, normalize: Option<f64>, log1p: bool) {
    let n_rows = csr.shape.0;
    for row in 0..n_rows {
        match (normalize, log1p) {
            (Some(target_sum), true) => {
                fused_normalize_log1p(&mut csr.data, &csr.indptr, row, target_sum)
            }
            (Some(target_sum), false) => normalize_row(&mut csr.data, &csr.indptr, row, target_sum),
            (None, true) => log1p_row(&mut csr.data, &csr.indptr, row),
            (None, false) => {}
        }
    }
}

/// Configuration for the streaming preprocess pipeline.
#[derive(Debug, Clone, Default)]
pub struct PreprocessConfig {
    /// Target sum for normalize_total. None = skip normalization.
    pub normalize_target_sum: Option<f64>,
    /// Whether to apply log1p after normalization.
    pub log1p: bool,
}

// ---------------------------------------------------------------------------
// Shared helpers for streaming_preprocess / streaming_save_layer
// ---------------------------------------------------------------------------

/// Build an output `FileHeader` from a source header, overriding `codec_id`.
///
/// All dynamic fields (nnz, catalog offsets, checksum) are zeroed — the writer
/// populates them during `finish()`.
fn build_output_header(src: &scx_format_io::FileHeader, codec_id: u8) -> scx_format_io::FileHeader {
    scx_format_io::FileHeader {
        format_version: src.format_version,
        n_obs: src.n_obs,
        n_vars: src.n_vars,
        shard_target_rows: src.shard_target_rows,
        codec_id,
        index_dtype: src.index_dtype,
        manifest_sequence: src.manifest_sequence + 1,
        ..Default::default()
    }
}

/// Write a provenance entry recording a preprocessing action.
fn write_preprocess_provenance(
    writer: &mut scx_format_io::ScxWriter,
    action: &str,
    params: &str,
) -> crate::Result<()> {
    writer.write_provenance(vec![scx_format_io::ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: action.to_string(),
        tool: "scx-engine".to_string(),
        params_json: params.to_string(),
        input_checksums: vec![],
    }])?;
    Ok(())
}

/// Carry the deletion-vector section into a preprocessed output.
///
/// Both streaming rewrites re-emit every shard at the source's `row_start` and
/// change only the *values*, so the global obs row indices a v2 deletion vector
/// stores stay valid verbatim — the same 1:1 argument `scx optimize` makes.
/// Without this the deleted cells come back: `build_output_header` starts from
/// `..Default::default()`, so the flag is not inherited, and no section is
/// written for `sync_from_catalog` to re-derive it from.
fn carry_deletion_vectors(
    reader: &scx_format_io::ScxReader,
    writer: &mut scx_format_io::ScxWriter,
) -> crate::Result<()> {
    if let Some(dv) = reader.read_deletion_vectors()? {
        writer.write_deletion_vectors(&dv)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public streaming pipelines
// ---------------------------------------------------------------------------

/// Streaming shard-by-shard preprocessing pipeline.
///
/// Reads the source SCX file one shard at a time, applies fused
/// normalize+log1p operations, and writes the result to a new SCX file.
///
/// **Preserved sections:** `obs`, `var`, `obsm`, and `uns` are copied from the
/// source, and the deletion-vector section is carried through unchanged (the
/// rewrite is 1:1 in obs row space, so its global row indices stay valid); a
/// fresh provenance entry is added. **Not preserved:** `adata.raw`,
/// layers, CSC sidecars, `varm`/`obsp`/`varp`, predicate indexes and detection
/// bitmaps are *not* carried over. Inputs carrying
/// `adata.raw`, and multimodal inputs, are rejected (see SCX-002) rather than
/// silently dropped or corrupted — extract the modality of interest first, or
/// run the transform before attaching raw.
///
/// Post-transformation data is always Float32 + Zstd codec, since
/// normalization produces float values that can't use the integer-only
/// Scx1 codec.
///
/// # Arguments
///
/// * `source_path` — path to the input SCX file
/// * `target_path` — path to write the output SCX file
/// * `config` — preprocessing operations to apply
///
/// # Returns
///
/// The number of shards processed on success.
/// Reject inputs whose full section layout the streaming rewrites cannot
/// faithfully reproduce (SCX-002). Both `streaming_preprocess` and
/// `streaming_save_layer` iterate `shards_sorted()`, which flattens CSR shards
/// across every modality with overlapping obs-row ranges; writing those back
/// against a single assembled obs axis would corrupt X. And neither carries
/// `adata.raw`, so a raw-bearing input would silently lose it. Fail loudly
/// instead of producing a semantically incomplete or corrupt file.
fn reject_unsupported_rewrite(reader: &scx_format_io::ScxReader, op: &str) -> crate::Result<()> {
    if reader.is_multimodal() {
        return Err(crate::EngineError::UnsupportedRewrite {
            op: op.to_string(),
            feature: "multimodal".to_string(),
            remedy: "extract a single modality with `scx subset --modality NAME` first".to_string(),
        });
    }
    if reader.has_raw() {
        return Err(crate::EngineError::UnsupportedRewrite {
            op: op.to_string(),
            feature: "adata.raw-bearing".to_string(),
            remedy: "raw would be dropped; run the transform before attaching raw, \
                     or drop raw first"
                .to_string(),
        });
    }
    Ok(())
}

pub fn streaming_preprocess(
    source_path: &std::path::Path,
    target_path: &std::path::Path,
    config: &PreprocessConfig,
) -> crate::Result<usize> {
    use byteorder::{LittleEndian, WriteBytesExt};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format_io::{ScxReader, ScxWriter};

    let reader = ScxReader::open(source_path)?;
    reject_unsupported_rewrite(&reader, "streaming_preprocess")?;
    let header = build_output_header(reader.header(), CodecId::Zstd as u8);
    let mut writer = ScxWriter::new(target_path, header)?;

    // Copy obs + var (before shards, matching original section order).
    // `read_obs` / `read_var` transparently handle both legacy and
    // Phase 2 row-sharded layouts; for sharded inputs the resulting
    // copy is single-section on the output (acceptable for fused_ops'
    // moderate-size paths).
    match reader.read_obs() {
        Ok(batch) => writer.write_obs(&batch)?,
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }
    match reader.read_var() {
        Ok(batch) => writer.write_var(&batch)?,
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }

    // Stream CSR shards: read → transform → write
    let shards = reader.catalog().shards_sorted();
    let value_encoding = ValueEncoding::Float32;
    let codec_id = CodecId::Zstd;

    for shard_entry in shards.iter() {
        let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;

        let n_rows = indptr.len().saturating_sub(1);
        let n_vars = reader.n_vars() as usize;
        let mut csr = ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data);
        apply_fused_ops(&mut csr, config.normalize_target_sum, config.log1p);

        let out_indptr: Vec<u64> = csr.indptr.iter().map(|&v| v as u64).collect();
        let out_indices: Vec<u32> = csr.indices.iter().map(|&v| v as u32).collect();
        let mut out_values = Vec::with_capacity(csr.data.len() * 4);
        for &v in &csr.data {
            out_values.write_f32::<LittleEndian>(v).unwrap();
        }

        let row_start = shard_entry.stats.as_ref().map_or(0u64, |s| s.row_start);

        writer.write_csr_shard(
            &out_indptr,
            &out_indices,
            &out_values,
            codec_id,
            value_encoding,
            row_start,
        )?;
    }

    let n_shards = shards.len();

    // Copy obsm + uns (after shards)
    match reader.read_all_obsm() {
        Ok(map) => {
            for (name, batch) in &map {
                writer.write_obsm(name, batch)?;
            }
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }
    match reader.read_uns() {
        Ok(json) => writer.write_uns(&json)?,
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }

    // Provenance
    let ops_desc = format!(
        "preprocess(normalize={}, log1p={})",
        config
            .normalize_target_sum
            .map_or("none".to_string(), |v| v.to_string()),
        config.log1p
    );
    write_preprocess_provenance(
        &mut writer,
        "streaming_preprocess",
        &format!("{{\"ops\":\"{ops_desc}\"}}"),
    )?;
    carry_deletion_vectors(&reader, &mut writer)?;

    writer.finish()?;

    Ok(n_shards)
}

/// Streaming shard-by-shard save-as-layer pipeline.
///
/// Copies `obs`, `var`, the original `X`, `obsm`, `uns` and the deletion-vector
/// section from the source SCX file to a new file, then adds the transformed X
/// data as a named layer (`LayerCsrShard` entries) plus a fresh provenance
/// entry.
///
/// **Not preserved:** pre-existing layers, `adata.raw`, CSC sidecars,
/// `varm`/`obsp`/`varp`, predicate indexes and detection
/// bitmaps are *not* carried over. Inputs carrying `adata.raw`, and multimodal
/// inputs, are rejected (see SCX-002) rather than silently dropped or
/// corrupted.
///
/// # Arguments
///
/// * `source_path` — path to the input SCX file
/// * `target_path` — path to write the output SCX file (must differ from source)
/// * `layer_name` — name for the new layer (e.g., "normalized")
/// * `config` — preprocessing operations to apply
pub fn streaming_save_layer(
    source_path: &std::path::Path,
    target_path: &std::path::Path,
    layer_name: &str,
    config: &PreprocessConfig,
) -> crate::Result<usize> {
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format_io::{ScxReader, ScxWriter};

    let reader = ScxReader::open(source_path)?;
    reject_unsupported_rewrite(&reader, "streaming_save_layer")?;
    let header = build_output_header(reader.header(), reader.header().codec_id);
    let mut writer = ScxWriter::new(target_path, header)?;

    // Copy obs + var (Phase 2 sharded → assembled into a single
    // section on the output, same trade-off as the streaming variant
    // above).
    match reader.read_obs() {
        Ok(batch) => writer.write_obs(&batch)?,
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }
    match reader.read_var() {
        Ok(batch) => writer.write_var(&batch)?,
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }

    // Copy original X shards unchanged (raw byte copy — no decode/re-encode)
    let shards = reader.catalog().shards_sorted();
    for (shard_idx, shard_entry) in shards.iter().enumerate() {
        let raw = reader.read_raw_shard_bytes(shard_entry)?;
        let stats = shard_entry
            .stats
            .clone()
            .ok_or_else(|| scx_format_io::ScxError::SectionNotFound("shard stats".into()))?;
        let nnz = stats.nnz;
        writer.write_raw_shard(
            raw,
            scx_format_io::SectionType::CsrShard,
            &format!("X_shard_{shard_idx}"),
            stats,
            nnz,
        )?;
    }

    // Write the transformed layer shards
    let layer_codec = CodecId::Zstd;
    let layer_encoding = ValueEncoding::Float32;

    for (shard_idx, shard_entry) in shards.iter().enumerate() {
        let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;

        let n_rows = indptr.len().saturating_sub(1);
        let n_vars = reader.n_vars() as usize;
        let mut csr = ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data);
        apply_fused_ops(&mut csr, config.normalize_target_sum, config.log1p);

        let out_indptr: Vec<u64> = csr.indptr.iter().map(|&v| v as u64).collect();
        let out_indices: Vec<u32> = csr.indices.iter().map(|&v| v as u32).collect();
        let out_values = encode_f32_values(&csr.data, layer_encoding)?;

        let row_start = shard_entry.stats.as_ref().map_or(0u64, |s| s.row_start);
        let shard = scx_format_io::ShardBuffers::new(
            &out_indptr,
            &out_indices,
            &out_values,
            layer_codec,
            layer_encoding,
        );
        writer.write_layer_csr_shard(layer_name, shard_idx as u32, row_start, shard)?;
    }

    let n_shards = shards.len();

    // Copy obsm + uns
    match reader.read_all_obsm() {
        Ok(map) => {
            for (name, batch) in &map {
                writer.write_obsm(name, batch)?;
            }
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }
    match reader.read_uns() {
        Ok(json) => writer.write_uns(&json)?,
        Err(scx_format_io::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }

    // Provenance
    write_preprocess_provenance(
        &mut writer,
        "streaming_save_layer",
        &format!("{{\"layer_name\":\"{layer_name}\"}}"),
    )?;
    carry_deletion_vectors(&reader, &mut writer)?;

    writer.finish()?;

    Ok(n_shards)
}

/// Encode f32 values to raw LE bytes for the given value encoding.
fn encode_f32_values(data: &[f32], encoding: scx_codec::ValueEncoding) -> crate::Result<Vec<u8>> {
    use byteorder::{LittleEndian, WriteBytesExt};
    match encoding {
        scx_codec::ValueEncoding::Uint8 => Ok(data.iter().map(|&v| v as u8).collect()),
        scx_codec::ValueEncoding::Uint16 => {
            let mut buf = Vec::with_capacity(data.len() * 2);
            for &v in data {
                buf.write_u16::<LittleEndian>(v as u16).unwrap();
            }
            Ok(buf)
        }
        scx_codec::ValueEncoding::Uint32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_u32::<LittleEndian>(v as u32).unwrap();
            }
            Ok(buf)
        }
        scx_codec::ValueEncoding::Float32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_f32::<LittleEndian>(v).unwrap();
            }
            Ok(buf)
        }
        scx_codec::ValueEncoding::Float16 => Err(crate::EngineError::SchemaError {
            column: "layer_values".to_string(),
            reason: "Float16 value encoding is not yet supported in \
                     encode_f32_values; use Float32 encoding instead"
                .to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 6×4 file with rows 1 and 4 logically deleted.
    ///
    /// The deletion vector is written directly rather than via
    /// `scx_ops::mark_deleted` because `scx-ops` depends on this crate, not the
    /// other way round. Same approach as `scx-ops`'s
    /// `optimize_preserves_deletion_vectors`.
    fn write_deleted_fixture(dir: &tempfile::TempDir) -> std::path::PathBuf {
        use arrow::array::{RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use scx_codec::{CodecId, ValueEncoding};
        use scx_format_io::deletion_vectors::DeletionVectors;
        use std::sync::Arc;

        let (n_obs, n_vars) = (6usize, 4usize);
        let path = dir.path().join("deleted.scx");
        let header = scx_format_io::header::FileHeader {
            n_obs: n_obs as u64,
            n_vars: n_vars as u64,
            ..Default::default()
        };
        let mut writer = scx_format_io::ScxWriter::new(&path, header).unwrap();

        let obs = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "cell_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                (0..n_obs).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        writer.write_obs(&obs).unwrap();
        let var = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "gene_id",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(
                (0..n_vars).map(|i| format!("gene_{i}")).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        writer.write_var(&var).unwrap();
        // Two nnz per row so every row survives normalize_total.
        let mut indptr = vec![0u64];
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for row in 0..n_obs {
            indices.push((row % n_vars) as u32);
            indices.push(((row + 1) % n_vars) as u32);
            values.push((row + 1) as u8);
            values.push((row + 2) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        let mut dv = DeletionVectors::new();
        dv.insert_global([1u32, 4]);
        writer.write_deletion_vectors(&dv).unwrap();
        writer.finish().unwrap();
        path
    }

    fn assert_carries_two_deletions(path: &std::path::Path, what: &str) {
        let out = scx_format_io::ScxReader::open(path).unwrap();
        assert!(
            out.header().has_deletion_vectors(),
            "{what}: deletion-vector flag must survive"
        );
        let dv = out
            .read_deletion_vectors()
            .unwrap()
            .unwrap_or_else(|| panic!("{what}: deletion-vector section present"));
        assert_eq!(dv.total_deleted(), 2, "{what}");
        assert!(dv.is_deleted_global(1) && dv.is_deleted_global(4), "{what}");
        // Carried, not applied — the transform is 1:1 in row space.
        assert_eq!(out.n_obs(), 6, "{what}");
        assert_eq!(
            out.read_all_csr_shards_filtered().unwrap().shape.0,
            4,
            "{what}: a deletion-aware read sees 6 - 2 rows"
        );
    }

    /// `pyscx.preprocess` rewrites values, never the row space — so dropping the
    /// deletion vector on the way out silently un-deletes cells, and the output
    /// looks perfectly well-formed while doing it.
    #[test]
    fn streaming_preprocess_carries_deletion_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_deleted_fixture(&dir);
        let out = dir.path().join("pp.scx");
        let config = PreprocessConfig {
            normalize_target_sum: Some(1e4),
            log1p: true,
        };
        streaming_preprocess(&src, &out, &config).unwrap();
        assert_carries_two_deletions(&out, "streaming_preprocess");
    }

    /// `pyscx.save_layer` keeps the original X byte-for-byte and adds a
    /// transformed layer beside it, so it is even more obviously 1:1.
    #[test]
    fn streaming_save_layer_carries_deletion_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let src = write_deleted_fixture(&dir);
        let out = dir.path().join("layered.scx");
        let config = PreprocessConfig {
            normalize_target_sum: Some(1e4),
            log1p: true,
        };
        streaming_save_layer(&src, &out, "normalized", &config).unwrap();
        assert_carries_two_deletions(&out, "streaming_save_layer");
    }

    /// Helper: create a test CSR matrix.
    ///
    /// 3×5 matrix:
    /// row 0: [0, 5, 0, 10, 0]   → nnz at cols 1,3; values 5,10; row_sum=15
    /// row 1: [1, 0, 3, 0, 7]    → nnz at cols 0,2,4; values 1,3,7; row_sum=11
    /// row 2: [0, 0, 2, 0, 0]    → nnz at col 2; value 2; row_sum=2
    fn sample_csr() -> ScxCsr {
        ScxCsr::new(
            (3, 5),
            vec![0, 2, 5, 6],
            vec![1, 3, 0, 2, 4, 2],
            vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0],
        )
        .unwrap()
    }

    // ---- normalize_row tests ----

    #[test]
    fn normalize_row_matches_scanpy() {
        // scanpy.pp.normalize_total with target_sum=10000:
        // row 0 sum = 15 → factor = 10000/15 = 666.667
        //   5.0 → 5*666.667 = 3333.333
        //   10.0 → 10*666.667 = 6666.667
        let mut csr = sample_csr();
        let target_sum = 10_000.0;

        normalize_row(&mut csr.data, &csr.indptr, 0, target_sum);

        let expected_0 = (5.0_f64 / 15.0 * target_sum) as f32;
        let expected_1 = (10.0_f64 / 15.0 * target_sum) as f32;
        assert!((csr.data[0] - expected_0).abs() < 1e-3);
        assert!((csr.data[1] - expected_1).abs() < 1e-3);

        // Other rows remain unchanged
        assert_eq!(csr.data[2], 1.0);
        assert_eq!(csr.data[3], 3.0);
        assert_eq!(csr.data[4], 7.0);
        assert_eq!(csr.data[5], 2.0);
    }

    #[test]
    fn normalize_row_all_rows() {
        let mut csr = sample_csr();
        let target = 1.0;
        for row in 0..3 {
            normalize_row(&mut csr.data, &csr.indptr, row, target);
        }
        // After normalizing each row to sum=1, check row sums
        for row in 0..3 {
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            let sum: f64 = csr.data[start..end].iter().map(|&v| v as f64).sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {} sum = {}", row, sum);
        }
    }

    // ---- log1p_row tests ----

    #[test]
    fn log1p_row_matches_numpy() {
        // numpy.log1p(5.0) = ln(6) ≈ 1.7918
        // numpy.log1p(10.0) = ln(11) ≈ 2.3979
        let mut csr = sample_csr();

        log1p_row(&mut csr.data, &csr.indptr, 0);

        let expected_0 = 5.0_f32.ln_1p();
        let expected_1 = 10.0_f32.ln_1p();
        assert!((csr.data[0] - expected_0).abs() < 1e-6);
        assert!((csr.data[1] - expected_1).abs() < 1e-6);

        // Other rows remain unchanged
        assert_eq!(csr.data[2], 1.0);
    }

    #[test]
    fn log1p_row_all_values() {
        let mut csr = sample_csr();
        for row in 0..3 {
            log1p_row(&mut csr.data, &csr.indptr, row);
        }
        // All values should be ln(original + 1)
        let original = [5.0_f32, 10.0, 1.0, 3.0, 7.0, 2.0];
        for (i, &orig) in original.iter().enumerate() {
            let expected = orig.ln_1p();
            assert!(
                (csr.data[i] - expected).abs() < 1e-6,
                "data[{}] = {}, expected {}",
                i,
                csr.data[i],
                expected
            );
        }
    }

    // ---- fused_normalize_log1p tests ----

    #[test]
    fn fused_matches_sequential() {
        // Fused should produce identical results to normalize-then-log1p
        let target = 10_000.0;

        // Sequential path
        let mut sequential = sample_csr();
        for row in 0..3 {
            normalize_row(&mut sequential.data, &sequential.indptr, row, target);
        }
        for row in 0..3 {
            log1p_row(&mut sequential.data, &sequential.indptr, row);
        }

        // Fused path
        let mut fused = sample_csr();
        for row in 0..3 {
            fused_normalize_log1p(&mut fused.data, &fused.indptr, row, target);
        }

        // Compare bit-level: both paths do f64 intermediate, cast to f32
        for i in 0..fused.data.len() {
            assert!(
                (fused.data[i] - sequential.data[i]).abs() < 1e-6,
                "mismatch at [{}]: fused={}, seq={}",
                i,
                fused.data[i],
                sequential.data[i]
            );
        }
    }

    // ---- zero-sum row test ----

    #[test]
    fn zero_sum_row_unchanged() {
        // Edge case: row with very small values that sum to effectively zero.
        // In CSR, rows with truly zero values shouldn't have entries, but
        // an empty row (nnz=0) should be handled gracefully.
        let mut csr = ScxCsr::new(
            (2, 3),
            vec![0, 0, 2], // row 0 is empty, row 1 has 2 entries
            vec![0, 2],
            vec![3.0, 7.0],
        )
        .unwrap();

        // Normalize empty row → no-op (no entries to modify)
        normalize_row(&mut csr.data, &csr.indptr, 0, 10_000.0);
        assert_eq!(csr.data, vec![3.0, 7.0]); // unchanged

        // Fused on empty row → no-op
        fused_normalize_log1p(&mut csr.data, &csr.indptr, 0, 10_000.0);
        assert_eq!(csr.data, vec![3.0, 7.0]); // unchanged

        // log1p on empty row → no-op
        log1p_row(&mut csr.data, &csr.indptr, 0);
        assert_eq!(csr.data, vec![3.0, 7.0]); // unchanged
    }

    // ---- single non-zero value per row ----

    #[test]
    fn single_nonzero_normalize() {
        // Row with a single non-zero value: after normalize, it should equal target_sum.
        let mut csr = ScxCsr::new((1, 5), vec![0, 1], vec![2], vec![42.0]).unwrap();

        normalize_row(&mut csr.data, &csr.indptr, 0, 10_000.0);
        // 42 / 42 * 10000 = 10000
        assert!((csr.data[0] - 10_000.0).abs() < 1e-3);
    }

    #[test]
    fn single_nonzero_fused() {
        let mut csr = ScxCsr::new((1, 5), vec![0, 1], vec![2], vec![42.0]).unwrap();

        fused_normalize_log1p(&mut csr.data, &csr.indptr, 0, 10_000.0);
        // ln(42/42 * 10000 + 1) = ln(10001)
        let expected = (10_001.0_f64).ln() as f32;
        assert!((csr.data[0] - expected).abs() < 1e-3);
    }

    // ---- apply_fused_ops tests ----

    #[test]
    fn apply_fused_ops_normalize_only() {
        let mut csr = sample_csr();
        apply_fused_ops(&mut csr, Some(1.0), false);
        for row in 0..3 {
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            let sum: f64 = csr.data[start..end].iter().map(|&v| v as f64).sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {} sum = {}", row, sum);
        }
    }

    #[test]
    fn apply_fused_ops_log1p_only() {
        let mut csr = sample_csr();
        let original_data = csr.data.clone();
        apply_fused_ops(&mut csr, None, true);
        for (i, &orig) in original_data.iter().enumerate() {
            let expected = orig.ln_1p();
            assert!(
                (csr.data[i] - expected).abs() < 1e-6,
                "data[{}] = {}, expected {}",
                i,
                csr.data[i],
                expected
            );
        }
    }

    #[test]
    fn apply_fused_ops_both() {
        let target = 10_000.0;

        // Sequential path for reference
        let mut expected = sample_csr();
        apply_fused_ops(&mut expected, Some(target), false);
        apply_fused_ops(&mut expected, None, true);

        // Fused path
        let mut actual = sample_csr();
        apply_fused_ops(&mut actual, Some(target), true);

        for i in 0..actual.data.len() {
            assert!(
                (actual.data[i] - expected.data[i]).abs() < 1e-6,
                "mismatch at [{}]: fused={}, seq={}",
                i,
                actual.data[i],
                expected.data[i]
            );
        }
    }

    #[test]
    fn apply_fused_ops_noop() {
        let mut csr = sample_csr();
        let original = csr.data.clone();
        apply_fused_ops(&mut csr, None, false);
        assert_eq!(csr.data, original);
    }
}
