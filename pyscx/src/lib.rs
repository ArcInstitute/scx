mod anndata;
mod experiment;
mod ops;
mod query;

#[cfg(feature = "cloud")]
mod cloud;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use experiment::PyExperiment;
use query::{PyQueryPipeline, PyQueryResult};
use scx_format::ScxError;

/// Convert an ScxError into a Python RuntimeError.
fn to_pyerr(e: ScxError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Open an SCX file and return a PyExperiment handle.
///
/// Example:
///     exp = pyscx.open("data.scx")
///     adata = exp.to_anndata()
#[pyfunction]
fn open(path: &str) -> PyResult<PyExperiment> {
    let reader = scx_format::ScxReader::open(path).map_err(to_pyerr)?;
    Ok(PyExperiment::new(reader, std::path::PathBuf::from(path)))
}

/// Convert an AnnData object to an SCX file.
///
/// Codec defaults to "auto", which selects the best codec based on value
/// distribution (Scx1 for small UMI counts, Zstd for large values or floats).
/// Explicit options: "none", "scx1", "zstd".
///
/// Example:
///     pyscx.from_anndata(adata, "output.scx")
///     pyscx.from_anndata(adata, "output.scx", codec="scx1", shard_size=8192)
#[pyfunction]
#[pyo3(signature = (adata, path, codec=None, shard_size=None))]
fn from_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
) -> PyResult<()> {
    anndata::from_anndata_impl(py, adata, path, codec, shard_size)
}

/// Convert a 10x HDF5 file to SCX via scanpy.
///
/// Reads the 10x file with scanpy.read_10x_h5(), then writes via from_anndata.
#[pyfunction]
#[pyo3(signature = (h5_path, scx_path, codec=None, shard_size=None))]
fn from_10x(
    py: Python<'_>,
    h5_path: &str,
    scx_path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
) -> PyResult<()> {
    let scanpy = py.import("scanpy")?;
    let adata = scanpy.call_method1("read_10x_h5", (h5_path,))?;
    anndata::from_anndata_impl(py, &adata, scx_path, codec, shard_size)
}

/// Convert a Cell Ranger MTX directory to SCX.
///
/// Reads the MTX directory (matrix.mtx[.gz], barcodes.tsv[.gz], features.tsv[.gz])
/// and writes an SCX file.
///
/// Example:
///     pyscx.from_mtx("/path/to/filtered_feature_bc_matrix", "output.scx")
#[pyfunction]
#[pyo3(signature = (mtx_dir, scx_path, codec=None, shard_size=None))]
fn from_mtx(
    mtx_dir: &str,
    scx_path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
) -> PyResult<()> {
    use scx_codec::CodecId;
    use scx_format::header::{FileHeader, MAGIC};
    use scx_format::provenance::ProvenanceEntry;
    use scx_format::select_codec;
    use scx_format::writer::ScxWriter;
    use std::time::{SystemTime, UNIX_EPOCH};

    let mtx_data = scx_mtx::read_mtx_directory(std::path::Path::new(mtx_dir))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let explicit_codec = match codec.unwrap_or("auto") {
        "auto" => None,
        "none" => Some(CodecId::None),
        "scx1" => Some(CodecId::Scx1),
        "zstd" => Some(CodecId::Zstd),
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown codec: '{}'. Use auto, none, scx1, or zstd.",
                other
            )))
        }
    };

    let target_rows = shard_size.unwrap_or(16384);
    let nnz = *mtx_data.indptr.last().unwrap_or(&0) as u64;

    // Detect encoding
    let (value_encoding, codec_id) = {
        let is_integer = mtx_data
            .data
            .iter()
            .all(|&v| v.is_finite() && v >= 0.0 && v == v.floor());
        let encoding = if !is_integer {
            scx_codec::ValueEncoding::Float32
        } else {
            let max_val: f64 = mtx_data
                .data
                .iter()
                .map(|&v| v as f64)
                .fold(0.0f64, f64::max);
            if max_val <= 255.0 {
                scx_codec::ValueEncoding::Uint8
            } else if max_val <= 65535.0 {
                scx_codec::ValueEncoding::Uint16
            } else {
                scx_codec::ValueEncoding::Uint32
            }
        };
        let raw_bytes: Vec<u8> = match encoding {
            scx_codec::ValueEncoding::Uint8 => mtx_data.data.iter().map(|&v| v as u8).collect(),
            scx_codec::ValueEncoding::Uint16 => {
                let mut buf = Vec::with_capacity(mtx_data.data.len() * 2);
                for &v in &mtx_data.data {
                    buf.extend_from_slice(&(v as u16).to_le_bytes());
                }
                buf
            }
            scx_codec::ValueEncoding::Uint32 => {
                let mut buf = Vec::with_capacity(mtx_data.data.len() * 4);
                for &v in &mtx_data.data {
                    buf.extend_from_slice(&(v as u32).to_le_bytes());
                }
                buf
            }
            _ => {
                let mut buf = Vec::with_capacity(mtx_data.data.len() * 4);
                for &v in &mtx_data.data {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf
            }
        };
        let c = match explicit_codec {
            Some(cid) => {
                if cid == CodecId::Scx1 && !encoding.is_integer() {
                    CodecId::Zstd
                } else {
                    cid
                }
            }
            None => select_codec(&raw_bytes, encoding),
        };
        (encoding, c)
    };

    let index_dtype: u8 = if mtx_data.n_vars <= 65535 { 0 } else { 1 };

    let header = FileHeader {
        magic: MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs: mtx_data.n_obs as u64,
        n_vars: mtx_data.n_vars as u64,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: target_rows,
        codec_id: codec_id as u8,
        index_dtype,
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
    };

    let mut writer = ScxWriter::new(std::path::Path::new(scx_path), header).map_err(to_pyerr)?;

    writer.write_obs(&mtx_data.obs).map_err(to_pyerr)?;
    writer.write_var(&mtx_data.var).map_err(to_pyerr)?;

    // Write CSR shards
    let target_rows_usize = target_rows as usize;
    let mut row_start = 0usize;
    while row_start < mtx_data.n_obs {
        let row_end = (row_start + target_rows_usize).min(mtx_data.n_obs);
        let shard_indptr_slice = &mtx_data.indptr[row_start..=row_end];
        let base = shard_indptr_slice[0];
        let shard_indptr: Vec<u64> = shard_indptr_slice
            .iter()
            .map(|&v| (v - base) as u64)
            .collect();
        let nnz_start = base as usize;
        let nnz_end = *shard_indptr_slice.last().unwrap() as usize;
        let shard_indices: Vec<u32> = mtx_data.indices[nnz_start..nnz_end]
            .iter()
            .map(|&v| v as u32)
            .collect();
        let shard_data = &mtx_data.data[nnz_start..nnz_end];
        let raw_values: Vec<u8> = match value_encoding {
            scx_codec::ValueEncoding::Uint8 => shard_data.iter().map(|&v| v as u8).collect(),
            scx_codec::ValueEncoding::Uint16 => {
                let mut buf = Vec::with_capacity(shard_data.len() * 2);
                for &v in shard_data {
                    buf.extend_from_slice(&(v as u16).to_le_bytes());
                }
                buf
            }
            scx_codec::ValueEncoding::Uint32 => {
                let mut buf = Vec::with_capacity(shard_data.len() * 4);
                for &v in shard_data {
                    buf.extend_from_slice(&(v as u32).to_le_bytes());
                }
                buf
            }
            _ => {
                let mut buf = Vec::with_capacity(shard_data.len() * 4);
                for &v in shard_data {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf
            }
        };
        writer
            .write_csr_shard(
                &shard_indptr,
                &shard_indices,
                &raw_values,
                codec_id,
                value_encoding,
                row_start as u64,
            )
            .map_err(to_pyerr)?;
        row_start = row_end;
    }

    // Write provenance
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "convert".to_string(),
            tool: "pyscx".to_string(),
            params_json: format!("{{\"input\":\"{}\",\"format\":\"mtx\"}}", mtx_dir),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;
    Ok(())
}

/// Convert an SCX file to a Cell Ranger–style MTX directory.
///
/// Output directory will contain: matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz
///
/// Example:
///     pyscx.to_mtx("data.scx", "/path/to/output_dir")
#[pyfunction]
fn to_mtx(scx_path: &str, output_dir: &str) -> PyResult<()> {
    scx_mtx::write_scx_to_mtx(
        std::path::Path::new(scx_path),
        std::path::Path::new(output_dir),
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn pyscx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core I/O
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(from_anndata, m)?)?;
    m.add_function(wrap_pyfunction!(from_10x, m)?)?;
    m.add_function(wrap_pyfunction!(from_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(to_mtx, m)?)?;

    // File operations (scx-ops)
    m.add_function(wrap_pyfunction!(ops::append, m)?)?;
    m.add_function(wrap_pyfunction!(ops::append_from_anndata, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mark_deleted, m)?)?;
    m.add_function(wrap_pyfunction!(ops::compact, m)?)?;
    m.add_function(wrap_pyfunction!(ops::rollback, m)?)?;
    m.add_function(wrap_pyfunction!(ops::merge, m)?)?;

    // Cloud operations (optional, behind "cloud" feature)
    #[cfg(feature = "cloud")]
    {
        m.add_function(wrap_pyfunction!(cloud::pull, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::push, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::cloud_optimize, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::explode, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::pack, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::open_cloud, m)?)?;
        m.add_class::<cloud::PyCloudExperiment>()?;
    }

    // Classes
    m.add_class::<PyExperiment>()?;
    m.add_class::<PyQueryPipeline>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<scx_loader::TrainingDataset>()?;
    Ok(())
}
