// pyscx::preprocess — streaming write-back preprocessing pipeline
//
// Exposes `pyscx.preprocess()` and `pyscx.save_layer()` as Python functions
// that read an SCX file shard-by-shard, apply fused operations (normalize_total,
// log1p), and write the result to a new SCX file or layer.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Streaming preprocess: read source SCX → apply ops shard-by-shard → write new SCX.
///
/// The output file contains the transformed X data with obs, var, obsm, and uns
/// copied from the source. adata.raw, layers, CSC sidecars, varm/obsp/varp,
/// predicate indexes, detection bitmaps, and deletion vectors are NOT carried
/// over; multimodal and raw-bearing inputs are rejected rather than silently
/// altered. Post-transformation data uses Float32+Zstd.
///
/// Parameters
/// ----------
/// source_path : str
///     Path to the input SCX file.
/// target_path : str
///     Path to write the output SCX file.
/// operations : list[str]
///     Operations to apply, in order. Supported: "normalize_total", "log1p".
/// target_sum : float, optional
///     Target sum for normalize_total. Default: 10000.0.
///
/// Returns
/// -------
/// int
///     The number of shards processed.
#[pyfunction]
#[pyo3(signature = (source_path, target_path, operations, target_sum=1e4))]
pub fn preprocess(
    source_path: &str,
    target_path: &str,
    operations: Vec<String>,
    target_sum: f64,
) -> PyResult<usize> {
    let config = parse_operations(&operations, target_sum)?;

    let n_shards = scx_engine::streaming_preprocess(
        std::path::Path::new(source_path),
        std::path::Path::new(target_path),
        &config,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(n_shards)
}

/// Save transformed data as a new layer in a copied SCX file.
///
/// Copies obs, var, the original X, obsm, and uns from the source SCX file,
/// then adds the transformed X data as a named layer (LayerCsrShard entries).
/// Pre-existing layers, adata.raw, CSC sidecars, varm/obsp/varp, predicate
/// indexes, detection bitmaps, and deletion vectors are NOT carried over.
/// Multimodal and raw-bearing inputs are rejected rather than silently
/// altered.
///
/// Parameters
/// ----------
/// source_path : str
///     Path to the input SCX file.
/// target_path : str
///     Path to write the output SCX file (copy with new layer).
/// layer_name : str
///     Name for the new layer (e.g., "normalized").
/// operations : list[str]
///     Operations to apply. Supported: "normalize_total", "log1p".
/// target_sum : float, optional
///     Target sum for normalize_total. Default: 10000.0.
///
/// Returns
/// -------
/// int
///     The number of shards processed.
#[pyfunction]
#[pyo3(signature = (source_path, target_path, layer_name, operations, target_sum=1e4))]
pub fn save_layer(
    source_path: &str,
    target_path: &str,
    layer_name: &str,
    operations: Vec<String>,
    target_sum: f64,
) -> PyResult<usize> {
    let config = parse_operations(&operations, target_sum)?;

    let n_shards = scx_engine::streaming_save_layer(
        std::path::Path::new(source_path),
        std::path::Path::new(target_path),
        layer_name,
        &config,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(n_shards)
}

/// Parse a list of operation names into a PreprocessConfig.
fn parse_operations(
    operations: &[String],
    target_sum: f64,
) -> PyResult<scx_engine::PreprocessConfig> {
    let mut config = scx_engine::PreprocessConfig::default();

    for op in operations {
        match op.as_str() {
            "normalize_total" => {
                config.normalize_target_sum = Some(target_sum);
            }
            "log1p" => {
                config.log1p = true;
            }
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown operation: '{}'. Supported: 'normalize_total', 'log1p'.",
                    other
                )));
            }
        }
    }

    Ok(config)
}
