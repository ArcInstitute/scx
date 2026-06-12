//! PFlog1pPF / shifted-CLR normalization — Booeshaghi et al. 2026 (CPU-native).
//!
//! The exact transform `Z = delta + baseline·1ᵀ` is dense, but it decomposes
//! into a sparse `delta` (= the lazy `NormalizeTotal{1/c}→Log1p` chain) plus a
//! per-cell `baseline`. This binding:
//!
//! * always writes the per-cell `baseline` to `adata.obs[baseline_key]`;
//! * `store ∈ {"pca","all"}` runs baseline-aware out-of-core randomized PCA
//!   (`scx_accel::pflog1ppf_pca`) → `adata.obsm[obsm_key]` + singular values in
//!   `adata.uns[f"{obsm_key}_singular_values"]`;
//! * `store ∈ {"dense","all"}` materializes the exact dense `Z`. Without
//!   `out=` it lands in `adata.layers[layer_out]`, guarded to in-memory-feasible
//!   sizes (`dense_max_elems`). With `out=<scx path>` it is streamed shard-by-shard
//!   to a new SCX file (Phase 4c, no size guard) in one of two reprs:
//!   `store_repr="delta_baseline"` (default, compact: Pcodec `delta` X + `baseline`
//!   obs column, reconstruct-on-read) or `store_repr="dense"` (literal full-density
//!   CSR, forced Zstd, small default `shard_size`).
//!
//! CPU-only: `device` is accepted for API symmetry but there is no GPU kernel.
//! PFlog1pPF requires **raw counts**; an already-transformed (lazy) `X` is
//! rejected (the raw-count guard).

use std::sync::Arc;

use numpy::{PyArray, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scx_accel::{pflog1ppf_baseline_from_delta, pflog1ppf_pca, PcaResult};
use scx_codec::CodecId;
use scx_format_io::section::SectionType;
use scx_format_io::shard_source::SingleShardSource;
use scx_format_io::{ModalityType, ProvenanceEntry, ScxWriter, ShardSource};
use scx_sparse::ScxCsr;

use crate::backed::ScxBackedSparseDataset;
use crate::convert::dtype::UnsFormat;
use crate::convert::from_anndata::build_output_header;
use crate::convert::h5ad::extract_scx_overrides;
use crate::convert::scx_to_scx::{for_each_coo_shard, for_each_dense_shard};
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};
use crate::to_pyerr;

use super::hvg::build_shard_source;
use super::util::extract_materialized_csr;

/// Default guard for the in-memory dense layer (`n_obs · n_vars` elements).
const DEFAULT_DENSE_MAX_ELEMS: usize = 200_000_000;

/// Default output shard height for `store_repr="dense"` (literal full-density
/// CSR). Kept small — peak RAM per shard ≈ `2 · shard_rows · n_vars · 4 B`, so
/// the 16384 default would be ≈6.5 GB/shard at 50k genes (spec §19.4c-iii).
const DENSE_DEFAULT_SHARD_ROWS: u32 = 2048;

/// Default output shard height for `store_repr="delta_baseline"` (compact,
/// `O(M)` — matches the workspace default).
const DELTA_DEFAULT_SHARD_ROWS: u32 = 16384;

/// Apply PFlog1pPF normalization to `adata`.
#[pyfunction]
#[pyo3(signature = (
    adata,
    c=1.0,
    layer=None,
    *,
    store="pca",
    n_components=50,
    n_oversamples=10,
    n_power_iterations=2,
    zero_center=true,
    random_state=0,
    obsm_key="X_pflog1ppf_pca",
    baseline_key="pflog1ppf_baseline",
    layer_out=None,
    out=None,
    store_repr="delta_baseline",
    shard_size=None,
    dense_max_elems=DEFAULT_DENSE_MAX_ELEMS,
    device="auto",
))]
#[allow(clippy::too_many_arguments)]
pub fn pflog1ppf(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    c: f64,
    layer: Option<&str>,
    store: &str,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    obsm_key: &str,
    baseline_key: &str,
    layer_out: Option<&str>,
    out: Option<&str>,
    store_repr: &str,
    shard_size: Option<u32>,
    dense_max_elems: usize,
    device: &str,
) -> PyResult<()> {
    if c <= 0.0 || c.is_nan() || c.is_infinite() {
        return Err(PyValueError::new_err(format!(
            "pflog1ppf: shift c must be positive and finite, got {c}"
        )));
    }
    let (want_pca, want_dense) = match store {
        "pca" => (true, false),
        "baseline" => (false, false),
        "dense" => (false, true),
        "all" => (true, true),
        other => {
            return Err(PyValueError::new_err(format!(
                "pflog1ppf: unknown store {other:?}; expected \"pca\", \"baseline\", \"dense\", or \"all\""
            )));
        }
    };
    match store_repr {
        "delta_baseline" | "dense" => {}
        other => {
            return Err(PyValueError::new_err(format!(
                "pflog1ppf: unknown store_repr {other:?}; expected \"delta_baseline\" or \"dense\""
            )));
        }
    }
    // `out=` redirects the dense materialization to an SCX file; it is only
    // meaningful when the transform is being materialized (`store` ∈ dense/all).
    if out.is_some() && !want_dense {
        return Err(PyValueError::new_err(
            "pflog1ppf: out= streams the materialized transform to disk; pass \
             store=\"dense\" or store=\"all\" (got store that does not materialize)",
        ));
    }

    // Select the source matrix (layer or X).
    let x = match layer {
        Some(name) => adata.getattr("layers")?.get_item(name)?,
        None => adata.getattr("X")?,
    };

    // ── Build the delta source (NormalizeTotal{1/c} → Log1p) ────────────────
    // Three X kinds: backed (out-of-core), lazy (raw-count guard), in-memory.
    let target_sum = 1.0 / c;

    if let Ok(backed) = x.cast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();
        let reader = Arc::clone(&backed_ref.backed);
        let n_vars = backed_ref.shape_val.1;
        let kept = backed_ref.kept_to_global.clone();
        let col_proj = backed_ref.col_projection_arc();
        // Physical-indexed (projected-aware) raw row sums — what NormalizeTotal
        // divides by (transforms run pre-deletion, pre-projection per shard).
        let phys_row_sums = if let Some(cols) = backed_ref.col_projection() {
            crate::projected_agg::row_sums_projected(&backed_ref.backed, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            backed_ref
                .backed
                .row_sums()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };
        drop(backed_ref);
        let transforms = vec![
            Transform::NormalizeTotal {
                row_sums: Arc::new(phys_row_sums),
                target_sum,
            },
            Transform::Log1p,
        ];
        let source = build_shard_source(&reader, &transforms, &kept, &col_proj, n_vars);
        let disk = DiskOut {
            out,
            store_repr,
            shard_size,
            c,
            source_kind: "backed",
        };
        return run_on_source(
            py,
            adata,
            &source,
            want_pca,
            want_dense,
            n_components,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            obsm_key,
            baseline_key,
            layer_out,
            &disk,
            dense_max_elems,
            device,
        );
    }

    if let Ok(lazy) = x.cast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        if !lazy_ref.transforms.is_empty() {
            return Err(PyValueError::new_err(
                "pflog1ppf requires raw counts, but X is a lazy-transformed dataset that \
                 already carries transforms (e.g. normalize_total / log1p). Run pflog1ppf on \
                 the raw-count X instead.",
            ));
        }
        let reader = Arc::clone(&lazy_ref.backed);
        let n_vars = lazy_ref.shape_val.1;
        let kept = lazy_ref.kept_to_global.clone();
        let col_proj = lazy_ref.col_projection.clone();
        drop(lazy_ref);
        let phys_row_sums = match &col_proj {
            Some(cols) => crate::projected_agg::row_sums_projected(&reader, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            None => reader
                .row_sums()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        let transforms = vec![
            Transform::NormalizeTotal {
                row_sums: Arc::new(phys_row_sums),
                target_sum,
            },
            Transform::Log1p,
        ];
        let source = build_shard_source(&reader, &transforms, &kept, &col_proj, n_vars);
        let disk = DiskOut {
            out,
            store_repr,
            shard_size,
            c,
            source_kind: "lazy",
        };
        return run_on_source(
            py,
            adata,
            &source,
            want_pca,
            want_dense,
            n_components,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            obsm_key,
            baseline_key,
            layer_out,
            &disk,
            dense_max_elems,
            device,
        );
    }

    // In-memory scipy/dense X → build the delta CSR directly, one shard.
    let raw = extract_materialized_csr(py, &x)?;
    let delta = delta_from_raw_csr(&raw, c)?;
    let source = SingleShardSource { csr: &delta };
    let disk = DiskOut {
        out,
        store_repr,
        shard_size,
        c,
        source_kind: "in_memory",
    };
    run_on_source(
        py,
        adata,
        &source,
        want_pca,
        want_dense,
        n_components,
        n_oversamples,
        n_power_iterations,
        zero_center,
        random_state,
        obsm_key,
        baseline_key,
        layer_out,
        &disk,
        dense_max_elems,
        device,
    )
}

/// Build the `delta` CSR `log1p(x_ij / (c·s_i))` from a raw-count CSR.
#[allow(clippy::needless_range_loop)]
fn delta_from_raw_csr(raw: &ScxCsr, c: f64) -> PyResult<ScxCsr> {
    let depths = raw.row_sums();
    let mut delta = raw.clone();
    for r in 0..raw.n_rows() {
        let depth = depths[r];
        if depth <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "pflog1ppf: cell {r} has non-positive depth {depth}; filter empty cells first"
            )));
        }
        let start = delta.indptr[r] as usize;
        let end = delta.indptr[r + 1] as usize;
        for v in &mut delta.data[start..end] {
            if !v.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "pflog1ppf: non-finite count {v} at cell {r}; counts must be finite"
                )));
            }
            if (*v as f64) < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "pflog1ppf: negative count {v} at cell {r}; counts must be non-negative"
                )));
            }
            *v = ((*v as f64) / (c * depth)).ln_1p() as f32;
        }
    }
    Ok(delta)
}

/// Streamed-to-disk materialization params (Phase 4c), bundled to keep the
/// `run_on_source` argument list manageable.
struct DiskOut<'a> {
    /// Target SCX path; `None` ⇒ in-memory dense layer (legacy behavior).
    out: Option<&'a str>,
    /// `"delta_baseline"` (compact) or `"dense"` (literal full-density).
    store_repr: &'a str,
    /// Output shard height override.
    shard_size: Option<u32>,
    /// Shift `c` (provenance only).
    c: f64,
    /// `"backed" | "lazy" | "in_memory"` (provenance only).
    source_kind: &'a str,
}

/// Compute baseline + (optionally) PCA / dense over a delta `ShardSource`,
/// writing results into `adata` and stamping the CPU route.
#[allow(clippy::too_many_arguments)]
fn run_on_source<S: ShardSource>(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    source: &S,
    want_pca: bool,
    want_dense: bool,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    obsm_key: &str,
    baseline_key: &str,
    layer_out: Option<&str>,
    disk: &DiskOut<'_>,
    dense_max_elems: usize,
    device: &str,
) -> PyResult<()> {
    let (n_obs, n_vars) = source.shape();

    // Baseline (always) → adata.obs[baseline_key].
    let baseline = pflog1ppf_baseline_from_delta(source)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let baseline_arr = PyArray::from_slice(py, &baseline);
    adata.getattr("obs")?.set_item(baseline_key, baseline_arr)?;

    if want_pca {
        let result: PcaResult = pflog1ppf_pca(
            source,
            &baseline,
            n_components,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        write_pca(py, adata, &result, obsm_key)?;
    }

    if want_dense {
        if let Some(out_path) = disk.out {
            // Phase 4c — stream the materialized transform to a new SCX file.
            // No in-memory size guard: peak RAM is bounded per shard. The
            // baseline was just stamped into adata.obs, so it rides into the
            // output file's obs automatically (compact reconstruct-on-read).
            stream_pflog1ppf_to_scx(py, adata, source, &baseline, out_path, disk)?;
        } else {
            let n_elems = n_obs.saturating_mul(n_vars);
            if n_elems > dense_max_elems {
                return Err(PyRuntimeError::new_err(format!(
                    "pflog1ppf: dense materialization is {n_obs}×{n_vars} = {n_elems} elements, \
                     over the dense_max_elems={dense_max_elems} guard. The exact PFlog1pPF transform \
                     is dense; pass out=<path.scx> to stream it to disk (any size), use store=\"pca\" \
                     for an out-of-core embedding, or raise dense_max_elems if you have the RAM."
                )));
            }
            let dense = materialize_dense(source, &baseline)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let arr = PyArray2::<f32>::from_vec2(py, &dense)?;
            let key = layer_out.unwrap_or("pflog1ppf");
            adata.getattr("layers")?.set_item(key, arr)?;
        }
    }

    super::route::write_accel_route(
        py,
        adata,
        "pflog1ppf",
        &super::route::cpu_only_exec_info(device),
    )?;
    Ok(())
}

/// Materialize the exact dense `Z = delta + baseline·1ᵀ` (f32, row-major).
fn materialize_dense<S: ShardSource>(
    source: &S,
    baseline: &[f64],
) -> Result<Vec<Vec<f32>>, scx_accel::AccelError> {
    let (n_obs, n_vars) = source.shape();
    let mut dense = vec![vec![0.0f32; n_vars]; n_obs];
    let mut row_base = 0usize;
    for shard_idx in 0..source.n_shards() {
        let csr = source.read_shard(shard_idx)?;
        let rows = csr.n_rows();
        for r in 0..rows {
            let cell = row_base + r;
            let b = baseline[cell] as f32;
            let out = &mut dense[cell];
            for v in out.iter_mut() {
                *v = b;
            }
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            for nz in start..end {
                let col = csr.indices[nz] as usize;
                out[col] += csr.data[nz];
            }
        }
        row_base += rows;
    }
    Ok(dense)
}

/// Write PFlog1pPF PCA results: `obsm[obsm_key]` (embeddings, f32) and
/// `uns[f"{obsm_key}_singular_values"]` (σ_i = sqrt(var_explained_i·(n−1))).
fn write_pca(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &PcaResult,
    obsm_key: &str,
) -> PyResult<()> {
    let embeddings = PyArray2::<f32>::from_vec2(
        py,
        &(0..result.n_obs)
            .map(|i| {
                (0..result.n_components)
                    .map(|j| result.embeddings[i * result.n_components + j] as f32)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>(),
    )?;
    adata.getattr("obsm")?.set_item(obsm_key, embeddings)?;

    let scale = (result.n_obs as f64 - 1.0).max(1.0);
    let singular: Vec<f64> = result
        .variance_explained
        .iter()
        .map(|&ve| (ve * scale).max(0.0).sqrt())
        .collect();
    let sv_arr = PyArray::from_slice(py, &singular);
    adata
        .getattr("uns")?
        .set_item(format!("{obsm_key}_singular_values"), sv_arr)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 4c — streaming materialize-to-SCX
// ─────────────────────────────────────────────────────────────────────────

/// Stream the materialized PFlog1pPF transform over the delta `ShardSource`
/// to a new SCX file, shard-by-shard (never `O(N·D)` in RAM). Two reprs:
///
/// * `"delta_baseline"` (compact): write `delta` as a sparse CSR `X`
///   (Pcodec) — `baseline` already lives in `adata.obs`, written below, so
///   readers reconstruct `Z = delta + baseline[:,None]` exactly.
/// * `"dense"` (literal): write the full-density `Z` as a CSR with forced
///   Zstd float values and a small default `shard_size`.
///
/// Mirrors the metadata-writing body of
/// [`crate::convert::scx_to_scx::route_scx_lazy_to_scx`].
fn stream_pflog1ppf_to_scx<S: ShardSource>(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    source: &S,
    baseline: &[f64],
    out_path: &str,
    disk: &DiskOut<'_>,
) -> PyResult<()> {
    let (n_obs_usize, n_vars_usize) = source.shape();
    let n_obs = n_obs_usize as u64;
    let n_vars = n_vars_usize as u64;
    let n_vars_u32 = u32::try_from(n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {n_vars} exceeds u32::MAX")))?;
    let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };

    let dense = disk.store_repr == "dense";
    let codec = if dense {
        CodecId::Zstd
    } else {
        CodecId::Pcodec
    };
    let shard_rows = disk.shard_size.unwrap_or(if dense {
        DENSE_DEFAULT_SHARD_ROWS
    } else {
        DELTA_DEFAULT_SHARD_ROWS
    });

    let header = build_output_header(
        n_obs,
        n_vars,
        shard_rows,
        codec,
        index_dtype,
        scx_format_io::CURRENT_FORMAT_VERSION,
    );
    let mut writer = ScxWriter::new(out_path, header).map_err(to_pyerr)?;

    // obs (carries `baseline` already) + var.
    let ov = extract_scx_overrides(py, adata, UnsFormat::Tagged)?;
    py.detach(|| -> Result<(), scx_format_io::ScxError> {
        writer.write_obs(&ov.obs)?;
        writer.write_var(&ov.var)?;
        Ok(())
    })
    .map_err(to_pyerr)?;

    // X shards — re-chunk to `shard_rows` windows within each physical shard.
    write_pflog1ppf_x_shards(
        py,
        &mut writer,
        source,
        baseline,
        dense,
        codec,
        n_vars_u32,
        index_dtype,
        shard_rows,
    )?;

    // obsm / varm / obsp / varp / uns (mirror route_scx_lazy_to_scx).
    py.detach(|| -> Result<(), scx_format_io::ScxError> {
        for (k, b) in &ov.obsm {
            for_each_dense_shard(
                b,
                shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varm {
            for_each_dense_shard(
                b,
                shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.obsp {
            for_each_coo_shard(
                b,
                shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varp {
            for_each_coo_shard(
                b,
                shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        if let Some(ref uns_json) = ov.uns {
            writer.write_uns(uns_json)?;
        }
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Provenance.
    let params_json = serde_json::json!({
        "c": disk.c,
        "repr": disk.store_repr,
        "source": disk.source_kind,
    });
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "pflog1ppf".to_string(),
            tool: format!("pyscx {}", env!("CARGO_PKG_VERSION")),
            params_json: params_json.to_string(),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;
    Ok(())
}

/// Stream X shards from the delta `ShardSource`, re-chunking each physical
/// shard into `shard_rows`-row windows so the dense path's peak RAM stays
/// bounded. Each output shard's rows come from a single physical shard.
#[allow(clippy::too_many_arguments)]
fn write_pflog1ppf_x_shards<S: ShardSource>(
    py: Python<'_>,
    writer: &mut ScxWriter,
    source: &S,
    baseline: &[f64],
    dense: bool,
    codec: CodecId,
    n_vars_u32: u32,
    index_dtype: u8,
    shard_rows: u32,
) -> PyResult<()> {
    let n_vars = n_vars_u32 as usize;
    let window = (shard_rows as usize).max(1);
    let mut global_row: u64 = 0;
    let mut out_idx = 0usize;
    for i in 0..source.n_shards() {
        let csr = source
            .read_shard(i)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let rows = csr.n_rows();
        if global_row + rows as u64 > baseline.len() as u64 {
            return Err(PyRuntimeError::new_err(format!(
                "pflog1ppf: shard {i} has {rows} rows, exceeding baseline length {} \
                 at global_row={global_row}",
                baseline.len()
            )));
        }
        let mut r0 = 0usize;
        while r0 < rows {
            let r1 = (r0 + window).min(rows);
            let (indptr, indices, values) = if dense {
                build_dense_window(&csr, baseline, global_row as usize, r0, r1, n_vars)
            } else {
                build_delta_window(&csr, r0, r1)
            };
            let name = format!("X_shard_{out_idx}");
            let pre = py
                .detach(|| {
                    scx_format_io::encode_one_shard(
                        &indptr,
                        &indices,
                        &values,
                        Some(codec),
                        index_dtype,
                        n_vars_u32,
                        global_row,
                        SectionType::CsrShard,
                        ModalityType::Rna,
                        name,
                    )
                })
                .map_err(to_pyerr)?;
            py.detach(|| writer.write_preencoded_shard(pre))
                .map_err(to_pyerr)?;
            global_row += (r1 - r0) as u64;
            out_idx += 1;
            r0 = r1;
        }
    }
    Ok(())
}

/// Slice rows `r0..r1` of a (canonical) delta CSR into a fresh shard-local
/// CSR triple (`indptr` rebased to 0), converting `i64`/`i32` → `u64`/`u32`.
fn build_delta_window(csr: &ScxCsr, r0: usize, r1: usize) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let base = csr.indptr[r0] as usize;
    let end = csr.indptr[r1] as usize;
    let indices: Vec<u32> = csr.indices[base..end].iter().map(|&v| v as u32).collect();
    let values: Vec<f32> = csr.data[base..end].to_vec();
    let indptr: Vec<u64> = (r0..=r1)
        .map(|r| (csr.indptr[r] as usize - base) as u64)
        .collect();
    (indptr, indices, values)
}

/// Build a full-density CSR triple for rows `r0..r1`: `Z = delta + baseline`.
/// Entries that land on exactly `0.0` are omitted — a missing CSR entry
/// reconstructs to `0.0`, so this is lossless and keeps the shard canonical
/// (`is_canonical_csr` forbids explicit zeros). `row_base` is the global row
/// index of `r0` (for `baseline` lookup).
fn build_dense_window(
    csr: &ScxCsr,
    baseline: &[f64],
    row_base: usize,
    r0: usize,
    r1: usize,
    n_vars: usize,
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let wrows = r1 - r0;
    let mut indptr: Vec<u64> = Vec::with_capacity(wrows + 1);
    indptr.push(0);
    let mut indices: Vec<u32> = Vec::with_capacity(wrows * n_vars);
    let mut values: Vec<f32> = Vec::with_capacity(wrows * n_vars);
    let mut rowbuf = vec![0f32; n_vars];
    for wr in 0..wrows {
        let b = baseline[row_base + wr] as f32;
        for v in rowbuf.iter_mut() {
            *v = b;
        }
        let start = csr.indptr[r0 + wr] as usize;
        let end = csr.indptr[r0 + wr + 1] as usize;
        for nz in start..end {
            rowbuf[csr.indices[nz] as usize] += csr.data[nz];
        }
        for (col, &v) in rowbuf.iter().enumerate() {
            if v != 0.0 {
                indices.push(col as u32);
                values.push(v);
            }
        }
        indptr.push(values.len() as u64);
    }
    (indptr, indices, values)
}
