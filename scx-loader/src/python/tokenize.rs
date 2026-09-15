//! The `pyscx.tokenize` namespace: the W6 kernels over a gathered CSR batch.
//!
//! Every entry takes a whole batch as `(indptr, indices, data)` rather than one
//! row, so the per-row loop runs in Rust on the loader's pool with the GIL
//! released — the point of the namespace is to delete that loop from Python, and
//! a per-row binding would keep it.
//!
//! Buffers are **moved** into numpy (`PyArray1::from_vec` adopts the
//! allocation), so nothing is copied at the boundary.
//!
//! The crop is exposed here without the withheld-gene masking that
//! `collate_cellset_gathered` layers on it: the mask is STATE3's query-panel
//! contract, not a property of a top-K crop, and duplicating its plumbing on a
//! second entry point is how the two would come to disagree. A caller that wants
//! the masked form calls `collate_cellset_gathered`.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

use crate::sparse_cellset::validate_indptr;
use crate::sparse_cellset_collate::PreprocessMode;
use crate::tokenize::bin::{BinEdges, BinTie};
use crate::tokenize::sample::WeightTransform;
use crate::tokenize::{bin, crop, rank, sample, transform, CsrRow};

use super::*;

/// A validated batch: the three CSR slices plus the row count they imply.
struct Batch<'a> {
    indptr: &'a [i64],
    indices: &'a [i32],
    data: &'a [f32],
    n_rows: usize,
}

/// Shared entry checks: contiguity, a well-formed `indptr`, and the row count.
fn rows_of<'a>(
    indptr: &'a PyReadonlyArray1<'a, i64>,
    indices: &'a PyReadonlyArray1<'a, i32>,
    data: &'a PyReadonlyArray1<'a, f32>,
) -> PyResult<Batch<'a>> {
    let err = |e| PyValueError::new_err(format!("array not contiguous: {e}"));
    let indptr = indptr.as_slice().map_err(err)?;
    let indices = indices.as_slice().map_err(err)?;
    let data = data.as_slice().map_err(err)?;
    if indices.len() != data.len() {
        return Err(PyValueError::new_err(format!(
            "indices len {} != data len {}",
            indices.len(),
            data.len()
        )));
    }
    // A public entry taking arbitrary numpy arrays: a non-monotonic or negative
    // `indptr` would slice out of bounds and panic across FFI.
    validate_indptr(indptr, data.len()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n_rows = indptr.len() - 1;
    Ok(Batch {
        indptr,
        indices,
        data,
        n_rows,
    })
}

/// Per-row keys for the seeded kernels.
///
/// Defaults to the row's position in this batch, which is reproducible only for
/// this batch. Pass the file's physical row ids (and the file identity from
/// `pyscx.downsample_file_identity`) so the draw is keyed on **content**: §3's
/// rule, and the reason a resumed run or a different worker count does not
/// silently change the sample.
fn row_keys(rows: Option<PyReadonlyArray1<'_, u64>>, n_rows: usize) -> PyResult<Vec<u64>> {
    match rows {
        None => Ok((0..n_rows as u64).collect()),
        Some(r) => {
            let r = r
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("array not contiguous: {e}")))?;
            if r.len() != n_rows {
                return Err(PyValueError::new_err(format!(
                    "rows len {} != n_rows {n_rows}",
                    r.len()
                )));
            }
            Ok(r.to_vec())
        }
    }
}

/// The version of the kernel contract this build implements.
#[pyfunction]
fn contract_version() -> u32 {
    crate::tokenize::TOKENIZE_CONTRACT_VERSION
}

/// `GENE_MASK` token id for a vocabulary of `n_genes_total` genes.
#[pyfunction]
fn gene_mask_id(n_genes_total: i64) -> i64 {
    crop::gene_mask_id(n_genes_total)
}

/// `PAD` token id for a vocabulary of `n_genes_total` genes.
#[pyfunction]
fn pad_id(n_genes_total: i64) -> i64 {
    crop::pad_id(n_genes_total)
}

/// Top-`k` crop per row, by value descending with gene id ascending on ties.
///
/// Returns `ids` / `values` / `mask` / `pad`, each `[n_rows * k]` flat. Unfilled
/// slots carry the PAD id; an empty row gets a single GENE_MASK token.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, k, n_genes_total))]
fn top_k<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f32>,
    k: usize,
    n_genes_total: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let Batch {
        indptr,
        indices,
        data,
        n_rows,
    } = rows_of(&indptr, &indices, &data)?;
    if k == 0 {
        return Err(PyValueError::new_err("k must be >= 1"));
    }
    let mut ids = vec![0i64; n_rows * k];
    let mut values = vec![0f32; n_rows * k];
    let mut mask = vec![0u8; n_rows * k];
    let mut pad = vec![0u8; n_rows * k];
    py.detach(|| {
        crate::pool::cpu_pool().install(|| {
            ids.par_chunks_mut(k)
                .zip(values.par_chunks_mut(k))
                .zip(mask.par_chunks_mut(k))
                .zip(pad.par_chunks_mut(k))
                .enumerate()
                .for_each_init(Vec::new, |order, (r, (((i, v), m), p))| {
                    let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
                    crop::top_k(
                        &crop::CropIn {
                            row: CsrRow {
                                gene_ids: &indices[lo..hi],
                                values: &data[lo..hi],
                            },
                            emit: &data[lo..hi],
                            withheld: None,
                            hide_readout: false,
                        },
                        &crop::CropConfig { k, n_genes_total },
                        order,
                        &mut crop::CropOut {
                            ids: i,
                            values: v,
                            mask: m,
                            pad: p,
                        },
                    );
                });
        })
    });
    let dict = PyDict::new(py);
    dict.set_item("ids", PyArray1::from_vec(py, ids))?;
    dict.set_item("values", PyArray1::from_vec(py, values))?;
    dict.set_item("mask", PyArray1::from_vec(py, mask))?;
    dict.set_item("pad", PyArray1::from_vec(py, pad))?;
    dict.set_item("n_rows", n_rows)?;
    dict.set_item("k", k)?;
    Ok(dict)
}

/// Rank tokens per row (Geneformer-class).
///
/// `gene_stats` is the per-gene normalisation statistic indexed by global gene
/// id — Geneformer's non-zero-median file. It and `vocabulary_version` are
/// inputs, never inferred, and the returned `norm_identity` is the 64-bit stamp
/// over both: record it with the run, because "rank order under statistics S at
/// vocabulary V" is only reproducible if S and V are named.
///
/// Returns `ids` `[n_rows * l_max]` and `lengths` `[n_rows]`. Slots past a row's
/// length are **undefined**, not padded: this kernel reports a length and the
/// consumer owns its padding token.
///
/// ⚠️ Ties break by gene id ascending. Geneformer's `np.argsort` default is
/// quicksort, which is not stable, so its order within an equal-value run
/// follows no rule and cannot be reproduced. See `docs/tokenize.md`.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, gene_stats, l_max, vocabulary_version, target_sum=1e4))]
#[allow(clippy::too_many_arguments)]
fn rank_tokens<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f32>,
    gene_stats: PyReadonlyArray1<'py, f32>,
    l_max: usize,
    vocabulary_version: String,
    target_sum: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let Batch {
        indptr,
        indices,
        data,
        n_rows,
    } = rows_of(&indptr, &indices, &data)?;
    if l_max == 0 {
        return Err(PyValueError::new_err("l_max must be >= 1"));
    }
    let stats = gene_stats
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("array not contiguous: {e}")))?;
    let norm = rank::PerGeneNorm::new(stats.into(), vocabulary_version)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let identity = norm.identity();

    let mut ids = vec![0i64; n_rows * l_max];
    let mut lengths = vec![0u32; n_rows];
    let out: Result<(), String> = py.detach(|| {
        crate::pool::cpu_pool().install(|| {
            ids.par_chunks_mut(l_max)
                .zip(lengths.par_iter_mut())
                .enumerate()
                .map(|(r, (slot, len))| {
                    let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
                    let mut scratch = (Vec::new(), Vec::new());
                    let n = rank::rank_tokens(
                        CsrRow {
                            gene_ids: &indices[lo..hi],
                            values: &data[lo..hi],
                        },
                        &norm,
                        target_sum,
                        &mut scratch.0,
                        &mut scratch.1,
                        slot,
                    )
                    .map_err(|e| e.to_string())?;
                    *len = n as u32;
                    Ok(())
                })
                .collect()
        })
    });
    out.map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("ids", PyArray1::from_vec(py, ids))?;
    dict.set_item("lengths", PyArray1::from_vec(py, lengths))?;
    dict.set_item("norm_identity", identity)?;
    dict.set_item("n_rows", n_rows)?;
    dict.set_item("l_max", l_max)?;
    Ok(dict)
}

/// Bin each row's expressed values (scGPT-class). Returns `bins` `[nnz]`, flat
/// and parallel to `data`.
///
/// `edges=None` recomputes per-cell quantile edges for every row, as scGPT's
/// `Preprocessor` does; supplying `edges` pins them corpus-wide, in which case
/// they are part of the tokeniser's identity and must be recorded with it.
///
/// `tie` picks how a value sitting exactly on an edge is placed:
/// `"left"` / `"right"` are `np.digitize`'s two deterministic bounds, and
/// `"seeded"` is the reference's randomised interpolation keyed on
/// `(seed, file_identity, row)` instead of numpy's global RNG.
///
/// ⚠️ scGPT's own `binning` is `"seeded"`-shaped but draws from numpy's global
/// RNG, so no output of this kernel reproduces its draws. The claim that does
/// hold is bracketing: every value it can emit lies between `"right"` and
/// `"left"`. See `docs/tokenize.md`.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_bins, edges=None, tie="left", seed=0,
                    file_identity=0, rows=None))]
#[allow(clippy::too_many_arguments)]
fn bin_values<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f32>,
    n_bins: usize,
    edges: Option<PyReadonlyArray1<'py, f64>>,
    tie: &str,
    seed: u64,
    file_identity: u64,
    rows: Option<PyReadonlyArray1<'py, u64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let Batch {
        indptr,
        data,
        n_rows,
        ..
    } = rows_of(&indptr, &indices, &data)?;
    let keys = row_keys(rows, n_rows)?;
    let edge_owner = edges;
    let fixed: Option<&[f64]> = match edge_owner.as_ref() {
        Some(e) => Some(
            e.as_slice()
                .map_err(|err| PyValueError::new_err(format!("array not contiguous: {err}")))?,
        ),
        None => None,
    };
    let make_tie = |row: u64| match tie {
        "left" => Ok(BinTie::Left),
        "right" => Ok(BinTie::Right),
        "seeded" => Ok(BinTie::SeededUniform {
            seed,
            file_identity,
            row,
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown tie {other:?} (expected \"left\", \"right\" or \"seeded\")"
        ))),
    };
    make_tie(0)?;

    let mut bins = vec![0i64; data.len()];
    let out: Result<(), String> = py.detach(|| {
        crate::pool::cpu_pool().install(|| {
            let spans: Vec<(usize, usize)> = (0..n_rows)
                .map(|r| (indptr[r] as usize, indptr[r + 1] as usize))
                .collect();
            split_by_spans(&mut bins, &spans)
                .into_par_iter()
                .enumerate()
                .map(|(r, slot)| {
                    let (lo, hi) = spans[r];
                    let t = match tie {
                        "right" => BinTie::Right,
                        "seeded" => BinTie::SeededUniform {
                            seed,
                            file_identity,
                            row: keys[r],
                        },
                        _ => BinTie::Left,
                    };
                    bin::bin_values(
                        &data[lo..hi],
                        match fixed {
                            Some(e) => BinEdges::Fixed(e),
                            None => BinEdges::PerCellQuantile,
                        },
                        n_bins,
                        t,
                        &mut Vec::new(),
                        slot,
                    )
                    .map_err(|e| e.to_string())
                })
                .collect()
        })
    });
    out.map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("bins", PyArray1::from_vec(py, bins))?;
    dict.set_item("n_rows", n_rows)?;
    dict.set_item("n_bins", n_bins)?;
    Ok(dict)
}

/// Expression-weighted gene sampling per row (UCE-class), **with replacement**.
///
/// Returns `ids` `[n_rows * n]` and `lengths` `[n_rows]`; a row with no positive
/// weight reports length 0 and leaves its slots untouched.
///
/// ⚠️ UCE draws from numpy's global RNG, so this reproduces its algorithm and
/// distribution, never its draws. Pass the file's real row ids in `rows` and its
/// `pyscx.downsample_file_identity` so the draw is keyed on content.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n, seed, file_identity, weight="log1p", rows=None))]
#[allow(clippy::too_many_arguments)]
fn sample_genes<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f32>,
    n: usize,
    seed: u64,
    file_identity: u64,
    weight: &str,
    rows: Option<PyReadonlyArray1<'py, u64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let Batch {
        indptr,
        indices,
        data,
        n_rows,
    } = rows_of(&indptr, &indices, &data)?;
    if n == 0 {
        return Err(PyValueError::new_err("n must be >= 1"));
    }
    let weight = match weight {
        "log1p" => WeightTransform::Log1p,
        "linear" => WeightTransform::Linear,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown weight {other:?} (expected \"log1p\" or \"linear\")"
            )))
        }
    };
    let keys = row_keys(rows, n_rows)?;

    let mut ids = vec![0i64; n_rows * n];
    let mut lengths = vec![0u32; n_rows];
    let out: Result<(), String> = py.detach(|| {
        crate::pool::cpu_pool().install(|| {
            ids.par_chunks_mut(n)
                .zip(lengths.par_iter_mut())
                .enumerate()
                .map(|(r, (slot, len))| {
                    let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
                    let drawn = sample::sample_genes(
                        CsrRow {
                            gene_ids: &indices[lo..hi],
                            values: &data[lo..hi],
                        },
                        weight,
                        seed,
                        file_identity,
                        keys[r],
                        &mut Vec::new(),
                        slot,
                    )
                    .map_err(|e| e.to_string())?;
                    *len = drawn as u32;
                    Ok(())
                })
                .collect()
        })
    });
    out.map_err(PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("ids", PyArray1::from_vec(py, ids))?;
    dict.set_item("lengths", PyArray1::from_vec(py, lengths))?;
    dict.set_item("n_rows", n_rows)?;
    dict.set_item("n", n)?;
    Ok(dict)
}

/// Apply one of the collator's preprocess modes to a batch, returning a new
/// `data` array of the same length.
///
/// `mode` is the same string `collate_cellset_gathered` accepts:
/// `"pass_through"`, `"log1p_raw"`, `"normalize_log1p"`, `"pflog_raw"`.
/// `pflog_raw` requires `pflog_alpha` and centres by `n_measured`, which must
/// then be the measured panel size — not the row's non-zero count.
#[pyfunction]
#[pyo3(signature = (indptr, data, mode, target_sum=1e4, pflog_alpha=None, n_measured=None))]
fn transform_values<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f32>,
    mode: &str,
    target_sum: f64,
    pflog_alpha: Option<f64>,
    n_measured: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let err = |e| PyValueError::new_err(format!("array not contiguous: {e}"));
    let indptr = indptr.as_slice().map_err(err)?;
    let data = data.as_slice().map_err(err)?;
    validate_indptr(indptr, data.len()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n_rows = indptr.len() - 1;
    let mode = PreprocessMode::parse(mode).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (alpha, measured) = if mode == PreprocessMode::PflogRaw {
        let a = pflog_alpha.ok_or_else(|| {
            PyValueError::new_err(
                "pflog_raw requires pflog_alpha (there is no dataset to estimate \
                                   from here)",
            )
        })?;
        if a <= 0.0 || !a.is_finite() {
            return Err(PyValueError::new_err(format!(
                "pflog_alpha must be positive and finite, got {a}"
            )));
        }
        let m = n_measured.ok_or_else(|| {
            PyValueError::new_err("pflog_raw requires n_measured (the centring denominator)")
        })?;
        if m == 0 {
            return Err(PyValueError::new_err("n_measured must be >= 1"));
        }
        (a, m)
    } else {
        (1.0, 1)
    };

    let mut out = vec![0f32; data.len()];
    py.detach(|| {
        crate::pool::cpu_pool().install(|| {
            let spans: Vec<(usize, usize)> = (0..n_rows)
                .map(|r| (indptr[r] as usize, indptr[r + 1] as usize))
                .collect();
            split_by_spans(&mut out, &spans)
                .into_par_iter()
                .enumerate()
                .for_each(|(r, slot)| {
                    let (lo, hi) = spans[r];
                    let src = &data[lo..hi];
                    match mode {
                        PreprocessMode::PassThrough => transform::pass_through(src, slot),
                        PreprocessMode::Log1pRaw => transform::log1p_raw(src, slot),
                        PreprocessMode::NormalizeLog1p => transform::normalize_log1p(
                            src,
                            slot,
                            target_sum,
                            transform::library_size(src),
                        ),
                        PreprocessMode::PflogRaw => {
                            transform::pflog_raw(src, slot, alpha, measured)
                        }
                    }
                });
        })
    });
    Ok(PyArray1::from_vec(py, out))
}

/// Per-row library size — the sum of each row's values, negatives clipped.
///
/// ⚠️ This is the sum of the row **as given**. On a panel-projected batch that is
/// the library size *after* feature filtering and is not the cell's sequencing
/// depth; a model whose normalisation statistic was computed against whole-cell
/// depth must not be fed this.
#[pyfunction]
fn library_size<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let err = |e| PyValueError::new_err(format!("array not contiguous: {e}"));
    let indptr = indptr.as_slice().map_err(err)?;
    let data = data.as_slice().map_err(err)?;
    validate_indptr(indptr, data.len()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n_rows = indptr.len() - 1;
    let mut out = vec![0f64; n_rows];
    py.detach(|| {
        crate::pool::cpu_pool().install(|| {
            out.par_iter_mut().enumerate().for_each(|(r, slot)| {
                let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
                *slot = transform::library_size(&data[lo..hi]);
            });
        })
    });
    Ok(PyArray1::from_vec(py, out))
}

/// Which panel positions each row measured: `1` where the row carries that gene.
///
/// Returns `[n_rows * panel.len()]` flat. "Measured", not "non-zero" — a gene the
/// row does not carry is absent from the CSR, which on a heterogeneous panel is
/// not the same claim as a zero count.
#[pyfunction]
fn measured_mask<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    panel: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let err = |e| PyValueError::new_err(format!("array not contiguous: {e}"));
    let indptr = indptr.as_slice().map_err(err)?;
    let indices = indices.as_slice().map_err(err)?;
    let panel = panel.as_slice().map_err(err)?;
    validate_indptr(indptr, indices.len()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n_rows = indptr.len() - 1;
    let width = panel.len().max(1);
    let mut out = vec![0u8; n_rows * panel.len()];
    py.detach(|| {
        crate::pool::cpu_pool().install(|| {
            out.par_chunks_mut(width).enumerate().for_each(|(r, slot)| {
                let (lo, hi) = (indptr[r] as usize, indptr[r + 1] as usize);
                transform::measured_mask(&indices[lo..hi], panel, slot);
            });
        })
    });
    Ok(PyArray1::from_vec(py, out))
}

/// Split a flat buffer into one mutable slice per row span.
///
/// The spans come straight from a validated `indptr`, so they are contiguous,
/// non-overlapping and in order — which is what makes `split_at_mut` sound here.
fn split_by_spans<'a, T>(buf: &'a mut [T], spans: &[(usize, usize)]) -> Vec<&'a mut [T]> {
    let mut rest = buf;
    let mut cursor = 0usize;
    let mut out = Vec::with_capacity(spans.len());
    for &(lo, hi) in spans {
        debug_assert_eq!(lo, cursor, "spans must tile the buffer in order");
        let (head, tail) = rest.split_at_mut(hi - lo);
        out.push(head);
        rest = tail;
        cursor = hi;
    }
    out
}

/// Register every kernel onto the `pyscx.tokenize` submodule.
///
/// Grouped into one helper for the same reason `pyscx`'s `register_*` helpers
/// are: adding a kernel touches this function, not a hundred-line block in
/// `pyscx/src/lib.rs`.
pub fn register_tokenize(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "CONTRACT_VERSION",
        crate::tokenize::TOKENIZE_CONTRACT_VERSION,
    )?;
    m.add_function(wrap_pyfunction!(contract_version, m)?)?;
    m.add_function(wrap_pyfunction!(gene_mask_id, m)?)?;
    m.add_function(wrap_pyfunction!(pad_id, m)?)?;
    m.add_function(wrap_pyfunction!(top_k, m)?)?;
    m.add_function(wrap_pyfunction!(rank_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(bin_values, m)?)?;
    m.add_function(wrap_pyfunction!(sample_genes, m)?)?;
    m.add_function(wrap_pyfunction!(transform_values, m)?)?;
    m.add_function(wrap_pyfunction!(library_size, m)?)?;
    m.add_function(wrap_pyfunction!(measured_mask, m)?)?;
    Ok(())
}
