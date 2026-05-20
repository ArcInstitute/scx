mod accel;
mod anndata;
pub(crate) mod backed;
mod experiment;
pub(crate) mod lazy_mapping;
pub(crate) mod lazy_transform;
pub(crate) mod mudata;
mod ops;
mod preprocess;
pub(crate) mod projected_agg;
mod query;

#[cfg(feature = "cloud")]
mod cloud;

use pyo3::exceptions::{PyFileNotFoundError, PyPermissionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use experiment::PyExperiment;
use query::{PyQueryPipeline, PyQueryResult};
use scx_format::ScxError;

/// Convert an ScxError into the most appropriate Python exception.
///
/// User-input format errors → ValueError; missing files → FileNotFoundError;
/// permission errors → PermissionError; everything else → RuntimeError.
pub(crate) fn to_pyerr(e: ScxError) -> PyErr {
    let msg = e.to_string();
    match &e {
        ScxError::InconsistentCsr
        | ScxError::NVarsOverflow(_)
        | ScxError::BlockRowsOverflow(_)
        | ScxError::BlockNnzOverflow(_) => PyValueError::new_err(msg),
        ScxError::Io(io_err) if io_err.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(msg)
        }
        ScxError::Io(io_err) if io_err.kind() == std::io::ErrorKind::PermissionDenied => {
            PyPermissionError::new_err(msg)
        }
        _ => PyRuntimeError::new_err(msg),
    }
}

/// Open an SCX file and return a PyExperiment handle.
///
/// Args:
///     path: Path to the SCX file.
///     verify: Verifies the file header and catalog checksum only.
///         Does NOT re-hash section payload bytes — for full payload
///         integrity (after write, after cloud pull, after transfer)
///         call `pyscx.validate(path)` (or `scx validate`). Default: True.
///
///         More precisely, with `verify=True` pyscx checks the header
///         magic/version and the trailing BLAKE3 checksum over the full
///         catalog. This authenticates the catalog payload (offsets,
///         lengths, per-section checksums) but does not touch section
///         bytes. Set to False for performance-sensitive paths where the
///         file is trusted (e.g., repeated reads of a file that was
///         already validated).
///
/// Example:
///     exp = pyscx.open("data.scx")
///     adata = exp.to_anndata()
///     # Fast open for trusted files:
///     exp = pyscx.open("data.scx", verify=False)
///     # Full per-section integrity check:
///     pyscx.validate("data.scx")
#[pyfunction]
#[pyo3(signature = (path, verify=true))]
fn open(path: &str, verify: bool) -> PyResult<PyExperiment> {
    let reader = if verify {
        scx_format::ScxReader::open(path)
    } else {
        scx_format::ScxReader::open_unchecked(path)
    }
    .map_err(to_pyerr)?;
    Ok(PyExperiment::new(reader, std::path::PathBuf::from(path)))
}

/// Validate all section checksums in an SCX file.
///
/// Opens the file with full catalog verification, then computes BLAKE3 of
/// every section's payload bytes and compares against the catalog's stored
/// checksum. This is the section-level integrity check — `pyscx.open()`
/// only verifies the catalog itself. Cost is proportional to the file's
/// total section bytes.
///
/// Returns a list of (section_name, passed) tuples. Raises RuntimeError if
/// any essential section (obs, var, CsrShard) fails.
///
/// Example:
///     results = pyscx.validate("data.scx")
///     for name, passed in results:
///         print(f"{name}: {'OK' if passed else 'FAIL'}")
#[pyfunction]
fn validate(path: &str) -> PyResult<Vec<(String, bool)>> {
    let reader = scx_format::ScxReader::open(path).map_err(to_pyerr)?;
    reader.validate().map_err(to_pyerr)
}

/// Convert an AnnData object to an SCX file.
///
/// Codec defaults to `"auto"`, which selects per shard via
/// `scx-format/src/codec_select.rs::select_codec()`:
/// Float32/Float16 values → Pcodec; integer values with floor-median ≤ 8
/// → Scx1 (Rice); larger integers → Zstd. Explicit options:
/// `"none"`, `"scx1"`, `"zstd"`, `"lz4"`, `"pcodec"`.
///
/// `adata.uns` is serialized as JSON. The `uns_format` kwarg selects the
/// envelope:
///
/// - `uns_format="tagged"` (default) wraps NumPy arrays / scalars,
///   `tuple`s, structured recarrays, and `pandas.Categorical` / `Index` /
///   `Series` in a `__scx_type__` JSON envelope. Numeric arrays are
///   stored as base64-encoded little-endian bytes, so dtype, shape, and
///   `NaN`/`Inf` round-trip bit-exact. Object/string arrays use a JSON
///   list of strings for portability. Plain dicts/lists/scalars pass
///   through as plain JSON.
/// - `uns_format="plain"` (legacy) collapses NumPy arrays / pandas
///   containers to plain Python lists on readback. Use this only if a
///   downstream pipeline relies on the old list types.
///
/// In both modes, non-finite raw Python `float` values and `bytes` objects
/// still raise `ValueError` rather than being silently coerced — finite
/// floats serialize as JSON numbers either way, and the in-array NaN/Inf
/// case is only handled by the base64 path under `"tagged"`.
///
/// By default `from_anndata()` does not mutate `adata.X` or `adata.layers`.
/// CSR inputs with unsorted indices are copied via
/// `scipy.sparse.csr_matrix.sorted_indices()` so the caller's matrices are
/// untouched. Pass `in_place=True` to allow in-place index sorting on the
/// caller's CSR matrices (saves one allocation per matrix; matches the
/// historical behavior). Note that even with `in_place=False`, dtype
/// conversion of `indptr` / `indices` / `data` to SCX's on-disk types
/// (`int64` / `int32` / `float32`) may allocate fresh numpy arrays — those
/// allocations never alias or mutate the caller's data.
///
/// **Backed AnnData auto-routes to streaming.** When `adata.isbacked` is
/// true and `adata.filename` (or `adata.file.filename`) resolves to a
/// readable h5ad path, the call dispatches to the streaming pipeline
/// instead of materialising `X`. Peak memory becomes bounded by one
/// shard's worth of CSR plus encode buffers, matching `from_h5ad`.
/// In-Python mutations to `obs` / `var` / `uns` / `obsm` / `varm` /
/// `obsp` / `varp` are preserved verbatim — the on-disk values are
/// overridden by the caller's. Layers are streamed directly from disk;
/// if the in-memory AnnData has layer mutations, a `UserWarning` fires
/// (use `from_h5ad` after rewriting an h5ad if you need them
/// preserved). Backed AnnDatas without a resolvable filename (e.g.
/// zarr-backed) raise `NotImplementedError`.
///
/// Example:
///     pyscx.from_anndata(adata, "output.scx")
///     pyscx.from_anndata(adata, "output.scx", codec="scx1", shard_size=8192)
///     pyscx.from_anndata(adata, "output.scx", in_place=True)
///     pyscx.from_anndata(adata, "output.scx", csc="always")
///     # Backed AnnData (auto-streams):
///     backed = sc.read_h5ad("big.h5ad", backed="r")
///     pyscx.from_anndata(backed, "big.scx")
///
/// `csc`: when `"always"`, also writes a CSC (column-major) sidecar.
///   `"off"` (default) emits CSR shards only. No `"auto"` mode — CSC is
///   opt-in by design (matches `scx convert --csc`).
///
/// `csc_cols_per_shard`: columns per emitted CSC shard (default 5000).
///   Pass `0` to disable the cap (single CSC shard, memory permitting).
#[pyfunction]
#[pyo3(signature = (
    adata, path, codec=None, shard_size=None, in_place=false, csc="off",
    csc_cols_per_shard=5000, uns_format="tagged",
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000,
    bitmap="off",
))]
#[allow(clippy::too_many_arguments)]
fn from_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    in_place: bool,
    csc: &str,
    csc_cols_per_shard: usize,
    uns_format: &str,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
) -> PyResult<()> {
    anndata::from_anndata_impl(
        py,
        adata,
        path,
        codec,
        shard_size,
        in_place,
        csc,
        csc_cols_per_shard,
        uns_format,
        index_obs.unwrap_or_default(),
        index_var.unwrap_or_default(),
        index_preset,
        index_auto_threshold,
        bitmap,
    )
}

/// Stream an h5ad file directly to SCX without materialising the full
/// X matrix in Python or Rust.
///
/// `pyscx.from_h5ad(path, out)` reads `path` from disk through the
/// `scx-convert` streaming pipeline and writes `out` shard-by-shard.
/// Peak memory is bounded by one shard's worth of CSR plus encode
/// buffers (plus the always-resident `indptr`, ~80 MB at 10M cells),
/// so this is the recommended entry point for h5ad files larger than
/// node RAM. For files that comfortably fit in memory, `from_anndata`
/// remains a touch faster.
///
/// `codec`, `shard_size`, `csc`, and `csc_cols_per_shard` mirror
/// `from_anndata` exactly.
///
/// Internally this opens the h5ad in `anndata.read_h5ad(path,
/// backed="r")` mode and delegates to the same backed-AnnData router
/// `from_anndata` uses on backed input. Backed mode only reads
/// metadata (obs / var / obsm / varm / uns) into Python; X and layers
/// stay on disk and are streamed shard-by-shard by `scx-convert`.
/// Going through Python's pyarrow on the metadata path preserves the
/// pandas `index_columns` schema metadata so `obs.index` round-trips
/// correctly through SCX → `to_anndata()`.
///
/// `csc="always"` performs a two-pass write: the streaming converter
/// emits CSR shards, then `scx_ops::rebuild_csc_inplace` regenerates
/// the CSC sidecar over the just-written file. Peak disk briefly
/// reaches ~2× the output size during the rebuild.
///
/// Source-layout handling:
///   * CSR-on-disk h5ad: native streaming path.
///   * Dense-on-disk h5ad: row-slab streaming with per-shard
///     sparsification (zero-drop). Use `dense_zero_epsilon` to
///     threshold near-zero values; default `0.0` matches
///     scipy's `csr_matrix(dense)` behaviour.
///   * CSC-on-disk h5ad: in-memory transpose when the file fits the
///     `memory_budget`; otherwise an external bucketed transpose to
///     `temp_dir` (`scipy.sum_duplicates` semantics on duplicate
///     coordinates).
///   * `varm` is preserved; `obsp` / `varp` come through only when
///     the on-disk h5ad has them in a form anndata exposes (matches
///     the non-streaming CLI converter).
///
/// Hardening / index kwargs:
///   * `strict_uns`: when `True`, the first unrepresentable `uns`
///     entry raises; default `False` emits a `UserWarning` per
///     skipped key (`SkippedUnsKey`).
///   * `memory_budget`: `"4G"`, `"512M"`, `"2GiB"`, or bytes. Caps
///     dense slabs and the CSC external-transpose buffers. Binary
///     prefixes only (`K/M/G/T`, `KiB/MiB/GiB/TiB`); decimal
///     `KB/MB/GB/TB` is rejected to avoid ambiguity.
///   * `temp_dir`: directory for CSC external transpose runs.
///     Cleaned on success and on drop; defaults to the system temp.
///   * `index_obs` / `index_var` / `index_preset`
///     (`cellxgene` | `perturbseq` | `training`) /
///     `index_auto_threshold`: materialise predicate indexes at
///     conversion time so `pyscx.open(...).query()` and
///     `scx pull --filter` can pushdown. Forced missing/unsupported
///     columns hard-error; preset misses emit
///     `MissingPresetIndexColumn`. Skipped for multimodal inputs.
///   * `bitmap`: `"off"` | `"auto"` | `"always"`. Writes per-shard
///     gene→local-row roaring bitmap sidecars (`SCXB`). `auto`
///     opts in for sparse X with `n_vars <= 1_000_000` and bitmap
///     size <= 15% of encoded CSR; ATAC modalities are eager under
///     `auto`.
///
/// Example:
///     pyscx.from_h5ad("big.h5ad", "big.scx")
///     pyscx.from_h5ad("big.h5ad", "big.scx", csc="always")
///     pyscx.from_h5ad("big.h5ad", "big.scx",
///                     memory_budget="4G", temp_dir="/scratch",
///                     index_preset="cellxgene", bitmap="auto")
#[cfg(feature = "hdf5")]
#[pyfunction]
#[pyo3(signature = (
    path, out, codec=None, shard_size=None, csc="off", csc_cols_per_shard=5000,
    uns_format="tagged", stream=true, strict_uns=false, dense_zero_epsilon=0.0,
    memory_budget=None, temp_dir=None,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000,
    bitmap="off", reader_threads=None, writer_queue_depth=4,
))]
#[allow(clippy::too_many_arguments)]
fn from_h5ad(
    py: Python<'_>,
    path: &str,
    out: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    csc: &str,
    csc_cols_per_shard: usize,
    uns_format: &str,
    stream: bool,
    strict_uns: bool,
    dense_zero_epsilon: f32,
    memory_budget: Option<Bound<'_, PyAny>>,
    temp_dir: Option<&str>,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
) -> PyResult<()> {
    let explicit_codec = anndata::parse_codec(codec)?;
    let csc_always = match csc {
        "off" => false,
        "always" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid csc value '{other}'; expected 'off' or 'always'"
            )));
        }
    };
    let uns_format_parsed = anndata::parse_uns_format(uns_format)?;
    let shard_target_rows = shard_size.unwrap_or(scx_format::DEFAULT_SHARD_TARGET_ROWS);
    let memory_budget_bytes = anndata::parse_memory_budget(memory_budget.as_ref())?;

    // Open the h5ad in backed mode so obs / var / obsm / varm / uns
    // are extractable as Python objects but X stays on disk. The
    // backed-router then runs the same Python → Arrow conversion path
    // `from_anndata` uses, which embeds pandas metadata in the obs /
    // var schemas — the bit that's missing if we let scx-convert's
    // pure-Rust `read_dataframe_group` build the records.
    let anndata_mod = py.import("anndata")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("backed", "r")?;
    let backed = anndata_mod.call_method("read_h5ad", (path,), Some(&kwargs))?;

    anndata::route_backed_anndata_to_streaming(
        py,
        &backed,
        out,
        explicit_codec,
        shard_target_rows,
        csc_always,
        csc_cols_per_shard,
        uns_format_parsed,
        stream,
        strict_uns,
        dense_zero_epsilon,
        memory_budget_bytes,
        temp_dir,
        index_obs.unwrap_or_default(),
        index_var.unwrap_or_default(),
        index_preset,
        index_auto_threshold,
        bitmap,
        reader_threads,
        writer_queue_depth,
    )
}

/// Convert a 10x HDF5 file to SCX via scanpy.
///
/// Reads the 10x file with scanpy.read_10x_h5(), then writes via from_anndata.
/// `in_place` mirrors the `from_anndata()` parameter; it has no observable
/// effect because the AnnData object returned by scanpy is freshly
/// constructed and has no other reference.
///
/// `csc`, `csc_cols_per_shard`, and `uns_format` mirror `from_anndata`
/// — see those docs.
#[pyfunction]
#[pyo3(signature = (
    h5_path, scx_path, codec=None, shard_size=None, in_place=false, csc="off",
    csc_cols_per_shard=5000, uns_format="tagged",
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000,
    bitmap="off",
))]
#[allow(clippy::too_many_arguments)]
fn from_10x(
    py: Python<'_>,
    h5_path: &str,
    scx_path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    in_place: bool,
    csc: &str,
    csc_cols_per_shard: usize,
    uns_format: &str,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
) -> PyResult<()> {
    let scanpy = py.import("scanpy").map_err(|e| {
        // Only rewrite when scanpy itself is the missing module — if scanpy
        // is installed but one of its transitive deps fails to import, we
        // must propagate the original error so users can diagnose it.
        if e.is_instance_of::<pyo3::exceptions::PyModuleNotFoundError>(py) {
            let missing = e
                .value(py)
                .getattr("name")
                .ok()
                .and_then(|n| n.extract::<String>().ok());
            if missing.as_deref() == Some("scanpy") {
                return pyo3::exceptions::PyModuleNotFoundError::new_err(
                    "pyscx.from_10x() requires scanpy. Install it with: \
                     pip install 'pyscx[10x]'",
                );
            }
        }
        e
    })?;
    let adata = scanpy.call_method1("read_10x_h5", (h5_path,))?;
    anndata::from_anndata_impl(
        py,
        &adata,
        scx_path,
        codec,
        shard_size,
        in_place,
        csc,
        csc_cols_per_shard,
        uns_format,
        index_obs.unwrap_or_default(),
        index_var.unwrap_or_default(),
        index_preset,
        index_auto_threshold,
        bitmap,
    )
}

/// Convert an h5mu file to a multimodal SCX v2 file (path-based,
/// streaming by default).
///
/// Mirrors `pyscx.from_h5ad` for h5mu inputs. When `stream=True`
/// (default) this calls `scx_convert::h5mu_to_scx_streaming`, which
/// processes one shard at a time per modality. Peak memory is
/// bounded by `shard_target_rows × max_n_vars × density × ~16`
/// bytes plus the always-resident outer obs.
///
/// `modalities`: optional list of modality names to include
/// (case-sensitive match against `/mod/{name}`). Unknown names
/// raise `ValueError` with the available list. `None` (default)
/// keeps every modality.
///
/// `modality_types`: optional dict mapping modality name → type
/// string (`"rna"`, `"protein"`, `"atac"`, `"spatial"`,
/// `"methylation"`, `"custom"`). Modalities not listed fall back
/// to inference and trigger a `UserWarning` per modality.
///
/// Example:
///     pyscx.from_h5mu("cite_seq.h5mu", "out.scx")
///     pyscx.from_h5mu("multiome.h5mu", "out.scx",
///                     modalities=["rna", "atac"],
///                     modality_types={"adt": "protein"})
#[cfg(feature = "hdf5")]
#[pyfunction]
#[pyo3(signature = (
    path, out, codec=None, shard_size=None, csc="off", csc_cols_per_shard=5000,
    stream=true, strict_uns=false, memory_budget=None, temp_dir=None,
    modalities=None, modality_types=None,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000,
    bitmap="off", reader_threads=None, writer_queue_depth=4,
))]
#[allow(clippy::too_many_arguments)]
fn from_h5mu(
    py: Python<'_>,
    path: &str,
    out: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    csc: &str,
    csc_cols_per_shard: usize,
    stream: bool,
    strict_uns: bool,
    memory_budget: Option<Bound<'_, PyAny>>,
    temp_dir: Option<&str>,
    modalities: Option<Vec<String>>,
    modality_types: Option<std::collections::HashMap<String, String>>,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
) -> PyResult<()> {
    mudata::from_h5mu_impl(
        py,
        path,
        out,
        codec,
        shard_size,
        csc,
        csc_cols_per_shard,
        stream,
        strict_uns,
        memory_budget,
        temp_dir,
        modalities,
        modality_types,
        index_obs.unwrap_or_default(),
        index_var.unwrap_or_default(),
        index_preset,
        index_auto_threshold,
        bitmap,
        reader_threads,
        writer_queue_depth,
    )
}

/// Convert an SCX file to h5ad.
///
/// Mirrors `pyscx.from_h5ad` in the opposite direction. Streams by
/// default — peak RSS is bounded by one shard's worth of CSR plus
/// encode buffers, matching the ingestion direction. For multimodal
/// SCX files, pass `modality="rna"` to extract a single modality as
/// h5ad; otherwise multimodal inputs raise (use `pyscx.to_h5mu`).
///
/// Args:
///     path: Source SCX file.
///     out: Destination h5ad file.
///     stream: Stream the conversion (default True). Set False for
///         the legacy materializing path.
///     modality: Modality name to extract (only valid on multimodal
///         SCX inputs).
///
/// Example:
///     pyscx.to_h5ad("data.scx", "data.h5ad")
///     pyscx.to_h5ad("cite.scx", "rna.h5ad", modality="rna")
#[cfg(feature = "hdf5")]
#[pyfunction]
#[pyo3(signature = (path, out, stream=true, modality=None, reader_threads=None, writer_queue_depth=4, memory_budget=None))]
#[allow(clippy::too_many_arguments)]
fn to_h5ad(
    py: Python<'_>,
    path: &str,
    out: &str,
    stream: bool,
    modality: Option<&str>,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
    memory_budget: Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    use std::path::Path;
    let memory_budget_bytes = anndata::parse_memory_budget(memory_budget.as_ref())?;
    let opts = scx_convert::ConvertOptions {
        stream,
        tool: "pyscx".into(),
        reader_threads,
        writer_queue_depth,
        memory_budget: memory_budget_bytes,
        ..Default::default()
    };
    py.allow_threads(|| -> Result<(), scx_convert::ConvertError> {
        let mut sink = scx_convert::WarningSink::log();
        match (modality, stream) {
            (Some(name), true) => scx_convert::scx_modality_to_h5ad_streaming(
                Path::new(path),
                Path::new(out),
                name,
                &opts,
                &mut sink,
            ),
            (Some(name), false) => {
                scx_convert::scx_modality_to_h5ad(Path::new(path), Path::new(out), name)
            }
            (None, true) => scx_convert::scx_to_h5ad_streaming(
                Path::new(path),
                Path::new(out),
                &opts,
                &mut sink,
            ),
            (None, false) => scx_convert::scx_to_h5ad(Path::new(path), Path::new(out), &mut sink),
        }
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Convert an SCX file to h5mu.
///
/// Mirrors `pyscx.from_h5mu` in the opposite direction. Streams by
/// default; per-modality `/mod/{name}/X` and any layers are written
/// shard-by-shard. Requires a multimodal SCX file.
///
/// Example:
///     pyscx.to_h5mu("cite.scx", "cite.h5mu")
#[cfg(feature = "hdf5")]
#[pyfunction]
#[pyo3(signature = (path, out, stream=true, reader_threads=None, writer_queue_depth=4, memory_budget=None))]
#[allow(clippy::too_many_arguments)]
fn to_h5mu(
    py: Python<'_>,
    path: &str,
    out: &str,
    stream: bool,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
    memory_budget: Option<Bound<'_, PyAny>>,
) -> PyResult<()> {
    use std::path::Path;
    let memory_budget_bytes = anndata::parse_memory_budget(memory_budget.as_ref())?;
    let opts = scx_convert::ConvertOptions {
        stream,
        tool: "pyscx".into(),
        reader_threads,
        writer_queue_depth,
        memory_budget: memory_budget_bytes,
        ..Default::default()
    };
    py.allow_threads(|| -> Result<(), scx_convert::ConvertError> {
        let mut sink = scx_convert::WarningSink::log();
        if stream {
            scx_convert::scx_to_h5mu_streaming(Path::new(path), Path::new(out), &opts, &mut sink)
        } else {
            scx_convert::scx_to_h5mu(Path::new(path), Path::new(out))
        }
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Convert a `mudata.MuData` object to a multimodal SCX v2 file.
///
/// Mirrors `from_anndata` for multi-modality inputs. The MuData's
/// outer `obs` is written as the global obs section
/// (`modality_id = 0`); each modality under `mu.mod` is registered
/// via `ScxWriter::add_modality` and gets its own `var`,
/// `CsrShard`, and `obsm` entries stamped with that modality's
/// `modality_id`.
///
/// Phase D MVP: emits a single CSR shard per modality. Use
/// `scx build-csc` afterwards to add CSC sidecars (the
/// `csc='always'` shortcut is a Phase D follow-on).
///
/// Example:
///     import mudata as md
///     mu = md.MuData({"rna": rna_adata, "adt": adt_adata})
///     pyscx.from_mudata(mu, "cite_seq.scx")
#[pyfunction]
#[pyo3(signature = (mu, path, codec=None, shard_size=None, csc="off", csc_cols_per_shard=5000, codec_per_modality=true))]
#[allow(clippy::too_many_arguments)]
fn from_mudata(
    py: Python<'_>,
    mu: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    csc: &str,
    csc_cols_per_shard: usize,
    codec_per_modality: bool,
) -> PyResult<()> {
    mudata::from_mudata_impl(
        py,
        mu,
        path,
        codec,
        shard_size,
        csc,
        csc_cols_per_shard,
        codec_per_modality,
    )
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
    scx_mtx::mtx_to_scx(
        std::path::Path::new(mtx_dir),
        std::path::Path::new(scx_path),
        shard_size.unwrap_or(16384),
        codec.unwrap_or("auto"),
        "pyscx",
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
    // Eagerly build the rayon global threadpool and force worker-thread
    // spawn at module import time. Without this, the first parallel accel
    // call (e.g. `calculate_qc_metrics`) pays a ~13 s cold-init tax that
    // scares users into thinking the op is slow.
    //
    // `build_global()` is idempotent — Err on already-initialised is harmless.
    // A trivial `par_iter` forces the worker threads to actually spawn
    // (rayon's pool is otherwise lazy on first use).
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    {
        use rayon::prelude::*;
        let _ = (0..rayon::current_num_threads().max(1))
            .into_par_iter()
            .map(|_| 0u64)
            .sum::<u64>();
    }

    // Core I/O
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(from_anndata, m)?)?;
    #[cfg(feature = "hdf5")]
    m.add_function(wrap_pyfunction!(from_h5ad, m)?)?;
    m.add_function(wrap_pyfunction!(from_10x, m)?)?;
    m.add_function(wrap_pyfunction!(from_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(to_mtx, m)?)?;
    #[cfg(feature = "hdf5")]
    m.add_function(wrap_pyfunction!(from_h5mu, m)?)?;
    m.add_function(wrap_pyfunction!(from_mudata, m)?)?;
    #[cfg(feature = "hdf5")]
    m.add_function(wrap_pyfunction!(to_h5ad, m)?)?;
    #[cfg(feature = "hdf5")]
    m.add_function(wrap_pyfunction!(to_h5mu, m)?)?;

    // Preprocessing pipeline
    m.add_function(wrap_pyfunction!(preprocess::preprocess, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess::save_layer, m)?)?;

    // File operations (scx-ops)
    register_ops(m)?;

    // Cloud operations (optional, behind "cloud" feature)
    #[cfg(feature = "cloud")]
    register_cloud(m)?;

    // Classes
    m.add_class::<PyExperiment>()?;
    m.add_class::<PyQueryPipeline>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<scx_loader::TrainingDataset>()?;
    m.add_class::<scx_loader::MultimodalTrainingDataset>()?;
    m.add_class::<scx_loader::IndexPlanDataset>()?;
    m.add_class::<backed::ScxBackedSparseDataset>()?;
    m.add_class::<backed::ScxBackedLayerDataset>()?;
    m.add_class::<backed::ScxBackedMuDataset>()?;
    m.add_class::<backed::ScxBackedMuModality>()?;
    m.add_class::<backed::ScxComparisonResult>()?;
    m.add_class::<lazy_transform::ScxLazyTransformedDataset>()?;
    m.add_class::<lazy_mapping::ScxLazyPairwiseMapping>()?;
    m.add_class::<lazy_mapping::ScxLazyVarmMapping>()?;
    m.add_class::<lazy_mapping::ScxLazyLayersMapping>()?;

    // Accelerators submodule.  Functions are grouped by domain into
    // `register_*` helpers so adding a new accelerator only touches one
    // helper — not this ~100-line registration block. Each helper adds
    // every function to the flat `accel_module` (so `pyscx.accel.pca`
    // continues to work — no Python-side API break).
    let accel_module = PyModule::new(m.py(), "accel")?;
    register_gpu(&accel_module)?;
    register_dim_reduction(&accel_module)?;
    register_neighbors(&accel_module)?;
    register_clustering(&accel_module)?;
    register_de(&accel_module)?;
    register_pseudobulk(&accel_module)?;
    register_batch_integration(&accel_module)?;
    register_preprocessing(&accel_module)?;
    register_filtering(&accel_module)?;
    register_hvg(&accel_module)?;
    register_col_aggs(&accel_module)?;
    register_eval_metrics(&accel_module)?;
    m.add_submodule(&accel_module)?;

    // Route Rust-side `log::*!` calls through Python's `logging` module so
    // Python users can configure severity/filtering/sinks via the standard
    // `logging.getLogger("pyscx")` API. Initialized once at module import;
    // a subsequent re-import is a no-op.
    let _ = pyo3_log::try_init();

    // Route Rust-side `tracing::*!` events to stderr when RUST_LOG is set.
    // Off by default — `try_init()` is a no-op on subsequent imports and
    // silent without RUST_LOG. The
    // `TrainingPipeline::{new, start_epoch, next_batch, drop, shutdown}`
    // spans + `decode` / `I/O` thread entry/exit traces surface here. We
    // use stderr rather than Python `logging` because
    // `tracing-subscriber → log → pyo3-log` would require an extra bridge
    // layer for negligible gain on a diagnostic path that is already
    // gated.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(true)
        .with_thread_names(true)
        .try_init();

    // Register backed classes as virtual subclasses of anndata.abc.CSRDataset.
    // This makes isinstance(x, CSRDataset) return True so AnnData accepts them.
    // Best-effort: if anndata isn't installed we skip silently (common on
    // stripped-down envs); but if the import succeeds and `register` raises
    // we surface the error via `log::warn!` so a user debugging why
    // `ad.AnnData(X=scx_backed)` rejects the object has actionable output.
    let py = m.py();
    match py.import("anndata.abc") {
        Ok(abc) => match abc.getattr("CSRDataset") {
            Ok(csr_dataset) => {
                for cls_name in [
                    "ScxBackedSparseDataset",
                    "ScxBackedLayerDataset",
                    "ScxLazyTransformedDataset",
                ] {
                    if let Err(err) = csr_dataset.call_method1("register", (m.getattr(cls_name)?,))
                    {
                        log::warn!(
                            "failed to register {cls_name} with anndata.abc.CSRDataset: {err}"
                        );
                    }
                }
            }
            Err(err) => log::warn!(
                "anndata.abc.CSRDataset lookup failed ({err}); backed datasets won't \
                 isinstance-check as CSRDataset."
            ),
        },
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyModuleNotFoundError>(py) => {
            // anndata not installed — scx is usable without it, no warning.
        }
        Err(err) => log::warn!(
            "unexpected error importing anndata.abc ({err}); backed datasets won't \
             isinstance-check as CSRDataset."
        ),
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Accelerator registration helpers
// ---------------------------------------------------------------------------
//
// Each `register_*` helper wires one domain of accelerator functions onto the
// flat `pyscx.accel` module. Grouping keeps the per-domain churn localized
// when a new function is added and documents intent at the call site in
// `pymodule`.  Every helper is a thin PyModule::add_function loop — no Python
// semantics change versus the pre-M7 flat registration.

fn register_gpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::gpu::gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(accel::gpu::estimate_gpu_memory, m)?)?;
    Ok(())
}

fn register_dim_reduction(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::pca::pca, m)?)?;
    m.add_function(wrap_pyfunction!(accel::umap::umap, m)?)?;
    Ok(())
}

fn register_neighbors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::neighbors::neighbors, m)?)?;
    Ok(())
}

fn register_clustering(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::leiden::leiden, m)?)?;
    Ok(())
}

fn register_de(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::de::rank_genes_groups, m)?)?;
    m.add_function(wrap_pyfunction!(accel::de::rank_genes_groups_df, m)?)?;
    m.add_function(wrap_pyfunction!(accel::de::pdex_ref, m)?)?;
    Ok(())
}

fn register_pseudobulk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::pseudobulk::pseudobulk_dex, m)?)?;
    Ok(())
}

fn register_batch_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::harmony::harmony_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(accel::lisi::compute_lisi, m)?)?;
    Ok(())
}

fn register_preprocessing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::preprocessing::normalize_total, m)?)?;
    m.add_function(wrap_pyfunction!(accel::preprocessing::log1p, m)?)?;
    m.add_function(wrap_pyfunction!(
        accel::preprocessing::calculate_qc_metrics,
        m
    )?)?;
    #[cfg(feature = "gpu")]
    m.add_class::<accel::preprocessing::ScxGpuNormalizeMarker>()?;
    Ok(())
}

fn register_filtering(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::filtering::filter_cells, m)?)?;
    m.add_function(wrap_pyfunction!(accel::filtering::filter_genes, m)?)?;
    m.add_function(wrap_pyfunction!(accel::filtering::subset_obs, m)?)?;
    Ok(())
}

fn register_hvg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::hvg::highly_variable_genes, m)?)?;
    Ok(())
}

fn register_col_aggs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::col_aggs::col_sums, m)?)?;
    m.add_function(wrap_pyfunction!(accel::col_aggs::col_nnz, m)?)?;
    m.add_function(wrap_pyfunction!(accel::col_aggs::col_min, m)?)?;
    m.add_function(wrap_pyfunction!(accel::col_aggs::col_max, m)?)?;
    m.add_function(wrap_pyfunction!(accel::col_aggs::col_var, m)?)?;
    Ok(())
}

fn register_eval_metrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::eval_metrics::pseudobulk_means, m)?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::perturbation_metrics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(accel::eval_metrics::energy_distance, m)?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::energy_distance_details,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::discrimination_score,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::knockdown_efficiency,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::clustering_agreement,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::adjusted_mutual_info,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::normalized_mutual_info,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::adjusted_rand_index,
        m
    )?)?;
    Ok(())
}

fn register_ops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ops::append, m)?)?;
    m.add_function(wrap_pyfunction!(ops::append_from_anndata, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mark_deleted, m)?)?;
    m.add_function(wrap_pyfunction!(ops::compact, m)?)?;
    m.add_function(wrap_pyfunction!(ops::rollback, m)?)?;
    m.add_function(wrap_pyfunction!(ops::merge, m)?)?;
    Ok(())
}

#[cfg(feature = "cloud")]
fn register_cloud(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cloud::pull, m)?)?;
    m.add_function(wrap_pyfunction!(cloud::push, m)?)?;
    m.add_function(wrap_pyfunction!(cloud::cloud_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(cloud::explode, m)?)?;
    m.add_function(wrap_pyfunction!(cloud::pack, m)?)?;
    m.add_function(wrap_pyfunction!(cloud::open_cloud, m)?)?;
    m.add_class::<cloud::PyCloudExperiment>()?;
    Ok(())
}
