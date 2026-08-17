mod accel;
pub(crate) mod anndata_hooks;
pub(crate) mod axis_align;
pub(crate) mod backed;
mod convert;
mod experiment;
#[cfg(feature = "hdf5")]
mod h5ad_metadata;
pub(crate) mod lazy_mapping;
pub(crate) mod lazy_transform;
pub(crate) mod mudata;
mod ops;
pub(crate) mod optional_deps;
mod preprocess;
pub(crate) mod projected_agg;
mod query;

#[cfg(feature = "cloud")]
mod cloud;

use pyo3::exceptions::{PyFileNotFoundError, PyPermissionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use experiment::{PyExperiment, PyGroupShard};
use query::{PyQueryPipeline, PyQueryResult};
use scx_format_io::{ScxError, ScxErrorClass, ScxReader};

/// Convert an ScxError into the most appropriate Python exception.
///
/// User-input format errors → ValueError; missing files → FileNotFoundError;
/// permission errors → PermissionError; everything else → RuntimeError.
/// Open a reader for a handle that will be **handed to Python** and outlive
/// this call — an `Experiment`, a backed dataset, a lazy mapping, a query
/// pipeline.
///
/// Always watched, so a read after the file is mutated behind the handle's
/// back raises instead of answering from a mapping of what the file used to
/// be. Its counterpart is a bare `ScxReader::open`, which is right for the
/// readers that live and die inside one call — `scx-ops` bracketing its own
/// mutation, a one-shot `validate`, a single GPU shard fetch — and which must
/// stay unwatched or those ops would fail against themselves.
///
/// Calling this rather than `ScxReader::open(..)?.watching()?` at each site is
/// the point: which of the two a site wants is a real decision, and it should
/// be greppable rather than a suffix someone can leave off.
pub(crate) fn open_handle_reader(path: impl AsRef<std::path::Path>) -> Result<ScxReader, ScxError> {
    ScxReader::open(path)?.watching()
}

/// [`open_handle_reader`], reusing an already-parsed `FullCatalog`. Same
/// contract; see `ScxReader::open_with_shared_catalog` for why the sharing
/// exists.
pub(crate) fn open_handle_reader_shared(
    path: impl AsRef<std::path::Path>,
    catalog: std::sync::Arc<scx_format_io::FullCatalog>,
) -> Result<ScxReader, ScxError> {
    ScxReader::open_with_shared_catalog(path, catalog)?.watching()
}

pub(crate) fn to_pyerr(e: ScxError) -> PyErr {
    use std::io::ErrorKind;
    let msg = e.to_string();
    match e.class() {
        // Bad input / inconsistent data (incl. the stale-sidecar error, whose
        // message already names the fix `scx build-csc` / `--rebuild-csc`).
        ScxErrorClass::Validation => PyValueError::new_err(msg),
        // File looks corrupt or was written by an incompatible/newer SCX.
        // ValueError (not RuntimeError) so callers can distinguish a bad
        // file from a transient runtime failure.
        ScxErrorClass::CorruptFile => PyValueError::new_err(format!(
            "{msg} — the file appears corrupt or was written by an incompatible \
             SCX version; re-run conversion to regenerate it"
        )),
        ScxErrorClass::Io(ErrorKind::NotFound) => PyFileNotFoundError::new_err(msg),
        ScxErrorClass::Io(ErrorKind::PermissionDenied) => PyPermissionError::new_err(msg),
        // A truncated / too-short file is a corruption signal, not a
        // transient runtime failure — surface it as ValueError.
        ScxErrorClass::Io(ErrorKind::UnexpectedEof) => PyValueError::new_err(format!(
            "{msg} — the file appears truncated or is not a valid SCX file"
        )),
        ScxErrorClass::Io(_) | ScxErrorClass::Other => PyRuntimeError::new_err(msg),
    }
}

/// Resolve an optional `shard_size` to a concrete row count, rejecting an
/// explicit zero (SCX-011). A zero shard size stalls every shard-boundary loop
/// (`row_end == (row_start + 0).min(n)` never advances → the writer hangs).
/// `None` falls back to `default`. The CLI already rejects zero via its
/// `positive_u32` parser; this closes the same gap for the Python API.
pub(crate) fn resolve_shard_size(shard_size: Option<u32>, default: u32) -> PyResult<u32> {
    match shard_size {
        Some(0) => Err(PyValueError::new_err(
            "shard_size must be > 0 (0 would stall the shard-writing loop and hang)",
        )),
        Some(n) => Ok(n),
        None => Ok(default),
    }
}

/// Convert a `scx_convert::ConvertError` into the most appropriate
/// Python exception. A missing input file raises `FileNotFoundError`
/// (not `RuntimeError`), a permission failure raises `PermissionError`,
/// an embedded `ScxError` is routed through [`to_pyerr`], and everything
/// else falls back to `RuntimeError`. Used by the h5ad/h5mu conversion
/// entry points so the common "I typed the wrong path" case surfaces as
/// the exception a Python user expects.
#[cfg(feature = "hdf5")]
pub(crate) fn convert_to_pyerr(e: scx_convert::ConvertError) -> PyErr {
    use scx_convert::ConvertError;
    match e {
        ConvertError::Io(io_err) if io_err.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(io_err.to_string())
        }
        ConvertError::Io(io_err) if io_err.kind() == std::io::ErrorKind::PermissionDenied => {
            PyPermissionError::new_err(io_err.to_string())
        }
        ConvertError::Scx(scx) => to_pyerr(scx),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Like [`convert_to_pyerr`] but with the input path in scope, so an existing
/// but non-HDF5 input (libhdf5 "file signature not found" / "unable to open
/// file") is reported as a clean `ValueError` naming the file and the expected
/// format, instead of leaking the raw libhdf5 `RuntimeError` (report E1). The
/// missing-file case is handled by the existence pre-check in the caller and
/// surfaces as `FileNotFoundError`.
#[cfg(feature = "hdf5")]
pub(crate) fn convert_to_pyerr_with_path(e: scx_convert::ConvertError, path: &str) -> PyErr {
    use scx_convert::ConvertError;
    if let ConvertError::Hdf5(ref h5) = e {
        let msg = h5.to_string();
        // This substring match is coupled to libhdf5's
        // error wording and could silently stop matching on a libhdf5 bump,
        // reverting to the opaque RuntimeError below. The fallback is graceful
        // (never wrong, just less friendly). The durable fix is a structured
        // signature-mismatch kind on `ConvertError::Hdf5` upstream in
        // scx-convert; until then, keep both known phrasings here.
        if msg.contains("file signature not found") || msg.contains("unable to open file") {
            return PyValueError::new_err(format!(
                "'{path}' is not a valid HDF5/h5ad file ({msg}). \
                 Expected an .h5ad file written by anndata.",
            ));
        }
    }
    convert_to_pyerr(e)
}

/// Open an SCX file and return an `Experiment` handle.
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
        scx_format_io::ScxReader::open(path)
    } else {
        scx_format_io::ScxReader::open_unchecked(path)
    }
    // Watch the file: an `Experiment` outlives the call that made it, and the
    // module-level mutators (`obs_import`, `compact`, `append`, …) take a path
    // and cannot reach it. Without this the handle answers from a mapping of
    // whatever the file used to be. See `scx_format_io::freshness`.
    .and_then(|r| r.watching())
    .map_err(to_pyerr)?;
    Ok(PyExperiment::new(reader, std::path::PathBuf::from(path)))
}

/// Validate section checksums (and, with `deep`, decode-level integrity) in an SCX file.
///
/// Opens the file with full catalog verification, then computes BLAKE3 of
/// every section's payload bytes and compares against the catalog's stored
/// checksum. This is the section-level integrity check — `pyscx.open()`
/// only verifies the catalog itself. Cost is proportional to the file's
/// total section bytes.
///
/// When `deep=True`, additionally decodes every sparse shard to verify the
/// v3 canonical CSR invariant (sorted column indices, no explicit zeros,
/// consistent indptr). Mirrors `scx validate --deep`. Canonical-CSR checks run
/// only on v3+ files (pre-v3 may legitimately carry unsorted shards). Deep
/// results are appended with `canonical-csr ` prefixed names and report `False`
/// rather than raising.
///
/// Returns a list of (section_name, passed) tuples. Raises RuntimeError if
/// any essential section (obs, var, CsrShard) checksum fails.
///
/// Example:
///     results = pyscx.validate("data.scx", deep=True)
///     for name, passed in results:
///         print(f"{name}: {'OK' if passed else 'FAIL'}")
#[pyfunction]
#[pyo3(signature = (path, deep=false))]
fn validate(py: Python<'_>, path: &str, deep: bool) -> PyResult<Vec<(String, bool)>> {
    let reader = scx_format_io::ScxReader::open(path).map_err(to_pyerr)?;
    let mut results = reader.validate().map_err(to_pyerr)?;
    if deep {
        // Deep validation re-decodes every shard (CPU-bound, pure Rust) — run
        // it off the GIL so other Python threads aren't blocked on large files.
        py.detach(|| deep_validate_into(&reader, &mut results));
    }
    Ok(results)
}

/// Append deep-validation results (canonical-CSR checks) to an existing checksum
/// result list. Shared by the top-level `validate` function and
/// `PyExperiment::validate`.
pub(crate) fn deep_validate_into(
    reader: &scx_format_io::ScxReader,
    results: &mut Vec<(String, bool)>,
) {
    // Canonical-CSR invariant is a v3 guarantee only; skip below v3 to avoid
    // false failures on legitimately-unsorted legacy shards.
    if reader.header().format_version >= 3 {
        for (name, passed) in reader.validate_canonical_csr_shards() {
            results.push((format!("canonical-csr {name}"), passed));
        }
    }
}

/// Convert an AnnData object to an SCX file.
///
/// `codec` is the intent axis (resolved by `scx_format::resolve_codec`):
/// `"auto"` (default) is cost-aware adaptive — per framed integer shard it
/// adopts `ShufDeltaZstd` when it wins by a margin, else the heuristic
/// (`Scx1` for floor-median ≤ 8, `Zstd` above); float always → `Pcodec`.
/// `"fast"` is the decode-max heuristic single-encode; `"compact"` is size-max
/// (adopts `ShufDeltaZstd` on ties; framed). Also accepted: the explicit codec
/// forces `"none"`, `"scx1"`, `"zstd"`, `"lz4"`, `"pcodec"`, `"shufdelta"`, and
/// `"compact-trial"` (framed; requires `row_group_rows > 0`).
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
/// By default `from_anndata()` does not mutate `adata.X`, `adata.layers`,
/// or `adata.raw.X`.
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
/// instead of materialising `X`. Peak memory becomes bounded by one X
/// shard's worth of CSR plus one row-shard per `obsm` / `varm` / `obsp`
/// / `varp` matrix (each shard is `shard_size × k × 4 B` for dense
/// embeddings, `shard_size × density × 16 B` for sparse pairwise).
/// `obs` / `var` are extracted from Python once (typically a few hundred
/// MB at most). In-Python mutations to `obs` / `var` / `uns` / `obsm` /
/// `varm` / `obsp` / `varp` are preserved when detected via top-level
/// key comparison against the on-disk h5py groups; if a user replaced a
/// value under an existing key in place, the on-disk version wins and a
/// `UserWarning` fires listing the sections routed from disk. Layers
/// are always streamed directly from disk; if the in-memory AnnData has
/// layer mutations, a separate `UserWarning` fires (use `from_h5ad`
/// after rewriting an h5ad if you need them preserved). Backed
/// AnnDatas without a resolvable filename (e.g. zarr-backed) raise
/// `NotImplementedError`.
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
/// `csc`: `"off"` (default) emits CSR shards only; `"always"` also writes
///   a CSC (column-major) sidecar; `"auto"` writes one when the dataset is
///   large enough to benefit (`n_obs >= 50000` and `n_vars >= 5000` by
///   default, tunable via the `SCX_CSC_AUTO_OBS_THRESHOLD` /
///   `SCX_CSC_AUTO_VARS_THRESHOLD` env vars). Matches `scx convert --csc`.
///
/// `csc_cols_per_shard`: columns per emitted CSC shard (default 5000).
///   Pass `0` to disable the cap (single CSC shard, memory permitting).
///
/// `memory_budget`: Optional budget that surfaces a `UserWarning` when
///   an obsm / varm / obsp / varp key's estimated peak footprint
///   exceeds the budget. Accepts `None` (no check), an int byte count,
///   or a binary-prefixed size string — `K`/`M`/`G`/`T` or
///   `KiB`/`MiB`/`GiB`/`TiB` (powers of 1024; decimal
///   `KB`/`MB`/`GB`/`TB` is rejected), e.g. `"4G"` / `"512MiB"`.
///   Warn-only — shard size is not derated. Applies to both the
///   in-memory and backed routing paths.
///
/// `force_legacy_metadata`: when True, write obs/var as a single
///   `ObsMetadata` / `VarMetadata` section regardless of size. Default
///   `False`: when `n_obs > shard_size` (or `n_vars > shard_size`)
///   `from_anndata` emits the Phase 2 sharded layout
///   (`ObsMetadataShard` / `VarMetadataShard`). The opt-out preserves
///   the legacy single-section layout for tools that haven't migrated
///   to `ScxReader::read_obs_shard` / `obs_shards()`. Honoured on both
///   branches: the in-memory writer applies it directly, and a backed
///   `X` (which routes through `scx-convert`'s streaming ingest) maps it
///   to `obs_shard_policy = Off`. On the backed branch it governs **obs
///   only** — that path never shards var at any `n_vars`, while the
///   in-memory one shards both.
#[pyfunction]
#[pyo3(signature = (
    adata, path, codec=None, shard_size=None, in_place=false, csc=None,
    csc_cols_per_shard=5000, uns_format="tagged",
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000,
    bitmap="off", memory_budget=None, force_legacy_metadata=false,
    sort_by=None, reverse=false,
    row_group_rows=scx_format_io::DEFAULT_ROW_GROUP_ROWS, row_group_target_nnz=None,
))]
#[allow(clippy::too_many_arguments)]
fn from_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    in_place: bool,
    csc: Option<&str>,
    csc_cols_per_shard: usize,
    uns_format: &str,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    memory_budget: Option<Bound<'_, PyAny>>,
    force_legacy_metadata: bool,
    sort_by: Option<Vec<String>>,
    reverse: bool,
    row_group_rows: Option<u32>,
    row_group_target_nnz: Option<u64>,
) -> PyResult<()> {
    let memory_budget_bytes = convert::parse_memory_budget(memory_budget.as_ref())?;
    let csc = scx_engine::index::resolve_csc_policy(csc, index_preset.as_deref());
    convert::from_anndata_impl(
        py,
        adata,
        path,
        codec,
        shard_size,
        in_place,
        &csc,
        csc_cols_per_shard,
        uns_format,
        index_obs.unwrap_or_default(),
        index_var.unwrap_or_default(),
        index_preset,
        index_auto_threshold,
        bitmap,
        memory_budget_bytes,
        force_legacy_metadata,
        sort_by.unwrap_or_default(),
        reverse,
        row_group_rows,
        row_group_target_nnz,
    )
}

/// Stream an h5ad file directly to SCX without materialising the full
/// X matrix in Python or Rust.
///
/// `pyscx.from_h5ad(path, out)` reads `path` from disk through the
/// `scx-convert` streaming pipeline and writes `out` shard-by-shard.
/// Peak memory is bounded by one X shard plus one row-shard per
/// `obsm` / `varm` / `obsp` / `varp` matrix (plus the always-resident
/// `indptr`, ~80 MB at 10M cells), so this is the recommended entry
/// point for h5ad files larger than node RAM. `obsm` / `varm` /
/// `obsp` / `varp` are read by hyperslab from h5py one row-range at a
/// time, then emitted as row-sharded sections — `obsm`-heavy inputs
/// (e.g. embeddings totalling tens of GB) no longer materialise their
/// matrices in memory. For files that comfortably fit in memory,
/// `from_anndata` remains a touch faster.
///
/// `codec`, `shard_size`, `csc`, and `csc_cols_per_shard` mirror
/// `from_anndata` exactly.
///
/// Internally this routes straight to `scx_convert::h5ad_to_scx_streaming`
/// with the on-disk h5ad path — no `anndata.read_h5ad` call, no
/// backed-AnnData round-trip. obs / var / uns are read via pure-Rust
/// HDF5 (`scx-convert/src/h5ad/read.rs`) when no override is supplied,
/// which dodges anndata's eager `obsm` materialisation on `read_h5ad`
/// (anndata 0.12 reads `obsm` into Python heap on every call, including
/// in `backed='r'` mode). The pure-Rust path stamps the same pandas
/// `index_columns` schema metadata used by the rest of the pipeline so
/// `obs.index` still round-trips through `to_anndata()`.
///
/// Supply `obs_override` / `var_override` / `uns_override` to inject
/// caller-mutated values (typically from
/// `pyscx.read_h5ad_metadata(path)`): row counts are validated against
/// the on-disk X shape; `uns_override` is a full-section replacement.
/// `obsm` / `varm` / `obsp` / `varp` are deliberately *not* exposed as
/// overrides — accepting them would re-introduce the eager-materialisation
/// OOM class this entrypoint exists to avoid.
///
/// `csc="always"` (or `csc="auto"` over a dataset above the size
/// thresholds — `n_obs >= 50000` and `n_vars >= 5000` by default, tunable
/// via `SCX_CSC_AUTO_OBS_THRESHOLD` / `SCX_CSC_AUTO_VARS_THRESHOLD`)
/// performs a two-pass write: the streaming converter emits CSR shards,
/// then `scx_ops::rebuild_csc_inplace` regenerates the CSC sidecar over the
/// just-written file. Peak disk briefly reaches ~2× the output size during
/// the rebuild.
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
///   * `shard_obs` (`"off"` | `"auto"` | `"always"`, default
///     `"auto"`): write obs as row-sharded `ObsMetadataShard`
///     sections. `"auto"` shards when `n_obs > shard_size`, the same
///     threshold `pyscx.from_anndata` and `pyscx.optimize(shard_obs=)`
///     use. Obs axis only — var is always a single section on import.
///     Sharded obs is what the streaming h5ad export path consumes;
///     it does not lower conversion peak memory.
///   * `strict_uns`: when `True`, the first unrepresentable `uns`
///     entry raises; default `False` emits a `UserWarning` per
///     skipped key (`SkippedUnsKey`).
///   * `memory_budget`: caps dense slabs and the CSC external-
///     transpose buffers. Accepts an int byte count or a binary-
///     prefixed size — `K`/`M`/`G`/`T` or `KiB`/`MiB`/`GiB`/`TiB`
///     (powers of 1024); decimal `KB`/`MB`/`GB`/`TB` is rejected to
///     avoid 1000-vs-1024 ambiguity. E.g. `"4G"` / `"512M"` / `"2GiB"`.
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
    path, out, codec=None, shard_size=None, shard_obs="auto", csc=None, csc_cols_per_shard=5000,
    uns_format="tagged", stream=true, strict_uns=false, dense_zero_epsilon=0.0,
    memory_budget=None, temp_dir=None,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000,
    bitmap="off", reader_threads=None, writer_queue_depth=4,
    sort_by=None, reverse=false,
    group_by=None, reference=None, group_target_bytes=None, group_max_bytes=None,
    group_pass="auto",
    obs_override=None, var_override=None, uns_override=None,
    row_group_rows=scx_format_io::DEFAULT_ROW_GROUP_ROWS, row_group_target_nnz=None,
))]
#[allow(clippy::too_many_arguments)]
fn from_h5ad(
    py: Python<'_>,
    path: &str,
    out: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    shard_obs: &str,
    csc: Option<&str>,
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
    sort_by: Option<Vec<String>>,
    reverse: bool,
    group_by: Option<String>,
    reference: Option<Bound<'_, PyAny>>,
    group_target_bytes: Option<Bound<'_, PyAny>>,
    group_max_bytes: Option<Bound<'_, PyAny>>,
    group_pass: &str,
    obs_override: Option<Bound<'_, PyAny>>,
    var_override: Option<Bound<'_, PyAny>>,
    uns_override: Option<Bound<'_, PyAny>>,
    row_group_rows: u32,
    row_group_target_nnz: Option<u64>,
) -> PyResult<()> {
    // A missing input is the common wrong-path case. The converter opens
    // the h5ad via `hdf5::File::open`, which surfaces as `ConvertError::Hdf5`
    // (not `Io`), so `convert_to_pyerr` cannot tell it apart from a real
    // HDF5 failure — pre-check here so the user gets `FileNotFoundError`.
    if !std::path::Path::new(path).exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "no such file: '{path}'"
        )));
    }
    // Resolve the codec intent axis (`auto`/`fast`/`compact` + explicit forces).
    let resolved = scx_format_io::resolve_codec(codec).map_err(PyValueError::new_err)?;
    let explicit_codec = resolved.explicit_codec;
    let codec_trial = resolved.codec_trial;
    let decode_target_parsed = resolved.decode_target;
    if resolved.requires_framing && row_group_rows == 0 {
        return Err(PyValueError::new_err(format!(
            "codec='{}' requires row_group_rows=N with N > 0 \
             (row-group-framed output for random-access-safe reads)",
            resolved.profile
        )));
    }
    let csc = scx_engine::index::resolve_csc_policy(csc, index_preset.as_deref());
    let csc_policy =
        scx_format_io::CscPolicy::parse(&csc).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let uns_format_parsed = convert::parse_uns_format(uns_format)?;
    let shard_target_rows =
        resolve_shard_size(shard_size, scx_format_io::DEFAULT_SHARD_TARGET_ROWS)?;
    let memory_budget_bytes = convert::parse_memory_budget(memory_budget.as_ref())?;
    let bitmap_policy = scx_format_io::BitmapPolicy::parse(bitmap)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Sort-on-convert (Phase 2) / grouped convert: the permuted
    // gather runs only on the streaming path, so either forces streaming on.
    let sort_by = sort_by.unwrap_or_default();
    let group_reference = crate::ops::parse_reference_spec(reference.as_ref())?;
    if group_reference.is_some() && group_by.is_none() {
        return Err(PyValueError::new_err(
            "reference requires group_by to be set",
        ));
    }
    let group_target_bytes_val = convert::parse_memory_budget(group_target_bytes.as_ref())?;
    let group_max_bytes_val = convert::parse_memory_budget(group_max_bytes.as_ref())?;
    let group_pass_val =
        scx_convert::GroupPass::parse(group_pass).map_err(PyValueError::new_err)?;
    let stream = stream || !sort_by.is_empty() || group_by.is_some();

    let has_override = obs_override.is_some() || var_override.is_some() || uns_override.is_some();
    if has_override && !stream {
        return Err(PyValueError::new_err(
            "obs_override / var_override / uns_override require stream=True; the \
             non-streaming h5ad_to_scx path does not apply overrides",
        ));
    }

    // Build StreamingOverrides from caller-supplied Python objects.
    // Validate obs / var row counts against the on-disk X shape so a
    // mismatched override doesn't silently desynchronise the SCX file.
    // obsm / varm / obsp / varp are intentionally not surfaced on the
    // Python API — accepting them would re-introduce the very obsm OOM
    // class this entrypoint exists to avoid.
    let mut overrides = scx_convert::StreamingOverrides::default();
    if obs_override.is_some() || var_override.is_some() {
        let path_buf = std::path::PathBuf::from(path);
        let mut shape_sink = scx_convert::WarningSink::log();
        let (n_obs_disk, n_vars_disk, _x_format) = py
            .detach(|| scx_convert::read_h5ad_x_shape_from_path(&path_buf, &mut shape_sink))
            .map_err(|e| PyRuntimeError::new_err(format!("read X shape '{path}': {e}")))?;
        convert::emit_python_warnings(py, &shape_sink)?;
        if let Some(obs) = obs_override.as_ref() {
            let rows: usize = obs.getattr("shape")?.get_item(0)?.extract()?;
            if rows != n_obs_disk {
                return Err(PyValueError::new_err(format!(
                    "obs_override has {rows} rows but X has n_obs={n_obs_disk}"
                )));
            }
            overrides.obs = Some(convert::pandas_to_record_batch(py, obs)?);
        }
        if let Some(var) = var_override.as_ref() {
            let rows: usize = var.getattr("shape")?.get_item(0)?.extract()?;
            if rows != n_vars_disk {
                return Err(PyValueError::new_err(format!(
                    "var_override has {rows} rows but X has n_vars={n_vars_disk}"
                )));
            }
            overrides.var = Some(convert::pandas_to_record_batch(py, var)?);
        }
    }
    if let Some(uns) = uns_override.as_ref() {
        // Replace-not-merge semantics: the supplied dict is the new uns
        // section in full, matching the existing backed-router behaviour
        // when section_keys_match returns false.
        overrides.uns = Some(convert::uns_py_to_json(py, uns, uns_format_parsed)?);
    }

    // Same parser `pyscx.optimize(shard_obs=)` uses, so the two cannot drift
    // on what `"auto"` means.
    let obs_shard_policy =
        scx_format_io::ObsShardPolicy::parse(shard_obs).map_err(PyValueError::new_err)?;

    let opts = scx_convert::ConvertOptions {
        shard_target_rows,
        codec: explicit_codec,
        csc: csc_policy,
        csc_cols_per_shard,
        // Framing on by default (G=256); `row_group_rows=0` opts out to unframed
        // v3. `ConvertOptions::framing()` treats `Some(0)` as unframed, but store
        // None for 0 so the v4 header bump is clean.
        row_group_rows: (row_group_rows != 0).then_some(row_group_rows),
        row_group_target_nnz,
        codec_trial,
        decode_target: decode_target_parsed,
        tool: "pyscx".into(),
        memory_budget: memory_budget_bytes,
        stream,
        strict_uns,
        dense_zero_epsilon,
        temp_dir: temp_dir.map(std::path::PathBuf::from),
        modalities: None,
        modality_types: Vec::new(),
        index_obs: index_obs.unwrap_or_default(),
        index_var: index_var.unwrap_or_default(),
        index_preset,
        index_auto_threshold,
        bitmap: bitmap_policy,
        obs_shard_policy,
        reader_threads,
        writer_queue_depth,
        sort_by,
        sort_reverse: reverse,
        group_by,
        reference: group_reference,
        group_target_bytes: group_target_bytes_val,
        group_max_bytes: group_max_bytes_val,
        group_pass: group_pass_val,
        // Export-only; this is an ingest direction.
        export_obs_keep_mask: None,
        export_min_counts: None,
    };

    let input = std::path::PathBuf::from(path);
    let output = std::path::PathBuf::from(out);
    let mut sink = scx_convert::WarningSink::log();
    if stream {
        py.detach(|| {
            scx_convert::h5ad_to_scx_streaming(&input, &output, &opts, &overrides, &mut sink)
        })
        .map_err(|e| convert_to_pyerr_with_path(e, path))?;
    } else {
        py.detach(|| scx_convert::h5ad_to_scx(&input, &output, &opts, &mut sink))
            .map_err(|e| convert_to_pyerr_with_path(e, path))?;
    }
    convert::emit_python_warnings(py, &sink)?;
    Ok(())
}

/// Convert a 10x HDF5 file to SCX via scanpy.
///
/// Reads the 10x file with scanpy.read_10x_h5(), then writes via from_anndata.
///
/// `csc`, `csc_cols_per_shard`, and `uns_format` mirror `from_anndata`
/// — see those docs. In particular `csc=None` (the default) resolves to
/// `"off"` unless an accel-ready `index_preset` (`training` /
/// `perturbseq`) upgrades it to `"auto"`; an explicit value always wins.
#[pyfunction]
#[pyo3(signature = (
    h5_path, scx_path, codec=None, shard_size=None, csc=None,
    csc_cols_per_shard=5000, uns_format="tagged",
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000,
    bitmap="off", memory_budget=None, force_legacy_metadata=false,
    row_group_rows=scx_format_io::DEFAULT_ROW_GROUP_ROWS, row_group_target_nnz=None,
))]
#[allow(clippy::too_many_arguments)]
fn from_10x(
    py: Python<'_>,
    h5_path: &str,
    scx_path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    csc: Option<&str>,
    csc_cols_per_shard: usize,
    uns_format: &str,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    memory_budget: Option<Bound<'_, PyAny>>,
    force_legacy_metadata: bool,
    row_group_rows: u32,
    row_group_target_nnz: Option<u64>,
) -> PyResult<()> {
    // The "only rewrite when scanpy itself is missing, otherwise propagate"
    // logic this site pioneered now lives in `optional_deps::import_optional`.
    let scanpy = optional_deps::import_optional(
        py,
        "scanpy",
        optional_deps::EXTRA_10X,
        "pyscx.from_10x()",
        "scanpy",
    )?;
    let adata = scanpy.call_method1("read_10x_h5", (h5_path,))?;
    let memory_budget_bytes = convert::parse_memory_budget(memory_budget.as_ref())?;
    // An explicit `csc` always wins; an unset one is upgraded to `auto` by an
    // accel-ready `index_preset`. Shared with `from_anndata` / `from_h5ad` /
    // `from_h5mu` and the `scx convert` CLI so the front-ends cannot drift —
    // this entry point previously pinned `"off"` and silently opted out.
    let csc = scx_engine::index::resolve_csc_policy(csc, index_preset.as_deref());
    convert::from_anndata_impl(
        py,
        &adata,
        scx_path,
        codec,
        shard_size,
        // `in_place` has no observable effect here: scanpy.read_10x_h5
        // returns a fresh AnnData with no other reference. The kwarg was
        // removed from the public signature (T3.7); pass false.
        false,
        &csc,
        csc_cols_per_shard,
        uns_format,
        index_obs.unwrap_or_default(),
        index_var.unwrap_or_default(),
        index_preset,
        index_auto_threshold,
        bitmap,
        memory_budget_bytes,
        force_legacy_metadata,
        Vec::new(), // sort_by (not exposed on the 10x reader)
        false,      // sort_reverse
        // Framing on by default (G=256); `row_group_rows=0` opts out to unframed.
        (row_group_rows != 0).then_some(row_group_rows),
        row_group_target_nnz,
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
/// `shard_obs` (`"off"` | `"auto"` | `"always"`, default `"auto"`):
/// write the shared outer obs as row-sharded `ObsMetadataShard`
/// sections; `"auto"` shards when `n_obs > shard_size`. Same knob and
/// same threshold as `pyscx.from_h5ad`. Per-modality `var` is always a
/// single section.
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
/// `csc`: the streaming path (default) cannot build per-modality CSC
/// sidecars — `csc="always"` raises and `csc="auto"` degrades to no-CSC
/// with a `UserWarning` when a modality would have qualified. Pass
/// `stream=False` to build per-modality CSC via the non-streaming path
/// (it materializes each modality's X). `csc="off"` (default) is unaffected.
///
/// Example:
///     pyscx.from_h5mu("cite_seq.h5mu", "out.scx")
///     pyscx.from_h5mu("multiome.h5mu", "out.scx",
///                     modalities=["rna", "atac"],
///                     modality_types={"adt": "protein"})
#[cfg(feature = "hdf5")]
#[pyfunction]
#[pyo3(signature = (
    path, out, codec=None, shard_size=None, shard_obs="auto", csc=None, csc_cols_per_shard=5000,
    stream=true, strict_uns=false, memory_budget=None, temp_dir=None,
    modalities=None, modality_types=None,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=1000,
    bitmap="off", reader_threads=None, writer_queue_depth=4,
    row_group_rows=scx_format_io::DEFAULT_ROW_GROUP_ROWS,
))]
#[allow(clippy::too_many_arguments)]
fn from_h5mu(
    py: Python<'_>,
    path: &str,
    out: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    shard_obs: &str,
    csc: Option<&str>,
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
    row_group_rows: u32,
) -> PyResult<()> {
    let csc = scx_engine::index::resolve_csc_policy(csc, index_preset.as_deref());
    mudata::from_h5mu_impl(
        py,
        path,
        out,
        codec,
        shard_size,
        shard_obs,
        &csc,
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
        row_group_rows,
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
///     obs_mask: Boolean numpy array selecting observations to keep.
///         Indexed in the GLOBAL / physical obs row space — its length
///         must equal `pyscx.open(path).n_obs_physical` (the file header
///         count), NOT `.n_obs` (the live, post-deletion count). Rows
///         already logically deleted stay dropped regardless of their
///         entry here: the mask is ANDed with the deletion-vector mask,
///         never substituted for it. Requires stream=True.
///     min_counts: Per-cell total-UMI floor. Keeps rows where
///         `X[i, :].sum() >= min_counts`, computed with one streaming
///         pass over the CSR shards (no materialization) in the same
///         global row space as obs_mask, and ANDed with it. Sums `X`,
///         not a layer — meaningless on an already-normalized matrix.
///         Requires stream=True.
///
/// Example:
///     pyscx.to_h5ad("data.scx", "data.h5ad")
///     pyscx.to_h5ad("cite.scx", "rna.h5ad", modality="rna")
///     # Result-preserving CellBender pre-trim on a raw all-droplet file.
///     pyscx.to_h5ad("raw.scx", "raw_trimmed.h5ad", min_counts=5)
#[cfg(feature = "hdf5")]
#[pyfunction]
#[pyo3(signature = (path, out, stream=true, modality=None, reader_threads=None, writer_queue_depth=4, memory_budget=None, obs_mask=None, min_counts=None))]
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
    obs_mask: Option<numpy::PyReadonlyArray1<'_, bool>>,
    min_counts: Option<f64>,
) -> PyResult<()> {
    use std::path::Path;
    let memory_budget_bytes = convert::parse_memory_budget(memory_budget.as_ref())?;

    // Copy the mask out of numpy while the GIL is held: `PyReadonlyArray1` is
    // GIL-bound and `!Send`, so it cannot cross the `py.detach` boundary below.
    let obs_mask_owned: Option<std::sync::Arc<[bool]>> = match obs_mask {
        None => None,
        Some(arr) => {
            let slice = arr.as_slice().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "obs_mask must be a contiguous 1-D bool array",
                )
            })?;
            Some(std::sync::Arc::from(slice.to_vec()))
        }
    };

    if let Some(mc) = min_counts {
        if !mc.is_finite() || mc < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "min_counts must be a finite non-negative number; got {mc}"
            )));
        }
    }

    // The legacy path applies deletion vectors inside the `scx-format-io`
    // readers rather than through a local mask, so it has nowhere to put a
    // caller mask — and it materializes the whole matrix, defeating the point.
    // Fail loudly rather than silently promoting to the streaming path, whose
    // stricter catalog-stats requirement could turn a working export into a
    // confusing failure.
    if !stream && (obs_mask_owned.is_some() || min_counts.is_some()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "obs_mask / min_counts require stream=True; the legacy materializing \
             export path cannot apply a caller-supplied row mask",
        ));
    }

    // Validate length here so the user gets ValueError rather than the
    // RuntimeError that a ConvertError maps to, and with a message that names
    // the coordinate system.
    if let Some(mask) = &obs_mask_owned {
        let n_obs = scx_format_io::ScxReader::open(Path::new(path))
            .map_err(to_pyerr)?
            .n_obs() as usize;
        if mask.len() != n_obs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "obs_mask length {} does not match the SCX file's physical n_obs {n_obs}. \
                 obs_mask is indexed in the GLOBAL (pre-deletion) obs row space — use \
                 `pyscx.open(path).n_obs_physical`, not `.n_obs` (the post-deletion \
                 live count).",
                mask.len()
            )));
        }
    }

    let opts = scx_convert::ConvertOptions {
        stream,
        tool: "pyscx".into(),
        reader_threads,
        writer_queue_depth,
        memory_budget: memory_budget_bytes,
        export_obs_keep_mask: obs_mask_owned,
        export_min_counts: min_counts,
        ..Default::default()
    };
    py.detach(|| -> Result<(), scx_convert::ConvertError> {
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
                scx_convert::scx_modality_to_h5ad(Path::new(path), Path::new(out), name, &mut sink)
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
    .map_err(convert_to_pyerr)
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
    let memory_budget_bytes = convert::parse_memory_budget(memory_budget.as_ref())?;
    let opts = scx_convert::ConvertOptions {
        stream,
        tool: "pyscx".into(),
        reader_threads,
        writer_queue_depth,
        memory_budget: memory_budget_bytes,
        ..Default::default()
    };
    py.detach(|| -> Result<(), scx_convert::ConvertError> {
        let mut sink = scx_convert::WarningSink::log();
        if stream {
            scx_convert::scx_to_h5mu_streaming(Path::new(path), Path::new(out), &opts, &mut sink)
        } else {
            scx_convert::scx_to_h5mu(Path::new(path), Path::new(out), &mut sink)
        }
    })
    .map_err(convert_to_pyerr)
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
#[pyo3(signature = (mu, path, codec=None, shard_size=None, csc="off", csc_cols_per_shard=5000, codec_per_modality=true, uns_format="tagged", row_group_rows=scx_format_io::DEFAULT_ROW_GROUP_ROWS))]
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
    uns_format: &str,
    row_group_rows: u32,
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
        uns_format,
        row_group_rows,
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
    py: Python<'_>,
    mtx_dir: &str,
    scx_path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
) -> PyResult<()> {
    let orientation = scx_mtx::mtx_to_scx(
        std::path::Path::new(mtx_dir),
        std::path::Path::new(scx_path),
        resolve_shard_size(shard_size, 16384)?,
        codec.unwrap_or("auto"),
        "pyscx",
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // A square matrix can't be disambiguated by dimension, so the reader
    // assumed the Cell Ranger default (features × barcodes) and transposed.
    // Surface that as a catchable warning — if the input was already
    // cells × genes, obs/var are now swapped.
    if orientation == scx_mtx::MtxOrientation::Ambiguous {
        py.import("warnings")?.call_method1(
            "warn",
            (
                "MTX matrix is square, so its orientation is ambiguous; assumed the Cell \
              Ranger default (features × barcodes) and transposed to cells × genes. If \
              your matrix was already cells × genes, obs and var are now swapped — verify \
              the resulting shape and var_names/obs_names.",
            ),
        )?;
    }
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
    #[cfg(feature = "hdf5")]
    m.add_function(wrap_pyfunction!(h5ad_metadata::read_h5ad_metadata, m)?)?;
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

    // Native cell-set collation (state3 "3A hybrid")
    m.add_function(wrap_pyfunction!(scx_loader::collate_cellset_gathered, m)?)?;
    // Seeded per-row count downsample (Phase 1B): the gather-stage primitive, plus
    // the path-identity helper that keys its RNG.
    m.add_function(wrap_pyfunction!(scx_loader::downsample_counts_csr, m)?)?;
    m.add_function(wrap_pyfunction!(scx_loader::downsample_file_identity, m)?)?;
    // Contract version for the native cell-set path. Bump on ANY change to what
    // the consumer's mirrored implementation must match: the kernel's
    // encoder-crop / masking / target semantics, **the accepted preprocess-mode
    // strings**, and the gather stage's value contract (clip, downsample).
    // Consumers (state3) assert this at `rust_collate` setup to fail loudly on
    // version skew. See scx-loader/src/sparse_cellset_collate.rs.
    //
    // v1 -> v2 covers two changes, one of them retroactive:
    //   * #356 moved the accepted mode strings from PFlog v2 (`pflog1ppf_raw`) to
    //     v4 (`pflog_raw` + a required `pflog_alpha`) WITHOUT bumping this — a
    //     different transform behind a version the assertion still called equal.
    //   * Phase 1B added the gather-stage negative clip and the optional seeded
    //     downsample, so the emitted CSR's values are no longer a pass-through.
    m.add("COLLATE_CELLSET_CONTRACT_VERSION", 2u32)?;

    // Build profile ("release" / "debug"). Benchmarks MUST run against a
    // release build — a debug `.so` runs ~4-10x slower uniformly and silently
    // poisons every timing. The comprehensive
    // harness asserts this in preflight (gate_candidate / run_parallel).
    m.add(
        "__build_profile__",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;

    // File operations (scx-ops)
    register_ops(m)?;

    // Cloud operations (optional, behind "cloud" feature)
    #[cfg(feature = "cloud")]
    register_cloud(m)?;

    // Classes
    m.add_class::<PyExperiment>()?;
    m.add_class::<PyGroupShard>()?;
    m.add_class::<PyQueryPipeline>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<scx_loader::TrainingDataset>()?;
    m.add_class::<scx_loader::MultimodalTrainingDataset>()?;
    m.add_class::<scx_loader::IndexPlanDataset>()?;
    m.add_class::<scx_loader::SparseCellSetDataset>()?;
    m.add_class::<backed::ScxBackedSparseDataset>()?;
    m.add_class::<backed::ScxBackedLayerDataset>()?;
    m.add_class::<backed::ScxBackedObsmDataset>()?;
    m.add_class::<backed::ScxBackedMuDataset>()?;
    m.add_class::<backed::ScxBackedMuModality>()?;
    m.add_class::<backed::ScxComparisonResult>()?;
    m.add_class::<lazy_transform::ScxLazyTransformedDataset>()?;
    m.add_class::<lazy_mapping::ScxLazyPairwiseMapping>()?;
    m.add_class::<lazy_mapping::ScxLazyVarmMapping>()?;
    m.add_class::<lazy_mapping::ScxLazyObsmMapping>()?;
    m.add_class::<lazy_mapping::ScxLazyLayersMapping>()?;
    m.add_class::<lazy_mapping::ScxLazyValueIterator>()?;
    m.add_class::<lazy_mapping::ScxLazyItemIterator>()?;
    #[cfg(feature = "hdf5")]
    m.add_class::<h5ad_metadata::PyH5adMetadata>()?;

    // Accelerators submodule.  Functions are grouped by domain into
    // `register_*` helpers so adding a new accelerator only touches one
    // helper — not this ~100-line registration block. Each helper adds
    // every function to the flat `accel_module` (so `pyscx.accel.pca`
    // continues to work — no Python-side API break).
    let accel_module = PyModule::new(m.py(), "accel")?;
    register_gpu(&accel_module)?;
    register_dim_reduction(&accel_module)?;
    register_neighbors(&accel_module)?;
    register_fused(&accel_module)?;
    register_clustering(&accel_module)?;
    register_de(&accel_module)?;
    register_pseudobulk(&accel_module)?;
    register_batch_integration(&accel_module)?;
    register_preprocessing(&accel_module)?;
    register_filtering(&accel_module)?;
    register_hvg(&accel_module)?;
    register_score_genes(&accel_module)?;
    register_pflog(&accel_module)?;
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
    //
    // `ScxBackedObsmDataset` is **dense**, but AnnData's `obsm`
    // (`AxisArrays`) re-validates every value on each public `adata.obsm[key]`
    // access via `coerce_array`, whose accepted-type allowlist for lazy dense
    // arrays is restricted to concrete classes we cannot subclass
    // (`h5py.Dataset` / `zarr.Array` / `dask.array`) plus the `CSRDataset` /
    // `CSCDataset` ABCs. Registering as `CSRDataset` is the only way to let
    // `adata.obsm[key]` return the backed row-gather dataset (so `m[idx]`
    // gathers `O(batch)` rows) rather than raising. The dataset's
    // `__getitem__` returns dense numpy, and it exposes `toarray` / `__array__`
    // so array-style consumers work; CSR-only methods (`.tocsr`) are
    // intentionally absent. See docs/scanpy.md.
    //
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
                    "ScxBackedObsmDataset",
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

    // Teach anndata to subset an SCX handle without materializing it —
    // `as_view` / `_subset` / `to_memory`. This is what lets the mutating
    // accelerators delegate to `_inplace_subset_{obs,var}` instead of
    // reimplementing the axis bookkeeping. See `anndata_hooks`.
    anndata_hooks::register_anndata_subset_hooks(m)?;

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
    m.add_function(wrap_pyfunction!(accel::gpu::gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(accel::gpu::gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(accel::gpu::estimate_gpu_memory, m)?)?;
    m.add_function(wrap_pyfunction!(accel::gpu::gpu_profile_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(accel::gpu::gpu_profile_reset, m)?)?;
    m.add_function(wrap_pyfunction!(accel::profile::cpu_profile_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(accel::profile::cpu_profile_reset, m)?)?;
    #[cfg(feature = "gpu")]
    {
        m.add_function(wrap_pyfunction!(accel::gpu_handoff::gpu_decode_shard, m)?)?;
        m.add_class::<accel::gpu_handoff::GpuShardCsr>()?;
        m.add_class::<accel::gpu_handoff::GpuCsrMatrix>()?;
        m.add_class::<accel::gpu_handoff::CudaArrayView>()?;
    }
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

fn register_fused(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::fused::pca_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(accel::fused::pca_neighbors_umap, m)?)?;
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
    m.add_function(wrap_pyfunction!(accel::nb_glm::nb_glm, m)?)?;
    m.add_function(wrap_pyfunction!(accel::nb_glm::pdex_nb_glm, m)?)?;
    m.add_function(wrap_pyfunction!(accel::nb_glm::nb_glm_profile_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(accel::nb_glm::nb_glm_profile_reset, m)?)?;
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
    m.add_function(wrap_pyfunction!(accel::filtering::subset_var, m)?)?;
    Ok(())
}

fn register_hvg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::hvg::highly_variable_genes, m)?)?;
    Ok(())
}

fn register_score_genes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::score_genes::score_genes, m)?)?;
    Ok(())
}

fn register_pflog(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::pflog::pflog, m)?)?;
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
    m.add_function(wrap_pyfunction!(ops::optimize, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sort, m)?)?;
    m.add_function(wrap_pyfunction!(ops::shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(ops::build_csc, m)?)?;
    m.add_function(wrap_pyfunction!(ops::rollback, m)?)?;
    m.add_function(wrap_pyfunction!(ops::merge, m)?)?;
    m.add_function(wrap_pyfunction!(ops::set_uns, m)?)?;
    m.add_function(wrap_pyfunction!(ops::modify_metadata, m)?)?;
    // Ungated: a delimited-table import must not require libhdf5.
    m.add_function(wrap_pyfunction!(ops::obs_import, m)?)?;
    m.add_function(wrap_pyfunction!(ops::diagnose_obs_key, m)?)?;
    m.add_function(wrap_pyfunction!(ops::doublet_import, m)?)?;
    m.add_function(wrap_pyfunction!(ops::doublet_tools, m)?)?;
    m.add_function(wrap_pyfunction!(ops::doublet_profiles, m)?)?;
    #[cfg(feature = "hdf5")]
    {
        m.add_function(wrap_pyfunction!(ops::cellbender_import, m)?)?;
        m.add_function(wrap_pyfunction!(ops::is_cellbender_h5, m)?)?;
    }
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
    m.add_function(wrap_pyfunction!(cloud::read_cloud, m)?)?;
    m.add_class::<cloud::PyCloudExperiment>()?;
    Ok(())
}
