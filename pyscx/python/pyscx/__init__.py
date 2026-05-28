"""pyscx — Python bindings for SCX (Sparse Cell eXpression System).

This module re-exports everything from the native Rust extension module
and provides pure-Python integration packages (e.g., scx_integrations).
"""

from importlib.metadata import (
    PackageNotFoundError as _PkgNotFound,
    version as _pkg_version,
)

try:
    __version__ = _pkg_version("pyscx")
except _PkgNotFound:
    # Editable install before `maturin develop` has materialised
    # distribution metadata — keep a sentinel rather than raising.
    __version__ = "0.0.0+dev"
del _pkg_version, _PkgNotFound

# Re-export everything from the native Rust extension module.
# The compiled .so/.pyd is named "pyscx.pyscx" internally by maturin.
from .pyscx import *  # noqa: F401, F403, E402
from .pyscx import ScxBackedSparseDataset, ScxBackedLayerDataset  # noqa: E402

# Register the Rust-side `accel`
# submodule under `sys.modules` so the dotted import idiom works
# symmetrically with the already-working `from pyscx import accel`. PyO3's
# `m.add_submodule(&accel_module)?` (see pyscx/src/lib.rs:774-787) exposes
# `accel` as an *attribute* on the parent C-extension module but does not
# populate `sys.modules`; Python's import machinery needs the entry there
# to resolve `import pyscx.accel as a`. `accel` is the only Rust-side
# submodule pyscx exposes today; other public surface comes from the
# `from .pyscx import *` line above. `setdefault` is the safe variant —
# if anything else has already registered the submodule (rewriting
# loaders, test harnesses, future Python), we don't clobber it.
import sys as _sys  # noqa: E402
from .pyscx import accel as _accel_submodule  # noqa: E402

_sys.modules.setdefault("pyscx.accel", _accel_submodule)
del _sys, _accel_submodule

# Thin Python wrappers around the native entry points so they accept
# `os.PathLike` (e.g. `pathlib.Path`) and — for the SCX-side
# converters (`to_h5ad` / `to_h5mu`) — a `PyExperiment` handle. The
# Rust bindings still want plain `str`; the wrappers coerce on the
# way in. `from_h5ad` / `from_h5mu` deliberately reject Experiment
# handles because their source must be an h5ad/h5mu file, not an
# already-open SCX file.
import os as _os                                   # noqa: E402

from .pyscx import open as _open_native            # noqa: E402
from .pyscx import validate as _validate_native    # noqa: E402

# N3-2026-05-21-Tier2: hdf5-gated entry points. The Rust side registers
# these four symbols under `#[cfg(feature = "hdf5")]` (pyscx/src/lib.rs).
# `[tool.maturin] features` defaults `hdf5` on, so the common case is
# `_HAS_HDF5 = True`. The try/except guard keeps `import pyscx` working
# in `--no-default-features` builds; calls into the wrappers below then
# raise a clean `NotImplementedError` via `_require_hdf5` instead of an
# `ImportError` on the very first `import pyscx`.
try:
    from .pyscx import from_h5ad as _from_h5ad_native  # noqa: E402
    from .pyscx import to_h5ad as _to_h5ad_native      # noqa: E402
    from .pyscx import from_h5mu as _from_h5mu_native  # noqa: E402
    from .pyscx import to_h5mu as _to_h5mu_native      # noqa: E402
    from .pyscx import read_h5ad_metadata as _read_h5ad_metadata_native  # noqa: E402
    _HAS_HDF5 = True
except ImportError:
    _from_h5ad_native = None
    _to_h5ad_native = None
    _from_h5mu_native = None
    _to_h5mu_native = None
    _read_h5ad_metadata_native = None
    _HAS_HDF5 = False


def _require_hdf5(fn_name):
    if not _HAS_HDF5:
        raise NotImplementedError(
            f"pyscx.{fn_name} requires the `hdf5` feature; rebuild via "
            "`cd pyscx && maturin develop --features hdf5` or install "
            "the prebuilt wheel from PyPI (bundles libhdf5)."
        )


def _coerce_path(p, *, allow_experiment: bool = True):
    """Coerce a path-like input to a plain str for the Rust bindings.

    Accepts:
      - `str` (returned as-is)
      - `os.PathLike` (e.g. `pathlib.Path`) — uses `__fspath__` via
        `os.fspath`, so subclasses whose `__str__` isn't overridden
        still resolve correctly.
      - a `pyscx.Experiment` handle, via its `.path` getter — only
        when `allow_experiment=True`. The `from_h5ad` / `from_h5mu`
        wrappers pass `allow_experiment=False` because their source
        must be an h5ad/h5mu file, not an open SCX Experiment.
    """
    if isinstance(p, str):
        return p
    # Real PathLike (pathlib.Path, etc.) takes priority over the
    # Experiment duck-type check so a path-like whose subclass happens
    # to expose `.path` for unrelated reasons still resolves via
    # __fspath__.
    if hasattr(p, "__fspath__"):
        fspath = _os.fspath(p)
        return fspath.decode() if isinstance(fspath, bytes) else fspath
    # PyExperiment exposes a `.path` getter returning str.
    path_attr = getattr(p, "path", None)
    if isinstance(path_attr, str):
        if not allow_experiment:
            raise TypeError(
                f"{type(p).__name__} (open SCX Experiment) is not a valid "
                "source here — the source must be an h5ad/h5mu file path, "
                "not an already-converted SCX file."
            )
        return path_attr
    # Last resort: re-raise os.fspath's TypeError with a scx-specific
    # message naming the expected types, so users hitting the boundary
    # see "pyscx expects ..." instead of the generic "expected str, bytes
    # or os.PathLike object, not list".
    try:
        fspath = _os.fspath(p)
    except TypeError:
        expected = "str | os.PathLike | pyscx.Experiment" if allow_experiment else "str | os.PathLike"
        raise TypeError(
            f"pyscx expects {expected}; got {type(p).__name__}"
        ) from None
    return fspath.decode() if isinstance(fspath, bytes) else fspath


def open(path, verify=True):  # noqa: A001 — intentional shadowing of builtins.open within the pyscx namespace
    """Open an SCX file as an `Experiment`. Accepts str or
    `os.PathLike` (e.g. `pathlib.Path`)."""
    return _open_native(_coerce_path(path), verify=verify)


def validate(path):
    """Validate an SCX file by walking its catalog and checking BLAKE3
    checksums. Accepts str or `os.PathLike`."""
    return _validate_native(_coerce_path(path))


def from_h5ad(path, out, **kwargs):
    """Convert an h5ad file to SCX, streaming by default.

    Accepts str or `os.PathLike` for `path` and `out`. (Source must be an
    h5ad file, not an open SCX Experiment.) Peak RSS is bounded by one
    shard's worth of CSR regardless of file size; obs/var/uns are read via
    pure-Rust HDF5 (no `anndata.read_h5ad`, no eager `obsm` allocation).

    Args:
        path: Source h5ad file (str or os.PathLike).
        out: Destination SCX file (str or os.PathLike).
        codec: Per-shard codec. None (default) auto-selects; also "scx1"
            (integer-only), "zstd", "pcodec" (best for float layers),
            "lz4", "none".
        shard_size: Rows per CSR X shard. None uses the default.
        csc: "off" (default) or "always". "always" adds a column-major
            sidecar via a two-pass CSR-then-rebuild write (transient disk
            ~2x the output); required for prefer_format="csc" accel paths.
        csc_cols_per_shard: Columns per CSC shard when csc="always"
            (default 5000); 0 = single CSC shard.
        uns_format: "tagged" (default) wraps NumPy/pandas containers in
            __scx_type__ envelopes for bit-exact round-trip; "plain"
            collapses to JSON primitives. Controls the envelope applied to
            uns_override; a no-op for the on-disk uns read.
        stream: Stream the conversion (default True). False falls back to
            the legacy materializing path (does not apply overrides).
        strict_uns: True raises on the first unrepresentable uns entry;
            False (default) emits a UserWarning per skipped key.
        dense_zero_epsilon: Threshold for dropping near-zero values when
            sparsifying a dense /X (default 0.0, matching
            scipy.csr_matrix(dense)).
        memory_budget: Caps dense row slabs and CSC external-transpose
            buffers, and derates reader_threads. Accepts an int byte count
            or a binary-prefixed size string — K/M/G/T or KiB/MiB/GiB/TiB
            (powers of 1024); decimal KB/MB/GB/TB is rejected. E.g. "4G" /
            "512M" / "2GiB". None = per-phase default.
        temp_dir: Scratch directory for the CSC external transpose (used
            only when memory_budget forces the external path). Defaults to
            the system temp dir.
        index_obs: List of obs column names to materialize predicate
            indexes for, so pyscx.open(...).query().filter_obs(...) pushes
            filters down. Forced missing columns hard-error.
        index_var: List of var column names to index.
        index_preset: Expand a curated column list: "cellxgene",
            "perturbseq", or "training". Preset misses emit
            MissingPresetIndexColumn.
        index_auto_threshold: Max cardinality for automatic categorical
            indexing (default 1000).
        bitmap: "off" (default), "auto", or "always". Writes per-shard
            gene->row detection bitmap sidecars consumed by
            detection_counts / cells_expressing.
        reader_threads: Parallel streaming-reader worker count. None
            (default) resolves to RAYON_NUM_THREADS or os.cpu_count(); 1
            forces the sequential coordinator; >1 requests rayon workers
            (output byte-identical). Requires a thread-safe libhdf5
            (conda-forge default); a non-threadsafe build falls back to
            sequential with a one-shot Hdf5NotThreadsafe warning.
        writer_queue_depth: Backpressure window between the encoder pool
            and the ordered writer (default 4); outstanding shards are
            capped at reader_threads + writer_queue_depth.
        obs_override: Optional pandas DataFrame used in place of the
            on-disk obs (shape[0] must equal on-disk n_obs), for
            read-mutate-write flows via pyscx.read_h5ad_metadata. Requires
            stream=True.
        var_override: Optional pandas DataFrame used in place of the
            on-disk var (shape[0] must equal on-disk n_vars). Requires
            stream=True.
        uns_override: Optional dict replacing the entire uns section (not
            merged). Requires stream=True.
    """
    _require_hdf5("from_h5ad")
    return _from_h5ad_native(
        _coerce_path(path, allow_experiment=False),
        _coerce_path(out),
        **kwargs,
    )


def read_h5ad_metadata(path, strict_uns=False):
    """Read obs / var / uns / X shape from an h5ad file via pure-Rust
    HDF5 readers, without going through `anndata.read_h5ad` (which
    eagerly materialises `obsm` on every call). Accepts str or
    `os.PathLike` for `path`. Returns an `H5adMetadata` object whose
    `obs`, `var`, `uns`, `n_obs`, `n_vars`, and `x_format` attributes
    can be inspected / mutated and passed back to
    `pyscx.from_h5ad(path, out, obs_override=..., uns_override=...)`
    for read-mutate-write flows that need to stay under tight memory
    budgets."""
    _require_hdf5("read_h5ad_metadata")
    return _read_h5ad_metadata_native(
        _coerce_path(path, allow_experiment=False),
        strict_uns=strict_uns,
    )


def to_h5ad(path, out, **kwargs):
    """Convert an SCX file to h5ad, streaming by default.

    Accepts str, `os.PathLike`, or a pyscx Experiment for `path`; str or
    `os.PathLike` for `out`. Mirror of `pyscx.from_h5ad`. Peak RSS is
    bounded by one shard's worth of CSR per matrix written. When deletion
    vectors are present, only kept rows are written.

    Args:
        path: Source SCX file (str, os.PathLike, or pyscx Experiment).
        out: Destination h5ad file (str or os.PathLike).
        stream: Stream the conversion (default True). False falls back to
            the legacy materializing path.
        modality: For a multimodal SCX file, the modality to extract as
            h5ad (e.g. "rna"); single-modality files ignore it and
            multimodal files raise without it (use pyscx.to_h5mu).
        reader_threads: Parallel shard-decoder worker count. None (default)
            resolves to RAYON_NUM_THREADS or os.cpu_count(); 1 forces
            sequential; >1 requests rayon workers (output byte-identical).
            HDF5 writes stay on the calling thread, so this does NOT require
            a thread-safe libhdf5 (unlike the ingest direction).
        writer_queue_depth: Bounded reorder-buffer depth between the
            decoder pool and the ordered HDF5 writer (default 4).
        memory_budget: Derates reader_threads against the exact per-shard
            byte size (from catalog nnz). Accepts an int byte count or a
            binary-prefixed size string — K/M/G/T or KiB/MiB/GiB/TiB
            (powers of 1024); decimal KB/MB/GB/TB is rejected. E.g. "4G" /
            "512M" / "2GiB". A single shard exceeding the budget raises;
            smaller mismatches emit ReaderThreadsDerated.
    """
    _require_hdf5("to_h5ad")
    return _to_h5ad_native(_coerce_path(path), _coerce_path(out), **kwargs)


def from_h5mu(path, out, **kwargs):
    """Convert an h5mu file to a multimodal SCX v2 file, streaming by default.

    Accepts str or `os.PathLike` for `path` and `out`. (Source must be an
    h5mu file, not an open SCX Experiment.) Mirrors `pyscx.from_h5ad`; each
    modality runs through the same dispatcher independently.

    Args:
        path: Source h5mu file (str or os.PathLike).
        out: Destination SCX file (str or os.PathLike).
        codec: Per-shard codec (see pyscx.from_h5ad). None auto-selects.
        shard_size: Rows per CSR X shard. None uses the default.
        csc: "off" (default) or "always" (column-major sidecar).
        csc_cols_per_shard: Columns per CSC shard when csc="always"
            (default 5000).
        stream: Stream the conversion (default True).
        strict_uns: True raises on the first unrepresentable uns entry;
            False (default) warns per skipped key.
        memory_budget: Caps dense slabs and CSC external-transpose buffers,
            and derates reader_threads. Int byte count or a binary-prefixed
            size string — K/M/G/T or KiB/MiB/GiB/TiB (powers of 1024);
            decimal KB/MB/GB/TB is rejected. E.g. "4G".
        temp_dir: Scratch directory for the CSC external transpose.
        modalities: Optional list of modality names to keep
            (case-sensitive). Unknown names raise with the available list.
        modality_types: Optional dict mapping modality name to one of
            "rna", "protein", "atac", "spatial", "methylation", "custom".
            Modalities not listed fall back to name inference and emit
            ModalityTypeInferred.
        index_obs: obs column names to index for query pushdown.
        index_var: var column names to index.
        index_preset: "cellxgene", "perturbseq", or "training".
        index_auto_threshold: Max cardinality for auto categorical
            indexing (default 1000).
        bitmap: "off" (default), "auto", or "always".
        reader_threads: Parallel reader worker count (see pyscx.from_h5ad);
            >1 requires a thread-safe libhdf5.
        writer_queue_depth: Encoder->writer backpressure window (default 4).
    """
    _require_hdf5("from_h5mu")
    return _from_h5mu_native(
        _coerce_path(path, allow_experiment=False),
        _coerce_path(out),
        **kwargs,
    )


def to_h5mu(path, out, **kwargs):
    """Convert a multimodal SCX file to h5mu, streaming by default.

    Accepts str, `os.PathLike`, or a pyscx Experiment for `path`; str or
    `os.PathLike` for `out`. Requires a multimodal SCX file
    (single-modality files raise — use `pyscx.to_h5ad`). Each modality's
    /mod/{name}/X and any layers are written shard-by-shard.

    Args:
        path: Source multimodal SCX file (str, os.PathLike, or Experiment).
        out: Destination h5mu file (str or os.PathLike).
        stream: Stream the conversion (default True).
        reader_threads: Parallel shard-decoder worker count (see
            pyscx.to_h5ad). HDF5 writes stay on the calling thread, so this
            does not require a thread-safe libhdf5.
        writer_queue_depth: Decoder->writer reorder-buffer depth (default 4).
        memory_budget: Derates reader_threads against the exact per-shard
            byte size. Int byte count or a binary-prefixed size string —
            K/M/G/T or KiB/MiB/GiB/TiB (powers of 1024); decimal
            KB/MB/GB/TB is rejected. E.g. "4G".
    """
    _require_hdf5("to_h5mu")
    return _to_h5mu_native(_coerce_path(path), _coerce_path(out), **kwargs)


def iter_chunks(adata, chunk_size="shard"):
    """Iterate over an AnnData in chunks, yielding fully materialized AnnData slices.

    When the AnnData has a backed SCX X matrix, ``chunk_size="shard"`` aligns
    chunks to the on-disk shard boundaries for optimal I/O.  Each yielded
    AnnData is a fully materialized copy with the correct obs/var/obsm
    metadata sliced to match.

    Args:
        adata: An ``anndata.AnnData`` object (backed or in-memory).
        chunk_size: ``"shard"`` (default) to align to SCX shard boundaries,
            or an ``int`` for fixed-size chunks of that many rows.

    Yields:
        ``anndata.AnnData`` — a fully materialized AnnData with
        ``~shard_size`` (or ``chunk_size``) cells.

    Example::

        adata = pyscx.open("atlas.scx").to_anndata(backed=True)
        for chunk in pyscx.iter_chunks(adata, chunk_size="shard"):
            sc.pp.normalize_total(chunk)
            results.append(chunk.X)
    """
    n_obs = adata.n_obs

    if chunk_size == "shard":
        # Try to get shard boundaries from the backed X matrix
        x = adata.X
        if isinstance(x, (ScxBackedSparseDataset, ScxBackedLayerDataset)):
            boundaries = x.shard_boundaries()
        else:
            # Fallback: non-backed AnnData — use default 16384-row chunks
            boundaries = [
                (i, min(i + 16384, n_obs)) for i in range(0, n_obs, 16384)
            ]
    elif isinstance(chunk_size, int) and chunk_size > 0:
        boundaries = [
            (i, min(i + chunk_size, n_obs)) for i in range(0, n_obs, chunk_size)
        ]
    else:
        raise ValueError(
            f"chunk_size must be 'shard' or a positive int, got {chunk_size!r}"
        )

    for start, end in boundaries:
        yield adata[start:end].copy()
