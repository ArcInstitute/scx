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
    _HAS_HDF5 = True
except ImportError:
    _from_h5ad_native = None
    _to_h5ad_native = None
    _from_h5mu_native = None
    _to_h5mu_native = None
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
    """Convert an h5ad file to SCX. Accepts str or `os.PathLike` for
    `path` and `out`. (Source must be an h5ad file, not an open SCX
    Experiment.)"""
    _require_hdf5("from_h5ad")
    return _from_h5ad_native(
        _coerce_path(path, allow_experiment=False),
        _coerce_path(out),
        **kwargs,
    )


def to_h5ad(path, out, **kwargs):
    """Convert an SCX file to h5ad. Accepts str, `os.PathLike`, or a
    pyscx Experiment for `path`; str or `os.PathLike` for `out`."""
    _require_hdf5("to_h5ad")
    return _to_h5ad_native(_coerce_path(path), _coerce_path(out), **kwargs)


def from_h5mu(path, out, **kwargs):
    """Convert an h5mu file to SCX. Accepts str or `os.PathLike` for
    `path` and `out`. (Source must be an h5mu file, not an open SCX
    Experiment.)"""
    _require_hdf5("from_h5mu")
    return _from_h5mu_native(
        _coerce_path(path, allow_experiment=False),
        _coerce_path(out),
        **kwargs,
    )


def to_h5mu(path, out, **kwargs):
    """Convert an SCX file to h5mu. Accepts str, `os.PathLike`, or a
    pyscx Experiment for `path`; str or `os.PathLike` for `out`."""
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
