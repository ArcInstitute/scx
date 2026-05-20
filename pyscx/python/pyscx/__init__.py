"""pyscx — Python bindings for SCX (Sparse Cell eXpression System).

This module re-exports everything from the native Rust extension module
and provides pure-Python integration packages (e.g., scx_integrations).
"""

# Re-export everything from the native Rust extension module.
# The compiled .so/.pyd is named "pyscx.pyscx" internally by maturin.
from .pyscx import *  # noqa: F401, F403
from .pyscx import ScxBackedSparseDataset, ScxBackedLayerDataset

# Thin Python wrappers around the native entry points so they accept
# `os.PathLike` (e.g. `pathlib.Path`) and — for the SCX-side
# converters — a `PyExperiment` handle. The Rust bindings still want
# plain `str`; the wrappers coerce on the way in. Source can be either
# an SCX path or a `pyscx.open(...).path` handle on `to_h5ad` /
# `to_h5mu`.
from .pyscx import open as _open_native            # noqa: E402
from .pyscx import validate as _validate_native    # noqa: E402
from .pyscx import from_h5ad as _from_h5ad_native  # noqa: E402
from .pyscx import to_h5ad as _to_h5ad_native      # noqa: E402
from .pyscx import from_h5mu as _from_h5mu_native  # noqa: E402
from .pyscx import to_h5mu as _to_h5mu_native      # noqa: E402


def _coerce_path(p):
    """Accept str, os.PathLike, or a pyscx Experiment handle (uses
    its `.path` attribute). Returns a plain str for the Rust bindings.

    Strings are returned as-is. `pathlib.Path` and other PathLike
    objects are stringified via `str()`, which invokes `__fspath__`.
    A `pyscx.Experiment` exposes its on-disk path via a `.path`
    getter (added on the Rust side); the wrapper unwraps it here so
    `pyscx.to_h5ad(exp, "/tmp/out.h5ad")` works idiomatically.
    """
    if isinstance(p, (str, bytes)):
        return p if isinstance(p, str) else p.decode()
    # PyExperiment exposes a `.path` getter returning str.
    path_attr = getattr(p, "path", None)
    if isinstance(path_attr, str):
        return path_attr
    # os.PathLike or anything else — let str() handle it.
    return str(p)


def open(path, verify=True):  # noqa: A001 — intentional shadowing of builtins.open within the pyscx namespace
    """Open an SCX file as an `Experiment`. Accepts str or
    `os.PathLike` (e.g. `pathlib.Path`)."""
    return _open_native(_coerce_path(path), verify=verify)


def validate(path):
    """Validate an SCX file by walking its catalog and checking BLAKE3
    checksums. Accepts str or `os.PathLike`."""
    return _validate_native(_coerce_path(path))


def from_h5ad(path, out, **kwargs):
    """Convert an h5ad file to SCX. Accepts str, `os.PathLike`, or a
    pyscx Experiment for `path`; str or `os.PathLike` for `out`."""
    return _from_h5ad_native(_coerce_path(path), _coerce_path(out), **kwargs)


def to_h5ad(path, out, **kwargs):
    """Convert an SCX file to h5ad. Accepts str, `os.PathLike`, or a
    pyscx Experiment for `path`; str or `os.PathLike` for `out`."""
    return _to_h5ad_native(_coerce_path(path), _coerce_path(out), **kwargs)


def from_h5mu(path, out, **kwargs):
    """Convert an h5mu file to SCX. Accepts str, `os.PathLike`, or a
    pyscx Experiment for `path`; str or `os.PathLike` for `out`."""
    return _from_h5mu_native(_coerce_path(path), _coerce_path(out), **kwargs)


def to_h5mu(path, out, **kwargs):
    """Convert an SCX file to h5mu. Accepts str, `os.PathLike`, or a
    pyscx Experiment for `path`; str or `os.PathLike` for `out`."""
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
