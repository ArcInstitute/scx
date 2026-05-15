"""pyscx — Python bindings for SCX (Sparse Cell eXpression System).

This module re-exports everything from the native Rust extension module
and provides pure-Python integration packages (e.g., scx_integrations).
"""

import math
import warnings

# Re-export everything from the native Rust extension module.
# The compiled .so/.pyd is named "pyscx.pyscx" internally by maturin.
from .pyscx import *  # noqa: F401, F403
from .pyscx import ScxBackedSparseDataset, ScxBackedLayerDataset

# Capture references to native impls before we override `from_anndata` below.
from .pyscx import from_anndata as _native_from_anndata
from .pyscx import append_from_anndata as _native_append_from_anndata

_DEFAULT_CHUNK_ROWS = 16384
# Slots that `append_from_anndata` does not extend. Carrying them on chunks
# would produce a file whose obsm/layers length disagrees with n_obs.
_ALIGNED_SLOTS_DROPPED_IN_CHUNKED = ("obsm", "varm", "obsp", "varp", "layers")


def _strip_aligned_slots(chunk, *, keep_uns):
    """Clear obsm/varm/obsp/varp/layers (and uns when ``keep_uns`` is False).

    Returns the same AnnData with those mappings cleared in place. Caller
    must own ``chunk`` (i.e. it is in-memory and not a view).
    """
    for slot in _ALIGNED_SLOTS_DROPPED_IN_CHUNKED:
        mapping = getattr(chunk, slot, None)
        if mapping is None:
            continue
        for k in list(mapping.keys()):
            del mapping[k]
    if not keep_uns:
        chunk.uns = {}
    return chunk


def from_anndata(
    adata,
    path,
    codec=None,
    shard_size=None,
    in_place=False,
    *,
    chunked=False,
    n_chunks=None,
):
    """Write an AnnData to an SCX file.

    By default the AnnData is written in a single pass. Pass ``chunked=True``
    to stream the conversion in row chunks — useful when ``adata`` is opened
    in ``backed='r'`` mode and ``X`` would not otherwise fit in memory. In
    chunked mode the first chunk is written with the native ``from_anndata``
    and subsequent chunks are appended with ``append_from_anndata``.

    Args:
        adata: ``anndata.AnnData`` (in-memory or backed).
        path: Output SCX path.
        codec: ``"auto"`` (default), ``"none"``, ``"scx1"``, ``"zstd"``,
            ``"lz4"``, or ``"pcodec"``.
        shard_size: Rows per shard. Defaults to the underlying writer's
            default (16384 in the pyscx path).
        in_place: Allow in-place CSR index sorting on caller's matrices.
        chunked: Stream the conversion in row chunks. Default ``False``.
        n_chunks: Number of row chunks to split ``adata`` into. Only used
            when ``chunked=True``. Defaults to ``ceil(n_obs / 16384)`` so
            each chunk is roughly one shard.

    Note:
        ``chunked=True`` only writes ``X``, ``obs``, ``var``, and (from the
        first chunk) ``uns`` — the underlying ``append_from_anndata``
        primitive does not extend ``obsm``/``varm``/``obsp``/``varp``/
        ``layers`` across appends, so those slots are dropped to keep the
        resulting file internally consistent. A ``UserWarning`` is emitted
        if any of those slots were non-empty on the input. Categorical
        dtypes and the obs index name on appended chunks may be normalised
        by ``append_from_anndata`` (the appended record batches encode the
        pandas index as a regular column), so a round-trip through chunked
        mode is not guaranteed to be byte-identical to a one-shot write
        for ``obs`` metadata — ``X`` is identical.

    Example::

        # One-shot (default)
        pyscx.from_anndata(adata, "out.scx")

        # Streamed from a backed h5ad — X is never fully materialized
        adata = sc.read_h5ad("huge.h5ad", backed="r")
        pyscx.from_anndata(adata, "out.scx", chunked=True, n_chunks=20)
    """
    if not chunked:
        return _native_from_anndata(
            adata, path, codec=codec, shard_size=shard_size, in_place=in_place
        )

    n_obs = int(adata.n_obs)
    if n_obs == 0:
        return _native_from_anndata(
            adata, path, codec=codec, shard_size=shard_size, in_place=in_place
        )

    if n_chunks is None:
        n_chunks = max(1, math.ceil(n_obs / _DEFAULT_CHUNK_ROWS))
    if not isinstance(n_chunks, int) or n_chunks <= 0:
        raise ValueError(
            f"n_chunks must be a positive int, got {n_chunks!r}"
        )
    n_chunks = min(n_chunks, n_obs)

    dropped = [
        slot
        for slot in _ALIGNED_SLOTS_DROPPED_IN_CHUNKED
        if len(getattr(adata, slot, {}) or {}) > 0
    ]
    if dropped:
        warnings.warn(
            "pyscx.from_anndata(chunked=True) drops "
            + ", ".join(f"adata.{s}" for s in dropped)
            + " because append_from_anndata does not extend these slots "
            "across chunks. Use chunked=False to preserve them.",
            UserWarning,
            stacklevel=2,
        )

    chunk_size = math.ceil(n_obs / n_chunks)
    is_backed = bool(getattr(adata, "isbacked", False))

    for i, start in enumerate(range(0, n_obs, chunk_size)):
        end = min(start + chunk_size, n_obs)
        sliced = adata[start:end]
        chunk = sliced.to_memory() if is_backed else sliced.copy()
        _strip_aligned_slots(chunk, keep_uns=(i == 0))

        if i == 0:
            _native_from_anndata(
                chunk, path, codec=codec, shard_size=shard_size, in_place=in_place
            )
        else:
            _native_append_from_anndata(
                path, chunk, codec=codec, shard_size=shard_size, in_place=in_place
            )


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
