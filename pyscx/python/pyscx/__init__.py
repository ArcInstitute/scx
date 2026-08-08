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


def _pflog_reconstruct(adata, baseline_key="pflog_baseline"):
    """Reconstruct the exact dense PFlog ``Z`` from a compact
    ``store_repr="delta_baseline"`` file written by ``accel.pflog(..., out=)``.

    Such a file stores the sparse ``delta`` as ``X`` and the per-cell
    ``baseline`` as an obs column, so ``Z = delta + baseline[:, None]``. This is
    the same cheap kernel the training loader applies on read; the compact file
    can equivalently be streamed through ``TrainingDataset`` with its transform
    mode off (precompute-once / train-many-epochs).

    .. note::
       This materializes the **entire** dense ``Z`` (``O(n_obs * n_vars)``
       memory). For atlas-scale data prefer the streaming training loader or the
       out-of-core ``store="pca"`` embedding instead of reconstructing in full.

    Parameters
    ----------
    adata
        An ``AnnData`` opened from the compact file (e.g.
        ``pyscx.open(path).to_anndata()``). ``adata.X`` is the ``delta`` matrix
        and ``adata.obs[baseline_key]`` the per-cell baseline.
    baseline_key
        Name of the obs column holding the baseline (default
        ``"pflog_baseline"``).

    Returns
    -------
    numpy.ndarray
        Dense ``float32`` array of shape ``(n_obs, n_vars)`` equal to the exact
        PFlog transform.
    """
    import numpy as _np

    if baseline_key not in adata.obs:
        raise KeyError(
            f"pflog_reconstruct: obs column {baseline_key!r} not found; this "
            "is not a compact delta_baseline PFlog file (or baseline_key is wrong)"
        )
    x = adata.X
    delta = _np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=_np.float32)
    baseline = _np.asarray(adata.obs[baseline_key], dtype=_np.float32)
    return delta + baseline[:, None]


# Expose as `pyscx.accel.pflog_reconstruct` (pure-Python companion to the
# native `accel.pflog` writer) and at top level for discoverability.
_accel_submodule.pflog_reconstruct = _pflog_reconstruct
pflog_reconstruct = _pflog_reconstruct

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
from .pyscx import from_anndata as _from_anndata_native  # noqa: E402
from .pyscx import obs_import as _obs_import_native  # noqa: E402
from .pyscx import diagnose_obs_key as _diagnose_obs_key_native  # noqa: E402
from .pyscx import doublet_import as _doublet_import_native  # noqa: E402
from .pyscx import modify_metadata as _modify_metadata_native  # noqa: E402
from .pyscx import set_uns as _set_uns_native      # noqa: E402

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


def _warn_if_deletions(src_path, out_fmt):
    """Emit a UserWarning when exporting a source that carries logical
    deletion vectors — those rows are dropped on export, so the written
    file has fewer cells than the SCX source. Cheap: header-only open.
    Never fails the export (best-effort)."""
    import warnings as _warnings
    try:
        exp = _open_native(src_path, verify=False)
        if getattr(exp, "has_deletions", False):
            _warnings.warn(
                f"source SCX file has logically-deleted rows; the exported "
                f"{out_fmt} will contain only the kept cells (row count is "
                f"smaller than the SCX source).",
                UserWarning,
                stacklevel=3,
            )
    except (OSError, RuntimeError, ValueError):
        # A header probe failure should never block the actual export;
        # the converter below will surface any real error. Narrow to the
        # expected I/O / open failures so a genuine binding bug
        # (TypeError / AttributeError) still surfaces during development.
        pass


def _coerce_obs_mask(mask):
    """Normalise a user obs_mask to a C-contiguous 1-D numpy bool array.

    Accepts numpy bool arrays, pandas boolean Series, and lists of bool.
    Non-bool dtypes are rejected rather than coerced: an int or float array
    silently becoming `!= 0` is exactly the kind of quiet wrong answer a row
    filter must never produce. Strided views are copied because the Rust side
    reads the buffer as a contiguous slice.
    """
    import numpy as _np

    # A pandas *nullable* boolean Series (dtype="boolean") becomes an object
    # array under np.asarray, so it needs its own branch. pd.NA is rejected
    # rather than defaulted: a missing entry has no defensible reading as
    # "keep" or "drop", and silently picking one would filter the wrong rows.
    if str(getattr(mask, "dtype", "")) == "boolean":
        if mask.isna().any():
            raise ValueError(
                "obs_mask contains pd.NA; a keep mask must be unambiguously "
                "True or False for every observation"
            )
        return _np.ascontiguousarray(mask.to_numpy(dtype=bool))

    arr = _np.asarray(mask)  # deliberately no dtype= — do NOT coerce
    if arr.dtype.kind != "b":
        raise TypeError(
            f"obs_mask must be a boolean array; got dtype {arr.dtype!r}. "
            "Pass a predicate result (e.g. `counts >= 500`), not the counts."
        )
    if arr.ndim != 1:
        raise ValueError(f"obs_mask must be 1-D; got shape {arr.shape}")
    return _np.ascontiguousarray(arr)


def _require_hdf5(fn_name):
    if not _HAS_HDF5:
        raise NotImplementedError(
            f"pyscx.{fn_name} requires the `hdf5` feature; rebuild via "
            "`cd pyscx && maturin develop --features hdf5` or install "
            "the prebuilt wheel from GitHub Releases (bundles libhdf5)."
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


def validate(path, deep=False):
    """Validate an SCX file by walking its catalog and checking BLAKE3
    checksums. Accepts str or `os.PathLike`.

    When ``deep=True``, additionally decodes every sparse shard to verify the
    v3 canonical CSR invariant (sorted column indices, no explicit zeros,
    consistent indptr) and verifies every decode sidecar (structural linkage
    + decode-parity). Mirrors ``scx validate --deep``. Deep-check results are
    appended with ``canonical-csr ``/``decode-sidecar `` prefixed names; they
    report ``False`` rather than raising."""
    return _validate_native(_coerce_path(path), deep)


def read(path, *, verify=True, **kwargs):
    """Read an SCX file into an `anndata.AnnData` in one call.

    The flat counterpart to `scanpy.read_h5ad` — shorthand for
    `pyscx.open(path).to_anndata(**kwargs)`. Accepts str or `os.PathLike`.

    Args:
        path: SCX file to read (str or os.PathLike).
        verify: Forwarded to `pyscx.open` — verify the catalog checksum
            on open (default True). Set False for trusted files.
        **kwargs: Forwarded to `Experiment.to_anndata` (e.g. `backed=True`,
            `var_names=[...]`, `obs_filter="..."`, `layers=[...]`). See
            `Experiment.to_anndata` for the full set.

    Returns:
        `anndata.AnnData`.

    Example:
        import pyscx
        adata = pyscx.read("data.scx")
        backed = pyscx.read("atlas.scx", backed=True)
    """
    return open(path, verify=verify).to_anndata(**kwargs)


def write(adata, path, **kwargs):
    """Write an `anndata.AnnData` to an SCX file in one call.

    The flat counterpart to `AnnData.write_h5ad` — shorthand for
    `pyscx.from_anndata(adata, path, **kwargs)`. Accepts str or
    `os.PathLike` for `path`.

    Args:
        adata: The `anndata.AnnData` (or backed/lazy SCX `X`) to write.
        path: Destination SCX file (str or os.PathLike).
        **kwargs: Forwarded to `pyscx.from_anndata` (e.g. `codec=...`,
            `shard_size=...`, `csc=...`, `index_preset=...`).

    Example:
        import pyscx
        pyscx.write(adata, "data.scx")
    """
    return _from_anndata_native(adata, _coerce_path(path), **kwargs)


def modify_metadata(path, **kwargs):
    """Replace metadata sections in place, without re-encoding `X`.

    Thin `os.PathLike`-accepting wrapper; see the native docstring
    (`help(pyscx.pyscx.modify_metadata)`) for the full contract. Every other
    path-taking entry point coerces, and this one did not — a `pathlib.Path`
    raised `TypeError: 'PosixPath' object is not an instance of 'str'`.

    Args:
        path: Target SCX file (str, os.PathLike, or an open Experiment).
        **kwargs: Forwarded to the native `modify_metadata` (`uns=`, `obs=`,
            `var=`, `obsm=`, `varm=`, `index_obs=`, …).
    """
    return _modify_metadata_native(_coerce_path(path), **kwargs)


def set_uns(path, uns):
    """Replace the whole `uns` block in place. `os.PathLike`-accepting wrapper.

    Args:
        path: Target SCX file (str, os.PathLike, or an open Experiment).
        uns: dict replacing the whole `uns` block (replace, not merge).
    """
    return _set_uns_native(_coerce_path(path), uns)


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
        csc: "off", "auto", or "always". "always" adds a column-major
            sidecar via a two-pass CSR-then-rebuild write (transient disk
            ~2x the output); required for prefer_format="csc" accel paths.
            When omitted, an accel-ready index_preset ("training" /
            "perturbseq") upgrades the default to "auto"; otherwise "off".
            An explicit value always wins.
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
        obs_mask: Boolean array selecting the observations to keep. Indexed
            in the GLOBAL / physical obs row space — its length must equal
            `pyscx.open(path).n_obs_physical` (the file header count), NOT
            `.n_obs` (the live, post-deletion count). Rows already logically
            deleted stay dropped regardless of their entry here: the mask is
            ANDed with the deletion-vector mask, never substituted for it.
            Accepts a numpy bool array, a pandas boolean Series, or a list of
            bool; a non-bool dtype raises rather than being coerced. Requires
            stream=True.
        min_counts: Per-cell total-UMI floor. Keeps rows where
            `X[i, :].sum() >= min_counts`, computed with one streaming pass
            over the CSR shards (no materialization) in the same global row
            space as obs_mask, and ANDed with it. Sums `X`, so it is
            meaningless on an already-normalized matrix. Requires stream=True.

    Example:
        # Result-preserving CellBender pre-trim on a raw all-droplet file:
        # CellBender's own prior estimation ignores droplets at or below its
        # --low-count-threshold, and never-analyzed barcodes are all-zero
        # rows in its output.
        pyscx.to_h5ad("raw.scx", "raw_trimmed.h5ad", min_counts=5)
    """
    _require_hdf5("to_h5ad")
    if kwargs.get("obs_mask") is not None:
        kwargs["obs_mask"] = _coerce_obs_mask(kwargs["obs_mask"])
    src = _coerce_path(path)
    _warn_if_deletions(src, "h5ad")
    return _to_h5ad_native(src, _coerce_path(out), **kwargs)


def _safe_batch_filename(batch) -> str:
    """Make an obs batch label safe to use as a filename.

    Batch labels are data, not identifiers: a `donor_id` of `../x` or `a/b`
    would otherwise place the export outside `out_dir`. Anything that is not
    alphanumeric, dash, dot or underscore becomes an underscore, and a label
    that reduces to nothing (or to a bare dot run) gets a positional fallback.
    """
    import re as _re

    text = _re.sub(r"[^A-Za-z0-9._-]", "_", str(batch))
    return text if text.strip("._") else "batch"


def _plan_batch_filenames(batches) -> dict:
    """Map each batch label to a unique filename stem.

    Sanitising alone is not enough: `batch/1`, `batch\\1` and `batch_1` all
    reduce to `batch_1`. Left unchecked that is a `FileExistsError` about a
    file this very call just wrote (with `overwrite=False`) or, worse, one
    batch silently overwriting another's export (`overwrite=True`) — the
    second batch's cells then never reach a tool and the first's results are
    gone. Disambiguate with an index suffix, which keeps the common case
    (labels that need no sanitising) unchanged.
    """
    stems, seen = {}, {}
    for i, b in enumerate(batches):
        stem = _safe_batch_filename(b)
        if stem in seen:
            stem = f"{stem}__{i}"
        seen[stem] = True
        stems[b] = stem
    return stems


def export_batches(path, out_dir, *, batch_key, key=None, batches=None,
                   on_ambiguous_key="error", overwrite=False, **kwargs):
    """Export one h5ad per batch, ready to run a per-sample tool on.

    Doublet callers are in-memory, single-sample tools, so the export is N
    per-batch files rather than one. That is the moat rather than a cost: the
    pooled atlas is never materialized and peak RSS is one library.

    **What this adds over the loop you would write yourself** is the key check.
    A tool sees only the h5ad it is handed, so if two of that file's cells carry
    the same key there is nothing to join its answers back on — and the failure
    only shows up much later, as a duplicate-key error at import time or, worse,
    as scores silently landing on the wrong cell. Measured on a real
    CELLxGENE-derived atlas, one batch in 1,086 had a duplicated index and it
    held 15.5% of the file's cells.

    **Two identities are checked, not one.** The tools read `obs_names` —
    Scrublet writes `barcode = adata.obs_names`, scDblFinder uses
    `colnames(sce)` — while the *import* joins on the resolved `key`. Those are
    the same column in the ordinary case, but not when auto-resolution picks a
    unique non-index column over a duplicated index. Checking only `key` would
    then pass a batch whose `obs_names` a tool cannot tell apart, which is the
    exact failure this guard exists to prevent, so both must be unique within a
    batch. `key_is_obs_index` in the result says whether they coincided.

    Args:
        path: Source SCX file (str, os.PathLike, or an open Experiment).
        out_dir: Directory for the per-batch h5ad files; created if absent.
        batch_key: obs column to split on (e.g. "donor_id", "sample_id").
        key: The join key the tool's output will carry back. None resolves it
            with `diagnose_obs_key`, the same way `obs_import` would — which
            prefers the obs index when it is unique, so the returned `key` is
            usually `"obs_names"`. That is the name to pass straight back to
            `obs_import(key=...)`; the physical `__index_level_0__` field is not
            a column you can address.
        batches: Restrict to these batch values. None exports every batch.
        on_ambiguous_key: What to do with a batch whose key is not unique
            *within that batch* — "error" (default) refuses before writing
            anything, "skip" omits it and records why, "warn" exports it
            anyway. Only choose "warn" if you have another way to rejoin.
        overwrite: Overwrite existing per-batch files. Off by default.
        **kwargs: Passed through to `pyscx.to_h5ad` (e.g. `min_counts`).

    Returns:
        dict with `key`, `key_is_globally_unique`, `out_dir`, `n_batches`,
        `n_cells_exported`, and `batches` — a list of per-batch dicts carrying
        `batch`, `path`, `n_cells`, `key_unique_within_batch` and, for anything
        not written, `skipped_reason`.

    Note:
        `key_is_globally_unique` is the one to read before planning the import.
        When it is True you can concatenate every tool output and import once,
        which is what you want because `overwrite` **replaces** rather than
        merges. When it is False the keys only distinguish cells inside their
        own batch, so you must import with a composite key that includes
        `batch_key`.

    Example:
        r = pyscx.export_batches("atlas.scx", "batches/", batch_key="donor_id")
        assert r["key_is_globally_unique"]      # else import per batch
        # ... run the tool on each r["batches"][i]["path"] ...
        pyscx.doublet_import("atlas.scx", "all_calls.csv", tool="scrublet")
    """
    import pathlib as _pathlib

    if on_ambiguous_key not in ("error", "skip", "warn"):
        raise ValueError(
            "on_ambiguous_key must be 'error', 'skip' or 'warn'; got "
            f"{on_ambiguous_key!r}"
        )

    src = _coerce_path(path)
    out_dir = _pathlib.Path(_coerce_path(out_dir, allow_experiment=False))

    # Resolve to a key that actually *works* rather than to whatever the plain
    # fallback order lands on: on a merged atlas the fallback is usually the
    # obs index, which is exactly the column that turns out to be duplicated.
    # The chosen key comes back in the result so the import can be given the
    # same one -- `obs_import` / `doublet_import` auto-resolve by fallback
    # order, so if the two differ the caller must pass `key=` there too.
    no_unique_candidate = ""
    if key is None:
        diag = diagnose_obs_key(src)
        suggestion = diag.get("suggestion")
        # A comma-joined suggestion is a *composite* — `diagnose_obs_key`
        # offers one when no single column is unique. It cannot be used here:
        # the exported h5ad identifies its rows by `obs_names` alone, so a
        # tool's output can only ever carry one component back. Falling
        # through to `resolved_key` means the per-batch guard below does the
        # refusing, with the full diagnosis attached — which is the accurate
        # message. Using the composite as a column name would fail with
        # "key 'a,b' is neither an obs column nor the obs index", which
        # explains nothing.
        if suggestion and "," in suggestion:
            suggestion = None
        # The obs index no longer needs a preference hard-coded here:
        # `diagnose_obs_key` ranks it first among usable keys, so the suggestion
        # already is the index whenever the index is unique. That is the right
        # place for the rule -- the index is what the tools hand back
        # (`adata.obs_names` / `colnames(sce)`), so export and import agree on a
        # key by construction instead of by two lists that could drift.
        key = suggestion or diag.get("resolved_key")
        if key is None:
            raise ValueError(
                "no obs column could serve as a join key: " + diag["summary"]
            )
        if not suggestion:
            # Fell back rather than found a unique column. Carried into the
            # per-batch refusal below so its remedy does not promise a key the
            # file does not have.
            no_unique_candidate = " " + diag["summary"]

    exp = _open_native(src)
    obs = exp.read_obs()
    # `read_obs()` returns PHYSICAL rows, which is exactly the row space
    # `to_h5ad(obs_mask=)` wants. It also *includes* logically deleted rows, so
    # the per-batch counts below can overcount; `to_h5ad` ANDs with the
    # deletion keep mask, so the exports themselves stay right.
    if batch_key not in obs.columns:
        raise ValueError(
            f"batch_key {batch_key!r} is not an obs column; columns are "
            f"{list(obs.columns)}"
        )
    if key in obs.columns:
        key_values = obs[key]
    elif obs.index.name == key or key in (
        "obs_names",
        "index",
        "_index",
        "__index_level_0__",
    ):
        key_values = obs.index.to_series()
    else:
        raise ValueError(
            f"key {key!r} is neither an obs column nor the obs index; columns "
            f"are {list(obs.columns)}"
        )

    globally_unique = bool(key_values.is_unique)

    # The identity the TOOLS see is the exported h5ad's `obs_names` — Scrublet
    # writes `barcode = adata.obs_names`, scDblFinder uses `colnames(sce)` —
    # and that is the obs index, not necessarily the resolved `key`. Checking
    # only `key` lets a batch through whose obs_names are duplicated, which is
    # precisely the silent wrong-join this helper exists to prevent: on a file
    # with a duplicated index but a unique `cell_uid`, auto-resolution picks
    # `cell_uid`, every batch passes, and the tool is handed a file whose rows
    # it cannot tell apart.
    #
    # So both are checked. `key` uniqueness is what the *import* needs;
    # obs_names uniqueness is what the *tool* needs, and a batch failing either
    # is unusable.
    index_values = obs.index.to_series()
    key_is_index = key_values.equals(index_values)

    # Plan every batch before writing any of them, so an "error" verdict costs
    # nothing rather than leaving a half-finished directory behind.
    plan = []
    values = obs[batch_key]
    wanted = list(batches) if batches is not None else list(values.drop_duplicates())
    for b in wanted:
        mask = (values == b).to_numpy()
        if not mask.any():
            raise ValueError(f"batch {b!r} matches no rows in {batch_key!r}")
        key_unique = bool(key_values[mask].is_unique)
        names_unique = bool(index_values[mask].is_unique)
        plan.append({"batch": b, "mask": mask, "n_cells": int(mask.sum()),
                     "key_unique_within_batch": key_unique and names_unique,
                     "obs_names_unique_within_batch": names_unique})

    bad = [p["batch"] for p in plan if not p["key_unique_within_batch"]]
    if bad and on_ambiguous_key == "error":
        n = sum(p["n_cells"] for p in plan if not p["key_unique_within_batch"])
        # Name whichever identity actually failed. "key X is not unique" is
        # actively misleading when X is unique and it was obs_names that
        # collided — the user would go looking for a better key and find that
        # the one they have is already fine.
        names_bad = [p["batch"] for p in plan
                     if not p["obs_names_unique_within_batch"]]
        detail = f"key {key!r} is not unique"
        if names_bad and not key_is_index:
            detail = (
                f"the exported obs_names are not unique (the resolved key "
                f"{key!r} is a different column, and a tool reads obs_names)"
            )
        raise ValueError(
            f"{detail} within {len(bad)} of {len(plan)} "
            f"batches ({n} cells): {bad[:5]}{' ...' if len(bad) > 5 else ''}. "
            "A tool run on those files could not tell two cells apart, so its "
            "output could not be joined back. Pass a key that is unique within "
            "each batch (pyscx.diagnose_obs_key lists the candidates), or "
            "on_ambiguous_key='skip' to export the rest." + no_unique_candidate
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stems = _plan_batch_filenames([p["batch"] for p in plan])
    results, exported = [], 0
    for p in plan:
        entry = {"batch": p["batch"], "n_cells": p["n_cells"],
                 "key_unique_within_batch": p["key_unique_within_batch"],
                 "obs_names_unique_within_batch": p["obs_names_unique_within_batch"]}
        if not p["key_unique_within_batch"] and on_ambiguous_key == "skip":
            entry["path"] = None
            entry["skipped_reason"] = (
                f"key {key!r} is not unique within this batch"
                if p["obs_names_unique_within_batch"]
                else "the exported obs_names are not unique within this batch"
            )
            results.append(entry)
            continue
        # The batch label comes from obs data, so it can contain a path
        # separator or `..`; joining it raw would write outside out_dir, and
        # two labels can sanitise to the same stem (see _plan_batch_filenames).
        dest = out_dir / f"{stems[p['batch']]}.h5ad"
        if dest.exists() and not overwrite:
            raise FileExistsError(
                f"{dest} already exists; pass overwrite=True to replace it"
            )
        to_h5ad(src, str(dest), obs_mask=p["mask"], **kwargs)
        entry["path"] = str(dest)
        results.append(entry)
        exported += p["n_cells"]

    return {
        "key": key,
        "key_is_globally_unique": globally_unique,
        # False means the tools will return obs_names while the import joins on
        # a different column — see the `key_is_index` note in the docstring.
        "key_is_obs_index": key_is_index,
        "batch_key": batch_key,
        "out_dir": str(out_dir),
        "n_batches": sum(1 for r in results if r.get("path")),
        "n_cells_exported": exported,
        "batches": results,
    }


def _coerce_key(key):
    """Normalise a `key=` / `source_key=` argument to a list of str, or None.

    A bare str is one component; any other iterable is a sequence of them. A
    non-str scalar (e.g. an int column label) is also one component -- iterating
    it would raise "'int' object is not iterable", which says nothing about the
    argument that was wrong. The native layer takes `Option<Vec<String>>`, so
    None must stay None: it is what selects auto-resolution rather than an empty
    composite.
    """
    if key is None:
        return None
    if isinstance(key, str):
        return [key]
    try:
        return [str(k) for k in key]
    except TypeError:
        return [str(key)]


def obs_import(path, table, *, key=None, source_key=None, **kwargs):
    """Import a delimited annotation table (CSV/TSV) as `obs` columns, in place.

    The generic importer behind the doublet-caller workflow: run a tool
    externally, have it write `barcode,score,call`, and land those columns on
    an existing SCX file. Nothing here is doublet-specific.

    Joins **by key string, never by row position** — a tool run per library
    returns rows in its own order, and a positional import would put every
    value on the wrong cell while still producing a correctly-shaped column.
    Target rows the table does not cover get `null`, never a fabricated `0`.

    In place via the same harness `append` uses: `X`, layers, `var`, the CSC
    sidecar, `.raw`, deletion vectors and predicate indexes are preserved, and
    `pyscx.rollback(path)` undoes the whole import.

    Args:
        path: Target SCX file (str, os.PathLike, or an open Experiment).
        table: Source .csv / .tsv / .txt, or an .h5ad whose `/obs` holds the
            columns (str or os.PathLike). The h5ad route needs a build with
            HDF5 support; without it the error says to write a CSV instead.
        key: Target-side join key. None auto-resolves with the same preference
            order the target side uses (obs index, then `barcode`/`cell_id`/…).
            A str names one column; `"obs_names"` names the obs index. A list of
            str builds a composite key — the right answer for a multi-library
            merge where `sample_id` + `barcode` is unique but neither is alone.
            The fusing separator is internal and not configurable.
        source_key: Source-side column(s) for the same key, when the table
            spells it differently. Pairs **positionally** with `key`, mirroring
            pandas `left_on` / `right_on`, so the two must have equal length::

                obs_import(t, "ml.csv", key=["sample_id", "obs_names"],
                           source_key=["sample_id", "barcode"])

            None means the source uses the `key` names, which is the common
            case and keeps the two sides impossible to desync.
        columns: Import only these source columns. None imports every non-key
            column.
        rename: `{source_name: new_name}`, applied before `prefix`.
        prefix: Prepended to every imported column name.
        keep_key_columns: Also import the key column(s) as ordinary annotations.
            Off by default — the key is usually already in obs.
        delimiter: One-character override. None sniffs from the extension
            (`.csv` / `.tsv` / `.tab`), then from the header line.
        status_column: Obs column recording "present"/"absent" per row.
        uns_key: `uns` key to merge the table's metadata under.
        uns_keys: `/uns` keys to carry across from an h5ad source. Opt-in --
            `/uns` routinely holds large arrays and types the reader skips, so
            nothing comes across unless named. A key that is not there is an
            error. Meaningless for a delimited table, and requesting one is an
            error rather than a silent no-op.
        overwrite: **Replaces, never merges.** A colliding column is dropped and
            rebuilt from this table alone, so importing several per-batch tables
            one after another keeps only the last. Concatenate them and import
            once. Without this, a collision is an error.
        on_missing_rows: "null" (default) leaves uncovered target rows NULL and
            marks them absent; "error" refuses. "zero" is an accepted alias for
            "null" — it names the shared policy enum, whose `zero` is literal
            only on `cellbender_import`, where a missing *matrix* row really is
            zeros.
        on_extra_rows: "warn" (default) skips source rows the target lacks;
            "error" refuses.
        dry_run: Run every validation and the join, then return the summary
            without writing. Also attaches a `key_diagnosis` to the result.

    Returns:
        dict with `n_obs`, `n_matched`, `n_target_rows_absent`,
        `n_source_rows_absent`, `obs_key_column`, `obs_columns_added`,
        `obs_index_dropped`, the source's `format` / `delimiter` (None for
        h5ad) / `n_rows_in_source` / `uns_keys_imported`, and (on a dry run)
        `key_diagnosis`.

    Example:
        # Look before you leap on a large file.
        r = pyscx.obs_import("atlas.scx", "calls.csv", dry_run=True)
        print(r["n_matched"], "of", r["n_obs"], "cells matched")
        pyscx.obs_import("atlas.scx", "calls.csv", status_column="dbl_status")
    """
    key = _coerce_key(key)
    source_key = _coerce_key(source_key)
    return _obs_import_native(_coerce_path(path), _coerce_path(table, allow_experiment=False),
                              key=key, source_key=source_key, **kwargs)


def diagnose_obs_key(path, key=None):
    """Report which obs columns could serve as an `obs_import` join key.

    Read-only. Reach for this when an import fails on a duplicated key: on a
    merged atlas the obvious candidates are often not unique, and the column
    that is may be one no fallback list would guess (on a CELLxGENE-derived
    file it is `soma_joinid`, with the obs index 10x-duplicated).

    Every name reported -- `resolved_key`, `unique_columns`, `suggestion` -- is
    one `obs_import(key=...)` accepts, including `"obs_names"` for the obs index.
    `unique_columns` is ordered best-candidate-first and holds only columns that
    can actually serve as a key; a unique column the join would refuse (a float,
    whose text form is not guaranteed to agree across two independently written
    sides) is listed separately under `unusable_unique_columns`.

    Returns a dict with `resolved_key`, `resolved_cardinality`,
    `unique_columns`, `unusable_unique_columns`, `unique_pairs`, `suggestion`
    and a printable `summary`.
    """
    return _diagnose_obs_key_native(_coerce_path(path), _coerce_key(key))


def doublet_import(path, table, *, tool, key=None, source_key=None, **kwargs):
    """Import a doublet caller's output, normalised to canonical obs columns.

    The doublet-specific wrapper over `obs_import`. Same in-place, key-joined,
    `pyscx.rollback`-able import — plus the one thing that needs per-tool
    knowledge: every caller names its score and call differently, and consensus
    code downstream should not have to branch on which tool ran.

    Writes, for `key_added="<K>"` (default: the tool name):

        obs["<K>_score"]      float32              higher = more doublet-like
        obs["<K>_predicted"]  boolean (nullable)   omitted when the tool has no call
        obs["<K>_status"]     object (str)         "present" / "absent"
        obs["<K>_<native>"]   ...              every other source column
        uns["<K>"]                             tool, source columns, join report

    These names match what a native SCX doublet run writes, so an imported
    result and a native one are drop-in comparable.

    Args:
        path: Target SCX file (str, os.PathLike, or an open Experiment).
        table: The caller's output .csv / .tsv, or an `.h5ad` whose `/obs`
            holds the columns — which is what the scanpy-resident tools write,
            since `sc.pp.scrublet(adata)` sets `adata.obs` in place. The h5ad
            route needs a build with HDF5 support; without it the error says to
            write a CSV instead.
        tool: One of `pyscx.doublet_tools()`: "scdblfinder", "scrublet",
            "doubletfinder", "doubletdetection", "solo", "scds", "generic".
        key: Target-side join key, exactly as for `obs_import`. None
            auto-resolves; `"obs_names"` names the obs index; a str
            names one column; a list builds a composite — the right answer for a
            multi-library merge where `sample_id` + `barcode` is unique but
            neither is alone.
        source_key: Source-side column(s) for the same key, exactly as for
            `obs_import` — use it when the caller's output spells the key
            differently from the target obs.
        key_added: Canonical prefix `<K>`. Defaults to `tool`, so two tools land
            side by side without colliding.
        score_column: Override the profile's score column. Required for
            `tool="generic"`.
        call_column: Override the profile's call column. Supplying one is also
            how you opt a score-only tool (scds) into a `<K>_predicted`.
        call_true / call_false: Override the text tokens meaning doublet and
            singlet. Needed when a tool version renames its classes. With only
            `call_true`, anything else non-null is treated as a singlet.
        keep_native_columns: Keep every other source column as `<K>_<native>`.
            True by default.
        delimiter: One-character override; None sniffs from the extension.
            Ignored for an h5ad source.
        uns_keys: `/uns` keys to carry across from an h5ad source, nested under
            `uns["<K>"]["source_uns"]` so the tool's own metadata keeps its
            names and cannot collide with the wrapper's record. Opt-in; a key
            that is not there is an error.
        overwrite: **Replaces, never merges.** Re-importing per-batch tables one
            after another keeps only the last — concatenate them and import once.
        on_missing_rows: "null" (default) leaves uncovered cells NULL and marks
            them absent ("zero" is an accepted alias for the same policy); "error"
            refuses.
        on_extra_rows: "warn" (default) skips source rows the target lacks;
            "error" refuses.
        dry_run: Validate and join without writing; also returns a
            `key_diagnosis`.

    Returns:
        dict with the `obs_import` join fields plus `tool`, `key_added`,
        `score_source_column`, `call_source_column`, `canonical_columns`,
        `native_columns`, `dropped_alias_columns`, `format` ("table" or
        "h5ad") and `uns_keys_imported`.

    Note:
        A tool that emits no call column never gets a `<K>_predicted`. The
        importer will not threshold a score on your behalf — that is a
        scientific decision it does not own.

    Example:
        # Look first: on a merged atlas the obvious key is often not unique.
        r = pyscx.doublet_import("atlas.scx", "calls.csv",
                                 tool="scdblfinder", dry_run=True)
        print(r["n_matched"], "of", r["n_obs"], "cells matched")
        pyscx.doublet_import("atlas.scx", "calls.csv", tool="scdblfinder")
    """
    return _doublet_import_native(_coerce_path(path),
                                  _coerce_path(table, allow_experiment=False),
                                  tool=tool, key=_coerce_key(key),
                                  source_key=_coerce_key(source_key), **kwargs)


_DOUBLET_CONSENSUS_METHODS = ("majority", "any", "all", "mean_rank")


def _consensus_calls(obs, column):
    """Normalise a canonical `<K>_predicted` column to (voted, called) arrays.

    **A null is not a vote.** That is the whole reason the importer refuses to
    fabricate a `0.0` for a cell a tool never saw, and the distinction has to
    survive to here: `voted` is False for those rows, so they neither support
    nor oppose a call. Collapsing them to False — which is what a plain
    `sum(...) >= k` over a nullable column does — would silently turn "no
    information" into "every tool said singlet".

    Accepts every dtype the round trip produces. `read_obs()` now returns pandas
    nullable `boolean` for any arrow boolean, which is also what
    `doublet_consensus` writes — but object holding `True`/`False`/`None`, plain
    `bool` and a strict 0/1 numeric column stay accepted, because a file written
    before that change, or a hand-built AnnData, still carries them. Those are
    the same tokens `scx_convert::doublet::coerce_call` accepts, so the two ends
    of the pipeline cannot drift on what a call is.
    """
    import numpy as _np
    import pandas as _pd

    col = obs[column]
    voted = ~_pd.isna(col).to_numpy()
    called = _np.zeros(len(col), dtype=bool)
    if not voted.any():
        return voted, called

    present = col.to_numpy(dtype=object)[voted]
    uniq = _pd.unique(present)

    if all(isinstance(v, (bool, _np.bool_)) for v in uniq):
        called[voted] = present.astype(bool)
        return voted, called

    # Strict 0/1 only. A column of arbitrary numbers is a score that was named
    # like a call, and thresholding it here is a scientific decision this
    # helper does not own.
    numeric = all(
        isinstance(v, (int, float, _np.integer, _np.floating))
        and not isinstance(v, bool)
        for v in uniq
    )
    if numeric and {float(v) for v in uniq} <= {0.0, 1.0}:
        called[voted] = present.astype("float64") == 1.0
        return voted, called

    bad = next(v for v in uniq if not isinstance(v, (bool, _np.bool_)))
    raise ValueError(
        f"obs[{column!r}] holds {bad!r}, which is not a doublet call. A "
        "canonical `<key>_predicted` column is a nullable boolean (or a strict "
        "0/1). A value like 'doublet' means `keys` names a tool's own source "
        "column rather than the canonical one the importer derives — pass the "
        "`key_added` you imported under (default: the tool name), not the "
        "tool's native column."
    )


def _consensus_scores(obs, column):
    """Read a canonical `<K>_score` column as float64 with NaN for null."""
    import numpy as _np
    import pandas as _pd

    try:
        numeric = _pd.to_numeric(obs[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"obs[{column!r}] is not numeric, so it cannot be ranked; "
            f"method='mean_rank' needs every key's score column. ({exc})"
        ) from exc
    return numeric.to_numpy(dtype="float64", na_value=_np.nan)


# Suffix of the column `doublet_consensus` writes for every method, used as the
# primary signal that a key is a consensus output rather than a caller (see
# `_consensus_output_columns`). Assumes nothing else writes it: a hand-built obs
# column named `<K>_n_tools_voting` would exclude `<K>` from `keys=None`
# discovery. Naming the key explicitly still works, and warns.
_CONSENSUS_VOTING_MARKER = "_n_tools_voting"


def _consensus_output_columns(obs, uns):
    """obs columns a *previous* `doublet_consensus` wrote.

    A consensus writes `<K>_predicted` (and `<K>_score` for `mean_rank`), which
    are exactly the columns discovery scans for — so without this, `keys=None`
    counts an earlier consensus as an extra voting "tool". That silently
    double-weights whichever callers fed it: with three callers where two agree,
    a stale 2-tool consensus voting as a 4th tool turned 10 of 50 cells from
    doublet to singlet, and reported `n_tools_voting = 4` for a 3-tool run.

    Two independent signals, union'd:

    * **`<K>_n_tools_voting`** — written *only* by `doublet_consensus`, for every
      method, and never by any `DoubletProfile`. This is the load-bearing one: it
      needs no `uns`, so it still works when another op dropped or rewrote uns,
      on a hand-built AnnData, and on files already polluted by the bug (no
      migration needed). It also covers both suffixes at once, catching a prior
      `mean_rank` consensus's `<K>_score` that a `_predicted`-only check misses.
    * **`columns_added` from each `uns["<X>_consensus"]` record** —
      authoritative when present, and stated in the same vocabulary discovery
      uses (obs column names) rather than inferred from key-name arithmetic.
      Matched on a positive consensus signature, so a tool legitimately imported
      under `key_added="foo_consensus"` is not mistaken for one. A partial or
      pre-fix record that fails that match is covered by the marker signal
      above — which is the reason this one need not guess.
    """
    derived = set()

    if isinstance(uns, dict):
        for name, rec in uns.items():
            if not (isinstance(name, str) and name.endswith("_consensus")):
                continue
            if not isinstance(rec, dict):
                continue
            k = name[: -len("_consensus")]
            # Match on a POSITIVE consensus signature, not just the key name. A
            # *tool* imported as `key_added="foo_consensus"` also lands at
            # `uns["foo_consensus"]` (doublet_import writes `uns["<K>"]`), and
            # treating that as a consensus would wrongly exclude the real tool
            # `foo` from discovery. A consensus record always carries `method`,
            # `key_added` and a non-empty `columns_added`; an import record
            # carries `tool` / `source_call_column` and none of those.
            cols = rec.get("columns_added")
            if not (
                rec.get("key_added") == k
                and isinstance(rec.get("method"), str)
                and isinstance(cols, (list, tuple))
                and cols
            ):
                continue
            derived.update(str(c) for c in cols)

    m = _CONSENSUS_VOTING_MARKER
    for c in obs.columns:
        if isinstance(c, str) and c.endswith(m) and len(c) > len(m):
            k = c[: -len(m)]
            derived.update({f"{k}_predicted", f"{k}_score"})
    return derived


def _resolve_consensus_keys(obs, uns, keys, method, key_added):
    """Resolve and validate the `keys` list against what is actually on obs.

    Returns `(keys, excluded, are_consensus)`: the resolved caller keys, the
    consensus keys discovery skipped (empty on the explicit-`keys` path), and
    the subset of `keys` that are themselves consensus outputs (empty on the
    discovery path, since those are excluded there).
    """
    import warnings as _warnings

    suffix = "_score" if method == "mean_rank" else "_predicted"
    all_suffixed = sorted(
        c[: -len(suffix)] for c in obs.columns
        if c.endswith(suffix) and len(c) > len(suffix)
    )
    derived = _consensus_output_columns(obs, uns)
    # Two complementary guards. The `!= key_added` scalar check covers the FIRST
    # run under a given `key_added` (no `<key_added>_consensus` record and no
    # `<key_added>_n_tools_voting` column exist yet, so neither signal fires);
    # `derived` covers every *other* consensus already on the file. Dropping
    # either one reopens the bug for one of those two cases.
    excluded = [
        k for k in all_suffixed
        if k != key_added and f"{k}{suffix}" in derived
    ]
    available = [
        k for k in all_suffixed
        if k != key_added and f"{k}{suffix}" not in derived
    ]

    if keys is None:
        if not available:
            why = (
                f" The only `<key>{suffix}` columns on obs are previous consensus "
                f"outputs ({excluded}), which are derived from callers rather "
                "than callers themselves."
                if excluded
                else ""
            )
            raise ValueError(
                f"no `<key>{suffix}` columns on obs, so there is nothing to "
                f"reach a consensus over.{why} Run `pyscx.doublet_import` for "
                "each tool first; obs columns present are "
                f"{list(obs.columns)}"
            )
        if excluded:
            _warnings.warn(
                f"doublet_consensus(keys=None) discovered {available} and "
                f"skipped {excluded}, which are previous consensus outputs "
                "rather than callers — counting one would double-weight the "
                "tools that fed it and inflate `n_tools_voting`. Pass "
                "`keys=[...]` explicitly to override.",
                UserWarning,
                stacklevel=3,
            )
        return available, excluded, []

    if isinstance(keys, str):
        keys = [keys]
    keys = [str(k) for k in keys]
    if not keys:
        raise ValueError("`keys` is empty; name at least one imported tool")
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(
            f"`keys` repeats {dup}; a tool listed twice would vote twice"
        )
    if key_added in keys:
        raise ValueError(
            f"key_added={key_added!r} is also in `keys`, so the consensus would "
            "read its own output. Pick a different key_added."
        )

    for k in keys:
        if f"{k}{suffix}" in obs.columns:
            continue
        if method != "mean_rank" and f"{k}_score" in obs.columns:
            # A score with no call. There are TWO reasons for that and they need
            # different remedies, so read `uns["<K>"]["call_column_status"]`
            # (written by `doublet_import`) instead of asserting one. The old
            # message claimed "That tool emits no call column" unconditionally,
            # which is false for any profile that declares one — e.g.
            # doubletdetection's `doublet_label` — and sent the user to the wrong
            # three fixes.
            rec = uns.get(k) if isinstance(uns, dict) else None
            rec = rec if isinstance(rec, dict) else {}
            status = rec.get("call_column_status")
            tool = rec.get("tool")
            expected = list(rec.get("expected_call_columns") or [])
            prefix = rec.get("expected_call_prefix")
            named = f"{tool!r}" if tool else f"{k!r}"
            common = (
                f"drop {k!r} from `keys`, use method='mean_rank' to combine "
                "scores instead, or re-import it with `call_column=` to derive "
                "a call."
            )
            if status == "declared_but_absent":
                want = f"one of {expected}" if expected else "a call column"
                if prefix:
                    want += f" or a column starting with {prefix!r}"
                raise ValueError(
                    f"obs has {k!r}'s score but no {f'{k}_predicted'!r}, so it "
                    f"cannot vote on a call. The {named} profile DOES emit a "
                    f"call column — it expects {want} — but the table you "
                    "imported carried none of those names, so the import wrote "
                    f"a score only (uns[{k!r}]['call_column_status'] == "
                    f"'declared_but_absent'). Re-import with `call_column=` "
                    "naming the column your table actually uses, or " + common
                )
            if status == "not_declared":
                raise ValueError(
                    f"obs has {k!r}'s score but no {f'{k}_predicted'!r}, so it "
                    f"cannot vote on a call. That tool ({named}) emits no call "
                    "column — " + common
                )
            # Unknown: imported by an older pyscx, uns dropped, or a hand-built
            # AnnData. Assert NEITHER cause rather than guess wrong.
            raise ValueError(
                f"obs has {k!r}'s score but no {f'{k}_predicted'!r}, so it "
                "cannot vote on a call. Either that tool emits no call column "
                "(scds), or the imported table did not carry the call column "
                f"its profile expects — uns[{k!r}]['call_column_status'] "
                "records which, for imports done by pyscx 0.12.1+. " + common
            )
        hint = (
            f" ({excluded} are previous consensus outputs, not callers, so "
            "discovery skips them; naming one explicitly is allowed.)"
            if excluded
            else ""
        )
        raise ValueError(
            f"obs has no {f'{k}{suffix}'!r} column. Keys available for "
            f"method={method!r}: {available or 'none'}{hint}"
        )

    # Naming another consensus explicitly is a coherent operation (reconciling
    # two disjoint tool panels, then combining), so allow it — but say what it
    # costs, because the common way to arrive here is hand-rolling discovery
    # from `read_obs().columns` and reproducing the bug by hand.
    are_consensus = [k for k in keys if f"{k}{suffix}" in derived]
    for k in are_consensus:
        sub = None
        if isinstance(uns, dict) and isinstance(uns.get(f"{k}_consensus"), dict):
            sub = uns[f"{k}_consensus"].get("keys")
        over = f", which is itself a consensus over {list(sub)}" if sub else ""
        _warnings.warn(
            f"`keys` names {k!r}{over} rather than a caller. Those tools "
            f"effectively vote twice, and `{key_added}_n_tools_voting` counts "
            f"{k!r} as one tool. Pass only caller keys, or keep this if a "
            "consensus-of-consensuses is what you intend.",
            UserWarning,
            stacklevel=3,
        )
    return keys, [], are_consensus


def doublet_consensus(target, *, keys=None, method="majority",
                      key_added="doublet", quantile=None, overwrite=False,
                      index_obs=None, index_preset=None):
    """Combine several doublet callers' imported calls into one consensus.

    Once N tools' results are canonical obs columns on one file, the consensus
    is arithmetic — no tool-specific knowledge left. This is the last step of
    the interop, and it is pure Python: nothing here knows what a doublet is.

    **Null-aware throughout.** A tool that never saw a cell does not vote on
    it, and a cell no tool voted on comes out `null` — never `False`. Those are
    different facts, and the importer preserved the difference precisely so it
    could be honoured here.

    Writes, for `key_added="<K>"`:

        obs["<K>_predicted"]         bool, nullable   null where nothing voted
        obs["<K>_n_tools_calling"]   int32            how many said doublet
        obs["<K>_n_tools_voting"]    int32            how many had an opinion
        obs["<K>_score"]             f32              mean_rank only
        uns["<K>_consensus"]                          inputs, rule and counts

    Read `<K>_n_tools_voting` before trusting a `False`: a 0 there means the
    cell was never assessed, which is why `<K>_predicted` is null beside it.

    Args:
        target: An SCX file (str, os.PathLike, or an open Experiment) — written
            in place, one atomic commit, undone by `pyscx.rollback(path)` — or
            an in-memory `AnnData`, mutated directly.
        keys: The `key_added` values the tools were imported under (default:
            the tool names). None discovers every key on obs that carries the
            column this `method` needs, and records what it found.
        method: How votes combine.
            "majority" — more than half of the voting tools said doublet. An
                even split is not a majority, so it is False, not null: the
                tools disagreed, which is information, unlike no vote at all.
            "any" — at least one voting tool said doublet.
            "all" — every voting tool said doublet. Tools that did not cover
                the cell are not counted against it.
            "mean_rank" — ignores the calls and combines the *scores*: each
                tool's score is ranked within the cells that tool covered and
                normalised to [0, 1], then averaged. Needs `quantile`.
        key_added: Prefix `<K>` for the written columns.
        quantile: For "mean_rank" only, and required there: the fraction of
            assessed cells to call doublets (e.g. 0.06 for an expected 6%
            doublet rate). Required rather than defaulted because picking a
            cutoff is a scientific decision this helper does not own — if you
            want the tools' own thresholds to decide, use "majority".
        overwrite: Replace existing `<K>_*` columns. Without it a collision is
            an error, so a second run cannot silently rewrite the first.
        index_obs / index_preset: Index the obs predicate index over these
            columns as part of the same commit, instead of the columns the file
            already indexes. Relevant only for a file target, and rarely
            needed: writing the consensus replaces obs, and a replaced obs
            carries its existing index forward, so pushdown survives this call
            untouched. Naming columns here *narrows* the index to them, and any
            column the file indexed that this list omits is reported as a
            `UserWarning`.

    Returns:
        The same dict written to `uns["<K>_consensus"]`: `method`, `keys`,
        `key_added`, `n_obs`, `n_predicted_doublet`, `n_predicted_singlet`,
        `n_no_vote`, `columns_added`, `per_key` counts, and for "mean_rank"
        the `quantile` and the `threshold` it resolved to.

    Note:
        A tool with no call column (scds) has no `<K>_predicted` and cannot
        vote on a call — naming it in `keys` for a call-based method is an
        error rather than a silent omission, since dropping a voter changes
        the result. It works fine under "mean_rank".

    Example:
        pyscx.doublet_import("atlas.scx", "scdbl.csv", tool="scdblfinder")
        pyscx.doublet_import("atlas.scx", "scrub.csv", tool="scrublet")
        r = pyscx.doublet_consensus("atlas.scx",
                                    keys=["scdblfinder", "scrublet"])
        print(r["n_predicted_doublet"], "doublets;", r["n_no_vote"], "unassessed")
    """
    import numpy as _np
    import pandas as _pd

    if method not in _DOUBLET_CONSENSUS_METHODS:
        raise ValueError(
            f"method must be one of {list(_DOUBLET_CONSENSUS_METHODS)}; got "
            f"{method!r}"
        )
    if not key_added:
        raise ValueError("key_added must be a non-empty string")
    if method == "mean_rank":
        if quantile is None:
            raise ValueError(
                "method='mean_rank' needs `quantile` — the fraction of "
                "assessed cells to call doublets (e.g. quantile=0.06). It has "
                "no default because a score cutoff is a scientific decision; "
                "use method='majority' to let each tool's own call decide."
            )
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError(
                f"quantile must be strictly between 0 and 1; got {quantile!r}"
            )
    elif quantile is not None:
        raise ValueError(
            f"`quantile` only applies to method='mean_rank'; got method="
            f"{method!r}. Its calls come from each tool's own threshold."
        )

    # An AnnData is mutated in place; anything path-like is read, computed and
    # written back. Checked before `_coerce_path` because an AnnData has no
    # __fspath__ and would otherwise fall through to its generic TypeError.
    is_adata = (
        not isinstance(target, str)
        and not hasattr(target, "__fspath__")
        and hasattr(target, "obs")
        and hasattr(target, "uns")
    )
    if is_adata:
        obs = target.obs
        # Guaranteed present: the `is_adata` duck-type above requires it.
        uns = target.uns
    else:
        path = _coerce_path(target)
        exp = _open_native(path)
        # Read uns HERE, before discovery, rather than after the compute: key
        # resolution needs it to tell a previous consensus's columns from a
        # caller's. `read_uns` is a metadata-only read (it does not touch obs,
        # var, obsm or X) and returns None when the file has no uns section.
        # This replaces the later read, so it is one fewer, not one more — and
        # the file is not mutated between here and `modify_metadata` below.
        uns = exp.read_uns()
        uns = dict(uns) if uns else {}
        # PHYSICAL rows — the row space `modify_metadata` validates against,
        # and the one logically-deleted rows still occupy.
        obs = exp.read_obs()

    keys, keys_excluded, keys_are_consensus = _resolve_consensus_keys(
        obs, uns, keys, method, key_added
    )

    n = len(obs)
    n_voting = _np.zeros(n, dtype=_np.int32)
    n_calling = _np.zeros(n, dtype=_np.int32)
    per_key = {}

    # `n_tools_calling` always counts tools whose own call was True, whatever
    # the method — so it stays a fact about the tools rather than about the
    # rule. `n_tools_voting` counts what fed *this* method's decision, which
    # for mean_rank is a score rather than a call.
    for k in keys:
        entry = {}
        if f"{k}_predicted" in obs.columns:
            voted, called = _consensus_calls(obs, f"{k}_predicted")
            entry["predicted_column"] = f"{k}_predicted"
            entry["n_calling"] = int(called.sum())
            n_calling += called
            if method != "mean_rank":
                entry["n_voting"] = int(voted.sum())
                n_voting += voted
        if method == "mean_rank":
            entry["score_column"] = f"{k}_score"
            covered = ~_np.isnan(_consensus_scores(obs, f"{k}_score"))
            entry["n_voting"] = int(covered.sum())
            n_voting += covered
        per_key[k] = entry

    voted_any = n_voting > 0
    record = {
        "method": method,
        "keys": list(keys),
        "key_added": key_added,
        "n_obs": int(n),
        "per_key": per_key,
        # What discovery narrowed, so the file itself records it. Empty unless
        # `keys=None` skipped a previous consensus output.
        "keys_excluded": list(keys_excluded),
        # The subset of the final `keys` that are themselves consensus outputs.
        # Non-empty only when the caller explicitly opted in, so an unusual
        # choice is visible to whoever inherits the file, not just to the
        # transient Python warning.
        "keys_that_are_consensus": list(keys_are_consensus),
    }

    # Columns are held as bare arrays, never Series, so assignment is
    # positional and never goes through index alignment. These values were
    # computed from these very rows, in this order, so alignment could only
    # ever be a no-op or a bug — and a merged atlas's obs index is routinely
    # duplicated (10x over on the CELLxGENE-derived file this work was
    # validated against), where pandas only tolerates alignment while the two
    # indexes compare equal.
    columns = {}
    if method == "mean_rank":
        # Rank within each tool's own covered cells, so a tool that scored
        # 3,000 cells and one that scored 300,000 contribute on the same [0, 1]
        # scale rather than the larger run dominating the average.
        ranks = _np.full((len(keys), n), _np.nan, dtype="float64")
        for i, k in enumerate(keys):
            s = _pd.Series(_consensus_scores(obs, f"{k}_score"))
            ranks[i] = s.rank(pct=True, method="average").to_numpy(dtype="float64")
        # Mean over the tools that scored each cell. Done by hand rather than
        # with nanmean because an all-NaN row — a cell no tool scored — is
        # expected here, and is the "no vote" case rather than a warning.
        covered_count = (~_np.isnan(ranks)).sum(axis=0)
        assessed = covered_count > 0
        mean_rank = _np.where(
            assessed,
            _np.nansum(ranks, axis=0) / _np.maximum(covered_count, 1),
            _np.nan,
        )
        if assessed.any():
            threshold = float(
                _np.quantile(mean_rank[assessed], 1.0 - float(quantile))
            )
            predicted = assessed & (mean_rank >= threshold)
        else:
            threshold = None
            predicted = _np.zeros(n, dtype=bool)
        columns[f"{key_added}_score"] = mean_rank.astype("float32")
        record["quantile"] = float(quantile)
        record["threshold"] = threshold
    elif method == "any":
        predicted = n_calling >= 1
    elif method == "all":
        predicted = (n_calling == n_voting) & voted_any
    else:  # majority — strictly more than half, so a tie is not a majority
        predicted = (n_calling * 2) > n_voting

    # The single line the whole design exists to make possible: a row nothing
    # voted on is null, not False.
    call = _pd.array(_np.asarray(predicted, dtype=bool), dtype="boolean")
    call[~voted_any] = _pd.NA
    columns[f"{key_added}_predicted"] = call
    columns[f"{key_added}_n_tools_calling"] = n_calling
    columns[f"{key_added}_n_tools_voting"] = n_voting

    existing = [c for c in columns if c in obs.columns]
    if existing and not overwrite:
        raise ValueError(
            f"obs already has {existing}; pass overwrite=True to replace them, "
            "or a different key_added to keep both."
        )

    record["n_predicted_doublet"] = int((predicted & voted_any).sum())
    record["n_predicted_singlet"] = int((~predicted & voted_any).sum())
    record["n_no_vote"] = int((~voted_any).sum())
    record["columns_added"] = list(columns)

    if is_adata:
        for name, values in columns.items():
            target.obs[name] = values
        target.uns[f"{key_added}_consensus"] = record
        return record

    for name, values in columns.items():
        obs[name] = values
    # `uns` was already read (and defaulted to {}) before key resolution, and
    # nothing has mutated the file since, so reuse it rather than re-reading.
    uns[f"{key_added}_consensus"] = record
    # obs and uns in ONE commit, so `pyscx.rollback` undoes the whole thing.
    # Two calls would leave a file that had been half-rolled-back.
    modify_metadata(path, obs=obs, uns=uns,
                    index_obs=index_obs, index_preset=index_preset)
    return record


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
        csc: "off", "auto", or "always" (column-major sidecar). When
            omitted, an accel-ready index_preset ("training" / "perturbseq")
            upgrades the default to "auto"; otherwise "off". An explicit
            value always wins. (Streaming h5mu cannot build per-modality
            CSC: "auto" degrades to no-CSC with a warning.)
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
    src = _coerce_path(path)
    _warn_if_deletions(src, "h5mu")
    return _to_h5mu_native(src, _coerce_path(out), **kwargs)


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
