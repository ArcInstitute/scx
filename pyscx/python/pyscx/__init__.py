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


def obs_import(path, table, *, key=None, **kwargs):
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
        key: Join key. None auto-resolves with the same preference order the
            target side uses (pandas index, then `barcode`/`cell_id`/…). A str
            names one column. A list of str builds a composite key — the right
            answer for a multi-library merge where `sample_id` + `barcode` is
            unique but neither is alone. Both sides are built by the same code,
            so the fusing separator is internal and not configurable.
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
        on_missing_rows: "zero" (default) marks uncovered target rows absent;
            "error" refuses.
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
    if key is not None:
        key = [key] if isinstance(key, str) else [str(k) for k in key]
    return _obs_import_native(_coerce_path(path), _coerce_path(table, allow_experiment=False),
                              key=key, **kwargs)


def diagnose_obs_key(path, key=None):
    """Report which obs columns could serve as an `obs_import` join key.

    Read-only. Reach for this when an import fails on a duplicated key: on a
    merged atlas the obvious candidates are often not unique, and the column
    that is may be one no fallback list would guess (on a CELLxGENE-derived
    file it is `soma_joinid`, with the obs index 10x-duplicated).

    Returns a dict with `resolved_key`, `resolved_cardinality`,
    `unique_columns`, `unique_pairs`, `suggestion` and a printable `summary`.
    """
    if key is not None:
        key = [key] if isinstance(key, str) else [str(k) for k in key]
    return _diagnose_obs_key_native(_coerce_path(path), key)


def doublet_import(path, table, *, tool, key=None, **kwargs):
    """Import a doublet caller's output, normalised to canonical obs columns.

    The doublet-specific wrapper over `obs_import`. Same in-place, key-joined,
    `pyscx.rollback`-able import — plus the one thing that needs per-tool
    knowledge: every caller names its score and call differently, and consensus
    code downstream should not have to branch on which tool ran.

    Writes, for `key_added="<K>"` (default: the tool name):

        obs["<K>_score"]      f32,  nullable   higher = more doublet-like
        obs["<K>_predicted"]  bool, nullable   omitted when the tool has no call
        obs["<K>_status"]     str              "present" / "absent"
        obs["<K>_<native>"]   ...              every other source column
        uns["<K>"]                             tool, source columns, join report

    These names match what a native SCX doublet run writes, so an imported
    result and a native one are drop-in comparable.

    Args:
        path: Target SCX file (str, os.PathLike, or an open Experiment).
        table: The caller's output .csv / .tsv. An `.h5ad` source is not
            supported yet — write `adata.obs[[...]].to_csv(...)` and import that.
        tool: One of `pyscx.doublet_tools()`: "scdblfinder", "scrublet",
            "doubletfinder", "doubletdetection", "solo", "scds", "generic".
        key: Join key, exactly as for `obs_import`. None auto-resolves; a str
            names one column; a list builds a composite — the right answer for a
            multi-library merge where `sample_id` + `barcode` is unique but
            neither is alone.
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
        on_missing_rows: "zero" (default) marks uncovered cells absent; "error"
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
    if key is not None:
        key = [key] if isinstance(key, str) else [str(k) for k in key]
    return _doublet_import_native(_coerce_path(path),
                                  _coerce_path(table, allow_experiment=False),
                                  tool=tool, key=key, **kwargs)


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
