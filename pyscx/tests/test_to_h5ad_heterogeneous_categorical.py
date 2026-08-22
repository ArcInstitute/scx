"""Streaming-export reconciliation of heterogeneous dict/plain categoricals
(SCX-STREAMING-EXPORT-DICT-RECONCILE).

PR #279 fixed the **read** path so a sharded obs axis whose categorical
columns mix ``Dictionary`` (original ``from_anndata`` shards) and plain
(appended shards) reassembles correctly. This module covers the **export**
follow-up: streaming `to_h5ad` / `to_h5mu` must fold the plain shards back
into a single h5ad categorical, matching ``to_anndata()``.

These started life as Phase-0 failing-first repros; after Phase 2 they are
green regression tests covering both **string** and **numeric** categoricals
(the numeric case is the one a string-only fix would silently miss — spec
§1.2.1).

**Phase-0 finding (kept for the record):** the spec originally claimed
``to_h5ad(stream=False)`` was a working workaround. It is not, for a
*sharded* obs axis: both ``write_scx_to_h5ad`` (stream=False) and
``write_scx_to_h5ad_streaming`` route obs through the same shard-stream writer
(``write_obs_streaming_or_eager`` only takes the eager ``read_obs`` branch when
``obs_metadata_shard_count() == 0``). So *both* flags were
broken before the fix and *both* are exercised here as a regression guard;
``to_anndata()`` is the authoritative read baseline. Category code **order** is
best-effort (spec §6) — equality is asserted **by value**, not by code.
"""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pandas.testing as pdt
import scipy.sparse as sp

import pyscx

# A ``shard_size`` smaller than ``n_obs`` forces the obs axis to be
# row-sharded (>=2 ``ObsMetadataShard`` sections). ``from_anndata`` writes
# those shards' categoricals as ``Dictionary``; a later ``append_from_anndata``
# writes its obs shard with plain (decoded) columns -> heterogeneous layout.
_N_OBS = 64
_N_VARS = 32
_SHARD = 32


def _base_x(n_obs: int, n_vars: int) -> sp.csr_matrix:
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 50, size=(n_obs, n_vars), dtype=np.int32).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.4] = 0.0
    return sp.csr_matrix(dense)


def _mk_adata_celltypes(
    n_obs: int, n_vars: int, perturbation: str, celltypes: list[str]
) -> anndata.AnnData:
    """AnnData with a **string** categorical ``cell_type`` drawn from a
    caller-supplied (possibly disjoint) vocabulary. Mirrors the helper in
    ``test_predicate_index_rewrite.py`` that stresses the append path's
    dictionary widen/unify/reconcile."""
    obs = pd.DataFrame(
        {
            "cell_id": [f"{perturbation}_{i}" for i in range(n_obs)],
            "cell_type": pd.Categorical(
                [celltypes[i % len(celltypes)] for i in range(n_obs)]
            ),
        },
        index=[f"{perturbation}_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=_base_x(n_obs, n_vars), obs=obs, var=var)


def _mk_adata_int_cat(
    n_obs: int, n_vars: int, perturbation: str, labels: list[int]
) -> anndata.AnnData:
    """AnnData with an **integer** categorical ``cluster`` (e.g. Leiden
    cluster labels). ``from_anndata`` stores it as ``Dictionary(_, Int64)``;
    ``append`` decodes it to a plain ``Int64`` shard -> the §1.2.1 numeric
    heterogeneous layout."""
    obs = pd.DataFrame(
        {
            "cell_id": [f"{perturbation}_{i}" for i in range(n_obs)],
            "cluster": pd.Categorical(
                [labels[i % len(labels)] for i in range(n_obs)]
            ),
        },
        index=[f"{perturbation}_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=_base_x(n_obs, n_vars), obs=obs, var=var)


def _mk_adata_float_cat(
    n_obs: int, n_vars: int, perturbation: str, labels: list[float]
) -> anndata.AnnData:
    """AnnData with a **float** categorical ``dose`` (e.g. dose levels).
    ``from_anndata`` stores it as ``Dictionary(_, Float64)``; ``append``
    decodes it to a plain ``Float64`` shard -> the §1.2.1 float heterogeneous
    layout (the 3.9 Float64 variant)."""
    obs = pd.DataFrame(
        {
            "cell_id": [f"{perturbation}_{i}" for i in range(n_obs)],
            "dose": pd.Categorical([labels[i % len(labels)] for i in range(n_obs)]),
        },
        index=[f"{perturbation}_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=_base_x(n_obs, n_vars), obs=obs, var=var)


def _build_heterogeneous(scx_path: str, base: anndata.AnnData, appended: anndata.AnnData):
    """from_anndata(base, sharded) then append_from_anndata(appended) ->
    a single-modality SCX whose obs is row-sharded with a categorical that
    is ``Dictionary`` in the base shards and plain in the appended shard.
    Returns the opened Experiment after asserting the sharding precondition."""
    pyscx.from_anndata(base, scx_path, shard_size=_SHARD)
    pre = pyscx.open(scx_path)
    assert pre.obs_metadata_shard_count >= 2, (
        "precondition: base obs must be row-sharded for the heterogeneous "
        f"repro (got {pre.obs_metadata_shard_count} shards)"
    )
    pyscx.append_from_anndata(scx_path, appended, shard_size=_SHARD)
    return pyscx.open(scx_path)


def _assert_export_matches_read(scx, exp, col, tmp_dir, tag, expected_union):
    """Both ``stream=True`` and ``stream=False`` export ``col`` as an h5ad
    categorical with the full unioned vocabulary, and each agrees per-row with
    ``to_anndata()``. Equality is by value (category code order is best-effort,
    spec §6), so we compare category *sets* and per-row string values."""
    ref = exp.to_anndata().obs[col]
    ref_vals = ref.astype(str).reset_index(drop=True)
    assert set(map(str, pd.unique(ref))) == expected_union

    for stream in (True, False):
        out_path = str(tmp_dir / f"{tag}_{'stream' if stream else 'mat'}.h5ad")
        pyscx.to_h5ad(scx, out_path, stream=stream)
        got = anndata.read_h5ad(out_path).obs[col]

        assert isinstance(got.dtype, pd.CategoricalDtype), (
            f"{tag} stream={stream}: column '{col}' should export as categorical, "
            f"got {got.dtype}"
        )
        assert set(map(str, got.cat.categories)) == expected_union, (
            f"{tag} stream={stream}: category set mismatch"
        )
        pdt.assert_series_equal(
            got.astype(str).reset_index(drop=True),
            ref_vals,
            check_names=False,
        )


# --------------------------------------------------------------------------
# string categorical (spec §3.2 / §3.3)
# --------------------------------------------------------------------------
def test_streaming_export_string_categorical_heterogeneous_roundtrips(tmp_dir):
    """A string categorical that is Dictionary(base)+plain(appended) exports
    (stream and materialising) as one categorical with the unioned vocab,
    matching ``to_anndata()``."""
    scx = str(tmp_dir / "het_str.scx")
    base = _mk_adata_celltypes(_N_OBS, _N_VARS, "DRUG_A", ["fibroblast", "epithelial"])
    appended = _mk_adata_celltypes(_N_OBS, _N_VARS, "DRUG_B", ["neuron", "astrocyte"])
    exp = _build_heterogeneous(scx, base, appended)
    assert exp.n_obs == 2 * _N_OBS

    _assert_export_matches_read(
        scx,
        exp,
        "cell_type",
        tmp_dir,
        "het_str",
        {"fibroblast", "epithelial", "neuron", "astrocyte"},
    )


# --------------------------------------------------------------------------
# numeric categorical (§1.2.1) — the case a string-only fix would miss
# --------------------------------------------------------------------------
def test_streaming_export_int_categorical_heterogeneous_roundtrips(tmp_dir):
    """An integer categorical that is Dictionary(_, Int64)(base)+plain
    Int64(appended) exports as one categorical with the unioned vocab
    {0,1,2,3}, matching ``to_anndata()``."""
    scx = str(tmp_dir / "het_int.scx")
    base = _mk_adata_int_cat(_N_OBS, _N_VARS, "DRUG_A", [0, 1])
    appended = _mk_adata_int_cat(_N_OBS, _N_VARS, "DRUG_B", [2, 3])
    exp = _build_heterogeneous(scx, base, appended)
    assert exp.n_obs == 2 * _N_OBS

    _assert_export_matches_read(
        scx,
        exp,
        "cluster",
        tmp_dir,
        "het_int",
        {"0", "1", "2", "3"},
    )


def test_streaming_export_float_categorical_heterogeneous_roundtrips(tmp_dir):
    """3.9 Float64 variant: a float categorical that is Dictionary(_, Float64)
    (base)+plain Float64(appended) exports as one categorical with the unioned
    vocab {0.5,1.5,2.5,3.5}, matching ``to_anndata()``."""
    scx = str(tmp_dir / "het_float.scx")
    base = _mk_adata_float_cat(_N_OBS, _N_VARS, "DRUG_A", [0.5, 1.5])
    appended = _mk_adata_float_cat(_N_OBS, _N_VARS, "DRUG_B", [2.5, 3.5])
    exp = _build_heterogeneous(scx, base, appended)
    assert exp.n_obs == 2 * _N_OBS

    _assert_export_matches_read(
        scx,
        exp,
        "dose",
        tmp_dir,
        "het_float",
        {"0.5", "1.5", "2.5", "3.5"},
    )
