"""Tests for in-place metadata replacement: pyscx.set_uns / modify_metadata.

These wrap scx_ops::set_uns / modify_metadata — replace SCX metadata sections
(uns / obs / var / obsm / varm) without re-encoding X.
"""

import numpy as np
import pandas as pd
import pytest

import pyscx


def test_set_uns_round_trip(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)

    new_uns = {"method": "updated", "k": 7, "labels": ["a", "b", "c"]}
    pyscx.set_uns(path, new_uns)

    uns = pyscx.open(path).to_anndata().uns
    assert uns["method"] == "updated"
    assert int(uns["k"]) == 7
    assert list(uns["labels"]) == ["a", "b", "c"]
    # Matrix untouched — file still opens and has the same shape.
    exp = pyscx.open(path)
    assert exp.n_obs == synthetic_adata.n_obs
    assert exp.n_vars == synthetic_adata.n_vars


def test_modify_metadata_obs_replace(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n = synthetic_adata.n_obs

    new_obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n)],
            "donor": ["donor_Z"] * n,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    pyscx.modify_metadata(path, obs=new_obs)

    obs = pyscx.open(path).to_anndata().obs
    assert len(obs) == n
    assert "donor" in obs.columns
    assert obs["donor"].iloc[0] == "donor_Z"


def test_modify_metadata_obs_wrong_shape_raises(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n = synthetic_adata.n_obs

    bad_obs = pd.DataFrame({"x": list(range(n - 1))})  # one row short
    with pytest.raises(ValueError):
        pyscx.modify_metadata(path, obs=bad_obs)


def test_modify_metadata_obs_dict_raises_typeerror(synthetic_adata, scx_from_adata):
    # Report E2: passing a column->values dict (a natural thing to try) instead
    # of a pandas DataFrame must raise a clear TypeError naming modify_metadata,
    # the parameter, and the fix — not an opaque pyarrow AttributeError.
    path = scx_from_adata(synthetic_adata)
    with pytest.raises(TypeError) as ei:
        pyscx.modify_metadata(path, obs={"qc_status": [1, 2, 3]})
    msg = str(ei.value)
    assert "modify_metadata(obs=...)" in msg
    assert "pandas DataFrame" in msg
    assert "dict" in msg


def test_modify_metadata_var_dict_raises_typeerror(synthetic_adata, scx_from_adata):
    # Same guard on the `var` parameter.
    path = scx_from_adata(synthetic_adata)
    with pytest.raises(TypeError) as ei:
        pyscx.modify_metadata(path, var={"gene_flag": [0, 1]})
    assert "modify_metadata(var=...)" in str(ei.value)


def test_modify_metadata_obs_index_rebuild(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n = synthetic_adata.n_obs

    new_obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n)],
            "batch": pd.Categorical(np.random.choice(["A", "B"], size=n)),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    # Request a predicate index over `batch` while replacing obs.
    pyscx.modify_metadata(path, obs=new_obs, index_obs=["batch"])

    # File reopens and the new obs is present.
    obs = pyscx.open(path).to_anndata().obs
    assert "batch" in obs.columns
    assert len(obs) == n


def test_modify_metadata_varm_replace(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n_vars = synthetic_adata.n_vars

    loadings = np.random.randn(n_vars, 4).astype(np.float32)
    pyscx.modify_metadata(path, varm={"PCs": loadings})

    adata = pyscx.open(path).to_anndata()
    assert "PCs" in adata.varm
    assert adata.varm["PCs"].shape == (n_vars, 4)


def test_set_uns_then_rollback(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    original = dict(pyscx.open(path).to_anndata().uns)

    pyscx.set_uns(path, {"state": "mutated"})
    assert pyscx.open(path).to_anndata().uns["state"] == "mutated"

    pyscx.rollback(path)
    restored = pyscx.open(path).to_anndata().uns
    assert "state" not in restored
    # Original keys are back.
    assert restored.get("species") == original.get("species")


def test_modify_metadata_empty_patch_raises(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    with pytest.raises(ValueError):
        pyscx.modify_metadata(path)


# ---------------------------------------------------------------------------
# The pandas index envelope must survive an in-place obs write
# ---------------------------------------------------------------------------
#
# `unify_dict_columns` (scx-ops) rebuilt the obs schema with `Schema::new(...)`,
# which dropped the schema-level `pandas` envelope naming `index_columns`. The
# h5ad exporter then fell back to "field 0 is the index" — true of a
# CLI-converted file, false after an obs rewrite — so **every exported cell was
# renamed to the value of the first string column**, silently.
#
# It only fired when a categorical obs column was present (a batch with no
# dictionary column takes an early return), which is why it looked intermittent:
# a file with no categoricals was fine, and so was any *second* mutation, the
# first having already decoded every categorical to plain strings.


@pytest.fixture()
def categorical_obs_scx(tmp_path):
    """An SCX file whose obs has a categorical column BEFORE the index field.

    Both details matter. The categorical is what sends the batch down the
    dictionary-rebuilding path, and the index not being field 0 is what makes
    the exporter's fallback pick the wrong column.
    """
    anndata = pytest.importorskip("anndata")
    sparse = pytest.importorskip("scipy.sparse")

    n, g = 12, 5
    X = sparse.csr_matrix(np.arange(n * g, dtype=np.float32).reshape(n, g))
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(np.repeat(["T cell", "B cell", "NK"], 4)),
            "n_counts": np.arange(n, dtype=np.float64),
        },
        index=[f"AAACCT-{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(g)])
    src = tmp_path / "src.h5ad"
    anndata.AnnData(X=X, obs=obs, var=var).write_h5ad(src)

    path = str(tmp_path / "f.scx")
    pyscx.from_h5ad(str(src), path)
    return path, list(obs.index)


def _exported_obs_names(scx_path, tmp_path, name):
    anndata = pytest.importorskip("anndata")
    out = tmp_path / name
    pyscx.to_h5ad(scx_path, str(out))
    return list(anndata.read_h5ad(out).obs_names)


def test_modify_metadata_then_export_keeps_obs_names(categorical_obs_scx, tmp_path):
    """The reported bug, end to end. No column is even added."""
    path, expected = categorical_obs_scx

    obs = pyscx.open(path).read_obs()
    pyscx.modify_metadata(path, obs=obs)

    assert _exported_obs_names(path, tmp_path, "out.h5ad") == expected


def test_append_then_export_keeps_obs_names(categorical_obs_scx, tmp_path):
    """`append` runs the same code over both the old and the new obs."""
    anndata = pytest.importorskip("anndata")
    sparse = pytest.importorskip("scipy.sparse")
    path, expected = categorical_obs_scx

    n, g = 4, 5
    extra = anndata.AnnData(
        X=sparse.csr_matrix(np.ones((n, g), dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "cell_type": pd.Categorical(["T cell"] * n),
                "n_counts": np.zeros(n, dtype=np.float64),
            },
            index=[f"EXTRA-{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(g)]),
    )
    extra_h5ad = tmp_path / "extra.h5ad"
    extra.write_h5ad(extra_h5ad)
    extra_scx = str(tmp_path / "extra.scx")
    pyscx.from_h5ad(str(extra_h5ad), extra_scx)   # append takes SCX, not h5ad
    pyscx.append(path, extra_scx)

    names = _exported_obs_names(path, tmp_path, "out.h5ad")
    assert names == expected + [f"EXTRA-{i}" for i in range(4)]


def test_obs_import_then_export_keeps_obs_names(categorical_obs_scx, tmp_path):
    """`attach_external_obs` shares the same helper, so `obs_import` /
    `doublet_import` / `cellbender_import` all inherited the bug."""
    path, expected = categorical_obs_scx

    table = tmp_path / "calls.csv"
    pd.DataFrame({
        "barcode": expected,
        "score": np.linspace(0, 1, len(expected)),
    }).to_csv(table, index=False)
    pyscx.obs_import(path, str(table))

    assert _exported_obs_names(path, tmp_path, "out.h5ad") == expected


def test_a_second_in_place_write_self_heals_the_envelope(categorical_obs_scx, tmp_path):
    """Two mutations in a row, which is the case that always worked.

    Verified against the pre-fix build: this one **passed** even then, and the
    reason is worth pinning. `doublet_import` drops the envelope, but it also
    decodes the categorical to plain strings; `doublet_consensus` then reads
    obs back through a path that re-stamps the envelope
    (`ensure_pandas_index_metadata`) and writes it out again — and with no
    dictionary column left there is nothing to trigger the schema rebuild, so
    the repair persists.

    That self-healing is exactly why the bug looked intermittent, and it is a
    real behaviour worth guarding: a future change to the read path's
    re-stamping would break this while leaving the single-mutation tests above
    green.
    """
    path, expected = categorical_obs_scx

    table = tmp_path / "scrub.csv"
    pd.DataFrame({
        "barcode": expected,
        "doublet_score": np.linspace(0, 1, len(expected)),
        "predicted_doublet": [True, False] * (len(expected) // 2),
    }).to_csv(table, index=False)
    pyscx.doublet_import(path, str(table), tool="scrublet")
    pyscx.doublet_consensus(path, keys=["scrublet"], method="any")

    assert _exported_obs_names(path, tmp_path, "out.h5ad") == expected


def test_the_categorical_stays_a_column_not_the_index(categorical_obs_scx, tmp_path):
    """The other half of the same assertion, from the failure's own direction:
    `cell_type` was being *promoted* to obs_names, so check it is still an
    ordinary column carrying its own values."""
    anndata = pytest.importorskip("anndata")
    path, _ = categorical_obs_scx

    pyscx.modify_metadata(path, obs=pyscx.open(path).read_obs())
    out = tmp_path / "out.h5ad"
    pyscx.to_h5ad(path, str(out))

    obs = anndata.read_h5ad(out).obs
    assert "cell_type" in obs.columns
    assert list(obs["cell_type"].astype(str))[:4] == ["T cell"] * 4
    assert "__index_level_0__" not in obs.columns


def test_projected_read_obs_keeps_its_index(categorical_obs_scx):
    """The second consumer of the envelope. `read_obs(columns=[...])` consults
    it to retain the index column in the projection; without it the returned
    frame silently loses its barcodes."""
    path, expected = categorical_obs_scx

    pyscx.modify_metadata(path, obs=pyscx.open(path).read_obs())

    projected = pyscx.open(path).read_obs(["n_counts"])
    assert list(projected.index) == expected
    assert list(projected.columns) == ["n_counts"]
