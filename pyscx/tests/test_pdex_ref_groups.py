"""`pdex_ref(groups=)` restricts the tested targets (REC-10, PR G).

A target is only ever compared with the reference — Mann-Whitney U,
pseudobulk fold change, per-target CPM filter, per-target BH — so dropping the
other targets before the kernel cannot change a selected target's rows. These
tests pin exactly that: the restricted frame equals the full frame filtered
to the requested targets, in the requested order, on every route this host can
run. Base-install safe: imports `_pdex_fixtures` only (no pdex, no polars).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pyscx
from _pdex_fixtures import REFERENCE, _make_adata

pytestmark = pytest.mark.filterwarnings("error::UserWarning")


def _run(adata, **kw):
    kw.setdefault("reference", REFERENCE)
    kw.setdefault("is_log1p", False)
    kw.setdefault("device", "cpu")
    return pyscx.accel.pdex_ref(adata, "target", **kw)


def _filtered(full: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    parts = [full[full["target"] == t] for t in targets]
    return pd.concat(parts).reset_index(drop=True)


def test_groups_equals_the_full_frame_filtered_in_request_order():
    adata = _make_adata()
    full = _run(adata)
    assert list(dict.fromkeys(full["target"])) == ["ko_a", "ko_b"]

    one = _run(adata, groups=["ko_b"])
    pd.testing.assert_frame_equal(one, _filtered(full, ["ko_b"]))

    both = _run(adata, groups=["ko_b", "ko_a"])
    assert list(dict.fromkeys(both["target"])) == ["ko_b", "ko_a"]
    pd.testing.assert_frame_equal(both, _filtered(full, ["ko_b", "ko_a"]))


def test_groups_with_cpm_filter_equals_the_filtered_full_frame():
    adata = _make_adata()
    full = _run(adata, cpm_filter=50.0)
    sub = _run(adata, cpm_filter=50.0, groups=["ko_a"])
    pd.testing.assert_frame_equal(sub, _filtered(full, ["ko_a"]))


def test_groups_on_backed_csr_and_csc_direct_routes(tmp_path):
    adata = _make_adata()
    full = _run(adata)
    path = str(tmp_path / "pdex.scx")
    pyscx.from_anndata(adata, path, csc="always", csc_cols_per_shard=4)

    backed = pyscx.open(path).to_anndata(backed=True)
    sub = _run(backed, groups=["ko_b"], prefer_format="csr")
    assert backed.uns["scx_accel"]["pdex_ref"]["route"] == "cpu_csr"
    pd.testing.assert_frame_equal(sub, _filtered(full, ["ko_b"]))

    csc = pyscx.open(path).to_anndata(backed=True)
    sub = _run(csc, groups=["ko_b"], prefer_format="csc")
    assert csc.uns["scx_accel"]["pdex_ref"]["route"] == "cpu_csc"
    pd.testing.assert_frame_equal(sub, _filtered(full, ["ko_b"]))


def test_excluded_targets_do_not_trip_the_unlabelled_cell_warning():
    """The unselected target's cells are dropped by relabelling, and that must
    not be reported as 'cells with no group label' — the module-level
    `error::UserWarning` filter turns any such warning into a failure."""
    _run(_make_adata(), groups=["ko_a"])


@pytest.mark.parametrize("na", [np.nan, None, pd.NA], ids=["np.nan", "None", "pd.NA"])
def test_a_genuinely_unlabelled_cell_still_warns_under_groups(na):
    """Every pandas missing value counts, not just the one that prints "nan".

    `astype("str")` renders `np.nan` as `"nan"`, `None` as `"None"` and `pd.NA`
    as `"<NA>"`. While missingness was read off that string, only the first was
    unlabelled here: the other two became levels of their own on a plain
    (non-categorical) column, i.e. phantom `pdex_ref` targets with no cells and
    no warning. `pandas.isna` decides it now.
    """
    adata = _make_adata()
    labels = adata.obs["target"].astype(object).to_numpy()
    labels[0] = na
    adata.obs["target"] = labels
    with pytest.warns(UserWarning, match="1 of 90 cells have no group label"):
        _run(adata, groups=["ko_a"])


def test_a_target_named_like_a_missing_value_is_a_real_target():
    """The other direction: `pdex_ref` must not steal a perturbation called
    `"nan"`. It is a legitimate label, and only `pandas.isna` can tell it from a
    genuine NaN."""
    adata = _make_adata()
    labels = adata.obs["target"].astype(object).to_numpy()
    labels[labels == "ko_b"] = "nan"
    adata.obs["target"] = labels
    df = _run(adata, groups=["nan"])
    assert set(df["target"]) == {"nan"}


def test_groups_errors():
    adata = _make_adata()
    with pytest.raises(ValueError, match=r'"ko_z" is not a level.*available: \["ko_a", "ko_b", "non-targeting"\]'):
        _run(adata, groups=["ko_a", "ko_z"])
    with pytest.raises(ValueError, match="more than once"):
        _run(adata, groups=["ko_a", "ko_a"])
    with pytest.raises(ValueError, match="at least one group"):
        _run(adata, groups=[])
    with pytest.raises(ValueError, match="is the reference group"):
        _run(adata, groups=["ko_a", REFERENCE])
    with pytest.raises(TypeError):
        _run(adata, groups="ko_a")


def test_route_metadata_is_unchanged_by_groups():
    adata = _make_adata()
    _run(adata, groups=["ko_a"])
    entry = adata.uns["scx_accel"]["pdex_ref"]
    assert entry["route"] == "cpu_dense"
    assert "groups" not in entry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert entry["use_raw"] is False
