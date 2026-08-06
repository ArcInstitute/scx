"""`pdex_ref`'s `is_log1p=None` auto-detection must not depend on X's layout.

`pdex_ref` picks its `GeomMeanMode` from whether the input is log1p-transformed:
log-space input has its means back-transformed with `expm1`, raw counts do not.
Get that wrong and `target_mean` / `ref_mean` / `log2_fold_change` are off by
orders of magnitude — `expm1(8) = 2980` against `8` — with no error and no
warning.

The probe used to answer `False` for any backed dataset ("touching X here would
force a load"), so a backed handle and an in-memory AnnData over byte-identical
data disagreed. These tests pin the three ways the question is now answered —
the `uns` annotation, a lazy transform chain, the catalog's integer value_max —
and the one case where it is refused instead of guessed.

Every comparison pins `device="cpu"` on both arms: on a GPU host an in-memory X
routes to rapids while a backed X stays native, and the test would be measuring
that instead.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx

GROUPBY = "target"
REFERENCE = "control"


def _counts_adata(n_obs: int = 60, n_vars: int = 12, seed: int = 3) -> ad.AnnData:
    """Raw-count AnnData with two groups and a deliberate per-group shift.

    Counts reach well past the heuristic's `< 30` threshold on purpose, so
    "these are raw counts" is the answer both the in-memory heuristic and the
    catalog's `value_max` must produce. A fixture whose max lands just under 30
    would make `is_log1p=True` correct and hide the very disagreement these
    tests are for — see `test_backed_low_max_counts_agree_with_in_memory` for
    the other side of the threshold.
    """
    rng = np.random.default_rng(seed)
    per_group = n_obs // 2
    base = rng.uniform(20.0, 60.0, size=(2, n_vars))
    base[1, : n_vars // 2] *= 3.0
    counts = np.vstack(
        [
            rng.poisson(base[0], size=(per_group, n_vars)),
            rng.poisson(base[1], size=(n_obs - per_group, n_vars)),
        ]
    ).astype(np.float32)
    obs = pd.DataFrame(
        {GROUPBY: [REFERENCE] * per_group + ["pert"] * (n_obs - per_group)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    assert adata.X.data.max() > 30.0, "fixture must read as raw counts"
    return adata


def _assert_close(a, b, what: str) -> None:
    for col in ("target_mean", "ref_mean", "log2_fold_change"):
        np.testing.assert_allclose(
            np.asarray(a[col], dtype=np.float64),
            np.asarray(b[col], dtype=np.float64),
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"{col} diverged: {what}",
        )


def test_backed_and_in_memory_agree_after_accel_log1p(tmp_path):
    """The headline case: `open(...) → accel.log1p → pdex_ref` must equal the
    in-memory `sc.pp.log1p` pipeline.

    The lazy log1p left no `uns["log1p"]`, and the probe refused to look at a
    backed X, so this pair silently took different mean modes. It now agrees on
    two independent grounds — the chain records the transform and the arm stamps
    the annotation.
    """
    sc = pytest.importorskip("scanpy")

    adata = _counts_adata()
    path = str(tmp_path / "counts.scx")
    pyscx.from_anndata(adata, path)

    backed = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.log1p(backed)

    in_memory = adata.copy()
    sc.pp.log1p(in_memory)

    backed_df = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, device="cpu"
    )
    memory_df = pyscx.accel.pdex_ref(
        in_memory, GROUPBY, reference=REFERENCE, device="cpu"
    )
    _assert_close(backed_df, memory_df, "backed lazy-log1p vs in-memory sc.pp.log1p")


def test_lazy_chain_is_detected_without_the_uns_stamp(tmp_path):
    """The transform chain alone is enough — the probe does not lean on `uns`.

    Deleting the stamp models a chain built by a caller that bypassed
    `accel.log1p`'s annotation, and previously hit
    `TypeError: max() got an unexpected keyword argument 'out'` because numpy
    was asked to reduce over a lazy dataset.
    """
    adata = _counts_adata()
    path = str(tmp_path / "counts.scx")
    pyscx.from_anndata(adata, path)

    backed = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.log1p(backed)
    del backed.uns["log1p"]

    stamped = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.log1p(stamped)

    unstamped_df = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, device="cpu"
    )
    stamped_df = pyscx.accel.pdex_ref(
        stamped, GROUPBY, reference=REFERENCE, device="cpu"
    )
    _assert_close(unstamped_df, stamped_df, "lazy chain without vs with uns stamp")

    # And it really is the log1p mode, not raw: forcing raw must differ.
    raw_mode_df = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, is_log1p=False, device="cpu"
    )
    assert not np.allclose(
        np.asarray(unstamped_df["target_mean"], dtype=np.float64),
        np.asarray(raw_mode_df["target_mean"], dtype=np.float64),
    ), "is_log1p detection is not actually changing the mean mode"


def test_backed_counts_agree_with_in_memory_counts(tmp_path):
    """Integer counts: the catalog's `value_max` answers exactly, and the
    backed answer matches the in-memory heuristic's."""
    adata = _counts_adata()
    path = str(tmp_path / "counts.scx")
    pyscx.from_anndata(adata, path)
    backed = pyscx.open(path).to_anndata(backed=True)

    backed_df = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, device="cpu"
    )
    memory_df = pyscx.accel.pdex_ref(
        adata.copy(), GROUPBY, reference=REFERENCE, device="cpu"
    )
    _assert_close(backed_df, memory_df, "backed raw counts vs in-memory raw counts")

    # The heuristic's own answer for counts data reaching well past 30.
    forced_raw = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, is_log1p=False, device="cpu"
    )
    _assert_close(backed_df, forced_raw, "auto-detected vs is_log1p=False")


def test_backed_low_max_counts_agree_with_in_memory(tmp_path):
    """Exercise the branch where the data *does* look log1p on a backed file.

    The backed arm must reach `is_log1p=True` from the catalog exactly where the
    in-memory arm reaches it from `max(X)`. This is the side of the threshold the
    pre-existing backed-vs-in-memory comparisons never touched: their fixture
    lands at a max of 30–31, so the in-memory heuristic returned `False` and
    matched the hard-coded backed `False` by luck rather than by agreement.
    """
    adata = _counts_adata()
    # Squash the range so every stored value is below the threshold.
    x = adata.X.tocsr()
    x.data = np.minimum(x.data, 5.0).astype(np.float32)
    adata.X = x
    assert adata.X.data.max() < 30.0

    path = str(tmp_path / "low.scx")
    pyscx.from_anndata(adata, path)
    backed = pyscx.open(path).to_anndata(backed=True)

    backed_df = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, device="cpu"
    )
    memory_df = pyscx.accel.pdex_ref(
        adata.copy(), GROUPBY, reference=REFERENCE, device="cpu"
    )
    _assert_close(backed_df, memory_df, "backed low-max vs in-memory low-max")


def test_backed_float_file_refuses_to_guess(tmp_path):
    """Float-encoded shards write `value_max = 0` by design, so the catalog can
    bound nothing. Refuse with an actionable message rather than silently
    answering `False` — the behaviour that made backed and in-memory disagree.
    """
    adata = _counts_adata()
    x = adata.X.tocsr()
    x.data = (x.data + 0.5).astype(np.float32)  # fractional → float encoding
    adata.X = x

    path = str(tmp_path / "float.scx")
    pyscx.from_anndata(adata, path)
    backed = pyscx.open(path).to_anndata(backed=True)

    with pytest.raises(ValueError, match="is_log1p"):
        pyscx.accel.pdex_ref(backed, GROUPBY, reference=REFERENCE, device="cpu")

    # Explicit answers are honoured, and they are not the same computation.
    as_log = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, is_log1p=True, device="cpu"
    )
    as_raw = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, is_log1p=False, device="cpu"
    )
    assert not np.allclose(
        np.asarray(as_log["target_mean"], dtype=np.float64),
        np.asarray(as_raw["target_mean"], dtype=np.float64),
    )


def test_uns_annotation_still_wins_on_a_float_file(tmp_path):
    """An annotated float file needs no value probe at all."""
    adata = _counts_adata()
    x = adata.X.tocsr()
    x.data = np.log1p(x.data).astype(np.float32)
    adata.X = x
    adata.uns["log1p"] = {"base": None}

    path = str(tmp_path / "logged.scx")
    pyscx.from_anndata(adata, path)
    backed = pyscx.open(path).to_anndata(backed=True)
    assert "log1p" in backed.uns

    backed_df = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, device="cpu"
    )
    forced = pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, is_log1p=True, device="cpu"
    )
    _assert_close(backed_df, forced, "uns-annotated float file vs is_log1p=True")


def test_probe_reads_the_selected_matrix_not_adata_x(tmp_path):
    """`use_raw=` / `layer=` select a different matrix; the probe must follow.

    Here `X` is log-space and `.raw` holds the counts, so probing `adata.X`
    would answer the question about the wrong matrix.
    """
    counts = _counts_adata()
    adata = counts.copy()
    adata.raw = counts
    x = adata.X.tocsr()
    x.data = np.log1p(x.data).astype(np.float32)
    adata.X = x

    auto = pyscx.accel.pdex_ref(
        adata, GROUPBY, reference=REFERENCE, use_raw=True, device="cpu"
    )
    as_raw_counts = pyscx.accel.pdex_ref(
        adata,
        GROUPBY,
        reference=REFERENCE,
        use_raw=True,
        is_log1p=False,
        device="cpu",
    )
    _assert_close(auto, as_raw_counts, "use_raw probe followed adata.X instead of .raw")


def test_rescaling_chain_without_log1p_refuses_to_guess(tmp_path):
    """A `normalize_total` chain detaches the values from the catalog.

    `normalize_total(target_sum=1e4)` over small counts produces values nowhere
    near what the shards recorded, so reading `value_max` would apply the
    heuristic to the wrong numbers — and to different numbers than the
    in-memory arm sees. The chain proves the data is not log1p, but the probe
    says so by refusing and naming the chain rather than by reading stale stats.
    """
    adata = _counts_adata()
    path = str(tmp_path / "counts.scx")
    pyscx.from_anndata(adata, path)

    backed = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.normalize_total(backed, target_sum=1e4)

    with pytest.raises(ValueError, match="normalize_total"):
        pyscx.accel.pdex_ref(backed, GROUPBY, reference=REFERENCE, device="cpu")

    # Explicit is fine, and log1p on top of the chain resolves it without one.
    pyscx.accel.pdex_ref(
        backed, GROUPBY, reference=REFERENCE, is_log1p=False, device="cpu"
    )
    pyscx.accel.log1p(backed)
    pyscx.accel.pdex_ref(backed, GROUPBY, reference=REFERENCE, device="cpu")


def test_empty_backed_matrix_refuses_rather_than_claiming_log1p(tmp_path):
    """An empty matrix is not evidence of anything, and must not read as log1p.

    The catalog can legitimately prove `value_max == 0` when `nnz == 0` — but
    "the maximum is 0" is a fact about an absent value set, not a measurement of
    one, and letting it flow into the `< 30` heuristic would answer
    `is_log1p=True` on no evidence. The in-memory arm raises here too (numpy
    cannot reduce an empty array), so refusing is what keeps the two layouts
    agreeing on this pathological input.
    """
    adata = _counts_adata()
    x = adata.X.tocsr()
    x.data[:] = 0.0
    x.eliminate_zeros()
    adata.X = x
    assert adata.X.nnz == 0

    path = str(tmp_path / "empty.scx")
    pyscx.from_anndata(adata, path)
    backed = pyscx.open(path).to_anndata(backed=True)

    with pytest.raises(ValueError, match="is_log1p"):
        pyscx.accel.pdex_ref(backed, GROUPBY, reference=REFERENCE, device="cpu")
