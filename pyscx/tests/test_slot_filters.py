"""`to_anndata` per-slot key filters: `obsp=` / `varp=` / `varm=`.

`layers=` and `obsm=` could already be narrowed; the other three aligned
slots could not, at any level. That matters because anndata decodes a whole
slot the moment the property is read — `AlignedMappingProperty.__get__` builds
an `AlignedActual`, whose `__init__` runs `_validate_value` over every entry —
so the first `adata.obsp` access pulls every obsp key off disk, however lazy
the individual bridges are. On a census-scale kNN graph that is an
`n_obs × n_obs` read the caller never asked for.

The contract mirrors `obsm=` exactly:

* `None` (default) — every key, byte-identical to before;
* `[]` — no keys, and **no bridge is constructed at all** (the existence gate
  is filter-aware, the same way `has_layers` already was), so nothing can be
  read even by accident;
* `["a"]` — that subset; an unknown key raises `KeyError`.

The `[]`-means-no-bridge choice is why the laziness assertions below run on a
*partial* filter rather than an empty one: a one-of-two filter proves the
bridge narrowed **and** stayed unmaterialised, which an empty bridge could not.
"""

import warnings

import numpy as np
import pytest

N_OBS, N_VARS = 24, 10


@pytest.fixture
def multi_key_path(tmp_dir):
    """An SCX file with **two** keys in each of the five aligned slots.

    Two per slot is load-bearing: a one-key slot cannot tell "the filter
    selected the right key" from "the filter was ignored".
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    rng = np.random.RandomState(11)
    x = (rng.random_sample((N_OBS, N_VARS)) * 10).astype(np.float32)
    x[rng.random_sample((N_OBS, N_VARS)) > 0.6] = 0.0

    adata = anndata.AnnData(
        X=sp.csr_matrix(x),
        obs=pd.DataFrame(
            {"group": ["a" if i % 2 else "b" for i in range(N_OBS)]},
            index=[f"cell_{i}" for i in range(N_OBS)],
        ),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(N_VARS)]),
        layers={
            "counts": sp.csr_matrix(x * 2),
            "spliced": sp.csr_matrix(x * 3),
        },
        obsm={
            "X_pca": rng.random_sample((N_OBS, 4)).astype(np.float32),
            "X_umap": rng.random_sample((N_OBS, 2)).astype(np.float32),
        },
        varm={
            "PCs": rng.random_sample((N_VARS, 4)).astype(np.float32),
            "loadings": rng.random_sample((N_VARS, 3)).astype(np.float32),
        },
        obsp={
            "connectivities": sp.csr_matrix(
                rng.random_sample((N_OBS, N_OBS)).astype(np.float32)
            ),
            "distances": sp.csr_matrix(
                rng.random_sample((N_OBS, N_OBS)).astype(np.float32)
            ),
        },
        varp={
            "corr": sp.csr_matrix(rng.random_sample((N_VARS, N_VARS)).astype(np.float32)),
            "cov": sp.csr_matrix(rng.random_sample((N_VARS, N_VARS)).astype(np.float32)),
        },
    )
    path = str(tmp_dir / "slots.scx")
    pyscx.from_anndata(adata, path)
    return path


# --------------------------------------------------------------------------
# Empty filter — the slot is gone, and so is its bridge.
# --------------------------------------------------------------------------


def test_obsp_empty_list_loads_nothing(multi_key_path):
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata(obsp=[])

    assert len(adata.obsp) == 0
    # No bridge at all — not an empty one. `_obsp` is anndata's own store.
    assert type(adata._obsp).__name__ != "ScxLazyPairwiseMapping"
    # And only obsp moved: the sibling slots are untouched.
    assert set(adata.varp) == {"corr", "cov"}
    assert set(adata.varm) == {"PCs", "loadings"}
    assert set(adata.layers) == {"counts", "spliced"}
    assert set(adata.obsm) == {"X_pca", "X_umap"}


def test_varp_and_varm_empty_lists_load_nothing(multi_key_path):
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata(varp=[], varm=[])

    assert len(adata.varp) == 0
    assert len(adata.varm) == 0
    assert type(adata._varp).__name__ != "ScxLazyPairwiseMapping"
    assert type(adata._varm).__name__ != "ScxLazyVarmMapping"
    assert set(adata.obsp) == {"connectivities", "distances"}


def test_every_slot_excluded_at_once(multi_key_path):
    """The shape arc-reactor actually asks for: X / obs / var / uns only."""
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata(
        layers=[], obsm=[], obsp=[], varp=[], varm=[]
    )

    assert len(adata.layers) == 0
    assert len(adata.obsm) == 0
    assert len(adata.obsp) == 0
    assert len(adata.varp) == 0
    assert len(adata.varm) == 0
    assert adata.shape == (N_OBS, N_VARS)
    assert adata.X.nnz > 0


def test_empty_filter_under_eager(multi_key_path):
    """`eager=True` takes a different branch — it materialises through the
    bridge into a plain dict instead of attaching it. The gate has to hold
    there too, or `obsp=[]` silently decodes everything."""
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata(eager=True, obsp=[], varm=[])

    assert len(adata.obsp) == 0
    assert len(adata.varm) == 0
    assert set(adata.varp) == {"corr", "cov"}


def test_empty_filter_under_backed(multi_key_path):
    """The backed path builds the same three bridges from its own code, so it
    would pass vacuously if only the eager site were gated."""
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata(backed=True, obsp=[], varp=[])

    assert len(adata.obsp) == 0
    assert len(adata.varp) == 0
    assert type(adata._obsp).__name__ != "ScxLazyPairwiseMapping"
    assert set(adata.varm) == {"PCs", "loadings"}


# --------------------------------------------------------------------------
# Partial filter — narrowed, and still lazy.
# --------------------------------------------------------------------------


def test_varm_partial_filter_narrows_and_stays_lazy(multi_key_path):
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata(varm=["PCs"])
    bridge = adata._varm

    assert type(bridge).__name__ == "ScxLazyVarmMapping"
    assert set(bridge.keys()) == {"PCs"}
    # Narrowed *and* deferred: the key set shrank without a decode.
    assert "1 keys, 0 materialized" in repr(bridge)

    value = adata.varm["PCs"]
    assert value.shape == (N_VARS, 4)
    assert "1 materialized" in repr(bridge)


def test_obsp_partial_filter_narrows_and_stays_lazy(multi_key_path):
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata(obsp=["distances"])
    bridge = adata._obsp

    assert type(bridge).__name__ == "ScxLazyPairwiseMapping"
    assert set(bridge.keys()) == {"distances"}
    assert "1 keys, 0 materialized" in repr(bridge)

    assert adata.obsp["distances"].shape == (N_OBS, N_OBS)
    assert "1 materialized" in repr(bridge)


def test_partial_filter_values_match_the_unfiltered_read(multi_key_path):
    """Selecting a key must not change what that key contains."""
    import pyscx

    full = pyscx.open(multi_key_path).to_anndata()
    picked = pyscx.open(multi_key_path).to_anndata(varm=["loadings"], varp=["cov"])

    np.testing.assert_array_equal(picked.varm["loadings"], full.varm["loadings"])
    np.testing.assert_array_equal(
        picked.varp["cov"].todense(), full.varp["cov"].todense()
    )


def test_partial_filter_under_preserve_slots_narrows_the_sliced_result(
    multi_key_path,
):
    """`preserve_slots=True` takes its own branch: eager assembly, then a
    pandas-mask slice. Unknown keys were covered there; that the filter actually
    narrows the surviving slots was not."""
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata(
        obs_filter="group == 'a'",
        preserve_slots=True,
        varm=["PCs"],
        obsp=[],
    )

    assert set(adata.varm) == {"PCs"}
    assert len(adata.obsp) == 0
    assert set(adata.varp) == {"corr", "cov"}
    # The row filter still applied, so this is not passing by returning nothing.
    assert adata.n_obs == N_OBS // 2


def test_partial_filter_under_backed(multi_key_path):
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata(backed=True, varm=["PCs"])

    assert set(adata.varm) == {"PCs"}
    assert adata.varm["PCs"].shape == (N_VARS, 4)


# --------------------------------------------------------------------------
# Unknown keys fail loud, naming the slot — as `obsm=` already does.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slot", ["obsp", "varp", "varm"])
def test_unknown_key_raises_keyerror_naming_the_slot(multi_key_path, slot):
    import pyscx

    with pytest.raises(KeyError) as exc:
        pyscx.open(multi_key_path).to_anndata(**{slot: ["not_a_key"]})

    msg = str(exc.value)
    assert slot in msg
    assert "not_a_key" in msg


@pytest.mark.parametrize("slot", ["obsp", "varp", "varm"])
def test_unknown_key_raises_under_backed_too(multi_key_path, slot):
    import pyscx

    with pytest.raises(KeyError):
        pyscx.open(multi_key_path).to_anndata(backed=True, **{slot: ["not_a_key"]})


@pytest.mark.parametrize("slot", ["obsm", "layers"])
def test_unknown_key_raises_for_the_older_slot_kwargs_too(multi_key_path, slot):
    """`obsm=` documented `KeyError` but only enforced it inside the eager
    assembler; `layers=` never enforced one at all and silently returned an
    empty slot. Both matter now that the query path's "does not load {slots}"
    warning is decided by the same filter: `slot_has_selected` cannot tell "the
    caller excluded this slot" from "the caller misspelled a key", so an
    unvalidated typo produced neither an error nor a warning naming the slot."""
    import pyscx

    with pytest.raises(KeyError) as exc:
        pyscx.open(multi_key_path).to_anndata(**{slot: ["not_a_key"]})

    assert "not_a_key" in str(exc.value)


@pytest.mark.parametrize("slot", ["obsm", "layers"])
def test_older_slot_kwargs_are_validated_on_the_query_path(multi_key_path, slot):
    import pyscx

    with pytest.raises(KeyError):
        pyscx.open(multi_key_path).to_anndata(
            obs_filter="group == 'a'", **{slot: ["not_a_key"]}
        )


@pytest.mark.parametrize("slot", ["obsm", "layers"])
def test_older_slot_kwargs_are_validated_under_backed(multi_key_path, slot):
    import pyscx

    with pytest.raises(KeyError):
        pyscx.open(multi_key_path).to_anndata(backed=True, **{slot: ["not_a_key"]})


def test_a_typo_never_masquerades_as_an_excluded_slot(tmp_dir):
    """The regression this guards, on the file shape that exposes it.

    On a file whose *only* aligned slot is obsm, a typo made `slot_has_selected`
    false, which removed obsm from the query path's drop warning — so the call
    returned with no `KeyError` and no warning at all. The single-slot fixture
    is load-bearing: on a file with five slots the other four keep the warning
    alive and the omission hides."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    rng = np.random.RandomState(3)
    x = rng.random_sample((N_OBS, N_VARS)).astype(np.float32)
    adata = anndata.AnnData(
        X=sp.csr_matrix(x),
        obs=pd.DataFrame(
            {"group": ["a" if i % 2 else "b" for i in range(N_OBS)]},
            index=[f"cell_{i}" for i in range(N_OBS)],
        ),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(N_VARS)]),
        obsm={"X_pca": rng.random_sample((N_OBS, 3)).astype(np.float32)},
    )
    path = str(tmp_dir / "obsm_only.scx")
    pyscx.from_anndata(adata, path)

    # A correct key still warns that the query path drops obsm.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pyscx.open(path).to_anndata(obs_filter="group == 'a'", obsm=["X_pca"])
    assert [w for w in rec if "does not load" in str(w.message)]

    # A typo raises rather than returning in silence.
    with pytest.raises(KeyError):
        pyscx.open(path).to_anndata(obs_filter="group == 'a'", obsm=["X_pca_typo"])


@pytest.mark.parametrize("slot", ["obsp", "varp", "varm"])
def test_unknown_key_raises_on_the_query_path_too(multi_key_path, slot):
    """The obs-filtered query path discards these slots anyway, which is
    exactly why it would be the one to swallow a typo -- the caller would see
    an empty slot and a "not loaded" warning and never learn the key was
    wrong. Validation lives at the entry point, above the branch."""
    import pyscx

    with pytest.raises(KeyError):
        pyscx.open(multi_key_path).to_anndata(
            obs_filter="group == 'a'", **{slot: ["not_a_key"]}
        )


@pytest.mark.parametrize("slot", ["obsp", "varp", "varm"])
def test_unknown_key_raises_under_preserve_slots_too(multi_key_path, slot):
    import pyscx

    with pytest.raises(KeyError):
        pyscx.open(multi_key_path).to_anndata(
            obs_filter="group == 'a'", preserve_slots=True, **{slot: ["not_a_key"]}
        )


# --------------------------------------------------------------------------
# The default is unchanged.
# --------------------------------------------------------------------------


def test_default_call_loads_every_slot(multi_key_path):
    import pyscx

    adata = pyscx.open(multi_key_path).to_anndata()

    assert set(adata.layers) == {"counts", "spliced"}
    assert set(adata.obsm) == {"X_pca", "X_umap"}
    assert set(adata.obsp) == {"connectivities", "distances"}
    assert set(adata.varp) == {"corr", "cov"}
    assert set(adata.varm) == {"PCs", "loadings"}


def test_explicit_none_matches_the_default(multi_key_path):
    import pyscx

    default = pyscx.open(multi_key_path).to_anndata()
    explicit = pyscx.open(multi_key_path).to_anndata(obsp=None, varp=None, varm=None)

    assert set(default.obsp) == set(explicit.obsp)
    assert set(default.varp) == set(explicit.varp)
    assert set(default.varm) == set(explicit.varm)


# --------------------------------------------------------------------------
# The query path's "these slots are being dropped" warning must not name a
# slot the caller deliberately excluded.
# --------------------------------------------------------------------------


def test_query_path_warning_omits_excluded_slots(multi_key_path):
    import pyscx

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pyscx.open(multi_key_path).to_anndata(obs_filter="group == 'a'", obsp=[])

    dropped = [str(w.message) for w in rec if "does not load" in str(w.message)]
    assert dropped, "the query path still drops the other slots, so it must warn"
    assert "obsp" not in dropped[0]
    # The slots the caller did NOT exclude are still named.
    assert "varp" in dropped[0]
    assert "varm" in dropped[0]


def test_query_path_warning_is_silent_when_all_slots_excluded(multi_key_path):
    import pyscx

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pyscx.open(multi_key_path).to_anndata(
            obs_filter="group == 'a'",
            layers=[],
            obsm=[],
            obsp=[],
            varp=[],
            varm=[],
        )

    assert not [w for w in rec if "does not load" in str(w.message)]


# --------------------------------------------------------------------------
# `modality=` cannot honour these, so it must refuse them rather than drop
# them on the floor (the existing contract for var_names / layers / obsm).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, named",
    [
        ({"obsp": ["connectivities"]}, "obsp"),
        ({"varp": ["corr"]}, "varp"),
        ({"varm": ["PCs"]}, "varm"),
        ({"raw": False}, "raw=False"),
    ],
)
def test_modality_rejects_slot_filters(multi_key_path, kwargs, named):
    """`to_anndata_backed_for_modality` takes no selection at all, so a filter
    reaching it would be dropped on the floor. Assert on the message, not just
    the exception type: a single-modality file also raises `ValueError` for
    "unknown modality", which would make the type check pass vacuously."""
    import pyscx

    with pytest.raises(ValueError) as exc:
        pyscx.open(multi_key_path).to_anndata(
            backed=True, modality="rna", **kwargs
        )

    assert named in str(exc.value)
    assert "does not support" in str(exc.value)
