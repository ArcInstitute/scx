"""The anndata private API pyscx registers against must still be there.

Phase 4.0b hands axis subsetting to anndata. That works because pyscx registers
SCX handle types with three `singledispatch` seams and then drives anndata's own
view/copy machinery. All of it is private and unversioned: a rename inside the
declared version bound would otherwise surface as an `AttributeError` or a
`NotImplementedError` from inside a user's ``filter_genes``.

One assertion per name, so an anndata upgrade names its casualty. See
``docs/compatibility-matrix.md`` for the tested range, and
``pyscx/src/anndata_hooks.rs`` for what each hook does.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("anndata")

# (module, attribute) for each `singledispatch` seam pyscx registers with.
SEAMS = [
    ("anndata._core.views", "as_view"),
    ("anndata._core.index", "_subset"),
    ("anndata._core.file_backing", "to_memory"),
]

# The AnnData internals `axis_align::rebuild_via_anndata` drives directly.
ANNDATA_METHODS = [
    "_inplace_subset_obs",
    "_inplace_subset_var",
    "_mutated_copy",
    "_init_as_actual",
]

# The view internals `axis_align::devirtualize_scx_view` reads to decide whether
# the AnnData it was handed is a view over a backed `X`. One entry per name, so
# an anndata upgrade names its own casualty.
ANNDATA_VIEW_ATTRS = [
    # Truthy on a view. Without it every accel op would try to detect a view by
    # catching the `TypeError` from anndata's X-setter.
    "is_view",
    # The parent a view was sliced from. Probed instead of `adata.X` because on
    # a view of an in-memory parent, reading `adata.X` performs a real
    # submatrix copy for a case the prologue then declines.
    "_adata_ref",
]

# Every SCX handle that can appear as `X`, a layer, or an aligned value.
HANDLE_CLASS_NAMES = [
    "ScxBackedSparseDataset",
    "ScxBackedLayerDataset",
    "ScxBackedObsmDataset",
    "ScxLazyTransformedDataset",
]


def _handle_classes():
    import pyscx

    return [getattr(pyscx, name) for name in HANDLE_CLASS_NAMES]


@pytest.fixture
def scx_hooks_path(tmp_dir):
    """A minimal backed SCX file for the behavioural compat checks."""
    import anndata
    import numpy as np
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    rng = np.random.RandomState(11)
    x = (rng.random_sample((20, 8)) * 10).astype(np.float32)
    adata = anndata.AnnData(
        X=sp.csr_matrix(x),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(8)]),
    )
    path = str(tmp_dir / "hooks_compat.scx")
    pyscx.from_anndata(adata, path)
    return path


@pytest.mark.parametrize(("module_name", "attr"), SEAMS)
def test_seam_exists_and_is_singledispatch(module_name, attr):
    module = importlib.import_module(module_name)
    hook = getattr(module, attr, None)
    assert hook is not None, f"{module_name}.{attr} is gone"
    assert hasattr(hook, "register") and hasattr(hook, "registry"), (
        f"{module_name}.{attr} is no longer a singledispatch; pyscx cannot "
        "register SCX handles with it"
    )


@pytest.mark.parametrize(("module_name", "attr"), SEAMS)
@pytest.mark.parametrize("cls_name", HANDLE_CLASS_NAMES)
def test_handle_is_registered(module_name, attr, cls_name):
    import pyscx

    hook = getattr(importlib.import_module(module_name), attr)
    cls = getattr(pyscx, cls_name)
    assert cls in hook.registry, (
        f"{cls_name} is not registered with {module_name}.{attr}; importing "
        "pyscx should have done that"
    )


@pytest.mark.parametrize("method", ANNDATA_METHODS)
def test_anndata_method_exists(method):
    from anndata import AnnData

    assert hasattr(AnnData, method), (
        f"AnnData.{method} is gone; pyscx.accel.filter_cells / filter_genes / "
        "subset_obs / subset_var drive it directly"
    )


@pytest.mark.parametrize("attr", ANNDATA_VIEW_ATTRS)
def test_anndata_view_attr_exists(attr, scx_hooks_path):
    """Asserted on an actual view, not on the class.

    `is_view` is a property and `_adata_ref` is only set on a view, so
    `hasattr(AnnData, ...)` would pass vacuously for one and fail spuriously
    for the other.
    """
    import numpy as np
    import pyscx

    adata = pyscx.open(scx_hooks_path).to_anndata(backed=True)
    view = adata[np.arange(adata.n_obs) % 2 == 0]
    assert hasattr(view, attr), (
        f"AnnData view has no {attr}; pyscx.accel's prologue "
        "(axis_align::devirtualize_scx_view) uses it to spot a view over a "
        "backed X before an accel op writes to it"
    )


@pytest.mark.parametrize("store", ["_layers", "_obsm", "_varm", "_obsp", "_varp"])
def test_raw_aligned_store_is_reachable(store):
    """The bypass detaches these to keep the lazy bridges off disk."""
    import anndata
    import numpy as np
    import pandas as pd

    adata = anndata.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g", "h"]),
    )
    assert hasattr(adata, store), (
        f"AnnData no longer keeps its aligned mapping in `{store}`; the "
        "lazy-bridge bypass in axis_align.rs reads and writes it directly"
    )


def test_view_x_does_not_copy_behaviourally(scx_hooks_path):
    """``view.X`` must be the *un-copied* ``_subset`` result.

    This is the single substitution `rebuild_via_anndata` makes against
    `_inplace_subset_var`: anndata's own routine goes through `.copy()`, which
    materializes an SCX handle on purpose. If `AnnData.X` on a view started
    copying, every in-place accelerator would silently materialize.

    Asserted on behaviour rather than on source text, so an anndata refactor
    that preserves the semantics does not fail the build.
    """
    import numpy as np
    import pyscx

    adata = pyscx.open(scx_hooks_path).to_anndata(backed=True)
    view = adata[np.arange(adata.n_obs) % 2 == 0]
    assert type(view.X).__name__ == "ScxBackedSparseDataset", (
        f"AnnData.X on a view returned {type(view.X).__name__}, not a lazy SCX "
        "handle — `rebuild_via_anndata` would materialize the matrix it is "
        "trying to keep on disk"
    )


def test_view_x_source_still_routes_through_subset():
    """Source-level tripwire for the same contract as the test above.

    Kept as an *early warning* that names the mechanism, not as the primary
    guard: it fires on a harmless anndata refactor, so it is deliberately
    paired with the behavioural check and stays a `pytest.fail` with a
    pointer rather than an assertion about correctness.
    """
    import inspect

    from anndata import AnnData

    src = inspect.getsource(AnnData.X.fget)
    if "_subset(self._adata_ref.X" not in src or ".copy()" in src:
        pytest.fail(
            "AnnData.X on a view no longer looks like an un-copied `_subset`. "
            "If test_view_x_does_not_copy_behaviourally still passes, the "
            "contract holds and this tripwire just needs updating."
        )


def test_registration_reports_success():
    """pyscx must believe its own registration worked.

    `axis_align` refuses to subset a backed X when it doesn't, so a silent
    registration failure would turn every `filter_genes` into a hard error —
    better caught here than there.
    """
    import numpy as np
    import pyscx

    # No public accessor for the flag; exercise the path that consults it.
    # A no-op mask returns before the check, so use one that drops a row.
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    adata = anndata.AnnData(
        X=sp.csr_matrix(np.eye(3, dtype=np.float32)),
        obs=pd.DataFrame(index=list("abc")),
        var=pd.DataFrame(index=list("xyz")),
    )
    pyscx.accel.subset_obs(adata, np.array([True, True, False]))
    assert adata.n_obs == 2


# ---------------------------------------------------------------------------
# is_backed_handle vs HANDLE_CLASSES — the drift guard
# ---------------------------------------------------------------------------


def _handle_instances(tmp_dir):
    """One live instance per handle class, keyed by class name.

    Deliberately a hand-written mapping: a newly registered handle class with
    no entry here makes the guard below fail loudly, which is the point.
    """
    import anndata
    import numpy as np
    import pyscx
    import scipy.sparse as sp

    rng = np.random.RandomState(5)
    x = (rng.random_sample((30, 10)) * 20).astype(np.float32)
    x[rng.random_sample((30, 10)) > 0.6] = 0.0
    adata = anndata.AnnData(
        X=sp.csr_matrix(x),
        layers={"counts": sp.csr_matrix(x)},
        obsm={"X_emb": rng.random_sample((30, 3)).astype(np.float32)},
    )
    path = str(tmp_dir / "handle_instances.scx")
    pyscx.from_anndata(adata, path)

    # `obsm=[…]` is what keeps an aligned store lazy rather than materializing.
    backed = pyscx.open(path).to_anndata(backed=True, obsm=["X_emb"])
    transformed = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.normalize_total(transformed, target_sum=1e4)

    out = {}
    for obj in (backed.X, backed.layers["counts"], backed.obsm["X_emb"], transformed.X):
        out[type(obj).__name__] = obj
    return out


def test_is_backed_handle_agrees_with_the_registered_handle_classes(tmp_dir):
    """`pyscx.is_backed_handle` must accept exactly the classes that
    `HANDLE_CLASSES` registers with anndata's seams.

    The Rust `HANDLE_CLASSES` list drives seam registration, while the
    predicate is a separate cast chain beside it. Those can drift silently: a
    fifth handle class added to the list would get its `as_view` / `_subset` /
    `to_memory` hooks registered and still be rejected by `is_backed_handle`,
    so every caller branching on it would take the wrong arm. The seam
    registry is used as the authority here precisely because it is the list's
    own output, not a second copy of it.
    """
    import pyscx

    hook = getattr(importlib.import_module("anndata._core.index"), "_subset")
    # pyo3 classes report `__module__ == "builtins"`, so key off the name.
    registered = {
        cls.__name__ for cls in hook.registry if cls.__name__.startswith("Scx")
    }
    assert registered == set(HANDLE_CLASS_NAMES), (
        "the SCX classes registered with anndata's _subset seam have changed; "
        f"registered={sorted(registered)} vs HANDLE_CLASS_NAMES="
        f"{sorted(HANDLE_CLASS_NAMES)}"
    )

    instances = _handle_instances(tmp_dir)
    missing = registered - set(instances)
    assert not missing, (
        f"no instance factory in this test for {sorted(missing)}; a handle "
        "class was added to HANDLE_CLASSES — extend `_handle_instances` and "
        "confirm `is_backed_handle` accepts it"
    )
    for name in sorted(registered):
        assert pyscx.is_backed_handle(instances[name]), (
            f"{name} is registered with anndata's seams but "
            "`is_backed_handle` rejects it — the cast chain in "
            "`anndata_hooks.rs` is out of sync with `HANDLE_CLASSES`"
        )
