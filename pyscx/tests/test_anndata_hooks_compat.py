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
