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


def test_view_x_does_not_copy():
    """``view.X`` must be the *un-copied* ``_subset`` result.

    This is the single substitution `rebuild_via_anndata` makes against
    `_inplace_subset_var`: anndata's own routine goes through `.copy()`, which
    materializes an SCX handle on purpose. If `AnnData.X` on a view started
    copying, every in-place accelerator would silently materialize.
    """
    import inspect

    from anndata import AnnData

    src = inspect.getsource(AnnData.X.fget)
    assert "_subset(self._adata_ref.X" in src, (
        "AnnData.X no longer resolves a view through `_subset`; check that "
        "`rebuild_via_anndata` still gets a lazy matrix out of `view.X`"
    )
    assert ".copy()" not in src, (
        "AnnData.X on a view now copies; `rebuild_via_anndata` would "
        "materialize the matrix it is trying to keep on disk"
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
