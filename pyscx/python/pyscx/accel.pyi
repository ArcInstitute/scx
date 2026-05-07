"""Type stubs for `pyscx.accel.*` `prefer_format` kwargs.

Intentionally narrow — mirrors the philosophy of the package-level
`__init__.pyi`. Only the `prefer_format` entry on each affected
function is typed; everything else falls back to `Any` via the
trailing `__getattr__`.

`prefer_format: Literal["csr", "csc"] = "csr"` is the explicit
opt-in surface from CSC-SUPPORT.md Phase G.1. Validation lives in
the Rust side (`PyValueError` on any other value, including `"auto"`).
"""

from __future__ import annotations

from typing import Any, Literal

# Type alias used by every affected entry.
PreferFormat = Literal["csr", "csc"]


# ---------------------------------------------------------------------------
# DE / HVG / pseudobulk
# ---------------------------------------------------------------------------


def rank_genes_groups(
    adata: Any,
    groupby: str,
    reference: str = "rest",
    n_genes: int | None = None,
    method: str = "wilcoxon",
    gene_chunk_size: int | None = None,
    stratify_by: list[str] | None = None,
    min_cells_per_stratum: int = 50,
    rankby_abs: bool = False,
    tie_correct: bool = False,
    prefer_format: PreferFormat = "csr",
) -> Any: ...


def highly_variable_genes(
    adata: Any,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    batch_key: str | None = None,
    span: float = 0.3,
    subset: bool = False,
    n_bins: int = 20,
    device: str = "auto",
    prefer_format: PreferFormat = "csr",
) -> None: ...


def pseudobulk_dex(
    adata: Any,
    groupby: list[str],
    test_col: str,
    reference: str,
    design: str | None = None,
    aggr_method: str = "sum",
    min_cells_per_group: int = 10,
    stratify_by: list[str] | None = None,
    min_cells_per_stratum: int = 50,
    prefer_format: PreferFormat = "csr",
    gene_indices: list[int] | None = None,
) -> Any: ...


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def calculate_qc_metrics(
    adata: Any,
    qc_vars: list[str] | None = None,
    log1p: bool = True,
    inplace: bool = True,
    prefer_format: PreferFormat = "csr",
) -> Any: ...


# ---------------------------------------------------------------------------
# Lower-level column aggregations (Phase F.5)
# ---------------------------------------------------------------------------


def col_sums(dataset: Any, prefer_format: PreferFormat = "csr") -> Any: ...
def col_nnz(dataset: Any, prefer_format: PreferFormat = "csr") -> Any: ...
def col_min(dataset: Any, prefer_format: PreferFormat = "csr") -> Any: ...
def col_max(dataset: Any, prefer_format: PreferFormat = "csr") -> Any: ...
def col_var(dataset: Any, prefer_format: PreferFormat = "csr") -> Any: ...


# ---------------------------------------------------------------------------
# PCA — `prefer_format="csc"` is rejected (raises ValueError on the Rust
# side). Stubbed here so the kwarg shape matches the Rust signature.
# ---------------------------------------------------------------------------


def pca(
    adata: Any,
    n_comps: int = 50,
    zero_center: bool = True,
    random_state: int = 0,
    n_oversamples: int = 10,
    n_power_iterations: int = 2,
    device: str = "auto",
    method: str = "auto",
    qr_method: str = "householder",
    prefer_format: PreferFormat = "csr",
) -> None: ...


# Catch-all for the rest of `pyscx.accel.*`.
def __getattr__(name: str) -> Any: ...
