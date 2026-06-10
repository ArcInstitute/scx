"""Type stubs for `pyscx.accel.*` `prefer_format` kwargs.

Intentionally narrow — mirrors the philosophy of the package-level
`__init__.pyi`. Only the `prefer_format` entry on each affected
function is typed; everything else falls back to `Any` via the
trailing `__getattr__`.

`prefer_format: Literal["csr", "csc"] = "csr"` is the explicit
opt-in surface for the column-major sidecar dispatch. Validation
lives in the Rust side (`PyValueError` on any other value,
including `"auto"`).
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
    device: str = "auto",
) -> Any: ...


def rank_genes_groups_df(
    adata: Any,
    groupby: str,
    reference: str = "rest",
    n_genes: int | None = None,
    gene_chunk_size: int | None = None,
    rankby_abs: bool = False,
    tie_correct: bool = False,
    device: str = "auto",
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
    layer: str | None = None,
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


def pdex_ref(
    adata: Any,
    groupby: str,
    *,
    reference: str = "non-targeting",
    is_log1p: bool | None = None,
    geometric_mean: bool = True,
    epsilon: float = 0.0,
    gene_chunk_size: int | None = None,
    prefer_format: PreferFormat = "csr",
    device: str = "auto",
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
# Lower-level column aggregations
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
    allow_tf32: bool = False,
    spmm_policy: str = "default",
    memory_budget: int | str | None = None,
) -> None: ...


# ---------------------------------------------------------------------------
# Graph / embedding / integration ops
# ---------------------------------------------------------------------------


def neighbors(
    adata: Any,
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
    random_state: int = 0,
    ef_construction: int = 200,
    ef_search: int = 200,
    device: str = "auto",
) -> None: ...


def umap(
    adata: Any,
    n_components: int = 2,
    n_epochs: int = 200,
    min_dist: float = 0.1,
    spread: float = 1.0,
    negative_sample_rate: int = 5,
    learning_rate: float = 1.0,
    random_state: int = 0,
    device: str = "auto",
) -> None: ...


def leiden(
    adata: Any,
    resolution: float = 1.0,
    key_added: str = "leiden",
    random_state: int = 0,
    n_iterations: int = 2,
    device: str = "auto",
    parallel: bool = False,
    theta: float = 1.0,
) -> None: ...


def harmony_integrate(
    adata: Any,
    key: Any,
    *,
    basis: str = "X_pca",
    adjusted_basis: str | None = None,
    n_clusters: int | None = None,
    theta: Any | None = None,
    sigma: float = 0.1,
    lamb: Any | None = None,
    alpha: float = 0.2,
    max_iter: int = 10,
    max_iter_kmeans: int = 4,
    epsilon_harmony: float = 1e-2,
    epsilon_kmeans: float = 1e-3,
    block_size: float = 0.05,
    batch_prop_cutoff: float = 1e-5,
    tau: float = 0.0,
    random_state: int = 0,
    device: str = "auto",
) -> None: ...


def compute_lisi(
    adata: Any,
    key: str,
    *,
    basis: str = "X_pca",
    perplexity: float = 30.0,
    n_neighbors: int | None = None,
) -> Any: ...


# Catch-all for the rest of `pyscx.accel.*`.
def __getattr__(name: str) -> Any: ...
