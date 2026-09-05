"""Type stubs for `pyscx.accel.*` `prefer_format` kwargs.

Intentionally narrow — mirrors the philosophy of the package-level
`__init__.pyi`. Only the `prefer_format` entry on each affected
function is typed; everything else falls back to `Any` via the
trailing `__getattr__`.

`prefer_format` selects the column-major sidecar dispatch. Most ops
take `Literal["csr", "csc"] = "csr"`; the DE ops (`rank_genes_groups`,
`pdex_ref`) additionally accept `"auto"` and **default to it** — `"auto"`
routes CSC-direct on CPU when a valid sidecar is present, else CSR.
Validation lives on the Rust side (`PyValueError` on any other value;
non-DE ops still reject `"auto"`).
"""

from __future__ import annotations

from typing import Any, Literal

# Non-DE ops: explicit CSR/CSC opt-in, default CSR.
PreferFormat = Literal["csr", "csc"]
# DE ops (rank_genes_groups / pdex_ref): additionally accept the "auto"
# capability-routed default.
DePreferFormat = Literal["auto", "csr", "csc"]


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
    prefer_format: DePreferFormat = "auto",
    device: str = "auto",
    use_raw: bool | None = None,
    layer: str | None = None,
    *,
    pts: bool = False,
    groups: list[str] | None = None,
    corr_method: str = "benjamini-hochberg",
) -> Any:
    """scanpy-compatible Wilcoxon rank-sum DE, written to ``adata.uns["rank_genes_groups"]``.

    ``pts=True`` adds scanpy's ``uns[key]["pts"]`` (and ``["pts_rest"]`` when
    ``reference="rest"``): ``genes × groups`` DataFrames of the fraction of
    cells with a nonzero value, indexed by var name, over every gene. It is
    one extra streaming pass over ``X`` on every route, and both frames
    round-trip through ``from_anndata`` / ``to_anndata`` and h5ad on the uns
    ``pandas.DataFrame`` envelope. ``groups=`` restricts
    which groups are *reported*, in the given order; "rest" is unchanged, so
    each group's statistics equal the unrestricted run's (and scanpy's); a
    named group (or named reference) with fewer than two cells raises, as in
    scanpy. ``pts=True`` refuses duplicate ``var_names`` (its table is joined
    by name). ``corr_method`` accepts only ``"benjamini-hochberg"`` (recorded
    in ``params``); any other value raises rather than silently applying BH.
    """
    ...


def rank_genes_groups_df(
    adata: Any,
    groupby: str | None = None,
    reference: str = "rest",
    n_genes: int | None = None,
    gene_chunk_size: int | None = None,
    rankby_abs: bool = False,
    tie_correct: bool = False,
    device: str = "auto",
    output: str = "pandas",
    *,
    group: str | list[str] | None = None,
    key: str = "rank_genes_groups",
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
) -> Any:
    """DE DataFrame in two modes (pass one).

    ``groupby=`` re-runs Wilcoxon and returns cell-eval ``DEResults`` columns.
    ``group=`` is the scanpy ``sc.get.rank_genes_groups_df`` alias: extracts the
    precomputed ``adata.uns[key]`` (no recompute) and returns scanpy's columns
    (``names, scores, logfoldchanges, pvals, pvals_adj``; a leading ``group``
    column when ``group`` is a list; ``pct_nz_group`` / ``pct_nz_reference``
    appended when ``uns[key]`` carries ``pts`` / ``pts_rest`` — i.e. after
    ``rank_genes_groups(pts=True)``). ``pval_cutoff``/``log2fc_min``/
    ``log2fc_max`` are scanpy-style row filters for the extraction path.
    ``n_genes`` is a pyscx extension (scanpy's extractor has none): top-N per
    group, before the filters. ``device`` is ignored in extract mode.

    ``group=None`` (or omitting it) with no ``groupby=`` extracts **every** group
    in ``adata.uns[key]``, with a leading ``group`` column — matching
    ``sc.get.rank_genes_groups_df``'s "All groups are returned if group is
    None". Both modes return a **pandas** DataFrame by default; pass
    ``output="polars"`` for the polars frame ``cell_eval`` consumes.
    """
    ...


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
    groupby: str | list[str] | None = None,
    test_col: str | None = None,
    reference: str | None = None,
    design: str | None = None,
    aggr_method: str = "sum",
    min_cells_per_group: int = 10,
    stratify_by: list[str] | None = None,
    min_cells_per_stratum: int = 50,
    prefer_format: PreferFormat = "csr",
    gene_indices: list[int] | None = None,
    n_cpus: int | None = None,
    backend: Literal["pydeseq2", "nb_glm"] | None = None,
    nbglm_options: dict[str, Any] | None = None,
    *,
    sample_cols: str | list[str] | None = None,
    sample_key: str | list[str] | None = None,
) -> Any:
    """Pseudobulk DE: Rust aggregation + PyDESeq2 or the Rust-native NB-GLM.

    ``groupby`` here is **not** what it is in ``rank_genes_groups``. There (and
    throughout scanpy) ``groupby`` names the column whose levels are compared;
    here it names the columns that together define one pseudobulk *sample* —
    condition **plus** replicate, e.g. ``["disease", "donor_id"]`` — and the
    compared column is ``test_col``. ``sample_cols=`` and ``sample_key=`` are
    aliases named for that role; all three accept a bare string as well as a
    list. Pass exactly one of the three.

    ``groupby`` / ``test_col`` / ``reference`` are all semantically required —
    they are typed optional only because the aliases make ``groupby`` optional
    and a required positional cannot follow an optional one.

    ``backend=None`` resolves to ``"nb_glm"``, the Rust-native NB-GLM, which
    needs no optional dependency. It defaulted to ``"pydeseq2"`` through v0.12;
    that engine is still available as ``backend="pydeseq2"`` (``pip install
    'pyscx[pydeseq2]'``) and is required for ``stratify_by`` and for
    ``aggr_method`` other than ``"sum"``.
    """
    ...


def pdex_ref(
    adata: Any,
    groupby: str,
    *,
    reference: str = "non-targeting",
    is_log1p: bool | None = None,
    geometric_mean: bool = True,
    epsilon: float = 1e-9,
    cpm_filter: float | None = None,
    gene_chunk_size: int | None = None,
    prefer_format: DePreferFormat = "auto",
    device: str = "auto",
    output: str = "pandas",
    use_raw: bool | None = None,
    layer: str | None = None,
    groups: list[str] | None = None,
) -> Any:
    """pdex ``mode="ref"`` DE: every ``groupby`` level vs ``reference``.

    ``groups=`` restricts the tested targets to those levels, reported in that
    order; each target's rows equal the unrestricted run's (a target is only
    ever compared with the reference) while the work scales with the number
    of targets asked for. Unknown names, repeats, an empty list and the
    reference itself are errors. A pyscx extension over upstream pdex.
    """
    ...


def pseudobulk_means(
    adata: Any,
    groupby: str | list[str],
    min_cells_per_group: int = 1,
    device: str = "auto",
) -> tuple[Any, list[str] | list[tuple[str, ...]]]:
    """Per-group mean expression: ``(means[P, G] float64, group_names)``.

    ``groupby`` is one obs column or a list of columns whose per-cell tuple
    defines a group (the same ``str | list[str]`` ``pseudobulk_dex`` takes).
    ``group_names`` is sorted lexicographically: a list of ``str`` for one
    column, a list of ``str`` tuples (one entry per column) for several.
    """
    ...


def pdex_nb_glm(
    adata: Any,
    groupby: str,
    reference: str,
    stratify_by: list[str] | None = None,
    min_cells_per_group: int = 10,
    min_cells_per_stratum: int = 50,
    is_log1p: bool | None = None,
    nbglm_options: dict[str, Any] | None = None,
    gene_chunk_size: int | None = None,
    prefer_format: str = "csr",
    device: str = "auto",
    design: str | None = None,
    output: str = "pandas",
) -> Any:
    """Pseudobulk NB-GLM DE in the cell-eval ``DEResults`` column schema.

    ``stratify_by`` is **required** in practice (it forms the pseudobulk
    replicates) and must be a list, not a bare string. Returns pandas by
    default; pass ``output="polars"`` for the container ``cell_eval`` consumes.
    """
    ...


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
    mask_var: Any | None = None,
) -> None: ...


# ---------------------------------------------------------------------------
# PFlog (v4) / shifted-log normalization on raw counts (Booeshaghi et al.,
# DOI 10.1101/2022.05.06.490859). Default `store="pca"` writes a baseline-aware
# PCA embedding to `adata.obsm[obsm_key]` and leaves `X` as raw counts (it does
# NOT transform `X` in place — pass `store="dense"` for the matrix). `alpha=None`
# estimates the NB overdispersion once from the matrix (pseudocount 1/(4α)) and
# stamps `adata.uns["pflog"]`; pass a float to pin it. `pflog_reconstruct` is a
# pure-Python companion exposed on this submodule (and at top level) by
# `pyscx/__init__.py`.
# ---------------------------------------------------------------------------


def pflog(
    adata: Any,
    *,
    alpha: float | None = None,
    layer: str | None = None,
    store: str = "pca",
    n_components: int = 50,
    n_oversamples: int = 10,
    n_power_iterations: int = 2,
    zero_center: bool = True,
    random_state: int = 0,
    obsm_key: str = "X_pflog_pca",
    baseline_key: str = "pflog_baseline",
    layer_out: str | None = None,
    out: str | None = None,
    store_repr: str = "delta_baseline",
    shard_size: int | None = None,
    dense_max_elems: int = 200_000_000,
    device: str = "auto",
) -> None: ...


def pflog_reconstruct(adata: Any, baseline_key: str = "pflog_baseline") -> Any: ...


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
    adjusted_basis: str | None = "X_pca_harmony",
    n_clusters: int | None = None,
    theta: Any | None = None,
    sigma: float = 0.1,
    lamb: Any | None = None,
    alpha: float = 0.2,
    max_iter: int = 10,
    max_iter_kmeans: int = 6,
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
    approximate_knn: bool = False,
) -> Any: ...


# CPU per-stage timing profiler (io / decode / reduction / marshalling).
# Buckets are populated only when `SCX_CPU_PROFILE=1` is set at process start.
def cpu_profile_snapshot() -> dict[str, Any]: ...


def cpu_profile_reset() -> None: ...


# Catch-all for the rest of `pyscx.accel.*`.
def __getattr__(name: str) -> Any: ...
