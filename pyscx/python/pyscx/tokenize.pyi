"""Type stubs for `pyscx.tokenize.*` — the W6 tokenisation kernels.

Intentionally narrow, mirroring the philosophy of the package-level
`__init__.pyi`: the parameters whose *values* are constrained are typed as
`Literal`, array arguments and return payloads fall back to `Any` via the
trailing `__getattr__`.

Every kernel takes a whole gathered CSR batch as `(indptr, indices, data)` and
returns numpy arrays moved (not copied) out of Rust, with the GIL released for
the computation.

Three of the reference tokenisers these kernels follow cannot be reproduced
exactly — scGPT's binning and UCE's sampler draw from numpy's global RNG, and
Geneformer's tie order is `np.argsort`'s unstable default. What each kernel
guarantees instead is stated in its docstring and in `docs/tokenize.md`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

# How a value sitting exactly on a bin edge is placed. "left"/"right" are
# `np.digitize`'s two deterministic bounds; "seeded" is scGPT's randomised
# interpolation, keyed on (seed, file_identity, row) instead of numpy's
# global RNG.
BinTie = Literal["left", "right", "seeded"]
# What the sampling weight is a function of. "log1p" is UCE's choice.
WeightTransform = Literal["log1p", "linear"]
# The collator's preprocess modes, as standalone transforms.
TransformMode = Literal["pass_through", "log1p_raw", "normalize_log1p", "pflog_raw"]

# Version of the kernel contract this build implements. Consumers assert it at
# setup so version skew fails loudly rather than mid-training.
CONTRACT_VERSION: int


def gene_mask_id(n_genes_total: int) -> int: ...


def pad_id(n_genes_total: int) -> int: ...


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def top_k(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    k: int,
    n_genes_total: int,
) -> dict[str, Any]: ...


def rank_tokens(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    gene_stats: np.ndarray,
    l_max: int,
    vocabulary_version: str,
    target_sum: float = 1e4,
) -> dict[str, Any]: ...


def bin_values(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_bins: int,
    # When given, must be exactly `n_bins - 1` finite non-decreasing edges, or
    # the emitted bins and the declared `n_bins` disagree.
    edges: np.ndarray | None = None,
    tie: BinTie = "left",
    seed: int = 0,
    file_identity: int = 0,
    rows: np.ndarray | None = None,
) -> dict[str, Any]: ...


def sample_genes(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n: int,
    seed: int,
    file_identity: int,
    weight: WeightTransform = "log1p",
    rows: np.ndarray | None = None,
) -> dict[str, Any]: ...


def transform_values(
    indptr: np.ndarray,
    data: np.ndarray,
    mode: TransformMode,
    target_sum: float = 1e4,
    pflog_alpha: float | None = None,
    n_measured: int | None = None,
) -> np.ndarray: ...


def library_size(indptr: np.ndarray, data: np.ndarray) -> np.ndarray: ...


def measured_mask(
    indptr: np.ndarray, indices: np.ndarray, panel: np.ndarray
) -> np.ndarray: ...


# Catch-all for the rest of `pyscx.tokenize.*`.
def __getattr__(name: str) -> Any: ...
