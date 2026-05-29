"""Predicate-index plumbing through merge / append / append_from_anndata
/ compact — the silent-data-loss surface fixed by the
`*_with_index_options` variants in scx-ops.

The Rust-side `predicate_index_rewrite.rs` integration tests already
verify the catalog-level shape of the output. These Python tests
exercise the pyscx kwargs and confirm `filter_obs` pushdown returns
correct results on the rewritten file (a fast smoke-test of the read
path; correctness vs. a full obs scan is the user-visible regression
that motivated the fix).
"""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx


def _mk_adata(n_obs: int, n_vars: int, perturbation: str) -> anndata.AnnData:
    """Synthetic AnnData with a single perturbation value across all
    rows. We use this to build per-pert inputs and then merge."""
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 50, size=(n_obs, n_vars), dtype=np.int32).astype(
        np.float32
    )
    # Inject some zeros so the matrix is genuinely sparse.
    dense[rng.random((n_obs, n_vars)) > 0.4] = 0.0
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {
            "cell_id": [f"{perturbation}_{i}" for i in range(n_obs)],
            "perturbation": pd.Categorical(
                [perturbation] * n_obs, categories=["DRUG_A", "DRUG_B"]
            ),
            "cell_type": pd.Categorical(
                ["fibroblast" if i % 2 == 0 else "epithelial" for i in range(n_obs)]
            ),
        },
        index=[f"{perturbation}_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=x, obs=obs, var=var)


def test_merge_with_index_obs_writes_predicate_index(tmp_dir):
    a = tmp_dir / "a.scx"
    b = tmp_dir / "b.scx"
    out = tmp_dir / "merged.scx"
    pyscx.from_anndata(
        _mk_adata(60, 16, "DRUG_A"),
        str(a),
        index_obs=["perturbation", "cell_type"],
    )
    pyscx.from_anndata(
        _mk_adata(60, 16, "DRUG_B"),
        str(b),
        index_obs=["perturbation", "cell_type"],
    )

    # Pre-fix: this would silently drop the predicate index. With the
    # new kwargs the merged output has it.
    pyscx.merge(
        [str(a), str(b)],
        str(out),
        index_obs=["perturbation", "cell_type"],
    )

    # Round-trip a filter that the index covers: must return the
    # correct row count (subset to DRUG_A) regardless of whether the
    # index was used internally.
    exp = pyscx.open(str(out))
    got = exp.query().filter_obs('perturbation == "DRUG_A"').count()
    assert got == 60


def test_merge_without_index_options_drops_predicate_index(tmp_dir):
    """Backwards compat: legacy `pyscx.merge(inputs, output)` call
    shape (no index kwargs) still works and produces a merged file
    with no predicate-index section."""
    a = tmp_dir / "a.scx"
    b = tmp_dir / "b.scx"
    out = tmp_dir / "merged.scx"
    pyscx.from_anndata(_mk_adata(30, 8, "DRUG_A"), str(a))
    pyscx.from_anndata(_mk_adata(30, 8, "DRUG_B"), str(b))

    pyscx.merge([str(a), str(b)], str(out))
    # The query should still return correct results — only latency
    # regresses, which is harder to assert here. Read-side smoke:
    exp = pyscx.open(str(out))
    assert exp.query().filter_obs('perturbation == "DRUG_A"').count() == 30


def test_compact_with_index_options_writes_predicate_index(tmp_dir):
    src = tmp_dir / "src.scx"
    out = tmp_dir / "compacted.scx"
    pyscx.from_anndata(_mk_adata(64, 16, "DRUG_A"), str(src))

    pyscx.compact(str(src), str(out), index_obs=["perturbation"])

    exp = pyscx.open(str(out))
    # All rows have perturbation=DRUG_A; filter returns all 64.
    assert exp.query().filter_obs('perturbation == "DRUG_A"').count() == 64


def test_compact_reshape_obs_no_index(tmp_dir):
    """reshape_obs=True with no index kwargs migrates legacy single-section
    obs to the sharded layout without building a predicate index. Asserts the
    read path round-trips (catalog-level shard shape is covered by the Rust
    integration tests)."""
    src = tmp_dir / "src.scx"
    out = tmp_dir / "reshaped.scx"
    # force_legacy_metadata + small shard_size: source has single-section
    # obs, and reshape will split it across multiple ObsMetadataShard rows.
    pyscx.from_anndata(
        _mk_adata(64, 16, "DRUG_A"),
        str(src),
        force_legacy_metadata=True,
        shard_size=16,
    )

    pyscx.compact(str(src), str(out), reshape_obs=True)

    exp = pyscx.open(str(out))
    assert exp.n_obs == 64
    # Pushdown still works post-reshape even without an explicit index.
    assert exp.query().filter_obs('perturbation == "DRUG_A"').count() == 64


def test_compact_reshape_obs_with_index(tmp_dir):
    """reshape_obs composes with the indexed compact path."""
    src = tmp_dir / "src.scx"
    out = tmp_dir / "reshaped_indexed.scx"
    pyscx.from_anndata(
        _mk_adata(64, 16, "DRUG_A"),
        str(src),
        force_legacy_metadata=True,
        shard_size=16,
    )

    pyscx.compact(str(src), str(out), reshape_obs=True, index_obs=["perturbation"])

    exp = pyscx.open(str(out))
    assert exp.n_obs == 64
    assert exp.query().filter_obs('perturbation == "DRUG_A"').count() == 64


def test_append_from_anndata_with_index_options_writes_predicate_index(tmp_dir):
    target = tmp_dir / "target.scx"
    pyscx.from_anndata(_mk_adata(32, 16, "DRUG_A"), str(target))

    new = _mk_adata(32, 16, "DRUG_B")
    pyscx.append_from_anndata(
        str(target),
        new,
        index_obs=["perturbation", "cell_type"],
    )

    exp = pyscx.open(str(target))
    # Post-append the file has both perturbations.
    assert exp.query().filter_obs('perturbation == "DRUG_A"').count() == 32
    assert exp.query().filter_obs('perturbation == "DRUG_B"').count() == 32


def test_append_with_index_options_writes_predicate_index(tmp_dir):
    target = tmp_dir / "target.scx"
    source = tmp_dir / "source.scx"
    pyscx.from_anndata(_mk_adata(20, 8, "DRUG_A"), str(target))
    pyscx.from_anndata(_mk_adata(20, 8, "DRUG_B"), str(source))

    pyscx.append(
        str(target),
        str(source),
        index_obs=["perturbation"],
    )

    exp = pyscx.open(str(target))
    assert exp.query().filter_obs('perturbation == "DRUG_A"').count() == 20
    assert exp.query().filter_obs('perturbation == "DRUG_B"').count() == 20


def test_merge_unknown_forced_column_raises(tmp_dir):
    a = tmp_dir / "a.scx"
    b = tmp_dir / "b.scx"
    out = tmp_dir / "merged.scx"
    pyscx.from_anndata(_mk_adata(8, 4, "DRUG_A"), str(a))
    pyscx.from_anndata(_mk_adata(8, 4, "DRUG_B"), str(b))

    # The forced obs column doesn't exist — must raise ValueError.
    with pytest.raises(ValueError, match="forced obs"):
        pyscx.merge(
            [str(a), str(b)],
            str(out),
            index_obs=["nonexistent_column"],
        )
    # Issue 1 (P1) — upfront validation must fail BEFORE the merged
    # output is created. A user who retries on ValueError should not
    # see a stale merged file from the prior attempt.
    assert not out.exists(), "merged output must not be on disk after ValueError"
