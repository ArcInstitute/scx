"""F2 — grouped-read API tests (read_group / read_reference / iter_group_shards).

The grouped layout is produced natively by `pyscx.sort(group_by=...)` (7.2a),
then these tests exercise the pyscx read surface — no `scx` subprocess.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def screen_adata():
    """A tiny perturbation-screen AnnData: target_gene grouping + a reference
    label, scattered so sorting must cluster them."""
    import anndata

    np.random.seed(7)
    genes = ["nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1"]
    n_obs, n_vars = len(genes), 6
    dense = np.random.randint(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    dense[np.random.random((n_obs, n_vars)) > 0.5] = 0
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"target_gene": pd.Categorical(genes)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=x, obs=obs, var=var), genes


def _sort_grouped(src, out, reference=None, shard_size=None):
    """Write a grouped SCX via the native pyscx sort (7.2a) — no subprocess."""
    import pyscx

    pyscx.sort(
        src,
        out,
        by=[],  # group_by becomes the leading key
        group_by="target_gene",
        reference=reference,
        shard_size=shard_size,
    )


def test_grouped_reads_roundtrip(screen_adata, scx_from_adata, tmp_dir):
    import pyscx

    adata, genes = screen_adata
    src = scx_from_adata(adata, "screen_src.scx")
    out = str(tmp_dir / "screen_grouped.scx")
    _sort_grouped(src, out, reference=["nt"])

    exp = pyscx.open(out)

    # group_labels covers every label.
    labels = set(exp.group_labels())
    assert {"nt", "MYC", "TP53", "GATA1"} <= labels

    # read_group("MYC") returns exactly the MYC cells.
    myc = exp.read_group("MYC")
    n_myc = sum(1 for g in genes if g == "MYC")
    assert myc.n_obs == n_myc
    assert myc.n_vars == 6
    assert sp.issparse(myc.X)
    assert all(myc.obs["target_gene"] == "MYC")

    # read_reference returns the nt cells.
    ref = exp.read_reference()
    assert ref is not None
    assert ref.n_obs == sum(1 for g in genes if g == "nt")
    assert all(ref.obs["target_gene"] == "nt")

    # iter_group_shards covers every non-reference label exactly once.
    seen = []
    for gs in exp.iter_group_shards():
        ad = gs.to_anndata()
        assert ad.n_obs == (gs.global_stop - gs.global_start)
        seen.extend(gs.labels)
    assert "nt" not in seen
    assert set(seen) == {"MYC", "TP53", "GATA1"}
    assert len(seen) == len(set(seen)), "no label may appear in two shards"

    # Unknown label → KeyError with suggestions.
    with pytest.raises(KeyError):
        exp.read_group("MYCN")


def test_grouped_read_reflects_mark_deleted(screen_adata, scx_from_adata, tmp_dir):
    """M5: a grouped read cached before mark_deleted must not keep returning
    just-deleted rows. mark_deleted resets the cached grouped pipeline (and the
    deleted-count cache), so a second read_group/read_reference on the same
    Experiment reflects the deletion."""
    import pyscx

    adata, genes = screen_adata
    src = scx_from_adata(adata, "screen_src.scx")
    out = str(tmp_dir / "screen_grouped.scx")
    _sort_grouped(src, out, reference=["nt"])

    exp = pyscx.open(out)

    # Populate the grouped-pipeline cache with the pre-deletion snapshot.
    n_myc_before = exp.read_group("MYC").n_obs
    n_ref_before = exp.read_reference().n_obs
    assert n_myc_before >= 1 and n_ref_before >= 1

    # Full-file (grouped/sorted) obs order → mask deleting one MYC + one nt row.
    tg = exp.to_anndata().obs["target_gene"].to_numpy()
    mask = np.zeros(len(tg), dtype=bool)
    mask[int(np.where(tg == "MYC")[0][0])] = True
    mask[int(np.where(tg == "nt")[0][0])] = True
    exp.mark_deleted(mask)

    # Same object: caches invalidated → deletion reflected (would still equal
    # *_before on the pre-fix stale-cache code).
    assert exp.read_group("MYC").n_obs == n_myc_before - 1
    assert exp.read_reference().n_obs == n_ref_before - 1

    # Cross-check against a fresh open of the mutated file.
    fresh = pyscx.open(out)
    assert fresh.read_group("MYC").n_obs == n_myc_before - 1
    assert fresh.read_reference().n_obs == n_ref_before - 1


def test_group_shard_per_label_reads(screen_adata, scx_from_adata, tmp_dir):
    """7.1d: GroupShard exposes per-label shard-local ranges and can read a
    single label out of a multi-group shard."""
    import pyscx

    adata, _genes = screen_adata
    src = scx_from_adata(adata, "screen_src.scx")
    out = str(tmp_dir / "screen_grouped.scx")
    # Large shards so several labels share one shard (exercises per-label slicing).
    _sort_grouped(src, out, reference=["nt"], shard_size=100)

    exp = pyscx.open(out)
    shards = exp.iter_group_shards()
    assert shards, "expected at least one non-reference shard"

    for gs in shards:
        groups = gs.groups  # dict[label] -> (local_start, local_stop)
        assert set(groups) == set(gs.labels)
        for label, (ls, le) in groups.items():
            # Local ranges are within the shard and ordered.
            assert 0 <= ls < le <= (gs.global_stop - gs.global_start)
            # Reading one label out of the shard matches the whole-file read_group.
            per_label = gs.read_group(label)
            whole = exp.read_group(label)
            assert per_label.n_obs == whole.n_obs == (le - ls)
            assert all(per_label.obs["target_gene"] == label)

    # Unknown label in a shard → KeyError.
    with pytest.raises(KeyError):
        shards[0].read_group("not_a_label")


def test_group_shard_stream_matches_whole_file(screen_adata, scx_from_adata, tmp_dir):
    """7.1c: streaming over iter_group_shards (shared pipeline) plus the
    reference reconstructs the full sorted file."""
    import pyscx

    adata, genes = screen_adata
    src = scx_from_adata(adata, "screen_src.scx")
    out = str(tmp_dir / "screen_grouped.scx")
    _sort_grouped(src, out, reference=["nt"], shard_size=3)

    exp = pyscx.open(out)
    # Sum of all non-reference shard rows + reference rows == total cells.
    shard_rows = sum(gs.to_anndata().n_obs for gs in exp.iter_group_shards())
    ref = exp.read_reference()
    ref_rows = 0 if ref is None else ref.n_obs
    assert shard_rows + ref_rows == len(genes)


def test_sort_reference_column_spec(scx_from_adata, tmp_dir):
    """7.2a: reference={'column': name} (boolean obs column) is accepted and
    isolates the flagged rows as the reference region."""
    import anndata

    import pyscx

    genes = ["nt", "MYC", "nt", "TP53", "MYC", "nt"]
    is_control = [g == "nt" for g in genes]
    n_obs, n_vars = len(genes), 5
    x = sp.csr_matrix(np.random.RandomState(3).randint(0, 9, (n_obs, n_vars)).astype(np.float32))
    obs = pd.DataFrame(
        {"target_gene": pd.Categorical(genes), "is_control": is_control},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    src = scx_from_adata(anndata.AnnData(X=x, obs=obs, var=var), "col_src.scx")
    out = str(tmp_dir / "col_grouped.scx")

    pyscx.sort(
        src, out, by=[], group_by="target_gene", reference={"column": "is_control"}
    )
    exp = pyscx.open(out)
    ref = exp.read_reference()
    assert ref is not None
    assert ref.n_obs == sum(is_control)
    assert all(ref.obs["target_gene"] == "nt")


def test_sort_reference_requires_group_by(screen_adata, scx_from_adata, tmp_dir):
    """7.2a: reference without group_by is a clean ValueError."""
    import pyscx

    adata, _ = screen_adata
    src = scx_from_adata(adata, "screen_src.scx")
    out = str(tmp_dir / "bad.scx")
    with pytest.raises(ValueError):
        pyscx.sort(src, out, by=["target_gene"], reference=["nt"])


def test_read_group_on_ungrouped_errors(query_adata, scx_from_adata):
    import pyscx

    path = scx_from_adata(query_adata, "ungrouped.scx")
    exp = pyscx.open(path)
    with pytest.raises(ValueError):
        exp.group_labels()
    with pytest.raises(ValueError):
        exp.read_group("anything")
