"""End-to-end tests for Cell Ranger MTX ingest (`pyscx.from_mtx`).

Cell Ranger writes `matrix.mtx` as **features × barcodes** (size line
`<genes> <cells> <nnz>`); SCX always stores cells × genes, so the reader
transposes. These tests lock in the orientation behaviour through the full
`from_mtx -> open -> to_anndata` path (the Rust tests in
`scx-mtx/tests/round_trip.rs` cover the CSR level).
"""

import gzip
import os

import numpy as np
import pytest

import pyscx


def _write_gz(path, data: bytes):
    with gzip.open(path, "wb") as fh:
        fh.write(data)


def write_cellranger_mtx(dir_path, mtx_body: str, barcodes, features):
    """Write a Cell Ranger v3-style MTX directory (all gzipped)."""
    os.makedirs(dir_path, exist_ok=True)
    _write_gz(os.path.join(dir_path, "matrix.mtx.gz"), mtx_body.encode())
    _write_gz(
        os.path.join(dir_path, "barcodes.tsv.gz"),
        ("\n".join(barcodes) + "\n").encode(),
    )
    # features.tsv: id, name, feature_type (tab-separated)
    feat_lines = [f"{fid}\t{name}\tGene Expression" for fid, name in features]
    _write_gz(
        os.path.join(dir_path, "features.tsv.gz"),
        ("\n".join(feat_lines) + "\n").encode(),
    )


def _to_dense(adata):
    x = adata.X
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def test_features_by_barcodes_to_anndata(tmp_path):
    """A genuine 4-genes × 3-cells Cell Ranger matrix reads back as a
    3-cells × 4-genes AnnData, with values correctly transposed and
    obs/var names attached to the right axes."""
    mtx_dir = tmp_path / "filtered_feature_bc_matrix"
    # 4 genes × 3 cells (features × barcodes); logical cells×genes matrix:
    #   (c0,g1)=1, (c0,g3)=2, (c1,g0)=3, (c2,g1)=4, (c2,g2)=5
    mtx_body = (
        "%%MatrixMarket matrix coordinate integer general\n"
        "4 3 5\n"
        "1 2 3\n"  # gene0, cell1 -> 3
        "2 1 1\n"  # gene1, cell0 -> 1
        "2 3 4\n"  # gene1, cell2 -> 4
        "3 3 5\n"  # gene2, cell2 -> 5
        "4 1 2\n"  # gene3, cell0 -> 2
    )
    barcodes = ["AAACCCAA-1", "BBBDDDBB-1", "CCCEEECC-1"]
    features = [
        ("ENSG001", "GeneA"),
        ("ENSG002", "GeneB"),
        ("ENSG003", "GeneC"),
        ("ENSG004", "GeneD"),
    ]
    write_cellranger_mtx(mtx_dir, mtx_body, barcodes, features)

    out = str(tmp_path / "out.scx")
    pyscx.from_mtx(str(mtx_dir), out)

    adata = pyscx.open(out).to_anndata()

    # Cells × genes orientation.
    assert adata.shape == (3, 4)
    assert list(adata.obs_names) == barcodes
    assert list(adata.var_names) == [f[0] for f in features]

    # Values survived the transpose into the right cells×genes cells.
    dense = _to_dense(adata)
    expected = np.array(
        [
            [0, 1, 0, 2],  # c0: g1=1, g3=2
            [3, 0, 0, 0],  # c1: g0=3
            [0, 4, 5, 0],  # c2: g1=4, g2=5
        ],
        dtype=dense.dtype,
    )
    np.testing.assert_array_equal(dense, expected)


def test_square_matrix_warns(tmp_path):
    """A square matrix is orientation-ambiguous; `from_mtx` assumes the
    Cell Ranger default and emits a catchable UserWarning."""
    mtx_dir = tmp_path / "square_mtx"
    # 2 × 2 with 2 barcodes + 2 features -> both interpretations fit.
    mtx_body = (
        "%%MatrixMarket matrix coordinate integer general\n"
        "2 2 2\n"
        "1 1 7\n"
        "2 2 9\n"
    )
    write_cellranger_mtx(
        mtx_dir,
        mtx_body,
        ["AAA-1", "BBB-1"],
        [("ENSG001", "GeneA"), ("ENSG002", "GeneB")],
    )

    out = str(tmp_path / "square.scx")
    with pytest.warns(UserWarning, match="ambiguous"):
        pyscx.from_mtx(str(mtx_dir), out)


def test_orientation_mismatch_raises(tmp_path):
    """A matrix whose dimensions match neither layout fails loud."""
    mtx_dir = tmp_path / "bad_mtx"
    # 2 × 2 matrix, but 3 barcodes and 4 features -> matches neither.
    mtx_body = (
        "%%MatrixMarket matrix coordinate integer general\n"
        "2 2 1\n"
        "1 1 7\n"
    )
    write_cellranger_mtx(
        mtx_dir,
        mtx_body,
        ["A-1", "B-1", "C-1"],
        [
            ("G1", "g1"),
            ("G2", "g2"),
            ("G3", "g3"),
            ("G4", "g4"),
        ],
    )

    out = str(tmp_path / "bad.scx")
    with pytest.raises(Exception, match="orientation mismatch"):
        pyscx.from_mtx(str(mtx_dir), out)


def _mtx_dir_with(tmp_path, n_cells):
    """A features × barcodes fixture with `n_cells` cells and 4 genes."""
    entries = [(1 + (i % 4), 1 + i, 1 + (i % 5)) for i in range(n_cells)]
    body = (
        "%%MatrixMarket matrix coordinate integer general\n"
        f"4 {n_cells} {len(entries)}\n"
        + "".join(f"{g} {c} {v}\n" for g, c, v in entries)
    )
    d = tmp_path / "fbm"
    write_cellranger_mtx(
        d,
        body,
        [f"CELL{i:04d}-1" for i in range(n_cells)],
        [(f"ENSG{i:03d}", f"Gene{i}") for i in range(4)],
    )
    return d


def test_from_mtx_shard_obs_policy(tmp_path):
    """`shard_obs` reaches `scx-mtx`.

    Until phase 6c's review round the flag was accepted on this route and did
    nothing: `run_convert` returned through the MTX dispatch before the policy
    was parsed, so `always` exited 0 having written one legacy `obs_metadata`
    section. Each assertion is paired with a control that must *not* shard, so
    "honoured" is distinguishable from "ignored".
    """
    mtx_dir = _mtx_dir_with(tmp_path, 12)

    for kwargs, want in (
        ({"shard_size": 100, "shard_obs": "always"}, 1),
        ({"shard_size": 100}, 0),  # auto, below threshold -> single section
        ({"shard_size": 4}, 3),  # auto, above threshold -> sharded
        ({"shard_size": 4, "shard_obs": "off"}, 0),
    ):
        out = str(tmp_path / f"out_{want}_{kwargs.get('shard_obs', 'auto')}.scx")
        pyscx.from_mtx(str(mtx_dir), out, **kwargs)
        exp = pyscx.open(out)
        assert exp.n_obs == 12
        assert exp.obs_metadata_shard_count == want, (
            f"from_mtx({kwargs}) -> {exp.obs_metadata_shard_count} obs shards, want {want}"
        )


def test_from_mtx_rejects_bad_shard_obs(tmp_path):
    mtx_dir = _mtx_dir_with(tmp_path, 8)
    with pytest.raises(ValueError, match="shard_obs"):
        pyscx.from_mtx(str(mtx_dir), str(tmp_path / "bad.scx"), shard_obs="sometimes")


# ---------------------------------------------------------------------------
# `integer` headers are taken at their word, and the MTX export is
# modality-scoped and streams shard by shard.
# ---------------------------------------------------------------------------


def _big_count_dir(tmp_path, value):
    """3 genes x 2 cells — deliberately non-square, so the orientation is
    unambiguous and the only warning a test can observe is the one it is
    asserting on."""
    d = tmp_path / "big_mtx"
    write_cellranger_mtx(
        d,
        "%%MatrixMarket matrix coordinate integer general\n"
        "3 2 2\n"
        "1 1 1\n"
        f"3 2 {value}\n",
        ["AAAC-1", "BBBC-1"],
        [("ENSG001", "GeneA"), ("ENSG002", "GeneB"), ("ENSG003", "GeneC")],
    )
    return d


def test_from_mtx_refuses_a_count_past_2_24(tmp_path):
    """`is_integer` used to be parsed for header validation and dropped, so
    every value went through `float32` and 16777217 silently became 16777216.

    The pipeline is f32 end to end, so the honest answer is to refuse rather
    than round quietly — the same posture the *read* side already takes."""
    mtx_dir = _big_count_dir(tmp_path, 16777217)
    out = tmp_path / "big.scx"
    with pytest.raises(RuntimeError, match="16777217"):
        pyscx.from_mtx(str(mtx_dir), str(out))
    assert not out.exists()


def test_from_mtx_allow_lossy_accepts_the_rounding_with_a_warning(tmp_path):
    mtx_dir = _big_count_dir(tmp_path, 16777217)
    out = tmp_path / "big_lossy.scx"
    with pytest.warns(UserWarning, match="rounded"):
        pyscx.from_mtx(str(mtx_dir), str(out), allow_lossy=True)
    adata = pyscx.open(str(out)).to_anndata()
    assert _to_dense(adata).max() == pytest.approx(16777216.0)


def test_from_mtx_at_the_2_24_boundary_is_unchanged(tmp_path):
    """Premise assertion: the guard fires *above* the limit, not at it. Without
    this the test above could pass against a gate that refused every count."""
    mtx_dir = _big_count_dir(tmp_path, 16777216)
    out = tmp_path / "boundary.scx"
    pyscx.from_mtx(str(mtx_dir), str(out))
    adata = pyscx.open(str(out)).to_anndata()
    assert _to_dense(adata).max() == pytest.approx(16777216.0)


def test_to_mtx_round_trips_and_accepts_modality_none(tmp_path):
    """The export streams shard by shard now; the observable contract is that a
    single-modality file still round-trips through it unchanged."""
    mtx_dir = tmp_path / "rt_in"
    write_cellranger_mtx(
        mtx_dir,
        "%%MatrixMarket matrix coordinate integer general\n"
        "4 3 5\n"
        "1 2 3\n"
        "2 1 1\n"
        "2 3 4\n"
        "3 3 5\n"
        "4 1 2\n",
        ["AAAC-1", "BBBC-1", "CCCC-1"],
        [("ENSG1", "A"), ("ENSG2", "B"), ("ENSG3", "C"), ("ENSG4", "D")],
    )
    scx_path = tmp_path / "rt.scx"
    pyscx.from_mtx(str(mtx_dir), str(scx_path))
    before = _to_dense(pyscx.open(str(scx_path)).to_anndata())

    out_dir = tmp_path / "rt_out"
    pyscx.to_mtx(str(scx_path), str(out_dir), modality=None)
    scx_again = tmp_path / "rt2.scx"
    pyscx.from_mtx(str(out_dir), str(scx_again))
    np.testing.assert_array_equal(
        _to_dense(pyscx.open(str(scx_again)).to_anndata()), before
    )


def test_to_mtx_rejects_a_modality_on_a_single_modality_file(tmp_path):
    mtx_dir = tmp_path / "single_in"
    write_cellranger_mtx(
        mtx_dir,
        "%%MatrixMarket matrix coordinate integer general\n3 2 1\n1 1 1\n",
        ["AAAC-1", "BBBC-1"],
        [("ENSG1", "A"), ("ENSG2", "B"), ("ENSG3", "C")],
    )
    scx_path = tmp_path / "single.scx"
    pyscx.from_mtx(str(mtx_dir), str(scx_path))
    with pytest.raises(RuntimeError, match="single-modality"):
        pyscx.to_mtx(str(scx_path), str(tmp_path / "out"), modality="rna")


def _mtx_fixture(tmp_path):
    mtx_dir = tmp_path / "mtx_csc"
    mtx_body = (
        "%%MatrixMarket matrix coordinate integer general\n"
        "4 3 5\n"
        "1 2 3\n"
        "2 1 1\n"
        "2 3 4\n"
        "3 3 5\n"
        "4 1 2\n"
    )
    barcodes = ["AAACCCAA-1", "BBBDDDBB-1", "CCCEEECC-1"]
    features = [(f"ENSG00{i}", f"Gene{i}") for i in range(4)]
    write_cellranger_mtx(mtx_dir, mtx_body, barcodes, features)
    return mtx_dir


@pytest.mark.parametrize("csc, expected", [(None, True), ("off", False), ("always", True)])
def test_from_mtx_follows_the_ingest_csc_default(tmp_path, monkeypatch, csc, expected):
    """`from_mtx` resolves `csc` like every other ingest entry point: unset
    is "auto" (the thresholds lowered to 0 so the tiny fixture clears it),
    "off" opts out, "always" builds regardless. It once took no `csc` at
    all and was CSR-only on any size."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")
    out = str(tmp_path / f"out_{csc}.scx")
    kwargs = {} if csc is None else {"csc": csc}
    pyscx.from_mtx(str(_mtx_fixture(tmp_path)), out, csc_cols_per_shard=2, **kwargs)
    assert pyscx.open(out).has_csc is expected


def test_from_mtx_auto_skips_below_the_threshold(tmp_path):
    """At the default thresholds a 3 x 4 matrix gets no sidecar."""
    out = str(tmp_path / "out.scx")
    pyscx.from_mtx(str(_mtx_fixture(tmp_path)), out)
    assert pyscx.open(out).has_csc is False


def test_from_mtx_budget_reaches_the_sidecar_build(tmp_path, monkeypatch):
    """`memory_budget` / `temp_dir` reach the sidecar post-pass: a valid budget
    still builds one, and an invalid budget is refused before anything is
    written rather than ignored."""
    monkeypatch.setenv("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
    monkeypatch.setenv("SCX_CSC_AUTO_VARS_THRESHOLD", "0")
    mtx_dir = _mtx_fixture(tmp_path)
    spill = tmp_path / "spill"
    spill.mkdir()
    out = str(tmp_path / "budgeted.scx")
    pyscx.from_mtx(str(mtx_dir), out, memory_budget="1M", temp_dir=str(spill))
    assert pyscx.open(out).has_csc is True

    bad = tmp_path / "bad.scx"
    with pytest.raises(ValueError):
        pyscx.from_mtx(str(mtx_dir), str(bad), memory_budget="10MB")
    assert not bad.exists()
