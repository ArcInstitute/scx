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
