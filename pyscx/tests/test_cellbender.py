"""Tests for `pyscx.cellbender_import`.

The import's only real risk is the join. CellBender's `_filtered.h5` stores
rows in descending-UMI order, so these fixtures deliberately reverse the row
order relative to the SCX target: a positional import would put every cell's
corrected counts on the wrong barcode and still produce a correctly-shaped
layer.
"""

import pathlib

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
h5py = pytest.importorskip("h5py")


def _write_cellbender_h5(path, barcodes, gene_ids, value_for, *, cell_prob=True):
    """A CellRanger-v3-shaped CellBender output.

    Stored as CSC of [genes x cells], which is CSR of [cells x genes]; row `r`
    holds `value_for(barcode)` at gene `r % n_genes`.
    """
    n_cells, n_genes = len(barcodes), len(gene_ids)
    indptr, indices, data = [0], [], []
    for r, bc in enumerate(barcodes):
        indices.append(r % n_genes)
        data.append(float(value_for(bc)))
        indptr.append(len(indices))

    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("shape", data=np.array([n_genes, n_cells], dtype="i4"))
        m.create_dataset("indptr", data=np.array(indptr, dtype="i8"))
        m.create_dataset("indices", data=np.array(indices, dtype="i4"))
        m.create_dataset("data", data=np.array(data, dtype="f4"))
        m.create_dataset("barcodes", data=np.array(barcodes, dtype=h5py.string_dtype()))
        ft = m.create_group("features")
        ft.create_dataset("id", data=np.array(gene_ids, dtype=h5py.string_dtype()))
        ft.create_dataset(
            "name",
            data=np.array([g.upper() for g in gene_ids], dtype=h5py.string_dtype()),
        )

        dl = f.create_group("droplet_latents")
        if cell_prob:
            dl.create_dataset(
                "cell_probability",
                data=np.array([0.5 + 0.01 * i for i in range(n_cells)], dtype="f4"),
            )
            dl.create_dataset(
                "barcode_indices_for_latents",
                data=np.arange(n_cells, dtype="i8"),
            )
        gl = f.create_group("global_latents")
        gl.create_dataset(
            "ambient_expression",
            data=np.array([0.01 * i for i in range(n_genes)], dtype="f4"),
        )
        md = f.create_group("metadata")
        md.create_dataset("estimator", data=np.array(["mckp"], dtype=h5py.string_dtype()))
        md.create_dataset("target_false_positive_rate", data=np.array([0.01], dtype="f4"))


@pytest.fixture
def target(synthetic_adata, scx_from_adata):
    """An SCX file plus its barcodes, in the target's own row order."""
    path = scx_from_adata(synthetic_adata, "target.scx")
    return path, list(synthetic_adata.obs_names), list(synthetic_adata.var_names)


def test_round_trip_joins_by_barcode_despite_reversed_source_order(target, tmp_dir):
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    # Reversed, like CellBender's descending-UMI filtered output.
    _write_cellbender_h5(cb, list(reversed(barcodes)), genes, lambda bc: barcodes.index(bc) + 1)

    summary = pyscx.cellbender_import(str(path), str(cb))
    assert summary["n_matched"] == len(barcodes)
    assert summary["n_target_rows_absent"] == 0
    assert summary["obs_key_column"]
    assert summary["gene_axis_match"] == "identical"

    adata = pyscx.open(path).to_anndata()
    layer = adata.layers["cellbender"]
    assert layer.shape == adata.shape

    dense = layer.toarray()
    for row in range(len(barcodes)):
        nz = dense[row].nonzero()[0]
        assert len(nz) == 1
        assert dense[row, nz[0]] == pytest.approx(row + 1), (
            f"row {row} received another cell's corrected counts"
        )


def test_emits_obs_var_and_uns_diagnostics(target, tmp_dir):
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    _write_cellbender_h5(cb, barcodes, genes, lambda bc: 1)

    pyscx.cellbender_import(str(path), str(cb))
    adata = pyscx.open(path).to_anndata()

    assert (adata.obs["cellbender_status"] == "present").all()
    assert adata.obs["cellbender_cell_probability"].notna().all()
    assert "cellbender_total_counts" in adata.obs
    assert len(adata.var["cellbender_ambient_expression"]) == adata.n_vars

    note = adata.uns["cellbender"]
    assert note["version"] == 1
    assert note["estimator"] == "mckp"
    assert note["n_rows_in_source"] == len(barcodes)


def test_unmatched_target_rows_are_null_not_zero(target, tmp_dir):
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    subset = barcodes[::2]
    _write_cellbender_h5(cb, subset, genes, lambda bc: 5)

    summary = pyscx.cellbender_import(str(path), str(cb))
    assert summary["n_matched"] == len(subset)
    assert summary["n_target_rows_absent"] == len(barcodes) - len(subset)

    adata = pyscx.open(path).to_anndata()
    status = adata.obs["cellbender_status"].to_numpy()
    prob = adata.obs["cellbender_cell_probability"].to_numpy()
    assert status[0] == "present" and status[1] == "absent"
    # NaN, not 0.0: a probability of zero is a claim CellBender never made.
    assert not np.isnan(prob[0])
    assert np.isnan(prob[1])


def test_dry_run_reports_the_join_without_writing(target, tmp_dir):
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    _write_cellbender_h5(cb, barcodes, genes, lambda bc: 1)
    before = pathlib.Path(path).read_bytes()

    summary = pyscx.cellbender_import(str(path), str(cb), dry_run=True)
    assert summary["dry_run"] is True
    assert summary["n_matched"] == len(barcodes)
    assert pathlib.Path(path).read_bytes() == before
    assert "cellbender" not in pyscx.open(path).to_anndata().layers


def test_reimport_requires_overwrite(target, tmp_dir):
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    _write_cellbender_h5(cb, barcodes, genes, lambda bc: 1)

    pyscx.cellbender_import(str(path), str(cb))
    with pytest.raises(ValueError, match="already exists"):
        pyscx.cellbender_import(str(path), str(cb))

    summary = pyscx.cellbender_import(str(path), str(cb), overwrite=True)
    assert summary["n_matched"] == len(barcodes)
    # Exactly one `cellbender` layer, at the target's height — not a doubled
    # shard family. (The fixture also carries an unrelated `raw` layer, which
    # must be left alone.)
    adata = pyscx.open(path).to_anndata()
    assert sorted(adata.layers) == ["cellbender", "raw"]
    assert adata.layers["cellbender"].shape == adata.shape


def test_zero_overlap_raises_with_example_keys(target, tmp_dir):
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    _write_cellbender_h5(cb, [f"sampleA_{b}" for b in barcodes], genes, lambda bc: 1)

    with pytest.raises(ValueError) as exc:
        pyscx.cellbender_import(str(path), str(cb))
    msg = str(exc.value)
    assert "no target row key matched" in msg
    assert "sampleA_" in msg, "the error must show a source example"


def test_x_and_provenance_survive_and_rollback_undoes_the_import(target, tmp_dir):
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    _write_cellbender_h5(cb, barcodes, genes, lambda bc: 1)

    before = pyscx.open(path).to_anndata()
    pyscx.cellbender_import(str(path), str(cb))

    after = pyscx.open(path).to_anndata()
    np.testing.assert_allclose(after.X.toarray(), before.X.toarray())

    pyscx.rollback(str(path))
    rolled = pyscx.open(path).to_anndata()
    assert "cellbender" not in rolled.layers
    assert "cellbender_status" not in rolled.obs
    np.testing.assert_allclose(rolled.X.toarray(), before.X.toarray())


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("on_missing_rows", "nope"),
        ("on_extra_rows", "nope"),
        ("gene_axis", "nope"),
    ],
)
def test_invalid_enum_arguments_raise(target, tmp_dir, kwarg, value):
    """Document the accepted spellings — a typo must fail loudly rather than
    falling back to a default policy."""
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    _write_cellbender_h5(cb, barcodes, genes, lambda bc: 1)

    with pytest.raises(ValueError, match=kwarg):
        pyscx.cellbender_import(str(path), str(cb), **{kwarg: value})


def test_is_cellbender_h5_discriminates(target, tmp_dir):
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    _write_cellbender_h5(cb, barcodes, genes, lambda bc: 1)
    assert pyscx.is_cellbender_h5(str(cb))
    assert not pyscx.is_cellbender_h5(str(path))


def test_latent_embedding_is_opt_in(target, tmp_dir):
    path, barcodes, genes = target
    cb = tmp_dir / "cb.h5"
    _write_cellbender_h5(cb, barcodes, genes, lambda bc: 1)
    with h5py.File(cb, "a") as f:
        f["droplet_latents"].create_dataset(
            "gene_expression_encoding",
            data=np.arange(len(barcodes) * 4, dtype="f4").reshape(len(barcodes), 4),
        )

    pyscx.cellbender_import(str(path), str(cb))
    assert "X_cellbender_latent" not in pyscx.open(path).to_anndata().obsm

    pyscx.rollback(str(path))
    summary = pyscx.cellbender_import(str(path), str(cb), latent_embedding=True)
    assert "X_cellbender_latent" in summary["obsm_keys_added"]
    adata = pyscx.open(path).to_anndata()
    assert adata.obsm["X_cellbender_latent"].shape == (adata.n_obs, 4)
