"""Tests for pyscx cloud operations (pull, push, explode, pack, cloud_optimize).

These tests use local filesystem paths only (no actual cloud access).
Requires: maturin develop --features cloud
"""

import os
import tempfile
import numpy as np
import pytest
import scipy.sparse as sp

# Skip all tests if pyscx was built without cloud features
_pyscx = pytest.importorskip("pyscx")
if not hasattr(_pyscx, "explode"):
    pytest.skip("pyscx built without cloud features", allow_module_level=True)


def _create_test_scx(path: str, n_obs: int = 100, n_vars: int = 50):
    """Create a test .scx file using pyscx.from_anndata."""
    import anndata
    import pyscx

    rng = np.random.default_rng(42)
    X = sp.random(n_obs, n_vars, density=0.1, format="csr", dtype=np.float32,
                   random_state=rng)
    X.data = np.round(X.data * 10).astype(np.float32)

    obs_data = {
        "cell_id": [f"cell_{i}" for i in range(n_obs)],
        "cell_type": [["T_cell", "B_cell", "Monocyte"][i % 3] for i in range(n_obs)],
    }
    var_data = {"gene_id": [f"gene_{i}" for i in range(n_vars)]}

    adata = anndata.AnnData(
        X=X,
        obs=obs_data,
        var=var_data,
    )
    pyscx.from_anndata(adata, path, codec="none")


class TestPullFromLocalExploded:
    """Test: Python pull from local exploded directory."""

    def test_pull_returns_stats_dict(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path)

            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)

            pulled_path = os.path.join(tmpdir, "pulled.scx")
            stats = pyscx.pull(exploded_dir, pulled_path)

            assert isinstance(stats, dict)
            assert "bytes_downloaded" in stats
            assert "sections_downloaded" in stats
            assert "elapsed_secs" in stats
            assert stats["bytes_downloaded"] > 0
            assert stats["sections_downloaded"] > 0

    def test_pulled_file_readable(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path)

            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)

            pulled_path = os.path.join(tmpdir, "pulled.scx")
            pyscx.pull(exploded_dir, pulled_path)

            exp = pyscx.open(pulled_path)
            assert exp.n_obs == 100
            assert exp.n_vars == 50


class TestExplodePackRoundtrip:
    """Test: Python explode → pack round-trip."""

    def test_roundtrip_preserves_data(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_path = os.path.join(tmpdir, "original.scx")
            _create_test_scx(orig_path)

            # Read original data
            exp_orig = pyscx.open(orig_path)
            adata_orig = exp_orig.to_anndata()

            # Explode → pack
            exploded_dir = os.path.join(tmpdir, "exploded.scxd")
            pyscx.explode(orig_path, exploded_dir)

            packed_path = os.path.join(tmpdir, "packed.scx")
            pyscx.pack(exploded_dir, packed_path)

            # Read round-tripped data
            exp_rt = pyscx.open(packed_path)
            adata_rt = exp_rt.to_anndata()

            assert adata_orig.n_obs == adata_rt.n_obs
            assert adata_orig.n_vars == adata_rt.n_vars

    def test_exploded_directory_has_expected_files(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path)

            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)

            assert os.path.exists(os.path.join(exploded_dir, "_catalog.bin"))
            assert os.path.exists(os.path.join(exploded_dir, "_header.bin"))
            assert os.path.exists(os.path.join(exploded_dir, "obs.arrow"))
            assert os.path.exists(os.path.join(exploded_dir, "var.arrow"))
            assert os.path.isdir(os.path.join(exploded_dir, "X"))


class TestCloudOptimize:
    """Test: Python cloud_optimize produces cloud-ready file."""

    def test_cloud_optimize_with_output(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path)

            optimized_path = os.path.join(tmpdir, "cloud_ready.scx")
            pyscx.cloud_optimize(scx_path, optimized_path)

            exp = pyscx.open(optimized_path)
            assert exp.n_obs == 100
            assert exp.n_vars == 50

    def test_cloud_optimize_in_place(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path)

            orig_size = os.path.getsize(scx_path)
            pyscx.cloud_optimize(scx_path)

            # File should still be readable after in-place optimize
            exp = pyscx.open(scx_path)
            assert exp.n_obs == 100
            assert exp.n_vars == 50

            # File should be larger due to front catalog
            new_size = os.path.getsize(scx_path)
            assert new_size >= orig_size


class TestPushPull:
    """Test: push → pull round-trip via Python bindings."""

    def test_push_returns_stats(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path)

            dest_dir = os.path.join(tmpdir, "pushed.scxd")
            os.makedirs(dest_dir)
            stats = pyscx.push(scx_path, dest_dir)

            assert isinstance(stats, dict)
            assert "bytes_uploaded" in stats
            assert "sections_uploaded" in stats
            assert stats["bytes_uploaded"] > 0

    def test_push_pull_roundtrip(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path)

            dest_dir = os.path.join(tmpdir, "pushed.scxd")
            os.makedirs(dest_dir)
            pyscx.push(scx_path, dest_dir)

            pulled_path = os.path.join(tmpdir, "pulled.scx")
            pyscx.pull(dest_dir, pulled_path)

            exp_orig = pyscx.open(scx_path)
            exp_pulled = pyscx.open(pulled_path)
            assert exp_orig.n_obs == exp_pulled.n_obs
            assert exp_orig.n_vars == exp_pulled.n_vars


class TestCloudReader:
    """Test: pyscx.open_cloud() reads header and catalog correctly."""

    def test_open_cloud_exploded_reads_header(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path)

            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)

            exp = pyscx.open_cloud(exploded_dir)
            assert exp.n_obs == 100
            assert exp.n_vars == 50

    def test_open_cloud_n_obs_returns_correct_count(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path, n_obs=200, n_vars=75)

            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)

            exp = pyscx.open_cloud(exploded_dir)
            assert exp.n_obs == 200
            assert exp.n_vars == 75

    def test_open_cloud_works_with_exploded_and_packed(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path)

            # Exploded
            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)
            exp_exploded = pyscx.open_cloud(exploded_dir)

            # Packed (non-cloud-ready)
            exp_packed = pyscx.open_cloud(scx_path)

            # Both should give same results
            assert exp_exploded.n_obs == exp_packed.n_obs
            assert exp_exploded.n_vars == exp_packed.n_vars
            assert exp_exploded.shard_count == exp_packed.shard_count

    def test_open_cloud_shape_and_keys(self):
        """D1: CloudExperiment exposes `.shape` and obs/var column accessors
        so users can discover the filter_obs vocabulary without collecting."""
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path, n_obs=120, n_vars=30)
            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)

            exp = pyscx.open_cloud(exploded_dir)
            assert exp.shape == (exp.n_obs, exp.n_vars) == (120, 30)
            assert "cell_type" in exp.obs_keys()
            assert "gene_id" in exp.var_keys()
            # Pandas index column is excluded, mirroring adata.obs.columns.
            assert "_index" not in exp.obs_keys()

    def test_open_cloud_missing_path_raises_filenotfound(self):
        """E1: a missing/wrong cloud path raises FileNotFoundError with an
        actionable message, not a raw object_store 404."""
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "does_not_exist.scxd")
            with pytest.raises(FileNotFoundError, match="no SCX data found"):
                pyscx.open_cloud(missing)


class TestCloudQuery:
    """Phase 7: pyscx.open_cloud(...).query().filter_obs(...).collect()
    must produce the same AnnData (X, obs, var) as the local
    pyscx.open(...).query().filter_obs(...).collect() pipeline on
    equivalent files. Verifies the SectionReader unification."""

    def test_query_filter_obs_matches_local_exploded(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path, n_obs=120, n_vars=40)

            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)

            local_adata = (
                pyscx.open(scx_path)
                .query()
                .filter_obs("cell_type == 'T_cell'")
                .collect()
                .to_anndata()
            )
            cloud_adata = (
                pyscx.open_cloud(exploded_dir)
                .query()
                .filter_obs("cell_type == 'T_cell'")
                .collect()
                .to_anndata()
            )

            assert cloud_adata.n_obs == local_adata.n_obs
            assert cloud_adata.n_vars == local_adata.n_vars
            assert cloud_adata.n_obs > 0
            # X matrices: same shape and nonzero structure
            np.testing.assert_array_equal(
                cloud_adata.X.toarray(), local_adata.X.toarray()
            )
            # obs cell_id column matches row-for-row
            assert list(cloud_adata.obs["cell_id"]) == list(local_adata.obs["cell_id"])

    def test_query_select_genes_matches_local(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path, n_obs=80, n_vars=30)

            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)

            genes = [0, 3, 7, 15, 22]
            local_adata = (
                pyscx.open(scx_path)
                .query()
                .filter_obs("cell_type == 'B_cell'")
                .select_genes(genes)
                .collect()
                .to_anndata()
            )
            cloud_adata = (
                pyscx.open_cloud(exploded_dir)
                .query()
                .filter_obs("cell_type == 'B_cell'")
                .select_genes(genes)
                .collect()
                .to_anndata()
            )

            assert cloud_adata.n_vars == 5
            assert cloud_adata.n_obs == local_adata.n_obs
            np.testing.assert_array_equal(
                cloud_adata.X.toarray(), local_adata.X.toarray()
            )
            assert list(cloud_adata.var["gene_id"]) == list(local_adata.var["gene_id"])

    def test_query_against_packed_cloud_ready_file(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path, n_obs=100, n_vars=50)

            optimized_path = os.path.join(tmpdir, "optimized.scx")
            pyscx.cloud_optimize(scx_path, optimized_path)

            local_adata = (
                pyscx.open(scx_path)
                .query()
                .filter_obs("cell_type == 'Monocyte'")
                .collect()
                .to_anndata()
            )
            cloud_adata = (
                pyscx.open_cloud(optimized_path)
                .query()
                .filter_obs("cell_type == 'Monocyte'")
                .collect()
                .to_anndata()
            )

            assert cloud_adata.n_obs == local_adata.n_obs
            np.testing.assert_array_equal(
                cloud_adata.X.toarray(), local_adata.X.toarray()
            )

    def test_query_reports_total_shards(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "test.scx")
            _create_test_scx(scx_path, n_obs=120, n_vars=30)

            exploded_dir = os.path.join(tmpdir, "test.scxd")
            pyscx.explode(scx_path, exploded_dir)

            local_result = (
                pyscx.open(scx_path)
                .query()
                .filter_obs("cell_type == 'T_cell'")
                .collect()
            )
            cloud_result = (
                pyscx.open_cloud(exploded_dir)
                .query()
                .filter_obs("cell_type == 'T_cell'")
                .collect()
            )

            assert cloud_result.total_shards == local_result.total_shards
            assert cloud_result.skipped_shards == local_result.skipped_shards


def _create_multimodal_scx(path: str):
    """Build a tiny CITE-seq (rna + adt) multimodal .scx via from_mudata."""
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")
    import pyscx

    rng = np.random.default_rng(0)
    n_obs, rna_n_vars, adt_n_vars = 24, 40, 8
    rna = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(0.4, size=(n_obs, rna_n_vars)).astype(np.float32))
    )
    rna.var_names = [f"g{i}" for i in range(rna_n_vars)]
    adt = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(0.4, size=(n_obs, adt_n_vars)).astype(np.float32))
    )
    adt.var_names = [f"a{i}" for i in range(adt_n_vars)]
    mu = mudata.MuData({"rna": rna, "adt": adt})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]
    pyscx.from_mudata(mu, path)


class TestCloudMultimodalDiscoverability:
    """B5: a multimodal file opened via open_cloud must expose its
    modalities (is_multimodal / modality_names / modality_info) instead
    of silently projecting to the primary (rna) modality."""

    def test_open_cloud_exposes_modalities(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "cite.scx")
            _create_multimodal_scx(scx_path)
            exploded_dir = os.path.join(tmpdir, "cite.scxd")
            pyscx.explode(scx_path, exploded_dir)

            exp = pyscx.open_cloud(exploded_dir)
            assert exp.is_multimodal is True
            assert exp.n_modalities == 2
            assert set(exp.modality_names) == {"rna", "adt"}

            # Parity with the local Experiment accessors.
            local = pyscx.open(scx_path)
            assert exp.modality_names == local.modality_names
            for name in ("rna", "adt"):
                mid = exp.modality_id(name)
                assert mid == local.modality_id(name)
                info = exp.modality_info(mid)
                assert info["name"] == name
                assert info["n_vars"] == local.modality_info(mid)["n_vars"]
                assert info["nnz"] == local.modality_info(mid)["nnz"]

            # Multimodality is visible in the repr (not silently hidden).
            assert "multimodal" in repr(exp)

    def test_open_cloud_single_modality_not_multimodal(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "single.scx")
            _create_test_scx(scx_path, n_obs=40, n_vars=20)
            exploded_dir = os.path.join(tmpdir, "single.scxd")
            pyscx.explode(scx_path, exploded_dir)

            exp = pyscx.open_cloud(exploded_dir)
            assert exp.is_multimodal is False
            assert exp.n_modalities == 0
            assert exp.modality_names == []
            assert exp.modality_id("rna") is None
            assert exp.modality_info(1) is None

    def test_cloud_to_mudata_raises_pull_locally(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "cite.scx")
            _create_multimodal_scx(scx_path)
            exploded_dir = os.path.join(tmpdir, "cite.scxd")
            pyscx.explode(scx_path, exploded_dir)

            exp = pyscx.open_cloud(exploded_dir)
            with pytest.raises(RuntimeError, match="pull the file locally"):
                exp.to_mudata()


def _create_grouped_exploded(tmpdir):
    """Build a grouped .scx (via native pyscx.sort, 7.2a) and explode it to a
    .scxd, returning (grouped_scx_path, exploded_dir)."""
    import anndata
    import pandas as pd
    import pyscx

    genes = ["nt", "MYC", "nt", "TP53", "MYC", "GATA1", "nt", "MYC", "GATA1", "GATA1", "TP53", "nt"]
    n_obs, n_vars = len(genes), 8
    rng = np.random.default_rng(7)
    dense = rng.integers(0, 20, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.6] = 0
    X = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"target_gene": pd.Categorical(genes), "cell_id": [f"cell_{i}" for i in range(n_obs)]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame({"gene_id": [f"gene_{i}" for i in range(n_vars)]},
                       index=[f"gene_{i}" for i in range(n_vars)])
    src = os.path.join(tmpdir, "src.scx")
    pyscx.from_anndata(anndata.AnnData(X=X, obs=obs, var=var), src, codec="none")
    grouped = os.path.join(tmpdir, "grouped.scx")
    pyscx.sort(src, grouped, by=[], group_by="target_gene", reference=["nt"], shard_size=3)
    exploded = os.path.join(tmpdir, "grouped.scxd")
    pyscx.explode(grouped, exploded)
    return grouped, exploded


class TestCloudGrouped:
    """Phase 7.3: open_cloud(...).read_group/read_reference/group_labels/
    iter_group_shards must match the local open(...) grouped reads on an
    equivalent exploded file (SectionReader unification over the cloud path)."""

    def test_read_group_matches_local(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            grouped, exploded = _create_grouped_exploded(tmpdir)
            local = pyscx.open(grouped).read_group("MYC")
            cloud = pyscx.open_cloud(exploded).read_group("MYC")
            assert cloud.n_obs == local.n_obs > 0
            np.testing.assert_array_equal(cloud.X.toarray(), local.X.toarray())
            assert list(cloud.obs["target_gene"]) == list(local.obs["target_gene"])
            assert list(cloud.obs["cell_id"]) == list(local.obs["cell_id"])
            assert all(cloud.obs["target_gene"] == "MYC")

    def test_read_reference_and_labels_match_local(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            grouped, exploded = _create_grouped_exploded(tmpdir)
            lref = pyscx.open(grouped).read_reference()
            cref = pyscx.open_cloud(exploded).read_reference()
            assert cref is not None and lref is not None
            assert cref.n_obs == lref.n_obs
            assert all(cref.obs["target_gene"] == "nt")
            assert set(pyscx.open_cloud(exploded).group_labels()) == set(
                pyscx.open(grouped).group_labels()
            )

    def test_iter_group_shards_matches_local(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            grouped, exploded = _create_grouped_exploded(tmpdir)
            exp = pyscx.open_cloud(exploded)
            shards = exp.iter_group_shards()
            assert shards
            total = 0
            seen = []
            for gs in shards:
                ad = gs.to_anndata()
                total += ad.n_obs
                seen.extend(gs.labels)
                lab = gs.labels[0]
                # Per-label read out of the cloud shard matches the whole-file read.
                assert gs.read_group(lab).n_obs == exp.read_group(lab).n_obs
            assert "nt" not in seen
            ref = exp.read_reference()
            assert total + (0 if ref is None else ref.n_obs) == exp.n_obs

    def test_unknown_label_and_ungrouped_errors(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            _, exploded = _create_grouped_exploded(tmpdir)
            with pytest.raises(KeyError):
                pyscx.open_cloud(exploded).read_group("NOPE")

            # Ungrouped exploded file → read_group raises ValueError (NotGrouped).
            plain = os.path.join(tmpdir, "plain.scx")
            _create_test_scx(plain, n_obs=30, n_vars=10)
            plain_exploded = os.path.join(tmpdir, "plain.scxd")
            pyscx.explode(plain, plain_exploded)
            with pytest.raises(ValueError):
                pyscx.open_cloud(plain_exploded).read_group("anything")


def _create_test_scx_with_uns(path: str, uns: dict | None, n_obs: int = 50, n_vars: int = 10):
    """Create a test .scx with an optional uns payload via from_anndata."""
    import anndata
    import pyscx

    rng = np.random.default_rng(7)
    X = sp.random(n_obs, n_vars, density=0.2, format="csr", dtype=np.float32,
                   random_state=rng)
    X.data = np.round(X.data * 10).astype(np.float32)
    adata = anndata.AnnData(
        X=X,
        obs={"cell_id": [f"cell_{i}" for i in range(n_obs)]},
        var={"gene_id": [f"gene_{i}" for i in range(n_vars)]},
    )
    if uns is not None:
        adata.uns.update(uns)
    pyscx.from_anndata(adata, path, codec="none")


class TestCloudReadUns:
    """Group A parity: CloudExperiment.read_uns() / uns_keys() must match
    the local Experiment on byte-identical (exploded) files."""

    def test_read_uns_parity_with_local(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "uns.scx")
            _create_test_scx_with_uns(scx_path, {"species": "human", "version": 2})
            exploded_dir = os.path.join(tmpdir, "uns.scxd")
            pyscx.explode(scx_path, exploded_dir)

            local = pyscx.open(scx_path)
            cloud = pyscx.open_cloud(exploded_dir)

            assert cloud.read_uns() == local.read_uns()
            assert set(cloud.uns_keys()) == set(local.uns_keys())
            assert cloud.read_uns()["species"] == "human"
            assert int(cloud.read_uns()["version"]) == 2

    def test_read_uns_dataframe_parity_with_local(self):
        """A `pandas.DataFrame` value decodes identically on the cloud path.

        The local and cloud readers share one decoder
        (`convert::uns::json_value_to_pyobject`), so this is really a check that
        the sharing still holds — but it is the arm that would catch a decoder
        wired into only one of them, and the parity test above cannot: `==` on
        two frames returns a *frame*, not a bool.
        """
        import pandas as pd

        import pyscx

        df = pd.DataFrame(
            {"v": [1.0, 2.0], "c": pd.Categorical(["a", "b"], ordered=True)},
            index=pd.Index(["r0", "r1"], name="row"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "uns_frame.scx")
            _create_test_scx_with_uns(scx_path, {"frame": df})
            exploded_dir = os.path.join(tmpdir, "uns_frame.scxd")
            pyscx.explode(scx_path, exploded_dir)

            local = pyscx.open(scx_path).read_uns()["frame"]
            cloud = pyscx.open_cloud(exploded_dir).read_uns()["frame"]

            assert isinstance(cloud, pd.DataFrame)
            pd.testing.assert_frame_equal(cloud, df, check_dtype=True)
            pd.testing.assert_frame_equal(cloud, local, check_dtype=True)

    def test_read_uns_none_when_absent(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "no_uns.scx")
            _create_test_scx_with_uns(scx_path, None)
            exploded_dir = os.path.join(tmpdir, "no_uns.scxd")
            pyscx.explode(scx_path, exploded_dir)

            cloud = pyscx.open_cloud(exploded_dir)
            assert cloud.read_uns() is None
            assert cloud.uns_keys() == []

    def test_read_uns_packed_file(self):
        """Works on a packed .scx (not just an exploded .scxd/ dir)."""
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "uns.scx")
            _create_test_scx_with_uns(scx_path, {"k": "v"})
            cloud = pyscx.open_cloud(scx_path)
            assert cloud.read_uns() == {"k": "v"}
            assert cloud.uns_keys() == ["k"]

    def test_read_uns_multimodal_modality_resolution(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "cite.scx")
            _create_multimodal_scx(scx_path)
            exploded_dir = os.path.join(tmpdir, "cite.scxd")
            pyscx.explode(scx_path, exploded_dir)

            cloud = pyscx.open_cloud(exploded_dir)
            assert set(cloud.modality_names) == {"rna", "adt"}

            # from_mudata writes no global / per-modality uns here, so known
            # modalities resolve and return None (proving the modality_id ->
            # read_uns_for path is wired), and unknown names raise KeyError.
            assert cloud.read_uns() is None
            assert cloud.read_uns(modality="rna") is None
            assert cloud.uns_keys(modality="adt") == []
            with pytest.raises(KeyError):
                cloud.read_uns(modality="does_not_exist")
            with pytest.raises(KeyError):
                cloud.uns_keys(modality="does_not_exist")


class TestCloudIntrospectionParity:
    """Group B parity: header-only introspection accessors on
    CloudExperiment must match the local Experiment on byte-identical
    (exploded / packed) files. All are O(1) header reads, no network I/O."""

    def test_header_getters_match_local(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "x.scx")
            _create_test_scx(scx_path, n_obs=80, n_vars=30)
            exploded_dir = os.path.join(tmpdir, "x.scxd")
            pyscx.explode(scx_path, exploded_dir)

            local = pyscx.open(scx_path)
            cloud = pyscx.open_cloud(exploded_dir)

            assert cloud.has_csc == local.has_csc
            assert cloud.has_deletions == local.has_deletions
            assert cloud.index_dtype == local.index_dtype
            assert cloud.n_obs_physical == local.n_obs_physical
            # A default file has no deletions, so physical == logical.
            assert cloud.n_obs_physical == cloud.n_obs == 80

    def test_path_returns_opened_url(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "x.scx")
            _create_test_scx(scx_path, n_obs=40, n_vars=20)
            exploded_dir = os.path.join(tmpdir, "x.scxd")
            pyscx.explode(scx_path, exploded_dir)

            # Both packed-file and exploded-dir layouts round-trip the URL.
            assert pyscx.open_cloud(exploded_dir).path == exploded_dir
            assert pyscx.open_cloud(scx_path).path == scx_path

    def test_info_string_and_prefix_parity(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "x.scx")
            _create_test_scx(scx_path, n_obs=60, n_vars=25)
            exploded_dir = os.path.join(tmpdir, "x.scxd")
            pyscx.explode(scx_path, exploded_dir)

            local = pyscx.open(scx_path)
            cloud = pyscx.open_cloud(exploded_dir)

            info = cloud.info()
            assert isinstance(info, str)
            for token in ("format_version=", "codec_id=", "nnz=", "has_csc=", "path="):
                assert token in info
            assert exploded_dir in info

            # Header fields are identical on byte-identical files; only the
            # trailing `path=...` differs. Compare the prefix up to `path=`.
            assert info.split(", path=")[0] == local.info().split(", path=")[0]

    def test_has_csc_true_with_sidecar(self):
        """A file written with a forced CSC sidecar reports has_csc=True,
        matching the local Experiment."""
        import anndata
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "csc.scx")
            rng = np.random.default_rng(3)
            X = sp.random(60, 20, density=0.3, format="csr", dtype=np.float32,
                          random_state=rng)
            X.data = np.round(X.data * 10).astype(np.float32)
            adata = anndata.AnnData(
                X=X,
                obs={"cell_id": [f"c{i}" for i in range(60)]},
                var={"gene_id": [f"g{i}" for i in range(20)]},
            )
            pyscx.from_anndata(adata, scx_path, codec="none", csc="always")

            local = pyscx.open(scx_path)
            exploded_dir = os.path.join(tmpdir, "csc.scxd")
            pyscx.explode(scx_path, exploded_dir)
            cloud = pyscx.open_cloud(exploded_dir)

            # Parity holds regardless of whether explode preserves the
            # sidecar; assert the local file actually has one so the
            # has_csc=True path is genuinely exercised.
            assert local.has_csc is True
            assert cloud.has_csc == local.has_csc


class TestCloudCatalogEnumParity:
    """Group C parity: CloudExperiment.obsm_keys() / varm_keys() /
    layer_names() must match the local Experiment on byte-identical files.
    All are pure in-memory catalog scans (catalog loaded at open)."""

    def test_enum_keys_match_local(self):
        import anndata
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "x.scx")
            rng = np.random.default_rng(11)
            X = sp.random(40, 12, density=0.3, format="csr", dtype=np.float32,
                          random_state=rng)
            X.data = np.round(X.data * 10).astype(np.float32)
            adata = anndata.AnnData(
                X=X,
                obs={"cell_id": [f"c{i}" for i in range(40)]},
                var={"gene_id": [f"g{i}" for i in range(12)]},
            )
            adata.obsm["X_pca"] = np.zeros((40, 5), dtype=np.float32)
            adata.obsm["X_umap"] = np.zeros((40, 2), dtype=np.float32)
            adata.varm["PCs"] = np.zeros((12, 5), dtype=np.float32)
            adata.layers["counts"] = adata.X.copy()
            pyscx.from_anndata(adata, scx_path, codec="none")

            exploded_dir = os.path.join(tmpdir, "x.scxd")
            pyscx.explode(scx_path, exploded_dir)

            local = pyscx.open(scx_path)
            cloud = pyscx.open_cloud(exploded_dir)

            assert set(cloud.obsm_keys()) == set(local.obsm_keys())
            assert set(cloud.varm_keys()) == set(local.varm_keys())
            assert set(cloud.layer_names()) == set(local.layer_names())
            # Sanity: the data we wrote is actually present.
            assert set(cloud.obsm_keys()) >= {"X_pca", "X_umap"}
            assert "PCs" in cloud.varm_keys()
            assert "counts" in cloud.layer_names()

    def test_enum_keys_empty_when_absent(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "x.scx")
            _create_test_scx(scx_path, n_obs=20, n_vars=8)
            exploded_dir = os.path.join(tmpdir, "x.scxd")
            pyscx.explode(scx_path, exploded_dir)

            cloud = pyscx.open_cloud(exploded_dir)
            assert cloud.obsm_keys() == []
            assert cloud.varm_keys() == []
            assert cloud.layer_names() == []


def _create_multimodal_scx(path: str, n_obs: int = 40, rna_vars: int = 12, adt_vars: int = 4):
    """Write a 2-modality (rna/adt) CITE-seq .scx via from_mudata, with a
    global `cell_type` obs column for filter_obs."""
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")
    import pyscx

    rng = np.random.default_rng(0)
    rna = sp.csr_matrix(rng.poisson(0.6, size=(n_obs, rna_vars)).astype(np.float32))
    adt = sp.csr_matrix(rng.poisson(0.6, size=(n_obs, adt_vars)).astype(np.float32))
    rna_ad = anndata.AnnData(X=rna)
    rna_ad.var_names = [f"g{i}" for i in range(rna_vars)]
    adt_ad = anndata.AnnData(X=adt)
    adt_ad.var_names = [f"a{i}" for i in range(adt_vars)]
    mu = mudata.MuData({"rna": rna_ad, "adt": adt_ad})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]
    mu.obs["cell_type"] = [["T_cell", "B_cell", "NK_cell"][i % 3] for i in range(n_obs)]
    pyscx.from_mudata(mu, path, codec="none")


class TestCloudModalityQuery:
    """Phase 6: open_cloud(...).query(modality=...) over an exploded .scxd/
    must match the local open(...).query(modality=...) pipeline."""

    def test_modality_query_matches_local(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "cite.scx")
            _create_multimodal_scx(scx_path, n_obs=45, rna_vars=12, adt_vars=4)
            exploded = os.path.join(tmpdir, "cite.scxd")
            pyscx.explode(scx_path, exploded)

            for modality, expect_vars in [("rna", 12), ("adt", 4)]:
                local = (
                    pyscx.open(scx_path)
                    .query(modality=modality)
                    .filter_obs("cell_type == 'T_cell'")
                    .collect()
                    .to_anndata()
                )
                cloud = (
                    pyscx.open_cloud(exploded)
                    .query(modality=modality)
                    .filter_obs("cell_type == 'T_cell'")
                    .collect()
                    .to_anndata()
                )
                assert cloud.n_vars == expect_vars
                assert cloud.n_obs == local.n_obs
                assert cloud.n_obs > 0
                np.testing.assert_array_equal(
                    cloud.X.toarray(), local.X.toarray()
                )
                assert list(cloud.var_names) == list(local.var_names)
                assert list(cloud.obs_names) == list(local.obs_names)

    def test_modality_select_genes_over_cloud(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "cite.scx")
            _create_multimodal_scx(scx_path, n_obs=30, rna_vars=12, adt_vars=4)
            exploded = os.path.join(tmpdir, "cite.scxd")
            pyscx.explode(scx_path, exploded)

            cloud = (
                pyscx.open_cloud(exploded)
                .query(modality="rna")
                .select_genes(["g0", "g5"])
                .collect()
                .to_anndata()
            )
            assert list(cloud.var_names) == ["g0", "g5"]

    def test_cloud_multimodal_without_modality_raises(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "cite.scx")
            _create_multimodal_scx(scx_path)
            exploded = os.path.join(tmpdir, "cite.scxd")
            pyscx.explode(scx_path, exploded)

            with pytest.raises(ValueError, match="multimodal"):
                pyscx.open_cloud(exploded).query()

    def test_cloud_unknown_modality_raises(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "cite.scx")
            _create_multimodal_scx(scx_path)
            exploded = os.path.join(tmpdir, "cite.scxd")
            pyscx.explode(scx_path, exploded)

            with pytest.raises(KeyError):
                pyscx.open_cloud(exploded).query(modality="atac")

    def test_read_cloud_modality(self):
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "cite.scx")
            _create_multimodal_scx(scx_path, n_obs=30, rna_vars=12, adt_vars=4)
            exploded = os.path.join(tmpdir, "cite.scxd")
            pyscx.explode(scx_path, exploded)

            adata = pyscx.read_cloud(
                exploded, obs_filter="cell_type == 'B_cell'", modality="rna"
            )
            assert adata.n_vars == 12
            assert adata.n_obs > 0

    def test_read_cloud_dtype_kwargs(self):
        """`read_cloud` is a one-call query, so it takes the decode kwargs.

        Both arms are exercised: a wide `data_dtype` (the typed decode) and an
        omitted one with `allow_lossy=True` (the f32 decode). The second is the
        one that regressed — the flag was accepted by the signature and dropped
        by the dispatcher, so the call was refused with a message denying the
        kwarg it had just been given.
        """
        import anndata
        import numpy as np
        import scipy.sparse as sp

        import pyscx

        big = 20_000_000
        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "big.scx")
            x = np.array([[big, 0], [1, 2], [3, 0], [4, 5]], dtype=np.int32)
            pyscx.from_anndata(
                anndata.AnnData(
                    X=sp.csr_matrix(x),
                    obs={"batch": ["a", "b", "a", "b"]},
                    var={"gene": ["g0", "g1"]},
                ),
                scx_path,
            )
            exploded = os.path.join(tmpdir, "big.scxd")
            pyscx.explode(scx_path, exploded)

            # Typed arm: exact at the requested width.
            adata = pyscx.read_cloud(exploded, data_dtype="float64")
            assert adata.X.data.dtype == np.float64
            assert int(adata.X.max()) == big

            # Default arm: refused without the opt-in, accepted with it.
            with pytest.raises(ValueError, match="allow_lossy"):
                pyscx.read_cloud(exploded)
            adata = pyscx.read_cloud(exploded, allow_lossy=True)
            assert adata.X.data.dtype == np.float32
            # Explicit float32 takes the same arm, and must honour it too.
            adata = pyscx.read_cloud(exploded, data_dtype="float32", allow_lossy=True)
            assert adata.X.data.dtype == np.float32

            # A dtype too narrow for the value is still refused.
            with pytest.raises(ValueError, match="allow_lossy"):
                pyscx.read_cloud(exploded, data_dtype="uint16")


class TestCloudReadVar:
    """`CloudExperiment.read_var` — parity with the local `Experiment`.

    `read_obs` existed on both handles; `read_var` was added locally first,
    which just moved the original dogfood F2 asymmetry one layer over: a
    cloud caller still had to `to_anndata()` the whole matrix over the network
    to get gene symbols. Exercised against a local exploded `.scxd`, which
    goes through the same `CloudReader` as a `gs://` URL.
    """

    def test_read_var_over_the_cloud_reader(self):
        import tempfile

        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "t.scx")
            _create_test_scx(scx_path, n_obs=30, n_vars=10)
            exploded = os.path.join(tmpdir, "t.scxd")
            pyscx.explode(scx_path, exploded)

            var = pyscx.open_cloud(exploded).read_var()
            assert len(var) == 10
            assert "gene_id" in var.columns
            assert list(var["gene_id"]) == [f"gene_{i}" for i in range(10)]

    def test_read_var_projection_keeps_the_index(self):
        import tempfile

        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "t.scx")
            _create_test_scx(scx_path, n_obs=30, n_vars=10)
            exploded = os.path.join(tmpdir, "t.scxd")
            pyscx.explode(scx_path, exploded)

            ce = pyscx.open_cloud(exploded)
            full = ce.read_var()
            sub = ce.read_var(columns=["gene_id"])
            assert list(sub.columns) == ["gene_id"]
            assert list(sub.index) == list(full.index)

    def test_read_var_matches_the_local_reader(self):
        """The cloud and local paths must agree on the same file."""
        import tempfile

        import pandas as pd
        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "t.scx")
            _create_test_scx(scx_path, n_obs=30, n_vars=10)
            exploded = os.path.join(tmpdir, "t.scxd")
            pyscx.explode(scx_path, exploded)

            local = pyscx.open(scx_path).read_var()
            cloud = pyscx.open_cloud(exploded).read_var()
            assert list(local.index) == list(cloud.index)
            assert set(local.columns) == set(cloud.columns)
            for col in local.columns:
                pd.testing.assert_series_equal(
                    local[col], cloud[col], check_dtype=False, check_categorical=False
                )

    def test_read_var_unknown_column_is_actionable(self):
        import tempfile

        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "t.scx")
            _create_test_scx(scx_path, n_obs=30, n_vars=10)
            exploded = os.path.join(tmpdir, "t.scxd")
            pyscx.explode(scx_path, exploded)

            with pytest.raises(KeyError) as excinfo:
                pyscx.open_cloud(exploded).read_var(columns=["nope"])
            msg = str(excinfo.value)
            assert "gene_id" in msg
            # The pandas index column is retained automatically and is often an
            # internal name — suggesting it would be noise.
            assert "__index_level_0__" not in msg

    def test_read_var_scopes_to_modality_over_the_cloud_reader(self):
        """Per-modality `var` routing is its own cloud code path.

        The cloud reader resolves `var/<modality>` sections over both packed
        and exploded layouts, so the modality kwarg is worth exercising here
        and not only against the local reader.
        """
        import tempfile

        import pyscx

        with tempfile.TemporaryDirectory() as tmpdir:
            scx_path = os.path.join(tmpdir, "mm.scx")
            _create_multimodal_scx(scx_path, n_obs=40, rna_vars=12, adt_vars=4)
            exploded = os.path.join(tmpdir, "mm.scxd")
            pyscx.explode(scx_path, exploded)

            # Both layouts, since the comment above claims both.
            for source in (exploded, scx_path):
                ce = pyscx.open_cloud(source)
                rna = ce.read_var(modality="rna")
                adt = ce.read_var(modality="adt")
                assert len(rna) == 12, source
                assert len(adt) == 4, source
                assert list(rna.index) == [f"g{i}" for i in range(12)], source

                # Agrees with the local reader on the same file.
                local_rna = pyscx.open(scx_path).read_var(modality="rna")
                assert list(local_rna.index) == list(rna.index), source

                with pytest.raises(KeyError, match="unknown modality"):
                    ce.read_var(modality="atac")
