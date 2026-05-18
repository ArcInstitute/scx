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
