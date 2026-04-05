"""Golden file regression tests for SCX format.

Phase 0 of the Sprint Regression Baseline (phase3_report.md §0.4–§0.5).

Reads golden SCX files from tests/reference_files/ and verifies:
  - Checksum validation passes
  - CSR arrays (indptr, indices, data) match JSON sidecars exactly
  - obs/var metadata matches
  - scipy CSR dtype, shape, and nnz match
  - BLAKE3 manifest hashes match
"""

import json
from pathlib import Path

import numpy as np
import pytest

import pyscx

GOLDEN_DIR = Path(__file__).resolve().parent / "../../tests/reference_files"


def _load_sidecars():
    """Yield (basename, scx_path, sidecar_dict) for each golden file."""
    json_files = sorted(GOLDEN_DIR.glob("golden_*.json"))
    if not json_files:
        pytest.skip(
            "No golden files found. Run: cargo test -p scx-integration-tests "
            "--test golden_files generate_golden_files -- --ignored"
        )
    for json_path in json_files:
        with open(json_path) as f:
            sidecar = json.load(f)
        basename = json_path.stem
        scx_path = str(json_path.with_suffix(".scx"))
        yield basename, scx_path, sidecar


# Collect parametrize IDs and args once at module level
_GOLDEN_PARAMS = list(_load_sidecars())
_GOLDEN_IDS = [p[0] for p in _GOLDEN_PARAMS]


@pytest.mark.parametrize("basename,scx_path,sidecar", _GOLDEN_PARAMS, ids=_GOLDEN_IDS)
class TestGoldenFiles:
    """Validate each golden SCX file against its JSON sidecar."""

    def test_validate(self, basename, scx_path, sidecar):
        """§0.4: All section checksums pass."""
        exp = pyscx.open(scx_path)
        results = exp.validate()
        for section_name, passed in results:
            assert passed, f"{basename}: checksum failed for section '{section_name}'"

    def test_shape_and_nnz(self, basename, scx_path, sidecar):
        """§0.4: n_obs, n_vars, nnz match expected values."""
        exp = pyscx.open(scx_path)
        assert exp.n_obs == sidecar["n_obs"], f"{basename}: n_obs mismatch"
        assert exp.n_vars == sidecar["n_vars"], f"{basename}: n_vars mismatch"
        assert exp.nnz == sidecar["nnz"], f"{basename}: nnz mismatch"

    def test_scipy_csr_properties(self, basename, scx_path, sidecar):
        """§0.4: scipy CSR dtype, shape, and nnz match."""
        adata = pyscx.open(scx_path).to_anndata()
        X = adata.X

        assert X.shape == (sidecar["n_obs"], sidecar["n_vars"]), f"{basename}: X.shape"
        assert X.nnz == sidecar["nnz"], f"{basename}: X.nnz"
        assert X.dtype == np.float32, f"{basename}: X.dtype is {X.dtype}, expected float32"

    def test_csr_arrays(self, basename, scx_path, sidecar):
        """§0.4: indptr, indices, data arrays match sidecar exactly."""
        adata = pyscx.open(scx_path).to_anndata()
        X = adata.X

        expected_indptr = np.array(sidecar["indptr"], dtype=np.int64)
        expected_indices = np.array(sidecar["indices"], dtype=np.int32)
        expected_data = np.array(sidecar["data"], dtype=np.float32)

        # indptr may be int32 or int64 depending on scipy; compare values
        np.testing.assert_array_equal(
            X.indptr.astype(np.int64), expected_indptr, err_msg=f"{basename}: indptr"
        )
        np.testing.assert_array_equal(
            X.indices.astype(np.int32), expected_indices, err_msg=f"{basename}: indices"
        )
        # Bit-exact f32 comparison
        np.testing.assert_array_equal(
            X.data, expected_data, err_msg=f"{basename}: data"
        )

    def test_metadata(self, basename, scx_path, sidecar):
        """§0.4: obs and var metadata match sidecar."""
        adata = pyscx.open(scx_path).to_anndata()

        assert list(adata.obs["cell_id"]) == sidecar["obs_cell_ids"], (
            f"{basename}: obs cell_id mismatch"
        )
        assert list(adata.obs["cell_type"]) == sidecar["obs_cell_types"], (
            f"{basename}: obs cell_type mismatch"
        )
        assert list(adata.var["gene_id"]) == sidecar["var_gene_ids"], (
            f"{basename}: var gene_id mismatch"
        )


def test_manifest_hashes():
    """§0.5: BLAKE3 hashes in MANIFEST.json match actual file hashes."""
    try:
        import blake3 as b3
    except ImportError:
        pytest.skip("blake3 Python package not installed")

    manifest_path = GOLDEN_DIR / "MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("MANIFEST.json not found")

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["algorithm"] == "blake3"

    for filename, expected_hash in manifest["files"].items():
        file_path = GOLDEN_DIR / filename
        assert file_path.exists(), f"Manifest references missing file: {filename}"
        actual_hash = b3.blake3(file_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, (
            f"BLAKE3 hash mismatch for {filename}: "
            f"got {actual_hash}, expected {expected_hash}"
        )
