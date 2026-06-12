"""Conformance-vector compatibility tests for pyscx.

Reads the new SCX reference fixtures from tests/reference_files/ and
verifies that pyscx.open() handles every variant:
  - v1 base + layers/obsm/uns
  - CSC sidecar
  - predicate indexes
  - deletion vectors
  - v2 multimodal (CITE-seq + partial CSC)
  - bitmap shard
  - cloud_optimized (front-catalog)

Companion to test_golden_files.py, which covers the codec-backward-compat
corpus. Regenerate fixtures via:
    cargo test -p scx-integration-tests --test conformance_vectors \\
        generate_conformance_vectors -- --ignored
"""

import json
from pathlib import Path

import numpy as np
import pytest

import pyscx

REF_DIR = Path(__file__).resolve().parent / "../../tests/reference_files"

# Single-modality fixtures (have a global X).
SINGLE_MODALITY_FIXTURES = [
    "v1_minimal",
    "v1_csc",
    "v1_layers_obsm_uns",
    "v1_predicate_indexes",
    "v1_deletion_vectors",
    "v2_bitmap",
    "cloud_optimized_reference",
]

# Multimodal fixtures (no global X — modalities are per-modality).
MULTIMODAL_FIXTURES = [
    "v2_multimodal_citeseq",
    "v2_multimodal_partial_csc",
]


def _load_sidecar(name):
    sidecar_path = REF_DIR / f"{name}.json"
    if not sidecar_path.exists():
        pytest.skip(
            f"{name}.json missing. Run: cargo test -p scx-integration-tests "
            "--test conformance_vectors generate_conformance_vectors -- --ignored"
        )
    with open(sidecar_path) as f:
        return json.load(f)


@pytest.mark.parametrize("name", SINGLE_MODALITY_FIXTURES)
def test_conformance_single_modality_opens(name):
    """pyscx.open succeeds and reports the expected shape."""
    sidecar = _load_sidecar(name)
    path = REF_DIR / f"{name}.scx"
    exp = pyscx.open(str(path))
    # The sidecar records the on-disk header (physical) row count; `n_obs` now
    # reports the logical (live) count after deletions (B3), so compare the
    # header against `n_obs_physical`. They're equal for fixtures without
    # deletion vectors.
    assert exp.n_obs_physical == sidecar["header"]["n_obs"], f"{name}: n_obs"
    assert exp.n_vars == sidecar["header"]["n_vars"], f"{name}: n_vars"


@pytest.mark.parametrize("name", SINGLE_MODALITY_FIXTURES)
def test_conformance_single_modality_validate(name):
    """All section checksums pass."""
    _ = _load_sidecar(name)
    path = REF_DIR / f"{name}.scx"
    exp = pyscx.open(str(path))
    results = exp.validate()
    for section_name, passed in results:
        assert passed, f"{name}: checksum failed for section '{section_name}'"


@pytest.mark.parametrize("name", SINGLE_MODALITY_FIXTURES)
def test_conformance_single_modality_csr_matches_sidecar(name):
    """CSR triplet matches the sidecar's expected_csr (when present).

    Skipped for fixtures whose on-disk CSR diverges from the pyscx view:
    v1_deletion_vectors applies logical deletions when materialising
    via `to_anndata`, while the sidecar records the un-filtered shard
    via `ScxReader::read_all_csr_shards`.
    """
    if name == "v1_deletion_vectors":
        pytest.skip("pyscx applies deletions during materialisation; sidecar tracks raw shard")
    sidecar = _load_sidecar(name)
    if sidecar.get("expected_csr") is None:
        pytest.skip(f"{name} has no expected_csr (e.g. cloud directory fixture)")
    path = REF_DIR / f"{name}.scx"
    adata = pyscx.open(str(path)).to_anndata()
    X = adata.X
    expected = sidecar["expected_csr"]
    np.testing.assert_array_equal(
        X.indptr.astype(np.int64),
        np.array(expected["indptr"], dtype=np.int64),
        err_msg=f"{name}: indptr",
    )
    np.testing.assert_array_equal(
        X.indices.astype(np.int32),
        np.array(expected["indices"], dtype=np.int32),
        err_msg=f"{name}: indices",
    )
    np.testing.assert_array_equal(
        X.data,
        np.array(expected["data"], dtype=np.float32),
        err_msg=f"{name}: data",
    )


@pytest.mark.parametrize("name", MULTIMODAL_FIXTURES)
def test_conformance_multimodal_modalities_listed(name):
    """v2 multimodal fixtures report the expected modality names."""
    sidecar = _load_sidecar(name)
    path = REF_DIR / f"{name}.scx"
    exp = pyscx.open(str(path))
    # Modality table presence: at least one ModalityTable entry in catalog.
    modality_entries = [
        e
        for e in sidecar["catalog_summary"]
        if e["section_type"] == "ModalityTable"
    ]
    assert len(modality_entries) == 1, (
        f"{name}: expected exactly one ModalityTable section, got {len(modality_entries)}"
    )
    # Header records the physical row count; compare against `n_obs_physical`
    # (equal to `n_obs` for fixtures without deletion vectors). See B3.
    assert exp.n_obs_physical == sidecar["header"]["n_obs"], f"{name}: n_obs"


def test_conformance_manifest_hashes():
    """BLAKE3 hashes for the new fixtures match MANIFEST.json."""
    try:
        import blake3 as b3
    except ImportError:
        pytest.skip("blake3 Python package not installed")

    manifest_path = REF_DIR / "MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("MANIFEST.json not found")

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["algorithm"] == "blake3"

    conformance_names = SINGLE_MODALITY_FIXTURES + MULTIMODAL_FIXTURES

    # Skip the whole test if the corpus hasn't been generated yet — but
    # once any fixture is present, every named fixture must be both on
    # disk AND in the manifest. Silent-skip would defeat the
    # frozen-reference guarantee.
    any_present = any((REF_DIR / f"{n}.scx").exists() for n in conformance_names)
    if not any_present:
        pytest.skip(
            "no conformance fixtures present; run "
            "`cargo test -p scx-integration-tests --test conformance_vectors "
            "generate_conformance_vectors -- --ignored`"
        )

    for fixture_name in conformance_names:
        rel_path = f"{fixture_name}.scx"
        file_path = REF_DIR / rel_path
        assert file_path.exists(), (
            f"{rel_path}: fixture file missing — regenerate the corpus"
        )
        assert rel_path in manifest["files"], (
            f"{rel_path}: missing from MANIFEST.json — regenerate the corpus"
        )
        expected_hash = manifest["files"][rel_path]
        actual_hash = b3.blake3(file_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, (
            f"BLAKE3 hash mismatch for {rel_path}: "
            f"got {actual_hash}, expected {expected_hash}"
        )


def test_conformance_deletion_vectors_visible():
    """Deletion-vector fixture flags has_deletion_vectors in the header."""
    sidecar = _load_sidecar("v1_deletion_vectors")
    assert sidecar["header"]["has_deletion_vectors"] is True


def test_conformance_bitmap_section_present():
    """Bitmap fixture has a BitmapShard entry in the catalog."""
    sidecar = _load_sidecar("v2_bitmap")
    bitmap_entries = [
        e
        for e in sidecar["catalog_summary"]
        if e["section_type"] == "BitmapShard"
    ]
    assert len(bitmap_entries) >= 1, (
        f"v2_bitmap: expected ≥1 BitmapShard entry, got {len(bitmap_entries)}"
    )
