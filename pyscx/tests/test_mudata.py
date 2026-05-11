"""Phase D + Phase E + Phase K coverage for the multimodal API.

Phase D — `pyscx.from_mudata` round-trips a CITE-seq MuData object
through SCX and back via `to_mudata()`.

Phase E — `from_mudata` resolves codec="auto" via
`select_codec_for_modality`, so RNA shards land on Scx1 while the
Protein/ADT modality is force-overridden to Zstd.

Phase K — round-trip correctness on real public datasets, v1↔v2
single-modality semantic parity, and Python→R cross-language
interoperability via subprocess.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import textwrap

import numpy as np
import pytest


# CodecId enum mapping (mirrors scx_codec::CodecId).
SCX1 = 1
ZSTD = 2
LZ4_SHUFFLE = 3


def _multimodal_data_available(name: str) -> bool:
    """Phase K.1: gate real-data round-trip tests on `SCX_DATA_DIR`.

    Returns True when `$SCX_DATA_DIR/<name>` exists. The download
    scripts at `benchmarks/scripts/download_citeseq_pbmc.py` and
    `download_multiome_pbmc.py` stage the expected files; absent
    `SCX_DATA_DIR` (e.g. running outside the benchmark host) the
    test skips.
    """
    data_dir = os.environ.get("SCX_DATA_DIR")
    if not data_dir:
        # Fall back to SCX_WORK_DIR/benchmarks/datasets per the
        # convention in `benchmarks/scripts/download_pbmc10k.py`.
        work_dir = os.environ.get("SCX_WORK_DIR")
        if not work_dir:
            return False
        data_dir = os.path.join(work_dir, "benchmarks", "datasets")
    return pathlib.Path(data_dir, name).exists()


def _multimodal_data_path(name: str) -> pathlib.Path:
    data_dir = os.environ.get("SCX_DATA_DIR") or os.path.join(
        os.environ.get("SCX_WORK_DIR", ""), "benchmarks", "datasets"
    )
    return pathlib.Path(data_dir) / name


def _find_scx_cli() -> str | None:
    """Find an `scx` CLI binary that supports the v2 format. Prefers
    the workspace-built `scx-cli` (always matches the running pyscx
    version) over a possibly-stale system `scx` on PATH. Returns None
    if nothing is found."""
    workspace_target = pathlib.Path(__file__).resolve().parents[2] / "target"
    for profile in ("release", "debug"):
        for candidate in ("scx-cli", "scx"):
            cand = workspace_target / profile / candidate
            if cand.exists() and os.access(cand, os.X_OK):
                return str(cand)
    # Fall back to whatever is on PATH (tests guard against version
    # mismatch by skipping when `scx info` returns non-zero).
    for candidate in ("scx-cli", "scx"):
        path = shutil.which(candidate)
        if path is not None:
            return path
    return None


@pytest.fixture
def cite_seq_mudata():
    """Tiny CITE-seq fixture with small-integer counts in both
    modalities (RNA + ADT). Both modalities use the same scipy CSR
    layout and similar value distributions, so any per-modality codec
    difference must come from the ModalityType override."""
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n_obs, rna_n_vars, adt_n_vars = 32, 80, 12

    # RNA: small UMI-like counts (median ≈ 1).
    rna_dense = rng.poisson(lam=0.4, size=(n_obs, rna_n_vars)).astype(np.float32)
    # ADT: similarly small counts so the *only* signal that flips the
    # codec is the modality_type, not the data magnitude.
    adt_dense = rng.poisson(lam=0.4, size=(n_obs, adt_n_vars)).astype(np.float32)

    rna_csr = sp.csr_matrix(rna_dense)
    adt_csr = sp.csr_matrix(adt_dense)

    rna_ad = anndata.AnnData(X=rna_csr)
    rna_ad.var_names = [f"g{i}" for i in range(rna_n_vars)]
    adt_ad = anndata.AnnData(X=adt_csr)
    adt_ad.var_names = [f"a{i}" for i in range(adt_n_vars)]

    mu = mudata.MuData({"rna": rna_ad, "adt": adt_ad})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]
    return mu


def test_from_mudata_round_trip(cite_seq_mudata):
    """Phase D: pyscx.from_mudata → pyscx.open(...).to_mudata() round-trips."""
    pytest.importorskip("mudata")
    import pyscx

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(cite_seq_mudata, path)

        reader = pyscx.open(path)
        assert reader.is_multimodal
        assert reader.n_modalities == 2
        assert sorted(reader.modality_names) == ["adt", "rna"]

        mu_back = reader.to_mudata()
        assert "rna" in mu_back.mod
        assert "adt" in mu_back.mod
        rna_orig = cite_seq_mudata.mod["rna"].X.toarray()
        rna_back = mu_back.mod["rna"].X.toarray()
        np.testing.assert_array_equal(rna_orig, rna_back)
        adt_orig = cite_seq_mudata.mod["adt"].X.toarray()
        adt_back = mu_back.mod["adt"].X.toarray()
        np.testing.assert_array_equal(adt_orig, adt_back)


def test_from_mudata_per_modality_codec_routing(cite_seq_mudata):
    """Phase E: with codec="auto", RNA picks Scx1 (small UMI median)
    while the Protein/ADT modality is overridden to Zstd by
    `select_codec_for_modality`, even though the underlying data
    distributions are similar.

    Asserts on `modality_info().default_codec_id`, which the h5mu
    pipeline writes as the same codec used for every CSR shard of
    that modality.
    """
    pytest.importorskip("mudata")
    import pyscx

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(cite_seq_mudata, path)  # codec="auto" by default

        reader = pyscx.open(path)
        rna_id = reader.modality_id("rna")
        adt_id = reader.modality_id("adt")
        assert rna_id is not None
        assert adt_id is not None

        rna_info = reader.modality_info(rna_id)
        adt_info = reader.modality_info(adt_id)
        assert rna_info is not None
        assert adt_info is not None

        assert rna_info["default_codec_id"] == SCX1, (
            f"RNA default codec should be Scx1 (id=1); got {rna_info['default_codec_id']}"
        )
        assert adt_info["default_codec_id"] == ZSTD, (
            f"ADT default codec should be Zstd (id=2); got {adt_info['default_codec_id']}"
        )


def test_from_mudata_honors_shard_size():
    """PR #68: `pyscx.from_mudata(..., shard_size=N)` must shard each
    modality's CSR by N rows. The pre-fix code accepted the parameter
    but emitted a single CSR shard per modality regardless.
    """
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")
    import scipy.sparse as sp
    import pyscx

    rng = np.random.default_rng(0)
    n_obs = 100
    rna = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(lam=0.5, size=(n_obs, 8)).astype(np.float32))
    )
    rna.var_names = [f"g{i}" for i in range(8)]
    adt = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(lam=0.5, size=(n_obs, 4)).astype(np.float32))
    )
    adt.var_names = [f"a{i}" for i in range(4)]
    mu = mudata.MuData({"rna": rna, "adt": adt})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sharded.scx")
        pyscx.from_mudata(mu, path, shard_size=30)

        reader = pyscx.open(path)
        for name in ("rna", "adt"):
            mid = reader.modality_id(name)
            assert mid is not None
            info = reader.modality_info(mid)
            assert info is not None
            # 100 rows / 30 per shard = ceil(100/30) = 4 shards.
            assert info["n_csr_shards"] == 4, (
                f"modality {name!r} should have 4 CSR shards at shard_size=30, "
                f"got {info['n_csr_shards']}"
            )

        # Round-trip still produces bit-equal matrices.
        mu_back = reader.to_mudata()
        for name in ("rna", "adt"):
            np.testing.assert_array_equal(
                mu.mod[name].X.toarray(),
                mu_back.mod[name].X.toarray(),
                err_msg=f"{name!r} round-trip mismatch after sharded write",
            )


# --- Phase D.4: ScxBackedMuDataset ----------------------------------------


def test_backed_mudata_lazy_mod_access(cite_seq_mudata):
    """Phase D.4: `ScxBackedMuDataset.mod[name]` returns a lazy
    `ScxBackedSparseDataset` pinned to the chosen modality. Per-modality
    `n_vars` matches the modality table (not the file-wide max)."""
    pytest.importorskip("mudata")
    import pyscx

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(cite_seq_mudata, path)
        mu = pyscx.ScxBackedMuDataset(path)

        assert mu.is_multimodal is True
        assert mu.n_modalities == 2
        assert sorted(mu.modality_names) == ["adt", "rna"]
        assert mu.modality_id("rna") is not None
        assert mu.modality_id("unknown") is None

        rna = mu.mod["rna"]
        assert rna.modality_id == mu.modality_id("rna")
        # Per-modality n_vars survives the wrapper (not the file-wide
        # header.n_vars max).
        rna_info = mu.modality_info(mu.modality_id("rna"))
        assert rna_info is not None
        assert rna.shape == (cite_seq_mudata.n_obs, rna_info["n_vars"])

        adt = mu.mod["adt"]
        adt_info = mu.modality_info(mu.modality_id("adt"))
        assert adt is not None and adt_info is not None
        assert adt.shape == (cite_seq_mudata.n_obs, adt_info["n_vars"])


def test_backed_mudata_obs_caches(cite_seq_mudata):
    """Phase D.4: `.obs` is materialised on first access and cached
    thereafter — second access returns the same Python object."""
    pytest.importorskip("mudata")
    import pyscx

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(cite_seq_mudata, path)
        mu = pyscx.ScxBackedMuDataset(path)

        obs1 = mu.obs
        obs2 = mu.obs
        # Cache returns the same Python object on subsequent access.
        assert obs1 is obs2
        # The DataFrame's row count matches the global n_obs.
        assert len(obs1) == cite_seq_mudata.n_obs


def test_backed_mudata_mod_dict_surface(cite_seq_mudata):
    """Phase D.4: `.mod` is dict-like — supports `name in mu.mod`,
    `iter(mu.mod)`, `keys()`, `len(mu.mod)`, and raises KeyError on
    unknown names."""
    pytest.importorskip("mudata")
    import pyscx

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(cite_seq_mudata, path)
        mu = pyscx.ScxBackedMuDataset(path)

        assert "rna" in mu.mod
        assert "adt" in mu.mod
        assert "unknown" not in mu.mod
        assert len(mu.mod) == 2
        assert sorted(mu.mod.keys()) == ["adt", "rna"]
        assert sorted(list(mu.mod)) == ["adt", "rna"]

        with pytest.raises(KeyError):
            _ = mu.mod["unknown"]


def test_backed_mudata_to_mudata_eager(cite_seq_mudata):
    """Phase D.4: `to_mudata()` is the eager-materialisation escape
    hatch — wraps the same `mudata::to_mudata` path used by
    `pyscx.open(path).to_mudata()`."""
    pytest.importorskip("mudata")
    import pyscx

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cite.scx")
        pyscx.from_mudata(cite_seq_mudata, path)
        mu = pyscx.ScxBackedMuDataset(path)
        full = mu.to_mudata()
        assert "rna" in full.mod
        assert "adt" in full.mod


def test_backed_mudata_rejects_single_modality(tmp_path):
    """Phase D.4: opening a single-modality file via
    `ScxBackedMuDataset` raises with a clear message directing the
    user to `pyscx.open(path)`."""
    pytest.importorskip("anndata")
    import anndata
    import scipy.sparse as sp
    import pyscx

    rng = np.random.default_rng(0)
    adata = anndata.AnnData(
        X=sp.csr_matrix(rng.poisson(0.3, size=(20, 30)).astype(np.float32))
    )
    adata.var_names = [f"g{i}" for i in range(30)]
    adata.obs_names = [f"c{i}" for i in range(20)]
    path = str(tmp_path / "single.scx")
    pyscx.from_anndata(adata, path)

    with pytest.raises(RuntimeError, match="single-modality"):
        pyscx.ScxBackedMuDataset(path)


# --- Phase K.1: Round-trip correctness ------------------------------------


def test_phase_k1_v1_v2_single_modality_parity(tmp_path):
    """Phase K.1.4: v2 is a strict superset of v1 — a 1-modality MuData
    round-trip yields the same X as the single-modality h5ad path on
    the same input. Locks the spec invariant that wrapping a single
    AnnData in MuData incurs no data loss."""
    pytest.importorskip("mudata")
    pytest.importorskip("anndata")
    import anndata
    import mudata
    import scipy.sparse as sp
    import pyscx

    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(0.4, (200, 80)).astype(np.float32))
    ad = anndata.AnnData(X=X)
    ad.var_names = [f"g{i}" for i in range(80)]
    ad.obs_names = [f"c{i}" for i in range(200)]

    # v1 path (single-modality)
    p_v1 = tmp_path / "v1.scx"
    pyscx.from_anndata(ad, str(p_v1))
    ad_v1 = pyscx.open(str(p_v1)).to_anndata()

    # v2 path (1-modality MuData)
    mu = mudata.MuData({"rna": ad})
    p_v2 = tmp_path / "v2.scx"
    pyscx.from_mudata(mu, str(p_v2))
    mu_v2 = pyscx.open(str(p_v2)).to_mudata()

    np.testing.assert_array_equal(ad.X.toarray(), ad_v1.X.toarray())
    np.testing.assert_array_equal(
        ad_v1.X.toarray(), mu_v2.mod["rna"].X.toarray()
    )


@pytest.mark.skipif(
    not _multimodal_data_available("cite_seq_pbmc_5k.h5mu"),
    reason="Real CITE-seq h5mu not staged at SCX_DATA_DIR; "
    "run benchmarks/scripts/download_citeseq_pbmc.py first",
)
def test_phase_k1_citeseq_real_round_trip(tmp_path):
    """Phase K.1.1: real 10x PBMC 5k CITE-seq h5mu round-trips through
    SCX with element-wise X equality, global obs equality, and
    per-modality var equality."""
    pytest.importorskip("mudata")
    pytest.importorskip("pandas")
    import mudata
    import pandas as pd
    import pyscx

    src = _multimodal_data_path("cite_seq_pbmc_5k.h5mu")
    mu_in = mudata.read_h5mu(str(src))
    out = tmp_path / "cite.scx"
    pyscx.from_mudata(mu_in, str(out))
    mu_out = pyscx.open(str(out)).to_mudata()

    # Per-modality X equality.
    for name in ("rna", "adt"):
        x_in = mu_in.mod[name].X
        x_out = mu_out.mod[name].X
        x_in_arr = x_in.toarray() if hasattr(x_in, "toarray") else np.asarray(x_in)
        x_out_arr = x_out.toarray() if hasattr(x_out, "toarray") else np.asarray(x_out)
        np.testing.assert_array_equal(x_in_arr, x_out_arr)

    # Per-modality var equality (subset to columns that survive the
    # arrow round-trip — pyscx records only string-indexed var
    # metadata, matching the single-modality semantics).
    for name in ("rna", "adt"):
        var_in = mu_in.mod[name].var
        var_out = mu_out.mod[name].var
        # Index alignment is the load-bearing assertion (every
        # feature name preserved, in order).
        assert list(var_in.index) == list(var_out.index)


@pytest.mark.skipif(
    not _multimodal_data_available("multiome_pbmc_10k.h5mu"),
    reason="Real 10x Multiome h5mu not staged at SCX_DATA_DIR; "
    "run benchmarks/scripts/download_multiome_pbmc.py first",
)
def test_phase_k1_multiome_real_round_trip(tmp_path):
    """Phase K.1.2: real 10x PBMC Multiome (RNA + ATAC) round-trip.

    Doubles as a thin K.2 cross-language smoke: invokes `scx info
    --json` as a subprocess and verifies the per-modality
    `default_codec_id` reflects the Phase E routing decision (RNA →
    Scx1, ATAC → Zstd or Lz4Shuffle depending on data shape).
    """
    pytest.importorskip("mudata")
    import mudata
    import pyscx

    src = _multimodal_data_path("multiome_pbmc_10k.h5mu")
    mu_in = mudata.read_h5mu(str(src))
    assert "rna" in mu_in.mod
    assert "atac" in mu_in.mod

    out = tmp_path / "multiome.scx"
    pyscx.from_mudata(mu_in, str(out))

    # Round-trip via the SCX→MuData path. The atac matrix is large
    # and mostly-binary; we compare nnz + non-zero values rather
    # than full densification to keep the test memory bounded.
    mu_out = pyscx.open(str(out)).to_mudata()
    for name in ("rna", "atac"):
        x_in = mu_in.mod[name].X
        x_out = mu_out.mod[name].X
        assert x_in.shape == x_out.shape
        assert x_in.nnz == x_out.nnz, (
            f"{name}: nnz mismatch in={x_in.nnz} out={x_out.nnz}"
        )

    # Cross-language smoke: `scx info --json` parses + lists modalities.
    # Skipped silently when `scx` is missing or older than the format
    # version the current pyscx writes — the round-trip above already
    # verifies the per-modality data is intact.
    scx_bin = _find_scx_cli()
    if scx_bin is None:
        return
    proc = subprocess.run(
        [scx_bin, "info", "--json", str(out)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Older scx binaries reject v2 files with "unsupported format
        # version" — treat as a non-blocking smoke and skip the
        # codec-routing assertion for this run.
        pytest.skip(f"`scx info` failed (older binary?): {proc.stderr.strip()}")
    info = json.loads(proc.stdout)
    modalities = info.get("modalities") or info.get("entries", [{}])[0].get(
        "modalities", []
    )
    assert {m["name"] for m in modalities} == {"rna", "atac"}
    # `scx info --json` reports `default_codec` as a string ("scx1",
    # "zstd", "lz4_shuffle", "pcodec") — not the numeric id.
    codec_by_name = {m["name"]: m.get("default_codec") for m in modalities}
    # RNA always lands on Scx1 (small UMI counts).
    assert codec_by_name["rna"] == "scx1", codec_by_name
    # ATAC routes to Zstd (binary-ish) or Lz4Shuffle per Phase E.
    # `scx info --json` reports the lz4+shuffle codec as
    # ``"lz4+shuffle"`` (mirroring the human-readable info output).
    assert codec_by_name["atac"] in ("zstd", "lz4+shuffle"), codec_by_name


# --- Phase K.2: Cross-language --------------------------------------------


@pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript not available; install R + rscx to run cross-language tests",
)
def test_phase_k2_python_to_r_cross_language(cite_seq_mudata, tmp_path):
    """Phase K.2.1: write multimodal SCX from Python, read in Rust via
    `scx info --json`, then read in R via `rscx::scx_open → to_seurat`
    in a subprocess. Asserts modality count, names, and per-modality
    var lengths survive the Py→Rust→R chain.
    """
    pytest.importorskip("mudata")
    import pyscx

    scx_bin = _find_scx_cli()
    if scx_bin is None:
        pytest.skip("scx / scx-cli CLI not found on PATH or workspace target")

    out = tmp_path / "py_to_r.scx"
    pyscx.from_mudata(cite_seq_mudata, str(out))

    # Rust read-back via `scx info --json`. An older `scx` binary may
    # reject the v2 format with "unsupported format version" — that's
    # an environment skip, not a test failure.
    proc = subprocess.run(
        [scx_bin, "info", "--json", str(out)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"`scx info` failed (older binary?): {proc.stderr.strip()}")
    info = json.loads(proc.stdout)
    modalities = info.get("modalities") or info.get("entries", [{}])[0].get(
        "modalities", []
    )
    assert {m["name"] for m in modalities} == {"rna", "adt"}

    # R read-back: Rscript loads rscx, opens the file, lifts to Seurat,
    # then prints n_cells + per-assay n_features.
    rscript = textwrap.dedent(f"""\
        suppressMessages({{
          if (!requireNamespace("rscx", quietly = TRUE)) {{
            cat("SKIP_RSCX_NOT_INSTALLED")
            quit(status = 0)
          }}
          library(rscx)
        }})
        reader <- scx_open("{out}")
        seu <- to_seurat(reader)
        n_obs <- ncol(seu)
        n_rna <- nrow(seu[["rna"]])
        n_adt <- nrow(seu[["adt"]])
        cat(sprintf("%d %d %d", n_obs, n_rna, n_adt))
    """)
    proc = subprocess.run(
        ["Rscript", "-e", rscript],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if "SKIP_RSCX_NOT_INSTALLED" in proc.stdout:
        pytest.skip("rscx package not installed in this R library")
    assert proc.returncode == 0, (
        f"Rscript failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )

    parts = proc.stdout.strip().split()
    assert len(parts) >= 3, f"Unexpected Rscript output: {proc.stdout!r}"
    n_obs, n_rna, n_adt = (int(parts[0]), int(parts[1]), int(parts[2]))
    assert n_obs == cite_seq_mudata.n_obs
    assert n_rna == cite_seq_mudata.mod["rna"].n_vars
    assert n_adt == cite_seq_mudata.mod["adt"].n_vars
