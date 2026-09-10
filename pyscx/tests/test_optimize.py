import os
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

import pyscx


def _create_v2_scx(path):
    """Create a minimal .scx that would benefit from optimize."""
    adata = pytest.importorskip("anndata").AnnData(
        X=sp.random(50, 100, density=0.3, format="csr", dtype=np.float32),
        obs=pd.DataFrame({"ct": ["A"] * 50}, index=[f"c{i}" for i in range(50)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(100)]),
    )
    pyscx.from_anndata(adata, path)


class TestOptimize:
    def test_optimize_to_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.scx")
            dst = os.path.join(tmp, "out.scx")
            _create_v2_scx(src)
            pyscx.optimize(src, dst)
            exp = pyscx.open(dst)
            assert exp.n_obs == 50
            assert exp.n_vars == 100

    def test_optimize_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.scx")
            _create_v2_scx(path)
            pyscx.optimize(path, path)  # in-place
            exp = pyscx.open(path)
            assert exp.n_obs == 50

    def test_optimize_codec_scx1(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.scx")
            dst = os.path.join(tmp, "out.scx")
            _create_v2_scx(src)
            pyscx.optimize(src, dst, codec="scx1")
            exp = pyscx.open(dst)
            assert exp.n_obs == 50

    def test_optimize_existing_output_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.scx")
            dst = os.path.join(tmp, "out.scx")
            _create_v2_scx(src)
            _create_v2_scx(dst)  # pre-existing, different file
            with pytest.raises(RuntimeError):
                pyscx.optimize(src, dst)

    def test_optimize_bad_codec_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.scx")
            _create_v2_scx(src)
            with pytest.raises((ValueError, RuntimeError)):
                pyscx.optimize(src, os.path.join(tmp, "out.scx"), codec="zstd")

    def test_optimize_default_codec_is_auto(self):
        """Ensure the default codec kwarg is 'auto' (no sidecar guarantee)."""
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.scx")
            dst = os.path.join(tmp, "out.scx")
            _create_v2_scx(src)
            # Should not raise — 'auto' is the default
            pyscx.optimize(src, dst)

    def test_optimize_memory_budget_accepts_str_int_and_none(self):
        """The budget caps the parallel re-encode's in-flight bytes.

        Three accepted spellings, matching every other pyscx ``memory_budget``:
        ``None`` (the 1 GiB default, which is *not* unbounded), an int of bytes,
        and a binary-prefixed string. A tiny budget is the documented way to pin
        the one-shard-at-a-time behaviour this call had before the re-encode
        became parallel, so it must succeed rather than refuse — a single
        shard's encode is irreducible.
        """
        for budget in (None, 1, 1024, "512M", "2GiB"):
            with tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "in.scx")
                dst = os.path.join(tmp, "out.scx")
                _create_v2_scx(src)
                pyscx.optimize(src, dst, memory_budget=budget)
                assert os.path.exists(dst), f"budget {budget!r} produced no output"

    # Byte identity across in-flight budgets is deliberately *not* asserted
    # here: it is a property of the Rust re-encode, pinned by
    # `optimize_honours_an_explicit_codec_at_every_concurrency` (which sweeps
    # the budget from one shard per chunk to all of them and compares the
    # catalog's entry order as well as the contents) and by cross-arm
    # per-section digests at census scale. What Python owns is the kwarg's
    # contract, which is what these tests cover.
    def test_optimize_rejects_a_decimal_memory_budget(self):
        """Decimal suffixes are ambiguous and rejected everywhere in scx."""
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.scx")
            _create_v2_scx(src)
            with pytest.raises(ValueError):
                pyscx.optimize(
                    src, os.path.join(tmp, "out.scx"), memory_budget="10GB"
                )
