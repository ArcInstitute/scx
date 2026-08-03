"""Hermetic tests for the doublet-interop benchmark (`doublet_interop`).

Fast and self-contained, mirroring `test_dataload_phase1d.py`: a small h5ad +
SCX fixture under an isolated ``SCX_WORK_DIR``, a **stub tool** standing in for
scDblFinder / Scrublet, and hand-computed metric cases.

A stub rather than the real callers because the alternative is a suite that
only runs on a machine with the `rscx` conda env, R 4.5, scDblFinder 1.24 and
scikit-image — which in practice means a suite nobody runs. The stub exercises
every line of the orchestration; what it cannot cover (does scDblFinder
actually emit `scDblFinder.class`?) is what the one real capture is for.

The failure this suite is most concerned with is silence. A benchmark can
degrade into reporting nothing far more easily than into reporting something
wrong: a tool whose env is missing is a legitimate skip, an empty pairwise dict
is legitimate when only one tool ran, and a batch cap is legitimate — each of
those, unwatched, turns a broken harness into an `available: False` nobody
reads. So the skips are asserted to be *legible* (a reason, a dropped-batch
list, a recorded environment), not merely present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyscx")
pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

_HAS_CONSENSUS = hasattr(pytest.importorskip("pyscx"), "doublet_consensus")
pytestmark = pytest.mark.skipif(
    not _HAS_CONSENSUS,
    reason="pyscx build predates `doublet_consensus` (interop Phase 7)",
)


# ---------------------------------------------------------------------------
# Metric unit tests — no fixture, no files
# ---------------------------------------------------------------------------


def _metrics():
    import importlib

    return importlib.import_module(
        "benchmarks.comprehensive.scripts.doublet.metrics"
    )


def test_pairwise_agreement_ignores_rows_only_one_tool_voted_on():
    # The Phase-7 null rule, one level up. Tool B never saw rows 2 and 3. If
    # those counted as agreement (both "not called"), agreement would be 4/4;
    # restricted to jointly-voted rows it is 1/2.
    m = _metrics()
    a = pd.array([True, False, True, False], dtype="boolean")
    b = pd.array([True, True, None, None], dtype="boolean")

    r = m.pairwise_call_agreement(a, b)

    assert r["n_both_voted"] == 2
    assert r["n_compared_of"] == 4
    assert r["agreement"] == pytest.approx(0.5)
    assert r["n_agree"] == 1 and r["n_disagree"] == 1


def test_kappa_is_undefined_rather_than_faked_when_one_class():
    # Two tools that both call everything a singlet agree 100% by construction.
    # Reporting kappa=1.0 (or 0.0) there would read as a finding about the
    # tools; it is a degenerate comparison and has to say so.
    m = _metrics()
    a = pd.array([False, False, False], dtype="boolean")
    b = pd.array([False, False, False], dtype="boolean")

    r = m.pairwise_call_agreement(a, b)

    assert r["agreement"] == pytest.approx(1.0)
    assert r["kappa"] is None
    assert "undefined" in r["reason"]


def test_kappa_corrects_for_chance():
    # 10 cells, each tool calls 1, and they agree on it. Raw agreement is a
    # flattering 100%; kappa is 1.0 here because they agree perfectly, but the
    # point is that the correction is applied at all.
    m = _metrics()
    a = pd.array([True] + [False] * 9, dtype="boolean")
    b = pd.array([True] + [False] * 9, dtype="boolean")
    assert m.pairwise_call_agreement(a, b)["kappa"] == pytest.approx(1.0)

    # Now they each call one cell, but different ones: raw agreement is still
    # 80%, and kappa is negative — worse than chance.
    c = pd.array([True] + [False] * 9, dtype="boolean")
    d = pd.array([False, True] + [False] * 8, dtype="boolean")
    r = m.pairwise_call_agreement(c, d)
    assert r["agreement"] == pytest.approx(0.8)
    assert r["kappa"] < 0


def test_spearman_is_rank_based_and_scale_free():
    # The tools' scores live on different scales; only the ordering compares.
    m = _metrics()
    a = [0.1, 0.2, 0.3, 0.4]
    b = [10.0, 200.0, 3000.0, 40000.0]
    r = m.score_rank_correlation(a, b)
    assert r["spearman"] == pytest.approx(1.0)
    assert r["n_both_scored"] == 4

    rev = m.score_rank_correlation(a, list(reversed(b)))
    assert rev["spearman"] == pytest.approx(-1.0)


def test_spearman_skips_rows_only_one_tool_scored():
    m = _metrics()
    r = m.score_rank_correlation([1.0, 2.0, 3.0, np.nan],
                                 [1.0, 2.0, 3.0, 99.0])
    assert r["n_both_scored"] == 3
    assert r["spearman"] == pytest.approx(1.0)


def test_spearman_declines_rather_than_returning_nan():
    m = _metrics()
    r = m.score_rank_correlation([1.0, 2.0], [1.0, 2.0])
    assert r["spearman"] is None
    assert "at least 3" in r["reason"]


def test_auroc_matches_a_hand_computed_case():
    m = _metrics()
    # Perfect separation.
    assert m.auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    # Perfectly wrong.
    assert m.auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    # All ties -> 0.5.
    assert m.auroc([0.5] * 4, [False, False, True, True]) == pytest.approx(0.5)
    # Undefined with one class.
    assert m.auroc([0.1, 0.2], [False, False]) is None


def test_truth_metrics_stratify_recall_by_doublet_kind():
    # Homotypic doublets are the hard ones; a pooled recall dominated by
    # heterotypic pairs is exactly the aggregate DOUBLET-DETECTION.md warns
    # against.
    m = _metrics()
    r = m.truth_metrics(
        scores=[0.9, 0.9, 0.1, 0.1],
        pred=pd.array([True, False, False, False], dtype="boolean"),
        truth_label=["doublet", "doublet", "singlet", "singlet"],
        kinds=["heterotypic", "homotypic", "real", "real"],
    )
    assert r["recall"] == pytest.approx(0.5)
    assert r["recall_by_kind"]["heterotypic"]["recall"] == pytest.approx(1.0)
    assert r["recall_by_kind"]["homotypic"]["recall"] == pytest.approx(0.0)
    assert r["precision"] == pytest.approx(1.0)


def test_called_rate_by_batch_splits_per_batch():
    m = _metrics()
    r = m.called_rate_by_batch(
        pd.array([True, True, False, False], dtype="boolean"),
        ["d1", "d1", "d2", "d2"],
    )
    assert r["d1"]["called_rate"] == pytest.approx(1.0)
    assert r["d2"]["called_rate"] == pytest.approx(0.0)


def test_roundtrip_fidelity_catches_a_shifted_join():
    m = _metrics()
    ok = m.roundtrip_fidelity([0.1, 0.2, 0.3],
                              pd.array([True, False, True], dtype="boolean"),
                              [0.1, 0.2, 0.3], [True, False, True])
    assert ok["max_score_delta"] == pytest.approx(0.0)
    assert ok["call_disagreements"] == 0

    # The same values, one row out of step — a positional import.
    bad = m.roundtrip_fidelity([0.2, 0.3, 0.1],
                               pd.array([False, True, True], dtype="boolean"),
                               [0.1, 0.2, 0.3], [True, False, True])
    assert bad["max_score_delta"] > 0.05
    assert bad["call_disagreements"] > 0


def test_roundtrip_refuses_misaligned_arrays():
    m = _metrics()
    with pytest.raises(ValueError, match="key-aligned"):
        m.roundtrip_fidelity([0.1, 0.2], [True, False], [0.1], [True])


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def _inject_mod():
    import importlib

    return importlib.import_module(
        "benchmarks.comprehensive.scripts.doublet.inject_doublets"
    )


def _tiny_adata(n_obs=60, n_vars=40, seed=0, with_types=True):
    import anndata as ad

    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(
        rng.integers(0, 8, size=(n_obs, n_vars)).astype(np.float32)
    )
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    if with_types:
        obs["cell_type"] = np.repeat(["T", "B", "NK"], n_obs // 3)[:n_obs]
    obs["donor_id"] = np.where(np.arange(n_obs) < n_obs // 2, "d1", "d2")
    # A NUMERIC and a CATEGORICAL column, not just strings. Real obs carries
    # both (`n_counts` on pbmc3k, `cell_type` on the atlas), and a fixture of
    # pure strings hid a dtype bug in the injection until it reached a real
    # dataset — the inherited rows became object-dtype and anndata refused to
    # write them as vlen strings.
    obs["n_counts"] = np.asarray(X.sum(axis=1)).ravel().astype("float64")
    obs["lane"] = pd.Categorical(np.where(np.arange(n_obs) % 2 == 0, "L1", "L2"))
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_injected_doublets_are_exactly_the_sum_of_their_parents(tmp_path):
    import anndata as ad

    mod = _inject_mod()
    src = _tiny_adata()
    out = tmp_path / "inj.h5ad"
    res = mod.inject_doublets(src, out, mod.InjectionSpec(rate=0.2, seed=7),
                              cell_type_key="cell_type")

    got = ad.read_h5ad(out)
    assert got.n_obs == res.n_real + res.n_injected
    dense = np.asarray(got.X.todense())
    names = list(got.obs_names)

    injected = got.obs[got.obs[mod.TRUTH_LABEL] == mod.DOUBLET]
    assert len(injected) == res.n_injected
    for name, row in injected.iterrows():
        a = names.index(row[mod.TRUTH_SOURCE_A])
        b = names.index(row[mod.TRUTH_SOURCE_B])
        # The recorded parents must reconstruct the row. Without this the
        # source-pair columns are decoration and a mislabelled truth set would
        # go unnoticed — every accuracy number downstream would be wrong.
        assert np.array_equal(dense[names.index(name)], dense[a] + dense[b])


def test_injection_never_pairs_a_cell_with_itself(tmp_path):
    # A self-pair is a scaled single cell labelled "doublet" — poison in the
    # truth set, and invisible unless checked.
    mod = _inject_mod()
    out = tmp_path / "inj.h5ad"
    mod.inject_doublets(_tiny_adata(), out, mod.InjectionSpec(rate=0.5, seed=3))

    import anndata as ad
    obs = ad.read_h5ad(out).obs
    inj = obs[obs[mod.TRUTH_LABEL] == mod.DOUBLET]
    # `.astype(str)` because anndata stores string obs columns as Categoricals,
    # and two Categoricals with different category sets refuse to compare.
    assert (inj[mod.TRUTH_SOURCE_A].astype(str)
            != inj[mod.TRUTH_SOURCE_B].astype(str)).all()


def test_injection_labels_homotypic_and_heterotypic_apart(tmp_path):
    import anndata as ad

    mod = _inject_mod()
    out = tmp_path / "inj.h5ad"
    res = mod.inject_doublets(_tiny_adata(), out,
                              mod.InjectionSpec(rate=0.4, seed=11),
                              cell_type_key="cell_type")
    obs = ad.read_h5ad(out).obs
    inj = obs[obs[mod.TRUTH_LABEL] == mod.DOUBLET]
    assert set(inj[mod.TRUTH_KIND]) <= {"heterotypic", "homotypic"}
    assert res.n_heterotypic + res.n_homotypic == res.n_injected
    # With three roughly equal cell types, both kinds must actually occur —
    # otherwise the stratified recall silently reports one number twice.
    assert res.n_heterotypic > 0
    assert res.n_homotypic > 0

    # And each label must match its own parents. Asserting only that both
    # kinds appear would pass on a labeller that assigned them at random, or
    # that called everything heterotypic in a fixture where that happens to be
    # the majority — homotypic doublets are the hard case and mislabelling
    # them would flatter every recall number computed from this truth set.
    types = obs["cell_type"].astype(str)
    for name, row in inj.iterrows():
        a, b = types[row[mod.TRUTH_SOURCE_A]], types[row[mod.TRUTH_SOURCE_B]]
        expected = "homotypic" if a == b else "heterotypic"
        assert row[mod.TRUTH_KIND] == expected, (name, a, b)


def test_heterotypic_only_never_emits_a_homotypic_pair(tmp_path):
    import anndata as ad

    mod = _inject_mod()
    out = tmp_path / "inj.h5ad"
    res = mod.inject_doublets(
        _tiny_adata(), out,
        mod.InjectionSpec(rate=0.3, seed=5, heterotypic_only=True),
        cell_type_key="cell_type",
    )
    assert res.n_homotypic == 0
    obs = ad.read_h5ad(out).obs
    inj = obs[obs[mod.TRUTH_LABEL] == mod.DOUBLET]
    assert set(inj[mod.TRUTH_KIND]) == {"heterotypic"}


def test_kind_is_unknown_rather_than_guessed_without_cell_types(tmp_path):
    import anndata as ad

    mod = _inject_mod()
    out = tmp_path / "inj.h5ad"
    mod.inject_doublets(_tiny_adata(with_types=False), out,
                        mod.InjectionSpec(rate=0.2, seed=1))
    obs = ad.read_h5ad(out).obs
    inj = obs[obs[mod.TRUTH_LABEL] == mod.DOUBLET]
    assert set(inj[mod.TRUTH_KIND]) == {"unknown"}


def test_injected_rows_inherit_the_batch_so_they_are_not_silently_dropped(tmp_path):
    # A synthetic row with a missing batch would vanish at the per-batch export
    # and never be scored, quietly shrinking the truth set.
    import anndata as ad

    mod = _inject_mod()
    out = tmp_path / "inj.h5ad"
    mod.inject_doublets(_tiny_adata(), out, mod.InjectionSpec(rate=0.2, seed=2))
    obs = ad.read_h5ad(out).obs
    assert obs["donor_id"].isna().sum() == 0
    assert set(obs["donor_id"]) == {"d1", "d2"}


def test_injection_records_its_evidence_type(tmp_path):
    # Category D truth must never be read as a hashing/genotype result; the
    # caveat travels with the data rather than living only in a doc.
    import anndata as ad

    mod = _inject_mod()
    out = tmp_path / "inj.h5ad"
    mod.inject_doublets(_tiny_adata(), out, mod.InjectionSpec(rate=0.2, seed=0))
    rec = ad.read_h5ad(out).uns["doublet_injection"]
    assert rec["category"] == "injected"
    assert "ambient" in rec["evidence_type"]


def test_an_impossible_heterotypic_request_fails_loudly(tmp_path):
    mod = _inject_mod()
    single = _tiny_adata(with_types=False)
    single.obs["cell_type"] = "only_one"
    with pytest.raises(RuntimeError, match="distinct"):
        mod.inject_doublets(
            single, tmp_path / "x.h5ad",
            mod.InjectionSpec(rate=0.2, seed=0, heterotypic_only=True),
            cell_type_key="cell_type",
        )


def test_a_bad_rate_is_refused(tmp_path):
    mod = _inject_mod()
    for rate in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="rate"):
            mod.inject_doublets(_tiny_adata(), tmp_path / "x.h5ad",
                                mod.InjectionSpec(rate=rate))


# ---------------------------------------------------------------------------
# Tool env resolution
# ---------------------------------------------------------------------------


def _env_mod():
    import importlib

    return importlib.import_module(
        "benchmarks.comprehensive.scripts.doublet._tool_env"
    )


def test_a_missing_tool_env_names_the_env_and_the_fix(monkeypatch, tmp_path):
    mod = _env_mod()
    monkeypatch.setenv("SCX_DOUBLET_SCRUBLET_PREFIX", str(tmp_path / "nope"))
    with pytest.raises(mod.ToolUnavailable) as ei:
        mod.resolve("scrublet")
    msg = str(ei.value)
    assert "scx-bench" in msg          # which env
    assert "scikit-image" in msg       # what to install
    assert str(tmp_path / "nope") in msg   # where it looked


def test_probe_reports_unavailable_rather_than_raising(monkeypatch, tmp_path):
    mod = _env_mod()
    monkeypatch.setenv("SCX_DOUBLET_SOLO_PREFIX", str(tmp_path / "nope"))
    rec = mod.probe("solo")
    assert rec["available"] is False
    assert rec["reason"]


def test_an_unknown_tool_is_refused():
    mod = _env_mod()
    with pytest.raises(mod.ToolUnavailable, match="unknown doublet tool"):
        mod.resolve("not_a_tool")


# ---------------------------------------------------------------------------
# End-to-end orchestration, with a stub tool
# ---------------------------------------------------------------------------


# A stand-in for a doublet caller. Scores by total counts, so an injected
# doublet (two cells summed) genuinely ranks above the singlets and the
# accuracy metrics measure something rather than noise.
_STUB = '''
import json, sys
cfg = json.loads(sys.stdin.read())
import anndata as ad, numpy as np, pandas as pd
a = ad.read_h5ad(cfg["h5ad_path"])
names = a.obs_names.astype(str)
tot = np.asarray(a.X.sum(axis=1)).ravel().astype("float64")
score = (tot - tot.min()) / max(tot.max() - tot.min(), 1e-9)
called = score >= np.quantile(score, 0.90)
pd.DataFrame({"barcode": names, SCORE_COL: score,
              CALL_COL: CALL_VALUES}).to_csv(cfg["out_csv"], index=False)
print(json.dumps({"tool": TOOL, "available": True,
                  "n_cells": int(a.n_obs), "n_called": int(called.sum()),
                  "score_column": SCORE_COL, "call_column": CALL_COL,
                  "versions": {"stub": "1.0"}}))
'''


def _stub_source(tool: str, score_col: str, call_col: str,
                 call_values: str) -> str:
    return (_STUB
            .replace("TOOL", repr(tool))
            .replace("SCORE_COL", repr(score_col))
            .replace("CALL_COL", repr(call_col))
            .replace("CALL_VALUES", call_values))


def _install_stub_tool(tmp_path, monkeypatch, tool: str, source: str) -> None:
    """Stand up a fake conda env whose bin/python runs *source*.

    The shim forwards ``-c`` to the real interpreter so the environment probe
    still works — a real conda python would, and a shim that swallowed every
    invocation would make the probe fail and the tool look unavailable, which
    is precisely the silent-skip failure this suite is guarding against.
    """
    env = tmp_path / f"stubenv_{tool}"
    (env / "bin").mkdir(parents=True, exist_ok=True)
    script = tmp_path / f"stub_{tool}.py"
    script.write_text(source)
    shim = env / "bin" / "python"
    shim.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "-c" ]; then exec "{sys.executable}" "$@"; fi\n'
        f'exec "{sys.executable}" "{script}"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv(f"SCX_DOUBLET_{tool.upper()}_PREFIX", str(env))

    from benchmarks.comprehensive.scripts.doublet import _tool_env, run_tool

    # The probe asserts the tool's real packages; a stub has none of them, so
    # point it at a stdlib module. This test exercises the probe MECHANISM
    # (does an unavailable tool surface as a legible skip?), not whether
    # scanpy happens to be installed in the interpreter running pytest.
    monkeypatch.setitem(_tool_env._PROBE_PACKAGES, tool, ["json"])
    original = _tool_env.TOOL_ENVS[tool]
    monkeypatch.setitem(
        _tool_env.TOOL_ENVS, tool,
        _tool_env.ToolEnv(tool=tool, kind="python", env_name=original.env_name,
                          package="json", install_hint=original.install_hint),
    )
    # Every stub reads an h5ad; the real scDblFinder takes the .scx.
    monkeypatch.setitem(run_tool.INPUT_KIND, tool, "h5ad")


@pytest.fixture()
def bench_env(tmp_path, monkeypatch):
    """Isolated work dir, a tiny dataset, and a stub standing in for Scrublet."""
    work = tmp_path / "work"
    data = work / "benchmarks" / "datasets"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SCX_WORK_DIR", str(work))
    monkeypatch.setenv("SCX_DATA_DIR", str(data))
    for m in (
        "benchmarks.comprehensive.bench_env",
        "benchmarks.comprehensive.config",
        "benchmarks.comprehensive.benchmarks.doublet_interop",
    ):
        sys.modules.pop(m, None)

    from benchmarks.comprehensive.config import DatasetConfig, FormatVariant

    ds = DatasetConfig(id="DBL", name="doublet_tiny", n_obs=60, n_vars=40,
                       protocol="synthetic", source="test", approx_h5ad_mb=1)
    _tiny_adata().write_h5ad(ds.h5ad_path)

    _install_stub_tool(tmp_path, monkeypatch, "scrublet",
                       _stub_source("scrublet", "doublet_score",
                                    "predicted_doublet", "called"))
    # scDblFinder and solo stay unresolvable, so the skip path is exercised on
    # every run of this fixture.
    monkeypatch.setenv("SCX_DOUBLET_SCDBLFINDER_PREFIX", str(tmp_path / "none"))
    monkeypatch.setenv("SCX_DOUBLET_SOLO_PREFIX", str(tmp_path / "none"))

    fv = FormatVariant("SCX (auto)", "scx_auto", "primary", "scx_runner",
                       {"codec": "auto"})
    return {"ds": ds, "fv": fv, "tmp_path": tmp_path,
            "monkeypatch": monkeypatch}


def _mod():
    import importlib

    return importlib.import_module(
        "benchmarks.comprehensive.benchmarks.doublet_interop"
    )


def _configure(mod, ds_name, **kw):
    """Register the tiny dataset in DOUBLET_SPEC for one test."""
    from benchmarks.comprehensive.scripts.doublet.inject_doublets import (
        InjectionSpec,
    )

    defaults = dict(
        category="injected", evidence_type="test",
        inject=InjectionSpec(rate=0.2, seed=0),
        cell_type_key="cell_type", tools=("scrublet",),
    )
    defaults.update(kw)
    mod.DOUBLET_SPEC[ds_name] = mod.DoubletDatasetSpec(**defaults)


def test_registered_in_all_benchmarks():
    from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS

    assert "doublet_interop" in ALL_BENCHMARKS


def test_skips_non_scx_formats(bench_env):
    from benchmarks.comprehensive.config import FormatVariant

    mod = _mod()
    # Register the dataset FIRST: without it the spec gate returns None and
    # this passes whether or not the format gate exists.
    _configure(mod, bench_env["ds"].name)
    h5ad_fv = FormatVariant("h5ad", "h5ad_gzip", "baseline", "h5ad_runner", {})
    assert mod.run(bench_env["ds"], h5ad_fv, n_runs=1) is None


def test_skips_a_dataset_with_no_spec(bench_env):
    mod = _mod()
    mod.DOUBLET_SPEC.pop(bench_env["ds"].name, None)
    assert mod.run(bench_env["ds"], bench_env["fv"], n_runs=1) is None


def test_end_to_end_produces_a_gateable_result(bench_env):
    mod = _mod()
    _configure(mod, bench_env["ds"].name)

    result = mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)

    assert result is not None
    assert result.benchmark == "doublet_interop"
    assert result.overall_passed is True

    extra = result.runs[0].extra
    # The plumbing tier: the tool's own numbers survived the key-joined import.
    assert extra["roundtrip_call_disagreements"] == 0
    assert extra["roundtrip_max_score_delta"] < 1e-5
    assert extra["import_matched_frac"] == pytest.approx(1.0)
    # The science tier: injected doublets are two cells summed, so a
    # total-counts score must separate them well.
    assert extra["scrublet_auroc"] > 0.9


def test_the_environment_of_every_tool_is_recorded(bench_env):
    # Checklist item 5. Which env produced a number is not derivable after the
    # fact, and per-job env routing is a known failure mode on this cluster.
    mod = _mod()
    _configure(mod, bench_env["ds"].name, tools=("scrublet", "solo"))

    result = mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)

    envs = result.metadata["tool_environments"]
    assert set(envs) == {"scrublet", "solo"}
    assert envs["scrublet"]["available"] is True
    assert envs["solo"]["available"] is False
    assert envs["solo"]["reason"]


def test_an_unavailable_tool_is_a_legible_skip_not_a_crash(bench_env):
    mod = _mod()
    _configure(mod, bench_env["ds"].name, tools=("scrublet", "scdblfinder"))

    result = mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)

    assert result.metadata["tools_ran"] == ["scrublet"]
    skipped = result.metadata["tools_skipped"]
    assert "scdblfinder" in skipped
    assert "rscx" in skipped["scdblfinder"]      # names the env
    assert result.runs[0].extra["n_tools_skipped"] == 1


def test_a_batch_cap_records_what_it_dropped(bench_env):
    # A silently truncated cohort reads downstream as full coverage. The
    # fixture has two donors; capping at one must say so.
    mod = _mod()
    _configure(mod, bench_env["ds"].name, batch_key="donor_id", max_batches=1)

    result = mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)

    assert result.metadata["batches_run"] == ["d1"]
    assert result.metadata["batches_dropped_by_cap"] == ["d2"]
    assert result.runs[0].extra["n_batches_dropped"] == 1


def test_per_batch_called_rates_are_reported(bench_env):
    mod = _mod()
    _configure(mod, bench_env["ds"].name, batch_key="donor_id")

    result = mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)

    rates = result.metadata["per_tool"]["scrublet"]["called_rate_by_batch"]
    assert set(rates) == {"d1", "d2"}
    assert all(0.0 <= v["called_rate"] <= 1.0 for v in rates.values())


def test_the_evidence_type_travels_with_the_numbers(bench_env):
    mod = _mod()
    _configure(mod, bench_env["ds"].name,
               evidence_type="computational injection (Category D)")

    result = mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)

    assert result.metadata["category"] == "injected"
    assert "Category D" in result.metadata["evidence_type"]


def test_an_unknown_call_token_from_a_tool_is_refused(bench_env):
    # A tool emitting a class token its own profile does not know must stop the
    # run, naming the value. Which layer refuses is not the contract — as it
    # happens `doublet_import` catches this one first, with a better message
    # than the benchmark's own comparison would have produced — but *something*
    # has to, because the alternative is a call column read as all-False and a
    # round-trip check that then passes on nonsense.
    mod = _mod()
    _configure(mod, bench_env["ds"].name)
    _install_stub_tool(
        bench_env["tmp_path"], bench_env["monkeypatch"], "scrublet",
        _stub_source("scrublet", "doublet_score", "predicted_doublet",
                     'np.where(called, "AMBIGUOUS", "singlet")'),
    )

    with pytest.raises((RuntimeError, ValueError)) as ei:
        mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)
    assert "singlet" in str(ei.value) or "AMBIGUOUS" in str(ei.value)


def test_the_roundtrip_comparison_refuses_a_token_it_cannot_map():
    # The benchmark's own guard, unit-tested directly. It covers the narrower
    # case the importer cannot: a token the *profile* accepts but that this
    # comparison would otherwise coerce, silently making the round-trip check
    # agree with itself.
    mod = _mod()
    assert list(mod._tool_calls(pd.Series(["doublet", "singlet"]),
                                "scdblfinder")) == [True, False]
    assert list(mod._tool_calls(pd.Series([True, False]), "scrublet")) == [True, False]
    with pytest.raises(RuntimeError, match="unrecognised call tokens"):
        mod._tool_calls(pd.Series(["maybe", "singlet"]), "scdblfinder")


def test_a_failing_tool_names_the_env_and_surfaces_its_stderr(bench_env):
    mod = _mod()
    _configure(mod, bench_env["ds"].name)
    shim = bench_env["tmp_path"] / "stubenv_scrublet" / "bin" / "python"
    shim.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "-c" ]; then exec "{sys.executable}" "$@"; fi\n'
        'echo "boom" >&2\nexit 3\n'
    )
    shim.chmod(0o755)

    with pytest.raises(RuntimeError) as ei:
        mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)
    msg = str(ei.value)
    assert "scrublet" in msg and "exit 3" in msg
    assert "boom" in msg          # the tool's own stderr survives


def test_two_tools_produce_pairwise_agreement_and_a_consensus(bench_env):
    # The comparison the checklist asks for needs two arms. A second stub
    # stands in for scDblFinder so the agreement/consensus path is covered
    # without R — including its native `scDblFinder.class` token spellings,
    # which is what makes the doublet_import profile do real work.
    mod = _mod()
    _configure(mod, bench_env["ds"].name, tools=("scrublet", "scdblfinder"))
    _install_stub_tool(
        bench_env["tmp_path"], bench_env["monkeypatch"], "scdblfinder",
        _stub_source("scdblfinder", "scDblFinder.score", "scDblFinder.class",
                     'np.where(called, "doublet", "singlet")'),
    )

    result = mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)

    assert sorted(result.metadata["tools_ran"]) == ["scdblfinder", "scrublet"]
    pair = result.metadata["pairwise"]
    assert len(pair) == 1
    entry = next(iter(pair.values()))
    assert entry["n_both_voted"] > 0
    assert entry["spearman"] is not None
    # Both stubs score by total counts, so they must agree completely.
    assert entry["agreement"] == pytest.approx(1.0)
    assert result.metadata["consensus"]["method"] == "majority"
    extra = result.runs[0].extra
    assert "agreement_raw__scrublet__scdblfinder" in extra
    assert "score_spearman__scrublet__scdblfinder" in extra


def test_a_key_unique_only_within_batches_is_refused_before_any_tool_runs(
        bench_env, tmp_path):
    # The nastiest key case, and the one `export_batches` alone does NOT catch:
    # barcodes repeated ACROSS donors are unique within every batch, so the
    # per-batch guard passes, and only the global join is ambiguous. Left
    # unchecked the tools would run for hours and the import would then either
    # refuse or — worse — put one donor's scores on another's cells.
    import anndata as ad

    mod = _mod()
    ds = bench_env["ds"]
    src = _tiny_adata()
    # Same 30 barcodes in each donor.
    half = src.n_obs // 2
    src.obs_names = [f"bc_{i % half}" for i in range(src.n_obs)]
    src.write_h5ad(ds.h5ad_path)
    _configure(mod, ds.name, batch_key="donor_id")

    with pytest.raises(RuntimeError) as ei:
        mod.run(ds, bench_env["fv"], n_runs=1)
    msg = str(ei.value)
    # Specifically the benchmark's own pre-flight refusal, not `doublet_import`
    # rejecting the duplicate key later. The importer does also catch this —
    # but only after every tool has already run, which on the real atlas slice
    # is hours of scDblFinder spent to learn something checkable in a second.
    assert "not unique across this file" in msg
    assert "composite-key path is not implemented" in msg


def test_a_tool_that_reports_unavailable_at_run_time_is_a_skip(bench_env):
    # Distinct from the probe path: here the environment resolves and the
    # probe's packages import, but the runner itself decides it cannot go —
    # a real Solo case, since `scvi` can import and still fail to find a GPU
    # or a compatible torch. It must degrade to a recorded skip, not a crash,
    # and not to a tool silently counted as having run and called nothing.
    mod = _mod()
    _configure(mod, bench_env["ds"].name, tools=("scrublet",))
    _install_stub_tool(
        bench_env["tmp_path"], bench_env["monkeypatch"], "scrublet",
        'import json, sys\n'
        'sys.stdin.read()\n'
        'print(json.dumps({"tool": "scrublet", "available": False,\n'
        '                  "reason": "no GPU visible"}))\n',
    )

    result = mod.run(bench_env["ds"], bench_env["fv"], n_runs=1)

    assert result.metadata["tools_ran"] == []
    assert result.metadata["tools_skipped"]["scrublet"] == "no GPU visible"
    assert result.runs[0].extra["n_tools_ran"] == 0
    # No tool ran, so there is nothing to claim passed.
    assert result.overall_passed is False


def test_injection_preserves_obs_column_dtypes(tmp_path):
    # The bug this pins reached a real capture: inheriting the parent row
    # column-by-column through `object` turned `n_counts` into Python floats,
    # the concatenated column became object-dtype, and anndata then tried to
    # write it as variable-length strings. It failed at h5ad-write time, long
    # after the injection "succeeded".
    import anndata as ad

    mod = _inject_mod()
    src = _tiny_adata()
    out = tmp_path / "inj.h5ad"
    mod.inject_doublets(src, out, mod.InjectionSpec(rate=0.2, seed=0),
                        cell_type_key="cell_type")

    got = ad.read_h5ad(out).obs
    assert got["n_counts"].dtype.kind == "f", got["n_counts"].dtype
    assert str(got["lane"].dtype) == "category"
    # And the inherited values are the first parent's, not nulls.
    assert got["n_counts"].isna().sum() == 0
    assert got["lane"].isna().sum() == 0
