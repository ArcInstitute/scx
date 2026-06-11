"""GPU CSC-direct parity tests for `pyscx.accel.rank_genes_groups(device="gpu")`.

Exercises the v3 CSC-direct Wilcoxon route (`scx_accel::
wilcoxon_rank_sum_gpu_chunked_v3_csc`, recorded as `gpu_csc_v3`) on a backed
SCX file with a CSC sidecar, cross-checking numerics against the CPU path and
asserting the recorded route is the CSC-direct one.

GPU DE v3 is the unconditional default route (the former `SCX_GPU_DE_V3` gate
was removed), so a backed SCX file with a CSC sidecar always takes the
`gpu_csc_v3` route — the assertion below proves it.

Skipped cleanly when `pyscx.accel.gpu_available()` is `False`.
"""

from __future__ import annotations

import numpy as np
import pytest

import anndata as ad  # noqa: E402

import pyscx  # noqa: E402

from _pdex_fixtures import _make_adata, REFERENCE  # noqa: E402


pytestmark = pytest.mark.skipif(
    not pyscx.accel.gpu_available(),
    reason="GPU not available (pyscx not built with gpu feature, or no CUDA device)",
)


def _open_with_csc(path, adata: ad.AnnData) -> ad.AnnData:
    """Round-trip ``adata`` through a CSC-equipped SCX file, opened backed."""
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=5)
    return pyscx.open(str(path)).to_anndata(backed=True)


def _result_to_gene_dict(adata: ad.AnnData):
    rgg = adata.uns["rank_genes_groups"]
    groups = list(rgg["names"].dtype.names)
    out = {}
    for g in groups:
        names = rgg["names"][g]
        scores = rgg["scores"][g]
        pvals = rgg["pvals"][g]
        logfc = rgg["logfoldchanges"][g]
        out[g] = {
            str(name): (float(scores[i]), float(pvals[i]), float(logfc[i]))
            for i, name in enumerate(names)
        }
    return out


def _compare(cpu, gpu) -> None:
    assert set(cpu.keys()) == set(gpu.keys()), f"group set mismatch: {set(cpu)} vs {set(gpu)}"
    for g in cpu:
        cpu_g, gpu_g = cpu[g], gpu[g]
        assert set(cpu_g.keys()) == set(gpu_g.keys()), f"gene set mismatch in group {g}"
        for gene, (s_c, p_c, l_c) in cpu_g.items():
            s_g, p_g, l_g = gpu_g[gene]
            if np.isfinite(s_c) and np.isfinite(s_g):
                assert abs(s_c - s_g) < 1e-6 or abs(s_c - s_g) / max(abs(s_c), 1e-12) < 1e-6, (
                    f"score mismatch group={g} gene={gene}: cpu={s_c}, gpu={s_g}"
                )
            assert abs(p_c - p_g) < 1e-9 or abs(p_c - p_g) / max(p_c, 1e-12) < 1e-6, (
                f"pval mismatch group={g} gene={gene}: cpu={p_c}, gpu={p_g}"
            )
            if np.isfinite(l_c) and np.isfinite(l_g):
                assert abs(l_c - l_g) < 1e-4 or abs(l_c - l_g) / max(abs(l_c), 1e-9) < 1e-4, (
                    f"logfc mismatch group={g} gene={gene}: cpu={l_c}, gpu={l_g}"
                )


def _route(adata: ad.AnnData) -> str | None:
    try:
        return adata.uns["scx_accel"]["rank_genes_groups"]["route"]
    except Exception:
        return None


def _run_parity(tmp_path, reference) -> None:
    base = _make_adata()
    cpu = _open_with_csc(tmp_path / "cpu.scx", base)
    gpu = _open_with_csc(tmp_path / "gpu.scx", base)

    pyscx.accel.rank_genes_groups(cpu, "target", reference=reference, tie_correct=True, device="cpu")
    pyscx.accel.rank_genes_groups(gpu, "target", reference=reference, tie_correct=True, device="gpu")

    _compare(_result_to_gene_dict(cpu), _result_to_gene_dict(gpu))

    route = _route(gpu)
    assert route == "gpu_csc_v3", f"expected CSC-direct route, got {route!r}"


def test_rank_genes_groups_gpu_csc_parity_one_vs_rest(tmp_path) -> None:
    _run_parity(tmp_path, "rest")


def test_rank_genes_groups_gpu_csc_parity_vs_reference(tmp_path) -> None:
    _run_parity(tmp_path, REFERENCE)
