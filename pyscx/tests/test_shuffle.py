"""1D — `pyscx.shuffle` / `scx sort --shuffle`: seeded global row permutation.

The engine-level correctness gates live in `scx-ops/src/sort_engine_tests.rs`
(strategy-differential, obsp remap, multimodal lockstep). These tests cover the
*Python and CLI surfaces*: the round-trip a consumer actually sees, the
reproducibility contract, and the argument validation.
"""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def clustered_adata():
    """40 cells whose obs order is strongly clustered by `cell_type` — the
    layout a shuffle exists to destroy. Each cell's X row is a unique signature
    (row index in column 0) so a desynced permutation is detectable by value,
    not just by cell_id."""
    import anndata

    rng = np.random.default_rng(11)
    n_obs, n_vars = 40, 12
    dense = rng.integers(0, 40, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.6] = 0
    # Column 0 is a per-row fingerprint: 1-based so it is never an implicit zero.
    dense[:, 0] = np.arange(1, n_obs + 1, dtype=np.float32)

    # Contiguous blocks: cells 0-9 A, 10-19 B, 20-29 C, 30-39 D.
    cell_type = np.repeat(["A", "B", "C", "D"], n_obs // 4)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(cell_type)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])
    return anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)


def _rows_by_cell(path):
    """{cell_id: dense X row} read back from an SCX file."""
    import pyscx

    adata = pyscx.open(path).to_anndata()
    dense = np.asarray(adata.X.todense())
    return {cid: dense[i].copy() for i, cid in enumerate(adata.obs_names)}


def _order(path):
    import pyscx

    return list(pyscx.open(path).read_obs().index)


# --- round-trip ------------------------------------------------------------


def test_shuffle_preserves_every_row_and_its_alignment(
    clustered_adata, scx_from_adata, tmp_dir
):
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    out = str(tmp_dir / "shuffled.scx")
    pyscx.shuffle(src, out, seed=42)

    before, after = _rows_by_cell(src), _rows_by_cell(out)
    assert set(before) == set(after), "row multiset must be preserved"
    for cid, row in after.items():
        np.testing.assert_array_equal(row, before[cid], err_msg=f"X row for {cid}")

    # obs is carried, not just cell_id.
    obs_after = pyscx.open(out).read_obs()
    obs_before = pyscx.open(src).read_obs()
    for cid in obs_after.index:
        assert obs_after.loc[cid, "cell_type"] == obs_before.loc[cid, "cell_type"]


def test_shuffle_actually_reorders(clustered_adata, scx_from_adata, tmp_dir):
    """The anti-tautology half: every assertion above also holds for a no-op."""
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    out = str(tmp_dir / "shuffled.scx")
    pyscx.shuffle(src, out, seed=42)
    assert _order(src) != _order(out)


def test_shuffle_breaks_up_the_clustered_blocks(
    clustered_adata, scx_from_adata, tmp_dir
):
    """The property the feature exists for, stated as a measurement rather than
    as "the order changed": in the input each cell_type occupies one contiguous
    run, so counting runs is a direct read of how clustered the file is."""
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    out = str(tmp_dir / "shuffled.scx")
    pyscx.shuffle(src, out, seed=42)

    def n_runs(path):
        labels = list(pyscx.open(path).read_obs()["cell_type"])
        return 1 + sum(a != b for a, b in zip(labels, labels[1:]))

    assert n_runs(src) == 4, "fixture must start perfectly clustered"
    # 40 cells over 4 labels: a uniform permutation gives ~30 runs. Anything
    # under 10 would mean residual block structure survived.
    assert n_runs(out) > 10, "shuffled file still looks clustered"


# --- reproducibility contract ----------------------------------------------


def test_same_seed_reproduces_the_same_order(
    clustered_adata, scx_from_adata, tmp_dir
):
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    a, b = str(tmp_dir / "a.scx"), str(tmp_dir / "b.scx")
    pyscx.shuffle(src, a, seed=1234)
    pyscx.shuffle(src, b, seed=1234)
    assert _order(a) == _order(b)


def test_different_seed_gives_a_different_order(
    clustered_adata, scx_from_adata, tmp_dir
):
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    a, b = str(tmp_dir / "a.scx"), str(tmp_dir / "b.scx")
    pyscx.shuffle(src, a, seed=1234)
    pyscx.shuffle(src, b, seed=1235)
    assert _order(a) != _order(b)


def test_seed_is_recorded_in_provenance(clustered_adata, scx_from_adata, tmp_dir):
    """The seed is the only record of the permutation — there is no key to
    re-derive it from — so a shuffled file that lost it is unreproducible."""
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    out = str(tmp_dir / "shuffled.scx")
    pyscx.shuffle(src, out, seed=7)

    entries = pyscx.open(out).provenance()
    sort_entries = [e for e in entries if e["action"] == "sort"]
    assert sort_entries, "shuffle records a sort provenance entry"
    params = json.loads(sort_entries[-1]["params_json"])
    assert params["shuffle"]["seed"] == 7
    assert params["by"] == []


def test_default_seed_is_42(clustered_adata, scx_from_adata, tmp_dir):
    """Matches `TrainingDataset(seed=42)`. Pinned because the default is what
    most callers will actually use, and changing it silently relayouts files."""
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    default, explicit = str(tmp_dir / "d.scx"), str(tmp_dir / "e.scx")
    pyscx.shuffle(src, default)
    pyscx.shuffle(src, explicit, seed=42)
    assert _order(default) == _order(explicit)


# --- composition with the rest of the op -----------------------------------


def test_shuffle_with_rebuild_csc(clustered_adata, scx_from_adata, tmp_dir):
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    out = str(tmp_dir / "shuffled.scx")
    pyscx.shuffle(src, out, seed=3, rebuild_csc=True)

    # The sidecar is rebuilt against the shuffled row order; reading it back
    # must agree with the row-major view cell-for-cell.
    assert pyscx.open(out).has_csc
    before, after = _rows_by_cell(src), _rows_by_cell(out)
    for cid, row in after.items():
        np.testing.assert_array_equal(row, before[cid])


def test_shuffle_under_a_memory_budget(clustered_adata, scx_from_adata, tmp_dir):
    """A budget routes the engine onto the external partition path. The
    permutation is computed in pass 0 either way, so the two must agree."""
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    plain = str(tmp_dir / "plain.scx")
    budgeted = str(tmp_dir / "budgeted.scx")
    pyscx.shuffle(src, plain, seed=5, shard_size=8)
    pyscx.shuffle(src, budgeted, seed=5, shard_size=8, memory_budget="1K")
    assert _order(plain) == _order(budgeted)
    assert _rows_by_cell(plain).keys() == _rows_by_cell(budgeted).keys()


def test_shuffle_survives_deletions(clustered_adata, scx_from_adata, tmp_dir):
    """Deletions are materialized away, as in `sort`. The permutation runs over
    live rows only, which is why a deleted-row file shuffles differently from
    the same file without deletions."""
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    pyscx.mark_deleted(src, [0, 5, 17])
    out = str(tmp_dir / "shuffled.scx")
    pyscx.shuffle(src, out, seed=9)

    ids = set(_order(out))
    assert len(ids) == 37
    assert {"cell_0", "cell_5", "cell_17"}.isdisjoint(ids)


# --- validation ------------------------------------------------------------


def test_shuffle_rejects_a_missing_input(tmp_dir):
    import pyscx

    with pytest.raises((RuntimeError, OSError, ValueError)):
        pyscx.shuffle(str(tmp_dir / "nope.scx"), str(tmp_dir / "out.scx"))


def test_shuffle_rejects_a_non_positive_shard_size(
    clustered_adata, scx_from_adata, tmp_dir
):
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    with pytest.raises(ValueError):
        pyscx.shuffle(src, str(tmp_dir / "out.scx"), shard_size=0)


def test_shuffle_signature_has_no_key_arguments():
    """`shuffle` is an order *source*, not `sort` with a flag. The engine
    rejects `by` / `group_by` / `reverse` alongside a shuffle, so this surface
    must not offer them at all — a `by=` that silently did nothing would be
    worse than the engine's refusal."""
    import inspect

    import pyscx

    params = set(inspect.signature(pyscx.shuffle).parameters)
    assert params.isdisjoint({"by", "reverse", "group_by", "reference"})
    assert "seed" in params


# --- CLI -------------------------------------------------------------------


def _scx_bin():
    """The *repo's* `scx`, never whatever is on PATH.

    There is a stale `scx` in `~/.local/bin` on at least one dev machine that
    predates the `sort` subcommand entirely; resolving through PATH turned these
    tests into a test of that binary. Skip when the repo binary has not been
    built rather than fall back."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        candidate = root / "target" / profile / "scx"
        if candidate.is_file():
            return str(candidate)
    return None


@pytest.mark.skipif(_scx_bin() is None, reason="scx CLI not on PATH")
def test_cli_shuffle_matches_the_python_surface(
    clustered_adata, scx_from_adata, tmp_dir
):
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    cli_out = str(tmp_dir / "cli.scx")
    py_out = str(tmp_dir / "py.scx")

    subprocess.run(
        [_scx_bin(), "sort", "--shuffle", "--seed", "13", src, cli_out],
        check=True,
        capture_output=True,
    )
    pyscx.shuffle(src, py_out, seed=13)
    assert _order(cli_out) == _order(py_out)


@pytest.mark.skipif(_scx_bin() is None, reason="scx CLI not on PATH")
@pytest.mark.parametrize(
    "extra,expected",
    [
        (["--by", "cell_type"], "--shuffle and --by"),
        (["--group-by", "cell_type"], "--shuffle and --group-by"),
        (["--reverse"], "--reverse is meaningless"),
    ],
)
def test_cli_rejects_shuffle_with_another_order_source(
    clustered_adata, scx_from_adata, tmp_dir, extra, expected
):
    src = scx_from_adata(clustered_adata, "src.scx")
    out = str(tmp_dir / f"out_{extra[0].strip('-')}.scx")
    proc = subprocess.run(
        [_scx_bin(), "sort", "--shuffle", *extra, src, out],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert expected in (proc.stderr + proc.stdout)


@pytest.mark.skipif(_scx_bin() is None, reason="scx CLI not on PATH")
def test_cli_seed_requires_shuffle(clustered_adata, scx_from_adata, tmp_dir):
    """`--seed` without `--shuffle` is a mistake worth catching at parse time —
    it would otherwise run a key sort and silently ignore the seed."""
    src = scx_from_adata(clustered_adata, "src.scx")
    proc = subprocess.run(
        [_scx_bin(), "sort", "--by", "cell_type", "--seed", "3", src,
         str(tmp_dir / "out.scx")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


# --- the two pre-write warnings --------------------------------------------
#
# These are diagnostics whose whole value is firing *before* a multi-hour
# rewrite, so "it compiles" is not evidence they work. Both halves are asserted:
# a warning that can never be silent is as useless as one that never fires.


def _shuffle_stderr(src, out, *extra):
    proc = subprocess.run(
        [_scx_bin(), "sort", "--shuffle", *extra, src, out],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stderr + proc.stdout


@pytest.mark.skipif(_scx_bin() is None, reason="scx CLI not on PATH")
def test_cli_warns_when_the_input_x_is_cross_row_coded(
    clustered_adata, scx_from_adata, tmp_dir
):
    src = scx_from_adata(clustered_adata, "src.scx")
    zstd_src = str(tmp_dir / "zstd.scx")
    scx1_src = str(tmp_dir / "scx1.scx")
    for dst, codec in ((zstd_src, "zstd"), (scx1_src, "scx1")):
        subprocess.run(
            [_scx_bin(), "sort", "--by", "cell_type", "--codec", codec, src, dst],
            check=True,
            capture_output=True,
        )

    needle = "compression spans rows"
    assert needle in _shuffle_stderr(zstd_src, str(tmp_dir / "z_out.scx"))
    assert needle not in _shuffle_stderr(scx1_src, str(tmp_dir / "s_out.scx"))

    # ...and an explicit --codec means the user already made the call.
    assert needle not in _shuffle_stderr(
        zstd_src, str(tmp_dir / "z_pinned.scx"), "--codec", "scx1"
    )


@pytest.mark.skipif(_scx_bin() is None, reason="scx CLI not on PATH")
def test_cli_size_warning_names_the_inputs_own_codec(
    clustered_adata, scx_from_adata, tmp_dir
):
    """The remediation the warning names must not *be* the failure mode.

    An earlier draft said "pass --codec scx1 for a size-neutral shuffle", which
    is true only relative to an scx1 input. On a shufdelta/zstd file the
    auto-mode growth IS the adaptive codec flipping to scx1, so that advice
    reproduces the blowup it warns about (tabula: 197.9 MB -> 413.3 MB either
    way). The size-preserving pin is the input's own codec."""
    src = scx_from_adata(clustered_adata, "src.scx")
    zstd_src = str(tmp_dir / "zstd.scx")
    subprocess.run(
        [_scx_bin(), "sort", "--by", "cell_type", "--codec", "zstd", src, zstd_src],
        check=True,
        capture_output=True,
    )
    out = _shuffle_stderr(zstd_src, str(tmp_dir / "out.scx"))
    assert "pin the input's own codec: `--codec zstd`" in out, out
    # ...and it must not present scx1 as the size-preserving option.
    assert "--codec scx1 for a size-neutral" not in out


@pytest.mark.skipif(_scx_bin() is None, reason="scx CLI not on PATH")
def test_cli_warns_when_the_shuffle_also_re_shards(
    clustered_adata, scx_from_adata, tmp_dir
):
    """Shard geometry is what quantises batch composition, so a shuffle that
    silently re-shards changes the very thing the user ran it to control.
    `--shard-size` defaults to 16384 (inherited from `sort`); this is the same
    trap that fabricated a 5.97x throughput ratio in 1D's own benchmark."""
    src = scx_from_adata(clustered_adata, "src.scx")
    sized = str(tmp_dir / "sized.scx")
    subprocess.run(
        [_scx_bin(), "sort", "--by", "cell_type", "--shard-size", "10",
         "--codec", "scx1", src, sized],
        check=True,
        capture_output=True,
    )
    needle = "differs from the input's"
    assert needle in _shuffle_stderr(sized, str(tmp_dir / "a.scx"))
    # Carrying the geometry through silences it — the negative half.
    assert needle not in _shuffle_stderr(
        sized, str(tmp_dir / "b.scx"), "--shard-size", "10"
    )


@pytest.mark.skipif(_scx_bin() is None, reason="scx CLI not on PATH")
def test_cli_warns_about_predicate_index_scatter_on_an_indexed_input(
    clustered_adata, scx_from_adata, tmp_dir
):
    """Gating this on the index *flags* alone left it silent in the common case:
    the rebuild also picks up auto-detected columns, so an already-indexed input
    re-emits an index even when `scx sort --shuffle` is passed no index flags at
    all. Observed on `pbmc10k_auto.scx`, which re-indexed `n_counts`."""
    src = scx_from_adata(clustered_adata, "src.scx")
    indexed = str(tmp_dir / "indexed.scx")
    subprocess.run(
        [_scx_bin(), "sort", "--by", "cell_type", "--index-obs", "cell_type",
         "--codec", "scx1", src, indexed],
        check=True,
        capture_output=True,
    )

    needle = "maximally scatters"
    # No index flags on the shuffle — the warning must still fire.
    assert needle in _shuffle_stderr(indexed, str(tmp_dir / "out.scx"))


# --- codec control ----------------------------------------------------------


def test_shuffle_preserves_a_pinned_codec(clustered_adata, scx_from_adata, tmp_dir):
    """Without a `codec=` kwarg the writer re-selects `auto` on every rewrite.

    This is not cosmetic: the 1D size benchmark swept six per-codec fixtures and
    got a byte-identical output from every one, because each was silently
    re-encoded to the same auto choice. The sweep was measuring auto-reselection,
    not whether a permutation grows that codec. Both `sort` and `shuffle` had the
    gap; the CLI never did."""
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    # Assert on *bytes*, not `Experiment.codec_id`: that attribute is the file
    # header's default, which the writer leaves at 0 while the real choice lives
    # in each shard header. Size is also the observable a caller pinning a codec
    # actually cares about.
    sizes = {}
    for codec in ("scx1", "none"):
        out = tmp_dir / f"{codec}.scx"
        pyscx.shuffle(src, str(out), seed=4, codec=codec)
        sizes[codec] = out.stat().st_size

    assert sizes["scx1"] < sizes["none"], (
        f"pinned codecs produced indistinguishable output: {sizes} — the kwarg "
        "is not reaching the writer"
    )


def test_sort_accepts_a_codec_too(clustered_adata, scx_from_adata, tmp_dir):
    """`docs/sharding.md` tells users to pin `--codec` when output size matters;
    before 1D that advice was unfollowable from Python."""
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    a, b = str(tmp_dir / "a.scx"), str(tmp_dir / "b.scx")
    pyscx.sort(src, a, by=["cell_type"], codec="scx1")
    pyscx.sort(src, b, by=["cell_type"], codec="none")
    import os

    assert os.path.getsize(a) < os.path.getsize(b)


def test_shuffle_rejects_an_unknown_codec(clustered_adata, scx_from_adata, tmp_dir):
    import pyscx

    src = scx_from_adata(clustered_adata, "src.scx")
    with pytest.raises(ValueError):
        pyscx.shuffle(src, str(tmp_dir / "out.scx"), codec="brotli")


@pytest.mark.skipif(_scx_bin() is None, reason="scx CLI not on PATH")
def test_cli_help_does_not_teach_the_wrong_codec_remediation():
    """`scx sort --help` is the first surface many users meet, so it must not
    contradict the runtime warning. It carried "pass `--codec scx1` for a
    size-neutral shuffle" for one commit *after* the engine warning had been
    rewritten to reject exactly that advice."""
    out = subprocess.run(
        [_scx_bin(), "sort", "--help"], capture_output=True, text=True, check=True
    ).stdout
    assert "--shuffle" in out
    assert "scx1` for a size-neutral" not in out
    assert "pin the INPUT's own codec" in out
