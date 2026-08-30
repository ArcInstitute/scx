"""Tests for the seeded count-downsample primitive (data-load Phase 1B).

Two surfaces are covered and deliberately cross-checked against each other:
``SparseCellSetDataset(downsample_*=...)``, which applies the draw inside the
gather, and the standalone ``pyscx.downsample_counts_csr``. They must agree —
otherwise a consumer that migrates between them silently changes its
augmentation.

Bit-parity with state3's numpy ``RawCountDownsampler`` is explicitly **not** a
goal (see ``scx-loader/src/downsample.rs``); what is pinned here is the
semantics, the distributions, and the invariances.
"""

import json
import pathlib

import numpy as np
import pytest

# The Rust-side golden fixture. Its source of truth is the Rust implementation
# (unlike encoder_crop_golden.json, whose source is state3's Python), and state3
# adopts a byte-identical copy.
_GOLDEN = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scx-loader"
    / "tests"
    / "data"
    / "downsample_golden.json"
)

_TARGET = 100


@pytest.fixture
def two_scx(synthetic_adata, tmp_dir):
    import pyscx

    p0 = str(tmp_dir / "f0.scx")
    p1 = str(tmp_dir / "f1.scx")
    pyscx.from_anndata(synthetic_adata, p0)
    pyscx.from_anndata(synthetic_adata, p1)
    return p0, p1


def _plan(file_ids, rows):
    """One single-set batch over the given (file, row) pairs."""
    n = len(rows)
    return (list(file_ids), list(rows), [0] * n, [0, n])


def _gather(ds, file_ids, rows):
    batches = list(ds.iter_with_plans(iter([_plan(file_ids, rows)])))
    assert len(batches) == 1
    return batches[0]


def _row(batch, j):
    lo, hi = int(batch["indptr"][j]), int(batch["indptr"][j + 1])
    return batch["indices"][lo:hi], batch["data"][lo:hi]


def _libs(batch):
    n = len(batch["indptr"]) - 1
    return [float(_row(batch, j)[1].sum()) for j in range(n)]


# ---------------------------------------------------------------------------
# The golden fixture, and the two surfaces agreeing on it
# ---------------------------------------------------------------------------


def test_standalone_primitive_reproduces_the_rust_golden():
    """The Python surface must reproduce the Rust fixture exactly.

    Without this, state3 could adopt the fixture and assert against it while the
    pyscx entry point it actually calls diverged.
    """
    import pyscx

    doc = json.loads(_GOLDEN.read_text())
    assert doc["cases"], "fixture has no cases"
    for case in doc["cases"]:
        counts = np.asarray(case["counts"], dtype=np.float32)
        indices = np.asarray(case["indices"], dtype=np.int32)
        out = pyscx.downsample_counts_csr(
            np.array([0, len(counts)], dtype=np.int64),
            indices,
            counts,
            np.array([case["row"]], dtype=np.uint64),
            np.array([case["file_identity"]], dtype=np.uint64),
            case["target_library_size"],
            case["method"],
            case["seed"],
        )
        np.testing.assert_array_equal(
            out["indices"], np.asarray(case["expected_indices"], dtype=np.int32),
            err_msg=f"indices mismatch [{case['name']}]",
        )
        np.testing.assert_array_equal(
            out["data"], np.asarray(case["expected_counts"], dtype=np.float32),
            err_msg=f"counts mismatch [{case['name']}]",
        )


def test_standalone_and_dataset_paths_agree(two_scx):
    """The two surfaces are one implementation; pin that they stay one.

    Gather without downsampling, downsample the result via the standalone
    primitive, and require it to equal what the loader produces when it does the
    draw itself.
    """
    import pyscx

    p0, _ = two_scx
    rows = [0, 3, 11, 42]

    plain = _gather(pyscx.SparseCellSetDataset([p0]), [0] * len(rows), rows)
    ident = pyscx.downsample_file_identity(p0)
    manual = pyscx.downsample_counts_csr(
        plain["indptr"],
        plain["indices"],
        plain["data"],
        np.asarray(rows, dtype=np.uint64),
        np.full(len(rows), ident, dtype=np.uint64),
        _TARGET,
        "multinomial",
        17,
    )

    inside = _gather(
        pyscx.SparseCellSetDataset(
            [p0],
            downsample_target_library_size=_TARGET,
            downsample_method="multinomial",
            downsample_seed=17,
        ),
        [0] * len(rows),
        rows,
    )
    np.testing.assert_array_equal(manual["indptr"], inside["indptr"])
    np.testing.assert_array_equal(manual["indices"], inside["indices"])
    np.testing.assert_array_equal(manual["data"], inside["data"])


# ---------------------------------------------------------------------------
# Target semantics
# ---------------------------------------------------------------------------


def test_multinomial_hits_the_target_exactly(two_scx):
    import pyscx

    p0, _ = two_scx
    ds = pyscx.SparseCellSetDataset(
        [p0],
        downsample_target_library_size=_TARGET,
        downsample_method="multinomial",
        downsample_seed=1,
    )
    b = _gather(ds, [0] * 8, list(range(8)))
    assert _libs(b) == [float(_TARGET)] * 8


def test_binomial_hits_the_target_in_expectation(two_scx):
    import pyscx

    p0, _ = two_scx
    ds = pyscx.SparseCellSetDataset(
        [p0],
        downsample_target_library_size=_TARGET,
        downsample_method="binomial",
        downsample_seed=1,
    )
    libs = _libs(_gather(ds, [0] * 60, list(range(60))))
    assert abs(np.mean(libs) - _TARGET) < 8, f"mean off target: {np.mean(libs)}"
    # If every draw landed exactly on target we would have written a multinomial.
    assert any(abs(l - _TARGET) > 0.5 for l in libs), "binomial draws all exact"


def test_without_downsample_counts_are_far_above_target(two_scx):
    """Anti-tautology for the two tests above.

    If the fixture's rows happened to sit below the target, the downsample would
    be a no-op and 'hits the target' would assert nothing.
    """
    import pyscx

    p0, _ = two_scx
    libs = _libs(_gather(pyscx.SparseCellSetDataset([p0]), [0] * 8, list(range(8))))
    assert min(libs) > _TARGET * 2, f"fixture too shallow to test a downsample: {libs}"


def test_never_upsamples_a_shallow_cell(two_scx):
    import pyscx

    p0, _ = two_scx
    plain = _libs(_gather(pyscx.SparseCellSetDataset([p0]), [0] * 8, list(range(8))))
    huge = max(plain) * 10
    for method in ("binomial", "multinomial"):
        ds = pyscx.SparseCellSetDataset(
            [p0],
            downsample_target_library_size=int(huge),
            downsample_method=method,
            downsample_seed=1,
        )
        got = _libs(_gather(ds, [0] * 8, list(range(8))))
        # Equal up to the integerisation below, never inflated toward `huge`.
        assert all(g <= p for g, p in zip(got, plain)), f"{method} upsampled"


def test_enabling_downsample_integerises_even_below_target(tmp_dir):
    """Mirrors the reference: turning downsampling on rints the whole corpus.

    Surprising enough that someone will read it as a bug later, so it is pinned
    with the reason attached.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    x = sp.csr_matrix(np.array([[1.4, 2.6, 0.0], [0.0, 3.5, 2.5]], dtype=np.float32))
    adata = anndata.AnnData(
        X=x,
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    p = str(tmp_dir / "frac.scx")
    pyscx.from_anndata(adata, p)

    ds = pyscx.SparseCellSetDataset(
        [p],
        downsample_target_library_size=10_000,  # far above every row
        downsample_method="multinomial",
        downsample_seed=1,
    )
    b = _gather(ds, [0, 0], [0, 1])
    # 1.4 -> 1, 2.6 -> 3; 3.5 -> 4 and 2.5 -> 2 (ties to EVEN, not away from zero).
    np.testing.assert_array_equal(_row(b, 0)[1], np.array([1.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(_row(b, 1)[1], np.array([4.0, 2.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Invariances — the properties the key design exists for
# ---------------------------------------------------------------------------


def test_deterministic_across_dataset_instances(two_scx):
    import pyscx

    p0, _ = two_scx

    def build():
        return pyscx.SparseCellSetDataset(
            [p0],
            downsample_target_library_size=_TARGET,
            downsample_method="multinomial",
            downsample_seed=5,
        )

    a = _gather(build(), [0] * 6, [1, 4, 9, 16, 25, 36])
    b = _gather(build(), [0] * 6, [1, 4, 9, 16, 25, 36])
    np.testing.assert_array_equal(a["data"], b["data"])
    np.testing.assert_array_equal(a["indices"], b["indices"])


def test_changing_the_seed_changes_the_draw(two_scx):
    """Anti-tautology for determinism: a config-ignoring implementation would
    otherwise pass the test above."""
    import pyscx

    p0, _ = two_scx
    rows = [1, 4, 9, 16, 25, 36]

    def libs_for(seed):
        ds = pyscx.SparseCellSetDataset(
            [p0],
            downsample_target_library_size=_TARGET,
            downsample_method="binomial",
            downsample_seed=seed,
        )
        return _libs(_gather(ds, [0] * len(rows), rows))

    assert libs_for(1) != libs_for(2)


def test_invariant_to_manifest_order(two_scx):
    """THE test that justifies keying on the resolved path.

    Under a ``file_id`` key — the obvious choice — reordering the manifest would
    silently redraw every cell while producing entirely plausible output.
    """
    import pyscx

    p0, p1 = two_scx
    rows = [2, 7, 13]

    def build(paths):
        return pyscx.SparseCellSetDataset(
            paths,
            downsample_target_library_size=_TARGET,
            downsample_method="multinomial",
            downsample_seed=9,
        )

    # p0 is file 0 in one manifest and file 1 in the other.
    a = _gather(build([p0, p1]), [0] * len(rows), rows)
    b = _gather(build([p1, p0]), [1] * len(rows), rows)
    np.testing.assert_array_equal(
        a["data"], b["data"], err_msg="the draw moved when the manifest was reordered"
    )


def test_invariant_to_a_manifest_subset(two_scx):
    """The debugging case: running on one file must draw as the full run did."""
    import pyscx

    p0, p1 = two_scx
    rows = [2, 7, 13]

    full = _gather(
        pyscx.SparseCellSetDataset(
            [p0, p1],
            downsample_target_library_size=_TARGET,
            downsample_seed=9,
        ),
        [0] * len(rows),
        rows,
    )
    subset = _gather(
        pyscx.SparseCellSetDataset(
            [p0],
            downsample_target_library_size=_TARGET,
            downsample_seed=9,
        ),
        [0] * len(rows),
        rows,
    )
    np.testing.assert_array_equal(full["data"], subset["data"])


def test_distinct_files_draw_distinctly(two_scx):
    """Guard on the above: two *different* files must not share a stream, or the
    identity is not reaching the key at all."""
    import pyscx

    p0, p1 = two_scx
    rows = [2, 7, 13]
    ds = pyscx.SparseCellSetDataset(
        [p0, p1],
        downsample_target_library_size=_TARGET,
        downsample_method="binomial",
        downsample_seed=9,
    )
    a = _libs(_gather(ds, [0] * len(rows), rows))
    b = _libs(_gather(ds, [1] * len(rows), rows))
    # The two files hold identical data, so identical libs would mean identical
    # streams.
    assert a != b, "both files drew identically"


def test_file_identity_is_alias_invariant(two_scx, tmp_dir):
    import pyscx

    p0, _ = two_scx
    direct = pyscx.downsample_file_identity(p0)
    round_about = str(tmp_dir / "sub" / ".." / pathlib.Path(p0).name)
    (tmp_dir / "sub").mkdir(exist_ok=True)
    assert pyscx.downsample_file_identity(round_about) == direct


# ---------------------------------------------------------------------------
# Distribution parity against numpy (moments, not values)
# ---------------------------------------------------------------------------


def test_binomial_moments_match_numpy():
    """Each element should behave like ``Binomial(n_i, target / library_size)``.

    Compared against numpy's own draws and against closed-form moments; bit
    parity is out of scope, so this is what "matches numpy" means here.
    """
    import pyscx

    counts = np.array([40.0, 60.0, 100.0, 200.0], dtype=np.float32)
    library = counts.sum()
    target = 100
    p = target / library
    n_draws = 600

    # One row per key -> `n_draws` independent draws of the same input.
    indptr = np.arange(n_draws + 1, dtype=np.int64) * len(counts)
    out = pyscx.downsample_counts_csr(
        indptr,
        np.tile(np.arange(len(counts), dtype=np.int32), n_draws),
        np.tile(counts, n_draws),
        np.arange(n_draws, dtype=np.uint64),
        np.full(n_draws, 0xFEED, dtype=np.uint64),
        target,
        "binomial",
        3,
    )
    # A hard squeeze can prune an element, so rebuild the dense per-draw matrix
    # rather than reshaping blindly.
    got = np.zeros((n_draws, len(counts)), dtype=np.float64)
    for r in range(n_draws):
        lo, hi = int(out["indptr"][r]), int(out["indptr"][r + 1])
        got[r, out["indices"][lo:hi]] = out["data"][lo:hi]

    rng = np.random.default_rng(0)
    ref = rng.binomial(counts.astype(np.int64), p, size=(n_draws, len(counts)))

    for i, n in enumerate(counts):
        theory_mean = n * p
        theory_sd = np.sqrt(n * p * (1 - p))
        se = theory_sd / np.sqrt(n_draws)
        assert abs(got[:, i].mean() - theory_mean) < 5 * se, (
            f"element {i}: mean {got[:, i].mean():.3f} vs theory {theory_mean:.3f}"
        )
        assert abs(got[:, i].mean() - ref[:, i].mean()) < 8 * se, (
            f"element {i} disagrees with numpy's own draws"
        )
        # Variance within a generous factor of the binomial variance — enough to
        # catch a sampler that is deterministic or wildly over-dispersed.
        assert 0.5 < got[:, i].var() / (n * p * (1 - p)) < 2.0, (
            f"element {i}: variance {got[:, i].var():.3f} vs theory "
            f"{n * p * (1 - p):.3f}"
        )


def test_multinomial_marginals_match_numpy():
    counts = np.array([40.0, 60.0, 100.0, 200.0], dtype=np.float32)
    probs = counts / counts.sum()
    target = 100
    n_draws = 600

    import pyscx

    indptr = np.arange(n_draws + 1, dtype=np.int64) * len(counts)
    out = pyscx.downsample_counts_csr(
        indptr,
        np.tile(np.arange(len(counts), dtype=np.int32), n_draws),
        np.tile(counts, n_draws),
        np.arange(n_draws, dtype=np.uint64),
        np.full(n_draws, 0xBEEF, dtype=np.uint64),
        target,
        "multinomial",
        3,
    )
    got = np.zeros((n_draws, len(counts)), dtype=np.float64)
    for r in range(n_draws):
        lo, hi = int(out["indptr"][r]), int(out["indptr"][r + 1])
        got[r, out["indices"][lo:hi]] = out["data"][lo:hi]

    # Exact total on every draw is the defining property.
    np.testing.assert_array_equal(got.sum(axis=1), np.full(n_draws, target))

    rng = np.random.default_rng(0)
    ref = rng.multinomial(target, probs, size=n_draws)
    for i, pr in enumerate(probs):
        theory_mean = target * pr
        se = np.sqrt(target * pr * (1 - pr)) / np.sqrt(n_draws)
        assert abs(got[:, i].mean() - theory_mean) < 5 * se, (
            f"element {i}: mean {got[:, i].mean():.3f} vs theory {theory_mean:.3f}"
        )
        assert abs(got[:, i].mean() - ref[:, i].mean()) < 8 * se, (
            f"element {i} disagrees with numpy's own draws"
        )


# ---------------------------------------------------------------------------
# Configuration errors — grouped at the public entry
# ---------------------------------------------------------------------------


def test_method_without_a_target_is_an_error(two_scx):
    """A typo that leaves a run un-augmented while looking configured is worth a
    hard error, not a silent no-op."""
    import pyscx

    p0, _ = two_scx
    with pytest.raises(ValueError, match="require downsample_target_library_size"):
        pyscx.SparseCellSetDataset([p0], downsample_method="binomial")


def test_seed_without_a_target_is_an_error(two_scx):
    import pyscx

    p0, _ = two_scx
    with pytest.raises(ValueError, match="require downsample_target_library_size"):
        pyscx.SparseCellSetDataset([p0], downsample_seed=7)


def test_zero_target_is_an_error(two_scx):
    import pyscx

    p0, _ = two_scx
    with pytest.raises(ValueError, match="must be > 0"):
        pyscx.SparseCellSetDataset([p0], downsample_target_library_size=0)


def test_unknown_method_names_the_accepted_set(two_scx):
    """`ValueError`, like every other argument check on this constructor.

    It used to be `RuntimeError` — not by design, but because this one error
    came from `DownsampleMethod::parse` and fell through `loader_err_to_py`'s
    default arm while the three checks beside it raised `ValueError` directly.
    Phase 9f moved the whole resolver into `downsample.rs` and maps its errors
    at the binding site, which made all four consistent. Pinned as `ValueError`
    rather than the bare `Exception` this asserted before, so the consistency
    cannot regress unnoticed.
    """
    import pyscx

    p0, _ = two_scx
    with pytest.raises(ValueError, match="hypergeometric"):
        pyscx.SparseCellSetDataset(
            [p0],
            downsample_target_library_size=_TARGET,
            downsample_method="hypergeometric",
        )


def test_standalone_primitive_rejects_an_unknown_method_with_valueerror(two_scx):
    """`ValueError`, matching the constructor and the zero-target check beside it.

    `downsample_counts_csr` is the second public downsample surface. Its
    `target_library_size == 0` check already raised `ValueError` while an unknown
    method fell through `loader_err_to_py`'s default arm to `RuntimeError` — two
    exception types for two argument checks in one function. Phase 9f made the
    `SparseCellSetDataset` constructor consistent; without this the two entry
    points disagreed with each other instead. Found by Cursor Agent in review.
    """
    import pyscx

    p0, _ = two_scx
    plain = _gather(pyscx.SparseCellSetDataset([p0]), [0, 0], [0, 1])
    with pytest.raises(ValueError, match="hypergeometric"):
        pyscx.downsample_counts_csr(
            plain["indptr"],
            plain["indices"],
            plain["data"],
            np.array([0, 1], dtype=np.uint64),
            np.array([1, 2], dtype=np.uint64),
            _TARGET,
            method="hypergeometric",
        )


def test_standalone_primitive_validates_its_array_lengths(two_scx):
    import pyscx

    p0, _ = two_scx
    plain = _gather(pyscx.SparseCellSetDataset([p0]), [0, 0], [0, 1])
    with pytest.raises(ValueError, match="rows len"):
        pyscx.downsample_counts_csr(
            plain["indptr"],
            plain["indices"],
            plain["data"],
            np.array([0], dtype=np.uint64),  # one row for a two-row batch
            np.array([], dtype=np.uint64),
            _TARGET,
        )
    with pytest.raises(ValueError, match="file_identities len"):
        pyscx.downsample_counts_csr(
            plain["indptr"],
            plain["indices"],
            plain["data"],
            np.array([0, 1], dtype=np.uint64),
            np.array([1, 2, 3], dtype=np.uint64),
            _TARGET,
        )


def test_standalone_primitive_defaults_to_multinomial(two_scx):
    """Same default as the reference config, so a migrating caller that omits
    `method` does not silently change distribution."""
    import pyscx

    p0, _ = two_scx
    plain = _gather(pyscx.SparseCellSetDataset([p0]), [0, 0], [0, 1])
    args = (
        plain["indptr"],
        plain["indices"],
        plain["data"],
        np.array([0, 1], dtype=np.uint64),
        np.array([], dtype=np.uint64),
    )
    implicit = pyscx.downsample_counts_csr(*args, _TARGET)
    explicit = pyscx.downsample_counts_csr(*args, _TARGET, "multinomial")
    np.testing.assert_array_equal(implicit["data"], explicit["data"])


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_downsample_prunes_sampled_zeros(two_scx):
    """A hard squeeze must shrink nnz and keep each row's indices ascending."""
    import pyscx

    p0, _ = two_scx
    rows = list(range(6))
    plain = _gather(pyscx.SparseCellSetDataset([p0]), [0] * 6, rows)
    squeezed = _gather(
        pyscx.SparseCellSetDataset(
            [p0], downsample_target_library_size=5, downsample_seed=2
        ),
        [0] * 6,
        rows,
    )
    assert len(squeezed["indices"]) < len(plain["indices"]), "nothing pruned"
    for j in range(6):
        idx, dat = _row(squeezed, j)
        assert len(idx) == len(dat)
        assert np.all(np.diff(idx) > 0), f"row {j} indices not ascending: {idx}"
        assert np.all(dat > 0), f"row {j} retained a zero: {dat}"


def test_clip_keeps_nnz_when_downsampling_is_off(tmp_dir):
    """The clip is unconditional; the zero-prune is not.

    A negative becomes an explicit zero rather than disappearing, so a caller
    that counts nnz sees the same structure with and without the clip — matching
    the reference, which prunes only inside its downsample branch.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    x = sp.csr_matrix(np.array([[5.0, -3.0, 7.0]], dtype=np.float32))
    adata = anndata.AnnData(
        X=x,
        obs=pd.DataFrame(index=["c0"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    p = str(tmp_dir / "neg.scx")
    pyscx.from_anndata(adata, p)

    b = _gather(pyscx.SparseCellSetDataset([p]), [0], [0])
    idx, dat = _row(b, 0)
    assert np.all(dat >= 0.0), f"negative leaked into the emitted CSR: {dat}"
    assert len(idx) == 3, f"clip changed nnz: {idx}"
    np.testing.assert_array_equal(dat, np.array([5.0, 0.0, 7.0], dtype=np.float32))


def test_malformed_indptr_raises_rather_than_panicking():
    """A non-monotonic `indptr` whose last entry looks right used to panic.

    `indptr.last() == nnz` is necessary but not sufficient: `[0, 3, 2]` over two
    non-zeros passes that check and then slices out of bounds inside the rayon
    map. A negative entry wraps through `as usize` and does the same. This is a
    public entry point taking arbitrary numpy arrays, so it must raise.
    """
    import pyscx

    indices = np.array([1, 2], dtype=np.int32)
    data = np.array([1.0, 2.0], dtype=np.float32)

    for label, indptr in [
        ("non-monotonic, last == nnz", [0, 3, 2]),
        ("negative entry", [0, -1, 2]),
        ("does not start at 0", [1, 2]),
    ]:
        arr = np.array(indptr, dtype=np.int64)
        with pytest.raises(ValueError, match="indptr"):
            pyscx.downsample_counts_csr(
                arr,
                indices,
                data,
                np.zeros(len(arr) - 1, dtype=np.uint64),
                np.array([], dtype=np.uint64),
                5,
            )
