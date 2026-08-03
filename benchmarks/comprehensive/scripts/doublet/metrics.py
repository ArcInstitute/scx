"""Agreement and accuracy metrics over canonical doublet obs columns.

Pure functions over arrays, deliberately free of pyscx, SLURM and file I/O so
they can be tested against hand-computed cases. Two tiers, kept apart because
they answer different questions:

* **Plumbing** — did the import land the tool's own numbers on the right cells?
  Deterministic, so it gates immediately.
* **Science** — do the tools agree with each other, and with labelled truth?
  Distributional, so its floors have to be calibrated from a real capture.

Every pairwise metric restricts to rows where **both** tools voted. This is the
Phase-7 null rule one level up: a tool that never saw a cell has no opinion
about it, and counting a non-vote as agreement would inflate every number here
in proportion to how little the tools overlapped — worst exactly when the
overlap is small enough for the comparison to be meaningless.

Neither AUROC nor Spearman pulls in scipy/sklearn: both are rank statistics
with exact closed forms, and the runners already live in envs whose package
sets we would rather not grow.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_calls",
    "pairwise_call_agreement",
    "score_rank_correlation",
    "called_rate_by_batch",
    "truth_metrics",
    "roundtrip_fidelity",
    "auroc",
]


def normalize_calls(values) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(voted, called)`` boolean arrays for a `<K>_predicted` column.

    Mirrors the contract `pyscx.doublet_consensus` reads: a null is *not a
    vote*. Accepts every dtype the SCX round trip produces — object with
    `None`, pandas nullable `boolean`, and plain `bool`.
    """
    import pandas as pd

    series = pd.Series(values)
    voted = ~pd.isna(series).to_numpy()
    called = np.zeros(len(series), dtype=bool)
    if voted.any():
        present = series.to_numpy(dtype=object)[voted]
        called[voted] = np.array([bool(v) for v in present], dtype=bool)
    return voted, called


def _both_voted(a, b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voted_a, called_a = normalize_calls(a)
    voted_b, called_b = normalize_calls(b)
    both = voted_a & voted_b
    return both, called_a, called_b


def pairwise_call_agreement(a, b) -> dict:
    """Raw agreement and Cohen's kappa between two tools' calls.

    Kappa as well as raw agreement because raw agreement is nearly useless on
    this problem: doublet rates are a few percent, so two tools that both call
    almost nothing agree ~95% of the time by construction. Kappa corrects for
    that chance agreement and is the number worth reading.
    """
    both, called_a, called_b = _both_voted(a, b)
    n = int(both.sum())
    out: dict = {
        "n_both_voted": n,
        "n_compared_of": int(len(both)),
        "agreement": None,
        "kappa": None,
        "n_agree": 0,
        "n_disagree": 0,
    }
    if n == 0:
        out["reason"] = "no row was voted on by both tools"
        return out

    ca, cb = called_a[both], called_b[both]
    agree = ca == cb
    out["n_agree"] = int(agree.sum())
    out["n_disagree"] = int(n - agree.sum())
    po = float(agree.mean())
    out["agreement"] = po

    # Chance agreement under independence of the two marginals.
    pa1, pb1 = float(ca.mean()), float(cb.mean())
    pe = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    if pe >= 1.0:
        # Both tools put every compared cell in the same single class, so
        # chance agreement is already 1 and kappa is 0/0. Report the reason
        # rather than a fabricated 0.0 or 1.0 — either would be read as a
        # finding about the tools instead of a degenerate comparison.
        out["reason"] = (
            "kappa undefined: both tools assigned every compared cell to a "
            "single class, so chance agreement is 1.0"
        )
        return out
    out["kappa"] = float((po - pe) / (1.0 - pe))
    return out


def score_rank_correlation(a, b) -> dict:
    """Spearman correlation between two tools' scores, over rows both scored.

    Spearman rather than Pearson because the tools' scores are on entirely
    different scales (scDblFinder's is roughly a probability, Scrublet's a
    simulated-neighbour ratio) and only their ordering is comparable.
    """
    import pandas as pd

    sa = pd.to_numeric(pd.Series(a), errors="coerce")
    sb = pd.to_numeric(pd.Series(b), errors="coerce")
    both = (~sa.isna() & ~sb.isna()).to_numpy()
    n = int(both.sum())
    out: dict = {"n_both_scored": n, "spearman": None}
    if n < 3:
        out["reason"] = f"need at least 3 jointly scored cells; got {n}"
        return out

    # Average ranks, then Pearson on the ranks — the definition of Spearman,
    # and tie-correct without scipy.
    ra = sa[both].rank(method="average").to_numpy(dtype="float64")
    rb = sb[both].rank(method="average").to_numpy(dtype="float64")
    if ra.std() == 0.0 or rb.std() == 0.0:
        out["reason"] = "a tool's score is constant over the compared cells"
        return out
    out["spearman"] = float(np.corrcoef(ra, rb)[0, 1])
    return out


def called_rate_by_batch(pred, batch) -> dict:
    """Per-batch called rate, over rows the tool actually voted on.

    Batch-level calibration is reported even when the global rate looks
    sensible: a tool that calls 6% overall but 30% in one library is
    miscalibrated in a way the pooled number hides
    (`DOUBLET-DETECTION.md` § Accuracy and diagnostic interpretation rules).
    """
    import pandas as pd

    voted, called = normalize_calls(pred)
    labels = pd.Series(batch).astype(str).to_numpy()
    out: dict[str, dict] = {}
    for value in sorted(set(labels[voted])) if voted.any() else []:
        mask = voted & (labels == value)
        n = int(mask.sum())
        out[value] = {
            "n_voted": n,
            "n_called": int(called[mask].sum()),
            "called_rate": float(called[mask].mean()) if n else None,
        }
    return out


def auroc(scores, positives) -> float | None:
    """Rank-based AUROC (Mann-Whitney U), tie-correct, no sklearn.

    Returns None when either class is empty, where AUROC is undefined.
    """
    import pandas as pd

    s = pd.to_numeric(pd.Series(scores), errors="coerce")
    pos = np.asarray(positives, dtype=bool)
    ok = (~s.isna()).to_numpy()
    s, pos = s[ok], pos[ok]
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = s.rank(method="average").to_numpy(dtype="float64")
    rank_sum = float(ranks[pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def truth_metrics(scores, pred, truth_label, *, kinds=None,
                  positive_value: str = "doublet") -> dict:
    """Score one tool against labelled truth, over rows it voted on.

    `kinds` (heterotypic / homotypic / real) stratifies the recall, because a
    pooled recall on injected data is dominated by whichever kind was sampled
    more, and homotypic doublets are the ones every method struggles with. An
    aggregate that hides that is the failure mode
    `DOUBLET-DETECTION.md` warns about explicitly.
    """
    import pandas as pd

    voted, called = normalize_calls(pred)
    labels = pd.Series(truth_label).astype(str).to_numpy()
    positive = labels == positive_value

    out: dict = {
        "positive_value": positive_value,
        "n_voted": int(voted.sum()),
        "n_positive": int((positive & voted).sum()),
        "n_negative": int((~positive & voted).sum()),
        "auroc": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    if not voted.any():
        out["reason"] = "the tool voted on no rows"
        return out

    s = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy()
    out["auroc"] = auroc(s[voted], positive[voted])

    tp = int((called & positive & voted).sum())
    fp = int((called & ~positive & voted).sum())
    fn = int((~called & positive & voted).sum())
    out.update(tp=tp, fp=fp, fn=fn)
    if tp + fp > 0:
        out["precision"] = tp / (tp + fp)
    if tp + fn > 0:
        out["recall"] = tp / (tp + fn)
    if out["precision"] and out["recall"]:
        p, r = out["precision"], out["recall"]
        out["f1"] = 2 * p * r / (p + r)

    if kinds is not None:
        kind_arr = pd.Series(kinds).astype(str).to_numpy()
        by_kind: dict[str, dict] = {}
        for kind in sorted(set(kind_arr[voted & positive])):
            mask = voted & positive & (kind_arr == kind)
            n = int(mask.sum())
            by_kind[kind] = {
                "n": n,
                "recall": float(called[mask].mean()) if n else None,
            }
        out["recall_by_kind"] = by_kind
    return out


def roundtrip_fidelity(file_scores, file_calls, tool_scores, tool_calls) -> dict:
    """Compare what the tool emitted with what came back off the SCX file.

    The mechanized form of the Phase-4 exit check, and the strongest thing this
    harness asserts cheaply: the arrays must already be aligned **by key** by
    the caller, so a positional import — which passes every row-count check —
    fails here and only here.
    """
    import pandas as pd

    fs = pd.to_numeric(pd.Series(file_scores), errors="coerce").to_numpy()
    ts = pd.to_numeric(pd.Series(tool_scores), errors="coerce").to_numpy()
    if len(fs) != len(ts):
        raise ValueError(
            f"score arrays must be key-aligned; got {len(fs)} vs {len(ts)}"
        )

    finite = ~(np.isnan(fs) | np.isnan(ts))
    max_delta = float(np.abs(fs[finite] - ts[finite]).max()) if finite.any() else 0.0

    _, file_called = normalize_calls(file_calls)
    _, tool_called = normalize_calls(tool_calls)
    disagreements = int((file_called != tool_called).sum())

    return {
        "n_compared": int(finite.sum()),
        "n_rows": int(len(fs)),
        "max_score_delta": max_delta,
        "call_disagreements": disagreements,
        # A row the tool scored but the file returned as null means the import
        # dropped it; counted separately from a value mismatch because the
        # causes are different (a failed join vs a bad value).
        "n_unmatched": int((~finite & ~np.isnan(ts)).sum()),
    }
