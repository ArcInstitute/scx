"""
§2.5 — rotation-invariant subspace metric for randomized PCA.

The pre-§2.5 gate floored randomized-PCA `cosine_sim_mean` at ~0.35 on
pbmc3k because the per-PC cosine drops below 1.0 just from the basis
rotation/permutation that the randomized-SVD path is free to return.
That floor is a sanity check, not a correctness gate — a real
regression that left the subspace mostly intact but corrupted one PC
would still pass.

The new ``_subspace_principal_cosines`` metric IS the strict gate. It
is invariant to basis rotation, column permutation, and column-sign
flip (the three free parameters of "same subspace, different basis"),
but drops below 1.0 the moment a component direction is genuinely
missing.

These tests cover:

  1. Identity inputs ⇒ all cosines are 1.0.
  2. Sign-flipped column ⇒ all cosines still 1.0 (per-PC cosine
     handles this; included for parity).
  3. Column-permuted columns ⇒ all cosines still 1.0 (per-PC cosine
     would FAIL here — this is the rotation invariance).
  4. Random-orthogonal-rotation of the basis ⇒ all cosines still 1.0
     (the marquee §2.5 case: randomized PCA returns this kind of
     rotated subspace by construction).
  5. One column substituted by an out-of-plane direction ⇒
     ``subspace_cos_min`` drops below 1.0, proving the gate fires on
     a real subspace regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.comprehensive.benchmarks.accel_pca import (
    _sign_agnostic_cosine_per_pc,
    _subspace_principal_cosines,
)


def _random_orthonormal_basis(n: int, k: int, seed: int = 0) -> np.ndarray:
    """Return an n×k matrix with orthonormal columns (a basis of a random
    k-subspace of R^n)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, k)).astype(np.float64)
    q, _ = np.linalg.qr(a)
    return q


def test_identical_subspaces_give_all_ones() -> None:
    a = _random_orthonormal_basis(200, 8, seed=42)
    cos = _subspace_principal_cosines(a, a)
    assert np.allclose(cos, 1.0, atol=1e-10)


def test_column_sign_flip_invariant() -> None:
    a = _random_orthonormal_basis(200, 8, seed=43)
    b = a.copy()
    b[:, 3] *= -1.0  # flip one PC's sign
    cos = _subspace_principal_cosines(a, b)
    assert np.allclose(cos, 1.0, atol=1e-10)


def test_column_permutation_invariant() -> None:
    """The per-PC sign-agnostic cosine FAILS on this case; the subspace
    metric must succeed.
    """
    a = _random_orthonormal_basis(200, 8, seed=44)
    perm = np.array([3, 0, 5, 7, 1, 4, 2, 6])
    b = a[:, perm]

    # Sanity: per-PC cosine is destroyed (no longer all 1.0 — except by
    # luck on PCs the permutation happens to leave in place).
    per_pc = _sign_agnostic_cosine_per_pc(a, b)
    assert (per_pc < 0.99).any(), "expected per-PC cosine to drop on permutation"

    # The subspace metric is invariant.
    cos = _subspace_principal_cosines(a, b)
    assert np.allclose(cos, 1.0, atol=1e-10)


def test_random_orthogonal_basis_rotation_invariant() -> None:
    """Marquee §2.5 case: randomized PCA returns a rotated basis of the
    same subspace. Per-PC cosine drops; subspace metric does not.
    """
    rng = np.random.default_rng(45)
    a = _random_orthonormal_basis(200, 8, seed=45)
    # Rotate within the column-span by a random orthogonal k×k matrix.
    rot, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    b = a @ rot

    per_pc = _sign_agnostic_cosine_per_pc(a, b)
    # Generic rotation should hit at least one PC well below 1.0.
    assert per_pc.min() < 0.95, (
        "expected per-PC cosine to drop under random rotation; "
        f"got min={per_pc.min():.4f}"
    )

    cos = _subspace_principal_cosines(a, b)
    assert np.allclose(cos, 1.0, atol=1e-10)


def test_subspace_regression_drops_min_cosine() -> None:
    """A genuinely corrupted component drops subspace_cos_min below 1.0.

    Construct ``b`` by replacing one of ``a``'s columns with a direction
    orthogonal to all of ``a``'s columns. The two subspaces now share
    only k-1 dimensions, so the smallest principal-angle cosine is 0.0.
    """
    n, k = 200, 8
    # Build a 2k-dim basis and split it: a = first k cols, the (k+1)th
    # col is orthogonal to all of a.
    full = _random_orthonormal_basis(n, 2 * k, seed=46)
    a = full[:, :k]
    b = a.copy()
    b[:, 4] = full[:, k]  # replace one PC with an out-of-plane direction

    cos = _subspace_principal_cosines(a, b)
    # k-1 angles agree (cosine=1) and one is fully orthogonal (cosine=0).
    assert cos[-1] < 1e-6, f"expected subspace_cos_min ~ 0, got {cos[-1]:.6f}"
    assert cos[0] > 1.0 - 1e-6


def test_clipped_into_unit_interval() -> None:
    """SVD on near-orthonormal product can return 1+ε; the helper must
    keep the cosine interpretation bounded."""
    a = _random_orthonormal_basis(200, 8, seed=47)
    cos = _subspace_principal_cosines(a, a)
    assert (cos <= 1.0).all()
    assert (cos >= 0.0).all()
