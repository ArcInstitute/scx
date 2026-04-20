"""
Canonical predicate set for the ``read_selective`` filtered-query scenario.

Each runner's ``read_filtered_query(path, predicate)`` consumes one of these
``Predicate`` instances and translates it into its native pushdown mechanism:

  - SCX       → catalog-level CategoryBitset pushdown
  - SLAF      → SQL ``WHERE`` clause against DuckDB
  - TileDB-SOMA → ``AxisQuery.value_filter``
  - Zarr      → densify + numpy boolean mask (no pushdown)
  - BPCells   → R-side boolean mask on obs metadata
  - Parquet   → row-group predicate via PyArrow filters

The predicate set is intentionally small and dataset-portable so the same
queries produce comparable timings across every dataset in ``config.DATASETS``.
Each entry carries a fallback value for datasets whose obs metadata uses a
different column name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Predicate types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EqPredicate:
    """Equality filter: ``obs[column] == value``."""
    column: str
    value: Any
    name: str

    def describe(self) -> str:
        return f"{self.column} == {self.value!r}"


@dataclass(frozen=True)
class GtPredicate:
    """Strict-greater-than filter: ``obs[column] > threshold``."""
    column: str
    threshold: float
    name: str

    def describe(self) -> str:
        return f"{self.column} > {self.threshold}"


@dataclass(frozen=True)
class RandomSamplePredicate:
    """Random row sample at a given fraction. Deterministic via ``seed``."""
    fraction: float
    seed: int
    name: str

    def describe(self) -> str:
        return f"random {self.fraction * 100:.1f}% sample (seed={self.seed})"


Predicate = EqPredicate | GtPredicate | RandomSamplePredicate


# ---------------------------------------------------------------------------
# Canonical set
# ---------------------------------------------------------------------------

# ``name`` is what reports group by. ``column`` names are the scanpy / SOMA
# conventions; datasets that use a different column should be excluded at the
# benchmark-runtime level (see ``read_selective._predicates_for_dataset``).
CANONICAL_PREDICATES: list[Predicate] = [
    EqPredicate(column="cell_type", value="T cell", name="cell_type_eq_t_cell"),
    GtPredicate(column="n_counts", threshold=1000, name="n_counts_gt_1000"),
    RandomSamplePredicate(fraction=0.01, seed=42, name="random_1pct"),
]


def default_predicates() -> list[Predicate]:
    """Return the shared canonical predicate list."""
    return list(CANONICAL_PREDICATES)
