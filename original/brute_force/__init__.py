"""Exact solvability datasets for small solitaire configurations."""

from .solvability import (
    ExactSolver,
    SolvabilityDataset,
    build_initial_state,
    expected_canonical_deal_count,
    iter_canonical_deals,
)

__all__ = [
    "ExactSolver",
    "SolvabilityDataset",
    "build_initial_state",
    "expected_canonical_deal_count",
    "iter_canonical_deals",
]
