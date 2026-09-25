"""Standalone exhaustive search for one-color solitaire."""

from .solvability import (
    ExactReserveSolver,
    SolvabilityDataset,
    build_reserve_state,
    expected_stock_collapsed_deal_count,
    iter_stock_collapsed_deals,
)

__all__ = [
    "ExactReserveSolver",
    "SolvabilityDataset",
    "build_reserve_state",
    "expected_stock_collapsed_deal_count",
    "iter_stock_collapsed_deals",
]
