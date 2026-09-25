from __future__ import annotations

import argparse
from collections import Counter
from itertools import permutations
from typing import Iterable


def passes_to_emit(draw_order: Iterable[int]) -> int:
    """Return passes needed to emit cards in the order 0, 1, ..., m - 1.

    A pass draws left-to-right onto a waste stack. Whenever the next required
    card is on top, it is removed and newly exposed required cards are removed
    too. Recycling preserves the draw order of the cards still in the waste.
    """
    remaining = list(draw_order)
    card_count = len(remaining)
    if sorted(remaining) != list(range(card_count)):
        raise ValueError("draw_order must be a permutation of range(m)")

    target = 0
    passes = 0
    while remaining:
        passes += 1
        waste: list[int] = []
        for card in remaining:
            waste.append(card)
            while waste and waste[-1] == target:
                waste.pop()
                target += 1
        if len(waste) == len(remaining):
            raise AssertionError("a complete pass must emit the next target")
        remaining = waste

    if target != card_count:
        raise AssertionError("not every card was emitted")
    return passes


def stock_order_pass_histogram(card_count: int) -> dict[int, int]:
    histogram = Counter(
        passes_to_emit(draw_order)
        for draw_order in permutations(range(card_count))
    )
    return dict(sorted(histogram.items()))


def worst_case_passes(card_count: int) -> int:
    """Compute P(m) exactly by checking all m! relative draw orders."""
    if card_count < 0:
        raise ValueError("card_count must be >= 0")
    if card_count == 0:
        return 0
    return max(stock_order_pass_histogram(card_count))


def universal_pass_bound(card_count: int) -> int:
    """A simple all-size bound: every pass emits at least one target card."""
    if card_count < 0:
        raise ValueError("card_count must be >= 0")
    return card_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove a stock-order scheduling bound by finite enumeration"
    )
    parser.add_argument("--cards", type=int, default=6)
    parser.add_argument("--passes", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card_count = args.cards
    available_passes = args.passes
    if card_count < 0 or available_passes < 0:
        raise ValueError("cards and passes must be >= 0")
    if card_count > 10:
        print(f"universal_pass_bound {universal_pass_bound(card_count)}")
        if available_passes >= universal_pass_bound(card_count):
            print("stock_order_bound proved")
        else:
            print("stock_order_bound inconclusive_without_a_sharper_bound")
        return

    histogram = stock_order_pass_histogram(card_count)
    worst = max(histogram, default=0)
    print(f"permutations_checked {sum(histogram.values())}")
    print(f"pass_histogram {histogram}")
    print(f"maximum_passes {worst}")
    print(f"available_passes {available_passes}")
    if worst <= available_passes:
        print("stock_order_bound proved")
    else:
        print("stock_order_bound disproved_for_arbitrary_target_orders")


if __name__ == "__main__":
    main()
