from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from functools import cache
import itertools
import json
import math
from pathlib import Path


Card = tuple[int, int]
Position = tuple[int, int]


def eligible_blocker_positions() -> tuple[tuple[Position, frozenset[Position]], ...]:
    """Target positions with at least three cards below in the same pile."""
    result = []
    for column in range(3, 7):
        for row in range(3, column + 1):
            below = frozenset((column, lower_row) for lower_row in range(row))
            result.append(((column, row), below))
    return tuple(result)


def lower_same_suit_cards(card: Card) -> frozenset[Card]:
    suit, rank = card
    return frozenset((suit, lower_rank) for lower_rank in range(1, rank))


def tableau_support_cards(card: Card) -> frozenset[Card]:
    suit, rank = card
    return frozenset(
        (other_suit, rank + 1)
        for other_suit in range(4)
        if other_suit % 2 != suit % 2
    )


def blocker_events() -> tuple[tuple[Card, Position, frozenset[Position]], ...]:
    return tuple(
        (target, position, below)
        for target in (
            (suit, rank)
            for suit in range(4)
            for rank in range(2, 13)
        )
        for position, below in eligible_blocker_positions()
    )


def count_unrestricted_completion(
    position_counts: tuple[int, int, int],
    unrestricted_cards: int,
    forbid_first_cards: int,
    forbid_second_cards: int,
) -> int:
    """Fill B1-only, B2-only, and B1-and-B2 labeled positions."""
    first_only, second_only, both = position_counts
    total = 0
    for from_forbid_second in range(
        min(first_only, forbid_second_cards) + 1
    ):
        first_factor = (
            math.comb(first_only, from_forbid_second)
            * math.perm(forbid_second_cards, from_forbid_second)
        )
        for from_forbid_first in range(
            min(second_only, forbid_first_cards) + 1
        ):
            unrestricted_positions = (
                first_only
                + second_only
                + both
                - from_forbid_second
                - from_forbid_first
            )
            if unrestricted_positions > unrestricted_cards:
                continue
            total += (
                first_factor
                * math.comb(second_only, from_forbid_first)
                * math.perm(forbid_first_cards, from_forbid_first)
                * math.perm(unrestricted_cards, unrestricted_positions)
            )
    return total


def count_pair_constraint(
    left: tuple[Card, Position, frozenset[Position]],
    right: tuple[Card, Position, frozenset[Position]],
    forbid_left_lower: bool,
    forbid_right_lower: bool,
) -> tuple[int, int]:
    """Count assignments for one term of pairwise inclusion-exclusion."""
    left_target, left_position, left_below = left
    right_target, right_position, right_below = right
    if left_position == right_position or left_target == right_target:
        return 0, 0

    fixed = {
        left_position: left_target,
        right_position: right_target,
    }
    observed_positions = (
        set(left_below)
        | set(right_below)
        | {left_position, right_position}
    )
    fill_positions = observed_positions - set(fixed)
    lower_sets = (
        lower_same_suit_cards(left_target),
        lower_same_suit_cards(right_target),
    )

    def forbidden_mask(card: Card) -> int:
        result = 0
        if forbid_left_lower and card in lower_sets[0]:
            result |= 1
        if forbid_right_lower and card in lower_sets[1]:
            result |= 2
        return result

    for position, card in fixed.items():
        mask = forbidden_mask(card)
        if (mask & 1 and position in left_below) or (
            mask & 2 and position in right_below
        ):
            return 0, len(observed_positions)

    required: dict[Card, set[Position]] = {}
    for target, below in (
        (left_target, left_below),
        (right_target, right_below),
    ):
        for support in tableau_support_cards(target):
            if support in fixed.values():
                support_position = next(
                    position
                    for position, card in fixed.items()
                    if card == support
                )
                if support_position not in below:
                    return 0, len(observed_positions)
                continue
            if support not in required:
                required[support] = set(fill_positions)
            required[support] &= set(below)

    for card, allowed in required.items():
        mask = forbidden_mask(card)
        required[card] = {
            position
            for position in allowed
            if not (
                (mask & 1 and position in left_below)
                or (mask & 2 and position in right_below)
            )
        }
        if not required[card]:
            return 0, len(observed_positions)

    position_masks = {
        position: (
            (1 if position in left_below else 0)
            | (2 if position in right_below else 0)
        )
        for position in fill_positions
    }
    initial_counts = Counter(position_masks.values())
    assignment_shapes: Counter[tuple[int, int, int]] = Counter()
    required_items = sorted(required.items(), key=lambda item: len(item[1]))
    used_positions: set[Position] = set()

    def assign_required(index: int, counts: Counter[int]) -> None:
        if index == len(required_items):
            assignment_shapes[(counts[1], counts[2], counts[3])] += 1
            return
        _card, allowed = required_items[index]
        for position in allowed:
            if position in used_positions:
                continue
            mask = position_masks[position]
            used_positions.add(position)
            counts[mask] -= 1
            assign_required(index + 1, counts)
            counts[mask] += 1
            used_positions.remove(position)

    assign_required(0, initial_counts)
    if not assignment_shapes:
        return 0, len(observed_positions)

    excluded_cards = set(fixed.values()) | set(required)
    card_counts = Counter(
        forbidden_mask((suit, rank))
        for suit in range(4)
        for rank in range(1, 14)
        if (suit, rank) not in excluded_cards
    )
    completions = 0
    for shape, assignment_count in assignment_shapes.items():
        completions += assignment_count * count_unrestricted_completion(
            shape,
            unrestricted_cards=card_counts[0],
            forbid_first_cards=card_counts[1],
            forbid_second_cards=card_counts[2],
        )
    return completions, len(observed_positions)


def count_blocker_event_pair(
    left: tuple[Card, Position, frozenset[Position]],
    right: tuple[Card, Position, frozenset[Position]],
) -> tuple[int, int]:
    unrestricted, positions = count_pair_constraint(left, right, False, False)
    no_left, _ = count_pair_constraint(left, right, True, False)
    no_right, _ = count_pair_constraint(left, right, False, True)
    neither, _ = count_pair_constraint(left, right, True, True)
    count = unrestricted - no_left - no_right + neither
    if count < 0:
        raise AssertionError("negative blocker-pair count")
    return count, positions


@cache
def blocker_bonferroni_components() -> tuple[Fraction, Fraction]:
    """Return the exact first- and second-order blocked-card sums."""
    events = blocker_events()
    single_sum = Fraction(0)
    for target, _position, below in events:
        lower_count = len(lower_same_suit_cards(target))
        remaining_positions = len(below) - 2
        assignments = math.perm(len(below), 2) * (
            math.perm(49, remaining_positions)
            - math.perm(49 - lower_count, remaining_positions)
        )
        single_sum += Fraction(
            assignments,
            math.perm(52, len(below) + 1),
        )

    pair_counts: dict[int, int] = defaultdict(int)
    for left, right in itertools.combinations(events, 2):
        assignments, observed_positions = count_blocker_event_pair(left, right)
        pair_counts[observed_positions] += assignments
    pair_sum = sum(
        (
            Fraction(assignments, math.perm(52, observed_positions))
            for observed_positions, assignments in pair_counts.items()
        ),
        start=Fraction(0),
    )
    return single_sum, pair_sum


def blocker_loss_lower_bound() -> Fraction:
    """Second-order Bonferroni bound for every static blocked-card event."""
    single_sum, pair_sum = blocker_bonferroni_components()
    return single_sum - pair_sum


def direct_foundation_win_lower_bound() -> Fraction:
    """Count deals following one fixed all-foundation target sequence."""
    denominator = math.factorial(24)
    for pile_size in range(1, 8):
        denominator *= math.factorial(pile_size)
    return Fraction(1, denominator)


def hoeffding_lower_bound(wins: int, games: int, alpha: float) -> float:
    if not 0 <= wins <= games or games < 1:
        raise ValueError("require 0 <= wins <= games and games >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    penalty = math.sqrt(math.log(1.0 / alpha) / (2.0 * games))
    return max(0.0, wins / games - penalty)


def binary_relative_entropy(observed: float, candidate: float) -> float:
    if not 0 <= observed <= 1 or not 0 <= candidate <= 1:
        raise ValueError("probabilities must be in [0, 1]")
    if observed == candidate:
        return 0.0
    if candidate == 0:
        return math.inf if observed > 0 else -math.log1p(-candidate)
    if candidate == 1:
        return math.inf if observed < 1 else -math.log(candidate)
    first = 0.0 if observed == 0 else observed * math.log(observed / candidate)
    second = (
        0.0
        if observed == 1
        else (1.0 - observed)
        * math.log((1.0 - observed) / (1.0 - candidate))
    )
    return first + second


def chernoff_kl_lower_bound(successes: int, trials: int, alpha: float) -> float:
    """Invert exp(-n D(observed || p)) <= alpha for a one-sided bound."""
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    observed = successes / trials
    if successes == 0:
        return 0.0
    target = math.log(1.0 / alpha) / trials
    lower = 0.0
    upper = observed
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if binary_relative_entropy(observed, midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bound standard-deck solvability")
    parser.add_argument("--wins", type=int)
    parser.add_argument("--games", type=int)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--exact-node-limit", type=int)
    parser.add_argument("--model-max-steps", type=int)
    parser.add_argument("--model-weights", type=float, nargs="+")
    parser.add_argument("--model-source")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--model-wins", type=int)
    parser.add_argument("--exact-fallbacks", type=int)
    parser.add_argument("--budget-exhaustions", type=int)
    parser.add_argument("--expanded-positions", type=int)
    parser.add_argument("--elapsed-seconds", type=float)
    parser.add_argument("--blocked-deals", type=int)
    parser.add_argument("--blocked-games", type=int)
    parser.add_argument("--blocked-seed", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_weights = args.model_weights
    if model_weights is None and args.model_source is not None:
        model_result = json.loads(
            Path(args.model_source).read_text(encoding="ascii")
        )
        feature_names = model_result.get("feature_names")
        selected_parameters = model_result.get("selected_parameters")
        if not isinstance(feature_names, list) or not isinstance(
            selected_parameters, dict
        ):
            raise ValueError("model source does not contain selected parameters")
        model_weights = [
            float(selected_parameters[name]) for name in feature_names
        ]
    blocker_single_sum, blocker_pair_sum = blocker_bonferroni_components()
    loss_lower = blocker_single_sum - blocker_pair_sum
    absolute_win_lower = direct_foundation_win_lower_bound()
    result: dict[str, object] = {
        "schema_version": 1,
        "quantity": "solvability probability of a uniformly random deal",
        "config": {
            "n": 13,
            "k": 2,
            "t": 7,
            "draw_count": 1,
            "max_recycles": 3,
        },
        "absolute_bounds": {
            "lower": float(absolute_win_lower),
            "lower_fraction": (
                f"{absolute_win_lower.numerator}/{absolute_win_lower.denominator}"
            ),
            "lower_method": "counted direct-to-foundation deal family",
            "upper": float(1 - loss_lower),
            "loss_lower": float(loss_lower),
            "blocker_event_count": len(blocker_events()),
            "blocker_single_sum": float(blocker_single_sum),
            "blocker_pair_sum": float(blocker_pair_sum),
            "loss_lower_fraction": (
                f"{loss_lower.numerator}/{loss_lower.denominator}"
            ),
            "upper_method": (
                "second-order Bonferroni count of permanently blocked "
                "rank-2-through-Queen tableau cards"
            ),
        },
    }
    has_win_sample = args.wins is not None or args.games is not None
    has_blocked_sample = (
        args.blocked_deals is not None or args.blocked_games is not None
    )
    tail_count = int(has_win_sample) + int(has_blocked_sample)
    tail_alpha = args.alpha / tail_count if tail_count else args.alpha

    if has_win_sample:
        if args.wins is None or args.games is None:
            raise ValueError("--wins and --games must be supplied together")
        statistical_lower = chernoff_kl_lower_bound(
            args.wins,
            args.games,
            tail_alpha,
        )
        constructive_sample = {
            "wins": args.wins,
            "games": args.games,
            "observed_rate": args.wins / args.games,
            "seed": args.seed,
            "exact_node_limit_per_model_failure": args.exact_node_limit,
            "model_max_steps": args.model_max_steps,
            "model_weights": model_weights,
            "model_source": args.model_source,
            "threads": args.threads,
            "model_wins": args.model_wins,
            "exact_fallbacks": args.exact_fallbacks,
            "exact_budget_exhaustions": args.budget_exhaustions,
            "expanded_positions": args.expanded_positions,
            "elapsed_seconds": args.elapsed_seconds,
            "joint_confidence": 1.0 - args.alpha,
            "tail_alpha": tail_alpha,
            "solvability_lower_bound": statistical_lower,
            "method": (
                "inverted one-sided Bernoulli KL-Chernoff bound for a fixed "
                "constructive solver on independent uniformly shuffled deals"
            ),
        }
        if (
            args.model_wins is not None
            and args.exact_fallbacks is not None
            and args.budget_exhaustions is not None
        ):
            exact_wins = args.wins - args.model_wins
            exact_proven_losses = (
                args.exact_fallbacks - exact_wins - args.budget_exhaustions
            )
            if exact_wins < 0 or exact_proven_losses < 0:
                raise ValueError("inconsistent constructive solver counts")
            constructive_sample["exact_wins"] = exact_wins
            constructive_sample["exact_proven_losses"] = exact_proven_losses
        result["constructive_sample"] = constructive_sample
    if has_blocked_sample:
        if args.blocked_deals is None or args.blocked_games is None:
            raise ValueError(
                "--blocked-deals and --blocked-games must be supplied together"
            )
        blocked_lower = chernoff_kl_lower_bound(
            args.blocked_deals,
            args.blocked_games,
            tail_alpha,
        )
        result["static_blocker_sample"] = {
            "blocked_deals": args.blocked_deals,
            "games": args.blocked_games,
            "observed_rate": args.blocked_deals / args.blocked_games,
            "seed": args.blocked_seed,
            "joint_confidence": 1.0 - args.alpha,
            "tail_alpha": tail_alpha,
            "unsolvability_lower_bound": blocked_lower,
            "solvability_upper_bound": 1.0 - blocked_lower,
            "method": (
                "inverted one-sided Bernoulli KL-Chernoff bound for the sound "
                "static blocked-card predicate on independent uniformly "
                "shuffled deals"
            ),
        }

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(encoded, encoding="ascii")
        temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
