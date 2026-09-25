from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
from typing import Iterator, Optional, Sequence


Deal = tuple[int, ...]
FACE_UP = 0x80
CARD_MASK = 0x7F
TABLEAU_PATTERN_RANK_ENUMERATION = (
    "one-color-tableau-pattern-partial-rank-v1"
)


def default_tableau_count(n: int, k: int = 2) -> int:
    """Triangular tableau giving the most even tableau/stock split.

    Exact ties go to the larger tableau.
    """
    if n < 1 or k < 1:
        raise ValueError("n and k must be positive")
    deck_size = n * k
    candidates = (
        columns
        for columns in range(1, deck_size + 1)
        if columns * (columns + 1) // 2 <= deck_size
    )
    return min(
        candidates,
        key=lambda columns: (
            abs(columns * (columns + 1) - deck_size),
            -columns,
        ),
    )


def tableau_card_count(n: int, k: int = 2, t: Optional[int] = None) -> int:
    columns = default_tableau_count(n, k) if t is None else t
    cards = columns * (columns + 1) // 2
    if columns < 1 or cards > n * k:
        raise ValueError("tableau does not fit in the deck")
    return cards


def suit_symmetry_maps(k: int) -> tuple[tuple[int, ...], ...]:
    """Every rule-preserving relabeling of the one-color suits."""
    if k < 1:
        raise ValueError("k must be positive")
    return tuple(permutations(range(k)))


def card_symmetry_maps(n: int, k: int) -> tuple[tuple[int, ...], ...]:
    maps = []
    for suit_map in suit_symmetry_maps(k):
        maps.append(
            tuple(
                suit_map[suit] * n + rank_index
                for suit in range(k)
                for rank_index in range(n)
            )
        )
    return tuple(maps)


def symmetry_orbit_size(k: int) -> int:
    return math.factorial(k)


def expected_canonical_deal_count(n: int, k: int) -> int:
    return math.factorial(n * k) // symmetry_orbit_size(k)


def expected_stock_collapsed_deal_count(
    n: int,
    k: int,
    tableau_cards: int,
) -> int:
    stock_cards = n * k - tableau_cards
    if stock_cards < 0:
        raise ValueError("tableau exceeds the deck")
    return (
        math.factorial(n * k)
        // math.factorial(stock_cards)
        // symmetry_orbit_size(k)
    )


def is_canonical_deal(
    deal: Deal,
    transforms: Sequence[Sequence[int]],
) -> bool:
    return all(
        tuple(transform[card] for card in deal) >= deal
        for transform in transforms
    )


def iter_canonical_deals(n: int, k: int = 2) -> Iterator[Deal]:
    """Slow full-order enumerator used only for small validation cases."""
    transforms = card_symmetry_maps(n, k)
    for deal in permutations(range(n * k)):
        if is_canonical_deal(deal, transforms):
            yield deal


def iter_canonical_suit_patterns(
    n: int,
    k: int = 2,
) -> Iterator[tuple[int, ...]]:
    """Restricted-growth suit strings for complete ordered deals."""
    if n < 1 or k < 1:
        raise ValueError("n and k must be positive")
    counts = [0] * k
    pattern: list[int] = []

    def extend(max_seen: int) -> Iterator[tuple[int, ...]]:
        if len(pattern) == n * k:
            yield tuple(pattern)
            return
        for suit in range(min(k - 1, max_seen + 1) + 1):
            if counts[suit] == n:
                continue
            counts[suit] += 1
            pattern.append(suit)
            yield from extend(max(max_seen, suit))
            pattern.pop()
            counts[suit] -= 1

    yield from extend(-1)


def expected_canonical_suit_pattern_count(n: int, k: int = 2) -> int:
    return (
        math.factorial(n * k)
        // math.factorial(n) ** k
        // math.factorial(k)
    )


def _rank_assignments(
    n: int,
    counts: Sequence[int],
) -> Iterator[tuple[tuple[int, ...], ...]]:
    selected: list[tuple[int, ...]] = []

    def extend(suit: int) -> Iterator[tuple[tuple[int, ...], ...]]:
        if suit == len(counts):
            yield tuple(selected)
            return
        for order in permutations(range(n), counts[suit]):
            selected.append(order)
            yield from extend(suit + 1)
            selected.pop()

    yield from extend(0)


def _deal_from_pattern(
    pattern: Sequence[int],
    ranks_by_suit: Sequence[Sequence[int]],
    n: int,
) -> Deal:
    occurrences = [0] * len(ranks_by_suit)
    deal = []
    for suit in pattern:
        rank_index = ranks_by_suit[suit][occurrences[suit]]
        occurrences[suit] += 1
        deal.append(suit * n + rank_index)
    return tuple(deal)


def iter_suit_pattern_rank_deals(
    n: int,
    k: int = 2,
) -> Iterator[Deal]:
    """Direct full-order canonical enumerator for small validation cases."""
    counts = [n] * k
    for pattern in iter_canonical_suit_patterns(n, k):
        for ranks_by_suit in _rank_assignments(n, counts):
            yield _deal_from_pattern(pattern, ranks_by_suit, n)


def iter_canonical_tableau_patterns(
    n: int,
    k: int,
    tableau_cards: int,
) -> Iterator[tuple[int, ...]]:
    """Suit-canonical ordered tableau prefixes."""
    if n < 1 or k < 1 or not 1 <= tableau_cards <= n * k:
        raise ValueError("invalid n, k, or tableau size")
    counts = [0] * k
    pattern: list[int] = []

    def extend(max_seen: int) -> Iterator[tuple[int, ...]]:
        if len(pattern) == tableau_cards:
            yield tuple(pattern)
            return
        for suit in range(min(k - 1, max_seen + 1) + 1):
            if counts[suit] == n:
                continue
            counts[suit] += 1
            pattern.append(suit)
            yield from extend(max(max_seen, suit))
            pattern.pop()
            counts[suit] -= 1

    yield from extend(-1)


def iter_stock_collapsed_deals(
    n: int,
    k: int = 2,
    t: Optional[int] = None,
) -> Iterator[Deal]:
    """Canonical tableau orders followed by the remaining cards sorted."""
    tableau_cards = tableau_card_count(n, k, t)
    for pattern in iter_canonical_tableau_patterns(
        n,
        k,
        tableau_cards,
    ):
        counts = [pattern.count(suit) for suit in range(k)]
        for ranks_by_suit in _rank_assignments(n, counts):
            tableau = _deal_from_pattern(pattern, ranks_by_suit, n)
            used = set(tableau)
            reserve = tuple(
                card for card in range(n * k) if card not in used
            )
            yield tableau + reserve


def card_id(card: int) -> int:
    return card & CARD_MASK


def face_up(card: int) -> bool:
    return bool(card & FACE_UP)


def make_face_up(card: int) -> int:
    return card | FACE_UP


def suit(card: int, n: int) -> int:
    return card_id(card) // n


def rank(card: int, n: int) -> int:
    return card_id(card) % n + 1


def can_build(card: int, destination: tuple[int, ...], n: int) -> bool:
    if not face_up(card):
        return False
    if not destination:
        return rank(card, n) == n
    return (
        face_up(destination[-1])
        and rank(card, n) + 1 == rank(destination[-1], n)
    )


def packed_stack(pile: tuple[int, ...], start: int, n: int) -> bool:
    stack = pile[start:]
    return bool(stack) and all(face_up(card) for card in stack) and all(
        rank(lower, n) == rank(upper, n) + 1
        for lower, upper in zip(stack, stack[1:])
    )


def _reveal_top(pile: tuple[int, ...]) -> tuple[int, ...]:
    if not pile:
        return pile
    return pile[:-1] + (make_face_up(pile[-1]),)


def _canonical_tableau(
    tableau: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(pile) for pile in tableau))


@dataclass(frozen=True)
class ReserveState:
    tableau: tuple[tuple[int, ...], ...]
    reserve: tuple[int, ...]
    foundations: tuple[int, ...]


def build_reserve_state(
    deal: Sequence[int],
    n: int,
    k: int = 2,
    t: Optional[int] = None,
) -> ReserveState:
    if len(deal) != n * k or set(deal) != set(range(n * k)):
        raise ValueError("deal must contain every card exactly once")
    columns = default_tableau_count(n, k) if t is None else t
    cursor = 0
    tableau = []
    for column in range(columns):
        pile = []
        for row in range(column + 1):
            card = deal[cursor]
            cursor += 1
            pile.append(make_face_up(card) if row == column else card)
        tableau.append(tuple(pile))
    return ReserveState(
        tableau=_canonical_tableau(tableau),
        reserve=tuple(sorted(deal[cursor:])),
        foundations=(0,) * k,
    )


def reserve_successors(
    state: ReserveState,
    n: int,
) -> Iterator[ReserveState]:
    tableau = state.tableau

    for reserve_index, raw_card in enumerate(state.reserve):
        card = make_face_up(raw_card)
        card_suit = suit(card, n)
        remaining = (
            state.reserve[:reserve_index]
            + state.reserve[reserve_index + 1 :]
        )
        if state.foundations[card_suit] + 1 == rank(card, n):
            foundations = list(state.foundations)
            foundations[card_suit] += 1
            yield ReserveState(tableau, remaining, tuple(foundations))
        for destination, pile in enumerate(tableau):
            if not can_build(card, pile, n):
                continue
            next_tableau = list(tableau)
            next_tableau[destination] = pile + (card,)
            yield ReserveState(
                _canonical_tableau(next_tableau),
                remaining,
                state.foundations,
            )

    for source, pile in enumerate(tableau):
        if not pile or not face_up(pile[-1]):
            continue
        top = pile[-1]
        top_suit = suit(top, n)
        if state.foundations[top_suit] + 1 == rank(top, n):
            next_tableau = list(tableau)
            next_tableau[source] = _reveal_top(pile[:-1])
            foundations = list(state.foundations)
            foundations[top_suit] += 1
            yield ReserveState(
                _canonical_tableau(next_tableau),
                state.reserve,
                tuple(foundations),
            )

        for start in range(len(pile)):
            if not packed_stack(pile, start, n):
                continue
            moving = pile[start:]
            for destination, destination_pile in enumerate(tableau):
                if destination == source or not can_build(
                    moving[0],
                    destination_pile,
                    n,
                ):
                    continue
                next_tableau = list(tableau)
                next_tableau[source] = _reveal_top(pile[:start])
                next_tableau[destination] = destination_pile + moving
                canonical = _canonical_tableau(next_tableau)
                if canonical == tableau:
                    continue
                yield ReserveState(
                    canonical,
                    state.reserve,
                    state.foundations,
                )


class ExactReserveSolver:
    """Independent, unoptimized exact reference for the reserve quotient."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.solvable: set[ReserveState] = set()
        self.unsolvable: set[ReserveState] = set()
        self.expanded_positions = 0

    def is_solvable(self, initial: ReserveState) -> bool:
        if initial in self.solvable:
            return True
        if initial in self.unsolvable:
            return False
        parents: dict[ReserveState, Optional[ReserveState]] = {
            initial: None
        }
        pending = [initial]
        while pending:
            state = pending.pop()
            if all(value == self.n for value in state.foundations):
                self._mark_path(state, parents)
                return True
            if state in self.solvable:
                self._mark_path(state, parents)
                return True
            self.expanded_positions += 1
            for next_state in reserve_successors(state, self.n):
                if next_state in self.solvable:
                    self._mark_path(state, parents)
                    return True
                if next_state in self.unsolvable or next_state in parents:
                    continue
                parents[next_state] = state
                pending.append(next_state)
        self.unsolvable.update(parents)
        return False

    def _mark_path(
        self,
        state: ReserveState,
        parents: dict[ReserveState, Optional[ReserveState]],
    ) -> None:
        current: Optional[ReserveState] = state
        while current is not None:
            self.solvable.add(current)
            current = parents[current]


@dataclass(frozen=True)
class OrderedState:
    tableau: tuple[tuple[int, ...], ...]
    stock: tuple[int, ...]
    waste: tuple[int, ...]
    foundations: tuple[int, ...]
    recycles: int


def build_ordered_state(
    deal: Sequence[int],
    n: int,
    k: int = 2,
    t: Optional[int] = None,
) -> OrderedState:
    reserve = build_reserve_state(deal, n, k, t)
    tableau_cards = tableau_card_count(n, k, t)
    return OrderedState(
        tableau=reserve.tableau,
        stock=tuple(reversed(deal[tableau_cards:])),
        waste=(),
        foundations=reserve.foundations,
        recycles=0,
    )


def ordered_successors(
    state: OrderedState,
    n: int,
    max_recycles: int = 3,
) -> Iterator[OrderedState]:
    if state.stock:
        yield OrderedState(
            state.tableau,
            state.stock[:-1],
            state.waste + (state.stock[-1],),
            state.foundations,
            state.recycles,
        )
    elif state.waste and state.recycles < max_recycles:
        yield OrderedState(
            state.tableau,
            tuple(reversed(state.waste)),
            (),
            state.foundations,
            state.recycles + 1,
        )

    if state.waste:
        card = make_face_up(state.waste[-1])
        card_suit = suit(card, n)
        waste = state.waste[:-1]
        if state.foundations[card_suit] + 1 == rank(card, n):
            foundations = list(state.foundations)
            foundations[card_suit] += 1
            yield OrderedState(
                state.tableau,
                state.stock,
                waste,
                tuple(foundations),
                state.recycles,
            )
        for destination, pile in enumerate(state.tableau):
            if not can_build(card, pile, n):
                continue
            tableau = list(state.tableau)
            tableau[destination] = pile + (card,)
            yield OrderedState(
                _canonical_tableau(tableau),
                state.stock,
                waste,
                state.foundations,
                state.recycles,
            )

    for source, pile in enumerate(state.tableau):
        if not pile or not face_up(pile[-1]):
            continue
        top = pile[-1]
        top_suit = suit(top, n)
        if state.foundations[top_suit] + 1 == rank(top, n):
            tableau = list(state.tableau)
            tableau[source] = _reveal_top(pile[:-1])
            foundations = list(state.foundations)
            foundations[top_suit] += 1
            yield OrderedState(
                _canonical_tableau(tableau),
                state.stock,
                state.waste,
                tuple(foundations),
                state.recycles,
            )
        for start in range(len(pile)):
            if not packed_stack(pile, start, n):
                continue
            moving = pile[start:]
            for destination, destination_pile in enumerate(state.tableau):
                if destination == source or not can_build(
                    moving[0],
                    destination_pile,
                    n,
                ):
                    continue
                tableau = list(state.tableau)
                tableau[source] = _reveal_top(pile[:start])
                tableau[destination] = destination_pile + moving
                canonical = _canonical_tableau(tableau)
                if canonical == state.tableau:
                    continue
                yield OrderedState(
                    canonical,
                    state.stock,
                    state.waste,
                    state.foundations,
                    state.recycles,
                )


def ordered_is_solvable(
    deal: Sequence[int],
    n: int,
    k: int = 2,
    t: Optional[int] = None,
    max_recycles: int = 3,
) -> bool:
    initial = build_ordered_state(deal, n, k, t)
    pending = [initial]
    visited = {initial}
    while pending:
        state = pending.pop()
        if all(value == n for value in state.foundations):
            return True
        for next_state in ordered_successors(state, n, max_recycles):
            if next_state not in visited:
                visited.add(next_state)
                pending.append(next_state)
    return False


def bit_at(bits: bytes, index: int) -> bool:
    if index < 0 or index >= len(bits) * 8:
        raise IndexError(index)
    return bool(bits[index // 8] & (1 << (index % 8)))


@dataclass(frozen=True)
class SolvabilityDataset:
    metadata: dict[str, object]
    bits: bytes

    @classmethod
    def load(cls, metadata_path: Path) -> "SolvabilityDataset":
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        bitset = metadata["bitset"]
        if not isinstance(bitset, dict):
            raise ValueError("invalid bitset metadata")
        bitset_path = metadata_path.parent / str(bitset["file"])
        compressed = bitset_path.read_bytes()
        if hashlib.sha256(compressed).hexdigest() != bitset["compressed_sha256"]:
            raise ValueError("compressed bitset checksum mismatch")
        bits = gzip.decompress(compressed)
        if hashlib.sha256(bits).hexdigest() != bitset["sha256"]:
            raise ValueError("bitset checksum mismatch")
        if len(bits) != int(bitset["uncompressed_bytes"]):
            raise ValueError("bitset length mismatch")
        return cls(metadata, bits)

    @property
    def deal_count(self) -> int:
        return int(self.metadata["canonical_deals"])

    def is_solvable(self, index: int) -> bool:
        if index < 0 or index >= self.deal_count:
            raise IndexError(index)
        return bit_at(self.bits, index)
