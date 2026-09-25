from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from itertools import permutations, product
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from solitaire.cards import Card
from solitaire.game import DeckConfig, GameState


Deal = tuple[int, ...]
PositionKey = tuple[object, ...]


def suit_symmetry_maps(k: int) -> tuple[tuple[int, ...], ...]:
    """Return every relabeling that preserves alternating-color rules."""
    if k < 1:
        raise ValueError("k must be >= 1")

    maps: list[tuple[int, ...]] = []
    for swap_colors in (0, 1):
        for color_zero in permutations(range(k)):
            for color_one in permutations(range(k)):
                color_permutations = (color_zero, color_one)
                mapping = []
                for suit in range(2 * k):
                    color = suit % 2
                    suit_in_color = suit // 2
                    mapped_color = color ^ swap_colors
                    mapped_in_color = color_permutations[color][suit_in_color]
                    mapping.append(2 * mapped_in_color + mapped_color)
                maps.append(tuple(mapping))
    return tuple(maps)


def card_symmetry_maps(n: int, k: int) -> tuple[tuple[int, ...], ...]:
    maps = []
    for suit_map in suit_symmetry_maps(k):
        card_map = []
        for suit in range(2 * k):
            for rank_index in range(n):
                card_map.append(suit_map[suit] * n + rank_index)
        maps.append(tuple(card_map))
    return tuple(maps)


def symmetry_orbit_size(k: int) -> int:
    return 2 * math.factorial(k) ** 2


def expected_canonical_deal_count(n: int, k: int) -> int:
    raw_deals = math.factorial(2 * n * k)
    orbit_size = symmetry_orbit_size(k)
    quotient, remainder = divmod(raw_deals, orbit_size)
    if remainder:
        raise AssertionError("suit symmetry group does not divide deal count")
    return quotient


def is_canonical_deal(deal: Deal, transforms: Sequence[Sequence[int]]) -> bool:
    """Whether deal is the lexicographic representative of its suit orbit."""
    for transform in transforms:
        if tuple(transform[card_id] for card_id in deal) < deal:
            return False
    return True


def iter_canonical_deals(n: int, k: int = 2) -> Iterator[Deal]:
    """Yield canonical deals in deterministic lexicographic order."""
    transforms = card_symmetry_maps(n, k)
    for deal in permutations(range(2 * n * k)):
        if is_canonical_deal(deal, transforms):
            yield deal


SUIT_PATTERN_RANK_ENUMERATION = "suit-pattern-rank-product-v1"
TABLEAU_PATTERN_RANK_ENUMERATION = "tableau-pattern-partial-rank-v1"


def iter_canonical_suit_patterns(n: int) -> Iterator[tuple[int, ...]]:
    """Yield one k=2 suit pattern per color/suit automorphism orbit."""
    if n < 1:
        raise ValueError("n must be >= 1")

    counts = [n - 1, n, n, n]
    pattern = [0]

    def extend(seen_suit_one: bool) -> Iterator[tuple[int, ...]]:
        if len(pattern) == 4 * n:
            yield tuple(pattern)
            return
        for suit in range(4):
            if counts[suit] == 0:
                continue
            if suit == 3 and not seen_suit_one:
                continue
            counts[suit] -= 1
            pattern.append(suit)
            yield from extend(seen_suit_one or suit == 1)
            pattern.pop()
            counts[suit] += 1

    yield from extend(False)


def expected_canonical_suit_pattern_count(n: int) -> int:
    return math.factorial(4 * n) // math.factorial(n) ** 4 // 8


def iter_suit_pattern_rank_deals(n: int, k: int = 2) -> Iterator[Deal]:
    """Directly generate canonical k=2 deals without scanning raw deals."""
    if k != 2:
        raise ValueError("direct suit-pattern enumeration currently requires k=2")

    rank_orders = tuple(permutations(range(n)))
    for pattern in iter_canonical_suit_patterns(n):
        for ranks_by_suit in product(rank_orders, repeat=4):
            occurrences = [0, 0, 0, 0]
            deal = []
            for suit in pattern:
                rank_index = ranks_by_suit[suit][occurrences[suit]]
                occurrences[suit] += 1
                deal.append(suit * n + rank_index)
            yield tuple(deal)


def iter_canonical_tableau_patterns(
    n: int,
    tableau_cards: int,
) -> Iterator[tuple[int, ...]]:
    """Yield direct k=2 suit representatives for an ordered tableau prefix."""
    counts = [1, 0, 0, 0]
    pattern = [0]

    def extend(seen_suit_one: bool) -> Iterator[tuple[int, ...]]:
        if len(pattern) == tableau_cards:
            yield tuple(pattern)
            return
        for suit in range(4):
            if counts[suit] == n:
                continue
            if suit == 3 and not seen_suit_one:
                continue
            counts[suit] += 1
            pattern.append(suit)
            yield from extend(seen_suit_one or suit == 1)
            pattern.pop()
            counts[suit] -= 1

    yield from extend(False)


def iter_stock_collapsed_deals(n: int, k: int = 2) -> Iterator[Deal]:
    """Yield suit-canonical tableau layouts with remaining stock sorted."""
    if k != 2:
        raise ValueError("stock-collapsed enumeration currently requires k=2")
    config = DeckConfig(n=n, k=k)
    columns = config.resolved_tableau_columns()
    tableau_cards = columns * (columns + 1) // 2
    for pattern in iter_canonical_tableau_patterns(n, tableau_cards):
        counts = [pattern.count(suit) for suit in range(4)]
        rank_orders = [tuple(permutations(range(n), count)) for count in counts]
        for ranks_by_suit in product(*rank_orders):
            occurrences = [0, 0, 0, 0]
            tableau = []
            for suit in pattern:
                rank_index = ranks_by_suit[suit][occurrences[suit]]
                occurrences[suit] += 1
                tableau.append(suit * n + rank_index)
            used = set(tableau)
            stock = [card for card in range(4 * n) if card not in used]
            yield tuple(tableau + stock)


def card_from_id(card_id: int, n: int, *, face_up: bool = False) -> Card:
    suit, rank_index = divmod(card_id, n)
    return Card(suit=suit, rank=rank_index + 1, face_up=face_up)


def build_initial_state(deal: Sequence[int], config: DeckConfig) -> GameState:
    """Build a state from tableau-deal order followed by stock-draw order."""
    if len(deal) != config.deck_size or set(deal) != set(range(config.deck_size)):
        raise ValueError("deal must be a permutation of every card id")

    cursor = 0
    tableau = []
    for column in range(config.resolved_tableau_columns()):
        pile = []
        for row in range(column + 1):
            if cursor == len(deal):
                break
            pile.append(card_from_id(deal[cursor], config.n, face_up=row == column))
            cursor += 1
        if pile and not pile[-1].face_up:
            pile[-1] = pile[-1].face_up_card()
        tableau.append(tuple(pile))

    # GameState draws with stock.pop(), so reverse the remaining draw-order cards.
    stock = tuple(
        card_from_id(card_id, config.n)
        for card_id in reversed(deal[cursor:])
    )
    return GameState(
        config=config,
        tableau=tuple(tableau),
        stock=stock,
        waste=(),
        foundations=tuple(0 for _ in range(config.suits)),
    )


class ExactSolver:
    """Prove whether a position can reach a win by exploring its state graph."""

    def __init__(self) -> None:
        self.solvable_positions: set[PositionKey] = set()
        self.unsolvable_positions: set[PositionKey] = set()
        self.expanded_positions = 0
        self.solvable_cache_hits = 0
        self.unsolvable_cache_hits = 0

    def is_solvable(self, initial_state: GameState) -> bool:
        initial_key = initial_state.position_key()
        if initial_state.is_won() or initial_key in self.solvable_positions:
            self.solvable_cache_hits += 1
            return True
        if initial_key in self.unsolvable_positions:
            self.unsolvable_cache_hits += 1
            return False

        parents: dict[PositionKey, Optional[PositionKey]] = {initial_key: None}
        pending = [initial_state]

        while pending:
            state = pending.pop()
            state_key = state.position_key()
            if state.is_won() or state_key in self.solvable_positions:
                self._mark_path_solvable(state_key, parents)
                return True

            self.expanded_positions += 1
            for move in state.legal_moves():
                next_state = state.apply_move(move)
                next_key = next_state.position_key()
                if next_state.is_won() or next_key in self.solvable_positions:
                    self._mark_path_solvable(state_key, parents)
                    return True
                if next_key in self.unsolvable_positions or next_key in parents:
                    continue
                parents[next_key] = state_key
                pending.append(next_state)

        # The complete reachable closure contains no win, so every visited
        # position is globally safe to memoize as unsolvable.
        self.unsolvable_positions.update(parents)
        return False

    def _mark_path_solvable(
        self,
        position: PositionKey,
        parents: dict[PositionKey, Optional[PositionKey]],
    ) -> None:
        current: Optional[PositionKey] = position
        while current is not None:
            self.solvable_positions.add(current)
            current = parents[current]


class BitsetBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.count = 0

    def append(self, value: bool) -> None:
        if self.count % 8 == 0:
            self.data.append(0)
        if value:
            self.data[-1] |= 1 << (self.count % 8)
        self.count += 1


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
        if len(compressed) != int(bitset["compressed_bytes"]):
            raise ValueError("compressed bitset length mismatch")
        compressed_digest = hashlib.sha256(compressed).hexdigest()
        if compressed_digest != bitset["compressed_sha256"]:
            raise ValueError("compressed bitset checksum mismatch")
        bits = gzip.decompress(compressed)
        deal_count = int(metadata["canonical_deals"])
        if int(bitset["bits"]) != deal_count:
            raise ValueError("bit count does not match canonical deal count")
        if len(bits) != (deal_count + 7) // 8:
            raise ValueError("uncompressed bitset length mismatch")
        if len(bits) != int(bitset["uncompressed_bytes"]):
            raise ValueError("recorded uncompressed bitset length mismatch")
        padding_bits = len(bits) * 8 - deal_count
        if padding_bits and bits[-1] >> (8 - padding_bits):
            raise ValueError("nonzero bitset padding")
        if hashlib.sha256(bits).hexdigest() != bitset["sha256"]:
            raise ValueError("bitset checksum mismatch")
        return cls(metadata=metadata, bits=bits)

    @property
    def deal_count(self) -> int:
        return int(self.metadata["canonical_deals"])

    def is_solvable(self, deal_index: int) -> bool:
        if deal_index < 0 or deal_index >= self.deal_count:
            raise IndexError(deal_index)
        return bit_at(self.bits, deal_index)

    def iter_deals(self) -> Iterable[tuple[Deal, bool]]:
        config = self.metadata["config"]
        if not isinstance(config, dict):
            raise ValueError("invalid config metadata")
        n = int(config["n"])
        k = int(config["k"])
        deal_order = self.metadata.get("deal_order", {})
        enumeration_id = (
            deal_order.get("enumeration_id") if isinstance(deal_order, dict) else None
        )
        if enumeration_id == SUIT_PATTERN_RANK_ENUMERATION:
            deals = iter_suit_pattern_rank_deals(n, k)
        elif enumeration_id == TABLEAU_PATTERN_RANK_ENUMERATION:
            deals = iter_stock_collapsed_deals(n, k)
        else:
            deals = iter_canonical_deals(n, k)
        for index, deal in enumerate(deals):
            yield deal, self.is_solvable(index)
