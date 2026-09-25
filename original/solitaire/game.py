from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Optional

from .cards import Card


class MoveKind:
    DRAW = "draw"
    RECYCLE = "recycle"
    WASTE_TO_FOUNDATION = "waste_to_foundation"
    TABLEAU_TO_FOUNDATION = "tableau_to_foundation"
    WASTE_TO_TABLEAU = "waste_to_tableau"
    TABLEAU_TO_TABLEAU = "tableau_to_tableau"


@dataclass(frozen=True, init=False)
class DeckConfig:
    n: int
    k: int
    t: Optional[int]
    draw_count: int = 1
    max_recycles: Optional[int] = 3
    allow_tableau_stack_splitting: bool = True

    def __init__(
        self,
        n: int = 13,
        k: int = 2,
        t: Optional[int] = None,
        draw_count: int = 1,
        max_recycles: Optional[int] = 3,
        allow_tableau_stack_splitting: bool = True,
        *,
        ranks: Optional[int] = None,
        suits: Optional[int] = None,
        tableau_columns: Optional[int] = None,
    ) -> None:
        """Configure the deck family.

        ``n`` is the number of ranks per suit, ``k`` is the number of suits per
        color, and ``t`` is the number of tableau piles. The legacy
        ``ranks``/``suits``/``tableau_columns`` names are accepted as aliases.
        """
        if ranks is not None:
            n = ranks
        if suits is not None:
            if suits % 2 != 0:
                raise ValueError("suits must be even because there are two colors")
            k = suits // 2
        if tableau_columns is not None:
            t = tableau_columns

        if n < 1:
            raise ValueError("n must be >= 1")
        if k < 1:
            raise ValueError("k must be >= 1")
        if t is not None and t < 1:
            raise ValueError("t must be >= 1")
        if draw_count < 1:
            raise ValueError("draw_count must be >= 1")
        if max_recycles is not None and max_recycles < 0:
            raise ValueError("max_recycles must be >= 0 or None")
        if not isinstance(allow_tableau_stack_splitting, bool):
            raise ValueError("allow_tableau_stack_splitting must be a boolean")

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "draw_count", draw_count)
        object.__setattr__(self, "max_recycles", max_recycles)
        object.__setattr__(self, "allow_tableau_stack_splitting", allow_tableau_stack_splitting)

    @property
    def suits(self) -> int:
        return 2 * self.k

    @property
    def ranks(self) -> int:
        return self.n

    @property
    def tableau_columns(self) -> Optional[int]:
        return self.t

    @property
    def deck_size(self) -> int:
        return self.suits * self.ranks

    def default_tableau_columns(self) -> int:
        target_cards = self.n * self.k
        columns = (math.isqrt(1 + 8 * target_cards) - 1) // 2
        if columns * (columns + 1) // 2 < target_cards:
            columns += 1
        return columns

    def resolved_tableau_columns(self) -> int:
        return self.default_tableau_columns() if self.t is None else self.t

    def rule_key(self) -> tuple[object, ...]:
        """Identity shared by parameter records and cached game positions."""
        return (
            self.n, self.k, self.resolved_tableau_columns(), self.draw_count,
            self.max_recycles, self.allow_tableau_stack_splitting,
        )


@dataclass(frozen=True)
class Move:
    kind: str
    source_index: Optional[int] = None
    dest_index: Optional[int] = None
    count: int = 1

    def __str__(self) -> str:
        parts = [self.kind]
        if self.source_index is not None:
            parts.append(f"src={self.source_index}")
        if self.dest_index is not None:
            parts.append(f"dst={self.dest_index}")
        if self.count != 1:
            parts.append(f"n={self.count}")
        return "(" + ", ".join(parts) + ")"


@dataclass(frozen=True)
class GameState:
    config: DeckConfig
    tableau: tuple[tuple[Card, ...], ...]
    stock: tuple[Card, ...]
    waste: tuple[Card, ...]
    foundations: tuple[int, ...]
    recycles_used: int = 0
    steps: int = 0

    def is_won(self) -> bool:
        return all(rank == self.config.ranks for rank in self.foundations)

    def foundation_progress(self) -> float:
        return sum(self.foundations) / self.config.deck_size

    def position_key(self) -> tuple[object, ...]:
        return (
            self.config.rule_key(),
            self.tableau,
            self.stock,
            self.waste,
            self.foundations,
            self.recycles_used if self.config.max_recycles is not None else 0,
        )

    def legal_moves(self) -> tuple[Move, ...]:
        moves: list[Move] = []

        if self.stock:
            moves.append(Move(MoveKind.DRAW))
        elif self.waste and (
            self.config.max_recycles is None
            or self.recycles_used < self.config.max_recycles
        ):
            moves.append(Move(MoveKind.RECYCLE))

        if self.waste:
            card = self.waste[-1]
            if self.can_move_to_foundation(card):
                moves.append(Move(MoveKind.WASTE_TO_FOUNDATION))
            for dest_index, dest in enumerate(self.tableau):
                if self.can_build_on_tableau(card, dest):
                    moves.append(Move(MoveKind.WASTE_TO_TABLEAU, dest_index=dest_index))

        for source_index, pile in enumerate(self.tableau):
            if not pile or not pile[-1].face_up:
                continue

            top_card = pile[-1]
            if self.can_move_to_foundation(top_card):
                moves.append(
                    Move(
                        MoveKind.TABLEAU_TO_FOUNDATION,
                        source_index=source_index,
                    )
                )

            for start in range(len(pile)):
                moving_stack = pile[start:]
                if not moving_stack[0].face_up:
                    continue
                if (
                    not self.config.allow_tableau_stack_splitting
                    and start > 0
                    and pile[start - 1].face_up
                ):
                    continue
                if not is_packed_tableau_stack(moving_stack):
                    continue
                count = len(moving_stack)
                moving_card = moving_stack[0]
                for dest_index, dest in enumerate(self.tableau):
                    if dest_index == source_index:
                        continue
                    if self.can_build_on_tableau(moving_card, dest):
                        moves.append(
                            Move(
                                MoveKind.TABLEAU_TO_TABLEAU,
                                source_index=source_index,
                                dest_index=dest_index,
                                count=count,
                            )
                        )

        return tuple(moves)

    def can_move_to_foundation(self, card: Card) -> bool:
        return card.face_up and self.foundations[card.suit] + 1 == card.rank

    def can_build_on_tableau(self, card: Card, dest: tuple[Card, ...]) -> bool:
        if not card.face_up:
            return False
        if not dest:
            return card.rank == self.config.ranks
        top = dest[-1]
        return (
            top.face_up
            and card.rank + 1 == top.rank
            and card.color != top.color
        )

    def apply_move(self, move: Move, *, validate: bool = False) -> "GameState":
        if validate and move not in self.legal_moves():
            raise ValueError(f"illegal move: {move}")

        tableau = [list(pile) for pile in self.tableau]
        stock = list(self.stock)
        waste = list(self.waste)
        foundations = list(self.foundations)
        recycles_used = self.recycles_used

        if move.kind == MoveKind.DRAW:
            for _ in range(min(self.config.draw_count, len(stock))):
                waste.append(stock.pop().face_up_card())

        elif move.kind == MoveKind.RECYCLE:
            stock = [card.face_down_card() for card in reversed(waste)]
            waste = []
            recycles_used = recycles_used + 1 if self.config.max_recycles is not None else 0

        elif move.kind == MoveKind.WASTE_TO_FOUNDATION:
            card = waste.pop()
            foundations[card.suit] += 1

        elif move.kind == MoveKind.TABLEAU_TO_FOUNDATION:
            source = require_index(move.source_index, "source_index")
            card = tableau[source].pop()
            foundations[card.suit] += 1
            reveal_tableau_top(tableau[source])

        elif move.kind == MoveKind.WASTE_TO_TABLEAU:
            dest = require_index(move.dest_index, "dest_index")
            tableau[dest].append(waste.pop().face_up_card())

        elif move.kind == MoveKind.TABLEAU_TO_TABLEAU:
            source = require_index(move.source_index, "source_index")
            dest = require_index(move.dest_index, "dest_index")
            moving_stack = tableau[source][-move.count :]
            del tableau[source][-move.count :]
            tableau[dest].extend(moving_stack)
            reveal_tableau_top(tableau[source])

        else:
            raise ValueError(f"unknown move kind: {move.kind}")

        return GameState(
            config=self.config,
            tableau=tuple(tuple(pile) for pile in tableau),
            stock=tuple(stock),
            waste=tuple(waste),
            foundations=tuple(foundations),
            recycles_used=recycles_used,
            steps=self.steps + 1,
        )

    def pretty(self) -> str:
        lines = [
            f"Foundations: {self.foundations}",
            f"Stock: {len(self.stock)} Waste: {self.waste[-1] if self.waste else '--'}",
        ]
        for index, pile in enumerate(self.tableau):
            cards = " ".join(str(card) for card in pile)
            lines.append(f"T{index}: {cards}")
        return "\n".join(lines)


def require_index(value: Optional[int], name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def reveal_tableau_top(pile: list[Card]) -> None:
    if pile and not pile[-1].face_up:
        pile[-1] = pile[-1].face_up_card()


def is_packed_tableau_stack(cards: Iterable[Card]) -> bool:
    stack = tuple(cards)
    if not stack or any(not card.face_up for card in stack):
        return False
    return all(
        lower.rank == upper.rank + 1 and lower.color != upper.color
        for lower, upper in zip(stack, stack[1:])
    )


def new_game(seed: Optional[int] = None, config: Optional[DeckConfig] = None) -> GameState:
    config = config or DeckConfig()
    rng = random.Random(seed)
    deck = [
        Card(suit=suit, rank=rank, face_up=False)
        for suit in range(config.suits)
        for rank in range(1, config.ranks + 1)
    ]
    rng.shuffle(deck)

    tableau: list[tuple[Card, ...]] = []
    for column in range(config.resolved_tableau_columns()):
        pile: list[Card] = []
        for row in range(column + 1):
            if not deck:
                break
            card = deck.pop()
            pile.append(card.face_up_card() if row == column else card.face_down_card())
        reveal_tableau_top(pile)
        tableau.append(tuple(pile))

    return GameState(
        config=config,
        tableau=tuple(tableau),
        stock=tuple(card.face_down_card() for card in deck),
        waste=(),
        foundations=tuple(0 for _ in range(config.suits)),
    )
