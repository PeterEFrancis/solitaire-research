from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Optional, Sequence

from .game import GameState, Move, MoveKind


PARAMETER_NAMES = (
    "safe_foundation",
    "reveal_hidden",
    "empty_king",
    "tableau_build",
    "stock_action",
    "foundation_move",
    "reveal_depth",
    "empty_source",
    "waste_to_tableau",
    "recycle",
    "destination_hidden",
    "destination_run",
    "blocks_foundation",
    "foundation_rank",
    "source_hidden",
    "creates_first_empty",
    "first_empty_with_king",
    "revealed_card_low_rank",
    "revealed_card_foundation_ready",
    "tableau_to_foundation",
    "tableau_to_tableau",
    "non_reveal_tableau_move",
    "productive_stack_length",
    "draw_playable",
    "waste_unlocks_playable",
    "empty_king_queen_access",
    "next_foundation_moves",
    "next_reveal_moves",
    "foundation_support_demand",
    "recycle_pressure",
    "draw_stock_remaining",
    "revealed_card_tableau_moves",
    "revealed_card_foundation_distance",
    "draw_foundation_ready",
    "draw_tableau_moves",
    "waste_unlocks_foundation_ready",
    "waste_unlocks_tableau_moves",
    "draw_buries_playable_waste",
    "waste_play_recycle_pressure",
    "next_empty_source_moves",
    "foundation_lag",
    "unsafe_foundation_distance",
)
FEATURE_COUNT = len(PARAMETER_NAMES)
DEFAULT_PARAMETERS = (
    8.0,
    5.0,
    0.0,
    -2.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


@dataclass(frozen=True)
class Decision:
    move: Move
    features: tuple[float, ...]
    probabilities: tuple[float, ...]
    grad_log_probability: tuple[float, ...]


class FiveParameterPlayer:
    """Softmax solitaire player with a linear, tunable move score."""

    def __init__(
        self,
        parameters: Optional[Sequence[float]] = None,
        *,
        temperature: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.parameters = list(DEFAULT_PARAMETERS if parameters is None else parameters)
        if len(self.parameters) != FEATURE_COUNT:
            raise ValueError(f"expected {FEATURE_COUNT} parameters")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.rng = rng or random.Random()

    def choose_move(
        self,
        state: GameState,
        *,
        greedy: bool = False,
        moves: Optional[Sequence[Move]] = None,
    ) -> Optional[Move]:
        decision = self.decide(state, greedy=greedy, moves=moves)
        return decision.move if decision else None

    def decide(
        self,
        state: GameState,
        *,
        greedy: bool = False,
        moves: Optional[Sequence[Move]] = None,
    ) -> Optional[Decision]:
        moves = state.legal_moves() if moves is None else tuple(moves)
        if not moves:
            return None

        features = tuple(move_features(state, move) for move in moves)
        scores = tuple(self.score(feature) / self.temperature for feature in features)

        if greedy:
            index = max(range(len(moves)), key=lambda i: scores[i])
            probabilities = tuple(1.0 if i == index else 0.0 for i in range(len(moves)))
            grad = tuple(0.0 for _ in range(FEATURE_COUNT))
            return Decision(moves[index], features[index], probabilities, grad)

        probabilities = softmax(scores)
        index = sample_index(probabilities, self.rng)
        expected_features = tuple(
            sum(probability * feature[i] for probability, feature in zip(probabilities, features))
            for i in range(FEATURE_COUNT)
        )
        grad = tuple(
            (features[index][i] - expected_features[i]) / self.temperature
            for i in range(FEATURE_COUNT)
        )
        return Decision(moves[index], features[index], probabilities, grad)

    def score(self, features: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(self.parameters, features))


def move_features(state: GameState, move: Move) -> tuple[float, ...]:
    reveals_hidden = move_reveals_hidden_card(state, move)
    empties_source = move_empties_source_column(state, move)
    fills_empty_with_king = move_fills_empty_column_with_king(state, move)
    next_state = state.apply_move(move)
    next_moves = next_state.legal_moves()

    safe_foundation = 1.0 if move_is_safe_foundation(state, move) else 0.0
    reveal_hidden = 1.0 if reveals_hidden else 0.0
    empty_king = 1.0 if fills_empty_with_king else 0.0
    tableau_build = float(move.count) if move.kind.endswith("_to_tableau") else 0.0
    stock_action = 1.0 if move.kind in (MoveKind.DRAW, MoveKind.RECYCLE) else 0.0
    foundation_move = 1.0 if foundation_card_for_move(state, move) else 0.0
    reveal_depth = float(hidden_cards_revealed_from_column(state, move))
    empty_source = 1.0 if empties_source else 0.0
    waste_to_tableau = 1.0 if move.kind == MoveKind.WASTE_TO_TABLEAU else 0.0
    recycle = 1.0 if move.kind == MoveKind.RECYCLE else 0.0
    destination = tableau_destination(state, move)
    destination_hidden = float(
        sum(not card.face_up for card in destination) if destination is not None else 0
    )
    destination_run = float(
        sum(card.face_up for card in destination) if destination is not None else 0
    )
    blocks_foundation = float(
        bool(destination) and state.can_move_to_foundation(destination[-1])
        if destination is not None
        else False
    )
    foundation_card = foundation_card_for_move(state, move)
    foundation_rank = float(foundation_card.rank if foundation_card else 0)
    source_hidden = float(hidden_cards_in_source(state, move))
    creates_first_empty = float(
        all(state.tableau) and empties_source
    )
    first_empty_with_king = float(
        creates_first_empty and has_playable_king_move(next_state)
    )
    revealed_card = revealed_card_for_move(next_state, move) if reveals_hidden else None
    revealed_card_low_rank = float(
        state.config.ranks + 1 - revealed_card.rank if revealed_card else 0
    )
    revealed_card_foundation_ready = float(
        revealed_card is not None and next_state.can_move_to_foundation(revealed_card)
    )
    tableau_to_foundation = float(move.kind == MoveKind.TABLEAU_TO_FOUNDATION)
    tableau_to_tableau = float(move.kind == MoveKind.TABLEAU_TO_TABLEAU)
    non_reveal_tableau_move = float(
        move.kind == MoveKind.TABLEAU_TO_TABLEAU
        and not reveals_hidden
        and not empties_source
    )
    productive_stack_length = float(
        move.count
        if move.kind == MoveKind.TABLEAU_TO_TABLEAU
        and (reveals_hidden or empties_source)
        else 0
    )
    draw_playable = float(
        move.kind == MoveKind.DRAW
        and bool(next_state.waste)
        and exposed_card_is_playable(next_state, next_state.waste[-1])
    )
    waste_unlocks_playable = float(
        move.kind in (MoveKind.WASTE_TO_FOUNDATION, MoveKind.WASTE_TO_TABLEAU)
        and bool(next_state.waste)
        and exposed_card_is_playable(next_state, next_state.waste[-1])
    )
    empty_king_queen_access = float(
        count_queen_moves_to_destination(next_state, next_moves, move.dest_index)
        if fills_empty_with_king
        else 0
    )
    next_foundation_moves = float(
        sum(foundation_card_for_move(next_state, candidate) is not None for candidate in next_moves)
    )
    next_reveal_moves = float(
        sum(move_reveals_hidden_card(next_state, candidate) for candidate in next_moves)
    )
    foundation_support_demand = float(
        visible_foundation_support_demand(state, foundation_card)
        if foundation_card is not None
        else 0
    )
    recycle_pressure = float(
        state.recycles_used + 1 if move.kind == MoveKind.RECYCLE else 0
    )
    draw_stock_remaining = float(
        len(state.stock) if move.kind == MoveKind.DRAW else 0
    )
    revealed_card_tableau_moves = float(
        count_tableau_destinations(next_state, revealed_card)
        if revealed_card is not None
        else 0
    )
    revealed_card_foundation_distance = float(
        max(
            0,
            revealed_card.rank
            - (next_state.foundations[revealed_card.suit] + 1),
        )
        if revealed_card is not None
        else 0
    )
    drawn_card = (
        next_state.waste[-1]
        if move.kind == MoveKind.DRAW and next_state.waste
        else None
    )
    draw_foundation_ready = float(
        drawn_card is not None and next_state.can_move_to_foundation(drawn_card)
    )
    draw_tableau_moves = float(
        count_tableau_destinations(next_state, drawn_card)
        if drawn_card is not None
        else 0
    )
    unlocked_waste_card = (
        next_state.waste[-1]
        if move.kind in (MoveKind.WASTE_TO_FOUNDATION, MoveKind.WASTE_TO_TABLEAU)
        and next_state.waste
        else None
    )
    waste_unlocks_foundation_ready = float(
        unlocked_waste_card is not None
        and next_state.can_move_to_foundation(unlocked_waste_card)
    )
    waste_unlocks_tableau_moves = float(
        count_tableau_destinations(next_state, unlocked_waste_card)
        if unlocked_waste_card is not None
        else 0
    )
    draw_buries_playable_waste = float(
        move.kind == MoveKind.DRAW
        and bool(state.waste)
        and exposed_card_is_playable(state, state.waste[-1])
    )
    waste_play_recycle_pressure = float(
        state.recycles_used
        if move.kind in (MoveKind.WASTE_TO_FOUNDATION, MoveKind.WASTE_TO_TABLEAU)
        else 0
    )
    next_empty_source_moves = float(
        sum(move_empties_source_column(next_state, candidate) for candidate in next_moves)
    )
    foundation_lag = float(
        max(state.foundations) - state.foundations[foundation_card.suit]
        if foundation_card is not None
        else 0
    )
    unsafe_foundation_distance = float(
        foundation_safety_gap(state, foundation_card)
        if foundation_card is not None
        else 0
    )
    return (
        safe_foundation,
        reveal_hidden,
        empty_king,
        tableau_build,
        stock_action,
        foundation_move,
        reveal_depth,
        empty_source,
        waste_to_tableau,
        recycle,
        destination_hidden,
        destination_run,
        blocks_foundation,
        foundation_rank,
        source_hidden,
        creates_first_empty,
        first_empty_with_king,
        revealed_card_low_rank,
        revealed_card_foundation_ready,
        tableau_to_foundation,
        tableau_to_tableau,
        non_reveal_tableau_move,
        productive_stack_length,
        draw_playable,
        waste_unlocks_playable,
        empty_king_queen_access,
        next_foundation_moves,
        next_reveal_moves,
        foundation_support_demand,
        recycle_pressure,
        draw_stock_remaining,
        revealed_card_tableau_moves,
        revealed_card_foundation_distance,
        draw_foundation_ready,
        draw_tableau_moves,
        waste_unlocks_foundation_ready,
        waste_unlocks_tableau_moves,
        draw_buries_playable_waste,
        waste_play_recycle_pressure,
        next_empty_source_moves,
        foundation_lag,
        unsafe_foundation_distance,
    )


def tableau_destination(
    state: GameState,
    move: Move,
) -> Optional[tuple]:
    if not move.kind.endswith("_to_tableau") or move.dest_index is None:
        return None
    return state.tableau[move.dest_index]


def hidden_cards_in_source(state: GameState, move: Move) -> int:
    if move.source_index is None or not move.kind.startswith("tableau_to_"):
        return 0
    return sum(not card.face_up for card in state.tableau[move.source_index])


def has_playable_king_move(state: GameState) -> bool:
    return any(
        move_fills_empty_column_with_king(state, move)
        for move in state.legal_moves()
    )


def revealed_card_for_move(state: GameState, move: Move):
    if move.source_index is None:
        return None
    source = state.tableau[move.source_index]
    return source[-1] if source else None


def exposed_card_is_playable(state: GameState, card) -> bool:
    return state.can_move_to_foundation(card) or any(
        state.can_build_on_tableau(card, destination)
        for destination in state.tableau
    )


def count_tableau_destinations(state: GameState, card) -> int:
    if card is None:
        return 0
    return sum(
        state.can_build_on_tableau(card, destination)
        for destination in state.tableau
    )


def count_queen_moves_to_destination(
    state: GameState,
    moves: Sequence[Move],
    destination_index: Optional[int],
) -> int:
    if destination_index is None:
        return 0
    return sum(
        candidate.kind.endswith("_to_tableau")
        and candidate.dest_index == destination_index
        and (card := moving_card_for_move(state, candidate)) is not None
        and card.rank == state.config.ranks - 1
        for candidate in moves
    )


def visible_foundation_support_demand(state: GameState, card) -> int:
    if card.rank <= 1:
        return 0
    needed_rank = card.rank - 1
    demand = int(
        bool(state.waste)
        and state.waste[-1].rank == needed_rank
        and state.waste[-1].color != card.color
    )
    for pile in state.tableau:
        demand += sum(
            candidate.face_up
            and candidate.rank == needed_rank
            and candidate.color != card.color
            for candidate in pile
        )
    return demand


def move_is_safe_foundation(state: GameState, move: Move) -> bool:
    card = foundation_card_for_move(state, move)
    if card is None:
        return False
    if card.rank == 1:
        return True

    opposite_color_foundations = [
        rank for suit, rank in enumerate(state.foundations) if suit % 2 != card.color
    ]
    return min(opposite_color_foundations) >= card.rank - 1


def foundation_safety_gap(state: GameState, card) -> int:
    opposite_color_foundations = [
        rank for suit, rank in enumerate(state.foundations) if suit % 2 != card.color
    ]
    return max(0, card.rank - 1 - min(opposite_color_foundations))


def foundation_card_for_move(state: GameState, move: Move):
    if move.kind == MoveKind.WASTE_TO_FOUNDATION:
        return state.waste[-1] if state.waste else None
    if move.kind == MoveKind.TABLEAU_TO_FOUNDATION and move.source_index is not None:
        source = state.tableau[move.source_index]
        return source[-1] if source else None
    return None


def move_reveals_hidden_card(state: GameState, move: Move) -> bool:
    if move.kind not in (MoveKind.TABLEAU_TO_FOUNDATION, MoveKind.TABLEAU_TO_TABLEAU):
        return False
    if move.source_index is None:
        return False
    source = state.tableau[move.source_index]
    remaining = len(source) - move.count
    return remaining > 0 and not source[remaining - 1].face_up


def hidden_cards_revealed_from_column(state: GameState, move: Move) -> int:
    if not move_reveals_hidden_card(state, move) or move.source_index is None:
        return 0
    return sum(not card.face_up for card in state.tableau[move.source_index])


def move_empties_source_column(state: GameState, move: Move) -> bool:
    if move.source_index is None:
        return False
    if move.kind not in (MoveKind.TABLEAU_TO_FOUNDATION, MoveKind.TABLEAU_TO_TABLEAU):
        return False
    return move.count == len(state.tableau[move.source_index])


def move_fills_empty_column_with_king(state: GameState, move: Move) -> bool:
    if not move.kind.endswith("_to_tableau") or move.dest_index is None:
        return False
    if state.tableau[move.dest_index]:
        return False
    moving_card = moving_card_for_move(state, move)
    return moving_card is not None and moving_card.rank == state.config.ranks


def moving_card_for_move(state: GameState, move: Move):
    if move.kind == MoveKind.WASTE_TO_TABLEAU:
        return state.waste[-1] if state.waste else None
    if move.kind == MoveKind.TABLEAU_TO_TABLEAU and move.source_index is not None:
        source = state.tableau[move.source_index]
        if move.count <= len(source):
            return source[-move.count]
    return None


def softmax(scores: Sequence[float]) -> tuple[float, ...]:
    top = max(scores)
    exp_scores = [math.exp(score - top) for score in scores]
    total = sum(exp_scores)
    return tuple(score / total for score in exp_scores)


def sample_index(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1
