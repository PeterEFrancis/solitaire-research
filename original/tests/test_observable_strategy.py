"""Check that proposed human strategies do not inspect unknown card identities.

The stock is conservatively treated as unknown even on repeat passes. Buried
waste is held fixed when testing perfect-memory features: under draw-one, every
waste card was previously exposed. These are feature/decision checks, not claims
about human performance or native/Python trajectory parity.
"""

from dataclasses import replace
import json
from pathlib import Path
import random
import unittest

from solitaire import Card, DeckConfig, FiveParameterPlayer, GameState, Move, MoveKind, new_game
from solitaire.player import PARAMETER_NAMES, move_features
from solitaire.train import moves_to_unseen_positions


CANDIDATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "brute_force/results/human-strategy-candidates.json"
)
REPORT = json.loads(CANDIDATE_PATH.read_text())
CURRENT_VISIBLE = tuple(REPORT["audit"]["current_visible_features"])
WASTE_HISTORY = tuple(REPORT["audit"]["waste_history_features"])


def public_position(state, *, remember_waste=False):
    """Only exposed cards/counts; optionally remember previously exposed waste."""
    return (
        state.config,
        tuple(
            tuple((card.suit, card.rank) if card.face_up else None for card in pile)
            for pile in state.tableau
        ),
        len(state.stock),
        tuple((card.suit, card.rank) for card in state.waste)
        if remember_waste else (
            len(state.waste),
            (state.waste[-1].suit, state.waste[-1].rank) if state.waste else None,
        ),
        state.foundations,
        state.recycles_used,
    )


def permute_unknown_cards(state, seed):
    """Preserve the deck multiset and every visible card and waste history."""
    unknown = [card for pile in state.tableau for card in pile if not card.face_up]
    unknown.extend(state.stock)
    random.Random(seed).shuffle(unknown)
    remaining = iter(unknown)
    tableau = tuple(
        tuple(card if card.face_up else next(remaining).face_down_card() for card in pile)
        for pile in state.tableau
    )
    stock = tuple(next(remaining).face_down_card() for _ in state.stock)
    return replace(state, tableau=tableau, stock=stock)


def selected_features(state, move, names):
    values = move_features(state, move)
    return tuple(values[PARAMETER_NAMES.index(name)] for name in names)


class ObservableStrategyTests(unittest.TestCase):
    def test_candidate_schema_and_feature_partition(self):
        self.assertEqual(tuple(REPORT["feature_names"]), PARAMETER_NAMES)
        groups = [CURRENT_VISIBLE, WASTE_HISTORY, REPORT["audit"]["unknown_identity_features"]]
        flattened = [name for group in groups for name in group]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(PARAMETER_NAMES))
        self.assertEqual(len(CURRENT_VISIBLE), 28)
        self.assertTrue(set(PARAMETER_NAMES[:17]).issubset(CURRENT_VISIBLE))
        for candidate in REPORT["candidates"]:
            with self.subTest(candidate=candidate["name"]):
                self.assertEqual(set(candidate["weights"]), set(PARAMETER_NAMES))
                nonzero = {name for name, weight in candidate["weights"].items() if weight}
                self.assertTrue(nonzero.issubset(CURRENT_VISIBLE))
                self.assertEqual(len(nonzero), candidate["nonzero_terms"])

    def test_observable_features_and_choices_survive_hidden_permutations(self):
        # Sample reachable positions, rather than only specially constructed
        # feature fixtures. The small fixed sample keeps this a fast regression
        # test; it includes states after draws, reveals, and foundation moves.
        players = [
            FiveParameterPlayer([candidate["weights"][name] for name in PARAMETER_NAMES])
            for candidate in REPORT["candidates"]
        ]
        for seed in range(5):
            state = new_game(seed=seed)
            rng = random.Random(seed + 901)
            for step in range(12):
                alternative = permute_unknown_cards(state, seed * 100 + step)
                self.assertEqual(public_position(state), public_position(alternative))
                self.assertEqual(state.legal_moves(), alternative.legal_moves())
                for move in state.legal_moves():
                    with self.subTest(seed=seed, step=step, move=move):
                        self.assertEqual(
                            selected_features(state, move, CURRENT_VISIBLE + WASTE_HISTORY),
                            selected_features(alternative, move, CURRENT_VISIBLE + WASTE_HISTORY),
                        )
                for player in players:
                    self.assertEqual(
                        player.choose_move(state, greedy=True),
                        player.choose_move(alternative, greedy=True),
                    )
                moves = state.legal_moves()
                if not moves:
                    break
                state = state.apply_move(rng.choice(moves))

    def test_oracle_draw_features_really_detect_unseen_identity(self):
        state = GameState(
            config=DeckConfig(n=4, k=1, t=2),
            tableau=((Card(1, 2, True),), (Card(0, 4, True),)),
            stock=(Card(0, 3), Card(0, 1)),
            waste=(),
            foundations=(0, 0),
        )
        alternative = replace(state, stock=tuple(reversed(state.stock)))
        draw = Move(MoveKind.DRAW)
        self.assertEqual(public_position(state), public_position(alternative))
        self.assertEqual(
            selected_features(state, draw, CURRENT_VISIBLE),
            selected_features(alternative, draw, CURRENT_VISIBLE),
        )
        for name in ("draw_playable", "draw_foundation_ready", "draw_tableau_moves"):
            with self.subTest(feature=name):
                self.assertNotEqual(
                    selected_features(state, draw, (name,)),
                    selected_features(alternative, draw, (name,)),
                )

    def test_king_queen_access_can_peek_at_hidden_source_queen(self):
        # Moving the visible red king reveals either a black queen or an ace.
        # The queen would immediately play onto that king, which is not known
        # when the candidate king move is being scored.
        state = GameState(
            config=DeckConfig(n=4, k=1, t=3),
            tableau=((Card(0, 3), Card(1, 4, True)), (), (Card(1, 2, True),)),
            stock=(Card(0, 1),),
            waste=(),
            foundations=(0, 0),
        )
        alternative = replace(
            state,
            tableau=((Card(0, 1), Card(1, 4, True)), (), (Card(1, 2, True),)),
            stock=(Card(0, 3),),
        )
        move = Move(MoveKind.TABLEAU_TO_TABLEAU, source_index=0, dest_index=1)
        self.assertIn(move, state.legal_moves())
        self.assertEqual(public_position(state), public_position(alternative))
        self.assertEqual(
            selected_features(state, move, CURRENT_VISIBLE),
            selected_features(alternative, move, CURRENT_VISIBLE),
        )
        self.assertNotEqual(
            selected_features(state, move, ("empty_king_queen_access",)),
            selected_features(alternative, move, ("empty_king_queen_access",)),
        )

    def test_waste_unlock_features_require_waste_history(self):
        # Both positions show the same top waste card. Swapping previously
        # exposed buried cards preserves the current view but changes memory.
        state = GameState(
            config=DeckConfig(n=4, k=1, t=2),
            tableau=((Card(1, 2, True),), (Card(0, 4, True),)),
            stock=(),
            waste=(Card(0, 3, True), Card(0, 1, True), Card(1, 3, True)),
            foundations=(0, 0),
        )
        alternative = replace(state, waste=(state.waste[1], state.waste[0], state.waste[2]))
        move = Move(MoveKind.WASTE_TO_TABLEAU, dest_index=1)
        self.assertIn(move, state.legal_moves())
        self.assertEqual(public_position(state), public_position(alternative))
        self.assertNotEqual(
            public_position(state, remember_waste=True),
            public_position(alternative, remember_waste=True),
        )
        self.assertEqual(
            selected_features(state, move, CURRENT_VISIBLE),
            selected_features(alternative, move, CURRENT_VISIBLE),
        )
        for name in WASTE_HISTORY:
            with self.subTest(feature=name):
                self.assertNotEqual(
                    selected_features(state, move, (name,)),
                    selected_features(alternative, move, (name,)),
                )

    def test_current_visible_scores_do_not_depend_on_buried_waste_order(self):
        for seed in range(5):
            state = new_game(seed=seed)
            for _ in range(4):
                state = state.apply_move(Move(MoveKind.DRAW))
            alternative = replace(state, waste=tuple(reversed(state.waste[:-1])) + state.waste[-1:])
            self.assertEqual(public_position(state), public_position(alternative))
            for move in state.legal_moves():
                self.assertEqual(
                    selected_features(state, move, CURRENT_VISIBLE),
                    selected_features(alternative, move, CURRENT_VISIBLE),
                )

    def test_cycle_filter_invariance_on_coupled_visible_history(self):
        state = GameState(
            config=DeckConfig(n=4, k=2, t=4),
            tableau=(
                (Card(0, 3, True), Card(1, 2, True)),
                (Card(1, 4, True),),
                (Card(0, 1), Card(2, 3, True)),
                (),
            ),
            stock=(Card(1, 1),),
            waste=(),
            foundations=(0, 0, 0, 0),
        )
        alternative = replace(
            state,
            tableau=state.tableau[:2] + ((Card(1, 1), Card(2, 3, True)), ()),
            stock=(Card(0, 1),),
        )
        seen, other_seen = {state.position_key()}, {alternative.position_key()}
        forward = Move(MoveKind.TABLEAU_TO_TABLEAU, source_index=0, dest_index=2)
        for before, visited in ((state, seen), (alternative, other_seen)):
            self.assertIn(forward, moves_to_unseen_positions(before, visited))
        state, alternative = state.apply_move(forward), alternative.apply_move(forward)
        seen.add(state.position_key())
        other_seen.add(alternative.position_key())
        self.assertEqual(public_position(state), public_position(alternative))
        self.assertEqual(
            moves_to_unseen_positions(state, seen),
            moves_to_unseen_positions(alternative, other_seen),
        )
        reverse = Move(MoveKind.TABLEAU_TO_TABLEAU, source_index=2, dest_index=0)
        self.assertIn(reverse, state.legal_moves())
        self.assertNotIn(reverse, moves_to_unseen_positions(state, seen))

    def test_ties_choose_first_generated_eligible_move(self):
        state = new_game(seed=7)
        player = FiveParameterPlayer([0.0] * len(PARAMETER_NAMES))
        self.assertEqual(player.choose_move(state, greedy=True), state.legal_moves()[0])
        reversed_moves = tuple(reversed(state.legal_moves()))
        self.assertEqual(
            player.choose_move(state, greedy=True, moves=reversed_moves), reversed_moves[0]
        )


if __name__ == "__main__":
    unittest.main()
