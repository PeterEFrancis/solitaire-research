import random
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from solitaire import Card, DeckConfig, FiveParameterPlayer, GameState, Move, MoveKind, new_game
from solitaire.parameter_store import load_parameters_for_config, save_parameter_record
from solitaire.player import FEATURE_COUNT, PARAMETER_NAMES, move_features
from solitaire.train import PolicyGradientTrainer, evaluate


class SolitaireEngineTests(unittest.TestCase):
    def test_new_game_preserves_all_cards(self):
        config = DeckConfig()
        state = new_game(seed=7, config=config)
        total_cards = (
            sum(len(pile) for pile in state.tableau)
            + len(state.stock)
            + len(state.waste)
            + sum(state.foundations)
        )
        self.assertEqual(total_cards, config.deck_size)
        self.assertEqual(len(state.tableau), config.default_tableau_columns())

    def test_default_tableau_columns_use_n_and_k(self):
        self.assertEqual(DeckConfig(n=13, k=2).resolved_tableau_columns(), 7)
        self.assertEqual(DeckConfig(n=3, k=2).resolved_tableau_columns(), 3)
        self.assertEqual(DeckConfig(n=4, k=1).resolved_tableau_columns(), 3)
        self.assertEqual(DeckConfig(n=3, k=2, t=4).resolved_tableau_columns(), 4)

    def test_position_key_ignores_step_counter(self):
        state = new_game(seed=9)
        self.assertEqual(state.position_key(), replace(state, steps=99).position_key())

    def test_small_deck_preserves_requested_tableau_columns(self):
        config = DeckConfig(n=4, k=1, t=7)
        state = new_game(seed=3, config=config)
        self.assertEqual(len(state.tableau), config.resolved_tableau_columns())
        total_cards = (
            sum(len(pile) for pile in state.tableau)
            + len(state.stock)
            + len(state.waste)
            + sum(state.foundations)
        )
        self.assertEqual(total_cards, config.deck_size)

    def test_every_legal_move_has_all_tunable_features(self):
        state = new_game(seed=11)
        for move in state.legal_moves():
            self.assertEqual(len(move_features(state, move)), FEATURE_COUNT)

    def test_draw_playable_looks_one_move_ahead(self):
        state = GameState(
            config=DeckConfig(n=3, k=1, t=2),
            tableau=((Card(1, 2, True),), (Card(0, 3, True),)),
            stock=(Card(0, 1, False),),
            waste=(),
            foundations=(0, 0),
        )
        features = move_features(state, Move(MoveKind.DRAW))
        self.assertEqual(features[PARAMETER_NAMES.index("draw_playable")], 1.0)

    def test_foundation_support_demand_counts_visible_opposite_card(self):
        state = GameState(
            config=DeckConfig(n=3, k=1, t=2),
            tableau=((Card(0, 2, True),), (Card(1, 1, True),)),
            stock=(),
            waste=(),
            foundations=(1, 0),
        )
        move = Move(MoveKind.TABLEAU_TO_FOUNDATION, source_index=0)
        features = move_features(state, move)
        self.assertEqual(
            features[PARAMETER_NAMES.index("foundation_support_demand")],
            1.0,
        )

    def test_reveal_features_see_the_new_tableau_card(self):
        state = GameState(
            config=DeckConfig(n=3, k=1, t=2),
            tableau=((Card(0, 1, False), Card(1, 2, True)), ()),
            stock=(),
            waste=(),
            foundations=(0, 1),
        )
        move = Move(MoveKind.TABLEAU_TO_FOUNDATION, source_index=0)
        features = move_features(state, move)
        self.assertEqual(
            features[PARAMETER_NAMES.index("revealed_card_low_rank")],
            3.0,
        )
        self.assertEqual(
            features[PARAMETER_NAMES.index("revealed_card_foundation_ready")],
            1.0,
        )

    def test_reveal_features_measure_tableau_access_and_foundation_distance(self):
        state = GameState(
            config=DeckConfig(n=4, k=1, t=2),
            tableau=(
                (Card(0, 3, False), Card(1, 1, True)),
                (Card(1, 4, True),),
            ),
            stock=(),
            waste=(),
            foundations=(0, 0),
        )
        features = move_features(
            state,
            Move(MoveKind.TABLEAU_TO_FOUNDATION, source_index=0),
        )
        self.assertEqual(
            features[PARAMETER_NAMES.index("revealed_card_tableau_moves")],
            1.0,
        )
        self.assertEqual(
            features[PARAMETER_NAMES.index("revealed_card_foundation_distance")],
            2.0,
        )

    def test_draw_features_distinguish_uses_and_buried_waste(self):
        state = GameState(
            config=DeckConfig(n=3, k=1, t=2),
            tableau=((Card(1, 2, True),), ()),
            stock=(Card(0, 1, False),),
            waste=(Card(1, 1, True),),
            foundations=(0, 0),
        )
        features = move_features(state, Move(MoveKind.DRAW))
        self.assertEqual(
            features[PARAMETER_NAMES.index("draw_foundation_ready")],
            1.0,
        )
        self.assertEqual(
            features[PARAMETER_NAMES.index("draw_tableau_moves")],
            1.0,
        )
        self.assertEqual(
            features[PARAMETER_NAMES.index("draw_buries_playable_waste")],
            1.0,
        )

    def test_foundation_features_measure_lag_and_safety_gap(self):
        lagging = GameState(
            config=DeckConfig(n=4, k=1, t=1),
            tableau=((Card(1, 2, True),),),
            stock=(),
            waste=(),
            foundations=(3, 1),
        )
        lagging_features = move_features(
            lagging,
            Move(MoveKind.TABLEAU_TO_FOUNDATION, source_index=0),
        )
        self.assertEqual(
            lagging_features[PARAMETER_NAMES.index("foundation_lag")],
            2.0,
        )
        self.assertEqual(
            lagging_features[PARAMETER_NAMES.index("unsafe_foundation_distance")],
            0.0,
        )

        unsafe = GameState(
            config=DeckConfig(n=4, k=1, t=1),
            tableau=((Card(0, 4, True),),),
            stock=(),
            waste=(),
            foundations=(3, 1),
        )
        unsafe_features = move_features(
            unsafe,
            Move(MoveKind.TABLEAU_TO_FOUNDATION, source_index=0),
        )
        self.assertEqual(
            unsafe_features[PARAMETER_NAMES.index("unsafe_foundation_distance")],
            2.0,
        )

    def test_player_can_finish_an_episode_without_crashing(self):
        config = DeckConfig(n=4, k=1, t=3, max_recycles=1)
        player = FiveParameterPlayer(rng=random.Random(1))
        result = evaluate(player, deck_config=config, games=5, max_steps=100, seed=2)
        self.assertEqual(result.games, 5)
        self.assertGreaterEqual(result.average_progress, 0.0)

    def test_training_updates_parameters(self):
        config = DeckConfig(n=5, k=1, t=3, max_recycles=1)
        player = FiveParameterPlayer(rng=random.Random(4))
        before = tuple(player.parameters)
        trainer = PolicyGradientTrainer(
            player,
            deck_config=config,
            learning_rate=0.01,
            max_steps=100,
            seed=5,
        )
        trainer.train(episodes=10)
        self.assertNotEqual(before, tuple(player.parameters))

    def test_adam_updates_only_selected_parameters(self):
        player = FiveParameterPlayer(parameters=[0.0] * FEATURE_COUNT)
        trainer = PolicyGradientTrainer(
            player,
            learning_rate=0.01,
            optimizer="adam",
            trainable_indices=[3],
        )
        trainer.apply_objective_gradient([1.0] * FEATURE_COUNT)
        self.assertNotEqual(player.parameters[3], 0.0)
        self.assertTrue(
            all(value == 0.0 for i, value in enumerate(player.parameters) if i != 3)
        )

    def test_parameter_store_round_trip_for_config(self):
        config = DeckConfig(n=3, k=2)
        parameters = [float(index) for index in range(FEATURE_COUNT)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            save_parameter_record(config, parameters, path=path)
            self.assertEqual(load_parameters_for_config(config, path), parameters)

    def test_standard_game_has_tuned_parameters(self):
        parameters = load_parameters_for_config(DeckConfig(n=13, k=2, t=7))
        self.assertIsNotNone(parameters)
        self.assertEqual(len(parameters), FEATURE_COUNT)


if __name__ == "__main__":
    unittest.main()
