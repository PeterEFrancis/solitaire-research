from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from solitaire import Card, DeckConfig, GameState, Move, MoveKind
from solitaire.parameter_store import load_parameters_for_config, save_parameter_record
from solitaire.player import (
    FEATURE_COUNT, PARAMETER_NAMES, hidden_information_features, move_features,
    waste_memory_features,
)
from solitaire.train import build_parser, moves_to_unseen_positions
from solitaire.variants import config_record, load_variant_policy, variant_config


class VariantRuleTests(unittest.TestCase):
    def state(self, config=None, *, stock=(), waste=(), tableau=((),), foundations=None):
        config = config or DeckConfig(n=4, k=2, t=len(tableau))
        return GameState(config, tableau, stock, waste, foundations or (0,) * config.suits)

    def test_draw_three_uses_top_of_packet_then_short_final_packet(self):
        config = DeckConfig(n=5, k=1, t=1, draw_count=3)
        state = self.state(config, stock=tuple(Card(0, rank, False) for rank in (5, 4, 3, 2, 1)))
        first = state.apply_move(Move(MoveKind.DRAW), validate=True)
        self.assertEqual([card.rank for card in first.waste], [1, 2, 3])
        self.assertEqual(first.waste[-1].rank, 3)
        second = first.apply_move(Move(MoveKind.DRAW), validate=True)
        self.assertEqual([card.rank for card in second.waste], [1, 2, 3, 4, 5])
        self.assertEqual(second.stock, ())
        recycled = second.apply_move(Move(MoveKind.RECYCLE), validate=True)
        self.assertEqual(recycled.stock, state.stock)
        self.assertEqual(recycled.recycles_used, 1)

    def test_unlimited_recycle_returns_to_same_position_and_is_suppressed(self):
        config = DeckConfig(n=4, k=1, t=1, draw_count=3, max_recycles=None)
        state = self.state(config, stock=tuple(Card(0, rank, False) for rank in (4, 3, 2)))
        drawn = state.apply_move(Move(MoveKind.DRAW))
        recycled = drawn.apply_move(Move(MoveKind.RECYCLE))
        self.assertEqual(recycled.recycles_used, 0)
        self.assertEqual(recycled.position_key(), state.position_key())
        self.assertNotIn(Move(MoveKind.RECYCLE), moves_to_unseen_positions(drawn, {state.position_key()}))
        self.assertEqual(state.position_key(), replace(state, recycles_used=999).position_key())

    def test_finite_pass_counts_and_rule_variants_have_distinct_position_keys(self):
        state = self.state(stock=(Card(0, 2, False),))
        self.assertNotEqual(state.position_key(), replace(state, recycles_used=1).position_key())
        for config in [replace(state.config, max_recycles=None), replace(state.config, draw_count=3), replace(state.config, allow_tableau_stack_splitting=False)]:
            self.assertNotEqual(state.position_key(), replace(state, config=config).position_key())

    def test_no_splitting_allows_whole_face_up_run_and_top_foundation_move(self):
        tableau = (
            (Card(3, 1, False), Card(0, 3, True), Card(1, 2, True)),
            (Card(1, 4, True),),
            (Card(2, 3, True),),
        )
        state = self.state(tableau=tableau, foundations=(0, 1, 0, 0))
        whole = Move(MoveKind.TABLEAU_TO_TABLEAU, source_index=0, dest_index=1, count=2)
        suffix = Move(MoveKind.TABLEAU_TO_TABLEAU, source_index=0, dest_index=2)
        foundation = Move(MoveKind.TABLEAU_TO_FOUNDATION, source_index=0)
        self.assertIn(suffix, state.legal_moves())
        restricted = replace(state, config=replace(state.config, allow_tableau_stack_splitting=False))
        self.assertIn(whole, restricted.legal_moves())
        self.assertIn(foundation, restricted.legal_moves())
        self.assertNotIn(suffix, restricted.legal_moves())
        moved = restricted.apply_move(whole, validate=True)
        self.assertTrue(moved.tableau[0][-1].face_up)

    def test_vegas_has_exactly_one_stock_pass(self):
        state = self.state(variant_config("vegas"), waste=(Card(0, 2, True),))
        self.assertNotIn(Move(MoveKind.RECYCLE), state.legal_moves())
        self.assertFalse(state.config.allow_tableau_stack_splitting)

    def test_unlimited_pressure_features_are_zero_even_for_legacy_counter(self):
        state = self.state(waste=(Card(0, 1, True),))
        state = replace(state, recycles_used=2)
        recycle = Move(MoveKind.RECYCLE)
        foundation = Move(MoveKind.WASTE_TO_FOUNDATION)
        recycle_index = PARAMETER_NAMES.index("recycle_pressure")
        waste_index = PARAMETER_NAMES.index("waste_play_recycle_pressure")
        self.assertEqual(move_features(state, recycle)[recycle_index], 3.0)
        self.assertEqual(move_features(state, foundation)[waste_index], 2.0)
        unlimited = replace(state, config=replace(state.config, max_recycles=None))
        self.assertEqual(move_features(unlimited, recycle)[recycle_index], 0.0)
        self.assertEqual(move_features(unlimited, foundation)[waste_index], 0.0)

    def test_draw_three_waste_lookahead_requires_hidden_information(self):
        names = {"waste_unlocks_playable", "waste_unlocks_foundation_ready", "waste_unlocks_tableau_moves"}
        self.assertEqual(waste_memory_features(DeckConfig()), names)
        self.assertTrue(names.isdisjoint(hidden_information_features(DeckConfig())))
        self.assertTrue(names <= hidden_information_features(DeckConfig(draw_count=3)))
        self.assertEqual(waste_memory_features(DeckConfig(draw_count=3)), frozenset())

    def test_cli_accepts_unlimited_sentinel_and_no_splitting(self):
        args = build_parser().parse_args(["--max-recycles", "-1", "--no-tableau-splitting"])
        self.assertEqual(args.max_recycles, -1)
        self.assertTrue(args.no_tableau_splitting)


class VariantParameterTests(unittest.TestCase):
    def test_legacy_records_only_match_historical_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            path.write_text(json.dumps({"records": [{"n": 13, "k": 2, "t": 7, "parameters": [1.0] * FEATURE_COUNT}]}))
            self.assertIsNotNone(load_parameters_for_config(variant_config("draw1_limited"), path))
            for name in ("draw1_unlimited", "draw3_limited", "draw3_unlimited", "vegas"):
                with self.subTest(name=name):
                    self.assertIsNone(load_parameters_for_config(variant_config(name), path))

    def test_variant_records_coexist_and_unlimited_survives_json_round_trip(self):
        names = ("draw1_limited", "draw1_unlimited", "draw3_limited", "draw3_unlimited", "vegas")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            for index, name in enumerate(names):
                save_parameter_record(variant_config(name), [float(index)] * FEATURE_COUNT, path=path)
            self.assertEqual(len(json.loads(path.read_text())["records"]), len(names))
            for index, name in enumerate(names):
                self.assertEqual(load_parameters_for_config(variant_config(name), path), [float(index)] * FEATURE_COUNT)

    def test_selected_policy_loader_returns_named_profile_and_rejects_absent_or_wrong_rules(self):
        config = variant_config("draw3_unlimited")
        record = {
            "config": config_record(config), "parameters": {"safe_foundation": 10},
            "policies": {"full": {"safe_foundation": 10}, "visible": {"reveal_hidden": 5}},
        }
        data = {"schema_version": 1, "variants": {"draw3_unlimited": record}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.json"
            path.write_text(json.dumps(data))
            loaded_config, weights = load_variant_policy("draw3_unlimited", policy="visible", path=path)
            self.assertEqual(loaded_config, config)
            self.assertEqual(weights[PARAMETER_NAMES.index("reveal_hidden")], 5.0)
            self.assertEqual(sum(weights), 5.0)
            with self.assertRaisesRegex(ValueError, "no 'profit'"):
                load_variant_policy("draw3_unlimited", policy="profit", path=path)
            with self.assertRaisesRegex(ValueError, "no selected policy"):
                load_variant_policy("vegas", path=path)
            record["config"]["max_recycles"] = 3
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "stored rules"):
                load_variant_policy("draw3_unlimited", path=path)

    def test_selected_policy_loader_rejects_unknown_nonfinite_and_conflicting_weights(self):
        record = {"config": config_record(variant_config("vegas")), "parameters": {}, "policies": {"full": {}}}
        data = {"schema_version": 1, "variants": {"vegas": record}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.json"
            for parameters in ({"unknown": 1}, {"safe_foundation": float("nan")}, {"safe_foundation": True}, {"safe_foundation": 8}):
                record["parameters"] = parameters
                path.write_text(json.dumps(data))
                with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                    load_variant_policy("vegas", path=path)


if __name__ == "__main__":
    unittest.main()
