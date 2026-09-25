import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from solitaire import Card, DeckConfig, GameState, Move, MoveKind
from solitaire.player import PARAMETER_NAMES
from solitaire.play_variant import PolicyRun, main, native_policy_moves, play_policy, select_run
from solitaire.variants import config_record, variant_config


class VariantPlaybackTests(unittest.TestCase):
    def test_empty_column_pruning_matches_native_selection(self):
        state = GameState(
            DeckConfig(n=3, k=2, t=4),
            ((Card(0, 3, True),), (Card(3, 1, False), Card(2, 3, True)), (), ()),
            (), (Card(1, 3, True),), (0, 0, 0, 0),
        )
        moves = native_policy_moves(state)
        self.assertIn(Move(MoveKind.WASTE_TO_TABLEAU, dest_index=2), moves)
        self.assertNotIn(Move(MoveKind.WASTE_TO_TABLEAU, dest_index=3), moves)
        self.assertFalse(any(move.kind == MoveKind.TABLEAU_TO_TABLEAU and move.source_index == 0 for move in moves))
        self.assertIn(Move(MoveKind.TABLEAU_TO_TABLEAU, source_index=1, dest_index=2), moves)
        self.assertNotIn(Move(MoveKind.TABLEAU_TO_TABLEAU, source_index=1, dest_index=3), moves)

    def test_trace_replays_to_a_legal_win_and_ties_follow_generator_order(self):
        initial = GameState(DeckConfig(n=1, k=1, t=1), ((Card(0, 1, True),),),
                            (Card(1, 1, False),), (), (0, 0))
        run = play_policy(initial, [0.0] * len(PARAMETER_NAMES), max_steps=10)
        self.assertTrue(run.final_state.is_won())
        self.assertEqual(run.trace[0]["move"]["kind"], MoveKind.DRAW)
        state = initial
        for step in run.trace:
            state = state.apply_move(Move(**step["move"]), validate=True)
        self.assertEqual(state, run.final_state)
        self.assertEqual(run.termination, "won")

    def test_unlimited_empty_pass_cycle_stops_before_budget(self):
        initial = GameState(
            DeckConfig(n=3, k=1, t=2, max_recycles=None),
            ((Card(0, 1, False), Card(0, 2, True)), (Card(1, 1, False), Card(1, 2, True))),
            (Card(0, 3, False), Card(1, 3, False)), (), (0, 0),
        )
        run = play_policy(initial, [0.0] * len(PARAMETER_NAMES), max_steps=1000)
        self.assertEqual(run.termination, "no_unseen_moves")
        self.assertEqual(len(run.trace), 2)

    def test_portfolio_chooses_wins_then_foundation_progress_then_shorter_trace(self):
        state = GameState(DeckConfig(n=1, k=1, t=1), ((),), (), (), (0, 0))
        runs = [PolicyRun(state, [], "no_unseen_moves"),
                PolicyRun(replace(state, foundations=(1, 0)), [{}, {}], "no_unseen_moves"),
                PolicyRun(replace(state, foundations=(1, 1)), [{}, {}, {}], "won"),
                PolicyRun(replace(state, foundations=(1, 1)), [{}], "won")]
        self.assertEqual(select_run(runs), 3)

    def test_cli_loads_frozen_portfolio_and_emits_reproducible_json(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_file = Path(directory) / "selected.json"
            policy_file.write_text(json.dumps({"schema_version": 1, "complete": True, "variants": {
                "vegas": {"config": config_record(variant_config("vegas")), "parameters": {},
                          "policies": {"full": {}}, "portfolio": [{}, {"safe_foundation": 8}]}
            }}))
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(main(["vegas", "--policy", "portfolio", "--policy-file", str(policy_file),
                                       "--seed", "9", "--max-steps", "3", "--format", "json"]), 0)
            report = json.loads(captured.getvalue())
            self.assertEqual(report["attempts"], 2)
            self.assertEqual(report["seed"], 9)
            self.assertIn("native benchmark", report["shuffle"])
            self.assertIn("full-information restarts", report["information"])
            self.assertEqual(report["net_dollars"], 5 * report["foundation_cards"] - 52)
            self.assertLessEqual(report["steps"], 3)
            self.assertEqual(len(report["members"]), 2)


if __name__ == "__main__":
    unittest.main()
