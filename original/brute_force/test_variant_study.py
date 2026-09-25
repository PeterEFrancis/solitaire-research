import contextlib
import gzip
import hashlib
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from solitaire.player import PARAMETER_NAMES, hidden_information_features, waste_memory_features
from solitaire.variants import config_record, variant_config
from brute_force import variant_study as study


class VariantStudyStatisticsTests(unittest.TestCase):
    def test_foundation_and_money_intervals_use_sample_variance(self):
        # Observations 0, 26, 52 have sample variance 676 and mean 26.
        result = study.statistics({
            "deals": 3, "wins": 1, "foundation_cards": 78,
            "foundation_cards_squared": 3380, "steps": 30,
        })
        margin = 1.96 * math.sqrt(676 / 3)
        self.assertEqual(result["mean_foundation_cards"], 26)
        self.assertEqual(result["mean_net_dollars"], 78)
        self.assertAlmostEqual(result["mean_foundation_cards_95_ci"][0], 26 - margin)
        self.assertAlmostEqual(result["mean_net_dollars_95_ci"][1], 78 + 5 * margin)

    def test_profit_selection_can_prefer_more_foundations_over_more_wins(self):
        frequent_partial = {"foundation_cards": 90, "wins": 0, "steps": 100}
        one_win = {"foundation_cards": 52, "wins": 1, "steps": 100}
        self.assertGreater(study.objective(frequent_partial, profit=True), study.objective(one_win, profit=True))
        self.assertLess(study.objective(frequent_partial), study.objective(one_win))

    def test_driver_profiles_match_loadable_rules_and_unlimited_normalization(self):
        weights = tuple(float(index + 1) for index in range(len(PARAMETER_NAMES)))
        pressure = {PARAMETER_NAMES.index(name) for name in ("recycle_pressure", "waste_play_recycle_pressure")}
        for profile in study.PROFILES:
            self.assertEqual(study.configuration(profile), config_record(variant_config(profile)))
            normalized = study.normalize(weights, profile)
            for index, value in enumerate(normalized):
                expected = 0 if study.PROFILES[profile][1] is None and index in pressure else weights[index]
                self.assertEqual(value, expected)


class VariantStudyProtocolTests(unittest.TestCase):
    def test_training_freezes_portfolio_without_accessing_test_seed(self):
        args = SimpleNamespace(binary=Path("unused"), train_deals=5, validation_deals=7,
                               threads=1, max_steps=40, mutations=0, finalists=2)
        full = tuple(1.0 for _ in PARAMETER_NAMES)
        simple = tuple(1.0 if name in study.SIMPLE else 0.0 for name in PARAMETER_NAMES)
        visible_names = [name for name in PARAMETER_NAMES if name not in (
            hidden_information_features(variant_config("draw3_unlimited"))
            | waste_memory_features(variant_config("draw3_unlimited"))
        )]

        def evaluate(binary, profile, weights, deals, seed, threads, max_steps):
            wins = int(abs(sum(weights))) % (deals + 1)
            row = {"deals": deals, "wins": wins, "foundation_cards": 52*wins + 10*(deals-wins),
                   "foundation_cards_squared": 52**2*wins + 100*(deals-wins), "steps": 10*deals, "cutoffs": 0}
            return row, b"", b""

        with patch.object(study, "evaluate", side_effect=evaluate) as evaluator, contextlib.redirect_stdout(io.StringIO()):
            training, frozen = study.train_profile(args, "draw3_unlimited", full, simple, visible_names)
        seeds = {call.args[4] for call in evaluator.call_args_list}
        self.assertEqual(seeds, {study.TRAIN_SEED, study.VALIDATION_SEED})
        self.assertNotIn(study.TEST_SEED, seeds)
        self.assertGreater(training["training_evaluations"], 0)
        self.assertLessEqual(len(frozen["portfolio"]), 6)
        self.assertIn(frozen["parameters"], frozen["portfolio"])
        for family, allowed in (("visible", visible_names), ("simple_eight", study.SIMPLE)):
            self.assertTrue(all(weight == 0 or name in allowed for name, weight in frozen["policies"][family].items()))

    def test_confirmation_reuses_fixed_policies_and_aggregates_restart_outcomes(self):
        args = SimpleNamespace(binary=Path("unused"), test_deals=3, threads=1, max_steps=100)
        def weights(value):
            return (float(value),) + (0.0,) * (len(PARAMETER_NAMES)-1)
        full, baseline, simple = weights(2), weights(0), weights(1)
        frozen = {"policies": {"full": study.named(full), "visible": study.named(baseline),
                               "simple_eight": study.named(simple)},
                  "portfolio": [study.named(baseline), study.named(full)]}
        fixtures = {
            0: (bytes([1, 0, 0]), bytes([52, 10, 20])),
            1: (bytes([0, 1, 0]), bytes([12, 52, 25])),
            2: (bytes([0, 0, 1]), bytes([8, 35, 52])),
        }
        def evaluate(binary, profile, selected, deals, seed, threads, max_steps, *, record):
            self.assertTrue(record)
            self.assertEqual(seed, study.TEST_SEED)
            wins, foundations = fixtures[int(selected[0])]
            return {"deals": deals, "wins": sum(wins), "foundation_cards": sum(foundations),
                    "foundation_cards_squared": sum(x*x for x in foundations), "steps": 100, "cutoffs": 0}, wins, foundations

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outcomes"
            output.mkdir()
            with patch.object(study, "RESULTS", root), patch.object(study, "evaluate", side_effect=evaluate) as evaluator, contextlib.redirect_stdout(io.StringIO()):
                report = study.confirm_profile(args, "vegas", frozen, baseline, simple, output)
            self.assertEqual(evaluator.call_count, 3)
            bank = report["evaluations"]["portfolio"]
            self.assertEqual(bank["attempts_per_deal"], 2)
            self.assertEqual(bank["wins"], 2)
            self.assertEqual(bank["foundation_cards"], 139)
            raw = gzip.decompress((root / bank["foundations_file"]).read_bytes())
            self.assertEqual(raw, bytes([52, 35, 52]))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), bank["foundations_sha256"])
            self.assertEqual(bank["paired_vs_stage8"]["net_wins"], 1)


if __name__ == "__main__":
    unittest.main()
