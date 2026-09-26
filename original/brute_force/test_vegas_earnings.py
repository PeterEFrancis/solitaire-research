import contextlib
from fractions import Fraction
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from brute_force.vegas_earnings import (
    POLICIES, VEGAS_CONFIG, analyze_outcomes, build_report, empirical_quantile,
    main, net_dollars,
)


class EarningsMathTests(unittest.TestCase):
    def test_support_has_no_break_even_and_maximum_is_208(self):
        support = [net_dollars(cards) for cards in range(53)]
        self.assertEqual(support[0], -52)
        self.assertEqual(support[10:12], [-2, 3])
        self.assertEqual(support[-1], 208)
        self.assertNotIn(0, support)
        for invalid in (-1, 53, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                net_dollars(invalid)

    def test_distribution_mean_sd_inverse_cdf_and_events(self):
        result = analyze_outcomes(bytes([0, 10, 11, 52]))
        self.assertEqual(result["mean_net_dollars"], 39.25)
        self.assertAlmostEqual(result["sd_net_dollars"], statistics.stdev([-52, -2, 3, 208]))
        self.assertEqual(result["quantiles_net_dollars"], {"q05": -52, "q25": -52, "q50": -2, "q75": 3, "q95": 208})
        self.assertEqual(result["median_net_dollars"], -2)
        self.assertEqual(result["events"]["profit"]["count"], 2)
        self.assertEqual(result["events"]["loss"]["count"], 2)
        self.assertEqual(result["events"]["full_loss"]["count"], 1)
        self.assertEqual(result["events"]["full_win"]["count"], 1)
        self.assertEqual([row["count"] for row in result["bins"]], [1, 0, 1, 1, 0, 0, 1])
        self.assertEqual(len(result["pmf"]), 53)
        self.assertEqual(sum(row["probability"] for row in result["pmf"]), 1)
        self.assertEqual(result["pmf"][10]["cumulative_probability"], 0.5)
        self.assertEqual(result["pmf"][-1]["cumulative_probability"], 1)

    def test_singleton_and_constant_samples_do_not_invent_variance(self):
        single = analyze_outcomes(bytes([0]))
        self.assertIsNone(single["sd_net_dollars"])
        self.assertIsNone(single["mean_net_dollars_95_ci"])
        self.assertEqual(single["events"]["full_loss"]["probability"], 1)
        self.assertLess(single["events"]["full_loss"]["wilson_95_ci"][0], 1)
        constant = analyze_outcomes(bytes([11] * 8))
        self.assertEqual(constant["sd_net_dollars"], 0)
        self.assertEqual(constant["mean_net_dollars_95_ci"], [3, 3])
        for invalid in (b"", bytes([53])):
            with self.assertRaises(ValueError):
                analyze_outcomes(invalid)

    def test_quantiles_skip_empty_support_and_use_exact_ranks(self):
        counts = [0] * 53
        counts[7] = counts[11] = 1
        self.assertEqual(empirical_quantile(counts, Fraction(0)), -17)
        self.assertEqual(empirical_quantile(counts, Fraction(1, 2)), -17)
        self.assertEqual(empirical_quantile(counts, Fraction(3, 4)), 3)
        self.assertEqual(empirical_quantile(counts, Fraction(1)), 3)
        with self.assertRaises(ValueError):
            empirical_quantile(counts, Fraction(11, 10))


class EarningsProvenanceTests(unittest.TestCase):
    def fixture(self, root):
        outcomes = bytes([0, 11, 52])
        frozen_path = root / "variant-study.selected-policies.json"
        frozen_path.write_text(json.dumps({"schema_version": 1, "complete": True,
                                           "variants": {"vegas": {"config": VEGAS_CONFIG}}}))
        evaluations = {}
        for name in POLICIES:
            filename = f"vegas.{name}.foundations.u8.gz"
            (root / filename).write_bytes(gzip.compress(outcomes, mtime=0))
            evaluations[name] = {
                "deals": 3, "foundation_cards": 63, "foundation_cards_squared": 2825,
                "wins": 1, "foundations_file": filename,
                "foundations_sha256": hashlib.sha256(outcomes).hexdigest(),
            }
        evaluations["portfolio"]["attempts_per_deal"] = 6
        confirmation = {"schema_version": 1, "complete": True,
                        "frozen_policies_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                        "variants": {"vegas": {"config": VEGAS_CONFIG, "evaluations": evaluations}}}
        confirmation_path = root / "variant-study.confirmation.json"
        confirmation_path.write_text(json.dumps(confirmation))
        return confirmation_path, confirmation

    def test_report_is_deterministic_and_check_detects_edited_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            first = build_report(results_dir=root)
            self.assertEqual(first, build_report(results_dir=root))
            self.assertEqual(tuple(first["policies"]), POLICIES)
            self.assertEqual(first["policies"]["portfolio"]["policy_type"], "full_information_restart_portfolio")
            self.assertIn("not an ordinary paid-play", first["policies"]["portfolio"]["interpretation"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--results-dir", str(root)]), 0)
                self.assertEqual(main(["--results-dir", str(root), "--check"]), 0)
            output = root / "vegas-earnings.json"
            saved = json.loads(output.read_text())
            saved["policies"]["profit"]["mean_net_dollars"] = 999
            output.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "stale or modified"):
                main(["--results-dir", str(root), "--check"])

    def test_raw_hash_corruption_is_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "vegas.profit.foundations.u8.gz").write_bytes(gzip.compress(bytes([1, 11, 52])))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build_report(results_dir=root)

    def test_output_cannot_overwrite_any_referenced_outcome_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            confirmation_path, confirmation = self.fixture(root)
            wins_path = root / "vegas.profit.wins.u8.gz"
            wins_path.write_bytes(gzip.compress(bytes([0, 0, 1]), mtime=0))
            confirmation["variants"]["vegas"]["evaluations"]["profit"]["wins_file"] = wins_path.name
            confirmation_path.write_text(json.dumps(confirmation))
            for path in (wins_path, root / "vegas.profit.foundations.u8.gz"):
                original = path.read_bytes()
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["--results-dir", str(root), "--output", str(path)])
                self.assertEqual(path.read_bytes(), original)

    def test_length_moments_and_full_win_counts_must_match(self):
        for key, value, message in (("deals", 4, "length mismatch"),
                                    ("foundation_cards", 64, "moment mismatch"),
                                    ("foundation_cards_squared", 2826, "moment mismatch"),
                                    ("wins", 2, "full-win count")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, confirmation = self.fixture(root)
                confirmation["variants"]["vegas"]["evaluations"]["profit"][key] = value
                path.write_text(json.dumps(confirmation))
                with self.assertRaisesRegex(ValueError, message):
                    build_report(results_dir=root)

    def test_changed_frozen_policy_artifact_and_incomplete_study_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            confirmation_path, confirmation = self.fixture(root)
            frozen = root / "variant-study.selected-policies.json"
            frozen.write_bytes(frozen.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "frozen policy checksum"):
                build_report(results_dir=root)
            confirmation["complete"] = False
            confirmation_path.write_text(json.dumps(confirmation))
            with self.assertRaisesRegex(ValueError, "must be complete"):
                build_report(results_dir=root)

    def test_wrong_rules_are_not_presented_as_vegas_earnings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, confirmation = self.fixture(root)
            confirmation["variants"]["vegas"]["config"] = {**VEGAS_CONFIG, "max_recycles": 3}
            path.write_text(json.dumps(confirmation))
            with self.assertRaisesRegex(ValueError, "single-pass"):
                build_report(results_dir=root)


if __name__ == "__main__":
    unittest.main()
