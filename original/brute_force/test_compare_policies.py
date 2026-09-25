import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from solitaire.player import PARAMETER_NAMES
from brute_force.compare_policies import (
    NORMAL_95_Z,
    load_policies,
    main,
    paired_statistics,
    validate_outcomes,
    validate_policies,
    wilson_interval,
)


class ComparisonStatisticsTests(unittest.TestCase):
    def test_wilson_interval_matches_reference_and_boundary_cases(self):
        lower, upper = wilson_interval(50, 100)
        self.assertAlmostEqual(lower, 0.4038315303659956)
        self.assertAlmostEqual(upper, 0.5961684696340044)
        self.assertAlmostEqual(wilson_interval(0, 100)[0], 0.0)
        self.assertAlmostEqual(wilson_interval(0, 100)[1], 0.03699349820698568)
        self.assertAlmostEqual(wilson_interval(100, 100)[1], 1.0)

    def test_paired_interval_accounts_for_shared_outcomes(self):
        result = paired_statistics(30, 10, 1000)
        self.assertEqual(result["net_wins"], 20)
        self.assertEqual(result["delta_percentage_points"], 2.0)
        expected_se = 100 * math.sqrt((0.04 - 0.02**2) / 1000)
        self.assertAlmostEqual(result["standard_error_percentage_points"], expected_se)
        lower, upper = result["paired_normal_95_ci_percentage_points"]
        self.assertAlmostEqual(lower, 2 - NORMAL_95_Z * expected_se)
        self.assertAlmostEqual(upper, 2 + NORMAL_95_Z * expected_se)
        reversed_result = paired_statistics(10, 30, 1000)
        reversed_lower, reversed_upper = reversed_result["paired_normal_95_ci_percentage_points"]
        self.assertAlmostEqual(reversed_lower, -upper)
        self.assertAlmostEqual(reversed_upper, -lower)

    def test_identical_policies_have_zero_paired_difference(self):
        result = paired_statistics(0, 0, 100)
        self.assertEqual(result["paired_normal_95_ci_percentage_points"], [0.0, 0.0])

    def test_invalid_statistical_counts_are_rejected(self):
        for wins, deals in [(0, 0), (-1, 10), (11, 10), (True, 10), (1.5, 10)]:
            with self.subTest(wins=wins, deals=deals), self.assertRaises(ValueError):
                wilson_interval(wins, deals)
        with self.assertRaises(ValueError):
            paired_statistics(6, 5, 10)


class ComparisonInputTests(unittest.TestCase):
    def test_missing_features_default_to_zero_in_canonical_order(self):
        policies = validate_policies({"partial": {"safe_foundation": 8}, "zero": {}})
        self.assertEqual(len(policies["partial"]), 42)
        self.assertEqual(policies["partial"][PARAMETER_NAMES.index("safe_foundation")], 8.0)
        self.assertEqual(sum(policies["partial"]), 8.0)
        self.assertEqual(policies["zero"], [0.0] * 42)

    def test_full_weight_mapping_is_preserved(self):
        supplied = {name: index / 10 for index, name in enumerate(PARAMETER_NAMES)}
        self.assertEqual(validate_policies({"full": supplied})["full"], list(supplied.values()))

    def test_malformed_weights_and_policy_shapes_are_rejected(self):
        invalid = [
            {}, [], {"": {}}, {"p": []}, {"p": {"typo": 1}},
            {"p": {"safe_foundation": True}}, {"p": {"safe_foundation": "8"}},
            {"p": {"safe_foundation": None}}, {"p": {"safe_foundation": math.nan}},
            {"p": {"safe_foundation": math.inf}}, {"p": {"safe_foundation": 10**1000}},
        ]
        for data in invalid:
            with self.subTest(data=repr(data)[:100]), self.assertRaises(ValueError):
                validate_policies(data)

    def test_duplicate_names_and_invalid_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policies.json"
            for raw in ['{"p": {}, "p": {}}', '{"p": {"safe_foundation": 1, "safe_foundation": 2}}', 'not json']:
                path.write_text(raw, encoding="utf-8")
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    load_policies(path)
            path.write_text(json.dumps({"valid": {}}), encoding="utf-8")
            policies, raw = load_policies(path)
            self.assertIn("valid", policies)
            self.assertEqual(raw, path.read_bytes())

    def test_native_outcome_validation_checks_length_values_and_count(self):
        validate_outcomes(bytes([0, 1, 1]), 3, 2)
        for data, deals, wins in [(bytes([1]), 2, 1), (bytes([2, 0]), 2, 1), (bytes([1, 1]), 2, 1)]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_outcomes(data, deals, wins)


@unittest.skipUnless(os.environ.get("NATIVE_SOLVER_TEST_BINARY"), "set NATIVE_SOLVER_TEST_BINARY for native integration")
class NativeComparisonIntegrationTests(unittest.TestCase):
    def test_reused_outcome_root_preserves_each_runs_evidence(self):
        binary = str(Path(os.environ["NATIVE_SOLVER_TEST_BINARY"]).resolve())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policies = root / "policies.json"
            policies.write_text(json.dumps({
                "baseline": {"safe_foundation": 8, "reveal_hidden": 5, "tableau_build": -2},
                "candidate": {"safe_foundation": 8, "reveal_hidden": 6, "tableau_build": -2},
            }), encoding="utf-8")
            outcomes_root = root / "outcomes"

            def run(seed):
                output = root / f"comparison-{seed}.json"
                self.assertEqual(main([
                    "--policies", str(policies), "--baseline", "baseline",
                    "--binary", binary, "--deals", "6", "--seed", str(seed),
                    "--threads", "1", "--max-steps", "200",
                    "--output", str(output), "--outcomes-dir", str(outcomes_root),
                ]), 0)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(report["complete"])
                self.assertEqual(report["execution_order"], ["baseline", "candidate"])
                return report

            first = run(2026092511)
            saved = {
                Path(policy["outcomes"]["path"]): Path(policy["outcomes"]["path"]).read_bytes()
                for policy in first["policies"].values()
            }
            second = run(2026092512)
            second_paths = {
                Path(policy["outcomes"]["path"])
                for policy in second["policies"].values()
            }
            self.assertTrue(set(saved).isdisjoint(second_paths))
            self.assertNotEqual(first["outcomes_directory"], second["outcomes_directory"])
            for path, data in saved.items():
                self.assertEqual(path.read_bytes(), data)
                self.assertEqual(len(data), 6)


if __name__ == "__main__":
    unittest.main()
