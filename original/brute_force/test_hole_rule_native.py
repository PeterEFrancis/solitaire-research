import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from brute_force.hole_rule_native import SOURCE, build_binary, patched_source, replace_once
from solitaire.player import PARAMETER_NAMES


class HoleRuleBuilderTests(unittest.TestCase):
    def test_transformation_is_guarded_and_does_not_modify_source(self):
        original = SOURCE.read_bytes()
        generated = patched_source(original.decode())
        self.assertEqual(SOURCE.read_bytes(), original)
        self.assertIn("count_holes(before, tableau_count) == 1", generated)
        self.assertIn("candidate_score -= g_hole_penalty", generated)
        with self.assertRaisesRegex(ValueError, "already contains"):
            patched_source(generated)
        with self.assertRaisesRegex(ValueError, "found 0"):
            patched_source(original.decode().replace("            Key choice_key{};", "            Key choice_key;"))
        with self.assertRaisesRegex(ValueError, "found 2"):
            replace_once("xx", "x", "y")

    def test_build_refuses_overwriting_original_source(self):
        with self.assertRaisesRegex(ValueError, "distinct paths"):
            build_binary(SOURCE)


@unittest.skipUnless(os.environ.get("HOLE_RULE_NATIVE_TESTS"), "set HOLE_RULE_NATIVE_TESTS=1 to compile and test the isolated experiment")
class HoleRuleNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="hole-rule-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.binary = cls.directory / "holes"
        cls.original_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        cls.manifest = build_binary(cls.binary)
        cls.baseline = cls.directory / "baseline"
        compiler, flags = cls.manifest["compiler"]["command"], cls.manifest["flags"]
        subprocess.run([*compiler, *flags, str(SOURCE.resolve()), "-o", str(cls.baseline)],
                       capture_output=True, text=True, check=True, timeout=180)
        generated = Path(cls.manifest["generated_source"]["path"]).read_text()
        generated = replace_once(generated, "private:\n", "public:\n")
        generated = replace_once(generated, "int main(int argc, char** argv)", "int native_program_main(int argc, char** argv)")
        harness_source = cls.directory / "harness.cpp"
        harness_source.write_text(generated + r'''
int main() {
    State one;
    one.tableau[0].size = one.tableau[1].size = 1;
    State fill = one;
    fill.tableau[2].size = 1;
    State two = one;
    two.tableau[1].size = 0;
    State two_to_one = two;
    two_to_one.tableau[2].size = 1;
    State swap = fill;
    swap.tableau[0].size = 0;
    State no_hole = fill;
    State revealed_before = one;
    revealed_before.tableau[0].size = 2;
    revealed_before.tableau[0].cards[0] = 0;  // hidden ace
    revealed_before.tableau[0].cards[1] = make_face_up(5);  // red king, n=3
    State revealed_after = revealed_before;
    revealed_after.tableau[0].size = 1;
    revealed_after.tableau[0].cards[0] = make_face_up(0);
    revealed_after.tableau[2].size = 1;
    revealed_after.tableau[2].cards[0] = make_face_up(5);
    State foundation_open = no_hole;
    foundation_open.tableau[0].size = 0;
    ++foundation_open.foundations[0];
    std::cout << consumes_last_hole(one, fill, 3) << " "
              << consumes_last_hole(two, two_to_one, 3) << " "
              << consumes_last_hole(one, swap, 3) << " "
              << consumes_last_hole(no_hole, no_hole, 3) << " "
              << consumes_last_hole(revealed_before, revealed_after, 3) << " "
              << consumes_last_hole(no_hole, foundation_open, 3) << "\n";

    State choice;
    choice.tableau[0].size = 1;
    choice.tableau[0].cards[0] = make_face_up(0);  // black ace
    choice.waste_size = 1;
    choice.waste[0] = make_face_up(5);  // red king
    std::array<double, kModelFeatureCount> weights{};
    weights[0] = 10.0;  // safe foundation
    weights[8] = 100.0; // waste to tableau
    for (double penalty : {0.0, 90.0, 91.0}) {
        g_hole_penalty = penalty;
        g_preserve_last_hole = false;
        Solver solver(3, 2, true, true, 1, weights, -1, 0, false, 3);
        solver.is_solvable(choice);
        std::cout << solver.model_steps << " " << solver.model_foundation_cards << "\n";
    }
    g_hole_penalty = 0.0;
    g_preserve_last_hole = true;
    Solver strict(3, 2, true, true, 1, weights, -1, 0, false, 3);
    strict.is_solvable(choice);
    std::cout << strict.model_steps << " " << strict.model_foundation_cards << "\n";
    std::vector<Candidate> strict_successors, plain_successors;
    strict.generate_successors(choice, strict_successors, false);
    g_preserve_last_hole = false;
    strict.generate_successors(choice, plain_successors, false);
    bool unchanged = strict_successors.size() == plain_successors.size();
    for (size_t i = 0; unchanged && i < strict_successors.size(); ++i) {
        unchanged = strict.position_key(strict_successors[i].state, false) ==
                    strict.position_key(plain_successors[i].state, false) &&
                    strict_successors[i].model_score == plain_successors[i].model_score;
    }
    std::cout << unchanged << "\n";
}
''', encoding="utf-8")
        cls.harness = cls.directory / "harness"
        subprocess.run([*compiler, *flags, str(harness_source), "-o", str(cls.harness)],
                       capture_output=True, text=True, check=True, timeout=180)

    def test_predicate_counts_actual_holes_and_score_ties_keep_generator_order(self):
        result = subprocess.run([str(self.harness)], capture_output=True, text=True, check=True)
        rows = [list(map(int, row.split())) for row in result.stdout.splitlines()]
        self.assertEqual(rows[0], [1, 0, 0, 0, 1, 0])
        self.assertEqual(rows[1], [1, 0])  # unconstrained king fill
        self.assertEqual(rows[2], [1, 0])  # adjusted tie still chooses earlier king move
        self.assertEqual(rows[3], [1, 1])  # larger penalty chooses legal foundation move
        self.assertEqual(rows[4], [1, 1])  # strict filter retains the foundation option
        self.assertEqual(rows[5], [1])     # legal successors and their features unchanged

    def benchmark(self, binary, label, extra=()):
        wins = self.directory / f"{label}.wins.u8"
        foundations = self.directory / f"{label}.foundations.u8"
        source = SOURCE.parent / "results/variant-study.selected-policies.json"
        parameters = json.loads(source.read_text())["variants"]["draw3_unlimited"]["parameters"]
        encoded = ",".join(format(parameters.get(name, 0), ".17g") for name in PARAMETER_NAMES)
        result = subprocess.run([
            str(binary), "--n", "13", "--draw-count", "3", "--max-recycles", "-1",
            "--model-only", "--benchmark-deals", "128", "--benchmark-seed", "2026102701",
            "--threads", "2", "--model-max-steps", "1000", "--model-weights", encoded,
            "--benchmark-outcomes", str(wins), "--benchmark-foundations", str(foundations), *extra,
        ], capture_output=True, text=True, check=True)
        return dict(line.split(" ", 1) for line in result.stdout.splitlines()), wins.read_bytes(), foundations.read_bytes()

    def test_zero_intervention_matches_fresh_native_binary_byte_for_byte(self):
        baseline, wins, foundations = self.benchmark(self.baseline, "baseline")
        for name, extra in (("default", ()), ("zero", ("--hole-penalty", "0"))):
            result, actual_wins, actual_foundations = self.benchmark(self.binary, name, extra)
            self.assertEqual(actual_wins, wins)
            self.assertEqual(actual_foundations, foundations)
            for field in baseline:
                if field not in ("elapsed_seconds", "deals_per_second", "benchmark_outcomes", "benchmark_foundations"):
                    self.assertEqual(result[field], baseline[field], field)

    def test_active_modes_preserve_legal_outcome_format_and_model_only_scope(self):
        for name, extra in (("penalty", ("--hole-penalty", "2.5")), ("strict", ("--preserve-last-hole",))):
            result, wins, foundations = self.benchmark(self.binary, name, extra)
            self.assertEqual(len(wins), 128)
            self.assertTrue(all(win == (cards == 52) for win, cards in zip(wins, foundations)))
            self.assertEqual(result["exact_fallbacks"], "0")
            self.assertEqual(result["preserve_last_hole"], "1" if name == "strict" else "0")
        base = [str(self.binary), "--n", "13", "--model-only", "--benchmark-deals", "1"]
        for extra in (("--hole-penalty", "nan"), ("--hole-penalty", "inf"), ("--hole-penalty", "-1"),
                      ("--hole-penalty", "2junk"), ("--hole-penalty", "1"), ("--preserve-last-hole",)):
            with self.subTest(extra=extra):
                result = subprocess.run([*base, *extra], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)

    def test_manifest_fingerprints_original_generated_binary_and_compiler(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), self.original_digest)
        for field in ("source", "generated_source", "binary", "builder"):
            record = self.manifest[field]
            self.assertEqual(hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest(), record["sha256"])
        self.assertEqual(len(self.manifest["compiler"]["executable_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
