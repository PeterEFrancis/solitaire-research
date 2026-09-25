"""Opt-in native benchmark checks; set NATIVE_SOLVER_TEST_BINARY to a built solver."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


NATIVE_BINARY = os.environ.get("NATIVE_SOLVER_TEST_BINARY")


@unittest.skipUnless(NATIVE_BINARY, "set NATIVE_SOLVER_TEST_BINARY to test native output")
class BenchmarkOutcomesTests(unittest.TestCase):
    def run_benchmark(self, threads, output=None):
        command = [
            str(Path(NATIVE_BINARY).resolve()),
            "--n", "4",
            "--model-only",
            "--model-max-steps", "120",
            "--benchmark-deals", "128",
            "--benchmark-seed", "20260925",
            "--threads", str(threads),
        ]
        if output is not None:
            command.extend(["--benchmark-outcomes", str(output)])
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return dict(line.split(" ", 1) for line in result.stdout.splitlines())

    def test_outcomes_match_aggregate_and_do_not_change_search(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outcomes.bin"
            output.write_bytes(b"old contents" * 100)
            recorded = self.run_benchmark(1, output)
            plain = self.run_benchmark(1)
            outcomes = output.read_bytes()
        self.assertEqual(len(outcomes), 128)
        self.assertLessEqual(set(outcomes), {0, 1})
        self.assertEqual(sum(outcomes), int(recorded["solvable_deals"]))
        self.assertEqual(recorded["benchmark_outcomes"], str(output))
        self.assertEqual(recorded["benchmark_outcomes_bytes"], "128")
        for name, value in plain.items():
            if name not in {"elapsed_seconds", "deals_per_second"}:
                self.assertEqual(recorded[name], value, name)
        self.assertNotIn("benchmark_outcomes", plain)

    def test_outcomes_are_in_deal_order_across_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "serial.bin"
            second = Path(directory) / "parallel.bin"
            self.run_benchmark(1, first)
            self.run_benchmark(3, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_outcomes_reject_nonbenchmark_and_overriding_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outcomes.bin"
            modes = [
                [],
                ["--benchmark-deals", "1", "--collapse-stock-order"],
                ["--benchmark-deals", "1", "--check-stock-orders", "1"],
                ["--benchmark-deals", "1", "--verify-stock-order", "unused"],
            ]
            for mode in modes:
                with self.subTest(mode=mode):
                    result = subprocess.run(
                        [str(Path(NATIVE_BINARY).resolve()), "--n", "4",
                         "--benchmark-outcomes", str(output), *mode],
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("--benchmark-outcomes requires", result.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
