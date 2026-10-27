"""Small saved-artifact corruption tests; no native executable is run."""

import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from brute_force.hole_rule_native import patched_source
from brute_force.hole_rule_study import comparison
from brute_force.variant_study import statistics
from brute_force.verify_hole_rule_study import (
    CONFIG, FAMILIES, PENALTIES, PRODUCTION_SEEDS, RESULTS, RULES,
    SOURCE_DIRECTORY, VerificationError, main, sha256, verify_study,
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def aggregate(cards, *, rule="baseline", penalty=0.0):
    row = statistics(dict(deals=len(cards), wins=cards.count(52), foundation_cards=sum(cards),
                          foundation_cards_squared=sum(x*x for x in cards), steps=500, cutoffs=0))
    row.pop("mean_net_dollars")
    row.pop("mean_net_dollars_95_ci")
    return dict(rule=rule, penalty=penalty, **row)


class SavedHoleStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = "fixture"
        self.paths = {name: self.root / f"fixture.{name}.json"
                      for name in ("protocol", "screen", "frozen", "confirmation")}
        self.binary = self.root / "experimental.bin"
        self.reference = self.root / "reference.bin"
        self.binary.write_bytes(b"experimental fixture binary; never executed")
        self.reference.write_bytes(b"reference fixture binary; never executed")
        build = {
            "schema_version": 1, "experiment": "last-empty-tableau-column policy intervention",
            "predicate": "empty_columns(before) == 1 and empty_columns(after) == 0",
            "options": {"hole_penalty_default": 0.0, "preserve_last_hole_default": False},
            "source": {"path": "native_solver.cpp", "sha256": sha256(SOURCE_DIRECTORY / "native_solver.cpp")},
            "builder": {"path": "hole_rule_native.py", "sha256": sha256(SOURCE_DIRECTORY / "hole_rule_native.py")},
            "generated_source": {"path": "generated.cpp", "sha256": hashlib.sha256(
                patched_source((SOURCE_DIRECTORY / "native_solver.cpp").read_text()).encode()).hexdigest()},
            "binary": {"path": str(self.binary), "sha256": sha256(self.binary)},
            "compiler": {"command": ["fixture-cxx"], "version": "fixture compiler",
                         "executable_sha256": "1" * 64,
                         "version_sha256": hashlib.sha256(b"fixture compiler").hexdigest()},
            "flags": ["-std=c++17"],
            "compile_command": ["fixture-cxx", "-std=c++17", "generated.cpp", "-o", str(self.binary)],
        }
        source = RESULTS / "variant-study.selected-policies.json"
        parameters = json.loads(source.read_text())["variants"]["draw3_unlimited"]["policies"]
        self.protocol = {
            "schema_version": 1, "variant": "draw3_unlimited", "config": CONFIG,
            "rule_definition": "A consuming move has exactly one empty tableau column before it and zero after it.",
            "soft_rule": "Subtract the penalty from the unchanged policy score only for consuming moves.",
            "strict_rule": "Exclude consuming moves; if no eligible unseen move remains, stop without backtracking.",
            "scope": "Preserve an existing last hole. Do not reward creating the first hole or penalize 2-to-1 empties or hole relocation.",
            "selection": "Per family, select the best strictly positive penalty on screen only by wins, foundation sum, fewer steps, then smaller penalty. Report baseline and strict separately. Never reselect from confirmation.",
            "families": list(FAMILIES), "penalties": list(PENALTIES),
            "seeds": {key: value + 10000 for key, value in PRODUCTION_SEEDS.items()},
            "options": dict(screen_deals=6, test_deals=6, max_steps=1000, threads=1, jobs=1, seed_offset=10000),
            "parameters": {family: parameters[family] for family in FAMILIES},
            "policy_source": source.name, "policy_source_sha256": sha256(source),
            "native_source_sha256": build["source"]["sha256"],
            "builder_source_sha256": build["builder"]["sha256"],
            "study_source_sha256": sha256(SOURCE_DIRECTORY / "hole_rule_study.py"),
            "reference_binary_sha256": sha256(self.reference), "build": build,
        }
        self.cards = {
            "baseline": bytes([52, 0, 11, 52, 25, 51]),
            "selected_soft": bytes([52, 52, 10, 50, 25, 51]),
            "strict": bytes([0, 0, 11, 52, 25, 51]),
        }
        self.screen = {
            "schema_version": 1, "complete": True,
            "baseline_parity": {family: dict(deals=1000, exact_match=True, wins_sha256="2"*64,
                                               foundations_sha256="3"*64) for family in FAMILIES},
            "families": {family: [aggregate(self.cards["baseline"], rule=rule, penalty=penalty)
                                   for rule, penalty in [("baseline", 0.0),
                                                         *(("soft", p) for p in PENALTIES), ("strict", 0.0)]]
                         for family in FAMILIES},
        }
        self.frozen = {"schema_version": 1, "complete": True,
                       "selected_penalties": {family: 0.5 for family in FAMILIES},
                       "confirmation_rules": list(RULES)}
        self.confirmation = {"schema_version": 1, "complete": True, "variant": "draw3_unlimited", "deals": 6,
                             "seed": self.protocol["seeds"]["confirmation"], "families": {}}
        (self.root / "fixture-outcomes").mkdir()
        for family in FAMILIES:
            rows = {}
            for name in RULES:
                cards = self.cards[name]
                wins = bytes(int(c == 52) for c in cards)
                row = aggregate(cards, rule="soft" if name == "selected_soft" else name,
                                penalty=0.5 if name == "selected_soft" else 0.0)
                for kind, data in (("wins", wins), ("foundations", cards)):
                    filename = f"fixture-outcomes/{family}.{name}.{kind}.u8.gz"
                    (self.root / filename).write_bytes(gzip.compress(data, mtime=0))
                    row[kind + "_file"] = filename
                    row[kind + "_sha256"] = hashlib.sha256(data).hexdigest()
                if name != "baseline":
                    row["paired_vs_baseline"] = comparison(wins, bytes(int(c == 52) for c in self.cards["baseline"]))
                rows[name] = row
            self.confirmation["families"][family] = rows
        self.relink()

    def relink(self):
        save(self.paths["protocol"], self.protocol)
        self.screen["protocol_sha256"] = sha256(self.paths["protocol"])
        save(self.paths["screen"], self.screen)
        self.frozen.update(protocol_sha256=sha256(self.paths["protocol"]), screen_sha256=sha256(self.paths["screen"]))
        save(self.paths["frozen"], self.frozen)
        self.confirmation.update(protocol_sha256=sha256(self.paths["protocol"]), frozen_sha256=sha256(self.paths["frozen"]))
        save(self.paths["confirmation"], self.confirmation)

    def verify(self, **kwargs):
        return verify_study(self.prefix, results_dir=self.root, **kwargs)

    def test_valid_fixture_portable_and_explicit_binary_checks(self):
        self.assertEqual(len(self.verify()), 10)
        self.assertIn("binary_checked=false", self.verify()[-1])
        self.assertIn("reference_binary_checked=true", self.verify(binary=self.binary, reference_binary=self.reference)[-1])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--prefix", "fixture", "--results-dir", str(self.root)]), 0)
        self.binary.write_bytes(b"changed binary")
        self.assertEqual(len(self.verify()), 10)
        with self.assertRaisesRegex(VerificationError, "experimental binary hash"):
            self.verify(binary=self.binary)

    def test_incomplete_and_broken_lineage_rejected(self):
        self.confirmation["complete"] = False
        self.relink()
        with self.assertRaisesRegex(VerificationError, "incomplete"):
            self.verify()
        self.confirmation["complete"] = True
        self.relink()
        self.paths["screen"].write_text(self.paths["screen"].read_text() + "\n")
        with self.assertRaisesRegex(VerificationError, "screen_sha256"):
            self.verify()

    def test_selection_recomputes_smaller_penalty_tie_break(self):
        self.frozen["selected_penalties"]["full"] = 1.0
        self.relink()
        with self.assertRaisesRegex(VerificationError, "selected penalty"):
            self.verify()

    def test_selection_recomputes_foundations_and_steps_ties(self):
        rows = self.screen["families"]["full"]
        rows[2] = aggregate(bytes([52, 1, 11, 52, 25, 51]), rule="soft", penalty=1.0)
        self.relink()
        with self.assertRaisesRegex(VerificationError, "selected penalty"):
            self.verify()
        rows[2] = aggregate(self.cards["baseline"], rule="soft", penalty=1.0)
        rows[2].update(steps=499, mean_steps=499/6)
        self.relink()
        with self.assertRaisesRegex(VerificationError, "selected penalty"):
            self.verify()

    def test_raw_corruption_and_hashed_wrong_length_rejected(self):
        row = self.confirmation["families"]["full"]["baseline"]
        path = self.root / row["wins_file"]
        path.write_bytes(gzip.compress(b"\0" * 6, mtime=0))
        with self.assertRaisesRegex(VerificationError, "SHA-256 mismatch"):
            self.verify()
        path.write_bytes(gzip.compress(b"\0" * 5, mtime=0))
        row["wins_sha256"] = hashlib.sha256(b"\0" * 5).hexdigest()
        self.relink()
        with self.assertRaisesRegex(VerificationError, "expected 6 bytes"):
            self.verify()

    def test_win_bytes_must_agree_with_52_foundations(self):
        row = self.confirmation["families"]["full"]["baseline"]
        data = bytes([0, 1, 0, 1, 0, 0])  # Same number of wins, different deal indices.
        (self.root / row["wins_file"]).write_bytes(gzip.compress(data, mtime=0))
        row["wins_sha256"] = hashlib.sha256(data).hexdigest()
        self.relink()
        with self.assertRaisesRegex(VerificationError, "52 cards iff win"):
            self.verify()

    def test_both_paired_intervals_are_recomputed(self):
        paired = self.confirmation["families"]["full"]["selected_soft"]["paired_vs_baseline"]
        for key in ("paired_normal_95_ci_percentage_points", "bonferroni_familywise_95_ci_percentage_points"):
            old = paired[key][:]
            paired[key][0] += 0.1
            self.relink()
            with self.subTest(key=key), self.assertRaisesRegex(VerificationError, key):
                self.verify()
            paired[key] = old

    def test_generated_source_and_seed_offset_are_checked(self):
        original = self.protocol["build"]["generated_source"]["sha256"]
        self.protocol["build"]["generated_source"]["sha256"] = "0" * 64
        self.relink()
        with self.assertRaisesRegex(VerificationError, "generated native source hash"):
            self.verify()
        self.protocol["build"]["generated_source"]["sha256"] = original
        self.protocol["seeds"]["screen"] += 1
        self.relink()
        with self.assertRaisesRegex(VerificationError, "phase seed offset"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
