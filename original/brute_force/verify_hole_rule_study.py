"""Verify saved last-hole experiments independently, without simulating games.

Run from original/: python3 -m brute_force.verify_hole_rule_study
Source hashes, frozen selections, indexed outcomes, and statistics are checked.
Binary hashes are checked only with explicit paths, permitting portable CI.
Screen aggregates and the recorded parity gate cannot be replayed from saved
bytes; this checks their consistency, not their execution or shuffle quality.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any
import zlib

from .verify_variant_study import (
    SIMPLE_FEATURES, VISIBLE_FEATURES, VerificationError, compare, expected_paired,
    expected_statistics, integer, read_json, read_outcome, require, sha256,
    verify_weights,
)


SOURCE_DIRECTORY = Path(__file__).resolve().parent
RESULTS = SOURCE_DIRECTORY / "results"
FAMILIES = ("full", "visible", "simple_eight")
PENALTIES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
RULES = ("baseline", "selected_soft", "strict")
CONFIG = dict(n=13, k=2, t=7, draw_count=3, max_recycles=None,
              allow_tableau_stack_splitting=True)
PRODUCTION_SEEDS = dict(parity=2026102700, screen=2026102701, confirmation=2026102799)
Z_95 = NormalDist().inv_cdf(0.975)
Z_FAMILY = NormalDist().inv_cdf(1 - 0.05 / 12)


def digest(value: Any, label: str) -> None:
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"{label}: invalid SHA-256")


def exact_keys(value: Any, keys, label: str) -> None:
    require(isinstance(value, dict) and set(value) == set(keys), f"{label}: unexpected keys")


def verify_aggregate(row: dict[str, Any], deals: int, max_steps: int, label: str) -> None:
    """Recalculate screen statistics from integer moments, not driver helpers."""
    compare(row["deals"], deals, label + ".deals")
    for key, upper in (("wins", deals), ("foundation_cards", 52 * deals),
                       ("foundation_cards_squared", 52**2 * deals),
                       ("steps", deals * max_steps), ("cutoffs", deals - row["wins"])):
        integer(row[key], 0, upper, label + "." + key)
    wins, total, squares = (row[k] for k in ("wins", "foundation_cards", "foundation_cards_squared"))
    require(52 * wins <= total <= 52 * wins + 51 * (deals - wins),
            label + ": foundation sum incompatible with win count")
    require(52**2 * wins <= squares <= 52**2 * wins + 51**2 * (deals - wins),
            label + ": squared foundation sum incompatible with win count")
    require(total <= squares <= 52 * total and (squares - total) % 2 == 0,
            label + ": invalid integer foundation moments")
    centered_numerator = deals * squares - total**2
    require(centered_numerator >= 0, label + ": negative foundation variance")
    require(row["cutoffs"] * max_steps <= row["steps"], label + ": cutoffs exceed executed steps")
    rate = wins / deals
    denominator = 1 + Z_95**2 / deals
    center = (rate + Z_95**2 / (2 * deals)) / denominator
    radius = Z_95 * math.sqrt(rate * (1 - rate) / deals + Z_95**2 / (4 * deals**2)) / denominator
    mean = total / deals
    margin = 1.96 * math.sqrt(centered_numerator / (deals**2 * max(1, deals - 1)))
    compare(row, {
        "win_rate": rate,
        "win_rate_wilson_95_ci": [max(0.0, center - radius), min(1.0, center + radius)],
        "mean_foundation_cards": mean,
        "mean_foundation_cards_95_ci": [mean - margin, mean + margin],
        "mean_steps": row["steps"] / deals,
    }, label)
    require("mean_net_dollars" not in row and "mean_net_dollars_95_ci" not in row,
            label + ": Vegas monetary summaries do not apply to this experiment")


def verify_sources(protocol: dict[str, Any], *, binary: Path | None,
                   reference_binary: Path | None) -> None:
    historical = read_json(RESULTS / "variant-study.build.json")
    compare(protocol["native_source_sha256"], historical["source_sha256"],
            "native source changed from historical variant study")
    for filename, key in (("native_solver.cpp", "native_source_sha256"),
                          ("hole_rule_study.py", "study_source_sha256"),
                          ("hole_rule_native.py", "builder_source_sha256")):
        compare(sha256(SOURCE_DIRECTORY / filename), protocol[key], filename + " source hash")
    compare(protocol["policy_source"], "variant-study.selected-policies.json", "policy_source")
    policy_path = RESULTS / protocol["policy_source"]
    compare(sha256(policy_path), protocol["policy_source_sha256"], "frozen policy source hash")
    policy_record = read_json(policy_path)["variants"]["draw3_unlimited"]
    compare(policy_record["config"], CONFIG, "original policy config")
    exact_keys(protocol["parameters"], FAMILIES, "protocol.parameters")
    for family in FAMILIES:
        weights = protocol["parameters"][family]
        allowed = VISIBLE_FEATURES if family == "visible" else SIMPLE_FEATURES if family == "simple_eight" else None
        verify_weights(weights, "draw3_unlimited", family, allowed)
        compare(weights, policy_record["policies"][family], family + " unchanged parameters")

    build = protocol["build"]
    compare(build, {
        "schema_version": 1,
        "experiment": "last-empty-tableau-column policy intervention",
        "predicate": "empty_columns(before) == 1 and empty_columns(after) == 0",
        "options": {"hole_penalty_default": 0.0, "preserve_last_hole_default": False},
        "source": {"sha256": protocol["native_source_sha256"]},
        "builder": {"sha256": protocol["builder_source_sha256"]},
    }, "build")
    # Reconstruct only the generated text, never compile or run it. Its hashed
    # builder and unchanged original source uniquely determine this output.
    from .hole_rule_native import patched_source
    generated = patched_source((SOURCE_DIRECTORY / "native_solver.cpp").read_text())
    compare(hashlib.sha256(generated.encode()).hexdigest(), build["generated_source"]["sha256"],
            "generated native source hash")
    for name in ("binary", "generated_source", "source", "builder"):
        digest(build[name]["sha256"], "build." + name)
        require(isinstance(build[name]["path"], str) and build[name]["path"], "missing build source path")
    compiler = build["compiler"]
    digest(compiler["executable_sha256"], "compiler executable")
    compare(hashlib.sha256(compiler["version"].encode()).hexdigest(), compiler["version_sha256"],
            "compiler version hash")
    require(isinstance(compiler["command"], list) and compiler["command"]
            and all(isinstance(v, str) for v in compiler["command"]), "invalid compiler command")
    require(isinstance(build["flags"], list) and all(isinstance(v, str) for v in build["flags"]),
            "invalid build flags")
    compare(build["compile_command"], [*compiler["command"], *build["flags"],
                                      build["generated_source"]["path"], "-o", build["binary"]["path"]],
            "recorded compile command")
    digest(protocol["reference_binary_sha256"], "reference binary")
    if binary is not None:
        compare(sha256(Path(binary)), build["binary"]["sha256"], "experimental binary hash")
    if reference_binary is not None:
        compare(sha256(Path(reference_binary)), protocol["reference_binary_sha256"], "reference binary hash")


def verify_study(prefix: str = "hole-preservation", *, results_dir: Path = RESULTS,
                 binary: Path | None = None, reference_binary: Path | None = None) -> list[str]:
    require(isinstance(prefix, str) and prefix and Path(prefix).name == prefix and "." not in prefix,
            "prefix must be a filename stem without dots")
    directory = Path(results_dir).resolve()
    paths = {name: directory / f"{prefix}.{name}.json"
             for name in ("protocol", "screen", "frozen", "confirmation")}
    documents = {name: read_json(path) for name, path in paths.items()}
    protocol, screen, frozen, confirmation = (documents[name] for name in paths)
    for name, document in documents.items():
        compare(document["schema_version"], 1, name + ".schema_version")
        if name != "protocol":
            require(document.get("complete") is True, name + ": study is incomplete")
            compare(document["protocol_sha256"], sha256(paths["protocol"]), name + ".protocol_sha256")
    compare(frozen["screen_sha256"], sha256(paths["screen"]), "frozen.screen_sha256")
    compare(confirmation["frozen_sha256"], sha256(paths["frozen"]), "confirmation.frozen_sha256")
    compare(protocol, {
        "variant": "draw3_unlimited", "config": CONFIG,
        "rule_definition": "A consuming move has exactly one empty tableau column before it and zero after it.",
        "soft_rule": "Subtract the penalty from the unchanged policy score only for consuming moves.",
        "strict_rule": "Exclude consuming moves; if no eligible unseen move remains, stop without backtracking.",
        "scope": "Preserve an existing last hole. Do not reward creating the first hole or penalize 2-to-1 empties or hole relocation.",
        "families": list(FAMILIES), "penalties": list(PENALTIES),
        "selection": "Per family, select the best strictly positive penalty on screen only by wins, foundation sum, fewer steps, then smaller penalty. Report baseline and strict separately. Never reselect from confirmation.",
    }, "protocol")
    options = protocol["options"]
    for name in ("screen_deals", "test_deals", "max_steps", "threads", "jobs"):
        integer(options[name], 1, 2**63 - 1, "options." + name)
    integer(options["seed_offset"], 0, 2**64 - 1 - max(PRODUCTION_SEEDS.values()), "options.seed_offset")
    exact_keys(protocol["seeds"], PRODUCTION_SEEDS, "seeds")
    for name, seed in protocol["seeds"].items():
        integer(seed, 0, 2**64 - 1, "seeds." + name)
    require(len(set(protocol["seeds"].values())) == 3, "phase seeds must be distinct")
    compare(protocol["seeds"], {name: seed + options["seed_offset"] for name, seed in PRODUCTION_SEEDS.items()},
            "phase seed offset")
    if prefix == "hole-preservation":
        compare(options, dict(screen_deals=20000, test_deals=250000, max_steps=1000, seed_offset=0), "production options")
        compare(protocol["seeds"], PRODUCTION_SEEDS, "production seeds")
    verify_sources(protocol, binary=binary, reference_binary=reference_binary)
    exact_keys(screen["baseline_parity"], FAMILIES, "baseline_parity")
    for family, parity in screen["baseline_parity"].items():
        compare(parity, {"deals": 1000, "exact_match": True}, "baseline_parity." + family)
        digest(parity["wins_sha256"], "parity wins")
        digest(parity["foundations_sha256"], "parity foundations")
    exact_keys(screen["families"], FAMILIES, "screen families")
    exact_keys(frozen["selected_penalties"], FAMILIES, "selected penalties")
    compare(frozen["confirmation_rules"], list(RULES), "frozen confirmation rules")
    for family, rows in screen["families"].items():
        expected = [("baseline", 0.0), *(("soft", p) for p in PENALTIES), ("strict", 0.0)]
        require(isinstance(rows, list) and len(rows) == len(expected), family + ": missing screening rows")
        compare([(r["rule"], r["penalty"]) for r in rows], expected, family + " screening candidates/order")
        for row in rows:
            verify_aggregate(row, options["screen_deals"], options["max_steps"], "screen." + family + "." + row["rule"])
        # This independently implements the declared ordering, including the
        # lower-penalty final tie break. Baseline and strict are never selected.
        best = sorted((r for r in rows if r["rule"] == "soft"),
                      key=lambda r: (-r["wins"], -r["foundation_cards"], r["steps"], r["penalty"]))[0]
        compare(frozen["selected_penalties"][family], best["penalty"], family + " selected penalty")
    compare(confirmation, {"variant": "draw3_unlimited", "seed": protocol["seeds"]["confirmation"],
                           "deals": options["test_deals"]}, "confirmation")
    exact_keys(confirmation["families"], FAMILIES, "confirmation families")
    summaries = []
    for family, rows in confirmation["families"].items():
        exact_keys(rows, RULES, "confirmation." + family)
        raw = {}
        for name, row in rows.items():
            label = family + "." + name
            compare(row["rule"], "soft" if name == "selected_soft" else name, label + ".rule")
            compare(row["penalty"], frozen["selected_penalties"][family] if name == "selected_soft" else 0.0,
                    label + ".penalty")
            verify_aggregate(row, options["test_deals"], options["max_steps"], label)
            for kind in ("wins", "foundations"):
                compare(row[kind + "_file"], f"{prefix}-outcomes/{label}.{kind}.u8.gz", label + "." + kind + "_file")
            wins = read_outcome(row, "wins", directory, options["test_deals"], label + ".wins")
            cards = read_outcome(row, "foundations", directory, options["test_deals"], label + ".foundations")
            require(all(bool(win) == (count == 52) for win, count in zip(wins, cards)), label + ": 52 cards iff win failed")
            expected = expected_statistics(sum(wins), cards)
            expected.pop("mean_net_dollars")
            expected.pop("mean_net_dollars_95_ci")
            compare(row, expected, label)
            raw[name] = wins
            summaries.append(f"{label}: {sum(wins):,}/{len(wins):,} wins; indexed outcomes verified")
        require("paired_vs_baseline" not in rows["baseline"], family + ": baseline must not be its own contrast")
        for name in ("selected_soft", "strict"):
            expected = expected_paired(raw[name], raw["baseline"])
            delta, se = expected["delta_percentage_points"], expected["standard_error_percentage_points"]
            expected.update(bonferroni_familywise_95_ci_percentage_points=[delta - Z_FAMILY * se, delta + Z_FAMILY * se],
                            familywise_comparisons=6)
            compare(rows[name]["paired_vs_baseline"], expected, family + "." + name + ".paired_vs_baseline")
    summaries.append(f"Lineage, frozen selections, source hashes, and six paired contrasts verified; "
                     f"binary_checked={str(binary is not None).lower()}, reference_binary_checked={str(reference_binary is not None).lower()}")
    return summaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="hole-preservation")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--reference-binary", type=Path)
    args = parser.parse_args(argv)
    try:
        lines = verify_study(args.prefix, results_dir=args.results_dir, binary=args.binary,
                             reference_binary=args.reference_binary)
    except (ValueError, KeyError, TypeError, OSError, EOFError, zlib.error) as error:
        parser.exit(1, f"Hole study verification failed: {error}\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
