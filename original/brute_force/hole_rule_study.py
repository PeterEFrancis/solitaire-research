"""Matched-deal test of preserving the last tableau hole in unlimited draw three.

Build the isolated native player with brute_force.hole_rule_native first.
The historical engine, weights, and study records are never changed. Positive
penalties are selected on screening deals, frozen, then tested on fresh deals.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import subprocess
import tempfile

from solitaire.player import PARAMETER_NAMES
from solitaire.variants import DEFAULT_VARIANT_POLICY_PATH, load_variant_policy, config_record
from .compare_policies import paired_statistics, sha256_file, validate_outcomes, write_report
from .variant_study import statistics


RESULTS = Path(__file__).resolve().parent / "results"
FAMILIES = ("full", "visible", "simple_eight")
PENALTIES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
SEEDS = {"parity": 2026102700, "screen": 2026102701, "confirmation": 2026102799}


def evaluate(binary, weights, rule, penalty, deals, seed, threads, max_steps, record=False):
    with tempfile.TemporaryDirectory(prefix="solitaire-hole-study-") as directory:
        wins_path, foundation_path = Path(directory) / "wins.u8", Path(directory) / "foundations.u8"
        command = [str(binary), "--n", "13", "--draw-count", "3", "--max-recycles", "-1",
                   "--model-only", "--model-max-steps", str(max_steps), "--model-weights",
                   ",".join(format(x, ".17g") for x in weights), "--benchmark-deals", str(deals),
                   "--benchmark-seed", str(seed), "--threads", str(threads)]
        if rule == "soft":
            command += ["--hole-penalty", str(penalty)]
        elif rule == "strict":
            command += ["--preserve-last-hole"]
        elif rule != "baseline":
            raise ValueError(f"unknown rule: {rule}")
        if record:
            command += ["--benchmark-outcomes", str(wins_path), "--benchmark-foundations", str(foundation_path)]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        fields = dict(line.split(" ", 1) for line in completed.stdout.splitlines())
        if (int(fields["benchmark_deals"]) != deals or int(fields["model_attempts"]) != deals
                or int(fields["exact_fallbacks"]) != 0):
            raise ValueError("unexpected native benchmark mode or sample size")
        if (float(fields.get("hole_penalty", 0)) != penalty
                or bool(int(fields.get("preserve_last_hole", 0))) != (rule == "strict")):
            raise ValueError("native executable did not report the requested hole intervention")
        row = {"deals": deals, "wins": int(fields["model_wins"]),
               "foundation_cards": int(fields["model_foundation_cards"]),
               "foundation_cards_squared": int(fields["model_foundation_cards_squared"]),
               "steps": int(fields["model_steps"]), "cutoffs": int(fields["model_cutoffs"])}
        wins = wins_path.read_bytes() if record else b""
        foundations = foundation_path.read_bytes() if record else b""
        if record:
            validate_outcomes(wins, deals, row["wins"])
            if (len(foundations) != deals or any(f > 52 for f in foundations)
                    or sum(foundations) != row["foundation_cards"]
                    or sum(f * f for f in foundations) != row["foundation_cards_squared"]
                    or any(bool(w) != (f == 52) for w, f in zip(wins, foundations))):
                raise ValueError("indexed outcomes disagree with native totals")
        result = statistics(row)
        # Vegas payout summaries from the shared helper are not relevant here.
        result.pop("mean_net_dollars")
        result.pop("mean_net_dollars_95_ci")
        return {"rule": rule, "penalty": penalty, **result}, wins, foundations


def selection_key(row):
    return row["wins"], row["foundation_cards"], -row["steps"], -row["penalty"]


def comparison(candidate, baseline):
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired outcomes must have equal positive length")
    result = paired_statistics(
        sum(a == 1 and b == 0 for a, b in zip(candidate, baseline)),
        sum(a == 0 and b == 1 for a, b in zip(candidate, baseline)), len(candidate))
    # Six predeclared contrasts: two rules against baseline in each family.
    margin = NormalDist().inv_cdf(1 - 0.05 / (2 * 6)) * result["standard_error_percentage_points"]
    delta = result["delta_percentage_points"]
    result["bonferroni_familywise_95_ci_percentage_points"] = [delta - margin, delta + margin]
    result["familywise_comparisons"] = 6
    return result


def run(args):
    seeds = {phase: seed + args.seed_offset for phase, seed in SEEDS.items()}
    prefix = RESULTS / args.output_prefix
    paths = {kind: prefix.with_suffix("." + kind + ".json")
             for kind in ("protocol", "screen", "frozen", "confirmation")}
    if any(path.exists() for path in paths.values()):
        raise ValueError("output already exists; use a fresh --output-prefix to preserve the evidence")
    config, _ = load_variant_policy("draw3_unlimited")
    weights = {family: load_variant_policy("draw3_unlimited", policy=family)[1] for family in FAMILIES}
    build = json.loads(args.build_manifest.read_text())
    binary_hash = sha256_file(args.binary)
    if build["binary"]["sha256"] != binary_hash:
        raise ValueError("experimental executable does not match its build manifest")
    if (build["source"]["sha256"] != sha256_file(Path(__file__).with_name("native_solver.cpp"))
            or build["builder"]["sha256"] != sha256_file(Path(__file__).with_name("hole_rule_native.py"))):
        raise ValueError("native source or experimental builder changed since compilation")
    protocol = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": "draw3_unlimited", "config": config_record(config),
        "rule_definition": "A consuming move has exactly one empty tableau column before it and zero after it.",
        "soft_rule": "Subtract the penalty from the unchanged policy score only for consuming moves.",
        "strict_rule": "Exclude consuming moves; if no eligible unseen move remains, stop without backtracking.",
        "scope": "Preserve an existing last hole. Do not reward creating the first hole or penalize 2-to-1 empties or hole relocation.",
        "penalties": list(PENALTIES), "families": list(FAMILIES), "seeds": seeds,
        "options": {key: getattr(args, key) for key in ("screen_deals", "test_deals", "max_steps", "threads", "jobs", "seed_offset")},
        "selection": "Per family, select the best strictly positive penalty on screen only by wins, foundation sum, fewer steps, then smaller penalty. Report baseline and strict separately. Never reselect from confirmation.",
        "uncertainty": "Paired normal 95% pointwise intervals and Bonferroni 95% familywise intervals for six predeclared contrasts; fixed-policy IID-uniform interpretation of seeded deals.",
        "information": {"full": "May inspect hidden identities", "visible": "Existing visible-position features", "simple_eight": "Existing eight observable features"},
        "additional_feature_information": "Last-hole consumption depends only on observable tableau occupancy.",
        "parameters": {f: dict(zip(PARAMETER_NAMES, w)) for f, w in weights.items()},
        "policy_source": DEFAULT_VARIANT_POLICY_PATH.name,
        "policy_source_sha256": sha256_file(DEFAULT_VARIANT_POLICY_PATH),
        "native_source_sha256": sha256_file(Path(__file__).with_name("native_solver.cpp")),
        "study_source_sha256": sha256_file(Path(__file__)),
        "builder_source_sha256": sha256_file(Path(__file__).with_name("hole_rule_native.py")),
        "reference_binary_sha256": sha256_file(args.reference_binary),
        "build": build,
    }
    write_report(paths["protocol"], protocol)

    parity = {}
    for family in FAMILIES:
        inputs = (weights[family], "baseline", 0.0, 1000, seeds["parity"], args.threads, args.max_steps)
        reference = evaluate(args.reference_binary, *inputs, record=True)
        experimental = evaluate(args.binary, *inputs, record=True)
        if reference != experimental:
            raise ValueError(f"zero-rule baseline differs from historical executable for {family}")
        parity[family] = {"deals": 1000, "wins_sha256": hashlib.sha256(reference[1]).hexdigest(),
                          "foundations_sha256": hashlib.sha256(reference[2]).hexdigest(), "exact_match": True}
    print("Baseline parity passed for all three policies on 1,000 matched deals each.", flush=True)

    screen = {"schema_version": 1, "complete": False, "baseline_parity": parity,
              "protocol_sha256": sha256_file(paths["protocol"]), "families": {f: [] for f in FAMILIES}}
    tasks = [(f, "baseline", 0.0) for f in FAMILIES]
    tasks += [(f, "soft", p) for f in FAMILIES for p in PENALTIES]
    tasks += [(f, "strict", 0.0) for f in FAMILIES]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(evaluate, args.binary, weights[f], rule, penalty, args.screen_deals,
                               seeds["screen"], args.threads, args.max_steps): f
                   for f, rule, penalty in tasks}
        for future in as_completed(pending):
            family = pending[future]
            row, _, _ = future.result()
            screen["families"][family].append(row)
            print(f"screen {family} {row['rule']} {row['penalty']:g}: {row['win_rate']:.4%}", flush=True)
            write_report(paths["screen"], screen)
    for rows in screen["families"].values():
        rows.sort(key=lambda row: (row["rule"], row["penalty"]))
    screen["complete"] = True
    write_report(paths["screen"], screen)
    selected = {f: max((r for r in screen["families"][f] if r["rule"] == "soft"), key=selection_key)["penalty"]
                for f in FAMILIES}
    frozen = {"schema_version": 1, "complete": True, "protocol_sha256": sha256_file(paths["protocol"]),
              "screen_sha256": sha256_file(paths["screen"]), "selected_penalties": selected,
              "confirmation_rules": ["baseline", "selected_soft", "strict"]}
    write_report(paths["frozen"], frozen)
    print("Frozen positive penalties:", selected, flush=True)

    outcome_dir = RESULTS / (args.output_prefix + "-outcomes")
    outcome_dir.mkdir(exist_ok=False)
    confirmation = {"schema_version": 1, "complete": False, "variant": "draw3_unlimited",
                    "frozen_sha256": sha256_file(paths["frozen"]), "protocol_sha256": sha256_file(paths["protocol"]),
                    "seed": seeds["confirmation"], "deals": args.test_deals, "families": {f: {} for f in FAMILIES}}
    raw = {f: {} for f in FAMILIES}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {}
        for family in FAMILIES:
            for name in ("baseline", "selected_soft", "strict"):
                rule = "soft" if name == "selected_soft" else name
                penalty = selected[family] if rule == "soft" else 0.0
                future = pool.submit(evaluate, args.binary, weights[family], rule, penalty, args.test_deals,
                                     seeds["confirmation"], args.threads, args.max_steps, True)
                pending[future] = (family, name)
        for future in as_completed(pending):
            family, name = pending[future]
            row, wins, foundations = future.result()
            for kind, data in (("wins", wins), ("foundations", foundations)):
                path = outcome_dir / f"{family}.{name}.{kind}.u8.gz"
                path.write_bytes(gzip.compress(data, mtime=0))
                row[kind + "_file"] = str(path.relative_to(RESULTS))
                row[kind + "_sha256"] = hashlib.sha256(data).hexdigest()
            confirmation["families"][family][name] = row
            raw[family][name] = wins
            print(f"confirmation {family} {name}: {row['wins']}/{row['deals']} ({row['win_rate']:.4%})", flush=True)
            write_report(paths["confirmation"], confirmation)
    for family in FAMILIES:
        for name in ("selected_soft", "strict"):
            confirmation["families"][family][name]["paired_vs_baseline"] = comparison(raw[family][name], raw[family]["baseline"])
    confirmation["complete"] = True
    write_report(paths["confirmation"], confirmation)
    print("Complete:", paths["confirmation"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--reference-binary", type=Path, default=Path(__file__).with_name("native_solver.variants"))
    parser.add_argument("--output-prefix", default="hole-preservation")
    parser.add_argument("--screen-deals", type=int, default=20000)
    parser.add_argument("--test-deals", type=int, default=250000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0, help="Use a nonzero offset for smoke checks, keeping production deals fresh")
    args = parser.parse_args()
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix or "." in args.output_prefix:
        parser.error("output-prefix must be a filename stem without dots")
    if min(args.screen_deals, args.test_deals, args.max_steps, args.threads, args.jobs) < 1:
        parser.error("deal counts, move limit, threads, and jobs must be positive")
    if args.seed_offset < 0 or max(SEEDS.values()) + args.seed_offset >= 2**64:
        parser.error("seed-offset must keep all seeds in the unsigned 64-bit range")
    args.binary, args.reference_binary = args.binary.resolve(), args.reference_binary.resolve()
    run(args)


if __name__ == "__main__":
    main()
