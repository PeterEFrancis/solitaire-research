"""Compare fixed policies on the same reproducible native benchmark deals.

This command supports only the existing standard rules: n=13, k=2, t=7,
draw one, three recycles, packed-stack splitting, and no foundation rollback.
The input JSON maps policy names to feature-name/weight objects. Omitted
features have weight zero. Policies run sequentially on identical seed/index
pairs; native outcome files contain one byte (0 or 1) per deal, not packed bits.

Example, from original/::

    python3 -m brute_force.compare_policies --policies policies.json \
        --baseline stage7 --deals 1000000 --seed 2026092501 \
        --output comparison.json

Intervals describe Monte Carlo sampling uncertainty for fixed policies under
the IID-uniform interpretation of seeded shuffles. They do not correct for
policy selection or multiple comparisons. The paired normal interval can be
unreliable with few discordant deals, and degenerates when none are observed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
from statistics import NormalDist
import subprocess
import tempfile
import time
from typing import Any, Sequence

from solitaire.player import PARAMETER_NAMES


STANDARD_CONFIG = {
    "n": 13,
    "k": 2,
    "t": 7,
    "draw_count": 1,
    "max_recycles": 3,
    "allow_tableau_stack_splitting": True,
    "allow_foundation_to_tableau": False,
}
NORMAL_95_Z = NormalDist().inv_cdf(0.975)


def validate_policies(data: object) -> dict[str, list[float]]:
    if not isinstance(data, dict) or not data:
        raise ValueError("policies must be a nonempty JSON object")
    policies = {}
    for name, supplied in data.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("policy names must be nonempty strings")
        if not isinstance(supplied, dict):
            raise ValueError(f"policy {name!r} must map feature names to weights")
        unknown = set(supplied) - set(PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"policy {name!r} has unknown features: {sorted(unknown, key=str)!r}")
        weights = []
        for feature in PARAMETER_NAMES:
            value = supplied.get(feature, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"policy {name!r}, feature {feature!r}: weight must be a number")
            try:
                weight = float(value)
            except OverflowError as error:
                raise ValueError(f"policy {name!r}, feature {feature!r}: weight must be finite") from error
            if not math.isfinite(weight):
                raise ValueError(f"policy {name!r}, feature {feature!r}: weight must be finite")
            weights.append(weight)
        policies[name] = weights
    return policies


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name!r}")
        result[name] = value
    return result


def load_policies(path: Path) -> tuple[dict[str, list[float]], bytes]:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_json_object)
    return validate_policies(data), raw


def validate_counts(wins: int, deals: int) -> None:
    if (
        isinstance(wins, bool)
        or isinstance(deals, bool)
        or not isinstance(wins, int)
        or not isinstance(deals, int)
        or deals < 1
        or not 0 <= wins <= deals
    ):
        raise ValueError("require integer deals >= 1 and 0 <= wins <= deals")


def wilson_interval(wins: int, deals: int) -> tuple[float, float]:
    """Two-sided Wilson score interval at 95% confidence."""
    validate_counts(wins, deals)
    proportion = wins / deals
    z_squared = NORMAL_95_Z**2
    denominator = 1.0 + z_squared / deals
    center = (proportion + z_squared / (2.0 * deals)) / denominator
    half_width = NORMAL_95_Z * math.sqrt(
        proportion * (1.0 - proportion) / deals
        + z_squared / (4.0 * deals**2)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def paired_statistics(b: int, c: int, deals: int) -> dict[str, Any]:
    """b=candidate-only wins; c=baseline-only wins; variance uses the plug-in MLE."""
    validate_counts(b, deals)
    validate_counts(c, deals)
    if b + c > deals:
        raise ValueError("discordant counts cannot exceed the number of deals")
    difference = (b - c) / deals
    standard_error = math.sqrt(max(0.0, (b + c) / deals - difference**2) / deals)
    half_width = NORMAL_95_Z * standard_error
    return {
        "deals": deals,
        "b_candidate_wins_baseline_loses": b,
        "c_baseline_wins_candidate_loses": c,
        "discordant_deals": b + c,
        "net_wins": b - c,
        "delta_percentage_points": 100.0 * difference,
        "standard_error_percentage_points": 100.0 * standard_error,
        "paired_normal_95_ci_percentage_points": [
            100.0 * (difference - half_width),
            100.0 * (difference + half_width),
        ],
    }


def validate_outcomes(outcomes: bytes, deals: int, wins: int) -> None:
    validate_counts(wins, deals)
    if len(outcomes) != deals:
        raise ValueError(f"native outcomes contain {len(outcomes)} bytes; expected {deals}")
    if any(value not in (0, 1) for value in outcomes):
        raise ValueError("native outcomes must contain only byte values 0 and 1")
    if sum(outcomes) != wins:
        raise ValueError("native outcome wins disagree with reported model_wins")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def compiler_metadata() -> dict[str, Any]:
    """Report an available compiler, without claiming it built the binary."""
    compiler = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        return {"path": None, "version": None, "binary_build_compiler": "unknown"}
    try:
        result = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True, timeout=5, check=True
        )
        return {
            "path": compiler,
            "version": result.stdout.strip(),
            "binary_build_compiler": "unknown",
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"path": compiler, "version": None, "error": str(error), "binary_build_compiler": "unknown"}


def evaluate_policy(
    binary: Path,
    weights: Sequence[float],
    *,
    deals: int,
    seed: int,
    threads: int,
    max_steps: int,
    outcomes_path: Path,
) -> tuple[dict[str, Any], bytes]:
    command = [
        str(binary), "--n", "13", "--max-recycles", "3",
        "--threads", str(threads), "--benchmark-deals", str(deals),
        "--benchmark-seed", str(seed), "--model-only",
        "--model-max-steps", str(max_steps),
        "--model-weights", ",".join(f"{weight:.17g}" for weight in weights),
        "--benchmark-outcomes", str(outcomes_path),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    wall_seconds = time.perf_counter() - started
    fields = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition(" ")
        if separator:
            if key in fields:
                raise ValueError(f"native output contains duplicate field {key!r}")
            fields[key] = value
    try:
        reported_deals = int(fields["benchmark_deals"])
        attempts = int(fields["model_attempts"])
        wins = int(fields["model_wins"])
        steps = int(fields["model_steps"])
        fallbacks = int(fields["exact_fallbacks"])
        native_elapsed = float(fields["elapsed_seconds"])
    except (KeyError, ValueError) as error:
        raise ValueError("native output is missing valid benchmark statistics") from error
    if reported_deals != deals or attempts != deals or fallbacks != 0:
        raise ValueError("native output does not describe the requested model-only benchmark")
    if not 0 <= steps <= deals * max_steps:
        raise ValueError("native model_steps is outside the requested step budget")
    if not math.isfinite(native_elapsed) or native_elapsed < 0:
        raise ValueError("native elapsed_seconds must be finite and nonnegative")
    outcomes = outcomes_path.read_bytes()
    validate_outcomes(outcomes, deals, wins)
    return {
        "wins": wins,
        "deals": deals,
        "win_rate": wins / deals,
        "wilson_95_ci": list(wilson_interval(wins, deals)),
        "model_steps": steps,
        "average_steps": steps / deals,
        "elapsed_seconds": native_elapsed,
        "wall_seconds": wall_seconds,
        "parameters": dict(zip(PARAMETER_NAMES, weights)),
        "outcomes": {
            "format": "one byte per deal in benchmark index order; 0=loss, 1=win",
            "bytes": len(outcomes),
            "sha256": hashlib.sha256(outcomes).hexdigest(),
        },
    }, outcomes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--binary", type=Path, default=Path(__file__).with_name("native_solver"))
    parser.add_argument("--deals", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.deals < 1 or args.threads < 1 or args.max_steps < 1:
        parser.error("deals, threads, and max steps must be positive")
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be an unsigned 64-bit integer")
    policies, raw_input = load_policies(args.policies)
    if args.baseline not in policies:
        parser.error("baseline must be a name in the policies input")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"binary is not an executable file: {binary}")
    if args.output.resolve() in (args.policies.resolve(), binary):
        parser.error("output must differ from the policies input and native binary")
    source = Path(__file__).with_name("native_solver.cpp").resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": dict(STANDARD_CONFIG),
        "baseline": args.baseline,
        "feature_names": list(PARAMETER_NAMES),
        "options": {"deals": args.deals, "seed": args.seed, "threads": args.threads, "max_steps": args.max_steps},
        "provenance": {
            "policies_input": {"path": str(args.policies.resolve()), "sha256": hashlib.sha256(raw_input).hexdigest()},
            "binary": {"path": str(binary), "sha256": sha256_file(binary)},
            "native_source": {"path": str(source), "sha256": sha256_file(source) if source.is_file() else None},
            "comparison_source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "available_compiler": compiler_metadata(),
            "source_binary_match_verified": False,
        },
        "uncertainty": {
            "confidence": 0.95,
            "per_policy": "two-sided Wilson score interval",
            "paired": "normal interval for mean(candidate outcome - baseline outcome), using plug-in variance",
            "assumption": "fixed policies and IID-uniform deals, interpreted through reproducible pseudorandom shuffles",
            "limitations": "No selection or multiple-comparison adjustment; paired normal intervals can be unreliable with few discordants and degenerate with none.",
        },
        "policies": {},
        "comparisons_to_baseline": {},
    }
    names = [args.baseline] + [name for name in policies if name != args.baseline]
    result["execution_order"] = names
    retained_directory = None
    if args.outcomes_dir is not None:
        args.outcomes_dir.mkdir(parents=True, exist_ok=True)
        retained_directory = Path(tempfile.mkdtemp(
            prefix="comparison-", dir=args.outcomes_dir.resolve()
        ))
    result["outcomes_directory"] = str(retained_directory) if retained_directory is not None else None
    write_report(args.output, result)
    with tempfile.TemporaryDirectory(prefix="solitaire-policy-comparison-") as temporary:
        destination = retained_directory if retained_directory is not None else Path(temporary)
        baseline_outcomes = b""
        for index, name in enumerate(names):
            name_hash = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
            outcomes_path = destination / f"{index:03d}-{name_hash}.outcomes.bin"
            policy_result, outcomes = evaluate_policy(
                binary, policies[name], deals=args.deals, seed=args.seed,
                threads=args.threads, max_steps=args.max_steps, outcomes_path=outcomes_path,
            )
            policy_result["outcomes"]["path"] = str(outcomes_path) if args.outcomes_dir is not None else None
            result["policies"][name] = policy_result
            if name == args.baseline:
                baseline_outcomes = outcomes
            else:
                b = c = 0
                for candidate, baseline in zip(outcomes, baseline_outcomes):
                    b += candidate == 1 and baseline == 0
                    c += candidate == 0 and baseline == 1
                result["comparisons_to_baseline"][name] = paired_statistics(b, c, args.deals)
            write_report(args.output, result)
            print(f"{name!r}: {policy_result['wins']}/{args.deals} ({policy_result['win_rate']:.5%})", flush=True)
    result["complete"] = True
    write_report(args.output, result)
    print(f"comparison saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
