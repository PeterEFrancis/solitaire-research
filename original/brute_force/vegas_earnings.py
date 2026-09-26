"""Recompute Vegas earnings distributions from verified saved study outcomes.

No games are simulated. Run ``python3 -m brute_force.vegas_earnings`` from
original/ to write results/vegas-earnings.json, or add ``--check`` to verify
that the saved report exactly matches the current source artifacts.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .compare_policies import NORMAL_95_Z, wilson_interval, write_report


RESULTS = Path(__file__).resolve().parent / "results"
POLICIES = ("profit", "full", "visible", "simple_eight", "stage8_baseline", "portfolio")
BINS = ((0, 0), (1, 5), (6, 10), (11, 15), (16, 25), (26, 51), (52, 52))
VEGAS_CONFIG = {
    "n": 13, "k": 2, "t": 7, "draw_count": 1, "max_recycles": 0,
    "allow_tableau_stack_splitting": False,
}
LABELS = {
    "profit": "Profit-tuned full-information policy",
    "full": "Win-tuned full-information policy",
    "visible": "Visible-information policy",
    "simple_eight": "Eight-feature visible policy",
    "stage8_baseline": "Stage-8 baseline",
    "portfolio": "Full-information restart portfolio",
}


def net_dollars(foundation_cards: int) -> int:
    if isinstance(foundation_cards, bool) or not isinstance(foundation_cards, int) or not 0 <= foundation_cards <= 52:
        raise ValueError("foundation card count must be an integer from 0 through 52")
    return 5 * foundation_cards - 52


def empirical_quantile(counts: Sequence[int], probability: Fraction) -> int:
    """Inverse empirical CDF, using the lowest observed value at probability 0."""
    if len(counts) != 53 or any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("expected 53 nonnegative integer PMF counts")
    sample_size = sum(counts)
    if sample_size == 0:
        raise ValueError("earnings sample must not be empty")
    probability = Fraction(probability)
    if not 0 <= probability <= 1:
        raise ValueError("quantile probability must be between 0 and 1")
    rank = max(1, (probability.numerator * sample_size + probability.denominator - 1) // probability.denominator)
    cumulative = 0
    for cards, count in enumerate(counts):
        cumulative += count
        if cumulative >= rank:
            return net_dollars(cards)
    raise AssertionError("quantile rank exceeds sample size")


def analyze_outcomes(outcomes: bytes) -> dict[str, Any]:
    """Return the complete discrete distribution, in net dollars per deal."""
    if not outcomes or any(cards > 52 for cards in outcomes):
        raise ValueError("expected a nonempty byte stream of foundation counts 0 through 52")
    counts = [0] * 53
    for cards in outcomes:
        counts[cards] += 1
    sample_size = len(outcomes)
    total = sum(count * net_dollars(cards) for cards, count in enumerate(counts))
    total_squared = sum(count * net_dollars(cards)**2 for cards, count in enumerate(counts))
    mean = total / sample_size
    if sample_size > 1:
        variance = (sample_size * total_squared - total**2) / (sample_size * (sample_size - 1))
        standard_deviation = math.sqrt(variance)
        margin = NORMAL_95_Z * standard_deviation / math.sqrt(sample_size)
        mean_interval = [mean - margin, mean + margin]
    else:
        standard_deviation = mean_interval = None

    def event(count: int) -> dict[str, Any]:
        return {"count": count, "probability": count / sample_size,
                "wilson_95_ci": list(wilson_interval(count, sample_size))}

    quantiles = {
        f"q{percent:02d}": empirical_quantile(counts, Fraction(percent, 100))
        for percent in (5, 25, 50, 75, 95)
    }
    return {
        "sample_size": sample_size,
        "mean_net_dollars": mean,
        "mean_net_dollars_95_ci": mean_interval,
        "median_net_dollars": quantiles["q50"],
        "sd_net_dollars": standard_deviation,
        "quantiles_net_dollars": quantiles,
        "events": {
            "profit": event(sum(counts[11:])),
            "loss": event(sum(counts[:11])),
            "full_loss": event(counts[0]),
            "full_win": event(counts[52]),
        },
        "bins": [
            {"foundation_cards_min": low, "foundation_cards_max": high,
             "net_dollars_min": net_dollars(low), "net_dollars_max": net_dollars(high),
             "label": str(low) if low == high else f"{low}–{high}",
             "count": sum(counts[low:high + 1]),
             "probability": sum(counts[low:high + 1]) / sample_size}
            for low, high in BINS
        ],
        "pmf": [
            {"foundation_cards": cards, "net_dollars": net_dollars(cards),
             "count": count, "probability": count / sample_size,
             "cumulative_probability": sum(counts[:cards + 1]) / sample_size}
            for cards, count in enumerate(counts)
        ],
    }


def require_vegas_config(value: object) -> None:
    if not isinstance(value, dict) or any(
        name not in value or type(value[name]) is not type(expected) or value[name] != expected
        for name, expected in VEGAS_CONFIG.items()
    ):
        raise ValueError("source does not use draw-one, single-pass, no-splitting Vegas rules")


def read_foundations(row: dict[str, Any], results_dir: Path) -> tuple[bytes, dict[str, Any]]:
    filename = row.get("foundations_file")
    if not isinstance(filename, str):
        raise ValueError("missing saved foundation outcome filename")
    source = (results_dir / filename).resolve()
    if Path(filename).is_absolute() or not source.is_relative_to(results_dir.resolve()):
        raise ValueError("foundation outcome file must be inside the results directory")
    compressed = source.read_bytes()
    try:
        outcomes = gzip.decompress(compressed)
    except (OSError, EOFError) as error:
        raise ValueError(f"invalid gzip foundation outcomes: {filename}") from error
    raw_digest = hashlib.sha256(outcomes).hexdigest()
    if raw_digest != row.get("foundations_sha256"):
        raise ValueError(f"foundation outcome checksum mismatch: {filename}")
    for field in ("deals", "foundation_cards", "foundation_cards_squared", "wins"):
        if type(row.get(field)) is not int or row[field] < 0:
            raise ValueError(f"invalid recorded foundation statistic: {field}")
    if row["deals"] < 1 or len(outcomes) != row["deals"]:
        raise ValueError(f"foundation outcome sample length mismatch: {filename}")
    if any(cards > 52 for cards in outcomes):
        raise ValueError(f"foundation outcome outside 0 through 52: {filename}")
    if (sum(outcomes) != row["foundation_cards"]
            or sum(cards * cards for cards in outcomes) != row["foundation_cards_squared"]):
        raise ValueError(f"foundation outcome moment mismatch: {filename}")
    if outcomes.count(52) != row["wins"]:
        raise ValueError(f"full-win count disagrees with saved foundation outcomes: {filename}")
    return outcomes, {
        "foundations_file": filename,
        "raw_outcomes_sha256": raw_digest,
        "compressed_outcomes_sha256": hashlib.sha256(compressed).hexdigest(),
    }


def build_report(prefix: str = "variant-study", *, results_dir: Path = RESULTS) -> dict[str, Any]:
    """Deterministically recompute the report after validating its saved sources."""
    if Path(prefix).name != prefix:
        raise ValueError("prefix must be a filename prefix")
    results_dir = Path(results_dir)
    confirmation_path = results_dir / f"{prefix}.confirmation.json"
    frozen_path = results_dir / f"{prefix}.selected-policies.json"
    raw_confirmation = confirmation_path.read_bytes()
    raw_frozen = frozen_path.read_bytes()
    confirmation = json.loads(raw_confirmation)
    frozen = json.loads(raw_frozen)
    if (not isinstance(confirmation, dict) or confirmation.get("complete") is not True
            or not isinstance(frozen, dict) or frozen.get("complete") is not True):
        raise ValueError("confirmation and frozen policy artifacts must be complete")
    if confirmation.get("schema_version") != 1 or frozen.get("schema_version") != 1:
        raise ValueError("expected schema_version 1 study artifacts")
    frozen_digest = hashlib.sha256(raw_frozen).hexdigest()
    if confirmation.get("frozen_policies_sha256") != frozen_digest:
        raise ValueError("frozen policy checksum disagrees with the confirmation artifact")
    try:
        vegas = confirmation["variants"]["vegas"]
        require_vegas_config(vegas["config"])
        require_vegas_config(frozen["variants"]["vegas"]["config"])
        evaluations = vegas["evaluations"]
    except (KeyError, TypeError) as error:
        raise ValueError("study artifacts lack a valid Vegas record") from error
    policies = {}
    sample_sizes = set()
    for name in POLICIES:
        if name not in evaluations or not isinstance(evaluations[name], dict):
            raise ValueError(f"missing Vegas policy evaluation: {name}")
        row = evaluations[name]
        outcomes, provenance = read_foundations(row, results_dir)
        policy = analyze_outcomes(outcomes)
        sample_sizes.add(policy["sample_size"])
        attempts = row.get("attempts_per_deal", 1)
        if type(attempts) is not int or attempts < 1:
            raise ValueError("attempts_per_deal must be a positive integer")
        portfolio = name == "portfolio"
        policy.update({
            "label": LABELS[name],
            "policy_type": "full_information_restart_portfolio" if portfolio else "single_policy",
            "information_group": "visible_information" if name in ("visible", "simple_eight") else "full_information",
            "attempts_per_deal": attempts,
            "interpretation": (
                "Full-information per-deal maximum foundation count across frozen restarts. The $52 cost is applied once to the best trajectory; this is not an ordinary paid-play policy or the earnings from separately paid attempts."
                if portfolio else
                "Net return from one fixed policy on each sampled deal; computer outcomes are not measured human performance."
            ),
            "provenance": provenance,
        })
        policies[name] = policy
    if len(sample_sizes) != 1:
        raise ValueError("policy sample sizes differ")
    return {
        "schema_version": 1,
        "source": {
            "confirmation_file": confirmation_path.name,
            "confirmation_sha256": hashlib.sha256(raw_confirmation).hexdigest(),
            "selected_policies_file": frozen_path.name,
            "selected_policies_sha256": frozen_digest,
        },
        "config": dict(VEGAS_CONFIG),
        "evaluation_options": confirmation.get("evaluation_options"),
        "rules": {
            "deck_cost_dollars": 52, "payout_per_foundation_card_dollars": 5,
            "net_dollars_formula": "5 * foundation_cards - 52",
            "break_even_possible": False, "first_profitable_foundation_count": 11,
            "min_net_dollars": -52, "max_net_dollars": 208,
        },
        "statistics": {
            "confidence": 0.95,
            "mean_interval": "Normal approximation using sample standard deviation divided by sqrt(N); null for N=1.",
            "sd_definition": "Sample standard deviation with denominator N-1; null for N=1.",
            "quantile_definition": "Inverse empirical CDF: smallest observed net return with cumulative probability at least q.",
            "median_definition": "Inverse empirical CDF at q=0.5; lower central value when an even sample straddles a gap.",
            "event_intervals": "Two-sided Wilson score intervals.",
            "limitations": "Sampling intervals assume IID-uniform deals under the seeded-shuffle interpretation and fixed policies. They are not adjusted for multiple comparisons and do not measure model or human-execution uncertainty.",
        },
        "policies": policies,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="variant-study")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.results_dir / ("vegas-earnings.json" if args.prefix == "variant-study" else f"{args.prefix}.vegas-earnings.json")
    report = build_report(args.prefix, results_dir=args.results_dir)
    protected = {
        (args.results_dir / report["source"]["confirmation_file"]).resolve(),
        (args.results_dir / report["source"]["selected_policies_file"]).resolve(),
    }
    confirmation = json.loads((args.results_dir / report["source"]["confirmation_file"]).read_bytes())
    for variant in confirmation["variants"].values():
        for row in variant.get("evaluations", {}).values():
            for key in ("foundations_file", "wins_file"):
                if isinstance(row.get(key), str):
                    protected.add((args.results_dir / row[key]).resolve())
    if output.resolve() in protected:
        parser.error("output must not overwrite a source artifact")
    if args.check:
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved != report:
            raise ValueError(f"saved Vegas earnings report is stale or modified: {output}")
        print(f"Vegas earnings report verified: {output}")
    else:
        write_report(output, report)
        print(f"Vegas earnings report saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
