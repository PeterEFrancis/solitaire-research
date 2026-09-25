from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from solitaire.player import PARAMETER_NAMES

from .tune_model import evaluate_native


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank model features by common-deal zero-weight ablation"
    )
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("brute_force/native_solver"))
    parser.add_argument("--n", type=int, default=13)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--t", type=int, default=7)
    parser.add_argument("--deals", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=940_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    return parser


def load_parameters(path: Path) -> list[float]:
    report = json.loads(path.read_text(encoding="ascii"))
    feature_names = report.get("feature_names")
    selected = report.get("selected_parameters")
    if feature_names != list(PARAMETER_NAMES) or not isinstance(selected, dict):
        raise ValueError("model source does not contain the current feature set")
    return [float(selected[name]) for name in PARAMETER_NAMES]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.k != 2 or args.deals < 1 or args.threads < 1 or args.max_steps < 1:
        raise ValueError("require k=2 and positive deals, threads, and max steps")

    parameters = load_parameters(args.model_source)
    baseline = evaluate_native(
        args.binary,
        parameters,
        n=args.n,
        deals=args.deals,
        seed=args.seed,
        threads=args.threads,
        max_steps=args.max_steps,
    )
    print(
        f"baseline {baseline.wins}/{baseline.deals} ({baseline.win_rate:.5%})",
        flush=True,
    )

    ablations = []
    for index, (name, weight) in enumerate(zip(PARAMETER_NAMES, parameters)):
        candidate = list(parameters)
        candidate[index] = 0.0
        evaluation = evaluate_native(
            args.binary,
            candidate,
            n=args.n,
            deals=args.deals,
            seed=args.seed,
            threads=args.threads,
            max_steps=args.max_steps,
        )
        loss = baseline.wins - evaluation.wins
        ablations.append(
            {
                "model_index": index + 1,
                "name": name,
                "weight": weight,
                "ablated": asdict(evaluation) | {"win_rate": evaluation.win_rate},
                "wins_lost": loss,
                "win_rate_loss_percentage_points": 100.0 * loss / args.deals,
            }
        )
        print(
            f"{index + 1:2d} {name} weight={weight:.8g} "
            f"wins_lost={loss:+d}",
            flush=True,
        )

    ablations.sort(key=lambda item: (-item["wins_lost"], item["model_index"]))
    for rank, item in enumerate(ablations, start=1):
        item["importance_rank"] = rank

    result = {
        "schema_version": 1,
        "config": {"n": args.n, "k": args.k, "t": args.t},
        "method": (
            "set one selected weight to zero at a time and compare wins on "
            "the same deals; larger positive wins_lost means greater empirical "
            "contribution in the fitted model"
        ),
        "model_source": str(args.model_source),
        "options": {
            "deals": args.deals,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "threads": args.threads,
        },
        "baseline": asdict(baseline) | {"win_rate": baseline.win_rate},
        "features": ablations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(args.output)
    print(f"output {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
