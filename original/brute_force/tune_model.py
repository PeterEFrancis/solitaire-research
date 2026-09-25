from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Sequence

from solitaire.game import DeckConfig
from solitaire.parameter_store import DEFAULT_PARAMETER_PATH, load_parameters_for_config
from solitaire.player import DEFAULT_PARAMETERS, FEATURE_COUNT, PARAMETER_NAMES


DEFAULT_STEPS = (
    1.0,
    1.0,
    0.5,
    0.25,
    0.25,
    0.5,
    0.25,
    0.5,
    0.5,
    0.5,
    0.05,
    0.1,
    0.5,
    0.1,
    0.25,
    0.5,
    0.5,
    0.1,
    1.0,
    0.25,
    0.25,
    0.5,
    0.25,
    0.5,
    0.5,
    0.5,
    0.25,
    0.5,
    0.5,
    0.25,
    0.025,
    0.5,
    0.1,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.25,
    0.5,
    0.1,
    0.1,
)


@dataclass(frozen=True)
class NativeEvaluation:
    deals: int
    wins: int
    model_steps: int
    seed: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.deals

    @property
    def score(self) -> tuple[int, int]:
        return self.wins, -self.model_steps


def parse_overrides(items: Sequence[str]) -> dict[str, float]:
    result = {}
    for item in items:
        name, separator, raw_value = item.partition("=")
        if not separator or name not in PARAMETER_NAMES:
            raise ValueError(f"expected NAME=VALUE for a known parameter: {item}")
        result[name] = float(raw_value)
    return result


def evaluate_native(
    binary: Path,
    parameters: Sequence[float],
    *,
    n: int,
    deals: int,
    seed: int,
    threads: int,
    max_steps: int,
) -> NativeEvaluation:
    encoded_parameters = ",".join(f"{value:.17g}" for value in parameters)
    command = [
        str(binary),
        "--n",
        str(n),
        "--threads",
        str(threads),
        "--benchmark-deals",
        str(deals),
        "--benchmark-seed",
        str(seed),
        "--model-only",
        "--model-max-steps",
        str(max_steps),
        "--model-weights",
        encoded_parameters,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition(" ")
        if separator:
            fields[key] = value
    return NativeEvaluation(
        deals=int(fields["benchmark_deals"]),
        wins=int(fields["model_wins"]),
        model_steps=int(fields["model_steps"]),
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune the native greedy model with common random deals"
    )
    parser.add_argument("--binary", type=Path, default=Path("brute_force/native_solver"))
    parser.add_argument("--parameter-file", type=Path, default=DEFAULT_PARAMETER_PATH)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Start from selected_parameters in a previous tuning result",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=13)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--t", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--train-deals", type=int, default=50_000)
    parser.add_argument("--validation-deals", type=int, default=100_000)
    parser.add_argument("--train-seed", type=int, default=710_001)
    parser.add_argument("--validation-seed", type=int, default=720_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--step-decay", type=float, default=0.5)
    parser.add_argument(
        "--parameter",
        action="append",
        choices=PARAMETER_NAMES,
        help="Tune only the named parameter; may be repeated",
    )
    parser.add_argument(
        "--initial",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a starting parameter",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rounds < 1 or args.train_deals < 1 or args.validation_deals < 1:
        raise ValueError("rounds and deal counts must be positive")
    if args.step_scale <= 0:
        raise ValueError("step scale must be positive")
    if not 0 < args.step_decay <= 1:
        raise ValueError("step decay must be in (0, 1]")

    config = DeckConfig(n=args.n, k=args.k, t=args.t)
    loaded = load_parameters_for_config(config, args.parameter_file)
    if loaded is None:
        raise ValueError(f"no parameter record for {config}")
    parameters = list(loaded)
    if args.resume is not None:
        resumed = json.loads(args.resume.read_text(encoding="ascii"))
        selected_parameters = resumed.get("selected_parameters", {})
        unknown_parameters = set(selected_parameters) - set(PARAMETER_NAMES)
        if unknown_parameters:
            raise ValueError(
                "resume file contains unknown parameters: "
                + ", ".join(sorted(unknown_parameters))
            )
        parameters = [
            float(selected_parameters.get(name, default))
            for name, default in zip(PARAMETER_NAMES, DEFAULT_PARAMETERS)
        ]
    for name, value in parse_overrides(args.initial).items():
        parameters[PARAMETER_NAMES.index(name)] = value

    tuned_indices = (
        [PARAMETER_NAMES.index(name) for name in args.parameter]
        if args.parameter
        else list(range(FEATURE_COUNT))
    )
    cache: dict[tuple[float, ...], NativeEvaluation] = {}

    def train_evaluation(candidate: Sequence[float]) -> NativeEvaluation:
        key = tuple(candidate)
        if key not in cache:
            cache[key] = evaluate_native(
                args.binary,
                candidate,
                n=args.n,
                deals=args.train_deals,
                seed=args.train_seed,
                threads=args.threads,
                max_steps=args.max_steps,
            )
        return cache[key]

    best_train = train_evaluation(parameters)
    initial_validation = evaluate_native(
        args.binary,
        parameters,
        n=args.n,
        deals=args.validation_deals,
        seed=args.validation_seed,
        threads=args.threads,
        max_steps=args.max_steps,
    )
    snapshots = [
        {
            "round": 0,
            "parameters": list(parameters),
            "train": asdict(best_train) | {"win_rate": best_train.win_rate},
            "validation": asdict(initial_validation)
            | {"win_rate": initial_validation.win_rate},
        }
    ]

    def write_result() -> dict[str, object]:
        selected = max(
            snapshots,
            key=lambda snapshot: (
                snapshot["validation"]["wins"],
                -snapshot["validation"]["model_steps"],
                snapshot["train"]["wins"],
                -snapshot["train"]["model_steps"],
            ),
        )
        result = {
            "schema_version": 1,
            "config": {"n": args.n, "k": args.k, "t": args.t},
            "feature_names": list(PARAMETER_NAMES),
            "feature_count": FEATURE_COUNT,
            "method": "coordinate ascent on common random training deals",
            "options": {
                "max_steps": args.max_steps,
                "step_decay": args.step_decay,
                "step_scale": args.step_scale,
                "threads": args.threads,
                "train_deals": args.train_deals,
                "train_seed": args.train_seed,
                "validation_deals": args.validation_deals,
                "validation_seed": args.validation_seed,
            },
            "selected_round": selected["round"],
            "selected_parameters": {
                name: value
                for name, value in zip(PARAMETER_NAMES, selected["parameters"])
            },
            "snapshots": snapshots,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        temporary.replace(args.output)
        return selected

    print(
        f"initial train={best_train.wins}/{best_train.deals} "
        f"({best_train.win_rate:.5%})",
        flush=True,
    )
    print(
        f"initial validation={initial_validation.wins}/"
        f"{initial_validation.deals} ({initial_validation.win_rate:.5%})",
        flush=True,
    )
    write_result()

    for round_index in range(args.rounds):
        round_scale = args.step_scale * args.step_decay**round_index
        for index in tuned_indices:
            step = DEFAULT_STEPS[index] * round_scale
            candidates = []
            for direction in (-1.0, 1.0):
                candidate = list(parameters)
                candidate[index] += direction * step
                candidates.append((candidate, train_evaluation(candidate)))
            candidate, evaluation = max(candidates, key=lambda item: item[1].score)
            if evaluation.score > best_train.score:
                parameters = candidate
                best_train = evaluation
                print(
                    f"round={round_index + 1} {PARAMETER_NAMES[index]}="
                    f"{parameters[index]:.8g} train={best_train.wins} "
                    f"({best_train.win_rate:.5%})",
                    flush=True,
                )

        validation = evaluate_native(
            args.binary,
            parameters,
            n=args.n,
            deals=args.validation_deals,
            seed=args.validation_seed,
            threads=args.threads,
            max_steps=args.max_steps,
        )
        snapshots.append(
            {
                "round": round_index + 1,
                "parameters": list(parameters),
                "train": asdict(best_train) | {"win_rate": best_train.win_rate},
                "validation": asdict(validation) | {"win_rate": validation.win_rate},
            }
        )
        print(
            f"round={round_index + 1} validation={validation.wins}/"
            f"{validation.deals} ({validation.win_rate:.5%})",
            flush=True,
        )
        write_result()

    selected = write_result()
    print(f"selected round={selected['round']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
