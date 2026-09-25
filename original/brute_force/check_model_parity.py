from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence

from solitaire.game import DeckConfig
from solitaire.player import FiveParameterPlayer, PARAMETER_NAMES
from solitaire.train import moves_to_unseen_positions

from .solvability import (
    BitsetBuilder,
    build_initial_state,
    expected_canonical_deal_count,
    iter_suit_pattern_rank_deals,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Python and native model outcomes on canonical deals"
    )
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("brute_force/native_solver"))
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    return parser


def load_model(path: Path) -> list[float]:
    report = json.loads(path.read_text(encoding="ascii"))
    feature_names = report.get("feature_names")
    selected = report.get("selected_parameters")
    if feature_names != list(PARAMETER_NAMES) or not isinstance(selected, dict):
        raise ValueError("model source does not contain the current feature set")
    return [float(selected[name]) for name in PARAMETER_NAMES]


def python_outcomes(n: int, weights: Sequence[float], max_steps: int) -> bytes:
    config = DeckConfig(n=n, k=2)
    player = FiveParameterPlayer(weights)
    outcomes = BitsetBuilder()
    for deal in iter_suit_pattern_rank_deals(n):
        state = build_initial_state(deal, config)
        seen = {state.position_key()}
        while not state.is_won() and state.steps < max_steps:
            moves = moves_to_unseen_positions(state, seen)
            move = player.choose_move(state, greedy=True, moves=moves)
            if move is None:
                break
            state = state.apply_move(move)
            seen.add(state.position_key())
        outcomes.append(state.is_won())
    expected = expected_canonical_deal_count(n, 2)
    if outcomes.count != expected:
        raise AssertionError(f"enumerated {outcomes.count} deals; expected {expected}")
    return bytes(outcomes.data)


def native_outcomes(
    binary: Path,
    n: int,
    weights: Sequence[float],
    threads: int,
    max_steps: int,
) -> bytes:
    encoded = ",".join(f"{weight:.17g}" for weight in weights)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "outcomes.bits"
        completion = Path(directory) / "completion.bits"
        subprocess.run(
            [
                str(binary),
                "--n",
                str(n),
                "--threads",
                str(threads),
                "--model-only",
                "--model-max-steps",
                str(max_steps),
                "--model-weights",
                encoded,
                "--output",
                str(output),
                "--completion",
                str(completion),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output.read_bytes()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n < 2 or args.n > 2:
        raise ValueError("the complete parity check is currently limited to n=2")
    weights = load_model(args.model_source)
    python_bits = python_outcomes(args.n, weights, args.max_steps)
    native_bits = native_outcomes(
        args.binary,
        args.n,
        weights,
        args.threads,
        args.max_steps,
    )
    if python_bits != native_bits:
        differing = sum(
            bin(left ^ right).count("1")
            for left, right in zip(python_bits, native_bits)
        )
        raise AssertionError(f"Python/native outcomes differ on {differing} deals")
    wins = sum(bin(byte).count("1") for byte in python_bits)
    deals = expected_canonical_deal_count(args.n, 2)
    print(f"python_native_parity true")
    print(f"canonical_deals {deals}")
    print(f"model_wins {wins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
