"""Play a selected variant policy and print its move trace.

Example: python3 -m solitaire.play_variant draw3_unlimited --policy full --seed 7
Use --policy portfolio for full-information restarts, or --format json for a
machine-readable trace. Python random.Random deals differ from native benchmark
deals with the same numeric seed. This command reproduces native move pruning
and greedy tie order, but does not reproduce the native shuffle algorithm.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .game import GameState, Move, MoveKind, new_game
from .player import FiveParameterPlayer, PARAMETER_NAMES
from .variants import (
    DEFAULT_VARIANT_POLICY_PATH, POLICY_NAMES, VARIANT_CONFIGS, config_record,
    load_variant_policy, parse_policy_parameters,
)


@dataclass
class PolicyRun:
    final_state: GameState
    trace: list[dict[str, Any]]
    termination: str


def native_policy_moves(state: GameState) -> tuple[Move, ...]:
    """Preserve generator order while applying native empty-column pruning."""
    result = []
    used_empty = set()
    for move in state.legal_moves():
        if move.kind in (MoveKind.WASTE_TO_TABLEAU, MoveKind.TABLEAU_TO_TABLEAU):
            if not state.tableau[move.dest_index]:
                if (
                    move.kind == MoveKind.TABLEAU_TO_TABLEAU
                    and move.count == len(state.tableau[move.source_index])
                ):
                    continue
                group = (move.kind, move.source_index, move.count)
                if group in used_empty:
                    continue
                used_empty.add(group)
        result.append(move)
    return tuple(result)


def describe_move(state: GameState, move: Move, next_state: GameState) -> str:
    if move.kind == MoveKind.DRAW:
        count = min(state.config.draw_count, len(state.stock))
        return f"Draw {count}; expose {next_state.waste[-1]}"
    if move.kind == MoveKind.RECYCLE:
        return "Recycle the waste into the stock"
    if move.kind == MoveKind.WASTE_TO_FOUNDATION:
        return f"Move {state.waste[-1]} from waste to foundation"
    if move.kind == MoveKind.TABLEAU_TO_FOUNDATION:
        return f"Move {state.tableau[move.source_index][-1]} from T{move.source_index + 1} to foundation"
    if move.kind == MoveKind.WASTE_TO_TABLEAU:
        return f"Move {state.waste[-1]} from waste to T{move.dest_index + 1}"
    card = state.tableau[move.source_index][-move.count]
    return f"Move {move.count}-card run headed by {card} from T{move.source_index + 1} to T{move.dest_index + 1}"


def play_policy(initial: GameState, weights: Sequence[float], max_steps: int = 1000) -> PolicyRun:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    player = FiveParameterPlayer(parameters=weights)
    state = initial
    seen = {state.position_key()}
    trace = []
    while not state.is_won() and len(trace) < max_steps:
        moves = tuple(
            move for move in native_policy_moves(state)
            if state.apply_move(move).position_key() not in seen
        )
        decision = player.decide(state, greedy=True, moves=moves)
        if decision is None:
            return PolicyRun(state, trace, "no_unseen_moves")
        next_state = state.apply_move(decision.move)
        trace.append({
            "step": len(trace) + 1,
            "move": asdict(decision.move),
            "description": describe_move(state, decision.move, next_state),
            "score": player.score(decision.features),
            "foundation_cards": sum(next_state.foundations),
            "stock_cards": len(next_state.stock),
            "waste_cards": len(next_state.waste),
        })
        state = next_state
        seen.add(state.position_key())
    return PolicyRun(state, trace, "won" if state.is_won() else "move_budget")


def select_run(runs: Sequence[PolicyRun]) -> int:
    if not runs:
        raise ValueError("portfolio must contain at least one run")
    return max(range(len(runs)), key=lambda index: (
        runs[index].final_state.is_won(),
        sum(runs[index].final_state.foundations),
        -len(runs[index].trace),
    ))


def state_record(state: GameState) -> dict[str, Any]:
    def cards(packet):
        return [asdict(card) for card in packet]
    return {
        "tableau": [cards(pile) for pile in state.tableau],
        "stock": cards(state.stock), "waste": cards(state.waste),
        "foundations": list(state.foundations), "recycles_used": state.recycles_used,
    }


def build_report(variant: str, policy: str, policy_path: Path, seed: int, max_steps: int) -> dict[str, Any]:
    raw_source = policy_path.read_bytes()
    source = json.loads(raw_source)
    if source.get("complete") is False:
        raise ValueError("selected policy file is still incomplete")
    config, weights = load_variant_policy(
        variant, policy="full" if policy == "portfolio" else policy, path=policy_path
    )
    if policy == "portfolio":
        portfolio = source["variants"][variant].get("portfolio")
        if not isinstance(portfolio, list) or not portfolio:
            raise ValueError(f"variant {variant!r} has no frozen portfolio")
        bank = [parse_policy_parameters(member) for member in portfolio]
        member_names = [f"portfolio_{index}" for index in range(len(bank))]
    else:
        bank, member_names = [weights], [policy]
    if policy_path.read_bytes() != raw_source:
        raise ValueError("policy file changed while loading; retry with the completed file")
    initial = new_game(seed=seed, config=config)
    runs = [play_policy(initial, member, max_steps) for member in bank]
    selected_index = select_run(runs)
    selected = runs[selected_index]
    report = {
        "schema_version": 1,
        "variant": variant, "config": config_record(config), "policy": policy,
        "seed": seed, "shuffle": "Python random.Random; same seed does not reproduce native benchmark deals",
        "max_steps_per_attempt": max_steps, "attempts": len(runs),
        "information": (
            "Portfolio uses full-information restarts from the same deal and selects a winning trajectory, otherwise maximum foundation cards; ties prefer fewer moves."
            if policy == "portfolio" else
            "Full/profit policies may inspect hidden cards. Visible/simple_eight scores use the visible-feature restriction; cycle avoidance uses complete position keys."
        ),
        "trace_information": "JSON initial/final states include hidden card identities. Text move descriptions expose only the final waste top on a draw.",
        "policy_source": str(policy_path.resolve()),
        "policy_source_sha256": hashlib.sha256(raw_source).hexdigest(),
        "selected_member": member_names[selected_index],
        "won": selected.final_state.is_won(),
        "foundation_cards": sum(selected.final_state.foundations),
        "steps": len(selected.trace), "termination": selected.termination,
        "total_attempt_steps": sum(len(run.trace) for run in runs),
        "members": [
            {"name": name, "won": run.final_state.is_won(),
             "foundation_cards": sum(run.final_state.foundations),
             "steps": len(run.trace), "termination": run.termination,
             "parameters": dict(zip(PARAMETER_NAMES, member))}
            for name, member, run in zip(member_names, bank, runs)
        ],
        "initial_state": state_record(initial), "final_state": state_record(selected.final_state),
        "trace": selected.trace,
    }
    if variant == "vegas":
        report["net_dollars"] = 5 * report["foundation_cards"] - 52
    return report


def format_text(report: dict[str, Any]) -> str:
    outcome = "Won" if report["won"] else "Stopped"
    lines = [
        f"{report['variant']} / {report['policy']}: {outcome}; {report['foundation_cards']}/52 foundation cards in {report['steps']} moves.",
        f"Seed {report['seed']}: {report['shuffle']}.",
        report["information"],
        f"Selected {report['selected_member']}; {report['attempts']} attempt(s), {report['total_attempt_steps']} total moves; termination={report['termination']}.",
    ]
    if "net_dollars" in report:
        lines.append(f"Vegas return: ${report['net_dollars']:+d} ($5 per foundation card minus $52).")
    lines.append("Tableau labels in this trace are T1 through T7; JSON move indices start at 0.")
    lines.extend(f"{step['step']:4d}. {step['description']} (score {step['score']:.6g})" for step in report["trace"])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=tuple(VARIANT_CONFIGS))
    parser.add_argument("--policy", choices=(*POLICY_NAMES, "portfolio"), default="full")
    parser.add_argument("--policy-file", type=Path, default=DEFAULT_VARIANT_POLICY_PATH)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.output is not None and args.output.resolve() == args.policy_file.resolve():
        parser.error("trace output must not overwrite the selected policies")
    report = build_report(args.variant, args.policy, args.policy_file, args.seed, args.max_steps)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n" if args.format == "json" else format_text(report)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
        print(f"Trace saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
