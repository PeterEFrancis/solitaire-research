from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Optional, Sequence

from .game import DeckConfig, GameState, Move, new_game
from .parameter_store import (
    DEFAULT_PARAMETER_PATH,
    load_parameter_data,
    load_parameters_for_config,
    save_parameter_record,
    find_record,
)
from .player import FEATURE_COUNT, PARAMETER_NAMES, FiveParameterPlayer


@dataclass(frozen=True)
class EpisodeResult:
    won: bool
    steps: int
    progress: float
    reward: float


@dataclass(frozen=True)
class EvaluationResult:
    games: int
    wins: int
    average_progress: float
    average_steps: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


class PolicyGradientTrainer:
    """Gradient descent on the negative reward-weighted log-probability loss."""

    def __init__(
        self,
        player: FiveParameterPlayer,
        *,
        deck_config: Optional[DeckConfig] = None,
        learning_rate: float = 0.03,
        max_steps: int = 1000,
        progress_reward_weight: float = 0.1,
        reveal_reward_weight: float = 0.1,
        baseline_decay: float = 0.95,
        batch_size: int = 1,
        optimizer: str = "sgd",
        gradient_clip: Optional[float] = None,
        frozen_prefix: int = 0,
        trainable_indices: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if progress_reward_weight < 0 or reveal_reward_weight < 0:
            raise ValueError("reward weights must be nonnegative")
        if not 0 <= baseline_decay < 1:
            raise ValueError("baseline_decay must be in [0, 1)")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be 'sgd' or 'adam'")
        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive or None")
        if not 0 <= frozen_prefix <= FEATURE_COUNT:
            raise ValueError(f"frozen_prefix must be in [0, {FEATURE_COUNT}]")
        if trainable_indices is not None and any(
            not 0 <= index < FEATURE_COUNT for index in trainable_indices
        ):
            raise ValueError("trainable indices are out of range")

        self.player = player
        self.deck_config = deck_config or DeckConfig()
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.progress_reward_weight = progress_reward_weight
        self.reveal_reward_weight = reveal_reward_weight
        self.baseline_decay = baseline_decay
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.gradient_clip = gradient_clip
        self.frozen_prefix = frozen_prefix
        self.trainable_indices = (
            set(trainable_indices) if trainable_indices is not None else None
        )
        self.rng = random.Random(seed)
        self.baseline: Optional[float] = None
        self.optimizer_step = 0
        self.adam_first_moment = [0.0 for _ in range(FEATURE_COUNT)]
        self.adam_second_moment = [0.0 for _ in range(FEATURE_COUNT)]

    def train(self, episodes: int) -> list[EpisodeResult]:
        history: list[EpisodeResult] = []
        for batch_start in range(0, episodes, self.batch_size):
            batch_count = min(self.batch_size, episodes - batch_start)
            batch: list[tuple[EpisodeResult, tuple[float, ...]]] = []
            for _ in range(batch_count):
                deal_seed = self.rng.randrange(2**63)
                item = self.run_episode(deal_seed, collect_gradient=True)
                history.append(item[0])
                batch.append(item)

            if self.batch_size == 1:
                result, grad_sum = batch[0]
                baseline = self.update_baseline(result.reward)
                advantages = (result.reward - baseline,)
                grad_sums = (grad_sum,)
            elif batch_count > 1:
                rewards = [result.reward for result, _ in batch]
                reward_mean = sum(rewards) / batch_count
                reward_variance = sum(
                    (reward - reward_mean) ** 2 for reward in rewards
                ) / batch_count
                reward_scale = math.sqrt(reward_variance) + 1e-8
                advantages = tuple(
                    (reward - reward_mean) / reward_scale for reward in rewards
                )
                grad_sums = tuple(grad_sum for _, grad_sum in batch)
                self.baseline = reward_mean
            else:
                continue

            objective_gradient = [0.0 for _ in range(FEATURE_COUNT)]
            for advantage, grad_sum in zip(advantages, grad_sums):
                for i, grad in enumerate(grad_sum):
                    objective_gradient[i] += advantage * grad / batch_count
            self.apply_objective_gradient(objective_gradient)

        return history

    def apply_objective_gradient(self, gradient: list[float]) -> None:
        gradient[: self.frozen_prefix] = [0.0] * self.frozen_prefix
        if self.trainable_indices is not None:
            gradient = [
                value if i in self.trainable_indices else 0.0
                for i, value in enumerate(gradient)
            ]
        if self.gradient_clip is not None:
            norm = math.sqrt(sum(value * value for value in gradient))
            if norm > self.gradient_clip:
                scale = self.gradient_clip / norm
                gradient = [value * scale for value in gradient]

        if self.optimizer == "sgd":
            for i, value in enumerate(gradient):
                self.player.parameters[i] += self.learning_rate * value
            return

        self.optimizer_step += 1
        beta1 = 0.9
        beta2 = 0.999
        for i, value in enumerate(gradient):
            self.adam_first_moment[i] = (
                beta1 * self.adam_first_moment[i] + (1.0 - beta1) * value
            )
            self.adam_second_moment[i] = (
                beta2 * self.adam_second_moment[i] + (1.0 - beta2) * value * value
            )
            first = self.adam_first_moment[i] / (1.0 - beta1**self.optimizer_step)
            second = self.adam_second_moment[i] / (1.0 - beta2**self.optimizer_step)
            self.player.parameters[i] += (
                self.learning_rate * first / (math.sqrt(second) + 1e-8)
            )

    def run_episode(
        self,
        deal_seed: int,
        *,
        collect_gradient: bool,
        greedy: bool = False,
    ) -> tuple[EpisodeResult, tuple[float, ...]]:
        state = new_game(seed=deal_seed, config=self.deck_config)
        grad_sum = [0.0 for _ in range(FEATURE_COUNT)]
        seen_positions = {state.position_key()}

        while not state.is_won() and state.steps < self.max_steps:
            moves = moves_to_unseen_positions(state, seen_positions)
            decision = self.player.decide(state, greedy=greedy, moves=moves)
            if decision is None:
                break
            if collect_gradient:
                for i, grad in enumerate(decision.grad_log_probability):
                    grad_sum[i] += grad
            state = state.apply_move(decision.move)
            seen_positions.add(state.position_key())

        result = episode_result(
            state,
            self.progress_reward_weight,
            self.reveal_reward_weight,
        )
        return result, tuple(grad_sum)

    def update_baseline(self, reward: float) -> float:
        if self.baseline is None:
            self.baseline = reward
        else:
            self.baseline = (
                self.baseline_decay * self.baseline
                + (1.0 - self.baseline_decay) * reward
            )
        return self.baseline


def episode_result(
    state: GameState,
    progress_reward_weight: float,
    reveal_reward_weight: float = 0.0,
) -> EpisodeResult:
    won = state.is_won()
    progress = state.foundation_progress()
    reward = (
        (1.0 if won else 0.0)
        + progress_reward_weight * progress
        + reveal_reward_weight * tableau_reveal_progress(state)
    )
    return EpisodeResult(
        won=won,
        steps=state.steps,
        progress=progress,
        reward=reward,
    )


def evaluate(
    player: FiveParameterPlayer,
    *,
    deck_config: Optional[DeckConfig] = None,
    games: int = 100,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    greedy: bool = True,
) -> EvaluationResult:
    deck_config = deck_config or DeckConfig()
    rng = random.Random(seed)
    wins = 0
    total_progress = 0.0
    total_steps = 0

    for _ in range(games):
        state = new_game(seed=rng.randrange(2**63), config=deck_config)
        seen_positions = {state.position_key()}
        while not state.is_won() and state.steps < max_steps:
            moves = moves_to_unseen_positions(state, seen_positions)
            move = player.choose_move(state, greedy=greedy, moves=moves)
            if move is None:
                break
            state = state.apply_move(move)
            seen_positions.add(state.position_key())
        wins += int(state.is_won())
        total_progress += state.foundation_progress()
        total_steps += state.steps

    return EvaluationResult(
        games=games,
        wins=wins,
        average_progress=total_progress / games if games else 0.0,
        average_steps=total_steps / games if games else 0.0,
    )


def moves_to_unseen_positions(
    state: GameState,
    seen_positions: set[tuple[object, ...]],
) -> tuple[Move, ...]:
    return tuple(
        move
        for move in state.legal_moves()
        if state.apply_move(move).position_key() not in seen_positions
    )


def tableau_reveal_progress(state: GameState) -> float:
    cards_left = state.config.deck_size
    initially_hidden = 0
    for column in range(state.config.resolved_tableau_columns()):
        dealt = min(column + 1, cards_left)
        initially_hidden += max(0, dealt - 1)
        cards_left -= dealt
    if initially_hidden == 0:
        return 1.0
    hidden_now = sum(
        not card.face_up for pile in state.tableau for card in pile
    )
    return 1.0 - hidden_now / initially_hidden


def format_parameters(parameters: Sequence[float]) -> str:
    return ", ".join(
        f"{name}={value:.3f}" for name, value in zip(PARAMETER_NAMES, parameters)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--progress-reward-weight", type=float, default=1.0)
    parser.add_argument("--reveal-reward-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--frozen-prefix", type=int, default=0)
    parser.add_argument(
        "--trainable-parameter",
        action="append",
        choices=PARAMETER_NAMES,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n", type=int, default=13, help="cards per suit")
    parser.add_argument("--k", type=int, default=2, help="suits per color")
    parser.add_argument("--t", type=int, default=None, help="tableau piles")
    parser.add_argument("--suits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ranks", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tableau-columns", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--draw-count", type=int, default=1)
    parser.add_argument("--max-recycles", type=int, default=3)
    parser.add_argument("--parameter-file", type=Path, default=DEFAULT_PARAMETER_PATH)
    parser.add_argument("--no-load-parameters", action="store_true")
    parser.add_argument("--no-save-parameters", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    n = args.ranks if args.ranks is not None else args.n
    k = args.k
    if args.suits is not None:
        if args.suits % 2 != 0:
            raise ValueError("--suits must be even because there are two colors")
        k = args.suits // 2
    t = args.tableau_columns if args.tableau_columns is not None else args.t
    deck_config = DeckConfig(
        n=n,
        k=k,
        t=t,
        draw_count=args.draw_count,
        max_recycles=args.max_recycles,
    )
    stored_parameters = None
    if not args.no_load_parameters:
        stored_parameters = load_parameters_for_config(deck_config, args.parameter_file)
    player = FiveParameterPlayer(
        parameters=stored_parameters,
        temperature=args.temperature,
        rng=random.Random(args.seed + 1),
    )
    trainer = PolicyGradientTrainer(
        player,
        deck_config=deck_config,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        progress_reward_weight=args.progress_reward_weight,
        reveal_reward_weight=args.reveal_reward_weight,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        gradient_clip=args.gradient_clip,
        frozen_prefix=args.frozen_prefix,
        trainable_indices=(
            [PARAMETER_NAMES.index(name) for name in args.trainable_parameter]
            if args.trainable_parameter
            else None
        ),
        seed=args.seed,
    )

    before = evaluate(
        player,
        deck_config=deck_config,
        games=args.eval_games,
        max_steps=args.max_steps,
        seed=args.seed + 2,
    )
    if args.episodes:
        trainer.train(args.episodes)
    after = evaluate(
        player,
        deck_config=deck_config,
        games=args.eval_games,
        max_steps=args.max_steps,
        seed=args.seed + 2,
    )

    saved = False
    if args.episodes and not args.no_save_parameters:
        existing_record = find_record(deck_config, load_parameter_data(args.parameter_file))
        existing_metrics = existing_record.get("metrics", {}) if existing_record else {}
        existing_win_rate = existing_metrics.get("win_rate")
        existing_progress = existing_metrics.get("average_progress", 0.0)
        after_score = (after.win_rate, after.average_progress)
        existing_score = (float(existing_win_rate or 0.0), float(existing_progress))
        if existing_win_rate is None or after_score >= existing_score:
            save_parameter_record(
                deck_config,
                player.parameters,
                path=args.parameter_file,
                evaluation=after,
                max_steps=args.max_steps,
                seed=args.seed + 2,
            )
            saved = True

    print(
        "Deck: "
        f"n={deck_config.n} k={deck_config.k} "
        f"t={deck_config.resolved_tableau_columns()} "
        f"suits={deck_config.suits} ranks={deck_config.ranks}"
    )
    print(
        "Parameter source: "
        f"{args.parameter_file if stored_parameters is not None else 'built-in defaults'}"
    )
    print(f"Before: wins={before.wins}/{before.games} win_rate={before.win_rate:.3f} progress={before.average_progress:.3f}")
    print(f"After:  wins={after.wins}/{after.games} win_rate={after.win_rate:.3f} progress={after.average_progress:.3f}")
    print(f"Parameters: {format_parameters(player.parameters)}")
    if args.episodes and not args.no_save_parameters:
        print(f"Parameter file: {'saved' if saved else 'kept existing better record'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
