"""Independently verify a completed variant study without running any games.

Checks artifact integrity, frozen selections, feature masks, indexed outcomes,
portfolio construction, and reported statistics. This verifies consistency of
the saved evidence, not random-number independence or global policy optimality.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any
import zlib

from solitaire.player import PARAMETER_NAMES


SOURCE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PREFIX = SOURCE_DIRECTORY / "results" / "variant-study"
PROFILES = {
    "draw1_limited": (1, 3, True),
    "draw1_unlimited": (1, None, True),
    "draw3_limited": (3, 3, True),
    "draw3_unlimited": (3, None, True),
    "vegas": (1, 0, False),
}
SIMPLE_FEATURES = {
    "safe_foundation", "reveal_hidden", "tableau_build", "stock_action",
    "waste_to_tableau", "reveal_depth", "recycle", "productive_stack_length",
}
UNKNOWN_FEATURES = {
    "revealed_card_low_rank", "revealed_card_foundation_ready", "draw_playable",
    "empty_king_queen_access", "next_foundation_moves", "next_reveal_moves",
    "revealed_card_tableau_moves", "revealed_card_foundation_distance",
    "draw_foundation_ready", "draw_tableau_moves", "next_empty_source_moves",
}
WASTE_HISTORY_FEATURES = {
    "waste_unlocks_playable", "waste_unlocks_foundation_ready", "waste_unlocks_tableau_moves",
}
VISIBLE_FEATURES = set(PARAMETER_NAMES) - UNKNOWN_FEATURES - WASTE_HISTORY_FEATURES
Z_95 = NormalDist().inv_cdf(0.975)


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise VerificationError(f"{path}: nonfinite JSON number {value}")

    result = json.loads(path.read_text(), object_pairs_hook=object_pairs, parse_constant=reject_constant)
    require(isinstance(result, dict), f"{path}: expected a JSON object")
    return result


def compare(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{label}: expected an object")
        for key, value in expected.items():
            require(key in actual, f"{label}: missing {key}")
            compare(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
                f"{label}: wrong array length")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{label}[{index}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
                f"{label}: {actual!r} != {expected!r}")
    elif type(expected) is int and type(actual) is float:
        # Training candidates can contain integer literals before the frozen
        # policy vector serializes their numerically identical float values.
        require(math.isfinite(actual) and actual == expected,
                f"{label}: {actual!r} != {expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected,
                f"{label}: {actual!r} != {expected!r}")


def integer(value: Any, lower: int, upper: int, label: str) -> None:
    require(type(value) is int and lower <= value <= upper,
            f"{label}: expected integer in [{lower}, {upper}]")


def verify_weights(weights: dict[str, Any], profile: str, label: str,
                   allowed: set[str] | None = None) -> None:
    require(isinstance(weights, dict) and set(weights) == set(PARAMETER_NAMES),
            f"{label}: expected all 42 named features")
    for name, value in weights.items():
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value), f"{label}.{name}: invalid weight")
        if allowed is not None and name not in allowed:
            require(value == 0, f"{label}.{name}: feature outside allowed information mask")
    if PROFILES[profile][1] is None:
        for name in ("recycle_pressure", "waste_play_recycle_pressure"):
            require(weights[name] == 0, f"{label}.{name}: unlimited pressure must be disabled")


def normalized(weights: dict[str, Any], profile: str) -> dict[str, Any]:
    result = dict(weights)
    if PROFILES[profile][1] is None:
        result.update(recycle_pressure=0.0, waste_play_recycle_pressure=0.0)
    return result


def config_for(profile: str) -> dict[str, Any]:
    draw, recycles, splitting = PROFILES[profile]
    return dict(n=13, k=2, t=7, draw_count=draw, max_recycles=recycles,
                allow_tableau_stack_splitting=splitting)


def expected_statistics(wins: int, cards: bytes) -> dict[str, Any]:
    n = len(cards)
    count = sum(cards)
    squares = sum(card * card for card in cards)
    rate = wins / n
    denominator = 1 + Z_95 * Z_95 / n
    center = (rate + Z_95 * Z_95 / (2 * n)) / denominator
    radius = Z_95 * math.sqrt(rate * (1 - rate) / n + Z_95 * Z_95 / (4 * n * n)) / denominator
    mean = count / n
    # Compute the centered sum directly, independently of the driver's moments.
    sum_squared_deviations = math.fsum((card - mean) ** 2 for card in cards)
    margin = 1.96 * math.sqrt(sum_squared_deviations / (n * max(1, n - 1)))
    return {
        "deals": n, "wins": wins, "foundation_cards": count,
        "foundation_cards_squared": squares, "win_rate": rate,
        "win_rate_wilson_95_ci": [max(0.0, center - radius), min(1.0, center + radius)],
        "mean_foundation_cards": mean,
        "mean_foundation_cards_95_ci": [mean - margin, mean + margin],
        "mean_net_dollars": 5 * mean - 52,
        "mean_net_dollars_95_ci": [5 * (mean - margin) - 52, 5 * (mean + margin) - 52],
    }


def expected_paired(candidate: bytes, baseline: bytes) -> dict[str, Any]:
    require(len(candidate) == len(baseline), "paired arrays have different lengths")
    n = len(candidate)
    differences = [left - right for left, right in zip(candidate, baseline)]
    b, c = differences.count(1), differences.count(-1)
    mean = sum(differences) / n
    variance = math.fsum((value - mean) ** 2 for value in differences) / n
    se = math.sqrt(variance / n)
    return {
        "deals": n, "b_candidate_wins_baseline_loses": b,
        "c_baseline_wins_candidate_loses": c, "discordant_deals": b + c,
        "net_wins": b - c, "delta_percentage_points": 100 * mean,
        "standard_error_percentage_points": 100 * se,
        "paired_normal_95_ci_percentage_points": [100 * (mean - Z_95 * se), 100 * (mean + Z_95 * se)],
    }


def read_outcome(row: dict[str, Any], kind: str, directory: Path, deals: int, label: str) -> bytes:
    relative = Path(row[kind + "_file"])
    require(not relative.is_absolute(), f"{label}: outcome path must be relative")
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()), f"{label}: outcome path escapes results directory")
    compressed = path.read_bytes()
    data = gzip.decompress(compressed)  # gzip CRC and stream length are validated here.
    require(len(data) == deals, f"{label}: expected {deals} bytes, found {len(data)}")
    require(hashlib.sha256(data).hexdigest() == row[kind + "_sha256"], f"{label}: outcome SHA-256 mismatch")
    if kind + "_compressed_sha256" in row:
        require(hashlib.sha256(compressed).hexdigest() == row[kind + "_compressed_sha256"],
                f"{label}: compressed SHA-256 mismatch")
    require(all(value <= (1 if kind == "wins" else 52) for value in data),
            f"{label}: invalid outcome byte")
    return data


def verify_study(prefix: Path = DEFAULT_PREFIX, *, binary: Path | None = None) -> list[str]:
    prefix = prefix.resolve()
    directory = prefix.parent
    paths = {kind: Path(str(prefix) + suffix) for kind, suffix in (
        ("protocol", ".protocol.json"), ("frozen", ".selected-policies.json"),
        ("training", ".training.json"), ("confirmation", ".confirmation.json"),
    )}
    documents = {name: read_json(path) for name, path in paths.items()}
    protocol, frozen, training, report = (documents[key] for key in ("protocol", "frozen", "training", "confirmation"))
    for name, document in documents.items():
        compare(document["schema_version"], 1, f"{name}.schema_version")
        if name != "protocol":
            require(document.get("complete") is True, f"{name}: study is incomplete")
    compare(report["protocol"], paths["protocol"].name, "confirmation.protocol")
    compare(report["frozen_policies_sha256"], sha256(paths["frozen"]), "confirmation.frozen_policies_sha256")
    options = protocol["options"]
    profiles = options["profiles"]
    require(isinstance(profiles, list) and profiles and len(set(profiles)) == len(profiles)
            and set(profiles) <= set(PROFILES), "protocol: invalid profiles")
    for name in ("frozen", "training", "confirmation"):
        require(set(documents[name]["variants"]) == set(profiles), f"{name}: profile set differs from protocol")
    for name in ("train_deals", "validation_deals", "test_deals", "max_steps", "threads"):
        integer(options[name], 1, 2**63 - 1, f"protocol.options.{name}")
    require(set(protocol["seeds"]) == {"train", "validation", "test"}, "protocol: unexpected seed fields")
    for name, seed in protocol["seeds"].items():
        integer(seed, 0, 2**64 - 1, f"protocol.seeds.{name}")
    require(len(set(protocol["seeds"].values())) == 3, "training, validation, and test seeds must differ")
    if "evaluation_options" in report:
        compare(report["evaluation_options"], dict(deals=options["test_deals"], seed=protocol["seeds"]["test"],
                                                   max_steps=options["max_steps"], threads=options["threads"]),
                "confirmation.evaluation_options")

    # A local rebuild may target a different platform. The recorded binary is
    # checked only when explicitly requested; source and manifest still agree.
    available_hashes = 0
    for path, expected, label in (
        (SOURCE_DIRECTORY / "native_solver.cpp", protocol["native_source_sha256"], "native source"),
        (SOURCE_DIRECTORY / "variant_study.py", protocol["experiment_source_sha256"], "experiment source"),
    ):
        if path.is_file():
            compare(sha256(path), expected, label + " SHA-256")
            available_hashes += 1
    build_path = Path(str(prefix) + ".build.json")
    if build_path.is_file():
        build = read_json(build_path)
        compare(build["source_sha256"], protocol["native_source_sha256"], "build manifest source SHA-256")
        compare(build["binary_sha256"], protocol["binary_sha256"], "build manifest binary SHA-256")
    if binary is not None:
        require(binary.is_file(), f"explicit binary does not exist: {binary}")
        compare(sha256(binary), protocol["binary_sha256"], "explicit binary SHA-256")
    for name, filename, key in (
        ("stage8", "k2_n13_t7.tuning-stage8.json", "selected_parameters"),
        ("simple_eight", "k2_n13_t7.human-frozen-policies.json", "simple_eight"),
    ):
        path = directory / filename
        if path.is_file():
            compare(sha256(path), protocol["baseline_source_sha256"][name], name + " baseline SHA-256")
            compare(protocol["baseline_parameters"][name], read_json(path)[key], name + " baseline parameters")
            available_hashes += 1
    audit_path = directory / "human-strategy-candidates.json"
    if audit_path.is_file():
        audit = read_json(audit_path)["audit"]
        require(set(audit["current_visible_features"]) == VISIBLE_FEATURES, "visible feature audit differs")
        require(set(audit["unknown_identity_features"]) == UNKNOWN_FEATURES, "unknown feature audit differs")
        require(set(audit["waste_history_features"]) == WASTE_HISTORY_FEATURES, "waste-memory feature audit differs")

    output = []
    deals, max_steps = options["test_deals"], options["max_steps"]
    total_rows = 0
    for profile in profiles:
        selected, trained, confirmed = (doc["variants"][profile] for doc in (frozen, training, report))
        for name, row in (("frozen", selected), ("training", trained), ("confirmation", confirmed)):
            compare(row["config"], config_for(profile), f"{profile}.{name}.config")
        families = {"full", "visible", "simple_eight"} | ({"profit"} if profile == "vegas" else set())
        require(set(selected["policies"]) == families, f"{profile}: wrong policy families")
        require(set(trained["validation"]) == families, f"{profile}: wrong validation families")
        compare(selected["parameters"], selected["policies"]["full"], f"{profile}.selected full parameters")
        for family, weights in selected["policies"].items():
            allowed = SIMPLE_FEATURES if family == "simple_eight" else VISIBLE_FEATURES if family == "visible" else None
            verify_weights(weights, profile, f"{profile}.{family}", allowed)
            validation = trained["validation"][family]
            require(isinstance(validation, list) and validation, f"{profile}.{family}: missing validation")
            for index, candidate in enumerate(validation):
                label = f"{profile}.{family}.validation[{index}]"
                compare(candidate["deals"], options["validation_deals"], label + ".deals")
                verify_weights(candidate["parameters"], profile, label, allowed)
                integer(candidate["wins"], 0, candidate["deals"], label + ".wins")
                integer(candidate["foundation_cards"], 0, 52 * candidate["deals"], label + ".foundation_cards")
                integer(candidate["steps"], 0, max_steps * candidate["deals"], label + ".steps")
            def objective(candidate):
                first, second = ("foundation_cards", "wins") if family == "profit" else ("wins", "foundation_cards")
                return candidate[first], candidate[second], -candidate["steps"]
            require(objective(validation[0]) == max(map(objective, validation)),
                    f"{profile}.{family}: frozen selection is not a validation winner")
            compare(weights, validation[0]["parameters"], f"{profile}.{family}: frozen validation choice")
        portfolio = selected["portfolio"]
        require(isinstance(portfolio, list) and 1 <= len(portfolio) <= 6, f"{profile}: invalid portfolio size")
        expected_members = []
        for weights in ([row["parameters"] for row in trained["validation"]["full"][:3]]
                        + [normalized(protocol["baseline_parameters"]["stage8"], profile),
                           selected["policies"]["visible"], selected["policies"]["simple_eight"]]):
            if weights not in expected_members:
                expected_members.append(weights)
        compare(portfolio, expected_members, f"{profile}.frozen portfolio")
        for index, weights in enumerate(portfolio):
            verify_weights(weights, profile, f"{profile}.portfolio[{index}]")

        weights_by_name = {
            "stage8_baseline": normalized(protocol["baseline_parameters"]["stage8"], profile),
            "simple_eight_baseline": normalized(protocol["baseline_parameters"]["simple_eight"], profile),
            **selected["policies"], **{f"portfolio_{i}": weights for i, weights in enumerate(portfolio)},
        }
        rows = confirmed["evaluations"]
        require(set(rows) == set(weights_by_name) | {"portfolio"}, f"{profile}: unexpected evaluation rows")
        raw, duplicate_policies = {}, {}
        for name, row in rows.items():
            label = f"{profile}.{name}"
            wins = read_outcome(row, "wins", directory, deals, label + ".wins")
            cards = read_outcome(row, "foundations", directory, deals, label + ".foundations")
            require(all(bool(win) == (count == 52) for win, count in zip(wins, cards)),
                    label + ": foundation count 52 must occur exactly on wins")
            compare(row, expected_statistics(sum(wins), cards), label)
            if name != "portfolio":
                integer(row["steps"], 0, max_steps * deals, label + ".steps")
                integer(row["cutoffs"], 0, deals - row["wins"], label + ".cutoffs")
                require(row["steps"] >= row["cutoffs"] * max_steps, label + ": cutoff count exceeds available steps")
                compare(row["mean_steps"], row["steps"] / deals, label + ".mean_steps")
                weights = weights_by_name[name]
                verify_weights(weights, profile, label + ".parameters",
                               SIMPLE_FEATURES if name == "simple_eight_baseline" else None)
                signature = tuple(weights[key] for key in PARAMETER_NAMES)
                if signature in duplicate_policies:
                    require((wins, cards) == duplicate_policies[signature], label + ": identical policies have different outcomes")
                duplicate_policies[signature] = wins, cards
            raw[name] = wins, cards
            total_rows += 1
        for name, row in rows.items():
            compare(row["paired_vs_stage8"], expected_paired(raw[name][0], raw["stage8_baseline"][0]),
                    f"{profile}.{name}.paired_vs_stage8")
        compare(rows["simple_eight"]["paired_vs_simple_baseline"],
                expected_paired(raw["simple_eight"][0], raw["simple_eight_baseline"][0]),
                profile + ".simple_eight.paired_vs_simple_baseline")
        bank = rows["portfolio"]
        member_names = [f"portfolio_{i}" for i in range(len(portfolio))]
        compare(bank["members"], member_names, profile + ".portfolio.members")
        compare(bank["attempts_per_deal"], len(member_names), profile + ".portfolio.attempts_per_deal")
        require(raw["portfolio"][0] == bytes(any(values) for values in zip(*(raw[name][0] for name in member_names))),
                profile + ": portfolio wins are not member OR")
        require(raw["portfolio"][1] == bytes(max(values) for values in zip(*(raw[name][1] for name in member_names))),
                profile + ": portfolio foundations are not member maximum")
        compare(bank["paired_vs_selected_full"], expected_paired(raw["portfolio"][0], raw["full"][0]),
                profile + ".portfolio.paired_vs_selected_full")
        output.append(f"verified {profile}: {len(rows)} policies, {deals:,} deals each, portfolio {bank['wins']:,} wins")
    output.append(f"verified {total_rows} policy rows; {available_hashes} source/baseline hashes match; binary checked={'yes' if binary is not None else 'no'}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX,
                        help="artifact prefix, excluding .confirmation.json")
    parser.add_argument("--binary", type=Path,
                        help="optionally verify this exact recorded binary; omit for portable source/data checks")
    args = parser.parse_args(argv)
    try:
        messages = verify_study(args.prefix, binary=args.binary)
    except (ValueError, KeyError, TypeError, OSError, EOFError, OverflowError, zlib.error) as error:
        parser.exit(1, f"verification failed: {error}\n")
    print("\n".join(messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
