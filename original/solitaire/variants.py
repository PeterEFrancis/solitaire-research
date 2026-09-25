"""Named standard-deck rules and explicit loading of study-selected policies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .game import DeckConfig
from .player import PARAMETER_NAMES


DEFAULT_VARIANT_POLICY_PATH = (
    Path(__file__).resolve().parent.parent
    / "brute_force/results/variant-study.selected-policies.json"
)
VARIANT_CONFIGS = {
    "draw1_limited": DeckConfig(n=13, k=2, t=7),
    "draw1_unlimited": DeckConfig(n=13, k=2, t=7, max_recycles=None),
    "draw3_limited": DeckConfig(n=13, k=2, t=7, draw_count=3),
    "draw3_unlimited": DeckConfig(n=13, k=2, t=7, draw_count=3, max_recycles=None),
    "vegas": DeckConfig(n=13, k=2, t=7, max_recycles=0, allow_tableau_stack_splitting=False),
}
POLICY_NAMES = ("full", "simple_eight", "visible", "profit")


def variant_config(variant_id: str) -> DeckConfig:
    try:
        return VARIANT_CONFIGS[variant_id]
    except KeyError as error:
        raise ValueError(f"unknown variant {variant_id!r}; choose from {tuple(VARIANT_CONFIGS)}") from error


def config_record(config: DeckConfig) -> dict[str, Any]:
    return {
        "n": config.n, "k": config.k, "t": config.resolved_tableau_columns(),
        "draw_count": config.draw_count, "max_recycles": config.max_recycles,
        "allow_tableau_stack_splitting": config.allow_tableau_stack_splitting,
    }


def parse_policy_parameters(value: object) -> list[float]:
    """Read named weights in canonical order; intentionally omitted weights are zero."""
    if not isinstance(value, dict):
        raise ValueError("policy parameters must be a feature-name/weight object")
    unknown = set(value) - set(PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown policy features: {sorted(unknown, key=str)!r}")
    result = []
    for name in PARAMETER_NAMES:
        weight = value.get(name, 0.0)
        if isinstance(weight, bool) or not isinstance(weight, (float, int)):
            raise ValueError(f"feature {name!r} must have a finite numeric weight")
        try:
            weight = float(weight)
        except OverflowError as error:
            raise ValueError(f"feature {name!r} must have a finite numeric weight") from error
        if not math.isfinite(weight):
            raise ValueError(f"feature {name!r} must have a finite numeric weight")
        result.append(weight)
    return result


def load_variant_policy(
    variant_id: str,
    *,
    policy: str = "full",
    path: Path = DEFAULT_VARIANT_POLICY_PATH,
) -> tuple[DeckConfig, list[float]]:
    """Load an explicit named profile, refusing mismatched rules or missing profiles.

    Schema 1 stores records in ``variants[variant_id]``. Each record has
    ``config``, ``parameters`` (the selected full policy), and ``policies``
    containing named full/simple_eight/visible/profit parameter maps. The full
    policy can be represented by parameters alone; if both copies exist they
    must agree. This loader never falls back to another variant's weights.
    """
    config = variant_config(variant_id)
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICY_NAMES}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("expected variant policy schema_version 1")
    records = data.get("variants")
    if not isinstance(records, dict) or variant_id not in records:
        raise ValueError(f"no selected policy record for variant {variant_id!r}")
    record = records[variant_id]
    if not isinstance(record, dict):
        raise ValueError("variant record must be an object")
    recorded_config = record.get("config")
    expected_config = config_record(config)
    if not isinstance(recorded_config, dict) or any(
        name not in recorded_config or recorded_config[name] != expected
        or type(recorded_config[name]) is not type(expected)
        for name, expected in expected_config.items()
    ):
        raise ValueError(f"stored rules do not match variant {variant_id!r}")
    named = record.get("policies", {})
    if not isinstance(named, dict):
        raise ValueError("named policies must be an object")
    if policy == "full":
        if "parameters" not in record:
            raise ValueError(f"variant {variant_id!r} has no selected full policy")
        weights = parse_policy_parameters(record["parameters"])
        if "full" in named and weights != parse_policy_parameters(named["full"]):
            raise ValueError("full policy disagrees with selected parameters")
    else:
        if policy not in named:
            raise ValueError(f"variant {variant_id!r} has no {policy!r} policy")
        weights = parse_policy_parameters(named[policy])
    return config, weights
