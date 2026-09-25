from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from .game import DeckConfig
from .player import DEFAULT_PARAMETERS, FEATURE_COUNT, PARAMETER_NAMES


DEFAULT_PARAMETER_PATH = Path(__file__).with_name("parameters.json")
PARAMETER_FILE_VERSION = 1


def load_parameter_data(path: Path = DEFAULT_PARAMETER_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"version": PARAMETER_FILE_VERSION, "records": []}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"parameter file must contain a JSON object: {path}")
    data.setdefault("version", PARAMETER_FILE_VERSION)
    data.setdefault("records", [])
    return data


def load_parameters_for_config(
    config: DeckConfig,
    path: Path = DEFAULT_PARAMETER_PATH,
) -> Optional[list[float]]:
    record = find_record(config, load_parameter_data(path))
    if record is None:
        return None
    parameters = record.get("parameters")
    if isinstance(parameters, list):
        values = [float(value) for value in parameters]
        if len(values) < FEATURE_COUNT:
            values.extend(DEFAULT_PARAMETERS[len(values) :])
    elif isinstance(parameters, dict):
        values = [
            float(parameters.get(name, default))
            for name, default in zip(PARAMETER_NAMES, DEFAULT_PARAMETERS)
        ]
    else:
        raise ValueError("parameter record must contain a list or object")
    if len(values) != FEATURE_COUNT:
        raise ValueError(f"expected {FEATURE_COUNT} stored parameters")
    return values


def save_parameter_record(
    config: DeckConfig,
    parameters: Sequence[float],
    *,
    path: Path = DEFAULT_PARAMETER_PATH,
    evaluation: Optional[Any] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    greedy: bool = True,
) -> None:
    if len(parameters) != FEATURE_COUNT:
        raise ValueError(f"expected {FEATURE_COUNT} parameters")

    data = load_parameter_data(path)
    records = data.setdefault("records", [])
    record = {
        "n": config.n,
        "k": config.k,
        "t": config.resolved_tableau_columns(),
        "parameters": {
            name: float(value) for name, value in zip(PARAMETER_NAMES, parameters)
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if evaluation is not None:
        record["metrics"] = {
            "wins": int(evaluation.wins),
            "games": int(evaluation.games),
            "win_rate": float(evaluation.win_rate),
            "average_progress": float(evaluation.average_progress),
            "average_steps": float(evaluation.average_steps),
            "max_steps": max_steps,
            "seed": seed,
            "greedy": greedy,
        }

    index = find_record_index(config, data)
    if index is None:
        records.append(record)
    else:
        records[index] = record

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
    temp_path.replace(path)


def find_record(config: DeckConfig, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    index = find_record_index(config, data)
    if index is None:
        return None
    return data["records"][index]


def find_record_index(config: DeckConfig, data: dict[str, Any]) -> Optional[int]:
    target = (config.n, config.k, config.resolved_tableau_columns())
    for index, record in enumerate(data.get("records", [])):
        record_key = (record.get("n"), record.get("k"), record.get("t"))
        if record_key == target:
            return index
    return None
