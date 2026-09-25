from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

from solitaire.game import DeckConfig

from .solvability import (
    SUIT_PATTERN_RANK_ENUMERATION,
    expected_canonical_deal_count,
    expected_canonical_suit_pattern_count,
    symmetry_orbit_size,
)


def write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


def update_summary(results_dir: Path, k: int) -> None:
    completed = []
    for metadata_path in results_dir.glob(f"k{k}_n*_t*.json"):
        completed.append(json.loads(metadata_path.read_text(encoding="ascii")))
    completed.sort(key=lambda item: int(item["config"]["n"]))
    if not completed:
        return

    next_n = int(completed[-1]["config"]["n"]) + 1
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "method": "exhaustive classification modulo color/suit automorphisms",
        "completed": completed,
        "stopped": {
            "n": next_n,
            "raw_deals": math.factorial(2 * next_n * k),
            "canonical_deals": expected_canonical_deal_count(next_n, k),
            "reason": "first uncompleted size; exhaustive enumeration is too large",
        },
    }
    write_json(results_dir / f"k{k}_summary.json", summary)


def parse_args() -> argparse.Namespace:
    default_results = Path(__file__).with_name("results")
    parser = argparse.ArgumentParser(description="Finalize a native solvability run")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--completion", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k != 2:
        raise ValueError("native finalization currently requires k=2")
    config = DeckConfig(n=args.n, k=args.k)
    stem = f"k{args.k}_n{args.n}_t{config.resolved_tableau_columns()}"
    raw_path = args.raw or args.results_dir / f"{stem}.native.bits.partial"
    completion_path = (
        args.completion
        or args.results_dir / f"{stem}.native.patterns.complete"
    )

    expected_patterns = expected_canonical_suit_pattern_count(args.n)
    completion = completion_path.read_bytes()
    if len(completion) != expected_patterns or any(value != 1 for value in completion):
        completed = sum(value == 1 for value in completion)
        raise ValueError(f"native run is incomplete: {completed}/{expected_patterns} patterns")

    canonical_deals = expected_canonical_deal_count(args.n, args.k)
    bits = raw_path.read_bytes()
    if len(bits) != (canonical_deals + 7) // 8:
        raise ValueError("native bitset has the wrong length")
    padding = len(bits) * 8 - canonical_deals
    if padding and bits[-1] >> (8 - padding):
        raise ValueError("native bitset has nonzero padding")
    popcounts = tuple(bin(value).count("1") for value in range(256))
    solvable = sum(popcounts[byte] for byte in bits)
    compressed = gzip.compress(bits, compresslevel=9, mtime=0)
    compressed_path = args.results_dir / f"{stem}.solvable.bits.gz"
    metadata_path = args.results_dir / f"{stem}.json"
    write_bytes(compressed_path, compressed)

    orbit_size = symmetry_orbit_size(args.k)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n": args.n,
            "k": args.k,
            "t": config.resolved_tableau_columns(),
            "draw_count": config.draw_count,
            "max_recycles": config.max_recycles,
        },
        "raw_deals": math.factorial(config.deck_size),
        "canonical_deals": canonical_deals,
        "symmetry": {
            "description": (
                "Permute suits within each color and optionally exchange the colors"
            ),
            "orbit_size": orbit_size,
        },
        "deal_order": {
            "card_id": "suit * n + (rank - 1)",
            "sequence": "tableau deal order, then stock draw order",
            "enumeration_id": SUIT_PATTERN_RANK_ENUMERATION,
            "enumeration": (
                "canonical suit patterns in lexicographic order, then the "
                "lexicographic Cartesian product of rank permutations by suit"
            ),
        },
        "canonical_suit_patterns": expected_patterns,
        "rank_assignments_per_pattern": math.factorial(args.n) ** 4,
        "solvable_canonical_deals": solvable,
        "unsolvable_canonical_deals": canonical_deals - solvable,
        "solvable_raw_deals": solvable * orbit_size,
        "unsolvable_raw_deals": (canonical_deals - solvable) * orbit_size,
        "solvability_rate": solvable / canonical_deals,
        "solver": {
            "engine": "brute_force/native_solver.cpp",
            "classification": "exact state-graph reachability",
            "tableau_pile_symmetry_in_state_cache": True,
        },
        "bitset": {
            "file": compressed_path.name,
            "bits": canonical_deals,
            "bit_order": "deal i is bit (i % 8), least-significant bit first",
            "uncompressed_bytes": len(bits),
            "compressed_bytes": len(compressed),
            "sha256": hashlib.sha256(bits).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        },
    }
    write_json(metadata_path, metadata)
    update_summary(args.results_dir, args.k)
    print(
        f"n={args.n}: {solvable:,}/{canonical_deals:,} canonical deals solvable "
        f"({solvable / canonical_deals:.6%})"
    )


if __name__ == "__main__":
    main()
