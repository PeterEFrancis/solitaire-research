from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional

from solitaire.game import DeckConfig

from .solvability import (
    BitsetBuilder,
    ExactSolver,
    build_initial_state,
    expected_canonical_deal_count,
    iter_canonical_deals,
    symmetry_orbit_size,
)


DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively classify symmetry-canonical solitaire deals."
    )
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--start-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=20)
    parser.add_argument("--max-canonical-deals", type=int, default=1_000_000)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path: Path, data: object) -> None:
    encoded = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_bytes(path, encoded)


def classify_size(
    n: int,
    k: int,
    results_dir: Path,
    progress_every: int,
) -> dict[str, object]:
    config = DeckConfig(n=n, k=k)
    canonical_deals = expected_canonical_deal_count(n, k)
    orbit_size = symmetry_orbit_size(k)
    solver = ExactSolver()
    outcomes = BitsetBuilder()
    solvable = 0
    started = time.monotonic()

    for index, deal in enumerate(iter_canonical_deals(n, k), start=1):
        outcome = solver.is_solvable(build_initial_state(deal, config))
        outcomes.append(outcome)
        solvable += int(outcome)
        if progress_every > 0 and index % progress_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"n={n}: {index:,}/{canonical_deals:,} deals "
                f"({index / elapsed:,.0f}/s)",
                flush=True,
            )

    if outcomes.count != canonical_deals:
        raise AssertionError(
            f"enumerated {outcomes.count} canonical deals; expected {canonical_deals}"
        )

    elapsed = time.monotonic() - started
    raw_bits = bytes(outcomes.data)
    compressed_bits = gzip.compress(raw_bits, compresslevel=9, mtime=0)
    stem = f"k{k}_n{n}_t{config.resolved_tableau_columns()}"
    bitset_path = results_dir / f"{stem}.solvable.bits.gz"
    metadata_path = results_dir / f"{stem}.json"
    write_bytes(bitset_path, compressed_bits)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "config": {
            "n": n,
            "k": k,
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
            "enumeration": (
                "lexicographic permutations filtered to the lexicographic minimum "
                "under suit/color symmetry"
            ),
        },
        "solvable_canonical_deals": solvable,
        "unsolvable_canonical_deals": canonical_deals - solvable,
        "solvable_raw_deals": solvable * orbit_size,
        "unsolvable_raw_deals": (canonical_deals - solvable) * orbit_size,
        "solvability_rate": solvable / canonical_deals,
        "elapsed_seconds": elapsed,
        "solver": {
            "expanded_positions": solver.expanded_positions,
            "solvable_positions_cached": len(solver.solvable_positions),
            "unsolvable_positions_cached": len(solver.unsolvable_positions),
            "solvable_cache_hits": solver.solvable_cache_hits,
            "unsolvable_cache_hits": solver.unsolvable_cache_hits,
        },
        "bitset": {
            "file": bitset_path.name,
            "bits": outcomes.count,
            "bit_order": "deal i is bit (i % 8), least-significant bit first",
            "uncompressed_bytes": len(raw_bits),
            "compressed_bytes": len(compressed_bits),
            "sha256": hashlib.sha256(raw_bits).hexdigest(),
            "compressed_sha256": hashlib.sha256(compressed_bits).hexdigest(),
        },
    }
    write_json(metadata_path, metadata)
    print(
        f"n={n}: {solvable:,}/{canonical_deals:,} solvable "
        f"({metadata['solvability_rate']:.2%}) in {elapsed:.2f}s",
        flush=True,
    )
    return metadata


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.k < 1 or args.start_n < 1 or args.max_n < args.start_n:
        raise ValueError("require k >= 1 and 1 <= start-n <= max-n")
    if args.max_canonical_deals < 1:
        raise ValueError("max-canonical-deals must be >= 1")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "k": args.k,
        "max_canonical_deals": args.max_canonical_deals,
        "completed": [],
        "stopped": None,
    }
    completed = summary["completed"]
    if not isinstance(completed, list):
        raise AssertionError("summary completed field must be a list")

    for n in range(args.start_n, args.max_n + 1):
        canonical_deals = expected_canonical_deal_count(n, args.k)
        if canonical_deals > args.max_canonical_deals:
            summary["stopped"] = {
                "n": n,
                "raw_deals": math.factorial(2 * n * args.k),
                "canonical_deals": canonical_deals,
                "reason": (
                    f"canonical deal count exceeds ceiling "
                    f"{args.max_canonical_deals:,}"
                ),
            }
            print(
                f"stopping before n={n}: {canonical_deals:,} canonical deals "
                f"exceeds ceiling {args.max_canonical_deals:,}",
                flush=True,
            )
            break
        completed.append(
            classify_size(n, args.k, args.results_dir, args.progress_every)
        )

    write_json(args.results_dir / f"k{args.k}_summary.json", summary)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
