from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

from solitaire.game import DeckConfig

from .solvability import TABLEAU_PATTERN_RANK_ENUMERATION


CHUNK_SIZE = 4 * 1024 * 1024
POPCOUNTS = tuple(bin(value).count("1") for value in range(256))


def parse_args() -> argparse.Namespace:
    results = Path(__file__).with_name("results")
    parser = argparse.ArgumentParser(description="Finalize stock-collapsed n=4 data")
    parser.add_argument("--results-dir", type=Path, default=results)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--completion", type=Path)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = DeckConfig(n=4, k=2)
    stem = "k2_n4_t4.stock_collapsed"
    raw_path = args.raw or args.results_dir / f"{stem}.bits.partial"
    completion_path = (
        args.completion or args.results_dir / f"{stem}.patterns.complete"
    )
    compressed_path = args.results_dir / f"{stem}.solvable.bits.gz"
    metadata_path = args.results_dir / f"{stem}.json"

    completion = completion_path.read_bytes()
    if len(completion) != 90_300 or any(value != 1 for value in completion):
        raise ValueError(
            f"incomplete stock-collapsed run: {sum(value == 1 for value in completion)}"
            f"/{len(completion)} patterns"
        )

    canonical_deals = 3_632_428_800
    expected_bytes = canonical_deals // 8
    if raw_path.stat().st_size != expected_bytes:
        raise ValueError("stock-collapsed bitset has the wrong size")

    raw_digest = hashlib.sha256()
    solvable = 0
    temporary = compressed_path.with_name(compressed_path.name + ".tmp")
    with raw_path.open("rb") as source, temporary.open("wb") as compressed_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=compressed_file,
            compresslevel=9,
            mtime=0,
        ) as target:
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    break
                raw_digest.update(chunk)
                solvable += sum(POPCOUNTS[byte] for byte in chunk)
                target.write(chunk)
    temporary.replace(compressed_path)

    if solvable != 3_579_359_602:
        raise ValueError(f"unexpected solvable count: {solvable}")
    orbit_size = 8 * math.factorial(6)
    raw_deals = math.factorial(16)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "status": "exhaustive",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n": 4,
            "k": 2,
            "t": config.resolved_tableau_columns(),
            "draw_count": config.draw_count,
            "max_recycles": config.max_recycles,
        },
        "raw_deals": raw_deals,
        "canonical_deals": canonical_deals,
        "canonical_tableau_patterns": 90_300,
        "equivalence": {
            "suit_color_orbit_size": 8,
            "stock_order_orbit_size": math.factorial(6),
            "combined_orbit_size": orbit_size,
            "stock_cards": 6,
            "stock_order_basis": (
                "Every reserve win prescribes an order for removing the six stock "
                "cards. Every relative draw order can emit that prescribed order "
                "within four passes using the waste stack."
            ),
            "stock_order_proof": {
                "file": "brute_force/prove_stock_order.py",
                "permutations_checked": math.factorial(6),
                "pass_histogram": {"1": 132, "2": 424, "3": 160, "4": 4},
                "max_passes_required": 4,
                "passes_available": config.max_recycles + 1,
            },
        },
        "deal_order": {
            "card_id": "suit * n + (rank - 1)",
            "sequence": "tableau deal order, then remaining card ids sorted as stock",
            "enumeration_id": TABLEAU_PATTERN_RANK_ENUMERATION,
            "enumeration": (
                "canonical tableau suit patterns in lexicographic order, followed "
                "by partial rank-permutation Cartesian products by suit"
            ),
        },
        "solvable_canonical_deals": solvable,
        "unsolvable_canonical_deals": canonical_deals - solvable,
        "solvable_raw_deals": solvable * orbit_size,
        "unsolvable_raw_deals": (canonical_deals - solvable) * orbit_size,
        "solvability_rate": solvable / canonical_deals,
        "solver": {
            "engine": "brute_force/native_solver.cpp",
            "classification": "exact state graph with unordered-reserve stock",
            "safe_foundation_moves_forced": True,
            "tableau_pile_symmetry_in_state_cache": True,
            "expanded_positions": 72_264_410_042,
        },
        "validation": {
            "n3_full_deals_checked": 59_875_200,
            "n3_reference_sha256": (
                "d9a420124a5387b9f23304bea527b161bd54cabb717f2646f02087479f94fa9e"
            ),
            "n4_tableaus_all_stock_orders_checked": 100_000,
            "n4_stock_order_comparisons": 72_000_000,
            "n4_ordinary_vs_reserve_block_deals": 4_649_472,
        },
        "bitset": {
            "file": compressed_path.name,
            "bits": canonical_deals,
            "bit_order": "deal i is bit (i % 8), least-significant bit first",
            "uncompressed_bytes": expected_bytes,
            "compressed_bytes": compressed_path.stat().st_size,
            "sha256": raw_digest.hexdigest(),
            "compressed_sha256": file_sha256(compressed_path),
        },
    }
    write_json(metadata_path, metadata)
    print(
        f"n=4 stock-collapsed: {solvable:,}/{canonical_deals:,} solvable "
        f"({solvable / canonical_deals:.6%}); compressed bytes "
        f"{compressed_path.stat().st_size:,}"
    )


if __name__ == "__main__":
    main()
