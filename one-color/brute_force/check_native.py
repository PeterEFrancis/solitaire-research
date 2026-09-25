from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

from .solvability import (
    ExactReserveSolver,
    build_reserve_state,
    iter_canonical_deals,
    iter_stock_collapsed_deals,
    ordered_is_solvable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the native exact search with the independent Python "
            "reference on reduced one-color deals"
        )
    )
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--stock-mode",
        choices=("reserve", "ordered"),
        default="reserve",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("brute_force/native_solver"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n < 1 or args.k != 2 or args.limit < 0:
        raise ValueError("require n >= 1, k=2, and limit >= 0")

    solver = ExactReserveSolver(args.n)
    expected = bytearray()
    deals: list[tuple[int, ...]] = []
    with tempfile.TemporaryDirectory(prefix="one-color-parity-") as directory:
        deal_path = Path(directory) / "deals.txt"
        native_path = Path(directory) / "native.outcomes"
        with deal_path.open("w", encoding="ascii") as deal_file:
            deal_iterator = (
                iter_canonical_deals(args.n, args.k)
                if args.stock_mode == "ordered"
                else iter_stock_collapsed_deals(args.n, args.k)
            )
            for index, deal in enumerate(deal_iterator):
                if args.limit and index >= args.limit:
                    break
                deals.append(deal)
                deal_file.write(",".join(map(str, deal)) + "\n")
                if args.stock_mode == "ordered":
                    expected.append(
                        ordered_is_solvable(deal, args.n, args.k)
                    )
                else:
                    expected.append(
                        solver.is_solvable(
                            build_reserve_state(deal, args.n, args.k)
                        )
                    )

        completed = subprocess.run(
            [
                str(args.binary),
                "--n",
                str(args.n),
                "--deals-file",
                str(deal_path),
                "--outcomes-file",
                str(native_path),
                "--stock-mode",
                args.stock_mode,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = native_path.read_bytes()

    if actual != expected:
        mismatch = next(
            index
            for index, (left, right) in enumerate(zip(expected, actual))
            if left != right
        )
        raise AssertionError(
            f"native mismatch at deal {mismatch}: {deals[mismatch]}; "
            f"python={expected[mismatch]} native={actual[mismatch]}"
        )

    print(completed.stdout, end="")
    print(f"python_expanded_positions {solver.expanded_positions}")
    print(f"matching_outcomes {len(expected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
