#!/usr/bin/env python3
"""Plot the saved Vegas payout PMF; requires matplotlib (only for this figure).

First run `cd original && python3 -m brute_force.vegas_earnings`.
Then run this script from the repository root. No simulation or fitting occurs.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "original/brute_force/results/vegas-earnings.json"


def dollars(value, decimals=0):
    return ("−" if value < 0 else "+" if value > 0 else "") + f"${abs(value):.{decimals}f}"


def plot(report_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    raw = report_path.read_bytes()
    report = json.loads(raw)
    row = report["policies"]["profit"]
    pmf = row["pmf"]
    assert len(pmf) == 53 and sum(p["count"] for p in pmf) == row["sample_size"]
    assert all(p["net_dollars"] == 5 * p["foundation_cards"] - 52 for p in pmf)
    x = [p["net_dollars"] for p in pmf]
    y = [p["probability"] for p in pmf]
    paper, ink, muted, green, red = "#faf9f5", "#1f3028", "#637067", "#244f3e", "#a84d39"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.parse_math": False,
        "text.color": ink, "axes.labelcolor": ink, "xtick.color": muted,
        "ytick.color": muted, "svg.hashsalt": "solitaire-vegas-earnings",
    })
    fig = plt.figure(figsize=(11.5, 7.0), facecolor=paper)
    fig.text(.08, .94, "Vegas: the distribution of net earnings", fontsize=21, weight="bold")
    fig.text(.08, .895, "Payout-focused computer policy · one attempt per deal · full information", color=muted, fontsize=11)
    cards = [
        ("MEAN NET RETURN", dollars(row["mean_net_dollars"], 2)),
        ("MEDIAN NET RETURN", dollars(row["median_net_dollars"])),
        ("CHANCE OF A PROFIT", f"{row['events']['profit']['probability']:.2%}"),
    ]
    for xpos, (label, value) in zip((.08, .39, .70), cards):
        fig.text(xpos, .822, label, color=muted, fontsize=9, weight="bold")
        fig.text(xpos, .768, value, fontsize=24, color=ink)

    ax = fig.add_axes((.08, .22, .87, .47), facecolor=paper)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d8ddd4", linewidth=.65)
    ax.bar(x, y, width=3.7, color=[red if v < 0 else green for v in x], zorder=3)
    ax.axvline(0, color=muted, linewidth=.9, linestyle=(0, (3, 4)))
    ax.set_xlim(-58, 216)
    ax.set_ylim(0, max(y) * 1.20)
    ax.set_ylabel("Share of deals", labelpad=11)
    ax.set_xlabel("Net earnings after the $52 entry cost", labelpad=13)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ticks = [-52, -27, -2, 23, 48, 73, 98, 123, 148, 173, 208]
    ax.set_xticks(ticks, [dollars(v) for v in ticks], fontsize=10)
    ax.tick_params(axis="both", length=0, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d8ddd4")
    ax.text(-28, max(y) * 1.115, f"Lose money: {row['events']['loss']['probability']:.2%}", ha="center", color=red, fontsize=10)
    ax.annotate(
        f"All 52 cards home\n+$208 · {row['events']['full_win']['probability']:.2%}",
        xy=(208, y[-1]), xytext=(137, max(y) * .66), fontsize=11, color=green,
        arrowprops={"arrowstyle": "-", "color": green, "connectionstyle": "angle,angleA=0,angleB=90,rad=7"},
        linespacing=1.6,
    )
    fig.text(.08, .105, "Each bar is one possible payout: $5 × foundation cards − $52. Zero is not a possible result.", fontsize=10, color=muted)
    fig.text(.08, .071, f"{row['sample_size']:,} held-out deals · draw one · one stock pass · whole-run tableau transfers only", fontsize=10, color=muted)
    fig.text(.08, .037, "Measured algorithm performance; hidden cards are available to this model. Human play was not tested.", fontsize=9, color=muted)
    output.parent.mkdir(parents=True, exist_ok=True)
    description = "Full 53-outcome PMF for frozen Vegas profit policy. Source JSON SHA-256: " + hashlib.sha256(raw).hexdigest()
    fig.savefig(output.with_suffix(".png"), dpi=190, metadata={"Description": description})
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None, "Description": description})
    plt.close(fig)
    print(f"Wrote {output.with_suffix('.png')} and .svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT.with_suffix(""), help="Output stem (PNG and SVG)")
    args = parser.parse_args()
    plot(args.report, args.output)


if __name__ == "__main__":
    main()
