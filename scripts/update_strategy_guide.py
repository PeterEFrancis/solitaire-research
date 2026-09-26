#!/usr/bin/env python3
"""Generate the rule-variant report from completed, frozen experiment records.

The original report remains below a marked historical heading. Generation is
idempotent and refuses incomplete or inconsistent inputs. To inspect the small
smoke study without changing the guide:

    python3 scripts/update_strategy_guide.py --prefix variant-smoke --output /tmp/guide-smoke.md
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "original/brute_force/results"
GUIDE = ROOT / "original/strategy-guide.md"
sys.path.insert(0, str(ROOT / "original"))
from solitaire.player import PARAMETER_NAMES  # noqa: E402

START = "<!-- BEGIN GENERATED VARIANT STUDY -->"
END = "<!-- END GENERATED VARIANT STUDY -->"
TITLE = "# Solitaire Research: how rules change strategy"
PROFILES = {
    "draw1_limited": ("Draw one / four passes", 1, 3, True),
    "draw1_unlimited": ("Draw one / unlimited", 1, None, True),
    "draw3_limited": ("Draw three / four passes", 3, 3, True),
    "draw3_unlimited": ("Draw three / unlimited", 3, None, True),
    "vegas": ("Vegas / one pass", 1, 0, False),
}
SIMPLE = {
    "S": "safe_foundation", "R": "reveal_hidden", "B": "tableau_build",
    "D": "stock_action", "W": "waste_to_tableau", "H": "reveal_depth",
    "C": "recycle", "P": "productive_stack_length",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid number: {label}")
    return value


def count(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid count: {label}")
    return value


def close(actual, expected, label):
    number(actual, label)
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), f"Inconsistent {label}")


def weights(value, label):
    require(isinstance(value, dict) and set(value) == set(PARAMETER_NAMES), f"Invalid parameter map: {label}")
    for key, item in value.items():
        number(item, f"{label}.{key}")
    return value


def wilson(wins, deals):
    z = 1.959963984540054
    p = wins / deals
    denominator = 1 + z * z / deals
    center = (p + z * z / (2 * deals)) / denominator
    half = z * math.sqrt(p * (1 - p) / deals + z * z / (4 * deals * deals)) / denominator
    return center - half, center + half


def check_pair(pair, row, baseline):
    require(isinstance(pair, dict), "Missing paired comparison")
    require(pair["deals"] == row["deals"], "Paired sample size differs")
    b = count(pair["b_candidate_wins_baseline_loses"], "paired b")
    c = count(pair["c_baseline_wins_candidate_loses"], "paired c")
    require(b + c <= row["deals"], "Too many discordant deals")
    require(b - c == pair["net_wins"] == row["wins"] - baseline["wins"], "Paired gain disagrees with wins")
    require(pair["discordant_deals"] == b + c, "Discordant count differs")
    delta = (b - c) / row["deals"]
    se = math.sqrt(max(0.0, (b + c) / row["deals"] - delta * delta) / row["deals"])
    close(pair["delta_percentage_points"], 100 * delta, "paired gain")
    close(pair["standard_error_percentage_points"], 100 * se, "paired standard error")
    interval = pair["paired_normal_95_ci_percentage_points"]
    require(isinstance(interval, list) and len(interval) == 2, "Invalid paired interval")
    z = 1.959963984540054
    for actual, expected in zip(interval, (100 * (delta - z * se), 100 * (delta + z * se))):
        close(actual, expected, "paired interval")


def check_evaluation(row, deals, results_dir):
    require(row["deals"] == deals, "Confirmation sample size differs from protocol")
    wins = count(row["wins"], "wins")
    require(wins <= deals, "Wins exceed deals")
    total = count(row["foundation_cards"], "foundation total")
    squared = count(row["foundation_cards_squared"], "foundation squared total")
    require(52 * wins <= total <= 52 * deals and squared <= 52 * total and total * total <= squared * deals,
            "Invalid foundation totals")
    close(row["win_rate"], wins / deals, "win rate")
    require(isinstance(row["win_rate_wilson_95_ci"], list) and len(row["win_rate_wilson_95_ci"]) == 2, "Invalid Wilson interval")
    for actual, expected in zip(row["win_rate_wilson_95_ci"], wilson(wins, deals)):
        close(actual, expected, "Wilson interval")
    mean = total / deals
    variance = max(0, squared / deals - mean * mean)
    margin = 1.96 * math.sqrt(variance / max(1, deals - 1))
    close(row["mean_foundation_cards"], mean, "mean foundation cards")
    close(row["mean_net_dollars"], 5 * mean - 52, "mean net dollars")
    for key, expected in (
        ("mean_foundation_cards_95_ci", (mean - margin, mean + margin)),
        ("mean_net_dollars_95_ci", (5 * (mean - margin) - 52, 5 * (mean + margin) - 52)),
    ):
        require(len(row[key]) == 2, f"Invalid {key}")
        for actual, target in zip(row[key], expected):
            close(actual, target, key)
    if "cutoffs" in row:
        require(count(row["cutoffs"], "cutoffs") <= deals - wins, "Invalid cutoff count")
        close(row["mean_steps"], count(row["steps"], "steps") / deals, "mean steps")
    for kind in ("wins", "foundations"):
        require((results_dir / row[kind + "_file"]).is_file(), f"Missing retained {kind} outcomes")
        require(re.fullmatch(r"[0-9a-f]{64}", row[kind + "_sha256"]), f"Invalid {kind} digest")


def load_study(prefix):
    records, paths = {}, {}
    for kind in ("protocol", "training", "confirmation", "selected-policies"):
        path = RESULTS / f"{prefix}.{kind}.json"
        paths[kind] = path
        records[kind] = json.loads(path.read_text(encoding="utf-8"))
        require(records[kind].get("schema_version") == 1, f"Unsupported {kind} schema")
        if kind != "protocol":
            require(records[kind].get("complete") is True, f"{kind} is incomplete")
    protocol, training, confirmation, frozen = (records[k] for k in records)
    options = protocol["options"]
    require(set(options["profiles"]) == set(PROFILES), "Expected all five rule variants")
    for key in ("train_deals", "validation_deals", "test_deals", "max_steps", "finalists"):
        count(options[key], key, 1)
    count(options["mutations"], "mutations")
    require(len(set(protocol["seeds"].values())) == 3, "Experiment splits must use distinct seeds")
    require(confirmation["protocol"] == paths["protocol"].name, "Wrong confirmation protocol")
    require(confirmation["frozen_policies_sha256"] == hashlib.sha256(paths["selected-policies"].read_bytes()).hexdigest(),
            "Frozen policies changed after confirmation")
    if prefix == "variant-study":
        expected = {"train_deals": 5000, "validation_deals": 50000, "test_deals": 250000,
                    "max_steps": 1000, "mutations": 24, "finalists": 4}
        require(all(options[key] == value for key, value in expected.items()), "Final study does not match its planned settings")
        require(protocol["seeds"]["test"] == 2026092699, "Wrong final confirmation seed")
    visible = set(json.loads((RESULTS / "human-strategy-candidates.json").read_text())["audit"]["current_visible_features"])
    for record in (training, confirmation, frozen):
        require(set(record["variants"]) == set(PROFILES), "Missing variant records")
    for profile, (_, draw, recycles, split) in PROFILES.items():
        expected = dict(n=13, k=2, t=7, draw_count=draw, max_recycles=recycles,
                        allow_tableau_stack_splitting=split)
        train, result, selected = (record["variants"][profile] for record in (training, confirmation, frozen))
        for record in (train, result, selected):
            require(record["config"] == expected, f"Wrong rules for {profile}")
        require(train["training_evaluations"] == len(train["history"]) > 0, "Invalid training history")
        require(all(row["deals"] == options["train_deals"] for row in train["history"]), "Wrong training sample size")
        families = {"full", "visible", "simple_eight"} | ({"profit"} if profile == "vegas" else set())
        require(set(selected["policies"]) == families == set(train["validation"]), "Missing policy families")
        for family in families:
            policy = weights(selected["policies"][family], f"{profile}.{family}")
            validation = train["validation"][family]
            require(validation and all(row["deals"] == options["validation_deals"] for row in validation), "Wrong validation sample size")
            require(policy == validation[0]["parameters"], "Selected policy differs from validation winner")
            if family in ("visible", "simple_eight"):
                allowed = visible if family == "visible" else set(SIMPLE.values())
                require(all(not value or name in allowed for name, value in policy.items()), f"Information restriction violated: {family}")
        require(selected["parameters"] == selected["policies"]["full"], "Full policy copies disagree")
        require(1 <= len(selected["portfolio"]) <= 6, "Invalid portfolio size")
        for policy in selected["portfolio"]:
            weights(policy, f"{profile}.portfolio")
        require(selected["policies"]["full"] in selected["portfolio"], "Portfolio omits selected full policy")
        evaluations = result["evaluations"]
        members = [f"portfolio_{index}" for index in range(len(selected["portfolio"]))]
        require(set(evaluations) == families | {"stage8_baseline", "simple_eight_baseline", "portfolio"} | set(members), "Unexpected confirmation policies")
        for row in evaluations.values():
            check_evaluation(row, options["test_deals"], RESULTS)
            check_pair(row["paired_vs_stage8"], row, evaluations["stage8_baseline"])
        bank = evaluations["portfolio"]
        require(bank["members"] == members and bank["attempts_per_deal"] == len(members), "Portfolio membership differs")
        require(bank["wins"] >= evaluations["full"]["wins"], "Portfolio lost a selected-full win")
        check_pair(bank["paired_vs_selected_full"], bank, evaluations["full"])
        check_pair(evaluations["simple_eight"]["paired_vs_simple_baseline"], evaluations["simple_eight"], evaluations["simple_eight_baseline"])
    return records


def pct(value, digits=4):
    return f"{100 * value:.{digits}f}%"


def ci(values, percent=False, digits=3):
    values = [value * (100 if percent else 1) for value in values]
    return f"{values[0]:.{digits}f} to {values[1]:.{digits}f}" + ("%" if percent else "")


def signed(value, digits=4):
    return f"{value:+.{digits}f}".replace("-", "−")


def money(value):
    return ("−" if value < 0 else "+") + f"${abs(value):.2f}"


def active(policy):
    return sum(value != 0 for value in policy.values())


def coefficient(value):
    return f"{value:.17g}".replace("-", "−")


def move_expression(constant=0, length=0, depth=0):
    """Render a numeric affine score, retaining future nonzero L/h terms."""
    parts = []
    for value, variable in ((constant, ""), (length, "L"), (depth, "h")):
        if value == 0:
            continue
        magnitude = "" if variable and abs(value) == 1 else coefficient(abs(value))
        term = magnitude + variable
        parts.append(("−" if value < 0 else "") + term if not parts
                     else (" − " if value < 0 else " + ") + term)
    return "".join(parts) or "0"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(":--" for _ in headers) + "|"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def vegas_earnings_section(earnings, frozen):
    policies = earnings["policies"]
    main = policies["profit"]
    events = main["events"]
    quantiles = main["quantiles_net_dollars"]
    attempts = len(frozen["vegas"]["portfolio"])
    labels = (
        ("profit", "Payout-focused", "Full deal"),
        ("full", "Win-focused full", "Full deal"),
        ("visible", "Visible", "Current visible position"),
        ("simple_eight", "Eight-feature", "Current visible position"),
        ("stage8_baseline", "Stage 8", "Full deal"),
        ("portfolio", "Restart portfolio", f"Full deal; {attempts} attempts"),
    )
    comparisons = []
    for name, label, information in labels:
        policy = policies[name]
        comparisons.append((label, information, money(policy["mean_net_dollars"]),
                            money(policy["median_net_dollars"]), pct(policy["events"]["profit"]["probability"]),
                            pct(policy["events"]["full_win"]["probability"])))
    groups = []
    for group in main["bins"]:
        low, high = group["foundation_cards_min"], group["foundation_cards_max"]
        cards = str(low) if low == high else f"{low}–{high}"
        net = money(5 * low - 52)
        if low != high:
            net += " to " + money(5 * high - 52)
        groups.append((cards, net, f"{group['count']:,}", pct(group["probability"])))
    objective_comparison = ""
    if (main["mean_net_dollars"] > policies["full"]["mean_net_dollars"]
            and events["full_win"]["probability"] < policies["full"]["events"]["full_win"]["probability"]):
        objective_comparison = (
            "The payout-focused policy earned " + money(main["mean_net_dollars"] - policies["full"]["mean_net_dollars"])
            + " more per deal on average than the win-focused full policy, despite completing fewer games. "
        )
    return "## Vegas: what do you earn per $52 deal? {#vegas-earnings}\n\n" + (
        "The advisor-inspired Vegas game uses draw one, one pass, and whole visible-run transfers. Pay **$52 for the deck and receive "
        "$5 per foundation card**: net earnings are **5 × foundation cards − 52**. Turning a card face up earns nothing. These are the "
        "rules of this experiment, not a claim about every casino.\n\n"
        f"The main earnings policy is the **payout-focused policy**, selected before confirmation to maximize foundation cards, with "
        f"{active(frozen['vegas']['policies']['profit'])} nonzero weights. On {main['sample_size']:,} fresh confirmation deals, its "
        f"**mean net return was {money(main['mean_net_dollars'])} per deal; the median was {money(main['median_net_dollars'])}.** "
        f"It made a profit on {pct(events['profit']['probability'])} of deals and a loss on {pct(events['loss']['probability'])}. "
        f"It lost the entire $52 stake on {pct(events['full_loss']['probability'])} and completed all 52 foundation cards on "
        f"{pct(events['full_win']['probability'])}. This fixed policy can inspect the full deal; these are computer results, not measured "
        "human earnings.\n\n"
        "**Exactly breaking even is impossible:** ten foundation cards return $50, a $2 loss; eleven return $55, a $3 profit. "
        "Possible net returns run from −$52 to +$208 in $5 steps.\n\n"
        "The 5th, 25th, 50th, 75th, and 95th percentiles of individual net returns were "
        + ", ".join(money(quantiles[key]) for key in ("q05", "q25", "q50", "q75", "q95")) + ", respectively. "
        "The mean's 95% confidence interval is "
        + " to ".join(money(value) for value in main["mean_net_dollars_95_ci"])
        + ". This normal-approximation interval describes uncertainty in the **average**, not a range containing 95% of individual "
        "deal outcomes. The return distribution is uneven: frequent losses coexist with a smaller chance of much larger positive payouts."
    ) + "\n\n" + (
        "![Distribution of net earnings from the payout-focused Vegas policy: losses are common, with a smaller positive-return tail "
        "extending to a $208 net win.](brute_force/results/vegas-earnings.png)\n\n"
        f"*Net earnings for the frozen payout-focused policy on {main['sample_size']:,} confirmation deals. The chart and table summarize "
        "the saved outcomes; no new deals or policy fitting were used.*"
    ) + "\n\n" + table(("Foundation cards", "Net earnings", "Deals", "Probability"), groups) + "\n\n" + (
        "The six existing policies show why complete-win rate and earnings are different objectives. “Visible” and “Eight-feature” use "
        "audited current-position information; the others can use the full deal. All single-policy rows are one fixed trajectory per deal."
    ) + "\n\n" + table(("Policy", "Information", "Mean net", "Median net", "Profit probability", "Full-win rate"), comparisons) + "\n\n" + (
        objective_comparison +
        f"The portfolio is hypothetical planning with {attempts} attempts on the **same known deal**, reporting the best foundation outcome. "
        f"Its displayed return charges one $52 stake for that selected trajectory; it is **not {attempts} independently paid plays**, nor an "
        "ordinary single-pass strategy.\n\n"
        "The [complete 53-point earnings distributions](brute_force/results/vegas-earnings.json) contain every foundation count, "
        "net return, probability, quantile, and source hash for all six policies. The "
        "[earnings analyzer](brute_force/vegas_earnings.py) reconstructs them from the frozen confirmation outcomes. This is a new "
        "summary of existing test data; it changes no policy and makes no new selection. From the repository root, verify it with:\n\n"
        "```bash\ncd original\npython3 -m brute_force.vegas_earnings --check\n```\n\n"
        "To regenerate the chart with the [plotting script](../scripts/plot_vegas_earnings.py), "
        "run `python3 scripts/plot_vegas_earnings.py` from the repository root "
        "with the optional Matplotlib dependency installed."
    )


def expanded_report(records, prefix, earnings=None):
    protocol = records["protocol"]
    options, seeds = protocol["options"], protocol["seeds"]
    frozen = records["selected-policies"]["variants"]
    variants = records["confirmation"]["variants"]
    deals = options["test_deals"]
    rule_rows, result_rows, gain_rows, cutoff_rows, coefficient_rows = [], [], [], [], []
    for profile, (label, draw, recycles, split) in PROFILES.items():
        selected, evaluations = frozen[profile], variants[profile]["evaluations"]
        rule_rows.append((label, draw, "Unlimited" if recycles is None else recycles + 1,
                          "Allowed" if split else "Whole visible run only"))
        def interval_rate(name):
            row = evaluations[name]
            return pct(row["win_rate"]) + " (" + ci(row["win_rate_wilson_95_ci"], percent=True) + ")"
        result_rows.append((label, pct(evaluations["stage8_baseline"]["win_rate"]), interval_rate("full"),
                            pct(evaluations["visible"]["win_rate"]), pct(evaluations["simple_eight"]["win_rate"]), interval_rate("portfolio")))
        def gain(pair):
            return signed(pair["delta_percentage_points"]) + " (" + ci(pair["paired_normal_95_ci_percentage_points"]) + ")"
        gain_rows.append((label, gain(evaluations["full"]["paired_vs_stage8"]),
                          gain(evaluations["portfolio"]["paired_vs_selected_full"]),
                          "/".join(str(active(selected["policies"][family])) for family in ("full", "visible", "simple_eight")),
                          len(selected["portfolio"])))
        attempts = sum(evaluations[name]["cutoffs"] for name in evaluations["portfolio"]["members"])
        cutoff_rows.append((label, *(f"{evaluations[family]['cutoffs']:,}" for family in ("stage8_baseline", "full", "visible", "simple_eight")),
                            f"{attempts:,} / {deals * len(selected['portfolio']):,}"))
        simple = selected["policies"]["simple_eight"]
        coefficient_rows.append((label, *(coefficient(simple[name]) for name in SIMPLE.values())))
    vegas = variants["vegas"]["evaluations"]
    vegas_rows = []
    for family, label in (("stage8_baseline", "Stage 8"), ("full", "Win-focused full"), ("visible", "Visible"),
                          ("simple_eight", "Eight-feature"), ("profit", "Foundation-focused"), ("portfolio", "Restart portfolio")):
        row = vegas[family]
        vegas_rows.append((label, pct(row["win_rate"]), f"{row['mean_foundation_cards']:.4f}", money(row["mean_net_dollars"]),
                           " to ".join(money(value) for value in row["mean_net_dollars_95_ci"])))
    training_counts = [records["training"]["variants"][profile]["training_evaluations"] for profile in PROFILES]
    simple_policies = [frozen[profile]["policies"]["simple_eight"] for profile in PROFILES]
    simplify = all(policy["tableau_build"] == -1 and policy["productive_stack_length"] == 1
                   and policy["reveal_depth"] == 1 for policy in simple_policies)
    score_note = " Their common B=−1, P=1, H=1 simplify the scores below." if simplify else ""
    score_columns = []
    for profile, policy in zip(PROFILES, simple_policies):
        s, r, b, d, w, h, c, p = (policy[name] for name in SIMPLE.values())
        score_columns.append((
            coefficient(d), "Unavailable" if PROFILES[profile][2] == 0 else coefficient(d + c),
            coefficient(b + w), coefficient(s), move_expression(r, b + p, h),
            move_expression(length=b + p), move_expression(length=b), move_expression(r, depth=h),
        ))
    score_rows = [(label, *(column[index] for column in score_columns)) for index, label in enumerate((
        "Draw", "Recycle", "Waste to tableau", "Foundation passing support test, no reveal",
        "Tableau transfer revealing a card", "Transfer emptying source, no reveal", "Other tableau transfer",
        "Bonus when a foundation move reveals a card",
    ))]
    vegas_simple = frozen["vegas"]["policies"]["simple_eight"]
    vegas_reveal = vegas_simple["reveal_hidden"] + 4 * vegas_simple["reveal_depth"]
    vegas_foundation = vegas_simple["safe_foundation"]
    examples = (
        "For example, in Vegas a revealing transfer with four face-down cards in its source scores "
        f"{coefficient(vegas_reveal)}" + (" (independent of run length)" if simplify else " before any length term") + ", "
        + ("beating" if vegas_reveal > vegas_foundation and simplify else "compared with")
        + f" a nonrevealing foundation move passing the support test at {coefficient(vegas_foundation)}; "
        f"drawing scores {coefficient(vegas_simple['stock_action'])}. "
        "In unlimited draw-one, recycling scores "
        f"{coefficient(simple_policies[1]['stock_action'] + simple_policies[1]['recycle'])}. "
        "A foundation move failing the support test starts at 0 and still receives any reveal bonus."
    )
    warning = ("> **Smoke-test preview.** These tiny samples test the reporting pipeline and are not research findings.\n\n"
               if deals < 1000 else "")
    sections = [
        warning + (
            f"This study compares five Klondike versions on **{deals:,} fresh deals per version**, using the same test seed and frozen policies. "
            f"It separates compact, observable strategies from full-information computer searches. The results measure these algorithms; "
            f"neither optimal play nor human performance has been established."
        ),
        "## The five games\n\n" + table(("Variant", "Cards per draw", "Maximum stock passes", "Splitting face-up stacks"), rule_rows) + "\n\n" + (
            "All versions use 52 cards, seven tableau columns, alternating descending building, kings in empty columns, and ascending suit "
            "foundations with no return moves. Four passes includes three recycles. Draw three turns up at most three remaining cards; only "
            "the waste's top card is playable. Vegas transfers move the entire visible run; individual top cards may still go to foundations.\n\n"
            f"**Unlimited is a rule, not an unlimited computation.** Every trajectory in this study has a {options['max_steps']:,}-move cap "
            "and repeated-position avoidance. A cutoff is a policy failure within that budget, not a proof that the deal is unsolvable."
        ),
        "## What the independent test found\n\n" + table(("Variant", "Stage 8 baseline", "Tuned full (95% interval)", "Tuned visible", "Tuned eight-feature", "Restart portfolio (95% interval)"), result_rows) + "\n\n" + (
            f"Stage 8 is freshly evaluated under each rule. Full policies can use {len(PARAMETER_NAMES)} features and inspect unknown cards. Visible policies use audited current-position "
            "features; compact policies use at most eight. Both exclude buried-waste memory and hidden identities. These computer rates include "
            "exact bookkeeping and cycle avoidance; human learnability was not tested.\n\n"
            "The portfolio restarts up to six frozen policies on the **same completely known deal**, accepting any winning trajectory or the "
            "highest foundation count. These separate attempts cost extra computation. The bank includes the selected full policy, preserving "
            "its wins. Success supplies legal winning paths, not an exact solvability probability.\n\n"
            "Win-rate intervals are 95% Wilson intervals. Gains below use approximate paired normal 95% intervals in percentage points, "
            "based on common-deal outcomes. They assume IID-uniform sampling as an interpretation of seeded shuffles. Intervals are marginal, "
            "not simultaneous, and describe frozen policies rather than global optima."
        ) + "\n\n" + table(("Variant", "Full minus stage 8 (points)", "Portfolio minus full (points)", "Nonzero weights: full / visible / eight", "Restart attempts"), gain_rows) + "\n\n" + (
            "Negative gains do not trigger reselection. Cross-rule comparisons also change selected weights, so they do not isolate a rules-only "
            "effect. Nonzero coefficient counts can include terms that never fire under a rule, such as Vegas recycling."
        ),
        "## A scorecard for each rule\n\n" + (
            "Score each eligible legal move and take the highest score. These are the frozen eight-feature weights. **L** is a transferred "
            "stack's length and **h** is the source column's face-down-card count before a revealing move. The feature called **safe foundation** "
            "is an ace, or a legal foundation move whose opposite-color foundations have both reached at least one rank below the moving card. "
            "This tests tableau support; it does not prove that immediately removing a waste card is best under draw-three packet timing. "
            "The benchmark does not force these moves, and failing the test does not make a foundation move illegal."
        ) + "\n\n" + table(("Variant", "S", "R", "B", "D", "W", "H", "C", "P"), coefficient_rows) + "\n\n" + (
            "Coefficients: foundation-support test **S**, reveal **R**, tableau cards **B**, stock action **D**, waste-to-tableau **W**, "
            "reveal depth **H**, recycle **C**, productive transfer **P**."
        ) + score_note + "\n\n" + table(("Move", "Draw 1 / four", "Draw 1 / unlimited", "Draw 3 / four", "Draw 3 / unlimited", "Vegas"), score_rows)
        + "\n\n" + examples + "\n\n" + (
            "Vegas has no recycle or partial-run transfer. Revealing and emptying are mutually exclusive. The program excludes repeated "
            "positions and redundant whole-column relocations to empty columns. Ties favor stock actions, then waste moves, then tableau "
            "sources left to right; foundations precede transfers, generated longest first. Equivalent empty destinations use the first column."
        ),
        "## Turning the scores into play\n\n" + (
            "Uncover blocked columns, use the foundation-support test, and distinguish productive transfers from rearrangement. The coefficients "
            "express competing priorities, not unconditional instructions. A reserve card may still be needed as an intermediate tableau support.\n\n"
            "Unlimited draw-one allows returning later, changing waste urgency compared with a last pass, without guaranteeing progress. Under "
            "draw three, removing cards changes later packet alignment. Inspect the exposed waste card; one-card stock habits need rethinking.\n\n"
            "Some buried draw-three packet cards were never exposed on top. Their identities are not automatically valid memory features. "
            "Visible and eight-feature policies therefore omit all three waste-history features in every variant. Full policies can inspect "
            "hidden identities through successor scoring; their and the portfolio's rates are not established human-performance rates."
        ),
        vegas_earnings_section(earnings, frozen) if earnings is not None else "## The Vegas objective: cards returned, not only complete wins\n\n" + (
            "This advisor-inspired experiment pays **$5 per foundation card, minus $52 for the deck**. Turning a card face up earns nothing. "
            "These explicitly defined rules are not a claim about all casinos. Optimizing complete wins can select a different policy from "
            "optimizing foundation cards."
        ) + "\n\n" + table(("Policy", "Win rate", "Mean foundation cards", "Mean net return", "95% interval for mean return"), vegas_rows) + "\n\n" + (
            f"The separately selected foundation-focused policy has {active(frozen['vegas']['policies']['profit'])} nonzero weights and was optimized "
            "for foundation count, breaking ties by wins and moves. Return intervals use a normal approximation and the recorded foundation "
            "totals and squared totals. They describe the fixed policy and shuffle model, without guaranteeing profit. Portfolio returns use "
            "full-information planning across multiple attempts."
        ),
        "## Search, confirmation, and computation limits\n\n" + (
            f"Training used {options['train_deals']:,} common deals (seed {seeds['train']}), rule-sensitive starting points, a coordinate sweep, "
            f"and {options['mutations']} sparse mutation proposals per family. The training record contains {min(training_counts):,}–{max(training_counts):,} "
            f"distinct candidate evaluations per variant across its families. Up to {options['finalists']} leading candidates plus the family's "
            f"starting policy were evaluated on {options['validation_deals']:,} different deals (seed {seeds['validation']}). Validation selected "
            "single policies and the restart bank. Complete wins were primary, except for the Vegas foundation objective.\n\n"
            f"All selections were frozen before the {deals:,}-deal confirmation (seed {seeds['test']}). Every frozen policy was reported. "
            "The test selected no weights or bank members. This search was not exhaustive. Unlimited-pass models fixed consumed-pass-pressure "
            "coefficients to zero.\n\n"
            f"The following counts reached the {options['max_steps']:,}-move cap without winning. Single-policy entries count deals out of "
            f"{deals:,}; the portfolio column counts capped attempts across its entire bank, so the same deal can appear more than once."
        ) + "\n\n" + table(("Variant", "Stage 8", "Full", "Visible", "Eight-feature", "Portfolio capped / total attempts"), cutoff_rows),
        "## Evidence and reproduction\n\n" + (
            f"The [frozen protocol](brute_force/results/{prefix}.protocol.json), "
            f"[training and validation record](brute_force/results/{prefix}.training.json), "
            f"[confirmation results](brute_force/results/{prefix}.confirmation.json), and "
            f"[selected weights](brute_force/results/{prefix}.selected-policies.json) record the rules, seeds, and outcomes. "
            "Indexed outcomes, source hashes, and executable hashes document provenance. Different C++ standard-library shuffles can produce "
            "different deals; exact replay requires matching the build.\n\n"
            "Build the native player from the repository root, then run a selected policy in Python:\n\n"
            "```bash\npython3 scripts/build_native.py\ncd original\n"
            "python3 -m solitaire.play_variant draw3_unlimited --policy simple_eight --seed 7\n```\n\n"
            "Python's seeded deal is not an indexed native replay. Use `--policy portfolio` for the restart bank; other variants are "
            "`draw1_limited`, `draw1_unlimited`, `draw3_limited`, and `vegas`. "
            "The [study runner](brute_force/variant_study.py) supplies the experimental protocol; "
            "[this report generator](../scripts/update_strategy_guide.py) derives tables from completed records. "
            "The historical report below preserves earlier policies, different samples, and its 500-move limit. Statements there about "
            "unexamined variants describe the earlier stage, now extended above."
        ),
    ]
    return "\n\n".join(sections)


def historical_report(original):
    if START in original or END in original:
        require(original.count(START) == original.count(END) == 1, "Unbalanced report markers")
        require(original.index(START) < original.index(END), "Reversed report markers")
        return original.split(END, 1)[1].lstrip()
    title = re.search(r"^#\s+(.+)$", original, re.MULTILINE)
    require(title is not None, "Historical guide has no title")
    body = original[:title.start()] + original[title.end():]
    output, fence = [], None
    for line in body.splitlines():
        token = re.match(r"^\s*(`{3,}|~{3,})", line)
        if token:
            fence = None if fence and token[1][0] == fence[0] and len(token[1]) >= len(fence) else token[1]
        if fence is None and re.match(r"^#{1,5}\s", line):
            line = "#" + line
        output.append(line)
    return "## Historical report: " + title[1] + "\n\n" + "\n".join(output).strip() + "\n"


def document(original, report):
    return TITLE + "\n\n" + START + "\n\n" + report + "\n\n" + END + "\n\n" + historical_report(original).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="variant-study", help="Result filename prefix")
    parser.add_argument("--output", type=Path, default=GUIDE, help="Destination; defaults to the canonical guide")
    parser.add_argument("--check", action="store_true", help="Check the destination matches current evidence without writing it")
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix:
        parser.error("--prefix must be a filename prefix")
    if args.prefix != "variant-study" and args.output.resolve() == GUIDE.resolve():
        parser.error("Use --output for a smoke/custom study; it must not overwrite the canonical guide")
    try:
        records = load_study(args.prefix)
        earnings = None
        if args.prefix == "variant-study":
            from brute_force.vegas_earnings import build_report
            earnings = json.loads((RESULTS / "vegas-earnings.json").read_text(encoding="utf-8"))
            require(earnings == build_report(), "Vegas earnings report is stale or inconsistent with its outcomes")
        report = expanded_report(records, args.prefix, earnings)
        output = document(GUIDE.read_text(encoding="utf-8"), report)
        require(document(output, report) == output, "Report generation is not idempotent")
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        parser.exit(1, f"Cannot generate report: {error}\n")
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != output:
            parser.exit(1, f"Report is stale: {args.output}\n")
        print(f"Report matches numerical evidence: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {args.output}; expanded section: {len(report.split()):,} words.")


if __name__ == "__main__":
    main()
