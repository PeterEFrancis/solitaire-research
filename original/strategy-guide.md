# Solitaire Research: how rules change strategy

<!-- BEGIN GENERATED VARIANT STUDY -->

This study compares five Klondike versions on **250,000 fresh deals per version**, using the same test seed and frozen policies. It separates compact, observable strategies from full-information computer searches. The results measure these algorithms; neither optimal play nor human performance has been established.

## The five games

| Variant | Cards per draw | Maximum stock passes | Splitting face-up stacks |
|:--|:--|:--|:--|
| Draw one / four passes | 1 | 4 | Allowed |
| Draw one / unlimited | 1 | Unlimited | Allowed |
| Draw three / four passes | 3 | 4 | Allowed |
| Draw three / unlimited | 3 | Unlimited | Allowed |
| Vegas / one pass | 1 | 1 | Whole visible run only |

All versions use 52 cards, seven tableau columns, alternating descending building, kings in empty columns, and ascending suit foundations with no return moves. Four passes includes three recycles. Draw three turns up at most three remaining cards; only the waste's top card is playable. Vegas transfers move the entire visible run; individual top cards may still go to foundations.

**Unlimited is a rule, not an unlimited computation.** Every trajectory in this study has a 1,000-move cap and repeated-position avoidance. A cutoff is a policy failure within that budget, not a proof that the deal is unsolvable.

## What the independent test found

| Variant | Stage 8 baseline | Tuned full (95% interval) | Tuned visible | Tuned eight-feature | Restart portfolio (95% interval) |
|:--|:--|:--|:--|:--|:--|
| Draw one / four passes | 54.3632% | 54.3632% (54.168 to 54.558%) | 50.9080% | 49.1440% | 65.3200% (65.133 to 65.506%) |
| Draw one / unlimited | 54.6848% | 56.0012% (55.807 to 56.196%) | 51.7768% | 50.2260% | 68.2348% (68.052 to 68.417%) |
| Draw three / four passes | 5.7500% | 10.9476% (10.826 to 11.071%) | 10.5976% | 10.3824% | 16.0768% (15.933 to 16.221%) |
| Draw three / unlimited | 13.5448% | 16.5932% (16.448 to 16.740%) | 15.7584% | 14.8720% | 28.2364% (28.060 to 28.413%) |
| Vegas / one pass | 0.7104% | 3.0440% (2.977 to 3.112%) | 2.6300% | 2.5828% | 3.8960% (3.821 to 3.973%) |

Stage 8 is freshly evaluated under each rule. Full policies can use 42 features and inspect unknown cards. Visible policies use audited current-position features; compact policies use at most eight. Both exclude buried-waste memory and hidden identities. These computer rates include exact bookkeeping and cycle avoidance; human learnability was not tested.

The portfolio restarts up to six frozen policies on the **same completely known deal**, accepting any winning trajectory or the highest foundation count. These separate attempts cost extra computation. The bank includes the selected full policy, preserving its wins. Success supplies legal winning paths, not an exact solvability probability.

Win-rate intervals are 95% Wilson intervals. Gains below use approximate paired normal 95% intervals in percentage points, based on common-deal outcomes. They assume IID-uniform sampling as an interpretation of seeded shuffles. Intervals are marginal, not simultaneous, and describe frozen policies rather than global optima.

| Variant | Full minus stage 8 (points) | Portfolio minus full (points) | Nonzero weights: full / visible / eight | Restart attempts |
|:--|:--|:--|:--|:--|
| Draw one / four passes | +0.0000 (0.000 to 0.000) | +10.9568 (10.834 to 11.079) | 35/23/8 | 5 |
| Draw one / unlimited | +1.3164 (1.167 to 1.465) | +12.2336 (12.105 to 12.362) | 34/21/8 | 6 |
| Draw three / four passes | +5.1976 (5.069 to 5.327) | +5.1292 (5.043 to 5.216) | 23/23/8 | 6 |
| Draw three / unlimited | +3.0484 (2.892 to 3.205) | +11.6432 (11.517 to 11.769) | 36/21/8 | 6 |
| Vegas / one pass | +2.3336 (2.269 to 2.398) | +0.8520 (0.816 to 0.888) | 36/23/8 | 6 |

Negative gains do not trigger reselection. Cross-rule comparisons also change selected weights, so they do not isolate a rules-only effect. Nonzero coefficient counts can include terms that never fire under a rule, such as Vegas recycling.

## A scorecard for each rule

Score each eligible legal move and take the highest score. These are the frozen eight-feature weights. **L** is a transferred stack's length and **h** is the source column's face-down-card count before a revealing move. The feature called **safe foundation** is an ace, or a legal foundation move whose opposite-color foundations have both reached at least one rank below the moving card. This tests tableau support; it does not prove that immediately removing a waste card is best under draw-three packet timing. The benchmark does not force these moves, and failing the test does not make a foundation move illegal.

| Variant | S | R | B | D | W | H | C | P |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| Draw one / four passes | 11.5 | 5 | −1 | −1 | 3.5 | 1 | −3.5 | 1 |
| Draw one / unlimited | 11 | 5 | −1 | −1 | 2 | 1 | 1.5 | 1 |
| Draw three / four passes | 13.5 | 7 | −1 | −2 | 2 | 1 | −3 | 1 |
| Draw three / unlimited | 11.5 | 3.5 | −1 | −0.5 | 2 | 1 | −1 | 1 |
| Vegas / one pass | 6.5 | 3 | −1 | −1 | 2.5 | 1 | 0.5 | 1 |

Coefficients: foundation-support test **S**, reveal **R**, tableau cards **B**, stock action **D**, waste-to-tableau **W**, reveal depth **H**, recycle **C**, productive transfer **P**. Their common B=−1, P=1, H=1 simplify the scores below.

| Move | Draw 1 / four | Draw 1 / unlimited | Draw 3 / four | Draw 3 / unlimited | Vegas |
|:--|:--|:--|:--|:--|:--|
| Draw | −1 | −1 | −2 | −0.5 | −1 |
| Recycle | −4.5 | 0.5 | −5 | −1.5 | Unavailable |
| Waste to tableau | 2.5 | 1 | 1 | 1 | 1.5 |
| Foundation passing support test, no reveal | 11.5 | 11 | 13.5 | 11.5 | 6.5 |
| Tableau transfer revealing a card | 5 + h | 5 + h | 7 + h | 3.5 + h | 3 + h |
| Transfer emptying source, no reveal | 0 | 0 | 0 | 0 | 0 |
| Other tableau transfer | −L | −L | −L | −L | −L |
| Bonus when a foundation move reveals a card | 5 + h | 5 + h | 7 + h | 3.5 + h | 3 + h |

For example, in Vegas a revealing transfer with four face-down cards in its source scores 7 (independent of run length), beating a nonrevealing foundation move passing the support test at 6.5; drawing scores −1. In unlimited draw-one, recycling scores 0.5. A foundation move failing the support test starts at 0 and still receives any reveal bonus.

Vegas has no recycle or partial-run transfer. Revealing and emptying are mutually exclusive. The program excludes repeated positions and redundant whole-column relocations to empty columns. Ties favor stock actions, then waste moves, then tableau sources left to right; foundations precede transfers, generated longest first. Equivalent empty destinations use the first column.

## Turning the scores into play

Uncover blocked columns, use the foundation-support test, and distinguish productive transfers from rearrangement. The coefficients express competing priorities, not unconditional instructions. A reserve card may still be needed as an intermediate tableau support.

Unlimited draw-one allows returning later, changing waste urgency compared with a last pass, without guaranteeing progress. Under draw three, removing cards changes later packet alignment. Inspect the exposed waste card; one-card stock habits need rethinking.

Some buried draw-three packet cards were never exposed on top. Their identities are not automatically valid memory features. Visible and eight-feature policies therefore omit all three waste-history features in every variant. Full policies can inspect hidden identities through successor scoring; their and the portfolio's rates are not established human-performance rates.

## The Vegas objective: cards returned, not only complete wins

This advisor-inspired experiment pays **$5 per foundation card, minus $52 for the deck**. Turning a card face up earns nothing. These explicitly defined rules are not a claim about all casinos. Optimizing complete wins can select a different policy from optimizing foundation cards.

| Policy | Win rate | Mean foundation cards | Mean net return | 95% interval for mean return |
|:--|:--|:--|:--|:--|
| Stage 8 | 0.7104% | 6.9587 | −$17.21 | −$17.31 to −$17.10 |
| Win-focused full | 3.0440% | 9.1737 | −$6.13 | −$6.31 to −$5.96 |
| Visible | 2.6300% | 8.9925 | −$7.04 | −$7.20 to −$6.87 |
| Eight-feature | 2.5828% | 8.9733 | −$7.13 | −$7.30 to −$6.97 |
| Foundation-focused | 2.8536% | 9.2548 | −$5.73 | −$5.90 to −$5.56 |
| Restart portfolio | 3.8960% | 10.0498 | −$1.75 | −$1.94 to −$1.56 |

The separately selected foundation-focused policy has 35 nonzero weights and was optimized for foundation count, breaking ties by wins and moves. Return intervals use a normal approximation and the recorded foundation totals and squared totals. They describe the fixed policy and shuffle model, without guaranteeing profit. Portfolio returns use full-information planning across multiple attempts.

## Search, confirmation, and computation limits

Training used 5,000 common deals (seed 2026092601), rule-sensitive starting points, a coordinate sweep, and 24 sparse mutation proposals per family. The training record contains 231–340 distinct candidate evaluations per variant across its families. Up to 4 leading candidates plus the family's starting policy were evaluated on 50,000 different deals (seed 2026092602). Validation selected single policies and the restart bank. Complete wins were primary, except for the Vegas foundation objective.

All selections were frozen before the 250,000-deal confirmation (seed 2026092699). Every frozen policy was reported. The test selected no weights or bank members. This search was not exhaustive. Unlimited-pass models fixed consumed-pass-pressure coefficients to zero.

The following counts reached the 1,000-move cap without winning. Single-policy entries count deals out of 250,000; the portfolio column counts capped attempts across its entire bank, so the same deal can appear more than once.

| Variant | Stage 8 | Full | Visible | Eight-feature | Portfolio capped / total attempts |
|:--|:--|:--|:--|:--|:--|
| Draw one / four passes | 0 | 0 | 0 | 0 | 0 / 1,250,000 |
| Draw one / unlimited | 0 | 0 | 0 | 0 | 0 / 1,500,000 |
| Draw three / four passes | 0 | 0 | 0 | 0 | 0 / 1,500,000 |
| Draw three / unlimited | 0 | 0 | 0 | 0 | 0 / 1,500,000 |
| Vegas / one pass | 0 | 0 | 0 | 0 | 0 / 1,500,000 |

## Evidence and reproduction

The [frozen protocol](brute_force/results/variant-study.protocol.json), [training and validation record](brute_force/results/variant-study.training.json), [confirmation results](brute_force/results/variant-study.confirmation.json), and [selected weights](brute_force/results/variant-study.selected-policies.json) record the rules, seeds, and outcomes. Indexed outcomes, source hashes, and executable hashes document provenance. Different C++ standard-library shuffles can produce different deals; exact replay requires matching the build.

Build the native player from the repository root, then run a selected policy in Python:

```bash
python3 scripts/build_native.py
cd original
python3 -m solitaire.play_variant draw3_unlimited --policy simple_eight --seed 7
```

Python's seeded deal is not an indexed native replay. Use `--policy portfolio` for the restart bank; other variants are `draw1_limited`, `draw1_unlimited`, `draw3_limited`, and `vegas`. The [study runner](brute_force/variant_study.py) supplies the experimental protocol; [this report generator](../scripts/update_strategy_guide.py) derives tables from completed records. The historical report below preserves earlier policies, different samples, and its 500-move limit. Statements there about unexamined variants describe the earlier stage, now extended above.

<!-- END GENERATED VARIANT STUDY -->

## Historical report: Two observable Klondike scorecards

These scorecards turn observable solitaire features into explicit playing rules. The eight-term card won 48.7172% of 250,000 untouched deals in computer simulation; the five-term card won 42.9332%. Both score moves using visible cards and simple counts, without unknown card identities or buried-waste memory. Human performance and learnability have not been measured.

The rules are draw one, at most three recycles (four stock passes), descending alternating-color tableau stacks, kings only in empty columns, and no returning cards from foundations. Partial face-up stacks may move. Advice about stock timing below applies to this limited-pass game.

### Independent comparison

All four policies below were fixed before this 250,000-deal test. Each saw the same deals and had a 500-move limit. Intervals are 95% Wilson intervals under the IID-uniform interpretation of reproducible seeded shuffles.

| Policy | Active scoring terms | Wins | Win rate | 95% interval | Average moves |
|:--|--:|--:|--:|:--|--:|
| Five-term card | 5 | 107,333 | 42.9332% | 42.739–43.127% | 239.99 |
| Eight-term card | 8 | 121,793 | 48.7172% | 48.521–48.913% | 114.04 |
| Older 17-feature model | 13 | 119,264 | 47.7056% | 47.510–47.901% | 129.18 |
| Visible-information model | 23 | 127,055 | 50.8220% | 50.626–51.018% | 127.98 |

The eight-term card added 2,529 wins over the older model, a gain of 1.0116 percentage points (approximate paired 95% interval: 0.9016–1.1216 points). The older model had 17 available features and 13 nonzero weights. Average moves includes failed games. The five-term card often makes many more moves, so the eight-term card is the more promising practical guide despite having three extra terms.

The 23-term visible-information model masks the unknown-card and waste-history terms of stage 8; it was not separately retuned under the information restriction.

Among these tested policies, moving from five to eight terms adds 5.7840 percentage points; going from eight to 23 adds another 2.1048 points. This is evidence of a useful compact strategy, not proof that either complexity class has been optimized completely.

### How to use a scorecard

List the legal moves, score each, and choose the highest score. Use one column throughout the game. Scores from different columns are not compared.

A foundation move is **safe** if its card is an ace, or both opposite-color foundations have reached at least one rank below that card. For example, a black 6 passes this test when both red foundations have reached 5. “Unsafe” here means it does not pass this sufficient safety test; such a move remains legal.

**L** is the number of cards transferred between tableau columns. **H** is the number of face-down cards in the source column immediately before a move that uncovers one of them. A move uncovers only the next card; H measures how deeply blocked that column was.

| Legal move | Five-term score | Eight-term score |
|:--|--:|--:|
| Draw the next stock card | −6 | −1 |
| Recycle the waste into the stock | −6 | −3 |
| Play the waste card onto the tableau | 3 | 1 |
| Move a card to a foundation | 40 if safe, otherwise 0 | 10 if safe, otherwise 0 |
| Transfer a tableau stack and uncover a face-down card | 20 − L | 5 + H |
| Transfer a tableau stack and empty its source, without uncovering a card | −L | 0 |
| Any other tableau transfer | −L | −L |

**Foundation reveal bonus:** If a foundation move also uncovers a face-down tableau card, add 20 in the five-term scorecard, or 5 + H in the eight-term scorecard.

The five-term column multiplies the original scores by four, eliminating fractions without changing choices. Its original weights are safe foundation +10, reveal +5, tableau build −0.25 per card, draw/recycle −1.5, and waste-to-tableau +1. The eight-term column needs no scaling: safe foundation +10, reveal +5, tableau build −1 per card, draw/recycle −1, waste-to-tableau +2, reveal depth +1, productive transfer +1 per card, and an extra recycle penalty −2.

For example, moving a three-card stack to uncover one of four hidden cards scores 17 in the five-term system and 9 in the eight-term system. Moving the same stack merely to empty its source scores −3 or 0, respectively; both beat drawing in their scorecard.

### The five-term strategy without arithmetic

For the 52-card game, the five-term scorecard is exactly this priority list, applied to eligible moves:

1. Make a safe foundation move, preferring one that uncovers a face-down card.
2. Uncover a face-down card. Prefer doing so by a foundation move; otherwise transfer the shortest available stack.
3. Play the waste card onto the tableau.
4. Make another foundation move.
5. Make a tableau transfer of one to five cards, preferring the shortest.
6. Draw, or recycle if the stock is exhausted and a recycle remains.
7. If no earlier action is available, make the shortest remaining tableau transfer.

A legal packed stack contains at most 13 cards, so every revealing transfer scores at least 7, above a waste-to-tableau move at 3. Nonrevealing six-card transfers tie drawing at −6; the computer draws first. This explains the cutoff after five cards. Apply the eligibility and tie rules below to reproduce the scorecard exactly.

### What the scores encourage

Both scorecards encourage safe foundation moves and uncovering tableau cards. A playable waste card scores above drawing, so the player usually uses it before covering it. The eight-term system gives extra value to uncovering deeply blocked columns and cancels the transfer penalty for moves that reveal cards or empty columns. It also makes recycling more costly than drawing. These are competing considerations, rather than an unconditional instruction to make any one kind of move.

A three-term system containing only safe foundations, reveals, and a tableau penalty misses stock timing: waste-to-tableau scores −1 while drawing scores 0. It therefore draws past useful waste cards whenever those are the competing choices. Adding features can change basic priorities, not merely refine rare decisions.

### Exact implementation and practical limits

The computer avoids previously visited positions and stops after 500 moves. Ties use generation order: draw/recycle first; then waste-to-foundation and waste-to-tableau, choosing destinations left to right; then tableau sources left to right, foundation first, longer transfers before shorter ones. Native evaluation omits relocating an entire column into an empty column and retains the first equivalent empty destination. Reproducing a measured computer win rate requires these details and reliable position memory. The scorecards themselves inspect no unknown cards and require no memory of buried waste, but a person following only their broad priorities has not been separately tested.

### Improvement to the full model

Before selecting the human scorecards, we improved the full-information stage-7 model by removing seven weights. Stage 8 keeps 35 active terms in the same 42-feature implementation. On a separate, frozen one-million-deal test it won 542,959 games (54.2959%), versus 539,896 (53.9896%) for stage 7: 3,063 additional wins, or 0.3063 percentage points. The approximate paired 95% interval for that gain is 0.2701–0.3425 points. Stage 8 is now the standard saved policy; stage-7 weights and historical results are preserved.

The removed terms are `non_reveal_tableau_move`, `empty_king`, `empty_king_queen_access`, `foundation_lag`, `draw_tableau_moves`, `foundation_rank`, and `tableau_to_tableau`. Removing a fitted term does not mean its underlying idea is always bad: other terms overlap with it. The full model still uses unknown-card information, and its million-deal rates use a different sample from the scorecard table.

### Which features are usable by a person?

The audit separates the 42 original features into 28 computed from the current visible position, three requiring memory of earlier waste cards, and 11 that can inspect an unknown card through simulated moves. The scorecards use only the first category. The additional memory features concern what playing the waste card would expose; under draw one those buried cards were previously visible. The unknown-card features include whether the next stock card is playable and the usefulness of a card about to be uncovered. Those cannot be presented as rules based only on what a player currently sees.

The practical core is safe foundation play, exposing hidden cards, using the waste before drawing past it, allowing productive stack transfers, and managing stock timing. The eight-term card adds depth of the blocked column, removes the stack-length penalty when a transfer reveals or clears a column, and discourages premature recycling. These results concern draw one with three recycles; they do not yet answer the advisor's draw-three or unlimited-recycle comparison.

### Experiment record and reproduction

The full-model screen evaluated ten candidates on 75,000 deals (seed 2026092501), selected among the baseline and three leading candidates on 200,000 different deals (2026092502), then confirmed one frozen candidate on one million deals (2026092599).

The human-strategy study screened 27 candidates on 25,000 training deals (2026092511), evaluated eight finalists on 200,000 validation deals (2026092512), then froze the best five-term, eight-term, and visible-information candidates plus the older reference. Every frozen policy was reported on 250,000 new deals (2026092598); no further tuning or selection used that sample.

Source files, parameters, build hashes, and indexed outcomes are recorded in:

- [Full-model confirmation](brute_force/results/k2_n13_t7.tuning-stage8.json)
- [Human-strategy protocol](brute_force/results/k2_n13_t7.human-strategy-protocol.json)
- [Human-strategy confirmation](brute_force/results/k2_n13_t7.human-strategy-confirmation.json)
- [Frozen human-policy weights](brute_force/results/k2_n13_t7.human-frozen-policies.json)
- [Feature-information audit and starting candidates](brute_force/results/human-strategy-candidates.json)

From `original/`, using the locally built native executable:

```bash
python3 -m brute_force.compare_policies \
  --policies brute_force/results/k2_n13_t7.human-frozen-policies.json \
  --baseline archived_17 --binary brute_force/native_solver.local \
  --deals 250000 --seed 2026092598 --threads 8 \
  --output /tmp/solitaire-human-strategy-replay.json
```

The local executable was built with Apple clang 17 and the MacOSX15.4 SDK. The saved ARM executable from earlier experiments is preserved separately. Different standard-library shuffle implementations may produce different deals; build provenance is therefore recorded, and all comparisons within a study use the same executable.

Validation includes 55 passing project tests, hidden-card-permutation checks for the observable features, native outcome-recording checks, and Python/native outcome parity for the improved model on all 5,040 canonical `n=2` deals. That parity check is a small-deck check, not a claim of identical trajectories for every standard-deck policy. The printed scorecards were also checked against the feature scorer on 677 legal moves across 300 reachable positions.
