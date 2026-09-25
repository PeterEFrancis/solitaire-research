# Brute-force solvability

September 25 update: the standard saved player is now stage 8, with 35 active
terms and an independently confirmed gain over stage 7. The newer
[strategy guide](../strategy-guide.md) documents this comparison and the
observable five-term/eight-term scorecards. The historical tuning results and
stage-4 solvability bounds described below remain preserved.

This folder classifies solitaire deals for a fixed `k`. The `n=1` through
`n=3` datasets are exhaustive modulo rule-preserving suit labels. The `n=4`
dataset additionally uses the proved fact that the order of its six stock cards
does not change solvability when four passes are available.

A model win is accepted as a constructive proof. Every model failure goes
through exact state-graph search, so the model cannot create a false positive.

Run the default `k=2` sweep from the repository root:

```bash
python3 -m brute_force.run
```

The sweep removes only rule-preserving suit labels: suits may be exchanged
within each color, and the two colors may be exchanged. For `k=2`, every orbit
contains eight raw deals. The canonical representative is the lexicographically
smallest deal in its orbit, so classifying it classifies all eight deals.

## Result format

Each packaged `(n, k, t)` result has two files in `results/`:

- `*.json` records the configuration, counts, deterministic deal ordering,
  checksums, and solver statistics.
- `*.solvable.bits.gz` stores one bit per canonical deal. Bit `i % 8` of byte
  `i // 8` is one when reduced deal `i` is solvable.

`SolvabilityDataset.load(...)` checks both compressed and uncompressed SHA-256
digests. `SolvabilityDataset.iter_deals()` regenerates canonical deals in order
and pairs them with their saved outcomes, ready for a later player benchmark.

The default sweep stops before a size with more than 1,000,000 canonical deals.
Raise `--max-canonical-deals` deliberately to attempt a larger exhaustive run.

For `n=3`, compile and run the checkpointed native exact solver:

```bash
clang++ -std=c++17 -O3 -DNDEBUG -pthread brute_force/native_solver.cpp \
  -o brute_force/native_solver
./brute_force/native_solver --n 3 --threads 8 \
  --output brute_force/results/k2_n3_t3.native.bits.partial \
  --completion brute_force/results/k2_n3_t3.native.patterns.complete
python3 -m brute_force.finalize_native --n 3
```

Restarting the same native command skips completed suit patterns. The native
move engine was checked deal-for-deal against the Python exact solver for all
5,040 canonical `n=2` deals, plus a mixed `n=3` sample containing known losses.

## Completed sweep

| n | k | t | Raw deals | Canonical deals | Solvable raw deals | Rate |
|---:|---:|---:|----------:|----------------:|-------------------:|-----:|
| 1 | 2 | 2 | 24 | 3 | 24 | 100.0000% |
| 2 | 2 | 3 | 40,320 | 5,040 | 40,032 | 99.2857% |
| 3 | 2 | 3 | 479,001,600 | 59,875,200 | 475,004,160 | 99.165464% |

The fully ordered-stock sweep stops before `n=4`: its 20,922,789,888,000 raw
deals reduce to 2,615,348,736,000 deals after the proven eight-way suit/color
reduction.

## n=4 stock-collapsed result

For `n=4`, the 10-card ordered tableau leaves six stock cards. Treating those
six cards as an unordered reserve adds a 720-way reduction, leaving
3,632,428,800 reduced deals in 90,300 canonical tableau-suit patterns.

| n | k | t | Reduced deals | Solvable | Unsolvable | Rate |
|---:|---:|---:|--------------:|---------:|-----------:|-----:|
| 4 | 2 | 4 | 3,632,428,800 | 3,579,359,602 | 53,069,198 | 98.539016% |

Under stock-order equivalence, this expands to 20,617,111,307,520 solvable and
305,678,580,480 unsolvable raw deals. The result bitset is 454,053,600 bytes
uncompressed and 8,517,753 bytes compressed.

### Stock-order proof

An unordered-reserve win removes the six stock cards in some order. Fix that as
the target order and relabel the cards `0` through `5`. For any actual stock
order, repeatedly do the following:

1. Draw cards onto the waste stack.
2. Whenever the next target card is on top, make its move from the reserve
   solution and then make the same intervening tableau and foundation moves.
3. Pop any newly exposed target cards; recycle the remaining waste when the
   stock is empty.

The extra cards waiting in the waste do not affect tableau or foundation move
legality. A played target card is in exactly the place it occupied in the
reserve solution, so induction reproduces the complete reserve win.

It remains only to bound the number of passes. The executable finite proof in
`brute_force/prove_stock_order.py` checks all `6! = 720` relative draw orders:

| Passes needed | Orders |
|--------------:|-------:|
| 1 | 132 |
| 2 | 424 |
| 3 | 160 |
| 4 | 4 |

Thus every prescribed removal order can be realized in at most four passes,
which is exactly the initial pass plus three permitted recycles. Conversely,
every ordered-stock win is plainly a legal reserve win. The two formulations
therefore have identical solvability for this six-card stock.

Run the finite part of the proof with:

```bash
python3 -m brute_force.prove_stock_order
```

As independent checks, all `n=3` deals had stock-invariant outcomes; 100,000
`n=4` tableaus had all 720 orders checked; and ordinary-stock and reserve search
agreed on 4,649,472 reduced `n=4` deals.

### General stock-order theorem

Let `m` be the number of stock cards and `R` the number of permitted recycles.
For a full triangular tableau,

```text
m = max(0, 2nk - t(t + 1) / 2).
```

Define `P(m)` as the maximum, over all `m!` relative draw orders, of the number
of waste-stack passes needed to reproduce an arbitrary prescribed reserve
removal order. The replay argument above proves stock-order equivalence whenever

```text
R + 1 >= P(m).
```

For every `m`, `P(m) <= m`: the next target card is present in the remaining
stock, so each complete pass removes at least that card. Consequently, allowing
at least `m - 1` recycles proves stock-order equivalence for any `n`, `k`, and
`t` under draw-one rules.

The sharper small-stock values begin as follows:

| Stock cards `m` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|----------------:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---:|
| `P(m)` | 1 | 1 | 2 | 2 | 3 | 4 | 5 | 5 | 6 | 7 |

The four-pass guarantee therefore ends at six stock cards. For example, the
relative order `(3, 5, 2, 6, 1, 4, 0)` needs five passes to emit the target
order `0, 1, ..., 6`. This shows that the reserve-replay proof cannot justify a
four-pass stock quotient once `m >= 7`; it does not by itself prove that a
particular larger solitaire deal is order-sensitive.

With `k=2`, default `t`, and three recycles:

| n | t | Stock cards | `P(m)` | Four-pass proof |
|--:|--:|------------:|-------:|:----------------|
| 1 | 2 | 1 | 1 | yes |
| 2 | 3 | 2 | 1 | yes |
| 3 | 3 | 6 | 4 | yes |
| 4 | 4 | 6 | 4 | yes |
| 5 | 4 | 10 | 7 | no |
| 6 | 5 | 9 | 6 | no |

For the standard `(n=13, k=2, t=7)` game, `m=24`. Twenty-three recycles are
a simple sufficient bound; four passes require a different, game-specific
argument and cannot be obtained from the general reserve-replay theorem.

Reproduce or resume the reduced run with:

```bash
./brute_force/native_solver --n 4 --threads 8 --collapse-stock-order \
  --stock-as-reserve \
  --output brute_force/results/k2_n4_t4.stock_collapsed.bits.partial \
  --completion brute_force/results/k2_n4_t4.stock_collapsed.patterns.complete
python3 -m brute_force.finalize_stock_collapsed
```

The complete sweep took about 93 minutes on eight threads and expanded
72,264,410,042 positions.

## n=4 performance

The full 17-parameter model was tuned for `(n=4, k=2, t=4)` and mirrored in
the native solver. Its move choices match the Python player deal-for-deal on the
complete `n=2` canonical set.

The native exact solver now forces provably safe foundation moves, canonicalizes
tableau-pile order in its state cache, and removes equivalent empty-column
successors. On one million deterministic random `n=4` deals:

| Mode | Model wins | Exact wins | Deals/second |
|------|-----------:|-----------:|-------------:|
| Model only | 952,272 | - | 459,934 |
| Model then exact | 952,272 | 985,209 | 137,132 |
| Exact only | - | 985,209 | 125,534 |

The model proves 95.23% of sampled deals and now improves total throughput by
about 9%. A direct suit-only sweep would still take roughly 0.60 machine-years
and need a 326,918,592,000-byte raw bitset. Collapsing stock order is the change
that makes the completed run practical.

## n=13 player tuning

The standard-game player now has 42 move features. Stage 5 made a fine
continuation of the original 31 weights. Stage 6 added and separately tuned 11
tactical features for revealed-card access, stock timing, empty-column
opportunities, and foundation safety. Stage 7 then refined all 42 weights on
250,000 common-random training deals and selected checkpoints on 500,000
separate validation deals.

On a final untouched million-deal seed, the stage-7 policy won 541,265 games
(54.1265%). The frozen stage-4 31-feature policy won 532,941 games (53.2941%)
on the same deals, so the new policy gained 8,324 wins or 0.8324 percentage
points. The tuning records are
`results/k2_n13_t7.tuning-stage5.json`,
`results/k2_n13_t7.tuning-stage6-new-features.json`, and
`results/k2_n13_t7.tuning-stage7.json`.

Check Python/native policy parity on every canonical `n=2` deal with:

```bash
python3 -m brute_force.check_model_parity \
  --model-source brute_force/results/k2_n13_t7.tuning-stage7.json
```

Reproduce the common-deal feature-ablation ranking in `notes.md` with:

```bash
python3 -m brute_force.rank_features \
  --model-source brute_force/results/k2_n13_t7.tuning-stage7.json \
  --output brute_force/results/k2_n13_t7.feature-ablation.json
```

The constructive probability bound below deliberately remains tied to the
older frozen stage-4 policy and its preregistered sample. Improving the current
player does not retroactively change that statistical result.

## n=13 probability bounds

Let `p` be the probability that a uniformly random deal is solvable under this
repository's standard rules: `(n=13, k=2, t=7)`, draw one, at most three
recycles, movable packed tableau stacks, and no foundation-to-tableau moves.

The unconditional, fully deterministic bounds are

```text
1 / 77811258016843059832113070080000000 <= p
                                              <= 43187711299211 / 43703874396000
1.285161075e-35                              <= p <= 0.988189534591097.
```

The lower bound counts one deliberately tiny family. Fix a legal order in
which all 52 cards go directly to their foundations. Require each of the seven
tableau piles, read from exposed card downward, and the 24-card stock, read in
draw order, to agree with that order. The relative-order constraints have
probability

```text
1 / (1! 2! 3! 4! 5! 6! 7! 24!).
```

For the upper bound, consider a tableau card of rank 2 through Queen. It is
permanently blocked when one lower card of its own suit and both opposite-color
cards one rank higher all lie beneath it in the same initial pile. The card
cannot reach its foundation, cannot build onto the tableau, and is not a King
that can enter an empty pile. Therefore every deal with such a card is
unsolvable.

There are 440 target-card/position events. Their exact probability sum is
`1490324 / 125423025`; the exact sum of all pair intersections is
`53429826907 / 742965864732000`. The second-order Bonferroni inequality gives

```text
P(unsolvable) >= sum P(E_i) - sum P(E_i intersect E_j)
              = 516163096789 / 43703874396000
              = 0.011810465408903011.
```

A fresh confirmatory run used the frozen 31-feature stage-4 policy followed by
search limited to 100,000 expanded nodes. It found 13,544 constructive wins in
20,000 seeded shuffles, including 10,631 wins from the policy alone. Every
accepted deal contains a complete legal solution; the 6,346 searches that
reached their node budget remain unresolved rather than being called losses.
Inverting the one-sided Bernoulli KL-Chernoff inequality at `alpha=10^-6`
gives, under the IID-uniform Monte Carlo interpretation,

```text
0.6596641225626343 <= p <= 0.988189534591097
```

with 99.9999% confidence for the lower endpoint; the upper endpoint is the
deterministic theorem above. The seeded C++ run is reproducible pseudorandom
sampling, so the absolute interval is the one that does not rely on a sampling
assumption. The previous 17-feature result is archived as
`results/k2_n13_t7.bounds-17-feature.json`; its confidence lower endpoint was
0.6327456382895958. Machine-readable current inputs and outputs are in
`results/k2_n13_t7.bounds.json`.

Recompute the exact count and confidence endpoint with:

```bash
python3 -m brute_force.standard_bounds --wins 13544 --games 20000 \
  --alpha 1e-6 --seed 2026071507 --exact-node-limit 100000 \
  --model-source brute_force/results/k2_n13_t7.tuning-stage4.json \
  --model-max-steps 500 --threads 8 --model-wins 10631 \
  --exact-fallbacks 9369 --budget-exhaustions 6346 \
  --expanded-positions 679980828 --elapsed-seconds 432.871 \
  --output brute_force/results/k2_n13_t7.bounds.json
```
