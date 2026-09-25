# One-color solitaire brute-force search

This folder contains only the exact-search code for one-color solitaire. A
deck has `k` interchangeable suits of one color and `n` ranks in each suit.
Tableau stacks build downward by rank without a color restriction.

The current native search is specialized to `k=2`. The default number of
tableau piles is the valid integer `t` that minimizes

```text
|t(t + 1) - nk|.
```

This minimizes the difference between the triangular tableau size
`t(t + 1)/2` and the remaining stock size. Exact ties go to the larger
tableau. For `(n, k) = (4, 2)`, this gives `t=2`, producing a `3/5` split
instead of the less even `6/2` split. For `(n, k) = (7, 2)`, it gives `t=3`,
with six tableau cards and eight stock cards.

## Exact reduction

The two suit labels are interchangeable. Stock order may also be discarded
when every desired reserve-removal order can be realized within the four
allowed stock passes. The finite proof applies to at most six stock cards,
covering the corrected defaults through `n=6`.

The corrected `n=7` game has eight stock cards, so its search preserves stock
order. It first obtains up to four exact reserve-winning removal orders. Each
one certifies all stock permutations that can replay it in four passes, and
the native ordered-state solver checks every remaining permutation exactly.

## Completed results

| n | t | Tableau/stock | Classified cases | Solvable | Rate |
|--:|--:|:--------------|-----------------:|---------:|-----:|
| 1 | 1 | 1 / 1 | 1 | 1 | 100% |
| 2 | 2 | 3 / 1 | 12 | 12 | 100% |
| 3 | 2 | 3 / 3 | 60 | 60 | 100% |
| 4 | 2 | 3 / 5 | 168 | 168 | 100% |
| 5 | 3 | 6 / 4 | 75,600 | 75,420 | 99.7619047619% |
| 6 | 3 | 6 / 6 | 332,640 | 332,100 | 99.8376623377% |
| 7 | 3 | 6 / 8 | 43,589,145,600 | 43,538,342,400 | 99.8834498834% |

Rows through `n=6` use the proved unordered-stock quotient, so their compact
case counts omit a uniform stock-order factor. The `n=7` row counts all ordered
stock deals after only the two-way suit-label symmetry reduction.

For `n=7`, exactly 1,260 of the 1,081,080 canonical tableau deals are
unsolvable. Every one of their 40,320 stock orders is unsolvable. Every stock
order of every other tableau deal is solvable, giving the exact fraction
`857/858`.

The next case, `n=8`, has 14,529,715,200 reduced cases. A one-million-case
benchmark processed about 75,764 cases per second, projecting roughly 53.3
hours and a 1,816,214,400-byte result bitset. That full run has not been
started; the benchmark's 97.8739% sample rate is not an exact result.

## Monte Carlo result for n=13

The balanced rule gives `t=5`, with 15 tableau cards and an 11-card ordered
stock. A deterministic one-million-deal Monte Carlo run using seed `20260718`
found:

```text
solvable       943,698
unsolvable      56,302
estimate      94.3698%
95% Wilson CI 94.32445% to 94.41481%
```

Every sampled deal was classified exactly. The classifier first searches the
unordered reserve relaxation. A reserve loss proves the ordered game
unsolvable; a four-pass replay certificate proves an ordered win; all
remaining deals go through full ordered-state search. The Monte Carlo
uncertainty concerns only extrapolation from the sample to all deals.

Of the one million sampled deals, 920,143 wins were replay-certified, 56,302
were proved unsolvable by the reserve relaxation, and 23,555 required full
ordered search. All 23,555 fallback searches were wins, so this sample
contained no deal whose solvability depended negatively on stock order. This
is empirical evidence, not a general stock-order theorem for eleven cards.

Reproduce the run with:

```bash
./brute_force/native_solver \
  --n 13 \
  --threads 8 \
  --stock-mode ordered \
  --benchmark-deals 1000000 \
  --benchmark-seed 20260718
```

## Files

- `brute_force/native_solver.cpp`: checkpointed multithreaded exhaustive
  solver used for the full `(7,2)` computation.
- `brute_force/solvability.py`: independent Python reference solver and deal
  enumerators.
- `brute_force/prove_stock_order.py`: finite verification of the stock-order
  reduction.
- `brute_force/check_native.py`: compares native outcomes with the independent
  Python solver.
- `brute_force/test_solvability.py`: small correctness checks.

## Build and run

From this folder:

```bash
clang++ -std=c++17 -O3 -DNDEBUG -pthread \
  brute_force/native_solver.cpp -o brute_force/native_solver

mkdir -p brute_force/results

./brute_force/native_solver \
  --n 7 \
  --threads 8 \
  --stock-mode ordered \
  --output brute_force/results/k2_n7_t3.ordered-counts.u16 \
  --completion brute_force/results/k2_n7_t3.patterns.complete
```

The exhaustive run is resumable. Repeating the same command skips completed
tableau suit patterns.
