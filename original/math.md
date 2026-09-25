# Mathematical bounds for the standard game

## Question and rules

Let `p` be the probability that a uniformly random deal is solvable for

```text
n = 13, k = 2, t = 7.
```

Thus the deck has 52 distinct cards, the tableau piles have sizes 1 through 7,
and the stock has 24 cards. The rules used here are draw one, at most three
recycles, movable packed face-up tableau stacks, and no
foundation-to-tableau moves. Solvability means that the complete deal is known
and some legal sequence of moves wins; it is not the win rate of a particular
human or learned policy.

## Result

The unconditional, deterministic bounds are

$$
\frac{1}{77811258016843059832113070080000000}
\le p \le
\frac{43187711299211}{43703874396000}.
$$

Numerically,

$$
1.2851610749996352 \mathbin{\times} 10^{-35}
\le p \le 0.988189534591097.
$$

A fresh constructive search using the frozen 31-feature policy gives the much
more useful one-sided confidence statement

$$
0.6596641225626343 \le p \le 0.988189534591097,
$$

or

$$
65.9664123\% \le p \le 98.8189535\%,
$$

with 99.9999% confidence for the lower endpoint under the IID-uniform Monte
Carlo interpretation. The upper endpoint is deterministic and has no sampling
assumption.

The current 42-feature stage-7 player was tuned after this sample was run. The
confidence statement therefore remains attached to the frozen 31-feature
stage-4 policy and is not recomputed from the later player's 54.1265% policy
win rate.

## Deterministic lower bound

Fix any total order of the 52 cards that respects Ace-through-King order within
each suit. View the initial deal as eight availability chains:

```text
the seven tableau piles, read from exposed card downward: 1, 2, ..., 7 cards
the stock, read in draw order:                            24 cards
```

If the relative card order in every chain agrees with the fixed total order,
then the cards can be moved directly to the foundations in that order. Whenever
a card is due, it is exposed at the head of its chain, and all lower cards of
its suit have already reached the foundation.

Among all `52!` ordered deals, the fraction satisfying these eight relative
order constraints is

$$
\frac{1}{1!2!3!4!5!6!7!24!}
=
\frac{1}{77811258016843059832113070080000000}.
$$

This family is deliberately tiny, but every deal in it is certainly solvable.

## Deterministic upper bound

Consider a tableau card `C` of rank 2 through Queen with at least three cards
beneath it in its initial pile. The card is permanently blocked if all of the
following are beneath `C` in that pile:

1. At least one lower-ranked card of the same suit.
2. The first opposite-color card one rank higher.
3. The second opposite-color card one rank higher.

The same-suit card prevents `C` from moving to its foundation. Both possible
tableau supports are inaccessible beneath `C`, and `C` is not a King that can
move to an empty pile. Therefore `C` can never leave its pile, so the deal is
unsolvable.

There are ten eligible tableau positions and 44 eligible target cards, giving
440 events `E_i`. For a target of rank `r` with `d` cards beneath it, write

$$
(a)_b = \frac{a!}{(a-b)!}.
$$

The exact event probability is

$$
\Pr(E_{r,d}) =
\frac{(d)_2\left((49)_{d-2}-(50-r)_{d-2}\right)}{(52)_{d+1}}.
$$

Summing all 440 event probabilities gives

$$
S_1 = \sum_i \Pr(E_i)
= \frac{1490324}{125423025}.
$$

Exact integer enumeration of every pair intersection gives

$$
S_2 = \sum_{i<j} \Pr(E_i \cap E_j)
= \frac{53429826907}{742965864732000}.
$$

The second-order Bonferroni inequality now proves

$$
\begin{aligned}
\Pr(\text{unsolvable})
&\ge S_1-S_2 \\
&= \frac{516163096789}{43703874396000} \\
&= 0.011810465408903011.
\end{aligned}
$$

Consequently,

$$
p \le 1-\Pr(\text{unsolvable})
\le \frac{43187711299211}{43703874396000}
=0.988189534591097.
$$

The pair count is a computer-assisted exact calculation: it enumerates the
`440 choose 2` event pairs and counts injections of distinct cards into the
affected labeled positions using integer arithmetic and inclusion-exclusion.
Floating point is used only to display the final fractions.

## Constructive confidence bound

The confirmatory run used a fixed solver consisting of the frozen 31-feature
stage-4 policy followed, when necessary, by search limited to 100,000 expanded
nodes. The policy, search limit, and confidence level were fixed before the new
seed was evaluated.
It produced

```text
deals                    20,000
constructive wins        13,544
model wins               10,631
exact fallbacks           9,369
budget-exhausted deals    6,346
observed win rate        0.6772
```

Every accepted win contains a legal solution. Budget-exhausted deals are
unresolved, not classified as losses. If `q` is the fixed solver's probability
of finding a solution, then `q <= p`.

For Bernoulli relative entropy

$$
D(x\mathbin\|y)
=x\log\frac{x}{y}+(1-x)\log\frac{1-x}{1-y},
$$

the one-sided Chernoff inequality bounds the chance of observing a success rate
at least `x` when the true rate is `y < x` by

$$
\exp\left(-N D(x\mathbin\|y)\right).
$$

With

$$
N=20000,\qquad x=\frac{13544}{20000},\qquad \alpha=10^{-6},
$$

solving

$$
N D(x\mathbin\|q_L)=\log(1/\alpha)
$$

gives

$$
q_L=0.6596641225626343.
$$

The resulting one-sided procedure has coverage at least `1-alpha = 0.999999`.
Since `q <= p`, the same value is a lower confidence bound for solvability.
The actual run used reproducible pseudorandom seeds, so this statement relies
on interpreting those shuffles as IID uniform deals; the deterministic bounds
above do not.

## Interpretation

The observed 67.72% is a solver success rate, not an estimate that only 67.72%
of deals are solvable. The unresolved deals may contain many additional wins.
The exact value of `p` for this repository's rules remains unknown.

For comparison, the archived 17-feature run found 13,012 constructive wins and
gave the lower endpoint 0.6327456382895958. The new independent run raises the
99.9999%-confidence lower endpoint by about 2.69185 percentage points. This is
a comparison of two independent samples rather than a paired-deal comparison.

Blake and Gent report `81.945% +/- 0.084%` for a related thoughtful Klondike
variant, which suggests that a value in the low 80s is plausible. Its draw,
redeal, stack-movement, and foundation rules are not identical, so that result
is context rather than a bound for this game:

<https://arxiv.org/abs/1906.12314>

## Reproduction

The exact counting and confidence calculation are implemented in
[`brute_force/standard_bounds.py`](brute_force/standard_bounds.py). The saved
inputs and outputs are in
[`brute_force/results/k2_n13_t7.bounds.json`](brute_force/results/k2_n13_t7.bounds.json).
The superseded 17-feature sample is preserved in
[`brute_force/results/k2_n13_t7.bounds-17-feature.json`](brute_force/results/k2_n13_t7.bounds-17-feature.json).

Run:

```bash
python3 -m brute_force.standard_bounds --wins 13544 --games 20000 \
  --alpha 1e-6 --seed 2026071507 --exact-node-limit 100000 \
  --model-source brute_force/results/k2_n13_t7.tuning-stage4.json \
  --model-max-steps 500 --threads 8 --model-wins 10631 \
  --exact-fallbacks 9369 --budget-exhaustions 6346 \
  --expanded-positions 679980828 --elapsed-seconds 432.871 \
  --output brute_force/results/k2_n13_t7.bounds.json
```
