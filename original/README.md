# Solitaire AI

This is a small, dependency-free Python foundation for solitaire experiments.
It implements a Klondike-style game engine plus a stochastic player with 42
tunable move-scoring parameters.

Game shape is controlled by:

1. `n`: cards per suit.
2. `k`: suits per color, with two colors total.
3. `t`: tableau piles. When omitted,
   `t = ceil((sqrt(1 + 8nk) - 1) / 2)`, the smallest tableau that initially
   holds at least half of the `2nk`-card deck.

The 42 parameters score these move features:

1. `safe_foundation`: moving a card to a foundation when the opposite color foundations have caught up enough that the move is unlikely to block tableau play.
2. `reveal_hidden`: uncovering a face-down tableau card.
3. `empty_king`: moving a king stack into an empty tableau column.
4. `tableau_build`: moving cards into tableau structure.
5. `stock_action`: drawing or recycling the stock.
6. `foundation_move`: moving any legal card to a foundation.
7. `reveal_depth`: the number of hidden cards in a column being uncovered.
8. `empty_source`: clearing a tableau column.
9. `waste_to_tableau`: placing the waste card onto the tableau.
10. `recycle`: recycling the waste back into the stock.
11. `destination_hidden`: hidden cards under a tableau destination.
12. `destination_run`: exposed cards already consolidated at a tableau destination.
13. `blocks_foundation`: covering a card that could move to its foundation.
14. `foundation_rank`: the rank of a card moving to its foundation.
15. `source_hidden`: hidden cards in the source tableau column.
16. `creates_first_empty`: creating the first empty tableau column.
17. `first_empty_with_king`: creating that space when a playable king is visible.
18. `revealed_card_low_rank`: preferring low-ranked cards when a move exposes a known face-down card.
19. `revealed_card_foundation_ready`: exposing a card that can immediately move to its foundation.
20. `tableau_to_foundation`: distinguishing foundation moves made from the tableau.
21. `tableau_to_tableau`: distinguishing moves between tableau piles.
22. `non_reveal_tableau_move`: moving between tableau piles without revealing a card or emptying the source.
23. `productive_stack_length`: stack length when a tableau move reveals a card or empties its source.
24. `draw_playable`: drawing a stock card that is immediately playable.
25. `waste_unlocks_playable`: removing the waste top to expose another immediately playable card.
26. `empty_king_queen_access`: accessible Queens that can build on a King moved to an empty pile.
27. `next_foundation_moves`: legal foundation moves available after the candidate move.
28. `next_reveal_moves`: hidden-card reveals available after the candidate move.
29. `foundation_support_demand`: visible lower opposite-color cards that still need a foundation-bound card as support.
30. `recycle_pressure`: which limited stock recycle the move consumes.
31. `draw_stock_remaining`: stock size when drawing.
32. `revealed_card_tableau_moves`: tableau destinations available to a newly revealed card.
33. `revealed_card_foundation_distance`: foundation advances still needed before a newly revealed card can move there.
34. `draw_foundation_ready`: drawing a card that can immediately move to its foundation.
35. `draw_tableau_moves`: tableau destinations available to the card just drawn.
36. `waste_unlocks_foundation_ready`: exposing a waste card that can immediately move to its foundation.
37. `waste_unlocks_tableau_moves`: tableau destinations available to the waste card just exposed.
38. `draw_buries_playable_waste`: drawing while the current waste card is playable.
39. `waste_play_recycle_pressure`: playing the waste later in the limited sequence of stock passes.
40. `next_empty_source_moves`: legal next moves that would clear a tableau column.
41. `foundation_lag`: how far the moved card's foundation trails the most advanced foundation.
42. `unsafe_foundation_distance`: how far a foundation move advances beyond the opposite-color safety frontier.

The player uses a softmax over legal moves:

```text
P(move) = softmax(parameters . features(move) / temperature)
```

Training uses policy-gradient descent on the negative reward objective. It
supports online SGD and normalized batched Adam updates, gradient clipping, and
selectively freezing parameters while strategic features are tuned.

## Run tests

```bash
python3 -m unittest discover
```

## Play or train

Evaluate the starting policy:

```bash
python3 -m solitaire.train --episodes 0 --eval-games 100
```

Train with batched Adam and evaluate:

```bash
python3 -m solitaire.train --episodes 5000 --batch-size 64 --optimizer adam \
  --gradient-clip 10 --eval-games 1000 --max-steps 500
```

Evaluate the current standard-deck policy:

```bash
python3 -m solitaire.train --n 13 --k 2 --t 7 --episodes 0 \
  --eval-games 1000 --max-steps 500
```

Training reads and writes parameter records in
`solitaire/parameters.json`, keyed by `(n, k, t)`. Current tuning focuses on
the standard `(n=13, k=2, t=7)` game.

The current standard policy is stage 8, selected on September 25, 2026. It
retains 35 nonzero weights in the existing 42-feature model. Starting from the
coordinate-tuned stage-7 policy, a feature-removal screen used 75,000 training
deals and separate 200,000-deal validation. The selected candidate was then
frozen before testing on one million untouched matched deals:

| Policy | Wins | Win rate |
|:--|--:|--:|
| Stage 7, 42 active terms | 539,896 | 53.9896% |
| Stage 8, 35 active terms | 542,959 | 54.2959% |

The gain is 3,063 wins, or 0.3063 percentage points (approximate paired 95%
interval: 0.2701 to 0.3425 points). Stage 8 sets seven former weights to zero:
`non_reveal_tableau_move`, `empty_king`, `empty_king_queen_access`,
`foundation_lag`, `draw_tableau_moves`, `foundation_rank`, and
`tableau_to_tableau`. Other weights are unchanged. Results, full weights,
build provenance, and indexed win/loss outcomes are saved in
`brute_force/results/k2_n13_t7.tuning-stage8.json` and its referenced files.

These are native heuristic-player win rates for draw one, three recycles,
movable partial tableau stacks, and no foundation rollback. The full model
uses information about unknown cards; its result is neither a measured human
win rate nor the underlying solvability probability. Historical stage-7
results remain in `notes.md` and the stage-7 artifact; its earlier 54.1265%
rate used a different sample. The older standard solvability bounds are
unchanged.

Reproduce further native tuning with:

```bash
python3 -m brute_force.tune_model \
  --resume brute_force/results/k2_n13_t7.tuning-stage8.json \
  --output brute_force/results/k2_n13_t7.tuning-next.json
```

Use a binary built for the current computer with `--binary` when needed.
For matched-deal policy comparisons, `python3 -m brute_force.compare_policies`
accepts a JSON mapping policy names to feature weights and reports per-policy
Wilson intervals and paired differences. `--outcomes-dir` retains one byte per
deal for each policy. A zero byte means the policy did not win within its move
budget; it does not prove that the deal is unsolvable.

## Playable strategies

[The strategy guide](strategy-guide.md) gives exact five-term and eight-term
scorecards using only visible cards and counts, including a version of the
five-term strategy that requires no arithmetic. On 250,000 untouched shared
deals, the five-term card won 42.9332%, the eight-term card 48.7172%, and a
23-term visible-information model 50.8220%. These are computer-executed
strategies with cycle avoidance and fixed tie-breaking, not measurements of
human play. The guide includes the rule set, full weights, uncertainty,
information restrictions, and reproduction commands.
