# Solitaire Research

Experiments in Klondike strategy, trained computer players, and exact solvability of smaller solitaire games.

**[Read the strategy guide](https://peterefrancis.com/solitaire-research/)** — the front page compares draw-one, draw-three, unlimited recycles, and the specified Vegas rules. It separates full-information computer models from observable scorecards and includes saved weights, uncertainty, and reproducible evidence. The canonical [Markdown guide](original/strategy-guide.md) is also available in this repository.

- [Klondike engine and research](original/README.md)
- [Exact solvability experiments](original/brute_force/README.md)
- [One-color solitaire research](one-color/README.md)

Source, saved parameters, result summaries, and packaged datasets are versioned. Local compiled executables, Python caches, and unfinished solver checkpoints are excluded; the large finished datasets remain available in compressed form.

## Run a selected player

Python 3.10+ runs the engine and playback without third-party packages. A C++17 compiler is needed for native benchmarks.

```sh
python3 scripts/build_native.py
cd original
python3 -m solitaire.play_variant draw3_unlimited --policy simple_eight --seed 7
python3 -m solitaire.play_variant vegas --policy profit --seed 7
python3 -m solitaire.play_variant draw1_unlimited --policy portfolio --seed 7 \
  --format json --output /tmp/solitaire-trace.json
```

Profiles: `draw1_limited`, `draw1_unlimited`, `draw3_limited`, `draw3_unlimited`, and `vegas`. Policies: `full`, `visible`, `simple_eight`, and `portfolio`; `profit` is additionally available for Vegas. Full/profit models can inspect hidden cards. The portfolio retries a known deal with several frozen policies; it is not an ordinary player's strategy. Python playback and native benchmarks use different shuffle implementations.

## Reproduce and check

From `original/`, `python3 -m brute_force.variant_study --output-prefix my-variant-study` repeats training, validation, and independent confirmation. It can take substantial time; the saved study uses 1,000 moves per policy attempt. Unlimited refers to the game rules, not an unlimited computation budget. See the guide for the exact protocol and platform caveats.

Run `python3 -m unittest discover` separately in `original/` and `one-color/`. Set `NATIVE_SOLVER_TEST_BINARY` to the absolute path of the built native executable to include native transition, scoring, and benchmark checks.

From `original/`, `python3 -m brute_force.verify_variant_study` independently checks saved outcomes, confidence intervals, paired comparisons, portfolio aggregation, and provenance. Rebuilt executables can differ by platform; use its optional `--binary` argument only to verify the exact recorded study executable.

The Pages site renders `docs/index.md`, generated from the canonical guide with `python3 scripts/build_pages.py`. GitHub checks that the page remains synchronized.
