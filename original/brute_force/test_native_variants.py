"""Opt-in native rule parity: NATIVE_SOLVER_TEST_BINARY selects the variant build.

The harness exposes Solver internals in a temporary source copy, without adding
test-only APIs to the engine. Set CXX/NATIVE_SOLVER_TEST_CXXFLAGS if needed.
NATIVE_SOLVER_REFERENCE_BINARY optionally checks against a historical build.
"""

from dataclasses import replace
from itertools import product
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import unittest

from solitaire import Card, DeckConfig, GameState, Move, MoveKind, new_game
from solitaire.player import PARAMETER_NAMES, move_features


NATIVE_BINARY = os.environ.get("NATIVE_SOLVER_TEST_BINARY")
REFERENCE_BINARY = os.environ.get("NATIVE_SOLVER_REFERENCE_BINARY")
SOURCE = Path(__file__).with_name("native_solver.cpp")


def encoded_card(card, n):
    return card.suit * n + card.rank - 1 + (128 if card.face_up else 0)


def state_object(state):
    n = state.config.n
    return {
        "stock": [encoded_card(card, n) for card in state.stock],
        "waste": [encoded_card(card, n) for card in state.waste],
        "foundations": list(state.foundations),
        "tableau": [[encoded_card(card, n) for card in pile] for pile in state.tableau],
        "recycles": state.recycles_used,
    }


def native_eligible_moves(state):
    """Preserve the native engine's historical empty-column symmetry pruning."""
    moves, used_empty = [], set()
    for move in state.legal_moves():
        if move.kind.endswith("_to_tableau") and not state.tableau[move.dest_index]:
            if (move.kind == MoveKind.TABLEAU_TO_TABLEAU
                    and move.count == len(state.tableau[move.source_index])):
                continue
            key = (move.kind, move.source_index, move.count)
            if key in used_empty:
                continue
            used_empty.add(key)
        moves.append(move)
    return moves


@unittest.skipUnless(NATIVE_BINARY, "set NATIVE_SOLVER_TEST_BINARY to test native variants")
class NativeVariantRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "clang++"))
        if not shutil.which(compiler[0]):
            raise unittest.SkipTest("C++ compiler unavailable for direct engine harness")
        cls.directory = tempfile.TemporaryDirectory(prefix="native-variant-harness-")
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(cls.directory.name) / "harness.cpp"
        cls.binary = Path(cls.directory.name) / "harness"
        production = SOURCE.read_text().replace("private:", "public:", 1)
        production = production.replace("int main(int argc, char** argv)", "int native_cli_main(int argc, char** argv)", 1)
        source.write_text(production + r'''
void print_state(const State& state, int tableau_count) {
    auto cards = [](const auto& array, int count) {
        std::cout << "[";
        for (int i = 0; i < count; ++i) {
            if (i) std::cout << ",";
            std::cout << static_cast<int>(array[i]);
        }
        std::cout << "]";
    };
    std::cout << "{\"stock\":";
    cards(state.stock, state.stock_size);
    std::cout << ",\"waste\":";
    cards(state.waste, state.waste_size);
    std::cout << ",\"foundations\":";
    cards(state.foundations, 4);
    std::cout << ",\"tableau\":[";
    for (int i = 0; i < tableau_count; ++i) {
        if (i) std::cout << ",";
        cards(state.tableau[i].cards, state.tableau[i].size);
    }
    std::cout << "],\"recycles\":" << static_cast<int>(state.recycles) << "}";
}
int main() {
    std::string command;
    while (std::cin >> command) {
        int n, t, draw, recycles, splitting, max_steps, value;
        std::cin >> n >> t >> draw >> recycles >> splitting >> max_steps;
        State state;
        std::cin >> value; state.recycles = value;
        std::cin >> value; state.stock_size = value;
        for (int i = 0; i < state.stock_size; ++i) {
            std::cin >> value; state.stock[i] = value;
        }
        std::cin >> value; state.waste_size = value;
        for (int i = 0; i < state.waste_size; ++i) {
            std::cin >> value; state.waste[i] = value;
        }
        for (int i = 0; i < 4; ++i) {
            std::cin >> value; state.foundations[i] = value;
        }
        for (int column = 0; column < t; ++column) {
            std::cin >> value; state.tableau[column].size = value;
            for (int i = 0; i < state.tableau[column].size; ++i) {
                std::cin >> value; state.tableau[column].cards[i] = value;
            }
        }
        std::array<double, kModelFeatureCount> weights{};
        Solver solver(n, t, true, true, max_steps, weights, recycles, 0, false, draw, splitting);
        if (command == "moves") {
            std::vector<Candidate> moves;
            solver.generate_successors(state, moves, false);
            std::vector<std::array<double, kModelFeatureCount>> features(moves.size());
            for (size_t feature = 0; feature < kModelFeatureCount; ++feature) {
                solver.model_weights_.fill(0.0);
                solver.model_weights_[feature] = 1.0;
                std::vector<Candidate> scored;
                solver.generate_successors(state, scored, false);
                if (scored.size() != moves.size()) return 2;
                for (size_t i = 0; i < moves.size(); ++i) {
                    features[i][feature] = scored[i].model_score;
                }
            }
            std::cout << "[";
            for (size_t i = 0; i < moves.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << "{\"state\":";
                print_state(moves[i].state, t);
                std::cout << ",\"features\":[";
                for (size_t feature = 0; feature < kModelFeatureCount; ++feature) {
                    if (feature) std::cout << ",";
                    std::cout << features[i][feature];
                }
                std::cout << "]}";
            }
            std::cout << "]\n";
        } else if (command == "key") {
            State normalized = state;
            normalized.recycles = 0;
            std::cout << (solver.position_key(state, false) == solver.position_key(normalized, false)) << "\n";
        } else if (command == "stress") {
            bool normalized = true;
            for (int i = 0; i < 2000; ++i) {
                std::vector<Candidate> moves;
                solver.generate_successors(state, moves, false);
                if (moves.empty()) return 3;
                state = moves[0].state;
                normalized = normalized && state.recycles == 0;
            }
            std::cout << normalized << "\n";
        } else if (command == "model") {
            bool won = solver.is_solvable(state);
            std::cout << "{\"won\":" << won
                      << ",\"steps\":" << solver.model_steps
                      << ",\"foundation_cards\":" << solver.model_foundation_cards
                      << ",\"foundation_cards_squared\":" << solver.model_foundation_cards_squared
                      << ",\"cutoffs\":" << solver.model_cutoffs << "}\n";
        } else return 4;
    }
}
''', encoding="utf-8")
        flags = shlex.split(os.environ.get("NATIVE_SOLVER_TEST_CXXFLAGS", ""))
        sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk")
        if not flags and platform.system() == "Darwin" and sdk.is_dir():
            flags = ["-isysroot", str(sdk), "-isystem", str(sdk / "usr/include/c++/v1")]
        subprocess.run([*compiler, "-std=c++17", "-O1", "-pthread", *flags,
                        str(source), "-o", str(cls.binary)],
                       check=True, capture_output=True, text=True, timeout=120)

    def query(self, state, command="moves", max_steps=500):
        config, data = state.config, state_object(state)
        tokens = [command, config.n, len(state.tableau), config.draw_count,
                  -1 if config.max_recycles is None else config.max_recycles,
                  int(config.allow_tableau_stack_splitting), max_steps,
                  data["recycles"], len(data["stock"]), *data["stock"],
                  len(data["waste"]), *data["waste"], *data["foundations"]]
        for pile in data["tableau"]:
            tokens.extend([len(pile), *pile])
        result = subprocess.run([str(self.binary)], input=" ".join(map(str, tokens)) + "\n",
                                check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def assert_successors_and_features_match(self, state):
        expected = [{"state": state_object(state.apply_move(move)),
                     "features": list(move_features(state, move))}
                    for move in native_eligible_moves(state)]
        self.assertEqual(self.query(state), expected)

    def test_draw_three_short_packet_and_recycle_order(self):
        state = GameState(
            DeckConfig(n=4, k=2, t=1, draw_count=3, max_recycles=None),
            ((Card(1, 4, True),),),
            (Card(0, 1), Card(0, 2), Card(0, 3), Card(1, 1), Card(1, 2)),
            (), (0, 0, 0, 0),
        )
        original_stock = state.stock
        self.assert_successors_and_features_match(state)
        state = state.apply_move(Move(MoveKind.DRAW))
        self.assertEqual([card.rank for card in state.waste], [2, 1, 3])
        self.assert_successors_and_features_match(state)
        state = state.apply_move(Move(MoveKind.DRAW))
        self.assertEqual(len(state.stock), 0)
        self.assertEqual(len(state.waste), 5)
        self.assert_successors_and_features_match(state)
        state = state.apply_move(Move(MoveKind.RECYCLE))
        self.assertEqual(state.stock, original_stock)
        self.assertEqual(state.recycles_used, 0)
        self.assert_successors_and_features_match(state)

    def test_no_splitting_moves_complete_exposed_run(self):
        state = GameState(
            DeckConfig(n=4, k=2, t=3, allow_tableau_stack_splitting=False),
            ((Card(2, 1), Card(0, 3, True), Card(1, 2, True)),
             (Card(3, 4, True),), (Card(2, 3, True),)),
            (), (), (0, 0, 0, 0),
        )
        partial = Move(MoveKind.TABLEAU_TO_TABLEAU, 0, 2, 1)
        complete = Move(MoveKind.TABLEAU_TO_TABLEAU, 0, 1, 2)
        self.assertNotIn(partial, state.legal_moves())
        self.assertIn(complete, state.legal_moves())
        self.assert_successors_and_features_match(state)
        split = replace(state, config=DeckConfig(n=4, k=2, t=3))
        self.assertIn(partial, split.legal_moves())
        self.assert_successors_and_features_match(split)

    def test_all_variant_features_match_python_on_reachable_states(self):
        for draw, recycles, splitting in product((1, 3), (0, 2, None), (True, False)):
            state = new_game(config=DeckConfig(n=4, k=2, draw_count=draw,
                                              max_recycles=recycles,
                                              allow_tableau_stack_splitting=splitting), seed=71)
            for step in range(4):
                with self.subTest(draw=draw, recycles=recycles, splitting=splitting, step=step):
                    self.assert_successors_and_features_match(state)
                moves = native_eligible_moves(state)
                if not moves:
                    break
                state = state.apply_move(moves[-1])

    def test_unlimited_keys_pressure_and_more_than_255_recycles(self):
        config = DeckConfig(n=3, k=2, t=1, max_recycles=None)
        state = GameState(config, ((),), (), (Card(0, 2, True),), (0, 0, 0, 0), 250)
        self.assertEqual(self.query(state, "key"), 1)
        finite = replace(state, config=DeckConfig(n=3, k=2, t=1, max_recycles=255))
        self.assertEqual(self.query(finite, "key"), 0)
        successors = self.query(state)
        self.assertEqual(successors[0]["state"]["recycles"], 0)
        for successor in successors:
            self.assertEqual(successor["features"][PARAMETER_NAMES.index("recycle_pressure")], 0)
            self.assertEqual(successor["features"][PARAMETER_NAMES.index("waste_play_recycle_pressure")], 0)
        self.assertEqual(self.query(replace(state, recycles_used=0), "stress"), 1)
        # The model stops at a genuine repeated stock position, not a pass cap.
        result = self.query(replace(state, recycles_used=0), "model", max_steps=1000)
        self.assertLess(result["steps"], 10)
        self.assertEqual(result["cutoffs"], 0)

    def test_foundation_totals_and_cutoffs_cover_each_stop_reason(self):
        config = DeckConfig(n=3, k=2, t=1, max_recycles=0)
        stalled = GameState(config, ((),), (), (), (1, 2, 0, 0))
        self.assertEqual(self.query(stalled, "model"), {
            "won": 0, "steps": 0, "foundation_cards": 3,
            "foundation_cards_squared": 9, "cutoffs": 0,
        })
        won = replace(stalled, foundations=(3, 3, 3, 3))
        self.assertEqual(self.query(won, "model")["foundation_cards_squared"], 144)
        self.assertEqual(self.query(won, "model")["cutoffs"], 0)
        drawing = replace(stalled, stock=(Card(0, 3),))
        limited = self.query(drawing, "model", max_steps=1)
        self.assertEqual(limited["cutoffs"], 1)
        self.assertEqual(limited["foundation_cards"], 3)
        final_card = replace(won, foundations=(2, 3, 3, 3), waste=(Card(0, 3, True),))
        last_step_win = self.query(final_card, "model", max_steps=1)
        self.assertEqual(last_step_win["won"], 1)
        self.assertEqual(last_step_win["cutoffs"], 0)


@unittest.skipUnless(NATIVE_BINARY, "set NATIVE_SOLVER_TEST_BINARY to test native variants")
class NativeVariantBenchmarkTests(unittest.TestCase):
    def benchmark(self, directory, options=(), threads=1, binary=NATIVE_BINARY, steps=120):
        output = Path(directory) / "wins.u8"
        foundations = Path(directory) / "foundations.u8"
        command = [str(Path(binary).resolve()), "--n", "4", "--model-only",
                   "--benchmark-deals", "64", "--benchmark-seed", "20260925",
                   "--model-max-steps", str(steps), "--threads", str(threads),
                   "--benchmark-outcomes", str(output),
                   "--benchmark-foundations", str(foundations), *options]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        fields = dict(line.split(" ", 1) for line in result.stdout.splitlines())
        return fields, output.read_bytes(), foundations.read_bytes()

    def test_default_options_preserve_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = self.benchmark(directory)
            explicit = self.benchmark(directory, ("--draw-count", "1", "--max-recycles", "3"))
        self.assertEqual(plain[1:], explicit[1:])
        for name in plain[0]:
            if name not in {"elapsed_seconds", "deals_per_second"}:
                self.assertEqual(plain[0][name], explicit[0][name], name)

    def test_indexed_foundations_match_totals_and_thread_order(self):
        with tempfile.TemporaryDirectory() as directory:
            for draw, recycles, splitting in product((1, 3), (0, 2, -1), (True, False)):
                options = ["--draw-count", str(draw), "--max-recycles", str(recycles)]
                if not splitting:
                    options.append("--no-tableau-splitting")
                with self.subTest(options=options):
                    fields, wins, cards = self.benchmark(directory, options)
                    parallel, parallel_wins, parallel_cards = self.benchmark(directory, options, 3)
                    self.assertEqual(wins, parallel_wins)
                    self.assertEqual(cards, parallel_cards)
                    self.assertEqual(len(cards), 64)
                    self.assertLessEqual(max(cards), 16)
                    self.assertEqual(bytes(int(card == 16) for card in cards), wins)
                    self.assertEqual(sum(cards), int(fields["model_foundation_cards"]))
                    self.assertEqual(sum(card * card for card in cards), int(fields["model_foundation_cards_squared"]))
                    self.assertEqual(fields["model_cutoffs"], parallel["model_cutoffs"])
                    self.assertEqual(fields["exact_fallbacks"], "0")
                    self.assertEqual(fields["allow_tableau_stack_splitting"], str(int(splitting)))
            fields, _, _ = self.benchmark(directory, steps=1)
            self.assertEqual(fields["model_cutoffs"], "64")

    @unittest.skipUnless(REFERENCE_BINARY, "set NATIVE_SOLVER_REFERENCE_BINARY for historical parity")
    def test_default_matches_historical_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            fields, outcomes, _ = self.benchmark(directory)
            path = Path(directory) / "wins.u8"
            result = subprocess.run([
                str(Path(REFERENCE_BINARY).resolve()), "--n", "4", "--model-only",
                "--benchmark-deals", "64", "--benchmark-seed", "20260925",
                "--model-max-steps", "120", "--threads", "1",
                "--benchmark-outcomes", str(path),
            ], check=True, capture_output=True, text=True)
            old = dict(line.split(" ", 1) for line in result.stdout.splitlines())
            self.assertEqual(path.read_bytes(), outcomes)
            for name in old:
                if name not in {"elapsed_seconds", "deals_per_second"}:
                    self.assertEqual(old[name], fields[name], name)

    def test_variants_reject_exact_checkpoints_and_invalid_rules(self):
        base = [str(Path(NATIVE_BINARY).resolve()), "--n", "4"]
        invalid = [
            ["--draw-count", "2"], ["--max-recycles", "-2"],
            ["--max-recycles", "256"],
            ["--draw-count", "3", "--benchmark-deals", "1"],
            ["--no-tableau-splitting", "--output", "unused", "--completion", "unused"],
            ["--max-recycles", "-1", "--check-stock-orders", "1"],
            ["--draw-count", "3", "--model-only", "--benchmark-deals", "1", "--stock-as-reserve"],
            ["--benchmark-foundations", "unused", "--benchmark-deals", "1"],
            ["--model-only", "--benchmark-deals", "1", "--benchmark-foundations", "same", "--benchmark-outcomes", "same"],
        ]
        for options in invalid:
            with self.subTest(options=options):
                result = subprocess.run(base + options, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
