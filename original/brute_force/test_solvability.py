import json
from pathlib import Path
import unittest

from solitaire.game import DeckConfig, MoveKind

from brute_force.prove_stock_order import (
    passes_to_emit,
    stock_order_pass_histogram,
    universal_pass_bound,
    worst_case_passes,
)
from brute_force.standard_bounds import (
    blocker_events,
    chernoff_kl_lower_bound,
    count_blocker_event_pair,
    count_unrestricted_completion,
    direct_foundation_win_lower_bound,
    hoeffding_lower_bound,
)
from brute_force.solvability import (
    ExactSolver,
    SolvabilityDataset,
    build_initial_state,
    expected_canonical_deal_count,
    expected_canonical_suit_pattern_count,
    iter_canonical_deals,
    iter_canonical_suit_patterns,
    iter_canonical_tableau_patterns,
    iter_stock_collapsed_deals,
    iter_suit_pattern_rank_deals,
    suit_symmetry_maps,
)


class BruteForceSolvabilityTests(unittest.TestCase):
    def test_k2_has_eight_rule_preserving_suit_symmetries(self):
        maps = suit_symmetry_maps(2)
        self.assertEqual(len(maps), 8)
        self.assertEqual(len(set(maps)), 8)
        self.assertTrue(all(sorted(mapping) == [0, 1, 2, 3] for mapping in maps))

    def test_n1_has_three_canonical_deals(self):
        deals = list(iter_canonical_deals(n=1, k=2))
        self.assertEqual(expected_canonical_deal_count(n=1, k=2), 3)
        self.assertEqual(len(deals), 3)

    def test_direct_enumerator_matches_lexicographic_orbits_for_n2(self):
        direct = set(iter_suit_pattern_rank_deals(n=2, k=2))
        lexicographic = set(iter_canonical_deals(n=2, k=2))
        self.assertEqual(direct, lexicographic)

    def test_n3_has_46200_canonical_suit_patterns(self):
        patterns = list(iter_canonical_suit_patterns(n=3))
        self.assertEqual(expected_canonical_suit_pattern_count(3), 46_200)
        self.assertEqual(len(patterns), 46_200)

    def test_n4_has_90300_canonical_tableau_patterns(self):
        self.assertEqual(
            sum(1 for _ in iter_canonical_tableau_patterns(n=4, tableau_cards=10)),
            90_300,
        )

    def test_stock_collapsed_deals_store_stock_in_sorted_order(self):
        deal = next(iter_stock_collapsed_deals(n=4, k=2))
        self.assertEqual(len(deal), 16)
        self.assertEqual(set(deal), set(range(16)))
        self.assertEqual(deal[10:], tuple(sorted(deal[10:])))

    def test_six_stock_cards_need_at_most_four_passes(self):
        self.assertEqual(
            stock_order_pass_histogram(6),
            {1: 132, 2: 424, 3: 160, 4: 4},
        )

    def test_seven_stock_cards_can_need_five_passes(self):
        witness = (3, 5, 2, 6, 1, 4, 0)
        self.assertEqual(passes_to_emit(witness), 5)
        self.assertEqual(worst_case_passes(7), 5)

    def test_each_pass_can_always_emit_at_least_one_target(self):
        self.assertEqual(universal_pass_bound(24), 24)

    def test_standard_probability_bound_helpers(self):
        self.assertGreater(
            chernoff_kl_lower_bound(600, 1_000, 1e-3),
            hoeffding_lower_bound(600, 1_000, 1e-3),
        )
        self.assertEqual(
            direct_foundation_win_lower_bound().numerator,
            1,
        )
        self.assertEqual(
            count_unrestricted_completion((1, 1, 1), 52, 0, 0),
            52 * 51 * 50,
        )
        self.assertEqual(
            count_unrestricted_completion((1, 0, 0), 10, 0, 5),
            15,
        )

    def test_blocker_pair_count_is_symmetric(self):
        events = blocker_events()
        left = events[36]
        right = events[177]
        self.assertEqual(
            count_blocker_event_pair(left, right),
            count_blocker_event_pair(right, left),
        )

    def test_deal_sequence_puts_remaining_cards_in_draw_order(self):
        config = DeckConfig(n=2, k=2)
        state = build_initial_state(tuple(range(8)), config)
        self.assertEqual([len(pile) for pile in state.tableau], [1, 2, 3])
        self.assertEqual(state.stock[-1].suit, 3)
        self.assertEqual(state.stock[-1].rank, 1)
        drawn = state.apply_move(state.legal_moves()[0])
        self.assertEqual(state.legal_moves()[0].kind, MoveKind.DRAW)
        self.assertEqual(drawn.waste[-1].suit, 3)
        self.assertEqual(drawn.waste[-1].rank, 1)

    def test_exact_solver_classifies_every_n1_canonical_deal(self):
        config = DeckConfig(n=1, k=2)
        solver = ExactSolver()
        outcomes = [
            solver.is_solvable(build_initial_state(deal, config))
            for deal in iter_canonical_deals(n=1, k=2)
        ]
        self.assertEqual(len(outcomes), 3)

    def test_saved_n2_dataset_is_valid_and_complete(self):
        metadata_path = Path(__file__).with_name("results") / "k2_n2_t3.json"
        dataset = SolvabilityDataset.load(metadata_path)
        self.assertEqual(dataset.deal_count, 5_040)
        self.assertEqual(
            sum(dataset.is_solvable(index) for index in range(dataset.deal_count)),
            5_004,
        )

    def test_saved_n3_dataset_is_valid_and_uses_direct_enumeration(self):
        metadata_path = Path(__file__).with_name("results") / "k2_n3_t3.json"
        dataset = SolvabilityDataset.load(metadata_path)
        self.assertEqual(dataset.deal_count, 59_875_200)
        self.assertEqual(dataset.metadata["solvable_canonical_deals"], 59_375_520)
        first_saved_deal, first_outcome = next(iter(dataset.iter_deals()))
        first_direct_deal = next(iter(iter_suit_pattern_rank_deals(n=3, k=2)))
        self.assertEqual(first_saved_deal, first_direct_deal)
        self.assertTrue(first_outcome)

    def test_n4_benchmark_records_completed_collapsed_sweep(self):
        benchmark_path = (
            Path(__file__).with_name("results") / "k2_n4_t4.benchmark.json"
        )
        benchmark = json.loads(benchmark_path.read_text(encoding="ascii"))
        self.assertEqual(
            benchmark["automorphism_reduction"]["canonical_deals"],
            expected_canonical_deal_count(4, 2),
        )
        self.assertEqual(
            benchmark["benchmark"]["exact_only"]["solvable_deals"],
            benchmark["benchmark"]["model_then_exact"]["solvable_deals"],
        )
        self.assertEqual(benchmark["tuning"]["full_parameter_count"], 17)
        self.assertEqual(
            benchmark["stock_collapsed_run"]["solvable_tableau_deals"],
            3_579_359_602,
        )

    def test_n4_collapsed_metadata_records_stock_order_proof(self):
        metadata_path = (
            Path(__file__).with_name("results")
            / "k2_n4_t4.stock_collapsed.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        self.assertEqual(metadata["status"], "exhaustive")
        self.assertEqual(metadata["canonical_deals"], 3_632_428_800)
        self.assertEqual(metadata["equivalence"]["combined_orbit_size"], 5_760)
        self.assertEqual(
            metadata["equivalence"]["stock_order_proof"]["max_passes_required"],
            4,
        )
        self.assertEqual(
            metadata["solvable_canonical_deals"]
            + metadata["unsolvable_canonical_deals"],
            metadata["canonical_deals"],
        )

    def test_standard_probability_bounds_are_recorded(self):
        metadata_path = (
            Path(__file__).with_name("results") / "k2_n13_t7.bounds.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        self.assertEqual(metadata["config"]["n"], 13)
        self.assertEqual(metadata["absolute_bounds"]["blocker_event_count"], 440)
        self.assertAlmostEqual(
            metadata["absolute_bounds"]["upper"],
            0.988189534591097,
        )
        self.assertEqual(metadata["constructive_sample"]["wins"], 13_544)
        self.assertEqual(metadata["constructive_sample"]["model_wins"], 10_631)
        self.assertEqual(metadata["constructive_sample"]["exact_wins"], 2_913)
        self.assertEqual(
            metadata["constructive_sample"]["exact_proven_losses"],
            110,
        )
        self.assertEqual(
            metadata["constructive_sample"]["model_source"],
            "brute_force/results/k2_n13_t7.tuning-stage4.json",
        )
        self.assertEqual(len(metadata["constructive_sample"]["model_weights"]), 31)
        self.assertGreater(
            metadata["constructive_sample"]["solvability_lower_bound"],
            0.659,
        )


if __name__ == "__main__":
    unittest.main()
