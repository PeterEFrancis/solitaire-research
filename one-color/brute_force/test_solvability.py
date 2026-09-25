from itertools import permutations
import unittest

from brute_force.prove_stock_order import (
    passes_to_emit,
    stock_order_pass_histogram,
    worst_case_passes,
)
from brute_force.solvability import (
    ExactReserveSolver,
    build_reserve_state,
    can_build,
    default_tableau_count,
    expected_canonical_deal_count,
    expected_canonical_suit_pattern_count,
    iter_canonical_deals,
    iter_canonical_suit_patterns,
    iter_canonical_tableau_patterns,
    iter_stock_collapsed_deals,
    iter_suit_pattern_rank_deals,
    make_face_up,
    ordered_is_solvable,
    suit_symmetry_maps,
    tableau_card_count,
)


class BruteForceSolvabilityTests(unittest.TestCase):
    def test_k2_has_two_rule_preserving_suit_symmetries(self):
        maps = suit_symmetry_maps(2)
        self.assertEqual(maps, ((0, 1), (1, 0)))

    def test_n2_has_twelve_canonical_deals(self):
        deals = list(iter_canonical_deals(n=2, k=2))
        self.assertEqual(expected_canonical_deal_count(n=2, k=2), 12)
        self.assertEqual(len(deals), 12)

    def test_direct_enumerator_matches_lexicographic_orbits_for_n2(self):
        direct = set(iter_suit_pattern_rank_deals(n=2, k=2))
        lexicographic = set(iter_canonical_deals(n=2, k=2))
        self.assertEqual(direct, lexicographic)

    def test_n3_has_ten_canonical_suit_patterns(self):
        patterns = list(iter_canonical_suit_patterns(n=3, k=2))
        self.assertEqual(expected_canonical_suit_pattern_count(3, 2), 10)
        self.assertEqual(len(patterns), 10)

    def test_default_tableau_uses_the_most_even_split(self):
        self.assertEqual(default_tableau_count(2, 2), 2)
        self.assertEqual(default_tableau_count(4, 2), 2)
        self.assertEqual(tableau_card_count(4, 2), 3)
        self.assertEqual(default_tableau_count(7, 2), 3)
        self.assertEqual(tableau_card_count(7, 2), 6)
        self.assertEqual(default_tableau_count(13, 2), 5)
        self.assertEqual(tableau_card_count(13, 2), 15)

    def test_n7_has_32_canonical_tableau_patterns(self):
        patterns = list(
            iter_canonical_tableau_patterns(n=7, k=2, tableau_cards=6)
        )
        self.assertEqual(len(patterns), 32)

    def test_n7_has_eight_stock_cards(self):
        self.assertEqual(
            14 - tableau_card_count(7, 2),
            8,
        )
        self.assertEqual(worst_case_passes(8), 5)

    def test_stock_collapsed_deals_store_stock_in_sorted_order(self):
        deal = next(iter_stock_collapsed_deals(n=3, k=2))
        tableau_cards = tableau_card_count(3, 2)
        self.assertEqual(len(deal), 6)
        self.assertEqual(set(deal), set(range(6)))
        self.assertEqual(
            deal[tableau_cards:],
            tuple(sorted(deal[tableau_cards:])),
        )

    def test_four_stock_cards_need_at_most_two_passes(self):
        self.assertEqual(stock_order_pass_histogram(4), {1: 14, 2: 10})
        self.assertEqual(worst_case_passes(4), 2)
        self.assertEqual(passes_to_emit((2, 0, 3, 1)), 2)

    def test_same_color_cards_can_build_by_rank(self):
        self.assertTrue(can_build(make_face_up(4), (make_face_up(5),), 3))

    def test_exact_solver_classifies_the_unique_n1_canonical_deal(self):
        solver = ExactReserveSolver(n=1)
        outcomes = [
            solver.is_solvable(build_reserve_state(deal, n=1, k=2))
            for deal in iter_canonical_deals(n=1, k=2)
        ]
        self.assertEqual(outcomes, [True])

    def test_reserve_and_every_stock_order_agree_for_n3(self):
        solver = ExactReserveSolver(n=3)
        tableau_cards = tableau_card_count(3, 2)
        for collapsed in iter_stock_collapsed_deals(n=3, k=2):
            expected = solver.is_solvable(
                build_reserve_state(collapsed, n=3, k=2)
            )
            tableau = collapsed[:tableau_cards]
            for stock in permutations(collapsed[tableau_cards:]):
                self.assertEqual(
                    ordered_is_solvable(tableau + stock, n=3, k=2),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
