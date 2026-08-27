"""Fisher information and active design: does it know when more data
cannot help, and does it pick the experiment that can?
"""
import math
import unittest

from qem_auditor.active_design import (
    DesignCandidate,
    DesignError,
    add,
    fisher_information,
    identifiability,
    is_symmetric,
    jacobi_eigen,
    log_det,
    quadratic_form,
    rank_d_optimal,
    rank_for_direction,
    recommend,
)


class EigenTest(unittest.TestCase):
    def test_diagonal_matrix_returns_its_diagonal_sorted(self):
        vals, _ = jacobi_eigen([[3.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(vals[0], 1.0)
        self.assertAlmostEqual(vals[1], 3.0)

    def test_known_two_by_two(self):
        # [[2,1],[1,2]] has eigenvalues 1 and 3
        vals, vecs = jacobi_eigen([[2.0, 1.0], [1.0, 2.0]])
        self.assertAlmostEqual(vals[0], 1.0, places=9)
        self.assertAlmostEqual(vals[1], 3.0, places=9)
        # eigenvector for lambda=1 is (1,-1)/sqrt2
        v = [row[0] for row in vecs]
        self.assertAlmostEqual(abs(v[0]), abs(v[1]), places=9)

    def test_eigenvalues_are_ascending_so_the_weakest_is_first(self):
        vals, _ = jacobi_eigen([[5.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 2.0]])
        self.assertEqual(vals, sorted(vals))

    def test_eigenvectors_are_orthonormal(self):
        _, vecs = jacobi_eigen([[4.0, 1.0], [1.0, 3.0]])
        col0 = [row[0] for row in vecs]
        col1 = [row[1] for row in vecs]
        self.assertAlmostEqual(sum(a * a for a in col0), 1.0, places=9)
        self.assertAlmostEqual(sum(a * b for a, b in zip(col0, col1)), 0.0, places=9)

    def test_asymmetric_input_is_rejected_with_a_reason(self):
        with self.assertRaises(DesignError) as ctx:
            jacobi_eigen([[1.0, 2.0], [3.0, 4.0]])
        self.assertIn("symmetric", str(ctx.exception))

    def test_ragged_matrix_is_rejected(self):
        with self.assertRaises(DesignError):
            jacobi_eigen([[1.0, 2.0], [3.0]])


class FisherTest(unittest.TestCase):
    def test_information_grows_with_measurements(self):
        one = fisher_information([[1.0]], [0.1])
        many = fisher_information([[1.0]] * 10, [0.1] * 10)
        self.assertAlmostEqual(many[0][0], 10 * one[0][0])

    def test_information_falls_with_noise(self):
        precise = fisher_information([[1.0]], [0.1])
        noisy = fisher_information([[1.0]], [1.0])
        self.assertGreater(precise[0][0], noisy[0][0])

    def test_result_is_symmetric_by_construction(self):
        f = fisher_information([[1.0, 0.5], [0.2, 1.0], [0.7, 0.3]], [0.1, 0.2, 0.1])
        self.assertTrue(is_symmetric(f))

    def test_mismatched_sigma_count_is_rejected(self):
        with self.assertRaises(DesignError) as ctx:
            fisher_information([[1.0], [1.0]], [0.1])
        self.assertIn("one uncertainty per", str(ctx.exception))

    def test_nonpositive_sigma_is_rejected(self):
        with self.assertRaises(DesignError):
            fisher_information([[1.0]], [0.0])


class IdentifiabilityTest(unittest.TestCase):
    """The finding that would have saved H4 iterations."""

    def _blind_design(self):
        # Only ever measures p0 - p1, so the SUM is invisible.
        return fisher_information([[1.0, -1.0], [1.0, -1.0], [2.0, -2.0]],
                                  [0.1, 0.1, 0.1])

    def test_a_blind_direction_is_detected(self):
        ident = identifiability(self._blind_design(), ["p_ZZ", "p_GPi2"])
        self.assertFalse(ident.is_identifiable)
        self.assertAlmostEqual(ident.lambda_min, 0.0, places=8)

    def test_the_blind_direction_is_named(self):
        ident = identifiability(self._blind_design(), ["p_ZZ", "p_GPi2"])
        described = ident.describe_weak_direction()
        self.assertIn("p_ZZ", described)
        self.assertIn("p_GPi2", described)

    def test_the_verdict_says_more_samples_will_not_help(self):
        ident = identifiability(self._blind_design(), ["p_ZZ", "p_GPi2"])
        self.assertIn("will not solve this", ident.verdict())
        self.assertIn("Change the design", ident.verdict())

    def test_a_well_posed_design_is_identifiable(self):
        f = fisher_information([[1.0, 0.0], [0.0, 1.0]], [0.1, 0.1])
        ident = identifiability(f, ["a", "b"])
        self.assertTrue(ident.is_identifiable)
        self.assertIn("More samples", ident.verdict())

    def test_condition_number_is_infinite_when_singular(self):
        self.assertEqual(identifiability(self._blind_design()).condition_number,
                         float("inf"))


class RankingTest(unittest.TestCase):
    def _blind(self):
        return fisher_information([[1.0, -1.0]] * 3, [0.1] * 3)

    def test_more_of_the_same_adds_nothing_along_a_blind_direction(self):
        """The whole point: free is not the same as useful."""
        current = self._blind()
        ident = identifiability(current)
        same = DesignCandidate("same_circuit",
                               fisher_information([[1.0, -1.0]] * 20, [0.1] * 20))
        ranked = rank_for_direction(current, [same], ident.weak_direction())
        self.assertAlmostEqual(ranked[0].score, 0.0, places=8)

    def test_a_circuit_that_sees_the_blind_direction_wins(self):
        current = self._blind()
        candidates = [
            DesignCandidate("same", fisher_information([[1.0, -1.0]] * 20, [0.1] * 20)),
            DesignCandidate("new_axis", fisher_information([[1.0, 0.0]], [0.1]),
                            cost_usd=25.79),
        ]
        _, ranked = recommend(current, candidates, ["p0", "p1"])
        self.assertEqual(ranked[0].candidate.candidate_id, "new_axis")

    def test_cost_matters_between_two_that_both_see_it(self):
        current = self._blind()
        ident = identifiability(current)
        cheap = DesignCandidate("cheap", fisher_information([[1.0, 0.0]], [0.1]),
                                cost_usd=25.0)
        pricey = DesignCandidate("pricey",
                                 fisher_information([[1.0, 0.0]] * 10, [0.1] * 10),
                                 cost_usd=6825.0)
        ranked = rank_for_direction(current, [cheap, pricey], ident.weak_direction())
        self.assertEqual(ranked[0].candidate.candidate_id, "cheap")

    def test_recommend_picks_the_directional_criterion_when_blind(self):
        _, ranked = recommend(self._blind(),
                              [DesignCandidate("x", fisher_information([[1.0, 0.0]], [0.1]))],
                              ["p0", "p1"])
        self.assertEqual(ranked[0].criterion, "directional")

    def test_recommend_picks_d_optimal_when_already_identifiable(self):
        current = fisher_information([[1.0, 0.0], [0.0, 1.0]], [0.1, 0.1])
        _, ranked = recommend(current,
                              [DesignCandidate("x", fisher_information([[1.0, 0.0]], [0.1]))])
        self.assertEqual(ranked[0].criterion, "D-optimal")

    def test_d_optimal_rewards_making_a_singular_design_identifiable(self):
        current = fisher_information([[1.0, -1.0]] * 3, [0.1] * 3)
        fixes = DesignCandidate("fixes", fisher_information([[1.0, 1.0]], [0.1]))
        does_not = DesignCandidate("does_not",
                                   fisher_information([[1.0, -1.0]] * 5, [0.1] * 5))
        ranked = rank_d_optimal(current, [fixes, does_not])
        self.assertEqual(ranked[0].candidate.candidate_id, "fixes")

    def test_empty_candidates_returns_empty(self):
        self.assertEqual(rank_for_direction(self._blind(), [], [1.0, 0.0]), [])


class LinearAlgebraHelpersTest(unittest.TestCase):
    def test_log_det_of_identity_is_zero(self):
        self.assertAlmostEqual(log_det([[1.0, 0.0], [0.0, 1.0]]), 0.0, places=9)

    def test_log_det_of_singular_is_negative_infinity(self):
        """Honest: infinite uncertainty in some direction."""
        self.assertEqual(log_det([[1.0, 1.0], [1.0, 1.0]]), float("-inf"))

    def test_quadratic_form(self):
        self.assertAlmostEqual(quadratic_form([[2.0, 0.0], [0.0, 3.0]], [1.0, 1.0]), 5.0)

    def test_quadratic_form_rejects_a_mismatched_vector(self):
        with self.assertRaises(DesignError):
            quadratic_form([[1.0, 0.0], [0.0, 1.0]], [1.0])

    def test_add_rejects_mismatched_sizes(self):
        with self.assertRaises(DesignError):
            add([[1.0]], [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
