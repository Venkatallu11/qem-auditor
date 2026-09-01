"""Power analysis: does it give a number a researcher can act on, and does
it refuse when the number would be meaningless?
"""
import math
import unittest

from qem_auditor import UncertaintyCoverage
from qem_auditor.power import (
    PowerError,
    analyze,
    analyze_experiment,
    chi2_quantile,
    check_scope,
    compare,
    interval_at_n,
    may_stop_early,
    mean_interval,
    normal_cdf,
    normal_quantile,
    power_at,
    rank_with_ties,
    required_n,
    sequential_alpha,
    sigma_upper_bound,
)

from .helpers import make_experiment


class NormalTest(unittest.TestCase):
    """Textbook values, since everything downstream rests on these."""

    def test_standard_quantiles(self):
        self.assertAlmostEqual(normal_quantile(0.975), 1.959964, places=5)
        self.assertAlmostEqual(normal_quantile(0.80), 0.841621, places=5)
        self.assertAlmostEqual(normal_quantile(0.50), 0.0, places=6)

    def test_cdf_inverts_quantile(self):
        for p in (0.01, 0.25, 0.5, 0.9, 0.975, 0.999):
            self.assertAlmostEqual(normal_cdf(normal_quantile(p)), p, places=6)

    def test_quantile_rejects_impossible_probability(self):
        for p in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(PowerError):
                normal_quantile(p)


class RequiredNTest(unittest.TestCase):
    def test_matches_the_textbook_case(self):
        # delta/sigma = 0.5, alpha=.05, beta=.20 -> n = 31.4 -> 32
        self.assertEqual(required_n(0.5, 1.0), 32)

    def test_rounds_up_never_down(self):
        """A sample size is a floor. Rounding a requirement down is how an
        under-powered study gets declared adequate."""
        n = required_n(0.5, 1.0)
        exact = ((normal_quantile(0.975) + normal_quantile(0.80)) / 0.5) ** 2
        self.assertGreaterEqual(n, exact)
        self.assertLess(n - exact, 1.0)

    def test_smaller_effects_need_more_runs(self):
        self.assertGreater(required_n(0.1, 1.0), required_n(0.5, 1.0))

    def test_zero_effect_is_rejected_with_a_reason(self):
        with self.assertRaises(PowerError) as ctx:
            required_n(0.0, 1.0)
        self.assertIn("infinite", str(ctx.exception))

    def test_nonpositive_sigma_is_rejected(self):
        with self.assertRaises(PowerError):
            required_n(0.5, 0.0)


class PowerAtTest(unittest.TestCase):
    def test_power_at_required_n_reaches_the_target(self):
        n = required_n(0.5, 1.0, alpha=0.05, beta=0.20)
        self.assertGreaterEqual(power_at(n, 0.5, 1.0), 0.80)

    def test_power_rises_with_n(self):
        self.assertLess(power_at(4, 0.5, 1.0), power_at(32, 0.5, 1.0))

    def test_power_is_bounded(self):
        self.assertLessEqual(power_at(10_000, 0.5, 1.0), 1.0)
        self.assertGreaterEqual(power_at(0, 0.5, 1.0), 0.0)


class SigmaBoundTest(unittest.TestCase):
    """A sample sd from 4 points is not sigma; it is a noisy guess at it."""

    def test_bound_exceeds_the_point_estimate(self):
        vals = [0.10, 0.12, 0.09, 0.11]
        from statistics import stdev

        self.assertGreater(sigma_upper_bound(vals), stdev(vals))

    def test_the_inflation_shrinks_as_n_grows(self):
        from statistics import stdev

        small = [float(i) for i in range(4)]
        large = [float(i) for i in range(20)]
        self.assertGreater(sigma_upper_bound(small) / stdev(small),
                           sigma_upper_bound(large) / stdev(large))

    def test_identical_values_cannot_bound_sigma(self):
        with self.assertRaises(PowerError):
            sigma_upper_bound([1.0, 1.0, 1.0])

    def test_chi2_quantile_is_sane(self):
        # chi2_{0.95}(1) ~ 3.841
        self.assertAlmostEqual(chi2_quantile(0.95, 1), 3.841, delta=0.15)


class ScopeMatchingTest(unittest.TestCase):
    """The design point: sigma must match what the claim has to survive."""

    def test_a_bootstrap_sigma_cannot_size_a_reproducibility_claim(self):
        gaps = check_scope(UncertaintyCoverage(shot_noise=True), "reproducible")
        self.assertIn("cross_submission", gaps)

    def test_a_full_coverage_sigma_has_no_gaps(self):
        full = UncertaintyCoverage(shot_noise=True, method_monte_carlo=True,
                                   cross_submission=True, noise_model=True)
        self.assertEqual(check_scope(full, "hardware_ready"), [])

    def test_scope_gaps_block_is_powered_however_large_n_is(self):
        """The historical mistake, in one assertion: a huge sample sized
        against the wrong variance is still not powered."""
        pa = analyze(effect_size=0.2, sigma=0.0015, current_n=10_000,
                     sigma_covers=UncertaintyCoverage(shot_noise=True),
                     claim_scope="reproducible")
        self.assertFalse(pa.is_powered)
        self.assertIn("wrong variance", pa.summary())

    def test_unknown_scope_is_rejected(self):
        with self.assertRaises(PowerError):
            check_scope(UncertaintyCoverage(), "vibes")


class MeanIntervalTest(unittest.TestCase):
    def test_interval_brackets_the_mean(self):
        ci = mean_interval([0.10, 0.12, 0.09, 0.11])
        self.assertLess(ci.low, ci.mean_kcal)
        self.assertGreater(ci.high, ci.mean_kcal)

    def test_more_runs_tighten_the_interval(self):
        vals = [0.10, 0.12, 0.09, 0.11]
        self.assertLess(interval_at_n(vals, 8).half_width_kcal,
                        mean_interval(vals).half_width_kcal)

    def test_conservative_interval_is_wider(self):
        vals = [0.10, 0.12, 0.09, 0.11]
        self.assertGreater(mean_interval(vals, conservative=True).half_width_kcal,
                           mean_interval(vals, conservative=False).half_width_kcal)

    def test_projection_never_narrows_below_the_request(self):
        vals = [0.10, 0.12, 0.09, 0.11]
        self.assertEqual(interval_at_n(vals, 2).n, 4)  # already past it


class SequentialTest(unittest.TestCase):
    def test_alpha_is_spent_across_looks(self):
        self.assertAlmostEqual(sequential_alpha(1, 5, 0.05), 0.01)

    def test_a_clear_separation_licenses_stopping(self):
        ok, why = may_stop_early([0.010, 0.011, 0.009, 0.012], threshold=0.25,
                                 total_looks=8)
        self.assertTrue(ok)
        self.assertIn("clears", why)

    def test_a_marginal_result_does_not(self):
        ok, why = may_stop_early([0.24, 0.26, 0.25, 0.27], threshold=0.25,
                                 total_looks=8)
        self.assertFalse(ok)
        self.assertIn("nominal alpha", why)

    def test_exceeding_the_planned_looks_is_refused(self):
        ok, why = may_stop_early([0.01] * 5, threshold=0.25, total_looks=3)
        self.assertFalse(ok)
        self.assertIn("planned looks", why)


class ExperimentIntegrationTest(unittest.TestCase):
    def test_analyzes_a_real_record(self):
        from benchmarks.h4_ancilla_qed import EXPERIMENT

        pa = analyze_experiment(EXPERIMENT, threshold_kcal=0.25)
        self.assertIsNotNone(pa)
        self.assertEqual(pa.current_n, 4)
        self.assertTrue(pa.sigma_is_upper_bound)
        self.assertGreater(pa.sigma_kcal, pa.sigma_point_kcal)

    def test_sigma_comes_from_independent_replicates_only(self):
        """Bootstrap replicates must not contribute to a between-run sigma."""
        from qem_auditor import Replicate, ReplicateKind

        exp = make_experiment()
        exp.outputs.replicates = [
            Replicate(0.10, ReplicateKind.BOOTSTRAP_RESAMPLE) for _ in range(8)
        ]
        self.assertIsNone(analyze_experiment(exp))

    def test_it_rederives_the_projects_own_replication_convention(self):
        """Powering against the scale the draws differ on gives ~8 --
        independently recovering the 8-draw target the project adopted by
        convention."""
        from benchmarks.h4_ancilla_qed import EXPERIMENT

        pa = analyze_experiment(EXPERIMENT, threshold_kcal=0.02)
        self.assertEqual(pa.required_n, 8)


if __name__ == "__main__":
    unittest.main()


class ComparisonTest(unittest.TestCase):
    """Whether a printed ranking is a claim the runs support.

    The shootout printed REM+ZNE above CDR on a gap of 0.14 between two
    numbers that move by ten times that between seeds. Ordering them
    asserted something the data did not, which is the over-claiming this
    package objects to everywhere else.
    """

    def test_a_large_gap_is_called(self):
        good = [1.0, 1.1, 0.9, 1.05]
        bad = [37.0, 37.2, 36.8, 37.1]
        result = compare("mitigated", good, "raw", bad)
        self.assertTrue(result.distinguishable)
        self.assertEqual(result.better, "mitigated")

    def test_a_gap_inside_the_noise_is_refused(self):
        a = [1.15, 2.5, 0.1, 0.9]
        b = [1.29, 2.1, 0.4, 1.2]
        result = compare("REM + ZNE", a, "CDR", b)
        self.assertFalse(result.distinguishable)
        self.assertIsNone(result.better,
                          "returning the nominal leader would hand back a tie silently")
        self.assertIn("not distinguishable", result.describe())

    def test_it_says_how_many_runs_would_settle_it(self):
        a = [1.15, 2.5, 0.1, 0.9]
        b = [1.29, 2.1, 0.4, 1.2]
        result = compare("a", a, "b", b)
        self.assertGreater(result.required_n, len(a))
        self.assertIn("would settle it", result.describe())

    def test_an_identical_pair_cannot_be_settled_at_any_n(self):
        """No finite experiment establishes that two methods are the same,
        so a run count is not quoted for a gap of zero."""
        values = [1.0, 2.0, 3.0, 4.0]
        result = compare("a", values, "b", list(values))
        self.assertEqual(result.gap, 0.0)
        self.assertIsNone(result.required_n)
        self.assertIn("no number of runs", result.describe())

    def test_the_difference_carries_both_methods_noise(self):
        """sqrt(2) wider than a single mean's error. Forgetting it is how
        two methods get called distinguishable on half the runs it takes."""
        a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]
        result = compare("a", a, "b", b)
        self.assertAlmostEqual(result.standard_error,
                               result.sigma * math.sqrt(2.0 / result.n))

    def test_unequal_run_counts_are_refused(self):
        with self.assertRaises(PowerError):
            compare("a", [1.0, 2.0, 3.0], "b", [1.0, 2.0])

    def test_more_runs_make_the_same_gap_resolvable(self):
        a = [1.0, 1.4, 0.6, 1.2]
        b = [1.3, 1.7, 0.9, 1.5]
        few = compare("a", a, "b", b)
        many = compare("a", a * 12, "b", b * 12)
        self.assertFalse(few.distinguishable)
        self.assertTrue(many.distinguishable)
        self.assertAlmostEqual(few.gap, many.gap)


class RankWithTiesTest(unittest.TestCase):

    def test_indistinguishable_methods_share_a_tier(self):
        tiers = rank_with_ties({
            "REM + ZNE": [1.15, 2.5, 0.1, 0.9],
            "CDR": [1.29, 2.1, 0.4, 1.2],
            "unmitigated": [37.0, 37.2, 36.8, 37.1],
        })
        self.assertIn(sorted(tiers[0]), [["CDR", "REM + ZNE"]])
        self.assertEqual(tiers[-1], ["unmitigated"])

    def test_a_clear_ordering_stays_one_per_tier(self):
        tiers = rank_with_ties({
            "best": [1.0, 1.1, 0.9, 1.0],
            "middle": [10.0, 10.1, 9.9, 10.0],
            "worst": [40.0, 40.1, 39.9, 40.0],
        })
        self.assertEqual(tiers, [["best"], ["middle"], ["worst"]])

    def test_tiers_are_ordered_best_first(self):
        tiers = rank_with_ties({
            "worst": [40.0, 40.1, 39.9, 40.0],
            "best": [1.0, 1.1, 0.9, 1.0],
        })
        self.assertEqual(tiers[0], ["best"])
