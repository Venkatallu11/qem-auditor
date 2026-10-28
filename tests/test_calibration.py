"""Are the quoted error bars the right size?

Validated against two real IonQ datasets, not against synthetic noise:
the 732 residuals of the first full real-hardware H4 energy run, and 39
earlier measurements whose exact answers are known from the prepared
state. Where a number here matches one the source project published
independently, that agreement is the test.
"""
import json
import math
import random
import unittest
from pathlib import Path

from qem_auditor.calibration import (CalibrationError, audit_against_truth,
                                     audit_groups, audit_residuals,
                                     compare_to_model)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ionq_full_run.json"


def load():
    return json.loads(FIXTURE.read_text())


class TheRealRunTest(unittest.TestCase):
    """84 circuits, 1,100 shots, job 01a138e7."""

    @classmethod
    def setUpClass(cls):
        data = load()["full_run"]
        # NOT cls.run: TestCase.run is what the framework calls to run
        # the test, and shadowing it with a dict breaks the runner.
        cls.meta = data
        cls.audit = audit_residuals([r["z"] for r in data["residuals"]],
                                    fitted_parameters=data["fitted_parameters"])

    def test_it_reproduces_the_published_chi2_per_dof(self):
        """Computed from the raw residuals, against the figure the source
        project published. This is the test the module exists to pass."""
        self.assertAlmostEqual(self.audit.chi2_per_dof,
                               self.meta["reported_chi2_dof"], places=9)
        self.assertAlmostEqual(self.audit.chi2_per_dof, 5.9286, places=4)

    def test_dividing_by_n_instead_of_dof_understates_it(self):
        """And understates it in the flattering direction, which is why
        fitted_parameters is a parameter and not an assumption."""
        naive = self.audit.chi2 / self.audit.n
        self.assertAlmostEqual(naive, 5.8071, places=4)
        self.assertLess(naive, self.audit.chi2_per_dof)
        self.assertEqual(self.audit.dof, 732 - 15)

    def test_the_bars_are_two_and_a_half_times_too_tight(self):
        self.assertAlmostEqual(self.audit.inflation, 2.435, places=3)
        self.assertFalse(self.audit.bars_are_honest)

    def test_it_sizes_what_the_model_is_missing(self):
        """sqrt(chi2/dof - 1): the unmodelled term, in units of the
        quoted sigma, added in quadrature."""
        self.assertAlmostEqual(self.audit.unmodelled_in_sigmas, 2.220,
                               places=3)

    def test_the_departure_is_enormous_and_reported_in_sigmas(self):
        """A p-value underflows to zero at this scale and stops carrying
        information, so the headline is a sigma count."""
        self.assertAlmostEqual(self.audit.significance_sigma, 46.0, delta=0.5)
        self.assertEqual(self.audit.p_value, 0.0)

    def test_coverage_is_far_below_gaussian(self):
        within1, within2, within3 = self.audit.coverage
        self.assertAlmostEqual(within1, 0.370, places=2)
        self.assertAlmostEqual(within3, 0.813, places=2)
        self.assertLess(within3, 0.90)  # a Gaussian would give 99.7%

    def test_this_run_is_mostly_scatter_not_bias(self):
        self.assertLess(self.audit.bias_share, 0.10)
        self.assertIn("more shots TIGHTENS each bar", self.audit.describe())


class GroupedTest(unittest.TestCase):
    """One chi-squared over a mixed set says less than it looks like."""

    @classmethod
    def setUpClass(cls):
        data = load()["full_run"]
        by_group = {}
        for row in data["residuals"]:
            by_group.setdefault(row["group"], []).append(row["z"])
        cls.grouped = audit_groups(by_group,
                                   fitted_parameters=data["fitted_parameters"])

    def test_the_groups_reproduce_the_published_breakdown(self):
        """Group 0's mean residual of -1.54 and mean square of 7.18 were
        published separately; both fall out of the same residuals."""
        group0 = self.grouped.groups[0]
        self.assertAlmostEqual(group0.chi2_per_dof, 7.179, places=3)
        self.assertAlmostEqual(group0.mean_z, -1.544, places=3)

    def test_two_different_problems_hide_in_one_headline(self):
        """Group 0 carries a systematic shift; group 2 carries the worst
        scatter. The single 5.93 shows neither."""
        self.assertEqual(self.grouped.worst, 2)
        self.assertGreater(self.grouped.groups[2].chi2_per_dof, 8.0)
        self.assertGreater(abs(self.grouped.groups[0].mean_z), 1.5)
        self.assertLess(abs(self.grouped.groups[2].mean_z), 0.5)

    def test_it_flags_a_concentrated_excess(self):
        self.assertTrue(self.grouped.concentrated)
        self.assertGreater(self.grouped.share_of_chi2(2),
                           self.grouped.share_of_points(2))
        self.assertIn("partly a fact about that subset",
                      self.grouped.describe())

    def test_the_pooled_number_matches_the_ungrouped_one(self):
        self.assertAlmostEqual(self.grouped.overall.chi2_per_dof, 5.9286,
                               places=4)


class KnownAnswersTest(unittest.TestCase):
    """39 measurements against exact expectation values."""

    @classmethod
    def setUpClass(cls):
        cls.checks = load()["correctness_checks"]
        cls.audit = audit_against_truth(
            [c["measured"] for c in cls.checks],
            [c["sigma"] for c in cls.checks],
            [c["exact"] for c in cls.checks])

    def test_there_are_thirty_nine_of_them(self):
        self.assertEqual(self.audit.n, 39)

    def test_it_reproduces_the_published_mean_z_squared(self):
        """Published as 2.73 from a z column rounded to one decimal;
        2.718 from the unrounded measurements."""
        self.assertAlmostEqual(self.audit.chi2 / self.audit.n, 2.72, places=2)

    def test_it_reproduces_the_published_three_sigma_coverage(self):
        self.assertEqual(round(self.audit.coverage[2] * 39), 37)

    def test_these_bars_are_too_tight_but_far_less_so(self):
        """1.65x against the full run's 2.43x -- these were correctness
        checks on one slot, not a fitted 84-circuit correction."""
        self.assertAlmostEqual(self.audit.inflation, 1.649, places=3)
        self.assertFalse(self.audit.bars_are_honest)


class HonestAndOverstatedBarsTest(unittest.TestCase):
    """The module has to pass a well-calibrated set, or it is only ever
    an accusation."""

    def test_well_calibrated_residuals_pass(self):
        rng = random.Random(20261028)
        audit = audit_residuals([rng.gauss(0, 1) for _ in range(2000)])
        self.assertTrue(audit.bars_are_honest, audit.describe())
        self.assertAlmostEqual(audit.chi2_per_dof, 1.0, delta=0.1)
        self.assertIn("nothing here says they are wrong", audit.describe())

    def test_bars_that_are_too_wide_are_also_called_wrong(self):
        """Two-sided on purpose: an overstated uncertainty is wrong, and
        a chi-squared far below 1 usually means a fit tracing its noise."""
        rng = random.Random(7)
        audit = audit_residuals([rng.gauss(0, 1) / 3 for _ in range(2000)])
        self.assertFalse(audit.bars_are_honest)
        self.assertIn("too WIDE", audit.describe())
        self.assertAlmostEqual(1 / audit.inflation, 3.0, delta=0.2)

    def test_a_pure_offset_is_all_bias(self):
        audit = audit_residuals([2.0] * 50)
        self.assertAlmostEqual(audit.bias_share, 1.0, places=9)
        self.assertIn("does not go away with more shots", audit.describe())

    def test_a_pure_spread_is_all_scatter(self):
        audit = audit_residuals([3.0, -3.0] * 25)
        self.assertAlmostEqual(audit.bias_share, 0.0, places=9)


class ModelComparisonTest(unittest.TestCase):
    """Did the noise model predict how WRONG the answer would be?"""

    def setUp(self):
        run = load()["full_run"]
        self.hardware = audit_residuals([r["z"] for r in run["residuals"]],
                                        fitted_parameters=run["fitted_parameters"])
        # The forte-1 simulator rehearsal of the same 84 circuits, whose
        # per-residual rows were not published -- only its chi2/dof of
        # 2.68. Reconstructed to sit at exactly that: chi2 = 2.68 * dof,
        # spread over the same 732 points with 15 fitted parameters, so
        # the scale follows from dof rather than from n.
        self.model = audit_residuals(
            [math.sqrt(2.68 * 717 / 732)] * 366
            + [-math.sqrt(2.68 * 717 / 732)] * 366,
            fitted_parameters=15)

    def test_excess_is_measured_from_one_not_from_the_model(self):
        """A model that reproduced the excess exactly would otherwise
        score as half right."""
        comparison = compare_to_model(self.hardware, self.model)
        self.assertAlmostEqual(comparison.excess_ratio, 4.9286 / 1.68,
                               places=2)
        self.assertIn("appeared only on hardware", comparison.describe())

    def test_a_model_predicting_no_excess_accounts_for_none_of_it(self):
        comparison = compare_to_model(self.hardware, audit_residuals([1.0, -1.0] * 50))
        self.assertEqual(comparison.excess_ratio, float("inf"))
        self.assertIn("predicted NO excess", comparison.describe())


class RefusalTest(unittest.TestCase):

    def test_no_residuals(self):
        with self.assertRaises(CalibrationError):
            audit_residuals([])

    def test_a_fit_with_no_degrees_of_freedom_left(self):
        with self.assertRaises(CalibrationError) as caught:
            audit_residuals([1.0, 2.0, 3.0], fitted_parameters=3)
        self.assertIn("says nothing about the error bars",
                      str(caught.exception))

    def test_a_measurement_quoted_with_no_uncertainty(self):
        with self.assertRaises(CalibrationError) as caught:
            audit_against_truth([1.0, 2.0], [0.1, 0.0], [1.0, 2.0])
        self.assertIn("cannot be tested against its error bar",
                      str(caught.exception))

    def test_mismatched_lengths_name_the_risk(self):
        with self.assertRaises(CalibrationError) as caught:
            audit_against_truth([1.0, 2.0], [0.1], [1.0, 2.0])
        self.assertIn("compared against the wrong", str(caught.exception))

    def test_negative_fitted_parameters(self):
        with self.assertRaises(CalibrationError):
            audit_residuals([1.0, 2.0], fitted_parameters=-1)

    def test_too_few_degrees_of_freedom_for_the_approximation(self):
        """Wilson-Hilferty is not trustworthy at single-digit dof, so it
        returns None instead of a number it cannot support."""
        audit = audit_residuals([1.0] * 5)
        self.assertIsNone(audit.significance_sigma)
        self.assertIsNone(audit.p_value)
        self.assertIsInstance(audit.bars_are_honest, bool)


if __name__ == "__main__":
    unittest.main()


class ThresholdTest(unittest.TestCase):
    """The threshold is 3 sigma because 2 was measurably too aggressive."""

    def test_two_sigma_falsely_accuses_honest_bars_at_scale(self):
        """The case that moved the default. 2,000 genuinely Gaussian
        residuals land at chi2/dof 0.929, which is 2.3 sigma from 1.0 --
        because chi2/dof concentrates as sqrt(2/dof) and the band gets
        narrow, not because anything is wrong with the bars."""
        rng = random.Random(20261028)
        audit = audit_residuals([rng.gauss(0, 1) for _ in range(2000)])
        self.assertFalse(audit.is_honest_at(2.0))
        self.assertTrue(audit.is_honest_at(3.0))

    def test_three_sigma_still_catches_everything_that_matters(self):
        run = load()["full_run"]
        audit = audit_residuals([r["z"] for r in run["residuals"]],
                                fitted_parameters=run["fitted_parameters"])
        self.assertFalse(audit.is_honest_at(3.0))
        self.assertFalse(audit.is_honest_at(10.0))  # it is at 46 sigma

    def test_the_small_dof_band_comes_from_the_spread_not_a_constant(self):
        """At 5 degrees of freedom chi2/dof has a spread of 0.63, so any
        fixed band is either useless or an accusation waiting to happen."""
        wide = audit_residuals([0.4] * 6)          # chi2/dof = 0.16
        self.assertTrue(wide.is_honest_at(3.0))    # inside 3 * 0.577
        self.assertFalse(wide.is_honest_at(1.0))
