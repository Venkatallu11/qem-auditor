"""The checks that run before anything is computed from a counts table.

The motivating failure is real and is reproduced here on real hardware
data: a vendor SDK returned raw counts where the client expected
probabilities and multiplied by the shot count. Every assertion about
what that does downstream is measured rather than asserted.
"""
import json
import math
import unittest
from pathlib import Path

from qem_auditor.counts import CountsError, check_counts
from qem_auditor.results import analyse, shot_noise

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ionq_forte_job.json"
OBSERVABLE = [("XYYX", 1.0)]


def real_counts() -> dict:
    return json.loads(FIXTURE.read_text())["counts"]


class CleanDataTest(unittest.TestCase):

    def test_a_real_hardware_table_passes(self):
        report = check_counts({"XYYX": real_counts()}, declared_shots=100)
        self.assertTrue(report.ok, report.describe())
        self.assertEqual(report.total_shots, 100)
        self.assertEqual(report.width, 5)

    def test_no_declared_shots_still_checks_structure(self):
        report = check_counts({"XYYX": real_counts()})
        self.assertTrue(report.ok)


class TheRealBugTest(unittest.TestCase):
    """Counts inflated by exactly the shot count, as actually happened."""

    def setUp(self):
        self.counts = real_counts()
        self.inflated = {bits: n * 100 for bits, n in self.counts.items()}

    def test_the_inflation_is_caught(self):
        report = check_counts({"XYYX": self.inflated}, declared_shots=100)
        self.assertFalse(report.ok)
        kinds = [p.kind for p in report.problems]
        self.assertEqual(kinds, ["shot_mismatch"])

    def test_nothing_about_the_estimate_looks_wrong(self):
        """Which is exactly why it needs a check rather than a reader."""
        honest = shot_noise({"XYYX": self.counts}, OBSERVABLE)
        broken = shot_noise({"XYYX": self.inflated}, OBSERVABLE)
        self.assertAlmostEqual(honest.estimate, broken.estimate, places=12)

    def test_the_bar_is_too_tight_by_the_square_root_of_the_factor(self):
        """The claim the report makes, checked against the arithmetic."""
        honest = shot_noise({"XYYX": self.counts}, OBSERVABLE)
        broken = shot_noise({"XYYX": self.inflated}, OBSERVABLE)
        self.assertAlmostEqual(honest.sigma / broken.sigma, math.sqrt(100),
                               places=6)
        problem = check_counts({"XYYX": self.inflated},
                               declared_shots=100).problems[0]
        self.assertIn("10.0x TOO TIGHT", problem.consequence)

    def test_analyse_refuses_rather_than_reporting_it(self):
        """A note under a 10x-too-tight interval is not enough. The
        interval must not be produced at all."""
        with self.assertRaises(CountsError) as caught:
            analyse({"XYYX": self.inflated}, OBSERVABLE, declared_shots=100)
        self.assertIn("shots were declared", str(caught.exception))

    def test_the_same_data_analyses_fine_when_it_is_not_corrupted(self):
        report = analyse({"XYYX": self.counts}, OBSERVABLE, declared_shots=100)
        self.assertEqual(report.noise.shots, 100)


class StructuralProblemsTest(unittest.TestCase):

    def test_probabilities_are_caught_as_probabilities(self):
        report = check_counts({"ZZ": {"00": 0.5, "11": 0.5}})
        self.assertIn("probabilities", [p.kind for p in report.problems])

    def test_a_fractional_count_is_refused(self):
        report = check_counts({"ZZ": {"00": 10.5, "11": 9.5}})
        self.assertIn("not_integral", [p.kind for p in report.problems])

    def test_a_negative_count_names_the_readout_correction(self):
        """Corrected weights ARE negative, legitimately. Re-analysing
        them as raw counts is the mistake, so the message says so."""
        report = check_counts({"ZZ": {"00": 120, "11": -8}})
        problem = next(p for p in report.problems if p.kind == "negative")
        self.assertIn("corrected WEIGHT", problem.consequence)

    def test_ragged_widths_across_settings_are_caught(self):
        report = check_counts({"ZZ": {"00": 10}, "XX": {"000": 10}})
        self.assertIn("ragged_width", [p.kind for p in report.problems])

    def test_a_declared_width_that_does_not_match_is_caught(self):
        report = check_counts({"ZZ": {"00": 10}}, declared_width=4)
        self.assertIn("width_mismatch", [p.kind for p in report.problems])

    def test_a_non_binary_key_is_caught(self):
        report = check_counts({"ZZ": {"0x": 10}})
        self.assertIn("not_a_bitstring", [p.kind for p in report.problems])

    def test_an_empty_table_is_not_zero_shots_of_information(self):
        report = check_counts({"ZZ": {}})
        self.assertIn("empty", [p.kind for p in report.problems])

    def test_no_settings_at_all(self):
        self.assertFalse(check_counts({}).ok)

    def test_a_deflated_table_reports_the_other_direction(self):
        report = check_counts({"ZZ": {"00": 25, "11": 25}}, declared_shots=200)
        problem = report.problems[0]
        self.assertIn("too wide", problem.consequence)
        self.assertIn("shots that were paid for", problem.consequence.lower())


class FoldsTest(unittest.TestCase):
    """Folds are checked structurally but not against the declared shot
    count, because running folds at a different shot count is legitimate
    and refusing it would reject honest data."""

    def test_folds_at_a_different_shot_count_are_accepted(self):
        counts = real_counts()
        half = {bits: n for bits, n in list(counts.items())[:3]}
        report = analyse({"XYYX": counts}, OBSERVABLE,
                         folds={1: {"XYYX": counts}, 3: {"XYYX": half}},
                         declared_shots=100)
        self.assertIsNotNone(report)

    def test_a_structurally_broken_fold_is_still_refused(self):
        with self.assertRaises(CountsError):
            analyse({"XYYX": real_counts()}, OBSERVABLE,
                    folds={1: {"XYYX": real_counts()},
                           3: {"XYYX": {"00000": -5}}},
                    declared_shots=100)


if __name__ == "__main__":
    unittest.main()
