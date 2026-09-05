"""The knowledge model, and the refutation it ships with.

A predictor nobody has validated is worse than no predictor: it answers
every question confidently and there is no way to tell from the inside
whether it learned anything or is repeating whatever the training set
contained most. That failure mode is this package's whole subject, so
these tests are mostly about the machinery that could embarrass it.
"""
import unittest

from qem_auditor.predict import (MINIMUM_CASES, NEAREST_ENOUGH, Predictor,
                                 TrainingCase, cross_validate)


def case(readout, gate, winner, shot=0.0):
    shares = {"READOUT": readout, "GATE_STOCHASTIC": gate}
    if shot:
        shares["SHOT_NOISE"] = shot
    return TrainingCase(budget_shares=shares, winner=winner)


def separable(n=8):
    """A corpus where the budget genuinely determines the winner."""
    cases = []
    for i in range(n):
        cases.append(case(0.9 - 0.01 * i, 0.1 + 0.01 * i, "REM (readout)"))
        cases.append(case(0.1 + 0.01 * i, 0.9 - 0.01 * i, "ZNE (fold 1,3,5)"))
    return cases


class RefusalTest(unittest.TestCase):

    def test_too_few_cases_means_no_prediction(self):
        predictor = Predictor().fit(separable(2))
        result = predictor.predict({"READOUT": 0.9, "GATE_STOCHASTIC": 0.1})
        self.assertFalse(result.confident)
        self.assertIn("not knowledge", result.reason)

    def test_an_out_of_distribution_budget_is_refused(self):
        """Guessing outside what the corpus measured is how a model earns
        trust it has not been given."""
        predictor = Predictor().fit(separable())
        result = predictor.predict({"DECOHERENCE": 1.0})
        self.assertFalse(result.confident)
        self.assertIn("outside what the corpus has measured", result.reason)
        self.assertLess(result.nearest_similarity, NEAREST_ENOUGH)

    def test_a_familiar_budget_gets_an_answer(self):
        predictor = Predictor().fit(separable())
        result = predictor.predict({"READOUT": 0.88, "GATE_STOCHASTIC": 0.12})
        self.assertTrue(result.confident)
        self.assertEqual(result.method, "REM (readout)")
        self.assertGreater(result.agreement, 0.5)


class ValidationTest(unittest.TestCase):

    def test_a_separable_corpus_beats_the_baseline(self):
        """Sanity: when the budget really does decide the winner, the
        machinery must be able to notice."""
        report = cross_validate(separable(), minimum_cases=4)
        self.assertTrue(report.beats_baseline, report.describe())
        self.assertGreater(report.skill, 0.2)

    def test_a_corpus_with_one_winner_shows_no_skill(self):
        """The important case. If every regime has the same winner, the
        features cannot carry information about it -- and a predictor
        that reports high accuracy here is measuring the baseline."""
        cases = [case(0.9 - 0.05 * i, 0.1 + 0.05 * i, "CDR (Clifford regression)")
                 for i in range(16)]
        report = cross_validate(cases, minimum_cases=4)
        self.assertAlmostEqual(report.accuracy, 1.0)
        self.assertAlmostEqual(report.baseline_accuracy, 1.0)
        self.assertFalse(report.beats_baseline)
        self.assertIn("no skill", report.describe())
        self.assertIn("statement about THIS corpus", report.describe())

    def test_the_baseline_is_the_majority_share_not_a_refitted_majority(self):
        """This test previously asserted the opposite, and the opposite
        was a trap.

        Refitting the majority per fold looks more even-handed. On a
        corpus split several ways it is not: removing one member of the
        leading class can hand the majority to a rival, so the "baseline"
        misses every case it would really have got right. On the sixteen
        measured cases in examples/knowledge_model.py -- split 7/6/3 --
        the refitted baseline scored 0 of 16 where always naming the most
        common winner scores 7. Publishing that would have shown the
        predictor beating a strawman of its own making.
        """
        cases = ([case(0.9, 0.1, "REM (readout)")] * 3
                 + [case(0.1, 0.9, "ZNE (fold 1,3,5)")] * 2)
        report = cross_validate(cases, minimum_cases=1)
        self.assertEqual(report.baseline_method, "REM (readout)")
        self.assertEqual(report.baseline_correct, 3)
        self.assertAlmostEqual(report.baseline_accuracy, 3 / 5)

    def test_a_tied_corpus_does_not_zero_the_baseline(self):
        """The exact pathology, pinned: an even split must not make the
        do-nothing strategy look like it never works."""
        cases = ([case(0.9, 0.1, "REM (readout)")] * 4
                 + [case(0.1, 0.9, "ZNE (fold 1,3,5)")] * 4)
        report = cross_validate(cases, minimum_cases=1)
        self.assertAlmostEqual(report.baseline_accuracy, 0.5)

    def test_a_refusal_counts_as_wrong(self):
        """Harsher than the predictor deserves, and the right reading of
        the question being asked: do the features help?"""
        cases = separable(3)
        report = cross_validate(cases, minimum_cases=MINIMUM_CASES)
        self.assertEqual(report.refused, len(cases))
        self.assertEqual(report.correct, 0)
        self.assertIn("counted as wrong", report.describe())

    def test_cross_validation_needs_something_to_hold_out(self):
        with self.assertRaises(ValueError):
            cross_validate([case(0.9, 0.1, "REM (readout)")])


class TrainingCaseTest(unittest.TestCase):

    def test_shares_that_are_not_shares_are_refused(self):
        with self.assertRaises(ValueError):
            TrainingCase(budget_shares={"READOUT": 0.9, "GATE_STOCHASTIC": 0.9},
                         winner="REM (readout)")

    def test_the_winner_is_whatever_was_measured(self):
        """Documented as a test because the tempting shortcut -- training
        on what the catalogue RECOMMENDED -- would teach a model to
        reproduce this package's own opinions rather than the physics."""
        entry = case(0.9, 0.1, "REM (readout)")
        self.assertEqual(entry.winner, "REM (readout)")
