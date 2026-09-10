"""The decision layer: from a table of results to a recommendation.

Everything needed to answer "which method should I use" existed here as
an EXAMPLE. A person with a circuit had to read the shootout and assemble
the judgement themselves, which is the work this project exists to
remove.
"""
import unittest

from qem_auditor.engine import MethodOutcome, consensus, recommend
from qem_auditor.prescribe import feasibility

EAGLE = {"ecr_error": 0.00311, "readout_error": 0.0293}


def outcome(name, errors, cost=1.0, sensitivity=1.0):
    return MethodOutcome(name=name, errors=tuple(errors), cost=cost,
                         sensitivity=sensitivity)


class DisqualificationTest(unittest.TestCase):

    def test_a_method_that_does_not_read_its_data_is_disqualified(self):
        """And it is disqualified while WINNING on accuracy, which is the
        whole point: in this project's own suite the fraud is not
        distinguishable from the best real method on error alone."""
        advice = recommend([
            outcome("fraud", [0.10, 0.11, 0.09, 0.10], cost=1, sensitivity=0.020),
            outcome("real", [0.40, 0.42, 0.38, 0.41], cost=5, sensitivity=1.05),
        ])
        self.assertEqual(advice.recommended, "real")
        self.assertIn("fraud", advice.disqualified)

    def test_a_method_never_attacked_is_not_assumed_honest(self):
        """An unrun control is not a pass -- the same rule every other
        gate here applies."""
        advice = recommend([outcome("unchecked", [0.1, 0.1, 0.1, 0.1],
                                    sensitivity=None)])
        self.assertIsNone(advice.recommended)
        self.assertIn("unchecked", advice.unverified)

    def test_that_rule_can_be_relaxed_deliberately(self):
        advice = recommend([outcome("unchecked", [0.1, 0.2, 0.1, 0.2],
                                    sensitivity=None)],
                           require_sensitivity=False)
        self.assertEqual(advice.recommended, "unchecked")


class TiesAreBrokenByCostTest(unittest.TestCase):
    """When the runs cannot separate two methods, ordering them by error
    orders noise. Cost is known, so cost decides -- which turns a coin
    flip dressed as a measurement into a defensible answer."""

    def test_the_cheaper_of_two_tied_methods_wins(self):
        advice = recommend([
            outcome("expensive", [1.10, 2.4, 0.2, 0.9], cost=50, sensitivity=0.9),
            outcome("cheap", [1.29, 2.1, 0.4, 1.2], cost=5, sensitivity=1.1),
        ])
        self.assertEqual(advice.recommended, "cheap")
        self.assertIn("expensive", advice.tied_with)
        self.assertIn("cheapest of a group", advice.reason)

    def test_a_clearly_better_method_wins_on_accuracy_not_cost(self):
        advice = recommend([
            outcome("accurate", [0.10, 0.11, 0.09, 0.10], cost=50, sensitivity=0.9),
            outcome("cheap but worse", [9.0, 9.1, 8.9, 9.0], cost=1, sensitivity=1.1),
        ])
        self.assertEqual(advice.recommended, "accurate")
        self.assertEqual(advice.tied_with, ())

    def test_the_report_says_cost_decided_rather_than_accuracy(self):
        advice = recommend([
            outcome("a", [1.10, 2.4, 0.2, 0.9], cost=50, sensitivity=0.9),
            outcome("b", [1.29, 2.1, 0.4, 1.2], cost=5, sensitivity=1.1),
        ])
        self.assertIn("cost decided, not accuracy", advice.format_report())


class FeasibilityRefusalTest(unittest.TestCase):

    def test_no_surviving_signal_means_no_recommendation(self):
        """A ranked table for an experiment that cannot run is a precise
        answer to a question nobody can ask."""
        advice = recommend(
            [outcome("anything", [1.0, 1.1, 0.9, 1.0], sensitivity=1.0)],
            feasibility=feasibility(5898, EAGLE, n_qubits=18))
        self.assertIsNone(advice.recommended)
        self.assertIn("no signal survives", advice.reason)
        self.assertIn("compilation problem", advice.reason)

    def test_a_runnable_circuit_still_gets_a_recommendation(self):
        advice = recommend(
            [outcome("something", [1.0, 1.1, 0.9, 1.0], sensitivity=1.0)],
            feasibility=feasibility(465, EAGLE, n_qubits=18))
        self.assertEqual(advice.recommended, "something")

    def test_no_outcomes_at_all_is_refused(self):
        with self.assertRaises(ValueError):
            recommend([])


class ConsensusTest(unittest.TestCase):
    """"Run everything and give me the answer" -- and the honest form of
    that is not one method's number."""

    def surviving(self):
        return [
            outcome("REM + ZNE", [1.15, 1.30, 0.95, 1.20], sensitivity=0.62),
            outcome("CDR", [1.29, 1.35, 1.20, 1.31], sensitivity=1.08),
            outcome("ZNE (exponential)", [1.90, 2.05, 1.80, 1.95], sensitivity=0.88),
            outcome("REM (iterative)", [1.22, 1.28, 1.18, 1.25], sensitivity=1.10),
        ]

    def test_a_fraud_is_excluded_from_the_average(self):
        """Including a fraud's number in a median lets it move the answer
        it was caught not reading."""
        result = consensus(self.surviving()
                           + [outcome("fraud", [0.1] * 4, sensitivity=0.02)])
        self.assertNotIn("fraud", result.methods)
        self.assertIn("fraud", result.excluded)
        self.assertGreater(result.estimate, 1.0)

    def test_the_bar_carries_both_terms(self):
        result = consensus(self.surviving(), statistical=0.05)
        self.assertAlmostEqual(result.statistical, 0.05)
        self.assertGreater(result.systematic, 0.0)
        self.assertAlmostEqual(
            result.uncertainty,
            (result.statistical ** 2 + result.systematic ** 2) ** 0.5)

    def test_disagreement_between_methods_is_reported_as_such(self):
        """The thing a single-method pipeline cannot compute: when the bar
        is dominated by disagreement, more shots will not narrow it."""
        result = consensus(self.surviving(), statistical=0.05)
        self.assertTrue(result.dominated_by_disagreement)
        report = result.format_report()
        self.assertIn("DISAGREEMENT", report)
        self.assertIn("More shots will not narrow it", report)

    def test_close_agreement_is_reported_as_statistical(self):
        agreeing = [outcome(f"m{i}", [1.20 + 0.001 * i] * 4, sensitivity=1.0)
                    for i in range(4)]
        result = consensus(agreeing, statistical=0.5)
        self.assertFalse(result.dominated_by_disagreement)
        self.assertIn("more shots would narrow it", result.format_report())

    def test_one_surviving_method_is_not_agreement(self):
        result = consensus([outcome("only", [1.2, 1.3, 1.1, 1.25],
                                    sensitivity=1.0)])
        self.assertEqual(result.systematic, 0.0)
        self.assertIn("not the same as agreement", result.reason)

    def test_no_survivor_means_no_answer(self):
        result = consensus([outcome("fraud", [0.1] * 4, sensitivity=0.02)])
        self.assertFalse(result.is_an_answer)
        self.assertIn("no relationship to the experiment", result.reason)

    def test_an_empty_set_is_refused(self):
        with self.assertRaises(ValueError):
            consensus([])
