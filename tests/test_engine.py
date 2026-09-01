"""The decision layer: from a table of results to a recommendation.

Everything needed to answer "which method should I use" existed here as
an EXAMPLE. A person with a circuit had to read the shootout and assemble
the judgement themselves, which is the work this project exists to
remove.
"""
import unittest

from qem_auditor.engine import MethodOutcome, recommend
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
