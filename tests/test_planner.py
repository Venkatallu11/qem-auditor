"""The planner: does it choose experiments that discriminate, and does it
refuse to spend when spending buys nothing?
"""
import unittest

from qem_auditor import audit
from qem_auditor.hypothesis import Hypothesis, HypothesisLedger, Observation
from qem_auditor.planner import (
    CandidateExperiment,
    Outcome,
    Recommendation,
    candidates_from_audit,
    expected_information_gain,
    next_experiment,
    plan,
)

from .helpers import make_experiment


def _ledger():
    return HypothesisLedger([Hypothesis(f"H{i}", f"hypothesis {i}", 1.0) for i in range(2)])


# Splits the two hypotheses cleanly: each outcome is near-certain under one.
DISCRIMINATING = CandidateExperiment(
    "decisive", "an experiment whose outcome depends on which hypothesis holds",
    outcomes=[Outcome("a", {"H0": 0.98, "H1": 0.02}),
              Outcome("b", {"H0": 0.02, "H1": 0.98})])

# Same outcome whatever is true: expensive, impressive, and uninformative.
USELESS = CandidateExperiment(
    "more_shots", "collect more shots, which every hypothesis predicts identically",
    cost_usd=500.0,
    outcomes=[Outcome("a", {"H0": 0.5, "H1": 0.5}),
              Outcome("b", {"H0": 0.5, "H1": 0.5})])


class InformationGainTest(unittest.TestCase):
    def test_a_discriminating_experiment_gains_almost_a_full_bit(self):
        gain = expected_information_gain(_ledger(), DISCRIMINATING)
        self.assertGreater(gain, 0.8)

    def test_an_experiment_every_hypothesis_predicts_alike_gains_nothing(self):
        self.assertAlmostEqual(expected_information_gain(_ledger(), USELESS), 0.0, places=6)

    def test_no_outcomes_means_no_gain(self):
        self.assertEqual(expected_information_gain(
            _ledger(), CandidateExperiment("x", "unspecified")), 0.0)

    def test_gain_is_never_negative(self):
        for cand in (DISCRIMINATING, USELESS):
            self.assertGreaterEqual(expected_information_gain(_ledger(), cand), 0.0)


class PlanTest(unittest.TestCase):
    def test_prefers_the_informative_experiment_over_the_expensive_one(self):
        rec, proposals, _ = plan(_ledger(), [USELESS, DISCRIMINATING])
        self.assertIs(rec, Recommendation.RUN)
        self.assertEqual(proposals[0].candidate.candidate_id, "decisive")

    def test_stops_when_nothing_would_discriminate(self):
        rec, _, reason = plan(_ledger(), [USELESS])
        self.assertIs(rec, Recommendation.STOP)
        self.assertIn("discrimination", reason)

    def test_stops_when_the_question_is_already_resolved(self):
        led = _ledger()
        led.update(Observation("e1", {"H1": 0.001}))
        rec, _, reason = plan(led, [DISCRIMINATING])
        self.assertIs(rec, Recommendation.STOP)
        self.assertIn("belief", reason)

    def test_budget_excludes_unaffordable_candidates_and_says_so(self):
        pricey = CandidateExperiment(
            "hardware_sweep", "273 circuits on real hardware", cost_usd=6825.0,
            outcomes=DISCRIMINATING.outcomes)
        rec, _, reason = plan(_ledger(), [pricey], budget_usd=170.0)
        self.assertIs(rec, Recommendation.STOP)
        self.assertIn("6,825", reason)
        self.assertIn("170", reason)

    def test_value_of_information_is_bits_per_dollar(self):
        costly = CandidateExperiment("c", "d", cost_usd=4.0, outcomes=DISCRIMINATING.outcomes)
        _, proposals, _ = plan(_ledger(), [costly])
        p = proposals[0]
        self.assertAlmostEqual(p.value_of_information, p.information_gain_bits / 4.0)

    def test_free_experiments_are_ranked_on_gain_not_infinite_value(self):
        _, proposals, _ = plan(_ledger(), [DISCRIMINATING])
        p = proposals[0]
        self.assertEqual(p.value_of_information, p.information_gain_bits)
        self.assertNotEqual(p.value_of_information, float("inf"))


class CandidatesFromAuditTest(unittest.TestCase):
    def test_a_clean_record_has_no_gaps_to_close(self):
        exp = make_experiment()
        self.assertEqual(candidates_from_audit(exp, audit(exp)), [])
        self.assertIsNone(next_experiment(exp, audit(exp)))

    def test_an_unrun_control_becomes_a_concrete_experiment(self):
        exp = make_experiment(adversarial_check=None)
        ids = [c.candidate_id for c in candidates_from_audit(exp, audit(exp))]
        self.assertIn("close_adversarial", ids)

    def test_a_failed_gate_is_a_gap_too(self):
        exp = make_experiment(determinism_check=False)
        ids = [c.candidate_id for c in candidates_from_audit(exp, audit(exp))]
        self.assertIn("close_determinism", ids)

    def test_every_candidate_carries_a_procedure_and_a_cost(self):
        exp = make_experiment(ideal_control=None, adversarial_check=None)
        for c in candidates_from_audit(exp, audit(exp)):
            with self.subTest(candidate=c.candidate_id):
                self.assertTrue(c.description.strip())
                self.assertTrue(c.resolves.strip())
                self.assertGreaterEqual(c.cost_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
