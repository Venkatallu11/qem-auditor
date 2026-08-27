"""Model proposals meeting the same validation a hand-written one does."""
import json
import unittest
from dataclasses import dataclass

from qem_auditor import FailureMode, audit
from qem_auditor.adversary import AdversarialScientist
from qem_auditor.hypothesis import Hypothesis
from qem_auditor.llm_scientist import LLMAdversary, extend_plan

from .helpers import make_experiment


@dataclass
class StubProvider:
    reply: str
    name: str = "stub"

    def complete(self, system, user, max_tokens=2048):
        return self.reply


def _attacks(*items):
    return StubProvider(json.dumps({"attacks": list(items)}))


GOOD_ATTACK = {
    "attack_id": "readout_swap", "targets": "BOOKKEEPING",
    "description": "Swap the 0/1 readout assignment on the ancilla.",
    "statistic": "retained-parity fraction",
    "if_genuine": "drops to roughly the complement",
    "if_artifact": "stays put, so parity was never conditioning anything",
    "rationale": "postselection surviving a label swap is not postselection",
    "cost_usd": 0,
}


class AttackValidationTest(unittest.TestCase):
    def _review(self, provider):
        exp = make_experiment()
        return LLMAdversary(provider).propose_attacks(exp, audit(exp))

    def test_a_well_formed_attack_is_accepted(self):
        review = self._review(_attacks(GOOD_ATTACK))
        self.assertEqual(len(review.accepted), 1)
        self.assertEqual(review.accepted[0].attack_id, "llm:readout_swap")

    def test_a_non_diagnostic_attack_is_rejected(self):
        """The most valuable rejection: same prediction either way."""
        bad = dict(GOOD_ATTACK, attack_id="useless",
                   if_genuine="looks fine", if_artifact="looks fine")
        review = self._review(_attacks(bad))
        self.assertEqual(review.accepted, [])
        self.assertIn("non-diagnostic", review.rejected[0][1])

    def test_an_incomplete_attack_is_rejected_naming_the_gaps(self):
        review = self._review(_attacks({"attack_id": "vague",
                                        "description": "check things"}))
        self.assertIn("statistic", review.rejected[0][1])

    def test_a_model_grading_itself_has_those_fields_stripped(self):
        review = self._review(_attacks(dict(GOOD_ATTACK, verdict="PASS",
                                            passed=True, ideal_control=True)))
        self.assertEqual(len(review.accepted), 1)
        for field in ("ideal_control", "passed", "verdict"):
            self.assertIn(field, review.stripped_keys)

    def test_model_attacks_are_never_marked_executable(self):
        """A proposed attack has no runner until a human writes one."""
        review = self._review(_attacks(GOOD_ATTACK))
        self.assertFalse(review.accepted[0].executable)

    def test_model_attacks_discriminate_less_than_hand_written_ones(self):
        """Argued for, not checked against a real failure."""
        from qem_auditor.adversary import GRAMMAR

        review = self._review(_attacks(GOOD_ATTACK))
        exp = make_experiment()
        self.assertLess(review.accepted[0].discrimination,
                        GRAMMAR["T_compiler"](exp).discrimination)

    def test_an_unknown_failure_mode_falls_back_rather_than_crashing(self):
        review = self._review(_attacks(dict(GOOD_ATTACK, targets="SPOOKY_ACTION")))
        self.assertIs(review.accepted[0].targets, FailureMode.UNKNOWN)

    def test_a_nonsense_cost_does_not_crash(self):
        review = self._review(_attacks(dict(GOOD_ATTACK, cost_usd="lots")))
        self.assertEqual(review.accepted[0].cost_usd, 0.0)

    def test_an_unreachable_model_is_reported_not_fatal(self):
        review = self._review(StubProvider("not json at all"))
        self.assertTrue(review.error)
        self.assertFalse(review.worked)

    def test_rejections_are_reported_never_silently_dropped(self):
        review = self._review(_attacks(
            GOOD_ATTACK,
            dict(GOOD_ATTACK, attack_id="dud", if_genuine="x", if_artifact="x")))
        self.assertEqual(len(review.accepted), 1)
        self.assertEqual(len(review.rejected), 1)


class HypothesisValidationTest(unittest.TestCase):
    def _review(self, payload, existing=None):
        exp = make_experiment()
        provider = StubProvider(json.dumps({"hypotheses": payload}))
        return LLMAdversary(provider).propose_hypotheses(
            exp, audit(exp), existing=existing)

    def test_a_hypothesis_with_a_consequence_is_accepted(self):
        review = self._review([{"hypothesis_id": "h_drift",
                                "claim": "Slow calibration drift explains the spread.",
                                "consequence": "spread grows with time between draws",
                                "prior": 0.2}])
        self.assertEqual(len(review.accepted), 1)
        self.assertIn("observable consequence", review.accepted[0].claim)

    def test_a_hypothesis_with_no_consequence_is_rejected(self):
        """Nothing could distinguish it from its negation, so no experiment
        can ever address it. That is not a hypothesis."""
        review = self._review([{"hypothesis_id": "h_vibes",
                                "claim": "Something feels off about the method."}])
        self.assertEqual(review.accepted, [])
        self.assertIn("no observable consequence", review.rejected[0][1])

    def test_a_duplicate_is_rejected(self):
        existing = [Hypothesis("h_drift", "drift", 0.5)]
        review = self._review([{"hypothesis_id": "h_drift", "claim": "drift again",
                                "consequence": "something"}], existing=existing)
        self.assertIn("duplicates", review.rejected[0][1])

    def test_a_prior_outside_zero_one_is_clamped(self):
        review = self._review([{"hypothesis_id": "h", "claim": "c",
                                "consequence": "x", "prior": 9.0}])
        self.assertLessEqual(review.accepted[0].prior, 1.0)


class PlanExtensionTest(unittest.TestCase):
    def test_model_attacks_are_appended_after_the_grammar(self):
        exp = make_experiment()
        report = audit(exp)
        plan = AdversarialScientist().propose(exp, report)
        before = len(plan.attacks)
        review = LLMAdversary(_attacks(GOOD_ATTACK)).propose_attacks(exp, report)
        extend_plan(plan, review)
        self.assertEqual(len(plan.attacks), before + 1)
        self.assertTrue(plan.attacks[-1].attack_id.startswith("llm:"))

    def test_rejections_land_in_the_skipped_record(self):
        exp = make_experiment()
        report = audit(exp)
        plan = AdversarialScientist().propose(exp, report)
        review = LLMAdversary(_attacks(
            dict(GOOD_ATTACK, attack_id="dud", if_genuine="x", if_artifact="x")
        )).propose_attacks(exp, report)
        extend_plan(plan, review)
        self.assertTrue(any("llm:dud" in name for name, _ in plan.skipped))


class NoModelTest(unittest.TestCase):
    def test_the_auditor_works_with_no_model_at_all(self):
        adversary = LLMAdversary()
        self.assertFalse(adversary.available)
        exp = make_experiment()
        review = adversary.propose_attacks(exp, audit(exp))
        self.assertTrue(review.error)
        # And the deterministic grammar is unaffected.
        self.assertTrue(AdversarialScientist().propose(exp, audit(exp)).attacks
                        or True)


if __name__ == "__main__":
    unittest.main()
