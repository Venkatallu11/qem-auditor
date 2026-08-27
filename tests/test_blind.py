"""Blind mode: is the answer actually hidden, and is the commitment
actually enforced?
"""
import unittest

from qem_auditor import Verdict, audit
from qem_auditor.blind import (
    BlindChallenge,
    BlindDecision,
    BlindError,
    auto_decide,
    redact,
)

from .helpers import make_experiment


class RedactionTest(unittest.TestCase):
    def test_every_outcome_quantity_is_removed(self):
        blind = redact(make_experiment())
        for field in ("raw_error_kcal", "mitigated_error_kcal", "q50_kcal",
                      "q95_kcal", "q99_kcal", "baseline_error_kcal"):
            self.assertIsNone(getattr(blind.outputs, field), field)

    def test_replicate_values_are_withheld_but_the_count_survives(self):
        exp = make_experiment()
        blind = redact(exp)
        self.assertEqual(len(blind.outputs.independent_replicates),
                         len(exp.outputs.independent_replicates))
        self.assertTrue(all(r.error_kcal is None for r in blind.outputs.replicates))

    def test_methodology_survives_redaction(self):
        """A blind auditor must still be able to judge HOW it was done."""
        blind = redact(make_experiment())
        self.assertIs(blind.controls.ideal_control, True)
        self.assertTrue(blind.outputs.uncertainty.is_complete)
        self.assertEqual(blind.outputs.n_replicates_target, 8)

    def test_notes_are_cleared(self):
        exp = make_experiment()
        exp.notes = "the exact energy is -2.1663 Ha"
        self.assertEqual(redact(exp).notes, "")

    def test_redaction_does_not_mutate_the_original(self):
        exp = make_experiment()
        redact(exp)
        self.assertIsNotNone(exp.outputs.mitigated_error_kcal)

    def test_a_redacted_record_is_not_malformed(self):
        """Withholding a value must not read as an inconsistent record --
        that would be a different finding entirely."""
        blind = redact(make_experiment())
        self.assertIsNot(audit(blind).verdict, Verdict.INVALID_RECORD)


class CommitmentTest(unittest.TestCase):
    def test_reveal_before_decide_is_refused(self):
        ch = BlindChallenge(make_experiment())
        with self.assertRaises(BlindError) as ctx:
            ch.reveal()
        self.assertIn("before decide", str(ctx.exception))

    def test_deciding_twice_is_refused(self):
        ch = BlindChallenge(make_experiment())
        ch.decide(BlindDecision(False, ["more draws"]))
        with self.assertRaises(BlindError):
            ch.decide(BlindDecision(True))

    def test_a_decision_cannot_certify_with_outstanding_evidence(self):
        with self.assertRaises(BlindError) as ctx:
            BlindDecision(would_certify=True, required_evidence=["more draws"])
        self.assertIn("not certification", str(ctx.exception))


class ScoringTest(unittest.TestCase):
    def test_correctly_withholding_scores_correct(self):
        exp = make_experiment(real_hardware_full_validation=False)
        self.assertIsNot(audit(exp).verdict, Verdict.CERTIFIED_UNDER_SCOPE)
        ch = BlindChallenge(exp)
        ch.decide(BlindDecision(False, ["hardware validation"]))
        self.assertTrue(ch.reveal().correct)

    def test_certifying_a_record_that_should_not_be_is_a_false_positive(self):
        exp = make_experiment(ideal_control=False)
        ch = BlindChallenge(exp)
        ch.decide(BlindDecision(True))
        result = ch.reveal()
        self.assertFalse(result.correct)
        self.assertIn("false positive", result.detail)

    def test_withholding_from_a_fully_certified_record_is_scored_wrong(self):
        exp = make_experiment()
        self.assertIs(audit(exp).verdict, Verdict.CERTIFIED_UNDER_SCOPE)
        ch = BlindChallenge(exp)
        ch.decide(BlindDecision(False, ["something"]))
        result = ch.reveal()
        self.assertFalse(result.correct)
        self.assertIn("met every bar", result.detail)


class AutoDecideTest(unittest.TestCase):
    def test_blind_certification_is_impossible_by_construction(self):
        """Gates needing the numbers go N/A, and an N/A is not a pass. So
        what is really scored is whether the RIGHT evidence is named."""
        ch = BlindChallenge(make_experiment())
        decision = auto_decide(ch.blinded)
        self.assertFalse(decision.would_certify)

    def test_it_names_the_missing_evidence(self):
        decision = auto_decide(redact(make_experiment()))
        self.assertTrue(decision.required_evidence)
        self.assertTrue(any("reproducibility" in r for r in decision.required_evidence))

    def test_the_flagship_challenge_is_answered_correctly(self):
        from benchmarks.h4_ancilla_qed import EXPERIMENT

        ch = BlindChallenge(EXPERIMENT, "flagship")
        ch.decide(auto_decide(ch.blinded))
        result = ch.reveal()
        self.assertTrue(result.correct)
        self.assertIs(result.true_verdict, Verdict.PROMISING)

    def test_every_benchmark_is_answered_correctly_blind(self):
        """None of the six should be certified, and the auditor must reach
        that without seeing any of their numbers."""
        import run_benchmarks

        for module in run_benchmarks.BENCHMARKS:
            with self.subTest(case=module.EXPERIMENT.experiment_id):
                ch = BlindChallenge(module.EXPERIMENT)
                ch.decide(auto_decide(ch.blinded))
                self.assertTrue(ch.reveal().correct)


if __name__ == "__main__":
    unittest.main()
