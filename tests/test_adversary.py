"""The adversarial generator: does it propose attacks that can actually
discriminate, and does it refuse ones that cannot?
"""
import unittest

from qem_auditor import FailureMode, Provenance, audit
from qem_auditor.adversary import (
    GRAMMAR,
    AdversarialScientist,
    Attack,
    NonDiagnosticAttack,
    Prediction,
    compose,
)

from .helpers import make_experiment


class PredictionTest(unittest.TestCase):
    """The structural rule: an attack both hypotheses predict identically
    is not an attack."""

    def test_identical_predictions_are_refused(self):
        with self.assertRaises(NonDiagnosticAttack) as ctx:
            Prediction("the error", "goes down", "goes down")
        self.assertIn("cannot discriminate", str(ctx.exception))

    def test_whitespace_does_not_disguise_a_non_prediction(self):
        with self.assertRaises(NonDiagnosticAttack):
            Prediction("the error", "goes down", "  goes down  ")

    def test_a_prediction_must_name_what_it_measures(self):
        with self.assertRaises(NonDiagnosticAttack):
            Prediction("   ", "a", "b")

    def test_a_diagnostic_prediction_is_accepted(self):
        p = Prediction("chi2/dof", "much worse when shuffled", "comparable when shuffled")
        self.assertTrue(p.statistic)


class GrammarTest(unittest.TestCase):
    def test_every_transformation_produces_a_diagnostic_attack(self):
        exp = make_experiment()
        for name, transformation in GRAMMAR.items():
            with self.subTest(transformation=name):
                attack = transformation(exp)
                self.assertIsInstance(attack, Attack)
                self.assertNotEqual(attack.prediction.if_genuine,
                                    attack.prediction.if_artifact)

    def test_every_attack_explains_why_it_exists(self):
        """Each traces to a real failure, not a generic best practice."""
        exp = make_experiment()
        for name, transformation in GRAMMAR.items():
            with self.subTest(transformation=name):
                self.assertGreater(len(transformation(exp).rationale), 80)

    def test_every_attack_targets_a_known_failure_mode(self):
        exp = make_experiment()
        for transformation in GRAMMAR.values():
            self.assertIsInstance(transformation(exp).targets, FailureMode)

    def test_exact_attacks_discriminate_more_sharply_than_fuzzy_ones(self):
        """Counting gates cannot be wrong; judging a shuffled refit can."""
        exp = make_experiment()
        self.assertGreater(GRAMMAR["T_compiler"](exp).discrimination,
                           GRAMMAR["T_label"](exp).discrimination)


class CompositionTest(unittest.TestCase):
    def test_composition_combines_both_transformations(self):
        c = compose(make_experiment(), "T_calibration", "T_compiler")
        self.assertEqual(c.attack_id, "T_calibration+T_compiler")
        self.assertIn("T_calibration", c.transformation)
        self.assertIn("T_compiler", c.transformation)

    def test_artifact_branch_is_disjunctive(self):
        """Any part firing falsifies the whole."""
        c = compose(make_experiment(), "T_calibration", "T_compiler")
        self.assertTrue(c.prediction.if_artifact.startswith("any of:"))
        self.assertTrue(c.prediction.if_genuine.startswith("all of:"))

    def test_composed_discrimination_takes_the_sharpest_part(self):
        exp = make_experiment()
        c = compose(exp, "T_label", "T_compiler")
        self.assertEqual(c.discrimination, GRAMMAR["T_compiler"](exp).discrimination)

    def test_composition_needs_two(self):
        with self.assertRaises(ValueError):
            compose(make_experiment(), "T_label")

    def test_unknown_transformation_is_rejected_by_name(self):
        with self.assertRaises(KeyError) as ctx:
            compose(make_experiment(), "T_label", "T_nonsense")
        self.assertIn("T_nonsense", str(ctx.exception))


class ProposerTest(unittest.TestCase):
    def test_an_unaudited_claim_draws_the_whole_grammar(self):
        from qem_auditor.schema import TranspilationStatus

        exp = make_experiment(ideal_control=None, target_leakage_check=None,
                              adversarial_check=None, determinism_check=None,
                              unitary_equivalence=None, free_parameter_floor_test=None,
                              extrapolation_in_domain=None)
        exp.controls.provenance.clear()
        exp.circuit.transpilation_status = TranspilationStatus.UNVERIFIED
        plan = AdversarialScientist().propose(exp, audit(exp))
        proposed = {a.transformation for a in plan.attacks}
        for name in GRAMMAR:
            self.assertIn(name, proposed)

    def test_a_control_the_auditor_measured_is_skipped(self):
        exp = make_experiment()
        exp.controls.record_measured("unitary_equivalence", True)
        plan = AdversarialScientist().propose(exp, audit(exp))
        self.assertIn("T_compiler", [n for n, _ in plan.skipped])

    def test_a_SELF_REPORTED_pass_is_still_attacked(self):
        """The whole point of an adversary: the claimant's word is not
        evidence, so a self-reported pass does not close the question."""
        exp = make_experiment()
        exp.controls.provenance.clear()  # everything self-reported
        plan = AdversarialScientist().propose(exp, audit(exp))
        self.assertIn("T_compiler", {a.transformation for a in plan.attacks})

    def test_skips_are_reported_not_silent(self):
        exp = make_experiment()
        exp.controls.record_measured("determinism_check", True)
        plan = AdversarialScientist().propose(exp, audit(exp))
        self.assertTrue(plan.skipped)
        for name, why in plan.skipped:
            self.assertTrue(why.strip())

    def test_executable_and_free_attacks_are_ranked_first(self):
        exp = make_experiment(unitary_equivalence=None)
        plan = AdversarialScientist().propose(exp, audit(exp))
        self.assertTrue(plan.attacks[0].executable)

    def test_the_proposer_never_issues_a_verdict(self):
        """It has no API for one. This test exists to keep it that way."""
        scientist = AdversarialScientist()
        for forbidden in ("verdict", "passed", "certify", "judge", "grade"):
            self.assertFalse(
                any(forbidden in attr.lower() for attr in dir(scientist)),
                f"the adversarial proposer must not expose {forbidden!r}")


if __name__ == "__main__":
    unittest.main()
