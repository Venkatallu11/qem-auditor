"""Verdict composition: which combinations of gate outcomes produce which
verdict, and -- most importantly -- that a single failed hard gate cannot
be outvoted by any amount of otherwise-excellent evidence.
"""
import unittest

from qem_auditor import Verdict, audit

from .helpers import make_experiment


class HardGatePrecedenceTest(unittest.TestCase):
    """Every case here starts from a record that would otherwise CERTIFY."""

    def test_broken_ideal_control_forces_invalid(self):
        self.assertIs(audit(make_experiment(ideal_control=False)).verdict, Verdict.INVALID)

    def test_target_leakage_forces_invalid(self):
        self.assertIs(audit(make_experiment(target_leakage_check=False)).verdict, Verdict.INVALID)

    def test_failed_adversarial_controls_force_invalid(self):
        self.assertIs(audit(make_experiment(adversarial_check=False)).verdict, Verdict.INVALID)

    def test_perfect_replication_does_not_rescue_a_broken_hard_gate(self):
        exp = make_experiment(ideal_control=False,
                              replicate_errors_kcal=[0.001] * 32,
                              mitigated_error_kcal=0.001,
                              q50_kcal=0.001, q95_kcal=0.002, q99_kcal=0.003)
        self.assertIs(audit(exp).verdict, Verdict.INVALID)

    def test_chemical_accuracy_alone_never_forces_invalid(self):
        """Missing the accuracy bar refutes the claim as stated -- it never
        makes the METHOD invalid, which is a claim about brokenness."""
        exp = make_experiment(mitigated_error_kcal=5.0, q50_kcal=5.0,
                              q95_kcal=9.0, q99_kcal=12.0,
                              replicate_errors_kcal=[5.0] * 8)
        self.assertIs(audit(exp).verdict, Verdict.REFUTED)


class InsufficientEvidenceTest(unittest.TestCase):
    def test_unrun_hard_gate_is_not_established_not_certified(self):
        """Silence is not evidence. An otherwise-perfect record whose
        adversarial controls were never run must not certify -- this is the
        exact hole that would let an untested claim through."""
        for missing in ("ideal_control", "target_leakage_check", "adversarial_check"):
            with self.subTest(missing=missing):
                exp = make_experiment(**{missing: None})
                self.assertIs(audit(exp).verdict, Verdict.NOT_ESTABLISHED)

    def test_no_replication_data_is_not_established(self):
        exp = make_experiment(reproducibility_checked=False, replicate_errors_kcal=[])
        self.assertIs(audit(exp).verdict, Verdict.NOT_ESTABLISHED)

    def test_no_accuracy_numbers_is_not_established(self):
        exp = make_experiment(mitigated_error_kcal=None, q95_kcal=None,
                              reproducibility_checked=False, replicate_errors_kcal=[])
        self.assertIs(audit(exp).verdict, Verdict.NOT_ESTABLISHED)


class CertificationTest(unittest.TestCase):
    def test_complete_clean_record_certifies(self):
        self.assertIs(audit(make_experiment()).verdict, Verdict.CERTIFIED_UNDER_SCOPE)

    def test_incomplete_replication_blocks_certification(self):
        exp = make_experiment(replicate_errors_kcal=[0.10] * 4)
        self.assertIs(audit(exp).verdict, Verdict.PROMISING)

    def test_missing_real_hardware_validation_blocks_certification(self):
        exp = make_experiment(real_hardware_full_validation=False)
        self.assertIs(audit(exp).verdict, Verdict.PROMISING)

    def test_missing_chemical_accuracy_blocks_certification(self):
        exp = make_experiment(q95_kcal=1.50, q99_kcal=2.00)
        self.assertIs(audit(exp).verdict, Verdict.REFUTED)


class ReportShapeTest(unittest.TestCase):
    def test_report_carries_every_gate_result(self):
        report = audit(make_experiment())
        self.assertEqual(len(report.gate_results), len(__import__(
            "qem_auditor.gates", fromlist=["gates"]).ALL_GATES))
        self.assertEqual(report.experiment_id, "fixture")
        self.assertEqual(report.integrity_violations, [])


if __name__ == "__main__":
    unittest.main()
