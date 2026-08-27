"""The gates added from the H4 failure library, each tested against the
distinction it exists to draw.
"""
import unittest

from qem_auditor import ClaimType, Replicate, ReplicateKind, UncertaintyCoverage, gates
from qem_auditor.schema import TranspilationStatus

from .helpers import make_experiment


class UnitaryEquivalenceGateTest(unittest.TestCase):
    def test_verified_modified_circuit_fails(self):
        exp = make_experiment(unitary_equivalence=None)
        exp.circuit.transpilation_status = TranspilationStatus.VERIFIED_MODIFIED
        self.assertIs(gates.unitary_equivalence_gate(exp).passed, False)

    def test_unverified_is_not_a_pass(self):
        exp = make_experiment(unitary_equivalence=None)
        exp.circuit.transpilation_status = TranspilationStatus.UNVERIFIED
        self.assertIsNone(gates.unitary_equivalence_gate(exp).passed)


class ReplicateIndependenceGateTest(unittest.TestCase):
    """The distinction that would have caught the 0.115 headline."""

    def test_bootstrap_replicates_are_not_replication(self):
        exp = make_experiment()
        exp.outputs.replicates = [
            Replicate(0.10, ReplicateKind.BOOTSTRAP_RESAMPLE) for _ in range(8)
        ]
        r = gates.replicate_independence_gate(exp)
        self.assertIs(r.passed, False)
        self.assertIn("shot noise", r.reason)

    def test_seed_splits_are_not_replication_either(self):
        exp = make_experiment()
        exp.outputs.replicates = [Replicate(0.10, ReplicateKind.SEED_SPLIT) for _ in range(8)]
        self.assertIs(gates.replicate_independence_gate(exp).passed, False)

    def test_independent_submissions_pass(self):
        self.assertIs(gates.replicate_independence_gate(make_experiment()).passed, True)

    def test_reproducibility_ignores_non_independent_replicates(self):
        """Eight bootstrap replicates must not satisfy an 8-replicate target."""
        exp = make_experiment()
        exp.outputs.replicates = [
            Replicate(0.10, ReplicateKind.BOOTSTRAP_RESAMPLE) for _ in range(8)
        ]
        self.assertIsNone(gates.reproducibility_gate(exp).passed)


class TailRiskGateTest(unittest.TestCase):
    """The distinction that killed cross-fitting despite a better median."""

    def test_rare_catastrophic_outliers_fail(self):
        exp = make_experiment(n_trials=32, n_outlier_trials=2)
        r = gates.tail_risk_gate(exp)
        self.assertIs(r.passed, False)
        self.assertIn("6.2%", r.reason)

    def test_heavy_tail_fails_even_with_no_flagged_outliers(self):
        exp = make_experiment(q50_kcal=0.10, q99_kcal=50.0, n_trials=80, n_outlier_trials=0)
        self.assertIs(gates.tail_risk_gate(exp).passed, False)

    def test_clean_tails_pass(self):
        self.assertIs(gates.tail_risk_gate(make_experiment()).passed, True)


class EvidenceScopeGateTest(unittest.TestCase):
    def test_all_four_axes_pass(self):
        self.assertIs(gates.evidence_scope_gate(make_experiment()).passed, True)

    def test_a_single_missing_axis_fails_and_names_it(self):
        exp = make_experiment(uncertainty=UncertaintyCoverage(
            shot_noise=True, method_monte_carlo=True, noise_model=True))
        r = gates.evidence_scope_gate(exp)
        self.assertIs(r.passed, False)
        self.assertIn("cross_submission", r.reason)

    def test_axes_are_not_ranked(self):
        """Neither of these is a superset of the other, so neither may pass."""
        drift_only = make_experiment(uncertainty=UncertaintyCoverage(
            shot_noise=True, cross_submission=True))
        model_only = make_experiment(uncertainty=UncertaintyCoverage(
            shot_noise=True, noise_model=True))
        self.assertIs(gates.evidence_scope_gate(drift_only).passed, False)
        self.assertIs(gates.evidence_scope_gate(model_only).passed, False)


class ClaimTypeTest(unittest.TestCase):
    """A relative improvement must not be graded against an absolute bar."""

    def test_absolute_claim_uses_chemical_accuracy_only(self):
        exp = make_experiment()
        self.assertIsNotNone(gates.chemical_accuracy_gate(exp).passed)
        self.assertIsNone(gates.improvement_gate(exp).passed)

    def test_relative_claim_uses_improvement_only(self):
        exp = make_experiment(claim_type=ClaimType.RELATIVE_IMPROVEMENT,
                              baseline_error_kcal=1.00, mitigated_error_kcal=0.10)
        self.assertIsNone(gates.chemical_accuracy_gate(exp).passed)
        self.assertIs(gates.improvement_gate(exp).passed, True)

    def test_a_real_improvement_that_misses_the_absolute_bar_still_improves(self):
        """The joint Schmidt frame's shape: 4x better than baseline, still
        nowhere near chemical accuracy. Grading it absolutely would call a
        real result a failure."""
        exp = make_experiment(claim_type=ClaimType.RELATIVE_IMPROVEMENT,
                              baseline_error_kcal=1.1788, mitigated_error_kcal=0.2920)
        self.assertIs(gates.improvement_gate(exp).passed, True)
        self.assertGreater(exp.outputs.mitigated_error_kcal, 0.25)

    def test_failing_to_beat_the_baseline_fails(self):
        exp = make_experiment(claim_type=ClaimType.RELATIVE_IMPROVEMENT,
                              baseline_error_kcal=1.79, mitigated_error_kcal=2.31)
        self.assertIs(gates.improvement_gate(exp).passed, False)


class FreeParameterAndDeterminismTest(unittest.TestCase):
    def test_unfloored_free_parameter_fails(self):
        self.assertIs(gates.free_parameter_floor_gate(
            make_experiment(free_parameter_floor_test=False)).passed, False)

    def test_nondeterminism_fails(self):
        self.assertIs(gates.determinism_gate(
            make_experiment(determinism_check=False)).passed, False)

    def test_wrong_direction_heldout_fails(self):
        self.assertIs(gates.extrapolation_domain_gate(
            make_experiment(extrapolation_in_domain=False)).passed, False)


if __name__ == "__main__":
    unittest.main()
