"""Each gate, exercised on all three of its outcomes: pass, fail, and the
'not enough data to judge' None case. The None case matters as much as
the other two -- conflating "not run" with "passed" is the single most
dangerous bug this project could have.
"""
import unittest

from qem_auditor import gates

from .helpers import make_experiment


class IdealControlGateTest(unittest.TestCase):
    def test_passes_when_control_recovered(self):
        r = gates.ideal_control_gate(make_experiment(ideal_control=True))
        self.assertIs(r.passed, True)

    def test_fails_when_control_broke(self):
        r = gates.ideal_control_gate(make_experiment(ideal_control=False))
        self.assertIs(r.passed, False)

    def test_not_run_is_none_not_pass(self):
        r = gates.ideal_control_gate(make_experiment(ideal_control=None))
        self.assertIsNone(r.passed)


class TargetLeakageGateTest(unittest.TestCase):
    def test_passes_when_no_leakage(self):
        self.assertIs(gates.target_leakage_gate(make_experiment(target_leakage_check=True)).passed, True)

    def test_fails_when_leakage_found(self):
        self.assertIs(gates.target_leakage_gate(make_experiment(target_leakage_check=False)).passed, False)

    def test_not_run_is_none_not_pass(self):
        self.assertIsNone(gates.target_leakage_gate(make_experiment(target_leakage_check=None)).passed)


class AdversarialGateTest(unittest.TestCase):
    def test_passes_when_negative_controls_failed_loudly(self):
        self.assertIs(gates.adversarial_gate(make_experiment(adversarial_check=True)).passed, True)

    def test_fails_when_garbage_input_still_looked_good(self):
        self.assertIs(gates.adversarial_gate(make_experiment(adversarial_check=False)).passed, False)

    def test_not_run_is_none_not_pass(self):
        self.assertIsNone(gates.adversarial_gate(make_experiment(adversarial_check=None)).passed)


class ReproducibilityGateTest(unittest.TestCase):
    def test_consistent_replicates_pass(self):
        r = gates.reproducibility_gate(make_experiment(replicate_errors_kcal=[0.10, 0.11, 0.09, 0.10]))
        self.assertIs(r.passed, True)

    def test_scattered_replicates_fail(self):
        r = gates.reproducibility_gate(
            make_experiment(replicate_errors_kcal=[0.01, 0.10, 5.00, 0.02])
        )
        self.assertIs(r.passed, False)
        self.assertIn("disagree", r.reason)

    def test_single_replicate_cannot_be_judged(self):
        r = gates.reproducibility_gate(make_experiment(replicate_errors_kcal=[0.10]))
        self.assertIsNone(r.passed)

    def test_unchecked_flag_cannot_be_judged(self):
        r = gates.reproducibility_gate(make_experiment(reproducibility_checked=False))
        self.assertIsNone(r.passed)

    def test_passing_below_target_says_so(self):
        r = gates.reproducibility_gate(make_experiment(replicate_errors_kcal=[0.10] * 4))
        self.assertIs(r.passed, True)
        self.assertIn("8-replicate target", r.reason)


class ChemicalAccuracyGateTest(unittest.TestCase):
    def test_prefers_q95_over_point_estimate(self):
        # Point estimate would clear the bar; the honest Q95 envelope does not.
        r = gates.chemical_accuracy_gate(make_experiment(mitigated_error_kcal=0.10, q95_kcal=0.90))
        self.assertIs(r.passed, False)
        self.assertIn("Q95", r.reason)

    def test_falls_back_to_point_estimate_without_q95(self):
        r = gates.chemical_accuracy_gate(make_experiment(q95_kcal=None, mitigated_error_kcal=0.10))
        self.assertIs(r.passed, True)
        self.assertIn("point estimate", r.reason)

    def test_no_numbers_at_all_cannot_be_judged(self):
        r = gates.chemical_accuracy_gate(make_experiment(q95_kcal=None, mitigated_error_kcal=None,
                                                         replicate_errors_kcal=[]))
        self.assertIsNone(r.passed)


class GateContractTest(unittest.TestCase):
    def test_every_gate_returns_a_named_result_with_a_reason(self):
        exp = make_experiment()
        for gate in gates.ALL_GATES:
            with self.subTest(gate=gate.__name__):
                r = gate(exp)
                self.assertTrue(r.name, "gate result must be named")
                self.assertTrue(r.reason.strip(), "gate result must explain itself")
                self.assertIn(r.passed, (True, False, None))


if __name__ == "__main__":
    unittest.main()
