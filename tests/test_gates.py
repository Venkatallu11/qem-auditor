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


class NumpyBooleanTest(unittest.TestCase):
    """A control set from a numpy comparison must not vanish.

    The gates separate "failed" from "never run" with `is False`, which
    is exact -- and numpy.bool_ is equal to False without being it. So
    `extrapolation_in_domain = error <= tolerance`, the most natural line
    anyone doing quantum work would write, used to store a value that
    read as "not recorded". A measured failure disappeared and the
    verdict softened from INVALID to NOT ESTABLISHED with nothing to show
    it had happened. Found by running a real audit, not by reading code.
    """

    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("numpy not installed")

    def test_a_numpy_false_is_stored_as_a_real_false(self):
        import numpy as np

        from qem_auditor import Controls

        controls = Controls()
        controls.extrapolation_in_domain = np.float64(5.1) <= np.float64(3.9)
        self.assertIs(controls.extrapolation_in_domain, False)

    def test_a_numpy_true_is_stored_as_a_real_true(self):
        import numpy as np

        from qem_auditor import Controls

        self.assertIs(Controls(ideal_control=np.True_).ideal_control, True)

    def test_the_constructor_normalises_too_not_only_assignment(self):
        import numpy as np

        from qem_auditor import Controls

        self.assertIs(Controls(determinism_check=np.False_).determinism_check, False)

    def test_a_numpy_failed_control_actually_fails_its_gate(self):
        """The end of the chain: the gate must see the failure."""
        import numpy as np

        from qem_auditor import gates

        from .helpers import make_experiment

        exp = make_experiment(ideal_control=np.False_)
        self.assertIs(gates.ideal_control_gate(exp).passed, False)

    def test_a_numpy_failure_still_reaches_the_verdict(self):
        import numpy as np

        from qem_auditor import Verdict, audit

        from .helpers import make_experiment

        self.assertIs(audit(make_experiment(target_leakage_check=np.False_)).verdict,
                      Verdict.INVALID)

    def test_an_ambiguous_value_is_refused_rather_than_guessed(self):
        """Silently reading 1, 0.0 or "no" as a verdict is the same
        mistake in a different coat."""
        from qem_auditor import Controls

        for value in (1, 0, 0.0, "no", "False", [], object()):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    Controls(ideal_control=value)

    def test_none_still_means_never_run(self):
        from qem_auditor import Controls

        self.assertIsNone(Controls(ideal_control=None).ideal_control)

    def test_a_numpy_integer_is_not_silently_a_boolean(self):
        import numpy as np

        from qem_auditor import Controls

        with self.assertRaises(TypeError):
            Controls(ideal_control=np.int64(1))
