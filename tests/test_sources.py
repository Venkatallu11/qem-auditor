"""Expectation sources: is the noise actually applied, and does the ideal
control stay noiseless whatever the adapter is configured with?
"""
import unittest

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from qem_auditor.adapters.sources import AerNoiseSource, StatevectorSource

    HAVE_AER = True
except ImportError:  # pragma: no cover
    HAVE_AER = False

from qem_auditor.adapters.base import MeasurementError


def _circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.3, 0)
    return qc


def _noise(p1=0.02, p2=0.05):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


@unittest.skipUnless(HAVE_AER, "qiskit-aer not installed")
class StatevectorSourceTest(unittest.TestCase):
    def test_it_is_noiseless(self):
        self.assertTrue(StatevectorSource().is_noiseless)

    def test_exact_matches_the_known_value(self):
        value = StatevectorSource().exact(_circuit(), SparsePauliOp("ZZ"))
        self.assertAlmostEqual(value, 0.955336, places=5)

    def test_sampling_is_close_but_not_identical(self):
        source = StatevectorSource()
        exact = source.exact(_circuit(), SparsePauliOp("ZZ"))
        sampled = source.sampled(_circuit(), SparsePauliOp("ZZ"), 20_000, 1)
        self.assertNotEqual(sampled, exact)
        self.assertLess(abs(sampled - exact), 0.05)

    def test_its_noiseless_twin_is_itself(self):
        source = StatevectorSource()
        self.assertIs(source.noiseless_twin(), source)


@unittest.skipUnless(HAVE_AER, "qiskit-aer not installed")
class AerNoiseSourceTest(unittest.TestCase):
    def test_noise_moves_the_expectation(self):
        clean = StatevectorSource().exact(_circuit(), SparsePauliOp("ZZ"))
        noisy = AerNoiseSource(_noise()).exact(_circuit(), SparsePauliOp("ZZ"))
        self.assertNotAlmostEqual(clean, noisy, places=3)
        self.assertLess(noisy, clean)  # depolarizing shrinks it

    def test_more_noise_moves_it_further(self):
        light = AerNoiseSource(_noise(0.005, 0.01)).exact(_circuit(), SparsePauliOp("ZZ"))
        heavy = AerNoiseSource(_noise(0.05, 0.15)).exact(_circuit(), SparsePauliOp("ZZ"))
        self.assertLess(heavy, light)

    def test_a_noise_model_that_never_fires_is_refused(self):
        """The trap: Aer happily returns a pure state, and those numbers
        would be reported as noisy. Failing is better."""
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 1), ["rz"])
        with self.assertRaises(MeasurementError) as ctx:
            AerNoiseSource(nm).exact(_circuit(), SparsePauliOp("ZZ"))
        self.assertIn("no effect", str(ctx.exception))
        self.assertIn("basis_gates", str(ctx.exception))

    def test_it_is_not_noiseless(self):
        self.assertFalse(AerNoiseSource(_noise()).is_noiseless)

    def test_its_noiseless_twin_is(self):
        twin = AerNoiseSource(_noise()).noiseless_twin()
        self.assertTrue(twin.is_noiseless)


@unittest.skipUnless(HAVE_AER, "qiskit-aer not installed")
class IdealControlInvariantTest(unittest.TestCase):
    """The invariant: the ideal control runs noiseless whatever the adapter
    was configured with. Running it under noise would quietly turn it into
    a different, much weaker check."""

    def test_ideal_control_stays_noiseless_on_a_noisy_adapter(self):
        from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

        adapter = QiskitAdapter(seed=3, source=AerNoiseSource(_noise()))
        self.assertTrue(adapter.is_noisy)
        measurement = adapter.measure_ideal_control(
            _circuit(), SparsePauliOp("ZZ"),
            lambda e: e(_circuit(), SparsePauliOp("ZZ")), shots=20_000)
        self.assertIn("noiseless", measurement.evidence["source"])

    def test_the_default_adapter_is_noiseless(self):
        from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

        self.assertFalse(QiskitAdapter().is_noisy)


@unittest.skipUnless(HAVE_AER, "qiskit-aer not installed")
class MitigationBenefitTest(unittest.TestCase):
    """The check the ideal control cannot make."""

    def _adapter(self):
        from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

        return QiskitAdapter(seed=3, source=AerNoiseSource(_noise(0.01, 0.04)))

    @staticmethod
    def _folded(base, n):
        qc = base.copy()
        for _ in range(n):
            qc.cx(0, 1)
            qc.cx(0, 1)
        return qc

    def test_a_real_mitigation_passes(self):
        obs = SparsePauliOp("ZZ")

        def zne(expectation):
            v = [expectation(self._folded(_circuit(), f), obs) for f in (0, 1, 2)]
            return 3 * v[0] - 3 * v[1] + v[2]

        m = self._adapter().measure_mitigation_benefit(
            _circuit(), obs, zne, shots=200_000, trials=8)
        self.assertIs(m.passed, True)
        self.assertGreater(m.evidence["median_improvement"], 1.1)

    def test_a_no_op_fails_even_though_it_passes_the_ideal_control(self):
        """The whole reason this control exists. A do-nothing mitigator
        cannot amplify noise it never touches, so the ideal control clears
        it trivially."""
        obs = SparsePauliOp("ZZ")
        adapter = self._adapter()

        def noop(expectation):
            return expectation(_circuit(), obs)

        ideal = adapter.measure_ideal_control(_circuit(), obs, noop, shots=200_000)
        self.assertIs(ideal.passed, True)

        benefit = adapter.measure_mitigation_benefit(
            _circuit(), obs, noop, shots=200_000, trials=8)
        self.assertIs(benefit.passed, False)
        self.assertIn("did not clearly help", benefit.detail)

    def test_a_single_trial_is_refused(self):
        """One draw cannot separate a real improvement from shot noise."""
        with self.assertRaises(MeasurementError):
            self._adapter().measure_mitigation_benefit(
                _circuit(), SparsePauliOp("ZZ"),
                lambda e: e(_circuit(), SparsePauliOp("ZZ")), trials=1)

    def test_a_noiseless_adapter_refuses_and_says_why(self):
        from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

        with self.assertRaises(MeasurementError) as ctx:
            QiskitAdapter().measure_mitigation_benefit(
                _circuit(), SparsePauliOp("ZZ"),
                lambda e: e(_circuit(), SparsePauliOp("ZZ")))
        self.assertIn("noisy source", str(ctx.exception))


class GateTest(unittest.TestCase):
    """No qiskit needed: the gate is pure record logic."""

    def test_unrun_is_not_a_pass(self):
        from qem_auditor import gates

        from .helpers import make_experiment

        exp = make_experiment(mitigation_benefit=None)
        self.assertIsNone(gates.mitigation_benefit_gate(exp).passed)

    def test_failing_it_does_not_make_the_record_invalid(self):
        """Not shown to work is not the same as shown to be broken."""
        from qem_auditor import Verdict, audit

        from .helpers import make_experiment

        exp = make_experiment(mitigation_benefit=False)
        self.assertIsNot(audit(exp).verdict, Verdict.INVALID)

    def test_but_it_blocks_certification(self):
        from qem_auditor import Verdict, audit

        from .helpers import make_experiment

        self.assertIs(audit(make_experiment()).verdict, Verdict.CERTIFIED_UNDER_SCOPE)
        self.assertIsNot(audit(make_experiment(mitigation_benefit=False)).verdict,
                         Verdict.CERTIFIED_UNDER_SCOPE)
        self.assertIsNot(audit(make_experiment(mitigation_benefit=None)).verdict,
                         Verdict.CERTIFIED_UNDER_SCOPE)


if __name__ == "__main__":
    unittest.main()
