"""The Qiskit adapter: does it actually catch the historical failures when
run against the real transpiler?

Skipped entirely when Qiskit is absent -- the core auditor has no
dependencies and must stay testable without them.
"""
import unittest

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp

    from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

    HAVE_QISKIT = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_QISKIT = False

from qem_auditor import Auditor, Provenance, Verdict
from qem_auditor.adapters.base import MeasurementError

from .helpers import make_experiment

BASIS = ["u", "cx"]


def _base():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.3, 0)
    return qc


def _folded(base, folds):
    qc = base.copy()
    for _ in range(folds):
        qc.cx(0, 1)
        qc.cx(0, 1)
    return qc


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class UnitaryEquivalenceTest(unittest.TestCase):
    def test_identical_circuits_are_equivalent(self):
        m = QiskitAdapter().measure_unitary_equivalence(_base(), _base())
        self.assertIs(m.passed, True)

    def test_a_genuinely_different_circuit_fails(self):
        other = QuantumCircuit(2)
        other.h(0)
        other.cx(0, 1)
        other.rx(1.7, 0)  # different angle
        self.assertIs(QiskitAdapter().measure_unitary_equivalence(_base(), other).passed,
                      False)

    def test_qubit_count_mismatch_fails_without_building_operators(self):
        self.assertIs(QiskitAdapter().measure_unitary_equivalence(
            _base(), QuantumCircuit(3)).passed, False)

    def test_a_circuit_with_measurements_reports_a_measurement_error(self):
        """Not silently a failure: an unmeasurable check is not a failed one."""
        qc = _base()
        qc.measure_all()
        with self.assertRaises(MeasurementError):
            QiskitAdapter().measure_unitary_equivalence(_base(), qc)


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class FoldSurvivalTest(unittest.TestCase):
    """The historical bug, reproduced against the real Qiskit transpiler."""

    def test_optimization_level_3_cancels_the_folds_and_is_caught(self):
        base = _base()
        submitted = transpile(_folded(base, 2), basis_gates=BASIS, optimization_level=3)
        m = QiskitAdapter().measure_fold_survival(base, submitted)
        self.assertIs(m.passed, False)
        self.assertIn("did NOT survive", m.detail)

    def test_optimization_level_0_preserves_them(self):
        base = _base()
        submitted = transpile(_folded(base, 2), basis_gates=BASIS, optimization_level=0)
        m = QiskitAdapter().measure_fold_survival(base, submitted)
        self.assertIs(m.passed, True)
        self.assertGreater(m.evidence["amplification_ratio"], 1.0)

    def test_unitary_equivalence_alone_would_have_missed_it(self):
        """The reason fold survival is a separate check: a cancelled fold
        pair leaves the unitary correct, so an equivalence check passes."""
        base = _base()
        submitted = transpile(_folded(base, 2), basis_gates=BASIS, optimization_level=3)
        self.assertIs(QiskitAdapter().measure_unitary_equivalence(base, submitted).passed,
                      True)
        self.assertIs(QiskitAdapter().measure_fold_survival(base, submitted).passed,
                      False)

    def test_folding_that_changes_the_unitary_fails(self):
        base = _base()
        broken = base.copy()
        broken.cx(0, 1)  # unpaired: changes the computation
        self.assertIs(QiskitAdapter().measure_fold_survival(base, broken).passed, False)


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class IdealControlTest(unittest.TestCase):
    OBS = property(lambda self: SparsePauliOp("ZZ"))

    def test_a_no_op_mitigator_passes(self):
        a = QiskitAdapter(seed=3)
        m = a.measure_ideal_control(_base(), SparsePauliOp("ZZ"),
                                    lambda expectation: expectation(_base(), SparsePauliOp("ZZ")),
                                    shots=20_000)
        self.assertIs(m.passed, True)

    def test_a_pathological_extrapolator_is_caught(self):
        """The 513x shape: huge coefficients amplify pure shot noise on a
        model with nothing to correct."""
        def mitigate(expectation):
            vals = [expectation(_folded(_base(), f), SparsePauliOp("ZZ")) for f in range(5)]
            coeffs = [70.0, -140.0, 90.0, -20.0, 1.0]
            return sum(c * v for c, v in zip(coeffs, vals))

        m = QiskitAdapter(seed=3).measure_ideal_control(
            _base(), SparsePauliOp("ZZ"), mitigate, shots=20_000)
        self.assertIs(m.passed, False)
        self.assertGreater(m.evidence["amplification"], 10.0)

    def test_a_pipeline_that_crashes_on_clean_input_is_a_measurement_error(self):
        def broken(expectation):
            raise ZeroDivisionError("no noise to divide by")

        with self.assertRaises(MeasurementError):
            QiskitAdapter().measure_ideal_control(_base(), SparsePauliOp("ZZ"), broken)

    def test_evidence_lets_a_reader_recompute_the_verdict(self):
        m = QiskitAdapter(seed=3).measure_ideal_control(
            _base(), SparsePauliOp("ZZ"),
            lambda e: e(_base(), SparsePauliOp("ZZ")), shots=20_000)
        for key in ("exact", "raw_error", "mitigated_error", "amplification",
                    "shots", "trials", "statistic"):
            self.assertIn(key, m.evidence)
        # The statistic is named, because a median over paired trials and a
        # single draw are different claims and the reader must not have to
        # guess which one this is.
        self.assertIn("median", m.evidence["statistic"])

    def test_the_amplification_is_stable_across_shot_counts(self):
        """A single-draw ratio is not: it once reported 43x for an
        estimator whose true amplification is about 4.4x."""
        adapter = QiskitAdapter(seed=3)
        obs = SparsePauliOp("ZZ")

        def zne(expectation):
            v = [expectation(_folded(_base(), f), obs) for f in (0, 1, 2)]
            return 3 * v[0] - 3 * v[1] + v[2]

        low = adapter.measure_ideal_control(_base(), obs, zne, shots=20_000)
        high = adapter.measure_ideal_control(_base(), obs, zne, shots=200_000)
        self.assertAlmostEqual(low.evidence["amplification"],
                               high.evidence["amplification"], delta=1.5)

    def test_a_single_trial_is_refused(self):
        from qem_auditor.adapters.base import MeasurementError

        with self.assertRaises(MeasurementError):
            QiskitAdapter().measure_ideal_control(
                _base(), SparsePauliOp("ZZ"),
                lambda e: e(_base(), SparsePauliOp("ZZ")), trials=1)


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class DeterminismTest(unittest.TestCase):
    def test_a_deterministic_computation_passes(self):
        self.assertIs(QiskitAdapter().measure_determinism(lambda: 1.25).passed, True)

    def test_a_nondeterministic_computation_fails(self):
        import itertools

        counter = itertools.count()
        m = QiskitAdapter().measure_determinism(lambda: float(next(counter)))
        self.assertIs(m.passed, False)
        self.assertIn("different results", m.detail)

    def test_default_tolerance_is_exact(self):
        """A 'close enough' default would hide the ordering and threading
        bugs this check exists to find."""
        vals = iter([1.0, 1.0 + 1e-12, 1.0])
        m = QiskitAdapter().measure_determinism(lambda: next(vals))
        self.assertIs(m.passed, False)

    def test_a_single_run_cannot_be_judged(self):
        with self.assertRaises(MeasurementError):
            QiskitAdapter().measure_determinism(lambda: 1.0, runs=1)


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class AuditorIntegrationTest(unittest.TestCase):
    def test_a_measured_control_is_marked_measured(self):
        exp = make_experiment()
        auditor = Auditor(adapter=QiskitAdapter())
        auditor.verify_determinism(exp, lambda: 1.0)
        self.assertIs(exp.controls.provenance_of("determinism_check"), Provenance.MEASURED)
        self.assertIs(exp.controls.determinism_check, True)

    def test_a_measured_failure_drives_the_verdict(self):
        exp = make_experiment()
        auditor = Auditor(adapter=QiskitAdapter())
        base = _base()
        submitted = transpile(_folded(base, 2), basis_gates=BASIS, optimization_level=3)
        auditor.verify_fold_survival(exp, base=base, submitted=submitted)
        self.assertIs(auditor.audit(exp).verdict, Verdict.INVALID)

    def test_the_measurement_reaches_the_diagnosis(self):
        exp = make_experiment()
        auditor = Auditor(adapter=QiskitAdapter(seed=3))

        def mitigate(expectation):
            vals = [expectation(_folded(_base(), f), SparsePauliOp("ZZ")) for f in range(5)]
            return sum(c * v for c, v in zip([70.0, -140.0, 90.0, -20.0, 1.0], vals))

        auditor.verify_ideal_control(exp, _base(), SparsePauliOp("ZZ"), mitigate,
                                     shots=20_000)
        result = auditor.audit(exp)
        self.assertIs(result.verdict, Verdict.INVALID)
        primary = result.analysis.primary
        self.assertIn("the auditor ran the pipeline", primary.evidence)

    def test_self_reported_records_cannot_certify(self):
        """The rule that makes this a verifier rather than a rubric."""
        exp = make_experiment()
        exp.controls.provenance.clear()
        self.assertIsNot(Auditor().audit(exp).verdict, Verdict.CERTIFIED_UNDER_SCOPE)


if __name__ == "__main__":
    unittest.main()
