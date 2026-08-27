"""The front door: bring a circuit, get a verdict -- without the auditor
assuming anything it did not measure.
"""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp

    from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

    HAVE_QISKIT = True
except ImportError:  # pragma: no cover
    HAVE_QISKIT = False

from qem_auditor import Auditor, Verdict
from qem_auditor.cli import EXIT_BAD_RECORD, EXIT_NOT_CERTIFIED, EXIT_OK, main
from qem_auditor.frontdoor import (
    VerificationInputs,
    build_experiment,
    describe_circuit,
    unverifiable_here,
)


def _circuit():
    qc = QuantumCircuit(2, name="probe")
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.3, 0)
    return qc


def _folded(base, n):
    qc = base.copy()
    for _ in range(n):
        qc.cx(0, 1)
        qc.cx(0, 1)
    return qc


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class DescribeTest(unittest.TestCase):
    def test_it_reads_the_circuit_rather_than_asking(self):
        spec = describe_circuit(_circuit())
        self.assertEqual(spec.n_qubits, 2)
        self.assertEqual(spec.n_2q_gates, 1)
        self.assertEqual(spec.n_1q_gates, 2)
        self.assertEqual(spec.circuit_id, "probe")

    def test_a_derived_record_asserts_no_controls(self):
        """Nothing is taken on the user's word, because nothing is asked."""
        exp = build_experiment(VerificationInputs(circuit=_circuit()))
        for control in ("ideal_control", "target_leakage_check", "adversarial_check",
                        "unitary_equivalence", "determinism_check",
                        "free_parameter_floor_test"):
            self.assertIsNone(getattr(exp.controls, control), control)
        self.assertFalse(exp.controls.reproducibility_checked)

    def test_no_replicates_are_invented(self):
        exp = build_experiment(VerificationInputs(circuit=_circuit()))
        self.assertEqual(exp.outputs.replicates, [])

    def test_supplied_replicates_are_marked_independent(self):
        from qem_auditor import ReplicateKind

        exp = build_experiment(VerificationInputs(circuit=_circuit(),
                                                  replicate_errors=(0.1, 0.11)))
        self.assertTrue(all(r.kind is ReplicateKind.INDEPENDENT_SUBMISSION
                            for r in exp.outputs.replicates))


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class VerifyTest(unittest.TestCase):
    def _verify(self, optimization_level, mitigator=None, amplified=True):
        qc = _circuit()
        obs = SparsePauliOp("ZZ")
        amplified_circuit = _folded(qc, 2)
        submitted = transpile(amplified_circuit, basis_gates=["u", "cx"],
                              optimization_level=optimization_level)
        return Auditor(adapter=QiskitAdapter(seed=7)).verify(
            circuit=qc, observable=obs, mitigator=mitigator,
            submitted_circuit=submitted,
            amplified_circuit=amplified_circuit if amplified else None,
            claim="test claim")

    def test_without_declaring_amplification_equivalence_is_all_it_can_ask(self):
        """Honest about its own limit: a fold pair leaves the unitary
        unchanged, so equivalence alone cannot catch a cancelled fold. The
        auditor needs to be told gates were inserted on purpose."""
        result = self._verify(3, amplified=False)
        self.assertIs(result.experiment.controls.unitary_equivalence, True)

    def test_it_measures_what_it_can(self):
        def mitigator(expectation):
            return expectation(_circuit(), SparsePauliOp("ZZ"))

        result = self._verify(0, mitigator)
        measured = {m.control for m in result.measurements}
        self.assertIn("unitary_equivalence", measured)
        self.assertIn("ideal_control", measured)
        self.assertIn("determinism_check", measured)

    def test_a_cancelled_fold_is_caught_from_the_circuit_alone(self):
        self.assertIs(self._verify(3).verdict, Verdict.INVALID)

    def test_a_clean_submission_is_not_invalid(self):
        self.assertIsNot(self._verify(0).verdict, Verdict.INVALID)

    def test_it_never_certifies_from_artifacts_alone(self):
        """Controls needing the researcher stay unrun, so certification is
        structurally out of reach here -- as it should be."""
        def mitigator(expectation):
            return expectation(_circuit(), SparsePauliOp("ZZ"))

        result = self._verify(0, mitigator)
        self.assertIsNot(result.verdict, Verdict.CERTIFIED_UNDER_SCOPE)
        self.assertFalse(result.passed)

    def test_it_says_what_it_cannot_establish(self):
        result = self._verify(0)
        names = [n for n, _ in result.outside_scope]
        self.assertIn("target_leakage", names)
        self.assertIn("reproducibility", names)
        for _, why in result.outside_scope:
            self.assertGreater(len(why), 20)

    def test_without_an_adapter_it_still_grades_but_measures_nothing(self):
        result = Auditor().verify(circuit=_circuit(), claim="x")
        self.assertEqual(result.measurements, [])
        self.assertIsNot(result.verdict, Verdict.CERTIFIED_UNDER_SCOPE)

    def test_a_mitigator_that_raises_does_not_become_a_pass(self):
        def broken(expectation):
            raise ZeroDivisionError("boom")

        result = self._verify(0, broken)
        ideal = next((m for m in result.measurements if m.control == "ideal_control"),
                     None)
        self.assertIsNone(ideal, "a crashing pipeline must not record a passing control")
        self.assertIsNone(result.experiment.controls.ideal_control)


class CheckCommandTest(unittest.TestCase):
    def _run(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(argv)
        return code, out.getvalue()

    def test_template_is_valid_python(self):
        code, out = self._run(["check", "--template"])
        self.assertEqual(code, EXIT_OK)
        compile(out, "template", "exec")

    def test_template_asserts_no_controls(self):
        """A starting point must not pre-fill anything as passed."""
        _, out = self._run(["check", "--template"])
        for banned in ("ideal_control =", "adversarial_check =", "verdict"):
            self.assertNotIn(banned, out)

    def test_a_file_without_a_circuit_is_rejected_helpfully(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.py"
            p.write_text("x = 1\n")
            code = main(["check", str(p)])
            self.assertEqual(code, EXIT_BAD_RECORD)

    def test_a_missing_file_is_rejected(self):
        self.assertEqual(main(["check", "/nonexistent/nope.py"]), EXIT_BAD_RECORD)

    def test_no_path_and_no_template_is_rejected(self):
        self.assertEqual(main(["check"]), EXIT_BAD_RECORD)

    @unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
    def test_the_template_round_trips_through_check(self):
        """The starting point we hand people must actually run."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.py"
            _, template = self._run(["check", "--template"])
            p.write_text(template)
            code, out = self._run(["check", str(p)])
            self.assertEqual(code, EXIT_NOT_CERTIFIED)
            self.assertIn("EXECUTED BY THE AUDITOR", out)
            self.assertIn("OUTSIDE WHAT THIS CHECK CAN ESTABLISH", out)

    @unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
    def test_it_can_save_the_derived_record(self):
        from qem_auditor import record

        with tempfile.TemporaryDirectory() as d:
            src, saved = Path(d) / "c.py", Path(d) / "rec.json"
            _, template = self._run(["check", "--template"])
            src.write_text(template)
            self._run(["check", str(src), "--save", str(saved)])
            self.assertTrue(saved.exists())
            record.load(saved)  # must be a valid record


class ScopeHonestyTest(unittest.TestCase):
    def test_every_unverifiable_control_explains_itself(self):
        for name, why in unverifiable_here():
            self.assertTrue(name.strip())
            self.assertGreater(len(why), 20, f"{name} needs a real explanation")


if __name__ == "__main__":
    unittest.main()
