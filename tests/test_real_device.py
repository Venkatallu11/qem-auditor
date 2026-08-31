"""The real-device audit, and the prediction it tested.

The auditor told the depolarizing run that "a result under one fixed
noise model predicts little about hardware". That is falsifiable, and
these tests pin the outcome of falsifying it: under IBM's measured
fake_kyiv calibration the same protocol's 5.53x improvement collapses to
roughly 1.2x, and the reason is an error gate folding structurally
cannot reach.
"""
import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

try:
    import qiskit_aer  # noqa: F401

    HAVE_AER = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_AER = False


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class CalibrationTest(unittest.TestCase):
    def test_the_pinned_calibration_matches_its_source(self):
        """The example pins fake_kyiv's numbers so it runs on aer alone.
        A pinned copy that silently drifts from its source is a
        transcription claiming to be a measurement, which is the thing
        this package exists to object to."""
        try:
            import qiskit_ibm_runtime  # noqa: F401
        except ImportError:
            self.skipTest("qiskit-ibm-runtime not installed; nothing to compare against")

        import real_device_audit as rd

        live = rd.calibration()
        for key, pinned in rd.MEASURED.items():
            with self.subTest(parameter=key):
                if isinstance(pinned, tuple):
                    for a, b in zip(pinned, live[key]):
                        self.assertAlmostEqual(a, b, places=12)
                else:
                    self.assertAlmostEqual(pinned, live[key], places=12)

    def test_the_pair_was_chosen_by_gate_error_not_by_result(self):
        try:
            from qiskit_ibm_runtime.fake_provider import FakeKyiv
        except ImportError:
            self.skipTest("qiskit-ibm-runtime not installed")

        import real_device_audit as rd

        device = FakeKyiv()
        errors = [device.properties().gate_error("ecr", list(p))
                  for p in device.coupling_map]
        self.assertAlmostEqual(rd.MEASURED["ecr_error"], min(errors), places=12)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class ReadoutErrorIsUnreachableTest(unittest.TestCase):
    """Gate folding amplifies what scales with gate count. Readout error
    happens once at measurement whatever the fold factor, so no
    extrapolation in that factor can remove it. This is the mechanism
    behind the collapse, isolated rather than asserted."""

    @classmethod
    def setUpClass(cls):
        import real_device_audit as rd

        cls.rd = rd
        cls.cal = rd.calibration()
        cls.gains = {}
        for label, switches in rd.ABLATION:
            raw, mitigated = rd.run_protocol(rd.device_noise(cls.cal, **switches))
            cls.gains[label] = statistics.median(raw) / statistics.median(mitigated)

    def test_zne_works_when_every_error_scales_with_gate_count(self):
        self.assertGreater(self.gains["gate + decoherence"], 2.0)

    def test_adding_readout_error_alone_destroys_the_gain(self):
        self.assertGreater(self.gains["gate errors only"], 1.5)
        self.assertLess(self.gains["gate + readout"], 1.5)

    def test_the_measured_device_is_in_the_readout_dominated_regime(self):
        """Which is why the hardware answer differs from the invented one."""
        self.assertLess(self.gains["all three, as measured"], 1.5)

    def test_readout_error_is_the_dominant_term_on_this_pair(self):
        self.assertGreater(self.cal["readout_error"], 5 * self.cal["ecr_error"])


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class VerdictTest(unittest.TestCase):
    def test_the_auditor_still_refuses_to_certify_and_names_the_reason(self):
        from qem_auditor import FailureMode, Verdict, audit, classify

        import real_device_audit as rd

        cal = rd.calibration()
        raw, mitigated = rd.run_protocol(rd.device_noise(cal))
        experiment = rd.build_record(raw, mitigated, cal)
        report = audit(experiment)

        self.assertIsNot(report.verdict, Verdict.CERTIFIED_UNDER_SCOPE)
        failed = [g.name for g in report.gate_results if g.passed is False]
        self.assertIn("evidence_scope", failed)
        modes = [d.mode for d in classify(experiment, report).diagnoses]
        self.assertIn(FailureMode.CALIBRATION_MISMATCH, modes)


if __name__ == "__main__":
    unittest.main()
