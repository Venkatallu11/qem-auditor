"""The mitigation IBM applies before anyone asks for any.

The IonQ reader is built against a real job record. This one is not: it
encodes the documented way the service RESOLVES options nobody set, and
documented behaviour can change under you without a single line of your
own code moving. So the last class here reads the installed
qiskit-ibm-runtime's own docstrings and fails if IBM's defaults stop
matching what this module claims about them -- which is the same
anti-staleness argument this package makes about everything else,
turned on itself.
"""
import unittest

from qem_auditor.vendor import (IBM_DEFAULT_RESILIENCE_LEVEL, cross_check_ibm,
                                ibm_estimator_mitigation)

try:
    from qiskit_ibm_runtime.options import (estimator_options,
                                            resilience_options,
                                            twirling_options)

    HAVE_IBM_RUNTIME = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_IBM_RUNTIME = False


class DefaultPathTest(unittest.TestCase):
    """What `Estimator(mode=backend)` does when you configure nothing."""

    def setUp(self):
        self.bare = ibm_estimator_mitigation()

    def test_the_default_level_is_not_zero(self):
        self.assertEqual(IBM_DEFAULT_RESILIENCE_LEVEL, 1)
        self.assertEqual(self.bare.resilience_level, 1)
        self.assertTrue(self.bare.level_defaulted)

    def test_two_methods_are_on_that_nobody_switched_on(self):
        self.assertTrue(self.bare.readout_mitigation)
        self.assertTrue(self.bare.measurement_twirling)
        self.assertEqual(
            set(self.bare.unrequested),
            {"readout error mitigation", "measurement twirling"})

    def test_the_expensive_ones_are_not(self):
        self.assertFalse(self.bare.gate_twirling)
        self.assertFalse(self.bare.zne)
        self.assertFalse(self.bare.pec)

    def test_a_bare_run_is_not_an_unmitigated_baseline(self):
        found = cross_check_ibm(self.bare, claims_unmitigated=True)
        self.assertEqual(len(found), 1)
        self.assertIn("never requested", found[0].reading)
        self.assertIn("left at its default", found[0].reading)


class ResolutionTest(unittest.TestCase):

    def test_level_zero_really_is_raw(self):
        raw = ibm_estimator_mitigation({"resilience_level": 0})
        self.assertEqual(raw.applied, ())
        self.assertIn("really is a raw run", raw.describe())
        self.assertEqual(cross_check_ibm(raw, claims_unmitigated=True), ())

    def test_level_two_turns_on_zne_and_gate_twirling(self):
        high = ibm_estimator_mitigation({"resilience_level": 2})
        self.assertTrue(high.zne)
        self.assertTrue(high.gate_twirling)
        self.assertTrue(high.readout_mitigation)

    def test_pec_never_turns_itself_on(self):
        """It is the one option with no level that flips it."""
        for level in (0, 1, 2):
            self.assertFalse(
                ibm_estimator_mitigation({"resilience_level": level}).pec)

    def test_pec_asked_for_is_recorded_as_asked_for(self):
        asked = ibm_estimator_mitigation(
            {"resilience_level": 1, "resilience": {"pec_mitigation": True}})
        self.assertTrue(asked.pec)
        self.assertIn("pec", asked.explicit)
        self.assertNotIn("PEC", asked.unrequested)

    def test_an_explicit_false_beats_the_level(self):
        off = ibm_estimator_mitigation(
            {"resilience_level": 2, "resilience": {"zne_mitigation": False}})
        self.assertFalse(off.zne)
        self.assertIn("zne", off.explicit)

    def test_explicitly_asked_for_is_not_a_surprise(self):
        """The contradiction still stands -- the baseline is still not
        unmitigated -- but it is not reported as something nobody chose."""
        asked = ibm_estimator_mitigation(
            {"resilience_level": 0, "resilience": {"measure_mitigation": True}})
        self.assertTrue(asked.readout_mitigation)
        self.assertEqual(asked.unrequested, ())
        found = cross_check_ibm(asked, claims_unmitigated=True)
        self.assertEqual(len(found), 1)
        self.assertNotIn("never requested", found[0].reading)

    def test_option_objects_read_the_same_as_dicts(self):
        """A live Estimator hands over nested objects, a saved payload
        hands over dicts. Blessing one shape would refuse the other."""

        class Resilience:
            measure_mitigation = False

        class Options:
            resilience_level = 0
            resilience = Resilience()

        from_object = ibm_estimator_mitigation(Options())
        from_dict = ibm_estimator_mitigation(
            {"resilience_level": 0, "resilience": {"measure_mitigation": False}})
        self.assertEqual(from_object.applied, from_dict.applied)
        self.assertEqual(from_object.explicit, from_dict.explicit)

    def test_a_claim_of_mitigation_is_not_contradicted(self):
        """Only an UNMITIGATED claim conflicts with mitigation."""
        self.assertEqual(cross_check_ibm(ibm_estimator_mitigation()), ())


@unittest.skipUnless(HAVE_IBM_RUNTIME, "needs qiskit-ibm-runtime")
class DefaultsStillHoldTest(unittest.TestCase):
    """Read IBM's own docstrings and fail if their defaults moved.

    This module's rules are transcribed, not observed. A transcription
    has a shelf life, and the failure mode is silent: the library
    changes, this keeps answering confidently, and the answer is about
    a version nobody is running. So the source is the test fixture.
    """

    @staticmethod
    def source(module) -> str:
        import inspect
        return " ".join(inspect.getsource(module).split())

    def test_the_default_resilience_level_is_still_one(self):
        text = self.source(estimator_options)
        self.assertIn("resilience_level", text)
        level = text.split("resilience_level", 1)[1]
        self.assertIn("Default: 1.", level[:1200])

    def test_readout_mitigation_still_switches_on_at_level_one(self):
        text = self.source(resilience_options)
        block = text.split("measure_mitigation", 1)[1][:900]
        self.assertIn("``False`` for resilience level 0", block)
        self.assertIn("``True`` for resilience levels 1 and 2", block)

    def test_zne_still_waits_for_level_two(self):
        block = self.source(resilience_options).split(
            "zne_mitigation", 1)[1][:900]
        self.assertIn("``False`` for resilience levels 0 and 1", block)
        self.assertIn("``True`` for resilience level 2", block)

    def test_pec_still_defaults_to_false(self):
        block = self.source(resilience_options).split(
            "pec_mitigation", 1)[1][:900]
        self.assertIn("Default: False.", block)

    def test_measurement_twirling_still_switches_on_at_level_one(self):
        block = self.source(twirling_options).split(
            "enable_measure", 1)[1][:900]
        self.assertIn("``False`` for resilience level 0", block)
        self.assertIn("``True`` for resilience levels 1 and 2", block)

    def test_gate_twirling_still_waits_for_level_two(self):
        block = self.source(twirling_options).split("enable_gates", 1)[1][:900]
        self.assertIn("``False`` for resilience levels 0 and 1", block)
        self.assertIn("``True`` for resilience level 2", block)


if __name__ == "__main__":
    unittest.main()
