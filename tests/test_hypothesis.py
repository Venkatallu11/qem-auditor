"""The hypothesis ledger: does evidence accumulate correctly, and does a
killed hypothesis stay killed?
"""
import math
import unittest

from qem_auditor.hypothesis import Hypothesis, HypothesisLedger, Observation, entropy


def _ledger(n=4):
    return HypothesisLedger([Hypothesis(f"H{i}", f"hypothesis {i}", 1.0) for i in range(n)])


class EntropyTest(unittest.TestCase):
    def test_uniform_over_four_is_two_bits(self):
        self.assertAlmostEqual(entropy({f"H{i}": 0.25 for i in range(4)}), 2.0)

    def test_certainty_is_zero_bits(self):
        self.assertAlmostEqual(entropy({"H0": 1.0, "H1": 0.0}), 0.0)

    def test_unnormalized_input_is_normalized(self):
        self.assertAlmostEqual(entropy({"H0": 2.0, "H1": 2.0}), 1.0)


class LedgerTest(unittest.TestCase):
    def test_priors_are_normalized(self):
        led = HypothesisLedger([Hypothesis("H0", "a", 3.0), Hypothesis("H1", "b", 1.0)])
        self.assertAlmostEqual(led.belief["H0"], 0.75)
        self.assertAlmostEqual(sum(led.belief.values()), 1.0)

    def test_evidence_against_drives_belief_down(self):
        led = _ledger()
        led.update(Observation("e1", {"H0": 0.01}, "ruled out by measurement"))
        self.assertLess(led.belief["H0"], 0.01)
        self.assertAlmostEqual(sum(led.belief.values()), 1.0)

    def test_a_killed_hypothesis_stays_killed(self):
        """Later evidence that says nothing about H0 must not revive it."""
        led = _ledger()
        led.update(Observation("e1", {"H0": 0.001}))
        after_kill = led.belief["H0"]
        led.update(Observation("e2", {"H1": 2.0}))
        led.update(Observation("e3", {"H2": 3.0}))
        self.assertLess(led.belief["H0"], after_kill)

    def test_silence_is_not_evidence(self):
        """A hypothesis the observation does not mention keeps likelihood 1,
        so its belief moves only by renormalization, never by assumption."""
        led = _ledger()
        before = led.belief["H3"]
        led.update(Observation("e1", {"H0": 1.0, "H1": 1.0, "H2": 1.0, "H3": 1.0}))
        self.assertAlmostEqual(led.belief["H3"], before)

    def test_evidence_is_recorded_on_the_right_side(self):
        led = _ledger()
        led.update(Observation("e1", {"H0": 5.0, "H1": 0.2}, "supports H0"))
        self.assertTrue(led.hypotheses["H0"].evidence_for)
        self.assertTrue(led.hypotheses["H1"].evidence_against)
        self.assertFalse(led.hypotheses["H0"].evidence_against)

    def test_observations_reduce_entropy_when_discriminating(self):
        led = _ledger()
        before = led.entropy
        led.update(Observation("e1", {"H0": 0.01, "H1": 0.01, "H2": 0.01}))
        self.assertLess(led.entropy, before)

    def test_resolution_threshold(self):
        led = _ledger()
        self.assertFalse(led.is_resolved())
        led.update(Observation("e1", {"H1": 0.001, "H2": 0.001, "H3": 0.001}))
        self.assertTrue(led.is_resolved())
        self.assertEqual(led.leading()[0], "H0")


class LedgerValidationTest(unittest.TestCase):
    def test_unknown_hypothesis_is_rejected(self):
        with self.assertRaises(KeyError):
            _ledger().update(Observation("e1", {"H_typo": 0.5}))

    def test_data_impossible_under_every_hypothesis_raises(self):
        """A finding in its own right: the explanation set is incomplete."""
        with self.assertRaises(ValueError):
            _ledger().update(Observation("e1", {f"H{i}": 0.0 for i in range(4)}))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            HypothesisLedger([Hypothesis("H0", "a", 1.0), Hypothesis("H0", "b", 1.0)])


class RealDecisionTrailTest(unittest.TestCase):
    """The H4 project's actual sequence, replayed."""

    def test_the_project_history_reproduces(self):
        led = HypothesisLedger([
            Hypothesis("H1", "1Q noise dominates", 0.25),
            Hypothesis("H2", "the ZNE extrapolator is stable", 0.25),
            Hypothesis("H3", "PEC suffices under a calibrated channel", 0.25),
            Hypothesis("H4", "ancilla parity + conditioned PEC reduces error", 0.25),
        ])
        led.update(Observation("task28a_1q_spectroscopy", {"H1": 0.02},
                               "measured 1Q noise far too small to dominate"))
        led.update(Observation("h4_all_gate_zne_ideal_control", {"H2": 0.001},
                               "513x blowup on a model with zero real noise"))
        led.update(Observation("task31h_robustness_envelope", {"H3": 0.05},
                               "Q95 51.22 kcal/mol once calibration uncertainty varied"))
        led.update(Observation("task39_ancilla_qed", {"H4": 3.0},
                               "4 independent draws in a 0.0105-0.0192 band"))
        self.assertEqual(led.leading()[0], "H4")
        self.assertTrue(led.is_resolved())
        for dead in ("H1", "H2", "H3"):
            self.assertLess(led.belief[dead], 0.05)


if __name__ == "__main__":
    unittest.main()
