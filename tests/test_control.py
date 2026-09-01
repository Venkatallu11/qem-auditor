"""The control experiment, and the two bars the ported version lacked.

Ported from `quantum-verifier`'s `falsify`. Its idea -- strip the
mechanism, keep everything else, and let the confounds cancel -- is
better than anything this package had for auditing a CLAIM rather than a
METHOD. What is added here is the arithmetic that says which of the
resulting numbers are findings.
"""
import unittest

from qem_auditor.control import (ControlError, distribution_shift,
                                 isolate_effect, total_variation)


def uniform(n_outcomes, shots, seed):
    """Counts from a uniform distribution, by a stdlib generator."""
    import random
    rng = random.Random(seed)
    table = {}
    for _ in range(shots):
        key = format(rng.randrange(n_outcomes), "012b")
        table[key] = table.get(key, 0) + 1
    return table


class IsolatedEffectTest(unittest.TestCase):

    def test_a_real_effect_clears_its_own_noise(self):
        effect = isolate_effect({"000": 3700, "111": 396}, {"000": 1873, "111": 2223},
                                marked={"000"}, removed=2)
        self.assertGreater(effect.effect, 0.4)
        self.assertTrue(effect.real)
        self.assertIn("the mechanism contributes", effect.describe())

    def test_an_effect_inside_the_noise_is_refused(self):
        """The ported version reports the number and interprets it in prose.
        At 4096 shots 0.01 and 0.35 differ in kind, not degree."""
        effect = isolate_effect({"010": 56, "000": 4040}, {"010": 55, "000": 4041},
                                marked={"010"}, removed=2)
        self.assertFalse(effect.real)
        self.assertIn("NOT distinguishable from zero", effect.describe())

    def test_it_says_how_many_shots_would_resolve_a_small_effect(self):
        effect = isolate_effect({"010": 60, "000": 4036}, {"010": 55, "000": 4041},
                                marked={"010"}, removed=2)
        self.assertFalse(effect.real)
        self.assertGreater(effect.shots_for_a_signal, 4096)

    def test_an_exactly_zero_effect_quotes_no_shot_count(self):
        """No finite number of shots establishes that a mechanism
        contributes nothing."""
        table = {"010": 55, "000": 4041}
        effect = isolate_effect(table, dict(table), marked={"010"}, removed=2)
        self.assertEqual(effect.effect, 0.0)
        self.assertIsNone(effect.shots_for_a_signal)
        self.assertIn("no number of shots", effect.describe())

    def test_a_run_with_no_shots_is_refused(self):
        with self.assertRaises(ControlError):
            isolate_effect({}, {"0": 10}, marked={"0"}, removed=1)


class DistributionShiftTest(unittest.TestCase):
    """TVD in discovery mode is biased upward, badly, and the bias grows
    with width. Two independent samples of the SAME distribution give
    roughly sqrt(outcomes/shots) -- about 0.28 at ten qubits and 4096
    shots, about 0.52 at twelve. Reading that against a scale where
    "0 = identical, 1 = completely different" calls noise a finding."""

    def test_two_samples_of_one_distribution_are_not_called_a_shift(self):
        a, b = uniform(1024, 4096, seed=1), uniform(1024, 4096, seed=2)
        shift = distribution_shift(a, b, replicates=60, seed=0)
        self.assertGreater(shift.observed, 0.2,
                           "raw TVD really is this large for identical distributions")
        self.assertFalse(shift.real)
        self.assertIn("no shift is established", shift.describe())

    def test_a_genuine_shift_clears_the_null(self):
        shift = distribution_shift({"000000000000": 4000, "111111111111": 96},
                                   uniform(1024, 4096, seed=3),
                                   replicates=60, seed=0)
        self.assertTrue(shift.real)
        self.assertGreater(shift.excess, 0.0)

    def test_the_null_moves_with_the_distribution_not_just_the_width(self):
        """Why a rule of thumb will not do. A concentrated distribution has
        far fewer effective outcomes, so its null is far lower -- measured
        at 0.011 against 0.279 for a near-uniform one at equal width and
        shots."""
        flat = distribution_shift(uniform(1024, 4096, 1), uniform(1024, 4096, 2),
                                  replicates=60, seed=0)
        sharp = distribution_shift({"000000000000": 2000, "111111111111": 2096},
                                   {"000000000000": 2050, "111111111111": 2046},
                                   replicates=60, seed=0)
        self.assertGreater(flat.null_median, 10 * sharp.null_median)

    def test_candidate_gainers_are_withheld_when_the_shift_is_not_real(self):
        """Ranking thousands of noisy differences and printing the top five
        is a selection: the largest is large whether or not anything
        happened. Offering them as 'candidate answers' invites a reader to
        chase noise."""
        a, b = uniform(1024, 4096, seed=1), uniform(1024, 4096, seed=2)
        self.assertEqual(distribution_shift(a, b, replicates=60, seed=0).gainers, ())

    def test_gainers_are_offered_when_it_is_real(self):
        shift = distribution_shift({"000000000000": 4000, "111111111111": 96},
                                   uniform(1024, 4096, seed=3),
                                   replicates=60, seed=0)
        self.assertIn("000000000000", shift.gainers)

    def test_total_variation_refuses_an_empty_run(self):
        with self.assertRaises(ControlError):
            total_variation({}, {"0": 5})


class BuildControlTest(unittest.TestCase):

    def setUp(self):
        try:
            from qiskit import QuantumCircuit  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("needs qiskit to build a circuit")

    def test_a_circuit_with_no_entangling_gate_is_refused(self):
        """The control would be the circuit itself, so the comparison
        isolates nothing while looking like it had."""
        from qiskit import QuantumCircuit
        from qem_auditor.control import build_control
        flat = QuantumCircuit(2, 2)
        flat.h(0)
        flat.measure([0, 1], [0, 1])
        with self.assertRaises(ControlError):
            build_control(flat)

    def test_everything_but_the_mechanism_survives(self):
        from qiskit import QuantumCircuit
        from qem_auditor.control import build_control
        ghz = QuantumCircuit(3, 3)
        ghz.h(0)
        ghz.cx(0, 1)
        ghz.cx(1, 2)
        ghz.measure(range(3), range(3))
        control, removed = build_control(ghz)
        self.assertEqual(removed, 2)
        self.assertEqual(control.num_qubits, ghz.num_qubits)
        self.assertEqual(control.num_clbits, ghz.num_clbits)
        self.assertEqual(control.count_ops().get("measure"), 3)
        self.assertEqual(control.count_ops().get("h"), 1)
        self.assertNotIn("cx", control.count_ops())


class PreflightGateTest(unittest.TestCase):
    """The bridge: this package's exact check, in the GO/BLOCK shape a
    preflight chain speaks."""

    ENCODE = staticmethod(lambda v: [(v >> 0) & 1, (v >> 1) & 1])

    def test_a_correct_circuit_goes(self):
        from qem_auditor.reversible import preflight_gate
        marks_two = [("x", [0]), ("cz", [0, 1]), ("x", [0])]
        verdict = preflight_gate(marks_two, lambda v: v == 2, 4, self.ENCODE)
        self.assertEqual(verdict["verdict"], "GO")

    def test_a_wrong_circuit_blocks_with_the_discrepancies_named(self):
        from qem_auditor.reversible import preflight_gate
        marks_two = [("x", [0]), ("cz", [0, 1]), ("x", [0])]
        verdict = preflight_gate(marks_two, lambda v: v == 3, 4, self.ENCODE)
        self.assertEqual(verdict["verdict"], "BLOCK")
        self.assertIn("specified but unmarked", verdict["reason"])

    def test_an_inapplicable_circuit_skips_rather_than_passing(self):
        """SKIP is not GO. Collapsing them would let an unchecked circuit
        through wearing a pass."""
        from qem_auditor.reversible import preflight_gate
        verdict = preflight_gate([("h", [0])], lambda v: True, 4, self.ENCODE)
        self.assertEqual(verdict["verdict"], "SKIP")
        self.assertIsNone(verdict["report"])
