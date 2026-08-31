"""Does any of it generalise beyond one molecule?

Every quantitative claim in this repository was measured on H2 in
STO-3G: two qubits, two CX gates. These pin what survived a second,
structurally different system and -- as importantly -- what did not.
"""
import statistics
import unittest

try:
    from qiskit_aer import AerSimulator

    HAVE_AER = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_AER = False

if HAVE_AER:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from benchmarks import tfim
    from benchmarks.methods import (METHODS, Sampler, ScrambledSampler,
                                    h2_system, unmitigated)
    from real_device_audit import calibration, device_noise

#: Low on purpose. Every assertion in this file is a large-margin
#: comparison -- readout share below 60%, the fraud on top, PEC worse
#: than REM+ZNE -- and shots buy runtime rather than confidence in any of
#: them. The H2 spread check keeps its own budget, because comparing a
#: median against a pinned spread is the one claim here that needs it.
SHOTS = 6_000
SEEDS = (101, 202)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class HamiltonianTest(unittest.TestCase):
    """Constructed here rather than transcribed, so there is nothing to
    have copied wrong."""

    def test_the_chain_has_the_terms_it_should(self):
        operator = tfim.hamiltonian(4)
        self.assertEqual(len(operator), 3 + 4)   # 3 bonds, 4 fields

    def test_a_chain_of_one_spin_is_refused(self):
        with self.assertRaises(ValueError):
            tfim.hamiltonian(1)

    def test_the_ground_energy_matches_an_independent_diagonalisation(self):
        import numpy as np

        operator = tfim.hamiltonian(4)
        self.assertAlmostEqual(
            tfim.exact_ground_energy(operator),
            float(np.linalg.eigvalsh(operator.to_matrix())[0]), places=10)

    def test_depth_is_a_knob(self):
        """The property the whole comparison rests on."""
        shallow, _ = tfim.gate_counts(tfim.trotter_circuit(4, 1))
        deep, _ = tfim.gate_counts(tfim.trotter_circuit(4, 8))
        self.assertEqual(deep, 8 * shallow)

    def test_an_evolution_of_no_steps_is_refused(self):
        with self.assertRaises(ValueError):
            tfim.trotter_circuit(4, 0)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class TrainingSetTest(unittest.TestCase):
    """The degenerate training set that nearly shipped.

    Replacing the timestep wholesale gave every training circuit the same
    exact value, so CDR's regression had slope zero and returned its
    intercept whatever it was shown -- and that constant happened to beat
    every real method. The scramble attack caught it at 0.000.
    """

    def test_training_targets_have_a_spread_to_regress_against(self):
        variants = tfim.system(4, 4).clifford_variants
        targets = [t for _, t in variants]
        self.assertGreater(max(targets) - min(targets), 0.5)

    def test_a_degenerate_training_set_is_refused_rather_than_fitted(self):
        with self.assertRaises(ValueError):
            tfim.near_clifford_variants(4, 4, 0.3, 1.0, 0.5, n_variants=1)

    def test_training_circuits_keep_the_depth_of_the_target(self):
        """Training on shallower circuits breaks the premise that the
        noise map transfers."""
        system = tfim.system(4, 4)
        target, _ = tfim.gate_counts(system.circuit)
        for circuit, _ in system.clifford_variants:
            with self.subTest():
                self.assertEqual(tfim.gate_counts(circuit)[0], target)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class GeneralisationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = calibration()
        cls.system = tfim.system(4, 4)
        cls.backend = AerSimulator(noise_model=device_noise(cls.cal))
        cls.errors = {}
        for name, method in METHODS.items():
            try:
                cls.errors[name] = statistics.median(
                    [cls.system.error(method(Sampler(cls.backend, SHOTS, s,
                                                     cls.system)))
                     for s in SEEDS])
            except ValueError:
                cls.errors[name] = None

    def test_readout_no_longer_dominates_on_a_deeper_circuit(self):
        """The H2 finding was a fact about a two-gate circuit. If it
        survived 24 gates the mechanism behind it would be wrong."""
        no_readout = AerSimulator(noise_model=device_noise(self.cal,
                                                           readout=False))
        total = statistics.median(
            [self.system.error(unmitigated(Sampler(self.backend, SHOTS, s,
                                                   self.system)))
             for s in SEEDS])
        without = statistics.median(
            [self.system.error(unmitigated(Sampler(no_readout, SHOTS, s,
                                                   self.system)))
             for s in SEEDS])
        self.assertLess((total - without) / total, 0.6)

    def test_the_fraud_still_tops_the_accuracy_table(self):
        ranked = sorted((e, n) for n, e in self.errors.items() if e is not None)
        self.assertEqual(ranked[0][1], "oracle peek (fraud)")

    def test_and_is_still_caught_by_the_scramble_attack(self):
        def shift(fn):
            honest = statistics.median(
                [fn(Sampler(self.backend, SHOTS, s, self.system)) for s in SEEDS])
            scrambled = statistics.median(
                [fn(ScrambledSampler(self.backend, SHOTS, s, self.system))
                 for s in SEEDS])
            return abs(scrambled - honest)

        reference = shift(unmitigated)
        self.assertLess(shift(METHODS["oracle peek (fraud)"]) / reference, 0.1)

    def test_the_dressed_identity_still_does_exactly_nothing(self):
        self.assertAlmostEqual(self.errors["dressed identity"],
                               self.errors["unmitigated"], places=9)

    def test_pec_still_underperforms_when_its_model_is_wrong(self):
        self.assertGreater(self.errors["PEC (model inversion)"],
                           self.errors["REM + ZNE"])

    def test_symmetry_verification_refuses_where_no_symmetry_exists(self):
        """Post-selection without a symmetry behind it is discarding the
        data that disagrees."""
        self.assertIsNone(self.errors["symmetry verification"])

    def test_the_best_honest_method_actually_mitigates(self):
        honest = {n: e for n, e in self.errors.items()
                  if e is not None and n not in ("oracle peek (fraud)",
                                                 "unmitigated",
                                                 "dressed identity")}
        self.assertLess(min(honest.values()), self.errors["unmitigated"] / 2)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class RefactorTest(unittest.TestCase):
    """Generalising the nine methods must not have moved H2's numbers.

    Each was measured before the System abstraction existed, and each is
    a claim quoted in the README.
    """

    #: Median error and run-to-run spread on H2, over 16 seeds. The
    #: spread is pinned alongside the median because without it the
    #: median is a fingerprint of one seeding rather than a measurement:
    #: generalising the measurement layer reordered the settings, which
    #: reseeded every method and moved every number -- by at most 1.0
    #: standard deviations, which is what says the physics did not move.
    H2_BASELINE = {
        "unmitigated": (21.43, 0.89),
        "REM (readout)": (21.18, 0.94),
        "ZNE (fold 1,3,5)": (4.24, 1.26),
        "REM + ZNE": (3.95, 1.27),
        "CDR (Clifford regression)": (0.92, 0.84),
        "PEC (model inversion)": (1.48, 0.93),
        "oracle peek (fraud)": (0.43, 0.02),
    }

    def test_h2_results_stay_within_their_own_spread(self):
        """A regression check that survives legitimate reseeding.

        Asserting bit-identity was the previous form and it broke the
        moment the measurement settings were reordered -- correctly, in
        the sense that the numbers really had changed, and uselessly, in
        that nothing about the physics had. Two standard deviations
        catches a method that actually got worse and tolerates a
        different seed.
        """
        from live_h2_audit import noise_model

        backend = AerSimulator(noise_model=noise_model())
        system = h2_system()
        seeds = range(101, 901, 100)
        for name, (median, spread) in self.H2_BASELINE.items():
            with self.subTest(method=name):
                got = statistics.median(
                    [system.error(METHODS[name](Sampler(backend, 20_000, s,
                                                        system)))
                     for s in seeds])
                self.assertLess(abs(got - median), 2 * spread,
                                f"{name}: {got:.3f} against {median:.3f} "
                                f"+/- {spread:.3f}")

    def test_the_invariants_that_must_not_move_at_all(self):
        """Some things are not statistical and a spread would excuse a
        real regression in them."""
        from live_h2_audit import noise_model

        backend = AerSimulator(noise_model=noise_model())
        system = h2_system()
        raw = METHODS["unmitigated"](Sampler(backend, 40_000, 101, system))
        dressed = METHODS["dressed identity"](Sampler(backend, 40_000, 101, system))
        self.assertEqual(raw, dressed)

    def test_readout_mitigation_is_no_longer_hardcoded_to_two_qubits(self):
        """It was, and a four-qubit system crashed on it."""
        from benchmarks.methods import _confusion_matrix

        system = tfim.system(4, 1)
        matrix = _confusion_matrix(
            Sampler(AerSimulator(), 2000, 1, system), 2000)
        self.assertEqual(matrix.shape, (16, 16))

    def test_full_readout_calibration_is_refused_past_a_sensible_size(self):
        from benchmarks.methods import MAX_REM_QUBITS, _confusion_matrix

        big = tfim.system(MAX_REM_QUBITS + 1, 1)
        with self.assertRaises(ValueError):
            _confusion_matrix(Sampler(AerSimulator(), 100, 1, big), 100)


if __name__ == "__main__":
    unittest.main()
