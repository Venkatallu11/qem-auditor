"""Does the engine work on a circuit nobody involved has seen?

The answer used to be no, silently. These pin that it is now yes, and
that the two methods which still refuse are refusing for reasons rather
than gaps.
"""
import statistics
import unittest

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    HAVE_AER = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_AER = False

if HAVE_AER:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from benchmarks.methods import (METHODS, Sampler, ScrambledSampler,
                                    TargetScrambledSampler, near_clifford_training,
                                    system_from_circuit, unmitigated)

#: Deliberately low. Every assertion here is a large-margin inequality --
#: the fraud scores 0.020 against a floor of 0.5, and a method beating raw
#: by 3x is asked to beat it by 3x -- so shots buy nothing but runtime.
#: The one test that needs precision, agreement with a statevector, asks
#: for its own.
SHOTS = 6_000
SEEDS = (101, 202)


def ansatz():
    """A hardware-efficient ansatz: the shape most people submit, and
    nothing like H2 or an Ising chain."""
    qc = QuantumCircuit(3, name="hardware_efficient")
    for q in range(3):
        qc.ry(0.6 + 0.3 * q, q)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for q in range(3):
        qc.ry(0.4 - 0.2 * q, q)
        qc.rz(0.5 + 0.1 * q, q)
    qc.cx(0, 2)
    qc.cx(0, 1)
    for q in range(3):
        qc.ry(0.25 * q, q)
    return qc


def observable():
    return SparsePauliOp.from_list([
        ("XYZ", 0.5), ("ZZI", 0.3), ("IXX", 0.2), ("ZIZ", 0.4), ("III", 0.1)])


def device():
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx", "cz"])
    noise.add_all_qubit_quantum_error(
        depolarizing_error(0.001, 1), ["h", "t", "ry", "rz", "sdg", "u", "sx", "x"])
    noise.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.03, 0.97]]))
    return noise


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class AnyCircuitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = system_from_circuit(ansatz(), observable())
        cls.backend = AerSimulator(noise_model=device())
        cls.errors, cls.refusals = {}, {}
        for name, method in METHODS.items():
            try:
                cls.errors[name] = statistics.median(
                    [cls.system.error(method(Sampler(cls.backend, SHOTS, s,
                                                     cls.system)))
                     for s in SEEDS])
            except ValueError as refusal:
                cls.refusals[name] = str(refusal)

    def test_a_mixed_basis_observable_is_measured_rather_than_guessed(self):
        """XYZ needs all three bases. The old layer popped one from a set
        and returned a wrong number."""
        noiseless = AerSimulator()
        # This one is a precision claim, so it pays for precision.
        got = unmitigated(Sampler(noiseless, 200_000, 1, self.system))
        self.assertAlmostEqual(got, self.system.exact, delta=0.03)

    def test_most_methods_run_on_a_circuit_they_have_never_seen(self):
        self.assertGreaterEqual(len(self.errors), 8)

    def test_training_circuits_are_generated_rather_than_demanded(self):
        self.assertGreaterEqual(len(self.system.clifford_variants), 2)

    def test_the_dressed_identity_still_does_exactly_nothing(self):
        self.assertAlmostEqual(self.errors["dressed identity"],
                               self.errors["unmitigated"], places=9)

    def test_the_fraud_still_tops_the_accuracy_table(self):
        best = min(self.errors, key=self.errors.get)
        self.assertEqual(best, "oracle peek (fraud)")

    def test_a_real_method_beats_raw_by_a_lot(self):
        honest = {n: e for n, e in self.errors.items()
                  if n not in ("oracle peek (fraud)", "unmitigated",
                               "dressed identity")}
        self.assertLess(min(honest.values()), self.errors["unmitigated"] / 3)

    def test_symmetry_verification_refuses_with_a_reason(self):
        """No error budget reveals whether a state obeys a symmetry.
        Refusing is the correct answer, not a gap."""
        self.assertIn("symmetry verification", self.refusals)
        self.assertIn("symmetry", self.refusals["symmetry verification"])


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class TargetOnlyScramblingTest(unittest.TestCase):
    """Scrambling a method's calibration as well as its experiment let a
    calibrated method absorb the distortion in its own fit.

    Measured on this circuit: CDR's fitted slope flipped from +1.24 to
    -0.32 while its target flipped sign too, the two cancelled, and it
    scored 0.390 -- below the floor, next to the fraud, for reading
    scrambled data twice rather than for not reading it.
    """

    @classmethod
    def setUpClass(cls):
        cls.system = system_from_circuit(ansatz(), observable())
        cls.backend = AerSimulator(noise_model=device())

    def sensitivity(self, method, sampler_class):
        def shift(fn):
            honest = statistics.median(
                [fn(Sampler(self.backend, SHOTS, s, self.system)) for s in SEEDS])
            scrambled = statistics.median(
                [fn(sampler_class(self.backend, SHOTS, s, self.system))
                 for s in SEEDS])
            return abs(scrambled - honest)

        return shift(method) / shift(unmitigated)

    def test_scrambling_everything_wrongly_flags_a_calibrated_method(self):
        self.assertLess(
            self.sensitivity(METHODS["CDR (Clifford regression)"],
                             ScrambledSampler), 0.5)

    def test_scrambling_the_target_only_clears_it(self):
        self.assertGreater(
            self.sensitivity(METHODS["CDR (Clifford regression)"],
                             TargetScrambledSampler), 0.5)

    def test_the_fraud_is_caught_either_way(self):
        for sampler_class in (ScrambledSampler, TargetScrambledSampler):
            with self.subTest(attack=sampler_class.__name__):
                self.assertLess(
                    self.sensitivity(METHODS["oracle peek (fraud)"],
                                     sampler_class), 0.1)

    def test_folded_measurements_are_not_treated_as_calibration(self):
        """A folded copy of the experiment is still the experiment.
        Classifying it as calibration made ZNE look perfectly
        data-independent."""
        self.assertGreater(
            self.sensitivity(METHODS["ZNE (fold 1,3,5)"],
                             TargetScrambledSampler), 0.3)

    def test_every_honest_method_clears_the_floor(self):
        """Checked on the methods that calibrate, which are the ones the
        attack's older form got wrong. Running all nine here duplicated
        what the shootout already covers at four times the cost."""
        for name in ("CDR (Clifford regression)", "REM (readout)", "REM + ZNE"):
            with self.subTest(method=name):
                self.assertGreater(
                    self.sensitivity(METHODS[name], TargetScrambledSampler), 0.5)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class TrainingGenerationTest(unittest.TestCase):
    def test_a_degenerate_training_set_is_reported_not_padded(self):
        """A circuit whose observable does not depend on its parameters
        gives CDR nothing to learn, and it should say so."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.3, 1)
        flat = SparsePauliOp.from_list([("ZZ", 1.0)])
        self.assertLessEqual(len(near_clifford_training(qc, flat)), 1)

    def test_a_parameter_sensitive_circuit_yields_a_spread(self):
        variants = near_clifford_training(ansatz(), observable())
        targets = [t for _, t in variants]
        self.assertGreater(max(targets) - min(targets), 0.1)


if __name__ == "__main__":
    unittest.main()
