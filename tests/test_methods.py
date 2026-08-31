"""Nine mitigation methods, and what the auditor is supposed to notice.

These pin the qualitative findings of examples/method_shootout.py. They
use fewer seeds than the example, because what is being asserted is which
way the comparisons go, not the third decimal place.
"""
import statistics
import unittest
from pathlib import Path

try:
    from qiskit_aer import AerSimulator

    HAVE_AER = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_AER = False

if HAVE_AER:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from benchmarks import methods as M
    from qem_auditor.estimation import group_terms
    from live_h2_audit import noise_model as invented_noise
    from real_device_audit import calibration, device_noise

SEEDS = (101, 202, 303)
SHOTS = 20_000


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class SetUpMixin(unittest.TestCase):
    @classmethod
    def backends(cls):
        return (AerSimulator(noise_model=invented_noise()),
                AerSimulator(noise_model=device_noise(calibration())))

    @staticmethod
    def median_error(method, backend):
        return statistics.median(
            [M.error_kcal(method(M.Sampler(backend, SHOTS, s))) for s in SEEDS])


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class ScrambleAttackTest(SetUpMixin):
    """The only thing separating a great method from one that peeks."""

    @classmethod
    def setUpClass(cls):
        cls.backend = AerSimulator(noise_model=device_noise(calibration()))
        cls.reference = M.scramble_shift(M.unmitigated, cls.backend, SHOTS, SEEDS)
        cls.sensitivity = {
            name: M.data_sensitivity(fn, cls.backend, SHOTS, SEEDS, cls.reference)
            for name, fn in M.METHODS.items()}

    def test_the_fraud_barely_moves_when_the_data_is_destroyed(self):
        self.assertLess(self.sensitivity["oracle peek (fraud)"], 0.1)

    def test_every_honest_method_moves_about_as_much_as_the_raw_estimate(self):
        for name, ratio in self.sensitivity.items():
            if name == "oracle peek (fraud)":
                continue
            with self.subTest(method=name):
                self.assertGreater(ratio, 0.5)

    def test_the_separation_is_not_a_tuned_threshold(self):
        """Honest methods cluster near 1 and the fraud sits near 0. The
        bar is placed in an empty gap, not fitted to either side."""
        honest = [r for n, r in self.sensitivity.items() if n != "oracle peek (fraud)"]
        self.assertGreater(min(honest), 5 * self.sensitivity["oracle peek (fraud)"])

    def test_accuracy_alone_would_crown_the_fraud(self):
        """Which is the whole argument for auditing rather than ranking."""
        errors = {name: self.median_error(fn, self.backend)
                  for name, fn in M.METHODS.items()}
        self.assertEqual(min(errors, key=errors.get), "oracle peek (fraud)")


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class DressedIdentityTest(SetUpMixin):
    def test_it_returns_exactly_the_unmitigated_value(self):
        backend = AerSimulator(noise_model=device_noise(calibration()))
        sampler_a = M.Sampler(backend, SHOTS, 101)
        sampler_b = M.Sampler(backend, SHOTS, 101)
        self.assertEqual(M.dressed_identity(sampler_a), M.unmitigated(sampler_b))

    def test_it_costs_as_much_as_a_real_method(self):
        """It is not a strawman: it runs every circuit ZNE runs."""
        backend = AerSimulator(noise_model=invented_noise())
        dressed = M.Sampler(backend, SHOTS, 101)
        M.dressed_identity(dressed)
        plain = M.Sampler(backend, SHOTS, 101)
        M.unmitigated(plain)
        self.assertGreater(dressed.circuits_run, plain.circuits_run)

    def test_it_passes_the_scramble_attack_that_catches_the_fraud(self):
        """Two different frauds need two different detectors. This one
        genuinely reads the data -- it just does nothing with it, which
        is what the improvement gate is for."""
        backend = AerSimulator(noise_model=device_noise(calibration()))
        ratio = M.data_sensitivity(M.dressed_identity, backend, SHOTS, SEEDS)
        self.assertGreater(ratio, 0.5)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class NoiseModelDependenceTest(SetUpMixin):
    """The finding that matters most: ranking methods on one noise model
    ranks nothing."""

    @classmethod
    def setUpClass(cls):
        cls.invented, cls.measured = cls.backends()

    def test_zne_wins_on_the_invented_noise_and_loses_on_the_measured_one(self):
        zne_invented = self.median_error(M.zne, self.invented)
        rem_invented = self.median_error(M.readout_mitigation, self.invented)
        self.assertLess(zne_invented, rem_invented)

        zne_measured = self.median_error(M.zne, self.measured)
        rem_measured = self.median_error(M.readout_mitigation, self.measured)
        self.assertLess(rem_measured, zne_measured)

    def test_readout_mitigation_does_nothing_where_there_is_no_readout_error(self):
        raw = self.median_error(M.unmitigated, self.invented)
        rem = self.median_error(M.readout_mitigation, self.invented)
        self.assertGreater(rem / raw, 0.9)

    def test_the_composition_beats_either_half_on_the_measured_noise(self):
        both = self.median_error(M.rem_then_zne, self.measured)
        self.assertLess(both, self.median_error(M.zne, self.measured))
        self.assertLess(both, self.median_error(M.readout_mitigation, self.measured))

    def test_cdr_is_the_one_method_that_barely_moves_between_them(self):
        """It learns the noise map instead of assuming its structure."""
        a = self.median_error(M.cdr, self.invented)
        b = self.median_error(M.cdr, self.measured)
        self.assertLess(max(a, b) / min(a, b), 2.0)

    def test_pec_collapses_when_its_assumed_model_is_wrong(self):
        """The case CALIBRATION_MISMATCH exists for, triggered honestly."""
        a = self.median_error(M.pec_model_inversion, self.invented)
        b = self.median_error(M.pec_model_inversion, self.measured)
        self.assertGreater(b / a, 3.0)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class SymmetryVerificationTest(SetUpMixin):
    def test_it_discards_shots_outside_the_physical_subspace(self):
        backend = AerSimulator(noise_model=device_noise(calibration()))
        system = M.h2_system()
        tables = M.basis_counts(M.ansatz(), backend, SHOTS, 101,
                                system.observable)
        settings, _ = group_terms(system.observable.paulis.to_labels())
        z_tables = [t for setting, t in zip(settings, tables)
                    if all(b in ("I", "Z") for b in setting)]
        self.assertTrue(z_tables, "no setting is measured in the Z basis")
        outside = sum(n for table in z_tables for b, n in table.items()
                      if b not in M.PHYSICAL_Z_STRINGS)
        self.assertGreater(outside, 0, "no errors to post-select away")

    def test_the_noiseless_state_never_leaves_that_subspace(self):
        """Which is what makes the post-selection sound rather than a
        convenient filter."""
        system = M.h2_system()
        tables = M.basis_counts(M.ansatz(), AerSimulator(), SHOTS, 101,
                                system.observable)
        settings, _ = group_terms(system.observable.paulis.to_labels())
        for setting, table in zip(settings, tables):
            if all(b in ("I", "Z") for b in setting):
                self.assertEqual(set(table) - set(M.PHYSICAL_Z_STRINGS), set())


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class LayeringTest(unittest.TestCase):
    """The auditor does not import the thing it audits.

    Checked on the import graph rather than on the file text: `qem_auditor`
    discusses benchmarks in its prose constantly, and a grep would confuse
    talking about them with depending on them.
    """

    def test_the_auditor_imports_nothing_from_the_benchmark_side(self):
        import ast

        import qem_auditor

        forbidden = {"benchmarks", "tests", "examples"}
        for path in Path(qem_auditor.__file__).parent.rglob("*.py"):
            tree = ast.parse(path.read_text())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    imported.add(node.module.split(".")[0])
            with self.subTest(module=path.name):
                self.assertEqual(imported & forbidden, set())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class TensoredReadoutMitigationTest(unittest.TestCase):
    """Readout mitigation that survives past six qubits.

    Full calibration needs 2**n preparation circuits, so it refuses at
    seven. The 18-qubit oracle that motivated this would have needed
    262,144 of them; the tensored form needs two, at any width. The
    price is an assumption -- that readout errors factorise -- and these
    tests pin both halves: that it tracks full REM where both run, and
    that it is registered as its own method rather than as full REM.
    """

    def backend(self):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(0.002, 2), ["cx"])
        noise.add_all_qubit_readout_error(ReadoutError([[0.94, 0.06], [0.08, 0.92]]))
        return AerSimulator(noise_model=noise)

    def test_it_recovers_most_of_what_full_rem_recovers(self):
        backend = self.backend()
        system = M.h2_system()
        raw, full, tensored = [], [], []
        for seed in (11, 22, 33):
            raw.append(abs(M.unmitigated(M.Sampler(backend, 20_000, seed)) - system.exact))
            full.append(abs(M.readout_mitigation(M.Sampler(backend, 20_000, seed))
                            - system.exact))
            tensored.append(abs(M.tensored_readout_mitigation(
                M.Sampler(backend, 20_000, seed)) - system.exact))
        median = lambda xs: sorted(xs)[len(xs) // 2]
        self.assertLess(median(tensored), median(raw) / 2,
                        "tensored REM should recover most of a readout-dominated error")
        # It discards correlations, so it is allowed to be worse than full
        # REM -- but not by a factor that would make it a different answer.
        self.assertLess(median(tensored), median(full) * 3)

    def test_it_is_registered_under_its_own_name(self):
        """Presenting a factorised estimator as full REM would be exactly
        the kind of unstated assumption this package audits for."""
        self.assertIn("REM (tensored)", M.METHODS)
        self.assertIsNot(M.METHODS["REM (tensored)"], M.METHODS["REM (readout)"])

    def test_full_rem_still_refuses_the_width_it_cannot_afford(self):
        with self.assertRaises(ValueError):
            M._confusion_matrix(_WideSampler(M.MAX_REM_QUBITS + 1), shots=10)

    def test_the_tensored_ceiling_is_set_by_the_dense_vector_not_the_circuits(self):
        with self.assertRaises(ValueError) as caught:
            M._tensored_confusion(_WideSampler(M.MAX_TENSORED_QUBITS + 1), shots=10)
        self.assertIn("sparse", str(caught.exception))
        self.assertGreater(M.MAX_TENSORED_QUBITS, M.MAX_REM_QUBITS)


class _WideSampler:
    """Just wide enough to trip a width guard, with nothing behind it.

    Only ever constructed inside a test that already requires qiskit.
    """

    def __init__(self, n_qubits):
        from qiskit import QuantumCircuit
        self.circuit = QuantumCircuit(n_qubits)
