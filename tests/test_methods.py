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
        tables = M.basis_counts(M.ansatz(), backend, SHOTS, 101)
        outside = sum(n for b, n in tables["Z"].items()
                      if b not in M.PHYSICAL_Z_STRINGS)
        self.assertGreater(outside, 0, "no errors to post-select away")

    def test_the_noiseless_state_never_leaves_that_subspace(self):
        """Which is what makes the post-selection sound rather than a
        convenient filter."""
        tables = M.basis_counts(M.ansatz(), AerSimulator(), SHOTS, 101)
        self.assertEqual(
            {b for b in tables["Z"]} - set(M.PHYSICAL_Z_STRINGS), set())


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
