"""Every mitigation method in the catalogue, and what the auditor is
supposed to notice about them.

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
        """None when the method refuses on every seed.

        A refusing method has no error to compare, and substituting one
        would rank it against methods that actually ran.
        """
        values = [M.attempt(method, M.Sampler(backend, SHOTS, s)) for s in SEEDS]
        errors = [M.error_kcal(v) for v in values if v is not None]
        return statistics.median(errors) if errors else None


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

    def test_a_method_that_refuses_reports_no_sensitivity_rather_than_a_number(self):
        """A refusal is a legitimate outcome, not a failed attack. There
        is no answer to test for data-dependence, and reporting a number
        would invent one. This arrived with the first refusing method and
        broke this class's setUpClass, which had assumed none ever would."""
        refused = [n for n, r in self.sensitivity.items() if r is None]
        for name in refused:
            with self.subTest(method=name):
                self.assertIsNone(self.sensitivity[name])

    def test_every_honest_method_moves_about_as_much_as_the_raw_estimate(self):
        for name, ratio in self.sensitivity.items():
            if name == "oracle peek (fraud)" or ratio is None:
                continue
            with self.subTest(method=name):
                self.assertGreater(ratio, 0.5)

    def test_the_separation_is_not_a_tuned_threshold(self):
        """Honest methods cluster near 1 and the fraud sits near 0. The
        bar is placed in an empty gap, not fitted to either side.

        Refusing methods are absent rather than scored: they produced no
        answer, so there is nothing to test for data-dependence.
        """
        honest = [r for n, r in self.sensitivity.items()
                  if n != "oracle peek (fraud)" and r is not None]
        self.assertGreater(min(honest), 5 * self.sensitivity["oracle peek (fraud)"])

    def test_accuracy_alone_would_crown_the_fraud(self):
        """Which is the whole argument for auditing rather than ranking."""
        errors = {name: self.median_error(fn, self.backend)
                  for name, fn in M.METHODS.items()}
        ran = {name: error for name, error in errors.items() if error is not None}
        self.assertEqual(min(ran, key=ran.get), "oracle peek (fraud)")
        self.assertGreater(len(ran), 10, "most methods should still run")


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


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class TensoredIsNotWorseThanFullRemTest(unittest.TestCase):
    """What the runs support about the cheap method, and no more.

    An earlier README draft quoted one seed -- 4.92 against 4.76 -- which
    reads as "tensored is slightly worse". Over six seeds the means are
    0.05 apart against a spread of 1.69, which supports "not measurably
    different", not "a little worse". Pinned because the tempting claim
    is the one that overstates.
    """

    def test_the_two_are_not_distinguishable_on_this_device(self):
        from qem_auditor.power import compare
        backend = AerSimulator(noise_model=device_noise(calibration()))
        seeds = (101, 202, 303, 404)
        full = [abs(M.readout_mitigation(M.Sampler(backend, 20_000, s)) - M.FCI)
                for s in seeds]
        tensored = [abs(M.tensored_readout_mitigation(M.Sampler(backend, 20_000, s))
                        - M.FCI) for s in seeds]
        self.assertFalse(compare("full", full, "tensored", tensored).distinguishable)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class TwirlingIsExactTest(unittest.TestCase):
    """A twirl must leave the ideal circuit alone. If it does not, the
    method is silently running a different experiment."""

    def setUp(self):
        try:
            import qiskit  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("needs qiskit")

    def test_every_conjugation_entry_is_right_up_to_a_sign(self):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, Pauli
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        cx = Operator(circuit).data
        for (a, b), (c, d) in M._CX_CONJUGATION.items():
            got = cx @ Pauli(b + a).to_matrix() @ cx.conj().T
            claimed = Pauli(d + c).to_matrix()
            self.assertTrue(
                np.allclose(got, claimed) or np.allclose(got, -claimed),
                f"CX ({a}x{b}) CX+ is neither +{c}x{d} nor -{c}x{d}")

    def test_exactly_two_entries_carry_a_minus_sign(self):
        """Pinned so the sign story in the comment stays true. The sign is
        unobservable -- it makes the net operation -CX, a global phase --
        but a reader deserves to know which entries carry it."""
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, Pauli
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        cx = Operator(circuit).data
        signed = {key for key, (c, d) in M._CX_CONJUGATION.items()
                  if np.allclose(cx @ Pauli(key[1] + key[0]).to_matrix() @ cx.conj().T,
                                 -Pauli(d + c).to_matrix())}
        self.assertEqual(signed, {("X", "Z"), ("Y", "Y")})

    def test_a_twirled_circuit_equals_the_original_up_to_global_phase(self):
        """The property that actually matters, asserted directly rather
        than inferred from the table."""
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        rng = np.random.default_rng(0)
        base = QuantumCircuit(3)
        base.h(0)
        base.cx(0, 1)
        base.rx(0.7, 2)
        base.cx(1, 2)
        base.cx(0, 2)
        target = Operator(base).data
        for _ in range(25):
            twirled = Operator(M.twirled_copy(base, rng)).data
            phase = np.vdot(target.reshape(-1), twirled.reshape(-1))
            phase = phase / abs(phase) if abs(phase) else 1.0
            self.assertTrue(np.allclose(twirled / phase, target))

    def test_twirling_preserves_the_measurement_structure(self):
        from qiskit import QuantumCircuit
        import numpy as np
        base = QuantumCircuit(2, 2)
        base.h(0)
        base.cx(0, 1)
        base.measure([0, 1], [0, 1])
        twirled = M.twirled_copy(base, np.random.default_rng(1))
        self.assertEqual(twirled.num_clbits, 2)
        self.assertEqual(twirled.count_ops().get("measure"), 2)
        self.assertEqual(twirled.count_ops().get("cx"), 1)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class ExponentialExtrapolationTest(unittest.TestCase):
    """It refuses when the data is not an exponential decay, which on the
    measured device noise model is what actually happens."""

    def test_a_clean_decay_extrapolates_to_the_right_place(self):
        # y = 1 + 2 * 0.5**x at x = 1, 3, 5
        values = [1 + 2 * 0.5 ** x for x in (1, 3, 5)]
        self.assertAlmostEqual(M._exponential_to_zero([1, 3, 5], values), 3.0)

    def test_non_monotonic_values_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            M._exponential_to_zero([1, 3, 5], [1.0, 0.5, 0.9])
        self.assertIn("not monotonic", str(caught.exception))

    def test_growth_rather_than_decay_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            M._exponential_to_zero([1, 3, 5], [1.0, 2.0, 4.0])
        self.assertIn("not a decay", str(caught.exception))

    def test_unequal_spacing_is_refused(self):
        with self.assertRaises(ValueError):
            M._exponential_to_zero([1, 2, 5], [1.0, 0.5, 0.25])


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class CatalogueTest(unittest.TestCase):

    def test_every_method_is_registered_once(self):
        self.assertEqual(len(M.METHODS), len(set(M.METHODS)))
        self.assertGreaterEqual(len(M.METHODS), 15)

    def test_the_new_fitting_methods_owe_a_held_out_check(self):
        """A method that fits something has data to hold out, so it must
        be in FITTING_METHODS or it escapes the check silently."""
        for name in ("ZNE (exponential)", "ZNE (Richardson)", "vnCDR"):
            self.assertIn(name, M.FITTING_METHODS, name)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class HeldOutValidationTest(unittest.TestCase):
    """A held-out check is only worth anything if it validates the model
    the method actually uses.

    For a while every non-CDR method here was validated by fitting a
    straight line through the folds, which meant the exponential and
    Richardson extrapolators were being asked a question about a model
    neither of them uses -- a pass that certified nothing and a failure
    that would have blamed the wrong assumption.
    """

    @classmethod
    def setUpClass(cls):
        cls.backend = AerSimulator(noise_model=invented_noise())

    def factory(self):
        return M.Sampler(self.backend, SHOTS, 101)

    def test_the_exponential_extrapolator_is_validated_by_an_exponential(self):
        seen = []
        original = M._exponential_fit

        def spy(scales, values):
            seen.append(list(scales))
            return original(scales, values)

        M._exponential_fit = spy
        try:
            M.heldout_ok("ZNE (exponential)", self.factory, 5.0)
        finally:
            M._exponential_fit = original
        # Fitted on the three upper folds and asked about the withheld
        # lowest one: an extrapolation, in the direction production uses.
        self.assertEqual(seen, [[3, 5, 7]])

    def test_richardson_keeps_a_degree_of_freedom_in_the_held_out_fit(self):
        seen = {}
        original = M._folded_energies

        def spy(sampler, folds, process, **kwargs):
            seen["folds"] = list(folds)
            return original(sampler, folds, process, **kwargs)

        M._folded_energies = spy
        try:
            M.heldout_ok("ZNE (Richardson)", self.factory, 5.0)
        finally:
            M._folded_energies = original
        # Five measured, four fitted with a quadratic, one withheld. A
        # quadratic through three points is interpolation wearing a
        # fit's clothes, which is the thing this method exists to avoid.
        self.assertEqual(seen["folds"], [1, 3, 5, 7, 9])

    def test_vncdr_is_validated_as_a_regression_not_an_extrapolation(self):
        calls = []
        original = M._heldout_extrapolation

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        M._heldout_extrapolation = spy
        try:
            M.heldout_ok("vnCDR", self.factory, 5.0)
        finally:
            M._heldout_extrapolation = original
        self.assertEqual(calls, [], "vnCDR fits a regression per fold; "
                                    "validating it as a plain extrapolator "
                                    "tests a method nobody proposed")

    def test_a_refusal_on_the_held_out_data_is_not_a_pass(self):
        """The tolerance is absurdly generous, so only a refusal can
        make this False -- and a refusal must, because an unrun control
        is not a passed one."""
        original = M._exponential_fit

        def refuse(scales, values):
            raise ValueError("the model does not describe this data")

        M._exponential_fit = refuse
        try:
            self.assertFalse(
                M.heldout_ok("ZNE (exponential)", self.factory, 1e9))
        finally:
            M._exponential_fit = original


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class ExponentialFitTest(unittest.TestCase):

    def test_the_fit_can_be_evaluated_away_from_zero(self):
        """Production wants the curve at zero; the held-out check wants
        it at a fold it withheld. One fit, evaluated twice."""
        offset, amplitude, ratio = M._exponential_fit(
            [3, 5, 7], [1 + 2 * 0.5 ** x for x in (3, 5, 7)])
        self.assertAlmostEqual(offset, 1.0, places=6)
        self.assertAlmostEqual(ratio, 0.5, places=6)
        self.assertAlmostEqual(offset + amplitude * ratio ** 1, 2.0, places=6)
        self.assertAlmostEqual(offset + amplitude, 3.0, places=6)
