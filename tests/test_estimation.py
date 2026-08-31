"""Measuring a general observable, on somebody else's circuit.

The layer this replaces assumed every term was all-Z or all-X and did not
check: it popped a basis from a set and measured a term like XYZ in an
arbitrary one, returning a number that was WRONG rather than absent,
while its docstring said silently averaging would be wrong. These tests
exist because that is the failure mode this package is supposed to catch
in other people's code.
"""
import unittest

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_aer import AerSimulator

    HAVE_AER = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_AER = False

from qem_auditor.estimation import (EstimationError, compatible, expectation,
                                    group_terms, parity, predicate_expectation,
                                    term_bases)


class TermBasisTest(unittest.TestCase):
    def test_labels_are_reversed_into_qubit_order(self):
        """Qiskit labels read qubit n-1 first. Getting this backwards is
        exactly the silent index error this package exists to catch."""
        self.assertEqual(term_bases("XYZ"), ("Z", "Y", "X"))

    def test_a_non_pauli_string_is_refused(self):
        for label in ("XQZ", "", "12"):
            with self.subTest(label=label):
                with self.assertRaises(EstimationError):
                    term_bases(label)

    def test_identity_imposes_nothing(self):
        self.assertTrue(compatible(("I", "I"), ("X", "Z")))

    def test_two_terms_wanting_different_bases_on_a_qubit_are_incompatible(self):
        self.assertFalse(compatible(("X", "I"), ("Z", "I")))

    def test_terms_of_different_widths_are_refused(self):
        with self.assertRaises(EstimationError):
            compatible(("X",), ("X", "Z"))


class GroupingTest(unittest.TestCase):
    def test_compatible_terms_share_a_setting(self):
        settings, assignment = group_terms(["ZZI", "IIZ", "ZII"])
        self.assertEqual(len(settings), 1)
        self.assertEqual(set(assignment.values()), {0})

    def test_incompatible_terms_do_not(self):
        settings, _ = group_terms(["XII", "ZII"])
        self.assertEqual(len(settings), 2)

    def test_the_identity_needs_no_circuit(self):
        _, assignment = group_terms(["III", "ZZZ"])
        self.assertIsNone(assignment["III"])

    def test_every_term_is_assigned_to_a_compatible_setting(self):
        """The property that makes the result correct rather than merely
        small."""
        labels = ["XYZ", "ZZI", "IXX", "ZIZ", "YYY"]
        settings, assignment = group_terms(labels)
        for label in labels:
            with self.subTest(label=label):
                self.assertTrue(
                    compatible(settings[assignment[label]], term_bases(label)))

    def test_grouping_is_deterministic(self):
        labels = ["XYZ", "ZZI", "IXX", "ZIZ"]
        self.assertEqual(group_terms(labels), group_terms(labels))


class ParityTest(unittest.TestCase):
    def test_an_empty_table_is_refused_rather_than_averaged(self):
        with self.assertRaises(EstimationError):
            parity({}, (0,))

    def test_no_qubits_is_the_identity(self):
        self.assertEqual(parity({"00": 10}, ()), 1.0)

    def test_even_parity_is_plus_one(self):
        self.assertAlmostEqual(parity({"11": 100}, (0, 1)), 1.0)

    def test_odd_parity_is_minus_one(self):
        self.assertAlmostEqual(parity({"10": 100}, (0, 1)), -1.0)


@unittest.skipUnless(HAVE_AER, "needs qiskit-aer")
class AgainstAStatevectorTest(unittest.TestCase):
    """The check that matters: does it get the right number?

    Random mixed-basis observables on random circuits, compared against
    an exact statevector. A layer that silently measured terms in the
    wrong basis passes every structural test and fails this one.
    """

    def measure(self, circuit, observable, shots=200_000):
        from qem_auditor.estimation import rotate_into_basis

        backend = AerSimulator()
        settings, assignment = group_terms(observable.paulis.to_labels())
        tables = []
        for index, setting in enumerate(settings):
            rotated = rotate_into_basis(circuit, setting)
            counts = backend.run(
                transpile(rotated, backend, optimization_level=0),
                shots=shots, seed_simulator=100 + index).result().get_counts()
            tables.append({b.replace(" ", "")[::-1]: c for b, c in counts.items()})
        return expectation(tables, observable, assignment)

    def random_case(self, seed):
        import numpy as np

        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 5))
        circuit = QuantumCircuit(n)
        for q in range(n):
            circuit.h(q)
            circuit.rx(float(rng.uniform(0, 3)), q)
        for q in range(n - 1):
            circuit.cx(q, q + 1)
            circuit.rz(float(rng.uniform(0, 3)), q + 1)
        labels = ["".join(rng.choice(list("IXYZ"), size=n)) for _ in range(4)]
        observable = SparsePauliOp.from_list(
            list(zip(labels, rng.uniform(-1, 1, size=len(labels)))))
        return circuit, observable

    def test_random_mixed_basis_observables_match_the_exact_value(self):
        for seed in range(5):
            circuit, observable = self.random_case(seed)
            with self.subTest(seed=seed):
                exact = float(
                    Statevector(circuit).expectation_value(observable).real)
                self.assertAlmostEqual(self.measure(circuit, observable), exact,
                                       delta=0.05)

    def test_a_term_needing_three_bases_is_measured_correctly(self):
        """The exact case the old layer got silently wrong."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.ry(0.7, 1)
        circuit.cx(0, 1)
        circuit.rz(0.4, 2)
        circuit.h(2)
        observable = SparsePauliOp.from_list([("XYZ", 1.0)])
        exact = float(Statevector(circuit).expectation_value(observable).real)
        self.assertAlmostEqual(self.measure(circuit, observable), exact,
                               delta=0.05)


if __name__ == "__main__":
    unittest.main()


class PredicateObservableTest(unittest.TestCase):
    """The observable an oracle actually has.

    "Did we land in the marked set" is diagonal, so one Z-basis setting
    measures it exactly -- but the 1097-state marked set that motivated
    this has no useful Pauli expansion, and `expectation` would have
    estimated thousands of terms to compute a fraction.
    """

    def test_it_counts_the_shots_the_predicate_accepts(self):
        counts = {"000": 40, "001": 30, "111": 30}
        self.assertAlmostEqual(
            predicate_expectation(counts, lambda v: v >= 1), 0.6)

    def test_the_default_decoding_is_little_endian_qubit_order(self):
        """Position i of a normalised bitstring is qubit i, so '001'
        means qubit 2 is set and the value is 4, not 1. Getting this
        backwards is silent and is why it is pinned."""
        self.assertAlmostEqual(
            predicate_expectation({"001": 10}, lambda v: v == 4), 1.0)
        self.assertAlmostEqual(
            predicate_expectation({"001": 10}, lambda v: v == 1), 0.0)

    def test_a_custom_decoding_is_used_when_given(self):
        counts = {"01": 30, "11": 70}
        pairs = predicate_expectation(
            counts, lambda xy: xy[0] == 1, decode=lambda b: (int(b[0]), int(b[1])))
        self.assertAlmostEqual(pairs, 0.7)  # only "11" decodes to a first entry of 1

    def test_negative_weights_from_readout_correction_are_kept(self):
        """Clipping them here would bias the result back toward the
        unmitigated value, making mitigation look like it did less."""
        counts = {"0": 110.0, "1": -10.0}
        self.assertAlmostEqual(predicate_expectation(counts, lambda v: v == 0), 1.1)

    def test_no_shots_is_refused_rather_than_returned_as_zero(self):
        with self.assertRaises(EstimationError):
            predicate_expectation({}, lambda v: True)

    def test_it_agrees_with_the_pauli_estimator_where_both_apply(self):
        """A single-qubit projector onto |1> is (I - Z)/2, so the two
        routes must give the same number. If they ever disagree, one of
        the two bit orderings is wrong."""
        counts = {"1": 700, "0": 300}
        by_predicate = predicate_expectation(counts, lambda v: v == 1)
        by_pauli = (1.0 - parity(counts, [0])) / 2
        self.assertAlmostEqual(by_predicate, by_pauli)
