"""The correctness gate that runs before any noise question.

The claim under test here is not "this mitigation helped" but "this
circuit computes what its author said". Every mitigation claim
presupposes it, and until the first outside circuit arrived nothing
checked it.
"""
import unittest

from qem_auditor.reversible import (NotReversible, audit_oracle, evaluate)


def marks(value_bits, sign=-1):
    """A tiny 2-qubit oracle marking |11>, written as (name, qubits)."""
    return [("ccz", [0, 1])] if value_bits else []


class EvaluateTest(unittest.TestCase):

    def test_negation_and_control(self):
        self.assertEqual(evaluate([("x", [0])], [0, 0]), ((1, 0), 1))
        self.assertEqual(evaluate([("cx", [0, 1])], [1, 0]), ((1, 1), 1))
        self.assertEqual(evaluate([("cx", [0, 1])], [0, 0]), ((0, 0), 1))
        self.assertEqual(evaluate([("ccx", [0, 1, 2])], [1, 1, 0]), ((1, 1, 1), 1))

    def test_phase_is_tracked_separately_from_bits(self):
        bits, sign = evaluate([("cz", [0, 1])], [1, 1])
        self.assertEqual(bits, (1, 1))
        self.assertEqual(sign, -1)
        self.assertEqual(evaluate([("cz", [0, 1])], [1, 0])[1], 1)

    def test_multi_controlled_forms_are_matched_by_shape(self):
        # c12z and mcx arrive from QASM without being enumerated anywhere.
        self.assertEqual(evaluate([("c12z", list(range(13)))], [1] * 13)[1], -1)
        self.assertEqual(evaluate([("mcx", [0, 1, 2, 3])], [1, 1, 1, 0])[0][3], 1)

    def test_a_non_permutation_gate_is_refused_by_name(self):
        """A Hadamard is not a bug in the circuit -- it means this check
        does not apply, which is worth saying rather than ignoring."""
        with self.assertRaises(NotReversible) as caught:
            evaluate([("h", [0])], [0])
        self.assertIn("h", str(caught.exception))


class AuditOracleTest(unittest.TestCase):

    def two_bit_oracle(self, marked):
        """Phase on exactly the listed 2-bit values, ancilla-free."""
        program = []
        for value in marked:
            zeros = [q for q in (0, 1) if not (value >> q) & 1]
            program += [("x", [q]) for q in zeros]
            program += [("cz", [0, 1])]
            program += [("x", [q]) for q in zeros]
        return program

    def encode(self, value):
        return [(value >> 0) & 1, (value >> 1) & 1]

    def test_a_correct_oracle_matches_its_specification(self):
        report = audit_oracle(self.two_bit_oracle([1, 2]),
                              predicate=lambda v: v in (1, 2),
                              n_inputs=4, encode=self.encode)
        self.assertTrue(report.matches_specification)
        self.assertEqual(report.marked, frozenset({1, 2}))
        self.assertEqual(report.discrepancies, ())

    def test_a_wrong_oracle_is_reported_in_both_directions(self):
        report = audit_oracle(self.two_bit_oracle([1, 3]),
                              predicate=lambda v: v in (1, 2),
                              n_inputs=4, encode=self.encode)
        self.assertFalse(report.matches_specification)
        self.assertEqual(report.false_negatives, frozenset({2}))
        self.assertEqual(report.false_positives, frozenset({3}))
        kinds = [d.kind for d in report.discrepancies]
        self.assertIn("specified but unmarked", kinds)
        self.assertIn("marked but not specified", kinds)

    def test_dirty_ancillas_are_caught_even_when_the_marking_is_right(self):
        """The uploaded oracle's real defect. The phases can be perfect
        and the circuit still useless inside amplitude amplification."""
        program = [("cx", [0, 2])] + self.two_bit_oracle([1, 2])
        report = audit_oracle(program, predicate=lambda v: v in (1, 2),
                              n_inputs=4, encode=lambda v: self.encode(v) + [0],
                              ancillas=[2])
        self.assertEqual(report.marked, frozenset({1, 2}))
        self.assertFalse(report.is_a_phase_oracle)
        self.assertFalse(report.matches_specification)
        self.assertEqual(report.dirty_ancillas, 2)
        self.assertIn("ancillas not restored",
                      [d.kind for d in report.discrepancies])

    def test_an_altered_input_register_is_caught(self):
        report = audit_oracle([("x", [0])], predicate=lambda v: False,
                              n_inputs=4, encode=self.encode)
        self.assertEqual(report.altered_inputs, 4)
        self.assertFalse(report.is_a_phase_oracle)

    def test_accuracy_flatters_a_sparse_specification(self):
        """Stated as a test because it is the reason the report never
        quotes accuracy without the counts beside it: a circuit that does
        nothing scores 75% against a specification marking one of four."""
        report = audit_oracle([], predicate=lambda v: v == 3,
                              n_inputs=4, encode=self.encode)
        self.assertEqual(report.accuracy, 0.75)
        self.assertFalse(report.matches_specification)

    def test_the_verdict_is_a_proof_over_the_whole_input_space(self):
        report = audit_oracle(self.two_bit_oracle([1, 2]),
                              predicate=lambda v: v in (1, 2),
                              n_inputs=4, encode=self.encode)
        self.assertEqual(report.n_inputs, 4)
