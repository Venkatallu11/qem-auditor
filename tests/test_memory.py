"""What this circuit reminds the auditor of.

The properties that matter: that structure is the key rather than names,
that memory reorders and warns without ever concluding, and that a thin
memory says so instead of acting confident.
"""
import json
import tempfile
import unittest
from pathlib import Path

from qem_auditor import FailureMode, Verdict
from qem_auditor.memory import (CaseMemory, CircuitFingerprint, PastCase,
                                case_from_audit, fingerprint_from_spec)
from qem_auditor.schema import CircuitSpec


def ucc(depth=8, family="ucc"):
    return CircuitFingerprint(n_qubits=2, two_qubit_gates=2, one_qubit_gates=6,
                              depth=depth, gate_names=("x", "rx", "rz", "h", "cx"),
                              observable_terms=5, measurement_bases=2, family=family)


def qaoa():
    return CircuitFingerprint(n_qubits=12, two_qubit_gates=80, one_qubit_gates=200,
                              depth=140, gate_names=("ecr", "sx", "rz"),
                              observable_terms=300, measurement_bases=4,
                              family="qaoa")


def case(name, fingerprint=None, verdict=Verdict.INVALID,
         failed=("unitary_equivalence",), attacks=("T_compiler",)):
    return PastCase(name, fingerprint or ucc(), verdict, failed, (), (), attacks)


class FingerprintTest(unittest.TestCase):
    def test_structure_is_the_key_not_the_name(self):
        """Two groups call the same ansatz different things, and one
        group calls two circuits the same across a refactor."""
        self.assertEqual(ucc(family="theirs").digest, ucc(family="ours").digest)

    def test_a_circuit_on_no_qubits_cannot_be_fingerprinted(self):
        with self.assertRaises(ValueError):
            CircuitFingerprint(n_qubits=0, two_qubit_gates=1,
                               one_qubit_gates=1, depth=1)

    def test_negative_counts_are_refused(self):
        with self.assertRaises(ValueError):
            CircuitFingerprint(n_qubits=2, two_qubit_gates=-1,
                               one_qubit_gates=1, depth=1)

    def test_gate_order_does_not_change_the_fingerprint(self):
        a = CircuitFingerprint(2, 2, 6, 8, ("cx", "h", "x"))
        b = CircuitFingerprint(2, 2, 6, 8, ("x", "h", "cx"))
        self.assertEqual(a.digest, b.digest)

    def test_identical_circuits_are_identical(self):
        self.assertEqual(ucc().resembles(ucc()), 1.0)

    def test_unrelated_circuits_score_low(self):
        self.assertLess(ucc().resembles(qaoa()), 0.2)

    def test_a_deeper_version_is_similar_but_not_identical(self):
        """Reporting circuits of different depth as 100% alike is an
        over-claim, and the family bonus used to produce exactly that."""
        score = ucc(8).resembles(ucc(9))
        self.assertGreater(score, 0.9)
        self.assertLess(score, 1.0)

    def test_similarity_falls_as_the_circuits_diverge(self):
        self.assertGreater(ucc(8).resembles(ucc(9)), ucc(8).resembles(ucc(40)))

    def test_it_is_symmetric(self):
        self.assertAlmostEqual(ucc(8).resembles(ucc(20)), ucc(20).resembles(ucc(8)))

    def test_a_different_declared_family_reduces_similarity(self):
        self.assertLess(ucc(family="qaoa").resembles(ucc(family="ucc")),
                        ucc(family="ucc").resembles(ucc(family="ucc")))

    def test_a_fingerprint_can_be_read_off_a_record(self):
        spec = CircuitSpec(circuit_id="h2", native_gate_set="x, rx, cx",
                           n_qubits=2, n_1q_gates=6, n_2q_gates=2)
        self.assertEqual(fingerprint_from_spec(spec).two_qubit_gates, 2)
        self.assertIn("cx", fingerprint_from_spec(spec).gate_names)


class RecallTest(unittest.TestCase):
    def setUp(self):
        self.memory = CaseMemory()
        self.memory.remember(case("run_a"))
        self.memory.remember(case("run_b", ucc(9),
                                  failed=("unitary_equivalence", "ideal_control")))
        self.memory.remember(case("run_c", ucc(7), Verdict.NOT_ESTABLISHED,
                                  failed=(), attacks=()))
        self.memory.remember(case("qaoa_big", qaoa(), Verdict.PROMISING,
                                  failed=(), attacks=()))

    def test_an_exact_structural_match_is_reported_as_seen_before(self):
        recalled = self.memory.recall(ucc())
        self.assertEqual([c.experiment_id for c in recalled.seen_before], ["run_a"])

    def test_similar_circuits_are_found_and_the_unrelated_one_is_not(self):
        found = {c.experiment_id for c, _ in self.memory.recall(ucc()).resembling}
        self.assertEqual(found, {"run_b", "run_c"})

    def test_the_checks_that_failed_most_often_come_first(self):
        first = self.memory.recall(ucc()).check_first[0]
        self.assertEqual(first[0], "unitary_equivalence")
        self.assertEqual(first[1], 2)

    def test_the_attacks_that_earned_their_keep_come_first(self):
        self.assertEqual(self.memory.recall(ucc()).attacks_first[0][0], "T_compiler")

    def test_an_unfamiliar_circuit_recalls_nothing_and_says_so(self):
        recalled = CaseMemory().recall(ucc())
        self.assertTrue(recalled.is_empty)
        self.assertIn("first time", recalled.format_recollection())

    def test_a_thin_memory_declines_to_be_relied_on(self):
        memory = CaseMemory()
        memory.remember(case("only_one"))
        recalled = memory.recall(ucc(9))
        self.assertFalse(recalled.worth_relying_on)
        self.assertIn("Too few", recalled.format_recollection())

    def test_enough_cases_earn_reliance(self):
        for i in range(4):
            self.memory.remember(case(f"extra_{i}", ucc(8 + i)))
        self.assertTrue(self.memory.recall(ucc()).worth_relying_on)

    def test_the_ordering_is_stable_across_calls(self):
        """An ordering that shuffled would send a user to a different
        check each time for no reason."""
        first = self.memory.recall(ucc()).check_first
        self.assertEqual(first, self.memory.recall(ucc()).check_first)


class MemoryAdvisesButNeverConvictsTest(unittest.TestCase):
    """The rule that keeps this an auditor rather than a reputation
    system: a method that failed once must still be able to be shown
    working."""

    def test_recall_returns_no_verdict_of_its_own(self):
        memory = CaseMemory()
        for i in range(5):
            memory.remember(case(f"bad_{i}", ucc(8 + i), Verdict.INVALID))
        recalled = memory.recall(ucc())
        self.assertFalse(hasattr(recalled, "verdict"))

    def test_the_report_says_the_gates_still_decide(self):
        memory = CaseMemory()
        memory.remember(case("bad"))
        self.assertIn("gates still decide",
                      memory.recall(ucc()).format_recollection())

    def test_a_clean_audit_of_a_previously_failing_circuit_is_unaffected(self):
        """Memory is not consulted by the gates at all, which is what
        makes this true by construction rather than by care."""
        from qem_auditor import audit

        from .helpers import make_experiment

        memory = CaseMemory()
        for i in range(5):
            memory.remember(case(f"bad_{i}", ucc(8 + i), Verdict.INVALID))
        self.assertIs(audit(make_experiment()).verdict, Verdict.CERTIFIED_UNDER_SCOPE)


class DuplicateAndPersistenceTest(unittest.TestCase):
    def test_the_same_case_remembered_twice_is_one_case(self):
        memory = CaseMemory()
        self.assertTrue(memory.remember(case("a")))
        self.assertFalse(memory.remember(case("a")))
        self.assertEqual(len(memory), 1)

    def test_a_different_outcome_on_the_same_circuit_is_new(self):
        memory = CaseMemory()
        memory.remember(case("a", verdict=Verdict.INVALID))
        self.assertTrue(memory.remember(case("a", verdict=Verdict.REFUTED)))

    def test_a_round_trip_preserves_what_was_found(self):
        memory = CaseMemory()
        memory.remember(PastCase("a", ucc(), Verdict.INVALID,
                                 ("ideal_control",), ("determinism",),
                                 (FailureMode.COMPILER_CANCELLATION,), ("T_sign",)))
        restored = CaseMemory.from_json(memory.to_json())
        case_back = restored.cases[0]
        self.assertEqual(case_back.failed_gates, ("ideal_control",))
        self.assertEqual(case_back.unrun_gates, ("determinism",))
        self.assertEqual(case_back.failure_modes,
                         (FailureMode.COMPILER_CANCELLATION,))
        self.assertEqual(case_back.attacks_that_fired, ("T_sign",))

    def test_it_is_a_readable_file(self):
        memory = CaseMemory()
        memory.remember(case("a"))
        entries = json.loads(memory.to_json())
        self.assertIn("fingerprint", entries[0])
        self.assertIn("verdict", entries[0])

    def test_loading_a_missing_file_gives_an_empty_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(CaseMemory.load(Path(tmp) / "nope.json")), 0)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            memory = CaseMemory()
            memory.remember(case("a"))
            memory.remember(case("b", ucc(9)))
            memory.save(path)
            self.assertEqual(len(CaseMemory.load(path)), 2)


class CaseFromAuditTest(unittest.TestCase):
    def test_it_records_what_failed_and_what_was_never_run(self):
        from qem_auditor import audit

        from .helpers import make_experiment

        exp = make_experiment(ideal_control=False, determinism_check=None)
        built = case_from_audit(exp, audit(exp), ucc())
        self.assertIn("ideal_control", built.failed_gates)
        self.assertIn("determinism", built.unrun_gates)
        self.assertIs(built.verdict, Verdict.INVALID)


if __name__ == "__main__":
    unittest.main()
