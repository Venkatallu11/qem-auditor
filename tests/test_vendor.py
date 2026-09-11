"""Reading the provider's own account of a run.

Every fixture here is a real IonQ job record (account identifiers
redacted, physics untouched). The checks are about the one party whose
statement nobody reads: the service that actually ran the circuit.
"""
import json
import unittest
from pathlib import Path

from qem_auditor.vendor import VendorRecordError, cross_check, ionq_job

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ionq_forte_job.json"


def record() -> dict:
    return json.loads(FIXTURE.read_text())


class ReadingTest(unittest.TestCase):

    def setUp(self):
        self.job = ionq_job(record())

    def test_it_reads_the_wrapper_or_the_bare_record(self):
        bare = ionq_job(record()["raw_job_metadata"])
        self.assertEqual(bare.job_id, self.job.job_id)

    def test_the_executed_gate_counts_are_the_vendors(self):
        self.assertEqual(self.job.one_qubit_gates, 120)
        self.assertEqual(self.job.two_qubit_gates, 11)

    def test_the_aggregate_wins_over_the_per_gate_breakdown(self):
        """IonQ's record carries BOTH `1q: 120` and `GPIq: 40` +
        `GPI2q: 80` in one mapping. Summing everything that looks like a
        one-qubit gate returns 240 for a circuit that ran 120."""
        counts = self.job.gate_counts
        self.assertIn("1q", counts)
        self.assertIn("GPI2q", counts)
        self.assertEqual(self.job.one_qubit_gates, counts["1q"])

    def test_a_qiskit_style_breakdown_with_no_aggregate_still_works(self):
        raw = record()
        raw["raw_job_metadata"]["stats"]["gate_counts"] = {
            "gpi2": 80, "gpi": 40, "zz": 11, "measure": 5, "barrier": 1}
        job = ionq_job(raw)
        self.assertEqual(job.one_qubit_gates, 120)
        self.assertEqual(job.two_qubit_gates, 11)

    def test_this_job_declares_no_vendor_mitigation(self):
        self.assertFalse(self.job.vendor_mitigated)
        self.assertIs(self.job.debiasing, False)

    def test_the_compilation_basis_is_read(self):
        self.assertEqual(self.job.gate_basis, "ZZ")

    def test_a_non_mapping_is_refused(self):
        with self.assertRaises(VendorRecordError):
            ionq_job([1, 2, 3])


class AbsenceIsNotADeclarationTest(unittest.TestCase):
    """A record that says nothing must not be read as saying "no"."""

    def test_a_missing_mitigation_block_is_unknown_not_false(self):
        raw = record()
        raw["raw_job_metadata"]["output"].pop("error_mitigation")
        job = ionq_job(raw)
        self.assertIsNone(job.debiasing)
        self.assertIn("none declared", job.describe())

    def test_a_batch_job_reads_its_children(self):
        """A multi-circuit job keeps mitigation on the children; the
        parent's empty output must not read as "nothing applied"."""
        raw = record()
        child = {"output": raw["raw_job_metadata"].pop("output"),
                 "stats": raw["raw_job_metadata"].pop("stats")}
        raw["raw_job_metadata"]["children_raw_job_metadata"] = [child]
        job = ionq_job(raw)
        self.assertEqual(job.gate_basis, "ZZ")
        self.assertEqual(job.one_qubit_gates, 120)

    def test_no_gate_counts_says_so_rather_than_borrowing_the_submitters(self):
        raw = record()
        raw["raw_job_metadata"]["stats"] = {}
        job = ionq_job(raw)
        self.assertEqual(job.gate_counts, {})
        self.assertIn("states no gate counts", job.describe())


class CrossCheckTest(unittest.TestCase):

    def setUp(self):
        self.job = ionq_job(record())

    def test_a_clean_record_contradicts_nothing(self):
        self.assertEqual(
            cross_check(self.job, claims_unmitigated=True,
                        declared_shots=100,
                        declared_one_qubit_gates=120,
                        declared_two_qubit_gates=11), ())

    def test_the_compilation_blowup_is_caught(self):
        found = cross_check(self.job, declared_one_qubit_gates=20)
        self.assertEqual(len(found), 1)
        self.assertIn("120 after compilation", found[0].vendor)

    def test_a_vendor_mitigated_baseline_is_the_finding_this_exists_for(self):
        raw = record()
        raw["raw_job_metadata"]["output"]["error_mitigation"]["debiasing"] = True
        job = ionq_job(raw)
        found = cross_check(job, claims_unmitigated=True)
        self.assertEqual(len(found), 1)
        self.assertIn("already mitigated", found[0].reading)

    def test_symmetry_verification_counts_as_vendor_mitigation_too(self):
        raw = record()
        raw["raw_job_metadata"]["output"]["error_mitigation"][
            "symmetry_verification"]["applied"] = True
        job = ionq_job(raw)
        self.assertTrue(job.vendor_mitigated)

    def test_a_shot_count_disagreement_is_about_the_interval(self):
        found = cross_check(self.job, declared_shots=1000)
        self.assertIn("shot-noise floor", found[0].reading)


if __name__ == "__main__":
    unittest.main()
