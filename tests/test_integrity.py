"""Record integrity: a record whose own numbers contradict each other is
unauditable, and must say so rather than quietly producing a verdict.

INVALID RECORD is kept distinct from INVALID on purpose -- "your evidence
is incoherent" is a different statement from "your method failed", and
collapsing them would let a bookkeeping mistake read as a scientific
refutation (or vice versa).
"""
import unittest

from qem_auditor import Verdict, audit
from qem_auditor.integrity import integrity_violations

from .helpers import make_experiment


class CleanRecordTest(unittest.TestCase):
    def test_clean_record_has_no_violations(self):
        self.assertEqual(integrity_violations(make_experiment()), [])

    def test_sparse_but_honest_record_is_not_a_violation(self):
        """Missing data is not incoherent data -- an early record with almost
        nothing filled in is under-established, never INVALID RECORD."""
        exp = make_experiment(ideal_control=None, target_leakage_check=None,
                              adversarial_check=None, reproducibility_checked=False,
                              raw_error_kcal=None, mitigated_error_kcal=None,
                              q95_kcal=None, replicate_errors_kcal=[])
        self.assertEqual(integrity_violations(exp), [])
        self.assertIs(audit(exp).verdict, Verdict.NOT_ESTABLISHED)


class ViolationTest(unittest.TestCase):
    def assert_violates(self, snippet, **overrides):
        v = integrity_violations(make_experiment(**overrides))
        self.assertTrue(any(snippet in x for x in v), f"expected {snippet!r} in {v}")
        self.assertIs(audit(make_experiment(**overrides)).verdict, Verdict.INVALID_RECORD)

    def test_empty_experiment_id(self):
        self.assert_violates("experiment_id is empty", experiment_id="   ")

    def test_empty_backend(self):
        self.assert_violates("backend is empty", backend="")

    def test_non_positive_shots(self):
        self.assert_violates("not a positive shot count", shots=0)

    def test_non_positive_replicate_target(self):
        self.assert_violates("not a positive target", n_replicates_target=0)

    def test_negative_error_magnitude(self):
        self.assert_violates("is negative", mitigated_error_kcal=-0.10)

    def test_negative_replicate(self):
        self.assert_violates("replicate_errors_kcal[2]",
                             replicate_errors_kcal=[0.10, 0.10, -0.10, 0.10])

    def test_reproducibility_claimed_without_data(self):
        self.assert_violates("reproducibility cannot be asserted",
                             reproducibility_checked=True, replicate_errors_kcal=[0.10])

    def test_q95_tighter_than_point_estimate(self):
        """A 95% envelope narrower than the error it envelopes is not a
        stricter result, it is a bookkeeping error -- and it would otherwise
        make a claim look BETTER at the chemical-accuracy gate."""
        self.assert_violates("cannot be", mitigated_error_kcal=0.40, q50_kcal=0.05,
                             q95_kcal=0.10, q99_kcal=0.15,
                             replicate_errors_kcal=[0.40] * 8)

    def test_headline_number_not_backed_by_its_own_replicates(self):
        """The best-draw-as-headline pattern: eight replicates around 0.5,
        reported as 0.01."""
        self.assert_violates("not what the replicates measured",
                             mitigated_error_kcal=0.01, q50_kcal=0.50, q95_kcal=0.60,
                             q99_kcal=0.70,
                             replicate_errors_kcal=[0.50, 0.52, 0.48, 0.51,
                                                    0.49, 0.50, 0.53, 0.47])

    def test_headline_within_replicate_scatter_is_fine(self):
        exp = make_experiment(mitigated_error_kcal=0.50, q50_kcal=0.50,
                              q95_kcal=0.60, q99_kcal=0.70,
                              replicate_errors_kcal=[0.48, 0.52, 0.49, 0.51])
        self.assertEqual(integrity_violations(exp), [])


class PrecedenceTest(unittest.TestCase):
    def test_integrity_outranks_a_failed_hard_gate(self):
        """If the record cannot be trusted to say what it says, the auditor
        should not go on to pronounce on the science it describes."""
        exp = make_experiment(ideal_control=False, shots=-1)
        self.assertIs(audit(exp).verdict, Verdict.INVALID_RECORD)

    def test_report_lists_the_violations(self):
        report = audit(make_experiment(shots=0, backend=""))
        self.assertEqual(len(report.integrity_violations), 2)


if __name__ == "__main__":
    unittest.main()
