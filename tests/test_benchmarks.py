"""The real historical cases, pinned.

These are the tests that matter most: they are not fixtures, they are
disclosed results from quantum-chemistry-vqe. If a future change to the
gates makes the 513x ZNE blowup pass, or makes the project's own best
result certify before its replication is finished, this file fails.
"""
import unittest

from qem_auditor import audit
from qem_auditor.integrity import integrity_violations

import run_benchmarks


class BenchmarkVerdictTest(unittest.TestCase):
    def test_every_benchmark_audits_to_its_expected_verdict(self):
        for module in run_benchmarks.BENCHMARKS:
            with self.subTest(case=module.EXPERIMENT.experiment_id):
                self.assertIs(audit(module.EXPERIMENT).verdict, module.EXPECTED_VERDICT)

    def test_every_benchmark_record_is_internally_consistent(self):
        """The benchmark records themselves must be clean: a real case that
        trips the integrity checks would be testing bookkeeping, not gates."""
        for module in run_benchmarks.BENCHMARKS:
            with self.subTest(case=module.EXPERIMENT.experiment_id):
                self.assertEqual(integrity_violations(module.EXPERIMENT), [])

    def test_runner_exits_zero_when_all_cases_match(self):
        self.assertEqual(run_benchmarks.main(), 0)


class ZneBlowupCaseTest(unittest.TestCase):
    def test_the_ideal_control_is_what_catches_it(self):
        from benchmarks.h4_zne_blowup import EXPERIMENT

        report = audit(EXPERIMENT)
        failed = [g.name for g in report.gate_results if g.passed is False]
        self.assertIn("ideal_control", failed)

    def test_mitigation_made_it_worse(self):
        from benchmarks.h4_zne_blowup import EXPERIMENT

        out = EXPERIMENT.outputs
        self.assertGreater(out.mitigated_error_kcal, out.raw_error_kcal * 100)


class AncillaQedCaseTest(unittest.TestCase):
    def test_all_hard_gates_are_clean(self):
        from benchmarks.h4_ancilla_qed import EXPERIMENT

        report = audit(EXPERIMENT)
        for g in report.gate_results:
            with self.subTest(gate=g.name):
                self.assertIsNot(g.passed, False)

    def test_it_still_refuses_to_certify(self):
        """Clean hard gates are not certification. 4 of 8 replication draws
        and no full real-hardware energy validation means PROMISING."""
        from benchmarks.h4_ancilla_qed import EXPERIMENT

        self.assertLess(len(EXPERIMENT.outputs.replicate_errors_kcal),
                        EXPERIMENT.outputs.n_replicates_target)
        self.assertFalse(EXPERIMENT.real_hardware_full_validation)

    def test_the_honest_q95_is_used_not_the_flattering_point_estimate(self):
        from benchmarks.h4_ancilla_qed import EXPERIMENT

        report = audit(EXPERIMENT)
        accuracy = next(g for g in report.gate_results if g.name == "chemical_accuracy")
        self.assertIn("Q95", accuracy.reason)


if __name__ == "__main__":
    unittest.main()
