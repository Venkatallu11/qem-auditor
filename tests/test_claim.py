"""The claim compiler: does it state limits as specifically as support?"""
import unittest

from qem_auditor import Verdict, audit
from qem_auditor.claim import compile_claim

from .helpers import make_experiment


class CompiledClaimTest(unittest.TestCase):
    def test_incomplete_replication_is_named_in_what_is_not_established(self):
        exp = make_experiment(replicate_errors_kcal=[0.10] * 4)
        claim = compile_claim(exp)
        self.assertTrue(any("4/8" in s for s in claim.not_established))

    def test_a_certified_record_has_nothing_outstanding(self):
        claim = compile_claim(make_experiment())
        self.assertIs(claim.verdict, Verdict.CERTIFIED_UNDER_SCOPE)
        self.assertEqual(claim.not_established, [])

    def test_every_verdict_states_what_it_licenses(self):
        for verdict in Verdict:
            with self.subTest(verdict=verdict):
                from qem_auditor.claim import _LICENCE
                self.assertTrue(_LICENCE[verdict].strip())

    def test_a_failure_carries_its_why_and_its_next_step(self):
        exp = make_experiment(ideal_control=False, raw_error_kcal=0.0652,
                              mitigated_error_kcal=33.48, q50_kcal=33.0,
                              q95_kcal=40.0, q99_kcal=50.0,
                              replicate_errors_kcal=[33.4, 33.5, 33.6, 33.4])
        claim = compile_claim(exp)
        self.assertIs(claim.verdict, Verdict.INVALID)
        self.assertTrue(claim.failure_analysis.diagnoses)
        self.assertIsNotNone(claim.next_experiment)
        rendered = claim.render()
        self.assertIn("WHY:", rendered)
        self.assertIn("NEXT EXPERIMENT:", rendered)

    def test_it_never_says_looks_good(self):
        """The compiler states what was shown, not how it feels about it."""
        for exp in (make_experiment(), make_experiment(ideal_control=False)):
            rendered = compile_claim(exp).render().lower()
            for banned in ("looks good", "looks promising", "seems to work", "great result"):
                self.assertNotIn(banned, rendered)

    def test_inapplicable_gates_are_not_reported_as_gaps(self):
        """chemical_accuracy goes N/A on a relative claim by design; that is
        not missing evidence."""
        from qem_auditor import ClaimType
        exp = make_experiment(claim_type=ClaimType.RELATIVE_IMPROVEMENT,
                              baseline_error_kcal=1.0, mitigated_error_kcal=0.1)
        claim = compile_claim(exp)
        self.assertFalse(any("chemical_accuracy" in s for s in claim.not_established))

    def test_support_and_limits_both_render(self):
        exp = make_experiment(replicate_errors_kcal=[0.10] * 4)
        rendered = compile_claim(exp).render()
        self.assertIn("SUPPORTED BY:", rendered)
        self.assertIn("NOT YET ESTABLISHED:", rendered)
        self.assertIn("PASS CRITERION:", rendered)


class BenchmarkClaimTest(unittest.TestCase):
    def test_every_benchmark_compiles_to_a_claim(self):
        import run_benchmarks

        for module in run_benchmarks.BENCHMARKS:
            exp = module.EXPERIMENT
            with self.subTest(case=exp.experiment_id):
                claim = compile_claim(exp, audit(exp))
                self.assertIs(claim.verdict, module.EXPECTED_VERDICT)
                self.assertTrue(claim.licence.strip())
                self.assertTrue(claim.render().strip())

    def test_no_benchmark_claims_more_than_it_showed(self):
        """Nothing in the suite is certified -- correctly, since none of the
        six has completed replication AND full hardware validation."""
        import run_benchmarks

        for module in run_benchmarks.BENCHMARKS:
            with self.subTest(case=module.EXPERIMENT.experiment_id):
                self.assertIsNot(audit(module.EXPERIMENT).verdict,
                                 Verdict.CERTIFIED_UNDER_SCOPE)


if __name__ == "__main__":
    unittest.main()
