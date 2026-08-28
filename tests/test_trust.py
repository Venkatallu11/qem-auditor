"""QEM-Trust scoring.

The properties tested here are the ones that make a benchmark worth
publishing: that hedging cannot earn a score, that one false endorsement
outweighs everything else, and that the suite reports its own resolution
honestly.
"""
import unittest

from qem_auditor import FailureMode, Verdict
from qem_auditor.trust import (
    Answer,
    Case,
    Pair,
    ErrorKind,
    Stance,
    TrustGrade,
    best_constant,
    builtin_auditor,
    classify_error,
    constant_auditor,
    score,
    wilson_interval,
)

from .helpers import make_experiment


def _case(case_id, truth, mode=None):
    return Case(case_id=case_id, experiment=make_experiment(), truth=truth,
                what_it_tests="a fixture, standing in for a real case",
                truth_mode=mode)


#: A synthetic suite with a case in every stance, so the scoring
#: properties can be tested without depending on the real six.
SUITE = [
    _case("broken_a", Verdict.INVALID, FailureMode.COMPILER_CANCELLATION),
    _case("broken_b", Verdict.REFUTED),
    _case("unclear", Verdict.NOT_ESTABLISHED),
    _case("sound", Verdict.CERTIFIED_UNDER_SCOPE),
]


class StanceTest(unittest.TestCase):
    def test_every_verdict_has_a_stance(self):
        from qem_auditor.trust import STANCE

        for verdict in Verdict:
            self.assertIn(verdict, STANCE)


class ErrorClassificationTest(unittest.TestCase):
    def test_exact_match_is_not_an_error(self):
        self.assertIs(classify_error(Verdict.INVALID, Verdict.INVALID),
                      ErrorKind.NONE)

    def test_certifying_a_broken_result_is_the_disqualifying_error(self):
        self.assertIs(classify_error(Verdict.INVALID, Verdict.CERTIFIED_UNDER_SCOPE),
                      ErrorKind.FALSE_ENDORSEMENT)

    def test_promising_a_broken_result_is_equally_disqualifying(self):
        """PROMISING reads as 'go build on this' too. The lattice position
        is softer; the consequence is not."""
        self.assertIs(classify_error(Verdict.REFUTED, Verdict.PROMISING),
                      ErrorKind.FALSE_ENDORSEMENT)

    def test_endorsing_something_merely_unproven_is_not_a_false_endorsement(self):
        """Overclaiming on an unproven result is bad, but it is not the
        same as endorsing something known to be broken, and collapsing
        the two would make the disqualifying rule meaningless."""
        self.assertIs(classify_error(Verdict.NOT_ESTABLISHED, Verdict.PROMISING),
                      ErrorKind.OVER_CLAIM)

    def test_withholding_from_a_broken_result_is_separated_from_hedging_on_a_good_one(self):
        self.assertIs(classify_error(Verdict.INVALID, Verdict.NOT_ESTABLISHED),
                      ErrorKind.MISSED_CONDEMNATION)
        self.assertIs(classify_error(Verdict.PROMISING, Verdict.NOT_ESTABLISHED),
                      ErrorKind.OVER_HEDGE)

    def test_confusing_two_withholding_verdicts_is_only_a_tier_slip(self):
        self.assertIs(classify_error(Verdict.MODEL_CONDITIONAL, Verdict.NOT_ESTABLISHED),
                      ErrorKind.TIER_SLIP)

    def test_condemning_a_sound_result_is_its_own_error(self):
        self.assertIs(classify_error(Verdict.CERTIFIED_UNDER_SCOPE, Verdict.INVALID),
                      ErrorKind.FALSE_CONDEMNATION)


class SkillTest(unittest.TestCase):
    def test_the_perpetual_hedger_scores_at_or_below_zero(self):
        report = score(constant_auditor(Verdict.NOT_ESTABLISHED), SUITE, "hedge")
        self.assertLessEqual(report.skill, 0.0)
        self.assertIs(report.grade, TrustGrade.NO_SKILL)

    def test_no_constant_answer_can_show_skill(self):
        """The zero point is defined as the BEST constant answer, so this
        must hold for every one of them, not just the obvious hedge."""
        for verdict in Verdict:
            with self.subTest(verdict=verdict.name):
                report = score(constant_auditor(verdict), SUITE, verdict.name)
                self.assertLessEqual(report.skill, 0.0)

    def test_a_perfect_auditor_scores_one(self):
        # Each case gets its own experiment object so the oracle can look
        # the answer up by identity rather than by reading the record.
        suite = [Case(case_id=c.case_id, experiment=make_experiment(),
                      truth=c.truth, what_it_tests=c.what_it_tests,
                      truth_mode=c.truth_mode) for c in SUITE]
        by_obj = {id(c.experiment): c.truth for c in suite}
        report = score(lambda exp: Answer(by_obj[id(exp)]), suite, "oracle")
        self.assertAlmostEqual(report.skill, 1.0)

    def test_an_auditor_worse_than_a_constant_scores_negative(self):
        report = score(constant_auditor(Verdict.CERTIFIED_UNDER_SCOPE), SUITE, "yes-man")
        self.assertLess(report.skill, 0.0)

    def test_best_constant_is_actually_the_best(self):
        verdict, credit = best_constant(SUITE)
        for other in Verdict:
            self.assertLessEqual(
                score(constant_auditor(other), SUITE, "x").credit, credit + 1e-12)


class DisqualificationTest(unittest.TestCase):
    def test_one_false_endorsement_disqualifies_an_otherwise_perfect_auditor(self):
        truths = {c.case_id: c.truth for c in SUITE}

        suite = [Case(case_id=c.case_id, experiment=make_experiment(),
                      truth=c.truth, what_it_tests=c.what_it_tests)
                 for c in SUITE]
        by_obj = {id(c.experiment): c.case_id for c in suite}

        def nearly(exp):
            case_id = by_obj[id(exp)]
            if case_id == "broken_a":
                return Answer(Verdict.CERTIFIED_UNDER_SCOPE)
            return Answer(truths[case_id])

        report = score(nearly, suite, "nearly-perfect")
        self.assertEqual(len(report.false_endorsements), 1)
        self.assertGreater(report.skill, 0.5)
        self.assertIs(report.grade, TrustGrade.DISQUALIFIED)

    def test_the_grade_says_why(self):
        report = score(constant_auditor(Verdict.CERTIFIED_UNDER_SCOPE), SUITE, "yes-man")
        self.assertIn("endorsed a known artifact", report.format_report())


class AttributionTest(unittest.TestCase):
    def test_naming_the_wrong_cause_does_not_cost_verdict_credit(self):
        """Verdict and diagnosis are scored on separate axes: an auditor
        that stops a bad result for the wrong reason still stopped it."""
        suite = [_case("broken", Verdict.INVALID, FailureMode.COMPILER_CANCELLATION)]
        report = score(
            lambda _e: Answer(Verdict.INVALID, FailureMode.DRIFT), suite, "misdiagnoser")
        self.assertEqual(report.exact, 1)
        self.assertEqual(report.attribution, (0, 1))

    def test_an_auditor_that_endorses_earns_no_attribution_score(self):
        suite = [_case("broken", Verdict.INVALID, FailureMode.COMPILER_CANCELLATION)]
        report = score(constant_auditor(Verdict.PROMISING), suite, "yes-man")
        self.assertIsNone(report.attribution)

    def test_a_withholding_verdict_still_owes_a_cause(self):
        """'Not established' is an answer with a reason behind it; an
        auditor that gives the verdict without the reason has not done
        the same work as one that names why."""
        suite = [_case("thin", Verdict.NOT_ESTABLISHED, FailureMode.UNDER_POWERED)]
        self.assertEqual(
            score(lambda _e: Answer(Verdict.NOT_ESTABLISHED), suite, "silent").attribution,
            (0, 1))
        self.assertEqual(
            score(lambda _e: Answer(Verdict.NOT_ESTABLISHED, FailureMode.UNDER_POWERED),
                  suite, "named").attribution,
            (1, 1))

    def test_a_case_cannot_pin_a_cause_to_an_endorsed_result(self):
        with self.assertRaises(ValueError):
            _case("sound", Verdict.CERTIFIED_UNDER_SCOPE, FailureMode.DRIFT)


class IntervalTest(unittest.TestCase):
    def test_the_interval_stays_inside_zero_and_one(self):
        for k in range(7):
            interval = wilson_interval(k, 6)
            self.assertGreaterEqual(interval.low, 0.0)
            self.assertLessEqual(interval.high, 1.0)

    def test_six_for_six_does_not_report_certainty(self):
        """The point of carrying the interval: a perfect run on six cases
        is consistent with being wrong a third of the time."""
        self.assertLess(wilson_interval(6, 6).low, 0.7)

    def test_more_cases_narrow_it(self):
        self.assertLess(
            wilson_interval(60, 60).high - wilson_interval(60, 60).low,
            wilson_interval(6, 6).high - wilson_interval(6, 6).low)

    def test_an_empty_suite_is_refused(self):
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)


class SuiteHygieneTest(unittest.TestCase):
    def test_an_empty_suite_scores_nothing(self):
        with self.assertRaises(ValueError):
            score(builtin_auditor, [], "x")

    def test_duplicate_cases_are_refused(self):
        duplicated = SUITE + [SUITE[0]]
        with self.assertRaises(ValueError):
            score(builtin_auditor, duplicated, "x")

    def test_a_case_must_say_what_it_discriminates(self):
        with self.assertRaises(ValueError):
            Case(case_id="x", experiment=make_experiment(),
                 truth=Verdict.INVALID, what_it_tests="   ")

    def test_a_bare_verdict_is_an_acceptable_answer(self):
        report = score(lambda _e: Verdict.INVALID, SUITE, "terse")
        self.assertEqual(report.n, len(SUITE))

    def test_returning_something_else_is_refused(self):
        with self.assertRaises(TypeError):
            score(lambda _e: "INVALID", SUITE, "confused")


class RealSuiteTest(unittest.TestCase):
    """The six disclosed cases, scored rather than merely pinned."""

    def setUp(self):
        from benchmarks.suite import CASES

        self.cases = CASES

    def test_the_suite_has_a_case_in_every_stance(self):
        from qem_auditor.trust import STANCE

        stances = {STANCE[c.truth] for c in self.cases}
        self.assertEqual(stances, set(Stance))

    def test_this_package_shows_skill_on_it(self):
        report = score(builtin_auditor, self.cases, "qem-auditor")
        self.assertGreater(report.skill, 0.0)
        self.assertEqual(report.false_endorsements, [])

    def test_the_hedger_does_not(self):
        report = score(constant_auditor(Verdict.NOT_ESTABLISHED), self.cases, "hedge")
        self.assertLessEqual(report.skill, 0.0)

    def test_the_runner_exits_zero(self):
        import run_trust

        self.assertEqual(run_trust.main(), 0)

    def test_the_truth_verdicts_come_from_the_case_files_not_from_the_auditor(self):
        """If someone 'fixes' a failing benchmark by copying the auditor's
        current output into the truth column, the suite silently becomes a
        regression test again. This pins the truths literally."""
        expected = {
            "h4_abstract_fold_compiler_cancellation": Verdict.INVALID,
            "h4_all_gate_zne_ideal_control": Verdict.INVALID,
            "h4_calibrated_pec_manifold_one_off": Verdict.NOT_ESTABLISHED,
            "h4_cross_fitted_manifold": Verdict.REFUTED,
            "h4_joint_schmidt_frame": Verdict.MODEL_CONDITIONAL,
            "h4_ancilla_qed_conditioned_pec": Verdict.PROMISING,
        }
        self.assertEqual({c.case_id: c.truth for c in self.cases}, expected)


if __name__ == "__main__":
    unittest.main()


class ProvenanceTest(unittest.TestCase):
    def test_constructed_is_the_default(self):
        """Claiming a case is a real disclosed result should take a
        deliberate act, not an omission."""
        from qem_auditor.trust import CaseProvenance

        self.assertIs(_case("x", Verdict.INVALID).provenance,
                      CaseProvenance.CONSTRUCTED)

    def test_the_disclosed_six_are_labelled_disclosed(self):
        from benchmarks.suite import CASES
        from qem_auditor.trust import CaseProvenance

        for case in CASES:
            with self.subTest(case=case.case_id):
                self.assertIs(case.provenance, CaseProvenance.DISCLOSED)

    def test_the_score_is_reported_per_provenance_not_only_blended(self):
        """A tool can score well on constructed cases by learning the
        schema. The split is what shows it."""
        from benchmarks.suite import ALL_CASES
        from qem_auditor.trust import CaseProvenance, number_reading_auditor

        report = score(number_reading_auditor, ALL_CASES, "number-reader")
        for provenance in CaseProvenance:
            self.assertIsNotNone(report.credit_on(provenance))
        self.assertIn("disclosed", report.format_report())
        self.assertIn("constructed", report.format_report())


class PairTest(unittest.TestCase):
    def _pair_suite(self):
        good = _case("good", Verdict.CERTIFIED_UNDER_SCOPE)
        bad = _case("bad", Verdict.INVALID)
        pair = Pair("p", "good", "bad", "one field", "because it matters")
        return [good, bad], [pair]

    def test_getting_one_member_right_earns_nothing(self):
        """Half a pair is what guessing looks like."""
        cases, pairs = self._pair_suite()
        report = score(constant_auditor(Verdict.INVALID), cases, "half", pairs)
        self.assertEqual(report.exact, 1)
        self.assertEqual(report.pair_score, (0, 1))

    def test_getting_both_right_earns_the_pair(self):
        cases, pairs = self._pair_suite()
        truth = {c.case_id: c.truth for c in cases}
        by_obj = {id(c.experiment): truth[c.case_id] for c in cases}
        report = score(lambda e: Answer(by_obj[id(e)]), cases, "oracle", pairs)
        self.assertEqual(report.pair_score, (1, 1))

    def test_no_pairs_means_no_pair_score(self):
        self.assertIsNone(score(builtin_auditor, SUITE, "x").pair_score)

    def test_a_pair_naming_a_missing_case_fails_at_scoring_time(self):
        cases, _ = self._pair_suite()
        stray = [Pair("p", "good", "not_in_suite", "a field", "a reason")]
        with self.assertRaises(ValueError):
            score(builtin_auditor, cases, "x", stray)

    def test_a_pair_cannot_be_a_case_with_itself(self):
        with self.assertRaises(ValueError):
            Pair("p", "same", "same", "a field", "a reason")

    def test_a_pair_must_state_what_differs_and_why(self):
        with self.assertRaises(ValueError):
            Pair("p", "a", "b", "  ", "a reason")
        with self.assertRaises(ValueError):
            Pair("p", "a", "b", "a field", "")


class ConstructedSuiteTest(unittest.TestCase):
    """The minimal pairs, and the evidence that they earn their place."""

    def setUp(self):
        from benchmarks.suite import ALL_CASES, CASES, PAIRS

        self.disclosed = CASES
        self.all_cases = ALL_CASES
        self.pairs = PAIRS

    def test_every_constructed_record_is_internally_consistent(self):
        from qem_auditor.integrity import integrity_violations
        from qem_auditor.trust import CaseProvenance

        for case in self.all_cases:
            if case.provenance is not CaseProvenance.CONSTRUCTED:
                continue
            with self.subTest(case=case.case_id):
                self.assertEqual(integrity_violations(case.experiment), [])

    def test_each_pair_members_differ_only_in_their_stated_respect(self):
        """If two pair members differ in several ways, a correct answer
        no longer attributes to the one thing the pair is about."""
        import dataclasses

        by_id = {c.case_id: c.experiment for c in self.all_cases}
        for pair in self.pairs:
            with self.subTest(pair=pair.pair_id):
                a, b = by_id[pair.case_a], by_id[pair.case_b]
                differing = [
                    f.name for f in dataclasses.fields(a)
                    if f.name not in ("experiment_id", "description", "claim")
                    and getattr(a, f.name) != getattr(b, f.name)
                ]
                # controls/outputs are the fields a pair is allowed to move.
                self.assertLessEqual(
                    set(differing) - {"controls", "outputs", "claim_type"}, set(),
                    f"{pair.pair_id} differs in {differing}")

    def test_the_pairs_make_the_top_verdict_reachable(self):
        """Before them, no case in the suite reached CERTIFIED UNDER
        SCOPE, so an auditor incapable of the top verdict scored the same
        as one that could produce it."""
        self.assertNotIn(Verdict.CERTIFIED_UNDER_SCOPE,
                         {c.truth for c in self.disclosed})
        self.assertIn(Verdict.CERTIFIED_UNDER_SCOPE,
                      {c.truth for c in self.all_cases})

    def test_the_pairs_expose_an_auditor_the_disclosed_cases_flatter(self):
        """The whole justification for the constructed half, measured
        rather than asserted."""
        from qem_auditor.trust import TrustGrade, number_reading_auditor

        flattered = score(number_reading_auditor, self.disclosed, "reader")
        self.assertGreater(flattered.skill, 0.0)
        self.assertIs(flattered.grade, TrustGrade.PARTIAL_SKILL)

        exposed = score(number_reading_auditor, self.all_cases, "reader", self.pairs)
        self.assertEqual(exposed.pair_score[0], 0)
        self.assertIs(exposed.grade, TrustGrade.DISQUALIFIED)

    def test_the_suite_now_exercises_gates_the_disclosed_cases_never_ran(self):
        from qem_auditor import audit

        def outcomes(cases):
            seen = {}
            for case in cases:
                for result in audit(case.experiment).gate_results:
                    seen.setdefault(result.name, set()).add(result.passed)
            return seen

        before, after = outcomes(self.disclosed), outcomes(self.all_cases)
        self.assertNotIn(True, before["mitigation_benefit"])
        self.assertIn(True, after["mitigation_benefit"])
        self.assertNotIn(True, before["independent_verification"])
        self.assertIn(True, after["independent_verification"])
