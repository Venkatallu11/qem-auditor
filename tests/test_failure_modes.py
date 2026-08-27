"""The classifier: does it name the right root cause, from the record
alone, without being shown what the author suspected?
"""
import unittest

from qem_auditor import FailureMode, UncertaintyCoverage, audit, classify
from qem_auditor.failure_modes import agreement_with_record

from .helpers import make_experiment


class ClassifierTest(unittest.TestCase):
    def test_clean_record_implicates_nothing(self):
        exp = make_experiment()
        self.assertEqual(classify(exp, audit(exp)).diagnoses, [])

    def test_ideal_control_blowup_is_conditioning_not_hardware_noise(self):
        """The distinction that makes the diagnosis useful: mitigation
        amplifying error on a NOISELESS model cannot be a noise problem."""
        exp = make_experiment(ideal_control=False, raw_error_kcal=0.0652,
                              mitigated_error_kcal=33.48)
        analysis = classify(exp, audit(exp))
        self.assertIs(analysis.primary.mode, FailureMode.EXTRAPOLATION_INSTABILITY)
        self.assertIn("513x", analysis.primary.evidence)
        self.assertIn("not hardware noise", analysis.primary.evidence)

    def test_scope_gap_is_attributed_to_the_axis_actually_missing(self):
        """A thorough noise-model envelope missing only cross-submission
        implicates drift, not calibration."""
        exp = make_experiment(uncertainty=UncertaintyCoverage(
            shot_noise=True, method_monte_carlo=True, noise_model=True))
        modes = classify(exp, audit(exp)).modes
        self.assertIn(FailureMode.DRIFT, modes)
        self.assertNotIn(FailureMode.CALIBRATION_MISMATCH, modes)

    def test_missing_noise_model_implicates_calibration(self):
        exp = make_experiment(uncertainty=UncertaintyCoverage(
            shot_noise=True, method_monte_carlo=True, cross_submission=True))
        modes = classify(exp, audit(exp)).modes
        self.assertIn(FailureMode.CALIBRATION_MISMATCH, modes)
        self.assertNotIn(FailureMode.DRIFT, modes)

    def test_one_diagnosis_per_mode(self):
        """Several rules can reach the same mode; the reader gets one
        finding with the evidence joined, not duplicates."""
        exp = make_experiment(ideal_control=False, extrapolation_in_domain=False,
                              raw_error_kcal=0.0652, mitigated_error_kcal=33.48)
        modes = classify(exp, audit(exp)).modes
        self.assertEqual(len(modes), len(set(modes)))

    def test_every_diagnosis_carries_specific_evidence_and_a_remedy(self):
        exp = make_experiment(ideal_control=False, determinism_check=False,
                              raw_error_kcal=0.10, mitigated_error_kcal=9.0)
        for d in classify(exp, audit(exp)).diagnoses:
            with self.subTest(mode=d.mode):
                self.assertTrue(d.evidence.strip())
                self.assertTrue(d.remedy.strip())
                # Evidence must be an observation, not the mode's own name.
                self.assertNotEqual(d.evidence.lower(), d.mode.value.lower())
                self.assertGreater(d.confidence, 0.0)

    def test_diagnoses_are_ordered_most_confident_first(self):
        exp = make_experiment(ideal_control=False, raw_error_kcal=0.10,
                              mitigated_error_kcal=9.0,
                              uncertainty=UncertaintyCoverage(shot_noise=True))
        confidences = [d.confidence for d in classify(exp, audit(exp)).diagnoses]
        self.assertEqual(confidences, sorted(confidences, reverse=True))


class IndependenceFromTheRecordTest(unittest.TestCase):
    def test_the_authors_suspicion_does_not_steer_the_diagnosis(self):
        """Whoever wrote the record does not get to pick the root cause."""
        honest = make_experiment(ideal_control=False, raw_error_kcal=0.0652,
                                 mitigated_error_kcal=33.48)
        misleading = make_experiment(ideal_control=False, raw_error_kcal=0.0652,
                                     mitigated_error_kcal=33.48)
        misleading.suspected_failure_modes = [FailureMode.SIGN_CONVENTION]
        self.assertEqual(classify(honest, audit(honest)).modes,
                         classify(misleading, audit(misleading)).modes)

    def test_agreement_is_reported_separately(self):
        exp = make_experiment(ideal_control=False, raw_error_kcal=0.0652,
                              mitigated_error_kcal=33.48)
        exp.suspected_failure_modes = [FailureMode.EXTRAPOLATION_INSTABILITY,
                                       FailureMode.SIGN_CONVENTION]
        agreement = agreement_with_record(exp, classify(exp, audit(exp)))
        self.assertIn("EXTRAPOLATION_INSTABILITY", agreement["confirmed"])
        self.assertIn("SIGN_CONVENTION", agreement["missed_by_classifier"])


if __name__ == "__main__":
    unittest.main()
