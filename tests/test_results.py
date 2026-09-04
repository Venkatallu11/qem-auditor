"""The door for people who already ran the experiment.

Every other entry point wants a circuit or a written record. This one
wants counts, which is what a person actually has when they come back
from a device -- and it needs no qiskit, because counts are dictionaries.
"""
import itertools
import math
import random
import statistics
import unittest

from qem_auditor.estimation import EstimationError
from qem_auditor.results import (analyse, apply_readout_correction,
                                 corrected_shot_noise, extrapolate_to_zero,
                                 readout_model, shot_noise)

TRUTH = {"00": 0.55, "01": 0.15, "10": 0.20, "11": 0.10}
EPSILON = [(0.06, 0.08), (0.05, 0.07)]
OBSERVABLE = [("ZZ", 0.4), ("II", -1.05)]


def confuse(distribution, epsilon=EPSILON):
    """Apply a known readout model, so tests have a ground truth."""
    width = len(next(iter(distribution)))
    out = {}
    for bits, p in distribution.items():
        for observed in itertools.product("01", repeat=width):
            weight = p
            for qubit in range(width):
                wrong_0, wrong_1 = epsilon[qubit]
                if bits[qubit] == "0":
                    weight *= wrong_0 if observed[qubit] == "1" else 1 - wrong_0
                else:
                    weight *= wrong_1 if observed[qubit] == "0" else 1 - wrong_1
            key = "".join(observed)
            out[key] = out.get(key, 0.0) + weight
    return out


def calibration_for(epsilon=EPSILON, width=2, shots=1e7):
    return {"prepared_0": {b: v * shots for b, v in confuse({"0" * width: 1.0}, epsilon).items()},
            "prepared_1": {b: v * shots for b, v in confuse({"1" * width: 1.0}, epsilon).items()}}


class ShotNoiseTest(unittest.TestCase):
    """The floor is exact, not a rule of thumb. It is the number a refusal
    rests on, so it is checked against resampling rather than algebra."""

    def test_the_variance_matches_repeated_experiments(self):
        distribution = {"ZZ": TRUTH}
        shots, trials = 4000, 1200
        exact = shot_noise({"ZZ": {b: p * shots for b, p in TRUTH.items()}}, OBSERVABLE)
        rng = random.Random(4)

        def draw():
            table = {}
            for bits in rng.choices(list(TRUTH), weights=list(TRUTH.values()), k=shots):
                table[bits] = table.get(bits, 0) + 1
            return {"ZZ": table}

        observed = statistics.stdev([shot_noise(draw(), OBSERVABLE).estimate
                                     for _ in range(trials)])
        error = observed / math.sqrt(2 * (trials - 1))
        self.assertLess(abs(observed - exact.sigma), 4 * error,
                        f"predicted {exact.sigma}, observed {observed}")

    def test_terms_sharing_a_setting_are_treated_as_correlated(self):
        """Adding term variances independently understates the floor, which
        is the wrong direction for a number whose job is to refuse."""
        counts = {"ZZ": {"00": 5000, "11": 5000}}
        together = shot_noise(counts, [("ZZ", 1.0), ("ZI", 1.0)])
        # ZZ is +1 on both outcomes here; ZI is +1 and -1. Their sum is
        # perfectly correlated with ZI alone, not the sum of two spreads.
        alone = shot_noise(counts, [("ZI", 1.0)])
        self.assertAlmostEqual(together.variance, alone.variance)

    def test_an_identity_term_moves_the_estimate_and_not_the_floor(self):
        counts = {"ZZ": {"00": 5000, "11": 5000}}
        without = shot_noise(counts, [("ZZ", 1.0)])
        with_identity = shot_noise(counts, [("ZZ", 1.0), ("II", -2.0)])
        self.assertAlmostEqual(with_identity.estimate, without.estimate - 2.0)
        self.assertAlmostEqual(with_identity.variance, without.variance)

    def test_a_term_with_no_measurements_is_refused_by_name(self):
        with self.assertRaises(EstimationError) as caught:
            shot_noise({"ZZ": {"00": 10}}, [("XX", 1.0)])
        self.assertIn("XX", str(caught.exception))

    def test_a_run_with_no_shots_is_refused(self):
        with self.assertRaises(EstimationError):
            shot_noise({"ZZ": {}}, [("ZZ", 1.0)])

    def test_it_says_how_many_shots_a_target_precision_needs(self):
        noise = shot_noise({"ZZ": {b: p * 4000 for b, p in TRUTH.items()}}, OBSERVABLE)
        target = noise.floor() / 4
        needed = noise.shots_for(target)
        self.assertGreater(needed, noise.shots)
        self.assertIsNone(noise.shots_for(noise.floor() * 2),
                          "a target already met needs no more shots")


class ReadoutCorrectionTest(unittest.TestCase):

    def test_it_recovers_the_true_distribution_exactly(self):
        model = readout_model(calibration_for(), 2)
        observed = {b: v * 1e6 for b, v in confuse(TRUTH).items()}
        recovered = apply_readout_correction(observed, model)
        mass = sum(recovered.values())
        for bits, p in TRUTH.items():
            self.assertAlmostEqual(recovered[bits] / mass, p, places=9)

    def test_a_singular_readout_model_is_refused(self):
        """A qubit whose two prepared states read identically cannot be
        corrected around -- that is a dead qubit or a mislabelled
        calibration, not a small number to divide by."""
        dead = {"prepared_0": {"0": 5000, "1": 5000},
                "prepared_1": {"0": 5000, "1": 5000}}
        model = readout_model(dead, 1)
        with self.assertRaises(EstimationError):
            apply_readout_correction({"0": 100, "1": 100}, model)

    def test_missing_calibration_states_are_named(self):
        with self.assertRaises(EstimationError) as caught:
            readout_model({"prepared_0": {"0": 10}}, 1)
        self.assertIn("prepared_1", str(caught.exception))


class CorrectionAmplifiesNoiseTest(unittest.TestCase):
    """The bug this module shipped in its own first draft.

    Correcting the counts and then measuring their spread reports the
    corrected distribution's spread, which has nothing to do with the
    estimate's uncertainty -- it produced `+- 0` on a mitigated estimate.
    Mitigation AMPLIFIES shot noise, and a pipeline reporting a tighter
    bar after mitigation is the exact failure this package exists to
    catch.
    """

    def test_the_corrected_estimate_has_a_wider_bar_not_a_narrower_one(self):
        model = readout_model(calibration_for(), 2)
        counts = {"ZZ": {b: v * 4000 for b, v in confuse(TRUTH).items()}}
        raw = shot_noise(counts, OBSERVABLE)
        corrected = corrected_shot_noise(counts, OBSERVABLE, model)
        self.assertGreater(corrected.sigma, raw.sigma)

    def test_the_amplification_matches_repeated_experiments(self):
        model = readout_model(calibration_for(), 2)
        observed_distribution = confuse(TRUTH)
        shots, trials = 4000, 900
        exact = corrected_shot_noise(
            {"ZZ": {b: p * shots for b, p in observed_distribution.items()}},
            OBSERVABLE, model)
        rng = random.Random(21)

        def draw():
            table = {}
            for bits in rng.choices(list(observed_distribution),
                                    weights=list(observed_distribution.values()),
                                    k=shots):
                table[bits] = table.get(bits, 0) + 1
            return {"ZZ": table}

        spread = statistics.stdev(
            [corrected_shot_noise(draw(), OBSERVABLE, model).estimate
             for _ in range(trials)])
        error = spread / math.sqrt(2 * (trials - 1))
        self.assertLess(abs(spread - exact.sigma), 4 * error,
                        f"predicted {exact.sigma}, observed {spread}")

    def test_correction_removes_the_bias_it_is_there_to_remove(self):
        model = readout_model(calibration_for(), 2)
        counts = {"ZZ": {b: v * 1e6 for b, v in confuse(TRUTH).items()}}
        true_value = 0.4 * sum(p * (1 if b in ("00", "11") else -1)
                               for b, p in TRUTH.items()) - 1.05
        raw = shot_noise(counts, OBSERVABLE).estimate
        corrected = corrected_shot_noise(counts, OBSERVABLE, model).estimate
        self.assertLess(abs(corrected - true_value), abs(raw - true_value))
        self.assertAlmostEqual(corrected, true_value, places=6)


class ExtrapolationTest(unittest.TestCase):

    def test_a_linear_signal_extrapolates_exactly(self):
        self.assertAlmostEqual(extrapolate_to_zero([1, 3, 5], [0.9, 0.7, 0.5]), 1.0)

    def test_an_order_needs_more_points_than_coefficients(self):
        """Fitting a parabola through three points is interpolation wearing
        a fit's clothes."""
        with self.assertRaises(EstimationError):
            extrapolate_to_zero([1, 3], [0.9, 0.7], order=2)

    def test_degenerate_scales_are_refused(self):
        with self.assertRaises(EstimationError):
            extrapolate_to_zero([3, 3], [0.9, 0.7])


class ReportTest(unittest.TestCase):

    def test_a_claim_below_the_shot_noise_floor_is_impossible(self):
        report = analyse({"ZZ": {b: p * 4000 for b, p in TRUTH.items()}},
                         OBSERVABLE, claimed_uncertainty=1e-6)
        self.assertTrue(report.claim_is_impossible)
        self.assertIn("BELOW the shot-noise floor", report.format_report())
        self.assertIn("shots would", report.format_report())

    def test_a_claim_that_clears_the_floor_is_not_condemned(self):
        report = analyse({"ZZ": {b: p * 4000 for b, p in TRUTH.items()}},
                         OBSERVABLE, claimed_uncertainty=1.0)
        self.assertFalse(report.claim_is_impossible)

    def test_missing_inputs_are_named_with_what_to_submit(self):
        report = analyse({"ZZ": {b: p * 4000 for b, p in TRUTH.items()}}, OBSERVABLE)
        text = report.format_report()
        self.assertIn("prepared_0", text)
        self.assertIn("fold factor", text)

    def test_every_advertised_method_is_actually_run(self):
        """Listing a method as available and then not running it would be
        the same unbacked claim this package objects to elsewhere."""
        report = analyse({"ZZ": {b: v * 4000 for b, v in confuse(TRUTH).items()}},
                         OBSERVABLE, calibration=calibration_for())
        run = {estimate.method for estimate in report.estimates}
        self.assertIn("unmitigated", run)
        self.assertIn("REM (tensored)", run)


class TermRoutingTest(unittest.TestCase):
    """One Z-basis measurement yields ZZ, ZI and IZ at once. Demanding a
    separate setting per term would make people collect several times the
    data they need, and is not how anyone runs an experiment."""

    def test_a_partial_term_is_measured_by_a_fuller_setting(self):
        counts = {"ZZ": {"00": 6000, "11": 2000, "01": 1000, "10": 1000}}
        estimate = shot_noise(counts, [("ZI", 1.0)]).estimate
        # ZI reads qubit 1 only: +1 on "00"/"01", -1 on "11"/"10"
        # (position 0 of the bitstring is qubit 0, so qubit 1 is index 1).
        expected = (6000 - 2000 - 1000 + 1000) / 10000
        self.assertAlmostEqual(estimate, expected)

    def test_several_terms_share_one_setting(self):
        counts = {"ZZ": {"00": 6000, "11": 2000, "01": 1000, "10": 1000}}
        combined = shot_noise(counts, [("ZZ", 1.0), ("ZI", 1.0), ("IZ", 1.0)])
        self.assertEqual(combined.settings, 1)
        self.assertEqual(combined.shots, 10000)

    def test_an_incompatible_basis_is_still_refused(self):
        """Routing is generous about identity, never about basis: an X term
        cannot be read off a Z measurement."""
        with self.assertRaises(EstimationError) as caught:
            shot_noise({"ZZ": {"00": 10}}, [("XI", 1.0)])
        self.assertIn("XI", str(caught.exception))

    def test_correlated_terms_do_not_get_independent_variances(self):
        counts = {"ZZ": {"00": 5000, "11": 5000}}
        together = shot_noise(counts, [("ZZ", 1.0), ("ZI", 1.0)])
        alone = shot_noise(counts, [("ZI", 1.0)])
        # On this data ZZ is +1 everywhere, so it adds a constant and no
        # spread; the pair's variance is ZI's, not a sum of two.
        self.assertAlmostEqual(together.variance, alone.variance)
