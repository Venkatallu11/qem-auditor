"""The fit-and-reconstruct interface, and the attacks it makes executable."""
import unittest
from statistics import mean

from qem_auditor.reconstruct import (
    FitData,
    shuffle_is_diagnostic,
    Measurement,
    ReconstructionError,
    compare_fit,
    flip_sign,
    reconstruction_spread,
    resample_shots,
    shuffle_labels,
    subsample_draws,
)

# Three labels: with only two, every group gets the same swap and a
# per-label model absorbs it -- see shuffle_is_diagnostic.
IDEAL = {"XX": 0.8, "YY": -0.6, "ZZ": 0.4}


def _data(draws=3, noise=0.0, seed=0):
    import random

    rng = random.Random(seed)
    return FitData([
        Measurement(slot=slot, label=label,
                    value=ideal * 0.9 + (rng.gauss(0, noise) if noise else 0.0),
                    sigma=noise or 0.01, shots=1000, draw=draw)
        for draw in range(draws) for slot in ("u0", "u1")
        for label, ideal in IDEAL.items()])


class GenuineModel:
    """Uses the label correspondence."""

    def fit(self, data):
        return {label: mean([m.value for m in data.measurements if m.label == label])
                / ideal for label, ideal in IDEAL.items()}

    def reconstruct(self, fit, data):
        return sum(mean([m.value for m in data.measurements if m.label == label])
                   / (fit.get(label) or 1.0) for label in IDEAL)

    def goodness_of_fit(self, fit, data):
        return sum(((m.value - IDEAL[m.label] * fit[m.label]) / (m.sigma or 1e-6)) ** 2
                   for m in data.measurements) / len(data.measurements)


class FlexibleModel:
    """One parameter per measurement: fits anything."""

    def fit(self, data):
        return {(m.slot, m.label, m.draw): m.value for m in data.measurements}

    def reconstruct(self, fit, data):
        return sum(fit.values())

    def goodness_of_fit(self, fit, data):
        return sum(((m.value - fit.get((m.slot, m.label, m.draw), m.value))
                    / (m.sigma or 1e-6)) ** 2
                   for m in data.measurements) / len(data.measurements)


class FitDataTest(unittest.TestCase):
    def test_empty_data_is_rejected(self):
        with self.assertRaises(ReconstructionError):
            FitData([])

    def test_copy_is_deep_enough_to_perturb_safely(self):
        data = _data()
        original = [m.label for m in data.measurements]
        shuffle_labels(data, seed=1)
        self.assertEqual([m.label for m in data.measurements], original)


class ShuffleTest(unittest.TestCase):
    def test_labels_move(self):
        data = _data(noise=0.01)
        self.assertNotEqual([m.label for m in shuffle_labels(data, 1).measurements],
                            [m.label for m in data.measurements])

    def test_values_are_untouched(self):
        data = _data(noise=0.01)
        self.assertEqual(sorted(m.value for m in shuffle_labels(data, 1).measurements),
                         sorted(m.value for m in data.measurements))

    def test_it_permutes_within_slot_and_draw(self):
        """The group must be one execution of one configuration. Permuting
        across draws would move values between draws, which changes more
        than the correspondence."""
        data = _data(noise=0.01)
        shuffled = shuffle_labels(data, seed=1)
        for slot in ("u0", "u1"):
            for draw in data.draws:
                before = sorted(m.value for m in data.measurements
                                if m.slot == slot and m.draw == draw)
                after = sorted(m.value for m in shuffled.measurements
                               if m.slot == slot and m.draw == draw)
                self.assertEqual(before, after, f"{slot}/{draw} multiset changed")

    def test_no_key_collisions_are_introduced(self):
        """A collision would let a model 'notice' the shuffle through lost
        data rather than through having used labels."""
        shuffled = shuffle_labels(_data(noise=0.01), seed=1)
        keys = {(m.slot, m.label, m.draw) for m in shuffled.measurements}
        self.assertEqual(len(keys), len(shuffled.measurements))

    def test_a_single_label_group_is_left_alone(self):
        data = FitData([Measurement("u0", "XX", 0.5, 0.01, 100, 0)])
        self.assertEqual(shuffle_labels(data, 1).measurements[0].label, "XX")


class DiagnosticStrengthTest(unittest.TestCase):
    """A limitation found by testing, now encoded rather than assumed away."""

    def test_two_labels_cannot_discriminate(self):
        data = FitData([Measurement("u0", label, 0.5, 0.01, 100, 0)
                        for label in ("XX", "YY")])
        diagnostic, why = shuffle_is_diagnostic(data)
        self.assertFalse(diagnostic)
        self.assertIn("systematic relabelling", why)

    def test_three_labels_can(self):
        self.assertTrue(shuffle_is_diagnostic(_data())[0])

    def test_one_label_cannot(self):
        data = FitData([Measurement("u0", "XX", 0.5, 0.01, 100, 0)])
        self.assertFalse(shuffle_is_diagnostic(data)[0])


class OtherPerturbationTest(unittest.TestCase):
    def test_flip_sign_negates_every_value(self):
        data = _data()
        self.assertEqual([m.value for m in flip_sign(data).measurements],
                         [-m.value for m in data.measurements])

    def test_resample_uses_sigma(self):
        data = _data()
        self.assertNotEqual([m.value for m in resample_shots(data, 1).measurements],
                            [m.value for m in data.measurements])

    def test_resample_leaves_a_measurement_with_no_uncertainty_alone(self):
        data = FitData([Measurement("u0", "XX", 0.5, sigma=0.0, shots=0)])
        self.assertEqual(resample_shots(data, 1).measurements[0].value, 0.5)

    def test_subsample_reduces_the_draws(self):
        data = _data(draws=4)
        self.assertEqual(len(subsample_draws(data, 2, seed=1).draws), 2)

    def test_subsampling_more_than_available_is_a_no_op(self):
        data = _data(draws=2)
        self.assertEqual(len(subsample_draws(data, 9, seed=1).draws), 2)

    def test_subsampling_to_nothing_is_rejected(self):
        with self.assertRaises(ReconstructionError):
            subsample_draws(_data(), 0)


class CompareFitTest(unittest.TestCase):
    """The core discrimination: does corrupting the data hurt the model?"""

    def test_a_genuine_model_notices_a_shuffle(self):
        data = _data(noise=0.01)
        comparison = compare_fit(GenuineModel(), data, shuffle_labels(data, 1))
        self.assertGreater(comparison.ratio, 5.0)
        self.assertTrue(comparison.model_noticed)

    def test_a_flexible_model_does_not(self):
        data = _data(noise=0.01)
        comparison = compare_fit(FlexibleModel(), data, shuffle_labels(data, 1))
        self.assertLess(comparison.ratio, 5.0)

    def test_the_flexible_model_fits_real_data_BETTER(self):
        """Which is why fit quality alone cannot be the test."""
        data = _data(noise=0.01)
        flexible = FlexibleModel()
        genuine = GenuineModel()
        self.assertLess(flexible.goodness_of_fit(flexible.fit(data), data),
                        genuine.goodness_of_fit(genuine.fit(data), data))

    def test_a_pipeline_that_crashes_on_corruption_counts_as_noticing(self):
        class Fragile(GenuineModel):
            def fit(self, data):
                if any(m.label == "YY" and m.value > 0 for m in data.measurements):
                    raise ValueError("unexpected sign")
                return super().fit(data)

        data = _data(noise=0.01)
        comparison = compare_fit(Fragile(), data, flip_sign(data))
        self.assertEqual(comparison.ratio, float("inf"))
        self.assertIn("loud failure", comparison.detail)

    def test_a_pipeline_that_fails_on_REAL_data_is_an_error_not_a_pass(self):
        class Broken:
            def fit(self, data):
                raise RuntimeError("boom")

            def reconstruct(self, fit, data):
                return 0.0

            def goodness_of_fit(self, fit, data):
                return 0.0

        data = _data()
        with self.assertRaises(ReconstructionError):
            compare_fit(Broken(), data, shuffle_labels(data, 1))


class SpreadTest(unittest.TestCase):
    def test_spread_is_measured_across_trials(self):
        data = _data(noise=0.01)
        _, spread = reconstruction_spread(
            GenuineModel(), data, lambda d, t: resample_shots(d, seed=t), trials=6)
        self.assertGreaterEqual(spread, 0.0)

    def test_too_few_successes_is_an_error_not_a_number(self):
        class AlwaysFails:
            def fit(self, data):
                raise RuntimeError("no")

            def reconstruct(self, fit, data):
                return 0.0

            def goodness_of_fit(self, fit, data):
                return 0.0

        with self.assertRaises(ReconstructionError):
            reconstruction_spread(AlwaysFails(), _data(),
                                  lambda d, t: resample_shots(d, seed=t))


if __name__ == "__main__":
    unittest.main()


class ScrambleTest(unittest.TestCase):
    def test_values_are_replaced_within_the_observed_range(self):
        from qem_auditor.reconstruct import scramble

        data = _data(noise=0.01)
        scrambled = scramble(data, seed=1)
        low = min(m.value for m in data.measurements)
        high = max(m.value for m in data.measurements)
        for m in scrambled.measurements:
            self.assertGreaterEqual(m.value, low)
            self.assertLessEqual(m.value, high)

    def test_it_actually_changes_the_data(self):
        from qem_auditor.reconstruct import scramble

        data = _data(noise=0.01)
        self.assertNotEqual([m.value for m in scramble(data, 1).measurements],
                            [m.value for m in data.measurements])

    def test_structure_is_preserved(self):
        """Only the values are garbage; slots, labels and draws stay put."""
        from qem_auditor.reconstruct import scramble

        data = _data(noise=0.01)
        scrambled = scramble(data, 1)
        self.assertEqual([(m.slot, m.label, m.draw) for m in scrambled.measurements],
                         [(m.slot, m.label, m.draw) for m in data.measurements])


class DegeneracyScanTest(unittest.TestCase):
    """The statistic is dependence on the DATA, not distance to the truth --
    an auditor holding the true answer would not need to be an auditor."""

    class _Degenerate:
        """Slides toward a fixed answer as the parameter goes to its floor."""

        def evaluate_at(self, name, value, data):
            measured = mean([m.value for m in data.measurements])
            return value * measured + (1.0 - value) * 42.0

    class _Honest:
        """Keeps depending on the data at every parameter value."""

        def evaluate_at(self, name, value, data):
            return mean([m.value for m in data.measurements]) * (1.0 + value)

    def test_a_degenerate_method_is_caught(self):
        from qem_auditor.reconstruct import degeneracy_scan

        scan = degeneracy_scan(self._Degenerate(), _data(noise=0.05),
                               "radius", floor=0.0, nominal=1.0)
        self.assertTrue(scan.degenerates)
        self.assertAlmostEqual(scan.divergence[0], 0.0, places=9)

    def test_an_honest_method_is_not(self):
        from qem_auditor.reconstruct import degeneracy_scan

        scan = degeneracy_scan(self._Honest(), _data(noise=0.05),
                               "radius", floor=0.5, nominal=1.0)
        self.assertFalse(scan.degenerates)

    def test_a_failing_evaluation_is_skipped_not_counted_as_zero(self):
        from qem_auditor.reconstruct import degeneracy_scan

        class Fragile:
            def evaluate_at(self, name, value, data):
                if value < 0.5:
                    raise ValueError("undefined below 0.5")
                return mean([m.value for m in data.measurements]) * value

        scan = degeneracy_scan(Fragile(), _data(noise=0.05), "r", 0.0, 1.0, steps=5)
        self.assertLess(len(scan.values), 5)
        self.assertFalse(scan.degenerates)


class NoiseEnvelopeTest(unittest.TestCase):
    class _Robust:
        def noise_parameters(self):
            return {"p": (0.99, 1.01)}

        def evaluate_under_noise(self, params, data):
            return mean([m.value for m in data.measurements]) / params["p"]

    class _Fragile:
        """Genuinely fragile: the assumed parameter can approach zero, and
        the correction divides by it."""

        def noise_parameters(self):
            return {"p": (0.02, 0.4)}

        def evaluate_under_noise(self, params, data):
            return mean([m.value for m in data.measurements]) / params["p"]

    def test_a_robust_method_has_a_tight_envelope(self):
        from qem_auditor.reconstruct import noise_envelope

        nominal, q95, samples = noise_envelope(self._Robust(), _data(noise=0.01))
        self.assertLess(q95 / abs(nominal), 0.1)
        self.assertGreater(len(samples), 1)

    def test_a_fragile_method_has_a_wide_one(self):
        from qem_auditor.reconstruct import noise_envelope

        nominal, q95, _ = noise_envelope(self._Fragile(), _data(noise=0.01),
                                         draws=64)
        self.assertGreater(q95 / abs(nominal), 1.0)

    def test_deviation_is_measured_against_the_nominal_not_a_truth(self):
        """The auditor has no absolute target and must not pretend to one."""
        from qem_auditor.reconstruct import noise_envelope

        nominal, q95, _ = noise_envelope(self._Robust(), _data(noise=0.01))
        self.assertGreater(abs(nominal), 0.0)
        self.assertGreaterEqual(q95, 0.0)

    def test_no_parameters_is_an_error(self):
        from qem_auditor.reconstruct import ReconstructionError, noise_envelope

        class Bare:
            def noise_parameters(self):
                return {}

            def evaluate_under_noise(self, params, data):
                return 0.0

        with self.assertRaises(ReconstructionError):
            noise_envelope(Bare(), _data())
