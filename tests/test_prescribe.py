"""From a refusal to a better experiment.

The properties tested here are the ones that separate advice from
folklore: that the recommendation follows from where the error IS, that
a method which cannot physically reach the dominant term is refused
however famous it is, and that an estimated budget is not allowed to
make quantitative promises.
"""
import unittest

from qem_auditor import Provenance
from qem_auditor.prescribe import (CATALOGUE, METHODS_BY_NAME, ErrorBudget,
                                   feasibility,
                                   ErrorSource, Scaling, SCALES_AS,
                                   budget_from_calibration, prescribe)

E = ErrorSource


def budget(contributions, provenance=Provenance.MEASURED):
    return ErrorBudget(contributions, provenance)


class BudgetTest(unittest.TestCase):
    def test_a_negative_contribution_is_refused(self):
        """An error budget with a negative term is a subtraction that went
        wrong, not a measurement."""
        with self.assertRaises(ValueError):
            ErrorBudget({E.READOUT: -1.0})

    def test_shares_and_ranking(self):
        b = budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 10.0})
        self.assertIs(b.dominant, E.READOUT)
        self.assertAlmostEqual(b.share(E.READOUT), 0.75)

    def test_two_close_leaders_are_not_decisive(self):
        """Two terms within 1.5x do not decide between methods that reach
        one each, and pretending they do is where invented confidence
        starts."""
        self.assertFalse(budget({E.READOUT: 10.0, E.GATE_STOCHASTIC: 9.0}).is_decisive)
        self.assertTrue(budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 9.0}).is_decisive)

    def test_shot_noise_and_ansatz_are_outside_what_mitigation_can_reach(self):
        b = budget({E.SHOT_NOISE: 50.0, E.GATE_STOCHASTIC: 50.0})
        self.assertAlmostEqual(b.reachable_ceiling, 0.5)
        self.assertAlmostEqual(b.best_possible_gain, 2.0)

    def test_an_empty_budget_has_no_dominant_term(self):
        self.assertIsNone(ErrorBudget({}).dominant)


class ScalingTest(unittest.TestCase):
    """The physics that decides what extrapolation can touch."""

    def test_readout_error_is_constant_under_folding(self):
        self.assertIs(SCALES_AS[E.READOUT], Scaling.CONSTANT)

    def test_gate_error_grows_with_gate_count(self):
        self.assertIs(SCALES_AS[E.GATE_STOCHASTIC], Scaling.WITH_GATE_COUNT)

    def test_every_source_has_a_scaling(self):
        for source in E:
            self.assertIn(source, SCALES_AS)

    def test_zne_does_not_claim_to_reach_anything_constant(self):
        """The whole finding, encoded: folding cannot reach a term that
        does not respond to folding."""
        zne = METHODS_BY_NAME["zero-noise extrapolation (ZNE)"]
        for source in zne.reaches:
            self.assertIsNot(SCALES_AS[source], Scaling.CONSTANT)


class PrescriptionTest(unittest.TestCase):
    def test_readout_dominated_does_not_recommend_zne_first(self):
        """The measured case, and the one everyone gets wrong: ZNE is the
        method people reach for, and it is the wrong one here."""
        consult = prescribe(budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 3.0}))
        self.assertIsNotNone(consult.leading)
        self.assertNotEqual(consult.leading.action, "zero-noise extrapolation (ZNE)")
        demoted = ([n for n, _ in consult.will_not_help]
                   + [p.action for p in consult.marginal])
        self.assertIn("zero-noise extrapolation (ZNE)", demoted)

    def test_gate_dominated_does_recommend_zne(self):
        consult = prescribe(budget({E.GATE_STOCHASTIC: 18.0, E.DECOHERENCE: 3.0}))
        self.assertIn("zero-noise extrapolation (ZNE)",
                      [p.action for p in consult.prescriptions])

    def test_the_reason_names_the_mechanism_not_a_preference(self):
        consult = prescribe(budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 3.0}))
        text = consult.format_consult()
        self.assertIn("extrapolation cannot reach it", text)

    def test_shot_noise_dominated_says_take_more_shots_not_mitigate(self):
        consult = prescribe(budget({E.SHOT_NOISE: 20.0, E.GATE_STOCHASTIC: 2.0}))
        self.assertEqual(consult.leading.action, "more shots")
        self.assertTrue(any("more shots" in action for action, _ in consult.structural))

    def test_an_inadequate_ansatz_gets_no_mitigation_at_all(self):
        """Mitigation removes noise; it cannot add expressiveness."""
        consult = prescribe(budget({E.ANSATZ: 40.0, E.GATE_STOCHASTIC: 2.0}))
        self.assertEqual(consult.prescriptions, ())
        self.assertTrue(any("Change the circuit" in action
                            for action, _ in consult.structural))

    def test_pec_is_withheld_until_its_one_assumption_is_checked(self):
        """It does not degrade when the model is wrong, it inverts."""
        b = budget({E.GATE_STOCHASTIC: 20.0})
        withheld = prescribe(b, noise_model_verified=False)
        self.assertIn("probabilistic error cancellation (PEC)",
                      [n for n, _ in withheld.will_not_help])
        allowed = prescribe(b, noise_model_verified=True)
        self.assertIn("probabilistic error cancellation (PEC)",
                      [p.action for p in allowed.prescriptions])

    def test_symmetry_verification_needs_the_caller_to_assert_a_symmetry(self):
        """Whether a state obeys a checkable symmetry is a fact about the
        physics that no error budget reveals."""
        b = budget({E.READOUT: 20.0, E.GATE_STOCHASTIC: 10.0})
        without = [n for n, _ in prescribe(b).will_not_help]
        self.assertIn("symmetry verification (post-selection)", without)
        with_symmetry = [p.action for p in
                         prescribe(b, symmetry_available=True).prescriptions]
        self.assertIn("symmetry verification (post-selection)", with_symmetry)

    def test_a_method_reaching_little_is_marginal_not_recommended(self):
        """Listing a 1.2x method beside a 20x one invites the wrong run.

        At 8 of 38 the gate term is real -- too big to dismiss, too small
        to be worth three times the shots -- which is the band `marginal`
        exists for. Below 15% a method is refused outright instead.
        """
        consult = prescribe(budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 8.0}))
        self.assertIn("zero-noise extrapolation (ZNE)",
                      [p.action for p in consult.marginal])
        self.assertNotIn("zero-noise extrapolation (ZNE)",
                         [p.action for p in consult.prescriptions])

    def test_below_the_marginal_band_a_method_is_refused_outright(self):
        consult = prescribe(budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 3.0}))
        self.assertIn("zero-noise extrapolation (ZNE)",
                      [n for n, _ in consult.will_not_help])

    def test_every_refusal_says_why(self):
        consult = prescribe(budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 3.0}))
        for name, reason in consult.will_not_help:
            with self.subTest(method=name):
                self.assertGreater(len(reason.strip()), 20)

    def test_an_empty_budget_recommends_measuring_one(self):
        consult = prescribe(ErrorBudget({}))
        self.assertEqual(consult.prescriptions, ())
        self.assertTrue(any("measure" in c for c in consult.caveats))


class EstimateHonestyTest(unittest.TestCase):
    """An estimate supports an ordering, not a number."""

    def test_an_estimated_budget_quotes_no_best_case(self):
        estimated = budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 5.0},
                           Provenance.SELF_REPORTED)
        for prescription in prescribe(estimated).prescriptions:
            with self.subTest(method=prescription.action):
                self.assertIsNone(prescription.best_case)

    def test_a_measured_budget_does(self):
        for prescription in prescribe(budget({E.READOUT: 30.0,
                                              E.GATE_STOCHASTIC: 5.0})).prescriptions:
            with self.subTest(method=prescription.action):
                self.assertIsNotNone(prescription.best_case)

    def test_an_estimated_budget_carries_a_caveat_saying_so(self):
        consult = prescribe(budget({E.READOUT: 30.0}, Provenance.SELF_REPORTED))
        self.assertTrue(any("not measured" in c for c in consult.caveats))


class CalibrationEstimateTest(unittest.TestCase):
    def test_it_finds_readout_dominant_on_the_measured_kyiv_numbers(self):
        """The check that licenses using this on hardware: it reaches the
        same conclusion as the ablation, which needs an exact answer."""
        estimated = budget_from_calibration(
            two_qubit_gates=2, one_qubit_gates=6, measured_qubits=2,
            two_qubit_error=0.0031126500103701993,
            one_qubit_error=0.00012952268350115682,
            readout_error=0.029296875, shots=40_000)
        self.assertIs(estimated.dominant, E.READOUT)

    def test_it_is_marked_as_an_estimate_not_a_measurement(self):
        estimated = budget_from_calibration(
            two_qubit_gates=2, one_qubit_gates=6, measured_qubits=2,
            two_qubit_error=0.01, one_qubit_error=0.001,
            readout_error=0.02, shots=1000)
        self.assertIs(estimated.provenance, Provenance.SELF_REPORTED)

    def test_shot_noise_shares_a_scale_with_the_other_terms(self):
        """Quoting it as an absolute error while everything else was a
        fraction inflated its share; caught by the prescription check."""
        estimated = budget_from_calibration(
            two_qubit_gates=2, one_qubit_gates=6, measured_qubits=2,
            two_qubit_error=0.003, one_qubit_error=0.0001,
            readout_error=0.03, shots=40_000)
        self.assertAlmostEqual(
            estimated.contributions[E.SHOT_NOISE], 1 / 40_000 ** 0.5)

    def test_more_shots_shrink_the_shot_noise_term(self):
        def estimate(shots):
            return budget_from_calibration(
                two_qubit_gates=2, one_qubit_gates=6, measured_qubits=2,
                two_qubit_error=0.003, one_qubit_error=0.0001,
                readout_error=0.03, shots=shots).share(E.SHOT_NOISE)

        self.assertLess(estimate(1_000_000), estimate(10_000))

    def test_an_impossible_error_rate_is_refused(self):
        with self.assertRaises(ValueError):
            budget_from_calibration(
                two_qubit_gates=1, one_qubit_gates=1, measured_qubits=1,
                two_qubit_error=1.5, one_qubit_error=0.0,
                readout_error=0.0, shots=100)


class CatalogueTest(unittest.TestCase):
    def test_every_mitigation_method_cites_measured_evidence(self):
        """A catalogue of folklore would rank exactly as confidently.

        "more shots" is exempt and is the only exemption: that shot noise
        falls as 1/sqrt(N) is a fact about sampling, and demanding an
        experiment for it would be theatre.
        """
        for method in CATALOGUE:
            if method.name == "more shots":
                continue
            with self.subTest(method=method.name):
                self.assertIn("measured", method.evidence)

    def test_the_one_exemption_cites_the_reason_it_needs_no_measurement(self):
        self.assertIn("1/sqrt(N)", METHODS_BY_NAME["more shots"].evidence)

    def test_every_method_states_what_it_assumes(self):
        for method in CATALOGUE:
            with self.subTest(method=method.name):
                self.assertTrue(method.assumes.strip())

    def test_partial_reach_is_declared_where_it_applies(self):
        symmetry = METHODS_BY_NAME["symmetry verification (post-selection)"]
        self.assertLess(symmetry.reach_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()


class FeasibilityTest(unittest.TestCase):
    """Ask whether there is a signal before ranking methods to improve it.

    Every method in the catalogue improves an estimate that exists. None
    creates one. Ranking methods for a circuit with 1e-8 survival is a
    precise answer to a question nobody can ask.
    """

    EAGLE = {"ecr_error": 0.00311, "readout_error": 0.0293}

    def test_a_shallow_circuit_leaves_something_to_mitigate(self):
        verdict = feasibility(465, self.EAGLE, n_qubits=18)
        self.assertTrue(verdict.is_mitigable)
        self.assertGreater(verdict.survival, 0.1)

    def test_a_deep_circuit_is_refused_rather_than_prescribed_for(self):
        verdict = feasibility(5898, self.EAGLE, n_qubits=18)
        self.assertFalse(verdict.is_mitigable)
        self.assertLess(verdict.survival, 1e-7)
        self.assertIn("no method mitigates this", verdict.format_verdict())

    def test_it_says_what_gate_count_would_be_affordable(self):
        """The actionable half. "Too deep" is a complaint; a number is a
        target, and it names compilation rather than mitigation as the
        thing that has to change."""
        verdict = feasibility(5898, self.EAGLE, n_qubits=18)
        affordable = verdict.affordable_two_qubit_gates
        self.assertLess(affordable, 5898)
        self.assertTrue(feasibility(affordable, self.EAGLE, 18).is_mitigable)
        self.assertFalse(feasibility(affordable + 50, self.EAGLE, 18).is_mitigable)

    def test_the_estimate_is_optimistic_on_purpose(self):
        """It counts gate and readout error only. A circuit this call
        declares hopeless is hopeless by a margin, since decoherence and
        crosstalk can only lower the number further."""
        verdict = feasibility(1000, self.EAGLE, n_qubits=18)
        self.assertAlmostEqual(verdict.survival,
                               verdict.gate_survival * verdict.readout_survival)

    def test_shots_needed_grows_as_the_inverse_square_of_survival(self):
        a = feasibility(1000, self.EAGLE, 18)
        b = feasibility(2000, self.EAGLE, 18)
        self.assertGreater(b.shots_for_a_signal, a.shots_for_a_signal)
        self.assertAlmostEqual(a.shots_for_a_signal * a.survival ** 2, 9.0)
