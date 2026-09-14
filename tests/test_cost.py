"""Pricing a method against a provider's real rate card.

The card under test is IonQ's own, from `GET /jobs/estimate`:
a $25.7899 minimum charged once per JOB, $0.000164 per one-qubit gate
per shot, $0.001121 per two-qubit gate per shot. The three quotes that
endpoint returned are reproduced here as fixtures, because a pricing
model that cannot reproduce the prices it came from is a guess.
"""
import unittest

from qem_auditor.cost import (RATE_CARDS, RateCard, affordable_methods,
                              circuits_for)

#: The real circuit behind this module, as EXECUTED on hardware.
ONE_Q, TWO_Q = 120, 11


class RateCardTest(unittest.TestCase):

    def setUp(self):
        self.rates = RATE_CARDS["ionq_forte"]

    def test_it_reproduces_the_providers_own_quotes(self):
        """51/11 gates at 1 shot, at 10,000 shots, and 125 circuits'
        worth merged into one job. All three came from the endpoint."""
        self.assertAlmostEqual(self.rates.quote(1, 51, 11, 1), 25.79, places=2)
        self.assertAlmostEqual(self.rates.quote(1, 51, 11, 10_000), 206.95,
                               places=2)
        self.assertAlmostEqual(self.rates.quote(125, 51, 11, 10_000),
                               25_868.75, places=2)

    def test_the_minimum_is_per_job_not_per_circuit(self):
        """The 125x quote is what settles this: merged into one job it
        came back at exactly 125.0x the one-circuit price, which only
        holds if the floor was never charged 125 times."""
        one = self.rates.quote(1, 51, 11, 10_000)
        merged = self.rates.quote(125, 51, 11, 10_000)
        self.assertAlmostEqual(merged / one, 125.0, places=6)

    def test_separate_jobs_pay_the_minimum_every_time(self):
        """Below the floor, batching is real money. Above it, nothing."""
        cheap_batched = self.rates.quote(5, ONE_Q, TWO_Q, 100, one_job=True)
        cheap_separate = self.rates.quote(5, ONE_Q, TWO_Q, 100, one_job=False)
        self.assertAlmostEqual(cheap_batched, 25.79, places=2)
        self.assertAlmostEqual(cheap_separate, 5 * 25.79, places=2)

        dear_batched = self.rates.quote(5, ONE_Q, TWO_Q, 10_000, one_job=True)
        dear_separate = self.rates.quote(5, ONE_Q, TWO_Q, 10_000, one_job=False)
        self.assertAlmostEqual(dear_batched, dear_separate, places=6)


class TwoRegimesTest(unittest.TestCase):
    """The correction this module needed. Two real jobs at 100 and 500
    shots both billed $25.79, and reading that as "cost is per circuit,
    shots are free" is wrong: both sat under the job minimum."""

    def setUp(self):
        self.rates = RATE_CARDS["ionq_forte"]

    def test_the_two_real_bills_reproduce(self):
        self.assertAlmostEqual(self.rates.quote(1, ONE_Q, TWO_Q, 100), 25.79,
                               places=2)
        self.assertAlmostEqual(self.rates.quote(1, ONE_Q, TWO_Q, 500), 25.79,
                               places=2)

    def test_the_break_even_is_computed_not_assumed(self):
        self.assertEqual(self.rates.break_even_shots(ONE_Q, TWO_Q), 806)

    def test_past_the_break_even_shots_are_not_free(self):
        """The question that caught the original error: at 2,000 shots
        does the money go up? It does -- 2.5x the floor."""
        self.assertAlmostEqual(self.rates.quote(1, ONE_Q, TWO_Q, 2_000), 64.02,
                               places=2)
        self.assertAlmostEqual(self.rates.quote(1, ONE_Q, TWO_Q, 10_000),
                               320.11, places=2)

    def test_cost_is_linear_in_shots_once_the_floor_stops_binding(self):
        at_2k = self.rates.quote(1, ONE_Q, TWO_Q, 2_000)
        at_4k = self.rates.quote(1, ONE_Q, TWO_Q, 4_000)
        self.assertAlmostEqual(at_4k / at_2k, 2.0, places=6)

    def test_compilation_is_a_price_and_not_only_an_error_budget(self):
        """Written with 20 one-qubit gates, executed with 120. Cost is
        charged per gate per shot, so the same blowup that worsened the
        error budget doubled the bill and halved the break-even."""
        written = self.rates.per_shot(20, TWO_Q)
        executed = self.rates.per_shot(ONE_Q, TWO_Q)
        self.assertAlmostEqual(executed / written, 2.05, places=2)
        self.assertEqual(self.rates.break_even_shots(20, TWO_Q), 1653)
        self.assertEqual(self.rates.break_even_shots(ONE_Q, TWO_Q), 806)

    def test_a_gateless_circuit_has_no_break_even(self):
        self.assertIsNone(self.rates.break_even_shots(0, 0))


class CircuitCountTest(unittest.TestCase):

    def test_post_selection_costs_no_extra_circuits(self):
        self.assertEqual(
            circuits_for("symmetry verification (post-selection)").circuits, 1)

    def test_post_selection_costs_shots_instead(self):
        """Measured on three real trapped-ion circuits: 91.5/90.2/90.1%."""
        cost = circuits_for("symmetry verification (post-selection)")
        self.assertEqual(cost.effective_shots(2000), 1800)

    def test_pec_is_orders_of_magnitude_more_expensive_than_zne(self):
        pec = circuits_for("probabilistic error cancellation (PEC)")
        zne = circuits_for("zero-noise extrapolation (ZNE)")
        self.assertGreater(pec.circuits / zne.circuits, 100)

    def test_the_full_confusion_matrix_grows_with_width(self):
        self.assertEqual(
            circuits_for("REM (full confusion matrix)", qubits=6).circuits,
            1 + 64)

    def test_vncdr_pays_for_both_assumptions(self):
        cdr = circuits_for("Clifford data regression (CDR)", training=5)
        vncdr = circuits_for("vnCDR", training=5, folds=3)
        self.assertEqual(vncdr.circuits, 3 * cdr.circuits)

    def test_an_unknown_method_refuses_rather_than_defaulting_to_one(self):
        with self.assertRaises(KeyError) as caught:
            circuits_for("a method nobody priced")
        self.assertIn("rather than defaulting", str(caught.exception))


class AffordabilityTest(unittest.TestCase):

    def setUp(self):
        self.methods = ["unmitigated", "zero-noise extrapolation (ZNE)",
                        "Clifford data regression (CDR)",
                        "probabilistic error cancellation (PEC)"]

    def report(self, shots):
        return affordable_methods(self.methods, budget_usd=170.0, shots=shots,
                                  one_qubit_gates=ONE_Q, two_qubit_gates=TWO_Q)

    def test_shots_change_what_is_affordable_above_the_break_even(self):
        """The bug this replaces asserted the opposite -- that a 20x
        shot increase changed nothing -- because it had no notion of a
        floor to cross."""
        cheap = {q.cost.method for q in self.report(500).affordable}
        dear = {q.cost.method for q in self.report(10_000).affordable}
        self.assertIn("Clifford data regression (CDR)", cheap)
        self.assertNotIn("Clifford data regression (CDR)", dear)

    def test_under_the_floor_shots_really_are_free(self):
        """Only for the methods actually under it. One circuit at 100
        and at 500 shots bills identically; that is the pair of real
        jobs this module started from."""
        single = "unmitigated"
        at_100 = next(q for q in self.report(100).quotes
                      if q.cost.method == single)
        at_500 = next(q for q in self.report(500).quotes
                      if q.cost.method == single)
        self.assertEqual(at_100.usd, at_500.usd)
        self.assertTrue(at_100.floor_binds)

    def test_the_break_even_divides_by_the_circuit_count(self):
        """Circuits in one job share one minimum, so a method needing
        many of them leaves the free regime almost immediately. PEC's
        500 circuits are past it after two shots, which is why it is
        never cheap however few shots you take."""
        rates = self.report(100).rates
        self.assertEqual(rates.break_even_shots(ONE_Q, TWO_Q, circuits=1), 806)
        self.assertEqual(rates.break_even_shots(ONE_Q, TWO_Q, circuits=500), 2)
        pec = next(q for q in self.report(100).quotes
                   if q.cost.method.startswith("probabilistic"))
        self.assertFalse(pec.floor_binds)
        self.assertGreater(pec.usd, 1_000)

    def test_the_regime_is_named_in_the_description(self):
        self.assertIn("more shots are genuinely free",
                      self.report(100).describe())
        over = self.report(2_000).describe()
        self.assertIn("stops being true", over)
        self.assertIn("2.5x", over)

    def test_a_quote_says_which_regime_it_is_in(self):
        self.assertTrue(self.report(100).quotes[0].floor_binds)
        self.assertFalse(self.report(10_000).quotes[0].floor_binds)

    def test_a_different_rate_card_changes_the_answer(self):
        """Nothing here is a fact about the methods."""
        flat = RateCard(job_minimum_usd=0.0, one_qubit_usd=0.0,
                        two_qubit_usd=0.0, as_of="hypothetical",
                        source="a free machine, for contrast")
        report = affordable_methods(["probabilistic error cancellation (PEC)"],
                                    budget_usd=1.0, shots=10_000,
                                    one_qubit_gates=ONE_Q, two_qubit_gates=TWO_Q,
                                    rates=flat)
        self.assertEqual(len(report.affordable), 1)


if __name__ == "__main__":
    unittest.main()
