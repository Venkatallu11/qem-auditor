"""Pricing a method in circuits, against a real bill.

The numbers under test come from two real submissions to IonQ
`qpu.forte-enterprise-1`: 100 shots billed $25.79, and 500 shots of the
same circuit billed the same. What that implies -- cost is per-circuit,
not per-shot -- is what these tests pin.
"""
import unittest

from qem_auditor.cost import (PRICING, Pricing, affordable_methods,
                              circuits_for)


class PricingTest(unittest.TestCase):

    def setUp(self):
        self.pricing = PRICING["ionq_forte_enterprise"]

    def test_the_measured_pair_reproduces(self):
        """100 and 500 shots of one circuit come out identical, because
        that is what the bill did."""
        cheap = self.pricing.quote(circuits=1, shots=100)
        dear = self.pricing.quote(circuits=1, shots=500)
        self.assertAlmostEqual(cheap, 25.79, places=2)
        self.assertEqual(cheap, dear)

    def test_the_per_shot_term_is_a_bound_not_a_measurement(self):
        """Recording it as zero would claim a measurement nobody made."""
        self.assertGreater(self.pricing.per_shot_usd_bound, 0)
        self.assertIn("not measured", self.pricing.source)

    def test_the_bound_comes_from_what_was_RECORDED_not_what_was_billed(self):
        """The second bill was written down as roughly $25, cents and
        all unrecorded. Claiming a cent-resolution bound would be a
        hundredfold over-claim off one missing decimal place."""
        self.assertAlmostEqual(self.pricing.per_shot_usd_bound, 1.00 / 400,
                               places=6)

    def test_a_known_per_shot_price_is_priced_rather_than_ignored(self):
        """The mirror of the bug above: a machine that really does
        charge per shot must not quote zero because the per-CIRCUIT
        term is all that was measured on a different machine."""
        metered = Pricing(per_circuit_usd=1.0, per_shot_usd_bound=1e-3,
                          per_shot_usd=1e-3, as_of="hypothetical",
                          source="a measured per-shot price")
        self.assertAlmostEqual(metered.quote(1, 2000), 3.0)
        self.assertAlmostEqual(metered.upper(1, 2000), 3.0)

    def test_the_bound_is_never_folded_into_the_price(self):
        """An upper bound reported as a price is a number larger than
        anything anyone was billed, inherited by everything downstream."""
        quoted = self.pricing.quote(1, 2000)
        upper = self.pricing.upper(1, 2000)
        self.assertEqual(quoted, 25.79)
        self.assertGreater(upper, quoted)

    def test_a_full_reconstruction_is_priced_in_circuits(self):
        """21 slots x 13 groups, the real experiment that was refused."""
        self.assertAlmostEqual(self.pricing.quote(273, 2000), 7040.67,
                               delta=1.0)


class CircuitCountTest(unittest.TestCase):

    def test_post_selection_costs_no_extra_circuits(self):
        cost = circuits_for("symmetry verification (post-selection)")
        self.assertEqual(cost.circuits, 1)

    def test_post_selection_costs_shots_instead(self):
        """Measured on three real trapped-ion circuits: 91.5/90.2/90.1%."""
        cost = circuits_for("symmetry verification (post-selection)")
        self.assertEqual(cost.effective_shots(2000), 1800)

    def test_pec_is_orders_of_magnitude_more_expensive_than_zne(self):
        pec = circuits_for("probabilistic error cancellation (PEC)")
        zne = circuits_for("zero-noise extrapolation (ZNE)")
        self.assertGreater(pec.circuits / zne.circuits, 100)

    def test_the_full_confusion_matrix_grows_with_width(self):
        narrow = circuits_for("REM (full confusion matrix)", qubits=2)
        wide = circuits_for("REM (full confusion matrix)", qubits=6)
        self.assertEqual(narrow.circuits, 1 + 4)
        self.assertEqual(wide.circuits, 1 + 64)

    def test_vncdr_pays_for_both_assumptions(self):
        cdr = circuits_for("Clifford data regression (CDR)", training=5)
        vncdr = circuits_for("vnCDR", training=5, folds=3)
        self.assertEqual(vncdr.circuits, 3 * cdr.circuits)

    def test_an_unknown_method_refuses_rather_than_defaulting_to_one(self):
        """Defaulting is how a prescription recommends something nobody
        can afford."""
        with self.assertRaises(KeyError) as caught:
            circuits_for("a method nobody priced")
        self.assertIn("rather than defaulting", str(caught.exception))


class AffordabilityTest(unittest.TestCase):

    def setUp(self):
        self.methods = ["unmitigated", "zero-noise extrapolation (ZNE)",
                        "Clifford data regression (CDR)",
                        "probabilistic error cancellation (PEC)"]
        self.report = affordable_methods(self.methods, budget_usd=170.0,
                                         shots=2000)

    def test_the_budget_that_was_actually_authorised(self):
        affordable = {q.cost.method for q in self.report.affordable}
        self.assertIn("unmitigated", affordable)
        self.assertIn("Clifford data regression (CDR)", affordable)
        self.assertNotIn("probabilistic error cancellation (PEC)", affordable)

    def test_shots_do_not_change_what_is_affordable(self):
        """The finding, stated as a test: on a per-circuit-priced device
        a 20x shot increase changes nothing about what fits."""
        richer = affordable_methods(self.methods, budget_usd=170.0,
                                    shots=40_000)
        self.assertEqual({q.cost.method for q in richer.affordable},
                         {q.cost.method for q in self.report.affordable})

    def test_the_best_method_in_the_benchmarks_nearly_exhausts_the_budget(self):
        cdr = next(q for q in self.report.quotes
                   if q.cost.method == "Clifford data regression (CDR)")
        self.assertGreater(cdr.usd / 170.0, 0.85)

    def test_the_description_says_cost_is_per_circuit(self):
        self.assertIn("per-CIRCUIT", self.report.describe())

    def test_a_different_pricing_model_changes_the_answer(self):
        """Nothing here is a fact about the methods. On a machine that
        charges per shot the ordering inverts."""
        per_shot = Pricing(per_circuit_usd=0.0, per_shot_usd_bound=1e-3,
                           per_shot_usd=1e-3, as_of="hypothetical",
                           source="a per-shot price, for contrast")
        report = affordable_methods(["unmitigated"], budget_usd=1.0,
                                    shots=2000, pricing=per_shot)
        self.assertGreater(report.quotes[0].usd, 1.0)


if __name__ == "__main__":
    unittest.main()
