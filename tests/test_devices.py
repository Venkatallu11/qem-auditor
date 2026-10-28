"""Device profiles across vendors, and what actually differs between them.

Written after this project spent its whole life calibrated against one
IBM Eagle chip. The tempting claim -- "different hardware needs a
different method" -- is mostly false, and one of these tests exists to
keep it false rather than let it creep back in.
"""
import unittest

from qem_auditor.devices import (PROFILES, Architecture, DeviceProfile,
                                 budget_for, compare, profile)
from qem_auditor.prescribe import ErrorSource, prescribe


class ProfileTest(unittest.TestCase):

    def test_every_profile_is_dated_and_sourced(self):
        """A device number with no date is a claim with no evidence, and
        this package exists to object to those."""
        for key, device in PROFILES.items():
            self.assertTrue(device.as_of, key)
            self.assertTrue(device.source, key)
            self.assertGreater(len(device.source), 20, key)

    def test_both_vendors_families_are_represented(self):
        vendors = {device.vendor for device in PROFILES.values()}
        for expected in ("IBM", "IonQ", "Quantinuum", "Rigetti"):
            self.assertIn(expected, vendors)
        architectures = {device.architecture for device in PROFILES.values()}
        self.assertIn(Architecture.SUPERCONDUCTING, architectures)
        self.assertIn(Architecture.TRAPPED_ION, architectures)

    def test_an_unknown_profile_lists_the_alternatives(self):
        with self.assertRaises(KeyError) as caught:
            profile("ibm_nonexistent")
        message = str(caught.exception)
        self.assertIn("ibm_eagle", message)
        self.assertIn("your own calibration", message)

    def test_your_own_calibration_replaces_the_representative_one(self):
        mine = profile("ibm_eagle").replace(two_qubit_error=0.001,
                                            readout_error=0.004)
        self.assertAlmostEqual(mine.two_qubit_error, 0.001)
        self.assertAlmostEqual(profile("ibm_eagle").two_qubit_error, 0.00311,
                               msg="replace must not mutate the registry")

    def test_the_readout_to_gate_ratio_separates_the_architectures(self):
        """The single most useful number for choosing a method: IBM Eagle
        is ~9x readout-dominated, IonQ Aria is not readout-dominated at
        all. Same circuit, different binding constraint."""
        self.assertGreater(profile("ibm_eagle").readout_to_gate, 5)
        self.assertLess(profile("ionq_aria").readout_to_gate, 2)


class ConnectivityTest(unittest.TestCase):
    """The large, reliable vendor effect -- and the one an architecture
    comparison silently gets wrong if it budgets the circuit as written."""

    def test_all_to_all_hardware_runs_what_you_wrote(self):
        for key in ("ionq_aria", "ionq_forte", "quantinuum_h2"):
            device = profile(key)
            self.assertTrue(device.all_to_all, key)
            self.assertEqual(device.effective_two_qubit_gates(100), 100, key)

    def test_nearest_neighbour_hardware_runs_more(self):
        for key in ("ibm_eagle", "ibm_heron", "rigetti_ankaa"):
            device = profile(key)
            self.assertFalse(device.all_to_all, key)
            self.assertGreater(device.effective_two_qubit_gates(100), 100, key)

    def test_a_fully_local_circuit_pays_nothing_for_routing(self):
        self.assertEqual(
            profile("ibm_eagle").effective_two_qubit_gates(100, locality=1.0), 100)

    def test_locality_must_be_a_fraction(self):
        with self.assertRaises(ValueError):
            profile("ibm_eagle").routing_multiplier(10, locality=1.5)

    def test_routing_is_applied_before_budgeting(self):
        """Budgeting the written circuit would flatter every
        nearest-neighbour device in the comparison."""
        rows = compare([profile("ibm_eagle"), profile("ionq_forte")],
                       two_qubit_gates=100, measured_qubits=4)
        by_vendor = {row["device"].vendor: row for row in rows}
        self.assertEqual(by_vendor["IonQ"]["executed_two_qubit_gates"], 100)
        self.assertGreater(by_vendor["IBM"]["executed_two_qubit_gates"], 100)


class WhatActuallyDiffersTest(unittest.TestCase):

    SHALLOW_WIDE = dict(two_qubit_gates=10, measured_qubits=6,
                        one_qubit_gates=20, shots=10_000)

    def test_the_dominant_error_can_differ_between_vendors(self):
        """Shallow and wide is the regime where machines disagree: the
        same circuit is readout-limited on IBM and gate-limited on IonQ
        Aria and Rigetti."""
        rows = compare(list(PROFILES.values()), **self.SHALLOW_WIDE)
        dominant = {row["dominant"] for row in rows} - {None}
        self.assertIn("READOUT", dominant)
        self.assertIn("GATE_STOCHASTIC", dominant)

    def test_the_top_method_does_not_differ_and_that_is_the_finding(self):
        """The claim this module set out to make -- that the recommended
        method flips between vendors -- is FALSE, and the example that
        would have asserted it says so instead. Pinned so it cannot creep
        back: CDR covers enough sources to top the ranking everywhere.

        If a future catalogue change makes the top method genuinely
        vendor-dependent, this test fails and the write-up must be
        revised with it. That is the intended behaviour, not a nuisance.
        """
        rows = compare(list(PROFILES.values()), **self.SHALLOW_WIDE)
        tops = set()
        for row in rows:
            advice = prescribe(row["budget"], noise_model_verified=False,
                               symmetry_available=False, shots=10_000)
            tops.add(advice.prescriptions[0].action)
        self.assertEqual(len(tops), 1,
                         f"the top method now differs by vendor: {tops}")

    def test_the_budget_composition_differs_a_lot(self):
        """Which is where the vendor actually matters. Quantinuum is
        shot-noise limited, Eagle carries far more decoherence -- and
        those imply different actions even under the same top method."""
        rows = {row["device"].name: row["budget"]
                for row in compare(list(PROFILES.values()), **self.SHALLOW_WIDE)}
        self.assertGreater(rows["H2"].share(ErrorSource.SHOT_NOISE),
                           rows["Ankaa-3"].share(ErrorSource.SHOT_NOISE))
        self.assertGreater(rows["Ankaa-3"].share(ErrorSource.GATE_STOCHASTIC),
                           rows["H2"].share(ErrorSource.GATE_STOCHASTIC))

    def test_survival_ranks_the_machines_even_when_the_method_does_not(self):
        rows = compare(list(PROFILES.values()), two_qubit_gates=60,
                       measured_qubits=4, one_qubit_gates=120)
        survivals = [row["feasibility"].survival for row in rows]
        self.assertEqual(survivals, sorted(survivals, reverse=True))
        self.assertGreater(survivals[0], 3 * survivals[-1])


class BudgetTest(unittest.TestCase):

    def test_a_budget_is_produced_for_every_profile(self):
        for key in PROFILES:
            budget = budget_for(profile(key), two_qubit_gates=20,
                                measured_qubits=4)
            total = sum(budget.share(source) for source in ErrorSource)
            self.assertAlmostEqual(total, 1.0, places=6, msg=key)

    def test_a_custom_profile_works_without_being_registered(self):
        mine = DeviceProfile(
            name="bench", vendor="me", architecture=Architecture.NEUTRAL_ATOM,
            two_qubit_error=0.01, readout_error=0.01, one_qubit_error=0.001,
            qubits=10, all_to_all=True, native_two_qubit="cz",
            as_of="2026-09", source="measured on my own bench this morning")
        budget = budget_for(mine, two_qubit_gates=10, measured_qubits=3)
        self.assertGreater(budget.share(ErrorSource.GATE_STOCHASTIC), 0)


class MeasuredIonQCalibrationTest(unittest.TestCase):
    """The two Forte profiles come from IonQ's calibration endpoint, and
    the tiers trade rather than rank."""

    def setUp(self):
        self.forte = PROFILES["ionq_forte"]
        self.enterprise = PROFILES["ionq_forte_enterprise"]

    def test_both_carry_the_characterisation_date_not_a_month(self):
        """A calibration figure without its date is a transcription."""
        for profile in (self.forte, self.enterprise):
            self.assertEqual(profile.as_of, "2026-08-09")
            self.assertIn("get_latest_calibration", profile.source)

    def test_enterprise_is_not_uniformly_better(self):
        """2q fidelity 0.9909 against 0.9952, SPAM 0.9968 against 0.9939.
        Nearly twice the gate error and noticeably better readout."""
        self.assertGreater(self.enterprise.two_qubit_error,
                           self.forte.two_qubit_error)
        self.assertLess(self.enterprise.readout_error,
                        self.forte.readout_error)
        self.assertAlmostEqual(
            self.enterprise.two_qubit_error / self.forte.two_qubit_error,
            1.90, places=2)

    def test_the_fidelities_round_trip(self):
        """Stored as errors; these are the fidelities IonQ returned."""
        self.assertAlmostEqual(1 - self.forte.two_qubit_error, 0.9952, places=4)
        self.assertAlmostEqual(1 - self.enterprise.two_qubit_error, 0.9909,
                               places=4)
        self.assertAlmostEqual(1 - self.enterprise.readout_error, 0.9968,
                               places=4)

    def test_aria_is_left_as_an_estimate_on_purpose(self):
        """Its calibration record returns a null 2q fidelity and a 1q
        figure of 0.4745. An unusable measurement does not beat a stated
        estimate, and the profile says so."""
        self.assertIn("data-quality gap", PROFILES["ionq_aria"].source)

    def test_the_swap_changes_magnitudes_but_not_the_real_runs_ranking(self):
        """Checked rather than asserted: on the real 84-circuit run's
        average circuit, gate error leads under both cards."""
        from qem_auditor.prescribe import budget_from_calibration

        tops = []
        for profile in (self.forte, self.enterprise):
            budget = budget_from_calibration(
                one_qubit_gates=155, two_qubit_gates=11, measured_qubits=5,
                shots=1100, one_qubit_error=profile.one_qubit_error,
                two_qubit_error=profile.two_qubit_error,
                readout_error=profile.readout_error)
            tops.append(max(budget.contributions.items(),
                            key=lambda kv: kv[1])[0])
        self.assertEqual(tops[0], tops[1])

    def test_the_swap_does_invert_it_on_a_shallow_wide_circuit(self):
        """Which is the case the comment in devices.py claims, and the
        case a first draft of that comment got wrong."""
        from qem_auditor.prescribe import budget_from_calibration

        tops = []
        for profile in (self.forte, self.enterprise):
            budget = budget_from_calibration(
                one_qubit_gates=10, two_qubit_gates=3, measured_qubits=8,
                shots=100_000, one_qubit_error=profile.one_qubit_error,
                two_qubit_error=profile.two_qubit_error,
                readout_error=profile.readout_error)
            tops.append(max(budget.contributions.items(),
                            key=lambda kv: kv[1])[0].name)
        self.assertEqual(tops, ["READOUT", "GATE_STOCHASTIC"])
