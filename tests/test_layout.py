"""Which qubits to run on, chosen by what is actually hurting you.

The property that matters is that the same coupling map gives DIFFERENT
answers for different error budgets. A layout chooser that always returns
the lowest-gate-error pair is the one everybody already has, and on a
readout-dominated device it is wrong.
"""
import itertools
import random
import time
import unittest

from qem_auditor import Provenance
from qem_auditor.layout import (DeviceLayout, QubitProperties, _connected_sets,
                                advise_layout, rank_placements, score_placement)
from qem_auditor.prescribe import METHODS_BY_NAME, ErrorBudget, ErrorSource

E = ErrorSource


def toy_device():
    """Qubits 1 and 2 have good readout and a terrible gate between them;
    0 and 1 have a good gate and bad readout on 0. Which pair is better
    is not a fact about the device."""
    return DeviceLayout(
        qubits={0: QubitProperties(0.05), 1: QubitProperties(0.01),
                2: QubitProperties(0.012), 3: QubitProperties(0.06)},
        edges={(0, 1): 0.002, (1, 2): 0.03, (2, 3): 0.004},
        name="toy")


def budget(contributions):
    return ErrorBudget(contributions, Provenance.MEASURED)


class DeviceTest(unittest.TestCase):
    def test_an_edge_naming_an_unknown_qubit_is_refused(self):
        with self.assertRaises(ValueError):
            DeviceLayout(qubits={0: QubitProperties(0.01)}, edges={(0, 9): 0.01})

    def test_a_self_loop_is_refused(self):
        with self.assertRaises(ValueError):
            DeviceLayout(qubits={0: QubitProperties(0.01)}, edges={(0, 0): 0.01})

    def test_edges_are_order_insensitive(self):
        device = DeviceLayout(
            qubits={0: QubitProperties(0.01), 1: QubitProperties(0.01)},
            edges={(1, 0): 0.005})
        self.assertEqual(device.gate_error(0, 1), device.gate_error(1, 0))

    def test_an_impossible_readout_error_is_refused(self):
        with self.assertRaises(ValueError):
            QubitProperties(readout_error=1.5)

    def test_unconnected_qubits_cannot_be_scored_silently(self):
        """A placement needing a connection the device lacks is a
        different circuit once routing inserts swaps."""
        with self.assertRaises(KeyError):
            toy_device().gate_error(0, 3)


class BudgetDrivesTheChoiceTest(unittest.TestCase):
    """The whole point of the module."""

    def test_readout_dominated_picks_the_low_readout_pair(self):
        advice = advise_layout(toy_device(),
                               budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 3.0}),
                               2, two_qubit_gates=2)
        self.assertEqual(advice.best.qubits, (1, 2))

    def test_gate_dominated_picks_the_low_gate_error_pair_instead(self):
        advice = advise_layout(toy_device(),
                               budget({E.READOUT: 3.0, E.GATE_STOCHASTIC: 30.0}),
                               2, two_qubit_gates=2)
        self.assertEqual(advice.best.qubits, (0, 1))

    def test_the_same_device_gives_opposite_answers(self):
        """Stated as its own test because it is the claim: a layout
        chooser that ignores the budget cannot be right for both."""
        device = toy_device()
        readout_pick = advise_layout(
            device, budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 3.0}), 2,
            two_qubit_gates=2).best.qubits
        gate_pick = advise_layout(
            device, budget({E.READOUT: 3.0, E.GATE_STOCHASTIC: 30.0}), 2,
            two_qubit_gates=2).best.qubits
        self.assertNotEqual(readout_pick, gate_pick)

    def test_a_source_the_budget_does_not_care_about_does_not_drive_the_choice(self):
        pure_readout = advise_layout(device := toy_device(),
                                     budget({E.READOUT: 30.0}), 2,
                                     two_qubit_gates=2)
        self.assertEqual(pure_readout.best.qubits, (1, 2))
        self.assertEqual(pure_readout.best.gate_cost,
                         score_placement((1, 2), device, budget({E.READOUT: 30.0}),
                                         two_qubit_gates=2).gate_cost)


class MethodChangesTheChoiceTest(unittest.TestCase):
    """Choosing qubits and choosing a method are one decision.

    Measured: on fake_kyiv the low-readout pair beats the low-gate pair
    13.87 to 36.46 unmitigated, and loses 11.02 to 6.18 once REM runs.
    """

    def test_planning_readout_mitigation_moves_the_optimum(self):
        device = toy_device()
        raw_budget = budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 3.0})
        raw_pick = advise_layout(device, raw_budget, 2,
                                 two_qubit_gates=2).best.qubits
        with_rem = advise_layout(
            device, raw_budget, 2,
            after_method=METHODS_BY_NAME["readout error mitigation (REM)"],
            two_qubit_gates=2).best.qubits
        self.assertNotEqual(raw_pick, with_rem)

    def test_the_report_says_it_scored_against_a_residual(self):
        advice = advise_layout(
            toy_device(), budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 3.0}), 2,
            after_method=METHODS_BY_NAME["readout error mitigation (REM)"],
            two_qubit_gates=2)
        self.assertIn("leaves behind", advice.format_advice())


class AdviceTest(unittest.TestCase):
    def test_it_reports_the_worst_placement_too(self):
        """Best alone does not tell a user whether placement matters on
        their device; best against worst does."""
        advice = advise_layout(toy_device(),
                               budget({E.READOUT: 30.0, E.GATE_STOCHASTIC: 3.0}),
                               2, current=(0, 1), two_qubit_gates=2)
        self.assertGreater(advice.worst.cost, advice.best.cost)
        self.assertIn("worst available", advice.format_advice())

    def test_a_placement_already_near_the_best_is_not_worth_moving(self):
        advice = advise_layout(toy_device(),
                               budget({E.READOUT: 3.0, E.GATE_STOCHASTIC: 30.0}),
                               2, current=(0, 1), two_qubit_gates=2)
        self.assertFalse(advice.worth_moving)
        self.assertIn("not worth the churn", advice.format_advice())

    def test_no_current_placement_means_no_comparison_is_invented(self):
        advice = advise_layout(toy_device(), budget({E.READOUT: 30.0}), 2,
                               two_qubit_gates=2)
        self.assertIsNone(advice.gain)
        self.assertFalse(advice.worth_moving)

    def test_a_repeated_qubit_is_refused(self):
        with self.assertRaises(ValueError):
            score_placement((1, 1), toy_device(), budget({E.READOUT: 1.0}))

    def test_every_connected_pair_is_considered(self):
        ranked = rank_placements(toy_device(), budget({E.READOUT: 30.0}), 2,
                                 two_qubit_gates=2)
        self.assertEqual(len(ranked), 3)

    def test_ranking_is_best_first(self):
        ranked = rank_placements(toy_device(), budget({E.READOUT: 30.0}), 2,
                                 two_qubit_gates=2)
        self.assertEqual([p.cost for p in ranked], sorted(p.cost for p in ranked))

    def test_a_placement_that_cannot_exist_is_refused(self):
        tiny = DeviceLayout(qubits={0: QubitProperties(0.01)}, edges={})
        with self.assertRaises(ValueError):
            advise_layout(tiny, budget({E.READOUT: 1.0}), 2)


if __name__ == "__main__":
    unittest.main()


class ConnectedSetEnumeration(unittest.TestCase):
    """The search is exhaustive, so it must produce each set once and only
    once -- and it must be able to say so before the heat death of the
    universe. Both halves were broken: the old path-walking version found
    every set correctly and reached them through every ordering, so on a
    127-qubit lattice it stopped answering at ten qubits without ever
    reaching the limit check that was supposed to refuse."""

    def _random_device(self, n, density, seed):
        rng = random.Random(seed)
        qubits = {i: QubitProperties(0.01) for i in range(n)}
        edges = {(a, b): 0.01 for a, b in itertools.combinations(range(n), 2)
                 if rng.random() < density}
        return DeviceLayout(qubits, edges)

    def _brute_force(self, device, size):
        found = set()
        for candidate in itertools.combinations(sorted(device.qubits), size):
            members, seen, frontier = set(candidate), {candidate[0]}, [candidate[0]]
            while frontier:
                for n in device.neighbours(frontier.pop()) & members:
                    if n not in seen:
                        seen.add(n)
                        frontier.append(n)
            if seen == members:
                found.add(candidate)
        return found

    def test_matches_brute_force_and_never_repeats(self):
        for seed in range(25):
            device = self._random_device(7, 0.45, seed)
            for size in range(1, 8):
                got = list(_connected_sets(device, size))
                self.assertEqual(len(got), len(set(got)),
                                 f"seed {seed} size {size} yielded duplicates")
                self.assertEqual(set(got), self._brute_force(device, size),
                                 f"seed {seed} size {size} disagrees with brute force")

    def test_refuses_promptly_instead_of_grinding(self):
        # A 60-qubit ring has far more than 20 connected 12-sets. The
        # point is not that it refuses -- it is that it refuses after
        # bounded work, which the path-walking version could not do.
        qubits = {i: QubitProperties(0.01) for i in range(60)}
        edges = {tuple(sorted((i, (i + 1) % 60))): 0.01 for i in range(60)}
        device = DeviceLayout(qubits, edges)
        started = time.monotonic()
        with self.assertRaises(OverflowError):
            list(_connected_sets(device, 12, limit=20))
        self.assertLess(time.monotonic() - started, 5.0)

    def test_candidates_restrict_the_search(self):
        device = toy_device()
        region = sorted(device.qubits)[:3]
        for placement in _connected_sets(device, 2, candidates=region):
            self.assertTrue(set(placement) <= set(region))

    def test_unknown_candidate_is_refused(self):
        with self.assertRaises(KeyError):
            list(_connected_sets(toy_device(), 2, candidates=[0, 999]))

    def test_advice_reports_the_region_it_searched(self):
        device = toy_device()
        region = sorted(device.qubits)[:3]
        advice = advise_layout(device, budget({E.READOUT: 30.0}), 2,
                               candidates=region)
        self.assertEqual(advice.region, tuple(region))
        self.assertIn("within the 3 qubits you named", advice.format_advice())
