"""The autonomous loop: does it stop honestly, and can it talk itself
into a verdict it did not earn?
"""
import copy
import unittest

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp

    from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

    HAVE_QISKIT = True
except ImportError:  # pragma: no cover
    HAVE_QISKIT = False

from qem_auditor import Verdict
from qem_auditor.agent import AuditAgent, Investigation

from .helpers import make_experiment

BASIS = ["u", "cx"]


def _base():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _folded(base, folds):
    qc = base.copy()
    for _ in range(folds):
        qc.cx(0, 1)
        qc.cx(0, 1)
    return qc


class StoppingTest(unittest.TestCase):
    def test_it_stops_and_says_why_when_nothing_can_run(self):
        investigation = AuditAgent().investigate(make_experiment())
        self.assertTrue(investigation.stopped_because)
        self.assertLessEqual(len(investigation.rounds), 1)

    def test_it_never_exceeds_its_round_limit(self):
        investigation = AuditAgent(max_rounds=2).investigate(make_experiment())
        self.assertLessEqual(len(investigation.rounds), 2)

    def test_a_terminal_verdict_ends_it_immediately(self):
        exp = make_experiment(ideal_control=False)
        investigation = AuditAgent().investigate(exp)
        self.assertIs(investigation.verdict, Verdict.INVALID)
        self.assertIn("disqualified", investigation.stopped_because)

    def test_the_stopping_reason_is_specific_not_generic(self):
        investigation = AuditAgent().investigate(make_experiment())
        self.assertGreater(len(investigation.stopped_because), 40)


class AuthorityTest(unittest.TestCase):
    """The agent decides whether to continue. It never decides truth."""

    def test_the_agent_cannot_certify(self):
        for attr in dir(AuditAgent):
            self.assertNotIn("certif", attr.lower())

    def test_every_round_verdict_comes_from_the_gates(self):
        from qem_auditor import audit

        exp = make_experiment(real_hardware_full_validation=False)
        investigation = AuditAgent().investigate(exp)
        for round_ in investigation.rounds:
            self.assertIs(round_.verdict, round_.report.verdict)

    def test_an_agent_run_does_not_upgrade_an_unproven_claim(self):
        exp = make_experiment(real_hardware_full_validation=False)
        before = __import__("qem_auditor").audit(exp).verdict
        AuditAgent().investigate(exp)
        after = __import__("qem_auditor").audit(exp).verdict
        self.assertIsNot(after, Verdict.CERTIFIED_UNDER_SCOPE)
        self.assertIs(before, after)


class BudgetTest(unittest.TestCase):
    def test_a_zero_budget_blocks_costly_attacks(self):
        agent = AuditAgent(budget_usd=0.0)
        investigation = agent.investigate(make_experiment())
        self.assertTrue(investigation.stopped_because)

    def test_spending_is_tracked(self):
        agent = AuditAgent(budget_usd=100.0)
        agent.investigate(make_experiment())
        self.assertGreaterEqual(agent.spent_usd, 0.0)


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class RealLoopTest(unittest.TestCase):
    def _run(self, optimization_level):
        exp = make_experiment()
        exp.controls.provenance.clear()
        base = _base()
        submitted = transpile(_folded(base, 2), basis_gates=BASIS,
                              optimization_level=optimization_level)
        agent = AuditAgent(adapter=QiskitAdapter(seed=5), max_rounds=3)
        return agent.investigate(
            copy.deepcopy(exp),
            base_circuit=base, submitted_circuit=submitted,
            circuit=base, observable=SparsePauliOp("ZZ"),
            mitigator=lambda e: e(base, SparsePauliOp("ZZ")),
            computation=lambda: 1.0)

    def test_it_finds_the_compiler_cancellation_on_its_own(self):
        investigation = self._run(optimization_level=3)
        self.assertIs(investigation.verdict, Verdict.INVALID)
        self.assertGreater(investigation.total_falsified, 0)

    def test_a_clean_submission_is_not_falsified(self):
        investigation = self._run(optimization_level=0)
        self.assertIsNot(investigation.verdict, Verdict.INVALID)

    def test_it_records_what_it_measured(self):
        investigation = self._run(optimization_level=0)
        rounds_with_attacks = [r for r in investigation.rounds if r.attack_report]
        self.assertTrue(rounds_with_attacks)
        self.assertTrue(rounds_with_attacks[0].attack_report.outcomes)

    def test_belief_moves_when_an_attack_falsifies(self):
        investigation = self._run(optimization_level=3)
        final = investigation.rounds[0]
        self.assertLess(final.belief["H_genuine"], 0.5)


if __name__ == "__main__":
    unittest.main()
