"""The executor: does it run what it can, and refuse to pretend about what
it cannot?
"""
import unittest

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp

    from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

    HAVE_QISKIT = True
except ImportError:  # pragma: no cover
    HAVE_QISKIT = False

from qem_auditor import audit
from qem_auditor.adversary import AdversarialScientist, GRAMMAR
from qem_auditor.executor import AttackExecutor, AttackReport

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


class NotRunSemanticsTest(unittest.TestCase):
    """The distinction that makes the whole thing honest: an attack that
    could not be run is not an attack the claim survived."""

    def test_an_unrunnable_attack_reports_survived_as_none(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        report = AttackExecutor().run(exp, plan)
        for outcome in report.outcomes:
            if not outcome.ran:
                self.assertIsNone(outcome.survived)
                self.assertNotEqual(outcome.survived, False)

    def test_unrun_attacks_are_counted_separately_from_survivors(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        report = AttackExecutor().run(exp, plan)
        self.assertTrue(report.not_run)
        for outcome in report.not_run:
            self.assertNotIn(outcome, report.survived)

    def test_missing_adapter_is_reported_not_silently_passed(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        report = AttackExecutor(adapter=None).run(exp, plan)
        self.assertTrue(any("adapter" in o.detail for o in report.not_run))

    def test_missing_artifacts_are_reported(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        report = AttackExecutor(adapter=object()).run(exp, plan)
        self.assertTrue(report.not_run)


class HookTest(unittest.TestCase):
    def test_a_domain_hook_runs_an_otherwise_unrunnable_attack(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))
        executor = AttackExecutor(hooks={
            "T_label": lambda e: ("artifact", "shuffled fit was comparable")})
        report = executor.run(exp, plan)
        label = next(o for o in report.outcomes if o.attack.transformation == "T_label")
        self.assertTrue(label.ran)
        self.assertIs(label.survived, False)

    def test_a_raising_hook_is_not_a_pass(self):
        exp = make_experiment()
        plan = AdversarialScientist().propose(exp, audit(exp))

        def broken(e):
            raise RuntimeError("fitting code blew up")

        report = AttackExecutor(hooks={"T_label": broken}).run(exp, plan)
        label = next(o for o in report.outcomes if o.attack.transformation == "T_label")
        self.assertIsNone(label.survived)
        self.assertIn("blew up", label.detail)


@unittest.skipUnless(HAVE_QISKIT, "qiskit not installed")
class RealExecutionTest(unittest.TestCase):
    """Against the real transpiler, reproducing the historical failure."""

    def _plan_and_run(self, optimization_level):
        exp = make_experiment()
        exp.controls.provenance.clear()
        plan = AdversarialScientist().propose(exp, audit(exp))
        base = _base()
        submitted = transpile(_folded(base, 2), basis_gates=BASIS,
                              optimization_level=optimization_level)
        return AttackExecutor(adapter=QiskitAdapter(seed=5)).run(
            exp, plan, base_circuit=base, submitted_circuit=submitted,
            computation=lambda: 1.0)

    def test_optimization_level_3_is_falsified_by_the_compiler_attack(self):
        report = self._plan_and_run(3)
        compiler = next(o for o in report.outcomes
                        if o.attack.transformation == "T_compiler")
        self.assertIs(compiler.survived, False)

    def test_optimization_level_0_survives_it(self):
        report = self._plan_and_run(0)
        compiler = next(o for o in report.outcomes
                        if o.attack.transformation == "T_compiler")
        self.assertIs(compiler.survived, True)

    def test_a_composed_attack_runs_every_part(self):
        exp = make_experiment()
        exp.controls.provenance.clear()
        plan = AdversarialScientist().propose(exp, audit(exp))
        composed = [a for a in plan.attacks if " o " in a.transformation]
        self.assertTrue(composed, "expected the proposer to compose interacting attacks")
        base = _base()
        submitted = transpile(_folded(base, 2), basis_gates=BASIS, optimization_level=3)
        report = AttackExecutor(adapter=QiskitAdapter(seed=5)).run(
            exp, plan, base_circuit=base, submitted_circuit=submitted,
            circuit=base, observable=SparsePauliOp("ZZ"),
            mitigator=lambda expectation: expectation(base, SparsePauliOp("ZZ")),
            computation=lambda: 1.0)
        out = next(o for o in report.outcomes if " o " in o.attack.transformation)
        self.assertTrue(out.ran)
        self.assertIs(out.survived, False)  # the compiler part falsifies it


if __name__ == "__main__":
    unittest.main()
