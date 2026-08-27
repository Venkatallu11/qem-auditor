#!/usr/bin/env python3
"""The closed loop: propose attacks, execute them, let the gates judge.

    claim -> audit -> what can still be wrong -> generate adversaries
          -> execute -> formal audit -> belief update -> next experiment

Nothing here is taken on trust. The adversarial scientist proposes the
attacks and commits in advance to what each outcome would mean; the
executor runs them against real circuits; the gates decide what actually
happened. The proposer never issues a verdict.

Needs qiskit: pip install -e ".[adapters]"
"""
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from qem_auditor import Auditor, Controls, Experiment, Outputs, UncertaintyCoverage, audit
from qem_auditor.adapters.qiskit_adapter import QiskitAdapter
from qem_auditor.adversary import AdversarialScientist
from qem_auditor.executor import AttackExecutor
from qem_auditor.hypothesis import Hypothesis, HypothesisLedger, Observation
from qem_auditor.planner import Recommendation, candidates_from_attacks, plan

BASIS = ["u", "cx"]
OBSERVABLE = SparsePauliOp("ZZ")


def base_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.3, 0)
    return qc


def folded(base: QuantumCircuit, folds: int) -> QuantumCircuit:
    qc = base.copy()
    for _ in range(folds):
        qc.cx(0, 1)
        qc.cx(0, 1)
    return qc


def zne_mitigator(expectation) -> float:
    """A Richardson extrapolation to the zero-noise limit."""
    base = base_circuit()
    vals = [expectation(folded(base, f), OBSERVABLE) for f in (0, 1, 2)]
    return 3 * vals[0] - 3 * vals[1] + vals[2]


def main() -> None:
    base = base_circuit()
    # The claimant's pipeline, transpiled the way they actually submit it.
    submitted = transpile(folded(base, 2), basis_gates=BASIS, optimization_level=3)

    exp = Experiment(
        experiment_id="zne_claim_under_attack",
        claim="ZNE gate folding recovers the zero-noise expectation value.",
        description="ZNE with gate folding, submitted after standard transpilation.",
        backend="aer_simulator",
        shots=20_000,
        controls=Controls(),
        outputs=Outputs(uncertainty=UncertaintyCoverage(shot_noise=True)),
    )

    print("=" * 72)
    print("1. AUDIT THE CLAIM AS SUBMITTED")
    print("=" * 72)
    report = audit(exp)
    print(f"  verdict: {report.verdict.value}")
    print(f"  {len(report.unrun_gates)} controls never run -- so most mechanisms are "
          f"still open")

    print()
    print("=" * 72)
    print("2. WHAT CAN STILL BE WRONG? GENERATE ADVERSARIES")
    print("=" * 72)
    attack_plan = AdversarialScientist().propose(exp, report)
    print(f"  {len(attack_plan.attacks)} attacks proposed, "
          f"{len(attack_plan.executable)} executable without domain hooks")
    for a in attack_plan.executable:
        print(f"    - {a.attack_id}: {a.prediction.statistic}")

    print()
    print("=" * 72)
    print("3. RANK BY INFORMATION GAIN")
    print("=" * 72)
    ledger = HypothesisLedger([
        Hypothesis("H_genuine", "The ZNE result is a real noise correction.", 0.5),
        Hypothesis("H_artifact", "The ZNE result is an artifact.", 0.5),
    ])
    candidates = candidates_from_attacks(attack_plan)
    recommendation, proposals, reason = plan(ledger, candidates)
    print(f"  {recommendation.value}: {reason}")
    for p in proposals[:3]:
        print(f"    {p.information_gain_bits:.3f} bits  {p.candidate.candidate_id}")

    print()
    print("=" * 72)
    print("4. EXECUTE")
    print("=" * 72)
    executor = AttackExecutor(adapter=QiskitAdapter(seed=11))
    attack_report = executor.run(
        exp, attack_plan,
        base_circuit=base, submitted_circuit=submitted,
        circuit=base, observable=OBSERVABLE, mitigator=zne_mitigator,
        computation=lambda: 1.0,
    )
    attack_report.print_report()

    print()
    print("=" * 72)
    print("5. THE GATES JUDGE, AND BELIEF UPDATES")
    print("=" * 72)
    auditor = Auditor(adapter=QiskitAdapter(seed=11))
    for outcome in attack_report.outcomes:
        if outcome.measurement is not None:
            exp.controls.record_measured(outcome.measurement.control,
                                         outcome.measurement.passed)
    result = auditor.audit(exp)
    print(f"  verdict: {result.verdict.value}")
    print(f"  licence: {result.claim.licence}")

    for outcome in attack_report.falsified_by:
        ledger.update(Observation(outcome.attack.attack_id,
                                  {"H_genuine": 0.05},
                                  outcome.detail[:80]))
    print(f"\n  belief after attacks: " +
          ", ".join(f"{h}={p:.3f}" for h, p in ledger.ranked()))

    if result.analysis.primary:
        d = result.analysis.primary
        print(f"\n  WHY: {d.mode.name}\n       {d.evidence}")


if __name__ == "__main__":
    main()
