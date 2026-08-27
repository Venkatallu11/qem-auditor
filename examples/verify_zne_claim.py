#!/usr/bin/env python3
"""End-to-end: verifying a ZNE claim you did not write.

This is the tool used the way it is meant to be used on someone else's
work. Nothing here is taken on trust -- the auditor builds the circuits,
runs the transpiler, executes the claimant's mitigation against a
noiseless model, and grades what it found.

Two scenarios, differing only in transpiler optimization level, which is
the exact difference that cost the H4 project a whole ZNE result.

Needs qiskit: pip install -r requirements-adapters.txt
"""
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from qem_auditor import Auditor, Controls, Experiment, Outputs, UncertaintyCoverage
from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

BASIS = ["u", "cx"]
OBSERVABLE = SparsePauliOp("ZZ")


def base_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.3, 0)
    return qc


def folded_circuit(base: QuantumCircuit, folds: int) -> QuantumCircuit:
    """Standard ZNE gate folding: insert CX.CX identity pairs to amplify
    noise without changing the unitary."""
    qc = base.copy()
    for _ in range(folds):
        qc.cx(0, 1)
        qc.cx(0, 1)
    return qc


def make_record(claim: str, optimization_level: int) -> Experiment:
    return Experiment(
        experiment_id=f"zne_claim_opt{optimization_level}",
        claim=claim,
        description=f"ZNE with gate folding, transpiled at optimization_level="
                    f"{optimization_level}.",
        backend="aer_simulator",
        shots=20_000,
        controls=Controls(),          # nothing asserted: the auditor will fill these in
        outputs=Outputs(uncertainty=UncertaintyCoverage(shot_noise=True)),
    )


def zne_mitigator(circuit, observable):
    """A Richardson-style extrapolation to the zero-noise limit, of the kind
    a real pipeline uses. Handed an expectation oracle, it queries three
    fold factors and extrapolates."""
    def mitigate(expectation) -> float:
        values = [expectation(folded_circuit(circuit, f), observable) for f in (0, 1, 2)]
        return 3 * values[0] - 3 * values[1] + values[2]
    return mitigate


def ill_conditioned_mitigator(circuit, observable):
    """A high-order extrapolation over a long lever arm -- the shape of the
    production estimator that turned a 0.0652 kcal/mol error into 33.48.
    The coefficients are enormous, so shot noise is amplified catastrophically
    while the systematic gain is speculative."""
    def mitigate(expectation) -> float:
        values = [expectation(folded_circuit(circuit, f), observable) for f in range(5)]
        # 5-point Richardson to lambda=0: coefficients grow fast with order.
        coeffs = [70.0, -140.0, 90.0, -20.0, 1.0]
        return sum(c * v for c, v in zip(coeffs, values))
    return mitigate


def run_scenario(optimization_level: int) -> None:
    print("=" * 72)
    print(f"SCENARIO: transpiled at optimization_level={optimization_level}")
    print("=" * 72)

    base = base_circuit()
    folded = folded_circuit(base, folds=2)
    submitted = transpile(folded, basis_gates=BASIS,
                          optimization_level=optimization_level)

    exp = make_record("ZNE gate folding recovers the zero-noise expectation value.",
                      optimization_level)
    auditor = Auditor(adapter=QiskitAdapter(seed=11))

    # 1. Did the noise-amplifying gates survive to execution?
    m = auditor.verify_fold_survival(exp, base=base, submitted=submitted)
    print(f"\n[{'PASS' if m.passed else 'FAIL'}] fold survival")
    print(f"      {m.detail}")

    # 2. Does the mitigation degrade a noiseless case?
    m = auditor.verify_ideal_control(exp, base, OBSERVABLE, zne_mitigator(base, OBSERVABLE),
                                     shots=20_000)
    print(f"\n[{'PASS' if m.passed else 'FAIL'}] ideal control")
    print(f"      {m.detail}")

    result = auditor.audit(exp)
    print(f"\nVERDICT: {result.verdict.value}")
    print(f"LICENCE: {result.claim.licence}")
    if result.analysis.primary:
        d = result.analysis.primary
        print(f"\nWHY: {d.mode.name}\n     {d.evidence}\n     remedy: {d.remedy}")
    print()


def run_ill_conditioned_scenario() -> None:
    """Circuits entirely correct; the estimator is the problem."""
    print("=" * 72)
    print("SCENARIO: correct circuits, ill-conditioned estimator")
    print("=" * 72)

    base = base_circuit()
    submitted = transpile(folded_circuit(base, folds=2), basis_gates=BASIS,
                          optimization_level=0)
    exp = make_record("High-order ZNE recovers the zero-noise expectation value.", 0)
    auditor = Auditor(adapter=QiskitAdapter(seed=11))

    m = auditor.verify_fold_survival(exp, base=base, submitted=submitted)
    print(f"\n[{'PASS' if m.passed else 'FAIL'}] fold survival")
    print(f"      {m.detail}")

    m = auditor.verify_ideal_control(
        exp, base, OBSERVABLE, ill_conditioned_mitigator(base, OBSERVABLE), shots=20_000)
    print(f"\n[{'PASS' if m.passed else 'FAIL'}] ideal control")
    print(f"      {m.detail}")

    result = auditor.audit(exp)
    print(f"\nVERDICT: {result.verdict.value}")
    if result.analysis.primary:
        d = result.analysis.primary
        print(f"\nWHY: {d.mode.name}\n     {d.evidence}\n     remedy: {d.remedy}")
    print()


def main() -> None:
    # optimization_level=3 cancels the fold pairs -- the historical bug.
    run_scenario(optimization_level=3)
    # optimization_level=0 preserves them, which is why the H4 project
    # hardcoded it everywhere after losing a result to this.
    run_scenario(optimization_level=0)
    # Circuits fine, estimator pathological -- the 513x shape.
    run_ill_conditioned_scenario()


if __name__ == "__main__":
    main()
