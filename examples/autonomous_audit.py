#!/usr/bin/env python3
"""Hand it an experiment; it investigates on its own.

The agent audits, works out what could still be wrong, proposes attacks
(from the built-in grammar, plus a language model if one is configured),
executes the ones it can, folds the results back in, and decides whether
to continue. It stops when nothing informative remains to buy, and says
why.

It never decides the claim is true. Every verdict comes from the gates.

Works with no language model at all. To add one -- a free local model is
plenty:

    ollama serve && ollama pull llama3.1
    export QEM_LLM_PROVIDER=openai
    export QEM_LLM_BASE_URL=http://localhost:11434/v1
    export QEM_LLM_MODEL=llama3.1

Needs qiskit for the executable attacks: pip install -e ".[adapters]"
"""
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from qem_auditor import Controls, Experiment, Outputs, UncertaintyCoverage
from qem_auditor.adapters.qiskit_adapter import QiskitAdapter
from qem_auditor.agent import AuditAgent
from qem_auditor.llm import provider_from_env
from qem_auditor.report import render_console

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
    base = base_circuit()
    vals = [expectation(folded(base, f), OBSERVABLE) for f in (0, 1, 2)]
    return 3 * vals[0] - 3 * vals[1] + vals[2]


def main() -> None:
    base = base_circuit()
    # The claimant's own submission path -- optimization_level=3, which is
    # where the historical fold-cancellation bug lives.
    submitted = transpile(folded(base, 2), basis_gates=BASIS, optimization_level=3)

    exp = Experiment(
        experiment_id="zne_claim_autonomous",
        claim="ZNE gate folding recovers the zero-noise expectation value.",
        description="ZNE with gate folding, submitted after standard transpilation.",
        backend="aer_simulator",
        shots=20_000,
        controls=Controls(),
        outputs=Outputs(uncertainty=UncertaintyCoverage(shot_noise=True)),
    )

    provider = provider_from_env()
    print(f"language model: {getattr(provider, 'name', 'none')}"
          f"{' (none configured -- running the deterministic grammar)' if provider.name == 'null' else ''}")

    agent = AuditAgent(adapter=QiskitAdapter(seed=11), provider=provider,
                       max_rounds=4, budget_usd=170.0)
    investigation = agent.investigate(
        exp,
        base_circuit=base, submitted_circuit=submitted,
        circuit=base, observable=OBSERVABLE, mitigator=zne_mitigator,
        computation=lambda: 1.0,
    )

    investigation.print_investigation()
    print()
    print(render_console(exp, investigation))


if __name__ == "__main__":
    main()
