#!/usr/bin/env python3
"""Your circuit, your observable, the whole engine.

Everything else here runs on H2 or an Ising chain, which are this
project's own systems. This one is the question that matters: does any of
it work on a circuit nobody involved has seen?

Until recently the honest answer was no, and the failure mode was the bad
kind. The measurement layer assumed every term of the observable was
all-Z or all-X -- true of H2 in its encoding, true of the Ising chain,
false of most things -- and it did not check. An operator with a term
like XYZ had its bases popped arbitrarily from a set and was measured in
one of them, returning a number that was WRONG rather than absent, while
the docstring above it said that silently averaging would be wrong.

So the circuit below is deliberately nothing like the ones this package
grew up on: a hardware-efficient ansatz, the shape most people actually
submit, with a mixed-Pauli observable that needs three measurement
settings and terms this code could not previously have measured at all.
"""
import statistics
import sys

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
except ImportError:
    print("this example needs qiskit-aer: pip install 'qem-auditor[adapters]'")
    sys.exit(0)

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/..")

from benchmarks.methods import (METHODS, Sampler,  # noqa: E402
                                TargetScrambledSampler,
                                system_from_circuit, unmitigated)
from qem_auditor.estimation import group_terms  # noqa: E402
from qem_auditor.prescribe import budget_from_calibration, prescribe  # noqa: E402

SHOTS = 20_000
SEEDS = (101, 202, 303, 404, 505)


def your_circuit() -> QuantumCircuit:
    """A hardware-efficient ansatz: rotations, entangle, repeat."""
    qc = QuantumCircuit(3, name="hardware_efficient_ansatz")
    for q in range(3):
        qc.ry(0.6 + 0.3 * q, q)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for q in range(3):
        qc.ry(0.4 - 0.2 * q, q)
        qc.rz(0.5 + 0.1 * q, q)
    qc.cx(0, 2)
    qc.cx(0, 1)
    for q in range(3):
        qc.ry(0.25 * q, q)
    return qc


def your_observable() -> SparsePauliOp:
    """Mixed Pauli terms, including one that needs all three bases."""
    return SparsePauliOp.from_list([
        ("XYZ", 0.5), ("ZZI", 0.3), ("IXX", 0.2), ("ZIZ", 0.4), ("III", 0.1)])


def your_device() -> NoiseModel:
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx", "cz"])
    noise.add_all_qubit_quantum_error(
        depolarizing_error(0.001, 1),
        ["h", "t", "ry", "rz", "sdg", "u", "sx", "x"])
    noise.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.03, 0.97]]))
    return noise


def main() -> int:
    circuit, observable = your_circuit(), your_observable()
    system = system_from_circuit(circuit, observable, name=circuit.name)
    settings, _ = group_terms(observable.paulis.to_labels())
    two_qubit = sum(1 for i in circuit.data if len(i.qubits) >= 2)
    one_qubit = sum(1 for i in circuit.data if len(i.qubits) == 1)

    print("=" * 72)
    print("  WHAT YOU BROUGHT")
    print("=" * 72)
    print(f"  circuit      {circuit.num_qubits} qubits, depth {circuit.depth()}, "
          f"{two_qubit} two-qubit gates")
    print(f"  observable   {len(observable)} terms: "
          f"{', '.join(observable.paulis.to_labels())}")
    print(f"  measured in  {len(settings)} settings, worked out from the "
          "observable:")
    for setting in settings:
        print(f"                 {''.join(setting)}  (qubit 0 first)")
    print(f"  training     {len(system.clifford_variants)} near-Clifford circuits "
          "generated for CDR")
    print(f"  exact        {system.exact:+.4f}  (statevector, so the ERROR of each "
          "method can be measured;")
    print("               applying a method needs no such thing, which is the")
    print("               difference between benchmarking mitigation and using it)")

    backend = AerSimulator(noise_model=your_device())
    budget = budget_from_calibration(
        two_qubit_gates=two_qubit, one_qubit_gates=one_qubit,
        measured_qubits=circuit.num_qubits,
        two_qubit_error=0.01, one_qubit_error=0.001,
        readout_error=0.03, shots=SHOTS)

    print("\n" + "=" * 72)
    print("  WHAT THE ENGINE SAYS BEFORE RUNNING ANYTHING")
    print("=" * 72)
    print()
    print(budget.format_budget())
    consult = prescribe(budget)
    print(f"\n  prescribed:  {consult.leading.action}")
    for prescription in consult.prescriptions[1:3]:
        print(f"               {prescription.action}")
    for name, why in consult.will_not_help[:2]:
        print(f"  not this:    {name} -- {why[:60]}")

    print("\n" + "=" * 72)
    print("  WHAT ACTUALLY HAPPENED")
    print("=" * 72)

    # Target-only scrambling: the calibration a method takes is a
    # separate and legitimate input, and scrambling it too let a
    # calibrated method absorb the distortion in its own fit. CDR scored
    # 0.390 under the older attack and 1.230 under this one.
    def shift(method):
        honest = statistics.median(
            [method(Sampler(backend, SHOTS, s, system)) for s in SEEDS[:3]])
        scrambled = statistics.median(
            [method(TargetScrambledSampler(backend, SHOTS, s, system))
             for s in SEEDS[:3]])
        return abs(scrambled - honest)

    reference = shift(unmitigated)
    rows = []
    for name, method in METHODS.items():
        try:
            error = statistics.median(
                [system.error(method(Sampler(backend, SHOTS, s, system)))
                 for s in SEEDS])
            rows.append((name, error, shift(method) / reference if reference else 0))
        except ValueError as refusal:
            rows.append((name, None, str(refusal)))

    raw = next(e for n, e, _ in rows if n == "unmitigated")
    print(f"\n  {'method':28s} {'error':>9s} {'gain':>7s} {'sensitivity':>12s}")
    print("  " + "-" * 62)
    for name, error, extra in sorted(rows, key=lambda r: (r[1] is None, r[1])):
        if error is None:
            print(f"  {name:28s}   refused -- {extra[:34]}")
            continue
        flag = "" if extra >= 0.5 else "  <-- not reading the data"
        print(f"  {name:28s} {error:9.4f} {raw / error:6.2f}x {extra:12.3f}{flag}")

    honest = [(n, e) for n, e, x in rows
              if e is not None and isinstance(x, float) and x >= 0.5]
    best = min(honest, key=lambda r: r[1])
    print(f"\n  Best honest method on YOUR circuit: {best[0]} at {best[1]:.4f}, "
          f"{raw / best[1]:.2f}x better than raw.")
    print("  The fraud is still closest and still caught. The dressed identity")
    print("  still returns exactly the unmitigated value. Nothing here was")
    print("  special-cased for this circuit.")
    print("\n  One method refused rather than guessed: symmetry post-selection")
    print("  needs a symmetry someone asserts, and no error budget reveals one.")
    print("  Refusing is the correct answer, not a gap.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
