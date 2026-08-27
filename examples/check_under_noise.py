#!/usr/bin/env python3
"""The same claim, audited noiselessly and then under real device noise.

The ideal control and the mitigation-benefit control ask opposite
questions, and passing one says nothing about the other:

    ideal control       does this method BREAK when there is no noise?
    mitigation benefit  does it HELP when there is?

A do-nothing mitigator passes the first trivially -- it cannot amplify
noise it never touches -- and fails the second. A badly conditioned
extrapolator can do the reverse. Only running both separates them.

Needs qiskit-aer: pip install -e ".[adapters]"
"""
import warnings

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qem_auditor import Auditor
from qem_auditor.adapters.qiskit_adapter import QiskitAdapter
from qem_auditor.adapters.sources import AerNoiseSource

warnings.filterwarnings("ignore")
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


def device_noise() -> NoiseModel:
    """A plain depolarizing model. Real backends give richer ones, but the
    interface is identical -- a source is a source."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.04, 2), ["cx"])
    return nm


def zne(expectation) -> float:
    """Richardson extrapolation to the zero-noise limit."""
    base = base_circuit()
    v = [expectation(folded(base, f), OBSERVABLE) for f in (0, 1, 2)]
    return 3 * v[0] - 3 * v[1] + v[2]


def noop(expectation) -> float:
    """No mitigation at all. Passes the ideal control trivially."""
    return expectation(base_circuit(), OBSERVABLE)


def main() -> None:
    adapter = QiskitAdapter(seed=3, source=AerNoiseSource(device_noise()))
    base = base_circuit()

    print(f"source: {adapter.source.name}")
    print(f"exact  <ZZ> = "
          f"{adapter.source.noiseless_twin().exact(base, OBSERVABLE):.6f}")
    print(f"noisy  <ZZ> = {adapter.source.exact(base, OBSERVABLE):.6f}")
    print()

    for name, mitigator in (("Richardson ZNE", zne),
                            ("no mitigation at all", noop)):
        print("=" * 70)
        print(name)
        print("=" * 70)

        ideal = adapter.measure_ideal_control(base, OBSERVABLE, mitigator,
                                              shots=200_000)
        benefit = adapter.measure_mitigation_benefit(base, OBSERVABLE, mitigator,
                                                     shots=200_000, trials=8)
        for label, m in (("ideal control    ", ideal),
                         ("mitigation benefit", benefit)):
            mark = "PASS" if m.passed else "FAIL" if m.passed is False else "N/A "
            print(f"  [{mark}] {label}: {m.detail}")
        print()

    print("=" * 70)
    print("READING THIS")
    print("=" * 70)
    print("""
  Both mitigators pass the ideal control. Only one of them does anything.

  That is the gap this control closes: the ideal control is a necessary
  condition, not a sufficient one, and a method can sail through it while
  being completely inert. Running it alone and calling the result verified
  would be a mistake the auditor should not let anyone make.

  Watch the numbers rather than the labels. Richardson's amplification on
  the noiseless model sits near its own coefficient norm -- the expected,
  benign cost of extrapolating -- while the no-op has nothing to amplify.
  Neither is pathological, so both pass. The pathological case is orders
  of magnitude worse, which is what the bar is set to catch.

  Note also what the ideal control does NOT do here: it stays noiseless
  even though the adapter was built with a noise model. Running it through
  device noise would quietly turn it into a different, much weaker check.
""")


if __name__ == "__main__":
    main()
