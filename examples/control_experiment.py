"""Is the effect the mechanism, or the apparatus?

The control-experiment idea is ported from `quantum-verifier`'s
`falsify`: build a second circuit that removes the entangling mechanism
the claim depends on and keeps everything else identical, run both, and
let the confounds cancel. It audits a CLAIM rather than a METHOD, and it
works when nobody knows the right answer -- which is the case this
package's own attacks did not cover.

What this run shows is the two places a reported number is not yet a
finding.

Run:  python examples/control_experiment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import QuantumCircuit, transpile                      # noqa: E402
from qiskit_aer import AerSimulator                               # noqa: E402

from qem_auditor.control import (build_control, distribution_shift,  # noqa: E402
                                 isolate_effect, total_variation)
from real_device_audit import calibration, device_noise           # noqa: E402

SHOTS = 4096


def run(backend, circuit, seed):
    transpiled = transpile(circuit, backend, optimization_level=0)
    counts = backend.run(transpiled, shots=SHOTS, seed_simulator=seed).result().get_counts()
    return {bits.replace(" ", ""): n for bits, n in counts.items()}


def heading(title):
    print(f"\n{title}\n" + "-" * len(title))


def main() -> int:
    noisy = AerSimulator(noise_model=device_noise(calibration()))

    ghz = QuantumCircuit(3, 3)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(1, 2)
    ghz.measure(range(3), range(3))
    control, removed = build_control(ghz)

    heading("A claim that is true: entanglement concentrates GHZ on 000 and 111")
    print(isolate_effect(run(noisy, ghz, 1), run(noisy, control, 2),
                         marked={"000", "111"}, removed=removed).describe())

    heading("The same circuit, claiming something it does not do")
    print(isolate_effect(run(noisy, ghz, 1), run(noisy, control, 2),
                         marked={"010"}, removed=removed).describe())
    print("\n  Both numbers come out of the same comparison. Only one of them")
    print("  is a finding, and an effect size reported without this bar does")
    print("  not say which.")

    heading("Discovery mode: TVD is not anchored at zero")
    ideal = AerSimulator()
    wide = QuantumCircuit(10, 10)
    wide.h(range(10))
    for qubit in range(0, 9, 2):
        wide.cz(qubit, qubit + 1)
    wide.measure(range(10), range(10))
    wide_control, _ = build_control(wide)
    a, b = run(ideal, wide, 3), run(ideal, wide_control, 4)

    print(f"  raw TVD: {total_variation(a, b):.4f}")
    print('  On a scale described as "0 = identical, 1 = completely different"')
    print("  that reads as a substantial effect. It is not one:\n")
    print(distribution_shift(a, b).describe())

    heading("And the same measurement where the mechanism really does move it")
    chain = QuantumCircuit(10, 10)
    chain.h(0)
    for qubit in range(9):
        chain.cx(qubit, qubit + 1)
    chain.measure(range(10), range(10))
    chain_control, _ = build_control(chain)
    print(distribution_shift(run(ideal, chain, 5), run(ideal, chain_control, 6)).describe())

    print("\n  The null moved from 0.28 to 0.01 between those two cases, at the")
    print("  same width and the same shots, because a concentrated")
    print("  distribution has far fewer effective outcomes. That is why the")
    print("  null is measured on the run's own counts rather than taken from")
    print("  a rule of thumb.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
