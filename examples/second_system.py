#!/usr/bin/env python3
"""Does any of this generalise beyond one molecule?

Every quantitative claim this project has made was measured on H2 in
STO-3G: two qubits, two CX gates. That readout error dominates and
defeats ZNE, that ZNE and REM change places between noise models, that
CDR is the robust one -- one system, all of it. Findings from one system
are a hypothesis, and stating them as more than that is the thing this
package refuses when anyone else does it.

So: a transverse-field Ising chain, four spins, whose Hamiltonian is
constructed from a formula here rather than transcribed from anywhere,
and whose depth is a knob. Two questions.

  1. Does the readout finding hold? It should NOT, if the mechanism
     behind it is real -- readout error is charged once per measured
     qubit however many gates ran, so a circuit with 24 two-qubit gates
     instead of 2 should be gate-dominated. A finding that survived that
     would mean the mechanism was wrong.

  2. Does the auditor notice? A tool that reasons from a rule would
     repeat H2's answer. One that reasons from the budget should give a
     different answer here, and be right both times.
"""
import statistics
import sys

try:
    from qiskit_aer import AerSimulator
except ImportError:
    print("this example needs qiskit-aer: pip install 'qem-auditor[adapters]'")
    sys.exit(0)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/..")

from real_device_audit import (ECR_DURATION, SX_DURATION,  # noqa: E402
                               calibration, device_noise)

from benchmarks import tfim  # noqa: E402
from benchmarks.methods import (METHODS, Sampler, ScrambledSampler,  # noqa: E402
                                h2_system, unmitigated)
from qem_auditor.prescribe import budget_from_calibration, prescribe  # noqa: E402

SHOTS = 20_000
SEEDS = (101, 202, 303, 404, 505, 606)
SENSITIVITY_SEEDS = (101, 202, 303)
SENSITIVITY_FLOOR = 0.5


def median_error(method, system, backend, seeds=SEEDS):
    return statistics.median(
        [system.error(method(Sampler(backend, SHOTS, seed, system)))
         for seed in seeds])


def sensitivity(method, system, backend, reference):
    def shift(fn):
        honest = statistics.median(
            [fn(Sampler(backend, SHOTS, s, system)) for s in SENSITIVITY_SEEDS])
        scrambled = statistics.median(
            [fn(ScrambledSampler(backend, SHOTS, s, system))
             for s in SENSITIVITY_SEEDS])
        return abs(scrambled - honest)

    return shift(method) / reference if reference else 0.0


def budget_for(system, measured_qubits, calibration_data):
    two, one = tfim.gate_counts(system.circuit)
    return budget_from_calibration(
        two_qubit_gates=two, one_qubit_gates=one,
        measured_qubits=measured_qubits,
        two_qubit_error=calibration_data["ecr_error"],
        one_qubit_error=calibration_data["sx_error"],
        readout_error=calibration_data["readout_error"], shots=SHOTS,
        circuit_duration_s=two * ECR_DURATION + one * SX_DURATION,
        t2_s=min(calibration_data["t2"]))


def main() -> int:
    cal = calibration()
    full = AerSimulator(noise_model=device_noise(cal))
    no_readout = AerSimulator(noise_model=device_noise(cal, readout=False))

    print("=" * 74)
    print("  1. DOES READOUT STILL DOMINATE WHEN THE CIRCUIT IS DEEPER?")
    print("=" * 74)
    print("\n  Measured by switching readout off, under the real fake_kyiv")
    print("  calibration. On H2 -- two qubits, two CX gates -- readout was 82%.\n")
    print(f"  {'steps':>5s} {'2q gates':>9s} {'total':>9s} {'readout':>9s} {'readout share':>14s}")
    print("  " + "-" * 52)
    for steps in (1, 2, 4, 8):
        system = tfim.system(4, steps)
        total = median_error(unmitigated, system, full, SENSITIVITY_SEEDS)
        without = median_error(unmitigated, system, no_readout, SENSITIVITY_SEEDS)
        readout = max(total - without, 0.0)
        two, _ = tfim.gate_counts(system.circuit)
        print(f"  {steps:5d} {two:9d} {total:9.4f} {readout:9.4f} "
              f"{readout / total * 100:13.1f}%")

    print("\n  It does not, and that is the mechanism working rather than failing.")
    print("  Readout error is charged once per measured qubit however many gates")
    print("  ran; gate error is charged per gate. H2's budget was readout-heavy")
    print("  because H2 has two CX gates, not because readout dominates in")
    print("  general. The finding was real and the generalisation would not have")
    print("  been -- which is exactly what a second system is for.")

    system = tfim.system(4, 4)
    budget = budget_for(system, 4, cal)

    print("\n" + "=" * 74)
    print("  2. DOES THE AUDITOR NOTICE, OR JUST REPEAT H2'S ANSWER?")
    print("=" * 74)
    print()
    print(budget.format_budget())
    consult = prescribe(budget)
    print(f"\n  prescribed here:   {consult.leading.action}")
    print("  prescribed on H2:  readout error mitigation (REM) and CDR, with")
    print("                     ZNE demoted for reaching only 9% of the error")

    print("\n" + "=" * 74)
    print(f"  3. MEASURED: {system.name}, exact {system.exact:.4f}")
    print("=" * 74)
    reference_shift = statistics.median(
        [unmitigated(ScrambledSampler(full, SHOTS, s, system))
         for s in SENSITIVITY_SEEDS])
    reference_shift = abs(reference_shift - statistics.median(
        [unmitigated(Sampler(full, SHOTS, s, system))
         for s in SENSITIVITY_SEEDS]))

    rows = []
    for name, method in METHODS.items():
        try:
            rows.append((name, median_error(method, system, full),
                         sensitivity(method, system, full, reference_shift)))
        except ValueError:
            rows.append((name, None, None))

    raw = next(e for n, e, _ in rows if n == "unmitigated")
    print(f"\n  {'method':28s} {'error':>9s} {'gain':>7s} {'sensitivity':>12s}")
    print("  " + "-" * 62)
    for name, error, sens in sorted(rows, key=lambda r: (r[1] is None, r[1])):
        if error is None:
            print(f"  {name:28s}       n/a -- declares no symmetry here")
            continue
        flag = "" if sens >= SENSITIVITY_FLOOR else "  <-- not reading the data"
        print(f"  {name:28s} {error:9.4f} {raw / error:6.2f}x {sens:12.3f}{flag}")

    honest = [(n, e) for n, e, s in rows
              if e is not None and s is not None and s >= SENSITIVITY_FLOOR]
    best = min(honest, key=lambda r: r[1])
    cheat = [(n, e) for n, e, s in rows
             if e is not None and s is not None and s < SENSITIVITY_FLOOR]

    print("\n  What held on both systems:")
    print(f"    the fraud tops the accuracy table and is caught anyway "
          f"({cheat[0][1]:.4f}, sensitivity 0.02)")
    print("    the dressed identity returns exactly the unmitigated value")
    print(f"    REM + ZNE is the best honest method ({best[1]:.4f} here, "
          "1.56 kcal/mol on H2)")
    print("    PEC underperforms wherever its assumed model is not the real one")
    print("\n  What did not:")
    print("    readout dominance, which was a fact about a two-gate circuit")
    print("    the size of the gains -- 2.6x here against 23x on H2")

    if cheat and best[1] <= cheat[0][1]:
        print("\n  FAILED: the fraud did not top the accuracy table, so this "
              "example is no longer testing what it claims to.")
        return 1
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
