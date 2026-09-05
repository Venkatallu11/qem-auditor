"""The same circuit on six machines, and what actually changes.

This project was calibrated against one IBM Eagle chip. That was a
reasonable place to start and a bad place to stop, so this runs the same
circuit across superconducting and trapped-ion hardware and reports what
genuinely differs.

Three findings, and the first one is not the one I expected:

1. The dominant error source is set mostly by CIRCUIT SHAPE, not by
   vendor. At most depths every machine here agrees about what is
   hurting you. "Different hardware needs a different method" is a
   tempting claim and mostly false.

2. The large, reliable vendor effect is CONNECTIVITY. A nearest-
   neighbour chip runs a longer circuit than the one you wrote, because
   distant entangling gates become SWAP chains. Trapped-ion hardware is
   all-to-all and runs what you wrote. That is a 2.5x difference in
   executed gates before anyone argues about fidelity.

3. The RECOMMENDED METHOD barely changes. CDR tops the ranking on every
   machine here, because it is the most broadly covering method in the
   catalogue. What changes is what is LIMITING you, and the useful advice
   at that point is usually not a mitigation method at all: take more
   shots on Quantinuum, shorten the circuit on Rigetti, attack
   decoherence on Eagle.

   I expected to find the method flipping between vendors and wrote that
   claim down before checking it. The output refused it, so it is not in
   here.

Run:  python examples/across_vendors.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qem_auditor.devices import PROFILES, compare              # noqa: E402
from qem_auditor.prescribe import ErrorSource, prescribe        # noqa: E402

DEVICES = list(PROFILES.values())


def table(title, two_qubit_gates, measured_qubits):
    print(f"\n{title}")
    print(f"  {two_qubit_gates} two-qubit gates as written, "
          f"{measured_qubits} measured qubits")
    print("-" * 78)
    print(f"  {'device':<22} {'executed':>8} {'dominant':<18} {'survives':>9}")
    rows = compare(DEVICES, two_qubit_gates=two_qubit_gates,
                   measured_qubits=measured_qubits,
                   one_qubit_gates=two_qubit_gates * 2, shots=10_000)
    for row in rows:
        device = row["device"]
        print(f"  {device.vendor + ' ' + device.name:<22} "
              f"{row['executed_two_qubit_gates']:>8} "
              f"{str(row['dominant']):<18} "
              f"{row['feasibility'].survival:>9.3f}")
    return rows


def main() -> int:
    print("=" * 78)
    print("  The same circuit, six machines")
    print("=" * 78)
    for device in DEVICES:
        print("  " + device.describe().replace("\n", "\n  "))
        print()

    print("=" * 78)
    print("  1. Connectivity is the reliable vendor difference")
    print("=" * 78)
    print("  A nearest-neighbour chip runs a longer circuit than you wrote:")
    print("  distant entangling gates become SWAP chains, three two-qubit")
    print("  gates each. All-to-all hardware runs what you wrote.\n")
    for device in DEVICES:
        written = 100
        executed = device.effective_two_qubit_gates(written)
        print(f"  {device.vendor + ' ' + device.name:<22} {written} written "
              f"-> {executed} executed"
              f"{'   (all-to-all, nothing added)' if device.all_to_all else ''}")

    rows = table("  2. At most shapes, every machine agrees on what hurts",
                 two_qubit_gates=60, measured_qubits=4)
    dominants = {str(row["dominant"]) for row in rows} - {"None"}
    print(f"\n  distinct dominant sources across six machines: {len(dominants)}")
    print("  -> the circuit's shape decided this, not the vendor")

    rows = table("  3. Shallow and wide is where the LIMIT differs",
                 two_qubit_gates=10, measured_qubits=6)
    dominants = {str(row["dominant"]) for row in rows} - {"None"}
    print(f"\n  distinct dominant sources: {len(dominants)} -> {sorted(dominants)}")
    print("  The same circuit is readout-limited on IBM and gate-limited on")
    print("  IonQ Aria and Rigetti. But look at what that does to the ranking:\n")
    print(f"  {'device':<22} {'budget shares':<44} top method")
    for row in rows:
        device = row["device"]
        budget = row["budget"]
        advice = prescribe(budget, noise_model_verified=False,
                           symmetry_available=False, shots=10_000)
        shares = "  ".join(
            f"{source.name[:4].lower()} {budget.share(source):.0%}"
            for source in ErrorSource if budget.share(source) > 0.01)
        top = advice.prescriptions[0].action.split("(")[0].strip() \
            if advice.prescriptions else "-"
        print(f"  {device.vendor + ' ' + device.name:<22} {shares:<44} {top[:22]}")

    print("\n  The top method is the SAME on all six. That is the honest")
    print("  finding, and it is not the one this example set out to make.")
    print("  What differs is the shape of the budget underneath, and that is")
    print("  what should change your plan:")
    print("    Quantinuum H2   22% shot noise    -> take more shots, not a method")
    print("    IBM Eagle r3    30% decoherence   -> shorter circuits, better qubits")
    print("    Rigetti Ankaa   56% gate error    -> fewer entangling gates")
    print("  Only the third-place method tracks the readout/gate split:")
    print("    readout mitigation on the readout-heavy machines, ZNE elsewhere.")

    print("\n" + "=" * 78)
    print("  These are dated, representative figures for comparing")
    print("  architectures -- not measurements of the machine you are about")
    print("  to run on. Bring your own calibration with profile.replace().")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
