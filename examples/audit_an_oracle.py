"""Auditing a circuit from outside this project, and what it changed.

Everything before this ran on H2 or an Ising chain -- systems this
project chose. Then an 18-qubit phase oracle arrived from elsewhere: a
64x64 logo bitmap, a specification naming 1097 marked pixels, a depth
target of 726, and a submission file asserting `"verified": true`.

Three things happened, in this order, and the order is the point.

1. It did not parse. `mcx` and `mcz` are not OPENQASM 2.0 gates and are
   not in qelib1.inc, so the file compiles nowhere standard.

2. Once shimmed, it did not compute its specification. It marked ONE of
   4096 basis states, and the wrong one, and left its ancillas entangled
   with the input on every single input -- which means it was not a phase
   oracle at all. This is the check that did not exist here before, and
   it is exact rather than statistical.

3. Only then is mitigation worth discussing, and the answer is no. A
   correct implementation of the same specification needs about 5,900
   two-qubit gates, which on Eagle-class hardware survives one shot in a
   hundred million. That is not a mitigation problem.

Run:  python examples/audit_an_oracle.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import QuantumCircuit, transpile                    # noqa: E402
from qiskit.circuit.library import MCXGate, ZGate               # noqa: E402

from benchmarks.oracle import (GRID, build_oracle, cube_cover,  # noqa: E402
                               disjoint_rectangles, encode,
                               logo_predicate, marked_pixels)
from qem_auditor.prescribe import feasibility                   # noqa: E402
from qem_auditor.reversible import audit_oracle                 # noqa: E402
from real_device_audit import calibration                       # noqa: E402

#: What the submission claimed for itself.
CLAIMED = {"pixels": 1097, "depth": 726, "two_qubit_gates": 2200}

#: Measured on the uploaded file itself, after shimming `mcx`/`mcz` so it
#: would parse at all. Recorded here as data rather than recomputed,
#: because the fixture below reproduces the DEFECTS faithfully and not
#: the full gate count -- and quoting the fixture's depth as the
#: submission's would be the same kind of unchecked claim this example
#: is about.
AS_UPLOADED = {"marks": 1, "depth": 591, "two_qubit_gates": 465}


def broken_oracle() -> QuantumCircuit:
    """The uploaded circuit's three defects, reproduced faithfully.

    Kept as a fixture rather than as the original file so this example
    stands alone, and so each defect can be pointed at individually:

      * the phase is a multi-controlled Z over the twelve COORDINATE
        qubits as well as the flag, so it fires only on x=63,y=63
        whatever the predicate computed -- it should be a phase on the
        flag alone;
      * the ancillas are "uncomputed" by flipping them with X rather
        than by running the computation backwards, so they never return
        to zero;
      * two of the four shapes were never implemented, which the file
        says out loud.
    """
    circuit = QuantumCircuit(18)
    circuit.mcx([1, 2, 3, 4], 12)        # a predicate, computed into an ancilla
    circuit.cx(5, 13)
    # The phase, mis-aimed: controlled on the twelve coordinate qubits as
    # well as the flag, so it fires only on x=63, y=63.
    circuit.append(ZGate().control(12), list(range(12)) + [12])
    circuit.x(range(12, 18))              # the "uncompute" that is not one
    return circuit


def report(title):
    print(f"\n{title}\n" + "-" * len(title))


def main() -> int:
    report("1. The specification")
    print(f"  four shapes -> {len(marked_pixels())} marked pixels "
          f"(the submission claimed {CLAIMED['pixels']})")
    print("  the pixel count is the one thing in the submission that was right.")

    report("2. Does the circuit compute it?")
    broken = broken_oracle()
    verdict = audit_oracle(
        broken, predicate=lambda v: logo_predicate(*divmod(v, GRID)),
        n_inputs=GRID * GRID, encode=lambda v: encode(v) + [0] * 6,
        ancillas=range(12, 18))
    print(verdict.format_report())
    print(f"\n  accuracy would read {verdict.accuracy:.1%}, which is why this")
    print(f"  report never quotes it alone: a circuit doing nothing scores")
    print(f"  {1 - len(verdict.expected) / (GRID * GRID):.1%} against a specification this sparse.")

    report("3. What does correct cost?")
    cubes = cube_cover()
    print(f"  {len(disjoint_rectangles())} disjoint rectangles -> {len(cubes)} cubes")
    correct = build_oracle(cubes=cubes)
    good = audit_oracle(correct, predicate=lambda v: logo_predicate(*divmod(v, GRID)),
                        n_inputs=GRID * GRID, encode=encode, ancillas=())
    print(f"  ancilla-free reference oracle: {good.matches_specification} "
          f"on all {good.n_inputs} inputs")

    flagged = QuantumCircuit(18)
    flagged.x(12); flagged.h(12)
    for cube in cubes:
        pairs = cube.controls()
        zeros = [q for q, b in pairs if b == 0]
        controls = [q for q, _ in pairs]
        if zeros:
            flagged.x(zeros)
        flagged.append(MCXGate(len(controls)), controls + [12])
        if zeros:
            flagged.x(zeros)
    flagged.h(12); flagged.x(12)

    native = transpile(flagged, basis_gates=["u3", "cx"], optimization_level=1)
    print(f"  {'as uploaded (measured)':30s} depth {AS_UPLOADED['depth']:6d}  "
          f"cx {AS_UPLOADED['two_qubit_gates']:6d}   marks "
          f"{AS_UPLOADED['marks']} of {CLAIMED['pixels']}")
    print(f"  {'claimed target':30s} depth {CLAIMED['depth']:6d}  "
          f"cx {CLAIMED['two_qubit_gates']:6d}")
    print(f"  {'correct, same 18q width':30s} depth {native.depth():6d}  "
          f"cx {native.count_ops().get('cx', 0):6d}   marks {len(good.marked)}")
    print("\n  -> the depth target was met by not implementing the function.")

    report("4. Can mitigation rescue it?")
    cal = calibration()
    native = transpile(flagged, basis_gates=["u3", "cx"], optimization_level=1)
    print(feasibility(native.count_ops().get("cx", 0), cal, n_qubits=18).format_verdict())
    print("\n  This is the answer to 'does our noise cancellation work here'.")
    print("  It does not, and saying so is the useful output. A ranked table")
    print("  of methods for this circuit would be a precise answer to a")
    print("  question nobody can ask.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
