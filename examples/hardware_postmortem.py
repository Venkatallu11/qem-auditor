#!/usr/bin/env python3
"""A real trapped-ion job, read by the auditor.

Every other example here builds an experiment and audits it. This one
starts where a hardware user actually starts: with a job record and a
counts table that came back from a real machine, and no idea what they
support.

The job is real. IonQ `qpu.forte-enterprise-1`, 2026-08-25, 5 qubits,
100 shots, from the sister project this package was built alongside. It
billed $25.79, and that number is the reason half this file exists.

Three questions get asked of it, in the order they matter:

1. Is the data what it claims to be? (counts.py)
2. What does the VENDOR say was done to it? (vendor.py)
3. What would doing this properly cost? (cost.py)

Nothing here needs qiskit. A hardware post-mortem runs in the
dependency-free install, because the person who most needs one has
counts and a bill, not a simulator.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qem_auditor.cost import PRICING, affordable_methods  # noqa: E402
from qem_auditor.counts import check_counts  # noqa: E402
from qem_auditor.prescribe import (ErrorSource,  # noqa: E402
                                   budget_from_calibration)
from qem_auditor.results import shot_noise  # noqa: E402
from qem_auditor.devices import PROFILES  # noqa: E402
from qem_auditor.vendor import cross_check, ionq_job  # noqa: E402

JOB = (Path(__file__).resolve().parent.parent / "tests" / "fixtures"
       / "ionq_forte_job.json")

#: What the submitter believed they were submitting: the circuit as
#: WRITTEN, before the service compiled it to its native basis.
AS_WRITTEN_ONE_QUBIT = 20
AS_WRITTEN_TWO_QUBIT = 11

#: The observable this circuit was built to measure, over the four
#: register qubits. Bit 0 is an ancilla carrying a parity check.
OBSERVABLE = [("XYYX", 1.0)]
TERMS = {"XYYX": [(1.0, (1, 2, 3, 4))]}


def rule(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def main() -> int:
    record = json.loads(JOB.read_text())
    job = ionq_job(record)
    counts = job.counts

    rule("THE JOB, AS THE VENDOR RECORDS IT")
    print(job.describe())

    rule("1. IS THE DATA WHAT IT CLAIMS TO BE?")
    report = check_counts({"XYYX": counts}, declared_shots=job.shots)
    print("  " + report.describe())

    # The failure this check exists for, run on the same real data. A
    # vendor SDK returned raw counts where the client expected
    # probabilities and multiplied by the shot count; every table came
    # back inflated by exactly that factor.
    inflated = {"XYYX": {bits: n * job.shots for bits, n in counts.items()}}
    print("\n  the same data with the real SDK bug applied:")
    for problem in check_counts(inflated, declared_shots=job.shots).problems:
        print("  - " + problem.describe())

    honest = shot_noise({"XYYX": counts}, OBSERVABLE)
    broken = shot_noise(inflated, OBSERVABLE)
    print(f"\n  estimate  {honest.estimate:+.6f} -> {broken.estimate:+.6f}   "
          "(unchanged: an expectation value is a weighted mean)")
    print(f"  sigma     {honest.sigma:.6f} -> {broken.sigma:.6f}   "
          f"({honest.sigma / broken.sigma:.1f}x too tight)")
    print("  The estimate survives the corruption untouched. Only the bar")
    print("  moves, and it moves in the direction that makes a claim look")
    print("  stronger. That is why this is checked before anything else.")

    rule("2. WHAT DID THE SERVICE DO THAT THE RECORD DOES NOT SAY?")
    contradictions = cross_check(
        job, claims_unmitigated=True, declared_shots=job.shots,
        declared_two_qubit_gates=AS_WRITTEN_TWO_QUBIT,
        declared_one_qubit_gates=AS_WRITTEN_ONE_QUBIT)
    if not contradictions:
        print("  nothing the vendor reports contradicts the record")
    for found in contradictions:
        print(found.describe())

    profile = PROFILES["ionq_forte"]
    print("\n  what that does to the error budget, on this device's own"
          " calibration:")
    gate_terms = {}
    for label, one, two in (("as written ", AS_WRITTEN_ONE_QUBIT, AS_WRITTEN_TWO_QUBIT),
                            ("as executed", job.one_qubit_gates, job.two_qubit_gates)):
        budget = budget_from_calibration(
            one_qubit_gates=one, two_qubit_gates=two,
            measured_qubits=5, shots=job.shots,
            one_qubit_error=profile.one_qubit_error,
            two_qubit_error=profile.two_qubit_error,
            readout_error=profile.readout_error)
        gate = budget.contributions[ErrorSource.GATE_STOCHASTIC]
        gate_terms[label.strip()] = gate
        top = max(budget.contributions.items(), key=lambda kv: kv[1])
        print(f"    {label}  {one:4d} 1q + {two:2d} 2q   "
              f"gate error {gate:.4f}   largest term: {top[0].name}")
    growth = gate_terms["as executed"] / gate_terms["as written"]
    print(f"  Compilation held the two-qubit count fixed and multiplied the")
    print(f"  one-qubit count by six, which grew the gate term {growth:.2f}x.")
    print("  A budget built on the gates as written is a budget for a")
    print("  circuit this machine never ran.")
    print()
    print("  Read the other column too, though: at 100 shots the SAMPLING")
    print("  term is larger than every hardware term put together, so the")
    print("  cheapest real improvement to this run is more shots -- which")
    print("  section 3 shows costs almost nothing on this machine.")

    rule("3. WHAT WOULD DOING THIS PROPERLY COST?")
    pricing = PRICING["ionq_forte_enterprise"]
    methods = ["unmitigated", "more shots",
               "symmetry verification (post-selection)",
               "readout error mitigation (REM)",
               "zero-noise extrapolation (ZNE)", "REM then ZNE",
               "Clifford data regression (CDR)", "vnCDR", "Pauli twirling",
               "probabilistic error cancellation (PEC)"]
    print(affordable_methods(methods, budget_usd=170.0, shots=2000,
                             pricing=pricing).describe())

    print("\n  Note which method the shootout crowns and which this table")
    print("  can afford. CDR is the most accurate honest method in this")
    print("  package's own benchmarks and costs 91% of the budget; PEC is")
    print("  in the same prescription and costs 76x it. An auditor that")
    print("  ranks methods on error alone will recommend one nobody can run.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
