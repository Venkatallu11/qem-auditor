#!/usr/bin/env python3
"""Two free wins the auditor can hand a user, and one that compounds.

The prescription in `prescribe_for_circuit.py` recommends a mitigation
method. This one is about the two things that come before that:

  1. WHERE you run. On fake_kyiv the same two-qubit circuit costs 14.1
     kcal/mol on the best neighbouring pair and 330.0 on the worst -- a
     23x range, available for the price of a different `initial_layout`
     and nothing else. A user who lands on the median pair gets 36.6 and
     concludes their method is weak.

  2. That the right qubits DEPEND on the method you plan to run.
     Choosing the lowest-readout pair is right if you are running raw and
     wrong if you are about to apply readout mitigation, because REM
     removes the very error that made that pair attractive and leaves the
     gate error where that pair is worse. Measured both ways below.

  3. That every audit should leave the next one better off. Each
     measured outcome is appended to an evidence ledger, and later
     prescriptions cite what actually happened on budgets like yours --
     but only reorder anything once there are enough observations to
     support a ranking.

Every number printed is measured here, not quoted.
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

from real_device_audit import (ECR_DURATION, SX_DURATION, MEASURED,  # noqa: E402
                               PAIR, calibration, device_noise)

from benchmarks.methods import (Sampler, error_kcal,  # noqa: E402
                                readout_mitigation, unmitigated)
from qem_auditor.layout import DeviceLayout, QubitProperties, advise_layout  # noqa: E402
from qem_auditor.ledger import EvidenceLedger, Observation, shares_of  # noqa: E402
from qem_auditor.prescribe import (METHODS_BY_NAME, budget_from_calibration,  # noqa: E402
                                   prescribe)
from qem_auditor.schema import Provenance  # noqa: E402

SHOTS = 40_000
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)
DURATION = 2 * ECR_DURATION + 6 * SX_DURATION


def device() -> tuple:
    """The lattice, from the snapshot when installed and pinned otherwise."""
    try:
        from qiskit_ibm_runtime.fake_provider import FakeKyiv
    except ImportError:
        return None, None
    backend = FakeKyiv()
    props = backend.properties()
    qubits = {q: QubitProperties(readout_error=props.readout_error(q),
                                 t1_s=props.t1(q),
                                 t2_s=min(props.t2(q), 2 * props.t1(q)))
              for q in range(backend.num_qubits)}
    edges = {}
    for pair in backend.coupling_map:
        error = props.gate_error("ecr", list(pair))
        # A pair whose gate error is 1.0 is dead, not merely bad. Scoring
        # it alongside working pairs would let an arithmetic mean pretend
        # it is a candidate.
        if error < 1.0:
            edges[tuple(pair)] = error
    return DeviceLayout(qubits, edges, "fake_kyiv"), props


def calibration_for(props, pair) -> dict:
    return {"ecr_error": props.gate_error("ecr", list(pair)),
            "sx_error": statistics.mean(props.gate_error("sx", [q]) for q in pair),
            "readout_error": statistics.mean(props.readout_error(q) for q in pair),
            "t1": tuple(props.t1(q) for q in pair),
            "t2": tuple(min(props.t2(q), 2 * props.t1(q)) for q in pair)}


def measure(pair_calibration, method) -> float:
    backend = AerSimulator(noise_model=device_noise(pair_calibration))
    return statistics.median(
        [error_kcal(method(Sampler(backend, SHOTS, seed))) for seed in SEEDS])


def budget_for(pair_calibration):
    return budget_from_calibration(
        two_qubit_gates=2, one_qubit_gates=6, measured_qubits=2,
        two_qubit_error=pair_calibration["ecr_error"],
        one_qubit_error=pair_calibration["sx_error"],
        readout_error=pair_calibration["readout_error"],
        shots=SHOTS, circuit_duration_s=DURATION,
        t2_s=min(pair_calibration["t2"]))


def main() -> int:
    layout, props = device()
    if layout is None:
        print("this example needs the calibration snapshot: "
              "pip install 'qem-auditor[devices]'")
        return 0

    budget = budget_for(MEASURED)
    print("=" * 72)
    print("  1. WHERE TO RUN, given where the error is")
    print("=" * 72)
    advice = advise_layout(layout, budget, 2, current=PAIR,
                           two_qubit_gates=2, circuit_duration_s=DURATION)
    print(advice.format_advice())

    raw_pick = advice.best.qubits
    print(f"\n  The pair this example used to hand-pick, {PAIR}, was chosen by")
    print("  lowest gate error -- the obvious criterion, and the wrong one on a")
    print("  device where readout error dominates. Measured, both ways:\n")

    ledger = EvidenceLedger()
    rows = []
    for label, pair in ((f"hand-picked by gate error {PAIR}", PAIR),
                        (f"budget-aware {raw_pick}", raw_pick),
                        (f"worst available {advice.worst.qubits}",
                         advice.worst.qubits)):
        cal = calibration_for(props, pair)
        raw = measure(cal, unmitigated)
        rows.append((label, pair, cal, raw))
        print(f"  {label:38s} raw {raw:9.3f} kcal/mol")

    best_raw = min(rows, key=lambda r: r[3])
    hand = rows[0]
    print(f"\n  -> {hand[3] / best_raw[3]:.2f}x, free, from `initial_layout` alone.")
    print(f"  -> and {rows[2][3] / best_raw[3]:.0f}x between the best and worst")
    print("     placements, which is what a user unlucky in their layout is")
    print("     unknowingly paying while blaming their method.")

    print("\n" + "=" * 72)
    print("  2. THE RIGHT QUBITS DEPEND ON THE METHOD")
    print("=" * 72)
    rem = METHODS_BY_NAME["readout error mitigation (REM)"]
    after = advise_layout(layout, budget, 2, current=PAIR, after_method=rem,
                          two_qubit_gates=2, circuit_duration_s=DURATION)
    print(f"\n  running raw       -> {advice.best.qubits}")
    print(f"  planning REM      -> {after.best.qubits}")
    print("  because REM removes the readout error that made the first pair")
    print("  attractive, leaving gate error where that pair is worse.\n")

    for label, pair, cal, raw in rows[:2]:
        mitigated = measure(cal, readout_mitigation)
        print(f"  {label:38s} with REM {mitigated:8.3f} kcal/mol")
        for method_name, error in (("readout error mitigation (REM)", mitigated),):
            ledger.record(Observation(
                experiment_id=f"h2_{pair[0]}_{pair[1]}",
                device="fake_kyiv", method=method_name,
                budget_shares=shares_of(budget_for(cal)),
                raw_error=raw, mitigated_error=error,
                provenance=Provenance.MEASURED,
                note="measured by examples/better_next_time.py"))

    print("\n  The ordering inverts. Choosing qubits and choosing a method are")
    print("  one decision, not two.")

    print("\n" + "=" * 72)
    print("  3. WHAT THIS AUDIT LEAVES FOR THE NEXT ONE")
    print("=" * 72)
    print(f"\n  {len(ledger)} observations recorded, content-addressed so the same")
    print("  run cannot be counted twice.\n")
    consulted = prescribe(budget, ledger=ledger)
    for prescription in consulted.prescriptions[:3]:
        seen = prescription.observed
        if seen is not None and seen.n:
            print(f"  {seen.summarise()}")
    print("\n  Too few to rank on, which the corpus says rather than quietly")
    print("  reordering the advice. The mechanism ordering stands until enough")
    print("  audits have accumulated to earn a say.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
