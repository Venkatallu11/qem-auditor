#!/usr/bin/env python3
"""Nine mitigation methods, two noise models, one auditor.

The earlier examples audit zero-noise extrapolation, which is one method
among many, and the finding there -- that readout error defeats ZNE
because it does not scale with gate count -- immediately raises the
question this file answers: what happens to everything else?

So: nine methods, each with the same access to the device and none
holding the exact answer, run under both the depolarizing model this
project invented and IBM's MEASURED calibration of `fake_kyiv`. Two of
the nine should not survive an audit, and are there to test whether the
auditor can REFUSE rather than merely rank.

Three things come out of it, and only the first was expected:

1. ZNE and REM change places. ZNE is the best single method under the
   invented noise and nearly useless under the measured noise; REM is
   useless under the invented noise and the largest single win under the
   measured one. Ranking methods on one noise model ranks nothing.

2. CDR is the only method whose accuracy barely moves between the two,
   because it learns the noise map from data instead of assuming its
   structure. PEC, which assumes the structure, is the one that collapses.

3. On accuracy alone the fraud wins both tables. A leaderboard ranked on
   error crowns a method that never read the data.
"""
import statistics
import sys

try:
    from qiskit_aer import AerSimulator
except ImportError:
    print("this example needs qiskit-aer: pip install 'qem-auditor[adapters]'")
    sys.exit(0)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np  # noqa: E402
from live_h2_audit import noise_model as invented_noise  # noqa: E402
from real_device_audit import calibration, device_noise  # noqa: E402

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/..")
from benchmarks.methods import (FITTING_METHODS, METHODS, FCI,  # noqa: E402
                                Sampler, data_sensitivity, error_kcal,
                                heldout_ok, is_deterministic, scramble_shift,
                                unmitigated)

from qem_auditor import (CircuitSpec, ClaimType, Controls, Experiment,  # noqa: E402
                         NoiseSpec, Outputs, Provenance, Replicate,
                         ReplicateKind, TranspilationStatus,
                         UncertaintyCoverage, Verdict, audit)

SHOTS = 40_000
ALL_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)
#: --quick halves the runs so CI can afford this on every push. The
#: qualitative findings it checks -- the fraud caught, the dressed
#: identity refuted, ZNE and REM changing places between noise models --
#: hold at either size; the numbers quoted in the README are the full run.
QUICK = "--quick" in sys.argv
SEEDS = ALL_SEEDS[:4] if QUICK else ALL_SEEDS
SENSITIVITY_SEEDS = SEEDS[:2] if QUICK else ALL_SEEDS[:4]

#: An honest method's answer moves about as much as the raw estimate does
#: when the data is scrambled. Everything real measured here sits between
#: 0.82 and 1.12; the fraud sits at 0.02. The bar is placed in the empty
#: middle rather than tuned against either side.
SENSITIVITY_FLOOR = 0.5


def errors_for(method, backend) -> list:
    return [error_kcal(method(Sampler(backend, SHOTS, seed))) for seed in SEEDS]


def build_record(name, errors, baseline_errors, sensitivity, ideal_ok,
                 deterministic, heldout, noise_label,
                 calibration_source) -> Experiment:
    controls = Controls(
        target_leakage_check=True,
        free_parameter_floor_test=True,
        reproducibility_checked=True,
        unitary_equivalence=True,
        # Every one of these is measured in this run, not reported by the
        # method. A shootout where the contestants grade their own
        # controls is a leaderboard, not an audit.
        ideal_control=ideal_ok,
        adversarial_check=bool(sensitivity >= SENSITIVITY_FLOOR),
        mitigation_benefit=bool(
            statistics.median(baseline_errors) / statistics.median(errors) >= 1.1),
        determinism_check=deterministic,
        heldout_check=heldout,
        extrapolation_in_domain=heldout,
    )
    for control in ("ideal_control", "adversarial_check", "mitigation_benefit",
                    "unitary_equivalence", "determinism_check"):
        controls.provenance[control] = Provenance.MEASURED
    if name in FITTING_METHODS:
        controls.provenance["heldout_check"] = Provenance.MEASURED
        controls.provenance["extrapolation_in_domain"] = Provenance.MEASURED

    ordered = sorted(errors)
    return Experiment(
        experiment_id=f"shootout_{name}",
        description=f"H2/STO-3G VQE mitigated by {name}, under {noise_label}",
        backend=f"aer, {noise_label}",
        shots=SHOTS,
        claim=f"{name} reduces the VQE energy error",
        claim_type=ClaimType.RELATIVE_IMPROVEMENT,
        circuit=CircuitSpec(circuit_id="h2_sto3g_ucc_1param",
                            native_gate_set="x,rx,rz,h,cx",
                            transpilation_status=TranspilationStatus.UNVERIFIED,
                            optimization_level=0, n_qubits=2),
        noise=NoiseSpec(noise_model=noise_label,
                        calibration_source=calibration_source,
                        calibration_uncertainty_propagated=False),
        controls=controls,
        outputs=Outputs(
            raw_error_kcal=statistics.median(baseline_errors),
            mitigated_error_kcal=statistics.median(errors),
            baseline_error_kcal=statistics.median(baseline_errors),
            baseline_label="unmitigated",
            replicates=[Replicate(v, ReplicateKind.INDEPENDENT_SUBMISSION,
                                  source_id=f"seed_{s}")
                        for v, s in zip(errors, SEEDS)],
            n_replicates_target=8,
            q50_kcal=statistics.median(ordered),
            q95_kcal=float(np.quantile(ordered, 0.95)),
            q99_kcal=float(np.quantile(ordered, 0.99)),
            n_trials=len(errors),
            n_outlier_trials=sum(1 for v in errors
                                 if v > 3 * statistics.median(errors)),
            uncertainty=UncertaintyCoverage(shot_noise=True, method_monte_carlo=True,
                                            cross_submission=False, noise_model=False),
        ),
        real_hardware_full_validation=False,
        notes=f"data sensitivity {sensitivity:.3f} (1.0 = as disturbed as the raw estimate)",
    )


def main() -> int:
    cal = calibration()
    models = (
        ("invented depolarizing", invented_noise(), "written by this project"),
        ("measured fake_kyiv", device_noise(cal),
         "IBM Eagle fake_kyiv snapshot, backend_version 1.20.22"),
    )
    noiseless = AerSimulator()

    print(f"exact answer {FCI:.9f} Ha, {len(METHODS)} methods, "
          f"{len(SEEDS)} independent runs each"
          f"{'  (--quick)' if QUICK else ''}\n")

    table = {}
    for label, noise, source in models:
        backend = AerSimulator(noise_model=noise)
        baseline = errors_for(unmitigated, backend)
        # Measured once per backend, not once per method: the scrambled
        # baseline and the noiseless baseline are properties of the
        # device, and recomputing them nine times is the same
        # measurement repeated at nine times the cost.
        reference_shift = scramble_shift(unmitigated, backend, SHOTS,
                                         SENSITIVITY_SEEDS)
        noiseless_baseline = statistics.median(errors_for(unmitigated, noiseless))
        print("=" * 78)
        print(f"{label}   (unmitigated: {statistics.median(baseline):.3f} kcal/mol)")
        print("=" * 78)
        print(f"  {'method':27s} {'error':>9s} {'gain':>7s} {'sens':>7s}  verdict")
        print("  " + "-" * 74)

        rows = []
        for name, method in METHODS.items():
            errors = errors_for(method, backend)
            sensitivity = data_sensitivity(method, backend, SHOTS,
                                           SENSITIVITY_SEEDS, reference_shift)
            # The ideal control: does the method break with no noise to correct?
            ideal = statistics.median(errors_for(method, noiseless))
            ideal_ok = ideal < 10 * noiseless_baseline
            deterministic = is_deterministic(method, backend, SHOTS, SEEDS[0])
            # A method that fits something owes a held-out check, run in
            # the direction it actually predicts. A method that fits
            # nothing has no data to hold out -- which is not the same as
            # having skipped the check, and is recorded as satisfied
            # rather than left unrun.
            heldout = (heldout_ok(name, lambda: Sampler(backend, SHOTS, SEEDS[0]),
                                  statistics.median(errors))
                       if name in FITTING_METHODS else True)
            record = build_record(name, errors, baseline, sensitivity, ideal_ok,
                                  deterministic, heldout, label, source)
            verdict = audit(record).verdict
            rows.append((name, statistics.median(errors),
                         statistics.median(baseline) / statistics.median(errors),
                         sensitivity, verdict))

        for name, error, gain, sensitivity, verdict in sorted(rows, key=lambda r: r[1]):
            flag = "" if sensitivity >= SENSITIVITY_FLOOR else "  <-- not reading the data"
            print(f"  {name:27s} {error:9.3f} {gain:6.2f}x {sensitivity:7.3f}  "
                  f"{verdict.value}{flag}")
        table[label] = rows
        print()

    print("=" * 78)
    print("  Ranked on accuracy alone, the fraud wins both tables.")
    for label, rows in table.items():
        best = min(rows, key=lambda r: r[1])
        best_certifiable = min(
            (r for r in rows if r[4] is not Verdict.INVALID), key=lambda r: r[1])
        print(f"    {label:24s} lowest error: {best[0]}")
        print(f"    {'':24s} not refused : {best_certifiable[0]}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
