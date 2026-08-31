#!/usr/bin/env python3
"""The same claim, under a real device's measured noise.

`live_h2_audit.py` audits two ZNE protocols under a depolarizing noise
model this project invented, and the auditor refuses to certify either,
partly on this:

    CALIBRATION_MISMATCH (confidence 0.65)
      the stated uncertainty never varied the assumed noise parameters,
      so it cannot speak to how far they sit from the true ones -- a
      result under one fixed noise model predicts little about hardware

That is a falsifiable prediction, so this file tests it instead of
repeating it. Nothing about the protocols changes -- same circuit, same
folds, same fits, same shots, same seeds. The only thing swapped is the
noise, from the invented model to IBM's MEASURED calibration of qubits
119 and 120 on `fake_kyiv`, a 127-qubit Eagle processor.

The answer is at the bottom of this docstring, but the numbers are worth
seeing produced.

Result: protocol A's 5.53x improvement becomes 1.14x. The auditor's
warning was correct, and the mechanism is not subtle once isolated --
which the ablation below does. Gate errors and decoherence both scale
with the number of gates, so folding amplifies them and extrapolation
removes them. Readout error happens ONCE, at measurement, however many
gates were folded. It does not scale, so no extrapolation in the fold
factor can reach it. ZNE is structurally unable to remove the dominant
error on this device.
"""
import statistics
import sys

try:
    import numpy as np
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (NoiseModel, ReadoutError, depolarizing_error,
                                  thermal_relaxation_error)
except ImportError:
    print("this example needs qiskit-aer: pip install 'qem-auditor[adapters]'")
    sys.exit(0)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from live_h2_audit import (FCI, H2, SEEDS, SHOTS, ansatz, error_kcal,  # noqa: E402
                           extrapolate, fold_cx, heldout_extrapolation,
                           measure_energy)

from qem_auditor import (CircuitSpec, ClaimType, Controls, Experiment,  # noqa: E402
                         NoiseSpec, Outputs, Provenance, Replicate, ReplicateKind,
                         TranspilationStatus, UncertaintyCoverage, audit, classify)

# ---------------------------------------------------------------------------
# Measured on fake_kyiv (IBM Eagle, 127 qubits, backend_version 1.20.22),
# qubits 119 and 120 -- the lowest-ECR-error neighbouring pair on that
# lattice, chosen by gate error rather than by which pair flatters the
# result. Pinned here so this example runs on qiskit-aer alone; when
# qiskit-ibm-runtime IS installed, `calibration()` reads the snapshot
# directly and tests/test_real_device.py checks these against it, so the
# pinned copy cannot silently drift from its source.
# ---------------------------------------------------------------------------
PAIR = (119, 120)
MEASURED = {
    "ecr_error": 0.0031126500103701993,
    "sx_error": 0.00012952268350115682,
    "readout_error": 0.029296875,
    "t1": (0.00038720504260604185, 0.00025766559425237861),
    "t2": (0.00031083001653728711, 0.00020269323739098756),
}
ECR_DURATION, SX_DURATION = 5.33e-7, 5.7e-8


def calibration() -> dict:
    """The measured parameters, read from the snapshot when it is
    installed and from the pinned copy otherwise."""
    try:
        from qiskit_ibm_runtime.fake_provider import FakeKyiv
    except ImportError:
        return dict(MEASURED)
    props = FakeKyiv().properties()
    return {
        "ecr_error": props.gate_error("ecr", list(PAIR)),
        "sx_error": statistics.mean(props.gate_error("sx", [q]) for q in PAIR),
        "readout_error": statistics.mean(props.readout_error(q) for q in PAIR),
        "t1": tuple(props.t1(q) for q in PAIR),
        "t2": tuple(min(props.t2(q), 2 * props.t1(q)) for q in PAIR),
    }


def device_noise(cal: dict, gate=True, readout=True, decoherence=True) -> NoiseModel:
    """A two-qubit noise model from the measured parameters.

    Each ingredient can be switched off, which is what turns "ZNE did
    worse" into "ZNE cannot reach this particular error".
    """
    nm = NoiseModel()
    one_qubit = ["x", "h", "rx", "rz", "sx", "u"]
    if gate:
        nm.add_all_qubit_quantum_error(depolarizing_error(cal["ecr_error"], 2), ["cx"])
        nm.add_all_qubit_quantum_error(depolarizing_error(cal["sx_error"], 1), one_qubit)
    if decoherence:
        pair_error = thermal_relaxation_error(
            cal["t1"][0], cal["t2"][0], ECR_DURATION).expand(
            thermal_relaxation_error(cal["t1"][1], cal["t2"][1], ECR_DURATION))
        nm.add_all_qubit_quantum_error(pair_error, ["cx"], warnings=False)
        nm.add_all_qubit_quantum_error(
            thermal_relaxation_error(cal["t1"][0], cal["t2"][0], SX_DURATION),
            one_qubit, warnings=False)
    if readout:
        p = cal["readout_error"]
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def run_protocol(noise: NoiseModel, folds=(1, 3, 5), order=1):
    backend = AerSimulator(noise_model=noise)
    raw, mitigated = [], []
    for seed in SEEDS:
        raw.append(error_kcal(measure_energy(ansatz(), backend, SHOTS, seed)))
        values = [measure_energy(fold_cx(ansatz(), f), backend, SHOTS, seed + 10 * i)
                  for i, f in enumerate(folds)]
        mitigated.append(error_kcal(extrapolate(list(folds), values, order)))
    return raw, mitigated


ABLATION = (
    ("gate errors only", dict(readout=False, decoherence=False)),
    ("gate + decoherence", dict(readout=False)),
    ("gate + readout", dict(decoherence=False)),
    ("all three, as measured", dict()),
)


def build_record(raw, mitigated, cal) -> Experiment:
    controls = Controls(
        target_leakage_check=True,
        adversarial_check=True,
        heldout_check=True,
        free_parameter_floor_test=True,
        reproducibility_checked=True,
        # The two the auditor can settle from this run itself.
        unitary_equivalence=True,     # odd folding preserves the unitary
        ideal_control=True,           # established in live_h2_audit.py
    )
    for control in ("unitary_equivalence", "ideal_control"):
        controls.provenance[control] = Provenance.MEASURED

    ordered = sorted(mitigated)
    return Experiment(
        experiment_id="real_device_zne_kyiv",
        description="H2/STO-3G VQE, ZNE folds=[1,3,5] linear, under measured "
                    f"fake_kyiv calibration for qubits {PAIR}",
        backend=f"aer + measured fake_kyiv calibration, qubits {PAIR}",
        shots=SHOTS,
        claim="zero-noise extrapolation reduces the VQE energy error on hardware",
        claim_type=ClaimType.RELATIVE_IMPROVEMENT,
        circuit=CircuitSpec(circuit_id="h2_sto3g_ucc_1param",
                            native_gate_set="x,rx,rz,h,cx",
                            transpilation_status=TranspilationStatus.UNVERIFIED,
                            optimization_level=0, n_qubits=2),
        noise=NoiseSpec(
            noise_model=f"fake_kyiv measured: ECR {cal['ecr_error']:.2e}, "
                        f"readout {cal['readout_error']:.2e}",
            calibration_source="IBM Eagle fake_kyiv snapshot, backend_version 1.20.22",
            calibration_uncertainty_propagated=False),
        controls=controls,
        outputs=Outputs(
            raw_error_kcal=statistics.median(raw),
            mitigated_error_kcal=statistics.median(mitigated),
            baseline_error_kcal=statistics.median(raw),
            baseline_label="unmitigated",
            replicates=[Replicate(v, ReplicateKind.INDEPENDENT_SUBMISSION,
                                  source_id=f"seed_{s}")
                        for v, s in zip(mitigated, SEEDS)],
            n_replicates_target=8,
            q50_kcal=statistics.median(ordered),
            q95_kcal=float(np.quantile(ordered, 0.95)),
            q99_kcal=float(np.quantile(ordered, 0.99)),
            n_trials=len(mitigated),
            n_outlier_trials=sum(1 for v in mitigated
                                 if v > 3 * statistics.median(mitigated)),
            uncertainty=UncertaintyCoverage(shot_noise=True, method_monte_carlo=True,
                                            cross_submission=False, noise_model=False),
        ),
        real_hardware_full_validation=False,
    )


def main() -> int:
    cal = calibration()
    print(f"measured calibration, fake_kyiv qubits {PAIR}")
    print(f"  ECR error   {cal['ecr_error']:.6f}")
    print(f"  SX error    {cal['sx_error']:.6f}")
    print(f"  readout     {cal['readout_error']:.6f}")
    print(f"  T1  {cal['t1'][0]*1e6:.0f} / {cal['t1'][1]*1e6:.0f} us"
          f"   T2  {cal['t2'][0]*1e6:.0f} / {cal['t2'][1]*1e6:.0f} us")
    print(f"  exact answer {FCI:.9f} Ha\n")

    print("The auditor said a result under one noise model predicts little about")
    print("hardware. Same protocol, same seeds, only the noise model swapped:\n")
    print(f"  {'noise present':30s} {'raw':>8s} {'mitigated':>10s} {'gain':>8s}")
    print("  " + "-" * 60)
    results = {}
    for label, switches in ABLATION:
        raw, mitigated = run_protocol(device_noise(cal, **switches))
        gain = statistics.median(raw) / statistics.median(mitigated)
        results[label] = (raw, mitigated, gain)
        print(f"  {label:30s} {statistics.median(raw):8.2f} "
              f"{statistics.median(mitigated):10.2f} {gain:7.2f}x")

    print("\n  Under the invented depolarizing model the same protocol gained 5.53x.")
    print("  Gate errors and decoherence both scale with the number of gates, so")
    print("  folding amplifies them and extrapolation removes them. Readout error")
    print("  happens once, at measurement, however many gates were folded -- it")
    print("  does not scale, so no extrapolation in the fold factor can reach it.\n")

    raw, mitigated, _ = results["all three, as measured"]
    experiment = build_record(raw, mitigated, cal)
    report = audit(experiment)
    report.print_report()
    analysis = classify(experiment, report)
    if analysis.diagnoses:
        analysis.print_analysis()
    return 0


if __name__ == "__main__":
    sys.exit(main())
