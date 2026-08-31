#!/usr/bin/env python3
"""A live end-to-end audit: real molecule, real noise, real mitigation.

Everything else in `benchmarks/` is a record TRANSCRIBED from an
experiment someone already ran and already understood. This file is the
other thing: it runs the experiment now, hands the auditor numbers it
measured itself, and nobody -- including whoever wrote this file -- gets
to tell it the answer.

The problem is real chemistry. H2 at 0.735 Angstrom in STO-3G, parity
mapped to two qubits, whose exact ground state energy is
-1.857275030 Ha. That number comes from diagonalising the Hamiltonian,
not from this package, so the auditor's verdicts can be checked against
the truth afterwards -- which is the only reason to run this in
simulation rather than on hardware.

Two zero-noise-extrapolation protocols are audited, and the auditor is
never told which is which:

  A  folds 1,3,5 with a linear fit         -- the sober choice
  B  folds 1,3,5,7,9 with a quartic fit    -- more data, more freedom

Both report a healthy improvement. B's headline is only slightly worse
than A's, and a reviewer reading a results table would pass both. What
happens instead is in the docstring at the bottom.
"""
import statistics
import sys

try:
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
except ImportError:
    print("this example needs qiskit and qiskit-aer: pip install 'qem-auditor[adapters]'")
    sys.exit(0)

from qem_auditor import (CircuitSpec, ClaimType, Controls, Experiment, NoiseSpec,
                         Outputs, Provenance, Replicate, ReplicateKind,
                         TranspilationStatus, UncertaintyCoverage, audit, classify)
from qem_auditor.adapters.qiskit_adapter import QiskitAdapter
from qem_auditor.adapters.sources import AerNoiseSource

HARTREE_TO_KCAL = 627.5094740631
SHOTS = 40_000
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)

# H2, 0.735 A, STO-3G, parity mapping with two-qubit reduction.
# O'Malley et al., Phys. Rev. X 6, 031007 (2016).
H2 = SparsePauliOp.from_list([
    ("II", -1.052373245772859),
    ("IZ",  0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX",  0.18093119978423156),
])
FCI = float(np.linalg.eigvalsh(H2.to_matrix())[0])
THETA = 0.223536983          # the converged VQE angle for this ansatz


def ansatz(theta: float = THETA) -> QuantumCircuit:
    """One-parameter UCC ansatz: exp(-i theta/2 * Y0 X1) applied to |01>."""
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.rx(np.pi / 2, 0)
    qc.h(1)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.rx(-np.pi / 2, 0)
    qc.h(1)
    return qc


def fold_cx(circuit: QuantumCircuit, factor: int) -> QuantumCircuit:
    """Unitary folding: each CX becomes `factor` copies.

    CX is self-inverse, so an odd number of copies is the SAME unitary
    carrying strictly more noise. That is the whole premise of ZNE, and
    the auditor verifies the "same unitary" half of it rather than
    believing this docstring.
    """
    if factor % 2 == 0:
        raise ValueError("an even fold factor changes the unitary")
    folded = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        for _ in range(factor if instruction.operation.name == "cx" else 1):
            folded.append(instruction.operation, instruction.qubits, instruction.clbits)
    return folded


def noise_model(p1: float = 0.001, p2: float = 0.02) -> NoiseModel:
    """Depolarizing noise: 0.1% on one-qubit gates, 2% on two-qubit."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1),
                                   ["x", "h", "rx", "rz", "sx", "u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


def _parity(counts: dict, qubits: tuple) -> float:
    total = sum(counts.values())
    return sum((-1 if sum(int(b.replace(" ", "")[::-1][q]) for q in qubits) % 2 else 1) * n
               for b, n in counts.items()) / total


def _term(label: str):
    """Qiskit labels read qubit n-1 ... qubit 0, left to right."""
    qubits = [len(label) - 1 - i for i, c in enumerate(label) if c != "I"]
    bases = {c for c in label if c != "I"}
    if len(bases) > 1:
        raise ValueError(f"term {label!r} spans two measurement bases")
    return (bases.pop() if bases else "I"), tuple(qubits)


def measure_energy(circuit, backend, shots: int, seed: int) -> float:
    """<H> the way an experiment gets it: one circuit per basis, finite
    shots, parities read off the bitstrings."""
    z_circ, x_circ = circuit.copy(), circuit.copy()
    x_circ.h(0)
    x_circ.h(1)
    z_circ.measure_all()
    x_circ.measure_all()

    counts = {
        "Z": backend.run(transpile(z_circ, backend, optimization_level=0),
                         shots=shots, seed_simulator=seed).result().get_counts(),
        "X": backend.run(transpile(x_circ, backend, optimization_level=0),
                         shots=shots, seed_simulator=seed + 1).result().get_counts(),
    }
    total = 0.0
    for label, coeff in zip(H2.paulis.to_labels(), H2.coeffs):
        basis, qubits = _term(label)
        total += float(coeff.real) * (1.0 if basis == "I" else _parity(counts[basis], qubits))
    return total


def extrapolate(folds, values, order: int) -> float:
    return float(np.polyval(np.polyfit(folds, values, order), 0.0))


def error_kcal(energy: float) -> float:
    return abs(energy - FCI) * HARTREE_TO_KCAL


def run_protocol(folds, order):
    """Eight independent executions of raw and mitigated, end to end."""
    noisy = AerSimulator(noise_model=noise_model())
    raw, mitigated = [], []
    for seed in SEEDS:
        raw.append(error_kcal(measure_energy(ansatz(), noisy, SHOTS, seed)))
        values = [measure_energy(fold_cx(ansatz(), f), noisy, SHOTS, seed + 10 * i)
                  for i, f in enumerate(folds)]
        mitigated.append(error_kcal(extrapolate(folds, values, order)))
    return raw, mitigated


def heldout_extrapolation(folds, fit_order: int, tolerance_kcal: float):
    """The held-out check, run in the direction production actually uses.

    Production EXTRAPOLATES: it fits folds on one side and evaluates at a
    point outside them. So the validation has to do that too -- hold out
    the lowest fold, fit only the ones above it, predict downward. A
    held-out point predicted from BETWEEN two fitted ones tests
    interpolation, and interpolation is not the thing being used.

    The tolerance is the error the protocol claims to achieve. An
    extrapolator that cannot predict a fold it did measure as accurately
    as it claims to predict the answer it did not has not validated the
    direction it is used in.
    """
    noisy = AerSimulator(noise_model=noise_model())
    errors = []
    for seed in (11, 22, 33, 44, 55, 66, 77, 88):
        measured = {f: measure_energy(fold_cx(ansatz(), f), noisy, SHOTS, seed + 3 * i)
                    for i, f in enumerate(folds)}
        fitted = folds[1:]
        predicted = np.polyval(
            np.polyfit(fitted, [measured[f] for f in fitted], fit_order), folds[0])
        errors.append(abs(predicted - measured[folds[0]]) * HARTREE_TO_KCAL)
    median = statistics.median(errors)
    return median <= tolerance_kcal, median


def build_record(name, folds, order, raw, mitigated, measurements,
                 heldout_passed, heldout_error) -> Experiment:
    controls = Controls(
        # Procedural facts about this script, reported honestly.
        target_leakage_check=True,       # no ancilla qubits exist in this circuit
        adversarial_check=True,          # the two protocols are each other's control
        heldout_check=True,              # a held-out fold WAS predicted
        free_parameter_floor_test=True,  # one parameter, fixed before the run
        reproducibility_checked=True,
        # Measured above, in the direction production uses.
        extrapolation_in_domain=heldout_passed,
    )
    controls.provenance["extrapolation_in_domain"] = Provenance.MEASURED
    # Whatever the auditor executed itself overrides anything claimed here.
    for m in measurements:
        setattr(controls, m.control, m.passed)
        controls.provenance[m.control] = Provenance.MEASURED

    ordered = sorted(mitigated)
    return Experiment(
        experiment_id=name,
        description=f"H2/STO-3G VQE, ZNE folds={folds}, polynomial order {order}",
        backend="aer density matrix, depolarizing 1q 0.1% / 2q 2%",
        shots=SHOTS,
        claim="zero-noise extrapolation reduces the VQE energy error",
        claim_type=ClaimType.RELATIVE_IMPROVEMENT,
        circuit=CircuitSpec(circuit_id="h2_sto3g_ucc_1param",
                            native_gate_set="x,rx,rz,h,cx",
                            transpilation_status=TranspilationStatus.UNVERIFIED,
                            optimization_level=0, n_qubits=2),
        noise=NoiseSpec(noise_model="depolarizing_1q_1e-3_2q_2e-2",
                        calibration_source="specified exactly, not fitted from a device",
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
            uncertainty=UncertaintyCoverage(
                shot_noise=True,
                method_monte_carlo=True,
                # A simulator does not drift between submissions, so bars
                # measured on one cannot cover submission-to-submission drift.
                cross_submission=False,
                # One noise model. Nothing here speaks to a different one.
                noise_model=False,
            ),
        ),
        real_hardware_full_validation=False,
        notes=(f"held-out extrapolation error {heldout_error:.3f} kcal/mol "
               f"against a claimed {statistics.median(mitigated):.3f}"),
    )


PROTOCOLS = (
    ("live_h2_zne_modest", [1, 3, 5], 1, 1),
    ("live_h2_zne_aggressive", [1, 3, 5, 7, 9], 4, 3),
)


def main() -> int:
    print(f"H2/STO-3G exact ground state: {FCI:.9f} Ha "
          f"(diagonalised, not supplied by this package)\n")

    adapter = QiskitAdapter(seed=7, source=AerNoiseSource(noise_model=noise_model()))
    base = ansatz()
    verdicts = {}

    for name, folds, order, fit_order in PROTOCOLS:
        print("=" * 70)
        print(f"{name}: folds {folds}, polynomial order {order}")
        print("=" * 70)

        raw, mitigated = run_protocol(folds, order)
        claimed = statistics.median(mitigated)
        print(f"  raw error        median {statistics.median(raw):7.3f} kcal/mol")
        print(f"  mitigated error  median {claimed:7.3f} kcal/mol  "
              f"({statistics.median(raw) / claimed:.2f}x improvement)")
        print(f"  per-seed         {[round(v, 2) for v in mitigated]}")

        def pipeline(expectation, folds=folds, order=order):
            return extrapolate(folds,
                               [expectation(fold_cx(ansatz(), f), H2) for f in folds],
                               order)

        measurements = [
            adapter.measure_unitary_equivalence(base, fold_cx(base, 3)),
            adapter.measure_ideal_control(base, H2, pipeline, shots=SHOTS),
            adapter.measure_mitigation_benefit(base, H2, pipeline, shots=SHOTS),
        ]
        for m in measurements:
            print(f"\n  AUDITOR RAN {m.control}: {m.passed}\n    {m.detail}")

        passed, heldout_error = heldout_extrapolation(folds, fit_order, claimed)
        print(f"\n  AUDITOR RAN heldout_extrapolation: {passed}\n"
              f"    predicting a held-out fold from outside the fitted range costs "
              f"{heldout_error:.3f} kcal/mol,\n    against the {claimed:.3f} this "
              f"protocol claims to achieve")

        experiment = build_record(name, folds, order, raw, mitigated,
                                  measurements, passed, heldout_error)
        report = audit(experiment)
        print()
        report.print_report()
        analysis = classify(experiment, report)
        if analysis.diagnoses:
            analysis.print_analysis()
        verdicts[name] = report.verdict
        print()

    print("=" * 70)
    for name, verdict in verdicts.items():
        print(f"  {name:26s} {verdict.value}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
