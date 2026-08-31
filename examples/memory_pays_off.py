#!/usr/bin/env python3
"""The third audit is cheaper than the first, because of the first two.

A tool that starts from zero every time makes every user rediscover what
the last one already found. This runs three audits of the same family of
circuit and shows what the third one gets for free.

Memory here is keyed on the CIRCUIT, not on the error budget. A budget
says what will help; a circuit says what went wrong last time, which is
the better predictor of what will go wrong this time -- the same ansatz
compiled the same way tends to break the same way.

What memory is allowed to do is reorder and warn. It never contributes to
a verdict: the third circuit is judged on its own evidence, and the last
section demonstrates that by auditing a clean circuit that memory
associates only with failures.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/..")

from qem_auditor import (CircuitSpec, Controls, Experiment, NoiseSpec,  # noqa: E402
                         Outputs, Provenance, Replicate, ReplicateKind,
                         TranspilationStatus, UncertaintyCoverage, Verdict,
                         audit, classify)
from qem_auditor.memory import (CaseMemory, CircuitFingerprint,  # noqa: E402
                                case_from_audit)

#: Controls this example can honestly claim, since it is describing a
#: hypothetical submission rather than running one. The examples that
#: MEASURE controls are the ones with qiskit in them; this one is about
#: memory, and needs no backend at all.
def record(experiment_id: str, unitary_equivalence: bool = True) -> Experiment:
    controls = Controls(
        ideal_control=True, target_leakage_check=True, adversarial_check=True,
        reproducibility_checked=True, unitary_equivalence=unitary_equivalence,
        heldout_check=True, extrapolation_in_domain=True,
        free_parameter_floor_test=True, determinism_check=True,
        mitigation_benefit=True)
    for control in ("unitary_equivalence", "ideal_control", "determinism_check"):
        controls.provenance[control] = Provenance.MEASURED
    return Experiment(
        experiment_id=experiment_id,
        description="two-qubit UCC ansatz submission",
        backend="example_backend", shots=20_000,
        claim="the mitigated result reaches chemical accuracy",
        circuit=CircuitSpec(
            circuit_id="ucc_2q", native_gate_set="x,rx,rz,h,cx", n_qubits=2,
            n_1q_gates=6, n_2q_gates=2,
            transpilation_status=(TranspilationStatus.VERIFIED_EQUIVALENT
                                  if unitary_equivalence
                                  else TranspilationStatus.VERIFIED_MODIFIED)),
        noise=NoiseSpec(noise_model="example", calibration_source="example"),
        controls=controls,
        outputs=Outputs(
            raw_error_kcal=1.20, mitigated_error_kcal=0.10,
            replicates=[Replicate(v, ReplicateKind.INDEPENDENT_SUBMISSION)
                        for v in (0.09, 0.10, 0.11, 0.10, 0.09, 0.12, 0.10, 0.10)],
            q50_kcal=0.10, q95_kcal=0.13, q99_kcal=0.15,
            n_trials=80, n_outlier_trials=0, n_replicates_target=8,
            baseline_error_kcal=1.20, baseline_label="unmitigated",
            uncertainty=UncertaintyCoverage(shot_noise=True, method_monte_carlo=True,
                                            cross_submission=True, noise_model=True)),
        real_hardware_full_validation=True)


def h2_like(depth: int) -> CircuitFingerprint:
    """The same two-qubit UCC ansatz, at slightly different depths --
    what a group actually submits over a week of work."""
    return CircuitFingerprint(
        n_qubits=2, two_qubit_gates=2, one_qubit_gates=6, depth=depth,
        gate_names=("x", "rx", "rz", "h", "cx"),
        observable_terms=5, measurement_bases=2, family="ucc_2q")


def audit_and_remember(memory, name, fingerprint, exp, attacks=()):
    recalled = memory.recall(fingerprint)
    report = audit(exp)
    analysis = classify(exp, report)
    memory.remember(case_from_audit(exp, report, fingerprint, analysis, attacks))
    return recalled, report


def main() -> int:
    memory = CaseMemory()

    print("=" * 72)
    print("  AUDIT 1  a circuit nobody has seen before")
    print("=" * 72)
    # The compiler optimised the folds back out: the classic failure.
    first = record("h2_ucc_monday", unitary_equivalence=False)
    recalled, report = audit_and_remember(
        memory, "run_1", h2_like(8), first, attacks=("T_compiler",))
    print("\n" + recalled.format_recollection())
    print(f"\n  verdict: {report.verdict.value}")
    print(f"  failed:  {', '.join(g.name for g in report.gate_results if g.passed is False)}")

    print("\n" + "=" * 72)
    print("  AUDIT 2  a slightly deeper version of the same ansatz")
    print("=" * 72)
    second = record("h2_ucc_tuesday", unitary_equivalence=False)
    recalled, report = audit_and_remember(
        memory, "run_2", h2_like(9), second, attacks=("T_compiler",))
    print("\n" + recalled.format_recollection())
    print(f"\n  verdict: {report.verdict.value}")

    print("\n" + "=" * 72)
    print("  AUDIT 3  a third, and now memory has something to say")
    print("=" * 72)
    third = record("h2_ucc_wednesday", unitary_equivalence=False)
    recalled, report = audit_and_remember(
        memory, "run_3", h2_like(10), third, attacks=("T_compiler",))
    print("\n" + recalled.format_recollection())
    print(f"\n  verdict: {report.verdict.value}")
    print("\n  A user arriving with this circuit is now told, before running")
    print("  anything, which check to run first and which attack found the")
    print("  problem last time. That is the whole saving: the expensive check")
    print("  goes first instead of last.")

    print("\n" + "=" * 72)
    print("  AUDIT 4  the same circuit family, but this one is CLEAN")
    print("=" * 72)
    clean = record("h2_ucc_thursday_fixed")
    recalled = memory.recall(h2_like(10))
    report = audit(clean)
    print(f"\n  memory associates this shape with {len(recalled.resembling) + len(recalled.seen_before)} "
          "prior cases, every one of them INVALID.")
    print(f"  verdict on this circuit's own evidence: {report.verdict.value}")
    if report.verdict is Verdict.CERTIFIED_UNDER_SCOPE:
        print("\n  Memory advised and did not convict. A method that failed once")
        print("  can still be shown working, which is the difference between an")
        print("  auditor and a reputation system -- and it holds by construction")
        print("  here, because the gates are never handed the memory at all.")
    else:
        print("\n  UNEXPECTED: memory appears to have influenced the verdict.")
        return 1
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
