"""Shared fixture builder for the test suite.

Deliberately builds a *clean, complete* record by default so each test
can break exactly one thing and attribute the resulting verdict to that
one thing.
"""
from __future__ import annotations

from qem_auditor import (
    CircuitSpec,
    Controls,
    Experiment,
    NoiseSpec,
    Outputs,
    Replicate,
    ReplicateKind,
    TranspilationStatus,
    UncertaintyCoverage,
)


def make_experiment(**overrides) -> Experiment:
    controls = Controls(
        ideal_control=True,
        target_leakage_check=True,
        adversarial_check=True,
        reproducibility_checked=True,
        unitary_equivalence=True,
        heldout_check=True,
        extrapolation_in_domain=True,
        free_parameter_floor_test=True,
        determinism_check=True,
    )
    outputs = Outputs(
        raw_error_kcal=1.20,
        mitigated_error_kcal=0.10,
        replicates=[Replicate(0.10, ReplicateKind.INDEPENDENT_SUBMISSION) for _ in range(8)],
        q50_kcal=0.10,
        q95_kcal=0.20,
        q99_kcal=0.30,
        n_trials=80,
        n_outlier_trials=0,
        n_replicates_target=8,
        uncertainty=UncertaintyCoverage(shot_noise=True, method_monte_carlo=True,
                                        cross_submission=True, noise_model=True),
    )
    exp = Experiment(
        experiment_id="fixture",
        description="clean fully-certifiable fixture",
        backend="test_backend",
        shots=20_000,
        circuit=CircuitSpec(circuit_id="fixture_circuit",
                            transpilation_status=TranspilationStatus.VERIFIED_EQUIVALENT),
        noise=NoiseSpec(noise_model="fixture_noise", calibration_source="fixture"),
        controls=controls,
        outputs=outputs,
        real_hardware_full_validation=True,
    )
    for key, value in overrides.items():
        if key == "replicate_errors_kcal":
            # Convenience: a bare list of numbers means independent
            # replicates, which is what most tests are exercising.
            exp.outputs.replicates = [
                Replicate(v, ReplicateKind.INDEPENDENT_SUBMISSION) for v in value
            ]
            continue
        if hasattr(exp.controls, key):
            setattr(exp.controls, key, value)
        elif hasattr(exp.outputs, key):
            setattr(exp.outputs, key, value)
        elif hasattr(exp, key):
            setattr(exp, key, value)
        else:
            raise AttributeError(f"no such field on Experiment/Controls/Outputs: {key}")
    return exp
