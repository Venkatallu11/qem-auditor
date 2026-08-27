"""Shared fixture builder for the test suite.

Deliberately builds a *clean, complete* record by default so each test
can break exactly one thing and attribute the resulting verdict to that
one thing.
"""
from __future__ import annotations

from qem_auditor import Controls, Experiment, Outputs


def make_experiment(**overrides) -> Experiment:
    controls = Controls(
        ideal_control=True,
        target_leakage_check=True,
        adversarial_check=True,
        reproducibility_checked=True,
    )
    outputs = Outputs(
        raw_error_kcal=1.20,
        mitigated_error_kcal=0.10,
        replicate_errors_kcal=[0.10] * 8,
        q95_kcal=0.20,
        n_replicates_target=8,
    )
    exp = Experiment(
        experiment_id="fixture",
        description="clean fully-certifiable fixture",
        backend="test_backend",
        shots=20_000,
        controls=controls,
        outputs=outputs,
        real_hardware_full_validation=True,
    )
    for key, value in overrides.items():
        if hasattr(exp.controls, key):
            setattr(exp.controls, key, value)
        elif hasattr(exp.outputs, key):
            setattr(exp.outputs, key, value)
        elif hasattr(exp, key):
            setattr(exp, key, value)
        else:
            raise AttributeError(f"no such field on Experiment/Controls/Outputs: {key}")
    return exp
