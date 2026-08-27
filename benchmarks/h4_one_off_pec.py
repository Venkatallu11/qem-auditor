"""QEM-Trust case 3: a real number, from a real submission, that was
never a result.

Source: quantum-chemistry-vqe, RESEARCH_LEDGER.md iterations 31-32. The
calibrated-PEC + covariance-manifold pipeline produced 0.115 kcal/mol on
one real submission -- comfortably inside chemical accuracy, and the best
number the project had ever seen. The same pipeline, on the same kind of
data, also produced 0.317 and 0.438. All three came from what was
loosely called "the same measurement."

The variance decomposition (iteration 32, Task B) found where the spread
actually lived, and it was not where anyone assumed: shot noise 0.0037,
manifold optimizer 0.234, submission-to-submission 0.070, and the
method's own PEC Monte Carlo 2.11 -- roughly 9x the optimizer, 30x
submission variation, and 570x pure shot noise. The instability was an
under-sampled correction (16 real twirled draws), not hardware
variability. That distinction matters because it changes the remedy
completely: more shots would have addressed the smallest term in the
budget.

This record is deliberately frozen at the moment the 0.115 was in hand,
with only the evidence that existed then. The point of the case is that
the auditor refuses it BEFORE the expensive robustness study that later
disowned it -- on nothing more than the observation that a single
submission's bootstrap bar cannot speak to reproducibility.

Expected auditor verdict: NOT ESTABLISHED. Not INVALID: nothing here was
shown to be wrong, and 0.115 is a real measurement. It simply was never
evidence for the claim it was used to support.
"""
from qem_auditor import (
    CircuitSpec,
    Controls,
    Experiment,
    FailureMode,
    NoiseSpec,
    Outputs,
    Replicate,
    ReplicateKind,
    TranspilationStatus,
    UncertaintyCoverage,
    Verdict,
)

# The 8 "replicates" behind the headline bar: bootstrap resamples of ONE
# submission's integer counts, which is what this project's standard
# "8-seed mean +- std" convention actually meant.
_BOOTSTRAP = [0.115, 0.121, 0.109, 0.118, 0.112, 0.124, 0.107, 0.119]

EXPERIMENT = Experiment(
    experiment_id="h4_calibrated_pec_manifold_one_off",
    claim="Calibrated PEC with covariance-manifold reconstruction reaches chemical accuracy on H4.",
    description=(
        "Calibrated PEC (21 slots, real literal twirling) + covariance-aware manifold "
        "reconstruction, one real ionq_simulator submission, 8-seed bootstrap error bar."
    ),
    backend="ionq_simulator (forte-1)",
    shots=20_000,
    circuit=CircuitSpec(
        circuit_id="h4_ef_native_k6",
        native_gate_set="IonQ native (GPi/GPi2/ZZ)",
        transpilation_status=TranspilationStatus.VERIFIED_EQUIVALENT,
        optimization_level=0,
    ),
    noise=NoiseSpec(
        noise_model="forte-1 named profile, PEC calibrated from measured p_ZZ/p_GPi",
        calibration_source="real measured sweep, single calibration point",
        calibration_uncertainty_propagated=False,
    ),
    controls=Controls(
        ideal_control=True,
        unitary_equivalence=True,
        target_leakage_check=True,
        free_parameter_floor_test=True,
        adversarial_check=True,
        heldout_check=True,
        extrapolation_in_domain=True,
        determinism_check=True,
        reproducibility_checked=True,  # asserted -- on bootstrap replicates
    ),
    outputs=Outputs(
        mitigated_error_kcal=0.115,
        replicates=[
            Replicate(v, ReplicateKind.BOOTSTRAP_RESAMPLE, "submission_0")
            for v in _BOOTSTRAP
        ],
        q95_kcal=0.124,
        uncertainty=UncertaintyCoverage(shot_noise=True),
        n_trials=8,
    ),
    real_hardware_full_validation=False,
    suspected_failure_modes=[FailureMode.MONTE_CARLO_VARIANCE, FailureMode.UNDER_POWERED],
    notes=(
        "The same pipeline also produced 0.317 and 0.438 kcal/mol. Iteration 32's variance "
        "decomposition: PEC Monte Carlo 2.11 vs shot noise 0.0037 -- the reported bar "
        "resampled the smallest term in the budget. A later robustness envelope over "
        "randomized noise-model parameters put Q95 at 51.22 kcal/mol, ~205x over target, "
        "and the 0.115 headline was disowned."
    ),
)

EXPECTED_VERDICT = Verdict.NOT_ESTABLISHED
EXPECTED_PRIMARY_FAILURE_MODE = FailureMode.UNDER_POWERED
