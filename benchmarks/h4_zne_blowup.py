"""QEM-Trust case 2: a genuine failure, caught by its own project before
it became a false headline.

Source: quantum-chemistry-vqe, RESEARCH_LEDGER.md (iteration 29). The
production all-gate ZNE pipeline's per-Pauli extrapolator was validated
by held-out cross-validation -- but that validation only ever tested
INTERPOLATION (predicting a held-out fold from inside the fitted range),
never the EXTRAPOLATION production actually uses (predicting fold=0 from
data entirely on the opposite side). A held-out check in the wrong
direction is not evidence for the direction used.

Feeding PURE statistical shot noise -- multinomial-sampled, 100,000 shots
x 8 seeds, from the exact noiseless `ideal` model, zero real hardware
noise at all -- through the production 756-curve two-stage extrapolation
code, verbatim, turned a raw(fold=1)=0.0652 kcal/mol error into
ALL-GATE-ZNE(fold=0)=33.48 kcal/mol: a 513x blowup, from shot noise
alone, on a model with zero real noise to correct.

The distinction the auditor should draw here is the one that makes the
finding useful. Not "ZNE failed" -- the physical fold response is fine.
The production ESTIMATOR is ill-conditioned at the point it is evaluated,
so the failure is numerical, not physical, and no amount of better
hardware would fix it.

Expected auditor verdict: INVALID.
"""
from qem_auditor import (
    CircuitSpec,
    Controls,
    Experiment,
    FailureMode,
    NoiseSpec,
    Outputs,
    TranspilationStatus,
    UncertaintyCoverage,
    Verdict,
)

EXPERIMENT = Experiment(
    experiment_id="h4_all_gate_zne_ideal_control",
    claim="All-gate ZNE recovers the zero-noise H4 energy from folded measurements.",
    description=(
        "All-gate ZNE (756-curve, two-stage held-out extrapolation), ideal-control "
        "test: pure statistical shot noise (100,000 shots x 8 seeds, exact noiseless "
        "`ideal` model, zero real hardware noise) fed through the unmodified production "
        "extrapolation code."
    ),
    backend="ionq_simulator (ideal noise model, zero real hardware noise injected)",
    shots=100_000,
    circuit=CircuitSpec(
        circuit_id="h4_ef_native_fold",
        native_gate_set="IonQ native (GPi/GPi2/ZZ)",
        transpilation_status=TranspilationStatus.VERIFIED_EQUIVALENT,
        optimization_level=0,
    ),
    noise=NoiseSpec(
        noise_model="ideal (noiseless) -- the control",
        calibration_source="n/a: no noise to calibrate",
    ),
    controls=Controls(
        ideal_control=False,  # the whole point of this record: it failed
        unitary_equivalence=True,  # the circuits were fine; the estimator was not
        heldout_check=True,        # a held-out check WAS run...
        extrapolation_in_domain=False,  # ...in the wrong direction
        target_leakage_check=None,
        adversarial_check=None,
        reproducibility_checked=False,
    ),
    outputs=Outputs(
        raw_error_kcal=0.0652,
        mitigated_error_kcal=33.48,
        q95_kcal=None,
        uncertainty=UncertaintyCoverage(shot_noise=True),
    ),
    real_hardware_full_validation=False,
    suspected_failure_modes=[FailureMode.EXTRAPOLATION_INSTABILITY],
    notes=(
        "513x blowup from shot noise alone, worse than the 21x seen in the actual real "
        "hardware run -- root cause was held-out validation testing interpolation, never "
        "the extrapolation direction production actually uses."
    ),
)

EXPECTED_VERDICT = Verdict.INVALID
EXPECTED_PRIMARY_FAILURE_MODE = FailureMode.EXTRAPOLATION_INSTABILITY
