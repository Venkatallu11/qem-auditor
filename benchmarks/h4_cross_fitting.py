"""QEM-Trust case 4: a principled bias correction whose median improved
and which was rejected anyway.

Source: quantum-chemistry-vqe, RESEARCH_LEDGER.md iteration 35, Task E.
The manifold estimator reconstructs every label as a SELF bilinear form
`a_hat @ P @ a_hat` where `a_hat` is itself a nonlinear fit of the SAME
noisy data. For such a plug-in estimator
`E[a^T P a] = mu^T P mu + Tr(P*Sigma)`: an extra bias term from the fit's
own sampling covariance that ordinary averaging cannot remove. The fix is
textbook and correct -- split each slot's 16 real draws into independent
8+8 halves, fit separately, and reconstruct through the symmetrized CROSS
form, which removes `Tr(P*Sigma)` exactly since `Cov(a_A, a_B)=0` by
construction.

It was tested on 32 real paired bootstrap trials and rejected. The naive
summary looked catastrophic (cross-fitted std 515.4 vs 1.4, "MSE up 5.7
million percent") and was correctly NOT taken at face value: 2 of 32
trials blew up to ~2100 kcal/mol and dominated everything. Excluding
those, cross-fitting's median is marginally BETTER (1.50 vs 1.62) while
its mean (2.31 vs 1.79), std (2.35 vs 1.40) and IQR are all worse, and it
wins 15/30 individual trials -- a coin flip.

The decisive finding was the tail, not the centre. Same-sample
reconstruction produced zero catastrophic outliers in the same test
(0/32); cross-fitting produced 2/32 (6.2%). Halving the data per fit
exposes each half to the estimator's already-documented nonconvex
instability, so a real bias-reduction mechanism is paid for with a real,
measured increase in tail risk.

This is the case that stops an auditor from ranking methods on point
estimates. On the median alone, cross-fitting wins.

Expected auditor verdict: REFUTED -- the claim that cross-fitting
improves the estimator is contradicted by its own trials.
"""
from qem_auditor import (
    CircuitSpec,
    ClaimType,
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
    experiment_id="h4_cross_fitted_manifold",
    claim="Cross-fitted reconstruction removes the plug-in bias and improves on same-sample reconstruction.",
    claim_type=ClaimType.RELATIVE_IMPROVEMENT,
    description=(
        "Symmetrized cross-fitted manifold reconstruction (8+8 independent half-fits per "
        "slot) vs the production same-sample estimator, 32 real paired bootstrap trials "
        "on iteration 31C's real checkpoint."
    ),
    backend="ionq_simulator (forte-1), reanalysis of real checkpointed data",
    shots=20_000,
    circuit=CircuitSpec(
        circuit_id="h4_ef_native_k6",
        native_gate_set="IonQ native (GPi/GPi2/ZZ)",
        transpilation_status=TranspilationStatus.VERIFIED_EQUIVALENT,
        optimization_level=0,
    ),
    noise=NoiseSpec(
        noise_model="forte-1 named profile",
        calibration_source="iteration 31C real checkpoint",
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
        reproducibility_checked=False,
    ),
    outputs=Outputs(
        # Robust re-analysis, the 2 blowup trials excluded, as the ledger reports it.
        mitigated_error_kcal=2.31,   # cross-fitted mean
        baseline_error_kcal=1.79,    # same-sample mean
        baseline_label="same-sample reconstruction",
        q50_kcal=1.50,
        n_trials=32,
        n_outlier_trials=2,          # ~2100 kcal/mol each; same-sample had 0/32
        uncertainty=UncertaintyCoverage(shot_noise=True, method_monte_carlo=True),
    ),
    real_hardware_full_validation=False,
    suspected_failure_modes=[FailureMode.HEAVY_TAIL, FailureMode.OPTIMIZER_INSTABILITY],
    notes=(
        "Mechanism understood, not merely observed: halving the data per fit exposes each "
        "half to fit_pure_state's documented nonconvex instability, so the real "
        "bias-removal benefit is outweighed by a real increase in each half's own "
        "estimation variance. Not adopted."
    ),
)

EXPECTED_VERDICT = Verdict.REFUTED
EXPECTED_PRIMARY_FAILURE_MODE = FailureMode.HEAVY_TAIL
