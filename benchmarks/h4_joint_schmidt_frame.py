"""QEM-Trust case 5: the project's largest real, validated improvement --
which still must not be certified.

Source: quantum-chemistry-vqe, RESEARCH_LEDGER.md iteration 36. A shared
Schmidt frame `U(theta) = U0 * expm(A(theta))` (A skew-symmetric, 15 free
parameters, orthogonality exact by construction) fitted jointly across
all 21 slots by nonlinear least squares, replacing 21 independent
per-slot fits. Motivated by an exact algebraic identity the production
code already relied on for minus-slots, checked directly against the code
before anything was built.

The bias-variance trade was stated UP FRONT, not discovered afterward:
imposing a shared-frame constraint on noisy data is a regularizer, not a
free improvement, and there is no physical law forcing slot (u_n+u_m)'s
actual noise deviation to equal the algebraic combination of the others'.

Evidence collected:
- Ideal-data recovery: 0.000000 kcal/mol, chi2/dof exactly 0.
- Adversarial rejection: real data fits 52x better than the same data
  with labels shuffled within each slot -- the model explains real
  structure rather than absorbing noise.
- A real nondeterminism bug found, root-caused and fixed, not papered
  over: identical-seed reruns landed in different local optima 40-50% of
  the time (chi2/dof 18-28x worse). Python's per-process hash
  randomization reordered the residual vector, and floating-point
  summation is not associative. More restarts made it WORSE. Fixed at the
  environment level after a local sorted() proved insufficient, then
  re-verified 6/6 identical.
- 80 paired bootstrap trials: MSE 2.8720 -> 0.3212 (88.8% reduction),
  Q95 4.14 -> 1.29, wins 66/80. Zero outliers above 20 kcal/mol in either
  method -- unlike cross-fitting, this regularization does not buy its
  central improvement with tail risk.
- 29 randomized Forte-like noise models: Q95 461.72 -> 16.48 (96.4%
  reduction), 5/29 catastrophic outliers -> 1/29. The win generalizes
  across noise models, not just resamples.

And still it is not certified. The uncertainty never included
cross-submission drift -- every trial resamples ONE real data collection,
while this project separately measured real drift of +-2.31 (forte-1) and
+-4.01 (aria-1) kcal/mol between independent submissions under the same
named noise profile.

Expected auditor verdict: VALID UNDER MODEL. The improvement is real and
adversarially validated; it is established under the noise models tested
and not under the platform's own run-to-run behaviour.
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
    experiment_id="h4_joint_schmidt_frame",
    claim="A jointly-fitted shared Schmidt frame beats 21 independent per-slot fits.",
    claim_type=ClaimType.RELATIVE_IMPROVEMENT,
    description=(
        "Joint 15-parameter shared Schmidt frame (skew-symmetric exponential map, "
        "Procrustes anchor, Levenberg-Marquardt over all 21 slots simultaneously) vs "
        "21 independent fit_pure_state calls. 80 paired bootstrap trials plus 29 "
        "randomized noise-model draws, on iteration 31C's real checkpoint."
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
        noise_model="randomized Forte-like envelope (p_ZZ, p_GPi, p_GPi2, coherent "
                    "angle bias per gate TYPE, readout, PEC calibration mismatch)",
        calibration_source="intervals justified by real measured sweeps",
        calibration_uncertainty_propagated=True,
    ),
    controls=Controls(
        ideal_control=True,          # 0.000000 kcal/mol, chi2/dof exactly 0
        unitary_equivalence=True,
        target_leakage_check=True,   # Procrustes anchor, no target peeking
        free_parameter_floor_test=True,
        adversarial_check=True,      # 52x vs shuffled labels
        heldout_check=True,
        extrapolation_in_domain=True,
        determinism_check=True,      # after the hash-order bug was found and fixed
        reproducibility_checked=False,
    ),
    outputs=Outputs(
        # Medians on both sides -- the headline and the baseline must be
        # the SAME statistic. (MSE: 2.8720 -> 0.3212, an 88.8% reduction,
        # recorded in the notes rather than mixed in here.)
        mitigated_error_kcal=0.2920,
        baseline_error_kcal=1.1788,
        baseline_label="21 independent per-slot fits (median)",
        q50_kcal=0.2920,
        q95_kcal=1.29,
        q99_kcal=2.44,
        n_trials=80,
        n_outlier_trials=0,            # 0/80, vs cross-fitting's 2/32
        # No independent replicates exist: all 80 trials, and all 29
        # noise-model draws, resample ONE real data collection. Recording
        # them as replicates would claim reproducibility evidence this
        # result does not have.
        replicates=[],
        uncertainty=UncertaintyCoverage(
            shot_noise=True,
            method_monte_carlo=True,
            noise_model=True,
            cross_submission=False,   # the gap that blocks certification
        ),
    ),
    real_hardware_full_validation=False,
    suspected_failure_modes=[FailureMode.DRIFT],
    notes=(
        "Largest real validated improvement across 36 iterations. Two limits disclosed "
        "by the project itself and not yet closed: no cross-submission drift in the bars, "
        "and Q95 ~1.29 kcal/mol still over the loosened 0.5 hardware acceptance bar -- "
        "reliability, not central tendency, remains the open question."
    ),
)

EXPECTED_VERDICT = Verdict.MODEL_CONDITIONAL
EXPECTED_PRIMARY_FAILURE_MODE = FailureMode.DRIFT
