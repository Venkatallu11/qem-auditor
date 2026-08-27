"""Real benchmark case #1: a genuine failure, caught by its own project
before it became a false headline.

Source: quantum-chemistry-vqe, RESEARCH_LEDGER.md (iteration 29). The
production all-gate ZNE pipeline's per-Pauli extrapolator was validated
by held-out cross-validation, but that validation only ever tested
INTERPOLATION (predicting a held-out fold from the fitted curve), never
the actual EXTRAPOLATION used in production (predicting fold=0 from data
on the opposite side). Feeding PURE statistical shot noise -- multinomial
-sampled, 100,000 shots x 8 seeds, from the exact noiseless `ideal`
model, zero real hardware noise at all -- through the production 756-
curve two-stage extrapolation code, verbatim, turned a
raw(fold=1)=0.0652 kcal/mol error into ALL-GATE-ZNE(fold=0)=33.48
kcal/mol: a 513x blowup, from shot noise alone, on a model with zero
real noise to correct.

This is exactly what an ideal-control gate exists to catch: if the
noiseless model doesn't recover a sane result, the method is broken
before hardware noise even enters the picture. Expected auditor verdict:
INVALID.
"""
from qem_auditor import Controls, Experiment, Outputs, Verdict

EXPERIMENT = Experiment(
    experiment_id="h4_all_gate_zne_ideal_control",
    description=(
        "All-gate ZNE (756-curve, two-stage held-out extrapolation), ideal-control "
        "test: pure statistical shot noise (100,000 shots x 8 seeds, exact noiseless "
        "`ideal` model, zero real hardware noise) fed through the unmodified production "
        "extrapolation code."
    ),
    backend="ionq_simulator (ideal noise model, zero real hardware noise injected)",
    shots=100_000,
    controls=Controls(
        ideal_control=False,  # the whole point of this record: it failed
        target_leakage_check=None,
        adversarial_check=None,
        reproducibility_checked=False,
    ),
    outputs=Outputs(
        raw_error_kcal=0.0652,
        mitigated_error_kcal=33.48,
        replicate_errors_kcal=[],
        q95_kcal=None,
    ),
    real_hardware_full_validation=False,
    notes=(
        "513x blowup from shot noise alone, worse than the 21x seen in the actual real "
        "hardware run -- root cause was held-out validation testing interpolation, never "
        "the extrapolation direction production actually uses."
    ),
)

# What this case must audit to. Asserted by run_benchmarks.py: if a change
# to the gates ever stops flagging the 513x blowup, the suite fails loudly
# rather than quietly re-blessing a known-bad result.
EXPECTED_VERDICT = Verdict.INVALID
