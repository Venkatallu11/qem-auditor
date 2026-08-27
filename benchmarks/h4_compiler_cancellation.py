"""QEM-Trust case 1: the executed circuit was not the designed circuit.

Source: quantum-chemistry-vqe, RESEARCH_LEDGER.md retrospective items 3
and 4. Abstract-gate ZNE folding inserted G.G^-1 pairs to amplify noise;
the transpiler recognized them as identities and optimized them straight
back out before submission. The locally-constructed circuit was exactly
right. The circuit that ran had no extra gates in it at all, so the
"noise-amplified" arm was measuring the same noise as the unamplified
one, and the extrapolation was fitting a slope through a variable that
never varied.

The same class of bug bit a second time from the other direction:
`optimization_level>=1` silently collapsed gate counts for SOME fitted
angles and not others (adaptive 2-qubit synthesis finding a cheaper
circuit near periodic special values), breaking CDR's core assumption
that training and target circuits are structurally identical. A
transpiler pass that is locally optimal per circuit is under no
obligation to be structurally consistent across a family of related
circuits -- and "optimize this circuit" and "keep this family
comparable" are different goals that one integer conflates.

This is the cheapest possible gate to run and it invalidates everything
downstream, which is why it sits second in ALL_GATES, right after the
ideal control.

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
    Verdict,
)

EXPERIMENT = Experiment(
    experiment_id="h4_abstract_fold_compiler_cancellation",
    claim="Abstract-gate ZNE folding amplifies noise as designed, enabling extrapolation to the zero-noise limit.",
    description=(
        "Abstract-gate ZNE on the H4 forged-energy circuits: G.G^-1 identity pairs "
        "inserted at the abstract-gate level to amplify noise, submitted through the "
        "standard transpile-and-run path."
    ),
    backend="ionq_simulator",
    shots=20_000,
    circuit=CircuitSpec(
        circuit_id="h4_ef_abstract_fold",
        native_gate_set="abstract (pre-native), folded before transpilation",
        transpilation_status=TranspilationStatus.VERIFIED_MODIFIED,
        optimization_level=1,
    ),
    noise=NoiseSpec(
        noise_model="ionq_simulator aria-1 / forte-1 named profiles",
        calibration_source="IonQ published profile",
    ),
    controls=Controls(
        unitary_equivalence=False,  # the whole point of this record
        ideal_control=None,
        target_leakage_check=None,
        adversarial_check=None,
        reproducibility_checked=False,
    ),
    outputs=Outputs(),
    suspected_failure_modes=[FailureMode.COMPILER_CANCELLATION],
    notes=(
        "Fixed procedurally rather than by patching a number: fold AFTER transpilation "
        "and submit with no further transpiler passes, and hardcode optimization_level=0 "
        "everywhere -- later enforced in code (qforge.ansatz.transpile_fixed has no such "
        "parameter at all) rather than left as a convention each new script had to remember."
    ),
)

EXPECTED_VERDICT = Verdict.INVALID
EXPECTED_PRIMARY_FAILURE_MODE = FailureMode.COMPILER_CANCELLATION
