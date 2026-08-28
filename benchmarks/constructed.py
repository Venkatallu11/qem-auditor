"""Constructed cases: minimal pairs that isolate one discrimination each.

The six disclosed cases in this package are real results whose truth was
settled by history -- someone ran the follow-up work and found out. They
are the better evidence and they are also, as a suite, exhausted: this
package scores 6/6 on them, which says the suite is out of resolution,
not that the auditor is reliable.

Two things were wrong with stopping there. The suite left six gates never
once observed failing, `mitigation_benefit` never observed running at
all, `independent_verification` never observed passing, and no case
anywhere near CERTIFIED UNDER SCOPE -- so an auditor that could not
produce the top verdict at all scored the same as one that could. And
aggregate accuracy is generous to an auditor that has learned which
records tend to look bad, because nothing forces it to react to a
specific difference.

Minimal pairs fix both. Each pair below is two records identical except
in one stated respect, where that respect is supposed to move the
verdict. Credit is all-or-nothing across the two: getting one right and
one wrong is what guessing looks like.

What these pairs do NOT do, and it should be said plainly: they do not
break saturation for this package. qem-auditor scores 12/12 on them and
6/6 on the pairs. That is close to guaranteed and it is not a
compliment -- the truth of a constructed case follows from the record as
written, by the same lattice this package implements, so the tool that
defines the rules will keep passing cases derived from them. Constructed
cases can only ever be hard for an auditor that reasons some OTHER way.
Breaking this package's saturation needs disclosed cases where the
follow-up work disagreed with the verdict, which are found by doing
experiments, not by writing records.

What they do instead is measured, not asserted: `number_reading_auditor`
scores +0.206 partial skill on the disclosed six and looks passable, then
solves 0 of 6 pairs and is disqualified for two false endorsements once
the pairs are in play. Every pair holds the numbers fixed and moves
something a numbers-only reading cannot see. They also make the top of
the lattice reachable for the first time -- before them no case in the
suite reached CERTIFIED UNDER SCOPE, `mitigation_benefit` was never once
observed running, and `independent_verification` was never observed
passing, so an auditor incapable of the top verdict scored the same as
one that could produce it.

These are CONSTRUCTED, and `qem_auditor.trust` scores them under that
label separately from the disclosed six. Their truth follows from the
record as written rather than from anyone's experiment, which is weaker
evidence about quantum error mitigation and stronger evidence about the
auditor. A suite that blended the two would let a tool that has merely
learned the schema look like it had learned the physics.
"""
from qem_auditor import (
    CircuitSpec,
    ClaimType,
    Controls,
    Experiment,
    NoiseSpec,
    Outputs,
    Provenance,
    Replicate,
    ReplicateKind,
    TranspilationStatus,
    UncertaintyCoverage,
    Verdict,
)
from qem_auditor.trust import Case, CaseProvenance, Pair

#: Controls the auditor could execute itself rather than take on trust.
_EXECUTABLE = ("unitary_equivalence", "ideal_control", "determinism_check")


def _clean(experiment_id: str, description: str, **overrides) -> Experiment:
    """A record with nothing wrong with it.

    Every case below starts here and breaks exactly one thing, so the
    verdict difference within a pair is attributable to that one thing
    and to nothing else.
    """
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
        mitigation_benefit=True,
    )
    for control in _EXECUTABLE:
        controls.provenance[control] = Provenance.MEASURED

    outputs = Outputs(
        raw_error_kcal=1.20,
        mitigated_error_kcal=0.10,
        replicates=[Replicate(v, ReplicateKind.INDEPENDENT_SUBMISSION)
                    for v in (0.09, 0.10, 0.11, 0.10, 0.09, 0.12, 0.10, 0.10)],
        q50_kcal=0.10,
        q95_kcal=0.13,
        q99_kcal=0.15,
        n_trials=80,
        n_outlier_trials=0,
        n_replicates_target=8,
        baseline_error_kcal=1.20,
        baseline_label="unmitigated",
        uncertainty=UncertaintyCoverage(shot_noise=True, method_monte_carlo=True,
                                        cross_submission=True, noise_model=True),
    )
    exp = Experiment(
        experiment_id=experiment_id,
        description=description,
        backend="constructed_backend",
        shots=20_000,
        claim="mitigated error reaches chemical accuracy",
        circuit=CircuitSpec(circuit_id="h4_sto3g_uccsd",
                            transpilation_status=TranspilationStatus.VERIFIED_EQUIVALENT),
        noise=NoiseSpec(noise_model="depolarizing_2q_5e-3",
                        calibration_source="constructed",
                        calibration_uncertainty_propagated=True),
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
            raise AttributeError(f"no field {key!r} on the record")
    return exp


# --------------------------------------------------------------------------
# P1. A hard gate that FAILED against one that was never RUN.
#
# The most confused distinction in the whole lattice, and the one an
# auditor built on "does anything look wrong" cannot draw at all: both
# records are missing a passing ideal control, and they deserve opposite
# treatment. One says the mitigation moves a noiseless answer -- that is
# a finding. The other says nobody looked -- that is a gap.
# --------------------------------------------------------------------------
FAILED_IDEAL_CONTROL = _clean(
    "c_ideal_control_failed",
    "mitigation shifts the answer on a noiseless model",
    ideal_control=False,
)
UNRUN_IDEAL_CONTROL = _clean(
    "c_ideal_control_unrun",
    "the noiseless control was never run",
    ideal_control=None,
)

# --------------------------------------------------------------------------
# P2. Eight independent submissions against eight bootstrap resamples of
# one. Byte-identical error numbers and spread; only the KIND differs.
#
# Resampling one submission measures how stable the estimator is on the
# data it already has. It cannot speak to whether a second submission
# would land anywhere near the first, which is the thing reproducibility
# means. An auditor reading the numbers alone cannot tell these apart.
# --------------------------------------------------------------------------
_BOOTSTRAP = _clean(
    "c_replicates_bootstrap",
    "eight bootstrap resamples of a single submission",
)
_BOOTSTRAP.outputs.replicates = [
    Replicate(r.error_kcal, ReplicateKind.BOOTSTRAP_RESAMPLE, source_id="submission_1")
    for r in _BOOTSTRAP.outputs.replicates
]
BOOTSTRAP_REPLICATES = _BOOTSTRAP
INDEPENDENT_REPLICATES = _clean(
    "c_replicates_independent",
    "eight independent submissions",
)

# --------------------------------------------------------------------------
# P3. The same controls, measured by the auditor against taken on the
# claimant's word.
#
# Certification is the one verdict that says "believe this". It should
# not rest on self-report for anything the auditor could have checked
# itself -- and an auditor with no notion of provenance sees two
# identical records here.
# --------------------------------------------------------------------------
_SELF_REPORTED = _clean(
    "c_controls_self_reported",
    "controls passed, all on the claimant's word",
)
for _control in _EXECUTABLE:
    _SELF_REPORTED.controls.provenance[_control] = Provenance.SELF_REPORTED
SELF_REPORTED_CONTROLS = _SELF_REPORTED
MEASURED_CONTROLS = _clean(
    "c_controls_measured",
    "the same controls, executed by the auditor",
)

# --------------------------------------------------------------------------
# P4. Under device noise the mitigation helps, or it does not.
#
# Failing this is not a refutation: a method aimed at a different noise
# regime has not been shown to be broken. But it has not been shown to
# work either, so certification waits and the verdict stops at PROMISING.
# An auditor with only pass and fail must get one of these two wrong.
# --------------------------------------------------------------------------
NO_BENEFIT_UNDER_NOISE = _clean(
    "c_no_benefit_under_noise",
    "clean record; the mitigation does not beat raw under device noise",
    mitigation_benefit=False,
)
BENEFIT_UNDER_NOISE = _clean(
    "c_benefit_under_noise",
    "clean record; the mitigation beats raw under device noise",
)

# --------------------------------------------------------------------------
# P5. Uncertainty that covers everything against uncertainty that covers
# only the noise model it was tuned on.
#
# The bars are the same width. What differs is what they were computed
# over, which is what decides whether the result travels off the model.
# --------------------------------------------------------------------------
_MODEL_ONLY = _clean(
    "c_scope_model_only",
    "bars cover shot noise and the noise model, nothing else",
)
_MODEL_ONLY.outputs.uncertainty = UncertaintyCoverage(shot_noise=True, noise_model=True)
MODEL_ONLY_SCOPE = _MODEL_ONLY
FULL_SCOPE = _clean(
    "c_scope_full",
    "bars cover shot noise, method Monte Carlo, cross-submission and model",
)

# --------------------------------------------------------------------------
# P6. The same shortfall, differently claimed.
#
# Identical numbers: 4.00 -> 1.10 kcal/mol, well outside chemical
# accuracy and a genuine 3.6x reduction. As an absolute-accuracy claim
# the evidence refutes it. As a relative-improvement claim the evidence
# supports it. An auditor that grades the number instead of the claim
# gets exactly one of these wrong, whichever way it leans.
# --------------------------------------------------------------------------
def _shortfall(experiment_id, description, claim_type, claim):
    exp = _clean(experiment_id, description)
    exp.outputs.raw_error_kcal = 4.00
    exp.outputs.mitigated_error_kcal = 1.10
    exp.outputs.replicates = [
        Replicate(v, ReplicateKind.INDEPENDENT_SUBMISSION)
        for v in (1.08, 1.10, 1.12, 1.10, 1.09, 1.13, 1.10, 1.10)
    ]
    exp.outputs.q50_kcal, exp.outputs.q95_kcal, exp.outputs.q99_kcal = 1.10, 1.13, 1.15
    exp.outputs.baseline_error_kcal, exp.outputs.baseline_label = 4.00, "unmitigated"
    exp.claim_type = claim_type
    exp.claim = claim
    return exp


ABSOLUTE_SHORTFALL = _shortfall(
    "c_claim_absolute_shortfall",
    "1.10 kcal/mol, claimed as chemical accuracy",
    ClaimType.ABSOLUTE_ACCURACY,
    "the mitigated result reaches chemical accuracy",
)
RELATIVE_SUCCESS = _shortfall(
    "c_claim_relative_success",
    "the same 1.10 kcal/mol, claimed as a 3.6x reduction",
    ClaimType.RELATIVE_IMPROVEMENT,
    "the mitigation reduces error 3.6x against unmitigated",
)


CASES = [
    Case("c_ideal_control_failed", FAILED_IDEAL_CONTROL, Verdict.INVALID,
         "a hard gate that actively failed. Separates auditors that "
         "distinguish a finding from a gap.",
         # Deliberately unpinned. A failed ideal control says the
         # mitigation moves a noiseless answer; it does not say why --
         # extrapolation, a sign error and a calibration mismatch all
         # produce it. The auditor declining to name a cause here is
         # correct behaviour, and pinning one would score a guess.
         truth_mode=None,
         provenance=CaseProvenance.CONSTRUCTED),
    Case("c_ideal_control_unrun", UNRUN_IDEAL_CONTROL, Verdict.NOT_ESTABLISHED,
         "the same hard gate, never run. Separates auditors that treat "
         "silence as a pass, and auditors that treat it as a failure.",
         provenance=CaseProvenance.CONSTRUCTED),

    Case("c_replicates_bootstrap", BOOTSTRAP_REPLICATES, Verdict.NOT_ESTABLISHED,
         "eight resamples of one submission, with the numbers of eight "
         "independent ones. Separates auditors that read replicate KIND.",
         provenance=CaseProvenance.CONSTRUCTED),
    Case("c_replicates_independent", INDEPENDENT_REPLICATES, Verdict.CERTIFIED_UNDER_SCOPE,
         "genuinely independent replicates. The control member of the "
         "bootstrap pair, and the case that shows an auditor can reach "
         "the top of the lattice at all.",
         provenance=CaseProvenance.CONSTRUCTED),

    Case("c_controls_self_reported", SELF_REPORTED_CONTROLS, Verdict.PROMISING,
         "every control passed on the claimant's word. Separates auditors "
         "that track WHO checked from auditors that read the checkbox.",
         provenance=CaseProvenance.CONSTRUCTED),
    Case("c_controls_measured", MEASURED_CONTROLS, Verdict.CERTIFIED_UNDER_SCOPE,
         "the same controls, executed rather than reported.",
         provenance=CaseProvenance.CONSTRUCTED),

    Case("c_no_benefit_under_noise", NO_BENEFIT_UNDER_NOISE, Verdict.PROMISING,
         "the mitigation does not help under device noise. Separates "
         "auditors that can withhold certification without condemning.",
         provenance=CaseProvenance.CONSTRUCTED),
    Case("c_benefit_under_noise", BENEFIT_UNDER_NOISE, Verdict.CERTIFIED_UNDER_SCOPE,
         "the same record where the mitigation does help.",
         provenance=CaseProvenance.CONSTRUCTED),

    Case("c_scope_model_only", MODEL_ONLY_SCOPE, Verdict.MODEL_CONDITIONAL,
         "bars that cover only the model they were tuned on. Separates "
         "auditors that ask what the uncertainty was computed OVER.",
         provenance=CaseProvenance.CONSTRUCTED),
    Case("c_scope_full", FULL_SCOPE, Verdict.CERTIFIED_UNDER_SCOPE,
         "the same bars, covering every axis the claim travels on.",
         provenance=CaseProvenance.CONSTRUCTED),

    Case("c_claim_absolute_shortfall", ABSOLUTE_SHORTFALL, Verdict.REFUTED,
         "1.10 kcal/mol claimed as chemical accuracy. Separates auditors "
         "that grade the CLAIM from auditors that grade the number.",
         # Also deliberately unpinned: the record is not underpowered and
         # nothing is broken in it. The method simply is not accurate
         # enough, and no failure mode in the grammar names that.
         truth_mode=None,
         provenance=CaseProvenance.CONSTRUCTED),
    Case("c_claim_relative_success", RELATIVE_SUCCESS, Verdict.CERTIFIED_UNDER_SCOPE,
         "the same 1.10 kcal/mol claimed as a 3.6x reduction, which the "
         "evidence supports.",
         provenance=CaseProvenance.CONSTRUCTED),
]

PAIRS = [
    Pair("p_failed_vs_unrun", "c_ideal_control_failed", "c_ideal_control_unrun",
         "the ideal control failed, or was never run",
         "A failed control is evidence against the method. An absent one is "
         "evidence about nobody having looked. Treating them alike either "
         "condemns honest incomplete work or lets silence pass for a control."),
    Pair("p_bootstrap_vs_independent", "c_replicates_bootstrap",
         "c_replicates_independent",
         "the replicates are resamples of one submission, or eight separate ones",
         "Resampling measures estimator stability on data already in hand. It "
         "says nothing about whether a second submission would land near the "
         "first, which is what reproducibility claims."),
    Pair("p_self_reported_vs_measured", "c_controls_self_reported",
         "c_controls_measured",
         "the passing controls were self-reported, or executed by the auditor",
         "Certification is the verdict that says believe this. It should not "
         "rest on the claimant's word for anything the auditor could check."),
    Pair("p_benefit_vs_none", "c_no_benefit_under_noise", "c_benefit_under_noise",
         "under device noise the mitigation helps, or it does not",
         "Not helping is not the same as being broken -- the method may be "
         "aimed at another noise regime. It withholds certification without "
         "condemning, which an auditor with only pass and fail cannot express."),
    Pair("p_scope_model_vs_full", "c_scope_model_only", "c_scope_full",
         "the uncertainty covers only the tuned noise model, or every axis",
         "Identical bars mean different things depending on what they were "
         "computed over. One result travels off the model and one does not."),
    Pair("p_absolute_vs_relative", "c_claim_absolute_shortfall",
         "c_claim_relative_success",
         "the same numbers claimed as absolute accuracy, or as relative improvement",
         "The evidence refutes one claim and supports the other. An auditor "
         "that grades the number rather than the claim must get one wrong."),
]
