"""Hard gates -- each one returns a GateResult, never a bare bool, so the
verdict layer can explain *why* something failed, not just that it did.

These are deliberately simple, auditable functions over an Experiment
record. No LLM involvement here on purpose: an AI should never be the
thing that decides whether a scientific claim passed. It can propose
experiments and interpret results in prose, but the gates are plain,
inspectable code.

Every gate below traces to a specific real disqualification in this
project's history. The docstrings name which one, so a reader can check
the gate against the failure it is supposed to catch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Optional

from .schema import (
    Experiment,
    Provenance,
    FailureMode,
    ReplicateKind,
    TranspilationStatus,
)


@dataclass
class GateResult:
    name: str
    passed: Optional[bool]  # None = not applicable / not enough data to judge
    reason: str
    implicates: list[FailureMode] = field(default_factory=list)
    """Failure modes this gate's FAILURE points to. Empty on pass."""


# --------------------------------------------------------------------------
# Phase 1 gates
# --------------------------------------------------------------------------

def ideal_control_gate(exp: Experiment) -> GateResult:
    """If a noiseless/ideal-model run of the SAME production code doesn't
    recover a near-zero (or at least non-degraded) result, the method is
    broken independent of any real hardware noise -- an automatic
    disqualifier. (This is exactly what caught the 513x ZNE blowup.)"""
    ic = exp.controls.ideal_control
    if ic is None:
        return GateResult("ideal_control", None, "not yet run")
    if ic is False:
        return GateResult("ideal_control", False,
                          "ideal/noiseless control did NOT recover a sane result -- "
                          "the method is broken independent of real hardware noise",
                          [FailureMode.EXTRAPOLATION_INSTABILITY, FailureMode.UNKNOWN])
    return GateResult("ideal_control", True, "ideal control passed")


def target_leakage_gate(exp: Experiment) -> GateResult:
    tl = exp.controls.target_leakage_check
    if tl is None:
        return GateResult("target_leakage", None, "not yet run")
    if tl is False:
        return GateResult("target_leakage", False,
                          "evidence of target leakage -- the known exact answer was used "
                          "to select or tune the method",
                          [FailureMode.TARGET_LEAKAGE])
    return GateResult("target_leakage", True, "no target leakage detected")


def adversarial_gate(exp: Experiment) -> GateResult:
    """Passed means the adversarial/negative controls (wrong parity,
    shuffled labels, wrong sign, etc.) DID fail loudly, as a genuine
    effect requires. If they did NOT fail -- i.e. garbage-in produced a
    similarly good-looking result -- the claim is not trustworthy."""
    adv = exp.controls.adversarial_check
    if adv is None:
        return GateResult("adversarial", None, "not yet run")
    if adv is False:
        return GateResult("adversarial", False,
                          "adversarial/negative controls did NOT fail as required -- "
                          "the result may be an artifact, not a genuine effect",
                          [FailureMode.TARGET_LEAKAGE, FailureMode.BOOKKEEPING])
    return GateResult("adversarial", True, "adversarial controls failed loudly, as required")


def reproducibility_gate(exp: Experiment, rel_tolerance: float = 0.5) -> GateResult:
    """Do the replicates agree with each other, and are there enough of
    them? Judged on INDEPENDENT replicates only -- see
    replicate_independence_gate for why bootstrap replicates cannot
    substitute."""
    reps = exp.outputs.independent_errors_kcal
    n_independent = len(exp.outputs.independent_replicates)
    n_target = exp.outputs.n_replicates_target
    if not exp.controls.reproducibility_checked or len(reps) < 2:
        withheld = n_independent - len(reps)
        note = (f"; {withheld} independent replicate(s) recorded without a value"
                if withheld else "")
        return GateResult("reproducibility", None,
                          f"insufficient independent replicate values to judge "
                          f"({len(reps)} usable){note}")
    m = mean(reps)
    spread = pstdev(reps)
    agree = spread <= rel_tolerance * m if m > 0 else spread == 0.0
    if not agree:
        return GateResult("reproducibility", False,
                          f"{len(reps)} independent replicates disagree beyond tolerance "
                          f"(mean={m:.4f}, spread={spread:.4f} kcal/mol)",
                          [FailureMode.DRIFT, FailureMode.MONTE_CARLO_VARIANCE,
                           FailureMode.OPTIMIZER_INSTABILITY])
    note = "" if len(reps) >= n_target else f" -- below this project's own {n_target}-replicate target"
    return GateResult("reproducibility", True,
                      f"{len(reps)}/{n_target} independent replicates collected and mutually "
                      f"consistent (mean={m:.4f} kcal/mol){note}")


def chemical_accuracy_gate(exp: Experiment, threshold_kcal: float = 0.25) -> GateResult:
    """Informational, not a hard fail: reports whether the best available
    uncertainty estimate (Q95 if present, else the point mitigated error)
    clears the standard 0.25 kcal/mol chemical-accuracy bar.

    Applies only to ABSOLUTE_ACCURACY claims. Grading a relative
    improvement against an absolute bar would call the joint Schmidt frame
    -- an 88.8% MSE reduction, this project's largest validated result --
    a failure, which it is not.
    """
    from .schema import ClaimType

    if exp.claim_type is ClaimType.RELATIVE_IMPROVEMENT:
        return GateResult("chemical_accuracy", None,
                          "not applicable: this is a relative-improvement claim, graded "
                          "against its baseline by improvement_gate")
    value = exp.outputs.q95_kcal if exp.outputs.q95_kcal is not None else exp.outputs.mitigated_error_kcal
    if value is None:
        return GateResult("chemical_accuracy", None, "no mitigated error or Q95 recorded")
    passed = value < threshold_kcal
    basis = "Q95" if exp.outputs.q95_kcal is not None else "point estimate"
    return GateResult("chemical_accuracy", passed,
                      f"{basis}={value:.4f} kcal/mol vs {threshold_kcal} kcal/mol target",
                      [] if passed else [FailureMode.UNDER_POWERED])


# --------------------------------------------------------------------------
# Phase 3 gates -- each from a specific real disqualification
# --------------------------------------------------------------------------

def unitary_equivalence_gate(exp: Experiment) -> GateResult:
    """Is the circuit that RAN the circuit that was designed?

    From the abstract-gate-folding failure: ZNE inserted G.G^-1 pairs that
    the compiler optimized straight back out before submission. The
    locally-constructed circuit was correct and the executed one was a
    different circuit, so every number downstream described an experiment
    nobody intended to run.
    """
    status = exp.circuit.transpilation_status
    explicit = exp.controls.unitary_equivalence
    if explicit is False or status is TranspilationStatus.VERIFIED_MODIFIED:
        return GateResult("unitary_equivalence", False,
                          "the SUBMITTED circuit does not implement the intended unitary -- "
                          "the executed experiment is not the designed one",
                          [FailureMode.COMPILER_CANCELLATION])
    if explicit is True or status is TranspilationStatus.VERIFIED_EQUIVALENT:
        return GateResult("unitary_equivalence", True,
                          "submitted circuit verified equivalent to the intended unitary")
    return GateResult("unitary_equivalence", None,
                      "submitted circuit never checked against the intended circuit")


def extrapolation_domain_gate(exp: Experiment) -> GateResult:
    """Was the method validated in the direction production actually uses?

    The 513x blowup passed its own held-out cross-validation. That
    validation only ever tested INTERPOLATION -- predicting a held-out
    fold from inside the fitted range -- while production EXTRAPOLATED to
    fold=0 from data entirely on one side. A held-out check in the wrong
    direction is not evidence for the direction used.
    """
    heldout = exp.controls.heldout_check
    in_domain = exp.controls.extrapolation_in_domain
    if in_domain is False:
        return GateResult("extrapolation_domain", False,
                          "production extrapolates outside the domain its held-out "
                          "validation tested -- the validation does not cover the use",
                          [FailureMode.EXTRAPOLATION_INSTABILITY])
    if heldout is False:
        return GateResult("extrapolation_domain", False,
                          "no held-out validation: the method was never tested against "
                          "data it did not see while fitting",
                          [FailureMode.EXTRAPOLATION_INSTABILITY])
    if in_domain is True and heldout is True:
        return GateResult("extrapolation_domain", True,
                          "held-out validation covers the direction production uses")
    return GateResult("extrapolation_domain", None,
                      "held-out validation direction not recorded")


def replicate_independence_gate(exp: Experiment) -> GateResult:
    """Bootstrap resampling is not replication.

    This project ran the identical circuit set through IonQ's simulator
    twice, independently, and the two energies differed by 3.27 kcal/mol
    (aria-1) and 2.94 (forte-1) -- three times its own reproducibility
    bar. Its standard '8-seed mean +- std' bars never showed this, because
    they resample ONE submission's counts and so measure shot noise only.
    A record that offers bootstrap replicates where independent ones are
    required is claiming evidence it does not have.
    """
    reps = exp.outputs.replicates
    if not reps:
        return GateResult("replicate_independence", None, "no replicates recorded")
    independent = exp.outputs.independent_replicates
    dependent = [r for r in reps if not r.kind.is_independent]
    if not independent and dependent:
        kinds = sorted({r.kind.name for r in dependent})
        return GateResult("replicate_independence", False,
                          f"all {len(dependent)} replicate(s) are {'/'.join(kinds)} -- "
                          "resampling one execution measures shot noise, not "
                          "run-to-run reproducibility",
                          [FailureMode.DRIFT, FailureMode.UNDER_POWERED])
    note = f" ({len(dependent)} additional non-independent replicate(s) not counted)" if dependent else ""
    return GateResult("replicate_independence", True,
                      f"{len(independent)} genuinely independent replicate(s) recorded{note}")


def determinism_gate(exp: Experiment) -> GateResult:
    """Does re-running the identical computation give the identical result?

    Hit twice in this project, both times silently. Python's per-process
    hash randomization reordered a residual vector fed to a nonconvex
    solver, and floating-point summation is not associative, so ~40-50% of
    identical-seed runs landed in a different local optimum (chi2/dof 18-28x
    worse). Separately, hash()-derived bootstrap seeds gave different
    numbers on every rerun against the SAME checkpointed data. Neither is
    visible from a single run.
    """
    det = exp.controls.determinism_check
    if det is None:
        return GateResult("determinism", None, "not yet run")
    if det is False:
        return GateResult("determinism", False,
                          "identical inputs did NOT reproduce an identical result across "
                          "runs -- the number reported is one draw from an unstated distribution",
                          [FailureMode.NONDETERMINISM, FailureMode.OPTIMIZER_INSTABILITY])
    return GateResult("determinism", True, "identical inputs reproduce identical results")


def free_parameter_floor_gate(exp: Experiment) -> GateResult:
    """Does every free parameter have a floor?

    The locally-perturbed CDR method was disqualified because its
    training-perturbation radius had none: as the radius shrinks toward 0
    the training circuit converges to the target circuit itself, so the
    method degenerates into classically re-evaluating the answer it was
    supposed to be measuring. Any simulator-only testbed can cheat this
    way, because 'exact' is one Statevector call away.
    """
    floor = exp.controls.free_parameter_floor_test
    if floor is None:
        return GateResult("free_parameter_floor", None, "not yet run")
    if floor is False:
        return GateResult("free_parameter_floor", False,
                          "a free parameter has no floor -- in its limit the method "
                          "degenerates toward re-evaluating the known answer",
                          [FailureMode.FREE_PARAMETER_DEGENERACY, FailureMode.TARGET_LEAKAGE])
    return GateResult("free_parameter_floor", True,
                      "every free parameter has a validated floor")


def tail_risk_gate(exp: Experiment, max_outlier_fraction: float = 0.02,
                   max_q99_ratio: float = 10.0) -> GateResult:
    """Reliability, not central tendency.

    Cross-fitted reconstruction had a marginally BETTER median (1.50 vs
    1.62) and was still rejected: 2 of 32 trials blew up to ~2100
    kcal/mol, a tail risk the same-sample method showed zero of (0/32).
    A method whose median improves while its tail explodes has not
    improved. Judged on the outlier fraction and on how far Q99 runs
    beyond the median.
    """
    out = exp.outputs
    frac = out.outlier_fraction
    findings = []
    if frac is not None and frac > max_outlier_fraction:
        findings.append(f"{out.n_outlier_trials}/{out.n_trials} trials "
                        f"({frac:.1%}) were catastrophic outliers")
    if out.q99_kcal is not None and out.q50_kcal not in (None, 0):
        ratio = out.q99_kcal / out.q50_kcal
        if ratio > max_q99_ratio:
            findings.append(f"Q99/Q50={ratio:.1f}x -- a heavy tail the median hides "
                            f"(Q50={out.q50_kcal:.4f}, Q99={out.q99_kcal:.4f} kcal/mol)")
    if findings:
        return GateResult("tail_risk", False, "; ".join(findings),
                          [FailureMode.HEAVY_TAIL, FailureMode.OPTIMIZER_INSTABILITY])
    if frac is None and out.q99_kcal is None:
        return GateResult("tail_risk", None, "no tail statistics recorded")
    detail = []
    if frac is not None:
        detail.append(f"{out.n_outlier_trials}/{out.n_trials} outlier trials")
    if out.q99_kcal is not None and out.q50_kcal not in (None, 0):
        detail.append(f"Q99/Q50={out.q99_kcal / out.q50_kcal:.1f}x")
    return GateResult("tail_risk", True, "tail behaviour within tolerance: " + ", ".join(detail))


def evidence_scope_gate(exp: Experiment) -> GateResult:
    """Does the stated uncertainty cover what the claim needs?

    This project's 0.115 kcal/mol headline was real under one fixed noise
    model and disowned once the noise model's own parameters were allowed
    to vary: Q95 went to 51.22 kcal/mol, ~205x over target. An uncertainty
    bar is only as broad as the thing it varied, and a claim can never be
    stronger than its bar.

    The four axes are checked independently, not ranked. The joint Schmidt
    frame propagated noise-model uncertainty but never cross-submission
    drift; earlier headline numbers had exactly the opposite gap. Neither
    is a superset of the other.
    """
    cov = exp.outputs.uncertainty
    if cov is None:
        return GateResult("evidence_scope", None, "uncertainty coverage not recorded")
    if cov.is_complete:
        return GateResult("evidence_scope", True,
                          f"uncertainty varies all four axes ({cov.describe()})")
    if not cov.covered:
        return GateResult("evidence_scope", False,
                          "the stated uncertainty varies nothing -- it is not an "
                          "uncertainty bar",
                          [FailureMode.CALIBRATION_MISMATCH, FailureMode.DRIFT])
    implicated = []
    if not cov.noise_model:
        implicated.append(FailureMode.CALIBRATION_MISMATCH)
    if not cov.cross_submission:
        implicated.append(FailureMode.DRIFT)
    if not cov.method_monte_carlo:
        implicated.append(FailureMode.MONTE_CARLO_VARIANCE)
    return GateResult("evidence_scope", False,
                      f"uncertainty varies {cov.describe()} but NOT "
                      f"{', '.join(cov.missing)} -- the bar cannot speak to what it "
                      f"never varied",
                      implicated)


def improvement_gate(exp: Experiment, min_ratio: float = 1.0) -> GateResult:
    """Grades a RELATIVE_IMPROVEMENT claim against its own baseline.

    The counterpart to chemical_accuracy_gate: a method can legitimately
    claim to beat a baseline without reaching an absolute target, and a
    method can fail to beat its baseline while sitting under one.
    """
    from .schema import ClaimType

    if exp.claim_type is not ClaimType.RELATIVE_IMPROVEMENT:
        return GateResult("improvement", None,
                          "not applicable: this is an absolute-accuracy claim")
    base = exp.outputs.baseline_error_kcal
    got = exp.outputs.mitigated_error_kcal
    if base is None or got is None:
        return GateResult("improvement", None, "no baseline or mitigated error recorded")
    if got <= 0:
        return GateResult("improvement", None, "mitigated error is not positive")
    ratio = base / got
    label = exp.outputs.baseline_label or "baseline"
    passed = ratio > min_ratio
    return GateResult("improvement", passed,
                      f"{ratio:.2f}x vs {label} ({base:.4f} -> {got:.4f} kcal/mol)",
                      [] if passed else [FailureMode.UNDER_POWERED])


# Controls an auditor can execute itself against the claimant's artifacts,
# rather than take on trust. The rest (target leakage, adversarial design,
# free-parameter floors) are procedural or domain-specific and currently
# depend on honest reporting -- which is stated plainly rather than papered
# over.
AUDITOR_VERIFIABLE = ("unitary_equivalence", "ideal_control", "determinism_check")


def mitigation_benefit_gate(exp: Experiment) -> GateResult:
    """Does the mitigation help once there is noise to correct?

    Deliberately NOT a hard gate. A method that fails to help under one
    noise model has not been shown to be broken -- it may be aimed at a
    different noise regime, or the model may be unrepresentative. But it
    has also not been shown to work, so certification waits.

    The pairing matters: ideal_control and this gate ask opposite
    questions, and passing one says nothing about the other. A no-op
    mitigator passes the ideal control trivially -- it cannot amplify
    noise it never touches -- and fails this one.
    """
    mb = exp.controls.mitigation_benefit
    if mb is None:
        return GateResult("mitigation_benefit", None,
                          "not yet run: whether mitigation helps under real noise is "
                          "untested")
    if mb is False:
        return GateResult("mitigation_benefit", False,
                          "under a real noise model the mitigation did not reduce the "
                          "error -- passing the ideal control shows only that it does "
                          "not break without noise",
                          [FailureMode.UNDER_POWERED, FailureMode.UNKNOWN])
    return GateResult("mitigation_benefit", True,
                      "the mitigation measurably reduces the error under real noise")


def independent_verification_gate(exp: Experiment) -> GateResult:
    """How much of this record did the auditor check for itself?

    A control the claimant asserts is a statement about their own work. A
    control the auditor executed is evidence. For the claimant's own
    project the distinction may not matter -- for a third party's claim it
    is the whole point, since a gate that trusts the claimant is not a
    gate.

    Never a hard failure: an honestly self-reported record is not a false
    one, and plenty of real evidence cannot be re-executed by a third
    party. It does block certification, which is a different bar.
    """
    verifiable = [c for c in AUDITOR_VERIFIABLE if getattr(exp.controls, c) is not None]
    if not verifiable:
        return GateResult("independent_verification", None,
                          "no auditor-verifiable control has been run at all")
    measured = [c for c in verifiable
                if exp.controls.provenance_of(c) is Provenance.MEASURED]
    unmeasured = [c for c in verifiable if c not in measured]
    if not measured:
        return GateResult("independent_verification", False,
                          f"every control is self-reported ({', '.join(unmeasured)}) -- "
                          f"nothing here was checked independently",
                          [FailureMode.UNKNOWN])
    if unmeasured:
        return GateResult("independent_verification", False,
                          f"measured by the auditor: {', '.join(measured)}; still "
                          f"self-reported: {', '.join(unmeasured)}",
                          [FailureMode.UNKNOWN])
    return GateResult("independent_verification", True,
                      f"every auditor-verifiable control was executed by the auditor "
                      f"({', '.join(measured)})")


ALL_GATES = [
    ideal_control_gate,
    unitary_equivalence_gate,
    target_leakage_gate,
    free_parameter_floor_gate,
    adversarial_gate,
    extrapolation_domain_gate,
    determinism_gate,
    replicate_independence_gate,
    reproducibility_gate,
    tail_risk_gate,
    evidence_scope_gate,
    mitigation_benefit_gate,
    independent_verification_gate,
    chemical_accuracy_gate,
    improvement_gate,
]
