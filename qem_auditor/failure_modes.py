"""Explains WHY a claim failed, in the vocabulary of this project's own
retrospective of every disqualification it recorded.

The difference this makes, in the project's own words: not "ZNE failed",
but "ZNE's physical fold response is valid; the production estimator
extrapolates outside the domain its held-out validation tested, so the
failure is numerical conditioning, not hardware noise." The first
sentence ends an investigation. The second one directs the next
experiment.

Every rule here fires on gate results and record fields, never on prose.
The classifier is deliberately never shown `exp.suspected_failure_modes`
-- whoever wrote the record does not get to steer the diagnosis. Its
findings are compared against that field only afterward, by the caller,
as a check on the classifier rather than an input to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

from .schema import Experiment, FailureMode
from .verdict import AuditReport


@dataclass
class Diagnosis:
    mode: FailureMode
    confidence: float  # 0-1, how directly the evidence implicates this mode
    evidence: str
    """What in the record supports this. Always a specific observation,
    never a restatement of the mode's name."""

    remedy: str = ""
    """The cheapest thing that would resolve or rule out this mode."""


@dataclass
class FailureAnalysis:
    experiment_id: str
    diagnoses: list[Diagnosis] = field(default_factory=list)

    @property
    def primary(self) -> Diagnosis | None:
        return self.diagnoses[0] if self.diagnoses else None

    @property
    def modes(self) -> list[FailureMode]:
        return [d.mode for d in self.diagnoses]

    def print_analysis(self) -> None:
        print(f"\n--- failure analysis: {self.experiment_id} ---")
        if not self.diagnoses:
            print("  no failure modes implicated")
            return
        for d in self.diagnoses:
            print(f"  {d.mode.name} (confidence {d.confidence:.2f})")
            print(f"    evidence: {d.evidence}")
            if d.remedy:
                print(f"    remedy:   {d.remedy}")


def classify(exp: Experiment, report: AuditReport) -> FailureAnalysis:
    """Derives failure modes from the gate results and the record's own
    numbers. Ordered most-confident first."""
    found: list[Diagnosis] = []
    by_name = {g.name: g for g in report.gate_results}
    out = exp.outputs

    def gate_failed(name: str) -> bool:
        g = by_name.get(name)
        return g is not None and g.passed is False

    # --- Direct, high-confidence: the gate itself identifies the mode ---

    if gate_failed("unitary_equivalence"):
        found.append(Diagnosis(
            FailureMode.COMPILER_CANCELLATION, 0.95,
            "the submitted circuit was verified NOT to implement the intended unitary, "
            "so every downstream number describes a circuit nobody intended to run",
            "fold or modify AFTER transpilation and submit with no further transpiler "
            "passes; verify the submitted circuit, not the constructed one"))

    if gate_failed("extrapolation_domain"):
        found.append(Diagnosis(
            FailureMode.EXTRAPOLATION_INSTABILITY, 0.9,
            "the estimator is used outside the domain its held-out validation tested -- "
            "validating interpolation does not validate extrapolation",
            "re-run the held-out check in the direction production actually uses "
            "(predict the production point from data on one side only)"))

    if gate_failed("free_parameter_floor"):
        found.append(Diagnosis(
            FailureMode.FREE_PARAMETER_DEGENERACY, 0.9,
            "a free parameter has no floor: in its limit the method converges on "
            "re-evaluating the answer it is supposed to be measuring",
            "impose and test a floor on the parameter, or drop the method"))

    if gate_failed("target_leakage"):
        found.append(Diagnosis(
            FailureMode.TARGET_LEAKAGE, 0.9,
            "the known exact answer influenced tuning or method selection",
            "refit with the target withheld; compare against a shuffled-label fit"))

    if gate_failed("determinism"):
        found.append(Diagnosis(
            FailureMode.NONDETERMINISM, 0.85,
            "identical inputs did not reproduce an identical result, so the reported "
            "number is one draw from an unstated distribution",
            "pin hash seed and BLAS threading, sort every collection feeding a "
            "nonconvex solver, then confirm N identical reruns before trusting a number"))

    # --- Inferred: the numbers implicate a mode the gates only hint at ---

    if gate_failed("ideal_control"):
        # An ideal control fails for a reason. If mitigation made a
        # noiseless case dramatically worse, that is conditioning, not noise.
        if out.raw_error_kcal not in (None, 0) and out.mitigated_error_kcal is not None:
            ratio = out.mitigated_error_kcal / out.raw_error_kcal
            if ratio > 2:
                found.append(Diagnosis(
                    FailureMode.EXTRAPOLATION_INSTABILITY, 0.85,
                    f"on a model with zero real noise to correct, mitigation amplified the "
                    f"error {ratio:.0f}x ({out.raw_error_kcal:.4f} -> "
                    f"{out.mitigated_error_kcal:.4f} kcal/mol) -- the failure is numerical "
                    f"conditioning of the estimator, not hardware noise",
                    "quantify the estimator's conditioning at the production point before "
                    "any further hardware spend"))
        if not any(d.mode is FailureMode.EXTRAPOLATION_INSTABILITY for d in found):
            found.append(Diagnosis(
                FailureMode.UNKNOWN, 0.5,
                "the ideal/noiseless control failed but the record does not carry raw and "
                "mitigated errors to localize why",
                "record raw vs mitigated error on the ideal control to separate "
                "conditioning from implementation"))

    if gate_failed("reproducibility"):
        reps = [r.error_kcal for r in out.independent_replicates]
        spread = pstdev(reps) if len(reps) > 1 else 0.0
        m = mean(reps) if reps else 0.0
        # Which mechanism? Drift and Monte Carlo variance look identical
        # in the spread alone; the record's scope tells them apart.
        if out.uncertainty is not None and out.uncertainty.cross_submission:
            found.append(Diagnosis(
                FailureMode.DRIFT, 0.7,
                f"{len(reps)} independent submissions of nominally identical work spread "
                f"{spread:.4f} kcal/mol around {m:.4f} -- run-to-run platform variation "
                f"that single-submission bootstrap bars cannot see",
                "report bars that include cross-submission drift; do not compare two "
                "single-submission numbers whose gap is smaller than the drift"))
        else:
            found.append(Diagnosis(
                FailureMode.MONTE_CARLO_VARIANCE, 0.6,
                f"replicates spread {spread:.4f} kcal/mol around {m:.4f} with no "
                f"cross-submission evidence to attribute it -- the method's own sampling "
                f"is the leading suspect",
                "decompose the variance by source (shot / method Monte Carlo / optimizer "
                "/ submission) before adding shots, which addresses only the smallest term"))

    if gate_failed("tail_risk"):
        frac = out.outlier_fraction
        detail = (f"{out.n_outlier_trials}/{out.n_trials} trials blew up"
                  if frac is not None else "Q99 runs far beyond the median")
        found.append(Diagnosis(
            FailureMode.HEAVY_TAIL, 0.8,
            f"{detail} -- the median is not the risk, and a method whose centre improves "
            f"while its tail explodes has not improved",
            "report Q95/Q99 alongside the median and rank methods on tail risk, not "
            "point estimates"))
        found.append(Diagnosis(
            FailureMode.OPTIMIZER_INSTABILITY, 0.5,
            "rare catastrophic trials in a pipeline containing a nonconvex fit are "
            "usually bad local optima rather than a physical effect",
            "log the per-trial fit diagnostics (chi2/dof, restart count) for the "
            "outlier trials specifically, rather than only the aggregate"))

    if gate_failed("evidence_scope"):
        # Name the mode that matches the axis actually missing. A bar that
        # varied the noise model but not the platform implicates drift; one
        # that varied neither implicates both. Attributing every scope gap
        # to calibration would have misdiagnosed the joint Schmidt frame,
        # whose noise-model envelope was the most thorough in the project.
        cov = out.uncertainty
        missing = cov.missing if cov else ["shot_noise", "method_monte_carlo",
                                           "cross_submission", "noise_model"]
        if "noise_model" in missing:
            found.append(Diagnosis(
                FailureMode.CALIBRATION_MISMATCH, 0.65,
                "the stated uncertainty never varied the assumed noise parameters, so it "
                "cannot speak to how far they sit from the true ones -- a result under one "
                "fixed noise model predicts little about hardware",
                "re-evaluate through a randomized noise-model envelope over calibration "
                "intervals justified by real measurements"))
        if "cross_submission" in missing:
            found.append(Diagnosis(
                FailureMode.DRIFT, 0.6,
                "every trial resamples ONE data collection, so the bar cannot see "
                "run-to-run platform variation -- separately measured on this platform at "
                "+-2.31 (forte-1) and +-4.01 (aria-1) kcal/mol between independent "
                "submissions under the same named profile",
                "re-execute the experiment as an independent submission and compare; on a "
                "free simulator this is the cheapest missing evidence there is"))
        if "method_monte_carlo" in missing:
            found.append(Diagnosis(
                FailureMode.MONTE_CARLO_VARIANCE, 0.55,
                "the method's own sampling (PEC draws, twirls, optimizer restarts) was "
                "never varied -- historically the largest single term in this pipeline's "
                "variance budget, at ~570x pure shot noise",
                "hold the data fixed and vary only the method's own draws to size this "
                "term before spending anything on more shots"))

    if gate_failed("replicate_independence"):
        kinds = sorted({r.kind.name for r in out.replicates if not r.kind.is_independent})
        found.append(Diagnosis(
            FailureMode.UNDER_POWERED, 0.8,
            f"the replicates offered are {'/'.join(kinds)}, which re-use one execution's "
            f"counts and therefore measure shot noise rather than reproducibility",
            "execute the experiment again, independently -- this is usually cheap on a "
            "free simulator and is the single highest-value missing evidence"))

    if gate_failed("chemical_accuracy") or gate_failed("improvement"):
        if not found:
            found.append(Diagnosis(
                FailureMode.UNDER_POWERED, 0.5,
                "every control the record carries is clean; the method simply does not "
                "reach its stated bar yet",
                "improve the method or lower the claim -- there is no bug to find here"))

    found.sort(key=lambda d: -d.confidence)
    return FailureAnalysis(exp.experiment_id, _merge(found))


def _merge(diagnoses: list[Diagnosis]) -> list[Diagnosis]:
    """One diagnosis per failure mode. Several rules can implicate the same
    mode from different angles -- the extrapolation case is reached both
    from the wrong-direction held-out check and from the ideal control's
    own blowup ratio -- and both observations are worth keeping, but as one
    finding rather than two. Keeps the highest confidence and joins the
    evidence."""
    merged: dict[FailureMode, Diagnosis] = {}
    for d in diagnoses:
        existing = merged.get(d.mode)
        if existing is None:
            merged[d.mode] = Diagnosis(d.mode, d.confidence, d.evidence, d.remedy)
            continue
        existing.evidence = f"{existing.evidence}; also: {d.evidence}"
        if not existing.remedy:
            existing.remedy = d.remedy
    return sorted(merged.values(), key=lambda d: -d.confidence)


def agreement_with_record(exp: Experiment, analysis: FailureAnalysis) -> dict:
    """Compares the classifier's independent diagnosis against whatever the
    record's author suspected. Reported as a check on the classifier, never
    fed into it."""
    suspected = set(exp.suspected_failure_modes)
    found = set(analysis.modes)
    return {
        "confirmed": sorted(m.name for m in suspected & found),
        "missed_by_classifier": sorted(m.name for m in suspected - found),
        "found_independently": sorted(m.name for m in found - suspected),
    }
