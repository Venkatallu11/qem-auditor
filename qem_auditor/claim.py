"""Compiles an audit into a scientific claim statement.

The output is deliberately shaped so that "what has been shown" and "what
has not" are the same size on the page. A report that lists supporting
evidence and then trails off into a vague limitations paragraph is how a
0.115 kcal/mol number becomes a headline; a report that states the
unclosed gaps as specifically as the supporting ones is much harder to
over-read.

Never says "looks good". Says exactly what has been demonstrated, exactly
what has not, and the cheapest experiment that closes the largest
remaining gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .failure_modes import FailureAnalysis, classify
from .planner import CandidateExperiment, candidates_from_audit
from .schema import ClaimType, Experiment
from .verdict import AuditReport, Verdict, audit

# What each verdict actually licenses someone to do with the result. The
# point of spelling this out is that a verdict name is easy to quote out
# of context; a sentence about what it permits is not.
_LICENCE = {
    Verdict.INVALID_RECORD: "Nothing. The record contradicts itself and cannot be read as evidence.",
    Verdict.INVALID: "Nothing. A hard control failed; the method is broken independent of the data.",
    Verdict.REFUTED: "Nothing as stated. The claim's own evidence contradicts it. Whether "
                     "the underlying method is worth further work is a separate question.",
    Verdict.CONFLICT: "Report wider bars. Independent measurements disagree by more than the "
                      "stated uncertainty, and there is no basis for preferring either.",
    Verdict.NOT_ESTABLISHED: "Nothing yet. Nothing here was shown to be wrong; the evidence "
                             "required to support the claim was never collected.",
    Verdict.MODEL_CONDITIONAL: "Cite only under the models tested, always with that condition "
                               "attached. Do not cite as an unconditional result.",
    Verdict.PROMISING: "Continue the work and cite as preliminary. Not a certified result.",
    Verdict.CERTIFIED_UNDER_SCOPE: "Cite within the stated scope. Outside that scope it is "
                                   "untested, not proven.",
}


@dataclass
class CompiledClaim:
    experiment_id: str
    claim: str
    verdict: Verdict
    supported_by: list[str] = field(default_factory=list)
    not_established: list[str] = field(default_factory=list)
    failure_analysis: FailureAnalysis | None = None
    next_experiment: CandidateExperiment | None = None
    pass_criterion: str = ""

    @property
    def licence(self) -> str:
        return _LICENCE[self.verdict]

    def render(self) -> str:
        lines = [
            f"CLAIM:      {self.claim or '(none stated)'}",
            f"EXPERIMENT: {self.experiment_id}",
            f"STATUS:     {self.verdict.value}",
            f"LICENCE:    {self.licence}",
            "",
            "SUPPORTED BY:",
        ]
        lines += [f"  + {s}" for s in self.supported_by] or ["  (nothing)"]
        lines += ["", "NOT YET ESTABLISHED:"]
        lines += [f"  - {s}" for s in self.not_established] or ["  (nothing outstanding)"]
        if self.failure_analysis and self.failure_analysis.diagnoses:
            lines += ["", "WHY:"]
            for d in self.failure_analysis.diagnoses:
                lines.append(f"  {d.mode.name} (confidence {d.confidence:.2f})")
                lines.append(f"    {d.evidence}")
        if self.next_experiment:
            cost = (f"${self.next_experiment.cost_usd:,.2f}"
                    if self.next_experiment.cost_usd > 0 else "no cost")
            lines += ["", "NEXT EXPERIMENT:",
                      f"  {self.next_experiment.description}",
                      f"  cost: {cost}"]
        if self.pass_criterion:
            lines += ["", f"PASS CRITERION: {self.pass_criterion}"]
        return "\n".join(lines)

    def print_claim(self) -> None:
        print("\n" + self.render())


def compile_claim(exp: Experiment, report: AuditReport | None = None) -> CompiledClaim:
    report = report or audit(exp)

    supported = [
        f"{g.name}: {g.reason}" for g in report.gate_results if g.passed is True
    ]
    outstanding = [
        f"{g.name}: {g.reason}"
        for g in report.gate_results
        if g.passed is False or (g.passed is None and g.name in _CLOSEABLE)
    ]
    outstanding += [f"record integrity: {v}" for v in report.integrity_violations]

    n_independent = len(exp.outputs.independent_replicates)
    target = exp.outputs.n_replicates_target
    if n_independent < target:
        outstanding.append(
            f"replication: {n_independent}/{target} independent draws completed")
    if not exp.real_hardware_full_validation:
        outstanding.append("real-hardware validation: not a full energy reconstruction")

    analysis = classify(exp, report)
    candidates = candidates_from_audit(exp, report)
    nxt = min(candidates, key=lambda c: c.cost_usd) if candidates else None

    if exp.claim_type is ClaimType.RELATIVE_IMPROVEMENT:
        base = exp.outputs.baseline_error_kcal
        criterion = (f"beat {exp.outputs.baseline_label or 'baseline'} "
                     f"({base} kcal/mol) on the same statistic, with tail risk no worse"
                     if base is not None else "beat the stated baseline")
    else:
        criterion = (f"Q95 < 0.25 kcal/mol across {target} independent draws, "
                     f"with calibration uncertainty propagated")

    return CompiledClaim(
        experiment_id=exp.experiment_id,
        claim=exp.claim,
        verdict=report.verdict,
        supported_by=supported,
        not_established=outstanding,
        failure_analysis=analysis,
        next_experiment=nxt,
        pass_criterion=criterion,
    )


# Gates whose "not yet run" state names a real, runnable experiment rather
# than an inapplicable check. improvement/chemical_accuracy go N/A by claim
# type, which is not a gap in the evidence.
_CLOSEABLE = frozenset({
    "ideal_control", "unitary_equivalence", "target_leakage", "free_parameter_floor",
    "adversarial", "extrapolation_domain", "determinism", "replicate_independence",
    "reproducibility", "tail_risk", "evidence_scope",
})
