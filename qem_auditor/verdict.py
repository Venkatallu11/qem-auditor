"""Combines gate results into one overall verdict.

The ordering below is the whole design. It runs from "this record cannot
be read" through "this claim is refuted" to "this claim is proven", and
every step down is a refusal to promote a claim past the evidence
actually collected.

    INVALID RECORD      the record contradicts itself; unauditable
    INVALID             a hard gate actively failed
    REFUTED             the claim as stated is contradicted by its own evidence
    CONFLICT            the evidence disagrees with itself
    NOT ESTABLISHED     a required control was never run
    MODEL-CONDITIONAL   holds, but only under an unvalidated model assumption
    PROMISING           clean so far, not fully proven
    CERTIFIED UNDER SCOPE   every gate passed, replication complete

Three of these exist because the H4 history produced results that fit
nowhere else. CONFLICT is for the Z2-tapered case: two equally real,
equally careful submissions of the identical circuit set that disagreed
by 3.27 kcal/mol -- there is no basis to call either wrong, only evidence
that a single submission carries more uncertainty than claimed.
MODEL-CONDITIONAL is for the joint Schmidt frame: a large, real,
adversarially-validated improvement whose uncertainty was never
propagated through cross-submission drift. NOT ESTABLISHED, distinct from
INVALID, is for the many results that were never disproven -- just never
tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import gates, integrity
from .schema import Experiment, FailureMode


class Verdict(Enum):
    INVALID_RECORD = "INVALID RECORD"      # the record contradicts itself; unauditable
    INVALID = "INVALID"                    # a hard gate actively failed
    REFUTED = "REFUTED"                    # the claim's own evidence contradicts it
    CONFLICT = "CONFLICT"                  # the evidence disagrees with itself
    NOT_ESTABLISHED = "NOT ESTABLISHED"    # a required control was never run
    MODEL_CONDITIONAL = "VALID UNDER MODEL"
    PROMISING = "PROMISING / REQUIRES FURTHER CERTIFICATION"
    CERTIFIED_UNDER_SCOPE = "CERTIFIED UNDER SCOPE"


# A failure here disqualifies the claim outright: the method is broken,
# or the result is an artifact, independent of how good the numbers look.
HARD_GATES = (
    gates.ideal_control_gate,
    gates.unitary_equivalence_gate,
    gates.target_leakage_gate,
    gates.free_parameter_floor_gate,
    gates.adversarial_gate,
    gates.extrapolation_domain_gate,
    gates.determinism_gate,
)

# A failure here means the evidence contradicts itself rather than
# refuting the claim -- the right response is wider bars, not a rejection.
CONFLICT_GATES = (gates.reproducibility_gate,)

# These must have actually passed before anything can be certified.
# Absence is not a pass.
REQUIRED_FOR_CERTIFICATION = (
    gates.replicate_independence_gate,
    gates.tail_risk_gate,
    gates.evidence_scope_gate,
)


def _name(gate) -> str:
    return gate.__name__.replace("_gate", "")


@dataclass
class AuditReport:
    experiment_id: str
    verdict: Verdict
    gate_results: list[gates.GateResult]
    integrity_violations: list[str] = field(default_factory=list)

    @property
    def failed_gates(self) -> list[gates.GateResult]:
        return [g for g in self.gate_results if g.passed is False]

    @property
    def unrun_gates(self) -> list[gates.GateResult]:
        return [g for g in self.gate_results if g.passed is None]

    @property
    def implicated_failure_modes(self) -> list[FailureMode]:
        seen: list[FailureMode] = []
        for g in self.failed_gates:
            for mode in g.implicates:
                if mode not in seen:
                    seen.append(mode)
        return seen

    def print_report(self) -> None:
        print(f"\n=== {self.experiment_id} ===")
        for violation in self.integrity_violations:
            print(f"  [BAD ] {'record_integrity':22} {violation}")
        for g in self.gate_results:
            status = "PASS" if g.passed is True else "FAIL" if g.passed is False else "N/A"
            print(f"  [{status:4}] {g.name:22} {g.reason}")
        print(f"  VERDICT: {self.verdict.value}")


def audit(exp: Experiment) -> AuditReport:
    violations = integrity.integrity_violations(exp)
    results = [g(exp) for g in gates.ALL_GATES]
    if violations:
        return AuditReport(exp.experiment_id, Verdict.INVALID_RECORD, results, violations)

    by_name = {r.name: r for r in results}

    def outcomes(group):
        return [by_name[_name(g)] for g in group if _name(g) in by_name]

    # 1. A hard gate that actively failed disqualifies the claim outright.
    if any(r.passed is False for r in outcomes(HARD_GATES)):
        return AuditReport(exp.experiment_id, Verdict.INVALID, results)

    # 2. Evidence that contradicts itself is neither a pass nor a refutation.
    if any(r.passed is False for r in outcomes(CONFLICT_GATES)):
        return AuditReport(exp.experiment_id, Verdict.CONFLICT, results)

    # 3. A hard gate that was never run is not a pass. Silence is not
    #    evidence: without the control there is nothing to promote,
    #    however complete the rest of the record looks.
    if any(r.passed is None for r in outcomes(HARD_GATES)):
        return AuditReport(exp.experiment_id, Verdict.NOT_ESTABLISHED, results)

    # 4. Is the claim itself supported? Absolute and relative claims are
    #    graded by different gates; exactly one of them applies.
    #
    #    Graded BEFORE replication is demanded, deliberately. It takes more
    #    evidence to bless a claim than to withhold a blessing: a claim its
    #    own trials contradict is refuted whether or not anyone replicated
    #    it, while replication is what a claim needs to be promoted.
    grading = [by_name["chemical_accuracy"], by_name["improvement"]]
    applicable = [g for g in grading if g.passed is not None]
    if any(g.passed is False for g in applicable):
        # The claim as stated is not supported by its own numbers. This is
        # a refutation of the claim, not a "not yet" -- a method that does
        # not beat the baseline it names has not beaten it, and a method
        # whose Q95 sits 205x over its stated bar has not reached it.
        # Whether the underlying method is worth further work is a
        # separate question the auditor does not answer.
        return AuditReport(exp.experiment_id, Verdict.REFUTED, results)

    if not applicable:
        return AuditReport(exp.experiment_id, Verdict.NOT_ESTABLISHED, results)

    # 5. A result that holds only under an unvalidated model assumption is
    #    real, and is not certified. Named for what it is rather than
    #    lumped in with "not enough evidence" -- but only when there is a
    #    model to be conditional ON. A claim whose uncertainty never varied
    #    the noise model has not been established under any model, however
    #    many times its single data collection was resampled; that is
    #    absence of evidence, and it falls through to NOT ESTABLISHED.
    scope = by_name["evidence_scope"]
    tails = by_name["tail_risk"]
    cov = exp.outputs.uncertainty
    has_model_envelope = cov is not None and cov.noise_model
    if (scope.passed is False or tails.passed is False) and has_model_envelope:
        return AuditReport(exp.experiment_id, Verdict.MODEL_CONDITIONAL, results)

    # 6. Past this point the claim is not contradicted -- so the question
    #    becomes whether enough was collected to promote it.
    if by_name["reproducibility"].passed is None:
        return AuditReport(exp.experiment_id, Verdict.NOT_ESTABLISHED, results)
    if scope.passed is False or tails.passed is False:
        return AuditReport(exp.experiment_id, Verdict.MODEL_CONDITIONAL, results)

    required = outcomes(REQUIRED_FOR_CERTIFICATION)

    n_independent = len(exp.outputs.independent_replicates)
    full_replication = n_independent >= exp.outputs.n_replicates_target
    if (all(r.passed is True for r in required)
            and full_replication
            and exp.real_hardware_full_validation):
        return AuditReport(exp.experiment_id, Verdict.CERTIFIED_UNDER_SCOPE, results)

    return AuditReport(exp.experiment_id, Verdict.PROMISING, results)
