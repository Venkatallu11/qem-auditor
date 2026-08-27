"""Combines gate results into one overall verdict. Hard gates (ideal
control, target leakage, adversarial) can force INVALID on their own --
no amount of good replication data overrides a broken ideal control.

Before any gate runs, the record itself is checked for internal
consistency (integrity.py). A self-contradictory record yields INVALID
RECORD, which is deliberately NOT the same as INVALID: one says the
method failed, the other says the evidence was never coherent enough to
ask the question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import gates, integrity
from .schema import Experiment


class Verdict(Enum):
    INVALID_RECORD = "INVALID RECORD"      # the record contradicts itself; unauditable
    INVALID = "INVALID"                    # a hard gate actively failed
    NOT_ESTABLISHED = "NOT ESTABLISHED"    # not enough evidence yet to judge
    PROMISING = "PROMISING / REQUIRES FURTHER CERTIFICATION"
    CERTIFIED_UNDER_SCOPE = "CERTIFIED UNDER SCOPE"


@dataclass
class AuditReport:
    experiment_id: str
    verdict: Verdict
    gate_results: list[gates.GateResult]
    integrity_violations: list[str] = field(default_factory=list)

    def print_report(self) -> None:
        print(f"\n=== {self.experiment_id} ===")
        for violation in self.integrity_violations:
            print(f"  [BAD ] {'record_integrity':20} {violation}")
        for g in self.gate_results:
            status = "PASS" if g.passed is True else "FAIL" if g.passed is False else "N/A"
            print(f"  [{status:4}] {g.name:20} {g.reason}")
        print(f"  VERDICT: {self.verdict.value}")


HARD_GATES = (gates.ideal_control_gate, gates.target_leakage_gate, gates.adversarial_gate)


def audit(exp: Experiment) -> AuditReport:
    violations = integrity.integrity_violations(exp)
    results = [g(exp) for g in gates.ALL_GATES]
    if violations:
        return AuditReport(exp.experiment_id, Verdict.INVALID_RECORD, results, violations)
    by_name = {r.name: r for r in results}

    hard_gate_names = [g.__name__.replace("_gate", "") for g in HARD_GATES]
    hard = [by_name[name] for name in hard_gate_names if name in by_name]

    # A hard gate that actively failed disqualifies the claim outright.
    if any(r.passed is False for r in hard):
        return AuditReport(exp.experiment_id, Verdict.INVALID, results)

    # A hard gate that was never run is not a pass. Silence is not
    # evidence: without the control there is nothing to promote, however
    # complete the rest of the record looks.
    if any(r.passed is None for r in hard):
        return AuditReport(exp.experiment_id, Verdict.NOT_ESTABLISHED, results)

    repro = by_name["reproducibility"]
    accuracy = by_name["chemical_accuracy"]

    if repro.passed is None or accuracy.passed is None:
        return AuditReport(exp.experiment_id, Verdict.NOT_ESTABLISHED, results)

    n_reps = len(exp.outputs.replicate_errors_kcal)
    full_replication = n_reps >= exp.outputs.n_replicates_target
    if repro.passed and accuracy.passed and full_replication and exp.real_hardware_full_validation:
        return AuditReport(exp.experiment_id, Verdict.CERTIFIED_UNDER_SCOPE, results)

    return AuditReport(exp.experiment_id, Verdict.PROMISING, results)
