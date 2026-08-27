"""Combines gate results into one overall verdict. Hard gates (ideal
control, target leakage, adversarial) can force INVALID on their own --
no amount of good replication data overrides a broken ideal control.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import gates
from .schema import Experiment


class Verdict(Enum):
    INVALID = "INVALID"                    # a hard gate actively failed
    NOT_ESTABLISHED = "NOT ESTABLISHED"    # not enough evidence yet to judge
    PROMISING = "PROMISING / REQUIRES FURTHER CERTIFICATION"
    CERTIFIED_UNDER_SCOPE = "CERTIFIED UNDER SCOPE"


@dataclass
class AuditReport:
    experiment_id: str
    verdict: Verdict
    gate_results: list[gates.GateResult]

    def print_report(self) -> None:
        print(f"\n=== {self.experiment_id} ===")
        for g in self.gate_results:
            status = "PASS" if g.passed is True else "FAIL" if g.passed is False else "N/A"
            print(f"  [{status:4}] {g.name:20} {g.reason}")
        print(f"  VERDICT: {self.verdict.value}")


HARD_GATES = (gates.ideal_control_gate, gates.target_leakage_gate, gates.adversarial_gate)


def audit(exp: Experiment) -> AuditReport:
    results = [g(exp) for g in gates.ALL_GATES]
    by_name = {r.name: r for r in results}

    hard_gate_names = {g.__name__.replace("_gate", "") for g in HARD_GATES}
    for name in hard_gate_names:
        r = by_name.get(name)
        if r is not None and r.passed is False:
            return AuditReport(exp.experiment_id, Verdict.INVALID, results)

    repro = by_name["reproducibility"]
    accuracy = by_name["chemical_accuracy"]

    if repro.passed is None or accuracy.passed is None:
        return AuditReport(exp.experiment_id, Verdict.NOT_ESTABLISHED, results)

    n_reps = len(exp.outputs.replicate_errors_kcal)
    full_replication = n_reps >= exp.outputs.n_replicates_target
    if repro.passed and accuracy.passed and full_replication and exp.real_hardware_full_validation:
        return AuditReport(exp.experiment_id, Verdict.CERTIFIED_UNDER_SCOPE, results)

    return AuditReport(exp.experiment_id, Verdict.PROMISING, results)
