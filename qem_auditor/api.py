"""The public entry point.

Everything the auditor can do, reachable without knowing how it is put
together:

    from qem_auditor import Auditor

    result = Auditor().audit("my_experiment.json")
    print(result.verdict)
    print(result.render())

And, with an adapter, the same thing but with the mechanizable controls
executed rather than believed:

    from qem_auditor.adapters.qiskit_adapter import QiskitAdapter

    auditor = Auditor(adapter=QiskitAdapter())
    auditor.verify_fold_survival(exp, base=base_circuit, submitted=submitted_circuit)
    auditor.verify_ideal_control(exp, circuit, observable, my_mitigation)
    result = auditor.audit(exp)

Every verify_* call records its finding on the experiment with MEASURED
provenance, so the resulting verdict can distinguish what the auditor
checked from what it was told.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import record
from .adapters.base import ControlMeasurement
from .claim import CompiledClaim, compile_claim
from .failure_modes import FailureAnalysis, classify
from .planner import CandidateExperiment, candidates_from_audit, next_experiment
from .schema import Experiment, FailureMode
from .verdict import AuditReport, Verdict, audit as _audit


@dataclass
class AuditResult:
    """Everything the auditor concluded, in one object."""

    experiment: Experiment
    report: AuditReport
    analysis: FailureAnalysis
    claim: CompiledClaim
    gaps: list[CandidateExperiment] = field(default_factory=list)
    measurements: list[ControlMeasurement] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        return self.report.verdict

    @property
    def failure_modes(self) -> list[FailureMode]:
        return self.analysis.modes

    @property
    def next_experiment(self) -> Optional[CandidateExperiment]:
        return min(self.gaps, key=lambda c: c.cost_usd) if self.gaps else None

    @property
    def passed(self) -> bool:
        """True only for verdicts that license using the result. Deliberately
        narrow: PROMISING is not a pass, it is permission to keep working."""
        return self.verdict is Verdict.CERTIFIED_UNDER_SCOPE

    def render(self) -> str:
        parts = [self.claim.render()]
        if self.measurements:
            parts.append("\nEXECUTED BY THE AUDITOR:")
            for m in self.measurements:
                mark = "PASS" if m.passed is True else "FAIL" if m.passed is False else "N/A"
                parts.append(f"  [{mark}] {m.control}: {m.detail}")
        return "\n".join(parts)

    def print_result(self) -> None:
        self.report.print_report()
        if self.analysis.diagnoses:
            self.analysis.print_analysis()
        self.claim.print_claim()


class Auditor:
    """Audits a quantum error-mitigation claim.

    With no adapter it grades a record as written -- useful, and dependent
    on the record being honest. With an adapter it executes the controls it
    can and grades what it found, which is what makes it a verifier rather
    than a rubric.
    """

    def __init__(self, adapter: Any | None = None) -> None:
        self.adapter = adapter
        self._measurements: list[ControlMeasurement] = []

    # -- auditing ------------------------------------------------------

    def audit(self, experiment: Experiment | str | Path | dict) -> AuditResult:
        exp = self._coerce(experiment)
        report = _audit(exp)
        return AuditResult(
            experiment=exp,
            report=report,
            analysis=classify(exp, report, self._measurements),
            claim=compile_claim(exp, report),
            gaps=candidates_from_audit(exp, report),
            measurements=list(self._measurements),
        )

    @staticmethod
    def _coerce(experiment: Experiment | str | Path | dict) -> Experiment:
        if isinstance(experiment, Experiment):
            return experiment
        if isinstance(experiment, dict):
            return record.from_dict(experiment)
        return record.load(experiment)

    # -- verification --------------------------------------------------

    def _apply(self, exp: Experiment, measurement: ControlMeasurement) -> ControlMeasurement:
        exp.controls.record_measured(measurement.control, measurement.passed)
        self._measurements.append(measurement)
        return measurement

    def _require_adapter(self) -> Any:
        if self.adapter is None:
            raise RuntimeError(
                "this Auditor has no backend adapter, so it can only grade a record as "
                "written. Pass one to execute controls: "
                "Auditor(adapter=QiskitAdapter())"
            )
        return self.adapter

    def verify_unitary_equivalence(self, exp: Experiment, intended, submitted) -> ControlMeasurement:
        """Does the submitted circuit implement the intended unitary?"""
        return self._apply(exp, self._require_adapter()
                           .measure_unitary_equivalence(intended, submitted))

    def verify_fold_survival(self, exp: Experiment, base, submitted) -> ControlMeasurement:
        """Did deliberately-inserted noise-amplifying gates survive transpilation?

        The check unitary equivalence alone cannot make, because a fold pair
        is supposed to leave the unitary unchanged.
        """
        return self._apply(exp, self._require_adapter()
                           .measure_fold_survival(base, submitted))

    def verify_ideal_control(self, exp: Experiment, circuit, observable,
                             mitigator, **kwargs) -> ControlMeasurement:
        """Run the claimant's mitigation against a noiseless model."""
        return self._apply(exp, self._require_adapter()
                           .measure_ideal_control(circuit, observable, mitigator, **kwargs))

    def verify_determinism(self, exp: Experiment, computation, **kwargs) -> ControlMeasurement:
        """Run the identical computation repeatedly and diff the results."""
        return self._apply(exp, self._require_adapter()
                           .measure_determinism(computation, **kwargs))

    # -- records -------------------------------------------------------

    @staticmethod
    def load(path: str | Path) -> Experiment:
        return record.load(path)

    @staticmethod
    def save(exp: Experiment, path: str | Path) -> None:
        record.save(exp, path)
