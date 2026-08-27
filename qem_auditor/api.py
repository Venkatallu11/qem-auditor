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
from .adversary import AdversarialScientist, AttackPlan
from .executor import AttackExecutor, AttackReport
from .frontdoor import VerificationInputs, build_experiment, unverifiable_here
from .power import PowerAnalysis, analyze_experiment
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
    attacks: Optional[AttackPlan] = None
    power: Optional[PowerAnalysis] = None
    outside_scope: list = field(default_factory=list)
    """(control, why) the auditor structurally cannot establish from the
    artifacts it was given. Distinguishes 'your method failed this' from
    'nobody could check this from a circuit alone'."""

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

    @property
    def open_attacks(self) -> list:
        """Attacks the claim has not been subjected to. An untested
        mechanism is not a mechanism that was ruled out."""
        return list(self.attacks.attacks) if self.attacks else []

    def render(self) -> str:
        parts = [self.claim.render()]
        if self.power is not None:
            parts.append(f"\nSTATISTICAL POWER:\n  {self.power.summary()}")
        if self.outside_scope:
            parts.append("\nOUTSIDE WHAT THIS CHECK CAN ESTABLISH:")
            for name, why in self.outside_scope:
                parts.append(f"  {name}: {why}")
        if self.attacks and self.attacks.attacks:
            parts.append(f"\nUNTESTED ATTACK SURFACE ({len(self.attacks.attacks)}):")
            for a in self.attacks.attacks:
                parts.append(f"  {a.attack_id}: {a.prediction.statistic}")
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

    def audit(self, experiment: Experiment | str | Path | dict,
              propose_attacks: bool = True) -> AuditResult:
        exp = self._coerce(experiment)
        report = _audit(exp)
        return AuditResult(
            experiment=exp,
            report=report,
            analysis=classify(exp, report, self._measurements),
            claim=compile_claim(exp, report),
            gaps=candidates_from_audit(exp, report),
            measurements=list(self._measurements),
            attacks=(AdversarialScientist().propose(exp, report)
                     if propose_attacks else None),
            power=analyze_experiment(exp),
        )

    # -- the front door ------------------------------------------------

    def verify(self, circuit: Any, observable: Any = None,
               mitigator: Any = None, *,
               submitted_circuit: Any = None,
               amplified_circuit: Any = None,
               claim: str = "", backend: str = "unspecified",
               shots: int = 20_000,
               replicate_errors: Any = (),
               experiment_id: str = "",
               run_attacks: bool = True,
               **agent_kwargs) -> AuditResult:
        """Audit a circuit directly, without writing a record first.

        Builds the record from the artifacts, executes every control it
        can, and reports the verdict. Controls it cannot execute stay
        unrun and are listed separately -- so a user can tell what their
        method failed apart from what nobody could check from a circuit.

        Needs an adapter to execute anything. Without one it still builds
        and grades the record, which is a weaker but honest answer.
        """
        inputs = VerificationInputs(
            circuit=circuit, observable=observable, mitigator=mitigator,
            submitted_circuit=submitted_circuit,
            amplified_circuit=amplified_circuit,
            claim=claim, backend=backend,
            shots=shots, replicate_errors=tuple(replicate_errors))
        exp = build_experiment(inputs, experiment_id)

        if self.adapter is not None:
            self._run_what_we_can(exp, inputs)

        result = self.audit(exp, propose_attacks=run_attacks)
        result.outside_scope = unverifiable_here()
        return result

    def _run_what_we_can(self, exp: Experiment, inputs: VerificationInputs) -> None:
        """Execute the mechanizable controls against the real artifacts."""
        submitted = inputs.submitted_circuit
        if submitted is not None:
            try:
                if inputs.amplified_circuit is not None:
                    # The pipeline deliberately inserts gates, so the
                    # question is whether they SURVIVED -- not merely
                    # whether the unitary matches, which a fold pair is
                    # designed to leave alone and which therefore passes
                    # precisely when the folds have been cancelled.
                    self.verify_fold_survival(exp, base=inputs.circuit,
                                              submitted=submitted)
                else:
                    self.verify_unitary_equivalence(exp, inputs.circuit, submitted)
            except Exception:
                pass  # reported as unrun, never as passed

        if inputs.mitigator is not None and inputs.observable is not None:
            try:
                self.verify_ideal_control(exp, inputs.circuit, inputs.observable,
                                          inputs.mitigator, shots=inputs.shots)
            except Exception:
                pass
            try:
                oracle_free = lambda: float(inputs.mitigator(
                    lambda c, o: self.adapter._exact_expectation(c, o)))
                self.verify_determinism(exp, oracle_free, runs=3)
            except Exception:
                pass

    def attack(self, experiment: Experiment | str | Path | dict) -> AttackPlan:
        """Propose falsification experiments for a claim.

        The proposer commits to what each outcome would mean before
        anything runs, and never issues a verdict of its own.
        """
        exp = self._coerce(experiment)
        return AdversarialScientist().propose(exp, _audit(exp))

    def run_attacks(self, exp: Experiment, plan: AttackPlan | None = None,
                    hooks: dict | None = None, **artifacts) -> AttackReport:
        """Execute the attacks that can be run, and record what they found.

        Pass `reconstructor=` and `fit_data=` to enable the fit-based
        attacks (T_label, T_sign, T_shot). They need no backend adapter --
        only your own fitting code behind the Reconstructor interface.

        Findings are written back onto the record with MEASURED
        provenance, so the verdict afterwards reflects what the auditor
        established rather than what it was told.
        """
        plan = plan or self.attack(exp)
        executor = AttackExecutor(adapter=self.adapter, hooks=hooks)
        report = executor.run(exp, plan, **artifacts)
        for outcome in report.outcomes:
            if outcome.measurement is not None:
                exp.controls.record_measured(outcome.measurement.control,
                                             outcome.measurement.passed)
                self._measurements.append(outcome.measurement)
        return report

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
