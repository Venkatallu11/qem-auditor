"""Runs the attacks the adversary proposes.

The third stage of the loop: the proposer writes an attack and what each
outcome would mean, this executes it, and the gates judge what came back.
The executor deliberately holds no opinion -- it returns measurements,
and never a verdict.

Only some of the grammar is mechanizable without domain hooks. That is
stated per attack rather than hidden: an attack this cannot run is
reported as needing a hook, never quietly reported as passing. The
distinction between "we ran it and it held" and "we could not run it" is
the whole difference between evidence and assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .adversary import Attack, AttackPlan
from .adapters.base import ControlMeasurement, MeasurementError
from .schema import Experiment


@dataclass
class AttackOutcome:
    """What happened when an attack was run, and which prediction it matched."""

    attack: Attack
    ran: bool
    measurement: Optional[ControlMeasurement] = None
    matched: str = ""
    """Which branch of the pre-registered prediction the result matched:
    'genuine', 'artifact', or '' when it could not be run."""

    detail: str = ""

    @property
    def survived(self) -> Optional[bool]:
        """True when the claim survived this attack, None when unrun.

        Note the asymmetry: surviving an attack is not proof, it is the
        absence of one refutation. Failing one is much stronger evidence
        than surviving one is.
        """
        if not self.ran:
            return None
        return self.matched == "genuine"

    def describe(self) -> str:
        if not self.ran:
            return f"[NOT RUN] {self.attack.attack_id}: {self.detail}"
        verdict = "SURVIVED" if self.survived else "FALSIFIED"
        return f"[{verdict}] {self.attack.attack_id}: {self.detail}"


@dataclass
class AttackReport:
    experiment_id: str
    outcomes: list[AttackOutcome] = field(default_factory=list)

    @property
    def falsified_by(self) -> list[AttackOutcome]:
        return [o for o in self.outcomes if o.survived is False]

    @property
    def survived(self) -> list[AttackOutcome]:
        return [o for o in self.outcomes if o.survived is True]

    @property
    def not_run(self) -> list[AttackOutcome]:
        return [o for o in self.outcomes if o.survived is None]

    def print_report(self) -> None:
        print(f"\n=== attack report: {self.experiment_id} ===")
        for o in self.outcomes:
            print("  " + o.describe())
        print(f"  {len(self.falsified_by)} falsified, {len(self.survived)} survived, "
              f"{len(self.not_run)} not run")
        if self.not_run:
            print("  (an attack that could not be run is not an attack the claim "
                  "survived)")


class AttackExecutor:
    """Executes the mechanizable part of the grammar.

    `hooks` supplies domain-specific callables for attacks this cannot run
    generically -- a label-shuffled refit, say, needs the claimant's own
    fitting code. A hook returns (matched_branch, detail).
    """

    def __init__(self, adapter: Any | None = None,
                 hooks: dict[str, Callable[[Experiment], tuple[str, str]]] | None = None) -> None:
        self.adapter = adapter
        self.hooks = hooks or {}

    def run(self, exp: Experiment, plan: AttackPlan, **artifacts) -> AttackReport:
        """Runs every attack it can. `artifacts` carries the circuits,
        observable and mitigator the executable attacks need."""
        report = AttackReport(exp.experiment_id)
        for attack in plan.attacks:
            report.outcomes.append(self._run_one(exp, attack, artifacts))
        return report

    def _run_one(self, exp: Experiment, attack: Attack,
                 artifacts: dict) -> AttackOutcome:
        hook = self.hooks.get(attack.transformation)
        if hook is not None:
            try:
                matched, detail = hook(exp)
            except Exception as e:
                return AttackOutcome(attack, False, detail=f"hook raised: {e}")
            return AttackOutcome(attack, True, matched=matched, detail=detail)

        # A composed attack runs each part; any part matching its artifact
        # branch falsifies the whole, since the branches are alternatives.
        if " o " in attack.transformation:
            return self._run_composed(exp, attack, artifacts)

        runner = _RUNNERS.get(attack.transformation)
        if runner is None:
            return AttackOutcome(
                attack, False,
                detail=f"no generic executor for {attack.transformation}; supply a hook "
                       f"with the claimant's own code")
        if self.adapter is None:
            return AttackOutcome(attack, False,
                                 detail="needs a backend adapter to execute")
        try:
            return runner(self, exp, attack, artifacts)
        except MeasurementError as e:
            return AttackOutcome(attack, False, detail=f"could not execute: {e}")


def _run_composed_impl(ex: "AttackExecutor", exp: Experiment, attack: Attack,
                       artifacts: dict) -> AttackOutcome:
    parts = [p.strip() for p in attack.transformation.split(" o ")]
    details, matched = [], "genuine"
    for name in parts:
        runner = _RUNNERS.get(name)
        if runner is None:
            return AttackOutcome(attack, False,
                                 detail=f"composed attack needs an executor for {name}")
        sub = runner(ex, exp, attack, artifacts)
        if not sub.ran:
            return AttackOutcome(attack, False,
                                 detail=f"{name} could not run: {sub.detail}")
        details.append(f"{name}: {sub.detail}")
        if sub.matched == "artifact":
            matched = "artifact"
    return AttackOutcome(attack, True, matched=matched, detail=" | ".join(details))


def _run_compiler(ex: AttackExecutor, exp: Experiment, attack: Attack,
                  artifacts: dict) -> AttackOutcome:
    base, submitted = artifacts.get("base_circuit"), artifacts.get("submitted_circuit")
    if base is None or submitted is None:
        return AttackOutcome(attack, False,
                             detail="needs base_circuit and submitted_circuit")
    m = ex.adapter.measure_fold_survival(base, submitted)
    return AttackOutcome(attack, True, m,
                         "genuine" if m.passed else "artifact", m.detail)


def _run_extrapolation(ex: AttackExecutor, exp: Experiment, attack: Attack,
                       artifacts: dict) -> AttackOutcome:
    circuit = artifacts.get("circuit")
    observable = artifacts.get("observable")
    mitigator = artifacts.get("mitigator")
    if circuit is None or observable is None or mitigator is None:
        return AttackOutcome(attack, False,
                             detail="needs circuit, observable and mitigator")
    m = ex.adapter.measure_ideal_control(
        circuit, observable, mitigator,
        shots=artifacts.get("shots", 20_000))
    return AttackOutcome(attack, True, m,
                         "genuine" if m.passed else "artifact", m.detail)


def _run_seed(ex: AttackExecutor, exp: Experiment, attack: Attack,
              artifacts: dict) -> AttackOutcome:
    computation = artifacts.get("computation")
    if computation is None:
        return AttackOutcome(attack, False,
                             detail="needs a `computation` callable to repeat")
    m = ex.adapter.measure_determinism(computation, runs=artifacts.get("runs", 3))
    return AttackOutcome(attack, True, m,
                         "genuine" if m.passed else "artifact", m.detail)


AttackExecutor._run_composed = _run_composed_impl  # type: ignore[attr-defined]

_RUNNERS = {
    "T_compiler": _run_compiler,
    "T_extrapolation": _run_extrapolation,
    "T_seed": _run_seed,
}
