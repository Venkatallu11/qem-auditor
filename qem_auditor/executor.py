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
from .reconstruct import (
    ReconstructionError,
    compare_fit,
    shuffle_is_diagnostic,
    flip_sign,
    reconstruction_spread,
    resample_shots,
    shuffle_labels,
    subsample_draws,
)
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
        # Only the circuit-level attacks need a backend adapter. The
        # fit-based ones (T_label, T_sign, T_shot) need the claimant's
        # reconstructor instead, and demanding an adapter for those would
        # report them unrunnable when they are perfectly runnable.
        if attack.transformation in _NEEDS_ADAPTER and self.adapter is None:
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
        if name in _NEEDS_ADAPTER and ex.adapter is None:
            return AttackOutcome(attack, False,
                                 detail=f"{name} needs a backend adapter to execute")
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

def _reconstruction_artifacts(artifacts: dict):
    """The pair every fit-based attack needs, or a reason it cannot run."""
    reconstructor = artifacts.get("reconstructor")
    data = artifacts.get("fit_data")
    if reconstructor is None or data is None:
        return None, None, ("needs `reconstructor` and `fit_data` -- implement "
                            "qem_auditor.reconstruct.Reconstructor over your own "
                            "fitting code")
    return reconstructor, data, ""


def _run_label(ex: "AttackExecutor", exp: Experiment, attack: Attack,
               artifacts: dict) -> AttackOutcome:
    """Shuffle labels within each slot, refit, and see whether the model
    noticed. A model that fits shuffled data as well as real data has not
    learned the physics."""
    reconstructor, data, why = _reconstruction_artifacts(artifacts)
    if reconstructor is None:
        return AttackOutcome(attack, False, detail=why)
    diagnostic, why = shuffle_is_diagnostic(data)
    if not diagnostic:
        return AttackOutcome(attack, False, detail=f"cannot judge: {why}")
    threshold = float(artifacts.get("label_min_ratio", 5.0))
    try:
        comparison = compare_fit(reconstructor, data,
                                 shuffle_labels(data, artifacts.get("seed", 0)))
    except ReconstructionError as e:
        return AttackOutcome(attack, False, detail=str(e))
    genuine = comparison.ratio >= threshold
    return AttackOutcome(
        attack, True, matched="genuine" if genuine else "artifact",
        detail=(f"{comparison.detail}; " + (
            f"the shuffled fit is {comparison.ratio:.1f}x worse, so the model is "
            f"using the label correspondence"
            if genuine else
            f"the shuffled fit is comparable (needed {threshold:.0f}x) -- the model "
            f"absorbs shuffled data as readily as real data, so its agreement with "
            f"the real data was never evidence")))


def _run_sign(ex: "AttackExecutor", exp: Experiment, attack: Attack,
              artifacts: dict) -> AttackOutcome:
    """Negate every measured value and refit. A correction that encodes a
    direction should fit the negation far worse."""
    reconstructor, data, why = _reconstruction_artifacts(artifacts)
    if reconstructor is None:
        return AttackOutcome(attack, False, detail=why)
    threshold = float(artifacts.get("sign_min_ratio", 5.0))
    try:
        comparison = compare_fit(reconstructor, data, flip_sign(data))
    except ReconstructionError as e:
        return AttackOutcome(attack, False, detail=str(e))
    genuine = comparison.ratio >= threshold
    return AttackOutcome(
        attack, True, matched="genuine" if genuine else "artifact",
        detail=(f"{comparison.detail}; " + (
            "the sign-flipped fit is much worse, so the correction is directional"
            if genuine else
            "the model fits negated data about as well -- it is flexible enough to "
            "explain either direction, so it is not encoding a physical sign")))


def _run_shot(ex: "AttackExecutor", exp: Experiment, attack: Attack,
              artifacts: dict) -> AttackOutcome:
    """Which knob actually moves the answer: the data's shot noise, or the
    method's own Monte Carlo draws?

    The prediction is deliberately not about magnitude but about WHICH
    dominates. In this project's measured budget the method's own sampling
    beat shot noise by ~570x, and a study that only resampled shots
    concluded more shots were the answer.
    """
    reconstructor, data, why = _reconstruction_artifacts(artifacts)
    if reconstructor is None:
        return AttackOutcome(attack, False, detail=why)
    trials = int(artifacts.get("shot_trials", 8))
    try:
        _, shot_spread = reconstruction_spread(
            reconstructor, data,
            lambda d, t: resample_shots(d, seed=t), trials=trials)
    except ReconstructionError as e:
        return AttackOutcome(attack, False, detail=f"shot resampling: {e}")

    draws = data.draws
    if len(draws) < 2:
        return AttackOutcome(
            attack, False,
            detail=(f"only {len(draws)} method draw(s) recorded, so the method's own "
                    f"sampling cannot be varied -- tag measurements with `draw` to "
                    f"make this comparison possible"))
    keep = max(1, len(draws) // 2)
    try:
        _, draw_spread = reconstruction_spread(
            reconstructor, data,
            lambda d, t: subsample_draws(d, keep, seed=t), trials=trials)
    except ReconstructionError as e:
        return AttackOutcome(attack, False, detail=f"draw subsampling: {e}")

    # Both spreads at zero means the reconstruction did not respond to
    # either knob, which is not evidence that shots dominate -- it is
    # evidence the comparison could not be made. Reporting it as a pass
    # would be exactly the "silence counts as a pass" error the gates
    # exist to prevent.
    scale = max(abs(shot_spread), abs(draw_spread))
    if scale <= 1e-12:
        return AttackOutcome(
            attack, False,
            detail=(f"the reconstruction did not move under either perturbation "
                    f"(shot spread {shot_spread:.3g}, draw spread {draw_spread:.3g}) "
                    f"-- nothing can be concluded about which term dominates. A "
                    f"reconstruction insensitive to its own input is worth "
                    f"investigating on its own account."))

    shots_dominate = shot_spread >= draw_spread
    ratio = (draw_spread / shot_spread) if shot_spread > 0 else float("inf")
    return AttackOutcome(
        attack, True, matched="genuine" if shots_dominate else "artifact",
        detail=(f"spread from shot noise {shot_spread:.6g}, from the method's own "
                f"draws {draw_spread:.6g} ({ratio:.1f}x); " + (
                    "shot noise dominates, so more shots is the right lever"
                    if shots_dominate else
                    "the method's own sampling dominates -- more shots would address "
                    "the smaller term while the real lever goes untouched")))


# Which runners talk to a quantum backend, as opposed to the claimant's
# own fitting code.
_NEEDS_ADAPTER = frozenset({"T_compiler", "T_extrapolation", "T_seed"})

_RUNNERS = {
    "T_label": _run_label,
    "T_sign": _run_sign,
    "T_shot": _run_shot,
    "T_compiler": _run_compiler,
    "T_extrapolation": _run_extrapolation,
    "T_seed": _run_seed,
}
