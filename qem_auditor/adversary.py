"""Generates experiments designed to destroy a claim.

The auditor's gates say what a claim must survive. This says how to
attack it. The division of labour is the same one the whole project rests
on and is worth stating precisely, because it is what keeps an LLM in the
loop without letting it grade itself:

    the proposer says   "this attack should distinguish H1 from H2, and
                         here is what each outcome would mean"
    the executor runs   the attack
    the gates say       what actually happened

A proposer never says "passed". It commits, in advance, to what each
outcome would imply -- and because it commits in advance, it cannot
reinterpret a bad result after seeing it. That pre-registration is the
mechanism, not a nicety.

**The structural rule this module enforces**: an attack is only an attack
if it predicts DIFFERENT outcomes under "the claim is genuine" and "the
claim is an artifact". An attack both hypotheses predict identically
cannot discriminate, however elaborate it looks, and `Attack` refuses to
be constructed as one. This is the same test the planner applies to
information gain, moved up to where attacks are written.

The transformations below are a grammar, not a list. Each `T` introduces
exactly one class of failure, they compose, and every one is drawn from a
failure this project actually suffered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .schema import Experiment, FailureMode
from .verdict import AuditReport


class NonDiagnosticAttack(ValueError):
    """An 'attack' whose outcome would mean the same thing either way."""


@dataclass(frozen=True)
class Prediction:
    """What each outcome would mean, committed to BEFORE running.

    Pre-registration is the point. A prediction written after seeing the
    result is not a prediction, and the failure mode it guards against --
    reinterpreting an inconvenient outcome as expected -- is the oldest
    one in science.
    """

    statistic: str
    """What gets measured. Must be computable without knowing the answer."""

    if_genuine: str
    """What this statistic should do if the claim is real."""

    if_artifact: str
    """What it should do if the claim is an artifact of the attacked mechanism."""

    def __post_init__(self) -> None:
        if not self.statistic.strip():
            raise NonDiagnosticAttack("a prediction must name what it measures")
        if self.if_genuine.strip() == self.if_artifact.strip():
            raise NonDiagnosticAttack(
                f"the prediction for {self.statistic!r} is identical under both "
                f"hypotheses ({self.if_genuine!r}), so the outcome cannot discriminate. "
                f"This is not an attack; it is an experiment that will confirm whatever "
                f"you already believe."
            )


@dataclass
class Attack:
    """One falsification experiment, with its meaning fixed in advance."""

    attack_id: str
    transformation: str
    targets: FailureMode
    description: str
    prediction: Prediction
    rationale: str
    cost_usd: float = 0.0
    executable: bool = False
    """True when an adapter can run this without domain-specific hooks."""

    discriminates: tuple[str, ...] = ()
    """Hypothesis ids this attack is meant to separate, when known."""

    discrimination: float = 0.9
    """How cleanly this attack separates its two hypotheses. Not uniform:
    counting gates is essentially exact, while judging whether a shuffled
    refit is 'comparable' involves a threshold and a noisy statistic. An
    attack that cannot be wrong should outrank one that can, and the
    planner's information gain reflects that only if this does."""

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, Prediction):
            raise NonDiagnosticAttack("an attack needs a Prediction, not a description")

    def describe(self) -> str:
        run = "executable" if self.executable else "needs a domain hook"
        cost = f"${self.cost_usd:,.2f}" if self.cost_usd else "free"
        return (f"[{self.attack_id}] {self.description}\n"
                f"    targets:  {self.targets.name}\n"
                f"    measures: {self.prediction.statistic}\n"
                f"    genuine:  {self.prediction.if_genuine}\n"
                f"    artifact: {self.prediction.if_artifact}\n"
                f"    why:      {self.rationale}\n"
                f"    cost:     {cost} ({run})")


# --------------------------------------------------------------------------
# The grammar: T : E -> E', each introducing exactly one failure class
# --------------------------------------------------------------------------

@dataclass
class Transformation:
    """A named, composable operation that deliberately breaks one thing.

    Composition matters: real failures arrive together. The H4 robustness
    envelope was T_calibration composed with T_coherent, and it behaved
    nothing like either alone -- an independent-per-instance coherent
    error produced Q95 of 827 kcal/mol, while the same magnitude applied
    per gate TYPE produced 0.21. Testing one transformation at a time
    would have missed that entirely.
    """

    name: str
    targets: FailureMode
    description: str
    build: Callable[[Experiment], Attack]

    def __call__(self, exp: Experiment) -> Attack:
        return self.build(exp)


def _attack(attack_id: str, transformation: str, targets: FailureMode,
            description: str, statistic: str, if_genuine: str, if_artifact: str,
            rationale: str, cost_usd: float = 0.0,
            executable: bool = False, discrimination: float = 0.9) -> Attack:
    return Attack(
        attack_id=attack_id, transformation=transformation, targets=targets,
        description=description,
        prediction=Prediction(statistic, if_genuine, if_artifact),
        rationale=rationale, cost_usd=cost_usd, executable=executable,
        discrimination=discrimination)


def t_label_shuffle(exp: Experiment) -> Attack:
    return _attack(
        "T_label", "T_label", FailureMode.TARGET_LEAKAGE,
        "Refit the correction with Pauli labels shuffled within each slot, then "
        "reconstruct the energy through the unchanged pipeline.",
        "chi2/dof of the shuffled fit, against the real fit",
        "shuffled fit is dramatically worse -- the model explains real structure "
        "(the joint Schmidt frame measured 52x)",
        "shuffled fit is comparable -- the model is flexible enough to absorb "
        "anything, so its agreement with real data was never evidence",
        "A model that fits shuffled data as well as real data has not learned the "
        "physics; it has learned to fit. This is the cheapest leakage test there is "
        "and it runs on data already collected.",
        executable=True, discrimination=0.9)


def t_sign_flip(exp: Experiment) -> Attack:
    return _attack(
        "T_sign", "T_sign", FailureMode.SIGN_CONVENTION,
        "Apply the correction with its sign inverted.",
        "final energy error",
        "error grows sharply and obviously -- the correction was doing real work "
        "in a specific direction",
        "error is unchanged or improves -- the correction was not the thing "
        "producing the result, or a sign convention is inconsistent somewhere",
        "A correction that helps in both directions is not a correction. This also "
        "catches the bookkeeping class of bug directly.",
        executable=True, discrimination=0.88)


def t_seed_perturb(exp: Experiment) -> Attack:
    return _attack(
        "T_seed", "T_seed", FailureMode.NONDETERMINISM,
        "Re-run the identical analysis with only the RNG seed and the environment's "
        "hash seed changed, several times.",
        "spread of the final estimate across reruns",
        "identical to floating-point tolerance -- the pipeline is deterministic "
        "given its inputs",
        "estimates differ materially -- the reported number is one draw from an "
        "unstated distribution",
        "This project hit this twice, invisibly: hash-order nondeterminism reordered "
        "a residual vector and tipped a nonconvex solver into different local optima "
        "40-50% of the time, and hash()-derived bootstrap seeds changed on every "
        "rerun against identical checkpointed data. Neither is visible from one run.",
        executable=True)


def t_calibration_shift(exp: Experiment) -> Attack:
    return _attack(
        "T_calibration", "T_calibration", FailureMode.CALIBRATION_MISMATCH,
        "Re-evaluate with the assumed noise parameters randomized over intervals "
        "justified by real measurements, applying coherent bias per gate TYPE "
        "rather than per gate instance.",
        "Q95 of the error across the randomized envelope",
        "Q95 stays near the nominal result -- the method tolerates the calibration "
        "uncertainty that actually exists",
        "Q95 explodes -- the result was an artifact of one lucky noise model",
        "The 0.115 kcal/mol headline was real under one fixed model and disowned "
        "at Q95=51.22 once the model's own parameters varied. Per-TYPE rather than "
        "per-instance matters: per-instance produced Q95=827 and was a modelling "
        "bug, not a finding.")


def t_compiler_optimize(exp: Experiment) -> Attack:
    return _attack(
        "T_compiler", "T_compiler", FailureMode.COMPILER_CANCELLATION,
        "Transpile the submitted circuit at a higher optimization level and compare "
        "both the unitary and the gate count against the intended circuit.",
        "gate count of the submitted circuit, and unitary equivalence",
        "gate count is preserved (or the pipeline pins optimization_level=0) -- "
        "what was designed is what executes",
        "gate count collapses while the unitary stays equivalent -- deliberately "
        "inserted gates were optimized back out, and the 'amplified' arm carries "
        "the same noise as the unamplified one",
        "Unitary equivalence alone cannot catch this, because a fold pair is "
        "SUPPOSED to leave the unitary unchanged. Both halves must be checked at "
        "once. This cost the project an entire ZNE result.",
        executable=True, discrimination=0.99)  # counting gates cannot be wrong


def t_extrapolation_domain(exp: Experiment) -> Attack:
    return _attack(
        "T_extrapolation", "T_extrapolation", FailureMode.EXTRAPOLATION_INSTABILITY,
        "Feed pure statistical noise from an exact noiseless model through the "
        "unmodified production estimator, and separately re-run the held-out check "
        "in the direction production actually extrapolates.",
        "error amplification ratio, mitigated over raw",
        "amplification stays near the estimator's own coefficient norm -- the "
        "expected, benign cost of extrapolating",
        "amplification is orders of magnitude larger -- the estimator is "
        "ill-conditioned at the point it is evaluated",
        "The 513x blowup passed its own held-out cross-validation, because that "
        "validation only ever tested interpolation while production extrapolated "
        "to fold=0 from data on one side. A held-out check in the wrong direction "
        "is not evidence for the direction used.",
        executable=True, discrimination=0.95)


def t_shot_reallocate(exp: Experiment) -> Attack:
    return _attack(
        "T_shot", "T_shot", FailureMode.MONTE_CARLO_VARIANCE,
        "Hold the total shot budget fixed and redistribute it across circuits, and "
        "separately vary only the method's own Monte Carlo draw count.",
        "which reallocation moves the final error most",
        "shot reallocation dominates -- statistical noise is the binding constraint "
        "and more shots would help",
        "the method's own draw count dominates -- more shots address the smallest "
        "term in the budget while the real lever goes untouched",
        "In the measured H4 variance budget: shot noise 0.0037, method Monte Carlo "
        "2.11. Roughly 570x. 'Run more shots' was the intuitive answer and the "
        "wrong one, and only decomposition revealed it.",
        executable=True, discrimination=0.85)


def t_target_leakage(exp: Experiment) -> Attack:
    return _attack(
        "T_leakage", "T_leakage", FailureMode.FREE_PARAMETER_DEGENERACY,
        "Drive each free parameter toward its limit and check whether the method "
        "converges on re-evaluating the known answer.",
        "the method's estimate as the parameter approaches its floor",
        "the estimate degrades or the method refuses -- the parameter has a real "
        "floor and the method depends on measurement",
        "the estimate converges on the exact answer -- the method is classically "
        "re-deriving what it claims to be measuring",
        "Locally-perturbed CDR was disqualified exactly here: as the training "
        "perturbation radius shrinks, the training circuit converges on the target "
        "circuit itself. Any simulator-only testbed can cheat this way, because "
        "'exact' is one Statevector call away.")


def t_correlation_break(exp: Experiment) -> Attack:
    return _attack(
        "T_correlation", "T_correlation", FailureMode.STRUCTURAL_NONIDENTIFIABILITY,
        "Fit the same data under a model that does not assume the shared structure "
        "the method relies on, and compare identifiability.",
        "smallest eigenvalue of the Fisher information",
        "the parameters remain identifiable without the shared-structure assumption "
        "-- the constraint is buying variance reduction, not existence",
        "the parameters are unidentifiable without it -- the result is an artifact "
        "of the assumption, and more samples will never fix it",
        "A shared-frame constraint on noisy data is a regularizer, not a free "
        "improvement, and there is no physical law forcing the assumed structure to "
        "hold. Worth separating 'this helps' from 'without this there is no answer'.")


GRAMMAR: dict[str, Transformation] = {
    t.name: t for t in [
        Transformation("T_label", FailureMode.TARGET_LEAKAGE,
                       "shuffle labels", t_label_shuffle),
        Transformation("T_sign", FailureMode.SIGN_CONVENTION,
                       "invert the correction's sign", t_sign_flip),
        Transformation("T_seed", FailureMode.NONDETERMINISM,
                       "perturb seeds and hash order", t_seed_perturb),
        Transformation("T_calibration", FailureMode.CALIBRATION_MISMATCH,
                       "randomize assumed noise parameters", t_calibration_shift),
        Transformation("T_compiler", FailureMode.COMPILER_CANCELLATION,
                       "raise transpiler optimization", t_compiler_optimize),
        Transformation("T_extrapolation", FailureMode.EXTRAPOLATION_INSTABILITY,
                       "evaluate outside the validated domain", t_extrapolation_domain),
        Transformation("T_shot", FailureMode.MONTE_CARLO_VARIANCE,
                       "reallocate the shot budget", t_shot_reallocate),
        Transformation("T_leakage", FailureMode.FREE_PARAMETER_DEGENERACY,
                       "drive free parameters to their floor", t_target_leakage),
        Transformation("T_correlation", FailureMode.STRUCTURAL_NONIDENTIFIABILITY,
                       "drop the assumed shared structure", t_correlation_break),
    ]
}


def compose(exp: Experiment, *names: str) -> Attack:
    """Compose transformations into one attack.

    Real failures arrive together, and composed failures are not the sum
    of their parts -- the calibration envelope's behaviour changed
    qualitatively depending on whether coherent bias was applied per gate
    type or per instance. An auditor that only ever tests one
    transformation at a time will miss interactions of exactly that kind.
    """
    if len(names) < 2:
        raise ValueError("composition needs at least two transformations")
    unknown = [n for n in names if n not in GRAMMAR]
    if unknown:
        raise KeyError(f"unknown transformation(s): {unknown}. "
                       f"Known: {sorted(GRAMMAR)}")
    parts = [GRAMMAR[n](exp) for n in names]
    return Attack(
        attack_id="+".join(names),
        transformation=" o ".join(names),
        targets=parts[0].targets,
        description=" THEN ".join(p.description for p in parts),
        prediction=Prediction(
            statistic="; ".join(p.prediction.statistic for p in parts),
            if_genuine="all of: " + "; ".join(p.prediction.if_genuine for p in parts),
            if_artifact="any of: " + "; ".join(p.prediction.if_artifact for p in parts)),
        rationale=("Composed deliberately: these failure classes interact, and testing "
                   "them separately can miss an interaction that dominates. "
                   + parts[0].rationale),
        cost_usd=sum(p.cost_usd for p in parts),
        executable=all(p.executable for p in parts),
        # A composed attack is at least as decisive as its sharpest part:
        # any one branch firing falsifies the claim.
        discrimination=max(p.discrimination for p in parts))


# --------------------------------------------------------------------------
# The proposer
# --------------------------------------------------------------------------

# Which control, if it passed, already answers what an attack would ask.
_CONTROL_FOR = {
    FailureMode.TARGET_LEAKAGE: "target_leakage",
    FailureMode.SIGN_CONVENTION: "adversarial",
    FailureMode.NONDETERMINISM: "determinism",
    FailureMode.CALIBRATION_MISMATCH: "evidence_scope",
    FailureMode.COMPILER_CANCELLATION: "unitary_equivalence",
    FailureMode.EXTRAPOLATION_INSTABILITY: "extrapolation_domain",
    FailureMode.FREE_PARAMETER_DEGENERACY: "free_parameter_floor",
}


@dataclass
class AttackPlan:
    experiment_id: str
    attacks: list[Attack] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    """(transformation, why) for attacks not proposed -- so the reader can
    see what was considered and ruled out, not just what was chosen."""

    @property
    def executable(self) -> list[Attack]:
        return [a for a in self.attacks if a.executable]

    def print_plan(self) -> None:
        print(f"\n=== adversarial plan: {self.experiment_id} ===")
        if not self.attacks:
            print("  no attacks proposed -- every mechanism this grammar covers has "
                  "already been tested by a passing control")
        for a in self.attacks:
            print(a.describe())
            print()
        for name, why in self.skipped:
            print(f"  (skipped {name}: {why})")


class AdversarialScientist:
    """Proposes experiments intended to destroy the claim.

    It never issues a verdict. It selects which mechanisms could still
    explain the result, writes an attack for each, and commits in advance
    to what each outcome would mean. Everything after that is the
    executor's and the gates' job.
    """

    def __init__(self, grammar: Optional[dict[str, Transformation]] = None) -> None:
        self.grammar = grammar or GRAMMAR

    def propose(self, exp: Experiment, report: AuditReport,
                include_composed: bool = True) -> AttackPlan:
        """Attacks for every mechanism not already ruled out by a passing,
        auditor-measured control."""
        by_name = {g.name: g for g in report.gate_results}
        plan = AttackPlan(exp.experiment_id)

        for name, transformation in self.grammar.items():
            control = _CONTROL_FOR.get(transformation.targets)
            gate = by_name.get(control) if control else None
            if gate is not None and gate.passed is True:
                # A control that PASSED still only settles the question if
                # the auditor ran it. A self-reported pass is the
                # claimant's word, which is what an adversary is for.
                from .schema import Provenance

                measured = (control in ("unitary_equivalence", "ideal_control",
                                        "determinism")
                            and exp.controls.provenance_of(
                                control if control != "determinism" else "determinism_check")
                            is Provenance.MEASURED)
                if measured:
                    plan.skipped.append(
                        (name, f"{control} passed and was measured by the auditor"))
                    continue
                plan.attacks.append(transformation(exp))
                continue
            plan.attacks.append(transformation(exp))

        # Executable and free first: an attack that can run now beats one
        # that needs a domain hook and a budget.
        plan.attacks.sort(key=lambda a: (not a.executable, a.cost_usd))

        if include_composed and len(plan.attacks) >= 2:
            interacting = [a.transformation for a in plan.attacks
                           if a.transformation in ("T_calibration", "T_extrapolation",
                                                   "T_compiler")]
            if len(interacting) >= 2:
                plan.attacks.append(compose(exp, *interacting[:2]))
        return plan
