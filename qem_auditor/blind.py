"""Auditing without being told the answer.

An auditor evaluated on records whose expected verdict is visible in the
same file is not being tested; it is being graded on a task it can see
the answer to. That is the target-leakage failure this project's own
benchmarks encode, applied to the auditor itself -- and it would be
embarrassing for a tool built to catch leakage to be validated by a
leaky protocol.

A `BlindChallenge` holds the full record privately and exposes a redacted
view with the outcome quantities removed. The auditor sees the controls,
the replication structure, and the uncertainty coverage -- everything
about HOW the experiment was done -- but not what it produced. It must
answer the question that actually matters:

    what evidence would I need before calling this credible?

Only then is the answer revealed and the decision scored.

What redaction removes and why: every error and quantile, because those
are computed against the known exact energy and therefore leak it; and
the expected verdict, obviously. What it keeps: controls, replicate kinds
and counts, uncertainty coverage, cost. A blind auditor should still be
able to say "four independent draws, calibration uncertainty propagated,
adversarial controls run" -- that is the methodological evidence, and
judging it without the answer is exactly the skill being tested.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from .schema import Experiment
from .verdict import Verdict, audit

# Outcome fields that encode the answer, directly or by construction.
_REDACTED_FIELDS = ("raw_error_kcal", "mitigated_error_kcal", "q50_kcal",
                    "q95_kcal", "q99_kcal", "baseline_error_kcal")


class BlindError(RuntimeError):
    """An operation that would have leaked or misused the hidden answer."""


def redact(exp: Experiment) -> Experiment:
    """A copy with the outcome quantities removed, structure intact."""
    blind = copy.deepcopy(exp)
    for name in _REDACTED_FIELDS:
        setattr(blind.outputs, name, None)
    # Replicate VALUES leak the answer; their kinds and count do not, and
    # those are what the methodological judgement rests on. Withheld is
    # None rather than a sentinel number, so nothing downstream can
    # mistake it for a measurement.
    for r in blind.outputs.replicates:
        r.error_kcal = None
    blind.notes = ""
    blind.claim = exp.claim  # the claim is public; its verification is not
    return blind


@dataclass
class BlindDecision:
    """What the auditor concluded without seeing the answer."""

    would_certify: bool
    required_evidence: list[str] = field(default_factory=list)
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.would_certify and self.required_evidence:
            raise BlindError(
                "a decision cannot both certify and list outstanding required "
                "evidence -- if evidence is missing, the answer is not certification")


@dataclass
class BlindResult:
    challenge_id: str
    decision: BlindDecision
    true_verdict: Verdict
    correct: bool
    detail: str = ""

    def describe(self) -> str:
        mark = "CORRECT" if self.correct else "WRONG"
        return (f"[{mark}] {self.challenge_id}: auditor "
                f"{'would certify' if self.decision.would_certify else 'withheld certification'}; "
                f"true verdict {self.true_verdict.value}. {self.detail}")


class BlindChallenge:
    """A record whose outcome is hidden until the auditor has committed.

    The commitment is enforced, not requested: `reveal` refuses until
    `decide` has been called, so a decision cannot be formed after
    glimpsing the answer.
    """

    def __init__(self, experiment: Experiment, challenge_id: str = "") -> None:
        self._truth = copy.deepcopy(experiment)
        self.challenge_id = challenge_id or experiment.experiment_id
        self._decision: Optional[BlindDecision] = None

    @property
    def blinded(self) -> Experiment:
        """The view the auditor is allowed to see."""
        return redact(self._truth)

    def decide(self, decision: BlindDecision) -> None:
        if self._decision is not None:
            raise BlindError(
                "this challenge already has a decision. Re-deciding after seeing "
                "anything further is precisely what blinding prevents.")
        self._decision = decision

    def reveal(self) -> BlindResult:
        if self._decision is None:
            raise BlindError(
                "reveal() before decide(): the answer cannot be shown until a "
                "decision is committed, or the blinding does nothing")
        true_verdict = audit(self._truth).verdict
        should_certify = true_verdict is Verdict.CERTIFIED_UNDER_SCOPE
        correct = self._decision.would_certify == should_certify
        if correct and not should_certify:
            detail = ("correctly withheld certification; outstanding evidence named: "
                      + (", ".join(self._decision.required_evidence) or "none"))
        elif correct:
            detail = "correctly certified"
        elif should_certify:
            detail = "withheld certification from a record that met every bar"
        else:
            detail = ("certified a record that did not meet every bar -- a false "
                      "positive, the failure this whole project exists to prevent")
        return BlindResult(self.challenge_id, self._decision, true_verdict,
                           correct, detail)


def auto_decide(blinded: Experiment) -> BlindDecision:
    """The auditor's own blind decision, from methodology alone.

    Runs the gates over the redacted record. Gates that need the outcome
    quantities go N/A -- and an N/A is not a pass, so a record judged
    without its numbers can never certify. That is the correct behaviour
    and worth being explicit about: blind certification is impossible by
    construction, so the real question this scores is whether the auditor
    names the RIGHT missing evidence.
    """
    report = audit(blinded)
    outstanding = [f"{g.name}: {g.reason}" for g in report.gate_results
                   if g.passed is False]
    outstanding += [f"{g.name}: not established" for g in report.gate_results
                    if g.passed is None]
    return BlindDecision(
        would_certify=report.verdict is Verdict.CERTIFIED_UNDER_SCOPE,
        required_evidence=outstanding,
        reasoning=f"blind verdict from methodology alone: {report.verdict.value}")
