"""A language model proposing hypotheses and attacks, validated by code.

The model widens the search; it does not widen what counts as evidence.
Every proposal it makes runs the same gauntlet a hand-written one does:

    an attack       must predict DIFFERENT outcomes under "genuine" and
                    "artifact", or `Prediction` refuses to build it
    a hypothesis    must name an observable consequence, or it is not a
                    hypothesis, it is a mood
    a control       is never accepted from a model at all

Rejections are reported rather than silently dropped. A model that
proposes six attacks of which two are non-diagnostic has told you
something useful about itself, and hiding that would make the auditor's
own behaviour unauditable -- which would be a poor look for this project
in particular.

The point of the model is the part the grammar cannot do: the grammar
encodes nine failure modes this project already suffered, and a real
claim may fail in a way nobody has met yet. The model is there to guess
at those. It is not there to decide whether any of them happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .adversary import Attack, AttackPlan, NonDiagnosticAttack, Prediction
from .hypothesis import Hypothesis
from .llm import LLMError, LLMProvider, NullProvider, Proposal, ask
from .schema import Experiment, FailureMode
from .verdict import AuditReport

SYSTEM = """You are an adversarial reviewer for quantum error-mitigation claims.

Your job is to try to DESTROY the claim you are shown, by proposing
experiments that would expose it as an artifact if it is one.

You do not decide whether anything passed. You never output a verdict,
a pass/fail, or an assertion that a control succeeded. Formal Python
gates decide what happened; you only propose what to try and state, in
advance, what each outcome would mean.

An attack is only useful if its outcome differs depending on whether the
claim is genuine. If you cannot state two DIFFERENT outcomes -- one
expected if the claim is real, one expected if it is an artifact -- then
it is not an attack and you must not propose it.

Reply with JSON only. No prose outside the JSON."""

ATTACK_PROMPT = """Here is a quantum error-mitigation claim under audit.

CLAIM: {claim}
DESCRIPTION: {description}
BACKEND: {backend}
MITIGATION CONTEXT: {notes}

The formal gates have already reported:
{gate_summary}

These mechanisms are ALREADY covered by an existing attack grammar, so do
NOT propose them again:
{covered}

Propose up to {n} NEW attacks targeting mechanisms the list above does not
cover. For each, give:

  "attack_id":   a short snake_case identifier
  "description": what to actually run, concretely
  "statistic":   what to measure, computable WITHOUT knowing the true answer
  "if_genuine":  what that statistic does if the claim is real
  "if_artifact": what it does if the claim is an artifact
  "rationale":   why this mechanism could plausibly explain the result
  "targets":     one of {failure_modes}
  "cost_usd":    rough cost, 0 for local/free analysis

Reply as: {{"attacks": [ ... ]}}"""

HYPOTHESIS_PROMPT = """Here is a quantum error-mitigation claim under audit.

CLAIM: {claim}
DESCRIPTION: {description}

The formal gates reported:
{gate_summary}

Current competing explanations:
{existing}

Propose up to {n} ADDITIONAL explanations that could still account for the
evidence and are not already listed. A hypothesis is only admissible if it
has an observable consequence -- something a measurement could come out
differently under. State that consequence explicitly.

For each give:
  "hypothesis_id": short snake_case identifier
  "claim":         the explanation, one sentence
  "consequence":   what would be observed differently if this were true
  "prior":         a rough weight between 0 and 1

Reply as: {{"hypotheses": [ ... ]}}"""


@dataclass
class ProposalReview:
    """What the model suggested, and what survived validation."""

    accepted: list = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    """(identifier, why) -- reported, never silently dropped."""

    stripped_keys: list[str] = field(default_factory=list)
    """Fields the model tried to set that it is not entitled to."""

    provider: str = ""
    error: str = ""

    @property
    def worked(self) -> bool:
        return bool(self.accepted)

    def print_review(self) -> None:
        if self.error:
            print(f"  model unavailable: {self.error}")
            return
        print(f"  {len(self.accepted)} accepted, {len(self.rejected)} rejected "
              f"(provider: {self.provider})")
        for name, why in self.rejected:
            print(f"    rejected {name}: {why}")
        if self.stripped_keys:
            print(f"    stripped fields the model may not set: "
                  f"{', '.join(self.stripped_keys)}")


def _gate_summary(report: AuditReport, limit: int = 12) -> str:
    lines = []
    for g in report.gate_results[:limit]:
        state = "PASS" if g.passed is True else "FAIL" if g.passed is False else "NOT RUN"
        lines.append(f"  [{state}] {g.name}: {g.reason}")
    return "\n".join(lines)


class LLMAdversary:
    """Proposes attacks and hypotheses beyond the built-in grammar."""

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self.provider = provider or NullProvider()

    @property
    def available(self) -> bool:
        return not isinstance(self.provider, NullProvider)

    def propose_attacks(self, exp: Experiment, report: AuditReport,
                        covered: Optional[list[str]] = None,
                        n: int = 4) -> ProposalReview:
        review = ProposalReview(provider=getattr(self.provider, "name", "unknown"))
        prompt = ATTACK_PROMPT.format(
            claim=exp.claim or "(none stated)",
            description=exp.description,
            backend=exp.backend,
            notes=exp.notes or "(none)",
            gate_summary=_gate_summary(report),
            covered="\n".join(f"  - {c}" for c in (covered or [])) or "  (none)",
            n=n,
            failure_modes=", ".join(m.name for m in FailureMode))
        try:
            proposal = ask(self.provider, SYSTEM, prompt, kind="attacks")
        except LLMError as e:
            review.error = str(e)
            return review
        review.stripped_keys = proposal.removed_keys
        for raw in _items(proposal, "attacks"):
            self._validate_attack(raw, review)
        return review

    def _validate_attack(self, raw: dict, review: ProposalReview) -> None:
        name = str(raw.get("attack_id") or raw.get("id") or "<unnamed>")
        if not isinstance(raw, dict):
            review.rejected.append((name, "not an object"))
            return
        missing = [k for k in ("description", "statistic", "if_genuine", "if_artifact")
                   if not str(raw.get(k, "")).strip()]
        if missing:
            review.rejected.append((name, f"missing required field(s): {', '.join(missing)}"))
            return
        try:
            targets = FailureMode[str(raw.get("targets", "UNKNOWN")).upper()]
        except KeyError:
            targets = FailureMode.UNKNOWN
        try:
            prediction = Prediction(str(raw["statistic"]), str(raw["if_genuine"]),
                                    str(raw["if_artifact"]))
        except NonDiagnosticAttack as e:
            # The most valuable rejection: the model proposed an
            # experiment whose outcome means the same thing either way.
            review.rejected.append((name, f"non-diagnostic: {e}"))
            return
        try:
            cost = float(raw.get("cost_usd", 0) or 0)
        except (TypeError, ValueError):
            cost = 0.0
        review.accepted.append(Attack(
            attack_id=f"llm:{name}",
            transformation=f"llm:{name}",
            targets=targets,
            description=str(raw["description"]),
            prediction=prediction,
            rationale=str(raw.get("rationale", "proposed by a language model")),
            cost_usd=max(0.0, cost),
            executable=False,  # a model-proposed attack needs a human-written runner
            # Lower than any hand-written attack: this one has not been
            # checked against a real failure, only argued for.
            discrimination=0.75,
        ))

    def propose_hypotheses(self, exp: Experiment, report: AuditReport,
                           existing: Optional[list[Hypothesis]] = None,
                           n: int = 3) -> ProposalReview:
        review = ProposalReview(provider=getattr(self.provider, "name", "unknown"))
        prompt = HYPOTHESIS_PROMPT.format(
            claim=exp.claim or "(none stated)",
            description=exp.description,
            gate_summary=_gate_summary(report),
            existing="\n".join(f"  - {h.hypothesis_id}: {h.claim}"
                               for h in (existing or [])) or "  (none)",
            n=n)
        try:
            proposal = ask(self.provider, SYSTEM, prompt, kind="hypotheses")
        except LLMError as e:
            review.error = str(e)
            return review
        review.stripped_keys = proposal.removed_keys
        known = {h.hypothesis_id for h in (existing or [])}
        for raw in _items(proposal, "hypotheses"):
            self._validate_hypothesis(raw, review, known)
        return review

    def _validate_hypothesis(self, raw: dict, review: ProposalReview,
                             known: set[str]) -> None:
        name = str(raw.get("hypothesis_id") or raw.get("id") or "<unnamed>")
        if not isinstance(raw, dict) or not str(raw.get("claim", "")).strip():
            review.rejected.append((name, "no claim stated"))
            return
        consequence = str(raw.get("consequence", "")).strip()
        if not consequence:
            # Without a consequence there is nothing an experiment could
            # do about it, so it can never be tested, confirmed or ruled
            # out. That is not a hypothesis.
            review.rejected.append(
                (name, "no observable consequence: nothing could distinguish this from "
                       "its negation, so no experiment can ever address it"))
            return
        if name in known:
            review.rejected.append((name, "duplicates an existing hypothesis"))
            return
        try:
            prior = float(raw.get("prior", 0.1) or 0.1)
        except (TypeError, ValueError):
            prior = 0.1
        review.accepted.append(Hypothesis(
            hypothesis_id=name,
            claim=f"{raw['claim']} (observable consequence: {consequence})",
            prior=max(0.0, min(1.0, prior))))


def _items(proposal: Proposal, key: str) -> list:
    payload = proposal.payload
    if isinstance(payload, dict):
        items = payload.get(key, [])
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [i for i in items if isinstance(i, dict)]


def extend_plan(plan: AttackPlan, review: ProposalReview) -> AttackPlan:
    """Fold accepted model proposals into a deterministic plan.

    Grammar attacks stay first: they are executable, they are drawn from
    failures that actually happened, and a speculative proposal should not
    displace a test that can run right now.
    """
    plan.attacks.extend(review.accepted)
    for name, why in review.rejected:
        plan.skipped.append((f"llm:{name}", why))
    return plan
