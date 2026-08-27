"""The auditor running by itself.

Give it an experiment and a budget; it audits, works out what could still
be wrong, proposes attacks, executes the ones it can, folds the results
back in, and decides whether to continue. It stops when there is nothing
informative left to buy, and says why.

    claim -> audit -> what can still be wrong -> generate adversaries
          -> execute -> formal audit -> belief update -> next experiment

What the agent is allowed to conclude is deliberately narrow. It never
decides a claim is true. It decides only whether to keep going, and every
verdict along the way comes from the gates. An agent that could talk
itself into `CERTIFIED` would be the exact failure this project exists to
prevent, so the stopping conditions are about evidence and cost, never
about confidence.

The language model is optional throughout. With none configured the
agent runs the deterministic grammar and reaches the same verdicts; the
model only widens the set of attacks considered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .adversary import AdversarialScientist, AttackPlan
from .executor import AttackExecutor, AttackReport
from .hypothesis import Hypothesis, HypothesisLedger, Observation
from .llm import LLMProvider, NullProvider
from .llm_scientist import LLMAdversary, ProposalReview, extend_plan
from .planner import Recommendation, candidates_from_attacks, plan as rank_plan
from .schema import Experiment
from .verdict import AuditReport, Verdict, audit

# Verdicts that end the investigation: the claim is disqualified, refuted,
# or unreadable, and no further attack changes that.
TERMINAL = (Verdict.INVALID, Verdict.INVALID_RECORD, Verdict.REFUTED)


@dataclass
class Round:
    """One pass of the loop."""

    number: int
    verdict: Verdict
    report: AuditReport
    attack_plan: Optional[AttackPlan] = None
    attack_report: Optional[AttackReport] = None
    llm_review: Optional[ProposalReview] = None
    belief: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def falsified(self) -> list:
        return self.attack_report.falsified_by if self.attack_report else []

    def summary(self) -> str:
        bits = [f"round {self.number}: {self.verdict.value}"]
        if self.attack_plan:
            bits.append(f"{len(self.attack_plan.attacks)} attacks")
        if self.attack_report:
            bits.append(f"{len(self.attack_report.falsified_by)} falsified, "
                        f"{len(self.attack_report.survived)} survived, "
                        f"{len(self.attack_report.not_run)} not run")
        return " | ".join(bits)


@dataclass
class Investigation:
    """The whole run: every round, the final verdict, and why it stopped."""

    experiment_id: str
    rounds: list[Round] = field(default_factory=list)
    stopped_because: str = ""
    ledger: Optional[HypothesisLedger] = None

    @property
    def final(self) -> Optional[Round]:
        return self.rounds[-1] if self.rounds else None

    @property
    def verdict(self) -> Optional[Verdict]:
        return self.final.verdict if self.final else None

    @property
    def total_falsified(self) -> int:
        return sum(len(r.falsified) for r in self.rounds)

    def print_investigation(self) -> None:
        print(f"\n=== investigation: {self.experiment_id} ===")
        for r in self.rounds:
            print(f"  {r.summary()}")
            for outcome in r.falsified:
                print(f"      FALSIFIED by {outcome.attack.attack_id}")
        print(f"  final: {self.verdict.value if self.verdict else 'none'}")
        print(f"  stopped: {self.stopped_because}")


class AuditAgent:
    """Runs the audit loop until there is nothing informative left to do."""

    def __init__(self, adapter: Any | None = None,
                 provider: Optional[LLMProvider] = None,
                 hooks: Optional[dict[str, Callable]] = None,
                 max_rounds: int = 5,
                 budget_usd: Optional[float] = None) -> None:
        self.adapter = adapter
        self.llm = LLMAdversary(provider or NullProvider())
        self.hooks = hooks or {}
        self.max_rounds = max_rounds
        self.budget_usd = budget_usd
        self.spent_usd = 0.0

    def investigate(self, exp: Experiment,
                    hypotheses: Optional[list[Hypothesis]] = None,
                    **artifacts) -> Investigation:
        ledger = HypothesisLedger(hypotheses or [
            Hypothesis("H_genuine", "The claimed improvement is real.", 0.5),
            Hypothesis("H_artifact", "The claimed improvement is an artifact.", 0.5),
        ])
        investigation = Investigation(exp.experiment_id, ledger=ledger)

        for number in range(1, self.max_rounds + 1):
            report = audit(exp)
            round_ = Round(number, report.verdict, report)

            if report.verdict in TERMINAL:
                round_.note = "terminal verdict"
                round_.belief = dict(ledger.belief)
                investigation.rounds.append(round_)
                investigation.stopped_because = (
                    f"{report.verdict.value}: the claim is disqualified, and no further "
                    f"attack changes that")
                return investigation

            plan = AdversarialScientist().propose(exp, report)
            if self.llm.available:
                covered = [a.transformation for a in plan.attacks]
                review = self.llm.propose_attacks(exp, report, covered=covered)
                round_.llm_review = review
                plan = extend_plan(plan, review)
            round_.attack_plan = plan

            runnable = [a for a in plan.attacks if a.executable]
            affordable = self._affordable(runnable)
            if not affordable:
                round_.belief = dict(ledger.belief)
                investigation.rounds.append(round_)
                investigation.stopped_because = self._why_nothing_runnable(
                    plan, runnable)
                return investigation

            # Rank what CAN be run, so the most decisive test goes first.
            _, proposals, _ = rank_plan(ledger, candidates_from_attacks(
                type(plan)(plan.experiment_id, affordable)))
            ordered_ids = [p.candidate.candidate_id.removeprefix("attack:")
                           for p in proposals]
            affordable.sort(key=lambda a: ordered_ids.index(a.attack_id)
                            if a.attack_id in ordered_ids else len(ordered_ids))

            executor = AttackExecutor(adapter=self.adapter, hooks=self.hooks)
            attack_report = executor.run(
                exp, type(plan)(plan.experiment_id, affordable), **artifacts)
            round_.attack_report = attack_report

            progressed = False
            for outcome in attack_report.outcomes:
                if outcome.measurement is not None:
                    exp.controls.record_measured(outcome.measurement.control,
                                                 outcome.measurement.passed)
                    progressed = True
                if outcome.survived is False:
                    ledger.update(Observation(
                        outcome.attack.attack_id, {"H_genuine": 0.05},
                        outcome.detail[:120]))
                elif outcome.survived is True:
                    # Surviving is weak evidence, and weighted as such: the
                    # absence of one refutation is not proof.
                    ledger.update(Observation(
                        outcome.attack.attack_id, {"H_genuine": 1.2},
                        outcome.detail[:120]))
            self.spent_usd += sum(a.cost_usd for a in affordable)
            round_.belief = dict(ledger.belief)
            investigation.rounds.append(round_)

            if not progressed:
                investigation.stopped_because = (
                    "a full round produced no new measurement -- every remaining attack "
                    "needs a domain hook or an artifact that was not supplied")
                return investigation

        investigation.stopped_because = (
            f"reached the {self.max_rounds}-round limit with attacks still outstanding")
        return investigation

    def _affordable(self, attacks: list) -> list:
        if self.budget_usd is None:
            return list(attacks)
        remaining = self.budget_usd - self.spent_usd
        return [a for a in attacks if a.cost_usd <= remaining]

    def _why_nothing_runnable(self, plan: AttackPlan, runnable: list) -> str:
        if not plan.attacks:
            return ("no attack remains: every mechanism the grammar covers has been "
                    "tested by a control the auditor measured")
        if not runnable:
            needs_hook = sorted({a.transformation for a in plan.attacks})
            return (f"{len(plan.attacks)} attack(s) remain but none can be executed "
                    f"generically -- these need a domain hook with the claimant's own "
                    f"code: {', '.join(needs_hook[:5])}")
        return (f"budget exhausted: ${self.spent_usd:,.2f} of "
                f"${self.budget_usd:,.2f} spent, and the cheapest remaining attack "
                f"costs more than the balance")
