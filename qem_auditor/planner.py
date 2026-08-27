"""Chooses the next experiment by value of information, and is allowed to
say stop.

Two things distinguish this from "run more shots". First, it ranks
candidates by how much they would actually resolve the open question per
unit cost, not by how much data they would produce -- in the H4 variance
budget, shot noise was 0.0037 against the method's own Monte Carlo at
2.11, so more shots would have bought a 570x-too-small improvement at
real expense. Second, it derives candidates from the auditor's own
findings: a gate that never ran IS the missing experiment, and its cost is
usually known.

Saying stop is a first-class output. When the leading hypothesis is
already resolved, or every candidate's information gain is negligible
against its cost, the recommendation is to stop and say why -- the H4
project needed exactly this before committing real money to a
$25.79-per-circuit backend where a full energy reconstruction would have
cost ~$6,825.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .hypothesis import HypothesisLedger, entropy
from .schema import Experiment
from .verdict import AuditReport, Verdict


class Recommendation(Enum):
    RUN = "RUN"
    STOP = "STOP"


@dataclass
class Outcome:
    """One way a candidate experiment could turn out, with how likely it is
    under each hypothesis."""

    label: str
    likelihoods: dict[str, float]


@dataclass
class CandidateExperiment:
    candidate_id: str
    description: str
    cost_usd: float = 0.0
    outcomes: list[Outcome] = field(default_factory=list)
    resolves: str = ""
    """Which audit gap this would close, when the candidate came from one."""


@dataclass
class Proposal:
    candidate: CandidateExperiment
    information_gain_bits: float
    cost_usd: float
    rationale: str = ""

    @property
    def value_of_information(self) -> float:
        """Bits per dollar. Free experiments are ranked on gain alone
        rather than dividing by zero -- on a free simulator, cost is not
        the scarce resource and pretending otherwise makes every free
        candidate infinitely attractive."""
        if self.cost_usd <= 0:
            return self.information_gain_bits
        return self.information_gain_bits / self.cost_usd


def expected_information_gain(ledger: HypothesisLedger,
                              candidate: CandidateExperiment) -> float:
    """IG(e) = H(H) - E_D[ H(H | D) ], in bits.

    How much the belief distribution is expected to sharpen, averaged over
    how the experiment could turn out. An experiment whose outcomes are
    equally likely under every hypothesis has an expected gain of zero: it
    cannot discriminate, however expensive or impressive it is.
    """
    if not candidate.outcomes:
        return 0.0
    prior = ledger.belief
    before = entropy(prior)
    expected_after = 0.0
    for outcome in candidate.outcomes:
        joint = {hid: p * outcome.likelihoods.get(hid, 1.0) for hid, p in prior.items()}
        p_outcome = sum(joint.values())
        if p_outcome <= 0:
            continue
        posterior = {hid: v / p_outcome for hid, v in joint.items()}
        expected_after += p_outcome * entropy(posterior)
    # Outcome likelihoods need not sum to one across outcomes; renormalize
    # the expectation by the total outcome mass so the gain stays a real
    # entropy difference rather than an artifact of unnormalized input.
    total_mass = sum(
        sum(p * o.likelihoods.get(hid, 1.0) for hid, p in prior.items())
        for o in candidate.outcomes
    )
    if total_mass <= 0:
        return 0.0
    return max(0.0, before - expected_after / total_mass)


def plan(ledger: HypothesisLedger,
         candidates: list[CandidateExperiment],
         budget_usd: float | None = None,
         min_gain_bits: float = 0.01,
         resolved_threshold: float = 0.9) -> tuple[Recommendation, list[Proposal], str]:
    """Ranks candidates by value of information and decides whether any of
    them is worth running.

    Returns (recommendation, ranked proposals, reason).
    """
    if ledger.is_resolved(resolved_threshold):
        hid, p = ledger.leading()
        return (Recommendation.STOP, [],
                f"{hid} already holds {p:.1%} of the belief -- further discrimination "
                f"between these hypotheses is not what is missing. Whether the supporting "
                f"result is trustworthy is a question for the gates, not for more data.")

    affordable = [c for c in candidates
                  if budget_usd is None or c.cost_usd <= budget_usd]
    unaffordable = [c for c in candidates if c not in affordable]

    proposals = [
        Proposal(c, expected_information_gain(ledger, c), c.cost_usd,
                 c.resolves or c.description)
        for c in affordable
    ]
    proposals.sort(key=lambda p: (-p.value_of_information, p.cost_usd))

    useful = [p for p in proposals if p.information_gain_bits >= min_gain_bits]
    if not useful:
        detail = ""
        if unaffordable:
            cheapest = min(unaffordable, key=lambda c: c.cost_usd)
            detail = (f" The cheapest experiment ruled out by budget is {cheapest.candidate_id} "
                      f"at ${cheapest.cost_usd:,.2f} against a ${budget_usd:,.2f} budget.")
        return (Recommendation.STOP, proposals,
                f"No affordable candidate is expected to gain more than {min_gain_bits} bits. "
                f"Spending here buys data, not discrimination.{detail}")

    best = useful[0]
    return (Recommendation.RUN, useful,
            f"{best.candidate.candidate_id}: {best.information_gain_bits:.3f} bits"
            + (f" at ${best.cost_usd:,.2f} ({best.value_of_information:.4f} bits/$)"
               if best.cost_usd > 0 else " at no cost"))


# --------------------------------------------------------------------------
# Deriving candidates from what the audit found missing
# --------------------------------------------------------------------------

# What it costs to close each gap, and what it buys. Costs are this
# project's own measured figures: free-simulator submissions cost nothing
# but time, while real hardware is ~$25.79 per circuit and is dominated by
# per-circuit overhead rather than shots (confirmed by a 100->500 shot
# probe that cost the same).
_GAP_REMEDIES: dict[str, tuple[str, float, str]] = {
    "ideal_control": (
        "Run the production pipeline unchanged against the exact noiseless model.",
        0.0, "the cheapest disqualifier there is, and it runs locally"),
    "unitary_equivalence": (
        "Compare the SUBMITTED circuit against the intended unitary, after transpilation.",
        0.0, "catches compiler cancellation before any submission is paid for"),
    "target_leakage": (
        "Refit with the target withheld; compare against a shuffled-label fit.",
        0.0, "local reanalysis of data already collected"),
    "free_parameter_floor": (
        "Sweep each free parameter to its limit and check the method does not "
        "degenerate toward re-evaluating the known answer.",
        0.0, "local; the test that disqualified locally-perturbed CDR"),
    "adversarial": (
        "Run the negative controls: wrong parity, shuffled labels, wrong sign.",
        0.0, "local reanalysis; a genuine effect must fail these loudly"),
    "extrapolation_domain": (
        "Re-run the held-out check in the direction production actually uses.",
        0.0, "local; this is what the 513x blowup's own validation skipped"),
    "determinism": (
        "Re-run the identical computation N times with the environment pinned "
        "and diff the outputs.",
        0.0, "local; invisible to any single run"),
    "replicate_independence": (
        "Execute the experiment again as a genuinely independent submission.",
        0.0, "free on ionq_simulator, and the single highest-value missing evidence "
             "when the only replicates on file are bootstrap resamples"),
    "reproducibility": (
        "Collect the remaining independent replication draws.",
        0.0, "free on ionq_simulator"),
    "tail_risk": (
        "Record Q95/Q99 and per-trial outlier diagnostics, not just the median.",
        0.0, "local reanalysis of trials already run"),
    "evidence_scope": (
        "Re-evaluate through a randomized noise-model envelope over calibration "
        "intervals justified by real measurements.",
        0.0, "local, compute-bound; the study that disowned the 0.115 headline"),
}


def candidates_from_audit(exp: Experiment, report: AuditReport) -> list[CandidateExperiment]:
    """Turns the audit's own gaps into concrete proposals.

    A gate that never ran is not an abstraction -- it is a specific
    experiment with a known procedure and a known cost, and it is usually
    the cheapest evidence available. Ordered as ALL_GATES is, so the
    cheapest disqualifiers come first.
    """
    out: list[CandidateExperiment] = []
    for gate in report.gate_results:
        # A gap is anything not actually passed: never run (None) and
        # actively failed (False) are both evidence the record still needs,
        # and a failed gate is often the more urgent of the two.
        if gate.passed is True:
            continue
        remedy = _GAP_REMEDIES.get(gate.name)
        if remedy is None:
            continue
        description, cost, why = remedy
        out.append(CandidateExperiment(
            candidate_id=f"close_{gate.name}",
            description=description,
            cost_usd=cost,
            resolves=f"{gate.name}: {gate.reason} -- {why}",
        ))
    return out


def next_experiment(exp: Experiment, report: AuditReport) -> CandidateExperiment | None:
    """The single cheapest piece of missing evidence, or None if the record
    has no gaps left to close."""
    candidates = candidates_from_audit(exp, report)
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.cost_usd)
