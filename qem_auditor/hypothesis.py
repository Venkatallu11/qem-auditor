"""A ledger of competing explanations, carried across experiments.

The H4 project's real decision trail was a sequence of hypotheses, each
believed, tested, and mostly killed: one-qubit noise dominates (measured:
no), all-gate ZNE will fix it (ideal control: no), more PEC draws will fix
it (33E: marginal), cross-fitting removes the nonlinear bias (35E: no, and
it adds tail risk), finer calibration fixes the robustness tail (37:
structurally non-identifiable). Every one of those was a real belief with
real evidence for and against, and every iteration re-derived the state of
play from prose.

This module holds that state explicitly. Beliefs are probabilities
updated by Bayes' rule against recorded observations, so evidence
accumulates instead of resetting, and a hypothesis can be driven to near
zero and stay there.

Deliberately NOT an LLM's summary of the evidence. The likelihoods are
supplied per observation and the arithmetic is plain, so the reasoning is
inspectable and reproducible -- the same separation the gates enforce.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Hypothesis:
    hypothesis_id: str
    claim: str
    prior: float
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Non-negative, not necessarily a probability: the ledger
        # normalizes, so an author may supply rough relative weights
        # ("twice as plausible as that one") without pretending to a
        # calibrated prior they do not have.
        if self.prior < 0:
            raise ValueError(
                f"prior weight for {self.hypothesis_id} must be non-negative, got {self.prior}")


@dataclass
class Observation:
    """One experiment's result, expressed as how likely it was under each
    hypothesis -- P(D | H_i), not a verdict about which hypothesis won."""

    experiment_id: str
    likelihoods: dict[str, float]
    summary: str = ""

    def __post_init__(self) -> None:
        for hid, lik in self.likelihoods.items():
            if lik < 0:
                raise ValueError(f"likelihood P(D|{hid}) cannot be negative, got {lik}")


def entropy(distribution: dict[str, float]) -> float:
    """Shannon entropy in bits. Zero means one hypothesis holds all the
    belief; higher means the evidence has not discriminated."""
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for p in distribution.values():
        q = p / total
        if q > 0:
            h -= q * math.log2(q)
    return h


class HypothesisLedger:
    """Competing explanations and their current credence."""

    def __init__(self, hypotheses: list[Hypothesis]) -> None:
        if not hypotheses:
            raise ValueError("a ledger needs at least one hypothesis")
        ids = [h.hypothesis_id for h in hypotheses]
        if len(set(ids)) != len(ids):
            raise ValueError("hypothesis ids must be unique")
        self.hypotheses = {h.hypothesis_id: h for h in hypotheses}
        total = sum(h.prior for h in hypotheses)
        if total <= 0:
            raise ValueError("priors must not all be zero")
        # Normalized so the ledger is a proper distribution even when the
        # author supplied rough weights rather than probabilities.
        self.belief = {h.hypothesis_id: h.prior / total for h in hypotheses}
        self.observations: list[Observation] = []

    def update(self, observation: Observation) -> dict[str, float]:
        """Bayes: P(H|D) proportional to P(D|H) P(H). Hypotheses the
        observation says nothing about keep their likelihood at 1, so
        silence never counts as evidence either way."""
        unknown = set(observation.likelihoods) - set(self.belief)
        if unknown:
            raise KeyError(f"observation references unknown hypotheses: {sorted(unknown)}")
        unnormalized = {
            hid: p * observation.likelihoods.get(hid, 1.0)
            for hid, p in self.belief.items()
        }
        total = sum(unnormalized.values())
        if total <= 0:
            raise ValueError(
                f"observation {observation.experiment_id!r} has zero likelihood under every "
                "hypothesis -- the hypothesis set cannot explain the data at all, which is "
                "itself a finding: add the explanation that is missing"
            )
        self.belief = {hid: v / total for hid, v in unnormalized.items()}
        self.observations.append(observation)
        for hid, lik in observation.likelihoods.items():
            note = f"{observation.experiment_id}: {observation.summary or 'observed'}"
            if lik > 1.0:
                self.hypotheses[hid].evidence_for.append(note)
            elif lik < 1.0:
                self.hypotheses[hid].evidence_against.append(note)
        return dict(self.belief)

    @property
    def entropy(self) -> float:
        return entropy(self.belief)

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.belief.items(), key=lambda kv: -kv[1])

    def leading(self) -> tuple[str, float]:
        return self.ranked()[0]

    def is_resolved(self, threshold: float = 0.9) -> bool:
        """One explanation holds most of the belief. Resolution is a
        stopping condition for the planner, not a certification -- the
        gates decide whether the leading hypothesis's supporting result is
        trustworthy, and that is a separate question."""
        return self.leading()[1] >= threshold

    def print_ledger(self) -> None:
        print(f"\n=== hypothesis ledger ({len(self.observations)} observations, "
              f"entropy {self.entropy:.3f} bits) ===")
        for hid, p in self.ranked():
            h = self.hypotheses[hid]
            print(f"  P={p:.3f}  {hid}")
            print(f"           {h.claim}")
            if h.evidence_for:
                print(f"           for:     {h.evidence_for[-1]}")
            if h.evidence_against:
                print(f"           against: {h.evidence_against[-1]}")
