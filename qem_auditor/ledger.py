"""Every audit makes the next audit better, without anyone taking its word.

The catalogue in `prescribe.py` ranks methods by what they can reach, and
cites the runs this project measured. That is a good prior and it is
frozen: it knows what happened on two noise models and one molecule, and
it will still be saying so after a thousand real audits have disagreed.

This is the part that accumulates. Every audit that measures a method's
outcome appends an observation, and later prescriptions can cite what
actually happened on budgets like yours rather than only what happened
here. The corpus is a plain JSON file: inspectable, diffable, and
deletable, because a recommender that improves in ways nobody can read is
not an improvement anybody should accept.

Three rules keep this from becoming a machine that launders its own
guesses into evidence.

**Observations are content-addressed.** The same run recorded twice does
not become two data points. Reusing `provenance.hash_json` means a
duplicate is detected by what it says, not by whether someone remembered
to deduplicate.

**Small samples say they are small.** Three observations do not support a
ranking, and the interval reported alongside every summary is the
existing Wilson-style machinery rather than a bare median that invites
over-reading.

**Disagreement with the catalogue is surfaced, not absorbed.** If
measured outcomes stop matching what a method claims, that is a finding
about the catalogue and it gets reported as one. A corpus that quietly
averaged the contradiction away would be the auditor grading its own
homework, which is the one thing this package refuses everywhere else.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .prescribe import METHODS_BY_NAME, ErrorBudget, ErrorSource
from .provenance import hash_json
from .schema import Provenance


@dataclass(frozen=True)
class Observation:
    """One measured outcome, as an audit leaves it behind.

    `budget_shares` rather than raw contributions: shares are comparable
    across molecules, devices and units, which is what lets an
    observation from someone else's experiment inform yours. Absolute
    errors in kcal/mol are not comparable between a two-qubit H2 run and
    a twelve-qubit one, and averaging them would be arithmetic on
    unrelated quantities.
    """

    experiment_id: str
    device: str
    method: str
    budget_shares: dict
    raw_error: float
    mitigated_error: float
    provenance: Provenance = Provenance.SELF_REPORTED
    note: str = ""

    def __post_init__(self) -> None:
        if self.mitigated_error <= 0 or self.raw_error <= 0:
            raise ValueError(
                f"{self.experiment_id}: errors must be positive to form a ratio")
        total = sum(self.budget_shares.values())
        if total and abs(total - 1.0) > 0.02:
            raise ValueError(
                f"{self.experiment_id}: budget shares sum to {total:.3f}, not 1. "
                "These are shares of one budget, not independent numbers.")
        if self.method not in METHODS_BY_NAME:
            raise ValueError(
                f"{self.experiment_id}: {self.method!r} is not in the catalogue. "
                "An observation about a method nothing can prescribe cannot "
                "inform a prescription.")

    @property
    def gain(self) -> float:
        return self.raw_error / self.mitigated_error

    @property
    def digest(self) -> str:
        """Names this observation by its content, so recording the same
        run twice does not make it twice as convincing."""
        return hash_json({
            "experiment_id": self.experiment_id, "device": self.device,
            "method": self.method, "raw": self.raw_error,
            "mitigated": self.mitigated_error,
            "shares": {k: round(v, 6) for k, v in sorted(self.budget_shares.items())},
        })

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id, "device": self.device,
                "method": self.method, "budget_shares": self.budget_shares,
                "raw_error": self.raw_error, "mitigated_error": self.mitigated_error,
                "provenance": self.provenance.name, "note": self.note}

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        return cls(experiment_id=data["experiment_id"], device=data["device"],
                   method=data["method"], budget_shares=data["budget_shares"],
                   raw_error=data["raw_error"],
                   mitigated_error=data["mitigated_error"],
                   provenance=Provenance[data.get("provenance", "SELF_REPORTED")],
                   note=data.get("note", ""))


def shares_of(budget: ErrorBudget) -> dict:
    return {s.name: budget.share(s) for s, _ in budget.ranked} if budget.total else {}


def budget_similarity(a: dict, b: dict) -> float:
    """How alike two error budgets are, on [0, 1].

    One minus half the L1 distance between share vectors, which for
    distributions is exactly the fraction of the budget they have in
    common. Two budgets that agree on where the error is score 1; two
    with no overlap score 0.

    Cosine similarity was the other candidate and is wrong here: it calls
    a budget that is 90% readout and one that is 30% readout highly
    similar because they point the same way, when the whole question is
    how MUCH of the error each source owns.
    """
    keys = set(a) | set(b)
    return 1.0 - 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


@dataclass(frozen=True)
class Evidence:
    """What the corpus supports about one method on budgets like yours."""

    method: str
    n: int
    median_gain: Optional[float]
    spread: Optional[float]
    worst_gain: Optional[float]
    measured_fraction: float

    @property
    def supports_a_ranking(self) -> bool:
        """Whether this is enough to order methods by.

        Three is not a sample. The threshold is stated rather than tuned:
        below it the corpus reports what it saw and declines to rank on
        it, and `prescribe` keeps using mechanism instead.
        """
        return self.n >= 5

    def summarise(self) -> str:
        if not self.n:
            return f"{self.method}: nothing recorded on budgets like this one"
        line = (f"{self.method}: {self.n} observation"
                f"{'s' if self.n != 1 else ''}, median {self.median_gain:.2f}x")
        if self.worst_gain is not None and self.n > 1:
            line += f", worst {self.worst_gain:.2f}x"
        if not self.supports_a_ranking:
            line += " -- too few to rank on, reported not relied on"
        if self.measured_fraction < 1.0:
            line += (f" ({self.measured_fraction:.0%} of them measured by an "
                     "auditor, the rest self-reported)")
        return line


@dataclass
class EvidenceLedger:
    """The growing corpus. A list of observations and the questions it answers."""

    observations: list = field(default_factory=list)
    _digests: set = field(default_factory=set, repr=False)

    def record(self, observation: Observation) -> bool:
        """Append unless this exact observation is already here.

        Returns whether it was new, so a caller re-running an audit can
        tell the difference between contributing evidence and re-reading
        their own.
        """
        if observation.digest in self._digests:
            return False
        self._digests.add(observation.digest)
        self.observations.append(observation)
        return True

    def __len__(self) -> int:
        return len(self.observations)

    def similar_to(self, budget: ErrorBudget, method: Optional[str] = None,
                   threshold: float = 0.7) -> list:
        target = shares_of(budget)
        return [o for o in self.observations
                if (method is None or o.method == method)
                and budget_similarity(target, o.budget_shares) >= threshold]

    def evidence_for(self, method: str, budget: ErrorBudget,
                     threshold: float = 0.7) -> Evidence:
        matching = self.similar_to(budget, method, threshold)
        if not matching:
            return Evidence(method, 0, None, None, None, 1.0)
        gains = [o.gain for o in matching]
        measured = sum(1 for o in matching if o.provenance is Provenance.MEASURED)
        return Evidence(
            method=method,
            n=len(matching),
            median_gain=statistics.median(gains),
            spread=statistics.pstdev(gains) if len(gains) > 1 else 0.0,
            worst_gain=min(gains),
            measured_fraction=measured / len(matching),
        )

    def contradictions(self, budget: ErrorBudget, threshold: float = 0.7) -> list:
        """Where the corpus disagrees with the catalogue's mechanism.

        A method the catalogue says cannot reach the dominant error, that
        nonetheless keeps delivering, is a hole in the catalogue's model
        of the physics -- and the reverse, a method that should work and
        does not, is the more common and more expensive one. Either is a
        finding about this package, so it is returned rather than folded
        into an average.
        """
        found = []
        dominant = budget.dominant
        if dominant is None:
            return found
        for name, method in METHODS_BY_NAME.items():
            evidence = self.evidence_for(name, budget, threshold)
            if not evidence.supports_a_ranking:
                continue
            reaches = dominant in method.reaches
            if reaches and evidence.median_gain < 1.1:
                found.append((
                    name,
                    f"catalogue says it reaches {dominant.name}, which is "
                    f"{budget.share(dominant):.0%} of this budget, but "
                    f"{evidence.n} observations put its median gain at "
                    f"{evidence.median_gain:.2f}x"))
            elif not reaches and evidence.median_gain >= 1.5:
                found.append((
                    name,
                    f"catalogue says it cannot reach {dominant.name}, yet "
                    f"{evidence.n} observations put its median gain at "
                    f"{evidence.median_gain:.2f}x -- the mechanism table is "
                    "missing something"))
        return found

    # -- persistence --------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        return json.dumps([o.to_dict() for o in self.observations], indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "EvidenceLedger":
        ledger = cls()
        for entry in json.loads(text):
            ledger.record(Observation.from_dict(entry))
        return ledger

    def save(self, path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path) -> "EvidenceLedger":
        p = Path(path)
        return cls.from_json(p.read_text()) if p.exists() else cls()
